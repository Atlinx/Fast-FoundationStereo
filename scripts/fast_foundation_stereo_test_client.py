#!/usr/bin/env python3

"""One-shot test client for fast_foundation_stereo_server.py.

Defaults to using images from ../demo_data and writes returned depth/disparity
into ../output_demo.
"""

import argparse
import json
import logging
import os
from typing import Tuple

import cv2
import imageio
import numpy as np
import zmq


def _read_intrinsics(path: str) -> Tuple[float, float, np.ndarray]:
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    K = np.array(list(map(float, lines[0].rstrip().split())), dtype=np.float32).reshape(3, 3)
    baseline = float(lines[1])
    fx = float(K[0, 0])
    return fx, baseline, K


def _as_hwc3(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        img = np.tile(img[..., None], (1, 1, 3))
    if img.shape[2] == 1:
        img = np.tile(img, (1, 1, 3))
    return img[..., :3]


def _decode_response_array(raw: bytes, shape, dtype: str) -> np.ndarray:
    return np.frombuffer(raw, dtype=np.dtype(dtype)).reshape(shape)


def main():
    code_dir = os.path.dirname(os.path.realpath(__file__))
    root_dir = os.path.normpath(os.path.join(code_dir, ".."))
    default_io_dir = os.path.join(root_dir, "demo_data")
    default_out_dir = os.path.join(root_dir, "output_demo")

    parser = argparse.ArgumentParser(description="Test client for Fast FoundationStereo ZMQ server")
    parser.add_argument("--address", type=str, default="tcp://127.0.0.1:8097", help="Server REQ/REP endpoint")
    parser.add_argument("--io_dir", type=str, default=default_io_dir, help="Directory containing left.png/right.png")
    parser.add_argument("--out_dir", type=str, default=default_out_dir, help="Directory to save depth/disparity outputs")
    parser.add_argument("--left_file", type=str, default="", help="Override left image path")
    parser.add_argument("--right_file", type=str, default="", help="Override right image path")
    parser.add_argument(
        "--intrinsic_file",
        type=str,
        default="",
        help="Path to intrinsics file (line1: 3x3 K flattened, line2: baseline).",
    )
    parser.add_argument("--timeout_ms", type=int, default=5000)
    parser.add_argument("--jpeg", type=int, default=0, help="Send JPEG-compressed images instead of raw")
    parser.add_argument("--jpeg_quality", type=int, default=95)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    left_file = args.left_file if args.left_file else os.path.join(args.io_dir, "left.png")
    right_file = args.right_file if args.right_file else os.path.join(args.io_dir, "right.png")

    if args.intrinsic_file:
        intrinsic_file = args.intrinsic_file
    else:
        io_k = os.path.join(args.io_dir, "K.txt")
        demo_k = os.path.join(root_dir, "demo_data", "K.txt")
        intrinsic_file = io_k if os.path.exists(io_k) else demo_k

    os.makedirs(args.out_dir, exist_ok=True)

    left = _as_hwc3(imageio.imread(left_file))
    right = _as_hwc3(imageio.imread(right_file))
    if left.shape[:2] != right.shape[:2]:
        raise ValueError(f"Image shapes must match, got {left.shape} vs {right.shape}")

    fx, baseline, _K = _read_intrinsics(intrinsic_file)
    logging.info("Using intrinsics: fx=%.4f baseline=%.6f m from %s", fx, baseline, intrinsic_file)
    logging.info("Left/right shape: %s", left.shape)

    meta = {
        "left_shape": list(left.shape),
        "right_shape": list(right.shape),
        "left_dtype": str(left.dtype),
        "right_dtype": str(right.dtype),
        "fx": float(fx),
        "baseline": float(baseline),
    }

    if args.jpeg:
        left_bgr = cv2.cvtColor(left, cv2.COLOR_RGB2BGR)
        right_bgr = cv2.cvtColor(right, cv2.COLOR_RGB2BGR)
        ok_l, left_blob = cv2.imencode(".jpg", left_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality])
        ok_r, right_blob = cv2.imencode(".jpg", right_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality])
        if not ok_l or not ok_r:
            raise RuntimeError("JPEG encoding failed")
        meta["left_mode"] = "jpeg"
        meta["right_mode"] = "jpeg"
        left_bytes = left_blob.tobytes()
        right_bytes = right_blob.tobytes()
    else:
        left_bytes = left.tobytes(order="C")
        right_bytes = right.tobytes(order="C")

    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.RCVTIMEO, args.timeout_ms)
    socket.setsockopt(zmq.SNDTIMEO, args.timeout_ms)
    socket.setsockopt(zmq.LINGER, 0)
    socket.connect(args.address)

    logging.info("Sending request to %s", args.address)
    socket.send_multipart([json.dumps(meta).encode("utf-8"), left_bytes, right_bytes])
    parts = socket.recv_multipart()
    socket.close(linger=0)
    context.term()

    res_meta = json.loads(parts[0].decode("utf-8"))
    if not res_meta.get("ok", False):
        raise RuntimeError(f"Server error: {res_meta.get('error', 'unknown')}" )
    if len(parts) < 3:
        raise RuntimeError("Malformed response from server")

    depth = _decode_response_array(parts[1], tuple(res_meta["depth_shape"]), res_meta["depth_dtype"]).astype(np.float32)
    disp = _decode_response_array(parts[2], tuple(res_meta["disp_shape"]), res_meta["disp_dtype"]).astype(np.float32)

    depth_npy = os.path.join(args.out_dir, "depth_from_server.npy")
    disp_npy = os.path.join(args.out_dir, "disp_from_server.npy")
    np.save(depth_npy, depth)
    np.save(disp_npy, disp)

    # Simple visualizations for quick sanity checks.
    finite_disp = np.where(np.isfinite(disp), disp, 0.0).astype(np.float32)
    disp_8u = cv2.normalize(finite_disp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    disp_vis = cv2.applyColorMap(disp_8u, cv2.COLORMAP_TURBO)
    cv2.imwrite(os.path.join(args.out_dir, "disp_from_server_vis.png"), disp_vis)

    finite_depth = np.where(np.isfinite(depth), depth, 0.0).astype(np.float32)
    depth_8u = cv2.normalize(finite_depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    depth_vis = cv2.applyColorMap(depth_8u, cv2.COLORMAP_VIRIDIS)
    cv2.imwrite(os.path.join(args.out_dir, "depth_from_server_vis.png"), depth_vis)

    logging.info("Server inference time: %.2f ms", float(res_meta.get("inference_ms", -1.0)))
    logging.info("Saved depth: %s", depth_npy)
    logging.info("Saved disparity: %s", disp_npy)


if __name__ == "__main__":
    main()