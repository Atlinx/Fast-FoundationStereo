#!/usr/bin/env python3

"""One-shot test client for fast_foundation_stereo_server.py.

Defaults to using images from ../demo_data and writes returned depth/disparity
into ../output_demo.
"""

import argparse
import json
import logging
import os
import sys
import time
from typing import Tuple

import cv2
import imageio
import numpy as np
import zmq

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f"{SCRIPT_DIR}/../")


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


def _vis_depth(depth: np.ndarray) -> np.ndarray:
    valid = np.isfinite(depth) & (depth > 0.0)
    depth_8u = np.zeros(depth.shape, dtype=np.uint8)
    if np.any(valid):
        dmin = float(depth[valid].min())
        dmax = float(depth[valid].max())
        if dmax > dmin:
            scaled = (depth[valid] - dmin) / (dmax - dmin)
            depth_8u[valid] = np.clip(scaled * 255.0, 0.0, 255.0).astype(np.uint8)
    return cv2.applyColorMap(depth_8u, cv2.COLORMAP_VIRIDIS)


def main():
    from Utils import o3d, toOpen3dCloud, vis_disparity

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
    parser.add_argument("--show_demo", type=int, default=1, help="Show left-right-depth preview window")
    parser.add_argument("--show_pc", type=int, default=1, help="Visualize point cloud in Open3D")
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

    fx, baseline, K = _read_intrinsics(intrinsic_file)
    logging.info("Using intrinsics: fx=%.4f baseline=%.6f m from %s", fx, baseline, intrinsic_file)
    logging.info("Left/right shape: %s", left.shape)

    meta_base = {
        "left_shape": list(left.shape),
        "right_shape": list(right.shape),
        "left_dtype": str(left.dtype),
        "right_dtype": str(right.dtype),
        "fx": float(fx),
        "fy": float(K[1, 1]),
        "cx": float(K[0, 2]),
        "cy": float(K[1, 2]),
        "baseline": float(baseline),
    }

    if args.jpeg:
        left_bgr = cv2.cvtColor(left, cv2.COLOR_RGB2BGR)
        right_bgr = cv2.cvtColor(right, cv2.COLOR_RGB2BGR)
        ok_l, left_blob = cv2.imencode(".jpg", left_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality])
        ok_r, right_blob = cv2.imencode(".jpg", right_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality])
        if not ok_l or not ok_r:
            raise RuntimeError("JPEG encoding failed")
        meta_base["left_mode"] = "jpeg"
        meta_base["right_mode"] = "jpeg"
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

    imageio.imwrite(os.path.join(args.out_dir, "left.png"), left)
    imageio.imwrite(os.path.join(args.out_dir, "right.png"), right)

    response_timing_ms = {}
    depth = None
    disp = None
    pcd = None

    for return_type in ("disparity", "depth", "point_cloud"):
        req_meta = dict(meta_base)
        req_meta["return_type"] = return_type
        logging.info("Sending %s request to %s", return_type, args.address)

        t0 = time.perf_counter()
        socket.send_multipart([json.dumps(req_meta).encode("utf-8"), left_bytes, right_bytes])
        parts = socket.recv_multipart()
        roundtrip_ms = (time.perf_counter() - t0) * 1000.0
        response_timing_ms[return_type] = roundtrip_ms

        res_meta = json.loads(parts[0].decode("utf-8"))
        if not res_meta.get("ok", False):
            raise RuntimeError(
                f"Server error for {return_type}: {res_meta.get('error', 'unknown')}"
            )

        got_type = res_meta.get("return_type")
        if got_type != return_type:
            raise RuntimeError(
                f"Response type mismatch: requested {return_type}, got {got_type}"
            )

        if return_type in ("disparity", "depth"):
            if len(parts) < 2:
                raise RuntimeError(f"Malformed {return_type} response from server")
            arr = _decode_response_array(
                parts[1],
                tuple(res_meta["output_shape"]),
                res_meta["output_dtype"],
            ).astype(np.float32)
            if return_type == "disparity":
                disp = arr
                disp_npy = os.path.join(args.out_dir, "disp_from_server.npy")
                np.save(disp_npy, disp)
                disp_vis = vis_disparity(disp, color_map=cv2.COLORMAP_TURBO)
                cv2.imwrite(os.path.join(args.out_dir, "disp_from_server_vis.png"), disp_vis)
                logging.info("Saved disparity: %s", disp_npy)
            else:
                depth = arr
                depth_npy = os.path.join(args.out_dir, "depth_from_server.npy")
                np.save(depth_npy, depth)
                depth_vis = _vis_depth(depth)
                cv2.imwrite(os.path.join(args.out_dir, "depth_from_server_vis.png"), depth_vis)
                depth_vis_rgb = depth_vis[:, :, ::-1]
                lr_depth_vis = np.concatenate([left, right, depth_vis_rgb], axis=1)
                imageio.imwrite(os.path.join(args.out_dir, "lr_depth_vis.png"), lr_depth_vis)
                logging.info("Saved depth: %s", depth_npy)
        else:
            if len(parts) < 3:
                raise RuntimeError("Malformed point_cloud response from server")
            points = _decode_response_array(
                parts[1],
                tuple(res_meta["points_shape"]),
                res_meta["points_dtype"],
            ).astype(np.float32)
            colors = _decode_response_array(
                parts[2],
                tuple(res_meta["colors_shape"]),
                res_meta["colors_dtype"],
            ).astype(np.uint8)
            pcd = toOpen3dCloud(points, colors)
            cloud_path = os.path.join(args.out_dir, "cloud_from_server.ply")
            o3d.io.write_point_cloud(cloud_path, pcd)
            logging.info("Saved point cloud: %s", cloud_path)

        logging.info(
            "%s round-trip: %.2f ms (server inference: %.2f ms)",
            return_type,
            roundtrip_ms,
            float(res_meta.get("inference_ms", -1.0)),
        )

    socket.close(linger=0)
    context.term()

    if args.show_demo and depth is not None:
        depth_vis = _vis_depth(depth)
        depth_vis_rgb = depth_vis[:, :, ::-1]
        lr_depth_vis = np.concatenate([left, right, depth_vis_rgb], axis=1)
        s = 1280.0 / lr_depth_vis.shape[1]
        resized = cv2.resize(
            lr_depth_vis,
            (int(lr_depth_vis.shape[1] * s), int(lr_depth_vis.shape[0] * s)),
            interpolation=cv2.INTER_LINEAR,
        )
        cv2.imshow("left|right|depth", resized[:, :, ::-1])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if args.show_pc and pcd is not None:
        logging.info("Visualizing point cloud. Press ESC to exit.")
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(pcd)
        vis.get_render_option().point_size = 1.0
        vis.get_render_option().background_color = np.array([0.5, 0.5, 0.5])
        if np.asarray(pcd.points).shape[0] > 0:
            ctr = vis.get_view_control()
            ctr.set_front([0, 0, -1])
            closest = np.asarray(pcd.points)[:, 2].argmin()
            ctr.set_lookat(np.asarray(pcd.points)[closest])
            ctr.set_up([0, -1, 0])
        vis.run()
        vis.destroy_window()

    logging.info("Round-trip summary (ms): %s", json.dumps(response_timing_ms, indent=2))


if __name__ == "__main__":
    main()
