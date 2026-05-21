#!/usr/bin/env python3

# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
ZeroMQ server for Fast FoundationStereo single-model inference.

Protocol (REQ/REP, multipart):
Request:
  part0: JSON metadata bytes, fields:
    {
      "left_shape": [H, W, C],
      "right_shape": [H, W, C],
      "left_dtype": "uint8",
      "right_dtype": "uint8",
    "left_mode": "raw" | "jpeg" (optional, default raw),
    "right_mode": "raw" | "jpeg" (optional, default raw),
      "fx": float,
      "baseline": float,
      "fy": float (optional),
      "cx": float (optional),
      "cy": float (optional)
    }
  part1: left image raw bytes
  part2: right image raw bytes

Response (success):
    part0: JSON metadata bytes, fields:
        {
            "ok": true,
            "return_type": "disparity" | "depth" | "point_cloud",
            "inference_ms": float,
            ...type-specific fields...
        }
    return_type == "disparity":
        part1: disparity raw bytes, with output_shape/output_dtype in part0
    return_type == "depth":
        part1: depth raw bytes, with output_shape/output_dtype in part0
    return_type == "point_cloud":
        part1: points raw bytes [N, 3] float32, with points_shape/points_dtype in part0
        part2: colors raw bytes [N, 3] uint8, with colors_shape/colors_dtype in part0

Response (error):
  part0: JSON metadata bytes, fields:
    {"ok": false, "error": "..."}
"""

import argparse
import json
import logging
import os
import sys
import time
from typing import Dict, Tuple

import cv2
import numpy as np
import torch
import yaml
import zmq

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
REPO_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))
sys.path.append(f"{SCRIPT_DIR}/../")

from Utils import depth2xyzmap, set_logging_format, set_seed

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class SingleEngineTrtRunner:
    """Minimal TensorRT runner for a single engine with named I/O."""

    def __init__(self, engine_path: str):
        import tensorrt as trt

        self.trt = trt
        self.logger = trt.Logger(trt.Logger.WARNING)

        with open(engine_path, "rb") as f:
            self.engine = trt.Runtime(self.logger).deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(
                f"Failed to deserialize TRT engine from {engine_path}. "
                f"This usually means the engine was built with a different "
                f"TensorRT version (yours: {trt.__version__}). "
                f"Rebuild with: trtexec --onnx=<your .onnx> "
                f"--saveEngine={engine_path} --fp16"
            )
        self.context = self.engine.create_execution_context()

    def _trt_to_torch_dtype(self, dt):
        trt = self.trt
        mapping = {
            trt.DataType.FLOAT: torch.float32,
            trt.DataType.HALF: torch.float16,
            trt.DataType.BF16: torch.bfloat16,
            trt.DataType.INT32: torch.int32,
            trt.DataType.INT8: torch.int8,
            trt.DataType.BOOL: torch.bool,
        }
        if dt not in mapping:
            raise RuntimeError(f"Unsupported TRT dtype: {dt}")
        return mapping[dt]

    def __call__(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        trt = self.trt

        for name, tensor in inputs.items():
            expected = self._trt_to_torch_dtype(self.engine.get_tensor_dtype(name))
            if tensor.dtype != expected:
                inputs[name] = tensor.to(expected)
            if not inputs[name].is_contiguous():
                inputs[name] = inputs[name].contiguous()
            self.context.set_input_shape(name, tuple(inputs[name].shape))

        out_names = [
            self.engine.get_tensor_name(i)
            for i in range(self.engine.num_io_tensors)
            if self.engine.get_tensor_mode(self.engine.get_tensor_name(i))
            == trt.TensorIOMode.OUTPUT
        ]

        outputs: Dict[str, torch.Tensor] = {}
        for name in out_names:
            shape = tuple(self.context.get_tensor_shape(name))
            dtype = self._trt_to_torch_dtype(self.engine.get_tensor_dtype(name))
            outputs[name] = torch.empty(shape, device="cuda", dtype=dtype)

        for name, tensor in inputs.items():
            self.context.set_tensor_address(name, int(tensor.data_ptr()))
        for name, tensor in outputs.items():
            self.context.set_tensor_address(name, int(tensor.data_ptr()))

        stream = torch.cuda.current_stream().cuda_stream
        if not self.context.execute_async_v3(stream):
            raise RuntimeError("TensorRT execution failed")

        return outputs


class OnnxRuntimeRunner:
    """Run inference via ONNX Runtime (GPU if available, else CPU)."""

    def __init__(self, onnx_path: str):
        import onnxruntime as ort

        providers = []
        if "CUDAExecutionProvider" in ort.get_available_providers():
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")
        logging.info("ONNX Runtime providers: %s", providers)
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]

    def __call__(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        feed = {}
        for name in self.input_names:
            tensor = inputs[name]
            if isinstance(tensor, torch.Tensor):
                tensor = tensor.cpu().float().numpy()
            feed[name] = tensor
        raw_outputs = self.session.run(self.output_names, feed)
        outputs = {}
        for name, arr in zip(self.output_names, raw_outputs):
            outputs[name] = torch.as_tensor(arr).cuda()
        return outputs


def normalize_imagenet(img_uint8: np.ndarray) -> np.ndarray:
    """Apply ImageNet normalization: (img/255 - mean) / std."""
    return ((img_uint8.astype(np.float32) / 255.0) - IMAGENET_MEAN) / IMAGENET_STD


def resolve_config(model_path: str) -> str:
    model_dir = os.path.dirname(model_path)
    base = os.path.splitext(os.path.basename(model_path))[0]
    candidates = [
        os.path.join(model_dir, f"{base}.yaml"),
        os.path.join(model_dir, "config.yaml"),
        os.path.join(model_dir, "onnx.yaml"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"No .yaml config found for {model_path}. Run make_single_onnx.py first."
    )


def find_model(model_dir: str) -> str:
    for ext in (".engine", ".onnx"):
        for f in os.listdir(model_dir):
            if f.endswith(ext):
                return os.path.join(model_dir, f)
    raise FileNotFoundError(
        f"No .engine or .onnx file found in {model_dir}. Run make_single_onnx.py first."
    )


def _as_hwc3(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        img = np.tile(img[..., None], (1, 1, 3))
    if img.shape[2] == 1:
        img = np.tile(img, (1, 1, 3))
    return img[..., :3]


def _decode_image(raw: bytes, shape: Tuple[int, ...], dtype: str) -> np.ndarray:
    arr = np.frombuffer(raw, dtype=np.dtype(dtype))
    return arr.reshape(shape)


def _decode_image_payload(raw: bytes, shape: Tuple[int, ...], dtype: str, mode: str) -> np.ndarray:
    if mode == "raw":
        return _decode_image(raw, shape, dtype)
    if mode == "jpeg":
        enc = np.frombuffer(raw, dtype=np.uint8)
        img_bgr = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError("Failed to decode JPEG image payload")
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    raise ValueError(f"Unsupported image mode: {mode}")


class FastFoundationStereoServer:
    def __init__(
        self,
        model_path: str,
        bind_address: str,
        remove_invisible: bool = True,
    ):
        cfg_path = resolve_config(model_path)
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        self.target_h, self.target_w = cfg["image_size"]
        self.remove_invisible = remove_invisible

        if model_path.endswith(".onnx"):
            self.runner = OnnxRuntimeRunner(model_path)
            backend = "onnxruntime"
        else:
            self.runner = SingleEngineTrtRunner(model_path)
            backend = "tensorrt"

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.setsockopt(zmq.RCVHWM, 1)
        self.socket.setsockopt(zmq.SNDHWM, 1)
        self.socket.bind(bind_address)

        logging.info("Loaded model: %s", model_path)
        logging.info("Backend: %s", backend)
        logging.info("Model target resolution: %dx%d", self.target_h, self.target_w)
        logging.info("Listening on %s", bind_address)

    def _infer_depth(
        self,
        img_left: np.ndarray,
        img_right: np.ndarray,
        fx_orig: float,
        baseline: float,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        orig_h, orig_w = img_left.shape[:2]
        sx = float(self.target_w) / float(orig_w)
        sy = float(self.target_h) / float(orig_h)

        if sx != 1.0 or sy != 1.0:
            img_left = cv2.resize(
                img_left, (self.target_w, self.target_h), interpolation=cv2.INTER_LINEAR
            )
            img_right = cv2.resize(
                img_right,
                (self.target_w, self.target_h),
                interpolation=cv2.INTER_LINEAR,
            )

        left_norm = normalize_imagenet(img_left)
        right_norm = normalize_imagenet(img_right)

        t_left = torch.as_tensor(left_norm).cuda().float()[None].permute(0, 3, 1, 2)
        t_right = (
            torch.as_tensor(right_norm).cuda().float()[None].permute(0, 3, 1, 2)
        )

        t0 = time.perf_counter()
        outputs = self.runner({"left_image": t_left, "right_image": t_right})
        torch.cuda.synchronize()
        infer_ms = (time.perf_counter() - t0) * 1000.0

        disp = (
            outputs["disparity"]
            .float()
            .detach()
            .cpu()
            .numpy()
            .reshape(self.target_h, self.target_w)
        )
        disp = np.clip(disp, 0.0, None)

        if sx != 1.0 or sy != 1.0:
            disp = cv2.resize(disp, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        disp = disp * (1.0 / sx)

        # Match run_demo_single_trt behavior: visualization uses raw disparity.
        # remove_invisible should only affect depth/point-cloud computation.
        disp_for_depth = disp.copy()
        if self.remove_invisible:
            _, xx = np.meshgrid(np.arange(orig_h), np.arange(orig_w), indexing="ij")
            invalid = (xx - disp_for_depth) < 0
            disp_for_depth[invalid] = np.inf

        depth = np.full_like(disp_for_depth, np.inf, dtype=np.float32)
        valid = np.isfinite(disp_for_depth) & (disp_for_depth > 0.0)
        depth[valid] = (fx_orig * baseline) / disp_for_depth[valid]
        return depth.astype(np.float32), disp.astype(np.float32), infer_ms

    def _handle_request(self, parts):
        if len(parts) != 3:
            raise ValueError("Request must have 3 parts: meta, left_bytes, right_bytes")

        meta = json.loads(parts[0].decode("utf-8"))
        left_mode = meta.get("left_mode", "raw")
        right_mode = meta.get("right_mode", "raw")
        left = _decode_image_payload(
            parts[1], tuple(meta["left_shape"]), meta["left_dtype"], left_mode
        )
        right = _decode_image_payload(
            parts[2], tuple(meta["right_shape"]), meta["right_dtype"], right_mode
        )

        left = _as_hwc3(left)
        right = _as_hwc3(right)
        if left.shape[:2] != right.shape[:2]:
            raise ValueError(
                f"Left/right image sizes differ: {left.shape[:2]} vs {right.shape[:2]}"
            )

        fx = float(meta["fx"])
        baseline = float(meta["baseline"])
        return_type = meta.get("return_type", "depth")
        if fx <= 0.0:
            raise ValueError("fx must be positive")
        if baseline <= 0.0:
            raise ValueError("baseline must be positive")
        if return_type not in {"disparity", "depth", "point_cloud"}:
            raise ValueError(
                f"Unsupported return_type '{return_type}'. "
                "Expected one of: disparity, depth, point_cloud"
            )

        depth, disp, inference_ms = self._infer_depth(left, right, fx, baseline)
        fps = 1000.0 / inference_ms if inference_ms > 0.0 else float("inf")
        logging.info(
            "Handled request return_type=%s shape=%dx%d inference=%.2f ms (%.2f FPS)",
            return_type,
            left.shape[1],
            left.shape[0],
            inference_ms,
            fps,
        )

        if return_type == "disparity":
            res_meta = {
                "ok": True,
                "return_type": "disparity",
                "output_shape": list(disp.shape),
                "output_dtype": str(disp.dtype),
                "inference_ms": inference_ms,
            }
            self.socket.send_multipart(
                [
                    json.dumps(res_meta).encode("utf-8"),
                    disp.tobytes(order="C"),
                ]
            )
            return

        if return_type == "depth":
            res_meta = {
                "ok": True,
                "return_type": "depth",
                "output_shape": list(depth.shape),
                "output_dtype": str(depth.dtype),
                "inference_ms": inference_ms,
            }
            self.socket.send_multipart(
                [
                    json.dumps(res_meta).encode("utf-8"),
                    depth.tobytes(order="C"),
                ]
            )
            return

        h, w = depth.shape[:2]
        fy = float(meta.get("fy", fx))
        cx = float(meta.get("cx", (w - 1.0) * 0.5))
        cy = float(meta.get("cy", (h - 1.0) * 0.5))
        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
        xyz_map = depth2xyzmap(depth, K)

        points = xyz_map.reshape(-1, 3)
        colors = left.reshape(-1, 3)
        valid = np.isfinite(points).all(axis=1) & (points[:, 2] > 0.0)
        points = points[valid].astype(np.float32, copy=False)
        colors = colors[valid].astype(np.uint8, copy=False)

        res_meta = {
            "ok": True,
            "return_type": "point_cloud",
            "points_shape": list(points.shape),
            "points_dtype": str(points.dtype),
            "colors_shape": list(colors.shape),
            "colors_dtype": str(colors.dtype),
            "inference_ms": inference_ms,
        }
        self.socket.send_multipart(
            [
                json.dumps(res_meta).encode("utf-8"),
                points.tobytes(order="C"),
                colors.tobytes(order="C"),
            ]
        )

    def serve_forever(self):
        while True:
            try:
                parts = self.socket.recv_multipart()
                self._handle_request(parts)
            except KeyboardInterrupt:
                logging.info("Shutdown requested")
                break
            except Exception as exc:  # pylint: disable=broad-except
                logging.exception("Failed processing request")
                err = {"ok": False, "error": str(exc)}
                self.socket.send_multipart([json.dumps(err).encode("utf-8")])


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Fast FoundationStereo ZMQ server (TensorRT single-engine)"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=f"{REPO_DIR}/output",
        help="Directory containing .engine/.onnx + matching YAML config",
    )
    parser.add_argument(
        "--model_file",
        type=str,
        default="",
        help="Explicit path to .engine or .onnx file (overrides auto-search)",
    )
    parser.add_argument(
        "--bind",
        type=str,
        default="tcp://*:8097",
        help="ZeroMQ REP bind endpoint",
    )
    parser.add_argument(
        "--remove_invisible",
        type=int,
        default=0,
        help="Mask disparities where right-image correspondence is out of bounds",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    set_logging_format()
    set_seed(0)
    torch.autograd.set_grad_enabled(False)

    model_path = args.model_file if args.model_file else find_model(args.model_dir)
    server = FastFoundationStereoServer(
        model_path=model_path,
        bind_address=args.bind,
        remove_invisible=bool(args.remove_invisible),
    )
    server.serve_forever()