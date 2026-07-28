from __future__ import annotations

import argparse
import gc
from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
import threading
import time
from typing import Any, Callable

import cv2
import numpy as np
import yaml

from panorama_demo.cuda_backend import cuda_status, reset_cuda_audit
from panorama_demo.inspection_multiview import render_inspection_multiview
from panorama_demo.metric_mosaic import render_metric_mosaic
from panorama_demo.session import load_rgbd_session


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stress the V1 renderers at a long metric span by scaling a real, "
            "already-solved trajectory. This is performance evidence only."
        )
    )
    parser.add_argument("session", type=Path)
    parser.add_argument("transforms", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/demo.yaml"),
    )
    parser.add_argument("--target-span-mm", type=float, default=20_000.0)
    parser.add_argument(
        "--stage",
        action="append",
        choices=("inspection", "metric"),
        dest="stages",
    )
    parser.add_argument(
        "--planar-surrogate-depth-mm",
        type=float,
        help=(
            "Reuse one real RGB source with a constant aligned depth for a "
            "content-consistent renderer-only stress workload."
        ),
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _load_poses(
    path: Path,
) -> tuple[list[int], list[np.ndarray]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("pose_convention", "").startswith("camera_to_world") is False:
        raise ValueError("Transforms do not declare camera_to_world poses")
    if payload.get("translation_unit") != "mm":
        raise ValueError("Transforms must use millimetres")
    nodes = payload.get("nodes")
    if not isinstance(nodes, list) or len(nodes) < 2:
        raise ValueError("Transforms require at least two nodes")
    frame_ids: list[int] = []
    poses: list[np.ndarray] = []
    for node in nodes:
        frame_id = int(node["node_id"])
        pose = np.asarray(node["camera_to_world"], dtype=np.float64)
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise ValueError(f"Invalid pose for frame {frame_id}")
        frame_ids.append(frame_id)
        poses.append(pose)
    if len(set(frame_ids)) != len(frame_ids):
        raise ValueError("Transform node IDs must be unique")
    return frame_ids, poses


def _scale_longitudinal_span(
    poses: list[np.ndarray],
    target_span_mm: float,
) -> tuple[list[np.ndarray], dict[str, object]]:
    if not np.isfinite(target_span_mm) or target_span_mm <= 0.0:
        raise ValueError("target-span-mm must be finite and positive")
    centers = np.asarray([pose[:3, 3] for pose in poses], dtype=np.float64)
    displacement = centers[-1] - centers[0]
    norm = float(np.linalg.norm(displacement))
    if norm <= 1e-6:
        raise ValueError("Trajectory endpoints do not define a scan direction")
    axis = displacement / norm
    relative = centers - centers[0]
    endpoint_longitudinal = relative @ axis
    step_lengths = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    if np.any(~np.isfinite(step_lengths)) or np.any(step_lengths <= 1e-6):
        raise ValueError("Trajectory contains a zero or invalid camera-centre step")
    longitudinal = np.concatenate(
        (np.zeros(1, dtype=np.float64), np.cumsum(step_lengths))
    )
    source_span = float(longitudinal[-1])
    scale = target_span_mm / source_span
    residual = relative - endpoint_longitudinal[:, None] * axis[None, :]
    scaled_centers = (
        centers[0]
        + longitudinal[:, None] * scale * axis[None, :]
        + residual
    )
    scaled: list[np.ndarray] = []
    for pose, center in zip(poses, scaled_centers, strict=True):
        transformed = pose.copy()
        transformed[:3, 3] = center
        scaled.append(transformed)
    return scaled, {
        "source_longitudinal_span_mm": source_span,
        "source_endpoint_axis_span_mm": float(endpoint_longitudinal[-1]),
        "target_longitudinal_span_mm": float(target_span_mm),
        "translation_scale": float(scale),
        "scan_axis_world": axis.tolist(),
        "longitudinal_parameter": "cumulative_real_camera_centre_path_length",
        "longitudinal_order_regularized": bool(
            np.any(np.diff(endpoint_longitudinal) <= 0.0)
        ),
        "rotation_modified": False,
        "lateral_and_depth_residual_modified": False,
        "pose_interpolation_count": 0,
    }


def _working_set_bytes() -> int:
    if os.name != "nt":
        return 0
    import ctypes
    from ctypes import wintypes

    class _Counters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
            ("PrivateUsage", ctypes.c_size_t),
        ]

    counters = _Counters()
    counters.cb = ctypes.sizeof(counters)
    handle = ctypes.windll.kernel32.GetCurrentProcess()
    get_memory_info = ctypes.windll.kernel32.K32GetProcessMemoryInfo
    get_memory_info.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(_Counters),
        wintypes.DWORD,
    ]
    get_memory_info.restype = wintypes.BOOL
    ok = get_memory_info(
        handle,
        ctypes.byref(counters),
        counters.cb,
    )
    return int(counters.WorkingSetSize) if ok else 0


def _measure_stage(
    name: str,
    operation: Callable[[], Any],
) -> tuple[Any, dict[str, object]]:
    stop = threading.Event()
    peak = _working_set_bytes()

    def monitor() -> None:
        nonlocal peak
        while not stop.wait(0.05):
            peak = max(peak, _working_set_bytes())

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    started = time.perf_counter()
    try:
        result = operation()
    finally:
        elapsed = time.perf_counter() - started
        peak = max(peak, _working_set_bytes())
        stop.set()
        thread.join()
    return result, {
        "stage": name,
        "elapsed_seconds": float(elapsed),
        "peak_process_working_set_bytes": int(peak),
    }


def main() -> int:
    args = _arguments()
    reset_cuda_audit()
    stages = args.stages or ["inspection", "metric"]
    session = load_rgbd_session(args.session)
    frame_ids, poses = _load_poses(args.transforms)
    by_id = {int(frame.frame_id): frame for frame in session.frames}
    missing = sorted(set(frame_ids) - set(by_id))
    if missing:
        raise ValueError(f"Session is missing transform frames: {missing}")
    frames = [by_id[frame_id] for frame_id in frame_ids]
    temporary_surrogate: tempfile.TemporaryDirectory[str] | None = None
    if args.planar_surrogate_depth_mm is not None:
        surrogate_depth = float(args.planar_surrogate_depth_mm)
        if (
            not np.isfinite(surrogate_depth)
            or surrogate_depth <= 0.0
            or surrogate_depth > np.iinfo(np.uint16).max
        ):
            raise ValueError(
                "planar-surrogate-depth-mm must fit a positive uint16 depth"
            )
        temporary_surrogate = tempfile.TemporaryDirectory(
            prefix="g305-v1-long-span-"
        )
        surrogate_path = (
            Path(temporary_surrogate.name) / "aligned_depth.png"
        )
        depth_image = np.full(
            (session.calibration.height, session.calibration.width),
            int(round(surrogate_depth)),
            dtype=np.uint16,
        )
        if not cv2.imwrite(str(surrogate_path), depth_image):
            raise RuntimeError("Could not write planar surrogate depth")
        source_color = frames[len(frames) // 2].color_path
        frames = [
            replace(
                frame,
                color_path=source_color,
                aligned_depth_path=surrogate_path,
                depth_scale_mm_per_unit=1.0,
            )
            for frame in frames
        ]
    scaled_poses, scale_audit = _scale_longitudinal_span(
        poses,
        float(args.target_span_mm),
    )
    config_payload = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    stitch = dict(config_payload["stitch"])
    report: dict[str, object] = {
        "schema": "gemini305-v1-long-span-performance-stress/v1",
        "status": "running",
        "evidence_scope": "renderer_performance_only",
        "formal_publication": False,
        "physical_long_scan_quality_evidence": False,
        "warning": (
            "RGB-D pixels come from the shorter real scan while only the "
            "longitudinal translations are scaled. Do not use these pixels "
            "or scores as long-distance tracking or panorama-quality evidence."
        ),
        "session": str(session.root),
        "transforms": str(args.transforms.resolve()),
        "frame_count": len(frames),
        "frame_ids": frame_ids,
        "trajectory_scaling": scale_audit,
        "planar_surrogate": {
            "enabled": args.planar_surrogate_depth_mm is not None,
            "constant_depth_mm": (
                None
                if args.planar_surrogate_depth_mm is None
                else float(args.planar_surrogate_depth_mm)
            ),
            "single_real_rgb_reused": (
                args.planar_surrogate_depth_mm is not None
            ),
            "physical_quality_evidence": False,
        },
        "stages_requested": list(stages),
        "stage_results": {},
    }
    overall_started = time.perf_counter()
    try:
        if "inspection" in stages:
            result, timing = _measure_stage(
                "inspection",
                lambda: render_inspection_multiview(
                    frames,
                    scaled_poses,
                    session.calibration,
                    config=stitch["inspection_multiview"],
                ),
            )
            timing.update(
                {
                    "image_shape": list(result.image_bgr.shape),
                    "panel_count": int(
                        result.metadata["layout"]["panel_count"]
                    ),
                    "strict_v1_inspection_complete": bool(
                        result.metadata["strict_v1_inspection_complete"]
                    ),
                    "resource_estimate": result.metadata[
                        "resource_estimate"
                    ],
                    "reference_inverse_maps": result.metadata[
                        "reference_inverse_maps"
                    ],
                }
            )
            report["stage_results"]["inspection"] = timing
            del result
            gc.collect()
        if "metric" in stages:
            result, timing = _measure_stage(
                "metric",
                lambda: render_metric_mosaic(
                    frames,
                    scaled_poses,
                    session.calibration,
                    config=stitch["metric_mosaic"],
                ),
            )
            timing.update(
                {
                    "image_shape": list(result.image_bgr.shape),
                    "canvas": result.metadata["canvas"],
                    "valid_pixel_count": int(
                        result.metadata["valid_pixel_count"]
                    ),
                    "estimated_peak_bytes": int(
                        result.metadata["estimated_peak_bytes"]
                    ),
                }
            )
            report["stage_results"]["metric"] = timing
            del result
            gc.collect()
        report["status"] = "passed"
    except Exception as exc:
        report["status"] = "failed"
        report["failure_type"] = type(exc).__name__
        report["failure_reason"] = str(exc)
        raise
    finally:
        report["elapsed_seconds"] = float(
            time.perf_counter() - overall_started
        )
        report["acceleration"] = cuda_status().as_dict()
        encoded = json.dumps(report, ensure_ascii=False, indent=2)
        if args.output is not None:
            output = args.output.resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            pending = output.with_name(f".{output.name}.pending")
            pending.write_text(encoded + "\n", encoding="utf-8")
            os.replace(pending, output)
        print(encoded)
        if temporary_surrogate is not None:
            temporary_surrogate.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
