from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import cv2

from .config import load_config
from .render import render_panorama
from .session import discover_frames, select_frames
from .stitch_common import build_aligner, read_bgr, write_bgr
from .unistitch_adapter import AlignmentError


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a side-scan panorama from original frames using UniStitch pair edges"
    )
    parser.add_argument("input", type=Path, help="Capture session, frames.csv, or color folder")
    parser.add_argument("--output", type=Path, default=Path("outputs/sequence"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--inference-width", type=int)
    parser.add_argument("--stride", type=int)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--max-canvas-megapixels", type=float)
    parser.add_argument(
        "--motion-model",
        choices=("translation", "similarity", "homography"),
        help="Constraint used when accumulating pair transforms (default: config value)",
    )
    parser.add_argument("--no-pair-previews", action="store_true")
    parser.add_argument(
        "--strict-unistitch",
        action="store_true",
        help="Reject instead of falling back to LightGlue+MAGSAC for sequence layout",
    )
    return parser


def _normalise(matrix: np.ndarray) -> np.ndarray:
    result = np.asarray(matrix, dtype=np.float64)
    if result.shape != (3, 3) or not np.isfinite(result).all():
        raise AlignmentError("Sequence transform is not a finite 3x3 matrix")
    if abs(result[2, 2]) < 1e-10:
        raise AlignmentError("Sequence transform is singular")
    return result / result[2, 2]


def regularize_pair_homography(
    homography: np.ndarray,
    image_shape: tuple[int, ...],
    motion_model: str,
) -> np.ndarray:
    """Project a pair homography onto the cart's mechanical motion model."""

    matrix = _normalise(homography)
    if motion_model == "homography":
        return matrix
    height, width = image_shape[:2]
    xs, ys = np.meshgrid(
        np.linspace(width * 0.1, width * 0.9, 5),
        np.linspace(height * 0.1, height * 0.9, 3),
    )
    source_grid = np.stack((xs.ravel(), ys.ravel()), axis=1).astype(np.float32)
    reference_grid = cv2.perspectiveTransform(source_grid[None], matrix)[0]
    if not np.isfinite(reference_grid).all():
        raise AlignmentError("Pair transform produced non-finite regularization samples")

    if motion_model == "translation":
        displacement = np.median(reference_grid - source_grid, axis=0)
        return np.array(
            [
                [1.0, 0.0, float(displacement[0])],
                [0.0, 1.0, float(displacement[1])],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
    if motion_model == "similarity":
        affine, _ = cv2.estimateAffinePartial2D(
            source_grid,
            reference_grid,
            method=cv2.LMEDS,
        )
        if affine is None:
            raise AlignmentError("Could not project pair transform to a similarity model")
        return _normalise(np.vstack((affine, [0.0, 0.0, 1.0])))
    raise ValueError(f"Unsupported sequence motion model: {motion_model}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    stitch_config = dict(config["stitch"])
    if args.strict_unistitch:
        stitch_config["allow_magsac_fallback"] = False
    stride = args.stride if args.stride is not None else int(stitch_config.get("stride", 1))
    max_frames = (
        args.max_frames
        if args.max_frames is not None
        else int(stitch_config.get("max_frames", 0)) or None
    )
    max_canvas_megapixels = (
        args.max_canvas_megapixels
        if args.max_canvas_megapixels is not None
        else float(stitch_config.get("max_canvas_megapixels", 250.0))
    )
    save_pair_previews = not args.no_pair_previews and bool(
        stitch_config.get("save_pair_previews", True)
    )
    motion_model = args.motion_model or str(
        stitch_config.get("sequence_motion_model", "translation")
    )
    if motion_model not in {"translation", "similarity", "homography"}:
        raise ValueError(f"Unsupported sequence motion model: {motion_model}")

    frames = select_frames(discover_frames(args.input), stride=stride, max_frames=max_frames)
    if len(frames) < 2:
        raise ValueError("Sequence stitching requires at least two selected color frames")
    images = [read_bgr(frame.color_path) for frame in frames]
    expected_shape = images[0].shape
    for frame, image in zip(frames, images, strict=True):
        if image.shape != expected_shape:
            raise ValueError(
                f"Frame {frame.frame_id} has shape {image.shape}; expected {expected_shape}"
            )

    output = args.output.expanduser().resolve()
    pair_dir = output / "pairs"
    output.mkdir(parents=True, exist_ok=True)
    if save_pair_previews:
        pair_dir.mkdir(parents=True, exist_ok=True)

    aligner = build_aligner(
        stitch_config,
        model=args.model,
        device=args.device,
        inference_width=args.inference_width,
    )
    transforms = [np.eye(3, dtype=np.float64)]
    pair_reports: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index in range(1, len(images)):
        pair_started = time.perf_counter()
        alignment = aligner.align(images[index - 1], images[index])
        sequence_pair_h = regularize_pair_homography(
            alignment.homography_source_to_reference,
            images[index].shape,
            motion_model,
        )
        global_transform = _normalise(
            transforms[-1] @ sequence_pair_h
        )
        transforms.append(global_transform)
        pair_report: dict[str, Any] = {
            "pair_index": index - 1,
            "reference_frame_id": frames[index - 1].frame_id,
            "source_frame_id": frames[index].frame_id,
            "reference_path": str(frames[index - 1].color_path),
            "source_path": str(frames[index].color_path),
            "elapsed_seconds": time.perf_counter() - pair_started,
            "sequence_homography_source_to_reference": sequence_pair_h.tolist(),
            **alignment.as_dict(),
        }
        if save_pair_previews:
            preview_path = write_bgr(
                pair_dir / f"{index - 1:04d}_{index:04d}.jpg",
                alignment.preview_bgr,
            )
            pair_report["preview_path"] = str(preview_path)
        pair_reports.append(pair_report)
        print(
            f"[{index}/{len(images) - 1}] frames "
            f"{frames[index - 1].frame_id}->{frames[index].frame_id}: "
            f"{alignment.layout_method}, {alignment.match_count} matches, "
            f"{alignment.median_reprojection_px:.2f}px"
        )

    panorama, canvas = render_panorama(
        images,
        transforms,
        max_megapixels=max_canvas_megapixels,
    )
    panorama_path = write_bgr(output / "panorama.jpg", panorama)
    transform_rows = [
        {
            "frame_id": frame.frame_id,
            "color_path": str(frame.color_path),
            "image_to_world": transform.tolist(),
        }
        for frame, transform in zip(frames, transforms, strict=True)
    ]
    (output / "transforms.json").write_text(
        json.dumps(transform_rows, indent=2), encoding="utf-8"
    )
    report: dict[str, Any] = {
        "schema": "gemini305-unistitch-sequence/v1",
        "input": str(args.input.expanduser().resolve()),
        "panorama": str(panorama_path),
        "frame_count": len(frames),
        "stride": stride,
        "sequence_motion_model": motion_model,
        "elapsed_seconds_excluding_model_load": time.perf_counter() - started,
        "canvas": canvas.as_dict(),
        "render_strategy": "original_frames_single_pass",
        "pair_local_warp": "UniStitch FFD/TPS previews",
        "sequence_layout": (
            "UniStitch global homography with optional MAGSAC validation fallback, "
            f"projected to {motion_model} motion"
        ),
        "depth_used": False,
        "pairs": pair_reports,
    }
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    args = _parser().parse_args()
    try:
        report = run(args)
    except (
        AlignmentError,
        FileNotFoundError,
        MemoryError,
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print(f"Panorama: {report['panorama']}")
    print(f"Report: {args.output.expanduser().resolve() / 'report.json'}")


if __name__ == "__main__":
    main()
