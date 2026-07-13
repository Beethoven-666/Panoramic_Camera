from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import cv2

from .config import load_config
from .errors import AlignmentError
from .quality import (
    MotionEstimate,
    assess_capture_quality,
    analyze_frame_quality,
    estimate_translation,
    resize_for_analysis,
    select_layout_from_motion_estimates,
    select_primary_scan_segment,
    select_render_indices_auto,
)
from .render import render_panorama, render_scan_panorama
from .session import SessionFrame, discover_frames, select_frames
from .stitch_common import build_aligner, read_bgr, write_bgr


_DELIVERY_FILES = (
    "delivery.json",
    "panorama.jpg",
    "report.json",
    "transforms.json",
    "render_transforms.json",
)


def _clear_delivery_files(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for name in _DELIVERY_FILES:
        (output / name).unlink(missing_ok=True)
    for pending in output.glob(".*.pending.*"):
        if pending.is_file():
            pending.unlink()


def _write_failure_report(output: Path, input_path: Path, exc: Exception) -> None:
    _clear_delivery_files(output)
    payload = {
        "schema": "gemini305-panorama-failure/v1",
        "failed_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(input_path.expanduser().resolve()),
        "error_type": type(exc).__name__,
        "message": str(exc),
        "deliverable_published": False,
    }
    pending = output / ".failure.pending.json"
    pending.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(pending, output / "failure.json")


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
        "--blend-mode",
        choices=("feather", "scan_seam"),
        help="Final sequence renderer (scan_seam avoids averaging all overlaps)",
    )
    parser.add_argument(
        "--render-frame-ids",
        help=(
            "Reserved diagnostic override; the delivery command rejects manual "
            "source selection"
        ),
    )
    parser.add_argument(
        "--translation-anchor-y",
        type=float,
        help="Vertical image fraction used to extract translation from each homography",
    )
    parser.add_argument("--scan-seam-margin", type=int)
    parser.add_argument("--scan-multiband-levels", type=int)
    parser.add_argument(
        "--scan-exposure-mode",
        choices=("none", "center_gain", "global_gain"),
    )
    parser.add_argument("--scan-seam-mask-sigma", type=float)
    parser.add_argument(
        "--scan-protect-region",
        action="append",
        metavar="FRAME_ID:X0:Y0:X1:Y1",
        help=(
            "Force one frame to own a near-field canvas rectangle; repeat for "
            "multiple objects"
        ),
    )
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
    translation_anchor_y: float | None = None,
) -> np.ndarray:
    """Project a pair homography onto the cart's mechanical motion model."""

    matrix = _normalise(homography)
    if motion_model == "homography":
        return matrix
    height, width = image_shape[:2]
    if translation_anchor_y is not None and not 0.0 <= translation_anchor_y <= 1.0:
        raise ValueError("translation_anchor_y must be between zero and one")
    if motion_model == "translation" and translation_anchor_y is not None:
        xs = np.linspace(width * 0.25, width * 0.75, 3)[None, :]
        ys = np.full_like(xs, height * translation_anchor_y)
    else:
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


def _parse_frame_ids(value: object) -> list[int]:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
        result = [int(part) for part in parts]
    elif isinstance(value, (list, tuple)):
        result = [int(item) for item in value]
    else:
        raise ValueError("render_frame_ids must be a comma-separated string or list")
    if len(result) != len(set(result)):
        raise ValueError("render_frame_ids cannot contain duplicates")
    if len(result) == 1:
        raise ValueError("render_frame_ids must contain at least two frame ids")
    return sorted(result)


def _ensure_publishable_quality(
    capture_quality: dict[str, object], render_metadata: dict[str, Any]
) -> None:
    """Keep diagnostic gate overrides from publishing an official delivery."""

    failures: list[str] = []
    if not bool(capture_quality.get("quality_pass", False)):
        failures.append("input capture quality did not pass")
    render_quality = render_metadata.get("quality_metrics")
    if not isinstance(render_quality, dict) or not bool(
        render_quality.get("quality_pass", False)
    ):
        failures.append("final render quality did not pass")
    if failures:
        raise RuntimeError("Delivery quality gate failed: " + "; ".join(failures))


def _parse_protected_regions(
    values: object,
    frames: list[SessionFrame],
) -> list[tuple[int, int, int, int, int]]:
    if values is None or values == "":
        return []
    rows = [values] if isinstance(values, str) else list(values)
    frame_index = {frame.frame_id: index for index, frame in enumerate(frames)}
    result: list[tuple[int, int, int, int, int]] = []
    for row in rows:
        parts = str(row).split(":")
        if len(parts) != 5:
            raise ValueError(
                "scan_protect_region must be FRAME_ID:X0:Y0:X1:Y1"
            )
        frame_id, x0, y0, x1, y1 = (int(part) for part in parts)
        if frame_id not in frame_index:
            raise ValueError(
                f"Protected-region frame {frame_id} is not a render frame"
            )
        result.append((frame_index[frame_id], x0, y0, x1, y1))
    return result


def interpolate_translation_transforms(
    layout_frames: list[SessionFrame],
    layout_transforms: list[np.ndarray],
    render_frames: list[SessionFrame],
) -> list[np.ndarray]:
    """Interpolate a dense frame's translation from the verified layout trajectory."""

    if len(layout_frames) != len(layout_transforms) or not layout_frames:
        raise ValueError("Layout frames and transforms must be non-empty and equal length")
    use_timestamps = all(frame.timestamp_us is not None for frame in layout_frames) and all(
        frame.timestamp_us is not None for frame in render_frames
    )
    coordinates = np.asarray(
        [
            float(frame.timestamp_us) if use_timestamps else float(frame.frame_id)
            for frame in layout_frames
        ],
        dtype=np.float64,
    )
    if np.any(np.diff(coordinates) <= 0):
        raise ValueError("Layout frame ids must be strictly increasing")
    matrices = [_normalise(transform) for transform in layout_transforms]
    for matrix in matrices:
        if not np.allclose(matrix[:2, :2], np.eye(2), atol=1e-6) or not np.allclose(
            matrix[2], [0.0, 0.0, 1.0], atol=1e-8
        ):
            raise ValueError("Dense render-frame interpolation requires translation layout")
    requested = np.asarray(
        [
            float(frame.timestamp_us) if use_timestamps else float(frame.frame_id)
            for frame in render_frames
        ],
        dtype=np.float64,
    )
    if requested.size == 0:
        raise ValueError("At least one render frame is required")
    if requested.min() < coordinates[0] or requested.max() > coordinates[-1]:
        raise ValueError(
            "Render frame ids must lie inside the aligned layout frame range "
            f"{int(coordinates[0])}-{int(coordinates[-1])}"
        )
    tx = np.asarray([matrix[0, 2] for matrix in matrices], dtype=np.float64)
    ty = np.asarray([matrix[1, 2] for matrix in matrices], dtype=np.float64)
    result: list[np.ndarray] = []
    for x, y in zip(
        np.interp(requested, coordinates, tx),
        np.interp(requested, coordinates, ty),
        strict=True,
    ):
        result.append(
            np.array(
                [[1.0, 0.0, float(x)], [0.0, 1.0, float(y)], [0.0, 0.0, 1.0]],
                dtype=np.float64,
            )
        )
    return result


def interpolate_motion_guided_transforms(
    layout_frames: list[SessionFrame],
    layout_transforms: list[np.ndarray],
    render_frames: list[SessionFrame],
    scan_frames: list[SessionFrame],
    scan_motions: list[MotionEstimate],
    scan_direction: int,
) -> list[np.ndarray]:
    """Interpolate dense poses by observed progress instead of assumed speed."""

    if len(scan_motions) != len(scan_frames) - 1 or scan_direction not in {-1, 1}:
        raise ValueError("Motion-guided interpolation inputs are inconsistent")
    scan_index = {frame.frame_id: index for index, frame in enumerate(scan_frames)}
    try:
        layout_indices = [scan_index[frame.frame_id] for frame in layout_frames]
        render_indices = [scan_index[frame.frame_id] for frame in render_frames]
    except KeyError as exc:
        raise ValueError("A dense render frame lies outside the selected scan segment") from exc
    progress = np.zeros(len(scan_frames), dtype=np.float64)
    recent_steps: list[float] = []
    for index, motion in enumerate(scan_motions):
        if motion.reliable:
            step = max(0.0, float(motion.dx) * scan_direction)
            recent_steps.append(step)
        else:
            step = float(np.median(recent_steps[-5:])) if recent_steps else 0.0
        progress[index + 1] = progress[index] + step
    anchor_progress = progress[layout_indices]
    if np.any(np.diff(anchor_progress) <= 1e-6):
        return interpolate_translation_transforms(
            layout_frames, layout_transforms, render_frames
        )
    matrices = [_normalise(transform) for transform in layout_transforms]
    for matrix in matrices:
        if not np.allclose(matrix[:2, :2], np.eye(2), atol=1e-6) or not np.allclose(
            matrix[2], [0.0, 0.0, 1.0], atol=1e-8
        ):
            raise ValueError("Motion-guided render interpolation requires translation layout")
    requested_progress = progress[render_indices]
    if (
        requested_progress.min() < anchor_progress[0] - 1e-6
        or requested_progress.max() > anchor_progress[-1] + 1e-6
    ):
        raise ValueError("Render frames must lie inside the aligned motion range")
    tx = np.asarray([matrix[0, 2] for matrix in matrices], dtype=np.float64)
    ty = np.asarray([matrix[1, 2] for matrix in matrices], dtype=np.float64)
    return [
        np.array(
            [[1.0, 0.0, float(x)], [0.0, 1.0, float(y)], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        for x, y in zip(
            np.interp(requested_progress, anchor_progress, tx),
            np.interp(requested_progress, anchor_progress, ty),
            strict=True,
        )
    ]


def _center_sharpness(image: np.ndarray) -> float:
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    roi = gray[
        int(round(height * 0.325)) : int(round(height * 0.95)),
        int(round(width * 0.345)) : int(round(width * 0.655)),
    ]
    return float(cv2.Laplacian(roi, cv2.CV_64F).var())


def _read_aligned_depth_mm(frame: SessionFrame) -> np.ndarray | None:
    if frame.depth_path is None:
        return None
    if (
        frame.depth_scale_mm_per_unit is None
        or not np.isfinite(frame.depth_scale_mm_per_unit)
        or frame.depth_scale_mm_per_unit <= 0.0
    ):
        raise ValueError(
            f"Frame {frame.frame_id} has aligned depth but no valid depth scale"
        )
    encoded = np.fromfile(frame.depth_path, dtype=np.uint8)
    depth = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    if depth is None or depth.ndim != 2:
        raise OSError(f"OpenCV could not decode aligned depth: {frame.depth_path}")
    return depth.astype(np.float32) * float(frame.depth_scale_mm_per_unit)


def select_scan_keyframes(
    frames: list[SessionFrame],
    images: list[np.ndarray],
    transforms: list[np.ndarray],
    max_keyframes: int,
) -> list[int]:
    """Pick sharp sources near evenly spaced scan positions."""

    if max_keyframes < 2:
        raise ValueError("scan_max_keyframes must be at least two")
    if len(frames) <= max_keyframes:
        return list(range(len(frames)))
    centers = []
    for image, transform in zip(images, transforms, strict=True):
        height, width = image.shape[:2]
        point = _normalise(transform) @ np.array([width * 0.5, height * 0.5, 1.0])
        centers.append(point[:2] / point[2])
    center_array = np.asarray(centers, dtype=np.float64)
    centered = center_array - center_array.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axis = vh[0]
    position = center_array @ axis
    targets = np.linspace(position.min(), position.max(), max_keyframes)
    spacing = max(float(np.ptp(position)) / max(1, max_keyframes - 1), 1.0)
    sharpness = np.asarray([_center_sharpness(image) for image in images])
    selected: list[int] = []
    for target in targets:
        distance = np.abs(position - target)
        candidates = np.flatnonzero(distance <= spacing * 0.65)
        if candidates.size == 0:
            candidates = np.asarray([int(np.argmin(distance))])
        score = sharpness[candidates] / (
            1.0 + 0.3 * np.square(distance[candidates] / (spacing * 0.65))
        )
        selected.append(int(candidates[int(np.argmax(score))]))
    return sorted(set(selected), key=lambda index: frames[index].frame_id)


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    stitch_config = dict(config["stitch"])
    output = args.output.expanduser().resolve()
    _clear_delivery_files(output)
    (output / "failure.json").unlink(missing_ok=True)
    if args.strict_unistitch:
        stitch_config["allow_magsac_fallback"] = False
    adaptive_layout = bool(stitch_config.get("adaptive_layout", True)) and (
        args.stride is None and args.max_frames is None
    )
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
    blend_mode = getattr(args, "blend_mode", None) or str(
        stitch_config.get("sequence_blend_mode", "scan_seam")
    )
    if blend_mode not in {"feather", "scan_seam"}:
        raise ValueError(f"Unsupported sequence blend mode: {blend_mode}")
    if blend_mode == "feather":
        raise ValueError(
            "feather is a diagnostic renderer and cannot publish a quality-gated delivery"
        )
    translation_anchor_y_value = getattr(args, "translation_anchor_y", None)
    configured_anchor = stitch_config.get("translation_anchor_y")
    translation_anchor_y = (
        float(translation_anchor_y_value)
        if translation_anchor_y_value is not None
        else (float(configured_anchor) if configured_anchor is not None else None)
    )
    scan_seam_margin_value = getattr(args, "scan_seam_margin", None)
    scan_seam_margin = (
        int(scan_seam_margin_value)
        if scan_seam_margin_value is not None
        else int(stitch_config.get("scan_seam_margin", 220))
    )
    scan_multiband_value = getattr(args, "scan_multiband_levels", None)
    scan_multiband_levels = (
        int(scan_multiband_value)
        if scan_multiband_value is not None
        else int(stitch_config.get("scan_multiband_levels", 4))
    )
    scan_exposure_mode = getattr(args, "scan_exposure_mode", None) or str(
        stitch_config.get("scan_exposure_mode", "center_gain")
    )
    scan_seam_mask_sigma_value = getattr(args, "scan_seam_mask_sigma", None)
    scan_seam_mask_sigma = (
        float(scan_seam_mask_sigma_value)
        if scan_seam_mask_sigma_value is not None
        else float(stitch_config.get("scan_seam_mask_sigma", 0.0))
    )
    protected_region_specs = getattr(args, "scan_protect_region", None)
    if protected_region_specs is None:
        protected_region_specs = stitch_config.get("scan_protected_regions")
    scan_auto_foreground = bool(stitch_config.get("scan_auto_foreground", True))
    scan_quality_gate = bool(stitch_config.get("scan_quality_gate", True))
    scan_max_keyframes = int(stitch_config.get("scan_max_keyframes", 14))
    render_frame_ids = _parse_frame_ids(
        getattr(args, "render_frame_ids", None)
        or stitch_config.get("sequence_render_frame_ids")
    )
    if render_frame_ids:
        raise ValueError(
            "render_frame_ids is a diagnostic override and cannot publish a "
            "complete quality-gated delivery"
        )
    save_pair_previews = not args.no_pair_previews and bool(
        stitch_config.get("save_pair_previews", True)
    )
    motion_model = args.motion_model or str(
        stitch_config.get("sequence_motion_model", "translation")
    )
    if motion_model not in {"translation", "similarity", "homography"}:
        raise ValueError(f"Unsupported sequence motion model: {motion_model}")

    all_frames = discover_frames(args.input)
    frame_ids = [frame.frame_id for frame in all_frames]
    if len(frame_ids) != len(set(frame_ids)) or any(
        later <= earlier for earlier, later in zip(frame_ids, frame_ids[1:])
    ):
        raise ValueError("Input frame ids must be unique and strictly increasing")
    analysis_width = int(stitch_config.get("analysis_width", 320))
    frame_qualities = []
    motion_estimates = []
    previous_analysis: np.ndarray | None = None
    analysis_shape: tuple[int, ...] | None = None
    for frame in all_frames:
        analysis = resize_for_analysis(read_bgr(frame.color_path), analysis_width)
        if analysis_shape is None:
            analysis_shape = analysis.shape
        elif analysis.shape != analysis_shape:
            raise ValueError(
                f"Frame {frame.frame_id} has inconsistent analysis shape {analysis.shape}"
            )
        frame_qualities.append(analyze_frame_quality(analysis))
        if adaptive_layout and previous_analysis is not None:
            motion_estimates.append(estimate_translation(previous_analysis, analysis))
        previous_analysis = analysis
    scan_frames = all_frames
    scan_qualities = frame_qualities
    scan_motions = motion_estimates
    segment_metadata: dict[str, Any] | None = None
    if adaptive_layout:
        assert analysis_shape is not None
        segment = select_primary_scan_segment(
            motion_estimates,
            image_width=analysis_shape[1],
            maximum_fraction=float(
                stitch_config.get("layout_max_displacement_fraction", 0.28)
            ),
        )
        scan_frames = all_frames[segment.start_index : segment.end_index + 1]
        scan_qualities = frame_qualities[
            segment.start_index : segment.end_index + 1
        ]
        scan_motions = motion_estimates[segment.start_index : segment.end_index]
        segment_metadata = segment.as_dict()
        segment_metadata["start_frame_id"] = scan_frames[0].frame_id
        segment_metadata["end_frame_id"] = scan_frames[-1].frame_id
    capture_quality = assess_capture_quality(
        scan_qualities,
        [frame.color_exposure_raw for frame in scan_frames],
        exposure_unit_us=float(stitch_config.get("color_exposure_unit_us", 100.0)),
        maximum_exposure_us=float(
            stitch_config.get("maximum_motion_exposure_us", 1200.0)
        ),
    )
    if bool(stitch_config.get("input_quality_gate", True)) and not bool(
        capture_quality["quality_pass"]
    ):
        reasons = "; ".join(str(value) for value in capture_quality["failure_reasons"])
        raise RuntimeError("Input capture quality gate failed: " + reasons)
    quality_by_id = {
        frame.frame_id: quality
        for frame, quality in zip(all_frames, frame_qualities, strict=True)
    }
    layout_metadata: dict[str, Any]
    if adaptive_layout:
        assert analysis_shape is not None
        adaptive_selection = select_layout_from_motion_estimates(
            scan_motions,
            frame_count=len(scan_frames),
            image_width=analysis_shape[1],
            target_fraction=float(
                stitch_config.get("layout_target_displacement_fraction", 0.18)
            ),
            maximum_fraction=float(
                stitch_config.get("layout_max_displacement_fraction", 0.28)
            ),
            max_selected=int(stitch_config.get("layout_max_frames", 160)),
        )
        frames = [scan_frames[index] for index in adaptive_selection.indices]
        layout_metadata = adaptive_selection.as_dict()
        layout_metadata["mode"] = "adaptive_visual_motion"
        layout_metadata["frame_ids"] = [frame.frame_id for frame in frames]
        layout_metadata["segment"] = segment_metadata
        layout_metadata["motion"] = [motion.as_dict() for motion in scan_motions]
    else:
        frames = select_frames(all_frames, stride=stride, max_frames=max_frames)
        layout_metadata = {
            "mode": "fixed_stride_override",
            "stride": stride,
            "max_frames": max_frames,
            "frame_ids": [frame.frame_id for frame in frames],
        }
    if len(frames) < 2:
        raise ValueError("Sequence stitching requires at least two selected color frames")
    images = [read_bgr(frame.color_path) for frame in frames]
    expected_shape = images[0].shape
    for frame, image in zip(frames, images, strict=True):
        if image.shape != expected_shape:
            raise ValueError(
                f"Frame {frame.frame_id} has shape {image.shape}; expected {expected_shape}"
            )
    if scan_seam_margin <= 0:
        scan_seam_margin = max(48, int(round(expected_shape[1] * 0.17)))

    pair_dir = output / "pairs"
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
            translation_anchor_y=translation_anchor_y,
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

    render_metadata: dict[str, Any]
    if blend_mode == "scan_seam":
        if render_frame_ids:
            by_id = {frame.frame_id: frame for frame in all_frames}
            missing_ids = [frame_id for frame_id in render_frame_ids if frame_id not in by_id]
            if missing_ids:
                raise ValueError(f"Render frame ids are missing from the input: {missing_ids}")
            render_frames = [by_id[frame_id] for frame_id in render_frame_ids]
            render_transforms = (
                interpolate_motion_guided_transforms(
                    frames,
                    transforms,
                    render_frames,
                    scan_frames,
                    scan_motions,
                    adaptive_selection.scan_direction,
                )
                if adaptive_layout
                else interpolate_translation_transforms(
                    frames, transforms, render_frames
                )
            )
            render_images = [read_bgr(frame.color_path) for frame in render_frames]
        else:
            dense_frames = [
                frame
                for frame in all_frames
                if frames[0].frame_id <= frame.frame_id <= frames[-1].frame_id
            ]
            dense_transforms = (
                interpolate_motion_guided_transforms(
                    frames,
                    transforms,
                    dense_frames,
                    scan_frames,
                    scan_motions,
                    adaptive_selection.scan_direction,
                )
                if adaptive_layout
                else interpolate_translation_transforms(
                    frames,
                    transforms,
                    dense_frames,
                )
            )
            dense_qualities = [quality_by_id[frame.frame_id] for frame in dense_frames]
            indices, render_selection = select_render_indices_auto(
                dense_qualities,
                dense_transforms,
                expected_shape,
                maximum_keyframes=scan_max_keyframes,
                target_spacing_fraction=float(
                    stitch_config.get("scan_target_spacing_fraction", 0.28)
                ),
            )
            render_frames = [dense_frames[index] for index in indices]
            render_images = [read_bgr(frame.color_path) for frame in render_frames]
            render_transforms = [dense_transforms[index] for index in indices]
        for frame, image in zip(render_frames, render_images, strict=True):
            if image.shape != expected_shape:
                raise ValueError(
                    f"Render frame {frame.frame_id} has shape {image.shape}; "
                    f"expected {expected_shape}"
                )
        protected_regions = _parse_protected_regions(
            protected_region_specs,
            render_frames,
        )
        render_depths = [_read_aligned_depth_mm(frame) for frame in render_frames]
        panorama, scan_render = render_scan_panorama(
            render_images,
            render_transforms,
            max_megapixels=max_canvas_megapixels,
            seam_margin=scan_seam_margin,
            multiband_levels=scan_multiband_levels,
            exposure_mode=scan_exposure_mode,
            seam_mask_sigma=scan_seam_mask_sigma,
            protected_regions=protected_regions,
            depth_maps_mm=render_depths,
            auto_foreground=scan_auto_foreground,
            quality_gate=scan_quality_gate,
        )
        canvas = scan_render.canvas
        render_metadata = scan_render.as_dict()
        render_metadata["frame_ids"] = [frame.frame_id for frame in render_frames]
        if not render_frame_ids:
            render_metadata["selection"] = render_selection
        render_metadata["source_quality"] = {
            str(frame.frame_id): quality_by_id[frame.frame_id].as_dict()
            for frame in render_frames
        }
        for region in render_metadata["protected_regions"]:
            region["owner_frame_id"] = render_frames[
                int(region["owner_index"])
            ].frame_id
        for region in render_metadata["automatic_protected_regions"]:
            region["owner_frame_id"] = render_frames[
                int(region["owner_index"])
            ].frame_id
        render_transform_rows = [
            {
                "frame_id": frame.frame_id,
                "color_path": str(frame.color_path),
                "image_to_world": transform.tolist(),
            }
            for frame, transform in zip(
                render_frames, render_transforms, strict=True
            )
        ]
    else:
        panorama, canvas = render_panorama(
            images,
            transforms,
            max_megapixels=max_canvas_megapixels,
        )
        render_metadata = {
            "source_count": len(images),
            "frame_ids": [frame.frame_id for frame in frames],
        }
        render_transform_rows = None
    _ensure_publishable_quality(capture_quality, render_metadata)
    panorama_path = output / "panorama.jpg"
    transform_rows = [
        {
            "frame_id": frame.frame_id,
            "color_path": str(frame.color_path),
            "image_to_world": transform.tolist(),
        }
        for frame, transform in zip(frames, transforms, strict=True)
    ]
    report: dict[str, Any] = {
        "schema": "gemini305-unistitch-sequence/v2",
        "input": str(args.input.expanduser().resolve()),
        "panorama": str(panorama_path),
        "frame_count": len(frames),
        "stride": None if adaptive_layout else stride,
        "layout_selection": layout_metadata,
        "input_quality": capture_quality,
        "sequence_motion_model": motion_model,
        "translation_anchor_y": translation_anchor_y,
        "elapsed_seconds_excluding_model_load": time.perf_counter() - started,
        "canvas": canvas.as_dict(),
        "render_strategy": blend_mode,
        "render": render_metadata,
        "pair_local_warp": "UniStitch FFD/TPS previews",
        "sequence_layout": (
            "Validated UniStitch global homography with preferred lower-error "
            f"LightGlue/MAGSAC layout, projected to {motion_model} motion"
        ),
        "depth_used": bool(
            blend_mode == "scan_seam"
            and any(depth is not None for depth in render_depths)
        ),
        "pairs": pair_reports,
    }
    pending_panorama = write_bgr(output / ".panorama.pending.jpg", panorama)
    pending_transforms = output / ".transforms.pending.json"
    pending_report = output / ".report.pending.json"
    pending_transforms.write_text(
        json.dumps(transform_rows, indent=2), encoding="utf-8"
    )
    pending_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    pending_render_transforms: Path | None = None
    if render_transform_rows is not None:
        pending_render_transforms = output / ".render_transforms.pending.json"
        pending_render_transforms.write_text(
            json.dumps(render_transform_rows, indent=2), encoding="utf-8"
        )
    os.replace(pending_panorama, panorama_path)
    os.replace(pending_transforms, output / "transforms.json")
    if pending_render_transforms is not None:
        os.replace(pending_render_transforms, output / "render_transforms.json")
    os.replace(pending_report, output / "report.json")
    delivery = {
        "schema": "gemini305-panorama-delivery/v1",
        "published_utc": datetime.now(timezone.utc).isoformat(),
        "quality_pass": True,
        "panorama": str(panorama_path),
        "report": str(output / "report.json"),
    }
    pending_delivery = output / ".delivery.pending.json"
    pending_delivery.write_text(json.dumps(delivery, indent=2), encoding="utf-8")
    os.replace(pending_delivery, output / "delivery.json")
    return report


def main() -> None:
    args = _parser().parse_args()
    try:
        report = run(args)
    except Exception as exc:
        _write_failure_report(args.output.expanduser().resolve(), args.input, exc)
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print(f"Panorama: {report['panorama']}")
    print(f"Report: {args.output.expanduser().resolve() / 'report.json'}")


if __name__ == "__main__":
    main()
