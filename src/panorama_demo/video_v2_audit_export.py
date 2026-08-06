"""Read-only central-strip archives for the candidate CUDA v2 renderer.

The primary CUDA data plane remains the only path that creates panorama RGB
or owner provenance.  This module runs only for ``artifact-level=audit``
after that result exists: it recreates the documented calibrated inverse-grid
tiles from the same real source files, applies a recorded accepted C7 scalar
correction when present, and writes BGRA evidence.  It never imports or calls
the historical CPU panorama renderer and has no route back into rendering.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
import shutil
from pathlib import Path
from typing import Mapping, Sequence

import cv2
import numpy as np

from .session import CameraIntrinsics, RGBDFrame
from .video_candidate_annotation_projection import (
    CandidateInverseMapSource,
    build_v2_c1_calibrated_inverse_sources,
)


@dataclass(frozen=True)
class V2CudaAuditExportContext:
    """Immutable real-source/grid inputs for a post-render audit archive."""

    sources: tuple[RGBDFrame, ...]
    strips: tuple[object, ...]
    calibration: CameraIntrinsics
    renderer: str
    include_adjacent_corridors: bool
    c7_export_corrections_bgr: Mapping[int, tuple[tuple[float, ...], tuple[float, ...]]] | None = None


def _linear_to_srgb(linear: np.ndarray) -> np.ndarray:
    return np.where(
        linear <= 0.0031308,
        linear * 12.92,
        1.055 * np.power(np.maximum(linear, 0.0), 1.0 / 2.4) - 0.055,
    )


def _apply_recorded_c7_correction(
    image_bgr: np.ndarray,
    *,
    correction: tuple[tuple[float, ...], tuple[float, ...]] | None,
) -> np.ndarray:
    """Apply the accepted C7 linear-light scalar correction to a BGRA source tile."""

    if correction is None:
        return image_bgr
    gain, bias = (np.asarray(value, dtype=np.float32) for value in correction)
    if gain.shape != (3,) or bias.shape != (3,) or not np.isfinite(gain).all() or not np.isfinite(bias).all():
        raise ValueError("C7 audit export correction must be finite BGR gain/bias triplets")
    srgb = image_bgr.astype(np.float32) / 255.0
    linear = np.where(srgb <= 0.04045, srgb / 12.92, np.power((srgb + 0.055) / 1.055, 2.4))
    corrected = np.clip(linear * gain.reshape(1, 1, 3) + bias.reshape(1, 1, 3), 0.0, 1.0)
    return np.rint(np.clip(_linear_to_srgb(corrected), 0.0, 1.0) * 255.0).astype(np.uint8)


def _decode_bgr(frame: RGBDFrame, calibration: CameraIntrinsics) -> np.ndarray:
    image = cv2.imread(str(frame.color_path), cv2.IMREAD_COLOR)
    if image is None or image.shape != (calibration.height, calibration.width, 3):
        raise RuntimeError(f"Could not decode real RGB source for CUDA audit export: {frame.color_path}")
    return np.ascontiguousarray(image)


def _stage_source_strips(
    context: V2CudaAuditExportContext,
    maps: Sequence[CandidateInverseMapSource],
    output_directory: Path,
) -> dict[str, object]:
    if output_directory.exists():
        raise RuntimeError("CUDA central-strip staging directory already exists")
    source_by_id = {int(frame.frame_id): frame for frame in context.sources}
    output_directory.mkdir(parents=True)
    images: list[dict[str, object]] = []
    try:
        for index, source in enumerate(maps):
            frame = source_by_id.get(int(source.frame_id))
            if frame is None:
                raise RuntimeError("CUDA audit export inverse tile references a non-selected source")
            bgr = _decode_bgr(frame, context.calibration)
            remapped = cv2.remap(
                bgr,
                np.asarray(source.source_map_x, dtype=np.float32),
                np.asarray(source.source_map_y, dtype=np.float32),
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0),
            )
            correction = None if context.c7_export_corrections_bgr is None else context.c7_export_corrections_bgr.get(int(source.frame_id))
            corrected = _apply_recorded_c7_correction(remapped, correction=correction)
            alpha = np.where(np.asarray(source.valid_mask, dtype=bool), 255, 0).astype(np.uint8)
            filename = f"central_strip_{index:04d}_frame_{int(source.frame_id):06d}.png"
            pending = output_directory / f".{filename}.pending.png"
            if not cv2.imwrite(str(pending), np.dstack((corrected, alpha)), [cv2.IMWRITE_PNG_COMPRESSION, 3]):
                raise RuntimeError(f"Could not write CUDA central strip {filename}")
            os.replace(pending, output_directory / filename)
            images.append({
                "filename": filename,
                "source_index": index,
                "frame_id": int(source.frame_id),
                "canvas_x0": int(source.canvas_x0),
                "width": int(corrected.shape[1]),
                "height": int(corrected.shape[0]),
                "valid_pixel_count": int(np.count_nonzero(source.valid_mask)),
                "photometric_correction": "c7_recorded_linear_light" if correction is not None else "identity",
            })
        (output_directory / "manifest.json").write_text(json.dumps({
            "schema": "gemini305-video-v2-cuda-central-strips/v1",
            "image_encoding": "PNG BGRA",
            "alpha_semantics": "255=valid_calibrated_rgb_sample; 0=invalid_remap_sample",
            "renderer": context.renderer,
            "export_mode": "read_only_post_render_exact_v2_grid_reference",
            "images": images,
        }, indent=2), encoding="utf-8")
    except Exception:
        shutil.rmtree(output_directory, ignore_errors=True)
        raise
    return {
        "schema": "gemini305-video-v2-cuda-central-strips/v1",
        "image_count": len(images),
        "image_encoding": "PNG BGRA",
        "alpha_semantics": "255=valid_calibrated_rgb_sample; 0=invalid_remap_sample",
        "renderer": context.renderer,
        "export_mode": "read_only_post_render_exact_v2_grid_reference",
        "primary_pixels_or_owner_modified": False,
    }


def _stage_owner_only(
    panorama_bgr: np.ndarray,
    owner_frame_id: np.ndarray,
    frame_ids: Sequence[int],
    output_directory: Path,
    *,
    renderer: str,
) -> dict[str, object]:
    if output_directory.exists():
        raise RuntimeError("CUDA owner-only central-strip staging directory already exists")
    if panorama_bgr.dtype != np.uint8 or panorama_bgr.shape[:2] != owner_frame_id.shape:
        raise ValueError("CUDA owner-only archive needs final uint8 panorama and matching owner map")
    output_directory.mkdir(parents=True)
    images: list[dict[str, object]] = []
    try:
        for index, frame_id in enumerate(frame_ids):
            mask = owner_frame_id == int(frame_id)
            columns = np.flatnonzero(np.any(mask, axis=0))
            empty = columns.size == 0
            if empty:
                x0, image, alpha = 0, np.zeros((1, 1, 3), dtype=np.uint8), np.zeros((1, 1), dtype=np.uint8)
            else:
                x0, x1 = int(columns[0]), int(columns[-1]) + 1
                image = np.ascontiguousarray(panorama_bgr[:, x0:x1])
                alpha = np.where(mask[:, x0:x1], 255, 0).astype(np.uint8)
            filename = f"central_strip_{index:04d}_frame_{int(frame_id):06d}.png"
            pending = output_directory / f".{filename}.pending.png"
            if not cv2.imwrite(str(pending), np.dstack((image, alpha)), [cv2.IMWRITE_PNG_COMPRESSION, 3]):
                raise RuntimeError(f"Could not write CUDA owner-only central strip {filename}")
            os.replace(pending, output_directory / filename)
            images.append({
                "filename": filename, "source_index": index, "frame_id": int(frame_id),
                "panorama_x0": x0, "width": int(image.shape[1]), "height": int(image.shape[0]),
                "owner_pixel_count": int(np.count_nonzero(mask)), "empty_owner": empty,
            })
        (output_directory / "manifest.json").write_text(json.dumps({
            "schema": "gemini305-video-v2-cuda-central-strips-owner-only/v1",
            "image_encoding": "PNG BGRA",
            "alpha_semantics": "255=final_owner_pixel; 0=not_owned_by_this_source",
            "renderer": renderer,
            "primary_pixels_or_owner_modified": False,
            "images": images,
        }, indent=2), encoding="utf-8")
    except Exception:
        shutil.rmtree(output_directory, ignore_errors=True)
        raise
    return {
        "schema": "gemini305-video-v2-cuda-central-strips-owner-only/v1",
        "image_count": len(images), "image_encoding": "PNG BGRA",
        "alpha_semantics": "255=final_owner_pixel; 0=not_owned_by_this_source",
        "renderer": renderer, "primary_pixels_or_owner_modified": False,
    }


def stage_v2_cuda_audit_exports(
    context: V2CudaAuditExportContext,
    *,
    panorama_bgr: np.ndarray,
    owner_frame_id: np.ndarray,
    central_strip_output_dir: Path,
    owner_only_output_dir: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    """Stage both read-only v2 CUDA BGRA archives before atomic publication."""

    ids = tuple(int(frame.frame_id) for frame in context.sources)
    maps = build_v2_c1_calibrated_inverse_sources(
        strips=context.strips,
        source_shapes={frame_id: (context.calibration.height, context.calibration.width) for frame_id in ids},
        canvas_shape=tuple(int(value) for value in owner_frame_id.shape),
        calibration={
            "fx": context.calibration.fx, "fy": context.calibration.fy,
            "cx": context.calibration.cx, "cy": context.calibration.cy,
            "distortion": context.calibration.distortion,
        },
        annotation_frame_ids=ids,
        include_adjacent_corridors=context.include_adjacent_corridors,
    )
    if tuple(int(item.frame_id) for item in maps) != ids:
        raise RuntimeError("CUDA audit export did not preserve the full real-source sequence")
    source_export = _stage_source_strips(context, maps, central_strip_output_dir)
    owner_export = _stage_owner_only(
        panorama_bgr, owner_frame_id, ids, owner_only_output_dir, renderer=context.renderer
    )
    return source_export, owner_export


__all__ = ["V2CudaAuditExportContext", "stage_v2_cuda_audit_exports"]
