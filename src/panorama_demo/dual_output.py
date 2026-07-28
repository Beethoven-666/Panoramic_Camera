"""Atomic staging and verification for V1.0 metric/inspection products."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import cv2
import numpy as np
import OpenEXR

from .metric_mosaic import MetricMosaicResult


_FINAL_NAMES = (
    "mosaic_metric.png",
    "mosaic_depth.exr",
    "mosaic_confidence.png",
    "mosaic_owner.png",
    "mosaic_meta.json",
    "mosaic_inspection.png",
    "inspection_owner.png",
    "mosaic_inspection_full_extent.png",
    "inspection_full_extent_owner.png",
    "inspection_meta.json",
)


@dataclass(frozen=True)
class StagedDualOutput:
    pending_by_final_name: Mapping[str, Path]
    metric_manifest: dict[str, object]
    inspection_manifest: dict[str, object]

    def commit(self, output: Path) -> None:
        import os

        for name in _FINAL_NAMES:
            os.replace(self.pending_by_final_name[name], output / name)


def dual_output_final_names() -> tuple[str, ...]:
    return _FINAL_NAMES


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_png_checked(path: Path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Could not write staged PNG: {path.name}")
    mode = cv2.IMREAD_UNCHANGED
    decoded = cv2.imread(str(path), mode)
    if decoded is None or decoded.dtype != image.dtype or decoded.shape != image.shape:
        raise RuntimeError(f"Staged PNG round-trip changed shape/dtype: {path.name}")
    if not np.array_equal(decoded, image):
        raise RuntimeError(f"Staged PNG round-trip changed pixels: {path.name}")


def _write_depth_exr_checked(path: Path, depth_mm: np.ndarray) -> None:
    values = np.ascontiguousarray(depth_mm, dtype=np.float32)
    with OpenEXR.File(
        {
            "compression": OpenEXR.ZIP_COMPRESSION,
            "g305DepthUnit": "mm",
            "g305InvalidDepth": "NaN",
        },
        {"Z": values},
    ) as output:
        output.write(str(path))
    with OpenEXR.File(str(path)) as source:
        decoded = np.asarray(source.channels()["Z"].pixels)
    if decoded.dtype != np.float32 or decoded.shape != values.shape:
        raise RuntimeError("Staged EXR round-trip changed depth shape/dtype")
    if not np.array_equal(np.isnan(decoded), np.isnan(values)):
        raise RuntimeError("Staged EXR round-trip changed invalid-depth mask")
    finite = np.isfinite(values)
    if not np.array_equal(decoded[finite], values[finite]):
        raise RuntimeError("Staged EXR round-trip changed metric depth values")


def _owner_png(owner_frame_id: np.ndarray, valid: np.ndarray) -> np.ndarray:
    owner = np.asarray(owner_frame_id, dtype=np.int64)
    mask = np.asarray(valid, dtype=bool)
    if owner.shape != mask.shape:
        raise RuntimeError("Owner raster does not match its valid mask")
    if np.any(owner[mask] < 0) or np.any(owner[mask] > 65534):
        raise RuntimeError("Owner frame ID cannot be encoded in uint16 PNG")
    encoded = np.zeros(owner.shape, dtype=np.uint16)
    encoded[mask] = (owner[mask] + 1).astype(np.uint16)
    return encoded


def stage_dual_output(
    output: Path,
    metric: MetricMosaicResult,
    *,
    inspection_bgr: np.ndarray,
    inspection_owner_frame_id: np.ndarray,
    inspection_full_extent_bgra: np.ndarray,
    inspection_full_extent_owner_frame_id: np.ndarray,
    inspection_metadata: Mapping[str, object],
) -> StagedDualOutput:
    """Stage and read back every V1.0 output before publication."""

    output.mkdir(parents=True, exist_ok=True)
    metric.validate()
    inspection = np.asarray(inspection_bgr)
    inspection_owner = np.asarray(inspection_owner_frame_id, dtype=np.int32)
    if (
        inspection.dtype != np.uint8
        or inspection.ndim != 3
        or inspection.shape[2] != 3
        or inspection_owner.shape != inspection.shape[:2]
    ):
        raise RuntimeError("Inspection RGB and owner products are not aligned")
    inspection_valid = inspection_owner >= 0
    if not np.all(inspection_valid):
        raise RuntimeError("Inspection crop contains an unowned output pixel")
    full_extent = np.asarray(inspection_full_extent_bgra)
    full_extent_owner = np.asarray(
        inspection_full_extent_owner_frame_id, dtype=np.int32
    )
    if (
        full_extent.dtype != np.uint8
        or full_extent.ndim != 3
        or full_extent.shape[2] != 4
        or full_extent_owner.shape != full_extent.shape[:2]
    ):
        raise RuntimeError(
            "Inspection full-extent RGBA and owner products are not aligned"
        )
    full_extent_valid = full_extent_owner >= 0
    if (
        not np.any(full_extent_valid)
        or np.any(full_extent[..., 3][full_extent_valid] != 255)
        or np.any(full_extent[..., 3][~full_extent_valid] != 0)
    ):
        raise RuntimeError(
            "Inspection full-extent alpha does not match its owner map"
        )

    pending = {
        name: output / f".{name}.pending"
        for name in _FINAL_NAMES
    }
    # Keep a real codec suffix at the end of image pending paths.
    pending.update(
        {
            "mosaic_metric.png": output / ".mosaic_metric.pending.png",
            "mosaic_depth.exr": output / ".mosaic_depth.pending.exr",
            "mosaic_confidence.png": output / ".mosaic_confidence.pending.png",
            "mosaic_owner.png": output / ".mosaic_owner.pending.png",
            "mosaic_meta.json": output / ".mosaic_meta.pending.json",
            "mosaic_inspection.png": output / ".mosaic_inspection.pending.png",
            "inspection_owner.png": output / ".inspection_owner.pending.png",
            "mosaic_inspection_full_extent.png": (
                output / ".mosaic_inspection_full_extent.pending.png"
            ),
            "inspection_full_extent_owner.png": (
                output / ".inspection_full_extent_owner.pending.png"
            ),
            "inspection_meta.json": output / ".inspection_meta.pending.json",
        }
    )
    try:
        _write_png_checked(pending["mosaic_metric.png"], metric.image_bgr)
        _write_depth_exr_checked(pending["mosaic_depth.exr"], metric.depth_mm)
        _write_png_checked(
            pending["mosaic_confidence.png"], metric.confidence_u16
        )
        _write_png_checked(
            pending["mosaic_owner.png"],
            _owner_png(metric.owner_frame_id, metric.valid_mask),
        )
        _write_png_checked(pending["mosaic_inspection.png"], inspection)
        _write_png_checked(
            pending["inspection_owner.png"],
            _owner_png(inspection_owner, inspection_valid),
        )
        _write_png_checked(
            pending["mosaic_inspection_full_extent.png"],
            full_extent,
        )
        _write_png_checked(
            pending["inspection_full_extent_owner.png"],
            _owner_png(full_extent_owner, full_extent_valid),
        )

        metric_hashes = {
            name: _sha256(pending[name])
            for name in (
                "mosaic_metric.png",
                "mosaic_depth.exr",
                "mosaic_confidence.png",
                "mosaic_owner.png",
            )
        }
        metric_manifest = {
            **metric.metadata,
            "files": {
                name: {
                    "path": name,
                    "sha256": digest,
                }
                for name, digest in metric_hashes.items()
            },
        }
        pending["mosaic_meta.json"].write_text(
            json.dumps(metric_manifest, indent=2),
            encoding="utf-8",
        )
        decoded_metric_meta = json.loads(
            pending["mosaic_meta.json"].read_text(encoding="utf-8")
        )
        if decoded_metric_meta.get("schema") != "gemini305-metric-mosaic/v1":
            raise RuntimeError("Metric metadata round-trip lost its schema")

        inspection_hashes = {
            name: _sha256(pending[name])
            for name in (
                "mosaic_inspection.png",
                "inspection_owner.png",
                "mosaic_inspection_full_extent.png",
                "inspection_full_extent_owner.png",
            )
        }
        inspection_manifest = {
            "schema": "gemini305-inspection-mosaic/v1",
            "method": (
                "trajectory_constrained_depth_aware_multiview_inspection"
            ),
            "image_shape": list(inspection.shape),
            "owner_encoding": "uint16_frame_id_plus_one_zero_invalid",
            "single_owner_pixel_count": int(np.count_nonzero(inspection_valid)),
            "unowned_pixel_count": int(
                inspection_valid.size - np.count_nonzero(inspection_valid)
            ),
            "full_extent_image_shape": list(full_extent.shape),
            "full_extent_single_owner_pixel_count": int(
                np.count_nonzero(full_extent_valid)
            ),
            "full_extent_transparent_pixel_count": int(
                full_extent_valid.size
                - np.count_nonzero(full_extent_valid)
            ),
            "renderer": dict(inspection_metadata),
            "files": {
                name: {
                    "path": name,
                    "sha256": digest,
                }
                for name, digest in inspection_hashes.items()
            },
        }
        pending["inspection_meta.json"].write_text(
            json.dumps(inspection_manifest, indent=2),
            encoding="utf-8",
        )
        decoded_inspection_meta = json.loads(
            pending["inspection_meta.json"].read_text(encoding="utf-8")
        )
        if (
            decoded_inspection_meta.get("schema")
            != "gemini305-inspection-mosaic/v1"
        ):
            raise RuntimeError("Inspection metadata round-trip lost its schema")
    except Exception:
        for path in pending.values():
            path.unlink(missing_ok=True)
        raise
    return StagedDualOutput(
        pending_by_final_name=pending,
        metric_manifest=metric_manifest,
        inspection_manifest=inspection_manifest,
    )
