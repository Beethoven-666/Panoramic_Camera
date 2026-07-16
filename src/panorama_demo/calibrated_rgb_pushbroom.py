"""Streaming, calibrated RGB-only pushbroom panorama renderer.

This module deliberately separates *pose evidence* from *pixel generation*.
The caller supplies already-validated, real camera-to-world SE(3) poses (normally
from the RGB-D Open3D/ORB-SLAM3 stages), but the renderer reads only colour files.
It neither reads aligned depth nor constructs a point cloud, TSDF, reference plane,
homography, accumulated 2-D motion, or interpolated pose.

Each source is inverse-remapped exactly once from raw calibrated RGB into a narrow
pose-levelled strip.  Narrow strips are spooled to a temporary directory, allowing
the exposure and seam passes to keep only adjacent strips in memory.  This is a
pushbroom image: its scan-coordinate scale is a robust *local RGB motion per real
camera-centre displacement* estimate, not an estimated 2-D camera trajectory.
"""

from __future__ import annotations

import math
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import cv2
import numpy as np

from .calibrated_remap import camera_points_to_source_pixels
from .render import largest_valid_rectangle
from .rgb_residual_alignment import (
    ResidualAlignmentConfig,
    ResidualAlignmentResult,
    SourceResidualWarp,
    audit_source_warps,
    extract_protected_component_fragments,
    extract_pair_evidence,
    measure_owner_boundary_geometry,
    preflight_sequence_owners,
    solve_background_se2,
)
from .session import CameraIntrinsics, RGBDFrame


_HARD_MAX_POSES = 160
_HARD_MAX_CANVAS_MEGAPIXELS = 200.0
_HARD_MAX_RESIDENT_STRIPS = 5


@dataclass(frozen=True)
class CalibratedRGBPushbroomConfig:
    """Closed safety/configuration surface for the RGB pushbroom renderer."""

    maximum_central_band_fraction: float = 0.20
    endpoint_outer_half_fov: bool = True
    # This is deliberately a *read-only* seam-search corridor, not a blend
    # width.  The two adjacent source strips may each reserve half of it as
    # calibrated support while their hard-owned cores remain unchanged.  The
    # actual common corridor is audited because the 20% central-strip limit
    # can make a requested width unavailable on sparse captures.
    seam_search_width_pixels: int = 64
    max_canvas_megapixels: float = 200.0
    max_aggregate_megapixels: float = 200.0
    max_pose_count: int = _HARD_MAX_POSES
    max_resident_frames: int = _HARD_MAX_RESIDENT_STRIPS
    minimum_valid_scale_pairs: int = 3
    scale_central_fraction: float = 0.20
    scale_low_gradient_quantile: float = 0.45
    scale_minimum_response: float = 0.10
    scale_max_relative_mad: float = 0.35
    residual_alignment: ResidualAlignmentConfig = field(
        default_factory=ResidualAlignmentConfig
    )

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, object] | None
    ) -> "CalibratedRGBPushbroomConfig":
        supplied = {} if value is None else dict(value)
        supplied.pop("mode", None)
        alignment_value = supplied.pop("residual_alignment", None)
        allowed = {
            "maximum_central_band_fraction",
            "endpoint_outer_half_fov",
            "seam_search_width_pixels",
            "max_canvas_megapixels",
            "max_aggregate_megapixels",
            "max_pose_count",
            "max_resident_frames",
            "minimum_valid_scale_pairs",
            "scale_central_fraction",
            "scale_low_gradient_quantile",
            "scale_minimum_response",
            "scale_max_relative_mad",
        }
        unknown = sorted(set(supplied) - allowed)
        if unknown:
            raise ValueError(
                "Unknown calibrated_rgb_pushbroom configuration keys: "
                + ", ".join(unknown)
            )
        try:
            result = cls(
                **supplied,
                residual_alignment=(
                    alignment_value
                    if isinstance(alignment_value, ResidualAlignmentConfig)
                    else ResidualAlignmentConfig.from_mapping(alignment_value)
                ),
            )
        except TypeError as exc:
            raise ValueError("Invalid calibrated_rgb_pushbroom configuration") from exc
        result._validate()
        return result

    def _validate(self) -> None:
        finite = np.asarray(
            [
                self.maximum_central_band_fraction,
                self.max_canvas_megapixels,
                self.max_aggregate_megapixels,
                self.scale_central_fraction,
                self.scale_low_gradient_quantile,
                self.scale_minimum_response,
                self.scale_max_relative_mad,
            ],
            dtype=np.float64,
        )
        if not np.isfinite(finite).all():
            raise ValueError("Calibrated RGB pushbroom settings must be finite")
        if not 0.0 < self.maximum_central_band_fraction <= 0.20:
            raise ValueError("maximum_central_band_fraction must be in (0, 0.20]")
        if type(self.endpoint_outer_half_fov) is not bool:
            raise ValueError("endpoint_outer_half_fov must be a boolean")
        if not 32 <= int(self.seam_search_width_pixels) <= 64:
            raise ValueError("seam_search_width_pixels must be in [32, 64]")
        if not 0.0 < self.max_canvas_megapixels <= _HARD_MAX_CANVAS_MEGAPIXELS:
            raise ValueError("max_canvas_megapixels must be in (0, 200]")
        if not 0.0 < self.max_aggregate_megapixels <= _HARD_MAX_CANVAS_MEGAPIXELS:
            raise ValueError("max_aggregate_megapixels must be in (0, 200]")
        if not 2 <= int(self.max_pose_count) <= _HARD_MAX_POSES:
            raise ValueError("max_pose_count must be in [2, 160]")
        if not 2 <= int(self.max_resident_frames) <= _HARD_MAX_RESIDENT_STRIPS:
            raise ValueError("max_resident_frames must be in [2, 5]")
        if int(self.minimum_valid_scale_pairs) < 1:
            raise ValueError("minimum_valid_scale_pairs must be positive")
        if not 0.0 < self.scale_central_fraction <= 1.0:
            raise ValueError("scale_central_fraction must be in (0, 1]")
        if not 0.0 < self.scale_low_gradient_quantile <= 1.0:
            raise ValueError("scale_low_gradient_quantile must be in (0, 1]")
        if self.scale_minimum_response < 0.0:
            raise ValueError("scale_minimum_response cannot be negative")
        if not 0.0 <= self.scale_max_relative_mad <= 1.0:
            raise ValueError("scale_max_relative_mad must be in [0, 1]")
        if not isinstance(self.residual_alignment, ResidualAlignmentConfig):
            raise ValueError("residual_alignment must be a ResidualAlignmentConfig")
        self.residual_alignment._validate()


@dataclass(frozen=True)
class RGBMotionScaleEstimate:
    """The RGB-only convention used to convert real millimetres to strip pixels."""

    pixels_per_mm: float
    valid_pair_count: int
    candidate_pair_count: int
    relative_mad: float
    samples: tuple[dict[str, float | int | bool], ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "method": "adjacent_rgb_motion_divided_by_real_se3_camera_displacement",
            "pixels_per_mm": self.pixels_per_mm,
            "valid_pair_count": self.valid_pair_count,
            "candidate_pair_count": self.candidate_pair_count,
            "relative_mad": self.relative_mad,
            "samples": [dict(value) for value in self.samples],
        }


@dataclass(frozen=True)
class PushbroomLayout:
    """All real-pose strip positions in one calibrated, pose-levelled canvas."""

    frame_ids: tuple[int, ...]
    source_scan_positions_mm: tuple[float, ...]
    source_centres_x: tuple[float, ...]
    owner_left_x: tuple[float, ...]
    owner_right_x: tuple[float, ...]
    support_left_x: tuple[int, ...]
    support_right_x: tuple[int, ...]
    owner_boundaries_x: tuple[float, ...]
    endpoint_outer_owner_intervals_x: tuple[tuple[float, float], ...]
    canvas_width: int
    canvas_height: int
    canvas_megapixels: float
    aggregate_megapixels: float
    maximum_source_strip_width: int
    pixels_per_mm: float
    temporal_scan_axis: tuple[float, float, float]
    level_camera_to_world_rotation: np.ndarray
    temporal_to_virtual_x_sign: float

    def as_dict(self) -> dict[str, object]:
        return {
            "coordinate_system": {
                "canvas_x": "real camera-centre scan position scaled by local RGB motion",
                "canvas_y": "pose-levelled calibrated virtual camera row",
                "translation_unit": "mm",
                "pixel_scale": "RGB pixels per real camera-centre millimetre",
            },
            "frame_ids": list(self.frame_ids),
            "source_scan_positions_mm": list(self.source_scan_positions_mm),
            "source_centres_x": list(self.source_centres_x),
            "owner_intervals_x": [
                [left, right]
                for left, right in zip(self.owner_left_x, self.owner_right_x, strict=True)
            ],
            "source_support_intervals_x": [
                [left, right]
                for left, right in zip(
                    self.support_left_x, self.support_right_x, strict=True
                )
            ],
            "owner_boundaries_x": list(self.owner_boundaries_x),
            "endpoint_policy": (
                "outward_half_fov"
                if self.endpoint_outer_owner_intervals_x
                else "central_strip_only"
            ),
            "endpoint_outer_owner_intervals_x": [
                [left, right] for left, right in self.endpoint_outer_owner_intervals_x
            ],
            "width": self.canvas_width,
            "height": self.canvas_height,
            "canvas_megapixels": self.canvas_megapixels,
            "aggregate_megapixels": self.aggregate_megapixels,
            "maximum_source_strip_width": self.maximum_source_strip_width,
            "pixels_per_mm": self.pixels_per_mm,
            "temporal_scan_axis": list(self.temporal_scan_axis),
            "level_camera_to_world_rotation": self.level_camera_to_world_rotation.tolist(),
            "temporal_to_virtual_x_sign": self.temporal_to_virtual_x_sign,
        }


@dataclass(frozen=True)
class PushbroomContribution:
    """One narrow RGB strip kept independently of all other source images."""

    source_index: int
    frame_id: int
    x0: int
    rgb: np.ndarray
    valid_mask: np.ndarray

    @property
    def x1(self) -> int:
        return self.x0 + int(self.rgb.shape[1])


@dataclass(frozen=True)
class PreviewContribution:
    """Low-resolution RGB-only analysis strip kept outside the output path.

    ``x0`` is expressed in the common preview-canvas grid.  The two inverse
    maps are raw calibrated source-pixel coordinates, retained only in the
    temporary preview spool so RGB residual evidence can be related back to
    the real camera image.  Preview samples are never copied into the formal
    panorama.
    """

    source_index: int
    frame_id: int
    x0: int
    canvas_scale: float
    rgb: np.ndarray
    valid_mask: np.ndarray
    source_map_x: np.ndarray
    source_map_y: np.ndarray

    @property
    def x1(self) -> int:
        return self.x0 + int(self.rgb.shape[1])


@dataclass(frozen=True)
class CalibratedRGBPushbroomResult:
    panorama: np.ndarray
    metadata: dict[str, object]


def _unit(vector: np.ndarray, label: str) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64).reshape(-1)
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError(f"{label} must be a finite three-vector")
    norm = float(np.linalg.norm(value))
    if norm <= 1e-9:
        raise RuntimeError(f"{label} is degenerate")
    return value / norm


def validate_camera_to_world(pose: np.ndarray) -> np.ndarray:
    """Validate one finite rigid camera-to-world transform without depth data."""

    value = np.asarray(pose, dtype=np.float64)
    if value.shape != (4, 4) or not np.isfinite(value).all():
        raise ValueError("camera_to_world must be a finite 4x4 matrix")
    if not np.allclose(value[3], (0.0, 0.0, 0.0, 1.0), atol=1e-8):
        raise ValueError("camera_to_world must have homogeneous final row [0, 0, 0, 1]")
    rotation = value[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        raise ValueError("camera_to_world rotation must be orthonormal")
    if not np.isclose(float(np.linalg.det(rotation)), 1.0, atol=1e-5):
        raise ValueError("camera_to_world rotation must have determinant +1")
    return value.copy()


def _trajectory_axes(poses: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray, float]:
    """Derive a level virtual-camera orientation from real poses only."""

    checked = [validate_camera_to_world(pose) for pose in poses]
    if len(checked) < 2:
        raise ValueError("Calibrated RGB pushbroom requires at least two poses")
    centres = np.asarray([pose[:3, 3] for pose in checked], dtype=np.float64)
    temporal = _unit(centres[-1] - centres[0], "camera-centre scan span")
    signed_steps = np.diff(centres, axis=0) @ temporal
    if not np.all(np.isfinite(signed_steps)) or np.any(signed_steps <= 1e-4):
        raise RuntimeError(
            "Calibrated RGB pushbroom requires strictly monotonic real camera centres"
        )

    down_candidates = np.asarray([pose[:3, 1] for pose in checked], dtype=np.float64)
    down = np.median(down_candidates, axis=0)
    down -= temporal * float(np.dot(down, temporal))
    down = _unit(down, "levelled camera-down direction")
    virtual_right = temporal.copy()
    virtual_forward = _unit(
        np.cross(virtual_right, down), "levelled camera-forward direction"
    )
    mean_forward = _unit(
        np.median(np.asarray([pose[:3, 2] for pose in checked]), axis=0),
        "mean camera-forward direction",
    )
    if float(np.dot(virtual_forward, mean_forward)) < 0.0:
        virtual_right *= -1.0
        virtual_forward = _unit(
            np.cross(virtual_right, down), "levelled camera-forward direction"
        )
    level_rotation = np.column_stack((virtual_right, down, virtual_forward))
    if not np.isclose(float(np.linalg.det(level_rotation)), 1.0, atol=1e-6):
        raise RuntimeError("Levelled RGB virtual camera is not right handed")
    return temporal, level_rotation, float(np.dot(temporal, virtual_right))


def _motion_row_value(value: object, name: str) -> object | None:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _read_bgr(path: Path, calibration: CameraIntrinsics) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not decode RGB source image: {path}")
    if image.shape[:2] != (calibration.height, calibration.width):
        raise ValueError(
            f"RGB source {path} has shape {image.shape[:2]}, expected "
            f"{(calibration.height, calibration.width)}"
        )
    return image


def _phase_motion_pixels(
    first: np.ndarray, second: np.ndarray, central_fraction: float
) -> tuple[float, float]:
    """Fallback local RGB shift measurement; it never produces a pose."""

    height, width = first.shape[:2]
    fraction = float(np.clip(central_fraction, 0.05, 1.0))
    half = max(8, min(width // 2, int(round(width * fraction * 0.5))))
    centre = width // 2
    x0, x1 = max(0, centre - half), min(width, centre + half)
    y0, y1 = max(0, int(round(height * 0.08))), min(height, int(round(height * 0.92)))
    first_gray = cv2.cvtColor(first[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(
        np.float32
    )
    second_gray = cv2.cvtColor(second[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(
        np.float32
    )
    if first_gray.shape[0] < 8 or first_gray.shape[1] < 8:
        return 0.0, 0.0
    window = cv2.createHanningWindow(
        (first_gray.shape[1], first_gray.shape[0]), cv2.CV_32F
    )
    try:
        shift, response = cv2.phaseCorrelate(first_gray, second_gray, window)
    except cv2.error:
        return 0.0, 0.0
    return abs(float(shift[0])), float(response)


def estimate_rgb_motion_pixels_per_mm(
    frames: Sequence[RGBDFrame],
    poses: Sequence[np.ndarray],
    calibration: CameraIntrinsics,
    config: CalibratedRGBPushbroomConfig,
    *,
    rgb_motions: Sequence[object] | None = None,
    motion_pixels_to_full_resolution: float = 1.0,
) -> RGBMotionScaleEstimate:
    """Estimate one layout-only RGB pixel scale from adjacent real SE(3) motion.

    ``rgb_motions`` may contain the pipeline's existing local thumbnail motion
    estimates.  They are used only as independent RGB measurements for a scalar
    scale; they never move, order, or create camera poses.  When unavailable,
    local phase correlation is used as a fail-closed fallback.
    """

    if len(frames) != len(poses) or len(frames) < 2:
        raise ValueError("RGB scale frames and poses must align and contain two frames")
    if not np.isfinite(float(motion_pixels_to_full_resolution)) or (
        motion_pixels_to_full_resolution <= 0.0
    ):
        raise ValueError("motion_pixels_to_full_resolution must be finite and positive")
    temporal, _, _ = _trajectory_axes(poses)
    centres = np.asarray([validate_camera_to_world(pose)[:3, 3] for pose in poses])
    displacements = np.diff(centres, axis=0) @ temporal
    samples: list[dict[str, float | int | bool]] = []
    candidates: list[float] = []
    for index, displacement in enumerate(displacements):
        pixels = 0.0
        response = 0.0
        used_thumbnail_motion = False
        reliable = True
        if rgb_motions is not None and index < len(rgb_motions):
            row = rgb_motions[index]
            raw_dx = _motion_row_value(row, "dx")
            raw_reliable = _motion_row_value(row, "reliable")
            try:
                pixels = abs(float(raw_dx)) * float(motion_pixels_to_full_resolution)
                response = 1.0
                reliable = bool(raw_reliable) if raw_reliable is not None else True
                used_thumbnail_motion = True
            except (TypeError, ValueError):
                pixels = 0.0
        if not used_thumbnail_motion:
            first = _read_bgr(frames[index].color_path, calibration)
            second = _read_bgr(frames[index + 1].color_path, calibration)
            pixels, response = _phase_motion_pixels(
                first, second, config.scale_central_fraction
            )
        accepted = bool(
            reliable
            and displacement > 1e-4
            and pixels > 0.25
            and response >= config.scale_minimum_response
            and np.isfinite(pixels)
        )
        pixels_per_mm = pixels / float(displacement) if accepted else 0.0
        samples.append(
            {
                "pair_index": index,
                "camera_displacement_mm": float(displacement),
                "rgb_motion_pixels": float(pixels),
                "pixels_per_mm": float(pixels_per_mm),
                "response": float(response),
                "used_thumbnail_motion": used_thumbnail_motion,
                "accepted": accepted,
            }
        )
        if accepted:
            candidates.append(float(pixels_per_mm))
    if len(candidates) < int(config.minimum_valid_scale_pairs):
        raise RuntimeError(
            "Calibrated RGB pushbroom has too few reliable adjacent RGB motion "
            "measurements for a real-SE(3) strip scale"
        )
    values = np.asarray(candidates, dtype=np.float64)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    relative_mad = mad / max(median, 1e-9)
    if (
        not np.isfinite(median)
        or median <= 1e-6
        or relative_mad > config.scale_max_relative_mad
    ):
        raise RuntimeError(
            "Calibrated RGB pushbroom RGB-motion scale is unstable; no depth or "
            "2-D fallback is permitted"
        )
    return RGBMotionScaleEstimate(
        pixels_per_mm=median,
        valid_pair_count=len(candidates),
        candidate_pair_count=len(samples),
        relative_mad=relative_mad,
        samples=tuple(samples),
    )


def build_calibrated_rgb_pushbroom_layout(
    frame_ids: Sequence[int],
    poses: Sequence[np.ndarray],
    calibration: CameraIntrinsics,
    scale: RGBMotionScaleEstimate,
    config: CalibratedRGBPushbroomConfig,
) -> PushbroomLayout:
    """Allocate contiguous real-pose owner intervals without a depth plane."""

    if len(frame_ids) != len(poses) or not 2 <= len(poses) <= config.max_pose_count:
        raise ValueError("Pushbroom layout requires 2-160 aligned real pose nodes")
    temporal, level_rotation, virtual_sign = _trajectory_axes(poses)
    centres_mm = np.asarray(
        [validate_camera_to_world(pose)[:3, 3] for pose in poses], dtype=np.float64
    )
    positions = centres_mm @ temporal
    if np.any(np.diff(positions) <= 1e-4):
        raise RuntimeError("Pushbroom camera centres are not strictly monotonic")
    centres_x_unshifted = (positions - positions[0]) * scale.pixels_per_mm
    steps = np.diff(centres_x_unshifted)
    if not np.isfinite(steps).all() or np.any(steps <= 0.25):
        raise RuntimeError("Pushbroom RGB scale produces collapsed adjacent strips")
    owner_edges = np.concatenate(
        (
            [centres_x_unshifted[0] - 0.5 * steps[0]],
            0.5 * (centres_x_unshifted[:-1] + centres_x_unshifted[1:]),
            [centres_x_unshifted[-1] + 0.5 * steps[-1]],
        )
    )
    if config.endpoint_outer_half_fov:
        # The first and last source own the scene that lies outward from the
        # scan.  Extend only those exterior intervals to the calibrated image
        # edge; every intermediate source remains a central strip.  The sign
        # accounts for a levelled virtual camera whose image-right direction
        # is opposite the chronological scan axis.
        if virtual_sign >= 0.0:
            first_outward_extent = float(calibration.cx)
            first_inward_extent = float(calibration.width - 1 - calibration.cx)
            last_inward_extent = float(calibration.cx)
            last_outward_extent = float(calibration.width - 1 - calibration.cx)
        else:
            first_outward_extent = float(calibration.width - 1 - calibration.cx)
            first_inward_extent = float(calibration.cx)
            last_inward_extent = float(calibration.width - 1 - calibration.cx)
            last_outward_extent = float(calibration.cx)
        owner_edges[0] = centres_x_unshifted[0] - first_outward_extent
        owner_edges[-1] = centres_x_unshifted[-1] + last_outward_extent
        if owner_edges[1] > centres_x_unshifted[0] + first_inward_extent:
            raise RuntimeError(
                "First endpoint cannot cover its midpoint owner interval within "
                "the calibrated RGB field of view"
            )
        if owner_edges[-2] < centres_x_unshifted[-1] - last_inward_extent:
            raise RuntimeError(
                "Last endpoint cannot cover its midpoint owner interval within "
                "the calibrated RGB field of view"
            )
    if np.any(np.diff(owner_edges) <= 1e-4):
        raise RuntimeError("Pushbroom endpoint and midpoint owner intervals collapse")
    origin = math.floor(float(owner_edges[0]))
    owner_edges -= origin
    centres_x = centres_x_unshifted - origin
    width = int(math.ceil(float(owner_edges[-1])))
    if width < 2:
        raise RuntimeError("Pushbroom RGB canvas is degenerate")
    megapixels = width * calibration.height / 1_000_000.0
    if megapixels > config.max_canvas_megapixels:
        raise MemoryError(
            f"Calibrated RGB pushbroom canvas is {megapixels:.1f} MP, above "
            f"the {config.max_canvas_megapixels:.1f} MP hard limit"
        )
    # Every intermediate raw source contributes only a genuinely narrow
    # central band.  The two scan endpoints are allowed to own their outward
    # calibrated half-FOV so the captured scene does not silently disappear at
    # either edge of the panorama.  If an intermediate hard-owned interval
    # cannot fit, fail closed and ask for denser real pose sampling or a lower
    # output scale.
    maximum_band_width = max(
        2, int(math.floor(calibration.width * config.maximum_central_band_fraction))
    )
    owner_left_int = np.ceil(owner_edges[:-1]).astype(np.int32)
    owner_right_int = np.ceil(owner_edges[1:]).astype(np.int32)
    owner_widths = owner_right_int - owner_left_int
    limited_owner_widths = (
        owner_widths[1:-1] if config.endpoint_outer_half_fov else owner_widths
    )
    if np.any(limited_owner_widths > maximum_band_width):
        raise RuntimeError(
            "An intermediate calibrated RGB pushbroom owner strip exceeds the "
            "configured narrow central-band limit; capture more real pose frames "
            "or lower output scale"
        )
    spare_width = np.maximum(0, maximum_band_width - owner_widths)
    # Keep the complete permitted central support on disk for photometric
    # statistics.  The *owner search* is selected separately below and remains
    # exactly 32--64 px wide, so this does not broaden a hard-owned strip or a
    # 2--8 px blend band.  Using the full 20% central strip for the global
    # colour solve avoids deriving a frame gain from a handful of pixels when a
    # foreground happens to cross the narrow seam corridor.
    left_padding = spare_width // 2
    right_padding = spare_width - left_padding
    # Endpoint exterior pixels are a direct calibrated copy, never a seam
    # support region.  Do not ask the inverse remap to sample beyond a physical
    # image edge merely to add seam padding.
    left_padding[0] = 0
    right_padding[-1] = 0
    support_left = owner_left_int - left_padding
    support_right = owner_right_int + right_padding
    support_left = np.clip(support_left, 0, width)
    support_right = np.clip(support_right, 0, width)
    if np.any(support_right - support_left < 2):
        raise RuntimeError("Pushbroom central strip support collapsed")
    canvas_megapixels = width * calibration.height / 1_000_000.0
    maximum_source_strip_width = int(np.max(support_right - support_left))
    aggregate_megapixels = (
        canvas_megapixels
        + 2.0 * calibration.height * maximum_source_strip_width / 1_000_000.0
    )
    if aggregate_megapixels > config.max_aggregate_megapixels:
        raise MemoryError(
            "Calibrated RGB pushbroom aggregate working set is "
            f"{aggregate_megapixels:.1f} MP, above the "
            f"{config.max_aggregate_megapixels:.1f} MP hard limit"
        )
    return PushbroomLayout(
        frame_ids=tuple(int(value) for value in frame_ids),
        source_scan_positions_mm=tuple(float(value) for value in positions),
        source_centres_x=tuple(float(value) for value in centres_x),
        owner_left_x=tuple(float(value) for value in owner_edges[:-1]),
        owner_right_x=tuple(float(value) for value in owner_edges[1:]),
        support_left_x=tuple(int(value) for value in support_left),
        support_right_x=tuple(int(value) for value in support_right),
        owner_boundaries_x=tuple(float(value) for value in owner_edges[1:-1]),
        endpoint_outer_owner_intervals_x=(
            (
                float(owner_edges[0]),
                float(centres_x[0]),
            ),
            (
                float(centres_x[-1]),
                float(owner_edges[-1]),
            ),
        )
        if config.endpoint_outer_half_fov
        else (),
        canvas_width=width,
        canvas_height=int(calibration.height),
        canvas_megapixels=float(canvas_megapixels),
        aggregate_megapixels=float(aggregate_megapixels),
        maximum_source_strip_width=maximum_source_strip_width,
        pixels_per_mm=float(scale.pixels_per_mm),
        temporal_scan_axis=tuple(float(value) for value in temporal),
        level_camera_to_world_rotation=level_rotation,
        temporal_to_virtual_x_sign=float(virtual_sign),
    )


def _as_virtual_coordinate_grids(
    canvas_x: np.ndarray, canvas_y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Normalise one-dimensional or matching two-dimensional virtual grids."""

    x = np.asarray(canvas_x, dtype=np.float64)
    y = np.asarray(canvas_y, dtype=np.float64)
    if x.ndim == 1 and y.ndim == 1:
        return (
            np.broadcast_to(x[None, :], (y.size, x.size)),
            np.broadcast_to(y[:, None], (y.size, x.size)),
        )
    if x.shape != y.shape or x.ndim != 2:
        raise ValueError("Virtual inverse-map coordinates must be matching 2-D grids")
    return x, y


def _build_nominal_inverse_map(
    layout: PushbroomLayout,
    calibration: CameraIntrinsics,
    camera_to_world: np.ndarray,
    source_index: int,
    canvas_x: np.ndarray,
    canvas_y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map a nominal levelled virtual strip into raw calibrated RGB pixels.

    This is intentionally the pre-existing calibrated mapping written as an
    independently testable constructor.  With one-dimensional coordinates its
    arithmetic and sampling grid are bit-for-bit the former ``render_frame``
    path, which lets the identity residual model prove it has not changed any
    output pixel.
    """

    if not 0 <= source_index < len(layout.frame_ids):
        raise IndexError("Pushbroom source index is out of range")
    pose = validate_camera_to_world(camera_to_world)
    x, y = _as_virtual_coordinate_grids(canvas_x, canvas_y)
    virtual_u = calibration.cx + layout.temporal_to_virtual_x_sign * (
        x - layout.source_centres_x[source_index]
    )
    ray_x = (virtual_u - calibration.cx) / calibration.fx
    ray_y = (y - calibration.cy) / calibration.fy
    rays = np.empty((*x.shape, 3), dtype=np.float64)
    rays[:, :, 0] = ray_x
    rays[:, :, 1] = ray_y
    rays[:, :, 2] = 1.0
    world_rays = rays @ layout.level_camera_to_world_rotation.T
    source_rays = world_rays @ pose[:3, :3]
    map_x, map_y, positive_z = camera_points_to_source_pixels(source_rays, calibration)
    valid = (
        positive_z
        & np.isfinite(map_x)
        & np.isfinite(map_y)
        & (map_x >= 0.0)
        & (map_x <= calibration.width - 1)
        & (map_y >= 0.0)
        & (map_y <= calibration.height - 1)
    )
    return (
        np.ascontiguousarray(map_x),
        np.ascontiguousarray(map_y),
        np.ascontiguousarray(valid),
    )


def _build_composite_inverse_map(
    layout: PushbroomLayout,
    calibration: CameraIntrinsics,
    camera_to_world: np.ndarray,
    source_index: int,
    canvas_x: np.ndarray,
    canvas_y: np.ndarray,
    *,
    residual_warp: object | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compose an audited residual inverse warp before calibrated ray mapping.

    A residual is not a camera pose and never alters the real SE(3) supplied
    by the pose pipeline.  ``None`` (and an explicitly identity residual)
    takes the nominal branch directly, preserving the Phase-0 identity
    baseline without any changed floating-point operations.
    """

    if residual_warp is None or bool(getattr(residual_warp, "is_identity", False)):
        return _build_nominal_inverse_map(
            layout,
            calibration,
            camera_to_world,
            source_index,
            canvas_x,
            canvas_y,
        )
    inverse = getattr(residual_warp, "inverse_virtual_coordinates", None)
    if not callable(inverse):
        raise TypeError("Residual warp must provide inverse_virtual_coordinates")
    x, y = _as_virtual_coordinate_grids(canvas_x, canvas_y)
    try:
        warped_x, warped_y = inverse(x, y)
    except Exception as exc:
        raise RuntimeError("Residual inverse warp could not map virtual coordinates") from exc
    warped_x = np.asarray(warped_x, dtype=np.float64)
    warped_y = np.asarray(warped_y, dtype=np.float64)
    if (
        warped_x.shape != x.shape
        or warped_y.shape != y.shape
        or not np.isfinite(warped_x).all()
        or not np.isfinite(warped_y).all()
    ):
        raise RuntimeError("Residual inverse warp returned invalid virtual coordinates")
    return _build_nominal_inverse_map(
        layout,
        calibration,
        camera_to_world,
        source_index,
        warped_x,
        warped_y,
    )


class CalibratedRGBPushbroomRenderer:
    """Single-remap calibrated RGB strip generator; it has no depth inputs."""

    def __init__(
        self,
        layout: PushbroomLayout,
        calibration: CameraIntrinsics,
        poses: Sequence[np.ndarray],
        residual_warps: Sequence[object | None] | None = None,
    ) -> None:
        if len(poses) != len(layout.frame_ids):
            raise ValueError("Pushbroom renderer poses must match layout sources")
        if residual_warps is not None and len(residual_warps) != len(poses):
            raise ValueError("Pushbroom residual warps must match layout sources")
        self.layout = layout
        self.calibration = calibration
        self.poses = tuple(validate_camera_to_world(pose) for pose in poses)
        self.residual_warps = (
            tuple(residual_warps) if residual_warps is not None else (None,) * len(poses)
        )
        # ``remap_count`` is kept as the legacy full-resolution-output counter.
        # Preview analysis is deliberately reported separately and cannot
        # satisfy the formal one-source/one-output-remap invariant.
        self.remap_count = 0
        self.full_resolution_output_remap_count = 0
        self.analysis_preview_remap_count = 0

    def _inverse_map(
        self,
        source_index: int,
        global_x: np.ndarray,
        virtual_v: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build the sole calibrated inverse map for a virtual strip grid."""

        warp = self.residual_warps[source_index]
        return _build_composite_inverse_map(
            self.layout,
            self.calibration,
            self.poses[source_index],
            source_index,
            global_x,
            virtual_v,
            residual_warp=warp,
        )

    def render_frame(self, frame: RGBDFrame, source_index: int) -> PushbroomContribution:
        if not 0 <= source_index < len(self.poses):
            raise IndexError("Pushbroom source index is out of range")
        if int(frame.frame_id) != self.layout.frame_ids[source_index]:
            raise ValueError("Pushbroom frame order must match real-pose layout order")
        image = _read_bgr(frame.color_path, self.calibration)
        x0 = self.layout.support_left_x[source_index]
        x1 = self.layout.support_right_x[source_index]
        height = self.layout.canvas_height
        global_x = np.arange(x0, x1, dtype=np.float64)
        virtual_v = np.arange(height, dtype=np.float64)
        map_x, map_y, valid = self._inverse_map(source_index, global_x, virtual_v)
        rgb = cv2.remap(
            image,
            map_x,
            map_y,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        self.remap_count += 1
        self.full_resolution_output_remap_count += 1
        return PushbroomContribution(
            source_index=source_index,
            frame_id=int(frame.frame_id),
            x0=x0,
            rgb=np.ascontiguousarray(rgb),
            valid_mask=np.ascontiguousarray(valid),
        )

    def render_preview_frame(
        self,
        frame: RGBDFrame,
        source_index: int,
        *,
        canvas_scale: float,
    ) -> PreviewContribution:
        """Inverse-remap one analysis-only preview in a shared canvas grid.

        The preview grid is derived from full virtual coordinates, rather than
        resizing a completed output strip.  Consequently its raw-coordinate
        inverse maps remain valid evidence for calibrated RGB/SE(3) checks.
        """

        if not 0 <= source_index < len(self.poses):
            raise IndexError("Pushbroom preview source index is out of range")
        if int(frame.frame_id) != self.layout.frame_ids[source_index]:
            raise ValueError("Pushbroom preview order must match real-pose layout order")
        scale = float(canvas_scale)
        if not np.isfinite(scale) or not 0.0 < scale <= 1.0:
            raise ValueError("Pushbroom preview canvas scale must be in (0, 1]")
        source_x0 = self.layout.support_left_x[source_index]
        source_x1 = self.layout.support_right_x[source_index]
        x0 = int(math.floor(source_x0 * scale))
        x1 = int(math.ceil(source_x1 * scale))
        height = max(1, int(math.ceil(self.layout.canvas_height * scale)))
        preview_x = np.arange(x0, x1, dtype=np.float64)
        global_x = (preview_x + 0.5) / scale - 0.5
        preview_y = np.arange(height, dtype=np.float64)
        virtual_v = (preview_y + 0.5) / scale - 0.5
        map_x, map_y, valid = self._inverse_map(source_index, global_x, virtual_v)
        image = _read_bgr(frame.color_path, self.calibration)
        rgb = cv2.remap(
            image,
            map_x,
            map_y,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        self.analysis_preview_remap_count += 1
        return PreviewContribution(
            source_index=source_index,
            frame_id=int(frame.frame_id),
            x0=x0,
            canvas_scale=scale,
            rgb=np.ascontiguousarray(rgb),
            valid_mask=np.ascontiguousarray(valid),
            source_map_x=np.ascontiguousarray(map_x),
            source_map_y=np.ascontiguousarray(map_y),
        )


def _save_contribution(path: Path, contribution: PushbroomContribution) -> None:
    np.savez_compressed(
        path,
        source_index=np.asarray(contribution.source_index, dtype=np.int32),
        frame_id=np.asarray(contribution.frame_id, dtype=np.int32),
        x0=np.asarray(contribution.x0, dtype=np.int32),
        rgb=contribution.rgb,
        valid=contribution.valid_mask.astype(np.uint8),
    )


def _load_contribution(path: Path) -> PushbroomContribution:
    with np.load(path, allow_pickle=False) as stored:
        return PushbroomContribution(
            source_index=int(stored["source_index"]),
            frame_id=int(stored["frame_id"]),
            x0=int(stored["x0"]),
            rgb=np.ascontiguousarray(stored["rgb"]),
            valid_mask=np.ascontiguousarray(stored["valid"].astype(bool)),
        )


def _save_preview(path: Path, contribution: PreviewContribution) -> None:
    """Spool analysis-only preview evidence; never publish it with a delivery."""

    np.savez_compressed(
        path,
        source_index=np.asarray(contribution.source_index, dtype=np.int32),
        frame_id=np.asarray(contribution.frame_id, dtype=np.int32),
        x0=np.asarray(contribution.x0, dtype=np.int32),
        canvas_scale=np.asarray(contribution.canvas_scale, dtype=np.float64),
        rgb=contribution.rgb,
        valid=contribution.valid_mask.astype(np.uint8),
        source_map_x=contribution.source_map_x,
        source_map_y=contribution.source_map_y,
    )


def _load_preview(path: Path) -> PreviewContribution:
    with np.load(path, allow_pickle=False) as stored:
        return PreviewContribution(
            source_index=int(stored["source_index"]),
            frame_id=int(stored["frame_id"]),
            x0=int(stored["x0"]),
            canvas_scale=float(stored["canvas_scale"]),
            rgb=np.ascontiguousarray(stored["rgb"]),
            valid_mask=np.ascontiguousarray(stored["valid"].astype(bool)),
            source_map_x=np.ascontiguousarray(stored["source_map_x"]),
            source_map_y=np.ascontiguousarray(stored["source_map_y"]),
        )


def _gradient_magnitude(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.magnitude(
        cv2.Scharr(gray, cv2.CV_32F, 1, 0),
        cv2.Scharr(gray, cv2.CV_32F, 0, 1),
    )


def _srgb_to_linear_bgr(image: np.ndarray) -> np.ndarray:
    """Decode a uint8 BGR image into the linear-light domain exactly once."""

    encoded = np.asarray(image, dtype=np.float32) / 255.0
    return np.where(
        encoded <= 0.04045,
        encoded / 12.92,
        np.power((encoded + 0.055) / 1.055, 2.4),
    ).astype(np.float32)


def _linear_to_srgb_bgr(linear: np.ndarray) -> np.ndarray:
    """Encode finite linear BGR samples back to uint8 sRGB."""

    values = np.clip(np.asarray(linear, dtype=np.float32), 0.0, 1.0)
    encoded = np.where(
        values <= 0.0031308,
        values * 12.92,
        1.055 * np.power(values, 1.0 / 2.4) - 0.055,
    )
    return np.rint(np.clip(encoded * 255.0, 0.0, 255.0)).astype(np.uint8)


def _apply_linear_bgr_gain(image: np.ndarray, gain_bgr: np.ndarray) -> np.ndarray:
    """Apply one finite three-channel gain in linear RGB, never gamma RGB."""

    gain = np.asarray(gain_bgr, dtype=np.float32).reshape(1, 1, 3)
    if not np.isfinite(gain).all() or np.any(gain < 0.45) or np.any(gain > 2.20):
        raise RuntimeError("Linear RGB photometric gain is outside the formal range")
    return _linear_to_srgb_bgr(_srgb_to_linear_bgr(image) * gain)


@dataclass(frozen=True)
class _RGBRiskDetails:
    mask: np.ndarray
    edge_offset_p95: float
    seed_pixel_count: int
    component_count: int


@dataclass(frozen=True)
class _PhotometricEdge:
    """One reliable adjacent safe-wall observation in linear BGR log space."""

    log_relation_bgr: np.ndarray
    support_pixels: int
    mad_bgr: np.ndarray
    raw_signed_l_delta: float
    safe_mask: np.ndarray | None = None


def _fill_risk_components(seed: np.ndarray, common: np.ndarray, bridge_radius: int) -> np.ndarray:
    """Close exposed risk edges and fill bounded foreground components."""

    bridge = max(3, int(bridge_radius) * 2 + 1)
    closed = cv2.morphologyEx(
        (np.asarray(seed, dtype=bool) & common).astype(np.uint8),
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (bridge, bridge)),
    ) > 0
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        closed.astype(np.uint8), connectivity=8
    )
    filled = closed.copy()
    for label in range(1, count):
        if int(stats[label, cv2.CC_STAT_AREA]) < 2:
            continue
        x0 = int(stats[label, cv2.CC_STAT_LEFT])
        y0 = int(stats[label, cv2.CC_STAT_TOP])
        x1 = x0 + int(stats[label, cv2.CC_STAT_WIDTH])
        y1 = y0 + int(stats[label, cv2.CC_STAT_HEIGHT])
        component = (labels[y0:y1, x0:x1] == label).astype(np.uint8)
        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        interior = np.zeros_like(component)
        cv2.drawContours(interior, contours, -1, 1, thickness=cv2.FILLED)
        filled[y0:y1, x0:x1] |= interior > 0
    return filled & common


def _rgb_risk_details(
    image0: np.ndarray,
    image1: np.ndarray,
    valid0: np.ndarray,
    valid1: np.ndarray,
    *,
    supplied_risk_mask: np.ndarray | None = None,
) -> _RGBRiskDetails:
    """Build RGB-only foreground risk from colour, edge distance and structure.

    A high Lab residual alone is deliberately sufficient.  The other two
    independent seeds detect a displaced dark object's exposed outline even
    when its uniform interior carries almost no gradient or Lab residual.
    """

    common = np.asarray(valid0, dtype=bool) & np.asarray(valid1, dtype=bool)
    if not np.any(common):
        return _RGBRiskDetails(
            mask=np.zeros(common.shape, dtype=np.uint8),
            edge_offset_p95=0.0,
            seed_pixel_count=0,
            component_count=0,
        )
    lab0 = cv2.cvtColor(image0, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab1 = cv2.cvtColor(image1, cv2.COLOR_BGR2LAB).astype(np.float32)
    difference = np.linalg.norm(lab0 - lab1, axis=2)
    values = difference[common]
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    lab_threshold = max(18.0, median + 3.5 * max(mad, 1.0))
    lab_seed = common & (difference >= lab_threshold)

    gray0 = cv2.cvtColor(image0, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gx0, gy0 = (
        cv2.Scharr(gray0, cv2.CV_32F, 1, 0),
        cv2.Scharr(gray0, cv2.CV_32F, 0, 1),
    )
    gx1, gy1 = (
        cv2.Scharr(gray1, cv2.CV_32F, 1, 0),
        cv2.Scharr(gray1, cv2.CV_32F, 0, 1),
    )
    magnitude0 = cv2.magnitude(gx0, gy0)
    magnitude1 = cv2.magnitude(gx1, gy1)
    magnitudes = np.concatenate((magnitude0[common], magnitude1[common]))
    # A displaced outline is caught by the symmetric edge-distance seed below.
    # The structure term is intentionally reserved for genuinely strong edges;
    # treating every modest Scharr fluctuation as foreground would turn normal
    # wall texture/illumination noise into a full-width risk component.
    gradient_threshold = max(80.0, float(np.percentile(magnitudes, 90.0)))
    denominator = np.maximum(magnitude0 * magnitude1, 1e-6)
    cosine = (gx0 * gx1 + gy0 * gy1) / denominator
    strong = (magnitude0 >= gradient_threshold) & (magnitude1 >= gradient_threshold)
    gradient_seed = common & strong & (
        (np.abs(magnitude0 - magnitude1) >= np.maximum(64.0, 0.65 * np.maximum(magnitude0, magnitude1)))
        | (cosine < 0.25)
    )

    edge0 = cv2.Canny(
        cv2.GaussianBlur(gray0, (3, 3), 0), 30, 90, L2gradient=True
    ) > 0
    edge1 = cv2.Canny(
        cv2.GaussianBlur(gray1, (3, 3), 0), 30, 90, L2gradient=True
    ) > 0
    distance0 = cv2.distanceTransform((~edge0).astype(np.uint8), cv2.DIST_L2, 3)
    distance1 = cv2.distanceTransform((~edge1).astype(np.uint8), cv2.DIST_L2, 3)
    edge_offset = (edge0 & common & (distance1 > 1.5)) | (
        edge1 & common & (distance0 > 1.5)
    )
    offsets = np.concatenate((distance1[edge0 & common], distance0[edge1 & common]))
    finite_offsets = offsets[
        np.isfinite(offsets)
        & (offsets > 0.0)
        # OpenCV uses FLT_MAX when an image contains no opposing edge at all;
        # that is not a measurable local edge displacement.
        & (offsets <= float(max(image0.shape[:2])))
    ]
    # P95 is a local registration-displacement statistic.  An edge with no
    # counterpart anywhere in the corridor remains a risk seed, but must not
    # turn a single unrelated missing edge into a 400 px global guard radius.
    local_offsets = finite_offsets[finite_offsets <= 16.0]
    edge_offset_p95 = (
        float(np.percentile(local_offsets, 95.0)) if local_offsets.size else 0.0
    )

    supplied = np.zeros(common.shape, dtype=bool)
    if supplied_risk_mask is not None:
        supplied_array = np.asarray(supplied_risk_mask)
        if supplied_array.shape != common.shape:
            raise ValueError("RGB disparity risk mask must match the strip overlap")
        if supplied_array.dtype not in {np.dtype(np.uint8), np.dtype(np.bool_)}:
            raise ValueError("RGB disparity risk mask must be uint8 or bool")
        supplied = supplied_array > 0
    seed = common & (lab_seed | edge_offset | gradient_seed | supplied)
    # Keep the bridge deliberately local.  A large p95 will subsequently make
    # the guard large and the constrained seam fail rather than silently grow
    # an arbitrary wall into a foreground mask.
    bridge_radius = min(8, max(2, int(math.ceil(edge_offset_p95))))
    filled = _fill_risk_components(seed, common, bridge_radius)
    risk = cv2.dilate(
        filled.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )
    risk[~common] = 0
    component_count = int(cv2.connectedComponents((risk > 0).astype(np.uint8))[0] - 1)
    return _RGBRiskDetails(
        mask=np.ascontiguousarray(risk),
        edge_offset_p95=edge_offset_p95,
        seed_pixel_count=int(np.count_nonzero(seed)),
        component_count=component_count,
    )


def _adaptive_multiband_levels(blend_width: int, requested_levels: int) -> int:
    if blend_width <= 2:
        natural = 1
    elif blend_width <= 4:
        natural = 2
    else:
        natural = 3
    return min(natural, max(1, int(requested_levels)))


def _risk_guard(
    risk: _RGBRiskDetails,
    common: np.ndarray,
    blend_width: int,
    requested_levels: int,
) -> tuple[np.ndarray, int, int]:
    """Fill/dilate risk components by edge shift plus pyramid support."""

    levels = _adaptive_multiband_levels(blend_width, requested_levels)
    pyramid_support = 1 << max(0, levels - 1)
    radius = max(2, int(math.ceil(risk.edge_offset_p95)) + pyramid_support + 2)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1)
    )
    guard = cv2.dilate((risk.mask > 0).astype(np.uint8), kernel) > 0
    # Treat a one-pixel residual gap between adjacent protected fragments as
    # part of the same foreground component.  Otherwise two independently
    # assigned fragments can leave an unusable one-pixel "safe" slit that no
    # 2--8 px owner boundary can traverse without touching the guard.
    guard = cv2.morphologyEx(
        guard.astype(np.uint8),
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    ) > 0
    guard &= np.asarray(common, dtype=bool)
    component_count = int(cv2.connectedComponents(guard.astype(np.uint8))[0] - 1)
    return guard, radius, component_count


def _safe_white_wall_mask(
    first: np.ndarray,
    second: np.ndarray,
    common: np.ndarray,
    risk_guard: np.ndarray,
    *,
    low_gradient_quantile: float,
) -> np.ndarray:
    """Return the conservative low-texture neutral wall support for one pair."""

    usable = np.asarray(common, dtype=bool) & ~np.asarray(risk_guard, dtype=bool)
    if int(np.count_nonzero(usable)) < 16:
        return np.zeros_like(usable)
    lab0 = cv2.cvtColor(first, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab1 = cv2.cvtColor(second, cv2.COLOR_BGR2LAB).astype(np.float32)
    grad0, grad1 = _gradient_magnitude(first), _gradient_magnitude(second)
    gradient_values = np.concatenate((grad0[usable], grad1[usable]))
    if not gradient_values.size:
        return np.zeros_like(usable)
    gradient_limit = float(
        np.clip(
            np.percentile(gradient_values, 100.0 * low_gradient_quantile),
            18.0,
            150.0,
        )
    )
    hsv0 = cv2.cvtColor(first, cv2.COLOR_BGR2HSV)
    hsv1 = cv2.cvtColor(second, cv2.COLOR_BGR2HSV)
    valid_distance = cv2.distanceTransform(usable.astype(np.uint8), cv2.DIST_L2, 3)
    unclipped = (
        (np.min(first, axis=2) >= 10)
        & (np.min(second, axis=2) >= 10)
        & (np.max(first, axis=2) <= 250)
        & (np.max(second, axis=2) <= 250)
        & (lab0[:, :, 0] >= 42.0)
        & (lab1[:, :, 0] >= 42.0)
        & (hsv0[:, :, 1] <= 48)
        & (hsv1[:, :, 1] <= 48)
        & (np.maximum(np.abs(lab0[:, :, 1] - 128.0), np.abs(lab0[:, :, 2] - 128.0)) <= 20.0)
        & (np.maximum(np.abs(lab1[:, :, 1] - 128.0), np.abs(lab1[:, :, 2] - 128.0)) <= 20.0)
    )
    candidates = (
        usable
        & (valid_distance >= 3.0)
        & unclipped
        & (grad0 <= gradient_limit)
        & (grad1 <= gradient_limit)
    )
    if int(np.count_nonzero(candidates)) < 16:
        return np.zeros_like(usable)
    linear0 = _srgb_to_linear_bgr(first)
    linear1 = _srgb_to_linear_bgr(second)
    provisional = np.median(
        np.log(np.maximum(linear0[candidates], 1e-4))
        - np.log(np.maximum(linear1[candidates], 1e-4)),
        axis=0,
    )
    adjusted_second = _linear_to_srgb_bgr(
        linear1 * np.exp(np.asarray(provisional, dtype=np.float32)).reshape(1, 1, 3)
    )
    adjusted_lab = cv2.cvtColor(adjusted_second, cv2.COLOR_BGR2LAB).astype(np.float32)
    residual = np.linalg.norm(lab0 - adjusted_lab, axis=2)
    residual_values = residual[candidates]
    residual_median = float(np.median(residual_values))
    residual_mad = float(np.median(np.abs(residual_values - residual_median)))
    residual_limit = max(3.0, residual_median + 3.5 * max(residual_mad, 0.5))
    return candidates & (residual <= residual_limit)


def _robust_log_channel_relation(values: np.ndarray, minimum: int) -> tuple[float, float] | None:
    """Return a trimmed Huber log-ratio estimate and its robust MAD."""

    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size < minimum:
        return None
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    keep = np.abs(finite - median) <= max(0.004, 3.0 * 1.4826 * mad)
    trimmed = finite[keep]
    # The initial safe support must meet the formal minimum.  Huber trimming
    # may legitimately discard a minority of tiny remap/interpolation
    # outliers, so require a still-substantial half of that support rather
    # than falsely declaring an otherwise clean narrow wall unusable.
    if trimmed.size < max(32, int(math.ceil(minimum * 0.5))):
        return None
    centre = float(np.median(trimmed))
    scale = max(0.0025, 1.4826 * float(np.median(np.abs(trimmed - centre))))
    weights = np.minimum(1.0, 1.5 * scale / np.maximum(np.abs(trimmed - centre), 1e-9))
    estimate = float(np.sum(weights * trimmed) / np.sum(weights))
    if not np.isfinite(estimate):
        return None
    return estimate, mad


def _safe_wall_log_relation(
    first: np.ndarray,
    second: np.ndarray,
    common: np.ndarray,
    risk_guard: np.ndarray,
    *,
    low_gradient_quantile: float,
) -> _PhotometricEdge | None:
    """Measure all three linear-BGR gain relations from safe neutral wall only."""

    safe = _safe_white_wall_mask(
        first,
        second,
        common,
        risk_guard,
        low_gradient_quantile=low_gradient_quantile,
    )
    support = int(np.count_nonzero(safe))
    # A narrow but genuinely clean corridor can contain only a few dozen wall
    # pixels; robust trimming still leaves a useful estimate at 64 samples.
    # Anything below that is a structural no-support failure.
    minimum = 64
    if support < minimum:
        return None
    linear0 = _srgb_to_linear_bgr(first)
    linear1 = _srgb_to_linear_bgr(second)
    ratios = np.log(np.maximum(linear0, 1e-4)) - np.log(np.maximum(linear1, 1e-4))
    relation = np.empty(3, dtype=np.float64)
    mad = np.empty(3, dtype=np.float64)
    for channel in range(3):
        robust = _robust_log_channel_relation(ratios[:, :, channel][safe], minimum)
        if robust is None:
            return None
        relation[channel], mad[channel] = robust
    lab0 = cv2.cvtColor(first, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab1 = cv2.cvtColor(second, cv2.COLOR_BGR2LAB).astype(np.float32)
    signed_l_delta = float(np.median((lab0[:, :, 0] - lab1[:, :, 0])[safe]) * (100.0 / 255.0))
    return _PhotometricEdge(
        log_relation_bgr=relation,
        support_pixels=support,
        mad_bgr=mad,
        raw_signed_l_delta=signed_l_delta,
        safe_mask=np.ascontiguousarray(safe),
    )


def _pair_overlap(
    first: PushbroomContribution, second: PushbroomContribution
) -> tuple[int, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    left, right = max(first.x0, second.x0), min(first.x1, second.x1)
    if right <= left:
        raise RuntimeError("Adjacent calibrated RGB strips have no real overlap")
    first_slice = slice(left - first.x0, right - first.x0)
    second_slice = slice(left - second.x0, right - second.x0)
    return (
        left,
        right,
        first.rgb[:, first_slice],
        second.rgb[:, second_slice],
        first.valid_mask[:, first_slice],
        second.valid_mask[:, second_slice],
    )


def _preview_pair_overlap(
    first: PreviewContribution, second: PreviewContribution
) -> tuple[
    int,
    int,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Return adjacent preview overlap plus both raw-coordinate inverse maps."""

    left, right = max(first.x0, second.x0), min(first.x1, second.x1)
    if right <= left:
        raise RuntimeError("Adjacent calibrated RGB previews have no real overlap")
    first_slice = slice(left - first.x0, right - first.x0)
    second_slice = slice(left - second.x0, right - second.x0)
    return (
        left,
        right,
        first.rgb[:, first_slice],
        second.rgb[:, second_slice],
        first.valid_mask[:, first_slice],
        second.valid_mask[:, second_slice],
        first.source_map_x[:, first_slice],
        first.source_map_y[:, first_slice],
        second.source_map_x[:, second_slice],
        second.source_map_y[:, second_slice],
    )


def _preview_canvas_scale(
    layout: PushbroomLayout, settings: ResidualAlignmentConfig
) -> float:
    """Bound preview analysis by formal width *and* pixel working-set caps."""

    full_pixels = int(layout.canvas_height) * int(layout.canvas_width)
    if full_pixels <= 0:
        raise RuntimeError("Calibrated RGB preview canvas has no pixels")
    width_scale = min(1.0, float(settings.analysis_width) / float(layout.canvas_width))
    pixel_scale = min(
        1.0,
        math.sqrt(
            float(settings.maximum_preview_megapixels) * 1_000_000.0
            / float(full_pixels)
        ),
    )
    return min(width_scale, pixel_scale)


def _residual_support_margins(layout: PushbroomLayout) -> tuple[float, ...]:
    """Return calibrated support spare for each source-owned interval."""

    margins: list[float] = []
    for left, right, support_left, support_right in zip(
        layout.owner_left_x,
        layout.owner_right_x,
        layout.support_left_x,
        layout.support_right_x,
        strict=True,
    ):
        # Endpoint outer FOV may sit directly on a calibrated image edge.  It
        # has no deformation spare, so a non-identity residual must pin that
        # source to identity instead of silently consuming an owner core.
        margins.append(max(0.0, min(float(left - support_left), float(support_right - right))))
    return tuple(margins)


def _residual_support_bounds(
    layout: PushbroomLayout,
) -> tuple[tuple[float, float, float, float], ...]:
    """Conservative virtual rectangles used to audit full SE(2) deformation."""

    bottom = float(layout.canvas_height - 1)
    return tuple(
        (
            float(left),
            float(right - 1),
            0.0,
            bottom,
        )
        for left, right in zip(
            layout.support_left_x, layout.support_right_x, strict=True
        )
    )


def _minimum_residual_support_margin(layout: PushbroomLayout) -> float:
    return min(_residual_support_margins(layout), default=0.0)


def _held_out_edge_metrics(
    evidence: Sequence[object],
) -> dict[str, float | int | None]:
    """Aggregate only immutable held-out edge samples for model comparison."""

    values: list[np.ndarray] = []
    accepted = 0
    held_out = 0
    for item in evidence:
        # Keep this helper structural: it only consumes the public evidence
        # arrays and does not need a renderer-specific class import.
        held_mask = np.asarray(getattr(item, "held_out_mask"), dtype=bool)
        accepted_mask = np.asarray(getattr(item, "accepted_mask"), dtype=bool)
        edge_steps = np.asarray(getattr(item, "edge_normal_step_pixels"), dtype=np.float64)
        samples = edge_steps[held_mask & accepted_mask]
        samples = samples[np.isfinite(samples)]
        if samples.size:
            values.append(samples)
        accepted += int(np.count_nonzero(held_mask & accepted_mask))
        held_out += int(np.count_nonzero(held_mask))
    merged = np.concatenate(values) if values else np.empty(0, dtype=np.float64)
    return {
        "held_out_correspondence_count": held_out,
        "held_out_accepted_count": accepted,
        "held_out_edge_step_p50_pixels": (
            float(np.percentile(merged, 50.0)) if merged.size else None
        ),
        "held_out_edge_step_p95_pixels": (
            float(np.percentile(merged, 95.0)) if merged.size else None
        ),
        "held_out_edge_step_max_pixels": float(np.max(merged)) if merged.size else None,
    }


def _identity_residual_alignment_result(
    layout: PushbroomLayout,
    calibration: CameraIntrinsics,
    settings: ResidualAlignmentConfig,
    evidence: Sequence[object],
    *,
    preview_remap_count: int,
    support_bounds: Sequence[tuple[float, float, float, float]] | None = None,
    held_out_additional: Mapping[str, float | int | str | bool | None] | None = None,
    working_set_additional: Mapping[str, int | float | str | bool | None] | None = None,
) -> ResidualAlignmentResult:
    """Build the Phase-0 selected model without changing any output mapping."""

    warps = tuple(
        SourceResidualWarp.identity(
            index,
            centre_x=float(layout.source_centres_x[index]),
            centre_y=float(calibration.cy),
        )
        for index in range(len(layout.frame_ids))
    )
    topology = audit_source_warps(
        warps,
        settings,
        support_margin_pixels=_minimum_residual_support_margin(layout),
        support_bounds=support_bounds,
    )
    if not topology.accepted:
        raise RuntimeError("Identity residual-alignment topology audit failed")
    held_out = _held_out_edge_metrics(evidence)
    if held_out_additional:
        held_out.update(held_out_additional)
    working_set: dict[str, int | float | bool | str | None] = {
        "analysis_preview_remap_count": int(preview_remap_count),
        "preview_streaming_maximum_resident_strips": 2,
        "preview_canvas_scale": _preview_canvas_scale(layout, settings),
        "preview_output_pixels": int(
            math.ceil(layout.canvas_height * _preview_canvas_scale(layout, settings))
            * math.ceil(layout.canvas_width * _preview_canvas_scale(layout, settings))
        ),
    }
    if working_set_additional:
        working_set.update(working_set_additional)
    return ResidualAlignmentResult(
        selected_model="identity",
        source_warps=warps,
        pair_evidence=tuple(evidence),
        held_out_metrics_before=held_out,
        held_out_metrics_after=dict(held_out),
        component_tracks=(),
        topology_audit=topology,
        working_set_audit=working_set,
    )


def _select_seam_search_corridor(
    left: int,
    right: int,
    first: np.ndarray,
    second: np.ndarray,
    valid0: np.ndarray,
    valid1: np.ndarray,
    *,
    nominal_boundary_x: float,
    requested_width: int,
    allow_short_endpoint_corridor: bool = False,
) -> tuple[int, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Restrict a wide calibrated overlap to the formal owner-search band.

    Full central-strip overlap is retained for reliable global photometry only.
    This helper is the sole path by which GraphCut, risk guards, owners, and
    MultiBand see a corridor, keeping their search at 32--64 px even when the
    source's legal 20% central support is substantially wider.
    """

    available = int(right - left)
    if available <= 0:
        raise RuntimeError("Adjacent calibrated RGB strips have no seam support")
    if available < 32 and not allow_short_endpoint_corridor:
        raise RuntimeError(
            "Interior calibrated RGB seam support is below the 32-pixel "
            "formal search minimum"
        )
    width = min(int(requested_width), available)
    if width < 2:
        raise RuntimeError("Calibrated RGB seam-search support collapsed")
    centre = int(round(float(nominal_boundary_x)))
    start = centre - width // 2
    start = int(np.clip(start, left, right - width))
    stop = start + width
    source = slice(start - left, stop - left)
    return (
        start,
        stop,
        first[:, source],
        second[:, source],
        valid0[:, source],
        valid1[:, source],
    )


def _solve_global_linear_rgb_gains(
    source_count: int, edges: Sequence[_PhotometricEdge | None]
) -> tuple[np.ndarray, dict[str, int | float | bool]]:
    """Jointly solve all 3N linear-light gains with robust smoothness.

    Every adjacent relation is mandatory.  Unlike sequential accumulation,
    this makes all measurements participate in a single IRLS problem and uses
    a second-difference penalty only on the *gain curve*, never on image
    pixels.  A mean-zero gauge preserves the overall scene brightness.
    """

    if len(edges) != source_count - 1:
        raise ValueError("Photometric observations must cover every adjacent pair")
    missing = [index for index, edge in enumerate(edges) if edge is None]
    if missing:
        raise RuntimeError(
            "No reliable safe white-wall photometric support for adjacent pair(s): "
            + ", ".join(str(index) for index in missing)
        )
    checked = [edge for edge in edges if edge is not None]
    assert len(checked) == source_count - 1
    relations = np.asarray([edge.log_relation_bgr for edge in checked], dtype=np.float64)
    mads = np.asarray([edge.mad_bgr for edge in checked], dtype=np.float64)
    support = np.asarray([edge.support_pixels for edge in checked], dtype=np.float64)
    if (
        relations.shape != (source_count - 1, 3)
        or not np.isfinite(relations).all()
        or not np.isfinite(mads).all()
        or np.any(mads < 0.0)
        or np.any(support <= 0.0)
    ):
        raise RuntimeError("Safe white-wall photometric observations are invalid")

    log_gains = np.empty((source_count, 3), dtype=np.float64)
    residuals = np.empty_like(relations)
    maximum_condition = 0.0
    iterations = 4
    # The robust wall statistics are measured from thousands of pixels on most
    # edges.  Keep a light second-difference prior to suppress an isolated bad
    # edge without biasing the channel relations enough to leave a visible
    # colour-temperature step at every owner boundary.
    smoothness = 0.08
    for channel in range(3):
        confidence = np.clip(support / 512.0, 0.25, 12.0) / np.maximum(
            np.square(np.maximum(mads[:, channel], 0.004) / 0.012), 1.0
        )
        base_weight = confidence / max(float(np.median(confidence)), 1e-9)
        base_weight = np.clip(base_weight, 0.10, 10.0)
        robust_weight = np.ones(source_count - 1, dtype=np.float64)
        solution = np.zeros(source_count, dtype=np.float64)
        for _ in range(iterations):
            rows: list[np.ndarray] = []
            rhs: list[float] = []
            for index, relation in enumerate(relations[:, channel]):
                row = np.zeros(source_count, dtype=np.float64)
                weight = math.sqrt(float(base_weight[index] * robust_weight[index]))
                row[index] = -weight
                row[index + 1] = weight
                rows.append(row)
                rhs.append(weight * float(relation))
            if source_count >= 3:
                curvature_weight = math.sqrt(smoothness)
                for index in range(1, source_count - 1):
                    row = np.zeros(source_count, dtype=np.float64)
                    row[index - 1 : index + 2] = (
                        curvature_weight,
                        -2.0 * curvature_weight,
                        curvature_weight,
                    )
                    rows.append(row)
                    rhs.append(0.0)
            gauge = np.full(source_count, math.sqrt(100.0 / source_count), dtype=np.float64)
            rows.append(gauge)
            rhs.append(0.0)
            matrix = np.vstack(rows)
            target = np.asarray(rhs, dtype=np.float64)
            rank = int(np.linalg.matrix_rank(matrix))
            condition = float(np.linalg.cond(matrix))
            maximum_condition = max(maximum_condition, condition)
            if rank != source_count or not np.isfinite(condition) or condition > 1e8:
                raise RuntimeError("Global RGB photometric solver is rank-deficient or ill-conditioned")
            solution, _, _, _ = np.linalg.lstsq(matrix, target, rcond=None)
            if not np.isfinite(solution).all():
                raise RuntimeError("Global RGB photometric solver returned non-finite gains")
            residual = np.diff(solution) - relations[:, channel]
            scale = max(0.0025, 1.4826 * float(np.median(np.abs(residual))))
            robust_weight = np.minimum(
                1.0, 1.5 * scale / np.maximum(np.abs(residual), 1e-9)
            )
        log_gains[:, channel] = solution
        residuals[:, channel] = np.diff(solution) - relations[:, channel]

    gains = np.exp(log_gains)
    if not np.isfinite(gains).all() or np.any(gains < 0.45) or np.any(gains > 2.20):
        raise RuntimeError("Global linear RGB gains exceed the formal 0.45-2.20 range")
    return gains, {
        "photometric_mode": "safe_wall_global_linear_rgb",
        "safe_exposure_pair_count": len(checked),
        "safe_exposure_pair_fraction": len(checked) / max(1, source_count - 1),
        "safe_exposure_support_pixel_count": int(np.sum(support)),
        "safe_exposure_support_pixel_count_per_channel": int(np.sum(support) * 3),
        "photometric_global_solver": True,
        "photometric_solver_irls_iterations": iterations,
        "photometric_solver_second_difference_lambda": smoothness,
        "photometric_solver_condition": maximum_condition,
        "photometric_log_residual_p95": float(np.percentile(np.abs(residuals), 95.0)),
        "photometric_edge_log_mad_p95": float(np.percentile(mads, 95.0)),
    }


def _graphcut_monotonic_owner(
    first: np.ndarray,
    second: np.ndarray,
    valid0: np.ndarray,
    valid1: np.ndarray,
    protected: np.ndarray,
    nominal_boundary: int,
    *,
    component_owner_constraints: Mapping[int, int] | None = None,
    owner_prior: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool, int, int, int]:
    """Constrain GraphCut to whole protected components and validate its seam.

    The previous implementation split a protected component at the nominal
    boundary and then rewrote the result with a row-wise ``max``.  Here the
    GraphCut masks receive an all-or-nothing owner for every connected risk
    component.  Optional sequence preflight constraints are keyed by the
    local connected-component label and use ``0`` / ``1`` for first / second
    owner.  Its returned owner map is used directly only when it is a valid
    monotonic prefix; otherwise a safe unblended hard cut is allowed.
    """

    height, width = valid0.shape
    common = np.asarray(valid0, dtype=bool) & np.asarray(valid1, dtype=bool)
    protected = np.asarray(protected, dtype=bool) & common
    nominal_boundary = int(np.clip(nominal_boundary, 0, width - 1))
    labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        protected.astype(np.uint8), connectivity=8
    )
    constraints = {} if component_owner_constraints is None else dict(component_owner_constraints)
    for label, owner in constraints.items():
        if not isinstance(label, (int, np.integer)) or not 1 <= int(label) < labels_count:
            raise ValueError("Component owner constraint references an unknown label")
        if int(owner) not in {0, 1}:
            raise ValueError("Component owner constraints must select owner 0 or 1")
    lower = np.full(height, -1, dtype=np.int32)
    upper = np.full(height, width - 1, dtype=np.int32)
    force0 = np.zeros_like(common)
    force1 = np.zeros_like(common)
    if owner_prior is not None:
        prior = np.asarray(owner_prior)
        if prior.shape != common.shape:
            raise ValueError("Owner prior must match the GraphCut corridor")
        if not np.isin(prior, (-1, 0, 1)).all():
            raise ValueError("Owner prior values must be -1, 0, or 1")
        force0 |= common & (prior == 0)
        force1 |= common & (prior == 1)
        if np.any(force0 & force1):
            raise ValueError("Owner prior selects both sources for one pixel")
        for row in range(height):
            first_columns = np.flatnonzero(force0[row])
            second_columns = np.flatnonzero(force1[row])
            if first_columns.size:
                lower[row] = max(lower[row], int(first_columns[-1]))
            if second_columns.size:
                upper[row] = min(upper[row], int(second_columns[0]) - 1)
    # Preserve a real contribution from both adjacent strips whenever a safe
    # corridor exists.  GraphCut otherwise has no terminal costs in this small
    # two-image overlap and can legally choose the same image for every row,
    # silently erasing a narrow intermediate pose node.  These are single-pixel
    # terminal constraints at the outermost safe columns, not a restriction of
    # the 32--64 px seam search itself.  Rows occupied completely by one
    # protected component deliberately get no such anchors; assigning that
    # component as a whole is safer than slicing it at an artificial terminal.
    anchor0 = np.full(height, -1, dtype=np.int32)
    anchor1 = np.full(height, -1, dtype=np.int32)
    for label in range(1, labels_count):
        if int(stats[label, cv2.CC_STAT_AREA]) == 0:
            continue
        component = labels == label
        xs = np.flatnonzero(np.any(component, axis=0))
        if not xs.size:
            continue
        left, right = int(xs[0]), int(xs[-1])
        constrained_owner = constraints.get(label)
        if constrained_owner is not None:
            choose_first = int(constrained_owner) == 0
        elif right <= nominal_boundary:
            choose_first = True
        elif left > nominal_boundary:
            choose_first = False
        else:
            # For a component straddling the nominal centre, choose exactly
            # one side according to the smaller necessary seam displacement.
            choose_first = (right - nominal_boundary) <= (nominal_boundary - left + 1)
        rows = np.flatnonzero(np.any(component, axis=1))
        if choose_first:
            force0 |= component
            for row in rows:
                lower[row] = max(lower[row], int(np.flatnonzero(component[row])[-1]))
        else:
            force1 |= component
            for row in rows:
                upper[row] = min(upper[row], int(np.flatnonzero(component[row])[0]) - 1)
    if np.any(force0 & force1):
        raise RuntimeError("Sequence owner constraints conflict in a protected component")
    safe_common = common & ~protected
    for row in range(height):
        safe_columns = np.flatnonzero(safe_common[row])
        if safe_columns.size < 2:
            continue
        # A left terminal can only belong to the first image and a right
        # terminal only to the second.  Keep them strictly ordered so a one
        # pixel slit cannot manufacture an impossible seam.
        left_candidates = safe_columns[safe_columns <= nominal_boundary]
        right_candidates = safe_columns[safe_columns > nominal_boundary]
        if not left_candidates.size or not right_candidates.size:
            continue
        left_terminal = int(left_candidates[0])
        right_terminal = int(right_candidates[-1])
        if left_terminal >= right_terminal:
            continue
        anchor0[row] = left_terminal
        anchor1[row] = right_terminal
    if np.any(lower > upper):
        raise RuntimeError("No safe hard-owner corridor around a protected RGB component")

    masks = [valid0.astype(np.uint8) * 255, valid1.astype(np.uint8) * 255]
    masks[1][force0] = 0
    masks[0][force1] = 0
    for row in range(height):
        if anchor0[row] >= 0:
            masks[1][row, anchor0[row]] = 0
        if anchor1[row] >= 0:
            masks[0][row, anchor1[row]] = 0

    def owners_from_prefix(
        candidate: np.ndarray, *, enforce_terminals: bool = False
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
        candidate = np.asarray(candidate, dtype=bool)
        if enforce_terminals:
            for row in range(height):
                if anchor0[row] >= 0 and not candidate[row, anchor0[row]]:
                    raise RuntimeError("GraphCut discarded a first-owner terminal")
                if anchor1[row] >= 0 and candidate[row, anchor1[row]]:
                    raise RuntimeError("GraphCut discarded a second-owner terminal")
        owner0 = valid0 & (~valid1 | candidate)
        owner1 = valid1 & (~valid0 | ~candidate)
        union = valid0 | valid1
        if np.any(union & ~(owner0 | owner1)) or np.any(owner0 & owner1):
            raise RuntimeError("RGB hard-owner seam left an invalid or multiply-owned pixel")
        cuts = np.full(height, nominal_boundary, dtype=np.int32)
        boundary_guard_pixels = 0
        for row in range(height):
            indices = np.flatnonzero(common[row])
            if not indices.size:
                continue
            prefix = owner0[row, indices]
            false_indices = np.flatnonzero(~prefix)
            if false_indices.size and np.any(prefix[false_indices[0] + 1 :]):
                raise RuntimeError("GraphCut returned a non-monotonic hard-owner seam")
            cut = int(indices[np.flatnonzero(prefix)[-1]]) if np.any(prefix) else int(indices[0] - 1)
            if cut < lower[row] or cut > upper[row]:
                raise RuntimeError("GraphCut violated a protected-component owner constraint")
            cuts[row] = cut
            # A component can legitimately own the complete endpoint overlap.
            # With only one owner present there is no actual inter-source
            # boundary in this row, so it cannot violate the boundary guard.
            if np.any(prefix) and np.any(~prefix):
                for position in (cut, cut + 1):
                    if 0 <= position < width and protected[row, position]:
                        boundary_guard_pixels += 1
        split_count = 0
        for label in range(1, labels_count):
            component = labels == label
            if np.any(owner0 & component) and np.any(owner1 & component):
                split_count += 1
        if split_count:
            raise RuntimeError("A protected RGB component was split between hard owners")
        if boundary_guard_pixels:
            raise RuntimeError("A hard-owner boundary intersects the RGB risk guard")
        return owner0, owner1, cuts, split_count, boundary_guard_pixels

    graphcut_candidate: np.ndarray | None = None
    try:
        finder = cv2.detail_GraphCutSeamFinder("COST_COLOR_GRAD")
        work_masks = [
            np.ascontiguousarray(masks[0]),
            np.ascontiguousarray(masks[1]),
        ]
        output = finder.find(
            [
                np.ascontiguousarray(first, dtype=np.float32),
                np.ascontiguousarray(second, dtype=np.float32),
            ],
            [(0, 0), (0, 0)],
            work_masks,
        )
        result_masks = work_masks if output is None else output
        if len(result_masks) == 2:
            candidate = result_masks[0].get() if hasattr(result_masks[0], "get") else result_masks[0]
            candidate = np.asarray(candidate, dtype=np.uint8)
            if candidate.shape == valid0.shape:
                graphcut_candidate = candidate > 0
    except cv2.error:
        graphcut_candidate = None

    if graphcut_candidate is not None:
        try:
            owner0, owner1, cuts, split_count, boundary_guard = owners_from_prefix(
                graphcut_candidate, enforce_terminals=True
            )
            return owner0, owner1, cuts, True, 0, split_count, boundary_guard
        except RuntimeError:
            # Do not repair a non-monotonic GraphCut by taking a row-wise
            # maximum.  The only permitted fallback is a safe direct hard cut.
            pass

    hard_candidate = np.zeros_like(common)
    hard_rows = 0
    for row in range(height):
        indices = np.flatnonzero(common[row])
        if not indices.size:
            continue
        if lower[row] >= width - 1:
            hard_candidate[row] = True
            hard_rows += 1
            continue
        if upper[row] < 0:
            hard_rows += 1
            continue
        candidates = np.arange(max(0, lower[row]), min(width - 1, upper[row]) + 1)
        candidates = candidates[
            ~protected[row, candidates]
            & ~protected[row, np.minimum(candidates + 1, width - 1)]
        ]
        if not candidates.size:
            # A protected component may span the complete search corridor in
            # this row (for example a horizontal hose).  It can still remain
            # intact if the entire row is assigned to its chosen owner; no
            # inter-source boundary then crosses the guard.  Competing forced
            # owners remain a genuine no-safe-channel structural failure.
            if lower[row] >= 0 and upper[row] == width - 1:
                hard_candidate[row] = True
                hard_rows += 1
                continue
            if upper[row] < width - 1 and lower[row] < 0:
                hard_rows += 1
                continue
            raise RuntimeError("No safe direct hard cut exists outside the RGB risk guard")
        cut = int(candidates[np.argmin(np.abs(candidates - nominal_boundary))])
        hard_candidate[row] = np.arange(width) <= cut
        hard_rows += 1
    owner0, owner1, cuts, split_count, boundary_guard = owners_from_prefix(hard_candidate)
    return owner0, owner1, cuts, False, hard_rows, split_count, boundary_guard


def _blend_safe_pair_zone(
    first: np.ndarray,
    second: np.ndarray,
    common: np.ndarray,
    protected: np.ndarray,
    safe_wall: np.ndarray,
    owner0: np.ndarray,
    owner1: np.ndarray,
    cuts: np.ndarray,
    blend_width: int,
    levels: int,
) -> tuple[np.ndarray, np.ndarray, int, int, int, int, bool]:
    """Use complementary owner masks in one local safe-wall MultiBand pass."""

    direct = np.zeros_like(first)
    direct[owner0] = first[owner0]
    direct[owner1] = second[owner1]
    if blend_width < 2:
        raise ValueError("Safe RGB blend width must be at least two pixels")
    x = np.arange(first.shape[1], dtype=np.int32)[None, :]
    left_span = (int(blend_width) - 1) // 2
    right_span = int(blend_width) - left_span - 1
    zone = (
        common
        & ~protected
        & safe_wall
        & (x >= cuts[:, None] - left_span)
        & (x <= cuts[:, None] + right_span)
    )
    effective_levels = _adaptive_multiband_levels(blend_width, levels)
    if not np.any(zone):
        return direct, zone, 0, effective_levels, 0, 0, False
    # Keep the pyramid local, but make each source's mask begin with its own
    # hard owner.  The two dilations overlap only within safe wall support;
    # they are never the same full-support 50/50 mask.
    support_radius = max(1, 1 << max(0, effective_levels - 1))
    support_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (support_radius * 2 + 1, support_radius * 2 + 1)
    )
    support = cv2.dilate(zone.astype(np.uint8), support_kernel) > 0
    support &= common & ~protected & safe_wall
    owner_radius = max(1, int(math.ceil(blend_width / 2.0)))
    owner_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (owner_radius * 2 + 1, owner_radius * 2 + 1)
    )
    mask0 = cv2.dilate((owner0 & safe_wall).astype(np.uint8), owner_kernel) > 0
    mask1 = cv2.dilate((owner1 & safe_wall).astype(np.uint8), owner_kernel) > 0
    mask0 &= support
    mask1 &= support
    zone &= mask0 & mask1
    if not np.any(zone):
        return (
            direct,
            zone,
            0,
            effective_levels,
            int(np.count_nonzero(mask0)),
            int(np.count_nonzero(mask1)),
            False,
        )
    if np.array_equal(mask0, mask1):
        # A tiny legal corridor (common in dense pose-node sequences) can make
        # both dilations cover the exact same support.  Feeding that pair would
        # recreate the forbidden 50/50 average, so retain the audited hard cut
        # and omit MultiBand for this pair instead of inventing a feather mask.
        return (
            direct,
            np.zeros_like(zone),
            0,
            effective_levels,
            int(np.count_nonzero(mask0)),
            int(np.count_nonzero(mask1)),
            False,
        )
    mask0_u8 = np.where(mask0, 255, 0).astype(np.uint8)
    mask1_u8 = np.where(mask1, 255, 0).astype(np.uint8)
    blender = cv2.detail_MultiBandBlender()
    blender.setNumBands(effective_levels)
    blender.prepare((0, 0, first.shape[1], first.shape[0]))
    try:
        blender.feed(np.ascontiguousarray(first, dtype=np.int16), mask0_u8, (0, 0))
        blender.feed(np.ascontiguousarray(second, dtype=np.int16), mask1_u8, (0, 0))
        blended, output_mask = blender.blend(None, None)
    except cv2.error as exc:
        raise RuntimeError("Safe RGB local MultiBand blending failed") from exc
    if hasattr(blended, "get"):
        blended = blended.get()
    if hasattr(output_mask, "get"):
        output_mask = output_mask.get()
    blended = np.clip(np.asarray(blended), 0, 255).astype(np.uint8)
    output_mask = np.asarray(output_mask, dtype=np.uint8)
    if blended.shape != first.shape or output_mask.shape != common.shape:
        raise RuntimeError("Safe RGB local MultiBand returned an invalid output")
    if np.any(zone & (output_mask == 0)):
        raise RuntimeError("Safe RGB local MultiBand produced a zero-weight wedge")
    if np.any(zone & protected):
        raise RuntimeError("RGB risk protection reached a MultiBand zone")
    direct[zone] = blended[zone]
    return (
        direct,
        zone,
        int(np.count_nonzero(zone)),
        effective_levels,
        int(np.count_nonzero(mask0)),
        int(np.count_nonzero(mask1)),
        True,
    )


def _delta_e00(lab0: np.ndarray, lab1: np.ndarray) -> np.ndarray:
    """Vectorised CIEDE2000 for standard (not uint8-encoded) CIE Lab."""

    first = np.asarray(lab0, dtype=np.float64)
    second = np.asarray(lab1, dtype=np.float64)
    l1, a1, b1 = first[..., 0], first[..., 1], first[..., 2]
    l2, a2, b2 = second[..., 0], second[..., 1], second[..., 2]
    c1 = np.hypot(a1, b1)
    c2 = np.hypot(a2, b2)
    mean_c = 0.5 * (c1 + c2)
    mean_c7 = np.power(mean_c, 7)
    g = 0.5 * (1.0 - np.sqrt(mean_c7 / (mean_c7 + 25.0**7)))
    a1p, a2p = (1.0 + g) * a1, (1.0 + g) * a2
    c1p, c2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1p = np.mod(np.degrees(np.arctan2(b1, a1p)), 360.0)
    h2p = np.mod(np.degrees(np.arctan2(b2, a2p)), 360.0)
    delta_l = l2 - l1
    delta_c = c2p - c1p
    hue_difference = h2p - h1p
    hue_difference = np.where(hue_difference > 180.0, hue_difference - 360.0, hue_difference)
    hue_difference = np.where(hue_difference < -180.0, hue_difference + 360.0, hue_difference)
    hue_difference = np.where((c1p * c2p) == 0.0, 0.0, hue_difference)
    delta_h = 2.0 * np.sqrt(c1p * c2p) * np.sin(np.radians(hue_difference / 2.0))
    mean_l = 0.5 * (l1 + l2)
    mean_cp = 0.5 * (c1p + c2p)
    mean_hp = 0.5 * (h1p + h2p)
    mean_hp = np.where(np.abs(h1p - h2p) > 180.0, mean_hp + 180.0, mean_hp)
    mean_hp = np.where((c1p * c2p) == 0.0, h1p + h2p, mean_hp)
    mean_hp = np.mod(mean_hp, 360.0)
    t = (
        1.0
        - 0.17 * np.cos(np.radians(mean_hp - 30.0))
        + 0.24 * np.cos(np.radians(2.0 * mean_hp))
        + 0.32 * np.cos(np.radians(3.0 * mean_hp + 6.0))
        - 0.20 * np.cos(np.radians(4.0 * mean_hp - 63.0))
    )
    delta_theta = 30.0 * np.exp(-np.square((mean_hp - 275.0) / 25.0))
    rc = 2.0 * np.sqrt(np.power(mean_cp, 7) / (np.power(mean_cp, 7) + 25.0**7))
    sl = 1.0 + (0.015 * np.square(mean_l - 50.0)) / np.sqrt(20.0 + np.square(mean_l - 50.0))
    sc = 1.0 + 0.045 * mean_cp
    sh = 1.0 + 0.015 * mean_cp * t
    rt = -np.sin(np.radians(2.0 * delta_theta)) * rc
    return np.sqrt(
        np.square(delta_l / sl)
        + np.square(delta_c / sc)
        + np.square(delta_h / sh)
        + rt * (delta_c / sc) * (delta_h / sh)
    )


def _bgr_to_cie_lab(image: np.ndarray) -> np.ndarray:
    """Convert uint8 BGR to continuous CIE Lab for a quality measurement."""

    values = np.ascontiguousarray(np.asarray(image, dtype=np.float32) / 255.0)
    return cv2.cvtColor(values, cv2.COLOR_BGR2LAB)


def _robust_wall_patch_samples(
    lab0: np.ndarray,
    lab1: np.ndarray,
    samples: np.ndarray,
    *,
    tile_height: int = 64,
    tile_width: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Return local median wall colours without filtering any rendered pixels.

    The metric is intentionally computed from independently safe pixels in an
    64x4 neighbourhood.  It rejects isolated inverse-remap/JPEG sample noise
    while retaining an owner-boundary colour step: a global per-frame colour
    shift is coherent in both dimensions and therefore survives each local
    median.  This helper is only a quality measurement; it never feeds a
    blend, owner decision, or output image.
    """

    safe = np.asarray(samples, dtype=bool)
    if lab0.shape != lab1.shape or safe.shape != lab0.shape[:2]:
        raise ValueError("Wall quality samples must match both Lab images")
    height, width = safe.shape
    first_patches: list[np.ndarray] = []
    second_patches: list[np.ndarray] = []
    height_step = max(2, int(tile_height))
    width_step = max(2, int(tile_width))
    for y0 in range(0, height, height_step):
        y1 = min(height, y0 + height_step)
        for x0 in range(0, width, width_step):
            x1 = min(width, x0 + width_step)
            patch = safe[y0:y1, x0:x1]
            minimum = max(8, int(math.ceil(0.25 * patch.size)))
            if int(np.count_nonzero(patch)) < minimum:
                continue
            first_patches.append(np.median(lab0[y0:y1, x0:x1][patch], axis=0))
            second_patches.append(np.median(lab1[y0:y1, x0:x1][patch], axis=0))
    if len(first_patches) < 4:
        # Sparse wall support is still a valid photometric observation, but it
        # cannot form four independent tiles.  Report its robust whole-support
        # medians rather than reverting to individual JPEG/remap samples.
        return (
            np.median(lab0[safe], axis=0, keepdims=True),
            np.median(lab1[safe], axis=0, keepdims=True),
        )
    return np.asarray(first_patches), np.asarray(second_patches)


def _white_wall_pair_metrics(
    first: np.ndarray,
    second: np.ndarray,
    safe_wall: np.ndarray,
    cuts: np.ndarray,
) -> tuple[
    float | None,
    float | None,
    float | None,
    int,
    np.ndarray,
    np.ndarray,
]:
    """Measure corrected ΔL*/ΔE00 near a real hard-owner boundary."""

    x = np.arange(first.shape[1], dtype=np.int32)[None, :]
    boundary = np.abs(x - cuts[:, None]) <= 2
    samples = np.asarray(safe_wall, dtype=bool) & boundary
    # A pair can legitimately have no neutral wall right at a foreground
    # detour.  Its wider safe-wall support still measures colour continuity.
    if int(np.count_nonzero(samples)) < 32:
        samples = np.asarray(safe_wall, dtype=bool)
    count = int(np.count_nonzero(samples))
    if count < 32:
        return (
            None,
            None,
            None,
            count,
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
        )
    lab0 = _bgr_to_cie_lab(first)
    lab1 = _bgr_to_cie_lab(second)
    first_samples, second_samples = _robust_wall_patch_samples(lab0, lab1, samples)
    delta_l = first_samples[:, 0] - second_samples[:, 0]
    delta_e = _delta_e00(first_samples, second_samples)
    return (
        float(np.percentile(np.abs(delta_l), 95.0)),
        float(np.percentile(delta_e, 95.0)),
        float(np.median(delta_l)),
        count,
        np.asarray(np.abs(delta_l), dtype=np.float64),
        np.asarray(delta_e, dtype=np.float64),
    )


def _periodic_stripe_energy(values: Sequence[float]) -> float | None:
    """High-frequency residual energy after removing an exposure drift line."""

    samples = np.asarray(values, dtype=np.float64)
    samples = samples[np.isfinite(samples)]
    if samples.size < 5:
        return None
    x = np.arange(samples.size, dtype=np.float64)
    trend = np.polyval(np.polyfit(x, samples, 1), x)
    residual = samples - trend
    smooth = np.convolve(np.pad(residual, (2, 2), mode="edge"), np.ones(5) / 5.0, mode="valid")
    return float(np.sqrt(np.mean(np.square(residual - smooth))))


def _write_owner_core(
    canvas: np.ndarray,
    valid_canvas: np.ndarray,
    owner_canvas: np.ndarray,
    contribution: PushbroomContribution,
    image: np.ndarray,
    left: float,
    right: float,
) -> None:
    x0 = max(contribution.x0, int(math.ceil(left)))
    x1 = min(contribution.x1, int(math.ceil(right)))
    if x1 <= x0:
        return
    source_slice = slice(x0 - contribution.x0, x1 - contribution.x0)
    destination = slice(x0, x1)
    usable = contribution.valid_mask[:, source_slice]
    if not np.any(usable):
        return
    source = image[:, source_slice]
    canvas_view = canvas[:, destination]
    valid_view = valid_canvas[:, destination]
    owner_view = owner_canvas[:, destination]
    canvas_view[usable] = source[usable]
    valid_view[usable] = True
    owner_view[usable] = contribution.source_index


def render_calibrated_rgb_pushbroom(
    frames: Sequence[RGBDFrame],
    poses: Sequence[np.ndarray],
    calibration: CameraIntrinsics,
    *,
    config: Mapping[str, object] | CalibratedRGBPushbroomConfig | None = None,
    rgb_motions: Sequence[object] | None = None,
    motion_pixels_to_full_resolution: float = 1.0,
    multiband_levels: int = 3,
    quality_gate: bool = True,
) -> CalibratedRGBPushbroomResult:
    """Render all real pose frames as a bounded-memory RGB-only pushbroom.

    The only samples written to ``panorama`` originate from a source frame's RGB
    array.  Aligned depth is intentionally absent from this function's inputs and
    implementation.
    """

    settings = (
        config
        if isinstance(config, CalibratedRGBPushbroomConfig)
        else CalibratedRGBPushbroomConfig.from_mapping(config)
    )
    if len(frames) != len(poses) or not 2 <= len(frames) <= settings.max_pose_count:
        raise ValueError("Calibrated RGB pushbroom requires 2-160 aligned frames/poses")
    if not 1 <= int(multiband_levels) <= 3:
        raise ValueError("Calibrated RGB pushbroom MultiBand levels must be in [1, 3]")
    checked_poses = [validate_camera_to_world(pose) for pose in poses]
    scale = estimate_rgb_motion_pixels_per_mm(
        frames,
        checked_poses,
        calibration,
        settings,
        rgb_motions=rgb_motions,
        motion_pixels_to_full_resolution=motion_pixels_to_full_resolution,
    )
    layout = build_calibrated_rgb_pushbroom_layout(
        [frame.frame_id for frame in frames], checked_poses, calibration, scale, settings
    )
    residual_support_bounds = _residual_support_bounds(layout)
    strip_paths: list[Path] = []
    with tempfile.TemporaryDirectory(prefix="g305-rgb-pushbroom-") as temporary:
        root = Path(temporary)
        # Phase 0/preview pass: calibrated low-resolution RGB evidence is
        # spooled independently and never enters the formal output canvas.
        analysis_renderer = CalibratedRGBPushbroomRenderer(
            layout, calibration, checked_poses
        )
        preview_scale = _preview_canvas_scale(layout, settings.residual_alignment)
        preview_evidence: list[object] = []
        preview_boundary_geometry: list[dict[str, object]] = []
        preview_pair_origins: list[tuple[float, float]] = []
        preview_evidence_pixels = 0
        preview_evidence_limit_pixels = int(
            math.floor(
                settings.residual_alignment.maximum_evidence_megapixels * 1_000_000.0
            )
        )
        previous_preview_path: Path | None = None
        for index, frame in enumerate(frames):
            preview = analysis_renderer.render_preview_frame(
                frame, index, canvas_scale=preview_scale
            )
            path = root / f"preview-{index:04d}.npz"
            _save_preview(path, preview)
            if previous_preview_path is None:
                previous_preview_path = path
                continue
            first_preview = _load_preview(previous_preview_path)
            second_preview = _load_preview(path)
            (
                preview_left,
                _preview_right,
                preview_rgb0,
                preview_rgb1,
                preview_valid0,
                preview_valid1,
                preview_map0_x,
                preview_map0_y,
                preview_map1_x,
                preview_map1_y,
            ) = _preview_pair_overlap(first_preview, second_preview)
            evidence_pixels = int(preview_rgb0.shape[0]) * int(preview_rgb0.shape[1])
            preview_evidence_pixels += evidence_pixels
            if preview_evidence_pixels > preview_evidence_limit_pixels:
                raise RuntimeError(
                    "RGB residual preview evidence exceeds the formal bounded-memory "
                    "pixel limit"
                )
            pair_index = index - 1
            preview_pair_origins.append((float(preview_left), float(preview_scale)))
            evidence = extract_pair_evidence(
                first_rgb=preview_rgb0,
                second_rgb=preview_rgb1,
                first_valid=preview_valid0,
                second_valid=preview_valid1,
                first_inverse_map=(preview_map0_x, preview_map0_y),
                second_inverse_map=(preview_map1_x, preview_map1_y),
                intrinsics=calibration,
                first_camera_to_world=checked_poses[pair_index],
                second_camera_to_world=checked_poses[index],
                pair_index=pair_index,
                frame_ids=(first_preview.frame_id, second_preview.frame_id),
                config=settings.residual_alignment,
            )
            preview_evidence.append(evidence)
            preview_boundary_geometry.append(
                {
                    "pair_index": pair_index,
                    "frame_ids": [first_preview.frame_id, second_preview.frame_id],
                    **measure_owner_boundary_geometry(
                        evidence,
                        boundary_x=(
                            layout.owner_boundaries_x[pair_index] * preview_scale
                            - preview_left
                        ),
                    ),
                }
            )
            # Only the immediately adjacent preview pair is retained on disk;
            # dense evidence itself is bounded globally by the explicit pixel
            # limit above and contains no output RGB strip.
            previous_preview_path.unlink(missing_ok=True)
            previous_preview_path = path
        if previous_preview_path is not None:
            previous_preview_path.unlink(missing_ok=True)
        preview_working_set = {
            "preview_evidence_pixel_count": preview_evidence_pixels,
            "preview_evidence_hard_limit_pixels": preview_evidence_limit_pixels,
            "preview_evidence_estimated_bytes": preview_evidence_pixels * 64,
            "preview_evidence_storage": "bounded_in_memory_analysis_only",
            "preview_streaming_maximum_resident_previews": 2,
        }
        identity_alignment = _identity_residual_alignment_result(
            layout,
            calibration,
            settings.residual_alignment,
            preview_evidence,
            preview_remap_count=analysis_renderer.analysis_preview_remap_count,
            support_bounds=residual_support_bounds,
            working_set_additional=preview_working_set,
        )
        candidate_warps, background_solver_metrics, candidate_topology = (
            solve_background_se2(
                tuple(preview_evidence),
                source_centres=tuple(
                    (float(centre_x), float(calibration.cy))
                    for centre_x in layout.source_centres_x
                ),
                pair_preview_origins=tuple(preview_pair_origins),
                config=settings.residual_alignment,
                support_margins_pixels=_residual_support_margins(layout),
                support_bounds=residual_support_bounds,
            )
        )
        if candidate_warps is None:
            residual_alignment = _identity_residual_alignment_result(
                layout,
                calibration,
                settings.residual_alignment,
                preview_evidence,
                preview_remap_count=analysis_renderer.analysis_preview_remap_count,
                support_bounds=residual_support_bounds,
                held_out_additional=background_solver_metrics,
                working_set_additional={
                    **preview_working_set,
                    "background_solver_selected": False,
                    "background_solver_reason": background_solver_metrics.get("reason"),
                },
            )
        else:
            before_metrics = dict(identity_alignment.held_out_metrics_before)
            after_metrics = dict(identity_alignment.held_out_metrics_after)
            before_metrics.update(
                {
                    key: value
                    for key, value in background_solver_metrics.items()
                    if "_before_" in key or key == "held_out_background_count"
                }
            )
            after_metrics.update(
                {
                    key: value
                    for key, value in background_solver_metrics.items()
                    if "_after_" in key or key == "held_out_background_count"
                }
            )
            residual_alignment = ResidualAlignmentResult(
                selected_model="background_se2",
                source_warps=tuple(candidate_warps),
                pair_evidence=tuple(preview_evidence),
                held_out_metrics_before=before_metrics,
                held_out_metrics_after=after_metrics,
                component_tracks=(),
                topology_audit=(
                    candidate_topology
                    if candidate_topology is not None
                    else identity_alignment.topology_audit
                ),
                working_set_audit={
                    **identity_alignment.working_set_audit,
                    **background_solver_metrics,
                    "background_solver_selected": True,
                },
            )
        # A non-identity model is not selected until held-out residual and
        # topology gates are implemented and pass.  Passing explicit identity
        # warps into the compositor exercises the same map boundary while the
        # direct nominal branch preserves all prior output pixels exactly.
        renderer = CalibratedRGBPushbroomRenderer(
            layout,
            calibration,
            checked_poses,
            residual_warps=residual_alignment.source_warps,
        )
        for index, frame in enumerate(frames):
            contribution = renderer.render_frame(frame, index)
            path = root / f"{index:04d}.npz"
            _save_contribution(path, contribution)
            strip_paths.append(path)

        photometric_edges: list[_PhotometricEdge | None] = []
        raw_risk_paths: list[Path] = []
        for index, (first_path, second_path) in enumerate(
            zip(strip_paths[:-1], strip_paths[1:], strict=True)
        ):
            first, second = _load_contribution(first_path), _load_contribution(second_path)
            (
                full_left,
                full_right,
                rgb0,
                rgb1,
                valid0,
                valid1,
            ) = _pair_overlap(first, second)
            raw_risk = _rgb_risk_details(rgb0, rgb1, valid0, valid1)
            # The safe-wall estimator excludes the complete protection band,
            # including the largest permitted local pyramid footprint.
            preliminary_guard, _, _ = _risk_guard(
                raw_risk,
                valid0 & valid1,
                blend_width=8,
                requested_levels=multiband_levels,
            )
            photometric_edges.append(
                _safe_wall_log_relation(
                    rgb0,
                    rgb1,
                    valid0 & valid1,
                    preliminary_guard,
                    low_gradient_quantile=settings.scale_low_gradient_quantile,
                )
            )
            (
                _seam_left,
                _seam_right,
                seam_rgb0,
                seam_rgb1,
                seam_valid0,
                seam_valid1,
            ) = _select_seam_search_corridor(
                full_left,
                full_right,
                rgb0,
                rgb1,
                valid0,
                valid1,
                nominal_boundary_x=layout.owner_boundaries_x[index],
                requested_width=settings.seam_search_width_pixels,
                allow_short_endpoint_corridor=index in {0, len(frames) - 2},
            )
            # Store the raw risk as evaluated in the exact owner-search
            # corridor.  A broad photometric overlap may join two foreground
            # regions outside that corridor; importing that remote connection
            # into GraphCut would falsely make a local safe path impossible.
            seam_raw_risk = _rgb_risk_details(
                seam_rgb0, seam_rgb1, seam_valid0, seam_valid1
            )
            raw_risk_path = root / f"risk-{index:04d}.npz"
            np.savez_compressed(
                raw_risk_path,
                mask=seam_raw_risk.mask,
                edge_offset_p95=np.asarray(
                    seam_raw_risk.edge_offset_p95, dtype=np.float64
                ),
                seed_pixel_count=np.asarray(
                    seam_raw_risk.seed_pixel_count, dtype=np.int32
                ),
                component_count=np.asarray(
                    seam_raw_risk.component_count, dtype=np.int32
                ),
            )
            raw_risk_paths.append(raw_risk_path)
        gains, exposure_metrics = _solve_global_linear_rgb_gains(
            len(frames), photometric_edges
        )

        # Rewrite each temporary calibrated strip after one linear-light
        # three-channel compensation.  Subsequent owner and blending passes
        # only consume these corrected strips, never compound a gain.
        for index, path in enumerate(strip_paths):
            contribution = _load_contribution(path)
            _save_contribution(
                path,
                PushbroomContribution(
                    source_index=contribution.source_index,
                    frame_id=contribution.frame_id,
                    x0=contribution.x0,
                    rgb=_apply_linear_bgr_gain(contribution.rgb, gains[index]),
                    valid_mask=contribution.valid_mask,
                ),
            )

        # Compare the same preselected safe-wall samples before and after the
        # single global gain pass.  This makes the stripe-energy audit an
        # apples-to-apples frame-sequence measurement rather than mixing a
        # broad photometric support with a later, foreground-detoured seam.
        raw_stripe_samples = [
            edge.raw_signed_l_delta
            for edge in photometric_edges
            if edge is not None
        ]
        corrected_stripe_samples: list[float] = []
        for index, (first_path, second_path) in enumerate(
            zip(strip_paths[:-1], strip_paths[1:], strict=True)
        ):
            edge = photometric_edges[index]
            if edge is None or edge.safe_mask is None:
                raise RuntimeError("Global RGB photometric audit lost safe-wall support")
            first, second = _load_contribution(first_path), _load_contribution(second_path)
            _, _, rgb0, rgb1, _, _ = _pair_overlap(first, second)
            safe = np.asarray(edge.safe_mask, dtype=bool)
            if safe.shape != rgb0.shape[:2] or not np.any(safe):
                raise RuntimeError("Global RGB photometric audit has an invalid safe-wall mask")
            lab0 = cv2.cvtColor(rgb0, cv2.COLOR_BGR2LAB).astype(np.float32)
            lab1 = cv2.cvtColor(rgb1, cv2.COLOR_BGR2LAB).astype(np.float32)
            corrected_stripe_samples.append(
                float(np.median((lab0[:, :, 0] - lab1[:, :, 0])[safe]) * (100.0 / 255.0))
            )

        # Preflight all pair-local protected components before any GraphCut is
        # permitted to write a final owner.  This pass only loads two corrected
        # strips at a time; it produces constraints/track summaries, not an
        # output owner image or an extra source remap.
        preflight_fragments_by_pair: list[tuple[object, ...]] = []
        for index, (first_path, second_path) in enumerate(
            zip(strip_paths[:-1], strip_paths[1:], strict=True)
        ):
            first, second = _load_contribution(first_path), _load_contribution(second_path)
            (
                full_left,
                full_right,
                full_image0,
                full_image1,
                full_valid0,
                full_valid1,
            ) = _pair_overlap(first, second)
            left, right, image0, image1, valid0, valid1 = _select_seam_search_corridor(
                full_left,
                full_right,
                full_image0,
                full_image1,
                full_valid0,
                full_valid1,
                nominal_boundary_x=layout.owner_boundaries_x[index],
                requested_width=settings.seam_search_width_pixels,
                allow_short_endpoint_corridor=index in {0, len(frames) - 2},
            )
            with np.load(raw_risk_paths[index], allow_pickle=False) as stored_risk:
                raw_risk_mask = np.asarray(stored_risk["mask"], dtype=np.uint8)
                raw_edge_offset = float(stored_risk["edge_offset_p95"])
            risk = _rgb_risk_details(
                image0,
                image1,
                valid0,
                valid1,
                supplied_risk_mask=raw_risk_mask,
            )
            risk = _RGBRiskDetails(
                mask=risk.mask,
                edge_offset_p95=max(raw_edge_offset, risk.edge_offset_p95),
                seed_pixel_count=risk.seed_pixel_count,
                component_count=risk.component_count,
            )
            owner_width = min(
                layout.owner_right_x[index] - layout.owner_left_x[index],
                layout.owner_right_x[index + 1] - layout.owner_left_x[index + 1],
            )
            blend_width = int(np.clip(math.floor(0.20 * owner_width), 2, 8))
            guard, _, _ = _risk_guard(
                risk,
                valid0 & valid1,
                blend_width,
                multiband_levels,
            )
            preflight_fragments_by_pair.append(
                extract_protected_component_fragments(
                    guard,
                    valid0,
                    valid1,
                    pair_index=index,
                    global_x0=left,
                    nominal_boundary_x=layout.owner_boundaries_x[index],
                )
            )
        sequence_owner_preflight = preflight_sequence_owners(
            preflight_fragments_by_pair
        )
        if not sequence_owner_preflight.accepted:
            raise RuntimeError(
                "RGB sequence owner preflight failed: "
                + str(sequence_owner_preflight.structural_failure_reason)
            )
        residual_alignment = ResidualAlignmentResult(
            selected_model=residual_alignment.selected_model,
            source_warps=residual_alignment.source_warps,
            pair_evidence=residual_alignment.pair_evidence,
            held_out_metrics_before=residual_alignment.held_out_metrics_before,
            held_out_metrics_after=residual_alignment.held_out_metrics_after,
            component_tracks=sequence_owner_preflight.component_tracks,
            topology_audit=residual_alignment.topology_audit,
            working_set_audit={
                **residual_alignment.working_set_audit,
                "sequence_owner_preflight": sequence_owner_preflight.as_dict(),
            },
        )

        canvas = np.zeros(
            (layout.canvas_height, layout.canvas_width, 3), dtype=np.uint8
        )
        valid_canvas = np.zeros((layout.canvas_height, layout.canvas_width), dtype=bool)
        owner_canvas = np.full(
            (layout.canvas_height, layout.canvas_width), -1, dtype=np.int16
        )
        for index, path in enumerate(strip_paths):
            contribution = _load_contribution(path)
            _write_owner_core(
                canvas,
                valid_canvas,
                owner_canvas,
                contribution,
                contribution.rgb,
                layout.owner_left_x[index],
                layout.owner_right_x[index],
            )

        blend_pixels = 0
        maximum_blend_risk = 0.0
        hard_cut_pairs = 0
        graphcut_pairs = 0
        protected_split_components = 0
        owner_boundary_guard_pixels = 0
        white_wall_l_samples: list[np.ndarray] = []
        white_wall_delta_e00_samples: list[np.ndarray] = []
        actual_search_widths: list[int] = []
        applied_multiband_levels: list[int] = []
        pair_metadata: list[dict[str, object]] = []
        for index, (first_path, second_path) in enumerate(
            zip(strip_paths[:-1], strip_paths[1:], strict=True)
        ):
            first, second = _load_contribution(first_path), _load_contribution(second_path)
            (
                full_left,
                full_right,
                full_image0,
                full_image1,
                full_valid0,
                full_valid1,
            ) = _pair_overlap(first, second)
            left, right, image0, image1, valid0, valid1 = _select_seam_search_corridor(
                full_left,
                full_right,
                full_image0,
                full_image1,
                full_valid0,
                full_valid1,
                nominal_boundary_x=layout.owner_boundaries_x[index],
                requested_width=settings.seam_search_width_pixels,
                allow_short_endpoint_corridor=index in {0, len(frames) - 2},
            )
            with np.load(raw_risk_paths[index], allow_pickle=False) as stored_risk:
                raw_risk_mask = np.asarray(stored_risk["mask"], dtype=np.uint8)
                raw_edge_offset = float(stored_risk["edge_offset_p95"])
            # Re-evaluate risk after the global linear-RGB correction while
            # retaining every raw risk component as an irreversible guard.
            risk = _rgb_risk_details(
                image0,
                image1,
                valid0,
                valid1,
                supplied_risk_mask=raw_risk_mask,
            )
            risk = _RGBRiskDetails(
                mask=risk.mask,
                edge_offset_p95=max(raw_edge_offset, risk.edge_offset_p95),
                seed_pixel_count=risk.seed_pixel_count,
                component_count=risk.component_count,
            )
            owner_width = min(
                layout.owner_right_x[index] - layout.owner_left_x[index],
                layout.owner_right_x[index + 1] - layout.owner_left_x[index + 1],
            )
            # This is a total blend-zone width, deliberately not a +/- radius:
            # using the latter would place roughly 40% of every narrow owner
            # strip inside a pyramid.  The capped 2--8 px zone stays below the
            # 20% global budget while retaining the requested adaptive rule.
            blend_width = int(np.clip(math.floor(0.20 * owner_width), 2, 8))
            guard, guard_radius, guard_component_count = _risk_guard(
                risk,
                valid0 & valid1,
                blend_width,
                multiband_levels,
            )
            nominal = int(round(layout.owner_boundaries_x[index])) - left
            # Endpoint outward half-FOV ownership can leave the midpoint one
            # pixel beyond the overlap support.  The nearest real corridor
            # column is the only valid nominal reference in that case.
            nominal = int(np.clip(nominal, 0, image0.shape[1] - 1))
            safe_wall = _safe_white_wall_mask(
                image0,
                image1,
                valid0 & valid1,
                guard,
                low_gradient_quantile=settings.scale_low_gradient_quantile,
            )
            try:
                (
                    owner0,
                    owner1,
                    cuts,
                    used_graphcut,
                    hard_rows,
                    split_count,
                    boundary_guard_count,
                ) = _graphcut_monotonic_owner(
                    image0,
                    image1,
                    valid0,
                    valid1,
                    guard,
                    nominal,
                    component_owner_constraints=(
                        sequence_owner_preflight.component_owner_constraints[index]
                    ),
                )
            except RuntimeError as exc:
                raise RuntimeError(
                    "No structurally safe RGB hard-owner seam for adjacent "
                    f"pair {index} ({first.frame_id}, {second.frame_id})"
                ) from exc
            (
                pair_image,
                blend_zone,
                pair_blend_pixels,
                pair_levels,
                mask0_pixels,
                mask1_pixels,
                masks_distinct,
            ) = _blend_safe_pair_zone(
                image0,
                image1,
                valid0 & valid1,
                guard,
                safe_wall,
                owner0,
                owner1,
                cuts,
                blend_width,
                multiband_levels,
            )
            if np.any(blend_zone & (risk.mask > 0)):
                raise RuntimeError("RGB risk pixels reached a MultiBand blend zone")
            if split_count or boundary_guard_count:
                raise RuntimeError("Hard-owner protection audit failed")
            (
                l_delta,
                delta_e00,
                _signed_l_delta,
                white_support,
                l_samples,
                delta_e00_samples,
            ) = _white_wall_pair_metrics(image0, image1, safe_wall, cuts)
            if l_delta is not None:
                white_wall_l_samples.append(l_samples)
            if delta_e00 is not None:
                white_wall_delta_e00_samples.append(delta_e00_samples)
            pair_valid = owner0 | owner1
            canvas_view = canvas[:, left:right]
            valid_view = valid_canvas[:, left:right]
            owner_view = owner_canvas[:, left:right]
            canvas_view[pair_valid] = pair_image[pair_valid]
            valid_view[pair_valid] = True
            owner_view[owner0] = index
            owner_view[owner1] = index + 1
            if not used_graphcut or hard_rows:
                hard_cut_pairs += 1
            graphcut_pairs += int(used_graphcut)
            blend_pixels += pair_blend_pixels
            protected_split_components += split_count
            owner_boundary_guard_pixels += boundary_guard_count
            actual_search_widths.append(int(right - left))
            applied_multiband_levels.append(pair_levels)
            photometric = photometric_edges[index]
            pair_metadata.append(
                {
                    "first_frame_id": first.frame_id,
                    "second_frame_id": second.frame_id,
                    "overlap_x": [left, right],
                    "search_corridor_width_pixels": int(right - left),
                    "search_corridor_target_width_pixels": int(
                        settings.seam_search_width_pixels
                    ),
                    "blend_width_pixels": blend_width,
                    "multiband_levels": pair_levels,
                    "blend_zone_pixel_count": pair_blend_pixels,
                    "blend_zone_risk_pixel_count": 0,
                    "risk_pixel_count": int(np.count_nonzero(risk.mask)),
                    "risk_seed_pixel_count": risk.seed_pixel_count,
                    "risk_edge_offset_p95_pixels": risk.edge_offset_p95,
                    "protected_guard_pixel_count": int(np.count_nonzero(guard)),
                    "protected_guard_radius_pixels": guard_radius,
                    "protected_component_count": guard_component_count,
                    "preflight_component_constraint_count": len(
                        sequence_owner_preflight.component_owner_constraints[index]
                    ),
                    "split_protected_component_count": split_count,
                    "owner_boundary_risk_guard_pixel_count": boundary_guard_count,
                    "safe_white_wall_pixel_count": white_support,
                    "safe_white_wall_delta_l_p95": l_delta,
                    "safe_white_wall_delta_e00_p95": delta_e00,
                    "multiband_mask0_pixel_count": mask0_pixels,
                    "multiband_mask1_pixel_count": mask1_pixels,
                    "multiband_masks_complementary": bool(
                        pair_blend_pixels == 0 or masks_distinct
                    ),
                    "photometric_safe_wall_support_pixels": (
                        photometric.support_pixels if photometric is not None else 0
                    ),
                    "photometric_log_mad_bgr": (
                        photometric.mad_bgr.tolist() if photometric is not None else None
                    ),
                    "graphcut_used": used_graphcut,
                    "hard_cut_row_count": hard_rows,
                }
            )

        if protected_split_components or owner_boundary_guard_pixels:
            raise RuntimeError("RGB hard-owner structural protection audit failed")
        white_wall_l_p95 = (
            float(np.percentile(np.concatenate(white_wall_l_samples), 95.0))
            if white_wall_l_samples
            else None
        )
        white_wall_delta_e00_p95 = (
            float(np.percentile(np.concatenate(white_wall_delta_e00_samples), 95.0))
            if white_wall_delta_e00_samples
            else None
        )
        white_wall_measurement_count = int(
            sum(samples.size for samples in white_wall_delta_e00_samples)
        )
        stripe_energy_before = _periodic_stripe_energy(raw_stripe_samples)
        stripe_energy_after = _periodic_stripe_energy(corrected_stripe_samples)
        stripe_reduction_required = (
            stripe_energy_before is not None and stripe_energy_before >= 0.20
        )
        stripe_energy_ratio = (
            stripe_energy_after / stripe_energy_before
            if stripe_energy_before is not None
            and stripe_energy_after is not None
            and stripe_energy_before > 1e-9
            else None
        )

        if np.any(valid_canvas & (owner_canvas < 0)):
            raise RuntimeError("RGB pushbroom valid output contains an unowned pixel")
        if not np.any(valid_canvas):
            raise RuntimeError("Calibrated RGB pushbroom produced no valid RGB pixels")
        crop = largest_valid_rectangle(valid_canvas)
        panorama = canvas[crop.y : crop.y + crop.height, crop.x : crop.x + crop.width]
        cropped_owner = owner_canvas[
            crop.y : crop.y + crop.height, crop.x : crop.x + crop.width
        ]
        source_owner_pixel_counts = [
            int(np.count_nonzero(cropped_owner == source_index))
            for source_index in range(len(frames))
        ]
        missing_sources = [
            int(frames[source_index].frame_id)
            for source_index, pixel_count in enumerate(source_owner_pixel_counts)
            if pixel_count == 0
        ]
        if missing_sources:
            raise RuntimeError(
                "Calibrated RGB pushbroom crop removed all owned pixels from "
                f"source frame(s): {missing_sources}"
            )
        endpoint_outer_owner_pixel_counts: list[int] = []
        endpoint_outer_trimmed_invalid_pixel_counts: list[int] = []
        endpoint_outer_trimmed_column_counts: list[int] = []
        if layout.endpoint_outer_owner_intervals_x:
            for source_index, (left, right) in zip(
                (0, len(frames) - 1),
                layout.endpoint_outer_owner_intervals_x,
                strict=True,
            ):
                x0 = max(crop.x, int(math.ceil(left)))
                x1 = min(crop.x + crop.width, int(math.ceil(right)))
                count = (
                    int(
                        np.count_nonzero(
                            owner_canvas[
                                crop.y : crop.y + crop.height,
                                x0:x1,
                            ]
                            == source_index
                        )
                    )
                    if x1 > x0
                    else 0
                )
                endpoint_outer_owner_pixel_counts.append(count)
                requested_left = max(0, int(math.ceil(left)))
                requested_right = min(layout.canvas_width, int(math.ceil(right)))
                trimmed_spans = (
                    (requested_left, min(requested_right, crop.x)),
                    (max(requested_left, crop.x + crop.width), requested_right),
                )
                trimmed_invalid = 0
                fully_valid_trimmed_columns = 0
                for trim_left, trim_right in trimmed_spans:
                    if trim_right <= trim_left:
                        continue
                    trimmed = valid_canvas[
                        crop.y : crop.y + crop.height,
                        trim_left:trim_right,
                    ]
                    trimmed_invalid += int(trimmed.size - np.count_nonzero(trimmed))
                    fully_valid_trimmed_columns += int(
                        np.count_nonzero(np.all(trimmed, axis=0))
                    )
                if fully_valid_trimmed_columns:
                    raise RuntimeError(
                        "Calibrated RGB pushbroom crop discarded a fully valid "
                        "endpoint outward calibrated field-of-view column"
                    )
                endpoint_outer_trimmed_invalid_pixel_counts.append(trimmed_invalid)
                endpoint_outer_trimmed_column_counts.append(
                    sum(max(0, right - left) for left, right in trimmed_spans)
                )
        if layout.endpoint_outer_owner_intervals_x and any(
            count == 0 for count in endpoint_outer_owner_pixel_counts
        ):
            raise RuntimeError(
                "Calibrated RGB pushbroom crop removed an endpoint outward "
                "calibrated field-of-view contribution"
            )
        crop_height_ratio = crop.height / float(layout.canvas_height)
        crop_width_ratio = crop.width / float(layout.canvas_width)
        blend_fraction = blend_pixels / max(1, int(np.count_nonzero(valid_canvas)))
        quality_metrics: dict[str, float | int | bool | None] = {
            "quality_pass": True,
            "crop_height_ratio": float(crop_height_ratio),
            "crop_width_ratio": float(crop_width_ratio),
            "blend_zone_fraction": float(blend_fraction),
            "blend_zone_risk_fraction": float(maximum_blend_risk),
            "blend_zone_risk_pixel_count": 0,
            "strict_owner_partition": True,
            "graphcut_pair_count": graphcut_pairs,
            "hard_cut_pair_count": hard_cut_pairs,
            "multiband_levels": max(applied_multiband_levels, default=0),
            "multiband_requested_max_levels": int(multiband_levels),
            "search_corridor_target_width_pixels": int(settings.seam_search_width_pixels),
            "search_corridor_width_min_pixels": min(actual_search_widths, default=0),
            "search_corridor_width_max_pixels": max(actual_search_widths, default=0),
            "search_corridor_target_pair_fraction": float(
                sum(width >= 32 for width in actual_search_widths)
                / max(1, len(actual_search_widths))
            ),
            "owner_boundary_risk_guard_pixel_count": owner_boundary_guard_pixels,
            "split_protected_component_count": protected_split_components,
            "safe_white_wall_boundary_l_delta_p95": white_wall_l_p95,
            "safe_white_wall_boundary_delta_e00_p95": white_wall_delta_e00_p95,
            "safe_white_wall_boundary_measurement_count": white_wall_measurement_count,
            "safe_white_wall_boundary_metric_tile_pixels": [64, 4],
            "periodic_stripe_energy_before": stripe_energy_before,
            "periodic_stripe_energy_after": stripe_energy_after,
            "periodic_stripe_energy_ratio": stripe_energy_ratio,
            "periodic_stripe_reduction_required": stripe_reduction_required,
            "periodic_stripe_reduction_target_ratio": 0.60,
            "source_remap_count": renderer.remap_count,
            "analysis_preview_remap_count": analysis_renderer.analysis_preview_remap_count,
            "full_resolution_output_remap_count": renderer.full_resolution_output_remap_count,
            "maximum_resident_strips": 2,
            "all_sources_have_owned_pixels": True,
            "endpoint_outer_half_fov_owner_pixel_counts": endpoint_outer_owner_pixel_counts,
            "endpoint_outer_half_fov_trimmed_invalid_pixel_counts": endpoint_outer_trimmed_invalid_pixel_counts,
            "endpoint_outer_half_fov_trimmed_column_counts": endpoint_outer_trimmed_column_counts,
            "endpoint_outer_half_fov_preserved": bool(
                layout.endpoint_outer_owner_intervals_x
            ),
            **exposure_metrics,
            "exposure_gain_min": float(np.min(gains)),
            "exposure_gain_max": float(np.max(gains)),
            "exposure_gain_min_bgr": [float(value) for value in np.min(gains, axis=0)],
            "exposure_gain_max_bgr": [float(value) for value in np.max(gains, axis=0)],
        }
        failures: list[str] = []
        if crop_height_ratio < 0.85:
            failures.append("less than 85% of levelled calibrated RGB height remains")
        if crop_width_ratio < 0.95:
            failures.append("less than 95% of the calibrated RGB strip canvas remains")
        if blend_fraction > 0.20:
            failures.append("RGB MultiBand zone exceeds 20% of valid output")
        if maximum_blend_risk != 0.0:
            failures.append("RGB risk entered a MultiBand blend zone")
        if white_wall_l_p95 is None or white_wall_l_p95 > 1.5:
            failures.append("safe white-wall owner-boundary P95 |delta L*| exceeds 1.5")
        if white_wall_delta_e00_p95 is None or white_wall_delta_e00_p95 > 2.0:
            failures.append("safe white-wall owner-boundary P95 delta E00 exceeds 2.0")
        if (
            stripe_reduction_required
            and (stripe_energy_ratio is None or stripe_energy_ratio > 0.60)
        ):
            failures.append("periodic vertical stripe energy was not reduced by 40%")
        quality_metrics["quality_pass"] = not failures
        if quality_gate and failures:
            raise RuntimeError("Calibrated RGB pushbroom quality gate failed: " + "; ".join(failures))
        if analysis_renderer.analysis_preview_remap_count != len(frames):
            raise RuntimeError("RGB residual preview did not remap every real source once")
        if renderer.full_resolution_output_remap_count != len(frames):
            raise RuntimeError(
                "Calibrated RGB pushbroom did not perform exactly one full-resolution "
                "output remap per real source"
            )
        residual_metadata = residual_alignment.as_dict()
        residual_metadata.update(
            {
                "backend": settings.residual_alignment.backend,
                "config": settings.residual_alignment.as_dict(),
                "analysis_preview_remap_count": analysis_renderer.analysis_preview_remap_count,
                "preview_remap_count": analysis_renderer.analysis_preview_remap_count,
                "full_resolution_output_remap_count": renderer.full_resolution_output_remap_count,
                "owner_boundary_geometry": preview_boundary_geometry,
            }
        )
        metadata: dict[str, object] = {
            "backend": "calibrated_rgb_pushbroom",
            "pixel_source": "calibrated_rgb_only",
            "depth_used_for_output_pixels": False,
            "point_cloud_constructed": False,
            "tsdf_constructed": False,
            "reference_plane_fitted": False,
            "single_inverse_remap_per_source": True,
            "interpolated_pose_count": 0,
            "layout": layout.as_dict(),
            "rgb_motion_scale": scale.as_dict(),
            "crop": crop.as_dict(),
            "source_count": len(frames),
            "frame_ids": [frame.frame_id for frame in frames],
            "source_owner_pixel_counts": source_owner_pixel_counts,
            "color_gain_channel_order": "RGB",
            "color_gain_application": "single_linear_rgb_pass",
            "color_gains": [
                [float(value) for value in gain[::-1]] for gain in gains
            ],
            "residual_alignment": residual_metadata,
            "pairs": pair_metadata,
            "quality_metrics": quality_metrics,
        }
        return CalibratedRGBPushbroomResult(panorama=panorama, metadata=metadata)
