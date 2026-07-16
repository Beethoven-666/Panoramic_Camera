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
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import cv2
import numpy as np

from .calibrated_remap import camera_points_to_source_pixels
from .render import compute_rgb_disparity_risk_mask, largest_valid_rectangle
from .session import CameraIntrinsics, RGBDFrame


_HARD_MAX_POSES = 160
_HARD_MAX_CANVAS_MEGAPIXELS = 200.0
_HARD_MAX_RESIDENT_STRIPS = 5


@dataclass(frozen=True)
class CalibratedRGBPushbroomConfig:
    """Closed safety/configuration surface for the RGB pushbroom renderer."""

    maximum_central_band_fraction: float = 0.20
    seam_half_width_pixels: int = 4
    max_canvas_megapixels: float = 200.0
    max_aggregate_megapixels: float = 200.0
    max_pose_count: int = _HARD_MAX_POSES
    max_resident_frames: int = _HARD_MAX_RESIDENT_STRIPS
    minimum_valid_scale_pairs: int = 3
    scale_central_fraction: float = 0.20
    scale_low_gradient_quantile: float = 0.45
    scale_minimum_response: float = 0.10
    scale_max_relative_mad: float = 0.35

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, object] | None
    ) -> "CalibratedRGBPushbroomConfig":
        supplied = {} if value is None else dict(value)
        supplied.pop("mode", None)
        allowed = {
            "maximum_central_band_fraction",
            "seam_half_width_pixels",
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
            result = cls(**supplied)
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
        if not 0 <= int(self.seam_half_width_pixels) <= 8:
            raise ValueError("seam_half_width_pixels must be in [0, 8]")
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
    # Each raw source contributes only a genuinely narrow central band.  At
    # low frame counts an owner interval itself can approach the configured
    # band limit, so reduce only the overlap guard; never widen the calibrated
    # remap beyond that limit.  If the hard-owned interval cannot fit, fail
    # closed and ask for denser real pose sampling or a lower output scale.
    maximum_band_width = max(
        2, int(math.floor(calibration.width * config.maximum_central_band_fraction))
    )
    owner_left_int = np.ceil(owner_edges[:-1]).astype(np.int32)
    owner_right_int = np.ceil(owner_edges[1:]).astype(np.int32)
    owner_widths = owner_right_int - owner_left_int
    if np.any(owner_widths > maximum_band_width):
        raise RuntimeError(
            "Calibrated RGB pushbroom owner strip exceeds the configured narrow "
            "central-band limit; capture more real pose frames or lower output scale"
        )
    spare_width = maximum_band_width - owner_widths
    left_padding = np.minimum(
        int(config.seam_half_width_pixels), spare_width // 2
    )
    right_padding = np.minimum(
        int(config.seam_half_width_pixels), spare_width - left_padding
    )
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


class CalibratedRGBPushbroomRenderer:
    """Single-remap calibrated RGB strip generator; it has no depth inputs."""

    def __init__(
        self,
        layout: PushbroomLayout,
        calibration: CameraIntrinsics,
        poses: Sequence[np.ndarray],
    ) -> None:
        if len(poses) != len(layout.frame_ids):
            raise ValueError("Pushbroom renderer poses must match layout sources")
        self.layout = layout
        self.calibration = calibration
        self.poses = tuple(validate_camera_to_world(pose) for pose in poses)
        self.remap_count = 0

    def render_frame(self, frame: RGBDFrame, source_index: int) -> PushbroomContribution:
        if not 0 <= source_index < len(self.poses):
            raise IndexError("Pushbroom source index is out of range")
        if int(frame.frame_id) != self.layout.frame_ids[source_index]:
            raise ValueError("Pushbroom frame order must match real-pose layout order")
        image = _read_bgr(frame.color_path, self.calibration)
        x0 = self.layout.support_left_x[source_index]
        x1 = self.layout.support_right_x[source_index]
        width = x1 - x0
        height = self.layout.canvas_height
        global_x = np.arange(x0, x1, dtype=np.float64)
        virtual_u = self.calibration.cx + self.layout.temporal_to_virtual_x_sign * (
            global_x - self.layout.source_centres_x[source_index]
        )
        virtual_v = np.arange(height, dtype=np.float64)
        ray_x = (virtual_u[None, :] - self.calibration.cx) / self.calibration.fx
        ray_y = (virtual_v[:, None] - self.calibration.cy) / self.calibration.fy
        rays = np.empty((height, width, 3), dtype=np.float64)
        rays[:, :, 0] = ray_x
        rays[:, :, 1] = ray_y
        rays[:, :, 2] = 1.0
        world_rays = rays @ self.layout.level_camera_to_world_rotation.T
        source_rays = world_rays @ self.poses[source_index][:3, :3]
        map_x, map_y, positive_z = camera_points_to_source_pixels(
            source_rays, self.calibration
        )
        valid = (
            positive_z
            & np.isfinite(map_x)
            & np.isfinite(map_y)
            & (map_x >= 0.0)
            & (map_x <= self.calibration.width - 1)
            & (map_y >= 0.0)
            & (map_y <= self.calibration.height - 1)
        )
        rgb = cv2.remap(
            image,
            map_x,
            map_y,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        self.remap_count += 1
        return PushbroomContribution(
            source_index=source_index,
            frame_id=int(frame.frame_id),
            x0=x0,
            rgb=np.ascontiguousarray(rgb),
            valid_mask=np.ascontiguousarray(valid),
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


def _gradient_magnitude(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.magnitude(
        cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3),
    )


def _safe_wall_log_gain(
    first: np.ndarray,
    second: np.ndarray,
    common: np.ndarray,
    risk: np.ndarray,
    *,
    low_gradient_quantile: float,
) -> tuple[float | None, int]:
    """Trimmed/Huber local brightness estimate that excludes foreground details."""

    usable = np.asarray(common, dtype=bool) & ~(np.asarray(risk) > 0)
    if int(np.count_nonzero(usable)) < 16:
        return None, 0
    lab0 = cv2.cvtColor(first, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab1 = cv2.cvtColor(second, cv2.COLOR_BGR2LAB).astype(np.float32)
    grad0, grad1 = _gradient_magnitude(first), _gradient_magnitude(second)
    gradient_values = np.concatenate((grad0[usable], grad1[usable]))
    if not gradient_values.size:
        return None, 0
    gradient_limit = float(
        np.clip(
            np.percentile(gradient_values, 100.0 * low_gradient_quantile),
            5.0,
            24.0,
        )
    )
    unclipped = (
        (lab0[:, :, 0] >= 12.0)
        & (lab0[:, :, 0] <= 243.0)
        & (lab1[:, :, 0] >= 12.0)
        & (lab1[:, :, 0] <= 243.0)
        & (np.max(first, axis=2) < 253)
        & (np.max(second, axis=2) < 253)
    )
    candidates = usable & unclipped & (grad0 <= gradient_limit) & (grad1 <= gradient_limit)
    minimum = max(16, min(256, int(math.ceil(np.count_nonzero(usable) * 0.02))))
    if int(np.count_nonzero(candidates)) < minimum:
        return None, 0
    ratios = np.log((lab0[:, :, 0] + 1.0) / (lab1[:, :, 0] + 1.0))
    values = ratios[candidates]
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    keep = candidates & (np.abs(ratios - median) <= max(0.015, 3.0 * 1.4826 * mad))
    if int(np.count_nonzero(keep)) < minimum:
        return None, 0
    values = ratios[keep]
    centre = float(np.median(values))
    scale = max(0.005, 1.4826 * float(np.median(np.abs(values - centre))))
    residual = np.abs(values - centre)
    weights = np.minimum(1.0, 1.5 * scale / np.maximum(residual, 1e-9))
    estimate = float(np.sum(weights * values) / np.sum(weights))
    return (estimate if np.isfinite(estimate) else None), int(np.count_nonzero(keep))


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


def _solve_smooth_gains(
    source_count: int, edges: Sequence[tuple[float | None, int]]
) -> tuple[np.ndarray, dict[str, int | float]]:
    """Solve a lightly smoothed 1-D gain curve; this never blurs image pixels."""

    log_gains = np.zeros(source_count, dtype=np.float64)
    support = 0
    valid_edges = 0
    for index, (relation, pixels) in enumerate(edges):
        if relation is None:
            log_gains[index + 1] = log_gains[index]
            continue
        log_gains[index + 1] = log_gains[index] + relation
        valid_edges += 1
        support += int(pixels)
    if source_count >= 3:
        # Smooth exposure *parameters* only.  This is intentionally not a
        # Gaussian or any other spatial blur of the output panorama.
        padded = np.pad(log_gains, (1, 1), mode="edge")
        log_gains = 0.25 * padded[:-2] + 0.50 * padded[1:-1] + 0.25 * padded[2:]
    log_gains -= float(np.median(log_gains))
    gains = np.exp(np.clip(log_gains, math.log(0.45), math.log(2.20)))
    return gains, {
        "safe_exposure_pair_count": valid_edges,
        "safe_exposure_support_pixel_count": support,
        "safe_exposure_pair_fraction": valid_edges / max(1, source_count - 1),
    }


def _apply_gain(image: np.ndarray, gain: float) -> np.ndarray:
    return np.clip(image.astype(np.float32) * gain, 0.0, 255.0).astype(np.uint8)


def _graphcut_monotonic_owner(
    first: np.ndarray,
    second: np.ndarray,
    valid0: np.ndarray,
    valid1: np.ndarray,
    protected: np.ndarray,
    nominal_boundary: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool, int]:
    """Run GraphCut, then reduce it to one monotonic hard owner per image row."""

    height, width = valid0.shape
    common = valid0 & valid1
    x = np.arange(width, dtype=np.int32)[None, :]
    masks = [valid0.astype(np.uint8) * 255, valid1.astype(np.uint8) * 255]
    left_protected = protected & (x <= nominal_boundary)
    right_protected = protected & ~left_protected
    masks[1][left_protected] = 0
    masks[0][right_protected] = 0
    graphcut_used = False
    graphcut_first = masks[0] > 0
    try:
        finder = cv2.detail_GraphCutSeamFinder("COST_COLOR_GRAD")
        output = finder.find(
            [
                np.ascontiguousarray(first, dtype=np.float32),
                np.ascontiguousarray(second, dtype=np.float32),
            ],
            [(0, 0), (0, 0)],
            [np.ascontiguousarray(masks[0]), np.ascontiguousarray(masks[1])],
        )
        if output is not None and len(output) == 2:
            candidate = output[0].get() if hasattr(output[0], "get") else output[0]
            candidate = np.asarray(candidate, dtype=np.uint8)
            if candidate.shape == valid0.shape:
                graphcut_first = candidate > 0
                graphcut_used = True
    except cv2.error:
        # A hard cut is explicitly permitted when no safe GraphCut channel
        # exists.  Never feather or average to conceal the failure.
        graphcut_used = False

    owner0 = np.zeros_like(valid0, dtype=bool)
    cuts = np.full(height, int(np.clip(nominal_boundary, 0, width - 1)), dtype=np.int32)
    hard_cut_rows = 0
    for row in range(height):
        common_indices = np.flatnonzero(common[row])
        if not common_indices.size:
            owner0[row] = valid0[row]
            continue
        proposed = np.flatnonzero(common[row] & graphcut_first[row])
        cut = int(proposed.max()) if proposed.size else int(nominal_boundary)
        protected_left = np.flatnonzero(protected[row] & (x[0] <= nominal_boundary))
        protected_right = np.flatnonzero(protected[row] & (x[0] > nominal_boundary))
        lower = int(protected_left.max()) if protected_left.size else 0
        upper = int(protected_right.min() - 1) if protected_right.size else width - 1
        if lower > upper:
            hard_cut_rows += 1
            cut = int(np.clip(nominal_boundary, 0, width - 1))
        else:
            cut = int(np.clip(cut, lower, upper))
        cuts[row] = cut
        owner0[row] = valid0[row] & ((x[0] <= cut) | ~valid1[row])
    owner1 = valid1 & ~owner0
    union = valid0 | valid1
    if np.any(union & ~(owner0 | owner1)) or np.any(owner0 & owner1):
        raise RuntimeError("RGB hard owner seam left an invalid or multiply-owned pixel")
    return owner0, owner1, cuts, graphcut_used, hard_cut_rows


def _blend_safe_pair_zone(
    first: np.ndarray,
    second: np.ndarray,
    common: np.ndarray,
    protected: np.ndarray,
    owner0: np.ndarray,
    owner1: np.ndarray,
    cuts: np.ndarray,
    blend_width: int,
    levels: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Use one local MultiBand blender only where the RGB-risk guard is clear."""

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
        & (x >= cuts[:, None] - left_span)
        & (x <= cuts[:, None] + right_span)
    )
    if not np.any(zone):
        return direct, zone, 0
    # Keep a small safe-only support around the actual zone.  In particular,
    # foreground/risk pixels are not fed into the local pyramid at all.
    support_radius = max(1, int(blend_width))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (support_radius * 2 + 1, support_radius * 2 + 1)
    )
    support = cv2.dilate(zone.astype(np.uint8), kernel) > 0
    support &= common & ~protected
    mask0 = np.where(support, 255, 0).astype(np.uint8)
    mask1 = mask0.copy()
    blender = cv2.detail_MultiBandBlender()
    blender.setNumBands(min(3, max(1, int(levels))))
    blender.prepare((0, 0, first.shape[1], first.shape[0]))
    try:
        blender.feed(np.ascontiguousarray(first, dtype=np.int16), mask0, (0, 0))
        blender.feed(np.ascontiguousarray(second, dtype=np.int16), mask1, (0, 0))
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
    return direct, zone, int(np.count_nonzero(zone))


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
    renderer = CalibratedRGBPushbroomRenderer(layout, calibration, checked_poses)
    strip_paths: list[Path] = []
    with tempfile.TemporaryDirectory(prefix="g305-rgb-pushbroom-") as temporary:
        root = Path(temporary)
        for index, frame in enumerate(frames):
            contribution = renderer.render_frame(frame, index)
            path = root / f"{index:04d}.npz"
            _save_contribution(path, contribution)
            strip_paths.append(path)

        gain_edges: list[tuple[float | None, int]] = []
        for first_path, second_path in zip(strip_paths[:-1], strip_paths[1:], strict=True):
            first, second = _load_contribution(first_path), _load_contribution(second_path)
            _, _, rgb0, rgb1, valid0, valid1 = _pair_overlap(first, second)
            risk = compute_rgb_disparity_risk_mask(
                rgb0, rgb1, valid0.astype(np.uint8) * 255, valid1.astype(np.uint8) * 255
            )
            relation, support = _safe_wall_log_gain(
                rgb0,
                rgb1,
                valid0 & valid1,
                risk,
                low_gradient_quantile=settings.scale_low_gradient_quantile,
            )
            gain_edges.append((relation, support))
        gains, exposure_metrics = _solve_smooth_gains(len(frames), gain_edges)

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
                _apply_gain(contribution.rgb, float(gains[index])),
                layout.owner_left_x[index],
                layout.owner_right_x[index],
            )

        blend_pixels = 0
        maximum_blend_risk = 0.0
        hard_cut_pairs = 0
        graphcut_pairs = 0
        pair_metadata: list[dict[str, object]] = []
        for index, (first_path, second_path) in enumerate(
            zip(strip_paths[:-1], strip_paths[1:], strict=True)
        ):
            first, second = _load_contribution(first_path), _load_contribution(second_path)
            left, right, raw0, raw1, valid0, valid1 = _pair_overlap(first, second)
            image0 = _apply_gain(raw0, float(gains[index]))
            image1 = _apply_gain(raw1, float(gains[index + 1]))
            # Re-evaluate residual risk after the safe-wall gain correction so
            # gain differences cannot make a white wall look like parallax.
            raw_risk = compute_rgb_disparity_risk_mask(
                raw0,
                raw1,
                valid0.astype(np.uint8) * 255,
                valid1.astype(np.uint8) * 255,
            )
            risk = compute_rgb_disparity_risk_mask(
                image0,
                image1,
                valid0.astype(np.uint8) * 255,
                valid1.astype(np.uint8) * 255,
                supplied_risk_mask=raw_risk,
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
            guard = cv2.dilate(
                (risk > 0).astype(np.uint8),
                cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (blend_width * 2 + 1, blend_width * 2 + 1)
                ),
            ) > 0
            nominal = int(round(layout.owner_boundaries_x[index])) - left
            owner0, owner1, cuts, used_graphcut, hard_rows = _graphcut_monotonic_owner(
                image0, image1, valid0, valid1, guard, nominal
            )
            pair_image, blend_zone, pair_blend_pixels = _blend_safe_pair_zone(
                image0,
                image1,
                valid0 & valid1,
                guard,
                owner0,
                owner1,
                cuts,
                blend_width,
                multiband_levels,
            )
            if np.any(blend_zone & (risk > 0)):
                raise RuntimeError("RGB risk pixels reached a MultiBand blend zone")
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
            pair_metadata.append(
                {
                    "first_frame_id": first.frame_id,
                    "second_frame_id": second.frame_id,
                    "overlap_x": [left, right],
                    "blend_width_pixels": blend_width,
                    "blend_zone_pixel_count": pair_blend_pixels,
                    "blend_zone_risk_pixel_count": 0,
                    "risk_pixel_count": int(np.count_nonzero(risk)),
                    "protected_guard_pixel_count": int(np.count_nonzero(guard)),
                    "graphcut_used": used_graphcut,
                    "hard_cut_row_count": hard_rows,
                }
            )

        if np.any(valid_canvas & (owner_canvas < 0)):
            raise RuntimeError("RGB pushbroom valid output contains an unowned pixel")
        if not np.any(valid_canvas):
            raise RuntimeError("Calibrated RGB pushbroom produced no valid RGB pixels")
        crop = largest_valid_rectangle(valid_canvas)
        panorama = canvas[crop.y : crop.y + crop.height, crop.x : crop.x + crop.width]
        crop_height_ratio = crop.height / float(layout.canvas_height)
        crop_width_ratio = crop.width / float(layout.canvas_width)
        blend_fraction = blend_pixels / max(1, int(np.count_nonzero(valid_canvas)))
        quality_metrics: dict[str, float | int | bool] = {
            "quality_pass": True,
            "crop_height_ratio": float(crop_height_ratio),
            "crop_width_ratio": float(crop_width_ratio),
            "blend_zone_fraction": float(blend_fraction),
            "blend_zone_risk_fraction": float(maximum_blend_risk),
            "blend_zone_risk_pixel_count": 0,
            "strict_owner_partition": True,
            "graphcut_pair_count": graphcut_pairs,
            "hard_cut_pair_count": hard_cut_pairs,
            "multiband_levels": int(multiband_levels),
            "source_remap_count": renderer.remap_count,
            "maximum_resident_strips": 2,
            **exposure_metrics,
            "exposure_gain_min": float(np.min(gains)),
            "exposure_gain_max": float(np.max(gains)),
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
        quality_metrics["quality_pass"] = not failures
        if quality_gate and failures:
            raise RuntimeError("Calibrated RGB pushbroom quality gate failed: " + "; ".join(failures))
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
            "color_gains": [[float(gain)] * 3 for gain in gains],
            "pairs": pair_metadata,
            "quality_metrics": quality_metrics,
        }
        return CalibratedRGBPushbroomResult(panorama=panorama, metadata=metadata)
