"""Audited dense SE(3) priors for real-frame video experiments only.

The frozen public video renderer still needs a genuine ORB pose for every
render source.  This module is a candidate-only helper: it never creates a
frame, colour, owner, global ORB trajectory entry, or production pose.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import cv2
import numpy as np

from .session import CameraIntrinsics, RGBDFrame


class DensePosePriorError(ValueError):
    """Dense experimental prior evidence is incomplete or unsafe."""


POSE_ORIGINS = frozenset(
    {"direct_orb_anchor", "interpolated_se3_prior", "refined_dense_prior"}
)


@dataclass(frozen=True)
class DensePosePriorConfig:
    """Closed fail-closed limits for bracketed experimental priors."""

    maximum_anchor_distance_us: int = 150_000
    maximum_forward_backward_p95_pixels: float = 1.5
    maximum_rgbd_residual_p95_pixels: float = 1.5
    maximum_refinement_translation_mm: float = 5.0
    maximum_refinement_rotation_degrees: float = 1.0
    minimum_audit_samples: int = 32

    def __post_init__(self) -> None:
        if (
            isinstance(self.maximum_anchor_distance_us, bool)
            or not isinstance(self.maximum_anchor_distance_us, int)
            or self.maximum_anchor_distance_us <= 0
        ):
            raise DensePosePriorError("maximum_anchor_distance_us must be positive")
        values = (
            self.maximum_forward_backward_p95_pixels,
            self.maximum_rgbd_residual_p95_pixels,
            self.maximum_refinement_translation_mm,
            self.maximum_refinement_rotation_degrees,
        )
        if not np.isfinite(values).all() or any(value <= 0.0 for value in values):
            raise DensePosePriorError("dense pose-prior limits must be finite and positive")
        if (
            isinstance(self.minimum_audit_samples, bool)
            or not isinstance(self.minimum_audit_samples, int)
            or self.minimum_audit_samples <= 0
        ):
            raise DensePosePriorError("minimum_audit_samples must be positive")


def _frame_identity(frame_id: int, timestamp_us: int, label: str) -> None:
    for name, value in (("frame_id", frame_id), ("timestamp_us", timestamp_us)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise DensePosePriorError(f"{label} {name} must be a non-negative integer")


def _pose(value: object, label: str) -> np.ndarray:
    pose = np.asarray(value, dtype=np.float64)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise DensePosePriorError(f"{label} camera_to_world must be a finite 4x4 matrix")
    if not np.allclose(pose[3], (0.0, 0.0, 0.0, 1.0), atol=1e-8):
        raise DensePosePriorError(f"{label} camera_to_world must be homogeneous SE(3)")
    rotation = pose[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or np.linalg.det(rotation) <= 0:
        raise DensePosePriorError(f"{label} camera_to_world must be a proper rigid pose")
    return np.ascontiguousarray(pose)


@dataclass(frozen=True)
class ORBPoseAnchor:
    """An immutable direct ORB-SLAM3 camera_to_world anchor."""

    frame_id: int
    timestamp_us: int
    camera_to_world: np.ndarray

    def __post_init__(self) -> None:
        _frame_identity(self.frame_id, self.timestamp_us, "ORB anchor")
        object.__setattr__(self, "camera_to_world", _pose(self.camera_to_world, "ORB anchor"))


@dataclass(frozen=True)
class DenseFrameAudit:
    """Dense image-motion and RGB-D residual evidence for one intermediate."""

    frame_id: int
    left_anchor_frame_id: int
    right_anchor_frame_id: int
    forward_backward_p95_pixels: float
    rgbd_residual_p95_pixels: float
    forward_backward_sample_count: int
    rgbd_residual_sample_count: int

    def __post_init__(self) -> None:
        for name in ("frame_id", "left_anchor_frame_id", "right_anchor_frame_id"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise DensePosePriorError(f"{name} must be a non-negative integer")
        if self.left_anchor_frame_id >= self.right_anchor_frame_id:
            raise DensePosePriorError("dense audit anchors must be chronological")
        values = (self.forward_backward_p95_pixels, self.rgbd_residual_p95_pixels)
        if not np.isfinite(values).all() or any(value < 0 for value in values):
            raise DensePosePriorError("dense audit residuals must be finite and non-negative")
        for name in ("forward_backward_sample_count", "rgbd_residual_sample_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise DensePosePriorError(f"{name} must be a non-negative integer")

    def accepted(self, config: DensePosePriorConfig) -> bool:
        return (
            self.forward_backward_sample_count >= config.minimum_audit_samples
            and self.rgbd_residual_sample_count >= config.minimum_audit_samples
            and self.forward_backward_p95_pixels <= config.maximum_forward_backward_p95_pixels
            and self.rgbd_residual_p95_pixels <= config.maximum_rgbd_residual_p95_pixels
        )

    def as_dict(self, config: DensePosePriorConfig) -> dict[str, object]:
        return {
            "frame_id": self.frame_id,
            "left_anchor_frame_id": self.left_anchor_frame_id,
            "right_anchor_frame_id": self.right_anchor_frame_id,
            "forward_backward_p95_pixels": self.forward_backward_p95_pixels,
            "rgbd_residual_p95_pixels": self.rgbd_residual_p95_pixels,
            "forward_backward_sample_count": self.forward_backward_sample_count,
            "rgbd_residual_sample_count": self.rgbd_residual_sample_count,
            "accepted": self.accepted(config),
        }


@dataclass(frozen=True)
class DensePosePrior:
    """Pose prior for one actual RGB-D frame plus complete origin/audit."""

    frame_id: int
    timestamp_us: int
    camera_to_world: np.ndarray
    source_pose_origin: str
    audit: dict[str, object]

    def __post_init__(self) -> None:
        _frame_identity(self.frame_id, self.timestamp_us, "dense pose prior")
        if self.source_pose_origin not in POSE_ORIGINS:
            raise DensePosePriorError("unsupported source_pose_origin")
        object.__setattr__(self, "camera_to_world", _pose(self.camera_to_world, "dense pose prior"))
        if not isinstance(self.audit, dict):
            raise DensePosePriorError("dense pose-prior audit must be an object")

    def as_dict(self) -> dict[str, object]:
        return {
            "frame_id": self.frame_id,
            "timestamp_us": self.timestamp_us,
            "camera_to_world": self.camera_to_world.tolist(),
            "source_pose_origin": self.source_pose_origin,
            "audit": dict(self.audit),
        }


def _quaternion(rotation: np.ndarray) -> np.ndarray:
    trace = float(np.trace(rotation))
    if trace > 0:
        scale = math.sqrt(trace + 1) * 2
        result = np.array((0.25 * scale, (rotation[2, 1] - rotation[1, 2]) / scale,
                           (rotation[0, 2] - rotation[2, 0]) / scale, (rotation[1, 0] - rotation[0, 1]) / scale))
    else:
        axis = int(np.argmax(np.diag(rotation)))
        if axis == 0:
            scale = math.sqrt(1 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2
            result = np.array(((rotation[2, 1] - rotation[1, 2]) / scale, 0.25 * scale,
                               (rotation[0, 1] + rotation[1, 0]) / scale, (rotation[0, 2] + rotation[2, 0]) / scale))
        elif axis == 1:
            scale = math.sqrt(1 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2
            result = np.array(((rotation[0, 2] - rotation[2, 0]) / scale, (rotation[0, 1] + rotation[1, 0]) / scale,
                               0.25 * scale, (rotation[1, 2] + rotation[2, 1]) / scale))
        else:
            scale = math.sqrt(1 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2
            result = np.array(((rotation[1, 0] - rotation[0, 1]) / scale, (rotation[0, 2] + rotation[2, 0]) / scale,
                               (rotation[1, 2] + rotation[2, 1]) / scale, 0.25 * scale))
    result /= np.linalg.norm(result)
    return result if result[0] >= 0 else -result


def _rotation(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = quat / np.linalg.norm(quat)
    return np.array(
        ((1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
         (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
         (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y))),
        dtype=np.float64,
    )


def interpolate_bracketed_se3(left: ORBPoseAnchor, right: ORBPoseAnchor, timestamp_us: int) -> np.ndarray:
    """SLERP/linear translation only inside a true temporal ORB bracket."""

    if not left.timestamp_us <= timestamp_us <= right.timestamp_us:
        raise DensePosePriorError("SE(3) interpolation cannot extrapolate beyond ORB anchors")
    if right.timestamp_us <= left.timestamp_us:
        raise DensePosePriorError("ORB bracket timestamps must increase")
    alpha = (timestamp_us - left.timestamp_us) / (right.timestamp_us - left.timestamp_us)
    first, second = _quaternion(left.camera_to_world[:3, :3]), _quaternion(right.camera_to_world[:3, :3])
    dot = float(np.dot(first, second))
    if dot < 0:
        second, dot = -second, -dot
    if dot > 0.9995:
        blended = first + alpha * (second - first)
    else:
        angle = math.acos(float(np.clip(dot, -1, 1)))
        blended = (math.sin((1 - alpha) * angle) * first + math.sin(alpha * angle) * second) / math.sin(angle)
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = _rotation(blended)
    result[:3, 3] = (1 - alpha) * left.camera_to_world[:3, 3] + alpha * right.camera_to_world[:3, 3]
    return result


def _refinement_metrics(delta: object, config: DensePosePriorConfig) -> tuple[np.ndarray, float, float]:
    refinement = _pose(delta, "dense refinement")
    translation = float(np.linalg.norm(refinement[:3, 3]))
    degrees = math.degrees(math.acos(float(np.clip((np.trace(refinement[:3, :3]) - 1) / 2, -1, 1))))
    if translation > config.maximum_refinement_translation_mm or degrees > config.maximum_refinement_rotation_degrees:
        raise DensePosePriorError("dense refinement exceeds its fixed SE(3) bounds")
    return refinement, translation, degrees


def build_dense_real_frame_pose_priors(
    frames: Sequence[RGBDFrame],
    orb_anchors: Sequence[ORBPoseAnchor],
    *,
    audits_by_frame_id: Mapping[int, DenseFrameAudit] | None = None,
    refinements_by_frame_id: Mapping[int, np.ndarray] | None = None,
    config: DensePosePriorConfig | None = None,
) -> tuple[DensePosePrior, ...]:
    """Produce only direct anchors or fully-audited bracketed real-frame priors."""

    settings = config or DensePosePriorConfig()
    real_frames, anchors = tuple(frames), tuple(orb_anchors)
    if not real_frames or not anchors:
        raise DensePosePriorError("dense priors require real frames and ORB anchors")
    if len({item.frame_id for item in real_frames}) != len(real_frames) or any(item.timestamp_us is None for item in real_frames):
        raise DensePosePriorError("real RGB-D frames need unique ids and timestamps")
    if any(b.timestamp_us <= a.timestamp_us for a, b in zip(real_frames, real_frames[1:])):
        raise DensePosePriorError("real RGB-D frames must be chronological")
    if len({item.frame_id for item in anchors}) != len(anchors) or any(
        b.timestamp_us <= a.timestamp_us for a, b in zip(anchors, anchors[1:])
    ):
        raise DensePosePriorError("ORB anchors must be unique and chronological")
    audits, refinements = dict(audits_by_frame_id or {}), dict(refinements_by_frame_id or {})
    if (set(audits) | set(refinements)) - {item.frame_id for item in real_frames}:
        raise DensePosePriorError("dense evidence references a non-real frame")
    by_id = {item.frame_id: item for item in anchors}
    result: list[DensePosePrior] = []
    for frame in real_frames:
        timestamp = int(frame.timestamp_us)
        direct = by_id.get(frame.frame_id)
        if direct is not None:
            if direct.timestamp_us != timestamp or frame.frame_id in audits or frame.frame_id in refinements:
                raise DensePosePriorError("a direct ORB anchor cannot be altered")
            result.append(DensePosePrior(frame.frame_id, timestamp, direct.camera_to_world, "direct_orb_anchor", {
                "schema": "gemini305-video-dense-real-frame-pose-prior/v1",
                "direct_orb_anchor": True, "no_extrapolation": True,
            }))
            continue
        bracket = next(
            (
                (a, b)
                for a, b in zip(anchors, anchors[1:])
                if a.timestamp_us < timestamp < b.timestamp_us
            ),
            None,
        )
        if bracket is None:
            raise DensePosePriorError("real intermediate has no enclosing ORB anchor bracket")
        left, right = bracket
        left_us, right_us = timestamp - left.timestamp_us, right.timestamp_us - timestamp
        if max(left_us, right_us) > settings.maximum_anchor_distance_us:
            raise DensePosePriorError("real intermediate exceeds the 150 ms anchor gate")
        audit = audits.get(frame.frame_id)
        if audit is None or (audit.left_anchor_frame_id, audit.right_anchor_frame_id) != (left.frame_id, right.frame_id) or not audit.accepted(settings):
            raise DensePosePriorError("real intermediate failed required dense image/RGB-D audit")
        pose, origin = interpolate_bracketed_se3(left, right, timestamp), "interpolated_se3_prior"
        evidence: dict[str, object] = {
            "schema": "gemini305-video-dense-real-frame-pose-prior/v1",
            "left_anchor_frame_id": left.frame_id, "right_anchor_frame_id": right.frame_id,
            "left_anchor_distance_us": left_us, "right_anchor_distance_us": right_us,
            "maximum_anchor_distance_us": settings.maximum_anchor_distance_us,
            "no_extrapolation": True, "dense_evidence": audit.as_dict(settings),
        }
        if frame.frame_id in refinements:
            delta, translation, degrees = _refinement_metrics(refinements[frame.frame_id], settings)
            pose, origin = _pose(delta @ pose, "refined dense prior"), "refined_dense_prior"
            evidence["bounded_refinement"] = {
                "translation_mm": translation, "rotation_degrees": degrees,
                "maximum_translation_mm": settings.maximum_refinement_translation_mm,
                "maximum_rotation_degrees": settings.maximum_refinement_rotation_degrees,
            }
        result.append(DensePosePrior(frame.frame_id, timestamp, pose, origin, evidence))
    return tuple(result)


def summarize_dense_frame_audit(
    *, frame_id: int, left_anchor_frame_id: int, right_anchor_frame_id: int,
    forward_backward_errors_pixels: np.ndarray, rgbd_residuals_pixels: np.ndarray,
) -> DenseFrameAudit:
    """Convert actual dense residual samples into immutable auditable p95s."""

    def clean(values: np.ndarray, label: str) -> np.ndarray:
        output = np.asarray(values, dtype=np.float64).reshape(-1)
        output = output[np.isfinite(output) & (output >= 0)]
        if not output.size:
            raise DensePosePriorError(f"{label} has no finite non-negative samples")
        return output
    fb, residual = clean(forward_backward_errors_pixels, "FB error"), clean(rgbd_residuals_pixels, "RGB-D residual")
    return DenseFrameAudit(frame_id, left_anchor_frame_id, right_anchor_frame_id,
                           float(np.percentile(fb, 95)), float(np.percentile(residual, 95)),
                           int(fb.size), int(residual.size))


def audit_dense_image_motion_and_rgbd_residual(
    source_bgr: np.ndarray, target_bgr: np.ndarray, source_depth_mm: np.ndarray,
    source_camera_to_world: np.ndarray, target_camera_to_world: np.ndarray,
    calibration: CameraIntrinsics,
) -> tuple[np.ndarray, np.ndarray]:
    """Measure real dense DIS FB and source-depth SE(3) flow residual samples."""

    source, target, depth = np.asarray(source_bgr), np.asarray(target_bgr), np.asarray(source_depth_mm, dtype=np.float64)
    if source.ndim != 3 or source.shape[2] != 3 or target.shape != source.shape or depth.shape != source.shape[:2]:
        raise DensePosePriorError("dense audit needs same-sized BGR images and aligned source depth")
    if source.shape[:2] != (calibration.height, calibration.width) or not np.isfinite(depth).all():
        raise DensePosePriorError("dense audit image/depth dimensions or values are invalid")
    source_pose, target_pose = _pose(source_camera_to_world, "source"), _pose(target_camera_to_world, "target")
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
    forward = dis.calc(cv2.cvtColor(source, cv2.COLOR_BGR2GRAY), cv2.cvtColor(target, cv2.COLOR_BGR2GRAY), None)
    backward = dis.calc(cv2.cvtColor(target, cv2.COLOR_BGR2GRAY), cv2.cvtColor(source, cv2.COLOR_BGR2GRAY), None)
    height, width = depth.shape
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    to_x, to_y = xx + forward[..., 0], yy + forward[..., 1]
    back_at_forward = cv2.remap(backward, to_x, to_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
    inside = (to_x >= 0) & (to_x < width - 1) & (to_y >= 0) & (to_y < height - 1)
    fb = np.linalg.norm(forward + back_at_forward, axis=2)
    transform = np.linalg.inv(target_pose) @ source_pose
    x, y = (xx - calibration.cx) * depth / calibration.fx, (yy - calibration.cy) * depth / calibration.fy
    points = np.stack((x, y, depth, np.ones_like(depth)), axis=-1) @ transform.T
    z = points[..., 2]
    u, v = calibration.fx * points[..., 0] / np.maximum(z, 1e-6) + calibration.cx, calibration.fy * points[..., 1] / np.maximum(z, 1e-6) + calibration.cy
    expected = np.stack((u - xx, v - yy), axis=-1)
    residual = np.linalg.norm(forward - expected, axis=2)
    valid = (depth > 0) & (z > 0) & (u >= 0) & (u < width - 1) & (v >= 0) & (v < height - 1)
    return np.ascontiguousarray(fb[inside & np.isfinite(fb)]), np.ascontiguousarray(residual[valid & np.isfinite(residual)])


__all__ = [
    "DenseFrameAudit", "DensePosePrior", "DensePosePriorConfig", "DensePosePriorError",
    "ORBPoseAnchor", "POSE_ORIGINS", "audit_dense_image_motion_and_rgbd_residual",
    "build_dense_real_frame_pose_priors", "interpolate_bracketed_se3", "summarize_dense_frame_audit",
]
