"""Robust pose-axis recovery and longest valid S1.2 scan selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .video_s12_config import S12ScanConfig
from .video_s12_pose import S12PoseQualification


@dataclass(frozen=True)
class S12ScanPoseAudit:
    frame_id: int
    progress_m: float
    cross_track_error_m: float
    direct_pose: bool
    pose_qualified: bool
    image_motion_direction_agreement: bool | None
    accepted: bool
    rejection_reasons: tuple[str, ...]


@dataclass(frozen=True)
class S12ScanSegment:
    start_frame_id: int
    end_frame_id: int
    candidate_frame_ids: tuple[int, ...]
    rail_origin_m: tuple[float, float, float]
    rail_axis: tuple[float, float, float]
    pose_displacement_m: float
    pose_audit: tuple[S12ScanPoseAudit, ...]

    @property
    def candidate_frame_count(self) -> int:
        return len(self.candidate_frame_ids)

    def as_dict(self) -> dict[str, object]:
        return {
            "start_frame_id": self.start_frame_id,
            "end_frame_id": self.end_frame_id,
            "candidate_frame_ids": list(self.candidate_frame_ids),
            "candidate_frame_count": self.candidate_frame_count,
            "rail_origin_m": list(self.rail_origin_m),
            "rail_axis": list(self.rail_axis),
            "pose_displacement_m": self.pose_displacement_m,
        }


def fit_s12_rail_axis(
    qualifications: Sequence[S12PoseQualification], config: S12ScanConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    qualified = [row for row in qualifications if row.qualified and row.record is not None]
    if len(qualified) < config.minimum_pose_count:
        raise RuntimeError(
            f"S1.2 structural fatal: only {len(qualified)} direct qualified poses; "
            f"requires {config.minimum_pose_count}"
        )
    points = np.asarray([row.record.camera_center_m for row in qualified], dtype=np.float64)
    rng = np.random.default_rng(12012)
    best_mask: np.ndarray | None = None
    best_score: tuple[int, float] | None = None
    count = len(points)
    # Endpoint pair is always considered; random pairs add outlier robustness.
    pairs = [(0, count - 1)]
    pairs.extend(
        tuple(int(value) for value in rng.choice(count, size=2, replace=False))
        for _ in range(config.rail_axis_ransac_iterations)
    )
    for left, right in pairs:
        delta = points[right] - points[left]
        norm = float(np.linalg.norm(delta))
        if norm <= 1e-12:
            continue
        direction = delta / norm
        offsets = points - points[left]
        residuals = np.linalg.norm(offsets - np.outer(offsets @ direction, direction), axis=1)
        mask = residuals <= config.rail_axis_inlier_threshold_m
        score = (int(mask.sum()), -float(np.median(residuals[mask])) if mask.any() else -float("inf"))
        if best_score is None or score > best_score:
            best_score, best_mask = score, mask
    if best_mask is None or int(best_mask.sum()) < 2:
        raise RuntimeError("S1.2 structural fatal: robust rail-axis fit has too few inliers")
    inliers = points[best_mask]
    origin = np.mean(inliers, axis=0)
    _, _, vh = np.linalg.svd(inliers - origin, full_matrices=False)
    axis = vh[0]
    temporal = np.arange(count, dtype=np.float64)
    temporal -= float(np.mean(temporal))
    temporal_direction = temporal @ points
    signed_temporal = float(temporal_direction @ axis)
    if abs(signed_temporal) <= 1e-12:
        for delta in np.diff(points, axis=0):
            signed_temporal = float(delta @ axis)
            if abs(signed_temporal) > 1e-12:
                break
    if signed_temporal < 0.0:
        axis = -axis
    progresses = (points - origin) @ axis
    return origin, axis, progresses


def _agreement_values(
    qualifications: Sequence[S12PoseQualification],
    image_motion_agreement: Mapping[int, bool] | Sequence[bool] | None,
) -> dict[int, bool]:
    if image_motion_agreement is None:
        return {}
    if isinstance(image_motion_agreement, Mapping):
        return {int(key): bool(value) for key, value in image_motion_agreement.items()}
    values = list(image_motion_agreement)
    if len(values) != max(0, len(qualifications) - 1):
        raise ValueError("S1.2 image-motion agreement must have one value per pose transition")
    return {
        qualifications[index + 1].frame_id: bool(value)
        for index, value in enumerate(values)
    }


def select_s12_scan_segment(
    qualifications: Sequence[S12PoseQualification],
    config: S12ScanConfig,
    *,
    image_motion_agreement: Mapping[int, bool] | Sequence[bool] | None = None,
) -> S12ScanSegment:
    """Choose one longest chronological, monotonic direct-pose scan.

    Image/pose direction evidence is accepted as an explicit audit input.  If
    it has not yet been measured, direction is recorded as unknown; no 2-D
    motion is invented or used as a pose replacement.
    """

    origin, axis, _ = fit_s12_rail_axis(qualifications, config)
    agreement = _agreement_values(qualifications, image_motion_agreement)
    audit: list[S12ScanPoseAudit] = []
    progress_by_index: dict[int, float] = {}
    for index, row in enumerate(qualifications):
        reasons = list(row.rejection_reasons)
        progress = float("nan")
        cross_track = float("inf")
        if row.qualified and row.record is not None:
            center = row.record.camera_center_m
            offset = center - origin
            progress = float(offset @ axis)
            cross_track = float(np.linalg.norm(offset - progress * axis))
            progress_by_index[index] = progress
            if cross_track > config.maximum_cross_track_error_m:
                reasons.append("cross_track_limit")
        motion_value = agreement.get(row.frame_id)
        if motion_value is False and config.require_image_motion_direction_agreement:
            reasons.append("image_pose_direction_disagreement")
        audit.append(
            S12ScanPoseAudit(
                frame_id=row.frame_id,
                progress_m=progress,
                cross_track_error_m=cross_track,
                direct_pose=row.direct_pose,
                pose_qualified=row.qualified,
                image_motion_direction_agreement=motion_value,
                accepted=not reasons,
                rejection_reasons=tuple(reasons),
            )
        )

    segments: list[list[int]] = []
    current: list[int] = []
    pause_span = 0
    source_gap_reasons = {"missing_trajectory_record", "not_direct_orb_pose"}
    for index, row in enumerate(audit):
        if not row.accepted:
            if set(row.rejection_reasons).issubset(source_gap_reasons):
                if (
                    current
                    and row.frame_id - audit[current[-1]].frame_id
                    > config.maximum_pause_span_frames
                ):
                    segments.append(current)
                    current = []
                    pause_span = 0
                continue
            if current:
                segments.append(current)
                current = []
            pause_span = 0
            continue
        if current:
            if row.frame_id - audit[current[-1]].frame_id > config.maximum_pause_span_frames:
                segments.append(current)
                current = []
                pause_span = 0
        if current:
            step = progress_by_index[index] - progress_by_index[current[-1]]
            if step < -config.maximum_backward_step_m:
                segments.append(current)
                current = []
                pause_span = 0
            elif step <= config.maximum_backward_step_m:
                pause_span += max(1, row.frame_id - audit[current[-1]].frame_id)
                if pause_span > config.maximum_pause_span_frames:
                    segments.append(current)
                    current = []
                    pause_span = 0
            else:
                pause_span = 0
        current.append(index)
    if current:
        segments.append(current)
    eligible: list[tuple[float, int, int, list[int]]] = []
    for indices in segments:
        displacement = progress_by_index[indices[-1]] - progress_by_index[indices[0]]
        if len(indices) >= config.minimum_pose_count and displacement >= config.minimum_scan_displacement_m:
            # Longest by frame count, then displacement, then earliest start.
            eligible.append((float(len(indices)), displacement, -indices[0], indices))
    if not eligible:
        raise RuntimeError("S1.2 structural fatal: longest valid scan segment is insufficient")
    indices = max(eligible)[3]
    selected = tuple(audit[index].frame_id for index in indices)
    displacement = progress_by_index[indices[-1]] - progress_by_index[indices[0]]
    return S12ScanSegment(
        start_frame_id=selected[0],
        end_frame_id=selected[-1],
        candidate_frame_ids=selected,
        rail_origin_m=tuple(float(value) for value in origin),
        rail_axis=tuple(float(value) for value in axis),
        pose_displacement_m=float(displacement),
        pose_audit=tuple(audit),
    )


detect_s12_scan_segment = select_s12_scan_segment


__all__ = [
    "S12ScanPoseAudit",
    "S12ScanSegment",
    "detect_s12_scan_segment",
    "fit_s12_rail_axis",
    "select_s12_scan_segment",
]
