"""D1 candidate-only dense real-frame source layout.

This is deliberately a source/pose gate, not a renderer.  It selects only
already-recorded RGB-D frames and annotates every accepted pose as either an
actual ORB anchor or a bounded, audited bracketed prior.  No pose is added to
ORB-SLAM3 and no pixel owner is inferred here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from .session import CameraIntrinsics, RGBDFrame
from .video_dense_pose_prior import (
    DensePosePrior,
    DensePosePriorConfig,
    DensePosePriorError,
    ORBPoseAnchor,
    audit_dense_image_motion_and_rgbd_residual,
    interpolate_bracketed_se3,
)


class DenseRealFrameLayoutError(RuntimeError):
    """D1 has insufficient real-frame pose or owner observability."""

    def __init__(self, message: str, *, diagnostics: Mapping[str, object] | None = None) -> None:
        super().__init__(message)
        self.diagnostics = dict(diagnostics or {})


@dataclass(frozen=True)
class DenseAdjacentFrameAudit:
    """Bidirectional dense evidence for one adjacent pair of actual sources.

    This deliberately identifies the two real source frames, rather than the
    two ORB anchors which bracket either source.  D1 may only use an
    interpolated pose after this adjacent-real-frame evidence is accepted.
    """

    left_frame_id: int
    right_frame_id: int
    left_source_pose_origin: str
    right_source_pose_origin: str
    forward_backward_p95_pixels: float
    rgbd_residual_p95_pixels: float
    forward_backward_sample_count: int
    rgbd_residual_sample_count: int

    def __post_init__(self) -> None:
        if self.left_frame_id >= self.right_frame_id:
            raise DenseRealFrameLayoutError("D1 adjacent audit frame ids must increase")
        if any(origin not in {"direct_orb_anchor", "interpolated_se3_prior", "refined_dense_prior"}
               for origin in (self.left_source_pose_origin, self.right_source_pose_origin)):
            raise DenseRealFrameLayoutError("D1 adjacent audit has invalid pose provenance")
        values = (self.forward_backward_p95_pixels, self.rgbd_residual_p95_pixels)
        if not np.isfinite(values).all() or any(value < 0.0 for value in values):
            raise DenseRealFrameLayoutError("D1 adjacent audit residuals must be finite and non-negative")
        if min(self.forward_backward_sample_count, self.rgbd_residual_sample_count) < 0:
            raise DenseRealFrameLayoutError("D1 adjacent audit sample counts must be non-negative")

    def accepted(self, config: DensePosePriorConfig) -> bool:
        return (
            self.forward_backward_sample_count >= config.minimum_audit_samples
            and self.rgbd_residual_sample_count >= config.minimum_audit_samples
            and self.forward_backward_p95_pixels <= config.maximum_forward_backward_p95_pixels
            and self.rgbd_residual_p95_pixels <= config.maximum_rgbd_residual_p95_pixels
        )

    def as_dict(self, config: DensePosePriorConfig) -> dict[str, object]:
        return {
            "left_frame_id": self.left_frame_id,
            "right_frame_id": self.right_frame_id,
            "left_source_pose_origin": self.left_source_pose_origin,
            "right_source_pose_origin": self.right_source_pose_origin,
            "forward_backward_p95_pixels": self.forward_backward_p95_pixels,
            "rgbd_residual_p95_pixels": self.rgbd_residual_p95_pixels,
            "forward_backward_sample_count": self.forward_backward_sample_count,
            "rgbd_residual_sample_count": self.rgbd_residual_sample_count,
            "maximum_forward_backward_p95_pixels": config.maximum_forward_backward_p95_pixels,
            "maximum_rgbd_residual_p95_pixels": config.maximum_rgbd_residual_p95_pixels,
            "accepted": self.accepted(config),
        }


@dataclass(frozen=True)
class DenseRealFrameLayoutConfig:
    real_source_fps: float = 24.0
    minimum_dense_prior_coverage: float = 0.95
    minimum_compact_object_support_coverage: float = 0.98

    def __post_init__(self) -> None:
        coverage = (self.minimum_dense_prior_coverage, self.minimum_compact_object_support_coverage)
        if not np.isfinite(coverage).all() or any(not 0.0 < value <= 1.0 for value in coverage):
            raise DenseRealFrameLayoutError("D1 coverage limits must be finite fractions in (0, 1]")
        if not np.isfinite(self.real_source_fps) or not 20.0 <= self.real_source_fps <= 30.0:
            raise DenseRealFrameLayoutError("D1 real source FPS must be finite and in [20, 30]")


@dataclass(frozen=True)
class DenseRealFrameLayout:
    frames: tuple[RGBDFrame, ...]
    priors: tuple[DensePosePrior, ...]
    audit: dict[str, object]

    @property
    def camera_to_world(self) -> tuple[np.ndarray, ...]:
        return tuple(item.camera_to_world for item in self.priors)

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": "gemini305-video-dense-real-frame-layout/v1",
            "source_frame_ids": [item.frame_id for item in self.frames],
            "source_pose_origins": [item.source_pose_origin for item in self.priors],
            "real_rgbd_sources_only": True,
            "no_extrapolated_poses": True,
            **self.audit,
        }


def select_dense_real_source_frames(
    frames: Sequence[RGBDFrame], *, real_source_fps: float,
) -> tuple[RGBDFrame, ...]:
    """Select a uniform cadence of recorded frames without synthesising one.

    Timestamp-grid targets map to the nearest later chronological real file.
    Keeping both scan endpoints makes an unbracketed endpoint visible to the
    strict pose gate rather than silently removing it from the experiment.
    """

    ordered = tuple(frames)
    if len(ordered) < 2:
        raise DenseRealFrameLayoutError("D1 requires at least two real scan frames")
    if not np.isfinite(real_source_fps) or not 20.0 <= real_source_fps <= 30.0:
        raise DenseRealFrameLayoutError("D1 real source FPS must be finite and in [20, 30]")
    timestamps = [item.timestamp_us for item in ordered]
    if any(value is None for value in timestamps) or any(
        int(right) <= int(left) for left, right in zip(timestamps, timestamps[1:])
    ):
        raise DenseRealFrameLayoutError("D1 real scan frames must have increasing timestamps")
    interval_us = 1_000_000.0 / float(real_source_fps)
    first_us, last_us = int(timestamps[0]), int(timestamps[-1])
    indices = [0]
    target_us = first_us + interval_us
    search_start = 1
    while target_us < last_us and search_start < len(ordered):
        # The index tie break selects the earlier real file deterministically.
        chosen = min(
            range(search_start, len(ordered)),
            key=lambda index: (abs(int(timestamps[index]) - target_us), index),
        )
        indices.append(chosen)
        search_start = chosen + 1
        target_us += interval_us
    if indices[-1] != len(ordered) - 1:
        indices.append(len(ordered) - 1)
    return tuple(ordered[index] for index in indices)


def _read_real_frame(frame: RGBDFrame, calibration: CameraIntrinsics) -> tuple[np.ndarray, np.ndarray]:
    colour = cv2.imread(str(frame.color_path), cv2.IMREAD_COLOR)
    depth = cv2.imread(str(frame.aligned_depth_path), cv2.IMREAD_UNCHANGED)
    if colour is None or colour.shape[:2] != (calibration.height, calibration.width):
        raise DenseRealFrameLayoutError(f"D1 cannot decode real RGB frame {frame.frame_id}")
    if depth is None or depth.dtype != np.uint16 or depth.shape != colour.shape[:2]:
        raise DenseRealFrameLayoutError(f"D1 cannot decode aligned depth frame {frame.frame_id}")
    if not np.isfinite(frame.depth_scale_mm_per_unit) or frame.depth_scale_mm_per_unit <= 0:
        raise DenseRealFrameLayoutError(f"D1 source {frame.frame_id} has invalid depth unit")
    return colour, depth.astype(np.float64) * float(frame.depth_scale_mm_per_unit)


def _provisional_bracketed_prior(
    frame: RGBDFrame,
    anchors: Sequence[ORBPoseAnchor],
    config: DensePosePriorConfig,
) -> DensePosePrior:
    """Create a bracket-only candidate prior before adjacent-pair evidence.

    This helper is private specifically because the returned prior has not
    yet been admitted to a D1 layout.  It enforces the same direct-anchor,
    no-extrapolation and 150 ms requirements as the final prior.
    """

    if frame.timestamp_us is None:
        raise DensePosePriorError("real RGB-D frame lacks timestamp")
    timestamp = int(frame.timestamp_us)
    direct = next((anchor for anchor in anchors if anchor.frame_id == frame.frame_id), None)
    if direct is not None:
        if direct.timestamp_us != timestamp:
            raise DensePosePriorError("direct ORB anchor timestamp differs from its real frame")
        return DensePosePrior(frame.frame_id, timestamp, direct.camera_to_world, "direct_orb_anchor", {
            "schema": "gemini305-video-dense-real-frame-pose-prior/v1",
            "direct_orb_anchor": True,
            "no_extrapolation": True,
        })
    bracket = next(
        ((left, right) for left, right in zip(anchors, anchors[1:])
         if left.timestamp_us < timestamp < right.timestamp_us),
        None,
    )
    if bracket is None:
        raise DensePosePriorError("real intermediate has no enclosing ORB anchor bracket")
    left, right = bracket
    left_us, right_us = timestamp - left.timestamp_us, right.timestamp_us - timestamp
    if max(left_us, right_us) > config.maximum_anchor_distance_us:
        raise DensePosePriorError("real intermediate exceeds the 150 ms anchor gate")
    return DensePosePrior(frame.frame_id, timestamp, interpolate_bracketed_se3(left, right, timestamp),
                          "interpolated_se3_prior", {
                              "schema": "gemini305-video-dense-real-frame-pose-prior/v1",
                              "left_anchor_frame_id": left.frame_id,
                              "right_anchor_frame_id": right.frame_id,
                              "left_anchor_distance_us": left_us,
                              "right_anchor_distance_us": right_us,
                              "maximum_anchor_distance_us": config.maximum_anchor_distance_us,
                              "no_extrapolation": True,
                          })


def _dense_audit_for_adjacent_sources(
    left_frame: RGBDFrame,
    left_prior: DensePosePrior,
    right_frame: RGBDFrame,
    right_prior: DensePosePrior,
    calibration: CameraIntrinsics,
) -> DenseAdjacentFrameAudit:
    """Measure D1 evidence only across adjacent actual dense RGB-D sources.

    Both directions are evaluated so each real source's aligned depth is
    audited.  This is intentionally not an intermediate-to-anchor union.
    """

    left_image, left_depth = _read_real_frame(left_frame, calibration)
    right_image, right_depth = _read_real_frame(right_frame, calibration)
    forward_fb, forward_rgbd = audit_dense_image_motion_and_rgbd_residual(
        left_image, right_image, left_depth, left_prior.camera_to_world,
        right_prior.camera_to_world, calibration,
    )
    backward_fb, backward_rgbd = audit_dense_image_motion_and_rgbd_residual(
        right_image, left_image, right_depth, right_prior.camera_to_world,
        left_prior.camera_to_world, calibration,
    )
    fb = np.concatenate((forward_fb, backward_fb))
    rgbd = np.concatenate((forward_rgbd, backward_rgbd))
    if not fb.size or not rgbd.size:
        raise DenseRealFrameLayoutError("D1 adjacent audit has no finite RGB-D evidence samples")
    return DenseAdjacentFrameAudit(
        left_frame_id=int(left_frame.frame_id),
        right_frame_id=int(right_frame.frame_id),
        left_source_pose_origin=left_prior.source_pose_origin,
        right_source_pose_origin=right_prior.source_pose_origin,
        forward_backward_p95_pixels=float(np.percentile(fb, 95)),
        rgbd_residual_p95_pixels=float(np.percentile(rgbd, 95)),
        forward_backward_sample_count=int(fb.size),
        rgbd_residual_sample_count=int(rgbd.size),
    )


def build_dense_real_frame_layout(
    frames: Sequence[RGBDFrame],
    orb_anchors: Sequence[ORBPoseAnchor],
    calibration: CameraIntrinsics,
    *,
    pose_config: DensePosePriorConfig | None = None,
    layout_config: DenseRealFrameLayoutConfig | None = None,
) -> DenseRealFrameLayout:
    """Build D1 sources from actual frames, rejecting failed priors fail-closed."""

    settings = pose_config or DensePosePriorConfig()
    gate = layout_config or DenseRealFrameLayoutConfig()
    ordered = tuple(frames)
    anchors = tuple(orb_anchors)
    if len(ordered) < 2 or len(anchors) < 2:
        raise DenseRealFrameLayoutError("D1 requires at least two real frames and two ORB anchors")
    frame_by_id = {int(item.frame_id): item for item in ordered}
    if len(frame_by_id) != len(ordered) or any(anchor.frame_id not in frame_by_id for anchor in anchors):
        raise DenseRealFrameLayoutError("D1 anchors must refer to unique real scan frames")
    # The dense graph is evidence at capture cadence (normally 60 FPS), not
    # the render cadence.  In particular, a 24 FPS source must never be
    # rejected merely because its next *render* source is 42 ms away: its
    # local 60 FPS incident edges are the only D1 image/RGB-D evidence.
    provisional: dict[int, DensePosePrior] = {}
    rejected_pose_reasons: dict[str, str] = {}
    for frame in ordered:
        try:
            provisional[frame.frame_id] = _provisional_bracketed_prior(frame, anchors, settings)
        except DensePosePriorError as error:
            rejected_pose_reasons[str(frame.frame_id)] = str(error)

    adjacent_audits: list[DenseAdjacentFrameAudit] = []
    unmeasured_edges: list[dict[str, object]] = []
    all_edge_records: list[dict[str, object]] = []
    evidence_by_frame_id: dict[int, list[dict[str, object]]] = {item.frame_id: [] for item in ordered}
    for left_frame, right_frame in zip(ordered, ordered[1:]):
        left_prior, right_prior = provisional.get(left_frame.frame_id), provisional.get(right_frame.frame_id)
        if left_prior is None or right_prior is None:
            record = {
                "left_frame_id": left_frame.frame_id, "right_frame_id": right_frame.frame_id,
                "accepted": False, "measurement_state": "missing_bracketed_prior",
            }
            unmeasured_edges.append(record)
            all_edge_records.append(record)
            continue
        try:
            edge = _dense_audit_for_adjacent_sources(left_frame, left_prior, right_frame, right_prior, calibration)
        except (DensePosePriorError, DenseRealFrameLayoutError, ValueError, cv2.error) as error:
            record = {
                "left_frame_id": left_frame.frame_id, "right_frame_id": right_frame.frame_id,
                "accepted": False, "measurement_state": "unmeasurable", "measurement_error": str(error),
            }
            unmeasured_edges.append(record)
            all_edge_records.append(record)
            continue
        record = edge.as_dict(settings)
        adjacent_audits.append(edge)
        all_edge_records.append(record)
        evidence_by_frame_id[left_frame.frame_id].append(record)
        evidence_by_frame_id[right_frame.frame_id].append(record)

    # A source requires all its applicable 60 FPS incident edges to be
    # measured and accepted.  This prevents a bad or unmeasured real frame
    # from rendering while still permitting a nearby auditable real file to
    # occupy a 20--30 FPS grid opportunity.
    eligible_ids: set[int] = set()
    rejection_by_frame: dict[int, str] = {}
    for index, frame in enumerate(ordered):
        if frame.frame_id not in provisional:
            rejection_by_frame[frame.frame_id] = rejected_pose_reasons[str(frame.frame_id)]
            continue
        incident = [entry for entry in all_edge_records if frame.frame_id in (entry["left_frame_id"], entry["right_frame_id"])]
        # Each endpoint has one incident edge and each interior file two.
        required_incident = 1 if index in {0, len(ordered) - 1} else 2
        if len(incident) != required_incident or not all(bool(entry.get("accepted")) for entry in incident):
            rejection_by_frame[frame.frame_id] = "local_60fps_incident_dense_evidence_failed"
            continue
        eligible_ids.add(frame.frame_id)

    grid_frames = select_dense_real_source_frames(ordered, real_source_fps=gate.real_source_fps)
    used_ids: set[int] = set()
    last_selected_timestamp_us = -1
    accepted: list[RGBDFrame] = []
    priors: list[DensePosePrior] = []
    grid_selection: list[dict[str, object]] = []
    for slot, target in enumerate(grid_frames):
        choices = [
            frame for frame in ordered
            if (
                frame.frame_id in eligible_ids
                and frame.frame_id not in used_ids
                and int(frame.timestamp_us) > last_selected_timestamp_us
            )
        ]
        chosen = min(choices, key=lambda frame: (abs(int(frame.timestamp_us) - int(target.timestamp_us)), frame.frame_id)) if choices else None
        record: dict[str, object] = {
            "grid_slot": slot, "target_frame_id": target.frame_id,
            "target_timestamp_us": int(target.timestamp_us),
        }
        if chosen is None:
            record.update({"selected": False, "omission_reason": "no_auditable_real_source"})
        else:
            used_ids.add(chosen.frame_id)
            last_selected_timestamp_us = int(chosen.timestamp_us)
            accepted.append(chosen)
            priors.append(provisional[chosen.frame_id])
            record.update({
                "selected": True, "selected_frame_id": chosen.frame_id,
                "selected_timestamp_us": int(chosen.timestamp_us),
                "grid_offset_us": int(chosen.timestamp_us) - int(target.timestamp_us),
            })
        grid_selection.append(record)

    coverage = len(accepted) / len(grid_frames)
    if coverage < gate.minimum_dense_prior_coverage:
        raise DenseRealFrameLayoutError(
            f"D1 dense real-frame auditable source coverage {coverage:.3f} is below {gate.minimum_dense_prior_coverage:.3f}",
            diagnostics={
                "schema": "gemini305-video-dense-real-frame-layout-diagnostics/v2",
                "stage": "dense_evidence_grid_source_coverage",
                "dense_prior_coverage": coverage, "dense_prior_coverage_gate": gate.minimum_dense_prior_coverage,
                "accepted_source_count": len(accepted), "candidate_source_count": len(grid_frames),
                "scan_frame_count": len(ordered),
                "candidate_source_frame_ids": [item.frame_id for item in grid_frames],
                "grid_source_selection": grid_selection,
                "rejected_pose_reasons": rejected_pose_reasons,
                "rejected_real_frame_reasons": {str(key): value for key, value in rejection_by_frame.items()},
                "adjacent_real_frame_audits": all_edge_records,
            },
        )
    if len(accepted) < 2:
        raise DenseRealFrameLayoutError("D1 has insufficient auditable real render sources")

    audited_priors: list[DensePosePrior] = []
    for prior in priors:
        audit = dict(prior.audit)
        audit["local_60fps_incident_evidence"] = evidence_by_frame_id[prior.frame_id]
        audited_priors.append(DensePosePrior(prior.frame_id, prior.timestamp_us, prior.camera_to_world,
                                             prior.source_pose_origin, audit))
    return DenseRealFrameLayout(
        frames=tuple(accepted), priors=tuple(audited_priors), audit={
            "dense_prior_coverage": coverage,
            "dense_prior_coverage_gate": gate.minimum_dense_prior_coverage,
            "real_source_fps": gate.real_source_fps,
            "accepted_source_count": len(accepted),
            "candidate_source_count": len(grid_frames), "scan_frame_count": len(ordered),
            "candidate_source_frame_ids": [item.frame_id for item in grid_frames],
            "rejected_real_frame_ids": [item.frame_id for item in grid_frames if item.frame_id not in used_ids],
            "grid_source_selection": grid_selection,
            "rejected_pose_reasons": rejected_pose_reasons,
            "rejected_real_frame_reasons": {str(key): value for key, value in rejection_by_frame.items()},
            "adjacent_real_frame_audits": all_edge_records,
        },
    )


def verify_dense_owner_observability(
    owner_frame_id: np.ndarray,
    *,
    compact_support_masks: Mapping[str, np.ndarray] = {},
    config: DenseRealFrameLayoutConfig | None = None,
) -> dict[str, object]:
    """Fail if a D1 canvas has unowned pixels or an under-covered compact object."""

    gate = config or DenseRealFrameLayoutConfig()
    owner = np.asarray(owner_frame_id)
    if owner.ndim != 2:
        raise DenseRealFrameLayoutError("D1 owner map must be two-dimensional")
    unowned = int(np.count_nonzero(owner < 0))
    if unowned:
        raise DenseRealFrameLayoutError(f"D1 output has {unowned} unowned pixels")
    coverage: dict[str, float] = {}
    for label, mask_value in compact_support_masks.items():
        mask = np.asarray(mask_value, dtype=bool)
        if mask.shape != owner.shape:
            raise DenseRealFrameLayoutError(f"D1 compact support {label!r} shape differs from owner map")
        total = int(np.count_nonzero(mask))
        if total == 0:
            raise DenseRealFrameLayoutError(f"D1 compact support {label!r} is empty")
        ratio = float(np.count_nonzero(mask & (owner >= 0)) / total)
        coverage[str(label)] = ratio
        if ratio < gate.minimum_compact_object_support_coverage:
            raise DenseRealFrameLayoutError(
                f"D1 compact support {label!r} coverage {ratio:.3f} is below "
                f"{gate.minimum_compact_object_support_coverage:.3f}"
            )
    return {"unowned_pixel_count": unowned, "compact_object_support_coverage": coverage,
            "compact_object_support_coverage_gate": gate.minimum_compact_object_support_coverage}


def compact_support_masks_from_projection(
    annotations: Mapping[str, Any],
    projection_payload: Mapping[str, Any],
    projection_masks: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Return every v2 compact-object consensus mask or fail closed.

    Projection is supplied by M2's owner-independent full-support adapter.
    This merely chooses which already-projected fixed measurements constitute
    D1's compact-object observability gate; it cannot affect rendering.
    """

    compact_groups = {
        str(entry.get("measurement_group", entry.get("id")))
        for entry in annotations.get("objects", [])
        if isinstance(entry, Mapping)
        and entry.get("role") == "compact_foreground_single_owner"
        and isinstance(entry.get("id"), str)
    }
    if not compact_groups:
        raise DenseRealFrameLayoutError("D1 requires v2 compact foreground object annotations")
    projected: dict[str, np.ndarray] = {}
    entries = projection_payload.get("objects")
    if not isinstance(entries, list):
        raise DenseRealFrameLayoutError("D1 full-support projection lacks object entries")
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        group = entry.get("measurement_group", entry.get("id"))
        key = entry.get("mask_key")
        if isinstance(group, str) and group in compact_groups and isinstance(key, str):
            mask = projection_masks.get(key)
            if mask is None:
                raise DenseRealFrameLayoutError(f"D1 full-support projection mask is missing for {group!r}")
            projected[group] = np.asarray(mask, dtype=bool)
    missing = sorted(compact_groups - set(projected))
    if missing:
        raise DenseRealFrameLayoutError(
            "D1 full-support projection omitted compact object group(s): " + ", ".join(missing)
        )
    return projected


__all__ = ["DenseRealFrameLayout", "DenseRealFrameLayoutConfig", "DenseRealFrameLayoutError",
           "build_dense_real_frame_layout", "compact_support_masks_from_projection",
           "verify_dense_owner_observability"]
