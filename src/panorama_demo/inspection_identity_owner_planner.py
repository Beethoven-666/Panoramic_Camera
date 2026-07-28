"""In-memory planning of pre-seam single-panel inspection owner intervals.

FastSAM polygons and DIS tracks are identity evidence only.  Every accepted
footprint is reconstructed again from the selected reference frame's aligned
depth and immutable camera-to-world pose.  The result contains only masks
which constrain an existing real RGB reference panel; it never carries or
modifies RGB, depth, pose, or sampling coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Mapping, Sequence

import cv2
import numpy as np

from .dis_track_direct_handoff import (
    DirectHandoffConfig,
    DirectProjectedObservation,
    evaluate_direct_track,
)
from .fastsam_dis_tracking import FastSAMDISTrack, FastSAMDISTrackingResult
from .inspection_fastsam_track import (
    FastSAMRGBDCandidate,
    polygon_mask as candidate_mask,
)
from .inspection_multiview import (
    InspectionForegroundIdentityOwner,
    InspectionMultiviewLayout,
    InspectionPreSeamHardOwnerInterval,
)
from .inspection_ocr_panel import (
    OCRSeededPanel,
    StableObjectTrackEvidence,
    audit_object_rich_interval,
    sample_mask_world_points,
    select_object_rich_neighbor_tracks,
    track_ocr_seeded_panels,
)
from .session import CameraIntrinsics


_DIRECT_STABLE_TRACK_GROUP_ID = 2_147_000_001
_SHELF_INVENTORY_GROUP_ID = 2_147_000_002


@dataclass(frozen=True)
class InspectionIdentityOwnerFrame:
    """One selected real reference-panel frame and its canvas validity."""

    panel_index: int
    source_index: int
    frame_id: int
    image_bgr: np.ndarray
    depth_mm: np.ndarray
    reliable_depth: np.ndarray
    camera_to_world: np.ndarray
    panel_valid_mask: np.ndarray


@dataclass(frozen=True)
class InspectionIdentityOwnerPlannerConfig:
    required_neighbor_track_count: int = 2
    maximum_source_adjacency_gap_pixels: int = 160
    minimum_vertical_overlap_ratio: float = 0.15
    maximum_interval_gap_pixels: float = 160.0
    footprint_sample_stride: int = 2
    lock_margin_pixels: int = 2
    minimum_projected_sample_count: int = 30


@dataclass(frozen=True)
class InspectionIdentityOwnerPlan:
    intervals: tuple[InspectionPreSeamHardOwnerInterval, ...]
    foreground_owners: tuple[InspectionForegroundIdentityOwner, ...]
    audit: dict[str, object]


@dataclass(frozen=True)
class InspectionDirectIdentityOwnerPlan:
    foreground_owners: tuple[InspectionForegroundIdentityOwner, ...]
    audit: dict[str, object]


@dataclass(frozen=True)
class ShelfInventoryOwnerConfig:
    """Fixed gates for complete objects standing on the yellow middle shelf."""

    minimum_source_depth_coverage_ratio: float = 0.90
    minimum_mask_bbox_fill_ratio: float = 0.20
    minimum_candidate_solidity: float = 0.35
    minimum_bbox_aspect_ratio: float = 0.12
    maximum_bbox_aspect_ratio: float = 6.0
    maximum_bbox_area_ratio: float = 0.12
    source_boundary_margin_pixels: int = 8
    minimum_yellow_row_coverage_ratio: float = 0.10
    minimum_yellow_shelf_run_pixels: int = 12
    shelf_contact_tolerance_ratio: float = 0.10
    hierarchy_containment_ratio: float = 0.88

    def validate(self) -> None:
        for name in (
            "minimum_source_depth_coverage_ratio",
            "minimum_mask_bbox_fill_ratio",
            "minimum_candidate_solidity",
            "maximum_bbox_area_ratio",
            "minimum_yellow_row_coverage_ratio",
            "shelf_contact_tolerance_ratio",
            "hierarchy_containment_ratio",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1]")
        if not (
            0.0
            < float(self.minimum_bbox_aspect_ratio)
            <= float(self.maximum_bbox_aspect_ratio)
        ):
            raise ValueError("Shelf inventory bbox aspect gates are invalid")
        if not 0 <= int(self.source_boundary_margin_pixels) <= 64:
            raise ValueError("Shelf inventory source margin is invalid")
        if int(self.minimum_yellow_shelf_run_pixels) < 2:
            raise ValueError("Shelf inventory yellow run is invalid")


@dataclass(frozen=True)
class InspectionShelfInventoryOwnerPlan:
    foreground_owners: tuple[InspectionForegroundIdentityOwner, ...]
    audit: dict[str, object]


@dataclass(frozen=True)
class _ProjectedStructure:
    footprint: np.ndarray
    x_span: tuple[float, float]
    y_span: tuple[float, float]
    sample_count: int
    valid_sample_count: int
    in_bounds_ratio: float


def _validate_pose(pose: np.ndarray) -> np.ndarray:
    value = np.asarray(pose, dtype=np.float64)
    if value.shape != (4, 4) or not np.isfinite(value).all():
        raise ValueError("Inspection identity owner pose must be finite 4x4")
    rotation = value[:3, :3]
    if (
        not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
        or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5)
        or not np.allclose(value[3], (0.0, 0.0, 0.0, 1.0), atol=1e-8)
    ):
        raise ValueError("Inspection identity owner pose must be rigid SE(3)")
    return value


def _validate_frames(
    frames: Sequence[InspectionIdentityOwnerFrame],
    *,
    layout: InspectionMultiviewLayout,
    intrinsics: CameraIntrinsics,
) -> dict[int, InspectionIdentityOwnerFrame]:
    by_frame: dict[int, InspectionIdentityOwnerFrame] = {}
    used_panels: set[int] = set()
    used_sources: set[int] = set()
    for frame in frames:
        frame_id = int(frame.frame_id)
        panel_index = int(frame.panel_index)
        source_index = int(frame.source_index)
        if (
            frame_id in by_frame
            or panel_index in used_panels
            or source_index in used_sources
        ):
            raise ValueError(
                "Inspection identity owner frames, sources, and panels must "
                "map one-to-one"
            )
        if source_index < 0:
            raise ValueError("Inspection identity owner source index is invalid")
        if panel_index < 0 or panel_index >= len(layout.panels):
            raise ValueError("Inspection identity owner frame has unknown panel")
        panel = layout.panels[panel_index]
        if int(panel.panel_index) != panel_index:
            raise ValueError("Inspection layout panel indices are inconsistent")
        image = np.asarray(frame.image_bgr)
        depth = np.asarray(frame.depth_mm)
        reliable = np.asarray(frame.reliable_depth)
        panel_valid = np.asarray(frame.panel_valid_mask)
        if (
            image.dtype != np.uint8
            or image.shape != (intrinsics.height, intrinsics.width, 3)
            or depth.shape != image.shape[:2]
            or reliable.dtype != np.bool_
            or reliable.shape != depth.shape
            or panel_valid.dtype != np.bool_
            or panel_valid.shape != (layout.height, layout.width)
        ):
            raise ValueError(
                "Inspection identity owner RGB-D or canvas validity is invalid"
            )
        _validate_pose(frame.camera_to_world)
        by_frame[frame_id] = frame
        used_panels.add(panel_index)
        used_sources.add(source_index)
    return by_frame


def _polygon_mask(
    polygon_xy: np.ndarray,
    shape: tuple[int, int],
) -> np.ndarray:
    polygon = np.asarray(polygon_xy, dtype=np.int32)
    mask = np.zeros(shape, dtype=np.uint8)
    if polygon.ndim == 2 and polygon.shape[0] >= 3 and polygon.shape[1] == 2:
        cv2.fillPoly(mask, [polygon], 1)
    return np.ascontiguousarray(mask > 0)


def _bbox_adjacent(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
    *,
    maximum_gap: int,
    minimum_vertical_overlap_ratio: float,
) -> bool:
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    horizontal_gap = max(0, ax - (bx + bw), bx - (ax + aw))
    vertical_overlap = max(0, min(ay + ah, by + bh) - max(ay, by))
    overlap_ratio = vertical_overlap / max(1, min(ah, bh))
    return bool(
        horizontal_gap <= maximum_gap
        and overlap_ratio >= minimum_vertical_overlap_ratio
    )


def _candidate_clarity(
    candidate: FastSAMRGBDCandidate,
    image_bgr: np.ndarray,
) -> float:
    x, y, width, height = candidate.bbox_xywh
    if (
        x < 0
        or y < 0
        or width <= 0
        or height <= 0
        or x + width > image_bgr.shape[1]
        or y + height > image_bgr.shape[0]
    ):
        raise ValueError("Stable DIS candidate bbox is outside its RGB frame")
    gray = cv2.cvtColor(
        image_bgr[y : y + height, x : x + width],
        cv2.COLOR_BGR2GRAY,
    )
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _measured_mask_properties(
    mask: np.ndarray,
    frame: InspectionIdentityOwnerFrame,
) -> tuple[float, float]:
    selected = np.asarray(mask, dtype=bool)
    area = int(np.count_nonzero(selected))
    if area == 0:
        return 0.0, math.inf
    depth = np.asarray(frame.depth_mm, dtype=np.float32)
    reliable = (
        np.asarray(frame.reliable_depth, dtype=bool)
        & np.isfinite(depth)
        & (depth > 0.0)
    )
    coverage = float(np.count_nonzero(selected & reliable) / area)
    lab = cv2.cvtColor(frame.image_bgr, cv2.COLOR_BGR2LAB)
    lightness = float(np.median(lab[..., 0][selected]))
    return coverage, lightness


def _track_candidate_by_frame(
    track: FastSAMDISTrack,
    result: FastSAMDISTrackingResult,
) -> dict[int, FastSAMRGBDCandidate]:
    selected: dict[int, FastSAMRGBDCandidate] = {}
    stable_ids = set(int(value) for value in track.stable_candidate_ids)
    for candidate_id in track.candidate_ids:
        if int(candidate_id) not in stable_ids:
            continue
        candidate = result.candidate_by_id.get(int(candidate_id))
        if candidate is None:
            raise ValueError(
                "Stable DIS track references an unknown FastSAM candidate"
            )
        if candidate.frame_id in selected:
            raise ValueError(
                "Stable DIS track has multiple candidates in one frame"
            )
        selected[int(candidate.frame_id)] = candidate
    return selected


def _middle_yellow_shelf_band(
    image_bgr: np.ndarray,
    *,
    config: ShelfInventoryOwnerConfig,
) -> tuple[int, int] | None:
    """Measure the dominant horizontal yellow shelf run in one real RGB frame."""

    image = np.asarray(image_bgr)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Shelf inventory RGB frame is invalid")
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Fixed physical colour gate.  It is deliberately broad enough for the
    # fixed-exposure greenhouse captures and is never estimated from the
    # external visual reference.
    yellow = (
        (hsv[..., 0] >= 10)
        & (hsv[..., 0] <= 45)
        & (hsv[..., 1] >= 55)
        & (hsv[..., 2] >= 40)
    )
    row_coverage = np.mean(yellow, axis=1)
    eligible = (
        (row_coverage >= config.minimum_yellow_row_coverage_ratio)
        & (
            np.arange(image.shape[0], dtype=np.int32)
            >= int(round(0.30 * image.shape[0]))
        )
    )
    best: tuple[int, int] | None = None
    start: int | None = None
    for row in range(eligible.size + 1):
        active = row < eligible.size and bool(eligible[row])
        if active and start is None:
            start = row
        if not active and start is not None:
            run = (start, row)
            if (
                run[1] - run[0]
                >= int(config.minimum_yellow_shelf_run_pixels)
                and (
                    best is None
                    or run[1] - run[0] > best[1] - best[0]
                    or (
                        run[1] - run[0] == best[1] - best[0]
                        and run[0] < best[0]
                    )
                )
            ):
                best = run
            start = None
    return best


def _mask_containment_ratio(first: np.ndarray, second: np.ndarray) -> float:
    a = np.asarray(first, dtype=bool)
    b = np.asarray(second, dtype=bool)
    if a.shape != b.shape:
        raise ValueError("Shelf inventory masks must be image-aligned")
    smaller = min(int(np.count_nonzero(a)), int(np.count_nonzero(b)))
    if smaller == 0:
        return 0.0
    return float(np.count_nonzero(a & b) / smaller)


def _select_panel_track(
    panels: Sequence[OCRSeededPanel],
) -> tuple[int, tuple[OCRSeededPanel, ...]] | None:
    tracks = track_ocr_seeded_panels(panels)
    if not tracks:
        return None
    ranked = sorted(
        enumerate(tracks),
        key=lambda item: (
            len(item[1]),
            float(
                np.median(
                    [panels[index].clarity_variance for index in item[1]]
                )
            ),
            -min(panels[index].source_index for index in item[1]),
        ),
        reverse=True,
    )
    track_ordinal, indices = ranked[0]
    return int(track_ordinal), tuple(panels[index] for index in indices)


def _stable_track_evidence(
    *,
    stable_tracks: Sequence[FastSAMDISTrack],
    tracking: FastSAMDISTrackingResult,
    panels_by_frame: dict[int, OCRSeededPanel],
    frames_by_id: dict[int, InspectionIdentityOwnerFrame],
    config: InspectionIdentityOwnerPlannerConfig,
) -> tuple[
    tuple[StableObjectTrackEvidence, ...],
    dict[int, dict[int, FastSAMRGBDCandidate]],
]:
    evidence: list[StableObjectTrackEvidence] = []
    observations: dict[int, dict[int, FastSAMRGBDCandidate]] = {}
    for track in stable_tracks:
        candidates = _track_candidate_by_frame(track, tracking)
        adjacent: list[FastSAMRGBDCandidate] = []
        clarities: list[float] = []
        for frame_id, candidate in candidates.items():
            panel = panels_by_frame.get(frame_id)
            frame = frames_by_id.get(frame_id)
            if panel is None or frame is None:
                continue
            if int(candidate.source_index) != int(frame.source_index):
                raise ValueError(
                    "Stable DIS candidate does not match the selected source"
                )
            if not _bbox_adjacent(
                candidate.bbox_xywh,
                panel.bbox_xywh,
                maximum_gap=max(
                    config.maximum_source_adjacency_gap_pixels,
                    int(round(0.25 * frame.image_bgr.shape[1])),
                ),
                minimum_vertical_overlap_ratio=(
                    config.minimum_vertical_overlap_ratio
                ),
            ):
                continue
            adjacent.append(candidate)
            clarities.append(_candidate_clarity(candidate, frame.image_bgr))
        if adjacent:
            observations[int(track.track_id)] = {
                int(item.frame_id): item for item in adjacent
            }
            measured = [
                _measured_mask_properties(
                    candidate_mask(
                        item,
                        frames_by_id[int(item.frame_id)].depth_mm.shape,
                    ),
                    frames_by_id[int(item.frame_id)],
                )
                for item in adjacent
            ]
            evidence.append(
                StableObjectTrackEvidence(
                    track_id=int(track.track_id),
                    observation_count=int(track.observation_count),
                    selected_panel_observation_count=len(adjacent),
                    common_frame_ids=tuple(
                        sorted(int(item.frame_id) for item in adjacent)
                    ),
                    median_lab_l=float(
                        np.median([item[1] for item in measured])
                    ),
                    clarity_variance=float(np.median(clarities)),
                    minimum_depth_coverage_ratio=float(
                        min(item[0] for item in measured)
                    ),
                    adjacent_to_panel=True,
                )
            )
    return tuple(evidence), observations


def _project_structure(
    points_world_mm: np.ndarray,
    *,
    layout: InspectionMultiviewLayout,
    intrinsics: CameraIntrinsics,
    panel_index: int,
    panel_valid_mask: np.ndarray,
    minimum_sample_count: int,
) -> _ProjectedStructure | None:
    points = np.asarray(points_world_mm, dtype=np.float64)
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or points.shape[0] < minimum_sample_count
        or not np.isfinite(points).all()
    ):
        return None
    panel = layout.panels[panel_index]
    relative = points - np.asarray(panel.center_world_mm, dtype=np.float64)
    q_scan = relative @ np.asarray(layout.scan_axis, dtype=np.float64)
    q_down = relative @ np.asarray(layout.down_axis, dtype=np.float64)
    q_normal = relative @ np.asarray(layout.normal_axis, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        x = (
            float(panel.canvas_offset_x)
            + intrinsics.cx
            + intrinsics.fx * q_scan / q_normal
        )
        y = (
            float(getattr(layout, "canvas_offset_y", 0.0))
            + intrinsics.cy
            + intrinsics.fy * q_down / q_normal
        )
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(q_normal)
    rounded_x = np.zeros(points.shape[0], dtype=np.int64)
    rounded_y = np.zeros(points.shape[0], dtype=np.int64)
    rounded_x[finite] = np.rint(x[finite]).astype(np.int64)
    rounded_y[finite] = np.rint(y[finite]).astype(np.int64)
    geometric = (
        finite
        & (q_normal > 0.0)
        & (rounded_x >= 0)
        & (rounded_x < layout.width)
        & (rounded_y >= 0)
        & (rounded_y < layout.height)
    )
    valid = geometric.copy()
    valid_indices = np.flatnonzero(geometric)
    valid[valid_indices] &= panel_valid_mask[
        rounded_y[valid_indices], rounded_x[valid_indices]
    ]
    valid_count = int(np.count_nonzero(valid))
    ratio = float(valid_count / points.shape[0])
    if valid_count < minimum_sample_count:
        return None
    footprint = np.zeros((layout.height, layout.width), dtype=np.uint8)
    footprint[rounded_y[valid], rounded_x[valid]] = 1
    # World samples arrive on a fixed two-pixel source stride.  Close only
    # the one-pixel sampling gaps; never replace the measured non-convex
    # silhouette with a convex hull.  Larger holes (fan openings, handles,
    # gaps between object parts) therefore stay transparent and cannot become
    # a pasted polygon of background colour.
    footprint = cv2.morphologyEx(
        footprint,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
    )
    footprint_bool = np.ascontiguousarray(footprint > 0)
    if np.any(footprint_bool & ~panel_valid_mask):
        return None
    return _ProjectedStructure(
        footprint=footprint_bool,
        x_span=(float(np.min(x[valid])), float(np.max(x[valid]))),
        y_span=(float(np.min(y[valid])), float(np.max(y[valid]))),
        sample_count=int(points.shape[0]),
        valid_sample_count=valid_count,
        in_bounds_ratio=ratio,
    )


def _row_contiguous_lock(
    footprint: np.ndarray,
    *,
    margin: int,
) -> np.ndarray:
    yy, xx = np.nonzero(footprint)
    if xx.size == 0:
        return np.zeros_like(footprint, dtype=bool)
    x0 = max(0, int(np.min(xx)) - margin)
    x1 = min(footprint.shape[1], int(np.max(xx)) + margin + 1)
    y0 = max(0, int(np.min(yy)) - margin)
    y1 = min(footprint.shape[0], int(np.max(yy)) + margin + 1)
    lock = np.zeros_like(footprint, dtype=bool)
    lock[y0:y1, x0:x1] = True
    return np.ascontiguousarray(lock)


def _group_track_id(
    *,
    panel_track_ordinal: int,
    object_track_ids: Sequence[int],
) -> int:
    payload = ",".join(
        str(value)
        for value in (panel_track_ordinal, *sorted(object_track_ids))
    ).encode("ascii")
    digest = hashlib.blake2s(payload, digest_size=4).digest()
    return int.from_bytes(digest, "little") & 0x7FFF_FFFF


def _empty_plan(
    *,
    reason: str,
    audit: dict[str, object],
) -> InspectionIdentityOwnerPlan:
    return InspectionIdentityOwnerPlan(
        intervals=(),
        foreground_owners=(),
        audit={
            "schema": "inspection-identity-owner-planner/v1",
            "policy": (
                "aligned_rgbd_true_pose_stable_identity_single_real_"
                "reference_panel_owner"
            ),
            **audit,
            "pass": False,
            "rejection_reason": reason,
            "interval_count": 0,
            "rgb_modified": False,
            "depth_modified": False,
            "pose_modified": False,
            "flow_used_to_warp_rgb_or_position": False,
        },
    )


def plan_inspection_identity_owner_intervals(
    *,
    frames: Sequence[InspectionIdentityOwnerFrame],
    tracking: FastSAMDISTrackingResult,
    layout: InspectionMultiviewLayout,
    intrinsics: CameraIntrinsics,
    ocr_seeded_panels: Sequence[OCRSeededPanel] = (),
    config: InspectionIdentityOwnerPlannerConfig | None = None,
) -> InspectionIdentityOwnerPlan:
    """Plan at most one audited object-rich single-panel owner interval.

    OCR evidence is optional at the API boundary.  The current formal planner
    deliberately returns an empty fail-closed plan when no stable OCR-seeded
    panel anchor exists; it never guesses an object group from proposal IDs.
    """

    selected = config or InspectionIdentityOwnerPlannerConfig()
    if selected.required_neighbor_track_count < 2:
        raise ValueError("Object-rich owner planning requires two neighbours")
    if selected.footprint_sample_stride <= 0:
        raise ValueError("Footprint sample stride must be positive")
    if tracking.flow_used_to_warp_rgb_or_position:
        raise ValueError(
            "Inspection identity flow must not warp RGB or object position"
        )
    frames_by_id = _validate_frames(
        frames, layout=layout, intrinsics=intrinsics
    )
    if not frames_by_id:
        return _empty_plan(
            reason="no_selected_reference_panel_frames",
            audit={"selected_reference_panel_frame_count": 0},
        )
    if not ocr_seeded_panels:
        return _empty_plan(
            reason="stable_ocr_seeded_panel_anchor_unavailable",
            audit={
                "selected_reference_panel_frame_count": len(frames_by_id),
                "ocr_seeded_panel_count": 0,
            },
        )
    panel_selection = _select_panel_track(ocr_seeded_panels)
    if panel_selection is None:
        return _empty_plan(
            reason="stable_ocr_seeded_panel_anchor_unavailable",
            audit={
                "selected_reference_panel_frame_count": len(frames_by_id),
                "ocr_seeded_panel_count": len(ocr_seeded_panels),
            },
        )
    panel_track_ordinal, panel_track = panel_selection
    for panel in panel_track:
        frame = frames_by_id.get(int(panel.frame_id))
        if frame is not None and int(panel.source_index) != int(
            frame.source_index
        ):
            raise ValueError(
                "OCR-seeded panel does not match the selected source"
            )
        if frame is not None and np.asarray(panel.mask).shape != (
            intrinsics.height,
            intrinsics.width,
        ):
            raise ValueError(
                "OCR-seeded panel mask does not match the selected RGB-D frame"
            )
    panels_by_frame = {
        int(panel.frame_id): panel
        for panel in panel_track
        if int(panel.frame_id) in frames_by_id
    }
    if len(panels_by_frame) < 2:
        return _empty_plan(
            reason="ocr_panel_track_has_fewer_than_two_selected_reference_frames",
            audit={
                "selected_reference_panel_frame_count": len(frames_by_id),
                "ocr_seeded_panel_count": len(ocr_seeded_panels),
                "selected_ocr_panel_observation_count": len(panels_by_frame),
            },
        )
    evidence, observations = _stable_track_evidence(
        stable_tracks=tracking.stable_tracks,
        tracking=tracking,
        panels_by_frame=panels_by_frame,
        frames_by_id=frames_by_id,
        config=selected,
    )
    object_track_ids = select_object_rich_neighbor_tracks(
        evidence,
        required_track_count=selected.required_neighbor_track_count,
    )
    if not object_track_ids:
        return _empty_plan(
            reason="fewer_than_required_stable_adjacent_object_tracks",
            audit={
                "selected_reference_panel_frame_count": len(frames_by_id),
                "selected_ocr_panel_observation_count": len(panels_by_frame),
                "stable_dis_track_count": len(tracking.stable_tracks),
                "eligible_adjacent_track_count": len(evidence),
                "eligible_adjacent_tracks": [
                    {
                        "track_id": int(item.track_id),
                        "observation_count": int(item.observation_count),
                        "selected_panel_observation_count": int(
                            item.selected_panel_observation_count
                        ),
                        "common_frame_ids": list(item.common_frame_ids),
                        "median_lab_l": float(item.median_lab_l),
                        "clarity_variance": float(
                            item.clarity_variance
                        ),
                        "minimum_depth_coverage_ratio": float(
                            item.minimum_depth_coverage_ratio
                        ),
                        "adjacent_to_panel": bool(
                            item.adjacent_to_panel
                        ),
                    }
                    for item in evidence
                ],
            },
        )
    common_frames = sorted(
        set(panels_by_frame)
        & set.intersection(
            *[set(observations[track_id]) for track_id in object_track_ids]
        )
    )
    candidates: list[
        tuple[
            float,
            int,
            InspectionPreSeamHardOwnerInterval,
            tuple[InspectionForegroundIdentityOwner, ...],
            dict[str, object],
        ]
    ] = []
    rejected_rows: list[dict[str, object]] = []
    for frame_id in common_frames:
        frame = frames_by_id[frame_id]
        panel = panels_by_frame[frame_id]
        structure_masks = [np.asarray(panel.mask, dtype=bool)]
        structure_depth_coverage: list[float] = []
        for track_id in object_track_ids:
            candidate = observations[track_id][frame_id]
            structure_masks.append(
                candidate_mask(candidate, frame.depth_mm.shape)
            )
        structure_depth_coverage = [
            _measured_mask_properties(mask, frame)[0]
            for mask in structure_masks
        ]
        points = [
            sample_mask_world_points(
                mask=mask,
                depth_mm=frame.depth_mm,
                reliable_depth=frame.reliable_depth,
                camera_to_world=frame.camera_to_world,
                intrinsics=intrinsics,
                stride=selected.footprint_sample_stride,
            )
            for mask in structure_masks
        ]
        projections = [
            _project_structure(
                item,
                layout=layout,
                intrinsics=intrinsics,
                panel_index=int(frame.panel_index),
                panel_valid_mask=frame.panel_valid_mask,
                minimum_sample_count=selected.minimum_projected_sample_count,
            )
            for item in points
        ]
        projection_ratios = [
            0.0 if item is None else item.in_bounds_ratio
            for item in projections
        ]
        spans = [
            (0.0, 0.0) if item is None else item.x_span
            for item in projections
        ]
        interval_audit = audit_object_rich_interval(
            projected_x_spans=spans,
            projected_in_bounds_ratios=projection_ratios,
            depth_coverage_ratios=structure_depth_coverage,
            source_width_pixels=intrinsics.width,
            maximum_gap_pixels=selected.maximum_interval_gap_pixels,
        )
        row: dict[str, object] = {
            "frame_id": int(frame_id),
            "panel_index": int(frame.panel_index),
            "selected_object_track_ids": list(object_track_ids),
            "structure_sample_counts": [
                int(item.shape[0]) for item in points
            ],
            "structure_projected_valid_ratios": projection_ratios,
            "structure_depth_coverage_ratios": structure_depth_coverage,
            "interval_gate": interval_audit,
        }
        if any(item is None for item in projections) or not bool(
            interval_audit["pass"]
        ):
            row.update(
                {
                    "pass": False,
                    "rejection_reason": (
                        "structure_projection_or_interval_gate_failed"
                    ),
                }
            )
            rejected_rows.append(row)
            continue
        projected = [item for item in projections if item is not None]
        union = np.logical_or.reduce([item.footprint for item in projected])
        lock = _row_contiguous_lock(
            union, margin=selected.lock_margin_pixels
        )
        lock_row_widths = np.count_nonzero(lock, axis=1)
        maximum_lock_row_width = int(
            np.max(lock_row_widths, initial=0)
        )
        missing_valid = int(
            np.count_nonzero(lock & ~frame.panel_valid_mask)
        )
        if missing_valid or maximum_lock_row_width > intrinsics.width:
            row.update(
                {
                    "pass": False,
                    "selected_panel_missing_valid_pixel_count": missing_valid,
                    "maximum_lock_row_width_pixels": maximum_lock_row_width,
                    "rejection_reason": (
                        "selected_real_panel_lacks_complete_corridor_coverage"
                        if missing_valid
                        else "owner_interval_exceeds_real_source_width"
                    ),
                }
            )
            rejected_rows.append(row)
            continue
        track_id = _group_track_id(
            panel_track_ordinal=panel_track_ordinal,
            object_track_ids=object_track_ids,
        )
        owner_interval = InspectionPreSeamHardOwnerInterval(
            track_id=track_id,
            panel_index=int(frame.panel_index),
            frame_id=int(frame_id),
            lock_mask=np.ascontiguousarray(lock),
            union_footprint=np.ascontiguousarray(union),
        )
        structure_identity_ids: tuple[int | None, ...] = (
            None,
            *(int(value) for value in object_track_ids),
        )
        structure_kinds = (
            "ocr_seeded_panel",
            *(
                "fastsam_stable_dis_track"
                for _ in object_track_ids
            ),
        )
        foreground_owners = tuple(
                InspectionForegroundIdentityOwner(
                    group_id=track_id,
                    structure_id=structure_id,
                    structure_kind=kind,
                    identity_track_id=identity_id,
                    panel_index=int(frame.panel_index),
                    target_panel_index=int(frame.panel_index),
                    frame_id=int(frame_id),
                    source_index=int(frame.source_index),
                    source_mask=np.ascontiguousarray(mask.copy()),
                    target_footprint=np.ascontiguousarray(
                        projection.footprint.copy()
                    ),
                    measured_depth_coverage_ratio=float(depth_coverage),
                    projected_in_bounds_ratio=float(
                        projection.in_bounds_ratio
                    ),
                )
                for structure_id, (
                    kind,
                    identity_id,
                    mask,
                    projection,
                    depth_coverage,
                ) in enumerate(zip(
                    structure_kinds,
                    structure_identity_ids,
                    structure_masks,
                    projected,
                    structure_depth_coverage,
                    strict=True,
                ))
        )
        clarity = min(
            _candidate_clarity(
                observations[track_id_value][frame_id],
                frame.image_bgr,
            )
            for track_id_value in object_track_ids
        )
        score = float(
            min(projection_ratios)
            + 0.01 * math.log1p(max(0.0, clarity))
        )
        row.update(
            {
                "pass": True,
                "rejection_reason": None,
                "automatic_selection_score": score,
                "lock_pixel_count": int(np.count_nonzero(lock)),
                "union_footprint_pixel_count": int(np.count_nonzero(union)),
                "selected_panel_missing_valid_pixel_count": 0,
                "maximum_lock_row_width_pixels": maximum_lock_row_width,
                "row_contiguous": True,
            }
        )
        candidates.append(
            (
                score,
                -frame_id,
                owner_interval,
                foreground_owners,
                row,
            )
        )
    if not candidates:
        return _empty_plan(
            reason="no_complete_single_real_panel_owner_corridor",
            audit={
                "selected_reference_panel_frame_count": len(frames_by_id),
                "selected_ocr_panel_observation_count": len(panels_by_frame),
                "selected_object_track_ids": list(object_track_ids),
                "common_candidate_frame_ids": common_frames,
                "candidate_audits": rejected_rows,
            },
        )
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    _, _, interval, foreground_owners, accepted_row = candidates[0]
    return InspectionIdentityOwnerPlan(
        intervals=(interval,),
        foreground_owners=foreground_owners,
        audit={
            "schema": "inspection-identity-owner-planner/v1",
            "policy": (
                "aligned_rgbd_true_pose_stable_identity_single_real_"
                "reference_panel_owner"
            ),
            "selected_reference_panel_frame_count": len(frames_by_id),
            "ocr_seeded_panel_count": len(ocr_seeded_panels),
            "selected_ocr_panel_observation_count": len(panels_by_frame),
            "stable_dis_track_count": len(tracking.stable_tracks),
            "selected_object_track_ids": list(object_track_ids),
            "common_candidate_frame_ids": common_frames,
            "selected_candidate": accepted_row,
            "rejected_candidate_audits": rejected_rows,
            "pass": True,
            "rejection_reason": None,
            "interval_count": 1,
            "foreground_identity_group_count": 1,
            "foreground_identity_owner_count": len(foreground_owners),
            "foreground_identity_structure_count": len(foreground_owners),
            "foreground_identity_policy": (
                "independent_source_masks_and_independent_true_rgbd_"
                "target_footprints_without_cross_structure_fill"
            ),
            "rgb_modified": False,
            "depth_modified": False,
            "pose_modified": False,
            "flow_used_to_warp_rgb_or_position": False,
            "delivery_grade_ceiling": "C",
            "handoff_outcome": "hard_cut_degraded",
        },
    )


def plan_direct_stable_track_identity_owners(
    *,
    frames: Sequence[InspectionIdentityOwnerFrame],
    tracking: FastSAMDISTrackingResult,
    layout: InspectionMultiviewLayout,
    intrinsics: CameraIntrinsics,
    existing_foreground_owners: Sequence[
        InspectionForegroundIdentityOwner
    ] = (),
    config: DirectHandoffConfig | Mapping[str, object] | None = None,
) -> InspectionDirectIdentityOwnerPlan:
    """Plan independent owners for stable FastSAM/DIS tracks without OCR.

    Every proposal mask is reconstructed from the selected source frame,
    sampled through aligned depth and its immutable SE(3), and projected to
    the median selected target panel.  DIS flow remains identity evidence and
    never supplies target geometry.
    """

    selected = (
        config
        if isinstance(config, DirectHandoffConfig)
        else DirectHandoffConfig.from_mapping(config)
    )
    selected.validate()
    if tracking.flow_used_to_warp_rgb_or_position:
        raise ValueError(
            "Inspection identity flow must not warp RGB or object position"
        )
    frames_by_id = _validate_frames(
        frames, layout=layout, intrinsics=intrinsics
    )
    frame_by_panel = {
        int(frame.panel_index): frame for frame in frames_by_id.values()
    }
    stable_track_ids = [
        int(track.track_id) for track in tracking.stable_tracks
    ]
    if len(stable_track_ids) != len(set(stable_track_ids)):
        raise ValueError("Stable DIS track IDs must be unique")
    existing_occupied = np.zeros(
        (layout.height, layout.width), dtype=bool
    )
    existing_rows: list[dict[str, object]] = []
    for owner in existing_foreground_owners:
        target = np.asarray(owner.target_footprint)
        source = np.asarray(owner.source_mask)
        if (
            target.dtype != np.bool_
            or target.shape != existing_occupied.shape
            or source.dtype != np.bool_
            or source.shape != (intrinsics.height, intrinsics.width)
            or not np.any(target)
            or not np.any(source)
        ):
            raise ValueError(
                "Existing foreground identity owner masks are invalid"
            )
        existing_occupied |= target
        existing_rows.append(
            {
                "group_id": int(owner.group_id),
                "structure_id": int(owner.structure_id),
                "panel_index": int(owner.panel_index),
                "frame_id": int(owner.frame_id),
                "target_pixel_count": int(np.count_nonzero(target)),
            }
        )
    existing_occupied_pixel_count = int(
        np.count_nonzero(existing_occupied)
    )
    direct_occupied = np.zeros_like(existing_occupied)
    runtime_rows: list[
        tuple[
                tuple[int, int, float, int],
                FastSAMDISTrack,
                DirectProjectedObservation,
            dict[int, tuple[
                InspectionIdentityOwnerFrame,
                FastSAMRGBDCandidate,
                np.ndarray,
                _ProjectedStructure,
                float,
            ]],
            dict[str, object],
        ]
    ] = []
    track_audits: list[dict[str, object]] = []
    for track in tracking.stable_tracks:
        candidates_by_frame = _track_candidate_by_frame(track, tracking)
        selected_candidates: list[
            tuple[InspectionIdentityOwnerFrame, FastSAMRGBDCandidate]
        ] = []
        for frame_id, candidate in candidates_by_frame.items():
            frame = frames_by_id.get(int(frame_id))
            if frame is None:
                continue
            if int(candidate.source_index) != int(frame.source_index):
                raise ValueError(
                    "Stable DIS candidate does not match the selected source"
                )
            selected_candidates.append((frame, candidate))
        selected_candidates.sort(
            key=lambda item: (
                int(item[0].panel_index),
                int(item[0].frame_id),
            )
        )
        source_panels = sorted(
            int(frame.panel_index) for frame, _ in selected_candidates
        )
        target_panel_index = (
            source_panels[len(source_panels) // 2]
            if len(source_panels) >= selected.minimum_projection_count
            else -1
        )
        target_frame = frame_by_panel.get(target_panel_index)
        projections: list[DirectProjectedObservation] = []
        projection_runtime: dict[
            int,
            tuple[
                InspectionIdentityOwnerFrame,
                FastSAMRGBDCandidate,
                np.ndarray,
                _ProjectedStructure,
                float,
            ],
        ] = {}
        source_rows: list[dict[str, object]] = []
        projection_rejections: list[dict[str, object]] = []
        for frame, candidate in selected_candidates:
            source_mask = candidate_mask(candidate, frame.depth_mm.shape)
            depth_coverage, _ = _measured_mask_properties(
                source_mask, frame
            )
            source_row = {
                "candidate_id": int(candidate.candidate_id),
                "frame_id": int(candidate.frame_id),
                "source_index": int(candidate.source_index),
                "source_panel_index": int(frame.panel_index),
                "source_mask_pixel_count": int(
                    np.count_nonzero(source_mask)
                ),
                "source_depth_coverage_ratio": depth_coverage,
            }
            source_rows.append(source_row)
            if (
                target_frame is None
                or depth_coverage
                < selected.minimum_source_depth_coverage_ratio
            ):
                projection_rejections.append(
                    {
                        **source_row,
                        "reason": (
                            "automatic_median_target_panel_unavailable"
                            if target_frame is None
                            else "source_depth_coverage_below_fixed_gate"
                        ),
                    }
                )
                continue
            points = sample_mask_world_points(
                mask=source_mask,
                depth_mm=frame.depth_mm,
                reliable_depth=frame.reliable_depth,
                camera_to_world=frame.camera_to_world,
                intrinsics=intrinsics,
                stride=2,
            )
            projection = _project_structure(
                points,
                layout=layout,
                intrinsics=intrinsics,
                panel_index=target_panel_index,
                panel_valid_mask=target_frame.panel_valid_mask,
                minimum_sample_count=30,
            )
            if projection is None or projection.in_bounds_ratio < 0.90:
                projection_rejections.append(
                    {
                        **source_row,
                        "reason": (
                            "true_rgbd_target_projection_failed_fixed_gate"
                        ),
                        "projected_in_bounds_ratio": (
                            0.0
                            if projection is None
                            else float(projection.in_bounds_ratio)
                        ),
                    }
                )
                continue
            clarity = _candidate_clarity(candidate, frame.image_bgr)
            observation = DirectProjectedObservation(
                candidate_id=int(candidate.candidate_id),
                frame_id=int(candidate.frame_id),
                source_panel_index=int(frame.panel_index),
                target_panel_index=target_panel_index,
                target_mask=np.ascontiguousarray(
                    projection.footprint.copy()
                ),
                target_image_bgr=np.asarray(frame.image_bgr),
                source_depth_coverage_ratio=depth_coverage,
                clarity=clarity,
                projection_audit={
                    "method": (
                        "source_mask_aligned_depth_true_se3_to_automatic_"
                        "median_target_panel"
                    ),
                    "world_sample_count": int(points.shape[0]),
                    "target_pixel_count": int(
                        np.count_nonzero(projection.footprint)
                    ),
                    "projected_in_bounds_ratio": float(
                        projection.in_bounds_ratio
                    ),
                    "target_panel_missing_valid_pixel_count": 0,
                    "rgb_modified": False,
                    "pose_modified": False,
                    "fitted_warp_used": False,
                    "generated_rgb_used": False,
                },
            )
            projections.append(observation)
            projection_runtime[int(candidate.candidate_id)] = (
                frame,
                candidate,
                np.ascontiguousarray(source_mask.copy()),
                projection,
                depth_coverage,
            )
        decision = evaluate_direct_track(
            int(track.track_id), projections, config=selected
        )
        audit = {
            **decision.audit,
            "stable_track_observation_count": int(track.observation_count),
            "stable_selected_panel_observation_count": len(
                selected_candidates
            ),
            "automatic_median_target_panel_index": target_panel_index,
            "source_observations": source_rows,
            "projection_rejections": projection_rejections,
        }
        track_audits.append(audit)
        if decision.accepted and decision.selected_observation is not None:
            score = (
                int(audit["consistent_projection_count"]),
                int(
                    np.count_nonzero(
                        decision.selected_observation.target_mask
                    )
                ),
                float(audit["selected_target_union_coverage_ratio"]),
                -int(track.track_id),
            )
            runtime_rows.append(
                (
                    score,
                    track,
                    decision.selected_observation,
                    projection_runtime,
                    audit,
                )
            )
    runtime_rows.sort(key=lambda item: item[0], reverse=True)
    accepted: list[InspectionForegroundIdentityOwner] = []
    rejected_conflicts: list[dict[str, object]] = []
    for (
        _,
        track,
        selected_observation_value,
        projection_runtime,
        audit,
    ) in runtime_rows:
        observation = selected_observation_value
        candidate_id = int(observation.candidate_id)
        runtime = projection_runtime.get(candidate_id)
        if runtime is None:
            raise RuntimeError(
                "Accepted direct identity observation lacks runtime source"
            )
        frame, candidate, source_mask, projection, depth_coverage = runtime
        target = projection.footprint
        existing_overlap_pixels = int(
            np.count_nonzero(target & existing_occupied)
        )
        direct_overlap_pixels = int(
            np.count_nonzero(target & direct_occupied)
        )
        overlap_pixels = existing_overlap_pixels + direct_overlap_pixels
        direct_overlap_ratio = float(
            direct_overlap_pixels / max(1, np.count_nonzero(target))
        )
        overlap_ratio = float(
            overlap_pixels / max(1, np.count_nonzero(target))
        )
        if (
            existing_overlap_pixels > 0
            or direct_overlap_ratio
            > selected.maximum_track_overlap_ratio
        ):
            audit["accepted"] = False
            audit["reason"] = (
                "direct_target_overlaps_existing_or_prior_identity_owner"
            )
            audit["identity_owner_overlap_pixel_count"] = overlap_pixels
            audit["identity_owner_overlap_ratio"] = overlap_ratio
            audit["existing_identity_owner_overlap_pixel_count"] = (
                existing_overlap_pixels
            )
            audit["prior_direct_identity_owner_overlap_pixel_count"] = (
                direct_overlap_pixels
            )
            audit["prior_direct_identity_owner_overlap_ratio"] = (
                direct_overlap_ratio
            )
            rejected_conflicts.append(audit)
            continue
        owner = InspectionForegroundIdentityOwner(
            group_id=_DIRECT_STABLE_TRACK_GROUP_ID,
            structure_id=int(track.track_id),
            structure_kind="fastsam_stable_dis_track_direct",
            identity_track_id=int(track.track_id),
            panel_index=int(frame.panel_index),
            target_panel_index=int(observation.target_panel_index),
            frame_id=int(candidate.frame_id),
            source_index=int(candidate.source_index),
            source_mask=np.ascontiguousarray(source_mask.copy()),
            target_footprint=np.ascontiguousarray(target.copy()),
            measured_depth_coverage_ratio=float(depth_coverage),
            projected_in_bounds_ratio=float(projection.in_bounds_ratio),
            reference_observation_masks=tuple(
                (
                    int(observation_frame.panel_index),
                    np.ascontiguousarray(observation_mask.copy()),
                )
                for (
                    observation_frame,
                    _,
                    observation_mask,
                    _,
                    _,
                ) in sorted(
                    projection_runtime.values(),
                    key=lambda item: int(item[0].panel_index),
                )
            ),
        )
        direct_occupied |= target
        audit["identity_owner_overlap_pixel_count"] = overlap_pixels
        audit["identity_owner_overlap_ratio"] = overlap_ratio
        audit["existing_identity_owner_overlap_pixel_count"] = 0
        audit["prior_direct_identity_owner_overlap_pixel_count"] = (
            direct_overlap_pixels
        )
        audit["prior_direct_identity_owner_overlap_ratio"] = (
            direct_overlap_ratio
        )
        audit["foreground_identity_owner_emitted"] = True
        accepted.append(owner)
    accepted_ids = {
        int(owner.identity_track_id)
        for owner in accepted
        if owner.identity_track_id is not None
    }
    public_tracks = [
        {
            **audit,
            "foreground_identity_owner_emitted": bool(
                int(audit["track_id"]) in accepted_ids
            ),
        }
        for audit in track_audits
    ]
    passed = bool(accepted)
    return InspectionDirectIdentityOwnerPlan(
        foreground_owners=tuple(accepted),
        audit={
            "schema": "inspection-direct-identity-owner-planner/v1",
            "policy": (
                "stable_fastsam_dis_identity_true_rgbd_direct_single_"
                "source_owner_without_ocr"
            ),
            "selected_reference_panel_frame_count": len(frames_by_id),
            "stable_track_count": len(tracking.stable_tracks),
            "existing_foreground_identity_owner_count": len(
                existing_foreground_owners
            ),
            "existing_foreground_identity_owner_pixel_count": int(
                existing_occupied_pixel_count
            ),
            "existing_foreground_identity_owners": existing_rows,
            "track_audits": public_tracks,
            "conflict_rejections": rejected_conflicts,
            "accepted_owner_count": len(accepted),
            "accepted_track_ids": [
                int(owner.identity_track_id)
                for owner in accepted
                if owner.identity_track_id is not None
            ],
            "ranking": (
                "consistent_projection_count_then_selected_target_area_"
                "then_union_coverage"
            ),
            "maximum_cross_track_target_overlap_ratio": float(
                selected.maximum_track_overlap_ratio
            ),
            "pass": passed,
            "rejection_reason": (
                None
                if passed
                else "no_direct_identity_owner_passed_fixed_gates"
            ),
            "rgb_modified": False,
            "depth_modified": False,
            "pose_modified": False,
            "flow_used_to_warp_rgb_or_position": False,
        },
    )


def plan_middle_shelf_inventory_identity_owners(
    *,
    frames: Sequence[InspectionIdentityOwnerFrame],
    tracking: FastSAMDISTrackingResult,
    layout: InspectionMultiviewLayout,
    intrinsics: CameraIntrinsics,
    config: ShelfInventoryOwnerConfig | None = None,
) -> InspectionShelfInventoryOwnerPlan:
    """Inventory complete stable objects on the measured yellow shelf.

    A track is eligible only when at least one selected real frame contains a
    compact, boundary-clear mask whose bottom meets that frame's measured
    yellow shelf band.  One most complete and centred real RGB-D observation
    is selected, then projected by aligned depth and immutable SE(3) to that
    same real source panel.  This keeps a complete unchanged reference-raster
    fallback available when the depth boundary contains holes.  Cross-track
    removal is limited to strict mask containment, so neighbouring objects
    remain independent owners.
    """

    selected = ShelfInventoryOwnerConfig() if config is None else config
    selected.validate()
    if tracking.flow_used_to_warp_rgb_or_position:
        raise ValueError(
            "Shelf inventory DIS flow must not warp RGB or object position"
        )
    frames_by_id = _validate_frames(
        frames, layout=layout, intrinsics=intrinsics
    )
    frame_by_panel = {
        int(frame.panel_index): frame for frame in frames_by_id.values()
    }
    shelf_bands = {
        int(frame.frame_id): _middle_yellow_shelf_band(
            frame.image_bgr, config=selected
        )
        for frame in frames_by_id.values()
    }
    runtimes: dict[int, dict[str, object]] = {}
    disposition_by_track: dict[int, dict[str, object]] = {}
    for track in tracking.stable_tracks:
        track_id = int(track.track_id)
        candidates_by_frame = _track_candidate_by_frame(track, tracking)
        observations: list[dict[str, object]] = []
        eligible: list[dict[str, object]] = []
        all_selected_masks: dict[int, np.ndarray] = {}
        for frame_id, candidate in sorted(candidates_by_frame.items()):
            frame = frames_by_id.get(int(frame_id))
            if frame is None:
                continue
            if int(candidate.source_index) != int(frame.source_index):
                raise ValueError(
                    "Shelf inventory candidate/source mapping is invalid"
                )
            mask = candidate_mask(candidate, frame.depth_mm.shape)
            all_selected_masks[int(frame_id)] = np.ascontiguousarray(mask)
            x, y, width, height = candidate.bbox_xywh
            x1, y1 = x + width, y + height
            area = int(np.count_nonzero(mask))
            bbox_area = int(width * height)
            bbox_area_ratio = float(
                bbox_area / max(1, intrinsics.width * intrinsics.height)
            )
            fill_ratio = float(area / max(1, bbox_area))
            aspect = float(width / max(1, height))
            margin = int(selected.source_boundary_margin_pixels)
            boundary_clear = bool(
                x >= margin
                and y >= margin
                and x1 <= intrinsics.width - margin
                and y1 <= intrinsics.height - margin
            )
            depth_coverage, _ = _measured_mask_properties(mask, frame)
            shelf_band = shelf_bands[int(frame_id)]
            contact = False
            if shelf_band is not None:
                tolerance = int(
                    round(
                        selected.shelf_contact_tolerance_ratio
                        * intrinsics.height
                    )
                )
                shelf_top, shelf_bottom = shelf_band
                contact = bool(
                    shelf_top - tolerance <= y1 <= shelf_bottom + tolerance
                    and y + 0.5 * height < shelf_top + tolerance
                )
            gates = {
                "selected_reference_frame": True,
                # A later ambiguous merge/split terminates propagation, but
                # it does not invalidate the already accepted stable
                # pre-termination observations.  Requiring a never-terminated
                # track discarded the charger, fan, white box, and most of
                # the shelf inventory even though each had several complete
                # measured RGB-D views.
                "stable_pretermination_observations_available": bool(
                    len(candidates_by_frame) >= 2
                ),
                "source_boundary_clear": boundary_clear,
                "bbox_area_within_limit": bool(
                    bbox_area_ratio <= selected.maximum_bbox_area_ratio
                ),
                "mask_bbox_fill_pass": bool(
                    fill_ratio >= selected.minimum_mask_bbox_fill_ratio
                ),
                "candidate_solidity_pass": bool(
                    float(candidate.solidity)
                    >= selected.minimum_candidate_solidity
                ),
                "bbox_aspect_pass": bool(
                    selected.minimum_bbox_aspect_ratio
                    <= aspect
                    <= selected.maximum_bbox_aspect_ratio
                ),
                "depth_coverage_pass": bool(
                    depth_coverage
                    >= selected.minimum_source_depth_coverage_ratio
                ),
                "measured_yellow_shelf_contact_pass": contact,
            }
            accepted = bool(all(gates.values()))
            centre_dx = abs(
                (x + 0.5 * width - intrinsics.cx)
                / max(1.0, 0.5 * intrinsics.width)
            )
            centre_dy = abs(
                (y + 0.5 * height - intrinsics.cy)
                / max(1.0, 0.5 * intrinsics.height)
            )
            boundary_margin = float(
                min(x, y, intrinsics.width - x1, intrinsics.height - y1)
            )
            clarity = _candidate_clarity(candidate, frame.image_bgr)
            rank = (
                boundary_margin / max(intrinsics.width, intrinsics.height),
                1.0 - min(1.0, centre_dx),
                1.0 - min(1.0, centre_dy),
                depth_coverage,
                area,
                clarity,
                -int(frame.panel_index),
            )
            row = {
                "candidate_id": int(candidate.candidate_id),
                "frame_id": int(frame_id),
                "source_panel_index": int(frame.panel_index),
                "source_bbox_xywh": [int(x), int(y), int(width), int(height)],
                "source_mask_pixel_count": area,
                "bbox_area_ratio": bbox_area_ratio,
                "mask_bbox_fill_ratio": fill_ratio,
                "candidate_solidity": float(candidate.solidity),
                "bbox_aspect_ratio": aspect,
                "source_depth_coverage_ratio": depth_coverage,
                "measured_yellow_shelf_band_y": (
                    None
                    if shelf_band is None
                    else [int(shelf_band[0]), int(shelf_band[1])]
                ),
                "gates": gates,
                "track_later_merge_split_terminated": bool(
                    track.merge_split_terminated
                ),
                "eligible_complete_shelf_observation": accepted,
                "selection_rank": [float(value) for value in rank[:-1]]
                + [int(rank[-1])],
            }
            observations.append(row)
            if accepted:
                eligible.append(
                    {
                        "rank": rank,
                        "frame": frame,
                        "candidate": candidate,
                        "mask": mask,
                        "depth_coverage": depth_coverage,
                    }
                )
        if not eligible:
            disposition_by_track[track_id] = {
                "track_id": track_id,
                "stable_observation_count": int(track.observation_count),
                "selected_reference_observation_count": len(observations),
                "inventory_disposition": "excluded_not_complete_shelf_object",
                "included_in_inventory": False,
                "mesh_preflight_required": False,
                "observations": observations,
            }
            continue
        eligible.sort(key=lambda item: item["rank"], reverse=True)
        source = eligible[0]
        source_frame = source["frame"]
        source_mask = np.asarray(source["mask"], dtype=bool)
        points = sample_mask_world_points(
            mask=source_mask,
            depth_mm=source_frame.depth_mm,
            reliable_depth=source_frame.reliable_depth,
            camera_to_world=source_frame.camera_to_world,
            intrinsics=intrinsics,
            stride=2,
        )
        projected: list[
            tuple[
                tuple[float, float, int],
                int,
                _ProjectedStructure,
            ]
        ] = []
        # The inventory fallback must remain an unchanged raster from the
        # selected real source panel when depth edges contain holes.  Keep the
        # spatial panel equal to that source panel; this is still positioned
        # by the real camera pose/layout and avoids inventing a cross-panel
        # reference-plane transfer.
        source_panel_index = int(source_frame.panel_index)
        for panel_index, target_frame in (
            (source_panel_index, frame_by_panel[source_panel_index]),
        ):
            projection = _project_structure(
                points,
                layout=layout,
                intrinsics=intrinsics,
                panel_index=panel_index,
                panel_valid_mask=target_frame.panel_valid_mask,
                minimum_sample_count=30,
            )
            if projection is None:
                continue
            yy, xx = np.nonzero(projection.footprint)
            target_cx = (
                float(layout.panels[panel_index].canvas_offset_x)
                + intrinsics.cx
            )
            target_cy = (
                float(getattr(layout, "canvas_offset_y", 0.0))
                + intrinsics.cy
            )
            centre_distance = float(
                abs(float(np.median(xx)) - target_cx)
                / max(1.0, intrinsics.width)
                + abs(float(np.median(yy)) - target_cy)
                / max(1.0, intrinsics.height)
            )
            projected.append(
                (
                    (
                        float(projection.in_bounds_ratio),
                        -centre_distance,
                        -int(panel_index),
                    ),
                    int(panel_index),
                    projection,
                )
            )
        projected.sort(key=lambda item: item[0], reverse=True)
        if not projected or projected[0][2].in_bounds_ratio < 0.90:
            disposition_by_track[track_id] = {
                "track_id": track_id,
                "stable_observation_count": int(track.observation_count),
                "selected_reference_observation_count": len(observations),
                "inventory_disposition": (
                    "excluded_true_rgbd_spatial_projection_incomplete"
                ),
                "included_in_inventory": False,
                "mesh_preflight_required": False,
                "observations": observations,
            }
            continue
        _, target_panel_index, projection = projected[0]
        runtimes[track_id] = {
            "track": track,
            "source": source,
            "projection": projection,
            "target_panel_index": target_panel_index,
            "all_selected_masks": all_selected_masks,
            "observations": observations,
            "median_area": float(
                np.median(
                    [
                        np.count_nonzero(value)
                        for value in all_selected_masks.values()
                    ]
                )
            ),
        }
        disposition_by_track[track_id] = {
            "track_id": track_id,
            "stable_observation_count": int(track.observation_count),
            "selected_reference_observation_count": len(observations),
            "inventory_disposition": "inventory_owner_candidate",
            "included_in_inventory": True,
            "mesh_preflight_required": True,
            "selected_frame_id": int(source_frame.frame_id),
            "selected_source_panel_index": int(source_frame.panel_index),
            "selected_target_panel_index": int(target_panel_index),
            "selection_policy": (
                "boundary_completeness_then_optical_centre_then_depth_"
                "coverage_area_and_clarity"
            ),
            "projected_in_bounds_ratio": float(
                projection.in_bounds_ratio
            ),
            "observations": observations,
        }

    # Remove only segmentation hierarchy duplicates.  Contacting or adjacent
    # masks are not merged: the smaller mask must be almost wholly contained
    # in the other in common source frames or in the common spatial panel.
    suppressed: dict[int, int] = {}
    track_ids = sorted(runtimes)
    for position, first_id in enumerate(track_ids):
        if first_id in suppressed:
            continue
        first = runtimes[first_id]
        for second_id in track_ids[position + 1 :]:
            if second_id in suppressed:
                continue
            second = runtimes[second_id]
            common = sorted(
                set(first["all_selected_masks"])
                & set(second["all_selected_masks"])
            )
            source_containment = (
                float(
                    np.median(
                        [
                            _mask_containment_ratio(
                                first["all_selected_masks"][frame_id],
                                second["all_selected_masks"][frame_id],
                            )
                            for frame_id in common
                        ]
                    )
                )
                if common
                else 0.0
            )
            target_containment = _mask_containment_ratio(
                first["projection"].footprint,
                second["projection"].footprint,
            )
            containment = max(source_containment, target_containment)
            if containment < selected.hierarchy_containment_ratio:
                continue
            first_area = float(first["median_area"])
            second_area = float(second["median_area"])
            if first_area == second_area:
                first_rank = first["source"]["rank"]
                second_rank = second["source"]["rank"]
                keep, remove = (
                    (first_id, second_id)
                    if first_rank >= second_rank
                    else (second_id, first_id)
                )
            else:
                # A FastSAM parent represents the complete object; its nested
                # child is a segmentation hierarchy duplicate, not a separate
                # shelf item.
                keep, remove = (
                    (first_id, second_id)
                    if first_area > second_area
                    else (second_id, first_id)
                )
            suppressed[remove] = keep
            disposition_by_track[remove].update(
                {
                    "inventory_disposition": (
                        "excluded_fastsam_hierarchy_duplicate"
                    ),
                    "included_in_inventory": False,
                    "mesh_preflight_required": False,
                    "hierarchy_parent_track_id": int(keep),
                    "hierarchy_source_containment_ratio": (
                        source_containment
                    ),
                    "hierarchy_target_containment_ratio": (
                        target_containment
                    ),
                }
            )
            if remove == first_id:
                break

    owners: list[InspectionForegroundIdentityOwner] = []
    for track_id in track_ids:
        if track_id in suppressed:
            continue
        runtime = runtimes[track_id]
        source = runtime["source"]
        frame = source["frame"]
        candidate = source["candidate"]
        projection = runtime["projection"]
        owners.append(
            InspectionForegroundIdentityOwner(
                group_id=_SHELF_INVENTORY_GROUP_ID,
                structure_id=int(track_id),
                structure_kind=(
                    "middle_yellow_shelf_stable_object_inventory"
                ),
                identity_track_id=int(track_id),
                panel_index=int(frame.panel_index),
                target_panel_index=int(runtime["target_panel_index"]),
                frame_id=int(candidate.frame_id),
                source_index=int(candidate.source_index),
                source_mask=np.ascontiguousarray(
                    np.asarray(source["mask"], dtype=bool)
                ),
                target_footprint=np.ascontiguousarray(
                    projection.footprint.copy()
                ),
                measured_depth_coverage_ratio=float(
                    source["depth_coverage"]
                ),
                projected_in_bounds_ratio=float(
                    projection.in_bounds_ratio
                ),
                reference_observation_masks=tuple(
                    (
                        int(frames_by_id[frame_id].panel_index),
                        np.ascontiguousarray(mask.copy()),
                    )
                    for frame_id, mask in sorted(
                        runtime["all_selected_masks"].items()
                    )
                ),
            )
        )
    included_ids = [
        int(owner.identity_track_id)
        for owner in owners
        if owner.identity_track_id is not None
    ]
    return InspectionShelfInventoryOwnerPlan(
        foreground_owners=tuple(owners),
        audit={
            "schema": "inspection-shelf-inventory-owner-planner/v1",
            "policy": (
                "stable_fastsam_dis_complete_middle_yellow_shelf_object_"
                "single_real_rgbd_owner"
            ),
            "reference_rgb_or_geometry_used": False,
            "track_ids_hardcoded": False,
            "selected_reference_panel_frame_count": len(frames_by_id),
            "stable_track_count": len(tracking.stable_tracks),
            "measured_yellow_shelf_bands": [
                {
                    "frame_id": int(frame_id),
                    "band_y": (
                        None
                        if band is None
                        else [int(band[0]), int(band[1])]
                    ),
                }
                for frame_id, band in sorted(shelf_bands.items())
            ],
            "inventory_owner_candidate_count": len(owners),
            "inventory_track_ids": included_ids,
            "mesh_preflight_required_track_ids": included_ids,
            "hierarchy_duplicate_count": len(suppressed),
            "hierarchy_duplicate_track_ids": sorted(suppressed),
            "adjacent_object_merge_used": False,
            "track_dispositions": [
                disposition_by_track[int(track.track_id)]
                for track in tracking.stable_tracks
            ],
            "all_stable_tracks_have_disposition": bool(
                len(disposition_by_track) == len(tracking.stable_tracks)
            ),
            "rgb_modified": False,
            "depth_modified": False,
            "pose_modified": False,
            "flow_used_to_warp_rgb_or_position": False,
        },
    )


__all__ = [
    "InspectionDirectIdentityOwnerPlan",
    "InspectionIdentityOwnerFrame",
    "InspectionIdentityOwnerPlan",
    "InspectionIdentityOwnerPlannerConfig",
    "InspectionShelfInventoryOwnerPlan",
    "ShelfInventoryOwnerConfig",
    "plan_direct_stable_track_identity_owners",
    "plan_inspection_identity_owner_intervals",
    "plan_middle_shelf_inventory_identity_owners",
]
