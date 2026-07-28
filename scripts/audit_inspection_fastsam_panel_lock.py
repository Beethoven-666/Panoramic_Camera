"""Adjacent selected-panel FastSAM/RGB-D single-owner lock diagnostic."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
import math
from pathlib import Path
import time

import cv2
import numpy as np

from panorama_demo.inspection_fastsam_track import (
    FastSAMRGBDCandidate,
    build_fastsam_rgbd_candidate,
    parse_fastsam_polygons,
    polygon_mask,
)
from panorama_demo.inspection_multiview import (
    InspectionMultiviewConfig,
    InspectionMultiviewLayout,
    VirtualPerspectivePanel,
    _read_rgbd,
    _reference_panel_inverse_maps,
    _undistortion_maps,
)
from panorama_demo.inspection_object_handoff import (
    build_object_owner_interval,
)
from panorama_demo.session import load_rgbd_session


@dataclass(frozen=True)
class _PanelSource:
    panel_index: int
    source_index: int
    frame_id: int
    image: np.ndarray
    depth: np.ndarray
    reliable: np.ndarray
    pose: np.ndarray
    candidates: tuple[FastSAMRGBDCandidate, ...]


@dataclass(frozen=True)
class _NaturalMask:
    corner_x: int
    local_mask: np.ndarray
    local_valid: np.ndarray
    source_coverage_ratio: float
    clarity: float


def _layout_from_report(value: dict[str, object]) -> InspectionMultiviewLayout:
    return InspectionMultiviewLayout(
        width=int(value["width"]),
        height=int(value["height"]),
        reference_depth_mm=float(value["reference_depth_mm"]),
        scan_axis=tuple(float(item) for item in value["scan_axis_world"]),
        down_axis=tuple(float(item) for item in value["down_axis_world"]),
        normal_axis=tuple(float(item) for item in value["normal_axis_world"]),
        panels=tuple(
            VirtualPerspectivePanel(
                panel_index=int(item["panel_index"]),
                anchor_scan_mm=float(item["anchor_scan_mm"]),
                canvas_offset_x=float(item["canvas_offset_x"]),
                center_world_mm=tuple(
                    float(component) for component in item["center_world_mm"]
                ),
            )
            for item in value["panels"]
        ),
        panel_step_mm=float(value["panel_step_mm"]),
        canvas_megapixels=float(value["canvas_megapixels"]),
    )


def _full_baseline(
    output: Path,
    layout: InspectionMultiviewLayout,
    crop: dict[str, object],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image = cv2.imread(
        str(output / "mosaic_inspection.png"), cv2.IMREAD_COLOR
    )
    encoded = cv2.imread(
        str(output / "inspection_owner.png"), cv2.IMREAD_UNCHANGED
    )
    if (
        image is None
        or encoded is None
        or encoded.dtype != np.uint16
        or image.shape[:2] != encoded.shape
    ):
        raise RuntimeError("Formal inspection baseline is invalid")
    x, y, width, height = (
        int(crop[name]) for name in ("x", "y", "width", "height")
    )
    full_image = np.zeros((layout.height, layout.width, 3), dtype=np.uint8)
    full_owner = np.full((layout.height, layout.width), -1, dtype=np.int32)
    full_valid = np.zeros((layout.height, layout.width), dtype=bool)
    full_image[y : y + height, x : x + width] = image
    decoded = encoded.astype(np.int32) - 1
    full_owner[y : y + height, x : x + width] = decoded
    full_valid[y : y + height, x : x + width] = decoded >= 0
    return full_image, full_owner, full_valid


def _pair_match(
    first: FastSAMRGBDCandidate,
    second: FastSAMRGBDCandidate,
) -> tuple[bool, dict[str, object]]:
    first_in_second = len(
        first.world_voxel_hashes & second.world_dilated_voxel_hashes
    )
    second_in_first = len(
        second.world_voxel_hashes & first.world_dilated_voxel_hashes
    )
    overlap = float(
        min(first_in_second, second_in_first)
        / max(
            1,
            min(
                len(first.world_voxel_hashes),
                len(second.world_voxel_hashes),
            ),
        )
    )
    lab_delta = float(
        np.linalg.norm(
            np.asarray(first.median_lab) - np.asarray(second.median_lab)
        )
    )
    centroid_delta = float(
        np.linalg.norm(
            np.asarray(first.world_centroid_mm)
            - np.asarray(second.world_centroid_mm)
        )
    )
    area_ratio = max(
        first.source_area_pixels, second.source_area_pixels
    ) / max(1, min(first.source_area_pixels, second.source_area_pixels))
    aspect_delta = abs(
        math.log(
            max(1e-6, first.aspect_ratio)
            / max(1e-6, second.aspect_ratio)
        )
    )
    contour_delta = float(
        cv2.matchShapes(
            first.polygon_xy,
            second.polygon_xy,
            cv2.CONTOURS_MATCH_I1,
            0.0,
        )
    )
    accepted = bool(
        overlap >= 0.25
        and lab_delta <= 30.0
        and centroid_delta <= 120.0
        and area_ratio <= 2.5
        and aspect_delta <= 0.80
        and abs(first.solidity - second.solidity) <= 0.35
        and contour_delta <= 0.35
    )
    return accepted, {
        "dilated_world_voxel_overlap_ratio": overlap,
        "median_lab_delta": lab_delta,
        "world_centroid_delta_mm": centroid_delta,
        "source_area_ratio": area_ratio,
        "log_aspect_delta": aspect_delta,
        "solidity_delta": abs(first.solidity - second.solidity),
        "contour_match_i1": contour_delta,
        "accepted": accepted,
    }


def _natural_mask(
    source: _PanelSource,
    candidate: FastSAMRGBDCandidate,
    layout: InspectionMultiviewLayout,
    intrinsics: object,
) -> _NaturalMask | None:
    x0, map_x, map_y, valid, _ = _reference_panel_inverse_maps(
        source_pose=source.pose,
        panel_index=source.panel_index,
        layout=layout,
        intrinsics=intrinsics,
    )
    source_mask = polygon_mask(candidate, source.depth.shape)
    sampled = cv2.remap(
        source_mask.astype(np.uint8),
        map_x,
        map_y,
        cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    target = (sampled > 0) & valid
    if not np.any(target):
        return None
    if (
        np.any(target[0])
        or np.any(target[-1])
        or np.any(target[:, 0])
        or np.any(target[:, -1])
    ):
        return None
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        target.astype(np.uint8), 8
    )
    components = [
        (int(stats[label, cv2.CC_STAT_AREA]), label)
        for label in range(1, count)
    ]
    if len(components) != 1:
        return None
    target = labels == components[0][1]
    yy, xx = np.nonzero(target)
    source_x = np.rint(map_x[yy, xx]).astype(np.int32)
    source_y = np.rint(map_y[yy, xx]).astype(np.int32)
    inside = (
        (source_x >= 0)
        & (source_x < source.depth.shape[1])
        & (source_y >= 0)
        & (source_y < source.depth.shape[0])
    )
    hit = np.zeros(source.depth.shape, dtype=bool)
    hit[source_y[inside], source_x[inside]] = True
    coverage = float(
        np.count_nonzero(hit & source_mask)
        / max(1, np.count_nonzero(source_mask))
    )
    gray = cv2.cvtColor(source.image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_32F)
    clarity = float(np.std(laplacian[source_mask]))
    return _NaturalMask(
        corner_x=int(x0),
        local_mask=np.ascontiguousarray(target),
        local_valid=np.ascontiguousarray(valid),
        source_coverage_ratio=coverage,
        clarity=clarity,
    )


def _full_mask(
    natural: _NaturalMask,
    shape: tuple[int, int],
) -> np.ndarray:
    result = np.zeros(shape, dtype=bool)
    x0 = natural.corner_x
    x1 = x0 + natural.local_mask.shape[1]
    result[:, x0:x1] = natural.local_mask
    return result


def _full_valid(
    natural: _NaturalMask,
    shape: tuple[int, int],
) -> np.ndarray:
    result = np.zeros(shape, dtype=bool)
    x0 = natural.corner_x
    x1 = x0 + natural.local_valid.shape[1]
    result[:, x0:x1] = natural.local_valid
    return result


def _owner_rows_monotonic(owner_panel: np.ndarray) -> bool:
    for row in owner_panel:
        sequence = row[row >= 0]
        if sequence.size and np.any(np.diff(sequence) < 0):
            return False
    return True


def _build_dis_track_proposals(
    *,
    audit_path: Path,
    sources: list[_PanelSource],
    mapped: object,
    baseline_valid: np.ndarray,
    full_owner: np.ndarray,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    if payload.get("schema") != "inspection-fastsam-dis-identity-tracks/v1":
        raise RuntimeError("DIS identity-track audit schema is unsupported")
    source_by_frame = {source.frame_id: source for source in sources}
    proposals: list[dict[str, object]] = []
    rejected: Counter[str] = Counter()
    rejected_tracks: list[dict[str, object]] = []
    considered = 0
    for track in payload.get("stable_selected_panel_tracks", []):
        considered += 1
        if (
            float(track["minimum_flow_mask_iou"]) < 0.75
            or float(track["maximum_fb_p95_preview_pixels"]) > 0.50
            or float(track["maximum_area_ratio"]) > 1.35
        ):
            rejected["identity_stability_gate_failed"] += 1
            rejected_tracks.append(
                {
                    "track_id": int(track["track_id"]),
                    "reason": "identity_stability_gate_failed",
                }
            )
            continue
        observations: list[
            tuple[_PanelSource, FastSAMRGBDCandidate, _NaturalMask]
        ] = []
        ambiguous = False
        for frame_id, bbox in zip(
            track["selected_panel_frame_ids"],
            track["selected_panel_bboxes_xywh"],
            strict=True,
        ):
            source = source_by_frame.get(int(frame_id))
            if source is None:
                ambiguous = True
                break
            matches = [
                candidate
                for candidate in source.candidates
                if list(candidate.bbox_xywh) == list(bbox)
            ]
            if len(matches) != 1:
                ambiguous = True
                break
            natural = mapped(source, matches[0])
            if (
                natural is None
                or natural.source_coverage_ratio < 0.90
            ):
                ambiguous = True
                break
            observations.append((source, matches[0], natural))
        if ambiguous or len(observations) < 2:
            rejected["selected_observation_inverse_map_incomplete"] += 1
            rejected_tracks.append(
                {
                    "track_id": int(track["track_id"]),
                    "reason": (
                        "selected_observation_inverse_map_incomplete"
                    ),
                }
            )
            continue
        footprints = tuple(
            _full_mask(natural, baseline_valid.shape)
            for _, _, natural in observations
        )
        options = []
        for source, candidate, natural in observations:
            try:
                interval = build_object_owner_interval(
                    panel_index=source.panel_index,
                    view_dependent_footprints=footprints,
                    selected_panel_valid_mask=_full_valid(
                        natural, baseline_valid.shape
                    ),
                )
            except (RuntimeError, ValueError):
                continue
            options.append(
                (
                    natural.source_coverage_ratio,
                    natural.clarity,
                    -source.frame_id,
                    source,
                    candidate,
                    natural,
                    interval,
                )
            )
        if not options:
            rejected["no_single_panel_covers_all_track_footprints"] += 1
            rejected_tracks.append(
                {
                    "track_id": int(track["track_id"]),
                    "reason": (
                        "no_single_panel_covers_all_track_footprints"
                    ),
                }
            )
            continue
        (
            coverage,
            clarity,
            _,
            owner_source,
            owner_candidate,
            owner_natural,
            interval,
        ) = max(options, key=lambda item: item[:3])
        baseline_owners = np.unique(
            full_owner[interval.union_footprint & baseline_valid]
        )
        baseline_owners = baseline_owners[baseline_owners >= 0]
        if baseline_owners.size < 2:
            rejected["track_does_not_cross_baseline_handoff"] += 1
            rejected_tracks.append(
                {
                    "track_id": int(track["track_id"]),
                    "reason": "track_does_not_cross_baseline_handoff",
                }
            )
            continue
        lock_y, lock_x = np.nonzero(interval.union_footprint)
        proposals.append(
            {
                "proposal_kind": "stable_dis_track",
                "track_id": int(track["track_id"]),
                "pair_index": int(
                    min(source.panel_index for source, _, _ in observations)
                ),
                "first_frame_id": int(observations[0][0].frame_id),
                "second_frame_id": int(observations[-1][0].frame_id),
                "selected_frame_id": int(owner_source.frame_id),
                "selected_panel_index": int(owner_source.panel_index),
                "selected_candidate_id": int(
                    owner_candidate.candidate_id
                ),
                "selected_source_coverage_ratio": float(coverage),
                "selected_clarity": float(clarity),
                "selected_panel_observation_count": len(observations),
                "baseline_owner_frame_ids": [
                    int(value) for value in baseline_owners
                ],
                "match": {
                    "dilated_world_voxel_overlap_ratio": float(
                        track["minimum_flow_mask_iou"]
                    ),
                    "minimum_flow_mask_iou": float(
                        track["minimum_flow_mask_iou"]
                    ),
                    "maximum_fb_p95_preview_pixels": float(
                        track["maximum_fb_p95_preview_pixels"]
                    ),
                    "maximum_area_ratio": float(
                        track["maximum_area_ratio"]
                    ),
                },
                "interval": interval,
                "owner_source": owner_source,
                "owner_natural": owner_natural,
                "lock_bbox_xywh": [
                    int(np.min(lock_x)),
                    int(np.min(lock_y)),
                    int(np.max(lock_x) - np.min(lock_x) + 1),
                    int(np.max(lock_y) - np.min(lock_y) + 1),
                ],
                "lock_pixel_count": int(
                    np.count_nonzero(interval.union_footprint)
                ),
                "interval_audit": interval.audit,
            }
        )
    return proposals, {
        "enabled": True,
        "audit_path": str(audit_path),
        "considered_track_count": considered,
        "proposal_count": len(proposals),
        "rejection_reason_counts": dict(rejected),
        "rejected_tracks": rejected_tracks,
        "minimum_flow_mask_iou": 0.75,
        "maximum_fb_p95_preview_pixels": 0.50,
        "maximum_area_ratio": 1.35,
        "minimum_natural_source_coverage_ratio": 0.90,
    }


def _mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    intersection = int(np.count_nonzero(first & second))
    union = int(np.count_nonzero(first | second))
    return float(intersection / union) if union else 0.0


def _masks_within_guard(
    first: np.ndarray,
    second: np.ndarray,
    guard_pixels: int,
) -> bool:
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (guard_pixels * 2 + 1, guard_pixels * 2 + 1),
    )
    dilated = cv2.dilate(first.astype(np.uint8), kernel) > 0
    return bool(np.any(dilated & second))


def _scene_world_compatible(
    proposals: list[dict[str, object]],
) -> bool:
    candidates = []
    for proposal in proposals:
        candidates.extend(
            (
                proposal["first_candidate"],
                proposal["second_candidate"],
            )
        )
    centroids = np.asarray(
        [item.world_centroid_mm for item in candidates], dtype=np.float64
    )
    spans = np.asarray(
        [item.world_spans_mm for item in candidates], dtype=np.float64
    )
    lower = centroids - 0.5 * spans
    upper = centroids + 0.5 * spans
    union_span = np.max(upper, axis=0) - np.min(lower, axis=0)
    centroid_span = np.ptp(centroids, axis=0)
    return bool(
        np.all(union_span <= np.asarray([500.0, 400.0, 350.0]))
        and float(np.linalg.norm(centroid_span)) <= 300.0
    )


def _build_scene_group_proposals(
    proposals: list[dict[str, object]],
    *,
    shape: tuple[int, int],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    ordered = sorted(
        proposals,
        key=lambda item: (
            item["selected_source_coverage_ratio"],
            item["match"]["dilated_world_voxel_overlap_ratio"],
            item["selected_clarity"],
            -item["lock_pixel_count"],
        ),
        reverse=True,
    )
    representatives: list[dict[str, object]] = []
    duplicate_count = 0
    for proposal in ordered:
        if any(
            int(existing["pair_index"]) == int(proposal["pair_index"])
            and _mask_iou(
                existing["interval"].union_footprint,
                proposal["interval"].union_footprint,
            )
            >= 0.65
            for existing in representatives
        ):
            duplicate_count += 1
            continue
        representatives.append(proposal)
    parent = list(range(len(representatives)))

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    adjacency_count = 0
    for first_index, first in enumerate(representatives):
        for second_index in range(first_index + 1, len(representatives)):
            second = representatives[second_index]
            if int(first["pair_index"]) != int(second["pair_index"]):
                continue
            if not _masks_within_guard(
                first["interval"].union_footprint,
                second["interval"].union_footprint,
                8,
            ):
                continue
            if not _scene_world_compatible([first, second]):
                continue
            first_root, second_root = find(first_index), find(second_index)
            if first_root != second_root:
                parent[max(first_root, second_root)] = min(
                    first_root, second_root
                )
            adjacency_count += 1
    components: dict[int, list[int]] = {}
    for index in range(len(representatives)):
        components.setdefault(find(index), []).append(index)
    compact_groups: list[list[int]] = []
    for component_indices in components.values():
        remaining = list(component_indices)
        while remaining:
            group = [remaining.pop(0)]
            changed = True
            while changed:
                changed = False
                for candidate_index in list(remaining):
                    candidate = representatives[candidate_index]
                    touches_group = any(
                        _masks_within_guard(
                            representatives[member][
                                "interval"
                            ].union_footprint,
                            candidate["interval"].union_footprint,
                            8,
                        )
                        for member in group
                    )
                    if not touches_group:
                        continue
                    proposed = [
                        representatives[member] for member in group
                    ] + [candidate]
                    if not _scene_world_compatible(proposed):
                        continue
                    group.append(candidate_index)
                    remaining.remove(candidate_index)
                    changed = True
            if len(group) >= 2:
                compact_groups.append(group)

    consumed: set[int] = set()
    group_proposals: list[dict[str, object]] = []
    rejected_groups: list[dict[str, object]] = []
    for indices in compact_groups:
        members = [representatives[index] for index in indices]
        pair_index = int(members[0]["pair_index"])
        options = []
        for side in ("first", "second"):
            source = members[0][f"{side}_source"]
            naturals = [member[f"{side}_natural"] for member in members]
            if any(
                natural.source_coverage_ratio < 0.90
                for natural in naturals
            ):
                continue
            footprints = tuple(
                footprint
                for member in members
                for footprint in (
                    member["first_footprint"],
                    member["second_footprint"],
                )
            )
            try:
                interval = build_object_owner_interval(
                    panel_index=source.panel_index,
                    view_dependent_footprints=footprints,
                    selected_panel_valid_mask=_full_valid(
                        naturals[0], shape
                    ),
                )
            except (RuntimeError, ValueError):
                continue
            options.append(
                (
                    min(
                        natural.source_coverage_ratio
                        for natural in naturals
                    ),
                    sum(natural.clarity for natural in naturals),
                    -source.frame_id,
                    side,
                    source,
                    naturals[0],
                    interval,
                )
            )
        if not options:
            rejected_groups.append(
                {
                    "member_count": len(members),
                    "pair_index": pair_index,
                    "reason": (
                        "no_common_panel_completely_covers_all_scene_masks"
                    ),
                }
            )
            continue
        (
            minimum_coverage,
            clarity_sum,
            _,
            selected_side,
            owner_source,
            owner_natural,
            interval,
        ) = max(options, key=lambda item: item[:3])
        lock_y, lock_x = np.nonzero(interval.lock_mask)
        group_proposals.append(
            {
                "proposal_kind": "scene_group",
                "pair_index": pair_index,
                "first_frame_id": int(members[0]["first_frame_id"]),
                "second_frame_id": int(members[0]["second_frame_id"]),
                "selected_frame_id": int(owner_source.frame_id),
                "selected_panel_index": int(owner_source.panel_index),
                "selected_scene_side": selected_side,
                "selected_source_coverage_ratio": float(minimum_coverage),
                "selected_clarity": float(clarity_sum),
                "baseline_owner_frame_ids": sorted(
                    {
                        int(value)
                        for member in members
                        for value in member["baseline_owner_frame_ids"]
                    }
                ),
                "match": {
                    "dilated_world_voxel_overlap_ratio": min(
                        float(
                            member["match"][
                                "dilated_world_voxel_overlap_ratio"
                            ]
                        )
                        for member in members
                    )
                },
                "interval": interval,
                "owner_source": owner_source,
                "owner_natural": owner_natural,
                "lock_bbox_xywh": [
                    int(np.min(lock_x)),
                    int(np.min(lock_y)),
                    int(np.max(lock_x) - np.min(lock_x) + 1),
                    int(np.max(lock_y) - np.min(lock_y) + 1),
                ],
                "lock_pixel_count": int(
                    np.count_nonzero(interval.lock_mask)
                ),
                "interval_audit": interval.audit,
                "scene_group_member_count": len(members),
                "scene_group_member_lock_bboxes_xywh": [
                    list(member["lock_bbox_xywh"]) for member in members
                ],
                "scene_group_member_candidate_ids": [
                    [
                        int(member["first_candidate_id"]),
                        int(member["second_candidate_id"]),
                    ]
                    for member in members
                ],
            }
        )
        consumed.update(indices)
    remaining = [
        proposal
        for index, proposal in enumerate(representatives)
        if index not in consumed
    ]
    return group_proposals + remaining, {
        "input_proposal_count": len(proposals),
        "duplicate_proposal_count": duplicate_count,
        "representative_proposal_count": len(representatives),
        "adjacency_edge_count": adjacency_count,
        "candidate_scene_group_count": len(compact_groups),
        "accepted_scene_group_proposal_count": len(group_proposals),
        "rejected_scene_groups": rejected_groups,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("session")
    parser.add_argument("formal_output")
    parser.add_argument("labels")
    parser.add_argument("--scene-groups", action="store_true")
    parser.add_argument("--dis-track-audit", type=Path)
    arguments = parser.parse_args()
    if arguments.scene_groups and arguments.dis_track_audit is not None:
        parser.error("--scene-groups and --dis-track-audit are exclusive")
    started = time.perf_counter()
    session_path = Path(arguments.session).expanduser().resolve()
    output = Path(arguments.formal_output).expanduser().resolve()
    labels_path = Path(arguments.labels).expanduser().resolve()
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    transforms = json.loads(
        (output / "transforms.json").read_text(encoding="utf-8")
    )
    render = report["render"]
    layout = _layout_from_report(render["layout"])
    config = InspectionMultiviewConfig.from_mapping(render["config"])
    session = load_rgbd_session(session_path)
    frame_by_id = {int(frame.frame_id): frame for frame in session.frames}
    pose_by_id = {
        int(item["node_id"]): np.asarray(
            item["camera_to_world"], dtype=np.float64
        )
        for item in transforms["nodes"]
    }
    full_image, full_owner, baseline_valid = _full_baseline(
        output, layout, render["crop"]
    )
    selected_rows = render["selected_panel_sources"]
    source_position_by_frame_id = {
        int(frame_id): index
        for index, frame_id in enumerate(render["frame_ids"])
    }
    maps = _undistortion_maps(session.calibration)
    sources: list[_PanelSource] = []
    raw_polygon_count = 0
    accepted_polygon_count = 0
    next_candidate_id = 0
    for row in selected_rows:
        panel_index = int(row["panel_index"])
        frame_id = int(row["frame_id"])
        source_index = source_position_by_frame_id[frame_id]
        image, depth, geometric_valid = _read_rgbd(
            frame_by_id[frame_id], session.calibration, maps
        )
        reliable = (
            geometric_valid
            & np.isfinite(depth)
            & (depth >= config.minimum_depth_mm)
            & (depth <= config.maximum_depth_mm)
        )
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        polygons = parse_fastsam_polygons(
            labels_path / f"{frame_id:08d}.txt",
            width=session.calibration.width,
            height=session.calibration.height,
        )
        raw_polygon_count += len(polygons)
        candidates: list[FastSAMRGBDCandidate] = []
        for polygon in polygons:
            candidate = build_fastsam_rgbd_candidate(
                candidate_id=next_candidate_id,
                source_index=source_index,
                frame_id=frame_id,
                polygon_xy=polygon,
                image_bgr=image,
                lab_image=lab,
                depth_mm=depth,
                reliable_depth=reliable,
                camera_to_world=pose_by_id[frame_id],
                intrinsics=session.calibration,
                reference_depth_mm=layout.reference_depth_mm,
            )
            next_candidate_id += 1
            if candidate is not None:
                candidates.append(candidate)
                accepted_polygon_count += 1
        sources.append(
            _PanelSource(
                panel_index=panel_index,
                source_index=source_index,
                frame_id=frame_id,
                image=np.ascontiguousarray(image),
                depth=np.ascontiguousarray(depth),
                reliable=np.ascontiguousarray(reliable),
                pose=np.ascontiguousarray(pose_by_id[frame_id]),
                candidates=tuple(candidates),
            )
        )

    natural_cache: dict[int, _NaturalMask | None] = {}

    def mapped(
        source: _PanelSource,
        candidate: FastSAMRGBDCandidate,
    ) -> _NaturalMask | None:
        if candidate.candidate_id not in natural_cache:
            natural_cache[candidate.candidate_id] = _natural_mask(
                source, candidate, layout, session.calibration
            )
        return natural_cache[candidate.candidate_id]

    pair_audits: list[dict[str, object]] = []
    proposals: list[dict[str, object]] = []
    for first, second in zip(sources[:-1], sources[1:], strict=True):
        match_count = 0
        pair_proposals = 0
        rejected_counts: Counter[str] = Counter()
        for first_candidate in first.candidates:
            for second_candidate in second.candidates:
                match, match_audit = _pair_match(
                    first_candidate, second_candidate
                )
                if not match:
                    continue
                match_count += 1
                first_natural = mapped(first, first_candidate)
                second_natural = mapped(second, second_candidate)
                if first_natural is None or second_natural is None:
                    rejected_counts["natural_inverse_map_incomplete"] += 1
                    continue
                if (
                    first_natural.source_coverage_ratio < 0.90
                    or second_natural.source_coverage_ratio < 0.90
                ):
                    rejected_counts[
                        "natural_inverse_map_source_coverage_below_0_90"
                    ] += 1
                    continue
                first_footprint = _full_mask(
                    first_natural, baseline_valid.shape
                )
                second_footprint = _full_mask(
                    second_natural, baseline_valid.shape
                )
                options = sorted(
                    (
                        (
                            first_natural.source_coverage_ratio,
                            first_natural.clarity,
                            -first.frame_id,
                            first,
                            first_candidate,
                            first_natural,
                        ),
                        (
                            second_natural.source_coverage_ratio,
                            second_natural.clarity,
                            -second.frame_id,
                            second,
                            second_candidate,
                            second_natural,
                        ),
                    ),
                    reverse=True,
                    key=lambda item: item[:3],
                )
                selected = None
                failure_reasons = []
                for (
                    _,
                    _,
                    _,
                    owner_source,
                    owner_candidate,
                    owner_natural,
                ) in options:
                    try:
                        interval = build_object_owner_interval(
                            panel_index=owner_source.panel_index,
                            view_dependent_footprints=(
                                first_footprint,
                                second_footprint,
                            ),
                            selected_panel_valid_mask=_full_valid(
                                owner_natural, baseline_valid.shape
                            ),
                        )
                    except (RuntimeError, ValueError) as exc:
                        failure_reasons.append(str(exc))
                        continue
                    selected = (
                        owner_source,
                        owner_candidate,
                        owner_natural,
                        interval,
                    )
                    break
                if selected is None:
                    rejected_counts[
                        "no_panel_completely_covers_monotone_union_interval"
                    ] += 1
                    continue
                (
                    owner_source,
                    owner_candidate,
                    owner_natural,
                    interval,
                ) = selected
                baseline_owners = np.unique(
                    full_owner[interval.union_footprint & baseline_valid]
                )
                baseline_owners = baseline_owners[baseline_owners >= 0]
                if baseline_owners.size < 2:
                    rejected_counts[
                        "footprint_union_not_crossed_by_baseline_handoff"
                    ] += 1
                    continue
                lock_y, lock_x = np.nonzero(interval.lock_mask)
                proposals.append(
                    {
                        "proposal_kind": "individual",
                        "pair_index": int(first.panel_index),
                        "first_frame_id": first.frame_id,
                        "second_frame_id": second.frame_id,
                        "first_candidate_id": first_candidate.candidate_id,
                        "second_candidate_id": second_candidate.candidate_id,
                        "first_candidate": first_candidate,
                        "second_candidate": second_candidate,
                        "first_source": first,
                        "second_source": second,
                        "first_natural": first_natural,
                        "second_natural": second_natural,
                        "first_footprint": first_footprint,
                        "second_footprint": second_footprint,
                        "selected_frame_id": owner_source.frame_id,
                        "selected_panel_index": owner_source.panel_index,
                        "selected_candidate_id": (
                            owner_candidate.candidate_id
                        ),
                        "selected_source_coverage_ratio": (
                            owner_natural.source_coverage_ratio
                        ),
                        "selected_clarity": owner_natural.clarity,
                        "baseline_owner_frame_ids": [
                            int(value) for value in baseline_owners
                        ],
                        "match": match_audit,
                        "interval": interval,
                        "owner_source": owner_source,
                        "owner_natural": owner_natural,
                        "lock_bbox_xywh": [
                            int(np.min(lock_x)),
                            int(np.min(lock_y)),
                            int(np.max(lock_x) - np.min(lock_x) + 1),
                            int(np.max(lock_y) - np.min(lock_y) + 1),
                        ],
                        "lock_pixel_count": int(
                            np.count_nonzero(interval.lock_mask)
                        ),
                        "interval_audit": interval.audit,
                    }
                )
                pair_proposals += 1
        pair_audits.append(
            {
                "pair_index": int(first.panel_index),
                "first_frame_id": first.frame_id,
                "second_frame_id": second.frame_id,
                "first_candidate_count": len(first.candidates),
                "second_candidate_count": len(second.candidates),
                "rgbd_world_match_count": match_count,
                "panel_lock_proposal_count": pair_proposals,
                "rejection_reason_counts": dict(rejected_counts),
            }
        )

    dis_track_audit: dict[str, object] = {"enabled": False}
    if arguments.dis_track_audit is not None:
        proposals, dis_track_audit = _build_dis_track_proposals(
            audit_path=arguments.dis_track_audit.expanduser().resolve(),
            sources=sources,
            mapped=mapped,
            baseline_valid=baseline_valid,
            full_owner=full_owner,
        )

    frame_to_panel = {
        int(source.frame_id): int(source.panel_index) for source in sources
    }
    owner_panel = np.full(full_owner.shape, -1, dtype=np.int16)
    for frame_id, panel_index in frame_to_panel.items():
        owner_panel[full_owner == frame_id] = np.int16(panel_index)
    diagnostic = full_image.copy()
    accepted_lock = np.zeros(baseline_valid.shape, dtype=bool)
    accepted: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []
    scene_group_audit: dict[str, object] = {
        "enabled": False,
        "input_proposal_count": len(proposals),
    }
    proposals_to_apply = proposals
    if arguments.scene_groups:
        proposals_to_apply, scene_group_payload = (
            _build_scene_group_proposals(
                proposals, shape=baseline_valid.shape
            )
        )
        scene_group_audit = {
            "enabled": True,
            **scene_group_payload,
        }
    mapped_panel_cache: dict[int, np.ndarray] = {}

    def mapped_panel_image(source: _PanelSource) -> np.ndarray:
        if source.panel_index not in mapped_panel_cache:
            _, map_x, map_y, _, _ = _reference_panel_inverse_maps(
                source_pose=source.pose,
                panel_index=source.panel_index,
                layout=layout,
                intrinsics=session.calibration,
            )
            mapped_panel_cache[source.panel_index] = cv2.remap(
                source.image,
                map_x,
                map_y,
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
        return mapped_panel_cache[source.panel_index]

    post_chain_overlay_count = 0
    for proposal in sorted(
        proposals_to_apply,
        key=lambda item: (
            item["selected_source_coverage_ratio"],
            item["match"]["dilated_world_voxel_overlap_ratio"],
            item["selected_clarity"],
            -item["lock_pixel_count"],
        ),
        reverse=True,
    ):
        interval = proposal["interval"]
        owner_source = proposal["owner_source"]
        owner_natural = proposal["owner_natural"]
        runtime_keys = {
            "interval",
            "owner_source",
            "owner_natural",
            "first_candidate",
            "second_candidate",
            "first_source",
            "second_source",
            "first_natural",
            "second_natural",
            "first_footprint",
            "second_footprint",
        }
        public_proposal = {
            key: value
            for key, value in proposal.items()
            if key not in runtime_keys
        }
        overlap = int(
            np.count_nonzero(interval.lock_mask & accepted_lock)
        )
        if overlap:
            rejected.append(
                {
                    **public_proposal,
                    "reason": "overlaps_another_accepted_panel_lock",
                    "accepted_lock_overlap_pixel_count": overlap,
                }
            )
            continue
        if proposal.get("proposal_kind") == "stable_dis_track":
            overlay_region = interval.union_footprint
            selected_valid = _full_valid(
                owner_natural, baseline_valid.shape
            )
            if not np.all(selected_valid[overlay_region]):
                rejected.append(
                    {
                        **public_proposal,
                        "reason": (
                            "selected_panel_does_not_cover_track_union"
                        ),
                    }
                )
                continue
            x0 = owner_natural.corner_x
            x1 = x0 + owner_natural.local_valid.shape[1]
            local_overlay = overlay_region[:, x0:x1]
            panel_image = mapped_panel_image(owner_source)
            diagnostic[:, x0:x1][local_overlay] = (
                panel_image[local_overlay]
            )
            accepted_lock |= overlay_region
            post_chain_overlay_count += 1
            accepted.append(
                {
                    **public_proposal,
                    "reason": (
                        "accepted_post_chain_stable_dis_track_overlay"
                    ),
                    "post_chain_overlay": True,
                    "background_owner_chain_modified": False,
                    "foreground_owner_island_excluded_from_"
                    "background_topology_audit": True,
                    "overlay_owner_frame_id": int(
                        owner_source.frame_id
                    ),
                    "overlay_pixel_count": int(
                        np.count_nonzero(overlay_region)
                    ),
                    "overlay_exactly_one_frame_owner": True,
                    "all_observation_footprints_replaced": True,
                    "protected_blend_intersection_pixel_count": 0,
                    "natural_position_only": True,
                    "translation_used": False,
                    "affine_used": False,
                    "warp_used": False,
                    "hole_fill_used": False,
                    "generated_rgb_used": False,
                    "single_real_rgb_owner": True,
                }
            )
            continue
        trial = owner_panel.copy()
        trial[interval.lock_mask] = np.int16(
            owner_source.panel_index
        )
        if not _owner_rows_monotonic(trial):
            if proposal.get("proposal_kind") == "scene_group":
                overlay_region = interval.union_footprint
                overlay_overlap = int(
                    np.count_nonzero(overlay_region & accepted_lock)
                )
                selected_valid = _full_valid(
                    owner_natural, baseline_valid.shape
                )
                if (
                    overlay_overlap == 0
                    and np.all(selected_valid[overlay_region])
                ):
                    x0 = owner_natural.corner_x
                    x1 = x0 + owner_natural.local_valid.shape[1]
                    local_overlay = overlay_region[:, x0:x1]
                    panel_image = mapped_panel_image(owner_source)
                    diagnostic[:, x0:x1][local_overlay] = (
                        panel_image[local_overlay]
                    )
                    accepted_lock |= overlay_region
                    post_chain_overlay_count += 1
                    accepted.append(
                        {
                            **public_proposal,
                            "reason": (
                                "accepted_post_chain_foreground_scene_"
                                "group_overlay"
                            ),
                            "post_chain_overlay": True,
                            "background_owner_chain_modified": False,
                            "foreground_owner_island_excluded_from_"
                            "background_topology_audit": True,
                            "overlay_owner_frame_id": int(
                                owner_source.frame_id
                            ),
                            "overlay_pixel_count": int(
                                np.count_nonzero(overlay_region)
                            ),
                            "overlay_exactly_one_frame_owner": True,
                            "all_observation_footprints_replaced": True,
                            "protected_blend_intersection_pixel_count": 0,
                            "natural_position_only": True,
                            "translation_used": False,
                            "affine_used": False,
                            "warp_used": False,
                            "hole_fill_used": False,
                            "generated_rgb_used": False,
                            "single_real_rgb_owner": True,
                        }
                    )
                    continue
            rejected.append(
                {
                    **public_proposal,
                    "reason": "panel_lock_breaks_closed_monotone_owner_chain",
                }
            )
            continue
        x0 = owner_natural.corner_x
        x1 = x0 + owner_natural.local_valid.shape[1]
        mapped_image = mapped_panel_image(owner_source)
        local_lock = interval.lock_mask[:, x0:x1]
        diagnostic[:, x0:x1][local_lock] = mapped_image[local_lock]
        owner_panel = trial
        accepted_lock |= interval.lock_mask
        accepted.append(
            {
                **public_proposal,
                "reason": "accepted_panel_native_single_owner_lock",
                "natural_position_only": True,
                "translation_used": False,
                "affine_used": False,
                "warp_used": False,
                "hole_fill_used": False,
                "generated_rgb_used": False,
                "single_real_rgb_owner": True,
            }
        )

    crop = render["crop"]
    crop_x, crop_y, crop_width, crop_height = (
        int(crop[name]) for name in ("x", "y", "width", "height")
    )
    baseline_crop = full_image[
        crop_y : crop_y + crop_height,
        crop_x : crop_x + crop_width,
    ]
    diagnostic_crop = diagnostic[
        crop_y : crop_y + crop_height,
        crop_x : crop_x + crop_width,
    ]
    comparison = np.hstack((baseline_crop, diagnostic_crop))
    for item in accepted:
        x, y, width, height = item["lock_bbox_xywh"]
        cv2.rectangle(
            comparison,
            (x - crop_x + crop_width, y - crop_y),
            (
                x + width - 1 - crop_x + crop_width,
                y + height - 1 - crop_y,
            ),
            (0, 255, 0),
            2,
        )
    name = (
        "diagnostic_fastsam_dis_track_lock"
        if arguments.dis_track_audit is not None
        else (
            "diagnostic_fastsam_scene_group"
            if arguments.scene_groups
            else "diagnostic_fastsam_panel_lock"
        )
    )
    diagnostic_path = output / f"{name}.png"
    comparison_path = output / f"{name}_before_after.jpg"
    audit_path = output / f"{name}_audit.json"
    if not cv2.imwrite(str(diagnostic_path), diagnostic_crop):
        raise RuntimeError("Could not write FastSAM panel-lock diagnostic")
    if not cv2.imwrite(str(comparison_path), comparison):
        raise RuntimeError("Could not write FastSAM panel-lock comparison")
    audit = {
        "schema": "inspection-fastsam-panel-lock-diagnostic/v1",
        "formal_output_modified": False,
        "formal_acceptance": False,
        "formal_acceptance_reason": (
            "isolated_adjacent_panel_diagnostic_not_connected_to_formal_chain"
        ),
        "model_role": "polygon_contour_proposals_only",
        "model_weight_sha256": (
            "c9f78716a81c7aff0d608ccc73e1b82a"
            "b3aaad86005049f6a92106a0be6d0844"
        ),
        "selected_panel_source_count": len(sources),
        "adjacent_pair_count": len(sources) - 1,
        "raw_polygon_count": raw_polygon_count,
        "rgbd_candidate_count": accepted_polygon_count,
        "panel_lock_proposal_count": len(proposals),
        "applied_proposal_count": len(proposals_to_apply),
        "scene_grouping": scene_group_audit,
        "dis_track_identity": dis_track_audit,
        "accepted_scene_group_count": sum(
            item.get("proposal_kind") == "scene_group"
            for item in accepted
        ),
        "accepted_post_chain_overlay_count": post_chain_overlay_count,
        "foreground_owner_islands_allowed": bool(
            post_chain_overlay_count
        ),
        "background_owner_topology_audit_excludes_foreground_overlays": bool(
            post_chain_overlay_count
        ),
        "accepted_panel_lock_count": len(accepted),
        "rejected_panel_lock_count": len(rejected),
        "accepted_lock_pixel_count": int(np.count_nonzero(accepted_lock)),
        "final_lock_rows_monotonic": _owner_rows_monotonic(owner_panel),
        "pair_audits": pair_audits,
        "accepted_panel_locks": accepted,
        "rejected_panel_locks": rejected,
        "manual_roi_used": False,
        "manual_frame_id_used": False,
        "model_position_used": False,
        "model_rgb_used": False,
        "translation_used": False,
        "affine_used": False,
        "warp_used": False,
        "hole_fill_used": False,
        "generated_rgb_used": False,
        "silent_fallback_allowed": False,
        "thresholds": {
            "minimum_dilated_world_voxel_overlap_ratio": 0.25,
            "maximum_lab_delta": 30.0,
            "maximum_world_centroid_delta_mm": 120.0,
            "maximum_contour_match_i1": 0.35,
            "minimum_natural_source_coverage_ratio": 0.90,
            "dis_track_minimum_flow_mask_iou": 0.75,
            "dis_track_maximum_fb_p95_preview_pixels": 0.50,
            "dis_track_maximum_area_ratio": 1.35,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "files": {
            "diagnostic": diagnostic_path.name,
            "before_after": comparison_path.name,
        },
    }
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(audit_path)
    print(diagnostic_path)
    print(comparison_path)
    print(
        json.dumps(
            {
                "raw_polygon_count": raw_polygon_count,
                "rgbd_candidate_count": accepted_polygon_count,
                "panel_lock_proposal_count": len(proposals),
                "accepted_panel_lock_count": len(accepted),
                "rejected_panel_lock_count": len(rejected),
                "elapsed_seconds": audit["elapsed_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
