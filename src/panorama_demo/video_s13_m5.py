"""M5 geometry/seam transactions and owner-only P2 rendering for S1.3."""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import cv2
import numpy as np

from .calibrated_remap import accelerated_remap, undistortion_maps
from .session import CameraIntrinsics
from .video_s12_schedule import S012Schedule, validate_s012_schedule
from .video_s13_alignment import (
    S13AlignmentCandidate,
    S13ApplicationBand,
    S13PairAlignment,
    estimate_s13_pair_alignment,
    reestimate_s13_final_corridor_alignment,
)
from .video_s13_quality import (
    long_horizontal_structure_metrics,
    long_horizontal_structure_nondegrading,
    seam_structure_metrics,
    sequence_structure_decision,
    structurally_non_degrading,
    symmetric_seam_structure_metrics,
)
from .video_s13_seam import S13SeamResult, select_s13_seam
from .video_s13_vertical import S13VerticalSolution


@dataclass(frozen=True)
class S13M5Pair:
    transaction: Mapping[str, object]
    seam_x_by_row: np.ndarray
    alignment: S13PairAlignment | None


@dataclass(frozen=True)
class S13P2Result:
    image: np.ndarray
    valid_mask: np.ndarray
    pixel_provenance: dict[str, np.ndarray]
    remap_invocations: int
    decoded_frame_ids: tuple[int, ...]


@dataclass(frozen=True)
class S13M5Result:
    pairs: tuple[S13M5Pair, ...]
    geometry_result: S13P2Result
    final_result: S13P2Result
    seam_overlay: np.ndarray
    selected_as_best: bool
    before_mean_score: float | None
    after_mean_score: float | None
    selection_audit: Mapping[str, object]
    performance: Mapping[str, float]


def _sha_array(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(value.shape).encode("ascii"))
        digest.update(value.tobytes())
    return digest.hexdigest()


def _sha_json(value: Mapping[str, object]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def _base_calibrated_map(
    schedule: S012Schedule,
    calibration: CameraIntrinsics,
    source_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    assignment = schedule.assignments[source_index]
    height, width = schedule.canvas_height, schedule.canvas_width
    canvas_x = np.broadcast_to(np.arange(width, dtype=np.float32)[None, :], (height, width))
    canvas_y = np.broadcast_to(np.arange(height, dtype=np.float32)[:, None], (height, width))
    calibrated_x = float(calibration.cx) + canvas_x - float(assignment.center_x)
    calibrated_y = canvas_y
    inverse = undistortion_maps(calibration)
    if inverse is None:
        source_u, source_v = calibrated_x, calibrated_y
    else:
        source_u = cv2.remap(inverse[0], calibrated_x, calibrated_y, cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=-1)
        source_v = cv2.remap(inverse[1], calibrated_x, calibrated_y, cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=-1)
    valid = (
        np.isfinite(source_u) & np.isfinite(source_v)
        & (source_u >= 0.0) & (source_u <= int(calibration.width) - 1)
        & (source_v >= 0.0) & (source_v <= int(calibration.height) - 1)
    )
    return source_u.astype(np.float32), source_v.astype(np.float32), valid


def _map_crop(
    schedule: S012Schedule,
    calibration: CameraIntrinsics,
    source_index: int,
    x0: int,
    x1: int,
    global_vertical_offset: float,
    candidate: S13AlignmentCandidate | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    assignment = schedule.assignments[source_index]
    height = schedule.canvas_height
    canvas_x = np.broadcast_to(np.arange(x0, x1, dtype=np.float32)[None, :], (height, x1 - x0))
    canvas_y = np.broadcast_to(np.arange(height, dtype=np.float32)[:, None], canvas_x.shape)
    delta_u = np.zeros(canvas_x.shape, dtype=np.float32)
    delta_v = np.zeros(canvas_x.shape, dtype=np.float32)
    if candidate is not None:
        overlap0, overlap1 = max(x0, candidate.x0), min(x1, candidate.x1)
        if overlap1 > overlap0:
            target = np.s_[:, overlap0 - x0:overlap1 - x0]
            source = np.s_[:, overlap0 - candidate.x0:overlap1 - candidate.x0]
            delta_u[target] = candidate.target_delta_u[source]
            delta_v[target] = candidate.target_delta_v[source]
    calibrated_x = float(calibration.cx) + canvas_x - float(assignment.center_x) + delta_u
    calibrated_y = canvas_y - float(global_vertical_offset) + delta_v
    inverse = undistortion_maps(calibration)
    if inverse is None:
        source_u, source_v = calibrated_x, calibrated_y
    else:
        source_u = cv2.remap(inverse[0], calibrated_x, calibrated_y, cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=-1)
        source_v = cv2.remap(inverse[1], calibrated_x, calibrated_y, cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=-1)
    valid = (
        np.isfinite(source_u) & np.isfinite(source_v)
        & (source_u >= 0.0) & (source_u <= int(calibration.width) - 1)
        & (source_v >= 0.0) & (source_v <= int(calibration.height) - 1)
    )
    return source_u.astype(np.float32), source_v.astype(np.float32), valid


def _sample_crop(
    raw: np.ndarray,
    maps: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    u, v, valid = maps
    sampled = accelerated_remap(raw, u, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return sampled, valid


def _pair_correspondences(
    left: np.ndarray,
    right: np.ndarray,
    left_valid: np.ndarray,
    right_valid: np.ndarray,
    *,
    x_offset: int,
) -> tuple[np.ndarray, np.ndarray]:
    left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
    mask = (left_valid & right_valid).astype(np.uint8) * 255
    points = cv2.goodFeaturesToTrack(left_gray, 400, 0.01, 4.0, mask=mask, blockSize=5)
    if points is None:
        return np.empty((0, 2), np.float64), np.empty((0, 2), np.float64)
    moved, status, _error = cv2.calcOpticalFlowPyrLK(
        left_gray, right_gray, points, None, winSize=(21, 21), maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    if moved is None or status is None:
        return np.empty((0, 2), np.float64), np.empty((0, 2), np.float64)
    source = points.reshape(-1, 2)
    target = moved.reshape(-1, 2)
    finite = (status.reshape(-1) > 0) & np.isfinite(source).all(axis=1) & np.isfinite(target).all(axis=1)
    source, target = source[finite].astype(np.float64), target[finite].astype(np.float64)
    source[:, 0] += x_offset
    target[:, 0] += x_offset
    return source, target


def _pair_domain(schedule: S012Schedule, pair_index: int, half_width: int = 48) -> tuple[int, int]:
    boundary = schedule.boundaries[pair_index + 1]
    return max(0, boundary - half_width), min(schedule.canvas_width, boundary + half_width)


def _allowed_application_bounds(schedule: S012Schedule, pair_index: int) -> tuple[int, int]:
    boundary = schedule.boundaries[pair_index + 1]
    previous = schedule.boundaries[pair_index] if pair_index > 0 else 0
    following = schedule.boundaries[pair_index + 2] if pair_index + 2 < len(schedule.boundaries) else schedule.canvas_width
    domain_left = int(math.ceil(0.5 * (previous + boundary)))
    domain_right = int(math.floor(0.5 * (boundary + following))) + 1
    left = max(domain_left, boundary - 8)
    right = min(domain_right, boundary + 9)
    if right - left < 3:
        left, right = max(0, boundary - 1), min(schedule.canvas_width, boundary + 2)
    return left, right


def _fallback_pair(
    schedule: S012Schedule,
    pair_index: int,
    parent_sha: str,
    frame_ids: tuple[int, int],
    reason: str,
    before: Mapping[str, object] | None = None,
    after: Mapping[str, object] | None = None,
) -> S13M5Pair:
    seam = np.full(schedule.canvas_height, schedule.boundaries[pair_index + 1], dtype=np.int32)
    core: dict[str, object] = {
        "transaction_id": f"m5-pair-{pair_index:04d}",
        "parent_stage_sha256": parent_sha,
        "pair_frame_ids": list(frame_ids),
        "motion_hypothesis_id": -1,
        "support_mask_sha256": None,
        "geometry_model": "C1_accepted_vertical_parent",
        "seam_model": "S0_midpoint_straight",
        "map_delta_sha256": None,
        "decision": "rolled_back",
        "before_metrics": dict(before) if before is not None else {"evaluable": False, "reason": reason, "score": None},
        "candidate_after_metrics": (
            dict(after) if after is not None else {"evaluable": False, "reason": reason, "score": None}
        ),
        "after_metrics": (
            dict(before) if before is not None else {"evaluable": False, "reason": reason, "score": None}
        ),
        "rollback_reason": reason,
        "final_corridor_geometry_reestimated_from_immutable_p0": False,
        "comparison_coordinate_policy": (
            "same_owner_geometry_plus_symmetric_base_candidate_seam"
        ),
    }
    core["result_stage_sha256"] = _sha_json(core)
    return S13M5Pair(core, seam, None)


def _compose_pair_preview(left: np.ndarray, right: np.ndarray, seam_local: np.ndarray) -> np.ndarray:
    height, width = left.shape[:2]
    x = np.arange(width, dtype=np.int32)[None, :]
    owner_right = x >= seam_local[:, None]
    return np.where(owner_right[..., None], right, left).astype(np.uint8)


def _audit_rendered_seam_change(
    before_image: np.ndarray,
    after_image: np.ndarray,
    before_seam_x_by_row: np.ndarray,
    after_seam_x_by_row: np.ndarray,
) -> dict[str, object]:
    """Audit two rendered owner topologies on both paths and held-out shoulders."""

    before_seam = np.asarray(before_seam_x_by_row, dtype=np.int32)
    after_seam = np.asarray(after_seam_x_by_row, dtype=np.int32)
    symmetric_before, symmetric_after = symmetric_seam_structure_metrics(
        before_image,
        after_image,
        before_seam,
        after_seam,
    )
    symmetric_ok, symmetric_reason = structurally_non_degrading(
        symmetric_before, symmetric_after
    )
    path_audits: dict[str, object] = {}
    paths_ok = True
    first_reason: str | None = None
    for name, path in (("before_owner", before_seam), ("after_owner", after_seam)):
        before_metrics = seam_structure_metrics(before_image, path)
        after_metrics = seam_structure_metrics(after_image, path)
        structure_ok, structure_reason = structurally_non_degrading(
            before_metrics, after_metrics
        )
        before_horizontal = long_horizontal_structure_metrics(before_image, path)
        after_horizontal = long_horizontal_structure_metrics(after_image, path)
        horizontal_ok, horizontal_reason = long_horizontal_structure_nondegrading(
            before_horizontal, after_horizontal
        )
        path_ok = bool(structure_ok and horizontal_ok)
        if not path_ok and first_reason is None:
            first_reason = structure_reason if not structure_ok else horizontal_reason
        paths_ok = paths_ok and path_ok
        path_audits[name] = {
            "eligible": path_ok,
            "reason": None if path_ok else (
                structure_reason if not structure_ok else horizontal_reason
            ),
            "before_metrics": before_metrics,
            "after_metrics": after_metrics,
            "before_horizontal_continuity": before_horizontal,
            "after_horizontal_continuity": after_horizontal,
        }
    eligible = bool(symmetric_ok and paths_ok)
    reason = None if eligible else (symmetric_reason if not symmetric_ok else first_reason)
    return {
        "eligible": eligible,
        "reason": reason,
        "comparison_policy": (
            "symmetric_before_and_after_owner_paths_with_held_out_shoulders"
        ),
        "before_seam_sha256": _sha_array(before_seam),
        "after_seam_sha256": _sha_array(after_seam),
        "before_image_sha256": _sha_array(before_image),
        "after_image_sha256": _sha_array(after_image),
        "symmetric_before_metrics": symmetric_before,
        "symmetric_after_metrics": symmetric_after,
        "symmetric_structure_eligible": bool(symmetric_ok),
        "symmetric_structure_reason": symmetric_reason,
        "path_audits": path_audits,
    }


def _audit_owner_seam_change(
    left_image: np.ndarray,
    right_image: np.ndarray,
    before_seam_x_by_row: np.ndarray,
    after_seam_x_by_row: np.ndarray,
) -> dict[str, object]:
    """Compose both owner choices from identical pixels before comparing them."""

    before_seam = np.asarray(before_seam_x_by_row, dtype=np.int32)
    after_seam = np.asarray(after_seam_x_by_row, dtype=np.int32)
    before_preview = _compose_pair_preview(left_image, right_image, before_seam)
    after_preview = _compose_pair_preview(left_image, right_image, after_seam)
    audit = _audit_rendered_seam_change(
        before_preview,
        after_preview,
        before_seam,
        after_seam,
    )
    return {
        **audit,
        "before_owner_seam_sha256": _sha_array(before_seam),
        "after_owner_seam_sha256": _sha_array(after_seam),
    }


def _select_held_out_safe_seam(
    left_image: np.ndarray,
    right_image: np.ndarray,
    seam_result: S13SeamResult,
    maximum_model: str,
) -> tuple[str, np.ndarray, dict[str, object]]:
    """Only retain a more complex seam after independent structure evidence."""

    candidates = seam_result.candidate_seams
    base = np.asarray(candidates["midpoint_straight"], dtype=np.int32)
    straight_value = candidates.get("shifted_straight")
    dp_value = candidates.get("monotone_dp")
    straight = None if straight_value is None else np.asarray(straight_value, dtype=np.int32)
    dp = None if dp_value is None else np.asarray(dp_value, dtype=np.int32)
    allowed_rank = {
        "midpoint_straight": 0,
        "shifted_straight": 1,
        "monotone_dp": 2,
    }[maximum_model]

    straight_audit: dict[str, object] | None = None
    straight_safe = False
    if straight is not None and allowed_rank >= 1:
        straight_audit = _audit_owner_seam_change(
            left_image, right_image, base, straight
        )
        straight_safe = bool(straight_audit["eligible"])

    dp_audit: dict[str, object] | None = None
    dp_safe = False
    if dp is not None and straight is not None and allowed_rank >= 2 and straight_safe:
        dp_audit = _audit_owner_seam_change(
            left_image, right_image, straight, dp
        )
        dp_safe = bool(dp_audit["eligible"])

    if dp_safe:
        selected_model, selected_seam = "monotone_dp", dp
        reason = "dp_cost_margin_and_held_out_structure_proved"
    elif straight_safe:
        selected_model, selected_seam = "shifted_straight", straight
        reason = (
            "dp_held_out_structure_not_proved"
            if allowed_rank >= 2 and dp is not None
            else "straight_held_out_structure_proved"
        )
    else:
        selected_model, selected_seam = "midpoint_straight", base
        reason = "straight_held_out_structure_not_proved"
    audit: dict[str, object] = {
        "input_maximum_model": maximum_model,
        "selected_model": selected_model,
        "selection_reason": reason,
        "base_seam_sha256": _sha_array(base),
        "shifted_straight_seam_sha256": None if straight is None else _sha_array(straight),
        "dp_seam_sha256": None if dp is None else _sha_array(dp),
        "selected_seam_sha256": _sha_array(selected_seam),
        "straight_vs_base": straight_audit,
        "dp_vs_straight": dp_audit,
    }
    return selected_model, selected_seam.copy(), audit


def estimate_s13_m5_transactions(
    schedule: S012Schedule,
    calibration: CameraIntrinsics,
    image_loader: Callable[[int], np.ndarray],
    vertical: S13VerticalSolution,
    *,
    parent_stage_sha256: str,
    maximum_seam_shift_px: int = 8,
) -> tuple[S13M5Pair, ...]:
    """Estimate every pair independently and atomically from immutable P0 grids."""

    validate_s012_schedule(schedule)
    pairs: list[S13M5Pair] = []
    raw_cache: dict[int, np.ndarray] = {}

    def raw(frame_id: int) -> np.ndarray:
        if frame_id not in raw_cache:
            raw_cache[frame_id] = np.asarray(image_loader(frame_id))
        return raw_cache[frame_id]

    for pair_index, (left_assignment, right_assignment) in enumerate(
        zip(schedule.assignments[:-1], schedule.assignments[1:])
    ):
        frame_ids = (left_assignment.frame_id, right_assignment.frame_id)
        try:
            x0, x1 = _pair_domain(schedule, pair_index)
            if x1 - x0 < 12:
                raise ValueError("final_corridor_too_narrow")
            left_maps = _map_crop(
                schedule, calibration, pair_index, x0, x1,
                vertical.global_offsets_px[pair_index], None,
            )
            right_maps = _map_crop(
                schedule, calibration, pair_index + 1, x0, x1,
                vertical.global_offsets_px[pair_index + 1], None,
            )
            left_image, left_valid = _sample_crop(raw(frame_ids[0]), left_maps)
            right_image, right_valid = _sample_crop(raw(frame_ids[1]), right_maps)
            reference, moving = _pair_correspondences(
                left_image, right_image, left_valid, right_valid, x_offset=x0
            )
            p0_u, p0_v, p0_valid = _base_calibrated_map(schedule, calibration, pair_index + 1)
            allowed_left, allowed_right = _allowed_application_bounds(schedule, pair_index)
            base_seam = np.full(schedule.canvas_height, schedule.boundaries[pair_index + 1], dtype=np.int32)
            base_band = S13ApplicationBand.straight(
                height=schedule.canvas_height, left_x=allowed_left, right_x=allowed_right
            )
            local_vertical = -np.asarray(vertical.local_row_residuals[pair_index], dtype=np.float64)
            preliminary = estimate_s13_pair_alignment(
                pair_index=pair_index,
                pair_frame_ids=frame_ids,
                non_reference_side="right",
                p0_source_u=p0_u,
                p0_source_v=p0_v,
                p0_valid=p0_valid,
                source_size=(int(calibration.width), int(calibration.height)),
                reference_points_xy=reference,
                non_reference_points_xy=moving,
                application_band=base_band,
                accepted_vertical_dy_by_row=local_vertical,
                alignment_shoulder=(x0, x1),
                vertical_accepted=bool(np.any(local_vertical != 0.0)),
            )
            right_geometry_maps = _map_crop(
                schedule, calibration, pair_index + 1, x0, x1,
                vertical.global_offsets_px[pair_index + 1], preliminary.selected,
            )
            right_geometry, right_geometry_valid = _sample_crop(raw(frame_ids[1]), right_geometry_maps)
            base_local = schedule.boundaries[pair_index + 1] - x0
            previous = schedule.boundaries[pair_index] - x0 if pair_index > 0 else None
            following = schedule.boundaries[pair_index + 2] - x0 if pair_index + 2 < len(schedule.boundaries) else None
            seam_result: S13SeamResult = select_s13_seam(
                left_image, right_geometry, left_valid, right_geometry_valid, base_local,
                maximum_shift_px=maximum_seam_shift_px,
                previous_boundary_x=previous,
                next_boundary_x=following,
            )
            selected_seam_model, selected_seam_local, preliminary_seam_audit = (
                _select_held_out_safe_seam(
                    left_image,
                    right_geometry,
                    seam_result,
                    seam_result.model,
                )
            )
            held_out_seam_audits: list[dict[str, object]] = [
                {"geometry_stage": "preliminary", **preliminary_seam_audit}
            ]
            final_reference, final_moving = _pair_correspondences(
                left_image, right_image, left_valid, right_valid, x_offset=x0
            )
            allowed_left_rows = np.full(schedule.canvas_height, allowed_left, dtype=np.int32)
            allowed_right_rows = np.full(schedule.canvas_height, allowed_right, dtype=np.int32)
            for geometry_iteration in range(3):
                final_seam = selected_seam_local + x0
                final_alignment = reestimate_s13_final_corridor_alignment(
                    final_seam_x_by_row=final_seam,
                    application_half_width_px=max(
                        2, min(8, (allowed_right - allowed_left - 1) // 2)
                    ),
                    allowed_left_x_by_row=allowed_left_rows,
                    allowed_right_x_by_row=allowed_right_rows,
                    pair_index=pair_index,
                    pair_frame_ids=frame_ids,
                    non_reference_side="right",
                    p0_source_u=p0_u,
                    p0_source_v=p0_v,
                    p0_valid=p0_valid,
                    source_size=(int(calibration.width), int(calibration.height)),
                    reference_points_xy=final_reference,
                    non_reference_points_xy=final_moving,
                    accepted_vertical_dy_by_row=local_vertical,
                    alignment_shoulder=(x0, x1),
                    vertical_accepted=bool(np.any(local_vertical != 0.0)),
                )
                final_right_maps = _map_crop(
                    schedule,
                    calibration,
                    pair_index + 1,
                    x0,
                    x1,
                    vertical.global_offsets_px[pair_index + 1],
                    final_alignment.selected,
                )
                final_right, final_right_valid = _sample_crop(
                    raw(frame_ids[1]), final_right_maps
                )
                audited_model, audited_seam, final_seam_audit = (
                    _select_held_out_safe_seam(
                        left_image,
                        final_right,
                        seam_result,
                        selected_seam_model,
                    )
                )
                held_out_seam_audits.append(
                    {
                        "geometry_stage": f"final_iteration_{geometry_iteration}",
                        **final_seam_audit,
                    }
                )
                if audited_model == selected_seam_model and np.array_equal(
                    audited_seam, selected_seam_local
                ):
                    break
                selected_seam_model, selected_seam_local = audited_model, audited_seam
            else:
                raise RuntimeError("held_out_seam_selection_did_not_converge")
            # The before and after previews must use the exact same owner
            # topology.  Otherwise the DP path can lower its own audit score
            # merely by moving to easier pixels without improving alignment.
            before_preview = _compose_pair_preview(
                left_image, right_image, selected_seam_local
            )
            after_preview = _compose_pair_preview(
                left_image, final_right, selected_seam_local
            )
            before_metrics = seam_structure_metrics(
                before_preview, selected_seam_local
            )
            candidate_after_metrics = seam_structure_metrics(
                after_preview, selected_seam_local
            )
            structure_ok, structure_reason = structurally_non_degrading(
                before_metrics, candidate_after_metrics
            )
            before_horizontal = long_horizontal_structure_metrics(
                before_preview, selected_seam_local
            )
            candidate_after_horizontal = long_horizontal_structure_metrics(
                after_preview, selected_seam_local
            )
            horizontal_ok, horizontal_reason = long_horizontal_structure_nondegrading(
                before_horizontal, candidate_after_horizontal
            )
            applied = bool(structure_ok and horizontal_ok)
            rollback_reason = structure_reason if not structure_ok else horizontal_reason
            after_metrics = candidate_after_metrics if applied else before_metrics
            after_horizontal = candidate_after_horizontal if applied else before_horizontal
            support = left_valid & final_right_valid
            selected = final_alignment.selected
            core: dict[str, object] = {
                "transaction_id": f"m5-pair-{pair_index:04d}",
                "parent_stage_sha256": parent_stage_sha256,
                "pair_frame_ids": list(frame_ids),
                "motion_hypothesis_id": -1,
                "support_mask_sha256": _sha_array(support),
                "geometry_model": selected.model if applied else "C1_accepted_vertical_parent",
                "seam_model": selected_seam_model if applied else "S0_midpoint_straight",
                "map_delta_sha256": _sha_array(selected.target_delta_u, selected.target_delta_v) if applied else None,
                "decision": "applied" if applied else "rolled_back",
                "before_metrics": before_metrics,
                "candidate_after_metrics": candidate_after_metrics,
                "after_metrics": after_metrics,
                "before_horizontal_continuity": before_horizontal,
                "candidate_after_horizontal_continuity": candidate_after_horizontal,
                "after_horizontal_continuity": after_horizontal,
                "rollback_reason": None if applied else rollback_reason,
                "comparison_coordinate_policy": (
                    "same_owner_geometry_plus_symmetric_base_candidate_seam"
                ),
                "geometry_comparison_seam_sha256": _sha_array(
                    selected_seam_local + x0
                ),
                "geometry_before_preview_sha256": _sha_array(before_preview),
                "geometry_after_preview_sha256": _sha_array(after_preview),
                "final_corridor_geometry_reestimated_from_immutable_p0": True,
                "reference_side": "left",
                "non_reference_side": "right",
                "alignment_shoulder_width_px": x1 - x0,
                "application_band_left_min": int(final_alignment.application_band.left_x_by_row.min()),
                "application_band_right_max": int(final_alignment.application_band.right_x_by_row.max()),
                "geometry_candidates": [
                    {
                        "model": candidate.model,
                        "accepted": candidate.accepted,
                        "failure_reason": candidate.failure_reason,
                        "metrics": dict(candidate.metrics),
                        "audit": dict(candidate.audit),
                    }
                    for candidate in final_alignment.candidates
                ],
                "base_seam_sha256": _sha_array(
                    seam_result.candidate_seams["midpoint_straight"] + x0
                ),
                "shifted_straight_seam_sha256": (
                    None
                    if "shifted_straight" not in seam_result.candidate_seams
                    else _sha_array(
                        seam_result.candidate_seams["shifted_straight"] + x0
                    )
                ),
                "dp_seam_sha256": (
                    None
                    if "monotone_dp" not in seam_result.candidate_seams
                    else _sha_array(seam_result.candidate_seams["monotone_dp"] + x0)
                ),
                "selected_seam_sha256": _sha_array(selected_seam_local + x0),
                "result_seam_sha256": _sha_array(
                    selected_seam_local + x0 if applied else base_seam
                ),
                "held_out_seam_selection_audits": held_out_seam_audits,
                "seam_audit": {
                    **dict(seam_result.audit),
                    "held_out_selected_model": selected_seam_model,
                },
                "seam_cost_components": dict(
                    seam_result.candidate_cost_components[selected_seam_model]
                ),
                "seam_fallback_chain": list(seam_result.fallback_chain),
            }
            core["result_stage_sha256"] = _sha_json(core)
            pairs.append(S13M5Pair(
                transaction=core,
                seam_x_by_row=selected_seam_local + x0 if applied else base_seam,
                alignment=final_alignment if applied else None,
            ))
        except Exception as exc:
            pairs.append(_fallback_pair(
                schedule, pair_index, parent_stage_sha256, frame_ids,
                f"{type(exc).__name__}:{exc}",
            ))
    return tuple(pairs)


def _seams_array(schedule: S012Schedule, pairs: Sequence[S13M5Pair], *, final: bool) -> np.ndarray:
    seams = []
    for index, pair in enumerate(pairs):
        if final:
            seams.append(np.asarray(pair.seam_x_by_row, dtype=np.int32))
        else:
            seams.append(np.full(schedule.canvas_height, schedule.boundaries[index + 1], dtype=np.int32))
    if not seams:
        return np.empty((0, schedule.canvas_height), dtype=np.int32)
    result = np.stack(seams)
    if np.any(np.diff(result, axis=0) < 0):
        raise ValueError("S1.3 M5 adjacent seams cross")
    if np.any(np.abs(result - np.asarray(schedule.boundaries[1:-1])[:, None]) > 8):
        raise ValueError("S1.3 M5 seam exceeds maximum shift")
    return result


def render_s13_p2_from_raw(
    schedule: S012Schedule,
    calibration: CameraIntrinsics,
    image_loader: Callable[[int], np.ndarray],
    vertical: S13VerticalSolution,
    pairs: Sequence[S13M5Pair],
    *,
    final_seams: bool,
    selected_hypothesis_ids: tuple[int, ...] | None = None,
    placement_methods: tuple[str, ...] | None = None,
) -> S13P2Result:
    """Formally remap each real contributor once from raw RGB for this P2 asset."""

    validate_s012_schedule(schedule)
    if len(pairs) != len(schedule.assignments) - 1:
        raise ValueError("S1.3 M5 pair transactions do not cover all render pairs")
    seams = _seams_array(schedule, pairs, final=final_seams)
    height, width = schedule.canvas_height, schedule.canvas_width
    owner_index = np.zeros((height, width), dtype=np.int32)
    for seam in seams:
        owner_index += np.arange(width, dtype=np.int32)[None, :] >= seam[:, None]
    output = np.zeros((height, width, 3), dtype=np.uint8)
    valid_full = np.zeros((height, width), dtype=bool)
    owner = np.full((height, width), -1, dtype=np.int32)
    assignment_full = np.full((height, width), -1, dtype=np.int32)
    source_u_full = np.full((height, width), np.nan, dtype=np.float32)
    source_v_full = np.full((height, width), np.nan, dtype=np.float32)
    geometry_tx = np.full((height, width), -1, dtype=np.int32)
    seam_tx = np.full((height, width), -1, dtype=np.int32)
    hypothesis = np.full((height, width), -1, dtype=np.int32)
    placement = np.full((height, width), -1, dtype=np.int16)
    decoded: list[int] = []
    for source_index, assignment in enumerate(schedule.assignments):
        mask = owner_index == source_index
        if not np.any(mask):
            continue
        columns = np.flatnonzero(np.any(mask, axis=0))
        x0, x1 = int(columns[0]), int(columns[-1]) + 1
        candidate = None
        if source_index > 0 and pairs[source_index - 1].alignment is not None:
            candidate = pairs[source_index - 1].alignment.selected
        maps = _map_crop(
            schedule, calibration, source_index, x0, x1,
            vertical.global_offsets_px[source_index], candidate,
        )
        raw = np.asarray(image_loader(assignment.frame_id))
        decoded.append(assignment.frame_id)
        if raw.shape != (height, int(calibration.width), 3) or raw.dtype != np.uint8:
            raise ValueError("S1.3 M5 raw RGB source shape/type changed")
        sampled, map_valid = _sample_crop(raw, maps)
        owned = mask[:, x0:x1] & map_valid
        roi = np.s_[:, x0:x1]
        output[roi][owned] = sampled[owned]
        valid_full[roi][owned] = True
        owner[roi][owned] = assignment.frame_id
        assignment_full[roi][owned] = source_index
        source_u_full[roi][owned] = maps[0][owned]
        source_v_full[roi][owned] = maps[1][owned]
        if source_index > 0:
            pair_index = source_index - 1
            geometry_tx[roi][owned] = pair_index
            seam_tx[roi][owned] = pair_index
        if selected_hypothesis_ids is not None:
            hypothesis[roi][owned] = int(selected_hypothesis_ids[source_index])
        placement[roi][owned] = source_index
    if len(decoded) != len(set(decoded)):
        raise ValueError("S1.3 M5 decoded one contributor more than once")
    pixel = {
        "owner_frame_id": owner,
        "owner_source_index": assignment_full.copy(),
        "assignment_index": assignment_full,
        "source_u": source_u_full,
        "source_v": source_v_full,
        "valid": valid_full,
        "selected_motion_hypothesis_id": hypothesis,
        "placement_method_code": placement,
        "placement_method_names": np.asarray(placement_methods or ()),
        "geometry_transaction_id": geometry_tx,
        "seam_transaction_id": seam_tx,
        "photometric_transaction_id": np.full((height, width), -1, np.int32),
        "blend_transaction_id": np.full((height, width), -1, np.int32),
        "secondary_frame_id": np.full((height, width), -1, np.int32),
        "secondary_source_u": np.full((height, width), np.nan, np.float32),
        "secondary_source_v": np.full((height, width), np.nan, np.float32),
        "secondary_weight": np.zeros((height, width), np.float32),
    }
    if not np.array_equal(valid_full, owner >= 0):
        raise ValueError("S1.3 M5 valid and owner topology disagree")
    finite = valid_full & (~np.isfinite(source_u_full) | ~np.isfinite(source_v_full))
    if np.any(finite):
        raise ValueError("S1.3 M5 valid source provenance is nonfinite")
    return S13P2Result(output, valid_full, pixel, len(decoded), tuple(decoded))


def _overlay_seams(image: np.ndarray, pairs: Sequence[S13M5Pair]) -> np.ndarray:
    overlay = image.copy()
    for pair in pairs:
        seam = np.asarray(pair.seam_x_by_row, dtype=np.int32)
        rows = np.arange(len(seam), dtype=np.int32)
        valid = (seam >= 0) & (seam < image.shape[1])
        overlay[rows[valid], seam[valid]] = (0, 0, 255)
    return overlay


def run_s13_m5(
    schedule: S012Schedule,
    calibration: CameraIntrinsics,
    image_loader: Callable[[int], np.ndarray],
    vertical: S13VerticalSolution,
    vertical_parent_image: np.ndarray,
    *,
    parent_stage_sha256: str,
    selected_hypothesis_ids: tuple[int, ...] | None = None,
    placement_methods: tuple[str, ...] | None = None,
) -> S13M5Result:
    started = time.perf_counter()
    tick = time.perf_counter()
    pairs = estimate_s13_m5_transactions(
        schedule, calibration, image_loader, vertical, parent_stage_sha256=parent_stage_sha256
    )
    geometry_seconds = time.perf_counter() - tick
    tick = time.perf_counter()
    geometry = render_s13_p2_from_raw(
        schedule, calibration, image_loader, vertical, pairs, final_seams=False,
        selected_hypothesis_ids=selected_hypothesis_ids, placement_methods=placement_methods,
    )
    final = render_s13_p2_from_raw(
        schedule, calibration, image_loader, vertical, pairs, final_seams=True,
        selected_hypothesis_ids=selected_hypothesis_ids, placement_methods=placement_methods,
    )
    seam_seconds = time.perf_counter() - tick
    # Geometry is compared on one owner topology.  Seam ownership is then
    # independently compared against the fixed-boundary render on both the
    # base and candidate paths, so neither half can authorize the other.
    before_metrics: list[Mapping[str, object]] = []
    after_metrics: list[Mapping[str, object]] = []
    audited_pair_indices: list[int] = []
    neutral_unevaluable_rollback_indices: list[int] = []
    applied_unevaluable_indices: list[int] = []
    horizontal_failure_indices: list[int] = []
    for pair_index, pair in enumerate(pairs):
        transaction = pair.transaction
        before = transaction.get("before_metrics")
        after = transaction.get("after_metrics")
        if (
            isinstance(before, Mapping)
            and isinstance(after, Mapping)
            and before.get("evaluable") is True
            and after.get("evaluable") is True
        ):
            before_metrics.append(before)
            after_metrics.append(after)
            audited_pair_indices.append(pair_index)
        elif transaction.get("decision") == "applied":
            applied_unevaluable_indices.append(pair_index)
        else:
            neutral_unevaluable_rollback_indices.append(pair_index)
        before_horizontal = transaction.get("before_horizontal_continuity")
        after_horizontal = transaction.get("after_horizontal_continuity")
        if isinstance(before_horizontal, Mapping) and isinstance(after_horizontal, Mapping):
            horizontal_ok, _reason = long_horizontal_structure_nondegrading(
                before_horizontal, after_horizontal
            )
            if not horizontal_ok:
                horizontal_failure_indices.append(pair_index)

    geometry_sequence_selected, geometry_selection_audit = sequence_structure_decision(
        before_metrics, after_metrics
    ) if before_metrics else (False, {"eligible": False, "reason": "no_evaluable_pair_transactions"})

    seam_output_pair_audits: list[dict[str, object]] = []
    seam_output_before: list[Mapping[str, object]] = []
    seam_output_after: list[Mapping[str, object]] = []
    seam_output_failure_indices: list[int] = []
    for pair_index, pair in enumerate(pairs):
        base_seam = np.full(
            schedule.canvas_height,
            schedule.boundaries[pair_index + 1],
            dtype=np.int32,
        )
        audit = _audit_rendered_seam_change(
            geometry.image,
            final.image,
            base_seam,
            np.asarray(pair.seam_x_by_row, dtype=np.int32),
        )
        seam_output_pair_audits.append({"pair_index": pair_index, **audit})
        symmetric_before = audit["symmetric_before_metrics"]
        symmetric_after = audit["symmetric_after_metrics"]
        if isinstance(symmetric_before, Mapping) and isinstance(symmetric_after, Mapping):
            seam_output_before.append(symmetric_before)
            seam_output_after.append(symmetric_after)
        if audit["eligible"] is not True:
            seam_output_failure_indices.append(pair_index)
    seam_output_sequence_selected, seam_output_sequence_audit = (
        sequence_structure_decision(
            seam_output_before,
            seam_output_after,
            maximum_component_failure_fraction=0.0,
            maximum_pair_score_failure_fraction=0.0,
            minimum_mean_improvement_fraction=0.0,
        )
        if seam_output_before
        else (False, {"eligible": False, "reason": "no_evaluable_seam_output_pairs"})
    )

    selected = bool(
        geometry_sequence_selected
        and not applied_unevaluable_indices
        and not horizontal_failure_indices
        and seam_output_sequence_selected
        and not seam_output_failure_indices
    )
    selection_reason = None
    if not geometry_sequence_selected:
        selection_reason = geometry_selection_audit.get("reason")
    elif applied_unevaluable_indices:
        selection_reason = "applied_geometry_transaction_unevaluable"
    elif horizontal_failure_indices:
        selection_reason = "geometry_horizontal_continuity_failure"
    elif not seam_output_sequence_selected:
        selection_reason = seam_output_sequence_audit.get("reason")
    elif seam_output_failure_indices:
        selection_reason = "seam_output_pair_non_degradation_not_proven"

    selection_audit = {
        **dict(geometry_selection_audit),
        "eligible": selected,
        "reason": selection_reason,
        "comparison_coordinate_policy": (
            "same_owner_geometry_plus_symmetric_base_candidate_seam"
        ),
        "geometry_sequence_eligible": bool(geometry_sequence_selected),
        "geometry_sequence_audit": dict(geometry_selection_audit),
        "audited_pair_indices": audited_pair_indices,
        "neutral_unevaluable_rollback_indices": neutral_unevaluable_rollback_indices,
        "applied_unevaluable_indices": applied_unevaluable_indices,
        "horizontal_continuity_failure_indices": horizontal_failure_indices,
        "seam_output_sequence_eligible": bool(seam_output_sequence_selected),
        "seam_output_sequence_audit": dict(seam_output_sequence_audit),
        "seam_output_failure_indices": seam_output_failure_indices,
        "seam_output_pair_audits": seam_output_pair_audits,
    }
    before_mean_value = selection_audit.get("before_mean_score")
    after_mean_value = selection_audit.get("after_mean_score")
    before_mean = float(before_mean_value) if isinstance(before_mean_value, (int, float)) else None
    after_mean = float(after_mean_value) if isinstance(after_mean_value, (int, float)) else None
    return S13M5Result(
        pairs=pairs,
        geometry_result=geometry,
        final_result=final,
        seam_overlay=_overlay_seams(final.image, pairs),
        selected_as_best=selected,
        before_mean_score=before_mean,
        after_mean_score=after_mean,
        selection_audit=selection_audit,
        performance={
            "geometry": geometry_seconds,
            "seam_and_p2_render": seam_seconds,
            "total_m5": time.perf_counter() - started,
        },
    )


__all__ = [
    "S13M5Pair", "S13M5Result", "S13P2Result", "estimate_s13_m5_transactions",
    "render_s13_p2_from_raw", "run_s13_m5",
]
