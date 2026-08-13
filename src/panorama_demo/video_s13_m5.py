"""M5 geometry/seam transactions and owner-only P2 rendering for S1.3."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import dataclass, replace
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
    SeamStructureFeatures,
    prepare_seam_structure,
    long_horizontal_structure_metrics,
    long_horizontal_structure_nondegrading,
    edge_registration_visual_suspect,
    pair_edge_registration_metrics,
    seam_structure_metrics,
    sequence_structure_decision,
    structurally_non_degrading,
    symmetric_seam_structure_metrics,
)
from .video_s13_seam import (
    S13SeamCandidate,
    S13SeamResult,
    build_s13_seam_candidates,
    rank_s13_seam_candidates,
    select_s13_seam,
)
from .video_s13_replay import S13P2ReplayPair
from .video_s13_m51_r2 import S13M51R2Config
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
    expected_support_mask: np.ndarray


@dataclass(frozen=True)
class S13M5Result:
    pairs: tuple[S13M5Pair, ...]
    replay_pairs: tuple[S13P2ReplayPair, ...]
    geometry_result: S13P2Result
    final_result: S13P2Result
    seam_overlay: np.ndarray
    hard_audit_passed: bool
    hard_audit: Mapping[str, object]
    diagnostic_quality: Mapping[str, object]
    selected_as_best: bool
    before_mean_score: float | None
    after_mean_score: float | None
    selection_audit: Mapping[str, object]
    performance: Mapping[str, float]


@dataclass(frozen=True)
class S13PairCorrespondences:
    reference_xy: np.ndarray
    moving_xy: np.ndarray
    audit: Mapping[str, object]

    def __iter__(self):
        """Keep legacy internal two-value unpacking while exposing the audit."""

        yield self.reference_xy
        yield self.moving_xy


def _sha_array(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(value.shape).encode("ascii"))
        digest.update(value.tobytes())
    return digest.hexdigest()


def _sha_json(value: Mapping[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


_V5_TRANSACTION_FLOAT_DECIMAL_PLACES = 6


def _canonicalize_v5_transaction_value(value: object) -> object:
    """Return the frozen JSON value representation used by M5.1-r2/v5.

    Candidate selection consumes the full-precision in-memory metrics before
    this function is called.  Quantization therefore affects only the sealed
    audit representation and its digest, not pixels, thresholds, or ranking.
    """

    if isinstance(value, Mapping):
        return {
            str(key): _canonicalize_v5_transaction_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_canonicalize_v5_transaction_value(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("S1.3 v5 transaction floats must be finite")
        rounded = round(number, _V5_TRANSACTION_FLOAT_DECIMAL_PLACES)
        return 0.0 if rounded == 0.0 else rounded
    return value


def _finalize_v5_transaction(value: Mapping[str, object]) -> dict[str, object]:
    """Canonicalize a v5 transaction and bind its normalized content hash."""

    without_digest = dict(value)
    without_digest.pop("result_stage_sha256", None)
    normalized = _canonicalize_v5_transaction_value(without_digest)
    if not isinstance(normalized, dict):  # pragma: no cover - Mapping guarantees this
        raise TypeError("S1.3 v5 transaction must be a mapping")
    normalized["result_stage_sha256"] = _sha_json(normalized)
    return normalized


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
    config: S13M51R2Config | None = None,
) -> S13PairCorrespondences:
    settings = config or S13M51R2Config()
    left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
    mask = (left_valid & right_valid).astype(np.uint8) * 255
    points = cv2.goodFeaturesToTrack(left_gray, 400, 0.01, 4.0, mask=mask, blockSize=5)
    empty = np.empty((0, 2), np.float64)
    empty_audit: dict[str, object] = {
        "detected_count": 0,
        "status_count": 0,
        "finite_count": 0,
        "target_in_bounds_count": 0,
        "target_valid_count": 0,
        "error_filtered_count": 0,
        "final_count": 0,
        "error_median": None,
        "error_mad": None,
        "error_limit": None,
        "raw_error_count": 0,
        "raw_error_median": None,
        "raw_error_mad": None,
        "raw_error_p50": None,
        "raw_error_p90": None,
        "raw_error_p95": None,
        "raw_error_p99": None,
        "raw_error_maximum": None,
        "filter_enabled": bool(settings.enabled),
        "legacy_decision_preserved": not settings.enabled,
        "gftt_call_count": 1,
        "forward_pyr_lk_call_count": 0,
        "backward_pyr_lk_call_count": 0,
    }
    if points is None:
        return S13PairCorrespondences(empty.copy(), empty.copy(), empty_audit)
    moved, status, error = cv2.calcOpticalFlowPyrLK(
        left_gray, right_gray, points, None, winSize=(21, 21), maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    empty_audit["detected_count"] = int(np.asarray(points).reshape(-1, 2).shape[0])
    empty_audit["forward_pyr_lk_call_count"] = 1
    if moved is None or status is None:
        return S13PairCorrespondences(empty.copy(), empty.copy(), empty_audit)
    source = np.asarray(points).reshape(-1, 2).astype(np.float64)
    target = np.asarray(moved).reshape(-1, 2).astype(np.float64)
    status_values = np.asarray(status).reshape(-1) > 0
    if len(source) != len(target) or len(source) != len(status_values):
        raise ValueError("S1.3 M5 PyrLK result count disagrees with detected points")
    finite_values = np.isfinite(source).all(axis=1) & np.isfinite(target).all(axis=1)
    legacy_keep = status_values & finite_values

    target_in_bounds = np.zeros(len(source), dtype=bool)
    target_in_bounds[legacy_keep] = (
        (target[legacy_keep, 0] >= 0.0)
        & (target[legacy_keep, 0] <= right_valid.shape[1] - 1.0)
        & (target[legacy_keep, 1] >= 0.0)
        & (target[legacy_keep, 1] <= right_valid.shape[0] - 1.0)
    )
    target_valid = np.zeros(len(source), dtype=bool)
    indexable = legacy_keep & target_in_bounds
    if np.any(indexable):
        # Finite/in-bounds is deliberately established before round/cast/index.
        tx = np.rint(target[indexable, 0]).astype(np.int32)
        ty = np.rint(target[indexable, 1]).astype(np.int32)
        target_valid[indexable] = np.asarray(right_valid, dtype=bool)[ty, tx]
    hard_keep = legacy_keep & target_in_bounds & target_valid

    error_values = None if error is None else np.asarray(error).reshape(-1).astype(np.float64)
    if error_values is not None and len(error_values) != len(source):
        raise ValueError("S1.3 M5 PyrLK error count disagrees with detected points")
    raw_error = (
        np.empty(0, dtype=np.float64)
        if error_values is None
        else error_values[legacy_keep & np.isfinite(error_values)]
    )
    distribution: dict[str, object] = {
        "raw_error_count": int(raw_error.size),
        "raw_error_median": None,
        "raw_error_mad": None,
        "raw_error_p50": None,
        "raw_error_p90": None,
        "raw_error_p95": None,
        "raw_error_p99": None,
        "raw_error_maximum": None,
    }
    if raw_error.size and settings.collect_lk_error_statistics:
        raw_median = float(np.median(raw_error))
        distribution.update({
            "raw_error_median": raw_median,
            "raw_error_mad": float(np.median(np.abs(raw_error - raw_median))),
            "raw_error_p50": float(np.percentile(raw_error, 50.0)),
            "raw_error_p90": float(np.percentile(raw_error, 90.0)),
            "raw_error_p95": float(np.percentile(raw_error, 95.0)),
            "raw_error_p99": float(np.percentile(raw_error, 99.0)),
            "raw_error_maximum": float(np.max(raw_error)),
        })

    final_keep = legacy_keep.copy()
    error_median = error_mad = error_limit = None
    error_filtered_count = int(hard_keep.sum())
    if settings.enabled:
        final_keep = hard_keep.copy()
        valid_error = (
            np.empty(0, dtype=np.float64)
            if error_values is None
            else error_values[hard_keep & np.isfinite(error_values)]
        )
        if valid_error.size >= settings.minimum_correspondence_count_for_error_filter:
            error_median = float(np.median(valid_error))
            error_mad = float(np.median(np.abs(valid_error - error_median)))
            sigma = 1.4826 * error_mad
            assert settings.lk_error_mad_floor is not None
            assert settings.lk_error_absolute_maximum is not None
            robust_limit = error_median + settings.lk_error_sigma_multiplier * max(
                sigma, settings.lk_error_mad_floor
            )
            error_limit = float(min(settings.lk_error_absolute_maximum, robust_limit))
            error_keep = np.isfinite(error_values) & (error_values <= error_limit)  # type: ignore[operator]
            final_keep &= error_keep
            error_filtered_count = int(final_keep.sum())

    selected_source = source[final_keep].copy()
    selected_target = target[final_keep].copy()
    selected_source[:, 0] += int(x_offset)
    selected_target[:, 0] += int(x_offset)
    audit: dict[str, object] = {
        "detected_count": int(len(source)),
        "status_count": int(status_values.sum()),
        "finite_count": int(legacy_keep.sum()),
        "target_in_bounds_count": int(indexable.sum()),
        "target_valid_count": int(hard_keep.sum()),
        "error_filtered_count": error_filtered_count,
        "final_count": int(final_keep.sum()),
        "error_median": error_median,
        "error_mad": error_mad,
        "error_limit": error_limit,
        **distribution,
        "filter_enabled": bool(settings.enabled),
        "legacy_decision_preserved": not settings.enabled,
        "gftt_call_count": 1,
        "forward_pyr_lk_call_count": 1,
        "backward_pyr_lk_call_count": 0,
    }
    return S13PairCorrespondences(selected_source, selected_target, audit)


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
    parent_result_sha256: str | None = None,
    p0_ancestor_completion_sha256: str | None = None,
    before: Mapping[str, object] | None = None,
    after: Mapping[str, object] | None = None,
    correspondence_filter: Mapping[str, object] | None = None,
    m51_r2_config: S13M51R2Config | None = None,
    topology_repair_used: bool = False,
) -> S13M5Pair:
    seam = np.full(schedule.canvas_height, schedule.boundaries[pair_index + 1], dtype=np.int32)
    core: dict[str, object] = {
        "schema": (
            "gemini305-video-s13-m5-pair-transaction/v3"
            if m51_r2_config is not None and m51_r2_config.enabled
            else "gemini305-video-s13-m5-pair-transaction/v2"
        ),
        "transaction_id": f"m5-pair-{pair_index:04d}",
        "parent_stage": "P1",
        "parent_stage_sha256": parent_sha,
        "parent_completion_sha256": parent_sha,
        "parent_result_sha256": parent_result_sha256,
        "p0_ancestor_completion_sha256": p0_ancestor_completion_sha256,
        "pair_frame_ids": list(frame_ids),
        "motion_hypothesis_id": -1,
        "support_mask_sha256": None,
        "geometry_model": "C1_accepted_vertical_parent",
        "seam_model": "S0_midpoint_straight",
        "map_delta_sha256": None,
        "decision": "rolled_back",
        "selection_policy": "minimum_local_objective_among_hard_safe_candidates",
        "selected_candidate_id": f"pair-{pair_index:04d}-S0-fallback",
        "selected_seam_model": "midpoint_straight",
        "selected_geometry_model": "C0_identity",
        "pair_hard_audit_passed": True,
        "fallback_used": True,
        "fallback_reason": reason,
        "candidate_generation": [],
        "candidate_evaluations": [],
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
    if correspondence_filter is not None:
        core["correspondence_filter"] = dict(correspondence_filter)
    if m51_r2_config is not None and m51_r2_config.enabled:
        core.update({
            "correspondence_filter": dict(correspondence_filter or {}),
            "seam_local_evidence": {
                "requested_half_widths_px": list(m51_r2_config.evidence_half_widths_px),
                "selected_half_width_px": None, "raw_count": 0,
                "selected_count": 0, "minimum_required_count": 36,
                "sufficient_for_train_held_out": False,
                "full_shoulder_fallback_used": False,
                "mode": "seam_local_insufficient",
            },
            "edge_registration": {
                "evaluable": False, "visual_suspect": False,
                "supported_block_count": 0, "supported_component_count": 0,
                "median_supported_abs_lag_px": None,
                "p95_supported_abs_lag_px": None, "ambiguous": False,
            },
            "selection_continued_for_structure": False,
            "selected_seam_rank": 0, "selected_geometry_rank": 0,
            "seam_fallback_used": True, "geometry_fallback_used": True,
            "topology_repair_used": bool(topology_repair_used),
            "unresolved_oblique_structure": False,
            "unexpected_exception_fallback": False,
            "micro_rescue": {"enabled": False, "attempted": False, "accepted": False},
        })
    if m51_r2_config is not None and m51_r2_config.enabled:
        core = _finalize_v5_transaction(core)
    else:
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
    *,
    before_features: SeamStructureFeatures | None = None,
    after_features: SeamStructureFeatures | None = None,
) -> dict[str, object]:
    """Audit two rendered owner topologies on both paths and held-out shoulders."""

    before_seam = np.asarray(before_seam_x_by_row, dtype=np.int32)
    after_seam = np.asarray(after_seam_x_by_row, dtype=np.int32)
    cached_before = before_features or prepare_seam_structure(before_image)
    cached_after = after_features or prepare_seam_structure(after_image)
    if cached_before is None or cached_after is None:
        raise ValueError("S1.3 seam audit images cannot build structure features")
    symmetric_before, symmetric_after = symmetric_seam_structure_metrics(
        cached_before,
        cached_after,
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
        before_metrics = seam_structure_metrics(cached_before, path)
        after_metrics = seam_structure_metrics(cached_after, path)
        structure_ok, structure_reason = structurally_non_degrading(
            before_metrics, after_metrics
        )
        before_horizontal = long_horizontal_structure_metrics(cached_before, path)
        after_horizontal = long_horizontal_structure_metrics(cached_after, path)
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
    parent_result_sha256: str | None = None,
    p0_ancestor_completion_sha256: str | None = None,
    maximum_seam_shift_px: int = 8,
    m51_r2_config: S13M51R2Config | None = None,
) -> tuple[S13M5Pair, ...]:
    """Evaluate peer seam candidates independently from immutable P1/P0 grids."""

    validate_s012_schedule(schedule)
    pairs: list[S13M5Pair] = []
    raw_cache: dict[int, np.ndarray] = {}
    successor = m51_r2_config or S13M51R2Config()
    instrumentation_requested = m51_r2_config is not None
    audit_all = os.environ.get("G305_S13_M5_AUDIT_ALL_CANDIDATES", "0") == "1"

    def raw(frame_id: int) -> np.ndarray:
        if frame_id not in raw_cache:
            raw_cache[frame_id] = np.asarray(image_loader(frame_id))
        return raw_cache[frame_id]

    for pair_index, (left_assignment, right_assignment) in enumerate(
        zip(schedule.assignments[:-1], schedule.assignments[1:])
    ):
        frame_ids = (left_assignment.frame_id, right_assignment.frame_id)
        correspondence_audit: Mapping[str, object] | None = None
        try:
            x0, x1 = _pair_domain(schedule, pair_index)
            if x1 - x0 < 12:
                raise ValueError("final_corridor_too_narrow")
            left_maps = _map_crop(schedule, calibration, pair_index, x0, x1,
                                  vertical.global_offsets_px[pair_index], None)
            right_maps = _map_crop(schedule, calibration, pair_index + 1, x0, x1,
                                   vertical.global_offsets_px[pair_index + 1], None)
            left_image, left_valid = _sample_crop(raw(frame_ids[0]), left_maps)
            right_image, right_valid = _sample_crop(raw(frame_ids[1]), right_maps)
            correspondence_result = _pair_correspondences(
                left_image, right_image, left_valid, right_valid, x_offset=x0,
                config=successor,
            )
            reference, moving = correspondence_result
            correspondence_audit = correspondence_result.audit
            p0_u, p0_v, p0_valid = _base_calibrated_map(
                schedule, calibration, pair_index + 1
            )
            allowed_left, allowed_right = _allowed_application_bounds(schedule, pair_index)
            base_band = S13ApplicationBand.straight(
                height=schedule.canvas_height, left_x=allowed_left, right_x=allowed_right
            )
            local_vertical = -np.asarray(vertical.local_row_residuals[pair_index], dtype=np.float64)
            preliminary = estimate_s13_pair_alignment(
                pair_index=pair_index, pair_frame_ids=frame_ids, non_reference_side="right",
                p0_source_u=p0_u, p0_source_v=p0_v, p0_valid=p0_valid,
                source_size=(int(calibration.width), int(calibration.height)),
                reference_points_xy=reference, non_reference_points_xy=moving,
                application_band=base_band, accepted_vertical_dy_by_row=local_vertical,
                alignment_shoulder=(x0, x1),
                vertical_accepted=bool(np.any(local_vertical != 0.0)),
            )
            preliminary_maps = _map_crop(
                schedule, calibration, pair_index + 1, x0, x1,
                vertical.global_offsets_px[pair_index + 1], preliminary.selected,
            )
            preliminary_right, preliminary_valid = _sample_crop(
                raw(frame_ids[1]), preliminary_maps
            )
            base_local = schedule.boundaries[pair_index + 1] - x0
            previous = schedule.boundaries[pair_index] - x0 if pair_index > 0 else None
            following = (
                schedule.boundaries[pair_index + 2] - x0
                if pair_index + 2 < len(schedule.boundaries) else None
            )
            seam_result = select_s13_seam(
                left_image, preliminary_right, left_valid, preliminary_valid, base_local,
                maximum_shift_px=maximum_seam_shift_px,
                previous_boundary_x=previous, next_boundary_x=following,
            )
            candidates, generation_statuses = build_s13_seam_candidates(pair_index, seam_result)
            ordered = rank_s13_seam_candidates(candidates)
            allowed_left_rows = np.full(schedule.canvas_height, allowed_left, dtype=np.int32)
            allowed_right_rows = np.full(schedule.canvas_height, allowed_right, dtype=np.int32)
            evaluations: list[dict[str, object]] = []
            selected_candidate: S13SeamCandidate | None = None
            selected_alignment: S13PairAlignment | None = None
            selected_before: Mapping[str, object] | None = None
            selected_after: Mapping[str, object] | None = None
            selected_before_horizontal: Mapping[str, object] | None = None
            selected_after_horizontal: Mapping[str, object] | None = None
            selected_edge_registration: Mapping[str, object] = {
                "evaluable": False, "visual_suspect": False,
                "supported_block_count": 0, "supported_component_count": 0,
                "median_supported_abs_lag_px": None,
                "p95_supported_abs_lag_px": None, "ambiguous": False,
            }
            selected_seam_rank = 0
            selected_geometry_rank = 0
            selection_continued_for_structure = False
            unresolved_oblique_structure = False
            hard_safe_baseline: tuple[object, ...] | None = None
            for seam_rank, candidate in enumerate(ordered):
                if selected_candidate is not None and not audit_all:
                    evaluations.append({
                        "candidate_id": candidate.candidate_id,
                        "model_code": candidate.model_code,
                        "model_name": candidate.model_name,
                        "generation_status": "generated",
                        "evaluation_status": "skipped_after_higher_rank_safe_candidate",
                        "local_objective": candidate.local_objective,
                        "hard_gate_passed": None,
                        "hard_gate_failures": [],
                        "generation_audit": dict(candidate.generation_audit),
                    })
                    continue
                seam_local = np.asarray(candidate.seam_x_by_row, dtype=np.int32)
                seam_global = seam_local + x0
                failures: list[str] = []
                if seam_local.shape != (schedule.canvas_height,):
                    failures.append("seam_shape_invalid")
                if not np.isfinite(seam_local).all():
                    failures.append("seam_nonfinite")
                if np.any(np.abs(np.diff(seam_local)) > 1):
                    failures.append("seam_row_step_invalid")
                search_left = int(seam_result.audit["search_left_x"])
                search_right = int(seam_result.audit["search_right_x"])
                if np.any((seam_local < search_left) | (seam_local > search_right)):
                    failures.append("seam_out_of_search_bounds")
                alignment: S13PairAlignment | None = None
                before_metrics: Mapping[str, object] = {"evaluable": False, "score": None}
                after_metrics: Mapping[str, object] = {"evaluable": False, "score": None}
                before_horizontal: Mapping[str, object] = {"observed": False}
                after_horizontal: Mapping[str, object] = {"observed": False}
                geometry_candidates: list[dict[str, object]] = []
                map_delta_sha: str | None = None
                corridor_bounds: list[int] | None = None
                edge_registration: Mapping[str, object] = {
                    "evaluable": False, "visual_suspect": False,
                    "supported_block_count": 0, "supported_component_count": 0,
                    "median_supported_abs_lag_px": None,
                    "p95_supported_abs_lag_px": None, "ambiguous": False,
                }
                visual_suspect = False
                geometry_rank = 0
                if not failures:
                    alignment = reestimate_s13_final_corridor_alignment(
                        final_seam_x_by_row=seam_global,
                        application_half_width_px=max(2, min(8, (allowed_right - allowed_left - 1) // 2)),
                        allowed_left_x_by_row=allowed_left_rows,
                        allowed_right_x_by_row=allowed_right_rows,
                        pair_index=pair_index, pair_frame_ids=frame_ids, non_reference_side="right",
                        p0_source_u=p0_u, p0_source_v=p0_v, p0_valid=p0_valid,
                        source_size=(int(calibration.width), int(calibration.height)),
                        reference_points_xy=reference, non_reference_points_xy=moving,
                        accepted_vertical_dy_by_row=local_vertical, alignment_shoulder=(x0, x1),
                        vertical_accepted=bool(np.any(local_vertical != 0.0)),
                        m51_r2_config=successor,
                    )
                    selected_map = alignment.selected
                    geometry_candidates = [
                        {"model": item.model, "accepted": item.accepted,
                         "failure_reason": item.failure_reason, "metrics": dict(item.metrics),
                         "audit": dict(item.audit)} for item in alignment.candidates
                    ]
                    for key, reason in (
                        ("map_finite", "geometry_map_nonfinite"),
                        ("map_in_source_bounds", "geometry_source_out_of_bounds"),
                        ("positive_jacobian", "geometry_nonpositive_jacobian"),
                    ):
                        if selected_map.audit.get(key) is not True:
                            failures.append(reason)
                    if float(selected_map.audit.get("support_retention", 0.0)) < 0.95:
                        failures.append("geometry_support_retention_insufficient")
                    if float(selected_map.audit.get("maximum_map_displacement_px", math.inf)) > 8.0:
                        failures.append("geometry_displacement_exceeded")
                    final_maps = _map_crop(
                        schedule, calibration, pair_index + 1, x0, x1,
                        vertical.global_offsets_px[pair_index + 1], selected_map,
                    )
                    final_right, final_valid = _sample_crop(raw(frame_ids[1]), final_maps)
                    before_preview = _compose_pair_preview(left_image, right_image, seam_local)
                    after_preview = _compose_pair_preview(left_image, final_right, seam_local)
                    before_features = prepare_seam_structure(before_preview)
                    after_features = prepare_seam_structure(after_preview)
                    if before_features is None or after_features is None:
                        raise ValueError("pair preview feature construction failed")
                    before_metrics = seam_structure_metrics(before_features, seam_local)
                    after_metrics = seam_structure_metrics(after_features, seam_local)
                    before_horizontal = long_horizontal_structure_metrics(before_features, seam_local)
                    after_horizontal = long_horizontal_structure_metrics(after_features, seam_local)
                    # Only catastrophic horizontal damage is a runtime gate.
                    from .video_s13_hard_audit import long_horizontal_structure_catastrophe_guard

                    horizontal_safe, horizontal_reason, horizontal_audit = (
                        long_horizontal_structure_catastrophe_guard(before_horizontal, after_horizontal)
                    )
                    if not horizontal_safe:
                        failures.append(str(horizontal_reason or "horizontal_structure_catastrophe"))
                    if successor.enabled and not failures:
                        left_edge_features = prepare_seam_structure(left_image)
                        final_edge_features = prepare_seam_structure(final_right)
                        if left_edge_features is None or final_edge_features is None:
                            raise ValueError("pair edge feature construction failed")
                        edge_registration = pair_edge_registration_metrics(
                            left_edge_features, final_edge_features, left_valid, final_valid, seam_local,
                            config=successor,
                        )
                        lk_p95_value = selected_map.metrics.get("residual_p95_px")
                        lk_p95 = (
                            float(lk_p95_value)
                            if isinstance(lk_p95_value, (int, float)) else None
                        )
                        visual_suspect = edge_registration_visual_suspect(
                            edge_registration, seam_local_lk_p95_px=lk_p95,
                            config=successor,
                        )
                        edge_registration = {
                            **dict(edge_registration), "visual_suspect": visual_suspect,
                        }

                        if visual_suspect:
                            selection_continued_for_structure = True
                            if hard_safe_baseline is None:
                                hard_safe_baseline = (
                                    candidate, alignment, before_metrics, after_metrics,
                                    before_horizontal, after_horizontal, edge_registration,
                                    seam_rank, 0,
                                )
                            clean_alternatives: list[tuple[float, int, int, object, object]] = []
                            simplicity = {
                                "C0_identity": 0, "C1_accepted_vertical": 1,
                                "C2_subpixel_translation": 2,
                                "C3_translation_tiny_rotation": 3,
                                "C4_bounded_light_affine": 4,
                            }
                            for alternate_index, alternate in enumerate(alignment.candidates):
                                if not alternate.accepted or alternate_index == alignment.selected_candidate_index:
                                    continue
                                if (
                                    alternate.audit.get("map_finite") is not True
                                    or alternate.audit.get("map_in_source_bounds") is not True
                                    or alternate.audit.get("positive_jacobian") is not True
                                    or float(alternate.audit.get("support_retention", 0.0)) < 0.95
                                    or float(alternate.audit.get("maximum_map_displacement_px", math.inf)) > 8.0
                                ):
                                    continue
                                alternate_maps = _map_crop(
                                    schedule, calibration, pair_index + 1, x0, x1,
                                    vertical.global_offsets_px[pair_index + 1], alternate,
                                )
                                alternate_right, alternate_valid = _sample_crop(
                                    raw(frame_ids[1]), alternate_maps
                                )
                                alternate_preview = _compose_pair_preview(
                                    left_image, alternate_right, seam_local
                                )
                                alternate_preview_features = prepare_seam_structure(alternate_preview)
                                alternate_edge_features = prepare_seam_structure(alternate_right)
                                if alternate_preview_features is None or alternate_edge_features is None:
                                    raise ValueError("alternate pair feature construction failed")
                                alternate_after = seam_structure_metrics(
                                    alternate_preview_features, seam_local
                                )
                                alternate_horizontal = long_horizontal_structure_metrics(
                                    alternate_preview_features, seam_local
                                )
                                alternate_safe, _alternate_reason, _alternate_audit = (
                                    long_horizontal_structure_catastrophe_guard(
                                        before_horizontal, alternate_horizontal
                                    )
                                )
                                if not alternate_safe:
                                    continue
                                alternate_edge = pair_edge_registration_metrics(
                                    left_edge_features, alternate_edge_features, left_valid,
                                    alternate_valid, seam_local, config=successor,
                                )
                                alt_lk_value = alternate.metrics.get("residual_p95_px")
                                alt_lk = (
                                    float(alt_lk_value)
                                    if isinstance(alt_lk_value, (int, float)) else None
                                )
                                if edge_registration_visual_suspect(
                                    alternate_edge, seam_local_lk_p95_px=alt_lk,
                                    config=successor,
                                ):
                                    continue
                                edge_p95_value = alternate_edge.get("p95_supported_abs_lag_px")
                                edge_p95 = (
                                    float(edge_p95_value)
                                    if isinstance(edge_p95_value, (int, float)) else math.inf
                                )
                                clean_alternatives.append((
                                    edge_p95, simplicity.get(alternate.model, 99),
                                    alternate_index, alternate_after,
                                    (alternate_horizontal, alternate_edge),
                                ))
                            if clean_alternatives:
                                minimum_edge = min(item[0] for item in clean_alternatives)
                                equivalent = [
                                    item for item in clean_alternatives
                                    if item[0] <= minimum_edge + successor.equivalent_edge_p95_tolerance_px
                                ]
                                chosen = min(equivalent, key=lambda item: item[1])
                                geometry_rank = int(chosen[2])
                                alignment = replace(
                                    alignment,
                                    selected_model=alignment.candidates[geometry_rank].model,
                                    selected_candidate_index=geometry_rank,
                                )
                                selected_map = alignment.selected
                                after_metrics = chosen[3]  # type: ignore[assignment]
                                alternate_details = chosen[4]
                                after_horizontal = alternate_details[0]  # type: ignore[index,assignment]
                                edge_registration = {
                                    **dict(alternate_details[1]),  # type: ignore[arg-type,index]
                                    "visual_suspect": False,
                                }
                                visual_suspect = False
                    map_delta_sha = _sha_array(selected_map.target_delta_u, selected_map.target_delta_v)
                    corridor_bounds = [
                        int(alignment.application_band.left_x_by_row.min()),
                        int(alignment.application_band.right_x_by_row.max()),
                    ]
                else:
                    horizontal_audit = {"passed": False, "not_evaluated": True}
                hard_safe = not failures
                passed = hard_safe
                if successor.enabled and hard_safe and visual_suspect:
                    passed = False
                evaluation = {
                    "candidate_id": candidate.candidate_id,
                    "model_code": candidate.model_code,
                    "model_name": candidate.model_name,
                    "generation_status": "generated",
                    "evaluation_status": (
                        "selected" if passed and selected_candidate is None
                        else "hard_safe_not_selected" if passed
                        else "visual_suspect_continued" if hard_safe and visual_suspect
                        else "rejected_hard_gate"
                    ),
                    "local_objective": candidate.local_objective,
                    "seam_sha256": _sha_array(seam_global),
                    "corridor_bounds": corridor_bounds,
                    "geometry_input_grid_sha256": _sha_array(p0_u, p0_v, p0_valid),
                    "geometry_model": None if alignment is None else alignment.selected.model,
                    "geometry_candidates": geometry_candidates,
                    "map_delta_sha256": map_delta_sha,
                    "horizontal_hard_audit": horizontal_audit,
                    "edge_registration": dict(edge_registration),
                    "visual_suspect": bool(visual_suspect),
                    "hard_gate_passed": hard_safe,
                    "hard_gate_failures": failures,
                    "diagnostic_metrics": {"before": before_metrics, "after": after_metrics},
                    "generation_audit": dict(candidate.generation_audit),
                }
                evaluations.append(evaluation)
                if passed and selected_candidate is None:
                    selected_candidate = candidate
                    selected_alignment = alignment
                    selected_before, selected_after = before_metrics, after_metrics
                    selected_before_horizontal, selected_after_horizontal = before_horizontal, after_horizontal
                    selected_edge_registration = edge_registration
                    selected_seam_rank = seam_rank
                    selected_geometry_rank = geometry_rank
            if selected_candidate is None and hard_safe_baseline is not None:
                (
                    selected_candidate, selected_alignment, selected_before, selected_after,
                    selected_before_horizontal, selected_after_horizontal,
                    selected_edge_registration, selected_seam_rank, selected_geometry_rank,
                ) = hard_safe_baseline
                unresolved_oblique_structure = True
                for row in evaluations:
                    if row.get("candidate_id") == selected_candidate.candidate_id:
                        row["evaluation_status"] = (
                            "selected_unresolved_hard_safe_baseline"
                        )
                        break
            if selected_candidate is None or selected_alignment is None:
                fallback = _fallback_pair(
                    schedule, pair_index, parent_stage_sha256, frame_ids,
                    "all_generated_candidates_failed_declared_hard_gates",
                    parent_result_sha256, p0_ancestor_completion_sha256,
                    correspondence_filter=correspondence_audit,
                    m51_r2_config=successor,
                )
                transaction = {
                    **dict(fallback.transaction),
                    "candidate_generation": [dict(item) for item in generation_statuses],
                    "candidate_evaluations": evaluations,
                }
                transaction = _finalize_v5_transaction(transaction)
                pairs.append(S13M5Pair(transaction, fallback.seam_x_by_row, None))
                continue
            selected_seam = np.asarray(selected_candidate.seam_x_by_row, dtype=np.int32) + x0
            selected_map = selected_alignment.selected
            core: dict[str, object] = {
                "schema": (
                    "gemini305-video-s13-m5-pair-transaction/v3"
                    if successor.enabled else "gemini305-video-s13-m5-pair-transaction/v2"
                ),
                "transaction_id": f"m5-pair-{pair_index:04d}",
                "parent_stage": "P1",
                "parent_stage_sha256": parent_stage_sha256,
                "parent_completion_sha256": parent_stage_sha256,
                "parent_result_sha256": parent_result_sha256,
                "p0_ancestor_completion_sha256": p0_ancestor_completion_sha256,
                "pair_frame_ids": list(frame_ids),
                "candidate_generation": [dict(item) for item in generation_statuses],
                "candidate_evaluations": evaluations,
                "selection_policy": (
                    "minimum_local_objective_with_supported_edge_residual_veto"
                    if successor.enabled
                    else "minimum_local_objective_among_hard_safe_candidates"
                ),
                "selected_candidate_id": selected_candidate.candidate_id,
                "selected_seam_model": selected_candidate.model_name,
                "selected_geometry_model": selected_map.model,
                "pair_hard_audit_passed": True,
                "fallback_used": selected_candidate.model_code == "S0" and selected_map.model == "C0_identity",
                "fallback_reason": (
                    "complex_candidates_failed_or_ranked_later" if selected_candidate.model_code == "S0" else None
                ),
                "motion_hypothesis_id": -1,
                "support_mask_sha256": _sha_array(left_valid & right_valid),
                "geometry_model": selected_map.model,
                "seam_model": selected_candidate.model_name,
                "map_delta_sha256": _sha_array(selected_map.target_delta_u, selected_map.target_delta_v),
                "decision": "applied",
                "before_metrics": dict(selected_before or {}),
                "candidate_after_metrics": dict(selected_after or {}),
                "after_metrics": dict(selected_after or {}),
                "before_horizontal_continuity": dict(selected_before_horizontal or {}),
                "candidate_after_horizontal_continuity": dict(selected_after_horizontal or {}),
                "after_horizontal_continuity": dict(selected_after_horizontal or {}),
                "rollback_reason": None,
                "comparison_coordinate_policy": "same_owner_geometry_plus_symmetric_base_candidate_seam",
                "geometry_comparison_seam_sha256": _sha_array(selected_seam),
                "selected_seam_sha256": _sha_array(selected_seam),
                "result_seam_sha256": _sha_array(selected_seam),
                "held_out_seam_selection_audits": [
                    {
                        "candidate_id": item.get("candidate_id"),
                        "evaluation_status": item.get("evaluation_status"),
                        "selected_seam_sha256": item.get("seam_sha256"),
                        "hard_gate_passed": item.get("hard_gate_passed"),
                        "hard_gate_failures": item.get("hard_gate_failures", []),
                    }
                    for item in evaluations
                    if item.get("evaluation_status") != "skipped_after_higher_rank_safe_candidate"
                ],
                "final_corridor_geometry_reestimated_from_immutable_p0": True,
                "geometry_candidates": evaluations,
                "seam_audit": dict(seam_result.audit),
                "seam_cost_components": dict(
                    seam_result.candidate_cost_components[selected_candidate.model_name]
                ),
                "seam_fallback_chain": list(seam_result.fallback_chain),
            }
            if instrumentation_requested:
                core["correspondence_filter"] = dict(correspondence_result.audit)
            if successor.enabled:
                transaction_edge = dict(selected_edge_registration)
                transaction_edge.setdefault(
                    "ambiguous",
                    bool(transaction_edge.get("multiple_layer_or_ambiguous", False)),
                )
                core.update({
                    "seam_local_evidence": dict(
                        selected_alignment.seam_local_evidence or {}
                    ),
                    "edge_registration": transaction_edge,
                    "selection_continued_for_structure": bool(
                        selection_continued_for_structure
                    ),
                    "selected_seam_rank": int(selected_seam_rank),
                    "selected_geometry_rank": int(selected_geometry_rank),
                    "seam_fallback_used": bool(selected_seam_rank > 0),
                    "geometry_fallback_used": bool(
                        selected_map.model in {"C0_identity", "C1_accepted_vertical"}
                    ),
                    "topology_repair_used": False,
                    "unresolved_oblique_structure": bool(unresolved_oblique_structure),
                    "unexpected_exception_fallback": False,
                    "micro_rescue": {
                        "enabled": False, "attempted": False, "accepted": False,
                    },
                })
            if successor.enabled:
                core = _finalize_v5_transaction(core)
            else:
                core["result_stage_sha256"] = _sha_json(core)
            pairs.append(S13M5Pair(core, selected_seam, selected_alignment))
        except Exception as exc:
            if successor.enabled:
                raise
            fallback = _fallback_pair(
                schedule, pair_index, parent_stage_sha256, frame_ids,
                f"{type(exc).__name__}:{exc}",
                parent_result_sha256,
                p0_ancestor_completion_sha256,
                correspondence_filter=(
                    correspondence_audit if instrumentation_requested else None
                ),
                m51_r2_config=m51_r2_config,
            )
            pairs.append(fallback)
    # One deterministic topology repair pass.  A local candidate may stay in
    # its own ordered search domain yet meet its neighbour on a row; only the
    # involved pairs fall back to their immutable P1 midpoint grids.
    if len(pairs) > 1:
        seams = np.stack([np.asarray(pair.seam_x_by_row, dtype=np.int32) for pair in pairs])
        crossing = np.flatnonzero(np.any(np.diff(seams, axis=0) < 1, axis=1))
        repair_indices = sorted({int(index) for value in crossing for index in (value, value + 1)})
        for pair_index in repair_indices:
            left = schedule.assignments[pair_index]
            right = schedule.assignments[pair_index + 1]
            original = pairs[pair_index]
            fallback = _fallback_pair(
                schedule, pair_index, parent_stage_sha256,
                (left.frame_id, right.frame_id),
                "seam_family_crossing_deterministic_midpoint_identity_repair",
                parent_result_sha256,
                p0_ancestor_completion_sha256,
                correspondence_filter=(
                    original.transaction.get("correspondence_filter")
                    if instrumentation_requested else None
                ),
                m51_r2_config=m51_r2_config,
                topology_repair_used=True,
            )
            transaction = {
                **dict(fallback.transaction),
                "candidate_generation": original.transaction.get("candidate_generation", []),
                "candidate_evaluations": original.transaction.get("candidate_evaluations", []),
            }
            if instrumentation_requested and "correspondence_filter" in original.transaction:
                transaction["correspondence_filter"] = original.transaction[
                    "correspondence_filter"
                ]
            if successor.enabled:
                transaction = _finalize_v5_transaction(transaction)
            else:
                transaction["result_stage_sha256"] = _sha_json(transaction)
            pairs[pair_index] = S13M5Pair(transaction, fallback.seam_x_by_row, None)
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


def build_s13_p2_replay(
    schedule: S012Schedule,
    calibration: CameraIntrinsics,
    vertical: S13VerticalSolution,
    pairs: Sequence[S13M5Pair],
    *,
    corridor_half_width_px: int = 8,
) -> tuple[S13P2ReplayPair, ...]:
    """Serialize selected P2 sampling without re-estimating geometry or seams."""

    if not 4 <= int(corridor_half_width_px) <= 16:
        raise ValueError("S1.3 P2 replay half-width must be in [4, 16]")
    seams = _seams_array(schedule, pairs, final=True)
    replay: list[S13P2ReplayPair] = []
    for pair_index, pair in enumerate(pairs):
        seam = seams[pair_index]
        x0 = max(0, int(seam.min()) - int(corridor_half_width_px))
        x1 = min(schedule.canvas_width, int(seam.max()) + int(corridor_half_width_px) + 1)
        left_candidate = (
            pairs[pair_index - 1].alignment.selected
            if pair_index > 0 and pairs[pair_index - 1].alignment is not None
            else None
        )
        right_candidate = pair.alignment.selected if pair.alignment is not None else None
        left_maps = _map_crop(
            schedule, calibration, pair_index, x0, x1,
            vertical.global_offsets_px[pair_index], left_candidate,
        )
        right_maps = _map_crop(
            schedule, calibration, pair_index + 1, x0, x1,
            vertical.global_offsets_px[pair_index + 1], right_candidate,
        )
        canvas_x = np.arange(x0, x1, dtype=np.int32)[None, :]
        owner_right = canvas_x >= seam[:, None]
        replay.append(S13P2ReplayPair(
            pair_index=pair_index,
            left_source_index=pair_index,
            right_source_index=pair_index + 1,
            left_frame_id=int(schedule.assignments[pair_index].frame_id),
            right_frame_id=int(schedule.assignments[pair_index + 1].frame_id),
            corridor_x0=x0,
            corridor_x1=x1,
            seam_x_by_row=seam.copy(),
            left_source_u=np.asarray(left_maps[0], dtype=np.float32),
            left_source_v=np.asarray(left_maps[1], dtype=np.float32),
            left_valid=np.asarray(left_maps[2], dtype=bool),
            right_source_u=np.asarray(right_maps[0], dtype=np.float32),
            right_source_v=np.asarray(right_maps[1], dtype=np.float32),
            right_valid=np.asarray(right_maps[2], dtype=bool),
            primary_owner_right_mask=owner_right,
            geometry_transaction_numeric_id=pair_index,
            seam_transaction_numeric_id=pair_index,
            parent_pair_transaction_sha256="",
        ))
    return tuple(replay)


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
    expected_support = np.zeros((height, width), dtype=bool)
    for source_index, assignment in enumerate(schedule.assignments):
        candidate = None
        if source_index > 0 and pairs[source_index - 1].alignment is not None:
            candidate = pairs[source_index - 1].alignment.selected
        expected_maps = _map_crop(
            schedule, calibration, source_index, 0, width,
            vertical.global_offsets_px[source_index], candidate,
        )
        expected_support |= expected_maps[2]
        mask = owner_index == source_index
        if not np.any(mask):
            continue
        columns = np.flatnonzero(np.any(mask, axis=0))
        x0, x1 = int(columns[0]), int(columns[-1]) + 1
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
    return S13P2Result(
        output, valid_full, pixel, len(decoded), tuple(decoded), expected_support
    )


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
    parent_result_sha256: str | None = None,
    p0_ancestor_completion_sha256: str | None = None,
    selected_hypothesis_ids: tuple[int, ...] | None = None,
    placement_methods: tuple[str, ...] | None = None,
    m51_r2_config: S13M51R2Config | None = None,
) -> S13M5Result:
    started = time.perf_counter()
    tick = time.perf_counter()
    pairs = estimate_s13_m5_transactions(
        schedule, calibration, image_loader, vertical,
        parent_stage_sha256=parent_stage_sha256,
        parent_result_sha256=parent_result_sha256,
        p0_ancestor_completion_sha256=p0_ancestor_completion_sha256,
        m51_r2_config=m51_r2_config,
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
    replay_pairs = build_s13_p2_replay(schedule, calibration, vertical, pairs)
    seam_seconds = time.perf_counter() - tick
    geometry_features = prepare_seam_structure(geometry.image)
    final_features = prepare_seam_structure(final.image)
    if geometry_features is None or final_features is None:
        raise ValueError("S1.3 P2 full-canvas feature construction failed")
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
            before_features=geometry_features,
            after_features=final_features,
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

    legacy_selected = bool(
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
        "eligible": legacy_selected,
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
    from .video_s13_hard_audit import audit_s13_p2_stage

    seams = _seams_array(schedule, pairs, final=True)
    hard_audit = audit_s13_p2_stage(
        valid_mask=final.valid_mask,
        pixel_provenance=final.pixel_provenance,
        seams_x_by_row=seams,
        assignment_frame_ids=tuple(item.frame_id for item in schedule.assignments),
        source_sizes=tuple(
            (int(calibration.width), int(calibration.height)) for _item in schedule.assignments
        ),
        pair_transaction_count=len(pairs),
        expected_support_mask=final.expected_support_mask,
    )
    hard_audit = {
        **dict(hard_audit),
        "parent_verification_deferred_to_stage_sealer": True,
        "pair_fallback_count": sum(
            bool(pair.transaction.get("fallback_used"))
            or pair.transaction.get("decision") == "rolled_back"
            for pair in pairs
        ),
        "horizontal_catastrophe_summary": {
            "rejected_candidate_count": sum(
                str(failure).startswith("horizontal_structure_")
                for pair in pairs
                for evaluation in pair.transaction.get("candidate_evaluations", [])
                if isinstance(evaluation, Mapping)
                for failure in evaluation.get("hard_gate_failures", [])
            ),
        },
    }
    diagnostic_quality = {
        "schema": "gemini305-video-s13-m5-diagnostic-quality/v2",
        "diagnostic_only": True,
        "runtime_authority": False,
        "legacy_minimum_improvement_fraction": 0.005,
        "legacy_policy_result": legacy_selected,
        "before_mean_score": before_mean,
        "after_mean_score": after_mean,
        "relative_change": (
            None if before_mean is None or after_mean is None or abs(before_mean) < 1e-12
            else (after_mean - before_mean) / abs(before_mean)
        ),
        "legacy_selection_audit": selection_audit,
    }
    correspondence_rows = [
        pair.transaction.get("correspondence_filter", {}) for pair in pairs
    ]
    evaluated_rows = [
        evaluation
        for pair in pairs
        for evaluation in pair.transaction.get("candidate_evaluations", [])
        if isinstance(evaluation, Mapping)
        and evaluation.get("evaluation_status") != "skipped_after_higher_rank_safe_candidate"
    ]
    return S13M5Result(
        pairs=pairs,
        replay_pairs=replay_pairs,
        geometry_result=geometry,
        final_result=final,
        seam_overlay=_overlay_seams(final.image, pairs),
        hard_audit_passed=hard_audit.get("passed") is True,
        hard_audit=hard_audit,
        diagnostic_quality=diagnostic_quality,
        selected_as_best=legacy_selected,
        before_mean_score=before_mean,
        after_mean_score=after_mean,
        selection_audit=selection_audit,
        performance={
            "geometry": geometry_seconds,
            "seam_and_p2_render": seam_seconds,
            "total_m5": time.perf_counter() - started,
            "gftt_call_count": sum(int(row.get("gftt_call_count", 0)) for row in correspondence_rows if isinstance(row, Mapping)),
            "forward_pyr_lk_call_count": sum(int(row.get("forward_pyr_lk_call_count", 0)) for row in correspondence_rows if isinstance(row, Mapping)),
            "backward_pyr_lk_call_count": 0,
            "full_canvas_feature_build_count": 2,
            "pair_feature_build_count": sum(2 for _row in evaluated_rows),
            "risk_pair_count": sum(bool(pair.transaction.get("selection_continued_for_structure")) for pair in pairs),
            "extra_seam_candidate_evaluation_count": sum(max(0, sum(1 for row in pair.transaction.get("candidate_evaluations", []) if isinstance(row, Mapping) and row.get("evaluation_status") != "skipped_after_higher_rank_safe_candidate") - 1) for pair in pairs),
            "extra_geometry_roi_sample_count": sum(max(0, int(pair.transaction.get("selected_geometry_rank", 0))) for pair in pairs),
            "micro_rescue_attempt_count": 0,
            "p2_full_resolution_render_count": 2,
            "extra_full_resolution_render_count": 0,
            "formal_raw_rgb_remap_invocations": geometry.remap_invocations + final.remap_invocations,
            "depth_call_count": 0,
            "dis_call_count": 0,
            "open3d_call_count": 0,
            "orbslam3_call_count_in_m5": 0,
            "m4_geometry_reestimation_count": 0,
            "gain_enumeration_count": 0,
        },
    )


__all__ = [
    "S13M5Pair", "S13M5Result", "S13P2Result", "S13PairCorrespondences",
    "build_s13_p2_replay",
    "estimate_s13_m5_transactions", "render_s13_p2_from_raw", "run_s13_m5",
]
