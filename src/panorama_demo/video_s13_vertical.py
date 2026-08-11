"""M4-only global scalar dy and pair-local row residual for S1.3."""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import asdict, dataclass, replace
from typing import Callable, Mapping

import cv2
import numpy as np

from .calibrated_remap import accelerated_remap, camera_matrix, undistortion_maps
from .session import CameraIntrinsics
from .video_s12_schedule import S012Schedule, validate_s012_schedule


@dataclass(frozen=True)
class S13VerticalPair:
    pair_index: int
    left_frame_id: int
    right_frame_id: int
    boundary_x: int
    shoulder_left_x: int
    shoulder_right_x: int
    application_left_x: int
    application_right_x: int
    measured_correction_dy_px: float
    phase_response: float
    supported_row_count: int
    local_residual_min_px: float
    local_residual_max_px: float
    status: str
    failure_reason: str | None


@dataclass(frozen=True)
class S13VerticalSolution:
    global_offsets_px: tuple[float, ...]
    local_row_residuals: tuple[np.ndarray, ...]
    pairs: tuple[S13VerticalPair, ...]
    selected_gain: float
    gain_scores: Mapping[str, float]
    shoulder_width_px: int
    audit: Mapping[str, object]
    gain_global_offsets_px: Mapping[str, tuple[float, ...]]
    gain_local_row_residuals: Mapping[str, tuple[np.ndarray, ...]]


@dataclass(frozen=True)
class S13P1Result:
    image: np.ndarray
    valid_mask: np.ndarray
    pixel_provenance: dict[str, np.ndarray]
    remap_invocations: int
    decoded_frame_ids: tuple[int, ...]


def _target_map(
    calibration: CameraIntrinsics,
    center_x: float,
    left_x: int,
    right_x: int,
    displacement_y: np.ndarray,
    inverse_maps: tuple[np.ndarray, np.ndarray] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height = int(calibration.height)
    x = float(calibration.cx) + np.arange(left_x, right_x, dtype=np.float64) - float(center_x)
    y = np.arange(height, dtype=np.float64)[:, None] - displacement_y.astype(np.float64)
    grid_x = np.broadcast_to(x[None, :], (height, len(x)))
    grid_y = np.broadcast_to(y, grid_x.shape)
    target_valid = (
        (grid_x >= 0.0) & (grid_x <= int(calibration.width) - 1)
        & (grid_y >= 0.0) & (grid_y <= int(calibration.height) - 1)
    )
    if inverse_maps is None:
        source_u, source_v = grid_x.astype(np.float32), grid_y.astype(np.float32)
    else:
        source_u = cv2.remap(inverse_maps[0], grid_x.astype(np.float32), grid_y.astype(np.float32),
                             cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=-1)
        source_v = cv2.remap(inverse_maps[1], grid_x.astype(np.float32), grid_y.astype(np.float32),
                             cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=-1)
    valid = (
        target_valid & np.isfinite(source_u) & np.isfinite(source_v)
        & (source_u >= 0.0) & (source_u <= int(calibration.width) - 1)
        & (source_v >= 0.0) & (source_v <= int(calibration.height) - 1)
    )
    return source_u, source_v, valid


def _sample_shoulder(
    image: np.ndarray,
    calibration: CameraIntrinsics,
    center_x: float,
    left_x: int,
    right_x: int,
) -> tuple[np.ndarray, np.ndarray]:
    displacement = np.zeros((int(calibration.height), 1), dtype=np.float32)
    u, v, valid = _target_map(
        calibration, center_x, left_x, right_x, displacement, undistortion_maps(calibration)
    )
    sampled = accelerated_remap(image, u, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    gray = cv2.cvtColor(sampled, cv2.COLOR_BGR2GRAY)
    return gray, valid


def _phase_vertical(left: np.ndarray, right: np.ndarray, valid: np.ndarray) -> tuple[float, float]:
    if left.shape != right.shape or min(left.shape) < 8 or int(valid.sum()) < 256:
        return 0.0, 0.0
    a = left.astype(np.float32)
    b = right.astype(np.float32)
    a[~valid] = 0.0
    b[~valid] = 0.0
    if float(a.std()) < 1e-6 or float(b.std()) < 1e-6:
        return 0.0, 0.0
    window = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F)
    shift, response = cv2.phaseCorrelate(a, b, window)
    if not np.isfinite(shift).all() or not math.isfinite(response):
        return 0.0, 0.0
    # shift_y describes where right content appears relative to left. The
    # target-space correction applied to the right source is the opposite.
    return float(np.clip(-shift[1], -12.0, 12.0)), float(response)


def _solve_offsets(observations: np.ndarray, weights: np.ndarray, gain: float) -> np.ndarray:
    count = len(observations) + 1
    if count <= 1:
        return np.zeros(count, dtype=np.float64)
    rows: list[np.ndarray] = []
    values: list[float] = []
    base_weights: list[float] = []
    for index, observation in enumerate(observations):
        row = np.zeros(count, dtype=np.float64)
        row[index], row[index + 1] = -1.0, 1.0
        rows.append(row)
        values.append(float(observation) * gain)
        base_weights.append(float(weights[index]))
    for index in range(1, count - 1):
        row = np.zeros(count, dtype=np.float64)
        row[index - 1:index + 2] = (0.15, -0.30, 0.15)
        rows.append(row)
        values.append(0.0)
        base_weights.append(1.0)
    # Weak identity priors prevent a long chain of tiny biased measurements
    # from turning into an unbounded vertical ramp. They are deliberately
    # weaker than a supported pair edge and preserve the mandatory gain=0
    # rollback candidate.
    for index in range(count):
        row = np.zeros(count, dtype=np.float64)
        row[index] = 1.0
        rows.append(row)
        values.append(0.0)
        base_weights.append(0.08)
    gauge = np.zeros(count, dtype=np.float64)
    gauge[0] = 1.0
    rows.append(gauge)
    values.append(0.0)
    base_weights.append(100.0)
    matrix = np.asarray(rows)
    target = np.asarray(values)
    robust = np.asarray(base_weights)
    solution = np.zeros(count, dtype=np.float64)
    for _ in range(6):
        weighted = np.sqrt(np.maximum(1e-6, robust))
        solution = np.linalg.lstsq(matrix * weighted[:, None], target * weighted, rcond=None)[0]
        residual = matrix @ solution - target
        scale = max(0.25, float(np.median(np.abs(residual))) * 1.4826)
        huber = np.minimum(1.0, 1.5 * scale / np.maximum(1e-9, np.abs(residual)))
        robust = np.asarray(base_weights) * huber
    return solution - float(np.median(solution))


def _local_rows(
    left: np.ndarray, right: np.ndarray, valid: np.ndarray, global_relative: float
) -> tuple[np.ndarray, int]:
    height = left.shape[0]
    residual = np.zeros(height, dtype=np.float32)
    if int(valid.sum()) < 256:
        return residual, 0
    flow = cv2.calcOpticalFlowFarneback(
        left, right, None, 0.5, 3, 19, 3, 5, 1.1, cv2.OPTFLOW_FARNEBACK_GAUSSIAN
    )
    texture = np.abs(cv2.Sobel(left, cv2.CV_32F, 1, 0, ksize=3))
    supported = np.zeros(height, dtype=bool)
    for row in range(height):
        mask = valid[row] & np.isfinite(flow[row, :, 1]) & (texture[row] > 3.0)
        if int(mask.sum()) < 12:
            continue
        row_correction = -float(np.median(flow[row, mask, 1])) - global_relative
        residual[row] = float(np.clip(row_correction, -2.0, 2.0))
        supported[row] = True
    if np.any(supported):
        smooth = cv2.GaussianBlur(residual[:, None], (1, 9), 0).reshape(-1)
        residual[supported] = smooth[supported]
        residual[~supported] = 0.0
    return residual, int(supported.sum())


def estimate_s13_vertical(
    schedule: S012Schedule,
    calibration: CameraIntrinsics,
    image_loader: Callable[[int], np.ndarray],
    *,
    shoulder_width_px: int = 96,
    gain_candidates: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0),
) -> S13VerticalSolution:
    camera_matrix(calibration)
    validate_s012_schedule(schedule)
    shoulder = int(np.clip(shoulder_width_px, 64, 128))
    cache: OrderedDict[int, np.ndarray] = OrderedDict()

    def load(frame_id: int) -> np.ndarray:
        if frame_id not in cache:
            cache[frame_id] = np.asarray(image_loader(frame_id))
            while len(cache) > 3:
                cache.popitem(last=False)
        cache.move_to_end(frame_id)
        return cache[frame_id]

    measurements: list[float] = []
    responses: list[float] = []
    sampled_pairs: list[tuple[np.ndarray, np.ndarray, np.ndarray, int, int]] = []
    half = shoulder // 2
    for pair_index, (left_assignment, right_assignment) in enumerate(
        zip(schedule.assignments[:-1], schedule.assignments[1:])
    ):
        boundary = schedule.boundaries[pair_index + 1]
        left_x, right_x = max(0, boundary - half), min(schedule.canvas_width, boundary + half)
        left_gray, left_valid = _sample_shoulder(
            load(left_assignment.frame_id), calibration, left_assignment.center_x, left_x, right_x
        )
        right_gray, right_valid = _sample_shoulder(
            load(right_assignment.frame_id), calibration, right_assignment.center_x, left_x, right_x
        )
        common = left_valid & right_valid
        correction, response = _phase_vertical(left_gray, right_gray, common)
        if response < 0.05:
            correction = 0.0
        measurements.append(correction)
        responses.append(response)
        sampled_pairs.append((left_gray, right_gray, common, left_x, right_x))
    observation = np.asarray(measurements, dtype=np.float64)
    weight = np.clip(np.asarray(responses, dtype=np.float64), 0.0, 1.0)
    gain_scores: dict[str, float] = {}
    candidates: dict[float, np.ndarray] = {}
    for gain in gain_candidates:
        offsets = _solve_offsets(observation, np.maximum(weight, 0.05), gain)
        predicted = np.diff(offsets)
        score = float(np.average(np.abs(observation - predicted), weights=np.maximum(weight, 0.05)))
        score += 0.02 * float(np.mean(np.abs(np.diff(offsets, n=2)))) if len(offsets) > 2 else 0.0
        score += 0.05 * float(np.mean(np.abs(offsets)))
        gain_scores[str(gain)] = score
        candidates[gain] = offsets
    selected_gain = min(gain_candidates, key=lambda gain: (gain_scores[str(gain)], gain))
    candidate_rows: dict[str, tuple[np.ndarray, ...]] = {}
    candidate_reports: dict[str, tuple[S13VerticalPair, ...]] = {}
    for gain in gain_candidates:
        gain_offsets = candidates[gain]
        local_rows: list[np.ndarray] = []
        reports: list[S13VerticalPair] = []
        for pair_index, ((left_gray, right_gray, common, left_x, right_x), response) in enumerate(
            zip(sampled_pairs, responses, strict=True)
        ):
            left_assignment = schedule.assignments[pair_index]
            right_assignment = schedule.assignments[pair_index + 1]
            relative = float(gain_offsets[pair_index + 1] - gain_offsets[pair_index])
            rows, supported = _local_rows(left_gray, right_gray, common, relative) if response >= 0.05 else (
                np.zeros(schedule.canvas_height, dtype=np.float32), 0
            )
            boundary = schedule.boundaries[pair_index + 1]
            application_width = min(16, max(0, right_assignment.right_x - boundary))
            application_right = boundary + application_width
            # This is only an evidence-bearing candidate.  M5's rendered
            # relative audit, not support count, decides whether it is used.
            candidate_available = supported > 0 and application_width > 0
            if not candidate_available:
                rows[...] = 0.0
            local_rows.append(rows)
            reports.append(S13VerticalPair(
                pair_index=pair_index, left_frame_id=left_assignment.frame_id,
                right_frame_id=right_assignment.frame_id, boundary_x=boundary,
                shoulder_left_x=left_x, shoulder_right_x=right_x,
                application_left_x=boundary, application_right_x=application_right,
                measured_correction_dy_px=float(measurements[pair_index]), phase_response=float(response),
                supported_row_count=supported, local_residual_min_px=float(rows.min()),
                local_residual_max_px=float(rows.max()), status="candidate" if candidate_available else "local_zero",
                failure_reason=None if candidate_available else "insufficient_supported_rows_or_band",
            ))
        candidate_rows[str(gain)] = tuple(local_rows)
        candidate_reports[str(gain)] = tuple(reports)
    offsets = candidates[selected_gain]
    local_rows = list(candidate_rows[str(selected_gain)])
    reports = list(candidate_reports[str(selected_gain)])
    return S13VerticalSolution(
        global_offsets_px=tuple(float(value) for value in offsets),
        local_row_residuals=tuple(local_rows), pairs=tuple(reports), selected_gain=float(selected_gain),
        gain_scores=gain_scores, shoulder_width_px=shoulder,
        audit={
            "model": "global_scalar_dy_plus_pair_local_row_residual",
            "global_solver": "huber_irls_with_second_difference_smoothing",
            "measurement_source": "immutable_P0_target_grids",
            "gain_candidates": list(gain_candidates),
            "selected_gain": float(selected_gain),
            "missing_rows_are_zero": True,
            "maximum_local_residual_px": 2.0,
            "translation_rotation_affine_seam_photometric_blend_depth_mesh_enabled": False,
            "pairs": [asdict(pair) for pair in reports],
        },
        gain_global_offsets_px={
            str(gain): tuple(float(value) for value in candidates[gain]) for gain in gain_candidates
        },
        gain_local_row_residuals=candidate_rows,
    )


def vertical_candidate_solution(
    solution: S13VerticalSolution,
    gain: float,
    *,
    accepted_local_pairs: tuple[bool, ...] | None,
) -> S13VerticalSolution:
    """Materialize one immutable-P0 vertical candidate after relative audits."""

    key = str(float(gain))
    offsets = solution.gain_global_offsets_px.get(key)
    rows = solution.gain_local_row_residuals.get(key)
    if offsets is None or rows is None:
        raise ValueError(f"S1.3 vertical gain candidate is unavailable: {gain}")
    if accepted_local_pairs is None:
        accepted_local_pairs = tuple(False for _ in solution.pairs)
    if len(accepted_local_pairs) != len(solution.pairs):
        raise ValueError("S1.3 local vertical decisions do not align with pairs")
    selected_rows: list[np.ndarray] = []
    selected_pairs: list[S13VerticalPair] = []
    for pair, candidate_rows, accepted in zip(solution.pairs, rows, accepted_local_pairs, strict=True):
        apply = bool(accepted and pair.supported_row_count > 0 and np.any(candidate_rows != 0.0))
        selected = np.asarray(candidate_rows, dtype=np.float32).copy() if apply else np.zeros_like(candidate_rows)
        selected_rows.append(selected)
        selected_pairs.append(replace(
            pair,
            local_residual_min_px=float(selected.min()),
            local_residual_max_px=float(selected.max()),
            status="applied" if apply else "rolled_back",
            failure_reason=None if apply else "local_structure_non_degradation_not_proven",
        ))
    return replace(
        solution,
        global_offsets_px=tuple(offsets),
        local_row_residuals=tuple(selected_rows),
        pairs=tuple(selected_pairs),
        selected_gain=float(gain),
        audit={**dict(solution.audit), "selected_gain": float(gain), "selection": "rendered_relative_structure"},
    )


def render_s13_p1_from_raw(
    schedule: S012Schedule,
    calibration: CameraIntrinsics,
    image_loader: Callable[[int], np.ndarray],
    solution: S13VerticalSolution,
) -> S13P1Result:
    validate_s012_schedule(schedule)
    if len(solution.global_offsets_px) != len(schedule.assignments):
        raise ValueError("S1.3 M4 global offsets do not align with P0 sources")
    height, width = schedule.canvas_height, schedule.canvas_width
    output = np.zeros((height, width, 3), dtype=np.uint8)
    valid_full = np.zeros((height, width), dtype=bool)
    owner = np.full((height, width), -1, dtype=np.int32)
    assignment_full = np.full((height, width), -1, dtype=np.int32)
    source_u_full = np.full((height, width), np.nan, dtype=np.float32)
    source_v_full = np.full((height, width), np.nan, dtype=np.float32)
    transaction = np.full((height, width), -1, dtype=np.int32)
    inverse_maps = undistortion_maps(calibration)
    decoded: list[int] = []
    for source_index, assignment in enumerate(schedule.assignments):
        if assignment.zero_width:
            continue
        image = np.asarray(image_loader(assignment.frame_id))
        decoded.append(assignment.frame_id)
        if image.shape != (height, int(calibration.width), 3) or image.dtype != np.uint8:
            raise ValueError("S1.3 M4 raw RGB source shape/type changed")
        roi_width = assignment.width
        displacement = np.full((height, roi_width), solution.global_offsets_px[source_index], dtype=np.float32)
        if source_index > 0:
            pair = solution.pairs[source_index - 1]
            band_width = pair.application_right_x - pair.application_left_x
            if band_width > 0:
                local_width = min(band_width, roi_width)
                taper = np.linspace(1.0, 0.0, local_width, endpoint=True, dtype=np.float32)
                displacement[:, :local_width] += solution.local_row_residuals[source_index - 1][:, None] * taper[None, :]
                transaction[:, assignment.left_x:assignment.left_x + local_width] = pair.pair_index
        u, v, valid = _target_map(
            calibration, assignment.center_x, assignment.left_x, assignment.right_x,
            displacement, inverse_maps,
        )
        sampled = accelerated_remap(image, u, v, cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        roi = np.s_[:, assignment.left_x:assignment.right_x]
        output[roi][valid] = sampled[valid]
        valid_full[roi] = valid
        owner_roi = owner[roi]
        assignment_roi = assignment_full[roi]
        source_u_roi = source_u_full[roi]
        source_v_roi = source_v_full[roi]
        owner_roi[valid] = assignment.frame_id
        assignment_roi[valid] = assignment.assignment_index
        source_u_roi[valid], source_v_roi[valid] = u[valid], v[valid]
    if len(decoded) != len(set(decoded)):
        raise ValueError("S1.3 M4 decoded one contributor more than once")
    pixel = {
        "owner_frame_id": owner,
        "owner_source_index": assignment_full.copy(),
        "assignment_index": assignment_full,
        "source_u": source_u_full,
        "source_v": source_v_full,
        "valid": valid_full,
        "geometry_transaction_id": transaction,
        "secondary_frame_id": np.full((height, width), -1, dtype=np.int32),
        "secondary_weight": np.zeros((height, width), dtype=np.float32),
    }
    if not np.array_equal(valid_full, owner >= 0):
        raise ValueError("S1.3 M4 valid and owner topology disagree")
    return S13P1Result(output, valid_full, pixel, len(decoded), tuple(decoded))


__all__ = [
    "S13P1Result", "S13VerticalPair", "S13VerticalSolution", "estimate_s13_vertical",
    "render_s13_p1_from_raw", "vertical_candidate_solution",
]
