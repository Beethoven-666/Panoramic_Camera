"""M4-only global scalar dy and pair-local row residual for S1.3."""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping

import cv2
import numpy as np

from .calibrated_remap import accelerated_remap, camera_matrix, undistortion_maps
from .session import CameraIntrinsics
from .video_s12_schedule import S012Schedule, validate_s012_schedule
from .video_s13_bundle import atomic_write_json, sha256_file, write_npz


S13_VERTICAL_SOLUTION_SCHEMA = "gemini305-video-s13-vertical-solution/v2"
S13_P1_COMPLETION_SCHEMA = "gemini305-video-s13-p1-vertical-completion/v2"


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


def _solution_dimensions(solution: S13VerticalSolution) -> tuple[int, int, int]:
    assignment_count = len(solution.global_offsets_px)
    pair_count = len(solution.pairs)
    if assignment_count < 1 or pair_count != assignment_count - 1:
        raise ValueError("S1.3 vertical solution pair and assignment counts disagree")
    if len(solution.local_row_residuals) != pair_count:
        raise ValueError("S1.3 vertical solution local residual count disagrees")
    heights = {int(np.asarray(rows).size) for rows in solution.local_row_residuals}
    if pair_count and (len(heights) != 1 or 0 in heights):
        raise ValueError("S1.3 vertical solution local residual heights disagree")
    canvas_height = heights.pop() if heights else 0
    return assignment_count, pair_count, canvas_height


def save_s13_vertical_solution(
    directory: Path,
    solution: S13VerticalSolution,
) -> dict[str, object]:
    """Write a versioned, hash-bound representation of a sealed P1 solution."""

    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    assignment_count, pair_count, canvas_height = _solution_dimensions(solution)
    arrays: dict[str, np.ndarray] = {
        "global_offsets_px": np.asarray(solution.global_offsets_px, dtype=np.float64),
    }
    local_keys: list[str] = []
    for index, rows in enumerate(solution.local_row_residuals):
        key = f"pair_{index:04d}_local_row_residual_px"
        local_keys.append(key)
        arrays[key] = np.asarray(rows).copy()

    gain_global_keys: dict[str, str] = {}
    gain_local_keys: dict[str, list[str]] = {}
    for gain_index, gain in enumerate(sorted(solution.gain_global_offsets_px, key=float)):
        global_key = f"gain_{gain_index:04d}_global_offsets_px"
        gain_global_keys[gain] = global_key
        arrays[global_key] = np.asarray(solution.gain_global_offsets_px[gain], dtype=np.float64)
        rows_for_gain = solution.gain_local_row_residuals.get(gain)
        if rows_for_gain is None or len(rows_for_gain) != pair_count:
            raise ValueError(f"S1.3 vertical solution gain local rows are incomplete: {gain}")
        keys: list[str] = []
        for pair_index, rows in enumerate(rows_for_gain):
            key = f"gain_{gain_index:04d}_pair_{pair_index:04d}_local_row_residual_px"
            keys.append(key)
            arrays[key] = np.asarray(rows).copy()
        gain_local_keys[gain] = keys
    if set(gain_global_keys) != set(solution.gain_local_row_residuals):
        raise ValueError("S1.3 vertical solution gain array sets disagree")
    if str(float(solution.selected_gain)) not in gain_global_keys:
        raise ValueError("S1.3 vertical solution selected gain is unavailable")
    if solution.shoulder_width_px < 1 or not all(math.isfinite(float(value)) for value in solution.gain_scores.values()):
        raise ValueError("S1.3 vertical solution scalar metadata is invalid")
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("S1.3 vertical solution contains nonfinite arrays")

    npz_path = root / "vertical_solution.npz"
    write_npz(npz_path, arrays)
    document: dict[str, object] = {
        "schema": S13_VERTICAL_SOLUTION_SCHEMA,
        "selected_gain": float(solution.selected_gain),
        "gain_scores": {str(key): float(value) for key, value in solution.gain_scores.items()},
        "shoulder_width_px": int(solution.shoulder_width_px),
        "pair_count": pair_count,
        "assignment_count": assignment_count,
        "canvas_height": canvas_height,
        "global_offsets_key": "global_offsets_px",
        "local_row_residual_keys": local_keys,
        "gain_global_offsets_keys": gain_global_keys,
        "gain_local_row_residual_keys": gain_local_keys,
        "pairs": [asdict(pair) for pair in solution.pairs],
        "audit": dict(solution.audit),
        "npz": "vertical_solution.npz",
        "npz_sha256": sha256_file(npz_path),
    }
    atomic_write_json(root / "vertical_solution.json", document)
    atomic_write_json(root / "pair_report.json", {
        "schema": "gemini305-video-s13-m4-pair-report/v2",
        "pairs": document["pairs"],
        "audit": document["audit"],
    })
    return document


def load_s13_vertical_solution(
    directory: Path,
    *,
    expected_completion_sha256: str | None = None,
) -> S13VerticalSolution:
    """Strictly reconstruct a P1 solution without estimating any M4 state."""

    import json

    root = Path(directory)
    try:
        document = json.loads((root / "vertical_solution.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        if (root / "vertical_solution.npz").is_file():
            raise ValueError("legacy_p1_not_replayable") from exc
        raise ValueError("S1.3 P1 vertical solution is missing or invalid") from exc
    if document.get("schema") != S13_VERTICAL_SOLUTION_SCHEMA:
        raise ValueError("legacy_p1_not_replayable")

    completion: Mapping[str, object] | None = None
    completion_path = root / "P1_completion.json"
    if expected_completion_sha256 is not None:
        if not completion_path.is_file() or sha256_file(completion_path) != expected_completion_sha256:
            raise ValueError("S1.3 P1 completion hash mismatch")
    if completion_path.is_file():
        try:
            loaded_completion = json.loads(completion_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("S1.3 P1 completion is invalid") from exc
        if not isinstance(loaded_completion, Mapping):
            raise ValueError("S1.3 P1 completion is invalid")
        completion = loaded_completion

    npz_name = document.get("npz")
    if npz_name != "vertical_solution.npz":
        raise ValueError("S1.3 P1 vertical solution NPZ path is invalid")
    npz_path = root / str(npz_name)
    expected_npz_sha = document.get("npz_sha256")
    if not isinstance(expected_npz_sha, str) or not npz_path.is_file() or sha256_file(npz_path) != expected_npz_sha:
        raise ValueError("S1.3 P1 vertical solution NPZ hash mismatch")
    json_sha = sha256_file(root / "vertical_solution.json")
    if completion is not None:
        if completion.get("schema") != S13_P1_COMPLETION_SCHEMA or completion.get("sealed") is not True:
            raise ValueError("legacy_p1_not_replayable")
        if completion.get("vertical_solution_json") != "vertical_solution.json":
            raise ValueError("S1.3 P1 vertical solution JSON path mismatch")
        if completion.get("vertical_solution_npz") != "vertical_solution.npz":
            raise ValueError("S1.3 P1 vertical solution NPZ path mismatch")
        if completion.get("vertical_solution_json_sha256") != json_sha:
            raise ValueError("S1.3 P1 vertical solution JSON hash mismatch")
        if completion.get("vertical_solution_npz_sha256") != expected_npz_sha:
            raise ValueError("S1.3 P1 completion vertical solution NPZ hash mismatch")
        selected = completion.get("selected_global_gain")
        if not isinstance(selected, (int, float)) or float(selected) != float(document.get("selected_gain")):
            raise ValueError("S1.3 P1 selected gain mismatch")

    try:
        assignment_count = int(document["assignment_count"])
        pair_count = int(document["pair_count"])
        canvas_height = int(document["canvas_height"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("S1.3 P1 vertical solution dimensions are invalid") from exc
    if assignment_count < 1 or pair_count != assignment_count - 1 or (pair_count and canvas_height < 1):
        raise ValueError("S1.3 P1 vertical solution dimensions disagree")
    pair_rows = document.get("pairs")
    audit = document.get("audit")
    if not isinstance(pair_rows, list) or len(pair_rows) != pair_count or not isinstance(audit, Mapping):
        raise ValueError("S1.3 P1 vertical solution pair metadata is invalid")
    try:
        pair_report = json.loads((root / "pair_report.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("S1.3 P1 pair report is missing or invalid") from exc
    if pair_report.get("pairs") != pair_rows or pair_report.get("audit") != audit:
        raise ValueError("S1.3 P1 pair report and vertical solution disagree")

    global_key = document.get("global_offsets_key")
    local_keys = document.get("local_row_residual_keys")
    gain_global_keys = document.get("gain_global_offsets_keys")
    gain_local_keys = document.get("gain_local_row_residual_keys")
    if global_key != "global_offsets_px" or not isinstance(local_keys, list) or len(local_keys) != pair_count:
        raise ValueError("S1.3 P1 vertical solution array mapping is invalid")
    if not isinstance(gain_global_keys, Mapping) or not isinstance(gain_local_keys, Mapping):
        raise ValueError("S1.3 P1 vertical gain array mapping is invalid")
    if set(gain_global_keys) != set(gain_local_keys):
        raise ValueError("S1.3 P1 vertical gain array sets disagree")

    with np.load(npz_path, allow_pickle=False) as archive:
        required = {str(global_key), *(str(key) for key in local_keys)}
        for gain, key in gain_global_keys.items():
            required.add(str(key))
            row_keys = gain_local_keys[gain]
            if not isinstance(row_keys, list) or len(row_keys) != pair_count:
                raise ValueError(f"S1.3 P1 vertical gain row mapping is invalid: {gain}")
            required.update(str(item) for item in row_keys)
        if set(archive.files) != required:
            raise ValueError("S1.3 P1 vertical solution NPZ keys disagree")
        global_offsets = np.asarray(archive[str(global_key)], dtype=np.float64)
        local_rows = tuple(np.asarray(archive[str(key)]).copy() for key in local_keys)
        gain_globals = {
            str(gain): tuple(float(value) for value in np.asarray(archive[str(key)], dtype=np.float64))
            for gain, key in gain_global_keys.items()
        }
        gain_rows = {
            str(gain): tuple(np.asarray(archive[str(key)]).copy() for key in gain_local_keys[gain])
            for gain in gain_local_keys
        }
    arrays = (global_offsets, *local_rows, *(np.asarray(value) for value in gain_globals.values()),
              *(row for rows in gain_rows.values() for row in rows))
    if global_offsets.shape != (assignment_count,) or any(row.shape != (canvas_height,) for row in local_rows):
        raise ValueError("S1.3 P1 vertical solution array shapes disagree")
    if any(np.asarray(value).shape != (assignment_count,) for value in gain_globals.values()):
        raise ValueError("S1.3 P1 vertical gain offset shapes disagree")
    if any(row.shape != (canvas_height,) for rows in gain_rows.values() for row in rows):
        raise ValueError("S1.3 P1 vertical gain row shapes disagree")
    if not all(np.isfinite(value).all() for value in arrays):
        raise ValueError("S1.3 P1 vertical solution contains nonfinite arrays")
    try:
        pairs = tuple(S13VerticalPair(**row) for row in pair_rows)
        gain_score_rows = document["gain_scores"]
        if not isinstance(gain_score_rows, Mapping):
            raise TypeError("gain_scores must be a mapping")
        gain_scores = {str(key): float(value) for key, value in gain_score_rows.items()}
        shoulder_width = int(document["shoulder_width_px"])
        selected_gain = float(document["selected_gain"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("S1.3 P1 vertical solution metadata is invalid") from exc
    if tuple(pair.pair_index for pair in pairs) != tuple(range(pair_count)):
        raise ValueError("S1.3 P1 vertical pair indices are invalid")
    if shoulder_width < 1 or not math.isfinite(selected_gain) or not all(
        math.isfinite(value) for value in gain_scores.values()
    ):
        raise ValueError("S1.3 P1 vertical scalar metadata is invalid")
    if str(float(selected_gain)) not in gain_globals:
        raise ValueError("S1.3 P1 selected gain is unavailable")
    return S13VerticalSolution(
        global_offsets_px=tuple(float(value) for value in global_offsets),
        local_row_residuals=local_rows,
        pairs=pairs,
        selected_gain=selected_gain,
        gain_scores=gain_scores,
        shoulder_width_px=shoulder_width,
        audit=dict(audit),
        gain_global_offsets_px=gain_globals,
        gain_local_row_residuals=gain_rows,
    )


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
    # M4 measurements are a small, CPU-owned decision input.  OpenCV's
    # fixed-point interpolation is the reference for its downstream rows;
    # using a float CUDA remap here can change a discrete vertical decision.
    sampled = cv2.remap(image, u, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
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


def _local_row_evidence(
    left: np.ndarray, right: np.ndarray, valid: np.ndarray
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return the gain-independent Farneback and texture evidence for one pair."""

    if int(valid.sum()) < 256:
        return None
    flow = cv2.calcOpticalFlowFarneback(
        left, right, None, 0.5, 3, 19, 3, 5, 1.1, cv2.OPTFLOW_FARNEBACK_GAUSSIAN
    )
    return flow[:, :, 1], np.abs(cv2.Sobel(left, cv2.CV_32F, 1, 0, ksize=3))


def _local_rows(
    left: np.ndarray,
    right: np.ndarray,
    valid: np.ndarray,
    global_relative: float,
    *,
    evidence: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, int]:
    height = left.shape[0]
    residual = np.zeros(height, dtype=np.float32)
    evidence = _local_row_evidence(left, right, valid) if evidence is None else evidence
    if evidence is None:
        return residual, 0
    flow_y, texture = evidence
    supported = np.zeros(height, dtype=bool)
    for row in range(height):
        mask = valid[row] & np.isfinite(flow_y[row]) & (texture[row] > 3.0)
        if int(mask.sum()) < 12:
            continue
        row_correction = -float(np.median(flow_y[row, mask])) - global_relative
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
    measurement_workers: int = 1,
) -> S13VerticalSolution:
    camera_matrix(calibration)
    validate_s012_schedule(schedule)
    shoulder = int(np.clip(shoulder_width_px, 64, 128))
    if measurement_workers not in (1, 2):
        raise ValueError("S1.3 vertical measurement supports only 1 or 2 workers")
    source_images = {
        assignment.frame_id: np.asarray(image_loader(assignment.frame_id))
        for assignment in schedule.assignments
    }

    def load(frame_id: int) -> np.ndarray:
        return source_images[frame_id]

    half = shoulder // 2

    def measure_pair(pair_index: int) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray, int, int]:
        left_assignment = schedule.assignments[pair_index]
        right_assignment = schedule.assignments[pair_index + 1]
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
        return correction, response, left_gray, right_gray, common, left_x, right_x

    pair_indices = tuple(range(len(schedule.assignments) - 1))
    if measurement_workers == 1:
        measured = tuple(measure_pair(pair_index) for pair_index in pair_indices)
    else:
        # Each measurement owns its remap/flow buffers.  Results are consumed
        # in pair order so the solver and every discrete decision remain exact.
        with ThreadPoolExecutor(
            max_workers=measurement_workers, thread_name_prefix="s13-m4"
        ) as pool:
            measured = tuple(pool.map(measure_pair, pair_indices))
    measurements = [item[0] for item in measured]
    responses = [item[1] for item in measured]
    sampled_pairs = [item[2:] for item in measured]
    observation = np.asarray(measurements, dtype=np.float64)
    weight = np.clip(np.asarray(responses, dtype=np.float64), 0.0, 1.0)
    local_evidence = tuple(
        _local_row_evidence(left, right, common)
        for left, right, common, _left_x, _right_x in sampled_pairs
    )
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
        for pair_index, ((left_gray, right_gray, common, left_x, right_x), response, evidence) in enumerate(
            zip(sampled_pairs, responses, local_evidence, strict=True)
        ):
            left_assignment = schedule.assignments[pair_index]
            right_assignment = schedule.assignments[pair_index + 1]
            relative = float(gain_offsets[pair_index + 1] - gain_offsets[pair_index])
            rows, supported = _local_rows(
                left_gray, right_gray, common, relative, evidence=evidence
            ) if response >= 0.05 else (
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
            "measurement_workers": measurement_workers,
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
    *,
    resident_stage: Any | None = None,
    resident_device_remap: Callable[[int, np.ndarray, np.ndarray, np.ndarray], Any] | None = None,
    reference_remap: bool = False,
) -> S13P1Result:
    validate_s012_schedule(schedule)
    if len(solution.global_offsets_px) != len(schedule.assignments):
        raise ValueError("S1.3 M4 global offsets do not align with P0 sources")
    height, width = schedule.canvas_height, schedule.canvas_width
    output = np.zeros((height, width, 3), dtype=np.uint8)
    if (resident_stage is None) != (resident_device_remap is None):
        raise ValueError("S1.3 P1 resident stage and remap must be supplied together")
    output_device = resident_stage.new_stage_canvas(height, width) if resident_stage is not None else None
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
        sampled = None if output_device is not None else (
            cv2.remap(image, u, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            if reference_remap else accelerated_remap(
                image, u, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
            )
        )
        roi = np.s_[:, assignment.left_x:assignment.right_x]
        if output_device is not None:
            resident_stage.compose_owner_roi(
                output_device, assignment.left_x, assignment.right_x,
                resident_device_remap(assignment.frame_id, image, u, v), valid,
            )
        else:
            assert sampled is not None
            output[roi][valid] = sampled[valid]
        valid_full[roi] = valid
        owner_roi = owner[roi]
        assignment_roi = assignment_full[roi]
        source_u_roi = source_u_full[roi]
        source_v_roi = source_v_full[roi]
        owner_roi[valid] = assignment.frame_id
        assignment_roi[valid] = assignment.assignment_index
        source_u_roi[valid], source_v_roi[valid] = u[valid], v[valid]
    if output_device is not None:
        output = resident_stage.download_stage_canvas("P1", output_device)
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


def render_s13_p1_local_patch_image(
    schedule: S012Schedule,
    calibration: CameraIntrinsics,
    image_loader: Callable[[int], np.ndarray],
    global_result: S13P1Result,
    solution: S13VerticalSolution,
    *,
    reference_remap: bool = False,
) -> np.ndarray:
    """Apply M4 local-row candidates over a rendered global-gain parent.

    The local bands are non-overlapping by contract.  This is used only while
    comparing the four gain candidates: it avoids re-remapping every source
    merely to evaluate a few narrow right-owner bands.
    """

    validate_s012_schedule(schedule)
    output = np.asarray(global_result.image).copy()
    inverse_maps = undistortion_maps(calibration)
    height = schedule.canvas_height
    for source_index, assignment in enumerate(schedule.assignments[1:], start=1):
        pair = solution.pairs[source_index - 1]
        band_width = pair.application_right_x - pair.application_left_x
        local_width = min(max(0, band_width), assignment.width)
        if local_width == 0 or not np.any(solution.local_row_residuals[source_index - 1]):
            continue
        displacement = np.full(
            (height, local_width), solution.global_offsets_px[source_index], np.float32
        )
        taper = np.linspace(1.0, 0.0, local_width, endpoint=True, dtype=np.float32)
        displacement += solution.local_row_residuals[source_index - 1][:, None] * taper
        left, right = assignment.left_x, assignment.left_x + local_width
        u, v, valid = _target_map(
            calibration, assignment.center_x, left, right, displacement, inverse_maps
        )
        sampled = (
            cv2.remap(
                np.asarray(image_loader(assignment.frame_id)), u, v, cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0,
            ) if reference_remap else accelerated_remap(
                np.asarray(image_loader(assignment.frame_id)), u, v, cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0,
            )
        )
        roi = output[:, left:right]
        roi[...] = 0
        roi[valid] = sampled[valid]
    return output


def render_s13_p1_exact_probe(
    schedule: S012Schedule,
    calibration: CameraIntrinsics,
    image_loader: Callable[[int], np.ndarray],
    solution: S13VerticalSolution,
    *,
    global_x0: int,
    global_x1: int,
) -> np.ndarray:
    """Render an exact global-canvas P1 slice without a full candidate canvas."""

    validate_s012_schedule(schedule)
    if not 0 <= global_x0 < global_x1 <= schedule.canvas_width:
        raise ValueError("S1.3 exact vertical probe is outside the canvas")
    height = schedule.canvas_height
    output = np.zeros((height, global_x1 - global_x0, 3), dtype=np.uint8)
    inverse_maps = undistortion_maps(calibration)
    for source_index, assignment in enumerate(schedule.assignments):
        left, right = max(global_x0, assignment.left_x), min(global_x1, assignment.right_x)
        if left >= right or assignment.zero_width:
            continue
        relative_x = np.arange(left - assignment.left_x, right - assignment.left_x)
        displacement = np.full((height, right - left), solution.global_offsets_px[source_index], np.float32)
        if source_index > 0:
            pair = solution.pairs[source_index - 1]
            local_width = min(pair.application_right_x - pair.application_left_x, assignment.width)
            active = relative_x < local_width
            if np.any(active):
                taper = np.linspace(1.0, 0.0, local_width, endpoint=True, dtype=np.float32)
                displacement[:, active] += (
                    solution.local_row_residuals[source_index - 1][:, None] * taper[relative_x[active]][None, :]
                )
        u, v, valid = _target_map(
            calibration, assignment.center_x, left, right, displacement, inverse_maps,
        )
        sampled = cv2.remap(
            np.asarray(image_loader(assignment.frame_id)), u, v, cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
        target = output[:, left - global_x0:right - global_x0]
        target[valid] = sampled[valid]
    return output


__all__ = [
    "S13P1Result", "S13VerticalPair", "S13VerticalSolution", "S13_P1_COMPLETION_SCHEMA",
    "S13_VERTICAL_SOLUTION_SCHEMA",
    "estimate_s13_vertical", "load_s13_vertical_solution", "render_s13_p1_from_raw",
    "save_s13_vertical_solution", "vertical_candidate_solution",
    "render_s13_p1_local_patch_image",
    "render_s13_p1_exact_probe",
]
