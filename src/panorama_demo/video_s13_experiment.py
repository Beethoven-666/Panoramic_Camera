"""Development-only M0--M3 runner for the isolated S1.3 candidate."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
import uuid
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from .video_algorithm import VideoAlgorithmSpec
from .video_s13_base_renderer import render_s13_p0
from .video_s13_bundle import (
    atomic_write_json,
    discard_staging,
    new_generation_staging,
    publish_generation,
    seal_p0,
    seal_stage,
    sha256_file,
    update_current_base,
    update_current_latest,
    verify_p0_completion,
    verify_stage,
    write_csv,
    write_image,
    write_npz,
)
from .video_s13_contract import (
    S13_FORMAL_M6_ALGORITHM_ID,
    S13_M51_R2_P2_COMPLETION_SCHEMA,
    S13_M51_R3_P2_COMPLETION_SCHEMA,
    S13_M51_R4_P2_COMPLETION_SCHEMA,
    load_s13_config,
)
from .video_s13_motion import measure_s13_motion
from .video_s13_m5 import (
    _finalize_v5_transaction,
    render_s13_component_roi_from_raw,
    run_s13_m5,
)
from .video_s13_m51_r4_component_chain import canonical_s13_support_sha256
from .video_s13_m61_evidence import (
    P2_V4_COMPLETION_SCHEMA,
    S13PhotometricEvidenceConfig,
    build_native_s13_m61_evidence,
)
from .video_s13_m61_config import load_s13_m61_effective_config
from .video_s13_m6 import run_s13_m6
from .video_s13_blend import BLEND_SCHEMA, blend_transaction_document
from .video_s13_p3_hard_audit import audit_s13_p3_stage, verify_sealed_s13_p3
from .video_s13_photometric import photometric_solution_document
from .video_s13_replay import (
    P2_REPLAY_SCHEMA,
    P2_REPLAY_V2_SCHEMA,
    load_verified_s13_p2_for_m6,
    replay_pair_arrays,
)
from .video_s13_progress import S13M3Layout, build_s13_m3_layout
from .video_s13_schedule import S13SchedulePlan, plan_s13_m3_schedule
from .video_s13_session import S13Session, load_s13_session, read_s13_rgb
from .video_s13_trajectory import S13Trajectory, load_s13_trajectory
from .video_s13_vertical import (
    estimate_s13_vertical,
    load_s13_vertical_solution,
    save_s13_vertical_solution,
)
from .video_s13_selection import select_s13_vertical_parent
from .video_s13_v6_r2_verifier import (
    canonical_source_map_slice_sha256,
    component_decision_stable_payload,
    component_decision_stable_sha256,
    segment_decision_stable_sha256,
    verify_s13_v6_r2_p2,
)


def _strong_edge_trace_metrics(
    image: np.ndarray, valid: np.ndarray
) -> tuple[dict[str, object], np.ndarray]:
    """Measure the dominant per-column edge trace in an external validation ROI."""

    if image.ndim != 3 or image.shape[:2] != valid.shape:
        raise ValueError("S1.3 box ROI image/valid shapes disagree")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.hypot(gx, gy)
    finite_values = magnitude[valid]
    threshold = (
        max(20.0, float(np.percentile(finite_values, 75.0)))
        if finite_values.size else math.inf
    )
    trace = np.full(image.shape[1], -1, dtype=np.int32)
    double_edge = np.zeros(image.shape[1], dtype=bool)
    for column in range(image.shape[1]):
        values = magnitude[:, column].copy()
        values[~valid[:, column]] = 0.0
        if values.size < 3:
            continue
        peaks = np.flatnonzero(
            (values >= np.roll(values, 1)) & (values >= np.roll(values, -1))
            & (values >= threshold)
        )
        peaks = peaks[(peaks > 0) & (peaks + 1 < values.size)]
        if peaks.size == 0:
            continue
        primary = min((int(row) for row in peaks), key=lambda row: (-values[row], row))
        trace[column] = primary
        double_edge[column] = any(
            abs(int(row) - primary) >= 3
            and values[int(row)] >= 0.65 * values[primary]
            for row in peaks if int(row) != primary
        )
    adjacent = (trace[:-1] >= 0) & (trace[1:] >= 0)
    steps = np.abs(np.diff(trace.astype(np.float64))[adjacent])
    second_evaluable = (trace[:-2] >= 0) & (trace[1:-1] >= 0) & (trace[2:] >= 0)
    second = np.abs(np.diff(trace.astype(np.float64), n=2)[second_evaluable])
    missing = trace < 0
    longest_break = 0
    current_break = 0
    for value in missing:
        current_break = current_break + 1 if value else 0
        longest_break = max(longest_break, current_break)
    signed_steps = np.diff(trace.astype(np.float64))[adjacent]
    nonzero_signs = np.sign(signed_steps[np.abs(signed_steps) >= 0.5])
    backtrack_count = int(np.count_nonzero(np.diff(nonzero_signs) != 0))
    overlay = image.copy()
    columns = np.flatnonzero(trace >= 0)
    overlay[trace[columns], columns] = (0, 0, 255)
    metrics: dict[str, object] = {
        "evaluable_column_count": int(columns.size),
        "coverage_fraction": float(columns.size / max(len(trace), 1)),
        "edge_step_p50_px": (
            float(np.percentile(steps, 50.0)) if steps.size else None
        ),
        "edge_step_p95_px": (
            float(np.percentile(steps, 95.0)) if steps.size else None
        ),
        "maximum_local_step_px": float(np.max(steps)) if steps.size else None,
        "second_difference_p95_px": (
            float(np.percentile(second, 95.0)) if second.size else None
        ),
        "break_length_px": int(longest_break),
        "double_edge_length_px": int(np.count_nonzero(double_edge)),
        "break_double_edge_union_length_px": int(
            np.count_nonzero(missing | double_edge)
        ),
        "backtrack_count": backtrack_count,
        "threshold": threshold if math.isfinite(threshold) else None,
    }
    return metrics, overlay


@dataclass(frozen=True)
class _ExternalBoxObservationSeed:
    """Exact observation support used only by the external box acceptance audit."""

    pair_index: int
    component_id: int
    support_sha256: str
    support_xy: np.ndarray
    obligation_id: str
    severe: bool
    evaluable: bool


def _support_line(support_xy: np.ndarray) -> tuple[float, float]:
    support = np.asarray(support_xy, dtype=np.float64)
    if support.ndim != 2 or support.shape[1] != 2 or support.shape[0] < 2:
        raise ValueError("S1.3 external edge support must contain at least two xy points")
    design = np.column_stack((support[:, 0], np.ones(support.shape[0])))
    slope, intercept = np.linalg.lstsq(design, support[:, 1], rcond=None)[0]
    return float(slope), float(intercept)


def _external_box_physical_tracks(
    observations: Sequence[_ExternalBoxObservationSeed],
    *,
    minimum_y_overlap_fraction: float,
    maximum_predicted_y_difference_px: float,
    maximum_normal_difference_degrees: float,
) -> tuple[dict[str, object], ...]:
    """Deterministically join adjacent-pair seeds without merging same-pair edges."""

    rows = tuple(sorted(
        observations,
        key=lambda row: (row.pair_index, row.component_id, row.support_sha256),
    ))
    parent = list(range(len(rows)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def join(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    by_pair: dict[int, list[int]] = {}
    line_by_index: dict[int, tuple[float, float]] = {}
    bbox_by_index: dict[int, tuple[float, float, float, float]] = {}
    for index, row in enumerate(rows):
        by_pair.setdefault(row.pair_index, []).append(index)
        line_by_index[index] = _support_line(row.support_xy)
        support = np.asarray(row.support_xy)
        bbox_by_index[index] = (
            float(np.min(support[:, 0])), float(np.min(support[:, 1])),
            float(np.max(support[:, 0]) + 1), float(np.max(support[:, 1]) + 1),
        )
    for pair_index in sorted(by_pair):
        left_indices = by_pair[pair_index]
        right_indices = by_pair.get(pair_index + 1, [])
        candidates: list[tuple[float, float, int, int]] = []
        for left_index in left_indices:
            lx0, ly0, lx1, ly1 = bbox_by_index[left_index]
            left_slope, left_intercept = line_by_index[left_index]
            for right_index in right_indices:
                rx0, ry0, rx1, ry1 = bbox_by_index[right_index]
                overlap = max(0.0, min(ly1, ry1) - max(ly0, ry0)) / max(
                    1.0, min(ly1 - ly0, ry1 - ry0)
                )
                if overlap < minimum_y_overlap_fraction:
                    continue
                right_slope, right_intercept = line_by_index[right_index]
                angle = abs(math.degrees(math.atan(left_slope) - math.atan(right_slope)))
                if angle > maximum_normal_difference_degrees:
                    continue
                common_x = 0.25 * (lx0 + lx1 + rx0 + rx1)
                predicted_difference = abs(
                    left_slope * common_x + left_intercept
                    - right_slope * common_x - right_intercept
                )
                if predicted_difference > maximum_predicted_y_difference_px:
                    continue
                candidates.append((predicted_difference, angle, left_index, right_index))
        # One-to-one greedy matching is deterministic.  Geometrically ambiguous
        # ties remain separate tracks rather than risking a physical-edge merge.
        used_left: set[int] = set()
        used_right: set[int] = set()
        for predicted_difference, angle, left_index, right_index in sorted(candidates):
            if left_index in used_left or right_index in used_right:
                continue
            left_alternatives = sorted(
                row for row in candidates if row[2] == left_index
            )
            right_alternatives = sorted(
                row for row in candidates if row[3] == right_index
            )
            best_cost = predicted_difference + angle
            ambiguous_left = len(left_alternatives) > 1 and (
                left_alternatives[1][0] + left_alternatives[1][1]
                <= best_cost + max(0.5, 0.10 * best_cost)
            )
            ambiguous_right = len(right_alternatives) > 1 and (
                right_alternatives[1][0] + right_alternatives[1][1]
                <= best_cost + max(0.5, 0.10 * best_cost)
            )
            if ambiguous_left or ambiguous_right:
                continue
            join(left_index, right_index)
            used_left.add(left_index)
            used_right.add(right_index)

    groups: dict[int, list[_ExternalBoxObservationSeed]] = {}
    for index, row in enumerate(rows):
        groups.setdefault(find(index), []).append(row)
    tracks: list[dict[str, object]] = []
    for members in groups.values():
        authority = sorted(
            (row.pair_index, row.component_id, row.support_sha256) for row in members
        )
        digest = hashlib.sha256(json.dumps(authority, separators=(",", ":")).encode()).hexdigest()
        support = np.unique(np.concatenate(
            [np.asarray(row.support_xy, dtype=np.int32) for row in members], axis=0
        ), axis=0)
        tracks.append({
            "track_id": f"box-edge-{digest[:20]}",
            "support_xy": support,
            "observation_authority": [
                {
                    "pair_index": pair_index,
                    "component_id": component_id,
                    "support_sha256": support_sha256,
                }
                for pair_index, component_id, support_sha256 in authority
            ],
            "obligation_ids": sorted(row.obligation_id for row in members),
        })
    return tuple(sorted(tracks, key=lambda row: str(row["track_id"])))


def _seeded_edge_trace_metrics(
    image: np.ndarray,
    valid: np.ndarray,
    tracks: Sequence[Mapping[str, object]],
    *,
    roi_origin_xy: tuple[int, int] = (0, 0),
    search_radius_px: int = 6,
) -> tuple[dict[str, object], np.ndarray]:
    """Trace each frozen physical edge in its own narrow, deterministic band."""

    if image.ndim != 3 or image.shape[:2] != valid.shape:
        raise ValueError("S1.3 box ROI image/valid shapes disagree")
    if search_radius_px < 1:
        raise ValueError("S1.3 external edge search radius must be positive")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.hypot(gx, gy)
    height, width = valid.shape
    origin_x, origin_y = roi_origin_xy
    overlay = image.copy()
    palette = ((0, 0, 255), (0, 255, 255), (255, 0, 255), (255, 255, 0))
    all_steps: list[np.ndarray] = []
    all_second: list[np.ndarray] = []
    track_metrics: list[dict[str, object]] = []
    total_evaluable = 0
    total_double = 0
    total_union = 0
    total_backtracks = 0
    longest_break_overall = 0
    for track_index, track in enumerate(sorted(tracks, key=lambda row: str(row["track_id"]))):
        support = np.asarray(track["support_xy"], dtype=np.float64).copy()
        support[:, 0] -= origin_x
        support[:, 1] -= origin_y
        slope, intercept = _support_line(support)
        normal = np.asarray((-slope, 1.0), dtype=np.float64)
        normal /= max(float(np.linalg.norm(normal)), 1e-12)
        directional = np.abs(gx * normal[0] + gy * normal[1])
        expected = slope * np.arange(width, dtype=np.float64) + intercept
        band_mask = np.zeros(valid.shape, dtype=bool)
        row_candidates: list[np.ndarray] = []
        for column, expected_row in enumerate(expected):
            lower = max(1, int(math.floor(expected_row)) - search_radius_px)
            upper = min(height - 1, int(math.ceil(expected_row)) + search_radius_px + 1)
            rows_for_column = np.arange(lower, upper, dtype=np.int32)
            row_candidates.append(rows_for_column)
            band_mask[rows_for_column, column] = True
        finite_values = directional[band_mask & valid]
        threshold = (
            max(20.0, float(np.percentile(finite_values, 75.0)))
            if finite_values.size else math.inf
        )
        scale = max(threshold, 1.0)
        costs: list[np.ndarray] = []
        predecessors: list[np.ndarray] = []
        for column, rows_for_column in enumerate(row_candidates):
            if not rows_for_column.size:
                costs.append(np.empty(0, dtype=np.float64))
                predecessors.append(np.empty(0, dtype=np.int32))
                continue
            emission = (
                -directional[rows_for_column, column] / scale
                + 0.05 * np.abs(rows_for_column.astype(np.float64) - expected[column])
            )
            emission[~valid[rows_for_column, column]] += 1_000_000.0
            if column == 0 or not costs[column - 1].size:
                costs.append(emission)
                predecessors.append(np.full(rows_for_column.size, -1, dtype=np.int32))
                continue
            previous_rows = row_candidates[column - 1]
            expected_step = expected[column] - expected[column - 1]
            transition = 0.30 * np.abs(
                rows_for_column[:, None].astype(np.float64)
                - previous_rows[None, :].astype(np.float64) - expected_step
            )
            combined = transition + costs[column - 1][None, :]
            previous = np.argmin(combined, axis=1).astype(np.int32)
            costs.append(emission + combined[np.arange(rows_for_column.size), previous])
            predecessors.append(previous)
        trace = np.full(width, -1, dtype=np.int32)
        segment_ends = [
            column for column in range(width)
            if row_candidates[column].size
            and (column + 1 == width or not row_candidates[column + 1].size)
        ]
        for segment_end in segment_ends:
            state = int(np.argmin(costs[segment_end]))
            column = segment_end
            while column >= 0 and row_candidates[column].size:
                trace[column] = int(row_candidates[column][state])
                if column == 0 or not row_candidates[column - 1].size:
                    break
                state = int(predecessors[column][state])
                column -= 1
        selected_columns = np.flatnonzero(trace >= 0)
        if selected_columns.size:
            selected_rows = trace[selected_columns]
            orientation_ratio = directional[selected_rows, selected_columns] / np.maximum(
                magnitude[selected_rows, selected_columns], 1e-6
            )
            supported = (
                valid[selected_rows, selected_columns]
                & (directional[selected_rows, selected_columns] >= threshold)
                & (orientation_ratio >= math.cos(math.radians(35.0)))
            )
            trace[selected_columns[~supported]] = -1
        double_edge = np.zeros(width, dtype=bool)
        for column in np.flatnonzero(trace >= 0):
            rows_for_column = row_candidates[column]
            values = directional[rows_for_column, column]
            local = np.flatnonzero(
                (values >= np.roll(values, 1)) & (values >= np.roll(values, -1))
                & (values >= threshold)
            )
            local = local[(local > 0) & (local + 1 < values.size)]
            primary_row = int(trace[column])
            primary_strength = float(directional[primary_row, column])
            double_edge[column] = any(
                abs(int(rows_for_column[index]) - primary_row) >= 3
                and float(values[index]) >= 0.65 * primary_strength
                and float(directional[int(rows_for_column[index]), column])
                / max(float(magnitude[int(rows_for_column[index]), column]), 1e-6)
                >= math.cos(math.radians(35.0))
                for index in local
            )
        adjacent = (trace[:-1] >= 0) & (trace[1:] >= 0)
        steps = np.abs(np.diff(trace.astype(np.float64))[adjacent])
        second_evaluable = (trace[:-2] >= 0) & (trace[1:-1] >= 0) & (trace[2:] >= 0)
        second = np.abs(np.diff(trace.astype(np.float64), n=2)[second_evaluable])
        missing = trace < 0
        longest_break = 0
        current_break = 0
        for value in missing:
            current_break = current_break + 1 if value else 0
            longest_break = max(longest_break, current_break)
        signed_steps = np.diff(trace.astype(np.float64))[adjacent]
        nonzero_signs = np.sign(signed_steps[np.abs(signed_steps) >= 0.5])
        backtracks = int(np.count_nonzero(np.diff(nonzero_signs) != 0))
        columns = np.flatnonzero(trace >= 0)
        color = palette[track_index % len(palette)]
        overlay[trace[columns], columns] = color
        union = int(np.count_nonzero(missing | double_edge))
        track_metrics.append({
            "track_id": str(track["track_id"]),
            "observation_authority": list(track.get("observation_authority", [])),
            "obligation_ids": list(track.get("obligation_ids", [])),
            "evaluable_column_count": int(columns.size),
            "coverage_fraction": float(columns.size / max(width, 1)),
            "edge_step_p95_px": float(np.percentile(steps, 95.0)) if steps.size else None,
            "maximum_local_step_px": float(np.max(steps)) if steps.size else None,
            "break_length_px": int(longest_break),
            "double_edge_length_px": int(np.count_nonzero(double_edge)),
            "break_double_edge_union_length_px": union,
            "backtrack_count": backtracks,
            "threshold": threshold if math.isfinite(threshold) else None,
        })
        if steps.size:
            all_steps.append(steps)
        if second.size:
            all_second.append(second)
        total_evaluable += int(columns.size)
        total_double += int(np.count_nonzero(double_edge))
        total_union += union
        total_backtracks += backtracks
        longest_break_overall = max(longest_break_overall, longest_break)
    steps = np.concatenate(all_steps) if all_steps else np.empty(0, dtype=np.float64)
    second = np.concatenate(all_second) if all_second else np.empty(0, dtype=np.float64)
    denominator = max(width * len(tracks), 1)
    return {
        "trace_mode": "exact_support_seeded_physical_tracks/v1",
        "track_count": len(tracks),
        "tracks": track_metrics,
        "evaluable_column_count": total_evaluable,
        "coverage_fraction": float(total_evaluable / denominator),
        "edge_step_p50_px": float(np.percentile(steps, 50.0)) if steps.size else None,
        "edge_step_p95_px": float(np.percentile(steps, 95.0)) if steps.size else None,
        "maximum_local_step_px": float(np.max(steps)) if steps.size else None,
        "second_difference_p95_px": (
            float(np.percentile(second, 95.0)) if second.size else None
        ),
        "break_length_px": int(longest_break_overall),
        "double_edge_length_px": total_double,
        "break_double_edge_union_length_px": total_union,
        "backtrack_count": total_backtracks,
        "threshold": None,
    }, overlay


def _box_target_coverage_summary(
    component_doc: Mapping[str, object],
    relevant_obligations: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Separate target identity, repair denominator, and non-severe anchors."""

    accepted_ids = {str(value) for value in component_doc.get("accepted_segment_ids", [])}
    resolved_authority = {
        (int(row["pair_index"]), int(row["component_id"]), str(row["support_sha256"]))
        for segment in component_doc.get("segments", [])
        if str(segment.get("segment_id")) in accepted_ids
        and segment.get("state") == "resolved"
        for row in segment.get("observation_authority", [])
    }
    repair_rows = [
        row for row in relevant_obligations
        if row.get("severe") is True and row.get("evaluable") is True
    ]
    anchor_rows = [
        row for row in relevant_obligations
        if row.get("severe") is not True and row.get("evaluable") is True
    ]
    unresolved = [
        row for row in repair_rows
        if (int(row["pair_index"]), int(row["component_id"]), str(row["support_sha256"]))
        not in resolved_authority
    ]
    return {
        "box_target_obligation_ids": [str(row["obligation_id"]) for row in relevant_obligations],
        "box_target_repair_obligation_ids": [str(row["obligation_id"]) for row in repair_rows],
        "box_target_anchor_obligation_ids": [str(row["obligation_id"]) for row in anchor_rows],
        "box_target_unresolved_repair_obligation_ids": [
            str(row["obligation_id"]) for row in unresolved
        ],
        "repair_obligations_covered": bool(repair_rows) and not unresolved,
    }


def _segment_authority_key(row: Mapping[str, object]) -> tuple[int, int, str]:
    return (
        int(row["pair_index"]), int(row["component_id"]),
        str(row["support_sha256"]),
    )


def _segment_authority_support(
    pairs: Sequence[object],
    observation_authority: Sequence[Mapping[str, object]],
) -> tuple[np.ndarray, tuple[dict[str, object], ...]]:
    """Collect only exact `(pair, component, support SHA)` segment evidence."""

    authority = tuple(sorted(_segment_authority_key(row) for row in observation_authority))
    if len(set(authority)) != len(authority):
        raise RuntimeError("S1.3 segment observation authority contains duplicates")
    available: dict[tuple[int, int, str], np.ndarray] = {}
    for pair in pairs:
        for observation, evidence in pair.component_evidence:
            key = (
                int(observation.pair_index), int(observation.component_id),
                str(observation.mask_sha256),
            )
            if key in available:
                raise RuntimeError("S1.3 exact observation authority is not unique")
            available[key] = np.asarray(evidence.support_xy, dtype=np.int32)
    missing = [key for key in authority if key not in available]
    if missing:
        raise RuntimeError(f"S1.3 segment exact observation authority is missing: {missing}")
    supports = [available[key] for key in authority]
    support = (
        np.unique(np.concatenate(supports, axis=0), axis=0)
        if supports else np.empty((0, 2), dtype=np.int32)
    )
    rows = tuple({
        "pair_index": key[0], "component_id": key[1],
        "support_sha256": key[2], "support_sample_count": int(available[key].shape[0]),
    } for key in authority)
    return support, rows


def _verify_segment_support_asset(
    transaction_root: Path,
    segment_document: Mapping[str, object],
    evidence_assets: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Independently reconstruct one segment union from sealed observation assets."""

    authority = tuple(
        sorted(_segment_authority_key(row)
               for row in segment_document.get("observation_authority", []))
    )
    evidence_by_key = {
        (
            int(row["pair_index"]), int(row["component_id"]),
            str(row["support_sha256"]),
        ): row
        for row in evidence_assets
    }
    if len(evidence_by_key) != len(evidence_assets):
        raise RuntimeError("S1.3 sealed observation evidence authority is not unique")
    supports: list[np.ndarray] = []
    for key in authority:
        row = evidence_by_key.get(key)
        if row is None:
            raise RuntimeError(f"S1.3 segment evidence asset is missing: {key}")
        asset = transaction_root / str(row["asset"])
        if sha256_file(asset) != str(row["asset_sha256"]):
            raise RuntimeError("S1.3 observation evidence asset SHA mismatch")
        with np.load(asset, allow_pickle=False) as payload:
            support = np.asarray(payload["support_xy"], dtype=np.int32)
        if canonical_s13_support_sha256(support) != key[2]:
            raise RuntimeError("S1.3 observation exact support SHA mismatch")
        supports.append(support)
    expected = (
        np.unique(np.concatenate(supports, axis=0), axis=0)
        if supports else np.empty((0, 2), dtype=np.int32)
    )
    segment_asset = transaction_root / str(segment_document["asset"])
    if sha256_file(segment_asset) != str(segment_document["asset_sha256"]):
        raise RuntimeError("S1.3 segment support asset SHA mismatch")
    with np.load(segment_asset, allow_pickle=False) as payload:
        actual = np.asarray(payload["support_xy"], dtype=np.int32)
        asset_pairs = np.asarray(payload["authority_pair_indices"], dtype=np.int32)
        asset_components = np.asarray(
            payload["authority_component_ids"], dtype=np.int32
        )
        asset_support_shas = tuple(str(value) for value in payload["authority_support_sha256"])
    expected_pairs = np.asarray([key[0] for key in authority], dtype=np.int32)
    expected_components = np.asarray([key[1] for key in authority], dtype=np.int32)
    expected_shas = tuple(key[2] for key in authority)
    passed = bool(
        np.array_equal(actual, expected)
        and np.array_equal(asset_pairs, expected_pairs)
        and np.array_equal(asset_components, expected_components)
        and asset_support_shas == expected_shas
    )
    if not passed:
        raise RuntimeError("S1.3 segment support asset does not match exact authority")
    return {
        "passed": True,
        "authority_count": len(authority),
        "support_sample_count": int(actual.shape[0]),
    }


def _component_obligation_outcomes(
    component_audit: Mapping[str, object],
) -> tuple[dict[str, object], ...]:
    """Explain every frozen obligation outcome without changing its denominator."""

    accepted = {str(value) for value in component_audit.get("accepted_segment_ids", [])}
    deferred = {str(value) for value in component_audit.get("deferred_segment_ids", [])}
    by_authority: dict[tuple[int, int, str], list[tuple[str, str]]] = {}
    for segment in component_audit.get("segments", []):
        segment_id = str(segment["segment_id"])
        state = str(segment.get("state", "unknown"))
        for row in segment.get("observation_authority", []):
            by_authority.setdefault(_segment_authority_key(row), []).append(
                (segment_id, state)
            )
    outcomes: list[dict[str, object]] = []
    for obligation in component_audit.get("baseline_c2e_obligations", []):
        key = _segment_authority_key(obligation)
        candidates = sorted(by_authority.get(key, []))
        resolved = [
            segment_id for segment_id, state in candidates
            if segment_id in accepted and state == "resolved"
        ]
        improved = [
            segment_id for segment_id, state in candidates
            if segment_id in accepted and state == "improved_unresolved"
        ]
        deferred_segments = [
            segment_id for segment_id, _state in candidates if segment_id in deferred
        ]
        if resolved:
            outcome, reason = "resolved", None
        elif improved:
            outcome, reason = "improved_unresolved", "accepted_segment_not_resolved"
        elif deferred_segments:
            outcome, reason = "deferred", "segment_budget_deferred"
        elif obligation.get("evaluable") is not True:
            outcome, reason = "unevaluable", "baseline_obligation_unevaluable"
        elif obligation.get("severe") is not True:
            outcome, reason = "safe_anchor", "nonsevere_obligation_requires_no_correction"
        elif candidates:
            outcome, reason = "unresolved", "candidate_segment_not_applied_resolved"
        else:
            outcome, reason = "uncovered", "no_segment_contains_exact_authority"
        outcomes.append({
            "obligation_id": str(obligation["obligation_id"]),
            "pair_index": key[0], "component_id": key[1],
            "support_sha256": key[2],
            "severe": obligation.get("severe") is True,
            "evaluable": obligation.get("evaluable") is True,
            "outcome": outcome, "reason": reason,
            "candidate_segment_ids": [row[0] for row in candidates],
            "resolved_by_segment_ids": resolved,
        })
    return tuple(outcomes)


REPORT_SCHEMA = "gemini305-video-s13-output-first-report/v1"
GENERATION_SCHEMA = "gemini305-video-s13-generation/v1"
P1_COMPLETION_SCHEMA = "gemini305-video-s13-p1-vertical-completion/v2"
P2_COMPLETION_SCHEMA = "gemini305-video-s13-p2-completion/v3"
P3_COMPLETION_SCHEMA = "gemini305-video-s13-p3-visual-completion/v1"
P4_MANUAL_C2E_COMPLETION_SCHEMA = (
    "gemini305-video-s13-p4-manual-c2e-completion/v1"
)


def _generation_id(session: S13Session, config_sha: str) -> str:
    payload = f"{session.root}|{config_sha}|{time.time_ns()}".encode("utf-8")
    return time.strftime("%Y%m%dT%H%M%S") + "-" + hashlib.sha256(payload).hexdigest()[:12]


def _owner_overlay(image: np.ndarray, owner: np.ndarray) -> np.ndarray:
    overlay = image.copy()
    boundary = np.zeros(owner.shape, dtype=bool)
    boundary[:, 1:] = owner[:, 1:] != owner[:, :-1]
    overlay[boundary] = (0, 0, 255)
    return overlay


def _plain_edge(edge: object) -> dict[str, object]:
    value = asdict(edge)  # type: ignore[arg-type]
    value.pop("observations", None)
    return {key: item for key, item in value.items()}


def _optimizer_all_fail_probe(enabled: bool) -> dict[str, object]:
    """Exercise the post-seal failure boundary without implementing M3+."""

    failures: list[dict[str, str]] = []
    if enabled:
        for name in ("vertical_selection", "geometry", "seam_transaction"):
            try:
                raise RuntimeError("injected optimizer failure after immutable P0")
            except RuntimeError as exc:
                failures.append({"optimizer": name, "error": str(exc)})
    return {
        "requested": enabled,
        "attempt_count": len(failures),
        "failures": failures,
        "all_failed": enabled and len(failures) == 3,
        "p0_parent_preserved": True,
    }


def _m31_report_statistics(
    motion: tuple[object, ...], layout: S13M3Layout, schedule_plan: S13SchedulePlan
) -> dict[str, object]:
    methods = Counter()
    for method in layout.progress.placement_methods[1:]:
        if method.startswith("motion_hypothesis") or method.startswith("F0_"):
            methods["F0"] += 1
        else:
            matched = next((f"F{index}" for index in range(1, 8) if method.startswith(f"F{index}_")), None)
            methods[matched or "unclassified"] += 1
    hypothesis_histogram = Counter(len(getattr(edge, "motion_hypotheses")) for edge in motion)
    render_source_count = sum(
        not assignment.zero_width for schedule in schedule_plan.schedules for assignment in schedule.assignments
    )
    render_pair_count = sum(
        max(0, sum(not assignment.zero_width for assignment in schedule.assignments) - 1)
        for schedule in schedule_plan.schedules
    )
    finite_source_u = [
        float(value)
        for schedule in schedule_plan.schedules
        for value in schedule.column_source_u[np.isfinite(schedule.column_source_u)]
    ]
    switch_count = sum(
        left.support_signature_sha256 != right.support_signature_sha256
        for left, right in zip(layout.lineage[:-1], layout.lineage[1:])
        if left.support_signature_sha256 is not None and right.support_signature_sha256 is not None
    )
    return {
        "raw_rgb_motion_edge_count": len(motion),
        "raw_rgb_adjacent_edge_count": sum(getattr(edge, "step") == 1 for edge in motion),
        "render_source_count": render_source_count,
        "render_pair_count": render_pair_count,
        "lineage_switch_count": switch_count,
        "fallback_counts": {f"F{index}": int(methods.get(f"F{index}", 0)) for index in range(8)},
        "fallback_unclassified_count": int(methods.get("unclassified", 0)),
        "hypothesis_count_histogram": {
            str(count): int(hypothesis_histogram.get(count, 0)) for count in range(4)
        },
        "source_u_statistics": {
            "minimum": min(finite_source_u) if finite_source_u else None,
            "maximum": max(finite_source_u) if finite_source_u else None,
            "span": max(finite_source_u) - min(finite_source_u) if finite_source_u else None,
            "maximum_internal_owner_width_px": max(
                (assignment.width for schedule in schedule_plan.schedules for assignment in schedule.assignments[1:-1]),
                default=0,
            ),
            "configured_cap_px": schedule_plan.source_u_cap_px,
        },
    }


def _run_m4_vertical(
    generation: Path,
    root: Path,
    generation_id: str,
    schedule: object,
    calibration: object,
    image_loader: object,
    p0_image: np.ndarray,
) -> dict[str, object]:
    from .video_s12_schedule import S012Schedule

    if not isinstance(schedule, S012Schedule):
        raise TypeError("S1.3 M4 requires one immutable P0 schedule")
    # Keep atomic staging names short enough for deeply nested Windows audit
    # assets.  Twelve UUID hex characters still provide ample per-run
    # collision resistance and avoid crossing legacy MAX_PATH at P2.
    pending = generation / f".P1.{uuid.uuid4().hex[:12]}.pending"
    final = generation / "P1"
    if final.exists():
        raise FileExistsError(f"S1.3 M4 P1 already exists: {final}")
    pending.mkdir()
    try:
        estimated = estimate_s13_vertical(schedule, calibration, image_loader)  # type: ignore[arg-type]
        selection = select_s13_vertical_parent(
            schedule, calibration, image_loader, estimated, p0_image  # type: ignore[arg-type]
        )
        solution = selection.solution
        result = selection.result
        write_image(pending / "vertical_panorama_owner_only.png", result.image)
        write_image(pending / "vertical_panorama_owner_only.jpg", result.image)
        write_image(pending / "vertical_valid_mask.png", result.valid_mask.astype(np.uint8) * 255)
        write_image(pending / "vertical_owner_boundary_overlay.png", _owner_overlay(
            result.image, result.pixel_provenance["owner_frame_id"]
        ))
        write_npz(pending / "vertical_pixel_provenance.npz", result.pixel_provenance)
        solution_document = save_s13_vertical_solution(pending, solution)
        atomic_write_json(pending / "vertical_selection.json", {
            **dict(selection.audit),
            "diagnostic_only": True,
            "runtime_authority": False,
            "forward_stage_selected": "P1",
        })
        p0_completion = generation / "P0" / "P0_completion.json"
        p0_completion_sha = sha256_file(p0_completion)
        result_asset = "vertical_panorama_owner_only.png"
        seal_stage(
            pending,
            completion_name="P1_completion.json",
            schema=P1_COMPLETION_SCHEMA,
            metadata={
                "generation_id": generation_id,
                "stage": "P1",
                "hard_audit_passed": True,
                "parent_stage": "P0",
                "parent_completion_sha256": p0_completion_sha,
                "p0_parent": "../P0/P0_completion.json",
                "p0_parent_sha256": p0_completion_sha,
                "models": ["global_scalar_dy", "pair_local_row_residual"],
                "excluded_models": [
                    "translation", "rotation", "affine", "seam", "graphcut", "photometric",
                    "blend", "depth", "mesh",
                ],
                "formal_raw_rgb_remap_invocations": result.remap_invocations,
                "formal_raw_rgb_unique_sources": len(result.decoded_frame_ids),
                "selected_global_gain": solution.selected_gain,
                "selected_vertical_parent": selection.stage,
                "vertical_solution_json": "vertical_solution.json",
                "vertical_solution_json_sha256": sha256_file(pending / "vertical_solution.json"),
                "vertical_solution_npz": "vertical_solution.npz",
                "vertical_solution_npz_sha256": str(solution_document["npz_sha256"]),
                "result_asset": result_asset,
                "result_asset_sha256": sha256_file(pending / result_asset),
                "pixel_provenance_sha256": sha256_file(pending / "vertical_pixel_provenance.npz"),
            },
        )
        os.replace(pending, final)
        verify_stage(final, completion_name="P1_completion.json", schema=P1_COMPLETION_SCHEMA)
        if sha256_file(generation / "P0" / "P0_completion.json") != p0_completion_sha:
            raise ValueError("S1.3 M4 immutable P0 parent changed during P1 commit")
        pointer = update_current_latest(
            root, generation, stage="P1", completion_name="P1_completion.json",
            completion_schema=P1_COMPLETION_SCHEMA, result_asset=result_asset,
        )
        return {
            "state": "P1_sealed",
            "panorama": str(final / "vertical_panorama_owner_only.png"),
            "owner_overlay": str(final / "vertical_owner_boundary_overlay.png"),
            "pixel_provenance": str(final / "vertical_pixel_provenance.npz"),
            "completion": str(final / "P1_completion.json"),
            "current_latest": str(root / "current_latest.json"),
            "selected_global_gain": solution.selected_gain,
            "pair_count": len(solution.pairs),
            "applied_pair_count": sum(pair.status == "applied" for pair in solution.pairs),
            "formal_raw_rgb_remap_invocations": result.remap_invocations,
            "selected_vertical_parent": selection.stage,
            "latest_stage": pointer["stage"],
        }
    except BaseException:
        discard_staging(pending)
        raise


def _run_m5(
    generation: Path,
    root: Path,
    generation_id: str,
    schedule: object,
    calibration: object,
    image_loader: object,
    p0_image: np.ndarray,
    *,
    selected_hypothesis_ids: tuple[int, ...],
    placement_methods: tuple[str, ...],
    p2_completion_schema: str = P2_COMPLETION_SCHEMA,
    m6_eligible: bool = True,
    m6_blocked_reason: str | None = None,
    m61_evidence_config: S13PhotometricEvidenceConfig | None = None,
    m61_evidence_branch: str = "formal",
    m61_evidence_run_id: str = "",
    raw_rgb_sha256_by_frame: Mapping[int, str] | None = None,
    m51_r2_config: object | None = None,
    candidate_identity: Mapping[str, object] | None = None,
) -> dict[str, object]:
    from .session import CameraIntrinsics
    from .video_s12_schedule import S012Schedule

    if not isinstance(schedule, S012Schedule) or not isinstance(calibration, CameraIntrinsics):
        raise TypeError("S1.3 M5 requires one immutable P0 schedule and calibration")
    pending = generation / f".P2.{uuid.uuid4().hex[:12]}.pending"
    final = generation / "P2"
    if final.exists():
        raise FileExistsError(f"S1.3 M5 P2 already exists: {final}")
    pending.mkdir()
    r4_enabled = p2_completion_schema == S13_M51_R4_P2_COMPLETION_SCHEMA
    replay_schema = P2_REPLAY_V2_SCHEMA if r4_enabled else P2_REPLAY_SCHEMA
    p0_completion_path = generation / "P0" / "P0_completion.json"
    p1_completion_path = generation / "P1" / "P1_completion.json"
    p0_completion_sha = sha256_file(p0_completion_path)
    p1_completion_sha = sha256_file(p1_completion_path)
    try:
        started = time.perf_counter()
        verify_p0_completion(generation / "P0")
        p1_completion = verify_stage(
            generation / "P1", completion_name="P1_completion.json",
            schema=P1_COMPLETION_SCHEMA,
        )
        vertical = load_s13_vertical_solution(
            generation / "P1", expected_completion_sha256=p1_completion_sha,
        )
        parent_asset_name = str(p1_completion["result_asset"])
        parent_asset = generation / "P1" / parent_asset_name
        if sha256_file(parent_asset) != p1_completion.get("result_asset_sha256"):
            raise ValueError("S1.3 M5 sealed P1 result hash mismatch")
        p1_image = cv2.imread(str(parent_asset), cv2.IMREAD_COLOR)
        if p1_image is None or p1_image.shape != p0_image.shape:
            raise ValueError("S1.3 M5 sealed P1 result image is unreadable")
        provenance_name = "vertical_pixel_provenance.npz"
        provenance_path = generation / "P1" / provenance_name
        if sha256_file(provenance_path) != p1_completion.get("pixel_provenance_sha256"):
            raise ValueError("S1.3 M5 sealed P1 provenance hash mismatch")
        with np.load(provenance_path, allow_pickle=False) as archive:
            p1_provenance = {name: np.asarray(archive[name]).copy() for name in archive.files}
        write_image(pending / "selected_vertical_parent.png", p1_image)
        write_image(pending / "selected_vertical_parent.jpg", p1_image)
        write_npz(pending / "selected_vertical_parent_provenance.npz", p1_provenance)
        atomic_write_json(pending / "p1_parent_reference.json", {
            "schema": "gemini305-video-s13-p1-parent-reference/v1",
            "parent_stage": "P1",
            "completion": "../P1/P1_completion.json",
            "completion_sha256": p1_completion_sha,
            "result_asset": f"../P1/{parent_asset_name}",
            "result_asset_sha256": p1_completion["result_asset_sha256"],
        })
        m5 = run_s13_m5(
            schedule, calibration, image_loader, vertical, p1_image,  # type: ignore[arg-type]
            parent_stage_sha256=p1_completion_sha,
            parent_result_sha256=str(p1_completion["result_asset_sha256"]),
            p0_ancestor_completion_sha256=p0_completion_sha,
            selected_hypothesis_ids=selected_hypothesis_ids,
            placement_methods=placement_methods,
            m51_r2_config=m51_r2_config,
        )
        write_image(pending / "geometry_panorama_owner_only.png", m5.geometry_result.image)
        write_image(pending / "geometry_and_seam_panorama_owner_only.png", m5.final_result.image)
        write_image(pending / "geometry_and_seam_panorama_owner_only.jpg", m5.final_result.image)
        write_image(pending / "seam_overlay.png", m5.seam_overlay)
        write_image(pending / "p2_valid_mask.png", m5.final_result.valid_mask.astype(np.uint8) * 255)
        write_npz(pending / "p2_pixel_provenance.npz", m5.final_result.pixel_provenance)
        if r4_enabled:
            validation_root = pending / "validation/box_component_chain"
            height, width = m5.final_result.image.shape[:2]
            x0, x1 = max(0, min(width, 1470)), max(0, min(width, 1610))
            y0, y1 = max(0, min(height, 280)), max(0, min(height, 350))
            if x1 <= x0 or y1 <= y0:
                x0, x1, y0, y1 = 0, width, 0, height
            validation_raw_cache: dict[int, np.ndarray] = {}

            def validation_image_loader(frame_id: int) -> np.ndarray:
                if frame_id not in validation_raw_cache:
                    validation_raw_cache[frame_id] = np.asarray(image_loader(frame_id))
                return validation_raw_cache[frame_id]

            component_doc = dict(m5.component_chain_audit or {})
            field_ids = {
                str(key): int(value)
                for key, value in dict(component_doc.get("field_id_table", {})).items()
            }
            obligation_by_authority = {
                (
                    int(row["pair_index"]), int(row["component_id"]),
                    str(row["support_sha256"]),
                ): row
                for row in component_doc.get("baseline_c2e_obligations", [])
            }
            box_observation_seeds: list[_ExternalBoxObservationSeed] = []
            for pair in m5.pairs:
                for observation, evidence in pair.component_evidence:
                    support = np.asarray(evidence.support_xy, dtype=np.int32)
                    support_in_roi = (
                        (support[:, 0] >= x0) & (support[:, 0] < x1)
                        & (support[:, 1] >= y0) & (support[:, 1] < y1)
                    )
                    if not np.any(support_in_roi):
                        continue
                    authority = (
                        int(observation.pair_index), int(observation.component_id),
                        str(observation.mask_sha256),
                    )
                    obligation = obligation_by_authority.get(authority)
                    if obligation is None:
                        raise RuntimeError(
                            "S1.3 box exact support has no frozen obligation authority"
                        )
                    box_observation_seeds.append(_ExternalBoxObservationSeed(
                        pair_index=authority[0],
                        component_id=authority[1],
                        support_sha256=authority[2],
                        support_xy=support,
                        obligation_id=str(obligation["obligation_id"]),
                        severe=obligation.get("severe") is True,
                        evaluable=obligation.get("evaluable") is True,
                    ))
            box_tracks = _external_box_physical_tracks(
                box_observation_seeds,
                minimum_y_overlap_fraction=float(
                    getattr(m51_r2_config, "minimum_component_y_overlap_fraction", 0.5)
                ),
                maximum_predicted_y_difference_px=float(
                    getattr(m51_r2_config, "maximum_predicted_y_disagreement_px", 6.0)
                ),
                maximum_normal_difference_degrees=float(
                    getattr(m51_r2_config, "maximum_component_normal_difference_degrees", 10.0)
                ),
            )
            current_roi, current_valid = render_s13_component_roi_from_raw(
                schedule, calibration, validation_image_loader, vertical, m5.pairs,
                (x0, y0, x1, y1), registry=None,
                preserve_vertical_parent=True,
            )
            candidate_roi, candidate_valid = render_s13_component_roi_from_raw(
                schedule, calibration, validation_image_loader, vertical, m5.pairs,
                (x0, y0, x1, y1), registry=m5.source_correction_registry,
                field_ids=field_ids,
                preserve_vertical_parent=True,
            )
            current_metrics, current_overlay = _seeded_edge_trace_metrics(
                current_roi, current_valid, box_tracks, roi_origin_xy=(x0, y0)
            )
            candidate_metrics, candidate_overlay = _seeded_edge_trace_metrics(
                candidate_roi, candidate_valid, box_tracks, roi_origin_xy=(x0, y0)
            )
            write_image(validation_root / "current_roi.png", current_roi)
            write_image(validation_root / "candidate_roi.png", candidate_roi)
            write_image(
                validation_root / "current_vs_candidate.png",
                np.concatenate((current_roi, candidate_roi), axis=1),
            )
            write_image(
                validation_root / "edge_trace_overlay.png",
                np.concatenate((current_overlay, candidate_overlay), axis=1),
            )
            mask_overlay = candidate_roi.copy()
            field = m5.final_result.pixel_provenance[
                "component_correction_field_id"
            ][y0:y1, x0:x1]
            corrected = field >= 0
            mask_overlay[corrected] = (
                0.5 * mask_overlay[corrected].astype(np.float32)
                + np.asarray((0, 127, 0), dtype=np.float32)
            ).astype(np.uint8)
            write_image(validation_root / "component_masks_overlay.png", mask_overlay)
            pair_rows = [
                row for row in component_doc.get("forward_reverse_hypotheses", [])
                if 68 <= int(row.get("pair_index", -1)) <= 78
            ]
            relevant_obligations = [
                row for row in component_doc.get("baseline_c2e_obligations", [])
                if str(row["obligation_id"]) in {
                    seed.obligation_id for seed in box_observation_seeds
                }
            ]
            relevant_obligations.sort(key=lambda row: (
                int(row["pair_index"]), int(row["component_id"]),
                str(row["support_sha256"]),
            ))
            coverage_summary = _box_target_coverage_summary(
                component_doc, relevant_obligations
            )
            obligations_covered = bool(coverage_summary["repair_obligations_covered"])
            baseline_union = int(current_metrics["break_double_edge_union_length_px"])
            candidate_union = int(candidate_metrics["break_double_edge_union_length_px"])
            break_gate = (
                candidate_union <= math.floor(0.5 * baseline_union)
                if baseline_union >= 4 else candidate_union <= baseline_union
            )
            p95 = candidate_metrics["edge_step_p95_px"]
            maximum_step = candidate_metrics["maximum_local_step_px"]
            coverage_gate = float(candidate_metrics["coverage_fraction"]) >= float(
                getattr(m51_r2_config, "minimum_evaluable_transition_fraction", 0.5)
            )
            box_target_repair_complete = bool(
                obligations_covered
                and coverage_gate
                and isinstance(p95, (int, float)) and float(p95) <= 1.0
                and isinstance(maximum_step, (int, float))
                and float(maximum_step) <= 1.5
                and break_gate
                and int(candidate_metrics["backtrack_count"])
                <= int(current_metrics["backtrack_count"])
            )
            failed_box_gates = [
                name for name, passed in {
                    "target_obligations_covered": obligations_covered,
                    "edge_coverage": coverage_gate,
                    "edge_step_p95": isinstance(p95, (int, float)) and float(p95) <= 1.0,
                    "maximum_local_step": isinstance(maximum_step, (int, float))
                    and float(maximum_step) <= 1.5,
                    "break_double_edge": break_gate,
                    "no_new_backtrack": int(candidate_metrics["backtrack_count"])
                    <= int(current_metrics["backtrack_count"]),
                }.items() if not passed
            ]
            atomic_write_json(validation_root / "pair_0068_0078_metrics.json", {
                "schema": "gemini305-video-s13-box-pair-metrics/v1",
                "roi_xyxy": [x0, y0, x1, y1],
                "pairs": pair_rows,
                "baseline_edge_trace": current_metrics,
                "candidate_edge_trace": candidate_metrics,
                "physical_edge_tracks": [
                    {
                        "track_id": row["track_id"],
                        "observation_authority": row["observation_authority"],
                        "obligation_ids": row["obligation_ids"],
                        "support_sample_count": int(
                            np.asarray(row["support_xy"]).shape[0]
                        ),
                    }
                    for row in box_tracks
                ],
                **coverage_summary,
                "box_target_repair_complete": box_target_repair_complete,
                "failed_gates": failed_box_gates,
                "reason": None if box_target_repair_complete
                else "external_roi_quality_gate_not_complete",
            })
            atomic_write_json(
                validation_root / "component_chain_audit.json", component_doc
            )
            atomic_write_json(validation_root / "source_offsets.json", {
                "schema": "gemini305-video-s13-box-source-offsets/v1",
                "segments": component_doc.get("segments", []),
            })
            atomic_write_json(validation_root / "forward_reverse_hypotheses.json", {
                "schema": "gemini305-video-s13-box-forward-reverse-hypotheses/v1",
                "hypotheses": pair_rows,
            })
        component_manifest_sha: str | None = None
        source_correction_manifest_sha: str | None = None
        source_map_manifest_sha: str | None = None
        if r4_enabled:
            component_audit = dict(m5.component_chain_audit or {})
            (pending / "component_chain_transactions").mkdir(parents=True, exist_ok=True)
            (pending / "source_corrections").mkdir(parents=True, exist_ok=True)
            (pending / "source_maps").mkdir(parents=True, exist_ok=True)
            evidence_assets: list[dict[str, object]] = []
            for pair in m5.pairs:
                for observation, evidence in pair.component_evidence:
                    asset = (
                        f"o_{observation.pair_index:04d}_"
                        f"{observation.component_id:04d}.npz"
                    )
                    write_npz(pending / "component_chain_transactions" / asset, {
                        "support_xy": evidence.support_xy,
                        "forward_scores": evidence.forward_scores,
                        "reverse_scores": evidence.reverse_scores,
                        "forward_correlations": evidence.forward_correlations,
                        "reverse_correlations": evidence.reverse_correlations,
                        "forward_support_counts": evidence.forward_support_counts,
                        "reverse_support_counts": evidence.reverse_support_counts,
                    })
                    evidence_assets.append({
                        "pair_index": observation.pair_index,
                        "component_id": observation.component_id,
                        "asset": asset,
                        "asset_sha256": sha256_file(
                            pending / "component_chain_transactions" / asset
                        ),
                        "support_sha256": observation.mask_sha256,
                    })
            segment_assets: list[dict[str, object]] = []
            hypothesis_by_authority = {
                (
                    int(row["pair_index"]), int(row["component_id"]),
                    str(row["support_sha256"]),
                ): row
                for row in component_audit.get("forward_reverse_hypotheses", [])
            }
            chain_by_id = {
                str(row["chain_id"]): row for row in component_audit.get("chains", [])
            }
            accepted_segment_ids = {
                str(value) for value in component_audit.get("accepted_segment_ids", [])
            }
            rejected_segment_ids = {
                str(value) for value in component_audit.get("rejected_segment_ids", [])
            }
            deferred_segment_ids = {
                str(value) for value in component_audit.get("deferred_segment_ids", [])
            }
            for segment in component_audit.get("segments", []):
                segment_id = str(segment["segment_id"])
                short_id = segment_id.rsplit("-", 1)[-1]
                json_name = f"s_{short_id}.json"
                npz_name = f"s_{short_id}.npz"
                observation_authority = list(segment.get("observation_authority", []))
                parent_chain_id = str(segment.get("parent_chain_id", ""))
                if not observation_authority and parent_chain_id in chain_by_id:
                    chain = chain_by_id[parent_chain_id]
                    component_by_pair = dict(zip(
                        chain.get("pair_indices", []), chain.get("component_ids", []),
                        strict=True,
                    ))
                    for pair_index in segment.get("pair_indices", []):
                        component_id = component_by_pair.get(pair_index)
                        matches = [
                            row for key, row in hypothesis_by_authority.items()
                            if key[:2] == (int(pair_index), int(component_id))
                        ]
                        if len(matches) != 1:
                            raise RuntimeError(
                                "S1.3 segment chain cannot resolve exact observation authority"
                            )
                        observation_authority.append({
                            "pair_index": int(pair_index),
                            "component_id": int(component_id),
                            "support_sha256": str(matches[0]["support_sha256"]),
                        })
                support, sealed_authority = _segment_authority_support(
                    m5.pairs, observation_authority
                )
                write_npz(pending / "component_chain_transactions" / npz_name, {
                    "support_xy": support,
                    "pair_indices": np.asarray(segment.get("pair_indices", []), np.int32),
                    "source_indices": np.asarray(segment.get("source_indices", []), np.int32),
                    "authority_pair_indices": np.asarray(
                        [row["pair_index"] for row in sealed_authority], np.int32
                    ),
                    "authority_component_ids": np.asarray(
                        [row["component_id"] for row in sealed_authority], np.int32
                    ),
                    "authority_support_sha256": np.asarray(
                        [row["support_sha256"] for row in sealed_authority]
                    ),
                })
                hypothesis_summary = []
                for row in sealed_authority:
                    hypothesis = hypothesis_by_authority.get(_segment_authority_key(row))
                    if hypothesis is None:
                        raise RuntimeError("S1.3 segment authority has no 13-hypothesis audit")
                    hypothesis_summary.append({
                        "pair_index": row["pair_index"],
                        "component_id": row["component_id"],
                        "support_sha256": row["support_sha256"],
                        "evidence_state": hypothesis.get("evidence_state"),
                        "forward_best_lag_px": hypothesis.get("forward_best_lag_px"),
                        "reverse_best_lag_px": hypothesis.get("reverse_best_lag_px"),
                        "forward_scores": hypothesis.get("forward_scores", []),
                        "reverse_scores": hypothesis.get("reverse_scores", []),
                        "forward_support_counts": hypothesis.get(
                            "forward_support_counts", []
                        ),
                        "reverse_support_counts": hypothesis.get(
                            "reverse_support_counts", []
                        ),
                    })
                dependency_rows = [
                    row for row in component_audit.get("dependency_groups", [])
                    if segment_id in row.get("segment_ids", [])
                ]
                affected_pair_indices = sorted({
                    int(pair_index)
                    for source_index in segment.get("source_indices", [])
                    for pair_index in (int(source_index) - 1, int(source_index))
                    if 0 <= pair_index < len(m5.pairs)
                } | {
                    int(pair_index) for row in dependency_rows
                    for pair_index in row.get("affected_pair_indices", [])
                })
                if segment_id in accepted_segment_ids:
                    selection_outcome = "conflict_winner_applied"
                elif segment_id in rejected_segment_ids:
                    selection_outcome = "conflict_or_quality_loser_rejected"
                elif segment_id in deferred_segment_ids:
                    selection_outcome = "budget_deferred"
                else:
                    selection_outcome = str(segment.get("state", "not_selected"))
                competing_winners = sorted(
                    accepted_segment_ids & {
                        str(value) for row in dependency_rows
                        for value in row.get("segment_ids", [])
                    } - {segment_id}
                )
                audit = dict(segment.get("audit", {}))
                segment_document = {
                    "schema": "gemini305-video-s13-component-segment-transaction/v1",
                    "config_sha256": str(candidate_identity["config_sha256"]),
                    **dict(segment),
                    "parent_segment_id": segment.get("parent_segment_id"),
                    "root_segment_id": segment.get("root_segment_id", segment_id),
                    "parent_chain_id": segment.get("parent_chain_id"),
                    "component_ids": [
                        int(row["component_id"]) for row in sealed_authority
                    ],
                    "observation_authority": list(sealed_authority),
                    "cut_reasons": {
                        "left": segment.get("left_cut_reason"),
                        "right": segment.get("right_cut_reason"),
                        "terminal": segment.get("reason"),
                    },
                    "hypothesis_13_summary": hypothesis_summary,
                    "affected_pair_closure": affected_pair_indices,
                    "hard_gates": {
                        "failures": list(audit.get("hard_gate_failures", [])),
                        "candidate_map_safety": audit.get("candidate_map_safety"),
                        "dependency_groups": dependency_rows,
                    },
                    "conflict_resolution": {
                        "outcome": selection_outcome,
                        "winner_segment_ids": (
                            [segment_id] if segment_id in accepted_segment_ids
                            else competing_winners
                        ),
                        "loser_segment_ids": (
                            [segment_id] if segment_id in rejected_segment_ids else []
                        ),
                    },
                    "asset": npz_name,
                    "asset_sha256": sha256_file(
                        pending / "component_chain_transactions" / npz_name
                    ),
                }
                segment_document["decision_payload_stable_sha256"] = (
                    segment_decision_stable_sha256(segment_document)
                )
                atomic_write_json(
                    pending / "component_chain_transactions" / json_name,
                    segment_document,
                )
                support_verification = _verify_segment_support_asset(
                    pending / "component_chain_transactions",
                    segment_document,
                    evidence_assets,
                )
                segment_assets.append({
                    "segment_id": segment_id,
                    "json": json_name,
                    "json_sha256": sha256_file(
                        pending / "component_chain_transactions" / json_name
                    ),
                    "npz": npz_name,
                    "npz_sha256": segment_document["asset_sha256"],
                    "support_verification": support_verification,
                    "support_authority_sha256": hashlib.sha256(
                        json.dumps(
                            [row["support_sha256"] for row in sealed_authority],
                            separators=(",", ":"),
                        ).encode("utf-8")
                    ).hexdigest(),
                })
            obligation_outcomes = _component_obligation_outcomes(component_audit)
            stable_payload = {
                "application_state": component_audit.get("application_state", "none"),
                "baseline_c2e_obligations": component_audit.get(
                    "baseline_c2e_obligations", []
                ),
                "chains": component_audit.get("chains", []),
                "segments": component_audit.get("segments", []),
                "dependency_groups": component_audit.get("dependency_groups", []),
                "component_match_matrices": component_audit.get(
                    "component_match_matrices", []
                ),
                "exact_evidence_propagation": component_audit.get(
                    "exact_evidence_propagation", {}
                ),
                "roi_candidate_pixels": component_audit.get("roi_candidate_pixels", 0),
                "roi_preview_count": component_audit.get("roi_preview_count", 0),
                "evidence_assets": evidence_assets,
                "segment_assets": segment_assets,
                "obligation_outcomes": obligation_outcomes,
                "split_lineage": component_audit.get("split_lineage", []),
            }
            component_manifest = {
                "schema": "gemini305-video-s13-component-chain-transactions/v1",
                "application_state": component_audit.get("application_state", "none"),
                "repair_complete": component_audit.get("repair_complete", False),
                "baseline_c2e_obligations": component_audit.get(
                    "baseline_c2e_obligations", []
                ),
                "chains": component_audit.get("chains", []),
                "segments": component_audit.get("segments", []),
                "dependency_groups": component_audit.get("dependency_groups", []),
                "component_match_matrices": component_audit.get(
                    "component_match_matrices", []
                ),
                "exact_evidence_propagation": component_audit.get(
                    "exact_evidence_propagation", {}
                ),
                "observation_count": component_audit.get("observation_count", 0),
                "chain_count": component_audit.get("chain_count", 0),
                "raw_partition_chain_count": component_audit.get(
                    "raw_partition_chain_count", 0
                ),
                "segment_count": component_audit.get("segment_count", 0),
                "roi_candidate_pixels": component_audit.get("roi_candidate_pixels", 0),
                "roi_preview_count": component_audit.get("roi_preview_count", 0),
                "forward_reverse_hypotheses": component_audit.get(
                    "forward_reverse_hypotheses", []
                ),
                "evidence_assets": evidence_assets,
                "segment_assets": segment_assets,
                "obligation_outcomes": obligation_outcomes,
                "split_lineage": component_audit.get("split_lineage", []),
                "segment_transaction_assets": [
                    {
                        "segment_id": row["segment_id"],
                        "asset": f"component_chain_transactions/{row['json']}",
                        "sha256": row["json_sha256"],
                        "evidence_asset": f"component_chain_transactions/{row['npz']}",
                        "evidence_sha256": row["npz_sha256"],
                    }
                    for row in segment_assets
                ],
                "accepted_segment_ids": component_audit.get("accepted_segment_ids", []),
                "rejected_segment_ids": component_audit.get("rejected_segment_ids", []),
                "deferred_segment_ids": component_audit.get("deferred_segment_ids", []),
                "unresolved_regions": component_audit.get("segments", []),
                "decision_payload_stable_sha256": "",
            }
            if stable_payload != component_decision_stable_payload(component_manifest):
                raise RuntimeError("component stable authority field selection disagrees")
            component_manifest["decision_payload_stable_sha256"] = (
                component_decision_stable_sha256(component_manifest)
            )
            atomic_write_json(
                pending / "component_chain_transactions/manifest.json",
                component_manifest,
            )
            component_manifest_sha = sha256_file(
                pending / "component_chain_transactions/manifest.json"
            )
            correction_sources: list[dict[str, object]] = []
            correction_asset_by_source: dict[int, dict[str, object]] = {}
            support_sha_by_segment = {
                str(row["segment_id"]): str(row["support_authority_sha256"])
                for row in segment_assets
            }
            patch_set = m5.component_patch_set
            if patch_set is not None:
                for source_index, corrections in patch_set.corrections_by_source.items():
                    asset = f"source_{source_index:04d}.npz"
                    correction_arrays: dict[str, np.ndarray] = {
                        "segment_ids": np.asarray([row.segment_id for row in corrections]),
                        "field_id": np.asarray([
                            component_audit.get("field_id_table", {}).get(
                                row.segment_id, -1
                            )
                            for row in corrections
                        ], dtype=np.int32),
                        "domains_xyxy": np.asarray([
                            (row.x0, row.y0, row.x1, row.y1) for row in corrections
                        ], dtype=np.int32),
                        "correction_sha256": np.asarray([
                            row.correction_sha256 for row in corrections
                        ]),
                        "support_authority_sha256": np.asarray([
                            support_sha_by_segment[row.segment_id]
                            for row in corrections
                        ]),
                    }
                    for index, correction in enumerate(corrections):
                        correction_arrays[f"delta_u_{index:04d}"] = correction.delta_u
                        correction_arrays[f"delta_v_{index:04d}"] = correction.delta_v
                        correction_arrays[f"weight_{index:04d}"] = correction.weight
                    write_npz(pending / "source_corrections" / asset, correction_arrays)
                    correction_asset_by_source[source_index] = {
                        "source_index": source_index,
                        "contributors": [row.segment_id for row in corrections],
                        "correction_asset": f"source_corrections/{asset}",
                        "correction_asset_sha256": sha256_file(
                            pending / "source_corrections" / asset
                        ),
                    }
            oracle_by_source = {
                oracle.source_index: oracle for oracle in m5.source_map_oracles
            }
            base_oracle_by_source = {
                oracle.source_index: oracle
                for oracle in m5.base_source_map_oracles
            }
            for source_index, oracle in sorted(oracle_by_source.items()):
                base_oracle = base_oracle_by_source.get(source_index)
                if base_oracle is None:
                    raise RuntimeError("S1.3 final oracle lacks pre-C2E base authority")
                correction_sources.append({
                    "source_index": source_index,
                    "contributors": correction_asset_by_source.get(
                        source_index, {}
                    ).get("contributors", []),
                    "source_map_oracle_sha256": oracle.oracle_sha256,
                    "base_source_map_oracle_sha256": base_oracle.oracle_sha256,
                    **correction_asset_by_source.get(source_index, {}),
                })
            segment_asset_by_id = {
                str(row["segment_id"]): row for row in segment_assets
            }
            correction_sha_by_segment = {
                str(segment_id): str(source["correction_asset_sha256"])
                for source in correction_sources
                if "correction_asset_sha256" in source
                for segment_id in source.get("contributors", [])
            }
            field_table = [
                {
                    "field_id": int(field_id),
                    "segment_id": str(segment_id),
                    "segment_transaction_sha256": segment_asset_by_id[
                        str(segment_id)
                    ]["json_sha256"],
                    "support_sha256": support_sha_by_segment[str(segment_id)],
                    "source_correction_asset_sha256": correction_sha_by_segment[
                        str(segment_id)
                    ],
                    "correction_rows": [
                        {
                            "source_index": int(source_index),
                            "row_index": int(row_index),
                            "domain_xyxy": [
                                int(correction.x0), int(correction.y0),
                                int(correction.x1), int(correction.y1),
                            ],
                            "correction_sha256": correction.correction_sha256,
                            "support_sha256": support_sha_by_segment[
                                str(segment_id)
                            ],
                            "source_correction_asset_sha256": (
                                correction_asset_by_source[source_index][
                                    "correction_asset_sha256"
                                ]
                            ),
                        }
                        for source_index, corrections in sorted(
                            patch_set.corrections_by_source.items()
                            if patch_set is not None else ()
                        )
                        for row_index, correction in enumerate(corrections)
                        if correction.segment_id == str(segment_id)
                    ],
                }
                for segment_id, field_id in sorted(
                    dict(component_audit.get("field_id_table", {})).items(),
                    key=lambda row: row[1],
                )
            ]
            source_correction_manifest = {
                "schema": "gemini305-video-s13-source-corrections/v1",
                "parent_component_transaction_manifest_sha256": component_manifest_sha,
                "application_state": component_audit.get("application_state", "none"),
                "contributors": component_audit.get("accepted_segment_ids", []),
                "field_id_table": field_table,
                "sources": correction_sources,
            }
            atomic_write_json(
                pending / "source_corrections/manifest.json",
                source_correction_manifest,
            )
            source_correction_manifest_sha = sha256_file(
                pending / "source_corrections/manifest.json"
            )
            source_rows: list[dict[str, object]] = []
            base_oracle_by_source = {
                oracle.source_index: oracle
                for oracle in m5.base_source_map_oracles
            }
            for oracle in m5.source_map_oracles:
                base_oracle = base_oracle_by_source.get(oracle.source_index)
                if base_oracle is None:
                    raise RuntimeError("S1.3 source-map oracle has no pre-C2E base authority")
                asset = f"source_{oracle.source_index:04d}.npz"
                base_asset = f"base_source_{oracle.source_index:04d}.npz"
                write_npz(pending / "source_maps" / asset, {
                    "source_index": np.asarray(oracle.source_index, dtype=np.int32),
                    "domain_xyxy": np.asarray(oracle.domain_xyxy, dtype=np.int32),
                    "u": oracle.u,
                    "v": oracle.v,
                    "valid": oracle.valid,
                    "field_id": oracle.field_id,
                })
                write_npz(pending / "source_maps" / base_asset, {
                    "source_index": np.asarray(base_oracle.source_index, dtype=np.int32),
                    "domain_xyxy": np.asarray(base_oracle.domain_xyxy, dtype=np.int32),
                    "u": base_oracle.u,
                    "v": base_oracle.v,
                    "valid": base_oracle.valid,
                    "field_id": base_oracle.field_id,
                })
                source_rows.append({
                    "source_index": oracle.source_index,
                    "domain_xyxy": list(oracle.domain_xyxy),
                    "asset": f"source_maps/{asset}",
                    "asset_sha256": sha256_file(pending / "source_maps" / asset),
                    "oracle_sha256": oracle.oracle_sha256,
                    "base_asset": f"source_maps/{base_asset}",
                    "base_asset_sha256": sha256_file(
                        pending / "source_maps" / base_asset
                    ),
                    "base_oracle_sha256": base_oracle.oracle_sha256,
                    "raw_source_size": [
                        int(calibration.width), int(calibration.height)
                    ],
                })
            atomic_write_json(pending / "source_maps/manifest.json", {
                "schema": "gemini305-video-s13-source-map-oracles/v1",
                "parent_source_correction_manifest_sha256": source_correction_manifest_sha,
                "source_count": len(source_rows),
                "sources": source_rows,
            })
            source_map_manifest_sha = sha256_file(pending / "source_maps/manifest.json")
        transaction_rows = [dict(pair.transaction) for pair in m5.pairs]
        if r4_enabled:
            def segment_affects_pair(
                segment: Mapping[str, object], pair_index: int
            ) -> bool:
                return pair_index in {
                    affected_pair
                    for source in segment.get("source_indices", [])
                    for affected_pair in (int(source) - 1, int(source))
                    if 0 <= affected_pair < len(m5.pairs)
                }

            for index, row in enumerate(transaction_rows):
                row["component_chain_c2e"] = {
                    "schema": "gemini305-video-s13-component-chain-c2e-pair-ref/v1",
                    "application_state": component_audit.get("application_state", "none"),
                    "reason": (
                        "no_actionable_component_segment"
                        if component_audit.get("application_state", "none") == "none"
                        else None
                    ),
                    "component_transaction_manifest_sha256": component_manifest_sha,
                    "source_correction_manifest_sha256": source_correction_manifest_sha,
                    "source_map_oracle_manifest_sha256": source_map_manifest_sha,
                    "refs": [
                        {
                            "segment_id": segment["segment_id"],
                            "segment_transaction_sha256": segment_asset_by_id[
                                str(segment["segment_id"])
                            ]["json_sha256"],
                            "role": "rescued" if segment.get("state") in {
                                "resolved", "improved_unresolved"
                            } else "cut",
                            "state": segment.get("state", "rejected"),
                            "field_id": (
                                component_audit.get("field_id_table", {}).get(
                                    str(segment["segment_id"]), -1
                                )
                            ),
                            "affected_source_indices": segment.get("source_indices", []),
                        }
                        for segment in component_audit.get("segments", [])
                        if segment_affects_pair(
                            segment, int(row["transaction_id"].split("-")[-1])
                        )
                    ],
                }
                transaction_rows[index] = _finalize_v5_transaction(row)
        atomic_write_json(pending / "pair_transactions.json", {
            "schema": (
                "gemini305-video-s13-m5-pair-transactions/v4"
                if r4_enabled
                else "gemini305-video-s13-m5-pair-transactions/v3"
                if p2_completion_schema == S13_M51_R3_P2_COMPLETION_SCHEMA
                else "gemini305-video-s13-m5-pair-transactions/v2"
                if p2_completion_schema == S13_M51_R2_P2_COMPLETION_SCHEMA
                else "gemini305-video-s13-m5-pair-transactions/v1"
            ),
            "all_pairs_reported": len(transaction_rows) == len(schedule.assignments) - 1,
            "pairs": transaction_rows,
            **({
                "component_transaction_manifest_sha256": component_manifest_sha,
                "source_correction_manifest_sha256": source_correction_manifest_sha,
                "source_map_oracle_manifest_sha256": source_map_manifest_sha,
            } if r4_enabled else {}),
        })
        atomic_write_json(pending / "p2_selection.json", dict(m5.selection_audit))
        hard_audit_document = dict(m5.hard_audit)
        if r4_enabled:
            hard_audit_document["component_chain_c2e"] = dict(
                m5.component_chain_audit or {}
            )
        atomic_write_json(pending / "hard_audit.json", hard_audit_document)
        atomic_write_json(
            pending / "diagnostic_quality_report.json", dict(m5.diagnostic_quality)
        )
        for pair_index, pair in enumerate(m5.pairs):
            atomic_write_json(
                pending / "pair_transactions" / f"pair_{int(pair.transaction['transaction_id'].split('-')[-1]):04d}.json",
                transaction_rows[pair_index],
            )
        aggregate_path = pending / "pair_transactions.json"
        aggregate_document = json.loads(aggregate_path.read_text(encoding="utf-8"))
        aggregate_document["pair_transaction_assets"] = [
            {
                "pair_index": pair_index,
                "asset": f"pair_transactions/pair_{pair_index:04d}.json",
                "sha256": sha256_file(
                    pending / "pair_transactions" / f"pair_{pair_index:04d}.json"
                ),
            }
            for pair_index in range(len(m5.pairs))
        ]
        atomic_write_json(aggregate_path, aggregate_document)
        replay_manifest_rows: list[dict[str, object]] = []
        replay_pairs = []
        oracle_by_source = {
            oracle.source_index: oracle for oracle in m5.source_map_oracles
        }
        for replay_pair in m5.replay_pairs:
            transaction_asset = (
                pending / "pair_transactions" / f"pair_{replay_pair.pair_index:04d}.json"
            )
            transaction_sha = sha256_file(transaction_asset)
            bound_pair = replace(
                replay_pair, parent_pair_transaction_sha256=transaction_sha
            )
            replay_asset = Path("pair_replay") / f"pair_{bound_pair.pair_index:04d}.npz"
            write_npz(pending / replay_asset, replay_pair_arrays(bound_pair))
            replay_row = {
                "pair_index": bound_pair.pair_index,
                "asset": replay_asset.as_posix(),
                "left_source_index": bound_pair.left_source_index,
                "right_source_index": bound_pair.right_source_index,
                "left_frame_id": bound_pair.left_frame_id,
                "right_frame_id": bound_pair.right_frame_id,
                "parent_pair_transaction": (
                    f"pair_transactions/pair_{bound_pair.pair_index:04d}.json"
                ),
                "parent_pair_transaction_sha256": transaction_sha,
            }
            if r4_enabled:
                def oracle_slice_sha(source_index: int) -> tuple[str, str]:
                    oracle = oracle_by_source[source_index]
                    return oracle.oracle_sha256, canonical_source_map_slice_sha256(
                        oracle,
                        (
                            bound_pair.corridor_x0,
                            0,
                            bound_pair.corridor_x1,
                            schedule.canvas_height,
                        ),
                    )

                left_oracle_sha, left_slice_sha = oracle_slice_sha(
                    bound_pair.left_source_index
                )
                right_oracle_sha, right_slice_sha = oracle_slice_sha(
                    bound_pair.right_source_index
                )
                replay_row.update({
                    "component_transaction_manifest_sha256": component_manifest_sha,
                    "source_correction_manifest_sha256": source_correction_manifest_sha,
                    "source_map_oracle_manifest_sha256": source_map_manifest_sha,
                    "left_source_map_oracle_sha256": left_oracle_sha,
                    "right_source_map_oracle_sha256": right_oracle_sha,
                    "left_source_map_slice_sha256": left_slice_sha,
                    "right_source_map_slice_sha256": right_slice_sha,
                    "relevant_segment_ids": [
                        row["segment_id"] for row in component_audit.get("segments", [])
                        if (
                            str(row["segment_id"]) in accepted_segment_ids
                            and segment_affects_pair(row, bound_pair.pair_index)
                        )
                    ],
                    "relevant_segment_transaction_sha256": [
                        segment_asset_by_id[str(row["segment_id"])]["json_sha256"]
                        for row in component_audit.get("segments", [])
                        if (
                            str(row["segment_id"]) in accepted_segment_ids
                            and segment_affects_pair(row, bound_pair.pair_index)
                        )
                    ],
                })
            replay_manifest_rows.append(replay_row)
            replay_pairs.append(bound_pair)
        write_npz(pending / "p2_seams.npz", {
            "seams_x_by_row": np.stack(
                [pair.seam_x_by_row for pair in replay_pairs]
            ).astype(np.int32),
            "base_boundaries_x": np.asarray(schedule.boundaries[1:-1], np.int32),
        })
        atomic_write_json(pending / "p2_replay_manifest.json", {
            "schema": replay_schema,
            "pair_count": len(replay_manifest_rows),
            "source_count": len(schedule.assignments),
            "maximum_secondary_corridor_width_px": 8,
            "reestimation_performed": False,
            "pairs": replay_manifest_rows,
            **({
                "component_transaction_manifest_sha256": component_manifest_sha,
                "source_correction_manifest_sha256": source_correction_manifest_sha,
                "source_map_oracle_manifest_sha256": source_map_manifest_sha,
                "aggregate_pair_transaction_manifest_sha256": sha256_file(
                    pending / "pair_transactions.json"
                ),
            } if r4_enabled else {}),
        })
        photometric_replay: Mapping[str, object] | None = None
        if p2_completion_schema == P2_V4_COMPLETION_SCHEMA:
            if m61_evidence_config is None or raw_rgb_sha256_by_frame is None:
                raise ValueError("S1.3 native P2/v4 requires frozen M6 photometric evidence inputs")
            if m61_evidence_config.canonical_sha256 != S13PhotometricEvidenceConfig().canonical_sha256:
                # The current formal identity freezes the evidence definition
                # used by the approved M6 thresholds.  A changed definition is
                # a successor candidate, not an implicit runtime override.
                raise ValueError("S1.3 native P2/v4 evidence config is not the approved formal contract")
            photometric_replay = build_native_s13_m61_evidence(
                pending,
                result_image=m5.final_result.image,
                valid_mask=m5.final_result.valid_mask,
                provenance=m5.final_result.pixel_provenance,
                transactions=transaction_rows,
                replay_pairs=replay_pairs,
                source_count=len(schedule.assignments),
                generation_id=generation_id,
                p1_parent_completion_sha256=p1_completion_sha,
                branch=m61_evidence_branch,
                run_id=m61_evidence_run_id,
                frame_image_loader=image_loader,  # type: ignore[arg-type]
                raw_rgb_sha256=raw_rgb_sha256_by_frame,
                config=m61_evidence_config,
            )
        elif p2_completion_schema not in (
            P2_COMPLETION_SCHEMA,
            S13_M51_R2_P2_COMPLETION_SCHEMA,
            S13_M51_R3_P2_COMPLETION_SCHEMA,
            S13_M51_R4_P2_COMPLETION_SCHEMA,
        ):
            raise ValueError("S1.3 P2 completion schema is unsupported")
        ranked = sorted(
            enumerate(m5.pairs),
            key=lambda item: float(item[1].transaction.get("after_metrics", {}).get("score") or -1.0),
            reverse=True,
        )[:10]
        crop_rows: list[dict[str, object]] = []
        for rank, (pair_index, pair) in enumerate(ranked):
            seam = np.asarray(pair.seam_x_by_row, dtype=np.int32)
            left = max(0, int(seam.min()) - 48)
            right = min(schedule.canvas_width, int(seam.max()) + 49)
            crop = m5.seam_overlay[:, left:right]
            name = f"rank_{rank:02d}_pair_{pair_index:04d}.png"
            write_image(pending / "worst_seam_crops" / name, crop)
            crop_rows.append({
                "rank": rank, "pair_index": pair_index, "left_x": left, "right_x": right,
                "after_score": pair.transaction.get("after_metrics", {}).get("score"), "asset": name,
            })
        atomic_write_json(pending / "worst_seam_crops" / "manifest.json", {
            "schema": "gemini305-video-s13-m5-worst-seam-crops/v1", "crops": crop_rows,
        })
        performance = {
            **dict(m5.performance),
            "input_and_preflight": 0.0,
            "trajectory": 0.0,
            "motion_measurement": 0.0,
            "motion_hypotheses": 0.0,
            "progress_and_layout": 0.0,
            "time_to_P0": None,
            "p1_verify_and_load": time.perf_counter() - started - float(m5.performance["total_m5"]),
            "vertical": 0.0,
            "geometry": float(m5.performance["geometry"]),
            "seam": float(m5.performance["seam_and_p2_render"]),
            "photometric": None,
            "blend": None,
            "repair": None,
            "artifact_export": None,
            "total_wall_time": time.perf_counter() - started,
            "total_m5": float(m5.performance["total_m5"]),
            "m4_reestimated_in_m5": False,
            "gain_enumeration_count_in_m5": int(m5.performance.get("gain_enumeration_count", 0)),
            "m4_selection_full_resolution_render_count": 0,
            "p2_full_resolution_render_count": int(m5.performance.get("p2_full_resolution_render_count", 2)),
            "peak_memory": None,
            "excluded_after_m5": ["photometric", "luminance_field", "feather", "multiband", "depth", "mesh", "source_rescue"],
        }
        atomic_write_json(pending / "performance.json", performance)
        if not m5.hard_audit_passed:
            atomic_write_json(generation / "P2_hard_failure.json", {
                "schema": "gemini305-video-s13-p2-hard-failure/v1",
                "generation_id": generation_id,
                "parent_stage": "P1",
                "parent_completion_sha256": p1_completion_sha,
                "hard_audit": dict(m5.hard_audit),
            })
            raise ValueError("S1.3 M5 P2 hard audit failed")
        result_asset = "geometry_and_seam_panorama_owner_only.png"
        selected_counts = Counter(
            str(pair.transaction.get("seam_model")) for pair in m5.pairs
        )
        completion_metadata: dict[str, object] = {
                "generation_id": generation_id,
                "stage": "P2",
                "hard_audit_passed": True,
                "parent_stage": "P1",
                "parent_completion_sha256": p1_completion_sha,
                "parent_result_sha256": p1_completion["result_asset_sha256"],
                "p0_ancestor_completion_sha256": p0_completion_sha,
                "result_asset": result_asset,
                "result_asset_sha256": sha256_file(pending / result_asset),
                "hard_audit_sha256": sha256_file(pending / "hard_audit.json"),
                "diagnostic_quality_report_sha256": sha256_file(
                    pending / "diagnostic_quality_report.json"
                ),
                "selected_as_best": bool(m5.selected_as_best),
                "selected_as_best_deprecated": True,
                "selected_as_best_runtime_authority": False,
                "selected_vertical_parent": "P1",
                "selected_global_gain": vertical.selected_gain,
                "p0_parent": "../P0/P0_completion.json",
                "p0_parent_sha256": p0_completion_sha,
                "p1_candidate_parent": "../P1/P1_completion.json",
                "p1_candidate_parent_sha256": p1_completion_sha,
                "formal_raw_rgb_remap_invocations": m5.final_result.remap_invocations,
                "formal_raw_rgb_unique_sources": len(m5.final_result.decoded_frame_ids),
                "all_pair_transaction_count": len(m5.pairs),
                "pair_transaction_count": len(m5.pairs),
                "applied_pair_transaction_count": sum(pair.transaction["decision"] == "applied" for pair in m5.pairs),
                "selected_seam_model_counts": dict(selected_counts),
                "pair_fallback_count": sum(
                    bool(pair.transaction.get("fallback_used")) for pair in m5.pairs
                ),
                "source_count": len(schedule.assignments),
                "p2_replay_schema": replay_schema,
                "p2_replay_pair_count": len(replay_pairs),
                "p2_replay_manifest_sha256": sha256_file(
                    pending / "p2_replay_manifest.json"
                ),
                "p2_seams_sha256": sha256_file(pending / "p2_seams.npz"),
                "before_mean_structure_score": m5.before_mean_score,
                "after_mean_structure_score": m5.after_mean_score,
                "selection_audit": dict(m5.selection_audit),
                "comparison_coordinate_policy": m5.selection_audit[
                    "comparison_coordinate_policy"
                ],
                "models": ["C0", "C1", "C2", "C3", "C4", "S0", "S1", "S2"],
                "excluded_models": [
                    "graphcut", "photometric_gain_bias", "luminance_field", "feather", "multiband",
                    "depth", "mesh", "source_rescue", "production_renderer",
                ],
                "m6_eligible": bool(m6_eligible),
                **(dict(candidate_identity or {}) if r4_enabled else {}),
                **({
                    "component_transaction_manifest_sha256": component_manifest_sha,
                    "source_correction_manifest_sha256": source_correction_manifest_sha,
                    "source_map_oracle_manifest_sha256": source_map_manifest_sha,
                    "aggregate_pair_transaction_manifest_sha256": sha256_file(
                        pending / "pair_transactions.json"
                    ),
                    "application_state": component_audit.get("application_state", "none"),
                    "repair_complete": component_audit.get("repair_complete", False),
                    "applied_segment_ids": component_audit.get("accepted_segment_ids", []),
                    "resolved_segment_ids": [
                        row["segment_id"] for row in component_audit.get("segments", [])
                        if row.get("state") == "resolved"
                    ],
                    "improved_segment_ids": [
                        row["segment_id"] for row in component_audit.get("segments", [])
                        if row.get("state") == "improved_unresolved"
                    ],
                    "rejected_segment_ids": component_audit.get("rejected_segment_ids", []),
                    "deferred_segment_ids": component_audit.get("deferred_segment_ids", []),
                    "provenance_schema": "gemini305-video-s13-p2-provenance/v6-r2",
                } if r4_enabled else {}),
            }
        if not m6_eligible:
            if not isinstance(m6_blocked_reason, str) or not m6_blocked_reason:
                raise ValueError("S1.3 P2-only completion requires an M6 blocked reason")
            completion_metadata["m6_blocked_reason"] = m6_blocked_reason
        if photometric_replay is not None:
            completion_metadata.update({
                "native_p2_v4": True,
                "p2_v3_parent": None,
                "photometric_replay_manifest_sha256": sha256_file(
                    pending / "photometric_replay/manifest.json"
                ),
                "owner_domain_manifest_sha256": photometric_replay[
                    "owner_domain_manifest_sha256"
                ],
                "graph_topology_manifest_sha256": photometric_replay[
                    "graph_topology_manifest_sha256"
                ],
                "graph_topology_sha256": photometric_replay["graph_topology_sha256"],
                "photometric_evidence_config_sha256": photometric_replay[
                    "photometric_evidence_config_sha256"
                ],
                "q_solver_invocations": 0,
                "q0_b0_only": True,
            })
        completion = seal_stage(
            pending,
            completion_name="P2_completion.json",
            schema=p2_completion_schema,
            metadata=completion_metadata,
        )
        verify_stage(pending, completion_name="P2_completion.json", schema=p2_completion_schema)
        if r4_enabled and candidate_identity is not None and not bool(
            candidate_identity.get("working_tree_dirty", True)
        ):
            # Semantic verification is part of the seal boundary.  Verify the
            # completed staging directory before it becomes the formal P2 so a
            # fail-closed rejection cannot leave a sealed-looking final tree.
            verify_s13_v6_r2_p2(pending)
        os.replace(pending, final)
        verify_stage(final, completion_name="P2_completion.json", schema=p2_completion_schema)
        if sha256_file(p0_completion_path) != p0_completion_sha:
            raise ValueError("S1.3 M5 immutable P0 completion changed")
        if sha256_file(p1_completion_path) != p1_completion_sha:
            raise ValueError("S1.3 M5 immutable original P1 completion changed")
        pointer = update_current_latest(
            root, generation, stage="P2", completion_name="P2_completion.json",
            completion_schema=p2_completion_schema, result_asset=result_asset,
        )
        return {
            "state": "P2_sealed",
            "panorama": str(final / "geometry_and_seam_panorama_owner_only.png"),
            "geometry": str(final / "geometry_panorama_owner_only.png"),
            "vertical_parent": str(final / "selected_vertical_parent.png"),
            "seam_overlay": str(final / "seam_overlay.png"),
            "pixel_provenance": str(final / "p2_pixel_provenance.npz"),
            "pair_transactions": str(final / "pair_transactions.json"),
            "worst_seam_crops": str(final / "worst_seam_crops"),
            "performance": str(final / "performance.json"),
            "completion": str(final / "P2_completion.json"),
            "current_latest": str(root / "current_latest.json"),
            "selected_as_best": bool(completion["selected_as_best"]),
            "selected_vertical_parent": "P1",
            "selected_global_gain": vertical.selected_gain,
            "hard_audit_passed": True,
            "latest_stage": pointer["stage"],
            "pair_count": len(m5.pairs),
            "applied_pair_count": sum(pair.transaction["decision"] == "applied" for pair in m5.pairs),
            "formal_raw_rgb_remap_invocations": m5.final_result.remap_invocations,
        }
    except BaseException:
        discard_staging(pending)
        raise


def _run_m6(
    generation: Path,
    root: Path,
    generation_id: str,
    image_loader: object,
    *,
    p2_completion_schema: str = P2_COMPLETION_SCHEMA,
) -> dict[str, object]:
    """Append P3 by replaying only sealed P2 state and original RGB."""

    if not callable(image_loader):
        raise TypeError("S1.3 M6 requires a raw RGB loader")
    final = generation / "P3"
    if final.exists():
        raise FileExistsError(f"S1.3 M6 P3 already exists: {final}")
    reviewed_before = (
        sha256_file(root / "current_reviewed.json")
        if (root / "current_reviewed.json").is_file() else None
    )
    preview_before = (
        sha256_file(root / "current_preview.json")
        if (root / "current_preview.json").is_file() else None
    )
    try:
        p2 = load_verified_s13_p2_for_m6(
            generation,
            p1_completion_schema=P1_COMPLETION_SCHEMA,
            p2_completion_schema=p2_completion_schema,
        )
    except Exception as exc:
        atomic_write_json(generation / "P3_preflight_failure.json", {
            "schema": "gemini305-video-s13-p3-preflight-failure/v1",
            "generation_id": generation_id,
            "parent_stage": "P2",
            "error_type": type(exc).__name__,
            "error": str(exc),
        })
        raise
    for attempt in range(2):
        pending = generation / f".P3.{uuid.uuid4().hex}.pending"
        pending.mkdir()
        force_fallback = attempt == 1
        try:
            started = time.perf_counter()
            p3 = run_s13_m6(
                p2, image_loader, force_identity_owner_only=force_fallback
            )
            atomic_write_json(pending / "p2_parent_reference.json", {
                "schema": "gemini305-video-s13-p2-parent-reference/v1",
                "parent_stage": "P2",
                "completion": "../P2/P2_completion.json",
                "completion_sha256": p2.completion_sha256,
                "result_asset": "../P2/geometry_and_seam_panorama_owner_only.png",
                "result_asset_sha256": p2.completion["result_asset_sha256"],
                "pixel_provenance_sha256": p2.completion["assets_sha256"][
                    "p2_pixel_provenance.npz"
                ],
            })
            write_image(pending / "photometric_owner_only.png", p3.photometric_owner_only)
            write_image(pending / "photometric_owner_only.jpg", p3.photometric_owner_only)
            write_image(pending / "visual_panorama.png", p3.visual_panorama)
            write_image(pending / "visual_panorama.jpg", p3.visual_panorama)
            write_image(pending / "p3_valid_mask.png", p3.valid_mask.astype(np.uint8) * 255)
            write_image(
                pending / "photometric_training_mask.png",
                p3.photometric_training_mask.astype(np.uint8) * 255,
            )
            write_image(
                pending / "photometric_heldout_mask.png",
                p3.photometric_heldout_mask.astype(np.uint8) * 255,
            )
            write_image(
                pending / "protected_structure_mask.png",
                p3.protected_structure_mask.astype(np.uint8) * 255,
            )
            write_image(
                pending / "safe_blend_mask.png",
                p3.safe_blend_mask.astype(np.uint8) * 255,
            )
            write_image(
                pending / "blend_weight_map.png",
                np.rint(np.clip(p3.blend_weight_map, 0.0, 1.0) * 255.0).astype(np.uint8),
            )
            write_npz(pending / "p3_pixel_provenance.npz", p3.pixel_provenance)
            solution_document = photometric_solution_document(p3.photometric_solution)
            atomic_write_json(pending / "photometric_solution.json", solution_document)
            write_npz(pending / "photometric_solution.npz", {
                "source_index": np.asarray(
                    [item.source_index for item in p3.photometric_solution.source_parameters], np.int32
                ),
                "frame_id": np.asarray(
                    [item.frame_id for item in p3.photometric_solution.source_parameters], np.int32
                ),
                "gain_bgr": np.asarray(
                    [item.gain_bgr for item in p3.photometric_solution.source_parameters], np.float32
                ),
                "bias_bgr": np.asarray(
                    [item.bias_bgr for item in p3.photometric_solution.source_parameters], np.float32
                ),
            })
            blend_rows = []
            for plan in p3.blend_plans:
                document = blend_transaction_document(plan.transaction)
                name = f"pair_{plan.transaction.pair_index:04d}.json"
                atomic_write_json(pending / "blend_transactions" / name, document)
                blend_rows.append({**document, "asset": f"blend_transactions/{name}"})
            atomic_write_json(pending / "blend_transactions.json", {
                "schema": BLEND_SCHEMA,
                "all_pairs_reported": len(blend_rows) == len(p2.replay_pairs),
                "pairs": blend_rows,
            })
            atomic_write_json(pending / "pair_photometric_report.json", {
                "schema": "gemini305-video-s13-pair-photometric-report/v1",
                "diagnostic_only": True,
                "runtime_authority": False,
                "pairs": [
                    {
                        "pair_index": sample.pair_index,
                        "train_sample_count": len(sample.train_left_rgb_linear),
                        "heldout_sample_count": len(sample.heldout_left_rgb_linear),
                        "safe_mask_sha256": sample.safe_mask_sha256,
                        "protected_mask_sha256": sample.protected_mask_sha256,
                    }
                    for sample in p3.photometric_samples
                ],
            })
            atomic_write_json(
                pending / "diagnostic_quality_report.json", dict(p3.diagnostic_quality)
            )
            # Export deterministic full-height risk/seam crops.  These are
            # evidence only and never participate in candidate selection.
            pair_quality = list(p3.diagnostic_quality.get("pair_reports", []))
            ranked_color = sorted(
                pair_quality,
                key=lambda row: float(row.get("p3_luminance_jump") or -1.0),
                reverse=True,
            )[:10]
            ranked_blur = sorted(
                enumerate(p3.blend_plans),
                key=lambda item: item[1].transaction.blended_pixel_count,
                reverse=True,
            )[:10]
            crop_manifest: list[dict[str, object]] = []
            for category, ranked in (("color", ranked_color), ("blur_ghost", ranked_blur)):
                for rank, item in enumerate(ranked):
                    pair_index = int(item[0] if isinstance(item, tuple) else item["pair_index"])
                    pair = p2.replay_pairs[pair_index]
                    x0 = max(0, int(pair.seam_x_by_row.min()) - 48)
                    x1 = min(p3.visual_panorama.shape[1], int(pair.seam_x_by_row.max()) + 49)
                    comparison = np.concatenate((
                        p2.result_image[:, x0:x1],
                        p3.photometric_owner_only[:, x0:x1],
                        p3.visual_panorama[:, x0:x1],
                    ), axis=1)
                    name = f"{category}_{rank:02d}_pair_{pair_index:04d}.png"
                    write_image(pending / "worst_visual_crops" / name, comparison)
                    crop_manifest.append({
                        "category": category, "rank": rank, "pair_index": pair_index,
                        "x0": x0, "x1": x1, "asset": name,
                    })
            protected_rows = np.count_nonzero(p3.protected_structure_mask, axis=1)
            for rank, center_y in enumerate(np.argsort(protected_rows)[-3:][::-1]):
                y0, y1 = max(0, int(center_y) - 48), min(p3.visual_panorama.shape[0], int(center_y) + 49)
                name = f"high_risk_horizontal_{rank:02d}.png"
                write_image(
                    pending / "worst_visual_crops" / name,
                    np.concatenate((p2.result_image[y0:y1], p3.visual_panorama[y0:y1]), axis=1),
                )
                crop_manifest.append({
                    "category": "high_risk_horizontal", "rank": rank,
                    "y0": y0, "y1": y1, "asset": name,
                })
            atomic_write_json(pending / "worst_visual_crops" / "manifest.json", {
                "schema": "gemini305-video-s13-p3-worst-visual-crops/v1",
                "crops": crop_manifest,
            })
            performance = dict(p3.performance)
            performance["artifact_export"] = time.perf_counter() - started - float(
                performance["total_m6"]
            )
            performance["total_m6"] = time.perf_counter() - started
            atomic_write_json(pending / "performance.json", performance)
            pending_hashes = {
                path.relative_to(pending).as_posix(): sha256_file(path)
                for path in sorted(pending.rglob("*")) if path.is_file()
            }
            audit_started = time.perf_counter()
            audit = audit_s13_p3_stage(
                p2, p3, pending_asset_sha256=pending_hashes
            )
            performance["p3_hard_audit"] = time.perf_counter() - audit_started
            performance["total_m6"] = time.perf_counter() - started
            atomic_write_json(pending / "performance.json", performance)
            atomic_write_json(pending / "hard_audit.json", audit)
            if audit.get("passed") is not True:
                if not force_fallback:
                    atomic_write_json(generation / "P3_first_attempt_failure.json", {
                        "schema": "gemini305-video-s13-p3-first-attempt-failure/v1",
                        "generation_id": generation_id,
                        "fallback": "Q0_identity+B0_owner_only",
                        "hard_audit": audit,
                    })
                    discard_staging(pending)
                    continue
                atomic_write_json(generation / "P3_failure_bundle.json", {
                    "schema": "gemini305-video-s13-p3-failure-bundle/v1",
                    "generation_id": generation_id,
                    "parent_stage": "P2",
                    "parent_completion_sha256": p2.completion_sha256,
                    "fallback_attempted": "Q0_identity+B0_owner_only",
                    "hard_audit": audit,
                })
                raise ValueError("S1.3 M6 Q0+B0 P3 hard audit failed")
            result_asset = "visual_panorama.png"
            seal_stage(
                pending,
                completion_name="P3_completion.json",
                schema=P3_COMPLETION_SCHEMA,
                metadata={
                    "generation_id": generation_id,
                    "stage": "P3",
                    "hard_audit_passed": True,
                    "parent_stage": "P2",
                    "parent_completion_sha256": p2.completion_sha256,
                    "parent_result_sha256": p2.completion["result_asset_sha256"],
                    "result_asset": result_asset,
                    "result_asset_sha256": sha256_file(pending / result_asset),
                    "pixel_provenance_sha256": sha256_file(
                        pending / "p3_pixel_provenance.npz"
                    ),
                    "hard_audit_sha256": sha256_file(pending / "hard_audit.json"),
                    "diagnostic_quality_report_sha256": sha256_file(
                        pending / "diagnostic_quality_report.json"
                    ),
                    "photometric_solution_sha256": sha256_file(
                        pending / "photometric_solution.json"
                    ),
                    "blend_transactions_sha256": sha256_file(
                        pending / "blend_transactions.json"
                    ),
                    "diagnostic_quality_passed": bool(
                        p3.diagnostic_quality.get("quality_pass")
                    ),
                    "diagnostic_quality_runtime_authority": False,
                    "formal_raw_rgb_unique_sources": performance[
                        "formal_raw_rgb_unique_sources"
                    ],
                    "formal_raw_rgb_remap_invocations": performance[
                        "formal_raw_rgb_remap_invocations"
                    ],
                    "maximum_real_contributors_per_pixel": 2 if np.any(
                        p3.blend_weight_map > 0.0
                    ) else 1,
                    "protected_blend_pixel_count": int(np.count_nonzero(
                        p3.protected_structure_mask & (p3.blend_weight_map > 0.0)
                    )),
                    "m4_reestimated_in_m6": 0,
                    "m5_reestimated_in_m6": 0,
                    "geometry_reestimated_in_m6": 0,
                    "seam_reestimated_in_m6": 0,
                    "trajectory_estimation_invocations": 0,
                    "open3d_invocations": 0,
                    "uses_depth": False,
                    "uses_tsdf": False,
                    "uses_graphcut": False,
                    "uses_mesh": False,
                    "uses_source_rescue": False,
                    "diagnostic_only": True,
                    "production_eligible": False,
                    "production_lock_eligible": False,
                },
            )
            os.replace(pending, final)
            verify_sealed_s13_p3(final, completion_schema=P3_COMPLETION_SCHEMA)
            if not all(
                (generation / name).is_file()
                and sha256_file(generation / name) == expected
                for name, expected in p2.immutable_sha256.items()
            ):
                raise ValueError("S1.3 M6 immutable P2 ancestry changed after P3 seal")
            pointer = update_current_latest(
                root, generation, stage="P3", completion_name="P3_completion.json",
                completion_schema=P3_COMPLETION_SCHEMA, result_asset=result_asset,
            )
            reviewed_after = (
                sha256_file(root / "current_reviewed.json")
                if (root / "current_reviewed.json").is_file() else None
            )
            preview_after = (
                sha256_file(root / "current_preview.json")
                if (root / "current_preview.json").is_file() else None
            )
            if reviewed_before != reviewed_after or preview_before != preview_after:
                raise ValueError("S1.3 M6 modified reviewed/preview pointers")
            return {
                "state": "P3_sealed",
                "panorama": str(final / result_asset),
                "photometric_owner_only": str(final / "photometric_owner_only.png"),
                "pixel_provenance": str(final / "p3_pixel_provenance.npz"),
                "hard_audit": str(final / "hard_audit.json"),
                "diagnostic_quality": str(final / "diagnostic_quality_report.json"),
                "performance": str(final / "performance.json"),
                "completion": str(final / "P3_completion.json"),
                "current_latest": str(root / "current_latest.json"),
                "latest_stage": pointer["stage"],
                "hard_audit_passed": True,
                "diagnostic_quality_passed": bool(
                    p3.diagnostic_quality.get("quality_pass")
                ),
                "fallback_rebuild_used": force_fallback,
                "blend_pixel_count": int(np.count_nonzero(p3.blend_weight_map > 0.0)),
                "protected_blend_pixel_count": 0,
                "formal_raw_rgb_remap_invocations": performance[
                    "formal_raw_rgb_remap_invocations"
                ],
            }
        except BaseException:
            if pending.exists():
                discard_staging(pending)
            raise
    raise AssertionError("S1.3 M6 fallback loop did not terminate")


def _run_manual_c2e_m7(
    generation: Path,
    root: Path,
    generation_id: str,
) -> dict[str, object]:
    """Seal the explicitly authorized M7 R0 winner without new pixels."""

    p3_root = generation / "P3"
    p3_completion_path = p3_root / "P3_completion.json"
    p3 = verify_stage(
        p3_root,
        completion_name="P3_completion.json",
        schema=P3_COMPLETION_SCHEMA,
    )
    if p3.get("hard_audit_passed") is not True:
        raise ValueError("S1.3 manual C2E M7 requires a hard-audited P3")
    result_name = str(p3.get("result_asset", "visual_panorama.png"))
    p3_image_path = p3_root / result_name
    p3_provenance_path = p3_root / "p3_pixel_provenance.npz"
    p3_image = cv2.imread(str(p3_image_path), cv2.IMREAD_COLOR)
    if p3_image is None:
        raise ValueError("S1.3 manual C2E M7 parent panorama is unreadable")
    try:
        with np.load(p3_provenance_path, allow_pickle=False) as stored:
            provenance = {
                name: np.asarray(stored[name]).copy() for name in stored.files
            }
    except (OSError, ValueError, KeyError) as exc:
        raise ValueError("S1.3 manual C2E M7 parent provenance is invalid") from exc

    final = generation / "P4"
    if final.exists():
        raise FileExistsError(f"S1.3 manual C2E M7 P4 already exists: {final}")
    pending = generation / f".P4.{uuid.uuid4().hex}.pending"
    pending.mkdir()
    started = time.perf_counter()
    try:
        write_image(pending / "final_panorama.png", p3_image)
        write_image(pending / "final_panorama.jpg", p3_image)
        write_npz(pending / "final_pixel_provenance.npz", provenance)
        atomic_write_json(pending / "manual_authorization.json", {
            "schema": "gemini305-video-s13-m7-manual-c2e-authorization/v1",
            "authorization": "apply_all_structurally_safe_c2e_corrections",
            "authorized_selection": "R0_keep_p3",
            "explicit_user_authorization_required": True,
            "new_pixel_generation": False,
            "quality_gates_runtime_authority": False,
            "structural_map_safety_runtime_authority": True,
        })
        final_png_sha = sha256_file(pending / "final_panorama.png")
        parent_png_sha = sha256_file(p3_image_path)
        pixels_equal = bool(np.array_equal(
            p3_image,
            cv2.imread(str(pending / "final_panorama.png"), cv2.IMREAD_COLOR),
        ))
        hard_audit = {
            "schema": "gemini305-video-s13-p4-manual-c2e-hard-audit/v1",
            "passed": pixels_equal,
            "selection": "R0_keep_p3",
            "parent_stage": "P3",
            "parent_completion_sha256": sha256_file(p3_completion_path),
            "parent_result_sha256": parent_png_sha,
            "final_result_sha256": final_png_sha,
            "pixel_exact_to_p3": pixels_equal,
            "new_pixel_generation": False,
            "p2_reestimated_in_m7": 0,
            "p3_reestimated_in_m7": 0,
        }
        atomic_write_json(pending / "hard_audit.json", hard_audit)
        if not pixels_equal:
            raise ValueError("S1.3 manual C2E M7 changed P3 pixels")
        completion = seal_stage(
            pending,
            completion_name="P4_completion.json",
            schema=P4_MANUAL_C2E_COMPLETION_SCHEMA,
            metadata={
                "generation_id": generation_id,
                "stage": "P4",
                "milestone": "M7",
                "parent_stage": "P3",
                "parent_completion_sha256": sha256_file(p3_completion_path),
                "parent_result_sha256": parent_png_sha,
                "result_asset": "final_panorama.png",
                "result_asset_sha256": final_png_sha,
                "pixel_provenance_sha256": sha256_file(
                    pending / "final_pixel_provenance.npz"
                ),
                "hard_audit_passed": True,
                "hard_audit_sha256": sha256_file(pending / "hard_audit.json"),
                "manual_authorization_sha256": sha256_file(
                    pending / "manual_authorization.json"
                ),
                "selected_candidate": "R0_keep_p3",
                "application_policy": "all_structurally_safe",
                "quality_gates_runtime_authority": False,
                "structural_map_safety_runtime_authority": True,
                "new_pixel_generation": False,
                "total_m7_seconds": time.perf_counter() - started,
                "diagnostic_only": True,
                "production_eligible": False,
                "production_lock_eligible": False,
            },
        )
        os.replace(pending, final)
        verify_stage(
            final,
            completion_name="P4_completion.json",
            schema=P4_MANUAL_C2E_COMPLETION_SCHEMA,
        )
        pointer = update_current_latest(
            root,
            generation,
            stage="P4",
            completion_name="P4_completion.json",
            completion_schema=P4_MANUAL_C2E_COMPLETION_SCHEMA,
            result_asset="final_panorama.png",
        )
        return {
            "state": "P4_sealed",
            "milestone": "M7",
            "panorama": str(final / "final_panorama.png"),
            "pixel_provenance": str(final / "final_pixel_provenance.npz"),
            "completion": str(final / "P4_completion.json"),
            "hard_audit": str(final / "hard_audit.json"),
            "latest_stage": pointer["stage"],
            "selected_candidate": "R0_keep_p3",
            "total_m7_seconds": float(completion["total_m7_seconds"]),
        }
    except BaseException:
        if pending.exists():
            discard_staging(pending)
        raise


def _run_formal_m6(
    generation: Path,
    root: Path,
    session: S13Session,
    candidate_config: Path,
) -> dict[str, object]:
    """Append the M6.1 implementation as the canonical S013 P3/v2 stage."""

    from .video_s13_m61_runner import run_s13_m61_acceptance

    result = run_s13_m61_acceptance(
        p2_root=generation / "P2",
        session=session.root,
        output=generation,
        candidate_config=candidate_config,
        pointer_root=root,
        raw_rgb_paths={frame.frame_id: frame.color_path for frame in session.frames},
    )
    final = Path(str(result["p3"])).resolve()
    completion = final / "P3_completion.json"
    return {
        "state": "P3_sealed",
        "implementation": "M6.1_formal_M6",
        "stage_acceptance": "hard_audit_sealed",
        "visual_acceptance_state": "not_granted",
        "m7_handoff_eligible": False,
        "panorama": str(final / "visual_panorama.png"),
        "photometric_owner_only": str(final / "photometric_owner_only.png"),
        "pixel_provenance": str(final / "p3_pixel_provenance.npz"),
        "hard_audit": str(final / "hard_audit.json"),
        "effective_config": str(final / "effective_config.json"),
        "blend_transactions": str(final / "blend_transactions.json"),
        "performance": str(final / "performance.json"),
        "completion": str(completion),
        "current_latest": str(root / "current_latest.json"),
        "latest_stage": result["pointer"]["stage"],
        "hard_audit_passed": True,
        "diagnostic_quality": str(generation / "validation_m61" / "quality_report.json"),
        "diagnostic_quality_runtime_authority": False,
        "reproducibility": dict(result["reproducibility"]),
    }


def _write_nonpanorama(
    staging: Path,
    session: S13Session,
    *,
    generation_id: str,
    trajectory_audit: Mapping[str, Any],
    run_started: float,
    layout_audit: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    target = staging / "nonpanorama"
    target.mkdir(parents=True)
    images = [read_s13_rgb(frame) for frame in session.frames]
    if len(images) == 1:
        from .video_s12_schedule import build_s012_schedule

        frame = session.frames[0]
        schedule = build_s012_schedule(
            (frame.frame_id,),
            (float(session.calibration.cx),),
            session.calibration,
            hard_internal_width_px=None,
        )
        rendered = render_s13_p0(
            schedule,
            session.calibration,
            lambda _frame_id: images[0],
            placement_methods=("origin",),
        )
        image = rendered.image
        asset_type, render_state = "calibrated_single_view" if session.calibration_mode == "calibrated_target" else "raw_single_view", "single_view_only"
        owner = rendered.pixel_provenance["owner_frame_id"]
        source_u = rendered.pixel_provenance["source_u"]
        source_v = rendered.pixel_provenance["source_v"]
        valid = rendered.valid_mask
    else:
        columns = [image[:, image.shape[1] // 2 : image.shape[1] // 2 + 1] for image in images]
        image = np.concatenate(columns, axis=1)
        asset_type, render_state = "temporal_slit_preview", "temporal_preview_only"
        owner = np.broadcast_to(np.asarray([frame.frame_id for frame in session.frames], dtype=np.int32)[None, :], image.shape[:2]).copy()
        source_u = np.broadcast_to(np.asarray([frame.width // 2 for frame in session.frames], dtype=np.float32)[None, :], image.shape[:2]).copy()
        source_v = np.broadcast_to(np.arange(image.shape[0], dtype=np.float32)[:, None], image.shape[:2]).copy()
        valid = np.ones(owner.shape, dtype=bool)
    write_image(target / "preview.png", image)
    write_image(target / "preview.jpg", image)
    write_npz(target / "pixel_provenance.npz", {
        "owner_frame_id": owner, "source_u": source_u, "source_v": source_v,
        "valid": valid, "panorama_claim": np.asarray("none"),
    })
    completion = {
        "schema": "gemini305-video-s13-nonpanorama-completion/v1",
        "generation_id": generation_id,
        "asset_type": asset_type,
        "render_state": render_state,
        "panorama_claim": "none",
        "diagnostic_only": True,
        "production_eligible": False,
        "manual_review_required": True,
        "assets_sha256": {name: sha256_file(target / name) for name in ("preview.png", "preview.jpg", "pixel_provenance.npz")},
        "time_to_visible_asset_seconds": time.perf_counter() - run_started,
        "trajectory": dict(trajectory_audit),
        "layout": dict(layout_audit or {}),
    }
    atomic_write_json(target / "completion.json", completion)
    return completion


def run_s13_experiment(
    *,
    input_path: Path,
    output: Path,
    candidate_config: Path,
    algorithm_spec: VideoAlgorithmSpec,
    trajectory_cache: Path | None,
    reuse_online_trajectory: bool,
    run_offline_orb: bool,
    ignore_pose: bool,
    config_path: Path | None,
    simulate_optimizer_all_fail: bool = False,
    run_m4: bool = True,
    run_m5: bool = True,
    run_m6: bool | None = None,
    resume_generation: Path | None = None,
    manual_c2e_forward_m7: bool = False,
) -> dict[str, Any]:
    run_started = time.perf_counter()
    config = load_s13_config(candidate_config)
    configured_algorithm_id = str(config.document["algorithm_id"])
    configured_implementation_id = str(config.document["implementation_id"])
    if (
        algorithm_spec.algorithm_id != configured_algorithm_id
        or algorithm_spec.implementation_id != configured_implementation_id
        or algorithm_spec.config_sha256 != candidate_config_sha(config.document)
    ):
        raise ValueError("S1.3 dispatch identity/config binding changed after validation")
    m51_r4_document_for_policy = config.component.get("m51_r4")
    manual_c2e_authorized = bool(
        manual_c2e_forward_m7
        and config.m51_r4_enabled
        and isinstance(m51_r4_document_for_policy, Mapping)
        and m51_r4_document_for_policy.get("application_policy")
        == "all_structurally_safe"
    )
    if manual_c2e_forward_m7 and not manual_c2e_authorized:
        raise ValueError(
            "S1.3 manual C2E M7 requires the v6-r2 all-structurally-safe policy"
        )
    if manual_c2e_forward_m7 and resume_generation is not None:
        raise ValueError("S1.3 manual C2E M7 requires a new complete M1-M7 run")
    if manual_c2e_authorized:
        run_m6 = True
    if config.p2_only and run_m6 is True and not manual_c2e_authorized:
        raise ValueError("S1.3 successor is P2-only and cannot run M6/P3")
    if config.p2_only and resume_generation is not None:
        raise ValueError("S1.3 successor is P2-only and cannot resume into M6/P3")
    if run_m6 is None:
        run_m6 = config.identity.default_stop_after != "P2"
    if run_m6 and not config.m6_eligible and not manual_c2e_authorized:
        raise ValueError("S1.3 identity is not M6 eligible")
    formal_m6 = configured_algorithm_id == S13_FORMAL_M6_ALGORITHM_ID
    from .video_s13_m51_r2 import S13M51R2Config, S13M51R3Config, S13M51R4Config

    m51_r2_document = config.component.get("m51_r2")
    if config.m51_r2_enabled:
        if not isinstance(m51_r2_document, Mapping):
            raise ValueError("S1.3 P2 successor M5.1-r2 configuration is missing")
        if config.m51_r4_enabled:
            m51_r3_document = config.component.get("m51_r3")
            m51_r4_document = config.component.get("m51_r4")
            if not isinstance(m51_r3_document, Mapping) or not isinstance(
                m51_r4_document, Mapping
            ):
                raise ValueError("S1.3 v6-r2 M5.1-r4 configuration is missing")
            arguments = dict(m51_r2_document)
            arguments.update(dict(m51_r4_document))
            arguments.update({
                "complete_reassessment_pair_indices": tuple(
                    int(value) for value in m51_r3_document["reassessment_pair_indices"]
                ),
                "component_local_ambiguity_enabled": True,
            })
            m51_r2_config = S13M51R4Config(**arguments)
        elif config.m51_r3_enabled:
            m51_r3_document = config.component.get("m51_r3")
            if not isinstance(m51_r3_document, Mapping):
                raise ValueError("S1.3 v6 M5.1-r3 configuration is missing")
            m51_r2_config = S13M51R3Config(
                **dict(m51_r2_document),
                complete_reassessment_pair_indices=tuple(
                    int(value)
                    for value in m51_r3_document["reassessment_pair_indices"]
                ),
                component_local_ambiguity_enabled=True,
            )
        else:
            m51_r2_config = S13M51R2Config(**dict(m51_r2_document))
    elif os.environ.get("G305_S13_M51_T0_STATS_ONLY", "0") == "1":
        # Development-only instrumentation for the frozen legacy path.  It
        # records PyrLK error distributions in a new generation but preserves
        # the historical correspondence set and every pixel decision.
        m51_r2_config = S13M51R2Config(enabled=False)
    else:
        m51_r2_config = None
    m61_evidence_config = S13PhotometricEvidenceConfig() if formal_m6 else None
    if m61_evidence_config is not None:
        # Validate every immutable M6 threshold/approval binding before the
        # generation can publish even P0.  Runtime evidence is generated later
        # from this exact frozen definition inside the native P2/v4 stage.
        load_s13_m61_effective_config(
            config.path,
            photometric_evidence_config=asdict(m61_evidence_config),
            config_root=config.path.parents[3],
        )
    root = output.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    stage_seconds: dict[str, float] = {}
    tick = time.perf_counter()
    try:
        session = load_s13_session(input_path, validation_workers=4)
    except Exception as exc:
        atomic_write_json(root / "S013_failure.json", {
            "schema": "gemini305-video-s13-failure/v1",
            "algorithm_id": configured_algorithm_id,
            "render_state": "input_fatal",
            "panorama_claim": "none",
            "trust_state": "invalid_input",
            "diagnostic_only": True,
            "production_eligible": False,
            "manual_review_required": True,
            "error_type": type(exc).__name__,
            "error": str(exc),
        })
        raise
    stage_seconds["input_and_preflight"] = time.perf_counter() - tick
    if resume_generation is not None:
        generation = resume_generation.expanduser().resolve()
        generations_root = (root / "generations").resolve()
        try:
            generation.relative_to(generations_root)
        except ValueError as exc:
            raise ValueError("S1.3 resume generation is outside the output root") from exc
        if generation.parent != generations_root or not generation.is_dir():
            raise ValueError("S1.3 resume generation must be one direct sealed generation")
        generation_manifest = json.loads(
            (generation / "generation_manifest.json").read_text(encoding="utf-8")
        )
        generation_algorithm = generation_manifest.get("algorithm")
        if not isinstance(generation_algorithm, Mapping) or any(
            generation_algorithm.get(key) != expected
            for key, expected in {
                "algorithm_id": configured_algorithm_id,
                "implementation_id": configured_implementation_id,
                "config_sha256": algorithm_spec.config_sha256,
            }.items()
        ):
            raise ValueError("S1.3 resume generation algorithm/config binding changed")
        current = json.loads((root / "current_latest.json").read_text(encoding="utf-8"))
        if current.get("generation_id") != generation.name or current.get("stage") != "P2":
            raise ValueError("S1.3 M6 resume requires current_latest=P2 for the generation")
        if any((generation / name).exists() for name in ("P3", "M6")):
            raise ValueError("S1.3 M6 resume target already has a P3/M6 stage")
        p2_schema = config.p2_completion_schema
        p2_completion_path = generation / "P2" / "P2_completion.json"
        p2_completion = verify_stage(
            generation / "P2", completion_name="P2_completion.json", schema=p2_schema,
        )
        if (
            p2_completion.get("generation_id") != generation.name
            or current.get("completion_sha256") != sha256_file(p2_completion_path)
            or current.get("completion")
            != p2_completion_path.relative_to(root).as_posix()
        ):
            raise ValueError("S1.3 M6 resume pointer is not bound to the sealed P2 parent")
        by_id = session.frame_by_id
        if formal_m6:
            m6 = _run_formal_m6(generation, root, session, config.path)
        else:
            m6 = _run_m6(
                generation, root, generation.name,
                lambda frame_id: read_s13_rgb(by_id[frame_id]),
            )
        report = {
            "schema": REPORT_SCHEMA,
            "algorithm_id": configured_algorithm_id,
            "implementation_id": configured_implementation_id,
            "generation_id": generation.name,
            "generation": str(generation),
            "panorama": m6["panorama"],
            "pixel_provenance": m6["pixel_provenance"],
            "completion": m6["completion"],
            "current_latest": m6["current_latest"],
            "final_stage": "P3",
            "resume_parent_stage": "P2",
            "m6": m6,
            "optimizer_state": "m6_sealed",
            "stage_acceptance": m6.get("stage_acceptance", "hard_audit_sealed"),
            "visual_acceptance_state": m6.get("visual_acceptance_state", "legacy_not_evaluated"),
            "m7_handoff_eligible": bool(m6.get("m7_handoff_eligible", False)),
            "diagnostic_only": True,
            "production_eligible": False,
            "production_lock_eligible": False,
            "manual_review_required": True,
        }
        atomic_write_json(root / "latest_run.json", report)
        return report
    tick = time.perf_counter()
    selected_trajectory_sources = sum(
        (trajectory_cache is not None, reuse_online_trajectory, run_offline_orb, ignore_pose)
    )
    if selected_trajectory_sources != 1:
        raise ValueError("S1.3 requires exactly one explicit trajectory source")
    try:
        trajectory = load_s13_trajectory(
            session,
            trajectory_cache=trajectory_cache,
            reuse_online_trajectory=reuse_online_trajectory,
            run_offline_orb=run_offline_orb,
            ignore_pose=ignore_pose,
            config_path=config_path,
        )
    except Exception as exc:
        requested_source = (
            "trajectory_cache" if trajectory_cache is not None
            else "online_trajectory" if reuse_online_trajectory
            else "offline_orbslam3"
        )
        trajectory = S13Trajectory(
            source=f"{requested_source}_invalid_rgb_fallback",
            source_path=trajectory_cache,
            poses_by_frame_id={},
            pose_origin_by_frame_id={},
            islands=(),
            audit={
                "source": requested_source,
                "trajectory_valid": False,
                "trajectory_error_type": type(exc).__name__,
                "trajectory_error": str(exc),
                "direct_pose_count": 0,
                "session_rgb_count": len(session.frames),
                "pose_supported": False,
                "complete_pose_coverage": False,
                "pose_islands": [],
                "interpolated_pose_count": 0,
                "extrapolated_pose_count": 0,
                "fallback": "rgb_motion_without_pose",
            },
        )
    stage_seconds["trajectory"] = time.perf_counter() - tick
    generation_id = _generation_id(session, algorithm_spec.config_sha256)
    staging = new_generation_staging(root, generation_id)
    generation = root / "generations" / generation_id
    try:
        atomic_write_json(staging / "generation_manifest.json", {
            "schema": GENERATION_SCHEMA,
            "generation_id": generation_id,
            "algorithm": algorithm_spec.as_dict(),
            "implemented_milestones": [
                "M0", "M1", "M2", "M3", *(("M4",) if run_m4 else ()),
                *(("M5",) if run_m4 and run_m5 else ()),
                *(("M6",) if run_m4 and run_m5 and run_m6 else ()),
                *(("M7",) if run_m4 and run_m5 and run_m6 and manual_c2e_authorized else ()),
            ],
            "excluded_milestones": (
                ["M8", "M9"] if run_m4 and run_m5 and run_m6 and manual_c2e_authorized
                else ["M7", "M8", "M9"] if run_m4 and run_m5 and run_m6
                else ["M6", "M7", "M8", "M9"] if run_m4 and run_m5
                else ["M5", "M6", "M7", "M8", "M9"] if run_m4
                else ["M4", "M5", "M6", "M7", "M8", "M9"]
            ),
            "reads_historical_s1_artifacts": False,
            "writes_production_delivery": False,
        })
        atomic_write_json(staging / "input_audit.json", {
            "root": str(session.root), "strict_session": session.strict_video is not None,
            "strict_error": session.strict_error, "trust_state": session.trust_state,
            "calibration_mode": session.calibration_mode, "real_rgb_frame_ids": [frame.frame_id for frame in session.frames],
            "skipped_rgb": list(session.skipped_rgb), "depth_used_for_p0": False,
        })
        atomic_write_json(staging / "trajectory_audit.json", dict(trajectory.audit))
        if len(session.frames) < 2:
            completion = _write_nonpanorama(staging, session, generation_id=generation_id, trajectory_audit=trajectory.audit, run_started=run_started)
            publish_generation(staging, generation)
            pointer = {"schema": "gemini305-video-s13-current-nonpanorama/v1", "generation_id": generation_id, "generation": str(generation.relative_to(root)).replace("\\", "/"), "completion_sha256": sha256_file(generation / "nonpanorama" / "completion.json")}
            atomic_write_json(root / "current_nonpanorama.json", pointer)
            return {"schema": REPORT_SCHEMA, **completion, "generation": str(generation), "panorama": None}

        tick = time.perf_counter()
        motion = measure_s13_motion(session.frames, analysis_width_px=config.analysis_width_px)
        stage_seconds["motion_measurement"] = time.perf_counter() - tick
        tick = time.perf_counter()
        m3_layout = build_s13_m3_layout(session.frames, motion, trajectory)
        progress = m3_layout.progress
        stage_seconds["progress_and_layout"] = time.perf_counter() - tick
        placement_methods = progress.placement_methods[1:]
        progress_deltas = np.diff(np.asarray(progress.centers_x, dtype=np.float64))
        pause_mask = np.asarray([method.startswith("F5_zero_duplicate") for method in placement_methods], dtype=bool)
        pause_expansion_count = int(np.count_nonzero(progress_deltas[pause_mask] > 1e-9))
        pause_expansion_px = float(np.maximum(progress_deltas[pause_mask], 0.0).sum())
        if pause_expansion_count or pause_expansion_px > 1e-9:
            raise ValueError("S1.3 observed pause expanded the panorama layout")
        zero_duplicate_edge_count = int(np.count_nonzero(pause_mask))
        subpixel_motion_edge_count = sum(method.endswith("_subpixel") for method in placement_methods)
        fallback_progress_edge_count = sum(
            method.startswith(("F3_", "F4_", "F7_")) for method in placement_methods
        )
        if not progress.spatial:
            completion = _write_nonpanorama(
                staging, session, generation_id=generation_id, trajectory_audit=trajectory.audit,
                run_started=run_started,
                layout_audit={
                    "layout_level": m3_layout.layout_level,
                    "spatial": False,
                    "canonical_scan_direction": m3_layout.canonical_scan_direction,
                    **dict(m3_layout.audit),
                },
            )
            atomic_write_json(staging / "motion_telemetry.json", {"edges": [_plain_edge(edge) for edge in motion]})
            publish_generation(staging, generation)
            pointer = {"schema": "gemini305-video-s13-current-nonpanorama/v1", "generation_id": generation_id, "generation": str(generation.relative_to(root)).replace("\\", "/"), "completion_sha256": sha256_file(generation / "nonpanorama" / "completion.json")}
            atomic_write_json(root / "current_nonpanorama.json", pointer)
            return {"schema": REPORT_SCHEMA, **completion, "generation": str(generation), "panorama": None}

        schedule_plan = plan_s13_m3_schedule(
            progress, motion, session.calibration,
            normal_target_advance_px=config.normal_target_advance_px,
            risky_target_advance_px=config.risky_target_advance_px,
            segment_break_pairs=m3_layout.segment_break_pairs,
        )
        selection = schedule_plan.selections[0]
        schedule = schedule_plan.schedules[0]
        p0 = staging / "P0"
        p0.mkdir()
        by_id = session.frame_by_id
        tick = time.perf_counter()
        hypothesis_by_frame = {step.target_frame_id: step.selected_hypothesis_id for step in m3_layout.lineage}
        panel_results = []
        panel_images = []
        for panel_index, (panel_selection, panel_schedule) in enumerate(
            zip(schedule_plan.selections, schedule_plan.schedules, strict=True)
        ):
            panel_result = render_s13_p0(
                panel_schedule, session.calibration, lambda frame_id: read_s13_rgb(by_id[frame_id]),
                placement_methods=panel_selection.placement_methods,
                selected_hypothesis_ids=tuple(
                    hypothesis_by_frame.get(frame_id, -1) for frame_id in panel_selection.frame_ids
                ),
            )
            panel_results.append(panel_result)
            panel_images.append(panel_result.image)
            if len(schedule_plan.schedules) > 1:
                panel_root = p0 / "panels" / f"panel_{panel_index:02d}"
                write_image(panel_root / "panorama_owner_only.png", panel_result.image)
                write_image(panel_root / "owner_boundary_overlay.png", _owner_overlay(
                    panel_result.image, panel_result.pixel_provenance["owner_frame_id"]
                ))
                write_npz(panel_root / "pixel_provenance.npz", panel_result.pixel_provenance)
                write_npz(panel_root / "column_provenance.npz", panel_result.column_provenance)
        result = panel_results[0]
        if len(panel_results) == 1:
            write_image(p0 / "base_panorama_owner_only.png", result.image)
            write_image(p0 / "base_panorama_owner_only.jpg", result.image)
            write_image(p0 / "base_valid_mask.png", result.valid_mask.astype(np.uint8) * 255)
            write_image(p0 / "owner_boundary_overlay.png", _owner_overlay(
                result.image, result.pixel_provenance["owner_frame_id"]
            ))
            write_npz(p0 / "base_pixel_provenance.npz", result.pixel_provenance)
            write_npz(p0 / "base_column_provenance.npz", result.column_provenance)
        if len(panel_images) > 1:
            gap_marker = np.zeros((panel_images[0].shape[0], 12, 3), dtype=np.uint8)
            gap_marker[:, :, 2] = 255
            overview_parts: list[np.ndarray] = []
            for panel_index, panel_image in enumerate(panel_images):
                if panel_index:
                    overview_parts.append(gap_marker)
                overview_parts.append(panel_image)
            write_image(p0 / "panel_navigation_overview_explicit_gaps.png", np.concatenate(overview_parts, axis=1))
        source_rows = [
            {
                "panel_index": panel_index,
                "assignment_index": assignment.assignment_index, "source_index": assignment.source_index,
                "frame_id": assignment.frame_id, "center_x": assignment.center_x, "left_x": assignment.left_x,
                "right_x": assignment.right_x, "width": assignment.width, "zero_width": assignment.zero_width,
                "placement_method": panel_selection.placement_methods[assignment.source_index],
                "risky": panel_selection.risk_by_frame_id.get(assignment.frame_id, False),
            }
            for panel_index, (panel_selection, panel_schedule) in enumerate(
                zip(schedule_plan.selections, schedule_plan.schedules, strict=True)
            )
            for assignment in panel_schedule.assignments
        ]
        write_csv(p0 / "base_sources.csv", source_rows, tuple(source_rows[0]))
        is_panel_set = len(schedule_plan.schedules) > 1
        render_state = "spatial_panel_set" if is_panel_set else "base_generated"
        panorama_claim = "none" if is_panel_set else (
            "pose_supported" if m3_layout.pose_supported else "visual_nonmetric"
        )
        report_statistics = _m31_report_statistics(motion, m3_layout, schedule_plan)
        layout = {
            "schema": "gemini305-video-s13-p0-layout/v1", "layout_level": m3_layout.layout_level,
            "render_state": render_state, "panorama_claim": panorama_claim,
            "metric_scale_claim": False, "pose_used_as_pixel_scale": False,
            "frame_ids": list(progress.frame_ids), "centers_x": list(progress.centers_x),
            "placement_methods": list(progress.placement_methods),
            "selected_frame_ids": [
                frame_id for panel_selection in schedule_plan.selections for frame_id in panel_selection.frame_ids
            ],
            "selected_centers_x": None if is_panel_set else list(selection.centers_x),
            "boundaries": None if is_panel_set else list(schedule.boundaries),
            "canvas_width": None if is_panel_set else schedule.canvas_width,
            "canvas_height": session.calibration.height,
            "pixels_per_meter_rgb_calibrated": m3_layout.pixels_per_meter,
            "pose_calibration_pairs": m3_layout.pose_calibration_pairs,
            "pose_calibration_reason": m3_layout.audit.get("pose_calibration_reason"),
            "pose_role": "direction_and_low_frequency_soft_prior_only",
            "canonical_scan_direction": m3_layout.canonical_scan_direction,
            "segment_break_pairs": [list(pair) for pair in m3_layout.segment_break_pairs],
            "motion_hypothesis_count_by_pair": {
                f"{edge.source_frame_id}:{edge.target_frame_id}": len(edge.motion_hypotheses) for edge in motion
            },
            "coherent_lineage": [asdict(step) for step in m3_layout.lineage],
            "source_u_fraction_cap": 0.20,
            "source_u_cap_px": schedule_plan.source_u_cap_px,
            "rescued_real_frame_ids": list(schedule_plan.rescued_frame_ids),
            "canvas_density_scale": schedule_plan.density_scale,
            "panel_count": len(schedule_plan.schedules),
            "panels": [
                {
                    "panel_index": index,
                    "frame_ids": list(panel_selection.frame_ids),
                    "centers_x": list(panel_selection.centers_x),
                    "canvas_width": panel_schedule.canvas_width,
                    "canvas_height": panel_schedule.canvas_height,
                    "artifact_root": f"panels/panel_{index:02d}" if is_panel_set else ".",
                }
                for index, (panel_selection, panel_schedule) in enumerate(
                    zip(schedule_plan.selections, schedule_plan.schedules, strict=True)
                )
            ],
            "panel_gap_frame_ids": [list(gap) for gap in schedule_plan.panel_gap_frame_ids],
            "source_u_cap_satisfied": schedule_plan.cap_satisfied,
            "panel_fallback_used": len(schedule_plan.schedules) > 1,
            "panel_overview_panorama_claim": "none" if len(schedule_plan.schedules) > 1 else "not_applicable",
            "midpoint_hard_owner": True, "transition_width_px": 0,
            "motion_graph_connected_telemetry": progress.adjacent_reliable_graph_connected,
            "motion_graph_connected_used_as_structural_gate": False,
            "direct_local_delta_used_as_structural_gate": False,
            "zero_duplicate_edge_count": zero_duplicate_edge_count,
            "subpixel_motion_edge_count": subpixel_motion_edge_count,
            "fallback_progress_edge_count": fallback_progress_edge_count,
            "observed_pause_expansion_count": pause_expansion_count,
            "observed_pause_expansion_px": pause_expansion_px,
            **report_statistics,
        }
        atomic_write_json(p0 / "base_layout.json", layout)
        atomic_write_json(staging / "motion_telemetry.json", {"edges": [_plain_edge(edge) for edge in motion]})
        atomic_write_json(staging / "motion_hypotheses.json", {
            "schema": "gemini305-video-s13-motion-hypotheses/v1",
            "maximum_hypotheses_per_pair": 3,
            "pairs": [_plain_edge(edge) for edge in motion],
        })
        atomic_write_json(staging / "coherent_lineage.json", {
            "schema": "gemini305-video-s13-coherent-lineage/v1",
            "selection": "whole_chain_dynamic_programming",
            "steps": [asdict(step) for step in m3_layout.lineage],
            "layout_audit": dict(m3_layout.audit),
        })
        stage_seconds["p0_render_and_export"] = time.perf_counter() - tick
        seal_p0(p0, generation_id=generation_id, metadata={
            "render_state": render_state, "panorama_claim": layout["panorama_claim"],
            "asset_semantics": "spatial_panel_set" if is_panel_set else "spatial_panorama",
            "trust_state": session.trust_state, "manual_review_required": True,
            "owner_policy": "fixed_midpoint", "remap_invocations": result.remap_invocations,
            "contributor_source_count": sum(
                not assignment.zero_width for panel_schedule in schedule_plan.schedules
                for assignment in panel_schedule.assignments
            ),
            "remap_invocations_all_panels": sum(item.remap_invocations for item in panel_results),
            "duplicate_write_count": sum(item.duplicate_write_count for item in panel_results),
        })
        atomic_write_json(staging / "report.json", {
            "schema": REPORT_SCHEMA, "generation_id": generation_id, "render_state": render_state,
            "panorama_claim": layout["panorama_claim"], "diagnostic_only": True, "production_eligible": False,
            "production_lock_eligible": False, "manual_review_required": True,
            "optimizer_state": (
                "all_failed_after_p0" if simulate_optimizer_all_fail
                else "m6_pending_after_p0" if run_m4 and run_m5 and run_m6 and not is_panel_set
                else "m5_pending_after_p0" if run_m4 and run_m5 and not is_panel_set
                else "m4_pending_after_p0" if run_m4 and not is_panel_set
                else "m3_complete_m4_not_started"
            ),
            "p0_parent": "P0/P0_completion.json", "raw_rgb_motion_pair_count": len(session.frames) - 1,
            "motion_graph_connected_telemetry": progress.adjacent_reliable_graph_connected,
            "motion_graph_disconnection_is_fatal": False, "direct_local_delta_is_fatal": False,
            "zero_duplicate_edge_count": layout["zero_duplicate_edge_count"],
            "subpixel_motion_edge_count": layout["subpixel_motion_edge_count"],
            "fallback_progress_edge_count": layout["fallback_progress_edge_count"],
            "observed_pause_expansion_count": layout["observed_pause_expansion_count"],
            "observed_pause_expansion_px": layout["observed_pause_expansion_px"],
            "trajectory": dict(trajectory.audit), "performance": {"stage_seconds": stage_seconds},
            **report_statistics,
        })
        publish_generation(staging, generation)
        pointer = update_current_base(
            root, generation, algorithm_id=configured_algorithm_id,
        )
        if not is_panel_set:
            update_current_latest(
                root, generation, stage="P0", completion_name="P0_completion.json",
                completion_schema="gemini305-video-s13-p0-completion/v1",
                result_asset="base_panorama_owner_only.png",
            )
        time_to_p0 = time.perf_counter() - run_started
        p0_hashes_before = dict(pointer["p0_assets_sha256"])
        # M2 has no optimizer. This injected path proves that any later optimizer
        # failure occurs strictly after the immutable completion and pointer.
        optimizer = _optimizer_all_fail_probe(simulate_optimizer_all_fail)
        p0_hashes_after = verify_p0_completion(generation / "P0")["assets_sha256"]
        optimizer["p0_hash_unchanged"] = p0_hashes_before == p0_hashes_after
        m4: dict[str, object] = {
            "state": "not_run",
            "reason": "optimizer_all_fail_probe" if simulate_optimizer_all_fail
            else "spatial_panel_set_requires_independent_panels" if is_panel_set
            else "disabled",
        }
        if run_m4 and not simulate_optimizer_all_fail and not is_panel_set:
            try:
                m4 = _run_m4_vertical(
                    generation, root, generation_id, schedule, session.calibration,
                    lambda frame_id: read_s13_rgb(by_id[frame_id]), result.image,
                )
            except Exception as exc:
                m4 = {
                    "state": "failed_p0_preserved",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                atomic_write_json(generation / "M4_failure.json", {
                    "schema": "gemini305-video-s13-m4-failure/v1",
                    "generation_id": generation_id,
                    "p0_parent_preserved": True,
                    **m4,
                })
        m5: dict[str, object] = {
            "state": "not_run",
            "reason": "optimizer_all_fail_probe" if simulate_optimizer_all_fail
            else "spatial_panel_set_requires_independent_panels" if is_panel_set
            else "m4_failed" if str(m4.get("state", "")).startswith("failed")
            else "disabled",
        }
        if (
            run_m4 and run_m5 and not simulate_optimizer_all_fail and not is_panel_set
            and m4.get("state") == "P1_sealed"
        ):
            try:
                m5 = _run_m5(
                    generation, root, generation_id, schedule, session.calibration,
                    lambda frame_id: read_s13_rgb(by_id[frame_id]), result.image,
                    selected_hypothesis_ids=tuple(
                        hypothesis_by_frame.get(frame_id, -1) for frame_id in selection.frame_ids
                    ),
                    placement_methods=selection.placement_methods,
                    p2_completion_schema=(
                        config.p2_completion_schema
                    ),
                    m6_eligible=(config.m6_eligible or manual_c2e_authorized),
                    m6_blocked_reason=(
                        "successor_p2_requires_fresh_four_branch_threshold_lineage"
                        if config.p2_only and not manual_c2e_authorized else None
                    ),
                    m61_evidence_config=m61_evidence_config,
                    m61_evidence_branch="formal",
                    m61_evidence_run_id=generation_id,
                    raw_rgb_sha256_by_frame=(
                        {
                            frame.frame_id: sha256_file(frame.color_path)
                            for frame in session.frames
                        }
                        if formal_m6 else None
                    ),
                    m51_r2_config=m51_r2_config,
                    candidate_identity={
                        "algorithm_id": configured_algorithm_id,
                        "implementation_id": config.document["implementation_id"],
                        "contract_schema": config.component["contract_schema"],
                        "config_sha256": algorithm_spec.config_sha256,
                        "candidate_manifest_sha256": algorithm_spec.candidate_manifest_sha256,
                        "candidate_config_sha256": algorithm_spec.config_sha256,
                        "s13_local_manifest_sha256": sha256_file(
                            config.path.parent / "candidate_manifest.json"
                        ),
                        "generation_manifest_sha256": sha256_file(
                            generation / "generation_manifest.json"
                        ),
                        "source_commit": algorithm_spec.source_commit,
                        "working_tree_dirty": algorithm_spec.working_tree_dirty,
                        "manual_c2e_forward_m7_authorized": (
                            manual_c2e_authorized
                        ),
                    },
                )
            except Exception as exc:
                m5 = {
                    "state": "failed_parent_preserved",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                atomic_write_json(generation / "M5_failure.json", {
                    "schema": "gemini305-video-s13-m5-failure/v1",
                    "generation_id": generation_id,
                    "p0_parent_preserved": True,
                    "p1_candidate_parent_preserved": True,
                    **m5,
                })
        m6: dict[str, object] = {
            "state": "not_run",
            "reason": "optimizer_all_fail_probe" if simulate_optimizer_all_fail
            else "spatial_panel_set_requires_independent_panels" if is_panel_set
            else "m5_failed" if str(m5.get("state", "")).startswith("failed")
            else "disabled",
        }
        if (
            run_m4 and run_m5 and run_m6 and not simulate_optimizer_all_fail
            and not is_panel_set and m5.get("state") == "P2_sealed"
        ):
            try:
                if formal_m6:
                    m6 = _run_formal_m6(
                        generation, root, session, config.path,
                    )
                else:
                    m6 = _run_m6(
                        generation, root, generation_id,
                        lambda frame_id: read_s13_rgb(by_id[frame_id]),
                        p2_completion_schema=config.p2_completion_schema,
                    )
            except Exception as exc:
                m6 = {
                    "state": "failed_p2_preserved",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                atomic_write_json(generation / "M6_failure.json", {
                    "schema": "gemini305-video-s13-m6-failure/v1",
                    "generation_id": generation_id,
                    "p2_parent_preserved": True,
                    **m6,
                })
        m7: dict[str, object] = {
            "state": "not_run",
            "reason": (
                "manual_c2e_forward_not_authorized"
                if not manual_c2e_authorized
                else "m6_failed"
            ),
        }
        if manual_c2e_authorized and m6.get("state") == "P3_sealed":
            try:
                m7 = _run_manual_c2e_m7(generation, root, generation_id)
            except Exception as exc:
                m7 = {
                    "state": "failed_p3_preserved",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                atomic_write_json(generation / "M7_failure.json", {
                    "schema": "gemini305-video-s13-m7-failure/v1",
                    "generation_id": generation_id,
                    "p3_parent_preserved": True,
                    **m7,
                })
        optimizer["p0_hash_unchanged_after_m5"] = (
            p0_hashes_before == verify_p0_completion(generation / "P0")["assets_sha256"]
        )
        if (generation / "P1" / "P1_completion.json").is_file():
            p1_document = json.loads(
                (generation / "P1" / "P1_completion.json").read_text(encoding="utf-8")
            )
            optimizer["p1_hash_unchanged_after_m5"] = all(
                (generation / "P1" / str(name)).is_file()
                and sha256_file(generation / "P1" / str(name)) == expected
                for name, expected in p1_document.get("assets_sha256", {}).items()
            )
        optimizer["p0_hash_unchanged_after_m4"] = (
            p0_hashes_before == verify_p0_completion(generation / "P0")["assets_sha256"]
        )
        optimizer_state = (
            "m7_sealed" if m7.get("state") == "P4_sealed"
            else "m7_failed" if str(m7.get("state", "")).startswith("failed")
            else "m6_sealed" if m6.get("state") == "P3_sealed"
            else "m6_failed" if str(m6.get("state", "")).startswith("failed")
            else "m5_sealed" if m5.get("state") == "P2_sealed"
            else "m5_failed" if str(m5.get("state", "")).startswith("failed")
            else "m4_sealed" if m4.get("state") == "P1_sealed"
            else "post_p0_not_run"
        )
        generation_report_path = generation / "report.json"
        generation_report = json.loads(generation_report_path.read_text(encoding="utf-8"))
        generation_report.update({
            "optimizer_state": optimizer_state,
            "m4": m4,
            "m5": m5,
            "m6": m6,
            "m7": m7,
            "optimizer": optimizer,
            "performance": {"stage_seconds": stage_seconds},
        })
        atomic_write_json(generation_report_path, generation_report)
        final_stage = (
            m7 if m7.get("state") == "P4_sealed"
            else m6 if m6.get("state") == "P3_sealed"
            else m5 if m5.get("state") == "P2_sealed"
            else m4 if m4.get("state") == "P1_sealed"
            else {}
        )
        # The historical v3 report keeps its P0 compatibility fields.  The
        # new v4 identity is the first contract whose primary result is the
        # latest sealed forward stage.
        primary_stage = final_stage if (
            formal_m6 or config.p2_only or manual_c2e_authorized
        ) else {}
        report = {
            "schema": REPORT_SCHEMA, "algorithm_id": configured_algorithm_id,
            "implementation_id": configured_implementation_id,
            "generation_id": generation_id,
            "generation": str(generation),
            "panorama": (
                None if is_panel_set
                else str(primary_stage.get("panorama", generation / "P0" / "base_panorama_owner_only.png"))
            ),
            "spatial_panel_set": str(generation / "P0" / "panels") if is_panel_set else None,
            "panel_navigation_overview": (
                str(generation / "P0" / "panel_navigation_overview_explicit_gaps.png") if is_panel_set else None
            ),
            "layout": str(generation / "P0" / "base_layout.json"),
            "owner_overlay": None if is_panel_set else str(generation / "P0" / "owner_boundary_overlay.png"),
            "pixel_provenance": (
                None if is_panel_set
                else str(primary_stage.get("pixel_provenance", generation / "P0" / "base_pixel_provenance.npz"))
            ),
            "column_provenance": None if is_panel_set else str(generation / "P0" / "base_column_provenance.npz"),
            "completion": str(primary_stage.get("completion", generation / "P0" / "P0_completion.json")),
            "final_stage": (
                "P4" if m7.get("state") == "P4_sealed"
                else "P3" if m6.get("state") == "P3_sealed"
                else "P2" if m5.get("state") == "P2_sealed"
                else "P1" if m4.get("state") == "P1_sealed"
                else "P0"
            ),
            "current_base": str(root / "current_base.json"), "time_to_P0_seconds": time_to_p0,
            "total_wall_seconds": time.perf_counter() - run_started, "stage_seconds": stage_seconds,
            "render_state": render_state, "panorama_claim": layout["panorama_claim"],
            "diagnostic_only": True, "production_eligible": False, "production_lock_eligible": False,
            "manual_review_required": True, "optimizer": optimizer,
            "optimizer_state": optimizer_state,
            "m4": m4,
            "m5": m5,
            "m6": m6,
            "m7": m7,
            "stage_acceptance": (
                m6.get("stage_acceptance") if formal_m6 else None
            ),
            "visual_acceptance_state": (
                m6.get("visual_acceptance_state") if formal_m6 else None
            ),
            "m7_handoff_eligible": (
                bool(m6.get("m7_handoff_eligible", False)) if formal_m6 else None
            ),
            "motion_graph_connected_telemetry": progress.adjacent_reliable_graph_connected,
            "motion_graph_disconnection_is_fatal": False, "direct_local_delta_is_fatal": False,
            **report_statistics,
        }
        atomic_write_json(root / "latest_run.json", report)
        return report
    except BaseException:
        if staging.exists():
            discard_staging(staging)
        raise


def candidate_config_sha(document: Mapping[str, Any]) -> str:
    from .video_algorithm import canonical_config_sha256

    return canonical_config_sha256(document)


__all__ = ["run_s13_experiment"]
