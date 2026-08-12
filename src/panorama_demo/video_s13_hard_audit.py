"""Fail-closed structural audits for the isolated S1.3 M5 stage.

This module only validates hard-owner geometry and provenance.  Image-quality
scores, colour correction, blending, depth completion, and later stages are
deliberately outside its scope.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class S13HardAuditConfig:
    """Versioned thresholds for M5 structural catastrophe protection."""

    schema: str = "gemini305-video-s13-m51-hard-audit-config/v1"
    maximum_seam_shift_px: int = 8
    minimum_owner_width_px: int = 1
    maximum_absolute_horizontal_lag_px: int = 2
    maximum_horizontal_lag_increase_px: int = 2
    maximum_zero_lag_correlation_drop: float = 0.20
    support_loss_check_minimum_rows: int = 64
    maximum_support_loss_fraction: float = 0.50
    minimum_catastrophe_support_rows: int = 8

    def validate(self) -> None:
        if self.maximum_seam_shift_px < 0:
            raise ValueError("maximum_seam_shift_px must be non-negative")
        if self.minimum_owner_width_px < 1:
            raise ValueError("minimum_owner_width_px must be positive")
        if self.maximum_absolute_horizontal_lag_px < 1:
            raise ValueError("maximum_absolute_horizontal_lag_px must be positive")
        if self.maximum_horizontal_lag_increase_px < 1:
            raise ValueError("maximum_horizontal_lag_increase_px must be positive")
        if not 0.0 <= self.maximum_zero_lag_correlation_drop <= 2.0:
            raise ValueError("maximum_zero_lag_correlation_drop must be in [0, 2]")
        if self.support_loss_check_minimum_rows < 1:
            raise ValueError("support_loss_check_minimum_rows must be positive")
        if not 0.0 <= self.maximum_support_loss_fraction < 1.0:
            raise ValueError("maximum_support_loss_fraction must be in [0, 1)")
        if self.minimum_catastrophe_support_rows < 1:
            raise ValueError("minimum_catastrophe_support_rows must be positive")


def _sha_array(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(str(value.shape).encode("ascii"))
    digest.update(value.tobytes())
    return digest.hexdigest()


def _finite_number(value: object) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if np.isfinite(result) else None


def long_horizontal_structure_catastrophe_guard(
    before: Mapping[str, object],
    after: Mapping[str, object],
    *,
    config: S13HardAuditConfig | None = None,
) -> tuple[bool, str | None, dict[str, object]]:
    """Reject only conspicuous long-horizontal-structure catastrophes.

    Small changes remain diagnostic.  The guard activates only when either
    side has sufficient measured support, avoiding hard decisions from sparse
    low-texture noise.
    """

    settings = config or S13HardAuditConfig()
    settings.validate()
    before_support = int(before.get("support_row_count", 0) or 0)
    after_support = int(after.get("support_row_count", 0) or 0)
    before_observed = before.get("observed") is True
    after_observed = after.get("observed") is True
    sufficient = max(before_support, after_support) >= settings.minimum_catastrophe_support_rows
    failures: list[str] = []

    before_lag = _finite_number(before.get("absolute_best_vertical_lag_px"))
    after_lag = _finite_number(after.get("absolute_best_vertical_lag_px"))
    before_zero = _finite_number(before.get("zero_lag_correlation"))
    after_zero = _finite_number(after.get("zero_lag_correlation"))

    if sufficient:
        if before_observed and not after_observed:
            failures.append("horizontal_structure_support_lost")
        if after_observed and after_lag is not None:
            if after_lag >= settings.maximum_absolute_horizontal_lag_px:
                failures.append("horizontal_structure_absolute_lag_catastrophe")
            if (
                before_observed
                and before_lag is not None
                and after_lag - before_lag
                >= settings.maximum_horizontal_lag_increase_px
            ):
                failures.append("horizontal_structure_lag_increase_catastrophe")
        if (
            before_observed
            and after_observed
            and before_zero is not None
            and after_zero is not None
            and before_zero - after_zero
            > settings.maximum_zero_lag_correlation_drop
        ):
            failures.append("horizontal_structure_zero_lag_correlation_drop")
        if before_observed and before_support >= settings.support_loss_check_minimum_rows:
            retained_fraction = after_support / max(before_support, 1)
            if retained_fraction < 1.0 - settings.maximum_support_loss_fraction:
                failures.append("horizontal_structure_support_catastrophe")

    passed = not failures
    reason = None if passed else failures[0]
    return passed, reason, {
        "schema": "gemini305-video-s13-horizontal-catastrophe-audit/v1",
        "passed": passed,
        "reason": reason,
        "failures": failures,
        "support_sufficient": sufficient,
        "before_observed": before_observed,
        "after_observed": after_observed,
        "before_support_row_count": before_support,
        "after_support_row_count": after_support,
        "before_absolute_lag_px": before_lag,
        "after_absolute_lag_px": after_lag,
        "zero_lag_correlation_drop": (
            None if before_zero is None or after_zero is None else before_zero - after_zero
        ),
        "config": asdict(settings),
    }


def audit_s13_seam_path(
    seam_x_by_row: np.ndarray,
    *,
    canvas_height: int,
    base_boundary_x: int,
    search_left_x: int | np.ndarray,
    search_right_x: int | np.ndarray,
    previous_seam_x_by_row: np.ndarray | None = None,
    next_seam_x_by_row: np.ndarray | None = None,
    config: S13HardAuditConfig | None = None,
) -> dict[str, object]:
    """Audit one seam without silently rounding invalid coordinates."""

    settings = config or S13HardAuditConfig()
    settings.validate()
    raw = np.asarray(seam_x_by_row)
    failures: list[str] = []
    shape_ok = raw.shape == (int(canvas_height),)
    if not shape_ok:
        failures.append("seam_shape_invalid")
    finite = bool(np.isfinite(raw).all()) if np.issubdtype(raw.dtype, np.number) else False
    if not finite:
        failures.append("seam_coordinates_nonfinite")
    integral = bool(finite and np.equal(raw, np.rint(raw)).all())
    if finite and not integral:
        failures.append("seam_coordinates_nonintegral")
    seam = np.rint(raw).astype(np.int64) if shape_ok and finite else np.empty(0, np.int64)

    left = np.broadcast_to(np.asarray(search_left_x, dtype=np.int64), (int(canvas_height),))
    right = np.broadcast_to(np.asarray(search_right_x, dtype=np.int64), (int(canvas_height),))
    bounds_valid = bool(np.all(left <= right))
    if not bounds_valid:
        failures.append("seam_search_bounds_invalid")
    in_bounds = bool(seam.size and np.all((seam >= left) & (seam <= right)))
    if shape_ok and integral and not in_bounds:
        failures.append("seam_outside_search_bounds")
    row_steps_valid = bool(seam.size and np.all(np.abs(np.diff(seam)) <= 1))
    if shape_ok and integral and not row_steps_valid:
        failures.append("seam_row_step_invalid")
    maximum_shift = int(np.max(np.abs(seam - int(base_boundary_x)))) if seam.size else None
    if maximum_shift is not None and maximum_shift > settings.maximum_seam_shift_px:
        failures.append("seam_maximum_shift_exceeded")

    if seam.size and previous_seam_x_by_row is not None:
        previous = np.asarray(previous_seam_x_by_row)
        if previous.shape != seam.shape or not np.isfinite(previous).all():
            failures.append("previous_seam_invalid")
        elif np.any(seam - previous < settings.minimum_owner_width_px):
            failures.append("seam_crosses_or_collapses_previous_owner")
    if seam.size and next_seam_x_by_row is not None:
        following = np.asarray(next_seam_x_by_row)
        if following.shape != seam.shape or not np.isfinite(following).all():
            failures.append("next_seam_invalid")
        elif np.any(following - seam < settings.minimum_owner_width_px):
            failures.append("seam_crosses_or_collapses_next_owner")

    return {
        "schema": "gemini305-video-s13-seam-path-hard-audit/v1",
        "passed": not failures,
        "failures": failures,
        "seam_sha256": _sha_array(raw),
        "shape_valid": shape_ok,
        "finite": finite,
        "integral_coordinates": integral,
        "search_bounds_valid": bounds_valid,
        "in_search_bounds": in_bounds,
        "allowed_row_steps_only": row_steps_valid,
        "maximum_shift_px": maximum_shift,
        "config": asdict(settings),
    }


def audit_s13_seam_family(
    seams_x_by_row: np.ndarray,
    *,
    assignment_count: int,
    canvas_height: int,
    canvas_width: int,
    config: S13HardAuditConfig | None = None,
) -> dict[str, object]:
    """Audit ordered seams and require every hard-owner region to be nonempty."""

    settings = config or S13HardAuditConfig()
    settings.validate()
    raw = np.asarray(seams_x_by_row)
    expected_shape = (max(0, int(assignment_count) - 1), int(canvas_height))
    failures: list[str] = []
    shape_ok = raw.shape == expected_shape
    if not shape_ok:
        failures.append("seam_family_shape_invalid")
    finite = bool(np.issubdtype(raw.dtype, np.number) and np.isfinite(raw).all())
    if not finite:
        failures.append("seam_family_nonfinite")
    integral = bool(finite and np.equal(raw, np.rint(raw)).all())
    if finite and not integral:
        failures.append("seam_family_nonintegral")
    seams = np.rint(raw).astype(np.int64) if shape_ok and finite else np.empty(expected_shape, np.int64)
    in_canvas = bool(
        seams.size == 0
        or np.all((seams >= settings.minimum_owner_width_px)
                  & (seams <= int(canvas_width) - settings.minimum_owner_width_px))
    )
    if shape_ok and integral and not in_canvas:
        failures.append("seam_family_outside_owner_domain")
    ordered = bool(seams.shape[0] < 2 or np.all(np.diff(seams, axis=0) >= settings.minimum_owner_width_px))
    if shape_ok and integral and not ordered:
        failures.append("seam_family_crossing_or_zero_width_owner")
    row_steps = bool(seams.size == 0 or np.all(np.abs(np.diff(seams, axis=1)) <= 1))
    if shape_ok and integral and not row_steps:
        failures.append("seam_family_row_step_invalid")

    if shape_ok and integral:
        boundaries = np.concatenate(
            (
                np.zeros((1, int(canvas_height)), dtype=np.int64),
                seams,
                np.full((1, int(canvas_height)), int(canvas_width), dtype=np.int64),
            ),
            axis=0,
        )
        owner_widths = np.diff(boundaries, axis=0)
        minimum_width = int(owner_widths.min()) if owner_widths.size else int(canvas_width)
    else:
        minimum_width = None
    if minimum_width is not None and minimum_width < settings.minimum_owner_width_px:
        failures.append("owner_width_below_minimum")

    return {
        "schema": "gemini305-video-s13-seam-family-hard-audit/v1",
        "passed": not failures,
        "failures": failures,
        "seam_family_sha256": _sha_array(raw),
        "expected_shape": list(expected_shape),
        "actual_shape": list(raw.shape),
        "finite": finite,
        "integral_coordinates": integral,
        "in_canvas": in_canvas,
        "strictly_ordered": ordered,
        "allowed_row_steps_only": row_steps,
        "minimum_owner_width_px": minimum_width,
        "config": asdict(settings),
    }


def _internal_hole_audit(expected: np.ndarray, valid: np.ndarray) -> dict[str, object]:
    hole = np.asarray(expected, dtype=bool) & ~np.asarray(valid, dtype=bool)
    invalid = (~np.asarray(valid, dtype=bool)).astype(np.uint8)
    component_count, labels = cv2.connectedComponents(invalid, connectivity=8)
    external_labels: set[int] = set()
    if labels.size:
        external_labels.update(int(value) for value in labels[0, :])
        external_labels.update(int(value) for value in labels[-1, :])
        external_labels.update(int(value) for value in labels[:, 0])
        external_labels.update(int(value) for value in labels[:, -1])
    internal = hole.copy()
    for label in external_labels:
        if label != 0:
            internal[labels == label] = False
    internal_labels = set(int(value) for value in np.unique(labels[internal]) if value != 0)
    return {
        "passed": not np.any(internal),
        "hole_pixel_count": int(hole.sum()),
        "internal_hole_pixel_count": int(internal.sum()),
        "internal_hole_component_count": len(internal_labels),
        "invalid_component_count": int(max(0, component_count - 1)),
        "external_invalid_component_count": len(external_labels - {0}),
    }


def audit_s13_p2_stage(
    *,
    valid_mask: np.ndarray,
    pixel_provenance: Mapping[str, np.ndarray],
    seams_x_by_row: np.ndarray,
    assignment_frame_ids: Sequence[int],
    source_sizes: Sequence[tuple[int, int]],
    pair_transaction_count: int,
    expected_support_mask: np.ndarray | None,
    config: S13HardAuditConfig | None = None,
) -> dict[str, object]:
    """Audit a rendered owner-only P2 before it can be sealed."""

    settings = config or S13HardAuditConfig()
    settings.validate()
    valid = np.asarray(valid_mask, dtype=bool)
    failures: list[str] = []
    if valid.ndim != 2:
        raise ValueError("valid_mask must be HxW")
    height, width = valid.shape
    assignments = tuple(int(value) for value in assignment_frame_ids)
    if len(assignments) < 1 or len(source_sizes) != len(assignments):
        raise ValueError("assignment_frame_ids and source_sizes must be nonempty and aligned")
    if int(pair_transaction_count) != len(assignments) - 1:
        failures.append("pair_transaction_count_mismatch")

    required = {
        "owner_frame_id", "owner_source_index", "source_u", "source_v", "valid",
        "geometry_transaction_id", "seam_transaction_id", "secondary_frame_id",
        "secondary_weight",
    }
    missing = sorted(required - set(pixel_provenance))
    if missing:
        failures.append("pixel_provenance_fields_missing")
    shape_failures: list[str] = []
    for name in required & set(pixel_provenance):
        if np.asarray(pixel_provenance[name]).shape != valid.shape:
            shape_failures.append(name)
    if shape_failures:
        failures.append("pixel_provenance_shape_mismatch")

    topology: dict[str, object] = {"missing_fields": missing, "shape_mismatch_fields": shape_failures}
    if not missing and not shape_failures:
        owner = np.asarray(pixel_provenance["owner_frame_id"])
        owner_index = np.asarray(pixel_provenance["owner_source_index"])
        provenance_valid = np.asarray(pixel_provenance["valid"], dtype=bool)
        source_u = np.asarray(pixel_provenance["source_u"])
        source_v = np.asarray(pixel_provenance["source_v"])
        geometry_tx = np.asarray(pixel_provenance["geometry_transaction_id"])
        seam_tx = np.asarray(pixel_provenance["seam_transaction_id"])
        valid_owner_consistent = bool(np.array_equal(valid, owner >= 0))
        provenance_valid_consistent = bool(np.array_equal(valid, provenance_valid))
        owner_index_valid = bool(
            np.all((owner_index[valid] >= 0) & (owner_index[valid] < len(assignments)))
            and np.all(owner_index[~valid] == -1)
        )
        owner_frame_consistent = bool(
            owner_index_valid
            and all(
                np.all(owner[valid & (owner_index == index)] == frame_id)
                for index, frame_id in enumerate(assignments)
            )
            and np.all(owner[~valid] == -1)
        )
        source_finite = bool(np.isfinite(source_u[valid]).all() and np.isfinite(source_v[valid]).all())
        source_bounds = True
        if owner_index_valid and source_finite:
            for index, (source_width, source_height) in enumerate(source_sizes):
                owned = valid & (owner_index == index)
                source_bounds = source_bounds and bool(
                    np.all((source_u[owned] >= 0.0) & (source_u[owned] <= int(source_width) - 1))
                    and np.all((source_v[owned] >= 0.0) & (source_v[owned] <= int(source_height) - 1))
                )
        else:
            source_bounds = False
        tx_valid = bool(
            np.all((geometry_tx[valid] >= -1) & (geometry_tx[valid] < pair_transaction_count))
            and np.all((seam_tx[valid] >= -1) & (seam_tx[valid] < pair_transaction_count))
            and np.all(geometry_tx[~valid] == -1)
            and np.all(seam_tx[~valid] == -1)
        )
        tx_owner_consistent = bool(
            tx_valid
            and np.all(geometry_tx[valid & (owner_index == 0)] == -1)
            and np.all(seam_tx[valid & (owner_index == 0)] == -1)
            and all(
                np.all(geometry_tx[valid & (owner_index == index)] == index - 1)
                and np.all(seam_tx[valid & (owner_index == index)] == index - 1)
                for index in range(1, len(assignments))
            )
        )
        secondary_owner_only = bool(
            np.all(np.asarray(pixel_provenance["secondary_frame_id"]) == -1)
            and np.all(np.asarray(pixel_provenance["secondary_weight"]) == 0.0)
        )
        checks = {
            "valid_owner_consistent": valid_owner_consistent,
            "provenance_valid_consistent": provenance_valid_consistent,
            "owner_source_index_valid": owner_index_valid,
            "owner_frame_consistent": owner_frame_consistent,
            "valid_source_coordinates_finite": source_finite,
            "valid_source_coordinates_in_bounds": source_bounds,
            "transaction_ids_valid": tx_valid,
            "transaction_ids_owner_consistent": tx_owner_consistent,
            "secondary_owner_only": secondary_owner_only,
        }
        topology.update(checks)
        failures.extend(name for name, passed in checks.items() if not passed)

    family = audit_s13_seam_family(
        seams_x_by_row,
        assignment_count=len(assignments),
        canvas_height=height,
        canvas_width=width,
        config=settings,
    )
    if family["passed"] is not True:
        failures.append("seam_family_hard_audit_failed")

    if expected_support_mask is None:
        holes = {"passed": False, "reason": "expected_support_mask_missing"}
        failures.append("expected_support_mask_missing")
    else:
        expected = np.asarray(expected_support_mask, dtype=bool)
        if expected.shape != valid.shape:
            holes = {"passed": False, "reason": "expected_support_mask_shape_mismatch"}
            failures.append("expected_support_mask_shape_mismatch")
        else:
            holes = _internal_hole_audit(expected, valid)
            if holes["passed"] is not True:
                failures.append("internal_holes_present")

    return {
        "schema": "gemini305-video-s13-p2-hard-audit/v1",
        "passed": not failures,
        "fatal_failures": failures,
        "owner_and_provenance": topology,
        "seam_family": family,
        "internal_holes": holes,
        "assignment_count": len(assignments),
        "pair_transaction_count": int(pair_transaction_count),
        "config": asdict(settings),
    }


__all__ = [
    "S13HardAuditConfig",
    "audit_s13_p2_stage",
    "audit_s13_seam_family",
    "audit_s13_seam_path",
    "long_horizontal_structure_catastrophe_guard",
]
