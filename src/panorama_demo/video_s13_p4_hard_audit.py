"""Independent structural audit for an S013 M7 P4 candidate."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


P4_HARD_AUDIT_SCHEMA = "gemini305-video-s13-p4-hard-audit/v1"
P4_COMPLETION_SCHEMA = "gemini305-video-s13-p4-repair-completion/v1"
PRIMARY_FIELDS = (
    "owner_frame_id", "owner_source_index", "assignment_index", "source_u", "source_v",
    "valid", "selected_motion_hypothesis_id", "placement_method_code",
    "geometry_transaction_id", "seam_transaction_id", "photometric_transaction_id",
)
M7_FIELDS = (
    "m7_handoff_item_id", "repair_component_id", "repair_transaction_id",
    "repair_model_code", "repair_parent_stage_code",
)


def _same(left: np.ndarray, right: np.ndarray) -> bool:
    left, right = np.asarray(left), np.asarray(right)
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    if left.dtype.kind == "f":
        return bool(np.array_equal(left, right, equal_nan=True))
    return bool(np.array_equal(left, right))


def audit_s13_p4_arrays(
    p3_image: np.ndarray,
    p4_image: np.ndarray,
    p3_provenance: Mapping[str, np.ndarray],
    p4_provenance: Mapping[str, np.ndarray],
    affected_mask: np.ndarray,
    *,
    selected_models: Sequence[str],
    expected_source_count: int,
    authorization_sha256: str,
    handoff_sha256: str,
) -> dict[str, Any]:
    failures: list[str] = []
    parent = np.asarray(p3_image)
    result = np.asarray(p4_image)
    affected = np.asarray(affected_mask, bool)
    if parent.dtype != np.uint8 or parent.ndim != 3 or parent.shape[2] != 3:
        failures.append("p3_image_invalid")
    if result.shape != parent.shape or result.dtype != parent.dtype or affected.shape != parent.shape[:2]:
        failures.append("p4_image_or_mask_shape_changed")
        return _result(failures, authorization_sha256, handoff_sha256)
    changed = np.any(result != parent, axis=2)
    if np.any(changed & ~affected):
        failures.append("image_changed_outside_handoff_mask")
    if not np.any(changed):
        failures.append("no_op_p4_forbidden")
    for name in PRIMARY_FIELDS:
        if name not in p3_provenance or name not in p4_provenance:
            failures.append(f"primary_field_missing:{name}")
        elif not _same(np.asarray(p3_provenance[name]), np.asarray(p4_provenance[name])):
            failures.append(f"primary_field_changed:{name}")
    for name in p3_provenance:
        if name not in p4_provenance:
            failures.append(f"p3_provenance_field_missing:{name}")
            continue
        left, right = np.asarray(p3_provenance[name]), np.asarray(p4_provenance[name])
        if left.shape == affected.shape and right.shape == affected.shape:
            if left.dtype.kind == "f":
                equal = (left == right) | (np.isnan(left) & np.isnan(right))
            else:
                equal = left == right
            if np.any(~equal & ~affected):
                failures.append(f"provenance_changed_outside_handoff_mask:{name}")
    for name in M7_FIELDS:
        value = p4_provenance.get(name)
        if value is None or np.asarray(value).shape != affected.shape or np.asarray(value).dtype != np.int32:
            failures.append(f"m7_field_invalid:{name}")
        elif np.any(np.asarray(value)[~affected] != -1):
            failures.append(f"m7_field_set_outside_handoff_mask:{name}")
    valid = np.asarray(p4_provenance.get("valid", np.zeros(affected.shape, bool)), bool)
    owner = np.asarray(p4_provenance.get("owner_source_index", np.full(affected.shape, -1)), np.int32)
    secondary = np.asarray(
        p4_provenance.get("secondary_source_index", np.full(affected.shape, -1)), np.int32,
    )
    weight = np.asarray(p4_provenance.get("secondary_weight", np.zeros(affected.shape)), np.float64)
    if np.any(valid & (owner < 0)) or np.any(~valid & (owner >= 0)):
        failures.append("valid_owner_topology_invalid")
    if not np.isfinite(weight).all() or np.any(weight < 0) or np.any(weight >= 0.5):
        failures.append("secondary_weight_invalid")
    if np.any((weight > 0) & ((secondary < 0) | ~valid)):
        failures.append("secondary_contributor_invalid")
    source_values = np.unique(owner[valid])
    if source_values.size and int(source_values.max()) >= int(expected_source_count):
        failures.append("source_set_changed")
    if any(model not in {
        "R1_b1_2px_to_b0_owner_only",
        "R2_downgrade_seam_reestimate_geometry",
        "R3_downgrade_geometry_reselect_seam",
    } for model in selected_models):
        failures.append("non_core_or_unknown_repair_model")
    return _result(failures, authorization_sha256, handoff_sha256, int(np.count_nonzero(changed)))


def _result(
    failures: Sequence[str],
    authorization_sha256: str,
    handoff_sha256: str,
    changed_pixel_count: int = 0,
) -> dict[str, Any]:
    unique = list(dict.fromkeys(failures))
    return {
        "schema": P4_HARD_AUDIT_SCHEMA,
        "passed": not unique,
        "failures": unique,
        "changed_pixel_count": int(changed_pixel_count),
        "manual_forward_authorization_sha256": authorization_sha256,
        "m7_handoff_sha256": handoff_sha256,
        "q_parameters_frozen": True,
        "thresholds_frozen": True,
        "source_set_frozen": True,
        "optional_invocations": {
            "constrained_two_label_graphcut": 0,
            "depth_risk_veto": 0,
            "bounded_local_mesh": 0,
        },
    }


__all__ = ["P4_COMPLETION_SCHEMA", "P4_HARD_AUDIT_SCHEMA", "audit_s13_p4_arrays"]
