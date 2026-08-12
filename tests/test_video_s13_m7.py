from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from panorama_demo.video_algorithm import load_algorithm_config
from panorama_demo.video_s13_contract import validate_s13_document
from panorama_demo.video_s13_p4_hard_audit import audit_s13_p4_arrays
from panorama_demo.video_s13_repair import (
    CORE_CANDIDATES,
    R1_B1_TO_B0,
    R2_SEAM_DOWNGRADE,
    R3_GEOMETRY_DOWNGRADE,
    S13CoreCandidate,
    r1_b1_to_b0_owner_only,
    validate_core_registry,
    validate_replayed_r2_candidate,
    validate_replayed_r3_candidate,
)


def _provenance() -> dict[str, np.ndarray]:
    shape = (3, 5)
    valid = np.ones(shape, bool)
    owner = np.zeros(shape, np.int32)
    owner[:, 3:] = 1
    secondary = np.full(shape, -1, np.int32)
    secondary[:, 2] = 1
    weight = np.zeros(shape, np.float32)
    weight[:, 2] = 0.25
    values: dict[str, np.ndarray] = {
        "owner_frame_id": owner.copy(),
        "owner_source_index": owner,
        "assignment_index": owner.copy(),
        "source_u": np.broadcast_to(np.arange(5, dtype=np.float32), shape).copy(),
        "source_v": np.broadcast_to(np.arange(3, dtype=np.float32)[:, None], shape).copy(),
        "valid": valid,
        "selected_motion_hypothesis_id": np.zeros(shape, np.int32),
        "placement_method_code": np.zeros(shape, np.int16),
        "geometry_transaction_id": np.zeros(shape, np.int32),
        "seam_transaction_id": np.zeros(shape, np.int32),
        "photometric_transaction_id": owner.copy(),
        "blend_transaction_id": np.where(weight > 0, 0, -1).astype(np.int32),
        "secondary_frame_id": np.where(weight > 0, 1, -1).astype(np.int32),
        "secondary_source_index": secondary,
        "secondary_source_u": np.where(weight > 0, 3.0, np.nan).astype(np.float32),
        "secondary_source_v": np.where(weight > 0, 1.0, np.nan).astype(np.float32),
        "secondary_weight": weight,
    }
    return values


def test_r1_only_turns_exact_frozen_b1_pair_into_b0() -> None:
    provenance = _provenance()
    image = np.full((3, 5, 3), 40, np.uint8)
    image[:, 2] = 90
    owner_only = image.copy()
    owner_only[:, 2] = 20
    affected = provenance["secondary_weight"] > 0
    result = r1_b1_to_b0_owner_only(
        image, owner_only, provenance, affected, pair_index=0,
    )
    assert result.model == R1_B1_TO_B0
    assert np.array_equal(result.image[affected], owner_only[affected])
    assert np.array_equal(result.image[~affected], image[~affected])
    assert np.all(result.provenance["secondary_weight"][affected] == 0)
    assert np.all(result.provenance["secondary_source_index"][affected] == -1)
    assert np.all(result.provenance["blend_transaction_id"][affected] == -1)
    assert np.array_equal(provenance["secondary_weight"], _provenance()["secondary_weight"])


def test_r1_rejects_roi_expansion_and_non_pair_masks() -> None:
    provenance = _provenance()
    image = np.zeros((3, 5, 3), np.uint8)
    expanded = provenance["secondary_weight"] > 0
    expanded[0, 1] = True
    with pytest.raises(ValueError, match="exact frozen B1 pair mask"):
        r1_b1_to_b0_owner_only(image, image, provenance, expanded, pair_index=0)


def test_core_registry_is_exact_and_contains_no_optional_candidate() -> None:
    validate_core_registry(CORE_CANDIDATES)
    with pytest.raises(ValueError, match="exactly ordered R0-R3"):
        validate_core_registry((*CORE_CANDIDATES, "constrained_two_label_graphcut"))


def test_r2_requires_new_geometry_from_p0_for_the_downgraded_seam() -> None:
    candidate = S13CoreCandidate(
        model=R2_SEAM_DOWNGRADE,
        image=np.zeros((1, 1, 3), np.uint8),
        provenance={},
        audit={
            "seam_direction": "S2_to_S1",
            "geometry_reestimated_from_immutable_p0": True,
            "final_corridor_geometry_reestimated": True,
            "q_parameters_frozen": True,
            "blend_model": "B0_owner_only",
            "thresholds_unchanged": True,
            "source_set_unchanged": True,
        },
    )
    validate_replayed_r2_candidate(candidate)
    invalid = S13CoreCandidate(candidate.model, candidate.image, {}, {**candidate.audit, "geometry_reestimated_from_immutable_p0": False})
    with pytest.raises(ValueError, match="R2 replay"):
        validate_replayed_r2_candidate(invalid)


def test_r3_requires_seam_reselection_for_simpler_geometry() -> None:
    candidate = S13CoreCandidate(
        model=R3_GEOMETRY_DOWNGRADE,
        image=np.zeros((1, 1, 3), np.uint8),
        provenance={},
        audit={
            "geometry_complexity_decreased": True,
            "geometry_replayed_from_immutable_p0": True,
            "seam_reselected_for_candidate_geometry": True,
            "q_parameters_frozen": True,
            "blend_model": "B0_owner_only",
            "thresholds_unchanged": True,
            "source_set_unchanged": True,
        },
    )
    validate_replayed_r3_candidate(candidate)
    invalid = S13CoreCandidate(candidate.model, candidate.image, {}, {**candidate.audit, "seam_reselected_for_candidate_geometry": False})
    with pytest.raises(ValueError, match="R3 replay"):
        validate_replayed_r3_candidate(invalid)


def test_p4_audit_rejects_noop_and_outside_component_changes() -> None:
    p3 = np.zeros((3, 5, 3), np.uint8)
    provenance = _provenance()
    affected = provenance["secondary_weight"] > 0
    p4_provenance = {name: value.copy() for name, value in provenance.items()}
    for name in (
        "m7_handoff_item_id", "repair_component_id", "repair_transaction_id",
        "repair_model_code", "repair_parent_stage_code",
    ):
        p4_provenance[name] = np.full(affected.shape, -1, np.int32)
        p4_provenance[name][affected] = 0
    noop = audit_s13_p4_arrays(
        p3, p3.copy(), provenance, p4_provenance, affected,
        selected_models=[R1_B1_TO_B0], expected_source_count=2,
        authorization_sha256="a" * 64, handoff_sha256="b" * 64,
    )
    assert "no_op_p4_forbidden" in noop["failures"]
    changed = p3.copy()
    changed[0, 0] = 1
    outside = audit_s13_p4_arrays(
        p3, changed, provenance, p4_provenance, affected,
        selected_models=[R1_B1_TO_B0], expected_source_count=2,
        authorization_sha256="a" * 64, handoff_sha256="b" * 64,
    )
    assert "image_changed_outside_handoff_mask" in outside["failures"]


def test_formal_config_rejects_enabling_any_m7_optional() -> None:
    path = Path("configs/video_candidates/s013/S013_output_first_progressive_dense_central_slit_v4.yaml")
    document = load_algorithm_config(path)
    component = document["components"]["s013_output_first_progressive_dense_central_slit"]
    for key in ("graphcut_enabled", "depth_risk_enabled", "bounded_mesh_enabled"):
        changed = deepcopy(document)
        changed["components"]["s013_output_first_progressive_dense_central_slit"]["repair"][key] = True
        with pytest.raises(ValueError, match="frozen/disabled"):
            validate_s13_document(changed, path=path)
    assert component["repair"]["q_parameters_frozen"] is True
