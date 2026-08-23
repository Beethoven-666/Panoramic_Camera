from __future__ import annotations

import copy
from pathlib import Path
from typing import Callable

import pytest
import yaml

from panorama_demo.video_algorithm import load_algorithm_config
from panorama_demo.video_algorithm_lock import read_algorithm_lock, verify_algorithm_lock
from panorama_demo.video_s13_contract import (
    S13_VISUAL_CONTINUITY_ALGORITHM_ID,
    S13_VISUAL_CONTINUITY_IMPLEMENTATION_ID,
)
from panorama_demo.video_s13_promotion import (
    S13_V11_CANDIDATE_CONFIG_SHA256,
    S13_V11_CANDIDATE_RELATIVE_PATH,
    S13_V11_CANDIDATE_SOURCE_COMMIT,
    S13V11PromotionError,
    canonical_s13_pixel_contract,
    s13_pixel_contract_sha256,
    verify_s13_v11_promotion,
)


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / S13_V11_CANDIDATE_RELATIVE_PATH


def _production_document() -> dict[str, object]:
    document = copy.deepcopy(load_algorithm_config(CANDIDATE))
    document.update(
        {
            "config_schema": "gemini305-video-algorithm/v1",
            "role": "production",
            "source_commit": "1" * 40,
            "config_sha256": "2" * 64,
            "promoted_from": {
                "candidate_path": S13_V11_CANDIDATE_RELATIVE_PATH,
                "candidate_id": S13_VISUAL_CONTINUITY_ALGORITHM_ID,
                "candidate_config_sha256": S13_V11_CANDIDATE_CONFIG_SHA256,
                "candidate_source_commit": S13_V11_CANDIDATE_SOURCE_COMMIT,
            },
        }
    )
    component = document["components"]["s013_output_first_progressive_dense_central_slit"]
    component["diagnostic_only"] = False
    component["production_eligible"] = True
    component["production_lock_eligible"] = True
    component["output"]["write_production_delivery"] = True
    component["output"]["write_all_stage_images"] = False
    component["runtime"]["stage_image_policy"] = "p3_only"
    component["runtime"]["write_performance_json"] = True
    return document


def _write_yaml(path: Path, document: dict[str, object]) -> Path:
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return path


def test_exact_v11_candidate_identity_and_contract_hash_are_frozen() -> None:
    document = load_algorithm_config(CANDIDATE)

    assert document["algorithm_id"] == S13_VISUAL_CONTINUITY_ALGORITHM_ID
    assert document["implementation_id"] == S13_VISUAL_CONTINUITY_IMPLEMENTATION_ID
    assert document["config_sha256"] == S13_V11_CANDIDATE_CONFIG_SHA256
    assert len(s13_pixel_contract_sha256(document)) == 64


def test_lifecycle_publication_and_observation_changes_preserve_pixel_contract(
    tmp_path: Path,
) -> None:
    candidate_document = load_algorithm_config(CANDIDATE)
    production_document = _production_document()
    production = _write_yaml(tmp_path / "production.yaml", production_document)

    verify_s13_v11_promotion(CANDIDATE, production)
    assert canonical_s13_pixel_contract(candidate_document) == (
        canonical_s13_pixel_contract(production_document)
    )
    assert s13_pixel_contract_sha256(candidate_document) == (
        s13_pixel_contract_sha256(production_document)
    )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda document: document["components"][
            "s013_output_first_progressive_dense_central_slit"
        ]["motion"].__setitem__("analysis_width_px", 423),
        lambda document: document["m5_seam_search"].__setitem__(
            "fallback_policy", "restore_parent_owner"
        ),
        lambda document: document["m63_photometric"]["q1r"].__setitem__(
            "maximum_gain", 1.09
        ),
        lambda document: document["components"][
            "s013_output_first_progressive_dense_central_slit"
        ].__setitem__("runtime_backend", "numpy_reference"),
        lambda document: document["components"][
            "s013_output_first_progressive_dense_central_slit"
        ]["runtime"].__setitem__("disk_resume", True),
    ],
)
def test_any_pixel_contract_change_rejects_promotion(
    tmp_path: Path, mutate: Callable[[dict[str, object]], None]
) -> None:
    production_document = _production_document()
    mutate(production_document)
    production = _write_yaml(tmp_path / "production.yaml", production_document)

    with pytest.raises(S13V11PromotionError, match="pixel contract differs"):
        verify_s13_v11_promotion(CANDIDATE, production)


def test_unknown_runtime_field_is_not_silently_treated_as_observation(
    tmp_path: Path,
) -> None:
    production_document = _production_document()
    component = production_document["components"][
        "s013_output_first_progressive_dense_central_slit"
    ]
    component["runtime"]["compute_shortcut"] = True
    production = _write_yaml(tmp_path / "production.yaml", production_document)

    with pytest.raises(S13V11PromotionError, match="pixel contract differs"):
        verify_s13_v11_promotion(CANDIDATE, production)


def test_promotion_provenance_and_no_fallback_are_exact(tmp_path: Path) -> None:
    for mutation in ("fallback", "provenance", "identity"):
        production_document = _production_document()
        if mutation == "fallback":
            production_document["allow_baseline_fallback"] = True
        elif mutation == "provenance":
            production_document["promoted_from"]["candidate_config_sha256"] = "0" * 64
        else:
            production_document["implementation_id"] = "s013_not_v11"
        production = _write_yaml(tmp_path / f"{mutation}.yaml", production_document)

        with pytest.raises(S13V11PromotionError):
            verify_s13_v11_promotion(CANDIDATE, production)


def test_candidate_declared_hash_cannot_hide_content_drift(tmp_path: Path) -> None:
    candidate_document = copy.deepcopy(load_algorithm_config(CANDIDATE))
    candidate_document["components"][
        "s013_output_first_progressive_dense_central_slit"
    ]["motion"]["analysis_width_px"] = 423
    candidate = _write_yaml(tmp_path / "candidate.yaml", candidate_document)
    production_document = _production_document()
    production_document["components"][
        "s013_output_first_progressive_dense_central_slit"
    ]["motion"]["analysis_width_px"] = 423
    production = _write_yaml(tmp_path / "production.yaml", production_document)

    with pytest.raises(S13V11PromotionError, match="candidate content changed"):
        verify_s13_v11_promotion(candidate, production)


def test_production_must_enable_eligible_delivery_lifecycle(tmp_path: Path) -> None:
    production_document = _production_document()
    production_document["components"][
        "s013_output_first_progressive_dense_central_slit"
    ]["production_lock_eligible"] = False
    production = _write_yaml(tmp_path / "production.yaml", production_document)

    with pytest.raises(S13V11PromotionError, match="lifecycle flags"):
        verify_s13_v11_promotion(CANDIDATE, production)


def test_repository_production_config_and_lock_are_exact_v11() -> None:
    production = ROOT / "configs/video_algorithms/s013_visual_continuity_v11_production.yaml"
    lock_path = production.with_suffix(".lock.json")

    verify_s13_v11_promotion(CANDIDATE, production)
    spec = verify_algorithm_lock(lock_path, expected_role="production")
    lock = read_algorithm_lock(lock_path, expected_role="production")

    assert spec.algorithm_id == S13_VISUAL_CONTINUITY_ALGORITHM_ID
    assert spec.implementation_id == S13_VISUAL_CONTINUITY_IMPLEMENTATION_ID
    assert spec.allow_baseline_fallback is False
    assert lock.dataset_lock_sha256 == (
        "42f989f5aa21863963beaa7fe8d6f9cf2df3c5243e766a49259735908c4e6397"
    )
    assert len(lock.dataset_lock_sha256) == 64
