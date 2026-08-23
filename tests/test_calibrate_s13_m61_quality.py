from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from panorama_demo.video_s13_m61_quality import (
    BASELINE_INPUT_SCHEMA,
    EXPECTED_BRANCHES,
    calibrate_m61_quality,
    load_baseline_input,
    write_proposed_calibration,
)


def _input(branch: str, *, model: str = "Q0_identity", blend: str = "B0_owner_only") -> dict[str, object]:
    pairs = []
    for index in range(2):
        pairs.append({
            "pair_index": index,
            "classification": "quality_safe_evaluable",
            "validation_sample_count": 256,
            "independent_validation_tile_count": 8,
            "vertical_block_count": 4,
            "train_validation_overlap_count": 0,
            "metrics": {
                "aggregate_linear_residual_median": 0.01 + index * 0.001,
                "aggregate_linear_residual_p95": 0.02 + index * 0.001,
                "pair_linear_residual_p95": 0.025 + index * 0.001,
                "delta_e00": 1.0 + index,
                "immediate_seam": 0.5,
                "gradient_ghost": 0.1,
                "owner_plateau": 0.8,
                "clipping_fraction": 0.001,
            },
            "calibration_replicates": {"aggregate_linear_residual_p95": [0.0202 + index * 0.001]},
        })
    return {
        "schema": BASELINE_INPUT_SCHEMA,
        "sealed": True,
        "branch": branch,
        "run_id": f"run-{branch}",
        "p2_completion_sha256": "a" * 64,
        "p2_canonical_image_sha256": "b" * 64,
        "graph_topology_sha256": "c" * 64,
        "photometric_model": model,
        "blend_model": blend,
        "pairs": pairs,
    }


def _write_inputs(root: Path) -> list[tuple[Path, dict[str, object]]]:
    result = []
    for branch in EXPECTED_BRANCHES:
        path = root / f"{branch}.json"
        payload = _input(branch)
        path.write_text(json.dumps(payload), encoding="utf-8")
        result.append((path, load_baseline_input(path)))
    return result


def test_calibration_writes_only_proposed_pause_assets(tmp_path: Path) -> None:
    assets = calibrate_m61_quality(_write_inputs(tmp_path))
    output = tmp_path / "metrics"
    written = write_proposed_calibration(output, assets)
    assert {path.name for path in written} == {
        "quality_metrics_baseline.json",
        "evaluability_by_pair.json",
        "mde_calibration.json",
        "quality_thresholds_m61_v2.proposed.json",
    }
    proposed = json.loads((output / "quality_thresholds_m61_v2.proposed.json").read_text())
    assert proposed["status"] == "proposed"
    assert proposed["candidate_execution_allowed"] is False
    assert proposed["allowed_photometric_models"] == ["Q0_identity"]
    assert proposed["allowed_blend_models"] == ["B0_owner_only"]
    tolerance = proposed["aggregate_validation_p95_non_regression"]
    assert tolerance["relative_tolerance"] == 0.02
    assert tolerance["absolute_tolerance_linear"] == 0.0005
    assert "0.0015" not in tolerance["formula"]
    assert (output / "baseline_atlases").is_dir()
    for forbidden in (
        "quality_thresholds_m61_v2.json",
        "candidate_manifest.json",
        "S013_output_first_progressive_dense_central_slit_m61_v2.yaml",
        "effective_config.json",
        "P3",
    ):
        assert not (output / forbidden).exists()


@pytest.mark.parametrize(
    ("model", "blend"),
    [("Q1_scalar_luminance_gain", "B0_owner_only"), ("Q0_identity", "B1_safe_feather_2px")],
)
def test_loader_rejects_non_q0_or_non_b0(tmp_path: Path, model: str, blend: str) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(_input("fast_direct", model=model, blend=blend)), encoding="utf-8")
    with pytest.raises(ValueError, match=r"Q0_identity\+B0_owner_only"):
        load_baseline_input(path)


def test_calibration_requires_four_unique_fixed_branches(tmp_path: Path) -> None:
    inputs = _write_inputs(tmp_path)
    with pytest.raises(ValueError, match="exactly four"):
        calibrate_m61_quality(inputs[:3])
    duplicate = [inputs[0], inputs[0], inputs[2], inputs[3]]
    with pytest.raises(ValueError, match="each fixed branch"):
        calibrate_m61_quality(duplicate)


def test_calibration_refuses_nonempty_output_directory(tmp_path: Path) -> None:
    assets = calibrate_m61_quality(_write_inputs(tmp_path))
    output = tmp_path / "metrics"
    output.mkdir()
    (output / "keep.txt").write_text("user data", encoding="utf-8")
    with pytest.raises(FileExistsError):
        write_proposed_calibration(output, assets)
    assert (output / "keep.txt").read_text(encoding="utf-8") == "user data"


def test_evaluability_is_fail_closed_on_train_validation_overlap(tmp_path: Path) -> None:
    inputs = _write_inputs(tmp_path)
    inputs[0][1]["pairs"][0]["train_validation_overlap_count"] = 1
    assets = calibrate_m61_quality(inputs)
    first = assets["evaluability"]["branches"][0]["pairs"][0]
    assert first["evaluable"] is False
    assert first["reason"] == "coverage_below_frozen_provisional_gate"


def test_cli_pauses_after_proposed_threshold(tmp_path: Path) -> None:
    inputs = _write_inputs(tmp_path)
    output = tmp_path / "cli-output"
    script = Path(__file__).parents[1] / "scripts" / "calibrate_s13_m61_quality.py"
    command = [sys.executable, str(script)]
    for path, _ in inputs:
        command.extend(("--baseline-input", str(path)))
    command.extend(("--output-dir", str(output)))
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    result = json.loads(completed.stdout)
    assert result["status"] == "paused_for_threshold_approval"
    assert result["candidate_execution_allowed"] is False
    assert not (output / "quality_thresholds_m61_v2.json").exists()
    assert not (output / "candidate_manifest.json").exists()
    assert not (output / "P3").exists()
