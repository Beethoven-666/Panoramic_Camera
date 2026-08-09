from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from panorama_demo.video_s1_config import (
    S1_ALGORITHM_ID,
    S1_CONFIG_SCHEMA,
    VideoS1ExperimentConfig,
    load_video_s1_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "video_candidates" / "S01_output_first_vertical_alignment_v1.yaml"


def test_default_s1_config_loads_the_isolated_output_first_experiment() -> None:
    config = load_video_s1_config(CONFIG)

    assert config.schema == S1_CONFIG_SCHEMA
    assert config.algorithm_id == S1_ALGORITHM_ID
    assert config.scan.motion_backend == "lk_ransac"
    assert config.scan.fallback_motion_backend == "phase_correlation"
    assert config.source_selection.normal_target_step_px == 12.0
    assert config.source_selection.risk_target_step_px == 8.0
    assert config.source_selection.shoulder_minimum_px == 32
    assert config.s0.transition_width_px == 1
    assert config.s0.synthetic_hole_fill is False
    assert config.s1.gain_candidates[0] == 0.0
    assert config.output.write_s0_panorama and config.output.write_s1_panorama
    assert config.as_dict()["algorithm_id"] == S1_ALGORITHM_ID


def test_s1_config_rejects_an_invalid_step_order() -> None:
    document = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    document["source_selection"]["risk_target_step_px"] = 14.0

    with pytest.raises(ValueError, match="risk <= normal"):
        VideoS1ExperimentConfig.from_mapping(document)


def test_s1_config_keeps_zero_gain_and_forbids_synthetic_fill() -> None:
    document = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    document["s1"]["gain_candidates"] = [0.25, 0.5, 1.0]
    with pytest.raises(ValueError, match="zero-correction fallback"):
        VideoS1ExperimentConfig.from_mapping(document)

    document = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    document["s0"]["synthetic_hole_fill"] = True
    with pytest.raises(ValueError, match="must remain false"):
        VideoS1ExperimentConfig.from_mapping(document)
