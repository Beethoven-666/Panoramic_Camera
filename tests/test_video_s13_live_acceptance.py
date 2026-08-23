from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from panorama_demo.video_delivery import publish_video_2d
from panorama_demo.video_s13_contract import (
    S13_VISUAL_CONTINUITY_ALGORITHM_ID,
    S13_VISUAL_CONTINUITY_IMPLEMENTATION_ID,
)
from panorama_demo.video_s13_live_acceptance import (
    S13V11EquivalenceError,
    compare_s13_v11_live_offline,
    require_s13_v11_live_offline_equivalence,
)


def _publish(root: Path, *, image_value: int = 7, owner_value: int = 10) -> None:
    root.mkdir(parents=True)
    image = np.full((3, 4, 3), image_value, dtype=np.uint8)
    owner = np.full((3, 4), owner_value, dtype=np.int32)
    report = {
        "delivery_state": "published",
        "manual_review_required": False,
        "strict_quality_pass": True,
        "grades": {"structural": "A", "visual": "A", "performance": "A", "overall": "A"},
        "algorithm": {
            "role": "production",
            "algorithm_id": S13_VISUAL_CONTINUITY_ALGORITHM_ID,
            "implementation_id": S13_VISUAL_CONTINUITY_IMPLEMENTATION_ID,
            "fallback_used": False,
        },
        "observability": {"report_level": "summary", "artifact_level": "minimal"},
        "input_sha256": {
            "manifest": "a" * 64,
            "calibration": "b" * 64,
            "frames_csv": "c" * 64,
        },
        "source_frame_ids": [10, 20],
        "schedule": {
            "canvas_shape": [3, 4],
            "boundaries": [0, 2, 4],
            "assignments": [{"frame_id": 10}, {"frame_id": 20}],
        },
        "owner_summary": {"shape": [3, 4]},
        "pair_seam_decisions": [{
            "pair_index": 0,
            "left_frame_id": 10,
            "right_frame_id": 20,
            "corridor_x0": 1,
            "corridor_x1": 3,
            "seam_x_by_row": [2, 2, 2],
            "m5_selected_seam_model": "midpoint_straight",
            "c2e_decision": "C0_keep_standard",
            "owner_only": False,
            "m5_transaction": {"candidate_evaluations": [{
                "candidate_id": "pair-0-S0",
                "model_code": "S0",
                "model_name": "midpoint_straight",
                "generation_status": "generated",
                "evaluation_status": "selected",
                "geometry_model": "C0_identity",
                "hard_gate_passed": True,
                "hard_gate_failures": [],
            }]},
        }],
        "m6_decisions": {
            "selected_photometric_model": "Q4c_quality_cut_component_scalar_gain",
            "blend_models": ["B1_narrow_feather"],
            "b0_count": 0,
            "b1_count": 1,
            "quality_cut_pair_indices": [],
            "m63_audit": {
                "selected_model": "Q4c",
                "candidate_models": ["Q1R", "Q4c"],
            },
            "photometric_candidate_audits": [
                {"model": "Q1R_robust_centered_scalar_gain", "selected": False},
                {"model": "Q4c_quality_cut_component_scalar_gain", "selected": True},
            ],
            "c2e_decisions": ["C0_keep_standard"],
            "c2e_owner_only_pair_indices": [],
        },
        "stage_pixel_sha256": {
            "P0": "0" * 64,
            "P1": "1" * 64,
            "P2": "2" * 64,
            "P3": "3" * 64,
        },
        "production_output_policy": {
            "stage_output_stages": [],
            "p3_stage_png_encoded": False,
            "formal_publication_encodes_p3_once": True,
        },
    }
    publish_video_2d(root, image, owner, report)
    (root / "video_timing.json").write_text(json.dumps({
        "final_2d": {
            "capture_stop_to_p3_memory_seconds": 1.0,
            "capture_stop_to_p3_published_seconds": 1.2,
        }
    }), encoding="utf-8")


def test_exact_formal_live_offline_publications_pass(tmp_path: Path) -> None:
    offline, live = tmp_path / "offline", tmp_path / "live"
    _publish(offline)
    _publish(live)

    report = require_s13_v11_live_offline_equivalence(offline, live)

    assert report["equivalent"] is True
    assert report["p3"]["differing_pixel_count"] == 0
    assert report["owner"]["differing_pixel_count"] == 0
    assert all(report["checks"].values())


def test_owner_difference_stops_at_p2(tmp_path: Path) -> None:
    offline, live = tmp_path / "offline", tmp_path / "live"
    _publish(offline)
    _publish(live, owner_value=20)

    report = compare_s13_v11_live_offline(offline, live)

    assert report["equivalent"] is False
    assert report["first_divergent_stage"] == "P2_owner_or_C2E"
    with pytest.raises(S13V11EquivalenceError) as caught:
        require_s13_v11_live_offline_equivalence(offline, live)
    assert caught.value.diagnostics["checks"]["owner_map_exact"] is False


def test_p3_pixel_difference_stops_at_p3(tmp_path: Path) -> None:
    offline, live = tmp_path / "offline", tmp_path / "live"
    _publish(offline)
    _publish(live, image_value=8)
    live_report_path = live / "video_report.json"
    live_report = json.loads(live_report_path.read_text(encoding="utf-8"))
    live_report["stage_pixel_sha256"]["P3"] = "d" * 64
    live_report_path.write_text(json.dumps(live_report), encoding="utf-8")

    report = compare_s13_v11_live_offline(offline, live)

    assert report["first_divergent_stage"] == "P3_memory_pixels"
    assert report["p3"]["differing_pixel_count"] == 12


def test_same_decoded_p3_with_different_png_bytes_stops_at_final_encode(
    tmp_path: Path,
) -> None:
    offline, live = tmp_path / "offline", tmp_path / "live"
    _publish(offline)
    _publish(live)
    image = cv2.imread(str(live / "video_panorama.png"), cv2.IMREAD_UNCHANGED)
    assert image is not None
    assert cv2.imwrite(
        str(live / "video_panorama.png"), image, [cv2.IMWRITE_PNG_COMPRESSION, 0]
    )

    report = compare_s13_v11_live_offline(offline, live)

    assert report["checks"]["p3_decoded_exact"] is True
    assert report["checks"]["p3_png_sha_exact"] is False
    assert report["first_divergent_stage"] == "final_png_encode"


def test_missing_timing_is_rejected(tmp_path: Path) -> None:
    offline, live = tmp_path / "offline", tmp_path / "live"
    _publish(offline)
    _publish(live)
    (live / "video_timing.json").unlink()

    with pytest.raises(ValueError, match="artifact is missing"):
        compare_s13_v11_live_offline(offline, live)


def test_stage_snapshot_is_rejected_by_output_policy(tmp_path: Path) -> None:
    offline, live = tmp_path / "offline", tmp_path / "live"
    _publish(offline)
    _publish(live)
    (live / "P0_base_panorama.png").write_bytes(b"not-a-formal-stage")

    report = compare_s13_v11_live_offline(offline, live)

    assert report["equivalent"] is False
    assert report["first_divergent_stage"] == "production_output_policy"
