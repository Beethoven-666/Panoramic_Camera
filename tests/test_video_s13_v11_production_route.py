from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import panorama_demo.video_delivery as video_delivery
import panorama_demo.video_pipeline as video_pipeline
import panorama_demo.video_s13_production as production
from panorama_demo.video_algorithm import VideoAlgorithmSpec
from panorama_demo.video_s13_contract import (
    S13_VISUAL_CONTINUITY_ALGORITHM_ID,
    S13_VISUAL_CONTINUITY_IMPLEMENTATION_ID,
)


def _spec(tmp_path: Path, **overrides: object) -> VideoAlgorithmSpec:
    values: dict[str, object] = {
        "role": "production",
        "algorithm_id": S13_VISUAL_CONTINUITY_ALGORITHM_ID,
        "implementation_id": S13_VISUAL_CONTINUITY_IMPLEMENTATION_ID,
        "config_path": tmp_path / "s013-v11-production.yaml",
        "config_sha256": "a" * 64,
        "source_commit": "b" * 40,
        "model_sha256": {},
        "allow_baseline_fallback": False,
    }
    values.update(overrides)
    return VideoAlgorithmSpec(**values)  # type: ignore[arg-type]


def _authority_report() -> dict[str, object]:
    return {
        "delivery_state": "published",
        "manual_review_required": False,
        "grades": {
            "structural": "A",
            "visual": "A",
            "performance": "A",
            "overall": "A",
        },
        "source_frame_ids": [1, 2],
        "schedule": [1, 2],
        "pair_seam_decisions": [{"pair_index": 0, "mode": "hard_owner"}],
        "m6_decisions": {"M63": "pair_guard_pass"},
    }


def _forbidden(*_args: object, **_kwargs: object) -> object:
    raise AssertionError("legacy video route was reached")


def test_exact_v11_production_routes_before_legacy_and_publishes_only_final_p3(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _spec(tmp_path)
    input_path = tmp_path / "session"
    input_path.mkdir()
    output = tmp_path / "output"
    calls: dict[str, object] = {}
    replacements: list[str] = []
    original_replace = video_delivery.os.replace

    def authority(**kwargs: object) -> production.S13V11AuthorityResult:
        calls.update(kwargs)
        return production.S13V11AuthorityResult(
            panorama=np.full((12, 24, 3), 80, dtype=np.uint8),
            owner_frame_id=np.full((12, 24), 1, dtype=np.int32),
            report=_authority_report(),
        )

    def record_replace(source: object, destination: object) -> None:
        replacements.append(Path(destination).name)
        original_replace(source, destination)

    monkeypatch.setattr(video_pipeline, "_lock_paths", lambda _path: ({}, tmp_path / "base.lock", tmp_path / "prod.lock"))
    monkeypatch.setattr(video_pipeline, "resolve_video_algorithm", lambda *_args, **_kwargs: spec)
    monkeypatch.setattr(video_pipeline, "_legacy_settings_for", _forbidden)
    monkeypatch.setattr(video_pipeline, "_baseline_legacy_settings", _forbidden)
    monkeypatch.setattr(video_pipeline, "write_observability_artifacts", lambda *_args: {})
    monkeypatch.setattr(production, "_run_s13_v11_authority", authority)
    monkeypatch.setattr(video_delivery.os, "replace", record_replace)

    report = video_pipeline.run_video_algorithm(
        input_path=input_path,
        output=output,
        role="production",
        defer_3d=True,
    )

    assert calls["stage_output_stages"] == ()
    assert report["algorithm"] == {
        "role": "production",
        "algorithm_id": S13_VISUAL_CONTINUITY_ALGORITHM_ID,
        "implementation_id": S13_VISUAL_CONTINUITY_IMPLEMENTATION_ID,
        "config_sha256": "a" * 64,
        "source_commit": "b" * 40,
        "working_tree_dirty": False,
        "model_sha256": {},
        "fallback_used": False,
        "execution_backend": "s013_v11_production_cuda",
    }
    assert replacements[-1] == "video_delivery.json"
    assert (output / "video_panorama.png").is_file()
    assert (output / "video_panorama.jpg").is_file()
    assert (output / "video_pixel_provenance.npz").is_file()
    assert not any(output.glob("P[0-3]_*.png"))
    published = json.loads((output / "video_delivery.json").read_text(encoding="utf-8"))
    assert published["algorithm_id"] == S13_VISUAL_CONTINUITY_ALGORITHM_ID
    assert published["algorithm_role"] == "production"
    assert published["fallback_used"] is False


@pytest.mark.parametrize(
    "changed",
    (
        {"implementation_id": "wrong"},
        {"algorithm_id": "wrong"},
        {"allow_baseline_fallback": True},
    ),
)
def test_v11_claim_with_wrong_identity_or_fallback_fails_before_legacy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    changed: dict[str, object],
) -> None:
    spec = replace(_spec(tmp_path), **changed)
    input_path = tmp_path / "session"
    input_path.mkdir()
    monkeypatch.setattr(video_pipeline, "_lock_paths", lambda _path: ({}, tmp_path / "base.lock", tmp_path / "prod.lock"))
    monkeypatch.setattr(video_pipeline, "resolve_video_algorithm", lambda *_args, **_kwargs: spec)
    monkeypatch.setattr(video_pipeline, "_legacy_settings_for", _forbidden)
    monkeypatch.setattr(video_pipeline, "_baseline_legacy_settings", _forbidden)
    monkeypatch.setattr(video_pipeline, "write_observability_artifacts", lambda *_args: {})
    monkeypatch.setattr(production, "_run_s13_v11_authority", _forbidden)

    with pytest.raises(ValueError, match="S013 V11"):
        video_pipeline.run_video_algorithm(
            input_path=input_path,
            output=tmp_path / "output",
            role="production",
        )


def test_v11_production_rejects_orb_trajectory_input_before_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _spec(tmp_path)
    input_path = tmp_path / "session"
    input_path.mkdir()
    monkeypatch.setattr(video_pipeline, "_lock_paths", lambda _path: ({}, tmp_path / "base.lock", tmp_path / "prod.lock"))
    monkeypatch.setattr(video_pipeline, "resolve_video_algorithm", lambda *_args, **_kwargs: spec)
    monkeypatch.setattr(production, "_run_s13_v11_authority", _forbidden)

    with pytest.raises(ValueError, match="ignore-pose"):
        video_pipeline.run_video_algorithm(
            input_path=input_path,
            output=tmp_path / "output",
            role="production",
            trajectory_cache=tmp_path / "trajectory.json",
        )


def test_default_authority_uses_production_config_ignore_pose_and_no_stage_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import panorama_demo.video_s13_contract as contract
    import panorama_demo.video_s13_cuda_fast_pipeline as cuda_pipeline
    import panorama_demo.video_s13_promotion as s13_promotion
    import panorama_demo.video_s13_session as s13_session
    import panorama_demo.video_s13_trajectory as s13_trajectory

    spec = _spec(tmp_path)
    session_root = tmp_path / "session"
    session_root.mkdir()
    for name in ("manifest.json", "calibration.json", "frames.csv"):
        (session_root / name).write_text(name, encoding="utf-8")
    session = SimpleNamespace(root=session_root, frames=(SimpleNamespace(frame_id=1), SimpleNamespace(frame_id=2)))
    bundle = SimpleNamespace(session=session, validated_rgb_handoff=object(), performance={})
    config = SimpleNamespace(
        document={
            "algorithm_id": S13_VISUAL_CONTINUITY_ALGORITHM_ID,
            "implementation_id": S13_VISUAL_CONTINUITY_IMPLEMENTATION_ID,
            "motion_execution": {"policy": "deferred_step4_unless_direction_fallback_required"},
            "m5_execution": {"mode": "candidate_final_authority"},
            "m5_pair_base_atlas": {"enabled": True},
            "m5_compact_p0_map": {"mode": "compact_exact_window", "full_reference_fallback": True},
            "m62_equivalence": {},
            "m62_execution": {"mode": "candidate_single_pass"},
            "m61_bootstrap": {"selection": {}},
            "m63_photometric": {},
        },
        component={"photometric": {}, "blend": {}, "m51_r2": None},
        runtime_backend="cupy_cuda_m63_robust_v7",
        analysis_width_px=424,
        normal_target_advance_px=8.0,
        risky_target_advance_px=5.0,
        m51_r2_enabled=False,
        m51_r3_enabled=False,
        m51_r4_enabled=False,
        m62_equivalence=True,
    )
    trajectory = SimpleNamespace(audit={"source": "ignore_pose", "direct_pose_count": 0})
    calls: dict[str, object] = {}

    def fake_trajectory(_session: object, **kwargs: object) -> object:
        calls["trajectory"] = kwargs
        return trajectory

    def fake_cuda(**kwargs: object) -> dict[str, object]:
        calls["runner"] = kwargs
        image = np.full((8, 16, 3), 40, dtype=np.uint8)
        owner = np.ones((8, 16), dtype=np.int32)
        return {
            "p3": SimpleNamespace(image=image),
            "authority": {
                "source_frame_ids": (1, 2),
                "schedule": {"canvas_shape": (8, 16)},
                "owner": {"frame_id_map": owner, "summary": {"shape": (8, 16)}},
                "pair_seam_decisions": (),
                "m6": {"selected_photometric_model": "Q0_identity"},
            },
            "timings": {"total": 1.0},
            "cuda_resident": {"fallback_count": 0},
        }

    monkeypatch.setattr(contract, "load_s13_config", lambda *_args, **_kwargs: config)
    monkeypatch.setattr(s13_promotion, "verify_s13_v11_promotion", lambda *_args: None)
    monkeypatch.setattr(s13_session, "load_s13_session_bundle", lambda *_args, **_kwargs: bundle)
    monkeypatch.setattr(s13_trajectory, "load_s13_trajectory", fake_trajectory)
    monkeypatch.setattr(cuda_pipeline, "run_s13_cuda_fast_pipeline", fake_cuda)

    result = production._run_s13_v11_authority(
        session_path=session_root,
        output=tmp_path / "output",
        algorithm_spec=spec,
        maximum_post_seconds=20.0,
    )

    assert calls["trajectory"] == {
        "trajectory_cache": None,
        "reuse_online_trajectory": False,
        "run_offline_orb": False,
        "ignore_pose": True,
        "config_path": None,
    }
    runner = calls["runner"]
    assert isinstance(runner, dict)
    assert runner["stage_output_stages"] == ()
    assert runner["analysis_width_px"] == 424
    assert runner["motion_execution_policy"] == "deferred_step4"
    assert np.array_equal(result.owner_frame_id, np.ones((8, 16), dtype=np.int32))
    assert result.report["trajectory"] == {"source": "ignore_pose", "direct_pose_count": 0}
