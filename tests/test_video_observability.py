from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from panorama_demo.video_algorithm import VideoAlgorithmSpec
from panorama_demo.video_observability import (
    ObservabilitySpec,
    clear_observability_artifacts,
    colorize_owner_map,
    owner_boundaries,
    owner_boundary_overlay,
    owner_component_report,
    write_audit_manifest,
    write_observability_artifacts,
)
from panorama_demo import video_pipeline


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_primary_delivery(output: Path) -> np.ndarray:
    output.mkdir(parents=True, exist_ok=True)
    panorama = np.full((4, 6, 3), (30, 90, 150), dtype=np.uint8)
    assert cv2.imwrite(str(output / "video_panorama.png"), panorama)
    assert cv2.imwrite(str(output / "video_panorama.jpg"), panorama)
    owner = np.array(
        [
            [-1, 1, 1, 2, 2, 2],
            [-1, 1, 1, 2, 2, 2],
            [3, 3, -1, 2, 2, 2],
            [3, 3, -1, 2, 2, 2],
        ],
        dtype=np.int32,
    )
    np.savez_compressed(output / "video_pixel_provenance.npz", owner_frame_id=owner)
    (output / "video_report.json").write_text("{}", encoding="utf-8")
    (output / "video_delivery.json").write_text("{}", encoding="utf-8")
    return owner


def _write_candidate_renderer_report(output: Path) -> None:
    """Minimal immutable report evidence for audit-sidecar tests."""

    report = {
        "schema": "gemini305-video-report/v2",
        "algorithm": {
            "role": "candidate",
            "algorithm_id": "C4_raft_rgbd_layered_mesh",
            "implementation_id": "video_visual_renderer_v2",
            "config_sha256": "a" * 64,
            "source_commit": "test",
            "model_sha256": {"raft": "b" * 64},
            "fallback_used": False,
        },
        "source_frame_ids": [10, 20],
        "renderer": {
            "backend": "video_visual_renderer_v2",
            "single_inverse_remap_per_source": True,
            "quality_metrics": {
                "candidate_mesh_evidence_pair_count": 1,
                "candidate_mesh_evidence_audits": [
                    {"first_frame_id": 10, "second_frame_id": 20, "accepted": True}
                ],
                "candidate_object_owner_lock_audits": [
                    {"first_frame_id": 10, "second_frame_id": 20, "locked_pixel_count": 3}
                ],
                "source_owner_pixel_counts": {"10": 12, "20": 12},
            },
            "video_photometric_flow_evidence": [
                {"pair_index": 0, "safe_evidence_pixel_count": 8}
            ],
            "video_global_photometric": {
                "accepted": False,
                "fail_closed_identity": True,
                "pairs": [{"left_source_index": 0, "right_source_index": 1}],
            },
        },
    }
    (output / "video_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )


def test_owner_provenance_visuals_and_component_report_are_deterministic():
    owner = np.array(
        [
            [-1, 4, 4, 9],
            [-1, 4, 4, 9],
            [9, 9, -1, 9],
        ],
        dtype=np.int32,
    )
    colour = colorize_owner_map(owner)
    assert colour.shape == (3, 4, 3)
    assert np.array_equal(colour[0, 0], (0, 0, 0))
    assert np.array_equal(colour[0, 1], colour[1, 1])
    assert not np.array_equal(colour[0, 1], colour[0, 3])

    boundary = owner_boundaries(owner)
    assert boundary.dtype == bool
    assert boundary[0, 2] and boundary[0, 3]
    assert not boundary[0, 0]
    panorama = np.full((3, 4, 3), 17, dtype=np.uint8)
    overlay = owner_boundary_overlay(panorama, owner)
    assert np.array_equal(overlay[0, 2], (255, 0, 255))
    assert np.array_equal(overlay[0, 0], panorama[0, 0])

    report = owner_component_report(owner)
    assert report["schema"] == "gemini305-video-owner-components/v1"
    assert report["owner_count"] == 2
    # Owner 9 has two spatially separate regions; the report must retain both.
    owner_nine = next(item for item in report["owners"] if item["frame_id"] == 9)
    assert owner_nine["component_count"] == 2
    assert owner_nine["pixel_count"] == 5


def test_provenance_export_writes_only_sidecars_and_audit_manifest_is_last(tmp_path):
    owner = _write_primary_delivery(tmp_path)
    primary_names = (
        "video_panorama.jpg",
        "video_panorama.png",
        "video_pixel_provenance.npz",
        "video_report.json",
        "video_delivery.json",
    )
    before = {name: _sha256(tmp_path / name) for name in primary_names}

    provenance = ObservabilitySpec(report_level="summary", artifact_level="provenance")
    exported = write_observability_artifacts(tmp_path, provenance)
    assert exported["published"] is True
    assert (tmp_path / "owner_map_color.png").is_file()
    assert (tmp_path / "owner_boundary_overlay.png").is_file()
    component = json.loads((tmp_path / "owner_component_report.json").read_text(encoding="utf-8"))
    assert component["owner_map"]["valid_pixel_count"] == int(np.count_nonzero(owner >= 0))
    assert {name: _sha256(tmp_path / name) for name in primary_names} == before

    audit = ObservabilitySpec(report_level="full", artifact_level="audit")
    exported = write_observability_artifacts(tmp_path, audit)
    manifest = write_audit_manifest(tmp_path, audit, exported)
    assert manifest["status"] == "published"
    disk_manifest = json.loads((tmp_path / "audit_manifest.json").read_text(encoding="utf-8"))
    assert disk_manifest["observability"] == audit.as_dict()
    assert disk_manifest["primary_artifacts"]["video_panorama.png"]["sha256"] == before[
        "video_panorama.png"
    ]
    assert {name: _sha256(tmp_path / name) for name in primary_names} == before


def test_full_audit_exports_candidate_pair_trace_without_changing_primary_panorama_or_owner(
    tmp_path,
):
    """Observability level must never change primary RGB or provenance bytes."""

    minimal_output = tmp_path / "minimal"
    audit_output = tmp_path / "audit"
    _write_primary_delivery(minimal_output)
    _write_primary_delivery(audit_output)
    _write_candidate_renderer_report(minimal_output)
    _write_candidate_renderer_report(audit_output)
    primary_names = ("video_panorama.jpg", "video_panorama.png", "video_pixel_provenance.npz")
    expected = {name: _sha256(minimal_output / name) for name in primary_names}

    minimal = write_observability_artifacts(
        minimal_output, ObservabilitySpec(report_level="summary", artifact_level="minimal")
    )
    assert minimal == {"artifact_level": "minimal", "published": False}
    assert not (minimal_output / "candidate_pair_audits.json").exists()
    assert not (minimal_output / "candidate_algorithm_trace.json").exists()

    exported = write_observability_artifacts(
        audit_output, ObservabilitySpec(report_level="full", artifact_level="audit")
    )
    assert {name: _sha256(audit_output / name) for name in primary_names} == expected
    assert set(exported["audit_artifacts"]) == {
        "candidate_pair_audits.json",
        "candidate_algorithm_trace.json",
    }
    pairs = json.loads((audit_output / "candidate_pair_audits.json").read_text(encoding="utf-8"))
    assert pairs["schema"] == "gemini305-video-candidate-pair-audits/v1"
    assert pairs["record_count"] == 4
    assert [group["report_path"] for group in pairs["groups"]] == [
        "renderer.quality_metrics.candidate_mesh_evidence_audits",
        "renderer.quality_metrics.candidate_object_owner_lock_audits",
        "renderer.video_photometric_flow_evidence",
        "renderer.video_global_photometric.pairs",
    ]
    trace = json.loads((audit_output / "candidate_algorithm_trace.json").read_text(encoding="utf-8"))
    assert trace["schema"] == "gemini305-video-candidate-algorithm-trace/v1"
    assert trace["algorithm"]["algorithm_id"] == "C4_raft_rgbd_layered_mesh"
    assert trace["pair_audit"]["record_count"] == 4
    # Structured pair evidence belongs only in the dedicated archive, not in
    # the compact trace summary where a consumer might mistake it for a new
    # renderer decision input.
    assert trace["renderer"]["quality_metrics"] == {
        "candidate_mesh_evidence_pair_count": 1,
    }

def test_observability_cleanup_removes_stale_visual_evaluation_but_not_primary(tmp_path):
    _write_primary_delivery(tmp_path)
    before = _sha256(tmp_path / "video_panorama.png")
    (tmp_path / "video_visual_evaluation.json").write_text("{}", encoding="utf-8")
    (tmp_path / "candidate_annotation_projection.json").write_text("{}", encoding="utf-8")
    np.savez_compressed(tmp_path / "candidate_annotation_projection_masks.npz", unused=np.zeros((1, 1)))
    (tmp_path / "candidate_pair_audits.json").write_text("{}", encoding="utf-8")
    (tmp_path / "candidate_algorithm_trace.json").write_text("{}", encoding="utf-8")
    clear_observability_artifacts(tmp_path)
    assert not (tmp_path / "video_visual_evaluation.json").exists()
    assert not (tmp_path / "candidate_annotation_projection.json").exists()
    assert not (tmp_path / "candidate_annotation_projection_masks.npz").exists()
    assert not (tmp_path / "candidate_pair_audits.json").exists()
    assert not (tmp_path / "candidate_algorithm_trace.json").exists()
    assert _sha256(tmp_path / "video_panorama.png") == before


def _baseline_spec(tmp_path: Path) -> VideoAlgorithmSpec:
    return VideoAlgorithmSpec(
        role="baseline",
        algorithm_id="test_baseline",
        implementation_id="test_renderer",
        config_path=tmp_path / "algorithm.yaml",
        config_sha256="a" * 64,
        source_commit="test",
        model_sha256={},
        allow_baseline_fallback=False,
    )


def test_pipeline_audit_failure_keeps_published_primary_and_records_manifest(
    monkeypatch, tmp_path
):
    output = tmp_path / "output"
    input_path = tmp_path / "session"
    input_path.mkdir()
    monkeypatch.setattr(
        video_pipeline,
        "_lock_paths",
        lambda _config: ({}, tmp_path / "baseline.lock", tmp_path / "production.lock"),
    )
    monkeypatch.setattr(
        video_pipeline,
        "resolve_video_algorithm",
        lambda *_args, **_kwargs: _baseline_spec(tmp_path),
    )
    monkeypatch.setattr(video_pipeline, "_legacy_settings_for", lambda _spec: {})

    from panorama_demo import video_panorama

    def fake_legacy(args):
        _write_primary_delivery(args.output)
        return {"panorama": str(args.output / "video_panorama.jpg")}

    monkeypatch.setattr(video_panorama, "run_legacy", fake_legacy)
    monkeypatch.setattr(
        video_pipeline,
        "write_observability_artifacts",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("audit encoder failed")),
    )

    result = video_pipeline.run_video_algorithm(
        input_path=input_path,
        output=output,
        role="baseline",
        observability=ObservabilitySpec(report_level="full", artifact_level="audit"),
    )

    assert result["audit_status"] == "failed"
    assert (output / "video_delivery.json").is_file()
    manifest = json.loads((output / "audit_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["error"]["message"] == "audit encoder failed"
