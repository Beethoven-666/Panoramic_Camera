from __future__ import annotations

import json

import numpy as np

from panorama_demo.orbslam3_bridge import ORBSLAM3Error
from panorama_demo.video_delivery import invalidate_video_delivery, publish_video_2d, write_video_failure
from panorama_demo.video_3d import _offline_glb_viewer


def test_video_delivery_is_published_last(tmp_path):
    image = np.full((12, 24, 3), 80, dtype=np.uint8)
    owner = np.zeros((12, 24), dtype=np.int32)
    report = publish_video_2d(tmp_path, image, owner, {"delivery_state": "published_degraded", "quality_grade": "C", "manual_review_required": True})
    assert (tmp_path / "video_delivery.json").is_file()
    assert (tmp_path / "video_panorama.png").is_file()
    assert report["schema"] == "gemini305-video-panorama-report/v1"
    delivery = json.loads((tmp_path / "video_delivery.json").read_text(encoding="utf-8"))
    assert delivery["quality_grade"] == "C"


def test_video_delivery_carries_fast_sla_and_invalidates_old_3d(tmp_path):
    (tmp_path / "video_tsdf_mesh.glb").write_bytes(b"old")
    invalidate_video_delivery(tmp_path)
    image = np.full((12, 24, 3), 80, dtype=np.uint8)
    owner = np.zeros((12, 24), dtype=np.int32)
    report = publish_video_2d(
        tmp_path,
        image,
        owner,
        {
            "delivery_state": "published_degraded",
            "quality_grade": "C",
            "manual_review_required": True,
            "preset": "fast",
            "performance": {
                "post_capture_seconds": 8.5,
                "maximum_post_seconds": 8.0,
                "within_post_capture_budget": False,
            },
        },
    )
    delivery = json.loads((tmp_path / "video_delivery.json").read_text(encoding="utf-8"))
    assert delivery["preset"] == "fast"
    assert delivery["within_post_capture_budget"] is False
    assert not (tmp_path / "video_tsdf_mesh.glb").exists()
    assert report["performance"]["post_capture_seconds"] == 8.5


def test_video_failure_records_native_attempt_audit(tmp_path):
    error = ORBSLAM3Error(
        "ORB-SLAM3 RGB-D failed (139)",
        attempt_audit=(
            {
                "attempt_index": 1,
                "returncode": 139,
                "signal": 11,
                "elapsed_seconds": 0.5,
                "accepted": False,
                "retry_reason": None,
            },
        ),
    )

    write_video_failure(tmp_path, tmp_path / "input", error)

    failure = json.loads((tmp_path / "video_failure.json").read_text(encoding="utf-8"))
    assert failure["orbslam3_execution_attempts"][0]["signal"] == 11


def test_video_delivery_publishes_staged_central_strips(tmp_path):
    image = np.full((12, 24, 3), 80, dtype=np.uint8)
    owner = np.zeros((12, 24), dtype=np.int32)
    pending = tmp_path / ".central_strips.pending"
    pending.mkdir()
    (pending / "central_strip_0000_frame_000000.png").write_bytes(b"png")
    pending_owner_only = tmp_path / ".central_strips_owner_only.pending"
    pending_owner_only.mkdir()
    (pending_owner_only / "central_strip_0000_frame_000000.png").write_bytes(b"png")
    report = publish_video_2d(
        tmp_path,
        image,
        owner,
        {
            "delivery_state": "published",
            "quality_grade": "A",
            "manual_review_required": False,
            "central_strip_export": {"image_count": 1},
            "central_strip_owner_only_export": {"image_count": 1},
        },
        pending_central_strips=pending,
        pending_central_strips_owner_only=pending_owner_only,
    )

    assert (tmp_path / "central_strips" / "central_strip_0000_frame_000000.png").is_file()
    assert report["central_strip_export"]["path"] == str(tmp_path / "central_strips")
    assert (tmp_path / "central_strips_owner_only" / "central_strip_0000_frame_000000.png").is_file()
    assert report["central_strip_owner_only_export"]["path"] == str(tmp_path / "central_strips_owner_only")
    delivery = json.loads((tmp_path / "video_delivery.json").read_text(encoding="utf-8"))
    assert delivery["central_strip_export"] == report["central_strip_export"]
    assert delivery["central_strip_owner_only_export"] == report["central_strip_owner_only_export"]


def test_video_3d_viewer_is_offline_and_references_its_sibling_glb():
    viewer = _offline_glb_viewer()
    assert "video_tsdf_mesh.glb" in viewer
    assert "getContext('webgl'" in viewer
    # Mirror the GLB node's 180-degree X rotation: Open3D +Y-down becomes
    # browser/glTF +Y-up, and forward +Z becomes viewer -Z.
    assert "=-(pos.data[3*i+1]-cy)/span" in viewer
    assert "=-(pos.data[3*i+2]-cz)/span" in viewer
    assert "model-viewer" not in viewer
    assert "https://" not in viewer
