from __future__ import annotations

import json

import numpy as np

from panorama_demo.video_delivery import publish_video_2d
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
