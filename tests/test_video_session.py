from __future__ import annotations

import json

import pytest

from panorama_demo.synthetic import generate_sequence
from panorama_demo.video_session import load_video_session


def test_video_v1_auto_session_is_accepted_for_c_compatibility(tmp_path):
    root = generate_sequence(tmp_path / "session", frame_count=4, frame_width=320, frame_height=200, step=40)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update({"schema": "panorama-demo-session/v1", "capture_mode": "continuous_rgbd_video_auto", "diagnostic_only": True, "formal_stitch_allowed": False, "writer_errors": []})
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    loaded = load_video_session(root)
    assert loaded.legacy_v1 is True
    assert loaded.capture_mode == "continuous_rgbd_video_auto"


def test_video_v2_requires_product_eligibility(tmp_path):
    root = generate_sequence(tmp_path / "session", frame_count=4, frame_width=320, frame_height=200, step=40)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update({"capture_mode": "continuous_rgbd_video_fixed_exposure", "writer_errors": [], "product_eligibility": {"photo_panorama": False, "video_panorama": True}})
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert load_video_session(root).product_eligible
    manifest["product_eligibility"]["video_panorama"] = False
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="not eligible"):
        load_video_session(root)
