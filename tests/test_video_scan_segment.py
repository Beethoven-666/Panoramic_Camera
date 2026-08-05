from __future__ import annotations

import json

from panorama_demo.synthetic import generate_sequence
from panorama_demo.video_scan_segment import analyse_video_scan
from panorama_demo.video_session import load_video_session


def test_video_scan_selects_contiguous_real_sources(tmp_path):
    root = generate_sequence(tmp_path / "session", frame_count=7, frame_width=320, frame_height=200, step=48)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update({"capture_mode": "continuous_rgbd_video_fixed_exposure", "writer_errors": [], "product_eligibility": {"photo_panorama": False, "video_panorama": True}})
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    session = load_video_session(root)
    qualities, motions, segment = analyse_video_scan(session.rgbd.frames)
    assert len(qualities) == 7
    assert len(motions) == 6
    assert {motion.method for motion in motions} <= {
        "dis_ultrafast", "dis_unreliable", "features", "phase"
    }
    assert 0 <= int(segment["start_index"]) < int(segment["end_index"]) < 7
