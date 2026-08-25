from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import panorama_demo.capture_orbbec as capture_orbbec
from panorama_demo.capture_orbbec import FramePacket, SessionWriter
from panorama_demo.session import RGBDFrame
from panorama_demo.synthetic import generate_sequence
from panorama_demo.video_online_state import (
    CAPTURE_FRAME_VALIDATION_SCHEMA,
    OnlineScanAccumulator,
    load_online_state,
    write_online_state,
)
from panorama_demo.video_scan_segment import analyse_video_scan
from panorama_demo.video_session import load_video_session


def _video_session(tmp_path):
    root = generate_sequence(
        tmp_path / "session", frame_count=4, frame_width=320, frame_height=200, step=40
    )
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "capture_mode": "continuous_rgbd_video_fixed_exposure",
            "writer_errors": [],
            "product_eligibility": {
                "photo_panorama": False,
                "video_panorama": True,
            },
        }
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return load_video_session(root)


def test_online_state_is_content_bound_and_reuses_capture_time_scan_facts(tmp_path) -> None:
    video = _video_session(tmp_path)
    qualities, motions, segment = analyse_video_scan(video.rgbd.frames)
    state_path = tmp_path / "online_state.json"
    write_online_state(
        state_path,
        root=video.rgbd.root,
        frames=video.rgbd.frames,
        qualities=qualities,
        motions=motions,
        segment=segment,
        origin="capture",
    )

    state = load_online_state(
        state_path, root=video.rgbd.root, frames=video.rgbd.frames
    )

    assert state.origin == "capture"
    assert not state.certifies_strict_frame_files
    assert state.segment == segment
    assert list(state.qualities) == qualities
    assert list(state.motions) == motions


def test_online_state_rejects_a_changed_source_file(tmp_path) -> None:
    video = _video_session(tmp_path)
    qualities, motions, segment = analyse_video_scan(video.rgbd.frames)
    state_path = tmp_path / "online_state.json"
    write_online_state(
        state_path,
        root=video.rgbd.root,
        frames=video.rgbd.frames,
        qualities=qualities,
        motions=motions,
        segment=segment,
        origin="capture",
    )
    video.rgbd.frames[0].color_path.write_bytes(b"changed")

    with pytest.raises(ValueError, match="frame bytes do not match"):
        load_online_state(state_path, root=video.rgbd.root, frames=video.rgbd.frames)


def test_capture_time_accumulator_uses_writer_byte_hashes_without_rereading_sources(
    tmp_path: Path,
) -> None:
    # The writer's digests are over the exact JPEG/PNG bytes it atomically
    # publishes, while analysis runs from the aligned capture colour buffers.
    (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")
    (tmp_path / "calibration.json").write_text("{}", encoding="utf-8")
    writer = SessionWriter(
        tmp_path,
        queue_size=4,
        jpeg_quality=98,
        depth_png_compression=0,
        save_raw_depth=False,
    )
    accumulator = OnlineScanAccumulator(motion_backend="feature")
    base = np.random.default_rng(7).integers(0, 255, (200, 320, 3), dtype=np.uint8)
    for frame_id in range(4):
        color = np.roll(base, frame_id * 30, axis=1)
        depth = np.full((200, 320), 1000, dtype=np.uint16)
        assert writer.submit(
            FramePacket(
                frame_id,
                color,
                depth,
                None,
                {"depth_scale_mm_per_unit": 1.0},
            )
        )
        accumulator.add(frame_id, color)
    writer.close()

    qualities, motions, segment = accumulator.finish()
    frames = [
        RGBDFrame(
            frame_id=item.frame_id,
            color_path=item.color_path,
            aligned_depth_path=item.aligned_depth_path,
            depth_scale_mm_per_unit=1.0,
        )
        for item in writer.written_rgbd_frames
    ]
    hashes = [
        {
            "frame_id": item.frame_id,
            "color_sha256": item.color_sha256,
            "aligned_depth_sha256": item.aligned_depth_sha256,
        }
        for item in writer.written_rgbd_frames
    ]
    state_path = tmp_path / "online_video_state.json"
    write_online_state(
        state_path,
        root=tmp_path,
        frames=frames,
        qualities=qualities,
        motions=motions,
        segment=segment,
        origin="capture",
        frame_file_sha256=hashes,
        capture_frame_validation={
            "schema": CAPTURE_FRAME_VALIDATION_SCHEMA,
            "frame_count": len(frames),
        },
    )

    state = load_online_state(state_path, root=tmp_path, frames=frames)
    assert state.origin == "capture"
    assert state.certifies_strict_frame_files
    assert state.segment == segment


def test_session_writer_latches_first_disk_failure_and_stops_accepting_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    encode_calls = 0

    def fail_on_depth(path, _extension, _image, _params, *, durable=True):
        nonlocal encode_calls
        del durable
        encode_calls += 1
        path.write_bytes(b"partial frame")
        if encode_calls == 2:
            raise OSError(28, "No space left on device")
        return "digest"

    monkeypatch.setattr(capture_orbbec, "_atomic_encode", fail_on_depth)
    writer = SessionWriter(
        tmp_path,
        queue_size=4,
        jpeg_quality=98,
        depth_png_compression=0,
        save_raw_depth=False,
    )
    packet = FramePacket(
        0,
        np.zeros((4, 4, 3), dtype=np.uint8),
        np.ones((4, 4), dtype=np.uint16),
        None,
        {"depth_scale_mm_per_unit": 1.0},
    )

    assert writer.submit(packet)
    assert writer._failed.wait(timeout=2)
    with pytest.raises(capture_orbbec.SessionWriterError, match="No space left"):
        writer.submit(packet)
    writer.close()

    assert writer.stats.write_errors == 1
    assert len(writer.stats.errors) == 1
    assert writer.stats.written == 0
    assert not (tmp_path / "color" / "00000000.jpg").exists()
    assert not (tmp_path / "depth_aligned" / "00000000.png").exists()


def test_session_writer_reports_frames_csv_initialization_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_open = Path.open

    def fail_frames_csv(path: Path, *args, **kwargs):
        if path.name == "frames.csv":
            raise OSError(28, "No space left on device")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_frames_csv)
    writer = SessionWriter(
        tmp_path,
        queue_size=1,
        jpeg_quality=98,
        depth_png_compression=0,
        save_raw_depth=False,
    )
    assert writer._failed.wait(timeout=2)
    writer.close()

    assert writer.stats.write_errors == 1
    assert writer.stats.errors == [
        "frames.csv initialization: [Errno 28] No space left on device"
    ]
