from __future__ import annotations

import cv2
import numpy as np

from panorama_demo.video_s13_frame_store import S13FrameStore
from panorama_demo.video_s13_motion import measure_s13_motion
from panorama_demo.video_s13_session import S13RenderFrame


def _frames(tmp_path, *, count: int = 5) -> tuple[S13RenderFrame, ...]:
    frames = []
    for index in range(count):
        image = np.zeros((48, 96, 3), dtype=np.uint8)
        cv2.rectangle(image, (12 + index * 3, 9), (42 + index * 3, 38), (80, 190, 230), -1)
        cv2.line(image, (0, 4 + index), (95, 31 + index), (240, 60, 30), 2)
        path = tmp_path / f"frame_{index:03d}.png"
        assert cv2.imwrite(str(path), image)
        frames.append(S13RenderFrame(index, path, index * 1_000, 96, 48))
    return tuple(frames)


def test_frame_store_two_worker_prefetch_is_motion_exact_and_decodes_once(tmp_path) -> None:
    frames = _frames(tmp_path)
    reference = measure_s13_motion(frames, analysis_width_px=64)

    store = S13FrameStore(maximum_bytes=1024 * 1024)
    analysis = store.prefetch_analysis(frames, 64, workers=2)
    gradients = tuple(store.analysis_gradient(frame, 64) for frame in frames)
    cached = measure_s13_motion(
        frames,
        analysis_width_px=64,
        prepared_analysis=analysis,
        prepared_gradients=gradients,
    )

    assert cached == reference
    assert all(store.raw_bgr(frame) is store.raw_bgr(frame) for frame in frames)
    report = store.report()
    assert report.raw_decode_count == len(frames)
    assert report.gray_build_count == len(frames)
    assert report.gradient_build_count == len(frames)
    assert report.prefetch_workers == 2
    assert report.eviction_count == 0


def test_frame_store_releases_analysis_without_dropping_raw_sources(tmp_path) -> None:
    frames = _frames(tmp_path, count=2)
    store = S13FrameStore(maximum_bytes=1024 * 1024)
    store.prefetch_analysis(frames, 64, workers=1)
    raw = store.raw_bgr(frames[0])
    before = store.report()
    store.release_analysis_arrays()
    after = store.report()

    assert store.raw_bgr(frames[0]) is raw
    assert after.resident_bytes < before.resident_bytes
    assert after.raw_decode_count == 2
    assert after.eviction_count == 0
