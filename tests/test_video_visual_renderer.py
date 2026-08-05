from __future__ import annotations

import numpy as np
import pytest

from panorama_demo.video_visual_renderer import (
    VideoVisualSeamConfig,
    VideoVisualSource,
    render_video_visual_sources,
    video_flow_correspondence_evidence,
)


def _source(frame_id: int, image: np.ndarray, depth: np.ndarray | None = None) -> VideoVisualSource:
    return VideoVisualSource(frame_id=frame_id, bgra=image.astype(np.uint8), depth_mm=depth)


def test_curved_hard_owner_seam_preserves_real_source_pixels() -> None:
    height, width = 48, 96
    old = np.full((height, width, 4), (30, 30, 30, 255), dtype=np.uint8)
    new = np.full_like(old, (150, 150, 150, 255))
    # A deliberately curved low-cost path for the hard seam.
    path = 32 + (7 * np.sin(np.arange(height) / 5.0)).astype(int)
    for row, column in enumerate(path):
        new[row, column] = old[row, column]
    result = render_video_visual_sources(
        (_source(4, old), _source(9, new)),
        config=VideoVisualSeamConfig(flow_enabled=False, maximum_step_pixels=4),
    )
    audit = result.seams[0]
    assert audit.curved_seam
    assert max(abs(audit.seam_x_by_row[row] - int(path[row])) for row in range(height)) <= 4
    assert set(np.unique(result.owner_frame_id[result.valid_mask])) == {4, 9}
    for frame_id, expected in ((4, old), (9, new)):
        selected = result.owner_frame_id == frame_id
        assert np.array_equal(result.bgra[selected], expected[selected])


def test_nearer_depth_conflict_is_protected_and_hard_owned() -> None:
    height, width = 40, 80
    old = np.full((height, width, 4), (10, 20, 30, 255), dtype=np.uint8)
    new = np.full_like(old, (80, 90, 100, 255))
    old_depth = np.full((height, width), 1000, dtype=np.float32)
    new_depth = old_depth.copy()
    new_depth[10:30, 28:52] = 500
    result = render_video_visual_sources(
        (_source(1, old, old_depth), _source(2, new, new_depth)),
        config=VideoVisualSeamConfig(flow_enabled=False),
    )
    patch = result.owner_frame_id[10:30, 28:52]
    assert np.all(patch == 2)
    assert result.seams[0].forced_nearer_owner_pixel_count == patch.size
    assert result.seams[0].protected_pixel_count > 0
    assert result.seams[0].depth_evidence_accepted
    assert result.seams[0].depth_rejection_reason is None
    assert np.all(result.depth_mm[10:30, 28:52] == 500)


def test_rejects_depth_evidence_when_protection_covers_most_of_overlap() -> None:
    """Noisy depth must not force a broad newest-frame ownership takeover."""

    height, width = 32, 64
    old = np.full((height, width, 4), (20, 30, 40, 255), dtype=np.uint8)
    new = np.full_like(old, (80, 90, 100, 255))
    old_depth = np.full((height, width), 1000, dtype=np.float32)
    # Alternating depths make every sample a local discontinuity.  This is
    # exactly the unreliable-depth case that previously created fragmented
    # near-shelf ownership in the fast compositor.
    new_depth = np.where(np.indices((height, width))[1] % 2, 500.0, 1000.0).astype(np.float32)
    result = render_video_visual_sources(
        (_source(1, old, old_depth), _source(2, new, new_depth)),
        config=VideoVisualSeamConfig(flow_enabled=False, maximum_protected_overlap_fraction=0.45),
    )
    audit = result.seams[0]
    assert audit.protected_pixel_count > (height * width) // 2
    assert not audit.depth_evidence_accepted
    assert audit.depth_rejection_reason == "protected_overlap_fraction_exceeds_limit"
    assert audit.forced_nearer_owner_pixel_count == 0
    assert set(np.unique(result.owner_frame_id[result.valid_mask])) == {1, 2}


def test_black_is_valid_content_and_owner_partition_is_strict() -> None:
    image = np.zeros((12, 16, 4), dtype=np.uint8)
    image[..., 3] = 255
    result = render_video_visual_sources((_source(17, image),))
    assert result.valid_mask.all()
    assert np.all(result.owner_frame_id == 17)
    assert np.all(result.bgra[..., :3] == 0)


def test_flow_evidence_is_recorded_without_warping_source_colour() -> None:
    height, width = 64, 96
    old = np.zeros((height, width, 4), dtype=np.uint8)
    old[..., 3] = 255
    old[:, 20:55, :3] = 160
    new = np.roll(old, 2, axis=1)
    result = render_video_visual_sources((_source(1, old), _source(2, new)))
    assert result.seams[0].reliable_flow_fraction is not None
    assert result.seams[0].method == "dis_flow_depth_protected_curved_hard_owner"
    for frame_id, expected in ((1, old), (2, new)):
        selected = result.owner_frame_id == frame_id
        assert np.array_equal(result.bgra[selected], expected[selected])


def test_pair_flow_correspondence_is_evidence_not_rendered_colour() -> None:
    height, width = 48, 80
    old = np.zeros((height, width, 4), dtype=np.uint8)
    old[..., 3] = 255
    old[:, 18:52, :3] = 190
    new = np.roll(old, 2, axis=1)
    evidence = video_flow_correspondence_evidence(old, new, np.ones((height, width), dtype=bool))
    assert evidence is not None
    residual, reliable, sampled_new = evidence
    assert reliable.mean() > 0.5
    # Textureless regions admit multiple equally valid DIS vectors; the
    # pair-wide median still demonstrates useful correspondence evidence.
    assert float(np.median(residual[reliable])) < 20.0
    assert sampled_new.shape == new.shape


def test_rejects_invalid_source_shapes_and_duplicate_ids() -> None:
    with pytest.raises(ValueError, match="HxWx4"):
        VideoVisualSource(1, np.zeros((2, 2, 3), dtype=np.uint8))
    image = np.zeros((2, 2, 4), dtype=np.uint8)
    with pytest.raises(ValueError, match="unique"):
        render_video_visual_sources((_source(1, image), _source(1, image)))


def test_accepts_a_streaming_source_iterator() -> None:
    image = np.zeros((3, 4, 4), dtype=np.uint8)
    image[..., 3] = 255

    def sources():
        yield _source(3, image)
        yield _source(4, image)

    result = render_video_visual_sources(sources(), config=VideoVisualSeamConfig(flow_enabled=False))
    assert result.seams[0].method == "rgb_depth_protected_curved_hard_owner"
