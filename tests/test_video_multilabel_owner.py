from __future__ import annotations

import numpy as np
import pytest

from panorama_demo.video_multilabel_owner import (
    MultilabelOwnerConfig,
    _solve_row,
    assert_c8_local_multilabel_owner,
    optimise_c8_local_multilabel_owner,
)
from panorama_demo.video_visual_renderer import VideoVisualSource


def _source(frame_id: int, start: int, end: int, *, shape: tuple[int, int] = (8, 30)) -> VideoVisualSource:
    image = np.zeros((*shape, 4), dtype=np.uint8)
    image[:, start:end, :3] = frame_id
    image[:, start:end, 3] = 255
    return VideoVisualSource(frame_id=frame_id, bgra=image)


def test_c8_local_owner_is_monotone_real_partition_and_does_not_return_colour() -> None:
    sources = (_source(10, 0, 14), _source(20, 7, 22), _source(30, 15, 30))
    result = optimise_c8_local_multilabel_owner(sources)
    assert not hasattr(result, "bgra")
    assert result.audit.window_frame_count == 3
    assert np.all(result.owner_frame_id[result.valid_mask] >= 0)
    assert set(np.unique(result.owner_frame_id[result.valid_mask])) <= {10, 20, 30}
    indexes = {10: 0, 20: 1, 30: 2}
    for row in result.owner_frame_id:
        chosen = row[row >= 0]
        assert np.all(np.diff([indexes[int(value)] for value in chosen]) >= 0)
    assert_c8_local_multilabel_owner(result, sources)
    audit = result.audit.as_dict()
    assert audit["creates_colour"] is False
    assert audit["creates_pose"] is False
    assert audit["one_real_owner_per_valid_pixel"] is True


def test_c8_honours_protected_and_object_owner_masks_without_splitting_them() -> None:
    sources = (_source(4, 0, 18), _source(5, 6, 25), _source(6, 14, 30))
    protected = np.full((8, 30), -1, dtype=np.int32)
    protected[:, 8:10] = 4
    object_mask = np.zeros((8, 30), dtype=bool)
    object_mask[:, 17:19] = True
    result = optimise_c8_local_multilabel_owner(
        sources, protected_owner_frame_id=protected, object_masks={5: object_mask}
    )
    assert np.all(result.owner_frame_id[:, 8:10] == 4)
    assert np.all(result.owner_frame_id[:, 17:19] == 5)
    assert result.audit.protected_owner_pixel_count == 16
    assert result.audit.object_owner_pixel_count == 16


def test_c8_rejects_infeasible_backwards_fixed_temporal_order() -> None:
    sources = (_source(1, 0, 20), _source(2, 0, 20), _source(3, 0, 20))
    protected = np.full((8, 30), -1, dtype=np.int32)
    protected[:, 5] = 3
    protected[:, 10] = 1
    with pytest.raises(ValueError, match="temporally infeasible"):
        optimise_c8_local_multilabel_owner(sources, protected_owner_frame_id=protected)


def test_c8_invalid_canvas_gap_breaks_the_monotonic_chain() -> None:
    # There is no owner between columns 1 and 3, so a later valid island is
    # independent rather than an impossible seam that crosses invalid canvas.
    cost = np.zeros((2, 5), dtype=np.float64)
    valid = np.array([True, True, False, True, True])
    fixed = np.array([1, -1, -1, 0, -1], dtype=np.int32)
    result = _solve_row(cost, valid, fixed, switch_penalty=1.0)
    assert result.tolist() == [1, 1, -1, 0, 0]


def test_c8_rejects_more_than_five_nonlocal_sources_and_unavailable_fixed_owner() -> None:
    sources = tuple(_source(index, index * 2, index * 2 + 15) for index in range(6))
    with pytest.raises(ValueError, match="2 to 5"):
        optimise_c8_local_multilabel_owner(sources)
    local = (_source(1, 0, 10), _source(2, 10, 20))
    protected = np.full((8, 30), -1, dtype=np.int32)
    protected[:, 15] = 1
    with pytest.raises(ValueError, match="real valid source pixel"):
        optimise_c8_local_multilabel_owner(local, protected_owner_frame_id=protected)


def test_c8_rejects_conflicting_or_reordered_constraints() -> None:
    first = _source(10, 10, 25)
    second = _source(11, 0, 20)
    with pytest.raises(ValueError, match="non-monotone placed support"):
        optimise_c8_local_multilabel_owner((first, second))
    sources = (_source(10, 0, 20), _source(11, 10, 30))
    protected = np.full((8, 30), -1, dtype=np.int32)
    protected[:, 12] = 10
    object_mask = np.zeros((8, 30), dtype=bool)
    object_mask[:, 12] = True
    with pytest.raises(ValueError, match="conflicts"):
        optimise_c8_local_multilabel_owner(
            sources, protected_owner_frame_id=protected, object_masks={11: object_mask}
        )


def test_c8_configuration_never_allows_a_larger_window() -> None:
    with pytest.raises(ValueError, match=r"\[2, 5\]"):
        MultilabelOwnerConfig(maximum_window_frames=6)
