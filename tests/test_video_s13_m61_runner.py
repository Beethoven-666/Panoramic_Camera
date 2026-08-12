from __future__ import annotations

import cv2
import numpy as np
import pytest
from types import SimpleNamespace

from panorama_demo.video_s13_m61_runner import (
    _LazyRawRGB,
    _apply_active_pixels,
    _correct,
    _load_session_rgb,
    _verify_raw_bindings,
    _write_p3_transaction_provenance,
)


def test_correct_uses_canonical_linear_gain_bias_rounding() -> None:
    image = np.asarray([[[0, 64, 255], [128, 200, 10]]], np.uint8)
    gain = np.asarray([1.0, 0.8, 1.0])
    bias = np.asarray([0.0, 0.01, -0.03])
    expected = np.clip(np.rint(np.clip(image / 255.0 * gain + bias, 0, 1) * 255), 0, 255).astype(np.uint8)
    assert np.array_equal(_correct(image, gain, bias), expected)


def test_raw_provenance_binding_is_fail_closed() -> None:
    metadata = {
        "left_frame_id": 10, "right_frame_id": 11,
        "raw_rgb_sha256": {"left": "a" * 64, "right": "b" * 64},
    }
    _verify_raw_bindings([(metadata, {})], {10: "a" * 64, 11: "b" * 64})
    try:
        _verify_raw_bindings([(metadata, {})], {10: "a" * 64, 11: "c" * 64})
    except ValueError as exc:
        assert "provenance SHA" in str(exc)
    else:
        raise AssertionError("mismatched raw RGB binding was accepted")


def test_lazy_raw_mapping_reads_current_bytes_without_cache(tmp_path) -> None:
    path = tmp_path / "raw.png"
    cv2.imwrite(str(path), np.full((2, 3, 3), 11, np.uint8))
    mapping = _LazyRawRGB({7: path})
    assert np.all(mapping[7] == 11)
    cv2.imwrite(str(path), np.full((2, 3, 3), 29, np.uint8))
    assert np.all(mapping[7] == 29)


def test_internal_raw_path_mapping_does_not_require_frames_csv(tmp_path) -> None:
    session = tmp_path / "session_without_frames_csv"
    session.mkdir()
    path = session / "rgb_0007.png"
    cv2.imwrite(str(path), np.full((2, 3, 3), 17, np.uint8))
    paths, hashes = _load_session_rgb(session, {7: path})
    assert paths == {7: path.resolve()}
    assert len(hashes[7]) == 64


def test_inactive_corridor_pixels_remain_owner() -> None:
    owner = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    visual = owner.copy()
    yy, xx = np.indices((2, 3))
    composed = np.full_like(owner, 251)
    active = np.zeros((2, 3), bool)
    active[0, 1] = True
    _apply_active_pixels(visual, yy, xx, composed, active)
    assert np.array_equal(visual[~active], owner[~active])
    assert np.all(visual[active] == 251)


def test_active_b1_pixels_have_one_traceable_transaction_id() -> None:
    shape = (2, 3)
    provenance = {
        "valid": np.ones(shape, bool),
        "owner_source_index": np.asarray([[0, 0, 1], [0, 1, 1]], np.int32),
        "photometric_transaction_id": np.full(shape, -1, np.int32),
    }
    active = np.asarray([[True, False], [False, True]])
    arrays = {
        "canvas_y": np.asarray([[0, 0], [1, 1]], np.int32),
        "canvas_x": np.asarray([[1, 2], [1, 2]], np.int32),
        "primary_owner_right": np.asarray([[False, True], [False, True]]),
        "left_source_u": np.asarray([[1, 2], [1, 2]], np.float32),
        "left_source_v": np.asarray([[0, 0], [1, 1]], np.float32),
        "right_source_u": np.asarray([[4, 5], [4, 5]], np.float32),
        "right_source_v": np.asarray([[0, 0], [1, 1]], np.float32),
    }
    metadata = {
        "left_frame_id": 10, "right_frame_id": 11,
        "left_source_index": 0, "right_source_index": 1,
    }
    plan = SimpleNamespace(
        active=active,
        secondary_weight=np.where(active, np.float32(0.25), np.float32(0.0)),
        transaction=SimpleNamespace(pair_index=7),
    )
    _write_p3_transaction_provenance(provenance, ((metadata, arrays),), (plan,))
    traced = provenance["blend_transaction_id"] == 7
    assert np.array_equal(traced, np.asarray([[False, True, False], [False, False, True]]))
    assert np.all(provenance["blend_transaction_id"][~traced] == -1)
    assert np.array_equal(
        provenance["photometric_transaction_id"], provenance["owner_source_index"]
    )

    with pytest.raises(ValueError, match="more than one blend transaction"):
        _write_p3_transaction_provenance(
            provenance, ((metadata, arrays), (metadata, arrays)), (plan, plan),
        )
