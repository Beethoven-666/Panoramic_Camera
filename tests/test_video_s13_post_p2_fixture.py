from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from panorama_demo.video_s13_post_p2_fixture import (
    POST_P2_FIXTURE_SCHEMA,
    export_s13_post_p2_fixture,
    load_s13_post_p2_fixture,
)
from panorama_demo.video_s13_replay import S13P2ReplayPair, S13VerifiedP2


def _p2(tmp_path: Path) -> S13VerifiedP2:
    shape = (3, 5)
    corridor = (3, 2)
    pair = S13P2ReplayPair(
        pair_index=0,
        left_source_index=0,
        right_source_index=1,
        left_frame_id=10,
        right_frame_id=11,
        corridor_x0=1,
        corridor_x1=3,
        seam_x_by_row=np.full(shape[0], 2, np.int32),
        left_source_u=np.zeros(corridor, np.float32),
        left_source_v=np.zeros(corridor, np.float32),
        left_valid=np.ones(corridor, bool),
        right_source_u=np.ones(corridor, np.float32),
        right_source_v=np.ones(corridor, np.float32),
        right_valid=np.ones(corridor, bool),
        primary_owner_right_mask=np.zeros(corridor, bool),
        geometry_transaction_numeric_id=3,
        seam_transaction_numeric_id=4,
        parent_pair_transaction_sha256="a" * 64,
    )
    valid = np.ones(shape, bool)
    provenance = {
        "valid": valid,
        "owner_source_index": np.tile(np.arange(shape[1]) >= 2, (shape[0], 1)).astype(np.int32),
        "source_u": np.zeros(shape, np.float32),
        "source_v": np.zeros(shape, np.float32),
    }
    return S13VerifiedP2(
        root=tmp_path,
        completion={"source_count": 2},
        completion_sha256="",
        result_image=np.full((*shape, 3), 37, np.uint8),
        valid_mask=valid,
        provenance=provenance,
        transactions=(),
        replay_pairs=(pair,),
        immutable_sha256={},
    )


def test_post_p2_fixture_round_trip_preserves_m6_authority(tmp_path, monkeypatch):
    p2 = _p2(tmp_path)
    root = export_s13_post_p2_fixture(
        tmp_path / "fixture",
        p2,
        session_path=tmp_path / "session",
        force_owner_only_pair_indices=frozenset({0}),
    )
    fake_frames = {10: object(), 11: object()}
    monkeypatch.setattr(
        "panorama_demo.video_s13_post_p2_fixture.load_s13_session",
        lambda *_args, **_kwargs: SimpleNamespace(frame_by_id=fake_frames),
    )
    monkeypatch.setattr(
        "panorama_demo.video_s13_post_p2_fixture.read_s13_rgb",
        lambda frame: np.full((2, 2, 3), 10 if frame is fake_frames[10] else 11, np.uint8),
    )
    loaded, image_loader, manifest = load_s13_post_p2_fixture(root)
    assert manifest["schema"] == POST_P2_FIXTURE_SCHEMA
    assert manifest["source_frame_ids"] == [10, 11]
    assert manifest["force_owner_only_pair_indices"] == [0]
    assert np.array_equal(loaded.result_image, p2.result_image)
    assert np.array_equal(loaded.valid_mask, p2.valid_mask)
    for name in ("owner_source_index", "source_u", "source_v"):
        assert np.array_equal(loaded.provenance[name], p2.provenance[name])
    assert loaded.replay_pairs[0].left_frame_id == 10
    assert int(image_loader(11)[0, 0, 0]) == 11


def test_post_p2_fixture_rejects_incomplete_source_mapping(tmp_path):
    p2 = _p2(tmp_path)
    broken = S13VerifiedP2(
        root=p2.root,
        completion={"source_count": 3},
        completion_sha256="",
        result_image=p2.result_image,
        valid_mask=p2.valid_mask,
        provenance=p2.provenance,
        transactions=(),
        replay_pairs=p2.replay_pairs,
        immutable_sha256={},
    )
    try:
        export_s13_post_p2_fixture(tmp_path / "broken", broken, session_path=tmp_path)
    except ValueError as exc:
        assert "mapping is incomplete" in str(exc)
    else:
        raise AssertionError("incomplete fixture source mapping was accepted")
