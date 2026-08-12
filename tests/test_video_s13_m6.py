from __future__ import annotations

from pathlib import Path

import numpy as np

from panorama_demo.video_s13_m6 import run_s13_m6
from panorama_demo.video_s13_replay import S13P2ReplayPair, S13VerifiedP2


def _verified_p2(tmp_path: Path) -> S13VerifiedP2:
    shape = (8, 12)
    owner_source = np.broadcast_to((np.arange(12) >= 6).astype(np.int32), shape).copy()
    valid = np.ones(shape, bool)
    provenance = {
        "owner_frame_id": np.where(owner_source == 0, 10, 11).astype(np.int32),
        "owner_source_index": owner_source,
        "assignment_index": owner_source.copy(),
        "source_u": np.broadcast_to(np.arange(12, dtype=np.float32), shape).copy(),
        "source_v": np.broadcast_to(np.arange(8, dtype=np.float32)[:, None], shape).copy(),
        "valid": valid,
        "selected_motion_hypothesis_id": np.zeros(shape, np.int32),
        "placement_method_code": owner_source.astype(np.int16),
        "placement_method_names": np.asarray(["a", "b"]),
        "geometry_transaction_id": np.zeros(shape, np.int32),
        "seam_transaction_id": np.zeros(shape, np.int32),
        "photometric_transaction_id": np.full(shape, -1, np.int32),
        "blend_transaction_id": np.full(shape, -1, np.int32),
        "secondary_frame_id": np.full(shape, -1, np.int32),
        "secondary_source_u": np.full(shape, np.nan, np.float32),
        "secondary_source_v": np.full(shape, np.nan, np.float32),
        "secondary_weight": np.zeros(shape, np.float32),
    }
    local_shape = (8, 8)
    pair = S13P2ReplayPair(
        0, 0, 1, 10, 11, 2, 10, np.full(8, 6, np.int32),
        np.broadcast_to(np.arange(2, 10, dtype=np.float32), local_shape).copy(),
        np.broadcast_to(np.arange(8, dtype=np.float32)[:, None], local_shape).copy(),
        np.ones(local_shape, bool),
        np.broadcast_to(np.arange(2, 10, dtype=np.float32), local_shape).copy(),
        np.broadcast_to(np.arange(8, dtype=np.float32)[:, None], local_shape).copy(),
        np.ones(local_shape, bool),
        np.broadcast_to(np.arange(2, 10)[None, :] >= 6, local_shape),
        0, 0, "a" * 64,
    )
    return S13VerifiedP2(
        tmp_path, {"source_count": 2}, "b" * 64,
        np.full((*shape, 3), 80, np.uint8), valid, provenance, ({},), (pair,), {},
    )


def test_m6_identity_owner_only_preserves_primary_provenance(tmp_path) -> None:
    p2 = _verified_p2(tmp_path)
    images = {10: np.full((8, 12, 3), 80, np.uint8), 11: np.full((8, 12, 3), 80, np.uint8)}
    result = run_s13_m6(p2, images.__getitem__, force_identity_owner_only=True)
    for name in (
        "owner_frame_id", "owner_source_index", "source_u", "source_v", "valid",
        "geometry_transaction_id", "seam_transaction_id",
    ):
        assert np.array_equal(result.pixel_provenance[name], p2.provenance[name], equal_nan=True)
    assert np.array_equal(result.visual_panorama, p2.result_image)
    assert result.performance["trajectory_estimation_invocations"] == 0
    assert result.performance["open3d_invocations"] == 0
    assert result.performance["formal_raw_rgb_remap_invocations"] == 2


def test_no_m6_completion_directory_name_is_created() -> None:
    source = Path("src/panorama_demo/video_s13_experiment.py").read_text(encoding="utf-8")
    assert 'generation / "M6"' not in source
    assert "M6_completion.json" not in source
    assert "P3_completion.json" in source
