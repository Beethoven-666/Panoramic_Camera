from __future__ import annotations

from dataclasses import replace

import numpy as np

from panorama_demo.video_s13_blend import S13BlendPlan, S13BlendTransaction
from panorama_demo.video_s13_m6 import S13P3Result
from panorama_demo.video_s13_p3_hard_audit import audit_s13_p3_stage
from panorama_demo.video_s13_photometric import (
    PHOTOMETRIC_BOUNDS_SCHEMA,
    PHOTOMETRIC_SCHEMA,
    S13PhotometricSolution,
    S13SourcePhotometricParameters,
)
from panorama_demo.video_s13_replay import S13P2ReplayPair, S13VerifiedP2


def _fixture(tmp_path):
    generation = tmp_path / "generation"
    for name in (
        "P0/P0_completion.json", "P1/P1_completion.json", "P2/P2_completion.json",
        "P2/geometry_and_seam_panorama_owner_only.png", "P2/p2_valid_mask.png",
        "P2/p2_pixel_provenance.npz", "P2/pair_transactions.json", "P2/hard_audit.json",
    ):
        path = generation / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
    from panorama_demo.video_s13_bundle import sha256_file
    immutable = {name: sha256_file(generation / name) for name in (
        "P0/P0_completion.json", "P1/P1_completion.json", "P2/P2_completion.json",
        "P2/geometry_and_seam_panorama_owner_only.png", "P2/p2_valid_mask.png",
        "P2/p2_pixel_provenance.npz", "P2/pair_transactions.json", "P2/hard_audit.json",
    )}
    shape = (4, 8)
    owner_source = np.broadcast_to((np.arange(8) >= 4).astype(np.int32), shape).copy()
    valid = np.ones(shape, bool)
    provenance = {
        "owner_frame_id": np.where(owner_source == 0, 10, 11).astype(np.int32),
        "owner_source_index": owner_source,
        "source_u": np.broadcast_to(np.arange(8, dtype=np.float32), shape).copy(),
        "source_v": np.broadcast_to(np.arange(4, dtype=np.float32)[:, None], shape).copy(),
        "valid": valid,
        "geometry_transaction_id": np.zeros(shape, np.int32),
        "seam_transaction_id": np.zeros(shape, np.int32),
    }
    local_shape = (4, 4)
    pair = S13P2ReplayPair(
        pair_index=0, left_source_index=0, right_source_index=1,
        left_frame_id=10, right_frame_id=11, corridor_x0=2, corridor_x1=6,
        seam_x_by_row=np.full(4, 4, np.int32),
        left_source_u=np.ones(local_shape, np.float32), left_source_v=np.ones(local_shape, np.float32),
        left_valid=np.ones(local_shape, bool), right_source_u=np.ones(local_shape, np.float32),
        right_source_v=np.ones(local_shape, np.float32), right_valid=np.ones(local_shape, bool),
        primary_owner_right_mask=np.broadcast_to(np.arange(2, 6)[None, :] >= 4, local_shape),
        geometry_transaction_numeric_id=0, seam_transaction_numeric_id=0,
        parent_pair_transaction_sha256="a" * 64,
    )
    p2 = S13VerifiedP2(
        root=generation / "P2", completion={"source_count": 2},
        completion_sha256="b" * 64, result_image=np.zeros((*shape, 3), np.uint8),
        valid_mask=valid, provenance=provenance, transactions=({},),
        replay_pairs=(pair,), immutable_sha256=immutable,
    )
    weight = np.zeros(local_shape, np.float32)
    plan = S13BlendPlan(
        S13BlendTransaction(0, 0, "a" * 64, "B0_owner_only", 0, 0, 0, 0, True, None, ()),
        weight, np.ones(local_shape, bool), np.zeros(local_shape, bool),
    )
    solution = S13PhotometricSolution(
        PHOTOMETRIC_SCHEMA, PHOTOMETRIC_BOUNDS_SCHEMA, "Q0_identity",
        tuple(S13SourcePhotometricParameters(i, 10 + i, "Q0_identity", (1, 1, 1), (0, 0, 0), i, 0, None) for i in range(2)),
        (), {}, {}, 1, 0.75, False,
    )
    p3_provenance = {**provenance,
        "photometric_transaction_id": owner_source.copy(),
        "blend_transaction_id": np.full(shape, -1, np.int32),
        "secondary_frame_id": np.full(shape, -1, np.int32),
        "secondary_source_index": np.full(shape, -1, np.int32),
        "secondary_source_u": np.full(shape, np.nan, np.float32),
        "secondary_source_v": np.full(shape, np.nan, np.float32),
        "secondary_weight": np.zeros(shape, np.float32),
    }
    performance = {key: 0 for key in (
        "m4_reestimated_in_m6", "m5_reestimated_in_m6", "geometry_reestimated_in_m6",
        "seam_reestimated_in_m6", "trajectory_estimation_invocations", "open3d_invocations",
        "depth_invocations", "tsdf_invocations",
    )}
    p3 = S13P3Result(
        np.zeros((*shape, 3), np.uint8), np.zeros((*shape, 3), np.uint8), valid,
        p3_provenance, solution, (), (plan,), np.zeros(shape, bool), np.zeros(shape, bool),
        np.zeros(shape, np.float32), np.zeros(shape, bool), np.zeros(shape, bool), {}, performance,
    )
    return p2, p3


def test_p3_hard_audit_accepts_owner_only_identity(tmp_path) -> None:
    p2, p3 = _fixture(tmp_path)
    audit = audit_s13_p3_stage(p2, p3)
    assert audit["passed"] is True
    assert audit["maximum_real_contributors_per_pixel"] == 1
    assert audit["protected_blend_pixel_count"] == 0


def test_p3_hard_audit_rejects_primary_owner_change(tmp_path) -> None:
    p2, p3 = _fixture(tmp_path)
    provenance = dict(p3.pixel_provenance)
    provenance["owner_source_index"] = provenance["owner_source_index"].copy()
    provenance["owner_source_index"][0, 0] = 1
    audit = audit_s13_p3_stage(p2, replace(p3, pixel_provenance=provenance))
    assert audit["passed"] is False
    assert "primary_owner_source_index_changed" in audit["failures"]


def test_p3_hard_audit_rejects_protected_blend(tmp_path) -> None:
    p2, p3 = _fixture(tmp_path)
    provenance = dict(p3.pixel_provenance)
    provenance["secondary_weight"] = provenance["secondary_weight"].copy()
    provenance["secondary_weight"][0, 3] = 0.2
    audit = audit_s13_p3_stage(
        p2,
        replace(p3, pixel_provenance=provenance, protected_structure_mask=np.ones((4, 8), bool)),
    )
    assert audit["passed"] is False
    assert "protected_structure_blended" in audit["failures"]
