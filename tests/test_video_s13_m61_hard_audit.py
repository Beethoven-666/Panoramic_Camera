from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from panorama_demo.video_s13_m61_hard_audit import (
    audit_m61_p3_files,
    promote_verified_pending,
    verify_promoted_m61_p3,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _fixture(root: Path) -> tuple[Path, Path, dict[str, object]]:
    p2, p3 = root / "P2", root / ".P3.pending"
    p2.mkdir(parents=True)
    p3.mkdir()
    shape = (3, 4)
    owner_source = np.zeros(shape, np.int32)
    primary = {
        "owner_frame_id": np.full(shape, 10, np.int32), "owner_source_index": owner_source,
        "source_u": np.broadcast_to(np.arange(4, dtype=np.float32), shape),
        "source_v": np.broadcast_to(np.arange(3, dtype=np.float32)[:, None], shape),
        "valid": np.ones(shape, bool), "geometry_transaction_id": np.zeros(shape, np.int32),
        "seam_transaction_id": np.zeros(shape, np.int32),
    }
    np.savez(p2 / "p2_pixel_provenance.npz", **primary)
    image = np.arange(36, dtype=np.uint8).reshape(3, 4, 3)
    cv2.imwrite(str(p2 / "geometry_and_seam_panorama_owner_only.png"), image)
    topology_path = p2 / "photometric_replay" / "graph_topology" / "topology.json"
    topology_path.parent.mkdir(parents=True)
    _write_json(topology_path, {
        "train_only": True,
        "components": [{"component_index": 0, "source_indices": [0]}],
    })
    p3_provenance = {**primary, "secondary_frame_id": np.full(shape, -1, np.int32),
                     "secondary_source_index": np.full(shape, -1, np.int32),
                     "secondary_source_u": np.full(shape, np.nan, np.float32),
                     "secondary_source_v": np.full(shape, np.nan, np.float32),
                     "secondary_weight": np.zeros(shape, np.float32)}
    np.savez(p3 / "p3_pixel_provenance.npz", **p3_provenance)
    cv2.imwrite(str(p3 / "visual_panorama.png"), image)
    cv2.imwrite(str(p3 / "photometric_owner_only.png"), image)
    cv2.imwrite(str(p3 / "p3_valid_mask.png"), np.full(shape, 255, np.uint8))
    np.savez(p3 / "pair_masks.npz", active=np.zeros(shape, bool), safe=np.ones(shape, bool),
             protected=np.zeros(shape, bool), common_valid=np.ones(shape, bool), expected_valid=np.ones(shape, bool))
    expected = {"parent": "a" * 64, "config": "b" * 64, "topology": _sha(topology_path)}
    _write_json(p3 / "p2_parent_reference.json", {"p2_completion_sha256": expected["parent"]})
    _write_json(p3 / "effective_config.json", {"effective_config_sha256": expected["config"], "graph_topology_sha256": expected["topology"]})
    _write_json(p3 / "performance.json", {
        "formal_render_attempt_count": 1, "formal_remap_count_by_source": [1],
        "hard_audit_render_invocations": 0, "fallback_identity_rebuild_count": 0,
        "forbidden_invocations": {"m4": 0, "m5": 0, "depth": 0, "open3d": 0, "tsdf": 0},
    })
    _write_json(p3 / "photometric_solution.json", {
        "selected_model": "Q0_identity", "graph_topology_sha256": expected["topology"],
        "components": [{"component_id": 0, "source_indices": [0], "selected_model": "Q0_identity"}],
        "sources": [{"source_index": 0, "frame_id": 10, "model": "Q0_identity", "gain_bgr": [1, 1, 1], "bias_bgr_linear": [0, 0, 0]}],
    })
    pair_asset = "blend_transactions/pair_0000.json"
    (p3 / "blend_transactions").mkdir()
    _write_json(p3 / pair_asset, {
        "pair_index": 0, "parent_completion_sha256": expected["parent"],
        "pair_masks_sha256": _sha(p3 / "pair_masks.npz"),
    })
    _write_json(p3 / "blend_transactions.json", {
        "parent_completion_sha256": expected["parent"],
        "pair_masks_sha256": _sha(p3 / "pair_masks.npz"), "fallback_reasons": [],
        "pairs": [{"pair_index": 0, "asset": pair_asset, "asset_sha256": _sha(p3 / pair_asset)}],
    })
    assets = {name: _sha(p3 / name) for name in (
        "visual_panorama.png", "photometric_owner_only.png", "p3_valid_mask.png",
        "p3_pixel_provenance.npz", "photometric_solution.json", "blend_transactions.json",
        "pair_masks.npz", "effective_config.json", "p2_parent_reference.json",
        "performance.json",
    )}
    assets[pair_asset] = _sha(p3 / pair_asset)
    _write_json(p3 / "seal_manifest.json", {"schema": "gemini305-video-s13-p3-seal-manifest/v2", "assets_sha256": assets})
    return p2, p3, {**expected, "raw": {10: image}}


def _audit(p2: Path, p3: Path, expected: dict[str, object]):
    return audit_m61_p3_files(p2, p3, expected_parent_sha256=expected["parent"],
                              expected_effective_config_sha256=expected["config"],
                              expected_topology_sha256=expected["topology"], raw_rgb_by_frame=expected["raw"])


def test_independent_audit_accepts_q0_b0_and_raw_replay(tmp_path: Path) -> None:
    p2, p3, expected = _fixture(tmp_path)
    assert _audit(p2, p3, expected)["passed"] is True


@pytest.mark.parametrize("mutation", ["primary", "weight", "uv", "parent", "config", "mask", "solution", "pixel"])
def test_mutations_fail_closed_without_identity_rebuild(tmp_path: Path, mutation: str) -> None:
    p2, p3, expected = _fixture(tmp_path)
    if mutation in {"primary", "weight", "uv"}:
        arrays = dict(np.load(p3 / "p3_pixel_provenance.npz", allow_pickle=False))
        if mutation == "primary":
            arrays["owner_source_index"][0, 0] = 1
        elif mutation == "weight":
            arrays["secondary_weight"][0, 0] = 0.6
        else:
            arrays["secondary_source_u"][0, 0] = np.inf
        np.savez(p3 / "p3_pixel_provenance.npz", **arrays)
    elif mutation == "parent":
        _write_json(p3 / "p2_parent_reference.json", {"p2_completion_sha256": "f" * 64})
    elif mutation == "config":
        _write_json(p3 / "effective_config.json", {
            "effective_config_sha256": "f" * 64,
            "graph_topology_sha256": expected["topology"],
        })
    elif mutation == "mask":
        masks = dict(np.load(p3 / "pair_masks.npz", allow_pickle=False))
        masks["active"][0, 0] = True
        np.savez(p3 / "pair_masks.npz", **masks)
    elif mutation == "solution":
        value = json.loads((p3 / "photometric_solution.json").read_text())
        value["graph_topology_sha256"] = "f" * 64
        _write_json(p3 / "photometric_solution.json", value)
    else:
        image = cv2.imread(str(p3 / "visual_panorama.png"))
        image[0, 0] ^= 1
        cv2.imwrite(str(p3 / "visual_panorama.png"), image)
    audit = _audit(p2, p3, expected)
    assert audit["passed"] is False
    assert audit["identity_rebuild_allowed"] is False
    assert audit["failure_action"] == "leave_pointer_at_p2_no_fallback_no_second_render"


def test_unclassified_fallback_and_allowlist_fail(tmp_path: Path) -> None:
    p2, p3, expected = _fixture(tmp_path)
    value = json.loads((p3 / "blend_transactions.json").read_text())
    value["fallback_reasons"] = ["anything_goes"]
    _write_json(p3 / "blend_transactions.json", value)
    manifest = json.loads((p3 / "seal_manifest.json").read_text())
    manifest["assets_sha256"]["unexpected.bin"] = "0" * 64
    _write_json(p3 / "seal_manifest.json", manifest)
    failures = _audit(p2, p3, expected)["failures"]
    assert "sealed_asset_allowlist_mismatch" in failures
    assert "fallback_reason_unclassified" in failures


def test_pointer_changes_only_after_passing_audit(tmp_path: Path) -> None:
    p2, pending, expected = _fixture(tmp_path)
    pointer = tmp_path / "current_latest.json"
    _write_json(pointer, {"stage": "P2"})
    failed = dict(_audit(p2, pending, expected))
    failed["passed"] = False
    with pytest.raises(ValueError):
        promote_verified_pending(pending, tmp_path / "P3", pointer, failed, generation_id="g")
    assert json.loads(pointer.read_text())["stage"] == "P2"
    audit = _audit(p2, pending, expected)
    promoted = promote_verified_pending(pending, tmp_path / "P3", pointer, audit, generation_id="g")
    assert promoted["stage"] == "P3"
    assert not pending.exists()
    completion = verify_promoted_m61_p3(
        p2, tmp_path / "P3", expected_parent_sha256=expected["parent"],
        expected_effective_config_sha256=expected["config"],
        expected_topology_sha256=expected["topology"], raw_rgb_by_frame=expected["raw"],
    )
    assert completion["stage"] == "P3"


def test_pair_transaction_parent_mutation_fails(tmp_path: Path) -> None:
    p2, p3, expected = _fixture(tmp_path)
    path = p3 / "blend_transactions" / "pair_0000.json"
    value = json.loads(path.read_text())
    value["parent_completion_sha256"] = "f" * 64
    _write_json(path, value)
    assert "pair_transaction_parent_invalid" in _audit(p2, p3, expected)["failures"]


def test_malformed_npz_is_structured_failure(tmp_path: Path) -> None:
    p2, p3, expected = _fixture(tmp_path)
    (p3 / "p3_pixel_provenance.npz").write_bytes(b"broken")
    audit = _audit(p2, p3, expected)
    assert audit["passed"] is False
    assert any(reason.startswith("malformed_asset:") for reason in audit["failures"])
