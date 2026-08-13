"""Sealed P2 replay assets consumed by the S1.3 M6 visual stage."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import cv2
import numpy as np

from .video_s13_bundle import sha256_file, verify_p0_completion, verify_stage


P2_REPLAY_SCHEMA = "gemini305-video-s13-p2-replay/v1"
P2_REPLAY_V2_SCHEMA = "gemini305-video-s13-p2-replay/v2"


@dataclass(frozen=True)
class S13P2ReplayPair:
    pair_index: int
    left_source_index: int
    right_source_index: int
    left_frame_id: int
    right_frame_id: int
    corridor_x0: int
    corridor_x1: int
    seam_x_by_row: np.ndarray
    left_source_u: np.ndarray
    left_source_v: np.ndarray
    left_valid: np.ndarray
    right_source_u: np.ndarray
    right_source_v: np.ndarray
    right_valid: np.ndarray
    primary_owner_right_mask: np.ndarray
    geometry_transaction_numeric_id: int
    seam_transaction_numeric_id: int
    parent_pair_transaction_sha256: str
    left_component_correction_field_id: np.ndarray | None = None
    right_component_correction_field_id: np.ndarray | None = None


@dataclass(frozen=True)
class S13VerifiedP2:
    root: Path
    completion: Mapping[str, object]
    completion_sha256: str
    result_image: np.ndarray
    valid_mask: np.ndarray
    provenance: Mapping[str, np.ndarray]
    transactions: tuple[Mapping[str, object], ...]
    replay_pairs: tuple[S13P2ReplayPair, ...]
    immutable_sha256: Mapping[str, str]


def replay_pair_arrays(pair: S13P2ReplayPair) -> dict[str, np.ndarray]:
    arrays = {
        "pair_index": np.asarray(pair.pair_index, dtype=np.int32),
        "left_source_index": np.asarray(pair.left_source_index, dtype=np.int32),
        "right_source_index": np.asarray(pair.right_source_index, dtype=np.int32),
        "left_frame_id": np.asarray(pair.left_frame_id, dtype=np.int32),
        "right_frame_id": np.asarray(pair.right_frame_id, dtype=np.int32),
        "corridor_x0": np.asarray(pair.corridor_x0, dtype=np.int32),
        "corridor_x1": np.asarray(pair.corridor_x1, dtype=np.int32),
        "seam_x_by_row": np.asarray(pair.seam_x_by_row, dtype=np.int32),
        "left_source_u": np.asarray(pair.left_source_u, dtype=np.float32),
        "left_source_v": np.asarray(pair.left_source_v, dtype=np.float32),
        "left_valid": np.asarray(pair.left_valid, dtype=bool),
        "right_source_u": np.asarray(pair.right_source_u, dtype=np.float32),
        "right_source_v": np.asarray(pair.right_source_v, dtype=np.float32),
        "right_valid": np.asarray(pair.right_valid, dtype=bool),
        "primary_owner_right_mask": np.asarray(pair.primary_owner_right_mask, dtype=bool),
        "geometry_transaction_numeric_id": np.asarray(
            pair.geometry_transaction_numeric_id, dtype=np.int32
        ),
        "seam_transaction_numeric_id": np.asarray(
            pair.seam_transaction_numeric_id, dtype=np.int32
        ),
        "parent_pair_transaction_sha256": np.asarray(pair.parent_pair_transaction_sha256),
    }
    if pair.left_component_correction_field_id is not None:
        arrays["left_component_correction_field_id"] = np.asarray(
            pair.left_component_correction_field_id, dtype=np.int32
        )
    if pair.right_component_correction_field_id is not None:
        arrays["right_component_correction_field_id"] = np.asarray(
            pair.right_component_correction_field_id, dtype=np.int32
        )
    return arrays


def _scalar(archive: Mapping[str, np.ndarray], name: str) -> int:
    value = np.asarray(archive[name])
    if value.shape != ():
        raise ValueError(f"S1.3 P2 replay scalar has invalid shape: {name}")
    return int(value.item())


def load_replay_pair(path: Path) -> S13P2ReplayPair:
    try:
        with np.load(path, allow_pickle=False) as stored:
            arrays = {name: np.asarray(stored[name]).copy() for name in stored.files}
    except (OSError, ValueError, KeyError) as exc:
        raise ValueError(f"S1.3 P2 replay pair is missing or invalid: {path.name}") from exc
    required = {
        "pair_index", "left_source_index", "right_source_index", "left_frame_id",
        "right_frame_id", "corridor_x0", "corridor_x1", "seam_x_by_row",
        "left_source_u", "left_source_v", "left_valid", "right_source_u",
        "right_source_v", "right_valid", "primary_owner_right_mask",
        "geometry_transaction_numeric_id", "seam_transaction_numeric_id",
        "parent_pair_transaction_sha256",
    }
    optional = {
        "left_component_correction_field_id",
        "right_component_correction_field_id",
    }
    if set(arrays) not in (required, required | optional):
        raise ValueError(f"S1.3 P2 replay fields disagree: {path.name}")
    x0, x1 = _scalar(arrays, "corridor_x0"), _scalar(arrays, "corridor_x1")
    seam = np.asarray(arrays["seam_x_by_row"], dtype=np.int32)
    shape = (seam.size, x1 - x0)
    if seam.ndim != 1 or x0 < 0 or x1 <= x0:
        raise ValueError(f"S1.3 P2 replay corridor is invalid: {path.name}")
    for name in (
        "left_source_u", "left_source_v", "left_valid", "right_source_u",
        "right_source_v", "right_valid", "primary_owner_right_mask",
    ):
        if arrays[name].shape != shape:
            raise ValueError(f"S1.3 P2 replay map shape disagrees: {path.name}:{name}")
    for name in optional & set(arrays):
        if arrays[name].shape != shape or arrays[name].dtype != np.int32:
            raise ValueError(f"S1.3 P2 replay field labels disagree: {path.name}:{name}")
    left_valid = np.asarray(arrays["left_valid"], dtype=bool)
    right_valid = np.asarray(arrays["right_valid"], dtype=bool)
    for prefix, valid in (("left", left_valid), ("right", right_valid)):
        u = np.asarray(arrays[f"{prefix}_source_u"], dtype=np.float32)
        v = np.asarray(arrays[f"{prefix}_source_v"], dtype=np.float32)
        if np.any(valid & (~np.isfinite(u) | ~np.isfinite(v))):
            raise ValueError(f"S1.3 P2 replay has nonfinite valid UV: {path.name}")
    parent_sha = str(np.asarray(arrays["parent_pair_transaction_sha256"]).item())
    if len(parent_sha) != 64:
        raise ValueError(f"S1.3 P2 replay transaction SHA is invalid: {path.name}")
    return S13P2ReplayPair(
        pair_index=_scalar(arrays, "pair_index"),
        left_source_index=_scalar(arrays, "left_source_index"),
        right_source_index=_scalar(arrays, "right_source_index"),
        left_frame_id=_scalar(arrays, "left_frame_id"),
        right_frame_id=_scalar(arrays, "right_frame_id"),
        corridor_x0=x0,
        corridor_x1=x1,
        seam_x_by_row=seam,
        left_source_u=np.asarray(arrays["left_source_u"], dtype=np.float32),
        left_source_v=np.asarray(arrays["left_source_v"], dtype=np.float32),
        left_valid=left_valid,
        right_source_u=np.asarray(arrays["right_source_u"], dtype=np.float32),
        right_source_v=np.asarray(arrays["right_source_v"], dtype=np.float32),
        right_valid=right_valid,
        primary_owner_right_mask=np.asarray(arrays["primary_owner_right_mask"], dtype=bool),
        geometry_transaction_numeric_id=_scalar(arrays, "geometry_transaction_numeric_id"),
        seam_transaction_numeric_id=_scalar(arrays, "seam_transaction_numeric_id"),
        parent_pair_transaction_sha256=parent_sha,
        left_component_correction_field_id=(
            np.asarray(arrays["left_component_correction_field_id"], dtype=np.int32)
            if "left_component_correction_field_id" in arrays else None
        ),
        right_component_correction_field_id=(
            np.asarray(arrays["right_component_correction_field_id"], dtype=np.int32)
            if "right_component_correction_field_id" in arrays else None
        ),
    )


def _load_json(path: Path, label: str) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"S1.3 {label} is missing or invalid") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"S1.3 {label} must be an object")
    return value


def load_verified_s13_p2_for_m6(
    generation: Path,
    *,
    p1_completion_schema: str,
    p2_completion_schema: str,
) -> S13VerifiedP2:
    """Load only hash-sealed P2 replay state; never invoke M4/M5 estimation."""

    p0, p1, p2 = generation / "P0", generation / "P1", generation / "P2"
    verify_p0_completion(p0)
    p1_completion = verify_stage(
        p1, completion_name="P1_completion.json", schema=p1_completion_schema
    )
    completion = verify_stage(
        p2, completion_name="P2_completion.json", schema=p2_completion_schema
    )
    p0_sha = sha256_file(p0 / "P0_completion.json")
    p1_sha = sha256_file(p1 / "P1_completion.json")
    if p1_completion.get("parent_completion_sha256") != p0_sha:
        raise ValueError("S1.3 P1 parent binding changed before M6")
    if completion.get("parent_stage") != "P1" or completion.get("parent_completion_sha256") != p1_sha:
        raise ValueError("S1.3 P2 parent binding changed before M6")
    if completion.get("hard_audit_passed") is not True:
        raise ValueError("S1.3 M6 requires a hard-audited P2")
    required_assets = {
        "geometry_and_seam_panorama_owner_only.png", "p2_valid_mask.png",
        "p2_pixel_provenance.npz", "pair_transactions.json", "hard_audit.json",
        "p2_replay_manifest.json", "p2_seams.npz",
    }
    assets = completion.get("assets_sha256")
    if not isinstance(assets, Mapping) or not required_assets.issubset(assets):
        raise ValueError("S1.3 P2 completion does not bind all M6 replay assets")
    hard_audit = _load_json(p2 / "hard_audit.json", "P2 hard audit")
    if hard_audit.get("passed") is not True and hard_audit.get("hard_audit_passed") is not True:
        raise ValueError("S1.3 P2 hard audit document did not pass")
    result = cv2.imread(str(p2 / "geometry_and_seam_panorama_owner_only.png"), cv2.IMREAD_COLOR)
    mask_u8 = cv2.imread(str(p2 / "p2_valid_mask.png"), cv2.IMREAD_GRAYSCALE)
    if result is None or mask_u8 is None or mask_u8.shape != result.shape[:2]:
        raise ValueError("S1.3 P2 result or valid mask is unreadable")
    with np.load(p2 / "p2_pixel_provenance.npz", allow_pickle=False) as stored:
        provenance = {name: np.asarray(stored[name]).copy() for name in stored.files}
    valid = np.asarray(provenance.get("valid"), dtype=bool)
    if valid.shape != result.shape[:2] or not np.array_equal(valid, mask_u8 > 0):
        raise ValueError("S1.3 P2 valid mask disagrees with provenance")
    transactions_doc = _load_json(p2 / "pair_transactions.json", "P2 pair transactions")
    transactions_value = transactions_doc.get("pairs")
    if not isinstance(transactions_value, Sequence) or isinstance(transactions_value, (str, bytes)):
        raise ValueError("S1.3 P2 pair transaction manifest is incomplete")
    transactions = tuple(
        item for item in transactions_value if isinstance(item, Mapping)
    )
    if len(transactions) != len(transactions_value):
        raise ValueError("S1.3 P2 pair transaction entry is invalid")
    replay_doc = _load_json(p2 / "p2_replay_manifest.json", "P2 replay manifest")
    if replay_doc.get("schema") != P2_REPLAY_SCHEMA:
        raise ValueError("S1.3 P2 replay schema is invalid")
    replay_entries = replay_doc.get("pairs")
    if not isinstance(replay_entries, Sequence) or isinstance(replay_entries, (str, bytes)):
        raise ValueError("S1.3 P2 replay manifest has no pairs")
    if len(replay_entries) != len(transactions):
        raise ValueError("S1.3 P2 replay pair count disagrees")
    replay_pairs: list[S13P2ReplayPair] = []
    for pair_index, (entry, transaction) in enumerate(zip(replay_entries, transactions, strict=True)):
        if not isinstance(entry, Mapping):
            raise ValueError("S1.3 P2 replay manifest entry is invalid")
        relative = Path(str(entry.get("asset", "")))
        if relative.is_absolute() or ".." in relative.parts or relative == Path("."):
            raise ValueError("S1.3 P2 replay asset path is unsafe")
        path = p2 / relative
        if assets.get(relative.as_posix()) != sha256_file(path):
            raise ValueError("S1.3 P2 replay asset is not completion-bound")
        transaction_path = p2 / "pair_transactions" / f"pair_{pair_index:04d}.json"
        transaction_sha = sha256_file(transaction_path)
        if assets.get(transaction_path.relative_to(p2).as_posix()) != transaction_sha:
            raise ValueError("S1.3 P2 pair transaction is not completion-bound")
        pair = load_replay_pair(path)
        if pair.pair_index != pair_index or pair.parent_pair_transaction_sha256 != transaction_sha:
            raise ValueError("S1.3 P2 replay transaction binding disagrees")
        if str(transaction.get("transaction_id")) != f"m5-pair-{pair_index:04d}":
            raise ValueError("S1.3 P2 transaction ordering changed")
        replay_pairs.append(pair)
    if int(completion.get("source_count", -1)) != len(replay_pairs) + 1:
        raise ValueError("S1.3 P2 replay does not cover every adjacent source pair")
    immutable_names = (
        "P0/P0_completion.json", "P1/P1_completion.json", "P2/P2_completion.json",
        "P2/geometry_and_seam_panorama_owner_only.png", "P2/p2_valid_mask.png",
        "P2/p2_pixel_provenance.npz", "P2/pair_transactions.json", "P2/hard_audit.json",
    )
    immutable = {name: sha256_file(generation / name) for name in immutable_names}
    return S13VerifiedP2(
        root=p2,
        completion=completion,
        completion_sha256=sha256_file(p2 / "P2_completion.json"),
        result_image=result,
        valid_mask=valid,
        provenance=provenance,
        transactions=transactions,
        replay_pairs=tuple(replay_pairs),
        immutable_sha256=immutable,
    )


__all__ = [
    "P2_REPLAY_SCHEMA", "S13P2ReplayPair", "S13VerifiedP2", "load_replay_pair",
    "load_verified_s13_p2_for_m6", "replay_pair_arrays",
]
