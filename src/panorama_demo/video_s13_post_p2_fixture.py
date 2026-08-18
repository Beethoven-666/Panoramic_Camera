"""Development-only portable post-P2 fixture for same-parent M6 comparisons."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .video_s13_replay import (
    S13VerifiedP2,
    load_replay_pair,
    replay_pair_arrays,
)
from .video_s13_session import load_s13_session, read_s13_rgb


POST_P2_FIXTURE_SCHEMA = "gemini305-video-s13-post-p2-fixture/v1"


def export_s13_post_p2_fixture(
    destination: Path,
    p2: S13VerifiedP2,
    *,
    session_path: Path,
    force_owner_only_pair_indices: frozenset[int] = frozenset(),
) -> Path:
    """Write a benchmark fixture without rerunning or modifying M0--M5."""

    root = destination.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    pairs_root = root / "pairs"
    pairs_root.mkdir(exist_ok=True)
    if not cv2.imwrite(str(root / "P2_image.png"), np.asarray(p2.result_image)):
        raise OSError("failed to write same-P2 fixture image")
    np.savez_compressed(
        root / "p2_maps.npz",
        valid=np.asarray(p2.valid_mask, dtype=bool),
        owner_source_index=np.asarray(p2.provenance["owner_source_index"], dtype=np.int32),
        source_u=np.asarray(p2.provenance["source_u"], dtype=np.float32),
        source_v=np.asarray(p2.provenance["source_v"], dtype=np.float32),
    )
    pair_entries: list[str] = []
    for pair in p2.replay_pairs:
        relative = f"pairs/pair_{pair.pair_index:04d}.npz"
        np.savez_compressed(root / relative, **replay_pair_arrays(pair))
        pair_entries.append(relative)
    source_frame_ids = [-1] * int(p2.completion["source_count"])
    for pair in p2.replay_pairs:
        source_frame_ids[pair.left_source_index] = pair.left_frame_id
        source_frame_ids[pair.right_source_index] = pair.right_frame_id
    if any(value < 0 for value in source_frame_ids):
        raise ValueError("same-P2 fixture source/frame mapping is incomplete")
    document = {
        "schema": POST_P2_FIXTURE_SCHEMA,
        "session_path": str(session_path.expanduser().resolve()),
        "canvas_shape": list(p2.valid_mask.shape),
        "source_count": len(source_frame_ids),
        "pair_count": len(pair_entries),
        "source_frame_ids": source_frame_ids,
        "p2_image": "P2_image.png",
        "p2_maps": "p2_maps.npz",
        "pairs": pair_entries,
        "force_owner_only_pair_indices": sorted(force_owner_only_pair_indices),
    }
    (root / "manifest.json").write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return root


def load_s13_post_p2_fixture(
    fixture: Path,
) -> tuple[S13VerifiedP2, Callable[[int], np.ndarray], dict[str, object]]:
    """Load one fixture without invoking trajectory, source selection, M4, or M5."""

    root = fixture.expanduser().resolve()
    document = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if document.get("schema") != POST_P2_FIXTURE_SCHEMA:
        raise ValueError("unsupported same-P2 fixture schema")
    image = cv2.imread(str(root / str(document["p2_image"])), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("same-P2 fixture image is unreadable")
    with np.load(root / str(document["p2_maps"]), allow_pickle=False) as stored:
        provenance = {name: np.asarray(stored[name]).copy() for name in stored.files}
    required = {"valid", "owner_source_index", "source_u", "source_v"}
    if set(provenance) != required:
        raise ValueError("same-P2 fixture map fields disagree")
    valid = np.asarray(provenance["valid"], dtype=bool)
    if valid.shape != image.shape[:2] or list(valid.shape) != document.get("canvas_shape"):
        raise ValueError("same-P2 fixture canvas shape disagrees")
    pairs = tuple(load_replay_pair(root / str(value)) for value in document["pairs"])
    source_count = int(document["source_count"])
    if len(pairs) != int(document["pair_count"]) or len(pairs) + 1 != source_count:
        raise ValueError("same-P2 fixture pair coverage disagrees")
    session = load_s13_session(Path(str(document["session_path"])), validation_workers=4)
    frame_by_id = session.frame_by_id
    cache: dict[int, np.ndarray] = {}

    def image_loader(frame_id: int) -> np.ndarray:
        if frame_id not in cache:
            cache[frame_id] = read_s13_rgb(frame_by_id[frame_id])
        return cache[frame_id]

    p2 = S13VerifiedP2(
        root=root,
        completion={"source_count": source_count},
        completion_sha256="",
        result_image=image,
        valid_mask=valid,
        provenance=provenance,
        transactions=tuple({"transaction_id": f"fixture-pair-{index:04d}"} for index in range(len(pairs))),
        replay_pairs=pairs,
        immutable_sha256={},
    )
    return p2, image_loader, document


__all__ = [
    "POST_P2_FIXTURE_SCHEMA",
    "export_s13_post_p2_fixture",
    "load_s13_post_p2_fixture",
]
