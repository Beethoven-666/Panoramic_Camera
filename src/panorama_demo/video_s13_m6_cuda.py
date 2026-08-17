"""Compact source-map authority for the CUDA-resident S1.3 M6 path.

The CPU M6 renderer builds one full-canvas map for every source.  CUDA v2
only needs each source's hard-owner columns plus its replay corridors, so this
module constructs that exact continuous ROI without changing replay semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .video_s13_replay import S13VerifiedP2


@dataclass(frozen=True)
class S13M6SourceROI:
    """The exact continuous canvas interval needed from one M6 source."""

    source_index: int
    frame_id: int
    x0: int
    x1: int
    map_u: np.ndarray
    map_v: np.ndarray
    mapped: np.ndarray
    owner_mask: np.ndarray


def build_s13_m6_source_rois(
    p2: S13VerifiedP2,
    frame_ids_by_source: tuple[int, ...],
) -> tuple[S13M6SourceROI, ...]:
    """Build exact M6 source ROIs while preserving replay conflict checks."""

    source_count = int(p2.completion["source_count"])
    if len(frame_ids_by_source) != source_count:
        raise ValueError("S1.3 M6 source/frame mapping is incomplete")
    valid = np.asarray(p2.valid_mask, dtype=bool)
    owner = np.asarray(p2.provenance["owner_source_index"], dtype=np.int32)
    source_u = np.asarray(p2.provenance["source_u"], dtype=np.float32)
    source_v = np.asarray(p2.provenance["source_v"], dtype=np.float32)
    height, width = valid.shape
    rois: list[S13M6SourceROI] = []
    for source_index, frame_id in enumerate(frame_ids_by_source):
        primary = valid & (owner == source_index)
        active_columns = np.any(primary, axis=0)
        for pair in p2.replay_pairs:
            if source_index not in (pair.left_source_index, pair.right_source_index):
                continue
            if source_index == pair.left_source_index:
                pair_valid = np.asarray(pair.left_valid, dtype=bool)
            else:
                pair_valid = np.asarray(pair.right_valid, dtype=bool)
            if np.any(pair_valid):
                active_columns[pair.corridor_x0:pair.corridor_x1] = True
        columns = np.flatnonzero(active_columns)
        if not columns.size:
            raise ValueError("S1.3 M6 source has no owner or replay samples")
        x0, x1 = int(columns[0]), int(columns[-1]) + 1
        shape = (height, x1 - x0)
        map_u = np.full(shape, -1.0, dtype=np.float32)
        map_v = np.full(shape, -1.0, dtype=np.float32)
        mapped = np.zeros(shape, dtype=bool)
        owner_mask = primary[:, x0:x1]
        map_u[owner_mask] = source_u[:, x0:x1][owner_mask]
        map_v[owner_mask] = source_v[:, x0:x1][owner_mask]
        mapped |= owner_mask
        for pair in p2.replay_pairs:
            if source_index == pair.left_source_index:
                pair_u, pair_v, pair_valid = pair.left_source_u, pair.left_source_v, pair.left_valid
            elif source_index == pair.right_source_index:
                pair_u, pair_v, pair_valid = pair.right_source_u, pair.right_source_v, pair.right_valid
            else:
                continue
            local = np.s_[:, pair.corridor_x0 - x0:pair.corridor_x1 - x0]
            existing = mapped[local] & pair_valid
            if np.any(existing):
                if (
                    np.max(np.abs(map_u[local][existing] - pair_u[existing]), initial=0.0) > 1e-3
                    or np.max(np.abs(map_v[local][existing] - pair_v[existing]), initial=0.0) > 1e-3
                ):
                    raise ValueError("S1.3 P2 replay gives conflicting source sampling")
            map_u[local][pair_valid] = pair_u[pair_valid]
            map_v[local][pair_valid] = pair_v[pair_valid]
            mapped[local] |= pair_valid
        rois.append(S13M6SourceROI(
            source_index=source_index, frame_id=int(frame_id), x0=x0, x1=x1,
            map_u=map_u, map_v=map_v, mapped=mapped, owner_mask=owner_mask,
        ))
    return tuple(rois)


def compact_s13_m6_pair(
    pair: object, rois_by_source: Mapping[int, S13M6SourceROI], corrected_by_source: Mapping[int, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Return a pair's corrected corridors from compact source ROI buffers."""

    def corridor(source_index: int) -> np.ndarray:
        roi = rois_by_source[source_index]
        corrected = corrected_by_source[source_index]
        return corrected[:, pair.corridor_x0 - roi.x0:pair.corridor_x1 - roi.x0]

    return corridor(pair.left_source_index), corridor(pair.right_source_index)


__all__ = ["S13M6SourceROI", "build_s13_m6_source_rois", "compact_s13_m6_pair"]
