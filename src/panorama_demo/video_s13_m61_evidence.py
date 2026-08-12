"""Immutable P2-v4 photometric evidence preparation for S1.3 M6.1.

This module deliberately stops at evidence and topology.  It does not fit or
select a photometric model and it never renders P3 pixels.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Mapping, Sequence

import cv2
import numpy as np

from .video_s13_bundle import atomic_write_json, seal_stage, sha256_file, write_npz
from .video_s13_m61_config import S13PhotometricEvidenceConfig
from .video_s13_replay import S13P2ReplayPair, S13VerifiedP2, load_verified_s13_p2_for_m6


P2_V3_COMPLETION_SCHEMA = "gemini305-video-s13-p2-completion/v3"
P2_V4_COMPLETION_SCHEMA = "gemini305-video-s13-p2-completion/v4"
PHOTOMETRIC_REPLAY_SCHEMA = "gemini305-video-s13-p2-photometric-replay/v1"
OWNER_DOMAIN_SCHEMA = "gemini305-video-s13-p2-owner-domain/v1"
GRAPH_TOPOLOGY_SCHEMA = "gemini305-video-s13-p2-photometric-graph-topology/v1"
CALIBRATION_INPUT_SCHEMA = "gemini305-video-s13-m61-calibration-input/v1"


def _array_sha(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(json.dumps(value.shape, separators=(",", ":")).encode("ascii"))
    digest.update(value.tobytes())
    return digest.hexdigest()


def _array_descriptors(arrays: Mapping[str, np.ndarray]) -> dict[str, object]:
    return {
        name: {"dtype": np.asarray(value).dtype.str, "shape": list(np.asarray(value).shape),
               "sha256": _array_sha(np.asarray(value))}
        for name, value in sorted(arrays.items())
    }


def _lab(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(np.asarray(image, np.float32) / 255.0, cv2.COLOR_BGR2Lab)


def _sample_lab(image: np.ndarray, u: np.ndarray, v: np.ndarray, support: np.ndarray) -> np.ndarray:
    sampled = cv2.remap(
        _lab(image), np.asarray(u, np.float32), np.asarray(v, np.float32),
        interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0.0, 0.0, 0.0),
    ).astype(np.float32)
    sampled[~support] = np.nan
    return sampled


def _split_roles(canvas_x: np.ndarray, canvas_y: np.ndarray,
                 config: S13PhotometricEvidenceConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tile_x = np.floor_divide(canvas_x, config.tile_size_px)
    tile_y = np.floor_divide(canvas_y, config.tile_size_px)
    # Two separated validation tiles per deterministic 5x5 super-tile.  A linear hash
    # modulo N made every tile a validation neighbour when N=8, silently
    # consuming the complete train split.  The 2-D lattice leaves a real,
    # auditable guard halo and a non-empty train interior.
    phase_x = np.mod(tile_x + config.split_seed, config.validation_modulus)
    phase_y = np.mod(tile_y + 3 * config.split_seed, config.validation_modulus)
    validation = ((phase_x == 0) & (phase_y == 0)) | ((phase_x == 2) & (phase_y == 2))
    guard = np.zeros(validation.shape, bool)
    for dy in range(-config.guard_halo_tiles, config.guard_halo_tiles + 1):
        for dx in range(-config.guard_halo_tiles, config.guard_halo_tiles + 1):
            if dx == 0 and dy == 0:
                continue
            neighbor_x = np.mod(phase_x + dx, config.validation_modulus)
            neighbor_y = np.mod(phase_y + dy, config.validation_modulus)
            guard |= ((neighbor_x == 0) & (neighbor_y == 0)) | (
                (neighbor_x == 2) & (neighbor_y == 2)
            )
    guard &= ~validation
    train = ~(validation | guard)
    return train, validation, guard


def _gradient_masks(left_lab: np.ndarray, right_lab: np.ndarray, support: np.ndarray,
                    config: S13PhotometricEvidenceConfig) -> tuple[np.ndarray, np.ndarray]:
    left_l = np.nan_to_num(left_lab[..., 0], nan=0.0)
    right_l = np.nan_to_num(right_lab[..., 0], nan=0.0)
    lx, ly = cv2.Sobel(left_l, cv2.CV_32F, 1, 0), cv2.Sobel(left_l, cv2.CV_32F, 0, 1)
    rx, ry = cv2.Sobel(right_l, cv2.CV_32F, 1, 0), cv2.Sobel(right_l, cv2.CV_32F, 0, 1)
    lm, rm = np.hypot(lx, ly), np.hypot(rx, ry)
    cosine = (lx * rx + ly * ry) / np.maximum(lm * rm, 1e-6)
    conflict = (lm > config.safe_gradient_max) & (rm > config.safe_gradient_max) & (
        cosine < config.gradient_direction_cosine_min
    )
    protected = support & ((lm >= config.protected_gradient_min) |
                           (rm >= config.protected_gradient_min) | conflict)
    safe = support & ~protected & (lm <= config.safe_gradient_max) & (
        rm <= config.safe_gradient_max
    )
    return safe, protected


def _expand_replay(pair: S13P2ReplayPair, canvas_width: int,
                   shoulder: int) -> dict[str, np.ndarray]:
    height = pair.seam_x_by_row.size
    x0 = max(0, int(np.min(pair.seam_x_by_row)) - shoulder)
    x1 = min(canvas_width, int(np.max(pair.seam_x_by_row)) + shoulder + 1)
    shape = (height, x1 - x0)
    arrays: dict[str, np.ndarray] = {
        "canvas_x": np.broadcast_to(np.arange(x0, x1, dtype=np.int32), shape).copy(),
        "canvas_y": np.broadcast_to(np.arange(height, dtype=np.int32)[:, None], shape).copy(),
        "seam_x_by_row": np.asarray(pair.seam_x_by_row, np.int32).copy(),
    }
    for side in ("left", "right"):
        arrays[f"{side}_source_u"] = np.full(shape, np.nan, np.float32)
        arrays[f"{side}_source_v"] = np.full(shape, np.nan, np.float32)
        arrays[f"{side}_geometry_support"] = np.zeros(shape, bool)
    copy_x0 = max(x0, pair.corridor_x0)
    copy_x1 = min(x1, pair.corridor_x1)
    if copy_x1 > copy_x0:
        destination = slice(copy_x0 - x0, copy_x1 - x0)
        source = slice(copy_x0 - pair.corridor_x0, copy_x1 - pair.corridor_x0)
        for side in ("left", "right"):
            arrays[f"{side}_source_u"][:, destination] = getattr(pair, f"{side}_source_u")[:, source]
            arrays[f"{side}_source_v"][:, destination] = getattr(pair, f"{side}_source_v")[:, source]
            arrays[f"{side}_geometry_support"][:, destination] = getattr(pair, f"{side}_valid")[:, source]
    arrays["common_audited_support"] = (
        arrays["left_geometry_support"] & arrays["right_geometry_support"]
    )
    arrays["upstream_sealed_support"] = arrays["common_audited_support"].copy()
    arrays["upstream_sealed_risk"] = np.zeros(shape, bool)
    arrays["primary_owner_right"] = arrays["canvas_x"] >= pair.seam_x_by_row[:, None]
    return arrays


def _coverage(
    common: np.ndarray, canvas_x: np.ndarray, seam: np.ndarray,
    *, canvas_width: int, shoulder_each_side_px: int,
) -> dict[str, object]:
    offset = np.abs(canvas_x - seam[:, None])
    per_row = np.max(np.where(common, offset, -1), axis=1)
    present = per_row[per_row >= 0]
    values = present if present.size else np.asarray([0], np.int32)
    return {
        "border_clipped_row_count": int(np.count_nonzero(
            (seam - shoulder_each_side_px < 0)
            | (seam + shoulder_each_side_px >= canvas_width)
        )),
        "row_valid_count": int(present.size),
        "shoulder_coverage_px": {
            "minimum": int(np.min(values)), "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95)),
        },
    }


def _topology(edges: Sequence[Mapping[str, object]], source_count: int) -> dict[str, object]:
    parent = list(range(source_count))

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    for edge in edges:
        if edge["accepted"]:
            left, right = int(edge["left_source_index"]), int(edge["right_source_index"])
            a, b = find(left), find(right)
            if a != b:
                parent[b] = a
    groups: dict[int, list[int]] = {}
    for source in range(source_count):
        groups.setdefault(find(source), []).append(source)
    components = [
        {"component_index": index, "source_indices": members,
         "single_source_identity_boundary": len(members) == 1}
        for index, members in enumerate(sorted(groups.values(), key=lambda item: item[0]))
    ]
    return {"edges": list(edges), "components": components,
            "component_count": len(components), "source_count": source_count}


def build_s13_m61_evidence(
    p2: S13VerifiedP2,
    output: Path,
    *,
    branch: str,
    run_id: str,
    generation_id: str,
    frame_image_loader: Callable[[int], np.ndarray],
    raw_rgb_sha256: Mapping[int, str],
    config: S13PhotometricEvidenceConfig,
) -> dict[str, object]:
    """Write model-independent replay, owner-domain and train-only topology sidecars."""

    replay_root = output / "photometric_replay"
    owner_root = replay_root / "owner_domain"
    topology_root = replay_root / "graph_topology"
    replay_root.mkdir(parents=True, exist_ok=True)
    owner_root.mkdir(parents=True, exist_ok=True)
    topology_root.mkdir(parents=True, exist_ok=True)
    height, width = p2.valid_mask.shape
    pair_entries: list[dict[str, object]] = []
    topology_edges: list[dict[str, object]] = []
    all_safe = np.zeros((height, width), bool)
    all_protected = np.zeros((height, width), bool)
    seam_exclusion = np.zeros((height, width), bool)
    p2_lab = _lab(p2.result_image)
    p2_l = p2_lab[..., 0]
    p2_gradient = np.hypot(
        cv2.Sobel(p2_l, cv2.CV_32F, 1, 0),
        cv2.Sobel(p2_l, cv2.CV_32F, 0, 1),
    )
    p2_clipped = np.max(p2.result_image, axis=2) >= config.clipping_high_u8

    image_cache: dict[int, np.ndarray] = {}
    for pair in p2.replay_pairs:
        arrays = _expand_replay(pair, width, config.diagnostic_shoulder_per_side_px)
        x, y = arrays["canvas_x"], arrays["canvas_y"]
        expected = p2.valid_mask[y, x]
        common = arrays["common_audited_support"] & expected
        arrays["expected_valid"] = expected
        left_image = image_cache.setdefault(pair.left_frame_id, frame_image_loader(pair.left_frame_id))
        right_image = image_cache.setdefault(pair.right_frame_id, frame_image_loader(pair.right_frame_id))
        for image, label in ((left_image, "left"), (right_image, "right")):
            if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
                raise ValueError(f"S013 M6.1 {label} raw RGB is unreadable or invalid")
        left_lab = _sample_lab(left_image, arrays["left_source_u"], arrays["left_source_v"], common)
        right_lab = _sample_lab(right_image, arrays["right_source_u"], arrays["right_source_v"], common)
        solver_clipped = common & ((left_lab[..., 0] >= 99.0) | (right_lab[..., 0] >= 99.0))
        solver_safe, solver_protected = _gradient_masks(left_lab, right_lab, common, config)
        solver_protected |= solver_clipped | arrays["upstream_sealed_risk"]
        solver_safe &= ~solver_protected
        quality_protected = expected & (
            (p2_gradient[y, x] >= config.protected_gradient_min)
            | p2_clipped[y, x]
            | arrays["upstream_sealed_risk"]
        )
        # Where bilateral source evidence exists, its stronger structure/risk
        # classification also protects the canonical-output quality domain.
        quality_protected |= solver_protected
        quality_safe = expected & ~quality_protected & (
            p2_gradient[y, x] <= config.safe_gradient_max
        )
        train, validation, guard = _split_roles(x, y, config)
        arrays.update({
            "left_lab": left_lab, "right_lab": right_lab,
            "solver_matched_safe": solver_safe,
            "solver_matched_protected": solver_protected,
            "quality_output_safe": quality_safe,
            "quality_output_protected": quality_protected,
            "general_safe": quality_safe, "protected": quality_protected,
            "neutral_safe": solver_safe
                            & (np.hypot(left_lab[..., 1], left_lab[..., 2]) <= config.neutral_chroma_max)
                            & (np.hypot(right_lab[..., 1], right_lab[..., 2]) <= config.neutral_chroma_max),
            "train": train, "validation": validation, "excluded_guard": guard,
            "row_valid": np.any(common, axis=1),
        })
        all_safe[y, x] |= quality_safe
        all_protected[y, x] |= quality_protected
        seam_exclusion[y, x] = True
        solver_train_safe = solver_safe & train
        quality_validation_safe = quality_safe & validation
        tile_ids = np.unique(np.stack((y[solver_train_safe] // config.tile_size_px,
                                       x[solver_train_safe] // config.tile_size_px), axis=1), axis=0) \
            if np.any(solver_train_safe) else np.empty((0, 2), np.int32)
        blocks = (np.unique(y[solver_train_safe] // config.vertical_block_px)
                  if np.any(solver_train_safe) else [])
        accepted = (int(np.count_nonzero(solver_train_safe)) >= config.topology_minimum_train_samples and
                    len(tile_ids) >= config.topology_minimum_train_tiles and
                    len(blocks) >= config.topology_minimum_vertical_blocks)
        edge = {
            "pair_index": pair.pair_index, "left_source_index": pair.left_source_index,
            "right_source_index": pair.right_source_index, "accepted": bool(accepted),
            "train_safe_sample_count": int(np.count_nonzero(solver_train_safe)),
            "train_tile_count": int(len(tile_ids)), "vertical_block_count": int(len(blocks)),
            "acceptance_reason": "accepted" if accepted else "insufficient_train_evidence",
        }
        topology_edges.append(edge)
        npz_relative = Path(f"pair_{pair.pair_index:04d}.npz")
        json_relative = Path(f"pair_{pair.pair_index:04d}.json")
        validation_tiles = (np.unique(
            np.stack((y[quality_validation_safe] // config.tile_size_px,
                      x[quality_validation_safe] // config.tile_size_px), axis=1), axis=0
        ) if np.any(quality_validation_safe) else np.empty((0, 2), np.int32))
        validation_blocks = (np.unique(y[quality_validation_safe] // config.vertical_block_px)
                             if np.any(quality_validation_safe) else np.empty(0, np.int32))
        mirror_x = 2 * pair.seam_x_by_row[:, None] - 1 - x
        mirror_inside = (mirror_x >= 0) & (mirror_x < width)
        mirror_clipped_x = np.clip(mirror_x, 0, width - 1)
        mirror_expected = p2.valid_mask[y, mirror_clipped_x] & mirror_inside
        mirror_protected = mirror_expected & (
            (p2_gradient[y, mirror_clipped_x] >= config.protected_gradient_min)
            | p2_clipped[y, mirror_clipped_x]
        )
        mirror_safe = mirror_expected & ~mirror_protected & (
            p2_gradient[y, mirror_clipped_x] <= config.safe_gradient_max
        )
        metric_validation = quality_validation_safe & mirror_safe
        arrays.update({
            "quality_mirror_canvas_x": mirror_x.astype(np.int32),
            "quality_mirror_inside": mirror_inside,
            "quality_mirror_safe": mirror_safe,
            "quality_metric_validation": metric_validation,
        })
        if np.any(metric_validation):
            from .video_offline_evaluation import _delta_e00

            output_lab = p2_lab[y, x]
            mirror_lab = p2_lab[y, mirror_clipped_x]
            linear = np.abs(output_lab[..., 0] - mirror_lab[..., 0])[metric_validation] / 100.0
            delta_e = _delta_e00(output_lab, mirror_lab)[metric_validation]
            left_domain = quality_validation_safe & (x < pair.seam_x_by_row[:, None])
            right_domain = quality_validation_safe & ~left_domain
            left_values = output_lab[..., 0][left_domain]
            right_values = output_lab[..., 0][right_domain]
            plateau = (float(abs(np.median(left_values) - np.median(right_values)) / 100.0)
                       if left_values.size and right_values.size else 0.0)
            gradient_difference = np.abs(
                p2_gradient[y, x] - p2_gradient[y, mirror_clipped_x]
            )[metric_validation] / 100.0
            clipping_fraction = float(np.mean(p2_clipped[y, x][expected & validation])) \
                if np.any(expected & validation) else 0.0
            metrics = {
                "aggregate_linear_residual_median": float(np.median(linear)),
                "aggregate_linear_residual_p95": float(np.percentile(linear, 95)),
                "pair_linear_residual_p95": float(np.percentile(linear, 95)),
                "delta_e00": float(np.median(delta_e)),
                "immediate_seam": float(np.median(linear)),
                "gradient_ghost": float(np.percentile(gradient_difference, 95)),
                "owner_plateau": plateau,
                "clipping_fraction": clipping_fraction,
            }
        else:
            metrics = {name: 0.0 for name in (
                "aggregate_linear_residual_median", "aggregate_linear_residual_p95",
                "pair_linear_residual_p95", "delta_e00", "immediate_seam",
                "gradient_ghost", "owner_plateau", "clipping_fraction",
            )}
        write_npz(replay_root / npz_relative, arrays)
        validation_count = int(np.count_nonzero(quality_validation_safe))
        overlap_count = int(np.count_nonzero(train & validation))
        quality_evaluable = bool(
            validation_count >= config.minimum_pair_validation_samples
            and len(validation_tiles) >= config.minimum_independent_validation_tiles
            and len(validation_blocks) >= config.minimum_vertical_blocks
            and overlap_count == 0
            and np.any(metric_validation)
        )
        pair_doc = {
            "schema": "gemini305-video-s13-photometric-pair-evidence/v2",
            "branch": branch, "run_id": run_id, "generation_id": generation_id,
            "pair_index": pair.pair_index, "left_source_index": pair.left_source_index,
            "right_source_index": pair.right_source_index,
            "left_frame_id": pair.left_frame_id, "right_frame_id": pair.right_frame_id,
            "bounds": [int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1],
            "shoulder_each_side_px": config.diagnostic_shoulder_per_side_px,
            "raw_rgb_sha256": {"left": raw_rgb_sha256[pair.left_frame_id],
                                  "right": raw_rgb_sha256[pair.right_frame_id]},
            "parent_narrow_replay_sha256": p2.completion["assets_sha256"][
                f"pair_replay/pair_{pair.pair_index:04d}.npz"
            ],
            "parent_pair_transaction_sha256": pair.parent_pair_transaction_sha256,
            "upstream_risk_provenance": "v3_has_no_pixel_risk_asset; false_is_not_a_safe_claim",
            "support_semantics": "bilateral audited support is solver-only",
            "quality_support_semantics": "canonical P2 expected-valid 64px shoulder",
            "coverage": _coverage(
                common, x, pair.seam_x_by_row, canvas_width=width,
                shoulder_each_side_px=config.diagnostic_shoulder_per_side_px,
            ),
            "statistics": {
                "common_audited_support_count": int(np.count_nonzero(common)),
                "solver_matched_safe_count": int(np.count_nonzero(solver_safe)),
                "solver_train_safe_count": int(np.count_nonzero(solver_train_safe)),
                "quality_output_safe_count": int(np.count_nonzero(quality_safe)),
                "quality_output_protected_count": int(np.count_nonzero(quality_protected)),
                "quality_validation_safe_count": validation_count,
                "metric_validation_count": int(np.count_nonzero(metric_validation)),
            },
            "arrays": _array_descriptors(arrays), "asset": npz_relative.as_posix(),
            "asset_sha256": sha256_file(replay_root / npz_relative),
        }
        atomic_write_json(replay_root / json_relative, pair_doc)
        pair_entries.append({
            "pair_index": pair.pair_index, "asset": npz_relative.as_posix(),
            "asset_sha256": pair_doc["asset_sha256"], "metadata": json_relative.as_posix(),
            "metadata_sha256": sha256_file(replay_root / json_relative),
            "quality_class": ("quality_safe_evaluable" if quality_evaluable else
                              "quality_insufficient_evidence" if np.any(quality_safe) else
                              "protected_only" if np.any(quality_protected) else
                              "quality_insufficient_evidence"),
            "statistics": pair_doc["statistics"], "metrics": metrics,
            "validation_sample_count": validation_count,
            "independent_validation_tile_count": int(len(validation_tiles)),
            "vertical_block_count": int(len(validation_blocks)),
            "train_validation_overlap_count": overlap_count,
        })

    owner_entries: list[dict[str, object]] = []
    owner_source = np.asarray(p2.provenance["owner_source_index"], np.int32)
    owner_frame = np.asarray(p2.provenance["owner_frame_id"], np.int32)
    source_u = np.asarray(p2.provenance["source_u"], np.float32)
    source_v = np.asarray(p2.provenance["source_v"], np.float32)
    train_full, validation_full, guard_full = _split_roles(
        np.broadcast_to(np.arange(width, dtype=np.int32), (height, width)),
        np.broadcast_to(np.arange(height, dtype=np.int32)[:, None], (height, width)), config,
    )
    source_count = int(p2.completion["source_count"])
    source_frame_ids: dict[int, int] = {}
    for pair in p2.replay_pairs:
        source_frame_ids[pair.left_source_index] = pair.left_frame_id
        source_frame_ids[pair.right_source_index] = pair.right_frame_id
    p2_lab = _lab(p2.result_image)
    p2_l = p2_lab[..., 0]
    p2_gradient = np.hypot(cv2.Sobel(p2_l, cv2.CV_32F, 1, 0),
                             cv2.Sobel(p2_l, cv2.CV_32F, 0, 1))
    for source_index in range(source_count):
        mask = p2.valid_mask & (owner_source == source_index)
        yy, xx = np.nonzero(mask)
        frames = np.unique(owner_frame[mask])
        if frames.size > 1:
            raise ValueError("P2 owner source does not have exactly one real frame identity")
        frame_id = int(frames[0]) if frames.size else source_frame_ids[source_index]
        source_image = image_cache.setdefault(frame_id, frame_image_loader(frame_id))
        lab = _lab(source_image)[
            np.clip(np.rint(source_v[mask]).astype(int), 0, image_cache[frame_id].shape[0] - 1),
            np.clip(np.rint(source_u[mask]).astype(int), 0, image_cache[frame_id].shape[1] - 1),
        ].astype(np.float32)
        owner_protected = p2_gradient[mask] >= config.protected_gradient_min
        protected = all_protected[mask] | owner_protected
        owner_safe = (p2_gradient[mask] <= config.safe_gradient_max) & (
            p2_lab[..., 0][mask] < 99.0
        )
        safe = (all_safe[mask] | owner_safe) & ~protected
        arrays = {
            "canvas_x": xx.astype(np.int32), "canvas_y": yy.astype(np.int32),
            "source_u": source_u[mask], "source_v": source_v[mask],
            "valid": np.ones(xx.size, bool), "expected_valid": np.ones(xx.size, bool),
            "upstream_support": (safe | protected), "safe": safe, "protected": protected,
            "train": train_full[mask], "validation": validation_full[mask],
            "excluded_guard": guard_full[mask],
            "central_stable_domain": owner_safe & ~seam_exclusion[mask] & ~protected,
            "seam_shoulder_exclusion": seam_exclusion[mask],
            "lab_profile_input": lab,
            "sample_class": np.where(protected, 2, np.where(safe, 1, 0)).astype(np.uint8),
            "canvas_tile_x": (xx // config.tile_size_px).astype(np.int32),
            "canvas_tile_y": (yy // config.tile_size_px).astype(np.int32),
        }
        npz_relative = Path(f"source_{source_index:04d}.npz")
        json_relative = Path(f"source_{source_index:04d}.json")
        write_npz(owner_root / npz_relative, arrays)
        doc = {
            "schema": "gemini305-video-s13-p2-owner-domain-source/v1",
            "source_index": source_index, "frame_id": frame_id,
            "p2_owner_identity": {"owner_source_index": source_index, "owner_frame_id": frame_id},
            "neighbor_owner_relations": {"left": source_index - 1 if source_index else None,
                                           "right": source_index + 1 if source_index + 1 < source_count else None},
            "raw_rgb_sha256": raw_rgb_sha256[frame_id], "asset": npz_relative.as_posix(),
            "asset_sha256": sha256_file(owner_root / npz_relative),
            "arrays": _array_descriptors(arrays),
        }
        atomic_write_json(owner_root / json_relative, doc)
        owner_entries.append({"source_index": source_index, "frame_id": frame_id,
                              "asset": npz_relative.as_posix(), "asset_sha256": doc["asset_sha256"],
                              "metadata": json_relative.as_posix(),
                              "metadata_sha256": sha256_file(owner_root / json_relative)})

    topology = {
        "schema": GRAPH_TOPOLOGY_SCHEMA, "branch": branch, "run_id": run_id,
        "generation_id": generation_id, "train_only": True,
        "photometric_evidence_config_sha256": config.canonical_sha256,
        **_topology(topology_edges, source_count),
        "contains_candidate_parameters": False,
    }
    atomic_write_json(topology_root / "topology.json", topology)
    topology_sha = sha256_file(topology_root / "topology.json")
    atomic_write_json(topology_root / "manifest.json", {
        "schema": GRAPH_TOPOLOGY_SCHEMA, "topology": "topology.json",
        "topology_sha256": topology_sha, "train_only": True,
        "photometric_evidence_config_sha256": config.canonical_sha256,
    })
    atomic_write_json(owner_root / "manifest.json", {
        "schema": OWNER_DOMAIN_SCHEMA, "branch": branch, "run_id": run_id,
        "generation_id": generation_id, "source_count": source_count,
        "photometric_evidence_config_sha256": config.canonical_sha256, "sources": owner_entries,
    })
    owner_manifest_sha = sha256_file(owner_root / "manifest.json")
    legacy_completion_sha = (
        p2.completion_sha256
        if isinstance(p2.completion_sha256, str) and len(p2.completion_sha256) == 64
        else None
    )
    manifest = {
        "schema": PHOTOMETRIC_REPLAY_SCHEMA, "branch": branch, "run_id": run_id,
        "generation_id": generation_id, "pair_count": len(pair_entries),
        "source_count": source_count, "shoulder_each_side_px": config.diagnostic_shoulder_per_side_px,
        "photometric_evidence_config": asdict(config),
        "photometric_evidence_config_sha256": config.canonical_sha256,
        "p1_parent_completion_sha256": p2.completion["parent_completion_sha256"],
        # A migrated historical generation has a sealed v3 parent.  The
        # formal forward path instead builds this sidecar while the native
        # P2/v4 tree is still pending, so there is intentionally no earlier
        # P2 completion to cite.  The final v4 completion binds this manifest
        # and every sidecar asset in one direction after this function returns.
        "p2_lineage": (
            "migrated_sealed_v3" if legacy_completion_sha is not None
            else "native_pending_v4"
        ),
        "p2_v3_completion_sha256": legacy_completion_sha,
        "p2_canonical_image_sha256": p2.completion["result_asset_sha256"],
        "p2_canonical_provenance_sha256": p2.completion["assets_sha256"]["p2_pixel_provenance.npz"],
        "p2_pair_transactions_sha256": p2.completion["assets_sha256"]["pair_transactions.json"],
        "p2_narrow_replay_manifest_sha256": p2.completion["assets_sha256"]["p2_replay_manifest.json"],
        "owner_domain_manifest": "owner_domain/manifest.json",
        "owner_domain_manifest_sha256": owner_manifest_sha,
        "graph_topology_manifest": "graph_topology/manifest.json",
        "graph_topology_manifest_sha256": sha256_file(topology_root / "manifest.json"),
        "graph_topology_sha256": topology_sha, "pairs": pair_entries,
        "candidate_solver_invocations": 0, "q0_b0_only": True,
    }
    atomic_write_json(replay_root / "manifest.json", manifest)
    return manifest


def build_native_s13_m61_evidence(
    p2_root: Path,
    *,
    result_image: np.ndarray,
    valid_mask: np.ndarray,
    provenance: Mapping[str, np.ndarray],
    transactions: Sequence[Mapping[str, object]],
    replay_pairs: Sequence[S13P2ReplayPair],
    source_count: int,
    generation_id: str,
    p1_parent_completion_sha256: str,
    branch: str,
    run_id: str,
    frame_image_loader: Callable[[int], np.ndarray],
    raw_rgb_sha256: Mapping[int, str],
    config: S13PhotometricEvidenceConfig,
) -> dict[str, object]:
    """Add M6 evidence to a pending native P2/v4 before its only seal.

    This is deliberately separate from :func:`prepare_s13_m61_p2`, which is
    retained for historical v3 acceptance migrations.  The caller must have
    already written the canonical P2 assets and narrow replay files.  Their
    real hashes seed the evidence lineage; the eventual P2/v4 completion then
    seals both those assets and everything produced here atomically.
    """

    p2_root = p2_root.resolve()
    required = (
        "geometry_and_seam_panorama_owner_only.png",
        "p2_pixel_provenance.npz",
        "pair_transactions.json",
        "p2_replay_manifest.json",
    )
    if any(not (p2_root / name).is_file() for name in required):
        raise ValueError("native P2/v4 evidence requires all canonical P2 assets")
    pair_assets = {
        f"pair_replay/pair_{pair.pair_index:04d}.npz": sha256_file(
            p2_root / f"pair_replay/pair_{pair.pair_index:04d}.npz"
        )
        for pair in replay_pairs
    }
    assets = {
        **pair_assets,
        "p2_pixel_provenance.npz": sha256_file(p2_root / "p2_pixel_provenance.npz"),
        "pair_transactions.json": sha256_file(p2_root / "pair_transactions.json"),
        "p2_replay_manifest.json": sha256_file(p2_root / "p2_replay_manifest.json"),
    }
    pending_view = S13VerifiedP2(
        root=p2_root,
        completion={
            "source_count": int(source_count),
            "generation_id": generation_id,
            "assets_sha256": assets,
            "parent_completion_sha256": p1_parent_completion_sha256,
            "result_asset_sha256": sha256_file(
                p2_root / "geometry_and_seam_panorama_owner_only.png"
            ),
        },
        completion_sha256="",
        result_image=np.asarray(result_image),
        valid_mask=np.asarray(valid_mask, bool),
        provenance={name: np.asarray(value) for name, value in provenance.items()},
        transactions=tuple(transactions),
        replay_pairs=tuple(replay_pairs),
        immutable_sha256={},
    )
    return build_s13_m61_evidence(
        pending_view,
        p2_root,
        branch=branch,
        run_id=run_id,
        generation_id=generation_id,
        frame_image_loader=frame_image_loader,
        raw_rgb_sha256=raw_rgb_sha256,
        config=config,
    )


def _frame_inputs(session: Path) -> tuple[dict[int, Path], dict[int, str]]:
    paths: dict[int, Path] = {}
    hashes: dict[int, str] = {}
    with (session / "frames.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            frame_id = int(row["frame_id"])
            relative = Path(row["color_path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("unsafe raw RGB path in frames.csv")
            path = session / relative
            paths[frame_id] = path
            hashes[frame_id] = sha256_file(path)
    return paths, hashes


def _read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"raw RGB input is unreadable: {path}")
    return image


def prepare_s13_m61_p2(
    parent_p2: Path,
    output_p2: Path,
    *, branch: str,
    session: Path,
    config: S13PhotometricEvidenceConfig | None = None,
) -> dict[str, object]:
    """Create a new immutable P2-v4 root without changing the sealed P2-v3 parent."""

    config = config or S13PhotometricEvidenceConfig()
    parent_p2, output_p2, session = parent_p2.resolve(), output_p2.resolve(), session.resolve()
    if parent_p2.name != "P2" or not (parent_p2 / "P2_completion.json").is_file():
        raise ValueError("parent_p2 must be a sealed P2 directory")
    if output_p2.exists():
        raise FileExistsError("P2-v4 output already exists; preparation is append-only")
    generation = parent_p2.parent
    p2 = load_verified_s13_p2_for_m6(
        generation, p1_completion_schema="gemini305-video-s13-p1-vertical-completion/v2",
        p2_completion_schema=P2_V3_COMPLETION_SCHEMA,
    )
    paths, raw_hashes = _frame_inputs(session)
    pending = output_p2.with_name(f".{output_p2.name}.{uuid.uuid4().hex[:12]}.pending")
    try:
        shutil.copytree(parent_p2, pending, ignore=shutil.ignore_patterns("P2_completion.json"))
        manifest = build_s13_m61_evidence(
            p2, pending, branch=branch, run_id=session.name,
            generation_id=str(p2.completion["generation_id"]),
            frame_image_loader=lambda frame_id: _read_rgb(paths[frame_id]),
            raw_rgb_sha256=raw_hashes, config=config,
        )
        regression_names = (
            "geometry_and_seam_panorama_owner_only.png", "p2_pixel_provenance.npz",
            "pair_transactions.json", "p2_replay_manifest.json",
        )
        regression = {name: {"parent_sha256": sha256_file(parent_p2 / name),
                             "prepared_sha256": sha256_file(pending / name),
                             "equal": sha256_file(parent_p2 / name) == sha256_file(pending / name)}
                      for name in regression_names}
        if not all(item["equal"] for item in regression.values()):
            raise ValueError("P2-v4 preparation changed a canonical P2-v3 asset")
        old = dict(p2.completion)
        for name in ("schema", "sealed", "assets_sha256"):
            old.pop(name, None)
        replay_manifest_sha = sha256_file(pending / "photometric_replay/manifest.json")
        completion = seal_stage(
            pending, completion_name="P2_completion.json", schema=P2_V4_COMPLETION_SCHEMA,
            metadata={**old, "stage": "P2", "generation_id": p2.completion["generation_id"],
                      "parent_p2_v3_completion_sha256": p2.completion_sha256,
                      "photometric_replay_manifest_sha256": replay_manifest_sha,
                      "owner_domain_manifest_sha256": manifest["owner_domain_manifest_sha256"],
                      "graph_topology_manifest_sha256": manifest["graph_topology_manifest_sha256"],
                      "graph_topology_sha256": manifest["graph_topology_sha256"],
                      "photometric_evidence_config_sha256": config.canonical_sha256,
                      "canonical_p2_regression": regression,
                      "q_solver_invocations": 0, "q0_b0_only": True},
        )
        completion_sha = sha256_file(pending / "P2_completion.json")
        calibration_input = {
            "schema": CALIBRATION_INPUT_SCHEMA, "branch": branch, "run_id": session.name,
            "sealed": True,
            "generation": str(p2.completion["generation_id"]),
            "p2_root": str(output_p2), "p2_completion_sha256": completion_sha,
            "p2_canonical_image_sha256": completion["result_asset_sha256"],
            "photometric_replay_manifest": "photometric_replay/manifest.json",
            "photometric_replay_manifest_sha256": replay_manifest_sha,
            "graph_topology": "photometric_replay/graph_topology/topology.json",
            "graph_topology_sha256": manifest["graph_topology_sha256"],
            "owner_domain_manifest": "photometric_replay/owner_domain/manifest.json",
            "owner_domain_manifest_sha256": manifest["owner_domain_manifest_sha256"],
            "pair_evidence": [{"pair_index": item["pair_index"],
                               "metadata": f"photometric_replay/{item['metadata']}",
                               "quality_class": item["quality_class"],
                               "statistics": item["statistics"]} for item in manifest["pairs"]],
            "pairs": [{"pair_index": item["pair_index"],
                       "metadata": f"photometric_replay/{item['metadata']}",
                       "classification": item["quality_class"],
                       "metrics": item["metrics"],
                       "validation_sample_count": item["validation_sample_count"],
                       "independent_validation_tile_count": item["independent_validation_tile_count"],
                       "vertical_block_count": item["vertical_block_count"],
                       "train_validation_overlap_count": item["train_validation_overlap_count"]}
                      for item in manifest["pairs"]],
            "photometric_model": "Q0_identity", "blend_model": "B0_owner_only",
            "q0_b0_only": True, "q_solver_invocations": 0,
            "note": "Written after the P2 seal to avoid a completion self-hash cycle.",
        }
        atomic_write_json(pending / "calibration_input.json", calibration_input)
        output_p2.parent.mkdir(parents=True, exist_ok=True)
        os.replace(pending, output_p2)
        return calibration_input
    finally:
        if pending.exists():
            shutil.rmtree(pending)


__all__ = [
    "CALIBRATION_INPUT_SCHEMA", "GRAPH_TOPOLOGY_SCHEMA", "OWNER_DOMAIN_SCHEMA",
    "P2_V4_COMPLETION_SCHEMA", "PHOTOMETRIC_REPLAY_SCHEMA",
    "S13PhotometricEvidenceConfig", "build_native_s13_m61_evidence",
    "build_s13_m61_evidence", "prepare_s13_m61_p2",
]
