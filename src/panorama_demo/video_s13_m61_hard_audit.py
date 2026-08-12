"""Fail-closed independent verifier and promotion state for S013 M6.1 P3.

The verifier consumes only sealed files.  It never calls the renderer, solver,
or fallback path, so a failed audit cannot rebuild an identity panorama.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

from .video_s13_bundle import (
    atomic_write_json,
    seal_stage,
    update_current_latest,
    verify_stage,
)
from .video_s13_m61_config import (
    M61_BLEND_MANIFEST_SCHEMA,
    M61_BLEND_TRANSACTION_SCHEMA,
    M61_P2_PHOTOMETRIC_REPLAY_SCHEMA,
    M61_P3_COMPLETION_SCHEMA,
    M61_P3_HARD_AUDIT_SCHEMA,
    M61_PERFORMANCE_SCHEMA,
)

HARD_AUDIT_SCHEMA = M61_P3_HARD_AUDIT_SCHEMA
COMPLETION_SCHEMA = M61_P3_COMPLETION_SCHEMA
P2_PARENT_REFERENCE_SCHEMA = "gemini305-video-s13-p2-parent-reference/v1"
SEAL_MANIFEST_SCHEMA = "gemini305-video-s13-p3-seal-manifest/v2"
CLASSIFIED_FALLBACK_REASONS = frozenset({
    "candidate_parameter_invalid", "candidate_non_regression_failed",
    "candidate_benefit_not_detectable", "blend_ineligible",
    "blend_benefit_not_detectable", "identity_selected_by_complexity_tie",
})
PRIMARY_FIELDS = (
    "owner_frame_id", "owner_source_index", "source_u", "source_v", "valid",
    "geometry_transaction_id", "seam_transaction_id",
)
SECONDARY_FIELDS = (
    "secondary_frame_id", "secondary_source_index", "secondary_source_u",
    "secondary_source_v", "secondary_weight",
)
REQUIRED_ASSETS = frozenset({
    "visual_panorama.png", "photometric_owner_only.png", "p3_valid_mask.png",
    "p3_pixel_provenance.npz", "photometric_solution.json",
    "blend_transactions.json", "pair_masks.npz", "effective_config.json",
    "performance.json", "p2_parent_reference.json",
})


def _allowed_asset(name: str) -> bool:
    if name in REQUIRED_ASSETS:
        return True
    parts = Path(name).parts
    return (
        len(parts) == 2 and parts[0] == "blend_transactions"
        and parts[1].startswith("pair_") and parts[1].endswith(".json")
        and parts[1][5:-5].isdigit()
    )


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"missing or malformed JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _npz(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            return {name: np.asarray(archive[name]) for name in archive.files}
    except (OSError, ValueError, KeyError) as exc:
        raise ValueError(f"missing or malformed NPZ: {path}") from exc


def _same(left: np.ndarray, right: np.ndarray) -> bool:
    return left.shape == right.shape and left.dtype == right.dtype and bool(
        np.array_equal(left, right, equal_nan=True) if left.dtype.kind == "f" else np.array_equal(left, right)
    )


def _load_color(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"missing or malformed image: {path}")
    return image


def _replay_pixels_full_canvas(
    provenance: Mapping[str, np.ndarray],
    sources_by_frame: Mapping[int, np.ndarray],
    parameters_by_frame: Mapping[int, tuple[np.ndarray, np.ndarray]],
    *,
    telemetry: dict[str, int] | None = None,
) -> np.ndarray:
    """Independently replay every valid pixel from raw RGB and provenance."""
    valid = np.asarray(provenance["valid"], bool)
    shape = valid.shape
    output = np.zeros((*shape, 3), np.float64)
    weight = np.asarray(provenance["secondary_weight"], np.float64)

    def sample(prefix: str, contribution: np.ndarray) -> None:
        frame = np.asarray(provenance[f"{prefix}frame_id" if prefix else "owner_frame_id"], np.int32)
        u = np.asarray(provenance[f"{prefix}source_u" if prefix else "source_u"], np.float64)
        v = np.asarray(provenance[f"{prefix}source_v" if prefix else "source_v"], np.float64)
        for frame_id in np.unique(frame[contribution]):
            if int(frame_id) not in sources_by_frame or int(frame_id) not in parameters_by_frame:
                raise ValueError("provenance references an unbound raw RGB frame")
            active = contribution & (frame == frame_id)
            image = np.asarray(sources_by_frame[int(frame_id)])
            x = u[active]
            y = v[active]
            if np.any(x < 0) or np.any(y < 0) or np.any(x > image.shape[1] - 1) or np.any(y > image.shape[0] - 1):
                raise ValueError("provenance UV is outside raw RGB")
            gain, bias = parameters_by_frame[int(frame_id)]
            # P2 is defined by the original calibrated OpenCV inverse remap,
            # not nearest-neighbour lookup.  Replaying with the same fixed
            # bilinear pixel-centre rule is independent of the formal renderer
            # while remaining byte-exact for Q0+B0.
            sampled = cv2.remap(
                image,
                np.asarray(u, np.float32),
                np.asarray(v, np.float32),
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )
            if telemetry is not None:
                telemetry["remap_invocation_count"] = telemetry.get("remap_invocation_count", 0) + 1
            linear = sampled[active].astype(np.float64) / 255.0
            corrected = np.rint(np.clip(linear * gain + bias, 0.0, 1.0) * 255.0)
            alpha = weight[active] if prefix else 1.0 - weight[active]
            output[active] += corrected * alpha[:, None]

    sample("", valid)
    sample("secondary_", valid & (weight > 0.0))
    return np.clip(np.rint(output), 0, 255).astype(np.uint8)


def _replay_pixels(
    provenance: Mapping[str, np.ndarray],
    sources_by_frame: Mapping[int, np.ndarray],
    parameters_by_frame: Mapping[int, tuple[np.ndarray, np.ndarray]],
    *,
    telemetry: dict[str, int] | None = None,
) -> np.ndarray:
    """Replay valid pixels with one decode and one compact remap per source.

    The reference implementation above remaps a full panorama-sized map once
    for primary and once for secondary contribution.  This exact path merges
    both coordinate lists for a frame, while retaining the original primary-
    then-secondary accumulation order.
    """
    valid = np.asarray(provenance["valid"], bool)
    output = np.zeros((*valid.shape, 3), np.float64)
    weight = np.asarray(provenance["secondary_weight"], np.float64)
    owner_frame = np.asarray(provenance["owner_frame_id"], np.int32)
    owner_u = np.asarray(provenance["source_u"], np.float32)
    owner_v = np.asarray(provenance["source_v"], np.float32)
    secondary_active = valid & (weight > 0.0)
    secondary_frame = np.asarray(provenance["secondary_frame_id"], np.int32)
    secondary_u = np.asarray(provenance["secondary_source_u"], np.float32)
    secondary_v = np.asarray(provenance["secondary_source_v"], np.float32)
    frame_ids = set(map(int, np.unique(owner_frame[valid])))
    if np.any(secondary_active):
        frame_ids.update(map(int, np.unique(secondary_frame[secondary_active])))
    available_frames = set(sources_by_frame)
    secondary_contributions: list[tuple[np.ndarray, np.ndarray]] = []
    for frame_id in sorted(frame_ids):
        if frame_id not in available_frames or frame_id not in parameters_by_frame:
            raise ValueError("provenance references an unbound raw RGB frame")
        primary = valid & (owner_frame == frame_id)
        secondary = secondary_active & (secondary_frame == frame_id)
        primary_count = int(np.count_nonzero(primary))
        secondary_count = int(np.count_nonzero(secondary))
        selected_u = np.concatenate((owner_u[primary], secondary_u[secondary]))
        selected_v = np.concatenate((owner_v[primary], secondary_v[secondary]))
        image = np.asarray(sources_by_frame[frame_id])
        if (
            np.any(selected_u < 0)
            or np.any(selected_v < 0)
            or np.any(selected_u > image.shape[1] - 1)
            or np.any(selected_v > image.shape[0] - 1)
        ):
            raise ValueError("provenance UV is outside raw RGB")
        sample_count = selected_u.size
        map_width = min(32_766, max(1, int(np.ceil(np.sqrt(sample_count)))))
        map_height = int(np.ceil(sample_count / map_width))
        u = np.full(map_height * map_width, -1, np.float32)
        v = np.full_like(u, -1)
        u[:sample_count], v[:sample_count] = selected_u, selected_v
        u, v = u.reshape(map_height, map_width), v.reshape(map_height, map_width)
        sampled = cv2.remap(
            image,
            u,
            v,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        ).reshape(-1, 3)[:sample_count]
        if telemetry is not None:
            telemetry["remap_invocation_count"] = telemetry.get("remap_invocation_count", 0) + 1
        gain, bias = parameters_by_frame[frame_id]
        corrected = np.rint(
            np.clip(sampled.astype(np.float64) / 255.0 * gain + bias, 0.0, 1.0) * 255.0
        )
        output[primary] += corrected[:primary_count] * (1.0 - weight[primary])[:, None]
        if secondary_count:
            secondary_contributions.append(
                (secondary, corrected[primary_count:] * weight[secondary][:, None])
            )
    for mask, contribution in secondary_contributions:
        output[mask] += contribution
    return np.clip(np.rint(output), 0, 255).astype(np.uint8)


def audit_m61_p3_files(
    p2_root: Path,
    p3_root: Path,
    *,
    expected_parent_sha256: str,
    expected_effective_config_sha256: str,
    expected_topology_sha256: str,
    raw_rgb_by_frame: Mapping[int, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Audit a pending or sealed P3 directory without changing it."""
    failures: list[str] = []
    try:
        seal = _json(p3_root / "seal_manifest.json")
        assets = seal.get("assets_sha256")
        if seal.get("schema") != SEAL_MANIFEST_SCHEMA or not isinstance(assets, dict):
            raise ValueError("invalid seal manifest")
        if not REQUIRED_ASSETS.issubset(assets) or any(not _allowed_asset(name) for name in assets):
            failures.append("sealed_asset_allowlist_mismatch")
        for name, digest in assets.items():
            path = p3_root / name
            if not path.is_file() or _sha(path) != digest:
                failures.append(f"sealed_asset_sha256_mismatch:{name}")

        parent = _json(p3_root / "p2_parent_reference.json")
        p2_completion_path = p2_root / "P2_completion.json"
        p2_completion = _json(p2_completion_path)
        if _sha(p2_completion_path) != expected_parent_sha256:
            failures.append("p2_completion_sha256_mismatch")
        p2_assets = p2_completion.get("assets_sha256")
        p2_result_asset = p2_completion.get("result_asset")
        p2_result_sha = p2_completion.get("result_asset_sha256")
        p2_result_relative = Path(p2_result_asset) if isinstance(p2_result_asset, str) else None
        p2_result_safe = bool(
            p2_result_relative is not None
            and not p2_result_relative.is_absolute()
            and ".." not in p2_result_relative.parts
            and p2_result_relative != Path(".")
        )
        p2_provenance_sha = (
            p2_assets.get("p2_pixel_provenance.npz") if isinstance(p2_assets, dict) else None
        )
        shared_parent_layout = p2_root == p3_root.parent / "P2"
        expected_completion_reference = (
            "../P2/P2_completion.json" if shared_parent_layout
            else str(p2_completion_path.resolve())
        )
        expected_result_reference = (
            f"../P2/{p2_result_asset}" if shared_parent_layout
            else str((p2_root / p2_result_relative).resolve()) if p2_result_safe else None
        )
        expected_provenance_reference = (
            "../P2/p2_pixel_provenance.npz" if shared_parent_layout
            else str((p2_root / "p2_pixel_provenance.npz").resolve())
        )
        if (
            parent.get("schema") != P2_PARENT_REFERENCE_SCHEMA
            or parent.get("parent_stage") != "P2"
            or parent.get("completion") != expected_completion_reference
            or parent.get("completion_sha256") != expected_parent_sha256
            or parent.get("p2_completion_sha256") != expected_parent_sha256
        ):
            failures.append("p2_parent_sha256_mismatch")
        if (
            not p2_result_safe
            or parent.get("result_asset") != expected_result_reference
            or parent.get("result_asset_sha256") != p2_result_sha
            or not (p2_root / p2_result_relative).is_file()
            or _sha(p2_root / p2_result_relative) != p2_result_sha
        ):
            failures.append("p2_parent_result_binding_mismatch")
        if (
            parent.get("pixel_provenance") != expected_provenance_reference
            or parent.get("pixel_provenance_sha256") != p2_provenance_sha
            or not isinstance(p2_provenance_sha, str)
            or _sha(p2_root / "p2_pixel_provenance.npz") != p2_provenance_sha
        ):
            failures.append("p2_parent_provenance_binding_mismatch")
        effective = _json(p3_root / "effective_config.json")
        if effective.get("effective_config_sha256") != expected_effective_config_sha256:
            failures.append("effective_config_sha256_mismatch")
        if effective.get("graph_topology_sha256") != expected_topology_sha256:
            failures.append("graph_topology_sha256_mismatch")
        topology_path = p2_root / "photometric_replay" / "graph_topology" / "topology.json"
        topology = _json(topology_path)
        if _sha(topology_path) != expected_topology_sha256 or topology.get("train_only") is not True:
            failures.append("sealed_train_topology_mismatch")
        topology_components = topology.get("components")
        if not isinstance(topology_components, list):
            failures.append("sealed_topology_components_invalid")
            topology_membership: list[list[int]] = []
        else:
            topology_membership = [list(map(int, component.get("source_indices", []))) for component in topology_components]
        performance = _json(p3_root / "performance.json")
        if performance.get("schema") != M61_PERFORMANCE_SCHEMA:
            failures.append("performance_schema_invalid")
        forbidden = performance.get("forbidden_invocations")
        if not isinstance(forbidden, dict) or any(value != 0 for value in forbidden.values()):
            failures.append("forbidden_invocation_nonzero")
        if performance.get("formal_render_attempt_count") != 1:
            failures.append("formal_render_attempt_count_invalid")
        if performance.get("hard_audit_render_invocations") != 0 or performance.get("fallback_identity_rebuild_count") != 0:
            failures.append("post_selection_render_or_identity_rebuild_detected")
        remaps = performance.get("formal_remap_count_by_source")
        if not isinstance(remaps, list) or not remaps or any(value != 1 for value in remaps):
            failures.append("formal_remap_once_invalid")

        p2 = _npz(p2_root / "p2_pixel_provenance.npz")
        p3 = _npz(p3_root / "p3_pixel_provenance.npz")
        for field in PRIMARY_FIELDS:
            if field not in p2 or field not in p3 or not _same(p2[field], p3[field]):
                failures.append(f"primary_field_mismatch:{field}")
        if any(field not in p3 for field in SECONDARY_FIELDS):
            failures.append("secondary_fields_missing")
            raise ValueError("secondary fields missing")
        valid = np.asarray(p3["valid"], bool)
        owner_frame = np.asarray(p3["owner_frame_id"])
        owner_source = np.asarray(p3["owner_source_index"])
        valid_mask_asset = cv2.imread(str(p3_root / "p3_valid_mask.png"), cv2.IMREAD_UNCHANGED)
        if (
            valid_mask_asset is None
            or valid_mask_asset.dtype != np.uint8
            or valid_mask_asset.ndim != 2
            or valid_mask_asset.shape != valid.shape
            or np.any((valid_mask_asset != 0) & (valid_mask_asset != 255))
            or not np.array_equal(valid_mask_asset == 255, valid)
        ):
            failures.append("p3_valid_mask_mismatch")
        if np.any(valid & ((owner_frame < 0) | (owner_source < 0))):
            failures.append("valid_pixel_missing_single_owner")
        weight = np.asarray(p3["secondary_weight"], np.float64)
        active = weight > 0.0
        if weight.shape != valid.shape or not np.isfinite(weight).all() or np.any(weight < 0) or np.any(weight >= 0.5):
            failures.append("secondary_weight_invalid")
        for field in SECONDARY_FIELDS[:-1]:
            if np.asarray(p3[field]).shape != valid.shape:
                failures.append(f"secondary_shape_invalid:{field}")
        secondary_frame = np.asarray(p3["secondary_frame_id"])
        secondary_source = np.asarray(p3["secondary_source_index"])
        secondary_u = np.asarray(p3["secondary_source_u"])
        secondary_v = np.asarray(p3["secondary_source_v"])
        if np.any(~active & ((secondary_frame != -1) | (secondary_source != -1))):
            failures.append("owner_only_secondary_identity_invalid")
        if np.any(active & ((secondary_frame < 0) | (secondary_source < 0))):
            failures.append("active_secondary_not_real")
        if np.any(active & (~np.isfinite(secondary_u) | ~np.isfinite(secondary_v))):
            failures.append("active_secondary_uv_nonfinite")
        if np.any(active & ~valid):
            failures.append("active_pixel_invalid")
        if "photometric_transaction_id" not in p3:
            failures.append("photometric_transaction_id_missing")
            photometric_transaction_id = np.full(valid.shape, -1, np.int32)
        else:
            photometric_transaction_id = np.asarray(p3["photometric_transaction_id"])
            if (
                photometric_transaction_id.shape != valid.shape
                or photometric_transaction_id.dtype != np.int32
                or np.any(~valid & (photometric_transaction_id != -1))
                or np.any(valid & (photometric_transaction_id != owner_source))
            ):
                failures.append("photometric_transaction_id_invalid")
        if "blend_transaction_id" not in p3:
            failures.append("blend_transaction_id_missing")
            blend_transaction_id = np.full(valid.shape, -1, np.int32)
        else:
            blend_transaction_id = np.asarray(p3["blend_transaction_id"])
            if (
                blend_transaction_id.shape != valid.shape
                or blend_transaction_id.dtype != np.int32
                or np.any(~active & (blend_transaction_id != -1))
                or np.any(active & (blend_transaction_id < 0))
            ):
                failures.append("blend_transaction_id_invalid")

        masks = _npz(p3_root / "pair_masks.npz")
        if set(masks) != {"active", "safe", "protected", "common_valid", "expected_valid"}:
            failures.append("pair_mask_fields_invalid")
        else:
            for name in masks:
                if masks[name].shape != valid.shape:
                    failures.append(f"pair_mask_shape_invalid:{name}")
            mask_active = np.asarray(masks["active"], bool)
            if not np.array_equal(mask_active, active):
                failures.append("active_pixel_mask_mismatch")
            eligible = np.asarray(masks["safe"], bool) & ~np.asarray(masks["protected"], bool) & np.asarray(masks["common_valid"], bool) & np.asarray(masks["expected_valid"], bool)
            if np.any(active & ~eligible):
                failures.append("active_pixel_outside_safe_eligibility")

        replay_manifest = _json(p2_root / "photometric_replay" / "manifest.json")
        replay_pairs = replay_manifest.get("pairs")
        pair_parent_bindings: dict[int, tuple[str, str]] = {}
        if (
            replay_manifest.get("schema") != M61_P2_PHOTOMETRIC_REPLAY_SCHEMA
            or not isinstance(replay_pairs, list)
        ):
            failures.append("p2_photometric_replay_manifest_invalid")
        else:
            for replay_entry in replay_pairs:
                pair_index = replay_entry.get("pair_index") if isinstance(replay_entry, dict) else None
                metadata_name = replay_entry.get("metadata") if isinstance(replay_entry, dict) else None
                asset_name = replay_entry.get("asset") if isinstance(replay_entry, dict) else None
                if (
                    not isinstance(pair_index, int)
                    or pair_index in pair_parent_bindings
                    or not isinstance(metadata_name, str)
                    or not isinstance(asset_name, str)
                ):
                    failures.append("p2_pair_replay_binding_invalid")
                    continue
                metadata_relative = Path(metadata_name)
                asset_relative = Path(asset_name)
                if (
                    metadata_relative.is_absolute()
                    or asset_relative.is_absolute()
                    or ".." in metadata_relative.parts
                    or ".." in asset_relative.parts
                ):
                    failures.append("p2_pair_replay_binding_invalid")
                    continue
                metadata_path = p2_root / "photometric_replay" / metadata_relative
                asset_path = p2_root / "photometric_replay" / asset_relative
                if (
                    not metadata_path.is_file()
                    or not asset_path.is_file()
                    or replay_entry.get("metadata_sha256") != _sha(metadata_path)
                    or replay_entry.get("asset_sha256") != _sha(asset_path)
                ):
                    failures.append("p2_pair_replay_binding_invalid")
                    continue
                metadata_document = _json(metadata_path)
                if metadata_document.get("pair_index") != pair_index:
                    failures.append("p2_pair_replay_binding_invalid")
                    continue
                narrow_sha = metadata_document.get("parent_narrow_replay_sha256")
                pair_transaction_sha = metadata_document.get("parent_pair_transaction_sha256")
                if not (
                    isinstance(narrow_sha, str) and len(narrow_sha) == 64
                    and isinstance(pair_transaction_sha, str) and len(pair_transaction_sha) == 64
                ):
                    failures.append("p2_pair_replay_binding_invalid")
                    continue
                pair_parent_bindings[pair_index] = (narrow_sha, pair_transaction_sha)

        transactions = _json(p3_root / "blend_transactions.json")
        if transactions.get("schema") != M61_BLEND_MANIFEST_SCHEMA:
            failures.append("blend_manifest_schema_invalid")
        if transactions.get("parent_completion_sha256") != expected_parent_sha256:
            failures.append("transaction_parent_sha256_mismatch")
        if transactions.get("pair_masks_sha256") != _sha(p3_root / "pair_masks.npz"):
            failures.append("transaction_pair_mask_sha256_mismatch")
        reasons = transactions.get("fallback_reasons", [])
        if not isinstance(reasons, list) or any(reason not in CLASSIFIED_FALLBACK_REASONS for reason in reasons):
            failures.append("fallback_reason_unclassified")
        pair_transactions = transactions.get("pairs", [])
        if not isinstance(pair_transactions, list):
            failures.append("pair_transaction_list_invalid")
        else:
            seen_pairs: set[int] = set()
            seen_transaction_ids: set[int] = set()
            for entry in pair_transactions:
                if not isinstance(entry, dict):
                    failures.append("pair_transaction_invalid")
                    continue
                pair_index = entry.get("pair_index")
                transaction_id = entry.get("transaction_id")
                asset = entry.get("asset")
                if not isinstance(pair_index, int) or pair_index < 0 or pair_index in seen_pairs:
                    failures.append("pair_transaction_order_or_identity_invalid")
                    continue
                seen_pairs.add(pair_index)
                if (
                    not isinstance(transaction_id, int)
                    or transaction_id < 0
                    or transaction_id in seen_transaction_ids
                    or transaction_id != pair_index
                ):
                    failures.append("blend_transaction_id_not_unique")
                    continue
                seen_transaction_ids.add(transaction_id)
                if asset != f"blend_transactions/pair_{pair_index:04d}.json" or assets.get(asset) != entry.get("asset_sha256"):
                    failures.append("pair_transaction_asset_binding_invalid")
                    continue
                document = _json(p3_root / asset)
                if (
                    document.get("schema") != M61_BLEND_TRANSACTION_SCHEMA
                    or document.get("pair_index") != pair_index
                    or document.get("transaction_id") != transaction_id
                    or document.get("parent_completion_sha256") != expected_parent_sha256
                ):
                    failures.append("pair_transaction_parent_invalid")
                if document.get("pair_masks_sha256") != _sha(p3_root / "pair_masks.npz"):
                    failures.append("pair_transaction_mask_sha256_mismatch")
                if pair_parent_bindings.get(pair_index) != (
                    document.get("parent_narrow_replay_sha256"),
                    document.get("parent_pair_transaction_sha256"),
                ):
                    failures.append("pair_transaction_replay_binding_invalid")
                if (
                    entry.get("model") != document.get("model")
                    or entry.get("active_pixel_count") != document.get("active_pixel_count")
                ):
                    failures.append("blend_manifest_declaration_mismatch")
                traced_count = int(np.count_nonzero(blend_transaction_id == transaction_id))
                if traced_count != document.get("active_pixel_count"):
                    failures.append("blend_transaction_pixel_trace_mismatch")
            if transactions.get("all_pairs_reported") is not True:
                failures.append("blend_manifest_incomplete")
            if seen_pairs != set(pair_parent_bindings):
                failures.append("blend_manifest_pair_set_mismatch")
            active_ids = set(map(int, np.unique(blend_transaction_id[active]))) if np.any(active) else set()
            if active_ids != {
                transaction_id for transaction_id in seen_transaction_ids
                if np.any(blend_transaction_id == transaction_id)
            }:
                failures.append("blend_transaction_pixel_identity_unbound")

        solution = _json(p3_root / "photometric_solution.json")
        if solution.get("graph_topology_sha256") != expected_topology_sha256:
            failures.append("solution_topology_sha256_mismatch")
        components = solution.get("components")
        sources = solution.get("sources")
        if not isinstance(components, list) or not isinstance(sources, list):
            failures.append("solution_component_structure_invalid")
            raise ValueError("solution structure invalid")
        membership = [int(source) for component in components for source in component.get("source_indices", [])]
        source_indices = [int(source["source_index"]) for source in sources]
        if sorted(membership) != sorted(source_indices) or len(membership) != len(set(membership)):
            failures.append("solution_component_membership_invalid")
        solution_membership = [list(map(int, component.get("source_indices", []))) for component in components]
        if solution_membership != topology_membership:
            failures.append("solution_changed_sealed_topology_membership")
        sources_by_index = {int(source["source_index"]): source for source in sources}
        identity_inside = any(
            component.get("selected_model") != "Q0_identity"
            and any(sources_by_index[index].get("model") == "Q0_identity" for index in component.get("source_indices", []))
            for component in components
        )
        if identity_inside:
            failures.append("identity_fallback_inside_solved_component")
        parameters: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        photometric_transaction_ids: set[int] = set()
        bounds = effective.get("photometric_bounds", {})
        minimum_gain = float(bounds.get("minimum_gain", 0.80))
        maximum_gain = float(bounds.get("maximum_gain", 1.25))
        maximum_bias = float(bounds.get("maximum_absolute_bias_linear", 0.03))
        for source in sources:
            frame_id = int(source["frame_id"])
            source_index = int(source["source_index"])
            transaction_id = source.get("photometric_transaction_id")
            gain = np.asarray(source["gain_bgr"], np.float64)
            bias = np.asarray(source["bias_bgr_linear"], np.float64)
            parameter = {
                "source_index": source_index,
                "frame_id": frame_id,
                "model": source.get("model"),
                "gain_bgr": source.get("gain_bgr"),
                "bias_bgr_linear": source.get("bias_bgr_linear"),
            }
            parameter_sha256 = hashlib.sha256(json.dumps(
                parameter, sort_keys=True, separators=(",", ":"), allow_nan=False,
            ).encode("utf-8")).hexdigest()
            if (
                transaction_id != source_index
                or transaction_id in photometric_transaction_ids
                or source.get("parameter_sha256") != parameter_sha256
            ):
                failures.append("photometric_transaction_binding_invalid")
            else:
                photometric_transaction_ids.add(transaction_id)
            if gain.shape != (3,) or bias.shape != (3,) or not np.isfinite(gain).all() or not np.isfinite(bias).all():
                failures.append("solution_parameter_invalid")
            elif np.any(gain < minimum_gain) or np.any(gain > maximum_gain) or np.any(np.abs(bias) > maximum_bias):
                failures.append("solution_parameter_out_of_bounds")
            parameters[frame_id] = (gain, bias)
        if not set(map(int, np.unique(photometric_transaction_id[valid]))).issubset(
            photometric_transaction_ids
        ):
            failures.append("photometric_transaction_pixel_trace_mismatch")

        visual = _load_color(p3_root / "visual_panorama.png")
        owner_only = _load_color(p3_root / "photometric_owner_only.png")
        p2_image = _load_color(p2_root / p2_result_relative)
        if (
            visual.shape != (*valid.shape, 3)
            or owner_only.shape != visual.shape
            or p2_image.shape != visual.shape
        ):
            failures.append("p3_image_or_valid_mask_shape_mismatch")
        q0_b0 = solution.get("selected_model") == "Q0_identity" and not np.any(active)
        if q0_b0 and (not np.array_equal(visual, p2_image) or not np.array_equal(owner_only, p2_image)):
            failures.append("q0_b0_canonical_equality_failed")
        if raw_rgb_by_frame is not None:
            replayed = _replay_pixels(p3, raw_rgb_by_frame, parameters)
            if not np.array_equal(replayed[valid], visual[valid]):
                failures.append("independent_raw_rgb_replay_mismatch")
    except (ValueError, KeyError, TypeError, IndexError) as exc:
        failures.append(f"malformed_asset:{type(exc).__name__}")
    failures = list(dict.fromkeys(failures))
    return {
        "schema": HARD_AUDIT_SCHEMA,
        "passed": not failures,
        "failures": failures,
        "failure_action": "leave_pointer_at_p2_no_fallback_no_second_render" if failures else None,
        "candidate_selection_authority": False,
        "identity_rebuild_allowed": False,
        "formal_render_attempts_after_audit": 0,
        "parent_completion_sha256": expected_parent_sha256,
        "effective_config_sha256": expected_effective_config_sha256,
        "graph_topology_sha256": expected_topology_sha256,
        "forbidden_invocations": forbidden if "forbidden" in locals() and isinstance(forbidden, dict) else {},
    }


def promote_verified_pending(
    pending: Path,
    final: Path,
    pointer_path: Path,
    audit: Mapping[str, Any],
    *,
    generation_id: str,
) -> dict[str, Any]:
    """Seal, promote, finally verify, and only then advance the shared pointer."""
    if audit.get("schema") != HARD_AUDIT_SCHEMA or audit.get("passed") is not True:
        raise ValueError("P3 pending tree did not pass independent hard audit")
    if final.exists() or not pending.is_dir():
        raise FileExistsError("P3 promotion requires an existing pending and absent final path")
    if pointer_path.name != "current_latest.json":
        raise ValueError("P3 promotion pointer must be current_latest.json")
    audit_path = pending / "hard_audit.json"
    atomic_write_json(audit_path, dict(audit))
    result_asset = "visual_panorama.png"
    result_path = pending / result_asset
    provenance_path = pending / "p3_pixel_provenance.npz"
    if not result_path.is_file() or not provenance_path.is_file():
        raise ValueError("P3 promotion result or provenance asset is missing")
    seal_stage(
        pending,
        completion_name="P3_completion.json",
        schema=COMPLETION_SCHEMA,
        metadata={
            "stage": "P3",
            "generation_id": generation_id,
            "hard_audit_passed": True,
            "parent_stage": "P2",
            "parent_completion_sha256": audit["parent_completion_sha256"],
            "result_asset": result_asset,
            "result_asset_sha256": _sha(result_path),
            "pixel_provenance_sha256": _sha(provenance_path),
            "hard_audit_sha256": _sha(audit_path),
            "effective_config_sha256": audit["effective_config_sha256"],
            "graph_topology_sha256": audit["graph_topology_sha256"],
            "diagnostic_only": True,
            "production": False,
            "production_eligible": False,
            "production_lock_eligible": False,
        },
    )
    verify_stage(pending, completion_name="P3_completion.json", schema=COMPLETION_SCHEMA)
    os.replace(pending, final)
    try:
        verify_promoted_m61_p3(
            final.parent / "P2",
            final,
            expected_parent_sha256=str(audit["parent_completion_sha256"]),
            expected_effective_config_sha256=str(audit["effective_config_sha256"]),
            expected_topology_sha256=str(audit["graph_topology_sha256"]),
        )
        return update_current_latest(
            pointer_path.parent,
            final.parent,
            stage="P3",
            completion_name="P3_completion.json",
            completion_schema=COMPLETION_SCHEMA,
            result_asset=result_asset,
        )
    except Exception:
        if final.exists():
            shutil.rmtree(final)
        raise


def verify_promoted_m61_p3(
    p2_root: Path,
    p3_root: Path,
    *,
    expected_parent_sha256: str,
    expected_effective_config_sha256: str,
    expected_topology_sha256: str,
    raw_rgb_by_frame: Mapping[int, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Re-run the independent audit and verify completion/audit SHA bindings."""
    completion = verify_stage(
        p3_root, completion_name="P3_completion.json", schema=COMPLETION_SCHEMA,
    )
    audit_path = p3_root / "hard_audit.json"
    sealed_audit = _json(audit_path)
    if (
        completion.get("stage") != "P3"
        or completion.get("parent_stage") != "P2"
        or completion.get("parent_completion_sha256") != expected_parent_sha256
        or completion.get("hard_audit_passed") is not True
        or completion.get("diagnostic_only") is not True
        or completion.get("production") is not False
        or completion.get("production_eligible") is not False
        or completion.get("production_lock_eligible") is not False
        or completion.get("effective_config_sha256") != expected_effective_config_sha256
        or completion.get("graph_topology_sha256") != expected_topology_sha256
    ):
        raise ValueError("invalid P3 completion")
    assets = completion.get("assets_sha256")
    result_asset = completion.get("result_asset")
    if (
        result_asset != "visual_panorama.png"
        or not isinstance(assets, dict)
        or completion.get("result_asset_sha256") != _sha(p3_root / result_asset)
        or assets.get(result_asset) != completion.get("result_asset_sha256")
        or completion.get("pixel_provenance_sha256") != _sha(
            p3_root / "p3_pixel_provenance.npz"
        )
        or assets.get("p3_pixel_provenance.npz") != completion.get(
            "pixel_provenance_sha256"
        )
    ):
        raise ValueError("P3 completion result/provenance binding is invalid")
    if completion.get("hard_audit_sha256") != _sha(audit_path):
        raise ValueError("P3 completion hard-audit SHA mismatch")
    if sealed_audit.get("schema") != HARD_AUDIT_SCHEMA or sealed_audit.get("passed") is not True:
        raise ValueError("sealed P3 hard audit did not pass")
    replayed = audit_m61_p3_files(
        p2_root, p3_root,
        expected_parent_sha256=expected_parent_sha256,
        expected_effective_config_sha256=expected_effective_config_sha256,
        expected_topology_sha256=expected_topology_sha256,
        raw_rgb_by_frame=raw_rgb_by_frame,
    )
    if replayed.get("passed") is not True:
        raise ValueError("promoted P3 no longer passes independent replay")
    return completion


__all__ = [
    "HARD_AUDIT_SCHEMA", "P2_PARENT_REFERENCE_SCHEMA", "audit_m61_p3_files", "promote_verified_pending",
    "verify_promoted_m61_p3",
]
