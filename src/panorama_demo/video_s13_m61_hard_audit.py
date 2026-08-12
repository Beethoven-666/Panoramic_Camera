"""Fail-closed independent verifier and promotion state for S013 M6.1 P3.

The verifier consumes only sealed files.  It never calls the renderer, solver,
or fallback path, so a failed audit cannot rebuild an identity panorama.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np


HARD_AUDIT_SCHEMA = "gemini305-video-s13-p3-hard-audit/v2"
COMPLETION_SCHEMA = "gemini305-video-s13-p3-visual-completion/v2"
POINTER_SCHEMA = "gemini305-video-s13-current-stage/v2"
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


def _replay_pixels(
    provenance: Mapping[str, np.ndarray],
    sources_by_frame: Mapping[int, np.ndarray],
    parameters_by_frame: Mapping[int, tuple[np.ndarray, np.ndarray]],
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
            linear = sampled[active].astype(np.float64) / 255.0
            corrected = np.rint(np.clip(linear * gain + bias, 0.0, 1.0) * 255.0)
            alpha = weight[active] if prefix else 1.0 - weight[active]
            output[active] += corrected * alpha[:, None]

    sample("", valid)
    sample("secondary_", valid & (weight > 0.0))
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
        if seal.get("schema") != "gemini305-video-s13-p3-seal-manifest/v2" or not isinstance(assets, dict):
            raise ValueError("invalid seal manifest")
        if not REQUIRED_ASSETS.issubset(assets) or any(not _allowed_asset(name) for name in assets):
            failures.append("sealed_asset_allowlist_mismatch")
        for name, digest in assets.items():
            path = p3_root / name
            if not path.is_file() or _sha(path) != digest:
                failures.append(f"sealed_asset_sha256_mismatch:{name}")

        parent = _json(p3_root / "p2_parent_reference.json")
        if parent.get("p2_completion_sha256") != expected_parent_sha256:
            failures.append("p2_parent_sha256_mismatch")
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

        transactions = _json(p3_root / "blend_transactions.json")
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
            for entry in pair_transactions:
                if not isinstance(entry, dict):
                    failures.append("pair_transaction_invalid")
                    continue
                pair_index = entry.get("pair_index")
                asset = entry.get("asset")
                if not isinstance(pair_index, int) or pair_index < 0 or pair_index in seen_pairs:
                    failures.append("pair_transaction_order_or_identity_invalid")
                    continue
                seen_pairs.add(pair_index)
                if asset != f"blend_transactions/pair_{pair_index:04d}.json" or assets.get(asset) != entry.get("asset_sha256"):
                    failures.append("pair_transaction_asset_binding_invalid")
                    continue
                document = _json(p3_root / asset)
                if document.get("pair_index") != pair_index or document.get("parent_completion_sha256") != expected_parent_sha256:
                    failures.append("pair_transaction_parent_invalid")
                if document.get("pair_masks_sha256") != _sha(p3_root / "pair_masks.npz"):
                    failures.append("pair_transaction_mask_sha256_mismatch")

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
        bounds = effective.get("photometric_bounds", {})
        minimum_gain = float(bounds.get("minimum_gain", 0.80))
        maximum_gain = float(bounds.get("maximum_gain", 1.25))
        maximum_bias = float(bounds.get("maximum_absolute_bias_linear", 0.03))
        for source in sources:
            frame_id = int(source["frame_id"])
            gain = np.asarray(source["gain_bgr"], np.float64)
            bias = np.asarray(source["bias_bgr_linear"], np.float64)
            if gain.shape != (3,) or bias.shape != (3,) or not np.isfinite(gain).all() or not np.isfinite(bias).all():
                failures.append("solution_parameter_invalid")
            elif np.any(gain < minimum_gain) or np.any(gain > maximum_gain) or np.any(np.abs(bias) > maximum_bias):
                failures.append("solution_parameter_out_of_bounds")
            parameters[frame_id] = (gain, bias)

        visual = _load_color(p3_root / "visual_panorama.png")
        owner_only = _load_color(p3_root / "photometric_owner_only.png")
        p2_image = _load_color(p2_root / "geometry_and_seam_panorama_owner_only.png")
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
    """Atomically promote only an already-passing pending tree, then its pointer."""
    if audit.get("schema") != HARD_AUDIT_SCHEMA or audit.get("passed") is not True:
        raise ValueError("P3 pending tree did not pass independent hard audit")
    if final.exists() or not pending.is_dir():
        raise FileExistsError("P3 promotion requires an existing pending and absent final path")
    audit_path = pending / "hard_audit.json"
    audit_path.write_text(json.dumps(dict(audit), sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    completion = {
        "schema": COMPLETION_SCHEMA, "stage": "P3", "generation_id": generation_id,
        "hard_audit_sha256": _sha(audit_path), "parent_completion_sha256": audit["parent_completion_sha256"],
        "effective_config_sha256": audit["effective_config_sha256"],
    }
    (pending / "P3_completion.json").write_text(json.dumps(completion, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(pending, final)
    pointer = {
        "schema": POINTER_SCHEMA, "stage": "P3", "generation_id": generation_id,
        "completion_sha256": _sha(final / "P3_completion.json"),
    }
    temporary = pointer_path.with_name(f".{pointer_path.name}.pending")
    temporary.write_text(json.dumps(pointer, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, pointer_path)
    return pointer


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
    completion = _json(p3_root / "P3_completion.json")
    audit_path = p3_root / "hard_audit.json"
    sealed_audit = _json(audit_path)
    if completion.get("schema") != COMPLETION_SCHEMA or completion.get("stage") != "P3":
        raise ValueError("invalid P3 completion")
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
    "HARD_AUDIT_SCHEMA", "audit_m61_p3_files", "promote_verified_pending",
    "verify_promoted_m61_p3",
]
