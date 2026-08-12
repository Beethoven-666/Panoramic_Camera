"""Authorized manual-forward S013 M7 preflight, Core evaluation, and P4 gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from .video_algorithm import canonical_config_sha256, load_algorithm_config
from .video_s13_bundle import (
    atomic_write_json,
    seal_stage,
    sha256_file,
    verify_stage,
    write_image,
    write_npz,
)
from .video_s13_p4_hard_audit import P4_COMPLETION_SCHEMA, audit_s13_p4_arrays
from .video_s13_repair import (
    CORE_CANDIDATES,
    R0_KEEP_P3,
    R1_B1_TO_B0,
    R2_SEAM_DOWNGRADE,
    R3_GEOMETRY_DOWNGRADE,
    ghost_material_count,
    r1_b1_to_b0_owner_only,
    validate_core_registry,
)


AUTHORIZATION_SCHEMA = "gemini305-video-s13-m7-manual-forward-authorization/v1"
HANDOFF_SCHEMA = "gemini305-video-s13-m7-handoff/v2"
P2_COMPLETION_SCHEMA = "gemini305-video-s13-p2-completion/v4"
P3_COMPLETION_SCHEMA = "gemini305-video-s13-p3-visual-completion/v2"
P3_SEAL_SCHEMA = "gemini305-video-s13-p3-seal-manifest/v2"
P3_AUDIT_SCHEMA = "gemini305-video-s13-p3-hard-audit/v2"
PREFLIGHT_SCHEMA = "gemini305-video-s13-m7-preflight/v2"
SUMMARY_SCHEMA = "gemini305-video-s13-m7-validation-summary/v1"
FORMAL_CANDIDATE_ID = "S013_output_first_progressive_dense_central_slit_v4"
FORMAL_IMPLEMENTATION_ID = "s013_output_first_progressive_dense_central_slit_m61_v2"
BRANCHES = ("fast_direct", "fast_ignore_pose", "slow_direct", "slow_ignore_pose")
ALLOWED_COMPONENT_CLASSES = frozenset({"protected", "geometry", "seam", "owner"})
DENIED_COMPONENT_TERMS = (
    "photometric", "brightness", "chroma", "white_balance", "white-balance",
    "metric_calibration", "q0", "q1", "q2", "q3", "q4", "threshold",
    "multiband", "expanded_feather", "source_set", "source_rescue", "insert_frame",
    "interpolated_frame", "cross_pair", "joint_repair",
)


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"S013 M7 missing or invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"S013 M7 JSON root must be an object: {path}")
    return value


def _npz(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            return {name: np.asarray(archive[name]) for name in archive.files}
    except (OSError, ValueError, KeyError) as exc:
        raise ValueError(f"S013 M7 missing or invalid NPZ: {path}") from exc


def _canonical_sha(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _safe_relative(root: Path, value: str, *, label: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts or relative == Path("."):
        raise ValueError(f"S013 M7 unsafe {label} path")
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"S013 M7 {label} escapes the acceptance root") from exc
    return path


def _verify_p3_seal(p3: Path) -> dict[str, str]:
    seal = _json(p3 / "seal_manifest.json")
    assets = seal.get("assets_sha256")
    if seal.get("schema") != P3_SEAL_SCHEMA or not isinstance(assets, dict):
        raise ValueError("S013 M7 P3 seal manifest is invalid")
    for name, expected in assets.items():
        relative = Path(str(name))
        if relative.is_absolute() or ".." in relative.parts or not isinstance(expected, str):
            raise ValueError("S013 M7 P3 seal contains an unsafe asset")
        asset = p3 / relative
        if not asset.is_file() or sha256_file(asset) != expected:
            raise ValueError(f"S013 M7 P3 sealed asset changed: {name}")
    for required in (
        "visual_panorama.png", "photometric_owner_only.png", "p3_pixel_provenance.npz",
        "p3_valid_mask.png", "hard_audit.json", "effective_config.json",
        "photometric_solution.json", "blend_transactions.json", "pair_masks.npz",
    ):
        if required not in assets and required != "hard_audit.json":
            raise ValueError(f"S013 M7 P3 seal lacks required asset: {required}")
        if not (p3 / required).is_file():
            raise ValueError(f"S013 M7 P3 lacks required asset: {required}")
    return {str(name): str(digest) for name, digest in assets.items()}


def _verify_branch(acceptance_root: Path, branch: str) -> dict[str, Any]:
    formal = acceptance_root / "runs_v4" / branch / "formal"
    p3 = formal / "P3"
    pointer_path = formal / "current_latest.json"
    pointer = _json(pointer_path)
    completion_path = p3 / "P3_completion.json"
    completion = _json(completion_path)
    completion_sha = sha256_file(completion_path)
    if (
        pointer.get("stage") != "P3"
        or pointer.get("completion_sha256") != completion_sha
        or completion.get("schema") != P3_COMPLETION_SCHEMA
        or completion.get("stage") != "P3"
        or completion.get("generation_id") != pointer.get("generation_id")
    ):
        raise ValueError(f"S013 M7 {branch} current_latest is not the accepted sealed P3")
    _verify_p3_seal(p3)
    audit_path = p3 / "hard_audit.json"
    audit = _json(audit_path)
    if (
        audit.get("schema") != P3_AUDIT_SCHEMA
        or audit.get("passed") is not True
        or completion.get("hard_audit_sha256") != sha256_file(audit_path)
        or audit.get("effective_config_sha256") != completion.get("effective_config_sha256")
    ):
        raise ValueError(f"S013 M7 {branch} P3 hard audit binding is invalid")
    parent_reference = _json(p3 / "p2_parent_reference.json")
    p2 = Path(str(parent_reference.get("p2_root", ""))).resolve()
    if not p2.is_dir():
        raise ValueError(f"S013 M7 {branch} P2 parent is missing")
    p2_completion_path = p2 / "P2_completion.json"
    p2_completion = _json(p2_completion_path)
    p2_completion_sha = sha256_file(p2_completion_path)
    if (
        p2_completion.get("schema") != P2_COMPLETION_SCHEMA
        or parent_reference.get("p2_completion_sha256") != p2_completion_sha
        or completion.get("parent_completion_sha256") != p2_completion_sha
    ):
        raise ValueError(f"S013 M7 {branch} P2-v4 parent binding is invalid")
    p2_assets = p2_completion.get("assets_sha256")
    if not isinstance(p2_assets, dict):
        raise ValueError(f"S013 M7 {branch} P2-v4 asset seal is missing")
    result_asset = str(p2_completion.get("result_asset", "geometry_and_seam_panorama_owner_only.png"))
    p2_result = p2 / result_asset
    p2_provenance = p2 / "p2_pixel_provenance.npz"
    for asset, expected in (
        (p2_result, p2_assets.get(result_asset)),
        (p2_provenance, p2_assets.get("p2_pixel_provenance.npz")),
    ):
        if not asset.is_file() or not isinstance(expected, str) or sha256_file(asset) != expected:
            raise ValueError(f"S013 M7 {branch} sealed P2 asset changed")
    solution = _json(p3 / "photometric_solution.json")
    source_count = len(solution.get("sources", [])) if isinstance(solution.get("sources"), list) else 0
    if source_count <= 1:
        raise ValueError(f"S013 M7 {branch} P3 source set is invalid")
    quality_path = formal / "post_quality_final" / "post_quality.json"
    quality = _json(quality_path)
    return {
        "branch": branch,
        "run_root": str(formal.resolve()),
        "generation_id": str(pointer["generation_id"]),
        "pointer_sha256": sha256_file(pointer_path),
        "p2_root": str(p2),
        "p2_completion_sha256": p2_completion_sha,
        "p2_result_asset": result_asset,
        "p2_result_sha256": sha256_file(p2_result),
        "p2_provenance_sha256": sha256_file(p2_provenance),
        "p3_completion_sha256": completion_sha,
        "p3_result_sha256": sha256_file(p3 / "visual_panorama.png"),
        "p3_owner_only_sha256": sha256_file(p3 / "photometric_owner_only.png"),
        "p3_provenance_sha256": sha256_file(p3 / "p3_pixel_provenance.npz"),
        "p3_hard_audit_sha256": sha256_file(audit_path),
        "p3_seal_manifest_sha256": sha256_file(p3 / "seal_manifest.json"),
        "effective_config_sha256": str(completion["effective_config_sha256"]),
        "photometric_solution_sha256": sha256_file(p3 / "photometric_solution.json"),
        "source_set_sha256": _canonical_sha(solution["sources"]),
        "source_count": source_count,
        "automatic_quality_decision_sha256": sha256_file(quality_path),
        "automatic_quality_state": str(quality.get("quality_state")),
    }


def _current_contract_binding(repo_root: Path) -> dict[str, str]:
    config_path = repo_root / "configs/video_candidates/s013/S013_output_first_progressive_dense_central_slit_v4.yaml"
    manifest_path = repo_root / "configs/video_candidates/s013/candidate_manifest.json"
    document = load_algorithm_config(config_path)
    if (
        document.get("algorithm_id") != FORMAL_CANDIDATE_ID
        or document.get("implementation_id") != FORMAL_IMPLEMENTATION_ID
    ):
        raise ValueError("S013 M7 formal v4 candidate identity changed")
    canonical = canonical_config_sha256(document)
    if document.get("config_sha256") != canonical:
        raise ValueError("S013 M7 formal v4 candidate self hash is invalid")
    manifest = _json(manifest_path)
    entry = manifest.get("candidates", {}).get(FORMAL_CANDIDATE_ID)
    if not isinstance(entry, dict) or entry.get("config_sha256") != canonical:
        raise ValueError("S013 M7 candidate manifest does not bind the v4 contract")
    return {
        "candidate_config": config_path.relative_to(repo_root).as_posix(),
        "candidate_config_sha256": canonical,
        "candidate_manifest": manifest_path.relative_to(repo_root).as_posix(),
        "candidate_manifest_file_sha256": sha256_file(manifest_path),
        "candidate_manifest_canonical_sha256": str(manifest.get("manifest_sha256")),
    }


def build_manual_forward_authorization(
    acceptance_root: Path,
    *,
    user_decision_text: str,
    authorized_by: str = "workspace_user",
    authorized_at_utc: str | None = None,
) -> dict[str, Any]:
    acceptance_root = acceptance_root.resolve()
    blocked_path = acceptance_root / "blocked_report.json"
    blocked_sha_before = sha256_file(blocked_path)
    blocked = _json(blocked_path)
    if (
        blocked.get("stage_exit") != "blocked"
        or blocked.get("m7_handoff_eligible") is not False
        or blocked.get("manual_acceptance_granted") is not False
    ):
        raise ValueError("S013 M7 manual forward requires the preserved automatic blocked record")
    branches = [_verify_branch(acceptance_root, branch) for branch in BRANCHES]
    repo_root = Path(__file__).resolve().parents[2]
    contract = _current_contract_binding(repo_root)
    timestamp = authorized_at_utc or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    if not user_decision_text.strip() or not authorized_by.strip():
        raise ValueError("S013 M7 manual forward requires explicit user decision and author identity")
    payload: dict[str, Any] = {
        "schema": AUTHORIZATION_SCHEMA,
        "candidate_id": FORMAL_CANDIDATE_ID,
        "implementation_id": FORMAL_IMPLEMENTATION_ID,
        "m6_stage_exit": "completed_manual_forward",
        "accepted_parent_stage": "P3",
        "automatic_visual_status": "blocked",
        "automatic_blocked_report": "blocked_report.json",
        "automatic_blocked_report_sha256": blocked_sha_before,
        "automatic_blocked_m7_handoff_eligible_preserved": False,
        "automatic_quality_records_preserved": True,
        "manual_acceptance_granted": True,
        "authorized_by": authorized_by.strip(),
        "authorized_at_utc": timestamp,
        "user_decision_text": user_decision_text,
        "user_decision_sha256": hashlib.sha256(user_decision_text.encode("utf-8")).hexdigest(),
        "m7_authorization_state": "authorized_manual_forward",
        "allowed_component_classes": ["geometry", "seam", "owner", "protected"],
        "denied_problem_classes": [
            "systemic_photometric", "brightness", "chroma", "white_balance",
            "Q0_Q4_resolve", "threshold_relaxation", "expanded_feather", "MultiBand",
            "adjacent_multi_seam_joint_repair", "cross_pair_joint_repair", "source_set_change",
            "inserted_or_interpolated_frame", "all_optional_candidates",
        ],
        "core_only": True,
        "q0_q3_frozen": True,
        "q4_and_low_frequency_field_disabled": True,
        "thresholds_frozen": True,
        "source_set_frozen": True,
        "optional_gates": {
            "constrained_two_label_graphcut": False,
            "depth_risk_veto": False,
            "bounded_local_mesh": False,
        },
        "automatic_quality_bindings": {
            "candidate_manifest_sha256": blocked.get("candidate_manifest_sha256"),
            "effective_config_sha256": blocked.get("effective_config_sha256"),
            "final_threshold_sha256": blocked.get("final_threshold_sha256"),
            "approval_witness_sha256": blocked.get("approval_witness_sha256"),
        },
        "manual_forward_contract": contract,
        "accepted_p3_branches": branches,
        "diagnostic_only": True,
        "runtime_authority": False,
        "production_eligible": False,
        "production_lock_eligible": False,
    }
    path = acceptance_root / "M6_manual_forward_authorization.json"
    atomic_write_json(path, payload)
    if sha256_file(blocked_path) != blocked_sha_before:
        raise ValueError("S013 M7 manual forward changed the automatic blocked evidence")
    return payload


def _load_color(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"S013 M7 missing image: {path}")
    return image


def _ghost_seed_mask(reference: np.ndarray, candidate: np.ndarray, active: np.ndarray) -> np.ndarray:
    reference_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY).astype(np.float32)
    candidate_gray = cv2.cvtColor(candidate, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g0 = np.hypot(
        cv2.Sobel(reference_gray, cv2.CV_32F, 1, 0),
        cv2.Sobel(reference_gray, cv2.CV_32F, 0, 1),
    )
    g1 = np.hypot(
        cv2.Sobel(candidate_gray, cv2.CV_32F, 1, 0),
        cv2.Sobel(candidate_gray, cv2.CV_32F, 0, 1),
    )
    return np.asarray(active, bool) & (g1 > g0 * 1.5 + 4) & (g0 >= 2)


def _branch_components(
    acceptance_root: Path,
    branch_binding: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    branch = str(branch_binding["branch"])
    formal = Path(str(branch_binding["run_root"]))
    p3 = formal / "P3"
    p2 = Path(str(branch_binding["p2_root"]))
    p2_image = _load_color(p2 / str(branch_binding["p2_result_asset"]))
    p3_image = _load_color(p3 / "visual_panorama.png")
    provenance = _npz(p3 / "p3_pixel_provenance.npz")
    owner = np.asarray(provenance["owner_source_index"], np.int32)
    secondary = np.asarray(provenance["secondary_source_index"], np.int32)
    active = np.asarray(provenance["secondary_weight"], np.float32) > 0
    pair_identity = np.minimum(owner, secondary)
    ghost = _ghost_seed_mask(p2_image, p3_image, active)
    pair_ids = sorted(int(value) for value in np.unique(pair_identity[ghost]) if value >= 0)
    replay = _json(p2 / "p2_replay_manifest.json")
    replay_pairs = replay.get("pairs")
    if not isinstance(replay_pairs, list):
        raise ValueError(f"S013 M7 {branch} P2 replay manifest is invalid")
    pair_by_index = {int(row["pair_index"]): row for row in replay_pairs if isinstance(row, dict)}
    maximum_components = max(3, math.ceil(0.05 * len(replay_pairs)))
    if len(pair_ids) > maximum_components:
        return [], [{
            "branch": branch,
            "reason": "local_ghost_pair_count_exceeds_manual_forward_component_cap",
            "pair_indices": pair_ids,
        }]
    components: list[dict[str, Any]] = []
    out_of_scope: list[dict[str, Any]] = []
    for pair_index in pair_ids:
        if any(abs(pair_index - other) == 1 for other in pair_ids):
            out_of_scope.append({
                "branch": branch, "pair_index": pair_index,
                "reason": "adjacent_residual_pairs_are_out_of_scope",
            })
            continue
        pair_mask = active & (pair_identity == pair_index)
        seed = ghost & (pair_identity == pair_index)
        if not np.any(pair_mask) or not np.any(seed):
            continue
        blend_transaction = p3 / "blend_transactions" / f"pair_{pair_index:04d}.json"
        blend_document = _json(blend_transaction)
        if blend_document.get("model") != "B1_narrow_feather_2px" or blend_document.get("total_width_px") != 2:
            out_of_scope.append({
                "branch": branch, "pair_index": pair_index,
                "reason": "local_signal_is_not_a_frozen_B1_2px_owner_component",
            })
            continue
        pair_row = pair_by_index.get(pair_index)
        if not isinstance(pair_row, dict):
            raise ValueError(f"S013 M7 {branch} pair {pair_index} lacks replay identity")
        parent_transaction = p2 / str(pair_row["parent_pair_transaction"])
        if sha256_file(parent_transaction) != pair_row.get("parent_pair_transaction_sha256"):
            raise ValueError(f"S013 M7 {branch} pair transaction changed")
        y, x = np.where(pair_mask)
        x0, x1 = max(0, int(x.min()) - 2), min(pair_mask.shape[1], int(x.max()) + 3)
        y0, y1 = max(0, int(y.min()) - 2), min(pair_mask.shape[0], int(y.max()) + 3)
        item_id = f"{branch}-owner-{pair_index:04d}"
        evidence_root = acceptance_root / "M7_handoff_evidence" / branch / item_id
        evidence_root.mkdir(parents=True, exist_ok=True)
        p2_crop = evidence_root / "p2_crop.png"
        p3_crop = evidence_root / "p3_crop.png"
        mask_path = evidence_root / "affected_mask.npz"
        write_image(p2_crop, p2_image[y0:y1, x0:x1])
        write_image(p3_crop, p3_image[y0:y1, x0:x1])
        write_npz(mask_path, {
            "affected_mask": pair_mask,
            "ghost_seed_mask": seed,
            "frozen_b1_active_mask": pair_mask,
        })
        evidence = {
            "schema": "gemini305-video-s13-m7-component-evidence/v1",
            "handoff_item_id": item_id,
            "branch": branch,
            "pair_index": pair_index,
            "classification": "owner",
            "risk": "blend_induced_local_ghost",
            "frozen_metric": "m61_gradient_ghost_predicate",
            "ghost_seed_pixel_count": int(np.count_nonzero(seed)),
            "affected_b1_pixel_count": int(np.count_nonzero(pair_mask)),
            "bbox_xyxy": [x0, y0, x1, y1],
            "p2_crop_sha256": sha256_file(p2_crop),
            "p3_crop_sha256": sha256_file(p3_crop),
            "affected_mask_sha256": sha256_file(mask_path),
            "diagnostic_only": True,
            "runtime_authority": False,
        }
        evidence_path = evidence_root / "evidence.json"
        atomic_write_json(evidence_path, evidence)
        components.append({
            "handoff_item_id": item_id,
            "branch": branch,
            "run_root": str(formal.resolve()),
            "generation_id": branch_binding["generation_id"],
            "authorization_mode": "completed_manual_forward",
            "component_class": "owner",
            "risk_classification": "blend_induced_local_ghost",
            "pair_index": pair_index,
            "left_source_index": int(pair_row["left_source_index"]),
            "right_source_index": int(pair_row["right_source_index"]),
            "left_frame_id": int(pair_row["left_frame_id"]),
            "right_frame_id": int(pair_row["right_frame_id"]),
            "bbox_xyxy": [x0, y0, x1, y1],
            "affected_mask": mask_path.relative_to(acceptance_root).as_posix(),
            "affected_mask_sha256": sha256_file(mask_path),
            "p2_crop": p2_crop.relative_to(acceptance_root).as_posix(),
            "p2_crop_sha256": sha256_file(p2_crop),
            "p3_crop": p3_crop.relative_to(acceptance_root).as_posix(),
            "p3_crop_sha256": sha256_file(p3_crop),
            "component_evidence": evidence_path.relative_to(acceptance_root).as_posix(),
            "component_evidence_sha256": sha256_file(evidence_path),
            "p2_pair_transaction": str(parent_transaction.resolve()),
            "p2_pair_transaction_sha256": sha256_file(parent_transaction),
            "p3_blend_transaction": str(blend_transaction.resolve()),
            "p3_blend_transaction_sha256": sha256_file(blend_transaction),
            "p2_completion_sha256": branch_binding["p2_completion_sha256"],
            "p3_completion_sha256": branch_binding["p3_completion_sha256"],
            "p3_result_sha256": branch_binding["p3_result_sha256"],
            "p3_provenance_sha256": branch_binding["p3_provenance_sha256"],
            "p3_hard_audit_sha256": branch_binding["p3_hard_audit_sha256"],
            "automatic_quality_decision_sha256": branch_binding["automatic_quality_decision_sha256"],
            "core_candidates": [R0_KEEP_P3, R1_B1_TO_B0],
            "optional_gates": {
                "constrained_two_label_graphcut": False,
                "depth_risk_veto": False,
                "bounded_local_mesh": False,
            },
            "roi_expansion_allowed": False,
        })
    return components, out_of_scope


def build_m7_handoff(acceptance_root: Path) -> dict[str, Any]:
    acceptance_root = acceptance_root.resolve()
    authorization_path = acceptance_root / "M6_manual_forward_authorization.json"
    authorization = _json(authorization_path)
    if (
        authorization.get("schema") != AUTHORIZATION_SCHEMA
        or authorization.get("m6_stage_exit") != "completed_manual_forward"
        or authorization.get("manual_acceptance_granted") is not True
        or authorization.get("m7_authorization_state") != "authorized_manual_forward"
    ):
        raise ValueError("S013 M7 manual-forward authorization is invalid")
    branch_bindings = authorization.get("accepted_p3_branches")
    if not isinstance(branch_bindings, list) or {row.get("branch") for row in branch_bindings} != set(BRANCHES):
        raise ValueError("S013 M7 authorization does not bind all accepted P3 branches")
    components: list[dict[str, Any]] = []
    discovered_out_of_scope: list[dict[str, Any]] = []
    for branch_binding in branch_bindings:
        branch_components, out_of_scope = _branch_components(acceptance_root, branch_binding)
        components.extend(branch_components)
        discovered_out_of_scope.extend(out_of_scope)
    known_out_of_scope = [
        {
            "class": "systemic_photometric",
            "reason": "wide-safe P95 benefit is below one percent on all four branches",
            "source": "blocked_report.json",
        },
        {
            "class": "brightness_chroma_white_balance",
            "reason": "systemic brightness/chroma/white-balance residuals remain M6/M6.x scope",
            "source": "blocked_report.json",
        },
        {
            "class": "metric_calibration",
            "reason": "frozen proxy threshold and post-render tail definitions disagree",
            "source": "blocked_report.json",
        },
        {
            "class": "Q4_low_frequency_field",
            "reason": "Q4 and nonzero low-frequency fields are disabled and not M7 components",
            "source": "manual_forward_authorization",
        },
        *discovered_out_of_scope,
    ]
    payload: dict[str, Any] = {
        "schema": HANDOFF_SCHEMA,
        "candidate_id": FORMAL_CANDIDATE_ID,
        "implementation_id": FORMAL_IMPLEMENTATION_ID,
        "authorization_mode": "completed_manual_forward",
        "manual_forward_authorization": authorization_path.name,
        "manual_forward_authorization_sha256": sha256_file(authorization_path),
        "automatic_blocked_report": "blocked_report.json",
        "automatic_blocked_report_sha256": authorization["automatic_blocked_report_sha256"],
        "automatic_quality_records_preserved": True,
        "accepted_parent_stage": "P3",
        "branch_bindings": branch_bindings,
        "components": sorted(components, key=lambda value: (str(value["branch"]), int(value["pair_index"]))),
        "known_out_of_scope": known_out_of_scope,
        "core_registry": list(CORE_CANDIDATES),
        "q0_q3_frozen": True,
        "q4_and_low_frequency_field_disabled": True,
        "thresholds_frozen": True,
        "source_set_frozen": True,
        "b1_total_width_px": 2,
        "b0_owner_only": True,
        "optional_gates": {
            "constrained_two_label_graphcut": False,
            "depth_risk_veto": False,
            "bounded_local_mesh": False,
        },
        "forbidden_operations": [
            "Q0_Q4_resolve", "threshold_change", "source_set_change", "inserted_frame",
            "expanded_feather", "MultiBand", "cross_pair_joint_repair", "roi_expansion",
        ],
        "diagnostic_only": True,
        "runtime_authority": False,
        "production_eligible": False,
        "production_lock_eligible": False,
    }
    atomic_write_json(acceptance_root / "M7_handoff.json", payload)
    return payload


def verify_m7_handoff(acceptance_root: Path) -> dict[str, Any]:
    acceptance_root = acceptance_root.resolve()
    failures: list[str] = []
    try:
        validate_core_registry(CORE_CANDIDATES)
        authorization_path = acceptance_root / "M6_manual_forward_authorization.json"
        handoff_path = acceptance_root / "M7_handoff.json"
        blocked_path = acceptance_root / "blocked_report.json"
        authorization = _json(authorization_path)
        handoff = _json(handoff_path)
        if authorization.get("schema") != AUTHORIZATION_SCHEMA:
            failures.append("authorization_schema_invalid")
        if (
            authorization.get("m6_stage_exit") != "completed_manual_forward"
            or authorization.get("manual_acceptance_granted") is not True
            or authorization.get("automatic_quality_records_preserved") is not True
        ):
            failures.append("manual_forward_authorization_invalid")
        if sha256_file(blocked_path) != authorization.get("automatic_blocked_report_sha256"):
            failures.append("automatic_blocked_report_sha256_mismatch")
        blocked = _json(blocked_path)
        if blocked.get("stage_exit") != "blocked" or blocked.get("m7_handoff_eligible") is not False:
            failures.append("automatic_blocked_state_was_overwritten")
        if handoff.get("schema") != HANDOFF_SCHEMA:
            failures.append("handoff_schema_invalid")
        if handoff.get("manual_forward_authorization_sha256") != sha256_file(authorization_path):
            failures.append("handoff_authorization_sha256_mismatch")
        if handoff.get("core_registry") != list(CORE_CANDIDATES):
            failures.append("core_registry_invalid")
        if any(handoff.get(key) is not True for key in (
            "q0_q3_frozen", "q4_and_low_frequency_field_disabled", "thresholds_frozen", "source_set_frozen",
        )):
            failures.append("frozen_contract_invalid")
        optional = handoff.get("optional_gates")
        if not isinstance(optional, dict) or any(value is not False for value in optional.values()):
            failures.append("optional_gate_enabled")
        live_bindings = {branch: _verify_branch(acceptance_root, branch) for branch in BRANCHES}
        authorization_bindings = {
            str(row.get("branch")): row for row in authorization.get("accepted_p3_branches", [])
            if isinstance(row, dict)
        }
        for branch, live in live_bindings.items():
            sealed = authorization_bindings.get(branch)
            if sealed is None or any(sealed.get(key) != live.get(key) for key in (
                "generation_id", "p2_completion_sha256", "p3_completion_sha256", "p3_result_sha256",
                "p3_provenance_sha256", "p3_hard_audit_sha256", "effective_config_sha256",
                "photometric_solution_sha256", "source_set_sha256", "source_count",
            )):
                failures.append(f"accepted_p3_binding_changed:{branch}")
        components = handoff.get("components")
        if not isinstance(components, list):
            failures.append("components_invalid")
            components = []
        pair_indices: dict[str, list[int]] = {branch: [] for branch in BRANCHES}
        for component in components:
            if not isinstance(component, dict):
                failures.append("component_not_object")
                continue
            branch = str(component.get("branch"))
            pair_index = component.get("pair_index")
            if branch not in pair_indices or not isinstance(pair_index, int):
                failures.append("component_branch_or_pair_invalid")
                continue
            pair_indices[branch].append(pair_index)
            if component.get("component_class") not in ALLOWED_COMPONENT_CLASSES:
                failures.append(f"component_class_forbidden:{branch}:{pair_index}")
            flattened = json.dumps(component, sort_keys=True).lower()
            if any(term in flattened for term in DENIED_COMPONENT_TERMS):
                failures.append(f"component_contains_out_of_scope_problem:{branch}:{pair_index}")
            if component.get("optional_gates") != {
                "constrained_two_label_graphcut": False,
                "depth_risk_veto": False,
                "bounded_local_mesh": False,
            }:
                failures.append(f"component_optional_gate_invalid:{branch}:{pair_index}")
            for path_key, sha_key in (
                ("affected_mask", "affected_mask_sha256"),
                ("p2_crop", "p2_crop_sha256"),
                ("p3_crop", "p3_crop_sha256"),
                ("component_evidence", "component_evidence_sha256"),
            ):
                path = _safe_relative(acceptance_root, str(component.get(path_key, "")), label=path_key)
                if not path.is_file() or sha256_file(path) != component.get(sha_key):
                    failures.append(f"component_evidence_sha256_mismatch:{branch}:{pair_index}:{path_key}")
            for path_key, sha_key in (
                ("p2_pair_transaction", "p2_pair_transaction_sha256"),
                ("p3_blend_transaction", "p3_blend_transaction_sha256"),
            ):
                path = Path(str(component.get(path_key, ""))).resolve()
                if not path.is_file() or sha256_file(path) != component.get(sha_key):
                    failures.append(f"component_transaction_sha256_mismatch:{branch}:{pair_index}")
        for branch, pairs in pair_indices.items():
            if len(pairs) != len(set(pairs)) or any(abs(left - right) == 1 for left in pairs for right in pairs if left < right):
                failures.append(f"adjacent_or_duplicate_component_pair:{branch}")
            pair_count = max(0, int(live_bindings[branch]["source_count"]) - 1)
            if len(pairs) > max(3, math.ceil(0.05 * pair_count)):
                failures.append(f"component_count_exceeds_cap:{branch}")
            if len(pairs) > 2:
                failures.append(f"safe_visible_seam_modification_cap_exceeded:{branch}")
    except (ValueError, KeyError, TypeError, OSError) as exc:
        failures.append(f"malformed_or_missing_evidence:{type(exc).__name__}:{exc}")
    failures = list(dict.fromkeys(failures))
    report = {
        "schema": PREFLIGHT_SCHEMA,
        "authorized": not failures,
        "failures": failures,
        "authorization_mode": "completed_manual_forward" if not failures else None,
        "original_automatic_status": "blocked",
        "automatic_quality_records_preserved": not any(
            failure == "automatic_blocked_state_was_overwritten" for failure in failures
        ),
        "core_registry": list(CORE_CANDIDATES),
        "optional_gates_all_disabled": True,
        "p4_created": False,
        "pointer_updated": False,
        "diagnostic_only": True,
        "runtime_authority": False,
    }
    atomic_write_json(acceptance_root / "M7_preflight.json", report)
    atomic_write_json(acceptance_root / "M7_handoff_validation.json", report)
    return report


def _selection_for_component(
    acceptance_root: Path,
    component: Mapping[str, Any],
) -> tuple[dict[str, Any], Any | None]:
    formal = Path(str(component["run_root"]))
    p3 = formal / "P3"
    p2_ref = _json(p3 / "p2_parent_reference.json")
    p2 = Path(str(p2_ref["p2_root"]))
    p2_completion = _json(p2 / "P2_completion.json")
    p2_image = _load_color(p2 / str(p2_completion.get("result_asset", "geometry_and_seam_panorama_owner_only.png")))
    p3_image = _load_color(p3 / "visual_panorama.png")
    owner_only = _load_color(p3 / "photometric_owner_only.png")
    provenance = _npz(p3 / "p3_pixel_provenance.npz")
    mask_archive = _npz(_safe_relative(
        acceptance_root, str(component["affected_mask"]), label="affected_mask",
    ))
    affected = np.asarray(mask_archive["affected_mask"], bool)
    before_ghost = ghost_material_count(p2_image, p3_image, affected)
    candidates = [{
        "model": R0_KEEP_P3,
        "status": "baseline_selected_unless_strict_winner",
        "ghost_material_count": before_ghost,
        "pixel_change_count": 0,
    }]
    winner = None
    if R1_B1_TO_B0 in component.get("core_candidates", []):
        r1 = r1_b1_to_b0_owner_only(
            p3_image, owner_only, provenance, affected,
            pair_index=int(component["pair_index"]),
        )
        after_ghost = ghost_material_count(p2_image, r1.image, affected)
        blend = _json(Path(str(component["p3_blend_transaction"])))
        immediate_before = blend.get("immediate_before_linear")
        immediate_after = blend.get("immediate_after_linear")
        effective = _json(p3 / "effective_config.json")
        mde = float(effective.get("selection", {}).get("selection_score_mde_linear", 0.0005))
        color_step_increase = (
            max(0.0, float(immediate_before) - float(immediate_after))
            if immediate_before is not None and immediate_after is not None else math.inf
        )
        outside_equal = bool(np.array_equal(r1.image[~affected], p3_image[~affected]))
        strict_winner = bool(
            before_ghost > after_ghost
            and int(r1.audit["pixel_change_count"]) > 0
            and color_step_increase <= mde
            and outside_equal
        )
        candidates.append({
            "model": R1_B1_TO_B0,
            "status": "winner" if strict_winner else "rejected",
            "ghost_material_count_before": before_ghost,
            "ghost_material_count_after": after_ghost,
            "ghost_material_improved": after_ghost < before_ghost,
            "immediate_color_step_increase_linear": color_step_increase,
            "maximum_allowed_increase_linear": mde,
            "no_new_color_step": color_step_increase <= mde,
            "outside_affected_exact": outside_equal,
            **dict(r1.audit),
        })
        if strict_winner:
            winner = r1
    candidates.extend([
        {"model": R2_SEAM_DOWNGRADE, "status": "not_applicable_component_class"},
        {"model": R3_GEOMETRY_DOWNGRADE, "status": "not_applicable_component_class"},
    ])
    selected = winner.model if winner is not None else R0_KEEP_P3
    return {
        "handoff_item_id": component["handoff_item_id"],
        "branch": component["branch"],
        "pair_index": component["pair_index"],
        "component_class": component["component_class"],
        "candidate_order": list(CORE_CANDIDATES),
        "candidates": candidates,
        "selected_model": selected,
        "actual_winner": winner is not None,
    }, winner


def _publish_p4(
    acceptance_root: Path,
    branch_binding: Mapping[str, Any],
    branch_components: Sequence[Mapping[str, Any]],
    selections: Sequence[tuple[Mapping[str, Any], Any]],
    *,
    authorization_sha256: str,
    handoff_sha256: str,
) -> dict[str, Any]:
    formal = Path(str(branch_binding["run_root"]))
    p3 = formal / "P3"
    final = formal / "P4"
    pending = formal / f".P4.{uuid.uuid4().hex}.pending"
    if final.exists() or any(path.name.startswith(".P4.") and path.name.endswith(".pending") for path in formal.iterdir()):
        raise FileExistsError(f"S013 M7 {branch_binding['branch']} P4 target/pending already exists")
    p3_image = _load_color(p3 / "visual_panorama.png")
    image = p3_image.copy()
    provenance = _npz(p3 / "p3_pixel_provenance.npz")
    affected_union = np.zeros(image.shape[:2], bool)
    selected_models: list[str] = []
    transactions: list[dict[str, Any]] = []
    component_index_by_item = {str(component["handoff_item_id"]): index for index, component in enumerate(branch_components)}
    try:
        pending.mkdir(parents=False, exist_ok=False)
        for transaction_index, (selection, candidate) in enumerate(selections):
            component = next(
                value for value in branch_components
                if value["handoff_item_id"] == selection["handoff_item_id"]
            )
            affected = _npz(_safe_relative(
                acceptance_root, str(component["affected_mask"]), label="affected_mask",
            ))["affected_mask"].astype(bool)
            if np.any(affected_union & affected):
                raise ValueError("S013 M7 winning component masks overlap")
            image[affected] = candidate.image[affected]
            for name, value in candidate.provenance.items():
                if np.asarray(value).shape == affected.shape:
                    provenance[name][affected] = np.asarray(value)[affected]
            affected_union |= affected
            selected_models.append(str(candidate.model))
            transactions.append({
                "transaction_id": transaction_index,
                "handoff_item_id": component["handoff_item_id"],
                "pair_index": component["pair_index"],
                "model": candidate.model,
                "affected_mask_sha256": component["affected_mask_sha256"],
                "parent_p3_completion_sha256": branch_binding["p3_completion_sha256"],
                "audit": dict(candidate.audit),
            })
        for name in (
            "m7_handoff_item_id", "repair_component_id", "repair_transaction_id",
            "repair_model_code", "repair_parent_stage_code",
        ):
            provenance[name] = np.full(image.shape[:2], -1, np.int32)
        for transaction_index, transaction in enumerate(transactions):
            component = next(value for value in branch_components if value["handoff_item_id"] == transaction["handoff_item_id"])
            affected = _npz(_safe_relative(
                acceptance_root, str(component["affected_mask"]), label="affected_mask",
            ))["affected_mask"].astype(bool)
            component_index = component_index_by_item[str(component["handoff_item_id"])]
            provenance["m7_handoff_item_id"][affected] = component_index
            provenance["repair_component_id"][affected] = component_index
            provenance["repair_transaction_id"][affected] = transaction_index
            provenance["repair_model_code"][affected] = 1
            provenance["repair_parent_stage_code"][affected] = 3
        audit = audit_s13_p4_arrays(
            p3_image, image, _npz(p3 / "p3_pixel_provenance.npz"), provenance, affected_union,
            selected_models=selected_models,
            expected_source_count=int(branch_binding["source_count"]),
            authorization_sha256=authorization_sha256,
            handoff_sha256=handoff_sha256,
        )
        if audit.get("passed") is not True:
            raise ValueError(f"S013 M7 P4 hard audit failed: {audit['failures']}")
        write_image(pending / "repaired_panorama.png", image)
        write_image(pending / "repaired_panorama.jpg", image)
        valid_asset = cv2.imread(str(p3 / "p3_valid_mask.png"), cv2.IMREAD_UNCHANGED)
        if valid_asset is None:
            raise ValueError("S013 M7 P3 valid mask is unreadable")
        write_image(pending / "p4_valid_mask.png", valid_asset)
        write_npz(pending / "p4_pixel_provenance.npz", provenance)
        atomic_write_json(pending / "p3_parent_reference.json", {
            "stage": "P3",
            "completion_sha256": branch_binding["p3_completion_sha256"],
            "result_sha256": branch_binding["p3_result_sha256"],
            "provenance_sha256": branch_binding["p3_provenance_sha256"],
            "hard_audit_sha256": branch_binding["p3_hard_audit_sha256"],
        })
        atomic_write_json(pending / "m7_handoff_reference.json", {
            "schema": HANDOFF_SCHEMA,
            "m7_handoff_sha256": handoff_sha256,
            "manual_forward_authorization_sha256": authorization_sha256,
        })
        atomic_write_json(pending / "repair_components.json", {
            "components": list(branch_components), "selected_models": selected_models,
        })
        atomic_write_json(pending / "repair_transactions.json", {"transactions": transactions})
        atomic_write_json(pending / "hard_audit.json", audit)
        atomic_write_json(pending / "diagnostic_quality_report.json", {
            "diagnostic_only": True, "runtime_authority": False,
            "component_selections": [selection for selection, _candidate in selections],
        })
        atomic_write_json(pending / "performance.json", {
            "core_only": True, "optional_invocations": 0,
            "q_solver_invocations": 0, "source_set_changes": 0,
        })
        completion = seal_stage(
            pending,
            completion_name="P4_completion.json",
            schema=P4_COMPLETION_SCHEMA,
            metadata={
                "stage": "P4",
                "generation_id": branch_binding["generation_id"],
                "hard_audit_passed": True,
                "parent_stage": "P3",
                "parent_completion_sha256": branch_binding["p3_completion_sha256"],
                "result_asset": "repaired_panorama.png",
                "result_asset_sha256": sha256_file(pending / "repaired_panorama.png"),
                "pixel_provenance_sha256": sha256_file(pending / "p4_pixel_provenance.npz"),
                "hard_audit_sha256": sha256_file(pending / "hard_audit.json"),
                "manual_forward_authorization_sha256": authorization_sha256,
                "m7_handoff_sha256": handoff_sha256,
                "actual_winner_count": len(selections),
                "diagnostic_only": True,
                "production_eligible": False,
                "production_lock_eligible": False,
            },
        )
        verify_stage(pending, completion_name="P4_completion.json", schema=P4_COMPLETION_SCHEMA)
        os.replace(pending, final)
        verify_stage(final, completion_name="P4_completion.json", schema=P4_COMPLETION_SCHEMA)
        pointer_path = formal / "current_latest.json"
        current = _json(pointer_path)
        if (
            current.get("stage") != "P3"
            or current.get("completion_sha256") != branch_binding["p3_completion_sha256"]
        ):
            raise ValueError("S013 M7 current_latest changed before P4 promotion")
        atomic_write_json(pointer_path, {
            "schema": "gemini305-video-s13-current-stage/v2",
            "stage": "P4",
            "generation_id": branch_binding["generation_id"],
            "completion_sha256": sha256_file(final / "P4_completion.json"),
        })
        return {
            "state": "P4_sealed",
            "p4": str(final),
            "completion_sha256": sha256_file(final / "P4_completion.json"),
            "pointer_stage": "P4",
            "actual_winner_count": len(selections),
            "completion": completion,
        }
    except Exception:
        if pending.is_dir():
            shutil.rmtree(pending)
        raise


def run_manual_forward_m7(
    acceptance_root: Path,
    *,
    user_decision_text: str,
    authorized_by: str = "workspace_user",
    authorized_at_utc: str | None = None,
) -> dict[str, Any]:
    acceptance_root = acceptance_root.resolve()
    build_manual_forward_authorization(
        acceptance_root,
        user_decision_text=user_decision_text,
        authorized_by=authorized_by,
        authorized_at_utc=authorized_at_utc,
    )
    build_m7_handoff(acceptance_root)
    preflight = verify_m7_handoff(acceptance_root)
    if preflight.get("authorized") is not True:
        report = {
            "schema": SUMMARY_SCHEMA,
            "state": "not_authorized",
            "failures": preflight["failures"],
            "p4_created": False,
            "continue_to_m8_with_stage": None,
        }
        atomic_write_json(acceptance_root / "M7_not_authorized.json", report)
        atomic_write_json(acceptance_root / "M7_validation_summary.json", report)
        return report
    handoff = _json(acceptance_root / "M7_handoff.json")
    authorization_sha = sha256_file(acceptance_root / "M6_manual_forward_authorization.json")
    handoff_sha = sha256_file(acceptance_root / "M7_handoff.json")
    components = handoff["components"]
    comparisons: list[dict[str, Any]] = []
    winners_by_branch: dict[str, list[tuple[Mapping[str, Any], Any]]] = {branch: [] for branch in BRANCHES}
    for component in components:
        comparison, winner = _selection_for_component(acceptance_root, component)
        comparisons.append(comparison)
        if winner is not None:
            winners_by_branch[str(component["branch"])].append((comparison, winner))
    comparison_report = {
        "schema": "gemini305-video-s13-m7-core-comparison/v1",
        "candidate_order": list(CORE_CANDIDATES),
        "components": comparisons,
        "q0_q3_frozen": True,
        "thresholds_frozen": True,
        "source_set_frozen": True,
        "optional_invocations": 0,
    }
    atomic_write_json(acceptance_root / "M7_core_comparisons.json", comparison_report)
    branch_bindings = {row["branch"]: row for row in handoff["branch_bindings"]}
    branch_results: list[dict[str, Any]] = []
    for branch in BRANCHES:
        winners = winners_by_branch[branch]
        branch_components = [value for value in components if value["branch"] == branch]
        if winners:
            branch_results.append({
                "branch": branch,
                **_publish_p4(
                    acceptance_root,
                    branch_bindings[branch],
                    branch_components,
                    winners,
                    authorization_sha256=authorization_sha,
                    handoff_sha256=handoff_sha,
                ),
            })
        else:
            branch_results.append({
                "branch": branch,
                "state": "no_winning_repair",
                "component_count": len(branch_components),
                "pointer_stage": "P3",
                "p4_created": False,
            })
    any_p4 = any(value["state"] == "P4_sealed" for value in branch_results)
    summary = {
        "schema": SUMMARY_SCHEMA,
        "state": "completed_with_p4" if any_p4 else "completed_no_winner_keep_p3",
        "authorization_mode": "completed_manual_forward",
        "manual_forward_authorization_sha256": authorization_sha,
        "m7_handoff_sha256": handoff_sha,
        "automatic_visual_status": "blocked",
        "automatic_quality_records_preserved": True,
        "component_count": len(components),
        "actual_winner_count": sum(len(value) for value in winners_by_branch.values()),
        "p4_created": any_p4,
        "branches": branch_results,
        "continue_to_m8": True,
        "continue_to_m8_with_stage": "P4_where_present_else_P3" if any_p4 else "P3",
        "current_reviewed_written": False,
        "optional_invocations": 0,
        "production_lock_written": False,
        "diagnostic_only": True,
        "runtime_authority": False,
    }
    atomic_write_json(acceptance_root / "M7_known_out_of_scope.json", {
        "schema": "gemini305-video-s13-m7-known-out-of-scope/v1",
        "known_out_of_scope": handoff["known_out_of_scope"],
        "diagnostic_only": True,
        "runtime_authority": False,
    })
    if not any_p4:
        atomic_write_json(acceptance_root / "M7_no_winning_repair.json", summary)
    atomic_write_json(acceptance_root / "M7_validation_summary.json", summary)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run authorized manual-forward S013 M7 Core")
    parser.add_argument("--acceptance-root", type=Path, required=True)
    parser.add_argument("--user-decision", required=True)
    parser.add_argument("--authorized-by", default="workspace_user")
    parser.add_argument("--authorized-at-utc")
    return parser


def main() -> None:
    args = _parser().parse_args()
    report = run_manual_forward_m7(
        args.acceptance_root,
        user_decision_text=args.user_decision,
        authorized_by=args.authorized_by,
        authorized_at_utc=args.authorized_at_utc,
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()


__all__ = [
    "AUTHORIZATION_SCHEMA", "HANDOFF_SCHEMA", "build_m7_handoff",
    "build_manual_forward_authorization", "run_manual_forward_m7", "verify_m7_handoff",
]
