"""Read-only four-branch acceptance and reproducibility summarization."""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


ACCEPTANCE_SCHEMA = "gemini305-video-s13-m61-acceptance-summary/v1"
REPRODUCIBILITY_SCHEMA = "gemini305-video-s13-m61-reproducibility/v1"
ATLAS_INDEX_SCHEMA = "gemini305-video-s13-m61-new-old-seam-atlas-index/v1"
EXPECTED_BRANCHES = ("fast_direct", "fast_ignore_pose", "slow_direct", "slow_ignore_pose")
REPRO_ASSETS = (
    "visual_panorama.png", "p3_pixel_provenance.npz", "photometric_solution.json",
    "blend_transactions.json",
)


@dataclass(frozen=True)
class BranchInputs:
    branch: str
    formal: Path
    rerun: Path
    old: Path
    p2: Path


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
        raise ValueError(f"missing or invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON asset must be an object: {path}")
    return value


def _quality(p3: Path) -> tuple[Path, dict[str, Any]]:
    candidates = (
        p3 / "quality_report.json", p3 / "diagnostic_quality_report.json",
        p3.parent / "validation_m61/quality_report.json",
        p3.parent.parent / "validation_m61/quality_report.json",
    )
    for path in candidates:
        if path.is_file():
            return path, _json(path)
    raise ValueError(f"quality report not found for P3: {p3}")


def _verify_p3(path: Path) -> dict[str, Any]:
    path = path.resolve()
    required = (*REPRO_ASSETS, "P3_completion.json", "performance.json", "hard_audit.json")
    for name in required:
        if not (path / name).is_file():
            raise ValueError(f"P3 required asset missing: {path / name}")
    completion = _json(path / "P3_completion.json")
    audit = _json(path / "hard_audit.json")
    performance = _json(path / "performance.json")
    quality_path, quality = _quality(path)
    if completion.get("schema") != "gemini305-video-s13-p3-visual-completion/v2":
        raise ValueError(f"unsupported P3 completion schema: {path}")
    if audit.get("schema") != "gemini305-video-s13-p3-hard-audit/v2" or audit.get("passed") is not True:
        raise ValueError(f"P3 hard audit did not pass: {path}")
    if completion.get("hard_audit_sha256") != _sha(path / "hard_audit.json"):
        raise ValueError(f"P3 completion hard-audit binding mismatch: {path}")
    forbidden = performance.get("forbidden_invocations", {})
    if not isinstance(forbidden, Mapping) or any(int(value) != 0 for value in forbidden.values()):
        raise ValueError(f"P3 has forbidden invocation: {path}")
    return {
        "root": str(path), "completion": completion, "completion_sha256": _sha(path / "P3_completion.json"),
        "quality_path": str(quality_path.resolve()), "quality": quality,
        "quality_sha256": _sha(quality_path), "performance": performance,
        "performance_sha256": _sha(path / "performance.json"), "hard_audit": audit,
        "hard_audit_sha256": _sha(path / "hard_audit.json"),
        "asset_sha256": {name: _sha(path / name) for name in REPRO_ASSETS},
    }


def _model_counts(solution: Mapping[str, Any]) -> dict[str, int]:
    models: list[str] = []
    selected = solution.get("selected_model")
    if isinstance(selected, str):
        models.append(selected)
    sources = solution.get("sources", [])
    if isinstance(sources, Sequence) and not isinstance(sources, (str, bytes)):
        for source in sources:
            if isinstance(source, Mapping) and isinstance(source.get("model"), str):
                models.append(str(source["model"]))
    result: dict[str, int] = {}
    for model in models:
        result[model] = result.get(model, 0) + 1
    return result


def _blend_counts(document: Mapping[str, Any], root: Path) -> dict[str, int]:
    result: dict[str, int] = {}
    for entry in document.get("pairs", []):
        if not isinstance(entry, Mapping):
            continue
        asset = entry.get("asset")
        if isinstance(asset, str) and (root / asset).is_file():
            model = _json(root / asset).get("model")
            if isinstance(model, str):
                result[model] = result.get(model, 0) + 1
    return result


def summarize_branch(inputs: BranchInputs) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    formal, rerun = _verify_p3(inputs.formal), _verify_p3(inputs.rerun)
    p2_completion = _json(inputs.p2 / "P2_completion.json")
    p2_image = inputs.p2 / "geometry_and_seam_panorama_owner_only.png"
    if p2_completion.get("schema") != "gemini305-video-s13-p2-completion/v4" or not p2_image.is_file():
        raise ValueError(f"invalid sealed P2-v4 input: {inputs.p2}")
    old_panorama = inputs.old / "visual_panorama.png"
    if not old_panorama.is_file():
        raise ValueError(f"old P3 panorama is missing: {old_panorama}")
    matches = {
        name: formal["asset_sha256"][name] == rerun["asset_sha256"][name]
        for name in REPRO_ASSETS
    }
    reproducible = all(matches.values())
    solution = _json(inputs.formal / "photometric_solution.json")
    blend_doc = _json(inputs.formal / "blend_transactions.json")
    quality = formal["quality"]
    performance = formal["performance"]
    summary = {
        "branch": inputs.branch,
        "p2_canonical_regression": {
            "p2_completion_sha256": _sha(inputs.p2 / "P2_completion.json"),
            "p2_canonical_image_sha256": _sha(p2_image),
            "passed": all(item.get("equal") is True for item in
                          p2_completion.get("canonical_p2_regression", {}).values()),
        },
        "effective_config_sha256": formal["completion"].get("effective_config_sha256"),
        "q_model_counts": _model_counts(solution),
        "q_component_count": len(solution.get("components", []))
            if isinstance(solution.get("components", []), list) else None,
        "blend_model_counts": _blend_counts(blend_doc, inputs.formal),
        "quality_state": quality.get("quality_state", quality.get("state")),
        "effective_manual_review_status": quality.get("effective_manual_review_status"),
        "quality_metrics": {
            name: quality.get(name) for name in (
                "wide_safe_delta_e00_median", "wide_safe_delta_e00_p95", "block_p95",
                "visible_seam_count", "owner_plateau", "slow_ignore_pair_21",
            )
        },
        "protected_blend_count": performance.get("protected_blend_pixel_count", 0),
        "formal_remap": performance.get("formal_raw_rgb_remap_invocations"),
        "audit_remap": performance.get("hard_audit_render_invocations", 0),
        "resources": {name: performance.get(name) for name in (
            "pre_audit_peak_process_rss_bytes", "audit_peak_process_rss_bytes",
            "whole_invocation_peak_process_rss_bytes", "peak_live_array_bytes",
            "maximum_resident_source_rois", "elapsed_seconds", "total_seconds",
            "maximum_post_seconds_soft_sla",
        )},
        "hard_audit_passed": True, "forbidden_invocations": performance.get("forbidden_invocations"),
        "reproducible": reproducible,
    }
    repro = {
        "branch": inputs.branch, "passed": reproducible, "asset_matches": matches,
        "formal": {"completion_sha256": formal["completion_sha256"],
                   "quality_sha256": formal["quality_sha256"], **formal["asset_sha256"]},
        "rerun": {"completion_sha256": rerun["completion_sha256"],
                  "quality_sha256": rerun["quality_sha256"], **rerun["asset_sha256"]},
    }
    atlas = {
        "branch": inputs.branch,
        "new_panorama": str((inputs.formal / "visual_panorama.png").resolve()),
        "rerun_panorama": str((inputs.rerun / "visual_panorama.png").resolve()),
        "old_panorama": str(old_panorama.resolve()),
        "p2_panorama": str(p2_image.resolve()),
        "new_quality_report": formal["quality_path"],
        "old_exists": True,
        "new_seam_diagnostics": sorted(
            str(path.resolve()) for root in (inputs.formal.parent / "validation_m61",
                                             inputs.formal.parent.parent / "validation_m61")
            if root.is_dir() for path in root.rglob("*.png")
        ),
        "old_seam_diagnostics": sorted(
            str(path.resolve()) for root in (inputs.old / "worst_visual_crops",
                                             inputs.old.parent / "validation_m6")
            if root.is_dir() for path in root.rglob("*.png")
        ),
    }
    return summary, repro, atlas


def summarize_acceptance(inputs: Sequence[BranchInputs], output: Path) -> tuple[Path, Path, Path]:
    if len(inputs) != 4 or tuple(sorted(item.branch for item in inputs)) != tuple(sorted(EXPECTED_BRANCHES)):
        raise ValueError("summary requires exactly the four fixed branches")
    output = output.resolve()
    if output.exists():
        raise FileExistsError("summary output must be a new absent directory")
    pending = output.with_name(f".{output.name}.{uuid.uuid4().hex[:12]}.pending")
    pending.mkdir(parents=True)
    try:
        summaries, repros, atlases = [], [], []
        for item in sorted(inputs, key=lambda value: value.branch):
            summary, repro, atlas = summarize_branch(item)
            summaries.append(summary)
            repros.append(repro)
            atlases.append(atlas)
        all_reproducible = all(item["passed"] for item in repros)
        all_hard_audit = all(item["hard_audit_passed"] for item in summaries)
        acceptance = {
            "schema": ACCEPTANCE_SCHEMA,
            "candidate_id": "S013_output_first_progressive_dense_central_slit_m61_v2",
            "readonly_semantic_baseline_commit": "c3aaa7ddbaf8aac5e00cded71ba9e4f1689b822a",
            "branches": summaries,
            "all_hard_audits_passed": all_hard_audit,
            "all_reproducible": all_reproducible,
            "section_27_report_skeleton": {
                "final_commit": None, "candidate_and_schema": None,
                "source_document_sha256": [], "p2_canonical_regression": None,
                "evidence_config_sha256": None, "effective_config_sha256": None,
                "threshold_sha256": None, "mde_sha256": None,
                "q_models_and_components": None, "b0_b1_counts": None,
                "quality_and_manual_state": None, "wide_safe_delta_e": None,
                "block_p95_visible_seam_owner_plateau": None, "slow_ignore_pair_21": None,
                "protected_blend_count": None, "formal_and_audit_remap": None,
                "rss_live_bytes_resident_roi_timing": None, "hard_audit": None,
                "reproducibility": None, "forbidden_invocations": None,
                "tests": None, "ruff": None, "compileall": None, "diff_check": None,
                "stage_exit": None, "m7_handoff_eligible": None,
                "manual_near_target_decision_witness": None, "blocked_reason": None,
            },
        }
        reproducibility = {"schema": REPRODUCIBILITY_SCHEMA, "passed": all_reproducible,
                           "branches": repros}
        atlas_index = {"schema": ATLAS_INDEX_SCHEMA, "coordinate_policy": "full_canvas_seam_index",
                       "branches": atlases}
        for name, payload in (("acceptance_summary.json", acceptance),
                              ("reproducibility.json", reproducibility),
                              ("new_old_seam_atlas_index.json", atlas_index)):
            (pending / name).write_text(
                json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
            )
        output.parent.mkdir(parents=True, exist_ok=True)
        os.replace(pending, output)
        return tuple(output / name for name in (
            "acceptance_summary.json", "reproducibility.json", "new_old_seam_atlas_index.json"
        ))
    finally:
        if pending.exists():
            for child in pending.iterdir():
                child.unlink()
            pending.rmdir()


def parse_branch_spec(value: str) -> BranchInputs:
    if "=" not in value:
        raise ValueError("branch spec must be branch=formal:rerun:old:p2")
    branch, encoded = value.split("=", 1)
    # Split separators before Windows drive roots, leaving the drive colon intact.
    parts = (re.split(r":(?=[A-Za-z]:[\\/])", encoded)
             if re.match(r"^[A-Za-z]:[\\/]", encoded) else encoded.split(":"))
    if len(parts) != 4:
        raise ValueError("branch spec must contain formal:rerun:old:p2")
    return BranchInputs(branch, *(Path(item).resolve() for item in parts))


__all__ = [
    "ACCEPTANCE_SCHEMA", "ATLAS_INDEX_SCHEMA", "BranchInputs", "EXPECTED_BRANCHES",
    "REPRODUCIBILITY_SCHEMA", "parse_branch_spec", "summarize_acceptance", "summarize_branch",
]
