"""Fail-closed live/offline equivalence audit for S013 Visual Continuity V11."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

from .video_delivery import VIDEO_DELIVERY_SCHEMA, VIDEO_REPORT_SCHEMA
from .video_s13_contract import (
    S13_VISUAL_CONTINUITY_ALGORITHM_ID,
    S13_VISUAL_CONTINUITY_IMPLEMENTATION_ID,
)


_STAGE_FILES = (
    "P0_base_panorama.png",
    "P1_vertical_panorama.png",
    "P2_geometry_and_seam_panorama.png",
    "P3_visual_panorama.png",
)
_TIMING_FIELDS = (
    "capture_stop_to_p3_memory_seconds",
    "capture_stop_to_p3_published_seconds",
)
_Q1R_MODEL = "Q1R_robust_centered_scalar_gain"
_Q4C_MODEL = "Q4c_quality_cut_component_scalar_gain"


class S13V11EquivalenceError(RuntimeError):
    def __init__(self, report: Mapping[str, object]) -> None:
        self.diagnostics = dict(report)
        stage = report.get("first_divergent_stage", "unknown")
        super().__init__(f"S013 V11 live/offline equivalence failed at {stage}")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"S013 V11 acceptance artifact is missing: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"S013 V11 acceptance artifact must be an object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _load_production(root: Path) -> dict[str, object]:
    destination = root.expanduser().resolve()
    delivery = _read_json(destination / "video_delivery.json")
    report = _read_json(destination / "video_report.json")
    if delivery.get("schema") != VIDEO_DELIVERY_SCHEMA:
        raise ValueError("S013 V11 acceptance requires a current video delivery")
    if report.get("schema") != VIDEO_REPORT_SCHEMA:
        raise ValueError("S013 V11 acceptance requires a current video report")
    algorithm = report.get("algorithm")
    if not isinstance(algorithm, Mapping) or any((
        algorithm.get("role") != "production",
        algorithm.get("algorithm_id") != S13_VISUAL_CONTINUITY_ALGORITHM_ID,
        algorithm.get("implementation_id") != S13_VISUAL_CONTINUITY_IMPLEMENTATION_ID,
        algorithm.get("fallback_used") is not False,
    )):
        raise ValueError("S013 V11 acceptance received a non-V11 or fallback delivery")
    panorama_path = destination / "video_panorama.png"
    panorama = cv2.imread(str(panorama_path), cv2.IMREAD_UNCHANGED)
    if panorama is None or panorama.dtype != np.uint8 or panorama.ndim != 3:
        raise ValueError("S013 V11 acceptance cannot decode the formal PNG")
    provenance_path = destination / "video_pixel_provenance.npz"
    try:
        with np.load(provenance_path, allow_pickle=False) as archive:
            owner = np.asarray(archive["owner_frame_id"])
    except (FileNotFoundError, KeyError, ValueError) as exc:
        raise ValueError("S013 V11 acceptance cannot load formal owner provenance") from exc
    if owner.shape != panorama.shape[:2] or not np.issubdtype(owner.dtype, np.integer):
        raise ValueError("S013 V11 acceptance owner provenance is invalid")
    timing = _read_json(destination / "video_timing.json")
    final_2d = timing.get("final_2d")
    if not isinstance(final_2d, Mapping):
        raise ValueError("S013 V11 acceptance timing lacks final_2d")
    timing_values: dict[str, float] = {}
    for field in _TIMING_FIELDS:
        value = final_2d.get(field)
        if not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0:
            raise ValueError(f"S013 V11 acceptance timing lacks finite final_2d.{field}")
        timing_values[field] = float(value)
    if timing_values[_TIMING_FIELDS[1]] < timing_values[_TIMING_FIELDS[0]]:
        raise ValueError("S013 V11 published P3 timing precedes in-memory P3")
    return {
        "root": destination,
        "delivery": delivery,
        "report": report,
        "panorama_path": panorama_path,
        "panorama": panorama,
        "owner": owner,
        "timing": timing_values,
    }


def _m6_contract(report: Mapping[str, object]) -> dict[str, object]:
    decisions = report.get("m6_decisions")
    if not isinstance(decisions, Mapping):
        raise ValueError("S013 V11 report lacks M6 decision evidence")
    required = (
        "selected_photometric_model",
        "blend_models",
        "b0_count",
        "b1_count",
        "m63_audit",
        "photometric_candidate_audits",
        "c2e_decisions",
        "c2e_owner_only_pair_indices",
    )
    if any(key not in decisions for key in required):
        raise ValueError("S013 V11 report has incomplete M63/B0/B1/C2E evidence")
    audits = decisions["photometric_candidate_audits"]
    if not isinstance(audits, (tuple, list)):
        raise ValueError("S013 V11 report lacks photometric candidate decisions")
    models = {
        item.get("model") for item in audits if isinstance(item, Mapping)
    }
    if _Q4C_MODEL not in models:
        raise ValueError("S013 V11 report lacks the Q4c candidate decision")
    canonical = {str(key): decisions[key] for key in decisions}
    if _Q1R_MODEL not in models:
        m63 = decisions["m63_audit"]
        quality_cuts = (
            m63.get("quality_cut_pair_indices") if isinstance(m63, Mapping) else None
        )
        selected_model = m63.get("selected_model") if isinstance(m63, Mapping) else None
        q4c_audit = next(
            (
                item
                for item in audits
                if isinstance(item, Mapping) and item.get("model") == _Q4C_MODEL
            ),
            None,
        )
        if (
            not isinstance(m63, Mapping)
            or selected_model not in ("Q0_identity", _Q4C_MODEL)
            or not isinstance(quality_cuts, (tuple, list))
            or not quality_cuts
            or not isinstance(q4c_audit, Mapping)
            or q4c_audit.get("selected") is (selected_model != _Q4C_MODEL)
        ):
            raise ValueError("S013 V11 report cannot explain the absent Q1R decision")
        canonical["q1r_quality_cut_outcome"] = {
            "status": "not_evaluated_under_quality_cut_split",
            "quality_cut_pair_indices": list(quality_cuts),
            "selected_model": selected_model,
            "q4c_selected": bool(q4c_audit["selected"]),
        }
    return canonical


def _pair_decision_contract(pairs: object) -> list[dict[str, object]]:
    if not isinstance(pairs, (tuple, list)):
        raise ValueError("S013 V11 report lacks pair seam decisions")
    canonical: list[dict[str, object]] = []
    for pair in pairs:
        if not isinstance(pair, Mapping):
            raise ValueError("S013 V11 pair seam decision is malformed")
        transaction = pair.get("m5_transaction")
        if not isinstance(transaction, Mapping):
            raise ValueError("S013 V11 pair seam transaction is missing")
        evaluations = transaction.get("candidate_evaluations")
        if not isinstance(evaluations, list):
            raise ValueError("S013 V11 pair seam evaluations are missing")
        canonical.append({
            "pair_index": pair.get("pair_index"),
            "left_frame_id": pair.get("left_frame_id"),
            "right_frame_id": pair.get("right_frame_id"),
            "corridor_x0": pair.get("corridor_x0"),
            "corridor_x1": pair.get("corridor_x1"),
            "seam_x_by_row": pair.get("seam_x_by_row"),
            "m5_selected_seam_model": pair.get("m5_selected_seam_model"),
            "c2e_decision": pair.get("c2e_decision"),
            "owner_only": pair.get("owner_only"),
            "candidate_outcomes": [
                {
                    "candidate_id": item.get("candidate_id"),
                    "model_code": item.get("model_code"),
                    "model_name": item.get("model_name"),
                    "generation_status": item.get("generation_status"),
                    "evaluation_status": item.get("evaluation_status"),
                    "geometry_model": item.get("geometry_model"),
                    "hard_gate_passed": item.get("hard_gate_passed"),
                    "hard_gate_failures": item.get("hard_gate_failures"),
                }
                for item in evaluations
                if isinstance(item, Mapping)
            ],
        })
    return canonical


def _first_divergent_stage(checks: Mapping[str, bool]) -> str | None:
    order = (
        ("same_input", "input"),
        ("source_frame_ids", "P0_source_selection"),
        ("schedule_canvas", "P0_schedule_canvas"),
        ("stage_P0", "P0_pixels"),
        ("stage_P1", "P1_pixels"),
        ("pair_seams", "P2_seam_decisions"),
        ("owner_map_exact", "P2_owner_or_C2E"),
        ("stage_P2", "P2_pixels"),
        ("m6_decisions", "P3_M63_B0_B1_C2E"),
        ("stage_P3", "P3_memory_pixels"),
        ("p3_decoded_exact", "P3_pixels"),
        ("p3_png_sha_exact", "final_png_encode"),
        ("production_output_policy", "production_output_policy"),
    )
    for name, stage in order:
        if checks.get(name) is False:
            return stage
    return None


def compare_s13_v11_live_offline(
    offline_root: Path,
    live_root: Path,
) -> dict[str, object]:
    """Compare two formal productions without accepting partial evidence."""

    offline = _load_production(offline_root)
    live = _load_production(live_root)
    offline_report = offline["report"]
    live_report = live["report"]
    assert isinstance(offline_report, Mapping) and isinstance(live_report, Mapping)
    offline_image = np.asarray(offline["panorama"])
    live_image = np.asarray(live["panorama"])
    same_shape = offline_image.shape == live_image.shape
    if same_shape:
        differing_pixels = int(np.count_nonzero(np.any(offline_image != live_image, axis=2)))
        maximum_channel_difference = int(
            np.max(np.abs(offline_image.astype(np.int16) - live_image.astype(np.int16)))
        )
    else:
        differing_pixels = -1
        maximum_channel_difference = -1

    stage_offline = offline_report.get("stage_pixel_sha256")
    stage_live = live_report.get("stage_pixel_sha256")
    stage_checks = {
        f"stage_{stage}": (
            isinstance(stage_offline, Mapping)
            and isinstance(stage_live, Mapping)
            and stage_offline.get(stage) == stage_live.get(stage)
            and _valid_sha256(stage_offline.get(stage))
        )
        for stage in ("P0", "P1", "P2", "P3")
    }
    forbidden_absent = all(
        not (Path(item["root"]) / name).exists()
        for item in (offline, live)
        for name in _STAGE_FILES
    )
    output_policies = (
        offline_report.get("production_output_policy"),
        live_report.get("production_output_policy"),
    )
    policy_valid = forbidden_absent and all(
        isinstance(policy, Mapping)
        and policy.get("stage_output_stages") == []
        and policy.get("p3_stage_png_encoded") is False
        and policy.get("formal_publication_encodes_p3_once") is True
        for policy in output_policies
    )
    offline_input = offline_report.get("input_sha256")
    live_input = live_report.get("input_sha256")
    offline_sources = offline_report.get("source_frame_ids")
    live_sources = live_report.get("source_frame_ids")
    offline_schedule = offline_report.get("schedule")
    live_schedule = live_report.get("schedule")
    offline_seams = offline_report.get("pair_seam_decisions")
    live_seams = live_report.get("pair_seam_decisions")
    checks = {
        "same_input": (
            isinstance(offline_input, Mapping)
            and isinstance(live_input, Mapping)
            and set(offline_input) == {"manifest", "calibration", "frames_csv"}
            and all(_valid_sha256(value) for value in offline_input.values())
            and offline_input == live_input
        ),
        "source_frame_ids": (
            isinstance(offline_sources, (tuple, list))
            and isinstance(live_sources, (tuple, list))
            and len(offline_sources) >= 2
            and offline_sources == live_sources
        ),
        "schedule_canvas": (
            isinstance(offline_schedule, Mapping)
            and isinstance(live_schedule, Mapping)
            and "canvas_shape" in offline_schedule
            and "boundaries" in offline_schedule
            and "assignments" in offline_schedule
            and offline_schedule == live_schedule
        ),
        **stage_checks,
        "pair_seams": (
            isinstance(offline_seams, (tuple, list))
            and isinstance(live_seams, (tuple, list))
            and isinstance(offline_sources, (tuple, list))
            and len(offline_seams) == len(offline_sources) - 1
            and _pair_decision_contract(offline_seams)
            == _pair_decision_contract(live_seams)
        ),
        "owner_map_exact": np.array_equal(offline["owner"], live["owner"]),
        "m6_decisions": _m6_contract(offline_report) == _m6_contract(live_report),
        "p3_decoded_exact": same_shape and differing_pixels == 0,
        "p3_png_sha_exact": _sha256(Path(offline["panorama_path"]))
        == _sha256(Path(live["panorama_path"])),
        "production_output_policy": policy_valid,
    }
    first = _first_divergent_stage(checks)
    return {
        "schema": "gemini305-video-s13-v11-live-offline-equivalence/v1",
        "equivalent": first is None,
        "first_divergent_stage": first,
        "checks": checks,
        "p3": {
            "offline_sha256": _sha256(Path(offline["panorama_path"])),
            "live_sha256": _sha256(Path(live["panorama_path"])),
            "differing_pixel_count": differing_pixels,
            "maximum_channel_difference": maximum_channel_difference,
        },
        "owner": {
            "shape_equal": np.asarray(offline["owner"]).shape
            == np.asarray(live["owner"]).shape,
            "differing_pixel_count": (
                int(np.count_nonzero(np.asarray(offline["owner"]) != live["owner"]))
                if np.asarray(offline["owner"]).shape == np.asarray(live["owner"]).shape
                else -1
            ),
        },
        "timing": {
            "offline": offline["timing"],
            "live": live["timing"],
        },
    }


def require_s13_v11_live_offline_equivalence(
    offline_root: Path,
    live_root: Path,
) -> dict[str, object]:
    report = compare_s13_v11_live_offline(offline_root, live_root)
    if report["equivalent"] is not True:
        raise S13V11EquivalenceError(report)
    return report


__all__ = [
    "S13V11EquivalenceError",
    "compare_s13_v11_live_offline",
    "require_s13_v11_live_offline_equivalence",
]
