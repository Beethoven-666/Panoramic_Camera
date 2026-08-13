"""Summarize the fixed four-run S1.3 M6 validation bundle."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


RUN_NAMES = ("fast_direct", "fast_ignore_pose", "slow_direct", "slow_ignore_pose")


def _json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def summarize(root: Path) -> dict[str, object]:
    runs: dict[str, object] = {}
    global_photometric: Counter[str] = Counter()
    global_blend: Counter[str] = Counter()
    for run_name in RUN_NAMES:
        run = root / run_name
        pointer = _json(run / "current_latest.json")
        if pointer.get("stage") != "P3":
            raise ValueError(f"{run_name} current_latest is not P3")
        generation = run / str(pointer["generation"])
        p3 = generation / "P3"
        completion = _json(p3 / "P3_completion.json")
        audit = _json(p3 / "hard_audit.json")
        diagnostic = _json(p3 / "diagnostic_quality_report.json")
        solution = _json(p3 / "photometric_solution.json")
        blend = _json(p3 / "blend_transactions.json")
        performance = _json(p3 / "performance.json")
        p2 = generation / "P2"
        p2_image = cv2.imread(
            str(p2 / "geometry_and_seam_panorama_owner_only.png"), cv2.IMREAD_COLOR
        )
        owner_image = cv2.imread(str(p3 / "photometric_owner_only.png"), cv2.IMREAD_COLOR)
        p3_image = cv2.imread(str(p3 / "visual_panorama.png"), cv2.IMREAD_COLOR)
        if p2_image is None or owner_image is None or p3_image is None:
            raise ValueError(f"{run_name} P2/P3 comparison images are unreadable")
        validation = run / "validation_m6"
        validation.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(validation / "p2_vs_p3_full.png"), np.concatenate((p2_image, p3_image), axis=1)):
            raise OSError("Failed to write P2/P3 comparison")
        if not cv2.imwrite(
            str(validation / "p2_vs_photometric_owner_only_full.png"),
            np.concatenate((p2_image, owner_image), axis=1),
        ):
            raise OSError("Failed to write P2/photometric-owner comparison")
        state_snapshot = {
            "current_latest": pointer,
            "current_reviewed": (
                _json(run / "current_reviewed.json")
                if (run / "current_reviewed.json").is_file() else None
            ),
            "current_preview": (
                _json(run / "current_preview.json")
                if (run / "current_preview.json").is_file() else None
            ),
        }
        (validation / "pointer_snapshot.json").write_text(
            json.dumps(state_snapshot, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        source_models = Counter(
            str(item["model"]) for item in solution.get("source_parameters", [])
        )
        blend_models = Counter(str(item["model"]) for item in blend.get("pairs", []))
        global_photometric.update(source_models)
        global_blend.update(blend_models)
        with np.load(p3 / "p3_pixel_provenance.npz", allow_pickle=False) as stored:
            weight = np.asarray(stored["secondary_weight"])
        reviewed = run / "current_reviewed.json"
        preview = run / "current_preview.json"
        runs[run_name] = {
            "generation": str(pointer["generation"]),
            "p3_completion_schema": completion.get("schema"),
            "parent_stage": completion.get("parent_stage"),
            "hard_audit_passed": audit.get("passed"),
            "diagnostic_quality_passed": diagnostic.get("quality_pass"),
            "photometric_model_counts": dict(source_models),
            "blend_model_counts": dict(blend_models),
            "blend_pixel_count": int(np.count_nonzero(weight > 0.0)),
            "protected_blend_pixel_count": audit.get("protected_blend_pixel_count"),
            "maximum_real_contributors_per_pixel": audit.get(
                "maximum_real_contributors_per_pixel"
            ),
            "formal_raw_rgb_remap_invocations": performance.get(
                "formal_raw_rgb_remap_invocations"
            ),
            "full_resolution_render_count": performance.get("full_resolution_render_count"),
            "trajectory_estimation_invocations": performance.get(
                "trajectory_estimation_invocations"
            ),
            "open3d_invocations": performance.get("open3d_invocations"),
            "m4_reestimated_in_m6": performance.get("m4_reestimated_in_m6"),
            "m5_reestimated_in_m6": performance.get("m5_reestimated_in_m6"),
            "geometry_reestimated_in_m6": performance.get("geometry_reestimated_in_m6"),
            "seam_reestimated_in_m6": performance.get("seam_reestimated_in_m6"),
            "total_m6_seconds": performance.get("total_m6"),
            "peak_memory_bytes_estimate": performance.get("peak_memory_bytes_estimate"),
            "current_latest": "P3",
            "current_reviewed": "present_unchanged" if reviewed.is_file() else "not_present",
            "current_preview": "present_unchanged" if preview.is_file() else "not_present",
        }
    return {
        "schema": "gemini305-video-s13-m6-validation/v1",
        "milestone": "M6",
        "pixel_stage": "P3",
        "parent_stage": "P2",
        "runs": runs,
        "photometric_model_counts": dict(global_photometric),
        "blend_model_counts": dict(global_blend),
        "m7_entered": False,
        "production_eligible": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summary = summarize(args.root.resolve())
    encoded = json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
