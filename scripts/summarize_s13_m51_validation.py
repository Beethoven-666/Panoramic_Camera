"""Summarize the fixed S013 M5.1 real-data validation without selecting a review winner."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np


RUN_NAMES = ("fast_direct", "fast_ignore_pose", "slow_direct", "slow_ignore_pose")


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _image_shape(path: Path) -> list[int]:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"unreadable validation image: {path}")
    return [int(image.shape[1]), int(image.shape[0])]


def _transactions(path: Path) -> list[dict[str, Any]]:
    return [
        _read(item) for item in sorted((path / "pair_transactions").glob("pair_*.json"))
    ]


def _run_summary(root: Path, name: str) -> dict[str, Any]:
    run_root = root / name
    latest_run = _read(run_root / "latest_run.json")
    latest = _read(run_root / "current_latest.json")
    generation = run_root / str(latest["generation"])
    p0, p1, p2 = generation / "P0", generation / "P1", generation / "P2"
    p0_completion = _read(p0 / "P0_completion.json")
    p1_completion = _read(p1 / "P1_completion.json")
    p2_completion = _read(p2 / "P2_completion.json")
    hard = _read(p2 / "hard_audit.json")
    diagnostic = _read(p2 / "diagnostic_quality_report.json")
    performance = _read(p2 / "performance.json")
    transactions = _transactions(p2)

    seam_counts = Counter(
        "midpoint_straight" if str(row.get("seam_model", "")).startswith("S0_")
        else str(row.get("seam_model"))
        for row in transactions
    )
    dp_generated = dp_evaluated = dp_selected = 0
    evaluation_status = Counter()
    hard_rejections = Counter()
    fallback_reasons = Counter()
    for row in transactions:
        if row.get("fallback_used"):
            fallback_reasons[str(row.get("fallback_reason") or "unspecified")] += 1
        for status in row.get("candidate_generation", []):
            if status.get("model_name") == "monotone_dp" and status.get("generation_status") == "generated":
                dp_generated += 1
        for evaluation in row.get("candidate_evaluations", []):
            if evaluation.get("model_name") != "monotone_dp":
                continue
            state = str(evaluation.get("evaluation_status"))
            evaluation_status[state] += 1
            if state not in {"generation_failed", "skipped_after_higher_rank_safe_candidate"}:
                dp_evaluated += 1
            if state == "selected":
                dp_selected += 1
            for reason in evaluation.get("hard_gate_failures", []):
                hard_rejections[str(reason)] += 1

    validation = run_root / "validation"
    validation.mkdir(parents=True, exist_ok=True)
    _write(validation / "current_latest_snapshot.json", latest)
    _write(validation / "seam_model_counts.json", dict(seam_counts))
    _write(validation / "dp_candidate_status.json", {
        "generated": dp_generated,
        "evaluated": dp_evaluated,
        "selected": dp_selected,
        "evaluation_status": dict(evaluation_status),
    })
    _write(validation / "hard_rejection_histogram.json", dict(hard_rejections))
    _write(validation / "fallback_histogram.json", dict(fallback_reasons))

    p0_time = float(latest_run["time_to_P0_seconds"])
    p1_time = max(0.0, (p1 / "P1_completion.json").stat().st_mtime - (p0 / "P0_completion.json").stat().st_mtime)
    p2_time = float(performance["total_m5"])
    return {
        "generation": latest["generation"],
        "latest_stage": latest["stage"],
        "hard_audit_passed": hard.get("passed") is True,
        "hard_audit_fatal_failures": hard.get("fatal_failures", []),
        "diagnostic_review_status": "not_reviewed",
        "legacy_policy_result": diagnostic.get("legacy_policy_result"),
        "stages": {
            "P0": {
                "size_px": _image_shape(p0 / "base_panorama_owner_only.png"),
                "contributors": p0_completion["contributor_source_count"],
                "seconds": p0_time,
            },
            "P1": {
                "size_px": _image_shape(p1 / "vertical_panorama_owner_only.png"),
                "contributors": p1_completion["formal_raw_rgb_unique_sources"],
                "seconds": p1_time,
                "time_basis": "completion_mtime_delta",
            },
            "P2": {
                "size_px": _image_shape(p2 / "geometry_and_seam_panorama_owner_only.png"),
                "contributors": p2_completion["formal_raw_rgb_unique_sources"],
                "seconds": p2_time,
            },
        },
        "seam_counts": dict(seam_counts),
        "dp_generated": dp_generated,
        "dp_evaluated": dp_evaluated,
        "dp_selected": dp_selected,
        "dp_evaluation_status": dict(evaluation_status),
        "hard_rejection_histogram": dict(hard_rejections),
        "pair_fallback_count": sum(fallback_reasons.values()),
        "fallback_histogram": dict(fallback_reasons),
        "time_to_p0": p0_time,
        "time_p1": p1_time,
        "time_m5": p2_time,
        "m4_reestimated_in_m5": performance["m4_reestimated_in_m5"],
        "m5_gain_enumeration_count": performance["gain_enumeration_count_in_m5"],
        "m5_full_resolution_render_count": performance["p2_full_resolution_render_count"],
        "formal_raw_rgb_remaps": p2_completion["formal_raw_rgb_remap_invocations"],
        "current_reviewed": "not_present_not_reviewed",
    }


def _fit_height(image: np.ndarray, height: int) -> np.ndarray:
    width = max(1, int(round(image.shape[1] * height / image.shape[0])))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def _comparison(root: Path, old_p2: Path, fast_generation: Path) -> None:
    validation = root / "validation"
    validation.mkdir(parents=True, exist_ok=True)
    old = cv2.imread(str(old_p2 / "geometry_and_seam_panorama_owner_only.png"), cv2.IMREAD_COLOR)
    new = cv2.imread(
        str(fast_generation / "P2" / "geometry_and_seam_panorama_owner_only.png"),
        cv2.IMREAD_COLOR,
    )
    if old is None or new is None:
        raise ValueError("old/new P2 comparison images are missing")
    height = min(old.shape[0], new.shape[0])
    old_fit, new_fit = _fit_height(old, height), _fit_height(new, height)
    gap = np.zeros((height, 12, 3), np.uint8)
    gap[:, :, 2] = 255
    contact = np.concatenate((old_fit, gap, new_fit), axis=1)
    cv2.imwrite(str(validation / "old_bad_vs_new_contact_sheet.png"), contact)

    crops = validation / "high_risk_horizontal_crops"
    crops.mkdir(parents=True, exist_ok=True)
    for index, (start_fraction, end_fraction) in enumerate(((0.0, 0.34), (0.33, 0.67), (0.66, 1.0))):
        y0, y1 = int(height * start_fraction), max(int(height * end_fraction), int(height * start_fraction) + 1)
        crop = np.concatenate((old_fit[y0:y1], gap[y0:y1], new_fit[y0:y1]), axis=1)
        cv2.imwrite(str(crops / f"horizontal_band_{index:02d}_old_vs_new.png"), crop)
    for name in ("seam_model_counts.json",):
        shutil.copy2(root / "fast_direct" / "validation" / name, validation / name)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--old-p2", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    runs = {name: _run_summary(root, name) for name in RUN_NAMES}
    fast_generation = root / "fast_direct" / str(_read(root / "fast_direct/current_latest.json")["generation"])
    _comparison(root, args.old_p2.resolve(), fast_generation)
    summary = {
        "schema": "gemini305-video-s13-m51-validation/v1",
        "policy": "forward_pipeline_hard_gate_diagnostic_only",
        "review_status": "not_reviewed",
        "m6_entered": False,
        "fast_input_resolution": {
            "plan_listed": "data/captures/video/run_20260804_162340",
            "replayed_old_validation_input": "data/captures/video/run_20260807_140140",
            "reason": "old validation commands, 129-frame audit, and trajectory all bind run_20260807_140140",
        },
        "old_m5_wall_seconds": {
            "fast_direct": 36.86894220000249,
            "fast_ignore_pose": 35.43491250000079,
            "slow_direct": 39.400342500004626,
            "slow_ignore_pose": 39.39295149999816,
        },
        "old_m5_full_resolution_render_count": 14,
        "old_m5_full_resolution_render_semantics": (
            "four gains times global/local/final M4 selection renders, repeated in M5, plus two P2 renders"
        ),
        "runs": runs,
    }
    _write(root / "validation_summary.json", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
