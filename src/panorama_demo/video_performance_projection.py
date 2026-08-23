"""Fail-closed fixed-run scaling projection for the locked video benchmark."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Callable, Mapping

from .video_runtime_environment import atomic_write_json


SCALING_PROJECTION_SCHEMA = "gemini305-video-scaling-projection/v1"
REQUIRED_PREFIXES = (0.25, 0.50, 0.75, 1.00)
PREFIX_LABELS = {
    0.25: "P25",
    0.50: "P50",
    0.75: "P75",
    1.00: "P100",
}
PREFIX_REPEATS = 3
FINAL_COLD_REPEATS = 1
FINAL_WARM_REPEATS = 5
PREFIX_MEASUREMENT_SCHEMA = "gemini305-video-fixed-prefix-measurements/v1"
FINAL_PERFORMANCE_SCHEMA = "gemini305-video-final-performance-evidence/v1"


class FixedRunPerformanceError(RuntimeError):
    """The supplied fixed-run evidence is incomplete or internally inconsistent."""


@dataclass(frozen=True)
class FixedRunMeasurementRequest:
    """An isolated requested measurement for a caller-owned, frozen runner.

    This module does not open a session, select an algorithm, reserve a holdout,
    or create a production lock.  The caller receives a unique output directory
    and must write the renderer's actual ``video_report.json`` there.  Keeping
    the renderer callable injected makes the evidence orchestration reusable
    while preventing this utility from silently choosing a data split.
    """

    label: str
    progress: float
    repeat: int
    run_kind: str
    output: Path


FixedRunMeasurementRunner = Callable[[FixedRunMeasurementRequest], object]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_algorithm_identity(report: Mapping[str, object]) -> dict[str, object]:
    algorithm = report.get("algorithm")
    if not isinstance(algorithm, Mapping):
        raise FixedRunPerformanceError("video_report.json lacks algorithm identity")
    identity: dict[str, object] = {}
    for key in ("role", "algorithm_id", "implementation_id", "config_sha256", "source_commit"):
        value = algorithm.get(key)
        if not isinstance(value, str) or not value:
            raise FixedRunPerformanceError(f"video_report.json algorithm.{key} must be a non-empty string")
        identity[key] = value
    model_sha256 = algorithm.get("model_sha256")
    if not isinstance(model_sha256, Mapping):
        raise FixedRunPerformanceError("video_report.json algorithm.model_sha256 must be an object")
    identity["model_sha256"] = dict(model_sha256)
    return identity


def _read_measurement_report(request: FixedRunMeasurementRequest) -> dict[str, object]:
    report_path = request.output / "video_report.json"
    if not report_path.is_file():
        raise FixedRunPerformanceError(
            f"{request.label} repeat {request.repeat} did not publish video_report.json"
        )
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FixedRunPerformanceError(f"Invalid measurement report: {report_path}") from exc
    if not isinstance(report, dict):
        raise FixedRunPerformanceError(f"Measurement report must be a JSON object: {report_path}")
    performance = report.get("performance")
    if not isinstance(performance, Mapping):
        raise FixedRunPerformanceError(f"Measurement report lacks performance: {report_path}")
    seconds = performance.get("primary_post_capture_seconds")
    if not isinstance(seconds, (int, float)) or isinstance(seconds, bool) or not math.isfinite(seconds) or seconds <= 0.0:
        raise FixedRunPerformanceError(
            f"Measurement primary_post_capture_seconds must be finite and positive: {report_path}"
        )
    grades = report.get("grades")
    if not isinstance(grades, Mapping) or grades.get("performance") != "A":
        raise FixedRunPerformanceError(f"Measurement performance grade is not A: {report_path}")
    return {
        "label": request.label,
        "progress": request.progress,
        "repeat": request.repeat,
        "run_kind": request.run_kind,
        "seconds": float(seconds),
        "report_path": str(report_path),
        "report_sha256": _sha256_file(report_path),
        "algorithm": _canonical_algorithm_identity(report),
    }


def _validate_output_root(output: str | Path, *, evidence_names: tuple[str, ...]) -> Path:
    root = Path(output).expanduser().resolve()
    existing = [root / name for name in evidence_names if (root / name).exists()]
    if existing:
        raise FixedRunPerformanceError(
            "Refusing to overwrite fixed-run evidence: " + ", ".join(str(path) for path in existing)
        )
    root.mkdir(parents=True, exist_ok=True)
    return root


def _collect_measurements(
    runner: FixedRunMeasurementRunner,
    *,
    output: Path,
    requests: tuple[tuple[str, float, int, str], ...],
) -> list[dict[str, object]]:
    """Run an exact caller-provided schedule and bind each report by hash."""

    records: list[dict[str, object]] = []
    identity_json: str | None = None
    for label, progress, repeat, run_kind in requests:
        target = output / "runs" / run_kind / label / f"repeat_{repeat:02d}"
        if target.exists():
            raise FixedRunPerformanceError(f"Refusing to reuse measurement output: {target}")
        request = FixedRunMeasurementRequest(
            label=label, progress=progress, repeat=repeat, run_kind=run_kind, output=target
        )
        runner(request)
        record = _read_measurement_report(request)
        encoded_identity = json.dumps(record["algorithm"], sort_keys=True, separators=(",", ":"))
        if identity_json is None:
            identity_json = encoded_identity
        elif identity_json != encoded_identity:
            raise FixedRunPerformanceError("Fixed-run measurements used more than one algorithm identity")
        records.append(record)
    return records


def _prefix_schedule() -> tuple[tuple[str, float, int, str], ...]:
    return tuple(
        (PREFIX_LABELS[progress], progress, repeat, "prefix")
        for progress in REQUIRED_PREFIXES
        for repeat in range(1, PREFIX_REPEATS + 1)
    )


def _prefix_summary(records: list[dict[str, object]]) -> tuple[dict[float, float], list[dict[str, object]]]:
    expected = {
        (PREFIX_LABELS[progress], progress, repeat, "prefix")
        for progress in REQUIRED_PREFIXES
        for repeat in range(1, PREFIX_REPEATS + 1)
    }
    observed = {
        (str(item.get("label")), float(item.get("progress")), int(item.get("repeat")), str(item.get("run_kind")))
        for item in records
    }
    if observed != expected or len(records) != PREFIX_REPEATS * len(REQUIRED_PREFIXES):
        raise FixedRunPerformanceError("Prefix evidence must contain exactly P25/P50/P75/P100 with three repeats each")
    samples: list[dict[str, object]] = []
    projection_seconds: dict[float, float] = {}
    for progress in REQUIRED_PREFIXES:
        values = sorted(float(item["seconds"]) for item in records if item["progress"] == progress)
        # Use the slowest of the three real runs.  A median here would make a
        # performance projection optimistic and could conceal timing variance.
        projection_seconds[progress] = values[-1]
        samples.append(
            {
                "label": PREFIX_LABELS[progress],
                "progress": progress,
                "repeat_count": len(values),
                "seconds": values,
                "projection_seconds_reducer": "maximum",
                "projection_seconds": values[-1],
            }
        )
    return projection_seconds, samples


def _write_evidence(path: Path, payload: Mapping[str, object]) -> None:
    """Write a new result only; benchmark evidence is never silently replaced."""

    if path.exists():
        raise FixedRunPerformanceError(f"Refusing to overwrite fixed-run evidence: {path}")
    atomic_write_json(path, dict(payload))


def run_fixed_prefix_measurements(
    runner: FixedRunMeasurementRunner,
    *,
    output: str | Path,
    nominal_full_length_m: float = 3.0,
    target_length_m: float = 20.0,
    safety_margin_seconds: float = 5.0,
) -> dict[str, object]:
    """Execute exactly 3 measurements for every fixed P25/P50/P75/P100 prefix.

    The returned projection is deliberately a *fixed-run performance
    projection*, never a claim that a real 20 m session passed.  This helper
    only calls the injected runner and writes under ``output``; it neither
    opens nor alters the locked dataset, invokes the first-holdout lifecycle,
    nor creates a production lock.
    """

    root = _validate_output_root(
        output, evidence_names=("fixed_prefix_measurements.json", "scaling_projection.json")
    )
    records = _collect_measurements(runner, output=root, requests=_prefix_schedule())
    projection_seconds, samples = _prefix_summary(records)
    projection = fixed_run_performance_projection(
        projection_seconds,
        nominal_full_length_m=nominal_full_length_m,
        target_length_m=target_length_m,
        safety_margin_seconds=safety_margin_seconds,
    )
    payload: dict[str, object] = {
        "schema": PREFIX_MEASUREMENT_SCHEMA,
        "claim": "fixed-run prefix measurement evidence",
        "does_not_run_holdout": True,
        "does_not_mutate_dataset": True,
        "does_not_create_production_lock": True,
        "required_prefixes": list(REQUIRED_PREFIXES),
        "required_repeats_per_prefix": PREFIX_REPEATS,
        "measurements": records,
        "prefix_summary": samples,
        "projection": projection,
        "performance_projection_gate_passed": projection["status"] == "passed",
    }
    _write_evidence(root / "fixed_prefix_measurements.json", payload)
    _write_evidence(root / "scaling_projection.json", projection)
    return payload


def run_final_fixed_run_performance_evidence(
    runner: FixedRunMeasurementRunner,
    *,
    output: str | Path,
    nominal_full_length_m: float = 3.0,
    target_length_m: float = 20.0,
    safety_margin_seconds: float = 5.0,
) -> dict[str, object]:
    """Run the complete final schedule: one cold, five warm, and 4 x 3 prefixes.

    This is an evidence orchestrator, not a lifecycle transition.  In
    particular it intentionally cannot run/reserve holdout or write a
    production lock; callers must arrange those immutable prerequisites before
    asking a frozen renderer to satisfy this schedule.
    """

    root = _validate_output_root(
        output, evidence_names=("final_performance_evidence.json", "scaling_projection.json")
    )
    full_schedule = (("P100", 1.0, 1, "cold"),) + tuple(
        ("P100", 1.0, repeat, "warm") for repeat in range(1, FINAL_WARM_REPEATS + 1)
    )
    full_records = _collect_measurements(runner, output=root, requests=full_schedule)
    prefix_records = _collect_measurements(runner, output=root, requests=_prefix_schedule())
    projection_seconds, prefix_summary = _prefix_summary(prefix_records)
    projection = fixed_run_performance_projection(
        projection_seconds,
        nominal_full_length_m=nominal_full_length_m,
        target_length_m=target_length_m,
        safety_margin_seconds=safety_margin_seconds,
    )
    cold_seconds = float(full_records[0]["seconds"])
    warm_seconds = [float(record["seconds"]) for record in full_records[1:]]
    warm_median = float(statistics.median(warm_seconds))
    warm_max = max(warm_seconds)
    warm_min = min(warm_seconds)
    relative_range = (warm_max - warm_min) / warm_median
    gates = {
        "exact_cold_repeats": len(full_records[:1]) == FINAL_COLD_REPEATS,
        "exact_warm_repeats": len(warm_seconds) == FINAL_WARM_REPEATS,
        "exact_prefix_repeats": len(prefix_records) == PREFIX_REPEATS * len(REQUIRED_PREFIXES),
        "cold_3m_le_12s": cold_seconds <= 12.0,
        "warm_median_3m_le_8s": warm_median <= 8.0,
        "warm_max_3m_le_9s": warm_max <= 9.0,
        "warm_relative_range_lt_3_percent": relative_range < 0.03,
        "performance_projection_passed": projection["status"] == "passed",
    }
    payload: dict[str, object] = {
        "schema": FINAL_PERFORMANCE_SCHEMA,
        "claim": "fixed-run final performance evidence",
        "does_not_prove_real_20m": True,
        "does_not_run_holdout": True,
        "does_not_mutate_dataset": True,
        "does_not_create_production_lock": True,
        "required_schedule": {
            "cold": FINAL_COLD_REPEATS,
            "warm": FINAL_WARM_REPEATS,
            "prefix_repeats": PREFIX_REPEATS,
            "prefixes": list(REQUIRED_PREFIXES),
        },
        "full_runs": full_records,
        "prefix_runs": prefix_records,
        "prefix_summary": prefix_summary,
        "summary": {
            "cold_seconds": cold_seconds,
            "warm_median_seconds": warm_median,
            "warm_max_seconds": warm_max,
            "warm_relative_range": relative_range,
        },
        "projection": projection,
        "gates": gates,
        "status": "passed" if all(gates.values()) else "failed",
    }
    _write_evidence(root / "final_performance_evidence.json", payload)
    _write_evidence(root / "scaling_projection.json", projection)
    return payload


def fixed_run_performance_projection(
    prefix_seconds: Mapping[float, float],
    *,
    nominal_full_length_m: float = 3.0,
    target_length_m: float = 20.0,
    safety_margin_seconds: float = 5.0,
) -> dict[str, object]:
    """Fit ``T(L)=T_fixed+kL`` and expose conservative evidence, never a claim.

    A maximum adjacent incremental slope is deliberately used for the
    conservative extrapolation.  This cannot prove a real 20 m run, and a
    failed projection is recorded rather than converted to a fallback.
    """

    expected = set(REQUIRED_PREFIXES)
    received = {float(key) for key in prefix_seconds}
    if received != expected:
        raise ValueError(f"prefix_seconds must contain exactly {list(REQUIRED_PREFIXES)}")
    if not all(math.isfinite(value) and value > 0.0 for value in prefix_seconds.values()):
        raise ValueError("prefix timings must be finite positive seconds")
    if not all(math.isfinite(value) and value > 0.0 for value in (nominal_full_length_m, target_length_m, safety_margin_seconds)):
        raise ValueError("projection dimensions and margin must be finite positive values")
    samples = sorted(
        (float(prefix) * nominal_full_length_m, float(prefix_seconds[prefix]))
        for prefix in REQUIRED_PREFIXES
    )
    mean_length = sum(length for length, _ in samples) / len(samples)
    mean_seconds = sum(seconds for _, seconds in samples) / len(samples)
    denominator = sum((length - mean_length) ** 2 for length, _ in samples)
    if denominator <= 0.0:
        raise ValueError("prefix lengths do not identify a scaling slope")
    slope = sum((length - mean_length) * (seconds - mean_seconds) for length, seconds in samples) / denominator
    fixed_seconds = mean_seconds - slope * mean_length
    incremental_slopes = [
        (later_seconds - earlier_seconds) / (later_length - earlier_length)
        for (earlier_length, earlier_seconds), (later_length, later_seconds) in zip(samples, samples[1:])
    ]
    max_incremental_slope = max(incremental_slopes)
    # Negative intercepts are not an admissible optimistic "fixed cost".
    conservative_fixed_seconds = max(0.0, fixed_seconds)
    linear_seconds = fixed_seconds + slope * target_length_m
    conservative_seconds = conservative_fixed_seconds + max_incremental_slope * target_length_m
    linear_pass = linear_seconds <= 50.0
    conservative_pass = conservative_seconds <= 55.0
    safety_margin_pass = conservative_seconds + safety_margin_seconds <= 60.0
    passed = linear_pass and conservative_pass and safety_margin_pass
    return {
        "schema": SCALING_PROJECTION_SCHEMA,
        "claim": "fixed-run performance projection",
        "does_not_prove_real_20m": True,
        "nominal_full_length_m": nominal_full_length_m,
        "target_length_m": target_length_m,
        "safety_margin_seconds": safety_margin_seconds,
        "prefix_samples": [
            {"progress": progress, "nominal_length_m": progress * nominal_full_length_m, "seconds": prefix_seconds[progress]}
            for progress in REQUIRED_PREFIXES
        ],
        "fit": {
            "model": "T(L) = T_fixed + k * L",
            "fixed_seconds": fixed_seconds,
            "linear_slope_seconds_per_m": slope,
            "maximum_incremental_slope_seconds_per_m": max_incremental_slope,
        },
        "projection": {
            "linear_seconds": linear_seconds,
            "conservative_seconds": conservative_seconds,
            "conservative_with_safety_margin_seconds": conservative_seconds + safety_margin_seconds,
        },
        "gates": {
            "linear_20m_le_50s": linear_pass,
            "conservative_20m_le_55s": conservative_pass,
            "conservative_plus_margin_le_60s": safety_margin_pass,
        },
        "status": "passed" if passed else "failed",
    }


def write_fixed_run_performance_projection(path: Path, prefix_seconds: Mapping[float, float], **kwargs: float) -> dict[str, object]:
    payload = fixed_run_performance_projection(prefix_seconds, **kwargs)
    atomic_write_json(path, payload)
    return payload
