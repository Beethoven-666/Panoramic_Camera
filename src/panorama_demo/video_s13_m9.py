"""S013 M9 profiling and exact-output comparison utilities.

M9 is diagnostic-only.  These helpers never render, mutate a sealed stage, or
advance a stage pointer.  Missing legacy timing fields remain explicitly
unavailable instead of being inferred from a total.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import threading
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .video_s13_evaluation import BRANCHES, verify_review_bundle
from .video_s13_m61_hard_audit import _replay_pixels, _replay_pixels_full_canvas


PROFILE_SCHEMA = "gemini305-video-s13-m9-profile/v1"
EQUIVALENCE_SCHEMA = "gemini305-video-s13-m9-exact-equivalence/v1"
COMPLETION_SCHEMA = "gemini305-video-s13-m9-benchmark-completion/v1"
STAGE_FIELDS = (
    "input_and_preflight", "trajectory_verify", "motion_measurement",
    "progress_and_layout", "time_to_P0", "P1", "P2_geometry", "P2_seam",
    "P2_render", "P2_export", "P3_replay_verify",
    "P3_photometric_sampling", "P3_photometric_solve", "P3_blend_analysis",
    "P3_full_resolution_render", "P3_export", "P4_risk_analysis",
    "P4_candidate_analysis", "P4_render", "P4_export", "evaluation_bundle",
    "total_wall_time",
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(value), sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


class CountingRawRGB(Mapping[int, np.ndarray]):
    """Lazy, non-caching raw RGB mapping with truthful decode accounting."""

    def __init__(self, paths: Mapping[int, Path]) -> None:
        self._paths = dict(paths)
        self.decode_count = 0
        self.decoded_frame_ids: list[int] = []

    def __getitem__(self, frame_id: int) -> np.ndarray:
        image = cv2.imread(str(self._paths[frame_id]), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"raw RGB is unreadable: {self._paths[frame_id]}")
        self.decode_count += 1
        self.decoded_frame_ids.append(int(frame_id))
        return image

    def __iter__(self) -> Iterator[int]:
        return iter(self._paths)

    def __len__(self) -> int:
        return len(self._paths)


@contextmanager
def _peak_rss_sampler() -> Iterator[dict[str, int]]:
    sample = {"peak": 0}
    stop = threading.Event()

    def poll() -> None:
        try:
            import psutil

            process = psutil.Process(os.getpid())
            while not stop.wait(0.01):
                sample["peak"] = max(sample["peak"], int(process.memory_info().rss))
            sample["peak"] = max(sample["peak"], int(process.memory_info().rss))
        except ImportError:
            sample["peak"] = 0

    thread = threading.Thread(target=poll, daemon=True)
    thread.start()
    try:
        yield sample
    finally:
        stop.set()
        thread.join()


def _run_paths(run: Mapping[str, Any]) -> tuple[Path, Path, dict[int, Path]]:
    completions = run["stage_completions"]
    p2 = Path(completions["P2"]["path"]).resolve().parent
    p3 = Path(completions["P3"]["path"]).resolve().parent
    session = Path(run["data"]["session_root"]).resolve()
    raw = {
        int(item["frame_id"]): session / str(item["path"])
        for item in run["data"]["rgb_files"]
    }
    return p2, p3, raw


def profile_s13_pipeline(lock_path: str | Path, output_path: str | Path, *, label: str) -> dict[str, Any]:
    """Profile the four sealed M8 branches without changing their outputs."""
    if label not in {"baseline", "optimized"}:
        raise ValueError("M9 profile label must be baseline or optimized")
    replayer = _replay_pixels_full_canvas if label == "baseline" else _replay_pixels
    replay_mode = "legacy_full_canvas_two_pass/v1" if label == "baseline" else "compact_union_roi_one_decode_remap/v1"
    lock_path, output_path = Path(lock_path).resolve(), Path(output_path).resolve()
    lock = _json(lock_path)
    runs_by_branch = {str(run["branch"]): run for run in lock.get("runs", [])}
    if set(runs_by_branch) != set(BRANCHES):
        raise ValueError("M9 profiling requires exactly the four M8 branches")
    rows: list[dict[str, Any]] = []
    for branch in BRANCHES:
        run = runs_by_branch[branch]
        p2, p3, raw_paths = _run_paths(run)
        p2_perf = _json(p2 / "performance.json")
        p3_perf = _json(p3 / "performance.json")
        solution = _json(p3 / "photometric_solution.json")
        parameters = {
            int(source["frame_id"]): (
                np.asarray(source["gain_bgr"], np.float64),
                np.asarray(source["bias_bgr_linear"], np.float64),
            )
            for source in solution["sources"]
        }
        with np.load(p3 / "p3_pixel_provenance.npz", allow_pickle=False) as archive:
            provenance = {name: np.asarray(archive[name]) for name in archive.files}
        valid = np.asarray(provenance["valid"], bool)
        visual = cv2.imread(str(p3 / "visual_panorama.png"), cv2.IMREAD_COLOR)
        if visual is None:
            raise ValueError("sealed P3 visual panorama is unreadable")
        raw = CountingRawRGB(raw_paths)
        replay_telemetry: dict[str, int] = {}
        with _peak_rss_sampler() as memory:
            tick = time.perf_counter()
            replay = replayer(provenance, raw, parameters, telemetry=replay_telemetry)
            if not np.array_equal(replay[valid], visual[valid]):
                raise ValueError("sealed P3 failed M9 independent raw-RGB replay")
            replay_seconds = time.perf_counter() - tick
            tick = time.perf_counter()
            verify_review_bundle(Path(run["bundle_manifest"]["path"]).parent)
            bundle_seconds = time.perf_counter() - tick
        stages: dict[str, float | None] = {name: None for name in STAGE_FIELDS}
        stages.update({
            "input_and_preflight": p2_perf.get("input_and_preflight"),
            "motion_measurement": p2_perf.get("motion_measurement"),
            "progress_and_layout": p2_perf.get("progress_and_layout"),
            "time_to_P0": p2_perf.get("time_to_P0"),
            "P2_geometry": p2_perf.get("geometry"),
            "P2_seam": p2_perf.get("seam"),
            "P3_replay_verify": replay_seconds,
            "P3_full_resolution_render": p3_perf.get("elapsed_seconds"),
            "evaluation_bundle": bundle_seconds,
            "total_wall_time": float(p2_perf["total_wall_time"]) + float(p3_perf["elapsed_seconds"]) + replay_seconds + bundle_seconds,
        })
        rows.append({
            "branch": branch,
            "generation_id": run["generation_id"],
            "reviewed_stage": run["reviewed_stage"],
            "stage_seconds": stages,
            "peak_rss_bytes": int(memory["peak"]),
            "peak_gpu_memory_bytes": None,
            "peak_gpu_memory_state": "not_measured_G305_CUDA_off",
            "decoded_source_count": len(set(raw.decoded_frame_ids)),
            "decode_invocation_count": raw.decode_count,
            "formal_remap_invocation_count": int(p3_perf["formal_raw_rgb_remap_invocations"]),
            "audit_replay_remap_invocation_count": replay_telemetry["remap_invocation_count"],
            "full_canvas_source_cache_created": bool(p3_perf["full_canvas_source_cache_created"]),
            "sealed_output": stage_fingerprint(p3),
            "measurement_notes": {
                "sealed_timing_source": "existing P2/P3 performance.json",
                "replay_and_bundle_timing_source": "time.perf_counter in this profiling invocation",
                "unavailable_stage_fields": [name for name, value in stages.items() if value is None],
                "p4": "not_present_no_real_M7_winner",
            },
        })
    document = {
        "schema": PROFILE_SCHEMA,
        "label": label,
        "replay_implementation": replay_mode,
        "diagnostic_only": True,
        "production_claim": False,
        "twenty_metre_sla_claim": False,
        "source_evaluation_lock": {"path": str(lock_path), "sha256": _sha(lock_path)},
        "runs": rows,
    }
    _write_json(output_path, document)
    return document


def stage_fingerprint(stage: str | Path) -> dict[str, Any]:
    stage = Path(stage).resolve()
    result: dict[str, Any] = {"root": str(stage), "files": {}}
    for path in sorted(stage.rglob("*")):
        if path.is_file():
            result["files"][path.relative_to(stage).as_posix()] = _sha(path)
    return result


def _same_array(left: np.ndarray, right: np.ndarray) -> bool:
    return left.shape == right.shape and left.dtype == right.dtype and bool(
        np.array_equal(left, right, equal_nan=True) if left.dtype.kind == "f" else np.array_equal(left, right)
    )


def compare_exact_file(left: str | Path, right: str | Path) -> dict[str, Any]:
    left, right = Path(left).resolve(), Path(right).resolve()
    suffix = left.suffix.lower()
    if suffix != right.suffix.lower():
        return {"exact": False, "reason": "suffix_mismatch"}
    result: dict[str, Any] = {"left": str(left), "right": str(right), "kind": suffix, "file_sha_equal": _sha(left) == _sha(right)}
    if suffix in {".png", ".jpg", ".jpeg"}:
        a, b = cv2.imread(str(left), cv2.IMREAD_UNCHANGED), cv2.imread(str(right), cv2.IMREAD_UNCHANGED)
        result["content_equal"] = a is not None and b is not None and _same_array(a, b)
    elif suffix == ".npz":
        with np.load(left, allow_pickle=False) as aa, np.load(right, allow_pickle=False) as bb:
            result["fields_equal"] = aa.files == bb.files
            result["content_equal"] = result["fields_equal"] and all(_same_array(aa[name], bb[name]) for name in aa.files)
    elif suffix == ".json":
        result["content_equal"] = json.loads(left.read_text(encoding="utf-8")) == json.loads(right.read_text(encoding="utf-8"))
    else:
        result["content_equal"] = result["file_sha_equal"]
    result["exact"] = bool(result["content_equal"])
    return result


def compare_profiles_exact(baseline_path: str | Path, optimized_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    baseline, optimized = _json(Path(baseline_path)), _json(Path(optimized_path))
    base_runs = {run["branch"]: run for run in baseline["runs"]}
    opt_runs = {run["branch"]: run for run in optimized["runs"]}
    runs = []
    for branch in BRANCHES:
        left = base_runs[branch]["sealed_output"]["files"]
        right = opt_runs[branch]["sealed_output"]["files"]
        runs.append({
            "branch": branch,
            "file_set_equal": set(left) == set(right),
            "all_file_sha256_equal": left == right,
            "p3_result_hash_equal": left.get("visual_panorama.png") == right.get("visual_panorama.png"),
            "provenance_hash_equal": left.get("p3_pixel_provenance.npz") == right.get("p3_pixel_provenance.npz"),
            "hard_audit_hash_equal": left.get("hard_audit.json") == right.get("hard_audit.json"),
            "completion_hash_equal": left.get("P3_completion.json") == right.get("P3_completion.json"),
        })
    exact = all(run["file_set_equal"] and run["all_file_sha256_equal"] for run in runs)
    document = {
        "schema": EQUIVALENCE_SCHEMA,
        "exact": exact,
        "classification": "A_exact_output_preserving" if exact else "non_exact_requires_M6_M8_rerun",
        "runs": runs,
        "baseline_sha256": _sha(Path(baseline_path)),
        "optimized_sha256": _sha(Path(optimized_path)),
    }
    _write_json(Path(output_path), document)
    return document


def summarize_benchmark(root: str | Path) -> dict[str, Any]:
    root = Path(root).resolve()
    baseline, optimized, equivalence = (_json(root / name) for name in ("baseline.json", "optimized.json", "equivalence.json"))
    baseline_by = {run["branch"]: run for run in baseline["runs"]}
    optimized_by = {run["branch"]: run for run in optimized["runs"]}
    rows = []
    for branch in BRANCHES:
        before = float(baseline_by[branch]["stage_seconds"]["P3_replay_verify"])
        after = float(optimized_by[branch]["stage_seconds"]["P3_replay_verify"])
        rows.append({"branch": branch, "baseline_replay_seconds": before, "optimized_replay_seconds": after, "speedup": before / after if after else None})
    summary = {"schema": "gemini305-video-s13-m9-four-run-summary/v1", "runs": rows, "exact": equivalence["exact"]}
    _write_json(root / "four_run_summary.json", summary)
    with (root / "stage_breakdown.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("profile", "branch", "stage", "seconds"))
        for profile in (baseline, optimized):
            for run in profile["runs"]:
                for stage in STAGE_FIELDS:
                    writer.writerow((profile["label"], run["branch"], stage, run["stage_seconds"][stage]))
    with (root / "memory.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("profile", "branch", "peak_rss_bytes", "peak_gpu_memory_bytes", "decode_invocations", "formal_remaps", "audit_remaps"))
        for profile in (baseline, optimized):
            for run in profile["runs"]:
                writer.writerow((profile["label"], run["branch"], run["peak_rss_bytes"], run["peak_gpu_memory_bytes"], run["decode_invocation_count"], run["formal_remap_invocation_count"], run["audit_replay_remap_invocation_count"]))
    asset_names = [
        "baseline.json", "optimized.json", "equivalence.json",
        "four_run_summary.json", "stage_breakdown.csv", "memory.csv",
    ]
    if (root / "M9_test_summary.json").is_file():
        asset_names.append("M9_test_summary.json")
    completion = {
        "schema": COMPLETION_SCHEMA,
        "diagnostic_only": True,
        "production_lock_created": False,
        "twenty_metre_sla_claim": False,
        "exact": equivalence["exact"],
        "assets_sha256": {name: _sha(root / name) for name in asset_names},
    }
    _write_json(root / "benchmark_completion.json", completion)
    return completion


__all__ = [
    "BRANCHES", "COMPLETION_SCHEMA", "EQUIVALENCE_SCHEMA", "PROFILE_SCHEMA",
    "STAGE_FIELDS", "compare_exact_file", "compare_profiles_exact",
    "profile_s13_pipeline", "stage_fingerprint", "summarize_benchmark",
]
