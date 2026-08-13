from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from pathlib import Path

import cv2
import numpy as np

from panorama_demo.video_s13_m61_hard_audit import (
    _replay_pixels,
    _replay_pixels_full_canvas,
)
from panorama_demo.video_s13_m9 import (
    BRANCHES,
    PROFILE_SCHEMA,
    STAGE_FIELDS,
    compare_exact_file,
    compare_profiles_exact,
    summarize_benchmark,
)


class _CountingImages(Mapping[int, np.ndarray]):
    def __init__(self, values: Mapping[int, np.ndarray], *, reverse: bool = False) -> None:
        self.values = dict(values)
        self.reverse = reverse
        self.reads = 0

    def __getitem__(self, key: int) -> np.ndarray:
        self.reads += 1
        return self.values[key]

    def __iter__(self) -> Iterator[int]:
        keys = sorted(self.values, reverse=self.reverse)
        return iter(keys)

    def __len__(self) -> int:
        return len(self.values)


def _fixture(shape: tuple[int, int] = (7, 9)) -> tuple[dict[str, np.ndarray], dict[int, np.ndarray], dict[int, tuple[np.ndarray, np.ndarray]]]:
    height, width = shape
    yy, xx = np.indices(shape)
    valid = np.ones(shape, bool)
    owner = np.where(xx < width // 2, 10, 20).astype(np.int32)
    secondary_weight = np.zeros(shape, np.float32)
    secondary_weight[:, max(1, width // 2 - 1): min(width, width // 2 + 1)] = 0.25
    secondary = np.where(owner == 10, 20, 10).astype(np.int32)
    provenance = {
        "valid": valid,
        "owner_frame_id": owner,
        "source_u": (xx + 0.25).astype(np.float32),
        "source_v": (yy + 0.125).astype(np.float32),
        "secondary_weight": secondary_weight,
        "secondary_frame_id": np.where(secondary_weight > 0, secondary, -1).astype(np.int32),
        "secondary_source_u": (xx + 0.5).astype(np.float32),
        "secondary_source_v": (yy + 0.375).astype(np.float32),
    }
    image_shape = (height + 2, width + 2, 3)
    first = np.arange(np.prod(image_shape), dtype=np.uint8).reshape(image_shape)
    second = np.flip(first, axis=1).copy()
    images = {10: first, 20: second}
    parameters = {
        10: (np.array([1.0, 0.95, 1.05]), np.array([0.0, 0.01, -0.01])),
        20: (np.array([0.9, 1.0, 1.1]), np.array([0.02, 0.0, -0.02])),
    }
    return provenance, images, parameters


def test_compact_replay_is_byte_exact_and_one_decode_remap_per_source() -> None:
    provenance, images, parameters = _fixture()
    reference = _replay_pixels_full_canvas(provenance, images, parameters)
    lazy = _CountingImages(images)
    telemetry: dict[str, int] = {}
    optimized = _replay_pixels(provenance, lazy, parameters, telemetry=telemetry)
    assert np.array_equal(optimized, reference)
    assert lazy.reads == len(images)
    assert telemetry == {"remap_invocation_count": len(images)}


def test_compact_replay_handles_more_than_opencv_short_rows_exactly() -> None:
    provenance, images, parameters = _fixture((400, 100))
    reference = _replay_pixels_full_canvas(provenance, images, parameters)
    optimized = _replay_pixels(provenance, images, parameters)
    assert np.array_equal(optimized, reference)


def test_cache_and_iteration_order_do_not_change_replay() -> None:
    provenance, images, parameters = _fixture()
    cached = _replay_pixels(provenance, images, parameters)
    reverse_lazy = _CountingImages(images, reverse=True)
    uncached = _replay_pixels(provenance, reverse_lazy, parameters)
    assert np.array_equal(cached, uncached)


def test_exact_file_comparison_covers_image_npz_and_json(tmp_path: Path) -> None:
    image = np.arange(36, dtype=np.uint8).reshape(3, 4, 3)
    for name in ("a.png", "b.png"):
        assert cv2.imwrite(str(tmp_path / name), image)
    np.savez_compressed(tmp_path / "a.npz", x=np.arange(5), y=np.array([np.nan]))
    np.savez_compressed(tmp_path / "b.npz", x=np.arange(5), y=np.array([np.nan]))
    (tmp_path / "a.json").write_text('{"b":2,"a":1}', encoding="utf-8")
    (tmp_path / "b.json").write_text('{"a":1,"b":2}', encoding="utf-8")
    assert compare_exact_file(tmp_path / "a.png", tmp_path / "b.png")["exact"]
    assert compare_exact_file(tmp_path / "a.npz", tmp_path / "b.npz")["exact"]
    assert compare_exact_file(tmp_path / "a.json", tmp_path / "b.json")["exact"]


def _profile(path: Path, hashes: Mapping[str, str], label: str) -> None:
    runs = []
    for branch in BRANCHES:
        stages = {name: 0.1 for name in STAGE_FIELDS}
        runs.append({
            "branch": branch,
            "stage_seconds": stages,
            "peak_rss_bytes": 1,
            "peak_gpu_memory_bytes": None,
            "decode_invocation_count": 1,
            "formal_remap_invocation_count": 1,
            "audit_replay_remap_invocation_count": 1,
            "sealed_output": {"files": dict(hashes)},
        })
    path.write_text(json.dumps({"schema": PROFILE_SCHEMA, "label": label, "runs": runs}), encoding="utf-8")


def test_profile_equivalence_fails_closed_on_any_sealed_hash_change(tmp_path: Path) -> None:
    baseline, optimized, output = tmp_path / "baseline.json", tmp_path / "optimized.json", tmp_path / "equivalence.json"
    _profile(baseline, {"visual_panorama.png": "a", "p3_pixel_provenance.npz": "b"}, "baseline")
    _profile(optimized, {"visual_panorama.png": "changed", "p3_pixel_provenance.npz": "b"}, "optimized")
    result = compare_profiles_exact(baseline, optimized, output)
    assert result["exact"] is False
    assert result["classification"] == "non_exact_requires_M6_M8_rerun"


def test_benchmark_summary_seals_required_diagnostic_artifacts(tmp_path: Path) -> None:
    hashes = {"visual_panorama.png": "a", "p3_pixel_provenance.npz": "b", "hard_audit.json": "c", "P3_completion.json": "d"}
    _profile(tmp_path / "baseline.json", hashes, "baseline")
    _profile(tmp_path / "optimized.json", hashes, "optimized")
    compare_profiles_exact(tmp_path / "baseline.json", tmp_path / "optimized.json", tmp_path / "equivalence.json")
    completion = summarize_benchmark(tmp_path)
    assert completion["exact"] is True
    assert completion["diagnostic_only"] is True
    assert completion["production_lock_created"] is False
    assert completion["twenty_metre_sla_claim"] is False
    assert set(completion["assets_sha256"]) == {
        "baseline.json", "optimized.json", "equivalence.json",
        "four_run_summary.json", "stage_breakdown.csv", "memory.csv",
    }
