from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pytest

import panorama_demo.stitch_sequence as sequence
from panorama_demo.synthetic import generate_sequence


@dataclass
class _FakeAlignment:
    homography_source_to_reference: np.ndarray
    preview_bgr: np.ndarray
    layout_method: str = "synthetic_known_translation"
    match_count: int = 200
    median_reprojection_px: float = 0.0

    def as_dict(self) -> dict[str, object]:
        return {
            "homography_source_to_reference": (
                self.homography_source_to_reference.tolist()
            ),
            "match_count": self.match_count,
            "layout_method": self.layout_method,
            "median_reprojection_px": self.median_reprojection_px,
        }


class _KnownTranslationAligner:
    def __init__(self, step: int) -> None:
        self.step = step

    def align(self, reference: np.ndarray, source: np.ndarray) -> _FakeAlignment:
        transform = np.array(
            [[1.0, 0.0, self.step], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        preview = np.hstack((reference, source))
        return _FakeAlignment(transform, preview)


def test_zero_parameter_sequence_publishes_one_complete_delivery(
    tmp_path: Path, monkeypatch
) -> None:
    session = generate_sequence(
        tmp_path / "session",
        frame_count=10,
        frame_width=640,
        frame_height=400,
        step=120,
        seed=19,
    )
    output = tmp_path / "output"
    monkeypatch.setattr(
        sequence,
        "build_aligner",
        lambda *_args, **_kwargs: _KnownTranslationAligner(120),
    )
    args = sequence._parser().parse_args(
        [str(session), "--output", str(output)]
    )

    report = sequence.run(args)

    panorama = cv2.imread(str(output / "panorama.jpg"), cv2.IMREAD_COLOR)
    assert panorama is not None
    assert panorama.shape[1] >= 1600
    assert panorama.shape[0] >= 390
    assert (output / "delivery.json").is_file()
    assert not (output / "failure.json").exists()
    assert report["layout_selection"]["mode"] == "adaptive_visual_motion"
    assert report["render"]["quality_metrics"]["quality_pass"] is True
    delivery = json.loads((output / "delivery.json").read_text(encoding="utf-8"))
    assert delivery["quality_pass"] is True
    assert np.all(np.max(panorama[[0, -1]], axis=2) > 0)
    assert np.all(np.max(panorama[:, [0, -1]], axis=2) > 0)


@pytest.mark.parametrize("activation", ["cli", "config"])
def test_diagnostic_mode_writes_only_non_delivery_artifacts(
    tmp_path: Path, monkeypatch, activation: str
) -> None:
    session = generate_sequence(
        tmp_path / "session",
        frame_count=10,
        frame_width=640,
        frame_height=400,
        step=120,
        seed=23,
    )
    manifest_path = session / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "capture_mode": "diagnostic_unrestricted_auto_exposure",
            "diagnostic_only": True,
            "formal_stitch_allowed": False,
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    output = tmp_path / "output"
    observed_config: dict[str, object] = {}

    def diagnostic_aligner(settings, **_kwargs):
        observed_config.update(settings)
        return _KnownTranslationAligner(120)

    monkeypatch.setattr(sequence, "build_aligner", diagnostic_aligner)
    original_capture_quality = sequence.assess_capture_quality
    original_render = sequence.render_scan_panorama

    def failing_capture_quality(*args, **kwargs):
        result = original_capture_quality(*args, **kwargs)
        result["quality_pass"] = False
        result["failure_reasons"] = ["forced test input failure"]
        return result

    def failing_render(*args, **kwargs):
        assert kwargs["quality_gate"] is False
        panorama, info = original_render(*args, **kwargs)
        info.quality_metrics["quality_pass"] = False
        return panorama, info

    monkeypatch.setattr(sequence, "assess_capture_quality", failing_capture_quality)
    monkeypatch.setattr(sequence, "render_scan_panorama", failing_render)
    arguments = [str(session), "--output", str(output)]
    if activation == "cli":
        arguments.append("--diagnostic-force")
    else:
        arguments.extend(
            [
                "--config",
                str(
                    Path(__file__).resolve().parents[1]
                    / "configs"
                    / "capture_unrestricted_auto_exposure.yaml"
                ),
            ]
        )
    args = sequence._parser().parse_args(arguments)

    report = sequence.run(args)

    panorama = cv2.imread(
        str(output / "diagnostic_panorama.jpg"), cv2.IMREAD_COLOR
    )
    assert panorama is not None
    assert (output / "diagnostic_report.json").is_file()
    assert not (output / "panorama.jpg").exists()
    assert not (output / "report.json").exists()
    assert not (output / "delivery.json").exists()
    assert not (output / "failure.json").exists()
    assert report["diagnostic_only"] is True
    assert report["deliverable_published"] is False
    assert report["input_capture"]["diagnostic_only"] is True
    assert report["input_quality"]["quality_pass"] is False
    assert report["render"]["quality_metrics"]["quality_pass"] is False
    assert observed_config["allow_magsac_fallback"] is True
    assert observed_config["prefer_magsac_layout"] is True
    assert observed_config["min_matches"] == 4
    assert observed_config["min_magsac_inlier_ratio"] == 0.0
    assert observed_config["max_unistitch_reprojection_px"] == 1_000_000.0
    assert report["diagnostic_geometry"]["official_thresholds_bypassed"] is True
