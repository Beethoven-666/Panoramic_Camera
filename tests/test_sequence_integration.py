from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

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
