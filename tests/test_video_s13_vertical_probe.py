from __future__ import annotations

import numpy as np

from panorama_demo.video_s13_quality import prepare_seam_structure, seam_structure_metrics
from panorama_demo.video_s13_vertical_probe import seam_structure_metrics_from_exact_probe


def test_exact_nine_column_probe_preserves_seam_metrics_for_curved_seam() -> None:
    rng = np.random.default_rng(20260817)
    image = rng.integers(0, 256, size=(480, 96, 3), dtype=np.uint8)
    seam = 48 + (np.arange(480, dtype=np.int32) % 3) - 1
    full = seam_structure_metrics(prepare_seam_structure(image), seam)
    x0, x1 = 44, 53
    probe = seam_structure_metrics_from_exact_probe(image[:, x0:x1], seam - x0)
    assert full == probe


def test_exact_probe_is_black_safe() -> None:
    image = np.zeros((480, 16, 3), dtype=np.uint8)
    seam = np.full(480, 8, dtype=np.int32)
    assert seam_structure_metrics(prepare_seam_structure(image), seam) == seam_structure_metrics_from_exact_probe(
        image[:, 4:13], seam - 4
    )
