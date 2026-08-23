"""Exact M4 seam probes in the original global canvas coordinate domain."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .video_s13_quality import prepare_seam_structure, seam_structure_metrics


@dataclass(frozen=True)
class S13VerticalProbeRequest:
    pair_index: int
    gain: float
    mode: str
    global_x0: int
    global_x1: int
    local_seam_x: int


@dataclass(frozen=True)
class S13VerticalProbeResult:
    request: S13VerticalProbeRequest
    image: np.ndarray
    metrics: Mapping[str, object]


def seam_structure_metrics_from_exact_probe(
    image: np.ndarray, seam_x_by_row: np.ndarray,
) -> Mapping[str, object]:
    """Run the unmodified structure metric on a probe with local seam coordinates."""

    features = prepare_seam_structure(image)
    return seam_structure_metrics(image if features is None else features, seam_x_by_row)


__all__ = [
    "S13VerticalProbeRequest", "S13VerticalProbeResult", "seam_structure_metrics_from_exact_probe",
]
