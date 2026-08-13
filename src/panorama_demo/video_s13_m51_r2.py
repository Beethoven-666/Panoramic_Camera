"""Frozen configuration surface for the isolated S1.3 M5.1-r2 candidate.

The default is deliberately legacy-compatible.  LK statistics may be collected
while ``enabled`` is false, but no correspondence is then removed beyond the
historic status/finite checks.  The two LK error limits have no guessed
defaults: an enabled candidate must supply values frozen from instrumentation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class S13M51R2Config:
    enabled: bool = False
    collect_lk_error_statistics: bool = True
    lk_error_sigma_multiplier: float = 3.0
    lk_error_mad_floor: float | None = None
    lk_error_absolute_maximum: float | None = None
    minimum_correspondence_count_for_error_filter: int = 16
    evidence_half_widths_px: tuple[int, ...] = (16, 24)
    moving_evidence_margin_px: int = 8
    block_height_px: int = 32
    block_stride_px: int = 16
    seam_support_radius_px: int = 6
    feature_halo_radius_px: int = 12
    normal_search_maximum_px: float = 3.0
    normal_search_step_px: float = 0.5
    minimum_edge_component_length_px: int = 12
    minimum_edge_support_count: int = 12
    minimum_edge_correlation: float = 0.65
    minimum_uniqueness_fraction: float = 0.10
    maximum_orientation_difference_degrees: float = 10.0
    suspect_lk_p95_px: float = 1.25
    suspect_edge_median_px: float = 0.75
    suspect_edge_p95_px: float = 1.0
    equivalent_edge_p95_tolerance_px: float = 0.15
    micro_rescue_enabled: bool = False

    def __post_init__(self) -> None:
        if not np.isfinite(self.lk_error_sigma_multiplier) or self.lk_error_sigma_multiplier < 0.0:
            raise ValueError("S1.3 M5.1-r2 LK error sigma multiplier is invalid")
        if self.minimum_correspondence_count_for_error_filter < 1:
            raise ValueError("S1.3 M5.1-r2 LK error filter count is invalid")
        for name in ("lk_error_mad_floor", "lk_error_absolute_maximum"):
            value = getattr(self, name)
            if value is not None and (not np.isfinite(value) or value < 0.0):
                raise ValueError(f"S1.3 M5.1-r2 {name} is invalid")
        if self.enabled and (
            self.lk_error_mad_floor is None or self.lk_error_absolute_maximum is None
        ):
            raise ValueError(
                "S1.3 M5.1-r2 enabled LK filtering requires frozen error limits"
            )
        widths = tuple(int(value) for value in self.evidence_half_widths_px)
        if widths != (16, 24):
            raise ValueError("S1.3 M5.1-r2 seam evidence widths must be exactly (16, 24)")
        if self.moving_evidence_margin_px < 0:
            raise ValueError("S1.3 M5.1-r2 moving evidence margin is invalid")


__all__ = ["S13M51R2Config"]
