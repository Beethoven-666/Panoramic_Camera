"""Frozen configuration surface for the isolated S1.3 M5.1-r2 candidate.

The default is deliberately legacy-compatible.  LK statistics may be collected
while ``enabled`` is false, but no correspondence is then removed beyond the
historic status/finite checks.  The two LK error limits have no guessed
defaults: an enabled candidate must supply values frozen from instrumentation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

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
        if self.micro_rescue_enabled:
            raise ValueError("S1.3 M5.1-r2 does not implement or enable C2E micro rescue")


@dataclass(frozen=True)
class S13M51R3Config(S13M51R2Config):
    """Explicit v6 successor enabling component-local edge ambiguity."""

    component_local_ambiguity_enabled: bool = True
    component_cluster_maximum_normal_difference_degrees: float = 10.0
    component_cluster_maximum_lag_difference_px: float = 0.75
    component_cluster_maximum_centroid_distance_px: float = 32.0
    complete_reassessment_pair_indices: tuple[int, ...] = (71,)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.component_local_ambiguity_enabled is not True:
            raise ValueError("S1.3 M5.1-r3 requires component-local ambiguity")
        for name in (
            "component_cluster_maximum_normal_difference_degrees",
            "component_cluster_maximum_lag_difference_px",
            "component_cluster_maximum_centroid_distance_px",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"S1.3 M5.1-r3 {name} is invalid")
        indices = tuple(int(value) for value in self.complete_reassessment_pair_indices)
        if indices != tuple(sorted(set(indices))) or any(value < 0 for value in indices):
            raise ValueError("S1.3 M5.1-r3 reassessment pair indices are invalid")

    def requires_complete_seam_reassessment(self, pair_index: int) -> bool:
        """Return the diagnostic-only seam audit scope; never enables a warp."""

        return int(pair_index) in self.complete_reassessment_pair_indices


_COMPONENT_MATCH_WEIGHT_KEYS = (
    "y_overlap",
    "predicted_y",
    "endpoint_distance",
    "correlation",
    "uniqueness",
)


@dataclass(frozen=True)
class S13M51R4Config(S13M51R3Config):
    """Frozen v6-r2 component-chain C2E contract.

    The constructor validates the complete runtime surface.  Sequence and
    mapping inputs are copied to immutable canonical representations so a
    caller cannot mutate a validated contract after construction.
    """

    enabled: bool = True
    lk_error_mad_floor: float | None = 1.0
    lk_error_absolute_maximum: float | None = 64.0
    component_chain_c2e_enabled: bool = True
    application_policy: str = "quality_gated"

    minimum_chain_pair_count: int = 2
    minimum_application_segment_pair_count: int = 1
    maximum_pair_gap: int = 1
    split_on_ambiguous_or_unevaluable_observation: bool = True
    split_on_solver_outlier: bool = True
    minimum_component_y_overlap_fraction: float = 0.50
    maximum_component_normal_difference_degrees: float = 10.0
    maximum_component_endpoint_distance_px: float = 8.0
    maximum_predicted_y_disagreement_px: float = 6.0
    minimum_component_match_margin_fraction: float = 0.10
    component_match_weights: Mapping[str, float] = field(
        default_factory=lambda: {
            "y_overlap": 0.30,
            "predicted_y": 0.25,
            "endpoint_distance": 0.20,
            "correlation": 0.15,
            "uniqueness": 0.10,
        }
    )
    minimum_signed_gradient_agreement: float = 0.10

    normal_search_minimum_px: float = -3.0
    normal_search_maximum_px: float = 3.0
    normal_search_step_px: float = 0.5
    minimum_c2e_correlation: float = 0.75
    minimum_c2e_uniqueness_fraction: float = 0.10
    maximum_forward_reverse_discrepancy_px: float = 0.5

    source_offset_regularization: float = 0.05
    maximum_solver_condition_number: float = 1_000_000.0
    solver_huber_delta_px: float = 0.75
    maximum_normalized_solver_residual: float = 3.0
    maximum_source_normal_offset_px: float = 3.0
    correction_gain_candidates: tuple[float, ...] = (1.0, 0.75, 0.5)

    edge_core_radius_px: int = 2
    normal_taper_radius_px: int = 8
    endpoint_taper_px: int = 12

    maximum_post_edge_p95_px: float = 1.0
    maximum_post_edge_step_px: float = 1.5
    minimum_evaluable_edge_columns: int = 12
    minimum_evaluable_transition_fraction: float = 0.50
    minimum_resolved_improvement_px: float = 0.10
    minimum_absolute_improvement_px: float = 0.5
    minimum_relative_improvement_fraction: float = 0.30
    minimum_break_length_reduction_fraction: float = 0.50
    maximum_non_target_p95_regression_px: float = 0.10
    maximum_non_target_step_regression_px: float = 0.25
    minimum_formal_owner_support_retention: float = 1.0
    minimum_evidence_sample_retention: float = 0.98
    minimum_jacobian: float = 0.5
    maximum_combined_map_displacement_px: float = 8.0
    maximum_halo_regression_px: float = 0.25

    allow_partial_application: bool = True
    maximum_segment_split_depth: int = 4
    maximum_candidates_per_segment: int = 3
    maximum_total_roi_candidate_pixels: int = 1_000_000
    maximum_exact_conflict_group_nodes: int = 12

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.application_policy not in {
            "quality_gated",
            "all_structurally_safe",
        }:
            raise ValueError("S1.3 M5.1-r4 application policy is invalid")
        object.__setattr__(
            self,
            "evidence_half_widths_px",
            tuple(int(value) for value in self.evidence_half_widths_px),
        )
        object.__setattr__(
            self,
            "complete_reassessment_pair_indices",
            tuple(int(value) for value in self.complete_reassessment_pair_indices),
        )
        if self.enabled is not True or self.component_chain_c2e_enabled is not True:
            raise ValueError("S1.3 M5.1-r4 component-chain C2E must be enabled")
        for name in (
            "split_on_ambiguous_or_unevaluable_observation",
            "split_on_solver_outlier",
            "allow_partial_application",
        ):
            if getattr(self, name) is not True:
                raise ValueError(f"S1.3 M5.1-r4 {name} must be enabled")

        integer_minima = {
            "minimum_chain_pair_count": 2,
            "minimum_application_segment_pair_count": 1,
            "maximum_pair_gap": 1,
            "minimum_evaluable_edge_columns": 1,
            "maximum_segment_split_depth": 0,
            "maximum_candidates_per_segment": 1,
            "maximum_total_roi_candidate_pixels": 1,
            "maximum_exact_conflict_group_nodes": 1,
        }
        for name, minimum in integer_minima.items():
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"S1.3 M5.1-r4 {name} must be an integer")
            if int(value) < minimum:
                raise ValueError(f"S1.3 M5.1-r4 {name} is out of range")
        if self.minimum_application_segment_pair_count > self.minimum_chain_pair_count:
            raise ValueError("S1.3 M5.1-r4 application segment cannot exceed chain minimum")

        for name in (
            "edge_core_radius_px",
            "normal_taper_radius_px",
            "endpoint_taper_px",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < 0:
                raise ValueError(f"S1.3 M5.1-r4 {name} is invalid")
        if self.normal_taper_radius_px < self.edge_core_radius_px:
            raise ValueError("S1.3 M5.1-r4 normal taper must contain the edge core")

        unit_interval = (
            "minimum_component_y_overlap_fraction",
            "minimum_component_match_margin_fraction",
            "minimum_c2e_correlation",
            "minimum_c2e_uniqueness_fraction",
            "minimum_evaluable_transition_fraction",
            "minimum_relative_improvement_fraction",
            "minimum_break_length_reduction_fraction",
            "minimum_evidence_sample_retention",
        )
        for name in unit_interval:
            value = float(getattr(self, name))
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"S1.3 M5.1-r4 {name} must be in [0, 1]")
        if (
            not np.isfinite(self.minimum_formal_owner_support_retention)
            or float(self.minimum_formal_owner_support_retention) != 1.0
        ):
            raise ValueError("S1.3 M5.1-r4 formal owner support retention must be exactly 1.0")
        if (
            not np.isfinite(self.minimum_signed_gradient_agreement)
            or not -1.0 <= float(self.minimum_signed_gradient_agreement) <= 1.0
        ):
            raise ValueError("S1.3 M5.1-r4 signed gradient agreement is invalid")

        positive = (
            "maximum_component_normal_difference_degrees",
            "maximum_component_endpoint_distance_px",
            "maximum_predicted_y_disagreement_px",
            "normal_search_step_px",
            "maximum_solver_condition_number",
            "solver_huber_delta_px",
            "maximum_normalized_solver_residual",
            "maximum_source_normal_offset_px",
            "maximum_post_edge_p95_px",
            "maximum_post_edge_step_px",
            "minimum_jacobian",
            "maximum_combined_map_displacement_px",
        )
        for name in positive:
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"S1.3 M5.1-r4 {name} must be finite and positive")
        if self.maximum_component_normal_difference_degrees > 180.0:
            raise ValueError("S1.3 M5.1-r4 component normal difference exceeds 180 degrees")

        nonnegative = (
            "maximum_forward_reverse_discrepancy_px",
            "source_offset_regularization",
            "minimum_resolved_improvement_px",
            "minimum_absolute_improvement_px",
            "maximum_non_target_p95_regression_px",
            "maximum_non_target_step_regression_px",
            "maximum_halo_regression_px",
        )
        for name in nonnegative:
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"S1.3 M5.1-r4 {name} must be finite and non-negative")

        weights = dict(self.component_match_weights)
        if tuple(sorted(weights)) != tuple(sorted(_COMPONENT_MATCH_WEIGHT_KEYS)):
            raise ValueError("S1.3 M5.1-r4 component match weight keys are invalid")
        normalized_weights: dict[str, float] = {}
        for name in _COMPONENT_MATCH_WEIGHT_KEYS:
            value = float(weights[name])
            if not np.isfinite(value) or value < 0.0:
                raise ValueError("S1.3 M5.1-r4 component match weights must be non-negative")
            normalized_weights[name] = value
        if abs(sum(normalized_weights.values()) - 1.0) > 1e-9:
            raise ValueError("S1.3 M5.1-r4 component match weights must sum to 1")
        object.__setattr__(
            self,
            "component_match_weights",
            MappingProxyType(normalized_weights),
        )

        search_minimum = float(self.normal_search_minimum_px)
        search_maximum = float(self.normal_search_maximum_px)
        search_step = float(self.normal_search_step_px)
        if not np.isfinite(search_minimum) or not np.isfinite(search_maximum):
            raise ValueError("S1.3 M5.1-r4 normal search endpoints must be finite")
        if search_minimum >= search_maximum:
            raise ValueError("S1.3 M5.1-r4 normal search endpoints are unordered")
        interval_count = (search_maximum - search_minimum) / search_step
        rounded_intervals = round(interval_count)
        if abs(interval_count - rounded_intervals) > 1e-9 or rounded_intervals + 1 != 13:
            raise ValueError("S1.3 M5.1-r4 normal search must produce exactly 13 states")

        gains = tuple(float(value) for value in self.correction_gain_candidates)
        if (
            not gains
            or len(gains) > self.maximum_candidates_per_segment
            or any(not np.isfinite(value) or not 0.0 < value <= 1.0 for value in gains)
            or tuple(sorted(set(gains), reverse=True)) != gains
            or gains != (1.0, 0.75, 0.5)
        ):
            raise ValueError("S1.3 M5.1-r4 correction gains must be (1.0, 0.75, 0.5)")
        object.__setattr__(self, "correction_gain_candidates", gains)

        if self.maximum_post_edge_p95_px > self.maximum_post_edge_step_px:
            raise ValueError("S1.3 M5.1-r4 post-edge P95 exceeds maximum step")
        if self.maximum_source_normal_offset_px > max(abs(search_minimum), abs(search_maximum)):
            raise ValueError("S1.3 M5.1-r4 source offset exceeds normal search range")
        if self.maximum_combined_map_displacement_px < self.maximum_source_normal_offset_px:
            raise ValueError("S1.3 M5.1-r4 combined displacement is below source offset")

    @property
    def normal_search_states_px(self) -> tuple[float, ...]:
        count = int(
            round(
                (self.normal_search_maximum_px - self.normal_search_minimum_px)
                / self.normal_search_step_px
            )
        )
        return tuple(
            float(self.normal_search_minimum_px + index * self.normal_search_step_px)
            for index in range(count + 1)
        )


__all__ = ["S13M51R2Config", "S13M51R3Config", "S13M51R4Config"]
