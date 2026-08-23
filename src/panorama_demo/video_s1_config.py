from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


S1_CONFIG_SCHEMA = "gemini305-video-s1-experiment/v1"
S1_ALGORITHM_ID = "S01_output_first_vertical_alignment_v1"
S11_CONFIG_SCHEMA = "gemini305-video-s1-experiment/v2"
S11_ALGORITHM_ID = "S011_object_safe_straight_handoff_v1"
S11_REPORT_SCHEMA = "gemini305-video-s1-experiment-report/v2"


def _section(document: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = document.get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _positive(name: str, value: float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class S1ScanConfig:
    analysis_width: int = 320
    motion_backend: str = "lk_ransac"
    fallback_motion_backend: str = "phase_correlation"
    maximum_workers: int = 4

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "S1ScanConfig":
        result = cls(**values)
        _positive("scan.analysis_width", result.analysis_width)
        _positive("scan.maximum_workers", result.maximum_workers)
        if result.motion_backend != "lk_ransac":
            raise ValueError("scan.motion_backend must be lk_ransac")
        if result.fallback_motion_backend != "phase_correlation":
            raise ValueError("scan.fallback_motion_backend must be phase_correlation")
        return result


@dataclass(frozen=True)
class S1SourceSelectionConfig:
    normal_target_step_px: float = 12.0
    risk_target_step_px: float = 8.0
    maximum_preferred_step_px: float = 18.0
    emergency_step_px: float = 24.0
    target_common_overlap_px: int = 128
    minimum_s1_overlap_px: int = 64
    shoulder_target_px: int = 64
    shoulder_minimum_px: int = 32
    shoulder_maximum_px: int = 96
    adjacent_frame_rescue: bool = True

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "S1SourceSelectionConfig":
        result = cls(**values)
        for name in (
            "normal_target_step_px", "risk_target_step_px", "maximum_preferred_step_px",
            "emergency_step_px", "target_common_overlap_px", "minimum_s1_overlap_px",
            "shoulder_target_px", "shoulder_minimum_px", "shoulder_maximum_px",
        ):
            _positive(f"source_selection.{name}", float(getattr(result, name)))
        if not (
            result.risk_target_step_px <= result.normal_target_step_px
            <= result.maximum_preferred_step_px <= result.emergency_step_px
        ):
            raise ValueError(
                "source_selection steps must satisfy risk <= normal <= maximum_preferred <= emergency"
            )
        if not result.shoulder_minimum_px <= result.shoulder_target_px <= result.shoulder_maximum_px:
            raise ValueError("source_selection shoulders must satisfy minimum <= target <= maximum")
        if result.minimum_s1_overlap_px > result.target_common_overlap_px:
            raise ValueError("minimum_s1_overlap_px cannot exceed target_common_overlap_px")
        return result


@dataclass(frozen=True)
class S1BaseRenderConfig:
    transition_width_px: int = 1
    fill_invalid_owner_from_adjacent_real_source: bool = True
    synthetic_hole_fill: bool = False

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "S1BaseRenderConfig":
        result = cls(**values)
        if result.transition_width_px not in (1, 2):
            raise ValueError("s0.transition_width_px must be 1 or 2")
        if result.synthetic_hole_fill:
            raise ValueError("s0.synthetic_hole_fill must remain false")
        return result


@dataclass(frozen=True)
class S1AlignmentConfig:
    horizontal_scale: float = 0.5
    preserve_full_vertical_resolution: bool = True
    window_height_px: int = 32
    window_step_px: int = 16
    minimum_window_height_px: int = 16
    maximum_window_height_px: int = 48
    horizontal_subwindow_width_px: int = 80
    horizontal_subwindow_count: int = 3
    minimum_horizontal_subwindow_width_px: int = 48
    search_radius_y_px: int = 16
    top_k_candidates: int = 5
    candidate_nms_radius_px: int = 2
    allow_subpixel_refinement: bool = True
    luminance_zncc_weight: float = 0.45
    vertical_gradient_zncc_weight: float = 0.40
    lab_color_weight: float = 0.15
    minimum_valid_fraction_for_candidate: float = 0.45
    minimum_2d_texture_score: float = 0.05
    left_right_sigma_px: float = 1.5
    candidate_margin_sigma: float = 0.08
    multi_x_consensus_sigma_px: float = 1.5
    use_ransac_soft_prior: bool = True
    ransac_minimum_points: int = 6
    ransac_residual_px: float = 2.0
    dp_first_order_weight: float = 1.0
    dp_second_order_weight: float = 0.5
    dp_missing_cost: float = 1.25
    dp_soft_prior_weight: float = 0.25
    smoothing_control_spacing_px: int = 80
    smoothing_data_weight: float = 1.0
    smoothing_first_order_weight: float = 0.25
    smoothing_second_order_weight: float = 2.0
    maximum_local_slope: float = 0.08
    gain_candidates: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0)
    gain_block_height_px: int = 64
    gain_smoothness_weight: float = 0.20
    warp_support_left_px: int = 64
    warp_support_right_px: int = 64
    symmetric_pair_warp: bool = True
    fade_function: str = "cosine"

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "S1AlignmentConfig":
        prepared = dict(values)
        if "gain_candidates" in prepared:
            prepared["gain_candidates"] = tuple(float(item) for item in prepared["gain_candidates"])
        result = cls(**prepared)
        for name in (
            "horizontal_scale", "window_height_px", "window_step_px", "minimum_window_height_px",
            "maximum_window_height_px", "horizontal_subwindow_width_px",
            "horizontal_subwindow_count", "minimum_horizontal_subwindow_width_px",
            "search_radius_y_px", "top_k_candidates", "candidate_nms_radius_px",
            "left_right_sigma_px", "multi_x_consensus_sigma_px", "ransac_minimum_points",
            "ransac_residual_px", "smoothing_control_spacing_px", "maximum_local_slope",
            "gain_block_height_px", "warp_support_left_px", "warp_support_right_px",
        ):
            _positive(f"s1.{name}", float(getattr(result, name)))
        if not 0 < result.horizontal_scale <= 1:
            raise ValueError("s1.horizontal_scale must be in (0, 1]")
        if not result.preserve_full_vertical_resolution:
            raise ValueError("s1.preserve_full_vertical_resolution must remain true")
        if not result.minimum_window_height_px <= result.window_height_px <= result.maximum_window_height_px:
            raise ValueError("s1 window height must be within its configured bounds")
        weights = result.luminance_zncc_weight + result.vertical_gradient_zncc_weight + result.lab_color_weight
        if abs(weights - 1.0) > 1e-6:
            raise ValueError("s1 candidate cost weights must sum to 1")
        for name in ("minimum_valid_fraction_for_candidate", "minimum_2d_texture_score"):
            value = float(getattr(result, name))
            if not 0 <= value <= 1:
                raise ValueError(f"s1.{name} must be in [0, 1]")
        if not result.gain_candidates or result.gain_candidates[0] != 0.0:
            raise ValueError("s1.gain_candidates must begin with the zero-correction fallback")
        if any(value < 0 or value > 1 for value in result.gain_candidates):
            raise ValueError("s1.gain_candidates must be in [0, 1]")
        if result.fade_function != "cosine":
            raise ValueError("s1.fade_function must be cosine")
        return result


@dataclass(frozen=True)
class S1OutputConfig:
    write_s0_panorama: bool = True
    write_s1_panorama: bool = True
    write_owner_map: bool = True
    write_pair_diagnostics: bool = True
    write_candidate_heatmaps: bool = True
    write_before_after_crops: bool = True
    write_performance_report: bool = True

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "S1OutputConfig":
        result = cls(**values)
        if not result.write_s0_panorama or not result.write_s1_panorama:
            raise ValueError("output must retain both S0 and S1 panoramas")
        return result


@dataclass(frozen=True)
class S11SourceSelectionConfig:
    initial_selection_policy: str = "inherit_s01_v1_exact"
    rescue_step_units: str = "full_resolution_px"
    minimum_rescue_improvement_px: float = 2.0
    maximum_refinement_rounds: int = 2
    maximum_insertions_per_original_pair: int = 2
    preserve_full_fov_endpoints: bool = True

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "S11SourceSelectionConfig":
        result = cls(**values)
        if result.initial_selection_policy != "inherit_s01_v1_exact":
            raise ValueError(
                "source_selection.initial_selection_policy must be inherit_s01_v1_exact"
            )
        if result.rescue_step_units != "full_resolution_px":
            raise ValueError("source_selection.rescue_step_units must be full_resolution_px")
        _positive(
            "source_selection.minimum_rescue_improvement_px",
            result.minimum_rescue_improvement_px,
        )
        if not 1 <= result.maximum_refinement_rounds <= 2:
            raise ValueError("source_selection.maximum_refinement_rounds must be in [1, 2]")
        if not 1 <= result.maximum_insertions_per_original_pair <= 2:
            raise ValueError(
                "source_selection.maximum_insertions_per_original_pair must be in [1, 2]"
            )
        if not result.preserve_full_fov_endpoints:
            raise ValueError("source_selection.preserve_full_fov_endpoints must remain true")
        return result


@dataclass(frozen=True)
class S11HandoffConfig:
    enabled: bool = True
    coordinate_domain: str = "calibrated_target_analysis"
    analysis_width_px: int = 424
    match_corridor_width_px: int = 128
    maximum_boundary_shift_px: int = 48
    minimum_internal_owner_width_px: int = 9
    minimum_match_count: int = 12
    minimum_protected_match_count: int = 4
    minimum_protected_vertical_span_px: int = 24
    cluster_neighbor_x_px: float = 24.0
    cluster_neighbor_y_px: float = 32.0
    cluster_relative_dx_tolerance_px: float = 1.0
    cluster_merge_gap_px: float = 2.0
    lk_forward_backward_max_px: float = 0.75
    maximum_vertical_match_error_px: float = 4.0
    protected_interval_min_width_px: float = 1.5
    minimum_background_relative_dx_px: float = 1.0
    conflict_guard_fullres_px: float = 3.0
    use_aligned_depth_for_risk: bool = True
    straight_seam_only: bool = True

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "S11HandoffConfig":
        result = cls(**values)
        if not result.enabled:
            raise ValueError("handoff.enabled must remain true")
        if result.coordinate_domain != "calibrated_target_analysis":
            raise ValueError("handoff.coordinate_domain must be calibrated_target_analysis")
        for name in (
            "analysis_width_px",
            "match_corridor_width_px",
            "maximum_boundary_shift_px",
            "minimum_internal_owner_width_px",
            "minimum_match_count",
            "minimum_protected_match_count",
            "minimum_protected_vertical_span_px",
            "cluster_neighbor_x_px",
            "cluster_neighbor_y_px",
            "cluster_relative_dx_tolerance_px",
            "cluster_merge_gap_px",
            "lk_forward_backward_max_px",
            "maximum_vertical_match_error_px",
            "protected_interval_min_width_px",
            "minimum_background_relative_dx_px",
            "conflict_guard_fullres_px",
        ):
            _positive(f"handoff.{name}", float(getattr(result, name)))
        if result.minimum_protected_match_count > result.minimum_match_count:
            raise ValueError("handoff protected match count cannot exceed minimum match count")
        if not result.straight_seam_only:
            raise ValueError("handoff.straight_seam_only must remain true")
        return result


@dataclass(frozen=True)
class S11OwnerConfig:
    protected_transition_width_px: int = 0
    safe_background_transition_width_px: int = 0
    allow_foreign_owner_fill: bool = False
    allow_adjacent_border_fill: bool = False

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "S11OwnerConfig":
        result = cls(**values)
        if result.protected_transition_width_px != 0:
            raise ValueError("owner.protected_transition_width_px must remain zero")
        if result.safe_background_transition_width_px != 0:
            raise ValueError("owner.safe_background_transition_width_px must remain zero")
        if result.allow_foreign_owner_fill:
            raise ValueError("owner.allow_foreign_owner_fill must remain false")
        if result.allow_adjacent_border_fill:
            raise ValueError("owner.allow_adjacent_border_fill must remain false")
        return result


@dataclass(frozen=True)
class S11AlignmentConfig:
    horizontal_scale: float = 0.5
    preserve_full_vertical_resolution: bool = True
    normal_coarse_search_radius_y_px: int = 6
    expanded_coarse_search_radius_y_px: int = 16
    coarse_window_height_px: int = 32
    coarse_window_step_px: int = 16
    fine_window_height_px: int = 24
    fine_window_step_px: int = 8
    fine_search_radius_y_px: float = 2.0
    fine_search_step_px: float = 0.25
    normal_top_k_candidates: int = 3
    expanded_top_k_candidates: int = 5
    candidate_nms_radius_px: int = 2
    minimum_independent_subwindow_agreement: int = 2
    maximum_subwindow_spread_px: float = 0.75
    trusted_row_taper_px: int = 8
    maximum_consecutive_missing_windows: int = 2
    fine_vertical_trigger_p95_px: float = 1.25
    warp_support_maximum_px: int = 16
    warp_support_owner_fraction: float = 0.45
    warp_support_minimum_px: int = 4
    protected_source_reference_warp: bool = True
    maximum_local_slope: float = 0.08
    fade_function: str = "cosine"

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "S11AlignmentConfig":
        result = cls(**values)
        for name in (
            "horizontal_scale",
            "normal_coarse_search_radius_y_px",
            "expanded_coarse_search_radius_y_px",
            "coarse_window_height_px",
            "coarse_window_step_px",
            "fine_window_height_px",
            "fine_window_step_px",
            "fine_search_radius_y_px",
            "fine_search_step_px",
            "normal_top_k_candidates",
            "expanded_top_k_candidates",
            "candidate_nms_radius_px",
            "minimum_independent_subwindow_agreement",
            "maximum_subwindow_spread_px",
            "trusted_row_taper_px",
            "fine_vertical_trigger_p95_px",
            "warp_support_maximum_px",
            "warp_support_minimum_px",
            "maximum_local_slope",
        ):
            _positive(f"s1.{name}", float(getattr(result, name)))
        if not 0 < result.horizontal_scale <= 1:
            raise ValueError("s1.horizontal_scale must be in (0, 1]")
        if not result.preserve_full_vertical_resolution:
            raise ValueError("s1.preserve_full_vertical_resolution must remain true")
        if result.normal_coarse_search_radius_y_px > result.expanded_coarse_search_radius_y_px:
            raise ValueError("s1 normal search radius cannot exceed expanded search radius")
        if result.normal_top_k_candidates > result.expanded_top_k_candidates:
            raise ValueError("s1 normal Top-K cannot exceed expanded Top-K")
        if result.maximum_consecutive_missing_windows < 0:
            raise ValueError("s1.maximum_consecutive_missing_windows must be non-negative")
        if result.warp_support_owner_fraction >= 0.5:
            raise ValueError("s1.warp_support_owner_fraction must be less than 0.5")
        if result.warp_support_minimum_px > result.warp_support_maximum_px:
            raise ValueError("s1 warp support minimum cannot exceed maximum")
        if not result.protected_source_reference_warp:
            raise ValueError("s1.protected_source_reference_warp must remain true")
        if result.fade_function != "cosine":
            raise ValueError("s1.fade_function must be cosine")
        return result


@dataclass(frozen=True)
class S11OutputConfig:
    mode: str = "audit"
    write_stage_a_nominal_midpoint: bool = True
    write_stage_b_handoff_only: bool = True
    write_stage_c_final: bool = True
    write_stage_provenance: bool = True
    write_pair_reports: bool = True
    write_acceptance_crops: bool = True
    write_performance_report: bool = True

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "S11OutputConfig":
        result = cls(**values)
        if result.mode not in {"minimal", "audit"}:
            raise ValueError("output.mode must be minimal or audit")
        if not (
            result.write_stage_a_nominal_midpoint
            and result.write_stage_b_handoff_only
            and result.write_stage_c_final
            and result.write_stage_provenance
        ):
            raise ValueError("output must retain Stage A/B/C and stage provenance")
        return result


@dataclass(frozen=True)
class VideoS1ExperimentConfig:
    schema: str
    algorithm_id: str
    scan: S1ScanConfig
    source_selection: S1SourceSelectionConfig
    s0: S1BaseRenderConfig
    s1: S1AlignmentConfig
    output: S1OutputConfig

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, document: Mapping[str, Any]) -> "VideoS1ExperimentConfig":
        if document.get("schema") != S1_CONFIG_SCHEMA:
            raise ValueError(f"schema must be {S1_CONFIG_SCHEMA}")
        if document.get("algorithm_id") != S1_ALGORITHM_ID:
            raise ValueError(f"algorithm_id must be {S1_ALGORITHM_ID}")
        return cls(
            schema=S1_CONFIG_SCHEMA,
            algorithm_id=S1_ALGORITHM_ID,
            scan=S1ScanConfig.from_mapping(_section(document, "scan")),
            source_selection=S1SourceSelectionConfig.from_mapping(_section(document, "source_selection")),
            s0=S1BaseRenderConfig.from_mapping(_section(document, "s0")),
            s1=S1AlignmentConfig.from_mapping(_section(document, "s1")),
            output=S1OutputConfig.from_mapping(_section(document, "output")),
        )


@dataclass(frozen=True)
class VideoS11ExperimentConfig:
    schema: str
    algorithm_id: str
    diagnostic_only: bool
    production_eligible: bool
    motion_role: str
    scan: S1ScanConfig
    source_selection: S11SourceSelectionConfig
    handoff: S11HandoffConfig
    owner: S11OwnerConfig
    s1: S11AlignmentConfig
    output: S11OutputConfig
    report_schema: str = S11_REPORT_SCHEMA

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, document: Mapping[str, Any]) -> "VideoS11ExperimentConfig":
        if document.get("schema") != S11_CONFIG_SCHEMA:
            raise ValueError(f"schema must be {S11_CONFIG_SCHEMA}")
        if document.get("algorithm_id") != S11_ALGORITHM_ID:
            raise ValueError(f"algorithm_id must be {S11_ALGORITHM_ID}")
        if document.get("diagnostic_only") is not True:
            raise ValueError("diagnostic_only must remain true")
        if document.get("production_eligible") is not False:
            raise ValueError("production_eligible must remain false")
        if document.get("motion_role") != "diagnostic_layout_evidence_only":
            raise ValueError(
                "motion_role must be diagnostic_layout_evidence_only"
            )
        source_selection = S11SourceSelectionConfig.from_mapping(
            _section(document, "source_selection")
        )
        handoff = S11HandoffConfig.from_mapping(_section(document, "handoff"))
        alignment = S11AlignmentConfig.from_mapping(_section(document, "s1"))
        if (
            handoff.minimum_internal_owner_width_px
            < 2 * alignment.warp_support_minimum_px + 1
        ):
            raise ValueError(
                "handoff.minimum_internal_owner_width_px must be at least "
                "2 * s1.warp_support_minimum_px + 1"
            )
        return cls(
            schema=S11_CONFIG_SCHEMA,
            algorithm_id=S11_ALGORITHM_ID,
            diagnostic_only=True,
            production_eligible=False,
            motion_role="diagnostic_layout_evidence_only",
            scan=S1ScanConfig.from_mapping(_section(document, "scan")),
            source_selection=source_selection,
            handoff=handoff,
            owner=S11OwnerConfig.from_mapping(_section(document, "owner")),
            s1=alignment,
            output=S11OutputConfig.from_mapping(_section(document, "output")),
        )


VideoS1Config = VideoS1ExperimentConfig | VideoS11ExperimentConfig


def load_video_s1_config(path: str | Path) -> VideoS1Config:
    config_path = Path(path)
    document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(document, Mapping):
        raise ValueError("S1 experiment config must be a mapping")
    if document.get("schema") == S11_CONFIG_SCHEMA:
        return VideoS11ExperimentConfig.from_mapping(document)
    return VideoS1ExperimentConfig.from_mapping(document)


def load_s1_config(path: str | Path) -> VideoS1Config:
    return load_video_s1_config(path)
