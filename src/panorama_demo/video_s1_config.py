from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


S1_CONFIG_SCHEMA = "gemini305-video-s1-experiment/v1"
S1_ALGORITHM_ID = "S01_output_first_vertical_alignment_v1"


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


def load_video_s1_config(path: str | Path) -> VideoS1ExperimentConfig:
    config_path = Path(path)
    document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(document, Mapping):
        raise ValueError("S1 experiment config must be a mapping")
    return VideoS1ExperimentConfig.from_mapping(document)


def load_s1_config(path: str | Path) -> VideoS1ExperimentConfig:
    return load_video_s1_config(path)
