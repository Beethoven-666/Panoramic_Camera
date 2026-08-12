"""Strict bootstrap configuration for the isolated S013 M6.1 candidate.

This module deliberately does not create candidate or threshold assets.  It
only validates already-written bootstrap evidence and, after an external
approval witness exists, builds the immutable effective configuration used by
the renderer and its auditors.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import yaml


M61_ALGORITHM_ID = "S013_output_first_progressive_dense_central_slit_m61_v2"
M61_IMPLEMENTATION_ID = "s013_output_first_progressive_dense_central_slit_m61_v2_preview"
M61_CONTRACT_SCHEMA = "gemini305-video-s13-output-first/v4"
M61_P2_COMPLETION_SCHEMA = "gemini305-video-s13-p2-completion/v4"
M61_P2_PHOTOMETRIC_REPLAY_SCHEMA = "gemini305-video-s13-p2-photometric-replay/v1"
M61_GRAPH_TOPOLOGY_SCHEMA = "gemini305-video-s13-p2-photometric-graph-topology/v1"
M61_PHOTOMETRIC_SOLUTION_SCHEMA = "gemini305-video-s13-photometric-solution/v2"
M61_PHOTOMETRIC_TRANSACTION_SCHEMA = "gemini305-video-s13-photometric-transaction/v2"
M61_PHOTOMETRIC_PAIR_EVIDENCE_SCHEMA = "gemini305-video-s13-photometric-pair-evidence/v2"
M61_LUMINANCE_FIELD_SCHEMA = "gemini305-video-s13-source-u-luminance-field/v1"
M61_BLEND_TRANSACTION_SCHEMA = "gemini305-video-s13-blend-transaction/v2"
M61_BLEND_MANIFEST_SCHEMA = "gemini305-video-s13-blend-transactions/v2"
M61_BLEND_MASKS_SCHEMA = "gemini305-video-s13-blend-pair-masks/v1"
M61_EFFECTIVE_CONFIG_SCHEMA = "gemini305-video-s13-m61-effective-config/v2"
M61_QUALITY_THRESHOLDS_SCHEMA = "gemini305-video-s13-m61-quality-thresholds/v2"
M61_VISUAL_QUALITY_SCHEMA = "gemini305-video-s13-p3-diagnostic-quality/v2"
M61_P3_HARD_AUDIT_SCHEMA = "gemini305-video-s13-p3-hard-audit/v2"
M61_PERFORMANCE_SCHEMA = "gemini305-video-s13-p3-performance/v2"
M61_P3_COMPLETION_SCHEMA = "gemini305-video-s13-p3-visual-completion/v2"
M61_DIAGNOSTICS_MANIFEST_SCHEMA = "gemini305-video-s13-m61-diagnostics-manifest/v1"
M61_MANUAL_REVIEW_MANIFEST_SCHEMA = "gemini305-video-s13-m61-manual-review-manifest/v1"
M61_MANUAL_REVIEW_DECISION_SCHEMA = "gemini305-video-s13-m61-manual-review-decision/v1"
M61_ACCEPTANCE_SUMMARY_SCHEMA = "gemini305-video-s13-m61-acceptance-summary/v1"
M61_ACCEPTANCE_SUMMARY_MANIFEST_SCHEMA = (
    "gemini305-video-s13-m61-acceptance-summary-manifest/v1"
)
M61_THRESHOLD_APPROVAL_SCHEMA = "gemini305-video-s13-m61-threshold-approval/v1"
M61_EVIDENCE_CONFIG_SCHEMA = "gemini305-video-s13-m61-photometric-evidence-config/v1"

M61_SCHEMA_REGISTRY = MappingProxyType({
    "contract": M61_CONTRACT_SCHEMA,
    "p2_completion": M61_P2_COMPLETION_SCHEMA,
    "p2_photometric_replay": M61_P2_PHOTOMETRIC_REPLAY_SCHEMA,
    "p2_photometric_graph_topology": M61_GRAPH_TOPOLOGY_SCHEMA,
    "photometric_solution": M61_PHOTOMETRIC_SOLUTION_SCHEMA,
    "photometric_transaction": M61_PHOTOMETRIC_TRANSACTION_SCHEMA,
    "photometric_pair_evidence": M61_PHOTOMETRIC_PAIR_EVIDENCE_SCHEMA,
    "luminance_field_disabled_contract": M61_LUMINANCE_FIELD_SCHEMA,
    "blend_transaction": M61_BLEND_TRANSACTION_SCHEMA,
    "blend_transactions_manifest": M61_BLEND_MANIFEST_SCHEMA,
    "blend_pair_masks_manifest": M61_BLEND_MASKS_SCHEMA,
    "effective_config": M61_EFFECTIVE_CONFIG_SCHEMA,
    "quality_thresholds": M61_QUALITY_THRESHOLDS_SCHEMA,
    "visual_quality": M61_VISUAL_QUALITY_SCHEMA,
    "p3_hard_audit": M61_P3_HARD_AUDIT_SCHEMA,
    "performance": M61_PERFORMANCE_SCHEMA,
    "p3_completion": M61_P3_COMPLETION_SCHEMA,
    "m61_diagnostics_manifest": M61_DIAGNOSTICS_MANIFEST_SCHEMA,
    "m61_manual_review_manifest": M61_MANUAL_REVIEW_MANIFEST_SCHEMA,
    "m61_manual_review_decision": M61_MANUAL_REVIEW_DECISION_SCHEMA,
    "m61_acceptance_summary": M61_ACCEPTANCE_SUMMARY_SCHEMA,
    "m61_acceptance_summary_manifest": M61_ACCEPTANCE_SUMMARY_MANIFEST_SCHEMA,
    "m61_threshold_approval": M61_THRESHOLD_APPROVAL_SCHEMA,
})


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    _validate_json_value(value, "config")
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")


def canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _validate_json_value(value: object, name: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"S013 M6.1 {name} must be finite")
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _validate_json_value(child, f"{name}[{index}]")
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str) or not key:
                raise ValueError(f"S013 M6.1 {name} keys must be non-empty strings")
            _validate_json_value(child, f"{name}.{key}")
        return
    raise ValueError(f"S013 M6.1 {name} is not canonical JSON data")


def _exact_keys(value: Mapping[str, Any], allowed: set[str], name: str) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"S013 M6.1 {name} has unknown keys: {sorted(unknown)}")


def _integer(value: object, name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"S013 M6.1 {name} must be an integer >= {minimum}")
    return value


def _fraction(value: object, name: str, *, include_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"S013 M6.1 {name} must be a finite fraction")
    result = float(value)
    lower_ok = result >= 0.0 if include_zero else result > 0.0
    if not math.isfinite(result) or not lower_ok or result >= 1.0:
        raise ValueError(f"S013 M6.1 {name} must be in {'[0, 1)' if include_zero else '(0, 1)'}")
    return result


def _positive(value: object, name: str, *, include_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"S013 M6.1 {name} must be finite")
    result = float(value)
    lower_ok = result >= 0.0 if include_zero else result > 0.0
    if not math.isfinite(result) or not lower_ok:
        raise ValueError(f"S013 M6.1 {name} must be finite and non-negative")
    return result


def _sha(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"S013 M6.1 {name} must be a SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"S013 M6.1 {name} must be a SHA-256 digest") from exc
    return value.lower()


@dataclass(frozen=True)
class S13PhotometricEvidenceConfig:
    schema: str = M61_EVIDENCE_CONFIG_SCHEMA
    diagnostic_shoulder_per_side_px: int = 64
    tile_size_px: int = 16
    guard_halo_tiles: int = 1
    split_seed: int = 20260804
    train_fraction: float = 0.75
    minimum_pair_validation_samples: int = 128
    minimum_independent_validation_tiles: int = 8
    minimum_vertical_blocks: int = 4
    maximum_samples_per_pair: int = 8192
    minimum_solver_pair_fraction: float = 0.80
    maximum_solver_insufficient_pair_fraction: float = 0.20
    require_bilateral_audited_support: bool = True
    graph_topology_version: str = "train-only-v1"
    validation_modulus: int = 5
    safe_gradient_max: float = 18.0
    protected_gradient_min: float = 32.0
    gradient_direction_cosine_min: float = -0.25
    clipping_high_u8: int = 250
    neutral_chroma_max: float = 12.0
    topology_minimum_train_samples: int = 64
    topology_minimum_train_tiles: int = 4
    topology_minimum_vertical_blocks: int = 4
    vertical_block_px: int = 64

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "S13PhotometricEvidenceConfig":
        if not isinstance(value, Mapping):
            raise ValueError("S013 M6.1 photometric evidence config must be a mapping")
        allowed = set(cls.__dataclass_fields__)
        _exact_keys(value, allowed, "photometric evidence config")
        merged = {**asdict(cls()), **dict(value)}
        if merged["schema"] != M61_EVIDENCE_CONFIG_SCHEMA:
            raise ValueError("S013 M6.1 photometric evidence config schema is invalid")
        if merged["diagnostic_shoulder_per_side_px"] != 64:
            raise ValueError("S013 M6.1 diagnostic shoulder is fixed at 64 px per side")
        if merged["tile_size_px"] != 16:
            raise ValueError("S013 M6.1 canvas split tile is fixed at 16 px")
        if merged["require_bilateral_audited_support"] is not True:
            raise ValueError("S013 M6.1 requires bilateral audited support")
        topology = merged["graph_topology_version"]
        if not isinstance(topology, str) or not topology.strip():
            raise ValueError("S013 M6.1 graph topology version is invalid")
        config = cls(
            schema=M61_EVIDENCE_CONFIG_SCHEMA,
            diagnostic_shoulder_per_side_px=64,
            tile_size_px=16,
            guard_halo_tiles=_integer(merged["guard_halo_tiles"], "guard_halo_tiles", minimum=1),
            split_seed=_integer(merged["split_seed"], "split_seed", minimum=0),
            train_fraction=_fraction(merged["train_fraction"], "train_fraction"),
            minimum_pair_validation_samples=_integer(
                merged["minimum_pair_validation_samples"], "minimum_pair_validation_samples", minimum=1
            ),
            minimum_independent_validation_tiles=_integer(
                merged["minimum_independent_validation_tiles"], "minimum_independent_validation_tiles", minimum=1
            ),
            minimum_vertical_blocks=_integer(
                merged["minimum_vertical_blocks"], "minimum_vertical_blocks", minimum=1
            ),
            maximum_samples_per_pair=_integer(
                merged["maximum_samples_per_pair"], "maximum_samples_per_pair", minimum=1
            ),
            minimum_solver_pair_fraction=_fraction(
                merged["minimum_solver_pair_fraction"], "minimum_solver_pair_fraction", include_zero=True
            ),
            maximum_solver_insufficient_pair_fraction=_fraction(
                merged["maximum_solver_insufficient_pair_fraction"],
                "maximum_solver_insufficient_pair_fraction",
                include_zero=True,
            ),
            require_bilateral_audited_support=True,
            graph_topology_version=topology,
            validation_modulus=_integer(merged["validation_modulus"], "validation_modulus", minimum=4),
            safe_gradient_max=float(merged["safe_gradient_max"]),
            protected_gradient_min=float(merged["protected_gradient_min"]),
            gradient_direction_cosine_min=float(merged["gradient_direction_cosine_min"]),
            clipping_high_u8=_integer(merged["clipping_high_u8"], "clipping_high_u8", minimum=1),
            neutral_chroma_max=float(merged["neutral_chroma_max"]),
            topology_minimum_train_samples=_integer(
                merged["topology_minimum_train_samples"], "topology_minimum_train_samples", minimum=1
            ),
            topology_minimum_train_tiles=_integer(
                merged["topology_minimum_train_tiles"], "topology_minimum_train_tiles", minimum=1
            ),
            topology_minimum_vertical_blocks=_integer(
                merged["topology_minimum_vertical_blocks"], "topology_minimum_vertical_blocks", minimum=1
            ),
            vertical_block_px=_integer(merged["vertical_block_px"], "vertical_block_px", minimum=1),
        )
        if config.minimum_solver_pair_fraction + config.maximum_solver_insufficient_pair_fraction < 1.0:
            raise ValueError("S013 M6.1 solver coverage fractions leave an unaudited gap")
        if config.maximum_samples_per_pair < config.minimum_pair_validation_samples:
            raise ValueError("S013 M6.1 maximum samples is below the validation minimum")
        if not all(math.isfinite(value) for value in (
            config.safe_gradient_max, config.protected_gradient_min,
            config.gradient_direction_cosine_min, config.neutral_chroma_max,
        )):
            raise ValueError("S013 M6.1 evidence thresholds must be finite")
        if not 0.0 <= config.safe_gradient_max < config.protected_gradient_min:
            raise ValueError("S013 M6.1 safe/protected gradient thresholds conflict")
        if not -1.0 <= config.gradient_direction_cosine_min <= 1.0:
            raise ValueError("S013 M6.1 gradient direction cosine is invalid")
        if config.clipping_high_u8 > 255:
            raise ValueError("S013 M6.1 clipping high value exceeds uint8")
        return config

    @property
    def canonical_sha256(self) -> str:
        return canonical_sha256(asdict(self))


@dataclass(frozen=True)
class S13LuminanceFieldDisabledConfig:
    schema: str = M61_LUMINANCE_FIELD_SCHEMA
    enabled: bool = False
    solver: str = "disabled"
    canonical_field_value: float = 0.0

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "S13LuminanceFieldDisabledConfig":
        if not isinstance(value, Mapping):
            raise ValueError("S013 M6.1 luminance-field config must be a mapping")
        _exact_keys(value, set(cls.__dataclass_fields__), "luminance-field config")
        merged = {**asdict(cls()), **dict(value)}
        if (
            merged["schema"] != M61_LUMINANCE_FIELD_SCHEMA
            or merged["enabled"] is not False
            or merged["solver"] != "disabled"
            or isinstance(merged["canonical_field_value"], bool)
            or not isinstance(merged["canonical_field_value"], (int, float))
            or not math.isfinite(float(merged["canonical_field_value"]))
            or float(merged["canonical_field_value"]) != 0.0
        ):
            raise ValueError("S013 M6.1 only supports the disabled zero luminance field")
        return cls()


@dataclass(frozen=True)
class S13PhotometricSolverConfig:
    huber_delta: float = 1.5
    irls_iterations: int = 8
    convergence_tolerance: float = 1e-6
    rank_tolerance: float = 1e-8
    maximum_condition_number: float = 1e8
    minimum_robust_scale_linear: float = 1e-4
    minimum_inlier_fraction: float = 0.50
    identity_regularization: float = 0.01
    first_difference_regularization: float = 0.05
    second_difference_regularization: float = 0.02
    minimum_gain: float = 0.80
    maximum_gain: float = 1.25
    maximum_absolute_bias_linear: float = 0.03
    canonical_identity_epsilon: float = 1e-4

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "S13PhotometricSolverConfig":
        if not isinstance(value, Mapping):
            raise ValueError("S013 M6.1 solver config must be a mapping")
        _exact_keys(value, set(cls.__dataclass_fields__), "solver config")
        merged = {**asdict(cls()), **dict(value)}
        result = cls(
            huber_delta=_positive(merged["huber_delta"], "huber_delta"),
            irls_iterations=_integer(merged["irls_iterations"], "irls_iterations", minimum=1),
            convergence_tolerance=_positive(
                merged["convergence_tolerance"], "convergence_tolerance"
            ),
            rank_tolerance=_positive(merged["rank_tolerance"], "rank_tolerance"),
            maximum_condition_number=_positive(
                merged["maximum_condition_number"], "maximum_condition_number"
            ),
            minimum_robust_scale_linear=_positive(
                merged["minimum_robust_scale_linear"], "minimum_robust_scale_linear"
            ),
            minimum_inlier_fraction=_fraction(
                merged["minimum_inlier_fraction"], "minimum_inlier_fraction"
            ),
            identity_regularization=_positive(
                merged["identity_regularization"], "identity_regularization", include_zero=True
            ),
            first_difference_regularization=_positive(
                merged["first_difference_regularization"],
                "first_difference_regularization",
                include_zero=True,
            ),
            second_difference_regularization=_positive(
                merged["second_difference_regularization"],
                "second_difference_regularization",
                include_zero=True,
            ),
            minimum_gain=_positive(merged["minimum_gain"], "minimum_gain"),
            maximum_gain=_positive(merged["maximum_gain"], "maximum_gain"),
            maximum_absolute_bias_linear=_positive(
                merged["maximum_absolute_bias_linear"],
                "maximum_absolute_bias_linear",
                include_zero=True,
            ),
            canonical_identity_epsilon=_positive(
                merged["canonical_identity_epsilon"], "canonical_identity_epsilon"
            ),
        )
        if result.minimum_gain >= 1.0 or result.maximum_gain <= 1.0:
            raise ValueError("S013 M6.1 gain bounds must strictly contain identity")
        if result.minimum_gain >= result.maximum_gain:
            raise ValueError("S013 M6.1 gain bounds conflict")
        if result.rank_tolerance >= 1.0 or result.maximum_condition_number <= 1.0:
            raise ValueError("S013 M6.1 rank/condition limits are invalid")
        return result


@dataclass(frozen=True)
class S13PhotometricSelectionConfig:
    aggregate_p95_nonreg_relative_tolerance: float = 0.02
    aggregate_p95_nonreg_absolute_tolerance_linear: float = 0.0005
    worst_pair_p95_nonreg_relative_tolerance: float = 0.02
    worst_pair_p95_nonreg_absolute_tolerance_linear: float = 0.0005
    minimum_aggregate_median_benefit_fraction: float = 0.01
    minimum_actionable_macro_p95_benefit_fraction: float = 0.05
    minimum_composite_benefit_fraction: float = 0.03
    selection_score_mde_linear: float = 0.0005
    require_macro_and_micro_nonregression: bool = True

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "S13PhotometricSelectionConfig":
        if not isinstance(value, Mapping):
            raise ValueError("S013 M6.1 selection config must be a mapping")
        _exact_keys(value, set(cls.__dataclass_fields__), "selection config")
        merged = {**asdict(cls()), **dict(value)}
        result = cls(
            aggregate_p95_nonreg_relative_tolerance=_fraction(
                merged["aggregate_p95_nonreg_relative_tolerance"],
                "aggregate_p95_nonreg_relative_tolerance",
                include_zero=True,
            ),
            aggregate_p95_nonreg_absolute_tolerance_linear=_positive(
                merged["aggregate_p95_nonreg_absolute_tolerance_linear"],
                "aggregate_p95_nonreg_absolute_tolerance_linear",
                include_zero=True,
            ),
            worst_pair_p95_nonreg_relative_tolerance=_fraction(
                merged["worst_pair_p95_nonreg_relative_tolerance"],
                "worst_pair_p95_nonreg_relative_tolerance",
                include_zero=True,
            ),
            worst_pair_p95_nonreg_absolute_tolerance_linear=_positive(
                merged["worst_pair_p95_nonreg_absolute_tolerance_linear"],
                "worst_pair_p95_nonreg_absolute_tolerance_linear",
                include_zero=True,
            ),
            minimum_aggregate_median_benefit_fraction=_fraction(
                merged["minimum_aggregate_median_benefit_fraction"],
                "minimum_aggregate_median_benefit_fraction",
                include_zero=True,
            ),
            minimum_actionable_macro_p95_benefit_fraction=_fraction(
                merged["minimum_actionable_macro_p95_benefit_fraction"],
                "minimum_actionable_macro_p95_benefit_fraction",
                include_zero=True,
            ),
            minimum_composite_benefit_fraction=_fraction(
                merged["minimum_composite_benefit_fraction"],
                "minimum_composite_benefit_fraction",
                include_zero=True,
            ),
            selection_score_mde_linear=_positive(
                merged["selection_score_mde_linear"], "selection_score_mde_linear"
            ),
            require_macro_and_micro_nonregression=merged[
                "require_macro_and_micro_nonregression"
            ],
        )
        if result.require_macro_and_micro_nonregression is not True:
            raise ValueError("S013 M6.1 requires macro and micro non-regression")
        # This exact initial boundary is an acceptance invariant, not a tuneable
        # approximation inherited from the superseded downloaded v2 plan.
        if (
            result.aggregate_p95_nonreg_relative_tolerance != 0.02
            or result.aggregate_p95_nonreg_absolute_tolerance_linear != 0.0005
        ):
            raise ValueError("S013 M6.1 aggregate P95 boundary is fixed at 2% + 0.0005 linear")
        return result

    def aggregate_p95_nonregression_passes(self, before: float, after: float) -> bool:
        before_value = _positive(before, "before aggregate P95", include_zero=True)
        after_value = _positive(after, "after aggregate P95", include_zero=True)
        allowance = max(
            self.aggregate_p95_nonreg_relative_tolerance * before_value,
            self.aggregate_p95_nonreg_absolute_tolerance_linear,
        )
        return after_value <= before_value + allowance


@dataclass(frozen=True)
class S13BlendConfig:
    eligible_models: tuple[str, ...] = ("B0_owner_only", "B1_narrow_feather_2px")
    ineligible_models: tuple[str, ...] = (
        "B2_safe_masked_multiband", "B3", "B4"
    )
    b1_total_width_px: int = 2
    maximum_color_contributors_per_pixel: int = 2
    protected_structure_weight: float = 0.0
    minimum_immediate_seam_benefit_dn: float = 0.5
    minimum_immediate_seam_benefit_mde_dn: float = 0.25

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "S13BlendConfig":
        if not isinstance(value, Mapping):
            raise ValueError("S013 M6.1 blend config must be a mapping")
        _exact_keys(value, set(cls.__dataclass_fields__), "blend config")
        merged = {**asdict(cls()), **dict(value)}
        eligible = tuple(merged["eligible_models"])
        ineligible = tuple(merged["ineligible_models"])
        result = cls(
            eligible_models=eligible,
            ineligible_models=ineligible,
            b1_total_width_px=_integer(merged["b1_total_width_px"], "b1_total_width_px", minimum=1),
            maximum_color_contributors_per_pixel=_integer(
                merged["maximum_color_contributors_per_pixel"],
                "maximum_color_contributors_per_pixel",
                minimum=1,
            ),
            protected_structure_weight=_positive(
                merged["protected_structure_weight"], "protected_structure_weight", include_zero=True
            ),
            minimum_immediate_seam_benefit_dn=_positive(
                merged["minimum_immediate_seam_benefit_dn"],
                "minimum_immediate_seam_benefit_dn",
                include_zero=True,
            ),
            minimum_immediate_seam_benefit_mde_dn=_positive(
                merged["minimum_immediate_seam_benefit_mde_dn"],
                "minimum_immediate_seam_benefit_mde_dn",
                include_zero=True,
            ),
        )
        if result.eligible_models != ("B0_owner_only", "B1_narrow_feather_2px"):
            raise ValueError("S013 M6.1 permits only B0 and B1 2 px")
        if result.b1_total_width_px != 2 or result.maximum_color_contributors_per_pixel != 2:
            raise ValueError("S013 M6.1 B1 width/contributor contract is fixed at two")
        if result.protected_structure_weight != 0.0:
            raise ValueError("S013 M6.1 protected pixels must remain owner-only")
        if not {"B2_safe_masked_multiband", "B3", "B4"}.issubset(result.ineligible_models):
            raise ValueError("S013 M6.1 must mark B2-B4 ineligible")
        return result


@dataclass(frozen=True)
class S13VisualQualityConfig:
    color_metric: str = "CIEDE2000"
    gaussian_sigma_px: float = 1.0
    block_widths_px: tuple[int, ...] = (8, 16, 32, 64)
    owner_plateau_minimum_width_px: int = 16
    minimum_required_metric_coverage_fraction: float = 0.95
    maximum_quality_insufficient_safe_pair_fraction: float = 0.0
    low_gradient_minimum_reference_energy: float = 1e-6
    maximum_canvas_megapixels: float = 200.0
    maximum_resident_source_rois: int = 5
    maximum_live_array_bytes: int = 1_500_000_000
    maximum_process_rss_bytes: int = 2_000_000_000
    maximum_post_seconds_soft_sla: float = 60.0

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "S13VisualQualityConfig":
        if not isinstance(value, Mapping):
            raise ValueError("S013 M6.1 visual quality config must be a mapping")
        _exact_keys(value, set(cls.__dataclass_fields__), "visual quality config")
        merged = {**asdict(cls()), **dict(value)}
        widths_value = merged["block_widths_px"]
        if not isinstance(widths_value, (list, tuple)):
            raise ValueError("S013 M6.1 block widths must be a sequence")
        widths = tuple(_integer(item, "block_width", minimum=1) for item in widths_value)
        result = cls(
            color_metric=str(merged["color_metric"]),
            gaussian_sigma_px=_positive(merged["gaussian_sigma_px"], "gaussian_sigma_px"),
            block_widths_px=widths,
            owner_plateau_minimum_width_px=_integer(
                merged["owner_plateau_minimum_width_px"],
                "owner_plateau_minimum_width_px",
                minimum=1,
            ),
            minimum_required_metric_coverage_fraction=_fraction(
                merged["minimum_required_metric_coverage_fraction"],
                "minimum_required_metric_coverage_fraction",
            ),
            maximum_quality_insufficient_safe_pair_fraction=_fraction(
                merged["maximum_quality_insufficient_safe_pair_fraction"],
                "maximum_quality_insufficient_safe_pair_fraction",
                include_zero=True,
            ),
            low_gradient_minimum_reference_energy=_positive(
                merged["low_gradient_minimum_reference_energy"],
                "low_gradient_minimum_reference_energy",
            ),
            maximum_canvas_megapixels=_positive(
                merged["maximum_canvas_megapixels"], "maximum_canvas_megapixels"
            ),
            maximum_resident_source_rois=_integer(
                merged["maximum_resident_source_rois"], "maximum_resident_source_rois", minimum=1
            ),
            maximum_live_array_bytes=_integer(
                merged["maximum_live_array_bytes"], "maximum_live_array_bytes", minimum=1
            ),
            maximum_process_rss_bytes=_integer(
                merged["maximum_process_rss_bytes"], "maximum_process_rss_bytes", minimum=1
            ),
            maximum_post_seconds_soft_sla=_positive(
                merged["maximum_post_seconds_soft_sla"], "maximum_post_seconds_soft_sla"
            ),
        )
        if result.color_metric != "CIEDE2000" or result.block_widths_px != (8, 16, 32, 64):
            raise ValueError("S013 M6.1 color metric/block scales are fixed")
        if result.maximum_canvas_megapixels > 200.0 or result.maximum_resident_source_rois > 5:
            raise ValueError("S013 M6.1 resource caps exceed the authorized envelope")
        if result.maximum_process_rss_bytes < result.maximum_live_array_bytes:
            raise ValueError("S013 M6.1 RSS cap is below the live-array cap")
        return result


@dataclass(frozen=True)
class S13M61EffectiveConfig:
    schema: str
    algorithm_id: str
    implementation_id: str
    candidate_config_sha256: str
    photometric_evidence_config: S13PhotometricEvidenceConfig
    photometric_evidence_config_sha256: str
    quality_thresholds: Mapping[str, Any]
    quality_thresholds_sha256: str
    threshold_approval: Mapping[str, Any]
    threshold_approval_sha256: str
    solver: S13PhotometricSolverConfig
    selection: S13PhotometricSelectionConfig
    blend: S13BlendConfig
    visual_quality: S13VisualQualityConfig
    luminance_field: S13LuminanceFieldDisabledConfig
    effective_config_sha256: str

    def canonical_document(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "algorithm_id": self.algorithm_id,
            "implementation_id": self.implementation_id,
            "candidate_config_sha256": self.candidate_config_sha256,
            "photometric_evidence_config": asdict(self.photometric_evidence_config),
            "photometric_evidence_config_sha256": self.photometric_evidence_config_sha256,
            "quality_thresholds": dict(self.quality_thresholds),
            "quality_thresholds_sha256": self.quality_thresholds_sha256,
            "threshold_approval": dict(self.threshold_approval),
            "threshold_approval_sha256": self.threshold_approval_sha256,
            "solver": asdict(self.solver),
            "selection": asdict(self.selection),
            "blend": asdict(self.blend),
            "visual_quality": asdict(self.visual_quality),
            "luminance_field": asdict(self.luminance_field),
        }


def _load_mapping(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"S013 M6.1 {label} does not exist: {path}")
    try:
        if path.suffix.lower() in {".yaml", ".yml"}:
            value = yaml.safe_load(path.read_text(encoding="utf-8"))
        else:
            value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, yaml.YAMLError) as exc:
        raise ValueError(f"S013 M6.1 {label} is invalid") from exc
    if not isinstance(value, dict):
        raise ValueError(f"S013 M6.1 {label} must be a mapping")
    _validate_json_value(value, label)
    return value


def _confined(path: str | Path, root: Path, label: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute() or ".." in candidate.parts or candidate == Path("."):
        raise ValueError(f"S013 M6.1 {label} path is unsafe")
    resolved_root = root.resolve()
    resolved = (resolved_root / candidate).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"S013 M6.1 {label} escapes its config root") from exc
    return resolved


def _validate_threshold(value: Mapping[str, Any], evidence_sha: str) -> dict[str, Any]:
    required = {
        "schema", "status", "photometric_evidence_config_sha256", "mde",
        "metrics", "baseline", "actionability", "proposed_threshold_sha256",
        "threshold_approval_sha256",
    }
    _exact_keys(value, required, "quality thresholds")
    if value.get("schema") != M61_QUALITY_THRESHOLDS_SCHEMA or value.get("status") != "final":
        raise ValueError("S013 M6.1 requires a final v2 quality-threshold asset")
    if _sha(value.get("photometric_evidence_config_sha256"), "threshold evidence SHA") != evidence_sha:
        raise ValueError("S013 M6.1 threshold evidence config SHA disagrees")
    _sha(value.get("threshold_approval_sha256"), "threshold approval SHA")
    _sha(value.get("proposed_threshold_sha256"), "proposed threshold SHA")
    for name in ("mde", "metrics", "baseline", "actionability"):
        if not isinstance(value.get(name), Mapping):
            raise ValueError(f"S013 M6.1 threshold {name} must be a mapping")
    return dict(value)


def _validate_approval(value: Mapping[str, Any], proposed_sha: str | None = None) -> dict[str, Any]:
    required = {
        "schema", "proposed_threshold_sha256", "baseline_quality_manifest_sha256",
        "mde_calibration_sha256", "four_p2_parents", "decision", "approver",
        "approved_at", "reason",
    }
    _exact_keys(value, required, "threshold approval")
    if value.get("schema") != M61_THRESHOLD_APPROVAL_SCHEMA or value.get("decision") != "approved":
        raise ValueError("S013 M6.1 threshold approval is missing or rejected")
    for name in (
        "proposed_threshold_sha256", "baseline_quality_manifest_sha256", "mde_calibration_sha256"
    ):
        _sha(value.get(name), name)
    if proposed_sha is not None and value["proposed_threshold_sha256"] != proposed_sha:
        raise ValueError("S013 M6.1 approval does not bind the proposed threshold")
    if not isinstance(value.get("approver"), str) or not value["approver"].strip():
        raise ValueError("S013 M6.1 approval requires an approver")
    if not isinstance(value.get("approved_at"), str) or not value["approved_at"].strip():
        raise ValueError("S013 M6.1 approval requires a timestamp")
    if not isinstance(value.get("reason"), str):
        raise ValueError("S013 M6.1 approval reason must be a string")
    parents = value.get("four_p2_parents")
    if not isinstance(parents, list) or len(parents) != 4:
        raise ValueError("S013 M6.1 approval must bind exactly four P2 parents")
    expected_branches = ["fast_direct", "fast_ignore_pose", "slow_direct", "slow_ignore_pose"]
    branches: list[str] = []
    for parent in parents:
        if not isinstance(parent, Mapping):
            raise ValueError("S013 M6.1 approval P2 parent is invalid")
        _exact_keys(
            parent,
            {"branch", "run_id", "p2_completion_sha256", "p2_canonical_image_sha256"},
            "threshold approval P2 parent",
        )
        if not isinstance(parent.get("branch"), str) or not isinstance(parent.get("run_id"), str):
            raise ValueError("S013 M6.1 approval P2 identity is invalid")
        branches.append(str(parent["branch"]))
        _sha(parent.get("p2_completion_sha256"), "P2 completion SHA")
        _sha(parent.get("p2_canonical_image_sha256"), "P2 image SHA")
    if branches != expected_branches:
        raise ValueError("S013 M6.1 approval P2 parents are not canonically branch-sorted")
    return dict(value)


def load_s13_m61_effective_config(
    candidate_path: str | Path,
    *,
    photometric_evidence_config: Mapping[str, Any],
    config_root: str | Path | None = None,
) -> S13M61EffectiveConfig:
    """Load an approved final M6.1 config; proposed thresholds cannot pass."""

    candidate_file = Path(candidate_path).expanduser().resolve()
    candidate = _load_mapping(candidate_file, "candidate config")
    if (
        candidate.get("candidate_id") != M61_ALGORITHM_ID
        or candidate.get("algorithm_id") != M61_ALGORITHM_ID
        or candidate.get("implementation_id") != M61_IMPLEMENTATION_ID
    ):
        raise ValueError("S013 M6.1 candidate identity is not exact")
    declared_config_sha = _sha(candidate.get("config_sha256"), "candidate config SHA")
    candidate_for_hash = dict(candidate)
    candidate_for_hash.pop("config_sha256", None)
    if declared_config_sha != canonical_sha256(candidate_for_hash):
        raise ValueError("S013 M6.1 candidate config SHA disagrees")
    bootstrap = candidate.get("m61_bootstrap")
    if not isinstance(bootstrap, Mapping):
        raise ValueError("S013 M6.1 candidate has no bootstrap bindings")
    _exact_keys(
        bootstrap,
        {
            "quality_thresholds_path", "quality_thresholds_sha256",
            "threshold_approval_path", "threshold_approval_sha256",
            "photometric_evidence_config_sha256", "luminance_field",
            "solver", "selection", "blend", "visual_quality",
        },
        "candidate bootstrap",
    )
    evidence = S13PhotometricEvidenceConfig.from_mapping(photometric_evidence_config)
    evidence_sha = evidence.canonical_sha256
    if _sha(bootstrap.get("photometric_evidence_config_sha256"), "candidate evidence SHA") != evidence_sha:
        raise ValueError("S013 M6.1 candidate evidence config SHA disagrees")
    root = Path(config_root).expanduser().resolve() if config_root is not None else candidate_file.parent
    threshold_path = _confined(str(bootstrap.get("quality_thresholds_path", "")), root, "threshold")
    approval_path = _confined(str(bootstrap.get("threshold_approval_path", "")), root, "approval")
    threshold = _validate_threshold(_load_mapping(threshold_path, "quality thresholds"), evidence_sha)
    approval = _validate_approval(
        _load_mapping(approval_path, "threshold approval"),
        proposed_sha=str(threshold["proposed_threshold_sha256"]),
    )
    threshold_sha = canonical_sha256(threshold)
    approval_sha = canonical_sha256(approval)
    if _sha(bootstrap.get("quality_thresholds_sha256"), "candidate threshold SHA") != threshold_sha:
        raise ValueError("S013 M6.1 candidate threshold SHA disagrees")
    if _sha(bootstrap.get("threshold_approval_sha256"), "candidate approval SHA") != approval_sha:
        raise ValueError("S013 M6.1 candidate approval SHA disagrees")
    if threshold["threshold_approval_sha256"] != approval_sha:
        raise ValueError("S013 M6.1 final threshold does not bind its approval witness")
    luminance = S13LuminanceFieldDisabledConfig.from_mapping(
        bootstrap.get("luminance_field", {})
    )
    solver = S13PhotometricSolverConfig.from_mapping(bootstrap.get("solver", {}))
    selection = S13PhotometricSelectionConfig.from_mapping(bootstrap.get("selection", {}))
    blend = S13BlendConfig.from_mapping(bootstrap.get("blend", {}))
    visual_quality = S13VisualQualityConfig.from_mapping(bootstrap.get("visual_quality", {}))
    document = {
        "schema": M61_EFFECTIVE_CONFIG_SCHEMA,
        "algorithm_id": M61_ALGORITHM_ID,
        "implementation_id": M61_IMPLEMENTATION_ID,
        "candidate_config_sha256": declared_config_sha,
        "photometric_evidence_config": asdict(evidence),
        "photometric_evidence_config_sha256": evidence_sha,
        "quality_thresholds": threshold,
        "quality_thresholds_sha256": threshold_sha,
        "threshold_approval": approval,
        "threshold_approval_sha256": approval_sha,
        "solver": asdict(solver),
        "selection": asdict(selection),
        "blend": asdict(blend),
        "visual_quality": asdict(visual_quality),
        "luminance_field": asdict(luminance),
    }
    return S13M61EffectiveConfig(
        schema=M61_EFFECTIVE_CONFIG_SCHEMA,
        algorithm_id=M61_ALGORITHM_ID,
        implementation_id=M61_IMPLEMENTATION_ID,
        candidate_config_sha256=declared_config_sha,
        photometric_evidence_config=evidence,
        photometric_evidence_config_sha256=evidence_sha,
        quality_thresholds=threshold,
        quality_thresholds_sha256=threshold_sha,
        threshold_approval=approval,
        threshold_approval_sha256=approval_sha,
        solver=solver,
        selection=selection,
        blend=blend,
        visual_quality=visual_quality,
        luminance_field=luminance,
        effective_config_sha256=canonical_sha256(document),
    )


__all__ = [
    name for name in globals()
    if name.startswith("M61_")
] + [
    "S13LuminanceFieldDisabledConfig", "S13M61EffectiveConfig",
    "S13BlendConfig", "S13PhotometricEvidenceConfig",
    "S13PhotometricSelectionConfig", "S13PhotometricSolverConfig",
    "S13VisualQualityConfig", "canonical_sha256",
    "load_s13_m61_effective_config",
]
