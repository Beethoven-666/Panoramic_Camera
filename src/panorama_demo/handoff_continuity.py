"""Scalar-only continuity audits for local foreground handoffs.

The formal renderer may use this module to decide whether an existing RGB
anchor can be kept at a foreground handoff or whether it must enter a bounded
local-deformation / hard-cut policy.  This module deliberately has no image,
depth, pose, mask persistence, or warp implementation.  Array inputs are
consumed immediately to produce scalar counts and percentiles; neither the
returned audit nor its JSON representation retains dense evidence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Mapping

import numpy as np


class HandoffDecision(str, Enum):
    """The next formal action selected by the continuity gate."""

    ANCHOR = "anchor"
    APAP_FLOW_FALLBACK = "apap_flow_fallback"


class HandoffMethod(str, Enum):
    """Final handoff methods reported in a delivery summary."""

    ANCHOR = "anchor"
    APAP = "apap"
    FLOW_MESH = "flow_mesh"
    HARD_CUT = "hard_cut"


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a non-negative integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a non-negative integer") from exc
    if result < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    if isinstance(value, (float, np.floating)) and not float(value).is_integer():
        raise ValueError(f"{name} must be a non-negative integer")
    return result


@dataclass(frozen=True)
class HandoffContinuityConfig:
    """Closed limits for an anchor-continuity decision.

    These values match the formal handoff proposal.  Callers may only tighten
    residual caps or raise support requirements; they cannot use configuration
    to turn a visibly discontinuous anchor into an accepted one.
    """

    maximum_normal_residual_p95_pixels: float = 0.75
    maximum_normal_residual_pixels: float = 2.0
    maximum_centreline_residual_p95_pixels: float = 1.0
    maximum_direction_delta_p95_degrees: float = 8.0
    minimum_same_layer_support_ratio: float = 0.70
    minimum_coverage_ratio: float = 1.0
    minimum_normal_residual_samples: int = 8
    minimum_centreline_residual_samples: int = 8
    minimum_direction_samples: int = 8
    minimum_same_layer_samples: int = 8
    minimum_coverage_samples: int = 8

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, object] | None
    ) -> "HandoffContinuityConfig":
        supplied = {} if value is None else dict(value)
        allowed = {
            "maximum_normal_residual_p95_pixels",
            "maximum_normal_residual_pixels",
            "maximum_centreline_residual_p95_pixels",
            "maximum_direction_delta_p95_degrees",
            "minimum_same_layer_support_ratio",
            "minimum_coverage_ratio",
            "minimum_normal_residual_samples",
            "minimum_centreline_residual_samples",
            "minimum_direction_samples",
            "minimum_same_layer_samples",
            "minimum_coverage_samples",
        }
        unknown = sorted(set(supplied) - allowed)
        if unknown:
            raise ValueError(
                "Unknown handoff_continuity configuration keys: " + ", ".join(unknown)
            )
        try:
            result = cls(**supplied)
        except TypeError as exc:
            raise ValueError("Invalid handoff_continuity configuration") from exc
        result.validate()
        return result

    def validate(self) -> None:
        caps = (
            (
                "maximum_normal_residual_p95_pixels",
                self.maximum_normal_residual_p95_pixels,
                0.75,
            ),
            (
                "maximum_normal_residual_pixels",
                self.maximum_normal_residual_pixels,
                2.0,
            ),
            (
                "maximum_centreline_residual_p95_pixels",
                self.maximum_centreline_residual_p95_pixels,
                1.0,
            ),
            (
                "maximum_direction_delta_p95_degrees",
                self.maximum_direction_delta_p95_degrees,
                8.0,
            ),
        )
        for name, value, ceiling in caps:
            number = _finite_number(value, name)
            if number <= 0.0 or number > ceiling:
                raise ValueError(f"{name} must be in (0, {ceiling}]")
        support_ratio = _finite_number(
            self.minimum_same_layer_support_ratio, "minimum_same_layer_support_ratio"
        )
        if not 0.70 <= support_ratio <= 1.0:
            raise ValueError("minimum_same_layer_support_ratio must be in [0.70, 1]")
        coverage_ratio = _finite_number(
            self.minimum_coverage_ratio, "minimum_coverage_ratio"
        )
        if not math.isclose(coverage_ratio, 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("minimum_coverage_ratio must equal 1 for a complete anchor")
        for name in (
            "minimum_normal_residual_samples",
            "minimum_centreline_residual_samples",
            "minimum_direction_samples",
            "minimum_same_layer_samples",
            "minimum_coverage_samples",
        ):
            count = _nonnegative_integer(getattr(self, name), name)
            if count < 8:
                raise ValueError(f"{name} must be at least 8")

    def as_dict(self) -> dict[str, float | int]:
        self.validate()
        return {
            "maximum_normal_residual_p95_pixels": float(
                self.maximum_normal_residual_p95_pixels
            ),
            "maximum_normal_residual_pixels": float(self.maximum_normal_residual_pixels),
            "maximum_centreline_residual_p95_pixels": float(
                self.maximum_centreline_residual_p95_pixels
            ),
            "maximum_direction_delta_p95_degrees": float(
                self.maximum_direction_delta_p95_degrees
            ),
            "minimum_same_layer_support_ratio": float(
                self.minimum_same_layer_support_ratio
            ),
            "minimum_coverage_ratio": float(self.minimum_coverage_ratio),
            "minimum_normal_residual_samples": int(self.minimum_normal_residual_samples),
            "minimum_centreline_residual_samples": int(
                self.minimum_centreline_residual_samples
            ),
            "minimum_direction_samples": int(self.minimum_direction_samples),
            "minimum_same_layer_samples": int(self.minimum_same_layer_samples),
            "minimum_coverage_samples": int(self.minimum_coverage_samples),
        }


@dataclass(frozen=True)
class HandoffContinuityAudit:
    """Scalar evidence and deterministic result for one proposed anchor.

    The class deliberately stores only scalar measurements and bounded strings.
    A failed continuity gate requests a later local APAP/flow candidate; it
    does not claim that such a candidate has been accepted.
    """

    pair_index: int
    instance_id: str
    candidate_anchor_frame_id: int
    normal_residual_sample_count: int
    normal_residual_p95_pixels: float | None
    normal_residual_max_pixels: float | None
    centreline_residual_sample_count: int
    centreline_residual_p95_pixels: float | None
    direction_sample_count: int
    direction_delta_p95_degrees: float | None
    same_layer_supported_pixel_count: int
    same_layer_candidate_pixel_count: int
    same_layer_support_ratio: float | None
    coverage_pixel_count: int
    coverage_required_pixel_count: int
    coverage_ratio: float | None
    delta_e00_sample_count: int = 0
    delta_e00_p95: float | None = None
    luminance_jump_sample_count: int = 0
    luminance_jump_p95: float | None = None
    accepted: bool = False
    decision: HandoffDecision = HandoffDecision.APAP_FLOW_FALLBACK
    rejection_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        pair_index = _nonnegative_integer(self.pair_index, "pair_index")
        anchor_id = _nonnegative_integer(
            self.candidate_anchor_frame_id, "candidate_anchor_frame_id"
        )
        instance_id = str(self.instance_id)
        if not instance_id:
            raise ValueError("instance_id must not be empty")
        count_names = (
            "normal_residual_sample_count",
            "centreline_residual_sample_count",
            "direction_sample_count",
            "same_layer_supported_pixel_count",
            "same_layer_candidate_pixel_count",
            "coverage_pixel_count",
            "coverage_required_pixel_count",
            "delta_e00_sample_count",
            "luminance_jump_sample_count",
        )
        counts = {
            name: _nonnegative_integer(getattr(self, name), name) for name in count_names
        }
        if counts["same_layer_supported_pixel_count"] > counts[
            "same_layer_candidate_pixel_count"
        ]:
            raise ValueError("same-layer support cannot exceed candidate support")
        if counts["coverage_pixel_count"] > counts["coverage_required_pixel_count"]:
            raise ValueError("coverage cannot exceed required support")
        metric_counts = (
            ("normal_residual_p95_pixels", "normal_residual_sample_count"),
            ("normal_residual_max_pixels", "normal_residual_sample_count"),
            ("centreline_residual_p95_pixels", "centreline_residual_sample_count"),
            ("direction_delta_p95_degrees", "direction_sample_count"),
            ("delta_e00_p95", "delta_e00_sample_count"),
            ("luminance_jump_p95", "luminance_jump_sample_count"),
        )
        for metric_name, count_name in metric_counts:
            metric = getattr(self, metric_name)
            if counts[count_name] == 0:
                if metric is not None:
                    raise ValueError(f"{metric_name} requires a positive sample count")
            elif metric is not None and _finite_number(metric, metric_name) < 0.0:
                raise ValueError(f"{metric_name} must be finite and non-negative")
        for ratio_name, numerator, denominator in (
            (
                "same_layer_support_ratio",
                counts["same_layer_supported_pixel_count"],
                counts["same_layer_candidate_pixel_count"],
            ),
            (
                "coverage_ratio",
                counts["coverage_pixel_count"],
                counts["coverage_required_pixel_count"],
            ),
        ):
            ratio = getattr(self, ratio_name)
            if denominator == 0:
                if ratio is not None:
                    raise ValueError(f"{ratio_name} requires a non-zero denominator")
            else:
                expected = float(numerator / denominator)
                if ratio is None or not math.isclose(
                    _finite_number(ratio, ratio_name), expected, rel_tol=0.0, abs_tol=1e-12
                ):
                    raise ValueError(f"{ratio_name} does not match its scalar counts")
        if not isinstance(self.decision, HandoffDecision):
            raise ValueError("decision must be a HandoffDecision")
        if type(self.accepted) is not bool:
            raise ValueError("accepted must be a boolean")
        if self.accepted != (self.decision is HandoffDecision.ANCHOR):
            raise ValueError("accepted continuity must use the anchor decision")
        if self.accepted and self.rejection_reasons:
            raise ValueError("accepted continuity cannot carry rejection reasons")
        if not self.accepted and not self.rejection_reasons:
            raise ValueError("rejected continuity requires at least one reason")
        if any(not isinstance(reason, str) or not reason for reason in self.rejection_reasons):
            raise ValueError("rejection reasons must be non-empty strings")
        object.__setattr__(self, "pair_index", pair_index)
        object.__setattr__(self, "candidate_anchor_frame_id", anchor_id)
        object.__setattr__(self, "instance_id", instance_id)
        for name, count in counts.items():
            object.__setattr__(self, name, count)

    def as_dict(self) -> dict[str, object]:
        """Return delivery-safe scalar metadata with no dense evidence."""

        return {
            "pair_index": int(self.pair_index),
            "instance_id": self.instance_id,
            "candidate_anchor_frame_id": int(self.candidate_anchor_frame_id),
            "normal_residual_sample_count": int(self.normal_residual_sample_count),
            "normal_residual_p95_px": self.normal_residual_p95_pixels,
            "normal_residual_max_px": self.normal_residual_max_pixels,
            "centreline_residual_sample_count": int(self.centreline_residual_sample_count),
            "centreline_residual_p95_px": self.centreline_residual_p95_pixels,
            "direction_sample_count": int(self.direction_sample_count),
            "direction_delta_p95_degrees": self.direction_delta_p95_degrees,
            "same_layer_supported_pixel_count": int(
                self.same_layer_supported_pixel_count
            ),
            "same_layer_candidate_pixel_count": int(
                self.same_layer_candidate_pixel_count
            ),
            "same_layer_support_ratio": self.same_layer_support_ratio,
            "coverage_pixel_count": int(self.coverage_pixel_count),
            "coverage_required_pixel_count": int(self.coverage_required_pixel_count),
            "coverage_ratio": self.coverage_ratio,
            "delta_e00_sample_count": int(self.delta_e00_sample_count),
            "delta_e00_p95": self.delta_e00_p95,
            "luminance_jump_sample_count": int(self.luminance_jump_sample_count),
            "luminance_jump_p95": self.luminance_jump_p95,
            "accepted": bool(self.accepted),
            "decision": self.decision.value,
            "rejection_reasons": list(self.rejection_reasons),
            "evidence_storage": "scalar_only",
        }


def _optional_scalar(value: object | None, name: str) -> float | None:
    if value is None:
        return None
    result = _finite_number(value, name)
    if result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else float(numerator / denominator)


def evaluate_handoff_continuity(
    *,
    pair_index: int,
    instance_id: str | int,
    candidate_anchor_frame_id: int,
    normal_residual_sample_count: int,
    normal_residual_p95_pixels: float | None,
    normal_residual_max_pixels: float | None,
    centreline_residual_sample_count: int,
    centreline_residual_p95_pixels: float | None,
    direction_sample_count: int,
    direction_delta_p95_degrees: float | None,
    same_layer_supported_pixel_count: int,
    same_layer_candidate_pixel_count: int,
    coverage_pixel_count: int,
    coverage_required_pixel_count: int,
    delta_e00_sample_count: int = 0,
    delta_e00_p95: float | None = None,
    luminance_jump_sample_count: int = 0,
    luminance_jump_p95: float | None = None,
    config: HandoffContinuityConfig | None = None,
) -> HandoffContinuityAudit:
    """Evaluate scalar handoff evidence from renderer-side measurements.

    This is the preferred integration API when a renderer has already reduced
    its temporary masks and residual arrays to counts/percentiles.
    """

    settings = HandoffContinuityConfig() if config is None else config
    if not isinstance(settings, HandoffContinuityConfig):
        raise TypeError("config must be a HandoffContinuityConfig")
    settings.validate()
    normal_count = _nonnegative_integer(
        normal_residual_sample_count, "normal_residual_sample_count"
    )
    centreline_count = _nonnegative_integer(
        centreline_residual_sample_count, "centreline_residual_sample_count"
    )
    direction_count = _nonnegative_integer(direction_sample_count, "direction_sample_count")
    same_layer_count = _nonnegative_integer(
        same_layer_supported_pixel_count, "same_layer_supported_pixel_count"
    )
    same_layer_total = _nonnegative_integer(
        same_layer_candidate_pixel_count, "same_layer_candidate_pixel_count"
    )
    coverage_count = _nonnegative_integer(coverage_pixel_count, "coverage_pixel_count")
    coverage_total = _nonnegative_integer(
        coverage_required_pixel_count, "coverage_required_pixel_count"
    )
    delta_count = _nonnegative_integer(delta_e00_sample_count, "delta_e00_sample_count")
    luminance_count = _nonnegative_integer(
        luminance_jump_sample_count, "luminance_jump_sample_count"
    )
    metrics = {
        "normal_residual_p95_pixels": _optional_scalar(
            normal_residual_p95_pixels, "normal_residual_p95_pixels"
        ),
        "normal_residual_max_pixels": _optional_scalar(
            normal_residual_max_pixels, "normal_residual_max_pixels"
        ),
        "centreline_residual_p95_pixels": _optional_scalar(
            centreline_residual_p95_pixels, "centreline_residual_p95_pixels"
        ),
        "direction_delta_p95_degrees": _optional_scalar(
            direction_delta_p95_degrees, "direction_delta_p95_degrees"
        ),
        "delta_e00_p95": _optional_scalar(delta_e00_p95, "delta_e00_p95"),
        "luminance_jump_p95": _optional_scalar(
            luminance_jump_p95, "luminance_jump_p95"
        ),
    }
    reasons: list[str] = []
    if normal_count < settings.minimum_normal_residual_samples:
        reasons.append("insufficient_normal_residual_support")
    elif (
        metrics["normal_residual_p95_pixels"] is None
        or metrics["normal_residual_max_pixels"] is None
    ):
        reasons.append("normal_residual_metrics_unavailable")
    else:
        if metrics["normal_residual_p95_pixels"] > settings.maximum_normal_residual_p95_pixels:
            reasons.append("normal_residual_p95_exceeded")
        if metrics["normal_residual_max_pixels"] > settings.maximum_normal_residual_pixels:
            reasons.append("normal_residual_maximum_exceeded")
    if centreline_count < settings.minimum_centreline_residual_samples:
        reasons.append("insufficient_centreline_support")
    elif metrics["centreline_residual_p95_pixels"] is None:
        reasons.append("centreline_residual_metric_unavailable")
    elif (
        metrics["centreline_residual_p95_pixels"]
        > settings.maximum_centreline_residual_p95_pixels
    ):
        reasons.append("centreline_residual_p95_exceeded")
    if direction_count < settings.minimum_direction_samples:
        reasons.append("insufficient_direction_support")
    elif metrics["direction_delta_p95_degrees"] is None:
        reasons.append("direction_metric_unavailable")
    elif (
        metrics["direction_delta_p95_degrees"]
        > settings.maximum_direction_delta_p95_degrees
    ):
        reasons.append("direction_delta_p95_exceeded")
    same_layer_ratio = _ratio(same_layer_count, same_layer_total)
    if same_layer_total < settings.minimum_same_layer_samples:
        reasons.append("insufficient_same_layer_support")
    elif same_layer_ratio is None or same_layer_ratio < settings.minimum_same_layer_support_ratio:
        reasons.append("same_layer_support_ratio_too_small")
    coverage_ratio = _ratio(coverage_count, coverage_total)
    if coverage_total < settings.minimum_coverage_samples:
        reasons.append("insufficient_coverage_support")
    elif coverage_ratio is None or coverage_ratio < settings.minimum_coverage_ratio:
        reasons.append("incomplete_anchor_coverage")
    accepted = not reasons
    return HandoffContinuityAudit(
        pair_index=pair_index,
        instance_id=str(instance_id),
        candidate_anchor_frame_id=candidate_anchor_frame_id,
        normal_residual_sample_count=normal_count,
        normal_residual_p95_pixels=metrics["normal_residual_p95_pixels"],
        normal_residual_max_pixels=metrics["normal_residual_max_pixels"],
        centreline_residual_sample_count=centreline_count,
        centreline_residual_p95_pixels=metrics["centreline_residual_p95_pixels"],
        direction_sample_count=direction_count,
        direction_delta_p95_degrees=metrics["direction_delta_p95_degrees"],
        same_layer_supported_pixel_count=same_layer_count,
        same_layer_candidate_pixel_count=same_layer_total,
        same_layer_support_ratio=same_layer_ratio,
        coverage_pixel_count=coverage_count,
        coverage_required_pixel_count=coverage_total,
        coverage_ratio=coverage_ratio,
        delta_e00_sample_count=delta_count,
        delta_e00_p95=metrics["delta_e00_p95"],
        luminance_jump_sample_count=luminance_count,
        luminance_jump_p95=metrics["luminance_jump_p95"],
        accepted=accepted,
        decision=(
            HandoffDecision.ANCHOR if accepted else HandoffDecision.APAP_FLOW_FALLBACK
        ),
        rejection_reasons=tuple(reasons),
    )


def _residual_samples(value: object, name: str) -> np.ndarray:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a non-empty numeric array")
    try:
        array = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a non-empty numeric array") from exc
    if not array.size or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite samples")
    return np.abs(array)


def _support_mask(value: object, name: str) -> np.ndarray:
    array = np.asarray(value)
    if not array.size:
        raise ValueError(f"{name} must not be empty")
    if array.dtype == np.dtype(bool):
        return array.reshape(-1)
    if array.dtype.kind not in {"i", "u"} or not np.all((array == 0) | (array == 1)):
        raise ValueError(f"{name} must be a boolean or 0/1 integer array")
    return array.astype(bool, copy=False).reshape(-1)


def _optional_samples(value: object | None, name: str) -> tuple[int, float | None]:
    if value is None:
        return 0, None
    samples = _residual_samples(value, name)
    return int(samples.size), float(np.percentile(samples, 95.0))


def build_handoff_continuity_audit(
    *,
    pair_index: int,
    instance_id: str | int,
    candidate_anchor_frame_id: int,
    normal_residual_pixels: object,
    centreline_residual_pixels: object,
    direction_delta_degrees: object,
    same_layer_support: object,
    coverage: object,
    delta_e00: object | None = None,
    luminance_jump: object | None = None,
    config: HandoffContinuityConfig | None = None,
) -> HandoffContinuityAudit:
    """Consume temporary arrays and return one scalar-only continuity audit.

    Signed normal/centreline residuals and orientation deltas are converted to
    magnitudes before percentile calculation.  ``same_layer_support`` and
    ``coverage`` are boolean (or 0/1 integer) arrays over their respective
    temporary candidate domains.
    """

    normal = _residual_samples(normal_residual_pixels, "normal_residual_pixels")
    centreline = _residual_samples(
        centreline_residual_pixels, "centreline_residual_pixels"
    )
    direction = _residual_samples(direction_delta_degrees, "direction_delta_degrees")
    same_layer = _support_mask(same_layer_support, "same_layer_support")
    coverage_mask = _support_mask(coverage, "coverage")
    delta_count, delta_p95 = _optional_samples(delta_e00, "delta_e00")
    luminance_count, luminance_p95 = _optional_samples(
        luminance_jump, "luminance_jump"
    )
    return evaluate_handoff_continuity(
        pair_index=pair_index,
        instance_id=instance_id,
        candidate_anchor_frame_id=candidate_anchor_frame_id,
        normal_residual_sample_count=int(normal.size),
        normal_residual_p95_pixels=float(np.percentile(normal, 95.0)),
        normal_residual_max_pixels=float(np.max(normal)),
        centreline_residual_sample_count=int(centreline.size),
        centreline_residual_p95_pixels=float(np.percentile(centreline, 95.0)),
        direction_sample_count=int(direction.size),
        direction_delta_p95_degrees=float(np.percentile(direction, 95.0)),
        same_layer_supported_pixel_count=int(np.count_nonzero(same_layer)),
        same_layer_candidate_pixel_count=int(same_layer.size),
        coverage_pixel_count=int(np.count_nonzero(coverage_mask)),
        coverage_required_pixel_count=int(coverage_mask.size),
        delta_e00_sample_count=delta_count,
        delta_e00_p95=delta_p95,
        luminance_jump_sample_count=luminance_count,
        luminance_jump_p95=luminance_p95,
        config=config,
    )


def _coerce_handoff_method(value: object) -> HandoffMethod:
    if isinstance(value, HandoffMethod):
        return value
    candidate = value
    if isinstance(value, Mapping):
        for key in ("method", "handoff_method", "decision"):
            if key in value:
                candidate = value[key]
                break
        else:
            raise ValueError(
                "Handoff summary mapping requires method, handoff_method, or decision"
            )
    if not isinstance(candidate, str):
        raise ValueError("Handoff method must be a supported string or HandoffMethod")
    aliases = {
        "anchor": HandoffMethod.ANCHOR,
        "apap": HandoffMethod.APAP,
        "flow_mesh": HandoffMethod.FLOW_MESH,
        "apap_plus_dense_flow": HandoffMethod.FLOW_MESH,
        "apap_flow": HandoffMethod.FLOW_MESH,
        "hard_cut": HandoffMethod.HARD_CUT,
        "hard_cut_degraded": HandoffMethod.HARD_CUT,
    }
    try:
        return aliases[candidate]
    except KeyError as exc:
        raise ValueError(f"Unsupported handoff method: {candidate}") from exc


def summarize_handoff_methods(
    methods: Iterable[HandoffMethod | str | Mapping[str, object]],
) -> dict[str, int]:
    """Return the bounded delivery summary for final handoff methods.

    The function accepts existing renderer-style metadata mappings containing
    ``method``, ``handoff_method``, or terminal ``decision``.  A continuity
    gate's provisional ``apap_flow_fallback`` decision is intentionally not
    accepted here: only a completed method may affect the published grade
    summary.
    """

    summary = {method.value: 0 for method in HandoffMethod}
    for value in methods:
        method = _coerce_handoff_method(value)
        summary[method.value] += 1
    return summary


__all__ = [
    "HandoffContinuityAudit",
    "HandoffContinuityConfig",
    "HandoffDecision",
    "HandoffMethod",
    "build_handoff_continuity_audit",
    "evaluate_handoff_continuity",
    "summarize_handoff_methods",
]
