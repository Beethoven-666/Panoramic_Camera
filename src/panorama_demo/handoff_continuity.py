"""Scalar-only background and foreground handoff audits.

The existing continuity gate evaluates whether a background anchor has enough
same-layer evidence for a later local-deformation candidate.  Foreground
handoffs are deliberately separate: their owner-only audit never authorizes
APAP, flow, blending, or deformation.  This module has no image, depth, pose,
mask persistence, or warp implementation.  Array inputs are consumed
immediately to produce scalar counts and percentiles; neither returned audit
nor its JSON representation retains dense evidence.
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


class ForegroundOwnerHandoffOutcome(str, Enum):
    """The only terminal actions permitted for a foreground owner boundary.

    These are intentionally not aliases for background handoff methods.  In
    particular, no APAP or flow outcome exists here: foreground pixels remain
    a single RGB owner throughout a planned owner run and its boundary.
    """

    KEEP_SOURCE = "keep_source"
    ADJACENT_OWNER_HANDOFF = "adjacent_owner_handoff"
    PAIR_LOCAL_HARD_OWNER = "pair_local_hard_owner"
    FULL_CORRIDOR_HARD_CUT = "full_corridor_hard_cut"
    STRUCTURAL_FAILURE = "structural_failure"


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


def _strict_bool(value: object, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a boolean")
    return bool(value)


def _required_identifier(value: object, name: str) -> str:
    if value is None or isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a non-empty identifier")
    result = str(value).strip()
    if not result:
        raise ValueError(f"{name} must be a non-empty identifier")
    return result


def _coerce_foreground_owner_handoff_outcome(
    value: ForegroundOwnerHandoffOutcome | str,
) -> ForegroundOwnerHandoffOutcome:
    if isinstance(value, ForegroundOwnerHandoffOutcome):
        return value
    if not isinstance(value, str):
        raise ValueError("Foreground handoff outcome must be a supported string")
    candidate = value.strip().lower()
    aliases = {
        outcome.value: outcome for outcome in ForegroundOwnerHandoffOutcome
    }
    aliases.update(
        {
            outcome.name.lower(): outcome
            for outcome in ForegroundOwnerHandoffOutcome
        }
    )
    try:
        return aliases[candidate]
    except KeyError as exc:
        raise ValueError(
            "Unsupported foreground owner handoff outcome: " + value
        ) from exc


def _current_pair_source_indices(
    value: object | None, *, pair_index: int
) -> tuple[int, int]:
    if value is None:
        return (int(pair_index), int(pair_index) + 1)
    if isinstance(value, (str, bytes)):
        raise ValueError("current_pair_source_indices must contain exactly two indices")
    try:
        values = tuple(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValueError(
            "current_pair_source_indices must contain exactly two indices"
        ) from exc
    if len(values) != 2:
        raise ValueError("current_pair_source_indices must contain exactly two indices")
    first = _nonnegative_integer(values[0], "current_pair_source_indices[0]")
    second = _nonnegative_integer(values[1], "current_pair_source_indices[1]")
    if first == second:
        raise ValueError("current_pair_source_indices must name two distinct sources")
    return (first, second)


def _reason_strings(value: Iterable[object], name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be an iterable of non-empty strings")
    try:
        values = tuple(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an iterable of non-empty strings") from exc
    reasons: list[str] = []
    for item in values:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{name} must contain only non-empty strings")
        reasons.append(item.strip())
    return tuple(reasons)


@dataclass(frozen=True)
class ForegroundOwnerHandoffAudit:
    """Fail-closed scalar audit for one foreground owner-run boundary.

    ``current_pair_source_indices`` defines the only two source owners that
    can legally participate in this pair.  A non-failure result proves that
    every foreground handoff pixel belongs to ``incoming_source_index``;
    merely proving ownership by either member of the pair is insufficient.

    The type has no APAP or flow outcome and serializes their authorization as
    false.  If an upstream caller attempts either mechanism, the evaluator
    records that attempt and returns ``STRUCTURAL_FAILURE`` instead.
    """

    pair_index: int
    track_id: str
    outgoing_run_id: str
    incoming_run_id: str
    outcome: ForegroundOwnerHandoffOutcome
    current_pair_source_indices: tuple[int, int]
    outgoing_source_index: int
    incoming_source_index: int
    handoff_pixel_count: int
    incoming_owner_pixel_count: int
    other_adjacent_owner_pixel_count: int
    nonadjacent_owner_pixel_count: int
    invalid_owner_pixel_count: int
    pair_corridor_pixel_count: int | None = None
    foreground_blend_pixel_count: int = 0
    foreground_deformation_pixel_count: int = 0
    prohibited_apap_or_flow_requested: bool = False
    prohibited_deformation_attempted: bool = False
    audit_complete: bool = True
    selection_reasons: tuple[str, ...] = ()
    rejection_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        pair_index = _nonnegative_integer(self.pair_index, "pair_index")
        track_id = _required_identifier(self.track_id, "track_id")
        outgoing_run_id = _required_identifier(self.outgoing_run_id, "outgoing_run_id")
        incoming_run_id = _required_identifier(self.incoming_run_id, "incoming_run_id")
        outcome = _coerce_foreground_owner_handoff_outcome(self.outcome)
        pair_sources = _current_pair_source_indices(
            self.current_pair_source_indices, pair_index=pair_index
        )
        outgoing_source = _nonnegative_integer(
            self.outgoing_source_index, "outgoing_source_index"
        )
        incoming_source = _nonnegative_integer(
            self.incoming_source_index, "incoming_source_index"
        )
        count_names = (
            "handoff_pixel_count",
            "incoming_owner_pixel_count",
            "other_adjacent_owner_pixel_count",
            "nonadjacent_owner_pixel_count",
            "invalid_owner_pixel_count",
            "foreground_blend_pixel_count",
            "foreground_deformation_pixel_count",
        )
        counts = {
            name: _nonnegative_integer(getattr(self, name), name) for name in count_names
        }
        if (
            counts["incoming_owner_pixel_count"]
            + counts["other_adjacent_owner_pixel_count"]
            + counts["nonadjacent_owner_pixel_count"]
            + counts["invalid_owner_pixel_count"]
            != counts["handoff_pixel_count"]
        ):
            raise ValueError("foreground handoff owner counts must partition support")
        corridor_count = (
            None
            if self.pair_corridor_pixel_count is None
            else _nonnegative_integer(
                self.pair_corridor_pixel_count, "pair_corridor_pixel_count"
            )
        )
        if corridor_count is not None and counts["handoff_pixel_count"] > corridor_count:
            raise ValueError("handoff support cannot exceed the pair corridor")
        if outcome is ForegroundOwnerHandoffOutcome.FULL_CORRIDOR_HARD_CUT and (
            corridor_count is None
            or corridor_count == 0
            or counts["handoff_pixel_count"] != corridor_count
        ):
            raise ValueError(
                "full-corridor hard cut requires complete non-empty corridor support"
            )
        prohibited_apap_or_flow = _strict_bool(
            self.prohibited_apap_or_flow_requested,
            "prohibited_apap_or_flow_requested",
        )
        prohibited_deformation = _strict_bool(
            self.prohibited_deformation_attempted,
            "prohibited_deformation_attempted",
        )
        audit_complete = _strict_bool(self.audit_complete, "audit_complete")
        selection_reasons = _reason_strings(self.selection_reasons, "selection_reasons")
        reasons = _reason_strings(self.rejection_reasons, "rejection_reasons")

        incoming_is_adjacent = incoming_source in pair_sources
        outgoing_is_adjacent = outgoing_source in pair_sources
        owner_topology_complete = (
            incoming_is_adjacent
            and counts["incoming_owner_pixel_count"]
            == counts["handoff_pixel_count"]
            and counts["other_adjacent_owner_pixel_count"] == 0
            and counts["nonadjacent_owner_pixel_count"] == 0
            and counts["invalid_owner_pixel_count"] == 0
        )
        owner_only_without_deformation = (
            counts["foreground_blend_pixel_count"] == 0
            and counts["foreground_deformation_pixel_count"] == 0
            and not prohibited_apap_or_flow
            and not prohibited_deformation
        )
        outcome_is_semantically_valid = (
            (
                outcome is not ForegroundOwnerHandoffOutcome.KEEP_SOURCE
                or outgoing_source == incoming_source
            )
            and (
                outcome
                is not ForegroundOwnerHandoffOutcome.ADJACENT_OWNER_HANDOFF
                or outgoing_source != incoming_source
            )
        )
        structurally_safe = (
            outcome is not ForegroundOwnerHandoffOutcome.STRUCTURAL_FAILURE
            and audit_complete
            and outgoing_is_adjacent
            and owner_topology_complete
            and owner_only_without_deformation
            and outcome_is_semantically_valid
        )
        if outcome is ForegroundOwnerHandoffOutcome.STRUCTURAL_FAILURE:
            if not reasons:
                raise ValueError("structural foreground handoff failure requires a reason")
        elif not structurally_safe:
            raise ValueError(
                "non-failure foreground handoff outcome requires complete adjacent "
                "incoming ownership and owner-only rendering"
            )

        object.__setattr__(self, "pair_index", pair_index)
        object.__setattr__(self, "track_id", track_id)
        object.__setattr__(self, "outgoing_run_id", outgoing_run_id)
        object.__setattr__(self, "incoming_run_id", incoming_run_id)
        object.__setattr__(self, "outcome", outcome)
        object.__setattr__(self, "current_pair_source_indices", pair_sources)
        object.__setattr__(self, "outgoing_source_index", outgoing_source)
        object.__setattr__(self, "incoming_source_index", incoming_source)
        for name, count in counts.items():
            object.__setattr__(self, name, count)
        object.__setattr__(self, "pair_corridor_pixel_count", corridor_count)
        object.__setattr__(self, "prohibited_apap_or_flow_requested", prohibited_apap_or_flow)
        object.__setattr__(self, "prohibited_deformation_attempted", prohibited_deformation)
        object.__setattr__(self, "audit_complete", audit_complete)
        object.__setattr__(self, "selection_reasons", selection_reasons)
        object.__setattr__(self, "rejection_reasons", reasons)

    @property
    def incoming_owner_is_adjacent(self) -> bool:
        """Whether the planned incoming owner belongs to the current pair."""

        return self.incoming_source_index in self.current_pair_source_indices

    @property
    def outgoing_owner_is_adjacent(self) -> bool:
        """Whether the planned outgoing owner belongs to the current pair."""

        return self.outgoing_source_index in self.current_pair_source_indices

    @property
    def incoming_owner_coverage_ratio(self) -> float:
        """Fraction of handoff support written by the named incoming owner."""

        if self.handoff_pixel_count == 0:
            return 1.0
        return float(self.incoming_owner_pixel_count / self.handoff_pixel_count)

    @property
    def incoming_owner_ownership_complete(self) -> bool:
        """Whether every audited handoff pixel has the named incoming owner."""

        return (
            self.incoming_owner_is_adjacent
            and self.incoming_owner_pixel_count == self.handoff_pixel_count
            and self.other_adjacent_owner_pixel_count == 0
            and self.nonadjacent_owner_pixel_count == 0
            and self.invalid_owner_pixel_count == 0
        )

    @property
    def owner_only_without_deformation(self) -> bool:
        """Foreground content must neither blend nor use a local deformation."""

        return (
            self.foreground_blend_pixel_count == 0
            and self.foreground_deformation_pixel_count == 0
            and not self.prohibited_apap_or_flow_requested
            and not self.prohibited_deformation_attempted
        )

    @property
    def structurally_safe(self) -> bool:
        """Whether the completed audit supports a non-failure terminal outcome."""

        return (
            self.outcome is not ForegroundOwnerHandoffOutcome.STRUCTURAL_FAILURE
            and self.audit_complete
            and self.outgoing_owner_is_adjacent
            and self.incoming_owner_ownership_complete
            and self.owner_only_without_deformation
        )

    @property
    def requires_manual_review(self) -> bool:
        """Hard-owner fallbacks are structurally safe but need C-grade review."""

        return self.outcome in {
            ForegroundOwnerHandoffOutcome.PAIR_LOCAL_HARD_OWNER,
            ForegroundOwnerHandoffOutcome.FULL_CORRIDOR_HARD_CUT,
        }

    def as_dict(self) -> dict[str, object]:
        """Return scalar-only handoff evidence suitable for renderer metadata."""

        return {
            "policy": "foreground_owner_only_handoff_audit_v1",
            "pair_index": int(self.pair_index),
            "track_id": self.track_id,
            "outgoing_run_id": self.outgoing_run_id,
            "incoming_run_id": self.incoming_run_id,
            "outcome": self.outcome.value,
            "current_pair_source_indices": [
                int(self.current_pair_source_indices[0]),
                int(self.current_pair_source_indices[1]),
            ],
            "outgoing_source_index": int(self.outgoing_source_index),
            "incoming_source_index": int(self.incoming_source_index),
            "handoff_pixel_count": int(self.handoff_pixel_count),
            "incoming_owner_pixel_count": int(self.incoming_owner_pixel_count),
            "other_adjacent_owner_pixel_count": int(
                self.other_adjacent_owner_pixel_count
            ),
            "nonadjacent_owner_pixel_count": int(self.nonadjacent_owner_pixel_count),
            "invalid_owner_pixel_count": int(self.invalid_owner_pixel_count),
            "pair_corridor_pixel_count": (
                None
                if self.pair_corridor_pixel_count is None
                else int(self.pair_corridor_pixel_count)
            ),
            "coverage_pixel_count": int(self.incoming_owner_pixel_count),
            "coverage_required_pixel_count": int(self.handoff_pixel_count),
            "incoming_owner_coverage_ratio": self.incoming_owner_coverage_ratio,
            "coverage_ratio": self.incoming_owner_coverage_ratio,
            "incoming_owner_is_adjacent": self.incoming_owner_is_adjacent,
            "outgoing_owner_is_adjacent": self.outgoing_owner_is_adjacent,
            "incoming_owner_ownership_complete": (
                self.incoming_owner_ownership_complete
            ),
            "foreground_owner_only": True,
            "owner_only_without_deformation": bool(
                self.owner_only_without_deformation
            ),
            "foreground_blend_pixel_count": int(self.foreground_blend_pixel_count),
            "foreground_deformation_pixel_count": int(
                self.foreground_deformation_pixel_count
            ),
            "apap_authorized": False,
            "flow_authorized": False,
            "local_deformation_allowed": False,
            "prohibited_apap_or_flow_requested": (
                self.prohibited_apap_or_flow_requested
            ),
            "prohibited_deformation_attempted": self.prohibited_deformation_attempted,
            "audit_complete": self.audit_complete,
            "structurally_safe": self.structurally_safe,
            "manual_review_required": self.requires_manual_review,
            "selection_reasons": list(self.selection_reasons),
            "rejection_reasons": list(self.rejection_reasons),
            "evidence_storage": "scalar_only",
        }


def evaluate_foreground_owner_handoff(
    *,
    pair_index: int,
    track_id: str | int,
    outgoing_run_id: str | int,
    incoming_run_id: str | int,
    outcome: ForegroundOwnerHandoffOutcome | str,
    outgoing_source_index: int,
    incoming_source_index: int,
    handoff_pixel_count: int,
    incoming_owner_pixel_count: int,
    other_adjacent_owner_pixel_count: int = 0,
    nonadjacent_owner_pixel_count: int = 0,
    invalid_owner_pixel_count: int = 0,
    pair_corridor_pixel_count: int | None = None,
    foreground_blend_pixel_count: int = 0,
    foreground_deformation_pixel_count: int = 0,
    current_pair_source_indices: tuple[int, int] | None = None,
    apap_authorized: bool = False,
    flow_authorized: bool = False,
    local_deformation_attempted: bool = False,
    audit_complete: bool = True,
    selection_reasons: Iterable[str] = (),
    rejection_reasons: Iterable[str] = (),
) -> ForegroundOwnerHandoffAudit:
    """Evaluate a planned foreground handoff from scalar owner evidence.

    The preferred renderer integration point is after a planned owner-run
    boundary has been rasterized, but before the pair metadata is published.
    ``incoming_owner_pixel_count`` must count pixels owned by the designated
    incoming source, not the union of both current-pair sources.  ``APAP`` and
    flow requests are converted into a structural failure; they cannot become
    a foreground authorization through this API.
    """

    normalized_pair_index = _nonnegative_integer(pair_index, "pair_index")
    requested_outcome = _coerce_foreground_owner_handoff_outcome(outcome)
    pair_sources = _current_pair_source_indices(
        current_pair_source_indices, pair_index=normalized_pair_index
    )
    outgoing_source = _nonnegative_integer(
        outgoing_source_index, "outgoing_source_index"
    )
    incoming_source = _nonnegative_integer(
        incoming_source_index, "incoming_source_index"
    )
    counts = {
        "handoff_pixel_count": _nonnegative_integer(
            handoff_pixel_count, "handoff_pixel_count"
        ),
        "incoming_owner_pixel_count": _nonnegative_integer(
            incoming_owner_pixel_count, "incoming_owner_pixel_count"
        ),
        "other_adjacent_owner_pixel_count": _nonnegative_integer(
            other_adjacent_owner_pixel_count, "other_adjacent_owner_pixel_count"
        ),
        "nonadjacent_owner_pixel_count": _nonnegative_integer(
            nonadjacent_owner_pixel_count, "nonadjacent_owner_pixel_count"
        ),
        "invalid_owner_pixel_count": _nonnegative_integer(
            invalid_owner_pixel_count, "invalid_owner_pixel_count"
        ),
        "foreground_blend_pixel_count": _nonnegative_integer(
            foreground_blend_pixel_count, "foreground_blend_pixel_count"
        ),
        "foreground_deformation_pixel_count": _nonnegative_integer(
            foreground_deformation_pixel_count, "foreground_deformation_pixel_count"
        ),
    }
    if (
        counts["incoming_owner_pixel_count"]
        + counts["other_adjacent_owner_pixel_count"]
        + counts["nonadjacent_owner_pixel_count"]
        + counts["invalid_owner_pixel_count"]
        != counts["handoff_pixel_count"]
    ):
        raise ValueError("foreground handoff owner counts must partition support")
    corridor_count = (
        None
        if pair_corridor_pixel_count is None
        else _nonnegative_integer(pair_corridor_pixel_count, "pair_corridor_pixel_count")
    )
    if corridor_count is not None and counts["handoff_pixel_count"] > corridor_count:
        raise ValueError("handoff support cannot exceed the pair corridor")
    requested_apap_or_flow = _strict_bool(apap_authorized, "apap_authorized") or _strict_bool(
        flow_authorized, "flow_authorized"
    )
    requested_deformation = _strict_bool(
        local_deformation_attempted, "local_deformation_attempted"
    )
    completed = _strict_bool(audit_complete, "audit_complete")
    normalized_selection_reasons = _reason_strings(
        selection_reasons, "selection_reasons"
    )
    reasons = list(_reason_strings(rejection_reasons, "rejection_reasons"))

    if outgoing_source not in pair_sources:
        reasons.append("outgoing_owner_is_not_current_pair_adjacent")
    if incoming_source not in pair_sources:
        reasons.append("incoming_owner_is_not_current_pair_adjacent")
    if counts["incoming_owner_pixel_count"] != counts["handoff_pixel_count"]:
        reasons.append("incoming_owner_coverage_incomplete")
    if counts["other_adjacent_owner_pixel_count"]:
        reasons.append("handoff_pixels_owned_by_other_adjacent_source")
    if counts["nonadjacent_owner_pixel_count"]:
        reasons.append("handoff_pixels_owned_by_nonadjacent_source")
    if counts["invalid_owner_pixel_count"]:
        reasons.append("handoff_pixels_without_valid_owner")
    if counts["foreground_blend_pixel_count"]:
        reasons.append("foreground_blending_detected")
    if counts["foreground_deformation_pixel_count"]:
        reasons.append("foreground_deformation_detected")
    if requested_apap_or_flow:
        reasons.append("foreground_apap_or_flow_authorization_prohibited")
    if requested_deformation:
        reasons.append("foreground_local_deformation_attempt_prohibited")
    if not completed:
        reasons.append("foreground_owner_handoff_audit_incomplete")
    if (
        requested_outcome is ForegroundOwnerHandoffOutcome.KEEP_SOURCE
        and outgoing_source != incoming_source
    ):
        reasons.append("keep_source_outcome_changes_owner")
    if (
        requested_outcome is ForegroundOwnerHandoffOutcome.ADJACENT_OWNER_HANDOFF
        and outgoing_source == incoming_source
    ):
        reasons.append("adjacent_owner_handoff_does_not_change_owner")
    if requested_outcome is ForegroundOwnerHandoffOutcome.FULL_CORRIDOR_HARD_CUT and (
        corridor_count is None
        or corridor_count == 0
        or counts["handoff_pixel_count"] != corridor_count
    ):
        reasons.append("full_corridor_hard_cut_lacks_complete_corridor_coverage")
    if requested_outcome is ForegroundOwnerHandoffOutcome.STRUCTURAL_FAILURE and not reasons:
        reasons.append("foreground_owner_handoff_marked_structural_failure")

    final_outcome = (
        ForegroundOwnerHandoffOutcome.STRUCTURAL_FAILURE
        if reasons
        else requested_outcome
    )
    return ForegroundOwnerHandoffAudit(
        pair_index=normalized_pair_index,
        track_id=track_id,
        outgoing_run_id=outgoing_run_id,
        incoming_run_id=incoming_run_id,
        outcome=final_outcome,
        current_pair_source_indices=pair_sources,
        outgoing_source_index=outgoing_source,
        incoming_source_index=incoming_source,
        handoff_pixel_count=counts["handoff_pixel_count"],
        incoming_owner_pixel_count=counts["incoming_owner_pixel_count"],
        other_adjacent_owner_pixel_count=counts["other_adjacent_owner_pixel_count"],
        nonadjacent_owner_pixel_count=counts["nonadjacent_owner_pixel_count"],
        invalid_owner_pixel_count=counts["invalid_owner_pixel_count"],
        pair_corridor_pixel_count=corridor_count,
        foreground_blend_pixel_count=counts["foreground_blend_pixel_count"],
        foreground_deformation_pixel_count=counts[
            "foreground_deformation_pixel_count"
        ],
        prohibited_apap_or_flow_requested=requested_apap_or_flow,
        prohibited_deformation_attempted=requested_deformation,
        audit_complete=completed,
        selection_reasons=normalized_selection_reasons,
        rejection_reasons=tuple(reasons),
    )


def _owner_assignment_array(value: object, name: str) -> np.ndarray:
    array = np.asarray(value)
    if not array.size or array.dtype == np.dtype(bool) or array.dtype.kind not in {"i", "u"}:
        raise ValueError(f"{name} must be a non-empty integer owner array")
    return array


def _same_shape_support_mask(
    value: object, *, name: str, shape: tuple[int, ...]
) -> np.ndarray:
    raw = np.asarray(value)
    mask = _support_mask(value, name)
    if raw.shape != shape:
        raise ValueError(f"{name} must have the same shape as handoff_support")
    return mask.reshape(shape)


def build_foreground_owner_handoff_audit(
    *,
    pair_index: int,
    track_id: str | int,
    outgoing_run_id: str | int,
    incoming_run_id: str | int,
    outcome: ForegroundOwnerHandoffOutcome | str,
    outgoing_source_index: int,
    incoming_source_index: int,
    handoff_support: object,
    owner_assignments: object,
    pair_corridor_pixel_count: int | None = None,
    current_pair_source_indices: tuple[int, int] | None = None,
    foreground_blend_support: object | None = None,
    foreground_deformation_support: object | None = None,
    apap_authorized: bool = False,
    flow_authorized: bool = False,
    local_deformation_attempted: bool = False,
    audit_complete: bool = True,
    selection_reasons: Iterable[str] = (),
    rejection_reasons: Iterable[str] = (),
) -> ForegroundOwnerHandoffAudit:
    """Consume temporary owner arrays and retain only scalar audit evidence."""

    raw_support = np.asarray(handoff_support)
    support = _same_shape_support_mask(
        handoff_support, name="handoff_support", shape=raw_support.shape
    )
    owners = _owner_assignment_array(owner_assignments, "owner_assignments")
    if owners.shape != support.shape:
        raise ValueError("owner_assignments must have the same shape as handoff_support")
    normalized_pair_index = _nonnegative_integer(pair_index, "pair_index")
    pair_sources = _current_pair_source_indices(
        current_pair_source_indices, pair_index=normalized_pair_index
    )
    incoming_source = _nonnegative_integer(
        incoming_source_index, "incoming_source_index"
    )
    if foreground_blend_support is None:
        blend = np.zeros_like(support, dtype=bool)
    else:
        blend = _same_shape_support_mask(
            foreground_blend_support,
            name="foreground_blend_support",
            shape=support.shape,
        )
    if foreground_deformation_support is None:
        deformation = np.zeros_like(support, dtype=bool)
    else:
        deformation = _same_shape_support_mask(
            foreground_deformation_support,
            name="foreground_deformation_support",
            shape=support.shape,
        )

    selected = owners[support]
    incoming = selected == incoming_source
    adjacent = (selected == pair_sources[0]) | (selected == pair_sources[1])
    valid = selected >= 0
    other_adjacent = adjacent & ~incoming
    nonadjacent = valid & ~adjacent
    invalid = ~valid
    return evaluate_foreground_owner_handoff(
        pair_index=normalized_pair_index,
        track_id=track_id,
        outgoing_run_id=outgoing_run_id,
        incoming_run_id=incoming_run_id,
        outcome=outcome,
        outgoing_source_index=outgoing_source_index,
        incoming_source_index=incoming_source,
        handoff_pixel_count=int(selected.size),
        incoming_owner_pixel_count=int(np.count_nonzero(incoming)),
        other_adjacent_owner_pixel_count=int(np.count_nonzero(other_adjacent)),
        nonadjacent_owner_pixel_count=int(np.count_nonzero(nonadjacent)),
        invalid_owner_pixel_count=int(np.count_nonzero(invalid)),
        pair_corridor_pixel_count=pair_corridor_pixel_count,
        foreground_blend_pixel_count=int(np.count_nonzero(support & blend)),
        foreground_deformation_pixel_count=int(np.count_nonzero(support & deformation)),
        current_pair_source_indices=pair_sources,
        apap_authorized=apap_authorized,
        flow_authorized=flow_authorized,
        local_deformation_attempted=local_deformation_attempted,
        audit_complete=audit_complete,
        selection_reasons=selection_reasons,
        rejection_reasons=rejection_reasons,
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
    "ForegroundOwnerHandoffAudit",
    "ForegroundOwnerHandoffOutcome",
    "HandoffContinuityAudit",
    "HandoffContinuityConfig",
    "HandoffDecision",
    "HandoffMethod",
    "build_foreground_owner_handoff_audit",
    "build_handoff_continuity_audit",
    "evaluate_foreground_owner_handoff",
    "evaluate_handoff_continuity",
    "summarize_handoff_methods",
]
