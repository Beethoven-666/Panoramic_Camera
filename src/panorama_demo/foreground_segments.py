"""Fail-closed foreground span and handoff planning for RGB owner decisions.

The calibrated pushbroom renderer remains responsible for the only RGB remap,
GraphCut, hard-owner write, and safe-wall MultiBand blend.  This module is a
pure planning layer placed before that renderer's owner solve.  It never
changes a pose, generates a pixel, or accepts a foreground warp.

The first formal version is intentionally conservative.  A cross-pair
foreground association is usable only when the caller supplies a shared
source-frame raw-footprint summary *and* bidirectional depth visibility
evidence.  Aligned-depth legacy inputs therefore remain ``IMAGE_REGION``
owner-only fragments: they are audited, but cannot fabricate a long-range
identity or loosen the existing protected-component rule.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, Sequence

import numpy as np

from .rgb_residual_alignment import ProtectedComponentFragment


class GeometryMode(str, Enum):
    """Evidence class for a foreground fragment.

    ``RIGID_PROXY`` is represented for audit compatibility only.  It is not a
    formal planning mode: the production path must reject it rather than turn
    an aligned-depth edge into a rigid geometric proxy.
    """

    DEPTH_OBSERVED = "depth_observed"
    RIGID_PROXY = "rigid_proxy"
    IMAGE_REGION = "image_region"
    UNKNOWN = "unknown"


def _finite(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


@dataclass(frozen=True)
class RawFootprintSummary:
    """A bounded, source-coordinate footprint summary for one component.

    The private occupancy grid is a small fixed-resolution summary in the raw
    RGB source coordinate system.  It lets the planner prove that two adjacent
    pair components actually touch the same *shared source* footprint without
    persisting a source map, RGB image, or depth image in a delivery sidecar.
    """

    source_index: int
    source_size: tuple[int, int]
    bounds_xyxy: tuple[float, float, float, float]
    sample_count: int
    occupancy: np.ndarray = field(repr=False)

    def __post_init__(self) -> None:
        if int(self.source_index) < 0:
            raise ValueError("Raw footprint source_index must be non-negative")
        width, height = (int(value) for value in self.source_size)
        if width <= 0 or height <= 0:
            raise ValueError("Raw footprint source_size must be positive")
        bounds = tuple(_finite(value, "Raw footprint bounds") for value in self.bounds_xyxy)
        if len(bounds) != 4 or bounds[2] < bounds[0] or bounds[3] < bounds[1]:
            raise ValueError("Raw footprint bounds must be ordered xyxy")
        if int(self.sample_count) <= 0:
            raise ValueError("Raw footprint requires at least one sample")
        occupancy = np.asarray(self.occupancy, dtype=bool)
        if occupancy.ndim != 2 or occupancy.shape[0] < 2 or occupancy.shape[1] < 2:
            raise ValueError("Raw footprint occupancy must be a 2-D grid")
        if not np.any(occupancy):
            raise ValueError("Raw footprint occupancy must not be empty")
        object.__setattr__(self, "source_size", (width, height))
        object.__setattr__(self, "bounds_xyxy", bounds)
        object.__setattr__(self, "sample_count", int(self.sample_count))
        object.__setattr__(self, "occupancy", np.ascontiguousarray(occupancy))

    @classmethod
    def from_source_coordinates(
        cls,
        *,
        source_index: int,
        source_size: tuple[int, int],
        map_x: np.ndarray,
        map_y: np.ndarray,
        mask: np.ndarray,
        grid_shape: tuple[int, int] = (32, 32),
    ) -> "RawFootprintSummary | None":
        """Summarise masked calibrated source coordinates without retaining maps."""

        width, height = (int(value) for value in source_size)
        rows, columns = (int(value) for value in grid_shape)
        if width <= 0 or height <= 0 or rows < 2 or columns < 2:
            raise ValueError("Raw footprint dimensions must be positive")
        source_x = np.asarray(map_x, dtype=np.float64)
        source_y = np.asarray(map_y, dtype=np.float64)
        component = np.asarray(mask, dtype=bool)
        if source_x.shape != source_y.shape or source_x.shape != component.shape:
            raise ValueError("Raw footprint maps and mask must share one shape")
        valid = (
            component
            & np.isfinite(source_x)
            & np.isfinite(source_y)
            & (source_x >= 0.0)
            & (source_y >= 0.0)
            & (source_x < float(width))
            & (source_y < float(height))
        )
        if not np.any(valid):
            return None
        x = source_x[valid]
        y = source_y[valid]
        occupancy = np.zeros((rows, columns), dtype=bool)
        cell_x = np.minimum(columns - 1, np.floor(x * columns / width).astype(np.int32))
        cell_y = np.minimum(rows - 1, np.floor(y * rows / height).astype(np.int32))
        occupancy[cell_y, cell_x] = True
        return cls(
            source_index=int(source_index),
            source_size=(width, height),
            bounds_xyxy=(float(x.min()), float(y.min()), float(x.max()), float(y.max())),
            sample_count=int(x.size),
            occupancy=occupancy,
        )

    @property
    def occupied_cell_count(self) -> int:
        return int(np.count_nonzero(self.occupancy))

    def overlap_iou(self, other: "RawFootprintSummary") -> float:
        """Return exact fixed-grid occupancy IoU for one shared source frame."""

        if (
            self.source_index != other.source_index
            or self.source_size != other.source_size
            or self.occupancy.shape != other.occupancy.shape
        ):
            return 0.0
        union = int(np.count_nonzero(self.occupancy | other.occupancy))
        if union == 0:
            return 0.0
        return float(np.count_nonzero(self.occupancy & other.occupancy) / union)

    def as_dict(self) -> dict[str, object]:
        """Return scalar-only audit evidence; never expose occupancy itself."""

        return {
            "source_index": int(self.source_index),
            "source_size": [int(value) for value in self.source_size],
            "bounds_xyxy": [float(value) for value in self.bounds_xyxy],
            "sample_count": int(self.sample_count),
            "occupancy_grid_shape": [
                int(self.occupancy.shape[0]),
                int(self.occupancy.shape[1]),
            ],
            "occupied_cell_count": self.occupied_cell_count,
        }


@dataclass(frozen=True)
class DepthAnchorToken:
    """Immutable identity for one two-pair direct signed-occlusion anchor.

    The two local pair tiles intentionally retain their own compact integer
    label images.  This token is the non-image bridge that proves that those
    two labels came from the *same* exact raw signed-occlusion match in their
    shared middle RGB-D source.  It is an owner-only identity, never a track,
    a pose estimate, or a deformation field.
    """

    shared_source_index: int
    left_pair_index: int
    right_pair_index: int
    left_direct_component_label: int
    right_direct_component_label: int

    def __post_init__(self) -> None:
        values = (
            self.shared_source_index,
            self.left_pair_index,
            self.right_pair_index,
            self.left_direct_component_label,
            self.right_direct_component_label,
        )
        if any(isinstance(value, (bool, np.bool_)) for value in values):
            raise ValueError("Depth anchor token fields must be integer identifiers")
        try:
            shared_source, left_pair, right_pair, left_label, right_label = (
                int(value) for value in values
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("Depth anchor token fields must be integer identifiers") from exc
        if (
            shared_source < 0
            or left_pair < 0
            or right_pair != left_pair + 1
            or shared_source != right_pair
            or left_label < 1
            or right_label < 1
        ):
            raise ValueError("Depth anchor token is not one shared adjacent-pair identity")
        object.__setattr__(self, "shared_source_index", shared_source)
        object.__setattr__(self, "left_pair_index", left_pair)
        object.__setattr__(self, "right_pair_index", right_pair)
        object.__setattr__(self, "left_direct_component_label", left_label)
        object.__setattr__(self, "right_direct_component_label", right_label)

    def as_dict(self) -> dict[str, int]:
        return {
            "shared_source_index": int(self.shared_source_index),
            "left_pair_index": int(self.left_pair_index),
            "right_pair_index": int(self.right_pair_index),
            "left_direct_component_label": int(self.left_direct_component_label),
            "right_direct_component_label": int(self.right_direct_component_label),
        }


@dataclass(frozen=True)
class ForegroundFragment:
    """One protected pair-corridor foreground component and its evidence."""

    pair_index: int
    component_label: int
    frame_ids: tuple[int, int]
    source_indices: tuple[int, int]
    global_bbox: tuple[int, int, int, int]
    local_mask: np.ndarray = field(repr=False)
    depth_anchor_local_mask: np.ndarray | None = field(default=None, repr=False)
    depth_anchor_token: DepthAnchorToken | None = None
    allowed_local_owners: tuple[int, ...] = (0, 1)
    preferred_local_owner: int | None = None
    edge_orientation_degrees: float | None = None
    local_width_pixels: float | None = None
    raw_footprints: tuple[RawFootprintSummary | None, RawFootprintSummary | None] = (
        None,
        None,
    )
    geometry_mode: GeometryMode = GeometryMode.UNKNOWN
    bidirectional_visibility_supported: bool = False
    natural_break_reason: str | None = None

    def __post_init__(self) -> None:
        if int(self.pair_index) < 0 or int(self.component_label) < 1:
            raise ValueError("Foreground fragment pair and component labels must be positive")
        if len(self.frame_ids) != 2 or len(self.source_indices) != 2:
            raise ValueError("Foreground fragment requires exactly two source frames")
        if self.source_indices[0] == self.source_indices[1] or min(self.source_indices) < 0:
            raise ValueError("Foreground fragment source indices must be distinct and non-negative")
        x, y, width, height = (int(value) for value in self.global_bbox)
        if width <= 0 or height <= 0:
            raise ValueError("Foreground fragment global_bbox must be non-empty")
        mask = np.asarray(self.local_mask, dtype=bool)
        if mask.shape != (height, width) or not np.any(mask):
            raise ValueError("Foreground fragment mask must fill its non-empty bbox shape")
        anchor_mask = (
            None
            if self.depth_anchor_local_mask is None
            else np.asarray(self.depth_anchor_local_mask, dtype=bool)
        )
        if anchor_mask is not None:
            if anchor_mask.shape != mask.shape or not np.any(anchor_mask):
                raise ValueError(
                    "Foreground depth anchor mask must be non-empty and match local_mask"
                )
            if np.any(anchor_mask & ~mask):
                raise ValueError("Foreground depth anchor mask must be a local_mask subset")
        if (anchor_mask is None) != (self.depth_anchor_token is None):
            raise ValueError(
                "Foreground depth anchor mask and token must be present together"
            )
        if self.depth_anchor_token is not None and not isinstance(
            self.depth_anchor_token, DepthAnchorToken
        ):
            raise ValueError("Foreground depth anchor token has the wrong type")
        owners = tuple(int(owner) for owner in self.allowed_local_owners)
        if not owners or any(owner not in {0, 1} for owner in owners):
            raise ValueError("Foreground fragment owners must be a non-empty subset of (0, 1)")
        if self.preferred_local_owner is not None and int(self.preferred_local_owner) not in owners:
            raise ValueError("Foreground fragment preferred owner must be completely available")
        if self.edge_orientation_degrees is not None:
            _finite(self.edge_orientation_degrees, "Foreground fragment orientation")
        width_pixels = self.local_width_pixels
        if width_pixels is not None and _finite(width_pixels, "Foreground fragment local width") <= 0.0:
            raise ValueError("Foreground fragment local width must be positive")
        if len(self.raw_footprints) != 2:
            raise ValueError("Foreground fragment needs two owner footprint slots")
        for owner, footprint in enumerate(self.raw_footprints):
            if footprint is not None and footprint.source_index != int(self.source_indices[owner]):
                raise ValueError("Foreground footprint source does not match local owner")
        if not isinstance(self.geometry_mode, GeometryMode):
            raise ValueError("Foreground fragment geometry_mode must be a GeometryMode")
        if type(self.bidirectional_visibility_supported) is not bool:
            raise ValueError("Foreground fragment visibility support must be boolean")
        if self.natural_break_reason is not None and not str(self.natural_break_reason):
            raise ValueError("Foreground fragment natural break reason must not be empty")
        object.__setattr__(self, "pair_index", int(self.pair_index))
        object.__setattr__(self, "component_label", int(self.component_label))
        object.__setattr__(self, "frame_ids", tuple(int(value) for value in self.frame_ids))
        object.__setattr__(self, "source_indices", tuple(int(value) for value in self.source_indices))
        object.__setattr__(self, "global_bbox", (x, y, width, height))
        object.__setattr__(self, "local_mask", np.ascontiguousarray(mask))
        object.__setattr__(
            self,
            "depth_anchor_local_mask",
            None if anchor_mask is None else np.ascontiguousarray(anchor_mask),
        )
        object.__setattr__(self, "allowed_local_owners", owners)
        object.__setattr__(
            self,
            "preferred_local_owner",
            None if self.preferred_local_owner is None else int(self.preferred_local_owner),
        )
        if width_pixels is None:
            width_pixels = float(min(width, height))
        object.__setattr__(self, "local_width_pixels", float(width_pixels))

    @property
    def reference(self) -> tuple[int, int]:
        return (self.pair_index, self.component_label)

    @property
    def pixel_count(self) -> int:
        return int(np.count_nonzero(self.local_mask))

    @property
    def anchor_pixel_count(self) -> int:
        """Number of depth-observed pixels reserved inside this guard."""

        if self.depth_anchor_local_mask is None:
            return 0
        return int(np.count_nonzero(self.depth_anchor_local_mask))

    def local_owner_for_source(self, source_index: int) -> int | None:
        for owner, candidate in enumerate(self.source_indices):
            if int(candidate) == int(source_index) and owner in self.allowed_local_owners:
                return owner
        return None

    def as_dict(self) -> dict[str, object]:
        return {
            "pair_index": int(self.pair_index),
            "component_label": int(self.component_label),
            "frame_ids": [int(value) for value in self.frame_ids],
            "source_indices": [int(value) for value in self.source_indices],
            "global_bbox": [int(value) for value in self.global_bbox],
            "pixel_count": self.pixel_count,
            "anchor_pixel_count": self.anchor_pixel_count,
            "depth_anchor_token": (
                None
                if self.depth_anchor_token is None
                else self.depth_anchor_token.as_dict()
            ),
            "allowed_local_owners": [int(value) for value in self.allowed_local_owners],
            "preferred_local_owner": self.preferred_local_owner,
            "edge_orientation_degrees": self.edge_orientation_degrees,
            "local_width_pixels": float(self.local_width_pixels),
            "geometry_mode": self.geometry_mode.value,
            "bidirectional_visibility_supported": self.bidirectional_visibility_supported,
            "natural_break_reason": self.natural_break_reason,
            "raw_footprints": [
                None if footprint is None else footprint.as_dict()
                for footprint in self.raw_footprints
            ],
        }


def foreground_fragment_from_protected(
    fragment: ProtectedComponentFragment,
    *,
    frame_ids: tuple[int, int],
    source_indices: tuple[int, int],
    geometry_mode: GeometryMode = GeometryMode.IMAGE_REGION,
    bidirectional_visibility_supported: bool = False,
    depth_anchor_local_mask: np.ndarray | None = None,
    depth_anchor_token: DepthAnchorToken | None = None,
    raw_footprints: tuple[RawFootprintSummary | None, RawFootprintSummary | None] = (
        None,
        None,
    ),
    natural_break_reason: str | None = None,
) -> ForegroundFragment:
    """Adapt the existing pair-local owner guard without changing its pixels."""

    if fragment.component_label is None:
        raise ValueError("Protected component lacks a stable component label")
    return ForegroundFragment(
        pair_index=int(fragment.pair_index),
        component_label=int(fragment.component_label),
        frame_ids=frame_ids,
        source_indices=source_indices,
        global_bbox=fragment.global_bbox,
        local_mask=fragment.local_mask,
        depth_anchor_local_mask=depth_anchor_local_mask,
        depth_anchor_token=depth_anchor_token,
        allowed_local_owners=fragment.allowed_owners,
        preferred_local_owner=fragment.preferred_owner,
        edge_orientation_degrees=fragment.edge_orientation,
        raw_footprints=raw_footprints,
        geometry_mode=geometry_mode,
        bidirectional_visibility_supported=bidirectional_visibility_supported,
        natural_break_reason=natural_break_reason,
    )


def build_foreground_fragments(
    fragments_by_pair: Sequence[Sequence[ProtectedComponentFragment]],
    *,
    frame_ids: Sequence[int],
    geometry_modes: Sequence[GeometryMode] | None = None,
    bidirectional_visibility: Sequence[bool] | None = None,
    geometry_mode_overrides: Mapping[tuple[int, int], GeometryMode] | None = None,
    bidirectional_visibility_overrides: Mapping[tuple[int, int], bool] | None = None,
    depth_anchor_masks: Mapping[tuple[int, int], np.ndarray] | None = None,
    depth_anchor_tokens: Mapping[tuple[int, int], DepthAnchorToken] | None = None,
    raw_footprints: Mapping[tuple[int, int, int], RawFootprintSummary] | None = None,
    natural_breaks: Mapping[tuple[int, int], str] | None = None,
) -> tuple[tuple[ForegroundFragment, ...], ...]:
    """Adapt pair guards into a pure v2 planning input.

    The default is deliberately ``IMAGE_REGION`` with no visibility evidence:
    that is the only honest formal interpretation of an aligned-only legacy
    input.  A caller may promote only a particular protected component by
    supplying its own per-fragment, bidirectionally verified depth evidence;
    a pair-wide RGB bounding box can never promote its neighbours.
    """

    pair_count = len(fragments_by_pair)
    if len(frame_ids) != pair_count + 1:
        raise ValueError("Foreground planning frame_ids must cover every adjacent pair")
    if geometry_modes is None:
        geometry_modes = (GeometryMode.IMAGE_REGION,) * pair_count
    if bidirectional_visibility is None:
        bidirectional_visibility = (False,) * pair_count
    if len(geometry_modes) != pair_count or len(bidirectional_visibility) != pair_count:
        raise ValueError("Foreground planning pair evidence has an invalid length")
    footprints = {} if raw_footprints is None else dict(raw_footprints)
    breaks = {} if natural_breaks is None else dict(natural_breaks)
    mode_overrides = (
        {} if geometry_mode_overrides is None else dict(geometry_mode_overrides)
    )
    visibility_overrides = (
        {}
        if bidirectional_visibility_overrides is None
        else dict(bidirectional_visibility_overrides)
    )
    anchor_masks = {} if depth_anchor_masks is None else dict(depth_anchor_masks)
    anchor_tokens = {} if depth_anchor_tokens is None else dict(depth_anchor_tokens)
    if set(anchor_masks) != set(anchor_tokens):
        raise ValueError("Depth anchor masks and tokens must use the same component references")
    known_references = {
        (pair_index, int(fragment.component_label))
        for pair_index, fragments in enumerate(fragments_by_pair)
        for fragment in fragments
        if fragment.component_label is not None
    }
    unknown_overrides = (
        set(mode_overrides)
        | set(visibility_overrides)
        | set(anchor_masks)
        | set(anchor_tokens)
    ) - known_references
    if unknown_overrides:
        raise ValueError("Foreground planning override references an unknown component")
    if not all(isinstance(mode, GeometryMode) for mode in mode_overrides.values()):
        raise ValueError("Foreground planning geometry overrides must contain GeometryMode")
    if not all(isinstance(value, (bool, np.bool_)) for value in visibility_overrides.values()):
        raise ValueError("Foreground planning visibility overrides must be boolean")
    result: list[tuple[ForegroundFragment, ...]] = []
    for pair_index, protected_fragments in enumerate(fragments_by_pair):
        mode = geometry_modes[pair_index]
        if not isinstance(mode, GeometryMode):
            raise ValueError("Foreground planning geometry_modes must contain GeometryMode")
        pair_fragments: list[ForegroundFragment] = []
        for protected in protected_fragments:
            if int(protected.pair_index) != pair_index or protected.component_label is None:
                raise ValueError("Foreground planning fragments must be ordered by pair")
            label = int(protected.component_label)
            reference = (pair_index, label)
            pair_fragments.append(
                foreground_fragment_from_protected(
                    protected,
                    frame_ids=(int(frame_ids[pair_index]), int(frame_ids[pair_index + 1])),
                    source_indices=(pair_index, pair_index + 1),
                    geometry_mode=mode_overrides.get(reference, mode),
                    bidirectional_visibility_supported=bool(
                        visibility_overrides.get(
                            reference, bidirectional_visibility[pair_index]
                        )
                    ),
                    depth_anchor_local_mask=anchor_masks.get(reference),
                    depth_anchor_token=anchor_tokens.get(reference),
                    raw_footprints=(
                        footprints.get((pair_index, label, 0)),
                        footprints.get((pair_index, label, 1)),
                    ),
                    natural_break_reason=breaks.get((pair_index, label)),
                )
            )
        result.append(tuple(pair_fragments))
    return tuple(result)


@dataclass(frozen=True)
class AnchorCandidate:
    """A complete-coverage anchor option for one span."""

    source_index: int
    frame_id: int
    complete_coverage: bool
    coverage_margin_pixels: float | None
    score: float | None
    rejection_reason: str | None = None

    def __post_init__(self) -> None:
        if int(self.source_index) < 0:
            raise ValueError("Anchor source_index must be non-negative")
        if self.coverage_margin_pixels is not None:
            _finite(self.coverage_margin_pixels, "Anchor coverage margin")
        if self.score is not None:
            _finite(self.score, "Anchor score")
        if not self.complete_coverage and self.rejection_reason is None:
            raise ValueError("Incomplete anchor candidates need a rejection reason")

    def as_dict(self) -> dict[str, object]:
        return {
            "source_index": int(self.source_index),
            "frame_id": int(self.frame_id),
            "complete_coverage": bool(self.complete_coverage),
            "coverage_margin_pixels": self.coverage_margin_pixels,
            "score": self.score,
            "rejection_reason": self.rejection_reason,
        }


@dataclass(frozen=True)
class ForegroundSpan:
    """One contiguous foreground region with exactly one RGB anchor."""

    segment_id: int
    span_id: int
    fragment_refs: tuple[tuple[int, int], ...]
    anchor_source_index: int
    anchor_frame_id: int
    geometry_mode: GeometryMode
    anchor_candidates: tuple[AnchorCandidate, ...]

    def __post_init__(self) -> None:
        if int(self.segment_id) < 0 or int(self.span_id) < 0 or not self.fragment_refs:
            raise ValueError("Foreground span needs an id and at least one fragment")
        if not isinstance(self.geometry_mode, GeometryMode):
            raise ValueError("Foreground span geometry_mode must be a GeometryMode")
        if not any(
            candidate.source_index == int(self.anchor_source_index)
            and candidate.frame_id == int(self.anchor_frame_id)
            and candidate.complete_coverage
            for candidate in self.anchor_candidates
        ):
            raise ValueError("Foreground span anchor must completely cover its support")

    def as_dict(self) -> dict[str, object]:
        return {
            "segment_id": int(self.segment_id),
            "span_id": int(self.span_id),
            "fragment_refs": [
                {"pair_index": int(pair), "component_label": int(label)}
                for pair, label in self.fragment_refs
            ],
            "anchor_source_index": int(self.anchor_source_index),
            "anchor_frame_id": int(self.anchor_frame_id),
            "geometry_mode": self.geometry_mode.value,
            "anchor_candidates": [candidate.as_dict() for candidate in self.anchor_candidates],
        }


@dataclass(frozen=True)
class HandoffZone:
    """An approved or explicitly rejected change between adjacent spans."""

    segment_id: int
    pair_index: int
    outgoing_anchor_source_index: int
    incoming_anchor_source_index: int
    accepted: bool
    reason: str
    cost_terms: Mapping[str, float | int | None]

    def __post_init__(self) -> None:
        if int(self.segment_id) < 0 or int(self.pair_index) < 0:
            raise ValueError("Handoff ids must be non-negative")
        if not self.reason:
            raise ValueError("Handoff reason must not be empty")
        for name, value in self.cost_terms.items():
            if value is not None:
                _finite(value, f"Handoff cost term {name}")

    def as_dict(self) -> dict[str, object]:
        return {
            "segment_id": int(self.segment_id),
            "pair_index": int(self.pair_index),
            "outgoing_anchor_source_index": int(self.outgoing_anchor_source_index),
            "incoming_anchor_source_index": int(self.incoming_anchor_source_index),
            "accepted": bool(self.accepted),
            "reason": self.reason,
            "cost_terms": dict(self.cost_terms),
        }


@dataclass(frozen=True)
class ForegroundSegment:
    """A connected set of evidence-backed foreground fragments."""

    segment_id: int
    fragment_refs: tuple[tuple[int, int], ...]
    span_ids: tuple[int, ...]
    geometry_mode: GeometryMode

    def as_dict(self) -> dict[str, object]:
        return {
            "segment_id": int(self.segment_id),
            "fragment_refs": [
                {"pair_index": int(pair), "component_label": int(label)}
                for pair, label in self.fragment_refs
            ],
            "span_ids": [int(value) for value in self.span_ids],
            "geometry_mode": self.geometry_mode.value,
        }


@dataclass(frozen=True)
class SegmentOwnerPlan:
    """The bounded plan compiled into existing GraphCut hard constraints."""

    component_owner_constraints: tuple[Mapping[int, int], ...]
    segments: tuple[ForegroundSegment, ...]
    spans: tuple[ForegroundSpan, ...]
    handoffs: tuple[HandoffZone, ...]
    rejected_associations: tuple[Mapping[str, object], ...]
    rejected_association_counts: Mapping[str, int]
    geometry_mode_counts: Mapping[str, int]
    structural_failure_reason: str | None = None

    @property
    def accepted(self) -> bool:
        return self.structural_failure_reason is None

    def as_dict(self) -> dict[str, object]:
        return {
            "backend": "foreground_segment_owner_plan_v2",
            "accepted": self.accepted,
            "structural_failure_reason": self.structural_failure_reason,
            "component_owner_constraints": [
                {str(label): int(owner) for label, owner in constraints.items()}
                for constraints in self.component_owner_constraints
            ],
            "segment_count": len(self.segments),
            "span_count": len(self.spans),
            "handoff_count": len(self.handoffs),
            "approved_handoff_count": sum(handoff.accepted for handoff in self.handoffs),
            "rejected_handoff_count": sum(not handoff.accepted for handoff in self.handoffs),
            "geometry_mode_counts": dict(self.geometry_mode_counts),
            "rejected_association_counts": dict(self.rejected_association_counts),
            "rejected_associations": [dict(value) for value in self.rejected_associations],
            "segments": [segment.as_dict() for segment in self.segments],
            "spans": [span.as_dict() for span in self.spans],
            "handoffs": [handoff.as_dict() for handoff in self.handoffs],
        }


@dataclass(frozen=True, eq=False)
class _CandidateLink:
    left: ForegroundFragment
    right: ForegroundFragment
    shared_source_index: int
    left_owner: int
    right_owner: int
    raw_footprint_iou: float
    orientation_delta_degrees: float
    width_ratio: float
    score: float

    @property
    def key(self) -> tuple[int, int, int, int]:
        return (
            self.left.pair_index,
            self.left.component_label,
            self.right.pair_index,
            self.right.component_label,
        )


_MINIMUM_RAW_FOOTPRINT_IOU = 0.35
_MAXIMUM_ORIENTATION_DELTA_DEGREES = 25.0
_MAXIMUM_WIDTH_RATIO = 1.50
_MAXIMUM_REJECTED_ASSOCIATION_EXAMPLES = 64


def _orientation_delta(first: float, second: float) -> float:
    return abs((first - second + 90.0) % 180.0 - 90.0)


def _candidate_link(
    left: ForegroundFragment, right: ForegroundFragment
) -> tuple[_CandidateLink | None, str | None]:
    """Validate all evidence required to join two adjacent pair fragments."""

    if right.pair_index != left.pair_index + 1:
        return None, "non_adjacent_pair"
    if left.geometry_mode is GeometryMode.RIGID_PROXY or right.geometry_mode is GeometryMode.RIGID_PROXY:
        return None, "rigid_proxy_forbidden_in_formal_owner_plan"
    if (
        left.geometry_mode is not GeometryMode.DEPTH_OBSERVED
        or right.geometry_mode is not GeometryMode.DEPTH_OBSERVED
    ):
        return None, "geometry_not_depth_observed"
    if not (left.bidirectional_visibility_supported and right.bidirectional_visibility_supported):
        return None, "missing_bidirectional_visibility"
    if left.natural_break_reason is not None or right.natural_break_reason is not None:
        return None, "natural_break"
    shared = sorted(set(left.source_indices) & set(right.source_indices))
    if len(shared) != 1:
        return None, "no_unique_shared_source"
    shared_source = shared[0]
    left_owner = left.local_owner_for_source(shared_source)
    right_owner = right.local_owner_for_source(shared_source)
    if left_owner is None or right_owner is None:
        return None, "shared_source_does_not_fully_cover_component"
    first_footprint = left.raw_footprints[left_owner]
    second_footprint = right.raw_footprints[right_owner]
    if first_footprint is None or second_footprint is None:
        return None, "missing_shared_raw_footprint"
    footprint_iou = first_footprint.overlap_iou(second_footprint)
    left_has_direct_anchor = left.depth_anchor_local_mask is not None
    right_has_direct_anchor = right.depth_anchor_local_mask is not None
    if left_has_direct_anchor != right_has_direct_anchor:
        return None, "incomplete_source_anchor_evidence"
    direct_source_anchor = left_has_direct_anchor and right_has_direct_anchor
    if direct_source_anchor:
        # A sparse anchor is allowed to bypass coarse guard shape heuristics
        # only when both pair-local labels carry the *same* exact raw
        # signed-occlusion identity.  The occupancy-grid IoU below is audit
        # information, not a substitute for this identity proof.
        left_token = left.depth_anchor_token
        right_token = right.depth_anchor_token
        if left_token is None or right_token is None:
            raise RuntimeError("Direct source anchor mask lacks its immutable token")
        if left_token != right_token:
            return None, "source_anchor_token_mismatch"
        if (
            left_token.shared_source_index != shared_source
            or left_token.left_pair_index != left.pair_index
            or left_token.right_pair_index != right.pair_index
        ):
            return None, "source_anchor_token_not_for_this_adjacent_pair"
    elif footprint_iou < _MINIMUM_RAW_FOOTPRINT_IOU:
        return None, "shared_raw_footprint_disjoint"
    if left.edge_orientation_degrees is None or right.edge_orientation_degrees is None:
        if not direct_source_anchor:
            return None, "missing_contour_orientation"
        # The direct raw signed-occlusion overlap is the identity proof for a
        # sparse source anchor.  A surrounding RGB guard need not itself have
        # one reliable contour orientation.
        orientation_delta = 0.0
    else:
        orientation_delta = _orientation_delta(
            float(left.edge_orientation_degrees), float(right.edge_orientation_degrees)
        )
    if not direct_source_anchor and orientation_delta > _MAXIMUM_ORIENTATION_DELTA_DEGREES:
        return None, "contour_orientation_discontinuity"
    width_ratio = max(
        float(left.local_width_pixels), float(right.local_width_pixels)
    ) / min(float(left.local_width_pixels), float(right.local_width_pixels))
    if not direct_source_anchor and width_ratio > _MAXIMUM_WIDTH_RATIO:
        return None, "local_width_discontinuity"
    # A deterministic scalar cost/score is enough in v2.0 because all
    # candidate anchors have already passed complete coverage.  Future
    # capture-time native depth/pose confidence can extend this score without
    # changing the owner contract.
    score = (
        1.0
        if direct_source_anchor
        else (
            0.70 * footprint_iou
            + 0.20 * (1.0 - orientation_delta / _MAXIMUM_ORIENTATION_DELTA_DEGREES)
            + 0.10 * (1.0 - (width_ratio - 1.0) / (_MAXIMUM_WIDTH_RATIO - 1.0))
        )
    )
    return (
        _CandidateLink(
            left=left,
            right=right,
            shared_source_index=shared_source,
            left_owner=left_owner,
            right_owner=right_owner,
            raw_footprint_iou=float(footprint_iou),
            orientation_delta_degrees=float(orientation_delta),
            width_ratio=float(width_ratio),
            score=float(np.clip(score, 0.0, 1.0)),
        ),
        None,
    )


def _candidate_components(
    candidates: Sequence[_CandidateLink],
) -> list[tuple[_CandidateLink, ...]]:
    """Partition unambiguous links into simple temporal chains."""

    by_node: dict[tuple[int, int], list[_CandidateLink]] = {}
    for link in candidates:
        by_node.setdefault(link.left.reference, []).append(link)
        by_node.setdefault(link.right.reference, []).append(link)
    unseen = set(candidates)
    components: list[tuple[_CandidateLink, ...]] = []
    while unseen:
        seed = min(unseen, key=lambda link: link.key)
        queue = [seed]
        component: set[_CandidateLink] = set()
        while queue:
            link = queue.pop()
            if link in component:
                continue
            component.add(link)
            unseen.discard(link)
            for node in (link.left.reference, link.right.reference):
                queue.extend(candidate for candidate in by_node[node] if candidate not in component)
        components.append(tuple(sorted(component, key=lambda link: link.key)))
    return components


def _maximum_weight_chain_matching(
    component: Sequence[_CandidateLink],
) -> tuple[_CandidateLink, ...]:
    """Use deterministic dynamic programming to choose non-conflicting spans."""

    if not component:
        return ()
    adjacency: dict[tuple[int, int], list[_CandidateLink]] = {}
    for link in component:
        adjacency.setdefault(link.left.reference, []).append(link)
        adjacency.setdefault(link.right.reference, []).append(link)
    if any(len(links) > 2 for links in adjacency.values()):
        raise RuntimeError("Foreground association component is not a simple chain")
    endpoints = sorted(node for node, links in adjacency.items() if len(links) == 1)
    if not endpoints:
        raise RuntimeError("Foreground association component has no temporal endpoint")
    node = endpoints[0]
    previous: _CandidateLink | None = None
    ordered: list[_CandidateLink] = []
    while True:
        next_links = [link for link in adjacency[node] if link is not previous]
        if not next_links:
            break
        link = next_links[0]
        ordered.append(link)
        node = link.right.reference if link.left.reference == node else link.left.reference
        previous = link
    if len(ordered) != len(component):
        raise RuntimeError("Foreground association chain traversal was incomplete")

    # ``states[index]`` contains the best non-overlapping edge set among the
    # first ``index`` ordered edges.  Equal scores use lexicographic link keys
    # so repeated executions emit the same span/handoff plan.
    states: list[tuple[float, tuple[_CandidateLink, ...]]] = [(0.0, ())]
    for index, link in enumerate(ordered, start=1):
        skip = states[index - 1]
        base = states[index - 2] if index >= 2 else states[0]
        take = (base[0] + link.score, (*base[1], link))
        if take[0] > skip[0] + 1e-12:
            states.append(take)
        elif skip[0] > take[0] + 1e-12:
            states.append(skip)
        else:
            take_key = tuple(candidate.key for candidate in take[1])
            skip_key = tuple(candidate.key for candidate in skip[1])
            states.append(take if take_key < skip_key else skip)
    return states[-1][1]


def _constraints_are_monotonic(
    fragments: Sequence[ForegroundFragment], constraints: Mapping[int, int]
) -> bool:
    """Mirror the renderer's protected-component monotonic owner invariant."""

    lower_by_row: dict[int, int] = {}
    upper_by_row: dict[int, int] = {}
    for fragment in fragments:
        owner = constraints.get(fragment.component_label)
        if owner is None:
            continue
        if owner not in {0, 1}:
            return False
        x0, y0, _, _ = fragment.global_bbox
        for local_y in np.flatnonzero(np.any(fragment.local_mask, axis=1)):
            local_x = np.flatnonzero(fragment.local_mask[int(local_y)])
            if not local_x.size:
                continue
            row = int(y0 + int(local_y))
            if owner == 0:
                lower_by_row[row] = max(
                    lower_by_row.get(row, -1), int(x0 + int(local_x[-1]))
                )
            else:
                upper_by_row[row] = min(
                    upper_by_row.get(row, int(x0 + fragment.local_mask.shape[1]) - 1),
                    int(x0 + int(local_x[0]) - 1),
                )
    return all(
        lower <= upper_by_row.get(row, lower) for row, lower in lower_by_row.items()
    )


def _span_for_link(
    *,
    segment_id: int,
    span_id: int,
    link: _CandidateLink,
) -> ForegroundSpan:
    source = link.shared_source_index
    left_frame = link.left.frame_ids[link.left_owner]
    right_frame = link.right.frame_ids[link.right_owner]
    if left_frame != right_frame:
        raise RuntimeError("Shared foreground anchor maps to different frame ids")
    candidate = AnchorCandidate(
        source_index=source,
        frame_id=left_frame,
        complete_coverage=True,
        coverage_margin_pixels=None,
        score=link.score,
    )
    return ForegroundSpan(
        segment_id=segment_id,
        span_id=span_id,
        fragment_refs=(link.left.reference, link.right.reference),
        anchor_source_index=source,
        anchor_frame_id=left_frame,
        geometry_mode=GeometryMode.DEPTH_OBSERVED,
        anchor_candidates=(candidate,),
    )


def _singleton_span(
    *,
    segment_id: int,
    span_id: int,
    fragment: ForegroundFragment,
) -> ForegroundSpan:
    owner = (
        fragment.preferred_local_owner
        if fragment.preferred_local_owner in fragment.allowed_local_owners
        else fragment.allowed_local_owners[0]
    )
    candidate = AnchorCandidate(
        source_index=fragment.source_indices[owner],
        frame_id=fragment.frame_ids[owner],
        complete_coverage=True,
        coverage_margin_pixels=None,
        score=0.0,
    )
    return ForegroundSpan(
        segment_id=segment_id,
        span_id=span_id,
        fragment_refs=(fragment.reference,),
        anchor_source_index=candidate.source_index,
        anchor_frame_id=candidate.frame_id,
        geometry_mode=fragment.geometry_mode,
        anchor_candidates=(candidate,),
    )


def plan_foreground_owners(
    fragments_by_pair: Sequence[Sequence[ForegroundFragment]],
) -> SegmentOwnerPlan:
    """Build deterministic spans and only safe hard-owner constraints.

    Connected long foreground evidence is solved as a maximum-weight matching
    over temporal association chains.  A shared component cannot be assigned
    to two different anchors, so an unapproved in-object handoff is retained
    as a rejected audit record rather than being hidden by a blend or warp.
    """

    mode_counts = {mode.value: 0 for mode in GeometryMode}
    for pair_index, fragments in enumerate(fragments_by_pair):
        for fragment in fragments:
            if fragment.pair_index != pair_index:
                return SegmentOwnerPlan(
                    component_owner_constraints=tuple({} for _ in fragments_by_pair),
                    segments=(),
                    spans=(),
                    handoffs=(),
                    rejected_associations=(),
                    rejected_association_counts={},
                    geometry_mode_counts=mode_counts,
                    structural_failure_reason="foreground_fragment_pair_order_mismatch",
                )
            mode_counts[fragment.geometry_mode.value] += 1

    rejected_counts: dict[str, int] = {}
    rejected_examples: list[dict[str, object]] = []

    def reject(
        reason: str, left: ForegroundFragment, right: ForegroundFragment
    ) -> None:
        rejected_counts[reason] = rejected_counts.get(reason, 0) + 1
        if len(rejected_examples) < _MAXIMUM_REJECTED_ASSOCIATION_EXAMPLES:
            rejected_examples.append(
                {
                    "reason": reason,
                    "left": {
                        "pair_index": left.pair_index,
                        "component_label": left.component_label,
                    },
                    "right": {
                        "pair_index": right.pair_index,
                        "component_label": right.component_label,
                    },
                }
            )

    raw_candidates: list[_CandidateLink] = []
    for pair_index in range(max(0, len(fragments_by_pair) - 1)):
        for left in fragments_by_pair[pair_index]:
            for right in fragments_by_pair[pair_index + 1]:
                candidate, reason = _candidate_link(left, right)
                if candidate is None:
                    # IMAGE_REGION/UNKNOWN fragments are intentionally not
                    # expanded into a cartesian report: their owner-only
                    # classification is already visible in mode_counts.
                    if reason not in {"geometry_not_depth_observed", "missing_bidirectional_visibility"}:
                        reject(str(reason), left, right)
                    continue
                raw_candidates.append(candidate)

    # A split/merge remains a natural association break.  Do not let a
    # maximum matching choose an arbitrary branch simply because both branches
    # have a similar raw-footprint score.
    outgoing_incidence: dict[tuple[int, int], int] = {}
    incoming_incidence: dict[tuple[int, int], int] = {}
    for candidate in raw_candidates:
        outgoing_incidence[candidate.left.reference] = (
            outgoing_incidence.get(candidate.left.reference, 0) + 1
        )
        incoming_incidence[candidate.right.reference] = (
            incoming_incidence.get(candidate.right.reference, 0) + 1
        )
    candidates: list[_CandidateLink] = []
    for candidate in raw_candidates:
        if (
            outgoing_incidence[candidate.left.reference] != 1
            or incoming_incidence[candidate.right.reference] != 1
        ):
            reject("split_merge_or_multiple_candidate_association", candidate.left, candidate.right)
            continue
        candidates.append(candidate)

    constraints: list[dict[int, int]] = [dict() for _ in fragments_by_pair]
    # A component with exactly one complete RGB owner is hard-constrained even
    # without depth association.  This duplicates the renderer's existing
    # fail-closed local rule and does not introduce a new colour path.
    for pair_index, fragments in enumerate(fragments_by_pair):
        for fragment in fragments:
            if len(fragment.allowed_local_owners) == 1:
                constraints[pair_index][fragment.component_label] = fragment.allowed_local_owners[0]

    selected: set[_CandidateLink] = set()
    for component in _candidate_components(candidates):
        try:
            selected.update(_maximum_weight_chain_matching(component))
        except RuntimeError:
            for candidate in component:
                reject("association_graph_not_a_simple_temporal_chain", candidate.left, candidate.right)

    active_links: list[_CandidateLink] = []
    for candidate in sorted(selected, key=lambda item: (-item.score, item.key)):
        proposed_left = dict(constraints[candidate.left.pair_index])
        proposed_right = dict(constraints[candidate.right.pair_index])
        proposed_left[candidate.left.component_label] = candidate.left_owner
        proposed_right[candidate.right.component_label] = candidate.right_owner
        if not (
            _constraints_are_monotonic(
                fragments_by_pair[candidate.left.pair_index], proposed_left
            )
            and _constraints_are_monotonic(
                fragments_by_pair[candidate.right.pair_index], proposed_right
            )
        ):
            reject("monotonic_owner_topology_conflict", candidate.left, candidate.right)
            continue
        constraints[candidate.left.pair_index] = proposed_left
        constraints[candidate.right.pair_index] = proposed_right
        active_links.append(candidate)

    # Each connected candidate component becomes one segment.  Selected links
    # create two-fragment anchored spans; unmatched nodes remain owner-only
    # singleton spans.  This is a deliberate no-fabrication fallback.
    component_nodes: list[set[tuple[int, int]]] = []
    for component in _candidate_components(candidates):
        component_nodes.append(
            {node for link in component for node in (link.left.reference, link.right.reference)}
        )
    all_fragments = {
        fragment.reference: fragment
        for fragments in fragments_by_pair
        for fragment in fragments
    }
    covered = {reference for nodes in component_nodes for reference in nodes}
    component_nodes.extend({reference} for reference in sorted(set(all_fragments) - covered))
    active_set = set(active_links)
    spans: list[ForegroundSpan] = []
    segments: list[ForegroundSegment] = []
    handoffs: list[HandoffZone] = []
    span_id = 0
    for segment_id, nodes in enumerate(sorted(component_nodes, key=lambda value: min(value))):
        links = [
            link
            for link in candidates
            if link.left.reference in nodes and link.right.reference in nodes
        ]
        active = [link for link in links if link in active_set]
        segment_span_ids: list[int] = []
        anchored_nodes: set[tuple[int, int]] = set()
        for link in sorted(active, key=lambda item: item.key):
            span = _span_for_link(segment_id=segment_id, span_id=span_id, link=link)
            spans.append(span)
            segment_span_ids.append(span_id)
            span_id += 1
            anchored_nodes.update((link.left.reference, link.right.reference))
        for reference in sorted(nodes - anchored_nodes):
            span = _singleton_span(
                segment_id=segment_id,
                span_id=span_id,
                fragment=all_fragments[reference],
            )
            spans.append(span)
            segment_span_ids.append(span_id)
            span_id += 1
        for link in sorted(links, key=lambda item: item.key):
            if link in active_set:
                continue
            handoffs.append(
                HandoffZone(
                    segment_id=segment_id,
                    pair_index=link.right.pair_index,
                    outgoing_anchor_source_index=link.shared_source_index,
                    incoming_anchor_source_index=link.right.source_indices[1],
                    accepted=False,
                    reason="continuous_foreground_requires_unapproved_handoff",
                    cost_terms={
                        "raw_footprint_iou": link.raw_footprint_iou,
                        "orientation_delta_degrees": link.orientation_delta_degrees,
                        "width_ratio": link.width_ratio,
                        "association_score": link.score,
                    },
                )
            )
        segment_mode = (
            GeometryMode.DEPTH_OBSERVED
            if any(all_fragments[reference].geometry_mode is GeometryMode.DEPTH_OBSERVED for reference in nodes)
            else all_fragments[min(nodes)].geometry_mode
        )
        segments.append(
            ForegroundSegment(
                segment_id=segment_id,
                fragment_refs=tuple(sorted(nodes)),
                span_ids=tuple(segment_span_ids),
                geometry_mode=segment_mode,
            )
        )

    return SegmentOwnerPlan(
        component_owner_constraints=tuple(constraints),
        segments=tuple(segments),
        spans=tuple(spans),
        handoffs=tuple(handoffs),
        rejected_associations=tuple(rejected_examples),
        rejected_association_counts=rejected_counts,
        geometry_mode_counts=mode_counts,
    )
