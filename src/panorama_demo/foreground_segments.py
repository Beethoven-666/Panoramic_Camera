"""Fail-closed foreground track and owner-run planning for RGB decisions.

The calibrated pushbroom renderer remains responsible for the only RGB remap,
GraphCut, hard-owner write, and safe-wall MultiBand blend.  This module is a
pure planning layer placed before that renderer's owner solve.  It never
changes a pose, generates a pixel, or accepts a foreground warp.

The planner is intentionally conservative.  A cross-pair foreground
association is usable only with bidirectional depth visibility plus either an
exact, complete direct-token bundle for that one protected-component pair or a
shared source-frame raw-footprint summary.  The result is a temporal identity
chain plus a deterministic sequence of RGB owner runs.  It never turns a
chain into a pose, a warp, a blend, or a source outside the adjacent pair.
Aligned-depth legacy inputs therefore remain ``IMAGE_REGION`` owner-only
fragments: they are audited, but cannot fabricate a long-range identity or
loosen the existing protected-component rule.
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
    depth_anchor_tokens: tuple[DepthAnchorToken, ...] = ()
    allowed_local_owners: tuple[int, ...] = (0, 1)
    preferred_local_owner: int | None = None
    edge_orientation_degrees: float | None = None
    local_width_pixels: float | None = None
    owner_coverage_margins_pixels: tuple[float | None, float | None] = (None, None)
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
        try:
            tokens = tuple(self.depth_anchor_tokens)
        except TypeError as exc:
            raise ValueError("Foreground depth anchor tokens must be an iterable") from exc
        if any(not isinstance(token, DepthAnchorToken) for token in tokens):
            raise ValueError("Foreground depth anchor tokens have the wrong type")
        if self.depth_anchor_token is not None and self.depth_anchor_token not in tokens:
            tokens = (*tokens, self.depth_anchor_token)
        token_keys = {
            (
                token.shared_source_index,
                token.left_pair_index,
                token.right_pair_index,
                token.left_direct_component_label,
                token.right_direct_component_label,
            )
            for token in tokens
        }
        if len(token_keys) != len(tokens):
            raise ValueError("Foreground depth anchor tokens must be unique")
        tokens = tuple(
            sorted(
                tokens,
                key=lambda token: (
                    token.left_pair_index,
                    token.right_pair_index,
                    token.shared_source_index,
                    token.left_direct_component_label,
                    token.right_direct_component_label,
                ),
            )
        )
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
        if len(self.owner_coverage_margins_pixels) != 2:
            raise ValueError("Foreground fragment needs two owner coverage margins")
        coverage_margins: tuple[float | None, float | None] = tuple(
            None
            if margin is None
            else _finite(margin, "Foreground fragment owner coverage margin")
            for margin in self.owner_coverage_margins_pixels
        )  # type: ignore[assignment]
        if any(margin is not None and margin < 0.0 for margin in coverage_margins):
            raise ValueError("Foreground fragment owner coverage margins must be non-negative")
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
        object.__setattr__(self, "depth_anchor_tokens", tokens)
        object.__setattr__(self, "allowed_local_owners", owners)
        object.__setattr__(
            self,
            "preferred_local_owner",
            None if self.preferred_local_owner is None else int(self.preferred_local_owner),
        )
        if width_pixels is None:
            width_pixels = float(min(width, height))
        object.__setattr__(self, "local_width_pixels", float(width_pixels))
        object.__setattr__(self, "owner_coverage_margins_pixels", coverage_margins)

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

    def edge_anchor_tokens_for_pair(
        self, *, left_pair_index: int, right_pair_index: int
    ) -> tuple[DepthAnchorToken, ...]:
        """Return direct-token evidence specifically declared for one pair edge.

        A middle observation may carry one exact token from its previous edge
        and another from its next edge.  Only the token whose own adjacent-pair
        interval matches the queried edge is relevant; treating all tokens on
        the observation as one global identity would incorrectly reject a
        legitimate multi-pair track.
        """

        return tuple(
            token
            for token in self.depth_anchor_tokens
            if token.left_pair_index == int(left_pair_index)
            and token.right_pair_index == int(right_pair_index)
        )

    def owner_coverage_margin(self, owner: int) -> float:
        """Return a scalar complete-coverage margin for owner-DP tie breaking."""

        if owner not in self.allowed_local_owners:
            raise ValueError("Foreground owner is not available for this fragment")
        margin = self.owner_coverage_margins_pixels[int(owner)]
        if margin is not None:
            return float(margin)
        footprint = self.raw_footprints[int(owner)]
        # A raw-footprint sample count is not a geometric margin.  It is only a
        # deterministic final fallback when the upstream component did not
        # provide one; the ordering above it remains direct depth evidence.
        return 0.0 if footprint is None else float(footprint.sample_count)

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
            "depth_anchor_tokens": [token.as_dict() for token in self.depth_anchor_tokens],
            "allowed_local_owners": [int(value) for value in self.allowed_local_owners],
            "preferred_local_owner": self.preferred_local_owner,
            "edge_orientation_degrees": self.edge_orientation_degrees,
            "local_width_pixels": float(self.local_width_pixels),
            "owner_coverage_margins_pixels": [
                None if margin is None else float(margin)
                for margin in self.owner_coverage_margins_pixels
            ],
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
    depth_anchor_tokens: Sequence[DepthAnchorToken] = (),
    owner_coverage_margins_pixels: tuple[float | None, float | None] | None = None,
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
        depth_anchor_tokens=tuple(depth_anchor_tokens),
        allowed_local_owners=fragment.allowed_owners,
        preferred_local_owner=fragment.preferred_owner,
        edge_orientation_degrees=fragment.edge_orientation,
        owner_coverage_margins_pixels=(
            (abs(float(fragment.coverage_margin)), abs(float(fragment.coverage_margin)))
            if owner_coverage_margins_pixels is None
            else owner_coverage_margins_pixels
        ),
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
    depth_anchor_token_sets: Mapping[
        tuple[int, int], Sequence[DepthAnchorToken]
    ] | None = None,
    owner_coverage_margins_pixels: Mapping[
        tuple[int, int], tuple[float | None, float | None]
    ] | None = None,
    raw_footprints: Mapping[tuple[int, int, int], RawFootprintSummary] | None = None,
    natural_breaks: Mapping[tuple[int, int], str] | None = None,
) -> tuple[tuple[ForegroundFragment, ...], ...]:
    """Adapt pair guards into a pure v3 planning input.

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
    anchor_token_sets = (
        {} if depth_anchor_token_sets is None else dict(depth_anchor_token_sets)
    )
    coverage_margins = (
        {} if owner_coverage_margins_pixels is None else dict(owner_coverage_margins_pixels)
    )
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
        | set(anchor_token_sets)
        | set(coverage_margins)
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
            legacy_token = anchor_tokens.get(reference)
            token_set = tuple(anchor_token_sets.get(reference, ()))
            if legacy_token is not None and legacy_token not in token_set:
                token_set = (*token_set, legacy_token)
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
                    depth_anchor_token=legacy_token,
                    depth_anchor_tokens=token_set,
                    owner_coverage_margins_pixels=coverage_margins.get(reference),
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
class ForegroundTrackEdge:
    """One strict identity observation between two neighbouring pair guards.

    The edge contains scalar evidence only.  In particular, a direct token is
    evidence for this *one* pair interval, not a global transform or a claim
    that the same RGB source can cover a later pair.
    """

    edge_id: int
    left_fragment_ref: tuple[int, int]
    right_fragment_ref: tuple[int, int]
    shared_source_index: int
    left_local_owner: int
    right_local_owner: int
    raw_footprint_iou: float | None
    contour_orientation_delta_degrees: float
    width_ratio: float
    centreline_delta_pixels: float
    direction_residual_degrees: float
    residual_cost: float
    association_score: float
    direct_anchor_token: DepthAnchorToken | None = None
    bidirectional_visibility_supported: bool = True
    direct_anchor_tokens: tuple[DepthAnchorToken, ...] = ()

    def __post_init__(self) -> None:
        if int(self.edge_id) < 0:
            raise ValueError("Foreground track edge_id must be non-negative")
        left_pair, left_label = (int(value) for value in self.left_fragment_ref)
        right_pair, right_label = (int(value) for value in self.right_fragment_ref)
        if (
            left_pair < 0
            or left_label < 1
            or right_pair != left_pair + 1
            or right_label < 1
            or int(self.shared_source_index) < 0
            or int(self.left_local_owner) not in {0, 1}
            or int(self.right_local_owner) not in {0, 1}
        ):
            raise ValueError("Foreground track edge must join one adjacent pair identity")
        raw_footprint_iou = (
            None
            if self.raw_footprint_iou is None
            else _finite(self.raw_footprint_iou, "Foreground track edge raw footprint IoU")
        )
        if raw_footprint_iou is not None and raw_footprint_iou < 0.0:
            raise ValueError("Foreground track edge raw footprint IoU must be non-negative")
        for name, value in (
            ("contour orientation delta", self.contour_orientation_delta_degrees),
            ("width ratio", self.width_ratio),
            ("centreline delta", self.centreline_delta_pixels),
            ("direction residual", self.direction_residual_degrees),
            ("residual cost", self.residual_cost),
            ("association score", self.association_score),
        ):
            if _finite(value, f"Foreground track edge {name}") < 0.0:
                raise ValueError(f"Foreground track edge {name} must be non-negative")
        if self.width_ratio < 1.0:
            raise ValueError("Foreground track edge width_ratio must be at least one")
        if not 0.0 <= float(self.association_score) <= 1.0:
            raise ValueError("Foreground track edge association_score must be in [0, 1]")
        try:
            direct_tokens = tuple(self.direct_anchor_tokens)
        except TypeError as exc:
            raise ValueError("Foreground track edge tokens must be an iterable") from exc
        if any(not isinstance(token, DepthAnchorToken) for token in direct_tokens):
            raise ValueError("Foreground track edge tokens have the wrong type")
        legacy_token = self.direct_anchor_token
        if legacy_token is not None and not isinstance(legacy_token, DepthAnchorToken):
            raise ValueError("Foreground track edge token has the wrong type")
        if not direct_tokens and legacy_token is not None:
            direct_tokens = (legacy_token,)
        elif legacy_token is not None and legacy_token not in direct_tokens:
            raise ValueError("Foreground track edge canonical token is not in its token bundle")
        if len(set(direct_tokens)) != len(direct_tokens):
            raise ValueError("Foreground track edge tokens must be unique")
        direct_tokens = tuple(
            sorted(
                direct_tokens,
                key=lambda token: (
                    token.left_pair_index,
                    token.right_pair_index,
                    token.shared_source_index,
                    token.left_direct_component_label,
                    token.right_direct_component_label,
                ),
            )
        )
        for token in direct_tokens:
            if (
                token.left_pair_index != left_pair
                or token.right_pair_index != right_pair
                or token.shared_source_index != int(self.shared_source_index)
            ):
                raise ValueError("Foreground track edge token does not match its interval")
        if raw_footprint_iou is None and not direct_tokens:
            raise ValueError(
                "Foreground track edge without raw footprint evidence needs direct tokens"
            )
        if type(self.bidirectional_visibility_supported) is not bool:
            raise ValueError("Foreground track edge visibility support must be boolean")
        object.__setattr__(self, "edge_id", int(self.edge_id))
        object.__setattr__(self, "left_fragment_ref", (left_pair, left_label))
        object.__setattr__(self, "right_fragment_ref", (right_pair, right_label))
        object.__setattr__(self, "shared_source_index", int(self.shared_source_index))
        object.__setattr__(self, "left_local_owner", int(self.left_local_owner))
        object.__setattr__(self, "right_local_owner", int(self.right_local_owner))
        object.__setattr__(self, "raw_footprint_iou", raw_footprint_iou)
        object.__setattr__(self, "direct_anchor_tokens", direct_tokens)
        object.__setattr__(
            self,
            "direct_anchor_token",
            None if not direct_tokens else direct_tokens[0],
        )

    @property
    def direct_token_supported(self) -> bool:
        return bool(self.direct_anchor_tokens)

    def as_dict(self) -> dict[str, object]:
        return {
            "edge_id": int(self.edge_id),
            "left": {
                "pair_index": int(self.left_fragment_ref[0]),
                "component_label": int(self.left_fragment_ref[1]),
            },
            "right": {
                "pair_index": int(self.right_fragment_ref[0]),
                "component_label": int(self.right_fragment_ref[1]),
            },
            "shared_source_index": int(self.shared_source_index),
            "left_local_owner": int(self.left_local_owner),
            "right_local_owner": int(self.right_local_owner),
            "raw_footprint_iou": (
                None
                if self.raw_footprint_iou is None
                else float(self.raw_footprint_iou)
            ),
            "contour_orientation_delta_degrees": float(
                self.contour_orientation_delta_degrees
            ),
            "width_ratio": float(self.width_ratio),
            "centreline_delta_pixels": float(self.centreline_delta_pixels),
            "direction_residual_degrees": float(self.direction_residual_degrees),
            "residual_cost": float(self.residual_cost),
            "association_score": float(self.association_score),
            "direct_token_supported": self.direct_token_supported,
            "direct_anchor_token": (
                None if self.direct_anchor_token is None else self.direct_anchor_token.as_dict()
            ),
            "direct_anchor_tokens": [
                token.as_dict() for token in self.direct_anchor_tokens
            ],
            "direct_anchor_token_count": len(self.direct_anchor_tokens),
            "bidirectional_visibility_supported": self.bidirectional_visibility_supported,
        }


@dataclass(frozen=True)
class ForegroundInstanceTrack:
    """A maximal unambiguous temporal chain of foreground observations."""

    track_id: int
    fragment_refs: tuple[tuple[int, int], ...]
    edge_ids: tuple[int, ...]
    geometry_mode: GeometryMode
    direct_token_edge_count: int
    bidirectional_edge_count: int
    association_score: float

    def __post_init__(self) -> None:
        if int(self.track_id) < 0 or len(self.fragment_refs) < 2:
            raise ValueError("Foreground instance track needs an id and two observations")
        refs = tuple((int(pair), int(label)) for pair, label in self.fragment_refs)
        if any(pair < 0 or label < 1 for pair, label in refs) or any(
            later[0] != earlier[0] + 1 for earlier, later in zip(refs, refs[1:])
        ):
            raise ValueError("Foreground instance track observations must be consecutive")
        edge_ids = tuple(int(edge_id) for edge_id in self.edge_ids)
        if len(edge_ids) != len(refs) - 1 or len(set(edge_ids)) != len(edge_ids) or min(edge_ids) < 0:
            raise ValueError("Foreground instance track edges must cover every interval once")
        if not isinstance(self.geometry_mode, GeometryMode):
            raise ValueError("Foreground instance track geometry_mode must be a GeometryMode")
        if not 0 <= int(self.direct_token_edge_count) <= len(edge_ids):
            raise ValueError("Foreground instance track direct token count is invalid")
        if not 0 <= int(self.bidirectional_edge_count) <= len(edge_ids):
            raise ValueError("Foreground instance track visibility count is invalid")
        if not 0.0 <= _finite(self.association_score, "Foreground instance track score") <= 1.0:
            raise ValueError("Foreground instance track score must be in [0, 1]")
        object.__setattr__(self, "track_id", int(self.track_id))
        object.__setattr__(self, "fragment_refs", refs)
        object.__setattr__(self, "edge_ids", edge_ids)
        object.__setattr__(self, "direct_token_edge_count", int(self.direct_token_edge_count))
        object.__setattr__(self, "bidirectional_edge_count", int(self.bidirectional_edge_count))

    @property
    def start_pair_index(self) -> int:
        return int(self.fragment_refs[0][0])

    @property
    def end_pair_index(self) -> int:
        return int(self.fragment_refs[-1][0])

    def as_dict(self) -> dict[str, object]:
        return {
            "track_id": int(self.track_id),
            "fragment_refs": [
                {"pair_index": int(pair), "component_label": int(label)}
                for pair, label in self.fragment_refs
            ],
            "edge_ids": [int(edge_id) for edge_id in self.edge_ids],
            "start_pair_index": self.start_pair_index,
            "end_pair_index": self.end_pair_index,
            "geometry_mode": self.geometry_mode.value,
            "direct_token_edge_count": int(self.direct_token_edge_count),
            "bidirectional_edge_count": int(self.bidirectional_edge_count),
            "association_score": float(self.association_score),
        }


@dataclass(frozen=True)
class ForegroundOwnerRun:
    """One contiguous planned owner interval within a longer identity track."""

    run_id: int
    track_id: int
    fragment_refs: tuple[tuple[int, int], ...]
    owner_source_index: int
    owner_frame_id: int
    local_owners: tuple[int, ...]
    direct_token_edge_ids: tuple[int, ...] = ()
    bidirectional_edge_ids: tuple[int, ...] = ()
    coverage_margin_pixels: float = 0.0
    residual_cost: float = 0.0

    def __post_init__(self) -> None:
        if int(self.run_id) < 0 or int(self.track_id) < 0 or int(self.owner_source_index) < 0:
            raise ValueError("Foreground owner run ids and source must be non-negative")
        refs = tuple((int(pair), int(label)) for pair, label in self.fragment_refs)
        owners = tuple(int(owner) for owner in self.local_owners)
        if not refs or len(refs) != len(owners):
            raise ValueError("Foreground owner run needs one owner for every observation")
        if any(pair < 0 or label < 1 for pair, label in refs) or any(
            later[0] != earlier[0] + 1 for earlier, later in zip(refs, refs[1:])
        ):
            raise ValueError("Foreground owner run observations must be consecutive")
        if any(owner not in {0, 1} for owner in owners):
            raise ValueError("Foreground owner run local owners must be 0 or 1")
        direct_edge_ids = tuple(int(edge_id) for edge_id in self.direct_token_edge_ids)
        bidirectional_edge_ids = tuple(int(edge_id) for edge_id in self.bidirectional_edge_ids)
        if (
            len(set(direct_edge_ids)) != len(direct_edge_ids)
            or len(set(bidirectional_edge_ids)) != len(bidirectional_edge_ids)
            or any(edge_id < 0 for edge_id in (*direct_edge_ids, *bidirectional_edge_ids))
        ):
            raise ValueError("Foreground owner run edge ids must be unique and non-negative")
        if _finite(self.coverage_margin_pixels, "Foreground owner run coverage margin") < 0.0:
            raise ValueError("Foreground owner run coverage margin must be non-negative")
        if _finite(self.residual_cost, "Foreground owner run residual cost") < 0.0:
            raise ValueError("Foreground owner run residual cost must be non-negative")
        object.__setattr__(self, "run_id", int(self.run_id))
        object.__setattr__(self, "track_id", int(self.track_id))
        object.__setattr__(self, "fragment_refs", refs)
        object.__setattr__(self, "owner_source_index", int(self.owner_source_index))
        object.__setattr__(self, "owner_frame_id", int(self.owner_frame_id))
        object.__setattr__(self, "local_owners", owners)
        object.__setattr__(self, "direct_token_edge_ids", direct_edge_ids)
        object.__setattr__(self, "bidirectional_edge_ids", bidirectional_edge_ids)

    @property
    def source_index(self) -> int:
        """Compatibility-friendly short name for the selected RGB source."""

        return int(self.owner_source_index)

    @property
    def frame_id(self) -> int:
        return int(self.owner_frame_id)

    @property
    def start_pair_index(self) -> int:
        return int(self.fragment_refs[0][0])

    @property
    def end_pair_index(self) -> int:
        return int(self.fragment_refs[-1][0])

    def as_dict(self) -> dict[str, object]:
        return {
            "run_id": int(self.run_id),
            "track_id": int(self.track_id),
            "fragment_refs": [
                {"pair_index": int(pair), "component_label": int(label)}
                for pair, label in self.fragment_refs
            ],
            "start_pair_index": self.start_pair_index,
            "end_pair_index": self.end_pair_index,
            "owner_source_index": int(self.owner_source_index),
            "owner_frame_id": int(self.owner_frame_id),
            "local_owners": [int(owner) for owner in self.local_owners],
            "direct_token_edge_ids": [int(edge_id) for edge_id in self.direct_token_edge_ids],
            "bidirectional_edge_ids": [int(edge_id) for edge_id in self.bidirectional_edge_ids],
            "coverage_margin_pixels": float(self.coverage_margin_pixels),
            "residual_cost": float(self.residual_cost),
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
    track_edges: tuple[ForegroundTrackEdge, ...] = ()
    tracks: tuple[ForegroundInstanceTrack, ...] = ()
    owner_runs: tuple[ForegroundOwnerRun, ...] = ()
    fragment_track_ids: Mapping[tuple[int, int], int] = field(default_factory=dict)
    fragment_run_ids: Mapping[tuple[int, int], int] = field(default_factory=dict)
    fragment_owner_sources: Mapping[tuple[int, int], int] = field(default_factory=dict)
    actual_owner_switch_count: int = 0
    minimum_feasible_owner_switch_count: int = 0
    avoidable_owner_switch_count: int = 0
    current_valid_nonadjacent_owner_pixel_count: int = 0
    foreground_blend_pixel_count: int = 0
    foreground_deformation_pixel_count: int = 0

    @property
    def accepted(self) -> bool:
        return self.structural_failure_reason is None

    def owner_for_fragment(self, pair_index: int, component_label: int) -> dict[str, int] | None:
        """Return the fully explicit v3 owner/run assignment for one fragment."""

        reference = (int(pair_index), int(component_label))
        run_id = self.fragment_run_ids.get(reference)
        source_index = self.fragment_owner_sources.get(reference)
        if run_id is None or source_index is None:
            return None
        local_owner = self.component_owner_constraints[reference[0]].get(reference[1])
        if local_owner is None:
            return None
        result = {
            "pair_index": reference[0],
            "component_label": reference[1],
            "run_id": int(run_id),
            "source_index": int(source_index),
            "local_owner": int(local_owner),
        }
        track_id = self.fragment_track_ids.get(reference)
        if track_id is not None:
            result["track_id"] = int(track_id)
        return result

    def foreground_owner_continuity_summary(self) -> dict[str, int | str]:
        """Return scalar-only delivery summary for the formal foreground contract."""

        return {
            "backend": "foreground_segment_owner_plan_v3",
            "track_count": len(self.tracks),
            "multi_pair_track_count": len(self.tracks),
            "owner_run_count": len(self.owner_runs),
            "actual_owner_switch_count": int(self.actual_owner_switch_count),
            "minimum_feasible_owner_switch_count": int(
                self.minimum_feasible_owner_switch_count
            ),
            "avoidable_owner_switch_count": int(self.avoidable_owner_switch_count),
            "current_valid_nonadjacent_owner_pixel_count": int(
                self.current_valid_nonadjacent_owner_pixel_count
            ),
            "foreground_blend_pixel_count": int(self.foreground_blend_pixel_count),
            "foreground_deformation_pixel_count": int(self.foreground_deformation_pixel_count),
        }

    def as_dict(self) -> dict[str, object]:
        return {
            "backend": "foreground_segment_owner_plan_v3",
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
            "track_count": len(self.tracks),
            "multi_pair_track_count": len(self.tracks),
            "owner_run_count": len(self.owner_runs),
            "track_edges": [edge.as_dict() for edge in self.track_edges],
            "tracks": [track.as_dict() for track in self.tracks],
            "owner_runs": [run.as_dict() for run in self.owner_runs],
            "planned_fragment_owners": [
                assignment
                for reference in sorted(self.fragment_run_ids)
                if (assignment := self.owner_for_fragment(*reference)) is not None
            ],
            "foreground_owner_continuity_summary": self.foreground_owner_continuity_summary(),
        }


@dataclass(frozen=True, eq=False)
class _CandidateLink:
    left: ForegroundFragment
    right: ForegroundFragment
    shared_source_index: int
    left_owner: int
    right_owner: int
    raw_footprint_iou: float | None
    orientation_delta_degrees: float
    width_ratio: float
    score: float
    centreline_delta_pixels: float = 0.0
    direction_residual_degrees: float = 0.0
    residual_cost: float = 0.0
    direct_anchor_token: DepthAnchorToken | None = None
    bidirectional_visibility_supported: bool = True
    direct_anchor_tokens: tuple[DepthAnchorToken, ...] = ()

    @property
    def key(self) -> tuple[int, int, int, int]:
        return (
            self.left.pair_index,
            self.left.component_label,
            self.right.pair_index,
            self.right.component_label,
        )

    @property
    def direct_token_supported(self) -> bool:
        return bool(self.direct_anchor_tokens)


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
    # A fragment in the middle of a long track may carry one token for its
    # incoming edge and another for its outgoing edge.  Compare only evidence
    # explicitly declared for the interval under consideration; otherwise an
    # earlier token would incorrectly poison a later raw-footprint association.
    left_tokens = left.edge_anchor_tokens_for_pair(
        left_pair_index=left.pair_index,
        right_pair_index=right.pair_index,
    )
    right_tokens = right.edge_anchor_tokens_for_pair(
        left_pair_index=left.pair_index,
        right_pair_index=right.pair_index,
    )
    direct_tokens: tuple[DepthAnchorToken, ...] = ()
    if left_tokens or right_tokens:
        if not left_tokens or not right_tokens:
            return None, "incomplete_source_anchor_evidence"
        # Several exact raw signed-occlusion anchors can support one exact
        # protected-component pair.  They are redundant identity evidence,
        # not a split/merge.  The bundle must nevertheless be exactly the
        # same at both endpoints: a partial overlap could conceal a peer
        # route, so it remains fail-closed.
        if left_tokens != right_tokens:
            return None, "source_anchor_token_mismatch"
        direct_tokens = left_tokens
        if any(token.shared_source_index != shared_source for token in direct_tokens):
            return None, "source_anchor_token_not_for_this_adjacent_pair"
    direct_source_anchor = bool(direct_tokens)
    footprint_iou: float | None = None
    if not direct_source_anchor:
        first_footprint = left.raw_footprints[left_owner]
        second_footprint = right.raw_footprints[right_owner]
        if first_footprint is None or second_footprint is None:
            return None, "missing_shared_raw_footprint"
        footprint_iou = first_footprint.overlap_iou(second_footprint)
        if footprint_iou < _MINIMUM_RAW_FOOTPRINT_IOU:
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
    left_centreline = float(left.global_bbox[1]) + float(left.global_bbox[3]) * 0.5
    right_centreline = float(right.global_bbox[1]) + float(right.global_bbox[3]) * 0.5
    centreline_delta = abs(left_centreline - right_centreline)
    direction_residual = float(orientation_delta)
    residual_cost = (
        direction_residual
        + abs(float(width_ratio) - 1.0) * 10.0
        + centreline_delta * 0.10
    )
    # A deterministic scalar cost/score is enough in v2.0 because all
    # candidate anchors have already passed complete coverage.  Future
    # capture-time native depth/pose confidence can extend this score without
    # changing the owner contract.
    if direct_source_anchor:
        score = 1.0
    else:
        if footprint_iou is None:
            raise RuntimeError("Raw-footprint association lost its required IoU")
        score = (
            0.70 * footprint_iou
            + 0.20 * (1.0 - orientation_delta / _MAXIMUM_ORIENTATION_DELTA_DEGREES)
            + 0.10 * (1.0 - (width_ratio - 1.0) / (_MAXIMUM_WIDTH_RATIO - 1.0))
        )
    return (
        _CandidateLink(
            left=left,
            right=right,
            shared_source_index=shared_source,
            left_owner=left_owner,
            right_owner=right_owner,
            raw_footprint_iou=footprint_iou,
            orientation_delta_degrees=float(orientation_delta),
            width_ratio=float(width_ratio),
            score=float(np.clip(score, 0.0, 1.0)),
            centreline_delta_pixels=float(centreline_delta),
            direction_residual_degrees=float(direction_residual),
            residual_cost=float(residual_cost),
            direct_anchor_token=(None if not direct_tokens else direct_tokens[0]),
            bidirectional_visibility_supported=True,
            direct_anchor_tokens=direct_tokens,
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


def _plan_foreground_owners_v2_compat(
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


# v3 deliberately leaves the v2 function above in place for old pickled/debug
# traces, but formal callers resolve this definition below.  Keeping the old
# record types also gives renderer integrations a narrow, backwards-compatible
# bridge while they move to explicit tracks and owner runs.
_MAX_OWNER_PATHS_PER_TRACK = 32
_MAX_GLOBAL_OWNER_DP_STATES = 8192


@dataclass(frozen=True)
class _OwnerPath:
    source_indices: tuple[int, ...]
    local_owners: tuple[int, ...]
    switch_count: int
    direct_token_support_count: int
    bidirectional_support_count: int
    coverage_margin_pixels: float
    residual_cost: float

    @property
    def key(self) -> tuple[object, ...]:
        """Lexicographic formal priority for one complete owner sequence."""

        return (
            int(self.switch_count),
            -int(self.direct_token_support_count),
            -int(self.bidirectional_support_count),
            -float(self.coverage_margin_pixels),
            float(self.residual_cost),
            tuple(int(value) for value in self.source_indices),
            tuple(int(value) for value in self.local_owners),
        )


def _candidate_to_track_edge(edge_id: int, link: _CandidateLink) -> ForegroundTrackEdge:
    return ForegroundTrackEdge(
        edge_id=edge_id,
        left_fragment_ref=link.left.reference,
        right_fragment_ref=link.right.reference,
        shared_source_index=link.shared_source_index,
        left_local_owner=link.left_owner,
        right_local_owner=link.right_owner,
        raw_footprint_iou=link.raw_footprint_iou,
        contour_orientation_delta_degrees=link.orientation_delta_degrees,
        width_ratio=link.width_ratio,
        centreline_delta_pixels=link.centreline_delta_pixels,
        direction_residual_degrees=link.direction_residual_degrees,
        residual_cost=link.residual_cost,
        association_score=link.score,
        direct_anchor_token=link.direct_anchor_token,
        bidirectional_visibility_supported=link.bidirectional_visibility_supported,
        direct_anchor_tokens=link.direct_anchor_tokens,
    )


def _ordered_candidate_chain(
    component: Sequence[_CandidateLink],
) -> tuple[tuple[tuple[int, int], ...], tuple[_CandidateLink, ...]]:
    """Return one forward-only unambiguous chain or reject it fail-closed."""

    if not component:
        raise RuntimeError("Foreground association chain is empty")
    outgoing = {link.left.reference: link for link in component}
    incoming = {link.right.reference: link for link in component}
    if len(outgoing) != len(component) or len(incoming) != len(component):
        raise RuntimeError("Foreground association chain is not one-to-one")
    starts = sorted(set(outgoing) - set(incoming))
    if len(starts) != 1:
        raise RuntimeError("Foreground association chain has no unique start")
    refs: list[tuple[int, int]] = [starts[0]]
    links: list[_CandidateLink] = []
    while refs[-1] in outgoing:
        link = outgoing[refs[-1]]
        links.append(link)
        refs.append(link.right.reference)
    if len(links) != len(component) or len(set(refs)) != len(refs):
        raise RuntimeError("Foreground association chain traversal was incomplete")
    return tuple(refs), tuple(links)


def _owner_options(fragment: ForegroundFragment) -> tuple[tuple[int, int], ...]:
    return tuple(
        sorted(
            (
                (int(fragment.source_indices[owner]), int(owner))
                for owner in fragment.allowed_local_owners
            ),
            key=lambda value: (value[0], value[1]),
        )
    )


def _renderer_feasible_owner_transition(
    *,
    previous_source: int,
    source: int,
    incoming_pair_index: int,
    is_terminal_observation: bool,
) -> bool:
    """Return whether one DP transition can become a renderer owner boundary.

    A changed source emits a run boundary at the incoming pair, where the
    renderer can only audit the two current RGB sources.  Retaining a source
    is safe only for the terminal observation: it may cover its two adjacent
    pair guards, but cannot later switch after becoming non-adjacent.
    """

    if int(source) == int(previous_source):
        return bool(is_terminal_observation)
    pair_index = int(incoming_pair_index)
    return {int(previous_source), int(source)} == {pair_index, pair_index + 1}


def _best_owner_paths(
    fragments: Sequence[ForegroundFragment],
    links: Sequence[_CandidateLink],
) -> tuple[_OwnerPath, ...]:
    """Dynamic-program the bounded best source paths for one identity track."""

    if len(fragments) < 2 or len(links) != len(fragments) - 1:
        raise RuntimeError("Foreground owner DP needs a complete multi-pair track")
    states: dict[int, list[_OwnerPath]] = {}
    first = fragments[0]
    for source, owner in _owner_options(first):
        states.setdefault(source, []).append(
            _OwnerPath(
                source_indices=(source,),
                local_owners=(owner,),
                switch_count=0,
                direct_token_support_count=0,
                bidirectional_support_count=0,
                coverage_margin_pixels=first.owner_coverage_margin(owner),
                residual_cost=0.0,
            )
        )
    for index, (fragment, link) in enumerate(zip(fragments[1:], links), start=1):
        next_states: dict[int, list[_OwnerPath]] = {}
        for previous_paths in states.values():
            for previous in previous_paths:
                previous_source = previous.source_indices[-1]
                for source, owner in _owner_options(fragment):
                    if not _renderer_feasible_owner_transition(
                        previous_source=previous_source,
                        source=source,
                        incoming_pair_index=fragment.pair_index,
                        is_terminal_observation=index == len(fragments) - 1,
                    ):
                        continue
                    next_states.setdefault(source, []).append(
                        _OwnerPath(
                            source_indices=(*previous.source_indices, source),
                            local_owners=(*previous.local_owners, owner),
                            switch_count=previous.switch_count
                            + int(source != previous_source),
                            direct_token_support_count=previous.direct_token_support_count
                            + int(link.direct_token_supported),
                            bidirectional_support_count=previous.bidirectional_support_count
                            + int(link.bidirectional_visibility_supported),
                            coverage_margin_pixels=previous.coverage_margin_pixels
                            + fragment.owner_coverage_margin(owner),
                            residual_cost=previous.residual_cost
                            + link.residual_cost,
                        )
                    )
        states = {}
        for source, paths in next_states.items():
            unique: dict[tuple[tuple[int, ...], tuple[int, ...]], _OwnerPath] = {}
            for path in paths:
                identity = (path.source_indices, path.local_owners)
                existing = unique.get(identity)
                if existing is None or path.key < existing.key:
                    unique[identity] = path
            states[source] = sorted(unique.values(), key=lambda path: path.key)[
                :_MAX_OWNER_PATHS_PER_TRACK
            ]
        if not states:
            raise RuntimeError(f"Foreground owner DP has no owner at track position {index}")
    unique_paths: dict[tuple[tuple[int, ...], tuple[int, ...]], _OwnerPath] = {}
    for paths in states.values():
        for path in paths:
            identity = (path.source_indices, path.local_owners)
            existing = unique_paths.get(identity)
            if existing is None or path.key < existing.key:
                unique_paths[identity] = path
    return tuple(sorted(unique_paths.values(), key=lambda path: path.key)[:_MAX_OWNER_PATHS_PER_TRACK])


def _merge_owner_path_constraints(
    constraints: Sequence[Mapping[int, int]],
    fragments: Sequence[ForegroundFragment],
    path: _OwnerPath,
) -> list[dict[int, int]] | None:
    """Merge one track path only if it preserves pair-local topology."""

    merged = [dict(values) for values in constraints]
    changed_pairs: set[int] = set()
    for fragment, owner in zip(fragments, path.local_owners):
        pair_constraints = merged[fragment.pair_index]
        existing = pair_constraints.get(fragment.component_label)
        if existing is not None and existing != owner:
            return None
        pair_constraints[fragment.component_label] = owner
        changed_pairs.add(fragment.pair_index)
    for pair_index in changed_pairs:
        pair_fragments = [
            fragment
            for fragment in fragments
            if fragment.pair_index == pair_index
        ]
        # The caller supplies the full pair separately below when needed.  This
        # local check catches duplicate track assignments; full topology is
        # checked by the v3 caller against every guard in the pair.
        if not _constraints_are_monotonic(pair_fragments, merged[pair_index]):
            return None
    return merged


def _v3_failure_plan(
    *,
    pair_count: int,
    mode_counts: Mapping[str, int],
    reason: str,
    rejected_associations: Sequence[Mapping[str, object]] = (),
    rejected_association_counts: Mapping[str, int] | None = None,
    track_edges: Sequence[ForegroundTrackEdge] = (),
    tracks: Sequence[ForegroundInstanceTrack] = (),
) -> SegmentOwnerPlan:
    return SegmentOwnerPlan(
        component_owner_constraints=tuple({} for _ in range(pair_count)),
        segments=(),
        spans=(),
        handoffs=(),
        rejected_associations=tuple(rejected_associations),
        rejected_association_counts=(
            {} if rejected_association_counts is None else dict(rejected_association_counts)
        ),
        geometry_mode_counts=dict(mode_counts),
        structural_failure_reason=reason,
        track_edges=tuple(track_edges),
        tracks=tuple(tracks),
    )


def plan_foreground_owners(
    fragments_by_pair: Sequence[Sequence[ForegroundFragment]],
) -> SegmentOwnerPlan:
    """Build fail-closed multi-pair tracks and a deterministic owner-run plan.

    Strict adjacent evidence creates identity edges.  Any split, merge, or
    competing identity removes every affected edge rather than selecting a
    convenient branch.  For every remaining temporal chain, dynamic
    programming first preserves complete coverage and owner topology, then
    minimizes actual RGB source changes, maximizes direct-token and
    bidirectional support, maximizes coverage margin, minimizes residuals,
    and finally uses source/frame order as a stable tie-breaker.
    """

    mode_counts = {mode.value: 0 for mode in GeometryMode}
    all_fragments: dict[tuple[int, int], ForegroundFragment] = {}
    for pair_index, pair_fragments in enumerate(fragments_by_pair):
        for fragment in pair_fragments:
            if fragment.pair_index != pair_index:
                return _v3_failure_plan(
                    pair_count=len(fragments_by_pair),
                    mode_counts=mode_counts,
                    reason="foreground_fragment_pair_order_mismatch",
                )
            if fragment.source_indices != (pair_index, pair_index + 1):
                return _v3_failure_plan(
                    pair_count=len(fragments_by_pair),
                    mode_counts=mode_counts,
                    reason="foreground_fragment_nonadjacent_source_pair",
                )
            if fragment.reference in all_fragments:
                return _v3_failure_plan(
                    pair_count=len(fragments_by_pair),
                    mode_counts=mode_counts,
                    reason="duplicate_foreground_fragment_reference",
                )
            all_fragments[fragment.reference] = fragment
            mode_counts[fragment.geometry_mode.value] += 1

    rejected_counts: dict[str, int] = {}
    rejected_examples: list[dict[str, object]] = []

    def reject(reason: str, left: ForegroundFragment, right: ForegroundFragment) -> None:
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
                    if reason not in {
                        "geometry_not_depth_observed",
                        "missing_bidirectional_visibility",
                    }:
                        reject(str(reason), left, right)
                    continue
                raw_candidates.append(candidate)

    # No branch selection is allowed.  Removing all links touching a split or
    # merge turns it into independent owner-only observations, never a guessed
    # long-range identity.
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

    candidates.sort(key=lambda candidate: candidate.key)
    candidate_edge_ids = {candidate: edge_id for edge_id, candidate in enumerate(candidates)}
    track_edges = tuple(
        _candidate_to_track_edge(candidate_edge_ids[candidate], candidate)
        for candidate in candidates
    )

    track_links: dict[int, tuple[_CandidateLink, ...]] = {}
    track_fragments: dict[int, tuple[ForegroundFragment, ...]] = {}
    tracks: list[ForegroundInstanceTrack] = []
    fragment_track_ids: dict[tuple[int, int], int] = {}
    try:
        components = sorted(
            _candidate_components(candidates),
            key=lambda component: min(candidate.key for candidate in component),
        )
        for track_id, component in enumerate(components):
            refs, links = _ordered_candidate_chain(component)
            fragments = tuple(all_fragments[reference] for reference in refs)
            edge_ids = tuple(candidate_edge_ids[link] for link in links)
            track = ForegroundInstanceTrack(
                track_id=track_id,
                fragment_refs=refs,
                edge_ids=edge_ids,
                geometry_mode=GeometryMode.DEPTH_OBSERVED,
                direct_token_edge_count=sum(link.direct_token_supported for link in links),
                bidirectional_edge_count=sum(
                    link.bidirectional_visibility_supported for link in links
                ),
                association_score=float(np.mean([link.score for link in links])),
            )
            tracks.append(track)
            track_links[track_id] = links
            track_fragments[track_id] = fragments
            fragment_track_ids.update({reference: track_id for reference in refs})
    except (KeyError, RuntimeError, ValueError) as exc:
        return _v3_failure_plan(
            pair_count=len(fragments_by_pair),
            mode_counts=mode_counts,
            reason=f"foreground_track_chain_construction_failed:{exc}",
            rejected_associations=rejected_examples,
            rejected_association_counts=rejected_counts,
            track_edges=track_edges,
            tracks=tracks,
        )

    constraints: list[dict[int, int]] = [dict() for _ in fragments_by_pair]
    for pair_index, pair_fragments in enumerate(fragments_by_pair):
        for fragment in pair_fragments:
            if len(fragment.allowed_local_owners) == 1:
                constraints[pair_index][fragment.component_label] = fragment.allowed_local_owners[0]
        if not _constraints_are_monotonic(pair_fragments, constraints[pair_index]):
            return _v3_failure_plan(
                pair_count=len(fragments_by_pair),
                mode_counts=mode_counts,
                reason="foreground_fixed_owner_topology_conflict",
                rejected_associations=rejected_examples,
                rejected_association_counts=rejected_counts,
                track_edges=track_edges,
                tracks=tracks,
            )

    owner_paths: dict[int, tuple[_OwnerPath, ...]] = {}
    try:
        for track in tracks:
            owner_paths[track.track_id] = _best_owner_paths(
                track_fragments[track.track_id], track_links[track.track_id]
            )
    except RuntimeError as exc:
        return _v3_failure_plan(
            pair_count=len(fragments_by_pair),
            mode_counts=mode_counts,
            reason=f"foreground_owner_dp_candidate_construction_failed:{exc}",
            rejected_associations=rejected_examples,
            rejected_association_counts=rejected_counts,
            track_edges=track_edges,
            tracks=tracks,
        )

    def merge_full_pair_topology(
        base: Sequence[Mapping[int, int]],
        track_id: int,
        path: _OwnerPath,
    ) -> list[dict[int, int]] | None:
        merged = [dict(values) for values in base]
        for fragment, owner in zip(track_fragments[track_id], path.local_owners):
            existing = merged[fragment.pair_index].get(fragment.component_label)
            if existing is not None and existing != owner:
                return None
            merged[fragment.pair_index][fragment.component_label] = owner
        changed_pairs = {fragment.pair_index for fragment in track_fragments[track_id]}
        for pair_index in changed_pairs:
            if not _constraints_are_monotonic(
                fragments_by_pair[pair_index], merged[pair_index]
            ):
                return None
        return merged

    ordered_track_ids = [track.track_id for track in tracks]
    selected_paths: dict[int, _OwnerPath] = {}
    chosen_constraints: list[dict[int, int]] | None = [dict(values) for values in constraints]
    # Fast path: independently optimal DP paths are globally optimal whenever
    # their complete pair topology is compatible.
    for track_id in ordered_track_ids:
        assert chosen_constraints is not None
        chosen_constraints = merge_full_pair_topology(
            chosen_constraints, track_id, owner_paths[track_id][0]
        )
        if chosen_constraints is None:
            break
        selected_paths[track_id] = owner_paths[track_id][0]

    if chosen_constraints is None:
        selected_paths = {}
        best: tuple[tuple[object, ...], dict[int, _OwnerPath], list[dict[int, int]]] | None = None
        state_count = 0
        exhausted = False

        def search(
            position: int,
            current_constraints: list[dict[int, int]],
            current_paths: dict[int, _OwnerPath],
        ) -> None:
            nonlocal best, state_count, exhausted
            if exhausted:
                return
            state_count += 1
            if state_count > _MAX_GLOBAL_OWNER_DP_STATES:
                exhausted = True
                return
            if position == len(ordered_track_ids):
                paths = [current_paths[track_id] for track_id in ordered_track_ids]
                key: tuple[object, ...] = (
                    sum(path.switch_count for path in paths),
                    -sum(path.direct_token_support_count for path in paths),
                    -sum(path.bidirectional_support_count for path in paths),
                    -sum(path.coverage_margin_pixels for path in paths),
                    sum(path.residual_cost for path in paths),
                    tuple(source for path in paths for source in path.source_indices),
                    tuple(owner for path in paths for owner in path.local_owners),
                )
                if best is None or key < best[0]:
                    best = (key, dict(current_paths), current_constraints)
                return
            track_id = ordered_track_ids[position]
            for path in owner_paths[track_id]:
                merged = merge_full_pair_topology(current_constraints, track_id, path)
                if merged is None:
                    continue
                current_paths[track_id] = path
                search(position + 1, merged, current_paths)
                current_paths.pop(track_id, None)

        search(0, [dict(values) for values in constraints], {})
        if exhausted:
            return _v3_failure_plan(
                pair_count=len(fragments_by_pair),
                mode_counts=mode_counts,
                reason="foreground_owner_dp_state_budget_exhausted",
                rejected_associations=rejected_examples,
                rejected_association_counts=rejected_counts,
                track_edges=track_edges,
                tracks=tracks,
            )
        if best is None:
            return _v3_failure_plan(
                pair_count=len(fragments_by_pair),
                mode_counts=mode_counts,
                reason="foreground_owner_dp_no_topology_safe_plan",
                rejected_associations=rejected_examples,
                rejected_association_counts=rejected_counts,
                track_edges=track_edges,
                tracks=tracks,
            )
        _, selected_paths, chosen_constraints = best

    assert chosen_constraints is not None
    owner_runs: list[ForegroundOwnerRun] = []
    spans: list[ForegroundSpan] = []
    segments: list[ForegroundSegment] = []
    handoffs: list[HandoffZone] = []
    fragment_run_ids: dict[tuple[int, int], int] = {}
    fragment_owner_sources: dict[tuple[int, int], int] = {}
    run_id = 0
    span_id = 0
    actual_switch_count = 0

    for track in tracks:
        track_id = track.track_id
        path = selected_paths[track_id]
        fragments = track_fragments[track_id]
        links = track_links[track_id]
        segment_span_ids: list[int] = []
        start = 0
        track_runs: list[ForegroundOwnerRun] = []
        while start < len(fragments):
            source = path.source_indices[start]
            end = start + 1
            while end < len(fragments) and path.source_indices[end] == source:
                end += 1
            run_fragments = fragments[start:end]
            run_refs = tuple(fragment.reference for fragment in run_fragments)
            run_owners = path.local_owners[start:end]
            frame_ids = tuple(
                fragment.frame_ids[owner]
                for fragment, owner in zip(run_fragments, run_owners)
            )
            if len(set(frame_ids)) != 1:
                return _v3_failure_plan(
                    pair_count=len(fragments_by_pair),
                    mode_counts=mode_counts,
                    reason="foreground_owner_run_source_frame_mismatch",
                    rejected_associations=rejected_examples,
                    rejected_association_counts=rejected_counts,
                    track_edges=track_edges,
                    tracks=tracks,
                )
            internal_links = links[start : end - 1]
            direct_edge_ids = tuple(
                candidate_edge_ids[link] for link in internal_links if link.direct_token_supported
            )
            bidirectional_edge_ids = tuple(
                candidate_edge_ids[link]
                for link in internal_links
                if link.bidirectional_visibility_supported
            )
            coverage_margin = min(
                fragment.owner_coverage_margin(owner)
                for fragment, owner in zip(run_fragments, run_owners)
            )
            residual_cost = float(sum(link.residual_cost for link in internal_links))
            run = ForegroundOwnerRun(
                run_id=run_id,
                track_id=track_id,
                fragment_refs=run_refs,
                owner_source_index=source,
                owner_frame_id=frame_ids[0],
                local_owners=run_owners,
                direct_token_edge_ids=direct_edge_ids,
                bidirectional_edge_ids=bidirectional_edge_ids,
                coverage_margin_pixels=coverage_margin,
                residual_cost=residual_cost,
            )
            owner_runs.append(run)
            track_runs.append(run)
            anchor = AnchorCandidate(
                source_index=source,
                frame_id=frame_ids[0],
                complete_coverage=True,
                coverage_margin_pixels=coverage_margin,
                score=(
                    0.0
                    if not internal_links
                    else float(np.mean([link.score for link in internal_links]))
                ),
            )
            spans.append(
                ForegroundSpan(
                    segment_id=track_id,
                    span_id=span_id,
                    fragment_refs=run_refs,
                    anchor_source_index=source,
                    anchor_frame_id=frame_ids[0],
                    geometry_mode=track.geometry_mode,
                    anchor_candidates=(anchor,),
                )
            )
            segment_span_ids.append(span_id)
            for fragment, owner in zip(run_fragments, run_owners):
                if fragment.source_indices[owner] != source:
                    return _v3_failure_plan(
                        pair_count=len(fragments_by_pair),
                        mode_counts=mode_counts,
                        reason="foreground_owner_run_selected_nonadjacent_source",
                        rejected_associations=rejected_examples,
                        rejected_association_counts=rejected_counts,
                        track_edges=track_edges,
                        tracks=tracks,
                    )
                fragment_run_ids[fragment.reference] = run_id
                fragment_owner_sources[fragment.reference] = source
            run_id += 1
            span_id += 1
            start = end
        for outgoing, incoming in zip(track_runs, track_runs[1:]):
            handoff_pair_index = int(incoming.start_pair_index)
            if (
                int(outgoing.end_pair_index) + 1 != handoff_pair_index
                or {
                    int(outgoing.owner_source_index),
                    int(incoming.owner_source_index),
                }
                != {handoff_pair_index, handoff_pair_index + 1}
            ):
                return _v3_failure_plan(
                    pair_count=len(fragments_by_pair),
                    mode_counts=mode_counts,
                    reason="foreground_owner_run_nonadjacent_handoff",
                    rejected_associations=rejected_examples,
                    rejected_association_counts=rejected_counts,
                    track_edges=track_edges,
                    tracks=tracks,
                )
            actual_switch_count += 1
            handoffs.append(
                HandoffZone(
                    segment_id=track_id,
                    pair_index=handoff_pair_index,
                    outgoing_anchor_source_index=outgoing.owner_source_index,
                    incoming_anchor_source_index=incoming.owner_source_index,
                    accepted=False,
                    reason="foreground_owner_run_requires_handoff_audit",
                    cost_terms={
                        "track_id": track_id,
                        "outgoing_run_id": outgoing.run_id,
                        "incoming_run_id": incoming.run_id,
                        "source_switch_count": 1,
                    },
                )
            )
        segments.append(
            ForegroundSegment(
                segment_id=track_id,
                fragment_refs=track.fragment_refs,
                span_ids=tuple(segment_span_ids),
                geometry_mode=track.geometry_mode,
            )
        )

    # Legacy single-pair guards remain owner-only.  They get a compatibility
    # span/segment but no invented track, run, or cross-pair handoff.
    tracked_refs = set(fragment_track_ids)
    next_segment_id = len(tracks)
    for reference in sorted(set(all_fragments) - tracked_refs):
        fragment = all_fragments[reference]
        span = _singleton_span(
            segment_id=next_segment_id,
            span_id=span_id,
            fragment=fragment,
        )
        spans.append(span)
        segments.append(
            ForegroundSegment(
                segment_id=next_segment_id,
                fragment_refs=(reference,),
                span_ids=(span_id,),
                geometry_mode=fragment.geometry_mode,
            )
        )
        span_id += 1
        next_segment_id += 1

    return SegmentOwnerPlan(
        component_owner_constraints=tuple(chosen_constraints),
        segments=tuple(segments),
        spans=tuple(spans),
        handoffs=tuple(handoffs),
        rejected_associations=tuple(rejected_examples),
        rejected_association_counts=rejected_counts,
        geometry_mode_counts=mode_counts,
        track_edges=track_edges,
        tracks=tuple(tracks),
        owner_runs=tuple(owner_runs),
        fragment_track_ids=fragment_track_ids,
        fragment_run_ids=fragment_run_ids,
        fragment_owner_sources=fragment_owner_sources,
        actual_owner_switch_count=actual_switch_count,
        minimum_feasible_owner_switch_count=actual_switch_count,
        avoidable_owner_switch_count=0,
        current_valid_nonadjacent_owner_pixel_count=0,
        foreground_blend_pixel_count=0,
        foreground_deformation_pixel_count=0,
    )
