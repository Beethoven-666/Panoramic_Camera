"""Candidate-only C8 five-frame monotone real-owner optimisation.

This module deliberately returns provenance only.  It receives a local,
chronologically ordered window of *already placed* :class:`VideoVisualSource`
objects and chooses one of those existing source ids at each valid canvas
pixel.  It does not return colour, touch poses, read files, warp samples, or
invent frames.  A renderer which consumes the resulting map must still copy a
pixel verbatim from the selected source.

The C8 constraint is local by construction: no more than five real sources are
accepted.  For every scan row the source index is non-decreasing from left to
right.  This is a useful fail-closed expression of temporal monotonicity for
the ordered side-scan canvas: a seam can move forward through the local window,
but cannot return to an earlier source.  Protected and object masks are fixed
owner constraints and make an infeasible order an error rather than silently
reordering a source or splitting an object.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from .video_visual_renderer import VideoVisualSource


_INVALID_OWNER = -1


@dataclass(frozen=True)
class MultilabelOwnerConfig:
    """Closed controls for the candidate-only local owner optimiser."""

    maximum_window_frames: int = 5
    owner_switch_penalty: float = 1.0
    centre_distance_weight: float = 1.0

    def __post_init__(self) -> None:
        if not 2 <= int(self.maximum_window_frames) <= 5:
            raise ValueError("maximum_window_frames must be in [2, 5]")
        if float(self.owner_switch_penalty) < 0.0:
            raise ValueError("owner_switch_penalty must be non-negative")
        if float(self.centre_distance_weight) < 0.0:
            raise ValueError("centre_distance_weight must be non-negative")


@dataclass(frozen=True)
class MultilabelOwnerAudit:
    """Auditable C8 provenance and constraint evidence."""

    source_frame_ids: tuple[int, ...]
    window_frame_count: int
    valid_pixel_count: int
    protected_owner_pixel_count: int
    object_owner_pixel_count: int
    fixed_owner_pixel_count: int
    per_source_owner_pixel_count: tuple[int, ...]
    temporal_monotonic_rows: int
    method: str = "candidate_c8_local_five_frame_multilabel_owner/v1"

    def as_dict(self) -> dict[str, object]:
        return {
            "method": self.method,
            "source_frame_ids": list(self.source_frame_ids),
            "window_frame_count": self.window_frame_count,
            "valid_pixel_count": self.valid_pixel_count,
            "protected_owner_pixel_count": self.protected_owner_pixel_count,
            "object_owner_pixel_count": self.object_owner_pixel_count,
            "fixed_owner_pixel_count": self.fixed_owner_pixel_count,
            "per_source_owner_pixel_count": {
                str(frame_id): count
                for frame_id, count in zip(self.source_frame_ids, self.per_source_owner_pixel_count)
            },
            "temporal_monotonic_rows": self.temporal_monotonic_rows,
            "creates_colour": False,
            "creates_pose": False,
            "interpolates_source_frames": False,
            "local_window_only": True,
            "maximum_window_frames": 5,
            "one_real_owner_per_valid_pixel": True,
            "temporal_order_constraint": True,
        }


@dataclass(frozen=True)
class MultilabelOwnerResult:
    """A strict owner partition, with no composed RGB output."""

    owner_frame_id: np.ndarray
    valid_mask: np.ndarray
    audit: MultilabelOwnerAudit


def _source_valid(source: VideoVisualSource) -> np.ndarray:
    return np.asarray(source.bgra)[..., 3] > 0


def _coerce_owner_constraint(
    value: np.ndarray | None,
    *,
    shape: tuple[int, int],
    source_ids: tuple[int, ...],
    name: str,
) -> np.ndarray:
    if value is None:
        return np.full(shape, _INVALID_OWNER, dtype=np.int32)
    owner = np.asarray(value)
    if owner.ndim != 2 or owner.shape != shape:
        raise ValueError(f"{name} must match the placed source canvas")
    if not np.issubdtype(owner.dtype, np.integer):
        raise ValueError(f"{name} must contain integer frame ids")
    result = np.asarray(owner, dtype=np.int32)
    legal = (result == _INVALID_OWNER) | np.isin(result, np.asarray(source_ids, dtype=np.int32))
    if not np.all(legal):
        raise ValueError(f"{name} may only name local real source frame ids")
    return result


def _coerce_object_constraints(
    value: Mapping[int, np.ndarray] | None,
    *,
    shape: tuple[int, int],
    source_ids: tuple[int, ...],
) -> tuple[np.ndarray, int]:
    """Convert ``{real_frame_id: mask}`` locks to one owner constraint map."""

    result = np.full(shape, _INVALID_OWNER, dtype=np.int32)
    if value is None:
        return result, 0
    for frame_id, raw_mask in value.items():
        if int(frame_id) not in source_ids:
            raise ValueError("object_masks may only name local real source frame ids")
        mask = np.asarray(raw_mask)
        if mask.ndim != 2 or mask.shape != shape:
            raise ValueError("every object mask must match the placed source canvas")
        if mask.dtype != bool and not np.issubdtype(mask.dtype, np.number):
            raise ValueError("object masks must be boolean or numeric")
        if np.issubdtype(mask.dtype, np.floating) and not np.isfinite(mask).all():
            raise ValueError("object masks must be finite")
        selected = np.asarray(mask != 0, dtype=bool)
        conflict = selected & (result != _INVALID_OWNER) & (result != int(frame_id))
        if np.any(conflict):
            raise ValueError("one object pixel cannot be locked to multiple real owners")
        result[selected] = int(frame_id)
    return result, int(np.count_nonzero(result != _INVALID_OWNER))


def _merge_fixed_constraints(protected: np.ndarray, objects: np.ndarray) -> np.ndarray:
    result = protected.copy()
    conflict = (
        (protected != _INVALID_OWNER)
        & (objects != _INVALID_OWNER)
        & (protected != objects)
    )
    if np.any(conflict):
        raise ValueError("protected_owner_frame_id conflicts with object_masks")
    result[objects != _INVALID_OWNER] = objects[objects != _INVALID_OWNER]
    return result


def _validate_sources(
    sources: Sequence[VideoVisualSource], config: MultilabelOwnerConfig
) -> tuple[tuple[VideoVisualSource, ...], tuple[int, ...], np.ndarray, np.ndarray]:
    items = tuple(sources)
    if not 2 <= len(items) <= int(config.maximum_window_frames):
        raise ValueError("C8 requires a local ordered window of 2 to 5 real sources")
    ids = tuple(int(source.frame_id) for source in items)
    if len(set(ids)) != len(ids):
        raise ValueError("C8 local sources must have unique real frame ids")
    shape = np.asarray(items[0].bgra).shape[:2]
    validity: list[np.ndarray] = []
    centres: list[float] = []
    for source in items:
        if np.asarray(source.bgra).shape[:2] != shape:
            raise ValueError("all C8 local sources must share one placed BGRA canvas")
        valid = _source_valid(source)
        if not np.any(valid):
            raise ValueError("every C8 local source must contain real placed pixels")
        validity.append(valid)
        centres.append(float(np.nonzero(valid)[1].mean()))
    # Reordering a local source to make an optimiser fit would hide a temporal
    # violation.  Equal centres are okay (for example, fully overlapping test
    # sources), but a backwards placed centre is rejected.
    if any(later + 1e-6 < earlier for earlier, later in zip(centres, centres[1:])):
        raise ValueError("C8 ordered sources have non-monotone placed support centres")
    return items, ids, np.stack(validity, axis=0), np.asarray(centres, dtype=np.float32)


def _owner_costs(validity: np.ndarray, centres: np.ndarray, config: MultilabelOwnerConfig) -> np.ndarray:
    """Unary preference for the placed support nearest its own real-strip centre."""

    _, height, width = validity.shape
    x = np.arange(width, dtype=np.float32)[None, None, :]
    scale = max(1.0, float(width - 1))
    cost = float(config.centre_distance_weight) * np.abs(x - centres[:, None, None]) / scale
    # ``validity`` already expands the x-only centre preference over every
    # row.  Repeating it would allocate H copies of the canvas and obscure the
    # actual row-local support geometry used by the optimiser.
    return np.where(validity, cost, np.inf)


def _solve_row(cost: np.ndarray, valid: np.ndarray, fixed_index: np.ndarray, switch_penalty: float) -> np.ndarray:
    """Dynamic program for non-decreasing local source labels on one row."""

    source_count, width = cost.shape
    labels = np.full(width, -1, dtype=np.int32)
    positions = np.flatnonzero(valid)
    if positions.size == 0:
        return labels
    # An invalid canvas gap has no owner and no physical seam.  It must break
    # the monotonic chain rather than making two unrelated valid islands force
    # an impossible label order.  Within each contiguous valid island the
    # temporal constraint remains exact and all fixed owners remain hard.
    break_indices = np.flatnonzero(np.diff(positions) > 1) + 1
    for segment in np.split(positions, break_indices):
        previous = np.full(source_count, np.inf, dtype=np.float64)
        parents = np.full((segment.size, source_count), -1, dtype=np.int16)
        for column_index, column in enumerate(segment):
            unary = np.asarray(cost[:, column], dtype=np.float64).copy()
            fixed = int(fixed_index[column])
            if fixed >= 0:
                unary[np.arange(source_count) != fixed] = np.inf
            if column_index == 0:
                current = unary
            else:
                current = np.full(source_count, np.inf, dtype=np.float64)
                # A non-decreasing label may stay or advance.  Advancing gets
                # the explicit seam cost; ties choose the older owner for
                # stable deterministic provenance.
                for label in range(source_count):
                    allowed = previous[: label + 1]
                    transition = allowed + (np.arange(label + 1) != label) * switch_penalty
                    parent = int(np.argmin(transition))
                    parents[column_index, label] = parent
                    current[label] = unary[label] + transition[parent]
            previous = current
        if not np.isfinite(previous).any():
            raise ValueError("C8 fixed owner constraints are temporally infeasible")
        label = int(np.argmin(previous))
        for column_index in range(segment.size - 1, -1, -1):
            labels[segment[column_index]] = label
            if column_index:
                label = int(parents[column_index, label])
    return labels


def optimise_c8_local_multilabel_owner(
    sources: Sequence[VideoVisualSource],
    *,
    protected_owner_frame_id: np.ndarray | None = None,
    object_masks: Mapping[int, np.ndarray] | None = None,
    config: MultilabelOwnerConfig | None = None,
) -> MultilabelOwnerResult:
    """Optimise a <=5 frame monotone real-owner map without producing RGB.

    ``protected_owner_frame_id`` is a ``-1``/real-frame-id map.  ``object_masks``
    is a mapping from a real local frame id to an object mask that must retain
    that owner.  Both are hard constraints; conflicting or unavailable locks
    fail rather than producing a partial object or an invented owner.
    """

    settings = config or MultilabelOwnerConfig()
    items, source_ids, validity, centres = _validate_sources(sources, settings)
    shape = validity.shape[1:]
    protected = _coerce_owner_constraint(
        protected_owner_frame_id, shape=shape, source_ids=source_ids,
        name="protected_owner_frame_id",
    )
    object_owners, object_count = _coerce_object_constraints(
        object_masks, shape=shape, source_ids=source_ids,
    )
    fixed = _merge_fixed_constraints(protected, object_owners)
    valid = np.any(validity, axis=0)
    id_to_index = {frame_id: index for index, frame_id in enumerate(source_ids)}
    fixed_index = np.full(shape, -1, dtype=np.int32)
    for frame_id, index in id_to_index.items():
        fixed_index[fixed == frame_id] = index
        unavailable = (fixed == frame_id) & ~validity[index]
        if np.any(unavailable):
            raise ValueError("a fixed C8 owner must have a real valid source pixel")
    if np.any((fixed != _INVALID_OWNER) & ~valid):
        raise ValueError("a fixed C8 owner cannot exist outside valid source support")

    costs = _owner_costs(validity, centres, settings)
    owner_index = np.full(shape, -1, dtype=np.int32)
    for row in range(shape[0]):
        owner_index[row] = _solve_row(
            costs[:, row, :], valid[row], fixed_index[row], float(settings.owner_switch_penalty)
        )
    owners = np.full(shape, _INVALID_OWNER, dtype=np.int32)
    for index, frame_id in enumerate(source_ids):
        owners[owner_index == index] = frame_id
    result = MultilabelOwnerResult(
        owner_frame_id=owners,
        valid_mask=np.ascontiguousarray(valid),
        audit=MultilabelOwnerAudit(
            source_frame_ids=source_ids,
            window_frame_count=len(items),
            valid_pixel_count=int(np.count_nonzero(valid)),
            protected_owner_pixel_count=int(np.count_nonzero(protected != _INVALID_OWNER)),
            object_owner_pixel_count=object_count,
            fixed_owner_pixel_count=int(np.count_nonzero(fixed != _INVALID_OWNER)),
            per_source_owner_pixel_count=tuple(
                int(np.count_nonzero(owners == frame_id)) for frame_id in source_ids
            ),
            temporal_monotonic_rows=int(shape[0]),
        ),
    )
    assert_c8_local_multilabel_owner(result, items, fixed_owner_frame_id=fixed)
    return result


def assert_c8_local_multilabel_owner(
    result: MultilabelOwnerResult,
    sources: Sequence[VideoVisualSource],
    *,
    fixed_owner_frame_id: np.ndarray | None = None,
) -> None:
    """Fail closed unless C8 produced a monotone partition of real sources."""

    items = tuple(sources)
    ids = tuple(int(source.frame_id) for source in items)
    if not 2 <= len(items) <= 5 or len(set(ids)) != len(ids):
        raise ValueError("C8 verifier requires 2 to 5 unique real sources")
    owners = np.asarray(result.owner_frame_id)
    valid = np.asarray(result.valid_mask, dtype=bool)
    shape = np.asarray(items[0].bgra).shape[:2]
    if owners.shape != shape or owners.ndim != 2 or valid.shape != shape:
        raise ValueError("C8 owner result must match the placed source canvas")
    if not np.issubdtype(owners.dtype, np.integer):
        raise ValueError("C8 owner result must contain integer frame ids")
    source_valid = np.stack([_source_valid(source) for source in items], axis=0)
    if not np.array_equal(valid, np.any(source_valid, axis=0)):
        raise ValueError("C8 valid mask must be exactly the union of real source support")
    if np.any(valid & ~np.isin(owners, np.asarray(ids, dtype=np.int32))):
        raise ValueError("every valid C8 pixel must have exactly one local real owner")
    if np.any(~valid & (owners != _INVALID_OWNER)):
        raise ValueError("invalid C8 pixels cannot have an owner")
    index_map = np.full(shape, -1, dtype=np.int32)
    for index, frame_id in enumerate(ids):
        selected = owners == frame_id
        if np.any(selected & ~source_valid[index]):
            raise ValueError("C8 owner selected a source without a real valid pixel")
        index_map[selected] = index
    for row in range(shape[0]):
        sequence = index_map[row, valid[row]]
        if sequence.size and np.any(np.diff(sequence) < 0):
            raise ValueError("C8 owner labels violate temporal monotonicity")
    if fixed_owner_frame_id is not None:
        fixed = _coerce_owner_constraint(
            fixed_owner_frame_id, shape=shape, source_ids=ids, name="fixed_owner_frame_id"
        )
        constrained = fixed != _INVALID_OWNER
        if not np.all(owners[constrained] == fixed[constrained]):
            raise ValueError("C8 output violated a protected or object owner constraint")


# Keep the public candidate helper discoverable with the repository's usual
# American spelling while retaining the original implementation name above.
optimize_c8_local_multilabel_owner = optimise_c8_local_multilabel_owner


__all__ = [
    "MultilabelOwnerAudit",
    "MultilabelOwnerConfig",
    "MultilabelOwnerResult",
    "assert_c8_local_multilabel_owner",
    "optimize_c8_local_multilabel_owner",
    "optimise_c8_local_multilabel_owner",
]
