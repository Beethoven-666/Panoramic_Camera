"""Candidate-only CUDA C8 local multi-label real-owner optimisation.

This primitive operates only on already-established local source support.  It
accepts 2--5 chronological real frame ids, GPU-resident unary costs and source
validity, then returns a GPU-resident owner partition.  It never accepts RGB,
depth, poses or source frames, and makes no dense device-to-host transfer.

The temporal constraint is deliberately strict: within each contiguous valid
island of every canvas row, owner labels may only stay or advance in the given
chronological order.  Invalid islands reset the local chain; they have no
owner, so forcing an order across them would be neither physical nor safe.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence


_INVALID_OWNER = -1


class CudaMultilabelOwnerError(ValueError):
    """A C8 local owner problem violates the CUDA real-source contract."""


@dataclass(frozen=True)
class CudaMultilabelOwnerConfig:
    """Closed candidate-only controls for the local C8 optimiser."""

    maximum_window_frames: int = 5
    owner_switch_penalty: float = 1.0
    enforce_non_decreasing_scan_order: bool = True

    def __post_init__(self) -> None:
        if not 2 <= int(self.maximum_window_frames) <= 5:
            raise ValueError("maximum_window_frames must be in [2, 5]")
        if not math.isfinite(float(self.owner_switch_penalty)) or float(self.owner_switch_penalty) < 0.0:
            raise ValueError("owner_switch_penalty must be finite and non-negative")
        if self.enforce_non_decreasing_scan_order is not True:
            raise ValueError("C8 may not disable the temporal/order constraint")


@dataclass(frozen=True)
class CudaMultilabelOwnerResult:
    """Strict device-resident owner provenance; no colour is produced."""

    owner_frame_id: Any
    valid_mask: Any
    audit: dict[str, object]


def _require_cuda_tensor(value: Any, *, shape: tuple[int, ...], device: Any, label: str) -> Any:
    if tuple(getattr(value, "shape", ())) != shape or getattr(value, "device", None) != device:
        raise CudaMultilabelOwnerError(f"{label} must match the unary-cost shape on one CUDA device")
    if not getattr(value, "is_cuda", False):
        raise CudaMultilabelOwnerError(f"{label} must be CUDA-resident")
    return value


def _validate_source_ids(source_frame_ids: Sequence[int], *, maximum_window_frames: int) -> tuple[int, ...]:
    ids = tuple(source_frame_ids)
    if not 2 <= len(ids) <= int(maximum_window_frames):
        raise CudaMultilabelOwnerError("C8 requires a local chronological window of 2 to 5 real sources")
    if any(not isinstance(frame_id, int) or isinstance(frame_id, bool) or frame_id < 0 for frame_id in ids):
        raise CudaMultilabelOwnerError("C8 source_frame_ids must be non-negative real integer frame ids")
    if len(set(ids)) != len(ids) or any(later <= earlier for earlier, later in zip(ids, ids[1:])):
        raise CudaMultilabelOwnerError("C8 source_frame_ids must be unique and strictly chronological")
    return ids


def _protected_indexes(torch: Any, owner: Any, source_ids: tuple[int, ...]) -> Any:
    """Map ``-1``/real-frame-id locks to GPU source-index locks."""

    indexes = torch.full(owner.shape, -1, dtype=torch.int64, device=owner.device)
    legal = owner == _INVALID_OWNER
    for index, frame_id in enumerate(source_ids):
        matched = owner == int(frame_id)
        indexes[matched] = int(index)
        legal = legal | matched
    if bool(torch.any(~legal).item()):
        raise CudaMultilabelOwnerError("protected_owner_frame_id may only name a local real source")
    return indexes


def optimise_cuda_c8_local_multilabel_owner(
    torch: Any,
    *,
    unary_cost: Any,
    source_valid_mask: Any,
    source_frame_ids: Sequence[int],
    protected_owner_frame_id: Any | None = None,
    config: CudaMultilabelOwnerConfig | None = None,
) -> CudaMultilabelOwnerResult:
    """Return one chronological real owner for every valid local canvas pixel.

    ``unary_cost`` and ``source_valid_mask`` have shape ``(K, H, W)`` and
    must already live on the same CUDA device.  A finite unary is required for
    every declared real source sample, preventing an infinity sentinel from
    quietly changing support.  ``protected_owner_frame_id`` is optional
    ``(H, W)`` ``-1``/real-frame-id hard ownership.  The returned tensors stay
    on device; the audit contains only scalar aggregate evidence.
    """

    settings = config or CudaMultilabelOwnerConfig()
    if getattr(unary_cost, "ndim", None) != 3 or not getattr(unary_cost, "is_cuda", False):
        raise CudaMultilabelOwnerError("unary_cost must be a CUDA-resident KxHxW floating tensor")
    if not getattr(unary_cost, "is_floating_point", lambda: False)():
        raise CudaMultilabelOwnerError("unary_cost must be floating point")
    source_count, height, width = (int(value) for value in unary_cost.shape)
    if height < 1 or width < 1:
        raise CudaMultilabelOwnerError("C8 unary_cost must have non-empty canvas dimensions")
    source_ids = _validate_source_ids(source_frame_ids, maximum_window_frames=settings.maximum_window_frames)
    if len(source_ids) != source_count:
        raise CudaMultilabelOwnerError("source_frame_ids length must equal unary_cost source dimension")
    valid_source = _require_cuda_tensor(
        source_valid_mask,
        shape=(source_count, height, width),
        device=unary_cost.device,
        label="source_valid_mask",
    ).bool()
    if bool(torch.any(torch.isnan(unary_cost)).item()):
        raise CudaMultilabelOwnerError("unary_cost cannot contain NaN")
    if bool(torch.any(valid_source & ~torch.isfinite(unary_cost)).item()):
        raise CudaMultilabelOwnerError("every declared real source sample requires a finite unary cost")

    shape = (height, width)
    if protected_owner_frame_id is None:
        protected_index = torch.full(shape, -1, dtype=torch.int64, device=unary_cost.device)
    else:
        protected = _require_cuda_tensor(
            protected_owner_frame_id,
            shape=shape,
            device=unary_cost.device,
            label="protected_owner_frame_id",
        )
        if getattr(protected, "dtype", None) not in (torch.int8, torch.int16, torch.int32, torch.int64):
            raise CudaMultilabelOwnerError("protected_owner_frame_id must contain integer real frame ids")
        protected_index = _protected_indexes(torch, protected.to(dtype=torch.int64), source_ids)

    valid = torch.any(valid_source, dim=0)
    protected_active = protected_index >= 0
    protected_available = valid_source.permute(1, 2, 0).gather(
        2, protected_index.clamp_min(0).unsqueeze(-1)
    ).squeeze(-1)
    if bool(torch.any(protected_active & ~protected_available).item()):
        raise CudaMultilabelOwnerError("a protected owner must have a genuine real source sample")
    if bool(torch.any(protected_active & ~valid).item()):
        raise CudaMultilabelOwnerError("a protected owner cannot exist outside valid source support")

    # Source support is authoritative.  There is no colour or interpolation in
    # this primitive: a label which lacks a genuine sample is simply infeasible.
    costs = unary_cost.to(dtype=torch.float32)
    infinity = torch.tensor(float("inf"), dtype=torch.float32, device=unary_cost.device)
    costs = torch.where(valid_source, costs, infinity)
    label_numbers = torch.arange(source_count, dtype=torch.int64, device=unary_cost.device)
    # ``previous source <= destination source`` is the temporal scan order.
    transition_allowed = label_numbers[:, None] <= label_numbers[None, :]
    switch_cost = (label_numbers[:, None] != label_numbers[None, :]).to(dtype=torch.float32)
    switch_cost = switch_cost * float(settings.owner_switch_penalty)
    transition_cost = torch.where(transition_allowed, switch_cost, infinity)

    # Dynamic programming is column-wise across all rows concurrently.  The
    # only Python loop enumerates canvas columns; every state/cost/parent tensor
    # remains resident on CUDA, including local backtracking provenance.
    parent = torch.full((height, width, source_count), -1, dtype=torch.int64, device=unary_cost.device)
    terminal = torch.full((height, width), -1, dtype=torch.int64, device=unary_cost.device)
    previous = torch.full((height, source_count), float("inf"), dtype=torch.float32, device=unary_cost.device)
    previous_active = torch.zeros((height,), dtype=torch.bool, device=unary_cost.device)
    for column in range(width):
        unary = costs[:, :, column].transpose(0, 1).contiguous()
        lock = protected_index[:, column]
        unary = torch.where(
            (lock[:, None] < 0) | (label_numbers[None, :] == lock[:, None]),
            unary,
            infinity,
        )
        active = valid[:, column]
        starts = active & ~previous_active
        candidates = previous[:, :, None] + transition_cost[None, :, :]
        best_cost, best_parent = torch.min(candidates, dim=1)
        current = torch.where(starts[:, None], unary, unary + best_cost)
        current = torch.where(active[:, None], current, infinity)
        if bool(torch.any(active & ~torch.any(torch.isfinite(current), dim=1)).item()):
            raise CudaMultilabelOwnerError("protected owners are temporally/order infeasible")
        terminal[:, column] = torch.where(active, torch.argmin(current, dim=1), -1)
        parent[:, column, :] = torch.where(
            (active & ~starts)[:, None], best_parent, torch.full_like(best_parent, -1)
        )
        previous = current
        previous_active = active

    # Backtrack every row on device.  When a valid island ends it seeds itself
    # from its own terminal optimum; an invalid gap deliberately resets order.
    owner_index = torch.full(shape, -1, dtype=torch.int64, device=unary_cost.device)
    next_active = torch.zeros((height,), dtype=torch.bool, device=unary_cost.device)
    state = torch.zeros((height,), dtype=torch.int64, device=unary_cost.device)
    rows = torch.arange(height, dtype=torch.int64, device=unary_cost.device)
    for column in range(width - 1, -1, -1):
        active = valid[:, column]
        ends = active & ~next_active
        current_state = torch.where(ends, terminal[:, column], state)
        if column < width - 1:
            continuing = active & next_active
            preceding = parent[rows, column + 1, state]
            current_state = torch.where(continuing, preceding, current_state)
        owner_index[:, column] = torch.where(active, current_state, -1)
        state = torch.where(active, current_state, state)
        next_active = active

    owners = torch.full(shape, _INVALID_OWNER, dtype=torch.int32, device=unary_cost.device)
    for index, frame_id in enumerate(source_ids):
        owners[owner_index == index] = int(frame_id)
    selected_valid = valid_source.permute(1, 2, 0).gather(
        2, owner_index.clamp_min(0).unsqueeze(-1)
    ).squeeze(-1)
    if bool(torch.any(valid & ~selected_valid).item()) or bool(torch.any(~valid & (owners != _INVALID_OWNER)).item()):
        raise CudaMultilabelOwnerError("C8 CUDA result violates exact real-owner topology")
    if bool(torch.any(protected_active & (owner_index != protected_index)).item()):
        raise CudaMultilabelOwnerError("C8 CUDA result violated a protected real owner")
    adjacent = valid[:, 1:] & valid[:, :-1]
    backwards = (owner_index[:, 1:] < owner_index[:, :-1]) & adjacent
    if bool(torch.any(backwards).item()):
        raise CudaMultilabelOwnerError("C8 CUDA result violates temporal/order monotonicity")

    audit = {
        "schema": "gemini305-video-cuda-multilabel-owner/v1",
        "method": "candidate_c8_cuda_local_multilabel_owner/v1",
        "output_residency": "device_tensor",
        "host_transfer_count": 0,
        "source_frame_ids": list(source_ids),
        "window_frame_count": int(source_count),
        "valid_pixel_count": int(valid.sum().item()),
        "protected_owner_pixel_count": int(protected_active.sum().item()),
        "per_source_owner_pixel_count": {
            str(frame_id): int((owners == frame_id).sum().item()) for frame_id in source_ids
        },
        "strict_single_owner": True,
        "temporal_order_constraint": True,
        "invalid_gaps_reset_chain": True,
        "creates_colour": False,
        "creates_pose": False,
        "creates_source": False,
        "interpolates_source_frames": False,
    }
    return CudaMultilabelOwnerResult(owner_frame_id=owners, valid_mask=valid, audit=audit)


# Retain both spellings for discoverability alongside the established CPU C8
# module.  They are identical CUDA-only candidate primitives.
optimize_cuda_c8_local_multilabel_owner = optimise_cuda_c8_local_multilabel_owner


__all__ = [
    "CudaMultilabelOwnerConfig",
    "CudaMultilabelOwnerError",
    "CudaMultilabelOwnerResult",
    "optimise_cuda_c8_local_multilabel_owner",
    "optimize_cuda_c8_local_multilabel_owner",
]
