"""CUDA-local C1 curved hard-owner optimisation for adjacent real sources.

The operation accepts an already-established pair-local corridor and a seam
cost.  It creates neither colour nor a source: every valid output pixel is
owned by exactly one of the two supplied real frame IDs.  Dynamic programming
uses first- and second-order row regularisation wholly on the provided torch
device; only scalar audit values leave it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


class CudaConstrainedOwnerError(ValueError):
    """The pair-local C1 owner problem is not auditable on one CUDA device."""


@dataclass(frozen=True)
class CudaConstrainedOwnerResult:
    owner_frame_id: Any
    seam_x_by_row: Any
    audit: dict[str, object]


def _require_mask(mask: Any, *, shape: tuple[int, int], device: Any, label: str) -> Any:
    if tuple(getattr(mask, "shape", ())) != shape or getattr(mask, "device", None) != device:
        raise CudaConstrainedOwnerError(f"{label} must be an HxW tensor on the seam-cost device")
    return mask.bool()


def constrained_curved_hard_owner(
    torch: Any,
    *,
    seam_cost: Any,
    first_valid_mask: Any,
    second_valid_mask: Any,
    first_frame_id: int,
    second_frame_id: int,
    corridor_x: tuple[int, int],
    protected_mask: Any | None = None,
    protected_owner_frame_id: Any | None = None,
    maximum_row_step_pixels: int = 4,
    first_order_penalty: float = 5.0,
    second_order_penalty: float = 3.0,
) -> CudaConstrainedOwnerResult:
    """Compute a pair-local curved C1 seam with a strict hard owner map.

    ``seam_cost`` is a device-resident HxW cost of placing a seam at each
    overlap pixel.  The corridor must be narrow and is the only region where
    both sources may compete.  Protected pixels preserve their supplied
    owner, which is how object/depth protection can constrain this C1 primitive
    without letting it make a protection decision itself.
    """

    if getattr(seam_cost, "ndim", None) != 2 or not getattr(seam_cost, "is_cuda", False):
        raise CudaConstrainedOwnerError("seam_cost must be a CUDA-resident HxW tensor")
    height, width = (int(value) for value in seam_cost.shape)
    if height < 2 or width < 8:
        raise CudaConstrainedOwnerError("C1 CUDA seam needs an HxW cost with H>=2 and W>=8")
    if not (isinstance(first_frame_id, int) and isinstance(second_frame_id, int) and 0 <= first_frame_id < second_frame_id):
        raise CudaConstrainedOwnerError("C1 needs two chronological non-negative real frame IDs")
    left, right = corridor_x
    if not (isinstance(left, int) and isinstance(right, int) and 0 <= left < right <= width and 8 <= right - left <= 256):
        raise CudaConstrainedOwnerError("C1 corridor must be an in-bounds 8..256 pixel interval")
    if not isinstance(maximum_row_step_pixels, int) or not 1 <= maximum_row_step_pixels <= 16:
        raise CudaConstrainedOwnerError("maximum_row_step_pixels must be an integer in [1, 16]")
    if not all(
        isinstance(value, (int, float)) and math.isfinite(value) and value >= 0.0
        for value in (first_order_penalty, second_order_penalty)
    ):
        raise CudaConstrainedOwnerError("C1 curvature penalties must be finite and non-negative")

    shape = (height, width)
    first_valid = _require_mask(first_valid_mask, shape=shape, device=seam_cost.device, label="first_valid_mask")
    second_valid = _require_mask(second_valid_mask, shape=shape, device=seam_cost.device, label="second_valid_mask")
    overlap = first_valid & second_valid
    corridor = torch.zeros(shape, dtype=torch.bool, device=seam_cost.device)
    corridor[:, left:right] = True
    support = overlap & corridor & torch.isfinite(seam_cost)
    if not bool(torch.any(support).item()):
        raise CudaConstrainedOwnerError("C1 corridor has no finite common real-source support")

    protected = torch.zeros(shape, dtype=torch.bool, device=seam_cost.device)
    protected_owner = None
    if protected_mask is not None or protected_owner_frame_id is not None:
        if protected_mask is None or protected_owner_frame_id is None:
            raise CudaConstrainedOwnerError("protected mask and owner must be supplied together")
        protected = _require_mask(protected_mask, shape=shape, device=seam_cost.device, label="protected_mask")
        if tuple(getattr(protected_owner_frame_id, "shape", ())) != shape or protected_owner_frame_id.device != seam_cost.device:
            raise CudaConstrainedOwnerError("protected owner must be an HxW tensor on the seam-cost device")
        protected_owner = protected_owner_frame_id.to(dtype=torch.int32)
        allowed = (protected_owner == first_frame_id) | (protected_owner == second_frame_id)
        if bool(torch.any(protected & ~allowed).item()):
            raise CudaConstrainedOwnerError("protected C1 pixels must already belong to one adjacent real source")
        # A protected owner with no corresponding genuine source is unsafe.
        first_protected = protected & (protected_owner == first_frame_id)
        second_protected = protected & (protected_owner == second_frame_id)
        if bool(torch.any(first_protected & ~first_valid).item()) or bool(torch.any(second_protected & ~second_valid).item()):
            raise CudaConstrainedOwnerError("protected C1 owner lacks a genuine source sample")

    local_width = right - left
    local_cost = seam_cost[:, left:right].to(dtype=torch.float32)
    local_support = support[:, left:right]
    # Rows with no overlap carry an identity seam state, so they cannot create
    # an arbitrary all-row owner cut through a gap.
    centre = local_width // 2
    local_cost = torch.where(local_support, local_cost, torch.full_like(local_cost, float("inf")))
    empty_rows = ~torch.any(local_support, dim=1)
    local_cost[empty_rows, centre] = 0.0
    deltas = torch.arange(
        -maximum_row_step_pixels,
        maximum_row_step_pixels + 1,
        dtype=torch.int64,
        device=seam_cost.device,
    )
    velocity_count = int(deltas.numel())
    zero_velocity = maximum_row_step_pixels
    infinity = torch.tensor(float("inf"), dtype=torch.float32, device=seam_cost.device)
    previous = torch.full((local_width, velocity_count), infinity, device=seam_cost.device)
    previous[:, zero_velocity] = local_cost[0]
    parent_x = torch.full((height, local_width, velocity_count), -1, dtype=torch.int16, device=seam_cost.device)
    parent_v = torch.full((height, local_width, velocity_count), -1, dtype=torch.int8, device=seam_cost.device)
    columns = torch.arange(local_width, device=seam_cost.device, dtype=torch.int64)
    velocity_change = (deltas[:, None] - deltas[None, :]).abs().to(dtype=torch.float32)
    for row in range(1, height):
        current = torch.full_like(previous, infinity)
        for velocity_index, velocity in enumerate(deltas):
            predecessor_columns = columns - velocity
            valid_columns = (predecessor_columns >= 0) & (predecessor_columns < local_width)
            current_columns = columns[valid_columns]
            if int(current_columns.numel()) == 0:
                continue
            candidates = previous[predecessor_columns[valid_columns]]
            candidates = candidates + float(first_order_penalty) * float(abs(int(velocity)))
            candidates = candidates + float(second_order_penalty) * velocity_change[velocity_index][None, :]
            values, predecessor_velocity = torch.min(candidates, dim=1)
            current[current_columns, velocity_index] = local_cost[row, current_columns] + values
            parent_x[row, current_columns, velocity_index] = predecessor_columns[valid_columns].to(dtype=torch.int16)
            parent_v[row, current_columns, velocity_index] = predecessor_velocity.to(dtype=torch.int8)
        previous = current
    terminal = torch.argmin(previous.reshape(-1))
    x = (terminal // velocity_count).to(dtype=torch.int64)
    velocity = (terminal % velocity_count).to(dtype=torch.int64)
    seam_local = torch.empty((height,), dtype=torch.int64, device=seam_cost.device)
    seam_local[-1] = x
    for row in range(height - 1, 0, -1):
        next_x = parent_x[row, x, velocity].to(dtype=torch.int64)
        next_velocity = parent_v[row, x, velocity].to(dtype=torch.int64)
        valid_parent = (next_x >= 0) & (next_velocity >= 0)
        x = torch.where(valid_parent, next_x, torch.tensor(centre, device=seam_cost.device, dtype=torch.int64))
        velocity = torch.where(valid_parent, next_velocity, torch.tensor(zero_velocity, device=seam_cost.device, dtype=torch.int64))
        seam_local[row - 1] = x
    seam = seam_local + left

    # Outside common support ownership is forced by genuine validity.  In the
    # common corridor, the curved seam is the sole owner decision.
    owner = torch.full(shape, -1, dtype=torch.int32, device=seam_cost.device)
    owner[first_valid & ~second_valid] = int(first_frame_id)
    owner[second_valid & ~first_valid] = int(second_frame_id)
    owner[overlap] = int(first_frame_id)
    xs = torch.arange(width, device=seam_cost.device, dtype=torch.int64)[None, :]
    second_side = overlap & corridor & (xs >= seam[:, None])
    owner[second_side] = int(second_frame_id)
    if protected_owner is not None:
        owner[protected] = protected_owner[protected]
    valid = first_valid | second_valid
    if bool(torch.any(valid & (owner < 0)).item()) or bool(torch.any(~valid & (owner >= 0)).item()):
        raise CudaConstrainedOwnerError("C1 CUDA result violates valid/owner topology")
    row_step = (seam[1:] - seam[:-1]).abs()
    maximum_observed_step = int(row_step.max().item()) if height > 1 else 0
    if maximum_observed_step > maximum_row_step_pixels:
        raise CudaConstrainedOwnerError("C1 dynamic-program seam exceeded the configured row-step limit")
    audit = {
        "schema": "gemini305-video-cuda-constrained-owner/v1",
        "output_residency": "device_tensor",
        "host_transfer_count": 0,
        "first_frame_id": int(first_frame_id),
        "second_frame_id": int(second_frame_id),
        "corridor_x": [int(left), int(right)],
        "corridor_width_px": int(local_width),
        "first_order_penalty": float(first_order_penalty),
        "second_order_penalty": float(second_order_penalty),
        "maximum_row_step_pixels": int(maximum_row_step_pixels),
        "maximum_observed_row_step_pixels": maximum_observed_step,
        "common_support_pixel_count": int(overlap.sum().item()),
        "protected_pixel_count": int(protected.sum().item()),
        "strict_single_owner": True,
        "creates_colour": False,
        "creates_source": False,
        "creates_pose": False,
    }
    return CudaConstrainedOwnerResult(owner, seam, audit)


__all__ = [
    "CudaConstrainedOwnerError",
    "CudaConstrainedOwnerResult",
    "constrained_curved_hard_owner",
]
