"""CUDA-local C5 protection and single-real-owner locking.

This candidate primitive derives protection only from aligned-depth evidence
of adjacent real sources.  It never accepts semantic or manual-annotation
masks, and never alters colour, poses, source-frame identity, or a mesh.  A
protected domain is changed only when one adjacent genuine source covers all
its pixels; otherwise the caller gets an explicit rejected lock and must
retain its prior hard owner.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


class CudaObjectOwnerLockError(ValueError):
    """C5 object/depth protection inputs are not safe CUDA pair evidence."""


@dataclass(frozen=True)
class CudaObjectOwnerLockResult:
    owner_frame_id: Any
    protected_mask: Any
    audit: dict[str, object]


def _mask(value: Any, *, shape: tuple[int, int], device: Any, label: str) -> Any:
    if tuple(getattr(value, "shape", ())) != shape or getattr(value, "device", None) != device:
        raise CudaObjectOwnerLockError(f"{label} must be an HxW tensor on the CUDA pair device")
    return value.bool()


def _expand(torch: Any, mask: Any, radius: int) -> Any:
    if radius == 0:
        return mask
    return torch.nn.functional.max_pool2d(
        mask.to(dtype=torch.float32).unsqueeze(0).unsqueeze(0),
        kernel_size=radius * 2 + 1,
        stride=1,
        padding=radius,
    )[0, 0].bool()


def cuda_depth_object_protection(
    torch: Any,
    *,
    first_depth_mm: Any,
    second_depth_mm: Any,
    depth_edge_guard_pixels: int = 3,
    absolute_tolerance_mm: float = 20.0,
    relative_tolerance: float = 0.02,
) -> tuple[Any, dict[str, object]]:
    """Derive conservative device-resident C5 object/depth protection."""

    if getattr(first_depth_mm, "ndim", None) != 2 or tuple(first_depth_mm.shape) != tuple(getattr(second_depth_mm, "shape", ())):
        raise CudaObjectOwnerLockError("adjacent aligned depth must be matching HxW tensors")
    if not getattr(first_depth_mm, "is_cuda", False) or second_depth_mm.device != first_depth_mm.device:
        raise CudaObjectOwnerLockError("C5 aligned depth must be resident on one CUDA device")
    height, width = (int(item) for item in first_depth_mm.shape)
    if height < 2 or width < 2:
        raise CudaObjectOwnerLockError("C5 aligned depth is too small")
    if not isinstance(depth_edge_guard_pixels, int) or not 0 <= depth_edge_guard_pixels <= 32:
        raise CudaObjectOwnerLockError("C5 depth-edge guard radius must be an integer in [0, 32]")
    if not all(isinstance(item, (int, float)) and math.isfinite(item) and item > 0.0 for item in (absolute_tolerance_mm, relative_tolerance)):
        raise CudaObjectOwnerLockError("C5 depth tolerances must be finite positive")
    first, second = first_depth_mm.float(), second_depth_mm.float()
    valid_first = torch.isfinite(first) & (first > 0.0)
    valid_second = torch.isfinite(second) & (second > 0.0)

    def edges(depth: Any, valid: Any) -> Any:
        result = ~valid
        tolerance = torch.maximum(
            torch.full_like(depth, float(absolute_tolerance_mm)),
            depth.abs() * float(relative_tolerance),
        )
        horizontal = valid[:, 1:] & valid[:, :-1] & ((depth[:, 1:] - depth[:, :-1]).abs() > torch.maximum(tolerance[:, 1:], tolerance[:, :-1]))
        vertical = valid[1:, :] & valid[:-1, :] & ((depth[1:, :] - depth[:-1, :]).abs() > torch.maximum(tolerance[1:, :], tolerance[:-1, :]))
        result[:, 1:] |= horizontal
        result[:, :-1] |= horizontal
        result[1:, :] |= vertical
        result[:-1, :] |= vertical
        return result

    depth_edges = _expand(torch, edges(first, valid_first) | edges(second, valid_second), depth_edge_guard_pixels)
    protected = depth_edges
    audit = {
        "schema": "gemini305-video-cuda-object-protection/v1",
        "output_residency": "device_tensor",
        "host_transfer_count": 0,
        "object_protected_pixel_count": 0,
        "depth_edge_protected_pixel_count": int(depth_edges.sum().item()),
        "protected_pixel_count": int(protected.sum().item()),
        "protection_input": "aligned_depth_only",
        "manual_measurement_annotations_used": False,
        "creates_colour": False,
        "creates_source": False,
        "creates_pose": False,
    }
    return protected, audit


def lock_cuda_protected_owner(
    torch: Any,
    *,
    owner_frame_id: Any,
    first_valid_mask: Any,
    second_valid_mask: Any,
    protected_mask: Any,
    first_frame_id: int,
    second_frame_id: int,
    preferred_owner_frame_id: int | None = None,
) -> CudaObjectOwnerLockResult:
    """Pin all protected pixels to one adjacent genuine owner or reject."""

    if getattr(owner_frame_id, "ndim", None) != 2 or not getattr(owner_frame_id, "is_cuda", False):
        raise CudaObjectOwnerLockError("owner_frame_id must be a CUDA-resident HxW tensor")
    if not (isinstance(first_frame_id, int) and isinstance(second_frame_id, int) and 0 <= first_frame_id < second_frame_id):
        raise CudaObjectOwnerLockError("C5 needs chronological non-negative adjacent real frame IDs")
    if preferred_owner_frame_id is not None and preferred_owner_frame_id not in (first_frame_id, second_frame_id):
        raise CudaObjectOwnerLockError("preferred C5 owner must be one adjacent real source")
    shape = tuple(int(item) for item in owner_frame_id.shape)
    first_valid = _mask(first_valid_mask, shape=shape, device=owner_frame_id.device, label="first_valid_mask")
    second_valid = _mask(second_valid_mask, shape=shape, device=owner_frame_id.device, label="second_valid_mask")
    protected = _mask(protected_mask, shape=shape, device=owner_frame_id.device, label="protected_mask")
    owner = owner_frame_id.to(dtype=torch.int32).clone()
    valid = first_valid | second_valid
    source_ids = (int(first_frame_id), int(second_frame_id))
    # Every existing valid owner must be a genuine adjacent source before C5
    # is allowed to constrain it.
    allowed_owner = (owner == source_ids[0]) | (owner == source_ids[1])
    if bool(torch.any(valid & ~allowed_owner).item()) or bool(torch.any(~valid & (owner >= 0)).item()):
        raise CudaObjectOwnerLockError("C5 input owner topology is not a two-real-source hard owner")
    protected_count = int(protected.sum().item())
    if protected_count == 0:
        return CudaObjectOwnerLockResult(owner, protected, {
            "schema": "gemini305-video-cuda-object-owner-lock/v1", "output_residency": "device_tensor",
            "host_transfer_count": 0, "accepted": True, "applied": False, "reason": "no_protected_pixels",
            "protected_pixel_count": 0, "creates_colour": False, "creates_source": False, "creates_pose": False,
        })
    first_covers = bool(torch.all(first_valid[protected]).item())
    second_covers = bool(torch.all(second_valid[protected]).item())
    if not first_covers and not second_covers:
        return CudaObjectOwnerLockResult(owner, protected, {
            "schema": "gemini305-video-cuda-object-owner-lock/v1", "output_residency": "device_tensor",
            "host_transfer_count": 0, "accepted": False, "applied": False, "reason": "no_single_genuine_source_covers_protection",
            "protected_pixel_count": protected_count, "creates_colour": False, "creates_source": False, "creates_pose": False,
        })
    if first_covers and second_covers:
        if preferred_owner_frame_id is not None:
            chosen = int(preferred_owner_frame_id)
        else:
            first_existing = int((owner[protected] == first_frame_id).sum().item())
            second_existing = int((owner[protected] == second_frame_id).sum().item())
            chosen = first_frame_id if first_existing >= second_existing else second_frame_id
    else:
        chosen = first_frame_id if first_covers else second_frame_id
    owner[protected] = int(chosen)
    return CudaObjectOwnerLockResult(owner, protected, {
        "schema": "gemini305-video-cuda-object-owner-lock/v1", "output_residency": "device_tensor",
        "host_transfer_count": 0, "accepted": True, "applied": True, "reason": None,
        "protected_pixel_count": protected_count, "locked_owner_frame_id": int(chosen),
        "creates_colour": False, "creates_source": False, "creates_pose": False, "strict_single_owner": True,
    })


__all__ = [
    "CudaObjectOwnerLockError", "CudaObjectOwnerLockResult", "cuda_depth_object_protection", "lock_cuda_protected_owner",
]
