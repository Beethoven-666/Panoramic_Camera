"""Candidate-only CUDA MultiBand blending for an already-owned safe corridor.

This module is deliberately narrower than a panorama compositor.  It receives
two *real*, already-resident source images plus the already-audited dominant
owner map.  It cannot select a source, change an owner, create a pose, or
write outside the caller-supplied safe-background region.  In particular it is
not imported by the photo renderer or the legacy video bridge.

The operation is a local Laplacian-pyramid blend.  It is useful only after a
real C1/C5 seam plan has produced a safe corridor; absent that evidence callers
must retain hard ownership instead of treating this as a fallback renderer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class CudaSafeMultiBandError(ValueError):
    """A proposed CUDA MultiBand operation violates the C6 safety contract."""


@dataclass(frozen=True)
class CudaSafeMultiBandResult:
    """Device-resident pixels, unchanged provenance, and scalar-only audit."""

    bgr: Any
    owner_frame_id: Any
    blend_mask: Any
    audit: dict[str, object]


def _require_same_cuda_device(*tensors: Any) -> str:
    """Reject CPU, cross-device, and non-tensor inputs before any blending."""

    devices: list[str] = []
    for tensor in tensors:
        device = getattr(tensor, "device", None)
        if device is None or getattr(device, "type", None) != "cuda":
            raise CudaSafeMultiBandError("C6 safe MultiBand requires CUDA-resident tensors")
        devices.append(str(device))
    if len(set(devices)) != 1:
        raise CudaSafeMultiBandError("C6 safe MultiBand inputs must share one CUDA device")
    return devices[0]


def _require_mask(mask: Any, *, name: str, shape: tuple[int, int], device: str) -> None:
    if tuple(getattr(mask, "shape", ())) != shape:
        raise CudaSafeMultiBandError(f"{name} must match the owner HxW shape")
    if str(getattr(mask, "device", "")) != device:
        raise CudaSafeMultiBandError(f"{name} must remain on the source CUDA device")
    if getattr(mask, "dtype", None) is None or str(mask.dtype) != "torch.bool":
        raise CudaSafeMultiBandError(f"{name} must be an explicit boolean safety mask")


def _owner_boundary_band(torch: Any, owner: Any, first_id: int, second_id: int, pixels: int) -> Any:
    """Return the local pair boundary expanded by the audited C6 band width."""

    pair_owner = (owner == first_id) | (owner == second_id)
    boundary = torch.zeros_like(pair_owner)
    horizontal = pair_owner[:, 1:] & pair_owner[:, :-1] & (owner[:, 1:] != owner[:, :-1])
    vertical = pair_owner[1:, :] & pair_owner[:-1, :] & (owner[1:, :] != owner[:-1, :])
    boundary[:, 1:] |= horizontal
    boundary[:, :-1] |= horizontal
    boundary[1:, :] |= vertical
    boundary[:-1, :] |= vertical
    # C6 is constrained to a narrow local corridor.  The max-pool performs a
    # binary dilation on CUDA; it never creates source colour or ownership.
    width = int(pixels) * 2 + 1
    return torch.nn.functional.max_pool2d(
        boundary.to(dtype=torch.float32).unsqueeze(0).unsqueeze(0),
        kernel_size=width,
        stride=1,
        padding=int(pixels),
    )[0, 0].bool()


def _laplacian_pyramid(torch: Any, image: Any, levels: int) -> list[Any]:
    """Build a compact CUDA Laplacian pyramid without a host materialisation."""

    gaussians = [image]
    for _ in range(1, int(levels)):
        current = gaussians[-1]
        if min(int(current.shape[-2]), int(current.shape[-1])) < 2:
            break
        gaussians.append(torch.nn.functional.avg_pool2d(current, kernel_size=2, stride=2))
    laplacians: list[Any] = []
    for index in range(len(gaussians) - 1):
        current, next_level = gaussians[index], gaussians[index + 1]
        upsampled = torch.nn.functional.interpolate(
            next_level,
            size=current.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        laplacians.append(current - upsampled)
    laplacians.append(gaussians[-1])
    return laplacians


def _multiband(torch: Any, first: Any, second: Any, second_owner: Any, levels: int) -> tuple[Any, int]:
    """Blend two CUDA images with their dominant-owner Gaussian pyramid."""

    first_pyramid = _laplacian_pyramid(torch, first, levels)
    second_pyramid = _laplacian_pyramid(torch, second, levels)
    alpha_pyramid = [second_owner.to(dtype=torch.float32).unsqueeze(0).unsqueeze(0)]
    while len(alpha_pyramid) < len(first_pyramid):
        alpha_pyramid.append(torch.nn.functional.avg_pool2d(alpha_pyramid[-1], kernel_size=2, stride=2))
    combined: list[Any] = []
    for first_level, second_level, alpha in zip(first_pyramid, second_pyramid, alpha_pyramid, strict=True):
        combined.append(first_level * (1.0 - alpha) + second_level * alpha)
    result = combined[-1]
    for level in range(len(combined) - 2, -1, -1):
        result = torch.nn.functional.interpolate(
            result,
            size=combined[level].shape[-2:],
            mode="bilinear",
            align_corners=False,
        ) + combined[level]
    return result, len(combined)


def blend_cuda_safe_multiband(
    torch: Any,
    *,
    first_bgr: Any,
    second_bgr: Any,
    owner_frame_id: Any,
    first_frame_id: int,
    second_frame_id: int,
    safe_background_mask: Any,
    protected_mask: Any,
    risk_mask: Any,
    band_pixels: int = 16,
    levels: int = 3,
) -> CudaSafeMultiBandResult:
    """Blend an already-owned pair only inside its supplied safe corridor.

    All large tensors stay on one CUDA device.  The owner map is returned
    unchanged and never used as a fractional provenance substitute: the
    separate ``blend_mask`` records where the two real sources contributed.
    ``protected_mask`` and ``risk_mask`` are required evidence; any overlap
    with the supplied safe area is a fail-closed error, even if the current
    owner boundary would not happen to reach it.
    """

    if type(first_frame_id) is not int or type(second_frame_id) is not int or first_frame_id == second_frame_id:
        raise CudaSafeMultiBandError("C6 MultiBand needs two distinct real integer frame ids")
    if not 16 <= int(band_pixels) <= 24:
        raise CudaSafeMultiBandError("C6 MultiBand band_pixels must remain in [16, 24]")
    # C6's planned ablation is exactly 3/4/5 pyramid levels.  A hard-owner
    # pair remains the fallback outside the audited safe band.
    if not 3 <= int(levels) <= 5:
        raise CudaSafeMultiBandError("C6 MultiBand levels must remain in [3, 5]")
    if tuple(getattr(first_bgr, "shape", ())) != tuple(getattr(second_bgr, "shape", ())):
        raise CudaSafeMultiBandError("C6 MultiBand real-source images must have identical CHW shape")
    if getattr(first_bgr, "ndim", None) != 3 or int(first_bgr.shape[0]) != 3:
        raise CudaSafeMultiBandError("C6 MultiBand images must be BGR CHW tensors")
    if getattr(first_bgr, "dtype", None) != getattr(second_bgr, "dtype", None):
        raise CudaSafeMultiBandError("C6 MultiBand real-source images must have one dtype")
    height, width = int(first_bgr.shape[1]), int(first_bgr.shape[2])
    if height < 2 or width < 2:
        raise CudaSafeMultiBandError("C6 MultiBand images are too small")
    if tuple(getattr(owner_frame_id, "shape", ())) != (height, width):
        raise CudaSafeMultiBandError("C6 MultiBand owner map must match image HxW")
    if getattr(owner_frame_id, "dtype", None) is None or not str(owner_frame_id.dtype).startswith("torch.int"):
        raise CudaSafeMultiBandError("C6 MultiBand owner map must be an integer tensor")
    device = _require_same_cuda_device(
        first_bgr,
        second_bgr,
        owner_frame_id,
        safe_background_mask,
        protected_mask,
        risk_mask,
    )
    _require_mask(safe_background_mask, name="safe_background_mask", shape=(height, width), device=device)
    _require_mask(protected_mask, name="protected_mask", shape=(height, width), device=device)
    _require_mask(risk_mask, name="risk_mask", shape=(height, width), device=device)
    owner = owner_frame_id
    pair_owner = (owner == first_frame_id) | (owner == second_frame_id)
    foreign_owner = (owner >= 0) & ~pair_owner
    if bool(foreign_owner.any().item()):
        raise CudaSafeMultiBandError("C6 pair owner map contains an unrelated real frame")
    if bool((safe_background_mask & ~pair_owner).any().item()):
        raise CudaSafeMultiBandError("C6 safe background must belong to the supplied real-source pair")
    if bool((safe_background_mask & (protected_mask | risk_mask)).any().item()):
        raise CudaSafeMultiBandError("C6 safe background overlaps a protected or risk pixel")

    with torch.cuda.device(first_bgr.device):
        boundary_band = _owner_boundary_band(torch, owner, first_frame_id, second_frame_id, int(band_pixels))
        blend_mask = safe_background_mask & pair_owner & boundary_band
        # A hard owner remains the base everywhere.  Invalid owner positions
        # stay black, which is an explicit invalid result rather than invented
        # RGB.  Valid pair pixels use exactly their dominant real source.
        first_selected = owner == first_frame_id
        second_selected = owner == second_frame_id
        base = torch.zeros_like(first_bgr)
        base[:, first_selected] = first_bgr[:, first_selected]
        base[:, second_selected] = second_bgr[:, second_selected]
        blend_pixels = int(blend_mask.sum().item())
        effective_levels = 0
        if blend_pixels:
            first_float = first_bgr.unsqueeze(0).to(dtype=torch.float32)
            second_float = second_bgr.unsqueeze(0).to(dtype=torch.float32)
            pyramid_blend, effective_levels = _multiband(
                torch,
                first_float,
                second_float,
                second_selected,
                int(levels),
            )
            if getattr(first_bgr.dtype, "is_floating_point", False):
                blended = pyramid_blend[0].to(dtype=first_bgr.dtype)
            else:
                blended = pyramid_blend[0].round().clamp_(0, 255).to(dtype=first_bgr.dtype)
            base[:, blend_mask] = blended[:, blend_mask]
    audit = {
        "schema": "gemini305-video-cuda-safe-multiband/v1",
        "candidate_only": True,
        "method": "cuda_laplacian_pyramid_safe_background",
        "output_residency": "device_tensor",
        "dense_host_transfer_count": 0,
        "scalar_audit_only": True,
        "participant_frame_ids": [int(first_frame_id), int(second_frame_id)],
        "owner_map_preserved": True,
        "strict_single_owner": True,
        "safe_background_pixel_count": int(safe_background_mask.sum().item()),
        "boundary_band_pixel_count": int(boundary_band.sum().item()),
        "blend_pixel_count": int(blend_mask.sum().item()),
        "protected_intersection_pixel_count": 0,
        "risk_intersection_pixel_count": 0,
        "band_pixels": int(band_pixels),
        "requested_levels": int(levels),
        "effective_levels": int(effective_levels),
        "applied": bool(blend_pixels),
        "outside_blend_is_hard_owner": True,
        "creates_composited_colour": True,
        "uses_only_real_source_colours": True,
        "creates_owner": False,
        "creates_pose": False,
    }
    return CudaSafeMultiBandResult(base, owner_frame_id, blend_mask, audit)


__all__ = [
    "CudaSafeMultiBandError",
    "CudaSafeMultiBandResult",
    "blend_cuda_safe_multiband",
]
