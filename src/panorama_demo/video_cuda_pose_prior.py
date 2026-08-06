"""CUDA-local RGB-D/ORB inverse-grid prior for one adjacent video corridor.

The candidate renderer normally lays strips out with a scalar scan coordinate.
That coordinate is deliberately *not* a pose substitute.  This helper is the
small, explicitly bounded place where an already-audited pair of real
``camera_to_world`` poses and aligned depths may improve an inverse sampling
grid.  It never creates a source, colour, owner, pose, or panorama depth.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence


class CudaPosePriorError(ValueError):
    """A local RGB-D/ORB pose prior cannot be established safely."""


@dataclass(frozen=True)
class CudaPoseInverseGridResult:
    """A first-source inverse grid corresponding to a real target corridor."""

    inverse_grid_xy: Any
    safe_mask: Any
    audit: dict[str, object]


def _distortion(values: Sequence[float] | None) -> tuple[float, ...]:
    raw = () if values is None else tuple(float(value) for value in values)
    if len(raw) not in (0, 4, 5, 8) or not all(math.isfinite(value) for value in raw):
        raise CudaPosePriorError("ORB pose prior distortion must contain 0, 4, 5, or 8 finite values")
    return raw + (0.0,) * (8 - len(raw))


def _distort_normalized(torch: Any, x: Any, y: Any, coefficients: tuple[float, ...]) -> tuple[Any, Any]:
    k1, k2, p1, p2, k3, k4, k5, k6 = coefficients
    radius2 = x.square() + y.square()
    numerator = 1.0 + k1 * radius2 + k2 * radius2.square() + k3 * radius2.pow(3)
    denominator = 1.0 + k4 * radius2 + k5 * radius2.square() + k6 * radius2.pow(3)
    scale = numerator / denominator.clamp_min(1.0e-12)
    return (
        x * scale + 2.0 * p1 * x * y + p2 * (radius2 + 2.0 * x.square()),
        y * scale + p1 * (radius2 + 2.0 * y.square()) + 2.0 * p2 * x * y,
    )


def _undistort_normalized(torch: Any, x_distorted: Any, y_distorted: Any, coefficients: tuple[float, ...]) -> tuple[Any, Any]:
    """Invert the calibrated Brown-Conrady mapping without leaving CUDA."""

    x, y = x_distorted.clone(), y_distorted.clone()
    # Eight fixed-point refinements are deterministic and sufficient for the
    # small calibrated Gemini colour distortion.  There is no host fallback.
    for _ in range(8):
        predicted_x, predicted_y = _distort_normalized(torch, x, y, coefficients)
        x = x + (x_distorted - predicted_x)
        y = y + (y_distorted - predicted_y)
    return x, y


def cuda_pose_inverse_grid_from_target_depth(
    torch: Any,
    *,
    target_inverse_grid_xy: Any,
    source_depth_mm: Any,
    target_depth_mm: Any,
    source_camera_to_world: Any,
    target_camera_to_world: Any,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    distortion: Sequence[float] | None = None,
    depth_absolute_tolerance_mm: float = 20.0,
    depth_relative_tolerance: float = 0.02,
) -> CudaPoseInverseGridResult:
    """Back-project a real target corridor into a real source inverse grid.

    ``target_inverse_grid_xy`` is a calibrated target-source sampling grid for
    an adjacent C1 corridor.  Its target depth samples are transformed by the
    two immutable ORB poses into source camera coordinates.  The source depth
    must agree at the resulting inverse location, which fail-closes occlusion,
    disocclusion, holes, and any uncalibrated/invalid correspondence.
    """

    values = (fx, fy, cx, cy, depth_absolute_tolerance_mm, depth_relative_tolerance)
    if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in values):
        raise CudaPosePriorError("ORB pose prior calibration and depth gates must be finite")
    if fx <= 0.0 or fy <= 0.0 or depth_absolute_tolerance_mm <= 0.0 or depth_relative_tolerance <= 0.0:
        raise CudaPosePriorError("ORB pose prior requires positive focal lengths and depth gates")
    coefficients = _distortion(distortion)
    if getattr(target_inverse_grid_xy, "ndim", None) != 3 or int(target_inverse_grid_xy.shape[-1]) != 2:
        raise CudaPosePriorError("target inverse grid must be CUDA HxWx2")
    if not bool(getattr(target_inverse_grid_xy, "is_cuda", False)):
        raise CudaPosePriorError("target inverse grid must remain CUDA-resident")
    if getattr(source_depth_mm, "ndim", None) != 2 or getattr(target_depth_mm, "ndim", None) != 2:
        raise CudaPosePriorError("source and target aligned depths must be HxW tensors")
    if source_depth_mm.device != target_inverse_grid_xy.device or target_depth_mm.device != target_inverse_grid_xy.device:
        raise CudaPosePriorError("ORB pose prior depths and inverse grid must use one CUDA device")
    source_height, source_width = (int(value) for value in source_depth_mm.shape)
    target_height, target_width = (int(value) for value in target_depth_mm.shape)
    if min(source_height, source_width, target_height, target_width) < 2:
        raise CudaPosePriorError("ORB pose prior source dimensions must be at least 2")
    for pose, label in ((source_camera_to_world, "source"), (target_camera_to_world, "target")):
        if tuple(getattr(pose, "shape", ())) != (4, 4) or pose.device != target_inverse_grid_xy.device:
            raise CudaPosePriorError(f"{label} camera_to_world must be a CUDA 4x4 matrix")
        if not bool(torch.isfinite(pose).all().item()):
            raise CudaPosePriorError(f"{label} camera_to_world is non-finite")

    grid = target_inverse_grid_xy.to(dtype=torch.float32)
    with torch.no_grad():
        target_depth = torch.nn.functional.grid_sample(
            target_depth_mm.to(dtype=torch.float32).unsqueeze(0).unsqueeze(0),
            grid.unsqueeze(0), mode="nearest", padding_mode="zeros", align_corners=True,
        )[0, 0]
        target_u = (grid[..., 0] + 1.0) * (float(target_width - 1) * 0.5)
        target_v = (grid[..., 1] + 1.0) * (float(target_height - 1) * 0.5)
        target_xd = (target_u - float(cx)) / float(fx)
        target_yd = (target_v - float(cy)) / float(fy)
        target_x, target_y = _undistort_normalized(torch, target_xd, target_yd, coefficients)
        homogeneous_target = torch.stack(
            (target_x * target_depth, target_y * target_depth, target_depth, torch.ones_like(target_depth)), dim=-1
        )
        source_from_target = torch.linalg.inv(source_camera_to_world.to(dtype=torch.float32)) @ target_camera_to_world.to(dtype=torch.float32)
        source_point = homogeneous_target @ source_from_target.T
        source_z = source_point[..., 2]
        source_x = source_point[..., 0] / source_z.clamp_min(1.0e-6)
        source_y = source_point[..., 1] / source_z.clamp_min(1.0e-6)
        source_xd, source_yd = _distort_normalized(torch, source_x, source_y, coefficients)
        source_u = float(fx) * source_xd + float(cx)
        source_v = float(fy) * source_yd + float(cy)
        source_grid = torch.stack(
            (2.0 * source_u / float(source_width - 1) - 1.0, 2.0 * source_v / float(source_height - 1) - 1.0), dim=-1
        ).contiguous()
        source_depth = torch.nn.functional.grid_sample(
            source_depth_mm.to(dtype=torch.float32).unsqueeze(0).unsqueeze(0),
            source_grid.unsqueeze(0), mode="nearest", padding_mode="zeros", align_corners=True,
        )[0, 0]
        target_inside = (grid[..., 0].abs() <= 1.0) & (grid[..., 1].abs() <= 1.0)
        source_inside = (source_grid[..., 0].abs() <= 1.0) & (source_grid[..., 1].abs() <= 1.0)
        target_valid = torch.isfinite(target_depth) & (target_depth > 0.0)
        projected_valid = torch.isfinite(source_z) & (source_z > 0.0) & torch.isfinite(source_grid).all(dim=-1)
        source_valid = torch.isfinite(source_depth) & (source_depth > 0.0)
        consistency_gate = torch.maximum(
            torch.full_like(source_z, float(depth_absolute_tolerance_mm)), source_z.abs() * float(depth_relative_tolerance)
        )
        depth_consistent = (source_depth - source_z).abs() <= consistency_gate
        safe = target_inside & source_inside & target_valid & projected_valid & source_valid & depth_consistent
    audit = {
        "schema": "gemini305-video-cuda-orb-rgbd-inverse-grid-prior/v1",
        "output_residency": "device_tensor",
        "host_transfer_count": 0,
        "uses_real_aligned_depth": True,
        "uses_real_orb_camera_to_world": True,
        "creates_colour": False,
        "creates_owner": False,
        "creates_pose": False,
        "creates_panorama_depth": False,
        "target_valid_pixel_count": int(target_valid.sum().item()),
        "source_projected_valid_pixel_count": int(projected_valid.sum().item()),
        "source_depth_valid_pixel_count": int(source_valid.sum().item()),
        "depth_consistent_pixel_count": int(depth_consistent.sum().item()),
        "safe_inverse_sample_pixel_count": int(safe.sum().item()),
        "depth_absolute_tolerance_mm": float(depth_absolute_tolerance_mm),
        "depth_relative_tolerance": float(depth_relative_tolerance),
    }
    return CudaPoseInverseGridResult(source_grid, safe, audit)


__all__ = [
    "CudaPoseInverseGridResult",
    "CudaPosePriorError",
    "cuda_pose_inverse_grid_from_target_depth",
]
