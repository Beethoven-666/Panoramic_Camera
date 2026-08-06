"""CUDA-local RGB-D layer safety for candidate residual meshes.

It accepts only adjacent real-source depth tensors and resident forward/backward
flow.  The result is a conservative same-layer background mask; it creates no
colour, source frame, pose, or dense panorama depth.
"""

from __future__ import annotations

import math
from typing import Any


class CudaDepthLayerError(ValueError):
    """A device-resident depth/flow input cannot establish local safety."""


def cuda_same_layer_safe_mask(
    torch: Any,
    *,
    first_depth_mm: Any,
    second_depth_mm: Any,
    forward_flow_xy: Any,
    backward_flow_xy: Any,
    absolute_tolerance_mm: float = 20.0,
    relative_tolerance: float = 0.02,
    forward_backward_maximum_error_px: float = 1.0,
    second_depth_is_already_forward_warped: bool = False,
) -> tuple[Any, dict[str, object]]:
    """Return a local, device-resident same-layer safety mask and scalar audit.

    A target depth is sampled at forward-flow coordinates.  Depth agreement
    uses ``max(20mm, 2% depth)`` by default; forward/backward agreement is
    required independently.  The small scalar counts in the audit are the
    only host-visible values.

    ``second_depth_is_already_forward_warped`` is deliberately explicit for
    the C4 renderer.  Its two depth images originate in different real camera
    coordinate systems, so C4 samples the target depth with its calibrated
    source-grid plus resident RAFT displacement first.  Re-sampling that
    already-corresponding local target with source-pixel RAFT values would be
    geometrically wrong.  The forward/backward gate remains active in either
    mode.
    """

    values = (absolute_tolerance_mm, relative_tolerance, forward_backward_maximum_error_px)
    if not all(isinstance(value, (int, float)) and math.isfinite(value) and value > 0.0 for value in values):
        raise CudaDepthLayerError("depth/flow gates must be finite positive values")
    first, second = first_depth_mm, second_depth_mm
    if getattr(first, "ndim", None) != 2 or tuple(first.shape) != tuple(getattr(second, "shape", ())):
        raise CudaDepthLayerError("adjacent depth tensors must be matching HxW")
    height, width = (int(value) for value in first.shape)
    if height < 2 or width < 2:
        raise CudaDepthLayerError("depth tensors are too small")
    for flow, label in ((forward_flow_xy, "forward"), (backward_flow_xy, "backward")):
        if tuple(getattr(flow, "shape", ())) != (height, width, 2):
            raise CudaDepthLayerError(f"{label} flow must be HxWx2")
        if flow.device != first.device:
            raise CudaDepthLayerError(f"{label} flow must remain on the depth device")
    if second.device != first.device:
        raise CudaDepthLayerError("adjacent depth tensors must remain on one device")
    dtype = torch.float32
    if second_depth_is_already_forward_warped:
        # The caller proved that the second depth has been sampled at the
        # forward-warped calibrated coordinates.  Both local arrays now share
        # a pixel domain, while the resident RAFT vectors remain raw-source
        # vectors used exclusively for the FB consistency gate below.
        sampled_second = second.to(dtype=dtype)
        sampled_backward = backward_flow_xy.to(dtype=dtype)
        inside = torch.ones((height, width), dtype=torch.bool, device=first.device)
    else:
        yy, xx = torch.meshgrid(
            torch.arange(height, device=first.device, dtype=dtype),
            torch.arange(width, device=first.device, dtype=dtype),
            indexing="ij",
        )
        map_x, map_y = xx + forward_flow_xy[..., 0], yy + forward_flow_xy[..., 1]
        grid = torch.stack(
            (2.0 * map_x / float(width - 1) - 1.0, 2.0 * map_y / float(height - 1) - 1.0),
            dim=-1,
        )
        sampled_second = torch.nn.functional.grid_sample(
            second.to(dtype=dtype).unsqueeze(0).unsqueeze(0),
            grid.unsqueeze(0),
            mode="nearest",
            padding_mode="zeros",
            align_corners=True,
        )[0, 0]
        sampled_backward = torch.nn.functional.grid_sample(
            backward_flow_xy.permute(2, 0, 1).unsqueeze(0).to(dtype=dtype),
            grid.unsqueeze(0),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )[0].permute(1, 2, 0)
        inside = (grid[..., 0].abs() <= 1.0) & (grid[..., 1].abs() <= 1.0)
    first_valid = torch.isfinite(first) & (first > 0.0)
    second_valid = torch.isfinite(sampled_second) & (sampled_second > 0.0)
    depth_gate = torch.maximum(
        torch.full_like(first, float(absolute_tolerance_mm)),
        first.abs() * float(relative_tolerance),
    )
    depth_consistent = (first.to(dtype=dtype) - sampled_second).abs() <= depth_gate
    fb_error = (forward_flow_xy.to(dtype=dtype) + sampled_backward).square().sum(dim=-1).sqrt()
    flow_consistent = fb_error <= float(forward_backward_maximum_error_px)
    safe = first_valid & second_valid & inside & depth_consistent & flow_consistent
    audit = {
        "schema": "gemini305-video-cuda-depth-layer-safety/v1",
        "output_residency": "device_tensor",
        "host_transfer_count": 0,
        "first_valid_pixel_count": int(first_valid.sum().item()),
        "second_sample_valid_pixel_count": int(second_valid.sum().item()),
        "depth_consistent_pixel_count": int(depth_consistent.sum().item()),
        "forward_backward_consistent_pixel_count": int(flow_consistent.sum().item()),
        "same_layer_safe_pixel_count": int(safe.sum().item()),
        "absolute_tolerance_mm": float(absolute_tolerance_mm),
        "relative_tolerance": float(relative_tolerance),
        "second_depth_is_already_forward_warped": bool(second_depth_is_already_forward_warped),
        "creates_colour": False,
        "creates_owner": False,
        "creates_pose": False,
    }
    return safe, audit


__all__ = ["CudaDepthLayerError", "cuda_same_layer_safe_mask"]
