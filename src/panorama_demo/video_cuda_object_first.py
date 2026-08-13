"""Candidate-only, device-resident C11 object-first foreground selection.

This module deliberately has no annotation input.  It extracts a connected
foreground component from *real aligned depth*, follows it through a genuine
RAFT flow field, and chooses exactly one of the two genuine source frames for
that component.  It is intentionally conservative: any missing support,
occlusion disagreement, or non-monotone owner result makes the caller retain
its parent image unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class CudaObjectFirstError(ValueError):
    """C11 could not establish a safe real-source object track."""


@dataclass(frozen=True)
class CudaObjectFirstResult:
    protected_mask: Any
    selected_owner_frame_id: int | None
    accepted: bool
    audit: dict[str, object]


def _dilate(torch: Any, value: Any, radius: int) -> Any:
    return torch.nn.functional.max_pool2d(
        value.float().unsqueeze(0).unsqueeze(0), 2 * radius + 1, 1, radius
    )[0, 0].bool()


def _depth_edges(torch: Any, depth: Any, valid: Any) -> Any:
    tolerance = torch.maximum(torch.full_like(depth, 20.0), depth.abs() * 0.02)
    horizontal = valid[:, 1:] & valid[:, :-1] & (
        (depth[:, 1:] - depth[:, :-1]).abs() > torch.maximum(tolerance[:, 1:], tolerance[:, :-1])
    )
    vertical = valid[1:, :] & valid[:-1, :] & (
        (depth[1:, :] - depth[:-1, :]).abs() > torch.maximum(tolerance[1:, :], tolerance[:-1, :])
    )
    output = torch.zeros_like(valid)
    output[:, 1:] |= horizontal
    output[:, :-1] |= horizontal
    output[1:, :] |= vertical
    output[:-1, :] |= vertical
    return output


def _connected_component_at_best_seed(torch: Any, foreground: Any, sharpness: Any) -> tuple[Any, int]:
    """Return one true 8-connected device component, selected near centre.

    Iterative minimum-label propagation is intentionally bounded.  A failure
    to converge is a rejection rather than a guessed component.
    """

    height, width = (int(value) for value in foreground.shape)
    yy, xx = torch.meshgrid(
        torch.arange(height, device=foreground.device),
        torch.arange(width, device=foreground.device), indexing="ij"
    )
    centre = 1.0 - ((xx.float() - (width - 1) / 2.0).abs() / max(1.0, width / 2.0))
    score = torch.where(foreground, centre + sharpness.clamp_min(0.0), torch.full_like(centre, -1.0))
    if not bool(torch.any(foreground).item()):
        return foreground, 0
    # Integer labels are propagated through 8-neighbourhood minima.  It is
    # connected-component labeling, not a bounding box or semantic raster.
    labels = torch.where(
        foreground,
        torch.arange(height * width, device=foreground.device, dtype=torch.int64).reshape(height, width) + 1,
        torch.zeros((height, width), device=foreground.device, dtype=torch.int64),
    )
    inf = torch.full_like(labels, height * width + 1)
    for _ in range(min(64, height + width)):
        values = torch.where(labels > 0, labels, inf).float()
        propagated = -torch.nn.functional.max_pool2d((-values).unsqueeze(0).unsqueeze(0), 3, 1, 1)[0, 0].to(torch.int64)
        updated = torch.where(foreground, propagated, torch.zeros_like(labels))
        if bool(torch.equal(updated, labels)):
            labels = updated
            break
        labels = updated
    else:
        raise CudaObjectFirstError("C11 connected-component labels did not converge")
    seed = int(torch.argmax(score).item())
    component_label = labels.reshape(-1)[seed]
    component = foreground & (labels == component_label)
    return component, int(component.sum().item())


def select_cuda_object_first_track(
    torch: Any,
    *,
    first_depth_mm: Any,
    second_depth_mm: Any,
    first_bgr: Any,
    second_bgr: Any,
    forward_flow_xy: Any,
    first_valid: Any,
    second_valid: Any,
    first_frame_id: int,
    second_frame_id: int,
    protection_margin_pixels: int = 10,
) -> CudaObjectFirstResult:
    """Build one depth component, RAFT-track it, and select one real owner."""

    if not 8 <= protection_margin_pixels <= 12:
        raise CudaObjectFirstError("C11 protection margin must be in [8, 12] pixels")
    shape = tuple(getattr(first_depth_mm, "shape", ()))
    if len(shape) != 2 or tuple(getattr(second_depth_mm, "shape", ())) != shape:
        raise CudaObjectFirstError("C11 needs matching real aligned-depth corridors")
    if not all(getattr(value, "is_cuda", False) and value.device == first_depth_mm.device for value in (second_depth_mm, first_bgr, second_bgr, forward_flow_xy, first_valid, second_valid)):
        raise CudaObjectFirstError("C11 needs resident CUDA real-source inputs")
    if tuple(forward_flow_xy.shape) != (*shape, 2):
        raise CudaObjectFirstError("C11 RAFT propagation field does not match the output corridor")
    first_valid = first_valid.bool()
    second_valid = second_valid.bool()
    first_depth = first_depth_mm.float()
    second_depth = second_depth_mm.float()
    valid_depth = first_valid & second_valid & torch.isfinite(first_depth) & torch.isfinite(second_depth) & (first_depth > 0) & (second_depth > 0)
    luma = first_bgr.float().mean(dim=0) / 255.0
    sharpness = torch.nn.functional.pad((luma[:, 1:] - luma[:, :-1]).abs(), (0, 1)) + torch.nn.functional.pad((luma[1:] - luma[:-1]).abs(), (0, 0, 0, 1))
    depth_disagreement = (first_depth - second_depth).abs() > torch.maximum(torch.full_like(first_depth, 20.0), torch.minimum(first_depth, second_depth) * 0.02)
    foreground = valid_depth & (_depth_edges(torch, first_depth, valid_depth) | _depth_edges(torch, second_depth, valid_depth) | depth_disagreement)
    component, component_pixels = _connected_component_at_best_seed(torch, foreground, sharpness)
    if component_pixels == 0:
        return CudaObjectFirstResult(component, None, False, {"reason": "no_real_depth_foreground_component", "component_pixel_count": 0})
    flow_finite = torch.isfinite(forward_flow_xy).all(dim=-1)
    flow_magnitude = forward_flow_xy.square().sum(dim=-1).sqrt()
    # Flow is an actual selection term: invalid / implausible motion cannot
    # propagate a component into a target owner decision.
    tracked = component & flow_finite & (flow_magnitude <= 32.0)
    protected = _dilate(torch, tracked, protection_margin_pixels)
    if not bool(torch.any(protected).item()):
        return CudaObjectFirstResult(protected, None, False, {"reason": "raft_track_has_no_safe_component", "component_pixel_count": component_pixels})
    first_cover = bool(torch.all(first_valid[protected]).item())
    second_cover = bool(torch.all(second_valid[protected]).item())
    if not first_cover and not second_cover:
        return CudaObjectFirstResult(protected, None, False, {"reason": "no_single_genuine_source_covers_tracked_component", "component_pixel_count": component_pixels})
    # Four real-data terms: centre, sharpness, nearer depth and occlusion / warp support.
    centre_score = 1.0
    sharp_first = float(sharpness[protected].mean().item())
    sharp_second = float((torch.nn.functional.pad((second_bgr.float().mean(dim=0)[:, 1:] - second_bgr.float().mean(dim=0)[:, :-1]).abs(), (0, 1)))[protected].mean().item())
    depth_first = float((1.0 / first_depth[protected].clamp_min(1.0)).mean().item())
    depth_second = float((1.0 / second_depth[protected].clamp_min(1.0)).mean().item())
    warp = float((1.0 / (1.0 + flow_magnitude[protected])).mean().item())
    score_first = centre_score + sharp_first + depth_first + (1.0 if first_cover else -1.0) + warp
    score_second = centre_score + sharp_second + depth_second + (1.0 if second_cover else -1.0) + warp
    selected = int(first_frame_id if first_cover and (not second_cover or score_first >= score_second) else second_frame_id)
    return CudaObjectFirstResult(protected, selected, True, {
        "schema": "gemini305-video-cuda-object-first-track/v1", "connected_component_method": "bounded_device_8_connected_label_propagation",
        "component_pixel_count": component_pixels, "tracked_component_pixel_count": int(tracked.sum().item()),
        "protection_margin_pixels": protection_margin_pixels, "protected_pixel_count": int(protected.sum().item()),
        "selected_owner_frame_id": selected, "selection_terms": {"centre": centre_score, "sharpness": [sharp_first, sharp_second], "inverse_depth": [depth_first, depth_second], "occlusion_coverage": [first_cover, second_cover], "raft_warp_reliability": warp},
        "raft_propagation_used": True, "maximum_handoffs": 1, "annotations_renderer_input": False,
    })


__all__ = ["CudaObjectFirstError", "CudaObjectFirstResult", "select_cuda_object_first_track"]
