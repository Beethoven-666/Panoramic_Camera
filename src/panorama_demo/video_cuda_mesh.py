"""Fail-closed CUDA local residual mesh fitting for v2 video candidates.

The mesh is intentionally a local displacement field, never a pose/source
substitute.  It is fitted from a training subset of already-resident adjacent
flow and can be used only when a disjoint held-out subset, boundary identity,
displacement, Jacobian and local-scale audits all pass on device.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CudaMeshResult:
    offset_xy: Any
    accepted_mask: Any
    audit: dict[str, object]


class CudaMeshError(ValueError):
    """CUDA mesh inputs cannot support a bounded local residual warp."""


@dataclass(frozen=True)
class CudaDisCorrespondence:
    """Device-resident, bounded dense inverse-search correspondence.

    OpenCV's DIS implementation is CPU-only in the supported runtime, so it
    cannot be placed between the resident calibration grid and mesh without a
    forbidden host round trip.  This implementation keeps the defining local
    dense inverse-search operation on CUDA: each output pixel chooses the
    lowest patch residual from a bounded inverse search in the *real* source
    samples.  It is deliberately not labelled as OpenCV DIS in the audit.
    """

    forward_xy: Any
    backward_xy: Any
    safe_mask: Any
    audit: dict[str, object]


def _p95(torch: Any, values: Any) -> float | None:
    if int(values.numel()) == 0:
        return None
    return float(torch.quantile(values.float(), 0.95).item())


def _dense_inverse_search(
    torch: Any,
    *,
    source_gray: Any,
    target_gray: Any,
    source_valid: Any,
    target_valid: Any,
    maximum_search_px: int,
) -> tuple[Any, Any, Any]:
    """Return source sampling offsets that minimise a local target residual."""

    candidates: list[Any] = []
    offsets: list[tuple[int, int]] = []
    invalid_cost = torch.tensor(1.0e6, dtype=torch.float32, device=source_gray.device)
    for dy in range(-maximum_search_px, maximum_search_px + 1):
        for dx in range(-maximum_search_px, maximum_search_px + 1):
            # ``roll(-offset)`` evaluates source[y + dy, x + dx] at the
            # target coordinate.  Wrapped pixels are made invalid below.
            shifted = torch.roll(source_gray, shifts=(-dy, -dx), dims=(0, 1))
            shifted_valid = torch.roll(source_valid.bool(), shifts=(-dy, -dx), dims=(0, 1))
            if dy > 0:
                shifted_valid[-dy:, :] = False
            elif dy < 0:
                shifted_valid[: -dy, :] = False
            if dx > 0:
                shifted_valid[:, -dx:] = False
            elif dx < 0:
                shifted_valid[:, : -dx] = False
            residual = (shifted - target_gray).abs()
            # The local footprint prevents a single textureless pixel from
            # becoming an unconstrained displacement estimate.
            residual = torch.nn.functional.avg_pool2d(
                residual[None, None], 3, stride=1, padding=1
            )[0, 0]
            usable = shifted_valid & target_valid.bool()
            candidates.append(torch.where(usable, residual, invalid_cost))
            offsets.append((dx, dy))
    costs = torch.stack(candidates, dim=0)
    best_cost, best_index = costs.min(dim=0)
    dx_table = torch.tensor([item[0] for item in offsets], dtype=torch.float32, device=source_gray.device)
    dy_table = torch.tensor([item[1] for item in offsets], dtype=torch.float32, device=source_gray.device)
    flow = torch.stack((dx_table[best_index], dy_table[best_index]), dim=-1)
    usable = best_cost < invalid_cost
    return flow, usable, best_cost


def estimate_cuda_dis_rgb_correspondence(
    torch: Any,
    *,
    first_bgr: Any,
    second_bgr: Any,
    first_valid_mask: Any,
    second_valid_mask: Any,
    maximum_search_px: int = 4,
    forward_backward_maximum_error_px: float = 1.5,
) -> CudaDisCorrespondence:
    """Estimate local CUDA DIS-style correspondence in one safe corridor.

    The inputs are the already calibrated adjacent real-source samples.  No
    frame, flow field, mesh, or mask is copied to the host; only scalar audit
    values are materialised after this function returns.
    """

    if not 1 <= int(maximum_search_px) <= 8:
        raise CudaMeshError("CUDA DIS search must be an integer in [1, 8]")
    if not isinstance(forward_backward_maximum_error_px, (int, float)) or not math.isfinite(forward_backward_maximum_error_px):
        raise CudaMeshError("CUDA DIS forward/backward limit must be finite")
    expected = tuple(getattr(first_bgr, "shape", ()))
    if len(expected) != 3 or expected[0] != 3 or tuple(getattr(second_bgr, "shape", ())) != expected:
        raise CudaMeshError("CUDA DIS requires matching 3xHxW real-source BGR tiles")
    height, width = int(expected[1]), int(expected[2])
    if height < 8 or width < 8:
        raise CudaMeshError("CUDA DIS needs an 8x8 adjacent risk corridor")
    for mask in (first_valid_mask, second_valid_mask):
        if tuple(getattr(mask, "shape", ())) != (height, width) or mask.device != first_bgr.device:
            raise CudaMeshError("CUDA DIS valid masks must be device HxW tensors")
    if second_bgr.device != first_bgr.device:
        raise CudaMeshError("CUDA DIS pair samples must remain on one device")
    first = first_bgr.to(dtype=torch.float32).mean(dim=0) / 255.0
    second = second_bgr.to(dtype=torch.float32).mean(dim=0) / 255.0
    forward, forward_valid, forward_cost = _dense_inverse_search(
        torch, source_gray=first, target_gray=second, source_valid=first_valid_mask,
        target_valid=second_valid_mask, maximum_search_px=int(maximum_search_px),
    )
    backward, backward_valid, _ = _dense_inverse_search(
        torch, source_gray=second, target_gray=first, source_valid=second_valid_mask,
        target_valid=first_valid_mask, maximum_search_px=int(maximum_search_px),
    )
    ys, xs = torch.meshgrid(
        torch.arange(height, device=first.device, dtype=torch.float32),
        torch.arange(width, device=first.device, dtype=torch.float32),
        indexing="ij",
    )
    sample_x = xs + forward[..., 0]
    sample_y = ys + forward[..., 1]
    grid = torch.stack(
        (2.0 * sample_x / float(width - 1) - 1.0, 2.0 * sample_y / float(height - 1) - 1.0), dim=-1
    )[None]
    sampled_backward = torch.nn.functional.grid_sample(
        backward.permute(2, 0, 1)[None], grid, mode="bilinear", padding_mode="zeros", align_corners=True
    )[0].permute(1, 2, 0)
    in_bounds = (sample_x >= 0.0) & (sample_x <= float(width - 1)) & (sample_y >= 0.0) & (sample_y <= float(height - 1))
    # High RGB structure is explicitly protected in C2; it may guide the
    # local correspondence but cannot receive a residual mesh without C4's
    # depth layers / C5 object protection.
    gx = torch.nn.functional.pad((first[:, 1:] - first[:, :-1]).abs(), (0, 1))
    gy = torch.nn.functional.pad((first[1:, :] - first[:-1, :]).abs(), (0, 0, 0, 1))
    gradient = gx + gy
    texture_threshold = torch.quantile(gradient, 0.80)
    fb_error = (forward + sampled_backward).square().sum(dim=-1).sqrt()
    safe = (
        forward_valid & backward_valid & in_bounds
        & (fb_error <= float(forward_backward_maximum_error_px))
        & (gradient <= texture_threshold)
    )
    audit = {
        "schema": "gemini305-video-cuda-dis-correspondence/v1",
        "backend": "torch_cuda_dense_inverse_search",
        "opencv_dis_used": False,
        "host_transfer_count": 0,
        "maximum_search_px": int(maximum_search_px),
        "forward_backward_maximum_error_px": float(forward_backward_maximum_error_px),
        "safe_pixel_count": int(safe.sum().item()),
        "forward_residual_p95": _p95(torch, forward_cost[forward_valid]),
        "forward_backward_error_p95_px": _p95(torch, fb_error[safe]),
        "high_structure_protected": True,
    }
    return CudaDisCorrespondence(forward, backward, safe, audit)


def fit_cuda_local_mesh(
    torch: Any,
    *,
    flow_xy: Any,
    training_mask: Any,
    held_out_mask: Any,
    safe_mask: Any,
    protected_mask: Any,
    maximum_displacement_px: float = 8.0,
    held_out_error_p95_maximum_px: float = 1.5,
    protected_boundary_taper_px: int = 0,
    identity_taper_mask: Any | None = None,
) -> CudaMeshResult:
    """Fit a bounded device-local mesh from train-only flow observations.

    This deliberately has no fallback warp: any failed audit returns a zero
    field and an all-false accepted mask.  The caller must retain the original
    hard owner in that case.
    """

    limits = (maximum_displacement_px, held_out_error_p95_maximum_px)
    if not all(isinstance(value, (int, float)) and math.isfinite(value) and 0.0 < value <= 8.0 for value in limits):
        raise CudaMeshError("CUDA mesh limits must be finite positive values <= 8 pixels")
    if not isinstance(protected_boundary_taper_px, int) or isinstance(protected_boundary_taper_px, bool) or not 0 <= protected_boundary_taper_px <= 8:
        raise CudaMeshError("CUDA mesh protected boundary taper must be an integer in [0, 8]")
    if getattr(flow_xy, "ndim", None) != 3 or int(flow_xy.shape[-1]) != 2:
        raise CudaMeshError("flow_xy must be a device HxWx2 tensor")
    height, width = int(flow_xy.shape[0]), int(flow_xy.shape[1])
    if height < 8 or width < 8:
        raise CudaMeshError("CUDA mesh needs at least 8x8 flow support")
    masks = (training_mask, held_out_mask, safe_mask, protected_mask)
    if any(tuple(getattr(mask, "shape", ())) != (height, width) for mask in masks):
        raise CudaMeshError("CUDA mesh masks must match flow HxW")
    if any(mask.device != flow_xy.device for mask in masks):
        raise CudaMeshError("CUDA mesh flow and masks must stay on one device")
    if identity_taper_mask is not None and (
        tuple(getattr(identity_taper_mask, "shape", ())) != (height, width)
        or identity_taper_mask.device != flow_xy.device
    ):
        raise CudaMeshError("CUDA mesh identity taper mask must match the device flow domain")
    train = training_mask.bool() & safe_mask.bool() & ~protected_mask.bool()
    held_out = held_out_mask.bool() & safe_mask.bool() & ~protected_mask.bool()
    disjoint = not bool((train & held_out).any().item())
    minimum_samples = 32
    base_audit: dict[str, object] = {
        "schema": "gemini305-video-cuda-local-mesh/v1",
        "output_residency": "device_tensor",
        "host_transfer_count": 0,
        "training_pixel_count": int(train.sum().item()),
        "held_out_pixel_count": int(held_out.sum().item()),
        "train_held_out_disjoint": disjoint,
        "maximum_displacement_px": float(maximum_displacement_px),
        "held_out_error_p95_maximum_px": float(held_out_error_p95_maximum_px),
        "creates_colour": False,
        "creates_owner": False,
        "creates_pose": False,
        "protected_boundary_identity_taper_px": int(protected_boundary_taper_px),
    }
    zero = torch.zeros_like(flow_xy, dtype=torch.float32)
    no_mesh = CudaMeshResult(zero, torch.zeros((height, width), dtype=torch.bool, device=flow_xy.device), base_audit)
    if not disjoint or int(train.sum().item()) < minimum_samples or int(held_out.sum().item()) < minimum_samples:
        return CudaMeshResult(zero, no_mesh.accepted_mask, {**base_audit, "accepted": False, "reason": "insufficient_disjoint_train_or_held_out_support"})
    flow = flow_xy.to(dtype=torch.float32)
    finite = torch.isfinite(flow).all(dim=-1)
    # Compute a local distance-to-protection field before fitting.  Only the
    # interior beyond the identity ramp may contribute observations or be
    # audited as an active mesh cell.  This prevents a held-out residual at a
    # deliberately owner-only boundary from vetoing a valid interior mesh.
    protected = protected_mask.bool()
    taper_protected = protected if identity_taper_mask is None else identity_taper_mask.bool()
    taper_radius = int(protected_boundary_taper_px)
    if taper_radius == 0:
        distance = torch.ones((height, width), dtype=torch.int32, device=flow.device)
        mesh_domain = torch.ones((height, width), dtype=torch.bool, device=flow.device)
    else:
        distance = torch.full((height, width), taper_radius + 1, dtype=torch.int32, device=flow.device)
        protected_float = taper_protected.to(dtype=torch.float32)[None, None]
        for radius in range(taper_radius + 1):
            if radius == 0:
                near = taper_protected
            else:
                near = torch.nn.functional.max_pool2d(
                    protected_float, 2 * radius + 1, stride=1, padding=radius
                )[0, 0].bool()
            distance = torch.where(
                (distance == taper_radius + 1) & near,
                torch.full_like(distance, radius),
                distance,
            )
        mesh_domain = distance >= taper_radius
    train = train & mesh_domain
    held_out = held_out & mesh_domain
    base_audit["training_pixel_count"] = int(train.sum().item())
    base_audit["held_out_pixel_count"] = int(held_out.sum().item())
    if int(train.sum().item()) < minimum_samples or int(held_out.sum().item()) < minimum_samples:
        return CudaMeshResult(zero, no_mesh.accepted_mask, {**base_audit, "accepted": False, "reason": "insufficient_interior_mesh_support"})
    if not bool((finite & train).any().item()) or not bool((finite & held_out).any().item()):
        return CudaMeshResult(zero, no_mesh.accepted_mask, {**base_audit, "accepted": False, "reason": "nonfinite_flow_support"})
    # A low-frequency 5x5 locally weighted field is the mesh prior.  Only
    # train pixels enter the estimator; held-out pixels are read afterwards.
    weights = (train & finite).to(dtype=torch.float32)
    numerator = torch.nn.functional.avg_pool2d(
        (flow * weights[..., None]).permute(2, 0, 1).unsqueeze(0), 5, stride=1, padding=2
    )[0].permute(1, 2, 0)
    denominator = torch.nn.functional.avg_pool2d(weights.unsqueeze(0).unsqueeze(0), 5, stride=1, padding=2)[0, 0]
    mesh = numerator / denominator.clamp_min(1e-6)[..., None]
    magnitude = mesh.square().sum(dim=-1).sqrt()
    mesh = mesh * (float(maximum_displacement_px) / magnitude.clamp_min(float(maximum_displacement_px)))[..., None]
    # A hard-owner/protected domain has zero mesh displacement.  Taper the
    # adjacent eight pixels to identity as well: abruptly zeroing an 8 px
    # local mesh at a depth/occlusion boundary creates a fold in the last
    # legal mesh cell.  The taper is purely local, does not add pixels to the
    # mesh domain, and preserves the zero-displacement contract at every
    # protected or image-boundary pixel.
    taper = (
        torch.ones_like(distance, dtype=torch.float32)
        if taper_radius == 0
        else (distance.to(dtype=torch.float32) / float(taper_radius)).clamp_(0.0, 1.0)
    )
    mesh = mesh * taper[..., None]
    mesh = torch.where(protected[..., None], torch.zeros_like(mesh), mesh)
    mesh[0, :, :] = 0.0
    mesh[-1, :, :] = 0.0
    mesh[:, 0, :] = 0.0
    mesh[:, -1, :] = 0.0
    magnitude = mesh.square().sum(dim=-1).sqrt()
    # Discrete Jacobian of source+residual mapping, inspected only where a
    # safe (unprotected) mesh could actually be sampled.
    du_dx = torch.zeros((height, width), device=flow.device)
    du_dy = torch.zeros((height, width), device=flow.device)
    dv_dx = torch.zeros((height, width), device=flow.device)
    dv_dy = torch.zeros((height, width), device=flow.device)
    du_dx[:, 1:-1] = 0.5 * (mesh[:, 2:, 0] - mesh[:, :-2, 0])
    du_dy[1:-1, :] = 0.5 * (mesh[2:, :, 0] - mesh[:-2, :, 0])
    dv_dx[:, 1:-1] = 0.5 * (mesh[:, 2:, 1] - mesh[:, :-2, 1])
    dv_dy[1:-1, :] = 0.5 * (mesh[2:, :, 1] - mesh[:-2, :, 1])
    determinant = (1.0 + du_dx) * (1.0 + dv_dy) - du_dy * dv_dx
    local_scale = determinant.abs().sqrt()
    # The identity boundary is a hard C2 contract.  Its adjacent finite
    # difference is intentionally not an interior mesh cell, otherwise a
    # valid zero boundary would falsely look like a fold against a nonzero
    # interior estimate.
    interior = torch.zeros((height, width), dtype=torch.bool, device=flow.device)
    interior[1:-1, 1:-1] = True
    accepted = safe_mask.bool() & ~protected_mask.bool() & finite & interior & mesh_domain
    held_out_error = (mesh - flow).square().sum(dim=-1).sqrt()[held_out & finite]
    max_disp = float(magnitude[accepted].max().item()) if bool(accepted.any().item()) else float("inf")
    min_jacobian = float(determinant[accepted].min().item()) if bool(accepted.any().item()) else float("-inf")
    min_scale = float(local_scale[accepted].min().item()) if bool(accepted.any().item()) else 0.0
    max_scale = float(local_scale[accepted].max().item()) if bool(accepted.any().item()) else float("inf")
    held_out_p95 = _p95(torch, held_out_error)
    boundary_max = float(torch.stack((mesh[0].abs().max(), mesh[-1].abs().max(), mesh[:, 0].abs().max(), mesh[:, -1].abs().max())).max().item())
    passed = bool(
        held_out_p95 is not None
        and held_out_p95 <= float(held_out_error_p95_maximum_px)
        and max_disp <= float(maximum_displacement_px)
        and min_jacobian >= 0.05
        and min_scale >= 0.70
        and max_scale <= 1.40
        and boundary_max == 0.0
    )
    audit = {
        **base_audit,
        "held_out_error_p95_px": held_out_p95,
        "maximum_observed_displacement_px": max_disp,
        "minimum_jacobian": min_jacobian,
        "minimum_local_scale": min_scale,
        "maximum_local_scale": max_scale,
        "boundary_identity_maximum_error_px": boundary_max,
        "accepted": passed,
        "reason": None if passed else "mesh_audit_rejected",
    }
    return CudaMeshResult(mesh if passed else zero, accepted if passed else no_mesh.accepted_mask, audit)


def fit_cuda_coarse_to_fine_local_mesh(
    torch: Any,
    *,
    flow_xy: Any,
    training_mask: Any,
    held_out_mask: Any,
    safe_mask: Any,
    protected_mask: Any,
    **kwargs: Any,
) -> CudaMeshResult:
    """Fit a C9 two-scale, still bounded positive-Jacobian local mesh.

    The coarse 9x9 train-only field is a prior, while the existing 5x5 fitter
    performs the final update and all final-grid Jacobian/scale audits.  No
    held-out sample enters either estimate; the existing fitter is still the
    sole authority that accepts a final inverse-sampling mesh.
    """

    if getattr(flow_xy, "ndim", None) != 3 or int(flow_xy.shape[-1]) != 2:
        raise CudaMeshError("coarse-to-fine mesh needs HxWx2 flow")
    train = training_mask.bool() & safe_mask.bool() & ~protected_mask.bool()
    weights = train.to(dtype=torch.float32)
    numerator = torch.nn.functional.avg_pool2d(
        (flow_xy.to(dtype=torch.float32) * weights[..., None]).permute(2, 0, 1)[None],
        9, stride=1, padding=4,
    )[0].permute(1, 2, 0)
    denominator = torch.nn.functional.avg_pool2d(weights[None, None], 9, stride=1, padding=4)[0, 0]
    coarse = numerator / denominator.clamp_min(1.0e-6)[..., None]
    # At locations with no training neighbourhood the unmodified observation
    # allows the final fitter's support checks to fail closed normally.
    coarse = torch.where((denominator > 0.0)[..., None], coarse, flow_xy.to(dtype=torch.float32))
    # Coarse plus local residual update, represented as a single final flow
    # field so the downstream maximum displacement/Jacobian/scale checks are
    # applied to exactly the grid that samples final output pixels.
    final_flow = 0.5 * coarse + 0.5 * flow_xy.to(dtype=torch.float32)
    result = fit_cuda_local_mesh(
        torch, flow_xy=final_flow, training_mask=training_mask,
        held_out_mask=held_out_mask, safe_mask=safe_mask,
        protected_mask=protected_mask, **kwargs,
    )
    return CudaMeshResult(
        result.offset_xy,
        result.accepted_mask,
        {
            **result.audit,
            "schema": "gemini305-video-cuda-coarse-to-fine-local-mesh/v1",
            "coarse_to_fine_levels": [9, 5],
            "coarse_prior_training_pixel_count": int(train.sum().item()),
            "final_grid_constraints_audited": True,
        },
    )


__all__ = [
    "CudaDisCorrespondence",
    "CudaMeshError",
    "CudaMeshResult",
    "estimate_cuda_dis_rgb_correspondence",
    "fit_cuda_local_mesh",
    "fit_cuda_coarse_to_fine_local_mesh",
]
