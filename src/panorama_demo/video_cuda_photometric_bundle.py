"""C13 candidate-only robust photometric bundle and illumination field.

The bundle deliberately builds on the audited C7 graph rather than a source-0
chain.  Its low-frequency field is applied before C1/C6 seam composition and
only through an observation-derived safe-background mask.  It has no owner,
pose, geometry, or annotation input authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .video_cuda_photometric import _linear_rgb, _srgb_from_linear


class CudaPhotometricBundleError(ValueError):
    """C13's safe illumination field contract was not met."""


@dataclass(frozen=True)
class CudaIlluminationFieldConfig:
    """Closed C13 controls; allowed field cells are exactly 64 or 96 px."""

    cell_width_pixels: int = 64
    cell_height_pixels: int = 96
    maximum_gain_deviation: float = 0.035
    field_strength: float = 0.20
    held_out_tile_modulus: int = 5
    held_out_tile_remainder: int = 0

    def validated(self) -> "CudaIlluminationFieldConfig":
        if (self.cell_width_pixels, self.cell_height_pixels) not in {(64, 96), (96, 64)}:
            raise CudaPhotometricBundleError("C13 field cells must be 64x96 or 96x64 pixels")
        if not 0.0 < self.maximum_gain_deviation <= 0.05:
            raise CudaPhotometricBundleError("C13 maximum field gain deviation must be in (0, .05]")
        if not 0.0 < self.field_strength <= 0.25:
            raise CudaPhotometricBundleError("C13 field strength must be in (0, .25]")
        if not isinstance(self.held_out_tile_modulus, int) or self.held_out_tile_modulus < 2:
            raise CudaPhotometricBundleError("C13 held-out modulus is invalid")
        if not isinstance(self.held_out_tile_remainder, int) or not 0 <= self.held_out_tile_remainder < self.held_out_tile_modulus:
            raise CudaPhotometricBundleError("C13 held-out remainder is invalid")
        return self


def apply_cuda_safe_illumination_field(
    torch: Any,
    *,
    corrected_bgr_srgb: Any,
    original_bgr_srgb: Any,
    safe_background_mask: Any,
    config: CudaIlluminationFieldConfig = CudaIlluminationFieldConfig(),
) -> tuple[Any, dict[str, object]]:
    """Apply a bounded 64/96-px field only at safe real-source pixels.

    The field is a blockwise, bilinearly upsampled residual from the accepted
    pre-seam global correction.  Outside ``safe_background_mask`` the result
    is byte-identical to the accepted global correction.  Its held-out split
    is spatial and deterministic; it is audit-only and never selects owners.
    """
    settings = config.validated()
    corrected = _linear_rgb(torch, corrected_bgr_srgb, label="C13 corrected real source")
    original = _linear_rgb(torch, original_bgr_srgb, label="C13 original real source")
    if corrected.device != original.device or tuple(corrected.shape) != tuple(original.shape):
        raise CudaPhotometricBundleError("C13 field sources must be same-shaped CUDA tensors")
    if tuple(getattr(safe_background_mask, "shape", ())) != tuple(corrected.shape[1:]) or safe_background_mask.device != corrected.device:
        raise CudaPhotometricBundleError("C13 safe background mask must be HxW on the source CUDA device")
    safe = safe_background_mask.bool()
    height, width = (int(corrected.shape[1]), int(corrected.shape[2]))
    if int(safe.sum().item()) == 0:
        raise CudaPhotometricBundleError("C13 illumination field has no safe background pixels")
    # A bounded low-frequency residual is measured only from the real-source
    # global correction.  It does not invent texture: block cells are pooled
    # and then bilinearly expanded, with unsafe pixels retained exactly.
    residual = (corrected - original) * safe.unsqueeze(0)
    counts = torch.nn.functional.avg_pool2d(
        safe.float().unsqueeze(0).unsqueeze(0),
        kernel_size=(settings.cell_height_pixels, settings.cell_width_pixels),
        stride=(settings.cell_height_pixels, settings.cell_width_pixels), ceil_mode=True,
    ).clamp_min(1.0e-6)
    pooled = torch.nn.functional.avg_pool2d(
        residual.unsqueeze(0), kernel_size=(settings.cell_height_pixels, settings.cell_width_pixels),
        stride=(settings.cell_height_pixels, settings.cell_width_pixels), ceil_mode=True,
    ) / counts
    field = torch.nn.functional.interpolate(pooled, size=(height, width), mode="bilinear", align_corners=False)[0]
    # Convert an additive linear residual to a small multiplicative field.
    denominator = corrected.clamp_min(0.05)
    gain_field = (1.0 + settings.field_strength * field / denominator).clamp(
        1.0 - settings.maximum_gain_deviation,
        1.0 + settings.maximum_gain_deviation,
    )
    candidate_linear = corrected * gain_field
    output_linear = torch.where(safe.unsqueeze(0), candidate_linear, corrected)
    encoded = _srgb_from_linear(torch, output_linear)
    if corrected_bgr_srgb.dtype == torch.uint8:
        output = encoded.mul(255.0).round().clamp_(0.0, 255.0).to(dtype=torch.uint8)
    else:
        output = encoded.to(dtype=corrected_bgr_srgb.dtype)
    changed = torch.any(output != corrected_bgr_srgb, dim=0) & safe
    rows = torch.arange(height, device=corrected.device).view(height, 1)
    columns = torch.arange(width, device=corrected.device).view(1, width)
    tiles = rows.div(settings.cell_height_pixels, rounding_mode="floor") + columns.div(settings.cell_width_pixels, rounding_mode="floor")
    held_out = safe & (tiles.remainder(settings.held_out_tile_modulus) == settings.held_out_tile_remainder)
    train = safe & ~held_out
    delta = (output_linear - corrected).abs().amax(dim=0)
    def p95(mask: Any) -> float:
        values = delta[mask]
        return float(torch.quantile(values, 0.95).item()) if int(values.numel()) else 0.0
    return output, {
        "schema": "gemini305-video-cuda-safe-illumination-field/v1",
        "candidate_only": True,
        "pre_seam_correction": True,
        "safe_background_only": True,
        "field_cell_pixels": [int(settings.cell_width_pixels), int(settings.cell_height_pixels)],
        "field_model": "blockwise_low_frequency_bilinear/v1",
        "safe_background_pixel_count": int(safe.sum().item()),
        "training_pixel_count": int(train.sum().item()),
        "held_out_pixel_count": int(held_out.sum().item()),
        "held_out_split": "deterministic_field_cells/v1",
        "training_delta_p95_linear": p95(train),
        "held_out_delta_p95_linear": p95(held_out),
        "actual_safe_output_affected_pixel_count": int(changed.sum().item()),
        "field_gain_minimum": float(gain_field.min().item()),
        "field_gain_maximum": float(gain_field.max().item()),
        "output_residency": "device_tensor",
        "dense_host_transfer_count": 0,
        "creates_owner": False,
        "creates_pose": False,
    }


__all__ = [
    "CudaIlluminationFieldConfig",
    "CudaPhotometricBundleError",
    "apply_cuda_safe_illumination_field",
]
