"""Candidate-only D2 monotonic depth-layer warp primitives.

This is deliberately independent of the retired C3/C4 free mesh routes.  A
D2 renderer supplies one horizontal knot vector per observed depth layer and
an optional bounded vertical residual.  The parameterisation itself makes
folds impossible: output knot increments are strictly positive and the open
uniform cubic B-spline is sampled for a strict derivative audit.  Callers must still reject an audit
which does not describe pixels actually resampled by the candidate renderer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import cv2


class D2MonotonicDepthLayerWarpError(ValueError):
    """D2 warp input or its fail-closed geometric audit is invalid."""


_LAYERS = ("far", "mid", "near")


@dataclass(frozen=True)
class D2MonotonicDepthLayerWarpConfig:
    """Immutable D2 geometry bounds, expressed in full-resolution pixels."""

    minimum_jacobian: float = 0.05
    minimum_scale: float = 0.70
    maximum_scale: float = 1.40
    maximum_vertical_residual_pixels: float = 1.5

    def __post_init__(self) -> None:
        if not (0.0 < self.minimum_jacobian <= self.minimum_scale <= self.maximum_scale):
            raise D2MonotonicDepthLayerWarpError("D2 horizontal scale bounds are invalid")
        if not np.isfinite(self.maximum_vertical_residual_pixels) or self.maximum_vertical_residual_pixels < 0.0:
            raise D2MonotonicDepthLayerWarpError("D2 maximum vertical residual must be finite and non-negative")


@dataclass(frozen=True)
class D2LayerWarp:
    """One observed depth layer's forward input-x to output-x mapping."""

    layer: str
    input_x: np.ndarray
    output_x: np.ndarray
    vertical_residual_y: np.ndarray


def _finite_vector(value: object, *, name: str, minimum_length: int = 2) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size < minimum_length or not np.isfinite(array).all():
        raise D2MonotonicDepthLayerWarpError(f"{name} must be a finite vector with at least {minimum_length} values")
    return array


def _open_uniform_cubic_knots(control_count: int) -> np.ndarray:
    if control_count < 4:
        raise D2MonotonicDepthLayerWarpError("D2 cubic B-spline requires at least four controls")
    interior_count = control_count - 4
    interior = np.linspace(0.0, 1.0, interior_count + 2, dtype=np.float64)[1:-1]
    return np.concatenate((np.zeros(4), interior, np.ones(4)))


def _evaluate_cubic_b_spline(controls: np.ndarray, parameters: np.ndarray) -> np.ndarray:
    """Evaluate an open-uniform cubic B-spline with Cox--de Boor bases."""

    knots = _open_uniform_cubic_knots(len(controls))
    parameter = np.clip(parameters, 0.0, 1.0)
    basis = np.zeros((len(parameter), len(controls)), dtype=np.float64)
    for index in range(len(controls)):
        basis[:, index] = (parameter >= knots[index]) & (parameter < knots[index + 1])
    basis[parameter == 1.0, -1] = 1.0
    for degree in range(1, 4):
        next_basis = np.zeros_like(basis)
        for index in range(len(controls)):
            left = knots[index + degree] - knots[index]
            right = knots[index + degree + 1] - knots[index + 1]
            if left > 0.0:
                next_basis[:, index] += ((parameter - knots[index]) / left) * basis[:, index]
            if right > 0.0 and index + 1 < len(controls):
                next_basis[:, index] += ((knots[index + degree + 1] - parameter) / right) * basis[:, index + 1]
        basis = next_basis
    return basis @ controls


def audit_d2_monotonic_depth_layer_warp(
    layers: Sequence[D2LayerWarp], *, actual_output_warp_pixel_count: int,
    config: D2MonotonicDepthLayerWarpConfig = D2MonotonicDepthLayerWarpConfig(),
) -> dict[str, object]:
    """Validate all three observed layers and return serialisable D2 evidence.

    The audit is intentionally fail-closed: identity parameters are geometrically
    valid but not an executed D2 component, so zero changed output pixels is
    rejected rather than being represented as a successful warp.
    """

    if not isinstance(actual_output_warp_pixel_count, int) or actual_output_warp_pixel_count <= 0:
        raise D2MonotonicDepthLayerWarpError("D2 requires actual_output_warp_pixel_count > 0")
    by_layer = {item.layer: item for item in layers}
    if set(by_layer) != set(_LAYERS) or len(layers) != len(_LAYERS):
        raise D2MonotonicDepthLayerWarpError("D2 requires exactly the far, mid, and near observed depth layers")
    audits: dict[str, object] = {}
    for layer in _LAYERS:
        warp = by_layer[layer]
        x = _finite_vector(warp.input_x, name=f"D2 {layer} input_x")
        out = _finite_vector(warp.output_x, name=f"D2 {layer} output_x")
        vertical = _finite_vector(warp.vertical_residual_y, name=f"D2 {layer} vertical_residual_y")
        if not (len(x) == len(out) == len(vertical)):
            raise D2MonotonicDepthLayerWarpError(f"D2 {layer} knot vectors must have matching lengths")
        if np.any(np.diff(x) <= 0.0):
            raise D2MonotonicDepthLayerWarpError(f"D2 {layer} input knots must be strictly increasing")
        increments = np.diff(out)
        if np.any(increments <= 0.0):
            raise D2MonotonicDepthLayerWarpError(f"D2 {layer} output increments must be strictly positive")
        parameter = np.linspace(0.0, 1.0, max(257, (len(x) - 1) * 64), dtype=np.float64)
        sampled_input = _evaluate_cubic_b_spline(x, parameter)
        mapped = _evaluate_cubic_b_spline(out, parameter)
        jacobian = np.gradient(mapped, parameter) / np.gradient(sampled_input, parameter)
        minimum_jacobian = float(np.min(jacobian))
        minimum_scale = float(np.min(jacobian))
        maximum_scale = float(np.max(jacobian))
        maximum_vertical = float(np.max(np.abs(vertical)))
        if minimum_jacobian < config.minimum_jacobian:
            raise D2MonotonicDepthLayerWarpError(f"D2 {layer} minimum Jacobian is below {config.minimum_jacobian}")
        if minimum_scale < config.minimum_scale or maximum_scale > config.maximum_scale:
            raise D2MonotonicDepthLayerWarpError(
                f"D2 {layer} horizontal scale is outside [{config.minimum_scale}, {config.maximum_scale}]"
            )
        if maximum_vertical > config.maximum_vertical_residual_pixels:
            raise D2MonotonicDepthLayerWarpError(
                f"D2 {layer} vertical residual exceeds {config.maximum_vertical_residual_pixels} px"
            )
        audits[layer] = {
            "parameterization": "open_uniform_cubic_b_spline", "knot_count": int(len(x)), "minimum_jacobian": minimum_jacobian,
            "horizontal_scale_range": [minimum_scale, maximum_scale],
            "maximum_vertical_residual_pixels": maximum_vertical,
            "positive_jacobian": True,
        }
    return {
        "renderer_component": "d2_monotonic_depth_layer_warp",
        "depth_layers": list(_LAYERS),
        "actual_output_warp_pixel_count": actual_output_warp_pixel_count,
        "minimum_jacobian_gate": config.minimum_jacobian,
        "horizontal_scale_gate": [config.minimum_scale, config.maximum_scale],
        "maximum_vertical_residual_pixels_gate": config.maximum_vertical_residual_pixels,
        "layers": audits,
    }


def layer_warp_from_mapping(layer: str, payload: Mapping[str, object]) -> D2LayerWarp:
    """Strictly deserialize one persisted D2 layer record for audit replay."""

    if layer not in _LAYERS or not isinstance(payload, Mapping):
        raise D2MonotonicDepthLayerWarpError("D2 layer payload is invalid")
    try:
        return D2LayerWarp(
            layer=layer,
            input_x=np.asarray(payload["input_x"], dtype=np.float64),
            output_x=np.asarray(payload["output_x"], dtype=np.float64),
            vertical_residual_y=np.asarray(payload["vertical_residual_y"], dtype=np.float64),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise D2MonotonicDepthLayerWarpError("D2 layer payload is malformed") from exc


def audit_d2_training_raft_and_lines(
    *, left_bgr: np.ndarray, right_bgr: np.ndarray, left_frame_id: int, right_frame_id: int,
    model_sha256: str, cuda_device: int = 0, minimum_line_length_px: int = 32,
) -> dict[str, object]:
    """Obtain candidate-only RAFT FB and automatic line evidence from real RGB.

    Labels are deliberately absent; long-line annotations remain evaluation
    data.  A missing local locked model, invalid FB field, or no automatic long
    line is a D2 rejection rather than a CPU/zero-flow fallback.
    """

    from .video_model_lock import verify_candidate_models
    from .video_raft_runtime import RAFTSmallRuntimeConfig, TorchvisionRAFTSmallRuntime

    if left_bgr.shape != right_bgr.shape or left_bgr.dtype != np.uint8 or left_bgr.ndim != 3:
        raise D2MonotonicDepthLayerWarpError("D2 RAFT evidence needs matching real BGR sources")
    locks = verify_candidate_models({"torchvision_raft_small_C_T_V2": model_sha256})
    if len(locks) != 1:
        raise D2MonotonicDepthLayerWarpError("D2 requires exactly one locked RAFT-small model")
    runtime = TorchvisionRAFTSmallRuntime(RAFTSmallRuntimeConfig(
        weights_path=locks[0].path, weights_sha256=locks[0].sha256, cuda_device=cuda_device,
    ))
    left_rgb = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2RGB)
    right_rgb = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2RGB)
    forward = runtime.estimate_pair(left_rgb, right_rgb, source_frame_id=left_frame_id, target_frame_id=right_frame_id)
    backward = runtime.estimate_pair(right_rgb, left_rgb, source_frame_id=right_frame_id, target_frame_id=left_frame_id)
    height, width = left_bgr.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    mapped_x, mapped_y = grid_x + forward.flow_xy[..., 0], grid_y + forward.flow_xy[..., 1]
    sampled_backward = cv2.remap(backward.flow_xy, mapped_x, mapped_y, cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
    fb = np.linalg.norm(forward.flow_xy + sampled_backward, axis=2)
    valid = np.isfinite(fb) & (mapped_x >= 0) & (mapped_x < width - 1) & (mapped_y >= 0) & (mapped_y < height - 1)
    if int(valid.sum()) < 256:
        raise D2MonotonicDepthLayerWarpError("D2 RAFT forward/backward evidence has insufficient valid support")
    fb_p95 = float(np.quantile(fb[valid], 0.95))
    if fb_p95 > 1.5:
        raise D2MonotonicDepthLayerWarpError("D2 RAFT forward/backward P95 exceeds 1.5 px")
    gray = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2GRAY)
    lines = cv2.HoughLinesP(cv2.Canny(gray, 48, 128), 1, np.pi / 180.0,
                             threshold=max(16, minimum_line_length_px // 2),
                             minLineLength=minimum_line_length_px, maxLineGap=4)
    line_count = 0 if lines is None else int(len(lines))
    if line_count == 0:
        raise D2MonotonicDepthLayerWarpError("D2 automatic long-line evidence found no valid line")
    return {
        "model": "torchvision_raft_small", "annotation_input": False,
        "left_frame_id": left_frame_id, "right_frame_id": right_frame_id,
        "forward_backward_p95_pixels": fb_p95, "forward_backward_gate_pixels": 1.5,
        "valid_forward_backward_sample_count": int(valid.sum()),
        "automatic_long_line_detector": "canny_houghlinesp",
        "minimum_line_length_px": minimum_line_length_px,
        "automatic_long_line_count": line_count,
        "raft_forward": forward.audit.as_dict(), "raft_backward": backward.audit.as_dict(),
    }


__all__ = [
    "D2LayerWarp", "D2MonotonicDepthLayerWarpConfig", "D2MonotonicDepthLayerWarpError",
    "audit_d2_monotonic_depth_layer_warp", "audit_d2_training_raft_and_lines", "layer_warp_from_mapping",
]
