"""Automatic, resident long-line evidence for the C9 experiment.

The detector consumes only calibrated real RGB samples and the adjacent
RAFT-small fields already resident on CUDA.  It deliberately has no notion of
measurement annotations: labels remain post-publication evidence only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CudaLongLineEvidence:
    tracked_mask: Any
    audit: dict[str, object]


def detect_and_track_cuda_long_lines(
    torch: Any,
    *,
    bgr: Any,
    forward_xy: Any,
    backward_xy: Any,
    safe_mask: Any,
    minimum_length_px: int = 32,
    forward_backward_maximum_error_px: float = 1.5,
) -> CudaLongLineEvidence:
    """Detect horizontal/vertical long edges and retain RAFT-tracked cells.

    This is intentionally conservative.  A cell must be part of a long,
    directionally coherent image edge *and* have a finite bidirectional RAFT
    observation.  Rejection simply leaves C4's hard-owner pixels intact.
    """

    if not isinstance(minimum_length_px, int) or not 16 <= minimum_length_px <= 160:
        raise ValueError("minimum long-line length must be an integer in [16, 160]")
    if bgr.ndim != 3 or int(bgr.shape[0]) != 3 or forward_xy.shape != backward_xy.shape:
        raise ValueError("long-line detector needs matching resident BGR and RAFT fields")
    height, width = int(bgr.shape[1]), int(bgr.shape[2])
    if tuple(forward_xy.shape) != (height, width, 2) or tuple(safe_mask.shape) != (height, width):
        raise ValueError("long-line detector tensors must share one HxW domain")
    gray = bgr.to(dtype=torch.float32).mean(dim=0) / 255.0
    gx = torch.nn.functional.pad(gray[:, 1:] - gray[:, :-1], (0, 1))
    gy = torch.nn.functional.pad(gray[1:, :] - gray[:-1, :], (0, 0, 0, 1))
    magnitude = (gx.square() + gy.square()).sqrt()
    finite_magnitude = magnitude[torch.isfinite(magnitude)]
    threshold = (
        torch.quantile(finite_magnitude, 0.85)
        if int(finite_magnitude.numel())
        else torch.tensor(float("inf"), device=bgr.device)
    )
    # A zero quantile arises in mostly flat real frames.  ``>= 0`` would turn
    # every background pixel into a fictional line, so retain only a genuine
    # nonzero gradient in that case.
    edge = magnitude >= threshold.clamp_min(1.0e-6)
    # A horizontal physical line has a predominantly vertical image gradient;
    # the converse applies to vertical lines.  Pooling only along the tangent
    # is the automatic long-support test, not an annotation-derived region.
    half = minimum_length_px // 2
    horizontal_support = torch.nn.functional.avg_pool2d(
        edge.to(dtype=torch.float32)[None, None], (1, minimum_length_px), stride=1,
        padding=(0, half), count_include_pad=False,
    )[0, 0, :, :width]
    vertical_support = torch.nn.functional.avg_pool2d(
        edge.to(dtype=torch.float32)[None, None], (minimum_length_px, 1), stride=1,
        padding=(half, 0), count_include_pad=False,
    )[0, 0, :height, :]
    horizontal = edge & (gy.abs() >= gx.abs()) & (horizontal_support >= 0.65)
    vertical = edge & (gx.abs() > gy.abs()) & (vertical_support >= 0.65)
    line_mask = horizontal | vertical
    fb_error = (forward_xy + backward_xy).square().sum(dim=-1).sqrt()
    finite = torch.isfinite(forward_xy).all(dim=-1) & torch.isfinite(backward_xy).all(dim=-1)
    tracked = line_mask & safe_mask.bool() & finite & (fb_error <= float(forward_backward_maximum_error_px))
    p95 = None
    if int(fb_error[tracked].numel()):
        p95 = float(torch.quantile(fb_error[tracked].float(), 0.95).item())
    return CudaLongLineEvidence(
        tracked_mask=tracked,
        audit={
            "schema": "gemini305-video-cuda-long-line-raft/v1",
            "annotation_input": False,
            "detector": "resident_sobel_tangent_support",
            "minimum_length_px": minimum_length_px,
            "edge_threshold": float(threshold.item()),
            "horizontal_line_pixel_count": int(horizontal.sum().item()),
            "vertical_line_pixel_count": int(vertical.sum().item()),
            "detected_line_pixel_count": int(line_mask.sum().item()),
            "raft_tracked_line_pixel_count": int(tracked.sum().item()),
            "raft_forward_backward_error_p95_px": p95,
        },
    )


__all__ = ["CudaLongLineEvidence", "detect_and_track_cuda_long_lines"]
