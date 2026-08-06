"""Candidate-only Torch CUDA tile compositor.

This is the CUDA data-plane used by the v2 video renderer.  It deliberately
does not choose source frames, create a pose, or solve a seam: those decisions
belong to the real-source planner and are supplied as an already-audited owner
map.  Its narrowly scoped job is to make the rendering half of the contract
testable: calibration inverse grid, a single composed ``grid_sample`` per
source, strict owner write, and final-only host copies.

The module is not imported by the photo pipeline and never falls back to
NumPy/CuPy.  A caller must explicitly opt into it for a candidate run and
record the returned CUDA audit before it can claim a resident path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .video_gpu_runtime import GpuVideoFrame, ResidentVideoFrameCache


class TorchCudaVideoRendererError(RuntimeError):
    """Raised when a candidate CUDA tile violates its real-source contract."""


@dataclass(frozen=True)
class CudaRenderSource:
    """One already-resident real source plus its one-shot target grid.

    ``inverse_grid_xy`` has ``H x W x 2`` normalised source coordinates.  It
    must be created on the same CUDA device as the source.  It combines every
    permitted geometric operation for that source (calibration, pose/scan
    layout, and an accepted residual mesh) before the only RGB interpolation.
    """

    frame_id: int
    frame: GpuVideoFrame
    inverse_grid_xy: Any


@dataclass(frozen=True)
class CudaTileResult:
    """A device-resident rendered tile with a strict real-frame owner map."""

    panorama_bgr: Any
    owner_frame_id: Any
    valid_mask: Any
    audit: dict[str, object]


def _torch(cache: ResidentVideoFrameCache) -> Any:
    return cache.torch_module


def _require_cuda_cache(cache: ResidentVideoFrameCache) -> None:
    if not cache.cuda_active:
        raise TorchCudaVideoRendererError(
            "Torch CUDA candidate renderer requires an active CUDA resident frame cache"
        )


def calibrated_inverse_grid(
    cache: ResidentVideoFrameCache,
    *,
    height: int,
    width: int,
    source_height: int | None = None,
    source_width: int | None = None,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    raw_cx: float | None = None,
    raw_cy: float | None = None,
    distortion: Sequence[float] = (),
) -> Any:
    """Build an on-device calibrated-to-raw inverse sampling grid.

    The supplied target coordinate system is calibrated pixel space.  The
    Brown--Conrady forward distortion maps it to the single raw RGB sampling
    coordinate consumed by ``grid_sample``.  The grid contains no colour,
    pose, owner, or inferred source content and remains resident on CUDA.
    """

    _require_cuda_cache(cache)
    if not isinstance(height, int) or not isinstance(width, int) or height < 2 or width < 2:
        raise TorchCudaVideoRendererError("calibration grid dimensions must be integers >= 2")
    # Target tiles are often narrower than their genuine source image.  The
    # calibration math is in raw-source pixels, so normalisation for
    # grid_sample must use the source extent, never the output tile extent.
    source_height = height if source_height is None else source_height
    source_width = width if source_width is None else source_width
    if (
        not isinstance(source_height, int)
        or not isinstance(source_width, int)
        or source_height < 2
        or source_width < 2
    ):
        raise TorchCudaVideoRendererError("calibration source dimensions must be integers >= 2")
    raw_cx = cx if raw_cx is None else raw_cx
    raw_cy = cy if raw_cy is None else raw_cy
    values = (fx, fy, cx, cy, raw_cx, raw_cy, *distortion)
    if not all(isinstance(value, (int, float)) and np.isfinite(value) for value in values):
        raise TorchCudaVideoRendererError("calibration parameters must be finite")
    if fx <= 0.0 or fy <= 0.0:
        raise TorchCudaVideoRendererError("calibration focal lengths must be positive")
    if len(distortion) not in (0, 4, 5, 8):
        raise TorchCudaVideoRendererError("distortion must use 0, 4, 5, or 8 Brown-Conrady coefficients")
    coeffs = tuple(float(value) for value in distortion) + (0.0,) * (8 - len(distortion))
    k1, k2, p1, p2, k3, k4, k5, k6 = coeffs
    torch = _torch(cache)
    with cache.compute_context():
        ys, xs = torch.meshgrid(
            torch.arange(height, device=cache.device, dtype=torch.float32),
            torch.arange(width, device=cache.device, dtype=torch.float32),
            indexing="ij",
        )
        x = (xs - float(cx)) / float(fx)
        y = (ys - float(cy)) / float(fy)
        radius2 = x.square() + y.square()
        numerator = 1.0 + k1 * radius2 + k2 * radius2.square() + k3 * radius2.pow(3)
        denominator = 1.0 + k4 * radius2 + k5 * radius2.square() + k6 * radius2.pow(3)
        scale = numerator / denominator.clamp_min(1e-12)
        raw_x = float(fx) * (x * scale + 2.0 * p1 * x * y + p2 * (radius2 + 2.0 * x.square())) + float(raw_cx)
        raw_y = float(fy) * (y * scale + p1 * (radius2 + 2.0 * y.square()) + 2.0 * p2 * x * y) + float(raw_cy)
        return torch.stack(
            (
                2.0 * raw_x / float(source_width - 1) - 1.0,
                2.0 * raw_y / float(source_height - 1) - 1.0,
            ),
            dim=-1,
        ).contiguous()


def compose_inverse_grid(
    calibration_grid_xy: Any,
    *,
    scan_offset_xy: tuple[float, float] = (0.0, 0.0),
    residual_mesh_offset_xy: Any | None = None,
    source_height: int | None = None,
    source_width: int | None = None,
) -> Any:
    """Compose scan and accepted-mesh offsets before the one RGB resample."""

    if not hasattr(calibration_grid_xy, "ndim") or calibration_grid_xy.ndim != 3 or calibration_grid_xy.shape[-1] != 2:
        raise TorchCudaVideoRendererError("calibration_grid_xy must be HxWx2")
    if not all(isinstance(value, (int, float)) and np.isfinite(value) for value in scan_offset_xy):
        raise TorchCudaVideoRendererError("scan_offset_xy must be finite")
    grid = calibration_grid_xy.clone()
    height, width = int(grid.shape[0]), int(grid.shape[1])
    source_height = height if source_height is None else source_height
    source_width = width if source_width is None else source_width
    if (
        not isinstance(source_height, int)
        or not isinstance(source_width, int)
        or source_height < 2
        or source_width < 2
    ):
        raise TorchCudaVideoRendererError("inverse-grid source dimensions must be integers >= 2")
    # The scan offset is supplied in source pixels, then converted to the
    # normalised grid domain before a single ``grid_sample`` occurs.
    grid[..., 0].add_(2.0 * float(scan_offset_xy[0]) / float(source_width - 1))
    grid[..., 1].add_(2.0 * float(scan_offset_xy[1]) / float(source_height - 1))
    if residual_mesh_offset_xy is not None:
        if tuple(residual_mesh_offset_xy.shape) != tuple(grid.shape):
            raise TorchCudaVideoRendererError("residual mesh offset must match calibration grid HxWx2")
        if str(residual_mesh_offset_xy.device) != str(grid.device):
            raise TorchCudaVideoRendererError("residual mesh must remain on the calibration grid device")
        grid[..., 0].add_(2.0 * residual_mesh_offset_xy[..., 0] / float(source_width - 1))
        grid[..., 1].add_(2.0 * residual_mesh_offset_xy[..., 1] / float(source_height - 1))
    return grid.contiguous()


class TorchCudaCandidateTileRenderer:
    """Strict-owner compositor for a <=5-source CUDA resident tile window."""

    def __init__(self, cache: ResidentVideoFrameCache) -> None:
        _require_cuda_cache(cache)
        self.cache = cache

    def estimate_raft_flow(
        self,
        raft_runtime: Any,
        *,
        source: GpuVideoFrame,
        target: GpuVideoFrame,
    ) -> tuple[Any, dict[str, object]]:
        """Run verified RAFT-small on resident real sRGB tensors only.

        The finite reduction is a permitted tiny scalar audit.  It happens
        before a flow can enter a mesh/seam solver and never materialises a
        dense flow on the host.
        """

        if source.frame_id == target.frame_id:
            raise TorchCudaVideoRendererError("RAFT needs two distinct adjacent real source frames")
        if getattr(raft_runtime, "device", None) != self.cache.device:
            raise TorchCudaVideoRendererError("RAFT runtime and resident source cache must use one CUDA device")
        estimate = getattr(raft_runtime, "estimate_pair_tensors", None)
        if not callable(estimate):
            raise TorchCudaVideoRendererError("RAFT runtime lacks resident tensor inference")
        with self.cache.compute_context():
            result = estimate(
                source.color_u8,
                target.color_u8,
                source_frame_id=source.frame_id,
                target_frame_id=target.frame_id,
            )
            flow = getattr(result, "flow_xy", None)
            if flow is None or str(getattr(flow, "device", "")) != self.cache.device:
                raise TorchCudaVideoRendererError("RAFT tensor flow left the selected CUDA device")
            finite = bool(_torch(self.cache).isfinite(flow).all().item())
        if not finite:
            raise TorchCudaVideoRendererError("RAFT tensor flow contains a non-finite value")
        raw_audit = getattr(getattr(result, "audit", None), "as_dict", lambda: {})()
        if not isinstance(raw_audit, dict):
            raise TorchCudaVideoRendererError("RAFT tensor flow lacks a structured audit")
        return flow, {
            **raw_audit,
            "flow_finite": True,
            "flow_finite_audit": "on_device_scalar",
            "output_residency": "device_tensor",
            "host_transfer_count": 0,
        }

    def render_hard_owner(
        self,
        sources: Sequence[CudaRenderSource],
        owner_frame_id: Any,
    ) -> CudaTileResult:
        """Render one tile from genuine sources with exactly one owner/pixel.

        Pixels with a negative owner are invalid.  Every other owner must
        occur in precisely one supplied source and samples that source once.
        A mesh may be represented by the already-composed grid but may never
        alter provenance.
        """

        torch = _torch(self.cache)
        if not sources or len(sources) > self.cache.config.maximum_resident_frames:
            raise TorchCudaVideoRendererError("CUDA tile needs between 1 and maximum resident real sources")
        if not hasattr(owner_frame_id, "ndim") or owner_frame_id.ndim != 2:
            raise TorchCudaVideoRendererError("owner_frame_id must be a device HxW integer map")
        if str(owner_frame_id.device) != self.cache.device:
            raise TorchCudaVideoRendererError("owner_frame_id must remain on the selected CUDA device")
        ids = [int(source.frame_id) for source in sources]
        if len(ids) != len(set(ids)):
            raise TorchCudaVideoRendererError("CUDA tile sources must have unique real frame ids")
        height, width = tuple(int(value) for value in owner_frame_id.shape)
        output = torch.zeros((3, height, width), dtype=torch.uint8, device=self.cache.device)
        valid = torch.zeros((height, width), dtype=torch.bool, device=self.cache.device)
        owner = owner_frame_id.to(dtype=torch.int32).contiguous()
        grid_sample_count = 0
        with self.cache.compute_context():
            for source in sources:
                if source.frame.frame_id != source.frame_id:
                    raise TorchCudaVideoRendererError("CUDA source frame identity differs from its real cache entry")
                grid = source.inverse_grid_xy
                if tuple(grid.shape) != (height, width, 2) or str(grid.device) != self.cache.device:
                    raise TorchCudaVideoRendererError("CUDA source grid must match tile shape and device")
                mask = owner == int(source.frame_id)
                if not bool(torch.any(mask).item()):
                    continue
                sampled_rgb = torch.nn.functional.grid_sample(
                    source.frame.color_u8.unsqueeze(0).to(dtype=torch.float32),
                    grid.unsqueeze(0),
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=True,
                )[0].round().clamp_(0, 255).to(dtype=torch.uint8)
                # Source frames are kept resident in RGB order.  The video
                # delivery boundary, and every ``panorama_bgr`` consumer,
                # uses OpenCV BGR order.  Convert exactly once here, after
                # the sole permitted RGB inverse sample and before any owner
                # composition can consume the tile.
                sampled_bgr = sampled_rgb[[2, 1, 0]]
                # A requested owner outside the inverse grid's valid range is
                # invalid rather than a black fabricated source sample.
                # ``compose_inverse_grid`` may introduce a one-ULP CUDA
                # rounding excursion at a mathematically exact border.  Keep
                # that genuine boundary sample; this epsilon is far below a
                # pixel and does not legitimise a real out-of-source lookup.
                inside = (grid[..., 0].abs() <= 1.0 + 1e-6) & (grid[..., 1].abs() <= 1.0 + 1e-6)
                accepted = mask & inside
                output[:, accepted] = sampled_bgr[:, accepted]
                valid |= accepted
                grid_sample_count += 1
        if bool(torch.any((owner >= 0) & ~valid).item()):
            raise TorchCudaVideoRendererError("A claimed owner lacks a valid genuine CUDA source sample")
        audit = {
            "schema": "gemini305-video-torch-cuda-tile-renderer/v1",
            "candidate_only": True,
            "source_frame_ids": ids,
            "source_count": len(ids),
            "grid_sample_count": grid_sample_count,
            "single_composed_inverse_sample_per_source": True,
            "strict_single_owner": True,
            "invalid_owner_pixel_count": 0,
            "output_residency": "device_tensor",
            "intermediate_d2h_count": 0,
        }
        return CudaTileResult(output, owner, valid, audit)

    def finalize(self, tile: CudaTileResult) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
        """Perform the only permitted large D2H transfers for a finished tile."""

        panorama = self.cache.copy_final_to_cpu(tile.panorama_bgr, artifact="panorama")
        provenance = self.cache.copy_final_to_cpu(tile.owner_frame_id, artifact="provenance")
        audit = {**tile.audit, "gpu_runtime": self.cache.audit()}
        return (
            panorama.permute(1, 2, 0).contiguous().numpy(),
            provenance.contiguous().numpy(),
            audit,
        )


__all__ = [
    "CudaRenderSource",
    "CudaTileResult",
    "TorchCudaCandidateTileRenderer",
    "TorchCudaVideoRendererError",
    "calibrated_inverse_grid",
    "compose_inverse_grid",
]
