"""Candidate-only v2 CUDA hard-owner renderer.

This is a deliberately narrow first production-grade data-plane: it renders a
pre-audited real-source strip plan on CUDA with each RGB-D source uploaded once
and an owner map constructed on-device.  Seam selection, depth mesh and
MultiBand remain separate planners; they may supply a richer composed grid in
a later v2 implementation, but cannot replace the real-source/owner contract
enforced here.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence

import numpy as np

from .video_algorithm_contract import (
    PreparedVideoAlgorithm,
    VideoAlgorithmContractError,
    VideoAlgorithmResult,
    VideoPanoramaAlgorithm,
)
from .video_cuda_renderer import (
    CudaRenderSource,
    TorchCudaCandidateTileRenderer,
    TorchCudaVideoRendererError,
    calibrated_inverse_grid,
    compose_inverse_grid,
)
from .video_cuda_constrained_owner import (
    CudaConstrainedOwnerError,
    constrained_curved_hard_owner,
)
from .video_gpu_runtime import ResidentVideoFrameCache, VideoGpuRuntimeConfig
from .video_raft_runtime import RAFTSmallRuntimeError
from .video_cuda_mesh import (
    CudaMeshError,
    estimate_cuda_dis_rgb_correspondence,
    fit_cuda_local_mesh,
)
from .video_cuda_depth_layers import CudaDepthLayerError, cuda_same_layer_safe_mask
from .video_cuda_pose_prior import CudaPosePriorError, cuda_pose_inverse_grid_from_target_depth
from .video_cuda_object_lock import (
    CudaObjectOwnerLockError,
    cuda_depth_object_protection,
    lock_cuda_protected_owner,
)
from .video_cuda_safe_multiband import (
    CudaSafeMultiBandError,
    blend_cuda_safe_multiband,
)
from .video_cuda_photometric import (
    CudaGlobalPhotometricResult,
    CudaPhotometricError,
    CudaPhotometricOverlap,
    apply_cuda_global_photometric_correction,
    solve_cuda_global_photometric,
)
from .video_cuda_multilabel_owner import (
    CudaMultilabelOwnerConfig,
    CudaMultilabelOwnerError,
    optimise_cuda_c8_local_multilabel_owner,
)


def _cuda_p95(torch: Any, values: Any) -> float | None:
    """Materialise only the scalar C3 forward/backward audit percentile."""

    if int(values.numel()) == 0:
        return None
    return float(torch.quantile(values.float(), 0.95).item())


def _median_exposure_anchor_index(exposures: Sequence[int | None]) -> int:
    """Choose C7's deterministic median-exposure gauge source.

    A fixed-exposure sequence has a tied median.  In that case choose the
    chronological middle source, rather than accidentally reverting to the
    source-0 chain that C7 is intended to replace.
    """

    if not exposures:
        raise VideoAlgorithmContractError("C7 needs at least one real source for its exposure anchor")
    # Formal session sources always carry positive exposure metadata.  The
    # temporal-middle fallback only keeps low-level synthetic renderer
    # fixtures (which have no session metadata) usable; it is recorded by the
    # caller's source audit and never applies to a public session route.
    if all(value is None for value in exposures):
        return len(exposures) // 2
    if not all(isinstance(value, int) and not isinstance(value, bool) and value > 0 for value in exposures):
        raise VideoAlgorithmContractError("C7 exposure metadata must be either all absent or all positive")
    median = sorted(int(value) for value in exposures)[len(exposures) // 2]
    candidates = [index for index, value in enumerate(exposures) if int(value) == median]
    return min(candidates, key=lambda index: (abs(index - len(exposures) // 2), index))


@dataclass(frozen=True)
class CudaRealSource:
    """One decoded, real RGB-D source available to the candidate renderer."""

    frame_id: int
    timestamp_us: int
    color_u8_rgb: np.ndarray
    depth_mm: np.ndarray
    camera_to_world: np.ndarray
    # The C7 global colour graph chooses its fixed gauge from real capture
    # metadata.  This is never used to synthesise exposure or to alter a
    # pose; it merely makes the documented median-exposure anchor explicit.
    color_exposure_raw: int | None = None
    # Legacy optional semantic raster.  Candidate C5--C8 never consume this
    # field: their protection is derived exclusively from aligned depth and
    # real-source support.  Keeping the field preserves low-level fixture/API
    # compatibility while preventing measurement annotations from becoming a
    # renderer input.
    object_mask: np.ndarray | None = None


@dataclass(frozen=True)
class CudaSourceStrip:
    """One non-overlapping output strip owned by exactly one real source."""

    frame_id: int
    output_x0: int
    source_x0: int
    width: int
    # Keep the unrounded layout centre for an expanded C1 pair window.  The
    # original four-field form remains valid for strict C0 unit fixtures.
    source_centre_x: float | None = field(default=None, compare=False)


class TorchCudaStripOwnerAlgorithm(VideoPanoramaAlgorithm):
    """Render an explicitly supplied, chronological real-source strip plan.

    The constructor receives the output layout from a separate audit/planner;
    it never estimates a 2-D pose or manufactures an intermediate source.  A
    frame can contribute one contiguous strip in this phase, which makes the
    H2D<=1 invariant mechanically enforceable while the planner evolves to
    tile windows and local mesh seams.
    """

    def __init__(
        self,
        *,
        sources: Sequence[CudaRealSource],
        strips: Sequence[CudaSourceStrip],
        output_height: int,
        output_width: int,
        calibration: Mapping[str, object],
        runtime_config: VideoGpuRuntimeConfig = VideoGpuRuntimeConfig(cuda_mode="required"),
    ) -> None:
        self.sources = tuple(sources)
        self.strips = tuple(strips)
        self.output_height = int(output_height)
        self.output_width = int(output_width)
        self.calibration = dict(calibration)
        self.runtime_config = runtime_config
        self._validate_static_plan()

    def _validate_static_plan(self) -> None:
        if self.output_height < 2 or self.output_width < 2:
            raise VideoAlgorithmContractError("v2 CUDA output dimensions must be >= 2")
        source_ids = tuple(item.frame_id for item in self.sources)
        if len(source_ids) < 2 or source_ids != tuple(sorted(source_ids)) or len(set(source_ids)) != len(source_ids):
            raise VideoAlgorithmContractError("v2 CUDA sources must be unique chronological real ids")
        strips_by_id = {strip.frame_id: strip for strip in self.strips}
        if not strips_by_id or not set(strips_by_id).issubset(source_ids) or len(strips_by_id) != len(self.strips):
            raise VideoAlgorithmContractError("v2 CUDA strips must belong to unique real sources")
        cursor = 0
        for strip in sorted(self.strips, key=lambda item: item.output_x0):
            if strip.output_x0 != cursor or strip.width < 1 or strip.source_x0 < 0:
                raise VideoAlgorithmContractError("v2 CUDA strips must be contiguous, non-empty, and non-overlapping")
            cursor += strip.width
        if cursor != self.output_width:
            raise VideoAlgorithmContractError("v2 CUDA strips must cover the complete output width")
        required = {"fx", "fy", "cx", "cy"}
        if set(self.calibration) < required:
            raise VideoAlgorithmContractError("v2 CUDA calibration requires fx/fy/cx/cy")
        for source in self.sources:
            color = np.asarray(source.color_u8_rgb)
            depth = np.asarray(source.depth_mm)
            pose = np.asarray(source.camera_to_world)
            if (
                color.dtype != np.uint8
                or color.ndim != 3
                or color.shape[2] != 3
                or color.shape[0] != self.output_height
                or color.shape[1] < 2
            ):
                raise VideoAlgorithmContractError("v2 CUDA source RGB must be uint8 with output height")
            if depth.shape != color.shape[:2] or pose.shape != (4, 4) or not np.isfinite(pose).all():
                raise VideoAlgorithmContractError("v2 CUDA source depth/pose is invalid")
            if source.object_mask is not None:
                object_mask = np.asarray(source.object_mask)
                if object_mask.dtype != np.bool_ or object_mask.shape != color.shape[:2]:
                    raise VideoAlgorithmContractError("v2 CUDA source object mask must be bool and source-shaped")
            strip = strips_by_id.get(source.frame_id)
            if strip is not None and strip.source_x0 + strip.width > color.shape[1]:
                raise VideoAlgorithmContractError("v2 CUDA strip lies outside its real source RGB")

    def prepare(
        self,
        *,
        session: Any,
        online_state: Any | None,
        context: Mapping[str, object],
    ) -> PreparedVideoAlgorithm:
        del session, online_state
        # Pair plans must be supplied by the audited C1/C8 planner.  The v2
        # renderer only accepts them after verifying their exact real source
        # sequence through PreparedVideoAlgorithm.
        pair_plans = context.get("pair_plans")
        if not isinstance(pair_plans, tuple):
            raise VideoAlgorithmContractError("v2 CUDA prepare requires immutable pair_plans")
        # This narrow data-plane executes only a literal one-owner strip
        # layout.  It must not accept a C1 curved seam, local mesh, object
        # lock, or MultiBand plan and then merely report it as completed.
        # Those plans are rejected until their corresponding device operator
        # is present in the render path.
        unsupported = [
            plan
            for plan in pair_plans
            if (
                plan.flow_backend != "none"
                or plan.use_raft_backward
                or plan.use_depth_mesh
                or plan.object_lock_required
                or plan.seam_mode != "hard_owner"
                or plan.blend_mode != "none"
            )
        ]
        if unsupported:
            raise VideoAlgorithmContractError(
                "torch_cuda_strip_owner_v2 executes only hard_owner/no-flow/no-mesh pair plans"
            )
        return PreparedVideoAlgorithm(
            source_frame_ids=tuple(source.frame_id for source in self.sources),
            camera_to_world=tuple(np.asarray(source.camera_to_world) for source in self.sources),
            pair_plans=pair_plans,
            context_audit={
                "renderer": "torch_cuda_strip_owner_v2",
                "real_sources_only": True,
                "interpolated_pose_count": 0,
                "components": {"cuda_calibration_and_strict_owner_data_plane": True},
            },
        )

    def render(self, prepared: PreparedVideoAlgorithm) -> VideoAlgorithmResult:
        if prepared.source_frame_ids != tuple(source.frame_id for source in self.sources):
            raise VideoAlgorithmContractError("prepared source ids differ from v2 CUDA real-source plan")
        cache = ResidentVideoFrameCache(self.runtime_config)
        tile_renderer = TorchCudaCandidateTileRenderer(cache)
        torch = cache.torch_module
        try:
            # No frame upload exists before this allocation, so it cannot
            # enter the cache's upload-dependent compute stream yet.
            panorama = torch.zeros(
                (3, self.output_height, self.output_width), dtype=torch.uint8, device=cache.device
            )
            owner = torch.full(
                (self.output_height, self.output_width), -1, dtype=torch.int32, device=cache.device
            )
            strips = {strip.frame_id: strip for strip in self.strips}
            for source in self.sources:
                frame = cache.upload(
                    frame_id=source.frame_id,
                    timestamp_us=source.timestamp_us,
                    color_u8=np.ascontiguousarray(source.color_u8_rgb),
                    depth_mm=np.ascontiguousarray(source.depth_mm),
                    pose_prior=np.ascontiguousarray(source.camera_to_world, dtype=np.float32),
                )
                strip = strips.get(source.frame_id)
                # A near-duplicate real node can legitimately have no final
                # owner pixels.  It nevertheless performs exactly one device
                # calibrated remap before discard, preserving source identity
                # and the one-upload/one-remap audit without inventing owner
                # coverage or skipping its real RGB-D input.
                remap_width = strip.width if strip is not None else int(source.color_u8_rgb.shape[1])
                remap_cx = (
                    float(self.calibration["cx"]) - float(strip.source_x0)
                    if strip is not None
                    else float(self.calibration["cx"])
                )
                grid = calibrated_inverse_grid(
                    cache,
                    height=self.output_height,
                    width=remap_width,
                    source_height=int(source.color_u8_rgb.shape[0]),
                    source_width=int(source.color_u8_rgb.shape[1]),
                    fx=float(self.calibration["fx"]),
                    fy=float(self.calibration["fy"]),
                    cx=remap_cx,
                    cy=float(self.calibration["cy"]),
                    raw_cx=float(self.calibration["cx"]),
                    raw_cy=float(self.calibration["cy"]),
                    distortion=tuple(float(item) for item in self.calibration.get("distortion", ())),
                )
                local_owner = torch.full(
                    (self.output_height, remap_width), source.frame_id,
                    dtype=torch.int32,
                    device=cache.device,
                )
                local = tile_renderer.render_hard_owner(
                    [CudaRenderSource(source.frame_id, frame, grid)], local_owner
                )
                with cache.compute_context():
                    if strip is not None:
                        # ``render_hard_owner`` already returns BGR, the
                        # canonical delivery order.  Do not permute again.
                        panorama[:, :, strip.output_x0 : strip.output_x0 + strip.width] = local.panorama_bgr
                        owner[:, strip.output_x0 : strip.output_x0 + strip.width] = local.owner_frame_id
                cache.release(source.frame_id)
            if bool(torch.any(owner < 0).item()):
                raise TorchCudaVideoRendererError("v2 CUDA strip plan left an unowned output pixel")
            panorama_cpu = cache.copy_final_to_cpu(panorama, artifact="panorama")
            owner_cpu = cache.copy_final_to_cpu(owner, artifact="provenance")
            audit = {
                "schema": "gemini305-video-visual-renderer-v2/v1",
                "renderer": "torch_cuda_strip_owner_v2",
                "candidate_only": True,
                "source_frame_ids": list(prepared.source_frame_ids),
                "pair_plans": [plan.as_dict() for plan in prepared.pair_plans],
                "interpolated_pose_count": 0,
                "strict_single_owner": True,
                "cuda_resident": True,
                "gpu_runtime": cache.audit(),
            }
            return VideoAlgorithmResult(
                panorama_bgr=panorama_cpu.permute(1, 2, 0).contiguous().numpy(),
                owner_frame_id=owner_cpu.contiguous().numpy(),
                source_frame_ids=prepared.source_frame_ids,
                algorithm_audit=audit,
            )
        finally:
            cache.close()


@dataclass(frozen=True)
class CudaC1ConstrainedOwnerConfig:
    """Closed, pair-local controls for the first C1 CUDA integration."""

    corridor_width_pixels: int = 96
    maximum_row_step_pixels: int = 4
    first_order_penalty: float = 5.0
    second_order_penalty: float = 3.0

    def __post_init__(self) -> None:
        if not 8 <= self.corridor_width_pixels <= 256:
            raise VideoAlgorithmContractError("C1 CUDA corridor width must be in [8, 256]")
        if not 1 <= self.maximum_row_step_pixels <= 16:
            raise VideoAlgorithmContractError("C1 CUDA maximum row step must be in [1, 16]")
        if self.first_order_penalty < 0.0 or self.second_order_penalty < 0.0:
            raise VideoAlgorithmContractError("C1 CUDA curvature penalties must be non-negative")


@dataclass(frozen=True)
class _CudaC1Window:
    frame_id: int
    output_x0: int
    output_x1: int
    panorama_bgr: Any
    valid_mask: Any
    inverse_grid: Any


class TorchCudaC1ConstrainedOwnerAlgorithm(TorchCudaStripOwnerAlgorithm):
    """CUDA C1 renderer with one calibrated remap for each real source.

    Each source is sampled exactly once into its owner strip plus its two
    bounded neighbouring risk corridors.  Adjacent windows then supply the
    actual device-resident RGB samples and validity masks to the C1 dynamic
    programme.  No host image is consulted until the final panorama and
    provenance downloads.
    """

    def __init__(self, *, c1_config: CudaC1ConstrainedOwnerConfig = CudaC1ConstrainedOwnerConfig(), **kwargs: Any) -> None:
        self.c1_config = c1_config
        super().__init__(**kwargs)
        if len(self.strips) != len(self.sources):
            raise VideoAlgorithmContractError(
                "C1 CUDA requires every chronological real source to retain a non-empty owner strip"
            )

    @staticmethod
    def _source_support(strip: CudaSourceStrip, source_width: int) -> tuple[int, int]:
        # The layout's scalar target-to-source relation is x_raw =
        # source_x0 + (x_output - output_x0).  It is only layout placement,
        # never a pose replacement or colour computation.
        return strip.output_x0 - strip.source_x0, strip.output_x0 - strip.source_x0 + source_width

    def _pair_corridors(self) -> tuple[tuple[int, int], ...]:
        strips = {item.frame_id: item for item in self.strips}
        result: list[tuple[int, int]] = []
        for first, second in zip(self.sources[:-1], self.sources[1:], strict=True):
            first_strip, second_strip = strips[first.frame_id], strips[second.frame_id]
            boundary = first_strip.output_x0 + first_strip.width
            if boundary != second_strip.output_x0:
                raise VideoAlgorithmContractError("C1 CUDA source strips must meet at each chronological boundary")
            first_support = self._source_support(first_strip, int(first.color_u8_rgb.shape[1]))
            second_support = self._source_support(second_strip, int(second.color_u8_rgb.shape[1]))
            shared_left = max(first_support[0], second_support[0], 0)
            shared_right = min(first_support[1], second_support[1], self.output_width)
            width = self.c1_config.corridor_width_pixels
            if shared_right - shared_left < width:
                raise VideoAlgorithmContractError(
                    "C1 CUDA pair has insufficient genuine calibrated common support for its locked corridor"
                )
            start = min(max(boundary - width // 2, shared_left), shared_right - width)
            result.append((int(start), int(start + width)))
        return tuple(result)

    def prepare(
        self,
        *,
        session: Any,
        online_state: Any | None,
        context: Mapping[str, object],
    ) -> PreparedVideoAlgorithm:
        del session, online_state
        pair_plans = context.get("pair_plans")
        if not isinstance(pair_plans, tuple):
            raise VideoAlgorithmContractError("v2 CUDA C1 prepare requires immutable pair_plans")
        invalid = [
            plan
            for plan in pair_plans
            if (
                plan.flow_backend != "none"
                or plan.use_raft_backward
                or plan.use_depth_mesh
                or plan.object_lock_required
                or plan.seam_mode != "curved_hard_owner"
                or plan.blend_mode != "none"
                or not plan.use_open3d
            )
        ]
        if invalid:
            raise VideoAlgorithmContractError(
                "torch_cuda_c1_constrained_owner_v2 executes only audited curved_hard_owner/no-flow/no-mesh pairs"
            )
        # Calculate and validate all pair corridors before an upload.  A C1
        # declaration cannot silently downgrade an unsupported pair to C0.
        self._pair_corridors()
        return PreparedVideoAlgorithm(
            source_frame_ids=tuple(source.frame_id for source in self.sources),
            camera_to_world=tuple(np.asarray(source.camera_to_world) for source in self.sources),
            pair_plans=pair_plans,
            context_audit={
                "renderer": "torch_cuda_c1_constrained_owner_v2",
                "real_sources_only": True,
                "interpolated_pose_count": 0,
                "components": {"cuda_calibration_and_c1_constrained_owner_data_plane": True},
            },
        )

    def _render_window(
        self,
        *,
        cache: ResidentVideoFrameCache,
        tile_renderer: TorchCudaCandidateTileRenderer,
        source: CudaRealSource,
        strip: CudaSourceStrip,
        output_x0: int,
        output_x1: int,
    ) -> tuple[Any, _CudaC1Window]:
        if output_x1 - output_x0 < 2:
            raise VideoAlgorithmContractError("C1 CUDA source window must be at least two pixels wide")
        frame = cache.upload(
            frame_id=source.frame_id,
            timestamp_us=source.timestamp_us,
            color_u8=np.ascontiguousarray(source.color_u8_rgb),
            depth_mm=np.ascontiguousarray(source.depth_mm),
            pose_prior=np.ascontiguousarray(source.camera_to_world, dtype=np.float32),
            # Semantic masks never enter a CUDA rendering upload.  Fixed
            # annotations are read only by the post-publication evaluator.
            object_mask=None,
        )
        centre = (
            float(strip.source_centre_x)
            if strip.source_centre_x is not None
            else float(self.calibration["cx"]) + float(strip.output_x0) - float(strip.source_x0)
        )
        grid = calibrated_inverse_grid(
            cache,
            height=self.output_height,
            width=output_x1 - output_x0,
            source_height=int(source.color_u8_rgb.shape[0]),
            source_width=int(source.color_u8_rgb.shape[1]),
            fx=float(self.calibration["fx"]),
            fy=float(self.calibration["fy"]),
            cx=centre - float(output_x0),
            cy=float(self.calibration["cy"]),
            raw_cx=float(self.calibration["cx"]),
            raw_cy=float(self.calibration["cy"]),
            distortion=tuple(float(item) for item in self.calibration.get("distortion", ())),
        )
        torch = cache.torch_module
        local_owner = torch.full(
            (self.output_height, output_x1 - output_x0), source.frame_id,
            dtype=torch.int32, device=cache.device,
        )
        tile = tile_renderer.render_hard_owner(
            [CudaRenderSource(source.frame_id, self._sampling_frame(source, frame), grid)], local_owner
        )
        return frame, _CudaC1Window(
            source.frame_id, output_x0, output_x1, tile.panorama_bgr, tile.valid_mask, grid
        )

    def _sampling_frame(self, source: CudaRealSource, frame: Any) -> Any:
        """Return the resident real source used for its one final remap.

        C7 overrides this narrow hook with a device-resident, globally
        accepted colour correction.  The frame identity, depth, object mask,
        pose, inverse grid, and owner are deliberately retained unchanged.
        """

        del source
        return frame

    def _before_c1_render(self, cache: ResidentVideoFrameCache) -> None:
        """Optional device-only preparation after the render cache exists."""

        del cache

    def _retain_source_frames_for_final_owner(self) -> bool:
        """Whether a descendant needs every real source after C1/C7 output.

        Normal C1--C7 operation releases an old local source once its last
        adjacent corridor has been rendered.  C8 is the sole exception: its
        bounded 2--5-source owner windows execute *after* C7/C6 and must
        resample resident original sources without a second H2D upload.
        """

        return False

    def _apply_final_post_owner(
        self,
        *,
        cache: ResidentVideoFrameCache,
        panorama_bgr: Any,
        owner_frame_id: Any,
        prepared: PreparedVideoAlgorithm,
    ) -> tuple[Any, Any, dict[str, object] | None]:
        """Optional whole-output candidate extension before the two final D2H copies."""

        del cache, prepared
        return panorama_bgr, owner_frame_id, None

    def _apply_pair_post_owner(
        self,
        *,
        cache: ResidentVideoFrameCache,
        tile_renderer: TorchCudaCandidateTileRenderer,
        first_frame: Any,
        second_frame: Any,
        first_window: _CudaC1Window,
        second_window: _CudaC1Window,
        corridor_output_x0: int,
        first_bgr: Any,
        second_bgr: Any,
        composed_bgr: Any,
        first_valid: Any,
        second_valid: Any,
        owner_frame_id: Any,
    ) -> tuple[Any, dict[str, object] | None]:
        """C1 has no colour post-process after its owner decision."""

        del cache, tile_renderer, first_frame, second_frame, first_window, second_window, corridor_output_x0
        del first_valid, second_valid, owner_frame_id
        return composed_bgr, None

    def render(self, prepared: PreparedVideoAlgorithm) -> VideoAlgorithmResult:
        if prepared.source_frame_ids != tuple(source.frame_id for source in self.sources):
            raise VideoAlgorithmContractError("prepared source ids differ from v2 CUDA C1 real-source plan")
        # Derived renderers such as C7 may need a larger audited resident
        # set for a global fit.  The base C1 route itself supplies a bound of
        # two, so retaining the configured limit keeps one-H2D enforcement
        # intact without hard-coding C1's window size into every descendant.
        cache = ResidentVideoFrameCache(self.runtime_config)
        tile_renderer = TorchCudaCandidateTileRenderer(cache)
        torch = cache.torch_module
        try:
            corridors = self._pair_corridors()
            strips = {strip.frame_id: strip for strip in self.strips}
            windows: list[tuple[int, int]] = []
            for index, source in enumerate(self.sources):
                strip = strips[source.frame_id]
                starts = [strip.output_x0]
                ends = [strip.output_x0 + strip.width]
                if index:
                    starts.append(corridors[index - 1][0])
                if index < len(corridors):
                    ends.append(corridors[index][1])
                support = self._source_support(strip, int(source.color_u8_rgb.shape[1]))
                window = max(min(starts), support[0], 0), min(max(ends), support[1], self.output_width)
                if window[0] > min(starts) or window[1] < max(ends):
                    raise VideoAlgorithmContractError("C1 CUDA source window would sample outside a genuine real RGB source")
                windows.append(window)
            self._before_c1_render(cache)
            panorama = torch.zeros((3, self.output_height, self.output_width), dtype=torch.uint8, device=cache.device)
            owner = torch.full((self.output_height, self.output_width), -1, dtype=torch.int32, device=cache.device)
            previous_frame = None
            previous_window: _CudaC1Window | None = None
            pair_audits: list[dict[str, object]] = []
            owner_changed = 0
            retain_sources = self._retain_source_frames_for_final_owner()
            for index, source in enumerate(self.sources):
                strip = strips[source.frame_id]
                frame, current = self._render_window(
                    cache=cache, tile_renderer=tile_renderer, source=source, strip=strip,
                    output_x0=windows[index][0], output_x1=windows[index][1],
                )
                base_left = strip.output_x0 - current.output_x0
                base_right = base_left + strip.width
                with cache.compute_context():
                    panorama[:, :, strip.output_x0 : strip.output_x0 + strip.width] = current.panorama_bgr[:, :, base_left:base_right]
                    owner[:, strip.output_x0 : strip.output_x0 + strip.width] = torch.where(
                        current.valid_mask[:, base_left:base_right],
                        torch.full((self.output_height, strip.width), source.frame_id, dtype=torch.int32, device=cache.device),
                        torch.full((self.output_height, strip.width), -1, dtype=torch.int32, device=cache.device),
                    )
                if previous_window is not None and previous_frame is not None:
                    x0, x1 = corridors[index - 1]
                    first_slice = slice(x0 - previous_window.output_x0, x1 - previous_window.output_x0)
                    second_slice = slice(x0 - current.output_x0, x1 - current.output_x0)
                    first_bgr = previous_window.panorama_bgr[:, :, first_slice]
                    second_bgr = current.panorama_bgr[:, :, second_slice]
                    first_valid = previous_window.valid_mask[:, first_slice]
                    second_valid = current.valid_mask[:, second_slice]
                    with cache.compute_context():
                        first_gray = first_bgr.to(dtype=torch.float32).mean(dim=0)
                        second_gray = second_bgr.to(dtype=torch.float32).mean(dim=0)
                        gradient = (first_gray[:, 1:] - first_gray[:, :-1]).abs()
                        gradient = torch.nn.functional.pad(gradient, (0, 1))
                        gradient -= torch.nn.functional.pad((second_gray[:, 1:] - second_gray[:, :-1]).abs(), (0, 1))
                        seam_cost = (first_gray - second_gray).abs() + 0.25 * gradient.abs()
                    try:
                        c1 = constrained_curved_hard_owner(
                            torch,
                            seam_cost=seam_cost,
                            first_valid_mask=first_valid,
                            second_valid_mask=second_valid,
                            first_frame_id=previous_window.frame_id,
                            second_frame_id=current.frame_id,
                            corridor_x=(0, x1 - x0),
                            maximum_row_step_pixels=self.c1_config.maximum_row_step_pixels,
                            first_order_penalty=self.c1_config.first_order_penalty,
                            second_order_penalty=self.c1_config.second_order_penalty,
                        )
                    except CudaConstrainedOwnerError as exc:
                        raise VideoAlgorithmContractError(f"C1 CUDA owner audit failed: {exc}") from exc
                    with cache.compute_context():
                        before = owner[:, x0:x1]
                        changed = int((before != c1.owner_frame_id).sum().item())
                        owner_changed += changed
                        composed = torch.where(
                            (c1.owner_frame_id == previous_window.frame_id)[None, :, :],
                            first_bgr,
                            second_bgr,
                        )
                    composed, post_audit = self._apply_pair_post_owner(
                        cache=cache,
                        tile_renderer=tile_renderer,
                        first_frame=previous_frame,
                        second_frame=frame,
                        first_window=previous_window,
                        second_window=current,
                        corridor_output_x0=x0,
                        first_bgr=first_bgr,
                        second_bgr=second_bgr,
                        composed_bgr=composed,
                        first_valid=first_valid,
                        second_valid=second_valid,
                        owner_frame_id=c1.owner_frame_id,
                    )
                    with cache.compute_context():
                        panorama[:, :, x0:x1] = composed
                        owner[:, x0:x1] = c1.owner_frame_id
                    pair_audit = {**c1.audit, "owner_pixels_changed_from_initial_hard_strip": changed}
                    if post_audit is not None:
                        pair_audit["post_owner"] = post_audit
                    pair_audits.append(pair_audit)
                    if not retain_sources:
                        cache.release(previous_frame.frame_id)
                previous_frame, previous_window = frame, current
            if previous_frame is not None and not retain_sources:
                cache.release(previous_frame.frame_id)
            panorama, owner, final_post_owner_audit = self._apply_final_post_owner(
                cache=cache,
                panorama_bgr=panorama,
                owner_frame_id=owner,
                prepared=prepared,
            )
            if bool(torch.any(owner < 0).item()):
                raise TorchCudaVideoRendererError("C1 CUDA render left an invalid or unowned output pixel")
            panorama_cpu = cache.copy_final_to_cpu(panorama, artifact="panorama")
            owner_cpu = cache.copy_final_to_cpu(owner, artifact="provenance")
            audit = {
                "schema": "gemini305-video-visual-renderer-v2/v1",
                "renderer": "torch_cuda_c1_constrained_owner_v2",
                "candidate_only": True,
                "source_frame_ids": list(prepared.source_frame_ids),
                "pair_plans": [plan.as_dict() for plan in prepared.pair_plans],
                "interpolated_pose_count": 0,
                "strict_single_owner": True,
                "cuda_resident": True,
                "c1_constrained_owner": {
                    "pair_count": len(pair_audits),
                    "pair_audits": pair_audits,
                    "owner_pixels_changed_from_initial_hard_strip": owner_changed,
                    "executed_and_affected_owner_output": owner_changed > 0,
                },
                "executed_candidate_components": {
                    "c1_constrained_owner": owner_changed > 0,
                },
                "gpu_runtime": cache.audit(),
            }
            if final_post_owner_audit is not None:
                audit["final_post_owner"] = final_post_owner_audit
            if retain_sources:
                for frame_id in tuple(cache.resident_frame_ids):
                    cache.release(frame_id)
            return VideoAlgorithmResult(
                panorama_bgr=panorama_cpu.permute(1, 2, 0).contiguous().numpy(),
                owner_frame_id=owner_cpu.contiguous().numpy(),
                source_frame_ids=prepared.source_frame_ids,
                algorithm_audit=audit,
            )
        finally:
            cache.close()


class TorchCudaC2DisResidualMeshAlgorithm(TorchCudaC1ConstrainedOwnerAlgorithm):
    """C2's C1 owner path plus an accepted CUDA DIS RGB residual mesh.

    The mesh has no authority over camera poses, source selection, or owner
    provenance.  It is a pair-local source-grid offset applied only where the
    C1 hard owner still selects the first real source and every mesh audit
    passes.  A rejected correspondence or mesh leaves C1's exact pixels in
    place, which is the required fail-closed fallback.
    """

    def __init__(self, **kwargs: Any) -> None:
        self._c2_pair_audits: list[dict[str, object]] = []
        super().__init__(**kwargs)

    def prepare(
        self,
        *,
        session: Any,
        online_state: Any | None,
        context: Mapping[str, object],
    ) -> PreparedVideoAlgorithm:
        del session, online_state
        pair_plans = context.get("pair_plans")
        if not isinstance(pair_plans, tuple):
            raise VideoAlgorithmContractError("v2 CUDA C2 prepare requires immutable pair_plans")
        invalid = [
            plan
            for plan in pair_plans
            if (
                plan.flow_backend != "dis"
                or plan.use_raft_backward
                or plan.use_depth_mesh
                or plan.object_lock_required
                or plan.seam_mode != "curved_hard_owner"
                or plan.blend_mode != "none"
                or not plan.use_open3d
            )
        ]
        if invalid:
            raise VideoAlgorithmContractError(
                "torch_cuda_c2_dis_residual_mesh_v2 executes only C1 curved_hard_owner plus audited DIS RGB meshes"
            )
        self._pair_corridors()
        return PreparedVideoAlgorithm(
            source_frame_ids=tuple(source.frame_id for source in self.sources),
            camera_to_world=tuple(np.asarray(source.camera_to_world) for source in self.sources),
            pair_plans=pair_plans,
            context_audit={
                "renderer": "torch_cuda_c2_dis_residual_mesh_v2",
                "real_sources_only": True,
                "interpolated_pose_count": 0,
                "components": {
                    "cuda_calibration_and_c1_constrained_owner_data_plane": True,
                    "cuda_dis_rgb_residual_mesh_data_plane": True,
                },
            },
        )

    def _apply_pair_post_owner(
        self,
        *,
        cache: ResidentVideoFrameCache,
        tile_renderer: TorchCudaCandidateTileRenderer,
        first_frame: Any,
        second_frame: Any,
        first_window: _CudaC1Window,
        second_window: _CudaC1Window,
        corridor_output_x0: int,
        first_bgr: Any,
        second_bgr: Any,
        composed_bgr: Any,
        first_valid: Any,
        second_valid: Any,
        owner_frame_id: Any,
    ) -> tuple[Any, dict[str, object] | None]:
        torch = cache.torch_module
        try:
            first_grid = first_window.inverse_grid[:, :, :]
            # ``first_window`` contains a larger per-source support window;
            # this hook sees exactly its corridor slice, so the grid must be
            # cut to the same HxW domain before composing a *single* final
            # calibration+mesh sample for the pixels that will use it.
            grid_width = int(first_bgr.shape[2])
            if first_grid.shape[1] < grid_width:
                raise VideoAlgorithmContractError("C2 CUDA grid cannot cover its real pair corridor")
            # The caller's first BGR slice has the exact corridor width.  Its
            # start is recoverable from the output windows and does not infer
            # a source pose or a new coordinate system.
            offset = int(corridor_output_x0 - first_window.output_x0)
            if offset < 0 or offset + grid_width > int(first_grid.shape[1]):
                raise VideoAlgorithmContractError("C2 CUDA pair corridor lies outside the first real window")
            second_offset = int(corridor_output_x0 - second_window.output_x0)
            if second_offset < 0 or second_offset + grid_width > int(second_window.inverse_grid.shape[1]):
                raise VideoAlgorithmContractError("C2 CUDA pair corridor lies outside the second real window")
            second_grid = second_window.inverse_grid[:, second_offset : second_offset + grid_width]
            source_height, source_width = int(first_frame.color_u8.shape[1]), int(first_frame.color_u8.shape[2])
            with cache.compute_context():
                # The target grid belongs to the adjacent real source at this
                # output corridor.  Reproject that source's *aligned* depth
                # with both immutable ORB camera_to_world poses, then require
                # a z-consistent real first-source sample.  This is an inverse
                # sampling prior only; no panorama depth, colour, owner, or
                # pose is created and every unsafe pixel remains exact C1.
                pose_prior = cuda_pose_inverse_grid_from_target_depth(
                    torch,
                    target_inverse_grid_xy=second_grid,
                    source_depth_mm=first_frame.depth_mm,
                    target_depth_mm=second_frame.depth_mm,
                    source_camera_to_world=first_frame.pose_prior,
                    target_camera_to_world=second_frame.pose_prior,
                    fx=float(self.calibration["fx"]),
                    fy=float(self.calibration["fy"]),
                    cx=float(self.calibration["cx"]),
                    cy=float(self.calibration["cy"]),
                    distortion=tuple(float(item) for item in self.calibration.get("distortion", ())),
                )
                pose_warped = tile_renderer.render_hard_owner(
                    [CudaRenderSource(first_window.frame_id, first_frame, pose_prior.inverse_grid_xy)],
                    torch.where(
                        pose_prior.safe_mask,
                        torch.full((int(first_valid.shape[0]), grid_width), first_window.frame_id, dtype=torch.int32, device=cache.device),
                        torch.full((int(first_valid.shape[0]), grid_width), -1, dtype=torch.int32, device=cache.device),
                    ),
                )
                pose_valid = pose_prior.safe_mask & pose_warped.valid_mask & second_valid.bool()
            correspondence = estimate_cuda_dis_rgb_correspondence(
                torch,
                first_bgr=pose_warped.panorama_bgr,
                second_bgr=second_bgr,
                first_valid_mask=pose_valid,
                second_valid_mask=second_valid,
            )
            height, width = int(first_valid.shape[0]), int(first_valid.shape[1])
            yy, xx = torch.meshgrid(
                torch.arange(height, device=cache.device),
                torch.arange(width, device=cache.device),
                indexing="ij",
            )
            # Spatially interleaved train/held-out supports are deterministic
            # and remain wholly inside this adjacent device corridor.
            train = ((xx + yy) & 1) == 0
            held_out = ~train
            mesh = fit_cuda_local_mesh(
                torch,
                flow_xy=correspondence.forward_xy,
                training_mask=train,
                held_out_mask=held_out,
                safe_mask=correspondence.safe_mask & pose_valid,
                protected_mask=~pose_valid,
            )
            audit: dict[str, object] = {
                "orb_rgbd_inverse_grid_prior": pose_prior.audit,
                "dis_correspondence": correspondence.audit,
                "mesh": mesh.audit,
                "fallback_to_c1_hard_owner": not bool(mesh.audit.get("accepted", False)),
                "actual_output_mesh_pixel_count": 0,
            }
            if not bool(mesh.audit.get("accepted", False)):
                self._c2_pair_audits.append(audit)
                return composed_bgr, audit
            composed_grid = compose_inverse_grid(
                pose_prior.inverse_grid_xy,
                residual_mesh_offset_xy=mesh.offset_xy,
                source_height=source_height,
                source_width=source_width,
            )
            # Resample only accepted, still-C1-owned evidence pixels.  A
            # residual mesh may legitimately leave the source bounds at a
            # corridor edge; those pixels are not a valid new owner claim and
            # must retain their exact C1 sample.
            composed_inside = (
                (composed_grid[..., 0].abs() <= 1.0 + 1e-6)
                & (composed_grid[..., 1].abs() <= 1.0 + 1e-6)
            )
            mesh_candidate = (
                mesh.accepted_mask
                & pose_valid
                & (owner_frame_id == first_window.frame_id)
                & composed_inside
            )
            local_owner = torch.where(
                mesh_candidate,
                torch.full((height, width), first_window.frame_id, dtype=torch.int32, device=cache.device),
                torch.full((height, width), -1, dtype=torch.int32, device=cache.device),
            )
            warped = tile_renderer.render_hard_owner(
                [CudaRenderSource(first_window.frame_id, first_frame, composed_grid)], local_owner
            )
            with cache.compute_context():
                apply = (
                    mesh_candidate
                    & (mesh.offset_xy.abs().sum(dim=-1) > 1.0e-6)
                    & warped.valid_mask
                    & (owner_frame_id == first_window.frame_id)
                )
                applied = int(apply.sum().item())
                result = torch.where(apply[None, :, :], warped.panorama_bgr, composed_bgr)
            audit["actual_output_mesh_pixel_count"] = applied
            audit["fallback_to_c1_hard_owner"] = applied == 0
            audit["mesh_applied_to_actual_output"] = applied > 0
            self._c2_pair_audits.append(audit)
            return result, audit
        except (CudaMeshError, CudaPosePriorError, TorchCudaVideoRendererError, VideoAlgorithmContractError) as exc:
            audit = {
                "mesh_audit_exception": str(exc),
                "fallback_to_c1_hard_owner": True,
                "actual_output_mesh_pixel_count": 0,
                "mesh_applied_to_actual_output": False,
            }
            self._c2_pair_audits.append(audit)
            return composed_bgr, audit

    def render(self, prepared: PreparedVideoAlgorithm) -> VideoAlgorithmResult:
        self._c2_pair_audits = []
        result = super().render(prepared)
        applied = sum(int(item.get("actual_output_mesh_pixel_count", 0)) for item in self._c2_pair_audits)
        audit = dict(result.algorithm_audit)
        audit["renderer"] = "torch_cuda_c2_dis_residual_mesh_v2"
        audit["c2_dis_mesh"] = {
            "pair_count": len(self._c2_pair_audits),
            "pair_audits": self._c2_pair_audits,
            "actual_output_mesh_pixel_count": applied,
            "executed_and_affected_output": applied > 0,
        }
        c1 = dict(audit.get("c1_constrained_owner", {}))
        c1_executed = bool(c1.get("executed_and_affected_owner_output", False))
        audit["executed_candidate_components"] = {
            "c1_constrained_owner": c1_executed,
            "c2_dis_mesh": applied > 0,
        }
        return VideoAlgorithmResult(
            panorama_bgr=result.panorama_bgr,
            owner_frame_id=result.owner_frame_id,
            source_frame_ids=result.source_frame_ids,
            algorithm_audit=audit,
            artifact_sources=result.artifact_sources,
        )


class TorchCudaC3RAFTResidualMeshAlgorithm(TorchCudaC1ConstrainedOwnerAlgorithm):
    """C3's C1 owner path plus verified, resident RAFT-small residual meshes.

    C3 is deliberately not a variant of C2's inverse-search implementation.
    It obtains both directions of flow from the locked RAFT-small runtime using
    the original adjacent RGB sources already resident in the cache.  The
    resulting full-source fields are sampled onto the one C1 corridor, audited
    there, and then composed into the first source's one final inverse grid.
    Thus RAFT never changes a pose, manufactures a source, or changes an owner.
    Any model, correspondence, or mesh failure retains the exact C1 colour and
    owner for that pair.
    """

    def __init__(
        self,
        *,
        raft_runtime: Any,
        forward_backward_maximum_error_px: float = 1.5,
        **kwargs: Any,
    ) -> None:
        if not isinstance(forward_backward_maximum_error_px, (int, float)) or not np.isfinite(
            forward_backward_maximum_error_px
        ) or not 0.0 < float(forward_backward_maximum_error_px) <= 8.0:
            raise VideoAlgorithmContractError(
                "C3 CUDA forward/backward maximum error must be finite in (0, 8]"
            )
        if raft_runtime is None:
            raise VideoAlgorithmContractError("C3 CUDA requires a verified RAFT-small runtime")
        self.raft_runtime = raft_runtime
        self.forward_backward_maximum_error_px = float(forward_backward_maximum_error_px)
        self._c3_pair_audits: list[dict[str, object]] = []
        super().__init__(**kwargs)

    def prepare(
        self,
        *,
        session: Any,
        online_state: Any | None,
        context: Mapping[str, object],
    ) -> PreparedVideoAlgorithm:
        del session, online_state
        pair_plans = context.get("pair_plans")
        if not isinstance(pair_plans, tuple):
            raise VideoAlgorithmContractError("v2 CUDA C3 prepare requires immutable pair_plans")
        invalid = [
            plan
            for plan in pair_plans
            if (
                plan.flow_backend != "raft_small"
                or not plan.use_raft_backward
                or plan.use_depth_mesh
                or plan.object_lock_required
                or plan.seam_mode != "curved_hard_owner"
                or plan.blend_mode != "none"
                or not plan.use_open3d
            )
        ]
        if invalid:
            raise VideoAlgorithmContractError(
                "torch_cuda_c3_raft_residual_mesh_v2 executes only C1 curved_hard_owner "
                "plus bidirectional RAFT-small RGB meshes"
            )
        self._pair_corridors()
        return PreparedVideoAlgorithm(
            source_frame_ids=tuple(source.frame_id for source in self.sources),
            camera_to_world=tuple(np.asarray(source.camera_to_world) for source in self.sources),
            pair_plans=pair_plans,
            context_audit={
                "renderer": "torch_cuda_c3_raft_residual_mesh_v2",
                "real_sources_only": True,
                "interpolated_pose_count": 0,
                "components": {
                    "cuda_calibration_and_c1_constrained_owner_data_plane": True,
                    "cuda_raft_small_bidirectional_rgb_residual_mesh_data_plane": True,
                },
            },
        )

    @staticmethod
    def _sample_field(torch: Any, field_xy: Any, grid: Any) -> Any:
        """Sample an HxWx2 resident source field onto an inverse-grid tile."""

        if getattr(field_xy, "ndim", None) != 3 or int(field_xy.shape[-1]) != 2:
            raise VideoAlgorithmContractError("C3 RAFT must return an HxWx2 resident flow field")
        if str(field_xy.device) != str(grid.device):
            raise VideoAlgorithmContractError("C3 RAFT flow and calibration grid must share one CUDA device")
        return torch.nn.functional.grid_sample(
            field_xy.permute(2, 0, 1).unsqueeze(0),
            grid.unsqueeze(0),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )[0].permute(1, 2, 0).contiguous()

    @staticmethod
    def _layout_residual_flow(
        torch: Any,
        *,
        forward_source_to_target_xy: Any,
        first_grid: Any,
        second_grid: Any,
        source_width: int,
        source_height: int,
        target_width: int,
        target_height: int,
    ) -> tuple[Any, Any]:
        """Subtract C1's already-applied raw-coordinate displacement.

        RAFT operates on the two full calibrated source images, hence its
        flow contains the large real camera translation between those raw
        coordinate systems.  C1 has already represented that translation by
        sampling ``first_grid`` and ``second_grid`` at the same canvas point.
        A local mesh must fit only the remaining inverse-sampling correction;
        treating the raw RAFT field as that correction incorrectly rejects
        otherwise-small residual meshes at the formal 8 px bound.
        """

        if tuple(first_grid.shape) != tuple(second_grid.shape):
            raise VideoAlgorithmContractError("C3 C1 grids must match for a local residual mesh")
        if tuple(forward_source_to_target_xy.shape) != (*tuple(first_grid.shape[:2]), 2):
            raise VideoAlgorithmContractError("C3 sampled RAFT field must match its C1 corridor grid")
        if min(source_width, source_height, target_width, target_height) < 2:
            raise VideoAlgorithmContractError("C3 source and target extents must support normalized grids")
        first_raw = torch.stack(
            (
                (first_grid[..., 0] + 1.0) * (float(source_width - 1) * 0.5),
                (first_grid[..., 1] + 1.0) * (float(source_height - 1) * 0.5),
            ),
            dim=-1,
        )
        second_raw = torch.stack(
            (
                (second_grid[..., 0] + 1.0) * (float(target_width - 1) * 0.5),
                (second_grid[..., 1] + 1.0) * (float(target_height - 1) * 0.5),
            ),
            dim=-1,
        )
        expected_source_to_target = second_raw - first_raw
        return expected_source_to_target - forward_source_to_target_xy, expected_source_to_target

    def _apply_pair_post_owner(
        self,
        *,
        cache: ResidentVideoFrameCache,
        tile_renderer: TorchCudaCandidateTileRenderer,
        first_frame: Any,
        second_frame: Any,
        first_window: _CudaC1Window,
        second_window: _CudaC1Window,
        corridor_output_x0: int,
        first_bgr: Any,
        second_bgr: Any,
        composed_bgr: Any,
        first_valid: Any,
        second_valid: Any,
        owner_frame_id: Any,
    ) -> tuple[Any, dict[str, object] | None]:
        torch = cache.torch_module
        try:
            forward_full, forward_audit = tile_renderer.estimate_raft_flow(
                self.raft_runtime, source=first_frame, target=second_frame
            )
            backward_full, backward_audit = tile_renderer.estimate_raft_flow(
                self.raft_runtime, source=second_frame, target=first_frame
            )
            source_height, source_width = int(first_frame.color_u8.shape[1]), int(first_frame.color_u8.shape[2])
            target_height, target_width = int(second_frame.color_u8.shape[1]), int(second_frame.color_u8.shape[2])
            if tuple(forward_full.shape) != (source_height, source_width, 2):
                raise VideoAlgorithmContractError("C3 forward RAFT flow does not match its real source extent")
            if tuple(backward_full.shape) != (target_height, target_width, 2):
                raise VideoAlgorithmContractError("C3 backward RAFT flow does not match its real source extent")
            grid_width = int(first_bgr.shape[2])
            offset = int(corridor_output_x0 - first_window.output_x0)
            if offset < 0 or offset + grid_width > int(first_window.inverse_grid.shape[1]):
                raise VideoAlgorithmContractError("C3 CUDA pair corridor lies outside the first real window")
            first_grid = first_window.inverse_grid[:, offset : offset + grid_width]
            if tuple(first_grid.shape[:2]) != tuple(first_valid.shape):
                raise VideoAlgorithmContractError("C3 CUDA calibration grid does not match its pair corridor")
            second_offset = int(corridor_output_x0 - second_window.output_x0)
            if second_offset < 0 or second_offset + grid_width > int(second_window.inverse_grid.shape[1]):
                raise VideoAlgorithmContractError("C3 CUDA pair corridor lies outside the second real window")
            second_grid = second_window.inverse_grid[:, second_offset : second_offset + grid_width]
            with cache.compute_context():
                forward = self._sample_field(torch, forward_full, first_grid)
                residual, expected = self._layout_residual_flow(
                    torch,
                    forward_source_to_target_xy=forward,
                    first_grid=first_grid,
                    second_grid=second_grid,
                    source_width=source_width,
                    source_height=source_height,
                    target_width=target_width,
                    target_height=target_height,
                )
                target_grid = first_grid.clone()
                target_grid[..., 0].add_(2.0 * forward[..., 0] / float(target_width - 1))
                target_grid[..., 1].add_(2.0 * forward[..., 1] / float(target_height - 1))
                backward = self._sample_field(torch, backward_full, target_grid)
                target_inside = (target_grid[..., 0].abs() <= 1.0) & (target_grid[..., 1].abs() <= 1.0)
                # C3 is RGB-only.  Until C4 introduces depth layers, strong
                # local RGB structure is conservatively retained by C1 rather
                # than allowing a mesh to cross a possible foreground edge.
                gray = first_bgr.to(dtype=torch.float32).mean(dim=0) / 255.0
                gx = torch.nn.functional.pad((gray[:, 1:] - gray[:, :-1]).abs(), (0, 1))
                gy = torch.nn.functional.pad((gray[1:, :] - gray[:-1, :]).abs(), (0, 0, 0, 1))
                gradient = gx + gy
                texture_threshold = torch.quantile(gradient, 0.80)
                fb_error = (forward + backward).square().sum(dim=-1).sqrt()
                safe = (
                    first_valid.bool()
                    & second_valid.bool()
                    & target_inside
                    & (fb_error <= self.forward_backward_maximum_error_px)
                    & (gradient <= texture_threshold)
                )
                height, width = int(first_valid.shape[0]), int(first_valid.shape[1])
                yy, xx = torch.meshgrid(
                    torch.arange(height, device=cache.device),
                    torch.arange(width, device=cache.device),
                    indexing="ij",
                )
                train = ((xx + yy) & 1) == 0
                held_out = ~train
            mesh = fit_cuda_local_mesh(
                torch,
                flow_xy=residual,
                training_mask=train,
                held_out_mask=held_out,
                safe_mask=safe,
                protected_mask=~safe,
            )
            audit: dict[str, object] = {
                "raft_forward": forward_audit,
                "raft_backward": backward_audit,
                "forward_backward_maximum_error_px": self.forward_backward_maximum_error_px,
                "forward_backward_error_p95_px": _cuda_p95(torch, fb_error[safe]),
                "raw_raft_flow_magnitude_p95_px": _cuda_p95(torch, forward.square().sum(dim=-1).sqrt()[safe]),
                "c1_expected_raw_flow_magnitude_p95_px": _cuda_p95(torch, expected.square().sum(dim=-1).sqrt()[safe]),
                "residual_flow_magnitude_p95_px": _cuda_p95(torch, residual.square().sum(dim=-1).sqrt()[safe]),
                "safe_pixel_count": int(safe.sum().item()),
                "high_structure_protected": True,
                "mesh": mesh.audit,
                "fallback_to_c1_hard_owner": not bool(mesh.audit.get("accepted", False)),
                "actual_output_mesh_pixel_count": 0,
            }
            if not bool(mesh.audit.get("accepted", False)):
                self._c3_pair_audits.append(audit)
                return composed_bgr, audit
            composed_grid = compose_inverse_grid(
                first_grid,
                residual_mesh_offset_xy=mesh.offset_xy,
                source_height=source_height,
                source_width=source_width,
            )
            local_owner = torch.full(
                (height, width), first_window.frame_id, dtype=torch.int32, device=cache.device
            )
            warped = tile_renderer.render_hard_owner(
                [CudaRenderSource(first_window.frame_id, first_frame, composed_grid)], local_owner
            )
            with cache.compute_context():
                apply = (
                    mesh.accepted_mask
                    & (mesh.offset_xy.abs().sum(dim=-1) > 1.0e-6)
                    & warped.valid_mask
                    & (owner_frame_id == first_window.frame_id)
                )
                applied = int(apply.sum().item())
                result = torch.where(apply[None, :, :], warped.panorama_bgr, composed_bgr)
            audit["actual_output_mesh_pixel_count"] = applied
            audit["fallback_to_c1_hard_owner"] = applied == 0
            audit["mesh_applied_to_actual_output"] = applied > 0
            self._c3_pair_audits.append(audit)
            return result, audit
        except (
            CudaMeshError,
            RAFTSmallRuntimeError,
            TorchCudaVideoRendererError,
            VideoAlgorithmContractError,
        ) as exc:
            audit = {
                "raft_or_mesh_exception": str(exc),
                "fallback_to_c1_hard_owner": True,
                "actual_output_mesh_pixel_count": 0,
                "mesh_applied_to_actual_output": False,
            }
            self._c3_pair_audits.append(audit)
            return composed_bgr, audit

    def render(self, prepared: PreparedVideoAlgorithm) -> VideoAlgorithmResult:
        self._c3_pair_audits = []
        result = super().render(prepared)
        applied = sum(int(item.get("actual_output_mesh_pixel_count", 0)) for item in self._c3_pair_audits)
        audit = dict(result.algorithm_audit)
        audit["renderer"] = "torch_cuda_c3_raft_residual_mesh_v2"
        audit["c3_raft_mesh"] = {
            "pair_count": len(self._c3_pair_audits),
            "pair_audits": self._c3_pair_audits,
            "actual_output_mesh_pixel_count": applied,
            "executed_and_affected_output": applied > 0,
        }
        c1 = dict(audit.get("c1_constrained_owner", {}))
        audit["executed_candidate_components"] = {
            "c1_constrained_owner": bool(c1.get("executed_and_affected_owner_output", False)),
            "c3_raft_mesh": applied > 0,
        }
        return VideoAlgorithmResult(
            panorama_bgr=result.panorama_bgr,
            owner_frame_id=result.owner_frame_id,
            source_frame_ids=result.source_frame_ids,
            algorithm_audit=audit,
            artifact_sources=result.artifact_sources,
        )


class TorchCudaC4RAFTDepthLayeredMeshAlgorithm(TorchCudaC3RAFTResidualMeshAlgorithm):
    """C4's C1+C3 route with an additional calibrated RGB-D same-layer gate.

    C4 neither inherits C2 nor relabels an RGB-only mesh as depth-aware.  A
    mesh pixel is sampled only when it passes the complete C3 RAFT/held-out
    mesh audit *and* C4's resident aligned-depth correspondence gate.  Depth
    rejection preserves the C1 hard-owner colour and provenance.
    """

    def __init__(self, **kwargs: Any) -> None:
        self._c4_pair_audits: list[dict[str, object]] = []
        super().__init__(**kwargs)

    def prepare(
        self,
        *,
        session: Any,
        online_state: Any | None,
        context: Mapping[str, object],
    ) -> PreparedVideoAlgorithm:
        del session, online_state
        pair_plans = context.get("pair_plans")
        if not isinstance(pair_plans, tuple):
            raise VideoAlgorithmContractError("v2 CUDA C4 prepare requires immutable pair_plans")
        invalid = [
            plan
            for plan in pair_plans
            if (
                plan.flow_backend != "raft_small"
                or not plan.use_raft_backward
                or not plan.use_depth_mesh
                or plan.object_lock_required
                or plan.seam_mode != "curved_hard_owner"
                or plan.blend_mode != "none"
                or not plan.use_open3d
            )
        ]
        if invalid:
            raise VideoAlgorithmContractError(
                "torch_cuda_c4_raft_rgbd_layered_mesh_v2 executes only C1 curved_hard_owner "
                "plus bidirectional RAFT-small and audited RGB-D same-layer meshes"
            )
        self._pair_corridors()
        return PreparedVideoAlgorithm(
            source_frame_ids=tuple(source.frame_id for source in self.sources),
            camera_to_world=tuple(np.asarray(source.camera_to_world) for source in self.sources),
            pair_plans=pair_plans,
            context_audit={
                "renderer": "torch_cuda_c4_raft_rgbd_layered_mesh_v2",
                "real_sources_only": True,
                "interpolated_pose_count": 0,
                "components": {
                    "cuda_calibration_and_c1_constrained_owner_data_plane": True,
                    "cuda_raft_small_bidirectional_rgb_residual_mesh_data_plane": True,
                    "cuda_rgbd_same_layer_mesh_protection_data_plane": True,
                },
            },
        )

    def _apply_pair_post_owner(
        self,
        *,
        cache: ResidentVideoFrameCache,
        tile_renderer: TorchCudaCandidateTileRenderer,
        first_frame: Any,
        second_frame: Any,
        first_window: _CudaC1Window,
        second_window: _CudaC1Window,
        corridor_output_x0: int,
        first_bgr: Any,
        second_bgr: Any,
        composed_bgr: Any,
        first_valid: Any,
        second_valid: Any,
        owner_frame_id: Any,
    ) -> tuple[Any, dict[str, object] | None]:
        torch = cache.torch_module
        try:
            forward_full, forward_audit = tile_renderer.estimate_raft_flow(
                self.raft_runtime, source=first_frame, target=second_frame
            )
            backward_full, backward_audit = tile_renderer.estimate_raft_flow(
                self.raft_runtime, source=second_frame, target=first_frame
            )
            source_height, source_width = int(first_frame.color_u8.shape[1]), int(first_frame.color_u8.shape[2])
            target_height, target_width = int(second_frame.color_u8.shape[1]), int(second_frame.color_u8.shape[2])
            if tuple(forward_full.shape) != (source_height, source_width, 2):
                raise VideoAlgorithmContractError("C4 forward RAFT flow does not match its real source extent")
            if tuple(backward_full.shape) != (target_height, target_width, 2):
                raise VideoAlgorithmContractError("C4 backward RAFT flow does not match its real source extent")
            grid_width = int(first_bgr.shape[2])
            offset = int(corridor_output_x0 - first_window.output_x0)
            if offset < 0 or offset + grid_width > int(first_window.inverse_grid.shape[1]):
                raise VideoAlgorithmContractError("C4 CUDA pair corridor lies outside the first real window")
            first_grid = first_window.inverse_grid[:, offset : offset + grid_width]
            if tuple(first_grid.shape[:2]) != tuple(first_valid.shape):
                raise VideoAlgorithmContractError("C4 CUDA calibration grid does not match its pair corridor")
            second_offset = int(corridor_output_x0 - second_window.output_x0)
            if second_offset < 0 or second_offset + grid_width > int(second_window.inverse_grid.shape[1]):
                raise VideoAlgorithmContractError("C4 CUDA pair corridor lies outside the second real window")
            second_grid = second_window.inverse_grid[:, second_offset : second_offset + grid_width]
            with cache.compute_context():
                # C4 is the depth-layered candidate: establish its local
                # first-source sampling prior from the adjacent real target
                # depth and immutable ORB poses before asking RAFT for the
                # remaining source-coordinate motion.  C1's scalar layout is
                # not treated as a pose substitute.  Unsafe reprojections
                # stay C1-owned below.
                pose_prior = cuda_pose_inverse_grid_from_target_depth(
                    torch,
                    target_inverse_grid_xy=second_grid,
                    source_depth_mm=first_frame.depth_mm,
                    target_depth_mm=second_frame.depth_mm,
                    source_camera_to_world=first_frame.pose_prior,
                    target_camera_to_world=second_frame.pose_prior,
                    fx=float(self.calibration["fx"]),
                    fy=float(self.calibration["fy"]),
                    cx=float(self.calibration["cx"]),
                    cy=float(self.calibration["cy"]),
                    distortion=tuple(float(item) for item in self.calibration.get("distortion", ())),
                )
                first_pose_grid = pose_prior.inverse_grid_xy
                forward = self._sample_field(torch, forward_full, first_pose_grid)
                residual, expected = self._layout_residual_flow(
                    torch,
                    forward_source_to_target_xy=forward,
                    first_grid=first_pose_grid,
                    second_grid=second_grid,
                    source_width=source_width,
                    source_height=source_height,
                    target_width=target_width,
                    target_height=target_height,
                )
                target_grid = first_pose_grid.clone()
                target_grid[..., 0].add_(2.0 * forward[..., 0] / float(target_width - 1))
                target_grid[..., 1].add_(2.0 * forward[..., 1] / float(target_height - 1))
                backward = self._sample_field(torch, backward_full, target_grid)
                target_inside = (target_grid[..., 0].abs() <= 1.0) & (target_grid[..., 1].abs() <= 1.0)
                # Both depth samples are real aligned RGB-D inputs.  They are
                # mapped with the same calibrated grids that define colour
                # sampling; no depth panorama is constructed or retained.
                first_depth = torch.nn.functional.grid_sample(
                    first_frame.depth_mm.unsqueeze(0).unsqueeze(0), first_pose_grid.unsqueeze(0),
                    mode="nearest", padding_mode="zeros", align_corners=True,
                )[0, 0]
                second_depth = torch.nn.functional.grid_sample(
                    second_frame.depth_mm.unsqueeze(0).unsqueeze(0), target_grid.unsqueeze(0),
                    mode="nearest", padding_mode="zeros", align_corners=True,
                )[0, 0]
                depth_safe, depth_audit = cuda_same_layer_safe_mask(
                    torch,
                    first_depth_mm=first_depth,
                    second_depth_mm=second_depth,
                    forward_flow_xy=forward,
                    backward_flow_xy=backward,
                    absolute_tolerance_mm=20.0,
                    relative_tolerance=0.02,
                    forward_backward_maximum_error_px=self.forward_backward_maximum_error_px,
                    second_depth_is_already_forward_warped=True,
                )
                first_pose_bgr = torch.nn.functional.grid_sample(
                    first_frame.color_u8.unsqueeze(0).to(dtype=torch.float32), first_pose_grid.unsqueeze(0),
                    mode="bilinear", padding_mode="zeros", align_corners=True,
                )[0].round().clamp_(0, 255).to(dtype=torch.uint8)[[2, 1, 0]]
                gray = first_pose_bgr.to(dtype=torch.float32).mean(dim=0) / 255.0
                gx = torch.nn.functional.pad((gray[:, 1:] - gray[:, :-1]).abs(), (0, 1))
                gy = torch.nn.functional.pad((gray[1:, :] - gray[:-1, :]).abs(), (0, 0, 0, 1))
                gradient = gx + gy
                texture_threshold = torch.quantile(gradient, 0.80)
                fb_error = (forward + backward).square().sum(dim=-1).sqrt()
                finite_flow = (
                    torch.isfinite(forward).all(dim=-1)
                    & torch.isfinite(backward).all(dim=-1)
                    & torch.isfinite(residual).all(dim=-1)
                    & torch.isfinite(fb_error)
                )
                c3_safe = (
                    first_valid.bool() & second_valid.bool() & pose_prior.safe_mask & target_inside
                    & (fb_error <= self.forward_backward_maximum_error_px)
                    & (gradient <= texture_threshold)
                    & finite_flow
                )
                safe = c3_safe & depth_safe
                height, width = int(first_valid.shape[0]), int(first_valid.shape[1])
                yy, xx = torch.meshgrid(
                    torch.arange(height, device=cache.device), torch.arange(width, device=cache.device), indexing="ij"
                )
                train = ((xx + yy) & 1) == 0
                held_out = ~train
            # The mesh fitter performs device-local interpolation internally;
            # even protected pixels must therefore be finite so an invalid
            # RAFT sample cannot poison neighboring accepted coefficients.
            # ``safe`` remains the authority for which real pixels may train,
            # validate, or reach output.
            finite_residual = torch.where(
                finite_flow[..., None], residual, torch.zeros_like(residual)
            )
            mesh = fit_cuda_local_mesh(
                torch, flow_xy=finite_residual, training_mask=train, held_out_mask=held_out,
                safe_mask=safe, protected_mask=~safe, protected_boundary_taper_px=8,
                # Taper only true RGB-D/ORB protection boundaries.  Strong
                # RGB texture remains owner-only but is not dilated into an
                # unrelated geometry exclusion band.
                identity_taper_mask=(~depth_safe) | (~pose_prior.safe_mask),
            )
            c3_mesh_candidate = mesh.accepted_mask & (mesh.offset_xy.abs().sum(dim=-1) > 1.0e-6)
            # These C3-admissible mesh locations are deliberately retained as
            # hard owner because their real aligned depths disagree.  The
            # accepted C4 mesh itself is limited to ``safe`` below, so this is
            # the exact excluded output domain, not a diagnostic-only mask.
            depth_protected = c3_safe & ~depth_safe
            audit: dict[str, object] = {
                "raft_forward": forward_audit,
                "raft_backward": backward_audit,
                "orb_rgbd_inverse_grid_prior": pose_prior.audit,
                "forward_backward_maximum_error_px": self.forward_backward_maximum_error_px,
                "forward_backward_error_p95_px": _cuda_p95(torch, fb_error[safe]),
                "raw_raft_flow_magnitude_p95_px": _cuda_p95(torch, forward.square().sum(dim=-1).sqrt()[safe]),
                "c1_expected_raw_flow_magnitude_p95_px": _cuda_p95(torch, expected.square().sum(dim=-1).sqrt()[safe]),
                "residual_flow_magnitude_p95_px": _cuda_p95(torch, residual.square().sum(dim=-1).sqrt()[safe]),
                "c3_rgb_safe_pixel_count": int(c3_safe.sum().item()),
                "depth_layers": depth_audit,
                "c4_same_layer_safe_pixel_count": int(safe.sum().item()),
                "non_finite_flow_protected_pixel_count": int((~finite_flow).sum().item()),
                "depth_protected_mesh_candidate_pixel_count": int(depth_protected.sum().item()),
                "mesh": mesh.audit,
                "fallback_to_c1_hard_owner": not bool(mesh.audit.get("accepted", False)),
                "actual_output_mesh_pixel_count": 0,
            }
            if not bool(mesh.audit.get("accepted", False)):
                self._c4_pair_audits.append(audit)
                return composed_bgr, audit
            composed_grid = compose_inverse_grid(
                first_pose_grid, residual_mesh_offset_xy=mesh.offset_xy,
                source_height=source_height, source_width=source_width,
            )
            composed_inside = (
                (composed_grid[..., 0].abs() <= 1.0 + 1e-6)
                & (composed_grid[..., 1].abs() <= 1.0 + 1e-6)
            )
            mesh_candidate = (
                c3_mesh_candidate
                & pose_prior.safe_mask
                & (owner_frame_id == first_window.frame_id)
                & composed_inside
            )
            local_owner = torch.where(
                mesh_candidate,
                torch.full((height, width), first_window.frame_id, dtype=torch.int32, device=cache.device),
                torch.full((height, width), -1, dtype=torch.int32, device=cache.device),
            )
            warped = tile_renderer.render_hard_owner(
                [CudaRenderSource(first_window.frame_id, first_frame, composed_grid)], local_owner
            )
            with cache.compute_context():
                apply = mesh_candidate & warped.valid_mask
                applied = int(apply.sum().item())
                result = torch.where(apply[None, :, :], warped.panorama_bgr, composed_bgr)
            audit["actual_output_mesh_pixel_count"] = applied
            audit["fallback_to_c1_hard_owner"] = applied == 0
            audit["mesh_applied_to_actual_output"] = applied > 0
            self._c4_pair_audits.append(audit)
            return result, audit
        except (
            CudaDepthLayerError, CudaMeshError, RAFTSmallRuntimeError,
            TorchCudaVideoRendererError, VideoAlgorithmContractError,
        ) as exc:
            audit = {
                "raft_depth_or_mesh_exception": str(exc),
                "fallback_to_c1_hard_owner": True,
                "actual_output_mesh_pixel_count": 0,
                "mesh_applied_to_actual_output": False,
            }
            self._c4_pair_audits.append(audit)
            return composed_bgr, audit

    def render(self, prepared: PreparedVideoAlgorithm) -> VideoAlgorithmResult:
        self._c4_pair_audits = []
        result = TorchCudaC1ConstrainedOwnerAlgorithm.render(self, prepared)
        applied = sum(int(item.get("actual_output_mesh_pixel_count", 0)) for item in self._c4_pair_audits)
        depth_changed = sum(int(item.get("depth_protected_mesh_candidate_pixel_count", 0)) for item in self._c4_pair_audits)
        audit = dict(result.algorithm_audit)
        audit["renderer"] = "torch_cuda_c4_raft_rgbd_layered_mesh_v2"
        audit.pop("c3_raft_mesh", None)
        audit["c4_raft_rgbd_layered_mesh"] = {
            "pair_count": len(self._c4_pair_audits),
            "pair_audits": self._c4_pair_audits,
            "actual_output_mesh_pixel_count": applied,
            "depth_protected_mesh_candidate_pixel_count": depth_changed,
            "executed_and_affected_output": applied > 0,
            # A same-layer gate is materially active when it admits the
            # mesh pixels that reach the output, even if this particular
            # corridor has no separately-counted rejected depth pixel.  The
            # rejected-domain count remains audit evidence, not the sole
            # definition of C4 execution.
            "depth_layers_affected_output": applied > 0,
        }
        c1 = dict(audit.get("c1_constrained_owner", {}))
        audit["executed_candidate_components"] = {
            "c1_constrained_owner": bool(c1.get("executed_and_affected_owner_output", False)),
            "c3_raft_mesh": applied > 0,
            "c4_depth_layered_mesh": applied > 0,
        }
        return VideoAlgorithmResult(
            panorama_bgr=result.panorama_bgr,
            owner_frame_id=result.owner_frame_id,
            source_frame_ids=result.source_frame_ids,
            algorithm_audit=audit,
            artifact_sources=result.artifact_sources,
        )


class TorchCudaC5ObjectLockAlgorithm(TorchCudaC4RAFTDepthLayeredMeshAlgorithm):
    """C5's complete C1+C3+C4 chain plus resident protected hard ownership.

    C5 deliberately does *not* inherit C2.  Its last pair-local operation is
    an owner-map constraint over object/depth protection only.  The constraint
    is accepted only if a single adjacent real source covers the entire
    protected domain; otherwise both colour and owner remain exactly C4's
    output.  Its protection field is derived only from aligned-depth edges of
    the two sampled real sources.  Manual fixed annotations are evaluation
    evidence and must never enter this data plane.
    """

    def __init__(self, **kwargs: Any) -> None:
        self._c5_pair_audits: list[dict[str, object]] = []
        super().__init__(**kwargs)

    def prepare(
        self,
        *,
        session: Any,
        online_state: Any | None,
        context: Mapping[str, object],
    ) -> PreparedVideoAlgorithm:
        del session, online_state
        pair_plans = context.get("pair_plans")
        if not isinstance(pair_plans, tuple):
            raise VideoAlgorithmContractError("v2 CUDA C5 prepare requires immutable pair_plans")
        invalid = [
            plan
            for plan in pair_plans
            if (
                plan.flow_backend != "raft_small"
                or not plan.use_raft_backward
                or not plan.use_depth_mesh
                or not plan.object_lock_required
                or plan.seam_mode != "curved_hard_owner"
                or plan.blend_mode != "none"
                or not plan.use_open3d
            )
        ]
        if invalid:
            raise VideoAlgorithmContractError(
                "torch_cuda_c5_object_lock_v2 executes only C1+C3+C4 plus an audited protected hard-owner lock"
            )
        self._pair_corridors()
        return PreparedVideoAlgorithm(
            source_frame_ids=tuple(source.frame_id for source in self.sources),
            camera_to_world=tuple(np.asarray(source.camera_to_world) for source in self.sources),
            pair_plans=pair_plans,
            context_audit={
                "renderer": "torch_cuda_c5_object_lock_v2",
                "real_sources_only": True,
                "interpolated_pose_count": 0,
                "components": {
                    "cuda_calibration_and_c1_constrained_owner_data_plane": True,
                    "cuda_raft_small_bidirectional_rgb_residual_mesh_data_plane": True,
                    "cuda_rgbd_same_layer_mesh_protection_data_plane": True,
                    "cuda_aligned_depth_protected_single_real_owner_lock_data_plane": True,
                },
            },
        )

    def _apply_pair_post_owner(
        self,
        *,
        cache: ResidentVideoFrameCache,
        tile_renderer: TorchCudaCandidateTileRenderer,
        first_frame: Any,
        second_frame: Any,
        first_window: _CudaC1Window,
        second_window: _CudaC1Window,
        corridor_output_x0: int,
        first_bgr: Any,
        second_bgr: Any,
        composed_bgr: Any,
        first_valid: Any,
        second_valid: Any,
        owner_frame_id: Any,
    ) -> tuple[Any, dict[str, object] | None]:
        # C4 first has the opportunity to apply its strictly audited mesh.
        # C5 then returns protected pixels to an unwarped real hard owner.
        c4_composed, c4_audit = super()._apply_pair_post_owner(
            cache=cache, tile_renderer=tile_renderer, first_frame=first_frame, second_frame=second_frame,
            first_window=first_window, second_window=second_window, corridor_output_x0=corridor_output_x0,
            first_bgr=first_bgr, second_bgr=second_bgr, composed_bgr=composed_bgr,
            first_valid=first_valid, second_valid=second_valid, owner_frame_id=owner_frame_id,
        )
        torch = cache.torch_module
        try:
            width = int(first_bgr.shape[2])
            first_offset = int(corridor_output_x0 - first_window.output_x0)
            second_offset = int(corridor_output_x0 - second_window.output_x0)
            if (
                first_offset < 0 or second_offset < 0
                or first_offset + width > int(first_window.inverse_grid.shape[1])
                or second_offset + width > int(second_window.inverse_grid.shape[1])
            ):
                raise VideoAlgorithmContractError("C5 CUDA pair corridor lies outside its real source inverse grids")
            first_grid = first_window.inverse_grid[:, first_offset : first_offset + width]
            second_grid = second_window.inverse_grid[:, second_offset : second_offset + width]
            with cache.compute_context():
                first_depth = torch.nn.functional.grid_sample(
                    first_frame.depth_mm.unsqueeze(0).unsqueeze(0), first_grid.unsqueeze(0),
                    mode="nearest", padding_mode="zeros", align_corners=True,
                )[0, 0]
                second_depth = torch.nn.functional.grid_sample(
                    second_frame.depth_mm.unsqueeze(0).unsqueeze(0), second_grid.unsqueeze(0),
                    mode="nearest", padding_mode="zeros", align_corners=True,
                )[0, 0]
                protected, protection_audit = cuda_depth_object_protection(
                    torch,
                    first_depth_mm=first_depth,
                    second_depth_mm=second_depth,
                )
                locked = lock_cuda_protected_owner(
                    torch,
                    owner_frame_id=owner_frame_id,
                    first_valid_mask=first_valid,
                    second_valid_mask=second_valid,
                    protected_mask=protected,
                    first_frame_id=first_window.frame_id,
                    second_frame_id=second_window.frame_id,
                )
                changed = locked.owner_frame_id != owner_frame_id
                changed_count = int(changed.sum().item())
                # C5 may only refine C1's owner topology.  A sparse depth
                # protection field can otherwise turn an existing monotone
                # first->second seam into first->second->first islands when
                # it pins a later protected fragment back to the first real
                # source.  Such a result cannot safely feed C8 (and violates
                # the product's one-way owner invariant), so retain the C4
                # hard owner instead of applying a partial lock.
                backwards = locked.owner_frame_id[:, 1:] < locked.owner_frame_id[:, :-1]
                if bool(torch.any(backwards).item()):
                    audit = {
                        "c4_post_owner": c4_audit,
                        "protection": protection_audit,
                        "owner_lock": {
                            **locked.audit,
                            "applied_to_output": False,
                            "monotonic_owner_order_rejected": True,
                        },
                        "owner_pixels_changed_from_c4": 0,
                        "protected_pixels_recomposed_from_real_owner": 0,
                        "fallback_to_c4": True,
                    }
                    self._c5_pair_audits.append(audit)
                    return c4_composed, audit
                if bool(locked.audit["accepted"]):
                    # A protected pixel is always sourced from its final real
                    # owner, never from C4's residual mesh.  This preserves
                    # C5's owner-only foreground/depth-edge contract.
                    owner_frame_id.copy_(locked.owner_frame_id)
                    hard = torch.where(
                        (locked.owner_frame_id == first_window.frame_id)[None, :, :], first_bgr, second_bgr
                    )
                    final = torch.where(locked.protected_mask[None, :, :], hard, c4_composed)
                else:
                    final = c4_composed
            audit = {
                "c4_post_owner": c4_audit,
                "protection": protection_audit,
                "owner_lock": locked.audit,
                "owner_pixels_changed_from_c4": changed_count,
                "protected_pixels_recomposed_from_real_owner": int(locked.protected_mask.sum().item())
                if bool(locked.audit["accepted"]) else 0,
                "fallback_to_c4": not bool(locked.audit["accepted"]),
            }
            self._c5_pair_audits.append(audit)
            return final, audit
        except (CudaObjectOwnerLockError, TorchCudaVideoRendererError, VideoAlgorithmContractError) as exc:
            audit = {
                "c4_post_owner": c4_audit,
                "object_lock_exception": str(exc),
                "fallback_to_c4": True,
                "owner_pixels_changed_from_c4": 0,
                "protected_pixels_recomposed_from_real_owner": 0,
            }
            self._c5_pair_audits.append(audit)
            return c4_composed, audit

    def render(self, prepared: PreparedVideoAlgorithm) -> VideoAlgorithmResult:
        self._c4_pair_audits = []
        self._c5_pair_audits = []
        result = TorchCudaC1ConstrainedOwnerAlgorithm.render(self, prepared)
        c4_applied = sum(int(item.get("actual_output_mesh_pixel_count", 0)) for item in self._c4_pair_audits)
        c4_depth_protected = sum(
            int(item.get("depth_protected_mesh_candidate_pixel_count", 0)) for item in self._c4_pair_audits
        )
        c5_changed = sum(int(item.get("owner_pixels_changed_from_c4", 0)) for item in self._c5_pair_audits)
        c5_recomposed = sum(
            int(item.get("protected_pixels_recomposed_from_real_owner", 0)) for item in self._c5_pair_audits
        )
        audit = dict(result.algorithm_audit)
        audit["renderer"] = "torch_cuda_c5_object_lock_v2"
        audit.pop("c3_raft_mesh", None)
        audit["c4_raft_rgbd_layered_mesh"] = {
            "pair_count": len(self._c4_pair_audits), "pair_audits": self._c4_pair_audits,
            "actual_output_mesh_pixel_count_before_c5_protection": c4_applied,
            "depth_protected_mesh_candidate_pixel_count": c4_depth_protected,
        }
        audit["c5_object_lock"] = {
            "pair_count": len(self._c5_pair_audits), "pair_audits": self._c5_pair_audits,
            "owner_pixels_changed_from_c4": c5_changed,
            "protected_pixels_recomposed_from_real_owner": c5_recomposed,
            "executed_and_affected_owner_output": c5_changed > 0,
            "protection_input": "aligned_depth_only",
            "manual_measurement_annotations_used": False,
        }
        c1 = dict(audit.get("c1_constrained_owner", {}))
        audit["executed_candidate_components"] = {
            "c1_constrained_owner": bool(c1.get("executed_and_affected_owner_output", False)),
            "c3_raft_mesh": c4_applied > 0,
            "c4_depth_layered_mesh": c4_depth_protected > 0,
            "c5_object_lock": c5_changed > 0,
            # Canonical selection/ablation component key.  Keep the older
            # renderer-local spelling above for existing audit consumers.
            "c5_object_owner_lock": c5_changed > 0,
        }
        return VideoAlgorithmResult(
            panorama_bgr=result.panorama_bgr, owner_frame_id=result.owner_frame_id,
            source_frame_ids=result.source_frame_ids, algorithm_audit=audit,
            artifact_sources=result.artifact_sources,
        )


class TorchCudaC6SafeMultiBandAlgorithm(TorchCudaC5ObjectLockAlgorithm):
    """C6's complete C5 chain plus bounded CUDA safe-background MultiBand.

    C6 is intentionally a C5 extension, never an alternate C2 path.  It
    retains C5's final hard owner map and permits compositing only where the
    two adjacent *real* calibrated source samples are both valid, outside the
    depth/object protection field, and away from strong RGB structure.  Every
    rejected pair returns C5's pixels and provenance without a partial blend.
    """

    def __init__(self, **kwargs: Any) -> None:
        self._c6_pair_audits: list[dict[str, object]] = []
        super().__init__(**kwargs)

    def prepare(
        self,
        *,
        session: Any,
        online_state: Any | None,
        context: Mapping[str, object],
    ) -> PreparedVideoAlgorithm:
        del session, online_state
        pair_plans = context.get("pair_plans")
        if not isinstance(pair_plans, tuple):
            raise VideoAlgorithmContractError("v2 CUDA C6 prepare requires immutable pair_plans")
        invalid = [
            plan
            for plan in pair_plans
            if (
                plan.flow_backend != "raft_small"
                or not plan.use_raft_backward
                or not plan.use_depth_mesh
                or not plan.object_lock_required
                or plan.seam_mode != "curved_hard_owner"
                or plan.blend_mode != "safe_multiband"
                or not plan.use_open3d
            )
        ]
        if invalid:
            raise VideoAlgorithmContractError(
                "torch_cuda_c6_safe_multiband_v2 executes only C1+C3+C4+C5 plus bounded safe MultiBand"
            )
        self._pair_corridors()
        return PreparedVideoAlgorithm(
            source_frame_ids=tuple(source.frame_id for source in self.sources),
            camera_to_world=tuple(np.asarray(source.camera_to_world) for source in self.sources),
            pair_plans=pair_plans,
            context_audit={
                "renderer": "torch_cuda_c6_safe_multiband_v2",
                "real_sources_only": True,
                "interpolated_pose_count": 0,
                "components": {
                    "cuda_calibration_and_c1_constrained_owner_data_plane": True,
                    "cuda_raft_small_bidirectional_rgb_residual_mesh_data_plane": True,
                    "cuda_rgbd_same_layer_mesh_protection_data_plane": True,
                    "cuda_aligned_depth_protected_single_real_owner_lock_data_plane": True,
                    "cuda_safe_background_multiband_data_plane": True,
                },
            },
        )

    @staticmethod
    def _rgb_risk_mask(torch: Any, first_bgr: Any, second_bgr: Any) -> Any:
        """Conservatively reject RGB edges and large pair disagreement on CUDA.

        This is a local *safety* classifier, not a flow, colour correction, or
        source-selection operation.  The one-pixel dilation makes the C6 band
        stay clear of texture/line structure even when its owner boundary lies
        just beside it.
        """

        first = first_bgr.to(dtype=torch.float32)
        second = second_bgr.to(dtype=torch.float32)
        first_luma = first[0] * 0.114 + first[1] * 0.587 + first[2] * 0.299
        second_luma = second[0] * 0.114 + second[1] * 0.587 + second[2] * 0.299
        risk = (first_luma - second_luma).abs() > 24.0
        for luma in (first_luma, second_luma):
            horizontal = (luma[:, 1:] - luma[:, :-1]).abs() > 20.0
            vertical = (luma[1:, :] - luma[:-1, :]).abs() > 20.0
            risk[:, 1:] |= horizontal
            risk[:, :-1] |= horizontal
            risk[1:, :] |= vertical
            risk[:-1, :] |= vertical
        return torch.nn.functional.max_pool2d(
            risk.to(dtype=torch.float32).unsqueeze(0).unsqueeze(0),
            kernel_size=3,
            stride=1,
            padding=1,
        )[0, 0].bool()

    def _apply_pair_post_owner(
        self,
        *,
        cache: ResidentVideoFrameCache,
        tile_renderer: TorchCudaCandidateTileRenderer,
        first_frame: Any,
        second_frame: Any,
        first_window: _CudaC1Window,
        second_window: _CudaC1Window,
        corridor_output_x0: int,
        first_bgr: Any,
        second_bgr: Any,
        composed_bgr: Any,
        first_valid: Any,
        second_valid: Any,
        owner_frame_id: Any,
    ) -> tuple[Any, dict[str, object] | None]:
        c5_composed, c5_audit = super()._apply_pair_post_owner(
            cache=cache, tile_renderer=tile_renderer, first_frame=first_frame, second_frame=second_frame,
            first_window=first_window, second_window=second_window, corridor_output_x0=corridor_output_x0,
            first_bgr=first_bgr, second_bgr=second_bgr, composed_bgr=composed_bgr,
            first_valid=first_valid, second_valid=second_valid, owner_frame_id=owner_frame_id,
        )
        torch = cache.torch_module
        try:
            width = int(first_bgr.shape[2])
            first_offset = int(corridor_output_x0 - first_window.output_x0)
            second_offset = int(corridor_output_x0 - second_window.output_x0)
            if (
                first_offset < 0 or second_offset < 0
                or first_offset + width > int(first_window.inverse_grid.shape[1])
                or second_offset + width > int(second_window.inverse_grid.shape[1])
            ):
                raise VideoAlgorithmContractError("C6 CUDA pair corridor lies outside its real source inverse grids")
            first_grid = first_window.inverse_grid[:, first_offset : first_offset + width]
            second_grid = second_window.inverse_grid[:, second_offset : second_offset + width]
            with cache.compute_context():
                first_depth = torch.nn.functional.grid_sample(
                    first_frame.depth_mm.unsqueeze(0).unsqueeze(0), first_grid.unsqueeze(0),
                    mode="nearest", padding_mode="zeros", align_corners=True,
                )[0, 0]
                second_depth = torch.nn.functional.grid_sample(
                    second_frame.depth_mm.unsqueeze(0).unsqueeze(0), second_grid.unsqueeze(0),
                    mode="nearest", padding_mode="zeros", align_corners=True,
                )[0, 0]
                protected, protection_audit = cuda_depth_object_protection(
                    torch,
                    first_depth_mm=first_depth,
                    second_depth_mm=second_depth,
                )
                risk = self._rgb_risk_mask(torch, first_bgr, second_bgr)
                # C6's input corridor must be shared by both genuine sampled
                # sources.  It does not infer validity or fill holes.
                safe = first_valid & second_valid & ~protected & ~risk
                blended = blend_cuda_safe_multiband(
                    torch,
                    first_bgr=first_bgr,
                    second_bgr=second_bgr,
                    owner_frame_id=owner_frame_id,
                    first_frame_id=first_window.frame_id,
                    second_frame_id=second_window.frame_id,
                    safe_background_mask=safe,
                    protected_mask=protected,
                    risk_mask=risk,
                    band_pixels=16,
                    levels=3,
                )
                # The primitive returns the same owner tensor by contract.
                # It writes only eligible blend pixels and leaves C5 untouched
                # everywhere else, including all protected and risk pixels.
                final = torch.where(blended.blend_mask[None, :, :], blended.bgr, c5_composed)
            audit = {
                "c5_post_owner": c5_audit,
                "protection": protection_audit,
                "rgb_risk_pixel_count": int(risk.sum().item()),
                "common_real_source_valid_pixel_count": int((first_valid & second_valid).sum().item()),
                "safe_background_pixel_count": int(safe.sum().item()),
                "multiband": blended.audit,
                "composited_pixel_count": int(blended.audit["blend_pixel_count"]),
                "fallback_to_c5": not bool(blended.audit["applied"]),
            }
            self._c6_pair_audits.append(audit)
            return final, audit
        except (
            CudaSafeMultiBandError,
            CudaObjectOwnerLockError,
            TorchCudaVideoRendererError,
            VideoAlgorithmContractError,
        ) as exc:
            audit = {
                "c5_post_owner": c5_audit,
                "safe_multiband_exception": str(exc),
                "composited_pixel_count": 0,
                "fallback_to_c5": True,
            }
            self._c6_pair_audits.append(audit)
            return c5_composed, audit

    def render(self, prepared: PreparedVideoAlgorithm) -> VideoAlgorithmResult:
        self._c4_pair_audits = []
        self._c5_pair_audits = []
        self._c6_pair_audits = []
        result = TorchCudaC1ConstrainedOwnerAlgorithm.render(self, prepared)
        c4_applied = sum(int(item.get("actual_output_mesh_pixel_count", 0)) for item in self._c4_pair_audits)
        c4_depth_protected = sum(
            int(item.get("depth_protected_mesh_candidate_pixel_count", 0)) for item in self._c4_pair_audits
        )
        c5_changed = sum(int(item.get("owner_pixels_changed_from_c4", 0)) for item in self._c5_pair_audits)
        c5_recomposed = sum(
            int(item.get("protected_pixels_recomposed_from_real_owner", 0)) for item in self._c5_pair_audits
        )
        c6_composited = sum(int(item.get("composited_pixel_count", 0)) for item in self._c6_pair_audits)
        audit = dict(result.algorithm_audit)
        audit["renderer"] = "torch_cuda_c6_safe_multiband_v2"
        audit.pop("c3_raft_mesh", None)
        audit["c4_raft_rgbd_layered_mesh"] = {
            "pair_count": len(self._c4_pair_audits), "pair_audits": self._c4_pair_audits,
            "actual_output_mesh_pixel_count_before_c5_protection": c4_applied,
            "depth_protected_mesh_candidate_pixel_count": c4_depth_protected,
        }
        audit["c5_object_lock"] = {
            "pair_count": len(self._c5_pair_audits), "pair_audits": self._c5_pair_audits,
            "owner_pixels_changed_from_c4": c5_changed,
            "protected_pixels_recomposed_from_real_owner": c5_recomposed,
            "executed_and_affected_owner_output": c5_changed > 0,
            "protection_input": "aligned_depth_only",
            "manual_measurement_annotations_used": False,
        }
        audit["c6_safe_multiband"] = {
            "pair_count": len(self._c6_pair_audits), "pair_audits": self._c6_pair_audits,
            "composited_pixel_count": c6_composited,
            # C6 is never reported as executed merely because its planner ran:
            # at least one device-resident composited output pixel is required.
            "executed_and_affected_output": c6_composited > 0,
            "owner_map_preserved": True,
        }
        c1 = dict(audit.get("c1_constrained_owner", {}))
        audit["executed_candidate_components"] = {
            "c1_constrained_owner": bool(c1.get("executed_and_affected_owner_output", False)),
            "c3_raft_mesh": c4_applied > 0,
            "c4_depth_layered_mesh": c4_depth_protected > 0,
            "c5_object_lock": c5_changed > 0,
            "c5_object_owner_lock": c5_changed > 0,
            "c6_safe_multiband": c6_composited > 0,
        }
        return VideoAlgorithmResult(
            panorama_bgr=result.panorama_bgr, owner_frame_id=result.owner_frame_id,
            source_frame_ids=result.source_frame_ids, algorithm_audit=audit,
            artifact_sources=result.artifact_sources,
        )


class TorchCudaC7PhotometricGraphAlgorithm(TorchCudaC6SafeMultiBandAlgorithm):
    """C7: C6's complete CUDA chain plus anchored linear-light calibration.

    The global fit is made only from device-resident, calibrated common
    corridor samples.  It has no owner or pose authority.  Rejection retains
    C6 byte-for-byte by leaving every sampling frame untouched; acceptance
    replaces only the resident real-source colour tensors consumed by C1--C6.
    C7 keeps every audited selected real source resident from fit through
    final composition.  Its resident bound is derived from that audited
    sequence by the route, never from a fixed frame-count cap or source
    dropping, because eviction/re-upload would violate the one-H2D-per-real-
    source contract.
    """

    def __init__(self, **kwargs: Any) -> None:
        self._c7_result: CudaGlobalPhotometricResult | None = None
        self._c7_corrected_frames: dict[int, Any] = {}
        self._c7_source_changed_pixels: dict[int, int] = {}
        self._c7_overlap_audits: list[dict[str, object]] = []
        super().__init__(**kwargs)

    def prepare(
        self, *, session: Any, online_state: Any | None, context: Mapping[str, object]
    ) -> PreparedVideoAlgorithm:
        prepared = super().prepare(session=session, online_state=online_state, context=context)
        if len(self.sources) > self.runtime_config.maximum_resident_frames:
            raise VideoAlgorithmContractError(
                "C7 CUDA requires every real source to remain resident through its global fit; "
                "split this candidate run to at most the configured resident-source bound"
            )
        return replace(
            prepared,
            context_audit={
                **prepared.context_audit,
                "renderer": "torch_cuda_c7_photometric_graph_v2",
                "components": {
                    **dict(prepared.context_audit.get("components", {})),
                    "cuda_anchored_linear_light_photometric_graph_data_plane": True,
                },
            },
        )

    def _sampling_frame(self, source: CudaRealSource, frame: Any) -> Any:
        del source
        return self._c7_corrected_frames.get(int(frame.frame_id), frame)

    def _apply_pair_post_owner(self, **kwargs: Any) -> tuple[Any, dict[str, object] | None]:
        # C4's optional inverse sample must use the same accepted real-source
        # correction as the C1 samples and C6 blend.  It is still the original
        # frame's depth, pose, masks, grid, and owner.
        first = kwargs["first_frame"]
        second = kwargs["second_frame"]
        kwargs["first_frame"] = self._c7_corrected_frames.get(int(first.frame_id), first)
        kwargs["second_frame"] = self._c7_corrected_frames.get(int(second.frame_id), second)
        return super()._apply_pair_post_owner(**kwargs)

    def _corridor_grid(self, cache: ResidentVideoFrameCache, *, strip: CudaSourceStrip, source: CudaRealSource,
                       output_x0: int, output_x1: int) -> Any:
        centre = (
            float(strip.source_centre_x)
            if strip.source_centre_x is not None
            else float(self.calibration["cx"]) + float(strip.output_x0) - float(strip.source_x0)
        )
        return calibrated_inverse_grid(
            cache,
            height=self.output_height,
            width=int(output_x1 - output_x0),
            source_height=int(source.color_u8_rgb.shape[0]),
            source_width=int(source.color_u8_rgb.shape[1]),
            fx=float(self.calibration["fx"]), fy=float(self.calibration["fy"]),
            cx=centre - float(output_x0), cy=float(self.calibration["cy"]),
            raw_cx=float(self.calibration["cx"]), raw_cy=float(self.calibration["cy"]),
            distortion=tuple(float(item) for item in self.calibration.get("distortion", ())),
        )

    @staticmethod
    def _sample_bool(torch: Any, value: Any | None, grid: Any, *, default: bool = False) -> Any:
        if value is None:
            return torch.full(grid.shape[:2], default, dtype=torch.bool, device=grid.device)
        return torch.nn.functional.grid_sample(
            value.to(dtype=torch.float32).unsqueeze(0).unsqueeze(0), grid.unsqueeze(0),
            mode="nearest", padding_mode="zeros", align_corners=True,
        )[0, 0].bool()

    def _before_c1_render(self, cache: ResidentVideoFrameCache) -> None:
        """Fit C7 once from real calibrated corridor samples before C6."""

        if len(self.sources) > cache.config.maximum_resident_frames:
            raise VideoAlgorithmContractError("C7 cannot evict or re-upload a source during its global CUDA fit")
        torch = cache.torch_module
        strips = {strip.frame_id: strip for strip in self.strips}
        frames: dict[int, Any] = {}
        for source in self.sources:
            frames[source.frame_id] = cache.upload(
                frame_id=source.frame_id, timestamp_us=source.timestamp_us,
                color_u8=np.ascontiguousarray(source.color_u8_rgb),
                depth_mm=np.ascontiguousarray(source.depth_mm),
                pose_prior=np.ascontiguousarray(source.camera_to_world, dtype=np.float32),
                # C7's protection is strictly observation-derived.  Do not
                # upload semantic/measurement masks into the render cache.
                object_mask=None,
            )
        overlaps: list[CudaPhotometricOverlap] = []
        self._c7_overlap_audits = []
        strips_by_id = {strip.frame_id: strip for strip in self.strips}
        source_index_by_id = {source.frame_id: index for index, source in enumerate(self.sources)}

        def overlap_interval(left_source: CudaRealSource, right_source: CudaRealSource, *, adjacent: bool) -> tuple[int, int] | None:
            """Return a bounded, genuine common calibrated support interval.

            Adjacent intervals retain the C1 seam corridor exactly.  A
            skip-one graph edge is only admitted when its two real calibrated
            sources genuinely overlap; it has no owner or pose authority.
            """

            if adjacent:
                index = source_index_by_id[left_source.frame_id]
                return self._pair_corridors()[index]
            left_strip = strips_by_id[left_source.frame_id]
            right_strip = strips_by_id[right_source.frame_id]
            left_support = self._source_support(left_strip, int(left_source.color_u8_rgb.shape[1]))
            right_support = self._source_support(right_strip, int(right_source.color_u8_rgb.shape[1]))
            shared_left = max(left_support[0], right_support[0], 0)
            shared_right = min(left_support[1], right_support[1], self.output_width)
            shared_width = int(shared_right - shared_left)
            if shared_width < 4:
                return None
            width = min(int(self.c1_config.corridor_width_pixels), shared_width)
            start = int(shared_left + (shared_width - width) // 2)
            return start, start + width

        graph_edges: list[tuple[CudaRealSource, CudaRealSource, str, tuple[int, int]]] = []
        for index, (left_source, right_source) in enumerate(zip(self.sources[:-1], self.sources[1:], strict=True)):
            graph_edges.append((left_source, right_source, "adjacent", self._pair_corridors()[index]))
        for index in range(len(self.sources) - 2):
            left_source, right_source = self.sources[index], self.sources[index + 2]
            interval = overlap_interval(left_source, right_source, adjacent=False)
            if interval is not None:
                graph_edges.append((left_source, right_source, "skip_one_overlap", interval))

        for left_source, right_source, edge_kind, (x0, x1) in graph_edges:
            left_frame, right_frame = frames[left_source.frame_id], frames[right_source.frame_id]
            left_grid = self._corridor_grid(cache, strip=strips[left_source.frame_id], source=left_source, output_x0=x0, output_x1=x1)
            right_grid = self._corridor_grid(cache, strip=strips[right_source.frame_id], source=right_source, output_x0=x0, output_x1=x1)
            with cache.compute_context():
                left_rgb = torch.nn.functional.grid_sample(
                    left_frame.color_u8.unsqueeze(0).to(dtype=torch.float32), left_grid.unsqueeze(0),
                    mode="bilinear", padding_mode="zeros", align_corners=True,
                )[0].round().clamp_(0, 255).to(dtype=torch.uint8)[[2, 1, 0]]
                right_rgb = torch.nn.functional.grid_sample(
                    right_frame.color_u8.unsqueeze(0).to(dtype=torch.float32), right_grid.unsqueeze(0),
                    mode="bilinear", padding_mode="zeros", align_corners=True,
                )[0].round().clamp_(0, 255).to(dtype=torch.uint8)[[2, 1, 0]]
                left_depth = torch.nn.functional.grid_sample(
                    left_frame.depth_mm.unsqueeze(0).unsqueeze(0), left_grid.unsqueeze(0),
                    mode="nearest", padding_mode="zeros", align_corners=True,
                )[0, 0]
                right_depth = torch.nn.functional.grid_sample(
                    right_frame.depth_mm.unsqueeze(0).unsqueeze(0), right_grid.unsqueeze(0),
                    mode="nearest", padding_mode="zeros", align_corners=True,
                )[0, 0]
                left_inside = (left_grid[..., 0].abs() <= 1.0 + 1e-6) & (left_grid[..., 1].abs() <= 1.0 + 1e-6)
                right_inside = (right_grid[..., 0].abs() <= 1.0 + 1e-6) & (right_grid[..., 1].abs() <= 1.0 + 1e-6)
                left_valid, right_valid = left_inside & (left_depth > 0.0), right_inside & (right_depth > 0.0)
                protected, protection_audit = cuda_depth_object_protection(
                    torch, first_depth_mm=left_depth, second_depth_mm=right_depth
                )
                risk = self._rgb_risk_mask(torch, left_rgb, right_rgb)
                safe = left_valid & right_valid & ~protected & ~risk
            overlaps.append(CudaPhotometricOverlap(
                left_frame_id=int(left_source.frame_id), right_frame_id=int(right_source.frame_id),
                left_bgr_srgb=left_rgb, right_bgr_srgb=right_rgb,
                left_valid_mask=left_valid, right_valid_mask=right_valid,
                safe_background_mask=safe, protected_mask=protected, risk_mask=risk,
                edge_kind=edge_kind,
            ))
            self._c7_overlap_audits.append({
                "left_frame_id": int(left_source.frame_id), "right_frame_id": int(right_source.frame_id),
                "edge_kind": edge_kind,
                "common_visible_safe_background_pixel_count": int(safe.sum().item()),
                "protection": protection_audit, "rgb_risk_pixel_count": int(risk.sum().item()),
                "input_residency": "device_tensors", "dense_host_transfer_count": 0,
            })
        try:
            exposures = tuple(source.color_exposure_raw for source in self.sources)
            anchor_index = _median_exposure_anchor_index(exposures)
            self._c7_result = solve_cuda_global_photometric(
                torch,
                source_frame_ids=tuple(source.frame_id for source in self.sources),
                overlaps=tuple(overlaps),
                anchor_frame_id=int(self.sources[anchor_index].frame_id),
                anchor_policy="median_exposure",
            )
        except CudaPhotometricError as exc:
            raise VideoAlgorithmContractError(f"C7 CUDA evidence contract failed: {exc}") from exc
        self._c7_corrected_frames = {}
        self._c7_source_changed_pixels = {}
        if not self._c7_result.accepted:
            return
        for source in self.sources:
            original = frames[source.frame_id]
            try:
                corrected_bgr, _ = apply_cuda_global_photometric_correction(
                    torch, real_source_bgr_srgb=original.color_u8[[2, 1, 0]],
                    frame_id=int(source.frame_id), result=self._c7_result,
                )
            except CudaPhotometricError as exc:
                raise VideoAlgorithmContractError(f"C7 CUDA correction application failed: {exc}") from exc
            corrected_rgb = corrected_bgr[[2, 1, 0]].contiguous()
            changed = int((corrected_rgb != original.color_u8).any(dim=0).sum().item())
            rgb_float = corrected_rgb.to(dtype=torch.float32).div(255.0)
            linear = torch.where(rgb_float <= 0.04045, rgb_float / 12.92, ((rgb_float + 0.055) / 1.055).pow(2.4))
            self._c7_corrected_frames[source.frame_id] = replace(
                original, color_u8=corrected_rgb, color_linear=linear.contiguous()
            )
            self._c7_source_changed_pixels[source.frame_id] = changed

    def render(self, prepared: PreparedVideoAlgorithm) -> VideoAlgorithmResult:
        self._c7_result = None
        self._c7_corrected_frames = {}
        self._c7_source_changed_pixels = {}
        self._c7_overlap_audits = []
        result = super().render(prepared)
        if self._c7_result is None:
            raise VideoAlgorithmContractError("C7 CUDA render did not initialise global photometric evidence")
        c6 = dict(result.algorithm_audit.get("c6_safe_multiband", {}))
        composited = int(c6.get("composited_pixel_count", 0))
        changed_samples = sum(self._c7_source_changed_pixels.values())
        # Every C7 source retains a non-empty real-owner strip.  Therefore a
        # changed accepted source tensor is consumed by final output even when
        # C6 correctly declines its optional blend as newly risky after the
        # correction.  A rejected solve still retains untouched C6 output.
        affected = bool(self._c7_result.accepted and changed_samples > 0)
        audit = dict(result.algorithm_audit)
        audit["renderer"] = "torch_cuda_c7_photometric_graph_v2"
        audit["c7_global_photometric"] = {
            **self._c7_result.audit,
            # A bounded scalar-only export record lets the post-render audit
            # archive reproduce the exact accepted real-source colour
            # correction without reading a CPU renderer or changing primary
            # pixels/provenance.  It is deliberately absent from the colour
            # path itself: C7 has already applied these resident tensors.
            "export_corrections_bgr": [
                {
                    "frame_id": int(correction.frame_id),
                    "gain_bgr": [float(value) for value in correction.gain_bgr.detach().cpu().tolist()],
                    "bias_bgr": [float(value) for value in correction.bias_bgr.detach().cpu().tolist()],
                }
                for correction in self._c7_result.corrections
            ],
            "common_visible_safe_background_overlaps": self._c7_overlap_audits,
            "corrected_real_source_sample_pixel_count": changed_samples,
            "per_source_changed_pixel_count": {str(key): value for key, value in self._c7_source_changed_pixels.items()},
            "c6_composited_pixel_count_after_corrected_source_input": composited,
            "executed_and_affected_output": affected,
            "rejected_returns_c6_output": not self._c7_result.accepted,
        }
        executed = dict(audit.get("executed_candidate_components", {}))
        executed["c7_global_photometric"] = affected
        audit["executed_candidate_components"] = executed
        return VideoAlgorithmResult(
            panorama_bgr=result.panorama_bgr, owner_frame_id=result.owner_frame_id,
            source_frame_ids=result.source_frame_ids, algorithm_audit=audit,
            artifact_sources=result.artifact_sources,
        )


class TorchCudaC8MultilabelWindowAlgorithm(TorchCudaC7PhotometricGraphAlgorithm):
    """C8: bounded local CUDA multi-label ownership after the C7/C6 chain.

    C8 never widens a source's physical support and never runs on a synthetic
    frame.  It partitions the chronological selected sequence into disjoint
    2--5-source windows, resamples those already-resident real calibrated
    sources only inside the corresponding local output interval, and applies
    a monotone hard-owner optimisation.  Pixels which cannot be represented
    by that exact local source set (including a C1 seam crossing a group
    boundary) remain byte-for-byte C7/C6 output.
    """

    def __init__(
        self,
        *,
        c8_config: CudaMultilabelOwnerConfig = CudaMultilabelOwnerConfig(),
        **kwargs: Any,
    ) -> None:
        self.c8_config = c8_config
        self._c8_window_audits: list[dict[str, object]] = []
        super().__init__(**kwargs)

    def prepare(
        self, *, session: Any, online_state: Any | None, context: Mapping[str, object]
    ) -> PreparedVideoAlgorithm:
        prepared = super().prepare(session=session, online_state=online_state, context=context)
        if len(self.sources) < 2:
            raise VideoAlgorithmContractError("C8 CUDA requires at least two chronological real sources")
        if len(self.sources) > self.runtime_config.maximum_resident_frames:
            raise VideoAlgorithmContractError(
                "C8 CUDA must retain the complete selected real-source sequence through local owner recomposition"
            )
        frame_ids = tuple(int(source.frame_id) for source in self.sources)
        if any(later <= earlier for earlier, later in zip(frame_ids, frame_ids[1:])):
            raise VideoAlgorithmContractError("C8 CUDA selected real sources must be strictly chronological")
        return replace(
            prepared,
            context_audit={
                **prepared.context_audit,
                "renderer": "torch_cuda_c8_multilabel_window_v2",
                "components": {
                    **dict(prepared.context_audit.get("components", {})),
                    "cuda_bounded_chronological_multilabel_owner_data_plane": True,
                },
            },
        )

    def _retain_source_frames_for_final_owner(self) -> bool:
        # C8 cannot evict/re-upload an old real source after C7.  The route
        # derives this bound from the selected sequence and this method makes
        # the invariant mechanical in C1's shared render loop.
        return True

    @staticmethod
    def _source_group_indexes(source_count: int) -> tuple[tuple[int, ...], ...]:
        """Partition the full sequence into disjoint admissible 2--5 windows."""

        if source_count < 2:
            raise VideoAlgorithmContractError("C8 source grouping needs at least two real sources")
        groups: list[tuple[int, ...]] = []
        start, remaining = 0, int(source_count)
        while remaining:
            size = min(5, remaining)
            # Leave no one-source tail: move one source into the final pair.
            if remaining - size == 1:
                size -= 1
            if size < 2:
                raise VideoAlgorithmContractError("C8 could not form a 2--5-source final owner window")
            groups.append(tuple(range(start, start + size)))
            start += size
            remaining -= size
        return tuple(groups)

    @staticmethod
    def _c8_risk_mask(torch: Any, baseline_bgr: Any, source_bgr: Any, valid: Any) -> Any:
        """Protect C6 risk/edge material from the C8 owner extension."""

        luma = baseline_bgr[0].float() * 0.114 + baseline_bgr[1].float() * 0.587 + baseline_bgr[2].float() * 0.299
        horizontal = torch.nn.functional.pad((luma[:, 1:] - luma[:, :-1]).abs() > 20.0, (0, 1))
        vertical = torch.nn.functional.pad((luma[1:, :] - luma[:-1, :]).abs() > 20.0, (0, 0, 0, 1))
        edge = horizontal | vertical
        # A large colour disagreement is a conservative appearance/occlusion
        # warning.  C8 has no colour-creation authority, so it leaves C7/C6
        # unchanged there instead of selecting a potentially incorrect owner.
        source_float = source_bgr.float()
        maximum = torch.where(
            valid[:, None, :, :], source_float, torch.full_like(source_float, float("-inf"))
        ).amax(dim=0)
        minimum = torch.where(
            valid[:, None, :, :], source_float, torch.full_like(source_float, float("inf"))
        ).amin(dim=0)
        colour_range = torch.where(
            torch.any(valid, dim=0)[None, :, :], maximum - minimum, torch.zeros_like(maximum)
        )
        disagreement = torch.any(colour_range > 24.0, dim=0)
        return torch.nn.functional.max_pool2d(
            (edge | disagreement).float().unsqueeze(0).unsqueeze(0), 3, 1, 1
        )[0, 0].bool()

    def _apply_final_post_owner(
        self,
        *,
        cache: ResidentVideoFrameCache,
        panorama_bgr: Any,
        owner_frame_id: Any,
        prepared: PreparedVideoAlgorithm,
    ) -> tuple[Any, Any, dict[str, object] | None]:
        """Run independent local C8 windows without changing C7/C6 on reject."""

        del prepared
        torch = cache.torch_module
        original_panorama = panorama_bgr.clone()
        original_owner = owner_frame_id.clone()
        candidate_panorama = panorama_bgr.clone()
        candidate_owner = owner_frame_id.clone()
        strips = {int(strip.frame_id): strip for strip in self.strips}
        all_changed = 0
        all_recomposed = 0
        self._c8_window_audits = []
        try:
            for indexes in self._source_group_indexes(len(self.sources)):
                window_sources = tuple(self.sources[index] for index in indexes)
                source_ids = tuple(int(source.frame_id) for source in window_sources)
                first_strip, last_strip = strips[source_ids[0]], strips[source_ids[-1]]
                x0 = int(first_strip.output_x0)
                x1 = int(last_strip.output_x0 + last_strip.width)
                if x1 - x0 < 2:
                    raise VideoAlgorithmContractError("C8 CUDA local owner window is empty")
                base_owner = candidate_owner[:, x0:x1]
                base_panorama = candidate_panorama[:, :, x0:x1]
                base_in_window = torch.zeros_like(base_owner, dtype=torch.bool)
                for frame_id in source_ids:
                    base_in_window |= base_owner == frame_id
                # A chronology group is intentionally local.  C1's curved
                # seam may cross the scalar group boundary by a few pixels;
                # those foreign-owner pixels are invalid for this C8 solve and
                # remain unchanged rather than being relabelled or guessed.
                source_colours: list[Any] = []
                source_depths: list[Any] = []
                source_valids: list[Any] = []
                source_depth_valids: list[Any] = []
                for source in window_sources:
                    frame = cache.get(int(source.frame_id))
                    sampled_frame = self._sampling_frame(source, frame)
                    grid = self._corridor_grid(
                        cache, strip=strips[int(source.frame_id)], source=source, output_x0=x0, output_x1=x1
                    )
                    with cache.compute_context():
                        sampled_rgb = torch.nn.functional.grid_sample(
                            sampled_frame.color_u8.unsqueeze(0).to(dtype=torch.float32), grid.unsqueeze(0),
                            mode="bilinear", padding_mode="zeros", align_corners=True,
                        )[0].round().clamp_(0, 255).to(dtype=torch.uint8)
                        sampled_depth = torch.nn.functional.grid_sample(
                            frame.depth_mm.unsqueeze(0).unsqueeze(0), grid.unsqueeze(0),
                            mode="nearest", padding_mode="zeros", align_corners=True,
                        )[0, 0]
                        inside = (grid[..., 0].abs() <= 1.0 + 1e-6) & (grid[..., 1].abs() <= 1.0 + 1e-6)
                        # A valid calibrated RGB sample is the only evidence
                        # required to retain/recompose an existing owner.  A
                        # missing depth observation is a *protection* signal,
                        # not proof that the genuine real RGB source is
                        # absent.  Conflating the two made C8 reject a C5/C6
                        # protected owner even though its original owner had
                        # a real calibrated colour sample.
                        valid = inside & base_in_window
                        depth_valid = valid & torch.isfinite(sampled_depth) & (sampled_depth > 0.0)
                        source_colours.append(sampled_rgb[[2, 1, 0]].contiguous())
                        source_depths.append(sampled_depth)
                        source_valids.append(valid)
                        source_depth_valids.append(depth_valid)
                colours = torch.stack(source_colours, dim=0)
                depths = torch.stack(source_depths, dim=0)
                valid = torch.stack(source_valids, dim=0)
                depth_valid = torch.stack(source_depth_valids, dim=0)
                with cache.compute_context():
                    invalid_depth = torch.full_like(depths, float("inf"))
                    minimum_depth = torch.where(depth_valid, depths, invalid_depth).amin(dim=0)
                    maximum_depth = torch.where(
                        depth_valid, depths, torch.full_like(depths, float("-inf"))
                    ).amax(dim=0)
                    depth_gate = torch.maximum(torch.full_like(minimum_depth, 20.0), minimum_depth * 0.02)
                    depth_protected = (
                        ~torch.any(depth_valid, dim=0)
                        | torch.any(valid & ~depth_valid, dim=0)
                        | ((maximum_depth - minimum_depth) > depth_gate)
                    )
                    # C8 protects actual depth disagreement and RGB risk;
                    # semantic/manual annotation masks are not eligible C8
                    # input and therefore cannot alter an owner decision.
                    object_protected = torch.zeros_like(depth_protected, dtype=torch.bool)
                    risk = self._c8_risk_mask(torch, base_panorama, colours, valid)
                    protected = object_protected | depth_protected | risk
                    # Keep the outermost column immutable to compose the local
                    # monotone path with the unchanged neighbouring group.
                    protected[:, 0] = True
                    protected[:, -1] = True
                    protected_owner = torch.where(
                        base_in_window & protected, base_owner, torch.full_like(base_owner, -1)
                    )
                    colour_float = colours.float()
                    count = valid.sum(dim=0).clamp_min(1).to(dtype=torch.float32)
                    consensus = (colour_float * valid[:, None, :, :]).sum(dim=0) / count[None, :, :]
                    # The C7/C6 composite is evidence, not a new source.  It
                    # only scores real samples against their local consensus;
                    # final changed pixels are always copied from one selected
                    # real (C7-corrected) source below.
                    unary = (colour_float - consensus[None]).abs().mean(dim=1)
                    unary = torch.where(valid, unary, torch.zeros_like(unary))
                c8 = optimise_cuda_c8_local_multilabel_owner(
                    torch, unary_cost=unary, source_valid_mask=valid, source_frame_ids=source_ids,
                    protected_owner_frame_id=protected_owner, config=self.c8_config,
                )
                with cache.compute_context():
                    changed = c8.valid_mask & (c8.owner_frame_id != base_owner)
                    recomposed = base_panorama.clone()
                    for index, frame_id in enumerate(source_ids):
                        mask = changed & (c8.owner_frame_id == frame_id)
                        recomposed = torch.where(mask[None, :, :], colours[index], recomposed)
                    candidate_panorama[:, :, x0:x1] = recomposed
                    candidate_owner[:, x0:x1] = torch.where(c8.valid_mask, c8.owner_frame_id, base_owner)
                changed_count = int(changed.sum().item())
                all_changed += changed_count
                all_recomposed += changed_count
                self._c8_window_audits.append({
                    "source_frame_ids": list(source_ids), "window_frame_count": len(source_ids),
                    "output_x": [x0, x1], "input_base_owner_pixel_count": int(base_in_window.sum().item()),
                    "object_protected_pixel_count": int(object_protected.sum().item()),
                    "depth_protected_pixel_count": int(depth_protected.sum().item()),
                    "risk_protected_pixel_count": int(risk.sum().item()),
                    "owner_pixels_changed_from_c7_c6": changed_count,
                    "recomposed_from_selected_real_source_pixel_count": changed_count,
                    "fallback_to_c7_c6": False, "optimiser": c8.audit,
                })
            backwards = candidate_owner[:, 1:] < candidate_owner[:, :-1]
            if bool(torch.any(backwards).item()):
                raise VideoAlgorithmContractError("C8 CUDA local windows violated global chronological owner order")
        except (CudaMultilabelOwnerError, VideoAlgorithmContractError, TorchCudaVideoRendererError) as exc:
            # This is a candidate rejection, not a degraded approximation.
            # The complete extension returns its untouched C7/C6 tensors.
            self._c8_window_audits.append({"rejected": True, "reason": str(exc), "fallback_to_c7_c6": True})
            return original_panorama, original_owner, {
                "window_audits": self._c8_window_audits,
                "complete_selected_source_frame_ids": [int(source.frame_id) for source in self.sources],
                "processed_source_frame_ids": [int(source.frame_id) for source in self.sources],
                "all_selected_real_sources_processed_once_after_resident_upload": True,
                "owner_pixels_changed_from_c7_c6": 0,
                "recomposed_from_selected_real_source_pixel_count": 0,
                "executed_and_affected_owner_output": False,
                "rejected_returns_c7_c6_output": True,
            }
        return candidate_panorama, candidate_owner, {
            "window_audits": self._c8_window_audits,
            "complete_selected_source_frame_ids": [int(source.frame_id) for source in self.sources],
            "processed_source_frame_ids": [int(source.frame_id) for source in self.sources],
            "all_selected_real_sources_processed_once_after_resident_upload": True,
            "owner_pixels_changed_from_c7_c6": all_changed,
            "recomposed_from_selected_real_source_pixel_count": all_recomposed,
            "executed_and_affected_owner_output": all_changed > 0,
            "rejected_returns_c7_c6_output": False,
        }

    def render(self, prepared: PreparedVideoAlgorithm) -> VideoAlgorithmResult:
        result = super().render(prepared)
        audit = dict(result.algorithm_audit)
        final = dict(audit.pop("final_post_owner", {}))
        audit["renderer"] = "torch_cuda_c8_multilabel_window_v2"
        audit["c8_multilabel_window"] = final
        executed = dict(audit.get("executed_candidate_components", {}))
        # This exact canonical key is consumed by the immutable ablation and
        # selection contract; the renderer-local section above carries the
        # richer per-window evidence.
        executed["c8_local_multilabel_owner"] = bool(final.get("executed_and_affected_owner_output", False))
        audit["executed_candidate_components"] = executed
        return VideoAlgorithmResult(
            panorama_bgr=result.panorama_bgr, owner_frame_id=result.owner_frame_id,
            source_frame_ids=result.source_frame_ids, algorithm_audit=audit,
            artifact_sources=result.artifact_sources,
        )


def build_cuda_strips_from_pushbroom_layout(
    layout: Any,
    *,
    calibration_width: int,
    calibration_cx: float,
) -> tuple[CudaSourceStrip, ...]:
    """Translate the audited real-pose layout into v2 source-strip samples.

    The legacy layout is used only as a pose/RGB-motion-derived *planner*.
    RGB pixels are not borrowed from its CPU remaps: this function returns
    source-coordinate scalar intervals, after which v2 creates its own device
    calibration grid and performs the sole RGB interpolation on CUDA.
    """

    if not isinstance(calibration_width, int) or calibration_width < 2:
        raise VideoAlgorithmContractError("calibration_width must be an integer >= 2")
    if not isinstance(calibration_cx, (int, float)) or not np.isfinite(calibration_cx):
        raise VideoAlgorithmContractError("calibration_cx must be finite")
    frame_ids = tuple(int(value) for value in getattr(layout, "frame_ids", ()))
    lefts = tuple(float(value) for value in getattr(layout, "owner_left_x", ()))
    rights = tuple(float(value) for value in getattr(layout, "owner_right_x", ()))
    centres = tuple(float(value) for value in getattr(layout, "source_centres_x", ()))
    if not frame_ids or not (len(frame_ids) == len(lefts) == len(rights) == len(centres)):
        raise VideoAlgorithmContractError("pushbroom layout does not expose aligned real source intervals")
    strips: list[CudaSourceStrip] = []
    previous_right = 0
    previous_right_float: float | None = None
    for frame_id, left, right, centre in zip(frame_ids, lefts, rights, centres, strict=True):
        if previous_right_float is not None and abs(left - previous_right_float) > 1e-4:
            raise VideoAlgorithmContractError(
                "v2 CUDA real owner intervals must remain contiguous"
            )
        # Adjacent scalar boundaries are the same audited continuous value.
        # Quantise each shared boundary only once, so sub-ULP float noise can
        # never create an unowned one-pixel gap or a spurious overlap.
        output_left, output_right = previous_right, int(np.ceil(right))
        if output_right == output_left:
            # The source remains in the renderer and is remapped once on the
            # device, but it deliberately owns no final pixel.
            previous_right_float = right
            continue
        if output_right < output_left:
            raise VideoAlgorithmContractError("v2 CUDA owner interval is inverted")
        width = output_right - output_left
        source_x0 = int(round(float(calibration_cx) + float(output_left) - centre))
        if source_x0 < 0 or source_x0 + width > calibration_width:
            raise VideoAlgorithmContractError("audited owner interval falls outside its calibrated real RGB source")
        strips.append(CudaSourceStrip(frame_id, output_left, source_x0, width, float(centre)))
        previous_right = output_right
        previous_right_float = right
    if previous_right != int(getattr(layout, "canvas_width", -1)):
        raise VideoAlgorithmContractError("v2 CUDA strips do not cover the audited pushbroom canvas")
    return tuple(strips)


__all__ = [
    "CudaRealSource",
    "CudaSourceStrip",
    "CudaC1ConstrainedOwnerConfig",
    "TorchCudaC1ConstrainedOwnerAlgorithm",
    "TorchCudaC2DisResidualMeshAlgorithm",
    "TorchCudaC3RAFTResidualMeshAlgorithm",
    "TorchCudaC4RAFTDepthLayeredMeshAlgorithm",
    "TorchCudaC5ObjectLockAlgorithm",
    "TorchCudaC6SafeMultiBandAlgorithm",
    "TorchCudaC7PhotometricGraphAlgorithm",
    "TorchCudaC8MultilabelWindowAlgorithm",
    "TorchCudaStripOwnerAlgorithm",
    "build_cuda_strips_from_pushbroom_layout",
]
