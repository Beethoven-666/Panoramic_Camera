"""CPU-only visual seam planning for the independent RGB-D video product.

This module intentionally has no session, pose, filesystem or renderer
dependency.  It composes *already calibrated and placed* BGRA source images in
chronological order.  Therefore it cannot make a frame real, create a pose, or
invent colour: every valid output pixel is copied verbatim from exactly one
supplied source and carries that source's frame id in ``owner_frame_id``.

The pair planner is deliberately conservative.  DIS optical flow is evidence
for seam cost only; it never warps pixels.  Aligned depth identifies holes,
depth discontinuities and conflicting foreground layers.  Those locations are
protected from a normal seam and, where both depths are valid and materially
different, receive the nearer real source as an explicit hard owner.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np


_INVALID_OWNER = -1


@dataclass(frozen=True)
class VideoVisualSeamConfig:
    """Closed, lightweight limits for a pairwise curved hard-owner seam."""

    flow_enabled: bool = True
    flow_preset: int = cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST
    flow_fb_error_pixels: float = 1.5
    depth_absolute_tolerance_mm: float = 20.0
    depth_relative_tolerance: float = 0.02
    depth_edge_guard_pixels: int = 3
    maximum_protected_overlap_fraction: float = 0.45
    maximum_step_pixels: int = 4
    step_penalty: float = 7.0
    protected_penalty: float = 4096.0

    def __post_init__(self) -> None:
        if self.flow_fb_error_pixels <= 0.0:
            raise ValueError("flow_fb_error_pixels must be positive")
        if self.depth_absolute_tolerance_mm <= 0.0:
            raise ValueError("depth_absolute_tolerance_mm must be positive")
        if not 0.0 < self.depth_relative_tolerance <= 1.0:
            raise ValueError("depth_relative_tolerance must be in (0, 1]")
        if not 0 <= self.depth_edge_guard_pixels <= 16:
            raise ValueError("depth_edge_guard_pixels must be in [0, 16]")
        if not 0.0 < self.maximum_protected_overlap_fraction <= 1.0:
            raise ValueError("maximum_protected_overlap_fraction must be in (0, 1]")
        if not 1 <= self.maximum_step_pixels <= 16:
            raise ValueError("maximum_step_pixels must be in [1, 16]")
        if self.step_penalty < 0.0 or self.protected_penalty <= 0.0:
            raise ValueError("seam penalties must be non-negative and positive")


@dataclass(frozen=True)
class VideoVisualSource:
    """One true, fully placed source image and its aligned depth evidence."""

    frame_id: int
    bgra: np.ndarray
    depth_mm: np.ndarray | None = None

    def __post_init__(self) -> None:
        image = np.asarray(self.bgra)
        if image.ndim != 3 or image.shape[2] != 4 or image.dtype != np.uint8:
            raise ValueError("bgra must be a uint8 HxWx4 image")
        if image.shape[0] == 0 or image.shape[1] == 0:
            raise ValueError("bgra must not be empty")
        if self.depth_mm is not None:
            depth = np.asarray(self.depth_mm)
            if depth.ndim != 2 or depth.shape != image.shape[:2]:
                raise ValueError("depth_mm must have the BGRA image height and width")
            if not np.issubdtype(depth.dtype, np.number):
                raise ValueError("depth_mm must be numeric")
            if np.any(np.isfinite(depth) & (depth < 0.0)):
                raise ValueError("depth_mm cannot contain negative values")


@dataclass(frozen=True)
class VideoVisualSeamAudit:
    """Scalar evidence for a single incoming-source hard-owner decision."""

    incoming_frame_id: int
    overlap_pixel_count: int
    reliable_flow_fraction: float | None
    protected_pixel_count: int
    depth_evidence_accepted: bool | None
    depth_rejection_reason: str | None
    forced_nearer_owner_pixel_count: int
    curved_seam: bool
    seam_x_by_row: tuple[int, ...]
    method: str


@dataclass(frozen=True)
class VideoVisualRenderResult:
    """Strictly owner-only composed image and the auditable seam decisions."""

    bgra: np.ndarray
    owner_frame_id: np.ndarray
    valid_mask: np.ndarray
    depth_mm: np.ndarray | None
    seams: tuple[VideoVisualSeamAudit, ...]


def _valid(image: np.ndarray) -> np.ndarray:
    return image[..., 3] > 0


def _coerce_depth(depth: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray | None:
    if depth is None:
        return None
    result = np.asarray(depth, dtype=np.float32)
    if result.shape != shape:
        raise ValueError("depth image shape mismatch")
    return result


def _depth_guard(depth: np.ndarray | None, config: VideoVisualSeamConfig) -> np.ndarray:
    if depth is None:
        return np.zeros((0, 0), dtype=bool)
    valid = np.isfinite(depth) & (depth > 0.0)
    guard = ~valid
    tolerance = np.maximum(
        config.depth_absolute_tolerance_mm,
        np.abs(depth) * config.depth_relative_tolerance,
    )
    horizontal = valid[:, 1:] & valid[:, :-1] & (
        np.abs(depth[:, 1:] - depth[:, :-1])
        > np.maximum(tolerance[:, 1:], tolerance[:, :-1])
    )
    vertical = valid[1:, :] & valid[:-1, :] & (
        np.abs(depth[1:, :] - depth[:-1, :])
        > np.maximum(tolerance[1:, :], tolerance[:-1, :])
    )
    guard[:, 1:] |= horizontal
    guard[:, :-1] |= horizontal
    guard[1:, :] |= vertical
    guard[:-1, :] |= vertical
    radius = config.depth_edge_guard_pixels
    if radius:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1,) * 2)
        guard = cv2.dilate(guard.astype(np.uint8), kernel).astype(bool)
    return guard


def _flow_reliability(
    old_bgra: np.ndarray,
    new_bgra: np.ndarray,
    overlap: np.ndarray,
    config: VideoVisualSeamConfig,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Return flow-corrected residual and a forward/backward reliability mask."""

    evidence = video_flow_correspondence_evidence(old_bgra, new_bgra, overlap, config)
    if evidence is None:
        return None, None
    residual, reliable, _ = evidence
    return residual, reliable


def video_flow_correspondence_evidence(
    old_bgra: np.ndarray,
    new_bgra: np.ndarray,
    overlap: np.ndarray,
    config: VideoVisualSeamConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Return pair-only flow evidence without changing any RGB render sample.

    The tuple is ``(residual, reliable, new_at_old_coordinates)``.  The
    sampled new image is a correspondence observation only; it must never be
    used as panorama colour.  The source compositor always copies the chosen
    RGB sample verbatim from one calibrated source image.
    """

    settings = config or VideoVisualSeamConfig()
    overlap = np.asarray(overlap, dtype=bool)
    if not settings.flow_enabled or int(overlap.sum()) < 64:
        return None
    old_gray = cv2.cvtColor(old_bgra, cv2.COLOR_BGRA2GRAY)
    new_gray = cv2.cvtColor(new_bgra, cv2.COLOR_BGRA2GRAY)
    dis = cv2.DISOpticalFlow_create(settings.flow_preset)
    forward = dis.calc(old_gray, new_gray, None)
    backward = dis.calc(new_gray, old_gray, None)
    height, width = old_gray.shape
    yy, xx = np.indices((height, width), dtype=np.float32)
    map_x = xx + forward[..., 0]
    map_y = yy + forward[..., 1]
    sampled_backward = cv2.remap(
        backward,
        map_x,
        map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(np.nan, np.nan),
    )
    fb = np.linalg.norm(forward + sampled_backward, axis=2)
    reliable = overlap & np.isfinite(fb) & (fb <= settings.flow_fb_error_pixels)
    sampled_new = cv2.remap(
        new_gray,
        map_x,
        map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    residual = np.abs(old_gray.astype(np.float32) - sampled_new.astype(np.float32))
    sampled_new_bgra = cv2.remap(
        np.asarray(new_bgra),
        map_x,
        map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )
    return residual, reliable, sampled_new_bgra


def _curved_seam(energy: np.ndarray, support: np.ndarray, config: VideoVisualSeamConfig) -> np.ndarray:
    """Find a vertically continuous minimum-energy x coordinate for each row."""

    height, width = energy.shape
    columns = np.where(np.any(support, axis=0))[0]
    if columns.size == 0:
        return np.full(height, width // 2, dtype=np.int32)
    # A pushbroom pair overlaps in a narrow corridor.  Restrict the dynamic
    # program to that real common support instead of iterating across the
    # entire panorama canvas for every incoming strip.
    left, right = int(columns[0]), int(columns[-1]) + 1
    finite_energy = np.where(support[:, left:right], energy[:, left:right], np.inf).astype(np.float32)
    local_width = finite_energy.shape[1]
    # Some rows can have only disjoint alpha support.  Permit the nearest
    # global centre there, then depth/validity ownership still wins afterwards.
    default_x = int(np.median(columns)) - left
    for row in range(height):
        if not np.isfinite(finite_energy[row]).any():
            finite_energy[row, default_x] = float(config.protected_penalty)
    cost = np.empty_like(finite_energy)
    parent = np.zeros((height, local_width), dtype=np.int32)
    cost[0] = finite_energy[0]
    steps = int(config.maximum_step_pixels)
    x_values = np.arange(local_width, dtype=np.int32)
    for row in range(1, height):
        previous = cost[row - 1]
        candidates: list[np.ndarray] = []
        parents: list[np.ndarray] = []
        for delta in range(-steps, steps + 1):
            predecessor = x_values + delta
            allowed = (predecessor >= 0) & (predecessor < local_width)
            values = np.full(local_width, np.inf, dtype=np.float32)
            values[allowed] = previous[predecessor[allowed]] + (
                config.step_penalty * abs(delta)
            )
            candidates.append(values)
            parents.append(np.clip(predecessor, 0, local_width - 1))
        stacked = np.stack(candidates, axis=0)
        choice = np.argmin(stacked, axis=0)
        best = np.take_along_axis(stacked, choice[None, :], axis=0)[0]
        parent[row] = np.stack(parents, axis=0)[choice, x_values]
        cost[row] = finite_energy[row] + best
    seam = np.empty(height, dtype=np.int32)
    seam[-1] = int(np.argmin(cost[-1]))
    for row in range(height - 1, 0, -1):
        seam[row - 1] = parent[row, seam[row]]
    return seam + left


def _merge_pair(
    image: np.ndarray,
    owners: np.ndarray,
    depth: np.ndarray | None,
    incoming: VideoVisualSource,
    config: VideoVisualSeamConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, VideoVisualSeamAudit]:
    new_image = np.asarray(incoming.bgra)
    if new_image.shape != image.shape:
        raise ValueError("all video visual sources must share one placed BGRA canvas")
    new_depth = _coerce_depth(incoming.depth_mm, image.shape[:2])
    old_valid, new_valid = _valid(image), _valid(new_image)
    overlap = old_valid & new_valid
    output = image.copy()
    result_owners = owners.copy()
    output_depth = None if depth is None and new_depth is None else np.zeros(image.shape[:2], dtype=np.float32)
    if output_depth is not None and depth is not None:
        output_depth[old_valid] = depth[old_valid]
    only_new = new_valid & ~old_valid
    output[only_new] = new_image[only_new]
    result_owners[only_new] = int(incoming.frame_id)
    if output_depth is not None and new_depth is not None:
        output_depth[only_new] = new_depth[only_new]
    if not np.any(overlap):
        return output, result_owners, output_depth, VideoVisualSeamAudit(
            incoming_frame_id=int(incoming.frame_id), overlap_pixel_count=0,
            reliable_flow_fraction=None, protected_pixel_count=0,
            depth_evidence_accepted=None, depth_rejection_reason=None,
            forced_nearer_owner_pixel_count=0, curved_seam=False,
            seam_x_by_row=(), method="disjoint_hard_owner",
        )

    old_guard = _depth_guard(depth, config) if depth is not None else np.zeros_like(overlap)
    new_guard = _depth_guard(new_depth, config) if new_depth is not None else np.zeros_like(overlap)
    protected = overlap & (old_guard | new_guard)
    protected_fraction = float(protected.sum()) / float(overlap.sum())
    # Depth from a moving RGB-D camera is only local visibility evidence.  A
    # corridor in which almost every overlapping pixel is a hole or depth
    # edge is not a trustworthy layer estimate -- in practice it causes the
    # compositor to flip broad shelf/background regions to the newest frame.
    # Reject that pair's depth evidence as a whole and retain a pure real-RGB
    # owner seam.  This is deliberately fail-closed: no pixel is warped,
    # blended, synthesized, or assigned an invented provenance.
    has_pair_depth = depth is not None and new_depth is not None
    depth_evidence_accepted = (
        has_pair_depth
        and protected_fraction <= config.maximum_protected_overlap_fraction
    )
    depth_rejection_reason = (
        "protected_overlap_fraction_exceeds_limit"
        if has_pair_depth and not depth_evidence_accepted
        else None
    )
    overlap_rows, overlap_columns = np.where(overlap)
    top, bottom = int(overlap_rows.min()), int(overlap_rows.max()) + 1
    left, right = int(overlap_columns.min()), int(overlap_columns.max()) + 1
    flow_residual_crop, reliable_crop = _flow_reliability(
        image[top:bottom, left:right],
        new_image[top:bottom, left:right],
        overlap[top:bottom, left:right],
        config,
    )
    old_gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY).astype(np.float32)
    new_gray = cv2.cvtColor(new_image, cv2.COLOR_BGRA2GRAY).astype(np.float32)
    residual = np.abs(old_gray - new_gray)
    reliable = None
    if flow_residual_crop is not None:
        residual[top:bottom, left:right] = flow_residual_crop
        reliable = np.zeros_like(overlap)
        reliable[top:bottom, left:right] = reliable_crop
    gradient_old = cv2.Sobel(old_gray, cv2.CV_32F, 1, 0, ksize=3)
    gradient_new = cv2.Sobel(new_gray, cv2.CV_32F, 1, 0, ksize=3)
    energy = residual + 0.25 * np.abs(gradient_old - gradient_new)
    if reliable is not None:
        energy += np.where(reliable, 0.0, config.protected_penalty * 0.25)
    if depth_evidence_accepted:
        energy += np.where(protected, config.protected_penalty, 0.0)
    seam = _curved_seam(energy, overlap, config)
    x = np.arange(image.shape[1], dtype=np.int32)[None, :]
    choose_new = overlap & (x > seam[:, None])

    # A true depth conflict is visibility evidence, not a colour interpolation.
    forced = np.zeros_like(overlap)
    if depth_evidence_accepted:
        old_depth_valid = np.isfinite(depth) & (depth > 0.0)
        new_depth_valid = np.isfinite(new_depth) & (new_depth > 0.0)
        tolerance = np.maximum(
            config.depth_absolute_tolerance_mm,
            np.minimum(depth, new_depth) * config.depth_relative_tolerance,
        )
        conflict = overlap & old_depth_valid & new_depth_valid & (np.abs(depth - new_depth) > tolerance)
        choose_new[conflict] = new_depth[conflict] < depth[conflict]
        forced = conflict
    output[choose_new] = new_image[choose_new]
    result_owners[choose_new] = int(incoming.frame_id)
    if output_depth is not None and new_depth is not None:
        output_depth[choose_new] = new_depth[choose_new]
    reliable_fraction = None if reliable is None else float(reliable[overlap].mean())
    return output, result_owners, output_depth, VideoVisualSeamAudit(
        incoming_frame_id=int(incoming.frame_id),
        overlap_pixel_count=int(overlap.sum()),
        reliable_flow_fraction=reliable_fraction,
        protected_pixel_count=int(protected.sum()),
        depth_evidence_accepted=depth_evidence_accepted,
        depth_rejection_reason=depth_rejection_reason,
        forced_nearer_owner_pixel_count=int(forced.sum()),
        curved_seam=bool(np.ptp(seam) > 0),
        seam_x_by_row=tuple(int(value) for value in seam),
        method=(
            "dis_flow_depth_protected_curved_hard_owner"
            if config.flow_enabled
            else "rgb_depth_protected_curved_hard_owner"
        ),
    )


def render_video_visual_sources(
    sources: Iterable[VideoVisualSource], *, config: VideoVisualSeamConfig | None = None
) -> VideoVisualRenderResult:
    """Owner-compose placed real sources without warping or blending any colour."""

    settings = config or VideoVisualSeamConfig()
    iterator = iter(sources)
    try:
        first = next(iterator)
    except StopIteration as error:
        raise ValueError("at least one real video visual source is required") from error
    ids = {int(first.frame_id)}
    image = np.asarray(first.bgra).copy()
    valid = _valid(image)
    owners = np.full(image.shape[:2], _INVALID_OWNER, dtype=np.int32)
    owners[valid] = int(first.frame_id)
    depth = _coerce_depth(first.depth_mm, image.shape[:2])
    depth = None if depth is None else depth.copy()
    audits: list[VideoVisualSeamAudit] = []
    for incoming in iterator:
        if int(incoming.frame_id) in ids:
            raise ValueError("video visual source frame ids must be unique")
        ids.add(int(incoming.frame_id))
        image, owners, depth, audit = _merge_pair(image, owners, depth, incoming, settings)
        audits.append(audit)
    valid = _valid(image)
    if np.any(valid & (owners == _INVALID_OWNER)) or np.any(~valid & (owners != _INVALID_OWNER)):
        raise RuntimeError("video visual owner topology is not a strict partition")
    return VideoVisualRenderResult(
        bgra=image,
        owner_frame_id=owners,
        valid_mask=valid,
        depth_mm=depth,
        seams=tuple(audits),
    )


__all__ = [
    "VideoVisualRenderResult",
    "VideoVisualSeamAudit",
    "VideoVisualSeamConfig",
    "VideoVisualSource",
    "video_flow_correspondence_evidence",
    "render_video_visual_sources",
]
