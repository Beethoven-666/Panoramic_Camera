"""C1 constrained, real-source hard-owner utilities for video experiments.

This is deliberately a candidate-level building block, rather than a public
renderer or a replacement for :mod:`video_panorama`.  It accepts only already
placed BGRA images, keeps their pixels unmodified, and records an owner map
whose non-negative values are exclusively the supplied real source frame ids.
It contains the C1-only work: 12/5px risk-aware source selection and a
pair-local constrained hard-owner seam with first/second-order regularisation,
long-line cost, and bounded owner cleanup.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import cv2
import numpy as np

from .video_visual_renderer import VideoVisualSource


_INVALID_OWNER = -1


@dataclass(frozen=True)
class ConstrainedOwnerConfig:
    """Closed C1 controls.  They cannot create RGB, sources, or poses."""

    normal_keyframe_step_pixels: float = 12.0
    risk_keyframe_step_pixels: float = 5.0
    seam_corridor_width_pixels: int = 96
    maximum_row_step_pixels: int = 4
    first_order_penalty: float = 5.0
    second_order_penalty: float = 3.0
    long_line_minimum_pixels: int = 24
    long_line_penalty: float = 192.0
    owner_cleanup_minimum_pixels: int = 12
    owner_cleanup_maximum_passes: int = 2

    def __post_init__(self) -> None:
        if not (0.0 < self.risk_keyframe_step_pixels <= self.normal_keyframe_step_pixels):
            raise ValueError("C1 requires positive risk step <= normal keyframe step")
        if not 8 <= self.seam_corridor_width_pixels <= 256:
            raise ValueError("seam_corridor_width_pixels must be in [8, 256]")
        if not 1 <= self.maximum_row_step_pixels <= 16:
            raise ValueError("maximum_row_step_pixels must be in [1, 16]")
        if self.first_order_penalty < 0.0 or self.second_order_penalty < 0.0:
            raise ValueError("seam curvature penalties must be non-negative")
        if self.long_line_minimum_pixels < 2 or self.long_line_penalty < 0.0:
            raise ValueError("long-line controls are invalid")
        if self.owner_cleanup_minimum_pixels < 1 or not 0 <= self.owner_cleanup_maximum_passes <= 8:
            raise ValueError("owner cleanup controls are invalid")


@dataclass(frozen=True)
class PairRisk:
    """Read-only C1 pair-risk evidence used for real-frame density only."""

    risk: bool
    overlap_pixel_count: int
    mean_luma_residual: float | None
    strong_edge_fraction: float | None
    reason: str


@dataclass(frozen=True)
class RiskAwareKeyframePlan:
    """An increasing subset of actual frame IDs; no pose/frame interpolation."""

    source_indices: tuple[int, ...]
    source_frame_ids: tuple[int, ...]
    risky_edge_count: int
    normal_target_step_pixels: float
    risk_target_step_pixels: float

    def as_dict(self) -> dict[str, object]:
        return {
            "source_indices": list(self.source_indices),
            "source_frame_ids": list(self.source_frame_ids),
            "risky_edge_count": self.risky_edge_count,
            "normal_target_step_pixels": self.normal_target_step_pixels,
            "risk_target_step_pixels": self.risk_target_step_pixels,
            "real_source_frames_only": True,
            "interpolated_poses": False,
        }


@dataclass(frozen=True)
class ConstrainedOwnerAudit:
    """Pair-local evidence sufficient to audit the C1 hard-owner decision."""

    first_frame_id: int
    second_frame_id: int
    risk: PairRisk
    corridor_x: tuple[int, int] | None
    seam_x_by_row: tuple[int, ...]
    line_constraint_pixel_count: int
    cleanup_component_count_before: int
    cleanup_component_count_after: int
    cleanup_reassigned_pixel_count: int
    method: str = "c1_constrained_real_source_hard_owner"


@dataclass(frozen=True)
class ConstrainedOwnerPairResult:
    """A strict pair composition copied verbatim from one real source per pixel."""

    bgra: np.ndarray
    owner_frame_id: np.ndarray
    valid_mask: np.ndarray
    audit: ConstrainedOwnerAudit


def _valid(image: np.ndarray) -> np.ndarray:
    return np.asarray(image)[..., 3] > 0


def assess_c1_pair_risk(first: VideoVisualSource, second: VideoVisualSource) -> PairRisk:
    """Assess RGB structural risk without warping or changing ownership."""

    first_image = np.asarray(first.bgra)
    second_image = np.asarray(second.bgra)
    if first_image.shape != second_image.shape:
        raise ValueError("C1 pair images must share one placed BGRA canvas")
    overlap = _valid(first_image) & _valid(second_image)
    count = int(overlap.sum())
    if count == 0:
        return PairRisk(True, 0, None, None, "no_common_real_source_support")
    first_gray = cv2.cvtColor(first_image, cv2.COLOR_BGRA2GRAY).astype(np.float32)
    second_gray = cv2.cvtColor(second_image, cv2.COLOR_BGRA2GRAY).astype(np.float32)
    residual = np.abs(first_gray - second_gray)
    edge = np.maximum(
        np.abs(cv2.Sobel(first_gray, cv2.CV_32F, 1, 0, ksize=3)),
        np.abs(cv2.Sobel(second_gray, cv2.CV_32F, 1, 0, ksize=3)),
    )
    mean_residual = float(residual[overlap].mean())
    strong_edge_fraction = float((edge[overlap] >= 48.0).mean())
    if mean_residual >= 24.0:
        reason = "high_luma_residual"
    elif strong_edge_fraction >= 0.12:
        reason = "strong_structural_edge_fraction"
    else:
        reason = "normal_pair"
    return PairRisk(reason != "normal_pair", count, mean_residual, strong_edge_fraction, reason)


def select_c1_risk_aware_keyframes(
    frame_ids: Sequence[int],
    edge_progress_pixels: Sequence[float],
    pair_risks: Sequence[PairRisk | bool],
    *,
    config: ConstrainedOwnerConfig | None = None,
) -> RiskAwareKeyframePlan:
    """Select a chronological subset of *real* source IDs at C1's 12/5px cadence."""

    ids = tuple(int(value) for value in frame_ids)
    progress = tuple(float(value) for value in edge_progress_pixels)
    if len(ids) < 2 or len(progress) != len(ids) - 1 or len(pair_risks) != len(progress):
        raise ValueError("C1 keyframe selection requires N ids, N-1 progress values, and N-1 risks")
    if len(set(ids)) != len(ids) or ids != tuple(sorted(ids)):
        raise ValueError("C1 keyframe frame ids must be unique and chronological")
    if not np.isfinite(progress).all() or any(value < 0.0 for value in progress):
        raise ValueError("C1 edge progress must be finite and non-negative")
    settings = config or ConstrainedOwnerConfig()
    selected = [0]
    accumulated = 0.0
    risky_count = 0
    risk_since_last_source = False
    for edge_index, value in enumerate(progress):
        risk = pair_risks[edge_index]
        risky = bool(risk.risk) if isinstance(risk, PairRisk) else bool(risk)
        risky_count += int(risky)
        risk_since_last_source |= risky
        accumulated += value
        target = (
            settings.risk_keyframe_step_pixels
            if risk_since_last_source
            else settings.normal_keyframe_step_pixels
        )
        if accumulated >= target:
            selected.append(edge_index + 1)
            accumulated = 0.0
            risk_since_last_source = False
    if selected[-1] != len(ids) - 1:
        selected.append(len(ids) - 1)
    indices = tuple(selected)
    return RiskAwareKeyframePlan(
        source_indices=indices,
        source_frame_ids=tuple(ids[index] for index in indices),
        risky_edge_count=risky_count,
        normal_target_step_pixels=settings.normal_keyframe_step_pixels,
        risk_target_step_pixels=settings.risk_keyframe_step_pixels,
    )


def _corridor(overlap: np.ndarray, width: int) -> tuple[np.ndarray, tuple[int, int] | None]:
    columns = np.where(np.any(overlap, axis=0))[0]
    if columns.size == 0:
        return np.zeros_like(overlap, dtype=bool), None
    left, right = int(columns[0]), int(columns[-1]) + 1
    actual_width = min(int(width), right - left)
    centre = (left + right) // 2
    x0 = max(left, centre - actual_width // 2)
    x1 = min(right, x0 + actual_width)
    x0 = max(left, x1 - actual_width)
    x = np.arange(overlap.shape[1])[None, :]
    return overlap & (x >= x0) & (x < x1), (x0, x1)


def _long_line_mask(first_gray: np.ndarray, second_gray: np.ndarray, support: np.ndarray, settings: ConstrainedOwnerConfig) -> np.ndarray:
    """Return an image-local cost mask for long, straight structural evidence."""

    edge_source = np.maximum(first_gray, second_gray).astype(np.uint8)
    edges = cv2.Canny(edge_source, 48, 128)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180.0, threshold=18, minLineLength=settings.long_line_minimum_pixels, maxLineGap=3)
    mask = np.zeros_like(support, dtype=np.uint8)
    if lines is None:
        return mask.astype(bool)
    for x0, y0, x1, y1 in lines.reshape(-1, 4):
        if float(np.hypot(x1 - x0, y1 - y0)) >= settings.long_line_minimum_pixels:
            cv2.line(mask, (int(x0), int(y0)), (int(x1), int(y1)), 255, 1)
    return (mask > 0) & support


def _constrained_seam(energy: np.ndarray, support: np.ndarray, bounds: tuple[int, int], settings: ConstrainedOwnerConfig) -> np.ndarray:
    """Dynamic programme with first *and* second order row curvature penalties."""

    height, _ = energy.shape
    x0, x1 = bounds
    local_energy = np.where(support[:, x0:x1], energy[:, x0:x1], np.inf).astype(np.float32)
    width = local_energy.shape[1]
    centre = width // 2
    # Rows without common support do not affect a pixel decision; making their
    # centre finite keeps the seam continuous across an alpha gap.
    for row in range(height):
        if not np.isfinite(local_energy[row]).any():
            local_energy[row, centre] = 0.0
    maximum_step = settings.maximum_row_step_pixels
    deltas = np.arange(-maximum_step, maximum_step + 1, dtype=np.int32)
    velocity_count = len(deltas)
    zero_velocity = maximum_step
    previous = np.full((width, velocity_count), np.inf, dtype=np.float32)
    previous[:, zero_velocity] = local_energy[0]
    parent_x = np.full((height, width, velocity_count), -1, dtype=np.int16)
    parent_v = np.full((height, width, velocity_count), -1, dtype=np.int8)
    xs = np.arange(width, dtype=np.int32)
    for row in range(1, height):
        current = np.full_like(previous, np.inf)
        for velocity_index, velocity in enumerate(deltas):
            predecessor_x = xs - velocity
            allowed = (predecessor_x >= 0) & (predecessor_x < width)
            allowed_x = xs[allowed]
            if not allowed_x.size:
                continue
            predecessor = predecessor_x[allowed]
            transition = previous[predecessor] + (
                settings.first_order_penalty * abs(int(velocity))
                + settings.second_order_penalty * np.abs(deltas - velocity)[None, :]
            )
            predecessor_velocity = np.argmin(transition, axis=1).astype(np.int8)
            current[allowed_x, velocity_index] = (
                local_energy[row, allowed_x]
                + transition[np.arange(allowed_x.size), predecessor_velocity]
            )
            parent_x[row, allowed_x, velocity_index] = predecessor.astype(np.int16)
            parent_v[row, allowed_x, velocity_index] = predecessor_velocity
        previous = current
    endpoint = int(np.argmin(previous))
    x, velocity = np.unravel_index(endpoint, previous.shape)
    seam = np.empty(height, dtype=np.int32)
    seam[-1] = int(x)
    for row in range(height - 1, 0, -1):
        predecessor_x = int(parent_x[row, x, velocity])
        predecessor_velocity = int(parent_v[row, x, velocity])
        if predecessor_x < 0 or predecessor_velocity < 0:
            predecessor_x, predecessor_velocity = centre, zero_velocity
        seam[row - 1] = predecessor_x
        x, velocity = predecessor_x, predecessor_velocity
    return seam + x0


def _cleanup_components(
    owners: np.ndarray,
    corridor: np.ndarray,
    first_id: int,
    second_id: int,
    first_valid: np.ndarray,
    second_valid: np.ndarray,
    fixed_owners: np.ndarray,
    settings: ConstrainedOwnerConfig,
) -> tuple[np.ndarray, int, int, int]:
    """Remove only tiny, non-fixed pair-owner islands inside the C1 corridor."""

    result = owners.copy()
    before = after = reassigned = 0
    for cleanup_pass in range(settings.owner_cleanup_maximum_passes):
        changed = False
        components_before_this_pass = 0
        for owner, alternate, alternate_valid in (
            (first_id, second_id, second_valid),
            (second_id, first_id, first_valid),
        ):
            labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(
                ((result == owner) & corridor).astype(np.uint8), connectivity=8
            )
            for label in range(1, labels_count):
                area = int(stats[label, cv2.CC_STAT_AREA])
                if area >= settings.owner_cleanup_minimum_pixels:
                    continue
                component = labels == label
                if np.any(fixed_owners[component] == owner) or not np.all(alternate_valid[component]):
                    continue
                components_before_this_pass += 1
                result[component] = alternate
                reassigned += area
                changed = True
        if cleanup_pass == 0:
            before = components_before_this_pass
        if not changed:
            break
    for owner in (first_id, second_id):
        labels_count, _, stats, _ = cv2.connectedComponentsWithStats(
            ((result == owner) & corridor).astype(np.uint8), connectivity=8
        )
        after += sum(
            int(stats[label, cv2.CC_STAT_AREA]) < settings.owner_cleanup_minimum_pixels
            for label in range(1, labels_count)
        )
    return result, before, after, reassigned


def render_c1_constrained_hard_owner_pair(
    first: VideoVisualSource,
    second: VideoVisualSource,
    *,
    config: ConstrainedOwnerConfig | None = None,
    fixed_owner_frame_id: np.ndarray | None = None,
) -> ConstrainedOwnerPairResult:
    """Compose two placed real sources with a pair-local C1 hard-owner seam.

    ``fixed_owner_frame_id`` is optional pair-local protection evidence.  Its
    only legal values are ``-1``, ``first.frame_id`` and ``second.frame_id``;
    fixed pixels cannot be cleaned or assigned to an unavailable source.
    """

    settings = config or ConstrainedOwnerConfig()
    if int(first.frame_id) == int(second.frame_id):
        raise ValueError("C1 constrained owner pair requires distinct real source ids")
    first_image = np.asarray(first.bgra)
    second_image = np.asarray(second.bgra)
    if first_image.shape != second_image.shape:
        raise ValueError("C1 pair images must share one placed BGRA canvas")
    first_valid, second_valid = _valid(first_image), _valid(second_image)
    valid = first_valid | second_valid
    overlap = first_valid & second_valid
    risk = assess_c1_pair_risk(first, second)
    corridor, bounds = _corridor(overlap, settings.seam_corridor_width_pixels)
    owners = np.full(valid.shape, _INVALID_OWNER, dtype=np.int32)
    owners[first_valid] = int(first.frame_id)
    owners[second_valid & ~first_valid] = int(second.frame_id)
    fixed = np.full(valid.shape, _INVALID_OWNER, dtype=np.int32)
    if fixed_owner_frame_id is not None:
        fixed = np.asarray(fixed_owner_frame_id, dtype=np.int32)
        if fixed.shape != valid.shape:
            raise ValueError("fixed_owner_frame_id must match the placed source canvas")
        legal = (fixed == _INVALID_OWNER) | (fixed == int(first.frame_id)) | (fixed == int(second.frame_id))
        if not np.all(legal):
            raise ValueError("fixed_owner_frame_id may only name this pair's real source ids")
        if np.any((fixed == int(first.frame_id)) & ~first_valid) or np.any((fixed == int(second.frame_id)) & ~second_valid):
            raise ValueError("fixed owner cannot select a source without a real pixel")
    if bounds is None:
        output = np.zeros_like(first_image)
        output[first_valid] = first_image[first_valid]
        output[second_valid & ~first_valid] = second_image[second_valid & ~first_valid]
        return ConstrainedOwnerPairResult(
            output, owners, valid,
            ConstrainedOwnerAudit(int(first.frame_id), int(second.frame_id), risk, None, (), 0, 0, 0, 0),
        )

    first_gray = cv2.cvtColor(first_image, cv2.COLOR_BGRA2GRAY).astype(np.float32)
    second_gray = cv2.cvtColor(second_image, cv2.COLOR_BGRA2GRAY).astype(np.float32)
    residual = np.abs(first_gray - second_gray)
    gradient = np.abs(cv2.Sobel(first_gray, cv2.CV_32F, 1, 0, ksize=3) - cv2.Sobel(second_gray, cv2.CV_32F, 1, 0, ksize=3))
    line_mask = _long_line_mask(first_gray, second_gray, corridor, settings)
    energy = residual + 0.25 * gradient + np.where(line_mask, settings.long_line_penalty, 0.0)
    seam = _constrained_seam(energy, corridor, bounds, settings)
    x = np.arange(valid.shape[1], dtype=np.int32)[None, :]
    # The non-corridor overlap is intentionally a deterministic old/new hard
    # split.  The only flexible boundary lies in the bounded adjacent corridor.
    owners[overlap & (x >= bounds[1])] = int(second.frame_id)
    owners[corridor & (x > seam[:, None])] = int(second.frame_id)
    owners[fixed == int(first.frame_id)] = int(first.frame_id)
    owners[fixed == int(second.frame_id)] = int(second.frame_id)
    owners, before, after, reassigned = _cleanup_components(
        owners, corridor, int(first.frame_id), int(second.frame_id), first_valid, second_valid, fixed, settings
    )
    output = np.zeros_like(first_image)
    output[owners == int(first.frame_id)] = first_image[owners == int(first.frame_id)]
    output[owners == int(second.frame_id)] = second_image[owners == int(second.frame_id)]
    if np.any(valid & (owners == _INVALID_OWNER)) or np.any(~valid & (owners != _INVALID_OWNER)):
        raise RuntimeError("C1 constrained owner topology is not a strict partition")
    if np.any((owners == int(first.frame_id)) & ~first_valid) or np.any((owners == int(second.frame_id)) & ~second_valid):
        raise RuntimeError("C1 owner selected an unavailable source pixel")
    return ConstrainedOwnerPairResult(
        output,
        owners,
        valid,
        ConstrainedOwnerAudit(
            int(first.frame_id), int(second.frame_id), risk, bounds,
            tuple(int(value) for value in seam), int(line_mask.sum()), before, after, reassigned,
        ),
    )


def assert_c1_real_source_owners(
    result: ConstrainedOwnerPairResult,
    sources: Iterable[VideoVisualSource],
) -> None:
    """Fail closed unless every valid output pixel is verbatim from its owner."""

    by_id = {int(source.frame_id): np.asarray(source.bgra) for source in sources}
    valid = np.asarray(result.valid_mask, dtype=bool)
    owners = np.asarray(result.owner_frame_id, dtype=np.int32)
    if np.any(valid & ~np.isin(owners, tuple(by_id))) or np.any(~valid & (owners != _INVALID_OWNER)):
        raise RuntimeError("C1 output contains a non-real or invalid owner")
    for frame_id, image in by_id.items():
        chosen = owners == frame_id
        if chosen.shape != image.shape[:2] or not np.array_equal(result.bgra[chosen], image[chosen]):
            raise RuntimeError("C1 output colour is not a verbatim real-source sample")


__all__ = [
    "ConstrainedOwnerAudit",
    "ConstrainedOwnerConfig",
    "ConstrainedOwnerPairResult",
    "PairRisk",
    "RiskAwareKeyframePlan",
    "assert_c1_real_source_owners",
    "assess_c1_pair_risk",
    "render_c1_constrained_hard_owner_pair",
    "select_c1_risk_aware_keyframes",
]
