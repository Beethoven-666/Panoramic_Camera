"""Candidate-only D3 real-source object compositor.

D3 is deliberately a small data-plane primitive: an automatically discovered
depth-connected foreground component is RAFT-gated, guarded, and copied from
one *real* source tile.  It has no annotation argument, no object warp, and
no blend path.  A caller must reject rather than approximate when neither
source fully supports the guarded component.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


class D3ObjectFirstCompositorError(ValueError):
    """D3 could not establish a safe single-real-source object owner."""


@dataclass(frozen=True)
class D3ObjectFirstCompositorResult:
    panorama_bgr: np.ndarray
    owner_frame_id: np.ndarray
    accepted: bool
    audit: dict[str, object]


def _validate_tile(name: str, value: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    array = np.asarray(value)
    if array.shape[:2] != shape:
        raise D3ObjectFirstCompositorError(f"D3 {name} shape does not match the owner canvas")
    return array


def compose_d3_object_first_dense_source(
    *, panorama_bgr: np.ndarray, owner_frame_id: np.ndarray,
    first_bgr: np.ndarray, second_bgr: np.ndarray,
    first_depth_mm: np.ndarray, second_depth_mm: np.ndarray,
    first_valid: np.ndarray, second_valid: np.ndarray,
    raft_forward_xy: np.ndarray, first_frame_id: int, second_frame_id: int,
    protection_margin_pixels: int = 10, source_support_gate: float = 0.98,
) -> D3ObjectFirstCompositorResult:
    """Apply one guarded, unblended D3 real-source object decision.

    The input tiles are already calibrated inverse-remap samples in the final
    canvas.  Consequently changing a pixel copies exactly one real source
    sample and changes its provenance owner in the same operation.
    """

    canvas = np.asarray(panorama_bgr)
    owner = np.asarray(owner_frame_id)
    if canvas.ndim != 3 or canvas.shape[2] != 3 or canvas.dtype != np.uint8:
        raise D3ObjectFirstCompositorError("D3 needs a uint8 BGR output canvas")
    if owner.shape != canvas.shape[:2] or owner.dtype.kind not in "iu":
        raise D3ObjectFirstCompositorError("D3 needs an integer owner canvas")
    if not 8 <= protection_margin_pixels <= 12:
        raise D3ObjectFirstCompositorError("D3 protection margin must be in [8, 12]")
    if not 0.98 <= source_support_gate <= 1.0:
        raise D3ObjectFirstCompositorError("D3 source support gate must be in [0.98, 1]")
    shape = owner.shape
    first = _validate_tile("first BGR", first_bgr, shape)
    second = _validate_tile("second BGR", second_bgr, shape)
    first_depth = _validate_tile("first depth", first_depth_mm, shape).astype(np.float32, copy=False)
    second_depth = _validate_tile("second depth", second_depth_mm, shape).astype(np.float32, copy=False)
    first_ok = _validate_tile("first validity", first_valid, shape).astype(bool, copy=False)
    second_ok = _validate_tile("second validity", second_valid, shape).astype(bool, copy=False)
    flow = np.asarray(raft_forward_xy, dtype=np.float32)
    if flow.shape != (*shape, 2) or not np.isfinite(flow).all():
        raise D3ObjectFirstCompositorError("D3 needs a finite RAFT forward field for the real source tiles")
    valid_depth = first_ok & second_ok & np.isfinite(first_depth) & np.isfinite(second_depth) & (first_depth > 0.0) & (second_depth > 0.0)
    # Segment *surfaces* rather than just a thin inter-frame disagreement
    # contour.  Depth discontinuities are barriers; their connected interiors
    # give D3 a whole physical object candidate (carton/cable/fan) whenever it
    # is jointly visible in the two real sources.
    reference_depth = np.minimum(first_depth, second_depth)
    tolerance = np.maximum(20.0, reference_depth * 0.02)
    horizontal = valid_depth[:, 1:] & valid_depth[:, :-1] & (
        np.abs(reference_depth[:, 1:] - reference_depth[:, :-1])
        > np.maximum(tolerance[:, 1:], tolerance[:, :-1])
    )
    vertical = valid_depth[1:] & valid_depth[:-1] & (
        np.abs(reference_depth[1:] - reference_depth[:-1])
        > np.maximum(tolerance[1:] , tolerance[:-1])
    )
    barriers = np.zeros(shape, dtype=bool)
    barriers[:, 1:] |= horizontal; barriers[:, :-1] |= horizontal
    barriers[1:] |= vertical; barriers[:-1] |= vertical
    barriers = cv2.dilate(barriers.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    foreground = valid_depth & ~barriers
    count, labels, stats, _ = cv2.connectedComponentsWithStats(foreground.astype(np.uint8), connectivity=8)
    if count <= 1:
        return D3ObjectFirstCompositorResult(canvas.copy(), owner.copy(), False, {"reason": "no_depth_connected_foreground_component", "annotations_renderer_input": False})
    # A central, sharp real component is less likely to be a strip boundary.
    gray = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
    sharp = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0)) + np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1))
    centre_x = (shape[1] - 1) / 2.0
    scores = []
    for label in range(1, count):
        mask = labels == label
        pixels = int(mask.sum())
        if pixels < 32:
            scores.append(float("-inf")); continue
        x = float(np.mean(np.nonzero(mask)[1]))
        # Prefer a coherent near surface.  This is entirely automatic: depth,
        # centre and sharpness are all measurements from the real source tile.
        near = 1.0 / max(1.0, float(np.median(reference_depth[mask])))
        scores.append(float(sharp[mask].mean()) + 600.0 * near + 1.0 - abs(x - centre_x) / max(1.0, centre_x) + min(1.0, pixels / 256.0))
    label = 1 + int(np.argmax(scores))
    component = labels == label
    guarded = cv2.dilate(component.astype(np.uint8), np.ones((2 * protection_margin_pixels + 1,) * 2, np.uint8)).astype(bool)
    magnitude = np.linalg.norm(flow, axis=2)
    guarded &= magnitude <= 32.0
    if not np.any(guarded):
        return D3ObjectFirstCompositorResult(canvas.copy(), owner.copy(), False, {"reason": "raft_track_has_no_safe_component", "annotations_renderer_input": False})
    support_first = float(np.mean(first_ok[guarded]))
    support_second = float(np.mean(second_ok[guarded]))
    if max(support_first, support_second) < source_support_gate:
        return D3ObjectFirstCompositorResult(canvas.copy(), owner.copy(), False, {"reason": "no_single_real_source_full_support", "source_support": [support_first, support_second], "annotations_renderer_input": False})
    sharpness = [float(sharp[guarded].mean()), float((np.abs(cv2.Sobel(cv2.cvtColor(second, cv2.COLOR_BGR2GRAY), cv2.CV_32F, 1, 0)))[guarded].mean())]
    scores = [support_first + sharpness[0] / 255.0, support_second + sharpness[1] / 255.0]
    selected_index = 0 if scores[0] >= scores[1] else 1
    selected_id = int((first_frame_id, second_frame_id)[selected_index])
    selected_bgr = (first, second)[selected_index]
    result_image, result_owner = canvas.copy(), owner.copy()
    result_image[guarded] = selected_bgr[guarded]
    result_owner[guarded] = selected_id
    adjacent_valid = (result_owner[:, 1:] >= 0) & (result_owner[:, :-1] >= 0)
    if np.any(adjacent_valid & (result_owner[:, 1:] < result_owner[:, :-1])):
        return D3ObjectFirstCompositorResult(canvas.copy(), owner.copy(), False, {"reason": "owner_temporal_monotonicity_rejected", "annotations_renderer_input": False})
    return D3ObjectFirstCompositorResult(result_image, result_owner, True, {
        "renderer_component": "d3_object_first_dense_source_compositor", "annotations_renderer_input": False,
        "component_method": "depth_connected_component", "raft_track_used": True,
        "object_flow_or_warp": False, "object_multiband": False, "maximum_object_handoffs": 1,
        "protection_margin_pixels": protection_margin_pixels, "component_pixel_count": int(component.sum()),
        "guarded_pixel_count": int(guarded.sum()), "selected_owner_frame_id": selected_id,
        "source_support": [support_first, support_second], "actual_output_object_pixel_count": int(guarded.sum()),
    })


def compose_d3_persistent_object_tracks(
    *, panorama_bgr: np.ndarray, owner_frame_id: np.ndarray,
    source_bgr: tuple[np.ndarray, ...], source_depth_mm: tuple[np.ndarray, ...],
    source_valid: tuple[np.ndarray, ...], raft_forward_xy: tuple[np.ndarray, ...],
    source_frame_ids: tuple[int, ...], protection_margin_pixels: int = 10,
    source_support_gate: float = 0.98,
) -> D3ObjectFirstCompositorResult:
    """Lock automatically discovered object surfaces across the dense chain.

    Unlike the pair primitive, this forms label-free tracks in the common
    calibrated canvas and selects a maximum of two ordered *real* owners for
    each tracked surface.  A single x handoff is the only permitted fallback
    when no one source sees the complete guarded object.  Annotations are not
    accepted by this API and cannot influence the data plane.
    """
    canvas, owner = np.asarray(panorama_bgr), np.asarray(owner_frame_id)
    if len(source_bgr) < 2 or not (len(source_bgr) == len(source_depth_mm) == len(source_valid) == len(source_frame_ids)):
        raise D3ObjectFirstCompositorError("D3 persistent tracks require aligned dense real sources")
    shape = owner.shape
    if canvas.shape[:2] != shape:
        raise D3ObjectFirstCompositorError("D3 persistent track canvas shape mismatch")
    if not 8 <= protection_margin_pixels <= 12:
        raise D3ObjectFirstCompositorError("D3 protection margin must be in [8, 12]")
    colours = tuple(_validate_tile("source BGR", value, shape) for value in source_bgr)
    depths = tuple(_validate_tile("source depth", value, shape).astype(np.float32, copy=False) for value in source_depth_mm)
    valids = tuple(_validate_tile("source validity", value, shape).astype(bool, copy=False) for value in source_valid)
    if len(raft_forward_xy) != len(colours) - 1:
        raise D3ObjectFirstCompositorError("D3 persistent tracks require one RAFT edge per dense source pair")
    raft_safe = np.zeros(shape, dtype=bool)
    for flow in raft_forward_xy:
        flow_array = np.asarray(flow, dtype=np.float32)
        if flow_array.shape != (*shape, 2) or not np.isfinite(flow_array).all():
            raise D3ObjectFirstCompositorError("D3 persistent tracks require finite real-source RAFT fields")
        raft_safe |= np.linalg.norm(flow_array, axis=2) <= 32.0
    depth_stack = np.stack(depths, axis=0)
    valid_stack = np.stack(tuple(valid & np.isfinite(depth) & (depth > 0.0) for valid, depth in zip(valids, depths, strict=True)), axis=0)
    # Median real depth gives a stable common-canvas surface observation.  A
    # discontinuity in any visible source is a boundary, never a color-only
    # segmentation signal.
    observed = np.any(valid_stack, axis=0)
    filled = np.where(valid_stack, depth_stack, np.nan)
    median = np.nanmedian(filled, axis=0).astype(np.float32)
    finite_median = np.where(np.isfinite(median), median, 0.0)
    local_min = cv2.erode(finite_median, np.ones((3, 3), np.uint8))
    local_max = cv2.dilate(finite_median, np.ones((3, 3), np.uint8))
    boundaries = observed & ((local_max - local_min) > np.maximum(20.0, finite_median * 0.02))
    boundaries = cv2.dilate(boundaries.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    count, labels, stats, _ = cv2.connectedComponentsWithStats((observed & ~boundaries).astype(np.uint8), connectivity=8)
    result_image, result_owner = canvas.copy(), owner.copy()
    sharpness: list[np.ndarray] = []
    for colour in colours:
        gray = cv2.cvtColor(colour, cv2.COLOR_BGR2GRAY)
        sharpness.append(np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0)) + np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1)))
    applied_tracks: list[dict[str, object]] = []
    # Components are intentionally evaluated without source annotations.  The
    # broad scene planes are rejected by their geometry, while compact depth
    # surfaces retain the carton, cable and fan as independent tracks.
    for label in range(1, count):
        x, y, width, height, pixels = (int(value) for value in stats[label])
        if pixels < 48 or width < 4 or height < 4:
            continue
        if width >= int(shape[1] * 0.80) and height <= int(shape[0] * 0.38):
            continue
        if width >= int(shape[1] * 0.94) and height >= int(shape[0] * 0.84):
            continue
        component = labels == label
        guarded = cv2.dilate(component.astype(np.uint8), np.ones((2 * protection_margin_pixels + 1,) * 2, np.uint8)).astype(bool)
        guarded &= raft_safe
        if int(guarded.sum()) < 48:
            continue
        # The pre-existing hard-owner map is monotonic.  A persistent object
        # decision may compress its many strip owners, but cannot select an
        # owner outside the object's observed source interval; doing so would
        # create an invalid temporal reversal at the guard boundary.
        existing_ids = result_owner[guarded & (result_owner >= 0)]
        if existing_ids.size == 0:
            continue
        existing_min, existing_max = int(existing_ids.min()), int(existing_ids.max())
        ys, xs = np.nonzero(guarded)
        x0, x1 = int(xs.min()), int(xs.max())
        # Visibility + center + sharpness + valid depth are the owner score;
        # invalid pixels are never copied.  Keep only the strongest twelve
        # real sources to make the bounded two-owner search deterministic.
        candidates: list[tuple[float, int]] = []
        centre = (shape[1] - 1) / 2.0
        for index, valid in enumerate(valid_stack):
            if not existing_min <= int(source_frame_ids[index]) <= existing_max:
                continue
            support = float(np.mean(valid[guarded]))
            if support <= 0.01:
                continue
            source_x = float(np.mean(np.nonzero(valid & guarded)[1])) if np.any(valid & guarded) else centre
            score = support + float(sharpness[index][guarded & valid].mean()) / 255.0 + 0.10 * (1.0 - abs(source_x - centre) / max(1.0, centre))
            candidates.append((score, index))
        candidates = sorted(candidates, reverse=True)[:12]
        if not candidates:
            continue
        best: tuple[float, int, int, int] | None = None
        # A one-handoff ordered split: left source <= right source in temporal
        # layout, so a successful replacement remains globally monotonic.
        for _, left in candidates:
            for _, right in candidates:
                if source_frame_ids[left] > source_frame_ids[right]:
                    continue
                for boundary in range(x0 - 1, x1 + 1, 4):
                    columns = np.arange(shape[1])[None, :]
                    left_mask = guarded & (columns <= boundary)
                    right_mask = guarded & ~left_mask
                    supported = (left_mask & valid_stack[left]) | (right_mask & valid_stack[right])
                    support = float(np.mean(supported[guarded]))
                    if support < source_support_gate:
                        continue
                    # Prefer no handoff, then a central/sharp source pair.
                    score = support + float(sharpness[left][left_mask & valid_stack[left]].mean() if np.any(left_mask & valid_stack[left]) else 0.0) / 255.0
                    score += float(sharpness[right][right_mask & valid_stack[right]].mean() if np.any(right_mask & valid_stack[right]) else 0.0) / 255.0
                    score -= 0.01 if left != right else 0.0
                    if best is None or score > best[0]:
                        best = (score, left, right, boundary)
        if best is None:
            continue
        _, left, right, boundary = best
        columns = np.arange(shape[1])[None, :]
        left_mask = guarded & (columns <= boundary) & valid_stack[left]
        right_mask = guarded & ~left_mask & valid_stack[right]
        proposal = result_owner.copy()
        proposal[left_mask] = int(source_frame_ids[left])
        proposal[right_mask] = int(source_frame_ids[right])
        adjacent_valid = (proposal[:, 1:] >= 0) & (proposal[:, :-1] >= 0)
        if np.any(adjacent_valid & (proposal[:, 1:] < proposal[:, :-1])):
            continue
        result_image[left_mask] = colours[left][left_mask]
        result_image[right_mask] = colours[right][right_mask]
        result_owner = proposal
        applied_tracks.append({
            "component_pixel_count": int(component.sum()), "guarded_pixel_count": int(guarded.sum()),
            "left_owner_frame_id": int(source_frame_ids[left]), "right_owner_frame_id": int(source_frame_ids[right]),
            "handoff_count": int(left != right), "boundary_x": int(boundary),
        })
    if not applied_tracks:
        return D3ObjectFirstCompositorResult(canvas.copy(), owner.copy(), False, {"reason": "no_persistent_depth_track_with_real_source_support", "annotations_renderer_input": False})
    changed = result_owner != owner
    return D3ObjectFirstCompositorResult(result_image, result_owner, True, {
        "renderer_component": "d3_object_first_dense_source_compositor", "annotations_renderer_input": False,
        "component_method": "persistent_depth_connected_component_track", "raft_track_used": True,
        "object_flow_or_warp": False, "object_multiband": False, "maximum_object_handoffs": 1,
        "protection_margin_pixels": protection_margin_pixels, "track_count": len(applied_tracks),
        "track_audits": applied_tracks, "actual_output_object_pixel_count": int(np.count_nonzero(changed)),
    })


__all__ = ["D3ObjectFirstCompositorError", "D3ObjectFirstCompositorResult", "compose_d3_object_first_dense_source", "compose_d3_persistent_object_tracks"]
