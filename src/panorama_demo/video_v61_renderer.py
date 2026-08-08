"""Complete-canvas, fail-closed v6.1 tail-guarded candidate renderer.

The candidate keeps placement/alignment evidence separate from the final RGB
sample.  An accepted preview alignment is composed into the full-resolution
inverse grid first; every true raw source is then remapped exactly once.  All
later geometry, owner and seam audits observe those aligned final samples.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping, Sequence

import cv2
import numpy as np

from .calibrated_rgb_pushbroom import CalibratedRGBPushbroomResult
from .video_final_sampling import VideoSamplingSource, sample_video_sources_once
from .video_graphcut_seam import VideoGraphCutAudit, solve_video_graphcut_seam
from .video_hard_guards import (
    VideoHardGuards,
    audit_guard_owner_intersection,
    build_video_hard_guards,
)
from .video_local_alignment import VideoLocalAlignmentConfig, fit_near_protected_alignment
from .video_near_blend import (
    VideoNearBlendConfig,
    apply_near_multiband,
    build_near_blend_eligible_mask,
)
from .video_rgb_quality import assess_video_rgb_quality
from .video_v61_geometry_gate import (
    V61GeometryAudit,
    V61GeometryGateConfig,
    evaluate_v61_geometry_gate,
)
from .video_v6_pair_renderer import build_v6_sampling_sources
from .video_visual_renderer import video_dis_pair_evidence


@dataclass(frozen=True)
class V61PairState:
    old_frame_id: int
    new_frame_id: int
    gate_state: str
    geometry: V61GeometryAudit | None
    alignment_accepted: bool
    alignment_model: str
    alignment_grid_applied: bool
    graphcut_called: bool
    graphcut_accepted: bool
    fallback_reason: str | None
    blend_pixel_count: int
    actual_old_owner_pixels: int
    actual_new_owner_pixels: int
    not_evaluable_reason: str | None
    effective_seam_audit: VideoGraphCutAudit | None = None
    owner_fragment_reassigned_pixels: int = 0
    owner_support_dropped_pixels: int = 0


def _preview_bgr(source: VideoSamplingSource, factor: int) -> tuple[np.ndarray, np.ndarray]:
    image = cv2.remap(
        source.raw_bgr,
        source.inverse_x[::factor, ::factor].astype(np.float32),
        source.inverse_y[::factor, ::factor].astype(np.float32),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    return image, np.asarray(source.valid_mask, bool)[::factor, ::factor]


def _preview_alignment_config(factor: int) -> VideoLocalAlignmentConfig:
    return VideoLocalAlignmentConfig(
        near_translation_target_px=3.0 / factor,
        near_translation_hard_px=6.0 / factor,
        near_homography_corner_displacement_hard_px=6.0 / factor,
        near_homography_held_out_fb_p95_max_px=1.0 / factor,
        near_homography_held_out_fb_abs_max_px=2.0 / factor,
    )


def _apply_output_matrix_to_grid(
    source: VideoSamplingSource, matrix: np.ndarray, support: np.ndarray,
) -> VideoSamplingSource:
    """Compose one audited output-coordinate matrix before final RGB remap."""
    height, width = source.valid_mask.shape
    value = np.asarray(matrix, np.float64)
    if value.shape != (3, 3) or not np.isfinite(value).all() or abs(float(np.linalg.det(value))) <= 1e-9:
        raise RuntimeError("v6.1 structural_failure: accepted alignment matrix is invalid")
    yy, xx = np.indices((height, width), dtype=np.float32)
    points = np.column_stack((xx.ravel(), yy.ravel(), np.ones(height * width, np.float32)))
    projected = points @ value.T
    denominator = projected[:, 2]
    if np.any(~np.isfinite(denominator)) or np.any(np.abs(denominator) <= 1e-9):
        raise RuntimeError("v6.1 structural_failure: alignment projected outside finite grid")
    target_x = (projected[:, 0] / denominator).reshape((height, width)).astype(np.float32)
    target_y = (projected[:, 1] / denominator).reshape((height, width)).astype(np.float32)
    adjusted_x = cv2.remap(
        source.inverse_x.astype(np.float32), target_x, target_y, cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=-1.0,
    )
    adjusted_y = cv2.remap(
        source.inverse_y.astype(np.float32), target_x, target_y, cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=-1.0,
    )
    adjusted_valid = cv2.remap(
        source.valid_mask.astype(np.uint8), target_x, target_y, cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
    ).astype(bool)
    mask = np.asarray(support, bool)
    if mask.shape != source.valid_mask.shape:
        raise RuntimeError("v6.1 structural_failure: alignment support has wrong canvas")
    return VideoSamplingSource(
        source.frame_id,
        source.raw_bgr,
        np.where(mask, adjusted_x, source.inverse_x),
        np.where(mask, adjusted_y, source.inverse_y),
        np.where(mask, adjusted_valid, source.valid_mask),
    )


def _prepare_aligned_sources(
    sources: tuple[VideoSamplingSource, ...], *, factor: int = 4,
) -> tuple[tuple[VideoSamplingSource, ...], dict[tuple[int, int], dict[str, object]]]:
    """Estimate on diagnostic previews and modify only final inverse grids."""
    adjusted = list(sources)
    audits: dict[tuple[int, int], dict[str, object]] = {}
    scaling = np.diag((float(factor), float(factor), 1.0))
    for index in range(1, len(adjusted)):
        old, old_valid = _preview_bgr(adjusted[index - 1], factor)
        new, new_valid = _preview_bgr(adjusted[index], factor)
        old_id, new_id = int(adjusted[index - 1].frame_id), int(adjusted[index].frame_id)
        record: dict[str, object] = {
            "old_frame_id": old_id,
            "new_frame_id": new_id,
            "preview_scale": factor,
            "accepted": False,
            "selected_model": "not_evaluable",
            "matrix_composed_into_final_inverse_grid": False,
            "safe_background_grid_pixel_count": 0,
        }
        evidence = video_dis_pair_evidence(
            np.dstack((old, old_valid.astype(np.uint8) * 255)),
            np.dstack((new, new_valid.astype(np.uint8) * 255)),
            old_valid & new_valid,
        )
        if evidence is None:
            record["rejection_reason"] = "no_preview_dis_evidence"
            audits[(old_id, new_id)] = record
            continue
        guards = build_video_hard_guards(
            old, new, evidence, old_valid=old_valid, new_valid=new_valid,
        )
        safe = old_valid & new_valid & np.asarray(evidence.reliable_mask, bool) & ~guards.protected
        alignment = fit_near_protected_alignment(
            evidence, support=safe, plane_verified=False,
            config=_preview_alignment_config(factor),
        )
        record.update(
            accepted=bool(alignment.audit.accepted),
            selected_model=str(alignment.audit.selected_model),
            rejection_reason=alignment.audit.rejection_reason,
        )
        if not alignment.audit.accepted or alignment.matrix is None:
            audits[(old_id, new_id)] = record
            continue
        full_matrix = scaling @ np.asarray(alignment.matrix, np.float64) @ np.linalg.inv(scaling)
        full_support = cv2.resize(
            safe.astype(np.uint8),
            (adjusted[index].valid_mask.shape[1], adjusted[index].valid_mask.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        adjusted[index] = _apply_output_matrix_to_grid(adjusted[index], full_matrix, full_support)
        record.update(
            matrix_composed_into_final_inverse_grid=True,
            safe_background_grid_pixel_count=int(full_support.sum()),
            maximum_displacement_preview_px=alignment.audit.maximum_displacement_px,
        )
        audits[(old_id, new_id)] = record
    return tuple(adjusted), audits


def _with_tail_guard(guards: VideoHardGuards, tail: np.ndarray) -> VideoHardGuards:
    tail = np.asarray(tail, bool)
    protected = np.asarray(guards.protected, bool) | tail
    return replace(
        guards,
        protected=protected,
        hard_owner_old=np.asarray(guards.hard_owner_old, bool) | tail,
    )


def _real_internal_handoff(
    audit: VideoGraphCutAudit,
    choose_new: np.ndarray,
    width: int,
    competition: np.ndarray,
) -> bool:
    seam = tuple(int(value) for value in audit.seam_x_by_row)
    common = np.asarray(competition, bool)
    seam_rows = [
        value
        for row, value in enumerate(seam)
        if np.any(common[row]) and value >= 0
    ]
    return bool(
        np.any(np.asarray(choose_new, bool) & common)
        and np.any(~np.asarray(choose_new, bool) & common)
        and len(seam) == choose_new.shape[0]
        and seam_rows
        and all(4 <= value <= width - 4 for value in seam_rows)
    )


def _old_to_new_monotone(labels: np.ndarray, valid: np.ndarray) -> bool:
    for row in range(labels.shape[0]):
        values = np.asarray(labels[row])[np.asarray(valid[row], bool)]
        if values.size > 1 and np.any(values[1:].astype(np.int64) < values[:-1].astype(np.int64)):
            return False
    return True


def _component_counts(mask: np.ndarray, minimum_pixels: int = 8) -> tuple[int, int]:
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        np.asarray(mask, np.uint8), connectivity=8,
    )
    sizes = stats[1:, cv2.CC_STAT_AREA]
    return max(0, count - 2), int(np.count_nonzero(sizes < minimum_pixels))


def _remove_tiny_support_fragments(
    source: VideoSamplingSource,
    *,
    minimum_pixels: int = 8,
) -> tuple[VideoSamplingSource, int]:
    """Drop isolated inverse-map specks before owner competition.

    Calibrated boundary interpolation can leave a handful of disconnected
    valid cells outside the source's main strip.  They cannot form a legal
    monotone owner region.  Components below the frozen GraphCut fragment
    threshold are excluded by topology only; retained RGB still comes from
    the source's single final inverse remap.
    """

    valid = np.asarray(source.valid_mask, bool)
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        valid.astype(np.uint8), connectivity=8,
    )
    cleaned = valid.copy()
    removed = 0
    for label in range(1, count):
        size = int(stats[label, cv2.CC_STAT_AREA])
        if size < minimum_pixels:
            cleaned[labels == label] = False
            removed += size
    if removed == 0:
        return source, 0
    if not np.any(cleaned):
        raise RuntimeError(
            "v6.1 structural_failure: support fragment cleanup removed a complete source"
        )
    return replace(source, valid_mask=np.ascontiguousarray(cleaned)), removed


def _repair_pair_owner_fragments(
    owner: np.ndarray,
    *,
    old_frame_id: int,
    new_frame_id: int,
    supports: dict[int, np.ndarray],
    maximum_repair_pixels: int = 64,
    allow_unsupported_drop: bool = False,
) -> tuple[np.ndarray, int, int]:
    """Give a tiny cut-created island to the adjacent real source if valid.

    This does not invent pixels or relax the island gate.  Reassignment is
    allowed only when every fragment pixel is covered by the competing
    adjacent source; the caller subsequently reruns the complete topology
    audit.  Components above the bounded repair size remain structural
    failures.
    """

    repaired = np.asarray(owner, np.int32).copy()
    total = 0
    dropped = 0
    for frame_id, alternate_id in (
        (int(old_frame_id), int(new_frame_id)),
        (int(new_frame_id), int(old_frame_id)),
    ):
        mask = repaired == frame_id
        count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), connectivity=8,
        )
        if count <= 2:
            continue
        sizes = [int(value) for value in stats[1:, cv2.CC_STAT_AREA]]
        largest_label = 1 + int(np.argmax(sizes))
        alternate_support = np.asarray(supports[alternate_id], bool)
        for label in range(1, count):
            size = int(stats[label, cv2.CC_STAT_AREA])
            if label == largest_label or size > maximum_repair_pixels:
                continue
            fragment = labels == label
            if np.all(alternate_support[fragment]):
                repaired[fragment] = alternate_id
                total += size
                continue
            if allow_unsupported_drop:
                transferable = fragment & alternate_support
                if np.any(transferable):
                    repaired[transferable] = alternate_id
                    total += int(np.count_nonzero(transferable))
                unsupported = fragment & ~alternate_support
                if np.any(unsupported):
                    # A non-adjacent older support must not create a third
                    # label island inside this pair handoff.  Exclude this
                    # bounded projection speck from every final support;
                    # otherwise no chronological monotone owner exists.
                    for candidate_support in supports.values():
                        candidate_support[unsupported] = False
                    repaired[unsupported] = -1
                    dropped += int(np.count_nonzero(unsupported))
    return repaired, total, dropped


def _validate_owner(
    owner: np.ndarray,
    supports: dict[int, np.ndarray],
    *,
    require_complete: bool,
) -> dict[str, object]:
    union = np.logical_or.reduce(tuple(np.asarray(mask, bool) for mask in supports.values()))
    valid = np.asarray(owner) >= 0
    known = np.isin(owner[valid], tuple(supports))
    supported = bool(known.all())
    if supported:
        for frame_id, support in supports.items():
            supported &= not np.any((owner == int(frame_id)) & ~np.asarray(support, bool))
    partition = bool(np.all(valid == union)) if require_complete else bool(np.all(~valid | union))
    monotone = _old_to_new_monotone(owner, valid)
    islands = fragments = 0
    per_source_components: dict[int, dict[str, object]] = {}
    for frame_id in supports:
        frame_owner = owner == int(frame_id)
        extra, small = _component_counts(frame_owner)
        islands += extra
        fragments += small
        if extra or small:
            count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
                frame_owner.astype(np.uint8), connectivity=8,
            )
            per_source_components[int(frame_id)] = {
                "component_count": max(0, int(count) - 1),
                "component_pixel_counts": [
                    int(value) for value in stats[1:, cv2.CC_STAT_AREA]
                ],
            }
    result = {
        "valid_pixel_exactly_one_owner": partition,
        "owner_source_support_pass": supported,
        "owner_frame_ids_chronological_monotone": monotone,
        "owner_island_count": int(islands),
        "small_fragment_count": int(fragments),
        "nontrivial_source_components": per_source_components,
        "valid_pixel_count": int(valid.sum()),
    }
    if not partition or not supported or not monotone or islands or fragments:
        raise RuntimeError(f"v6.1 structural_failure: owner topology invalid: {result}")
    return result


def _hard_owner_handoff(
    owner: np.ndarray,
    output: np.ndarray,
    *,
    old_bgr: np.ndarray,
    new_bgr: np.ndarray,
    old_valid: np.ndarray,
    new_valid: np.ndarray,
    old_frame_id: int,
    new_frame_id: int,
    supports: Mapping[int, np.ndarray],
    guards: VideoHardGuards | None = None,
    corridor_left: int = 0,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Extend one true new owner through one deterministic vertical handoff."""
    height, width = owner.shape
    _yy, xx = np.indices((height, width), dtype=np.int32)
    old_owned_common = (owner == old_frame_id) & old_valid & new_valid
    candidates = np.flatnonzero(np.any(old_owned_common, axis=0))
    if candidates.size == 0:
        candidates = np.flatnonzero(np.any(old_valid & new_valid, axis=0))
    boundary = width if candidates.size == 0 else int((candidates.min() + candidates.max() + 1) // 2)
    if guards is not None:
        old_columns = np.flatnonzero(np.any(guards.hard_owner_old, axis=0))
        new_columns = np.flatnonzero(np.any(guards.hard_owner_new, axis=0))
        minimum = corridor_left + int(old_columns.max()) + 1 if old_columns.size else 0
        maximum = corridor_left + int(new_columns.min()) if new_columns.size else width
        if minimum > maximum:
            raise RuntimeError("v6.1 structural_failure: no monotone hard-owner handoff satisfies guards")
        boundary = min(max(boundary, minimum), maximum)
    # Keep an already-valid earlier real owner when the immediately previous
    # source has a projection hole.  Treating ``~old_valid`` alone as a new
    # owner mandate creates a left-side incoming-source island and reverses
    # chronological labels.  New-only pixels are claimed only when no prior
    # source owns them; the normal right-side handoff remains unchanged.
    choose_new = new_valid & ((xx >= boundary) | (owner < 0))
    if guards is not None:
        crop = choose_new[:, corridor_left : corridor_left + guards.protected.shape[1]]
        if audit_guard_owner_intersection(crop, guards):
            raise RuntimeError("v6.1 structural_failure: degraded owner intersects a hard guard")
    candidate = owner.copy()
    candidate[choose_new] = int(new_frame_id)
    candidate, reassigned, dropped = _repair_pair_owner_fragments(
        candidate,
        old_frame_id=old_frame_id,
        new_frame_id=new_frame_id,
        supports=supports,
        allow_unsupported_drop=True,
    )
    current_supports = {
        frame_id: mask
        for frame_id, mask in supports.items()
        if frame_id <= new_frame_id
    }
    _validate_owner(candidate, current_supports, require_complete=True)
    rendered = output.copy()
    rendered[candidate == int(old_frame_id)] = old_bgr[candidate == int(old_frame_id)]
    rendered[candidate == int(new_frame_id)] = new_bgr[candidate == int(new_frame_id)]
    return candidate, rendered, reassigned, dropped


def _effective_handoff_audit(
    owner: np.ndarray,
    old_frame_id: int,
    new_frame_id: int,
    *,
    graphcut_called: bool,
    graphcut_accepted: bool,
    rejection_reason: str | None,
) -> VideoGraphCutAudit:
    """Measure the actual final owner boundary, including degraded hard cuts."""
    seam: list[int] = []
    monotone = True
    for row in range(owner.shape[0]):
        columns = np.flatnonzero((owner[row] == old_frame_id) | (owner[row] == new_frame_id))
        if columns.size == 0:
            seam.append(-1)
            continue
        labels = owner[row, columns] == new_frame_id
        if np.any(labels[:-1] & ~labels[1:]):
            monotone = False
        old_columns = columns[~labels]
        new_columns = columns[labels]
        if old_columns.size and new_columns.size:
            seam.append(int(new_columns.min()))
        else:
            seam.append(-1)
    known = [value for value in seam if value >= 0]
    maximum = None if len(known) < 2 else max(
        abs(right - left) for left, right in zip(known, known[1:])
    )
    islands, fragments = _component_counts(owner == new_frame_id)
    topology = bool(monotone and islands == 0 and fragments == 0)
    accepted = bool(graphcut_accepted and topology and known and (maximum is None or maximum <= 1))
    return VideoGraphCutAudit(
        graphcut_called,
        False,
        tuple(seam),
        maximum,
        islands,
        fragments,
        True,
        accepted,
        None if accepted else rejection_reason or "effective_hard_owner_handoff",
        0,
    )


def _geometry_report(audit: V61GeometryAudit | None) -> dict[str, object] | None:
    if audit is None:
        return None
    method = getattr(audit, "as_report_dict", None)
    if callable(method):
        return dict(method())
    result: dict[str, object] = {}
    for key, value in vars(audit).items():
        if isinstance(value, (bool, str, int, float)) or value is None:
            result[key] = value.item() if isinstance(value, np.generic) else value
    return result


def _config_report(config: V61GeometryGateConfig) -> dict[str, object]:
    return {
        key: value.item() if isinstance(value, np.generic) else value
        for key, value in vars(config).items()
        if isinstance(value, (bool, str, int, float, np.generic)) or value is None
    }


def _pair_report(state: V61PairState, observation: object | None) -> dict[str, object]:
    seam = state.effective_seam_audit
    report: dict[str, object] = {
        "old_frame_id": state.old_frame_id,
        "new_frame_id": state.new_frame_id,
        "gate_state": state.gate_state,
        "geometry": _geometry_report(state.geometry),
        "alignment_accepted": state.alignment_accepted,
        "alignment_model": state.alignment_model,
        "alignment_grid_applied": state.alignment_grid_applied,
        "graphcut_called": state.graphcut_called,
        "graphcut_accepted": state.graphcut_accepted,
        "fallback_reason": state.fallback_reason,
        "blend_pixel_count": state.blend_pixel_count,
        "actual_old_owner_pixels": state.actual_old_owner_pixels,
        "actual_new_owner_pixels": state.actual_new_owner_pixels,
        "not_evaluable_reason": state.not_evaluable_reason,
        "owner_fragment_reassigned_pixels": state.owner_fragment_reassigned_pixels,
        "owner_support_dropped_pixels": state.owner_support_dropped_pixels,
    }
    if seam is not None:
        report["effective_owner_handoff"] = {
            "evaluated": state.not_evaluable_reason is None,
            "evaluated_seam_rows": int(sum(value >= 0 for value in seam.seam_x_by_row)),
            "maximum_adjacent_row_step_px": seam.maximum_adjacent_row_step_px,
            "owner_island_count": seam.owner_island_count,
            "small_fragment_count": seam.small_fragment_count,
            "valid_pixel_exactly_one_owner": seam.valid_pixel_exactly_one_owner,
            "double_edge_count": None if observation is None else int(observation.double_edge_count),
            "ghost_count": None if observation is None else int(observation.ghost_count),
        }
    return report


def _coerce_geometry_config(
    value: V61GeometryGateConfig | Mapping[str, object] | None,
) -> V61GeometryGateConfig:
    if value is None:
        return V61GeometryGateConfig()
    if isinstance(value, V61GeometryGateConfig):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("geometry_gate_config must be a V61GeometryGateConfig or mapping")
    return V61GeometryGateConfig(**dict(value))


def _audit_open3d_edges(
    frames: Sequence[object], open3d_edges: Sequence[object],
) -> dict[str, object]:
    """Bind actual CUDA Open3D evidence to every adjacent real source."""
    frame_ids = tuple(int(getattr(frame, "frame_id")) for frame in frames)
    if len(open3d_edges) != len(frame_ids) - 1:
        raise RuntimeError("v6.1 structural_failure: Open3D must audit every adjacent render edge")
    summaries: list[dict[str, object]] = []
    quality_pass = True
    for (reference_id, source_id), edge in zip(
        zip(frame_ids[:-1], frame_ids[1:], strict=True), open3d_edges, strict=True,
    ):
        actual_reference = getattr(edge, "reference_node_id", None)
        actual_source = getattr(edge, "source_node_id", None)
        if (actual_reference, actual_source) != (reference_id, source_id):
            raise RuntimeError(
                "v6.1 structural_failure: Open3D edge does not bind its adjacent real sources"
            )
        if getattr(edge, "structurally_valid", None) is not True:
            raise RuntimeError("v6.1 structural_failure: Open3D edge is not structurally valid")
        backend = str(getattr(edge, "backend", ""))
        if backend != "open3d_tensor_cuda_rgbd":
            raise RuntimeError(
                "v6.1 structural_failure: Open3D edge did not use open3d_tensor_cuda_rgbd"
            )
        reliable = getattr(edge, "reliable", None) is True
        reasons = tuple(str(value) for value in getattr(edge, "failure_reasons", ()))
        reliable = bool(reliable and not reasons)
        quality_pass &= reliable
        summaries.append({
            "reference_node_id": reference_id,
            "source_node_id": source_id,
            "backend": backend,
            "structurally_valid": True,
            "reliable": reliable,
            "failure_reasons": list(reasons),
        })
    return {
        "valid": True,
        "sample_count": len(summaries),
        "audit_passed": bool(quality_pass),
        "edges": summaries,
    }


def render_video_v61_real_sources(
    sources: tuple[VideoSamplingSource, ...],
    *,
    geometry_gate_config: V61GeometryGateConfig | Mapping[str, object] | None = None,
    orb_pose_count: int | None = None,
    open3d_edge_count: int | None = None,
    open3d_edge_evidence: Mapping[str, object] | None = None,
) -> CalibratedRGBPushbroomResult:
    """Render N true sources; every pair ends GraphCut, degraded hard owner, or F."""
    if len(sources) < 2:
        raise ValueError("v6.1 requires at least two direct real sources")
    ids = tuple(int(source.frame_id) for source in sources)
    if ids != tuple(sorted(set(ids))):
        raise RuntimeError("v6.1 structural_failure: source ids must be unique and chronological")
    shape = sources[0].valid_mask.shape
    if any(source.valid_mask.shape != shape for source in sources):
        raise RuntimeError("v6.1 structural_failure: every source must share the complete canvas")
    config = _coerce_geometry_config(geometry_gate_config)
    aligned_sources, alignment_audits = _prepare_aligned_sources(sources)
    cleaned_records = tuple(
        _remove_tiny_support_fragments(source) for source in aligned_sources
    )
    aligned_sources = tuple(record[0] for record in cleaned_records)
    support_fragment_cleanup = {
        int(source.frame_id): int(record[1])
        for source, record in zip(aligned_sources, cleaned_records, strict=True)
    }
    sampled = sample_video_sources_once(aligned_sources)
    supports = {
        int(source.frame_id): np.asarray(source.valid_mask, bool).copy()
        for source in aligned_sources
    }
    owner = np.full(shape, -1, np.int32)
    owner[supports[ids[0]]] = ids[0]
    output = sampled[0][1].copy()
    states: list[V61PairState] = []
    dis_evidence_count = 0

    for (old_source, old_bgr), (new_source, new_bgr) in zip(sampled[:-1], sampled[1:], strict=True):
        old_id, new_id = int(old_source.frame_id), int(new_source.frame_id)
        alignment = alignment_audits[(old_id, new_id)]
        alignment_accepted = bool(alignment["accepted"])
        alignment_applied = bool(alignment["matrix_composed_into_final_inverse_grid"])
        alignment_model = str(alignment["selected_model"])
        old_valid, new_valid = supports[old_id], supports[new_id]
        common = old_valid & new_valid
        rows, columns = np.where(common)
        geometry: V61GeometryAudit | None = None
        guards: VideoHardGuards | None = None
        left = 0
        evidence = None
        fallback_reason: str | None = None
        graphcut_called = False
        graphcut_accepted = False
        blend_pixels = 0
        fragment_reassigned_pixels = 0
        support_dropped_pixels = 0
        rollback: tuple[np.ndarray, np.ndarray] | None = None

        if rows.size:
            corridor_width = min(160, int(columns.max()) - int(columns.min()) + 1)
            if corridor_width >= 96:
                left = max(
                    0,
                    min(shape[1] - corridor_width, (int(columns.min()) + int(columns.max()) + 1 - corridor_width) // 2),
                )
                right = left + corridor_width
                old_crop, new_crop = old_bgr[:, left:right], new_bgr[:, left:right]
                old_crop_valid, new_crop_valid = old_valid[:, left:right], new_valid[:, left:right]
                evidence = video_dis_pair_evidence(
                    np.dstack((old_crop, old_crop_valid.astype(np.uint8) * 255)),
                    np.dstack((new_crop, new_crop_valid.astype(np.uint8) * 255)),
                    old_crop_valid & new_crop_valid,
                )
                if evidence is not None:
                    dis_evidence_count += 1
                    base_guards = build_video_hard_guards(
                        old_crop, new_crop, evidence,
                        old_valid=old_crop_valid, new_valid=new_crop_valid,
                    )
                    geometry = evaluate_v61_geometry_gate(
                        old_crop,
                        new_crop,
                        evidence,
                        support=old_crop_valid & new_crop_valid,
                        protected=base_guards.protected,
                        config=config,
                    )
                    guards = _with_tail_guard(base_guards, geometry.tail_guard)
                else:
                    fallback_reason = "no_fb_dis_evidence"
            else:
                fallback_reason = "corridor_below_96px"
        else:
            fallback_reason = "no_common_real_support"

        gate_pass = bool(
            evidence is not None
            and geometry is not None
            and geometry.accepted
            and alignment_accepted
            and alignment_applied
            and guards is not None
        )
        if not gate_pass and fallback_reason is None:
            if not alignment_accepted or not alignment_applied:
                fallback_reason = str(alignment.get("rejection_reason") or "alignment_not_applied")
            elif geometry is not None:
                fallback_reason = geometry.rejection_reason or "geometry_gate_failed"

        if gate_pass:
            graphcut_called = True
            assert evidence is not None and geometry is not None and guards is not None
            right = left + guards.protected.shape[1]
            old_crop, new_crop = old_bgr[:, left:right], new_bgr[:, left:right]
            old_crop_valid, new_crop_valid = old_valid[:, left:right], new_valid[:, left:right]
            try:
                graphcut = solve_video_graphcut_seam(
                    old_crop,
                    new_crop,
                    old_crop_valid,
                    new_crop_valid,
                    hard_owner_old=guards.hard_owner_old,
                    hard_owner_new=guards.hard_owner_new,
                )
                graphcut = replace(graphcut, audit=replace(graphcut.audit, canvas_x_offset=left))
                graphcut_failures: list[str] = []
                if not graphcut.audit.accepted:
                    graphcut_failures.append(graphcut.audit.rejection_reason or "graphcut_audit_rejected")
                if audit_guard_owner_intersection(graphcut.choose_new, guards):
                    graphcut_failures.append("hard_guard_owner_intersection")
                if not _real_internal_handoff(
                    graphcut.audit,
                    graphcut.choose_new,
                    right - left,
                    old_crop_valid & new_crop_valid,
                ):
                    graphcut_failures.append("no_real_internal_two_owner_handoff")
                if not _old_to_new_monotone(
                    graphcut.choose_new.astype(np.int8), old_crop_valid | new_crop_valid,
                ):
                    graphcut_failures.append("graphcut_owner_not_old_to_new_monotone")
                safe = not graphcut_failures
                if not safe:
                    fallback_reason = ",".join(graphcut_failures)
                else:
                    rollback = (owner.copy(), output.copy())
                    crop_owner = owner[:, left:right]
                    crop_owner[graphcut.choose_new] = new_id
                    owner[:, left:right] = crop_owner
                    owner[new_valid & ~old_valid] = new_id
                    owner, fragment_reassigned_pixels, _graphcut_dropped = (
                        _repair_pair_owner_fragments(
                        owner,
                        old_frame_id=old_id,
                        new_frame_id=new_id,
                        supports=supports,
                        allow_unsupported_drop=False,
                    ))
                    output[owner == old_id] = old_bgr[owner == old_id]
                    output[owner == new_id] = new_bgr[owner == new_id]
                    _validate_owner(
                        owner,
                        {frame_id: support for frame_id, support in supports.items() if frame_id <= new_id},
                        require_complete=True,
                    )
                    eligible = build_near_blend_eligible_mask(
                        old_crop_valid, new_crop_valid, evidence, guards,
                    )
                    blended, _band, blend = apply_near_multiband(
                        old_crop,
                        new_crop,
                        output[:, left:right],
                        graphcut.choose_new,
                        eligible,
                        guards,
                        config=VideoNearBlendConfig(near_width_px=2),
                    )
                    output[:, left:right] = blended
                    blend_pixels = int(blend.band_pixel_count)
                    graphcut_accepted = True
                    rollback = None
            except (RuntimeError, ValueError, cv2.error) as error:
                if rollback is not None:
                    owner, output = rollback
                    rollback = None
                fallback_reason = f"graphcut_exception:{type(error).__name__}"

        if not graphcut_accepted:
            (
                owner,
                output,
                fragment_reassigned_pixels,
                support_dropped_pixels,
            ) = _hard_owner_handoff(
                owner,
                output,
                old_bgr=old_bgr,
                new_bgr=new_bgr,
                old_valid=old_valid,
                new_valid=new_valid,
                old_frame_id=old_id,
                new_frame_id=new_id,
                supports=supports,
                guards=guards,
                corridor_left=left,
            )
            blend_pixels = 0
        states.append(V61PairState(
            old_id,
            new_id,
            "graphcut_accepted" if graphcut_accepted else "hard_owner_degraded",
            geometry,
            alignment_accepted,
            alignment_model,
            alignment_applied,
            graphcut_called,
            graphcut_accepted,
            None if graphcut_accepted else fallback_reason or "graphcut_not_accepted",
            blend_pixels,
            0,
            0,
            None,
            owner_fragment_reassigned_pixels=fragment_reassigned_pixels,
            owner_support_dropped_pixels=support_dropped_pixels,
        ))

    owner_audit = _validate_owner(owner, supports, require_complete=True)
    effective_audits: list[VideoGraphCutAudit] = []
    effective_state_indices: list[int] = []
    final_states: list[V61PairState] = []
    for index, state in enumerate(states):
        audit = _effective_handoff_audit(
            owner,
            state.old_frame_id,
            state.new_frame_id,
            graphcut_called=state.graphcut_called,
            graphcut_accepted=state.graphcut_accepted,
            rejection_reason=state.fallback_reason,
        )
        evaluated_rows = sum(value >= 0 for value in audit.seam_x_by_row)
        reason = None if evaluated_rows >= 2 else "no_effective_final_two_owner_handoff"
        if reason is None:
            effective_audits.append(audit)
            effective_state_indices.append(index)
        final_states.append(replace(
            state,
            actual_old_owner_pixels=int(np.count_nonzero(owner == state.old_frame_id)),
            actual_new_owner_pixels=int(np.count_nonzero(owner == state.new_frame_id)),
            not_evaluable_reason=reason,
            effective_seam_audit=audit,
        ))
    states = final_states
    valid = owner >= 0
    quality = assess_video_rgb_quality(output, owner, valid, tuple(effective_audits))
    observations: list[object | None] = [None] * len(states)
    for index, observation in zip(effective_state_indices, quality.seam_observations, strict=True):
        observations[index] = observation
    degraded = any(state.gate_state == "hard_owner_degraded" for state in states)
    all_seams_evaluable = len(effective_audits) == len(states)
    strict = bool(quality.strict_quality_pass and not degraded and all_seams_evaluable)
    failure_reasons = list(quality.failure_reasons)
    if degraded:
        failure_reasons.append("hard_owner_degraded")
    if not all_seams_evaluable:
        failure_reasons.append("effective_seam_not_evaluable")

    pose_count = len(sources) if orb_pose_count is None else int(orb_pose_count)
    edge_count = len(sources) - 1 if open3d_edge_count is None else int(open3d_edge_count)
    if open3d_edge_evidence is None:
        edge_record: dict[str, object] = {
            "valid": edge_count == len(sources) - 1,
            "sample_count": edge_count,
            "audit_passed": edge_count == len(sources) - 1,
            "prevalidated_by_real_source_caller": True,
        }
    else:
        edge_record = dict(open3d_edge_evidence)
    evidence_records = {
        "orb_anchor_trajectory": {
            "valid": pose_count == len(sources),
            "sample_count": pose_count,
            "audit_passed": pose_count == len(sources),
        },
        "open3d_rgbd_edges": edge_record,
        "dis_forward_backward": {
            "valid": dis_evidence_count == len(sources) - 1,
            "sample_count": dis_evidence_count,
            "audit_passed": dis_evidence_count == len(sources) - 1,
        },
    }
    component_applied = int(valid.sum())
    evidence_pass = all(bool(record["audit_passed"]) for record in evidence_records.values())
    if not evidence_pass:
        strict = False
        failure_reasons.append("required_component_evidence_not_passed")
    metadata: dict[str, object] = {
        "schema": "video-v61-tail-guarded-full-panorama/v1",
        "renderer": "V61_tail_guarded_full_panorama",
        "candidate_only": True,
        "geometry_gate_config": _config_report(config),
        "pair_states": [
            _pair_report(state, observation)
            for state, observation in zip(states, observations, strict=True)
        ],
        "alignment_execution": [alignment_audits[key] for key in sorted(alignment_audits)],
        "owner_audit": owner_audit,
        "support_fragment_cleanup": {
            "minimum_component_pixels": 8,
            "maximum_handoff_component_repair_pixels": 64,
            "removed_pixel_count_by_source": support_fragment_cleanup,
            "removed_pixel_count": int(sum(support_fragment_cleanup.values())),
            "handoff_unsupported_pixel_count": int(
                sum(state.owner_support_dropped_pixels for state in states)
            ),
        },
        "quality_metrics": {
            "quality_pass": strict,
            "strict_quality_pass": strict,
            "failure_reasons": failure_reasons,
            "grade": "B" if strict else "C",
            "manual_review_required": not strict,
            "seam_step_p95_px": quality.seam_step_p95_px,
            "seam_step_abs_max_px": quality.seam_step_abs_max_px,
            "staircase_run_count": quality.staircase_run_count,
            "double_edge_count": quality.double_edge_count,
            "ghost_count": quality.ghost_count,
            "evaluated_seam_pair_count": len(effective_audits),
        },
        "raw_rgb_once_sampling": {
            "source_frame_ids": list(ids),
            "source_sampling_call_count": len(sources),
            "full_resolution_inverse_remap_call_count_by_source": [1] * len(sources),
            "diagnostic_preview_scale": 4,
            "exactly_once": True,
        },
        "component_evidence": evidence_records,
        "component_execution": {
            "v61_tail_guarded_full_panorama": {
                "required": True,
                "initialized": True,
                "applied_to_output": component_applied > 0,
                "applied_output_pixel_count": component_applied,
            },
        },
        "executed_candidate_components": {
            "v61_tail_guarded_full_panorama": component_applied > 0,
        },
        "candidate_run_state": "completed" if component_applied > 0 else "invalid_component_execution",
        "selection_eligible": bool(component_applied > 0 and evidence_pass and strict),
    }
    return CalibratedRGBPushbroomResult(output, metadata, owner_frame_id=owner)


def render_video_v61_candidate(
    frames: tuple[object, ...],
    poses: tuple[np.ndarray, ...],
    calibration: object,
    *,
    pushbroom_config: dict[str, object],
    rgb_motions: list[object],
    motion_pixels_to_full_resolution: float,
    geometry_gate_config: V61GeometryGateConfig | Mapping[str, object] | None = None,
    open3d_edges: Sequence[object] = (),
    **_ignored: object,
) -> CalibratedRGBPushbroomResult:
    if len(poses) != len(frames):
        raise RuntimeError("v6.1 structural_failure: every render source requires a direct ORB pose")
    open3d_evidence = _audit_open3d_edges(frames, open3d_edges)
    sources = build_v6_sampling_sources(
        frames,
        poses,
        calibration,
        pushbroom_config=pushbroom_config,
        rgb_motions=rgb_motions,
        motion_pixels_to_full_resolution=motion_pixels_to_full_resolution,
    )
    return render_video_v61_real_sources(
        sources,
        geometry_gate_config=geometry_gate_config,
        orb_pose_count=len(poses),
        open3d_edge_count=len(open3d_edges),
        open3d_edge_evidence=open3d_evidence,
    )


__all__ = ["V61PairState", "render_video_v61_candidate", "render_video_v61_real_sources"]
