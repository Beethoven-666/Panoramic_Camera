"""Phase 1.3 diagnostic-only visual seam calibration samples.

This module is deliberately not a renderer, CLI entry point, candidate lock,
or production dependency.  It uses real captured RGB pairs to make blinded
human-review crops.  GraphCut is authorised here solely to create those
samples; a protected pixel is always a hard owner and no depth creates colour.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

import cv2
import numpy as np

from .video_graphcut_seam import solve_video_graphcut_seam
from .video_hard_guards import VideoHardGuards, audit_guard_owner_intersection, build_video_hard_guards
from .video_local_alignment import fit_near_protected_alignment
from .video_near_blend import VideoNearBlendConfig, apply_near_multiband, build_near_blend_eligible_mask
from .video_object_mask import build_video_object_masks
from .video_v61_blocker_poc import (
    _aligned_new_image, _alignment_config, _capture_rgb_paths, _central_corridor, _coarse_strip_placement,
    default_v61_blocker_specs,
)
from .video_visual_renderer import VideoDISPairEvidence, video_dis_pair_evidence


@dataclass(frozen=True)
class Phase13VisualCalibrationConfig:
    """One fixed diagnostic calibration recipe; never data-set tuned."""

    corridor_width_px: int = 160
    blend_width_px: int = 2
    core_half_width_px: int = 4
    context_half_width_px: int = 16
    handoff_anchor_width_px: int = 4
    diagnostic_only: bool = True

    def __post_init__(self) -> None:
        if not self.diagnostic_only:
            raise ValueError("Phase 1.3 GraphCut is diagnostic-only")
        if self.corridor_width_px != 160 or self.blend_width_px not in {2, 3, 4}:
            raise ValueError("Phase 1.3 uses one fixed 160px corridor and 2--4px narrow blend")
        if self.core_half_width_px != 4 or self.context_half_width_px < 16 or self.handoff_anchor_width_px != 4:
            raise ValueError("Phase 1.3 review windows are fixed at 4px core and >=16px context")


@dataclass(frozen=True)
class Phase13Pair:
    session_name: str
    session_root: Path
    left_frame_id: int
    right_frame_id: int


@dataclass(frozen=True)
class Phase13PairResult:
    sample_id: str
    split: str
    generated: bool
    reason: str | None
    graphcut_called: bool
    graphcut_accepted: bool
    graphcut_guard_intersection_pixels: int
    blend_guard_intersection_pixels: int
    protected_pixel_count: int
    occlusion_pixel_count: int
    low_confidence_pixel_count: int
    anomalous_edgelet_pixel_count: int
    seam_coordinates: tuple[int, ...]
    core_metrics: dict[str, float | int | None]
    context_metrics: dict[str, float | int | None]
    old_owner_pixel_count: int = 0
    new_owner_pixel_count: int = 0
    interior_seam_row_count: int = 0


EvidenceFactory = Callable[[np.ndarray, np.ndarray, np.ndarray], VideoDISPairEvidence | None]


def phase13_pairs(captures_root: Path) -> tuple[Phase13Pair, ...]:
    """Expand the already frozen A--M--B evidence into adjacent real pairs."""
    pairs: list[Phase13Pair] = []
    for spec in default_v61_blocker_specs(Path(captures_root)):
        ids = (spec.left_frame_id, *spec.middle_frame_ids, spec.right_frame_id)
        pairs.extend(Phase13Pair(spec.name, spec.session_root, left, right) for left, right in zip(ids[:-1], ids[1:], strict=True))
    return tuple(pairs)


def _sample_id(pair: Phase13Pair) -> str:
    digest = hashlib.sha256(f"phase13-review-v1/{pair.session_name}/{pair.left_frame_id}/{pair.right_frame_id}".encode()).hexdigest()
    return f"sample-{digest[:12]}"


def _split(pair: Phase13Pair) -> str:
    """Frozen before rendering: one third held out, without looking at RGB metrics."""
    digest = hashlib.sha256(f"phase13-split-v1/{pair.session_name}/{pair.left_frame_id}/{pair.right_frame_id}".encode()).digest()
    return "held_out" if digest[0] % 3 == 0 else "calibration"


def frozen_phase13_split(captures_root: Path) -> dict[str, object]:
    pairs = phase13_pairs(captures_root)
    return {
        "schema": "video-v61-phase13-split-lock/v1",
        "purpose": "Frozen calibration/held-out assignment before visual samples are generated",
        "assignment_rule": "SHA-256 phase13-split-v1 tuple; held_out when first byte mod 3 is zero",
        "pairs": [
            {"sample_id": _sample_id(pair), "session": pair.session_name, "left_frame_id": pair.left_frame_id,
             "right_frame_id": pair.right_frame_id, "split": _split(pair)}
            for pair in pairs
        ],
    }


def _read_pair(pair: Phase13Pair) -> tuple[np.ndarray, np.ndarray]:
    paths = _capture_rgb_paths(pair.session_root)
    images: list[np.ndarray] = []
    for frame_id in (pair.left_frame_id, pair.right_frame_id):
        path = paths.get(frame_id)
        if path is None:
            raise ValueError("Phase 1.3 only accepts a frames.csv-listed real capture RGB frame")
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"cannot decode frozen RGB capture frame: {path}")
        images.append(image)
    return images[0], images[1]


def _edgelet_protection(old: np.ndarray, new: np.ndarray, evidence: VideoDISPairEvidence) -> np.ndarray:
    old_edge = cv2.Canny(cv2.cvtColor(old, cv2.COLOR_BGR2GRAY), 80, 160) > 0
    new_edge = cv2.Canny(cv2.cvtColor(new, cv2.COLOR_BGR2GRAY), 80, 160) > 0
    residual = np.asarray(evidence.gradient_residual, np.float32)
    finite = residual[np.isfinite(residual)]
    threshold = float(np.percentile(finite, 90.0)) if finite.size else float("inf")
    anomalous = (old_edge ^ new_edge) | ((old_edge | new_edge) & (residual >= threshold))
    return cv2.dilate(anomalous.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)


def _protected_guards(
    old: np.ndarray, new: np.ndarray, evidence: VideoDISPairEvidence, old_valid: np.ndarray, new_valid: np.ndarray,
) -> tuple[VideoHardGuards, dict[str, int]]:
    base = build_video_hard_guards(old, new, evidence, old_valid=old_valid.copy(), new_valid=new_valid.copy())
    anomalous = _edgelet_protection(old, new, evidence)
    low_confidence = ~np.asarray(evidence.reliable_mask, bool)
    objects = build_video_object_masks(evidence, strong_protection=base.protected | anomalous | low_confidence)
    protected = base.protected | anomalous | low_confidence | np.asarray(evidence.occlusion_risk_mask, bool) | objects.protected_mask
    protected &= old_valid | new_valid
    # Diagnostic policy: any protected/uncertain location remains the old real
    # owner where possible; only an old-invalid pixel can name the new owner.
    hard_old = protected & old_valid
    hard_new = protected & ~old_valid & new_valid
    guards = VideoHardGuards(
        base.line_guard, base.object_outer_boundary, base.thin_structure, base.occlusion_risk,
        protected, hard_old, hard_new, base.audit,
    )
    return guards, {
        "protected_pixel_count": int(protected.sum()),
        "occlusion_pixel_count": int(np.count_nonzero(evidence.occlusion_risk_mask)),
        "low_confidence_pixel_count": int(np.count_nonzero(low_confidence)),
        "anomalous_edgelet_pixel_count": int(np.count_nonzero(anomalous)),
    }


def _fixed_handoff_anchors(
    guards: VideoHardGuards, old_valid: np.ndarray, new_valid: np.ndarray, width_px: int,
) -> tuple[VideoHardGuards, int, int]:
    """Set fixed, safe outer anchors so GraphCut must evaluate a real handoff.

    The anchors never override a protected owner.  They are a single global
    Phase-1.3 configuration, not a data-set-specific seam adjustment.
    """
    old_valid, new_valid = np.asarray(old_valid, bool), np.asarray(new_valid, bool)
    # The original guard builder defaults every protected pixel to old.  That
    # is safe for a one-owner fallback, but it creates unavoidable islands in
    # any real left-to-right handoff.  Here each *whole* protected component
    # receives one fixed owner by its centroid relative to the corridor
    # midline; the seam still cannot enter it.  This is a global POC rule,
    # never a per-pair data-cost or threshold adjustment.
    protected = np.asarray(guards.protected, bool)
    labels_count, labels = cv2.connectedComponents(protected.astype(np.uint8), connectivity=8)
    hard_old, hard_new = np.zeros_like(protected), np.zeros_like(protected)
    for label in range(1, labels_count):
        component = labels == label
        median_x = float(np.median(np.nonzero(component)[1]))
        prefer_new = median_x >= (protected.shape[1] / 2.0)
        assign_new = component & new_valid & (prefer_new | ~old_valid)
        hard_new |= assign_new
        hard_old |= component & ~assign_new & old_valid
    safe = old_valid & new_valid & ~protected
    columns = np.arange(safe.shape[1])[None, :]
    old_anchor = safe & (columns < width_px)
    new_anchor = safe & (columns >= safe.shape[1] - width_px)
    hard_old |= old_anchor
    hard_new |= new_anchor
    if np.any(hard_old & hard_new):
        raise RuntimeError("fixed diagnostic handoff anchors overlapped a protected owner")
    return VideoHardGuards(
        guards.line_guard, guards.object_outer_boundary, guards.thin_structure, guards.occlusion_risk,
        guards.protected, hard_old, hard_new, guards.audit,
    ), int(old_anchor.sum()), int(new_anchor.sum())


def _seam_windows(seam: Iterable[int], width: int, half_width: int) -> tuple[np.ndarray, np.ndarray]:
    left, right = np.zeros((480, width), bool), np.zeros((480, width), bool)
    for row, coordinate in enumerate(seam):
        if coordinate < 0:
            continue
        left[row, max(0, coordinate - half_width):coordinate] = True
        right[row, coordinate:min(width, coordinate + half_width)] = True
    return left, right


def _p95(values: np.ndarray) -> float | None:
    finite = np.asarray(values, np.float64)
    finite = finite[np.isfinite(finite)]
    return None if not finite.size else float(np.percentile(finite, 95.0))


def _review_metrics(output: np.ndarray, seam: tuple[int, ...], config: Phase13VisualCalibrationConfig) -> tuple[dict[str, float | int | None], dict[str, float | int | None]]:
    height, width = output.shape[:2]
    core_left, core_right = _seam_windows(seam, width, config.core_half_width_px)
    # Pair corresponding pixels measured at equal distance from the seam.
    lab = cv2.cvtColor(output, cv2.COLOR_BGR2LAB).astype(np.float32)
    gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY).astype(np.float32)
    grad = cv2.magnitude(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1))
    delta_e: list[float] = []
    luma: list[float] = []
    gradient: list[float] = []
    line_break = 0
    edges = cv2.Canny(gray.astype(np.uint8), 80, 160) > 0
    for row, x in enumerate(seam):
        if x < config.core_half_width_px or x + config.core_half_width_px >= width:
            continue
        for offset in range(1, config.core_half_width_px + 1):
            a, b = x - offset, x + offset - 1
            delta_e.append(float(np.linalg.norm(lab[row, a] - lab[row, b])))
            luma.append(abs(float(gray[row, a] - gray[row, b])))
            gradient.append(abs(float(grad[row, a] - grad[row, b])))
        if x >= 2 and x + 2 < width and bool(edges[row, x - 2]) != bool(edges[row, x + 1]):
            line_break += 1
    context_left, context_right = _seam_windows(seam, width, config.context_half_width_px)
    vertical = np.zeros_like(edges)
    vertical[1:-1] = edges[1:-1] & (edges[:-2] | edges[2:])
    double = ghost = 0
    for row, x in enumerate(seam):
        if x < 2 or x + 2 >= width:
            continue
        window = vertical[row, x - 2:x + 3]
        starts = int(window[0]) + int(np.count_nonzero(~window[:-1] & window[1:]))
        double += int(starts >= 2)
        ghost += int(starts >= 2 and bool(window[0]) and bool(window[-1]))
    core = {
        "core_pixels_each_side": int(min(core_left.sum(), core_right.sum())),
        "lab_delta_e_p95": _p95(np.asarray(delta_e)), "luma_step_p95": _p95(np.asarray(luma)),
        "gradient_step_p95": _p95(np.asarray(gradient)),
        "not_evaluable": ([] if delta_e else ["lab_delta_e_p95", "luma_step_p95", "gradient_step_p95"]),
    }
    context = {
        "context_pixels_each_side": int(min(context_left.sum(), context_right.sum())),
        "double_edge_count": int(double), "ghost_count": int(ghost), "line_continuity_break_suspect_count": int(line_break),
        "not_evaluable": ([] if any(x >= 2 and x + 2 < width for x in seam) else ["double_edge_count", "ghost_count", "line_continuity_break_suspect_count"]),
    }
    return core, context


def render_phase13_pair(pair: Phase13Pair, *, config: Phase13VisualCalibrationConfig | None = None, evidence_factory: EvidenceFactory | None = None) -> tuple[Phase13PairResult, np.ndarray | None, np.ndarray | None]:
    """Make one review crop; whole-overlap edge-absolute residual is recorded nowhere as a veto."""
    settings = config or Phase13VisualCalibrationConfig()
    sample_id, split = _sample_id(pair), _split(pair)
    old_full, new_raw = _read_pair(pair)
    coarse, _, _, _ = _coarse_strip_placement(old_full, new_raw)
    coarse_new, coarse_valid = _aligned_new_image(new_raw, coarse)
    old, new = _central_corridor(old_full, settings.corridor_width_px), _central_corridor(coarse_new, settings.corridor_width_px)
    old_valid = np.ones(old.shape[:2], bool)
    new_valid = _central_corridor(coarse_valid.astype(np.uint8), settings.corridor_width_px).astype(bool)
    factory = evidence_factory or (lambda a, b, support: video_dis_pair_evidence(np.dstack((a, support.astype(np.uint8) * 255)), np.dstack((b, support.astype(np.uint8) * 255)), support))
    evidence = factory(old, new, old_valid & new_valid)
    if evidence is None:
        return Phase13PairResult(sample_id, split, False, "no_fb_dis_evidence", False, False, 0, 0, 0, 0, 0, 0, (), {}, {}), None, None
    guards, counts = _protected_guards(old, new, evidence, old_valid, new_valid)
    alignment = fit_near_protected_alignment(evidence, support=np.asarray(evidence.reliable_mask, bool) & ~guards.protected, plane_verified=False, config=_alignment_config())
    if not alignment.audit.accepted or alignment.matrix is None:
        return Phase13PairResult(sample_id, split, False, alignment.audit.rejection_reason or "alignment_rejected", False, False, 0, 0, **counts, seam_coordinates=(), core_metrics={}, context_metrics={}), None, None
    final_new_full, final_valid_full = _aligned_new_image(new_raw, coarse @ alignment.matrix)
    aligned_new = _central_corridor(final_new_full, settings.corridor_width_px)
    aligned_valid = _central_corridor(final_valid_full.astype(np.uint8), settings.corridor_width_px).astype(bool)
    aligned_guards, counts = _protected_guards(old, aligned_new, evidence, old_valid, aligned_valid)
    aligned_guards, old_anchor_count, new_anchor_count = _fixed_handoff_anchors(
        aligned_guards, old_valid, aligned_valid, settings.handoff_anchor_width_px,
    )
    try:
        graphcut = solve_video_graphcut_seam(old, aligned_new, old_valid, aligned_valid, hard_owner_old=aligned_guards.hard_owner_old, hard_owner_new=aligned_guards.hard_owner_new)
    except RuntimeError:
        return Phase13PairResult(sample_id, split, False, "graphcut_runtime_failure", True, False, 0, 0, **counts, seam_coordinates=(), core_metrics={}, context_metrics={}), None, None
    violation = audit_guard_owner_intersection(graphcut.choose_new, aligned_guards)
    if not graphcut.audit.accepted or violation:
        return Phase13PairResult(sample_id, split, False, graphcut.audit.rejection_reason or "protected_owner_violation", True, False, violation, 0, **counts, seam_coordinates=graphcut.audit.seam_x_by_row, core_metrics={}, context_metrics={}), None, None
    overlap = old_valid & aligned_valid
    old_owner_count = int(np.count_nonzero(overlap & ~graphcut.choose_new))
    new_owner_count = int(np.count_nonzero(overlap & graphcut.choose_new))
    interior_rows = sum(
        int(settings.core_half_width_px <= coordinate <= settings.corridor_width_px - settings.core_half_width_px)
        for coordinate in graphcut.audit.seam_x_by_row
    )
    # A boundary-labelled corridor is a valid GraphCut topology but not an
    # actual two-frame handoff and therefore cannot enter human calibration.
    if old_owner_count == 0 or new_owner_count == 0 or interior_rows != old.shape[0]:
        return Phase13PairResult(
            sample_id, split, False, "not_a_real_two_owner_interior_handoff", True, True, violation, 0,
            **counts, seam_coordinates=graphcut.audit.seam_x_by_row, core_metrics={}, context_metrics={},
            old_owner_pixel_count=old_owner_count, new_owner_pixel_count=new_owner_count,
            interior_seam_row_count=interior_rows,
        ), None, None
    output = old.copy()
    output[graphcut.choose_new] = aligned_new[graphcut.choose_new]
    eligible = build_near_blend_eligible_mask(old_valid, aligned_valid, evidence, aligned_guards)
    output, _band, blend = apply_near_multiband(old, aligned_new, output, graphcut.choose_new, eligible, aligned_guards, config=VideoNearBlendConfig(near_width_px=settings.blend_width_px))
    owner = np.where(graphcut.choose_new, 255, 0).astype(np.uint8)
    core, context = _review_metrics(output, graphcut.audit.seam_x_by_row, settings)
    return Phase13PairResult(sample_id, split, True, None, True, True, violation, blend.guard_intersection_pixel_count, **counts, seam_coordinates=graphcut.audit.seam_x_by_row, core_metrics=core, context_metrics=context, old_owner_pixel_count=old_owner_count, new_owner_pixel_count=new_owner_count, interior_seam_row_count=interior_rows), output, owner


def build_phase13_review_package(
    captures_root: Path, output_root: Path, *, config: Phase13VisualCalibrationConfig | None = None,
    evidence_factory: EvidenceFactory | None = None,
) -> dict[str, object]:
    """Write an intentionally blinded annotation package and private evidence ledger."""
    settings, root = config or Phase13VisualCalibrationConfig(), Path(output_root)
    images, owners, seams = (root / name for name in ("images", "owners", "seams"))
    for directory in (images, owners, seams):
        directory.mkdir(parents=True, exist_ok=True)
    split_lock = frozen_phase13_split(captures_root)
    (root / "split.lock.json").write_text(json.dumps(split_lock, indent=2) + "\n", encoding="utf-8")
    results: list[Phase13PairResult] = []
    for pair in phase13_pairs(captures_root):
        result, crop, owner = render_phase13_pair(pair, config=settings, evidence_factory=evidence_factory)
        results.append(result)
        if result.generated and crop is not None and owner is not None:
            cv2.imwrite(str(images / f"{result.sample_id}.png"), crop)
            cv2.imwrite(str(owners / f"{result.sample_id}.png"), owner)
            (seams / f"{result.sample_id}.json").write_text(json.dumps({"seam_x_by_row": result.seam_coordinates}) + "\n", encoding="utf-8")
    annotations = {"schema": "video-v61-phase13-human-annotation/v2", "instructions": "Review each genuine two-owner, full-resolution crop at 100%; leave values null until human labelled.", "samples": [{"sample_id": r.sample_id, "image": f"images/{r.sample_id}.png", "owner_map": f"owners/{r.sample_id}.png", "seam_coordinates": f"seams/{r.sample_id}.json", "seam_visible": None, "gradient_break_visible": None, "double_edge_or_ghost": None, "line_break": None, "confidence": None} for r in results if r.generated]}
    (root / "annotation_manifest.json").write_text(json.dumps(annotations, indent=2) + "\n", encoding="utf-8")
    ledger = {"schema": "video-v61-phase13-visual-evidence/v2", "scope": "isolated diagnostic POC; not a production or candidate lock", "fixed_config": asdict(settings), "split_lock_sha256": hashlib.sha256((root / "split.lock.json").read_bytes()).hexdigest(), "graphcut_call_count": sum(r.graphcut_called for r in results), "guard_intersection_pixel_count": sum(r.graphcut_guard_intersection_pixels + r.blend_guard_intersection_pixels for r in results), "real_two_owner_sample_count": sum(r.generated for r in results), "results": [asdict(r) for r in results]}
    (root / "evidence_manifest.json").write_text(json.dumps(ledger, indent=2) + "\n", encoding="utf-8")
    return ledger


__all__ = ["Phase13Pair", "Phase13PairResult", "Phase13VisualCalibrationConfig", "build_phase13_review_package", "frozen_phase13_split", "phase13_pairs", "render_phase13_pair"]
