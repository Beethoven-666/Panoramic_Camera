"""In-memory fixed-gate FastSAM RGB-D identity tracking.

FastSAM contributes contour proposals only.  Aligned depth, immutable
camera-to-world poses, measured colour/shape agreement, and adjacent-frame
DIS flow provide identity evidence.  Flow never changes pose, proposal
geometry, depth, or output RGB.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Protocol, Sequence

import cv2
import numpy as np

from .inspection_fastsam_track import (
    FastSAMRGBDCandidate,
    build_fastsam_rgbd_candidate,
    flow_forward_backward_consistency,
    flow_predict_mask,
    polygon_mask,
    select_unambiguous_one_to_one_matches,
)
from .session import CameraIntrinsics


class PolygonProposal(Protocol):
    polygon_xy: np.ndarray


@dataclass(frozen=True)
class FastSAMExactMaskProposal:
    """Memory-bounded proposal retaining exact bbox-local mask topology."""

    polygon_xy: np.ndarray
    bbox_xywh: tuple[int, int, int, int]
    exact_mask_bbox: np.ndarray


@dataclass(frozen=True)
class FastSAMDISFrameInput:
    frame_id: int
    image_bgr: np.ndarray
    depth_mm: np.ndarray
    camera_to_world: np.ndarray
    proposals: Sequence[PolygonProposal | np.ndarray]
    geometric_valid: np.ndarray | None = None


@dataclass(frozen=True)
class FastSAMDISConfig:
    preview_scale: float = 0.25
    minimum_depth_mm: float = 200.0
    maximum_depth_mm: float = 3000.0
    maximum_fb_error_preview_pixels: float = 0.75
    minimum_flow_predicted_mask_iou: float = 0.35
    maximum_lab_delta: float = 30.0
    maximum_world_centroid_delta_mm: float = 80.0
    maximum_area_ratio: float = 2.0
    maximum_log_aspect_delta: float = 0.65
    maximum_solidity_delta: float = 0.30
    maximum_contour_match_i1: float = 0.35
    mutual_best_ambiguity_margin: float = 0.05
    minimum_stable_observations: int = 2


@dataclass(frozen=True)
class FastSAMDISFrameResult:
    source_index: int
    frame_id: int
    candidates: tuple[FastSAMRGBDCandidate, ...]
    identity_masks_preview: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class FastSAMDISTrack:
    track_id: int
    candidate_ids: tuple[int, ...]
    frame_ids: tuple[int, ...]
    observation_count: int
    stable_candidate_ids: tuple[int, ...]
    stable_frame_ids: tuple[int, ...]
    maximum_area_ratio: float
    minimum_flow_mask_iou: float
    maximum_fb_p95_preview_pixels: float
    merge_split_terminated: bool


@dataclass(frozen=True)
class FastSAMDISTrackingResult:
    frames: tuple[FastSAMDISFrameResult, ...]
    tracks: tuple[FastSAMDISTrack, ...]
    stable_tracks: tuple[FastSAMDISTrack, ...]
    pair_audits: tuple[dict[str, object], ...]
    candidate_by_id: dict[int, FastSAMRGBDCandidate]
    flow_role: str = "candidate_identity_evidence_only"
    flow_used_to_warp_rgb_or_position: bool = False


def _proposal_polygon(proposal: PolygonProposal | np.ndarray) -> np.ndarray:
    value = proposal if isinstance(proposal, np.ndarray) else proposal.polygon_xy
    polygon = np.asarray(value, dtype=np.int32)
    if polygon.ndim != 2 or polygon.shape[1] != 2:
        raise ValueError("FastSAM proposal polygon must have Nx2 coordinates")
    return np.ascontiguousarray(polygon)


def _proposal_exact_mask_bbox(
    proposal: PolygonProposal | np.ndarray,
) -> np.ndarray | None:
    value = getattr(proposal, "exact_mask_bbox", None)
    if value is None:
        full = getattr(proposal, "mask", None)
        if full is not None:
            full_value = np.asarray(full, dtype=bool)
            polygon = _proposal_polygon(proposal)
            x, y, width, height = cv2.boundingRect(polygon)
            if (
                full_value.ndim != 2
                or x < 0
                or y < 0
                or x + width > full_value.shape[1]
                or y + height > full_value.shape[0]
            ):
                raise ValueError("FastSAM proposal exact mask is malformed")
            value = full_value[y : y + height, x : x + width]
    return (
        None
        if value is None
        else np.ascontiguousarray(np.asarray(value, dtype=bool))
    )


def _preview_mask(
    candidate: FastSAMRGBDCandidate,
    shape: tuple[int, int],
    scale: float,
) -> np.ndarray:
    full_shape = (
        max(1, int(round(shape[0] / scale))),
        max(1, int(round(shape[1] / scale))),
    )
    full = polygon_mask(candidate, full_shape).astype(np.uint8)
    mask = cv2.resize(full, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return np.ascontiguousarray(mask > 0)


def _mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    intersection = int(np.count_nonzero(first & second))
    union = int(np.count_nonzero(first | second))
    return float(intersection / union) if union else 0.0


def _candidate_pair_audit(
    first: FastSAMRGBDCandidate,
    second: FastSAMRGBDCandidate,
    predicted_mask: np.ndarray,
    second_mask: np.ndarray,
    config: FastSAMDISConfig,
) -> tuple[bool, float, dict[str, object]]:
    iou = _mask_iou(predicted_mask, second_mask)
    lab_delta = float(
        np.linalg.norm(np.asarray(first.median_lab) - np.asarray(second.median_lab))
    )
    centroid_delta = float(
        np.linalg.norm(
            np.asarray(first.world_centroid_mm) - np.asarray(second.world_centroid_mm)
        )
    )
    area_ratio = max(first.source_area_pixels, second.source_area_pixels) / max(
        1, min(first.source_area_pixels, second.source_area_pixels)
    )
    aspect_delta = abs(
        math.log(max(1e-6, first.aspect_ratio) / max(1e-6, second.aspect_ratio))
    )
    solidity_delta = abs(first.solidity - second.solidity)
    contour_delta = float(
        cv2.matchShapes(
            first.polygon_xy,
            second.polygon_xy,
            cv2.CONTOURS_MATCH_I1,
            0.0,
        )
    )
    accepted = bool(
        iou >= config.minimum_flow_predicted_mask_iou
        and lab_delta <= config.maximum_lab_delta
        and centroid_delta <= config.maximum_world_centroid_delta_mm
        and area_ratio <= config.maximum_area_ratio
        and aspect_delta <= config.maximum_log_aspect_delta
        and solidity_delta <= config.maximum_solidity_delta
        and contour_delta <= config.maximum_contour_match_i1
    )
    score = iou - 0.01 * lab_delta - 0.0025 * centroid_delta - 0.20 * contour_delta
    return accepted, score, {
        "flow_predicted_mask_iou": iou,
        "median_lab_delta": lab_delta,
        "world_centroid_delta_mm": centroid_delta,
        "source_area_ratio": area_ratio,
        "log_aspect_delta": aspect_delta,
        "solidity_delta": solidity_delta,
        "contour_match_i1": contour_delta,
        "score": score,
        "pass": accepted,
    }


def track_fastsam_dis_frames(
    frames: Iterable[FastSAMDISFrameInput],
    *,
    intrinsics: CameraIntrinsics,
    reference_depth_mm: float,
    stable_frame_ids: Sequence[int] | None = None,
    config: FastSAMDISConfig | None = None,
) -> FastSAMDISTrackingResult:
    """Build fixed-gate adjacent-frame tracks entirely from in-memory inputs."""
    settings = config or FastSAMDISConfig()
    requested_stable_ids = (
        {int(value) for value in stable_frame_ids}
        if stable_frame_ids is not None
        else None
    )
    preview_size = (
        int(round(intrinsics.width * settings.preview_scale)),
        int(round(intrinsics.height * settings.preview_scale)),
    )
    grays: list[np.ndarray] = []
    frame_results: list[FastSAMDISFrameResult] = []
    candidates: list[FastSAMRGBDCandidate] = []
    for source_index, frame in enumerate(frames):
        image = np.asarray(frame.image_bgr, dtype=np.uint8)
        depth = np.asarray(frame.depth_mm, dtype=np.float32)
        if image.shape != (intrinsics.height, intrinsics.width, 3):
            raise ValueError(f"frame {frame.frame_id} RGB dimensions do not match intrinsics")
        if depth.shape != image.shape[:2]:
            raise ValueError(f"frame {frame.frame_id} depth dimensions do not match RGB")
        geometric_valid = (
            np.ones(depth.shape, dtype=bool)
            if frame.geometric_valid is None
            else np.asarray(frame.geometric_valid, dtype=bool)
        )
        if geometric_valid.shape != depth.shape:
            raise ValueError(f"frame {frame.frame_id} geometric-valid mask is invalid")
        reliable = (
            geometric_valid
            & np.isfinite(depth)
            & (depth >= settings.minimum_depth_mm)
            & (depth <= settings.maximum_depth_mm)
        )
        gray = cv2.resize(
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
            preview_size,
            interpolation=cv2.INTER_AREA,
        )
        grays.append(np.ascontiguousarray(gray))
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        frame_candidates: list[FastSAMRGBDCandidate] = []
        for proposal in frame.proposals:
            candidate = build_fastsam_rgbd_candidate(
                candidate_id=len(candidates),
                source_index=source_index,
                frame_id=int(frame.frame_id),
                polygon_xy=_proposal_polygon(proposal),
                exact_mask_bbox=_proposal_exact_mask_bbox(proposal),
                image_bgr=image,
                lab_image=lab,
                depth_mm=depth,
                reliable_depth=reliable,
                camera_to_world=np.asarray(frame.camera_to_world, dtype=np.float64),
                intrinsics=intrinsics,
                reference_depth_mm=float(reference_depth_mm),
            )
            if candidate is not None:
                candidates.append(candidate)
                frame_candidates.append(candidate)
        masks = tuple(
            _preview_mask(candidate, gray.shape, settings.preview_scale)
            for candidate in frame_candidates
        )
        frame_results.append(
            FastSAMDISFrameResult(
                source_index=source_index,
                frame_id=int(frame.frame_id),
                candidates=tuple(frame_candidates),
                identity_masks_preview=masks,
            )
        )
    if len(frame_results) < 2:
        raise ValueError("FastSAM DIS tracking requires at least two frames")
    stable_ids = (
        requested_stable_ids
        if requested_stable_ids is not None
        else {frame.frame_id for frame in frame_results}
    )

    outgoing: dict[int, int] = {}
    incoming: dict[int, int] = {}
    edge_audit: dict[tuple[int, int], dict[str, object]] = {}
    pair_audits: list[dict[str, object]] = []
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    for pair_index in range(len(frame_results) - 1):
        first_frame = frame_results[pair_index]
        second_frame = frame_results[pair_index + 1]
        first_candidates = first_frame.candidates
        second_candidates = second_frame.candidates
        row: dict[str, object] = {
            "first_frame_id": first_frame.frame_id,
            "second_frame_id": second_frame.frame_id,
            "first_candidate_count": len(first_candidates),
            "second_candidate_count": len(second_candidates),
            "valid_edge_count": 0,
            "one_to_one_match_count": 0,
            "fb_rejected_source_count": 0,
        }
        if not first_candidates or not second_candidates:
            pair_audits.append(row)
            continue
        forward = dis.calc(grays[pair_index], grays[pair_index + 1], None)
        backward = dis.calc(grays[pair_index + 1], grays[pair_index], None)
        valid = np.zeros((len(first_candidates), len(second_candidates)), dtype=bool)
        score = np.full(valid.shape, -np.inf, dtype=np.float64)
        metrics: dict[tuple[int, int], dict[str, object]] = {}
        for first_index, (first_candidate, first_mask) in enumerate(
            zip(first_candidates, first_frame.identity_masks_preview, strict=True)
        ):
            fb = flow_forward_backward_consistency(
                first_mask,
                forward,
                backward,
                maximum_error_pixels=settings.maximum_fb_error_preview_pixels,
            )
            if not fb["pass"]:
                row["fb_rejected_source_count"] = int(row["fb_rejected_source_count"]) + 1
                continue
            predicted = flow_predict_mask(first_mask, backward)
            for second_index, (second_candidate, second_mask) in enumerate(
                zip(second_candidates, second_frame.identity_masks_preview, strict=True)
            ):
                accepted, value, audit = _candidate_pair_audit(
                    first_candidate,
                    second_candidate,
                    predicted,
                    second_mask,
                    settings,
                )
                if accepted:
                    valid[first_index, second_index] = True
                    score[first_index, second_index] = value
                    metrics[(first_index, second_index)] = {
                        **audit,
                        "flow_forward_backward": fb,
                    }
        row["valid_edge_count"] = int(np.count_nonzero(valid))
        matches = select_unambiguous_one_to_one_matches(
            valid,
            score,
            ambiguity_margin=settings.mutual_best_ambiguity_margin,
        )
        row["one_to_one_match_count"] = len(matches)
        for first_index, second_index in matches:
            first_id = first_candidates[first_index].candidate_id
            second_id = second_candidates[second_index].candidate_id
            outgoing[first_id] = second_id
            incoming[second_id] = first_id
            edge_audit[(first_id, second_id)] = metrics[(first_index, second_index)]
        pair_audits.append(row)

    candidate_by_id = {candidate.candidate_id: candidate for candidate in candidates}
    tracks: list[FastSAMDISTrack] = []
    visited: set[int] = set()
    for candidate in candidates:
        candidate_id = candidate.candidate_id
        if candidate_id in incoming or candidate_id in visited:
            continue
        sequence: list[int] = []
        edge_rows: list[dict[str, object]] = []
        current = candidate_id
        while current not in visited:
            visited.add(current)
            sequence.append(current)
            next_id = outgoing.get(current)
            if next_id is None:
                break
            edge_rows.append(edge_audit[(current, next_id)])
            current = next_id
        if len(sequence) < 2:
            continue
        stable_candidates = tuple(
            value for value in sequence if candidate_by_id[value].frame_id in stable_ids
        )
        tracks.append(
            FastSAMDISTrack(
                track_id=len(tracks),
                candidate_ids=tuple(sequence),
                frame_ids=tuple(candidate_by_id[value].frame_id for value in sequence),
                observation_count=len(sequence),
                stable_candidate_ids=stable_candidates,
                stable_frame_ids=tuple(
                    candidate_by_id[value].frame_id for value in stable_candidates
                ),
                maximum_area_ratio=max(float(row["source_area_ratio"]) for row in edge_rows),
                minimum_flow_mask_iou=min(
                    float(row["flow_predicted_mask_iou"]) for row in edge_rows
                ),
                maximum_fb_p95_preview_pixels=max(
                    float(row["flow_forward_backward"]["p95_error_pixels"])
                    for row in edge_rows
                ),
                merge_split_terminated=bool(
                    sequence[-1] not in outgoing
                    and candidate_by_id[sequence[-1]].source_index < len(frame_results) - 1
                ),
            )
        )
    stable_tracks = sorted(
        (
            track
            for track in tracks
            if len(track.stable_candidate_ids) >= settings.minimum_stable_observations
        ),
        key=lambda track: (
            len(track.stable_candidate_ids),
            track.observation_count,
            -track.maximum_area_ratio,
        ),
        reverse=True,
    )
    return FastSAMDISTrackingResult(
        frames=tuple(frame_results),
        tracks=tuple(tracks),
        stable_tracks=tuple(stable_tracks),
        pair_audits=tuple(pair_audits),
        candidate_by_id=candidate_by_id,
    )


__all__ = [
    "FastSAMDISConfig",
    "FastSAMDISFrameInput",
    "FastSAMDISFrameResult",
    "FastSAMDISTrack",
    "FastSAMDISTrackingResult",
    "track_fastsam_dis_frames",
]
