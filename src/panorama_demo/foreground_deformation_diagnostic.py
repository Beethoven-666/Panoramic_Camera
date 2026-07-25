"""Independent full-chain diagnostic for experimental foreground deformation.

This command deliberately does not add a renderer option to ``g305-panorama``.
It asks the ordinary calibrated renderer for a complete, current RGB-D/ORB
source chain, captures a read-only adjacent-pair analysis callback, and
publishes only the resulting before/after pair view plus a scalar audit.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Mapping, Sequence

import cv2
import numpy as np

from . import stitch_sequence
from .calibrated_rgb_pushbroom import (
    CalibratedRGBPushbroomConfig,
    CalibratedRGBPushbroomResult,
    render_calibrated_rgb_pushbroom,
)
from .foreground_deformation import (
    ForegroundDeformationExperimentConfig,
    ForegroundTrackEvidence,
    attempt_foreground_deformation,
)
from .foreground_segments import ForegroundFragment, GeometryMode, SegmentOwnerPlan
from .session import CameraIntrinsics, RGBDFrame


def _nonnegative_pair_index(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("pair index must be an integer") from exc
    if result < 0:
        raise argparse.ArgumentTypeError("pair index must be non-negative")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a diagnostic-only adjacent-pair A/B view or an experimental "
            "foreground-deformation panorama from a strict Gemini 305 RGB-D session"
        )
    )
    parser.add_argument("input", type=Path, help="Calibrated RGB-D capture session")
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/foreground_deformation_diagnostic")
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--pair-index",
        type=_nonnegative_pair_index,
        default=48,
        help=(
            "Zero-based adjacent pair position in the complete current render "
            "chain (default: 48, inspecting nodes 48 and 49)"
        ),
    )
    parser.add_argument(
        "--whole-panorama",
        action="store_true",
        help=(
            "Run the same experimental gate on every adjacent pair and publish one "
            "diagnostic-only full panorama. --pair-index is ignored in this mode."
        ),
    )
    return parser


def _scalar_tree(value: object) -> object:
    """Fail closed if a diagnostic report attempts to retain dense evidence."""

    if value is None or isinstance(value, (bool, str, int, np.integer)):
        return value if not isinstance(value, np.integer) else int(value)
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        if not math.isfinite(numeric):
            raise RuntimeError("foreground deformation audit contains a non-finite scalar")
        return numeric
    if isinstance(value, np.ndarray):
        raise RuntimeError("foreground deformation audit attempted to publish dense data")
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key, nested in value.items():
            if not isinstance(key, str):
                raise RuntimeError("foreground deformation audit keys must be strings")
            lowered = key.lower()
            if (
                lowered.endswith("_mask")
                or lowered.endswith("_map")
                or lowered.endswith("_path")
                or lowered in {"image", "rgb", "depth", "flow"}
            ):
                raise RuntimeError("foreground deformation audit attempted to publish dense data")
            result[key] = _scalar_tree(nested)
        return result
    if isinstance(value, (tuple, list)):
        return [_scalar_tree(item) for item in value]
    raise RuntimeError("foreground deformation audit contains a non-scalar value")


def _native_pair_residual(
    first: np.ndarray, second: np.ndarray, valid: np.ndarray
) -> dict[str, object]:
    """Measure an original-resolution RGB edge displacement without a warp."""

    gray0 = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.cvtColor(second, cv2.COLOR_BGR2GRAY)
    forward = cv2.calcOpticalFlowFarneback(
        gray0,
        gray1,
        None,
        0.5,
        3,
        21,
        5,
        7,
        1.5,
        0,
    )
    gx = cv2.Sobel(gray0, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray0, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.hypot(gx, gy)
    values = magnitude[np.asarray(valid, dtype=bool)]
    if not values.size:
        return {
            "strong_edge_count": 0,
            "edge_offset_p95_pixels": None,
            "edge_offset_maximum_pixels": None,
        }
    threshold = max(12.0, float(np.percentile(values, 60.0)))
    edge = np.asarray(valid, dtype=bool) & (magnitude >= threshold)
    offsets = np.hypot(forward[:, :, 0], forward[:, :, 1])[edge]
    return {
        "strong_edge_count": int(offsets.size),
        "edge_offset_p95_pixels": (
            None if not offsets.size else float(np.percentile(offsets, 95.0))
        ),
        "edge_offset_maximum_pixels": (
            None if not offsets.size else float(np.max(offsets))
        ),
    }


def _corridor_bounds(
    *,
    overlap_x: tuple[int, int],
    nominal_owner_boundary_x: float,
    requested_width: int,
) -> tuple[int, int]:
    left, right = (int(value) for value in overlap_x)
    if right - left < 96:
        raise RuntimeError(
            "Foreground deformation diagnostic requires a 96-160 px adjacent corridor"
        )
    width = min(int(requested_width), right - left)
    if not 96 <= width <= 160:
        raise RuntimeError("Foreground deformation diagnostic corridor escaped [96, 160]")
    initial_left = int(round(float(nominal_owner_boundary_x) - width / 2.0))
    x0 = min(max(initial_left, left), right - width)
    return int(x0), int(x0 + width)


def _fragment_mask_in_corridor(
    fragment: ForegroundFragment,
    *,
    corridor_x: tuple[int, int],
    image_height: int,
) -> np.ndarray:
    x0, x1 = corridor_x
    result = np.zeros((image_height, x1 - x0), dtype=bool)
    fragment_x, fragment_y, fragment_width, fragment_height = fragment.global_bbox
    common_left = max(int(fragment_x), x0)
    common_right = min(int(fragment_x + fragment_width), x1)
    common_top = max(int(fragment_y), 0)
    common_bottom = min(int(fragment_y + fragment_height), image_height)
    if common_right <= common_left or common_bottom <= common_top:
        return result
    source_x = slice(common_left - int(fragment_x), common_right - int(fragment_x))
    source_y = slice(common_top - int(fragment_y), common_bottom - int(fragment_y))
    result[
        common_top:common_bottom,
        common_left - x0 : common_right - x0,
    ] = np.asarray(fragment.local_mask, dtype=bool)[source_y, source_x]
    return result


def _track_evidence(
    fragment: ForegroundFragment,
    plan: SegmentOwnerPlan,
) -> ForegroundTrackEvidence | None:
    track_id = plan.fragment_track_ids.get(fragment.reference)
    if track_id is None:
        return None
    track = next((row for row in plan.tracks if row.track_id == track_id), None)
    if track is None:
        return None
    # ``plan_foreground_owners`` has already rejected all split/merge candidate
    # graphs before making a track.  Keep the remaining object-specific gates
    # conservative: a natural break remains an endpoint/joint/occlusion veto.
    complete = set(fragment.allowed_local_owners) == {0, 1}
    reciprocal = bool(fragment.bidirectional_visibility_supported)
    direct = bool(track.direct_token_edge_count and track.bidirectional_edge_count)
    no_break = fragment.natural_break_reason is None
    return ForegroundTrackEvidence(
        track_id=int(track.track_id),
        association_score=float(track.association_score),
        one_to_one=True,
        no_split_merge=True,
        complete_source_coverage=complete,
        bidirectional_visibility=reciprocal,
        contour_correspondence=direct,
        centreline_correspondence=direct,
        no_real_joint=no_break,
        no_object_endpoint=no_break,
        no_occlusion_or_disocclusion=(
            reciprocal and fragment.geometry_mode is GeometryMode.DEPTH_OBSERVED
        ),
        native_resolution=True,
    )


class _ForegroundDeformationCollector:
    """Read-only callback passed to the calibrated renderer's diagnostic seam."""

    def __init__(
        self,
        *,
        pair_index: int,
        config: ForegroundDeformationExperimentConfig,
    ) -> None:
        self.pair_index = int(pair_index)
        self.config = config
        self.candidate_bgr: np.ndarray | None = None
        self.candidate_mask: np.ndarray | None = None
        self.metadata: dict[str, object] | None = None

    def __call__(self, **context: object) -> None:
        pair_index = int(context["pair_index"])
        if pair_index != self.pair_index:
            raise RuntimeError("foreground deformation callback received the wrong pair")
        first = np.asarray(context["first_bgr"], dtype=np.uint8)
        second = np.asarray(context["second_bgr"], dtype=np.uint8)
        first_valid = np.asarray(context["first_valid"], dtype=bool)
        second_valid = np.asarray(context["second_valid"], dtype=bool)
        if (
            first.ndim != 3
            or first.shape != second.shape
            or first.shape[2] != 3
            or first_valid.shape != first.shape[:2]
            or second_valid.shape != first.shape[:2]
        ):
            raise RuntimeError("foreground deformation callback pair evidence is malformed")
        overlap_x = tuple(int(value) for value in context["overlap_x"])
        if len(overlap_x) != 2:
            raise RuntimeError("foreground deformation callback overlap is malformed")
        corridor_x = _corridor_bounds(
            overlap_x=(overlap_x[0], overlap_x[1]),
            nominal_owner_boundary_x=float(context["nominal_owner_boundary_x"]),
            requested_width=int(self.config.analysis_corridor_width_pixels),
        )
        local_left = corridor_x[0] - overlap_x[0]
        local_right = corridor_x[1] - overlap_x[0]
        first_crop = first[:, local_left:local_right].copy()
        second_crop = second[:, local_left:local_right].copy()
        valid0 = first_valid[:, local_left:local_right].copy()
        valid1 = second_valid[:, local_left:local_right].copy()
        common = valid0 & valid1
        residual = _native_pair_residual(first_crop, second_crop, common)
        boundary = int(
            np.clip(
                round(float(context["nominal_owner_boundary_x"])) - corridor_x[0],
                0,
                first_crop.shape[1] - 1,
            )
        )
        # The actual before panel is taken later from the completed formal
        # renderer crop.  Keep only a sparse foreground replacement here so
        # the formal GraphCut/same-layer mesh/MultiBand background is retained
        # exactly in both panels.
        candidate_bgr = np.zeros_like(first_crop)
        candidate_mask = np.zeros(first_crop.shape[:2], dtype=bool)
        fragments = tuple(context["foreground_fragments"])
        if not all(isinstance(fragment, ForegroundFragment) for fragment in fragments):
            raise RuntimeError("foreground deformation callback fragments are malformed")
        plan = context["foreground_owner_plan"]
        if not isinstance(plan, SegmentOwnerPlan):
            raise RuntimeError("foreground deformation callback owner plan is malformed")
        fragment_masks = {
            fragment.reference: _fragment_mask_in_corridor(
                fragment,
                corridor_x=corridor_x,
                image_height=first_crop.shape[0],
            )
            for fragment in fragments
        }
        audits: list[dict[str, object]] = []
        accepted_count = 0
        active_pixel_count = 0
        for fragment in fragments:
            evidence = _track_evidence(fragment, plan)
            instance_mask = fragment_masks[fragment.reference]
            if evidence is None:
                audits.append(
                    {
                        "policy": "foreground_local_inverse_mesh_diagnostic_v1",
                        "candidate": False,
                        "accepted": False,
                        "reason": "no_high_confidence_non_split_merge_foreground_track",
                        "pair_index": pair_index,
                        "frame_ids": [int(value) for value in context["frame_ids"]],
                        "foreground_instance_pixel_count": int(np.count_nonzero(instance_mask)),
                        "native_pair_residual": residual,
                    }
                )
                continue
            assignment = plan.owner_for_fragment(*fragment.reference)
            if assignment is None:
                audits.append(
                    {
                        "policy": "foreground_local_inverse_mesh_diagnostic_v1",
                        "candidate": False,
                        "accepted": False,
                        "reason": "foreground_track_has_no_complete_owner_run",
                        "pair_index": pair_index,
                        "frame_ids": [int(value) for value in context["frame_ids"]],
                        "foreground_instance_pixel_count": int(
                            np.count_nonzero(instance_mask)
                        ),
                    }
                )
                continue
            source_indices = tuple(int(value) for value in context["source_indices"])
            desired_source = int(assignment["source_index"])
            if desired_source != source_indices[1]:
                # The current implementation is deliberately one-directional:
                # source 1 may be inverse-sampled into the first source's
                # coordinate system.  Forcing source 1 when the complete
                # foreground run prefers source 0 would violate continuity,
                # so it remains a hard-owner no-op rather than swapping roles.
                audits.append(
                    {
                        "policy": "foreground_local_inverse_mesh_diagnostic_v1",
                        "candidate": False,
                        "accepted": False,
                        "reason": "foreground_continuity_prefers_reference_source",
                        "pair_index": pair_index,
                        "frame_ids": [int(value) for value in context["frame_ids"]],
                        "source_indices": list(source_indices),
                        "planned_owner_source_index": desired_source,
                        "foreground_instance_pixel_count": int(
                            np.count_nonzero(instance_mask)
                        ),
                    }
                )
                continue
            complete_coverage = instance_mask & ~(valid0 & valid1)
            if np.any(complete_coverage):
                audits.append(
                    {
                        "policy": "foreground_local_inverse_mesh_diagnostic_v1",
                        "candidate": False,
                        "accepted": False,
                        "reason": "incomplete_dual_source_foreground_coverage",
                        "pair_index": pair_index,
                        "frame_ids": [int(value) for value in context["frame_ids"]],
                        "source_indices": list(source_indices),
                        "foreground_instance_pixel_count": int(
                            np.count_nonzero(instance_mask)
                        ),
                        "uncovered_foreground_pixel_count": int(
                            np.count_nonzero(complete_coverage)
                        ),
                    }
                )
                continue
            protected = np.zeros_like(instance_mask)
            for reference, other_mask in fragment_masks.items():
                if reference != fragment.reference:
                    protected |= other_mask
            result = attempt_foreground_deformation(
                first_crop,
                second_crop,
                instance_mask,
                evidence,
                config=self.config,
                reference_valid_mask=valid0,
                source_valid_mask=valid1,
                source_foreground_mask=instance_mask,
                protected_mask=protected,
                owner_boundary_x=float(boundary),
            )
            audit = dict(result.as_dict())
            audit.update(
                {
                    "pair_index": pair_index,
                    "frame_ids": [int(value) for value in context["frame_ids"]],
                    "source_indices": [
                        int(value) for value in context["source_indices"]
                    ],
                    "corridor_x": [int(value) for value in corridor_x],
                    "geometry_triggered": bool(context["geometry_triggered"]),
                    "native_pair_residual": residual,
                }
            )
            if result.accepted:
                # Foreground continuity comes first: the completed baseline
                # already holds this run from source 1; candidate output only
                # replaces those same pixels with one inverse-sampled RGB
                # owner.  There is no alpha, MultiBand, APAP, or nonadjacent
                # source.
                candidate_bgr[instance_mask] = result.warped_source_bgr[
                    instance_mask
                ]
                candidate_mask |= instance_mask
                accepted_count += 1
                active_pixel_count += int(np.count_nonzero(result.active_mask))
            audits.append(audit)
        if not audits:
            p95 = residual["edge_offset_p95_pixels"]
            maximum = residual["edge_offset_maximum_pixels"]
            no_deformation_needed = (
                (p95 is None or float(p95) <= 0.75)
                and (maximum is None or float(maximum) <= 2.0)
            )
            audits.append(
                {
                    "policy": "foreground_local_inverse_mesh_diagnostic_v1",
                    "candidate": False,
                    "accepted": False,
                    "reason": (
                        "no_measurable_foreground_seam_residual"
                        if no_deformation_needed
                        else "no_high_confidence_non_split_merge_foreground_track"
                    ),
                    "pair_index": pair_index,
                    "frame_ids": [int(value) for value in context["frame_ids"]],
                    "corridor_x": [int(value) for value in corridor_x],
                    "native_pair_residual": residual,
                }
            )
        self.candidate_bgr = candidate_bgr
        self.candidate_mask = candidate_mask
        self.metadata = {
            "pair_index": pair_index,
            "frame_ids": [int(value) for value in context["frame_ids"]],
            "source_indices": [int(value) for value in context["source_indices"]],
            "corridor_x": [int(value) for value in corridor_x],
            "corridor_width_pixels": int(corridor_x[1] - corridor_x[0]),
            "nominal_owner_boundary_x": float(context["nominal_owner_boundary_x"]),
            "foreground_deformation_audits": audits,
            "accepted_foreground_instance_count": accepted_count,
            "foreground_deformation_pixel_count": active_pixel_count,
            "owner_policy": "foreground_continuity_single_source_hard_owner_only",
            "graphcut_crosses_accepted_foreground_instance_count": 0,
            "alpha_blend_pixel_count": 0,
            "multiband_pixel_count": 0,
            "nonadjacent_owner_pixel_count": 0,
            "two_source_foreground_handoff_pixel_count": 0,
            "pose_rewrite_detected": False,
            "color_generation_detected": False,
            "global_flow_or_apap_used": False,
        }

    def panels_from_baseline(
        self, baseline: CalibratedRGBPushbroomResult
    ) -> tuple[np.ndarray, dict[str, object]]:
        """Make a raw A/B pair crop while retaining the formal background."""

        if (
            self.candidate_bgr is None
            or self.candidate_mask is None
            or self.metadata is None
        ):
            raise RuntimeError("foreground deformation callback did not produce a panel")
        crop = baseline.metadata.get("crop")
        if not isinstance(crop, Mapping):
            raise RuntimeError("foreground deformation baseline omitted its crop map")
        try:
            crop_x = int(crop["x"])
            crop_y = int(crop["y"])
            crop_width = int(crop["width"])
            crop_height = int(crop["height"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("foreground deformation baseline crop is malformed") from exc
        panorama = np.asarray(baseline.panorama, dtype=np.uint8)
        if panorama.shape != (crop_height, crop_width, 3):
            raise RuntimeError("foreground deformation baseline crop/image mismatch")
        corridor = self.metadata.get("corridor_x")
        if not (
            isinstance(corridor, list)
            and len(corridor) == 2
            and all(isinstance(value, int) for value in corridor)
        ):
            raise RuntimeError("foreground deformation callback corridor is malformed")
        corridor_left, corridor_right = (int(value) for value in corridor)
        x0 = max(corridor_left, crop_x)
        x1 = min(corridor_right, crop_x + crop_width)
        if x1 - x0 < 96:
            raise RuntimeError(
                "foreground deformation final crop does not retain its 96 px corridor"
            )
        source_y0 = crop_y
        source_y1 = crop_y + crop_height
        if source_y0 < 0 or source_y1 > self.candidate_mask.shape[0]:
            raise RuntimeError("foreground deformation crop escaped pair canvas rows")
        local_x0 = x0 - corridor_left
        local_x1 = x1 - corridor_left
        before = np.ascontiguousarray(
            panorama[:, x0 - crop_x : x1 - crop_x].copy()
        )
        after = before.copy()
        candidate_mask = self.candidate_mask[source_y0:source_y1, local_x0:local_x1]
        candidate_bgr = self.candidate_bgr[source_y0:source_y1, local_x0:local_x1]
        if candidate_mask.shape != before.shape[:2] or candidate_bgr.shape != before.shape:
            raise RuntimeError("foreground deformation candidate crop shape mismatch")
        after[candidate_mask] = candidate_bgr[candidate_mask]
        metadata = dict(self.metadata)
        metadata.update(
            {
                "panel_mapping": {
                    "before": {"columns": [0, int(before.shape[1])]},
                    "after": {
                        "columns": [
                            int(before.shape[1]),
                            int(before.shape[1] * 2),
                        ]
                    },
                },
                "baseline_graphcut_background_retained": True,
                "baseline_crop": {
                    "x": crop_x,
                    "y": crop_y,
                    "width": crop_width,
                    "height": crop_height,
                },
                "rendered_corridor_x": [int(x0), int(x1)],
            }
        )
        return np.ascontiguousarray(np.hstack((before, after))), metadata


class _ForegroundDeformationPanoramaCollector:
    """Collect every independent pair audit and compose accepted RGB replacements.

    This is intentionally a diagnostic-only post-composite: the ordinary renderer
    completes its full hard-owner/GraphCut/MultiBand baseline first.  Only an
    accepted foreground instance may then replace those same pixels with one
    adjacent, inverse-sampled RGB source.  The baseline renderer never consumes
    a result from this object.
    """

    def __init__(
        self,
        *,
        pair_indices: Sequence[int],
        config: ForegroundDeformationExperimentConfig,
    ) -> None:
        normalized = tuple(sorted({int(value) for value in pair_indices}))
        if not normalized:
            raise ValueError("foreground deformation panorama collector needs pairs")
        self._collectors = {
            pair_index: _ForegroundDeformationCollector(
                pair_index=pair_index,
                config=config,
            )
            for pair_index in normalized
        }

    def __call__(self, **context: object) -> None:
        pair_index = int(context["pair_index"])
        collector = self._collectors.get(pair_index)
        if collector is None:
            raise RuntimeError("foreground deformation panorama received an unknown pair")
        collector(**context)

    @staticmethod
    def _mark_final_application(
        metadata: dict[str, object],
        *,
        applied: bool,
        reason: str,
    ) -> None:
        audits = metadata.get("foreground_deformation_audits")
        if not isinstance(audits, list):
            raise RuntimeError("foreground deformation panorama collector omitted audits")
        for audit in audits:
            if not isinstance(audit, dict):
                raise RuntimeError("foreground deformation panorama audit is malformed")
            if bool(audit.get("accepted", False)):
                audit["final_composite_applied"] = bool(applied)
                audit["final_composite_reason"] = reason

    def panorama_from_baseline(
        self, baseline: CalibratedRGBPushbroomResult
    ) -> tuple[np.ndarray, dict[str, object]]:
        """Overlay only non-overlapping accepted foreground source samples."""

        crop = baseline.metadata.get("crop")
        if not isinstance(crop, Mapping):
            raise RuntimeError("foreground deformation baseline omitted its crop map")
        try:
            crop_x = int(crop["x"])
            crop_y = int(crop["y"])
            crop_width = int(crop["width"])
            crop_height = int(crop["height"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("foreground deformation baseline crop is malformed") from exc
        baseline_panorama = np.asarray(baseline.panorama, dtype=np.uint8)
        if baseline_panorama.shape != (crop_height, crop_width, 3):
            raise RuntimeError("foreground deformation baseline crop/image mismatch")

        panorama = np.ascontiguousarray(baseline_panorama.copy())
        coverage = np.zeros(panorama.shape[:2], dtype=np.uint16)
        proposals: list[dict[str, object]] = []
        per_pair: list[dict[str, object]] = []
        all_audits: list[dict[str, object]] = []
        candidate_instance_count = 0

        for pair_index, collector in self._collectors.items():
            if (
                collector.candidate_bgr is None
                or collector.candidate_mask is None
                or collector.metadata is None
            ):
                raise RuntimeError(
                    "foreground deformation panorama callback did not produce every pair"
                )
            metadata = collector.metadata
            audits = metadata.get("foreground_deformation_audits")
            if not isinstance(audits, list) or not all(
                isinstance(audit, dict) for audit in audits
            ):
                raise RuntimeError("foreground deformation panorama audit is malformed")
            all_audits.extend(audits)
            candidate_instance_count += int(
                metadata.get("accepted_foreground_instance_count", 0)
            )

            corridor = metadata.get("corridor_x")
            if not (
                isinstance(corridor, list)
                and len(corridor) == 2
                and all(isinstance(value, int) for value in corridor)
            ):
                raise RuntimeError("foreground deformation callback corridor is malformed")
            corridor_left, corridor_right = (int(value) for value in corridor)
            x0 = max(corridor_left, crop_x)
            x1 = min(corridor_right, crop_x + crop_width)
            source_y0 = crop_y
            source_y1 = crop_y + crop_height
            if source_y0 < 0 or source_y1 > collector.candidate_mask.shape[0]:
                raise RuntimeError("foreground deformation crop escaped pair canvas rows")
            if x1 <= x0:
                self._mark_final_application(
                    metadata,
                    applied=False,
                    reason="final_panorama_crop_excludes_candidate",
                )
                per_pair.append(dict(metadata))
                continue
            local_x0 = x0 - corridor_left
            local_x1 = x1 - corridor_left
            candidate_mask = collector.candidate_mask[
                source_y0:source_y1, local_x0:local_x1
            ]
            candidate_bgr = collector.candidate_bgr[
                source_y0:source_y1, local_x0:local_x1
            ]
            if (
                candidate_mask.shape != (crop_height, x1 - x0)
                or candidate_bgr.shape != (crop_height, x1 - x0, 3)
            ):
                raise RuntimeError("foreground deformation candidate crop shape mismatch")
            output_x0 = x0 - crop_x
            output_x1 = x1 - crop_x
            if np.any(candidate_mask):
                coverage_view = coverage[:, output_x0:output_x1]
                coverage_view[candidate_mask] += 1
                proposals.append(
                    {
                        "pair_index": pair_index,
                        "metadata": metadata,
                        "mask": candidate_mask,
                        "bgr": candidate_bgr,
                        "x0": output_x0,
                        "x1": output_x1,
                    }
                )
            per_pair.append(dict(metadata))

        applied_instance_count = 0
        applied_active_pixel_count = 0
        overlap_rejected_instance_count = 0
        for proposal in proposals:
            x0 = int(proposal["x0"])
            x1 = int(proposal["x1"])
            mask = np.asarray(proposal["mask"], dtype=bool)
            bgr = np.asarray(proposal["bgr"], dtype=np.uint8)
            metadata = proposal["metadata"]
            if not isinstance(metadata, dict):
                raise RuntimeError("foreground deformation panorama proposal is malformed")
            if np.any(coverage[:, x0:x1][mask] > 1):
                self._mark_final_application(
                    metadata,
                    applied=False,
                    reason="overlapping_accepted_foreground_deformation_candidates",
                )
                overlap_rejected_instance_count += int(
                    metadata.get("accepted_foreground_instance_count", 0)
                )
                continue
            destination = panorama[:, x0:x1]
            destination[mask] = bgr[mask]
            self._mark_final_application(
                metadata,
                applied=True,
                reason="accepted_single_source_foreground_inverse_mesh",
            )
            applied_instance_count += int(
                metadata.get("accepted_foreground_instance_count", 0)
            )
            applied_active_pixel_count += int(
                metadata.get("foreground_deformation_pixel_count", 0)
            )

        metadata = {
            "mode": "full_panorama_experimental_foreground_deformation",
            "pair_count": len(self._collectors),
            "pair_indices": [int(value) for value in self._collectors],
            "pair_diagnostics": per_pair,
            "foreground_deformation_audits": all_audits,
            "mesh_accepted_foreground_instance_count": candidate_instance_count,
            "applied_foreground_instance_count": applied_instance_count,
            "overlap_rejected_foreground_instance_count": overlap_rejected_instance_count,
            "foreground_deformation_pixel_count": applied_active_pixel_count,
            "baseline_graphcut_background_retained": True,
            "owner_policy": "foreground_continuity_single_source_hard_owner_only",
            "graphcut_crosses_accepted_foreground_instance_count": 0,
            "alpha_blend_pixel_count": 0,
            "multiband_pixel_count": 0,
            "nonadjacent_owner_pixel_count": 0,
            "two_source_foreground_handoff_pixel_count": 0,
            "pose_rewrite_detected": False,
            "color_generation_detected": False,
            "global_flow_or_apap_used": False,
        }
        return panorama, metadata


def render_foreground_deformation_pair_diagnostic(
    frames: Sequence[RGBDFrame],
    poses: Sequence[np.ndarray],
    calibration: CameraIntrinsics,
    *,
    pair_index: int,
    experiment_config: ForegroundDeformationExperimentConfig,
    config: Mapping[str, object] | CalibratedRGBPushbroomConfig | None = None,
    rgb_motions: Sequence[object] | None = None,
    motion_pixels_to_full_resolution: float = 1.0,
    multiband_levels: int = 3,
    quality_gate: bool = True,
) -> CalibratedRGBPushbroomResult:
    """Run one full-chain renderer pass and retain a read-only pair A/B panel."""

    source_frames = tuple(frames)
    source_poses = tuple(poses)
    if len(source_frames) != len(source_poses) or len(source_frames) < 2:
        raise ValueError(
            "Foreground deformation diagnostic requires matching full frame/pose chains"
        )
    if isinstance(pair_index, bool) or not isinstance(pair_index, (int, np.integer)):
        raise TypeError("foreground deformation diagnostic pair index must be an integer")
    requested_pair = int(pair_index)
    if not 0 <= requested_pair < len(source_frames) - 1:
        raise IndexError("foreground deformation diagnostic pair index is not adjacent")
    experiment_config.validate()
    collector = _ForegroundDeformationCollector(
        pair_index=requested_pair,
        config=experiment_config,
    )
    source_motions = tuple(rgb_motions) if rgb_motions is not None else None
    baseline = render_calibrated_rgb_pushbroom(
        source_frames,
        source_poses,
        calibration,
        config=config,
        rgb_motions=source_motions,
        motion_pixels_to_full_resolution=motion_pixels_to_full_resolution,
        multiband_levels=multiband_levels,
        quality_gate=quality_gate,
        foreground_deformation_diagnostic_pair_index=requested_pair,
        foreground_deformation_diagnostic_callback=collector,
    )
    panorama, foreground_metadata = collector.panels_from_baseline(baseline)
    metadata = {
        "source_chain": {
            "source_count": len(source_frames),
            "frame_ids": [int(frame.frame_id) for frame in source_frames],
            "pair_frame_ids": [
                int(source_frames[requested_pair].frame_id),
                int(source_frames[requested_pair + 1].frame_id),
            ],
            "all_real_pose_nodes": True,
            "pose_rewrite_detected": False,
        },
        "foreground_deformation_experiment": experiment_config.as_dict(),
        "foreground_deformation": foreground_metadata,
        "baseline_render": {
            "source_remap_count": int(
                baseline.metadata["quality_metrics"]["source_remap_count"]
            ),
            "full_resolution_output_remap_count": int(
                baseline.metadata["quality_metrics"]["full_resolution_output_remap_count"]
            ),
            "analysis_preview_remap_count": int(
                baseline.metadata["quality_metrics"]["analysis_preview_remap_count"]
            ),
            "layout": {
                key: baseline.metadata["layout"][key]
                for key in ("width", "height", "frame_ids", "owner_boundaries_x")
            },
        },
    }
    return CalibratedRGBPushbroomResult(
        panorama=panorama,
        metadata=_scalar_tree(metadata),  # type: ignore[arg-type]
    )


def render_foreground_deformation_panorama_diagnostic(
    frames: Sequence[RGBDFrame],
    poses: Sequence[np.ndarray],
    calibration: CameraIntrinsics,
    *,
    experiment_config: ForegroundDeformationExperimentConfig,
    config: Mapping[str, object] | CalibratedRGBPushbroomConfig | None = None,
    rgb_motions: Sequence[object] | None = None,
    motion_pixels_to_full_resolution: float = 1.0,
    multiband_levels: int = 3,
    quality_gate: bool = True,
) -> CalibratedRGBPushbroomResult:
    """Build one full diagnostic panorama from accepted adjacent foreground meshes.

    This is deliberately not a formal renderer.  It first completes the normal
    full-resolution baseline, then overlays only accepted, non-overlapping
    foreground RGB source samples in-memory.  A rejected candidate remains the
    baseline hard-owner result.
    """

    source_frames = tuple(frames)
    source_poses = tuple(poses)
    if len(source_frames) != len(source_poses) or len(source_frames) < 2:
        raise ValueError(
            "Foreground deformation diagnostic requires matching full frame/pose chains"
        )
    experiment_config.validate()
    pair_indices = tuple(range(len(source_frames) - 1))
    collector = _ForegroundDeformationPanoramaCollector(
        pair_indices=pair_indices,
        config=experiment_config,
    )
    source_motions = tuple(rgb_motions) if rgb_motions is not None else None
    baseline = render_calibrated_rgb_pushbroom(
        source_frames,
        source_poses,
        calibration,
        config=config,
        rgb_motions=source_motions,
        motion_pixels_to_full_resolution=motion_pixels_to_full_resolution,
        multiband_levels=multiband_levels,
        quality_gate=quality_gate,
        foreground_deformation_diagnostic_pair_indices=pair_indices,
        foreground_deformation_diagnostic_callback=collector,
    )
    panorama, foreground_metadata = collector.panorama_from_baseline(baseline)
    metadata = {
        "source_chain": {
            "source_count": len(source_frames),
            "frame_ids": [int(frame.frame_id) for frame in source_frames],
            "pair_count": len(pair_indices),
            "all_real_pose_nodes": True,
            "pose_rewrite_detected": False,
        },
        "foreground_deformation_experiment": experiment_config.as_dict(),
        "foreground_deformation": foreground_metadata,
        "baseline_render": {
            "source_remap_count": int(
                baseline.metadata["quality_metrics"]["source_remap_count"]
            ),
            "full_resolution_output_remap_count": int(
                baseline.metadata["quality_metrics"]["full_resolution_output_remap_count"]
            ),
            "analysis_preview_remap_count": int(
                baseline.metadata["quality_metrics"]["analysis_preview_remap_count"]
            ),
            "layout": {
                key: baseline.metadata["layout"][key]
                for key in ("width", "height", "frame_ids", "owner_boundaries_x")
            },
        },
    }
    return CalibratedRGBPushbroomResult(
        panorama=panorama,
        metadata=_scalar_tree(metadata),  # type: ignore[arg-type]
    )


def _foreground_deformation_renderer(
    *,
    render_frames: Sequence[RGBDFrame],
    render_poses: Sequence[np.ndarray],
    calibration: CameraIntrinsics,
    config: Mapping[str, object],
    rgb_motions: Sequence[object] | None,
    motion_pixels_to_full_resolution: float,
    multiband_levels: int,
    pair_index: int | None,
    whole_panorama: bool,
    experiment_config: ForegroundDeformationExperimentConfig,
) -> CalibratedRGBPushbroomResult:
    if whole_panorama:
        return render_foreground_deformation_panorama_diagnostic(
            render_frames,
            render_poses,
            calibration,
            experiment_config=experiment_config,
            config=config,
            rgb_motions=rgb_motions,
            motion_pixels_to_full_resolution=motion_pixels_to_full_resolution,
            multiband_levels=multiband_levels,
            quality_gate=False,
        )
    if pair_index is None:
        raise ValueError("foreground deformation pair diagnostic requires a pair index")
    return render_foreground_deformation_pair_diagnostic(
        render_frames,
        render_poses,
        calibration,
        pair_index=pair_index,
        experiment_config=experiment_config,
        config=config,
        rgb_motions=rgb_motions,
        motion_pixels_to_full_resolution=motion_pixels_to_full_resolution,
        multiband_levels=multiband_levels,
        quality_gate=False,
    )


def main() -> None:
    args = _parser().parse_args()
    try:
        report = stitch_sequence.run(
            args,
            foreground_deformation_diagnostic_renderer=_foreground_deformation_renderer,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print(f"Diagnostic panorama: {report['panorama']}")
    print(f"Diagnostic report: {report['report']}")
    print("Diagnostic only: no delivery.json was published")


if __name__ == "__main__":
    main()
