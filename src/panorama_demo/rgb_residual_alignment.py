"""RGB-only residual-alignment evidence and safety primitives.

This module deliberately has no depth, point-cloud, TSDF, homography, or
two-dimensional-camera-pose dependency.  It analyses low-resolution calibrated
RGB previews and their raw-coordinate inverse maps.  The formal renderer may
use its explicitly audited :class:`SourceResidualWarp` objects *only before*
constructing the calibrated virtual ray; a warp never replaces the real RGB-D
SE(3) trajectory.

The first integration phase is diagnostic/identity-only.  The data structures
and deterministic held-out partition are intentionally useful before any warp
is allowed to change output pixels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import cv2
import numpy as np


_FORBIDDEN_BACKENDS = {
    "apap",
    "tps",
    "homography",
    "dibr",
    "depth_projection",
    "depth",
    "optical_flow_warp",
}


def _finite_float(value: object, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


@dataclass(frozen=True)
class ResidualAlignmentConfig:
    """Closed configuration surface for RGB residual analysis.

    The bounds here are formal hard limits, not quality thresholds that a
    diagnostic flag may relax.  A caller can choose an identity result when
    evidence is insufficient; it cannot request a larger or un-audited warp.
    """

    backend: str = "se3_epipolar_hierarchical_rgb"
    analysis_width: int = 640
    maximum_preview_megapixels: float = 2.0
    maximum_evidence_megapixels: float = 3.0
    held_out_fraction: float = 0.20
    held_out_seed: int = 20301117
    owner_track_consistency: bool = True
    background_model: str = "se2"
    maximum_residual_displacement_pixels: float = 8.0
    maximum_background_roll_degrees: float = 0.25
    maximum_flow_fb_error_pixels: float = 1.0
    maximum_epipolar_error_pixels: float = 1.0
    # Rigid-normal fitting is intentionally not a formal renderer model.  It
    # belongs to the separately declared diagnostic phase; the formal path
    # presently uses only whole-component hard ownership.
    component_model: str = "owner_only"
    maximum_edge_step_p95_pixels: float = 1.5
    maximum_edge_step_pixels: float = 3.0
    cut_mesh_formal_enabled: bool = False

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, object] | None
    ) -> "ResidualAlignmentConfig":
        supplied = {} if value is None else dict(value)
        allowed = {
            "backend",
            "analysis_width",
            "maximum_preview_megapixels",
            "maximum_evidence_megapixels",
            "held_out_fraction",
            "held_out_seed",
            "owner_track_consistency",
            "background_model",
            "maximum_residual_displacement_pixels",
            "maximum_background_roll_degrees",
            "maximum_flow_fb_error_pixels",
            "maximum_epipolar_error_pixels",
            "component_model",
            "maximum_edge_step_p95_pixels",
            "maximum_edge_step_pixels",
            "cut_mesh_formal_enabled",
        }
        unknown = sorted(set(supplied) - allowed)
        if unknown:
            raise ValueError(
                "Unknown residual_alignment configuration keys: " + ", ".join(unknown)
            )
        try:
            result = cls(**supplied)
        except TypeError as exc:
            raise ValueError("Invalid residual_alignment configuration") from exc
        result._validate()
        return result

    def _validate(self) -> None:
        if self.backend.lower() in _FORBIDDEN_BACKENDS:
            raise ValueError("Residual alignment forbids projective/depth warp backends")
        if self.backend != "se3_epipolar_hierarchical_rgb":
            raise ValueError(
                "residual_alignment.backend must be se3_epipolar_hierarchical_rgb"
            )
        if not 40 <= int(self.analysis_width) <= 640:
            raise ValueError("residual_alignment.analysis_width must be in [40, 640]")
        for name, ceiling in (
            ("maximum_preview_megapixels", 2.0),
            ("maximum_evidence_megapixels", 3.0),
        ):
            value = _finite_float(getattr(self, name), name)
            if not 0.0 < value <= ceiling:
                raise ValueError(f"{name} must be in (0, {ceiling}]")
        fraction = _finite_float(self.held_out_fraction, "held_out_fraction")
        if not 0.0 < fraction < 0.5:
            raise ValueError("residual_alignment.held_out_fraction must be in (0, 0.5)")
        if not isinstance(self.held_out_seed, (int, np.integer)):
            raise ValueError("residual_alignment.held_out_seed must be an integer")
        if type(self.owner_track_consistency) is not bool:
            raise ValueError("owner_track_consistency must be a boolean")
        if self.background_model not in {"identity", "se2"}:
            raise ValueError("background_model must be identity or se2")
        if self.component_model != "owner_only":
            raise ValueError(
                "component_model must remain owner_only; rigid-normal is diagnostic-only"
            )
        if type(self.cut_mesh_formal_enabled) is not bool or self.cut_mesh_formal_enabled:
            raise ValueError("cut_mesh_formal_enabled must remain false for formal rendering")
        displacement = _finite_float(
            self.maximum_residual_displacement_pixels,
            "maximum_residual_displacement_pixels",
        )
        if not 0.0 < displacement <= 8.0:
            raise ValueError("maximum residual displacement must be in (0, 8]")
        roll = _finite_float(
            self.maximum_background_roll_degrees,
            "maximum_background_roll_degrees",
        )
        if not 0.0 < roll <= 0.25:
            raise ValueError("maximum background roll must be in (0, 0.25]")
        for name, ceiling in (
            ("maximum_flow_fb_error_pixels", 1.0),
            ("maximum_epipolar_error_pixels", 1.0),
            ("maximum_edge_step_p95_pixels", 1.5),
            ("maximum_edge_step_pixels", 3.0),
        ):
            value = _finite_float(getattr(self, name), name)
            if not 0.0 < value <= ceiling:
                raise ValueError(f"{name} must be in (0, {ceiling}]")

    def as_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "analysis_width": int(self.analysis_width),
            "maximum_preview_megapixels": float(self.maximum_preview_megapixels),
            "maximum_evidence_megapixels": float(self.maximum_evidence_megapixels),
            "held_out_fraction": float(self.held_out_fraction),
            "held_out_seed": int(self.held_out_seed),
            "owner_track_consistency": self.owner_track_consistency,
            "background_model": self.background_model,
            "maximum_residual_displacement_pixels": float(
                self.maximum_residual_displacement_pixels
            ),
            "maximum_background_roll_degrees": float(
                self.maximum_background_roll_degrees
            ),
            "maximum_flow_fb_error_pixels": float(self.maximum_flow_fb_error_pixels),
            "maximum_epipolar_error_pixels": float(self.maximum_epipolar_error_pixels),
            "component_model": self.component_model,
            "maximum_edge_step_p95_pixels": float(self.maximum_edge_step_p95_pixels),
            "maximum_edge_step_pixels": float(self.maximum_edge_step_pixels),
            "cut_mesh_formal_enabled": False,
        }


@dataclass(frozen=True)
class PairResidualEvidence:
    """Adjacent-preview RGB evidence with explicit unsupported regions."""

    pair_index: int
    frame_ids: tuple[int, int]
    candidate_count: int
    accepted_count: int
    held_out_count: int
    held_out_accepted_count: int
    observable_mask: np.ndarray = field(repr=False)
    accepted_mask: np.ndarray = field(repr=False)
    held_out_mask: np.ndarray = field(repr=False)
    flow_uncertain_mask: np.ndarray = field(repr=False)
    occluded_mask: np.ndarray = field(repr=False)
    forward_flow: np.ndarray = field(repr=False)
    backward_flow: np.ndarray = field(repr=False)
    forward_backward_error: np.ndarray = field(repr=False)
    structure_minimum_eigenvalue: np.ndarray = field(repr=False)
    epipolar_residual_pixels: np.ndarray = field(repr=False)
    edge_normal_step_pixels: np.ndarray = field(repr=False)
    edge_orientation_delta_degrees: np.ndarray = field(repr=False)
    metrics: Mapping[str, object] = field(default_factory=dict)

    @property
    def unobservable_count(self) -> int:
        return int(self.candidate_count - np.count_nonzero(self.observable_mask))

    def as_dict(self) -> dict[str, object]:
        return {
            "pair_index": int(self.pair_index),
            "frame_ids": [int(value) for value in self.frame_ids],
            "candidate_count": int(self.candidate_count),
            "accepted_count": int(self.accepted_count),
            "held_out_count": int(self.held_out_count),
            "held_out_accepted_count": int(self.held_out_accepted_count),
            "unobservable_count": self.unobservable_count,
            "flow_uncertain_count": int(np.count_nonzero(self.flow_uncertain_mask)),
            "occluded_count": int(np.count_nonzero(self.occluded_mask)),
            "metrics": dict(self.metrics),
        }


@dataclass(frozen=True)
class ProtectedComponentFragment:
    """One pair-local risk fragment that a hard owner must keep whole."""

    pair_index: int
    global_bbox: tuple[int, int, int, int]
    local_mask: np.ndarray = field(repr=False)
    allowed_owners: tuple[int, ...] = (0, 1)
    coverage_margin: float = 0.0
    edge_orientation: float | None = None
    uncertainty: float | None = None
    component_label: int | None = None
    preferred_owner: int | None = None

    def __post_init__(self) -> None:
        mask = np.asarray(self.local_mask, dtype=bool)
        if mask.ndim != 2 or not np.any(mask):
            raise ValueError("Protected component fragment requires a non-empty 2-D mask")
        if not self.allowed_owners or any(owner not in {0, 1} for owner in self.allowed_owners):
            raise ValueError("Protected component owners must be a non-empty subset of (0, 1)")
        if not math.isfinite(float(self.coverage_margin)):
            raise ValueError("Protected component coverage margin must be finite")
        if self.component_label is not None and int(self.component_label) < 1:
            raise ValueError("Protected component label must be positive when supplied")
        if self.preferred_owner is not None and int(self.preferred_owner) not in {0, 1}:
            raise ValueError("Protected component preferred owner must be 0 or 1")
        object.__setattr__(self, "local_mask", np.ascontiguousarray(mask))

    def as_dict(self) -> dict[str, object]:
        return {
            "pair_index": int(self.pair_index),
            "global_bbox": [int(value) for value in self.global_bbox],
            "pixel_count": int(np.count_nonzero(self.local_mask)),
            "allowed_owners": [int(value) for value in self.allowed_owners],
            "coverage_margin": float(self.coverage_margin),
            "edge_orientation": self.edge_orientation,
            "uncertainty": self.uncertainty,
            "component_label": self.component_label,
            "preferred_owner": self.preferred_owner,
        }


@dataclass(frozen=True)
class ComponentTrack:
    """A high-confidence one-to-one association across neighbouring pairs."""

    track_id: int
    fragments: tuple[ProtectedComponentFragment, ...]
    confidence: float
    ambiguous: bool = False

    def __post_init__(self) -> None:
        if not self.fragments:
            raise ValueError("Component track must contain at least one fragment")
        if not math.isfinite(float(self.confidence)) or not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("Component track confidence must be in [0, 1]")

    def as_dict(self) -> dict[str, object]:
        return {
            "track_id": int(self.track_id),
            "confidence": float(self.confidence),
            "ambiguous": bool(self.ambiguous),
            "fragments": [fragment.as_dict() for fragment in self.fragments],
        }


@dataclass(frozen=True)
class SequenceOwnerPreflight:
    """Pair-local hard constraints and only unambiguous cross-pair tracks."""

    component_owner_constraints: tuple[Mapping[int, int], ...]
    owner_priors: tuple[Mapping[int, int], ...]
    component_tracks: tuple[ComponentTrack, ...]
    ambiguous_association_count: int
    candidate_track_count: int = 0
    rejected_track_count: int = 0
    structural_failure_reason: str | None = None

    @property
    def accepted(self) -> bool:
        return self.structural_failure_reason is None

    def as_dict(self) -> dict[str, object]:
        return {
            "accepted": self.accepted,
            "structural_failure_reason": self.structural_failure_reason,
            "component_owner_constraints": [
                {str(label): int(owner) for label, owner in constraints.items()}
                for constraints in self.component_owner_constraints
            ],
            "owner_priors": [
                {str(label): int(owner) for label, owner in prior.items()}
                for prior in self.owner_priors
            ],
            "track_count": len(self.component_tracks),
            "candidate_track_count": int(self.candidate_track_count),
            "rejected_track_count": int(self.rejected_track_count),
            "ambiguous_association_count": int(self.ambiguous_association_count),
            "tracks": [track.as_dict() for track in self.component_tracks],
        }


def extract_protected_component_fragments(
    protected_mask: np.ndarray,
    valid_first: np.ndarray,
    valid_second: np.ndarray,
    *,
    pair_index: int,
    global_x0: int,
    nominal_boundary_x: float,
) -> tuple[ProtectedComponentFragment, ...]:
    """Turn a pair risk guard into whole-owner fragments without depth input."""

    protected = np.asarray(protected_mask, dtype=bool)
    first = _as_valid_mask(valid_first, protected.shape, "valid_first")
    second = _as_valid_mask(valid_second, protected.shape, "valid_second")
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        protected.astype(np.uint8), connectivity=8
    )
    fragments: list[ProtectedComponentFragment] = []
    for label in range(1, count):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        if width <= 0 or height <= 0:
            continue
        local = labels[y : y + height, x : x + width] == label
        available_first = bool(np.all(first[y : y + height, x : x + width][local]))
        available_second = bool(np.all(second[y : y + height, x : x + width][local]))
        allowed = tuple(
            owner
            for owner, available in ((0, available_first), (1, available_second))
            if available
        )
        if not allowed:
            # It should be impossible for a component derived from common
            # valid RGB, but preserve fail-closed behaviour if caller input is
            # inconsistent rather than guessing a partial owner.
            raise RuntimeError("Protected RGB component has no complete source owner")
        yy, xx = np.nonzero(local)
        orientation: float | None = None
        if len(xx) >= 3:
            coordinates = np.column_stack((xx, yy)).astype(np.float32)
            try:
                direction = np.asarray(
                    cv2.fitLine(coordinates, cv2.DIST_L2, 0, 0.01, 0.01)
                ).reshape(-1)
                orientation = float(
                    math.degrees(math.atan2(float(direction[1]), float(direction[0])))
                )
            except cv2.error:
                orientation = None
        component_centre = global_x0 + x + 0.5 * max(0, width - 1)
        preferred = 0 if component_centre <= nominal_boundary_x else 1
        # Signed margin documents the exact pair-local nominal choice without
        # pretending it is a cross-pair identity or a depth estimate.
        coverage_margin = float(nominal_boundary_x - component_centre)
        fragments.append(
            ProtectedComponentFragment(
                pair_index=int(pair_index),
                global_bbox=(global_x0 + x, y, width, height),
                local_mask=local,
                allowed_owners=allowed,
                coverage_margin=coverage_margin,
                edge_orientation=orientation,
                uncertainty=None,
                component_label=label,
                preferred_owner=preferred if preferred in allowed else allowed[0],
            )
        )
    return tuple(fragments)


def _bbox_iou(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    left, top = max(ax, bx), max(ay, by)
    right, bottom = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    intersection = max(0, right - left) * max(0, bottom - top)
    union = aw * ah + bw * bh - intersection
    return float(intersection / union) if union else 0.0


def _constraints_allow_monotonic_owner(
    fragments: Sequence[ProtectedComponentFragment],
    constraints: Mapping[int, int],
) -> bool:
    """Check whether whole-component owners admit one left-to-right cut.

    A pair corridor has exactly one monotonic boundary: every forced first
    source fragment must stay on or left of it, and every forced second source
    fragment must stay strictly to its right.  The check uses the true local
    component masks rather than their coarse bounding boxes, so a slanted or
    disconnected row does not create a false conflict.  It is deliberately a
    preflight check: incompatible *associations* are discarded before the
    renderer reaches GraphCut, while a locally impossible protected scene
    remains a structural failure.
    """

    lower_by_row: dict[int, int] = {}
    upper_by_row: dict[int, int] = {}
    for fragment in fragments:
        label = fragment.component_label
        if label is None or label not in constraints:
            continue
        owner = int(constraints[label])
        if owner not in {0, 1}:
            return False
        x0, y0, _, _ = fragment.global_bbox
        mask = np.asarray(fragment.local_mask, dtype=bool)
        for local_y in np.flatnonzero(np.any(mask, axis=1)):
            local_x = np.flatnonzero(mask[int(local_y)])
            if not local_x.size:
                continue
            row = int(y0 + int(local_y))
            if owner == 0:
                lower_by_row[row] = max(
                    lower_by_row.get(row, -1),
                    int(x0 + int(local_x[-1])),
                )
            else:
                upper_by_row[row] = min(
                    upper_by_row.get(row, int(x0 + mask.shape[1]) - 1),
                    int(x0 + int(local_x[0]) - 1),
                )
    return all(
        lower <= upper_by_row.get(row, lower)
        for row, lower in lower_by_row.items()
    )


def preflight_sequence_owners(
    fragments_by_pair: Sequence[Sequence[ProtectedComponentFragment]],
) -> SequenceOwnerPreflight:
    """Choose safe pair owners before GraphCut and retain only unique tracks.

    A track is created only for an unambiguous one-to-one bbox/orientation
    association in consecutive pair corridors.  Split/merge or several equally
    plausible matches are reported as ambiguous and deliberately receive no
    cross-pair force.  This keeps the final renderer adjacent-source-only.
    """

    constraints: list[dict[int, int]] = []
    priors: list[dict[int, int]] = []
    for fragments in fragments_by_pair:
        pair_constraints: dict[int, int] = {}
        pair_prior: dict[int, int] = {}
        for fragment in fragments:
            if fragment.component_label is None:
                return SequenceOwnerPreflight(
                    component_owner_constraints=tuple(constraints),
                    owner_priors=tuple(priors),
                    component_tracks=(),
                    ambiguous_association_count=0,
                    structural_failure_reason="fragment_without_component_label",
                )
            owner = (
                fragment.preferred_owner
                if fragment.preferred_owner in fragment.allowed_owners
                else fragment.allowed_owners[0]
            )
            pair_constraints[int(fragment.component_label)] = int(owner)
            pair_prior[int(fragment.component_label)] = int(owner)
        constraints.append(pair_constraints)
        priors.append(pair_prior)

    # First collect pairwise reciprocal links without changing any owner.  A
    # middle fragment can otherwise be linked to both its predecessor and its
    # successor: one link requires it to use the first source and the other
    # requires it to use the second.  "Last write wins" would silently make an
    # impossible owner topology.  Such a chain junction is ambiguous by
    # definition and must retain only its pair-local owner decision.
    candidate_links: list[
        tuple[float, int, ProtectedComponentFragment, ProtectedComponentFragment]
    ] = []
    ambiguous = 0
    for pair_index in range(len(fragments_by_pair) - 1):
        left_fragments = tuple(fragments_by_pair[pair_index])
        right_fragments = tuple(fragments_by_pair[pair_index + 1])
        left_candidates: list[list[tuple[float, int]]] = [
            [] for _ in left_fragments
        ]
        right_candidates: list[list[tuple[float, int]]] = [
            [] for _ in right_fragments
        ]
        for left_index, left in enumerate(left_fragments):
            for right_index, right in enumerate(right_fragments):
                overlap = _bbox_iou(left.global_bbox, right.global_bbox)
                if overlap <= 0.20:
                    continue
                if (
                    left.edge_orientation is not None
                    and right.edge_orientation is not None
                    and abs((left.edge_orientation - right.edge_orientation + 90.0) % 180.0 - 90.0)
                    > 25.0
                ):
                    continue
                left_candidates[left_index].append((overlap, right_index))
                right_candidates[right_index].append((overlap, left_index))

        # A component association is usable only when the candidate graph is
        # reciprocal one-to-one.  A split or merge remains a pair-local owner
        # decision; forcing it across pairs would create a false component
        # identity and can split a protected foreground edge.
        ambiguous_nodes: set[tuple[str, int]] = set()
        for left_index, candidates in enumerate(left_candidates):
            if candidates and len(candidates) != 1:
                ambiguous_nodes.add(("left", left_index))
        for right_index, candidates in enumerate(right_candidates):
            if candidates and len(candidates) != 1:
                ambiguous_nodes.add(("right", right_index))
        ambiguous += len(ambiguous_nodes)

        for left_index, candidates in enumerate(left_candidates):
            if len(candidates) != 1:
                continue
            overlap, right_index = candidates[0]
            if len(right_candidates[right_index]) != 1:
                continue
            left = left_fragments[left_index]
            right = right_fragments[right_index]
            # Pair (i-1, i)'s second owner and pair (i, i+1)'s first owner
            # refer to the shared real source i.  Enforce it only if both
            # fragments can be completely covered by that source.
            if 1 not in left.allowed_owners or 0 not in right.allowed_owners:
                ambiguous += 1
                continue
            assert left.component_label is not None and right.component_label is not None
            candidate_links.append((float(overlap), pair_index, left, right))

    incident_count: dict[tuple[int, int], int] = {}
    for _, _, left, right in candidate_links:
        assert left.component_label is not None and right.component_label is not None
        for fragment in (left, right):
            key = (int(fragment.pair_index), int(fragment.component_label))
            incident_count[key] = incident_count.get(key, 0) + 1

    tracks: list[ComponentTrack] = []
    rejected_track_count = 0
    track_id = 0
    # Prefer the most strongly overlapping non-conflicting association, while
    # retaining deterministic tie-breaking for audits and tests.
    for overlap, pair_index, left, right in sorted(
        candidate_links,
        key=lambda item: (
            -item[0],
            item[1],
            int(item[2].component_label or 0),
            int(item[3].component_label or 0),
        ),
    ):
        assert left.component_label is not None and right.component_label is not None
        left_key = (int(left.pair_index), int(left.component_label))
        right_key = (int(right.pair_index), int(right.component_label))
        if incident_count[left_key] != 1 or incident_count[right_key] != 1:
            ambiguous += 1
            rejected_track_count += 1
            continue

        # An otherwise one-to-one link may still reverse the only legal seam
        # order when combined with a separate link in the same pair.  Reject
        # that association before rendering; do not relax the risk guard or
        # overwrite a prior constraint.
        proposed_left = dict(constraints[pair_index])
        proposed_right = dict(constraints[pair_index + 1])
        proposed_left[int(left.component_label)] = 1
        proposed_right[int(right.component_label)] = 0
        if not (
            _constraints_allow_monotonic_owner(
                fragments_by_pair[pair_index], proposed_left
            )
            and _constraints_allow_monotonic_owner(
                fragments_by_pair[pair_index + 1], proposed_right
            )
        ):
            ambiguous += 1
            rejected_track_count += 1
            continue

        constraints[pair_index] = proposed_left
        constraints[pair_index + 1] = proposed_right
        priors[pair_index][int(left.component_label)] = 1
        priors[pair_index + 1][int(right.component_label)] = 0
        tracks.append(
            ComponentTrack(
                track_id=track_id,
                fragments=(left, right),
                confidence=float(overlap),
                ambiguous=False,
            )
        )
        track_id += 1

    for pair_index, fragments in enumerate(fragments_by_pair):
        if not _constraints_allow_monotonic_owner(
            fragments, constraints[pair_index]
        ):
            return SequenceOwnerPreflight(
                component_owner_constraints=tuple(constraints),
                owner_priors=tuple(priors),
                component_tracks=tuple(tracks),
                ambiguous_association_count=ambiguous,
                candidate_track_count=len(candidate_links),
                rejected_track_count=rejected_track_count,
                structural_failure_reason=(
                    "pair_local_component_owners_do_not_admit_a_monotonic_seam:"
                    f"{pair_index}"
                ),
            )
    return SequenceOwnerPreflight(
        component_owner_constraints=tuple(constraints),
        owner_priors=tuple(priors),
        component_tracks=tuple(tracks),
        ambiguous_association_count=ambiguous,
        candidate_track_count=len(candidate_links),
        rejected_track_count=rejected_track_count,
    )


@dataclass(frozen=True)
class SourceResidualWarp:
    """Small source-local inverse SE(2) residual in virtual canvas pixels."""

    source_index: int = -1
    translation_x: float = 0.0
    translation_y: float = 0.0
    roll_degrees: float = 0.0
    centre_x: float = 0.0
    centre_y: float = 0.0

    def __post_init__(self) -> None:
        values = np.asarray(
            (
                self.translation_x,
                self.translation_y,
                self.roll_degrees,
                self.centre_x,
                self.centre_y,
            ),
            dtype=np.float64,
        )
        if not np.isfinite(values).all():
            raise ValueError("Source residual warp values must be finite")

    @classmethod
    def identity(
        cls, source_index: int = -1, *, centre_x: float = 0.0, centre_y: float = 0.0
    ) -> "SourceResidualWarp":
        return cls(source_index=source_index, centre_x=centre_x, centre_y=centre_y)

    @property
    def is_identity(self) -> bool:
        return (
            self.translation_x == 0.0
            and self.translation_y == 0.0
            and self.roll_degrees == 0.0
        )

    @property
    def displacement_magnitude(self) -> float:
        return float(math.hypot(self.translation_x, self.translation_y))

    def inverse_virtual_coordinates(
        self, x: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply ``W_i^-1`` before virtual-ray construction.

        Forward ``W_i`` is a source-local translation then tiny rotation around
        ``centre``.  The output-to-source direction below is therefore the
        inverse rotation of ``p - translation``.  No camera pose is modified.
        """

        xx = np.asarray(x, dtype=np.float64)
        yy = np.asarray(y, dtype=np.float64)
        if xx.shape != yy.shape:
            raise ValueError("Residual inverse coordinates must have matching shapes")
        angle = math.radians(-self.roll_degrees)
        cos_angle = math.cos(angle)
        sin_angle = math.sin(angle)
        dx = xx - self.centre_x - self.translation_x
        dy = yy - self.centre_y - self.translation_y
        return (
            np.ascontiguousarray(self.centre_x + cos_angle * dx - sin_angle * dy),
            np.ascontiguousarray(self.centre_y + sin_angle * dx + cos_angle * dy),
        )

    def forward_virtual_coordinates(
        self, x: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply ``W_i`` for held-out residual and round-trip audits only."""

        xx = np.asarray(x, dtype=np.float64)
        yy = np.asarray(y, dtype=np.float64)
        if xx.shape != yy.shape:
            raise ValueError("Residual forward coordinates must have matching shapes")
        angle = math.radians(self.roll_degrees)
        cos_angle = math.cos(angle)
        sin_angle = math.sin(angle)
        dx = xx - self.centre_x
        dy = yy - self.centre_y
        return (
            np.ascontiguousarray(
                self.centre_x + cos_angle * dx - sin_angle * dy + self.translation_x
            ),
            np.ascontiguousarray(
                self.centre_y + sin_angle * dx + cos_angle * dy + self.translation_y
            ),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "source_index": int(self.source_index),
            "translation_x_pixels": float(self.translation_x),
            "translation_y_pixels": float(self.translation_y),
            "roll_degrees": float(self.roll_degrees),
            "centre_x_pixels": float(self.centre_x),
            "centre_y_pixels": float(self.centre_y),
            "identity": self.is_identity,
        }


@dataclass(frozen=True)
class WarpTopologyAudit:
    """Compact structural result for a selected set of residual inverse maps."""

    source_count: int
    finite: bool
    maximum_displacement_pixels: float
    maximum_roll_degrees: float
    support_margin_pixels: float | None
    inverse_roundtrip_error_pixels: float | None
    accepted: bool
    failure_reason: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "source_count": int(self.source_count),
            "finite": bool(self.finite),
            "maximum_displacement_pixels": float(self.maximum_displacement_pixels),
            "maximum_roll_degrees": float(self.maximum_roll_degrees),
            "support_margin_pixels": self.support_margin_pixels,
            "inverse_roundtrip_error_pixels": self.inverse_roundtrip_error_pixels,
            "accepted": bool(self.accepted),
            "failure_reason": self.failure_reason,
        }


@dataclass(frozen=True)
class ResidualAlignmentResult:
    """Selected minimum-complexity RGB residual model and its audit summary."""

    selected_model: str
    source_warps: tuple[SourceResidualWarp, ...]
    pair_evidence: tuple[PairResidualEvidence, ...]
    held_out_metrics_before: Mapping[str, float | int | None]
    held_out_metrics_after: Mapping[str, float | int | None]
    component_tracks: tuple[ComponentTrack, ...]
    topology_audit: WarpTopologyAudit
    working_set_audit: Mapping[str, int | float | bool]

    def as_dict(self) -> dict[str, object]:
        return {
            "selected_model": self.selected_model,
            "per_source_parameters": [warp.as_dict() for warp in self.source_warps],
            "evidence": [evidence.as_dict() for evidence in self.pair_evidence],
            "held_out_metrics_before": dict(self.held_out_metrics_before),
            "held_out_metrics_after": dict(self.held_out_metrics_after),
            "component_audit": {
                "track_count": len(self.component_tracks),
                "tracks": [track.as_dict() for track in self.component_tracks],
            },
            "topology_audit": self.topology_audit.as_dict(),
            "working_set_audit": dict(self.working_set_audit),
        }


def _normalise_inverse_map(value: object | None, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray] | None:
    if value is None:
        return None
    if isinstance(value, (tuple, list)) and len(value) == 2:
        map_x, map_y = (np.asarray(item, dtype=np.float64) for item in value)
    else:
        array = np.asarray(value, dtype=np.float64)
        if array.shape == (*shape, 2):
            map_x, map_y = array[:, :, 0], array[:, :, 1]
        else:
            raise ValueError("Inverse map must be (map_x, map_y) or HxWx2")
    if map_x.shape != shape or map_y.shape != shape:
        raise ValueError("Inverse map must match the RGB preview shape")
    return np.ascontiguousarray(map_x), np.ascontiguousarray(map_y)


def _as_bgr_image(value: np.ndarray, name: str) -> np.ndarray:
    image = np.asarray(value)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError(f"{name} must be a uint8 BGR HxWx3 preview")
    return np.ascontiguousarray(image)


def _as_valid_mask(value: np.ndarray, shape: tuple[int, int], name: str) -> np.ndarray:
    mask = np.asarray(value, dtype=bool)
    if mask.shape != shape:
        raise ValueError(f"{name} must match the RGB preview")
    return np.ascontiguousarray(mask)


def build_held_out_partition(
    shape: tuple[int, int],
    *,
    seed: int,
    frame_ids: tuple[int, int],
    fraction: float = 0.20,
) -> np.ndarray:
    """Return a stable spatial 80/20 partition independent of processing order."""

    if len(shape) != 2 or shape[0] <= 0 or shape[1] <= 0:
        raise ValueError("Held-out partition shape must be positive HxW")
    if not isinstance(seed, (int, np.integer)):
        raise ValueError("Held-out partition seed must be an integer")
    if len(frame_ids) != 2:
        raise ValueError("Held-out partition requires two frame identifiers")
    portion = _finite_float(fraction, "held-out fraction")
    if not 0.0 < portion < 0.5:
        raise ValueError("Held-out fraction must be in (0, 0.5)")
    yy, xx = np.indices(shape, dtype=np.uint64)
    # This is deliberately uint64 arithmetic: overflow is the desired hash
    # behaviour, so suppress NumPy's otherwise noisy scalar-overflow warning.
    with np.errstate(over="ignore"):
        state = (
            np.uint64(int(seed) & ((1 << 64) - 1))
            ^ (xx * np.uint64(0x9E3779B185EBCA87))
            ^ (yy * np.uint64(0xC2B2AE3D27D4EB4F))
            ^ (
                np.uint64(int(frame_ids[0]) & ((1 << 64) - 1))
                * np.uint64(0x165667B19E3779F9)
            )
            ^ (
                np.uint64(int(frame_ids[1]) & ((1 << 64) - 1))
                * np.uint64(0xD6E8FEB86659FD93)
            )
        )
        state ^= state >> np.uint64(30)
        state *= np.uint64(0xBF58476D1CE4E5B9)
        state ^= state >> np.uint64(27)
        state *= np.uint64(0x94D049BB133111EB)
        state ^= state >> np.uint64(31)
    # Comparing a fixed 16-bit fraction avoids platform-dependent float
    # rounding at the split boundary.
    threshold = int(round(portion * 65536.0))
    return np.ascontiguousarray((state & np.uint64(0xFFFF)) < np.uint64(threshold))


def _flow_pair(first_gray: np.ndarray, second_gray: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute DIS candidates and their forward/backward consistency error."""

    height, width = first_gray.shape
    def unavailable() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        zero = np.zeros((height, width, 2), dtype=np.float32)
        return (
            zero,
            zero.copy(),
            np.full((height, width), np.inf, dtype=np.float32),
            np.zeros((height, width), dtype=bool),
        )

    if min(height, width) < 8:
        return unavailable()
    try:
        dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
        dis.setFinestScale(1)
        forward = np.asarray(dis.calc(first_gray, second_gray, None), dtype=np.float32)
        backward = np.asarray(dis.calc(second_gray, first_gray, None), dtype=np.float32)
    except cv2.error:
        # A failed optical-flow invocation is unknown RGB evidence.  Returning
        # a zero vector here would fabricate a perfect correspondence and can
        # incorrectly select a residual correction on a broken preview pass.
        return unavailable()
    xx, yy = np.meshgrid(
        np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32)
    )
    target_x = xx + forward[:, :, 0]
    target_y = yy + forward[:, :, 1]
    inside = (
        np.isfinite(target_x)
        & np.isfinite(target_y)
        & (target_x >= 0.0)
        & (target_x <= width - 1)
        & (target_y >= 0.0)
        & (target_y <= height - 1)
    )
    sampled_backward_x = cv2.remap(
        backward[:, :, 0], target_x, target_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan
    )
    sampled_backward_y = cv2.remap(
        backward[:, :, 1], target_x, target_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan
    )
    error = np.hypot(
        forward[:, :, 0] + sampled_backward_x,
        forward[:, :, 1] + sampled_backward_y,
    ).astype(np.float32)
    error[~inside] = np.inf
    return forward, backward, error, inside


def _sample_map(array: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return cv2.remap(
        np.asarray(array, dtype=np.float32),
        np.asarray(x, dtype=np.float32),
        np.asarray(y, dtype=np.float32),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=np.nan,
    ).astype(np.float64)


def _sample_valid_mask(mask: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Sample an already-eroded valid support mask at a flow target."""

    sampled = cv2.remap(
        np.asarray(mask, dtype=np.uint8),
        np.asarray(x, dtype=np.float32),
        np.asarray(y, dtype=np.float32),
        cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return np.ascontiguousarray(sampled != 0)


def _eroded_analysis_support(valid: np.ndarray) -> np.ndarray:
    """Exclude every feature/flow sample whose local footprint touches invalid RGB."""

    # Corner/Scharr/Canny all have a local footprint; DIS can also be biased
    # by a black inverse-remap border.  A five-pixel support kernel combined
    # with inpainted analysis-only grayscale prevents that border becoming a
    # source of residual evidence while preserving valid RGB output pixels.
    kernel = np.ones((5, 5), dtype=np.uint8)
    return np.ascontiguousarray(
        cv2.erode(np.asarray(valid, dtype=np.uint8), kernel, iterations=1) != 0
    )


def _analysis_gray(rgb: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Create RGB-only flow input without exposing invalid black remap edges."""

    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    invalid = ~np.asarray(valid, dtype=bool)
    if not np.any(invalid):
        return gray
    try:
        return cv2.inpaint(
            gray,
            (invalid.astype(np.uint8) * 255),
            3.0,
            cv2.INPAINT_NS,
        )
    except cv2.error:
        # This preview pair remains safe because all samples touching invalid
        # RGB are removed by the eroded support.  A neutral value avoids a
        # black edge if inpainting is unavailable on a malformed input.
        result = gray.copy()
        result[invalid] = 127
        return result


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(vector, dtype=np.float64)
    return np.array(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)), dtype=np.float64)


def _epipolar_sampson_pixels(
    first_map: tuple[np.ndarray, np.ndarray] | None,
    second_map: tuple[np.ndarray, np.ndarray] | None,
    forward: np.ndarray,
    inside: np.ndarray,
    intrinsics: object | None,
    first_camera_to_world: np.ndarray | None,
    second_camera_to_world: np.ndarray | None,
) -> np.ndarray:
    """Evaluate true-SE(3) Sampson residuals in approximate pixel units."""

    shape = forward.shape[:2]
    result = np.full(shape, np.inf, dtype=np.float64)
    if (
        first_map is None
        or second_map is None
        or intrinsics is None
        or first_camera_to_world is None
        or second_camera_to_world is None
    ):
        return result
    try:
        fx = _finite_float(getattr(intrinsics, "fx"), "intrinsics.fx")
        fy = _finite_float(getattr(intrinsics, "fy"), "intrinsics.fy")
        cx = _finite_float(getattr(intrinsics, "cx"), "intrinsics.cx")
        cy = _finite_float(getattr(intrinsics, "cy"), "intrinsics.cy")
        pose0 = np.asarray(first_camera_to_world, dtype=np.float64)
        pose1 = np.asarray(second_camera_to_world, dtype=np.float64)
        if pose0.shape != (4, 4) or pose1.shape != (4, 4):
            return result
        r10 = pose1[:3, :3].T @ pose0[:3, :3]
        t10 = pose1[:3, :3].T @ (pose0[:3, 3] - pose1[:3, 3])
        if not np.isfinite(r10).all() or not np.isfinite(t10).all() or np.linalg.norm(t10) <= 1e-9:
            return result
        essential = _skew(t10) @ r10
        height, width = shape
        xx, yy = np.meshgrid(
            np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32)
        )
        target_x = xx + forward[:, :, 0]
        target_y = yy + forward[:, :, 1]
        map0_x, map0_y = first_map
        map1_x = _sample_map(second_map[0], target_x, target_y)
        map1_y = _sample_map(second_map[1], target_x, target_y)
        x0 = (map0_x - cx) / fx
        y0 = (map0_y - cy) / fy
        x1 = (map1_x - cx) / fx
        y1 = (map1_y - cy) / fy
        p0 = np.stack((x0, y0, np.ones_like(x0)), axis=-1)
        p1 = np.stack((x1, y1, np.ones_like(x1)), axis=-1)
        ep0 = p0 @ essential.T
        ep1 = p1 @ essential
        numerator = np.sum(p1 * ep0, axis=-1)
        denominator = np.square(ep0[:, :, 0]) + np.square(ep0[:, :, 1]) + np.square(ep1[:, :, 0]) + np.square(ep1[:, :, 1])
        valid = inside & np.isfinite(numerator) & np.isfinite(denominator) & (denominator > 1e-12)
        result[valid] = np.abs(numerator[valid]) / np.sqrt(denominator[valid]) * math.sqrt(fx * fy)
    except (AttributeError, TypeError, ValueError, np.linalg.LinAlgError):
        return np.full(shape, np.inf, dtype=np.float64)
    return result


def _finite_percentile(values: np.ndarray, percentile: float) -> float | None:
    checked = np.asarray(values, dtype=np.float64)
    checked = checked[np.isfinite(checked)]
    if not checked.size:
        return None
    return float(np.percentile(checked, percentile))


def extract_pair_evidence(
    *,
    first_rgb: np.ndarray,
    second_rgb: np.ndarray,
    first_valid: np.ndarray,
    second_valid: np.ndarray,
    first_inverse_map: object | None = None,
    second_inverse_map: object | None = None,
    intrinsics: object | None = None,
    first_camera_to_world: np.ndarray | None = None,
    second_camera_to_world: np.ndarray | None = None,
    pair_index: int = 0,
    frame_ids: tuple[int, int] = (0, 1),
    config: ResidualAlignmentConfig | Mapping[str, object] | None = None,
) -> PairResidualEvidence:
    """Extract low-resolution RGB/SE(3) evidence without estimating a pose.

    DIS produces candidates only.  A candidate is accepted only when texture,
    forward/backward flow, common calibrated validity, and (when supplied)
    true SE(3) epipolar evidence agree.  Uniform white/black interior and
    unsupported/occluded pixels are explicitly marked unobservable rather than
    reported as zero residual.
    """

    settings = (
        config
        if isinstance(config, ResidualAlignmentConfig)
        else ResidualAlignmentConfig.from_mapping(config)
    )
    first = _as_bgr_image(first_rgb, "first_rgb")
    second = _as_bgr_image(second_rgb, "second_rgb")
    if first.shape != second.shape:
        raise ValueError("Adjacent RGB previews must have the same shape")
    shape = first.shape[:2]
    valid0 = _as_valid_mask(first_valid, shape, "first_valid")
    valid1 = _as_valid_mask(second_valid, shape, "second_valid")
    if len(frame_ids) != 2:
        raise ValueError("Pair evidence requires exactly two frame IDs")
    maps0 = _normalise_inverse_map(first_inverse_map, shape)
    maps1 = _normalise_inverse_map(second_inverse_map, shape)
    base_valid0 = valid0.copy()
    base_valid1 = valid1.copy()
    if maps0 is not None:
        base_valid0 &= np.isfinite(maps0[0]) & np.isfinite(maps0[1])
    if maps1 is not None:
        base_valid1 &= np.isfinite(maps1[0]) & np.isfinite(maps1[1])
    # ``candidate_same_coordinate`` remains an audit count for the literal
    # calibrated preview overlap.  Actual feature/flow evidence is stricter:
    # its source and flow target must each have an eroded valid footprint.
    candidate_same_coordinate = base_valid0 & base_valid1
    support0 = _eroded_analysis_support(base_valid0)
    support1 = _eroded_analysis_support(base_valid1)
    gray0 = _analysis_gray(first, base_valid0)
    gray1 = _analysis_gray(second, base_valid1)
    forward, backward, fb_error, inside = _flow_pair(gray0, gray1)
    xx, yy = np.meshgrid(
        np.arange(shape[1], dtype=np.float32), np.arange(shape[0], dtype=np.float32)
    )
    target_x = xx + forward[:, :, 0]
    target_y = yy + forward[:, :, 1]
    target_support = _sample_valid_mask(support1, target_x, target_y)
    common = support0 & target_support & inside
    structure0 = cv2.cornerMinEigenVal(gray0, blockSize=3, ksize=3)
    structure1 = cv2.cornerMinEigenVal(gray1, blockSize=3, ksize=3)
    sampled_structure1 = _sample_map(structure1, target_x, target_y)
    structure = np.minimum(
        structure0.astype(np.float64), sampled_structure1
    ).astype(np.float32)
    structure_values = structure0[support0]
    texture_threshold = (
        max(1e-3, float(np.percentile(structure_values, 55.0)) * 0.25)
        if structure_values.size
        else math.inf
    )
    # Keep source texture observability separate from a successful flow call:
    # a failed DIS pair is observable-but-unaccepted, rather than fabricated as
    # a zero-motion match.  Target texture/validity is required below for every
    # accepted correspondence.
    observable = support0 & (structure0 >= texture_threshold)
    target_texture = (
        common
        & np.isfinite(sampled_structure1)
        & (sampled_structure1 >= texture_threshold)
    )
    flow_uncertain = base_valid0 & (
        ~inside
        | ~target_support
        | ~np.isfinite(fb_error)
        | (fb_error > settings.maximum_flow_fb_error_pixels)
    )
    epipolar = _epipolar_sampson_pixels(
        maps0,
        maps1,
        forward,
        common,
        intrinsics,
        first_camera_to_world,
        second_camera_to_world,
    )
    has_epipolar_support = maps0 is not None and maps1 is not None and intrinsics is not None and first_camera_to_world is not None and second_camera_to_world is not None
    epipolar_rejected = has_epipolar_support & (~np.isfinite(epipolar) | (epipolar > settings.maximum_epipolar_error_pixels))
    occluded = base_valid0 & target_support & inside & np.isfinite(fb_error) & (
        fb_error > settings.maximum_flow_fb_error_pixels
    )

    gx0 = cv2.Scharr(gray0, cv2.CV_32F, 1, 0)
    gy0 = cv2.Scharr(gray0, cv2.CV_32F, 0, 1)
    gx1 = cv2.Scharr(gray1, cv2.CV_32F, 1, 0)
    gy1 = cv2.Scharr(gray1, cv2.CV_32F, 0, 1)
    magnitude0 = cv2.magnitude(gx0, gy0)
    magnitude1 = cv2.magnitude(gx1, gy1)
    strength_values = (
        np.concatenate((magnitude0[common], magnitude1[common]))
        if np.any(common)
        else np.empty(0)
    )
    edge_threshold = max(48.0, float(np.percentile(strength_values, 80.0)) if strength_values.size else math.inf)
    edge0 = support0 & (magnitude0 >= edge_threshold) & (
        cv2.Canny(gray0, 30, 90, L2gradient=True) > 0
    )
    edge1 = support1 & (magnitude1 >= edge_threshold) & (
        cv2.Canny(gray1, 30, 90, L2gradient=True) > 0
    )
    distance0 = cv2.distanceTransform((~edge0).astype(np.uint8), cv2.DIST_L2, 3)
    distance1 = cv2.distanceTransform((~edge1).astype(np.uint8), cv2.DIST_L2, 3)
    edge_step = np.full(shape, np.nan, dtype=np.float32)
    edge_step[edge0] = distance1[edge0]
    edge_step[edge1] = np.maximum(np.nan_to_num(edge_step[edge1], nan=0.0), distance0[edge1])
    orientation0 = np.degrees(np.arctan2(gy0, gx0))
    orientation1 = np.degrees(np.arctan2(gy1, gx1))
    sampled_orientation1 = cv2.remap(orientation1.astype(np.float32), target_x, target_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
    orientation_delta = np.abs((orientation0 - sampled_orientation1 + 90.0) % 180.0 - 90.0).astype(np.float32)
    accepted = observable & target_texture & ~flow_uncertain & ~epipolar_rejected
    held_out = build_held_out_partition(
        shape,
        seed=settings.held_out_seed,
        frame_ids=(int(frame_ids[0]), int(frame_ids[1])),
        fraction=settings.held_out_fraction,
    )
    edge_values = edge_step[np.isfinite(edge_step) & common]
    metric_values = {
        "flow_fb_error_p50_pixels": _finite_percentile(
            fb_error[candidate_same_coordinate], 50.0
        ),
        "flow_fb_error_p95_pixels": _finite_percentile(
            fb_error[candidate_same_coordinate], 95.0
        ),
        "epipolar_error_p50_pixels": _finite_percentile(epipolar[accepted], 50.0) if has_epipolar_support else None,
        "epipolar_error_p95_pixels": _finite_percentile(epipolar[accepted], 95.0) if has_epipolar_support else None,
        "edge_normal_step_p50_pixels": _finite_percentile(edge_values, 50.0),
        "edge_normal_step_p95_pixels": _finite_percentile(edge_values, 95.0),
        "edge_normal_step_max_pixels": float(np.max(edge_values)) if edge_values.size else None,
        "edge_orientation_delta_p95_degrees": _finite_percentile(orientation_delta[edge0 & inside], 95.0),
        "structure_minimum_eigenvalue_p50": _finite_percentile(
            structure[common], 50.0
        ),
        "epipolar_available": bool(has_epipolar_support),
        "unobservable": bool(not np.any(observable)),
    }
    return PairResidualEvidence(
        pair_index=int(pair_index),
        frame_ids=(int(frame_ids[0]), int(frame_ids[1])),
        candidate_count=int(np.count_nonzero(candidate_same_coordinate)),
        accepted_count=int(np.count_nonzero(accepted)),
        held_out_count=int(np.count_nonzero(candidate_same_coordinate & held_out)),
        held_out_accepted_count=int(np.count_nonzero(accepted & held_out)),
        observable_mask=np.ascontiguousarray(observable),
        accepted_mask=np.ascontiguousarray(accepted),
        held_out_mask=np.ascontiguousarray(held_out),
        flow_uncertain_mask=np.ascontiguousarray(flow_uncertain),
        occluded_mask=np.ascontiguousarray(occluded),
        forward_flow=np.ascontiguousarray(forward),
        backward_flow=np.ascontiguousarray(backward),
        forward_backward_error=np.ascontiguousarray(fb_error),
        structure_minimum_eigenvalue=np.ascontiguousarray(structure),
        epipolar_residual_pixels=np.ascontiguousarray(epipolar),
        edge_normal_step_pixels=np.ascontiguousarray(edge_step),
        edge_orientation_delta_degrees=np.ascontiguousarray(orientation_delta),
        metrics=metric_values,
    )


def measure_owner_boundary_geometry(
    evidence: PairResidualEvidence,
    *,
    boundary_x: int | float,
    half_width: int = 2,
) -> dict[str, object]:
    """Summarise only observable evidence near a proposed hard-owner boundary."""

    width = evidence.observable_mask.shape[1]
    centre = int(round(float(boundary_x)))
    radius = max(0, int(half_width))
    x = np.arange(width, dtype=np.int32)[None, :]
    region = np.abs(x - centre) <= radius
    usable = region & evidence.observable_mask & ~evidence.flow_uncertain_mask
    edge_steps = evidence.edge_normal_step_pixels[usable]
    edge_steps = edge_steps[np.isfinite(edge_steps)]
    return {
        "boundary_x": centre,
        "half_width": radius,
        "observable_pixel_count": int(np.count_nonzero(usable)),
        "flow_uncertain_pixel_count": int(np.count_nonzero(region & evidence.flow_uncertain_mask)),
        "occluded_pixel_count": int(np.count_nonzero(region & evidence.occluded_mask)),
        "edge_normal_step_p50_pixels": _finite_percentile(edge_steps, 50.0),
        "edge_normal_step_p95_pixels": _finite_percentile(edge_steps, 95.0),
        "edge_normal_step_max_pixels": float(np.max(edge_steps)) if edge_steps.size else None,
        "unobservable": bool(not np.any(usable)),
    }


def audit_source_warps(
    warps: Sequence[SourceResidualWarp],
    config: ResidualAlignmentConfig | Mapping[str, object] | None = None,
    *,
    support_margin_pixels: float | Sequence[float] | None = None,
    support_bounds: Sequence[tuple[float, float, float, float]] | None = None,
) -> WarpTopologyAudit:
    """Check non-bypassable amplitude/finite/support invariants for SE(2)."""

    settings = (
        config
        if isinstance(config, ResidualAlignmentConfig)
        else ResidualAlignmentConfig.from_mapping(config)
    )
    checked = tuple(warps)
    if support_bounds is None:
        bounds: tuple[tuple[float, float, float, float] | None, ...] = (
            None,
        ) * len(checked)
    else:
        raw_bounds = tuple(
            tuple(float(value) for value in item) for item in support_bounds
        )
        if len(raw_bounds) != len(checked) or any(
            len(item) != 4 or not np.isfinite(item).all() for item in raw_bounds
        ):
            raise ValueError("Residual support bounds must match source warps")
        bounds = raw_bounds

    def deformation(
        warp: SourceResidualWarp, bounds_item: tuple[float, float, float, float] | None
    ) -> float:
        if bounds_item is None:
            return warp.displacement_magnitude
        left, right, top, bottom = bounds_item
        if right < left or bottom < top:
            raise ValueError("Residual support bounds must be ordered left/right/top/bottom")
        x = np.asarray((left, left, right, right), dtype=np.float64)
        y = np.asarray((top, bottom, top, bottom), dtype=np.float64)
        warped_x, warped_y = warp.forward_virtual_coordinates(x, y)
        return float(np.max(np.hypot(warped_x - x, warped_y - y)))

    per_warp_displacement = tuple(
        deformation(warp, bounds_item)
        for warp, bounds_item in zip(checked, bounds, strict=True)
    )
    displacement = max(per_warp_displacement, default=0.0)
    roll = max((abs(float(warp.roll_degrees)) for warp in checked), default=0.0)
    finite = all(
        math.isfinite(value)
        for warp in checked
        for value in (warp.translation_x, warp.translation_y, warp.roll_degrees, warp.centre_x, warp.centre_y)
    )
    if support_margin_pixels is None:
        per_source_margin: tuple[float, ...] | None = None
        minimum_margin: float | None = None
    elif isinstance(support_margin_pixels, (int, float, np.floating, np.integer)):
        minimum_margin = float(support_margin_pixels)
        per_source_margin = (minimum_margin,) * len(checked)
    else:
        per_source_margin = tuple(float(value) for value in support_margin_pixels)
        if len(per_source_margin) != len(checked) or not np.isfinite(per_source_margin).all():
            raise ValueError("Per-source residual support margins must match source warps")
        minimum_margin = min(per_source_margin, default=0.0)
    reason: str | None = None
    if not finite:
        reason = "non-finite residual parameter"
    elif displacement > settings.maximum_residual_displacement_pixels:
        reason = "residual displacement exceeds formal hard limit"
    elif roll > settings.maximum_background_roll_degrees:
        reason = "residual roll exceeds formal hard limit"
    elif per_source_margin is not None and any(
        displacement_at_bounds > margin
        for displacement_at_bounds, margin in zip(
            per_warp_displacement, per_source_margin, strict=True
        )
    ):
        reason = "residual displacement exceeds calibrated support spare"
    return WarpTopologyAudit(
        source_count=len(checked),
        finite=finite,
        maximum_displacement_pixels=float(displacement),
        maximum_roll_degrees=float(roll),
        support_margin_pixels=minimum_margin,
        inverse_roundtrip_error_pixels=0.0 if finite else None,
        accepted=reason is None,
        failure_reason=reason,
    )


def _preview_correspondences(
    evidence: PairResidualEvidence,
    *,
    preview_origin_x: float,
    preview_scale: float,
    held_out: bool,
    maximum_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Lift accepted preview flow samples into full virtual-canvas coordinates."""

    scale = _finite_float(preview_scale, "preview_scale")
    if not 0.0 < scale <= 1.0:
        raise ValueError("preview_scale must be in (0, 1]")
    accepted = np.asarray(evidence.accepted_mask, dtype=bool)
    partition = np.asarray(evidence.held_out_mask, dtype=bool)
    mask = accepted & (partition if held_out else ~partition)
    positions = np.flatnonzero(mask.ravel())
    if positions.size > maximum_samples:
        # Deterministic spreading prevents a dense textured tile from taking
        # over the global solution while retaining the held-out separation.
        select = np.linspace(0, positions.size - 1, maximum_samples, dtype=np.int64)
        positions = positions[select]
    if not positions.size:
        return (
            np.empty((0, 2), dtype=np.float64),
            np.empty((0, 2), dtype=np.float64),
            np.empty(0, dtype=np.float64),
        )
    yy, xx = np.unravel_index(positions, mask.shape)
    flow = np.asarray(evidence.forward_flow, dtype=np.float64)[yy, xx]
    fb = np.asarray(evidence.forward_backward_error, dtype=np.float64)[yy, xx]
    pi = np.column_stack(
        (
            (xx.astype(np.float64) + preview_origin_x + 0.5) / scale - 0.5,
            (yy.astype(np.float64) + 0.5) / scale - 0.5,
        )
    )
    qj = np.column_stack(
        (
            (xx.astype(np.float64) + flow[:, 0] + preview_origin_x + 0.5) / scale
            - 0.5,
            (yy.astype(np.float64) + flow[:, 1] + 0.5) / scale - 0.5,
        )
    )
    finite = np.isfinite(pi).all(axis=1) & np.isfinite(qj).all(axis=1) & np.isfinite(fb)
    return pi[finite], qj[finite], fb[finite]


def _append_normal_row(
    matrix: np.ndarray,
    vector: np.ndarray,
    indices: Sequence[int],
    coefficients: Sequence[float],
    target: float,
    weight: float,
) -> None:
    """Accumulate one compact weighted least-squares row into normal equations."""

    checked_indices = np.asarray(indices, dtype=np.intp)
    checked_coefficients = np.asarray(coefficients, dtype=np.float64)
    if not math.isfinite(weight) or weight <= 0.0:
        return
    outer = weight * np.outer(checked_coefficients, checked_coefficients)
    matrix[np.ix_(checked_indices, checked_indices)] += outer
    vector[checked_indices] += weight * checked_coefficients * float(target)


def _row_residual(
    solution: np.ndarray,
    indices: Sequence[int],
    coefficients: Sequence[float],
    target: float,
) -> float:
    return float(np.dot(solution[np.asarray(indices, dtype=np.intp)], coefficients) - target)


def _background_held_out_metrics(
    evidence: Sequence[PairResidualEvidence],
    warps: Sequence[SourceResidualWarp],
    pair_preview_origins: Sequence[tuple[float, float]],
) -> dict[str, float | int | None]:
    """Evaluate selected warps only on correspondences excluded from the solve."""

    before: list[np.ndarray] = []
    after: list[np.ndarray] = []
    for index, item in enumerate(evidence):
        origin, scale = pair_preview_origins[index]
        pi, qj, _ = _preview_correspondences(
            item,
            preview_origin_x=origin,
            preview_scale=scale,
            held_out=True,
            maximum_samples=512,
        )
        if not len(pi):
            continue
        before.append(np.linalg.norm(qj - pi, axis=1))
        first_x, first_y = warps[index].forward_virtual_coordinates(pi[:, 0], pi[:, 1])
        second_x, second_y = warps[index + 1].forward_virtual_coordinates(
            qj[:, 0], qj[:, 1]
        )
        after.append(np.hypot(first_x - second_x, first_y - second_y))
    before_values = np.concatenate(before) if before else np.empty(0, dtype=np.float64)
    after_values = np.concatenate(after) if after else np.empty(0, dtype=np.float64)
    return {
        "held_out_background_count": int(after_values.size),
        "held_out_background_residual_p50_before_pixels": _finite_percentile(
            before_values, 50.0
        ),
        "held_out_background_residual_p95_before_pixels": _finite_percentile(
            before_values, 95.0
        ),
        "held_out_background_residual_max_before_pixels": (
            float(np.max(before_values)) if before_values.size else None
        ),
        "held_out_background_residual_p50_after_pixels": _finite_percentile(
            after_values, 50.0
        ),
        "held_out_background_residual_p95_after_pixels": _finite_percentile(
            after_values, 95.0
        ),
        "held_out_background_residual_max_after_pixels": (
            float(np.max(after_values)) if after_values.size else None
        ),
    }


def solve_background_se2(
    evidence: Sequence[PairResidualEvidence],
    *,
    source_centres: Sequence[tuple[float, float]],
    pair_preview_origins: Sequence[tuple[float, float]],
    config: ResidualAlignmentConfig | Mapping[str, object] | None = None,
    support_margins_pixels: Sequence[float] | None = None,
    support_bounds: Sequence[tuple[float, float, float, float]] | None = None,
) -> tuple[tuple[SourceResidualWarp, ...] | None, dict[str, float | int | str | bool | None], WarpTopologyAudit | None]:
    """Solve one low-DoF global background residual from training evidence.

    The variables are a translation and tiny roll for each *real* source.  All
    adjacent observations participate at once; pair warps are never chained or
    accumulated.  The deterministic held-out partition is excluded from this
    solve and is used only for the returned quality metrics.

    ``None`` is an intentional fail-closed result: insufficient texture,
    missing adjacent evidence, ill-conditioned geometry, exceeded support, or
    weak held-out improvement leaves the renderer in identity/owner-only mode.
    """

    settings = (
        config
        if isinstance(config, ResidualAlignmentConfig)
        else ResidualAlignmentConfig.from_mapping(config)
    )
    centres = tuple((float(x), float(y)) for x, y in source_centres)
    source_count = len(centres)
    metrics: dict[str, float | int | str | bool | None] = {
        "solver": "global_huber_irls_se2",
        "selected": False,
        "reason": None,
        "training_correspondence_count": 0,
        "held_out_background_count": 0,
    }
    if settings.background_model != "se2":
        metrics["reason"] = "background_model_identity"
        return None, metrics, None
    if source_count < 2 or len(evidence) != source_count - 1:
        metrics["reason"] = "incomplete_adjacent_evidence"
        return None, metrics, None
    if len(pair_preview_origins) != len(evidence):
        raise ValueError("Pair preview origins must cover every adjacent evidence pair")
    if support_margins_pixels is None:
        margins = (float("inf"),) * source_count
    else:
        margins = tuple(float(value) for value in support_margins_pixels)
        if len(margins) != source_count or not np.isfinite(margins).all():
            raise ValueError("Residual support margins must match all source centres")
    dimensions = 3 * source_count
    # Each data row has at most six non-zero coefficients.  Keep compact rows
    # rather than a potentially 160 x 800 x 480 dense matrix.
    rows: list[tuple[tuple[int, ...], tuple[float, ...], float, float]] = []
    pair_training_counts: list[int] = []
    held_out_edge_p95: list[float] = []
    held_out_edge_max: list[float] = []
    for pair_index, item in enumerate(evidence):
        edge_mask = (
            np.asarray(item.accepted_mask, dtype=bool)
            & np.asarray(item.held_out_mask, dtype=bool)
        )
        edge_values = np.asarray(item.edge_normal_step_pixels, dtype=np.float64)[
            edge_mask
        ]
        edge_values = edge_values[np.isfinite(edge_values)]
        if not edge_values.size:
            metrics["reason"] = f"no_held_out_edge_support_pair_{pair_index}"
            return None, metrics, None
        edge_p95 = float(np.percentile(edge_values, 95.0))
        edge_maximum = float(np.max(edge_values))
        held_out_edge_p95.append(edge_p95)
        held_out_edge_max.append(edge_maximum)
        if edge_p95 > settings.maximum_edge_step_p95_pixels:
            metrics["reason"] = f"held_out_edge_p95_exceeds_limit_pair_{pair_index}"
            metrics["held_out_edge_p95_pixels"] = edge_p95
            return None, metrics, None
        if edge_maximum > settings.maximum_edge_step_pixels:
            metrics["reason"] = f"held_out_edge_max_exceeds_limit_pair_{pair_index}"
            metrics["held_out_edge_max_pixels"] = edge_maximum
            return None, metrics, None
        origin, scale = pair_preview_origins[pair_index]
        pi, qj, fb = _preview_correspondences(
            item,
            preview_origin_x=origin,
            preview_scale=scale,
            held_out=False,
            maximum_samples=128,
        )
        # Rotation is unobservable for a line/one-tile correspondence cloud.
        if len(pi) < 24:
            metrics["reason"] = f"insufficient_training_support_pair_{pair_index}"
            return None, metrics, None
        covariance = np.cov(pi.T)
        if covariance.shape != (2, 2) or np.min(np.linalg.eigvalsh(covariance)) < 2.0:
            metrics["reason"] = f"rank_deficient_background_pair_{pair_index}"
            return None, metrics, None
        pair_training_counts.append(int(len(pi)))
        source0, source1 = pair_index, pair_index + 1
        centre0, centre1 = centres[source0], centres[source1]
        # FB-consistent pixels have a bounded but non-zero robust confidence.
        weights = np.clip(1.0 / np.maximum(0.10, 0.25 + fb), 0.20, 4.0)
        for point0, point1, weight in zip(pi, qj, weights, strict=True):
            px, py = point0
            qx, qy = point1
            rows.append(
                (
                    (3 * source0, 3 * source0 + 2, 3 * source1, 3 * source1 + 2),
                    (1.0, -(py - centre0[1]), -1.0, qy - centre1[1]),
                    float(qx - px),
                    float(weight),
                )
            )
            rows.append(
                (
                    (3 * source0 + 1, 3 * source0 + 2, 3 * source1 + 1, 3 * source1 + 2),
                    (1.0, px - centre0[0], -1.0, -(qx - centre1[0])),
                    float(qy - py),
                    float(weight),
                )
            )
    metrics["training_correspondence_count"] = int(sum(pair_training_counts))
    metrics["training_pair_min_count"] = min(pair_training_counts, default=0)
    metrics["held_out_edge_step_p95_maximum_pixels"] = max(
        held_out_edge_p95, default=None
    )
    metrics["held_out_edge_step_maximum_pixels"] = max(
        held_out_edge_max, default=None
    )
    # Endpoint outward calibrated support has no spare beyond the image edge.
    # Pin any source with no geometric margin to identity; intermediate sources
    # still solve globally against those real-pose anchors.
    fixed_sources = {0}
    fixed_sources.update(index for index, margin in enumerate(margins) if margin <= 1e-9)
    solution = np.zeros(dimensions, dtype=np.float64)
    condition = math.inf
    for _iteration in range(4):
        normal = np.zeros((dimensions, dimensions), dtype=np.float64)
        target = np.zeros(dimensions, dtype=np.float64)
        for indices, coefficients, value, base_weight in rows:
            residual = _row_residual(solution, indices, coefficients, value)
            huber = min(1.0, 1.5 / max(abs(residual), 1e-9))
            _append_normal_row(
                normal,
                target,
                indices,
                coefficients,
                value,
                base_weight * huber,
            )
        # A light second-difference prior damps a single false RGB match while
        # preserving affine-in-time motion (whose second derivative is zero).
        for index in range(1, source_count - 1):
            for component, regularization in ((0, 0.04), (1, 0.04), (2, 6.0)):
                _append_normal_row(
                    normal,
                    target,
                    (3 * (index - 1) + component, 3 * index + component, 3 * (index + 1) + component),
                    (1.0, -2.0, 1.0),
                    0.0,
                    regularization,
                )
        for source in fixed_sources:
            for component in range(3):
                _append_normal_row(
                    normal,
                    target,
                    (3 * source + component,),
                    (1.0,),
                    0.0,
                    1e8,
                )
        try:
            condition = float(np.linalg.cond(normal))
            if not math.isfinite(condition) or condition > 1e10:
                metrics["reason"] = "ill_conditioned_background_solver"
                metrics["solver_condition"] = condition
                return None, metrics, None
            solution = np.linalg.solve(normal, target)
        except np.linalg.LinAlgError:
            metrics["reason"] = "singular_background_solver"
            return None, metrics, None
        if not np.isfinite(solution).all():
            metrics["reason"] = "non_finite_background_solver"
            return None, metrics, None
    warps = tuple(
        SourceResidualWarp(
            source_index=index,
            translation_x=float(solution[3 * index]),
            translation_y=float(solution[3 * index + 1]),
            roll_degrees=float(math.degrees(solution[3 * index + 2])),
            centre_x=centres[index][0],
            centre_y=centres[index][1],
        )
        for index in range(source_count)
    )
    topology = audit_source_warps(
        warps,
        settings,
        support_margin_pixels=margins,
        support_bounds=support_bounds,
    )
    if not topology.accepted:
        metrics["reason"] = topology.failure_reason or "background_topology_audit_failed"
        return None, metrics, topology
    held_out = _background_held_out_metrics(evidence, warps, pair_preview_origins)
    metrics.update(held_out)
    metrics["solver_condition"] = condition
    before = held_out["held_out_background_residual_p95_before_pixels"]
    after = held_out["held_out_background_residual_p95_after_pixels"]
    if before is None or after is None:
        metrics["reason"] = "no_held_out_background_support"
        return None, metrics, topology
    before_value = float(before)
    after_value = float(after)
    if after_value > 2.0:
        metrics["reason"] = "held_out_background_p95_above_2px"
        return None, metrics, topology
    if before_value > 2.0 and after_value > 0.70 * before_value:
        metrics["reason"] = "held_out_background_improvement_below_30_percent"
        return None, metrics, topology
    if before_value <= 2.0 and after_value > before_value + 0.25:
        metrics["reason"] = "held_out_background_regression_above_0_25px"
        return None, metrics, topology
    # Do not disturb an already-aligned panorama merely because the allowed
    # degradation threshold permits it: selection itself requires an observed
    # improvement, while the 0.25 px bound protects a later forced diagnostic.
    if after_value >= before_value - 0.05:
        metrics["reason"] = "no_material_held_out_background_improvement"
        return None, metrics, topology
    metrics["selected"] = True
    metrics["reason"] = "held_out_background_improvement"
    return warps, metrics, topology
