from __future__ import annotations

import numpy as np
import pytest

from panorama_demo.foreground_segments import (
    DepthAnchorToken,
    GeometryMode,
    RawFootprintSummary,
    build_foreground_fragments,
    foreground_fragment_from_protected,
    plan_foreground_owners,
)
from panorama_demo.rgb_residual_alignment import (
    ProtectedComponentFragment,
    preflight_sequence_owners,
)


def _protected(
    pair_index: int,
    label: int,
    *,
    x: int = 0,
    allowed: tuple[int, ...] = (0, 1),
    preferred: int = 0,
) -> ProtectedComponentFragment:
    mask = np.ones((8, 12), dtype=bool)
    return ProtectedComponentFragment(
        pair_index=pair_index,
        global_bbox=(x, 20, mask.shape[1], mask.shape[0]),
        local_mask=mask,
        allowed_owners=allowed,
        coverage_margin=3.0,
        edge_orientation=4.0,
        component_label=label,
        preferred_owner=preferred,
    )


def _footprint(source_index: int, *, x0: float) -> RawFootprintSummary:
    grid_y, grid_x = np.mgrid[0:10, 0:12]
    return RawFootprintSummary.from_source_coordinates(
        source_index=source_index,
        source_size=(160, 120),
        map_x=x0 + grid_x.astype(np.float64),
        map_y=30.0 + grid_y.astype(np.float64),
        mask=np.ones(grid_x.shape, dtype=bool),
    )


def _anchor_token(
    *,
    left_pair_index: int = 0,
    left_direct_component_label: int = 3,
    right_direct_component_label: int = 5,
) -> DepthAnchorToken:
    return DepthAnchorToken(
        shared_source_index=left_pair_index + 1,
        left_pair_index=left_pair_index,
        right_pair_index=left_pair_index + 1,
        left_direct_component_label=left_direct_component_label,
        right_direct_component_label=right_direct_component_label,
    )


def _depth_fragment(
    pair_index: int,
    label: int,
    *,
    footprints: tuple[RawFootprintSummary, RawFootprintSummary],
) -> object:
    return foreground_fragment_from_protected(
        _protected(pair_index, label, x=pair_index * 18),
        frame_ids=(100 + pair_index, 101 + pair_index),
        source_indices=(pair_index, pair_index + 1),
        geometry_mode=GeometryMode.DEPTH_OBSERVED,
        bidirectional_visibility_supported=True,
        raw_footprints=footprints,
    )


def test_depth_observed_three_pair_chain_uses_deterministic_spans_not_a_fake_handoff() -> None:
    shared_one = _footprint(1, x0=20.0)
    shared_two = _footprint(2, x0=70.0)
    fragments = (
        (
            _depth_fragment(
                0,
                1,
                footprints=(_footprint(0, x0=5.0), shared_one),
            ),
        ),
        (
            _depth_fragment(
                1,
                1,
                footprints=(shared_one, shared_two),
            ),
        ),
        (
            _depth_fragment(
                2,
                1,
                footprints=(shared_two, _footprint(3, x0=120.0)),
            ),
        ),
    )

    plan = plan_foreground_owners(fragments)

    assert plan.accepted
    # Equal-cost links are deterministically resolved toward the earlier span.
    assert plan.component_owner_constraints[0] == {1: 1}
    assert plan.component_owner_constraints[1] == {1: 0}
    assert plan.component_owner_constraints[2] == {}
    assert len(plan.segments) == 1
    assert len(plan.spans) == 2
    assert len(plan.handoffs) == 1
    assert plan.handoffs[0].accepted is False
    assert plan.handoffs[0].reason == "continuous_foreground_requires_unapproved_handoff"


def test_disjoint_shared_raw_footprints_do_not_create_a_cross_pair_segment() -> None:
    first = _depth_fragment(
        0,
        1,
        footprints=(_footprint(0, x0=5.0), _footprint(1, x0=10.0)),
    )
    second = _depth_fragment(
        1,
        1,
        footprints=(_footprint(1, x0=110.0), _footprint(2, x0=130.0)),
    )

    plan = plan_foreground_owners(((first,), (second,)))

    assert plan.component_owner_constraints == ({}, {})
    assert plan.rejected_association_counts["shared_raw_footprint_disjoint"] == 1
    assert len(plan.segments) == 2
    assert len(plan.spans) == 2


def test_depth_observed_shared_source_anchor_locks_each_pair_fragment() -> None:
    """Depth evidence, not pair-local preference, chooses a hose anchor.

    The two pair-local components prefer opposite outer RGB sources.  Their
    only common source is frame 101 and the raw source-coordinate footprint
    proves that it sees the same foreground instance in both pair corridors.
    A valid anchor must therefore lock pair 0 to its second local owner and
    pair 1 to its first local owner; the legacy bbox preflight cannot rewrite
    either decision.
    """

    first_protected = _protected(0, 7, x=36, preferred=0)
    second_protected = _protected(1, 13, x=54, preferred=1)
    shared = _footprint(1, x0=42.0)
    first = foreground_fragment_from_protected(
        first_protected,
        frame_ids=(100, 101),
        source_indices=(0, 1),
        geometry_mode=GeometryMode.DEPTH_OBSERVED,
        bidirectional_visibility_supported=True,
        depth_anchor_local_mask=np.ones(first_protected.local_mask.shape, dtype=bool),
        depth_anchor_token=_anchor_token(),
        raw_footprints=(_footprint(0, x0=12.0), shared),
    )
    second = foreground_fragment_from_protected(
        second_protected,
        frame_ids=(101, 102),
        source_indices=(1, 2),
        geometry_mode=GeometryMode.DEPTH_OBSERVED,
        bidirectional_visibility_supported=True,
        depth_anchor_local_mask=np.ones(second_protected.local_mask.shape, dtype=bool),
        depth_anchor_token=_anchor_token(),
        raw_footprints=(shared, _footprint(2, x0=94.0)),
    )

    plan = plan_foreground_owners(((first,), (second,)))

    assert plan.accepted
    assert plan.component_owner_constraints == ({7: 1}, {13: 0})
    assert len(plan.spans) == 1
    span = plan.spans[0]
    assert span.fragment_refs == ((0, 7), (1, 13))
    assert span.anchor_source_index == 1
    assert span.anchor_frame_id == 101
    assert all(candidate.complete_coverage for candidate in span.anchor_candidates)
    preflight = preflight_sequence_owners(
        ((first_protected,), (second_protected,)),
        locked_component_owner_constraints=plan.component_owner_constraints,
    )
    assert preflight.accepted
    assert preflight.component_owner_constraints == ({7: 1}, {13: 0})


def test_different_sparse_anchor_tokens_cannot_link_on_coarse_footprint_overlap() -> None:
    """A common 32x32 footprint is audit only, never token identity."""

    first_protected = _protected(0, 7, x=36)
    second_protected = _protected(1, 13, x=54)
    shared = _footprint(1, x0=42.0)
    first = foreground_fragment_from_protected(
        first_protected,
        frame_ids=(100, 101),
        source_indices=(0, 1),
        geometry_mode=GeometryMode.DEPTH_OBSERVED,
        bidirectional_visibility_supported=True,
        depth_anchor_local_mask=np.ones(first_protected.local_mask.shape, dtype=bool),
        depth_anchor_token=_anchor_token(left_direct_component_label=3),
        raw_footprints=(_footprint(0, x0=12.0), shared),
    )
    second = foreground_fragment_from_protected(
        second_protected,
        frame_ids=(101, 102),
        source_indices=(1, 2),
        geometry_mode=GeometryMode.DEPTH_OBSERVED,
        bidirectional_visibility_supported=True,
        depth_anchor_local_mask=np.ones(second_protected.local_mask.shape, dtype=bool),
        depth_anchor_token=_anchor_token(right_direct_component_label=11),
        raw_footprints=(shared, _footprint(2, x0=94.0)),
    )

    plan = plan_foreground_owners(((first,), (second,)))

    assert plan.component_owner_constraints == ({}, {})
    assert plan.rejected_association_counts["source_anchor_token_mismatch"] == 1
    assert len(plan.spans) == 2


def test_aligned_only_adapter_stays_image_region_owner_only() -> None:
    protected = ((_protected(0, 1),), (_protected(1, 1),))

    fragments = build_foreground_fragments(
        protected,
        frame_ids=(10, 11, 12),
    )
    plan = plan_foreground_owners(fragments)

    assert plan.geometry_mode_counts[GeometryMode.IMAGE_REGION.value] == 2
    assert plan.component_owner_constraints == ({}, {})
    assert len(plan.segments) == 2
    assert not plan.handoffs


def test_component_depth_override_does_not_promote_neighbouring_rgb_guard() -> None:
    first = _protected(0, 1, x=0)
    neighbouring_rgb_guard = _protected(0, 2, x=18)
    first_footprint = _footprint(0, x0=5.0)
    second_footprint = _footprint(1, x0=25.0)

    fragments = build_foreground_fragments(
        ((first, neighbouring_rgb_guard),),
        frame_ids=(10, 11),
        geometry_mode_overrides={(0, 1): GeometryMode.DEPTH_OBSERVED},
        bidirectional_visibility_overrides={(0, 1): True},
        raw_footprints={
            (0, 1, 0): first_footprint,
            (0, 1, 1): second_footprint,
        },
    )

    by_label = {fragment.component_label: fragment for fragment in fragments[0]}
    assert by_label[1].geometry_mode is GeometryMode.DEPTH_OBSERVED
    assert by_label[1].bidirectional_visibility_supported
    assert by_label[2].geometry_mode is GeometryMode.IMAGE_REGION
    assert not by_label[2].bidirectional_visibility_supported


def test_depth_anchor_mask_is_audited_and_limited_to_its_protected_component() -> None:
    protected = _protected(0, 1)
    anchor = np.zeros(protected.local_mask.shape, dtype=bool)
    anchor[2:5, 4:9] = True

    fragment = build_foreground_fragments(
        ((protected,),),
        frame_ids=(10, 11),
        depth_anchor_masks={(0, 1): anchor},
        depth_anchor_tokens={(0, 1): _anchor_token()},
    )[0][0]

    assert fragment.anchor_pixel_count == 15
    assert np.array_equal(fragment.depth_anchor_local_mask, anchor)
    assert fragment.as_dict()["anchor_pixel_count"] == 15


@pytest.mark.parametrize(
    "anchor",
    (
        np.ones((4, 5), dtype=bool),
        np.zeros((8, 12), dtype=bool),
        np.pad(np.ones((1, 1), dtype=bool), ((0, 7), (12, 0))),
    ),
)
def test_depth_anchor_mask_must_be_a_nonempty_local_mask_subset(anchor: np.ndarray) -> None:
    with pytest.raises(ValueError, match="depth anchor mask"):
        build_foreground_fragments(
            ((_protected(0, 1),),),
            frame_ids=(10, 11),
            depth_anchor_masks={(0, 1): anchor},
            depth_anchor_tokens={(0, 1): _anchor_token()},
        )


def test_locked_segment_owner_constraint_cannot_be_overwritten_by_legacy_bbox_track() -> None:
    first = _protected(0, 1, preferred=1)
    second = _protected(1, 1, preferred=0)

    preflight = preflight_sequence_owners(
        ((first,), (second,)),
        locked_component_owner_constraints=({1: 0}, {}),
    )

    assert preflight.accepted
    assert preflight.component_owner_constraints[0] == {1: 0}
    assert preflight.component_owner_constraints[1] == {1: 0}
    assert preflight.component_tracks == ()
    assert preflight.rejected_track_count == 1


def test_raw_footprint_audit_is_scalar_only() -> None:
    footprint = _footprint(3, x0=40.0)

    audit = footprint.as_dict()

    assert "occupancy" not in audit
    assert audit["occupied_cell_count"] > 0
    assert footprint.overlap_iou(footprint) == 1.0
