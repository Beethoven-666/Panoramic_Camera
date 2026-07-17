from __future__ import annotations

import numpy as np

from panorama_demo.foreground_segments import (
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
