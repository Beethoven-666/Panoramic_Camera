from __future__ import annotations

import cv2
import numpy as np
import pytest

import panorama_demo.rgb_residual_alignment as residual_alignment_module
from panorama_demo.rgb_residual_alignment import (
    ResidualAlignmentConfig,
    ProtectedComponentFragment,
    SourceResidualWarp,
    audit_source_warps,
    build_held_out_partition,
    extract_pair_evidence,
    measure_owner_boundary_geometry,
    preflight_sequence_owners,
    solve_background_se2,
)


def _translated_edge_preview(offset: int) -> tuple[np.ndarray, np.ndarray]:
    image = np.full((96, 128, 3), 190, dtype=np.uint8)
    cv2.rectangle(image, (28, 20), (92, 76), (20, 20, 20), thickness=cv2.FILLED)
    cv2.line(image, (18, 84), (106, 12), (72, 72, 72), thickness=2)
    translated = cv2.warpAffine(
        image,
        np.array(((1.0, 0.0, float(offset)), (0.0, 1.0, 0.0)), dtype=np.float32),
        (image.shape[1], image.shape[0]),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return image, translated


def test_held_out_partition_is_deterministic_and_pair_specific() -> None:
    first = build_held_out_partition((80, 120), seed=7, frame_ids=(10, 11))
    again = build_held_out_partition((80, 120), seed=7, frame_ids=(10, 11))
    other = build_held_out_partition((80, 120), seed=7, frame_ids=(11, 12))

    assert np.array_equal(first, again)
    assert not np.array_equal(first, other)
    assert 0.16 < float(np.mean(first)) < 0.24


@pytest.mark.parametrize("offset", (0, 1, 2, 4))
def test_edge_step_metric_tracks_known_preview_offset(offset: int) -> None:
    first, second = _translated_edge_preview(offset)
    valid = np.ones(first.shape[:2], dtype=bool)
    evidence = extract_pair_evidence(
        first_rgb=first,
        second_rgb=second,
        first_valid=valid,
        second_valid=valid,
        frame_ids=(3, 4),
    )

    measured = evidence.metrics["edge_normal_step_p95_pixels"]
    assert measured is not None
    assert abs(float(measured) - offset) <= 0.6
    geometry = measure_owner_boundary_geometry(evidence, boundary_x=28)
    assert geometry["unobservable"] is False


@pytest.mark.parametrize("value", (0, 255))
def test_flat_white_or_black_preview_is_explicitly_unobservable(value: int) -> None:
    image = np.full((64, 80, 3), value, dtype=np.uint8)
    valid = np.ones(image.shape[:2], dtype=bool)
    evidence = extract_pair_evidence(
        first_rgb=image,
        second_rgb=image.copy(),
        first_valid=valid,
        second_valid=valid,
    )

    assert evidence.metrics["unobservable"] is True
    assert evidence.accepted_count == 0
    assert evidence.metrics["edge_normal_step_p95_pixels"] is None


def test_missing_common_rgb_coverage_never_becomes_zero_error() -> None:
    first, second = _translated_edge_preview(2)
    valid0 = np.ones(first.shape[:2], dtype=bool)
    valid1 = valid0.copy()
    valid1[:, 40:80] = False
    evidence = extract_pair_evidence(
        first_rgb=first,
        second_rgb=second,
        first_valid=valid0,
        second_valid=valid1,
    )

    assert evidence.candidate_count == int(np.count_nonzero(valid0 & valid1))
    assert not np.any(evidence.accepted_mask[:, 40:80])


def test_flow_target_in_invalid_second_preview_never_becomes_accepted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A texture sample is invalid when its forward-flow target is invalid."""

    rng = np.random.default_rng(17)
    image = rng.integers(0, 256, size=(64, 96, 3), dtype=np.uint8)

    class _FixedHorizontalDIS:
        def __init__(self) -> None:
            self.calls = 0

        def setFinestScale(self, _: int) -> None:
            pass

        def calc(
            self, first: np.ndarray, second: np.ndarray, _: object
        ) -> np.ndarray:
            del second
            flow = np.zeros((*first.shape, 2), dtype=np.float32)
            flow[:, :, 0] = 5.0 if self.calls == 0 else -5.0
            self.calls += 1
            return flow

    dis = _FixedHorizontalDIS()
    monkeypatch.setattr(
        residual_alignment_module.cv2,
        "DISOpticalFlow_create",
        lambda *_: dis,
    )
    valid0 = np.ones(image.shape[:2], dtype=bool)
    valid1 = valid0.copy()
    # Starts at x=25..29 are valid in both previews, but their +5 flow target
    # falls into this invalid second-preview hole.
    valid1[:, 30:50] = False
    evidence = extract_pair_evidence(
        first_rgb=image,
        second_rgb=image.copy(),
        first_valid=valid0,
        second_valid=valid1,
        frame_ids=(7, 8),
    )

    assert np.any(evidence.accepted_mask[:, :20])
    assert not np.any(evidence.accepted_mask[:, 25:30])


def test_dis_failure_marks_all_candidates_unaccepted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A DIS exception is unknown flow evidence, never a zero-flow success."""

    rng = np.random.default_rng(23)
    image = rng.integers(0, 256, size=(64, 96, 3), dtype=np.uint8)

    def _raise_dis_error(*_: object) -> object:
        raise residual_alignment_module.cv2.error("forced DIS failure")

    monkeypatch.setattr(
        residual_alignment_module.cv2,
        "DISOpticalFlow_create",
        _raise_dis_error,
    )
    valid = np.ones(image.shape[:2], dtype=bool)
    evidence = extract_pair_evidence(
        first_rgb=image,
        second_rgb=image.copy(),
        first_valid=valid,
        second_valid=valid,
        frame_ids=(9, 10),
    )

    assert np.any(evidence.observable_mask)
    assert evidence.accepted_count == 0
    assert np.all(evidence.flow_uncertain_mask[valid])


def test_invalid_black_hole_cannot_supply_accepted_texture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The black invalid border itself cannot become a false RGB feature."""

    class _ZeroFlowDIS:
        def setFinestScale(self, _: int) -> None:
            pass

        def calc(
            self, first: np.ndarray, second: np.ndarray, _: object
        ) -> np.ndarray:
            del second
            return np.zeros((*first.shape, 2), dtype=np.float32)

    monkeypatch.setattr(
        residual_alignment_module.cv2,
        "DISOpticalFlow_create",
        lambda *_: _ZeroFlowDIS(),
    )
    image = np.full((64, 96, 3), 180, dtype=np.uint8)
    image[20:44, 30:58] = 0
    valid = np.ones(image.shape[:2], dtype=bool)
    valid[20:44, 30:58] = False
    evidence = extract_pair_evidence(
        first_rgb=image,
        second_rgb=image.copy(),
        first_valid=valid,
        second_valid=valid,
        frame_ids=(11, 12),
    )

    assert evidence.candidate_count == int(np.count_nonzero(valid))
    assert evidence.accepted_count == 0
    assert not np.any(evidence.accepted_mask)


def test_source_residual_warp_is_inverse_and_bounded_by_topology_audit() -> None:
    warp = SourceResidualWarp(
        source_index=2,
        translation_x=2.5,
        translation_y=-1.0,
        roll_degrees=0.2,
        centre_x=64.0,
        centre_y=48.0,
    )
    source_x = np.array(((31.0, 57.0), (89.0, 102.0)), dtype=np.float64)
    source_y = np.array(((13.0, 44.0), (70.0, 82.0)), dtype=np.float64)
    angle = np.deg2rad(warp.roll_degrees)
    cos_angle, sin_angle = np.cos(angle), np.sin(angle)
    dx, dy = source_x - warp.centre_x, source_y - warp.centre_y
    output_x = warp.centre_x + cos_angle * dx - sin_angle * dy + warp.translation_x
    output_y = warp.centre_y + sin_angle * dx + cos_angle * dy + warp.translation_y
    recovered_x, recovered_y = warp.inverse_virtual_coordinates(output_x, output_y)

    np.testing.assert_allclose(recovered_x, source_x, atol=1e-10)
    np.testing.assert_allclose(recovered_y, source_y, atol=1e-10)
    audit = audit_source_warps((warp,), support_margin_pixels=3.0)
    assert audit.accepted is True
    rejected = audit_source_warps(
        (SourceResidualWarp(translation_x=9.0),), support_margin_pixels=10.0
    )
    assert rejected.accepted is False


def test_roll_deformation_cannot_consume_calibrated_support_spare() -> None:
    """Support auditing measures the full corner displacement, not translation only."""

    audit = audit_source_warps(
        (
            SourceResidualWarp(
                translation_x=0.0,
                translation_y=0.0,
                roll_degrees=0.25,
                centre_x=0.0,
                centre_y=0.0,
            ),
        ),
        support_margin_pixels=0.1,
        support_bounds=((0.0, 200.0, 0.0, 0.0),),
    )

    assert audit.accepted is False
    assert audit.maximum_displacement_pixels > 0.8
    assert audit.failure_reason == "residual displacement exceeds calibrated support spare"


def test_config_is_closed_and_rejects_projective_or_depth_backends() -> None:
    with pytest.raises(ValueError, match="Unknown residual_alignment"):
        ResidualAlignmentConfig.from_mapping({"unknown": True})
    with pytest.raises(ValueError, match="forbids"):
        ResidualAlignmentConfig.from_mapping({"backend": "homography"})


def test_background_model_defaults_to_identity() -> None:
    """The formal renderer must never opt into a global RGB SE(2) warp."""

    assert ResidualAlignmentConfig().background_model == "identity"
    warps, metrics, audit = solve_background_se2(
        (),
        source_centres=((64.0, 48.0), (64.0, 48.0)),
        pair_preview_origins=(),
    )

    assert warps is None
    assert audit is None
    assert metrics["selected"] is False
    assert metrics["solver"] == "identity_not_run"
    assert metrics["reason"] == "background_model_identity"


def test_global_background_se2_uses_training_only_and_improves_held_out() -> None:
    """The retained solver utility requires an explicit historical opt-in."""

    images = [_translated_edge_preview(offset)[1] for offset in range(5)]
    valid = np.ones(images[0].shape[:2], dtype=bool)
    evidence = tuple(
        extract_pair_evidence(
            first_rgb=images[index],
            second_rgb=images[index + 1],
            first_valid=valid,
            second_valid=valid,
            pair_index=index,
            frame_ids=(index, index + 1),
        )
        for index in range(len(images) - 1)
    )
    warps, metrics, audit = solve_background_se2(
        evidence,
        source_centres=tuple((64.0, 48.0) for _ in images),
        pair_preview_origins=tuple((0.0, 1.0) for _ in evidence),
        support_margins_pixels=tuple(8.0 for _ in images),
        config=ResidualAlignmentConfig(background_model="se2"),
    )

    assert warps is not None
    assert audit is not None and audit.accepted is True
    assert metrics["selected"] is True
    assert float(metrics["held_out_background_residual_p95_after_pixels"]) < 0.5 * float(
        metrics["held_out_background_residual_p95_before_pixels"]
    )


def test_global_background_se2_rejects_held_out_edge_step_above_formal_limit() -> None:
    """A model cannot be selected when held-out edge displacement is unsafe."""

    images = [_translated_edge_preview(offset)[1] for offset in range(5)]
    valid = np.ones(images[0].shape[:2], dtype=bool)
    evidence = tuple(
        extract_pair_evidence(
            first_rgb=images[index],
            second_rgb=images[index + 1],
            first_valid=valid,
            second_valid=valid,
            pair_index=index,
            frame_ids=(index, index + 1),
        )
        for index in range(len(images) - 1)
    )
    warps, metrics, audit = solve_background_se2(
        evidence,
        source_centres=tuple((64.0, 48.0) for _ in images),
        pair_preview_origins=tuple((0.0, 1.0) for _ in evidence),
        support_margins_pixels=tuple(8.0 for _ in images),
        config=ResidualAlignmentConfig(
            background_model="se2",
            maximum_edge_step_p95_pixels=0.5,
        ),
    )

    assert warps is None
    assert audit is None
    assert str(metrics["reason"]).startswith("held_out_edge_p95_exceeds_limit")


def test_owner_preflight_keeps_a_unique_cross_pair_component_on_shared_source() -> None:
    fragment0 = ProtectedComponentFragment(
        pair_index=0,
        global_bbox=(40, 12, 16, 8),
        local_mask=np.ones((8, 16), dtype=bool),
        component_label=1,
        preferred_owner=0,
    )
    fragment1 = ProtectedComponentFragment(
        pair_index=1,
        global_bbox=(42, 12, 16, 8),
        local_mask=np.ones((8, 16), dtype=bool),
        component_label=1,
        preferred_owner=1,
    )

    preflight = preflight_sequence_owners(((fragment0,), (fragment1,)))

    assert preflight.accepted is True
    assert len(preflight.component_tracks) == 1
    # Pair 0's second owner and pair 1's first owner are the shared source.
    assert preflight.component_owner_constraints[0] == {1: 1}
    assert preflight.component_owner_constraints[1] == {1: 0}


def test_owner_preflight_marks_split_merge_associations_ambiguous() -> None:
    """Two fragments cannot both form a unique track to one next-pair fragment."""

    left_first = ProtectedComponentFragment(
        pair_index=0,
        global_bbox=(40, 12, 20, 10),
        local_mask=np.ones((10, 20), dtype=bool),
        component_label=1,
        preferred_owner=0,
    )
    left_second = ProtectedComponentFragment(
        pair_index=0,
        global_bbox=(60, 12, 20, 10),
        local_mask=np.ones((10, 20), dtype=bool),
        component_label=2,
        preferred_owner=0,
    )
    merged = ProtectedComponentFragment(
        pair_index=1,
        global_bbox=(45, 12, 35, 10),
        local_mask=np.ones((10, 35), dtype=bool),
        component_label=1,
        preferred_owner=1,
    )

    preflight = preflight_sequence_owners(((left_first, left_second), (merged,)))

    assert preflight.accepted is True
    assert preflight.component_tracks == ()
    assert preflight.ambiguous_association_count >= 1
    # Keep only independent pair-local decisions; an ambiguous merge must not
    # force either left fragment onto the shared second source.
    assert preflight.component_owner_constraints == ({1: 0, 2: 0}, {1: 1})


def test_owner_preflight_discards_a_three_pair_chain_junction() -> None:
    """A middle fragment cannot belong to both adjacent shared sources."""

    first = ProtectedComponentFragment(
        pair_index=0,
        global_bbox=(40, 12, 16, 8),
        local_mask=np.ones((8, 16), dtype=bool),
        component_label=1,
        preferred_owner=0,
    )
    middle = ProtectedComponentFragment(
        pair_index=1,
        global_bbox=(42, 12, 16, 8),
        local_mask=np.ones((8, 16), dtype=bool),
        component_label=1,
        preferred_owner=0,
    )
    last = ProtectedComponentFragment(
        pair_index=2,
        global_bbox=(44, 12, 16, 8),
        local_mask=np.ones((8, 16), dtype=bool),
        component_label=1,
        preferred_owner=1,
    )

    preflight = preflight_sequence_owners(((first,), (middle,), (last,)))

    assert preflight.accepted is True
    assert preflight.candidate_track_count == 2
    assert preflight.rejected_track_count == 2
    assert preflight.component_tracks == ()
    assert preflight.ambiguous_association_count >= 2
    # The former implementation overwrote the middle fragment from owner 0 to
    # owner 1.  It must instead retain its pair-local decision.
    assert preflight.component_owner_constraints == ({1: 0}, {1: 0}, {1: 1})


def test_owner_preflight_drops_disjoint_tracks_that_reverse_a_pair_seam() -> None:
    """Two valid links may still be jointly impossible in their shared pair."""

    incoming = ProtectedComponentFragment(
        pair_index=0,
        global_bbox=(48, 10, 12, 10),
        local_mask=np.ones((10, 12), dtype=bool),
        component_label=1,
        preferred_owner=0,
    )
    middle_right = ProtectedComponentFragment(
        pair_index=1,
        global_bbox=(48, 10, 12, 10),
        local_mask=np.ones((10, 12), dtype=bool),
        component_label=1,
        preferred_owner=0,
    )
    middle_left = ProtectedComponentFragment(
        pair_index=1,
        global_bbox=(8, 10, 12, 10),
        local_mask=np.ones((10, 12), dtype=bool),
        component_label=2,
        preferred_owner=0,
    )
    outgoing = ProtectedComponentFragment(
        pair_index=2,
        global_bbox=(8, 10, 12, 10),
        local_mask=np.ones((10, 12), dtype=bool),
        component_label=1,
        preferred_owner=1,
    )

    preflight = preflight_sequence_owners(
        ((incoming,), (middle_right, middle_left), (outgoing,))
    )

    assert preflight.accepted is True
    assert preflight.candidate_track_count == 2
    assert preflight.rejected_track_count == 1
    assert len(preflight.component_tracks) == 1
    # The second association would force the left component to source 1 while
    # the right component is already forced to source 0, which has no
    # monotonic hard-owner solution.  Its pair-local owner remains intact.
    assert preflight.component_owner_constraints[1] == {1: 0, 2: 0}
