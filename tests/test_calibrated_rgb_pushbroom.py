from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

import panorama_demo.calibrated_rgb_pushbroom as pushbroom_module
from panorama_demo.calibrated_rgb_pushbroom import (
    CalibratedRGBPushbroomConfig,
    _PhotometricEdge,
    _adaptive_multiband_levels,
    _apply_linear_bgr_gain,
    _blend_safe_pair_zone,
    _graphcut_monotonic_owner,
    _solve_global_linear_rgb_gains,
    _srgb_to_linear_bgr,
    estimate_rgb_motion_pixels_per_mm,
    render_calibrated_rgb_pushbroom,
)
from panorama_demo.session import RGBDSession, load_rgbd_session
from panorama_demo.synthetic import generate_sequence


def _make_rgb_pushbroom_input(tmp_path: Path, *, seed: int) -> tuple[RGBDSession, list[np.ndarray]]:
    root = generate_sequence(
        tmp_path / "session",
        frame_count=5,
        frame_width=320,
        frame_height=200,
        # Dense real pose nodes leave a 32--64 px calibrated seam-search
        # corridor inside the unchanged 20% central source-band limit.
        step=16,
        seed=seed,
    )
    # The general renderer fixture isolates calibrated-strip, photometric and
    # ownership contracts from the generator's deliberately dense foreground
    # clutter.  Dedicated tests below cover protected foreground components.
    for index in range(5):
        image = np.full(
            (200, 320, 3),
            (180 + 2 * index, 182 + index, 184),
            dtype=np.uint8,
        )
        assert cv2.imwrite(str(root / "color" / f"{index:08d}.jpg"), image)
    session = load_rgbd_session(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    trajectory = manifest["known_trajectory"]
    assert trajectory["transform"] == "camera_to_world"
    return (
        session,
        [
            np.asarray(row["matrix_row_major"], dtype=np.float64).reshape(4, 4)
            for row in trajectory["poses"]
        ],
    )


def _reliable_adjacent_rgb_motions(frame_count: int) -> list[dict[str, object]]:
    """Synthetic frames advance by 16 RGB pixels at the far background."""

    return [{"dx": 16.0, "reliable": True} for _ in range(frame_count - 1)]


def test_pushbroom_uses_every_real_frame_once_with_bounded_strip_residency(
    tmp_path: Path,
) -> None:
    session, poses = _make_rgb_pushbroom_input(tmp_path, seed=17)
    motions = _reliable_adjacent_rgb_motions(len(session.frames))

    result = render_calibrated_rgb_pushbroom(
        session.frames,
        poses,
        session.calibration,
        rgb_motions=motions,
    )

    metadata = result.metadata
    metrics = metadata["quality_metrics"]
    assert len(motions) == len(session.frames) - 1
    assert metadata["source_count"] == len(session.frames)
    assert metadata["frame_ids"] == [frame.frame_id for frame in session.frames]
    assert metadata["single_inverse_remap_per_source"] is True
    assert metadata["interpolated_pose_count"] == 0
    assert metrics["source_remap_count"] == len(session.frames)
    assert 2 <= metrics["maximum_resident_strips"] <= 5
    assert len(metadata["rgb_motion_scale"]["samples"]) == len(session.frames) - 1
    assert metadata["layout"]["endpoint_policy"] == "outward_half_fov"
    assert len(metadata["layout"]["endpoint_outer_owner_intervals_x"]) == 2
    assert metadata["layout"]["maximum_source_strip_width"] > 64
    supports = metadata["layout"]["source_support_intervals_x"]
    assert all(right - left <= 64 for left, right in supports[1:-1])
    assert all(count > 0 for count in metadata["source_owner_pixel_counts"])
    assert all(
        count > 0
        for count in metrics["endpoint_outer_half_fov_owner_pixel_counts"]
    )
    assert metrics["endpoint_outer_half_fov_preserved"] is True
    assert metrics["endpoint_outer_half_fov_trimmed_column_counts"] == [0, 0]
    assert metrics["endpoint_outer_half_fov_trimmed_invalid_pixel_counts"] == [0, 0]


def test_pushbroom_keeps_outward_endpoint_coverage_when_virtual_x_is_reversed(
    tmp_path: Path,
) -> None:
    session, poses = _make_rgb_pushbroom_input(tmp_path, seed=19)
    reversed_poses = []
    for pose in poses:
        reversed_pose = pose.copy()
        reversed_pose[:3, :3] = np.diag((-1.0, 1.0, -1.0))
        reversed_poses.append(reversed_pose)

    result = render_calibrated_rgb_pushbroom(
        session.frames,
        reversed_poses,
        session.calibration,
        rgb_motions=_reliable_adjacent_rgb_motions(len(session.frames)),
    )

    layout = result.metadata["layout"]
    metrics = result.metadata["quality_metrics"]
    assert layout["temporal_to_virtual_x_sign"] < 0.0
    assert layout["endpoint_policy"] == "outward_half_fov"
    assert all(
        count > 0
        for count in metrics["endpoint_outer_half_fov_owner_pixel_counts"]
    )


def test_pushbroom_pair_blends_are_narrow_and_never_include_rgb_risk(
    tmp_path: Path,
) -> None:
    session, poses = _make_rgb_pushbroom_input(tmp_path, seed=23)
    result = render_calibrated_rgb_pushbroom(
        session.frames,
        poses,
        session.calibration,
        rgb_motions=_reliable_adjacent_rgb_motions(len(session.frames)),
    )

    pairs = result.metadata["pairs"]
    metrics = result.metadata["quality_metrics"]
    assert len(pairs) == len(session.frames) - 1
    assert all(2 <= pair["blend_width_pixels"] <= 8 for pair in pairs)
    # Endpoint supports may be shortened by the outward-half-FOV policy, but
    # interior seams retain a 32--64 px search corridor that is independent
    # from the 2--8 px output blend band.
    assert all(2 <= pair["search_corridor_width_pixels"] <= 64 for pair in pairs)
    assert all(
        32 <= pair["search_corridor_width_pixels"] <= 64 for pair in pairs[1:-1]
    )
    assert all(
        pair["search_corridor_width_pixels"] > pair["blend_width_pixels"]
        for pair in pairs
    )
    assert all(pair["blend_zone_risk_pixel_count"] == 0 for pair in pairs)
    assert metrics["blend_zone_risk_pixel_count"] == 0
    assert metrics["blend_zone_risk_fraction"] == 0.0
    assert metrics["blend_zone_fraction"] <= 0.20


def test_pushbroom_crops_from_valid_mask_and_preserves_valid_black_rgb(
    tmp_path: Path,
) -> None:
    session, poses = _make_rgb_pushbroom_input(tmp_path, seed=31)
    black = np.zeros(
        (session.calibration.height, session.calibration.width, 3), dtype=np.uint8
    )
    # Preserve a neutral wall rail solely for the mandatory photometric
    # measurement; the large black region remains valid output content.
    black[:40, :, :] = 190
    for frame in session.frames:
        assert cv2.imwrite(str(frame.color_path), black)
        # Rendering after this deletion proves output pixels are not read from
        # the strict session's aligned-depth files.
        frame.aligned_depth_path.unlink()

    result = render_calibrated_rgb_pushbroom(
        session.frames,
        poses,
        session.calibration,
        rgb_motions=_reliable_adjacent_rgb_motions(len(session.frames)),
    )

    crop = result.metadata["crop"]
    assert result.metadata["depth_used_for_output_pixels"] is False
    assert result.panorama.shape[:2] == (crop["height"], crop["width"])
    assert crop["width"] > 0 and crop["height"] > 0
    assert np.any(np.all(result.panorama == 0, axis=2))


def test_pushbroom_rejects_insufficient_reliable_rgb_motion_scale(
    tmp_path: Path,
) -> None:
    session, poses = _make_rgb_pushbroom_input(tmp_path, seed=41)
    invalid_motions = [
        {"dx": 0.0, "reliable": False} for _ in range(len(session.frames) - 1)
    ]

    with pytest.raises(RuntimeError, match="too few reliable adjacent RGB motion"):
        estimate_rgb_motion_pixels_per_mm(
            session.frames,
            poses,
            session.calibration,
            CalibratedRGBPushbroomConfig(),
            rgb_motions=invalid_motions,
        )


def test_pushbroom_rejects_unstable_rgb_motion_scale(tmp_path: Path) -> None:
    session, poses = _make_rgb_pushbroom_input(tmp_path, seed=47)
    unstable_motions = [
        {"dx": value, "reliable": True} for value in (60.0, 120.0, 1200.0, 2400.0)
    ]

    with pytest.raises(RuntimeError, match="RGB-motion scale is unstable"):
        estimate_rgb_motion_pixels_per_mm(
            session.frames,
            poses,
            session.calibration,
            CalibratedRGBPushbroomConfig(),
            rgb_motions=unstable_motions,
        )


def test_global_linear_rgb_gain_solver_is_joint_per_channel_and_fail_closed() -> None:
    source_count = 5
    # Each BGR channel has its own linear log-gain slope.  The mean-zero gauge
    # is known analytically, and affine gain curves are not biased by the
    # second-difference regularizer.
    slope_bgr = np.array((0.045, -0.025, 0.030), dtype=np.float64)
    expected_log_gains = (
        np.arange(source_count, dtype=np.float64)[:, None] - 2.0
    ) * slope_bgr[None, :]
    edges = [
        _PhotometricEdge(
            log_relation_bgr=slope_bgr.copy(),
            support_pixels=1024,
            mad_bgr=np.full(3, 0.005, dtype=np.float64),
            raw_signed_l_delta=0.0,
        )
        for _ in range(source_count - 1)
    ]

    gains, metrics = _solve_global_linear_rgb_gains(source_count, edges)

    assert metrics["photometric_mode"] == "safe_wall_global_linear_rgb"
    assert metrics["photometric_global_solver"] is True
    np.testing.assert_allclose(np.log(gains), expected_log_gains, atol=1e-8)
    assert not np.allclose(gains[:, 0], gains[:, 1])
    assert not np.allclose(gains[:, 1], gains[:, 2])

    with pytest.raises(RuntimeError, match="No reliable safe white-wall"):
        _solve_global_linear_rgb_gains(source_count, [edges[0], None, *edges[2:]])


def test_linear_rgb_gain_application_is_per_channel_not_gamma_scalar() -> None:
    encoded = np.full((4, 5, 3), (96, 144, 192), dtype=np.uint8)
    gain_bgr = np.array((1.20, 0.82, 1.08), dtype=np.float64)

    corrected = _apply_linear_bgr_gain(encoded, gain_bgr)
    expected_linear = _srgb_to_linear_bgr(encoded) * gain_bgr.reshape(1, 1, 3)
    actual_linear = _srgb_to_linear_bgr(corrected)

    # Encoding quantization is the only allowed discrepancy; a scalar applied
    # to gamma-encoded RGB would fail this linear-light comparison.
    np.testing.assert_allclose(actual_linear, expected_linear, atol=0.005)
    assert not np.all(corrected[:, :, 0] == corrected[:, :, 1])
    assert not np.all(corrected[:, :, 1] == corrected[:, :, 2])


def test_protected_foreground_component_is_owned_wholly_by_one_source() -> None:
    height, width = 32, 64
    first = np.full((height, width, 3), 190, dtype=np.uint8)
    second = np.full((height, width, 3), 190, dtype=np.uint8)
    valid = np.ones((height, width), dtype=bool)
    # A horizontal hose spans the nominal seam.  Its uniform interior is part
    # of the supplied connected protection component, so the GraphCut owner
    # must not split it at the nominal centre.
    protected = np.zeros((height, width), dtype=bool)
    protected[12:20, 18:46] = True
    first[protected] = (20, 20, 20)
    second[protected] = (20, 20, 20)

    owner0, owner1, cuts, _, _, split_count, boundary_guard_count = (
        _graphcut_monotonic_owner(
            first,
            second,
            valid,
            valid,
            protected,
            nominal_boundary=32,
        )
    )

    assert np.all(owner0[protected]) or np.all(owner1[protected])
    assert not np.any(owner0[protected] & owner1[protected])
    assert split_count == 0
    assert boundary_guard_count == 0
    assert np.all(cuts[12:20] < 18) or np.all(cuts[12:20] >= 46)


def test_local_multiband_uses_distinct_owner_masks_and_adaptive_levels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    height, width = 12, 20
    first = np.full((height, width, 3), (80, 120, 160), dtype=np.uint8)
    second = np.full((height, width, 3), (120, 160, 200), dtype=np.uint8)
    common = np.ones((height, width), dtype=bool)
    protected = np.zeros((height, width), dtype=bool)
    safe_wall = common.copy()
    owner0 = np.zeros_like(common)
    owner0[:, :8] = True
    owner1 = common & ~owner0
    cuts = np.full(height, 7, dtype=np.int32)
    captured_masks: list[np.ndarray] = []
    captured_images: list[np.ndarray] = []

    class _CapturingBlender:
        def setNumBands(self, _: int) -> None:
            pass

        def prepare(self, _: tuple[int, int, int, int]) -> None:
            pass

        def feed(
            self, image: np.ndarray, mask: np.ndarray, _: tuple[int, int]
        ) -> None:
            captured_images.append(np.asarray(image).copy())
            captured_masks.append(np.asarray(mask).copy())

        def blend(
            self, _: None, __: None
        ) -> tuple[np.ndarray, np.ndarray]:
            return (
                captured_images[0],
                np.where(captured_masks[0] | captured_masks[1], 255, 0).astype(
                    np.uint8
                ),
            )

    monkeypatch.setattr(
        pushbroom_module.cv2,
        "detail_MultiBandBlender",
        _CapturingBlender,
    )
    (
        _,
        zone,
        pixels,
        levels,
        mask0_pixels,
        mask1_pixels,
        masks_distinct,
    ) = _blend_safe_pair_zone(
        first,
        second,
        common,
        protected,
        safe_wall,
        owner0,
        owner1,
        cuts,
        blend_width=6,
        levels=3,
    )

    assert pixels == int(np.count_nonzero(zone)) > 0
    assert levels == 3
    assert mask0_pixels > 0 and mask1_pixels > 0
    assert masks_distinct is True
    assert len(captured_masks) == 2
    assert not np.array_equal(captured_masks[0], captured_masks[1])
    assert _adaptive_multiband_levels(2, 3) == 1
    assert _adaptive_multiband_levels(4, 3) == 2
    assert _adaptive_multiband_levels(8, 3) == 3


def test_preview_residual_diagnostics_keep_identity_output_and_remap_counts_separate(
    tmp_path: Path,
) -> None:
    session, poses = _make_rgb_pushbroom_input(tmp_path, seed=53)
    motions = _reliable_adjacent_rgb_motions(len(session.frames))
    default_result = render_calibrated_rgb_pushbroom(
        session.frames,
        poses,
        session.calibration,
        rgb_motions=motions,
    )
    identity_result = render_calibrated_rgb_pushbroom(
        session.frames,
        poses,
        session.calibration,
        config={"residual_alignment": {"background_model": "identity"}},
        rgb_motions=motions,
    )

    # Preview evidence uses a separate low-resolution inverse remap.  It is
    # forbidden from changing a formal source sample while the selected model
    # remains identity.
    assert np.array_equal(default_result.panorama, identity_result.panorama)
    alignment = default_result.metadata["residual_alignment"]
    metrics = default_result.metadata["quality_metrics"]
    assert alignment["selected_model"] == "identity"
    assert alignment["analysis_preview_remap_count"] == len(session.frames)
    assert alignment["full_resolution_output_remap_count"] == len(session.frames)
    assert metrics["analysis_preview_remap_count"] == len(session.frames)
    assert metrics["full_resolution_output_remap_count"] == len(session.frames)
    assert len(alignment["evidence"]) == len(session.frames) - 1
    working = alignment["working_set_audit"]
    assert working["preview_streaming_maximum_resident_previews"] == 2
    assert (
        working["preview_evidence_pixel_count"]
        <= working["preview_evidence_hard_limit_pixels"]
    )
    assert working["preview_evidence_storage"] == "bounded_in_memory_analysis_only"


def test_preview_evidence_budget_fails_before_formal_full_resolution_remaps(
    tmp_path: Path,
) -> None:
    session, poses = _make_rgb_pushbroom_input(tmp_path, seed=59)

    with pytest.raises(RuntimeError, match="preview evidence exceeds"):
        render_calibrated_rgb_pushbroom(
            session.frames,
            poses,
            session.calibration,
            rgb_motions=_reliable_adjacent_rgb_motions(len(session.frames)),
            config={
                "residual_alignment": {
                    "maximum_evidence_megapixels": 0.000001,
                }
            },
        )
