from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from panorama_demo.central_strip import (
    CentralStripLayout,
    ScanAxes,
    _prepare_central_strip_render_domains,
    _validate_time_order,
    fit_reference_plane,
    render_central_strip_diagnostic,
    validate_central_strip_config,
)
from panorama_demo.render import PrewarpedScanSource
from panorama_demo.session import load_rgbd_session
from panorama_demo.synthetic import generate_sequence


def _settings() -> dict[str, object]:
    return {
        "enabled": True,
        "reference_scale_mode": "robust_aligned_depth_plane",
        "orientation_mode": "verified_camera_to_world",
        "maximum_central_band_fraction": 0.20,
        "minimum_pair_overlap_pixels": 96,
        "exposure_mode": "global_gain",
        "multiband_levels": 5,
    }


def _known_poses(session: Path) -> list[np.ndarray]:
    manifest = json.loads((session / "manifest.json").read_text(encoding="utf-8"))
    return [
        np.asarray(row["matrix_row_major"], dtype=np.float64).reshape(4, 4)
        for row in manifest["known_trajectory"]["poses"]
    ]


def _fronto_parallel_axes() -> ScanAxes:
    return ScanAxes(
        scan_axis=(1.0, 0.0, 0.0),
        up_axis=(0.0, -1.0, 0.0),
        normal_axis=(0.0, 0.0, 1.0),
    )


def _prewarped_source(color_valid_mask: np.ndarray) -> PrewarpedScanSource:
    """Construct a minimal real-mask prewarped source for domain-gate tests."""

    height, width = color_valid_mask.shape
    empty_depth = np.zeros((height, width), dtype=np.float32)
    empty_validity = np.zeros((height, width), dtype=np.uint8)
    return PrewarpedScanSource(
        rgb=np.zeros((height, width, 3), dtype=np.uint8),
        color_valid_mask=color_valid_mask,
        surface_depth_mm=empty_depth,
        surface_depth_valid_mask=empty_validity,
        camera_depth_mm=empty_depth,
        camera_depth_valid_mask=empty_validity,
        projected_center_xy=(float(width) / 2.0, float(height) / 2.0),
    )


def test_central_strip_uses_real_plane_and_keeps_rgb_depth_holes_separate(
    tmp_path: Path,
) -> None:
    # At 640 px the fixed 20% central field can genuinely satisfy the fixed
    # 96-pixel pair-overlap admission rule.  This is deliberately not a small
    # test image that would hide an impossible production configuration.
    root = generate_sequence(
        tmp_path / "session",
        frame_count=7,
        frame_width=640,
        frame_height=400,
        step=20,
        seed=31,
    )
    before_hole = load_rgbd_session(root)
    target = before_hole.frames[3].aligned_depth_path
    depth = cv2.imread(str(target), cv2.IMREAD_UNCHANGED)
    assert depth is not None
    depth[130:260, 270:390] = 0
    assert cv2.imwrite(str(target), depth)

    session = load_rgbd_session(root)
    result = render_central_strip_diagnostic(
        plane_frames=session.frames,
        plane_poses=_known_poses(root),
        render_frames=session.frames,
        render_poses=_known_poses(root),
        calibration=session.calibration,
        config=_settings(),
        sharpness_scores=[1.0] * len(session.frames),
    )

    assert result.panorama.dtype == np.uint8
    assert result.panorama.shape[2] == 3
    assert result.metadata["geometry_claim"] == "reference_plane_only"
    assert result.metadata["interpolated_pose_count"] == 0
    assert result.metadata["reference_plane"]["inlier_count"] > 0
    assert result.metadata["reference_plane"]["measured_depth_validation_coverage"] > 0.90
    assert result.metadata["reference_plane"]["quality"]["fit_quality_pass"] is True
    assert result.metadata["reference_plane"]["quality"]["distinct_candidate_count"] == 1
    assert result.metadata["layout"]["aggregate_megapixels"] <= 80.0
    source = result.metadata["sources"][3]
    assert source["rgb_without_measured_depth_pixel_count"] > 0
    assert len(result.metadata["adjacent_motion"]) == len(session.frames) - 1
    assert all(
        row["expected_scan_motion_blur_pixels"][0] >= 0.0
        for row in result.metadata["adjacent_motion"]
    )
    assert result.metadata["render"]["quality_metrics"]["strict_owner_partition"] is True


def test_central_strip_rejects_capture_order_reversal() -> None:
    poses = []
    for x in (0.0, 100.0, 60.0):
        pose = np.eye(4, dtype=np.float64)
        pose[0, 3] = x
        poses.append(pose)
    axes = ScanAxes(
        scan_axis=(1.0, 0.0, 0.0),
        up_axis=(0.0, -1.0, 0.0),
        normal_axis=(0.0, 0.0, 1.0),
    )

    with pytest.raises(RuntimeError, match="reverses"):
        _validate_time_order(poses, axes)


def test_central_strip_rejects_ambiguous_two_plane_measured_depth(
    tmp_path: Path,
) -> None:
    root = generate_sequence(
        tmp_path / "ambiguous-planes",
        frame_count=6,
        frame_width=640,
        frame_height=400,
        step=20,
        seed=43,
    )
    for path in sorted((root / "depth_aligned").glob("*.png")):
        depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        assert depth is not None
        # Two equally extensive, stable fronto-parallel depth planes are not a
        # unique measured reference plane.  The gate must not choose one by
        # proximity or capture order.
        depth[:, : depth.shape[1] // 2] = 1800
        assert cv2.imwrite(str(path), depth)

    session = load_rgbd_session(root)
    with pytest.raises(RuntimeError, match="ambiguous competing reference planes"):
        fit_reference_plane(
            session.frames,
            _known_poses(root),
            session.calibration,
            _fronto_parallel_axes(),
        )


def test_central_strip_rejects_plane_with_only_narrow_calibrated_support(
    tmp_path: Path,
) -> None:
    root = generate_sequence(
        tmp_path / "narrow-plane-support",
        frame_count=6,
        frame_width=640,
        frame_height=400,
        step=20,
        seed=47,
    )
    for path in sorted((root / "depth_aligned").glob("*.png")):
        depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        assert depth is not None
        # A 4.7% vertical strip has ample samples for a numerical TLS fit, but
        # cannot establish the required calibrated image-area support.
        depth.fill(0)
        centre = depth.shape[1] // 2
        depth[:, centre - 15 : centre + 15] = 2400
        assert cv2.imwrite(str(path), depth)

    session = load_rgbd_session(root)
    with pytest.raises(
        RuntimeError, match="lacks calibrated image-area support across the scan"
    ):
        fit_reference_plane(
            session.frames,
            _known_poses(root),
            session.calibration,
            _fronto_parallel_axes(),
        )


def test_central_strip_rejects_pair_without_enough_active_cross_scan_rows() -> None:
    height, width = 100, 220
    # The pair has a wide real common RGB interval, but it is active in only
    # 79% of cross-scan rows.  This checks the independent vertical-coverage
    # admission gate rather than merely the 96-pixel horizontal-overlap gate.
    color_valid = np.zeros((height, width), dtype=np.uint8)
    color_valid[:79, :] = 255
    layout = CentralStripLayout(
        width=width,
        height=height,
        pixels_per_mm=1.0,
        scan_min_mm=0.0,
        scan_max_mm=float(width - 1),
        up_min_mm=0.0,
        up_max_mm=float(height - 1),
        source_scan_positions_mm=(55.0, 165.0),
        source_support_intervals_mm=((0.0, 180.0), (40.0, float(width - 1))),
        owner_intervals_mm=((0.0, 110.0), (110.0, float(width - 1))),
        owner_boundaries_x=(110.0,),
        source_center_xy=((55.0, 50.0), (165.0, 50.0)),
        pair_overlap_pixels=(140.0,),
        canvas_megapixels=height * width / 1_000_000.0,
        aggregate_megapixels=2.0 * height * width / 1_000_000.0,
    )

    with pytest.raises(RuntimeError, match="do not cover 80% of the required cross-scan"):
        _prepare_central_strip_render_domains(
            [_prewarped_source(color_valid), _prewarped_source(color_valid)],
            layout,
            minimum_pair_overlap_pixels=96,
        )


@pytest.mark.parametrize(
    "override, message",
    [
        ({"unknown": True}, "Unknown"),
        ({"maximum_central_band_fraction": 0.15}, "fixed at 0.20"),
        ({"minimum_pair_overlap_pixels": 64}, "fixed at 96"),
        ({"multiband_levels": 4}, "fixed at the audited value"),
    ],
)
def test_central_strip_configuration_is_closed(
    override: dict[str, object], message: str
) -> None:
    config = _settings()
    config.update(override)
    with pytest.raises(ValueError, match=message):
        validate_central_strip_config(config, require_enabled=True)
