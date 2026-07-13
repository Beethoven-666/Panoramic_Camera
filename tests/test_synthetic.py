from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from panorama_demo.session import (
    discover_frames,
    load_rgbd_session,
    read_aligned_depth_mm,
)
from panorama_demo.synthetic import (
    FAR_DEPTH_MM,
    NEAR_DEPTH_MM,
    generate_sequence,
)


def _read_color_frames(directory: Path) -> list[np.ndarray]:
    images = []
    for path in sorted((directory / "color").glob("*.jpg")):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        assert image is not None
        images.append(image)
    return images


def _read_depth_frames(directory: Path) -> list[np.ndarray]:
    images = []
    for path in sorted((directory / "depth_aligned").glob("*.png")):
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        assert image is not None
        images.append(image)
    return images


def test_generate_sequence_writes_replayable_session(tmp_path: Path) -> None:
    output = tmp_path / "session"

    result = generate_sequence(
        output,
        frame_count=4,
        frame_width=96,
        frame_height=72,
        step=18,
        seed=11,
    )

    assert result == output.resolve()
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema"] == "panorama-demo-session/v2"
    assert manifest["synthetic"] is True
    assert manifest["frame_count"] == 4
    assert manifest["known_step_px"] == 18
    assert manifest["frame_size"] == [96, 72]
    assert manifest["scene"] == "plane"
    assert manifest["capture_options"]["align"] == "software"
    assert manifest["known_trajectory"]["transform"] == "camera_to_world"
    assert manifest["known_trajectory"]["translation_unit"] == "millimetres"

    frames = discover_frames(output)
    assert [frame.frame_id for frame in frames] == [0, 1, 2, 3]
    assert [frame.timestamp_us for frame in frames] == [0, 100_000, 200_000, 300_000]
    assert all(frame.depth_path is not None for frame in frames)
    assert all(frame.depth_scale_mm_per_unit == 1.0 for frame in frames)
    assert [frame.color_path.name for frame in frames] == [
        "00000000.jpg",
        "00000001.jpg",
        "00000002.jpg",
        "00000003.jpg",
    ]

    images = _read_color_frames(output)
    assert len(images) == 4
    assert all(image.shape == (72, 96, 3) for image in images)
    assert all(image.dtype == np.uint8 for image in images)
    depths = _read_depth_frames(output)
    assert len(depths) == 4
    assert all(depth.shape == (72, 96) for depth in depths)
    assert all(depth.dtype == np.uint16 for depth in depths)
    assert all(np.all(depth == FAR_DEPTH_MM) for depth in depths)

    strict_session = load_rgbd_session(output)
    assert len(strict_session.frames) == 4
    assert strict_session.calibration.width == 96
    assert strict_session.calibration.height == 72
    np.testing.assert_array_equal(
        read_aligned_depth_mm(strict_session.frames[0]),
        np.full((72, 96), FAR_DEPTH_MM, dtype=np.float32),
    )


def test_generate_sequence_csv_uses_relative_portable_paths(tmp_path: Path) -> None:
    output = generate_sequence(
        tmp_path / "session", frame_count=2, frame_width=80, frame_height=64, step=12
    )

    with (output / "frames.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert rows == [
        {
            "frame_id": "0",
            "color_device_timestamp_us": "0",
            "color_exposure": "8",
            "color_gain": "16",
            "depth_scale_mm_per_unit": "1.0",
            "color_path": "color/00000000.jpg",
            "aligned_depth_path": "depth_aligned/00000000.png",
            "raw_depth_path": "",
        },
        {
            "frame_id": "1",
            "color_device_timestamp_us": "100000",
            "color_exposure": "8",
            "color_gain": "16",
            "depth_scale_mm_per_unit": "1.0",
            "color_path": "color/00000001.jpg",
            "aligned_depth_path": "depth_aligned/00000001.png",
            "raw_depth_path": "",
        },
    ]


def test_generate_sequence_is_deterministic_for_same_seed(tmp_path: Path) -> None:
    first = generate_sequence(
        tmp_path / "first",
        frame_count=3,
        frame_width=88,
        frame_height=66,
        step=14,
        seed=42,
    )
    second = generate_sequence(
        tmp_path / "second",
        frame_count=3,
        frame_width=88,
        frame_height=66,
        step=14,
        seed=42,
    )

    first_images = _read_color_frames(first)
    second_images = _read_color_frames(second)
    assert len(first_images) == len(second_images) == 3
    for first_image, second_image in zip(first_images, second_images, strict=True):
        np.testing.assert_array_equal(first_image, second_image)
    for first_depth, second_depth in zip(
        _read_depth_frames(first), _read_depth_frames(second), strict=True
    ):
        np.testing.assert_array_equal(first_depth, second_depth)


def test_generate_sequence_changes_texture_for_different_seed(tmp_path: Path) -> None:
    first = generate_sequence(
        tmp_path / "first",
        frame_count=2,
        frame_width=80,
        frame_height=64,
        step=12,
        seed=1,
    )
    second = generate_sequence(
        tmp_path / "second",
        frame_count=2,
        frame_width=80,
        frame_height=64,
        step=12,
        seed=2,
    )

    differences = [
        np.any(left != right)
        for left, right in zip(
            _read_color_frames(first), _read_color_frames(second), strict=True
        )
    ]
    assert any(differences)


def test_generate_single_frame_sequence(tmp_path: Path) -> None:
    output = generate_sequence(
        tmp_path / "single",
        frame_count=1,
        frame_width=70,
        frame_height=60,
        step=20,
        seed=3,
    )

    frames = discover_frames(output)
    assert len(frames) == 1
    assert frames[0].frame_id == 0
    assert frames[0].depth_path is not None
    assert _read_color_frames(output)[0].shape == (60, 70, 3)


def test_generate_sequence_writes_capture_compatible_calibration(tmp_path: Path) -> None:
    output = generate_sequence(
        tmp_path / "session", frame_count=2, frame_width=80, frame_height=64, step=12
    )

    calibration = json.loads(
        (output / "calibration.json").read_text(encoding="utf-8")
    )

    assert calibration["color_intrinsic"] == {
        "width": 80,
        "height": 64,
        "fx": 68.0,
        "fy": 68.0,
        "cx": 39.5,
        "cy": 31.5,
    }
    assert calibration["depth_intrinsic"] == calibration["color_intrinsic"]
    assert calibration["depth_alignment"] == {
        "aligned_to": "color",
        "enabled": True,
        "method": "synthetic",
    }
    assert calibration["depth_to_color"]["rotation_row_major"] == [
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
    ]


def test_generate_sequence_records_known_se3_trajectory(tmp_path: Path) -> None:
    output = generate_sequence(
        tmp_path / "session", frame_count=3, frame_width=80, frame_height=64, step=17
    )
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    poses = manifest["known_trajectory"]["poses"]

    assert [pose["frame_id"] for pose in poses] == [0, 1, 2]
    matrices = [
        np.asarray(pose["matrix_row_major"], dtype=np.float64).reshape(4, 4)
        for pose in poses
    ]
    expected_step_mm = 17 * FAR_DEPTH_MM / (80 * 0.85)
    for index, matrix in enumerate(matrices):
        np.testing.assert_allclose(matrix[:3, :3], np.eye(3))
        np.testing.assert_allclose(matrix[:3, 3], [index * expected_step_mm, 0, 0])
        np.testing.assert_allclose(matrix[3], [0, 0, 0, 1])


@pytest.mark.parametrize(
    ("scene", "required_depths"),
    [
        ("plane", {FAR_DEPTH_MM}),
        ("layered", {NEAR_DEPTH_MM, FAR_DEPTH_MM}),
        ("occlusion", {NEAR_DEPTH_MM, FAR_DEPTH_MM}),
        ("depth_hole", {0, NEAR_DEPTH_MM, FAR_DEPTH_MM}),
        ("dynamic_object", {NEAR_DEPTH_MM - 200, NEAR_DEPTH_MM, FAR_DEPTH_MM}),
    ],
)
def test_generate_sequence_covers_rgbd_scene_variants(
    tmp_path: Path, scene: str, required_depths: set[int]
) -> None:
    output = generate_sequence(
        tmp_path / scene,
        frame_count=4,
        frame_width=120,
        frame_height=80,
        step=18,
        seed=5,
        scene=scene,
    )

    observed = set(
        np.unique(np.concatenate([depth.reshape(-1) for depth in _read_depth_frames(output)]))
    )

    assert required_depths <= observed
    assert len(load_rgbd_session(output).frames) == 4


def test_dynamic_object_has_non_camera_motion(tmp_path: Path) -> None:
    output = generate_sequence(
        tmp_path / "dynamic",
        frame_count=3,
        frame_width=120,
        frame_height=80,
        step=18,
        scene="dynamic_object",
    )
    dynamic_masks = [depth == NEAR_DEPTH_MM - 200 for depth in _read_depth_frames(output)]
    centroids = [float(np.flatnonzero(mask.any(axis=0)).mean()) for mask in dynamic_masks]

    assert centroids[0] < centroids[1] < centroids[2]


def test_generate_sequence_rejects_unknown_scene(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unsupported synthetic scene"):
        generate_sequence(tmp_path / "bad", scene="mirror")
