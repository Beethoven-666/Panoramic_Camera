from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np

from panorama_demo.session import discover_frames
from panorama_demo.synthetic import generate_sequence


def _read_color_frames(directory: Path) -> list[np.ndarray]:
    images = []
    for path in sorted((directory / "color").glob("*.jpg")):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
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
    assert manifest == {
        "schema": "panorama-demo-session/v1",
        "synthetic": True,
        "frame_count": 4,
        "known_step_px": 18,
        "frame_size": [96, 72],
    }

    frames = discover_frames(output)
    assert [frame.frame_id for frame in frames] == [0, 1, 2, 3]
    assert [frame.timestamp_us for frame in frames] == [0, 100_000, 200_000, 300_000]
    assert all(frame.depth_path is None for frame in frames)
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
            "color_path": "color/00000000.jpg",
        },
        {
            "frame_id": "1",
            "color_device_timestamp_us": "100000",
            "color_path": "color/00000001.jpg",
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
    assert _read_color_frames(output)[0].shape == (60, 70, 3)
