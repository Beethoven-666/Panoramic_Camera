from __future__ import annotations

import csv
from pathlib import Path

import pytest

from panorama_demo.session import SessionFrame, discover_frames, select_frames


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fixture")
    return path.resolve()


def test_discover_single_image(tmp_path: Path) -> None:
    image = _touch(tmp_path / "frame.PNG")

    frames = discover_frames(image)

    assert frames == [SessionFrame(frame_id=0, color_path=image)]


def test_discover_directory_uses_sorted_color_subdirectory(tmp_path: Path) -> None:
    second = _touch(tmp_path / "color" / "10.JPG")
    first = _touch(tmp_path / "color" / "02.png")
    _touch(tmp_path / "color" / "ignore.txt")
    _touch(tmp_path / "root_image.jpg")

    frames = discover_frames(tmp_path)

    assert [frame.frame_id for frame in frames] == [0, 1]
    assert [frame.color_path for frame in frames] == [first, second]
    assert all(frame.depth_path is None for frame in frames)


def test_discover_csv_supports_alias_columns_and_utf8_bom(tmp_path: Path) -> None:
    color0 = _touch(tmp_path / "rgb" / "a.png")
    color1 = _touch(tmp_path / "rgb" / "b.jpg")
    depth0 = _touch(tmp_path / "depth" / "a.png")
    csv_path = tmp_path / "frames.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["pair_id", "image_path", "depth_path", "timestamp_us"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "pair_id": "7",
                "image_path": "rgb/a.png",
                "depth_path": "depth/a.png",
                "timestamp_us": "12345",
            }
        )
        writer.writerow(
            {
                "pair_id": "9",
                "image_path": "rgb/b.jpg",
                "depth_path": "",
                "timestamp_us": "",
            }
        )
        writer.writerow(
            {
                "pair_id": "10",
                "image_path": "",
                "depth_path": "",
                "timestamp_us": "999",
            }
        )

    frames = discover_frames(csv_path)

    assert frames == [
        SessionFrame(7, color0, depth0, 12345),
        SessionFrame(9, color1, None, None),
    ]


def test_discover_csv_prefers_capture_column_names(tmp_path: Path) -> None:
    color = _touch(tmp_path / "color" / "0001.jpg")
    aligned_depth = _touch(tmp_path / "aligned" / "0001.png")
    _touch(tmp_path / "raw_depth" / "0001.png")
    csv_path = tmp_path / "frames.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "frame_id",
                "color_path",
                "aligned_depth_path",
                "depth_path",
                "color_device_timestamp_us",
                "timestamp_us",
                "depth_scale_mm_per_unit",
                "color_exposure",
                "color_gain",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "frame_id": "4",
                "color_path": "color/0001.jpg",
                "aligned_depth_path": "aligned/0001.png",
                "depth_path": "raw_depth/0001.png",
                "color_device_timestamp_us": "800",
                "timestamp_us": "700",
                "depth_scale_mm_per_unit": "0.1",
                "color_exposure": "8",
                "color_gain": "24",
            }
        )

    assert discover_frames(tmp_path) == [
        SessionFrame(4, color, aligned_depth, 800, 0.1, 8, 24)
    ]


def test_discover_rejects_missing_csv_color_image(tmp_path: Path) -> None:
    (tmp_path / "frames.csv").write_text(
        "frame_id,color_path\n0,color/missing.png\n", encoding="utf-8"
    )

    with pytest.raises(FileNotFoundError, match="missing color images"):
        discover_frames(tmp_path)


@pytest.mark.parametrize("name", ["frame.raw", "frames.json"])
def test_discover_rejects_unsupported_file(tmp_path: Path, name: str) -> None:
    path = _touch(tmp_path / name)

    with pytest.raises(ValueError, match="Unsupported input file"):
        discover_frames(path)


def test_discover_rejects_nonexistent_path(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        discover_frames(tmp_path / "does-not-exist")


def test_select_frames_applies_stride_then_limit() -> None:
    frames = [SessionFrame(index, Path(f"{index}.jpg")) for index in range(8)]

    assert [frame.frame_id for frame in select_frames(frames, stride=2)] == [0, 2, 4, 6]
    assert [
        frame.frame_id for frame in select_frames(frames, stride=2, max_frames=2)
    ] == [0, 2]
    assert select_frames(frames, max_frames=0) == frames


def test_select_frames_rejects_invalid_stride() -> None:
    with pytest.raises(ValueError, match="stride must be at least 1"):
        select_frames([], stride=0)
