from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from panorama_demo.session import (
    CameraIntrinsics,
    RGBDFrame,
    SessionFrame,
    discover_frames,
    load_rgbd_session,
    load_session_manifest,
    read_aligned_depth_mm,
    select_frames,
)


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fixture")
    return path.resolve()


def _write_rgbd_session(
    root: Path,
    *,
    width: int = 32,
    height: int = 24,
    calibration_width: int | None = None,
    calibration_height: int | None = None,
    alignment_in_calibration: bool = False,
    manifest_align: str | None = "software",
    aligned_depth_path: str = "depth_aligned/00000000.png",
    raw_depth_path: str = "",
    depth_scale: str = "0.1",
    depth_shape: tuple[int, int] | None = None,
    timestamp: str = "123456",
    color_exposure: str = "8",
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    color = np.full((height, width, 3), (20, 40, 60), dtype=np.uint8)
    color_path = root / "color" / "00000000.jpg"
    color_path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(color_path), color)
    actual_depth_shape = depth_shape or (height, width)
    depth = np.full(actual_depth_shape, 10_000, dtype=np.uint16)
    depth_path = root / aligned_depth_path
    depth_path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(depth_path), depth)

    calibration: dict[str, object] = {
        "color_intrinsic": {
            "width": calibration_width or width,
            "height": calibration_height or height,
            "fx": 25.0,
            "fy": 25.5,
            "cx": (calibration_width or width) / 2.0,
            "cy": (calibration_height or height) / 2.0,
        },
        "color_distortion": {
            "k1": 0.0,
            "k2": 0.0,
            "k3": 0.0,
            "k4": 0.0,
            "k5": 0.0,
            "k6": 0.0,
            "p1": 0.0,
            "p2": 0.0,
        },
    }
    if alignment_in_calibration:
        calibration["depth_alignment"] = {
            "aligned_to": "color",
            "enabled": True,
        }
    (root / "calibration.json").write_text(
        json.dumps(calibration), encoding="utf-8"
    )
    manifest: dict[str, object] = {
        "schema": "panorama-demo-session/v1",
        "clean_shutdown": True,
    }
    if manifest_align is not None:
        manifest["capture_options"] = {"align": manifest_align}
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with (root / "frames.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "frame_id",
                "color_device_timestamp_us",
                "color_exposure",
                "color_gain",
                "depth_scale_mm_per_unit",
                "color_path",
                "aligned_depth_path",
                "raw_depth_path",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "frame_id": 0,
                "color_device_timestamp_us": timestamp,
                "color_exposure": color_exposure,
                "color_gain": 16,
                "depth_scale_mm_per_unit": depth_scale,
                "color_path": "color/00000000.jpg",
                "aligned_depth_path": aligned_depth_path,
                "raw_depth_path": raw_depth_path,
            }
        )
    return root


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


@pytest.mark.parametrize("input_kind", ["session", "csv", "color", "image"])
def test_load_session_manifest_from_supported_session_inputs(
    tmp_path: Path, input_kind: str
) -> None:
    color = _touch(tmp_path / "color" / "0001.jpg")
    csv_path = tmp_path / "frames.csv"
    csv_path.write_text("frame_id,color_path\n0,color/0001.jpg\n", encoding="utf-8")
    (tmp_path / "manifest.json").write_text(
        '{"diagnostic_only": true, "capture_mode": "test"}',
        encoding="utf-8",
    )
    inputs = {
        "session": tmp_path,
        "csv": csv_path,
        "color": tmp_path / "color",
        "image": color,
    }

    manifest = load_session_manifest(inputs[input_kind])

    assert manifest == {"diagnostic_only": True, "capture_mode": "test"}


def test_load_session_manifest_rejects_invalid_json(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_text("not-json", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid session manifest"):
        load_session_manifest(tmp_path)


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


def test_load_rgbd_session_validates_capture_contract(tmp_path: Path) -> None:
    root = _write_rgbd_session(tmp_path / "run")

    session = load_rgbd_session(root)

    assert session.root == root.resolve()
    assert session.depth_alignment == "color"
    assert session.calibration == CameraIntrinsics(
        width=32,
        height=24,
        fx=25.0,
        fy=25.5,
        cx=16.0,
        cy=12.0,
        distortion=(0.0,) * 8,
    )
    assert session.calibration.matrix.tolist() == [
        [25.0, 0.0, 16.0],
        [0.0, 25.5, 12.0],
        [0.0, 0.0, 1.0],
    ]
    assert session.frames == (
        RGBDFrame(
            frame_id=0,
            color_path=(root / "color/00000000.jpg").resolve(),
            aligned_depth_path=(root / "depth_aligned/00000000.png").resolve(),
            depth_scale_mm_per_unit=0.1,
            timestamp_us=123_456,
            color_exposure_raw=8,
            color_gain=16,
        ),
    )
    assert session.frames[0].depth_path == session.frames[0].aligned_depth_path


@pytest.mark.parametrize("clean_shutdown", [False, None])
def test_load_rgbd_session_rejects_incomplete_project_session(
    tmp_path: Path, clean_shutdown: bool | None
) -> None:
    root = _write_rgbd_session(tmp_path)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if clean_shutdown is None:
        manifest.pop("clean_shutdown")
    else:
        manifest["clean_shutdown"] = clean_shutdown
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="not cleanly closed"):
        load_rgbd_session(root)


def test_load_rgbd_session_requires_manifest(tmp_path: Path) -> None:
    root = _write_rgbd_session(tmp_path)
    (root / "manifest.json").unlink()

    with pytest.raises(ValueError, match="requires manifest.json"):
        load_rgbd_session(root)


def test_load_rgbd_session_accepts_calibration_alignment_marker(
    tmp_path: Path,
) -> None:
    root = _write_rgbd_session(
        tmp_path / "run",
        alignment_in_calibration=True,
        manifest_align=None,
    )

    assert len(load_rgbd_session(root / "frames.csv").frames) == 1


def test_read_aligned_depth_converts_device_units_to_mm(tmp_path: Path) -> None:
    root = _write_rgbd_session(tmp_path / "run", depth_scale="0.1")
    frame = load_rgbd_session(root).frames[0]

    depth_mm = read_aligned_depth_mm(frame)

    assert depth_mm.dtype == np.float32
    assert depth_mm.shape == (24, 32)
    np.testing.assert_allclose(depth_mm, 1000.0)


def test_load_rgbd_session_rejects_missing_or_invalid_calibration(
    tmp_path: Path,
) -> None:
    root = _write_rgbd_session(tmp_path / "missing")
    (root / "calibration.json").unlink()
    with pytest.raises(FileNotFoundError, match="Missing calibration.json"):
        load_rgbd_session(root)

    invalid_root = _write_rgbd_session(tmp_path / "invalid")
    (invalid_root / "calibration.json").write_text("not-json", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid calibration.json"):
        load_rgbd_session(invalid_root)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("fx", float("nan"), "must be finite and positive"),
        ("fx", 0.0, "must be finite and positive"),
        ("cx", 100.0, "principal point"),
    ],
)
def test_load_rgbd_session_rejects_invalid_color_intrinsics(
    tmp_path: Path, field: str, value: float, message: str
) -> None:
    root = _write_rgbd_session(tmp_path / "run")
    path = root / "calibration.json"
    calibration = json.loads(path.read_text(encoding="utf-8"))
    calibration["color_intrinsic"][field] = value
    path.write_text(json.dumps(calibration), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_rgbd_session(root)


def test_load_rgbd_session_rejects_calibration_image_size_mismatch(
    tmp_path: Path,
) -> None:
    root = _write_rgbd_session(
        tmp_path / "run", calibration_width=31, calibration_height=24
    )

    with pytest.raises(ValueError, match="does not match calibration"):
        load_rgbd_session(root)


def test_load_rgbd_session_requires_explicit_alignment_marker(tmp_path: Path) -> None:
    root = _write_rgbd_session(tmp_path / "run", manifest_align=None)

    with pytest.raises(ValueError, match="explicitly declare depth aligned to color"):
        load_rgbd_session(root)


@pytest.mark.parametrize(
    "align_mode", ["none", "disabled", "off", "raw", "depth", "ir", "hardware", "typo"]
)
def test_load_rgbd_session_rejects_disabled_alignment(
    tmp_path: Path, align_mode: str
) -> None:
    root = _write_rgbd_session(tmp_path / align_mode, manifest_align=align_mode)

    with pytest.raises(ValueError, match="explicitly declare depth aligned to color"):
        load_rgbd_session(root)


def test_load_rgbd_session_rejects_unproven_foreign_software_alignment(
    tmp_path: Path,
) -> None:
    root = _write_rgbd_session(tmp_path / "run", manifest_align="software")
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema"] = "foreign-session/v1"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported manifest schema"):
        load_rgbd_session(root)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"timestamp": ""}, "color timestamp"),
        ({"timestamp": "-1"}, "color timestamp"),
        ({"color_exposure": ""}, "color_exposure metadata"),
        ({"color_exposure": "0"}, "color_exposure metadata"),
    ],
)
def test_load_rgbd_session_requires_complete_timing_and_exposure_metadata(
    tmp_path: Path, kwargs: dict[str, str], message: str
) -> None:
    root = _write_rgbd_session(tmp_path / "run", **kwargs)

    with pytest.raises(ValueError, match=message):
        load_rgbd_session(root)


def test_load_rgbd_session_does_not_accept_depth_path_alias(tmp_path: Path) -> None:
    root = _write_rgbd_session(tmp_path / "run")
    csv_path = root / "frames.csv"
    csv_path.write_text(
        "frame_id,color_path,depth_path,depth_scale_mm_per_unit\n"
        "0,color/00000000.jpg,depth_aligned/00000000.png,0.1\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing formal RGB-D columns"):
        load_rgbd_session(root)


@pytest.mark.parametrize("raw_alias", ["same_file", "raw_directory", "unproven_directory"])
def test_load_rgbd_session_rejects_raw_depth_masquerading_as_aligned(
    tmp_path: Path, raw_alias: str
) -> None:
    if raw_alias == "same_file":
        root = _write_rgbd_session(
            tmp_path / "run",
            raw_depth_path="depth_aligned/00000000.png",
        )
    elif raw_alias == "raw_directory":
        root = _write_rgbd_session(
            tmp_path / "run",
            aligned_depth_path="depth_raw/00000000.png",
        )
    else:
        root = _write_rgbd_session(
            tmp_path / "run",
            aligned_depth_path="depth/00000000.png",
        )

    with pytest.raises(ValueError, match="raw depth|depth_aligned directory"):
        load_rgbd_session(root)


@pytest.mark.parametrize("scale", ["", "0", "-0.1", "nan", "inf"])
def test_load_rgbd_session_rejects_invalid_depth_scale(
    tmp_path: Path, scale: str
) -> None:
    root = _write_rgbd_session(tmp_path / "run", depth_scale=scale)

    with pytest.raises(ValueError, match="depth scale"):
        load_rgbd_session(root)


def test_load_rgbd_session_rejects_missing_corrupt_or_mismatched_depth(
    tmp_path: Path,
) -> None:
    missing = _write_rgbd_session(tmp_path / "missing")
    (missing / "depth_aligned/00000000.png").unlink()
    with pytest.raises(FileNotFoundError, match="Missing aligned depth image"):
        load_rgbd_session(missing)

    corrupt = _write_rgbd_session(tmp_path / "corrupt")
    (corrupt / "depth_aligned/00000000.png").write_bytes(b"not-a-png")
    with pytest.raises(ValueError, match="Could not decode aligned depth image"):
        load_rgbd_session(corrupt)

    mismatch = _write_rgbd_session(tmp_path / "mismatch", depth_shape=(23, 32))
    with pytest.raises(ValueError, match="does not match color/calibration"):
        load_rgbd_session(mismatch)
