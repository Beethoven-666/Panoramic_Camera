from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


SYNTHETIC_SCENES = {
    "plane",
    "layered",
    "occlusion",
    "depth_hole",
    "dynamic_object",
}
FAR_DEPTH_MM = 2400
NEAR_DEPTH_MM = 900
DEPTH_SCALE_MM_PER_UNIT = 1.0


def _write_image(path: Path, image: np.ndarray, parameters: list[int]) -> None:
    if not cv2.imwrite(str(path), image, parameters):
        raise OSError(f"OpenCV could not write synthetic image: {path}")


def _add_static_near_layer(
    color: np.ndarray,
    depth: np.ndarray,
    *,
    frame_index: int,
    frame_count: int,
    step_px: int,
    fx: float,
    cx: float,
) -> None:
    """Render several world-fixed near panels with genuine lateral parallax."""

    height, width = depth.shape
    camera_step_mm = float(step_px) * FAR_DEPTH_MM / fx
    camera_x_mm = frame_index * camera_step_mm
    scan_extent_mm = max(0, frame_count - 1) * camera_step_mm
    near_fov_mm = NEAR_DEPTH_MM * width / fx
    spacing_mm = max(near_fov_mm * 0.55, camera_step_mm * 1.5, 1.0)
    first_center_mm = -near_fov_mm * 0.25
    last_center_mm = scan_extent_mm + near_fov_mm * 0.75
    panel_centers = np.arange(first_center_mm, last_center_mm, spacing_mm)
    half_width_px = max(3, int(round(width * 0.075)))
    top = max(1, int(round(height * 0.18)))
    bottom = min(height - 2, int(round(height * 0.84)))

    for panel_index, center_world_mm in enumerate(panel_centers):
        center_u = int(
            round(fx * (float(center_world_mm) - camera_x_mm) / NEAR_DEPTH_MM + cx)
        )
        left = max(0, center_u - half_width_px)
        right = min(width - 1, center_u + half_width_px)
        if left > right:
            continue
        panel_color = (
            45 + (37 * panel_index) % 150,
            75 + (53 * panel_index) % 145,
            90 + (29 * panel_index) % 135,
        )
        cv2.rectangle(color, (left, top), (right, bottom), panel_color, -1)
        cv2.rectangle(color, (left, top), (right, bottom), (235, 235, 235), 1)
        stripe_x = int(np.clip(center_u, left, right))
        cv2.line(color, (stripe_x, top), (stripe_x, bottom), (20, 20, 20), 1)
        depth[top : bottom + 1, left : right + 1] = NEAR_DEPTH_MM


def _add_dynamic_object(
    color: np.ndarray, depth: np.ndarray, *, frame_index: int
) -> None:
    height, width = depth.shape
    radius = max(3, min(width, height) // 12)
    span = max(1, width - 2 * radius - 2)
    center_x = radius + 1 + (frame_index * max(3, width // 9)) % span
    center_y = max(radius + 1, int(round(height * 0.58)))
    center_y = min(height - radius - 2, center_y)
    cv2.circle(color, (center_x, center_y), radius, (210, 35, 220), -1)
    cv2.circle(color, (center_x, center_y), radius, (250, 250, 250), 1)
    yy, xx = np.ogrid[:height, :width]
    mask = (xx - center_x) ** 2 + (yy - center_y) ** 2 <= radius**2
    depth[mask] = max(1, NEAR_DEPTH_MM - 200)


def _apply_scene_geometry(
    color: np.ndarray,
    *,
    scene: str,
    frame_index: int,
    frame_count: int,
    step_px: int,
    fx: float,
    cx: float,
) -> np.ndarray:
    depth = np.full(color.shape[:2], FAR_DEPTH_MM, dtype=np.uint16)
    if scene != "plane":
        _add_static_near_layer(
            color,
            depth,
            frame_index=frame_index,
            frame_count=frame_count,
            step_px=step_px,
            fx=fx,
            cx=cx,
        )
    if scene == "depth_hole":
        height, width = depth.shape
        x0, x1 = width // 3, min(width, width // 3 + max(2, width // 10))
        y0, y1 = height // 3, min(height, height // 3 + max(2, height // 5))
        depth[y0:y1, x0:x1] = 0
    elif scene == "dynamic_object":
        _add_dynamic_object(color, depth, frame_index=frame_index)
    return depth


def _calibration_payload(
    *, frame_width: int, frame_height: int, fx: float, fy: float
) -> dict[str, object]:
    intrinsic = {
        "width": frame_width,
        "height": frame_height,
        "fx": fx,
        "fy": fy,
        "cx": (frame_width - 1) / 2.0,
        "cy": (frame_height - 1) / 2.0,
    }
    distortion = {
        "k1": 0.0,
        "k2": 0.0,
        "k3": 0.0,
        "k4": 0.0,
        "k5": 0.0,
        "k6": 0.0,
        "p1": 0.0,
        "p2": 0.0,
    }
    return {
        "schema": "panorama-demo-calibration/v2",
        "color_intrinsic": intrinsic,
        "color_distortion": distortion,
        "depth_intrinsic": dict(intrinsic),
        "depth_distortion": dict(distortion),
        "depth_to_color": {
            "rotation_row_major": np.eye(3, dtype=np.float64).reshape(-1).tolist(),
            "translation_mm": [0.0, 0.0, 0.0],
        },
        "depth_alignment": {
            "aligned_to": "color",
            "enabled": True,
            "method": "synthetic",
        },
    }


def _known_trajectory(
    *, frame_count: int, step_px: int, fx: float
) -> dict[str, object]:
    camera_step_mm = float(step_px) * FAR_DEPTH_MM / fx
    poses: list[dict[str, object]] = []
    for frame_id in range(frame_count):
        camera_to_world = np.eye(4, dtype=np.float64)
        camera_to_world[0, 3] = frame_id * camera_step_mm
        poses.append(
            {
                "frame_id": frame_id,
                "matrix_row_major": camera_to_world.reshape(-1).tolist(),
            }
        )
    return {
        "transform": "camera_to_world",
        "translation_unit": "millimetres",
        "camera_axes": {"x": "right", "y": "down", "z": "forward"},
        "world_scan_axis": "+x",
        "poses": poses,
    }


def generate_sequence(
    output: Path,
    *,
    frame_count: int = 10,
    frame_width: int = 640,
    frame_height: int = 400,
    step: int = 120,
    seed: int = 7,
    scene: str = "plane",
) -> Path:
    scene = scene.strip().lower()
    if scene not in SYNTHETIC_SCENES:
        choices = ", ".join(sorted(SYNTHETIC_SCENES))
        raise ValueError(f"Unsupported synthetic scene {scene!r}; choose one of: {choices}")
    if frame_count < 1:
        raise ValueError("frame_count must be at least 1")
    if frame_width < 16 or frame_height < 16:
        raise ValueError("synthetic frame dimensions must be at least 16 pixels")
    if step < 0:
        raise ValueError("step must be non-negative")

    output = output.resolve()
    color_dir = output / "color"
    depth_dir = output / "depth_aligned"
    color_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    world_width = frame_width + step * max(0, frame_count - 1)
    world = np.full((frame_height, world_width, 3), (48, 74, 44), dtype=np.uint8)

    # Repeated vertical racks deliberately make the sequence less trivial.
    for x in range(30, world_width, 95):
        cv2.rectangle(world, (x, 10), (x + 10, frame_height - 10), (150, 160, 165), -1)
        cv2.line(world, (x - 22, frame_height // 2), (x + 35, frame_height // 2), (130, 140, 145), 5)
    y_margin = min(20, max(1, frame_height // 4))
    for _ in range(max(80, world_width // 8)):
        x = int(rng.integers(0, world_width))
        y = int(rng.integers(y_margin, max(y_margin + 1, frame_height - y_margin)))
        radius = int(rng.integers(3, 14))
        color = tuple(int(value) for value in rng.integers(40, 235, size=3))
        cv2.circle(world, (x, y), radius, color, -1, lineType=cv2.LINE_AA)
    for index, x in enumerate(range(60, world_width, 260)):
        cv2.putText(
            world,
            f"R{index:02d}",
            (x, 70 + 45 * (index % 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (245, 245, 245),
            2,
            cv2.LINE_AA,
        )

    fx = float(frame_width) * 0.85
    fy = fx
    cx = (frame_width - 1) / 2.0
    rows: list[dict[str, object]] = []
    for index in range(frame_count):
        x = index * step
        frame = world[:, x : x + frame_width].copy()
        gain = 0.94 + 0.012 * index
        frame = np.clip(frame.astype(np.float32) * gain, 0, 255).astype(np.uint8)
        depth = _apply_scene_geometry(
            frame,
            scene=scene,
            frame_index=index,
            frame_count=frame_count,
            step_px=step,
            fx=fx,
            cx=cx,
        )
        color_relative = Path("color") / f"{index:08d}.jpg"
        depth_relative = Path("depth_aligned") / f"{index:08d}.png"
        _write_image(
            output / color_relative,
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, 97],
        )
        _write_image(
            output / depth_relative,
            depth,
            [cv2.IMWRITE_PNG_COMPRESSION, 1],
        )
        rows.append(
            {
                "frame_id": index,
                "color_device_timestamp_us": index * 100_000,
                "color_exposure": 8,
                "color_gain": 16,
                "depth_scale_mm_per_unit": DEPTH_SCALE_MM_PER_UNIT,
                "color_path": color_relative.as_posix(),
                "aligned_depth_path": depth_relative.as_posix(),
                "raw_depth_path": "",
            }
        )

    with (output / "frames.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "calibration.json").write_text(
        json.dumps(
            _calibration_payload(
                frame_width=frame_width,
                frame_height=frame_height,
                fx=fx,
                fy=fy,
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "panorama-demo-session/v2",
                "synthetic": True,
                "frame_count": frame_count,
                "known_step_px": step,
                "frame_size": [frame_width, frame_height],
                "scene": scene,
                "capture_options": {
                    "width": frame_width,
                    "height": frame_height,
                    "align": "software",
                    "frame_sync": True,
                },
                "frame_sync": True,
                "clean_shutdown": True,
                "queue_drops": 0,
                "write_errors": 0,
                "depth_scale_mm_per_unit": DEPTH_SCALE_MM_PER_UNIT,
                "synthetic_depth_mm": {
                    "far": FAR_DEPTH_MM,
                    "near": NEAR_DEPTH_MM,
                    "invalid": 0,
                },
                "known_trajectory": _known_trajectory(
                    frame_count=frame_count,
                    step_px=step,
                    fx=fx,
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a deterministic side-scan demo sequence")
    parser.add_argument("--output", type=Path, default=Path("data/synthetic/demo"))
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=400)
    parser.add_argument("--step", type=int, default=120)
    parser.add_argument("--scene", choices=sorted(SYNTHETIC_SCENES), default="plane")
    args = parser.parse_args()
    destination = generate_sequence(
        args.output,
        frame_count=args.frames,
        frame_width=args.width,
        frame_height=args.height,
        step=args.step,
        scene=args.scene,
    )
    print(destination)


if __name__ == "__main__":
    main()
