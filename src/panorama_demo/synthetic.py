from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


def generate_sequence(
    output: Path,
    *,
    frame_count: int = 10,
    frame_width: int = 640,
    frame_height: int = 400,
    step: int = 120,
    seed: int = 7,
) -> Path:
    output = output.resolve()
    color_dir = output / "color"
    color_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    world_width = frame_width + step * max(0, frame_count - 1)
    world = np.full((frame_height, world_width, 3), (48, 74, 44), dtype=np.uint8)

    # Repeated vertical racks deliberately make the sequence less trivial.
    for x in range(30, world_width, 95):
        cv2.rectangle(world, (x, 10), (x + 10, frame_height - 10), (150, 160, 165), -1)
        cv2.line(world, (x - 22, frame_height // 2), (x + 35, frame_height // 2), (130, 140, 145), 5)
    for _ in range(max(80, world_width // 8)):
        x = int(rng.integers(0, world_width))
        y = int(rng.integers(20, frame_height - 20))
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

    rows: list[dict[str, object]] = []
    for index in range(frame_count):
        x = index * step
        frame = world[:, x : x + frame_width].copy()
        gain = 0.94 + 0.012 * index
        frame = np.clip(frame.astype(np.float32) * gain, 0, 255).astype(np.uint8)
        relative = Path("color") / f"{index:08d}.jpg"
        cv2.imwrite(str(output / relative), frame, [cv2.IMWRITE_JPEG_QUALITY, 97])
        rows.append(
            {
                "frame_id": index,
                "color_device_timestamp_us": index * 100_000,
                "color_path": relative.as_posix(),
            }
        )

    with (output / "frames.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "panorama-demo-session/v1",
                "synthetic": True,
                "frame_count": frame_count,
                "known_step_px": step,
                "frame_size": [frame_width, frame_height],
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
    args = parser.parse_args()
    destination = generate_sequence(
        args.output,
        frame_count=args.frames,
        frame_width=args.width,
        frame_height=args.height,
        step=args.step,
    )
    print(destination)


if __name__ == "__main__":
    main()
