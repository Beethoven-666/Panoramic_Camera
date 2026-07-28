"""Render Ultralytics segmentation polygons without detector boxes or labels.

This is an independent diagnostic helper.  It does not participate in the
formal RGB-D renderer and never modifies a source image.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("labels", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--minimum-area", type=int, default=250)
    return parser


def _colour(index: int) -> tuple[int, int, int]:
    hue = int((index * 47) % 180)
    hsv = np.uint8([[[hue, 220, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return tuple(int(value) for value in bgr)


def main() -> None:
    args = _parser().parse_args()
    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(args.image)
    height, width = image.shape[:2]
    overlay = image.copy()
    outlines: list[tuple[int, np.ndarray, tuple[int, int, int]]] = []
    for line_index, line in enumerate(
        args.labels.read_text(encoding="utf-8").splitlines()
    ):
        values = np.asarray([float(value) for value in line.split()])
        if values.size < 7 or (values.size - 1) % 2:
            continue
        points = values[1:].reshape(-1, 2)
        polygon = np.rint(
            points * np.asarray([width, height], dtype=np.float64)
        ).astype(np.int32)
        area = float(abs(cv2.contourArea(polygon)))
        if area < int(args.minimum_area):
            continue
        colour = _colour(line_index)
        cv2.fillPoly(overlay, [polygon], colour)
        outlines.append((line_index, polygon, colour))
    rendered = cv2.addWeighted(image, 0.55, overlay, 0.45, 0.0)
    for line_index, polygon, colour in outlines:
        cv2.polylines(rendered, [polygon], True, colour, 2, cv2.LINE_AA)
        moments = cv2.moments(polygon)
        if abs(moments["m00"]) <= 1e-6:
            continue
        center = (
            int(round(moments["m10"] / moments["m00"])),
            int(round(moments["m01"] / moments["m00"])),
        )
        cv2.putText(
            rendered,
            str(line_index),
            center,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            rendered,
            str(line_index),
            center,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), rendered):
        raise RuntimeError(f"Could not write {args.output}")


if __name__ == "__main__":
    main()
