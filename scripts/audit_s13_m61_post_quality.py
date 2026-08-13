"""Write reporting-only M6.1 post-render quality and top-10 atlas."""

from __future__ import annotations

import argparse
import json
import os
import uuid
from pathlib import Path

import cv2
import numpy as np

from panorama_demo.video_s13_m61_post_quality import (
    evaluate_m61_post_render_quality,
    write_m61_top10_atlas,
)


def _image(path: Path) -> np.ndarray:
    value = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if value is None:
        raise ValueError(f"unreadable image: {path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p2-root", required=True, type=Path)
    parser.add_argument("--p3-root", required=True, type=Path)
    parser.add_argument("--old-p3", required=True, type=Path)
    parser.add_argument("--threshold", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("post-quality output must be new")
    pairs = []
    for path in sorted((args.p2_root / "photometric_replay").glob("pair_*.npz")):
        with np.load(path, allow_pickle=False) as archive:
            pairs.append({name: np.asarray(archive[name]) for name in archive.files})
    with np.load(args.p2_root / "p2_pixel_provenance.npz", allow_pickle=False) as archive:
        owner = np.asarray(archive["owner_source_index"])
        valid = np.asarray(archive["valid"], bool)
    with np.load(args.p3_root / "pair_masks.npz", allow_pickle=False) as archive:
        active = np.asarray(archive["active"], bool)
    report = evaluate_m61_post_render_quality(
        _image(args.p2_root / "geometry_and_seam_panorama_owner_only.png"),
        _image(args.p3_root / "visual_panorama.png"),
        _image(args.old_p3 / "visual_panorama.png"),
        pairs, owner, valid,
        json.loads(args.threshold.read_text(encoding="utf-8")),
        blend_active_mask=active,
    )
    pending = args.output.with_name(f".{args.output.name}.{uuid.uuid4().hex}.pending")
    pending.mkdir(parents=True)
    (pending / "post_quality.json").write_text(
        json.dumps(report, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    write_m61_top10_atlas(pending / "top10_atlas", _image(args.p3_root / "visual_panorama.png"), pairs, report)
    os.replace(pending, args.output)
    print(json.dumps({"quality_state": report["quality_state"], "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
