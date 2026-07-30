"""Export an independently reproducible, real ORB-SLAM3 RGB-D trajectory.

This command deliberately invokes the same strict-session and ORB-SLAM3 bridge
used by the formal pipeline.  It never reads a historical pose sidecar and it
never substitutes Open3D odometry for a missing ORB-SLAM3 pose.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

from .config import load_config
from .orbslam3_bridge import ORBSLAM3Config, run_orbslam3_rgbd
from .session import load_rgbd_session


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export the complete current ORB-SLAM3 RGB-D trajectory"
    )
    parser.add_argument("input", type=Path, help="Strict calibrated RGB-D session")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("orbslam3_trajectory.json"),
        help="Atomic JSON output path (default: ./orbslam3_trajectory.json)",
    )
    parser.add_argument("--config", type=Path, help="Optional merged YAML override")
    return parser


def export_trajectory(
    input_path: str | Path,
    output_path: str | Path,
    *,
    config_path: str | Path | None = None,
) -> dict[str, object]:
    """Run ORB-SLAM3 over the complete strict session and atomically export it."""

    session = load_rgbd_session(input_path)
    config = load_config(config_path)
    stitch = config.get("stitch")
    if not isinstance(stitch, dict):
        raise ValueError("Configuration is missing stitch settings")
    orb_config = ORBSLAM3Config.from_mapping(stitch.get("orbslam3_rgbd"))
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Staging is deliberately independent from the final artifact.  It may
    # contain ORB's temporary undistorted inputs, while the output contains
    # only scalar audit data and complete metric camera-to-world transforms.
    with tempfile.TemporaryDirectory(prefix="g305-orbslam3-", dir=destination.parent) as work:
        trajectory = run_orbslam3_rgbd(
            session.frames,
            session.calibration,
            Path(work),
            config=orb_config,
        )
        if tuple(trajectory.tracked_frame_ids) != tuple(frame.frame_id for frame in session.frames):
            raise RuntimeError("ORB-SLAM3 did not track every requested frame")
        matrices: list[dict[str, object]] = []
        for frame in session.frames:
            matrix = np.asarray(trajectory.poses_by_frame_id.get(frame.frame_id), dtype=np.float64)
            if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
                raise RuntimeError(f"ORB-SLAM3 returned no finite pose for frame {frame.frame_id}")
            matrices.append({"frame_id": frame.frame_id, "camera_to_world": matrix.tolist()})
        payload = trajectory.as_dict(input_frame_count=len(session.frames))
        payload.update(
            {
                "schema": "gemini305-orbslam3-trajectory/v1",
                "session": str(session.root),
                "complete_tracking_required": True,
                "poses": matrices,
            }
        )
        pending = destination.parent / f".{destination.name}.pending"
        pending.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(pending, destination)
    return payload


def main() -> None:
    args = _parser().parse_args()
    try:
        payload = export_trajectory(args.input, args.output, config_path=args.config)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print(f"ORB-SLAM3 trajectory: {Path(args.output).expanduser().resolve()}")
    print(f"Tracked frames: {payload['tracked_frame_count']}/{payload['input_frame_count']}")


if __name__ == "__main__":
    main()
