"""Export an independently reproducible, real ORB-SLAM3 RGB-D trajectory.

This command deliberately invokes the same strict-session and ORB-SLAM3 bridge
used by the formal pipeline.  It never reads a historical pose sidecar and it
never substitutes Open3D odometry for a missing ORB-SLAM3 pose.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

from .config import load_config
from .orbslam3_bridge import ORBSLAM3Config, run_orbslam3_rgbd
from .session import load_rgbd_session


TRAJECTORY_SCHEMA = "gemini305-orbslam3-trajectory/v2"
TRAJECTORY_AUDIT_SCHEMA = "gemini305-orbslam3-trajectory-audit/v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_copy(source: Path, destination: Path) -> None:
    pending = destination.with_name(f".{destination.name}.pending")
    shutil.copyfile(source, pending)
    os.replace(pending, destination)


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
    stdout_output = destination.with_suffix(".stdout.txt")
    stderr_output = destination.with_suffix(".stderr.txt")
    audit_output = destination.with_suffix(".audit.json")
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
            if frame.timestamp_us is None or frame.timestamp_us < 0:
                raise RuntimeError(
                    f"ORB-SLAM3 source frame {frame.frame_id} has no valid timestamp"
                )
            matrices.append(
                {
                    "frame_id": frame.frame_id,
                    "timestamp_us": int(frame.timestamp_us),
                    "pose_status": "valid",
                    "pose_kind": "direct_orbslam3",
                    "pose_origin": "offline_orbslam3_rgbd_full_session",
                    "tracking_state": "tracked",
                    "camera_to_world": matrix.tolist(),
                }
            )
        payload = trajectory.as_dict(input_frame_count=len(session.frames))
        payload.update(
            {
                "schema": TRAJECTORY_SCHEMA,
                "session": str(session.root),
                "complete_tracking_required": True,
                "pose_record_contract": "explicit-direct-orbslam3-camera-to-world/v1",
                "untracked_frame_ids": [],
                "stdout_file": stdout_output.name,
                "stderr_file": stderr_output.name,
                "tum_file": destination.with_suffix(".tum.txt").name,
                "association_file": destination.with_suffix(".association.txt").name,
                "poses": matrices,
            }
        )
        stdout_source = Path(trajectory.stdout_path)
        stderr_source = Path(trajectory.stderr_path)
        if not stdout_source.is_file() or not stderr_source.is_file():
            raise RuntimeError("ORB-SLAM3 did not preserve stdout/stderr audit files")
        _atomic_copy(stdout_source, stdout_output)
        _atomic_copy(stderr_source, stderr_output)
        _atomic_copy(Path(trajectory.trajectory_path), destination.with_suffix(".tum.txt"))
        _atomic_copy(Path(trajectory.association_path), destination.with_suffix(".association.txt"))
        pending = destination.parent / f".{destination.name}.pending"
        pending.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(pending, destination)
        config_resolved = (
            Path(config_path).expanduser().resolve() if config_path is not None else None
        )
        effective_config = json.dumps(
            stitch.get("orbslam3_rgbd"), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        audit = {
            "schema": TRAJECTORY_AUDIT_SCHEMA,
            "trajectory": destination.name,
            "trajectory_schema": TRAJECTORY_SCHEMA,
            "trajectory_sha256": _sha256(destination),
            "input_frame_count": len(session.frames),
            "tracked_frame_count": len(trajectory.tracked_frame_ids),
            "untracked_frame_ids": [],
            "config_path": str(config_resolved) if config_resolved is not None else None,
            "config_sha256": (
                _sha256(config_resolved) if config_resolved is not None else None
            ),
            "effective_orbslam3_config_sha256": hashlib.sha256(
                effective_config
            ).hexdigest(),
            "stdout_file": stdout_output.name,
            "stdout_sha256": _sha256(stdout_output),
            "stderr_file": stderr_output.name,
            "stderr_sha256": _sha256(stderr_output),
        }
        audit_pending = audit_output.with_name(f".{audit_output.name}.pending")
        audit_pending.write_text(json.dumps(audit, indent=2), encoding="utf-8")
        os.replace(audit_pending, audit_output)
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
