"""Freeze a verified video-report ORB chain for deterministic experiments.

The public video renderer accepts a trajectory cache only when it describes
the exact current session.  This module materialises that cache from a prior
*published* video report without estimating, interpolating, or otherwise
altering any pose.  It is deliberately separate from the renderer so an
experiment can reuse a completed full real ORB chain deterministically.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np


TRAJECTORY_CACHE_SCHEMA = "gemini305-video-experiment-trajectory-cache/v1"
VIDEO_REPORT_SCHEMA = "gemini305-video-panorama-report/v2"
VIDEO_DELIVERY_SCHEMA = "gemini305-video-panorama-delivery/v2"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, *, description: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid {description}: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{description.capitalize()} must contain a JSON object")
    return payload


def _session_root(input_path: Path) -> Path:
    root = input_path.expanduser().resolve()
    return root if root.is_dir() else root.parent


def session_input_sha256(input_path: Path) -> dict[str, str]:
    """Return the exact session hashes checked by ``video_panorama``."""

    root = _session_root(input_path)
    names = {
        "manifest": "manifest.json",
        "calibration": "calibration.json",
        "frames_csv": "frames.csv",
    }
    hashes: dict[str, str] = {}
    for key, name in names.items():
        path = root / name
        if not path.is_file():
            raise ValueError(f"Video session is missing {name}: {root}")
        hashes[key] = _sha256(path)
    return hashes


def _require_pose(value: object, *, frame_id: int) -> None:
    pose = np.asarray(value, dtype=np.float64)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError(f"Video report pose for frame {frame_id} is not finite 4x4 SE(3)")
    rotation = pose[:3, :3]
    if not np.allclose(pose[3], (0.0, 0.0, 0.0, 1.0), atol=1e-8):
        raise ValueError(f"Video report pose for frame {frame_id} has invalid homogeneous row")
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-4) or not np.isclose(
        np.linalg.det(rotation), 1.0, atol=1e-4
    ):
        raise ValueError(f"Video report pose for frame {frame_id} is not rigid")


def _require_exact_hashes(value: object, *, current: dict[str, str]) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError("Video report lacks input_sha256")
    recorded = {key: value.get(key) for key in current}
    if not all(isinstance(item, str) and len(item) == 64 for item in recorded.values()):
        raise ValueError("Video report has malformed input_sha256")
    if recorded != current:
        raise ValueError("Video report input hashes do not match this video session")
    return dict(current)


def _verified_orb_from_report(
    report: dict[str, Any], *, current_hashes: dict[str, str]
) -> dict[str, Any]:
    if report.get("schema") != VIDEO_REPORT_SCHEMA:
        raise ValueError("Trajectory freeze requires a completed v2 video report")
    if report.get("delivery_state") not in {"published", "published_degraded"}:
        raise ValueError("Trajectory freeze requires a published video report")
    grades = report.get("grades")
    if not isinstance(grades, dict) or grades.get("structural") != "A":
        raise ValueError("Trajectory freeze requires structural grade A")
    if report.get("orb_tracking_source") != "real_time_spaced_video_frames_only":
        raise ValueError("Video report does not certify real video ORB tracking sources")
    _require_exact_hashes(report.get("input_sha256"), current=current_hashes)

    tracked = report.get("orb_tracking_frame_ids")
    all_motion = report.get("motion_analysis_frame_ids")
    orb = report.get("orbslam3")
    if not isinstance(tracked, list) or not isinstance(all_motion, list) or not isinstance(orb, dict):
        raise ValueError("Video report lacks complete ORB trajectory metadata")
    if not tracked or not all(isinstance(frame_id, int) for frame_id in tracked):
        raise ValueError("Video report has malformed ORB tracking frame ids")
    if len(set(tracked)) != len(tracked) or tracked != sorted(tracked):
        raise ValueError("Video report ORB tracking frames must be unique and chronological")
    if not all(isinstance(frame_id, int) for frame_id in all_motion):
        raise ValueError("Video report has malformed motion-analysis frame ids")
    motion_positions = {frame_id: index for index, frame_id in enumerate(all_motion)}
    if any(frame_id not in motion_positions for frame_id in tracked) or any(
        motion_positions[left] >= motion_positions[right]
        for left, right in zip(tracked, tracked[1:])
    ):
        raise ValueError("Video report ORB tracking frames are not a real chronological scan subset")

    ids = orb.get("tracked_frame_ids")
    poses = orb.get("camera_to_world")
    if ids != tracked or not isinstance(poses, list) or len(poses) != len(ids):
        raise ValueError("Video report ORB pose array does not exactly cover tracked real frames")
    if orb.get("pose_convention") != "camera_to_world" or orb.get("translation_unit") != "mm":
        raise ValueError("Video report ORB trajectory convention is unsupported")
    for frame_id, pose in zip(ids, poses, strict=True):
        _require_pose(pose, frame_id=frame_id)
    attempts = orb.get("attempts", [])
    if not isinstance(attempts, list) or not all(isinstance(item, dict) for item in attempts):
        raise ValueError("Video report ORB execution-attempt audit is malformed")
    return {
        "tracked_frame_ids": list(tracked),
        # deepcopy preserves the report's exact real pose JSON values.  The
        # numerical checks above are validation only, never pose processing.
        "camera_to_world": deepcopy(poses),
        "attempts": deepcopy(attempts),
        "pose_convention": "camera_to_world",
        "translation_unit": "mm",
    }


def freeze_verified_trajectory(
    *, input_path: Path, report_path: Path, output_path: Path
) -> dict[str, Any]:
    """Create one immutable cache from a published, input-matching report.

    The function never overwrites a different cache.  Repeating the exact
    freeze is idempotent; otherwise the caller must choose a new output path.
    """

    report_path = report_path.expanduser().resolve()
    if report_path.is_dir():
        report_path = report_path / "video_report.json"
    report = _read_json(report_path, description="video report")
    delivery_path = report_path.parent / "video_delivery.json"
    delivery = _read_json(delivery_path, description="video delivery")
    if delivery.get("schema") != VIDEO_DELIVERY_SCHEMA or delivery.get("report") != report_path.name:
        raise ValueError("Trajectory freeze requires the report's matching atomic video delivery")
    if delivery.get("delivery_state") != report.get("delivery_state"):
        raise ValueError("Video delivery state does not match video report")

    current_hashes = session_input_sha256(input_path)
    orb = _verified_orb_from_report(report, current_hashes=current_hashes)
    payload = {
        "schema": TRAJECTORY_CACHE_SCHEMA,
        "input_sha256": current_hashes,
        "orbslam3": orb,
        "source_video_report_sha256": _sha256(report_path),
        "source_video_delivery_sha256": _sha256(delivery_path),
        "source_report_schema": VIDEO_REPORT_SCHEMA,
        "cache_origin": "published_verified_real_orbslam3_video_report",
    }
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    target = output_path.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.is_file() and target.read_bytes() == encoded:
            return {"cache": str(target), "created": False, "tracked_frame_count": len(orb["tracked_frame_ids"])}
        raise FileExistsError(f"Refusing to overwrite a different trajectory cache: {target}")
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=target.parent, prefix=f".{target.name}.", suffix=".pending", delete=False
    ) as handle:
        pending = Path(handle.name)
        try:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            pending.unlink(missing_ok=True)
            raise
    try:
        os.replace(pending, target)
    finally:
        pending.unlink(missing_ok=True)
    return {"cache": str(target), "created": True, "tracked_frame_count": len(orb["tracked_frame_ids"])}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Freeze a verified real ORB-SLAM3 video trajectory for deterministic experiments"
    )
    parser.add_argument("input", type=Path, help="Original completed RGB-D video session")
    parser.add_argument("--report", type=Path, required=True, help="Published video_report.json or its directory")
    parser.add_argument("--output", type=Path, required=True, help="New trajectory cache JSON path")
    return parser


def main() -> None:
    args = _parser().parse_args()
    try:
        frozen = freeze_verified_trajectory(
            input_path=args.input, report_path=args.report, output_path=args.output
        )
    except Exception as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    action = "created" if frozen["created"] else "already exists"
    print(f"Trajectory cache {action}: {frozen['cache']}")


if __name__ == "__main__":
    main()
