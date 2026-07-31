"""Independent continuous-video panorama orchestration.

This intentionally does not import ``stitch_sequence``: photo publication and
video publication have different eligibility and marker schemas.
"""

from __future__ import annotations

import argparse
import hashlib
import tempfile
import time
from pathlib import Path
from typing import Any


from .calibrated_rgb_pushbroom import render_calibrated_rgb_pushbroom
from .config import load_config
from .orbslam3_bridge import run_orbslam3_rgbd
from .quality import assess_capture_quality
from .rgbd_odometry import RGBDOdometryConfig, estimate_pair_rgbd_odometry
from .video_delivery import invalidate_video_delivery, publish_video_2d, write_video_failure
from .video_3d import publish_video_3d
from .video_scan_segment import analyse_video_scan
from .video_session import load_video_session
from .video_source_selection import select_video_render_sources


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build an independent C/A/B video panorama")
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/video_sequence"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--defer-3d", action="store_true")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.expanduser().resolve()
    input_path = args.input.expanduser().resolve()
    invalidate_video_delivery(output)
    started = time.perf_counter()
    two_d_published = False
    try:
        config = load_config(args.config)
        stitch = dict(config.get("stitch", {}))
        video = load_video_session(input_path)
        qualities, motions, segment = analyse_video_scan(video.rgbd.frames, analysis_width=int(stitch.get("analysis_width", 320)))
        first, last = int(segment["start_index"]), int(segment["end_index"])
        scan_frames = tuple(video.rgbd.frames[first:last + 1])
        sources = select_video_render_sources(scan_frames)
        trajectory_config = dict(stitch.get("orbslam3_rgbd", {}))
        # A video source cannot be rendered without a genuine ORB pose.
        trajectory_config["minimum_tracked_fraction"] = 1.0
        with tempfile.TemporaryDirectory(prefix="g305-video-orbslam3-") as work:
            trajectory = run_orbslam3_rgbd(sources, video.rgbd.calibration, work, config=trajectory_config)
        poses = [trajectory.poses_by_frame_id[frame.frame_id] for frame in sources]
        odometry_config = RGBDOdometryConfig.from_mapping(stitch.get("rgbd_odometry"))
        edges = [
            estimate_pair_rgbd_odometry(left, right, video.rgbd.calibration, config=odometry_config, reference_node_id=left.frame_id, source_node_id=right.frame_id)
            for left, right in zip(sources, sources[1:])
        ]
        if len(edges) != len(sources) - 1:
            raise RuntimeError("Video Open3D audit did not cover every selected source edge")
        push_config = dict(stitch.get("calibrated_rgb_pushbroom", {}))
        push_config["max_pose_count"] = None
        push = render_calibrated_rgb_pushbroom(
            list(sources), poses, video.rgbd.calibration, config=push_config,
            rgb_motions=motions[first:last],
            motion_pixels_to_full_resolution=video.rgbd.calibration.width / float(stitch.get("analysis_width", 320)),
            multiband_levels=int(dict(stitch.get("scan_seam", {})).get("multiband_levels", 3)),
            quality_gate=False,
        )
        capture_quality = assess_capture_quality(
            qualities[first:last + 1], [frame.color_exposure_raw for frame in sources],
            exposure_unit_us=float(stitch.get("color_exposure_unit_us", 100.0)),
            maximum_exposure_us=float(stitch.get("maximum_motion_exposure_us", 1200.0)),
        )
        # Auto exposure and any strict-quality failure are deliberately C, not
        # structural F, after all trajectory/owner requirements succeeded.
        strict = bool(capture_quality["quality_pass"]) and video.capture_mode == "continuous_rgbd_video_fixed_exposure"
        grade = "A" if strict else "C"
        report: dict[str, Any] = {
            "input": str(input_path), "capture_mode": video.capture_mode,
            "legacy_v1_compatibility": video.legacy_v1,
            "delivery_state": "published" if strict else "published_degraded",
            "quality_grade": grade, "strict_quality_pass": strict,
            "manual_review_required": not strict,
            "defer_3d": bool(args.defer_3d), "scan_segment": segment,
            "source_frame_ids": [frame.frame_id for frame in sources],
            "input_sha256": {"manifest": _sha256(video.rgbd.root / "manifest.json"), "calibration": _sha256(video.rgbd.root / "calibration.json")},
            "capture_quality": capture_quality,
            "orbslam3": {
                "tracked_frame_ids": list(trajectory.tracked_frame_ids),
                "camera_to_world": [trajectory.poses_by_frame_id[frame_id].tolist() for frame_id in trajectory.tracked_frame_ids],
                "attempts": list(trajectory.attempt_audit),
                "pose_convention": "camera_to_world",
                "translation_unit": "mm",
            },
            "open3d_edges": [edge.as_dict() for edge in edges],
            "renderer": dict(push.metadata), "elapsed_seconds": time.perf_counter() - started,
        }
        published = publish_video_2d(output, push.panorama, push.owner_frame_id, report)
        two_d_published = True
        if not args.defer_3d:
            # This is deliberately an independent post-publication delivery:
            # a TSDF failure must never revoke the valid 2-D C/A/B marker.
            try:
                publish_video_3d(output, input_path=input_path, config=config)
            except Exception:
                # The independent 3-D helper has atomically recorded its own
                # failure marker.  Keep the already published 2-D delivery.
                published["three_d_delivery_state"] = "failed"
        return published
    except Exception as exc:
        if not two_d_published:
            write_video_failure(output, input_path, exc)
        raise


def main() -> None:
    args = _parser().parse_args()
    try:
        report = run(args)
    except Exception as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    print(f"Video panorama: {report['panorama']}")


if __name__ == "__main__":
    main()
