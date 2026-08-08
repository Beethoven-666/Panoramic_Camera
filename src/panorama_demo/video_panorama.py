"""Independent continuous-video panorama orchestration.

Photo and video publication intentionally remain separate.  Video fast mode
uses cheap full-sequence motion to select a smaller set of *real* render
sources, while ORB-SLAM3 still tracks the complete selected scan and Open3D
still audits every adjacent render-source edge.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from .calibrated_rgb_pushbroom import render_calibrated_rgb_pushbroom
from .config import load_config
from .orbslam3_bridge import run_orbslam3_rgbd
from .quality import assess_capture_quality
from .rgbd_odometry import (
    RGBDOdometryConfig,
    create_open3d_rgbd_odometry_backend,
    estimate_prepared_pair_rgbd_odometry,
    prepare_rgbd_odometry_frame,
)
from .video_delivery import (
    invalidate_video_delivery,
    publish_video_2d,
    write_invalid_candidate_experiment,
    write_video_failure,
)
from .video_3d import publish_video_3d
from .video_motion_resampler import (
    MotionResamplingConfig,
    compose_selected_motions,
    insert_v6_real_rescue_sources,
    reroute_v6_failed_pairs_with_real_sources,
    select_render_keyframes,
)
from .video_online_state import load_online_state
from .video_performance import VideoPerformanceProfiler
from .video_scan_segment import analyse_video_scan
from .video_session import load_video_session
from .video_source_selection import (
    assess_video_source_frontality,
    plan_frontality_owner_spans,
)
from .video_runtime_environment import atomic_write_json


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_pose(value: object, *, frame_id: int) -> np.ndarray:
    pose = np.asarray(value, dtype=np.float64)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError(f"Trajectory cache pose for frame {frame_id} is not finite 4x4 SE(3)")
    rotation = pose[:3, :3]
    if not np.allclose(pose[3], (0.0, 0.0, 0.0, 1.0), atol=1e-8):
        raise ValueError(f"Trajectory cache pose for frame {frame_id} has invalid homogeneous row")
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-4) or not np.isclose(
        np.linalg.det(rotation), 1.0, atol=1e-4
    ):
        raise ValueError(f"Trajectory cache pose for frame {frame_id} is not rigid")
    return pose


def _restrict_scan_to_progress_interval(
    frames: tuple,
    qualities: list,
    motions: list,
    interval: tuple[float, float] | None,
) -> tuple[tuple, list, list, dict[str, object] | None]:
    """Restrict rendering/tracking to actual sources in a locked motion interval.

    The scan analyser still establishes direction on the complete input, but
    all downstream decode, ORB tracking, Open3D edge audit and rendering use
    only the returned contiguous real-source subset.  No endpoint is
    interpolated and an interval with too little measured motion fails closed.
    """

    if interval is None:
        return frames, qualities, motions, None
    if len(frames) < 2 or len(motions) != len(frames) - 1 or len(qualities) != len(frames):
        raise ValueError("Progress restriction requires aligned scan frames, qualities, and motions")
    start, end = (float(interval[0]), float(interval[1]))
    if not (np.isfinite(start) and np.isfinite(end) and 0.0 <= start < end <= 1.0):
        raise ValueError("scan_progress_interval must satisfy 0 <= START < END <= 1")
    # The locked split has one explicit coordinate: cumulative *reliable
    # horizontal* motion.  Do not substitute Euclidean movement here, because
    # that could put the same real source in a different split than its fixed
    # measurement annotation.  The helper also rejects reverse reliable edges
    # instead of turning them into pseudo-progress.
    from .video_split import build_source_progress_evidence, source_progress_by_frame

    evidence = build_source_progress_evidence(frames, motions)
    progress_by_id = source_progress_by_frame(evidence)
    progress = np.asarray(
        [
            progress_by_id[
                int(frame.frame_id if hasattr(frame, "frame_id") else frame)
            ]
            for frame in frames
        ],
        dtype=np.float64,
    )
    # Nearest *contained* source nodes provide an unambiguous, real-frame
    # subset.  An interval must contain two sources, otherwise no genuine RGB-D
    # edge can be audited.
    selected = np.flatnonzero((progress >= start) & (progress <= end))
    if selected.size < 2:
        raise ValueError("scan_progress_interval contains fewer than two real source frames")
    first, last = int(selected[0]), int(selected[-1])
    if not np.array_equal(selected, np.arange(first, last + 1)):
        raise RuntimeError("Progress restriction selected a non-contiguous source sequence")
    return (
        frames[first : last + 1],
        qualities[first : last + 1],
        motions[first:last],
        {
            "requested": [start, end],
            "selected_source_indices": [first, last],
            "selected_source_progress": [float(progress[first]), float(progress[last])],
            "progress_coordinate": "cumulative_reliable_horizontal_motion",
            "source_progress_evidence_sha256": evidence["content_sha256"],
            "selection": "real_contiguous_sources_only",
        },
    )


def _cached_trajectory(
    path: Path,
    *,
    frames: tuple,
    input_sha256: dict[str, str],
    require_capture_provenance: bool,
) -> tuple[dict[int, np.ndarray], tuple[int, ...], list[dict[str, object]]]:
    """Load a prior full real ORB chain only after exact-input verification."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid trajectory cache: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Trajectory cache must contain an object")
    frame_ids = [frame.frame_id for frame in frames]
    cached_hashes = payload.get("input_sha256")
    # Both cache forms are valid only for the exact on-disk session contract.
    # In particular, ``frames.csv`` binds frame ids to the real RGB/depth
    # files that the cache pose chain was produced from.  Comparing only the
    # manifest and calibration would permit a changed source sequence to reuse
    # a trajectory from different real frames.
    hashes_match = cached_hashes == input_sha256
    if not hashes_match:
        raise ValueError("Trajectory cache input hashes do not match this video session")
    if require_capture_provenance:
        if payload.get("schema") != "gemini305-online-orbslam3-trajectory/v1" or payload.get("capture_origin") != "writer_committed_files":
            raise ValueError("Online trajectory lacks capture-time provenance")
        source_hashes = payload.get("source_file_sha256")
        if not isinstance(source_hashes, list):
            raise ValueError("Online trajectory lacks committed source-file hashes")
        hash_by_id = {
            int(item.get("frame_id")): item
            for item in source_hashes
            if isinstance(item, dict) and isinstance(item.get("frame_id"), int)
        }
        for frame in frames:
            recorded = hash_by_id.get(frame.frame_id)
            if recorded is None or recorded.get("color_sha256") != _sha256(frame.color_path) or recorded.get("aligned_depth_sha256") != _sha256(frame.aligned_depth_path):
                raise ValueError("Online trajectory source-file hashes do not match this video session")
    else:
        # A caller-provided cache is trusted only when it was atomically
        # materialised from a completed report by the dedicated verifier.  Do
        # not accept arbitrary report fragments or generic pose JSON here.
        from .video_trajectory_cache import TRAJECTORY_CACHE_SCHEMA

        if payload.get("schema") != TRAJECTORY_CACHE_SCHEMA:
            raise ValueError("Trajectory cache is not a verified experiment trajectory cache")
    orb = payload.get("orbslam3", payload)
    if not isinstance(orb, dict):
        raise ValueError("Trajectory cache lacks orbslam3 data")
    ids = orb.get("tracked_frame_ids")
    poses = orb.get("camera_to_world")
    if not isinstance(ids, list) or not isinstance(poses, list) or len(ids) != len(poses):
        raise ValueError("Trajectory cache has inconsistent tracked-frame and pose arrays")
    cached_poses = {
        int(frame_id): _require_pose(pose, frame_id=int(frame_id))
        for frame_id, pose in zip(ids, poses, strict=True)
    }
    if len(cached_poses) != len(ids) or any(frame_id not in cached_poses for frame_id in frame_ids):
        raise ValueError("Trajectory cache must cover every real frame in the selected scan")
    pose_by_id = {frame_id: cached_poses[frame_id] for frame_id in frame_ids}
    attempts = orb.get("attempts", orb.get("execution_attempts", []))
    audit = [dict(item) for item in attempts if isinstance(item, dict)]
    return pose_by_id, tuple(frame_ids), audit


def _resolve_cached_trajectory_path(args: argparse.Namespace, session_root: Path) -> tuple[Path, bool] | None:
    if args.trajectory_cache is not None:
        return args.trajectory_cache.expanduser().resolve(), False
    if args.reuse_online_trajectory:
        candidate = session_root / "online_orbslam3_trajectory.json"
        if candidate.is_file():
            return candidate, True
    return None


def _online_cached_tracking_frame_ids(path: Path) -> tuple[int, ...]:
    """Return the chronological real-frame IDs declared by an online chain.

    This is deliberately only a *planning* read.  The caller must subsequently
    pass the selected frames to :func:`_cached_trajectory`, which verifies the
    immutable session lock, capture provenance, source bytes, and rigid poses
    before any cached pose is used.  Reading these IDs here prevents a locked
    progress slice from re-phasing a capture-time 8 FPS chain at a different
    candidate tracking rate.
    """

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid trajectory cache: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Trajectory cache must contain an object")
    if (
        payload.get("schema") != "gemini305-online-orbslam3-trajectory/v1"
        or payload.get("capture_origin") != "writer_committed_files"
    ):
        raise ValueError("Online trajectory lacks capture-time provenance")
    orb = payload.get("orbslam3", payload)
    if not isinstance(orb, dict) or not isinstance(orb.get("tracked_frame_ids"), list):
        raise ValueError("Online trajectory lacks tracked real-frame IDs")
    try:
        ids = tuple(int(frame_id) for frame_id in orb["tracked_frame_ids"])
    except (TypeError, ValueError) as exc:
        raise ValueError("Online trajectory has invalid tracked real-frame IDs") from exc
    if len(ids) < 2 or any(next_id <= frame_id for frame_id, next_id in zip(ids, ids[1:])):
        raise ValueError("Online trajectory tracked real-frame IDs must be strictly increasing")
    return ids


def _contained_online_tracking_indices(scan_frames: tuple, cached_frame_ids: tuple[int, ...]) -> tuple[int, ...]:
    """Keep only certified cached anchors fully contained in a real-frame slice."""

    cached = set(cached_frame_ids)
    indices = tuple(index for index, frame in enumerate(scan_frames) if int(frame.frame_id) in cached)
    if len(indices) < 2:
        raise ValueError(
            "scan_progress_interval contains fewer than two certified online ORB real frames"
        )
    return indices


def _select_real_orb_tracking_indices(frames, *, target_fps: float) -> tuple[int, ...]:
    """Select chronological real frames for the video ORB chain.

    Fast video keeps every received frame for motion/risk analysis, but tracks
    only a true time-spaced subset.  No skipped frame is eligible to render.
    """

    if not np.isfinite(target_fps) or target_fps <= 0.0:
        raise ValueError("video fast ORB target FPS must be finite and positive")
    interval_us = 1_000_000.0 / target_fps
    selected = [0]
    previous = frames[0].timestamp_us
    if previous is None:
        raise ValueError("video ORB tracking source lacks timestamp")
    for index, frame in enumerate(frames[1:], start=1):
        if frame.timestamp_us is None:
            raise ValueError("video ORB tracking source lacks timestamp")
        if float(frame.timestamp_us - previous) >= interval_us:
            selected.append(index)
            previous = frame.timestamp_us
    if selected[-1] != len(frames) - 1:
        selected.append(len(frames) - 1)
    return tuple(selected)


def _tracking_fps_candidates(values: object) -> tuple[float, ...]:
    """Validate the ordered, direct-ORB FPS alternatives for v6 development."""

    if not isinstance(values, (tuple, list)) or not values:
        raise ValueError("tracking FPS candidates must be a non-empty list")
    candidates = tuple(float(value) for value in values)
    if not np.isfinite(candidates).all() or any(value <= 0.0 for value in candidates):
        raise ValueError("tracking FPS candidates must be finite and positive")
    if tuple(sorted(candidates)) != candidates or len(set(candidates)) != len(candidates):
        raise ValueError("tracking FPS candidates must be strictly increasing")
    return candidates


def run_direct_orb_tracking_gate(
    input_path: Path,
    output: Path,
    *,
    fps_candidates: tuple[float, ...] = (8.0, 12.0, 16.0),
    config_path: Path | None = None,
    fast_orbslam3_config: dict[str, object] | None = None,
) -> dict[str, object]:
    """Measure T0/T1/T2 using only real frames and direct ORB poses.

    This v6 Phase-1 command intentionally stops before source selection or
    rendering.  A failed rate remains observable in the report; it never
    receives an interpolated pose, DIS replacement, or an Open3D substitute.
    """

    candidates = _tracking_fps_candidates(fps_candidates)
    root = input_path.expanduser().resolve()
    root = root if root.is_dir() else root.parent
    destination = output.expanduser().resolve()
    config = load_config(config_path)
    stitch = dict(config.get("stitch", {}))
    video_settings = dict(stitch.get("video_panorama", {}))
    validation_workers = int(video_settings.get("fast_session_validation_workers", 4))
    scan_workers = int(video_settings.get("fast_scan_analysis_workers", 4))
    if validation_workers < 1 or scan_workers < 1:
        raise ValueError("tracking gate worker counts must be positive")
    video = load_video_session(
        root,
        validate_frame_files=True,
        validation_workers=validation_workers,
    )
    qualities, motions, segment = analyse_video_scan(
        video.rgbd.frames,
        analysis_width=int(stitch.get("analysis_width", 320)),
        motion_backend=str(video_settings.get("motion_backend", "dis")),
        workers=scan_workers,
    )
    first, last = int(segment["start_index"]), int(segment["end_index"])
    scan_frames = tuple(video.rgbd.frames[first : last + 1])
    if len(scan_frames) < 2:
        raise RuntimeError("Direct ORB tracking gate has fewer than two real scan frames")
    trajectory_base = dict(stitch.get("orbslam3_rgbd", {}))
    if fast_orbslam3_config is not None:
        trajectory_base.update(dict(fast_orbslam3_config))
    trajectory_base["minimum_tracked_fraction"] = 1.0

    results: list[dict[str, object]] = []
    selected_id: str | None = None
    for index, fps in enumerate(candidates):
        candidate_id = f"T{index}"
        tracking_indices = _select_real_orb_tracking_indices(scan_frames, target_fps=fps)
        tracking_frames = tuple(scan_frames[item] for item in tracking_indices)
        expected_ids = [int(frame.frame_id) for frame in tracking_frames]
        tracking_motions = compose_selected_motions(motions[first:last], tracking_indices)
        source_steps = np.asarray(
            [
                float(segment["scan_direction"]) * float(motion.dx)
                * (video.rgbd.calibration.width / float(stitch.get("analysis_width", 320)))
                for motion in tracking_motions
            ],
            dtype=np.float64,
        )
        if not np.isfinite(source_steps).all() or np.any(source_steps <= 0.0):
            raise RuntimeError("tracking candidate has non-positive measured scan progress")
        source_centres_x = tuple(
            float(value) for value in np.concatenate(([0.0], np.cumsum(source_steps)))
        )
        frontality = assess_video_source_frontality(
            tracking_frames, video.rgbd.calibration
        )
        owner_plan = plan_frontality_owner_spans(
            frontality, video.rgbd.calibration, source_centres_x
        )
        frontality_records = [record.as_dict() for record in frontality]
        frontality_pass = len(frontality_records) == len(expected_ids)
        started = time.perf_counter()
        row: dict[str, object] = {
            "tracking_candidate_id": candidate_id,
            "tracking_fps": fps,
            "requested_real_frame_count": len(expected_ids),
            "requested_frame_ids": expected_ids,
            "direct_orb_pose_count": 0,
            "direct_orb_pose_coverage": 0.0,
            "full_direct_orb_chain_available": False,
            "frontality_coverage_pass": frontality_pass,
            "frontality_coverage_status": (
                "calibrated_per_source_spans_available"
                if frontality_pass
                else "frontality_span_coverage_incomplete"
            ),
            "source_frontality": frontality_records,
            "frontality_owner_plan": owner_plan.as_dict(),
        }
        try:
            with tempfile.TemporaryDirectory(prefix=f"g305-v6-{candidate_id.lower()}-orb-") as work:
                trajectory = run_orbslam3_rgbd(
                    tracking_frames,
                    video.rgbd.calibration,
                    work,
                    config=trajectory_base,
                )
            tracked_ids = [int(frame_id) for frame_id in trajectory.tracked_frame_ids]
            coverage = len(tracked_ids) / len(expected_ids)
            complete = tracked_ids == expected_ids and len(trajectory.poses_by_frame_id) == len(expected_ids)
            row.update(
                {
                    "direct_orb_pose_count": len(tracked_ids),
                    "direct_orb_pose_coverage": coverage,
                    "full_direct_orb_chain_available": complete,
                    "orb_attempt_audit": [dict(item) for item in trajectory.attempt_audit],
                    "elapsed_seconds": time.perf_counter() - started,
                    "status": "direct_chain_complete" if complete else "direct_chain_incomplete",
                }
            )
        except Exception as exc:
            row.update(
                {
                    "elapsed_seconds": time.perf_counter() - started,
                    "status": "direct_chain_failed",
                    "failure_reason": f"{type(exc).__name__}: {exc}",
                }
            )
        results.append(row)
        if selected_id is None and row["full_direct_orb_chain_available"] is True:
            selected_id = candidate_id

    result: dict[str, object] = {
        "schema": "gemini305-video-direct-orb-tracking-gate/v1",
        "session_root": str(root),
        "scan_segment": dict(segment),
        "scan_real_frame_count": len(scan_frames),
        "tracking_candidates": results,
        "selected_tracking_candidate_id": selected_id,
        "full_direct_orb_chain_available": selected_id is not None,
        "selection_status": (
            "direct_orb_and_calibrated_frontality_ready"
            if selected_id is not None
            else "no_direct_orb_survivor"
        ),
        "interpolated_poses": False,
        "open3d_pose_replacement": False,
        "dis_pose_replacement": False,
    }
    atomic_write_json(destination / "tracking_gate.json", result)
    return result


def run_legacy(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.expanduser().resolve()
    input_path = args.input.expanduser().resolve()
    invalidate_video_delivery(output)
    profiler = VideoPerformanceProfiler()
    two_d_published = False
    try:
        with profiler.stage("config_and_session"):
            config = load_config(args.config)
            stitch = dict(config.get("stitch", {}))
            video_settings = dict(stitch.get("video_panorama", {}))
            legacy_renderer = str(video_settings.get("fast_renderer", "visual_seam"))
            if legacy_renderer not in {
                "audited_visual",
                "visual_seam",
                "hard_owner_diagnostic",
                "v6_graphcut_candidate",
                "v61_tail_guarded_candidate",
            }:
                raise ValueError(
                    "video_panorama.fast_renderer must be audited_visual, "
                    "visual_seam, hard_owner_diagnostic, v6_graphcut_candidate, or v61_tail_guarded_candidate"
                )
            candidate_c1_constrained_owner = bool(
                video_settings.get("candidate_c1_constrained_owner", False)
            )
            candidate_c1_config = video_settings.get("candidate_c1_config")
            if candidate_c1_config is not None and not isinstance(candidate_c1_config, dict):
                raise ValueError("candidate_c1_config must be a mapping")
            candidate_mesh_evidence = video_settings.get("candidate_mesh_evidence")
            if candidate_mesh_evidence is not None and not isinstance(candidate_mesh_evidence, dict):
                raise ValueError("candidate_mesh_evidence must be a mapping")
            candidate_c9_long_line_minimum_length_px = video_settings.get("candidate_c9_long_line_minimum_length_px")
            if candidate_c9_long_line_minimum_length_px is not None and not isinstance(candidate_c9_long_line_minimum_length_px, int):
                raise ValueError("candidate_c9_long_line_minimum_length_px must be an integer")
            candidate_object_owner_lock = bool(video_settings.get("candidate_object_owner_lock", False))
            candidate_safe_multiband = bool(video_settings.get("candidate_safe_multiband", False))
            candidate_global_photometric = bool(video_settings.get("candidate_global_photometric", False))
            candidate_multilabel_owner = bool(video_settings.get("candidate_multilabel_owner", False))
            candidate_robust_photometric_bundle = bool(video_settings.get("candidate_robust_photometric_bundle", False))
            candidate_depth_conditioned_layout = bool(video_settings.get("candidate_depth_conditioned_layout", False))
            candidate_joint_owner_final_grid = bool(video_settings.get("candidate_joint_owner_final_grid", False))
            candidate_object_first_compositor = bool(video_settings.get("candidate_object_first_compositor", False))
            candidate_object_first_protection_margin_pixels = video_settings.get("candidate_object_first_protection_margin_pixels")
            candidate_dense_real_frame_layout = video_settings.get("candidate_dense_real_frame_layout")
            if candidate_dense_real_frame_layout is not None and not isinstance(candidate_dense_real_frame_layout, dict):
                raise ValueError("candidate_dense_real_frame_layout must be a mapping")
            candidate_d2_monotonic_depth_layer_warp = video_settings.get(
                "candidate_d2_monotonic_depth_layer_warp"
            )
            if candidate_d2_monotonic_depth_layer_warp is not None and not isinstance(
                candidate_d2_monotonic_depth_layer_warp, dict
            ):
                raise ValueError("candidate_d2_monotonic_depth_layer_warp must be a mapping")
            candidate_d3_object_first_dense_source_compositor = video_settings.get(
                "candidate_d3_object_first_dense_source_compositor"
            )
            if candidate_d3_object_first_dense_source_compositor is not None and not isinstance(
                candidate_d3_object_first_dense_source_compositor, dict
            ):
                raise ValueError("candidate_d3_object_first_dense_source_compositor must be a mapping")
            if candidate_c1_constrained_owner and legacy_renderer != "hard_owner_diagnostic":
                raise ValueError(
                    "candidate C1 constrained owner requires the isolated hard-owner renderer"
                )
            maximum_post_seconds = (
                args.maximum_post_seconds
                if args.maximum_post_seconds is not None
                else video_settings.get("maximum_post_seconds")
            )
            if maximum_post_seconds is not None:
                maximum_post_seconds = float(maximum_post_seconds)
                if not np.isfinite(maximum_post_seconds) or maximum_post_seconds <= 0.0:
                    raise ValueError("maximum_post_seconds must be finite and positive")
            publish_auxiliary_exports = bool(video_settings.get("fast_publish_auxiliary_exports", False))
            fast_enable_geometry_assist = bool(
                video_settings.get("fast_enable_geometry_assist", False)
            )
            fast_odometry_prepare_workers = int(
                video_settings.get("fast_odometry_prepare_workers", 4)
            )
            if fast_odometry_prepare_workers < 1:
                raise ValueError("fast_odometry_prepare_workers must be positive")
            fast_session_validation_workers = int(
                video_settings.get("fast_session_validation_workers", 1)
            )
            fast_scan_analysis_workers = int(
                video_settings.get("fast_scan_analysis_workers", 1)
            )
            if fast_session_validation_workers < 1 or fast_scan_analysis_workers < 1:
                raise ValueError("video fast worker counts must be positive")
            requested_online_state = (
                args.online_state.expanduser().resolve()
                if args.online_state is not None
                else None
            )
            if requested_online_state is None:
                session_candidate_root = (
                    input_path if input_path.is_dir() else input_path.parent
                )
                capture_state = session_candidate_root / "online_video_state.json"
                if capture_state.is_file():
                    requested_online_state = capture_state
            video = load_video_session(
                input_path,
                validate_frame_files=requested_online_state is None,
                validation_workers=(
                    fast_session_validation_workers
                ),
            )
            online_state = (
                load_online_state(
                    requested_online_state,
                    root=video.rgbd.root,
                    frames=video.rgbd.frames,
                )
                if requested_online_state is not None
                else None
            )
            if online_state is not None and not online_state.certifies_strict_frame_files:
                # An offline-prepared state may save scan analysis, but it
                # never replaces the strict per-file RGB/depth decoder audit.
                video = load_video_session(
                    input_path,
                    validate_frame_files=True,
                    validation_workers=(
                        fast_session_validation_workers
                    ),
                )
            input_hashes = {
                "manifest": _sha256(video.rgbd.root / "manifest.json"),
                "calibration": _sha256(video.rgbd.root / "calibration.json"),
                "frames_csv": _sha256(video.rgbd.root / "frames.csv"),
            }
        with profiler.stage("scan_analysis"):
            if online_state is None:
                qualities, motions, segment = analyse_video_scan(
                    video.rgbd.frames,
                    analysis_width=int(stitch.get("analysis_width", 320)),
                    motion_backend=str(video_settings.get("motion_backend", "dis")),
                workers=fast_scan_analysis_workers,
                )
            else:
                qualities = list(online_state.qualities)
                motions = list(online_state.motions)
                segment = dict(online_state.segment)
        first, last = int(segment["start_index"]), int(segment["end_index"])
        scan_frames = tuple(video.rgbd.frames[first : last + 1])
        scan_qualities = qualities[first : last + 1]
        scan_motions = motions[first:last]
        source_progress_evidence: dict[str, object] | None = None
        if getattr(args, "measurement_annotations", None) is not None:
            # This uses the complete selected scan before a locked experiment
            # interval reduces it.  It is measurement provenance only: it
            # cannot add a render source, infer a pose, or influence the
            # renderer's source/keyframe selection.
            from .video_split import build_source_progress_evidence

            source_progress_evidence = build_source_progress_evidence(
                scan_frames, scan_motions
            )
        scan_frames, scan_qualities, scan_motions, progress_restriction = (
            _restrict_scan_to_progress_interval(
                scan_frames,
                scan_qualities,
                scan_motions,
                getattr(args, "scan_progress_interval", None),
            )
        )
        full_scan_ids = [frame.frame_id for frame in scan_frames]
        # An online cache was certified at capture time against a particular
        # set of real 8 FPS frames.  A candidate-only progress restriction is
        # allowed to select a contained portion of that chain, but it must not
        # re-run the time-spacing phase on the shortened slice: doing so picks
        # different real files and rightly fails the cache source-hash audit.
        # Caller-provided verified experiment caches retain their historical
        # behaviour; only capture-provenance online reuse has this exception.
        cache = _resolve_cached_trajectory_path(args, video.rgbd.root)
        if cache is not None and cache[1]:
            cached_frame_ids = _online_cached_tracking_frame_ids(cache[0])
            tracking_indices = _contained_online_tracking_indices(scan_frames, cached_frame_ids)
        else:
            tracking_indices = _select_real_orb_tracking_indices(
                scan_frames,
                target_fps=float(video_settings.get("fast_orb_target_fps", 20.0)),
            )
        tracking_frames = tuple(scan_frames[index] for index in tracking_indices)
        tracking_motions = compose_selected_motions(
            scan_motions,
            tracking_indices,
            # Capture-certified online anchors may be wholly contained in a
            # development/validation slice without coinciding with its
            # endpoints.  They are trajectory evidence only, never a render
            # source selection, so composing their measured interior spans is
            # valid and must not re-phase the capture-time chain.
            require_scan_endpoints=not (cache is not None and cache[1]),
        )
        tracking_qualities = [scan_qualities[index] for index in tracking_indices]
        tracking_ids = [frame.frame_id for frame in tracking_frames]
        with profiler.stage("orbslam3_trajectory"):
            if cache is not None:
                cache_path, capture_provenance = cache
                poses_by_id, tracked_ids, attempts = _cached_trajectory(
                    cache_path,
                    frames=tracking_frames,
                    input_sha256=input_hashes,
                    require_capture_provenance=capture_provenance,
                )
                trajectory_reused = True
            else:
                trajectory_config = dict(stitch.get("orbslam3_rgbd", {}))
                fast_orb_config = video_settings.get("fast_orbslam3_rgbd")
                if fast_orb_config is not None:
                    if not isinstance(fast_orb_config, dict):
                        raise ValueError("legacy baseline ORB settings must be a mapping")
                    trajectory_config.update(fast_orb_config)
                # The entire real scan remains the sole ORB-SLAM3 chain.  A
                # later render subset is provenance-only, never an interpolated pose.
                trajectory_config["minimum_tracked_fraction"] = 1.0
                with tempfile.TemporaryDirectory(prefix="g305-video-orbslam3-") as work:
                    trajectory = run_orbslam3_rgbd(
                        tracking_frames, video.rgbd.calibration, work, config=trajectory_config
                    )
                if list(trajectory.tracked_frame_ids) != tracking_ids:
                    raise RuntimeError("ORB-SLAM3 did not track every real video scan frame")
                poses_by_id = dict(trajectory.poses_by_frame_id)
                tracked_ids = tuple(trajectory.tracked_frame_ids)
                attempts = list(trajectory.attempt_audit)
                trajectory_reused = False
        with profiler.stage("render_keyframe_selection"):
            plan = select_render_keyframes(
                tracking_frames,
                tracking_motions,
                full_resolution_scale=video.rgbd.calibration.width
                / float(stitch.get("analysis_width", 320)),
                frame_width=video.rgbd.calibration.width,
                qualities=tracking_qualities,
                config=MotionResamplingConfig.from_mapping(
                    video_settings.get("motion_resampling")
                ),
            )
            # V6's measured 8 px source-spacing contract is the first N_req
            # replan.  A selected gap exceeding it cannot honestly be called
            # one safe geometry patch, so insert at most four *already direct
            # ORB-tracked* midpoint sources before Open3D edges and the final
            # one-pass RGB sampling.  This is source selection only: no frame
            # or pose is synthesized and every resulting edge is audited.
            if legacy_renderer in {"v6_graphcut_candidate", "v61_tail_guarded_candidate"}:
                plan = insert_v6_real_rescue_sources(
                    tracking_frames,
                    tracking_motions,
                    plan,
                    full_resolution_scale=video.rgbd.calibration.width
                    / float(stitch.get("analysis_width", 320)),
                    maximum_step_pixels=8.0,
                    maximum_rescues=4,
                )
            sources, source_indices = plan.frames, plan.source_indices
            render_plan = plan.as_dict()
            selected_motions = compose_selected_motions(tracking_motions, source_indices)
            frontality_scale = video.rgbd.calibration.width / float(
                stitch.get("analysis_width", 320)
            )
            source_steps = np.asarray(
                [
                    float(plan.scan_direction) * float(motion.dx) * frontality_scale
                    for motion in selected_motions
                ],
                dtype=np.float64,
            )
            if not np.isfinite(source_steps).all() or np.any(source_steps <= 0.0):
                raise RuntimeError(
                    "Dynamic frontality source selection requires strictly positive "
                    "measured scan progress between adjacent real sources"
                )
            source_centres_x = tuple(
                float(value)
                for value in np.concatenate(([0.0], np.cumsum(source_steps)))
            )
            frontality_records = assess_video_source_frontality(
                sources, video.rgbd.calibration
            )
            frontality_owner_plan = plan_frontality_owner_spans(
                frontality_records,
                video.rgbd.calibration,
                source_centres_x,
            )
            render_plan["frontality"] = {
                "source_records": [record.as_dict() for record in frontality_records],
                "owner_plan": frontality_owner_plan.as_dict(),
                "source_centre_evidence": "measured_adjacent_rgb_motion_only",
            }
            poses = [poses_by_id[frame.frame_id] for frame in sources]
            dense_layout = None
            if candidate_dense_real_frame_layout is not None:
                from .video_dense_pose_prior import DensePosePriorConfig, ORBPoseAnchor
                from .video_dense_real_frame_layout import (
                    DenseRealFrameLayoutConfig, build_dense_real_frame_layout,
                )

                anchors = tuple(
                    ORBPoseAnchor(int(frame.frame_id), int(frame.timestamp_us), poses_by_id[frame.frame_id])
                    for frame in tracking_frames
                )
                dense_layout = build_dense_real_frame_layout(
                    scan_frames, anchors, video.rgbd.calibration,
                    pose_config=DensePosePriorConfig(),
                    layout_config=DenseRealFrameLayoutConfig(
                        real_source_fps=float(candidate_dense_real_frame_layout.get("real_source_fps", 24.0)),
                        minimum_dense_prior_coverage=float(candidate_dense_real_frame_layout.get("minimum_dense_prior_coverage", 0.95)),
                        minimum_compact_object_support_coverage=float(candidate_dense_real_frame_layout.get("minimum_compact_object_support_coverage", 0.98)),
                    ),
                )
                sources = dense_layout.frames
                poses = list(dense_layout.camera_to_world)
                source_indices = tuple(
                    next(index for index, frame in enumerate(scan_frames) if frame.frame_id == source.frame_id)
                    for source in sources
                )
                selected_motions = compose_selected_motions(
                    scan_motions, source_indices, require_scan_endpoints=False
                )
                render_plan = dense_layout.as_dict()
        with profiler.stage("open3d_render_edge_audit"):
            odometry_mapping = stitch.get("rgbd_odometry")
            # The frozen baseline does not skip or approximate any required
            # local edge; it uses its pinned working resolution and solver.
            fast_odometry = video_settings.get("fast_rgbd_odometry")
            if fast_odometry is not None:
                if not isinstance(fast_odometry, dict):
                    raise ValueError("legacy baseline odometry settings must be a mapping")
                odometry_mapping = {**dict(odometry_mapping or {}), **fast_odometry}
            odometry_config = RGBDOdometryConfig.from_mapping(odometry_mapping)
            def _prepare_source(source: object):
                return prepare_rgbd_odometry_frame(
                    source,
                    video.rgbd.calibration,
                    config=odometry_config,
                    fallback_id=source.frame_id,
                )

            # Disk-backed RGB/depth decode and calibrated preparation are
            # independent for each real source. Keep final edge estimation
            # serial on its single CUDA backend, but overlap this CPU/I/O work.
            workers = min(fast_odometry_prepare_workers, len(sources))
            if workers > 1:
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    prepared_records = list(executor.map(_prepare_source, sources))
            else:
                prepared_records = [_prepare_source(source) for source in sources]
            prepared_sources = []
            working_intrinsics = None
            for prepared, prepared_intrinsics in prepared_records:
                if working_intrinsics is None:
                    working_intrinsics = prepared_intrinsics
                elif prepared_intrinsics != working_intrinsics:
                    raise RuntimeError("Video Open3D sources have inconsistent working intrinsics")
                prepared_sources.append(prepared)
            if working_intrinsics is None:
                raise RuntimeError("Video renderer selected no Open3D sources")
            edge_backend = create_open3d_rgbd_odometry_backend()
            edges = [
                estimate_prepared_pair_rgbd_odometry(
                    left,
                    right,
                    working_intrinsics,
                    config=odometry_config,
                    backend=edge_backend,
                    reference_node_id=source_left.frame_id,
                    source_node_id=source_right.frame_id,
                )
                for left, right, source_left, source_right in zip(
                    prepared_sources[:-1],
                    prepared_sources[1:],
                    sources[:-1],
                    sources[1:],
                    strict=True,
                )
            ]
            if len(edges) != len(sources) - 1:
                raise RuntimeError("Video Open3D audit did not cover every selected source edge")
        with profiler.stage("calibrated_render_and_exports"):
            push_config = dict(stitch.get("calibrated_rgb_pushbroom", {}))
            push_config["max_pose_count"] = None
            if not fast_enable_geometry_assist:
                geometry_settings = dict(push_config.get("geometry_assisted_seam", {}))
                geometry_settings["enabled"] = False
                push_config["geometry_assisted_seam"] = geometry_settings
            # Fast primary delivery is permitted to publish before optional
            # strip archives.  Audit (or explicit config opt-in) retains the
            # complete staged archive contract.
            central_strip_output_dir = (
                output / ".central_strips.pending" if publish_auxiliary_exports else None
            )
            central_strip_owner_only_output_dir = (
                output / ".central_strips_owner_only.pending"
                if publish_auxiliary_exports
                else None
            )
            v2_cuda_mode = getattr(args, "v2_cuda_renderer_mode", None)
            if v2_cuda_mode is None and bool(getattr(args, "v2_cuda_strict_owner", False)):
                v2_cuda_mode = "strict_owner"
            if v2_cuda_mode is not None:
                unsupported_v2_components = any(
                    (
                        candidate_mesh_evidence is not None
                        and v2_cuda_mode not in {
                            "c2_dis_residual_mesh",
                            "c3_raft_residual_mesh",
                            "c4_raft_rgbd_layered_mesh",
                            "c5_object_lock",
                            "c6_safe_multiband",
                            "c7_photometric_graph",
                            "c8_multilabel_window",
                            "c13_robust_photometric_bundle",
                            "c10_depth_conditioned_layout",
                            "c11_object_first_foreground_compositor",
                            "c12_joint_owner_final_grid",
                            "c9_positive_jacobian_line_mesh",
                        },
                        candidate_object_owner_lock and v2_cuda_mode not in {"c5_object_lock", "c6_safe_multiband", "c7_photometric_graph", "c8_multilabel_window", "c13_robust_photometric_bundle"},
                        candidate_safe_multiband and v2_cuda_mode not in {"c6_safe_multiband", "c7_photometric_graph", "c8_multilabel_window", "c13_robust_photometric_bundle"},
                        candidate_global_photometric and v2_cuda_mode not in {"c7_photometric_graph", "c8_multilabel_window", "c13_robust_photometric_bundle"},
                        candidate_multilabel_owner and v2_cuda_mode not in {"c8_multilabel_window", "c13_robust_photometric_bundle"},
                        candidate_robust_photometric_bundle and v2_cuda_mode != "c13_robust_photometric_bundle",
                        candidate_depth_conditioned_layout and v2_cuda_mode not in {"c10_depth_conditioned_layout", "c11_object_first_foreground_compositor", "c12_joint_owner_final_grid"},
                        candidate_joint_owner_final_grid and v2_cuda_mode != "c12_joint_owner_final_grid",
                        candidate_object_first_compositor and v2_cuda_mode != "c11_object_first_foreground_compositor",
                    )
                )
                if unsupported_v2_components or (
                    v2_cuda_mode == "strict_owner" and candidate_c1_constrained_owner
                ) or (
                    v2_cuda_mode == "c1_constrained_owner" and not candidate_c1_constrained_owner
                ) or (
                    v2_cuda_mode == "c2_dis_residual_mesh"
                    and (
                        not candidate_c1_constrained_owner
                        or not isinstance(candidate_mesh_evidence, dict)
                        or candidate_mesh_evidence.get("flow_backend") != "dis"
                        or bool(candidate_mesh_evidence.get("require_depth_safety", True))
                    )
                ) or (
                    v2_cuda_mode == "c3_raft_residual_mesh"
                    and (
                        not candidate_c1_constrained_owner
                        or not isinstance(candidate_mesh_evidence, dict)
                        or candidate_mesh_evidence.get("flow_backend") != "raft"
                        or bool(candidate_mesh_evidence.get("require_depth_safety", True))
                    )
                ) or (
                    v2_cuda_mode == "c4_raft_rgbd_layered_mesh"
                    and (
                        not candidate_c1_constrained_owner
                        or not isinstance(candidate_mesh_evidence, dict)
                        or candidate_mesh_evidence.get("flow_backend") != "raft"
                        or not bool(candidate_mesh_evidence.get("require_depth_safety", False))
                    )
                ) or (
                    v2_cuda_mode == "c10_depth_conditioned_layout"
                    and (
                        not candidate_c1_constrained_owner
                        or not isinstance(candidate_mesh_evidence, dict)
                        or candidate_mesh_evidence.get("flow_backend") != "raft"
                        or not bool(candidate_mesh_evidence.get("require_depth_safety", False))
                        or not candidate_depth_conditioned_layout
                    )
                ) or (
                    v2_cuda_mode == "c11_object_first_foreground_compositor"
                    and (
                        not candidate_c1_constrained_owner
                        or not isinstance(candidate_mesh_evidence, dict)
                        or candidate_mesh_evidence.get("flow_backend") != "raft"
                        or not bool(candidate_mesh_evidence.get("require_depth_safety", False))
                        or not candidate_object_first_compositor
                        or not isinstance(candidate_object_first_protection_margin_pixels, int)
                        or not 8 <= candidate_object_first_protection_margin_pixels <= 12
                    )
                ) or (
                    v2_cuda_mode == "c12_joint_owner_final_grid"
                    and (
                        not candidate_c1_constrained_owner
                        or not isinstance(candidate_mesh_evidence, dict)
                        or candidate_mesh_evidence.get("flow_backend") != "raft"
                        or not bool(candidate_mesh_evidence.get("require_depth_safety", False))
                        or not candidate_joint_owner_final_grid
                    )
                ) or (
                    v2_cuda_mode == "c9_positive_jacobian_line_mesh"
                    and (
                        not candidate_c1_constrained_owner
                        or not isinstance(candidate_mesh_evidence, dict)
                        or candidate_mesh_evidence.get("flow_backend") != "raft"
                        or not bool(candidate_mesh_evidence.get("require_depth_safety", False))
                        or not isinstance(candidate_c9_long_line_minimum_length_px, int)
                    )
                ) or (
                    v2_cuda_mode == "c5_object_lock"
                    and (
                        not candidate_c1_constrained_owner
                        or not isinstance(candidate_mesh_evidence, dict)
                        or candidate_mesh_evidence.get("flow_backend") != "raft"
                        or not bool(candidate_mesh_evidence.get("require_depth_safety", False))
                        or not candidate_object_owner_lock
                    )
                ) or (
                    v2_cuda_mode == "c6_safe_multiband"
                    and (
                        not candidate_c1_constrained_owner
                        or not isinstance(candidate_mesh_evidence, dict)
                        or candidate_mesh_evidence.get("flow_backend") != "raft"
                        or not bool(candidate_mesh_evidence.get("require_depth_safety", False))
                        or not candidate_object_owner_lock
                        or not candidate_safe_multiband
                    )
                ) or (
                    v2_cuda_mode == "c7_photometric_graph"
                    and (
                        not candidate_c1_constrained_owner
                        or not isinstance(candidate_mesh_evidence, dict)
                        or candidate_mesh_evidence.get("flow_backend") != "raft"
                        or not bool(candidate_mesh_evidence.get("require_depth_safety", False))
                        or not candidate_object_owner_lock
                        or not candidate_safe_multiband
                        or not candidate_global_photometric
                    )
                ) or (
                    v2_cuda_mode == "c8_multilabel_window"
                    and (
                        not candidate_c1_constrained_owner
                        or not isinstance(candidate_mesh_evidence, dict)
                        or candidate_mesh_evidence.get("flow_backend") != "raft"
                        or not bool(candidate_mesh_evidence.get("require_depth_safety", False))
                        or not candidate_object_owner_lock
                        or not candidate_safe_multiband
                        or not candidate_global_photometric
                        or not candidate_multilabel_owner
                    )
                ) or (
                    v2_cuda_mode == "c13_robust_photometric_bundle"
                    and (
                        not candidate_c1_constrained_owner
                        or not isinstance(candidate_mesh_evidence, dict)
                        or candidate_mesh_evidence.get("flow_backend") != "raft"
                        or not bool(candidate_mesh_evidence.get("require_depth_safety", False))
                        or not candidate_object_owner_lock
                        or not candidate_safe_multiband
                        or not candidate_global_photometric
                        or not candidate_multilabel_owner
                        or not candidate_robust_photometric_bundle
                    )
                ):
                    raise RuntimeError(
                        "v2 CUDA route cannot claim components that are not wired into its selected data plane"
                    )
                if v2_cuda_mode == "strict_owner":
                    from .video_v2_route import render_cuda_strict_owner_v2

                    push = render_cuda_strict_owner_v2(
                        sources=sources,
                        camera_to_world=poses,
                        calibration=video.rgbd.calibration,
                        pushbroom_config=push_config,
                        selected_motions=selected_motions,
                        motion_pixels_to_full_resolution=video.rgbd.calibration.width
                        / float(stitch.get("analysis_width", 320)),
                        cuda_device=int(video_settings.get("cuda_device", 0)),
                    )
                elif v2_cuda_mode == "c1_constrained_owner":
                    from .video_v2_route import render_cuda_c1_constrained_owner_v2

                    push = render_cuda_c1_constrained_owner_v2(
                        sources=sources,
                        camera_to_world=poses,
                        calibration=video.rgbd.calibration,
                        pushbroom_config=push_config,
                        selected_motions=selected_motions,
                        motion_pixels_to_full_resolution=video.rgbd.calibration.width
                        / float(stitch.get("analysis_width", 320)),
                        c1_config=candidate_c1_config,
                        annotations=getattr(args, "measurement_annotations", None),
                        cuda_device=int(video_settings.get("cuda_device", 0)),
                    )
                elif v2_cuda_mode == "c2_dis_residual_mesh":
                    from .video_v2_route import render_cuda_c2_dis_residual_mesh_v2

                    push = render_cuda_c2_dis_residual_mesh_v2(
                        sources=sources,
                        camera_to_world=poses,
                        calibration=video.rgbd.calibration,
                        pushbroom_config=push_config,
                        selected_motions=selected_motions,
                        motion_pixels_to_full_resolution=video.rgbd.calibration.width
                        / float(stitch.get("analysis_width", 320)),
                        c1_config=candidate_c1_config,
                        cuda_device=int(video_settings.get("cuda_device", 0)),
                    )
                elif v2_cuda_mode == "c3_raft_residual_mesh":
                    from .video_v2_route import render_cuda_c3_raft_residual_mesh_v2

                    push = render_cuda_c3_raft_residual_mesh_v2(
                        sources=sources,
                        camera_to_world=poses,
                        calibration=video.rgbd.calibration,
                        pushbroom_config=push_config,
                        selected_motions=selected_motions,
                        motion_pixels_to_full_resolution=video.rgbd.calibration.width
                        / float(stitch.get("analysis_width", 320)),
                        c1_config=candidate_c1_config,
                        cuda_device=int(video_settings.get("cuda_device", 0)),
                    )
                elif v2_cuda_mode == "c4_raft_rgbd_layered_mesh":
                    from .video_v2_route import render_cuda_c4_raft_rgbd_layered_mesh_v2

                    push = render_cuda_c4_raft_rgbd_layered_mesh_v2(
                        sources=sources,
                        camera_to_world=poses,
                        calibration=video.rgbd.calibration,
                        pushbroom_config=push_config,
                        selected_motions=selected_motions,
                        motion_pixels_to_full_resolution=video.rgbd.calibration.width
                        / float(stitch.get("analysis_width", 320)),
                        c1_config=candidate_c1_config,
                        cuda_device=int(video_settings.get("cuda_device", 0)),
                    )
                elif v2_cuda_mode == "c9_positive_jacobian_line_mesh":
                    from .video_v2_route import render_cuda_c9_positive_jacobian_line_mesh_v2

                    push = render_cuda_c9_positive_jacobian_line_mesh_v2(
                        sources=sources, camera_to_world=poses, calibration=video.rgbd.calibration,
                        pushbroom_config=push_config, selected_motions=selected_motions,
                        motion_pixels_to_full_resolution=video.rgbd.calibration.width / float(stitch.get("analysis_width", 320)),
                        c1_config=candidate_c1_config, cuda_device=int(video_settings.get("cuda_device", 0)),
                        long_line_minimum_length_px=candidate_c9_long_line_minimum_length_px,
                    )
                elif v2_cuda_mode == "c10_depth_conditioned_layout":
                    from .video_v2_route import render_cuda_c10_depth_conditioned_layout_v2

                    push = render_cuda_c10_depth_conditioned_layout_v2(
                        sources=sources, camera_to_world=poses, calibration=video.rgbd.calibration,
                        pushbroom_config=push_config, selected_motions=selected_motions,
                        motion_pixels_to_full_resolution=video.rgbd.calibration.width / float(stitch.get("analysis_width", 320)),
                        c1_config=candidate_c1_config, cuda_device=int(video_settings.get("cuda_device", 0)),
                    )
                elif v2_cuda_mode == "c11_object_first_foreground_compositor":
                    from .video_v2_route import render_cuda_c11_object_first_foreground_compositor_v2

                    push = render_cuda_c11_object_first_foreground_compositor_v2(
                        sources=sources, camera_to_world=poses, calibration=video.rgbd.calibration,
                        pushbroom_config=push_config, selected_motions=selected_motions,
                        motion_pixels_to_full_resolution=video.rgbd.calibration.width / float(stitch.get("analysis_width", 320)),
                        c1_config=candidate_c1_config, cuda_device=int(video_settings.get("cuda_device", 0)),
                        protection_margin_pixels=candidate_object_first_protection_margin_pixels,
                    )
                elif v2_cuda_mode == "c12_joint_owner_final_grid":
                    from .video_v2_route import render_cuda_c12_joint_owner_final_grid_v2

                    push = render_cuda_c12_joint_owner_final_grid_v2(
                        sources=sources, camera_to_world=poses, calibration=video.rgbd.calibration,
                        pushbroom_config=push_config, selected_motions=selected_motions,
                        motion_pixels_to_full_resolution=video.rgbd.calibration.width / float(stitch.get("analysis_width", 320)),
                        c1_config=candidate_c1_config, cuda_device=int(video_settings.get("cuda_device", 0)),
                    )
                elif v2_cuda_mode == "c5_object_lock":
                    from .video_v2_route import render_cuda_c5_object_lock_v2

                    push = render_cuda_c5_object_lock_v2(
                        sources=sources,
                        camera_to_world=poses,
                        calibration=video.rgbd.calibration,
                        pushbroom_config=push_config,
                        selected_motions=selected_motions,
                        motion_pixels_to_full_resolution=video.rgbd.calibration.width
                        / float(stitch.get("analysis_width", 320)),
                        c1_config=candidate_c1_config,
                        cuda_device=int(video_settings.get("cuda_device", 0)),
                    )
                elif v2_cuda_mode == "c6_safe_multiband":
                    from .video_v2_route import render_cuda_c6_safe_multiband_v2

                    push = render_cuda_c6_safe_multiband_v2(
                        sources=sources,
                        camera_to_world=poses,
                        calibration=video.rgbd.calibration,
                        pushbroom_config=push_config,
                        selected_motions=selected_motions,
                        motion_pixels_to_full_resolution=video.rgbd.calibration.width
                        / float(stitch.get("analysis_width", 320)),
                        c1_config=candidate_c1_config,
                        cuda_device=int(video_settings.get("cuda_device", 0)),
                    )
                elif v2_cuda_mode == "c7_photometric_graph":
                    from .video_v2_route import render_cuda_c7_photometric_graph_v2

                    push = render_cuda_c7_photometric_graph_v2(
                        sources=sources,
                        camera_to_world=poses,
                        calibration=video.rgbd.calibration,
                        pushbroom_config=push_config,
                        selected_motions=selected_motions,
                        motion_pixels_to_full_resolution=video.rgbd.calibration.width
                        / float(stitch.get("analysis_width", 320)),
                        c1_config=candidate_c1_config,
                        cuda_device=int(video_settings.get("cuda_device", 0)),
                    )
                elif v2_cuda_mode == "c8_multilabel_window":
                    from .video_v2_route import render_cuda_c8_multilabel_window_v2

                    push = render_cuda_c8_multilabel_window_v2(
                        sources=sources,
                        camera_to_world=poses,
                        calibration=video.rgbd.calibration,
                        pushbroom_config=push_config,
                        selected_motions=selected_motions,
                        motion_pixels_to_full_resolution=video.rgbd.calibration.width
                        / float(stitch.get("analysis_width", 320)),
                        c1_config=candidate_c1_config,
                        cuda_device=int(video_settings.get("cuda_device", 0)),
                    )
                elif v2_cuda_mode == "c13_robust_photometric_bundle":
                    from .video_v2_route import render_cuda_c13_robust_photometric_bundle_v2

                    push = render_cuda_c13_robust_photometric_bundle_v2(
                        sources=sources, camera_to_world=poses, calibration=video.rgbd.calibration,
                        pushbroom_config=push_config, selected_motions=selected_motions,
                        motion_pixels_to_full_resolution=video.rgbd.calibration.width
                        / float(stitch.get("analysis_width", 320)),
                        c1_config=candidate_c1_config,
                        cuda_device=int(video_settings.get("cuda_device", 0)),
                    )
                else:
                    raise RuntimeError("unknown v2 CUDA renderer mode")
            else:
                if legacy_renderer in {"v6_graphcut_candidate", "v61_tail_guarded_candidate"}:
                    if legacy_renderer == "v61_tail_guarded_candidate":
                        from .video_v61_renderer import render_video_v61_candidate as candidate_renderer
                    else:
                        from .video_v6_pair_renderer import render_video_v6_candidate as candidate_renderer
                    push = candidate_renderer(
                        tuple(sources), tuple(poses), video.rgbd.calibration,
                        pushbroom_config=push_config, rgb_motions=selected_motions,
                        motion_pixels_to_full_resolution=video.rgbd.calibration.width
                        / float(stitch.get("analysis_width", 320)),
                        frontality_hard_spans={
                            int(record.frame_id): tuple(int(value) for value in record.general_hard_span)
                            for record in frontality_records
                        },
                    )
                    # A rejected true GraphCut is evidence to reroute a real
                    # source, not permission to synthesize a pose or loosen a
                    # topology gate.  One bounded replan can replace the
                    # speculative four rescue slots with midpoint frames in
                    # the actual failing direct-ORB gaps.  Its result must be
                    # completely re-audited before it may become final.
                    initial_expanded = push.metadata.get("expanded_real_owner_pair_frame_ids", ())
                    failed_pairs = tuple(
                        (int(pair[0]), int(pair[1]))
                        for pair in initial_expanded
                        if isinstance(pair, (tuple, list)) and len(pair) == 2
                    )
                    reroute_audit: dict[str, object] = {
                        "attempted": bool(failed_pairs),
                        "failed_pair_frame_ids": [list(pair) for pair in failed_pairs],
                        "accepted": False,
                    }
                    if failed_pairs:
                        rerouted_plan = reroute_v6_failed_pairs_with_real_sources(
                            tracking_frames, plan,
                            failed_pair_frame_ids=failed_pairs, maximum_rescues=4,
                        )
                        if rerouted_plan.source_indices != plan.source_indices:
                            rerouted_sources = rerouted_plan.frames
                            rerouted_indices = rerouted_plan.source_indices
                            rerouted_motions = compose_selected_motions(
                                tracking_motions, rerouted_indices,
                            )
                            rerouted_steps = np.asarray([
                                float(rerouted_plan.scan_direction) * float(motion.dx) * frontality_scale
                                for motion in rerouted_motions
                            ], dtype=np.float64)
                            if not np.isfinite(rerouted_steps).all() or np.any(rerouted_steps <= 0.0):
                                raise RuntimeError("v6 reroute selected a non-monotone real source chain")
                            rerouted_frontality = assess_video_source_frontality(
                                rerouted_sources, video.rgbd.calibration,
                            )
                            rerouted_owner_plan = plan_frontality_owner_spans(
                                rerouted_frontality, video.rgbd.calibration,
                                tuple(float(value) for value in np.concatenate(([0.0], np.cumsum(rerouted_steps)))),
                            )
                            rerouted_poses = [poses_by_id[frame.frame_id] for frame in rerouted_sources]
                            # The reroute has new final adjacency, so each of
                            # those real RGB-D edges receives a new actual
                            # Open3D audit before any rerendered pixels exist.
                            rerouted_records = [_prepare_source(source) for source in rerouted_sources]
                            rerouted_prepared: list[object] = []
                            rerouted_intrinsics = None
                            for prepared, prepared_intrinsics in rerouted_records:
                                if rerouted_intrinsics is None:
                                    rerouted_intrinsics = prepared_intrinsics
                                elif prepared_intrinsics != rerouted_intrinsics:
                                    raise RuntimeError("v6 reroute Open3D sources have inconsistent intrinsics")
                                rerouted_prepared.append(prepared)
                            if rerouted_intrinsics is None:
                                raise RuntimeError("v6 reroute selected no real sources")
                            rerouted_backend = create_open3d_rgbd_odometry_backend()
                            rerouted_edges = [
                                estimate_prepared_pair_rgbd_odometry(
                                    left, right, rerouted_intrinsics, config=odometry_config,
                                    backend=rerouted_backend,
                                    reference_node_id=source_left.frame_id,
                                    source_node_id=source_right.frame_id,
                                )
                                for left, right, source_left, source_right in zip(
                                    rerouted_prepared[:-1], rerouted_prepared[1:],
                                    rerouted_sources[:-1], rerouted_sources[1:], strict=True,
                                )
                            ]
                            if len(rerouted_edges) != len(rerouted_sources) - 1:
                                raise RuntimeError("v6 reroute Open3D audit did not cover every real source edge")
                            rerouted_push = candidate_renderer(
                                tuple(rerouted_sources), tuple(rerouted_poses), video.rgbd.calibration,
                                pushbroom_config=push_config, rgb_motions=rerouted_motions,
                                motion_pixels_to_full_resolution=video.rgbd.calibration.width
                                / float(stitch.get("analysis_width", 320)),
                                frontality_hard_spans={
                                    int(record.frame_id): tuple(int(value) for value in record.general_hard_span)
                                    for record in rerouted_frontality
                                },
                            )
                            def _v6_failure_score(result: object) -> tuple[int, int, int]:
                                metadata = getattr(result, "metadata", {})
                                metrics = metadata.get("quality_metrics", {}) if isinstance(metadata, dict) else {}
                                expanded = metadata.get("expanded_real_owner_pair_frame_ids", ()) if isinstance(metadata, dict) else ()
                                return (
                                    len(expanded),
                                    int(metrics.get("double_edge_count", 0)),
                                    int(metrics.get("ghost_count", 0)),
                                )
                            initial_score = _v6_failure_score(push)
                            rerouted_score = _v6_failure_score(rerouted_push)
                            reroute_audit.update({
                                "rerouted_source_frame_ids": [int(frame.frame_id) for frame in rerouted_sources],
                                "reroute_rescue_frame_ids": list(rerouted_plan.rescue_source_frame_ids),
                                "initial_failure_score": list(initial_score),
                                "rerouted_failure_score": list(rerouted_score),
                            })
                            # All three are v6 hard visual/structural gates.
                            # A reroute must Pareto-improve them: reducing an
                            # owner expansion cannot buy a worse double edge
                            # or ghosting result.
                            reroute_improves = (
                                all(candidate <= initial for candidate, initial in zip(
                                    rerouted_score, initial_score, strict=True,
                                ))
                                and rerouted_score != initial_score
                            )
                            if reroute_improves:
                                sources, source_indices, selected_motions, poses = (
                                    rerouted_sources, rerouted_indices, rerouted_motions, rerouted_poses,
                                )
                                plan, edges = rerouted_plan, rerouted_edges
                                frontality_records, frontality_owner_plan = rerouted_frontality, rerouted_owner_plan
                                render_plan = rerouted_plan.as_dict()
                                render_plan["frontality"] = {
                                    "source_records": [record.as_dict() for record in rerouted_frontality],
                                    "owner_plan": rerouted_owner_plan.as_dict(),
                                    "source_centre_evidence": "measured_adjacent_rgb_motion_only",
                                }
                                push = rerouted_push
                                reroute_audit["accepted"] = True
                            else:
                                reroute_audit["rejection_reason"] = "rerouted_failure_score_not_better"
                        else:
                            reroute_audit["rejection_reason"] = "no_new_real_midpoint_source_available"
                    push.metadata["v6_targeted_real_source_reroute"] = reroute_audit
                else:
                    push = render_calibrated_rgb_pushbroom(
                    list(sources),
                    poses,
                    video.rgbd.calibration,
                    config=push_config,
                    rgb_motions=selected_motions,
                    motion_pixels_to_full_resolution=video.rgbd.calibration.width
                    / float(stitch.get("analysis_width", 320)),
                    multiband_levels=int(dict(stitch.get("scan_seam", {})).get("multiband_levels", 3)),
                    quality_gate=False,
                    central_strip_output_dir=central_strip_output_dir,
                    central_strip_owner_only_output_dir=central_strip_owner_only_output_dir,
                    fast_hard_owner=(
                        legacy_renderer == "hard_owner_diagnostic"
                        or candidate_c1_constrained_owner
                    ),
                    fast_visual_owner=legacy_renderer == "visual_seam",
                    fast_visual_use_depth=bool(
                        int(render_plan.get("high_risk_edge_count", 0)) > 0
                    ),
                    candidate_c1_constrained_owner=candidate_c1_constrained_owner,
                    candidate_c1_config=candidate_c1_config,
                    candidate_mesh_evidence=candidate_mesh_evidence,
                    candidate_object_owner_lock=candidate_object_owner_lock,
                    candidate_safe_multiband=candidate_safe_multiband,
                    candidate_global_photometric=candidate_global_photometric,
                    candidate_multilabel_owner=candidate_multilabel_owner,
                    candidate_d2_monotonic_depth_layer_warp=candidate_d2_monotonic_depth_layer_warp,
                    candidate_d3_object_first_dense_source_compositor=candidate_d3_object_first_dense_source_compositor,
                    # D1's compact-support measurement is deliberately a
                    # separate full-support audit; it must not activate the
                    # older C1-only annotation renderer branch.
                    candidate_measurement_annotations=(
                        None if candidate_dense_real_frame_layout is not None
                        else getattr(args, "measurement_annotations", None)
                    ),
                    candidate_full_support_annotation_frame_ids=(
                        tuple(sorted({
                            int(entry["frame_id"])
                            for entry in getattr(args, "measurement_annotations", {}).get("objects", [])
                            if isinstance(entry, dict)
                            and entry.get("role") == "compact_foreground_single_owner"
                            and isinstance(entry.get("frame_id"), int)
                        }))
                        if candidate_dense_real_frame_layout is not None
                        and isinstance(getattr(args, "measurement_annotations", None), dict)
                        else ()
                    ),
                    )
            if v2_cuda_mode is not None and publish_auxiliary_exports:
                # The v2 archive is a strictly post-render, read-only export
                # of actual real-source calibrated tiles and final owner
                # pixels.  It intentionally does not call the historical CPU
                # renderer or feed RGB/owner data back into the CUDA result.
                from .video_v2_audit_export import stage_v2_cuda_audit_exports

                audit_context = getattr(push, "audit_export_context", None)
                if audit_context is None:
                    raise RuntimeError("v2 CUDA renderer did not provide its required audit export context")
                # Archive generation is read-only evidence.  It is nested in
                # the renderer stage for lifecycle simplicity, but explicitly
                # excluded from the primary delivery SLA.
                with profiler.audit_export():
                    source_export, owner_export = stage_v2_cuda_audit_exports(
                        audit_context,
                        panorama_bgr=push.panorama,
                        owner_frame_id=push.owner_frame_id,
                        central_strip_output_dir=central_strip_output_dir,
                        owner_only_output_dir=central_strip_owner_only_output_dir,
                    )
                push.metadata["central_strip_export"] = source_export
                push.metadata["central_strip_owner_only_export"] = owner_export
        d1_measurement_payload = None
        d1_measurement_masks = None
        d1_compact_support_masks = None
        if candidate_dense_real_frame_layout is not None:
            from .video_candidate_annotation_projection import build_candidate_annotation_projection
            from .video_dense_real_frame_layout import compact_support_masks_from_projection

            annotations = getattr(args, "measurement_annotations", None)
            sources_for_measurement = push.measurement_full_support_sources
            if not isinstance(annotations, dict) or not sources_for_measurement:
                raise RuntimeError("D1 requires M2 full-support compact annotation measurement context")
            d1_measurement_payload, d1_measurement_masks = build_candidate_annotation_projection(
                annotations, sources=sources_for_measurement,
                final_owner_frame_id=push.owner_frame_id,
                crop_xywh=(0, 0, int(push.panorama.shape[1]), int(push.panorama.shape[0])),
                horizontal_flip=bool(push.metadata.get("orientation", {}).get("horizontal_flip", False)),
            )
            d1_compact_support_masks = compact_support_masks_from_projection(
                annotations, d1_measurement_payload, d1_measurement_masks
            )
        with profiler.stage("quality_and_report"):
            capture_quality = assess_capture_quality(
                scan_qualities,
                [frame.color_exposure_raw for frame in scan_frames],
                exposure_unit_us=float(stitch.get("color_exposure_unit_us", 100.0)),
                maximum_exposure_us=float(stitch.get("maximum_motion_exposure_us", 1200.0)),
            )
            renderer_quality = dict(push.metadata.get("quality_metrics", {}))
            dense_owner_observability = None
            if candidate_dense_real_frame_layout is not None:
                from .video_dense_real_frame_layout import (
                    DenseRealFrameLayoutConfig, verify_dense_owner_observability,
                )
                if push.owner_frame_id is None:
                    raise RuntimeError("D1 renderer omitted its required hard-owner provenance map")
                dense_owner_observability = verify_dense_owner_observability(
                    push.owner_frame_id,
                    compact_support_masks=d1_compact_support_masks or {},
                    config=DenseRealFrameLayoutConfig(
                        real_source_fps=float(candidate_dense_real_frame_layout.get("real_source_fps", 24.0)),
                        minimum_dense_prior_coverage=float(candidate_dense_real_frame_layout.get("minimum_dense_prior_coverage", 0.95)),
                        minimum_compact_object_support_coverage=float(candidate_dense_real_frame_layout.get("minimum_compact_object_support_coverage", 0.98)),
                    ),
                )
                # D1's contract records two evidence-only gates and one
                # actual output component.  These records describe the
                # already-rendered real-frame layout; they do not feed back
                # into source selection, poses, or owner pixels.
                applied = int(np.count_nonzero(push.owner_frame_id >= 0))
                push.metadata["component_evidence"] = {
                    "orb_anchor_trajectory": {
                        "valid": True, "sample_count": len(tracking_frames), "audit_passed": True,
                    },
                    "dense_real_frame_pose_audit": {
                        "valid": bool(dense_layout.audit["dense_prior_coverage"] >= dense_layout.audit["dense_prior_coverage_gate"]),
                        "sample_count": len(dense_layout.frames), "audit_passed": True,
                    },
                }
                if candidate_d2_monotonic_depth_layer_warp is None:
                    push.metadata["component_execution"] = {
                        "d1_dense_real_frame_hard_owner": {
                            "required": True, "initialized": True, "applied_to_output": applied > 0,
                            "applied_output_pixel_count": applied,
                        }
                    }
                    push.metadata["executed_candidate_components"] = {
                        "d1_dense_real_frame_hard_owner": applied > 0,
                    }
                    push.metadata["candidate_run_state"] = "completed"
                    push.metadata["selection_eligible"] = True
                else:
                    d2_audit = push.metadata.get("d2_monotonic_depth_layer_warp")
                    if not isinstance(d2_audit, dict):
                        raise RuntimeError("D2 renderer omitted its required depth-layer warp audit")
                    d2_pixels = d2_audit.get("actual_output_warp_pixel_count")
                    if not isinstance(d2_pixels, int) or d2_pixels <= 0:
                        raise RuntimeError("D2 renderer did not apply a real output warp")
                    execution = {
                        "d2_monotonic_depth_layer_warp": {
                            "required": True, "initialized": True,
                            "applied_to_output": True, "applied_output_pixel_count": d2_pixels,
                        }
                    }
                    executed = {
                        "d2_monotonic_depth_layer_warp": True,
                    }
                    if candidate_d3_object_first_dense_source_compositor is not None:
                        d3_audit = push.metadata.get("d3_object_first_dense_source_compositor")
                        if not isinstance(d3_audit, dict):
                            raise RuntimeError("D3 renderer omitted its required real-source object audit")
                        d3_pixels = d3_audit.get("actual_output_object_pixel_count")
                        if not isinstance(d3_pixels, int) or d3_pixels <= 0 or d3_audit.get("object_flow_or_warp") is not False or d3_audit.get("object_multiband") is not False:
                            raise RuntimeError("D3 renderer did not apply an unblended real-source object owner")
                        execution["d3_object_first_dense_source_compositor"] = {
                            "required": True, "initialized": True,
                            "applied_to_output": True, "applied_output_pixel_count": d3_pixels,
                        }
                        executed["d3_object_first_dense_source_compositor"] = True
                    push.metadata["component_execution"] = execution
                    push.metadata["executed_candidate_components"] = executed
                    push.metadata["component_evidence"]["depth_visibility_evidence"] = {
                        "valid": True, "sample_count": int(d2_audit.get("observed_depth_pixel_count", 0)),
                        "audit_passed": True,
                    }
                    raft_line = d2_audit.get("training_raft_line_evidence")
                    if not isinstance(raft_line, dict):
                        raise RuntimeError("D2 renderer omitted required RAFT/automatic-line evidence")
                    push.metadata["component_evidence"]["raft_correspondence_evidence"] = {
                        "valid": True,
                        "sample_count": int(raft_line.get("valid_forward_backward_sample_count", 0)),
                        "audit_passed": float(raft_line.get("forward_backward_p95_pixels", float("inf"))) <= 1.5,
                    }
                    push.metadata["component_evidence"]["automatic_long_line_evidence"] = {
                        "valid": True, "sample_count": int(raft_line.get("automatic_long_line_count", 0)),
                        "audit_passed": int(raft_line.get("automatic_long_line_count", 0)) > 0,
                    }
                    push.metadata["candidate_run_state"] = "completed"
                    push.metadata["selection_eligible"] = True
            strict = (
                bool(capture_quality["quality_pass"])
                and video.capture_mode == "continuous_rgbd_video_fixed_exposure"
                and bool(renderer_quality.get("quality_pass", False))
            )
            performance = profiler.as_dict(maximum_post_seconds=maximum_post_seconds)
            budget_exceeded = performance["within_post_capture_budget"] is False
            grade = "A" if strict and not budget_exceeded else "C"
            algorithm = getattr(args, "algorithm_spec", None)
            if isinstance(algorithm, dict):
                algorithm_report = dict(algorithm)
            else:
                # This fallback identity is used only by the isolated legacy
                # runner.  Public and experiment CLIs always pass a registry
                # resolved immutable spec.
                algorithm_report = {
                    "role": "baseline",
                    "algorithm_id": "legacy_fast_b07b561",
                    "implementation_id": "legacy_visual_seam",
                    "config_sha256": "legacy-unlocked",
                    "source_commit": "legacy-unlocked",
                    "model_sha256": {},
                    "fallback_used": False,
                }
            executed_components = push.metadata.get("executed_candidate_components")
            if isinstance(executed_components, dict):
                # Renderer-provided evidence is copied verbatim only after
                # rendering; registry intent alone can never claim C1 ran.
                algorithm_report["executed_candidate_components"] = dict(executed_components)
            component_execution = push.metadata.get("component_execution")
            if isinstance(component_execution, dict):
                algorithm_report["component_execution"] = dict(component_execution)
            component_evidence = push.metadata.get("component_evidence")
            if isinstance(component_evidence, dict):
                algorithm_report["component_evidence"] = dict(component_evidence)
            candidate_run_state = push.metadata.get("candidate_run_state")
            if isinstance(candidate_run_state, str):
                algorithm_report["candidate_run_state"] = candidate_run_state
            component_selection_eligible = push.metadata.get("selection_eligible")
            if isinstance(component_selection_eligible, bool):
                algorithm_report["component_execution_selection_eligible"] = component_selection_eligible
            observability = getattr(args, "observability", None)
            if not isinstance(observability, dict):
                observability = {
                    "report_level": "summary",
                    "artifact_level": "minimal",
                }
            # Development/validation and audit invocations are not a
            # production 3 m minimal benchmark.  Their wall-clock timing is
            # retained as diagnostic evidence only (NE), never as a
            # performance claim that candidate selection could consume.
            performance_evaluable = (
                (getattr(args, "evaluation_scope", None) or "exploratory_full_scan")
                not in {"development_only", "validation_only"}
                and str(observability.get("artifact_level", "minimal")) != "audit"
            )
            performance_grade = (
                ("A" if not budget_exceeded else "C") if performance_evaluable else "NE"
            )
            performance["performance_evaluation"] = (
                "evaluated_primary_delivery" if performance_evaluable else "not_evaluated_validation_or_audit"
            )
            overall_grade = "A" if grade == "A" else "C"
            invalid_component_execution = (
                algorithm_report.get("role") == "candidate"
                and algorithm_report.get("candidate_run_state") == "invalid_component_execution"
            )
            if invalid_component_execution:
                strict = False
                overall_grade = "F"
            report: dict[str, Any] = {
                "input": str(input_path),
                "capture_mode": video.capture_mode,
                "legacy_v1_compatibility": video.legacy_v1,
                "delivery_state": (
                    "experiment_invalid" if invalid_component_execution
                    else ("published" if grade == "A" else "published_degraded")
                ),
                "algorithm": algorithm_report,
                "observability": dict(observability),
                "grades": {
                    "structural": "A",
                    "implementation": "F" if invalid_component_execution else "A",
                    "visual": overall_grade,
                    "performance": performance_grade,
                    "overall": overall_grade,
                },
                "strict_quality_pass": strict,
                "manual_review_required": grade != "A" or invalid_component_execution,
                "defer_3d": bool(args.defer_3d),
                "online_state": {
                    "reused": online_state is not None,
                    "origin": online_state.origin if online_state is not None else None,
                    "strict_frame_files_certified": (
                        online_state.certifies_strict_frame_files
                        if online_state is not None
                        else False
                    ),
                    "path": str(requested_online_state) if requested_online_state else None,
                },
                "scan_segment": segment,
                "source_progress_evidence": (
                    {
                        "schema": source_progress_evidence["schema"],
                        "coordinate": source_progress_evidence["coordinate"],
                        "content_sha256": source_progress_evidence["content_sha256"],
                        "measurement_only": True,
                    }
                    if source_progress_evidence is not None
                    else None
                ),
                "scan_progress_restriction": progress_restriction,
                "evaluation_scope": (
                    getattr(args, "evaluation_scope", None) or "exploratory_full_scan"
                ),
                "motion_analysis_frame_ids": full_scan_ids,
                "orb_tracking_frame_ids": tracking_ids,
                "orb_tracking_source": "real_time_spaced_video_frames_only",
                "untracked_motion_analysis_frames_rendered": False,
                "render_keyframes": render_plan,
                "source_frame_ids": [frame.frame_id for frame in sources],
                "input_sha256": input_hashes,
                "capture_quality": capture_quality,
                "performance": performance,
                "orbslam3": {
                    "tracked_frame_ids": list(tracked_ids),
                    "camera_to_world": [poses_by_id[frame_id].tolist() for frame_id in tracked_ids],
                    "attempts": attempts,
                    "reused_validated_trajectory": trajectory_reused,
                    "pose_convention": "camera_to_world",
                    "translation_unit": "mm",
                },
                "open3d_edges": [edge.as_dict() for edge in edges],
                "renderer": dict(push.metadata),
            }
            if dense_layout is not None:
                report["dense_real_frame_layout"] = dense_layout.as_dict()
                report["dense_owner_observability"] = dense_owner_observability
            if budget_exceeded:
                report["performance_degradation_reason"] = "post_capture_sla_exceeded"
            if publish_auxiliary_exports:
                central_strip_export = push.metadata.get("central_strip_export")
                owner_only_export = push.metadata.get("central_strip_owner_only_export")
                if not isinstance(central_strip_export, dict) or not isinstance(owner_only_export, dict):
                    raise RuntimeError("Video renderer did not stage required central-strip exports")
                report["central_strip_export"] = dict(central_strip_export)
                report["central_strip_owner_only_export"] = dict(owner_only_export)
            else:
                report["auxiliary_exports"] = {
                    "central_strips": "not_published_fast",
                    "central_strips_owner_only": "not_published_fast",
                }
        if invalid_component_execution:
            published = write_invalid_candidate_experiment(output, report)
            two_d_published = True
            return published
        with profiler.stage("atomic_2d_delivery"):
            published = publish_video_2d(
                output,
                push.panorama,
                push.owner_frame_id,
                report,
                pending_central_strips=central_strip_output_dir,
                pending_central_strips_owner_only=central_strip_owner_only_output_dir,
            )
        two_d_published = True
        # C5--C8 retain only immutable calibrated-grid geometry here; fixed
        # labels are first read after atomic primary publication.  This keeps
        # their offline visual evaluation available without giving labels any
        # route into RGB, owner, seam, pose, or source selection.
        annotations = getattr(args, "measurement_annotations", None)
        measurement_payload = (
            d1_measurement_payload if d1_measurement_payload is not None
            else push.measurement_projection_payload
        )
        measurement_masks = (
            d1_measurement_masks if d1_measurement_masks is not None
            else push.measurement_projection_masks
        )
        if (
            measurement_payload is None
            and measurement_masks is None
            and isinstance(annotations, dict)
            and getattr(push, "post_publication_measurement_context", None) is not None
        ):
            try:
                from .video_v2_route import build_v2_post_publication_measurement_projection

                measurement_payload, measurement_masks = build_v2_post_publication_measurement_projection(
                    context=push.post_publication_measurement_context,
                    annotations=annotations,
                    final_owner_frame_id=push.owner_frame_id,
                )
            except Exception as exc:
                # Labels are evaluation evidence only; projection failure must
                # not revoke the already-published primary artifact.
                published["offline_visual_evaluation_error"] = str(exc)
        if (
            measurement_payload is not None
            and measurement_masks is not None
        ):
            # Projection evidence is explicitly post-publication and
            # non-primary.  It has no route back to renderer decisions.
            from .video_candidate_annotation_projection import (
                write_candidate_annotation_projection_sidecar,
            )

            with profiler.offline_evaluation():
                projection_path, masks_path = write_candidate_annotation_projection_sidecar(
                    output / "video_annotation_projection.json",
                    measurement_payload,
                    measurement_masks,
                )
            published["candidate_annotation_projection"] = {
                "path": str(projection_path),
                "masks": str(masks_path),
                "measurement_only": True,
            }
            if isinstance(annotations, dict):
                # This evidence is deliberately generated *after* the atomic
                # primary delivery.  It reloads the projection sidecar and
                # evaluates only published pixels/provenance, so it cannot
                # influence source selection, poses, owner labels or RGB.
                try:
                    from .video_offline_evaluation import (
                        evaluate_offline_visual_annotations,
                        load_panorama_annotation_projection,
                        write_offline_evaluation,
                    )

                    with profiler.offline_evaluation():
                        projection = load_panorama_annotation_projection(
                            projection_path,
                            annotations=annotations,
                            panorama_shape=push.panorama.shape[:2],
                        )
                        evaluation = evaluate_offline_visual_annotations(
                            push.panorama,
                            push.owner_frame_id,
                            annotations=annotations,
                            projection=projection,
                        )
                        evaluation_path = write_offline_evaluation(
                            output / "visual_metrics.json", evaluation
                        )
                    published["offline_visual_evaluation"] = {
                        "path": str(evaluation_path),
                        "measurement_only": True,
                    }
                except Exception as exc:
                    # Evidence must never revoke an already atomically
                    # published candidate result.  Absence of this sidecar is
                    # nevertheless fail-closed for future selection.
                    published["offline_visual_evaluation_error"] = str(exc)
        if source_progress_evidence is not None:
            # Freeze full-scan, real-frame progress after primary publication.
            # A mismatch with a fixed annotation is reported as an immutable
            # measurement failure; it never changes RGB, owner, poses, or the
            # annotations themselves.
            from .video_annotations import audit_annotation_source_progress
            from .video_split import (
                source_progress_by_frame,
                write_or_verify_source_progress_evidence,
            )

            with profiler.offline_evaluation():
                evidence_path = output / "video_source_progress_evidence.json"
                evidence = write_or_verify_source_progress_evidence(
                    evidence_path, source_progress_evidence
                )
                audit = audit_annotation_source_progress(
                    args.measurement_annotations,
                    source_progress_by_frame(evidence),
                )
                audit["source_progress_evidence_sha256"] = evidence["content_sha256"]
                audit["selection_eligible"] = bool(audit["verified"])
                audit_path = output / "video_annotation_source_progress_audit.json"
                pending = audit_path.with_name(f".{audit_path.name}.pending")
                try:
                    pending.write_text(
                        json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8"
                    )
                    os.replace(pending, audit_path)
                finally:
                    pending.unlink(missing_ok=True)
            published["candidate_annotation_source_progress"] = {
                "evidence": str(evidence_path),
                "audit": str(audit_path),
                "verified": bool(audit["verified"]),
                "selection_eligible": bool(audit["selection_eligible"]),
            }
        # ``video_report.json`` is already part of the atomically published
        # primary delivery.  Persist the final post-publication timing in a
        # separate audit sidecar rather than mutating that primary report.
        # Consumers can therefore see all three clocks without mistaking
        # annotation/audit work for renderer latency.
        final_timing = profiler.as_dict(maximum_post_seconds=maximum_post_seconds)
        timing_path = output / "video_timing.json"
        pending_timing = timing_path.with_name(f".{timing_path.name}.pending")
        try:
            pending_timing.write_text(
                json.dumps(final_timing, indent=2, sort_keys=True), encoding="utf-8"
            )
            os.replace(pending_timing, timing_path)
        finally:
            pending_timing.unlink(missing_ok=True)
        published["performance_timing"] = {
            "path": str(timing_path),
            **final_timing,
            "measurement_only": True,
        }
        if not args.defer_3d:
            try:
                publish_video_3d(output, input_path=input_path, config=config)
            except Exception:
                published["three_d_delivery_state"] = "failed"
        return published
    except Exception as exc:
        if not two_d_published:
            write_video_failure(output, input_path, exc)
        raise


def main() -> None:
    # The public command intentionally exposes only the frozen production
    # algorithm.  Baseline and candidate work live in video_experiment; the
    # retained implementation above is called there through video_pipeline.
    from .video_pipeline import production_parser, run_production

    if "--preset" in sys.argv:
        raise SystemExit(
            "ERROR: --preset has been removed. Development uses "
            "g305-video-experiment baseline/candidate; audit uses "
            "--report-level full --artifact-level audit."
        )
    args = production_parser().parse_args()
    try:
        report = run_production(args)
    except Exception as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    print(f"Video panorama: {report['panorama']}")


if __name__ == "__main__":
    main()
