"""Formal continuous RGB-D capture with incremental S013 V11 preview and final P3."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from .capture_orbbec import build_parser as capture_parser
from .capture_orbbec import run_video_capture
from .video_algorithm_registry import resolve_video_algorithm
from .video_pipeline import _lock_paths, run_video_algorithm
from .video_s13_contract import (
    S13_VISUAL_CONTINUITY_ALGORITHM_ID,
    S13_VISUAL_CONTINUITY_IMPLEMENTATION_ID,
)
from .video_s13_live import S13V11LiveObserver


_CAPTURE_ZERO_COUNTERS = (
    "online_orb_tracker_construct_count",
    "capture_orbslam3_process_count",
    "capture_orbslam3_frame_submit_count",
    "capture_open3d_import_count",
    "capture_open3d_tsdf_call_count",
    "capture_3d_process_count",
)


def build_parser() -> argparse.ArgumentParser:
    parser = capture_parser()
    parser.description = "Capture continuous RGB-D with S013 V11 live preview and formal P3"
    parser.add_argument(
        "--panorama-output",
        type=Path,
        default=Path("outputs/video_live"),
        help="Formal two-dimensional output directory",
    )
    parser.add_argument("--maximum-post-seconds", type=float, default=60.0)
    parser.add_argument(
        "--post-3d",
        action="store_true",
        help="Spawn isolated ORB-SLAM3/TSDF processing after the 2-D delivery",
    )
    parser.set_defaults(wait_for_camera=True)
    return parser


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    pending = path.with_name(f".{path.name}.pending")
    pending.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(pending, path)


def _require_capture_zero(runtime_audit: object) -> dict[str, object]:
    if not isinstance(runtime_audit, dict):
        raise RuntimeError("Live capture did not publish its runtime audit")
    nonzero = {
        name: runtime_audit.get(name)
        for name in _CAPTURE_ZERO_COUNTERS
        if runtime_audit.get(name) != 0
    }
    if nonzero:
        raise RuntimeError(f"Live capture invoked forbidden heavy processing: {nonzero}")
    return dict(runtime_audit)


def run(args: argparse.Namespace) -> dict[str, object]:
    if bool(args.photo_mode):
        raise ValueError("g305-video-live accepts only continuous RGB-D video capture")
    if bool(args.diagnostic_online_orbslam3):
        raise ValueError("g305-video-live forbids capture-time ORB-SLAM3")
    _, baseline_lock, production_lock = _lock_paths(args.config)
    spec = resolve_video_algorithm(
        "production",
        baseline_lock=baseline_lock,
        production_lock=production_lock,
        candidate_config=None,
    )
    if (
        spec.role != "production"
        or spec.algorithm_id != S13_VISUAL_CONTINUITY_ALGORITHM_ID
        or spec.implementation_id != S13_VISUAL_CONTINUITY_IMPLEMENTATION_ID
        or spec.allow_baseline_fallback
    ):
        raise ValueError("g305-video-live requires the exact locked S013 V11 production algorithm")

    observer = S13V11LiveObserver(
        production_config_sha256=spec.config_sha256,
        preview_output=args.panorama_output,
    )
    try:
        session_root = run_video_capture(args, observer=observer)
    except BaseException:
        # Device discovery happens after the observer starts its worker
        # threads. Ensure Ctrl+C or an early SDK failure cannot leave those
        # non-daemon threads keeping the live command alive.
        observer.on_capture_stopping()
        observer.close_authority()
        raise
    handoff = observer.freeze_handoff()
    snapshot = observer.snapshot()
    manifest = json.loads((session_root / "manifest.json").read_text(encoding="utf-8"))
    runtime_audit = _require_capture_zero(manifest.get("capture_runtime_audit"))
    eligibility = manifest.get("product_eligibility")
    if (
        manifest.get("clean_shutdown") is not True
        or not isinstance(eligibility, dict)
        or eligibility.get("video_panorama") is not True
    ):
        raise RuntimeError("Live capture did not close as an eligible continuous RGB-D session")
    if not handoff.committed_frames:
        raise RuntimeError("Live capture produced no committed RGB-D frames")
    if (
        int(manifest.get("queue_drops", -1)) != 0
        or int(manifest.get("write_errors", -1)) != 0
        or int(manifest.get("received_frames", -1)) != int(manifest.get("written_frames", -2))
    ):
        raise RuntimeError("Live capture writer did not close with exact zero-drop coverage")

    capture_report = {
        "schema": "gemini305-s013-v11-live-capture/v1",
        "algorithm_id": spec.algorithm_id,
        "implementation_id": spec.implementation_id,
        "production_config_sha256": spec.config_sha256,
        "capture_runtime_audit": runtime_audit,
        "received_frames": manifest.get("received_frames"),
        "written_frames": manifest.get("written_frames"),
        "queue_drops": manifest.get("queue_drops"),
        "write_errors": manifest.get("write_errors"),
        "max_queue_depth": manifest.get("max_queue_depth"),
        "preview_updates": snapshot.preview_updates,
        "preview_skipped_due_to_load": snapshot.preview_skipped_due_to_load,
        "preview_failures": snapshot.preview_failures,
        "preview_first_display_seconds": (
            None
            if snapshot.first_preview_monotonic_ns is None
            else (
                snapshot.first_preview_monotonic_ns
                - handoff.capture_started_monotonic_ns
            ) / 1_000_000_000.0
        ),
        "preview_latency_p95_ms": snapshot.preview_latency_p95_ms,
        "preview_authority": "non_authoritative_live_preview",
        "handoff_reuse_level": handoff.reuse_level,
    }
    _write_json_atomic(session_root / "live_capture_report.json", capture_report)
    published = run_video_algorithm(
        input_path=session_root,
        output=args.panorama_output,
        role="production",
        config_path=args.config,
        maximum_post_seconds=args.maximum_post_seconds,
        defer_3d=not bool(args.post_3d),
        live_handoff=handoff,
    )
    return {"session": str(session_root), "capture": capture_report, "delivery": published}


def main() -> None:
    args = build_parser().parse_args()
    try:
        run(args)
    except KeyboardInterrupt:
        print("Live video cancelled while waiting for camera.", file=sys.stderr)
        raise SystemExit(130) from None
    except Exception as exc:
        print(f"Live video failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()


__all__ = ["build_parser", "run"]
