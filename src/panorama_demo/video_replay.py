"""Replay a locked RGB-D session into the capture-time scan-state contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

from .config import load_config
from .video_online_state import OnlineScanAccumulator, write_online_state
from .video_runtime_environment import atomic_write_json, capture_runtime_environment
from .video_session import load_video_session


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay a continuous RGB-D video session")
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--mode", choices=("realtime", "unpaced"), default="unpaced")
    parser.add_argument("--write-online-state", type=Path, required=True)
    parser.add_argument(
        "--reference-online-state", type=Path,
        help="Require replay scan facts to equal this content-bound online state.",
    )
    parser.add_argument("--config", type=Path)
    return parser


def _scan_facts_sha256(path: Path) -> str:
    """Hash only facts that replay is allowed to reproduce, not file locations."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid online state: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid online state: {path}")
    facts = {
        key: payload.get(key)
        for key in ("input_sha256", "frame_file_sha256", "qualities", "motions", "scan_segment")
    }
    return hashlib.sha256(
        json.dumps(facts, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()


def run(args: argparse.Namespace) -> dict[str, object]:
    session = load_video_session(args.session, validate_frame_files=True)
    config = load_config(args.config)
    stitch = dict(config.get("stitch", {}))
    runtime = dict(stitch.get("video_runtime", {}))
    accumulator = OnlineScanAccumulator(
        analysis_width=int(stitch.get("analysis_width", 320)),
        motion_backend=str(runtime.get("motion_backend", "dis")),
    )
    previous_timestamp: int | None = None
    started = time.perf_counter()
    import cv2

    for frame in session.rgbd.frames:
        if args.mode == "realtime" and previous_timestamp is not None and frame.timestamp_us is not None:
            delay = max(0.0, (frame.timestamp_us - previous_timestamp) / 1_000_000.0)
            if delay:
                time.sleep(delay)
        image = cv2.imread(str(frame.color_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not decode replay source {frame.color_path}")
        accumulator.add(frame.frame_id, image)
        previous_timestamp = frame.timestamp_us
    qualities, motions, segment = accumulator.finish()
    destination = args.write_online_state.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    staged = destination.with_name(f".{destination.stem}.replay-staged{destination.suffix}")
    write_online_state(
        staged,
        root=session.rgbd.root,
        frames=session.rgbd.frames,
        qualities=qualities,
        motions=motions,
        segment=segment,
        origin="offline_prepare",
    )
    try:
        replay_digest = _scan_facts_sha256(staged)
        reference = getattr(args, "reference_online_state", None)
        if reference is not None:
            reference_digest = _scan_facts_sha256(Path(reference).expanduser().resolve())
            if replay_digest != reference_digest:
                raise ValueError("Replay scan facts do not match the supplied reference online state")
        os.replace(staged, destination)
    finally:
        staged.unlink(missing_ok=True)
    environment_path = destination.parent / "replay_environment.json"
    atomic_write_json(environment_path, capture_runtime_environment())
    result: dict[str, object] = {
        "schema": "gemini305-video-replay-result/v1",
        "online_state": str(destination),
        "mode": args.mode,
        "frame_count": len(session.rgbd.frames),
        "elapsed_seconds": time.perf_counter() - started,
        "scan_facts_sha256": replay_digest,
        "reference_matched": reference is not None,
        "environment": str(environment_path),
    }
    # The elapsed time is evidence, while the scan-facts digest is the stable
    # replay output.  No replay failure is silently converted to a different
    # scan state or tracking input.
    atomic_write_json(destination.parent / "replay_result.json", result)
    return result


def main() -> None:
    args = _parser().parse_args()
    try:
        report = run(args)
    except Exception as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    print(f"Online replay state: {report['online_state']}")
