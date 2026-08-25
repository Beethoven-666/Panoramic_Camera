#!/usr/bin/env python3
"""Thin native-H0 harness that calls only the public Gemini305VideoSDK."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Callable


def _status(status: str, reason_code: str, detail: dict[str, object] | None = None) -> dict[str, object]:
    return {"status": status, "reason_code": reason_code, "detail": detail or {}}


def run(args: argparse.Namespace, *, sdk_factory: Callable[[], Any] | None = None) -> dict[str, object]:
    if args.runs < 1 or args.warmups < 0:
        raise ValueError("runs must be positive and warmups non-negative")
    root = args.artifact_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=False)
    if sdk_factory is None:
        from panorama_demo import Gemini305VideoSDK

        sdk_factory = Gemini305VideoSDK
    sdk = sdk_factory()
    records: list[dict[str, object]] = []
    last_success: Any | None = None
    total = args.warmups + args.runs
    for sequence in range(total):
        warmup = sequence < args.warmups
        formal_index = sequence - args.warmups + 1
        label = f"warmup_{sequence + 1:02d}" if warmup else f"run_{formal_index:02d}"
        run_root = root / label
        run_root.mkdir()
        started = time.perf_counter()
        try:
            job = sdk.capture_and_process(
                capture_root=run_root / "sessions",
                output_dir=run_root / "2d",
                width=args.width,
                height=args.height,
                fps=args.fps,
                duration_seconds=args.duration,
                video_exposure_us=args.video_exposure_us,
                preview_window=not args.no_preview,
                maximum_post_seconds=args.maximum_post_seconds,
            )
            result = job.result()
            record = {
                "label": label, "warmup": warmup, "status": "PASS", "reason_code": "P3_PUBLISHED",
                "wall_seconds": time.perf_counter() - started, "job_state": job.state.value,
                "session": str(result.session), "output": str(result.output_dir),
                "delivery": str(result.delivery_path), "authority": result.authority,
                "grade": result.overall_grade, "manual_review_required": result.manual_review_required,
                "capture_stop_to_p3_published_seconds": result.primary_post_capture_seconds,
            }
            if not warmup:
                last_success = result
        except BaseException as exc:
            record = {
                "label": label, "warmup": warmup, "status": "FAIL", "reason_code": "SDK_CAPTURE_FAILED",
                "wall_seconds": time.perf_counter() - started, "error_type": type(exc).__name__,
                "error": str(exc),
            }
        (run_root / "sdk_result.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        records.append(record)
    formal = [record for record in records if not record["warmup"]]
    success_count = sum(record["status"] == "PASS" for record in formal)
    post_3d = _status("NOT_EXECUTED", "NOT_REQUESTED")
    if args.post_3d_last:
        if last_success is None:
            post_3d = _status("NOT_EXECUTED", "NO_SUCCESSFUL_SESSION")
        else:
            try:
                result_3d = sdk.run_post_3d(last_success.session, last_success.output_dir)
                post_3d = _status("PASS", "THREE_D_PUBLISHED", {
                    "output": str(result_3d.output_dir), "delivery": str(result_3d.delivery_path)})
            except BaseException as exc:
                post_3d = _status("FAIL", "POST_3D_FAILED_2D_PRESERVED", {
                    "error_type": type(exc).__name__, "error": str(exc),
                    "two_d_delivery": str(last_success.delivery_path)})
    passed = success_count == args.runs and (not args.post_3d_last or post_3d["status"] == "PASS")
    result = {
        "schema": "gemini305-sdk-h0-acceptance/v1", "status": "PASS" if passed else "FAIL",
        "reason_code": "SDK_H0_COMPLETE" if passed else "SDK_H0_INCOMPLETE",
        "successful_runs": success_count, "required_runs": args.runs, "records": records,
        "post_3d_last": post_3d,
        "cli_crosscheck": _status("NOT_EXECUTED", "RUN_SEPARATELY_WITH_G305_VIDEO_LIVE"),
    }
    (root / "sdk_h0_acceptance.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--width", type=int, default=848)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--video-exposure-us", type=int, default=100)
    parser.add_argument("--maximum-post-seconds", type=float, default=60.0)
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument("--post-3d-last", action="store_true")
    return parser


def main() -> None:
    result = run(_parser().parse_args())
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
