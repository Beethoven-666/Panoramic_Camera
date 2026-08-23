"""Run exact locked S013 V11 on a real acceptance session without role relabeling."""

from __future__ import annotations

import argparse
from pathlib import Path

from panorama_demo.video_algorithm_registry import resolve_video_algorithm
from panorama_demo.video_pipeline import _lock_paths
from panorama_demo.video_s13_production import run_s13_v11_production


def main() -> None:
    parser = argparse.ArgumentParser(description="Run locked S013 V11 offline acceptance")
    parser.add_argument("session", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--maximum-post-seconds", type=float, default=60.0)
    args = parser.parse_args()
    _, baseline_lock, production_lock = _lock_paths(args.config)
    spec = resolve_video_algorithm(
        "production",
        baseline_lock=baseline_lock,
        production_lock=production_lock,
        candidate_config=None,
    )
    report = run_s13_v11_production(
        session_path=args.session,
        output=args.output,
        algorithm_spec=spec,
        maximum_post_seconds=args.maximum_post_seconds,
    )
    print(f"Video panorama: {report['panorama']}")


if __name__ == "__main__":
    main()
