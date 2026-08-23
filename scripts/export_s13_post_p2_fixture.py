"""Run the S1.3 candidate once and export its exact post-C2E P2 fixture."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--candidate-config", type=Path, required=True)
    pose = parser.add_mutually_exclusive_group(required=True)
    pose.add_argument("--trajectory-cache", type=Path)
    pose.add_argument("--ignore-pose", action="store_true")
    args = parser.parse_args()
    command = [
        sys.executable,
        "-c",
        "from panorama_demo.video_experiment import main; main()",
        "--algorithm",
        "candidate",
        "--output",
        str(args.output),
        "--candidate-config",
        str(args.candidate_config),
        "--post-p2-fixture",
        str(args.fixture),
        "--m62-execution-mode",
        "candidate_single_pass",
        str(args.input),
    ]
    if args.ignore_pose:
        command.append("--ignore-pose")
    else:
        command.extend(("--trajectory-cache", str(args.trajectory_cache)))
    subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
