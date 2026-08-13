from __future__ import annotations

import argparse
from pathlib import Path

from panorama_demo.video_s13_m9 import profile_s13_pipeline


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-lock", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--label", required=True, choices=("baseline", "optimized"))
    args = parser.parse_args()
    profile_s13_pipeline(args.evaluation_lock, args.output, label=args.label)
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
