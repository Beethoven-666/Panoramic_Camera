from __future__ import annotations

import argparse
from pathlib import Path

from panorama_demo.video_s13_m9 import compare_profiles_exact


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--optimized", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = compare_profiles_exact(args.baseline, args.optimized, args.output)
    print(args.output.resolve())
    return 0 if result["exact"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
