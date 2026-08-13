from __future__ import annotations

import argparse
from pathlib import Path

from panorama_demo.video_s13_m9 import summarize_benchmark


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-root", required=True, type=Path)
    args = parser.parse_args()
    summarize_benchmark(args.benchmark_root)
    print(args.benchmark_root.resolve() / "benchmark_completion.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
