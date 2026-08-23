from __future__ import annotations

import argparse
from pathlib import Path

from panorama_demo.video_s13_m9 import compare_profiles_exact, profile_s13_pipeline, summarize_benchmark


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the sealed four-branch M9 benchmark")
    parser.add_argument("--evaluation-lock", required=True, type=Path)
    parser.add_argument("--benchmark-root", required=True, type=Path)
    parser.add_argument("--profile", required=True, choices=("baseline", "optimized"))
    args = parser.parse_args()
    output = args.benchmark_root / f"{args.profile}.json"
    profile_s13_pipeline(args.evaluation_lock, output, label=args.profile)
    if (args.benchmark_root / "baseline.json").is_file() and (args.benchmark_root / "optimized.json").is_file():
        compare_profiles_exact(args.benchmark_root / "baseline.json", args.benchmark_root / "optimized.json", args.benchmark_root / "equivalence.json")
        summarize_benchmark(args.benchmark_root)
    print(output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
