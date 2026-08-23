from __future__ import annotations
import argparse
from pathlib import Path
from panorama_demo.video_s13_evaluation import build_evaluation_lock


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--acceptance-root", type=Path, required=True)
    p.add_argument("--evaluation-id", required=True)
    p.add_argument("--test-summary", type=Path, required=True)
    p.add_argument("--allow-dirty", action="store_true")
    p.add_argument("--m9-benchmark-completion", type=Path)
    a = p.parse_args()
    print(
        build_evaluation_lock(
            Path(__file__).resolve().parents[1],
            a.acceptance_root.resolve(),
            a.evaluation_id,
            a.test_summary.resolve(),
            allow_dirty=a.allow_dirty,
            m9_benchmark_completion=(
                a.m9_benchmark_completion.resolve()
                if a.m9_benchmark_completion is not None
                else None
            ),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
