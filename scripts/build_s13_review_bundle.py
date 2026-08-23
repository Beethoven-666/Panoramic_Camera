from __future__ import annotations
import argparse
from pathlib import Path
from panorama_demo.video_s13_evaluation import BRANCHES, build_review_bundle


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--acceptance-root", type=Path, required=True)
    p.add_argument("--evaluation-id", required=True)
    a = p.parse_args()
    repo = Path(__file__).resolve().parents[1]
    for branch in BRANCHES:
        print(
            build_review_bundle(
                repo, a.acceptance_root.resolve(), a.evaluation_id, branch
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
