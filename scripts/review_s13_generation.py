from __future__ import annotations
import argparse
from pathlib import Path
from panorama_demo.video_s13_review import review_generation


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--acceptance-root", type=Path, required=True)
    p.add_argument("--evaluation-id", required=True)
    p.add_argument("--branch", required=True)
    p.add_argument("--stage", choices=("P2", "P3", "P4"), required=True)
    p.add_argument("--reviewer", required=True)
    p.add_argument(
        "--review-method", choices=("manual", "offline_dataset"), required=True
    )
    p.add_argument("--note", default="")
    a = p.parse_args()
    bundle = a.acceptance_root / "evaluation" / a.evaluation_id / a.branch
    print(
        review_generation(
            a.acceptance_root.resolve(),
            bundle.resolve(),
            a.branch,
            stage=a.stage,
            reviewer=a.reviewer,
            review_method=a.review_method,
            note=a.note,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
