from __future__ import annotations
import argparse
from pathlib import Path
from panorama_demo.video_s13_bundle import atomic_write_json, sha256_file
from panorama_demo.video_s13_evaluation import BRANCHES, _json, verify_evaluation_lock


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--acceptance-root", type=Path, required=True)
    p.add_argument("--evaluation-id", required=True)
    a = p.parse_args()
    root = a.acceptance_root.resolve()
    lock = root / "evaluation" / a.evaluation_id / "evaluation.lock.json"
    verify_evaluation_lock(lock)
    runs = []
    for b in BRANCHES:
        q = root / "runs_v4" / b / "formal/current_reviewed.json"
        d = _json(q)
        runs.append(
            {
                "branch": b,
                "stage": d["stage"],
                "generation_id": d["generation_id"],
                "pointer_sha256": sha256_file(q),
            }
        )
    out = root / "M8_validation_summary.json"
    atomic_write_json(
        out,
        {
            "schema": "gemini305-video-s13-m8-validation-summary/v1",
            "state": "completed_diagnostic_review_p3",
            "evaluation_id": a.evaluation_id,
            "runs": runs,
            "p4_created": False,
            "production_lock_written": False,
            "automatic_quality_remains_blocked": True,
            "evaluation_lock_sha256": sha256_file(lock),
        },
    )
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
