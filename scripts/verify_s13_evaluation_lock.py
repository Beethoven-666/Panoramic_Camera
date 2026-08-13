from __future__ import annotations
import argparse
from pathlib import Path
from panorama_demo.video_s13_evaluation import verify_evaluation_lock


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("lock", type=Path)
    a = p.parse_args()
    print(verify_evaluation_lock(a.lock.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
