from __future__ import annotations

import argparse
import json
from pathlib import Path

from panorama_demo.video_s13_m61_summary import parse_branch_spec, summarize_acceptance


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize four formal/rerun S013 M6.1 branches")
    parser.add_argument(
        "--branch", action="append", required=True,
        help="Repeat four times: branch=formal_p3:rerun_p3:old_p3:p2_v4",
    )
    parser.add_argument("--output", type=Path, required=True, help="New absent output directory")
    args = parser.parse_args(argv)
    written = summarize_acceptance([parse_branch_spec(value) for value in args.branch], args.output)
    print(json.dumps({"written": [str(path) for path in written]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
