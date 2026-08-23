"""Create the S013 M6.1 Q0+B0 proposed quality threshold and stop."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from panorama_demo.video_s13_m61_quality import (
    calibrate_m61_quality,
    load_baseline_input,
    write_proposed_calibration,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-input",
        action="append",
        required=True,
        type=Path,
        help="Step-2 branch-specific sealed P2-v4 calibration_input.json; repeat exactly four times",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if len(args.baseline_input) != 4:
        raise SystemExit("--baseline-input must be supplied exactly four times")
    loaded = [(path.resolve(), load_baseline_input(path.resolve())) for path in args.baseline_input]
    assets = calibrate_m61_quality(loaded)
    written = write_proposed_calibration(args.output_dir, assets)
    print(json.dumps({
        "status": "paused_for_threshold_approval",
        "candidate_execution_allowed": False,
        "written": [str(path) for path in written],
        "proposed_threshold_sha256": __import__("hashlib").sha256(written[-1].read_bytes()).hexdigest(),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
