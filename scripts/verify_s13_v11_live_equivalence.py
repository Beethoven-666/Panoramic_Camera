"""Verify exact formal live/offline S013 V11 production equivalence."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from panorama_demo.video_s13_live_acceptance import compare_s13_v11_live_offline


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fail closed unless live and offline S013 V11 productions are exact"
    )
    parser.add_argument("offline", type=Path)
    parser.add_argument("live", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    report = compare_s13_v11_live_offline(args.offline, args.live)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pending = args.output.with_name(f".{args.output.name}.pending")
    pending.write_text(json.dumps(report, indent=2), encoding="utf-8")
    os.replace(pending, args.output)
    return 0 if report["equivalent"] is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
