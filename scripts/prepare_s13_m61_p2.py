from __future__ import annotations

import argparse
import json
from pathlib import Path

from panorama_demo.video_s13_m61_evidence import (
    S13PhotometricEvidenceConfig,
    prepare_s13_m61_p2,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare one branch-specific sealed S013 P2-v4")
    parser.add_argument("--parent-p2", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--evidence-config", type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    config = S13PhotometricEvidenceConfig()
    if args.evidence_config is not None:
        value = json.loads(args.evidence_config.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("photometric evidence config must be a JSON object")
        config = S13PhotometricEvidenceConfig.from_mapping(value)
    result = prepare_s13_m61_p2(
        args.parent_p2, args.output, branch=args.branch, session=args.session, config=config
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
