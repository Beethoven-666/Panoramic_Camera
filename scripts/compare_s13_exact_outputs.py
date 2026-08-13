from __future__ import annotations

import argparse
import json
from pathlib import Path

from panorama_demo.video_s13_m9 import compare_exact_file


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    args = parser.parse_args()
    result = compare_exact_file(args.left, args.right)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["exact"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
