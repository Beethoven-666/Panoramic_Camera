#!/usr/bin/env python3
"""Finalize the camera/host binding once after recorded replay and inventory."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qualification.native_h0.binding import finalize_binding


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    args = parser.parse_args()
    finalize_binding(args.raw_root)


if __name__ == "__main__":
    main()
