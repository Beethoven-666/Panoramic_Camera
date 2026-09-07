#!/usr/bin/env python3
"""Sole writer of gemini305-sdk-native-acceptance/v3 native status."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qualification.native_h0.aggregator import publish


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("raw-root", "native-binding", "candidate-index", "software-acceptance", "output-directory"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--subject-verification", type=Path,
                        help="Staging candidate paths to revalidate when a native binding could not be created; cannot qualify hardware")
    args = parser.parse_args()
    result = publish(args.raw_root, args.native_binding, args.candidate_index,
                     args.software_acceptance, args.output_directory, subject_verification=args.subject_verification)
    print(result["status"])
    return 0 if result["hardware_qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
