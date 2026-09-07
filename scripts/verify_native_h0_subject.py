#!/usr/bin/env python3
"""Read-only frozen candidate audit on a staging host; never a native H0 binding."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qualification.native_h0.binding import verify_candidate, verify_software
from qualification.native_h0.common import read, write_new


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--software-acceptance", type=Path, required=True)
    parser.add_argument("--candidate-index", type=Path, required=True)
    parser.add_argument("--phase3-freeze", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # Stage4 paths are only locators, all bytes are rechecked against frozen identities.
    old = read(args.software_acceptance)["binding"]
    candidate = verify_candidate(args.candidate_index, args.phase3_freeze,
        {k: v["path"] for k, v in old["artifacts"].items()}, old["extracted_roots"]["base"],
        old["extracted_roots"]["three_d_addon"], old["orb_runtime"]["manifest_path"])
    software = verify_software(args.software_acceptance, candidate)
    write_new(args.output, {"schema": "gemini305-native-h0-subject-verification/v1", "status": "PASS",
        "candidate": candidate, "software_acceptance": software, "software_ready": True,
        "hardware_qualified": False, "long_duration_qualified": False, "release_ready": False,
        "signature_status": "UNSIGNED", "scope": "READ_ONLY_FROZEN_SUBJECT_AUDIT"})


if __name__ == "__main__":
    main()
