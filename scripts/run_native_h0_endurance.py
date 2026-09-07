#!/usr/bin/env python3
"""Run complete native endurance with a calibrated disk budget."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qualification.native_h0.binding import revalidate_binding
from qualification.native_h0.campaign import Campaign, run_worker, verify_operation
from qualification.native_h0.common import read, require
from qualification.native_h0.inventory import validate_inventory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--native-binding", type=Path, required=True)
    args = parser.parse_args()
    binding = read(args.native_binding)
    revalidate_binding(binding, binding["candidate"]["candidate_index"]["path"], binding["software_acceptance"]["path"])
    campaign = Campaign(args.raw_root, binding)
    require(not validate_inventory(campaign.root, binding)["errors"], "Native inventory must pass")
    verify_operation(campaign.root, binding, "preflight/native-replay")
    profile = read(campaign.root / "profile.json")
    for name in ("storage-calibration-60s", "continuous-30m", "continuous-2h"):
        with campaign.operation("endurance/" + name) as root:
            run_worker("endurance", root, binding, name=name, profile=profile,
                       calibration=str(campaign.root / "endurance/storage-calibration-60s"))


if __name__ == "__main__":
    main()
