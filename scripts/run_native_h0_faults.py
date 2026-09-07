"""Run the eight physical fault cases; unplug actions use recorded checkpoints."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qualification.native_h0.binding import revalidate_binding
from qualification.native_h0.campaign import Campaign, run_worker, verify_operation
from qualification.native_h0.common import read, require
from qualification.native_h0.fault_cases import FAULT_CASES, validate_fault_campaign, validate_loopback_mount
from qualification.native_h0.inventory import validate_inventory


def main():
    import argparse
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--native-binding", type=Path, required=True)
    parser.add_argument("--loopback-root", type=Path, required=True)
    args = parser.parse_args()
    binding = read(args.native_binding)
    revalidate_binding(binding, binding["candidate"]["candidate_index"]["path"], binding["software_acceptance"]["path"])
    campaign = Campaign(args.raw_root, binding)
    require(not validate_inventory(campaign.root, binding)["errors"], "Native inventory must pass")
    verify_operation(campaign.root, binding, "preflight/native-replay")
    validate_loopback_mount(args.loopback_root)
    for case in FAULT_CASES:
        with campaign.operation("faults/" + case) as root:
            run_worker("fault", root, binding, case=case, loopback_root=str(args.loopback_root))
    result = validate_fault_campaign(campaign.root, binding)
    print(json.dumps(result, indent=2))
    return 2 if result["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
