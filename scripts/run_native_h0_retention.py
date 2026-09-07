"""Run fresh native retention success, failure-preservation, and recovery cases."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qualification.native_h0.binding import revalidate_binding
from qualification.native_h0.campaign import Campaign, run_worker, verify_operation
from qualification.native_h0.common import read, require
from qualification.native_h0.inventory import validate_inventory
from qualification.native_h0.retention import RETENTION_CASES, validate_retention


def main():
    import argparse
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--native-binding", type=Path, required=True)
    args = parser.parse_args()
    binding = read(args.native_binding)
    revalidate_binding(binding, binding["candidate"]["candidate_index"]["path"], binding["software_acceptance"]["path"])
    campaign = Campaign(args.raw_root, binding)
    require(not validate_inventory(campaign.root, binding)["errors"], "Native inventory must pass")
    verify_operation(campaign.root, binding, "preflight/native-replay")
    for case in RETENTION_CASES:
        with campaign.operation("retention/" + case) as root:
            run_worker("retention", root, binding, case=case)
    result = validate_retention(campaign.root, binding)
    print(json.dumps(result, indent=2))
    return 2 if result["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
