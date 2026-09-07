#!/usr/bin/env python3
"""Bind a native campaign to the immutable SDK candidate and external tools."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qualification.native_h0.binding import create_binding
from qualification.native_h0.common import write_new


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("candidate-index", "software-acceptance", "phase3-freeze", "base-archive", "addon-archive",
                 "source-archive", "base-root", "addon-root", "orb-manifest", "frozen-session",
                 "tools-archive", "tools-manifest", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--host-inventory", type=Path)
    parser.add_argument("--camera-inventory", type=Path)
    parser.add_argument("--usb-inventory", type=Path)
    parser.add_argument("--h0-id", required=True)
    args = parser.parse_args()
    result = create_binding(args.candidate_index, args.software_acceptance, args.phase3_freeze,
        {"base": args.base_archive, "three_d_addon": args.addon_archive, "source_compliance": args.source_archive},
        args.base_root, args.addon_root, args.orb_manifest, args.frozen_session,
        args.tools_archive, args.tools_manifest, args.host_inventory, args.camera_inventory,
        h0_id=args.h0_id, usb_inventory=args.usb_inventory)
    write_new(args.output, result)


if __name__ == "__main__":
    main()
