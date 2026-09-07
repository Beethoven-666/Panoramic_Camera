"""Standalone native H0 inventory; does not write hardware qualification status."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qualification.native_h0.inventory import collect_inventory, read_json, inventory_errors


def main():
    import argparse
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--storage-root", required=True, type=Path)
    parser.add_argument("--camera-lock-root", type=Path, default=Path("/var/lock/gemini305-sdk"))
    parser.add_argument("--native-binding", type=Path, required=True)
    parser.add_argument("--native-replay", type=Path)
    parser.add_argument("--udev-rule", type=Path)
    args = parser.parse_args()
    binding = read_json(args.native_binding)
    verified = False
    if args.native_replay:
        from qualification.native_h0.campaign import verify_operation
        from qualification.native_h0.equivalence import verify_native_equivalence
        campaign_root = args.output_directory.resolve().parent
        replay_root = verify_operation(campaign_root, binding, "preflight/native-replay")
        supplied = args.native_replay.resolve()
        if supplied not in {replay_root, replay_root / "replay_result.json"}:
            raise ValueError("Replay path is not this H0 campaign's native replay")
        replay = read_json(replay_root / "replay_result.json")
        session = Path(binding["frozen_session"]["root"])
        verify_native_equivalence(replay_root / "sdk", replay_root / "cli", session_root=session)
        verify_native_equivalence(replay_root / "sdk", Path(replay["reference_2d"]), session_root=session)
        verified = True
    records = collect_inventory(args.output_directory, args.storage_root, args.camera_lock_root,
                                binding=binding, replay_verified=verified, udev_rule=args.udev_rule)
    validation_binding = dict(binding)
    if not binding.get("camera_serial"):
        validation_binding.update(camera_serial=records["camera"].get("device", {}).get("serial_number"),
            usb_port_path=next(iter(records["usb"].get("matched_ports", [])), None),
            udev_rule_sha256=records["camera"].get("udev_rule", {}).get("sha256"))
    errors = inventory_errors(records, validation_binding)
    print(json.dumps({"status": "BLOCKED" if errors else "INVENTORY_COLLECTED", "errors": errors}, indent=2))
    return 2 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
