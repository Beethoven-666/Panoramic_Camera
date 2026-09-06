#!/usr/bin/env bash
set -euo pipefail
# Compatibility command; elevation must be explicitly requested by the user.
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
exec bash "$ROOT/packaging/linux/setup_orbbec_udev.sh" "$@"
