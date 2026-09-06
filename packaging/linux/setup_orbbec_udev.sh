#!/usr/bin/env bash
set -euo pipefail
[[ $EUID -eq 0 ]] || { echo 'Run explicitly: sudo ./setup_orbbec_udev.sh' >&2; exit 1; }
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# Official v2.9.3 rules, source commit 2f6561c28255d805b34aa00a690199ce40e96c81.
exec sh "$ROOT/orbbec-official/install_udev_rules.sh"
