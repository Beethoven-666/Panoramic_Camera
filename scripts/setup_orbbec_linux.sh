#!/usr/bin/env bash
set -euo pipefail

ORBBEC_COMMIT=63994096ce6a0c07a235500802ce5379490404b7
SOURCE_ROOT=${G305_ORBBEC_SDK_SOURCE:-/home/zyh/build/OrbbecSDK_v2-2.8.6}

if (($#)); then
  if [[ "$1" != "--source-root" || $# -ne 2 ]]; then
    echo "usage: $0 [--source-root PATH]" >&2
    exit 2
  fi
  SOURCE_ROOT=$2
fi
SOURCE_ROOT=$(readlink -m -- "$SOURCE_ROOT")

if [[ ! -d "$SOURCE_ROOT/.git" ]]; then
  [[ ! -e "$SOURCE_ROOT" ]] || {
    echo "Source path exists but is not a Git checkout: $SOURCE_ROOT" >&2
    exit 1
  }
  git clone https://github.com/orbbec/OrbbecSDK_v2.git "$SOURCE_ROOT"
fi
git -C "$SOURCE_ROOT" fetch --depth 1 origin "$ORBBEC_COMMIT"
git -C "$SOURCE_ROOT" checkout --detach "$ORBBEC_COMMIT"
[[ "$(git -C "$SOURCE_ROOT" rev-parse HEAD)" == "$ORBBEC_COMMIT" ]] || exit 1
[[ -z "$(git -C "$SOURCE_ROOT" status --short)" ]] || {
  echo "Refusing to use a modified Orbbec SDK source tree" >&2
  exit 1
}

OFFICIAL_INSTALLER="$SOURCE_ROOT/scripts/env_setup/install_udev_rules.sh"
OFFICIAL_RULES="$SOURCE_ROOT/scripts/env_setup/99-obsensor-libusb.rules"
[[ -f "$OFFICIAL_INSTALLER" && -f "$OFFICIAL_RULES" ]] || {
  echo "Pinned Orbbec SDK source lacks its official udev installer" >&2
  exit 1
}
echo "About to run the official Orbbec SDK v2.8.6 udev installer with sudo."
sudo sh "$OFFICIAL_INSTALLER"
