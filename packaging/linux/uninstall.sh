#!/usr/bin/env bash
set -euo pipefail
ARGS=("$@")
PYTHON=
while (($#)); do
  case "$1" in
    --python) PYTHON=${2:?}; shift 2 ;;
    *) shift ;;
  esac
done
[[ -n "$PYTHON" ]] || { echo "--python is required" >&2; exit 2; }
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec "$PYTHON" "$SCRIPT_DIR/install_runtime.py" "${ARGS[@]}" --action uninstall
