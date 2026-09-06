#!/usr/bin/env bash
set -euo pipefail

ORB_COMMIT=4452a3c4ab75b1cde34e5505a36ec3f9edcdc4c4
PANGOLIN_COMMIT=aff6883c83f3fd7e8268a9715e84266c42e2efe3
JOBS=${G305_BUILD_JOBS:-2}
WORK_ROOT=
INSTALL_ROOT=
FINAL_INSTALL_ROOT=

usage() {
  echo "usage: $0 --work-root PATH --install-root PATH" >&2
}

while (($#)); do
  case "$1" in
    --work-root) WORK_ROOT=${2:?}; shift 2 ;;
    --install-root) INSTALL_ROOT=${2:?}; shift 2 ;;
    *) usage; exit 2 ;;
  esac
done
[[ -n "$WORK_ROOT" && -n "$INSTALL_ROOT" ]] || { usage; exit 2; }
[[ "$JOBS" =~ ^[12]$ ]] || { echo "G305_BUILD_JOBS must be 1 or 2" >&2; exit 2; }

WORK_ROOT=$(readlink -m -- "$WORK_ROOT")
INSTALL_ROOT=$(readlink -m -- "$INSTALL_ROOT")
FINAL_INSTALL_ROOT="$INSTALL_ROOT"
INSTALL_ROOT="${FINAL_INSTALL_ROOT}.unrelocated"
[[ ! -e "$FINAL_INSTALL_ROOT" ]] || { echo "Install root already exists" >&2; exit 1; }
command -v patchelf >/dev/null || { echo "patchelf is required for relative RPATH" >&2; exit 1; }
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PATCH_PATH="$SCRIPT_DIR/patches/orbslam3-g305-headless-runners.patch"
ORB_SOURCE="$WORK_ROOT/ORB_SLAM3"
PANGOLIN_SOURCE="$WORK_ROOT/Pangolin"
PANGOLIN_INSTALL="$WORK_ROOT/pangolin-install"

for target in "$ORB_SOURCE" "$PANGOLIN_SOURCE" "$PANGOLIN_INSTALL" "$INSTALL_ROOT"; do
  if [[ -e "$target" ]]; then
    echo "Refusing to reuse an existing build/install path: $target" >&2
    exit 1
  fi
done
mkdir -p -- "$WORK_ROOT"

git clone https://github.com/stevenlovegrove/Pangolin.git "$PANGOLIN_SOURCE"
git -C "$PANGOLIN_SOURCE" checkout --detach "$PANGOLIN_COMMIT"
cmake -S "$PANGOLIN_SOURCE" -B "$PANGOLIN_SOURCE/build-g305" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DCMAKE_INSTALL_PREFIX="$PANGOLIN_INSTALL" \
  -DBUILD_EXAMPLES=OFF -DBUILD_TOOLS=OFF -DBUILD_PANGOLIN_GUI_VARS=OFF
cmake --build "$PANGOLIN_SOURCE/build-g305" --parallel "$JOBS"
cmake --install "$PANGOLIN_SOURCE/build-g305"

git clone https://github.com/UZ-SLAMLab/ORB_SLAM3.git "$ORB_SOURCE"
git -C "$ORB_SOURCE" checkout --detach "$ORB_COMMIT"
git -C "$ORB_SOURCE" apply --check "$PATCH_PATH"
git -C "$ORB_SOURCE" apply "$PATCH_PATH"

cmake -S "$ORB_SOURCE/Thirdparty/DBoW2" -B "$ORB_SOURCE/Thirdparty/DBoW2/build-g305" \
  -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build "$ORB_SOURCE/Thirdparty/DBoW2/build-g305" --parallel "$JOBS"
cmake -S "$ORB_SOURCE/Thirdparty/g2o" -B "$ORB_SOURCE/Thirdparty/g2o/build-g305" \
  -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build "$ORB_SOURCE/Thirdparty/g2o/build-g305" --parallel "$JOBS"
cmake -S "$ORB_SOURCE/Thirdparty/Sophus" -B "$ORB_SOURCE/Thirdparty/Sophus/build-g305" \
  -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build "$ORB_SOURCE/Thirdparty/Sophus/build-g305" --parallel "$JOBS"

if [[ ! -f "$ORB_SOURCE/Vocabulary/ORBvoc.txt" ]]; then
  tar -xf "$ORB_SOURCE/Vocabulary/ORBvoc.txt.tar.gz" -C "$ORB_SOURCE/Vocabulary"
fi
mkdir -p "$INSTALL_ROOT/lib" "$INSTALL_ROOT/Thirdparty/DBoW2/lib" \
  "$INSTALL_ROOT/Thirdparty/g2o/lib" "$INSTALL_ROOT/Examples/RGB-D" \
  "$INSTALL_ROOT/Vocabulary"
cmake -S "$ORB_SOURCE" -B "$ORB_SOURCE/build-g305" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DCMAKE_PREFIX_PATH="$PANGOLIN_INSTALL" \
  -DCMAKE_BUILD_RPATH="$INSTALL_ROOT/lib;$INSTALL_ROOT/Thirdparty/DBoW2/lib;$INSTALL_ROOT/Thirdparty/g2o/lib;$PANGOLIN_INSTALL/lib"
cmake --build "$ORB_SOURCE/build-g305" --parallel "$JOBS" \
  --target rgbd_tum_headless rgbd_g305_stream_headless

cp "$ORB_SOURCE/Examples/RGB-D/rgbd_tum_headless" "$INSTALL_ROOT/Examples/RGB-D/"
cp "$ORB_SOURCE/Examples/RGB-D/rgbd_g305_stream_headless" "$INSTALL_ROOT/Examples/RGB-D/"
cp "$ORB_SOURCE/Vocabulary/ORBvoc.txt" "$INSTALL_ROOT/Vocabulary/"
cp "$ORB_SOURCE/lib/libORB_SLAM3.so" "$INSTALL_ROOT/lib/"
cp "$ORB_SOURCE/Thirdparty/DBoW2/lib/libDBoW2.so" "$INSTALL_ROOT/Thirdparty/DBoW2/lib/"
cp "$ORB_SOURCE/Thirdparty/g2o/lib/libg2o.so" "$INSTALL_ROOT/Thirdparty/g2o/lib/"
cp "$ORB_SOURCE/LICENSE" "$INSTALL_ROOT/COPYING"

for executable in rgbd_tum_headless rgbd_g305_stream_headless; do
  path="$INSTALL_ROOT/Examples/RGB-D/$executable"
  test -x "$path"
  if ldd "$path" | grep -q 'not found'; then
    ldd "$path" >&2
    exit 1
  fi
done

python3 - "$INSTALL_ROOT" "$ORB_COMMIT" "$PANGOLIN_COMMIT" "$PATCH_PATH" <<'PY'
import hashlib
import json
import pathlib
import platform
import subprocess
import sys

root = pathlib.Path(sys.argv[1])
orb_commit, pangolin_commit = sys.argv[2:4]
patch = pathlib.Path(sys.argv[4])
executables = [
    root / "Examples/RGB-D/rgbd_tum_headless",
    root / "Examples/RGB-D/rgbd_g305_stream_headless",
]
payload = {
    "schema": "gemini305-orbslam3-linux-runtime/v1",
    "orbslam3_commit": orb_commit,
    "pangolin_commit": pangolin_commit,
    "patch_sha256": hashlib.sha256(patch.read_bytes()).hexdigest(),
    "compiler": subprocess.check_output(["c++", "--version"], text=True).splitlines()[0],
    "cmake": subprocess.check_output(["cmake", "--version"], text=True).splitlines()[0],
    "platform": platform.platform(),
    "artifacts": {},
}
for path in [*executables, root / "Vocabulary/ORBvoc.txt"]:
    payload["artifacts"][str(path.relative_to(root))] = {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
    }
payload["ldd"] = {
    path.name: subprocess.check_output(["ldd", str(path)], text=True).splitlines()
    for path in executables
}
(root / "runtime-manifest.json").write_text(
    json.dumps(payload, indent=2) + "\n", encoding="utf-8"
)
PY

python3 "$SCRIPT_DIR/relocate_orb_runtime.py" --source "$INSTALL_ROOT" --output "$FINAL_INSTALL_ROOT"
echo "Internal ORB-SLAM3 Runtime installed at $FINAL_INSTALL_ROOT; no public distribution approval"
