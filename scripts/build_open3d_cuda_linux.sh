#!/usr/bin/env bash
set -euo pipefail

OPEN3D_COMMIT=1e7b17438687a0b0c1e5a7187321ac7044afe275
SOURCE_ROOT=/home/zyh/build/Open3D-0.19
BUILD_ROOT=/home/zyh/build/Open3D-0.19/build-cuda12.8-sm120
WHEEL_DIR=/home/zyh/build/wheels
PYTHON=python
JOBS=${G305_BUILD_JOBS:-2}

usage() {
  echo "usage: $0 [--source-root PATH] [--build-root PATH] [--wheel-dir PATH] [--python PATH]" >&2
}

while (($#)); do
  case "$1" in
    --source-root) SOURCE_ROOT=${2:?}; shift 2 ;;
    --build-root) BUILD_ROOT=${2:?}; shift 2 ;;
    --wheel-dir) WHEEL_DIR=${2:?}; shift 2 ;;
    --python) PYTHON=${2:?}; shift 2 ;;
    *) usage; exit 2 ;;
  esac
done
[[ "$JOBS" =~ ^[12]$ ]] || { echo "G305_BUILD_JOBS must be 1 or 2" >&2; exit 2; }
# ExternalProject build steps launch their own build tools. This environment
# limit is inherited by those nested Ninja processes; --parallel below only
# constrains the top-level build.
export CMAKE_BUILD_PARALLEL_LEVEL="$JOBS"
command -v nvcc >/dev/null || { echo "CUDA 12.8 nvcc is required" >&2; exit 1; }
CMAKE=$(command -v cmake)
NVCC_VERSION=$(nvcc --version | sed -n 's/.*release \([0-9.]*\).*/\1/p' | tail -n1)
[[ "$NVCC_VERSION" == 12.8* ]] || {
  echo "Open3D Runtime requires CUDA compiler 12.8, found ${NVCC_VERSION:-unknown}" >&2
  exit 1
}

SOURCE_ROOT=$(readlink -m -- "$SOURCE_ROOT")
BUILD_ROOT=$(readlink -m -- "$BUILD_ROOT")
WHEEL_DIR=$(readlink -m -- "$WHEEL_DIR")
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CMAKE4_PATCH="$SCRIPT_DIR/patches/open3d-0.19-cmake4-policy.patch"

if [[ ! -d "$SOURCE_ROOT/.git" ]]; then
  [[ ! -e "$SOURCE_ROOT" ]] || {
    echo "Open3D source path exists but is not a Git checkout: $SOURCE_ROOT" >&2
    exit 1
  }
  git clone https://github.com/isl-org/Open3D.git "$SOURCE_ROOT"
fi
git -C "$SOURCE_ROOT" fetch --depth 1 origin "$OPEN3D_COMMIT"
git -C "$SOURCE_ROOT" checkout --detach "$OPEN3D_COMMIT"
# Restore only the known compatibility patch before checking cleanliness so an
# interrupted build can be resumed without accepting unrelated source edits.
if git -C "$SOURCE_ROOT" apply --reverse --check "$CMAKE4_PATCH" 2>/dev/null; then
  git -C "$SOURCE_ROOT" apply --reverse "$CMAKE4_PATCH"
fi
[[ -z "$(git -C "$SOURCE_ROOT" status --short)" ]] || {
  echo "Open3D source must be clean before applying compatibility patches" >&2
  exit 1
}

if git -C "$SOURCE_ROOT" apply --check "$CMAKE4_PATCH"; then
  git -C "$SOURCE_ROOT" apply "$CMAKE4_PATCH"
elif ! grep -q 'CMAKE_POLICY_VERSION_MINIMUM=3.5' \
  "$SOURCE_ROOT/3rdparty/find_dependencies.cmake" || \
  ! grep -q 'CMAKE_POLICY_VERSION_MINIMUM=3.5' \
  "$SOURCE_ROOT/3rdparty/vtk/CMakeLists.txt"; then
  echo "Open3D CMake 4 compatibility patch is neither applicable nor already present" >&2
  exit 1
fi

EXTRA_CMAKE_ARGS=()
if [[ "${G305_OPEN3D_APPLY_CCCL_PATCH:-0}" == 1 ]]; then
  CUDA13_PATCH="$SCRIPT_DIR/patches/open3d-0.19-cuda13-cccl.patch"
  if git -C "$SOURCE_ROOT" apply --check "$CUDA13_PATCH"; then
    git -C "$SOURCE_ROOT" apply "$CUDA13_PATCH"
  elif ! grep -q OPEN3D_THRUST_INCLUDE_DIR "$SOURCE_ROOT/3rdparty/stdgpu/stdgpu.cmake"; then
    echo "Open3D CUDA13/CCCL patch is neither applicable nor already present" >&2
    exit 1
  fi
  EXTRA_CMAKE_ARGS+=(
    -DOPEN3D_CUDA13_STDGPU_PATCH="$SCRIPT_DIR/patches/stdgpu-cuda13-device-properties.patch"
  )
elif [[ "${G305_OPEN3D_APPLY_CCCL_PATCH:-0}" != 0 ]]; then
  echo "G305_OPEN3D_APPLY_CCCL_PATCH must be 0 or 1" >&2
  exit 2
fi

if [[ -e "$BUILD_ROOT" && ! -f "$BUILD_ROOT/CMakeCache.txt" ]]; then
  echo "Refusing to overwrite an unknown Open3D build path: $BUILD_ROOT" >&2
  exit 1
fi
mkdir -p "$WHEEL_DIR"
"$CMAKE" -S "$SOURCE_ROOT" -B "$BUILD_ROOT" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DCMAKE_CUDA_COMPILER="$(command -v nvcc)" \
  -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DPython3_EXECUTABLE="$(readlink -f -- "$PYTHON")" \
  -DBUILD_CUDA_MODULE=ON \
  -DBUILD_PYTHON_MODULE=ON \
  -DBUILD_WITH_CUDA_STATIC=ON \
  -DBUILD_COMMON_CUDA_ARCHS=OFF \
  -DBUILD_GUI=OFF \
  -DBUILD_WEBRTC=OFF \
  -DBUILD_JUPYTER_EXTENSION=OFF \
  -DBUILD_EXAMPLES=OFF \
  -DBUILD_UNIT_TESTS=OFF \
  -DBUILD_BENCHMARKS=OFF \
  -DBUILD_ISPC_MODULE=OFF \
  -DBUILD_AZURE_KINECT=OFF \
  -DBUILD_LIBREALSENSE=OFF \
  -DBUILD_PYTORCH_OPS=OFF \
  -DBUILD_TENSORFLOW_OPS=OFF \
  -DBUNDLE_OPEN3D_ML=OFF \
  "${EXTRA_CMAKE_ARGS[@]}"
"$CMAKE" --build "$BUILD_ROOT" --target pip-package --parallel "$JOBS"

mapfile -t WHEELS < <(find "$BUILD_ROOT/lib" -type f -name 'open3d-*.whl' -print)
[[ ${#WHEELS[@]} -eq 1 ]] || {
  echo "Expected exactly one Open3D wheel, found ${#WHEELS[@]}" >&2
  exit 1
}
cp "${WHEELS[0]}" "$WHEEL_DIR/"
echo "$WHEEL_DIR/$(basename "${WHEELS[0]}")"
