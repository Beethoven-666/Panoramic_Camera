# Linux native Runtime and CLI

Linux L0 targets Ubuntu 22.04 and executes ORB-SLAM3 as a native ELF process. It does not call
`wsl.exe`, translate `/mnt/*` paths, or fall back to the legacy Windows-to-WSL bridge.

## Build boundaries

For the 0.3.0rc1 offline base/addon bundles use [INSTALL_LINUX.md](INSTALL_LINUX.md).
The following source build helpers are developer tools, not the offline installation entry.
The controlled variant is Ubuntu 22.04 x86_64, CPython 3.10 and sm_120.

The pinned build entry points are:

```bash
bash scripts/build_orbslam3_linux.sh
bash scripts/install_linux_python_runtime.sh
bash scripts/build_open3d_cuda_linux.sh
sudo bash packaging/linux/setup_orbbec_udev.sh   # user explicitly installs official rules
```

Set `G305_BUILD_JOBS=1` or `2` for Open3D. The build script also propagates that limit to nested
third-party Ninja builds through `CMAKE_BUILD_PARALLEL_LEVEL`.

The ORB build is pinned to commit `4452a3c4ab75b1cde34e5505a36ec3f9edcdc4c4`; the headless
runner patch is GPL-3.0-or-later and the generated external Runtime is not bundled in the Python
wheel. Open3D is pinned to `0.19` source commit `1e7b17438687a0b0c1e5a7187321ac7044afe275`, CUDA 12.8,
and architecture 120 for the RTX 5060 L1 machine. Other architectures require a separate build
and validation; changing a flag does not qualify another GPU.

The Open3D build also needs the GLFW/X11 development headers even with the GUI disabled:

```bash
sudo apt-get install -y --no-install-recommends \
  libxcursor-dev libxinerama-dev libxi-dev libxrandr-dev
```

`setup_orbbec_udev.sh` invokes verbatim official v2.9.3 rules from commit
`2f6561c28255d805b34aa00a690199ce40e96c81`, matching native SDK 2.9.3.
The ordinary installer never invokes this script or sudo. Do not run capture as root.

## Runtime configuration

Machine paths belong in a site YAML or `VideoSDKConfig`, never in immutable production YAML/lock:

```yaml
stitch:
  orbslam3_rgbd:
    runtime_kind: native_linux
    root: /home/zyh/opt/g305-orbslam3
    executable: Examples/RGB-D/rgbd_tum_headless
    stream_executable: Examples/RGB-D/rgbd_g305_stream_headless
    vocabulary: Vocabulary/ORBvoc.txt
```

Run the required CUDA smoke before product validation:

```bash
export G305_CUDA=required
python scripts/verify_open3d_cuda.py
python -c 'from panorama_demo import Gemini305VideoSDK; print(Gemini305VideoSDK().doctor())'
```

## Five public CLI entries

```bash
g305-capture --help
g305-video-live --help
g305-video-panorama --help
g305-video-post-3d --help
g305-orbslam3-trajectory --help
```

`g305-video-panorama` and SDK 2-D use locked S013 V11 with `G305_CUDA=required`; they never run ORB
or Open3D before 2-D publication. `g305-video-post-3d` runs native ORB and TSDF afterward. A 3-D
failure writes `3d/video_3d_failure.json` and cannot revoke the existing `video_delivery.json`.

Before the first committed frame, live capture can wait for camera hot-plug. After the first
commit, disconnect stops that session; reconnect cannot append to it. Use
`--no-wait-for-camera` for immediate failure. Ctrl+C and the SDK job cancellation event stop the
wait/capture path cooperatively.

## What validation claims mean

Recorded-session replay proves software correctness, pixel equivalence, native ORB and post-3D on
that recording. It is not a new physical-camera run. Native non-root Gemini 305 capture, udev,
metadata/sync readback and five physical SDK runs remain `NOT_EXECUTED/WAITING_FOR_H0` until H0 is
performed on an Ubuntu machine with the camera attached.

Doctor reports capabilities only; compatibility readiness fields are null. Only the acceptance
aggregator can issue source/wheel/bundle/variant/platform-bound status artifacts after checking
the actual raw runs. Phase 2 does not execute native H0 or issue any readiness state.
