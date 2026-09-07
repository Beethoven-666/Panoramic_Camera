"""Read-only native H0 host probes; camera access is gated by native replay."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time


IDENTITY_FIELDS = ("h0_id", "subject_source_commit", "h0_tool_commit")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def bound_errors(value, binding, label):
    return [f"{label}: missing/mismatched {key}" for key in IDENTITY_FIELDS
            if not binding.get(key) or value.get(key) != binding[key]]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def command(args):
    try:
        result = subprocess.run(args, text=True, capture_output=True, timeout=20, check=False)
        return {"argv": args, "returncode": result.returncode,
                "stdout": result.stdout, "stderr": result.stderr}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"argv": args, "status": "NOT_EXECUTED", "reason": str(exc)}


def _text(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def collect_host():
    release = {}
    for line in (_text("/etc/os-release") or "").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            release[key] = value.strip('"')
    return {"system": platform.system(), "os_release": release,
            "machine_id": _text("/etc/machine-id"), "hostname": platform.node(),
            "architecture": platform.machine(), "python_implementation": platform.python_implementation(),
            "python_version": list(sys.version_info[:3]), "python_executable": sys.executable,
            "effective_uid": os.geteuid() if hasattr(os, "geteuid") else None,
            "glibc": command(["getconf", "GNU_LIBC_VERSION"]),
            "virtualization": command(["systemd-detect-virt"]),
            "proc_version": _text("/proc/version"), "kernel": platform.release(),
            "container_markers": [p for p in ("/.dockerenv", "/run/.containerenv") if Path(p).exists()],
            "pythonpath": os.environ.get("PYTHONPATH"), "g305_cuda": os.environ.get("G305_CUDA")}


def collect_gpu():
    result = {"nvidia_smi": command(["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total,temperature.gpu,compute_cap", "--format=csv,noheader,nounits"])}
    try:
        import cupy as cp
        # A RawKernel proves execution on the selected device, not merely import.
        out = cp.empty(8, dtype=cp.int32)
        kernel = cp.RawKernel('extern "C" __global__ void h0(int* x) { int i=threadIdx.x; x[i]=i*3+7; }', "h0")
        kernel((1,), (8,), (out,))
        cp.cuda.runtime.deviceSynchronize()
        props = cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())
        result["cupy"] = {"version": cp.__version__, "runtime_version": cp.cuda.runtime.runtimeGetVersion(),
                           "device_index": cp.cuda.runtime.getDevice(), "major": props["major"], "minor": props["minor"],
                           "kernel_output": cp.asnumpy(out).tolist()}
    except Exception as exc:
        result["cupy"] = {"status": "NOT_EXECUTED", "reason": f"{type(exc).__name__}: {exc}"}
    try:
        from panorama_demo.sdk_runtime import addon_python
        interpreter = addon_python()
        if interpreter is None:
            raise RuntimeError("Installed offline Open3D addon interpreter is missing")
        code = """import json, sys
import open3d as o3d
t = o3d.core.Tensor([1., 2.], device=o3d.core.Device('CUDA:0'))
print(json.dumps({'version': o3d.__version__, 'build_cuda_module': o3d._build_config.get('BUILD_CUDA_MODULE'),
 'cuda_available': o3d.core.cuda.is_available(), 'device_count': o3d.core.cuda.device_count(),
 'cuda_result': (t+t).cpu().numpy().tolist(), 'addon_python': sys.executable}))
"""
        probe = command([str(interpreter), "-c", code])
        if probe.get("returncode") != 0:
            raise RuntimeError(f"Installed Open3D addon CUDA probe failed: {probe}")
        result["open3d"] = json.loads(probe["stdout"].splitlines()[-1])
        result["open3d"]["probe_command"] = probe
    except Exception as exc:
        result.setdefault("open3d", {}).update(status="NOT_EXECUTED", reason=f"{type(exc).__name__}: {exc}")
    return result


def collect_storage(storage_root):
    path = Path(storage_root).resolve()
    disk = shutil.disk_usage(path)
    return {"path": str(path), "free_bytes": disk.free, "total_bytes": disk.total,
            "findmnt": command(["findmnt", "-J", "-T", str(path), "-o", "SOURCE,TARGET,FSTYPE,OPTIONS"])}


def collect_camera(lock_root, *, replay_verified):
    if not replay_verified:
        return {"status": "NOT_EXECUTED", "reason": "Native recorded replay must pass before camera probe"}
    result = {"probe_camera": True, "lock_root": str(lock_root)}
    try:
        import pyorbbecsdk as ob
        from panorama_demo.camera_lease import CameraLease
        from panorama_demo.capture_orbbec import _device_info
        from panorama_demo.photo_capture import select_fastest_rgbd_profiles
        result.update(wrapper_version=importlib.metadata.version("pyorbbecsdk2"), native_sdk_version=str(ob.get_version()))
        with CameraLease(Path(lock_root), "discovery", "native-h0-inventory"):
            context = ob.Context()
            devices = context.query_devices()
            result["device_count"] = int(devices.get_count())
            if result["device_count"] != 1:
                raise RuntimeError("Native H0 requires exactly one connected camera")
            device = devices.get_device_by_index(0)
            result["device"] = _device_info(device)
            pipeline = ob.Pipeline(device)
            color, depth = select_fastest_rgbd_profiles(
                pipeline.get_stream_profile_list(ob.OBSensorType.COLOR_SENSOR),
                pipeline.get_stream_profile_list(ob.OBSensorType.DEPTH_SENSOR),
                width=848, height=480, requested_fps=60, sdk=ob,
                color_formats=("RGB", "BGR", "YUYV", "MJPG"),
                trigger_out_delay_us=7000, trigger_to_image_delay_us=8000)
            result["profiles"] = [{"width": p.get_width(), "height": p.get_height(), "fps": p.get_fps(),
                                   "format": str(p.get_format()).split(".")[-1]} for p in (color, depth)]
            config = device.get_multi_device_sync_config()
            result["sync_readback"] = {"sync_mode": str(config.mode), "trigger_out_enable": bool(config.trigger_out_enable),
                                       "trigger_out_delay_us": int(config.trigger_out_delay_us),
                                       "trigger_to_image_delay_us": int(config.trigger_to_image_delay_us)}
            result["ordinary_user_open_read"] = True
        result["lock_root_writable"] = Path(lock_root).is_dir() and os.access(lock_root, os.W_OK)
        result["lock_root_mode"] = oct(Path(lock_root).stat().st_mode & 0o777)
    except Exception as exc:
        result.update(status="NOT_EXECUTED", reason=f"{type(exc).__name__}: {exc}")
    return result


def collect_usb(camera):
    serial = camera.get("device", {}).get("serial_number")
    devices = []
    for path in Path("/sys/bus/usb/devices").glob("*"):
        if _text(path / "idVendor") == "2bc5" and _text(path / "idProduct") == "0840":
            devices.append({"port_path": path.name, "sysfs_realpath": str(path.resolve()),
                            **{key: _text(path / key) for key in ("serial", "speed", "busnum", "devnum", "bMaxPower", "manufacturer", "product")}})
    return {"devices": devices, "camera_serial": serial, "usb_tree": command(["lsusb", "-t"]),
            "matched_ports": [item["port_path"] for item in devices if serial and item["serial"] == serial]}


def collect_inventory(output_root, storage_root, lock_root, *, binding=None, replay_verified=False, udev_rule=None):
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    identity = {key: (binding or {}).get(key) for key in IDENTITY_FIELDS}
    host = collect_host()
    native = (host["system"] == "Linux" and host["os_release"].get("ID") == "ubuntu"
              and host["os_release"].get("VERSION_ID") == "22.04" and host["architecture"] == "x86_64"
              and host["effective_uid"] not in {None, 0} and not host["container_markers"]
              and host["virtualization"].get("stdout", "").strip() == "none"
              and "microsoft" not in str(host["proc_version"]).lower())
    camera = collect_camera(lock_root, replay_verified=replay_verified and native)
    if udev_rule:
        path = Path(udev_rule)
        camera["udev_rule"] = {"path": str(path), "sha256": sha256(path), "mode": oct(path.stat().st_mode & 0o777)}
    records = {"host": host, "gpu": collect_gpu(), "storage": collect_storage(storage_root),
               "camera": camera, "usb": collect_usb(camera)}
    for name, value in records.items():
        value.update(identity, schema=f"gemini305-native-h0-{name}/v1", collected_unix_seconds=time.time())
        with (output / f"{name}_inventory.json").open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2)
    return records


def inventory_errors(records, binding):
    errors = []
    def require(condition, reason):
        if not condition:
            errors.append(reason)
    for name in ("host", "gpu", "storage", "camera", "usb"):
        errors.extend(bound_errors(records.get(name, {}), binding, name))
    for name in ("host", "camera"):
        if binding.get(name) is not None and records.get(name) != binding[name]:
            errors.append(f"{name}: inventory differs from bound raw inventory")
    host = records.get("host", {})
    require(host.get("system") == "Linux" and host.get("os_release", {}).get("ID") == "ubuntu"
            and host.get("os_release", {}).get("VERSION_ID") == "22.04", "Host must be native Ubuntu 22.04")
    require(host.get("architecture") == "x86_64", "Host must be x86_64")
    require(host.get("python_implementation") == "CPython" and host.get("python_version", [])[:2] == [3, 10], "CPython 3.10 required")
    require(isinstance(host.get("effective_uid"), int) and host["effective_uid"] > 0, "Non-root required")
    virt = host.get("virtualization", {})
    require(virt.get("stdout", "").strip() == "none" and virt.get("returncode") == 1
            and not host.get("container_markers") and "microsoft" not in (str(host.get("proc_version")) + str(host.get("kernel"))).lower(), "Bare metal required; virtualization unknown or detected")
    try:
        version = host["glibc"]["stdout"].strip().split()[-1]
        require(tuple(map(int, version.split(".")[:2])) >= (2, 35), "glibc >=2.35 required")
    except (KeyError, ValueError, IndexError):
        errors.append("glibc not measured")
    require(not host.get("pythonpath"), "PYTHONPATH prohibited")
    require(host.get("g305_cuda") == "required", "G305_CUDA=required required")
    gpu = records.get("gpu", {})
    cupy, o3d = gpu.get("cupy", {}), gpu.get("open3d", {})
    require(gpu.get("nvidia_smi", {}).get("returncode") == 0, "NVIDIA driver unavailable")
    require((cupy.get("major"), cupy.get("minor")) == (12, 0) and cupy.get("kernel_output") == [i * 3 + 7 for i in range(8)], "sm_120 actual CuPy kernel required")
    require(o3d.get("version") == "0.19.0+1e7b17438" and o3d.get("build_cuda_module") is True
            and o3d.get("cuda_available") is True and o3d.get("device_count", 0) >= 1
            and o3d.get("cuda_result") == [2., 4.], "Private Open3D actual CUDA required")
    expected_addon = binding.get("installed_open3d", {}).get("addon_python")
    if expected_addon:
        require(o3d.get("addon_python") == expected_addon, "Open3D probe interpreter differs from bound addon")
    storage = records.get("storage", {})
    try:
        fs = json.loads(storage["findmnt"]["stdout"])["filesystems"][0]
        require(fs["fstype"] in {"ext4", "xfs"} and str(fs["source"]).startswith("/dev/"), "Local ext4/xfs required")
    except (KeyError, IndexError, TypeError, ValueError):
        errors.append("Local filesystem not measured")
    require(storage.get("free_bytes", 0) >= 30 * 1024**3, "At least 30 GiB free required")
    camera = records.get("camera", {})
    device = camera.get("device", {})
    require(camera.get("probe_camera") is True and camera.get("device_count") == 1, "Exactly one probed camera required")
    require(device.get("name") in {"Gemini 305", "Orbbec Gemini 305"} and device.get("vid") == 0x2BC5
            and device.get("pid") == 0x0840 and device.get("firmware_version") == "1.0.70", "Gemini 305 firmware/VID/PID mismatch")
    require(bool(device.get("serial_number")) and device.get("serial_number") == binding.get("camera_serial"), "Camera serial binding mismatch")
    require(camera.get("wrapper_version") == "2.1.2+g305.1" and camera.get("native_sdk_version") == "2.9.3", "Camera SDK version mismatch")
    require(camera.get("ordinary_user_open_read") is True and camera.get("lock_root_writable") is True, "Camera nonroot/lock access missing")
    require(bool(camera.get("sync_readback")), "Trigger Out property readback required")
    profiles = camera.get("profiles", [])
    require(len(profiles) == 2 and all((p.get("width"), p.get("height"), p.get("fps")) == (848, 480, 60) for p in profiles)
            and profiles[1].get("format") == "Y16", "Exact 848x480 RGB-D @60 profile required")
    rule = camera.get("udev_rule", {})
    require(bool(rule.get("sha256")) and rule.get("sha256") == binding.get("udev_rule_sha256"), "udev rule not bound to candidate")
    usb = records.get("usb", {})
    matched = [d for d in usb.get("devices", []) if d.get("serial") == binding.get("camera_serial")]
    try:
        require(len(matched) == 1 and float(matched[0]["speed"]) >= 5000
                and matched[0]["port_path"] == binding.get("usb_port_path"), "USB3 actual speed/port binding required")
    except (ValueError, TypeError, KeyError, IndexError):
        errors.append("USB actual speed not measured")
    return errors


def validate_inventory(root, binding):
    records, errors = {}, []
    for name in ("host", "gpu", "storage", "camera", "usb"):
        path = Path(root) / "preflight" / f"{name}_inventory.json"
        try:
            records[name] = read_json(path)
        except (OSError, ValueError) as exc:
            errors.append(f"{name}: {exc}")
    errors.extend(inventory_errors(records, binding))
    return {"errors": errors, "evidence": records}
