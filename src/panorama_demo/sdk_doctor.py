from __future__ import annotations

import json
from importlib import metadata
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

from .paths import PROJECT_ROOT, runtime_resource
from .video_s13_contract import (
    S13_VISUAL_CONTINUITY_ALGORITHM_ID,
    S13_VISUAL_CONTINUITY_IMPLEMENTATION_ID,
)


_VALID_STATUS = {"PASS", "FAIL", "NOT_CHECKED", "BLOCKED"}


@dataclass(frozen=True)
class SDKDoctorCheck:
    status: str
    reason_code: str
    detail: Mapping[str, object]
    required_for_base: bool
    required_for_addon: bool

    def __post_init__(self) -> None:
        if self.status not in _VALID_STATUS:
            raise ValueError(f"Invalid SDK doctor status: {self.status}")


@dataclass(frozen=True)
class SDKDoctorReport:
    schema: str
    checks: Mapping[str, SDKDoctorCheck]
    software_ready: None = None
    hardware_qualified: None = None
    release_ready: None = None
    milestone: None = None

    def __post_init__(self) -> None:
        if any(value is not None for value in (self.software_ready, self.hardware_qualified,
                                               self.release_ready, self.milestone)):
            raise ValueError("Doctor cannot issue qualifications; use acceptance artifacts")

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "checks": {name: asdict(value) for name, value in self.checks.items()},
            "software_ready": self.software_ready,
            "hardware_qualified": self.hardware_qualified,
            "release_ready": self.release_ready,
            "milestone": self.milestone,
            "qualification_source": "acceptance_artifact_only",
        }

    def __str__(self) -> str:
        return json.dumps(self.as_dict(), indent=2)


def _check(
    status: str,
    reason_code: str,
    detail: Mapping[str, object],
    *,
    software: bool,
    native: bool,
) -> SDKDoctorCheck:
    return SDKDoctorCheck(status, reason_code, dict(detail), software, native)


def run_sdk_doctor(
    *,
    orbslam3_root: Path,
    orbslam3_executable: str,
    orbslam3_vocabulary: str,
    orb_runtime_kind: str,
    camera_lock_root: Path = Path("/var/lock/gemini305-sdk"),
    probe_camera: bool = False,
    probe_addon: bool = True,
) -> SDKDoctorReport:
    checks: dict[str, SDKDoctorCheck] = {}
    supported = sys.version_info[:2] == (3, 10) and sys.implementation.name == "cpython"
    checks["python"] = _check(
        "PASS" if supported else "FAIL",
        "SUPPORTED_PYTHON" if supported else "UNSUPPORTED_PYTHON",
        {"version": sys.version.split()[0], "executable": sys.executable},
        software=True,
        native=True,
    )

    resources = (
        "configs/demo.yaml",
        "configs/video_algorithms/s013_visual_continuity_v11_production.yaml",
        "configs/video_algorithms/s013_visual_continuity_v11_production.lock.json",
        "configs/video_algorithms/baseline_legacy_fast_b07b561.yaml",
        "configs/video_algorithms/baseline_legacy_fast_b07b561.lock.json",
        "configs/video_candidates/s013/S013_output_first_progressive_dense_central_slit_v4_cuda_m63_visual_continuity_v11.yaml",
        "configs/video_candidates/s013/quality_thresholds_m61_v2.json",
        "artifacts/S013_M6_1_metrics_baseline_v2/threshold_approval.json",
    )
    try:
        resolved_resources = [str(runtime_resource(value)) for value in resources]
        resource_error = None
    except Exception as exc:
        resolved_resources = []
        resource_error = f"{type(exc).__name__}: {exc}"
    checks["package_resources"] = _check(
        "PASS" if resource_error is None else "FAIL",
        "RESOURCES_COMPLETE" if resource_error is None else "PACKAGE_RESOURCE_MISSING",
        {"root": str(PROJECT_ROOT), "paths": resolved_resources, "error": resource_error},
        software=True,
        native=True,
    )

    try:
        from .video_algorithm_registry import resolve_video_algorithm
        from .video_pipeline import _lock_paths

        _, baseline_lock, production_lock = _lock_paths(None)
        spec = resolve_video_algorithm(
            "production",
            baseline_lock=baseline_lock,
            production_lock=production_lock,
            candidate_config=None,
        )
        identity_ok = (
            spec.algorithm_id == S13_VISUAL_CONTINUITY_ALGORITHM_ID
            and spec.implementation_id == S13_VISUAL_CONTINUITY_IMPLEMENTATION_ID
            and spec.config_sha256
            == "3a49f8cbaa622592eff41b2a9e34cb518c6067d74d15574cf01faeea0cc8fafc"
            and spec.allow_baseline_fallback is False
        )
        identity_detail = {
            "algorithm_id": spec.algorithm_id,
            "implementation_id": spec.implementation_id,
            "config_sha256": spec.config_sha256,
            "allow_baseline_fallback": spec.allow_baseline_fallback,
        }
    except Exception as exc:
        identity_ok = False
        identity_detail = {"error": f"{type(exc).__name__}: {exc}"}
    checks["production_identity"] = _check(
        "PASS" if identity_ok else "FAIL",
        "V11_IDENTITY_EXACT" if identity_ok else "PRODUCTION_IDENTITY_INVALID",
        identity_detail,
        software=True,
        native=True,
    )

    try:
        import cupy as cp

        cupy_count = int(cp.cuda.runtime.getDeviceCount())
        if cupy_count:
            cp.cuda.Device(0).use()
            cp.zeros(1, dtype=cp.uint8).sum().get()
            probe = cp.eye(2, dtype=cp.float32)
            cublas_probe = float((probe @ probe).sum().get())
        properties = cp.cuda.runtime.getDeviceProperties(0) if cupy_count else {}
        architecture = int(properties.get("major", 0)) * 10 + int(properties.get("minor", 0))
        cupy_ok = cupy_count > 0 and architecture == 120
        cupy_detail: dict[str, object] = {
            "version": cp.__version__,
            "device_count": cupy_count,
            "runtime_version": int(cp.cuda.runtime.runtimeGetVersion()),
            "cublas_probe": cublas_probe if cupy_count else None,
            "gpu_compute_capability": architecture,
            "supported_variant": "sm_120",
        }
    except Exception as exc:
        cupy_ok = False
        cupy_detail = {"error": f"{type(exc).__name__}: {exc}"}
    checks["cupy_cuda"] = _check(
        "PASS" if cupy_ok else "FAIL",
        "CUPY_CUDA_AVAILABLE" if cupy_ok else "CUPY_CUDA_UNAVAILABLE",
        cupy_detail,
        software=True,
        native=True,
    )

    try:
        if not probe_addon:
            raise RuntimeError("Addon probe deferred while capture is active")
        from .sdk_runtime import addon_python

        interpreter = addon_python()
        if interpreter is None:
            raise RuntimeError("Open3D addon is not installed")
        completed = subprocess.run([str(interpreter), "-c",
            "import json,open3d as o; print(json.dumps(dict(version=o.__version__,"
            "build_cuda_module=bool(o._build_config.get('BUILD_CUDA_MODULE')),"
            "cuda_available=bool(o.core.cuda.is_available()),device_count=o.core.cuda.device_count())))"],
            capture_output=True, text=True, timeout=60, check=True)
        open3d_detail = json.loads(completed.stdout.strip().splitlines()[-1])
        open3d_detail["interpreter"] = str(interpreter)
        open3d_ok = (open3d_detail["version"] == "0.19.0+1e7b17438"
                      and open3d_detail["build_cuda_module"] and open3d_detail["cuda_available"])
    except Exception as exc:
        open3d_ok = False
        open3d_detail = {"error": f"{type(exc).__name__}: {exc}"}
    checks["open3d_cuda"] = _check(
        "PASS" if open3d_ok else "BLOCKED",
        "OPEN3D_CUDA_AVAILABLE" if open3d_ok else "THREE_D_ADDON_MISSING_OR_UNAVAILABLE",
        open3d_detail,
        software=False,
        native=True,
    )

    executable = (orbslam3_root / orbslam3_executable).expanduser().resolve()
    vocabulary = (orbslam3_root / orbslam3_vocabulary).expanduser().resolve()
    orb_ok = (
        orb_runtime_kind == "native_linux"
        and sys.platform.startswith("linux")
        and executable.is_file()
        and os.access(executable, os.X_OK)
        and vocabulary.is_file()
    )
    orb_detail: dict[str, object] = {
        "runtime_kind": orb_runtime_kind,
        "executable": str(executable),
        "vocabulary": str(vocabulary),
    }
    if orb_ok:
        completed = subprocess.run(
            ["ldd", str(executable)], capture_output=True, text=True, check=False
        )
        orb_ok = completed.returncode == 0 and "not found" not in completed.stdout
        orb_detail["ldd_returncode"] = completed.returncode
        orb_detail["ldd_missing"] = "not found" in completed.stdout
    checks["orbslam3_external_runtime"] = _check(
        "PASS" if orb_ok else "BLOCKED",
        "NATIVE_ORB_RUNTIME_AVAILABLE" if orb_ok else "THREE_D_RUNTIME_MISSING_OR_UNLICENSED",
        orb_detail,
        software=False,
        native=True,
    )

    try:
        import pyorbbecsdk as ob

        native_version = str(ob.get_version())
        wrapper_version = metadata.version("pyorbbecsdk2")
        wrapper_ok = wrapper_version == "2.1.2+g305.1" and native_version == "2.9.3"
        wrapper_detail: dict[str, object] = {"sdk_version": native_version,
            "wrapper_version": wrapper_version, "expected_native_sdk": "2.9.3",
            "upstream_wrapper": "2.1.2", "metadata_variant": "2.1.2+g305.1"}
    except Exception as exc:
        wrapper_ok = False
        wrapper_detail = {"error": f"{type(exc).__name__}: {exc}"}
        ob = None
    checks["orbbec_wrapper"] = _check(
        "PASS" if wrapper_ok else "FAIL",
        "ORBBEC_WRAPPER_AVAILABLE" if wrapper_ok else "ORBBEC_WRAPPER_MISSING",
        wrapper_detail,
        software=True,
        native=True,
    )

    camera_count = 0
    camera_error: str | None = None
    camera_info = []
    if ob is not None and probe_camera:
        try:
            from .camera_lease import CameraLease
            from .capture_orbbec import _device_info

            with CameraLease(camera_lock_root, "discovery", "doctor"):
                devices = ob.Context().query_devices()
                camera_count = int(devices.get_count())
                camera_info = [_device_info(devices.get_device_by_index(index)) for index in range(camera_count)]
        except Exception as exc:
            camera_error = f"{type(exc).__name__}: {exc}"
    checks["camera_device"] = _check(
        "PASS" if camera_count > 0 else "BLOCKED" if camera_error else "NOT_CHECKED",
        "CAMERA_DETECTED" if camera_count > 0 else "CAMERA_NOT_PROBED_OR_UNAVAILABLE",
        {"device_count": camera_count, "error": camera_error, "devices": camera_info,
         "supported_firmware": ["1.0.70"], "vid": 11205, "pid": 2112},
        software=False,
        native=True,
    )

    disk = shutil.disk_usage(PROJECT_ROOT)
    checks["host_disk_space"] = _check(
        "PASS" if disk.free >= 10 * 1024**3 else "FAIL",
        "DISK_SPACE_AVAILABLE" if disk.free >= 10 * 1024**3 else "HOST_DISK_SPACE",
        {"path": str(PROJECT_ROOT), "free_bytes": disk.free, "total_bytes": disk.total},
        software=True,
        native=True,
    )
    checks["orb_distribution_license"] = _check(
        "BLOCKED",
        "ORB_LICENSE",
        {"wheel_contains_orb": False, "commercial_distribution_approved": False},
        software=False,
        native=True,
    )
    lock_ok = camera_lock_root.is_dir() and os.access(camera_lock_root, os.W_OK)
    checks["camera_lock_root"] = _check("PASS" if lock_ok else "BLOCKED",
        "CAMERA_LOCK_ROOT_AVAILABLE" if lock_ok else "CAMERA_LOCK_ROOT_UNAVAILABLE",
        {"path": str(camera_lock_root), "required_for_capture": True}, software=False, native=False)
    return SDKDoctorReport(
        schema="gemini305-sdk-doctor/v2",
        checks=checks,
    )


def main() -> None:
    import argparse
    from contextlib import redirect_stdout
    import io

    parser = argparse.ArgumentParser(description="Report capabilities; never issue SDK qualification")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--orb-runtime-root", type=Path, default=Path.home() / "opt/g305-orbslam3")
    parser.add_argument("--camera-lock-root", type=Path, default=Path("/var/lock/gemini305-sdk"))
    parser.add_argument("--probe-camera", action="store_true")
    args = parser.parse_args()
    with redirect_stdout(io.StringIO()):
        report = run_sdk_doctor(orbslam3_root=args.orb_runtime_root,
            orbslam3_executable="Examples/RGB-D/rgbd_tum_headless", orbslam3_vocabulary="Vocabulary/ORBvoc.txt",
            orb_runtime_kind="native_linux", camera_lock_root=args.camera_lock_root, probe_camera=args.probe_camera)
    payload = str(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    if any(check.status != "PASS" for check in report.checks.values() if check.required_for_base):
        raise SystemExit(1)


if __name__ == "__main__":
    main()


__all__ = ["SDKDoctorCheck", "SDKDoctorReport", "run_sdk_doctor"]
