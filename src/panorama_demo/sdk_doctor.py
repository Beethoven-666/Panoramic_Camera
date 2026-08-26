from __future__ import annotations

import json
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


_VALID_STATUS = {"PASS", "PASS_WITH_WARNING", "FAIL", "NOT_EXECUTED", "BLOCKED"}


@dataclass(frozen=True)
class SDKDoctorCheck:
    status: str
    reason_code: str
    detail: Mapping[str, object]
    required_for_software_ready: bool
    required_for_native_final: bool

    def __post_init__(self) -> None:
        if self.status not in _VALID_STATUS:
            raise ValueError(f"Invalid SDK doctor status: {self.status}")


@dataclass(frozen=True)
class SDKDoctorReport:
    schema: str
    checks: Mapping[str, SDKDoctorCheck]
    software_ready: bool
    hardware_qualified: bool
    release_ready: bool
    milestone: str | None

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "checks": {name: asdict(value) for name, value in self.checks.items()},
            "software_ready": self.software_ready,
            "hardware_qualified": self.hardware_qualified,
            "release_ready": self.release_ready,
            "milestone": self.milestone,
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
) -> SDKDoctorReport:
    checks: dict[str, SDKDoctorCheck] = {}
    supported = (3, 10) <= sys.version_info[:2] < (3, 13)
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
        cupy_ok = cupy_count > 0
        cupy_detail: dict[str, object] = {
            "version": cp.__version__,
            "device_count": cupy_count,
            "runtime_version": int(cp.cuda.runtime.runtimeGetVersion()),
            "cublas_probe": cublas_probe if cupy_count else None,
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
        import open3d as o3d

        build = dict(getattr(o3d, "_build_config", {}))
        open3d_ok = bool(build.get("BUILD_CUDA_MODULE")) and bool(
            o3d.core.cuda.is_available()
        )
        open3d_detail: dict[str, object] = {
            "version": o3d.__version__,
            "build_cuda_module": bool(build.get("BUILD_CUDA_MODULE")),
            "cuda_available": bool(o3d.core.cuda.is_available()),
            "device_count": int(o3d.core.cuda.device_count()),
        }
    except Exception as exc:
        open3d_ok = False
        open3d_detail = {"error": f"{type(exc).__name__}: {exc}"}
    checks["open3d_cuda"] = _check(
        "PASS" if open3d_ok else "FAIL",
        "OPEN3D_CUDA_AVAILABLE" if open3d_ok else "OPEN3D_CUDA_UNAVAILABLE",
        open3d_detail,
        software=True,
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
        "PASS" if orb_ok else "FAIL",
        "NATIVE_ORB_RUNTIME_READY" if orb_ok else "NATIVE_ORB_RUNTIME_MISSING",
        orb_detail,
        software=True,
        native=True,
    )

    try:
        import pyorbbecsdk as ob

        wrapper_ok = True
        wrapper_detail: dict[str, object] = {"sdk_version": str(ob.get_version())}
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
    if ob is not None:
        try:
            camera_count = int(ob.Context().query_devices().get_count())
        except Exception as exc:
            camera_error = f"{type(exc).__name__}: {exc}"
    checks["camera_device"] = _check(
        "PASS" if camera_count > 0 else "NOT_EXECUTED",
        "CAMERA_DETECTED" if camera_count > 0 else "NO_CAMERA",
        {"device_count": camera_count, "error": camera_error},
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

    software_ready = all(
        value.status in {"PASS", "PASS_WITH_WARNING"}
        for value in checks.values()
        if value.required_for_software_ready
    )
    hardware_qualified = software_ready and all(
        value.status in {"PASS", "PASS_WITH_WARNING"}
        for value in checks.values()
        if value.required_for_native_final
        and value is not checks["orb_distribution_license"]
    )
    return SDKDoctorReport(
        schema="gemini305-sdk-doctor/v1",
        checks=checks,
        software_ready=software_ready,
        hardware_qualified=hardware_qualified,
        release_ready=hardware_qualified
        and checks["orb_distribution_license"].status == "PASS",
        milestone=(
            "SDK_HARDWARE_QUALIFIED"
            if hardware_qualified
            else "SDK_SOFTWARE_READY"
            if software_ready
            else None
        ),
    )


__all__ = ["SDKDoctorCheck", "SDKDoctorReport", "run_sdk_doctor"]
