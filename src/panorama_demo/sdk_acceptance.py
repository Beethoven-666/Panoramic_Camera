"""Read actual acceptance evidence. Capability reports alone cannot qualify a runtime."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import platform
import builtins
import subprocess
import sys
from contextlib import contextmanager

import numpy as np


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def native_host():
    return (
        platform.system() == "Linux"
        and "microsoft" not in platform.release().lower()
        and "wsl" not in platform.release().lower()
    )


def binding(source_commit, wheel, bundle_checksums, runtime_variant, platform_name):
    require(
        len(source_commit) == 40
        and all(c in "0123456789abcdef" for c in source_commit),
        "Invalid source commit",
    )
    return {
        "source_commit": source_commit,
        "project_wheel_sha256": hashlib.sha256(Path(wheel).read_bytes()).hexdigest(),
        "bundle_sha256": hashlib.sha256(
            Path(bundle_checksums).read_bytes()
        ).hexdigest(),
        "bundle_sha256_scope": "checksums.sha256 content",
        "runtime_variant": runtime_variant,
        "platform": platform_name,
    }


def verify_binding(evidence, expected):
    require(
        evidence.get("binding") == expected,
        "Acceptance source/wheel/bundle/variant/platform binding mismatch",
    )


def verify_doctor(path):
    report = read(path)
    checks = report["checks"]
    if isinstance(checks, dict):
        checks = list(checks.values())
    base = [c for c in checks if c.get("required_for_base")]
    require(
        bool(base) and all(c["status"] == "PASS" for c in base),
        "Doctor base capability failed or missing",
    )
    require(
        all(
            report.get(k) is None
            for k in ("software_ready", "hardware_qualified", "release_ready")
        ),
        "Doctor cannot issue qualification",
    )
    return report


def verify_2d(root, comparison_root=None, *, require_process_audit=True):
    from .video_s13_contract import (
        S13_VISUAL_CONTINUITY_ALGORITHM_ID,
        S13_VISUAL_CONTINUITY_IMPLEMENTATION_ID,
    )
    from .video_s13_live_acceptance import compare_s13_v11_live_offline

    root = Path(root)
    delivery, report, timing = (
        read(root / name)
        for name in ("video_delivery.json", "video_report.json", "video_timing.json")
    )
    algorithm = report["algorithm"]
    require(
        algorithm.get("algorithm_id") == S13_VISUAL_CONTINUITY_ALGORITHM_ID
        and algorithm.get("implementation_id")
        == S13_VISUAL_CONTINUITY_IMPLEMENTATION_ID,
        "Wrong production identity",
    )
    require(
        algorithm.get("role") == "production"
        and algorithm.get("fallback_used") is False
        and delivery.get("fallback_used") is False,
        "Fallback or non-production output",
    )
    require(
        algorithm.get("config_sha256")
        == "3a49f8cbaa622592eff41b2a9e34cb518c6067d74d15574cf01faeea0cc8fafc",
        "Wrong production config",
    )
    require(
        report["trajectory"].get("direct_pose_count") == 0
        and report["trajectory"].get("pose_supported") is False,
        "2-D consumed pose authority",
    )
    seconds = timing["final_2d"]["capture_stop_to_p3_published_seconds"]
    require(
        isinstance(seconds, (int, float)) and math.isfinite(seconds) and seconds >= 0,
        "Invalid actual P3 publish timing",
    )
    # Existing canonical comparison checks P0-P3, schedule, pair seams, M6,
    # PNG pixels and owner/valid/all normalized provenance arrays directly.
    comparison = compare_s13_v11_live_offline(
        root, Path(comparison_root) if comparison_root else root
    )
    require(comparison["equivalent"] is True, "Canonical SDK/CLI equivalence failed")
    if require_process_audit:
        audit = read(root / "process_audit.json")
        require(
            audit.get("schema") == "gemini305-2d-process-audit/v1"
            and audit.get("completed") is True,
            "Missing completed 2-D process audit",
        )
        require(
            all(
                audit.get(k) == 0
                for k in (
                    "orb_calls",
                    "open3d_imports",
                    "three_d_calls",
                    "online_orb_constructions",
                )
            ),
            "Heavy Runtime used before 2-D publication",
        )
    return {
        "status": "PASS",
        "capture_stop_to_p3_published_seconds": seconds,
        "comparison": comparison,
    }


def verify_3d(root, *, failure=False):
    root = Path(root)
    if failure:
        read(root / "video_3d_failure.json")
        evidence = read(root / "failure_isolation.json")
        require(evidence.get("status") == "PASS", "Missing failure isolation evidence")
        before = Path(evidence["before_2d"])
        after = Path(evidence["after_2d"])
        for name in (
            "video_delivery.json",
            "video_panorama.png",
            "video_pixel_provenance.npz",
        ):
            require(
                (before / name).read_bytes() == (after / name).read_bytes(),
                "3-D failure changed 2-D: " + name,
            )
        return {"status": "PASS", "failure_isolation": True}
    delivery = read(root / "video_3d_delivery.json")
    require(
        delivery.get("delivery_state") in ("published", "published_degraded"),
        "3-D not published",
    )
    trajectory = read(root / "orbslam3_trajectory.json")
    poses = np.asarray(trajectory["camera_to_world"], dtype=float)
    require(
        poses.ndim == 3
        and poses.shape[1:] == (4, 4)
        and len(poses) >= 2
        and np.isfinite(poses).all(),
        "Missing finite real ORB trajectory",
    )
    require(
        trajectory.get("pose_convention") == "camera_to_world",
        "Wrong ORB pose convention",
    )
    require(
        np.allclose(poses[:, 3, :], [0, 0, 0, 1])
        and np.allclose(np.linalg.det(poses[:, :3, :3]), 1, atol=1e-4),
        "Nonrigid ORB pose",
    )
    require(any(root.glob("*.glb")), "Missing 3-D GLB")
    return {"status": "PASS", "tracked_poses": len(poses)}


def verify_capture_contracts(paths):
    from .session import load_rgbd_session
    from .video_session import load_video_session

    require(
        set(paths) == {"fixed", "auto", "photo"},
        "Need fixed, auto and photo capture contracts",
    )
    for mode, path in paths.items():
        manifest = read(Path(path) / "manifest.json")
        require(manifest.get("clean_shutdown") is True, "Unclean " + mode + " capture")
        if mode == "photo":
            load_rgbd_session(Path(path))
            require(manifest.get("formal_stitch_allowed") is True, "Photo not eligible")
        else:
            load_video_session(Path(path), validate_frame_files=True)
            expected = (
                "continuous_rgbd_video_fixed_exposure"
                if mode == "fixed"
                else "continuous_rgbd_video_auto"
            )
            require(
                manifest.get("capture_mode") == expected,
                "Wrong " + mode + " capture contract",
            )
    return {"status": "PASS", "contracts": sorted(paths)}


@contextmanager
def audit_2d_process(output):
    """Acceptance-harness guard, installed before SDK/CLI entry and removed after publication."""
    report = {
        "schema": "gemini305-2d-process-audit/v1",
        "orb_calls": 0,
        "open3d_imports": 0,
        "three_d_calls": 0,
        "online_orb_constructions": 0,
        "completed": False,
    }
    require(
        not any(n == "open3d" or n.startswith("open3d.") for n in sys.modules),
        "Open3D already imported before audited 2-D",
    )
    original_import = builtins.__import__
    original_popen = subprocess.Popen

    def guarded_import(name, *args, **kwargs):
        if name == "open3d" or name.startswith("open3d."):
            report["open3d_imports"] += 1
            raise RuntimeError("Open3D import during capture/2-D")
        if name.endswith("video_online_orb"):
            report["online_orb_constructions"] += 1
            raise RuntimeError("Online ORB module during capture/2-D")
        return original_import(name, *args, **kwargs)

    def guarded_popen(command, *args, **kwargs):
        text = str(command).lower()
        if any(token in text for token in ("orbslam", "rgbd_tum", "rgbd_g305")):
            report["orb_calls"] += 1
            raise RuntimeError("ORB subprocess before 2-D publication")
        if any(token in text for token in ("video_3d", "post-3d")):
            report["three_d_calls"] += 1
            raise RuntimeError("3-D subprocess before 2-D publication")
        return original_popen(command, *args, **kwargs)

    builtins.__import__ = guarded_import
    subprocess.Popen = guarded_popen
    try:
        yield
        report["completed"] = True
    finally:
        builtins.__import__ = original_import
        subprocess.Popen = original_popen
        Path(output).mkdir(parents=True, exist_ok=True)
        with (Path(output) / "process_audit.json").open("x") as handle:
            json.dump(report, handle, indent=2)
