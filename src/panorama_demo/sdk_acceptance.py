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


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def binding(source_commit, wheel, bundle_archive, runtime_variant, platform_name):
    require(
        len(source_commit) == 40
        and all(c in "0123456789abcdef" for c in source_commit),
        "Invalid source commit",
    )
    return {
        "source_commit": source_commit,
        "project_wheel_sha256": sha256(wheel),
        "base_archive_sha256": sha256(bundle_archive),
        "runtime_variant": runtime_variant,
        "platform": platform_name,
    }


def verify_binding(evidence, expected):
    require(
        evidence.get("binding", evidence) == expected,
        "Acceptance source/wheel/bundle/variant/platform binding mismatch",
    )


def verify_doctor(path, profile="base"):
    report = read(path)
    require(profile in {"base", "full_software", "native_h0"}, "Unknown doctor profile")
    named_checks = report["checks"]
    checks = named_checks
    if isinstance(checks, dict):
        checks = list(checks.values())
    base = [c for c in checks if c.get("required_for_base")]
    require(
        bool(base) and all(c["status"] == "PASS" for c in base),
        "Doctor base capability failed or missing",
    )
    if profile in {"full_software", "native_h0"}:
        require(isinstance(named_checks, dict), "Full doctor needs named checks")
        for name in ("open3d_cuda", "orbslam3_external_runtime", "orbbec_wrapper"):
            require(named_checks.get(name, {}).get("status") == "PASS",
                    "Doctor full software capability failed: " + name)
    if profile == "native_h0":
        require(native_host(), "WSL cannot issue native H0")
        require(named_checks.get("camera_device", {}).get("status") == "PASS",
                "Native H0 requires a camera")
    require(
        all(
            report.get(k) is None
            for k in ("software_ready", "hardware_qualified", "release_ready")
        ),
        "Doctor cannot issue qualification",
    )
    return report


def verify_2d_contract(root, expected_binding=None, *, require_process_audit=True):
    from .video_s13_contract import (
        S13_VISUAL_CONTINUITY_ALGORITHM_ID,
        S13_VISUAL_CONTINUITY_IMPLEMENTATION_ID,
    )
    from .video_s13_live_acceptance import (
        _load_production, _m6_contract, _pair_decision_contract, _valid_sha256,
        _STAGE_FILES,
    )

    root = Path(root)
    product = _load_production(root)
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
        isinstance(seconds, (int, float)) and math.isfinite(seconds) and 0 <= seconds <= 60,
        "Invalid actual P3 publish timing or 60 second SLA exceeded",
    )
    require(all(_valid_sha256(report.get("stage_pixel_sha256", {}).get(stage))
                for stage in ("P0", "P1", "P2", "P3")), "Missing P0-P3 pixel evidence")
    sources = report.get("source_frame_ids")
    require(isinstance(sources, list) and len(sources) >= 2 and len(set(sources)) == len(sources),
            "Missing source frame IDs")
    require(set(np.unique(product["owner"][product["owner"] >= 0])).issubset(set(sources)),
            "Owner map contains a nonexistent source")
    schedule = report.get("schedule", {})
    require(all(name in schedule for name in ("canvas_shape", "boundaries", "assignments")),
            "Missing schedule evidence")
    pairs = _pair_decision_contract(report.get("pair_seam_decisions"))
    require(len(pairs) == len(sources) - 1, "Incomplete pair seam evidence")
    _m6_contract(report)
    policy = report.get("production_output_policy", {})
    require(policy.get("stage_output_stages") == []
            and policy.get("p3_stage_png_encoded") is False
            and policy.get("formal_publication_encodes_p3_once") is True
            and all(not (root / name).exists() for name in _STAGE_FILES),
            "Invalid formal stage output policy")
    inputs = report.get("input_sha256", {})
    require(set(inputs) == {"manifest", "calibration", "frames_csv"}
            and all(_valid_sha256(v) for v in inputs.values()), "Missing input identity")
    if expected_binding is not None:
        require(inputs == expected_binding["frozen_session"]["input_sha256"],
                "Frozen session input binding mismatch")
    reuse = report.get("live_handoff", report.get("frozen_p0_reuse", {}))
    if reuse.get("p0_reused"):
        require(reuse.get("full_m0_m3_recomputed") is False,
                "P0 reuse contradicts full recomputation")
        require(reuse.get("reuse_level") == "frozen_p0_authority_v1"
                and reuse.get("preview_pixels_directly_published") is False,
                "Invalid frozen P0 authority")
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
    }


def verify_2d_equivalence(root, comparison_root, expected_binding=None):
    from .video_s13_live_acceptance import compare_s13_v11_live_offline

    require(Path(root).resolve() != Path(comparison_root).resolve(),
            "SDK/CLI self-comparison is forbidden")
    verify_2d_contract(root, expected_binding)
    verify_2d_contract(comparison_root, expected_binding)
    comparison = compare_s13_v11_live_offline(Path(root), Path(comparison_root))
    require(comparison["equivalent"] is True, "Canonical SDK/CLI equivalence failed")
    return comparison


def verify_2d(root, comparison_root=None, *, require_process_audit=True):
    result = verify_2d_contract(root, require_process_audit=require_process_audit)
    if comparison_root is not None:
        result["comparison"] = verify_2d_equivalence(root, comparison_root)
    return result


def verify_orb_runtime(identity, runtime_root=None):
    """Check the relocated native binaries, not the pre-patchelf build hashes."""
    manifest = Path(identity["manifest_path"])
    require(sha256(manifest) == identity["manifest_sha256"], "ORB Runtime manifest changed")
    portable = Path(identity["portability_manifest_path"])
    require(sha256(portable) == identity["portability_manifest_sha256"],
            "ORB portability manifest changed")
    original, relocation = read(manifest), read(portable)
    require(original["orbslam3_commit"] == relocation["orbslam3_commit"]
            == identity["orbslam3_commit"], "ORB source commit mismatch")
    runtime_root = Path(runtime_root or identity["root"]).expanduser().resolve()
    files = dict(relocation["files"])
    files["Vocabulary/ORBvoc.txt"] = original["artifacts"]["Vocabulary/ORBvoc.txt"]
    require(files == identity["files"], "ORB bound file inventory mismatch")
    for name, record in files.items():
        require(sha256(runtime_root / name) == record["sha256"], "ORB Runtime file changed: " + name)
    executable = runtime_root / "Examples/RGB-D/rgbd_tum_headless"
    require(executable.read_bytes()[:4] == b"\x7fELF", "ORB executable is not native ELF")
    if sys.platform.startswith("linux"):
        completed = subprocess.run(["ldd", str(executable)], capture_output=True, text=True, check=False)
        require(completed.returncode == 0 and "not found" not in completed.stdout,
                "ORB dependencies unresolved")
        for line in completed.stdout.splitlines():
            if "=> /" not in line:
                continue
            dependency = Path(line.split("=>", 1)[1].strip().split()[0]).resolve()
            require("build" not in dependency.parts and "/tmp/" not in str(dependency),
                    "ORB dependency resolves to a build directory")
            if dependency.name.startswith(("libORB", "libpango_", "libg2o", "libDBoW")):
                require(dependency.is_relative_to(runtime_root),
                        "ORB private dependency escaped relocated Runtime")
    return {"status": "PASS", "orbslam3_commit": identity["orbslam3_commit"],
            "manifest_sha256": identity["manifest_sha256"]}


def verify_native_orb(root, expected_binding=None):
    """Validate each fresh native execution; cross-run pose equality is not a gate."""
    root = Path(root)
    trajectory = read(root / "orbslam3_trajectory.json")
    require(trajectory.get("backend") == "orbslam3_rgbd_native_linux"
            and trajectory.get("config", {}).get("runtime_kind") == "native_linux",
            "Missing real native ORB backend")
    require(trajectory.get("pose_convention") == "camera_to_world"
            and trajectory.get("translation_unit") == "mm", "Wrong ORB pose convention")
    attempts = trajectory.get("execution_attempts", [])
    require(attempts and attempts[-1].get("accepted") is True
            and attempts[-1].get("returncode") == 0, "No accepted fresh ORB execution")
    ids = trajectory["tracked_frame_ids"]
    records = trajectory.get("poses")
    if records is not None:
        require([r["frame_id"] for r in records] == ids, "ORB pose/frame mapping differs")
        require(all(r.get("pose_kind") == "direct_orbslam3" and r.get("pose_status") == "valid"
                    for r in records), "Synthetic or missing ORB pose")
        poses = np.asarray([r["camera_to_world"] for r in records], dtype=float)
        timestamps = np.asarray([r["timestamp_us"] for r in records], dtype=float)
    else:
        poses = np.asarray(trajectory["camera_to_world"], dtype=float)
        timestamps = np.asarray(trajectory["timestamps_us"], dtype=float)
        require(trajectory.get("interpolated_pose_count") == 0
                and trajectory.get("extrapolated_pose_count") == 0, "Interpolated ORB chain")
    require(len(ids) >= 2 and ids == sorted(set(ids)) and len(poses) == len(ids),
            "Invalid ORB frame IDs")
    require(timestamps.shape == (len(ids),) and np.isfinite(timestamps).all()
            and (timestamps >= 0).all() and (np.diff(timestamps) > 0).all(),
            "Invalid ORB timestamp mapping")
    require(poses.shape == (len(ids), 4, 4) and np.isfinite(poses).all(),
            "Missing finite real ORB trajectory")
    rotations = poses[:, :3, :3]
    require(np.allclose(poses[:, 3, :], [0, 0, 0, 1], atol=1e-8)
            and np.allclose(np.linalg.det(rotations), 1, atol=1e-4)
            and np.allclose(rotations @ rotations.transpose(0, 2, 1), np.eye(3), atol=1e-4),
            "Nonrigid ORB pose")
    count = trajectory["input_frame_count"]
    require(count >= len(ids) and trajectory["tracked_frame_count"] == len(ids)
            and math.isclose(trajectory["tracked_fraction"], len(ids) / count)
            and len(ids) / count >= trajectory["config"]["minimum_tracked_fraction"],
            "ORB tracked fraction below configured gate")
    for name in ("stdout_file", "stderr_file", "tum_file", "association_file"):
        require((root / trajectory[name]).is_file(), "Missing native ORB evidence: " + name)
    tum = np.loadtxt(root / trajectory["tum_file"], ndmin=2)
    require(tum.shape == (len(ids), 8) and np.isfinite(tum).all()
            and (np.diff(tum[:, 0]) > 0).all(), "Invalid native TUM trajectory")
    require(np.allclose(tum[:, 0], timestamps / 1_000_000, rtol=0, atol=2.1e-6),
            "Native TUM timestamps do not map to exported frames")
    from .orbslam3_bridge import _read_tum_trajectory
    from types import SimpleNamespace
    reparsed = _read_tum_trajectory(root / trajectory["tum_file"],
        [SimpleNamespace(frame_id=frame_id) for frame_id in ids], timestamps / 1_000_000)
    require(np.allclose([reparsed[frame_id] for frame_id in ids], poses, rtol=0, atol=1e-8),
            "Exported poses differ from this run's native TUM output")
    if expected_binding is not None:
        verify_orb_runtime(expected_binding["orb_runtime"], trajectory["config"]["root"])
        from .video_session import load_video_session
        session = load_video_session(Path(expected_binding["frozen_session"]["root"]),
                                     validate_frame_files=False)
        frames = {frame.frame_id: frame for frame in session.rgbd.frames}
        require(all(frame_id in frames and timestamps[i] == frames[frame_id].timestamp_us
                    for i, frame_id in enumerate(ids)), "ORB frames/timestamps differ from session")
        command = trajectory.get("command", [])
        executable = Path(trajectory["config"]["root"]).expanduser() / trajectory["config"]["executable"]
        require(command and Path(command[0]).resolve() == executable.resolve(),
                "ORB command did not execute bound native Runtime")
    return {"status": "PASS", "tracked_poses": len(ids), "tracked_frame_ids": ids,
            "camera_to_world": poses.tolist(), "timestamp_us": timestamps.tolist(),
            "tracked_fraction": len(ids) / count, "cross_run_pose_equality_required": False}


verify_orb_trajectory = verify_native_orb


def verify_3d(root, *, failure=False, expected_binding=None):
    root = Path(root)
    if failure:
        read(root / "video_3d_failure.json")
        require(not (root / "video_3d_delivery.json").exists(), "Failed 3-D was published")
        evidence = read(root / "failure_isolation.json")
        require(evidence.get("status") == "PASS", "Missing failure isolation evidence")
        require(evidence.get("orb_failure_injected") is True
                and evidence.get("child_executed") is True
                and evidence.get("result_loadable") is True
                and evidence.get("typed_error") == "ThreeDProcessingError"
                and evidence.get("task_state") in {"THREE_D_FAILED_2D_PRESERVED", "COMPLETED_WITH_WARNINGS"},
                "Missing actual typed ORB failure and preserved 2-D result")
        before = Path(evidence["before_2d"])
        after = Path(evidence["after_2d"])
        for name in (
            "video_delivery.json",
            "video_panorama.png",
            "video_pixel_provenance.npz",
            "video_panorama.jpg",
            "video_report.json",
            "video_timing.json",
        ):
            require(
                sha256(before / name) == sha256(after / name)
                == evidence["before_sha256"][name] == evidence["after_sha256"][name],
                "3-D failure changed 2-D: " + name,
            )
        return {"status": "PASS", "failure_isolation": True}
    delivery = read(root / "video_3d_delivery.json")
    require(
        delivery.get("delivery_state") in ("published", "published_degraded"),
        "3-D not published",
    )
    orb = verify_native_orb(root, expected_binding)
    acceleration = delivery["audit"]["acceleration"]
    require(acceleration.get("requested") == "required"
            and acceleration.get("selected") == "cuda"
            and acceleration.get("fallback_used") is False
            and delivery["audit"]["backend"] == "open3d_tensor_cuda_voxel_block_grid_display_only",
            "3-D did not use Open3D CUDA required")
    for key in ("mesh", "mobile_mesh"):
        path = root / delivery[key]
        with path.open("rb") as handle:
            header = handle.read(12)
        import struct
        require(len(header) == 12 and header[:4] == b"glTF"
                and struct.unpack("<II", header[4:]) == (2, path.stat().st_size),
                "Invalid " + key + " GLB")
    viewer = (root / delivery["viewer"]).read_text(encoding="utf-8")
    require("https://" not in viewer and "http://" not in viewer, "Viewer requires network")
    timing = read(root / "video_3d_timing.json")
    require(timing.get("status") == "published"
            and timing["isolation"]["runs_in_child_process"] is True
            and timing["isolation"]["orb_is_post_capture_only"] is True,
            "Missing isolated post-publication 3-D timing")
    require(timing["isolation"]["two_d_delivery_sha256"] == sha256(root.parent / "video_delivery.json"),
            "3-D references wrong 2-D delivery")
    return {"status": "PASS", "tracked_poses": orb["tracked_poses"],
            "cuda_required": True, "orb": orb}


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
