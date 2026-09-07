"""Fresh, bound native campaign operations; SDK code is imported only at execution."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

from .common import read, require, sha256, write_new

SCHEMA = "gemini305-native-h0-campaign/v2"
LABELS = ("warmup_01", "run_01", "run_02", "run_03", "run_04", "run_05")
TWO_D = ("video_delivery.json", "video_panorama.png", "video_panorama.jpg",
         "video_pixel_provenance.npz", "video_report.json", "video_timing.json")


def relative(root, path):
    return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()


def bound_path(root, name):
    path = (Path(root) / name).resolve()
    require(not Path(name).is_absolute() and path.is_relative_to(Path(root).resolve()),
            "Evidence must belong to this H0 root")
    return path


def binding_identity(binding):
    return {"h0_id": binding["h0_id"],
            "subject_source_commit": binding["subject"]["source_commit"],
            "h0_tool_commit": binding["qualification_tool"]["commit"]}


def seal(root):
    """Write actual evidence checksums after all producers have closed their files."""
    root = Path(root)
    with (root / "evidence.sha256").open("x", encoding="utf-8") as stream:
        for path in sorted(root.rglob("*")):
            if path.is_file() and path != root / "evidence.sha256":
                stream.write(f"{sha256(path)}  {relative(root, path)}\n")


def verify_seal(root):
    root = Path(root)
    declared = set()
    with (root / "evidence.sha256").open(encoding="utf-8") as stream:
        for line in stream:
            digest, name = line.rstrip("\n").split("  ", 1)
            require(name not in declared, "Duplicate evidence checksum")
            declared.add(name)
            require(sha256(bound_path(root, name)) == digest, "Evidence changed: " + name)
    actual = {relative(root, p) for p in root.rglob("*")
              if p.is_file() and p != root / "evidence.sha256"}
    require(actual == declared, "Missing or added evidence files")


def seal_campaign(root):
    root = Path(root).resolve()
    require(not (root / "final").exists(), "Seal raw evidence before qualification output")
    seal(root)


def verify_final_evidence_seal(root, binding):
    root = Path(root).resolve()
    Campaign(root, binding)
    declared = set()
    with (root / "evidence.sha256").open(encoding="utf-8") as stream:
        for line in stream:
            digest, name = line.rstrip("\n").split("  ", 1)
            require(name not in declared and not name.startswith("final/"), "Invalid campaign evidence seal")
            declared.add(name)
            require(sha256(bound_path(root, name)) == digest, "Campaign evidence changed: " + name)
    actual = {relative(root, p) for p in root.rglob("*") if p.is_file()
              and p != root / "evidence.sha256" and not p.is_relative_to(root / "final")}
    require(actual == declared, "Campaign evidence added or removed after seal")
    require(sha256(root / "profile.json") == read(root / "profile_lock.json")["sha256"], "Pre-run profile changed")
    return {"status": "PASS", "file_count": len(declared)}


class Campaign:
    def __init__(self, root, binding, *, create=False):
        self.root, self.binding = Path(root).resolve(), binding
        if create:
            self.root.mkdir(parents=True, exist_ok=False)
            write_new(self.root / "campaign.json", {
                "schema": SCHEMA, **binding_identity(binding), "created_ns": time.time_ns(),
                "pid": os.getpid(), "root": str(self.root), "planned_sdk_labels": list(LABELS)})
            write_new(self.root / "bootstrap_binding.json", binding)
        state = read(self.root / "campaign.json")
        require(state["schema"] == SCHEMA and state["root"] == str(self.root), "Wrong H0 campaign root")
        require(all(state.get(k) == v for k, v in binding_identity(binding).items()), "Wrong campaign binding")

    @contextmanager
    def operation(self, name):
        root = bound_path(self.root, name)
        root.mkdir(parents=True, exist_ok=False)
        operation = {"schema": "gemini305-native-h0-operation/v1", **binding_identity(self.binding),
                     "name": name, "started_ns": time.time_ns(), "pid": os.getpid()}
        write_new(root / "operation_start.json", operation)
        try:
            yield root
        except BaseException as exc:
            write_new(root / "operation_end.json", {**operation, "ended_ns": time.time_ns(),
                "status": "FAIL", "error_type": type(exc).__name__, "error": str(exc)})
            seal(root)
            raise
        else:
            write_new(root / "operation_end.json", {**operation, "ended_ns": time.time_ns(), "status": "PASS"})
            seal(root)


def verify_operation(root, binding, name):
    path = bound_path(root, name)
    start, end = read(path / "operation_start.json"), read(path / "operation_end.json")
    require(start["schema"] == "gemini305-native-h0-operation/v1", "Wrong operation schema")
    require(all(start.get(k) == v and end.get(k) == v for k, v in binding_identity(binding).items()),
            "Unbound or stale H0 operation: " + name)
    require(start["name"] == end["name"] == name and end["status"] == "PASS"
            and end["started_ns"] == start["started_ns"]
            and end["ended_ns"] >= start["started_ns"] >= read(Path(root) / "campaign.json")["created_ns"],
            "Failed or invalid operation interval: " + name)
    verify_seal(path)
    return path


def sdk_config(binding, root, **overrides):
    from panorama_demo import VideoSDKConfig
    from panorama_demo.sdk_state import RetentionPolicy
    options = dict(cuda_mode="required", preview_enabled=True, post_3d_enabled=False,
                   retention_policy=RetentionPolicy.KEEP, camera_lock_root="/var/lock/gemini305-sdk",
                   state_root=Path(root) / "sdk-state", orb_runtime_kind="native_linux",
                   orbslam3_root=binding["candidate"]["orb_runtime"]["root"])
    options.update(overrides)
    return VideoSDKConfig(**options)


def verify_usb_snapshots(root, binding):
    """Require this operation's camera to stay on its bound USB3 port."""
    matched = []
    for name in ("usb_before.json", "usb_after.json"):
        record = read(Path(root) / name)
        require(all(record.get(k) == v for k, v in binding_identity(binding).items()),
                "Unbound operation USB snapshot")
        devices = [item for item in record.get("devices", []) if item.get("serial") == binding["camera_serial"]]
        require(len(devices) == 1, "Operation USB snapshot must identify exactly one bound camera")
        device = devices[0]
        require(float(device["speed"]) >= 5000 and device["port_path"] == binding["usb_port_path"],
                "Operation USB speed/port changed")
        matched.append(device)
    require(all(matched[0].get(key) is not None and matched[0][key] == matched[1].get(key)
                for key in ("port_path", "sysfs_realpath", "busnum", "devnum")),
            "USB device changed or re-enumerated during normal capture")


@contextmanager
def usb_capture_boundary(root, binding):
    from .inventory import collect_usb

    def snapshot(name):
        write_new(Path(root) / name, {**binding_identity(binding), "observed_ns": time.time_ns(),
                                     **collect_usb(binding["camera"])})

    snapshot("usb_before.json")
    try:
        yield
    finally:
        snapshot("usb_after.json")
    verify_usb_snapshots(root, binding)


def capture_once(sdk, run_root, *, duration_seconds=3, exposure_us=100, wait_for_camera=True, telemetry=False):
    from panorama_demo.sdk_acceptance import audit_2d_process
    from .observer import NativeCaptureObserver
    from contextlib import nullcontext
    run_root = Path(run_root)
    output = run_root / "2d"
    started = time.time_ns()
    with NativeCaptureObserver(run_root / "capture_observation.json") as observer, audit_2d_process(output):
        if telemetry:
            import psutil
            from .endurance import TelemetryRecorder
            previous = [time.monotonic(), psutil.Process().io_counters().write_bytes]

            def runtime_sample():
                row = observer.snapshot_runtime()
                now, written = time.monotonic(), psutil.Process().io_counters().write_bytes
                row["write_bytes_per_second"] = (written - previous[1]) / (now - previous[0])
                previous[:] = [now, written]
                return row

            recorder = TelemetryRecorder(run_root, os.getpid(), run_root, runtime_sample)
        else:
            recorder = nullcontext()
        with recorder:
            job = sdk.capture_and_process(capture_root=run_root / "sessions", output_dir=output,
                width=848, height=480, fps=60, duration_seconds=duration_seconds,
                video_exposure_us=exposure_us, preview_window=False, wait_for_camera=wait_for_camera,
                maximum_post_seconds=60)
            result = job.result()
    record = {"session": str(result.session), "output": str(result.output_dir),
              "job_id": result.job_id, "state": job.state.value, "started_ns": started,
              "ended_ns": time.time_ns(), "duration_seconds": duration_seconds,
              "video_exposure_us": exposure_us, "preview_enabled": sdk.config.preview_enabled, "preview_window": False,
              "retention_policy": sdk.config.retention_policy.value, "formal_2d_published": result.formal_2d_published,
              "grade": result.overall_grade, "manual_review_required": result.manual_review_required}
    write_new(run_root / "capture_result.json", record)
    return record


def run_worker(operation, root, binding, **options):
    root = Path(root)
    request = {"operation": operation, "root": str(root), "binding": binding, **options}
    request_path = root / f"request-{uuid.uuid4().hex}.json"
    write_new(request_path, request)
    env = os.environ.copy()
    require(not env.get("PYTHONPATH") and not env.get("PYTHONHOME"), "Source import environment forbidden")
    env["G305_CUDA"] = "required"
    script = Path(__file__).resolve().parents[2] / "scripts" / "native_h0_worker.py"
    with (root / f"{operation}.log").open("xb") as log:
        result = subprocess.run([sys.executable, str(script), str(request_path)], cwd=root,
                                env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
    require(result.returncode == 0, f"H0 {operation} worker failed: {result.returncode}; see {root}")


def cli_replay(session, output):
    from panorama_demo.sdk_acceptance import audit_2d_process
    from panorama_demo.video_panorama import main
    with audit_2d_process(output):
        previous = sys.argv
        try:
            sys.argv = ["g305-video-panorama", str(session), "--output", str(output),
                        "--maximum-post-seconds", "60", "--defer-3d"]
            main()
        finally:
            sys.argv = previous


def verify_smoke(root, binding, profile):
    """A failed ten-second camera/profile smoke must stop the hardware campaign."""
    from .capture_verifier import verify_native_capture_session
    from panorama_demo.sdk_acceptance import verify_2d
    record = read(Path(root) / "capture_result.json")
    require(record["duration_seconds"] == 10 and record["formal_2d_published"] is True,
            "Profile smoke did not produce a formal ten-second capture")
    verify_native_capture_session(record["session"], expected_serial=binding["camera_serial"],
        requested_exposure_us=100, max_sync_delta_us=profile["max_sync_delta_us"],
        observation_path=Path(root) / "capture_observation.json")
    verify_2d(record["output"])


def post_3d(root, binding, session, source):
    from panorama_demo import Gemini305VideoSDK, VideoPanoramaResult
    from panorama_demo.sdk_state import ThreeDProcessingError
    from panorama_demo.sdk_acceptance import verify_3d
    root, source = Path(root), Path(source)
    original = {name: sha256(source / name) for name in TWO_D}
    for case in ("success", "failure-isolation"):
        destination = root / case / "2d"
        destination.mkdir(parents=True, exist_ok=False)
        for name in TWO_D:
            shutil.copyfile(source / name, destination / name)
        options = {} if case == "success" else {"orbslam3_executable": "H0_NONEXISTENT_ORB_EXECUTABLE"}
        sdk = Gemini305VideoSDK(sdk_config(binding, root / case, **options))
        if case == "failure-isolation":
            require(not (Path(sdk.config.orbslam3_root) / options["orbslam3_executable"]).exists(),
                    "Injected ORB executable must not exist before execution")
        typed_error = None
        try:
            sdk.run_post_3d(session, destination)
        except ThreeDProcessingError as exc:
            typed_error = type(exc).__name__
            if case == "success":
                raise
        finally:
            sdk.release_control()
        after = {name: sha256(destination / name) for name in TWO_D}
        require(after == original == {name: sha256(source / name) for name in TWO_D}, "3-D altered two-dimensional delivery")
        VideoPanoramaResult.load(session, destination)
        if case == "failure-isolation":
            require(typed_error == "ThreeDProcessingError", "Missing actual typed post-3D failure")
            boundary = read(destination / "3d" / "process_boundary.json")
            require(boundary["child_pid"] != boundary["parent_pid"], "Failure injection child was not executed")
            require(not (Path(sdk.config.orbslam3_root) / options["orbslam3_executable"]).exists(), "Injected ORB must not exist")
            write_new(destination / "3d" / "failure_isolation.json", {
                "status": "PASS", "orb_failure_injected": True, "child_executed": True,
                "result_loadable": True, "typed_error": typed_error,
                "task_state": "THREE_D_FAILED_2D_PRESERVED", "before_2d": str(source),
                "after_2d": str(destination), "before_sha256": original, "after_sha256": after})
        verify_3d(destination / "3d", failure=case != "success",
                  expected_binding={"orb_runtime": binding["candidate"]["orb_runtime"],
                                    "frozen_session": {"root": str(session)}})
    write_new(root / "post_3d.json", {"session": str(session), "source": str(source), "before_sha256": original})


def worker(request):
    """Invoked only under the installed candidate interpreter, outside a source checkout."""
    import importlib.util
    from .binding import installed_wheel_identity
    binding, root = request["binding"], Path(request["root"])
    installed_wheel_identity(binding["candidate"]["project_wheel"]["path"], "gemini305-rgbd-panorama")
    spec = importlib.util.find_spec("panorama_demo")
    require(spec and Path(spec.origin).resolve().is_relative_to(Path(sys.prefix).resolve()),
            "H0 cannot import SDK from a checkout")
    if request["operation"] == "retention":
        from .retention import run_retention_case
        run_retention_case(request["case"], root, binding)
    elif request["operation"] == "endurance":
        from .endurance_campaign import run_endurance_session
        run_endurance_session(root, binding, request["name"], request["profile"], request.get("calibration"))
    elif request["operation"] == "fault":
        from .fault_cases import run_fault_case
        run_fault_case(request["case"], root, binding, sdk_config(binding, root),
                       loopback_root=request.get("loopback_root"))
    elif request["operation"] == "lease-probe":
        from .fault_cases import lease_probe
        lease_probe(root, sdk_config(binding, root), request["serial"], job_root=request.get("job_root"))
    elif request["operation"] == "cli-replay":
        cli_replay(request["session"], request["output"])
    elif request["operation"] == "post-3d":
        post_3d(root, binding, request["session"], request["source"])
    elif request["operation"] == "replay":
        from panorama_demo import Gemini305VideoSDK
        from panorama_demo.sdk_acceptance import audit_2d_process
        from .equivalence import verify_native_equivalence
        sdk = Gemini305VideoSDK(sdk_config(binding, root))
        session = Path(binding["frozen_session"]["root"])
        with audit_2d_process(root / "sdk"):
            sdk.process_session(session, root / "sdk")
        cli_replay(session, root / "cli")
        verify_native_equivalence(root / "sdk", root / "cli", session_root=session)
        verify_native_equivalence(root / "sdk", Path(request["reference_2d"]), session_root=session)
        post_3d(root / "post-3d", binding, session, root / "sdk")
        write_new(root / "replay_result.json", {"status": "PASS", **binding_identity(binding),
            "reference_2d": request["reference_2d"], "session": str(session), "completed_ns": time.time_ns()})
    elif request["operation"] in {"cli-live", "fixed", "auto", "photo"}:
        from panorama_demo.camera_lease import CameraLease
        from .observer import NativeCaptureObserver
        from panorama_demo.sdk_acceptance import audit_2d_process
        from contextlib import nullcontext
        mode = request["operation"]
        if mode == "cli-live":
            from panorama_demo.video_live import main
            argv = ["g305-video-live", "--panorama-output", str(root / "2d"), "--maximum-post-seconds", "60"]
        else:
            from panorama_demo.capture_orbbec import main
            argv = ["g305-capture"]
        argv += ["--width", "848", "--height", "480", "--fps", "60", "--output", str(root / "sessions")]
        if mode == "photo":
            argv += ["--photo-mode", "--max-frames", "10"]
        else:
            argv += ["--duration", "3", "--no-preview"]
            if mode != "auto":
                argv += ["--video-exposure-us", "100"]
        lease = CameraLease("/var/lock/gemini305-sdk", binding["camera_serial"], binding["h0_id"])
        previous = sys.argv
        try:
            with usb_capture_boundary(root, binding), NativeCaptureObserver(root / "capture_observation.json"), (audit_2d_process(root / "2d") if mode == "cli-live" else nullcontext()):
                lease.acquire()
                try:
                    sys.argv = argv
                    main()
                finally:
                    lease.release()
        finally:
            sys.argv = previous
        sessions = list((root / "sessions").rglob("manifest.json"))
        require(len(sessions) == 1, "CLI operation must create exactly one fresh session")
        write_new(root / "capture_result.json", {"session": str(sessions[0].parent), "output": str(root / "2d"),
            "mode": mode, "manifest_sha256": sha256(sessions[0]), "argv": argv})
    else:
        raise ValueError("Unknown H0 worker operation")


def validate_campaign(root, binding):
    """Reopen and revalidate raw evidence. Cached suite PASS flags have no authority."""
    from .capture_verifier import verify_native_capture_session
    from .preview_verifier import verify_native_preview
    from .equivalence import verify_native_equivalence
    from .performance import summarize_performance
    from panorama_demo.sdk_acceptance import verify_2d, verify_3d, verify_doctor
    root = Path(root)
    errors, evidence = [], {}
    try:
        Campaign(root, binding)
        profile = read(root / "profile.json")
        require(profile["width"] == 848 and profile["height"] == 480 and profile["fps"] == 60,
                "H0 profile changed")
        serial = binding["camera_serial"]
        replay = verify_operation(root, binding, "preflight/native-replay")
        from .binding import stage4_reference_identity
        reference = Path(read(replay / "replay_result.json")["reference_2d"])
        require(stage4_reference_identity(reference, binding["software_acceptance"]["path"])
                == read(replay / "stage4_reference_identity.json"), "Phase 4 replay reference identity changed")
        session = Path(binding["frozen_session"]["root"])
        verify_native_equivalence(replay / "sdk", replay / "cli", session_root=session)
        verify_native_equivalence(replay / "sdk", reference, session_root=session)
        verify_3d(replay / "post-3d/success/2d/3d",
                  expected_binding={"orb_runtime": binding["candidate"]["orb_runtime"],
                                    "frozen_session": {"root": str(session)}})
        verify_doctor(root / "preflight/doctor-native-h0.json", profile="native_h0")
        actual = sorted(p.name for p in (root / "sdk-physical").iterdir() if p.is_dir())
        require(actual == sorted(LABELS), "Exactly one warmup and five planned samples required")
        roots = {}
        for label in LABELS:
            path = verify_operation(root, binding, "sdk-physical/" + label)
            verify_usb_snapshots(path, binding)
            result = read(path / "capture_result.json")
            session = bound_path(root, relative(root, result["session"]))
            output = bound_path(root, relative(root, result["output"]))
            require(result["duration_seconds"] == 3 and result["video_exposure_us"] == 100
                    and result["preview_enabled"] is True and result["preview_window"] is False
                    and result["retention_policy"] == "keep", "Changed H0 run settings")
            capture = verify_native_capture_session(session, expected_serial=serial,
                observation_path=path / "capture_observation.json", max_sync_delta_us=profile["max_sync_delta_us"])
            verify_native_preview(path, observation_path=path / "capture_observation.json")
            cross = verify_operation(root, binding, "cli-crosscheck/" + label)
            verify_native_equivalence(output, cross / "2d", session_root=session,
                                      require_standard_quality=label != "warmup_01")
            require(result["formal_2d_published"] is True, "Emergency result cannot qualify normal run")
            if label != "warmup_01":
                report, delivery = read(output / "video_report.json"), read(output / "video_delivery.json")
                require(report["grades"]["overall"] in {"A", "B"} and delivery["manual_review_required"] is False,
                        "H0 quality gate failed: " + label)
                roots[label] = output
            evidence[label] = capture
        evidence["performance"] = summarize_performance(roots)
        require(evidence["performance"]["status"] == "PASS", "H0 performance gate failed: " + str(evidence["performance"]))
        for mode in ("fixed", "auto", "photo", "cli-live", "profile-smoke"):
            name = "capture-contracts/" + mode if mode in {"fixed", "auto", "photo"} else mode
            path = verify_operation(root, binding, name)
            verify_usb_snapshots(path, binding)
            result = read(path / "capture_result.json")
            session = bound_path(root, relative(root, result["session"]))
            verify_native_capture_session(session, expected_serial=serial, mode=mode if mode in {"auto", "photo"} else "fixed",
                requested_exposure_us=None if mode == "auto" else 100,
                observation_path=path / "capture_observation.json", max_sync_delta_us=profile["max_sync_delta_us"])
            if mode == "cli-live":
                verify_2d(path / "2d")
        path = verify_operation(root, binding, "post-3d")
        final_session = read(root / "sdk-physical/run_05/capture_result.json")["session"]
        require(read(path / "post_3d.json")["session"] == final_session, "Post-3D must use fifth formal session")
        verify_3d(path / "success/2d/3d",
                  expected_binding={"orb_runtime": binding["candidate"]["orb_runtime"],
                                    "frozen_session": {"root": final_session}})
        verify_3d(path / "failure-isolation/2d/3d", failure=True)
        scene = read(root / "scene/h0_scene.json")
        require(scene.get("h0_id") == binding["h0_id"], "Standard scene belongs to another H0")
        for key in ("operator", "rail_or_vehicle", "movement_distance_m", "target_speed_m_s",
                    "actual_speed_record", "scan_direction", "nearest_object_distance_m",
                    "lighting", "camera_mount", "usb_cable_port", "sync_tolerance_approval"):
            require(scene.get(key) is not None and scene.get(key) != "", "Missing scene evidence: " + key)
        require(isinstance(scene.get("reference_images"), list) and bool(scene["reference_images"]), "Scene photographs required")
        for image in scene["reference_images"]:
            require(bound_path(root, image).is_file(), "Missing scene photograph")
        review = read(root / "scene/h0_visual_review.json")
        require(review["h0_id"] == binding["h0_id"] and bool(review["operator"]), "Unbound visual review")
        require(set(review["runs"]) == set(LABELS[1:]), "Five visual reviews required")
        for label, row in review["runs"].items():
            require(row["accepted"] is True and row["png_sha256"] == sha256(roots[label] / "video_panorama.png")
                    and bool(row["reviewed_utc"]), "Visual review failed or stale")
    except (ValueError, KeyError, TypeError, OSError) as exc:
        errors.append(str(exc))
    return {"status": "FAIL" if errors else "PASS", "errors": errors, "evidence": evidence}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-binding", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--phase", choices=("replay", "core", "seal"), required=True)
    parser.add_argument("--reference-2d", type=Path)
    parser.add_argument("--profile", type=Path, required=True)
    args = parser.parse_args()
    from .binding import revalidate_binding, stage4_reference_identity
    binding = read(args.native_binding)
    revalidate_binding(binding, binding["candidate"]["candidate_index"]["path"], binding["software_acceptance"]["path"])
    if args.phase == "replay":
        require(args.reference_2d is not None, "Stage4 2-D reference required")
        require(not binding.get("inventory_inputs"), "Replay must precede final camera/host inventory binding")
        reference_identity = stage4_reference_identity(args.reference_2d, binding["software_acceptance"]["path"])
        campaign = Campaign(args.raw_root, binding, create=True)
        write_new(campaign.root / "profile.json", read(args.profile))
        write_new(campaign.root / "profile_lock.json", {"sha256": sha256(campaign.root / "profile.json"),
                  "locked_ns": time.time_ns(), **binding_identity(binding)})
        with campaign.operation("preflight/native-replay") as root:
            write_new(root / "stage4_reference_identity.json", reference_identity)
            run_worker("replay", root, binding, reference_2d=str(args.reference_2d.resolve()))
        return
    campaign = Campaign(args.raw_root, binding)
    require(read(campaign.root / "native_h0_binding.json") == binding,
            "Core and seal require this campaign's finalized native binding")
    require(read(args.profile) == read(campaign.root / "profile.json"), "Pre-run profile changed")
    require(sha256(campaign.root / "profile.json") == read(campaign.root / "profile_lock.json")["sha256"], "Profile bytes changed")
    if args.phase == "seal":
        seal_campaign(campaign.root)
        return
    from .inventory import validate_inventory
    inventory = validate_inventory(campaign.root, binding)
    require(not inventory["errors"], "Native inventory failed: " + str(inventory))
    verify_operation(campaign.root, binding, "preflight/native-replay")
    from panorama_demo import Gemini305VideoSDK
    from panorama_demo.sdk_acceptance import verify_doctor
    verify_doctor(campaign.root / "preflight/doctor-native-h0.json", profile="native_h0")
    sdk = Gemini305VideoSDK(sdk_config(binding, campaign.root))
    with campaign.operation("profile-smoke") as root:
        with usb_capture_boundary(root, binding):
            capture_once(sdk, root, duration_seconds=10)
        verify_smoke(root, binding, read(campaign.root / "profile.json"))
    failures = []
    for label in LABELS:
        try:
            with campaign.operation("sdk-physical/" + label) as root:
                with usb_capture_boundary(root, binding):
                    result = capture_once(sdk, root)
            with campaign.operation("cli-crosscheck/" + label) as root:
                run_worker("cli-replay", root, binding, session=result["session"], output=str(root / "2d"))
        except Exception as exc:
            failures.append({"label": label, "error": str(exc)})
    sdk.release_control()
    write_new(campaign.root / "sdk-physical/suite.json", {"planned_labels": list(LABELS), "failures": failures})
    for mode in ("cli-live", "fixed", "auto", "photo"):
        name = mode if mode == "cli-live" else "capture-contracts/" + mode
        with campaign.operation(name) as root:
            run_worker(mode, root, binding)
    fifth = campaign.root / "sdk-physical/run_05/capture_result.json"
    if fifth.exists():
        record = read(fifth)
        with campaign.operation("post-3d") as root:
            run_worker("post-3d", root, binding, session=record["session"], source=record["output"])
    require(not failures, "Fixed 1+5 campaign contains failures; replacement samples forbidden")


if __name__ == "__main__":
    main()
