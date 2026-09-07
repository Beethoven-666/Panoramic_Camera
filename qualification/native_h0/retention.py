"""Verify actual deletion/preservation and journal recovery on physical sessions."""
from __future__ import annotations

from pathlib import Path
import json
import time

from .inventory import bound_errors, read_json, sha256

RETENTION_CASES = ("all-success", "interrupted-cleanup", "warning", "emergency", "3d-failure", "user-session")


def run_retention_case(case, root, binding):
    from panorama_demo import Gemini305VideoSDK
    from .campaign import capture_once, sdk_config
    from .fault_cases import run_fault_case
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if case not in RETENTION_CASES:
        raise ValueError("Unknown retention case")
    options = {"retention_policy": "delete_after_all_success", "post_3d_enabled": case in {"all-success", "interrupted-cleanup", "3d-failure"}}
    if case == "3d-failure":
        options["orbslam3_executable"] = "H0_NONEXISTENT_ORB_EXECUTABLE"
    config = sdk_config(binding, root, **options)
    sdk = Gemini305VideoSDK(config)
    def capture(destination):
        destination.mkdir(parents=True, exist_ok=True)
        if case in {"warning", "emergency"}:
            return run_fault_case("post-formal-salvage" if case == "warning" else "early-emergency",
                                  destination / "physical-fault", binding, config, owned_session=True)
        from .observer import NativeCaptureObserver
        with NativeCaptureObserver(destination / "capture_observation.json"):
            job = sdk.start_capture(destination, width=848, height=480, fps=60,
                duration_seconds=3, video_exposure_us=100, preview_window=False)
            result = job.result()
        return {"session": str(result.session), "output": str(result.output_dir)}
    try:
        if case == "user-session":
            keep_sdk = Gemini305VideoSDK(sdk_config(binding, root / "fresh-input"))
            try:
                return run_user_session_retention(root, binding,
                    lambda destination: capture_once(keep_sdk, destination, duration_seconds=3), sdk)
            finally:
                keep_sdk.release_control()
        return run_owned_retention_case(case, root, binding, lambda destination: capture(destination / "capture"))
    finally:
        sdk.release_control()


def run_owned_retention_case(case, root, binding, capture_action):
    """Run a fresh SDK capture callable while observing its real cleanup boundary.

The caller configures DELETE_AFTER_ALL_SUCCESS and post-3D for success cases.
Warning/emergency cases require a physical fault action during this invocation;
no prior session path is accepted. For interruption, raise one OSError at the
SDK-owned tombstone removal, then call the candidate's real resume_cleanup.
"""
    if case not in RETENTION_CASES or case == "user-session":
        raise ValueError("Expected an owned physical retention case")
    from panorama_demo import sdk_retention
    from .campaign import binding_identity, relative
    root = Path(root)
    if (root / "before.json").exists():
        raise FileExistsError(root / "before.json")
    original_cleanup = sdk_retention.cleanup_owned_session
    original_remove = sdk_retention.shutil.rmtree
    observed = {}

    def cleanup(record, session, **options):
        if observed:
            raise RuntimeError("Retention case must create exactly one fresh SDK session")
        session = Path(session).resolve()
        session.relative_to(root.resolve())
        outputs = {relative(root, name): {"sha256": sha256(name), "bytes": Path(name).stat().st_size}
                   for name in options["output_sha256"]}
        # The SDK's cleanup hash input omits failed 3-D files; preservation
        # qualification must still verify that real failure/log output survives.
        for name in options["output_sha256"]:
            three_d = Path(name).parent / "3d"
            if three_d.is_dir():
                for output_file in three_d.rglob("*"):
                    if output_file.is_file():
                        outputs[relative(root, output_file)] = {"sha256": sha256(output_file), "bytes": output_file.stat().st_size}
                break
        before = {"outputs": outputs, "raw_inputs": snapshot_files(session),
                  "captured_unix_seconds": time.time()}
        (root / "before.json").write_text(json.dumps(before, indent=2))
        observed.update(record=record, session=session, options=options)
        tombstone = session.with_name(session.name + ".delete_pending")

        def interrupted_remove(path, *args, **kwargs):
            if Path(path).resolve() == tombstone:
                (root / "interrupted_job.json").write_text(json.dumps(record.payload, indent=2))
                raise OSError("Native H0 one-shot cleanup interruption at owned tombstone")
            return original_remove(path, *args, **kwargs)

        if case == "interrupted-cleanup":
            sdk_retention.shutil.rmtree = interrupted_remove
        try:
            return original_cleanup(record, session, **options)
        finally:
            sdk_retention.shutil.rmtree = original_remove

    sdk_retention.cleanup_owned_session = cleanup
    try:
        capture_action(root)
    finally:
        sdk_retention.cleanup_owned_session = original_cleanup
        sdk_retention.shutil.rmtree = original_remove
    if not observed:
        raise RuntimeError("Fresh SDK capture did not reach real retention boundary")
    record = observed["record"]
    if case == "interrupted-cleanup":
        value = sdk_retention.resume_cleanup(record)
        (root / "recovery_call.json").write_text(json.dumps({"function": "resume_cleanup",
            "return_value": value, "unix_seconds": time.time()}))
    observation = {**binding_identity(binding), "execution_kind": "native_physical",
                   "retention_policy": observed["options"]["policy"].value,
                   "session_relative_path": relative(root, observed["session"]),
                   "outputs_relative_path": ".", "job_record_relative_path": relative(root, record.root / "state.json")}
    with (root / "observations.json").open("x", encoding="utf-8") as stream:
        json.dump(observation, stream, indent=2)
    return observation


def run_user_session_retention(root, binding, fresh_capture_action, sdk):
    """Capture the input now, then exercise public process_session default ownership."""
    from .campaign import binding_identity, relative
    root = Path(root)
    captured = fresh_capture_action(root / "fresh-input")
    session = Path(captured["session"]).resolve()
    session.relative_to(root.resolve())
    raw = snapshot_files(session)
    result = sdk.process_session(session, root / "processed", defer_3d=True)
    job = sdk.get_job(result.job_id)
    before = {"raw_inputs": raw, "outputs": snapshot_files(root / "processed")}
    (root / "before.json").write_text(json.dumps(before, indent=2))
    observation = {**binding_identity(binding), "execution_kind": "native_physical",
                   "retention_policy": sdk.config.retention_policy.value,
                   "session_relative_path": relative(root, session), "outputs_relative_path": "processed",
                   "job_record_relative_path": relative(root, job._record.root / "state.json")}
    with (root / "observations.json").open("x", encoding="utf-8") as stream:
        json.dump(observation, stream, indent=2)
    return observation


def snapshot_files(root):
    """SHA evidence is required here to prove preserved outputs/inputs unchanged."""
    root = Path(root)
    return {str(path.relative_to(root)): {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in sorted(root.rglob("*")) if path.is_file()}


def verify_snapshot(root, snapshot):
    errors = []
    root = Path(root).resolve()
    if not snapshot:
        return ["Empty pre-operation snapshot"]
    for name, expected in snapshot.items():
        path = (root / name).resolve()
        try:
            path.relative_to(root)
            if not path.is_file() or path.stat().st_size != expected["bytes"] or sha256(path) != expected["sha256"]:
                errors.append(f"Changed/missing preserved file: {name}")
        except (ValueError, OSError, KeyError):
            errors.append(f"Invalid preservation evidence: {name}")
    return errors


def validate_retention(root, binding):
    errors, evidence = [], {}
    root = Path(root).resolve()
    for case in RETENTION_CASES:
        base = root / "retention" / case
        try:
            from .campaign import verify_operation
            verify_operation(root, binding, "retention/" + case)
            record = read_json(base / "observations.json")
            errors.extend(bound_errors(record, binding, case))
            evidence[case] = record
            if record.get("execution_kind") != "native_physical":
                errors.append(f"{case}: physical session required")
            before = read_json(base / "before.json")
            job_path = (base / record["job_record_relative_path"]).resolve()
            job_path.relative_to(root)
            job = read_json(job_path)
            if job.get("schema") != "gemini305-sdk-job/v1":
                errors.append(f"{case}: real SDK job record missing")
            output = (base / record["outputs_relative_path"]).resolve()
            output.relative_to(root)
            errors.extend(f"{case}: {e}" for e in verify_snapshot(output, before["outputs"]))
            session = (base / record["session_relative_path"]).resolve()
            session.relative_to(root)
            if case in {"all-success", "interrupted-cleanup"}:
                cleanup = job.get("cleanup", {})
                if (job.get("owns_session") is not True or cleanup.get("state") != "completed"
                        or session.exists() or session.with_name(session.name + ".delete_pending").exists()):
                    errors.append(f"{case}: SDK-owned session deletion not completed")
                if record.get("retention_policy") != "delete_after_all_success" or not before.get("raw_inputs"):
                    errors.append(f"{case}: default retention/input snapshot missing")
                outputs = before["outputs"]
                if not any(name.endswith("video_delivery.json") for name in outputs) or not any(name.endswith("video_3d_delivery.json") for name in outputs):
                    errors.append(f"{case}: 2-D and requested 3-D delivery absent")
                if job.get("failure") or any(not w.startswith("cleanup_pending:") for w in job.get("warnings", [])):
                    errors.append(f"{case}: successful deletion scenario has failure/warning")
                if case == "interrupted-cleanup":
                    interrupted = read_json(base / "interrupted_job.json")
                    if interrupted.get("cleanup", {}).get("state") not in {"rename_pending", "delete_pending"}:
                        errors.append(f"{case}: actual interrupted journal absent")
                    event = read_json(base / "recovery_call.json")
                    if event.get("function") != "resume_cleanup" or event.get("return_value") is not True:
                        errors.append(f"{case}: actual recovery call evidence absent")
            else:
                errors.extend(f"{case}: {e}" for e in verify_snapshot(session, before["raw_inputs"]))
                if job.get("cleanup", {}).get("state") == "completed":
                    errors.append(f"{case}: preserved input unexpectedly cleaned")
                if case == "user-session" and job.get("owns_session") is not False:
                    errors.append(f"{case}: process_session input ownership invalid")
                if case == "warning" and not job.get("warnings"):
                    errors.append(f"{case}: actual warning missing")
                if case == "emergency" and not any(name.endswith("emergency_delivery.json") for name in before["outputs"]):
                    errors.append(f"{case}: emergency delivery missing")
                if case == "3d-failure" and not any(name.endswith(("video_3d_failure.json", "three_d_failure.json")) for name in before["outputs"]):
                    errors.append(f"{case}: actual 3-D failure artifact missing")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append(f"{case}: NOT_EXECUTED/invalid retention evidence: {exc}")
    return {"errors": errors, "evidence": evidence}
