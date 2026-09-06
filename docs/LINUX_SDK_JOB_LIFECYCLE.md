# Linux SDK job lifecycle (phase 1 candidate)

`Gemini305VideoSDK` preserves the photo SDK API and adds managed continuous RGB-D
jobs. The production renderer remains the exact locked S013 V11 ignore-pose path.
No camera or 3-D runtime is initialized by the constructor or state queries.

```python
from panorama_demo import Gemini305VideoSDK, VideoSDKConfig

sdk = Gemini305VideoSDK(VideoSDKConfig())
job = sdk.start_capture('/data/g305')
status = sdk.get_status(job.job_id)
# Called by the application's stop action:
sdk.stop_capture(job.job_id)
result = job.result()
print(result.panorama_png, result.formal_2d_published)
sdk.release_control()
```

The lock directory `/var/lock/gemini305-sdk` must be explicitly provisioned on a
local filesystem and writable by camera operators. A kernel flock serializes
discovery before Orbbec access and holds the actual serial lease through camera
shutdown and writer drain. A second client can read state/Preview but cannot
cancel another client's job. Kernel locks are released after controller death;
lock files are never deleted to decide ownership.

Managed jobs use `output_root/jobs/<uuid>/` with atomic `state.json`, append-only
`events.jsonl`, `control.json`, `session/`, `preview/`, `2d/`, `checkpoints/`,
`failures/` and `cleanup/`. The state index defaults to
`~/.local/state/gemini305-sdk`. `capture_and_process(capture_root, output_dir)`
retains its external-input convention. `process_session(session, output_dir)`
always treats input as user-owned and refuses an already published output.

States advance CREATED → WAITING_FOR_CAMERA → WARMING_UP → CAPTURING → STOPPING →
FINALIZING_2D → PUBLISHED_2D → optional PROCESSING_3D → COMPLETED or
COMPLETED_WITH_WARNINGS. No-data stops end CANCELLED_NO_DATA; unrecoverable errors
end FAILED. Only zero-commit preflight errors permit bounded retries (three SDK
attempts); a committed frame permanently prevents reconnecting that session.
SIGINT/SIGTERM in the continuous CLI request cooperative stop rather than aborting
the writer. The SDK uses explicit stop/cancel methods and does not replace host
application signal handlers.

Disk preflight runs before creating the capture session. Runtime checks use free
bytes/inodes and measured committed bytes, at least every second or 60 frames.
The reserve defaults to 10 GiB plus a minimum 120-second capture runway calculation
and finalization scratch budget. LOW_DISK stops capture before writing into the
reserve. Three-dimensional work has an additional budget and cannot delay stop.

The journal fsyncs JPEG, aligned depth, CSV and metadata before advancing its
durable checkpoint (first frame, then at most 30 frames or one second). Recovery
verifies that checkpoint and each included file, falls back to the previous
checkpoint when needed, and writes a new derived session. Original failure,
shutdown and timestamp evidence is preserved. A timestamp regression prohibits
formal video qualification, including in the derived session.

Insufficient formal input produces `emergency_panorama.png`, separate emergency
provenance/report and `emergency_delivery.json`; it never produces a formal
`video_delivery.json`. Reliable adjacent motion can authorize measured narrow
central strips; otherwise the deterministic best complete JPEG is returned.
`VideoPanoramaResult.formal_2d_published` is false and manual review is required.

The independent online worker reads only canonical decoded committed JPEGs,
persists a disk ledger and two generations of mapped P0, and freezes a verified
sealed prefix plus mutable tail. It keeps at most eight decoded sources and four
prepared frames. Preview is optional, latest-only and non-authoritative. Preview
failure cannot invalidate capture or frozen P0. Damaged P0 falls back to complete
execution of the same V11, never a legacy renderer.

`run_post_3d(result)` accepts only published formal 2-D and supervises a separate
process after 2-D returns. A watchdog/cancel kills its process tree, records failure
under `2d/3d/`, and checks that delivery, PNG and provenance stayed unchanged.
The default capture API defers 3-D. Requested 3-D failure leaves usable 2-D and a
COMPLETED_WITH_WARNINGS job. ORB/Open3D are never imported by this supervisor.

Default retention deletes only SDK-owned sessions after verified, warning-free
requested outputs. External input, salvage, emergency, manual review and failed
3-D retain data. Cleanup records its intended rename before `.delete_pending`
and deletion. `list_recoverable_jobs()` is read-only; `recover_job()` explicitly
resumes cleanup or creates a new recovery job after the old controller exits.
Construction never launches recovery or camera work.

Public exceptions derive from `SDKError` and carry `error_code`, `stage`,
`recoverable`, `artifact_path`, `cause_type` and `as_dict()`. Stable codes include
CAMERA_BUSY, CAMERA_LOCK_ROOT_UNAVAILABLE, CAMERA_PREFLIGHT_FAILED, LOW_DISK,
JOB_CONTROL_NOT_OWNED, PANORAMA_PROCESSING_ERROR, THREE_D_PROCESSING_ERROR,
THREE_D_TIMEOUT and THREE_D_CANCELLED.

Phase 1 evidence is development-level Windows synthetic tests and WSL recorded
RGB-D replay. It does not constitute native Ubuntu H0, physical capture, speed,
software-ready, hardware-qualified or release-ready acceptance.
