# VMware Ubuntu H0 — external qualification tools

These tools qualify only SDK `0.3.0rc1`, subject commit
`2abb132043e7c5ae1805a2ed1dd91516ebf2f874`, runtime variant
`ubuntu22.04-x86_64-py310-sm120`. The tool commit is a separate identity. The
project wheel and all frozen SDK archives remain immutable. Tool fixes require a
new tools archive and a fresh campaign; SDK/runtime/installer/config fixes stop
this campaign and require a new `0.3.0rc2` through phases 3, 4 and 5.

The execution requirements are the user-supplied
`GEMINI305_LINUX_SDK_PHASE_5_NATIVE_H0_CODEX_IMPLEMENTATION_PLAN.md` (phase 5 only),
as amended by the user's 2026-09-07 instruction: replace native bare-metal H0
with VMware virtual-machine acceptance. This amendment changes only the host
environment gate. Ubuntu 22.04, CPython 3.10, non-root execution, real CUDA
compute capability 12.0, frozen offline Runtime, camera, USB, pixel equivalence,
quality, performance, fault, retention and duration gates remain mandatory.
The legacy filenames and v3 schema identifier remain stable; every new binding,
tools manifest and final status identifies
`qualification_environment=VMWARE_UBUNTU_22_04` and
`bare_metal_qualified=false`. A passing `hardware_qualified` is scoped solely
to this VMware campaign and also requires `vmware_qualified=true`.
Historical native bindings/tools cannot qualify this amended campaign.
The old `run_sdk_h0_acceptance.py` and `aggregate_sdk_acceptance.py --kind native`
must not issue phase 5 qualification. Only `aggregate_native_h0_acceptance.py`
writes `native_acceptance_status.json`, using
`gemini305-sdk-native-acceptance/v3`.

## Status of this implementation

External tools and synthetic negative tests are implemented. The clean-tree
Windows repository suite completed with 2262 passed, 0 failed and 10 explained
skips; the native-H0 tool suite completed with 138 passed under both Windows
and Linux CPython 3.10. Ruff, compileall and diff checks passed. The independent
archive was extracted and its deployed members were checked against its manifest.
Those counts describe the previous bare-metal tool revision, before the VMware
amendment. Current revision checks are recorded in the VMware execution report.
The VMware tool suite passed 161 tests on Windows; focused checks include
VMware host admission, rejection of old native bindings/tools, and preserving
the real CUDA and full-campaign gates. Ruff and compileall passed.
U22 (`E:\VMware\U22\Ubuntu 64 位.vmx`) was started and inspected through VMware
Tools as normal user `z` (uid 1000). The guest reports Ubuntu 22.04.5,
CPython 3.10.12 and `systemd-detect-virt=vmware`. Its PCI inventory exposes only
VMware SVGA II graphics, no NVIDIA PCI device; `nvidia-smi` and `/dev/nvidia*`
are absent. The required CUDA CC 12.0 is consequently unavailable on this
configuration. No GPU driver or frozen runtime was modified to disguise this.
No camera evidence is supplied. Test runs on Windows/WSL are tool development
checks only. Hardware, performance, scene quality, physical faults, retention
and endurance remain NOT_EXECUTED until the unchanged compute gate can pass.

These checks establish phase 5A tool-development validation only. They do not
establish native hardware readiness or qualify any of the physical suites.
The final implementation report records the exact implemented and unverified
parts, complete test counts, tool commit, archive identity, and blockers.

## Frozen identities

| Artifact | SHA-256 |
| --- | --- |
| Base | `5e488e8f7145a9aec9da13998a01e507dfac818b9f04d2b1bbbcc72df4eb18e0` |
| Addon | `59db7a1a90178afb2605763510762f3e917707476501f72971ac4594b5fad0c4` |
| Source compliance | `bcc60c197b65f145480f6d89d5d415fc444881a97ef7d607512f61c868cc93b5` |
| Project wheel | `c99b58a389821a6146cc77866c1c631fdc81875f07f940a2edb4b1724ea00cc8` |

Open3D, ORB, production lock, index and software-qualification identities are
read from the frozen index/binding and checked against actual files; this table
does not replace those checks. The source archive is never rebuilt by H0.

## Build and transport the independent tools

Commit only `qualification/native_h0/`, the independent native-H0 scripts,
`tests/test_native_h0_*.py`, and this document. Build from that clean commit:

```bash
python3.10 scripts/build_native_h0_tools.py --help
```

The builder reads exact Git blobs, checks the diff scope against the subject,
and creates a fresh output directory containing
`gemini305-native-h0-tools-<commit>.tar.zst`,
`native-h0-tools-manifest.json`, and `native-h0-tools-checksums.sha256`.
It uses the existing `zstd` executable offline. No dependency download is needed.
The manifest contains tool members and hashes, Python 3.10 requirement, schema
support and build time. The tools contain no project/Open3D wheel, ORB executable,
Vocabulary, capture data or private key. Keep the external manifest and checksum
file beside the extracted tools, as documented by `--help`.

## VMware preparation and preflight order

Use a non-root account in a VMware Ubuntu 22.04 guest, x86_64, CPython 3.10,
glibc >=2.35. `systemd-detect-virt` must return `vmware` with exit code 0;
bare metal, other hypervisors, WSL, containers and unknown detection block this
VMware-specific campaign. GPU capability must be 12.0 with the
frozen private CUDA Open3D and a working CuPy kernel. Local storage must be ext4
or xfs, with at least 30 GiB free for short tests.

Check the guest's actual PCI/CUDA visibility before transporting large frozen
archives. A virtual SVGA adapter or host GPU inventory is not CUDA execution
inside this guest. Missing CUDA stops before recorded replay and camera probe;
CPU fallback is not authorized by the host-environment amendment.

1. Copy the same base/addon/source archives, candidate index, phase 3 freeze,
   phase 4 software status/binding, recorded session and reference results,
   external ORB Runtime and the H0 tools. Verify all supplied checksums.
2. Extract through the frozen archive verifier. Install base and addon using
   their frozen installers with `--offline`. Never use editable install or
   source-tree imports. Unset `PYTHONPATH` and `PYTHONHOME` and export
   `G305_CUDA=required`. Use the installed candidate interpreter for all H0
   scripts; the tools add only their own root to Python's import path.
3. Run `create_native_h0_binding.py` without camera/host inventory for the
   **bootstrap binding**, with a new `--h0-id`. This revalidates every subject
   input, the installed project/Open3D members and the tools archive. A
   bootstrap binding cannot qualify hardware.
4. Prepare a pre-run profile JSON with `width=848`, `height=480`, `fps=60` and an
   explicitly approved `max_sync_delta_us`. No frozen candidate synchronization
   tolerance was found; therefore the tool requires a pre-run value and its
   approval/provenance rather than fitting one to observed results. The value
   must exclude a different 60 Hz frame. Keep the approval in the scene record.
5. `run_native_h0_campaign.py --phase replay --raw-root NEW_ROOT
   --native-binding BOOTSTRAP --profile PROFILE --reference-2d STAGE4_2D` creates
   an exclusive root and runs fresh CLI/SDK recorded replay, exact reference
   comparison and actual native post-3D in isolated processes. A failure stops
   before any camera probing. The reference must be an actual phase 4 sample.
6. Run `collect_native_h0_inventory.py` **only after** the bound replay succeeds,
   and then the installed `g305-sdk-doctor --probe-camera --orb-runtime-root ...
   --camera-lock-root /var/lock/gemini305-sdk --output
   ROOT/preflight/doctor-native-h0.json`.
7. Finalize the native binding using the collected host/camera/USB inventories
   with the same H0 ID; `finalize_native_h0_binding.py --help` describes the
   required bootstrap/root/input arguments. Core execution requires this final
   binding and `verify_doctor(..., profile="native_h0")`.

The inventory records actual model/serial/firmware, VID/PID, wrapper/native SDK,
exact profiles, sysfs speed/port, driver/GPU identity, filesystem and permissions.
Require Gemini 305 firmware 1.0.70, wrapper 2.1.2+g305.1, native SDK 2.9.3, USB
speed >=5000 Mbit/s and exactly one camera. Install only the frozen udev rule;
create `/var/lock/gemini305-sdk` for the normal user. Installation of this rule,
lock-root creation and explicit loopback mounting are the limited sudo actions;
SDK capture itself never runs as root. Read each script's `--help` for exact
paths; do not replace missing evidence with an expected-value JSON file.

## Physical campaign

Record `scene/h0_scene.json` with operator, rail/vehicle, distance, target and
measured speed, direction, nearest object, lighting, mount pose, reference photos
and the fixed USB cable/port. Five samples use the same single-direction route
and approximately 0.5m nearest-object scene with straight and slanted seam edges.

`run_native_h0_campaign.py --phase core` performs a fresh 10-second profile smoke
and exactly `warmup_01`, `run_01` ... `run_05`. Each SDK run uses 848x480 at 60FPS,
3s, 100us exposure, CUDA required, internal Preview enabled/window disabled,
and `RetentionPolicy.KEEP` (the frozen SDK spelling of KEEP_ALWAYS). It keeps the
same SDK instance/Python process across the six samples. Each captured session
is replayed through the CLI in its own worker. A failed sample is preserved and
remaining planned samples are recorded; no sixth replacement sample is added.

The orchestrator then produces fresh actual CLI live and fixed/auto/photo
captures. Ordinary CLI `--no-preview` closes preview; the independent SDK suite
proves enabled internal Preview. The actual candidate behavior is recorded,
never changed through the H0 tools. Capture CLI calls hold a camera lease in the
external harness because bare `g305-capture` itself does not acquire that lease.

The observer calls the installed Python functions unchanged, streams boundary
events, and records real frame metadata, sync readbacks, writer drain, hardware
close, camera lease and Preview publications. Observer errors fail validation.
Every normal frame and its real RGB/depth files are verified, including actual
57–63FPS timing, no drops/errors/regressions, exposure, sync and final output-gate
readback. First-formal-frame recovery into the same session is rejected.

SDK/CLI comparison checks P0–P3 canonical pixels, valid/owner arrays, all
provenance members, sources, schedule, seam transactions, M6 decisions, decoded
P3 and formal PNG. Main formal samples require A/B and no manual review. The
only timing statistic is
`video_timing.json:final_2d.capture_stop_to_p3_published_seconds`:
every sample <=60s, five-sample median <=15s and linear empirical P95 <=20s.

The fifth formal session supplies actual external-ORB/Open3D-CUDA post-3D and an
actual missing-ORB-executable failure. Both retain all six 2-D file hashes;
failure must produce a typed `ThreeDProcessingError`, a real child boundary and
a reloadable 2-D result. ORB poses need finite rigid SE(3) and the configured
tracking gate, never cross-run 1mm/0.1-degree equality.

The operator must review each actual main PNG and write
`scene/h0_visual_review.json` with H0 ID, operator, and `runs.run_01..run_05`, each
containing `accepted`, `reviewed_utc` and the reviewed `png_sha256`. Do not
preapprove unseen images. Seal the scene inputs along with the campaign.

## Faults, retention and endurance

The fault runner uses separate fresh operation directories for no-wait,
wait-cancel, pre-formal-reconnect, post-formal-salvage, early-emergency,
low-disk, camera-lease and SIGTERM. Native operator checkpoints record intent
and confirmation; actual observed events must also prove the fault occurred.
No unplug operation can be replaced by a synthetic exception. Use a dedicated
pre-mounted loopback ext4 filesystem inside this H0 root for low-disk capture.
Do not fill the system filesystem. Preserve it until validation/sealing, then
explicitly unmount and remove only the named loopback image after collecting
the required evidence. Never use SIGKILL as the qualification stop case.

Retention runs use fresh SDK-owned sessions with DELETE_AFTER_ALL_SUCCESS,
including real post-3D, one interruption at the SDK-owned deletion tombstone and
actual journal recovery. Warning, emergency and 3-D failure preserve input;
`process_session()` keeps a newly captured user-owned session. Normal 1+5 and
endurance continue to use KEEP_ALWAYS.

`run_native_h0_endurance.py` executes a real 60-second rate measurement, then a
single uninterrupted 1800-second fixed-exposure capture and a single
7200-second default auto/lock capture. All use the same 848x480/60FPS Preview
profile. Capacity before each long capture is measured bytes/sec * duration *
1.25 +20GiB reserve. Insufficient space blocks the run and qualification.

Telemetry samples actual process-tree RSS, GPU memory/load/temperature,
threads/FDs, memory/swap, disk/throughput, queues, committed frames, checkpoints,
P0 prefix/tail and lease state. Unavailable counters remain NOT_EXECUTED with a
reason. Capture-phase evidence requires complete coverage, no resource runaway,
3GiB RSS/GPU ceilings, zero capture errors, P0 reuse >=0.90,
`full_m0_m3_recomputed=false` and P3 publication <=60s. Raw kernel logs are
required for OOM/Xid/USB/thermal checks. Long data is not substituted with many
short sessions or reused from another campaign.

## Final aggregation and stopping

When every raw suite and operator review is complete, run the campaign `seal`
phase before any final status. This checksummed raw tree, including the locked
profile and scene review, is rehashed by the unique aggregator. Every operation
has a fresh-root start/end receipt and its own evidence checksums. Missing,
changed, foreign or NOT_EXECUTED evidence blocks hardware qualification.

```bash
"$PY" "$TOOLS/scripts/aggregate_native_h0_acceptance.py" \
  --raw-root "$H0" --native-binding "$H0/native_h0_binding.json" \
  --candidate-index "$RC/release-candidate-index.json" \
  --software-acceptance "$RC/software_acceptance_status.json" \
  --output-directory "$H0/final"
```

For a staging-only blocked report, `verify_native_h0_subject.py` produces an
actual frozen-input audit and the aggregator's `--subject-verification` option
can preserve verified software readiness without inventing a native binding.
That path always remains hardware_qualified=false and long_duration_qualified=false.

All gates must pass before hardware_qualified=true and long_duration_qualified=true.
Always retain software_ready=true only when its actual binding passes,
release_ready=false, signature_status=UNSIGNED, external ORB distribution
BLOCKED/ORB_LICENSE and owner EULA WAITING_FOR_OWNER_APPROVAL. Do not create a
release tag, sign a release, publish ORB binaries or continue to phase 6.

## Tool verification

Run focused native-H0 negative tests and the full repository pytest suite,
ruff on qualification/scripts/tests, compileall and git diff --check. Full
pytest includes a disposable wheel-build smoke test that requires a clean Git
tree; it must never replace any frozen project wheel. Supply the existing two
historical source annotation images to a fresh worktree for its existing test.
Record skips by reason. Synthetic/tool tests are not camera, CUDA, ORB,
performance or physical acceptance.
