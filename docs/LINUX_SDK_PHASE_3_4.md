# Linux SDK phase 3 and 4

The implementation baseline is `codex/linux-sdk-0.3.0rc1` at
`2a71cebe627df1bdc87d95aa68d7b25b19fccbf1`. Work continues on
`codex/linux-sdk-0.3.0rc1-phase3-4`; the S013 production branch is not the build baseline.

## Branch reconciliation

The merge base with production commit `48817c28dc23672b7d444a45e896696bea09eb0e`
is `da9687fc7c3e6cbb9ea7ec98bb1d85eebb7e4b21` (SDK ahead 20, behind 1).
The production-only camera preflight change is semantically covered:

- `capture_orbbec.py` retains the preflight/control/no-warmup-frame error types,
  post-lock timeout classification, and retryable USB transport cause traversal.
- Device discovery recreates the SDK Context for retryable transport failures,
  respects no-wait and cancellation, and bounds transport recovery to eight retries.
- `video_sdk_jobs.py` owns the newer durable capture lifecycle. It retries pre-formal
  warm-up/control failures for at most three attempts and preserves failure evidence.
- No-wait fails on the first attempt. Cancellation releases the camera lease.
- Once a formal frame is committed, even a preflight-class exception finalizes the
  safe committed prefix; it cannot reconnect and append into that session. An
  insufficient prefix produces independent emergency 2-D with warnings.

The bounded retry policy is the approved Linux SDK behavior, replacing the older
unbounded SDK loop. No merge or cherry-pick is needed. The unchanged baseline's
`test_capture_calibration.py` and `test_video_sdk.py` pass (113 tests), including
transport discovery recovery, control/warm-up/post-lock recovery, no-wait,
cancellation, unrelated errors, and committed-frame emergency handling.
These are unit tests, not physical USB disconnect qualification.

## Candidate and qualification boundaries

Build from a clean committed WSL ext4 checkout using the exact build lock. Independently
build the project wheel and base/addon/source compliance archives twice. The release
index identifies the actual archive bytes; content checksum identities remain separate.
The public archives exclude ORB executables and vocabulary. Open3D is the exact private
CUDA wheel in the addon. Acceptance tools travel inside the base archive.

Phase 3 install/relocation/replay/uninstall/tamper evidence can establish only
`RC_BUNDLE_FROZEN`, with all readiness flags false and signature status `UNSIGNED`.
Phase 4 consumes build_a exclusively from outside Git/source trees, with `PYTHONPATH`
and `PYTHONHOME` unset. Its immutable binding identifies archives, wheel, production
lock, runtime variant, external ORB, private Open3D and the frozen RGB-D session.

Required suites are CLI 2-D, SDK 2-D, native ORB and SDK post-3D, each with one warm-up
and five retained formal runs. Formal artifacts are checked every run. Distinct CLI
and SDK results must be exactly equivalent; ORB is checked for real native execution,
valid SE(3), identity and frame/time mapping, without a new pose equality threshold.
Post-3D includes an actual public SDK failure injection that preserves all published
2-D file hashes. Only `aggregate_sdk_acceptance.py --kind software` may write the
software acceptance status. Any phase 4 code fix supersedes the candidate and restarts
phase 3 followed by all phase 4 runs.

WSL USB/IP is supplemental. Native Ubuntu H0, hardware qualification, release readiness,
release tags and formal signing are outside this task. Generated reports and large
artifacts stay outside version control; incomplete tests never establish readiness.
