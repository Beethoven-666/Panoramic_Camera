# Linux L1 acceptance record

Use one new artifact root per attempt. Never overwrite or remove failed samples.

```text
acceptance_id:
git_head:
git_status:
wheel_path:
wheel_sha256:
runtime_manifest:
platform_label:
frozen_session: L1-FROZEN-001
frozen_manifest_sha256:
frozen_calibration_sha256:
frozen_frames_csv_sha256:
```

Required software evidence:

| Gate | Status | Reason code | Artifact |
| --- | --- | --- | --- |
| Five Linux CLI help/integration | NOT_EXECUTED | PENDING | |
| CLI frozen 2-D, warm-up + 5/5 | NOT_EXECUTED | PENDING | |
| SDK frozen 2-D, warm-up + 5/5 | NOT_EXECUTED | PENDING | |
| SDK/CLI exact P0-P3/provenance/decision equivalence | NOT_EXECUTED | PENDING | |
| SDK post-3D, warm-up + 5/5 | NOT_EXECUTED | PENDING | |
| Native ORB correctness and replay | NOT_EXECUTED | PENDING | |
| Wheel external-venv resources/help | NOT_EXECUTED | PENDING | |
| Doctor production/CUDA/Open3D/ORB/wrapper/disk | NOT_EXECUTED | PENDING | |

Report `capture_stop_to_p3_published_seconds` as 2-D user wait. Keep
`primary_post_capture_seconds`, ORB, TSDF and total 3-D time separate. Compare performance only when
input identity and source/pair/canvas/owner workload match. Failed benchmark runs remain in success
rate and statistics input.

Native camera acceptance is a later milestone:

```json
{
  "native_acceptance_status": "NOT_EXECUTED",
  "reason_code": "WAITING_FOR_H0",
  "hardware_qualified": false
}
```

Recorded replay must never be relabeled as non-root physical capture evidence.
