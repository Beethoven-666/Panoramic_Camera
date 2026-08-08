# Measurement validity report

- Fixed validation annotations: frames 249 and 257 only.
- Evaluator: gemini305-video-offline-visual-evaluation/v2.
- Projection is owner-independent; final owner is consulted only by the final measurement.
- C1, C5, C6 and C8 produced v2 sidecars. C2/C3 were invalid component executions and C4 had no valid projection sidecar, so none can be interpreted as a passing visual result without the required evidence.
- The new selection remains fail-closed: absent, inconsistent, or non-evaluable measurements are measurement-invalid rather than visual passes.
