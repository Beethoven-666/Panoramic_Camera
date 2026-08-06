# Extended candidate decision

C9–C13 have all undergone locked `validation_only` experiments using the fixed session, immutable annotation split, verified real ORB trajectory, and evaluator v2. The quality-gate lock remains unchanged (`6a627a4fa548e4e161b70b1babe6fb9c625d7d6dfe33eca2b0b12e99b421b484`).

| Candidate | Physical validation conclusion |
| --- | --- |
| C9 | All line-preserving mesh proposals rejected by existing positive-Jacobian, scale, and depth-safety audits; zero final mesh pixels. |
| C10 | No depth-conditioned layout grid passed the existing final-output safety audit. |
| C11 | No foreground compositor pixels: its optional background refinement did not provide a safe changed grid. |
| C12 | Both genuine local windows (7 and 5 sources) attempted the real RAFT/depth joint-owner solve and failed because a protected owner lacked a genuine source sample. |
| C13 | The robust photometric bundle made no eligible final-output correction; its inherited graph/component audit remained fail-closed. |

These are candidate execution / geometry-observability failures under unchanged gates, not successful visual results. `algorithm_selection_extended_v1.json` therefore remains `not_selectable`. No holdout has been consumed and no production lock exists. C14 is not justified: none of C9–C13 demonstrated a validation improvement that could be combined without creating a misleading hybrid.

The conditional holdout, independent 3 m validation, production freeze, and 20 m handoff are intentionally not reached because their prerequisite—an eligible validation candidate—was not met.
