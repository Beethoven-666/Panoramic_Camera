import numpy as np
import pytest

from panorama_demo.video_s13_runtime_state import S13RuntimeContext, S13Stage, S13StageResult


def _result(stage: S13Stage, revision: int, parent: int | None, run_id: str = "run") -> S13StageResult:
    return S13StageResult(run_id, stage, revision, parent, np.zeros((3, 4, 3), np.uint8), np.ones((3, 4), bool))


def test_parent_chain_accepts_only_p0_to_p3() -> None:
    runtime = S13RuntimeContext("run")
    p0 = runtime.initialize_p0(_result(S13Stage.P0, 0, None))
    p1 = runtime.commit(expected_parent=S13Stage.P0, candidate=_result(S13Stage.P1, 1, p0.revision))
    p2 = runtime.commit(expected_parent=S13Stage.P1, candidate=_result(S13Stage.P2, 2, p1.revision))
    p3 = runtime.commit(expected_parent=S13Stage.P2, candidate=_result(S13Stage.P3, 3, p2.revision))
    assert runtime.current is p3


def test_parent_chain_rejects_invalid_candidate_without_changing_current() -> None:
    runtime = S13RuntimeContext("run")
    p0 = runtime.initialize_p0(_result(S13Stage.P0, 0, None))
    with pytest.raises(RuntimeError, match="another run"):
        runtime.commit(expected_parent=S13Stage.P0, candidate=_result(S13Stage.P1, 1, 0, "other"))
    assert runtime.current is p0
    with pytest.raises(RuntimeError, match="skipped"):
        runtime.commit(expected_parent=S13Stage.P0, candidate=_result(S13Stage.P2, 1, 0))
    assert runtime.current is p0
