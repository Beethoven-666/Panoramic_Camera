from __future__ import annotations

import numpy as np
import pytest

from panorama_demo.video_s13_luminance_field import (
    S13LuminanceFieldDisabledConfig,
    canonical_zero_luminance_fields,
    luminance_field_document,
    solve_nonzero_luminance_field,
    validate_disabled_luminance_fields,
)


def test_q4_disabled_contract_is_canonical_and_auditable() -> None:
    S13LuminanceFieldDisabledConfig()
    arrays = canonical_zero_luminance_fields([10, 11], sample_count=3)
    validate_disabled_luminance_fields(arrays, frame_ids=[10, 11])
    document = luminance_field_document(arrays)
    assert document["q4_state"] == "disabled"
    assert document["solver_invocations"] == 0
    assert np.count_nonzero(arrays["source_u_field"]) == 0


def test_q4_rejects_enable_nonzero_and_solver_invocation() -> None:
    with pytest.raises(ValueError):
        S13LuminanceFieldDisabledConfig(enabled=True)
    arrays = canonical_zero_luminance_fields([10], sample_count=1)
    arrays["source_u_field"][0, 0] = 0.01
    with pytest.raises(ValueError):
        validate_disabled_luminance_fields(arrays, frame_ids=[10])
    with pytest.raises(RuntimeError):
        solve_nonzero_luminance_field()
