from __future__ import annotations

import numpy as np

from panorama_demo.video_s13_m61_blend import (
    INELIGIBLE_MODELS,
    M61BlendCandidate,
    b1_two_pixel_weights,
    compose_and_canonicalize_m61_blend,
    plan_m61_blends,
)


def _candidate(*, pair: int = 0, seam: int = 4, left_value: int = 0,
               right_value: int = 100, safe: bool = True) -> M61BlendCandidate:
    shape = (4, 8)
    owner_right = np.broadcast_to(np.arange(8)[None, :] >= seam, shape).copy()
    return M61BlendCandidate(
        pair, np.full(4, seam, np.int32), 0, 8, owner_right,
        np.full(shape, safe), np.zeros(shape, bool), np.ones(shape, bool),
        np.ones(shape, bool), np.full((*shape, 3), left_value, np.uint8),
        np.full((*shape, 3), right_value, np.uint8), evidence_strength=1.0,
    )


def test_b1_pixel_center_golden_is_exactly_two_pixels_at_quarter_weight() -> None:
    candidate = _candidate()
    weight = b1_two_pixel_weights(candidate)
    assert np.array_equal(np.flatnonzero(weight[0]), np.asarray([3, 4]))
    assert np.array_equal(np.unique(weight[weight > 0]), np.asarray([0.25], np.float32))
    plan = plan_m61_blends((candidate,))[0]
    assert plan.transaction.schema == "gemini305-video-s13-blend-transaction/v2"
    assert plan.transaction.model == "B1_narrow_feather_2px"
    assert plan.transaction.total_width_px == 2
    assert plan.transaction.ineligible_models == INELIGIBLE_MODELS
    assert not np.any(plan.active & plan.protected)


def test_b1_requires_immediate_benefit_mde() -> None:
    candidate = _candidate()
    candidate = M61BlendCandidate(
        **{**candidate.__dict__, "immediate_benefit_mde_linear": 1.0}
    )
    plan = plan_m61_blends((candidate,))[0]
    assert plan.transaction.model == "B0_owner_only"
    assert plan.transaction.fallback_reason == "immediate_benefit_below_mde"


def test_conflicting_pairs_use_risk_evidence_benefit_pair_order() -> None:
    weaker = _candidate(pair=0)
    stronger = M61BlendCandidate(**{
        **_candidate(pair=1).__dict__, "structural_risk": 0.0, "evidence_strength": 2.0,
    })
    plans = plan_m61_blends((weaker, stronger))
    assert plans[0].transaction.model == "B0_owner_only"
    assert plans[0].transaction.fallback_reason == "global_pair_conflict"
    assert plans[1].transaction.model == "B1_narrow_feather_2px"


def test_uint8_noop_rebuilds_all_payload_as_b0() -> None:
    candidate = _candidate(left_value=80, right_value=81)
    plan = plan_m61_blends((candidate,))[0]
    output, final, provenance = compose_and_canonicalize_m61_blend(candidate, plan)
    assert final.transaction.model == "B0_owner_only"
    assert final.transaction.fallback_reason == "actual_uint8_noop_canonicalized"
    assert final.transaction.active_pixel_count == 0
    assert not np.any(provenance["active"])
    assert not np.any(provenance["secondary_weight"])
    assert np.all(provenance["blend_transaction_id"] == -1)
    assert np.array_equal(output, np.where(candidate.owner_right[..., None],
                                           candidate.right_rgb, candidate.left_rgb))


def test_protected_and_non_common_pixels_never_activate() -> None:
    candidate = _candidate()
    protected = candidate.protected.copy()
    protected[:, 3] = True
    safe = candidate.safe.copy()
    safe[:, 3] = False
    common = candidate.common_valid.copy()
    common[:, 4] = False
    candidate = M61BlendCandidate(**{
        **candidate.__dict__, "safe": safe, "protected": protected, "common_valid": common,
    })
    weight = b1_two_pixel_weights(candidate)
    assert not np.any(weight)
    assert plan_m61_blends((candidate,))[0].transaction.model == "B0_owner_only"
