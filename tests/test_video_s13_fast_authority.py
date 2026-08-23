from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from panorama_demo.video_s13_fast_pipeline import _build_fast_authority


def test_fast_authority_exposes_exact_in_memory_decisions() -> None:
    assignments = (
        SimpleNamespace(
            assignment_index=0, source_index=0, frame_id=10, center_x=2.0,
            left_x=0, right_x=2, zero_width=False,
        ),
        SimpleNamespace(
            assignment_index=1, source_index=1, frame_id=20, center_x=4.0,
            left_x=2, right_x=4, zero_width=False,
        ),
    )
    schedule = SimpleNamespace(
        canvas_height=2, canvas_width=4, canvas_left=0, canvas_right=4,
        boundaries=(0, 2, 4), assignments=assignments,
    )
    seam = np.array([2, 3], dtype=np.int32)
    replay = SimpleNamespace(
        left_frame_id=10, right_frame_id=20, corridor_x0=1, corridor_x1=4,
        seam_x_by_row=seam,
    )
    pair = SimpleNamespace(transaction={"selected_seam_model": "monotone_dp"})
    m5 = SimpleNamespace(pairs=(pair,))
    owner_source = np.array([[0, 0, 1, 1], [0, 0, 0, 1]], dtype=np.int32)
    owner_frame = np.array([[10, 10, 20, 20], [10, 10, 10, 20]], dtype=np.int32)
    blend_plans = (
        SimpleNamespace(transaction=SimpleNamespace(model="B1_narrow_feather")),
    )
    plan = SimpleNamespace(
        photometric_solution=SimpleNamespace(
            model_family="Q4c_quality_cut_component_scalar_gain"
        ),
        blend_plans=blend_plans,
        m63=SimpleNamespace(quality_cut_pair_indices=(0,)),
    )

    authority = _build_fast_authority(
        schedule=schedule,
        m5=m5,
        selected_replay=(replay,),
        c2e=("C1_owner_only",),
        owner_only_pairs=frozenset({0}),
        provenance={
            "owner_source_index": owner_source,
            "owner_frame_id": owner_frame,
        },
        m62={"plan": plan},
    )

    assert authority["source_frame_ids"] == (10, 20)
    assert authority["schedule"]["canvas_shape"] == (2, 4)
    assert authority["owner"]["source_index_map"] is owner_source
    assert authority["owner"]["frame_id_map"] is owner_frame
    assert authority["owner"]["summary"]["source_pixel_counts"] == (
        {"source_index": 0, "frame_id": 10, "pixel_count": 5},
        {"source_index": 1, "frame_id": 20, "pixel_count": 3},
    )
    decision = authority["pair_seam_decisions"][0]
    assert decision["seam_x_by_row"] is seam
    assert decision["m5_selected_seam_model"] == "monotone_dp"
    assert decision["c2e_decision"] == "C1_owner_only"
    assert decision["owner_only"] is True
    assert authority["m6"] == {
        "selected_photometric_model": "Q4c_quality_cut_component_scalar_gain",
        "blend_models": ("B1_narrow_feather",),
        "b0_count": 0,
        "b1_count": 1,
        "quality_cut_pair_indices": (0,),
        "c2e_decisions": ("C1_owner_only",),
        "c2e_owner_only_pair_indices": (0,),
    }
