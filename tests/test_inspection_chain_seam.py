from __future__ import annotations

import numpy as np
import pytest

from panorama_demo.inspection_chain_seam import (
    ChainSeamConfig,
    PairCorridorEvidence,
    PanelLocalEvidence,
    audit_panel_chain_topology,
    select_adaptive_nominal_boundaries,
    solve_adjacent_panel_chain,
)


def _full_panels(
    count: int, *, height: int = 32, width: int = 420
) -> list[np.ndarray]:
    return [np.ones((height, width), dtype=bool) for _ in range(count)]


def test_chain_solver_produces_only_adjacent_monotone_closed_owners() -> None:
    height, width = 32, 420
    panels = _full_panels(3, height=height, width=width)
    yy = np.arange(height, dtype=np.float32)[:, None]
    xx = np.arange(width, dtype=np.float32)[None, :]
    first_target = 138.0 + 5.0 * np.sin(yy / 5.0)
    second_target = 282.0 + 4.0 * np.cos(yy / 6.0)
    costs = [
        np.broadcast_to(np.abs(xx - first_target), (height, width)).copy(),
        np.broadcast_to(np.abs(xx - second_target), (height, width)).copy(),
    ]

    result = solve_adjacent_panel_chain(
        panels,
        [140.0, 280.0],
        pair_costs=costs,
        config=ChainSeamConfig(corridor_width_pixels=96),
    )

    assert result.audit["pass"] is True
    assert result.audit["coverage_closed"] is True
    assert result.audit["backward_owner_transition_count"] == 0
    assert result.audit["nonadjacent_owner_transition_count"] == 0
    assert result.audit["row_with_repeated_owner_count"] == 0
    assert len(result.seams) == 2
    assert np.all(result.owner_panel_index >= 0)
    for row in result.owner_panel_index:
        compressed = row[np.r_[True, row[1:] != row[:-1]]]
        assert compressed.tolist() == [0, 1, 2]


def test_pair_coverage_feasibility_closes_the_corridor() -> None:
    height, width = 12, 220
    left = np.ones((height, width), dtype=bool)
    right = np.ones((height, width), dtype=bool)
    left[:, 125:] = False
    right[:, :100] = False
    cost = np.broadcast_to(
        np.abs(np.arange(width, dtype=np.float32)[None, :] - 105.0),
        (height, width),
    ).copy()

    result = solve_adjacent_panel_chain(
        [left, right],
        [110.0],
        pair_costs=[cost],
        target_valid_mask=np.ones((height, width), dtype=bool),
        config=ChainSeamConfig(corridor_width_pixels=96),
    )

    seam = result.seams[0].seam_x_by_row
    assert np.all((seam >= 99) & (seam <= 124))
    assert result.audit["owner_source_coverage_failure_pixel_count"] == 0
    assert result.audit["unowned_target_pixel_count"] == 0


def test_unique_coverage_rows_may_reposition_boundary_without_fake_seam() -> None:
    height, width = 12, 220
    left = np.ones((height, width), dtype=bool)
    right = np.ones((height, width), dtype=bool)
    # Top and bottom rows meet only at opposite corridor edges.  There is no
    # physical two-view seam to constrain while either side has unique
    # coverage, but the middle rows retain a normal shared seam.
    left[:3, 158:] = False
    right[:3, :158] = False
    left[9:, 63:] = False
    right[9:, :63] = False

    result = solve_adjacent_panel_chain(
        [left, right],
        [110.0],
        target_valid_mask=np.ones((height, width), dtype=bool),
        config=ChainSeamConfig(corridor_width_pixels=96),
    )

    seam = result.seams[0]
    assert np.all(seam.seam_x_by_row[:3] == 157)
    assert np.all(seam.seam_x_by_row[9:] == 62)
    assert seam.as_dict()["coverage_relaxed_transition_count"] > 0
    assert result.audit["pass"] is True
    assert result.audit["owner_source_coverage_failure_pixel_count"] == 0


def test_gap_between_adjacent_panels_fails_closed() -> None:
    height, width = 12, 220
    left = np.ones((height, width), dtype=bool)
    right = np.ones((height, width), dtype=bool)
    left[:, 101:] = False
    right[:, :120] = False

    with pytest.raises(RuntimeError, match="feasible closed boundary"):
        solve_adjacent_panel_chain(
            [left, right],
            [110.0],
            target_valid_mask=np.ones((height, width), dtype=bool),
            config=ChainSeamConfig(corridor_width_pixels=96),
        )


def test_locked_foreground_component_is_not_split_by_the_seam() -> None:
    height, width = 24, 420
    panels = _full_panels(3, height=height, width=width)
    locked = np.full((height, width), -1, dtype=np.int16)
    locked[6:18, 132:161] = 0
    locked[4:20, 263:289] = 2

    result = solve_adjacent_panel_chain(
        panels,
        [140.0, 280.0],
        locked_owner_panel_index=locked,
        config=ChainSeamConfig(corridor_width_pixels=128),
    )

    assert np.all(result.owner_panel_index[6:18, 132:161] == 0)
    assert np.all(result.owner_panel_index[4:20, 263:289] == 2)
    assert np.all(result.seams[0].seam_x_by_row[6:18] >= 160)
    assert np.all(result.seams[1].seam_x_by_row[4:20] < 263)
    assert result.audit["locked_owner_mismatch_pixel_count"] == 0


def test_full_chain_lock_constrains_every_adjacent_boundary() -> None:
    height, width = 24, 420
    panels = _full_panels(3, height=height, width=width)
    locked = np.full((height, width), -1, dtype=np.int16)
    # This owner-1 object sits entirely between the two pair corridors.
    # Both boundaries must nevertheless leave it on the owner-1 side.
    locked[5:19, 196:224] = 1

    result = solve_adjacent_panel_chain(
        panels,
        [140.0, 280.0],
        locked_owner_panel_index=locked,
        config=ChainSeamConfig(corridor_width_pixels=96),
    )

    assert np.all(result.owner_panel_index[5:19, 196:224] == 1)
    assert np.all(result.seams[0].seam_x_by_row[5:19] < 196)
    assert np.all(result.seams[1].seam_x_by_row[5:19] >= 223)
    assert result.audit["locked_owner_mismatch_pixel_count"] == 0


def test_impossible_full_chain_lock_fails_during_seam_solving() -> None:
    height, width = 24, 420
    panels = _full_panels(3, height=height, width=width)
    locked = np.full((height, width), -1, dtype=np.int16)
    # Pair 0 is limited to x=76..203. Owner 0 at x>=220 would require
    # its boundary to move outside that audited corridor.
    locked[5:19, 220:236] = 0

    with pytest.raises(RuntimeError, match="feasible closed boundary"):
        solve_adjacent_panel_chain(
            panels,
            [140.0, 280.0],
            locked_owner_panel_index=locked,
            config=ChainSeamConfig(corridor_width_pixels=128),
        )


def test_adaptive_boundary_moves_away_from_foreground_depth_risk() -> None:
    height, width = 64, 420
    panels = _full_panels(3, height=height, width=width)
    first_risk = np.zeros((height, width), dtype=bool)
    second_risk = np.zeros((height, width), dtype=bool)
    first_risk[:, 122:159] = True
    second_risk[:, 262:299] = True
    config = ChainSeamConfig(
        corridor_width_pixels=96,
        adaptive_boundary_maximum_shift_pixels=48,
        adaptive_boundary_risk_guard_pixels=8,
    )

    selection = select_adaptive_nominal_boundaries(
        panels,
        [140.0, 280.0],
        [first_risk, second_risk],
        config=config,
    )
    audit = selection.as_dict()

    assert audit["moved_pair_count"] == 2
    assert audit["mean_risk_occupancy_before"] == pytest.approx(1.0)
    assert audit["mean_risk_occupancy_after"] == pytest.approx(0.0)
    assert audit["mean_risk_occupancy_reduction"] == pytest.approx(1.0)
    assert audit["corridors_nonoverlapping"] is True
    for pair in audit["pairs"]:
        assert pair["corridor_width_pixels"] == 96
        assert pair["risk_occupancy_after"] < pair["risk_occupancy_before"]

    result = solve_adjacent_panel_chain(
        panels,
        selection.selected_boundaries_x,
        config=config,
    )
    assert result.audit["pass"] is True
    assert result.audit["nonadjacent_owner_transition_count"] == 0


def test_adaptive_boundaries_jointly_preserve_corridor_nonoverlap() -> None:
    height, width = 48, 520
    panels = _full_panels(3, height=height, width=width)
    first_risk = np.zeros((height, width), dtype=bool)
    second_risk = np.zeros((height, width), dtype=bool)
    # Local minima would move the first boundary right and the second left.
    # Joint selection must stop before their 128 px corridors overlap.
    first_risk[:, :220] = True
    second_risk[:, 300:] = True
    config = ChainSeamConfig(
        corridor_width_pixels=128,
        adaptive_boundary_maximum_shift_pixels=64,
        adaptive_boundary_risk_guard_pixels=6,
    )

    selection = select_adaptive_nominal_boundaries(
        panels,
        [190.0, 330.0],
        [first_risk, second_risk],
        config=config,
    )

    left, right = selection.selected_boundaries_x
    assert right - left >= 128
    assert selection.as_dict()["corridors_nonoverlapping"] is True


def test_adaptive_boundaries_move_as_chain_to_cover_object_lock() -> None:
    height, width = 32, 520
    panels = _full_panels(4, height=height, width=width)
    risks = [
        np.zeros((height, width), dtype=bool)
        for _ in range(3)
    ]
    locked = np.full((height, width), -1, dtype=np.int16)
    # Owner 1 needs boundary 1 at or beyond x=290. The original 96-pixel
    # corridor ends at x=284, so boundary 1 and the later non-overlapping
    # corridor must move right together.
    locked[7:25, 270:291] = 1

    config = ChainSeamConfig(
        corridor_width_pixels=96,
        adaptive_boundary_maximum_shift_pixels=64,
    )
    selection = select_adaptive_nominal_boundaries(
        panels,
        [140.0, 236.0, 332.0],
        risks,
        locked_owner_panel_index=locked,
        config=config,
    )
    selected = selection.selected_boundaries_x

    assert selected[1] >= 243.0
    assert selected[2] >= 339.0
    assert all(
        row["selected_corridor_lock_compatible"] is True
        for row in selection.pair_audits
    )
    result = solve_adjacent_panel_chain(
        panels,
        selected,
        locked_owner_panel_index=locked,
        config=config,
    )
    assert np.all(result.owner_panel_index[7:25, 270:291] == 1)


def test_compact_evidence_is_pixel_equivalent_to_legacy_canvas_inputs() -> None:
    height, width = 17, 320
    panel_bounds = ((0, 190), (55, 275), (135, 320))
    full_panels = _full_panels(3, height=height, width=width)
    for panel, (x0, x1) in zip(
        full_panels, panel_bounds, strict=True
    ):
        panel[:, :x0] = False
        panel[:, x1:] = False
    full_panels[0][3:6, 160:166] = False
    full_panels[1][7:10, 250:257] = False
    full_panels[2][11:14, 140:147] = False
    compact_panels = [
        PanelLocalEvidence(
            corner_x=x0,
            values=np.ascontiguousarray(panel[:, x0:x1]),
            canvas_width=width,
        )
        for panel, (x0, x1) in zip(
            full_panels, panel_bounds, strict=True
        )
    ]

    full_risks = [
        np.zeros((height, width), dtype=bool),
        np.zeros((height, width), dtype=bool),
    ]
    full_risks[0][:, 96:105] = True
    full_risks[1][:, 216:225] = True
    risk_bounds = ((40, 160), (160, 280))
    compact_risks = [
        PairCorridorEvidence(
            corner_x=x0,
            values=np.ascontiguousarray(risk[:, x0:x1]),
            canvas_width=width,
        )
        for risk, (x0, x1) in zip(
            full_risks, risk_bounds, strict=True
        )
    ]
    target = np.ones((height, width), dtype=bool)
    config = ChainSeamConfig(
        corridor_width_pixels=96,
        maximum_row_step_pixels=2,
        adaptive_boundary_maximum_shift_pixels=12,
        adaptive_boundary_risk_guard_pixels=4,
    )
    legacy_selection = select_adaptive_nominal_boundaries(
        full_panels,
        [100.0, 220.0],
        full_risks,
        target_valid_mask=target,
        config=config,
    )
    compact_selection = select_adaptive_nominal_boundaries(
        compact_panels,
        [100.0, 220.0],
        compact_risks,
        target_valid_mask=target,
        config=config,
    )
    assert (
        compact_selection.selected_boundaries_x
        == legacy_selection.selected_boundaries_x
    )

    yy = np.arange(height, dtype=np.float32)[:, None]
    xx = np.arange(width, dtype=np.float32)[None, :]
    first_target = 101.0 + 3.0 * np.sin(yy / 3.0)
    second_target = 219.0 + 2.0 * np.cos(yy / 4.0)
    full_costs = [
        np.abs(xx - first_target).astype(np.float32),
        np.abs(xx - second_target).astype(np.float32),
    ]
    compact_costs: list[PairCorridorEvidence] = []
    for cost, nominal_x in zip(
        full_costs,
        compact_selection.selected_boundaries_x,
        strict=True,
    ):
        corridor_x0 = int(round(nominal_x)) - 48
        compact_costs.append(
            PairCorridorEvidence(
                corner_x=corridor_x0,
                values=np.ascontiguousarray(
                    cost[:, corridor_x0 : corridor_x0 + 96]
                ),
                canvas_width=width,
            )
        )
    locked = np.full((height, width), -1, dtype=np.int16)
    locked[:, 104:112] = 0
    locked[:, 207:215] = 2

    legacy = solve_adjacent_panel_chain(
        full_panels,
        legacy_selection.selected_boundaries_x,
        pair_costs=full_costs,
        target_valid_mask=target,
        locked_owner_panel_index=locked,
        config=config,
    )
    compact = solve_adjacent_panel_chain(
        compact_panels,
        compact_selection.selected_boundaries_x,
        pair_costs=compact_costs,
        target_valid_mask=target,
        locked_owner_panel_index=locked,
        config=config,
    )

    assert np.array_equal(
        compact.owner_panel_index, legacy.owner_panel_index
    )
    assert np.array_equal(compact.valid_mask, legacy.valid_mask)
    for compact_seam, legacy_seam in zip(
        compact.seams, legacy.seams, strict=True
    ):
        assert np.array_equal(
            compact_seam.seam_x_by_row,
            legacy_seam.seam_x_by_row,
        )
        for row, seam_x in enumerate(compact_seam.seam_x_by_row):
            assert compact.owner_panel_index[row, seam_x] == (
                compact_seam.left_panel_index
            )
            assert compact.owner_panel_index[row, seam_x + 1] == (
                compact_seam.right_panel_index
            )
    for key in (
        "coverage_closed",
        "owner_source_coverage_failure_pixel_count",
        "locked_owner_mismatch_pixel_count",
        "backward_owner_transition_count",
        "nonadjacent_owner_transition_count",
        "owner_map_seam_mismatch_pixel_count",
        "pass",
    ):
        assert compact.audit[key] == legacy.audit[key]

    compact_panel_bytes = sum(
        np.asarray(item.values).nbytes for item in compact_panels
    )
    compact_risk_bytes = sum(
        np.asarray(item.values).nbytes for item in compact_risks
    )
    compact_cost_bytes = sum(
        np.asarray(item.values).nbytes for item in compact_costs
    )
    assert compact_panel_bytes < len(full_panels) * height * width
    assert compact_risk_bytes < len(full_risks) * height * width
    assert compact_cost_bytes == 2 * height * 96 * 4
    assert compact_cost_bytes < 2 * height * width * 4


def test_forbidden_region_makes_one_connected_path_detour() -> None:
    height, width = 28, 220
    panels = _full_panels(2, height=height, width=width)
    forbidden = np.zeros((height, width), dtype=bool)
    forbidden[8:21, 105:116] = True
    cost = np.broadcast_to(
        np.abs(np.arange(width, dtype=np.float32)[None, :] - 110.0),
        (height, width),
    ).copy()

    result = solve_adjacent_panel_chain(
        panels,
        [110.0],
        pair_costs=[cost],
        seam_forbidden_masks=[forbidden],
        config=ChainSeamConfig(
            corridor_width_pixels=96,
            maximum_row_step_pixels=2,
        ),
    )

    path = result.seams[0].seam_x_by_row
    assert np.all((path[8:21] < 104) | (path[8:21] > 115))
    assert int(np.max(np.abs(np.diff(path)))) <= 2
    assert result.audit["seams"][0]["full_height_closed_path"] is True


def test_topology_audit_detects_backward_repeated_owner_island() -> None:
    height, width = 8, 220
    panels = _full_panels(2, height=height, width=width)
    result = solve_adjacent_panel_chain(
        panels,
        [110.0],
        config=ChainSeamConfig(corridor_width_pixels=96),
    )
    broken = result.owner_panel_index.copy()
    broken[:, 150:155] = 0

    audit = audit_panel_chain_topology(
        broken,
        result.valid_mask,
        panels,
        result.seams,
        maximum_row_step_pixels=4,
    )

    assert audit["pass"] is False
    assert audit["backward_owner_transition_count"] > 0
    assert audit["row_with_repeated_owner_count"] == height
    assert audit["owner_map_seam_mismatch_pixel_count"] > 0


def test_topology_audit_detects_nonadjacent_owner_jump() -> None:
    height, width = 8, 420
    panels = _full_panels(3, height=height, width=width)
    result = solve_adjacent_panel_chain(
        panels,
        [140.0, 280.0],
        config=ChainSeamConfig(corridor_width_pixels=96),
    )
    broken = result.owner_panel_index.copy()
    broken[:, 40:45] = 2

    audit = audit_panel_chain_topology(
        broken,
        result.valid_mask,
        panels,
        result.seams,
        maximum_row_step_pixels=4,
    )

    assert audit["pass"] is False
    assert audit["nonadjacent_owner_transition_count"] > 0
    assert audit["adjacent_pair_only"] is False


def test_topology_audit_does_not_bridge_transparent_valid_islands() -> None:
    height, width = 8, 300
    panels = _full_panels(3, height=height, width=width)
    target = np.zeros((height, width), dtype=bool)
    target[:, :90] = True
    target[:, 210:] = True

    result = solve_adjacent_panel_chain(
        panels,
        [100.0, 200.0],
        target_valid_mask=target,
        config=ChainSeamConfig(corridor_width_pixels=96),
    )

    assert np.all(result.owner_panel_index[:, :90] == 0)
    assert np.all(result.owner_panel_index[:, 210:] == 2)
    assert result.audit["backward_owner_transition_count"] == 0
    assert result.audit["nonadjacent_owner_transition_count"] == 0
    assert result.audit["owner_order_monotone"] is True
    assert result.audit["adjacent_pair_only"] is True
    assert result.audit["pass"] is True


@pytest.mark.parametrize("width", [95, 161])
def test_corridor_width_outside_formal_range_is_rejected(width: int) -> None:
    with pytest.raises(ValueError, match=r"\[96, 160\]"):
        solve_adjacent_panel_chain(
            _full_panels(2, height=8, width=220),
            [110.0],
            config=ChainSeamConfig(corridor_width_pixels=width),
        )
