"""Topology-closed adjacent-panel seams for side-scan inspection mosaics.

The solver deliberately does not compose RGB.  It partitions a requested
canvas coverage into a left-to-right chain of panel-index owners.  Each of the
``N - 1`` boundaries is solved only between adjacent panels inside one bounded
corridor, so the resulting background ownership cannot jump backwards, skip a
panel, or form disconnected per-row owner islands.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class ChainSeamConfig:
    """Closed limits for the adjacent-panel dynamic-programming solver."""

    corridor_width_pixels: int = 128
    maximum_row_step_pixels: int = 4
    smoothness_penalty: float = 0.25
    nominal_boundary_penalty: float = 0.05
    adaptive_boundary_maximum_shift_pixels: int = 64
    adaptive_boundary_risk_guard_pixels: int = 12
    adaptive_boundary_minimum_common_coverage_ratio: float = 0.50
    adaptive_boundary_shift_penalty: float = 0.05

    def validate(self) -> None:
        if not 96 <= int(self.corridor_width_pixels) <= 160:
            raise ValueError(
                "inspection chain seam corridor width must be in [96, 160]"
            )
        if not 0 <= int(self.maximum_row_step_pixels) <= 16:
            raise ValueError(
                "inspection chain seam maximum row step must be in [0, 16]"
            )
        if not 0 <= int(self.adaptive_boundary_maximum_shift_pixels) <= 160:
            raise ValueError(
                "inspection chain seam adaptive boundary maximum shift "
                "must be in [0, 160]"
            )
        if not 1 <= int(self.adaptive_boundary_risk_guard_pixels) <= 32:
            raise ValueError(
                "inspection chain seam adaptive boundary risk guard "
                "must be in [1, 32]"
            )
        if not (
            0.0
            < float(
                self.adaptive_boundary_minimum_common_coverage_ratio
            )
            <= 1.0
        ):
            raise ValueError(
                "inspection chain seam adaptive boundary minimum common "
                "coverage ratio must be in (0, 1]"
            )
        for name, value in (
            ("smoothness_penalty", self.smoothness_penalty),
            ("nominal_boundary_penalty", self.nominal_boundary_penalty),
            (
                "adaptive_boundary_shift_penalty",
                self.adaptive_boundary_shift_penalty,
            ),
        ):
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(
                    f"inspection chain seam {name} must be finite and non-negative"
                )


@dataclass(frozen=True)
class AdaptiveBoundarySelection:
    """Risk-aware nominal boundaries chosen before per-row seam solving."""

    original_boundaries_x: tuple[float, ...]
    selected_boundaries_x: tuple[float, ...]
    corridor_width_pixels: int
    pair_audits: tuple[dict[str, object], ...]

    def as_dict(self) -> dict[str, object]:
        before = [
            float(value["risk_occupancy_before"])
            for value in self.pair_audits
        ]
        after = [
            float(value["risk_occupancy_after"])
            for value in self.pair_audits
        ]
        return {
            "method": (
                "adjacent_overlap_foreground_depth_risk_occupancy_"
                "with_global_nonoverlap_dp"
            ),
            "enabled": True,
            "pair_count": len(self.pair_audits),
            "corridor_width_pixels": int(self.corridor_width_pixels),
            "original_boundaries_x": [
                float(value) for value in self.original_boundaries_x
            ],
            "selected_boundaries_x": [
                float(value) for value in self.selected_boundaries_x
            ],
            "moved_pair_count": int(
                sum(
                    not math.isclose(original, selected, abs_tol=0.25)
                    for original, selected in zip(
                        self.original_boundaries_x,
                        self.selected_boundaries_x,
                        strict=True,
                    )
                )
            ),
            "mean_risk_occupancy_before": (
                float(np.mean(before)) if before else 0.0
            ),
            "mean_risk_occupancy_after": (
                float(np.mean(after)) if after else 0.0
            ),
            "mean_risk_occupancy_reduction": (
                float(np.mean(before) - np.mean(after)) if before else 0.0
            ),
            "corridors_nonoverlapping": True,
            "pairs": [dict(value) for value in self.pair_audits],
        }


@dataclass(frozen=True)
class AdjacentPanelSeam:
    """One top-to-bottom boundary between two adjacent panel indices."""

    pair_index: int
    left_panel_index: int
    right_panel_index: int
    corridor_x0: int
    corridor_x1: int
    nominal_x: float
    seam_x_by_row: np.ndarray
    terminal_cost: float
    feasible_candidate_count: int
    relaxed_transition_by_row: np.ndarray | None = None

    def as_dict(self) -> dict[str, object]:
        path = np.asarray(self.seam_x_by_row, dtype=np.int32)
        steps = np.abs(np.diff(path))
        relaxed = (
            np.zeros(path.shape, dtype=bool)
            if self.relaxed_transition_by_row is None
            else np.asarray(self.relaxed_transition_by_row, dtype=bool)
        )
        constrained_steps = (
            steps[~(relaxed[:-1] | relaxed[1:])]
            if relaxed.shape == path.shape
            else steps
        )
        return {
            "pair_index": int(self.pair_index),
            "left_panel_index": int(self.left_panel_index),
            "right_panel_index": int(self.right_panel_index),
            "corridor_x0": int(self.corridor_x0),
            "corridor_x1": int(self.corridor_x1),
            "corridor_width_pixels": int(
                self.corridor_x1 - self.corridor_x0
            ),
            "nominal_x": float(self.nominal_x),
            "minimum_seam_x": int(np.min(path)),
            "maximum_seam_x": int(np.max(path)),
            "maximum_row_step_pixels": (
                int(np.max(constrained_steps))
                if constrained_steps.size
                else 0
            ),
            "coverage_relaxed_transition_count": int(
                np.count_nonzero(relaxed[:-1] | relaxed[1:])
            ),
            "terminal_cost": float(self.terminal_cost),
            "feasible_candidate_count": int(self.feasible_candidate_count),
        }


@dataclass(frozen=True)
class ChainSeamResult:
    """Panel-index owner raster plus its adjacent seams and scalar audit."""

    owner_panel_index: np.ndarray
    valid_mask: np.ndarray
    seams: tuple[AdjacentPanelSeam, ...]
    audit: dict[str, object]

    def validate(self) -> None:
        owner = np.asarray(self.owner_panel_index)
        valid = np.asarray(self.valid_mask, dtype=bool)
        if owner.dtype != np.int16 or owner.shape != valid.shape:
            raise RuntimeError(
                "inspection chain seam owner/valid rasters are misaligned"
            )
        if np.any(owner[valid] < 0) or np.any(owner[~valid] != -1):
            raise RuntimeError(
                "inspection chain seam owner validity contract failed"
            )
        if self.audit.get("pass") is not True:
            raise RuntimeError("inspection chain seam topology audit failed")


@dataclass(frozen=True)
class PanelLocalEvidence:
    """One panel raster stored only across its real canvas footprint."""

    corner_x: int
    values: np.ndarray
    canvas_width: int

    def validate(self, *, name: str) -> None:
        array = np.asarray(self.values)
        if array.ndim != 2 or array.shape[0] <= 0 or array.shape[1] <= 0:
            raise ValueError(
                f"inspection chain seam {name} must be a non-empty HxW array"
            )
        if (
            type(self.corner_x) is not int
            or type(self.canvas_width) is not int
            or self.corner_x < 0
            or self.canvas_width <= 0
            or self.corner_x + array.shape[1] > self.canvas_width
        ):
            raise ValueError(
                f"inspection chain seam {name} footprint is outside the canvas"
            )

    @property
    def height(self) -> int:
        return int(np.asarray(self.values).shape[0])

    @property
    def width(self) -> int:
        return int(np.asarray(self.values).shape[1])

    def window(
        self,
        x0: int,
        x1: int,
        *,
        dtype: np.dtype,
        fill: bool | float = False,
    ) -> np.ndarray:
        """Return one canvas-coordinate window without expanding the panel."""

        if not 0 <= x0 <= x1 <= self.canvas_width:
            raise ValueError(
                "inspection panel evidence window is outside the canvas"
            )
        output = np.full(
            (self.height, x1 - x0),
            fill,
            dtype=dtype,
        )
        source_x0 = max(x0, self.corner_x)
        source_x1 = min(x1, self.corner_x + self.width)
        if source_x1 > source_x0:
            output[
                :,
                source_x0 - x0 : source_x1 - x0,
            ] = np.asarray(self.values, dtype=dtype)[
                :,
                source_x0 - self.corner_x : source_x1 - self.corner_x,
            ]
        return output


@dataclass(frozen=True)
class PairCorridorEvidence:
    """One adjacent-pair raster stored only over its relevant x interval."""

    corner_x: int
    values: np.ndarray
    canvas_width: int

    def validate(self, *, name: str) -> None:
        array = np.asarray(self.values)
        if array.ndim != 2 or array.shape[0] <= 0 or array.shape[1] <= 0:
            raise ValueError(
                f"inspection chain seam {name} must be a non-empty HxW array"
            )
        if (
            type(self.corner_x) is not int
            or type(self.canvas_width) is not int
            or self.corner_x < 0
            or self.canvas_width <= 0
            or self.corner_x + array.shape[1] > self.canvas_width
        ):
            raise ValueError(
                f"inspection chain seam {name} footprint is outside the canvas"
            )

    @property
    def height(self) -> int:
        return int(np.asarray(self.values).shape[0])

    @property
    def width(self) -> int:
        return int(np.asarray(self.values).shape[1])

    def window(
        self,
        x0: int,
        x1: int,
        *,
        dtype: np.dtype,
        fill: bool | float,
    ) -> np.ndarray:
        """Return one requested corridor, padding outside evidence footprint."""

        if not 0 <= x0 <= x1 <= self.canvas_width:
            raise ValueError(
                "inspection pair evidence window is outside the canvas"
            )
        output = np.full(
            (self.height, x1 - x0),
            fill,
            dtype=dtype,
        )
        source_x0 = max(x0, self.corner_x)
        source_x1 = min(x1, self.corner_x + self.width)
        if source_x1 > source_x0:
            output[
                :,
                source_x0 - x0 : source_x1 - x0,
            ] = np.asarray(self.values, dtype=dtype)[
                :,
                source_x0 - self.corner_x : source_x1 - self.corner_x,
            ]
        return output


def _as_bool_masks(
    values: Sequence[np.ndarray | PanelLocalEvidence], *, name: str
) -> tuple[PanelLocalEvidence, ...]:
    if len(values) < 2:
        raise ValueError("inspection chain seam needs at least two panels")
    masks: list[PanelLocalEvidence] = []
    for value in values:
        if isinstance(value, PanelLocalEvidence):
            value.validate(name=name)
            masks.append(
                PanelLocalEvidence(
                    corner_x=int(value.corner_x),
                    values=np.asarray(value.values, dtype=bool),
                    canvas_width=int(value.canvas_width),
                )
            )
        else:
            array = np.asarray(value, dtype=bool)
            if array.ndim != 2:
                raise ValueError(
                    f"inspection chain seam {name} must be aligned HxW arrays"
                )
            masks.append(
                PanelLocalEvidence(
                    corner_x=0,
                    values=array,
                    canvas_width=int(array.shape[1]),
                )
            )
    height = masks[0].height
    canvas_width = masks[0].canvas_width
    if any(
        mask.height != height or mask.canvas_width != canvas_width
        for mask in masks
    ):
        raise ValueError(
            f"inspection chain seam {name} must be aligned HxW arrays"
        )
    return tuple(masks)


def _optional_pair_arrays(
    values: Sequence[np.ndarray | PairCorridorEvidence] | None,
    *,
    pair_count: int,
    shape: tuple[int, int],
    name: str,
    dtype: np.dtype,
) -> tuple[PairCorridorEvidence, ...]:
    if values is None:
        fill = False if np.issubdtype(dtype, np.bool_) else 0.0
        return tuple(
            PairCorridorEvidence(
                corner_x=0,
                values=np.full((shape[0], 1), fill, dtype=dtype),
                canvas_width=shape[1],
            )
            for _ in range(pair_count)
        )
    if len(values) != pair_count:
        raise ValueError(
            f"inspection chain seam {name} must contain N-1 arrays"
        )
    arrays: list[PairCorridorEvidence] = []
    for value in values:
        if isinstance(value, PairCorridorEvidence):
            value.validate(name=name)
            evidence = PairCorridorEvidence(
                corner_x=int(value.corner_x),
                values=np.asarray(value.values, dtype=dtype),
                canvas_width=int(value.canvas_width),
            )
        else:
            array = np.asarray(value, dtype=dtype)
            if array.shape != shape:
                raise ValueError(
                    f"inspection chain seam {name} arrays must match the canvas"
                )
            evidence = PairCorridorEvidence(
                corner_x=0,
                values=array,
                canvas_width=shape[1],
            )
        if (
            evidence.height != shape[0]
            or evidence.canvas_width != shape[1]
        ):
            raise ValueError(
                f"inspection chain seam {name} arrays must match the canvas"
            )
        arrays.append(evidence)
    return tuple(arrays)


def _corridor_bounds(
    nominal_x: float, *, width: int, canvas_width: int
) -> tuple[int, int]:
    if canvas_width < width:
        raise ValueError(
            "inspection chain seam canvas is narrower than its corridor"
        )
    center = int(round(float(nominal_x)))
    x0 = center - width // 2
    x0 = min(max(0, x0), canvas_width - width)
    return x0, x0 + width


def _validate_corridors(
    nominal_boundaries_x: Sequence[float],
    *,
    panel_count: int,
    canvas_width: int,
    config: ChainSeamConfig,
) -> tuple[tuple[int, int], ...]:
    if len(nominal_boundaries_x) != panel_count - 1:
        raise ValueError(
            "inspection chain seam needs one nominal boundary per adjacent pair"
        )
    nominal = np.asarray(nominal_boundaries_x, dtype=np.float64)
    if (
        nominal.ndim != 1
        or not np.isfinite(nominal).all()
        or np.any(np.diff(nominal) <= 0.0)
    ):
        raise ValueError(
            "inspection chain seam nominal boundaries must be finite and increasing"
        )
    corridors = tuple(
        _corridor_bounds(
            float(value),
            width=int(config.corridor_width_pixels),
            canvas_width=canvas_width,
        )
        for value in nominal
    )
    if any(
        corridors[index][1] > corridors[index + 1][0]
        for index in range(len(corridors) - 1)
    ):
        raise ValueError(
            "inspection chain seam corridors overlap; ownership could cross"
        )
    return corridors


def _boundary_candidate_metrics(
    *,
    center_x: int,
    original_x: float,
    common: np.ndarray,
    target: np.ndarray,
    risk: np.ndarray,
    config: ChainSeamConfig,
    origin_x: int = 0,
) -> dict[str, float | int]:
    width = target.shape[1]
    radius = int(config.adaptive_boundary_risk_guard_pixels)
    local_center_x = int(center_x) - int(origin_x)
    local_band_x0 = max(0, local_center_x - radius)
    local_band_x1 = min(width, local_center_x + radius + 1)
    local_target = target[:, local_band_x0:local_band_x1]
    local_common = (
        common[:, local_band_x0:local_band_x1] & local_target
    )
    target_support = int(np.count_nonzero(local_target))
    common_support = int(np.count_nonzero(local_common))
    common_ratio = (
        float(common_support / target_support)
        if target_support
        else 0.0
    )
    risk_pixels = int(
        np.count_nonzero(
            risk[:, local_band_x0:local_band_x1] & local_common
        )
    )
    risk_occupancy = (
        float(risk_pixels / common_support) if common_support else 1.0
    )
    maximum_shift = max(
        1, int(config.adaptive_boundary_maximum_shift_pixels)
    )
    normalized_shift = abs(float(center_x) - float(original_x)) / float(
        maximum_shift
    )
    score = (
        risk_occupancy
        + 0.50 * (1.0 - common_ratio)
        + float(config.adaptive_boundary_shift_penalty) * normalized_shift
    )
    return {
        "center_x": int(center_x),
        "band_x0": int(local_band_x0 + origin_x),
        "band_x1": int(local_band_x1 + origin_x),
        "common_support_pixel_count": common_support,
        "common_coverage_ratio": common_ratio,
        "risk_pixel_count": risk_pixels,
        "risk_occupancy": risk_occupancy,
        "normalized_shift": normalized_shift,
        "score": float(score),
    }


def select_adaptive_nominal_boundaries(
    panel_valid_masks: Sequence[np.ndarray | PanelLocalEvidence],
    nominal_boundaries_x: Sequence[float],
    boundary_risk_masks: Sequence[
        np.ndarray | PairCorridorEvidence
    ],
    *,
    target_valid_mask: np.ndarray | None = None,
    locked_owner_panel_index: np.ndarray | None = None,
    config: ChainSeamConfig | Mapping[str, object] | None = None,
) -> AdaptiveBoundarySelection:
    """Move adjacent nominal boundaries away from foreground/depth risk.

    Candidate corridors remain within real adjacent-panel overlap, retain the
    configured 96--160 pixel width, and are selected jointly so neighboring
    corridors cannot overlap.  The function changes only nominal handoff
    locations; the full per-row seam is still solved independently afterwards.
    """

    if isinstance(config, ChainSeamConfig):
        selected_config = config
    else:
        payload = {} if config is None else dict(config)
        unknown = sorted(
            set(payload) - set(ChainSeamConfig.__dataclass_fields__)
        )
        if unknown:
            raise ValueError(
                f"unknown inspection chain seam configuration keys: {unknown}"
            )
        selected_config = ChainSeamConfig(**payload)
    selected_config.validate()

    panels = _as_bool_masks(panel_valid_masks, name="panel valid masks")
    height = panels[0].height
    width = panels[0].canvas_width
    pair_count = len(panels) - 1
    if target_valid_mask is None:
        target = np.zeros((height, width), dtype=bool)
        for panel in panels:
            x0 = panel.corner_x
            x1 = x0 + panel.width
            target[:, x0:x1] |= np.asarray(
                panel.values, dtype=bool
            )
    else:
        target = np.asarray(target_valid_mask, dtype=bool)
    if target.shape != (height, width) or not np.any(target):
        raise ValueError(
            "inspection adaptive boundary target is empty or misaligned"
        )
    locked = (
        np.full((height, width), -1, dtype=np.int16)
        if locked_owner_panel_index is None
        else np.asarray(locked_owner_panel_index, dtype=np.int16)
    )
    if locked.shape != (height, width):
        raise ValueError(
            "inspection adaptive boundary locked owner mask must match "
            "the canvas"
        )
    if np.any((locked < -1) | (locked >= len(panels))):
        raise ValueError(
            "inspection adaptive boundary locked owner contains an "
            "invalid panel index"
        )
    if np.any((locked >= 0) & ~target):
        raise ValueError(
            "inspection adaptive boundary cannot lock an owner outside "
            "target coverage"
        )
    risks = _optional_pair_arrays(
        boundary_risk_masks,
        pair_count=pair_count,
        shape=(height, width),
        name="boundary risk masks",
        dtype=np.dtype(bool),
    )
    original = tuple(float(value) for value in nominal_boundaries_x)
    _validate_corridors(
        original,
        panel_count=len(panels),
        canvas_width=width,
        config=selected_config,
    )

    maximum_shift = int(
        selected_config.adaptive_boundary_maximum_shift_pixels
    )
    corridor_width = int(selected_config.corridor_width_pixels)
    minimum_common = float(
        selected_config.adaptive_boundary_minimum_common_coverage_ratio
    )
    candidate_rows: list[list[dict[str, float | int]]] = []
    lock_bounds: list[tuple[np.ndarray, np.ndarray]] = []
    planning_windows: list[
        tuple[int, np.ndarray, np.ndarray, np.ndarray]
    ] = []
    for pair_index, original_x in enumerate(original):
        canvas_x = np.arange(width, dtype=np.int32)
        must_be_left = (locked >= 0) & (locked <= pair_index)
        must_be_right = locked > pair_index
        locked_lower = np.max(
            np.where(must_be_left, canvas_x[None, :], -1),
            axis=1,
        )
        locked_upper = (
            np.min(
                np.where(
                    must_be_right,
                    canvas_x[None, :],
                    width,
                ),
                axis=1,
            )
            - 1
        )
        lock_bounds.append((locked_lower, locked_upper))
        center = int(round(original_x))
        minimum_center = max(
            corridor_width // 2,
            center - maximum_shift,
        )
        maximum_center = min(
            width - (corridor_width - corridor_width // 2),
            center + maximum_shift,
        )
        minimum_corridor = _corridor_bounds(
            float(minimum_center),
            width=corridor_width,
            canvas_width=width,
        )
        maximum_corridor = _corridor_bounds(
            float(maximum_center),
            width=corridor_width,
            canvas_width=width,
        )
        radius = int(
            selected_config.adaptive_boundary_risk_guard_pixels
        )
        planning_x0 = max(
            0,
            min(minimum_corridor[0], minimum_center - radius),
        )
        planning_x1 = min(
            width,
            max(maximum_corridor[1], maximum_center + radius + 1),
        )
        local_target = target[:, planning_x0:planning_x1]
        common = (
            panels[pair_index].window(
                planning_x0,
                planning_x1,
                dtype=np.dtype(bool),
            )
            & panels[pair_index + 1].window(
                planning_x0,
                planning_x1,
                dtype=np.dtype(bool),
            )
            & local_target
        )
        local_risk = risks[pair_index].window(
            planning_x0,
            planning_x1,
            dtype=np.dtype(bool),
            fill=False,
        )
        planning_windows.append(
            (planning_x0, common, local_target, local_risk)
        )
        common_columns = np.any(common, axis=0)
        by_center: dict[int, dict[str, float | int]] = {}
        for candidate_x in range(minimum_center, maximum_center + 1):
            corridor_x0, corridor_x1 = _corridor_bounds(
                float(candidate_x),
                width=corridor_width,
                canvas_width=width,
            )
            if np.any(
                np.maximum(locked_lower, corridor_x0)
                > np.minimum(locked_upper, corridor_x1 - 1)
            ):
                continue
            if not np.all(
                common_columns[
                    corridor_x0 - planning_x0 :
                    corridor_x1 - planning_x0
                ]
            ):
                continue
            metrics = _boundary_candidate_metrics(
                center_x=candidate_x,
                original_x=original_x,
                common=common,
                target=local_target,
                risk=local_risk,
                config=selected_config,
                origin_x=planning_x0,
            )
            if float(metrics["common_coverage_ratio"]) < minimum_common:
                continue
            by_center[candidate_x] = metrics

        # The original closed topology is always a deterministic fallback.
        # Its later coverage audit remains fail-closed if it is not actually
        # usable, while adaptive selection itself never makes compatibility
        # worse merely because the overlap mask is conservative.
        original_corridor_x0, original_corridor_x1 = _corridor_bounds(
            float(center),
            width=corridor_width,
            canvas_width=width,
        )
        original_lock_compatible = not np.any(
            np.maximum(locked_lower, original_corridor_x0)
            > np.minimum(locked_upper, original_corridor_x1 - 1)
        )
        if center not in by_center and original_lock_compatible:
            by_center[center] = _boundary_candidate_metrics(
                center_x=center,
                original_x=original_x,
                common=common,
                target=local_target,
                risk=local_risk,
                config=selected_config,
                origin_x=planning_x0,
            )
            by_center[center]["fallback_original_candidate"] = 1
        if not by_center:
            raise RuntimeError(
                "inspection adaptive boundary has no lock-compatible "
                f"candidate for pair {pair_index}"
            )
        candidate_rows.append(
            [by_center[key] for key in sorted(by_center)]
        )

    cumulative: list[np.ndarray] = []
    predecessors: list[np.ndarray] = []
    for pair_index, candidates in enumerate(candidate_rows):
        local_cost = np.asarray(
            [float(value["score"]) for value in candidates],
            dtype=np.float64,
        )
        if pair_index == 0:
            cumulative.append(local_cost)
            predecessors.append(
                np.full(local_cost.shape, -1, dtype=np.int32)
            )
            continue
        previous = candidate_rows[pair_index - 1]
        previous_cost = cumulative[-1]
        current_cost = np.full(local_cost.shape, np.inf, dtype=np.float64)
        current_predecessor = np.full(
            local_cost.shape, -1, dtype=np.int32
        )
        for current_index, current in enumerate(candidates):
            current_x0, _ = _corridor_bounds(
                float(current["center_x"]),
                width=corridor_width,
                canvas_width=width,
            )
            best_key: tuple[float, float, int] | None = None
            best_previous = -1
            for previous_index, previous_value in enumerate(previous):
                _, previous_x1 = _corridor_bounds(
                    float(previous_value["center_x"]),
                    width=corridor_width,
                    canvas_width=width,
                )
                if previous_x1 > current_x0:
                    continue
                key = (
                    float(previous_cost[previous_index]),
                    abs(
                        float(previous_value["center_x"])
                        - original[pair_index - 1]
                    ),
                    int(previous_value["center_x"]),
                )
                if best_key is None or key < best_key:
                    best_key = key
                    best_previous = previous_index
            if best_previous >= 0:
                current_cost[current_index] = (
                    local_cost[current_index]
                    + previous_cost[best_previous]
                )
                current_predecessor[current_index] = best_previous
        if not np.any(np.isfinite(current_cost)):
            raise RuntimeError(
                "inspection adaptive boundaries cannot preserve non-overlap"
            )
        cumulative.append(current_cost)
        predecessors.append(current_predecessor)

    terminal_candidates = candidate_rows[-1]
    terminal_keys = [
        (
            float(cumulative[-1][index]),
            abs(float(value["center_x"]) - original[-1]),
            int(value["center_x"]),
            index,
        )
        for index, value in enumerate(terminal_candidates)
    ]
    terminal = min(terminal_keys)[-1]
    selected_indices = [terminal]
    for pair_index in range(pair_count - 1, 0, -1):
        terminal = int(predecessors[pair_index][terminal])
        if terminal < 0:
            raise RuntimeError(
                "inspection adaptive boundary backtracking failed"
            )
        selected_indices.append(terminal)
    selected_indices.reverse()

    selected_boundaries = tuple(
        float(candidate_rows[index][selected_indices[index]]["center_x"])
        for index in range(pair_count)
    )
    _validate_corridors(
        selected_boundaries,
        panel_count=len(panels),
        canvas_width=width,
        config=selected_config,
    )
    pair_audits: list[dict[str, object]] = []
    for pair_index, (original_x, selected_x) in enumerate(
        zip(original, selected_boundaries, strict=True)
    ):
        planning_x0, common, local_target, local_risk = (
            planning_windows[pair_index]
        )
        before = _boundary_candidate_metrics(
            center_x=int(round(original_x)),
            original_x=original_x,
            common=common,
            target=local_target,
            risk=local_risk,
            config=selected_config,
            origin_x=planning_x0,
        )
        after = _boundary_candidate_metrics(
            center_x=int(round(selected_x)),
            original_x=original_x,
            common=common,
            target=local_target,
            risk=local_risk,
            config=selected_config,
            origin_x=planning_x0,
        )
        corridor_x0, corridor_x1 = _corridor_bounds(
            selected_x,
            width=corridor_width,
            canvas_width=width,
        )
        pair_audits.append(
            {
                "pair_index": pair_index,
                "left_panel_index": pair_index,
                "right_panel_index": pair_index + 1,
                "original_nominal_x": original_x,
                "selected_nominal_x": selected_x,
                "boundary_shift_pixels": selected_x - original_x,
                "corridor_x0": corridor_x0,
                "corridor_x1": corridor_x1,
                "corridor_width_pixels": corridor_width,
                "candidate_count": len(candidate_rows[pair_index]),
                "risk_occupancy_before": float(
                    before["risk_occupancy"]
                ),
                "risk_occupancy_after": float(
                    after["risk_occupancy"]
                ),
                "risk_occupancy_reduction": float(
                    before["risk_occupancy"]
                    - after["risk_occupancy"]
                ),
                "common_coverage_ratio_before": float(
                    before["common_coverage_ratio"]
                ),
                "common_coverage_ratio_after": float(
                    after["common_coverage_ratio"]
                ),
                "selected_score": float(after["score"]),
                "adaptive_move_applied": not math.isclose(
                    original_x, selected_x, abs_tol=0.25
                ),
                "locked_owner_constraint_pixel_count": int(
                    np.count_nonzero(locked >= 0)
                ),
                "selected_corridor_lock_compatible": bool(
                    not np.any(
                        np.maximum(
                            lock_bounds[pair_index][0],
                            corridor_x0,
                        )
                        > np.minimum(
                            lock_bounds[pair_index][1],
                            corridor_x1 - 1,
                        )
                    )
                ),
            }
        )
    return AdaptiveBoundarySelection(
        original_boundaries_x=original,
        selected_boundaries_x=selected_boundaries,
        corridor_width_pixels=corridor_width,
        pair_audits=tuple(pair_audits),
    )


def _validate_exclusive_cores(
    panel_valid_masks: Sequence[PanelLocalEvidence],
    target_valid_mask: np.ndarray,
    corridors: Sequence[tuple[int, int]],
) -> None:
    width = target_valid_mask.shape[1]
    for panel_index, panel_valid in enumerate(panel_valid_masks):
        x0 = 0 if panel_index == 0 else corridors[panel_index - 1][1]
        x1 = (
            width
            if panel_index == len(panel_valid_masks) - 1
            else corridors[panel_index][0]
        )
        if x1 <= x0:
            continue
        missing = (
            target_valid_mask[:, x0:x1]
            & ~panel_valid.window(
                x0,
                x1,
                dtype=np.dtype(bool),
            )
        )
        if np.any(missing):
            raise RuntimeError(
                "inspection chain seam exclusive panel core lacks closed coverage"
            )


def _pair_allowed_candidates(
    *,
    pair_index: int,
    corridor_x0: int,
    left_valid: np.ndarray,
    right_valid: np.ndarray,
    target_valid: np.ndarray,
    seam_forbidden: np.ndarray,
    locked_owner: np.ndarray,
) -> np.ndarray:
    local_target = target_valid
    left_bad = local_target & ~left_valid
    right_bad = local_target & ~right_valid
    left_prefix_bad = np.cumsum(left_bad, axis=1) > 0
    right_suffix_bad = (
        np.cumsum(right_bad[:, ::-1], axis=1)[:, ::-1] > 0
    )
    right_bad_after = np.zeros(right_bad.shape, dtype=bool)
    right_bad_after[:, :-1] = right_suffix_bad[:, 1:]
    allowed = ~left_prefix_bad & ~right_bad_after

    local_forbidden = seam_forbidden & local_target
    boundary_forbidden = local_forbidden.copy()
    boundary_forbidden[:, :-1] |= local_forbidden[:, 1:]
    allowed &= ~boundary_forbidden

    local_x = np.arange(
        corridor_x0,
        corridor_x0 + local_target.shape[1],
        dtype=np.int32,
    )
    canvas_x = np.arange(locked_owner.shape[1], dtype=np.int32)
    must_be_left = (locked_owner >= 0) & (
        locked_owner <= pair_index
    )
    must_be_right = locked_owner > pair_index
    lower = np.max(
        np.where(must_be_left, canvas_x[None, :], -1), axis=1
    )
    upper = (
        np.min(
            np.where(
                must_be_right,
                canvas_x[None, :],
                locked_owner.shape[1],
            ),
            axis=1,
        )
        - 1
    )
    allowed &= local_x[None, :] >= lower[:, None]
    allowed &= local_x[None, :] <= upper[:, None]
    return allowed


def _solve_path(
    energy: np.ndarray,
    allowed: np.ndarray,
    *,
    maximum_step: int,
    smoothness_penalty: float,
    relaxed_rows: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    height, width = energy.shape
    cost = np.where(allowed, energy, np.inf).astype(np.float64)
    if np.any(~np.any(np.isfinite(cost), axis=1)):
        raise RuntimeError(
            "inspection chain seam has a row without a feasible closed boundary"
        )
    previous = cost[0].copy()
    relaxed = (
        np.zeros(height, dtype=bool)
        if relaxed_rows is None
        else np.asarray(relaxed_rows, dtype=bool)
    )
    if relaxed.shape != (height,):
        raise ValueError(
            "inspection chain seam relaxed-row mask is misaligned"
        )
    back = np.full((height, width), -1, dtype=np.int16)
    delta_order = [0]
    for distance in range(1, maximum_step + 1):
        delta_order.extend((-distance, distance))
    deltas = np.asarray(delta_order, dtype=np.int16)
    columns = np.arange(width, dtype=np.int32)

    for row in range(1, height):
        if relaxed[row] or relaxed[row - 1]:
            source = int(np.argmin(previous))
            source_cost = float(previous[source])
            current = cost[row] + source_cost
            back[row, np.isfinite(current)] = np.int16(source)
            previous = current
            continue
        candidates = np.full(
            (deltas.size, width), np.inf, dtype=np.float64
        )
        for delta_index, delta_value in enumerate(deltas):
            source = columns + int(delta_value)
            inside = (source >= 0) & (source < width)
            candidates[delta_index, inside] = (
                previous[source[inside]]
                + smoothness_penalty * abs(int(delta_value))
            )
        best_delta_index = np.argmin(candidates, axis=0)
        best_cost = candidates[best_delta_index, columns]
        current = cost[row] + best_cost
        source = columns + deltas[best_delta_index].astype(np.int32)
        source[~np.isfinite(current)] = -1
        back[row] = source.astype(np.int16)
        previous = current

    terminal = int(np.argmin(previous))
    terminal_cost = float(previous[terminal])
    if not math.isfinite(terminal_cost):
        raise RuntimeError(
            "inspection chain seam has no top-to-bottom feasible path"
        )
    path = np.empty(height, dtype=np.int32)
    path[-1] = terminal
    for row in range(height - 1, 0, -1):
        source = int(back[row, path[row]])
        if source < 0:
            raise RuntimeError(
                "inspection chain seam backtracking reached an invalid state"
            )
        path[row - 1] = source
    return path, terminal_cost


def _expected_owner_from_seams(
    shape: tuple[int, int],
    seams: Sequence[AdjacentPanelSeam],
    target_valid_mask: np.ndarray,
) -> np.ndarray:
    height, width = shape
    columns = np.arange(width, dtype=np.int32)[None, :]
    owner = np.zeros((height, width), dtype=np.int16)
    for seam in seams:
        boundary = np.asarray(seam.seam_x_by_row, dtype=np.int32)
        owner += (columns > boundary[:, None]).astype(np.int16)
    owner[~target_valid_mask] = -1
    return owner


def audit_panel_chain_topology(
    owner_panel_index: np.ndarray,
    target_valid_mask: np.ndarray,
    panel_valid_masks: Sequence[np.ndarray | PanelLocalEvidence],
    seams: Sequence[AdjacentPanelSeam],
    *,
    locked_owner_panel_index: np.ndarray | None = None,
    maximum_row_step_pixels: int = 4,
) -> dict[str, object]:
    """Return a scalar fail-closed audit of one panel-chain partition."""

    panels = _as_bool_masks(panel_valid_masks, name="panel valid masks")
    target = np.asarray(target_valid_mask, dtype=bool)
    owner = np.asarray(owner_panel_index)
    canvas_shape = (panels[0].height, panels[0].canvas_width)
    if target.shape != canvas_shape or owner.shape != target.shape:
        raise ValueError(
            "inspection chain seam audit arrays must match the canvas"
        )
    panel_count = len(panels)
    expected_pair_count = panel_count - 1
    invalid_owner_range = int(
        np.count_nonzero(
            target & ((owner < 0) | (owner >= panel_count))
        )
    )
    unowned = int(np.count_nonzero(target & (owner < 0)))
    outside_owned = int(np.count_nonzero(~target & (owner >= 0)))

    source_coverage_failure = 0
    for panel_index, panel_valid in enumerate(panels):
        panel_x0 = panel_valid.corner_x
        panel_x1 = panel_x0 + panel_valid.width
        source_coverage_failure += int(
            np.count_nonzero(
                target[:, :panel_x0]
                & (owner[:, :panel_x0] == panel_index)
            )
        )
        source_coverage_failure += int(
            np.count_nonzero(
                target[:, panel_x1:]
                & (owner[:, panel_x1:] == panel_index)
            )
        )
        source_coverage_failure += int(
            np.count_nonzero(
                target[:, panel_x0:panel_x1]
                & (owner[:, panel_x0:panel_x1] == panel_index)
                & ~np.asarray(panel_valid.values, dtype=bool)
            )
        )

    backward = 0
    nonadjacent = 0
    repeated_rows = 0
    maximum_runs = 0
    for row in range(owner.shape[0]):
        target_columns = np.flatnonzero(target[row])
        values = owner[row, target_columns]
        if not values.size:
            continue
        compressed = values[
            np.r_[True, values[1:] != values[:-1]]
        ]
        differences = np.diff(compressed)
        backward += int(np.count_nonzero(differences < 0))
        # A transparent/invalid gap has no RGB owner and therefore no panel
        # boundary to cross.  Do not manufacture a non-adjacent transition
        # by concatenating two disconnected valid islands.  Monotonicity is
        # still checked across the complete row above, while adjacency is
        # checked independently inside every spatially connected valid run.
        run_starts = np.r_[
            0,
            np.flatnonzero(np.diff(target_columns) > 1) + 1,
        ]
        run_ends = np.r_[run_starts[1:], target_columns.size]
        for run_start, run_end in zip(
            run_starts, run_ends, strict=True
        ):
            run_values = values[run_start:run_end]
            run_compressed = run_values[
                np.r_[True, run_values[1:] != run_values[:-1]]
            ]
            nonadjacent += int(
                np.count_nonzero(np.abs(np.diff(run_compressed)) > 1)
            )
        repeated_rows += int(
            np.unique(compressed).size != compressed.size
        )
        maximum_runs = max(maximum_runs, int(compressed.size))

    seam_audits: list[dict[str, object]] = []
    invalid_seam_count = 0
    previous_path: np.ndarray | None = None
    for pair_index, seam in enumerate(seams):
        path = np.asarray(seam.seam_x_by_row, dtype=np.int32)
        correct_pair = (
            seam.pair_index == pair_index
            and seam.left_panel_index == pair_index
            and seam.right_panel_index == pair_index + 1
        )
        full_height = path.shape == (owner.shape[0],)
        in_corridor = bool(
            full_height
            and np.all(path >= seam.corridor_x0)
            and np.all(path < seam.corridor_x1)
        )
        steps = np.abs(np.diff(path)) if full_height else np.empty(0)
        relaxed = (
            np.zeros(path.shape, dtype=bool)
            if seam.relaxed_transition_by_row is None
            else np.asarray(seam.relaxed_transition_by_row, dtype=bool)
        )
        constrained_steps = (
            steps[~(relaxed[:-1] | relaxed[1:])]
            if full_height and relaxed.shape == path.shape
            else steps
        )
        step_valid = bool(
            full_height
            and (
                not constrained_steps.size
                or int(np.max(constrained_steps))
                <= int(maximum_row_step_pixels)
            )
        )
        ordered = bool(
            previous_path is None
            or (full_height and np.all(previous_path < path))
        )
        seam_pass = correct_pair and full_height and in_corridor and step_valid and ordered
        invalid_seam_count += int(not seam_pass)
        row = seam.as_dict()
        row.update(
            {
                "correct_adjacent_pair": bool(correct_pair),
                "full_height_closed_path": bool(full_height),
                "inside_corridor": bool(in_corridor),
                "row_step_within_limit": bool(step_valid),
                "strictly_after_previous_seam": bool(ordered),
                "pass": bool(seam_pass),
            }
        )
        seam_audits.append(row)
        if full_height:
            previous_path = path

    expected_owner_mismatch = owner.size
    if len(seams) == expected_pair_count and invalid_seam_count == 0:
        expected = _expected_owner_from_seams(
            owner.shape, seams, target
        )
        expected_owner_mismatch = int(
            np.count_nonzero(owner != expected)
        )

    locked_mismatch = 0
    if locked_owner_panel_index is not None:
        locked = np.asarray(locked_owner_panel_index, dtype=np.int16)
        if locked.shape != target.shape:
            raise ValueError(
                "inspection chain seam locked owner mask must match the canvas"
            )
        locked_valid = locked >= 0
        locked_mismatch = int(
            np.count_nonzero(locked_valid & (owner != locked))
        )

    passed = (
        len(seams) == expected_pair_count
        and invalid_seam_count == 0
        and invalid_owner_range == 0
        and unowned == 0
        and outside_owned == 0
        and source_coverage_failure == 0
        and backward == 0
        and nonadjacent == 0
        and repeated_rows == 0
        and expected_owner_mismatch == 0
        and locked_mismatch == 0
    )
    return {
        "schema": "inspection-adjacent-panel-chain-topology/v1",
        "panel_count": panel_count,
        "expected_pair_count": expected_pair_count,
        "actual_pair_count": len(seams),
        "target_valid_pixel_count": int(np.count_nonzero(target)),
        "owned_target_pixel_count": int(
            np.count_nonzero(target & (owner >= 0))
        ),
        "unowned_target_pixel_count": unowned,
        "owned_outside_target_pixel_count": outside_owned,
        "invalid_owner_range_pixel_count": invalid_owner_range,
        "owner_source_coverage_failure_pixel_count": (
            source_coverage_failure
        ),
        "backward_owner_transition_count": backward,
        "nonadjacent_owner_transition_count": nonadjacent,
        "row_with_repeated_owner_count": repeated_rows,
        "maximum_owner_runs_per_row": maximum_runs,
        "owner_map_seam_mismatch_pixel_count": expected_owner_mismatch,
        "locked_owner_mismatch_pixel_count": locked_mismatch,
        "invalid_seam_count": invalid_seam_count,
        "coverage_closed": bool(unowned == 0 and source_coverage_failure == 0),
        "owner_order_monotone": bool(
            backward == 0 and nonadjacent == 0 and repeated_rows == 0
        ),
        "adjacent_pair_only": bool(nonadjacent == 0),
        "seams": seam_audits,
        "pass": bool(passed),
    }


def solve_adjacent_panel_chain(
    panel_valid_masks: Sequence[np.ndarray | PanelLocalEvidence],
    nominal_boundaries_x: Sequence[float],
    *,
    pair_costs: Sequence[
        np.ndarray | PairCorridorEvidence
    ] | None = None,
    seam_forbidden_masks: Sequence[
        np.ndarray | PairCorridorEvidence
    ] | None = None,
    target_valid_mask: np.ndarray | None = None,
    locked_owner_panel_index: np.ndarray | None = None,
    config: ChainSeamConfig | Mapping[str, object] | None = None,
) -> ChainSeamResult:
    """Solve a topology-closed owner partition for ordered panels.

    Evidence may use legacy full-canvas arrays or compact panel/corridor
    containers. ``locked_owner_panel_index`` may assign foreground pixels to
    one complete real panel before seam solving; every adjacent boundary is
    constrained to leave those pixels on the requested side.
    """

    if isinstance(config, ChainSeamConfig):
        selected = config
    else:
        payload = {} if config is None else dict(config)
        unknown = sorted(
            set(payload) - set(ChainSeamConfig.__dataclass_fields__)
        )
        if unknown:
            raise ValueError(
                f"unknown inspection chain seam configuration keys: {unknown}"
            )
        selected = ChainSeamConfig(**payload)
    selected.validate()

    panels = _as_bool_masks(panel_valid_masks, name="panel valid masks")
    height = panels[0].height
    width = panels[0].canvas_width
    pair_count = len(panels) - 1
    if target_valid_mask is None:
        target = np.zeros((height, width), dtype=bool)
        for panel in panels:
            x0 = panel.corner_x
            x1 = x0 + panel.width
            target[:, x0:x1] |= np.asarray(
                panel.values, dtype=bool
            )
    else:
        target = np.asarray(target_valid_mask, dtype=bool)
    if target.shape != (height, width) or not np.any(target):
        raise ValueError(
            "inspection chain seam target valid mask is empty or misaligned"
        )

    costs = _optional_pair_arrays(
        pair_costs,
        pair_count=pair_count,
        shape=(height, width),
        name="pair costs",
        dtype=np.dtype(np.float32),
    )
    if any(
        not np.isfinite(value.values).all()
        or np.any(value.values < 0.0)
        for value in costs
    ):
        raise ValueError(
            "inspection chain seam pair costs must be finite and non-negative"
        )
    forbidden = _optional_pair_arrays(
        seam_forbidden_masks,
        pair_count=pair_count,
        shape=(height, width),
        name="seam forbidden masks",
        dtype=np.dtype(bool),
    )
    locked = (
        np.full((height, width), -1, dtype=np.int16)
        if locked_owner_panel_index is None
        else np.asarray(locked_owner_panel_index, dtype=np.int16)
    )
    if locked.shape != (height, width):
        raise ValueError(
            "inspection chain seam locked owner mask must match the canvas"
        )
    if np.any((locked < -1) | (locked >= len(panels))):
        raise ValueError(
            "inspection chain seam locked owner contains an invalid panel index"
        )
    if np.any((locked >= 0) & ~target):
        raise ValueError(
            "inspection chain seam cannot lock an owner outside target coverage"
        )
    for panel_index, panel_valid in enumerate(panels):
        x0 = panel_valid.corner_x
        x1 = x0 + panel_valid.width
        lacks_coverage = (
            np.any(locked[:, :x0] == panel_index)
            or np.any(locked[:, x1:] == panel_index)
            or np.any(
                (locked[:, x0:x1] == panel_index)
                & ~np.asarray(panel_valid.values, dtype=bool)
            )
        )
        if lacks_coverage:
            raise RuntimeError(
                "inspection chain seam locked owner lacks real panel coverage"
            )

    corridors = _validate_corridors(
        nominal_boundaries_x,
        panel_count=len(panels),
        canvas_width=width,
        config=selected,
    )
    _validate_exclusive_cores(panels, target, corridors)

    seams: list[AdjacentPanelSeam] = []
    for pair_index, ((x0, x1), nominal_x) in enumerate(
        zip(corridors, nominal_boundaries_x, strict=True)
    ):
        left_valid = panels[pair_index].window(
            x0,
            x1,
            dtype=np.dtype(bool),
        )
        right_valid = panels[pair_index + 1].window(
            x0,
            x1,
            dtype=np.dtype(bool),
        )
        local_target = target[:, x0:x1]
        local_forbidden = forbidden[pair_index].window(
            x0,
            x1,
            dtype=np.dtype(bool),
            fill=False,
        )
        allowed = _pair_allowed_candidates(
            pair_index=pair_index,
            corridor_x0=x0,
            left_valid=left_valid,
            right_valid=right_valid,
            target_valid=local_target,
            seam_forbidden=local_forbidden,
            locked_owner=locked,
        )
        local_x = np.arange(x0, x1, dtype=np.float64)
        energy = costs[pair_index].window(
            x0,
            x1,
            dtype=np.dtype(np.float32),
            fill=0.0,
        ).astype(
            np.float64,
            copy=False,
        )
        energy += (
            float(selected.nominal_boundary_penalty)
            * np.abs(local_x[None, :] - float(nominal_x))
            / max(1.0, float(x1 - x0))
        )
        try:
            local_path, terminal_cost = _solve_path(
                energy,
                allowed,
                maximum_step=int(selected.maximum_row_step_pixels),
                smoothness_penalty=float(selected.smoothness_penalty),
                relaxed_rows=~np.any(
                    left_valid & right_valid & local_target,
                    axis=1,
                ),
            )
        except RuntimeError as exc:
            raise RuntimeError(
                "inspection chain seam pair "
                f"{pair_index} ({pair_index}->{pair_index + 1}) failed: "
                f"{exc}"
            ) from exc
        relaxed_rows = ~np.any(
            left_valid & right_valid & local_target,
            axis=1,
        )
        seams.append(
            AdjacentPanelSeam(
                pair_index=pair_index,
                left_panel_index=pair_index,
                right_panel_index=pair_index + 1,
                corridor_x0=x0,
                corridor_x1=x1,
                nominal_x=float(nominal_x),
                seam_x_by_row=np.ascontiguousarray(
                    local_path + x0, dtype=np.int32
                ),
                terminal_cost=terminal_cost,
                feasible_candidate_count=int(np.count_nonzero(allowed)),
                relaxed_transition_by_row=np.ascontiguousarray(
                    relaxed_rows, dtype=bool
                ),
            )
        )

    owner = _expected_owner_from_seams(
        (height, width), seams, target
    )
    audit = audit_panel_chain_topology(
        owner,
        target,
        panels,
        seams,
        locked_owner_panel_index=locked,
        maximum_row_step_pixels=int(selected.maximum_row_step_pixels),
    )
    if audit["pass"] is not True:
        raise RuntimeError(
            "inspection chain seam could not produce a closed monotone "
            f"topology: {audit}"
        )
    result = ChainSeamResult(
        owner_panel_index=owner,
        valid_mask=np.ascontiguousarray(target),
        seams=tuple(seams),
        audit=audit,
    )
    result.validate()
    return result


__all__ = [
    "AdaptiveBoundarySelection",
    "AdjacentPanelSeam",
    "ChainSeamConfig",
    "ChainSeamResult",
    "PairCorridorEvidence",
    "PanelLocalEvidence",
    "audit_panel_chain_topology",
    "select_adaptive_nominal_boundaries",
    "solve_adjacent_panel_chain",
]
