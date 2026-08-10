"""Fail-closed configuration contract for the standalone S1.2 experiment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import yaml


S12_CONFIG_SCHEMA = "gemini305-video-s12-standalone/v1"
S12_ALGORITHM_ID = "S012_standalone_auto_anchor_dense_central_slit_v1"


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"S1.2 configuration section {name!r} must be a mapping")
    return dict(value)


def _number(section: Mapping[str, Any], key: str, *, minimum: float = 0.0) -> float:
    value = section.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"S1.2 configuration {key!r} must be numeric")
    result = float(value)
    if not result >= minimum or result == float("inf"):
        raise ValueError(f"S1.2 configuration {key!r} must be finite and >= {minimum}")
    return result


def _integer(section: Mapping[str, Any], key: str, *, minimum: int = 0) -> int:
    value = section.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"S1.2 configuration {key!r} must be an integer >= {minimum}")
    return value


@dataclass(frozen=True)
class S12TrajectoryConfig:
    maximum_timestamp_delta_ms: float
    maximum_relative_roll_deg: float
    maximum_relative_pitch_deg: float
    maximum_relative_yaw_deg: float
    maximum_source_pruning_rounds: int
    audit_rotation: bool


@dataclass(frozen=True)
class S12ScanConfig:
    minimum_pose_count: int
    minimum_scan_displacement_m: float
    rail_axis_ransac_iterations: int
    rail_axis_inlier_threshold_m: float
    maximum_cross_track_error_m: float
    maximum_backward_step_m: float
    maximum_pause_span_frames: int
    require_image_motion_direction_agreement: bool


@dataclass(frozen=True)
class S12Config:
    """Validated immutable view plus typed M0/M1 settings."""

    trajectory: S12TrajectoryConfig
    scan: S12ScanConfig
    raw: Mapping[str, Any]

    def section(self, name: str) -> Mapping[str, Any]:
        return MappingProxyType(_mapping(self.raw.get(name), name))


_REQUIRED_SECTIONS = {
    "trajectory",
    "scan",
    "motion_graph",
    "anchors",
    "horizontal_solver",
    "source_selection",
    "schedule",
    "open3d_audit",
    "render",
    "vertical",
    "tracks",
    "acceptance",
    "output",
}


def _require_fixed(section: Mapping[str, Any], values: Mapping[str, object], name: str) -> None:
    for key, expected in values.items():
        if section.get(key) != expected:
            raise ValueError(
                f"S1.2 {name}.{key} is fixed at {expected!r}, got {section.get(key)!r}"
            )


def parse_s12_config(document: Mapping[str, Any]) -> S12Config:
    raw = dict(document)
    if raw.get("schema") != S12_CONFIG_SCHEMA:
        raise ValueError(f"S1.2 configuration schema must be {S12_CONFIG_SCHEMA!r}")
    if raw.get("algorithm_id") != S12_ALGORITHM_ID:
        raise ValueError(f"S1.2 algorithm_id must be {S12_ALGORITHM_ID!r}")
    if raw.get("diagnostic_only") is not True or raw.get("production_eligible") is not False:
        raise ValueError("S1.2 Stage A is diagnostic-only and not production eligible")
    missing = sorted(_REQUIRED_SECTIONS - raw.keys())
    if missing:
        raise ValueError("S1.2 configuration is missing sections: " + ", ".join(missing))

    sections = {name: _mapping(raw[name], name) for name in _REQUIRED_SECTIONS}
    trajectory = sections["trajectory"]
    _require_fixed(
        trajectory,
        {
            "require_direct_orb_pose": True,
            "allow_interpolated_pose": False,
            "allow_extrapolated_pose": False,
            "require_uniform_pose_origin": True,
        },
        "trajectory",
    )
    scan = sections["scan"]
    _require_fixed(
        sections["motion_graph"],
        {
            "preserve_reliable_measurements": True,
            "apply_median_filter": False,
            "interpolate_unreliable_edges": False,
            "extrapolate_edges": False,
        },
        "motion_graph",
    )
    _require_fixed(
        sections["source_selection"],
        {"permit_virtual_rgb_source": False},
        "source_selection",
    )
    _require_fixed(
        sections["schedule"],
        {"permit_dynamic_pair_boundary": False, "permit_object_owner": False},
        "schedule",
    )
    _require_fixed(
        sections["open3d_audit"],
        {"allow_rgb_generation": False, "allow_pose_replacement": False, "allow_tsdf": False},
        "open3d_audit",
    )
    _require_fixed(
        sections["render"],
        {
            "transition_width_px": 0,
            "color_gain_enabled": False,
            "rotation_enabled": False,
            "synthetic_hole_fill": False,
            "foreign_source_fill": False,
        },
        "render",
    )
    _require_fixed(
        sections["vertical"],
        {
            "enabled_for_stage_a": False,
            "allow_canvas_interpolated_offset": False,
            "allow_pair_local_warp": False,
            "allow_row_dependent_warp": False,
        },
        "vertical",
    )
    return S12Config(
        trajectory=S12TrajectoryConfig(
            maximum_timestamp_delta_ms=_number(trajectory, "maximum_timestamp_delta_ms"),
            maximum_relative_roll_deg=_number(trajectory, "maximum_relative_roll_deg"),
            maximum_relative_pitch_deg=_number(trajectory, "maximum_relative_pitch_deg"),
            maximum_relative_yaw_deg=_number(trajectory, "maximum_relative_yaw_deg"),
            maximum_source_pruning_rounds=_integer(
                trajectory, "maximum_source_pruning_rounds"
            ),
            audit_rotation=trajectory.get("audit_rotation") is True,
        ),
        scan=S12ScanConfig(
            minimum_pose_count=_integer(scan, "minimum_pose_count", minimum=2),
            minimum_scan_displacement_m=_number(scan, "minimum_scan_displacement_m"),
            rail_axis_ransac_iterations=_integer(
                scan, "rail_axis_ransac_iterations", minimum=1
            ),
            rail_axis_inlier_threshold_m=_number(
                scan, "rail_axis_inlier_threshold_m", minimum=1e-12
            ),
            maximum_cross_track_error_m=_number(scan, "maximum_cross_track_error_m"),
            maximum_backward_step_m=_number(scan, "maximum_backward_step_m"),
            maximum_pause_span_frames=_integer(scan, "maximum_pause_span_frames"),
            require_image_motion_direction_agreement=(
                scan.get("require_image_motion_direction_agreement") is True
            ),
        ),
        raw=MappingProxyType(raw),
    )


def load_s12_config(path_or_mapping: str | Path | Mapping[str, Any]) -> S12Config:
    if isinstance(path_or_mapping, Mapping):
        return parse_s12_config(path_or_mapping)
    path = Path(path_or_mapping).expanduser().resolve()
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ValueError(f"Invalid S1.2 configuration: {path}") from exc
    if not isinstance(document, Mapping):
        raise ValueError("S1.2 configuration root must be a mapping")
    return parse_s12_config(document)


__all__ = [
    "S12_ALGORITHM_ID",
    "S12_CONFIG_SCHEMA",
    "S12Config",
    "S12ScanConfig",
    "S12TrajectoryConfig",
    "load_s12_config",
    "parse_s12_config",
]
