"""Atomic Stage-A runner for the standalone automatic-anchor S1.2 experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import yaml

from .video_s12_anchor import (
    S12AnchorConfig,
    S12AnchorEvidence,
    build_anchor_evidence,
    select_automatic_anchors,
)
from .video_s12_config import S12_ALGORITHM_ID, S12Config, load_s12_config
from .video_s12_metrics import (
    audit_open3d_source_edges,
    percentiles,
    select_unique_sources,
)
from .video_s12_motion_graph import (
    S12MotionGraphError,
    S12MotionGraphConfig,
    S12MotionNode,
    build_motion_graph,
    calibrate_analysis_image,
    estimate_direct_motion_edge,
    solve_horizontal_centers,
)
from .video_s12_renderer import render_s012_stage_a
from .video_s12_scan import select_s12_scan_segment
from .video_s12_schedule import build_s012_schedule
from .video_s12_session import S12Session, load_s12_session


REPORT_SCHEMA = "gemini305-video-s12-standalone-report/v1"
FAILURE_SCHEMA = "gemini305-video-s12-failure/v1"
COMPLETION_SCHEMA = "gemini305-video-s12-stage-a-completion/v1"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the isolated S1.2 automatic-anchor dense central-slit experiment"
    )
    parser.add_argument("input", type=Path, help="Strict continuous RGB-D video session")
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=_default_config())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("a", "b"), required=True)
    parser.add_argument("--stage-a-lock", type=Path)
    parser.add_argument("--output-mode", choices=("audit", "minimal"), default=None)
    return parser


def _default_config() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "configs/video_candidates/S012_standalone_auto_anchor_dense_central_slit_v1.yaml"
    )


def _validation_regions_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "configs/video_candidates/S012_validation_regions_v1.json"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _plain(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _plain(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    if isinstance(value, list):
        return [_plain(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(_plain(dict(row)))


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[2],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _git_provenance() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=root,
        check=False,
        capture_output=True,
    )
    diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD", "--", "src", "tests", "configs", "README.md"],
        cwd=root,
        check=False,
        capture_output=True,
    )
    return {
        "commit_sha": _git_commit(),
        "worktree_dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
        "tracked_diff_sha256": (
            hashlib.sha256(diff.stdout).hexdigest() if diff.returncode == 0 else None
        ),
    }


def _atomic_write_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        _write_json(temporary, payload)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _motion_config(config: S12Config) -> S12MotionGraphConfig:
    motion = config.section("motion_graph")
    solver = config.section("horizontal_solver")
    local = motion.get("local_edge", {})
    anchor_long = motion.get("anchor_long", {})
    return S12MotionGraphConfig(
        analysis_width_px=int(motion["analysis_width_px"]),
        lk_forward_backward_max_px=float(motion["lk_forward_backward_max_px"]),
        ransac_residual_px=float(motion["ransac_residual_px"]),
        minimum_inlier_count=int(motion["minimum_inlier_count"]),
        minimum_inlier_ratio=float(motion["minimum_inlier_ratio"]),
        minimum_model_inlier_count=int(motion["minimum_model_inlier_count"]),
        minimum_fb_retention_ratio=float(motion["minimum_fb_retention_ratio"]),
        minimum_model_inlier_ratio_retained=float(
            motion["minimum_model_inlier_ratio_retained"]
        ),
        minimum_effective_inlier_ratio_detected=float(
            motion["minimum_effective_inlier_ratio_detected"]
        ),
        minimum_grid_coverage=float(motion["minimum_grid_coverage"]),
        minimum_vertical_span_fraction=float(motion["minimum_vertical_span_fraction"]),
        maximum_parallax_cluster_ratio=float(motion["maximum_parallax_cluster_ratio"]),
        minimum_edge_confidence=float(motion.get("minimum_edge_confidence", 0.0)),
        adjacent_candidate_steps=tuple(int(value) for value in motion["adjacent_candidate_steps"]),
        skip_candidate_steps=tuple(int(value) for value in motion["skip_candidate_steps"]),
        local_edge_maximum_frame_gap=int(
            local.get("maximum_frame_gap", motion.get("maximum_direct_edge_frame_gap", 64))
        ),
        local_edge_lk_window_size=int(local.get("lk_window_size", 21)),
        local_edge_lk_max_level=int(local.get("lk_max_level", 3)),
        anchor_long_use_local_frame_gap=bool(anchor_long.get("use_local_frame_gap", False)),
        anchor_long_maximum_frame_gap=int(anchor_long.get("maximum_frame_gap", 256)),
        anchor_long_minimum_spacing_px=float(anchor_long.get("minimum_spacing_px", 64.0)),
        anchor_long_maximum_spacing_px=float(anchor_long.get("maximum_spacing_px", 144.0)),
        anchor_long_require_timestamp_audit=bool(
            anchor_long.get("require_timestamp_audit", True)
        ),
        anchor_long_minimum_timestamp_gap_ms=float(
            anchor_long.get("minimum_timestamp_gap_ms", 0.001)
        ),
        anchor_long_maximum_timestamp_gap_ms=float(
            anchor_long.get("maximum_timestamp_gap_ms", 10_000.0)
        ),
        anchor_long_require_predicted_overlap=bool(
            anchor_long.get("require_predicted_overlap", True)
        ),
        anchor_long_minimum_predicted_overlap_fraction=float(
            anchor_long.get("minimum_predicted_overlap_fraction", 0.50)
        ),
        anchor_long_require_direct_image_match=bool(
            anchor_long.get("require_direct_image_match", True)
        ),
        anchor_long_use_provisional_initial_flow=bool(
            anchor_long["use_provisional_initial_flow"]
        ),
        anchor_long_initial_flow_source=str(anchor_long["initial_flow_source"]),
        anchor_long_feature_domain=str(anchor_long["feature_domain"]),
        anchor_long_feature_slit_width_px=int(anchor_long["feature_slit_width_px"]),
        anchor_long_preprocessing=str(anchor_long["preprocessing"]),
        anchor_long_lk_window_size=int(anchor_long.get("lk_window_size", 31)),
        anchor_long_lk_max_level=int(anchor_long.get("lk_max_level", 4)),
        huber_delta_px=float(solver["huber_delta_px"]),
        irls_iterations=int(solver["irls_iterations"]),
        minimum_increment_px=float(solver["minimum_increment_px"]),
        maximum_reliable_edge_adjustment_px=float(
            solver["maximum_reliable_edge_adjustment_px"]
        ),
    ).validated()


def _anchor_config(config: S12Config) -> S12AnchorConfig:
    values = config.section("anchors")
    return S12AnchorConfig(
        target_spacing_px=float(values["target_spacing_px"]),
        minimum_spacing_px=float(values["minimum_spacing_px"]),
        maximum_spacing_px=float(values["maximum_spacing_px"]),
        target_search_radius_px=float(values["target_search_radius_px"]),
        minimum_anchor_edge_confidence=float(values["minimum_anchor_edge_confidence"]),
        minimum_anchor_image_quality=float(values["minimum_anchor_image_quality"]),
        maximum_direct_local_path_difference_px=float(
            values.get("maximum_direct_local_path_difference_px", 2.0)
        ),
    ).validated()


def _expected_motion_sign(session: S12Session, segment: Any) -> int:
    first = session.trajectory.by_frame_id()[segment.start_frame_id]
    assert first.camera_to_world is not None
    rail_world = np.asarray(segment.rail_axis, dtype=np.float64)
    rail_camera = first.camera_to_world[:3, :3].T @ rail_world
    if abs(float(rail_camera[0])) < 1e-6:
        raise RuntimeError("S1.2 structural fatal: rail motion has no auditable camera-x direction")
    # A static scene moves opposite to camera translation in image x.
    return -1 if rail_camera[0] > 0.0 else 1


def _image_quality(gray: np.ndarray) -> float:
    sharpness = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    contrast = float(gray.std())
    return float(np.clip(0.5 * sharpness / (sharpness + 80.0) + 0.5 * contrast / 64.0, 0, 1))


def _build_nodes(session: S12Session, segment: Any, motion: S12MotionGraphConfig):
    frame_by_id = {frame.frame_id: frame for frame in session.video.rgbd.frames}
    audit_by_id = {row.frame_id: row for row in segment.pose_audit}
    calibration = session.video.rgbd.calibration
    nodes: list[S12MotionNode] = []
    target_matrix: np.ndarray | None = None
    for node_index, frame_id in enumerate(segment.candidate_frame_ids):
        frame = frame_by_id[frame_id]
        image = cv2.imread(str(frame.color_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"S1.2 RGB source cannot be decoded: {frame.color_path}")
        gray, current_matrix = calibrate_analysis_image(
            image,
            calibration.matrix,
            np.asarray(calibration.distortion, dtype=np.float64),
            analysis_width_px=motion.analysis_width_px,
        )
        if target_matrix is None:
            target_matrix = current_matrix
        elif not np.allclose(target_matrix, current_matrix):
            raise RuntimeError("S1.2 calibrated analysis domains are inconsistent")
        nodes.append(
            S12MotionNode(
                node_index=node_index,
                frame_id=frame_id,
                calibrated_gray=gray,
                pose_progress_m=float(audit_by_id[frame_id].progress_m),
                image_quality=_image_quality(gray),
                timestamp_ms=(
                    None if frame.timestamp_us is None else float(frame.timestamp_us) / 1000.0
                ),
                calibrated_cx=float(current_matrix[0, 2]),
            )
        )
    if target_matrix is None:
        raise RuntimeError("S1.2 structural fatal: scan selected no motion nodes")
    return tuple(nodes), target_matrix


def _build_anchor_evidence(
    session: S12Session,
    segment: Any,
    graph: Any,
    scan_config: Any | None = None,
) -> tuple[S12AnchorEvidence, ...]:
    """Bind anchor eligibility to pose, scan, and independent edge evidence."""
    qualification_by_id = {row.frame_id: row for row in session.pose_qualifications}
    scan_by_id = {row.frame_id: row for row in segment.pose_audit}
    progress = [float(node.pose_progress_m) for node in graph.nodes]
    direct_pose_qualified: list[bool] = []
    attitude_qualified: list[bool] = []
    paused: list[bool] = []
    backward: list[bool] = []
    maximum_backward_step_m = (
        0.0
        if scan_config is None
        else float(scan_config.maximum_backward_step_m)
    )
    paused_nodes: set[int] = set()
    if scan_config is not None:
        for index in range(1, len(progress)):
            if progress[index] - progress[index - 1] <= scan_config.maximum_backward_step_m:
                paused_nodes.update((index - 1, index))
    for index, node in enumerate(graph.nodes):
        qualification = qualification_by_id[node.frame_id]
        scan = scan_by_id[node.frame_id]
        neighbours = []
        if index:
            neighbours.append(progress[index] - progress[index - 1])
        if index + 1 < len(progress):
            neighbours.append(progress[index + 1] - progress[index])
        direct_pose_qualified.append(bool(qualification.direct_pose))
        attitude_qualified.append(
            bool(
                not any(
                    reason in {"roll_limit", "pitch_limit", "yaw_limit", "rotation_limit"}
                    for reason in qualification.rejection_reasons
                )
            )
        )
        # The selected scan has already applied the configured bounded pause
        # span. Do not invent a stricter per-node epsilon pause rule here.
        paused.append(
            index in paused_nodes or "paused_motion" in scan.rejection_reasons
        )
        backward.append(
            bool(
                any(delta < -maximum_backward_step_m for delta in neighbours)
                or any(
                    reason in {"backward_motion", "non_monotonic_progress"}
                    for reason in scan.rejection_reasons
                )
            )
        )
    return build_anchor_evidence(
        graph,
        direct_pose_qualified=direct_pose_qualified,
        attitude_qualified=attitude_qualified,
        paused=paused,
        backward=backward,
    )


def _write_motion_diagnostics(
    output: Path,
    *,
    graph: Any,
    evidence: Sequence[S12AnchorEvidence],
) -> None:
    edge_rows = [edge.as_dict() for edge in graph.edges]
    edge_fields = tuple(edge_rows[0]) if edge_rows else (
        "left_node", "right_node", "left_frame_id", "right_frame_id", "kind",
        "reliable", "rejection_reasons",
    )
    _write_csv(output / "edge_diagnostics.csv", edge_rows, edge_fields)
    _write_csv(output / "motion_edges.csv", edge_rows, edge_fields)
    try:
        graph.assert_reliable_connected()
        graph_connected = True
    except S12MotionGraphError:
        graph_connected = False
    _write_json(
        output / "motion_graph.json",
        {
            "coordinate_domain": graph.coordinate_domain,
            "nodes": [
                {
                    "node_index": node.node_index,
                    "frame_id": node.frame_id,
                    "pose_progress_m": node.pose_progress_m,
                    "image_quality": node.image_quality,
                }
                for node in graph.nodes
            ],
            "edges": edge_rows,
            "connected": graph_connected,
        },
    )
    reason_counts: dict[str, int] = {}
    overlap_counts: dict[str, int] = {}
    for edge in graph.edges:
        reasons = tuple(edge.rejection_reasons)
        for reason in reasons:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        for left in range(len(reasons)):
            for right in range(left + 1, len(reasons)):
                key = "+".join(sorted((reasons[left], reasons[right])))
                overlap_counts[key] = overlap_counts.get(key, 0) + 1
    _write_json(
        output / "edge_rejection_summary.json",
        {
            "edge_count": len(graph.edges),
            "rejected_edge_count": sum(not edge.reliable for edge in graph.edges),
            "reason_counts": reason_counts,
            "overlapping_reason_counts": overlap_counts,
        },
    )
    reliable_incident = [0] * len(graph.nodes)
    rejected_by_node: list[set[str]] = [set() for _ in graph.nodes]
    for edge in graph.edges:
        for node_index in (edge.left_node, edge.right_node):
            if edge.reliable:
                reliable_incident[node_index] += 1
            else:
                rejected_by_node[node_index].update(edge.rejection_reasons)
    evidence_rows = []
    for index, (node, item) in enumerate(zip(graph.nodes, evidence)):
        row = {"node_index": index, "frame_id": node.frame_id, **item.as_dict()}
        row["reliable_local_edge_indices"] = ";".join(
            str(value) for value in item.reliable_local_edge_indices
        )
        row["severe_motion_cluster_incident_edge_indices"] = ";".join(
            str(value) for value in item.severe_motion_cluster_incident_edge_indices
        )
        row["rejection_reasons"] = ";".join(item.rejection_reasons)
        row["rejected_incident_reasons"] = ";".join(sorted(rejected_by_node[index]))
        evidence_rows.append(row)
    _write_csv(output / "anchor_evidence.csv", evidence_rows, tuple(evidence_rows[0]))
    _write_csv(
        output / "anchor_candidate_diagnostics.csv",
        (),
        (
            "left_node", "right_node", "left_frame_id", "right_frame_id", "kind",
            "frame_gap", "timestamp_gap_ms", "provisional_spacing_px",
            "predicted_overlap_fraction", "direct_image_match", "reliable",
            "confidence", "measured_dx_px", "rejection_reasons",
        ),
    )
    endpoint_indices = sorted(set((0, len(graph.nodes) - 1, *range(max(0, len(graph.nodes) - 20), len(graph.nodes)))))
    _write_json(
        output / "endpoint_support_diagnostics.json",
        {
            "nodes": [
                {
                    "node_index": index,
                    "frame_id": graph.nodes[index].frame_id,
                    "reliable_incident_edge_count": reliable_incident[index],
                    "rejected_incident_reasons": sorted(rejected_by_node[index]),
                }
                for index in endpoint_indices
            ]
        },
    )
    _write_json(
        output / "motion_graph_diagnostics.json",
        {
            "node_count": len(graph.nodes),
            "edge_count": len(graph.edges),
            "reliable_edge_count": sum(edge.reliable for edge in graph.edges),
            "edge_kind_counts": {
                kind: sum(edge.kind == kind for edge in graph.edges)
                for kind in ("adjacent", "skip", "anchor_long")
            },
            "rejection_reason_counts": reason_counts,
        },
    )


def _write_early_session_audit(
    output: Path,
    *,
    config: S12Config,
    session: S12Session,
    trajectory_path: Path,
) -> None:
    qualification_rows = [
        {
            "frame_id": row.frame_id,
            "timestamp_us": row.timestamp_us,
            "direct_pose": row.direct_pose,
            "qualified": row.qualified,
            "timestamp_delta_ms": row.timestamp_delta_ms,
            "roll_deg": row.roll_deg,
            "pitch_deg": row.pitch_deg,
            "yaw_deg": row.yaw_deg,
            "rejection_reasons": ";".join(row.rejection_reasons),
        }
        for row in session.pose_qualifications
    ]
    _write_csv(output / "pose_qualification.csv", qualification_rows, tuple(qualification_rows[0]))
    _write_json(
        output / "trajectory_audit.json",
        {
            "path": str(trajectory_path.resolve()),
            "sha256": _sha256(trajectory_path),
            "schema": session.trajectory.schema,
            "pose_origin": session.trajectory.pose_origin,
            "direct_pose_count": len(session.trajectory.direct_records),
        },
    )
    (output / "config_snapshot.yaml").write_text(
        yaml.safe_dump(_plain(dict(config.raw)), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def _publish(staging: Path, output: Path) -> Path | None:
    backup = output.with_name(f".{output.name}.backup-{uuid.uuid4().hex}")
    moved = False
    try:
        if output.exists():
            os.replace(output, backup)
            moved = True
        os.replace(staging, output)
    except BaseException:
        if moved and backup.exists() and not output.exists():
            os.replace(backup, output)
        raise
    return backup if moved else None


def _load_validation_regions(run_name: str) -> Mapping[str, Any]:
    path = _validation_regions_path()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "gemini305-video-s12-validation-regions/v1":
        raise ValueError("S1.2 validation-region schema is invalid")
    runs = payload.get("runs")
    if not isinstance(runs, Mapping):
        raise ValueError("S1.2 validation regions require a runs mapping")
    selected = runs.get(run_name, {})
    if not isinstance(selected, Mapping):
        raise ValueError("S1.2 run validation regions must be a mapping")
    return selected


def _write_region_crops(
    output: Path,
    render: Any,
    schedule: Any,
    graph: Any,
    full_centers: Sequence[float],
    session: S12Session,
    calibration: Any,
    run_name: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    regions = _load_validation_regions(run_name)
    center_by_frame = {
        node.frame_id: float(full_centers[index]) for index, node in enumerate(graph.nodes)
    }
    frame_by_id = {frame.frame_id: frame for frame in session.video.rgbd.frames}
    records: list[dict[str, Any]] = []
    failures: list[str] = []
    fixed = output / "fixed_regions"
    worst = output / "worst_regions"
    fixed.mkdir(parents=True, exist_ok=True)
    worst.mkdir(parents=True, exist_ok=True)
    for name, value in regions.items():
        if not isinstance(value, Mapping):
            raise ValueError(f"S1.2 validation region {name!r} must be an object")
        frame_id = int(value["reference_frame_id"])
        bbox = [int(item) for item in value["source_bbox"]]
        if len(bbox) != 4 or bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
            raise ValueError(f"S1.2 validation region {name!r} has an invalid bbox")
        center = center_by_frame.get(frame_id)
        frame = frame_by_id.get(frame_id)
        if center is None or frame is None:
            failures.append(f"fixed_region_missing_full_solution:{name}:{frame_id}")
            records.append({"name": name, "reference_frame_id": frame_id, "available": False})
            continue
        x0 = int(np.floor(center + bbox[0] - calibration.cx))
        x1 = int(np.ceil(center + bbox[2] - calibration.cx))
        y0, y1 = bbox[1], bbox[3]
        x0, x1 = max(0, x0), min(render.image.shape[1], x1)
        y0, y1 = max(0, y0), min(render.image.shape[0], y1)
        if x0 >= x1 or y0 >= y1:
            failures.append(f"fixed_region_outside_canvas:{name}")
            records.append({"name": name, "reference_frame_id": frame_id, "available": False})
            continue
        region_dir = fixed / name
        region_dir.mkdir(parents=True, exist_ok=True)
        reference = cv2.imread(str(frame.color_path), cv2.IMREAD_COLOR)
        if reference is None:
            raise OSError(f"Failed to decode S1.2 fixed-region reference {frame_id}")
        rx0, ry0, rx1, ry1 = bbox
        rx0, rx1 = max(0, rx0), min(reference.shape[1], rx1)
        ry0, ry1 = max(0, ry0), min(reference.shape[0], ry1)
        if not cv2.imwrite(str(region_dir / "reference.png"), reference[ry0:ry1, rx0:rx1]):
            raise OSError(f"Failed to write S1.2 fixed-region reference {name}")
        if not cv2.imwrite(str(region_dir / "panorama.png"), render.image[y0:y1, x0:x1]):
            raise OSError(f"Failed to write S1.2 fixed-region panorama {name}")
        owners = render.owner_frame_id[y0:y1, x0:x1]
        overlay = np.zeros((*owners.shape, 3), dtype=np.uint8)
        owner_ids = sorted(int(item) for item in np.unique(owners) if item >= 0)
        for owner_id in owner_ids:
            overlay[owners == owner_id] = (
                (owner_id * 37 + 53) % 256,
                (owner_id * 67 + 97) % 256,
                (owner_id * 101 + 193) % 256,
            )
        if not cv2.imwrite(str(region_dir / "owner_overlay.png"), overlay):
            raise OSError(f"Failed to write S1.2 fixed-region owner overlay {name}")
        column_owners = render.column_frame_id[x0:x1]
        switch_columns = [
            x0 + index
            for index in range(1, len(column_owners))
            if int(column_owners[index]) != int(column_owners[index - 1])
        ]
        provenance = {
            "name": name,
            "reference_frame_id": frame_id,
            "source_bbox": bbox,
            "canvas_bbox": [x0, y0, x1, y1],
            "owner_frame_ids": owner_ids,
            "source_frame_ids": sorted(
                int(item.frame_id)
                for item in schedule.assignments
                if item.right_x > x0 and item.left_x < x1 and not item.zero_width
            ),
            "owner_switch_columns": switch_columns,
            "manual_review": {"status": "pending"},
            "lock_eligible": False,
        }
        _write_json(region_dir / "provenance.json", provenance)
        records.append(
            {
                "name": name,
                "reference_frame_id": frame_id,
                "source_bbox": bbox,
                "canvas_bbox": [x0, y0, x1, y1],
                "available": True,
                "directory": f"fixed_regions/{name}",
                "manual_review": {"status": "pending"},
                "lock_eligible": False,
            }
        )
    internal = sorted(schedule.assignments[1:-1], key=lambda item: item.width, reverse=True)[:5]
    for rank, item in enumerate(internal):
        x0, x1 = max(0, item.left_x - 48), min(render.image.shape[1], item.right_x + 48)
        if not cv2.imwrite(str(worst / f"width_{rank:02d}_frame_{item.frame_id}.png"), render.image[:, x0:x1]):
            raise OSError("Failed to write S1.2 worst-region crop")
    _write_json(fixed / "manifest.json", {"regions": records})
    if failures:
        raise RuntimeError(
            "S1.2 structural fatal: required fixed regions are incomplete: "
            + ", ".join(failures)
        )
    return records, failures


def _write_bundle(
    output: Path,
    *,
    config: S12Config,
    config_path: Path,
    session: S12Session,
    segment: Any,
    graph: Any,
    anchors: Any,
    full_centers: Sequence[float],
    selected: Sequence[Any],
    schedule: Any,
    render: Any,
    open3d_rows: Sequence[Mapping[str, Any]],
    stage_seconds: Mapping[str, float],
    trajectory_path: Path,
) -> dict[str, Any]:
    if not cv2.imwrite(str(output / "stage_a_dense_slit.png"), render.image):
        raise OSError("Failed to write S1.2 Stage A image")
    np.savez_compressed(output / "owner_frame_id.npz", owner_frame_id=render.owner_frame_id)
    if not cv2.imwrite(str(output / "valid_mask.png"), render.valid_mask.astype(np.uint8) * 255):
        raise OSError("Failed to write S1.2 valid mask")
    np.savez_compressed(
        output / "source_uv.npz",
        source_u=render.source_u,
        source_v=render.source_v,
        assignment_index=render.assignment_index,
    )
    np.savez_compressed(
        output / "column_provenance.npz",
        column_frame_id=render.column_frame_id,
        column_source_u=render.column_source_u,
        column_assignment_index=render.column_assignment_index,
    )

    qualification_rows = [
        {
            "frame_id": row.frame_id,
            "timestamp_us": row.timestamp_us,
            "direct_pose": row.direct_pose,
            "qualified": row.qualified,
            "timestamp_delta_ms": row.timestamp_delta_ms,
            "roll_deg": row.roll_deg,
            "pitch_deg": row.pitch_deg,
            "yaw_deg": row.yaw_deg,
            "rejection_reasons": ";".join(row.rejection_reasons),
        }
        for row in session.pose_qualifications
    ]
    _write_csv(output / "pose_qualification.csv", qualification_rows, tuple(qualification_rows[0]))
    _write_json(output / "scan_segment.json", segment.as_dict())
    edge_rows = [edge.as_dict() for edge in graph.edges]
    _write_csv(output / "motion_edges.csv", edge_rows, tuple(edge_rows[0]))
    _write_json(
        output / "motion_graph.json",
        {
            "coordinate_domain": graph.coordinate_domain,
            "nodes": [
                {
                    "node_index": node.node_index,
                    "frame_id": node.frame_id,
                    "pose_progress_m": node.pose_progress_m,
                    "image_quality": node.image_quality,
                }
                for node in graph.nodes
            ],
            "edges": edge_rows,
            "connected": True,
        },
    )
    _write_json(output / "anchor_selection.json", anchors.as_dict())
    anchor_rows = [edge.as_dict() for edge in anchors.long_edges]
    _write_csv(output / "anchor_edges.csv", anchor_rows, tuple(anchor_rows[0]))
    horizontal_rows = [
        {
            "source_index": index,
            "frame_id": graph.nodes[index].frame_id,
            "center_x": full_centers[index],
            "increment_from_previous_px": 0.0 if index == 0 else full_centers[index] - full_centers[index - 1],
        }
        for index in range(len(full_centers))
    ]
    _write_csv(output / "horizontal_solution.csv", horizontal_rows, tuple(horizontal_rows[0]))
    source_rows = [_plain(item) for item in selected]
    _write_csv(output / "source_selection.csv", source_rows, tuple(source_rows[0]))
    _write_json(output / "open3d_edge_audit.json", {"edges": open3d_rows})
    schedule_rows = [asdict(item) for item in schedule.assignments]
    _write_csv(output / "strip_schedule.csv", schedule_rows, tuple(schedule_rows[0]))
    region_rows, region_failures = _write_region_crops(
        output,
        render,
        schedule,
        graph,
        full_centers,
        session,
        session.video.rgbd.calibration,
        session.root.name,
    )

    increments = np.diff(np.asarray([item.center_x for item in selected], dtype=np.float64))
    internal_widths = [item.width for item in schedule.assignments[1:-1] if item.width]
    valid_fraction = float(np.mean(render.valid_mask))
    acceptance_reasons = list(region_failures)
    minimum_valid = float(config.section("acceptance")["minimum_valid_pixel_fraction"])
    if valid_fraction < minimum_valid:
        acceptance_reasons.append("minimum_valid_pixel_fraction")
    performance = {
        "stage_seconds": dict(stage_seconds),
        "algorithm_seconds": float(sum(stage_seconds.values())),
        "audit_export_seconds": 0.0,
        "atomic_publish_seconds": 0.0,
        "final_wall_seconds": 0.0,
        "slowest_stage": max(stage_seconds, key=stage_seconds.get),
        "slowest_stage_seconds": float(max(stage_seconds.values())),
        "target_seconds": 20.0,
        "target_pass": False,
        "runtime_finalized": False,
        "decode_count": len(render.decoded_frame_ids),
        "remap_invocations": render.remap_invocations,
    }
    report = {
        "schema": REPORT_SCHEMA,
        "algorithm_id": S12_ALGORITHM_ID,
        "stage": "a",
        "diagnostic_only": True,
        "production_eligible": False,
        "structural_valid": True,
        "code_provenance": _git_provenance(),
        "input": str(session.root),
        "trajectory": {
            "path": str(trajectory_path.resolve()),
            "hash": _sha256(trajectory_path),
            "schema": session.trajectory.schema,
            "pose_origin": session.trajectory.pose_origin,
            "direct_pose_count": len(session.trajectory.direct_records),
        },
        "scan": segment.as_dict(),
        "motion_graph": {
            "reliability_gate_mode": "split_v1",
            "reliability_gates": {
                "minimum_model_inlier_count": 16,
                "minimum_fb_retention_ratio": 0.45,
                "minimum_model_inlier_ratio_retained": 0.45,
                "minimum_effective_inlier_ratio_detected": 0.25,
                "maximum_parallax_cluster_ratio": 0.45,
            },
            "compatibility_only_fields": {
                "minimum_inlier_count": 16,
                "minimum_inlier_ratio": 0.45,
            },
            "node_count": len(graph.nodes),
            "adjacent_edge_count": sum(edge.kind == "adjacent" for edge in graph.edges),
            "skip_edge_count": sum(edge.kind == "skip" for edge in graph.edges),
            "anchor_edge_count": len(anchors.long_edges),
            "rejected_edge_count": sum(not edge.reliable for edge in graph.edges),
            "connected": True,
        },
        "anchors": {
            "count": len(anchors.anchor_nodes),
            "frame_ids": [graph.nodes[index].frame_id for index in anchors.anchor_nodes],
            "spacing": percentiles(
                [full_centers[right] - full_centers[left] for left, right in zip(anchors.anchor_nodes, anchors.anchor_nodes[1:])]
            ),
            "all_direct_edges_pass": all(edge.reliable for edge in anchors.long_edges),
            "all_two_path_checks_pass": all(bool(item.local_path_edge_indices) for item in anchors.interval_audits),
        },
        "horizontal": {
            "source_count": len(selected),
            "center_increment": percentiles(increments.tolist()),
            "edge_residual_p95_px": float(np.percentile(np.abs(anchors.final_solution.edge_residuals_px), 95)),
            "global_rescale_applied": False,
            "median_filter_applied": False,
        },
        "schedule": {
            "canvas_width": schedule.canvas_width,
            "canvas_height": schedule.canvas_height,
            "nonzero_source_count": sum(not item.zero_width for item in schedule.assignments),
            "internal_width": percentiles(internal_widths),
            "preferred_width_exceed_count": sum(width > int(config.section("source_selection")["preferred_internal_strip_width_px"]) for width in internal_widths),
            "hard_width_exceed_count": sum(width > int(config.section("source_selection")["hard_internal_strip_width_px"]) for width in internal_widths),
            "unassigned_internal_columns": 0,
            "multiply_assigned_columns": 0,
        },
        "open3d": {
            "edge_count": len(open3d_rows),
            "all_adjacent_final_edges_pass": all(bool(row["quality_pass"]) for row in open3d_rows),
        },
        "provenance": {
            "valid_pixel_fraction": valid_fraction,
            "duplicate_write_count": render.duplicate_write_count,
            "consistency_pass": bool(np.array_equal(render.valid_mask, render.owner_frame_id >= 0)),
        },
        "fixed_regions": region_rows,
        "manual_review": {"status": "pending"},
        "lock_eligible": False,
        "vertical": {},
        "heldout_tracks": {},
        "performance": performance,
        "acceptance": {
            "pass": not acceptance_reasons,
            "structural_fatal": False,
            "reasons": acceptance_reasons,
        },
    }
    _write_json(output / "performance.json", performance)
    _write_json(output / "report.json", report)
    _write_json(
        output / "trajectory_audit.json",
        {
            "path": str(trajectory_path.resolve()),
            "sha256": _sha256(trajectory_path),
            "schema": session.trajectory.schema,
            "pose_origin": session.trajectory.pose_origin,
            "direct_pose_count": len(session.trajectory.direct_records),
        },
    )
    (output / "config_snapshot.yaml").write_text(
        yaml.safe_dump(_plain(dict(config.raw)), sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    return report


def _run_stage_a(
    input_path: Path,
    trajectory_path: Path,
    config_path: Path,
    staging: Path,
    *,
    output_mode: str | None,
    open3d_backend: Any | None,
) -> dict[str, Any]:
    started = time.perf_counter()
    stages: dict[str, float] = {}
    _write_json(staging / "_run_state.json", {"failed_stage": "session_and_trajectory"})
    tick = time.perf_counter()
    config = load_s12_config(config_path)
    if output_mode is not None and output_mode != config.section("output")["mode"]:
        raw = _plain(dict(config.raw))
        raw["output"]["mode"] = output_mode
        config = load_s12_config(raw)
    session = load_s12_session(input_path, trajectory_path, config, validation_workers=4)
    _write_early_session_audit(
        staging,
        config=config,
        session=session,
        trajectory_path=trajectory_path,
    )
    stages["session_and_trajectory"] = time.perf_counter() - tick

    tick = time.perf_counter()
    _write_json(staging / "_run_state.json", {"failed_stage": "scan_and_motion_graph"})
    segment = select_s12_scan_segment(session.pose_qualifications, config.scan)
    motion_config = _motion_config(config)
    nodes, analysis_matrix = _build_nodes(session, segment, motion_config)
    motion_sign = _expected_motion_sign(session, segment)
    graph = build_motion_graph(
        nodes,
        expected_image_motion_sign=motion_sign,
        config=motion_config,
        require_connected=False,
    )
    evidence = _build_anchor_evidence(session, segment, graph, config.scan)
    _write_motion_diagnostics(staging, graph=graph, evidence=evidence)
    _write_json(staging / "initial_scan_segment.json", segment.as_dict())
    _write_json(staging / "scan_segment.json", segment.as_dict())
    direction = {
        graph.nodes[edge.right_node].frame_id: bool(
            edge.pose_direction_agreement and edge.reliable
        )
        for edge in graph.edges
        if edge.kind == "adjacent"
    }
    checked_segment = select_s12_scan_segment(
        session.pose_qualifications, config.scan, image_motion_agreement=direction
    )
    _write_json(staging / "checked_scan_segment.json", checked_segment.as_dict())
    _write_json(staging / "scan_segment.json", checked_segment.as_dict())
    if checked_segment.candidate_frame_ids != segment.candidate_frame_ids:
        segment = checked_segment
        nodes, analysis_matrix = _build_nodes(session, segment, motion_config)
        graph = build_motion_graph(
            nodes, expected_image_motion_sign=motion_sign, config=motion_config
        )
    else:
        segment = checked_segment
        graph.assert_reliable_connected()
    evidence = _build_anchor_evidence(session, segment, graph, config.scan)
    _write_motion_diagnostics(staging, graph=graph, evidence=evidence)
    _write_json(staging / "scan_segment.json", segment.as_dict())
    stages["scan_and_motion_graph"] = time.perf_counter() - tick

    tick = time.perf_counter()
    _write_json(staging / "_run_state.json", {"failed_stage": "anchors_horizontal_schedule"})
    provisional_for_long = solve_horizontal_centers(
        graph,
        calibrated_cx=float(analysis_matrix[0, 2]),
        config=motion_config,
    )
    anchor_candidate_rows: list[dict[str, Any]] = []

    def estimate_anchor_candidate(left: int, right: int) -> Any:
        direct = estimate_direct_motion_edge(
            graph.nodes[left], graph.nodes[right], kind="anchor_long",
            expected_image_motion_sign=motion_sign, config=motion_config,
            provisional_spacing_px=(
                provisional_for_long.centers_px[right]
                - provisional_for_long.centers_px[left]
            ),
        )
        row = direct.as_dict()
        row["rejection_reasons"] = ";".join(direct.rejection_reasons)
        anchor_candidate_rows.append(row)
        _write_csv(
            staging / "anchor_candidate_diagnostics.csv",
            anchor_candidate_rows,
            tuple(anchor_candidate_rows[0]),
        )
        return direct

    anchors = select_automatic_anchors(
        graph,
        evidence,
        calibrated_cx=float(analysis_matrix[0, 2]),
        long_edge_estimator=estimate_anchor_candidate,
        anchor_config=_anchor_config(config),
        solver_config=motion_config,
        full_resolution_scale=(
            session.video.rgbd.calibration.width / float(motion_config.analysis_width_px)
        ),
    )
    graph = graph.with_edges(anchors.long_edges)
    scale = session.video.rgbd.calibration.width / float(motion_config.analysis_width_px)
    analysis_cx = float(analysis_matrix[0, 2])
    full_cx = float(session.video.rgbd.calibration.cx)
    full_centers = tuple(
        full_cx + (value - analysis_cx) * scale for value in anchors.final_solution.centers_px
    )
    source_cfg = config.section("source_selection")
    selected = list(select_unique_sources(
        [node.frame_id for node in graph.nodes],
        full_centers,
        anchor_frame_ids=[graph.nodes[index].frame_id for index in anchors.anchor_nodes],
        image_quality_by_frame={node.frame_id: node.image_quality for node in graph.nodes},
        duplicate_tolerance_px=float(source_cfg["duplicate_center_tolerance_px"]),
    ))
    schedule = build_s012_schedule(
        [item.frame_id for item in selected],
        [item.center_x for item in selected],
        session.video.rgbd.calibration,
        hard_internal_width_px=int(source_cfg["hard_internal_strip_width_px"]),
    )
    stages["anchors_horizontal_schedule"] = time.perf_counter() - tick

    tick = time.perf_counter()
    _write_json(staging / "_run_state.json", {"failed_stage": "open3d_final_edge_audit"})
    frame_by_id = {frame.frame_id: frame for frame in session.video.rgbd.frames}
    pose_by_id = {
        record.frame_id: record.camera_to_world for record in session.trajectory.direct_records
    }
    anchor_ids = {graph.nodes[index].frame_id for index in anchors.anchor_nodes}
    maximum_pruning_rounds = config.trajectory.maximum_source_pruning_rounds
    for pruning_round in range(maximum_pruning_rounds + 1):
        selected_frames = [frame_by_id[item.frame_id] for item in selected]
        selected_poses = [pose_by_id[item.frame_id] for item in selected]
        open3d_rows = audit_open3d_source_edges(
            selected_frames,
            selected_poses,
            session.video.rgbd.calibration,
            backend=open3d_backend,
        )
        failed = [row for row in open3d_rows if not row["quality_pass"]]
        if not failed:
            break
        if pruning_round >= maximum_pruning_rounds:
            pairs = [f"{row['reference_node_id']}->{row['source_node_id']}" for row in failed]
            raise RuntimeError(
                "S1.2 structural fatal: final Open3D edge audit failed after bounded pruning: "
                + ", ".join(pairs)
            )
        failed_ids = {
            int(value)
            for row in failed
            for value in (row["reference_node_id"], row["source_node_id"])
        }
        removable = [
            item
            for item in selected[1:-1]
            if item.frame_id in failed_ids and item.frame_id not in anchor_ids
        ]
        if not removable:
            raise RuntimeError(
                "S1.2 structural fatal: failed Open3D edge is anchor/end-point constrained"
            )
        victim = min(removable, key=lambda item: (item.image_quality, item.frame_id))
        selected = [item for item in selected if item.frame_id != victim.frame_id]
        # Rebuilding the immutable schedule here ensures pruning never creates
        # a >24 px internal slit or leaves an unassigned column.
        schedule = build_s012_schedule(
            [item.frame_id for item in selected],
            [item.center_x for item in selected],
            session.video.rgbd.calibration,
            hard_internal_width_px=int(source_cfg["hard_internal_strip_width_px"]),
        )
    stages["open3d_final_edge_audit"] = time.perf_counter() - tick

    tick = time.perf_counter()
    _write_json(staging / "_run_state.json", {"failed_stage": "stage_a_roi_remap"})
    render = render_s012_stage_a(
        schedule,
        session.video.rgbd.calibration,
        lambda frame_id: cv2.imread(str(frame_by_id[frame_id].color_path), cv2.IMREAD_COLOR),
    )
    stages["stage_a_roi_remap"] = time.perf_counter() - tick
    tick = time.perf_counter()
    _write_json(staging / "_run_state.json", {"failed_stage": "audit_export"})
    algorithm_seconds = time.perf_counter() - started
    report = _write_bundle(
        staging,
        config=config,
        config_path=config_path,
        session=session,
        segment=segment,
        graph=graph,
        anchors=anchors,
        full_centers=full_centers,
        selected=selected,
        schedule=schedule,
        render=render,
        open3d_rows=open3d_rows,
        stage_seconds=stages,
        trajectory_path=trajectory_path,
    )
    stages["audit_export"] = time.perf_counter() - tick
    report["performance"]["algorithm_seconds"] = algorithm_seconds
    report["performance"]["audit_export_seconds"] = stages["audit_export"]
    report["performance"]["stage_seconds"] = dict(stages)
    report["performance"]["slowest_stage"] = max(stages, key=stages.get)
    report["performance"]["slowest_stage_seconds"] = float(max(stages.values()))
    _write_json(staging / "performance.json", report["performance"])
    _write_json(staging / "report.json", report)
    (staging / "_run_state.json").unlink(missing_ok=True)
    return report


def _finalize_completion(
    output: Path,
    report: dict[str, Any],
    *,
    run_started: float,
    atomic_publish_seconds: float,
) -> dict[str, Any]:
    performance = report["performance"]
    performance["atomic_publish_seconds"] = float(atomic_publish_seconds)
    performance["final_wall_seconds"] = 0.0
    performance["runtime_finalized"] = False
    performance["target_pass"] = False
    report["lock_eligible"] = False
    report["manual_review"] = {"status": "pending"}
    _atomic_write_json(output / "performance.json", performance)
    _atomic_write_json(output / "report.json", report)
    report_hash = _sha256(output / "report.json")
    completion: dict[str, Any] = {
        "schema": COMPLETION_SCHEMA,
        "runtime_finalized": False,
        "algorithm_seconds": performance["algorithm_seconds"],
        "audit_export_seconds": performance["audit_export_seconds"],
        "atomic_publish_seconds": performance["atomic_publish_seconds"],
        "final_wall_seconds": 0.0,
        "target_seconds": performance["target_seconds"],
        "target_pass": False,
        "bundle_path": str(output),
        "bundle_report_sha256": report_hash,
        "manual_review": {"status": "pending"},
        "lock_eligible": False,
    }
    # The first marker write is deliberately invalid but lets the final wall
    # include a real marker serialization and atomic replace. Only the second,
    # hash-bound runtime_finalized marker is the commit point.
    marker_started = time.perf_counter()
    _atomic_write_json(output / "S012_stage_a_completion.json", completion)
    marker_finished = time.perf_counter()
    performance["completion_marker_seconds"] = float(marker_finished - marker_started)
    performance["final_wall_seconds"] = float(marker_finished - run_started)
    performance["runtime_finalized"] = True
    performance["target_pass"] = bool(
        performance["final_wall_seconds"] <= performance["target_seconds"]
    )
    reasons = report["acceptance"]["reasons"]
    if not performance["target_pass"] and "stage_a_runtime_target" not in reasons:
        reasons.append("stage_a_runtime_target")
    report["acceptance"]["pass"] = not reasons
    _atomic_write_json(output / "performance.json", performance)
    _atomic_write_json(output / "report.json", report)
    report_hash = _sha256(output / "report.json")
    completion.update(
        {
            "runtime_finalized": True,
            "completion_marker_seconds": performance["completion_marker_seconds"],
            "final_wall_seconds": performance["final_wall_seconds"],
            "target_pass": performance["target_pass"],
            "bundle_report_sha256": report_hash,
        }
    )
    _atomic_write_json(output / "S012_stage_a_completion.json", completion)
    return report


def _publish_failure_bundle(
    staging: Path,
    *,
    output: Path,
    input_path: Path,
    trajectory_path: Path,
    config_path: Path,
    exc: BaseException,
    run_started: float,
) -> Path:
    failure_output = output.parent / f"{output.name}.failure_bundle"
    if not staging.exists():
        staging.mkdir(parents=True)
        for name in (
            "edge_diagnostics.csv",
            "edge_rejection_summary.json",
            "anchor_evidence.csv",
            "anchor_candidate_diagnostics.csv",
            "endpoint_support_diagnostics.json",
            "motion_graph_diagnostics.json",
            "motion_edges.csv",
            "motion_graph.json",
            "scan_segment.json",
            "pose_qualification.csv",
        ):
            source = output / name
            if source.is_file():
                shutil.copy2(source, staging / name)
    state_path = staging / "_run_state.json"
    failed_stage = "completion_marker"
    if state_path.is_file():
        try:
            failed_stage = str(json.loads(state_path.read_text(encoding="utf-8"))["failed_stage"])
        except (KeyError, TypeError, ValueError):
            failed_stage = "unknown"
    state_path.unlink(missing_ok=True)
    if config_path.is_file():
        shutil.copy2(config_path, staging / "config_snapshot.yaml")
    trajectory_hash = _sha256(trajectory_path) if trajectory_path.is_file() else None
    failure_payload = {
        "schema": FAILURE_SCHEMA,
        "algorithm_id": S12_ALGORITHM_ID,
        "stage": "a",
        "structural_fatal": True,
        "structural_valid": False,
        "production_eligible": False,
        "acceptance": {"pass": False, "structural_fatal": True},
        "lock_eligible": False,
        "error_type": type(exc).__name__,
        "message": str(exc),
        "input": str(input_path),
        "trajectory": {"path": str(trajectory_path), "sha256": trajectory_hash},
        "config": {"path": str(config_path), "sha256": _sha256(config_path) if config_path.is_file() else None},
        **_git_provenance(),
        "failed_stage": failed_stage,
        "wall_seconds_before_failure": float(time.perf_counter() - run_started),
    }
    _write_json(staging / "failure.json", failure_payload)
    failure_backup = _publish(staging, failure_output)
    if failure_backup is not None and failure_backup.exists():
        shutil.rmtree(failure_backup)
    return failure_output


def run(
    input_path: Path,
    *,
    trajectory_path: Path,
    config_path: Path | None = None,
    output: Path,
    stage: str = "a",
    stage_a_lock: Path | None = None,
    output_mode: str | None = None,
    open3d_backend: Any | None = None,
) -> dict[str, Any]:
    if stage != "a":
        raise ValueError("Stage B is intentionally not implemented before Stage A review")
    if stage_a_lock is not None:
        raise ValueError("--stage-a-lock is only valid for the future Stage B implementation")
    run_started = time.perf_counter()
    selected_config = (config_path or _default_config()).expanduser().resolve()
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(f".{output.name}.staging-{uuid.uuid4().hex}")
    failure = output.parent / f"{output.name}.failure.json"
    staging.mkdir()
    published_backup: Path | None = None
    try:
        report = _run_stage_a(
            input_path,
            trajectory_path,
            selected_config,
            staging,
            output_mode=output_mode,
            open3d_backend=open3d_backend,
        )
        publish_started = time.perf_counter()
        published_backup = _publish(staging, output)
        publish_seconds = time.perf_counter() - publish_started
        report = _finalize_completion(
            output,
            report,
            run_started=run_started,
            atomic_publish_seconds=publish_seconds,
        )
        if published_backup is not None and published_backup.exists():
            shutil.rmtree(published_backup)
        failure.unlink(missing_ok=True)
        return report
    except BaseException as exc:
        failure_staging = staging
        if not failure_staging.exists():
            failure_staging = output.with_name(
                f".{output.name}.failure-staging-{uuid.uuid4().hex}"
            )
            if output.exists():
                os.replace(output, failure_staging)
            if published_backup is not None and published_backup.exists():
                os.replace(published_backup, output)
        failure_bundle = _publish_failure_bundle(
            failure_staging,
            output=output,
            input_path=Path(input_path).expanduser().resolve(),
            trajectory_path=Path(trajectory_path).expanduser().resolve(),
            config_path=selected_config,
            exc=exc,
            run_started=run_started,
        )
        _write_json(
            failure,
            {
                "schema": FAILURE_SCHEMA,
                "algorithm_id": S12_ALGORITHM_ID,
                "stage": "a",
                "structural_fatal": True,
                "error_type": type(exc).__name__,
                "message": str(exc),
                "input": str(Path(input_path).expanduser().resolve()),
                "trajectory": str(Path(trajectory_path).expanduser().resolve()),
                "failure_bundle": str(failure_bundle),
            },
        )
        raise


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        run(
            args.input,
            trajectory_path=args.trajectory,
            config_path=args.config,
            output=args.output,
            stage=args.stage,
            stage_a_lock=args.stage_a_lock,
            output_mode=args.output_mode,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
