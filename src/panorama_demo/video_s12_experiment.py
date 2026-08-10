"""Atomic Stage-A runner for the standalone automatic-anchor S1.2 experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
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
    select_automatic_anchors,
)
from .video_s12_config import S12_ALGORITHM_ID, S12Config, load_s12_config
from .video_s12_metrics import (
    audit_open3d_source_edges,
    percentiles,
    select_unique_sources,
)
from .video_s12_motion_graph import (
    S12MotionGraphConfig,
    S12MotionNode,
    build_motion_graph,
    calibrate_analysis_image,
    estimate_direct_motion_edge,
)
from .video_s12_renderer import render_s012_stage_a
from .video_s12_scan import select_s12_scan_segment
from .video_s12_schedule import build_s012_schedule
from .video_s12_session import S12Session, load_s12_session


REPORT_SCHEMA = "gemini305-video-s12-standalone-report/v1"
FAILURE_SCHEMA = "gemini305-video-s12-failure/v1"


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


def _motion_config(config: S12Config) -> S12MotionGraphConfig:
    motion = config.section("motion_graph")
    solver = config.section("horizontal_solver")
    return S12MotionGraphConfig(
        analysis_width_px=int(motion["analysis_width_px"]),
        lk_forward_backward_max_px=float(motion["lk_forward_backward_max_px"]),
        ransac_residual_px=float(motion["ransac_residual_px"]),
        minimum_inlier_count=int(motion["minimum_inlier_count"]),
        minimum_inlier_ratio=float(motion["minimum_inlier_ratio"]),
        minimum_grid_coverage=float(motion["minimum_grid_coverage"]),
        minimum_vertical_span_fraction=float(motion["minimum_vertical_span_fraction"]),
        maximum_parallax_cluster_ratio=float(motion["maximum_parallax_cluster_ratio"]),
        adjacent_candidate_steps=tuple(int(value) for value in motion["adjacent_candidate_steps"]),
        skip_candidate_steps=tuple(int(value) for value in motion["skip_candidate_steps"]),
        maximum_direct_edge_frame_gap=int(motion["maximum_direct_edge_frame_gap"]),
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
            )
        )
    if target_matrix is None:
        raise RuntimeError("S1.2 structural fatal: scan selected no motion nodes")
    return tuple(nodes), target_matrix


def _publish(staging: Path, output: Path) -> None:
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
    if backup.exists():
        shutil.rmtree(backup)


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
    image: np.ndarray,
    schedule: Any,
    calibration: Any,
    run_name: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    regions = _load_validation_regions(run_name)
    center_by_frame = {item.frame_id: item.center_x for item in schedule.assignments}
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
        if center is None:
            failures.append(f"fixed_region_source_not_selected:{name}:{frame_id}")
            records.append({"name": name, "reference_frame_id": frame_id, "available": False})
            continue
        x0 = int(np.floor(center + bbox[0] - calibration.cx))
        x1 = int(np.ceil(center + bbox[2] - calibration.cx))
        y0, y1 = bbox[1], bbox[3]
        x0, x1 = max(0, x0), min(image.shape[1], x1)
        y0, y1 = max(0, y0), min(image.shape[0], y1)
        if x0 >= x1 or y0 >= y1:
            failures.append(f"fixed_region_outside_canvas:{name}")
            records.append({"name": name, "reference_frame_id": frame_id, "available": False})
            continue
        filename = f"{name}.png"
        if not cv2.imwrite(str(fixed / filename), image[y0:y1, x0:x1]):
            raise OSError(f"Failed to write S1.2 fixed region {name}")
        records.append(
            {
                "name": name,
                "reference_frame_id": frame_id,
                "source_bbox": bbox,
                "canvas_bbox": [x0, y0, x1, y1],
                "available": True,
                "file": f"fixed_regions/{filename}",
            }
        )
    internal = sorted(schedule.assignments[1:-1], key=lambda item: item.width, reverse=True)[:5]
    for rank, item in enumerate(internal):
        x0, x1 = max(0, item.left_x - 48), min(image.shape[1], item.right_x + 48)
        if not cv2.imwrite(str(worst / f"width_{rank:02d}_frame_{item.frame_id}.png"), image[:, x0:x1]):
            raise OSError("Failed to write S1.2 worst-region crop")
    _write_json(fixed / "manifest.json", {"regions": records})
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
        output, render.image, schedule, session.video.rgbd.calibration, session.root.name
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
        "total_seconds": float(sum(stage_seconds.values())),
        "slowest_stage": max(stage_seconds, key=stage_seconds.get),
        "slowest_stage_seconds": float(max(stage_seconds.values())),
        "target_seconds": 20.0,
        "target_pass": float(sum(stage_seconds.values())) <= 20.0,
        "decode_count": len(render.decoded_frame_ids),
        "remap_invocations": render.remap_invocations,
    }
    if not performance["target_pass"]:
        acceptance_reasons.append("stage_a_runtime_target")
    report = {
        "schema": REPORT_SCHEMA,
        "algorithm_id": S12_ALGORITHM_ID,
        "stage": "a",
        "diagnostic_only": True,
        "production_eligible": False,
        "structural_valid": True,
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
    tick = time.perf_counter()
    config = load_s12_config(config_path)
    if output_mode is not None and output_mode != config.section("output")["mode"]:
        raw = _plain(dict(config.raw))
        raw["output"]["mode"] = output_mode
        config = load_s12_config(raw)
    session = load_s12_session(input_path, trajectory_path, config, validation_workers=4)
    stages["session_and_trajectory"] = time.perf_counter() - tick

    tick = time.perf_counter()
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
    direction = {
        graph.nodes[edge.right_node].frame_id: bool(edge.pose_direction_agreement and edge.reliable)
        for edge in graph.edges
        if edge.kind == "adjacent"
    }
    checked_segment = select_s12_scan_segment(
        session.pose_qualifications, config.scan, image_motion_agreement=direction
    )
    if checked_segment.candidate_frame_ids != segment.candidate_frame_ids:
        segment = checked_segment
        nodes, analysis_matrix = _build_nodes(session, segment, motion_config)
        graph = build_motion_graph(nodes, expected_image_motion_sign=motion_sign, config=motion_config)
    else:
        segment = checked_segment
        graph.assert_reliable_connected()
    stages["scan_and_motion_graph"] = time.perf_counter() - tick

    tick = time.perf_counter()
    evidence = tuple(S12AnchorEvidence() for _ in nodes)
    anchors = select_automatic_anchors(
        graph,
        evidence,
        calibrated_cx=float(analysis_matrix[0, 2]),
        long_edge_estimator=lambda left, right: estimate_direct_motion_edge(
            graph.nodes[left], graph.nodes[right], kind="anchor_long",
            expected_image_motion_sign=motion_sign, config=motion_config,
        ),
        anchor_config=_anchor_config(config),
        solver_config=motion_config,
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
    render = render_s012_stage_a(
        schedule,
        session.video.rgbd.calibration,
        lambda frame_id: cv2.imread(str(frame_by_id[frame_id].color_path), cv2.IMREAD_COLOR),
    )
    stages["stage_a_roi_remap"] = time.perf_counter() - tick
    tick = time.perf_counter()
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
    report["performance"]["wall_seconds"] = time.perf_counter() - started
    _write_json(staging / "performance.json", report["performance"])
    _write_json(staging / "report.json", report)
    return report


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
    selected_config = (config_path or _default_config()).expanduser().resolve()
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(f".{output.name}.staging-{uuid.uuid4().hex}")
    failure = output.parent / f"{output.name}.failure.json"
    staging.mkdir()
    try:
        report = _run_stage_a(
            input_path,
            trajectory_path,
            selected_config,
            staging,
            output_mode=output_mode,
            open3d_backend=open3d_backend,
        )
        _publish(staging, output)
        failure.unlink(missing_ok=True)
        return report
    except BaseException as exc:
        if staging.exists():
            shutil.rmtree(staging)
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
