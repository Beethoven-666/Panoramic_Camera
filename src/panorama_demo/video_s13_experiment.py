"""Development-only M0--M2 runner for the isolated S1.3 candidate."""

from __future__ import annotations

import hashlib
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .video_algorithm import VideoAlgorithmSpec
from .video_s13_base_renderer import render_s13_p0
from .video_s13_bundle import (
    atomic_write_json,
    discard_staging,
    new_generation_staging,
    publish_generation,
    seal_p0,
    sha256_file,
    update_current_base,
    verify_p0_completion,
    write_csv,
    write_image,
    write_npz,
)
from .video_s13_contract import S13_ALGORITHM_ID, load_s13_config
from .video_s13_motion import build_basic_s13_progress, measure_s13_motion
from .video_s13_schedule import build_s13_midpoint_schedule, select_dense_s13_sources
from .video_s13_session import S13Session, load_s13_session, read_s13_rgb
from .video_s13_trajectory import S13Trajectory, load_s13_trajectory


REPORT_SCHEMA = "gemini305-video-s13-output-first-report/v1"
GENERATION_SCHEMA = "gemini305-video-s13-generation/v1"


def _generation_id(session: S13Session, config_sha: str) -> str:
    payload = f"{session.root}|{config_sha}|{time.time_ns()}".encode("utf-8")
    return time.strftime("%Y%m%dT%H%M%S") + "-" + hashlib.sha256(payload).hexdigest()[:12]


def _owner_overlay(image: np.ndarray, owner: np.ndarray) -> np.ndarray:
    overlay = image.copy()
    boundary = np.zeros(owner.shape, dtype=bool)
    boundary[:, 1:] = owner[:, 1:] != owner[:, :-1]
    overlay[boundary] = (0, 0, 255)
    return overlay


def _plain_edge(edge: object) -> dict[str, object]:
    value = asdict(edge)  # type: ignore[arg-type]
    return {key: item for key, item in value.items()}


def _optimizer_all_fail_probe(enabled: bool) -> dict[str, object]:
    """Exercise the post-seal failure boundary without implementing M3+."""

    failures: list[dict[str, str]] = []
    if enabled:
        for name in ("future_geometry", "future_seam", "future_photometric"):
            try:
                raise RuntimeError("injected optimizer failure after immutable P0")
            except RuntimeError as exc:
                failures.append({"optimizer": name, "error": str(exc)})
    return {
        "requested": enabled,
        "attempt_count": len(failures),
        "failures": failures,
        "all_failed": enabled and len(failures) == 3,
        "p0_parent_preserved": True,
    }


def _write_nonpanorama(
    staging: Path,
    session: S13Session,
    *,
    generation_id: str,
    trajectory_audit: Mapping[str, Any],
    run_started: float,
) -> dict[str, Any]:
    target = staging / "nonpanorama"
    target.mkdir(parents=True)
    images = [read_s13_rgb(frame) for frame in session.frames]
    if len(images) == 1:
        from .video_s12_schedule import build_s012_schedule

        frame = session.frames[0]
        schedule = build_s012_schedule(
            (frame.frame_id,),
            (float(session.calibration.cx),),
            session.calibration,
            hard_internal_width_px=None,
        )
        rendered = render_s13_p0(
            schedule,
            session.calibration,
            lambda _frame_id: images[0],
            placement_methods=("origin",),
        )
        image = rendered.image
        asset_type, render_state = "calibrated_single_view" if session.calibration_mode == "calibrated_target" else "raw_single_view", "single_view_only"
        owner = rendered.pixel_provenance["owner_frame_id"]
        source_u = rendered.pixel_provenance["source_u"]
        source_v = rendered.pixel_provenance["source_v"]
        valid = rendered.valid_mask
    else:
        columns = [image[:, image.shape[1] // 2 : image.shape[1] // 2 + 1] for image in images]
        image = np.concatenate(columns, axis=1)
        asset_type, render_state = "temporal_slit_preview", "temporal_preview_only"
        owner = np.broadcast_to(np.asarray([frame.frame_id for frame in session.frames], dtype=np.int32)[None, :], image.shape[:2]).copy()
        source_u = np.broadcast_to(np.asarray([frame.width // 2 for frame in session.frames], dtype=np.float32)[None, :], image.shape[:2]).copy()
        source_v = np.broadcast_to(np.arange(image.shape[0], dtype=np.float32)[:, None], image.shape[:2]).copy()
        valid = np.ones(owner.shape, dtype=bool)
    write_image(target / "preview.png", image)
    write_image(target / "preview.jpg", image)
    write_npz(target / "pixel_provenance.npz", {
        "owner_frame_id": owner, "source_u": source_u, "source_v": source_v,
        "valid": valid, "panorama_claim": np.asarray("none"),
    })
    completion = {
        "schema": "gemini305-video-s13-nonpanorama-completion/v1",
        "generation_id": generation_id,
        "asset_type": asset_type,
        "render_state": render_state,
        "panorama_claim": "none",
        "diagnostic_only": True,
        "production_eligible": False,
        "manual_review_required": True,
        "assets_sha256": {name: sha256_file(target / name) for name in ("preview.png", "preview.jpg", "pixel_provenance.npz")},
        "time_to_visible_asset_seconds": time.perf_counter() - run_started,
        "trajectory": dict(trajectory_audit),
    }
    atomic_write_json(target / "completion.json", completion)
    return completion


def run_s13_experiment(
    *,
    input_path: Path,
    output: Path,
    candidate_config: Path,
    algorithm_spec: VideoAlgorithmSpec,
    trajectory_cache: Path | None,
    reuse_online_trajectory: bool,
    run_offline_orb: bool,
    ignore_pose: bool,
    config_path: Path | None,
    simulate_optimizer_all_fail: bool = False,
) -> dict[str, Any]:
    run_started = time.perf_counter()
    config = load_s13_config(candidate_config)
    if algorithm_spec.algorithm_id != S13_ALGORITHM_ID or algorithm_spec.config_sha256 != candidate_config_sha(config.document):
        raise ValueError("S1.3 dispatch identity/config binding changed after validation")
    root = output.expanduser().resolve()
    stage_seconds: dict[str, float] = {}
    tick = time.perf_counter()
    try:
        session = load_s13_session(input_path, validation_workers=4)
    except Exception as exc:
        atomic_write_json(root / "S013_failure.json", {
            "schema": "gemini305-video-s13-failure/v1",
            "algorithm_id": S13_ALGORITHM_ID,
            "render_state": "input_fatal",
            "panorama_claim": "none",
            "trust_state": "invalid_input",
            "diagnostic_only": True,
            "production_eligible": False,
            "manual_review_required": True,
            "error_type": type(exc).__name__,
            "error": str(exc),
        })
        raise
    stage_seconds["input_and_preflight"] = time.perf_counter() - tick
    tick = time.perf_counter()
    selected_trajectory_sources = sum(
        (trajectory_cache is not None, reuse_online_trajectory, run_offline_orb, ignore_pose)
    )
    if selected_trajectory_sources != 1:
        raise ValueError("S1.3 requires exactly one explicit trajectory source")
    try:
        trajectory = load_s13_trajectory(
            session,
            trajectory_cache=trajectory_cache,
            reuse_online_trajectory=reuse_online_trajectory,
            run_offline_orb=run_offline_orb,
            ignore_pose=ignore_pose,
            config_path=config_path,
        )
    except Exception as exc:
        requested_source = (
            "trajectory_cache" if trajectory_cache is not None
            else "online_trajectory" if reuse_online_trajectory
            else "offline_orbslam3"
        )
        trajectory = S13Trajectory(
            source=f"{requested_source}_invalid_rgb_fallback",
            source_path=trajectory_cache,
            poses_by_frame_id={},
            pose_origin_by_frame_id={},
            islands=(),
            audit={
                "source": requested_source,
                "trajectory_valid": False,
                "trajectory_error_type": type(exc).__name__,
                "trajectory_error": str(exc),
                "direct_pose_count": 0,
                "session_rgb_count": len(session.frames),
                "pose_supported": False,
                "complete_pose_coverage": False,
                "pose_islands": [],
                "interpolated_pose_count": 0,
                "extrapolated_pose_count": 0,
                "fallback": "rgb_motion_without_pose",
            },
        )
    stage_seconds["trajectory"] = time.perf_counter() - tick
    generation_id = _generation_id(session, algorithm_spec.config_sha256)
    staging = new_generation_staging(root, generation_id)
    generation = root / "generations" / generation_id
    try:
        atomic_write_json(staging / "generation_manifest.json", {
            "schema": GENERATION_SCHEMA,
            "generation_id": generation_id,
            "algorithm": algorithm_spec.as_dict(),
            "implemented_milestones": ["M0", "M1", "M2"],
            "excluded_milestones": ["M3", "M4", "M5", "M6", "M7", "M8", "M9"],
            "reads_historical_s1_artifacts": False,
            "writes_production_delivery": False,
        })
        atomic_write_json(staging / "input_audit.json", {
            "root": str(session.root), "strict_session": session.strict_video is not None,
            "strict_error": session.strict_error, "trust_state": session.trust_state,
            "calibration_mode": session.calibration_mode, "real_rgb_frame_ids": [frame.frame_id for frame in session.frames],
            "skipped_rgb": list(session.skipped_rgb), "depth_used_for_p0": False,
        })
        atomic_write_json(staging / "trajectory_audit.json", dict(trajectory.audit))
        if len(session.frames) < 2:
            completion = _write_nonpanorama(staging, session, generation_id=generation_id, trajectory_audit=trajectory.audit, run_started=run_started)
            publish_generation(staging, generation)
            pointer = {"schema": "gemini305-video-s13-current-nonpanorama/v1", "generation_id": generation_id, "generation": str(generation.relative_to(root)).replace("\\", "/"), "completion_sha256": sha256_file(generation / "nonpanorama" / "completion.json")}
            atomic_write_json(root / "current_nonpanorama.json", pointer)
            return {"schema": REPORT_SCHEMA, **completion, "generation": str(generation), "panorama": None}

        tick = time.perf_counter()
        motion = measure_s13_motion(session.frames, analysis_width_px=config.analysis_width_px)
        stage_seconds["motion_measurement"] = time.perf_counter() - tick
        tick = time.perf_counter()
        progress = build_basic_s13_progress(session.frames, motion)
        stage_seconds["progress_and_layout"] = time.perf_counter() - tick
        placement_methods = progress.placement_methods[1:]
        progress_deltas = np.diff(np.asarray(progress.centers_x, dtype=np.float64))
        pause_mask = np.asarray([method == "zero_duplicate" for method in placement_methods], dtype=bool)
        pause_expansion_count = int(np.count_nonzero(progress_deltas[pause_mask] > 1e-9))
        pause_expansion_px = float(np.maximum(progress_deltas[pause_mask], 0.0).sum())
        if pause_expansion_count or pause_expansion_px > 1e-9:
            raise ValueError("S1.3 observed pause expanded the panorama layout")
        zero_duplicate_edge_count = int(np.count_nonzero(pause_mask))
        subpixel_motion_edge_count = sum(method.endswith("_subpixel") for method in placement_methods)
        fallback_progress_edge_count = sum(
            method in {"session_median", "temporal_order"} for method in placement_methods
        )
        if not progress.spatial:
            completion = _write_nonpanorama(staging, session, generation_id=generation_id, trajectory_audit=trajectory.audit, run_started=run_started)
            atomic_write_json(staging / "motion_telemetry.json", {"edges": [_plain_edge(edge) for edge in motion]})
            publish_generation(staging, generation)
            pointer = {"schema": "gemini305-video-s13-current-nonpanorama/v1", "generation_id": generation_id, "generation": str(generation.relative_to(root)).replace("\\", "/"), "completion_sha256": sha256_file(generation / "nonpanorama" / "completion.json")}
            atomic_write_json(root / "current_nonpanorama.json", pointer)
            return {"schema": REPORT_SCHEMA, **completion, "generation": str(generation), "panorama": None}

        selection = select_dense_s13_sources(
            progress, motion, session.calibration,
            normal_target_advance_px=config.normal_target_advance_px,
            risky_target_advance_px=config.risky_target_advance_px,
        )
        schedule = build_s13_midpoint_schedule(selection, session.calibration)
        p0 = staging / "P0"
        p0.mkdir()
        by_id = session.frame_by_id
        tick = time.perf_counter()
        result = render_s13_p0(
            schedule, session.calibration, lambda frame_id: read_s13_rgb(by_id[frame_id]),
            placement_methods=selection.placement_methods,
        )
        write_image(p0 / "base_panorama_owner_only.png", result.image)
        write_image(p0 / "base_panorama_owner_only.jpg", result.image)
        write_image(p0 / "base_valid_mask.png", result.valid_mask.astype(np.uint8) * 255)
        write_image(p0 / "owner_boundary_overlay.png", _owner_overlay(result.image, result.pixel_provenance["owner_frame_id"]))
        write_npz(p0 / "base_pixel_provenance.npz", result.pixel_provenance)
        write_npz(p0 / "base_column_provenance.npz", result.column_provenance)
        source_rows = [
            {
                "assignment_index": assignment.assignment_index, "source_index": assignment.source_index,
                "frame_id": assignment.frame_id, "center_x": assignment.center_x, "left_x": assignment.left_x,
                "right_x": assignment.right_x, "width": assignment.width, "zero_width": assignment.zero_width,
                "placement_method": selection.placement_methods[assignment.source_index],
                "risky": selection.risk_by_frame_id.get(assignment.frame_id, False),
            }
            for assignment in schedule.assignments
        ]
        write_csv(p0 / "base_sources.csv", source_rows, tuple(source_rows[0]))
        layout = {
            "schema": "gemini305-video-s13-p0-layout/v1", "layout_level": "M2_basic_rgb_progress",
            "panorama_claim": "pose_supported" if trajectory.audit.get("complete_pose_coverage") is True else "visual_nonmetric",
            "metric_scale_claim": False, "pose_used_as_pixel_scale": False,
            "frame_ids": list(progress.frame_ids), "centers_x": list(progress.centers_x),
            "placement_methods": list(progress.placement_methods), "selected_frame_ids": list(selection.frame_ids),
            "selected_centers_x": list(selection.centers_x), "boundaries": list(schedule.boundaries),
            "canvas_width": schedule.canvas_width, "canvas_height": schedule.canvas_height,
            "midpoint_hard_owner": True, "transition_width_px": 0,
            "motion_graph_connected_telemetry": progress.adjacent_reliable_graph_connected,
            "motion_graph_connected_used_as_structural_gate": False,
            "direct_local_delta_used_as_structural_gate": False,
            "zero_duplicate_edge_count": zero_duplicate_edge_count,
            "subpixel_motion_edge_count": subpixel_motion_edge_count,
            "fallback_progress_edge_count": fallback_progress_edge_count,
            "observed_pause_expansion_count": pause_expansion_count,
            "observed_pause_expansion_px": pause_expansion_px,
        }
        atomic_write_json(p0 / "base_layout.json", layout)
        atomic_write_json(staging / "motion_telemetry.json", {"edges": [_plain_edge(edge) for edge in motion]})
        stage_seconds["p0_render_and_export"] = time.perf_counter() - tick
        seal_p0(p0, generation_id=generation_id, metadata={
            "render_state": "base_generated", "panorama_claim": layout["panorama_claim"],
            "trust_state": session.trust_state, "manual_review_required": True,
            "owner_policy": "fixed_midpoint", "remap_invocations": result.remap_invocations,
            "contributor_source_count": sum(not assignment.zero_width for assignment in schedule.assignments),
            "duplicate_write_count": result.duplicate_write_count,
        })
        atomic_write_json(staging / "report.json", {
            "schema": REPORT_SCHEMA, "generation_id": generation_id, "render_state": "base_generated",
            "panorama_claim": layout["panorama_claim"], "diagnostic_only": True, "production_eligible": False,
            "production_lock_eligible": False, "manual_review_required": True,
            "optimizer_state": "all_failed_after_p0" if simulate_optimizer_all_fail else "not_started_m2",
            "p0_parent": "P0/P0_completion.json", "pair_count": len(session.frames) - 1,
            "motion_graph_connected_telemetry": progress.adjacent_reliable_graph_connected,
            "motion_graph_disconnection_is_fatal": False, "direct_local_delta_is_fatal": False,
            "zero_duplicate_edge_count": layout["zero_duplicate_edge_count"],
            "subpixel_motion_edge_count": layout["subpixel_motion_edge_count"],
            "fallback_progress_edge_count": layout["fallback_progress_edge_count"],
            "observed_pause_expansion_count": layout["observed_pause_expansion_count"],
            "observed_pause_expansion_px": layout["observed_pause_expansion_px"],
            "trajectory": dict(trajectory.audit), "performance": {"stage_seconds": stage_seconds},
        })
        publish_generation(staging, generation)
        pointer = update_current_base(root, generation)
        time_to_p0 = time.perf_counter() - run_started
        p0_hashes_before = dict(pointer["p0_assets_sha256"])
        # M2 has no optimizer. This injected path proves that any later optimizer
        # failure occurs strictly after the immutable completion and pointer.
        optimizer = _optimizer_all_fail_probe(simulate_optimizer_all_fail)
        p0_hashes_after = verify_p0_completion(generation / "P0")["assets_sha256"]
        optimizer["p0_hash_unchanged"] = p0_hashes_before == p0_hashes_after
        report = {
            "schema": REPORT_SCHEMA, "algorithm_id": S13_ALGORITHM_ID, "generation_id": generation_id,
            "generation": str(generation), "panorama": str(generation / "P0" / "base_panorama_owner_only.png"),
            "layout": str(generation / "P0" / "base_layout.json"),
            "owner_overlay": str(generation / "P0" / "owner_boundary_overlay.png"),
            "pixel_provenance": str(generation / "P0" / "base_pixel_provenance.npz"),
            "column_provenance": str(generation / "P0" / "base_column_provenance.npz"),
            "completion": str(generation / "P0" / "P0_completion.json"),
            "current_base": str(root / "current_base.json"), "time_to_P0_seconds": time_to_p0,
            "total_wall_seconds": time.perf_counter() - run_started, "stage_seconds": stage_seconds,
            "render_state": "base_generated", "panorama_claim": layout["panorama_claim"],
            "diagnostic_only": True, "production_eligible": False, "production_lock_eligible": False,
            "manual_review_required": True, "optimizer": optimizer,
            "motion_graph_connected_telemetry": progress.adjacent_reliable_graph_connected,
            "motion_graph_disconnection_is_fatal": False, "direct_local_delta_is_fatal": False,
        }
        atomic_write_json(root / "latest_run.json", report)
        return report
    except BaseException:
        if staging.exists():
            discard_staging(staging)
        raise


def candidate_config_sha(document: Mapping[str, Any]) -> str:
    from .video_algorithm import canonical_config_sha256

    return canonical_config_sha256(document)


__all__ = ["run_s13_experiment"]
