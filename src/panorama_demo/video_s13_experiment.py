"""Development-only M0--M3 runner for the isolated S1.3 candidate."""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .video_algorithm import VideoAlgorithmSpec
from .video_s13_base_renderer import render_s13_p0
from .video_s13_bundle import (
    atomic_write_json,
    discard_staging,
    invalidate_current_preview,
    new_generation_staging,
    publish_generation,
    seal_p0,
    seal_stage,
    sha256_file,
    update_current_base,
    update_current_preview,
    verify_p0_completion,
    verify_stage,
    write_csv,
    write_image,
    write_npz,
)
from .video_s13_contract import S13_ALGORITHM_ID, load_s13_config
from .video_s13_motion import measure_s13_motion
from .video_s13_m5 import run_s13_m5
from .video_s13_progress import S13M3Layout, build_s13_m3_layout
from .video_s13_schedule import S13SchedulePlan, plan_s13_m3_schedule
from .video_s13_session import S13Session, load_s13_session, read_s13_rgb
from .video_s13_trajectory import S13Trajectory, load_s13_trajectory
from .video_s13_vertical import estimate_s13_vertical, render_s13_p1_from_raw
from .video_s13_selection import select_s13_vertical_parent


REPORT_SCHEMA = "gemini305-video-s13-output-first-report/v1"
GENERATION_SCHEMA = "gemini305-video-s13-generation/v1"
P2_COMPLETION_SCHEMA = "gemini305-video-s13-p2-completion/v1"


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
    value.pop("observations", None)
    return {key: item for key, item in value.items()}


def _optimizer_all_fail_probe(enabled: bool) -> dict[str, object]:
    """Exercise the post-seal failure boundary without implementing M3+."""

    failures: list[dict[str, str]] = []
    if enabled:
        for name in ("vertical_selection", "geometry", "seam_transaction"):
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


def _m31_report_statistics(
    motion: tuple[object, ...], layout: S13M3Layout, schedule_plan: S13SchedulePlan
) -> dict[str, object]:
    methods = Counter()
    for method in layout.progress.placement_methods[1:]:
        if method.startswith("motion_hypothesis") or method.startswith("F0_"):
            methods["F0"] += 1
        else:
            matched = next((f"F{index}" for index in range(1, 8) if method.startswith(f"F{index}_")), None)
            methods[matched or "unclassified"] += 1
    hypothesis_histogram = Counter(len(getattr(edge, "motion_hypotheses")) for edge in motion)
    render_source_count = sum(
        not assignment.zero_width for schedule in schedule_plan.schedules for assignment in schedule.assignments
    )
    render_pair_count = sum(
        max(0, sum(not assignment.zero_width for assignment in schedule.assignments) - 1)
        for schedule in schedule_plan.schedules
    )
    finite_source_u = [
        float(value)
        for schedule in schedule_plan.schedules
        for value in schedule.column_source_u[np.isfinite(schedule.column_source_u)]
    ]
    switch_count = sum(
        left.support_signature_sha256 != right.support_signature_sha256
        for left, right in zip(layout.lineage[:-1], layout.lineage[1:])
        if left.support_signature_sha256 is not None and right.support_signature_sha256 is not None
    )
    return {
        "raw_rgb_motion_edge_count": len(motion),
        "raw_rgb_adjacent_edge_count": sum(getattr(edge, "step") == 1 for edge in motion),
        "render_source_count": render_source_count,
        "render_pair_count": render_pair_count,
        "lineage_switch_count": switch_count,
        "fallback_counts": {f"F{index}": int(methods.get(f"F{index}", 0)) for index in range(8)},
        "fallback_unclassified_count": int(methods.get("unclassified", 0)),
        "hypothesis_count_histogram": {
            str(count): int(hypothesis_histogram.get(count, 0)) for count in range(4)
        },
        "source_u_statistics": {
            "minimum": min(finite_source_u) if finite_source_u else None,
            "maximum": max(finite_source_u) if finite_source_u else None,
            "span": max(finite_source_u) - min(finite_source_u) if finite_source_u else None,
            "maximum_internal_owner_width_px": max(
                (assignment.width for schedule in schedule_plan.schedules for assignment in schedule.assignments[1:-1]),
                default=0,
            ),
            "configured_cap_px": schedule_plan.source_u_cap_px,
        },
    }


def _run_m4_vertical(
    generation: Path,
    root: Path,
    generation_id: str,
    schedule: object,
    calibration: object,
    image_loader: object,
) -> dict[str, object]:
    from .video_s12_schedule import S012Schedule

    if not isinstance(schedule, S012Schedule):
        raise TypeError("S1.3 M4 requires one immutable P0 schedule")
    pending = generation / f".P1.{uuid.uuid4().hex}.pending"
    final = generation / "P1"
    if final.exists():
        raise FileExistsError(f"S1.3 M4 P1 already exists: {final}")
    pending.mkdir()
    try:
        solution = estimate_s13_vertical(schedule, calibration, image_loader)  # type: ignore[arg-type]
        result = render_s13_p1_from_raw(schedule, calibration, image_loader, solution)  # type: ignore[arg-type]
        write_image(pending / "vertical_panorama_owner_only.png", result.image)
        write_image(pending / "vertical_panorama_owner_only.jpg", result.image)
        write_image(pending / "vertical_valid_mask.png", result.valid_mask.astype(np.uint8) * 255)
        write_image(pending / "vertical_owner_boundary_overlay.png", _owner_overlay(
            result.image, result.pixel_provenance["owner_frame_id"]
        ))
        write_npz(pending / "vertical_pixel_provenance.npz", result.pixel_provenance)
        write_npz(pending / "vertical_solution.npz", {
            "global_offsets_px": np.asarray(solution.global_offsets_px, dtype=np.float32),
            **{
                f"pair_{index:04d}_local_row_residual_px": rows
                for index, rows in enumerate(solution.local_row_residuals)
            },
        })
        atomic_write_json(pending / "pair_report.json", {
            "schema": "gemini305-video-s13-m4-pair-report/v1",
            "pairs": [asdict(pair) for pair in solution.pairs],
            "audit": dict(solution.audit),
        })
        p0_completion = generation / "P0" / "P0_completion.json"
        p0_completion_sha = sha256_file(p0_completion)
        assets = sorted(path for path in pending.rglob("*") if path.is_file())
        completion = {
            "schema": "gemini305-video-s13-p1-vertical-completion/v1",
            "generation_id": generation_id,
            "stage": "P1",
            "sealed": True,
            "p0_parent": "../P0/P0_completion.json",
            "p0_parent_sha256": p0_completion_sha,
            "models": ["global_scalar_dy", "pair_local_row_residual"],
            "excluded_models": [
                "translation", "rotation", "affine", "seam", "graphcut", "photometric", "blend", "depth", "mesh",
            ],
            "formal_raw_rgb_remap_invocations": result.remap_invocations,
            "formal_raw_rgb_unique_sources": len(result.decoded_frame_ids),
            "selected_global_gain": solution.selected_gain,
            "assets_sha256": {path.relative_to(pending).as_posix(): sha256_file(path) for path in assets},
        }
        atomic_write_json(pending / "P1_completion.json", completion)
        os.replace(pending, final)
        for name, expected in completion["assets_sha256"].items():
            asset = final / str(name)
            if not asset.is_file() or sha256_file(asset) != expected:
                raise ValueError(f"S1.3 M4 sealed P1 asset hash mismatch: {name}")
        if sha256_file(generation / "P0" / "P0_completion.json") != p0_completion_sha:
            raise ValueError("S1.3 M4 immutable P0 parent changed during P1 commit")
        return {
            "state": "P1_vertical_candidate_generated",
            "panorama": str(final / "vertical_panorama_owner_only.png"),
            "owner_overlay": str(final / "vertical_owner_boundary_overlay.png"),
            "pixel_provenance": str(final / "vertical_pixel_provenance.npz"),
            "completion": str(final / "P1_completion.json"),
            "current_preview": None,
            "selected_global_gain": solution.selected_gain,
            "pair_count": len(solution.pairs),
            "applied_pair_count": sum(pair.status == "applied" for pair in solution.pairs),
            "formal_raw_rgb_remap_invocations": result.remap_invocations,
        }
    except BaseException:
        discard_staging(pending)
        raise


def _run_m5(
    generation: Path,
    root: Path,
    generation_id: str,
    schedule: object,
    calibration: object,
    image_loader: object,
    p0_image: np.ndarray,
    *,
    selected_hypothesis_ids: tuple[int, ...],
    placement_methods: tuple[str, ...],
) -> dict[str, object]:
    from .session import CameraIntrinsics
    from .video_s12_schedule import S012Schedule

    if not isinstance(schedule, S012Schedule) or not isinstance(calibration, CameraIntrinsics):
        raise TypeError("S1.3 M5 requires one immutable P0 schedule and calibration")
    pending = generation / f".P2.{uuid.uuid4().hex}.pending"
    final = generation / "P2"
    if final.exists():
        raise FileExistsError(f"S1.3 M5 P2 already exists: {final}")
    pending.mkdir()
    p0_completion_path = generation / "P0" / "P0_completion.json"
    p1_completion_path = generation / "P1" / "P1_completion.json"
    p0_completion_sha = sha256_file(p0_completion_path)
    p1_completion_sha = sha256_file(p1_completion_path)
    try:
        started = time.perf_counter()
        base_solution = estimate_s13_vertical(schedule, calibration, image_loader)  # type: ignore[arg-type]
        selection = select_s13_vertical_parent(
            schedule, calibration, image_loader, base_solution, p0_image  # type: ignore[arg-type]
        )
        write_image(pending / "selected_vertical_parent.png", selection.result.image)
        write_image(pending / "selected_vertical_parent.jpg", selection.result.image)
        write_npz(pending / "selected_vertical_parent_provenance.npz", selection.result.pixel_provenance)
        atomic_write_json(pending / "vertical_selection.json", dict(selection.audit))
        parent_stage_sha = sha256_file(pending / "vertical_selection.json")
        m5 = run_s13_m5(
            schedule, calibration, image_loader, selection.solution, selection.result.image,  # type: ignore[arg-type]
            parent_stage_sha256=parent_stage_sha,
            selected_hypothesis_ids=selected_hypothesis_ids,
            placement_methods=placement_methods,
        )
        write_image(pending / "geometry_panorama_owner_only.png", m5.geometry_result.image)
        write_image(pending / "geometry_and_seam_panorama_owner_only.png", m5.final_result.image)
        write_image(pending / "geometry_and_seam_panorama_owner_only.jpg", m5.final_result.image)
        write_image(pending / "seam_overlay.png", m5.seam_overlay)
        write_image(pending / "p2_valid_mask.png", m5.final_result.valid_mask.astype(np.uint8) * 255)
        write_npz(pending / "p2_pixel_provenance.npz", m5.final_result.pixel_provenance)
        transaction_rows = [dict(pair.transaction) for pair in m5.pairs]
        atomic_write_json(pending / "pair_transactions.json", {
            "schema": "gemini305-video-s13-m5-pair-transactions/v1",
            "all_pairs_reported": len(transaction_rows) == len(schedule.assignments) - 1,
            "pairs": transaction_rows,
        })
        atomic_write_json(pending / "p2_selection.json", dict(m5.selection_audit))
        for pair in m5.pairs:
            atomic_write_json(
                pending / "pair_transactions" / f"pair_{int(pair.transaction['transaction_id'].split('-')[-1]):04d}.json",
                dict(pair.transaction),
            )
        ranked = sorted(
            enumerate(m5.pairs),
            key=lambda item: float(item[1].transaction.get("after_metrics", {}).get("score") or -1.0),
            reverse=True,
        )[:10]
        crop_rows: list[dict[str, object]] = []
        for rank, (pair_index, pair) in enumerate(ranked):
            seam = np.asarray(pair.seam_x_by_row, dtype=np.int32)
            left = max(0, int(seam.min()) - 48)
            right = min(schedule.canvas_width, int(seam.max()) + 49)
            crop = m5.seam_overlay[:, left:right]
            name = f"rank_{rank:02d}_pair_{pair_index:04d}.png"
            write_image(pending / "worst_seam_crops" / name, crop)
            crop_rows.append({
                "rank": rank, "pair_index": pair_index, "left_x": left, "right_x": right,
                "after_score": pair.transaction.get("after_metrics", {}).get("score"), "asset": name,
            })
        atomic_write_json(pending / "worst_seam_crops" / "manifest.json", {
            "schema": "gemini305-video-s13-m5-worst-seam-crops/v1", "crops": crop_rows,
        })
        performance = {
            "input_and_preflight": 0.0,
            "trajectory": 0.0,
            "motion_measurement": 0.0,
            "motion_hypotheses": 0.0,
            "progress_and_layout": 0.0,
            "time_to_P0": None,
            "vertical": time.perf_counter() - started - float(m5.performance["total_m5"]),
            "geometry": float(m5.performance["geometry"]),
            "seam": float(m5.performance["seam_and_p2_render"]),
            "photometric": None,
            "blend": None,
            "repair": None,
            "artifact_export": None,
            "total_wall_time": time.perf_counter() - started,
            "peak_memory": None,
            "excluded_after_m5": ["photometric", "luminance_field", "feather", "multiband", "depth", "mesh", "source_rescue"],
        }
        atomic_write_json(pending / "performance.json", performance)
        completion = seal_stage(
            pending,
            completion_name="P2_completion.json",
            schema=P2_COMPLETION_SCHEMA,
            metadata={
                "generation_id": generation_id,
                "stage": "P2",
                "selected_as_best": bool(m5.selected_as_best),
                "selected_vertical_parent": selection.stage,
                "selected_global_gain": selection.selected_gain,
                "p0_parent": "../P0/P0_completion.json",
                "p0_parent_sha256": p0_completion_sha,
                "p1_candidate_parent": "../P1/P1_completion.json",
                "p1_candidate_parent_sha256": p1_completion_sha,
                "formal_raw_rgb_remap_invocations": m5.final_result.remap_invocations,
                "formal_raw_rgb_unique_sources": len(m5.final_result.decoded_frame_ids),
                "all_pair_transaction_count": len(m5.pairs),
                "applied_pair_transaction_count": sum(pair.transaction["decision"] == "applied" for pair in m5.pairs),
                "before_mean_structure_score": m5.before_mean_score,
                "after_mean_structure_score": m5.after_mean_score,
                "selection_audit": dict(m5.selection_audit),
                "comparison_coordinate_policy": m5.selection_audit[
                    "comparison_coordinate_policy"
                ],
                "models": ["C0", "C1", "C2", "C3", "C4", "S0", "S1", "S2"],
                "excluded_models": [
                    "graphcut", "photometric_gain_bias", "luminance_field", "feather", "multiband",
                    "depth", "mesh", "source_rescue", "production_renderer",
                ],
            },
        )
        os.replace(pending, final)
        verify_stage(final, completion_name="P2_completion.json", schema=P2_COMPLETION_SCHEMA)
        if sha256_file(p0_completion_path) != p0_completion_sha:
            raise ValueError("S1.3 M5 immutable P0 completion changed")
        if sha256_file(p1_completion_path) != p1_completion_sha:
            raise ValueError("S1.3 M5 immutable original P1 completion changed")
        pointer = None
        if completion["selected_as_best"] is True:
            pointer = update_current_preview(
                root, generation, stage="P2", completion_name="P2_completion.json",
                completion_schema=P2_COMPLETION_SCHEMA, p0_parent_sha256=p0_completion_sha,
            )
        return {
            "state": "P2_selected" if completion["selected_as_best"] else "P2_generated_parent_retained",
            "panorama": str(final / "geometry_and_seam_panorama_owner_only.png"),
            "geometry": str(final / "geometry_panorama_owner_only.png"),
            "vertical_parent": str(final / "selected_vertical_parent.png"),
            "seam_overlay": str(final / "seam_overlay.png"),
            "pixel_provenance": str(final / "p2_pixel_provenance.npz"),
            "pair_transactions": str(final / "pair_transactions.json"),
            "worst_seam_crops": str(final / "worst_seam_crops"),
            "performance": str(final / "performance.json"),
            "completion": str(final / "P2_completion.json"),
            "current_preview": str(root / "current_preview.json") if pointer is not None else None,
            "selected_as_best": bool(completion["selected_as_best"]),
            "selected_vertical_parent": selection.stage,
            "selected_global_gain": selection.selected_gain,
            "pair_count": len(m5.pairs),
            "applied_pair_count": sum(pair.transaction["decision"] == "applied" for pair in m5.pairs),
            "formal_raw_rgb_remap_invocations": m5.final_result.remap_invocations,
        }
    except BaseException:
        discard_staging(pending)
        raise


def _write_nonpanorama(
    staging: Path,
    session: S13Session,
    *,
    generation_id: str,
    trajectory_audit: Mapping[str, Any],
    run_started: float,
    layout_audit: Mapping[str, Any] | None = None,
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
        "layout": dict(layout_audit or {}),
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
    run_m4: bool = True,
    run_m5: bool = True,
) -> dict[str, Any]:
    run_started = time.perf_counter()
    config = load_s13_config(candidate_config)
    if algorithm_spec.algorithm_id != S13_ALGORITHM_ID or algorithm_spec.config_sha256 != candidate_config_sha(config.document):
        raise ValueError("S1.3 dispatch identity/config binding changed after validation")
    root = output.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    invalidate_current_preview(root)
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
            "implemented_milestones": [
                "M0", "M1", "M2", "M3", *(("M4",) if run_m4 else ()),
                *(("M5",) if run_m4 and run_m5 else ()),
            ],
            "excluded_milestones": (
                ["M6", "M7", "M8", "M9"] if run_m4 and run_m5
                else ["M5", "M6", "M7", "M8", "M9"] if run_m4
                else ["M4", "M5", "M6", "M7", "M8", "M9"]
            ),
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
        m3_layout = build_s13_m3_layout(session.frames, motion, trajectory)
        progress = m3_layout.progress
        stage_seconds["progress_and_layout"] = time.perf_counter() - tick
        placement_methods = progress.placement_methods[1:]
        progress_deltas = np.diff(np.asarray(progress.centers_x, dtype=np.float64))
        pause_mask = np.asarray([method.startswith("F5_zero_duplicate") for method in placement_methods], dtype=bool)
        pause_expansion_count = int(np.count_nonzero(progress_deltas[pause_mask] > 1e-9))
        pause_expansion_px = float(np.maximum(progress_deltas[pause_mask], 0.0).sum())
        if pause_expansion_count or pause_expansion_px > 1e-9:
            raise ValueError("S1.3 observed pause expanded the panorama layout")
        zero_duplicate_edge_count = int(np.count_nonzero(pause_mask))
        subpixel_motion_edge_count = sum(method.endswith("_subpixel") for method in placement_methods)
        fallback_progress_edge_count = sum(
            method.startswith(("F3_", "F4_", "F7_")) for method in placement_methods
        )
        if not progress.spatial:
            completion = _write_nonpanorama(
                staging, session, generation_id=generation_id, trajectory_audit=trajectory.audit,
                run_started=run_started,
                layout_audit={
                    "layout_level": m3_layout.layout_level,
                    "spatial": False,
                    "canonical_scan_direction": m3_layout.canonical_scan_direction,
                    **dict(m3_layout.audit),
                },
            )
            atomic_write_json(staging / "motion_telemetry.json", {"edges": [_plain_edge(edge) for edge in motion]})
            publish_generation(staging, generation)
            pointer = {"schema": "gemini305-video-s13-current-nonpanorama/v1", "generation_id": generation_id, "generation": str(generation.relative_to(root)).replace("\\", "/"), "completion_sha256": sha256_file(generation / "nonpanorama" / "completion.json")}
            atomic_write_json(root / "current_nonpanorama.json", pointer)
            return {"schema": REPORT_SCHEMA, **completion, "generation": str(generation), "panorama": None}

        schedule_plan = plan_s13_m3_schedule(
            progress, motion, session.calibration,
            normal_target_advance_px=config.normal_target_advance_px,
            risky_target_advance_px=config.risky_target_advance_px,
            segment_break_pairs=m3_layout.segment_break_pairs,
        )
        selection = schedule_plan.selections[0]
        schedule = schedule_plan.schedules[0]
        p0 = staging / "P0"
        p0.mkdir()
        by_id = session.frame_by_id
        tick = time.perf_counter()
        hypothesis_by_frame = {step.target_frame_id: step.selected_hypothesis_id for step in m3_layout.lineage}
        panel_results = []
        panel_images = []
        for panel_index, (panel_selection, panel_schedule) in enumerate(
            zip(schedule_plan.selections, schedule_plan.schedules, strict=True)
        ):
            panel_result = render_s13_p0(
                panel_schedule, session.calibration, lambda frame_id: read_s13_rgb(by_id[frame_id]),
                placement_methods=panel_selection.placement_methods,
                selected_hypothesis_ids=tuple(
                    hypothesis_by_frame.get(frame_id, -1) for frame_id in panel_selection.frame_ids
                ),
            )
            panel_results.append(panel_result)
            panel_images.append(panel_result.image)
            if len(schedule_plan.schedules) > 1:
                panel_root = p0 / "panels" / f"panel_{panel_index:02d}"
                write_image(panel_root / "panorama_owner_only.png", panel_result.image)
                write_image(panel_root / "owner_boundary_overlay.png", _owner_overlay(
                    panel_result.image, panel_result.pixel_provenance["owner_frame_id"]
                ))
                write_npz(panel_root / "pixel_provenance.npz", panel_result.pixel_provenance)
                write_npz(panel_root / "column_provenance.npz", panel_result.column_provenance)
        result = panel_results[0]
        if len(panel_results) == 1:
            write_image(p0 / "base_panorama_owner_only.png", result.image)
            write_image(p0 / "base_panorama_owner_only.jpg", result.image)
            write_image(p0 / "base_valid_mask.png", result.valid_mask.astype(np.uint8) * 255)
            write_image(p0 / "owner_boundary_overlay.png", _owner_overlay(
                result.image, result.pixel_provenance["owner_frame_id"]
            ))
            write_npz(p0 / "base_pixel_provenance.npz", result.pixel_provenance)
            write_npz(p0 / "base_column_provenance.npz", result.column_provenance)
        if len(panel_images) > 1:
            gap_marker = np.zeros((panel_images[0].shape[0], 12, 3), dtype=np.uint8)
            gap_marker[:, :, 2] = 255
            overview_parts: list[np.ndarray] = []
            for panel_index, panel_image in enumerate(panel_images):
                if panel_index:
                    overview_parts.append(gap_marker)
                overview_parts.append(panel_image)
            write_image(p0 / "panel_navigation_overview_explicit_gaps.png", np.concatenate(overview_parts, axis=1))
        source_rows = [
            {
                "panel_index": panel_index,
                "assignment_index": assignment.assignment_index, "source_index": assignment.source_index,
                "frame_id": assignment.frame_id, "center_x": assignment.center_x, "left_x": assignment.left_x,
                "right_x": assignment.right_x, "width": assignment.width, "zero_width": assignment.zero_width,
                "placement_method": panel_selection.placement_methods[assignment.source_index],
                "risky": panel_selection.risk_by_frame_id.get(assignment.frame_id, False),
            }
            for panel_index, (panel_selection, panel_schedule) in enumerate(
                zip(schedule_plan.selections, schedule_plan.schedules, strict=True)
            )
            for assignment in panel_schedule.assignments
        ]
        write_csv(p0 / "base_sources.csv", source_rows, tuple(source_rows[0]))
        is_panel_set = len(schedule_plan.schedules) > 1
        render_state = "spatial_panel_set" if is_panel_set else "base_generated"
        panorama_claim = "none" if is_panel_set else (
            "pose_supported" if m3_layout.pose_supported else "visual_nonmetric"
        )
        report_statistics = _m31_report_statistics(motion, m3_layout, schedule_plan)
        layout = {
            "schema": "gemini305-video-s13-p0-layout/v1", "layout_level": m3_layout.layout_level,
            "render_state": render_state, "panorama_claim": panorama_claim,
            "metric_scale_claim": False, "pose_used_as_pixel_scale": False,
            "frame_ids": list(progress.frame_ids), "centers_x": list(progress.centers_x),
            "placement_methods": list(progress.placement_methods),
            "selected_frame_ids": [
                frame_id for panel_selection in schedule_plan.selections for frame_id in panel_selection.frame_ids
            ],
            "selected_centers_x": None if is_panel_set else list(selection.centers_x),
            "boundaries": None if is_panel_set else list(schedule.boundaries),
            "canvas_width": None if is_panel_set else schedule.canvas_width,
            "canvas_height": session.calibration.height,
            "pixels_per_meter_rgb_calibrated": m3_layout.pixels_per_meter,
            "pose_calibration_pairs": m3_layout.pose_calibration_pairs,
            "pose_calibration_reason": m3_layout.audit.get("pose_calibration_reason"),
            "pose_role": "direction_and_low_frequency_soft_prior_only",
            "canonical_scan_direction": m3_layout.canonical_scan_direction,
            "segment_break_pairs": [list(pair) for pair in m3_layout.segment_break_pairs],
            "motion_hypothesis_count_by_pair": {
                f"{edge.source_frame_id}:{edge.target_frame_id}": len(edge.motion_hypotheses) for edge in motion
            },
            "coherent_lineage": [asdict(step) for step in m3_layout.lineage],
            "source_u_fraction_cap": 0.20,
            "source_u_cap_px": schedule_plan.source_u_cap_px,
            "rescued_real_frame_ids": list(schedule_plan.rescued_frame_ids),
            "canvas_density_scale": schedule_plan.density_scale,
            "panel_count": len(schedule_plan.schedules),
            "panels": [
                {
                    "panel_index": index,
                    "frame_ids": list(panel_selection.frame_ids),
                    "centers_x": list(panel_selection.centers_x),
                    "canvas_width": panel_schedule.canvas_width,
                    "canvas_height": panel_schedule.canvas_height,
                    "artifact_root": f"panels/panel_{index:02d}" if is_panel_set else ".",
                }
                for index, (panel_selection, panel_schedule) in enumerate(
                    zip(schedule_plan.selections, schedule_plan.schedules, strict=True)
                )
            ],
            "panel_gap_frame_ids": [list(gap) for gap in schedule_plan.panel_gap_frame_ids],
            "source_u_cap_satisfied": schedule_plan.cap_satisfied,
            "panel_fallback_used": len(schedule_plan.schedules) > 1,
            "panel_overview_panorama_claim": "none" if len(schedule_plan.schedules) > 1 else "not_applicable",
            "midpoint_hard_owner": True, "transition_width_px": 0,
            "motion_graph_connected_telemetry": progress.adjacent_reliable_graph_connected,
            "motion_graph_connected_used_as_structural_gate": False,
            "direct_local_delta_used_as_structural_gate": False,
            "zero_duplicate_edge_count": zero_duplicate_edge_count,
            "subpixel_motion_edge_count": subpixel_motion_edge_count,
            "fallback_progress_edge_count": fallback_progress_edge_count,
            "observed_pause_expansion_count": pause_expansion_count,
            "observed_pause_expansion_px": pause_expansion_px,
            **report_statistics,
        }
        atomic_write_json(p0 / "base_layout.json", layout)
        atomic_write_json(staging / "motion_telemetry.json", {"edges": [_plain_edge(edge) for edge in motion]})
        atomic_write_json(staging / "motion_hypotheses.json", {
            "schema": "gemini305-video-s13-motion-hypotheses/v1",
            "maximum_hypotheses_per_pair": 3,
            "pairs": [_plain_edge(edge) for edge in motion],
        })
        atomic_write_json(staging / "coherent_lineage.json", {
            "schema": "gemini305-video-s13-coherent-lineage/v1",
            "selection": "whole_chain_dynamic_programming",
            "steps": [asdict(step) for step in m3_layout.lineage],
            "layout_audit": dict(m3_layout.audit),
        })
        stage_seconds["p0_render_and_export"] = time.perf_counter() - tick
        seal_p0(p0, generation_id=generation_id, metadata={
            "render_state": render_state, "panorama_claim": layout["panorama_claim"],
            "asset_semantics": "spatial_panel_set" if is_panel_set else "spatial_panorama",
            "trust_state": session.trust_state, "manual_review_required": True,
            "owner_policy": "fixed_midpoint", "remap_invocations": result.remap_invocations,
            "contributor_source_count": sum(
                not assignment.zero_width for panel_schedule in schedule_plan.schedules
                for assignment in panel_schedule.assignments
            ),
            "remap_invocations_all_panels": sum(item.remap_invocations for item in panel_results),
            "duplicate_write_count": sum(item.duplicate_write_count for item in panel_results),
        })
        atomic_write_json(staging / "report.json", {
            "schema": REPORT_SCHEMA, "generation_id": generation_id, "render_state": render_state,
            "panorama_claim": layout["panorama_claim"], "diagnostic_only": True, "production_eligible": False,
            "production_lock_eligible": False, "manual_review_required": True,
            "optimizer_state": (
                "all_failed_after_p0" if simulate_optimizer_all_fail
                else "m5_pending_after_p0" if run_m4 and run_m5 and not is_panel_set
                else "m4_pending_after_p0" if run_m4 and not is_panel_set
                else "m3_complete_m4_not_started"
            ),
            "p0_parent": "P0/P0_completion.json", "raw_rgb_motion_pair_count": len(session.frames) - 1,
            "motion_graph_connected_telemetry": progress.adjacent_reliable_graph_connected,
            "motion_graph_disconnection_is_fatal": False, "direct_local_delta_is_fatal": False,
            "zero_duplicate_edge_count": layout["zero_duplicate_edge_count"],
            "subpixel_motion_edge_count": layout["subpixel_motion_edge_count"],
            "fallback_progress_edge_count": layout["fallback_progress_edge_count"],
            "observed_pause_expansion_count": layout["observed_pause_expansion_count"],
            "observed_pause_expansion_px": layout["observed_pause_expansion_px"],
            "trajectory": dict(trajectory.audit), "performance": {"stage_seconds": stage_seconds},
            **report_statistics,
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
        m4: dict[str, object] = {
            "state": "not_run",
            "reason": "optimizer_all_fail_probe" if simulate_optimizer_all_fail
            else "spatial_panel_set_requires_independent_panels" if is_panel_set
            else "disabled",
        }
        if run_m4 and not simulate_optimizer_all_fail and not is_panel_set:
            try:
                m4 = _run_m4_vertical(
                    generation, root, generation_id, schedule, session.calibration,
                    lambda frame_id: read_s13_rgb(by_id[frame_id]),
                )
            except Exception as exc:
                m4 = {
                    "state": "failed_p0_preserved",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                atomic_write_json(generation / "M4_failure.json", {
                    "schema": "gemini305-video-s13-m4-failure/v1",
                    "generation_id": generation_id,
                    "p0_parent_preserved": True,
                    **m4,
                })
        m5: dict[str, object] = {
            "state": "not_run",
            "reason": "optimizer_all_fail_probe" if simulate_optimizer_all_fail
            else "spatial_panel_set_requires_independent_panels" if is_panel_set
            else "m4_failed" if str(m4.get("state", "")).startswith("failed")
            else "disabled",
        }
        if (
            run_m4 and run_m5 and not simulate_optimizer_all_fail and not is_panel_set
            and m4.get("state") == "P1_vertical_candidate_generated"
        ):
            try:
                m5 = _run_m5(
                    generation, root, generation_id, schedule, session.calibration,
                    lambda frame_id: read_s13_rgb(by_id[frame_id]), result.image,
                    selected_hypothesis_ids=tuple(
                        hypothesis_by_frame.get(frame_id, -1) for frame_id in selection.frame_ids
                    ),
                    placement_methods=selection.placement_methods,
                )
            except Exception as exc:
                m5 = {
                    "state": "failed_parent_preserved",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                atomic_write_json(generation / "M5_failure.json", {
                    "schema": "gemini305-video-s13-m5-failure/v1",
                    "generation_id": generation_id,
                    "p0_parent_preserved": True,
                    "p1_candidate_parent_preserved": True,
                    **m5,
                })
        optimizer["p0_hash_unchanged_after_m5"] = (
            p0_hashes_before == verify_p0_completion(generation / "P0")["assets_sha256"]
        )
        if (generation / "P1" / "P1_completion.json").is_file():
            p1_document = json.loads(
                (generation / "P1" / "P1_completion.json").read_text(encoding="utf-8")
            )
            optimizer["p1_hash_unchanged_after_m5"] = all(
                (generation / "P1" / str(name)).is_file()
                and sha256_file(generation / "P1" / str(name)) == expected
                for name, expected in p1_document.get("assets_sha256", {}).items()
            )
        optimizer["p0_hash_unchanged_after_m4"] = (
            p0_hashes_before == verify_p0_completion(generation / "P0")["assets_sha256"]
        )
        optimizer_state = (
            "m5_selected" if m5.get("state") == "P2_selected"
            else "m5_generated_parent_retained" if m5.get("state") == "P2_generated_parent_retained"
            else "m5_failed" if str(m5.get("state", "")).startswith("failed")
            else "m4_candidate_generated" if m4.get("state") == "P1_vertical_candidate_generated"
            else "post_p0_not_run"
        )
        generation_report_path = generation / "report.json"
        generation_report = json.loads(generation_report_path.read_text(encoding="utf-8"))
        generation_report.update({
            "optimizer_state": optimizer_state,
            "m4": m4,
            "m5": m5,
            "optimizer": optimizer,
            "performance": {"stage_seconds": stage_seconds},
        })
        atomic_write_json(generation_report_path, generation_report)
        report = {
            "schema": REPORT_SCHEMA, "algorithm_id": S13_ALGORITHM_ID, "generation_id": generation_id,
            "generation": str(generation),
            "panorama": None if is_panel_set else str(generation / "P0" / "base_panorama_owner_only.png"),
            "spatial_panel_set": str(generation / "P0" / "panels") if is_panel_set else None,
            "panel_navigation_overview": (
                str(generation / "P0" / "panel_navigation_overview_explicit_gaps.png") if is_panel_set else None
            ),
            "layout": str(generation / "P0" / "base_layout.json"),
            "owner_overlay": None if is_panel_set else str(generation / "P0" / "owner_boundary_overlay.png"),
            "pixel_provenance": None if is_panel_set else str(generation / "P0" / "base_pixel_provenance.npz"),
            "column_provenance": None if is_panel_set else str(generation / "P0" / "base_column_provenance.npz"),
            "completion": str(generation / "P0" / "P0_completion.json"),
            "current_base": str(root / "current_base.json"), "time_to_P0_seconds": time_to_p0,
            "total_wall_seconds": time.perf_counter() - run_started, "stage_seconds": stage_seconds,
            "render_state": render_state, "panorama_claim": layout["panorama_claim"],
            "diagnostic_only": True, "production_eligible": False, "production_lock_eligible": False,
            "manual_review_required": True, "optimizer": optimizer,
            "optimizer_state": optimizer_state,
            "m4": m4,
            "m5": m5,
            "motion_graph_connected_telemetry": progress.adjacent_reliable_graph_connected,
            "motion_graph_disconnection_is_fatal": False, "direct_local_delta_is_fatal": False,
            **report_statistics,
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
