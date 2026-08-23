"""The no-generation, in-memory S1.3 M0--M6 execution path.

This module is intentionally separate from the historical sealed-stage
implementation.  It has no resume, seal, pointer, replay-file, or audit-file
authority.  The only user-visible artifacts are the four stage PNG snapshots.
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any, Callable, Literal, Mapping

import numpy as np

from .video_s13_base_renderer import render_s13_p0
from .video_s13_c2e import select_s13_fast_c2e
from .video_s13_m5 import (
    reset_s13_m5_resident_batch, reset_s13_m5_resident_remap, reset_s13_m5_runtime_unsealed,
    run_s13_m5, set_s13_m5_resident_batch, set_s13_m5_resident_remap,
    set_s13_m5_runtime_unsealed,
)
from .video_s13_m6 import run_s13_m6, run_s13_m6_cuda_v2, run_s13_m6_cuda_v3
from .video_s13_motion import measure_s13_motion, reliable_step1_direction_evidence
from .video_s13_frame_store import S13FrameStore
from .video_s13_progress import build_s13_m3_layout
from .video_s13_replay import S13VerifiedP2
from .video_s13_runtime_state import S13RuntimeContext, S13Stage, S13StageResult
from .video_s13_schedule import plan_s13_m3_schedule
from .video_s13_selection import select_s13_vertical_parent
from .video_s13_session import S13Session, S13ValidatedRgbHandoff
from .video_s13_stage_writer import S13StageImageWriter
from .video_s13_trajectory import S13Trajectory
from .video_s13_vertical import estimate_s13_vertical, render_s13_p1_from_raw


def _submit_immutable_stage(
    writer: S13StageImageWriter,
    stage: str,
    image: np.ndarray,
) -> None:
    """Transfer an immutable stage buffer to the asynchronous PNG writer."""

    if not writer.is_enabled(stage):
        return
    buffer = np.asarray(image)
    if not buffer.flags.c_contiguous:
        raise ValueError("S1.3 fast stage image must be contiguous before writer transfer")
    buffer.setflags(write=False)
    writer.submit_owned_host_image(stage, buffer)


def _measure_fast_motion(
    frames,
    *,
    analysis_width_px: int,
    prepared_analysis,
    prepared_gradients,
    policy: Literal["full_reference", "deferred_step4"],
):
    motion_profile: dict[str, object] = {}
    if policy == "full_reference":
        motion = measure_s13_motion(
            frames, analysis_width_px=analysis_width_px,
            prepared_analysis=prepared_analysis,
            prepared_gradients=prepared_gradients, profile=motion_profile,
        )
        direction_evidence = reliable_step1_direction_evidence(motion)
        step4_required = True
        step4_reason = "full_reference"
    elif policy == "deferred_step4":
        primary_profile: dict[str, object] = {}
        primary_motion = measure_s13_motion(
            frames, analysis_width_px=analysis_width_px, steps=(1, 2),
            prepared_analysis=prepared_analysis,
            prepared_gradients=prepared_gradients, profile=primary_profile,
        )
        direction_evidence = reliable_step1_direction_evidence(primary_motion)
        step4_required = not direction_evidence
        step4_reason = (
            "no_reliable_step1_direction"
            if step4_required else "reliable_step1_direction_available"
        )
        if step4_required:
            step4_profile: dict[str, object] = {}
            step4_motion = measure_s13_motion(
                frames, analysis_width_px=analysis_width_px, steps=(4,),
                prepared_analysis=prepared_analysis,
                prepared_gradients=prepared_gradients, profile=step4_profile,
            )
            motion = (*primary_motion, *step4_motion)
            motion_profile = {
                **primary_profile,
                **{
                    key: value for key, value in step4_profile.items()
                    if key.startswith("step4_")
                },
                "profiled_wall_seconds": float(primary_profile["profiled_wall_seconds"])
                + float(step4_profile["profiled_wall_seconds"]),
                "profiling_bookkeeping_seconds": float(
                    primary_profile["profiling_bookkeeping_seconds"]
                ) + float(step4_profile["profiling_bookkeeping_seconds"]),
                "requested_steps": [1, 2, 4],
            }
        else:
            motion = primary_motion
            motion_profile = primary_profile
    else:
        raise ValueError(f"unsupported S1.3 motion execution policy: {policy}")
    step4_computed = any(edge.step == 4 for edge in motion)
    motion_profile.update({
        "step4_requested": bool(step4_required),
        "step4_computed": step4_computed,
        "step4_requirement_reason": step4_reason,
    })
    execution = {
        "schema": "gemini305-video-s13-motion-execution/v1",
        "policy": (
            "full_reference" if policy == "full_reference"
            else "deferred_step4_unless_direction_fallback_required"
        ),
        "primary_steps": [1, 2],
        "fallback_steps": [4],
        "reliable_step1_direction_count": len(direction_evidence),
        "step4_required": bool(step4_required),
        "step4_requirement_reason": step4_reason,
        "step4_computed": step4_computed,
        "merged_edge_order": list(dict.fromkeys(edge.step for edge in motion)),
    }
    return motion, motion_profile, execution


def _build_fast_authority(
    *, schedule, m5, selected_replay, c2e, owner_only_pairs, provenance, m62,
) -> dict[str, object]:
    """Expose the exact in-memory pixel authorities for production comparison.

    Arrays deliberately remain in memory.  The production wrapper decides
    which summaries belong in JSON and publishes the owner-frame map as pixel
    provenance without making the candidate runner perform extra I/O.
    """

    source_frame_ids = tuple(int(item.frame_id) for item in schedule.assignments)
    owner_source = np.asarray(provenance["owner_source_index"], dtype=np.int32)
    owner_frame = np.asarray(provenance["owner_frame_id"], dtype=np.int32)
    unique_sources, source_counts = np.unique(owner_source, return_counts=True)
    frame_by_source = {
        int(item.source_index): int(item.frame_id) for item in schedule.assignments
    }
    source_pixel_counts = tuple({
        "source_index": int(source_index),
        "frame_id": frame_by_source.get(int(source_index), -1),
        "pixel_count": int(pixel_count),
    } for source_index, pixel_count in zip(unique_sources, source_counts, strict=True))

    pair_seam_decisions = tuple({
        "pair_index": int(index),
        "left_frame_id": int(replay.left_frame_id),
        "right_frame_id": int(replay.right_frame_id),
        "corridor_x0": int(replay.corridor_x0),
        "corridor_x1": int(replay.corridor_x1),
        "seam_x_by_row": np.asarray(replay.seam_x_by_row, dtype=np.int32),
        "m5_selected_seam_model": str(
            pair.transaction.get(
                "selected_seam_model", pair.transaction.get("seam_model", "")
            )
        ),
        "m5_transaction": dict(pair.transaction),
        "c2e_decision": str(c2e[index]),
        "owner_only": index in owner_only_pairs,
    } for index, (pair, replay) in enumerate(
        zip(m5.pairs, selected_replay, strict=True)
    ))

    plan = None if m62 is None else m62.get("plan")
    if plan is None:
        selected_photometric_model = None
        blend_models: tuple[str, ...] = ()
        quality_cut_pair_indices: tuple[int, ...] = ()
        m63_audit: dict[str, object] = {}
        photometric_candidate_audits: tuple[dict[str, object], ...] = ()
    else:
        selected_photometric_model = str(plan.photometric_solution.model_family)
        blend_models = tuple(str(item.transaction.model) for item in plan.blend_plans)
        quality_cut_pair_indices = tuple(
            int(value)
            for value in (() if plan.m63 is None else plan.m63.quality_cut_pair_indices)
        )
        m63_audit = {} if plan.m63 is None else dict(plan.m63.audit)
        photometric_candidate_audits = tuple(
            dict(value) for value in plan.photometric_solution.candidate_audits
        )

    assignments = tuple({
        "assignment_index": int(item.assignment_index),
        "source_index": int(item.source_index),
        "frame_id": int(item.frame_id),
        "center_x": float(item.center_x),
        "left_x": int(item.left_x),
        "right_x": int(item.right_x),
        "zero_width": bool(item.zero_width),
    } for item in schedule.assignments)
    return {
        "source_frame_ids": source_frame_ids,
        "schedule": {
            "canvas_shape": (int(schedule.canvas_height), int(schedule.canvas_width)),
            "canvas_left": int(schedule.canvas_left),
            "canvas_right": int(schedule.canvas_right),
            "boundaries": tuple(int(value) for value in schedule.boundaries),
            "assignments": assignments,
        },
        "owner": {
            "source_index_map": owner_source,
            "frame_id_map": owner_frame,
            "summary": {
                "shape": tuple(int(value) for value in owner_source.shape),
                "source_pixel_counts": source_pixel_counts,
                "invalid_pixel_count": int(np.count_nonzero(owner_source < 0)),
            },
        },
        "pair_seam_decisions": pair_seam_decisions,
        "m6": {
            "selected_photometric_model": selected_photometric_model,
            "blend_models": blend_models,
            "b0_count": int(sum(model == "B0_owner_only" for model in blend_models)),
            "b1_count": int(sum(model.startswith("B1_") for model in blend_models)),
            "quality_cut_pair_indices": quality_cut_pair_indices,
            "m63_audit": m63_audit,
            "photometric_candidate_audits": photometric_candidate_audits,
            "c2e_decisions": tuple(str(value) for value in c2e),
            "c2e_owner_only_pair_indices": tuple(sorted(int(value) for value in owner_only_pairs)),
        },
    }


def run_s13_fast_pipeline(
    *,
    session: S13Session,
    trajectory: S13Trajectory,
    output: Path,
    analysis_width_px: int,
    normal_target_advance_px: float,
    risky_target_advance_px: float,
    m51_r2_config: object | None,
    resident_runtime: Any | None = None,
    p0_resident_remap: Callable[[int, np.ndarray, np.ndarray, np.ndarray], np.ndarray] | None = None,
    p0_resident_device_remap: Callable[[int, np.ndarray, np.ndarray, np.ndarray], Any] | None = None,
    vertical_exact_seam_probes: bool = False,
    p1_reference_remap: bool = False,
    m6_cuda_v2: bool = False,
    m6_cuda_v3: bool = False,
    m62_equivalence: bool = False,
    m62_options: Mapping[str, object] | None = None,
    post_p2_fixture: Path | None = None,
    validated_rgb_handoff: S13ValidatedRgbHandoff | None = None,
    motion_execution_policy: Literal["full_reference", "deferred_step4"] = "full_reference",
    m5_execution_mode: Literal["full_reference", "candidate_final_authority"] = "full_reference",
    m5_pair_base_atlas: bool = False,
    m5_p0_map_mode: Literal["full_reference", "compact_exact_window"] = "full_reference",
    m5_compact_full_reference_fallback: bool = True,
    stage_output_stages: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Render P0--P3 once, keeping every parent and decision in memory."""

    if len(session.frames) < 2:
        raise ValueError("S1.3 fast pipeline requires at least two RGB frames")
    started = time.perf_counter()
    timings: dict[str, float] = {}
    frame_by_id = session.frame_by_id
    validated_rgb_bytes = (
        0 if validated_rgb_handoff is None else validated_rgb_handoff.retained_bytes
    )
    frame_store = S13FrameStore(maximum_bytes=max(
        512 * 1024 * 1024,
        validated_rgb_bytes + 32 * 1024 * 1024,
    ))
    handoff_tick = time.perf_counter()
    if validated_rgb_handoff is not None:
        frame_store.adopt_validated_raw_bulk(
            frame_by_id, validated_rgb_handoff.take_all()
        )
    timings["m0_m3.handoff"] = time.perf_counter() - handoff_tick

    def image_loader(frame_id: int) -> np.ndarray:
        return frame_store.raw_bgr(frame_by_id[frame_id])

    runtime = S13RuntimeContext(uuid.uuid4().hex)
    writer = S13StageImageWriter(
        output, max_pending=4, enabled_stages=stage_output_stages,
    )
    try:
        tick = time.perf_counter()
        prepared_analysis = frame_store.prefetch_analysis(
            session.frames, analysis_width_px, workers=2
        )
        prepared_gradients = tuple(
            frame_store.analysis_gradient(frame, analysis_width_px)
            for frame in session.frames
        )
        motion, motion_profile, motion_execution = _measure_fast_motion(
            session.frames, analysis_width_px=analysis_width_px,
            prepared_analysis=prepared_analysis,
            prepared_gradients=prepared_gradients,
            policy=motion_execution_policy,
        )
        del prepared_analysis, prepared_gradients
        frame_store.release_analysis_arrays()
        layout = build_s13_m3_layout(session.frames, motion, trajectory)
        if not layout.progress.spatial:
            raise ValueError("S1.3 fast pipeline has no spatial scan segment")
        schedule_plan = plan_s13_m3_schedule(
            layout.progress,
            motion,
            session.calibration,
            normal_target_advance_px=normal_target_advance_px,
            risky_target_advance_px=risky_target_advance_px,
            segment_break_pairs=layout.segment_break_pairs,
        )
        if len(schedule_plan.schedules) != 1:
            raise ValueError("S1.3 fast pipeline does not publish panel sets")
        selection, schedule = schedule_plan.selections[0], schedule_plan.schedules[0]
        frame_store.retain_raw({
            assignment.frame_id
            for assignment in schedule.assignments
            if not assignment.zero_width
        })
        timings["m0_m3.motion_and_layout"] = time.perf_counter() - tick
        preload_tick = time.perf_counter()
        if resident_runtime is not None:
            resident_runtime.preload_sources({
                assignment.frame_id: image_loader(assignment.frame_id)
                for assignment in schedule.assignments if not assignment.zero_width
            })
        timings["m0_m3.preload"] = time.perf_counter() - preload_tick
        hypothesis_by_frame = {
            step.target_frame_id: step.selected_hypothesis_id for step in layout.lineage
        }
        render_tick = time.perf_counter()
        p0_render = render_s13_p0(
            schedule,
            session.calibration,
            image_loader,
            placement_methods=selection.placement_methods,
            selected_hypothesis_ids=tuple(
                hypothesis_by_frame.get(frame_id, -1) for frame_id in selection.frame_ids
            ),
            resident_remap=p0_resident_remap,
            resident_device_remap=p0_resident_device_remap,
            resident_stage=resident_runtime if p0_resident_device_remap is not None else None,
        )
        timings["p0.render"] = time.perf_counter() - render_tick
        p0 = runtime.initialize_p0(S13StageResult(
            runtime.run_id, S13Stage.P0, 0, None, p0_render.image,
            p0_render.valid_mask, {"schedule": schedule, "selection": selection,
                                   "motion": motion, "layout": layout},
        ))
        submit_tick = time.perf_counter()
        _submit_immutable_stage(writer, "P0", p0.image)
        timings["p0.submit"] = time.perf_counter() - submit_tick
        timings["m0_m3"] = time.perf_counter() - tick

        tick = time.perf_counter()
        estimate_tick = time.perf_counter()
        vertical = estimate_s13_vertical(schedule, session.calibration, image_loader)
        timings["m4.estimate"] = time.perf_counter() - estimate_tick
        selection_tick = time.perf_counter()
        vertical_selection = select_s13_vertical_parent(
            schedule, session.calibration, image_loader, vertical, p0.image,
            exact_seam_probes=vertical_exact_seam_probes,
            reference_final_remap=p1_reference_remap,
        )
        timings["m4.candidate_decision"] = time.perf_counter() - selection_tick
        final_render_tick = time.perf_counter()
        p1_render = (
            render_s13_p1_from_raw(
                schedule, session.calibration, image_loader, vertical_selection.solution,
                resident_stage=resident_runtime,
                resident_device_remap=p0_resident_device_remap,
            ) if p0_resident_device_remap is not None and not (
                vertical_exact_seam_probes or p1_reference_remap
            ) else vertical_selection.result
        )
        timings["m4.final_p1_render"] = time.perf_counter() - final_render_tick
        p1 = runtime.commit(expected_parent=S13Stage.P0, candidate=S13StageResult(
            runtime.run_id, S13Stage.P1, 1, p0.revision, p1_render.image,
            p1_render.valid_mask,
            {"vertical": vertical_selection.solution, "schedule": schedule,
             "selection": selection, "p0": p0_render},
        ))
        submit_tick = time.perf_counter()
        _submit_immutable_stage(writer, "P1", p1.image)
        timings["m4.submit"] = time.perf_counter() - submit_tick
        timings["m4"] = timings["m4.total"] = time.perf_counter() - tick

        tick = time.perf_counter()
        unsealed_token = set_s13_m5_runtime_unsealed(True)
        resident_m5_token = set_s13_m5_resident_remap(
            resident_runtime.remap_host_source if p0_resident_device_remap is not None else None
        )
        resident_m5_batch_token = None
        if p0_resident_device_remap is not None:
            from .video_s13_cuda_runtime import S13DeviceRemapRequest

            def resident_m5_batch(entries):
                return resident_runtime.remap_batch_to_host(tuple(
                    S13DeviceRemapRequest(
                        request_id=f"m5-base-{index}-{frame_id}", frame_id=frame_id,
                        map_u=map_u, map_v=map_v,
                    )
                    for index, (frame_id, _raw, map_u, map_v) in enumerate(entries)
                ))

            resident_m5_batch_token = set_s13_m5_resident_batch(resident_m5_batch)
        try:
            # P2 is the source-map authority for C2E and M6.  Keep its final
            # pixels on the established CPU reference remap path: composing a
            # sparse owner canvas on CUDA can turn a source-map discontinuity
            # into an entire visible vertical strip. CUDA remains responsible
            # for bounded M5 corridor sampling, but never for P2 publication.
            final_image_composer = None
            m5 = run_s13_m5(
                schedule, session.calibration, image_loader, vertical_selection.solution, p1.image,
                parent_stage_sha256="in-memory-p1", parent_result_sha256="in-memory-p1",
                p0_ancestor_completion_sha256="in-memory-p0",
                selected_hypothesis_ids=tuple(
                    hypothesis_by_frame.get(frame_id, -1) for frame_id in selection.frame_ids
                ),
                placement_methods=selection.placement_methods,
                m51_r2_config=m51_r2_config,
                final_image_composer=final_image_composer,
                execution_mode=m5_execution_mode,
                pair_base_atlas=m5_pair_base_atlas,
                p0_map_mode=m5_p0_map_mode,
                compact_full_reference_fallback=m5_compact_full_reference_fallback,
            )
        finally:
            if resident_m5_batch_token is not None:
                reset_s13_m5_resident_batch(resident_m5_batch_token)
            reset_s13_m5_resident_remap(resident_m5_token)
            reset_s13_m5_runtime_unsealed(unsealed_token)
        p2 = runtime.commit(expected_parent=S13Stage.P1, candidate=S13StageResult(
            runtime.run_id, S13Stage.P2, 2, p1.revision,
            m5.final_result.image,
            m5.final_result.valid_mask,
            {"m5": m5, "schedule": schedule, "selection": selection},
        ))
        submit_tick = time.perf_counter()
        _submit_immutable_stage(writer, "P2", p2.image)
        timings["m5.submit"] = time.perf_counter() - submit_tick
        timings["m5"] = timings["m5.total"] = time.perf_counter() - tick

        tick = time.perf_counter()
        c2e_tick = time.perf_counter()
        selected_replay, c2e, owner_only_pairs = select_s13_fast_c2e(
            m5.pairs, m5.replay_pairs, image_loader,
            resident_remap=(
                resident_runtime.remap_resident_frame
                if p0_resident_device_remap is not None else None
            ),
        )
        timings["c2e.decision"] = time.perf_counter() - c2e_tick
        # C3 can replace a right-source map in a compact pair corridor.  M6's
        # source-map assembler starts from P2 owner provenance, so mirror that
        # same in-memory map only for pixels owned by the changed source.
        changed_replay = any(
            not np.array_equal(original.right_source_u, chosen.right_source_u)
            or not np.array_equal(original.right_source_v, chosen.right_source_v)
            for original, chosen in zip(m5.replay_pairs, selected_replay, strict=True)
        )
        provenance = m5.final_result.pixel_provenance
        c2e_copy_count = 0
        if changed_replay:
            provenance = dict(provenance)
            for name in ("source_u", "source_v", "valid"):
                provenance[name] = np.asarray(provenance[name]).copy()
            c2e_copy_count = 1
        owner_source = np.asarray(provenance["owner_source_index"], dtype=np.int32)
        for original, chosen in zip(m5.replay_pairs, selected_replay, strict=True):
            if np.array_equal(original.right_source_u, chosen.right_source_u) and np.array_equal(original.right_source_v, chosen.right_source_v):
                continue
            roi = np.s_[:, chosen.corridor_x0:chosen.corridor_x1]
            owned = owner_source[roi] == chosen.right_source_index
            for name, values in (("source_u", chosen.right_source_u), ("source_v", chosen.right_source_v)):
                provenance[name][roi][owned] = values[owned]
            provenance["valid"][roi][owned] = chosen.right_valid[owned]
        timings["c2e.provenance_patch"] = time.perf_counter() - c2e_tick - timings["c2e.decision"]
        timings["c2e.total"] = time.perf_counter() - c2e_tick
        p2_runtime = S13VerifiedP2(
            root=output,
            completion={"source_count": len(schedule.assignments)},
            completion_sha256="",
            result_image=p2.image,
            valid_mask=p2.valid,
            provenance=provenance,
            transactions=tuple(pair.transaction for pair in m5.pairs),
            replay_pairs=selected_replay,
            immutable_sha256={},
        )
        if post_p2_fixture is not None:
            from .video_s13_post_p2_fixture import export_s13_post_p2_fixture

            export_s13_post_p2_fixture(
                post_p2_fixture,
                p2_runtime,
                session_path=session.root,
                force_owner_only_pair_indices=frozenset(owner_only_pairs),
            )
        m62: dict[str, Any] | None = None
        if m62_equivalence:
            from .video_s13_m62_runner import run_s13_m62_cpu_authoritative
            from .video_s13_blend import S13BlendConfig
            from .video_s13_photometric import S13PhotometricConfig
            from .video_s13_m63_solver import S13M63Config
            options = dict(m62_options or {})
            photometric_values = dict(options.get("photometric", {}))
            photometric_values.update(dict(options.get("selection", {})))
            blend_values = dict(options.get("blend", {}))
            allowed_photo = set(S13PhotometricConfig.__dataclass_fields__)
            allowed_blend = set(S13BlendConfig.__dataclass_fields__)
            execution_mode = str(dict(options.get("execution", {})).get(
                "mode", "parity_test"
            ))
            p3_render, m62 = run_s13_m62_cpu_authoritative(
                p2_runtime, image_loader, cuda_runtime=resident_runtime,
                retain_runtime_details=execution_mode in {"shadow_audit", "parity_test"},
                force_owner_only_pair_indices=owner_only_pairs,
                photometric_config=S13PhotometricConfig(**{
                    key: value for key, value in photometric_values.items() if key in allowed_photo
                }),
                blend_config=S13BlendConfig(**{
                    key: value for key, value in blend_values.items() if key in allowed_blend
                }),
                execution_mode=execution_mode,
                m63_config=S13M63Config.from_document(options.get("m63")),
            )
        elif m6_cuda_v3:
            if p0_resident_device_remap is None or resident_runtime is None:
                raise ValueError("S1.3 M6 CUDA v3 requires the resident device runtime")
            p3_render = run_s13_m6_cuda_v3(
                p2_runtime, image_loader, cuda_runtime=resident_runtime,
                retain_runtime_details=False, force_owner_only_pair_indices=owner_only_pairs,
            )
        elif m6_cuda_v2:
            if p0_resident_device_remap is None or resident_runtime is None:
                raise ValueError("S1.3 M6 CUDA v2 requires the resident device runtime")
            p3_render = run_s13_m6_cuda_v2(
                p2_runtime, image_loader, cuda_runtime=resident_runtime,
                retain_runtime_details=False,
                force_owner_only_pair_indices=owner_only_pairs,
            )
        else:
            p3_render = run_s13_m6(
                p2_runtime, image_loader, retain_runtime_details=False,
                force_owner_only_pair_indices=owner_only_pairs,
            )
        p3 = runtime.commit(expected_parent=S13Stage.P2, candidate=S13StageResult(
            runtime.run_id, S13Stage.P3, 3, p2.revision, p3_render.visual_panorama,
            p3_render.valid_mask,
            {"selected_c2e_candidates": c2e, "m6_performance": p3_render.performance},
        ))
        submit_tick = time.perf_counter()
        _submit_immutable_stage(writer, "P3", p3.image)
        timings["m6.submit"] = time.perf_counter() - submit_tick
        timings["m6"] = timings["m6.total"] = time.perf_counter() - tick
        paths = writer.close()
    except BaseException:
        writer.close()
        raise
    timings["writer.close_wait"] = writer.close_wait_seconds
    timings["total"] = time.perf_counter() - started
    authority = _build_fast_authority(
        schedule=schedule,
        m5=m5,
        selected_replay=selected_replay,
        c2e=c2e,
        owner_only_pairs=owner_only_pairs,
        provenance=provenance,
        m62=m62,
    )
    return {
        "run_id": runtime.run_id,
        "paths": tuple(str(path) for path in paths),
        "timings": timings,
        "p0": p0,
        "p1": p1,
        "p2": p2,
        "p3": p3,
        "c2e": c2e,
        "m7_call_count": 0,
        "c2e_call_count": len(c2e),
        "sha_call_count": 0,
        "png_write_count": len(paths),
        "stage_output_stages": writer.enabled_stages,
        "jpg_write_count": 0,
        "json_write_count": 0,
        "npz_write_count": 0,
        "writer_audit": {
            "writer_snapshot_copy_count": writer.snapshot_copy_count,
            "writer_submit_wait_seconds": writer.submit_wait_seconds,
            "writer_close_wait_seconds": writer.close_wait_seconds,
            "writer_pending_peak": writer.pending_peak,
            "writer_submit_blocking_count": writer.submit_blocking_count,
        },
        "c2e_full_provenance_copy_count": c2e_copy_count,
        "m6_performance": dict(p3_render.performance),
        "m5_performance": dict(m5.performance),
        "motion_profile": motion_profile,
        "motion_execution_policy": motion_execution,
        "m4_probe": {
            "mode": "legacy_exact_probe" if vertical_exact_seam_probes else "full_reference",
            "plan_seconds": 0.0,
            "global_atlas_render_seconds": 0.0,
            "full_local_atlas_render_seconds": 0.0,
            "metric_seconds": float(timings.get("m4.candidate_decision", 0.0)),
            "final_full_p1_guard_seconds": 0.0,
            "slot_count": len(vertical.pairs),
            "packed_width_px": 0,
            "source_remap_count": 0,
            "legacy_probe_shadow_count": 0,
            "full_reference_shadow_count": 0,
            "metric_mismatch_count": 0,
            "decision_mismatch_count": 0,
            "fallback_count": 0,
        },
        "m5_expected_support": dict(
            m5.performance.get("m5_expected_support", {})
        ),
        "m62": m62,
        "frame_store": frame_store.report().__dict__,
        "authority": authority,
    }


__all__ = ["run_s13_fast_pipeline"]
