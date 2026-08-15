"""The no-generation, in-memory S1.3 M0--M6 execution path.

This module is intentionally separate from the historical sealed-stage
implementation.  It has no resume, seal, pointer, replay-file, or audit-file
authority.  The only user-visible artifacts are the four stage PNG snapshots.
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from .video_s13_base_renderer import render_s13_p0
from .video_s13_m5 import run_s13_m5
from .video_s13_m6 import run_s13_m6
from .video_s13_motion import measure_s13_motion
from .video_s13_progress import build_s13_m3_layout
from .video_s13_replay import S13VerifiedP2
from .video_s13_runtime_state import S13RuntimeContext, S13Stage, S13StageResult
from .video_s13_schedule import plan_s13_m3_schedule
from .video_s13_selection import select_s13_vertical_parent
from .video_s13_session import S13Session, read_s13_rgb
from .video_s13_stage_writer import S13StageImageWriter
from .video_s13_trajectory import S13Trajectory
from .video_s13_vertical import estimate_s13_vertical


def run_s13_fast_pipeline(
    *,
    session: S13Session,
    trajectory: S13Trajectory,
    output: Path,
    analysis_width_px: int,
    normal_target_advance_px: float,
    risky_target_advance_px: float,
    m51_r2_config: object | None,
) -> dict[str, Any]:
    """Render P0--P3 once, keeping every parent and decision in memory."""

    if len(session.frames) < 2:
        raise ValueError("S1.3 fast pipeline requires at least two RGB frames")
    started = time.perf_counter()
    timings: dict[str, float] = {}
    frame_by_id = session.frame_by_id
    decoded: dict[int, np.ndarray] = {}

    def image_loader(frame_id: int) -> np.ndarray:
        image = decoded.get(frame_id)
        if image is None:
            image = read_s13_rgb(frame_by_id[frame_id])
            decoded[frame_id] = image
        return image

    runtime = S13RuntimeContext(uuid.uuid4().hex)
    writer = S13StageImageWriter(output)
    try:
        tick = time.perf_counter()
        motion = measure_s13_motion(session.frames, analysis_width_px=analysis_width_px)
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
        hypothesis_by_frame = {
            step.target_frame_id: step.selected_hypothesis_id for step in layout.lineage
        }
        p0_render = render_s13_p0(
            schedule,
            session.calibration,
            image_loader,
            placement_methods=selection.placement_methods,
            selected_hypothesis_ids=tuple(
                hypothesis_by_frame.get(frame_id, -1) for frame_id in selection.frame_ids
            ),
        )
        p0 = runtime.initialize_p0(S13StageResult(
            runtime.run_id, S13Stage.P0, 0, None, p0_render.image,
            p0_render.valid_mask, {"schedule": schedule, "selection": selection,
                                   "motion": motion, "layout": layout},
        ))
        writer.submit_host_image("P0", p0.image)
        timings["m0_m3"] = time.perf_counter() - tick

        tick = time.perf_counter()
        vertical = estimate_s13_vertical(schedule, session.calibration, image_loader)
        vertical_selection = select_s13_vertical_parent(
            schedule, session.calibration, image_loader, vertical, p0.image
        )
        p1 = runtime.commit(expected_parent=S13Stage.P0, candidate=S13StageResult(
            runtime.run_id, S13Stage.P1, 1, p0.revision, vertical_selection.result.image,
            vertical_selection.result.valid_mask,
            {"vertical": vertical_selection.solution, "schedule": schedule,
             "selection": selection, "p0": p0_render},
        ))
        writer.submit_host_image("P1", p1.image)
        timings["m4"] = time.perf_counter() - tick

        tick = time.perf_counter()
        m5 = run_s13_m5(
            schedule, session.calibration, image_loader, vertical_selection.solution, p1.image,
            parent_stage_sha256="in-memory-p1", parent_result_sha256="in-memory-p1",
            p0_ancestor_completion_sha256="in-memory-p0",
            selected_hypothesis_ids=tuple(
                hypothesis_by_frame.get(frame_id, -1) for frame_id in selection.frame_ids
            ),
            placement_methods=selection.placement_methods,
            m51_r2_config=m51_r2_config,
        )
        p2 = runtime.commit(expected_parent=S13Stage.P1, candidate=S13StageResult(
            runtime.run_id, S13Stage.P2, 2, p1.revision, m5.final_result.image,
            m5.final_result.valid_mask,
            {"m5": m5, "schedule": schedule, "selection": selection},
        ))
        writer.submit_host_image("P2", p2.image)
        timings["m5"] = time.perf_counter() - tick

        tick = time.perf_counter()
        p2_runtime = S13VerifiedP2(
            root=output,
            completion={"source_count": len(schedule.assignments)},
            completion_sha256="",
            result_image=p2.image,
            valid_mask=p2.valid,
            provenance=m5.final_result.pixel_provenance,
            transactions=tuple(pair.transaction for pair in m5.pairs),
            replay_pairs=m5.replay_pairs,
            immutable_sha256={},
        )
        # The old M7 contributes no image candidate: its only selected R0 path
        # copied P3 to P4.  Every pair therefore enters automatic C2E and keeps
        # C0 until a future local candidate has a demonstrated improvement.
        p3_render = run_s13_m6(p2_runtime, image_loader)
        c2e = tuple("C0_keep_standard" for _ in m5.pairs)
        p3 = runtime.commit(expected_parent=S13Stage.P2, candidate=S13StageResult(
            runtime.run_id, S13Stage.P3, 3, p2.revision, p3_render.visual_panorama,
            p3_render.valid_mask,
            {"selected_c2e_candidates": c2e, "m6_performance": p3_render.performance},
        ))
        writer.submit_host_image("P3", p3.image)
        timings["m6"] = time.perf_counter() - tick
        paths = writer.close()
    except BaseException:
        writer.close()
        raise
    timings["total"] = time.perf_counter() - started
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
        "sha_call_count": 0,
        "png_write_count": 4,
        "jpg_write_count": 0,
        "json_write_count": 0,
        "npz_write_count": 0,
    }


__all__ = ["run_s13_fast_pipeline"]
