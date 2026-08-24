"""Committed-ledger S013 M0--M3 shadow and online-P0 state types."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import time
from typing import Mapping

import numpy as np

from .session import CameraIntrinsics
from .video_s13_base_renderer import (
    S13P0AssignmentRender,
    S13P0Result,
    render_s13_p0_assignment,
)
from .video_s13_contract import (
    S13_VISUAL_CONTINUITY_ALGORITHM_ID,
    S13_VISUAL_CONTINUITY_IMPLEMENTATION_ID,
)
from .video_s13_fast_pipeline import S13P0Continuation
from .video_s13_motion import (
    S13MotionAccumulator,
    S13MotionSnapshot,
    prepare_s13_motion_frame,
)
from .video_s13_online_checkpoint import (
    S13OnlineCheckpointChain,
    array_sha256,
)
from .video_s13_progress import (
    S13M3Layout,
    build_s13_m3_layout,
    selected_hypothesis_ids_for_spatial_sources,
)
from .video_s13_schedule import plan_s13_m3_schedule
from .video_s13_session import S13RenderFrame, read_s13_rgb
from .video_s13_runtime_state import S13Stage, S13StageResult
from .video_s13_trajectory import S13Trajectory


@dataclass(frozen=True)
class S13OnlineFrontiers:
    committed_frame_index: int = -1
    prepared_frame_index: int = -1
    motion_complete_frame_index: int = -1
    layout_current_frame_index: int = -1
    selected_source_count: int = 0
    sealed_source_count: int = 0
    sealed_canvas_x_end: int = 0
    mutable_canvas_x_begin: int = 0
    preview_frame_index: int = -1


@dataclass(frozen=True)
class S13ScheduleSemanticAssignment:
    frame_id: int
    quantized_center_x: int
    left_x: int
    right_x: int
    placement_method: str
    selected_hypothesis_id: int


@dataclass(frozen=True)
class S13OnlineShadowSnapshot:
    frontiers: S13OnlineFrontiers
    motion: S13MotionSnapshot
    layout: S13M3Layout | None
    selection: object | None
    schedule: object | None
    semantic_assignments: tuple[S13ScheduleSemanticAssignment, ...]
    current_p0: S13P0Result | None


@dataclass(frozen=True)
class S13FrozenP0Authority:
    schema: str
    algorithm_id: str
    implementation_id: str
    production_config_sha256: str
    session_root: Path
    manifest_sha256: str
    calibration_sha256: str
    frames_csv_sha256: str
    committed_frames: tuple[object, ...]
    continuation: S13P0Continuation
    sealed_prefix_source_count: int
    finalized_tail_source_count: int
    checkpoint_chain_sha256: str
    p0_pixel_sha256: str
    p0_prefix_reused_fraction: float
    final_metadata_closure_seconds: float
    tail_finalize_seconds: float
    rollback_checkpoint_used: bool
    rollback_reasons: tuple[str, ...]
    full_m0_m3_recomputed: bool = False
    reuse_level: str = "frozen_p0_authority_v1"


def calibration_from_live_document(document: Mapping[str, object]) -> CameraIntrinsics:
    intrinsic = document.get("color_intrinsic")
    distortion = document.get("color_distortion", {})
    if not isinstance(intrinsic, Mapping) or not isinstance(distortion, Mapping):
        raise ValueError("S1.3 online shadow requires color calibration")
    return CameraIntrinsics(
        width=int(intrinsic["width"]),
        height=int(intrinsic["height"]),
        fx=float(intrinsic["fx"]),
        fy=float(intrinsic["fy"]),
        cx=float(intrinsic["cx"]),
        cy=float(intrinsic["cy"]),
        distortion=tuple(
            float(distortion.get(name, 0.0))
            for name in ("k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6")
        ),
    )


def _ignore_pose_trajectory() -> S13Trajectory:
    return S13Trajectory(
        source="ignore_pose_online_shadow",
        source_path=None,
        poses_by_frame_id={},
        pose_origin_by_frame_id={},
        islands=(),
        audit={"source": "ignore_pose", "direct_pose_count": 0},
    )


def semantic_assignments(
    schedule: object,
    selection: object,
    selected_hypothesis_ids: tuple[int, ...],
) -> tuple[S13ScheduleSemanticAssignment, ...]:
    assignments = getattr(schedule, "assignments")
    placement_methods = getattr(selection, "placement_methods")
    return tuple(
        S13ScheduleSemanticAssignment(
            frame_id=int(item.frame_id),
            quantized_center_x=int(round(float(item.center_x) * 1_000_000.0)),
            left_x=int(item.left_x),
            right_x=int(item.right_x),
            placement_method=str(placement_methods[index]),
            selected_hypothesis_id=int(selected_hypothesis_ids[index]),
        )
        for index, item in enumerate(assignments)
    )


class S13OnlineP0Engine:
    """R1 committed-ledger shadow; it does not render or reuse formal pixels."""

    def __init__(
        self,
        *,
        session_root: Path,
        calibration: CameraIntrinsics,
        analysis_width_px: int = 424,
        normal_target_advance_px: float = 8.0,
        risky_target_advance_px: float = 5.0,
    ) -> None:
        self.session_root = session_root.resolve()
        self.calibration = calibration
        self.analysis_width_px = int(analysis_width_px)
        self.normal_target_advance_px = float(normal_target_advance_px)
        self.risky_target_advance_px = float(risky_target_advance_px)
        self._frames: list[S13RenderFrame] = []
        self._frame_by_id: dict[int, S13RenderFrame] = {}
        self._raw_by_frame_id: dict[int, np.ndarray] = {}
        self._assignment_cache: dict[
            tuple[int, S13ScheduleSemanticAssignment], S13P0AssignmentRender
        ] = {}
        self._current_renders: tuple[S13P0AssignmentRender, ...] = ()
        self._previous_semantic: tuple[S13ScheduleSemanticAssignment, ...] = ()
        self._checkpoint_chain = S13OnlineCheckpointChain()
        self._canonical_directions: set[int] = set()
        self._last_layout_commit_ns = 0
        self._layout_interval_ns = 200_000_000
        self._motion = S13MotionAccumulator()
        self._snapshot = S13OnlineShadowSnapshot(
            frontiers=S13OnlineFrontiers(),
            motion=self._motion.snapshot(),
            layout=None,
            selection=None,
            schedule=None,
            semantic_assignments=(),
            current_p0=None,
        )

    def _raw(self, frame_id: int) -> np.ndarray:
        image = self._raw_by_frame_id.get(frame_id)
        if image is None:
            image = read_s13_rgb(self._frame_by_id[frame_id])
            self._raw_by_frame_id[frame_id] = image
        return image

    def _render_current_p0(
        self,
        schedule: object,
        selection: object,
        semantic: tuple[S13ScheduleSemanticAssignment, ...],
        hypothesis_ids: tuple[int, ...],
    ) -> S13P0Result:
        assignments = getattr(schedule, "assignments")
        placement_methods = getattr(selection, "placement_methods")
        height = int(getattr(schedule, "canvas_height"))
        width = int(getattr(schedule, "canvas_width"))
        image = np.zeros((height, width, 3), dtype=np.uint8)
        cache: dict[
            tuple[int, S13ScheduleSemanticAssignment], S13P0AssignmentRender
        ] = {}
        renders: list[S13P0AssignmentRender] = []
        for index, assignment in enumerate(assignments):
            if assignment.zero_width:
                continue
            key = (index, semantic[index])
            rendered = self._assignment_cache.get(key)
            if rendered is None:
                rendered = render_s13_p0_assignment(
                    assignment,
                    self.calibration,
                    self._raw(int(assignment.frame_id)),
                    placement_method=str(placement_methods[index]),
                    selected_hypothesis_id=int(hypothesis_ids[index]),
                )
            cache[key] = rendered
            renders.append(rendered)
            image[:, rendered.left_x:rendered.right_x] = rendered.image_roi
        self._assignment_cache = cache
        self._current_renders = tuple(renders)
        if not renders:
            raise ValueError("S1.3 current P0 has no positive-width assignment")
        spatial_names = tuple(
            name for name, value in renders[0].pixel_provenance_roi.items()
            if np.asarray(value).ndim == 2
        )
        pixel: dict[str, np.ndarray] = {}
        for name in spatial_names:
            sample = np.asarray(renders[0].pixel_provenance_roi[name])
            shape = (height, width)
            if np.issubdtype(sample.dtype, np.floating):
                fill = 0.0 if name in {"secondary_weight"} else np.nan
            elif np.issubdtype(sample.dtype, np.bool_):
                fill = False
            else:
                fill = -1
            pixel[name] = np.full(shape, fill, dtype=sample.dtype)
        for rendered in renders:
            roi = np.s_[:, rendered.left_x:rendered.right_x]
            for name in spatial_names:
                pixel[name][roi] = rendered.pixel_provenance_roi[name]
        pixel["placement_method_names"] = np.asarray(placement_methods)
        column = {
            "owner_frame_id": np.asarray(getattr(schedule, "column_frame_id")).copy(),
            "owner_source_index": np.asarray(
                getattr(schedule, "column_assignment_index")
            ).copy(),
            "assignment_index": np.asarray(
                getattr(schedule, "column_assignment_index")
            ).copy(),
            "source_u": np.asarray(getattr(schedule, "column_source_u")).copy(),
            "valid": np.asarray(getattr(schedule, "column_frame_id")) >= 0,
        }
        valid = np.asarray(pixel["valid"], dtype=bool)
        result = S13P0Result(
            image=np.ascontiguousarray(image),
            valid_mask=valid,
            pixel_provenance=pixel,
            column_provenance=column,
            remap_invocations=len(renders),
            duplicate_write_count=0,
        )
        result.image.setflags(write=False)
        result.valid_mask.setflags(write=False)
        return result

    def append_committed(
        self, identity: object, *, force_layout: bool = False
    ) -> S13OnlineShadowSnapshot:
        frame_id = int(getattr(identity, "frame_id"))
        color_path = Path(getattr(identity, "color_path")).resolve()
        color_path.relative_to(self.session_root)
        frame = S13RenderFrame(
            frame_id=frame_id,
            color_path=color_path,
            timestamp_us=int(getattr(identity, "timestamp_us")),
            width=int(self.calibration.width),
            height=int(self.calibration.height),
        )
        prepared = prepare_s13_motion_frame(
            frame, analysis_width_px=self.analysis_width_px, include_source_identity=True
        )
        if prepared.source_jpeg_sha256 != str(getattr(identity, "color_sha256")):
            raise ValueError("S1.3 committed JPEG identity changed before online M0")
        self._frames.append(frame)
        self._frame_by_id[frame_id] = frame
        self._motion.append(prepared)
        motion = self._motion.snapshot(deferred_step4=True)
        commit_ns = int(getattr(identity, "commit_monotonic_ns", 0))
        layout_due = (
            force_layout
            or len(self._frames) == 2
            or self._snapshot.layout is None
            or commit_ns <= 0
            or self._last_layout_commit_ns <= 0
            or commit_ns - self._last_layout_commit_ns >= self._layout_interval_ns
        )
        if not layout_due:
            previous = self._snapshot
            last_index = len(self._frames) - 1
            self._snapshot = S13OnlineShadowSnapshot(
                frontiers=S13OnlineFrontiers(
                    committed_frame_index=last_index,
                    prepared_frame_index=last_index,
                    motion_complete_frame_index=last_index,
                    layout_current_frame_index=(
                        previous.frontiers.layout_current_frame_index
                    ),
                    selected_source_count=previous.frontiers.selected_source_count,
                    sealed_source_count=previous.frontiers.sealed_source_count,
                    sealed_canvas_x_end=previous.frontiers.sealed_canvas_x_end,
                    mutable_canvas_x_begin=previous.frontiers.mutable_canvas_x_begin,
                    preview_frame_index=previous.frontiers.preview_frame_index,
                ),
                motion=motion,
                layout=previous.layout,
                selection=previous.selection,
                schedule=previous.schedule,
                semantic_assignments=previous.semantic_assignments,
                current_p0=previous.current_p0,
            )
            return self._snapshot
        self._last_layout_commit_ns = commit_ns
        layout = selection = schedule = None
        semantic: tuple[S13ScheduleSemanticAssignment, ...] = ()
        current_p0: S13P0Result | None = None
        sealed_count = 0
        if len(self._frames) >= 2:
            layout = build_s13_m3_layout(
                tuple(self._frames), motion.edges, _ignore_pose_trajectory()
            )
            self._canonical_directions.add(int(layout.canonical_scan_direction))
            if layout.progress.spatial:
                plan = plan_s13_m3_schedule(
                    layout.progress,
                    motion.edges,
                    self.calibration,
                    normal_target_advance_px=self.normal_target_advance_px,
                    risky_target_advance_px=self.risky_target_advance_px,
                    segment_break_pairs=layout.segment_break_pairs,
                    canonical_scan_direction=layout.canonical_scan_direction,
                )
                if len(plan.schedules) == 1:
                    selection, schedule = plan.selections[0], plan.schedules[0]
                    hypothesis_ids = selected_hypothesis_ids_for_spatial_sources(
                        layout, selection.frame_ids
                    )
                    semantic = semantic_assignments(
                        schedule, selection, hypothesis_ids
                    )
                    common = 0
                    for previous, current in zip(self._previous_semantic, semantic):
                        if previous != current:
                            break
                        common += 1
                    sealed_count = min(common, max(0, len(semantic) - 4))
                    current_p0 = self._render_current_p0(
                        schedule, selection, semantic, hypothesis_ids
                    )
                    self._checkpoint_chain.advance(
                        semantic, self._current_renders, sealed_count
                    )
                    self._previous_semantic = semantic
        last_index = len(self._frames) - 1
        canvas_width = 0 if schedule is None else int(getattr(schedule, "canvas_width"))
        self._snapshot = S13OnlineShadowSnapshot(
            frontiers=S13OnlineFrontiers(
                committed_frame_index=last_index,
                prepared_frame_index=last_index,
                motion_complete_frame_index=last_index,
                layout_current_frame_index=last_index,
                selected_source_count=len(semantic),
                sealed_source_count=sealed_count,
                sealed_canvas_x_end=(
                    0 if schedule is None or sealed_count == 0
                    else int(getattr(schedule, "assignments")[sealed_count - 1].right_x)
                ),
                mutable_canvas_x_begin=(
                    0 if schedule is None or sealed_count == 0
                    else int(getattr(schedule, "assignments")[sealed_count].left_x)
                ),
                preview_frame_index=last_index if current_p0 is not None else -1,
            ),
            motion=motion,
            layout=layout,
            selection=selection,
            schedule=schedule,
            semantic_assignments=semantic,
            current_p0=current_p0,
        )
        if schedule is not None and canvas_width <= 0:
            raise RuntimeError("S1.3 online shadow produced an empty schedule")
        return self._snapshot

    def snapshot(self) -> S13OnlineShadowSnapshot:
        return self._snapshot

    @staticmethod
    def _file_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def freeze(
        self,
        *,
        committed_frames: tuple[object, ...],
        production_config_sha256: str,
    ) -> S13FrozenP0Authority:
        """Close M3, rerender only the non-checkpoint suffix, and freeze P0."""

        if len(self._frames) != len(committed_frames) or len(self._frames) < 2:
            raise RuntimeError("S1.3 frozen P0 does not cover the committed ledger")
        closure_started = time.perf_counter()
        motion = self._motion.snapshot(deferred_step4=True)
        layout = build_s13_m3_layout(
            tuple(self._frames), motion.edges, _ignore_pose_trajectory()
        )
        if not layout.progress.spatial:
            raise RuntimeError("S1.3 frozen P0 final closure is not spatial")
        plan = plan_s13_m3_schedule(
            layout.progress,
            motion.edges,
            self.calibration,
            normal_target_advance_px=self.normal_target_advance_px,
            risky_target_advance_px=self.risky_target_advance_px,
            segment_break_pairs=layout.segment_break_pairs,
            canonical_scan_direction=layout.canonical_scan_direction,
        )
        if len(plan.schedules) != 1:
            raise RuntimeError("S1.3 frozen P0 final closure produced panel sets")
        selection, schedule = plan.selections[0], plan.schedules[0]
        hypothesis_ids = selected_hypothesis_ids_for_spatial_sources(
            layout, selection.frame_ids
        )
        final_semantic = semantic_assignments(schedule, selection, hypothesis_ids)
        maximum_reuse = max(0, len(final_semantic) - 4)
        if self._snapshot.semantic_assignments == final_semantic:
            self._checkpoint_chain.advance(
                final_semantic, self._current_renders, maximum_reuse
            )
        consistent = self._checkpoint_chain.longest_consistent_prefix(
            final_semantic, self._current_renders
        )
        reuse_count = min(consistent, maximum_reuse)
        reasons: list[str] = []
        if consistent < len(self._checkpoint_chain.checkpoints):
            reasons.append("sealed_prefix_checkpoint_mismatch")
        if len(self._canonical_directions) > 1:
            reasons.append("canonical_direction_changed")
            reuse_count = 0
        if layout.segment_break_pairs:
            reasons.append("final_segment_break")
            reuse_count = 0
        if self._snapshot.semantic_assignments != final_semantic:
            reasons.append("final_schedule_changed")
        closure_seconds = time.perf_counter() - closure_started

        tail_started = time.perf_counter()
        self._assignment_cache = {
            key: value
            for key, value in self._assignment_cache.items()
            if key[0] < reuse_count and key[1] == final_semantic[key[0]]
        }
        final_p0 = self._render_current_p0(
            schedule, selection, final_semantic, hypothesis_ids
        )
        tail_seconds = time.perf_counter() - tail_started
        self._snapshot = S13OnlineShadowSnapshot(
            frontiers=S13OnlineFrontiers(
                committed_frame_index=len(self._frames) - 1,
                prepared_frame_index=len(self._frames) - 1,
                motion_complete_frame_index=len(self._frames) - 1,
                layout_current_frame_index=len(self._frames) - 1,
                selected_source_count=len(final_semantic),
                sealed_source_count=reuse_count,
                sealed_canvas_x_end=(
                    0 if reuse_count == 0
                    else int(schedule.assignments[reuse_count - 1].right_x)
                ),
                mutable_canvas_x_begin=(
                    0 if reuse_count == 0
                    else int(schedule.assignments[reuse_count].left_x)
                ),
                preview_frame_index=len(self._frames) - 1,
            ),
            motion=motion,
            layout=layout,
            selection=selection,
            schedule=schedule,
            semantic_assignments=final_semantic,
            current_p0=final_p0,
        )
        p0_result = S13StageResult(
            run_id="frozen-online-p0",
            stage=S13Stage.P0,
            revision=0,
            parent_revision=None,
            image=final_p0.image,
            valid=final_p0.valid_mask,
            runtime_data={
                "schedule": schedule,
                "selection": selection,
                "motion": motion.edges,
                "layout": layout,
            },
        )
        continuation = S13P0Continuation(
            p0=p0_result,
            p0_render=final_p0,
            schedule=schedule,
            selection=selection,
            motion=motion.edges,
            layout=layout,
            selected_hypothesis_ids=hypothesis_ids,
            motion_profile={"source": "committed_online_accumulator"},
            motion_execution={
                "policy": "deferred_step4_unless_direction_fallback_required",
                "step4_computed": motion.step4_computed,
                "step4_selected": motion.step4_selected,
            },
        )
        return S13FrozenP0Authority(
            schema="gemini305-s013-frozen-p0-authority/v1",
            algorithm_id=S13_VISUAL_CONTINUITY_ALGORITHM_ID,
            implementation_id=S13_VISUAL_CONTINUITY_IMPLEMENTATION_ID,
            production_config_sha256=production_config_sha256,
            session_root=self.session_root,
            manifest_sha256=self._file_sha256(self.session_root / "manifest.json"),
            calibration_sha256=self._file_sha256(self.session_root / "calibration.json"),
            frames_csv_sha256=self._file_sha256(self.session_root / "frames.csv"),
            committed_frames=committed_frames,
            continuation=continuation,
            sealed_prefix_source_count=reuse_count,
            finalized_tail_source_count=len(final_semantic) - reuse_count,
            checkpoint_chain_sha256=self._checkpoint_chain.chain_sha256,
            p0_pixel_sha256=array_sha256(final_p0.image, schema="s013-stage-pixels/v1"),
            p0_prefix_reused_fraction=(
                float(reuse_count) / float(len(final_semantic))
                if final_semantic else 0.0
            ),
            final_metadata_closure_seconds=closure_seconds,
            tail_finalize_seconds=tail_seconds,
            rollback_checkpoint_used=bool(reasons),
            rollback_reasons=tuple(reasons),
        )


__all__ = [
    "S13OnlineFrontiers",
    "S13FrozenP0Authority",
    "S13OnlineP0Engine",
    "S13OnlineShadowSnapshot",
    "S13ScheduleSemanticAssignment",
    "calibration_from_live_document",
    "semantic_assignments",
]
