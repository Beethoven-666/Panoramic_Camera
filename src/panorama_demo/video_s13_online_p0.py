"""Committed-ledger S013 M0--M3 shadow and online-P0 state types."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .session import CameraIntrinsics
from .video_s13_motion import (
    S13MotionAccumulator,
    S13MotionSnapshot,
    prepare_s13_motion_frame,
)
from .video_s13_progress import (
    S13M3Layout,
    build_s13_m3_layout,
    selected_hypothesis_ids_for_spatial_sources,
)
from .video_s13_schedule import plan_s13_m3_schedule
from .video_s13_session import S13RenderFrame
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
        self._motion = S13MotionAccumulator()
        self._snapshot = S13OnlineShadowSnapshot(
            frontiers=S13OnlineFrontiers(),
            motion=self._motion.snapshot(),
            layout=None,
            selection=None,
            schedule=None,
            semantic_assignments=(),
        )

    def append_committed(self, identity: object) -> S13OnlineShadowSnapshot:
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
        self._motion.append(prepared)
        motion = self._motion.snapshot(deferred_step4=True)
        layout = selection = schedule = None
        semantic: tuple[S13ScheduleSemanticAssignment, ...] = ()
        if len(self._frames) >= 2:
            layout = build_s13_m3_layout(
                tuple(self._frames), motion.edges, _ignore_pose_trajectory()
            )
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
        last_index = len(self._frames) - 1
        canvas_width = 0 if schedule is None else int(getattr(schedule, "canvas_width"))
        self._snapshot = S13OnlineShadowSnapshot(
            frontiers=S13OnlineFrontiers(
                committed_frame_index=last_index,
                prepared_frame_index=last_index,
                motion_complete_frame_index=last_index,
                layout_current_frame_index=last_index,
                selected_source_count=len(semantic),
                sealed_source_count=0,
                sealed_canvas_x_end=0,
                mutable_canvas_x_begin=0,
                preview_frame_index=-1,
            ),
            motion=motion,
            layout=layout,
            selection=selection,
            schedule=schedule,
            semantic_assignments=semantic,
        )
        if schedule is not None and canvas_width <= 0:
            raise RuntimeError("S1.3 online shadow produced an empty schedule")
        return self._snapshot

    def snapshot(self) -> S13OnlineShadowSnapshot:
        return self._snapshot


__all__ = [
    "S13OnlineFrontiers",
    "S13OnlineP0Engine",
    "S13OnlineShadowSnapshot",
    "S13ScheduleSemanticAssignment",
    "calibration_from_live_document",
    "semantic_assignments",
]
