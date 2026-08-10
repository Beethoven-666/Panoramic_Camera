"""Standalone S1.2 session assembly without any S0/S1 baseline input."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2

from .video_s12_config import S12Config
from .video_s12_pose import S12PoseQualification, S12Trajectory, load_s12_trajectory, qualify_s12_poses
from .video_session import VideoSession, load_video_session


@dataclass(frozen=True)
class S12Session:
    video: VideoSession
    trajectory: S12Trajectory
    pose_qualifications: tuple[S12PoseQualification, ...]

    @property
    def root(self) -> Path:
        return self.video.rgbd.root

    @property
    def pose_qualified_frame_ids(self) -> tuple[int, ...]:
        return tuple(row.frame_id for row in self.pose_qualifications if row.qualified)


def _validate_rgb_files(video: VideoSession) -> None:
    calibration = video.rgbd.calibration
    for frame in video.rgbd.frames:
        if not frame.color_path.is_file():
            raise ValueError(f"S1.2 RGB frame is missing: {frame.color_path}")
        image = cv2.imread(str(frame.color_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"S1.2 RGB frame cannot be decoded: {frame.color_path}")
        if image.shape[:2] != (calibration.height, calibration.width):
            raise ValueError(
                f"S1.2 RGB frame {frame.frame_id} dimensions do not match calibration"
            )


def load_s12_session(
    input_path: str | Path,
    trajectory_path: str | Path,
    config: S12Config,
    *,
    validate_rgb_files: bool = True,
    validation_workers: int = 1,
) -> S12Session:
    """Read only current session artifacts and an explicit direct-pose trajectory.

    Depth image pixels are intentionally not decoded here.  They remain paths
    on ``RGBDFrame`` and may only be read later by the isolated Open3D audit.
    """

    video = load_video_session(
        input_path,
        validate_frame_files=False,
        validation_workers=validation_workers,
    )
    if validate_rgb_files:
        _validate_rgb_files(video)
    trajectory = load_s12_trajectory(trajectory_path)
    if trajectory.schema != "gemini305-orbslam3-trajectory/v2":
        raise ValueError("S1.2 Stage A requires gemini305-orbslam3-trajectory/v2")
    session_ids = {frame.frame_id for frame in video.rgbd.frames}
    unknown = sorted(record.frame_id for record in trajectory.records if record.frame_id not in session_ids)
    if unknown:
        raise ValueError(f"S1.2 trajectory contains frames outside the session: {unknown[:8]}")
    qualifications = qualify_s12_poses(video.rgbd.frames, trajectory, config.trajectory)
    return S12Session(
        video=video,
        trajectory=trajectory,
        pose_qualifications=qualifications,
    )


__all__ = ["S12Session", "load_s12_session"]
