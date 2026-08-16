"""Compare every selected P0 resident remap with OpenCV on a real session."""

from __future__ import annotations

import argparse

import cv2
import numpy as np

from panorama_demo.calibrated_remap import undistortion_maps
from panorama_demo.video_s12_renderer import _roi_inverse_map
from panorama_demo.video_s13_cuda_runtime import S13CudaRuntime
from panorama_demo.video_s13_motion import measure_s13_motion
from panorama_demo.video_s13_progress import build_s13_m3_layout
from panorama_demo.video_s13_schedule import plan_s13_m3_schedule
from panorama_demo.video_s13_session import load_s13_session, read_s13_rgb
from panorama_demo.video_s13_trajectory import load_s13_trajectory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    args = parser.parse_args()
    session = load_s13_session(args.input)
    trajectory = load_s13_trajectory(
        session, trajectory_cache=None, reuse_online_trajectory=False,
        run_offline_orb=False, ignore_pose=True, config_path=None,
    )
    motion = measure_s13_motion(session.frames, analysis_width_px=424)
    layout = build_s13_m3_layout(session.frames, motion, trajectory)
    schedule = plan_s13_m3_schedule(
        layout.progress, motion, session.calibration,
        normal_target_advance_px=8, risky_target_advance_px=5,
        segment_break_pairs=layout.segment_break_pairs,
    ).schedules[0]
    images = {
        assignment.frame_id: read_s13_rgb(session.frame_by_id[assignment.frame_id])
        for assignment in schedule.assignments if not assignment.zero_width
    }
    runtime = S13CudaRuntime()
    try:
        runtime.preload_sources(images)
        inverse = undistortion_maps(session.calibration)
        for assignment in schedule.assignments:
            if assignment.zero_width:
                continue
            image = images[assignment.frame_id]
            map_u, map_v, _ = _roi_inverse_map(
                session.calibration, assignment.center_x, assignment.left_x,
                assignment.right_x, inverse,
            )
            actual = runtime.remap_resident_frame(assignment.frame_id, image, map_u, map_v)
            expected = cv2.remap(
                image, map_u, map_v, cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0,
            )
            differing = int(np.count_nonzero(np.any(actual != expected, axis=2)))
            if differing:
                y, x = np.argwhere(np.any(actual != expected, axis=2))[0]
                print(
                    assignment.frame_id, assignment.left_x, assignment.right_x, differing,
                    "example", int(y), int(x), float(map_u[y, x]), float(map_v[y, x]),
                    actual[y, x].tolist(), expected[y, x].tolist(),
                )
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
