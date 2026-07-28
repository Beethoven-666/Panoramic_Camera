"""Run the in-memory FastSAM ONNX + RGB-D DIS tracker on a frame interval."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import time

import cv2
import numpy as np

from panorama_demo.fastsam_dis_tracking import (
    FastSAMDISConfig,
    FastSAMDISFrameInput,
    track_fastsam_dis_frames,
)
from panorama_demo.fastsam_onnx import FastSAMOnnxRunner, summarize_onnxruntime_profile
from panorama_demo.inspection_multiview import (
    InspectionMultiviewConfig,
    _read_rgbd,
    _undistortion_maps,
)
from panorama_demo.session import load_rgbd_session


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("session", type=Path)
    parser.add_argument("formal_output", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--first-frame", type=int, default=8)
    parser.add_argument("--last-frame", type=int, default=36)
    args = parser.parse_args()
    started = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=True)
    report = json.loads((args.formal_output / "report.json").read_text(encoding="utf-8"))
    transforms = json.loads(
        (args.formal_output / "transforms.json").read_text(encoding="utf-8")
    )
    inspection = InspectionMultiviewConfig.from_mapping(report["render"]["config"])
    session = load_rgbd_session(args.session)
    maps = _undistortion_maps(session.calibration)
    pose_by_id = {
        int(item["node_id"]): np.asarray(item["camera_to_world"], dtype=np.float64)
        for item in transforms["nodes"]
    }
    selected_frames = tuple(
        int(item["frame_id"])
        for item in report["render"]["selected_panel_sources"]
        if args.first_frame <= int(item["frame_id"]) <= args.last_frame
    )
    source_frames = [
        frame
        for frame in sorted(session.frames, key=lambda value: int(value.frame_id))
        if args.first_frame <= int(frame.frame_id) <= args.last_frame
    ]
    if any(int(frame.frame_id) not in pose_by_id for frame in source_frames):
        raise RuntimeError("requested diagnostic interval has a missing real pose")
    runner = FastSAMOnnxRunner(args.model, enable_profiling=True)

    def frame_inputs():
        for index, frame in enumerate(source_frames):
            frame_id = int(frame.frame_id)
            raw = cv2.imread(str(frame.color_path), cv2.IMREAD_COLOR)
            if raw is None:
                raise RuntimeError(f"could not decode FastSAM RGB frame {frame_id}")
            proposals = runner.predict(raw)
            image, depth, geometric_valid = _read_rgbd(frame, session.calibration, maps)
            print(
                f"F{frame_id}: proposals={len(proposals)}",
                flush=True,
            )
            yield FastSAMDISFrameInput(
                frame_id=frame_id,
                image_bgr=image,
                depth_mm=depth,
                camera_to_world=pose_by_id[frame_id],
                proposals=proposals,
                geometric_valid=geometric_valid,
            )

    result = track_fastsam_dis_frames(
        frame_inputs(),
        intrinsics=session.calibration,
        reference_depth_mm=float(report["render"]["layout"]["reference_depth_mm"]),
        stable_frame_ids=selected_frames,
        config=FastSAMDISConfig(
            minimum_depth_mm=float(inspection.minimum_depth_mm),
            maximum_depth_mm=float(inspection.maximum_depth_mm),
        ),
    )
    profile_source = runner.end_profiling()
    profile = summarize_onnxruntime_profile(profile_source) if profile_source else {}
    if profile_source and profile_source.is_file():
        shutil.copy2(profile_source, args.output / "onnxruntime_profile.json")
    track_by_id = {track.track_id: track for track in result.tracks}
    expected = {
        0: {
            "minimum_observations": 20,
            "required_frames": (8, 20, 26, 32),
            "required_stable_frames": (8, 20, 26),
        },
        49: {
            "minimum_observations": 20,
            "required_frames": (14, 20, 26, 36),
            "required_stable_frames": (20, 26, 36),
        },
    }
    def satisfies(track, requirements):
        return bool(
            track is not None
            and track.observation_count >= requirements["minimum_observations"]
            and set(requirements["required_frames"]).issubset(track.frame_ids)
            and set(requirements["required_stable_frames"]).issubset(track.stable_frame_ids)
        )

    target_audits = {}
    for legacy_track_id, requirements in expected.items():
        exact_track = track_by_id.get(legacy_track_id)
        equivalent_tracks = [
            track for track in result.tracks if satisfies(track, requirements)
        ]
        observed = (
            exact_track
            if satisfies(exact_track, requirements)
            else equivalent_tracks[0]
            if len(equivalent_tracks) == 1
            else None
        )
        target_audits[f"T{legacy_track_id}"] = {
            "pass": observed is not None,
            "identity_continuity_pass": observed is not None,
            "legacy_ordinal_track_id": legacy_track_id,
            "observed_ordinal_track_id": (
                observed.track_id if observed is not None else None
            ),
            "exact_ordinal_track_id_preserved": bool(
                observed is not None and observed.track_id == legacy_track_id
            ),
            "ordinal_id_note": (
                "Track identity is established by fixed-gate frame continuity; "
                "ordinal IDs may shift when the proposal set gains or loses an "
                "earlier candidate."
            ),
            "equivalent_track_candidate_ids": [
                track.track_id for track in equivalent_tracks
            ],
            "requirements": requirements,
            "observed": asdict(observed) if observed is not None else None,
        }
    cuda_profile = profile.get("providers", {}).get("CUDAExecutionProvider", {})
    conv_providers = profile.get("operator_providers", {}).get("Conv", [])
    cuda_verified = bool(
        cuda_profile.get("node_events", 0)
        and "CUDAExecutionProvider" in conv_providers
        and not runner.diagnostic_cpu_fallback_used
    )
    passed = bool(cuda_verified and all(value["pass"] for value in target_audits.values()))
    output = {
        "schema": "gemini305-fastsam-onnx-dis-tracking-audit/v1",
        "verdict": "pass" if passed else "fail",
        "formal_output_modified": False,
        "labels_read": False,
        "historical_audit_sidecar_read": False,
        "frame_interval": [args.first_frame, args.last_frame],
        "stable_frame_ids": selected_frames,
        "flow_role": result.flow_role,
        "flow_used_to_warp_rgb_or_position": result.flow_used_to_warp_rgb_or_position,
        "frame_candidate_counts": {
            str(frame.frame_id): len(frame.candidates) for frame in result.frames
        },
        "matched_edge_count": sum(
            int(row["one_to_one_match_count"]) for row in result.pair_audits
        ),
        "track_count": len(result.tracks),
        "stable_track_count": len(result.stable_tracks),
        "target_track_audits": target_audits,
        "stable_tracks": [asdict(track) for track in result.stable_tracks],
        "pair_audits": result.pair_audits,
        "execution": {
            "mode": runner.execution_mode,
            "active_providers": runner.active_providers,
            "diagnostic_cpu_fallback_used": runner.diagnostic_cpu_fallback_used,
            "cuda_execution_verified": cuda_verified,
            "profile": profile,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    output_path = args.output / "tracking_audit.json"
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(output_path)
    print(
        json.dumps(
            {
                "verdict": output["verdict"],
                "track_count": output["track_count"],
                "stable_track_count": output["stable_track_count"],
                "T0": target_audits["T0"]["pass"],
                "T49": target_audits["T49"]["pass"],
                "cuda_execution_verified": cuda_verified,
                "elapsed_seconds": output["elapsed_seconds"],
            },
            indent=2,
        )
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
