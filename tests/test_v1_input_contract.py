from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from panorama_demo.session import (
    CameraIntrinsics,
    RGBDFrame,
    RGBDSession,
)
from panorama_demo.v1_input_contract import (
    audit_v1_input_sidecars,
    load_v1_camera_yaml,
    load_v1_transforms_json,
)


DISTORTION = (-1.0, 0.5, 0.001, -0.002, -0.1, -0.9, 0.45, -0.08)


def _session(root: Path, frame_ids: tuple[int, ...] = (10, 11)) -> RGBDSession:
    intrinsics = CameraIntrinsics(
        width=1280,
        height=800,
        fx=613.25,
        fy=613.125,
        cx=636.75,
        cy=394.0,
        distortion=DISTORTION,
    )
    frames = tuple(
        RGBDFrame(
            frame_id=frame_id,
            color_path=root / "color" / f"{frame_id}.jpg",
            aligned_depth_path=root / "depth_aligned" / f"{frame_id}.png",
            depth_scale_mm_per_unit=0.1,
            timestamp_us=frame_id * 1000,
            color_exposure_raw=8,
        )
        for frame_id in frame_ids
    )
    return RGBDSession(
        root=root,
        calibration=intrinsics,
        frames=frames,
        manifest={"schema": "panorama-demo-session/v1", "clean_shutdown": True},
    )


def _write_camera(path: Path, *, fx: float = 613.25, scale: float = 0.1) -> None:
    path.write_text(
        "\n".join(
            [
                "schema: g305-camera/v1",
                "image_width: 1280",
                "image_height: 800",
                "camera_coordinates: opencv_x_right_y_down_z_forward",
                "camera_matrix:",
                "  rows: 3",
                "  cols: 3",
                (
                    "  data: "
                    f"[{fx}, 0, 636.75, 0, 613.125, 394.0, 0, 0, 1]"
                ),
                "distortion_model: opencv_rational",
                "distortion_coefficients:",
                "  rows: 1",
                "  cols: 8",
                "  data: [-1.0, 0.5, 0.001, -0.002, -0.1, -0.9, 0.45, -0.08]",
                f"depth_scale_mm_per_unit: {scale}",
                "depth_alignment: color",
            ]
        ),
        encoding="utf-8",
    )


def _trajectory_payload(frame_ids: tuple[int, ...] = (10, 11)) -> dict:
    nodes = []
    for index, frame_id in enumerate(frame_ids):
        pose = np.eye(4, dtype=np.float64)
        pose[0, 3] = index * 20.0
        nodes.append(
            {"node_id": frame_id, "camera_to_world": pose.tolist()}
        )
    return {
        "schema": "rgbd-pose-graph/v1",
        "translation_unit": "mm",
        "pose_convention": "camera_to_world",
        "camera_coordinates": "opencv_x_right_y_down_z_forward",
        "nodes": nodes,
        "edge_residuals": [
            {
                "source_node_id": frame_ids[0],
                "target_node_id": frame_ids[-1],
                "translation_residual_mm": 0.25,
                "rotation_residual_deg": 0.01,
            }
        ],
        "global_trajectory": {
            "backend": "orbslam3_rgbd_wsl",
            "pose_convention": "camera_to_world",
            "translation_unit": "mm",
            "tracked_frame_ids": list(frame_ids),
            "tracked_fraction": 1.0,
        },
        "tracked_frame_ids": list(frame_ids),
    }


def test_camera_yaml_matches_formal_calibration_and_frames(tmp_path: Path) -> None:
    session = _session(tmp_path)
    path = tmp_path / "camera.yaml"
    _write_camera(path)

    audit = load_v1_camera_yaml(path, session=session)

    assert audit.intrinsics == session.calibration
    assert audit.depth_scale_mm_per_unit == 0.1
    assert "authoritative" in audit.authority


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda text: text.replace("image_width: 1280", "image_width: 640"), "1280x800"),
        (lambda text: text.replace("613.25, 0", "614.25, 0"), "calibration.json"),
        (
            lambda text: text.replace(
                "depth_scale_mm_per_unit: 0.1",
                "depth_scale_mm_per_unit: 1.0",
            ),
            "frames.csv",
        ),
    ],
)
def test_camera_yaml_rejects_contract_mismatch(
    tmp_path: Path, mutation, match: str
) -> None:
    session = _session(tmp_path)
    path = tmp_path / "camera.yaml"
    _write_camera(path)
    path.write_text(mutation(path.read_text(encoding="utf-8")), encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        load_v1_camera_yaml(path, session=session)


def test_transforms_loader_returns_ordered_rigid_mm_audit(tmp_path: Path) -> None:
    path = tmp_path / "transforms.json"
    path.write_text(json.dumps(_trajectory_payload()), encoding="utf-8")

    audit = load_v1_transforms_json(path, required_frame_ids=[11, 10])

    assert audit.node_ids == (10, 11)
    assert audit.tracked_frame_ids == (10, 11)
    assert audit.backend == "orbslam3_rgbd_wsl"
    assert audit.pose_for_audit(11)[0, 3] == 20.0
    ordered = audit.poses_for_audit([11, 10])
    assert [pose[0, 3] for pose in ordered] == [20.0, 0.0]
    assert "not_an_orbslam3_fallback" in audit.authority


def test_transforms_loader_accepts_existing_explicit_convention_sentences(
    tmp_path: Path,
) -> None:
    payload = _trajectory_payload()
    payload["pose_convention"] = (
        "camera_to_world maps camera coordinates into the first pose-node camera frame"
    )
    payload["camera_coordinates"] = (
        "OpenCV/Open3D color camera coordinates: +x right, +y down, +z forward"
    )
    path = tmp_path / "transforms.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    audit = load_v1_transforms_json(path, required_frame_ids=[10, 11])

    assert audit.node_ids == (10, 11)


def test_transforms_loader_requires_complete_used_frame_coverage(
    tmp_path: Path,
) -> None:
    path = tmp_path / "transforms.json"
    path.write_text(json.dumps(_trajectory_payload((10,))), encoding="utf-8")

    with pytest.raises(ValueError, match="completely cover.*11"):
        load_v1_transforms_json(path, required_frame_ids=[10, 11])


@pytest.mark.parametrize("defect", ["unit", "pose", "untracked", "nan"])
def test_transforms_loader_rejects_untrusted_sidecar(
    tmp_path: Path, defect: str
) -> None:
    payload = _trajectory_payload()
    if defect == "unit":
        payload["translation_unit"] = "m"
    elif defect == "pose":
        payload["nodes"][1]["camera_to_world"][0][0] = 2.0
    elif defect == "untracked":
        payload["global_trajectory"]["tracked_frame_ids"] = [10]
        payload["tracked_frame_ids"] = [10]
    else:
        payload["edge_residuals"][0]["translation_residual_mm"] = float("nan")
    path = tmp_path / "transforms.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError):
        load_v1_transforms_json(path, required_frame_ids=[10, 11])


def test_sidecars_are_optional_and_never_change_formal_authority(
    tmp_path: Path,
) -> None:
    session = _session(tmp_path)

    absent = audit_v1_input_sidecars(session)

    assert absent.camera is None
    assert absent.trajectory is None
    assert absent.required_frame_ids == (10, 11)
    assert absent.as_dict()["sidecars_change_formal_authority"] is False

    _write_camera(tmp_path / "camera.yaml")
    (tmp_path / "transforms.json").write_text(
        json.dumps(_trajectory_payload()), encoding="utf-8"
    )
    present = audit_v1_input_sidecars(session, used_frame_ids=[11])

    assert present.camera is not None
    assert present.trajectory is not None
    assert present.trajectory.pose_for_audit(11)[0, 3] == 20.0


def test_transforms_json_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    path = tmp_path / "transforms.json"
    path.write_text(
        '{"translation_unit":"mm","translation_unit":"m"}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate key"):
        load_v1_transforms_json(path)
