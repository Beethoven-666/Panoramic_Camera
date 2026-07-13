from __future__ import annotations

import json
import math
from typing import Any

import cv2
import numpy as np
import pytest

import panorama_demo.rgbd_odometry as odometry
from panorama_demo.rgbd_odometry import (
    Open3DUnavailableError,
    PoseEdge,
    PoseGraphError,
    PoseGraphResult,
    PoseQualityThresholds,
    estimate_pair_rgbd_odometry,
    optimize_rgbd_pose_graph,
    validate_pose_trajectory,
)


def _intrinsics(width: int = 32, height: int = 24) -> dict[str, float | int]:
    return {
        "width": width,
        "height": height,
        "fx": 30.0,
        "fy": 30.0,
        "cx": (width - 1) * 0.5,
        "cy": (height - 1) * 0.5,
    }


def _frame(frame_id: int, *, depth_mm: float = 1000.0) -> dict[str, Any]:
    return {
        "frame_id": frame_id,
        # Black is valid image content; depth and geometry define validity.
        "color_rgb": np.zeros((24, 32, 3), dtype=np.uint8),
        "aligned_depth_mm": np.full((24, 32), depth_mm, dtype=np.float32),
    }


def _pose(
    x_mm: float = 0.0,
    y_mm: float = 0.0,
    z_mm: float = 0.0,
    *,
    rotation_deg: float = 0.0,
) -> np.ndarray:
    theta = math.radians(rotation_deg)
    cosine = math.cos(theta)
    sine = math.sin(theta)
    matrix = np.asarray(
        [
            [cosine, -sine, 0.0, x_mm],
            [sine, cosine, 0.0, y_mm],
            [0.0, 0.0, 1.0, z_mm],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return matrix


def _edge(
    reference: int,
    source: int,
    transform: np.ndarray,
    *,
    failure_reasons: tuple[str, ...] = (),
    converged: bool = True,
) -> PoseEdge:
    return PoseEdge(
        reference_node_id=reference,
        source_node_id=source,
        source_to_reference=transform,
        converged=converged,
        fitness=0.90,
        rmse_mm=5.0,
        information=np.eye(6, dtype=np.float64),
        reference_valid_depth_ratio=0.95,
        source_valid_depth_ratio=0.94,
        failure_reasons=failure_reasons,
        backend="fake_rgbd",
    )


class _PairBackend:
    name = "fake_rgbd"

    def __init__(self, measurement: dict[str, Any]) -> None:
        self.measurement = measurement
        self.reference_depth_mm: np.ndarray | None = None

    def estimate_pair(self, **kwargs: Any) -> dict[str, Any]:
        self.reference_depth_mm = kwargs["reference"].depth_mm.copy()
        assert kwargs["source"].depth_mm.dtype == np.float32
        return self.measurement


class _EchoPoseGraphBackend:
    name = "fake_pose_graph"

    def __init__(self, replacement: list[np.ndarray] | None = None) -> None:
        self.replacement = replacement
        self.initial: tuple[np.ndarray, ...] | None = None
        self.edges: tuple[PoseEdge, ...] | None = None

    def optimize_pose_graph(self, **kwargs: Any) -> list[np.ndarray]:
        self.initial = kwargs["initial_camera_to_world"]
        self.edges = kwargs["edges"]
        if self.replacement is not None:
            return self.replacement
        return [pose.copy() for pose in self.initial]


def _result_from_poses(poses: list[np.ndarray]) -> PoseGraphResult:
    node_ids = tuple(range(len(poses)))
    edges = []
    residuals = []
    for reference, source in zip(node_ids, node_ids[1:]):
        transform = np.linalg.inv(poses[reference]) @ poses[source]
        edges.append(_edge(reference, source, transform))
        residuals.append(
            {
                "reference_node_id": reference,
                "source_node_id": source,
                "translation_residual_mm": 0.0,
                "rotation_residual_deg": 0.0,
            }
        )
    return PoseGraphResult(
        node_ids=node_ids,
        camera_to_world=tuple(poses),
        edges=tuple(edges),
        optimized=True,
        connected=True,
        reference_node_id=0,
        backend="fake_pose_graph",
        edge_residuals=tuple(residuals),
    )


def test_pair_estimation_uses_source_to_reference_and_public_mm_units() -> None:
    source_to_reference = _pose(125.0, 2.0, -3.0, rotation_deg=1.0)
    backend = _PairBackend(
        {
            "converged": True,
            "source_to_reference": source_to_reference,
            "information": np.eye(6, dtype=np.float64),
            "fitness": 0.88,
            "rmse_mm": 7.5,
            "backend": "fake_rgbd",
        }
    )

    edge = estimate_pair_rgbd_odometry(
        _frame(40),
        _frame(41),
        _intrinsics(),
        backend=backend,
    )

    assert edge.reference_node_id == 40
    assert edge.source_node_id == 41
    np.testing.assert_allclose(edge.source_to_reference, source_to_reference)
    assert edge.rmse_mm == pytest.approx(7.5)
    assert edge.rmse_m == pytest.approx(0.0075)
    assert edge.reliable
    assert backend.reference_depth_mm is not None
    assert float(backend.reference_depth_mm[0, 0]) == pytest.approx(1000.0)
    payload = edge.as_dict()
    assert payload["translation_unit"] == "mm"
    assert payload["translation_mm"] == pytest.approx([125.0, 2.0, -3.0])


def test_pair_estimation_requires_real_fitness_and_rmse_metrics() -> None:
    backend = _PairBackend(
        {
            "converged": True,
            "source_to_reference": _pose(100.0),
            "information": np.eye(6, dtype=np.float64),
        }
    )

    edge = estimate_pair_rgbd_odometry(
        _frame(0), _frame(1), _intrinsics(), backend=backend
    )

    assert not edge.reliable
    assert any("fitness" in reason for reason in edge.failure_reasons)
    assert any("RMSE" in reason for reason in edge.failure_reasons)
    assert edge.as_dict()["fitness"] is None


def test_aligned_depth_device_units_are_converted_to_mm_before_backend(
    tmp_path,
) -> None:
    color_path = tmp_path / "color.png"
    depth_path = tmp_path / "aligned.png"
    assert cv2.imwrite(
        str(color_path), np.zeros((24, 32, 3), dtype=np.uint8)
    )
    assert cv2.imwrite(
        str(depth_path), np.full((24, 32), 2000, dtype=np.uint16)
    )
    backend = _PairBackend(
        {
            "converged": True,
            "source_to_reference": _pose(100.0),
            "information": np.eye(6),
            "fitness": 0.8,
            "rmse_mm": 5.0,
        }
    )
    reference = {
        "frame_id": 0,
        "color_path": color_path,
        "aligned_depth_path": depth_path,
        "depth_scale_mm_per_unit": 0.5,
    }

    estimate_pair_rgbd_odometry(
        reference,
        reference | {"frame_id": 1},
        _intrinsics(),
        backend=backend,
    )

    assert backend.reference_depth_mm is not None
    assert float(backend.reference_depth_mm[0, 0]) == pytest.approx(1000.0)


def test_pair_estimation_rejects_raw_depth_as_aligned_depth(tmp_path) -> None:
    frame = {
        "frame_id": 0,
        "color_path": tmp_path / "color.png",
        "depth_path": tmp_path / "raw.png",
        "depth_scale_mm_per_unit": 1.0,
    }
    with pytest.raises(ValueError, match="aligned_depth_path.*raw depth"):
        estimate_pair_rgbd_odometry(
            frame, frame | {"frame_id": 1}, _intrinsics(), backend=_PairBackend({})
        )


def test_open3d_is_loaded_only_when_default_backend_is_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_open3d(name: str) -> Any:
        assert name == "open3d"
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(odometry.importlib, "import_module", missing_open3d)

    with pytest.raises(Open3DUnavailableError, match="no 2-D.*fallback"):
        estimate_pair_rgbd_odometry(_frame(0), _frame(1), _intrinsics())


def test_formal_config_cannot_disable_calibration_depth_or_pose_graph() -> None:
    with pytest.raises(ValueError, match="cannot disable"):
        odometry.RGBDOdometryConfig(require_aligned_depth=False)
    with pytest.raises(ValueError, match="cannot disable"):
        odometry.RGBDOdometryConfig(require_calibration=False)
    with pytest.raises(ValueError, match="cannot be disabled"):
        odometry.PoseGraphConfig(enabled=False)


def test_open3d_019_odometry_option_names_are_applied_and_verified() -> None:
    class OfficialOption:
        depth_min = 0.0
        depth_max = 4.0
        depth_diff_max = 0.03
        iteration_number_per_pyramid_level = [20, 10, 5]

    option = OfficialOption()
    selected = odometry._configure_open3d_odometry_option(  # noqa: SLF001
        option, odometry.RGBDOdometryConfig()
    )

    assert selected == {
        "depth_min": "depth_min",
        "depth_max": "depth_max",
        "depth_diff_max": "depth_diff_max",
        "iteration_number_per_pyramid_level": (
            "iteration_number_per_pyramid_level"
        ),
    }
    assert option.depth_min == pytest.approx(0.15)
    assert option.depth_max == pytest.approx(10.0)
    assert option.depth_diff_max == pytest.approx(0.07)
    assert option.iteration_number_per_pyramid_level == [20, 10, 5]


def test_known_legacy_odometry_option_names_are_explicitly_supported() -> None:
    class LegacyOption:
        min_depth = 0.0
        max_depth = 4.0
        max_depth_diff = 0.03
        iteration_number_per_pyramid_level = [20, 10, 5]

    option = LegacyOption()
    selected = odometry._configure_open3d_odometry_option(  # noqa: SLF001
        option, odometry.RGBDOdometryConfig()
    )

    assert selected["depth_min"] == "min_depth"
    assert selected["depth_max"] == "max_depth"
    assert selected["depth_diff_max"] == "max_depth_diff"
    assert option.min_depth == pytest.approx(0.15)


def test_missing_or_ignored_open3d_odometry_option_is_fatal() -> None:
    class MissingOption:
        pass

    with pytest.raises(RuntimeError, match="does not expose.*depth_min"):
        odometry._configure_open3d_odometry_option(  # noqa: SLF001
            MissingOption(), odometry.RGBDOdometryConfig()
        )

    class IgnoredOption:
        @property
        def depth_min(self) -> float:
            return 0.0

        @depth_min.setter
        def depth_min(self, _value: float) -> None:
            return None

    with pytest.raises(RuntimeError, match="did not retain depth_min"):
        odometry._set_open3d_option(  # noqa: SLF001
            IgnoredOption(), "depth_min", 0.15
        )


def test_pose_graph_composes_source_to_reference_edges_as_camera_to_world() -> None:
    first_edge = _edge(10, 20, _pose(100.0, rotation_deg=2.0))
    second_edge = _edge(20, 30, _pose(90.0))
    backend = _EchoPoseGraphBackend()

    result = optimize_rgbd_pose_graph(
        [10, 20, 30], [first_edge, second_edge], backend=backend
    )

    expected_second = first_edge.source_to_reference
    expected_third = expected_second @ second_edge.source_to_reference
    np.testing.assert_allclose(result.camera_to_world[0], np.eye(4), atol=1e-10)
    np.testing.assert_allclose(result.camera_to_world[1], expected_second, atol=1e-10)
    np.testing.assert_allclose(result.camera_to_world[2], expected_third, atol=1e-10)
    assert backend.initial is not None
    assert all(row["translation_residual_mm"] < 1e-8 for row in result.edge_residuals)
    assert result.as_dict()["translation_unit"] == "mm"


def test_pose_graph_can_use_an_edge_stored_in_the_reverse_direction() -> None:
    # The edge maps node 0 into node 1; propagation must invert it to recover
    # node 1 camera_to_world in the first-node world frame.
    reverse_edge = _edge(1, 0, _pose(-100.0))

    result = optimize_rgbd_pose_graph(
        [0, 1], [reverse_edge], backend=_EchoPoseGraphBackend()
    )

    assert result.camera_to_world[1][0, 3] == pytest.approx(100.0)


def test_pose_graph_rejects_a_disconnected_rgbd_graph() -> None:
    with pytest.raises(PoseGraphError, match="disconnected"):
        optimize_rgbd_pose_graph(
            [0, 1, 2, 3],
            [_edge(0, 1, _pose(100.0)), _edge(2, 3, _pose(100.0))],
            backend=_EchoPoseGraphBackend(),
        )


def test_pose_graph_rejects_missing_required_adjacent_edge_even_if_connected() -> None:
    with pytest.raises(PoseGraphError, match="Required adjacent RGB-D edge 1<->2"):
        optimize_rgbd_pose_graph(
            [0, 1, 2],
            [_edge(0, 1, _pose(100.0)), _edge(0, 2, _pose(200.0))],
            backend=_EchoPoseGraphBackend(),
        )


def test_diagnostic_edge_bypass_keeps_structural_se3_and_connectivity_gates() -> None:
    low_quality = _edge(
        0,
        1,
        _pose(100.0),
        failure_reasons=("RGB-D odometry fitness is below the safety threshold",),
    )
    with pytest.raises(PoseGraphError, match="no reliable measurement"):
        optimize_rgbd_pose_graph(
            [0, 1], [low_quality], backend=_EchoPoseGraphBackend()
        )

    result = optimize_rgbd_pose_graph(
        [0, 1],
        [low_quality],
        backend=_EchoPoseGraphBackend(),
        enforce_edge_quality=False,
    )
    assert result.connected

    non_finite = _edge(0, 1, _pose(100.0))
    non_finite.source_to_reference[0, 3] = np.nan
    with pytest.raises(PoseGraphError, match="disconnected"):
        optimize_rgbd_pose_graph(
            [0, 1],
            [non_finite],
            backend=_EchoPoseGraphBackend(),
            enforce_edge_quality=False,
        )


def test_pose_graph_rejects_non_finite_optimized_pose() -> None:
    invalid_pose = _pose(100.0)
    invalid_pose[0, 3] = np.nan
    backend = _EchoPoseGraphBackend([np.eye(4), invalid_pose])

    with pytest.raises(PoseGraphError, match="non-finite or non-rigid"):
        optimize_rgbd_pose_graph(
            [0, 1], [_edge(0, 1, _pose(100.0))], backend=backend
        )


@pytest.mark.parametrize("diagonal", [(2.0, 1.0, 1.0), (-1.0, 1.0, 1.0)])
def test_pose_edge_records_non_orthogonal_or_reflected_rotation_as_invalid(
    diagonal: tuple[float, float, float],
) -> None:
    transform = _pose(100.0)
    transform[:3, :3] = np.diag(diagonal)
    edge = _edge(0, 1, transform)

    assert not edge.transform_is_valid
    assert not edge.structurally_valid
    assert edge.as_dict()["finite_se3"] is False


@pytest.mark.parametrize(
    "information",
    [
        np.diag([-1.0, 2.0, 2.0, 2.0, 2.0, 2.0]),
        np.diag([0.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
    ],
)
def test_pose_edge_rejects_indefinite_or_singular_information(
    information: np.ndarray,
) -> None:
    edge = PoseEdge(
        reference_node_id=0,
        source_node_id=1,
        source_to_reference=_pose(100.0),
        converged=True,
        fitness=0.9,
        rmse_mm=5.0,
        information=information,
        reference_valid_depth_ratio=0.9,
        source_valid_depth_ratio=0.9,
    )

    assert not edge.information_is_valid
    assert not edge.structurally_valid


def test_valid_horizontal_pose_trajectory_passes_and_is_json_serializable() -> None:
    result = optimize_rgbd_pose_graph(
        [0, 1, 2],
        [_edge(0, 1, _pose(100.0)), _edge(1, 2, _pose(100.0))],
        backend=_EchoPoseGraphBackend(),
    )

    report = validate_pose_trajectory(result)

    assert report.quality_pass
    assert report.failure_reasons == ()
    assert report.metrics["scan_direction"] == 1
    assert report.metrics["scan_span_mm"] == pytest.approx(200.0)
    json.dumps(report.as_dict(), allow_nan=False)
    json.dumps(result.as_dict(), allow_nan=False)


def test_pose_trajectory_rejects_direction_reversal() -> None:
    result = _result_from_poses(
        [_pose(0.0), _pose(100.0), _pose(94.0), _pose(200.0)]
    )

    report = validate_pose_trajectory(result)

    assert not report.quality_pass
    assert any("unidirectional" in reason for reason in report.failure_reasons)


@pytest.mark.parametrize(
    ("poses", "expected_reason"),
    [
        ([_pose(), _pose(100.0, 90.0)], "vertical motion"),
        ([_pose(), _pose(100.0, 0.0, 130.0)], "forward motion"),
        ([_pose(), _pose(100.0, rotation_deg=7.0)], "step rotation"),
    ],
)
def test_pose_trajectory_rejects_cross_axis_drift_and_rotation(
    poses: list[np.ndarray], expected_reason: str
) -> None:
    report = validate_pose_trajectory(_result_from_poses(poses))

    assert not report.quality_pass
    assert any(expected_reason in reason for reason in report.failure_reasons)


def test_pose_trajectory_rejects_pose_graph_residual() -> None:
    result = _result_from_poses([_pose(), _pose(100.0)])
    result = PoseGraphResult(
        node_ids=result.node_ids,
        camera_to_world=result.camera_to_world,
        edges=result.edges,
        optimized=True,
        connected=True,
        reference_node_id=0,
        edge_residuals=(
            {
                "reference_node_id": 0,
                "source_node_id": 1,
                "translation_residual_mm": 31.0,
                "rotation_residual_deg": 0.0,
            },
        ),
    )

    report = validate_pose_trajectory(result)

    assert not report.quality_pass
    assert any("translation residual" in reason for reason in report.failure_reasons)


def test_pose_trajectory_rejects_non_finite_se3_and_three_missing_edges() -> None:
    invalid = _pose(300.0)
    invalid[2, 2] = np.nan
    result = PoseGraphResult(
        node_ids=(0, 1, 2, 3),
        camera_to_world=(_pose(), _pose(100.0), _pose(200.0), invalid),
        edges=(
            _edge(
                0,
                3,
                _pose(300.0),
                failure_reasons=("loop edge is not an adjacent measurement",),
            ),
        ),
        optimized=True,
        connected=True,
        reference_node_id=0,
        edge_residuals=(
            {
                "reference_node_id": 0,
                "source_node_id": 3,
                "translation_residual_mm": 0.0,
                "rotation_residual_deg": 0.0,
            },
        ),
    )

    report = validate_pose_trajectory(
        result,
        thresholds=PoseQualityThresholds(require_all_adjacent_edges=False),
    )

    assert not report.quality_pass
    assert any("Three or more" in reason for reason in report.failure_reasons)
    assert any("non-finite" in reason for reason in report.failure_reasons)
