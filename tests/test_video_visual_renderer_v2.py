from __future__ import annotations

import importlib
from types import SimpleNamespace

import numpy as np
import pytest

from panorama_demo.video_algorithm_contract import PairPlan, VideoAlgorithmContractError
from panorama_demo.video_gpu_runtime import VideoGpuRuntimeConfig
from panorama_demo.video_raft_runtime import RAFTSmallRuntimeError
from panorama_demo.video_visual_renderer_v2 import (
    CudaC1ConstrainedOwnerConfig,
    CudaRealSource,
    CudaSourceStrip,
    TorchCudaC1ConstrainedOwnerAlgorithm,
    TorchCudaC2DisResidualMeshAlgorithm,
    TorchCudaC3RAFTResidualMeshAlgorithm,
    TorchCudaC4RAFTDepthLayeredMeshAlgorithm,
    TorchCudaC5ObjectLockAlgorithm,
    TorchCudaC6SafeMultiBandAlgorithm,
    TorchCudaC7PhotometricGraphAlgorithm,
    TorchCudaC8MultilabelWindowAlgorithm,
    TorchCudaC12JointOwnerFinalGridAlgorithm,
    TorchCudaStripOwnerAlgorithm,
    _finalize_component_execution,
    _median_exposure_anchor_index,
    build_cuda_strips_from_pushbroom_layout,
)


def _torch():
    return importlib.import_module("torch")


def _source(frame_id: int, red: int, blue: int) -> CudaRealSource:
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    rgb[..., 0] = red
    rgb[..., 2] = blue
    return CudaRealSource(frame_id, frame_id * 10, rgb, np.full((8, 8), 600, dtype=np.uint16), np.eye(4))


def _source_with_red_ramp(frame_id: int) -> CudaRealSource:
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    rgb[..., 0] = np.arange(8, dtype=np.uint8)[None, :] * 20
    return CudaRealSource(frame_id, frame_id * 10, rgb, np.full((8, 8), 600, dtype=np.uint16), np.eye(4))


def test_component_execution_requires_final_output_pixels_and_preserves_c4_parent_lineage():
    audit: dict[str, object] = {
        "c1_constrained_owner": {"owner_pixels_changed_from_initial_hard_strip": 12},
        "c4_raft_rgbd_layered_mesh": {
            "pair_audits": [
                {"actual_output_mesh_pixel_count": 7, "maximum_mesh_displacement_px": 2.5},
                {"actual_output_mesh_pixel_count": 0, "fallback_to_c1_hard_owner": True},
            ],
        },
    }

    _finalize_component_execution(
        audit,
        required_components=("c1_constrained_owner", "c3_raft_mesh", "c4_depth_layered_mesh"),
    )

    records = audit["component_execution"]
    assert isinstance(records, dict)
    assert records["c3_raft_mesh"]["applied_output_pixel_count"] == 7
    assert records["c3_raft_mesh"]["applied_to_output"] is True
    assert records["c4_depth_layered_mesh"]["fallback_pair_count"] == 1
    assert audit["candidate_run_state"] == "completed"


def test_component_execution_marks_zero_output_required_component_invalid():
    audit: dict[str, object] = {
        "c1_constrained_owner": {"owner_pixels_changed_from_initial_hard_strip": 12},
        "c3_raft_mesh": {"pair_audits": [{"actual_output_mesh_pixel_count": 0, "fallback_to_c1_hard_owner": True}]},
    }

    _finalize_component_execution(audit, required_components=("c1_constrained_owner", "c3_raft_mesh"))

    assert audit["candidate_run_state"] == "invalid_component_execution"
    assert audit["selection_eligible"] is False
    assert audit["component_execution_failure_components"] == ["c3_raft_mesh"]


def test_component_execution_requires_c10_depth_conditioned_grid_to_reach_output():
    audit: dict[str, object] = {
        "c1_constrained_owner": {"owner_pixels_changed_from_initial_hard_strip": 12},
        "c4_raft_rgbd_layered_mesh": {"pair_audits": [{"actual_output_mesh_pixel_count": 5}]},
        "c10_depth_conditioned_multi_perspective_layout": {
            "pair_audits": [{"actual_output_layout_pixel_count": 5}],
        },
    }

    _finalize_component_execution(
        audit,
        required_components=(
            "c1_constrained_owner", "c3_raft_mesh", "c4_depth_layered_mesh", "c10_depth_conditioned_layout",
        ),
    )

    records = audit["component_execution"]
    assert isinstance(records, dict)
    assert records["c10_depth_conditioned_layout"]["applied_output_pixel_count"] == 5
    assert audit["candidate_run_state"] == "completed"


def test_component_execution_requires_c12_joint_owner_grid_to_recompose_actual_output():
    audit: dict[str, object] = {
        "c1_constrained_owner": {"owner_pixels_changed_from_initial_hard_strip": 12},
        "c4_raft_rgbd_layered_mesh": {"pair_audits": [{"actual_output_mesh_pixel_count": 5}]},
        "c10_depth_conditioned_multi_perspective_layout": {"pair_audits": [{"actual_output_layout_pixel_count": 5}]},
        "c12_joint_owner_final_grid": {
            "window_audits": [{"actual_output_joint_owner_grid_pixel_count": 9}],
        },
    }

    _finalize_component_execution(
        audit,
        required_components=(
            "c1_constrained_owner", "c3_raft_mesh", "c4_depth_layered_mesh",
            "c10_depth_conditioned_layout", "c12_joint_owner_final_grid",
        ),
    )

    records = audit["component_execution"]
    assert isinstance(records, dict)
    assert records["c12_joint_owner_final_grid"]["applied_output_pixel_count"] == 9
    assert records["c12_joint_owner_final_grid"]["applied_to_output"] is True


def test_c12_component_execution_keeps_a_real_window_rejection_observable():
    audit: dict[str, object] = {
        "c12_joint_owner_final_grid": {
            "window_audits": [
                {"c12_exception": "no feasible genuine owner", "fallback_to_c10": True,
                 "actual_output_joint_owner_grid_pixel_count": 0},
            ],
        },
    }
    _finalize_component_execution(audit, required_components=("c12_joint_owner_final_grid",))

    record = audit["component_execution"]["c12_joint_owner_final_grid"]
    assert record["attempted_pair_count"] == 1
    assert record["fallback_pair_count"] == 1
    assert record["rejection_reasons"] == {"c12_exception": 1}


def test_c7_median_exposure_anchor_uses_real_metadata_and_deterministic_fixture_fallback():
    # Real session metadata selects the actual median exposure source.
    assert _median_exposure_anchor_index((4, 6, 10, 20, 30)) == 2
    # Low-level source fixtures intentionally have no SessionFrame metadata;
    # their fallback must remain deterministic without weakening session IO.
    assert _median_exposure_anchor_index((None, None, None, None)) == 2
    with pytest.raises(VideoAlgorithmContractError, match="all absent or all positive"):
        _median_exposure_anchor_index((6, None, 8))


def test_c12_partitions_a_full_real_sequence_into_only_five_to_seven_source_solver_windows():
    assert TorchCudaC12JointOwnerFinalGridAlgorithm._source_window_indexes(5) == (  # noqa: SLF001
        (0, 1, 2, 3, 4),
    )
    assert TorchCudaC12JointOwnerFinalGridAlgorithm._source_window_indexes(12) == (  # noqa: SLF001
        (0, 1, 2, 3, 4, 5, 6), (7, 8, 9, 10, 11),
    )
    with pytest.raises(VideoAlgorithmContractError, match="at least five"):
        TorchCudaC12JointOwnerFinalGridAlgorithm._source_window_indexes(4)  # noqa: SLF001


@pytest.mark.skipif(not _torch().cuda.is_available(), reason="requires actual CUDA")
def test_v2_cuda_strip_algorithm_renders_real_sources_once_with_owner_provenance():
    algorithm = TorchCudaStripOwnerAlgorithm(
        sources=(_source(10, 20, 0), _source(20, 0, 180)),
        strips=(CudaSourceStrip(10, 0, 0, 4), CudaSourceStrip(20, 4, 4, 4)),
        output_height=8,
        output_width=8,
        calibration={"fx": 4.0, "fy": 4.0, "cx": 3.5, "cy": 3.5, "distortion": ()},
        runtime_config=VideoGpuRuntimeConfig(cuda_mode="required", maximum_resident_frames=1),
    )
    pair = PairPlan(10, 20, 0, "none", False, False, True, False, "hard_owner", "none")
    prepared = algorithm.prepare(session=None, online_state=None, context={"pair_plans": (pair,)})

    result = algorithm.render(prepared)

    assert result.panorama_bgr.shape == (8, 8, 3)
    assert np.all(result.owner_frame_id[:, :4] == 10)
    assert np.all(result.owner_frame_id[:, 4:] == 20)
    assert np.all(result.panorama_bgr[:, :4, 2] == 20)
    assert np.all(result.panorama_bgr[:, 4:, 0] == 180)
    runtime = result.algorithm_audit["gpu_runtime"]
    assert runtime["h2d_frame_upload_count"] == 2
    assert runtime["intermediate_d2h_count"] == 0
    assert runtime["final_d2h_copy_count"] == 2
    assert prepared.context_audit["components"]["cuda_calibration_and_strict_owner_data_plane"] is True


@pytest.mark.skipif(not _torch().cuda.is_available(), reason="requires actual CUDA")
def test_v2_cuda_strip_uses_full_raw_source_extent_for_inverse_grid_normalisation():
    algorithm = TorchCudaStripOwnerAlgorithm(
        sources=(_source_with_red_ramp(10), _source_with_red_ramp(20)),
        strips=(CudaSourceStrip(10, 0, 2, 3), CudaSourceStrip(20, 3, 5, 3)),
        output_height=8,
        output_width=6,
        calibration={"fx": 4.0, "fy": 4.0, "cx": 3.5, "cy": 3.5, "distortion": ()},
        runtime_config=VideoGpuRuntimeConfig(cuda_mode="required", maximum_resident_frames=1),
    )
    pair = PairPlan(10, 20, 0, "none", False, False, True, False, "hard_owner", "none")

    result = algorithm.render(algorithm.prepare(session=None, online_state=None, context={"pair_plans": (pair,)}))

    # RGB source channel 0 becomes BGR channel 2.  The two strips must sample
    # raw columns 2..4 and 5..7 rather than normalising against width=3.
    assert np.array_equal(result.panorama_bgr[4, :, 2], np.array([40, 60, 80, 100, 120, 140], dtype=np.uint8))


def test_v2_cuda_strip_planner_uses_audited_real_pose_layout_not_cpu_rgb_remaps():
    class Layout:
        frame_ids = (10, 20)
        owner_left_x = (0.0, 4.0)
        owner_right_x = (4.0, 8.0)
        source_centres_x = (3.5, 7.5)
        canvas_width = 8

    strips = build_cuda_strips_from_pushbroom_layout(Layout(), calibration_width=8, calibration_cx=3.5)

    assert strips == (CudaSourceStrip(10, 0, 0, 4), CudaSourceStrip(20, 4, 0, 4))


def test_v2_cuda_strip_planner_keeps_redundant_real_node_out_of_final_owner_map():
    class Layout:
        frame_ids = (10, 15, 20)
        owner_left_x = (0.0, 4.0, 4.0)
        owner_right_x = (4.0, 4.0, 8.0)
        source_centres_x = (3.5, 4.0, 7.5)
        canvas_width = 8

    strips = build_cuda_strips_from_pushbroom_layout(Layout(), calibration_width=8, calibration_cx=3.5)

    assert strips == (CudaSourceStrip(10, 0, 0, 4), CudaSourceStrip(20, 4, 0, 4))


def test_v2_cuda_strip_rejects_planned_components_it_does_not_execute():
    algorithm = TorchCudaStripOwnerAlgorithm(
        sources=(_source(10, 20, 0), _source(20, 0, 180)),
        strips=(CudaSourceStrip(10, 0, 0, 4), CudaSourceStrip(20, 4, 4, 4)),
        output_height=8,
        output_width=8,
        calibration={"fx": 4.0, "fy": 4.0, "cx": 3.5, "cy": 3.5},
    )
    planned_c1 = PairPlan(10, 20, 1, "none", False, False, True, False, "curved_hard_owner", "none")

    with pytest.raises(VideoAlgorithmContractError, match="executes only hard_owner"):
        algorithm.prepare(session=None, online_state=None, context={"pair_plans": (planned_c1,)})


def test_c1_cuda_prepare_requires_real_overlap_and_curved_owner_plan():
    source = lambda frame_id: CudaRealSource(  # noqa: E731
        frame_id=frame_id,
        timestamp_us=frame_id,
        color_u8_rgb=np.zeros((12, 192, 3), dtype=np.uint8),
        depth_mm=np.full((12, 192), 1000.0, dtype=np.float32),
        camera_to_world=(
            np.eye(4, dtype=np.float64)
            if frame_id == 10
            else np.asarray(
                [[1.0, 0.0, 0.0, 800.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
                dtype=np.float64,
            )
        ),
    )
    algorithm = TorchCudaC1ConstrainedOwnerAlgorithm(
        sources=(source(10), source(20)),
        strips=(
            CudaSourceStrip(10, 0, 48, 80, 48.0),
            CudaSourceStrip(20, 80, 48, 80, 128.0),
        ),
        output_height=12,
        output_width=160,
        calibration={"fx": 100.0, "fy": 100.0, "cx": 96.0, "cy": 6.0},
        c1_config=CudaC1ConstrainedOwnerConfig(corridor_width_pixels=96),
    )
    plan = PairPlan(
        left_frame_id=10,
        right_frame_id=20,
        risk_level=1,
        flow_backend="none",
        use_raft_backward=False,
        use_depth_mesh=False,
        use_open3d=True,
        object_lock_required=False,
        seam_mode="curved_hard_owner",
        blend_mode="none",
    )

    prepared = algorithm.prepare(session=None, online_state=None, context={"pair_plans": (plan,)})

    assert prepared.context_audit["renderer"] == "torch_cuda_c1_constrained_owner_v2"


@pytest.mark.skipif(not _torch().cuda.is_available(), reason="requires actual CUDA")
def test_c1_cuda_renderer_changes_owner_only_from_device_resident_real_pair_samples():
    source = lambda frame_id, value: CudaRealSource(  # noqa: E731
        frame_id=frame_id,
        timestamp_us=frame_id,
        color_u8_rgb=np.full((12, 192, 3), value, dtype=np.uint8),
        depth_mm=np.full((12, 192), 1000.0, dtype=np.float32),
        camera_to_world=np.eye(4, dtype=np.float64),
    )
    algorithm = TorchCudaC1ConstrainedOwnerAlgorithm(
        sources=(source(10, 24), source(20, 196)),
        strips=(
            CudaSourceStrip(10, 0, 48, 80, 48.0),
            CudaSourceStrip(20, 80, 48, 80, 128.0),
        ),
        output_height=12,
        output_width=160,
        calibration={"fx": 100.0, "fy": 100.0, "cx": 96.0, "cy": 6.0},
        runtime_config=VideoGpuRuntimeConfig(cuda_mode="required", maximum_resident_frames=2),
    )
    plan = PairPlan(10, 20, 1, "none", False, False, True, False, "curved_hard_owner", "none")

    result = algorithm.render(algorithm.prepare(session=None, online_state=None, context={"pair_plans": (plan,)}))

    c1 = result.algorithm_audit["c1_constrained_owner"]
    assert c1["executed_and_affected_owner_output"] is True
    assert c1["owner_pixels_changed_from_initial_hard_strip"] > 0
    assert result.algorithm_audit["executed_candidate_components"] == {"c1_constrained_owner": True}
    assert set(np.unique(result.owner_frame_id)) == {10, 20}
    runtime = result.algorithm_audit["gpu_runtime"]
    assert runtime["h2d_frame_upload_count"] == 2
    assert runtime["intermediate_d2h_count"] == 0


@pytest.mark.skipif(not _torch().cuda.is_available(), reason="requires actual CUDA")
def test_c2_cuda_dis_mesh_resamples_accepted_real_owner_pixels_without_host_round_trip():
    height, width = 16, 192
    rng = np.random.default_rng(20260805)
    first_rgb = rng.integers(20, 235, (height, width, 3), dtype=np.uint8)
    # The C1 layout below maps the second source 80 raw pixels behind the
    # first.  An additional one-pixel local residual is therefore represented
    # by -81; C2 must discover it and apply it through a composed source grid.
    second_rgb = np.roll(first_rgb, -81, axis=1)
    source = lambda frame_id, image: CudaRealSource(  # noqa: E731
        frame_id=frame_id,
        timestamp_us=frame_id,
        color_u8_rgb=image,
        depth_mm=np.full((height, width), 1000.0, dtype=np.float32),
        camera_to_world=(
            np.eye(4, dtype=np.float64)
            if frame_id == 10
            else np.asarray(
                [[1.0, 0.0, 0.0, 800.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
                dtype=np.float64,
            )
        ),
    )
    algorithm = TorchCudaC2DisResidualMeshAlgorithm(
        sources=(source(10, first_rgb), source(20, second_rgb)),
        strips=(
            CudaSourceStrip(10, 0, 48, 80, 48.0),
            CudaSourceStrip(20, 80, 48, 80, 128.0),
        ),
        output_height=height,
        output_width=160,
        calibration={"fx": 100.0, "fy": 100.0, "cx": 96.0, "cy": 7.5},
        runtime_config=VideoGpuRuntimeConfig(cuda_mode="required", maximum_resident_frames=2),
    )
    plan = PairPlan(10, 20, 1, "dis", False, False, True, False, "curved_hard_owner", "none")

    result = algorithm.render(algorithm.prepare(session=None, online_state=None, context={"pair_plans": (plan,)}))

    c2 = result.algorithm_audit["c2_dis_mesh"]
    assert c2["executed_and_affected_output"] is True
    assert c2["actual_output_mesh_pixel_count"] > 0
    pair = c2["pair_audits"][0]
    assert pair["orb_rgbd_inverse_grid_prior"]["uses_real_orb_camera_to_world"] is True
    assert pair["orb_rgbd_inverse_grid_prior"]["uses_real_aligned_depth"] is True
    assert pair["orb_rgbd_inverse_grid_prior"]["creates_panorama_depth"] is False
    assert pair["mesh"]["train_held_out_disjoint"] is True
    assert pair["mesh"]["accepted"] is True
    assert pair["mesh_applied_to_actual_output"] is True
    assert result.algorithm_audit["executed_candidate_components"] == {
        "c1_constrained_owner": True,
        "c2_dis_mesh": True,
    }
    runtime = result.algorithm_audit["gpu_runtime"]
    assert runtime["h2d_frame_upload_count"] == 2
    assert runtime["intermediate_d2h_count"] == 0


@pytest.mark.skipif(not _torch().cuda.is_available(), reason="requires actual CUDA")
def test_c3_cuda_uses_bidirectional_resident_raft_and_applies_accepted_mesh():
    height, width = 16, 192
    rng = np.random.default_rng(20260806)
    first_rgb = rng.integers(20, 235, (height, width, 3), dtype=np.uint8)
    second_rgb = np.roll(first_rgb, -81, axis=1)

    class ResidentRAFT:
        def __init__(self) -> None:
            self.device = "cuda:0"
            self.calls: list[tuple[int, int, tuple[int, ...]]] = []

        def estimate_pair_tensors(self, source_rgb, target_rgb, *, source_frame_id, target_frame_id):
            self.calls.append((source_frame_id, target_frame_id, tuple(source_rgb.shape)))
            assert source_rgb.device.type == "cuda"
            assert target_rgb.device.type == "cuda"
            # C1 already accounts for -80/+80 raw pixels between these
            # source grids.  The real full-image RAFT displacement is
            # -81/+81, leaving the intended +1/-1 local residual after C1
            # layout subtraction.  Return source-coordinate flow, never a
            # pre-warped residual or host array.
            direction = -81.0 if source_frame_id == 10 else 81.0
            flow = _torch().zeros(
                (int(source_rgb.shape[1]), int(source_rgb.shape[2]), 2),
                device=source_rgb.device,
                dtype=_torch().float32,
            )
            flow[..., 0] = direction
            return SimpleNamespace(
                flow_xy=flow,
                audit=SimpleNamespace(
                    as_dict=lambda: {
                        "model": "test_locked_raft_small",
                        "source_frame_id": source_frame_id,
                        "target_frame_id": target_frame_id,
                        "output_residency": "device_tensor",
                        "host_transfer_count": 0,
                    }
                ),
            )

    source = lambda frame_id, image: CudaRealSource(  # noqa: E731
        frame_id=frame_id,
        timestamp_us=frame_id,
        color_u8_rgb=image,
        depth_mm=np.full((height, width), 1000.0, dtype=np.float32),
        camera_to_world=np.eye(4, dtype=np.float64),
    )
    raft = ResidentRAFT()
    algorithm = TorchCudaC3RAFTResidualMeshAlgorithm(
        sources=(source(10, first_rgb), source(20, second_rgb)),
        strips=(
            CudaSourceStrip(10, 0, 48, 80, 48.0),
            CudaSourceStrip(20, 80, 48, 80, 128.0),
        ),
        output_height=height,
        output_width=160,
        calibration={"fx": 100.0, "fy": 100.0, "cx": 96.0, "cy": 7.5},
        runtime_config=VideoGpuRuntimeConfig(cuda_mode="required", maximum_resident_frames=2),
        raft_runtime=raft,
    )
    plan = PairPlan(10, 20, 1, "raft_small", True, False, True, False, "curved_hard_owner", "none")

    result = algorithm.render(algorithm.prepare(session=None, online_state=None, context={"pair_plans": (plan,)}))

    assert raft.calls == [(10, 20, (3, height, width)), (20, 10, (3, height, width))]
    c3 = result.algorithm_audit["c3_raft_mesh"]
    assert c3["executed_and_affected_output"] is True
    pair = c3["pair_audits"][0]
    assert pair["mesh"]["accepted"] is True
    assert pair["mesh_applied_to_actual_output"] is True
    assert pair["raft_forward"]["host_transfer_count"] == 0
    assert pair["raft_backward"]["host_transfer_count"] == 0
    assert "c2_dis_mesh" not in result.algorithm_audit["executed_candidate_components"]
    assert result.algorithm_audit["executed_candidate_components"]["c3_raft_mesh"] is True
    runtime = result.algorithm_audit["gpu_runtime"]
    assert runtime["h2d_frame_upload_count"] == 2
    assert runtime["intermediate_d2h_count"] == 0


@pytest.mark.skipif(not _torch().cuda.is_available(), reason="requires actual CUDA")
def test_c3_cuda_raft_error_retains_safe_c1_owner_output():
    height, width = 12, 192

    class FailingResidentRAFT:
        device = "cuda:0"

        def estimate_pair_tensors(self, *_args, **_kwargs):
            raise RAFTSmallRuntimeError("synthetic locked-model inference failure")

    source = lambda frame_id, value: CudaRealSource(  # noqa: E731
        frame_id=frame_id,
        timestamp_us=frame_id,
        color_u8_rgb=np.full((height, width, 3), value, dtype=np.uint8),
        depth_mm=np.full((height, width), 1000.0, dtype=np.float32),
        camera_to_world=np.eye(4, dtype=np.float64),
    )
    algorithm = TorchCudaC3RAFTResidualMeshAlgorithm(
        sources=(source(10, 24), source(20, 196)),
        strips=(
            CudaSourceStrip(10, 0, 48, 80, 48.0),
            CudaSourceStrip(20, 80, 48, 80, 128.0),
        ),
        output_height=height,
        output_width=160,
        calibration={"fx": 100.0, "fy": 100.0, "cx": 96.0, "cy": 5.5},
        runtime_config=VideoGpuRuntimeConfig(cuda_mode="required", maximum_resident_frames=2),
        raft_runtime=FailingResidentRAFT(),
    )
    plan = PairPlan(10, 20, 1, "raft_small", True, False, True, False, "curved_hard_owner", "none")

    result = algorithm.render(algorithm.prepare(session=None, online_state=None, context={"pair_plans": (plan,)}))

    c3 = result.algorithm_audit["c3_raft_mesh"]
    assert c3["executed_and_affected_output"] is False
    assert c3["pair_audits"][0]["fallback_to_c1_hard_owner"] is True
    assert "synthetic locked-model inference failure" in c3["pair_audits"][0]["raft_or_mesh_exception"]
    assert result.algorithm_audit["executed_candidate_components"]["c3_raft_mesh"] is False
    assert np.all(result.owner_frame_id >= 0)


@pytest.mark.skipif(not _torch().cuda.is_available(), reason="requires actual CUDA")
def test_c4_cuda_mesh_requires_c3_audit_and_real_same_layer_depth_before_sampling():
    height, width = 16, 192
    rng = np.random.default_rng(20260808)
    first_rgb = rng.integers(20, 235, (height, width, 3), dtype=np.uint8)
    second_rgb = np.roll(first_rgb, -81, axis=1)

    class ResidentRAFT:
        device = "cuda:0"

        def estimate_pair_tensors(self, source_rgb, _target_rgb, *, source_frame_id, target_frame_id):
            assert source_rgb.device.type == "cuda"
            # Full-image flow includes C1's -80/+80 raw-grid displacement;
            # only its remaining +1/-1 is eligible as the local mesh.
            direction = -81.0 if source_frame_id == 10 else 81.0
            flow = _torch().zeros(
                (int(source_rgb.shape[1]), int(source_rgb.shape[2]), 2),
                device=source_rgb.device, dtype=_torch().float32,
            )
            flow[..., 0] = direction
            return SimpleNamespace(
                flow_xy=flow,
                audit=SimpleNamespace(as_dict=lambda: {
                    "model": "test_locked_raft_small", "source_frame_id": source_frame_id,
                    "target_frame_id": target_frame_id, "output_residency": "device_tensor",
                    "host_transfer_count": 0,
                }),
            )

    first_depth = np.full((height, width), 1000.0, dtype=np.float32)
    second_depth = np.full((height, width), 1000.0, dtype=np.float32)
    # Matching real aligned depths leave a same-layer, pose-consistent
    # background corridor for C4's independently held-out residual mesh.
    source = lambda frame_id, image, depth: CudaRealSource(  # noqa: E731
        frame_id=frame_id, timestamp_us=frame_id, color_u8_rgb=image,
        depth_mm=depth,
        # The C1 grids differ by 80 raw pixels; the immutable real-pose
        # prior therefore maps the target grid back by an 800 mm X baseline
        # at fx=100 and z=1000 mm before RAFT fits its remaining 1 px.
        camera_to_world=(
            np.eye(4, dtype=np.float64)
            if frame_id == 10
            else np.asarray(
                [[1.0, 0.0, 0.0, 800.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
                dtype=np.float64,
            )
        ),
    )
    algorithm = TorchCudaC4RAFTDepthLayeredMeshAlgorithm(
        sources=(source(10, first_rgb, first_depth), source(20, second_rgb, second_depth)),
        strips=(
            CudaSourceStrip(10, 0, 48, 80, 48.0),
            CudaSourceStrip(20, 80, 48, 80, 128.0),
        ),
        output_height=height, output_width=160,
        calibration={"fx": 100.0, "fy": 100.0, "cx": 96.0, "cy": 7.5},
        runtime_config=VideoGpuRuntimeConfig(cuda_mode="required", maximum_resident_frames=2),
        raft_runtime=ResidentRAFT(),
    )
    plan = PairPlan(10, 20, 1, "raft_small", True, True, True, False, "curved_hard_owner", "none")

    result = algorithm.render(algorithm.prepare(session=None, online_state=None, context={"pair_plans": (plan,)}))

    c4 = result.algorithm_audit["c4_raft_rgbd_layered_mesh"]
    pair = c4["pair_audits"][0]
    assert pair["depth_layers"]["absolute_tolerance_mm"] == 20.0
    assert pair["depth_layers"]["relative_tolerance"] == 0.02
    assert pair["depth_layers"]["forward_backward_consistent_pixel_count"] > 0
    prior = pair["orb_rgbd_inverse_grid_prior"]
    assert prior["target_valid_pixel_count"] == prior["safe_inverse_sample_pixel_count"] > 0
    assert pair["depth_protected_mesh_candidate_pixel_count"] == 0
    assert pair["mesh"]["train_held_out_disjoint"] is True
    assert pair["mesh"]["accepted"] is True
    assert c4["actual_output_mesh_pixel_count"] > 0
    assert result.algorithm_audit["executed_candidate_components"] == {
        "c1_constrained_owner": True,
        "c3_raft_mesh": True,
        "c4_depth_layered_mesh": True,
    }
    runtime = result.algorithm_audit["gpu_runtime"]
    assert runtime["h2d_frame_upload_count"] == 2
    assert runtime["intermediate_d2h_count"] == 0


@pytest.mark.skipif(not _torch().cuda.is_available(), reason="requires actual CUDA")
def test_c5_cuda_object_lock_ignores_semantic_masks_and_uses_real_aligned_depth_protection():
    """C5 must run C4 first, then pin only a depth-observed corridor."""

    height, width = 16, 192
    rng = np.random.default_rng(20260809)
    first_rgb = rng.integers(20, 235, (height, width, 3), dtype=np.uint8)
    second_rgb = np.roll(first_rgb, -81, axis=1)

    class ResidentRAFT:
        device = "cuda:0"

        def estimate_pair_tensors(self, source_rgb, _target_rgb, *, source_frame_id, target_frame_id):
            assert source_rgb.device.type == "cuda"
            direction = 1.0 if source_frame_id == 10 else -1.0
            flow = _torch().zeros(
                (int(source_rgb.shape[1]), int(source_rgb.shape[2]), 2),
                device=source_rgb.device, dtype=_torch().float32,
            )
            flow[..., 0] = direction
            return SimpleNamespace(
                flow_xy=flow,
                audit=SimpleNamespace(as_dict=lambda: {
                    "model": "test_locked_raft_small", "source_frame_id": source_frame_id,
                    "target_frame_id": target_frame_id, "output_residency": "device_tensor",
                    "host_transfer_count": 0,
                }),
            )

    depth = np.full((height, width), 1000.0, dtype=np.float32)
    depth[:, 96:] = 1300.0
    source = lambda frame_id, image, object_mask: CudaRealSource(  # noqa: E731
        frame_id=frame_id, timestamp_us=frame_id, color_u8_rgb=image,
        depth_mm=depth,
        camera_to_world=np.eye(4, dtype=np.float64), object_mask=object_mask,
    )
    # A deliberately all-true semantic raster proves that C5 never uploads or
    # consumes annotation-like input.  The only protection comes from depth.
    first_object = np.ones((height, width), dtype=bool)
    algorithm = TorchCudaC5ObjectLockAlgorithm(
        sources=(source(10, first_rgb, first_object), source(20, second_rgb, None)),
        strips=(
            CudaSourceStrip(10, 0, 48, 80, 48.0),
            CudaSourceStrip(20, 80, 48, 80, 128.0),
        ),
        output_height=height, output_width=160,
        calibration={"fx": 100.0, "fy": 100.0, "cx": 96.0, "cy": 7.5},
        runtime_config=VideoGpuRuntimeConfig(cuda_mode="required", maximum_resident_frames=2),
        raft_runtime=ResidentRAFT(),
    )
    plan = PairPlan(10, 20, 1, "raft_small", True, True, True, True, "curved_hard_owner", "none")

    result = algorithm.render(algorithm.prepare(session=None, online_state=None, context={"pair_plans": (plan,)}))

    c5 = result.algorithm_audit["c5_object_lock"]
    pair = c5["pair_audits"][0]
    assert pair["protection"]["output_residency"] == "device_tensor"
    assert pair["protection"]["object_protected_pixel_count"] == 0
    assert pair["protection"]["depth_edge_protected_pixel_count"] > 0
    assert pair["protection"]["manual_measurement_annotations_used"] is False
    assert pair["owner_lock"]["accepted"] is True
    assert pair["owner_lock"]["host_transfer_count"] == 0
    # The observation-derived edge is already owned by C4's same real source,
    # so a no-change result is the required fail-closed outcome rather than a
    # reason to use the supplied semantic raster.
    assert c5["owner_pixels_changed_from_c4"] == 0
    assert c5["protected_pixels_recomposed_from_real_owner"] > 0
    assert result.algorithm_audit["executed_candidate_components"]["c5_object_lock"] is False
    assert c5["protection_input"] == "aligned_depth_only"
    assert c5["manual_measurement_annotations_used"] is False
    runtime = result.algorithm_audit["gpu_runtime"]
    assert runtime["h2d_frame_upload_count"] == 2
    assert runtime["intermediate_d2h_count"] == 0


@pytest.mark.skipif(not _torch().cuda.is_available(), reason="requires actual CUDA")
def test_c6_cuda_multiband_composites_only_shared_low_risk_background_and_preserves_owner():
    """C6 must extend C5 and never turn blended provenance into a new owner."""

    height, width = 16, 192
    # Low-gradient, mutually valid real sources make an eligible background
    # corridor.  Their small (below-risk-threshold) colour offset still makes
    # the device MultiBand result observably distinct from the C5 hard cut.
    ramp = np.rint(np.linspace(80, 160, width, dtype=np.float32)).astype(np.uint8)
    first_rgb = np.repeat(ramp[None, :, None], height, axis=0)
    first_rgb = np.repeat(first_rgb, 3, axis=2)
    # The small, smooth pair difference is lowest at the middle so C1 keeps
    # both genuine owners in its corridor, while it remains below C6's
    # conservative texture/disagreement risk thresholds.
    offset = np.rint(np.abs(np.arange(width, dtype=np.float32) - width / 2.0) * (20.0 / (width / 2.0))).astype(np.uint8)
    second_rgb = np.repeat((100 + offset)[None, :, None], height, axis=0)
    second_rgb = np.repeat(second_rgb, 3, axis=2)

    class ResidentRAFT:
        device = "cuda:0"

        def estimate_pair_tensors(self, source_rgb, _target_rgb, *, source_frame_id, target_frame_id):
            flow = _torch().zeros(
                (int(source_rgb.shape[1]), int(source_rgb.shape[2]), 2),
                device=source_rgb.device, dtype=_torch().float32,
            )
            return SimpleNamespace(
                flow_xy=flow,
                audit=SimpleNamespace(as_dict=lambda: {
                    "model": "test_locked_raft_small", "source_frame_id": source_frame_id,
                    "target_frame_id": target_frame_id, "output_residency": "device_tensor",
                    "host_transfer_count": 0,
                }),
            )

    source = lambda frame_id, image: CudaRealSource(  # noqa: E731
        frame_id=frame_id, timestamp_us=frame_id, color_u8_rgb=image,
        depth_mm=np.full((height, width), 1000.0, dtype=np.float32),
        camera_to_world=np.eye(4, dtype=np.float64), object_mask=None,
    )
    algorithm = TorchCudaC6SafeMultiBandAlgorithm(
        sources=(source(10, first_rgb), source(20, second_rgb)),
        strips=(
            CudaSourceStrip(10, 0, 48, 80, 48.0),
            CudaSourceStrip(20, 80, 48, 80, 128.0),
        ),
        output_height=height, output_width=160,
        calibration={"fx": 100.0, "fy": 100.0, "cx": 96.0, "cy": 7.5},
        runtime_config=VideoGpuRuntimeConfig(cuda_mode="required", maximum_resident_frames=2),
        raft_runtime=ResidentRAFT(),
    )
    plan = PairPlan(10, 20, 1, "raft_small", True, True, True, True, "curved_hard_owner", "safe_multiband")

    result = algorithm.render(algorithm.prepare(session=None, online_state=None, context={"pair_plans": (plan,)}))

    c6 = result.algorithm_audit["c6_safe_multiband"]
    pair = c6["pair_audits"][0]
    assert pair["common_real_source_valid_pixel_count"] > 0
    assert pair["safe_background_pixel_count"] > 0
    assert pair["multiband"]["output_residency"] == "device_tensor"
    assert pair["multiband"]["owner_map_preserved"] is True
    assert pair["composited_pixel_count"] > 0
    assert c6["executed_and_affected_output"] is True
    assert result.algorithm_audit["executed_candidate_components"]["c6_safe_multiband"] is True
    assert set(np.unique(result.owner_frame_id).tolist()) == {10, 20}
    runtime = result.algorithm_audit["gpu_runtime"]
    assert runtime["h2d_frame_upload_count"] == 2
    assert runtime["intermediate_d2h_count"] == 0


@pytest.mark.skipif(not _torch().cuda.is_available(), reason="requires actual CUDA")
def test_c7_cuda_global_photometric_changes_real_source_samples_before_c6_composition():
    """C7 must fit/apply on CUDA and prove a non-identity C6 output impact."""

    height, width = 16, 192
    ramp = np.rint(np.linspace(80, 160, width, dtype=np.float32)).astype(np.uint8)
    first_rgb = np.repeat(ramp[None, :, None], height, axis=0)
    first_rgb = np.repeat(first_rgb, 3, axis=2)
    # Smooth, jointly visible source disagreement remains below C6's risk
    # threshold while providing an affine linear-light C7 correction signal.
    second_rgb = np.clip(first_rgb.astype(np.int16) + 10, 0, 255).astype(np.uint8)

    class ResidentRAFT:
        device = "cuda:0"

        def estimate_pair_tensors(self, source_rgb, _target_rgb, *, source_frame_id, target_frame_id):
            flow = _torch().zeros((int(source_rgb.shape[1]), int(source_rgb.shape[2]), 2), device=source_rgb.device)
            return SimpleNamespace(flow_xy=flow, audit=SimpleNamespace(as_dict=lambda: {
                "model": "test_locked_raft_small", "source_frame_id": source_frame_id,
                "target_frame_id": target_frame_id, "output_residency": "device_tensor", "host_transfer_count": 0,
            }))

    source = lambda frame_id, image: CudaRealSource(  # noqa: E731
        frame_id=frame_id, timestamp_us=frame_id, color_u8_rgb=image,
        depth_mm=np.full((height, width), 1000.0, dtype=np.float32), camera_to_world=np.eye(4), object_mask=None,
    )
    strips = (CudaSourceStrip(10, 0, 48, 80, 48.0), CudaSourceStrip(20, 80, 48, 80, 128.0))
    plan = PairPlan(10, 20, 1, "raft_small", True, True, True, True, "curved_hard_owner", "safe_multiband")
    common = dict(
        sources=(source(10, first_rgb), source(20, second_rgb)), strips=strips,
        output_height=height, output_width=160,
        calibration={"fx": 100.0, "fy": 100.0, "cx": 96.0, "cy": 7.5},
        raft_runtime=ResidentRAFT(), c1_config=CudaC1ConstrainedOwnerConfig(),
    )
    c6 = TorchCudaC6SafeMultiBandAlgorithm(
        **common, runtime_config=VideoGpuRuntimeConfig(cuda_mode="required", maximum_resident_frames=2)
    )
    c7 = TorchCudaC7PhotometricGraphAlgorithm(
        **common, runtime_config=VideoGpuRuntimeConfig(cuda_mode="required", maximum_resident_frames=5)
    )
    c6_result = c6.render(c6.prepare(session=None, online_state=None, context={"pair_plans": (plan,)}))
    c7_result = c7.render(c7.prepare(session=None, online_state=None, context={"pair_plans": (plan,)}))

    audit = c7_result.algorithm_audit["c7_global_photometric"]
    assert audit["accepted"] is True, audit
    assert audit["linear_light"] is True
    assert audit["common_visible_safe_background_overlaps"][0]["common_visible_safe_background_pixel_count"] > 0
    assert audit["corrected_real_source_sample_pixel_count"] > 0
    assert audit["executed_and_affected_output"] is True
    assert c7_result.algorithm_audit["executed_candidate_components"]["c7_global_photometric"] is True
    assert not np.array_equal(c6_result.panorama_bgr, c7_result.panorama_bgr)
    runtime = c7_result.algorithm_audit["gpu_runtime"]
    assert runtime["h2d_frame_upload_count"] == 2
    assert runtime["intermediate_d2h_count"] == 0


@pytest.mark.skipif(not _torch().cuda.is_available(), reason="requires actual CUDA")
def test_c8_cuda_multilabel_recomposes_resident_real_sources_and_preserves_depth_owner():
    """C8 must run 2--5 real sources locally, without upload/replay or fake pixels."""

    height, source_width, output_width = 16, 256, 192

    def image(value: int) -> np.ndarray:
        return np.full((height, source_width, 3), value, dtype=np.uint8)

    depth_step = np.full((height, source_width), 1000.0, dtype=np.float32)
    depth_step[:, 96:] = 1300.0
    source = lambda frame_id, value, mask=None, depth=None: CudaRealSource(  # noqa: E731
        frame_id=frame_id, timestamp_us=frame_id, color_u8_rgb=image(value),
        depth_mm=np.full((height, source_width), 1000.0, dtype=np.float32) if depth is None else depth,
        camera_to_world=np.eye(4), object_mask=mask,
    )

    class ResidentRAFT:
        device = "cuda:0"

        def estimate_pair_tensors(self, source_rgb, _target_rgb, *, source_frame_id, target_frame_id):
            flow = _torch().zeros((int(source_rgb.shape[1]), int(source_rgb.shape[2]), 2), device=source_rgb.device)
            return SimpleNamespace(flow_xy=flow, audit=SimpleNamespace(as_dict=lambda: {
                "model": "test_locked_raft_small", "source_frame_id": source_frame_id,
                "target_frame_id": target_frame_id, "output_residency": "device_tensor", "host_transfer_count": 0,
            }))

    # Supplying an annotation-like mask is intentional: C8 must ignore it.
    semantic_mask = np.ones((height, source_width), dtype=np.bool_)
    sources = (source(10, 95), source(20, 110, semantic_mask, depth_step), source(30, 105))
    # Every source has genuine inverse-map support over the local C8 output
    # window; the strips themselves remain the chronological C1 base owners.
    strips = (
        CudaSourceStrip(10, 0, 0, 64, 0.0),
        CudaSourceStrip(20, 64, 64, 64, 64.0),
        CudaSourceStrip(30, 128, 128, 64, 128.0),
    )
    plans = tuple(
        PairPlan(left.frame_id, right.frame_id, 1, "raft_small", True, True, True, True, "curved_hard_owner", "safe_multiband")
        for left, right in zip(sources[:-1], sources[1:], strict=True)
    )
    common = dict(
        sources=sources, strips=strips, output_height=height, output_width=output_width,
        calibration={"fx": 100.0, "fy": 100.0, "cx": 96.0, "cy": 7.5},
        runtime_config=VideoGpuRuntimeConfig(cuda_mode="required", maximum_resident_frames=3),
        raft_runtime=ResidentRAFT(), c1_config=CudaC1ConstrainedOwnerConfig(),
    )
    c7 = TorchCudaC7PhotometricGraphAlgorithm(**common)
    c8 = TorchCudaC8MultilabelWindowAlgorithm(**common)
    prepared_c7 = c7.prepare(session=None, online_state=None, context={"pair_plans": plans})
    prepared_c8 = c8.prepare(session=None, online_state=None, context={"pair_plans": plans})
    c7_result = c7.render(prepared_c7)
    result = c8.render(prepared_c8)

    c8_audit = result.algorithm_audit["c8_multilabel_window"]
    assert result.algorithm_audit["renderer"] == "torch_cuda_c8_multilabel_window_v2"
    assert c8_audit["rejected_returns_c7_c6_output"] is False, c8_audit
    assert c8_audit["window_audits"][0]["window_frame_count"] == 3
    assert c8_audit["window_audits"][0]["optimiser"]["output_residency"] == "device_tensor"
    assert c8_audit["window_audits"][0]["object_protected_pixel_count"] == 0
    assert c8_audit["window_audits"][0]["depth_protected_pixel_count"] > 0
    assert "c8_local_multilabel_owner" in result.algorithm_audit["executed_candidate_components"]
    assert {"c1_constrained_owner", "c3_raft_mesh", "c4_depth_layered_mesh", "c5_object_owner_lock", "c6_safe_multiband", "c7_global_photometric"}.issubset(
        result.algorithm_audit["executed_candidate_components"]
    )
    assert result.owner_frame_id[5, 96] == c7_result.owner_frame_id[5, 96]
    assert set(np.unique(result.owner_frame_id)).issubset({10, 20, 30})
    assert np.all(result.owner_frame_id[:, 1:] >= result.owner_frame_id[:, :-1])
    runtime = result.algorithm_audit["gpu_runtime"]
    assert runtime["h2d_frame_upload_count"] == 3
    assert runtime["intermediate_d2h_count"] == 0
    assert runtime["final_d2h_copy_count"] == 2
