from __future__ import annotations

import json

import numpy as np

from panorama_demo.rapidocr_onnx_adapter import (
    audit_profile_cuda_execution,
    normalize_rapidocr_result,
    summarize_onnxruntime_profile,
)


def test_result_normalization_is_deterministic_and_validated() -> None:
    raw = [
        [
            [[20, 20], [30, 20], [30, 30], [20, 30]],
            " Wave\u3000Share ",
            0.91,
        ],
        [
            [[2, 2], [10, 2], [10, 8], [2, 8]],
            "ＡＢＣ",
            0.75,
        ],
        [[[0, 0], [1, 0], [2, 0], [3, 0]], "bad", 0.9],
        [[[0, 0], [1, 0], [1, 1], [0, 1]], "", 0.9],
    ]
    detections = normalize_rapidocr_result(
        raw, image_width=40, image_height=40
    )
    assert [item.text for item in detections] == ["ABC", "Wave Share"]
    assert detections[0].score == 0.75
    assert detections[0].polygon_xy.shape == (4, 2)
    assert detections[0].polygon_xy.dtype == np.float32


def test_cuda_heavy_compute_with_cpu_shape_nodes_passes() -> None:
    audit = audit_profile_cuda_execution(
        {
            "detection": {
                "provider_node_events": {
                    "CUDAExecutionProvider": 20,
                    "CPUExecutionProvider": 2,
                },
                "operator_providers": {
                    "Conv": ["CUDAExecutionProvider"],
                    "Shape": ["CPUExecutionProvider"],
                },
            }
        },
        allow_shape_control_cpu=True,
    )
    assert audit["pass"] is True
    assert audit["heavy_cuda_operator_kind_count"] == 1


def test_cpu_heavy_compute_is_rejected_even_when_cuda_is_registered() -> None:
    audit = audit_profile_cuda_execution(
        {
            "recognition": {
                "provider_node_events": {
                    "CUDAExecutionProvider": 4,
                    "CPUExecutionProvider": 20,
                },
                "operator_providers": {
                    "Conv": [
                        "CPUExecutionProvider",
                        "CUDAExecutionProvider",
                    ],
                    "Shape": ["CPUExecutionProvider"],
                },
            }
        },
        allow_shape_control_cpu=True,
    )
    assert audit["pass"] is False
    assert "recognition:heavy_compute_operator_on_cpu" in audit["failures"]


def test_whole_graph_cpu_fallback_is_rejected() -> None:
    audit = audit_profile_cuda_execution(
        {
            "detection": {
                "provider_node_events": {"CPUExecutionProvider": 40},
                "operator_providers": {
                    "Conv": ["CPUExecutionProvider"],
                },
            }
        },
        allow_shape_control_cpu=True,
    )
    assert audit["pass"] is False
    assert "detection:executed_without_cuda_node" in audit["failures"]
    assert "detection:heavy_compute_operator_on_cpu" in audit["failures"]


def test_non_shape_cpu_operator_is_rejected() -> None:
    audit = audit_profile_cuda_execution(
        {
            "detection": {
                "provider_node_events": {
                    "CUDAExecutionProvider": 20,
                    "CPUExecutionProvider": 1,
                },
                "operator_providers": {
                    "Conv": ["CUDAExecutionProvider"],
                    "Relu": ["CPUExecutionProvider"],
                },
            }
        },
        allow_shape_control_cpu=True,
    )
    assert audit["pass"] is False
    assert (
        "detection:non_shape_control_operator_on_cpu" in audit["failures"]
    )


def test_profile_summary_records_provider_and_operator(tmp_path) -> None:
    path = tmp_path / "profile.json"
    path.write_text(
        json.dumps(
            [
                {
                    "name": "conv_kernel_time",
                    "dur": 17,
                    "args": {
                        "provider": "CUDAExecutionProvider",
                        "op_name": "Conv",
                    },
                },
                {
                    "name": "shape_kernel_time",
                    "dur": 2,
                    "args": {
                        "provider": "CPUExecutionProvider",
                        "op_name": "Shape",
                    },
                },
            ]
        ),
        encoding="utf-8",
    )
    summary = summarize_onnxruntime_profile(path)
    assert summary["provider_node_events"]["CUDAExecutionProvider"] == 1
    assert summary["operator_providers"]["Conv"] == [
        "CUDAExecutionProvider"
    ]
