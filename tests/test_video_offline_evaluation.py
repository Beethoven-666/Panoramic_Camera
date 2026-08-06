from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from panorama_demo.video_offline_evaluation import (
    PANORAMA_PROJECTION_SCHEMA,
    VideoOfflineEvaluationError,
    evaluate_delivery_artifacts,
    evaluate_offline_visual_annotations,
    load_panorama_annotation_projection,
    write_offline_evaluation,
)
from panorama_demo.video_offline_evaluation import _line_metrics_from_observations


def _annotations() -> dict[str, object]:
    return {
        "schema": "gemini305-video-source-annotations/v1",
        "source_frames": {"7": {"color_path": "color/00000007.jpg", "scan_progress": 0.2}},
        "objects": [{"id": "object", "frame_id": 7, "polygon": [[0, 0], [1, 0], [1, 1]]}],
        "lines": [{"id": "line", "frame_id": 7, "points": [[0, 0], [1, 0]]}],
        "safe_background": [{"id": "background", "frame_id": 7, "polygon": [[0, 0], [1, 0], [1, 1]]}],
    }


def _projection() -> dict[str, object]:
    return {
        "schema": PANORAMA_PROJECTION_SCHEMA,
        "measurement_only": True,
        "panorama_shape": [16, 24],
        "objects": [{"id": "object", "frame_id": 7, "polygon": [[2, 2], [20, 2], [20, 13], [2, 13]]}],
        "lines": [{"id": "line", "frame_id": 7, "points": [[2, 8], [20, 8]]}],
        "safe_background": [{"id": "background", "frame_id": 7, "polygon": [[2, 2], [20, 2], [20, 13], [2, 13]]}],
    }


def _panorama_and_owner() -> tuple[np.ndarray, np.ndarray]:
    image = np.zeros((16, 24, 3), dtype=np.uint8)
    image[:, :12] = (20, 20, 20)
    image[:, 12:] = (20, 20, 20)
    cv2.line(image, (2, 8), (20, 8), (255, 255, 255), 1)
    owner = np.full((16, 24), 7, dtype=np.int32)
    return image, owner


def test_offline_evaluation_is_explicitly_unavailable_without_projection():
    image, owner = _panorama_and_owner()
    result = evaluate_offline_visual_annotations(image, owner, annotations=_annotations())
    assert result["measurement_only"] is True
    assert result["automatic_grade_promotion_allowed"] is False
    assert result["owner_topology"]["active_owner_count"] == 1
    assert result["object_integrity"]["object"]["status"] == "not_evaluable"
    assert result["line_continuity"]["line"]["status"] == "not_evaluable"


def test_projection_backed_evaluation_measures_owner_line_and_background(tmp_path):
    image, owner = _panorama_and_owner()
    projection_path = tmp_path / "projection.json"
    projection_path.write_text(json.dumps(_projection()), encoding="utf-8")
    projection = load_panorama_annotation_projection(
        projection_path, annotations=_annotations(), panorama_shape=image.shape[:2]
    )
    result = evaluate_offline_visual_annotations(image, owner, annotations=_annotations(), projection=projection)
    assert result["object_integrity"]["object"]["owner_count"] == 1
    assert result["object_integrity"]["object"]["hard_gate_pass"] is True
    assert result["line_continuity"]["line"]["status"] == "evaluated"
    assert result["safe_background"]["background"]["status"] == "not_evaluable"


def test_safe_background_perfect_owner_boundary_passes_zero_difference_gate(tmp_path):
    """An exact colour match is a measured zero, rather than missing data."""

    image, owner = _panorama_and_owner()
    owner[:, 12:] = 8
    projection_path = tmp_path / "projection.json"
    projection_path.write_text(json.dumps(_projection()), encoding="utf-8")
    projection = load_panorama_annotation_projection(
        projection_path, annotations=_annotations(), panorama_shape=image.shape[:2]
    )

    result = evaluate_offline_visual_annotations(image, owner, annotations=_annotations(), projection=projection)
    safe = result["safe_background"]["background"]
    assert safe["status"] == "evaluated"
    assert safe["delta_e00_p95"] == 0.0
    assert safe["brightness_step_p95_percent"] == 0.0
    assert safe["hard_gate_pass"] is True


def test_line_metric_keeps_documented_two_pixel_step_and_five_degree_orientation_failures():
    """v2's dense-line plumbing cannot weaken the immutable line gates."""

    metrics = _line_metrics_from_observations(({
        "sample_count": 4,
        "offsets": np.asarray([0.0, 2.0, 0.0, 2.0]),
        "steps": np.asarray([2.0, 2.0, 2.0]),
        "orientation_error": np.asarray([5.0, 5.0, 5.0, 5.0]),
    },))
    assert metrics["line_step_p95_px"] == 2.0
    assert metrics["line_orientation_delta_p95_degrees"] == 5.0
    assert metrics["hard_gate_pass"] is False


def test_projection_rejects_wrong_frame_or_out_of_bounds(tmp_path):
    payload = _projection()
    payload["objects"][0]["frame_id"] = 99
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(VideoOfflineEvaluationError, match="frame_id"):
        load_panorama_annotation_projection(path, annotations=_annotations(), panorama_shape=(16, 24))


def test_explicit_paired_measurement_groups_aggregate_only_projected_members_and_ignore_mask_perimeter(tmp_path):
    """249/257-style pairs are one read-only measurement per annotation kind."""

    annotations = {
        "schema": "gemini305-video-source-annotations/v1",
        "source_frames": {
            "249": {"color_path": "color/00000249.jpg", "scan_progress": 0.39},
            "257": {"color_path": "color/00000257.jpg", "scan_progress": 0.40},
        },
        "objects": [
            {"id": "object_249", "frame_id": 249, "measurement_group": "paired_object", "polygon": [[0, 0], [1, 0], [1, 1]]},
            {"id": "object_257", "frame_id": 257, "measurement_group": "paired_object", "polygon": [[0, 0], [1, 0], [1, 1]]},
        ],
        "lines": [
            {"id": "line_249", "frame_id": 249, "measurement_group": "paired_line", "points": [[0, 0], [1, 0]]},
            {"id": "line_257", "frame_id": 257, "measurement_group": "paired_line", "points": [[0, 0], [1, 0]]},
        ],
        "safe_background": [
            {"id": "safe_249", "frame_id": 249, "measurement_group": "paired_safe", "polygon": [[0, 0], [1, 0], [1, 1]]},
            {"id": "safe_257", "frame_id": 257, "measurement_group": "paired_safe", "polygon": [[0, 0], [1, 0], [1, 1]]},
        ],
    }
    height, width = 16, 24
    image = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.line(image, (2, 4), (10, 4), (255, 255, 255), 1)
    cv2.line(image, (13, 10), (21, 10), (255, 255, 255), 1)
    owner = np.full((height, width), 8, dtype=np.int32)
    # Each object mask is owned by 7 while its perimeter is owned by 8.  The
    # old `owner_boundaries(owner) & mask` logic incorrectly treated that
    # perimeter transition as an internal object seam.
    object_249 = np.zeros((height, width), dtype=bool)
    object_249[2:6, 2:6] = True
    object_257 = np.zeros((height, width), dtype=bool)
    object_257[8:12, 14:18] = True
    owner[object_249 | object_257] = 7
    safe = np.zeros((height, width), dtype=bool)
    safe[2:14, 10:14] = True
    path = tmp_path / "projection.json"
    np.savez_compressed(
        tmp_path / "projection_masks.npz",
        object_249=object_249, object_257=object_257, safe_249=safe,
    )
    payload = {
        "schema": PANORAMA_PROJECTION_SCHEMA,
        "measurement_only": True,
        "panorama_shape": [height, width],
        "mask_artifact": "projection_masks.npz",
        "objects": [
            {"id": "object_249", "frame_id": 249, "measurement_group": "paired_object", "mask_key": "object_249"},
            {"id": "object_257", "frame_id": 257, "measurement_group": "paired_object", "mask_key": "object_257"},
        ],
        "lines": [
            {"id": "line_249", "frame_id": 249, "measurement_group": "paired_line", "points": [[2, 4], [10, 4]]},
            {"id": "line_257", "frame_id": 257, "measurement_group": "paired_line", "points": [[13, 10], [21, 10]]},
        ],
        "safe_background": [
            {"id": "safe_249", "frame_id": 249, "measurement_group": "paired_safe", "mask_key": "safe_249"},
            # 257 is deliberately absent: a group must aggregate visible
            # members only, not infer a projection from its owner label.
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    projection = load_panorama_annotation_projection(path, annotations=annotations, panorama_shape=(height, width))
    result = evaluate_offline_visual_annotations(image, owner, annotations=annotations, projection=projection)

    assert set(result["object_integrity"]) == {"paired_object"}
    object_metrics = result["object_integrity"]["paired_object"]
    assert object_metrics["projected_member_count"] == 2
    assert object_metrics["object_internal_seam_count"] == 0
    assert object_metrics["object_internal_seam_pixel_count"] == 0
    assert object_metrics["hard_gate_pass"] is True
    assert set(result["line_continuity"]) == {"paired_line"}
    assert result["line_continuity"]["paired_line"]["edge_sample_count"] >= 6
    assert result["line_continuity"]["paired_line"]["line_step_p95_px"] == 0.0
    assert set(result["safe_background"]) == {"paired_safe"}
    assert result["safe_background"]["paired_safe"]["projected_member_count"] == 1


def test_delivery_evaluator_and_sidecar_do_not_modify_primary_artifacts(tmp_path):
    image, owner = _panorama_and_owner()
    assert cv2.imwrite(str(tmp_path / "video_panorama.png"), image)
    np.savez_compressed(tmp_path / "video_pixel_provenance.npz", owner_frame_id=owner)
    annotations_path = tmp_path / "annotations.json"
    annotations_path.write_text(json.dumps(_annotations()), encoding="utf-8")
    image_before = (tmp_path / "video_panorama.png").read_bytes()
    owner_before = (tmp_path / "video_pixel_provenance.npz").read_bytes()
    result = evaluate_delivery_artifacts(tmp_path, annotations_path=annotations_path)
    sidecar = write_offline_evaluation(tmp_path / "video_visual_evaluation.json", result)
    assert sidecar.is_file()
    assert (tmp_path / "video_panorama.png").read_bytes() == image_before
    assert (tmp_path / "video_pixel_provenance.npz").read_bytes() == owner_before
