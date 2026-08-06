from __future__ import annotations

import numpy as np

from panorama_demo.video_candidate_annotation_projection import (
    CandidateInverseMapSource,
    build_candidate_annotation_projection,
    build_v2_c1_calibrated_inverse_sources,
    write_candidate_annotation_projection_sidecar,
)
from panorama_demo.video_visual_renderer_v2 import CudaSourceStrip
from panorama_demo.video_offline_evaluation import (
    evaluate_offline_visual_annotations,
    load_panorama_annotation_projection,
)


def _annotations() -> dict[str, object]:
    return {
        "schema": "gemini305-video-source-annotations/v1",
        "source_frames": {"7": {"color_path": "color/00000007.jpg", "scan_progress": 0.0}},
        "objects": [{"id": "box", "frame_id": 7, "polygon": [[2, 2], [6, 2], [6, 5], [2, 5]]}],
        "lines": [{"id": "edge", "frame_id": 7, "points": [[1, 4], [8, 4]]}],
        "safe_background": [{"id": "wall", "frame_id": 7, "polygon": [[0, 0], [2, 0], [2, 7], [0, 7]]}],
    }


def test_candidate_projection_uses_inverse_maps_and_final_owner_without_rgb(tmp_path):
    height, width = 8, 10
    source = CandidateInverseMapSource(
        frame_id=7,
        canvas_x0=2,
        source_map_x=np.tile(np.arange(width, dtype=np.float32), (height, 1)),
        source_map_y=np.tile(np.arange(height, dtype=np.float32)[:, None], (1, width)),
        valid_mask=np.ones((height, width), dtype=bool),
        raw_shape=(height, width),
    )
    owner = np.full((height, width), 7, dtype=np.int32)
    owner[2:6, 2:5] = 9  # final provenance must remove only part of the raw polygon.
    annotations = _annotations()
    annotations["objects"][0]["measurement_group"] = "box_pair"  # type: ignore[index]
    annotations["lines"][0]["measurement_group"] = "edge_pair"  # type: ignore[index]
    annotations["safe_background"][0]["measurement_group"] = "wall_pair"  # type: ignore[index]
    payload, masks = build_candidate_annotation_projection(
        annotations, sources=[source], final_owner_frame_id=owner,
        crop_xywh=(2, 0, width, height), horizontal_flip=False,
    )
    assert payload["measurement_only"] is True
    assert payload["projection_method"] == "candidate_calibrated_inverse_map_owner_filtered"
    assert payload["objects"][0]["measurement_group"] == "box_pair"
    assert masks["objects__box"].dtype == bool
    assert not np.any(masks["objects__box"][2:6, 2:5])
    projection_path, mask_path = write_candidate_annotation_projection_sidecar(
        tmp_path / "candidate_annotation_projection.json", payload, masks
    )
    assert projection_path.is_file() and mask_path.is_file()
    loaded = load_panorama_annotation_projection(
        projection_path, annotations=annotations, panorama_shape=(height, width)
    )
    panorama = np.full((height, width, 3), 20, dtype=np.uint8)
    evaluation = evaluate_offline_visual_annotations(panorama, owner, annotations=annotations, projection=loaded)
    assert evaluation["object_integrity"]["box_pair"]["valid_pixel_count"] == int(masks["objects__box"].sum())


def test_candidate_projection_omits_unavailable_annotated_source():
    source = CandidateInverseMapSource(
        frame_id=8, canvas_x0=0,
        source_map_x=np.zeros((2, 2), dtype=np.float32), source_map_y=np.zeros((2, 2), dtype=np.float32),
        valid_mask=np.ones((2, 2), dtype=bool), raw_shape=(2, 2),
    )
    payload, masks = build_candidate_annotation_projection(
        _annotations(), sources=[source], final_owner_frame_id=np.full((2, 2), 8, dtype=np.int32),
        crop_xywh=(0, 0, 2, 2), horizontal_flip=False,
    )
    assert not masks
    assert {entry["reason"] for entry in payload["omitted"]} == {"annotated_source_not_a_candidate_render_source"}


def test_v2_c1_projection_uses_actual_c1_windows_and_calibrated_grid_not_owner_geometry():
    """The C1 source map reaches its actual corridor, not merely its owner strip."""

    height, canvas_width = 8, 40
    strips = (
        CudaSourceStrip(7, 0, 0, 20, 20.0),
        CudaSourceStrip(8, 20, 10, 20, 30.0),
    )
    maps = build_v2_c1_calibrated_inverse_sources(
        strips=strips,
        source_shapes={7: (height, 80), 8: (height, 80)},
        canvas_shape=(height, canvas_width),
        calibration={"fx": 100.0, "fy": 100.0, "cx": 20.0, "cy": 3.5, "distortion": ()},
        annotation_frame_ids=(8,),
        corridor_width_pixels=8,
    )

    assert len(maps) == 1
    source = maps[0]
    # C1's 8px seam corridor is [16, 24), so the second source has an
    # actual rendered grid window [16, 40), not just its [20, 40) base strip.
    assert source.canvas_x0 == 16
    assert source.source_map_x.shape == (height, 24)
    # At global canvas x=20, source 8's C1 grid maps to raw x=10.  This
    # checks the source-centre target convention rather than owner geometry.
    assert source.source_map_x[4, 4] == 10.0
    assert source.source_map_y[4, 4] == 4.0

    annotations = {
        "objects": [{"id": "box", "frame_id": 8, "polygon": [[10, 2], [14, 2], [14, 5], [10, 5]]}],
        "lines": [{"id": "edge", "frame_id": 8, "points": [[10, 4], [14, 4]]}],
        "safe_background": [{"id": "wall", "frame_id": 8, "polygon": [[10, 0], [11, 0], [11, 7], [10, 7]]}],
    }
    owner = np.full((height, canvas_width), 7, dtype=np.int32)
    owner[:, 20:25] = 8
    owner[2:6, 22] = 7
    payload, masks = build_candidate_annotation_projection(
        annotations,
        sources=maps,
        final_owner_frame_id=owner,
        crop_xywh=(0, 0, canvas_width, height),
        horizontal_flip=False,
    )
    assert payload["projection_method"] == "candidate_calibrated_inverse_map_owner_filtered"
    # Source 8's raw polygon projects using the C1 grid; final ownership
    # removes the one explicitly foreign-provenance column.
    assert masks["objects__box"].shape == owner.shape
    assert not np.any(masks["objects__box"][2:6, 22])
    assert np.any(masks["objects__box"][:, 20:25])
