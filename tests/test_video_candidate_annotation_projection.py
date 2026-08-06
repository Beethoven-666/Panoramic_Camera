from __future__ import annotations

import numpy as np

from panorama_demo.video_candidate_annotation_projection import (
    CandidateInverseMapSource,
    apply_final_grid_updates,
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


def test_candidate_projection_uses_inverse_maps_without_owner_filtering(tmp_path):
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
    owner[2:6, 2:5] = 9  # Must not erase a source measurement projection.
    annotations = _annotations()
    annotations["objects"][0]["measurement_group"] = "box_pair"  # type: ignore[index]
    annotations["lines"][0]["measurement_group"] = "edge_pair"  # type: ignore[index]
    annotations["safe_background"][0]["measurement_group"] = "wall_pair"  # type: ignore[index]
    payload, masks = build_candidate_annotation_projection(
        annotations, sources=[source], final_owner_frame_id=owner,
        crop_xywh=(2, 0, width, height), horizontal_flip=False,
    )
    assert payload["measurement_only"] is True
    assert payload["projection_method"] == "candidate_calibrated_inverse_map_owner_independent_consensus"
    assert payload["objects"][0]["measurement_group"] == "box_pair"
    assert masks["objects__consensus__box_pair"].dtype == bool
    assert np.any(masks["objects__consensus__box_pair"][2:6, 2:5])
    projection_path, mask_path = write_candidate_annotation_projection_sidecar(
        tmp_path / "candidate_annotation_projection.json", payload, masks
    )
    assert projection_path.is_file() and mask_path.is_file()
    loaded = load_panorama_annotation_projection(
        projection_path, annotations=annotations, panorama_shape=(height, width)
    )
    panorama = np.full((height, width, 3), 20, dtype=np.uint8)
    evaluation = evaluate_offline_visual_annotations(panorama, owner, annotations=annotations, projection=loaded)
    assert evaluation["object_integrity"]["box_pair"]["valid_pixel_count"] == int(masks["objects__consensus__box_pair"].sum())


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


def test_final_grid_update_changes_only_the_actual_mesh_output_domain_without_owner_filtering():
    """C2--C4 evidence must use final inverse samples, never nominal C1 maps."""

    source = CandidateInverseMapSource(
        frame_id=7,
        canvas_x0=10,
        source_map_x=np.tile(np.arange(6, dtype=np.float64), (4, 1)),
        source_map_y=np.tile(np.arange(4, dtype=np.float64)[:, None], (1, 6)),
        valid_mask=np.ones((4, 6), dtype=bool),
        raw_shape=(4, 6),
    )
    normalized = np.zeros((4, 3, 2), dtype=np.float32)
    normalized[..., 0] = -0.2  # raw x=2 for a six-pixel source
    normalized[..., 1] = -1.0
    applied = np.zeros((4, 3), dtype=bool)
    applied[1, 1] = True
    result = apply_final_grid_updates(
        [source],
        [{
            "frame_id": 7,
            "canvas_x0": 12,
            "normalized_grid_xy": normalized,
            "applied_mask": applied,
            "source_shape": [4, 6],
        }],
    )
    updated = result[0]
    assert np.isclose(updated.source_map_x[1, 3], 2.0)
    assert np.isclose(updated.source_map_y[1, 3], 0.0)
    # An unapplied mesh cell retains the calibrated C1 map exactly.
    assert updated.source_map_x[1, 2] == source.source_map_x[1, 2]
    assert bool(updated.valid_mask[1, 3])


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
    assert payload["projection_method"] == "candidate_calibrated_inverse_map_owner_independent_consensus"
    # Source 8's raw polygon projects using the C1 grid. Foreign provenance
    # remains measurable; it is precisely what the object gate later audits.
    assert masks["objects__consensus__box"].shape == owner.shape
    assert np.any(masks["objects__consensus__box"][2:6, 22])
    assert np.any(masks["objects__consensus__box"][:, 20:25])


def test_v2_projection_consensus_survives_changed_owner_and_records_source_masks(tmp_path):
    """A two-source object/background measurement is independent of ownership."""

    height, width = 16, 24
    yy, xx = np.indices((height, width), dtype=np.float32)
    first = CandidateInverseMapSource(7, 0, xx, yy, np.ones_like(xx, dtype=bool), (height, width))
    # The second source's physical object is translated two raw pixels; its
    # final inverse grid correctly lands it on the same panorama structure.
    second = CandidateInverseMapSource(8, 0, xx - 2.0, yy, np.ones_like(xx, dtype=bool), (height, width))
    annotations = {
        "objects": [
            {"id": "box_7", "frame_id": 7, "measurement_group": "box", "polygon": [[6, 4], [11, 4], [11, 10], [6, 10]]},
            {"id": "box_8", "frame_id": 8, "measurement_group": "box", "polygon": [[4, 4], [9, 4], [9, 10], [4, 10]]},
        ],
        "lines": [],
        "safe_background": [
            {"id": "wall_7", "frame_id": 7, "measurement_group": "wall", "polygon": [[1, 1], [4, 1], [4, 14], [1, 14]]},
            {"id": "wall_8", "frame_id": 8, "measurement_group": "wall", "polygon": [[-1, 1], [2, 1], [2, 14], [-1, 14]]},
        ],
    }
    owner = np.full((height, width), 99, dtype=np.int32)  # changed after rendering
    payload, masks = build_candidate_annotation_projection(
        annotations, sources=(first, second), final_owner_frame_id=owner,
        crop_xywh=(0, 0, width, height), horizontal_flip=False,
    )
    assert {item["measurement_group"] for item in payload["measurement_groups"]} == {"box", "wall"}
    assert all(item["measurement_state"] == "evaluated" for item in payload["measurement_groups"])
    assert "objects__source_projected__box_7" in masks
    assert "objects__source_projected__box_8" in masks
    assert np.any(masks["objects__consensus__box"])
    projection_path, _ = write_candidate_annotation_projection_sidecar(tmp_path / "projection.json", payload, masks)
    loaded = load_panorama_annotation_projection(projection_path, annotations={"schema": "gemini305-video-source-annotations/v1", "source_frames": {}, **annotations}, panorama_shape=(height, width))
    # The object is evaluated against the actual owner map rather than
    # disappearing merely because its original source no longer owns pixels.
    result = evaluate_offline_visual_annotations(np.zeros((height, width, 3), np.uint8), owner, annotations={"schema": "gemini305-video-source-annotations/v1", "source_frames": {}, **annotations}, projection=loaded)
    assert result["object_integrity"]["box"]["owner_count"] == 1


def test_v2_projection_rejects_inconsistent_pair_and_dense_line_preserves_curvature(tmp_path):
    height, width = 24, 32
    yy, xx = np.indices((height, width), dtype=np.float32)
    curved = CandidateInverseMapSource(7, 0, xx, yy + 1.5 * np.sin(xx / 3.0), np.ones_like(xx, dtype=bool), (height, width))
    far = CandidateInverseMapSource(8, 0, xx - 10.0, yy, np.ones_like(xx, dtype=bool), (height, width))
    annotations = {
        "objects": [
            {"id": "far_7", "frame_id": 7, "measurement_group": "far", "polygon": [[8, 4], [14, 4], [14, 10], [8, 10]]},
            {"id": "far_8", "frame_id": 8, "measurement_group": "far", "polygon": [[8, 4], [14, 4], [14, 10], [8, 10]]},
        ],
        "lines": [{"id": "curve", "frame_id": 7, "points": [[3, 12], [27, 12]]}],
        "safe_background": [],
    }
    payload, masks = build_candidate_annotation_projection(
        annotations, sources=(curved, far), final_owner_frame_id=np.zeros((height, width), dtype=np.int32),
        crop_xywh=(0, 0, width, height), horizontal_flip=False,
    )
    far_group = next(item for item in payload["measurement_groups"] if item["measurement_group"] == "far")
    assert far_group["measurement_state"] == "projection_inconsistent"
    line = payload["lines"][0]["points"]
    assert len(line) > 8
    assert len({round(point[1], 2) for point in line}) > 2
    projection_path, _ = write_candidate_annotation_projection_sidecar(tmp_path / "projection.json", payload, masks)
    source_annotations = {"schema": "gemini305-video-source-annotations/v1", "source_frames": {}, **annotations}
    loaded = load_panorama_annotation_projection(projection_path, annotations=source_annotations, panorama_shape=(height, width))
    result = evaluate_offline_visual_annotations(np.zeros((height, width, 3), np.uint8), np.zeros((height, width), dtype=np.int32), annotations=source_annotations, projection=loaded)
    assert result["object_integrity"]["far"]["reason"] == "projection_inconsistent"
