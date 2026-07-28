from __future__ import annotations

import numpy as np

from panorama_demo.inspection_supervoxel_track import (
    segment_world_supervoxels,
)


def test_supervoxel_lab_gate_separates_touching_objects() -> None:
    first = np.asarray(
        [[x, y, 500.0] for x in (0.0, 10.0) for y in (0.0, 10.0)]
    )
    second = first + np.asarray([20.0, 0.0, 0.0])
    points = np.vstack((first, second))
    lab = np.vstack(
        (
            np.tile((220.0, 128.0, 128.0), (len(first), 1)),
            np.tile((30.0, 128.0, 128.0), (len(second), 1)),
        )
    )
    normals = np.tile((0.0, 0.0, 1.0), (len(points), 1))
    result = segment_world_supervoxels(
        points_world_mm=points,
        lab=lab,
        normals_world=normals,
        normal_valid=np.ones(len(points), dtype=bool),
        voxel_size_mm=10.0,
        remove_structural_planes=False,
    )
    first_tracks = np.unique(result.sample_track_id[: len(first)])
    second_tracks = np.unique(result.sample_track_id[len(first) :])
    assert first_tracks.size == 1
    assert second_tracks.size == 1
    assert first_tracks[0] != second_tracks[0]


def test_supervoxel_normal_boundary_separates_same_colour_surfaces() -> None:
    horizontal = np.asarray(
        [[x, y, 500.0] for x in (0.0, 10.0) for y in (0.0, 10.0)]
    )
    vertical = np.asarray(
        [[20.0, y, z] for y in (0.0, 10.0) for z in (500.0, 510.0)]
    )
    points = np.vstack((horizontal, vertical))
    lab = np.tile((160.0, 128.0, 128.0), (len(points), 1))
    normals = np.vstack(
        (
            np.tile((0.0, 0.0, 1.0), (len(horizontal), 1)),
            np.tile((1.0, 0.0, 0.0), (len(vertical), 1)),
        )
    )
    result = segment_world_supervoxels(
        points_world_mm=points,
        lab=lab,
        normals_world=normals,
        normal_valid=np.ones(len(points), dtype=bool),
        voxel_size_mm=10.0,
        remove_structural_planes=False,
    )
    assert np.unique(
        result.sample_track_id[: len(horizontal)]
    ).item() != np.unique(
        result.sample_track_id[len(horizontal) :]
    ).item()
