"""Dense RGB-D side-scan fusion for the ORB-SLAM3 trajectory path.

The Gemini 305 can return no depth on specular or low-texture parts of an
otherwise well-exposed colour image.  Projecting each depth pixel as a single
point therefore leaves black holes even when camera tracking is excellent.
This module uses two complementary, metric layers:

* a TSDF built from every real tracked RGB-D pose preserves the geometry and
  colour of foreground surfaces; and
* calibrated colour is reprojected to the dominant TSDF background plane to
  provide dense texture where the sensor reported no usable depth.

The plane is estimated from the TSDF mesh, never from a fabricated pose.  The
TSDF foreground is then composited back over the planar background only where
its measured surface departs from that plane.  This is suited to a side scan
of a mostly planar wall with nearby objects such as equipment or vegetation.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from .rgbd_projection import PinholeIntrinsics, ProjectionCanvas, RGBDProjectionFrame
from .session import RGBDFrame, read_aligned_depth_mm


@dataclass(frozen=True)
class DenseFusionConfig:
    """Bounded settings for TSDF geometry and dense planar colour recovery."""

    voxel_length_mm: float = 10.0
    sdf_truncation_mm: float = 40.0
    maximum_depth_mm: float = 2000.0
    plane_histogram_bin_mm: float = 20.0
    foreground_offset_mm: float = 35.0
    ray_chunk_rows: int = 128
    foreground_overlay: bool = True
    minimum_crop_coverage: float = 0.85
    minimum_row_or_column_coverage: float = 0.50

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any] | None = None
    ) -> "DenseFusionConfig":
        if value is None:
            return cls()
        payload = dict(value)
        known = set(cls.__dataclass_fields__)
        unknown = sorted(set(payload) - known)
        if unknown:
            raise ValueError(f"Unknown dense_tsdf configuration keys: {unknown}")
        config = cls(**payload)
        positive = (
            ("voxel_length_mm", config.voxel_length_mm),
            ("sdf_truncation_mm", config.sdf_truncation_mm),
            ("maximum_depth_mm", config.maximum_depth_mm),
            ("plane_histogram_bin_mm", config.plane_histogram_bin_mm),
            ("foreground_offset_mm", config.foreground_offset_mm),
        )
        if any(not math.isfinite(value) or value <= 0.0 for _, value in positive):
            raise ValueError("Dense TSDF distances must be finite positive millimetres")
        if config.sdf_truncation_mm < config.voxel_length_mm:
            raise ValueError("Dense TSDF sdf_truncation_mm must cover one voxel")
        if config.ray_chunk_rows < 1:
            raise ValueError("Dense TSDF ray_chunk_rows must be positive")
        if not 0.0 < config.minimum_crop_coverage <= 1.0:
            raise ValueError("Dense TSDF minimum_crop_coverage must be in (0, 1]")
        if not 0.0 < config.minimum_row_or_column_coverage <= 1.0:
            raise ValueError(
                "Dense TSDF minimum_row_or_column_coverage must be in (0, 1]"
            )
        return config


@dataclass(frozen=True)
class DenseFusionResult:
    image: np.ndarray
    valid_mask: np.ndarray
    metadata: dict[str, object]


def _require_open3d() -> Any:
    try:
        import open3d as o3d
    except ImportError as exc:  # pragma: no cover - depends on local optional runtime
        raise RuntimeError("Dense TSDF fusion requires Open3D 0.19") from exc
    return o3d


def _undistortion_maps(
    intrinsics: PinholeIntrinsics,
) -> tuple[np.ndarray, np.ndarray] | None:
    distortion = np.asarray(intrinsics.distortion, dtype=np.float64)
    if distortion.size == 0 or not np.any(distortion):
        return None
    return cv2.initUndistortRectifyMap(
        intrinsics.matrix,
        distortion,
        None,
        intrinsics.matrix,
        (intrinsics.width, intrinsics.height),
        cv2.CV_32FC1,
    )


def _undistort(
    color: np.ndarray,
    depth_mm: np.ndarray,
    maps: tuple[np.ndarray, np.ndarray] | None,
) -> tuple[np.ndarray, np.ndarray]:
    if maps is None:
        return color, np.asarray(depth_mm, dtype=np.float32)
    map_x, map_y = maps
    return (
        cv2.remap(
            color,
            map_x,
            map_y,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        ),
        cv2.remap(
            np.asarray(depth_mm, dtype=np.float32),
            map_x,
            map_y,
            cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        ),
    )


def _decode_color(path: Path) -> np.ndarray:
    encoded = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR) if encoded.size else None
    if image is None:
        raise OSError(f"Could not decode RGB-D colour image: {path}")
    return image


def _open3d_intrinsics(o3d: Any, intrinsics: PinholeIntrinsics) -> Any:
    return o3d.camera.PinholeCameraIntrinsic(
        intrinsics.width,
        intrinsics.height,
        intrinsics.fx,
        intrinsics.fy,
        intrinsics.cx,
        intrinsics.cy,
    )


def _camera_to_world_m(camera_to_world_mm: np.ndarray) -> np.ndarray:
    pose = np.asarray(camera_to_world_mm, dtype=np.float64).copy()
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError("Dense TSDF pose must be a finite 4x4 camera_to_world matrix")
    pose[:3, 3] /= 1000.0
    return pose


def _integrate_tsdf(
    all_frames: Sequence[RGBDFrame],
    all_poses: Sequence[np.ndarray],
    intrinsics: PinholeIntrinsics,
    maps: tuple[np.ndarray, np.ndarray] | None,
    config: DenseFusionConfig,
) -> Any:
    if len(all_frames) != len(all_poses) or not all_frames:
        raise ValueError("Dense TSDF inputs must contain aligned frames and poses")
    o3d = _require_open3d()
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=config.voxel_length_mm / 1000.0,
        sdf_trunc=config.sdf_truncation_mm / 1000.0,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )
    intrinsic = _open3d_intrinsics(o3d, intrinsics)
    for frame, pose in zip(all_frames, all_poses, strict=True):
        color = _decode_color(frame.color_path)
        depth = read_aligned_depth_mm(frame)
        color, depth = _undistort(color, depth, maps)
        bounded_depth = np.where(
            np.isfinite(depth) & (depth > 0.0) & (depth <= config.maximum_depth_mm),
            depth,
            0.0,
        ).astype(np.uint16)
        if not np.any(bounded_depth):
            continue
        rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(rgb),
            o3d.geometry.Image(bounded_depth),
            depth_scale=1000.0,
            depth_trunc=config.maximum_depth_mm / 1000.0,
            convert_rgb_to_intensity=False,
        )
        volume.integrate(rgbd, intrinsic, np.linalg.inv(_camera_to_world_m(pose)))
    mesh = volume.extract_triangle_mesh()
    if len(mesh.vertices) < 3 or len(mesh.triangles) < 1:
        raise RuntimeError("Dense TSDF fusion did not reconstruct a mesh")
    return mesh


def _dominant_background_plane_mm(
    vertices_mm: np.ndarray,
    normal_axis: np.ndarray,
    bin_width_mm: float,
) -> float:
    normal_depth = vertices_mm @ normal_axis
    finite = normal_depth[np.isfinite(normal_depth)]
    if finite.size < 3:
        raise RuntimeError("Dense TSDF mesh has no finite world-normal depth")
    low, high = np.percentile(finite, (1.0, 99.0))
    edges = np.arange(low, high + bin_width_mm, bin_width_mm, dtype=np.float64)
    if edges.size < 2:
        return float(np.median(finite))
    counts, edges = np.histogram(finite, bins=edges)
    index = int(np.argmax(counts))
    selected = finite[(finite >= edges[index]) & (finite < edges[index + 1])]
    return float(np.median(selected)) if selected.size else float(np.median(finite))


def _dense_plane_background(
    frames: Sequence[RGBDProjectionFrame],
    intrinsics: PinholeIntrinsics,
    canvas: ProjectionCanvas,
    plane_depth_mm: float,
    maps: tuple[np.ndarray, np.ndarray] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Texture the dominant TSDF plane with the nearest real camera at each pixel."""

    if not frames:
        raise ValueError("Dense plane texture requires at least one source")
    height, width = canvas.height, canvas.width
    image = np.zeros((height, width, 3), dtype=np.uint8)
    valid = np.zeros((height, width), dtype=bool)
    owner_distance = np.full((height, width), np.inf, dtype=np.float32)
    scan_axis = np.asarray(canvas.scan_axis, dtype=np.float64)
    down_axis = -np.asarray(canvas.up_axis, dtype=np.float64)
    normal_axis = np.asarray(canvas.normal_axis, dtype=np.float64)
    min_scan, min_down, _, _ = canvas.world_bounds
    scan_values = min_scan + (
        np.arange(width, dtype=np.float64) + 0.5
    ) / canvas.pixels_per_mm

    for frame in frames:
        color, _ = _undistort(frame.rgb, frame.depth_mm, maps)
        # ``cv2.undistort`` keeps the target image size, filling the parts of
        # that rectangle which map outside the calibrated source image with
        # black.  Those pixels are not scene content; if they are allowed to
        # own the plane texture they create long black wedges in an otherwise
        # valid panorama.  Carry an explicit validity mask through the same
        # remapping operation before selecting this source.
        if maps is None:
            color_valid = np.ones(color.shape[:2], dtype=np.uint8)
        else:
            color_valid = cv2.remap(
                np.full(frame.rgb.shape[:2], 255, dtype=np.uint8),
                maps[0],
                maps[1],
                cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
        pose = np.asarray(frame.camera_to_world, dtype=np.float64)
        rotation = pose[:3, :3]
        translation = pose[:3, 3]
        camera_scan_position = float(np.dot(translation, scan_axis))
        for row_start in range(0, height, 128):
            row_stop = min(height, row_start + 128)
            down_values = min_down + (
                np.arange(row_start, row_stop, dtype=np.float64) + 0.5
            ) / canvas.pixels_per_mm
            scan_grid, down_grid = np.meshgrid(
                scan_values, down_values, indexing="xy"
            )
            world = (
                scan_grid[..., None] * scan_axis
                + down_grid[..., None] * down_axis
                + plane_depth_mm * normal_axis
            )
            # The project convention is row-vector points: world = camera R^T+t.
            camera = (world - translation) @ rotation
            z = camera[..., 2]
            safe_z = np.where(z > 1e-6, z, 1.0)
            u = intrinsics.fx * camera[..., 0] / safe_z + intrinsics.cx
            v = intrinsics.fy * camera[..., 1] / safe_z + intrinsics.cy
            in_frame = (
                (z > 1e-6)
                & (u >= 0.0)
                & (u < intrinsics.width - 1)
                & (v >= 0.0)
                & (v < intrinsics.height - 1)
            )
            warped = cv2.remap(
                color,
                u.astype(np.float32),
                v.astype(np.float32),
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )
            sampled_source_valid = cv2.remap(
                color_valid,
                u.astype(np.float32),
                v.astype(np.float32),
                cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            ) > 0
            distance = np.abs(scan_grid - camera_scan_position).astype(np.float32)
            best = owner_distance[row_start:row_stop]
            replace = in_frame & sampled_source_valid & (distance < best)
            image_slice = image[row_start:row_stop]
            image_slice[replace] = warped[replace]
            best[replace] = distance[replace]
            valid[row_start:row_stop][replace] = True
    return image, valid, owner_distance


def _tsdf_foreground_overlay(
    mesh: Any,
    canvas: ProjectionCanvas,
    plane_depth_mm: float,
    config: DenseFusionConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Raycast the fused mesh orthographically and keep only non-plane surfaces."""

    o3d = _require_open3d()
    vertices_m = np.asarray(mesh.vertices, dtype=np.float64)
    triangles = np.asarray(mesh.triangles, dtype=np.int64)
    colors = np.asarray(mesh.vertex_colors, dtype=np.float64)
    if colors.shape != vertices_m.shape:
        raise RuntimeError("Dense TSDF mesh has no per-vertex colour")
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    height, width = canvas.height, canvas.width
    output = np.zeros((height, width, 3), dtype=np.uint8)
    valid = np.zeros((height, width), dtype=bool)
    depth_mm = np.zeros((height, width), dtype=np.float32)
    scan_axis = np.asarray(canvas.scan_axis, dtype=np.float64)
    down_axis = -np.asarray(canvas.up_axis, dtype=np.float64)
    normal_axis = np.asarray(canvas.normal_axis, dtype=np.float64)
    vertices_normal_mm = (vertices_m * 1000.0) @ normal_axis
    origin_normal_m = float(vertices_normal_mm.max() + 50.0) / 1000.0
    min_scan, min_down, _, _ = canvas.world_bounds
    x_values = min_scan + (
        np.arange(width, dtype=np.float64) + 0.5
    ) / canvas.pixels_per_mm
    invalid_primitive = np.iinfo(np.uint32).max
    direction = (-normal_axis).astype(np.float32)

    for row_start in range(0, height, config.ray_chunk_rows):
        row_stop = min(height, row_start + config.ray_chunk_rows)
        y_values = min_down + (
            np.arange(row_start, row_stop, dtype=np.float64) + 0.5
        ) / canvas.pixels_per_mm
        scan_grid, down_grid = np.meshgrid(x_values, y_values, indexing="xy")
        origins = (
            scan_grid[..., None] * scan_axis
            + down_grid[..., None] * down_axis
            + origin_normal_m * normal_axis
        ).astype(np.float32)
        rays = np.concatenate(
            (origins, np.broadcast_to(direction, origins.shape)), axis=-1
        ).reshape(-1, 6)
        answers = scene.cast_rays(o3d.core.Tensor(rays))
        primitive_ids = answers["primitive_ids"].numpy()
        hit = primitive_ids != invalid_primitive
        if not np.any(hit):
            continue
        uv = answers["primitive_uvs"].numpy()
        triangles_for_hits = triangles[primitive_ids[hit]]
        u = uv[hit, 0:1]
        v = uv[hit, 1:2]
        rgb = (
            colors[triangles_for_hits[:, 0]] * (1.0 - u - v)
            + colors[triangles_for_hits[:, 1]] * u
            + colors[triangles_for_hits[:, 2]] * v
        )
        hit_depth = origin_normal_m * 1000.0 - (
            answers["t_hit"].numpy()[hit] * 1000.0
        )
        foreground = np.abs(hit_depth - plane_depth_mm) >= config.foreground_offset_mm
        local_rgb = np.zeros((rays.shape[0], 3), dtype=np.uint8)
        local_valid = np.zeros(rays.shape[0], dtype=bool)
        local_depth = np.zeros(rays.shape[0], dtype=np.float32)
        hit_indices = np.flatnonzero(hit)
        selected = hit_indices[foreground]
        local_rgb[selected] = np.clip(rgb[foreground] * 255.0, 0.0, 255.0).astype(
            np.uint8
        )
        local_valid[selected] = True
        local_depth[selected] = hit_depth[foreground].astype(np.float32)
        output[row_start:row_stop] = cv2.cvtColor(
            local_rgb.reshape(row_stop - row_start, width, 3), cv2.COLOR_RGB2BGR
        )
        valid[row_start:row_stop] = local_valid.reshape(row_stop - row_start, width)
        depth_mm[row_start:row_stop] = local_depth.reshape(row_stop - row_start, width)
    return output, valid, depth_mm


def _crop_dense_result(
    image: np.ndarray,
    valid: np.ndarray,
    *,
    minimum_row_or_column_coverage: float,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    if not np.any(valid):
        raise RuntimeError("Dense RGB-D fusion produced no textured background")
    # A single valid point at the top/bottom of a warped source should not make
    # a long black strip part of the delivered image.  Camera-field-of-view
    # limits can still be marked valid geometrically, so use non-black texture
    # only to locate the outer crop.  This does not reclassify dark scene
    # content (for example a cable or a hose) in the delivered valid mask.
    textured = valid & (np.max(image, axis=2) > 8)
    rows = np.flatnonzero(
        textured.mean(axis=1) >= minimum_row_or_column_coverage
    )
    columns = np.flatnonzero(
        textured.mean(axis=0) >= minimum_row_or_column_coverage
    )
    if rows.size and columns.size:
        x = int(columns[0])
        y = int(rows[0])
        width = int(columns[-1] - columns[0] + 1)
        height = int(rows[-1] - rows[0] + 1)
    else:
        x, y, width, height = cv2.boundingRect(valid.astype(np.uint8))
    if width <= 0 or height <= 0:
        raise RuntimeError("Dense RGB-D fusion produced an empty crop")
    return (
        image[y : y + height, x : x + width],
        valid[y : y + height, x : x + width],
        (x, y, width, height),
    )


def fuse_dense_rgbd_side_scan(
    background_frames: Sequence[RGBDProjectionFrame],
    all_depth_frames: Sequence[RGBDFrame],
    all_camera_to_world: Sequence[np.ndarray],
    intrinsics: PinholeIntrinsics,
    canvas: ProjectionCanvas,
    *,
    config: DenseFusionConfig | Mapping[str, Any] | None = None,
) -> DenseFusionResult:
    """Build a dense wall/foreground panorama from RGB-D frames and real poses."""

    selected_config = (
        config if isinstance(config, DenseFusionConfig) else DenseFusionConfig.from_mapping(config)
    )
    maps = _undistortion_maps(intrinsics)
    mesh = _integrate_tsdf(
        all_depth_frames,
        all_camera_to_world,
        intrinsics,
        maps,
        selected_config,
    )
    vertices_mm = np.asarray(mesh.vertices, dtype=np.float64) * 1000.0
    plane_depth_mm = _dominant_background_plane_mm(
        vertices_mm,
        np.asarray(canvas.normal_axis, dtype=np.float64),
        selected_config.plane_histogram_bin_mm,
    )
    background, background_valid, _ = _dense_plane_background(
        background_frames,
        intrinsics,
        canvas,
        plane_depth_mm,
        maps,
    )
    foreground_pixels = 0
    if selected_config.foreground_overlay:
        foreground, foreground_valid, _ = _tsdf_foreground_overlay(
            mesh, canvas, plane_depth_mm, selected_config
        )
        background[foreground_valid] = foreground[foreground_valid]
        foreground_pixels = int(np.count_nonzero(foreground_valid))
    image, crop_valid, crop = _crop_dense_result(
        background,
        background_valid,
        minimum_row_or_column_coverage=selected_config.minimum_row_or_column_coverage,
    )
    coverage = float(np.mean(crop_valid))
    metadata: dict[str, object] = {
        "backend": "tsdf_plane_dense_rgbd",
        "config": asdict(selected_config),
        "canvas_width": canvas.width,
        "canvas_height": canvas.height,
        "crop": {"x": crop[0], "y": crop[1], "width": crop[2], "height": crop[3]},
        "background_plane_normal_depth_mm": plane_depth_mm,
        "tsdf_vertex_count": int(len(mesh.vertices)),
        "tsdf_triangle_count": int(len(mesh.triangles)),
        "background_valid_pixel_count": int(np.count_nonzero(background_valid)),
        "foreground_overlay_pixel_count": foreground_pixels,
        "crop_valid_coverage": coverage,
        "quality_metrics": {
            "quality_pass": coverage >= selected_config.minimum_crop_coverage,
            "crop_valid_coverage": coverage,
            "minimum_crop_coverage": selected_config.minimum_crop_coverage,
            "foreground_overlay_pixel_count": foreground_pixels,
        },
    }
    return DenseFusionResult(image=image, valid_mask=crop_valid, metadata=metadata)
