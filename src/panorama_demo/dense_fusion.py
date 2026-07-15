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

import json
import math
import struct
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
    foreground_neighbor_depth_tolerance_mm: float = 70.0
    minimum_foreground_support_neighbors: int = 1
    foreground_tsdf_hole_kernel_size: int = 15
    ray_chunk_rows: int = 128
    foreground_overlay: bool = True
    minimum_crop_coverage: float = 0.85
    minimum_row_or_column_coverage: float = 0.98
    minimum_inscribed_crop_height_fraction: float = 0.90
    mask_close_kernel_size: int = 3
    mask_erode_pixels: int = 1

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
            (
                "foreground_neighbor_depth_tolerance_mm",
                config.foreground_neighbor_depth_tolerance_mm,
            ),
        )
        if any(not math.isfinite(value) or value <= 0.0 for _, value in positive):
            raise ValueError("Dense TSDF distances must be finite positive millimetres")
        if config.sdf_truncation_mm < config.voxel_length_mm:
            raise ValueError("Dense TSDF sdf_truncation_mm must cover one voxel")
        if config.ray_chunk_rows < 1:
            raise ValueError("Dense TSDF ray_chunk_rows must be positive")
        if not 1 <= config.minimum_foreground_support_neighbors <= 4:
            raise ValueError(
                "Dense TSDF minimum_foreground_support_neighbors must be in [1, 4]"
            )
        if (
            config.foreground_tsdf_hole_kernel_size < 1
            or config.foreground_tsdf_hole_kernel_size > 31
            or config.foreground_tsdf_hole_kernel_size % 2 == 0
        ):
            raise ValueError(
                "Dense TSDF foreground_tsdf_hole_kernel_size must be odd in [1, 31]"
            )
        if not 0.0 < config.minimum_crop_coverage <= 1.0:
            raise ValueError("Dense TSDF minimum_crop_coverage must be in (0, 1]")
        if not 0.0 < config.minimum_row_or_column_coverage <= 1.0:
            raise ValueError(
                "Dense TSDF minimum_row_or_column_coverage must be in (0, 1]"
            )
        if not 0.0 < config.minimum_inscribed_crop_height_fraction <= 1.0:
            raise ValueError(
                "Dense TSDF minimum_inscribed_crop_height_fraction must be in (0, 1]"
            )
        if config.mask_close_kernel_size not in {1, 3, 5}:
            raise ValueError("Dense TSDF mask_close_kernel_size must be 1, 3, or 5")
        if not 0 <= config.mask_erode_pixels <= 3:
            raise ValueError("Dense TSDF mask_erode_pixels must be in [0, 3]")
        return config


@dataclass(frozen=True)
class DenseFusionResult:
    image: np.ndarray
    valid_mask: np.ndarray
    foreground_mask: np.ndarray
    tsdf_mesh_glb: bytes
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Undistort colour/depth and derive the actual camera-field valid mask."""

    if maps is None:
        return (
            color,
            np.asarray(depth_mm, dtype=np.float32),
            np.ones(depth_mm.shape, dtype=bool),
        )
    map_x, map_y = maps
    source_valid = np.full(color.shape[:2], 255, dtype=np.uint8)
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
        cv2.remap(
            source_valid,
            map_x,
            map_y,
            cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        ) > 0,
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
        color, depth, undistort_valid = _undistort(color, depth, maps)
        bounded_depth = np.where(
            undistort_valid
            & np.isfinite(depth)
            & (depth > 0.0)
            & (depth <= config.maximum_depth_mm),
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


def _mesh_to_glb(mesh: Any) -> bytes:
    """Encode a TSDF mesh with the Open3D RGB-D axes converted to glTF axes."""

    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    triangles = np.asarray(mesh.triangles, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices):
        raise RuntimeError("TSDF mesh has no exportable vertices")
    if triangles.ndim != 2 or triangles.shape[1] != 3 or not len(triangles):
        raise RuntimeError("TSDF mesh has no exportable triangles")
    if not np.isfinite(vertices).all() or np.any(triangles < 0):
        raise RuntimeError("TSDF mesh contains invalid glTF geometry")
    if int(triangles.max()) >= len(vertices):
        raise RuntimeError("TSDF mesh triangle index exceeds vertex count")

    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
    if normals.shape != vertices.shape or not np.isfinite(normals).all():
        raise RuntimeError("TSDF mesh has no finite vertex normals")
    colors = np.asarray(mesh.vertex_colors, dtype=np.float64)
    if colors.shape != vertices.shape or not np.isfinite(colors).all():
        colors = np.ones(vertices.shape, dtype=np.float64)
    colors_u8 = np.clip(np.rint(colors * 255.0), 0, 255).astype(np.uint8)
    index_dtype = np.uint16 if len(vertices) <= np.iinfo(np.uint16).max else np.uint32
    indices = np.asarray(triangles.reshape(-1), dtype=index_dtype)

    binary = bytearray()
    buffer_views: list[dict[str, int]] = []

    def append_view(values: np.ndarray, target: int) -> int:
        while len(binary) % 4:
            binary.append(0)
        offset = len(binary)
        encoded = np.ascontiguousarray(values).tobytes()
        binary.extend(encoded)
        buffer_views.append(
            {"buffer": 0, "byteOffset": offset, "byteLength": len(encoded), "target": target}
        )
        return len(buffer_views) - 1

    position_view = append_view(vertices, 34962)
    normal_view = append_view(normals, 34962)
    color_view = append_view(colors_u8, 34962)
    index_view = append_view(indices, 34963)
    index_component_type = 5123 if index_dtype is np.uint16 else 5125
    accessors: list[dict[str, object]] = [
        {
            "bufferView": position_view,
            "componentType": 5126,
            "count": int(len(vertices)),
            "type": "VEC3",
            "min": vertices.min(axis=0).astype(float).tolist(),
            "max": vertices.max(axis=0).astype(float).tolist(),
        },
        {
            "bufferView": normal_view,
            "componentType": 5126,
            "count": int(len(normals)),
            "type": "VEC3",
        },
        {
            "bufferView": color_view,
            "componentType": 5121,
            "normalized": True,
            "count": int(len(colors_u8)),
            "type": "VEC3",
        },
        {
            "bufferView": index_view,
            "componentType": index_component_type,
            "count": int(indices.size),
            "type": "SCALAR",
        },
    ]
    document: dict[str, object] = {
        "asset": {"version": "2.0", "generator": "gemini305-rgbd-panorama"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        # Open3D RGB-D geometry uses +Y down and +Z forward.  glTF/browser
        # viewers use +Y up and look along -Z.  A 180-degree X rotation is a
        # proper (right-handed) coordinate conversion, unlike a Y reflection.
        "nodes": [
            {
                "mesh": 0,
                "name": "Gemini 305 TSDF mesh",
                "rotation": [1.0, 0.0, 0.0, 0.0],
            }
        ],
        "meshes": [
            {
                "name": "tsdf_mesh",
                "primitives": [
                    {
                        "attributes": {"POSITION": 0, "NORMAL": 1, "COLOR_0": 2},
                        "indices": 3,
                        "material": 0,
                        "mode": 4,
                    }
                ],
            }
        ],
        "materials": [
            {
                "pbrMetallicRoughness": {
                    "baseColorFactor": [1.0, 1.0, 1.0, 1.0],
                    "metallicFactor": 0.0,
                    "roughnessFactor": 1.0,
                },
                "doubleSided": True,
            }
        ],
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": buffer_views,
        "accessors": accessors,
    }
    json_chunk = json.dumps(document, separators=(",", ":")).encode("utf-8")
    json_chunk += b" " * ((-len(json_chunk)) % 4)
    binary_chunk = bytes(binary)
    binary_chunk += b"\x00" * ((-len(binary_chunk)) % 4)
    total_length = 12 + 8 + len(json_chunk) + 8 + len(binary_chunk)
    return b"".join(
        (
            struct.pack("<4sII", b"glTF", 2, total_length),
            struct.pack("<I4s", len(json_chunk), b"JSON"),
            json_chunk,
            struct.pack("<I4s", len(binary_chunk), b"BIN\x00"),
            binary_chunk,
        )
    )


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
        color, _, color_valid = _undistort(frame.rgb, frame.depth_mm, maps)
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
                color_valid.astype(np.uint8),
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


def _camera_to_wall_direction(
    camera_to_world: Sequence[np.ndarray],
    canvas: ProjectionCanvas,
    plane_depth_mm: float,
) -> float:
    """Return the signed normal direction from the cameras toward the wall."""

    normal_axis = np.asarray(canvas.normal_axis, dtype=np.float64)
    camera_normal = np.asarray(
        [np.asarray(pose, dtype=np.float64)[:3, 3] @ normal_axis for pose in camera_to_world],
        dtype=np.float64,
    )
    if not camera_normal.size or not np.isfinite(camera_normal).all():
        raise ValueError("Dense foreground fusion requires finite camera poses")
    difference = float(plane_depth_mm - np.median(camera_normal))
    if abs(difference) < 1e-3:
        raise RuntimeError("Dense foreground plane coincides with the camera side")
    return 1.0 if difference > 0.0 else -1.0


def _supported_depth_mask(
    depth_mm: np.ndarray,
    valid: np.ndarray,
    *,
    tolerance_mm: float,
    minimum_neighbors: int,
) -> np.ndarray:
    """Reject isolated samples and depth-edge flying points without erasing edges.

    A real foreground contour still has at least one adjacent sample at a
    similar range.  An isolated flying point does not, so this test removes
    the latter while retaining the measured outline of an extinguisher or hose.
    """

    depth = np.asarray(depth_mm, dtype=np.float32)
    supported = np.zeros(depth.shape, dtype=np.uint8)
    padded_depth = np.pad(depth, 1, mode="constant", constant_values=np.nan)
    padded_valid = np.pad(valid, 1, mode="constant", constant_values=False)
    center = depth
    for row_offset, column_offset in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        neighbor = padded_depth[
            1 + row_offset : 1 + row_offset + depth.shape[0],
            1 + column_offset : 1 + column_offset + depth.shape[1],
        ]
        neighbor_valid = padded_valid[
            1 + row_offset : 1 + row_offset + depth.shape[0],
            1 + column_offset : 1 + column_offset + depth.shape[1],
        ]
        supported += (
            neighbor_valid & (np.abs(neighbor - center) <= tolerance_mm)
        ).astype(np.uint8)
    return valid & (supported >= minimum_neighbors)


def _depth_point_foreground_overlay(
    all_frames: Sequence[RGBDFrame],
    all_poses: Sequence[np.ndarray],
    intrinsics: PinholeIntrinsics,
    canvas: ProjectionCanvas,
    plane_depth_mm: float,
    view_direction: float,
    maps: tuple[np.ndarray, np.ndarray] | None,
    config: DenseFusionConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    """Use measured foreground RGB-D points to fill holes in the TSDF layer."""

    if len(all_frames) != len(all_poses) or not all_frames:
        raise ValueError("Dense foreground point fusion requires aligned frames and poses")
    height, width = canvas.height, canvas.width
    output = np.zeros((height, width, 3), dtype=np.uint8)
    valid = np.zeros((height, width), dtype=bool)
    # Larger values are closer to the camera-side virtual observation plane.
    virtual_depth = np.full((height, width), -np.inf, dtype=np.float32)
    normal_axis = np.asarray(canvas.normal_axis, dtype=np.float64)
    support_count = 0
    foreground_count = 0
    projected_count = 0

    for frame, pose in zip(all_frames, all_poses, strict=True):
        color = _decode_color(frame.color_path)
        depth = read_aligned_depth_mm(frame)
        color, depth, undistort_valid = _undistort(color, depth, maps)
        depth_valid = (
            undistort_valid
            & np.isfinite(depth)
            & (depth > 0.0)
            & (depth <= config.maximum_depth_mm)
        )
        supported = _supported_depth_mask(
            depth,
            depth_valid,
            tolerance_mm=config.foreground_neighbor_depth_tolerance_mm,
            minimum_neighbors=config.minimum_foreground_support_neighbors,
        )
        support_count += int(np.count_nonzero(supported))
        rows, columns = np.nonzero(supported)
        if not rows.size:
            continue
        point_depth = depth[rows, columns].astype(np.float64)
        camera_points = np.empty((rows.size, 3), dtype=np.float64)
        camera_points[:, 0] = (
            (columns.astype(np.float64) - intrinsics.cx)
            * point_depth
            / intrinsics.fx
        )
        camera_points[:, 1] = (
            (rows.astype(np.float64) - intrinsics.cy) * point_depth / intrinsics.fy
        )
        camera_points[:, 2] = point_depth
        camera_to_world = np.asarray(pose, dtype=np.float64)
        world_points = (
            camera_points @ camera_to_world[:3, :3].T + camera_to_world[:3, 3]
        )
        normal_depth = world_points @ normal_axis
        foreground = (
            (plane_depth_mm - normal_depth) * view_direction
            >= config.foreground_offset_mm
        )
        foreground_count += int(np.count_nonzero(foreground))
        if not np.any(foreground):
            continue
        world_points = world_points[foreground]
        normal_depth = normal_depth[foreground]
        colours = color[rows[foreground], columns[foreground]]
        canvas_points = canvas.world_to_canvas(world_points)
        x = np.floor(canvas_points[:, 0]).astype(np.int64)
        y = np.floor(canvas_points[:, 1]).astype(np.int64)
        in_canvas = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        if not np.any(in_canvas):
            continue
        x = x[in_canvas]
        y = y[in_canvas]
        colours = colours[in_canvas]
        normal_depth = normal_depth[in_canvas]
        flat = y * width + x
        # For one source frame retain the point closest to the virtual camera
        # at every output pixel; then compare it with the global z-buffer.
        camera_side_depth = -view_direction * normal_depth
        order = np.lexsort((camera_side_depth, flat))
        ordered_flat = flat[order]
        keep = np.empty(order.size, dtype=bool)
        keep[:-1] = ordered_flat[:-1] != ordered_flat[1:]
        keep[-1] = True
        local_winners = order[keep]
        candidate_flat = flat[local_winners]
        candidate_depth = camera_side_depth[local_winners].astype(np.float32)
        zbuffer = virtual_depth.reshape(-1)
        replace = candidate_depth > zbuffer[candidate_flat]
        if not np.any(replace):
            continue
        selected = local_winners[replace]
        selected_flat = flat[selected]
        output.reshape(-1, 3)[selected_flat] = colours[selected]
        virtual_depth.reshape(-1)[selected_flat] = camera_side_depth[selected]
        valid.reshape(-1)[selected_flat] = True
        projected_count += int(selected.size)

    return output, valid, virtual_depth, {
        "supported_depth_point_count": support_count,
        "foreground_candidate_point_count": foreground_count,
        "foreground_zbuffer_pixel_count": int(np.count_nonzero(valid)),
        "foreground_zbuffer_updates": projected_count,
    }


def _tsdf_foreground_overlay(
    mesh: Any,
    canvas: ProjectionCanvas,
    plane_depth_mm: float,
    camera_to_world: Sequence[np.ndarray],
    config: DenseFusionConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Raycast foreground from the real camera side of the dominant wall."""

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
    view_direction = _camera_to_wall_direction(camera_to_world, canvas, plane_depth_mm)
    camera_normal = np.asarray(
        [np.asarray(pose, dtype=np.float64)[:3, 3] @ normal_axis for pose in camera_to_world],
        dtype=np.float64,
    )
    origin_normal_mm = float(
        np.min(camera_normal) - 50.0
        if view_direction > 0.0
        else np.max(camera_normal) + 50.0
    )
    min_scan, min_down, _, _ = canvas.world_bounds
    x_values = min_scan + (
        np.arange(width, dtype=np.float64) + 0.5
    ) / canvas.pixels_per_mm
    invalid_primitive = np.iinfo(np.uint32).max

    for row_start in range(0, height, config.ray_chunk_rows):
        row_stop = min(height, row_start + config.ray_chunk_rows)
        y_values = min_down + (
            np.arange(row_start, row_stop, dtype=np.float64) + 0.5
        ) / canvas.pixels_per_mm
        scan_grid, down_grid = np.meshgrid(x_values, y_values, indexing="xy")
        origins = (
            scan_grid[..., None] * scan_axis
            + down_grid[..., None] * down_axis
            + origin_normal_mm * normal_axis
        ).astype(np.float32) / 1000.0
        rays = np.concatenate(
            (
                origins,
                np.broadcast_to(
                    (view_direction * normal_axis).astype(np.float32), origins.shape
                ),
            ),
            axis=-1,
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
        hit_depth = origin_normal_mm + view_direction * (
            answers["t_hit"].numpy()[hit] * 1000.0
        )
        foreground = (
            (plane_depth_mm - hit_depth) * view_direction
            >= config.foreground_offset_mm
        )
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


def _largest_inscribed_rectangle(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Find the largest axis-aligned rectangle containing only true mask pixels."""

    if mask.ndim != 2 or not np.any(mask):
        return (0, 0, 0, 0)
    height, width = mask.shape
    histogram = np.zeros(width, dtype=np.int32)
    best_area = 0
    best = (0, 0, 0, 0)
    for row_index in range(height):
        histogram = np.where(mask[row_index], histogram + 1, 0)
        stack = [-1]
        for column_index in range(width + 1):
            current_height = int(histogram[column_index]) if column_index < width else 0
            while stack[-1] >= 0 and histogram[stack[-1]] >= current_height:
                top = stack.pop()
                rectangle_height = int(histogram[top])
                left = stack[-1] + 1
                rectangle_width = column_index - left
                area = rectangle_height * rectangle_width
                if area > best_area:
                    best_area = area
                    best = (
                        left,
                        row_index - rectangle_height + 1,
                        rectangle_width,
                        rectangle_height,
                    )
            stack.append(column_index)
    return best


def _longest_contiguous_span(indices: np.ndarray) -> tuple[int, int] | None:
    if not indices.size:
        return None
    split = np.flatnonzero(np.diff(indices) > 1) + 1
    runs = np.split(indices, split)
    selected = max(runs, key=len)
    return int(selected[0]), int(selected[-1])


def _fallback_visible_rectangle(
    mask: np.ndarray, minimum_coverage: float
) -> tuple[int, int, int, int]:
    """Use the documented high-coverage row/column fallback crop."""

    rows = _longest_contiguous_span(
        np.flatnonzero(mask.mean(axis=1) >= minimum_coverage)
    )
    columns = _longest_contiguous_span(
        np.flatnonzero(mask.mean(axis=0) >= minimum_coverage)
    )
    if rows is None or columns is None:
        x, y, width, height = cv2.boundingRect(mask.astype(np.uint8))
        return int(x), int(y), int(width), int(height)
    return (
        columns[0],
        rows[0],
        columns[1] - columns[0] + 1,
        rows[1] - rows[0] + 1,
    )


def _safe_visible_mask(valid: np.ndarray, config: DenseFusionConfig) -> np.ndarray:
    """Close tiny validity holes, then inset the safe visible region by one pixel."""

    binary = np.asarray(valid, dtype=np.uint8) * 255
    if config.mask_close_kernel_size > 1:
        kernel = np.ones(
            (config.mask_close_kernel_size, config.mask_close_kernel_size), dtype=np.uint8
        )
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    if config.mask_erode_pixels:
        binary = cv2.erode(
            binary,
            np.ones((3, 3), dtype=np.uint8),
            iterations=config.mask_erode_pixels,
        )
    return binary > 0


def _tsdf_local_holes(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    """Return only small holes enclosed by TSDF foreground, never its exterior."""

    if kernel_size == 1:
        return np.zeros(mask.shape, dtype=bool)
    binary = np.asarray(mask, dtype=np.uint8) * 255
    closed = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        np.ones((kernel_size, kernel_size), dtype=np.uint8),
    )
    candidates = (closed > 0) & ~np.asarray(mask, dtype=bool)
    component_count, labels = cv2.connectedComponents(candidates.astype(np.uint8), 8)
    if component_count <= 1:
        return candidates
    boundary_labels = np.unique(
        np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1]))
    )
    boundary_labels = boundary_labels[boundary_labels != 0]
    if boundary_labels.size:
        candidates[np.isin(labels, boundary_labels)] = False
    return candidates


def _crop_dense_result(
    image: np.ndarray,
    valid: np.ndarray,
    foreground_mask: np.ndarray,
    *,
    config: DenseFusionConfig,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    tuple[int, int, int, int],
    dict[str, object],
]:
    """Crop exclusively from the true projected visibility mask, never RGB values."""

    if not np.any(valid):
        raise RuntimeError("Dense RGB-D fusion produced no visible panorama pixels")
    safe_visible = _safe_visible_mask(valid, config)
    if not np.any(safe_visible):
        raise RuntimeError("Dense RGB-D fusion has no visible pixels after safe inset")
    bounds = cv2.boundingRect(safe_visible.astype(np.uint8))
    strict_crop = _largest_inscribed_rectangle(safe_visible)
    minimum_height = config.minimum_inscribed_crop_height_fraction * bounds[3]
    if strict_crop[3] >= minimum_height:
        crop = strict_crop
        strategy = "maximum_safe_inscribed_rectangle"
    else:
        crop = _fallback_visible_rectangle(
            safe_visible, config.minimum_row_or_column_coverage
        )
        strategy = "row_column_visible_coverage_fallback"
    x, y, width, height = crop
    if width <= 0 or height <= 0:
        raise RuntimeError("Dense RGB-D fusion produced an empty crop")
    cropped_visible = safe_visible[y : y + height, x : x + width]
    return (
        image[y : y + height, x : x + width],
        cropped_visible,
        foreground_mask[y : y + height, x : x + width],
        crop,
        {
            "strategy": strategy,
            "safe_visible_pixel_count": int(np.count_nonzero(safe_visible)),
            "safe_visible_bounds": {
                "x": int(bounds[0]),
                "y": int(bounds[1]),
                "width": int(bounds[2]),
                "height": int(bounds[3]),
            },
            "strict_crop": {
                "x": int(strict_crop[0]),
                "y": int(strict_crop[1]),
                "width": int(strict_crop[2]),
                "height": int(strict_crop[3]),
            },
        },
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
    tsdf_mesh_glb = _mesh_to_glb(mesh)
    vertices_mm = np.asarray(mesh.vertices, dtype=np.float64) * 1000.0
    plane_depth_mm = _dominant_background_plane_mm(
        vertices_mm,
        np.asarray(canvas.normal_axis, dtype=np.float64),
        selected_config.plane_histogram_bin_mm,
    )
    view_direction = _camera_to_wall_direction(
        all_camera_to_world, canvas, plane_depth_mm
    )
    background, background_valid, _ = _dense_plane_background(
        background_frames,
        intrinsics,
        canvas,
        plane_depth_mm,
        maps,
    )
    foreground_mask = np.zeros(background_valid.shape, dtype=bool)
    tsdf_foreground_pixels = 0
    depth_point_foreground_pixels = 0
    depth_point_metadata: dict[str, int] = {
        "supported_depth_point_count": 0,
        "foreground_candidate_point_count": 0,
        "foreground_zbuffer_pixel_count": 0,
        "foreground_zbuffer_updates": 0,
    }
    if selected_config.foreground_overlay:
        tsdf_foreground, tsdf_valid, _ = _tsdf_foreground_overlay(
            mesh,
            canvas,
            plane_depth_mm,
            all_camera_to_world,
            selected_config,
        )
        depth_foreground, depth_valid, _, depth_point_metadata = (
            _depth_point_foreground_overlay(
                all_depth_frames,
                all_camera_to_world,
                intrinsics,
                canvas,
                plane_depth_mm,
                view_direction,
                maps,
                selected_config,
            )
        )
        # TSDF is the preferred continuous geometry.  Measured RGB-D points
        # are allowed only where that raycast has a real hole; wall texture
        # never fills either kind of foreground hole.
        local_tsdf_holes = _tsdf_local_holes(
            tsdf_valid, selected_config.foreground_tsdf_hole_kernel_size
        )
        depth_fallback = depth_valid & local_tsdf_holes
        background[tsdf_valid] = tsdf_foreground[tsdf_valid]
        background[depth_fallback] = depth_foreground[depth_fallback]
        foreground_mask = tsdf_valid | depth_fallback
        tsdf_foreground_pixels = int(np.count_nonzero(tsdf_valid))
        depth_point_foreground_pixels = int(np.count_nonzero(depth_fallback))
    visible = background_valid | foreground_mask
    image, crop_valid, cropped_foreground, crop, crop_metadata = _crop_dense_result(
        background,
        visible,
        foreground_mask,
        config=selected_config,
    )
    coverage = float(np.mean(crop_valid))
    foreground_pixels = int(np.count_nonzero(cropped_foreground))
    metadata: dict[str, object] = {
        "backend": "tsdf_plane_dense_rgbd",
        "config": asdict(selected_config),
        "canvas_width": canvas.width,
        "canvas_height": canvas.height,
        "crop": {"x": crop[0], "y": crop[1], "width": crop[2], "height": crop[3]},
        "background_plane_normal_depth_mm": plane_depth_mm,
        "tsdf_vertex_count": int(len(mesh.vertices)),
        "tsdf_triangle_count": int(len(mesh.triangles)),
        "tsdf_mesh_glb_byte_count": int(len(tsdf_mesh_glb)),
        "background_valid_pixel_count": int(np.count_nonzero(background_valid)),
        "visible_pixel_count": int(np.count_nonzero(visible)),
        "foreground_view_direction": view_direction,
        "foreground_tsdf_pixel_count": tsdf_foreground_pixels,
        "foreground_depth_point_fallback_pixel_count": depth_point_foreground_pixels,
        "foreground_tsdf_local_hole_pixel_count": int(
            np.count_nonzero(local_tsdf_holes)
        )
        if selected_config.foreground_overlay
        else 0,
        "foreground_overlay_pixel_count": foreground_pixels,
        "foreground_depth_point_audit": depth_point_metadata,
        "crop_valid_coverage": coverage,
        "crop_visibility": crop_metadata,
        "quality_metrics": {
            "quality_pass": coverage >= selected_config.minimum_crop_coverage,
            "crop_valid_coverage": coverage,
            "minimum_crop_coverage": selected_config.minimum_crop_coverage,
            "foreground_overlay_pixel_count": foreground_pixels,
        },
    }
    return DenseFusionResult(
        image=image,
        valid_mask=crop_valid,
        foreground_mask=cropped_foreground,
        tsdf_mesh_glb=tsdf_mesh_glb,
        metadata=metadata,
    )
