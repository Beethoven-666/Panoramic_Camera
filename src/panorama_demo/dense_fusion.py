"""Dense RGB-D side-scan fusion for the ORB-SLAM3 trajectory path.

The Gemini 305 can return no depth on specular or low-texture parts of an
otherwise well-exposed colour image.  Projecting each depth pixel as a single
point therefore leaves black holes even when camera tracking is excellent.
This module uses two complementary, metric layers:

* a TSDF built from every real tracked RGB-D pose preserves foreground
  geometry and ownership; and
* calibrated source RGB is reprojected to a fitted wall plane only where that
  source is known to contain wall, never an object in front of it.

The plane is estimated from the TSDF mesh, never from a fabricated pose.  The
TSDF colours are deliberately not used as final foreground texture: a TSDF
averages observations and therefore softens labels, contours, and thin hoses.
Foreground colour is sampled again from a geometrically consistent real RGB-D
observation.  This is suited to a side scan of a mostly planar wall with
nearby objects such as equipment or vegetation.
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

    voxel_length_mm: float = 5.0
    sdf_truncation_mm: float = 20.0
    maximum_depth_mm: float = 2000.0
    plane_histogram_bin_mm: float = 20.0
    plane_fit_half_band_mm: float = 30.0
    plane_fit_huber_delta_mm: float = 10.0
    maximum_plane_fit_p95_mm: float = 30.0
    maximum_plane_tilt_deg: float = 8.0
    minimum_plane_inlier_fraction: float = 0.04
    minimum_plane_inlier_count: int = 500
    background_core_offset_mm: float = 20.0
    foreground_offset_mm: float = 35.0
    foreground_neighbor_depth_tolerance_mm: float = 70.0
    minimum_foreground_support_neighbors: int = 1
    depth_discontinuity_hard_edge_mm: float = 80.0
    foreground_exclusion_dilate_px_at_640: int = 1
    background_occlusion_margin_mm: float = 25.0
    foreground_texture_depth_tolerance_mm: float = 25.0
    minimum_foreground_multiview_support: int = 2
    minimum_depth_foreground_multiview_support: int = 2
    minimum_depth_foreground_baseline_mm: float = 40.0
    depth_foreground_consistency_tolerance_mm: float = 25.0
    foreground_evidence_dilate_px: int = 2
    foreground_component_single_source_coverage: float = 0.95
    foreground_texture_tile_size_px: int = 32
    foreground_texture_tile_primary_coverage: float = 0.20
    source_wall_orientation_tolerance_deg: float = 25.0
    source_wall_global_depth_tolerance_mm: float = 140.0
    source_wall_ransac_inlier_mm: float = 20.0
    source_wall_minimum_inlier_count: int = 256
    foreground_alpha_radius_px_at_640: float = 2.0
    maximum_foreground_alpha_radius_px: int = 4
    maximum_background_sources: int = 32
    foreground_tsdf_hole_kernel_size: int = 31
    ray_chunk_rows: int = 128
    foreground_overlay: bool = True
    minimum_crop_coverage: float = 0.85
    minimum_row_or_column_coverage: float = 0.98
    minimum_inscribed_crop_height_fraction: float = 0.90
    minimum_crop_width_fraction: float = 0.95
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
            ("plane_fit_half_band_mm", config.plane_fit_half_band_mm),
            ("plane_fit_huber_delta_mm", config.plane_fit_huber_delta_mm),
            ("maximum_plane_fit_p95_mm", config.maximum_plane_fit_p95_mm),
            ("background_core_offset_mm", config.background_core_offset_mm),
            ("foreground_offset_mm", config.foreground_offset_mm),
            (
                "foreground_neighbor_depth_tolerance_mm",
                config.foreground_neighbor_depth_tolerance_mm,
            ),
            (
                "depth_discontinuity_hard_edge_mm",
                config.depth_discontinuity_hard_edge_mm,
            ),
            ("background_occlusion_margin_mm", config.background_occlusion_margin_mm),
            (
                "foreground_texture_depth_tolerance_mm",
                config.foreground_texture_depth_tolerance_mm,
            ),
            (
                "minimum_depth_foreground_baseline_mm",
                config.minimum_depth_foreground_baseline_mm,
            ),
            (
                "depth_foreground_consistency_tolerance_mm",
                config.depth_foreground_consistency_tolerance_mm,
            ),
            (
                "foreground_alpha_radius_px_at_640",
                config.foreground_alpha_radius_px_at_640,
            ),
            (
                "source_wall_global_depth_tolerance_mm",
                config.source_wall_global_depth_tolerance_mm,
            ),
            (
                "source_wall_ransac_inlier_mm",
                config.source_wall_ransac_inlier_mm,
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
        if not 0.0 < config.maximum_plane_tilt_deg <= 45.0:
            raise ValueError("Dense TSDF maximum_plane_tilt_deg must be in (0, 45]")
        if not 0.0 < config.source_wall_orientation_tolerance_deg <= 45.0:
            raise ValueError(
                "Dense TSDF source_wall_orientation_tolerance_deg must be in (0, 45]"
            )
        if config.source_wall_minimum_inlier_count < 64:
            raise ValueError(
                "Dense TSDF source_wall_minimum_inlier_count must be at least 64"
            )
        if not 0.0 < config.minimum_plane_inlier_fraction <= 1.0:
            raise ValueError("Dense TSDF minimum_plane_inlier_fraction must be in (0, 1]")
        if config.minimum_plane_inlier_count < 32:
            raise ValueError("Dense TSDF minimum_plane_inlier_count must be at least 32")
        if config.foreground_exclusion_dilate_px_at_640 < 0:
            raise ValueError("Dense TSDF foreground exclusion dilation cannot be negative")
        if not 0.0 < config.foreground_component_single_source_coverage <= 1.0:
            raise ValueError(
                "Dense TSDF foreground_component_single_source_coverage must be in (0, 1]"
            )
        if not 8 <= config.foreground_texture_tile_size_px <= 128:
            raise ValueError(
                "Dense TSDF foreground_texture_tile_size_px must be in [8, 128]"
            )
        if not 0.0 < config.foreground_texture_tile_primary_coverage <= 1.0:
            raise ValueError(
                "Dense TSDF foreground_texture_tile_primary_coverage must be in (0, 1]"
            )
        if not 1 <= config.minimum_foreground_multiview_support <= 8:
            raise ValueError(
                "Dense TSDF minimum_foreground_multiview_support must be in [1, 8]"
            )
        if not 1 <= config.minimum_depth_foreground_multiview_support <= 8:
            raise ValueError(
                "Dense TSDF minimum_depth_foreground_multiview_support must be in [1, 8]"
            )
        if not 0 <= config.foreground_evidence_dilate_px <= 4:
            raise ValueError(
                "Dense TSDF foreground evidence dilation must be in [0, 4] pixels"
            )
        if not 1 <= config.maximum_foreground_alpha_radius_px <= 8:
            raise ValueError(
                "Dense TSDF maximum_foreground_alpha_radius_px must be in [1, 8]"
            )
        if not 2 <= config.maximum_background_sources <= 32:
            raise ValueError("Dense TSDF maximum_background_sources must be in [2, 32]")
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
        if not 0.0 < config.minimum_crop_width_fraction <= 1.0:
            raise ValueError(
                "Dense TSDF minimum_crop_width_fraction must be in (0, 1]"
            )
        if config.mask_close_kernel_size not in {1, 3}:
            raise ValueError("Dense TSDF mask_close_kernel_size must be 1 or 3")
        if not 0 <= config.mask_erode_pixels <= 3:
            raise ValueError("Dense TSDF mask_erode_pixels must be in [0, 3]")
        return config


@dataclass(frozen=True)
class TSDFMeshVisualizationConfig:
    """Display-only TSDF export settings, deliberately separate from rendering."""

    enabled: bool = True
    voxel_length_mm: float = 5.0
    sdf_truncation_mm: float = 20.0
    maximum_depth_mm: float = 10000.0

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any] | None = None
    ) -> "TSDFMeshVisualizationConfig":
        payload = dict(value or {})
        unknown = sorted(set(payload) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(
                "Unknown tsdf_visualization configuration keys: " + str(unknown)
            )
        config = cls(**payload)
        if not isinstance(config.enabled, bool):
            raise ValueError("tsdf_visualization.enabled must be a boolean")
        for name, number in (
            ("voxel_length_mm", config.voxel_length_mm),
            ("sdf_truncation_mm", config.sdf_truncation_mm),
            ("maximum_depth_mm", config.maximum_depth_mm),
        ):
            if not math.isfinite(number) or number <= 0.0:
                raise ValueError(f"tsdf_visualization.{name} must be finite and positive")
        if config.sdf_truncation_mm < config.voxel_length_mm:
            raise ValueError(
                "tsdf_visualization.sdf_truncation_mm must cover one voxel"
            )
        return config


@dataclass(frozen=True)
class DenseFusionResult:
    image: np.ndarray
    valid_mask: np.ndarray
    foreground_mask: np.ndarray
    background_exclusion_mask: np.ndarray
    tsdf_foreground_mask: np.ndarray
    depth_fallback_mask: np.ndarray
    depth_multiview_foreground_mask: np.ndarray
    foreground_alpha: np.ndarray
    foreground_source_id: np.ndarray
    foreground_confidence: np.ndarray
    background_source_id: np.ndarray
    tsdf_mesh_glb: bytes
    metadata: dict[str, object]

    @property
    def audit_images(self) -> dict[str, np.ndarray]:
        """Final-crop audit rasters, all aligned with ``image`` exactly."""

        return {
            "background_exclusion_mask": self.background_exclusion_mask,
            "tsdf_foreground_mask": self.tsdf_foreground_mask,
            "depth_fallback_mask": self.depth_fallback_mask,
            "depth_multiview_foreground_mask": self.depth_multiview_foreground_mask,
            "foreground_alpha": self.foreground_alpha,
            "foreground_source_id": self.foreground_source_id,
            "foreground_confidence": self.foreground_confidence,
            "background_source_id": self.background_source_id,
        }


@dataclass(frozen=True)
class WallPlaneModel:
    """A metric world-space wall plane whose normal points toward the cameras."""

    normal_world: np.ndarray
    offset_mm: float
    rmse_mm: float
    p95_residual_mm: float
    inlier_fraction: float
    inlier_count: int
    tilt_deg: float

    def signed_distance_mm(self, world_points_mm: np.ndarray) -> np.ndarray:
        points = np.asarray(world_points_mm, dtype=np.float64)
        return points @ self.normal_world + self.offset_mm

    def as_dict(self) -> dict[str, object]:
        return {
            "normal_world": np.asarray(self.normal_world, dtype=float).tolist(),
            "offset_mm": float(self.offset_mm),
            "rmse_mm": float(self.rmse_mm),
            "p95_residual_mm": float(self.p95_residual_mm),
            "inlier_fraction": float(self.inlier_fraction),
            "inlier_count": int(self.inlier_count),
            "tilt_deg": float(self.tilt_deg),
            "signed_distance_positive_direction": "toward_camera",
        }


@dataclass(frozen=True)
class SourceForegroundMasks:
    """Three-state source-space ownership masks for one real RGB-D frame."""

    frame_id: int
    undistort_valid: np.ndarray
    depth_valid: np.ndarray
    depth_edge: np.ndarray
    foreground_core: np.ndarray
    unknown_band: np.ndarray
    tsdf_silhouette: np.ndarray
    background_safe: np.ndarray
    background_sample_safe: np.ndarray
    confidence: np.ndarray
    wall_residual_rmse_mm: float
    wall_residual_p95_mm: float


@dataclass(frozen=True)
class PreparedDenseSource:
    """One undistorted true source, retained only while it is being sampled."""

    frame_id: int
    color: np.ndarray
    depth_mm: np.ndarray
    undistort_valid: np.ndarray
    camera_to_world: np.ndarray
    sharpness: float


@dataclass(frozen=True)
class ForegroundGeometry:
    """TSDF geometry only; final RGB always comes from a true source frame."""

    hard_mask: np.ndarray
    surface_hit_mask: np.ndarray
    world_points_mm: np.ndarray
    normal_world: np.ndarray
    surface_depth_mm: np.ndarray
    triangle_id: np.ndarray


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


def export_tsdf_mesh(
    frames: Sequence[RGBDFrame],
    camera_to_world: Sequence[np.ndarray],
    intrinsics: PinholeIntrinsics,
    *,
    config: TSDFMeshVisualizationConfig | Mapping[str, Any] | None = None,
) -> tuple[bytes, dict[str, object]]:
    """Export a coloured TSDF GLB without invoking any panorama code.

    The caller supplies already validated real RGB-D poses.  This helper never
    creates a plane, foreground mask, ownership map, or RGB panorama; its GLB
    is an inspection-only artifact.
    """

    selected = (
        config
        if isinstance(config, TSDFMeshVisualizationConfig)
        else TSDFMeshVisualizationConfig.from_mapping(config)
    )
    if not selected.enabled:
        raise ValueError("Cannot export a TSDF mesh when tsdf_visualization is disabled")
    integration_config = DenseFusionConfig(
        voxel_length_mm=selected.voxel_length_mm,
        sdf_truncation_mm=selected.sdf_truncation_mm,
        maximum_depth_mm=selected.maximum_depth_mm,
    )
    mesh = _integrate_tsdf(
        frames,
        camera_to_world,
        intrinsics,
        _undistortion_maps(intrinsics),
        integration_config,
    )
    glb = _mesh_to_glb(mesh)
    return glb, {
        "backend": "open3d_scalable_tsdf_display_only",
        "frame_count": len(frames),
        "vertex_count": int(len(mesh.vertices)),
        "triangle_count": int(len(mesh.triangles)),
        "glb_byte_count": len(glb),
        "translation_unit": "mm",
        "configuration": {
            "voxel_length_mm": selected.voxel_length_mm,
            "sdf_truncation_mm": selected.sdf_truncation_mm,
            "maximum_depth_mm": selected.maximum_depth_mm,
        },
        "display_only": True,
        "participates_in_panorama": False,
    }


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


def _normalise(vector: np.ndarray, label: str) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(value))
    if not math.isfinite(norm) or norm < 1e-9:
        raise RuntimeError(f"Dense TSDF has no finite {label}")
    return value / norm


def _mode_plane_candidates(
    vertices_mm: np.ndarray,
    vertex_normals: np.ndarray | None,
    canvas_normal: np.ndarray,
    config: DenseFusionConfig,
) -> np.ndarray:
    """Select the broad, canvas-aligned wall band before robust plane fitting."""

    depth = np.asarray(vertices_mm, dtype=np.float64) @ canvas_normal
    finite = np.isfinite(depth)
    if int(np.count_nonzero(finite)) < config.minimum_plane_inlier_count:
        raise RuntimeError("Dense TSDF mesh has too few finite vertices for wall fitting")
    values = depth[finite]
    low, high = np.percentile(values, (1.0, 99.0))
    edges = np.arange(
        low,
        high + config.plane_histogram_bin_mm,
        config.plane_histogram_bin_mm,
        dtype=np.float64,
    )
    if edges.size < 2:
        center = float(np.median(values))
    else:
        counts, edges = np.histogram(values, bins=edges)
        index = int(np.argmax(counts))
        center = float((edges[index] + edges[index + 1]) * 0.5)
    candidate = finite & (np.abs(depth - center) <= config.plane_fit_half_band_mm)
    if vertex_normals is not None and vertex_normals.shape == vertices_mm.shape:
        normal_valid = np.isfinite(vertex_normals).all(axis=1)
        normal_length = np.linalg.norm(vertex_normals, axis=1)
        alignment = np.zeros(vertices_mm.shape[0], dtype=np.float64)
        usable = normal_valid & (normal_length > 1e-9)
        alignment[usable] = np.abs(
            (vertex_normals[usable] / normal_length[usable, None]) @ canvas_normal
        )
        # The broad wall should be close to the side-scan normal.  This avoids
        # using the long face of a foreground cylinder as a plane in a crowded
        # band, while accepting a genuinely tilted wall.
        candidate &= alignment >= math.cos(math.radians(25.0))
    if int(np.count_nonzero(candidate)) < config.minimum_plane_inlier_count:
        # A mesh may lack reliable normals on a low-texture white wall.  The
        # depth band itself remains a valid deterministic initialiser.
        candidate = finite & (
            np.abs(depth - center) <= config.plane_fit_half_band_mm
        )
    if int(np.count_nonzero(candidate)) < config.minimum_plane_inlier_count:
        raise RuntimeError("Dense TSDF wall band has too few candidate vertices")
    return candidate


def _fit_wall_plane_model(
    mesh: Any,
    canvas: ProjectionCanvas,
    camera_to_world: Sequence[np.ndarray],
    config: DenseFusionConfig,
) -> WallPlaneModel:
    """Fit a robust full 3-D wall plane from the dominant TSDF mesh band.

    The histogram only gives an initial depth band.  The returned plane is a
    total-least-squares Huber fit in world millimetres, with its sign fixed so
    positive signed distance always means "toward the camera".  There is no
    scalar-depth fallback: a bad plane must fail before it can turn wall into
    foreground ownership.
    """

    vertices_mm = np.asarray(mesh.vertices, dtype=np.float64) * 1000.0
    if vertices_mm.ndim != 2 or vertices_mm.shape[1] != 3:
        raise RuntimeError("Dense TSDF mesh has invalid vertex coordinates")
    if mesh.has_vertex_normals():
        normals = np.asarray(mesh.vertex_normals, dtype=np.float64)
    else:
        normals = None
    canvas_normal = _normalise(np.asarray(canvas.normal_axis), "canvas normal")
    candidates = _mode_plane_candidates(vertices_mm, normals, canvas_normal, config)
    points = vertices_mm[candidates]
    # The histogram band can contain several nearby parallel TSDF layers.  A
    # deterministic RANSAC resolves that ambiguity before the Huber fit; using
    # total least squares on the whole band would average those layers and turn
    # its residual into a false foreground band.
    voxel = 10.0
    voxel_key = np.floor(points / voxel).astype(np.int64)
    _, representative = np.unique(voxel_key, axis=0, return_index=True)
    sample_points = points[np.sort(representative)]
    if sample_points.shape[0] < config.minimum_plane_inlier_count:
        sample_points = points
    rng = np.random.default_rng(305_132_427)
    ransac_threshold = min(15.0, config.plane_fit_half_band_mm)
    best_normal: np.ndarray | None = None
    best_offset = 0.0
    best_count = -1
    best_span = -1.0
    best_p95 = math.inf
    scan_axis = _normalise(np.asarray(canvas.scan_axis), "canvas scan axis")
    down_axis = _normalise(-np.asarray(canvas.up_axis), "canvas down axis")
    for _ in range(768):
        selected = rng.choice(len(sample_points), size=3, replace=False)
        a, b, c = sample_points[selected]
        cross = np.cross(b - a, c - a)
        if float(np.linalg.norm(cross)) < 1e-5:
            continue
        normal = _normalise(cross, "RANSAC wall normal")
        tilt = math.degrees(
            math.acos(float(np.clip(abs(np.dot(normal, canvas_normal)), -1.0, 1.0)))
        )
        if tilt > min(25.0, config.maximum_plane_tilt_deg + 12.0):
            continue
        offset = -float(np.dot(normal, a))
        residual = np.abs(sample_points @ normal + offset)
        inlier = residual <= ransac_threshold
        count = int(np.count_nonzero(inlier))
        if count < 3:
            continue
        inlier_points = sample_points[inlier]
        scan_span = float(np.ptp(inlier_points @ scan_axis))
        down_span = float(np.ptp(inlier_points @ down_axis))
        span = scan_span * down_span
        p95 = float(np.percentile(residual[inlier], 95.0))
        if (
            count > best_count
            or (count == best_count and span > best_span)
            or (count == best_count and math.isclose(span, best_span) and p95 < best_p95)
        ):
            best_normal = normal
            best_offset = offset
            best_count = count
            best_span = span
            best_p95 = p95
    if best_normal is None:
        raise RuntimeError("Dense TSDF wall RANSAC could not find a plane")
    residual = points @ best_normal + best_offset
    fit_points = points[np.abs(residual) <= ransac_threshold]
    if fit_points.shape[0] < config.minimum_plane_inlier_count:
        raise RuntimeError("Dense TSDF wall RANSAC has too few full-resolution inliers")
    normal = best_normal
    offset = best_offset
    inlier_points = fit_points
    for _ in range(8):
        residual = inlier_points @ normal + offset
        absolute = np.abs(residual)
        weights = np.minimum(
            1.0, config.plane_fit_huber_delta_mm / np.maximum(absolute, 1e-9)
        )
        total_weight = float(weights.sum())
        centroid = (inlier_points * weights[:, None]).sum(axis=0) / total_weight
        centered = inlier_points - centroid
        covariance = (centered * weights[:, None]).T @ centered / total_weight
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        if not np.isfinite(eigenvalues).all() or eigenvalues[1] <= 1e-8:
            raise RuntimeError("Dense TSDF wall vertices do not span a plane")
        normal = _normalise(eigenvectors[:, 0], "wall normal")
        offset = -float(np.dot(normal, centroid))

    # Do not let the Huber refinement jump from the RANSAC wall layer onto a
    # nearby parallel TSDF layer.  Retaining the deterministic RANSAC model is
    # a valid robust result; it is fundamentally different from a forbidden
    # histogram/median fallback.
    refined_count = int(
        np.count_nonzero(
            candidates
            & (np.abs(vertices_mm @ normal + offset) <= ransac_threshold)
        )
    )
    if refined_count < int(0.75 * fit_points.shape[0]):
        normal = best_normal
        offset = best_offset
    residual_all = vertices_mm @ normal + offset
    inlier = candidates & (np.abs(residual_all) <= ransac_threshold)
    inlier_count = int(np.count_nonzero(inlier))
    inlier_fraction = float(inlier_count / max(1, len(vertices_mm)))
    if inlier_count < config.minimum_plane_inlier_count or (
        inlier_fraction < config.minimum_plane_inlier_fraction
    ):
        raise RuntimeError(
            "Dense TSDF wall fit has insufficient inliers "
            f"({inlier_count} vertices, {inlier_fraction:.3f} fraction; "
            f"RANSAC candidate={best_count}, full-band={len(points)}, "
            f"refined={refined_count})"
        )
    inlier_residual = residual_all[inlier]
    rmse = float(np.sqrt(np.mean(inlier_residual**2)))
    p95 = float(np.percentile(np.abs(inlier_residual), 95.0))
    tilt = math.degrees(
        math.acos(float(np.clip(abs(np.dot(normal, canvas_normal)), -1.0, 1.0)))
    )
    centers = np.asarray(
        [np.asarray(pose, dtype=np.float64)[:3, 3] for pose in camera_to_world],
        dtype=np.float64,
    )
    camera_signed = centers @ normal + offset
    if not np.isfinite(camera_signed).all() or abs(float(np.median(camera_signed))) < 1e-3:
        raise RuntimeError("Dense TSDF wall plane has no unambiguous camera side")
    if float(np.median(camera_signed)) < 0.0:
        normal = -normal
        offset = -offset
    if p95 > config.maximum_plane_fit_p95_mm:
        raise RuntimeError(
            "Dense TSDF wall plane residual is too high "
            f"({p95:.1f} mm > {config.maximum_plane_fit_p95_mm:.1f} mm)"
        )
    if tilt > config.maximum_plane_tilt_deg:
        raise RuntimeError(
            "Dense TSDF wall plane tilt is too high "
            f"({tilt:.2f} deg > {config.maximum_plane_tilt_deg:.2f} deg)"
        )
    return WallPlaneModel(
        normal_world=normal,
        offset_mm=offset,
        rmse_mm=rmse,
        p95_residual_mm=p95,
        inlier_fraction=inlier_fraction,
        inlier_count=inlier_count,
        tilt_deg=tilt,
    )


def _wall_plane_world_grid(
    canvas: ProjectionCanvas,
    plane: WallPlaneModel,
    scan_values: np.ndarray,
    down_values: np.ndarray,
) -> np.ndarray:
    """Intersect each canvas scan/down line with the fitted 3-D wall plane."""

    scan_axis = _normalise(np.asarray(canvas.scan_axis), "canvas scan axis")
    down_axis = _normalise(-np.asarray(canvas.up_axis), "canvas down axis")
    normal_axis = _normalise(np.asarray(canvas.normal_axis), "canvas normal")
    denominator = float(np.dot(plane.normal_world, normal_axis))
    if abs(denominator) < 1e-5:
        raise RuntimeError("Dense TSDF wall plane is parallel to the projection rays")
    scan_grid, down_grid = np.meshgrid(scan_values, down_values, indexing="xy")
    base = (
        scan_grid[..., None] * scan_axis
        + down_grid[..., None] * down_axis
    )
    distance = -(base @ plane.normal_world + plane.offset_mm) / denominator
    return base + distance[..., None] * normal_axis


def _source_sharpness(color: np.ndarray) -> float:
    """A bounded source-local detail score used only to break valid ties."""

    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    if max(gray.shape) > 640:
        scale = 640.0 / max(gray.shape)
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return float(np.var(cv2.Laplacian(gray, cv2.CV_32F)))


def _prepare_projection_source(
    frame: RGBDProjectionFrame,
    maps: tuple[np.ndarray, np.ndarray] | None,
) -> PreparedDenseSource:
    color, depth, valid = _undistort(frame.rgb, frame.depth_mm, maps)
    return PreparedDenseSource(
        frame_id=int(frame.frame_id),
        color=color,
        depth_mm=depth,
        undistort_valid=valid,
        camera_to_world=np.asarray(frame.camera_to_world, dtype=np.float64),
        sharpness=_source_sharpness(color),
    )


def _prepare_session_source(
    frame: RGBDFrame,
    pose: np.ndarray,
    maps: tuple[np.ndarray, np.ndarray] | None,
) -> PreparedDenseSource:
    color = _decode_color(frame.color_path)
    depth = read_aligned_depth_mm(frame)
    color, depth, valid = _undistort(color, depth, maps)
    return PreparedDenseSource(
        frame_id=int(frame.frame_id),
        color=color,
        depth_mm=depth,
        undistort_valid=valid,
        camera_to_world=np.asarray(pose, dtype=np.float64),
        sharpness=_source_sharpness(color),
    )


def _depth_edge_mask(
    depth_mm: np.ndarray,
    valid: np.ndarray,
    threshold_mm: float,
) -> np.ndarray:
    """Mark both sides of hard range discontinuities as an unknown safety band."""

    depth = np.asarray(depth_mm, dtype=np.float32)
    usable = np.asarray(valid, dtype=bool)
    edge = np.zeros(depth.shape, dtype=bool)
    horizontal = (
        usable[:, 1:]
        & usable[:, :-1]
        & (np.abs(depth[:, 1:] - depth[:, :-1]) >= threshold_mm)
    )
    vertical = (
        usable[1:, :]
        & usable[:-1, :]
        & (np.abs(depth[1:, :] - depth[:-1, :]) >= threshold_mm)
    )
    edge[:, 1:] |= horizontal
    edge[:, :-1] |= horizontal
    edge[1:, :] |= vertical
    edge[:-1, :] |= vertical
    return edge


def _adaptive_support_tolerance_mm(
    depth_mm: np.ndarray,
    config: DenseFusionConfig,
) -> np.ndarray:
    """Depth-adaptive neighbour tolerance, deliberately capped against bridging."""

    depth = np.asarray(depth_mm, dtype=np.float32)
    upper = min(40.0, config.foreground_neighbor_depth_tolerance_mm)
    return np.clip(np.maximum(20.0, depth * 0.02), 20.0, upper).astype(np.float32)


def _scaled_source_kernel_radius(
    config: DenseFusionConfig,
    width: int,
) -> int:
    return int(
        round(config.foreground_exclusion_dilate_px_at_640 * width / 640.0)
    )


def _predicted_wall_depth_mm(
    source: PreparedDenseSource,
    intrinsics: PinholeIntrinsics,
    plane: WallPlaneModel,
) -> np.ndarray:
    """Camera-z depth of the world wall plane at every undistorted source ray."""

    height, width = source.depth_mm.shape
    columns = np.arange(width, dtype=np.float64)
    rows = np.arange(height, dtype=np.float64)
    x, y = np.meshgrid(
        (columns - intrinsics.cx) / intrinsics.fx,
        (rows - intrinsics.cy) / intrinsics.fy,
        indexing="xy",
    )
    rotation = source.camera_to_world[:3, :3]
    # Row-vector convention: world = camera @ R.T + t.
    normal_camera = rotation.T @ plane.normal_world
    camera_signed = float(
        source.camera_to_world[:3, 3] @ plane.normal_world + plane.offset_mm
    )
    denominator = (
        normal_camera[0] * x + normal_camera[1] * y + normal_camera[2]
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        predicted = -camera_signed / denominator
    return np.where(
        np.isfinite(predicted) & (predicted > 0.0), predicted, 0.0
    ).astype(np.float32)


def _camera_plane_depth_mm(normal_camera: np.ndarray, offset_mm: float) -> float:
    """Return the plane depth at the calibrated principal ray."""

    normal = _normalise(np.asarray(normal_camera, dtype=np.float64), "source wall normal")
    if abs(float(normal[2])) < 1e-6:
        raise RuntimeError("Source wall plane is parallel to the camera principal ray")
    depth = -float(offset_mm) / float(normal[2])
    if not math.isfinite(depth) or depth <= 0.0:
        raise RuntimeError("Source wall plane is behind the camera")
    return depth


def _fit_constrained_source_wall_plane(
    source: PreparedDenseSource,
    intrinsics: PinholeIntrinsics,
    plane: WallPlaneModel,
    depth_valid: np.ndarray,
    config: DenseFusionConfig,
) -> tuple[np.ndarray, float, float, float]:
    """Fit one source-local wall plane without allowing an object to become it.

    The fitted TSDF plane supplies a *prior*, not a per-pixel foreground label:
    its camera-space normal constrains local RANSAC and its principal-ray depth
    bounds the permitted offset.  The final local plane is independently fit
    to this source's true depth.  Consequently a large, nearly parallel
    extinguisher face cannot replace the wall merely because it has more valid
    depth pixels, while a small RGB-D/trajectory bias is still absorbed by the
    local model.
    """

    predicted = _predicted_wall_depth_mm(source, intrinsics, plane)
    usable = (
        np.asarray(depth_valid, dtype=bool)
        & np.isfinite(predicted)
        & (predicted > 0.0)
    )
    if int(np.count_nonzero(usable)) < config.source_wall_minimum_inlier_count:
        raise RuntimeError(
            f"Source {source.frame_id} has too little depth for local wall fitting"
        )
    # The global model merely selects a plausible wall *band*.  Object faces
    # farther than this cannot seed RANSAC, even if they are locally dominant.
    depth = source.depth_mm.astype(np.float64)
    seed = usable & (
        np.abs(predicted.astype(np.float64) - depth)
        <= config.source_wall_global_depth_tolerance_mm
    )
    minimum_seed_count = max(
        config.source_wall_minimum_inlier_count,
        int(np.count_nonzero(usable)) // 200,
    )
    if int(np.count_nonzero(seed)) < minimum_seed_count:
        raise RuntimeError(
            f"Source {source.frame_id} has insufficient local depth near the wall prior"
        )
    rows, columns = np.nonzero(seed)
    z = depth[rows, columns]
    points = np.stack(
        (
            (columns.astype(np.float64) - intrinsics.cx) * z / intrinsics.fx,
            (rows.astype(np.float64) - intrinsics.cy) * z / intrinsics.fy,
            z,
        ),
        axis=1,
    )
    expected_normal = _normalise(
        source.camera_to_world[:3, :3].T @ plane.normal_world,
        "source wall prior normal",
    )
    expected_offset = float(
        source.camera_to_world[:3, 3] @ plane.normal_world + plane.offset_mm
    )
    expected_depth = _camera_plane_depth_mm(expected_normal, expected_offset)
    # A bounded, deterministic representative set keeps this operation cheap
    # on 1280x800 inputs yet samples all parts of the image rather than only a
    # contiguous row-major prefix.
    if len(points) > 4096:
        rng = np.random.default_rng(305_000 + int(source.frame_id))
        sample_indices = rng.choice(len(points), size=4096, replace=False)
        sample_points = points[sample_indices]
    else:
        sample_points = points
    if len(sample_points) < 3:
        raise RuntimeError(f"Source {source.frame_id} has too few wall samples")
    normal_cosine = math.cos(math.radians(config.source_wall_orientation_tolerance_deg))
    rng = np.random.default_rng(1_305_000 + int(source.frame_id))
    best_normal: np.ndarray | None = None
    best_offset = 0.0
    best_count = -1
    best_span = -1.0
    best_p95 = math.inf
    for _ in range(192):
        selected = rng.choice(len(sample_points), size=3, replace=False)
        a, b, c = sample_points[selected]
        cross = np.cross(b - a, c - a)
        if float(np.linalg.norm(cross)) < 1e-6:
            continue
        normal = _normalise(cross, "source RANSAC wall normal")
        if float(np.dot(normal, expected_normal)) < 0.0:
            normal = -normal
        if float(np.dot(normal, expected_normal)) < normal_cosine:
            continue
        offset = -float(np.dot(normal, a))
        try:
            candidate_depth = _camera_plane_depth_mm(normal, offset)
        except RuntimeError:
            continue
        if (
            abs(candidate_depth - expected_depth)
            > config.source_wall_global_depth_tolerance_mm
        ):
            continue
        residual = np.abs(sample_points @ normal + offset)
        inlier = residual <= config.source_wall_ransac_inlier_mm
        count = int(np.count_nonzero(inlier))
        if count < 3:
            continue
        inlier_points = sample_points[inlier]
        span = float(
            np.ptp(inlier_points[:, 0]) * np.ptp(inlier_points[:, 1])
        )
        p95 = float(np.percentile(residual[inlier], 95.0))
        if (
            count > best_count
            or (count == best_count and span > best_span)
            or (count == best_count and math.isclose(span, best_span) and p95 < best_p95)
        ):
            best_normal = normal
            best_offset = offset
            best_count = count
            best_span = span
            best_p95 = p95
    if best_normal is None:
        raise RuntimeError(f"Source {source.frame_id} local wall RANSAC failed")
    residual = np.abs(points @ best_normal + best_offset)
    inlier = residual <= config.source_wall_ransac_inlier_mm
    if int(np.count_nonzero(inlier)) < minimum_seed_count:
        raise RuntimeError(
            f"Source {source.frame_id} local wall RANSAC has too few inliers"
        )
    normal = best_normal
    offset = best_offset
    inlier_points = points[inlier]
    for _ in range(6):
        residual = inlier_points @ normal + offset
        weights = np.minimum(
            1.0,
            config.plane_fit_huber_delta_mm / np.maximum(np.abs(residual), 1e-6),
        )
        weight_sum = float(weights.sum())
        centroid = (inlier_points * weights[:, None]).sum(axis=0) / weight_sum
        centered = inlier_points - centroid
        covariance = (centered * weights[:, None]).T @ centered / weight_sum
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        if not np.isfinite(eigenvalues).all() or eigenvalues[1] <= 1e-8:
            raise RuntimeError(f"Source {source.frame_id} local wall fit is singular")
        refined_normal = _normalise(eigenvectors[:, 0], "refined source wall normal")
        if float(np.dot(refined_normal, expected_normal)) < 0.0:
            refined_normal = -refined_normal
        refined_offset = -float(np.dot(refined_normal, centroid))
        # Never let robust fitting drift from the admissible global-wall
        # neighbourhood into a foreground parallel layer.
        if (
            float(np.dot(refined_normal, expected_normal)) < normal_cosine
            or abs(_camera_plane_depth_mm(refined_normal, refined_offset) - expected_depth)
            > config.source_wall_global_depth_tolerance_mm
        ):
            break
        normal = refined_normal
        offset = refined_offset
        residual = np.abs(points @ normal + offset)
        inlier_points = points[residual <= config.source_wall_ransac_inlier_mm]
        if len(inlier_points) < minimum_seed_count:
            raise RuntimeError(
                f"Source {source.frame_id} local wall refinement lost its inliers"
            )
    final_error = inlier_points @ normal + offset
    return (
        normal,
        offset,
        float(np.sqrt(np.mean(final_error**2))),
        float(np.percentile(np.abs(final_error), 95.0)),
    )


def _source_wall_relative_depth_mm(
    source: PreparedDenseSource,
    intrinsics: PinholeIntrinsics,
    plane: WallPlaneModel,
    depth_valid: np.ndarray,
    config: DenseFusionConfig,
) -> tuple[np.ndarray, float, float]:
    """Measure a depth sample in front of this source's constrained wall plane."""

    normal, offset, rmse, p95 = _fit_constrained_source_wall_plane(
        source, intrinsics, plane, depth_valid, config
    )
    height, width = source.depth_mm.shape
    columns = np.arange(width, dtype=np.float64)
    rows = np.arange(height, dtype=np.float64)
    x, y = np.meshgrid(
        (columns - intrinsics.cx) / intrinsics.fx,
        (rows - intrinsics.cy) / intrinsics.fy,
        indexing="xy",
    )
    denominator = normal[0] * x + normal[1] * y + normal[2]
    with np.errstate(divide="ignore", invalid="ignore"):
        predicted = -offset / denominator
    predicted = np.where(
        np.isfinite(predicted) & (predicted > 0.0), predicted, 0.0
    )
    relative = predicted - source.depth_mm.astype(np.float64)
    return relative.astype(np.float32), rmse, p95


def _classify_source_foreground(
    source: PreparedDenseSource,
    intrinsics: PinholeIntrinsics,
    plane: WallPlaneModel,
    config: DenseFusionConfig,
    *,
    tsdf_silhouette: np.ndarray | None = None,
    wall_relative_cache: dict[int, tuple[np.ndarray, float, float]] | None = None,
) -> SourceForegroundMasks:
    """Build the source-space core/unknown/background-safe ownership states.

    Invalid depth remains eligible for wall texture *unless* a TSDF silhouette
    later says foreground occupies that ray.  That distinction is what lets a
    white wall recover RGB through a depth hole without painting through a
    reflective extinguisher label.
    """

    depth = source.depth_mm
    depth_valid = (
        source.undistort_valid
        & np.isfinite(depth)
        & (depth > 0.0)
        & (depth <= config.maximum_depth_mm)
    )
    supported = _supported_depth_mask(
        depth,
        depth_valid,
        tolerance_mm=_adaptive_support_tolerance_mm(depth, config),
        minimum_neighbors=config.minimum_foreground_support_neighbors,
    )
    depth_edge = _depth_edge_mask(
        depth, depth_valid, config.depth_discontinuity_hard_edge_mm
    )
    if tsdf_silhouette is None:
        silhouette = np.zeros(depth.shape, dtype=bool)
    else:
        silhouette = np.asarray(tsdf_silhouette, dtype=bool)
        if silhouette.shape != depth.shape:
            raise ValueError("TSDF source silhouette must match the undistorted source")
    cached_relative = (
        wall_relative_cache.get(source.frame_id)
        if wall_relative_cache is not None
        else None
    )
    if cached_relative is None:
        cached_relative = _source_wall_relative_depth_mm(
            source, intrinsics, plane, depth_valid, config
        )
        if wall_relative_cache is not None:
            wall_relative_cache[source.frame_id] = cached_relative
    relative, residual_rmse, residual_p95 = cached_relative
    raw_foreground_core = (
        depth_valid
        & supported
        & (relative >= config.foreground_offset_mm)
    )
    if np.any(silhouette):
        # The TSDF silhouette is already source-view z-buffered and checked
        # against this source's aligned depth.  Do not enlarge it before
        # semantic association: a 3x3 expansion of sparse projection points
        # was enough to turn a 40k-pixel extinguisher silhouette into a
        # 200k+ wall exclusion.  The final, separately configured two-pixel
        # exclusion dilation below remains the only bilinear-sampling guard.
        near_tsdf_foreground = silhouette
    else:
        # A caller without TSDF geometry is a legacy/diagnostic source; retain
        # the depth-only classification rather than silently inventing a mask.
        near_tsdf_foreground = np.ones(depth.shape, dtype=bool)
    # Raw RGB-D is a complement for TSDF geometry, not an independent global
    # wall segmenter.  Its source-local residual can still be noisy on white
    # walls, so only let it extend ownership in a small TSDF foreground
    # neighbourhood.  This prevents an entire white wall from becoming an
    # "unknown" texture exclusion because of a smooth sensor bias.
    foreground_core = raw_foreground_core & near_tsdf_foreground
    # A wall point is trustworthy only near the fitted plane and away from a
    # range edge.  Everything else with valid depth becomes an exclusion band;
    # it is safer to leave an occasional wall pixel blank than sample object
    # RGB into the background layer.
    background_core = (
        depth_valid
        & supported
        & (
            ~near_tsdf_foreground
            | (~depth_edge & (relative <= config.background_core_offset_mm))
        )
    )
    unknown_band = (
        depth_valid
        & near_tsdf_foreground
        & ~(foreground_core | background_core)
    )
    exclusion = (
        foreground_core | unknown_band | (depth_edge & near_tsdf_foreground) | silhouette
    )
    radius = _scaled_source_kernel_radius(config, source.color.shape[1])
    if radius:
        kernel = np.ones((radius * 2 + 1, radius * 2 + 1), dtype=np.uint8)
        exclusion = cv2.dilate(exclusion.astype(np.uint8), kernel) > 0
    background_safe = source.undistort_valid & ~exclusion
    background_sample_safe = cv2.erode(
        background_safe.astype(np.uint8), np.ones((3, 3), dtype=np.uint8)
    ) > 0
    confidence = np.zeros(depth.shape, dtype=np.float32)
    confidence[background_core] = 1.0
    confidence[foreground_core] = 1.0
    confidence[unknown_band] = 0.5
    confidence[depth_edge] = 0.25
    return SourceForegroundMasks(
        frame_id=source.frame_id,
        undistort_valid=source.undistort_valid,
        depth_valid=depth_valid,
        depth_edge=depth_edge,
        foreground_core=foreground_core,
        unknown_band=unknown_band,
        tsdf_silhouette=silhouette,
        background_safe=background_safe,
        background_sample_safe=background_sample_safe,
        confidence=confidence,
        wall_residual_rmse_mm=residual_rmse,
        wall_residual_p95_mm=residual_p95,
    )


def _dense_plane_background(
    sources: Sequence[PreparedDenseSource],
    source_masks: Sequence[SourceForegroundMasks],
    intrinsics: PinholeIntrinsics,
    canvas: ProjectionCanvas,
    plane: WallPlaneModel,
    config: DenseFusionConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    """Texture a fitted wall plane from only source-space safe wall samples.

    The background never samples an excluded foreground/unknown texel.  The
    additional measured-depth test catches an object even if a source mask was
    weakened by a local depth hole.  Both checks run before a candidate can own
    an output pixel, which makes the ownership invariant mechanically true.
    """

    if not sources:
        raise ValueError("Dense plane texture requires at least one source")
    if len(sources) != len(source_masks):
        raise ValueError("Dense plane texture sources and masks must have equal length")
    height, width = canvas.height, canvas.width
    image = np.zeros((height, width, 3), dtype=np.uint8)
    valid = np.zeros((height, width), dtype=bool)
    owner_distance = np.full((height, width), np.inf, dtype=np.float32)
    source_id = np.full((height, width), -1, dtype=np.int32)
    blocked_by_foreground = np.zeros((height, width), dtype=bool)
    audit = {
        "background_candidate_in_frame_count": 0,
        "background_rejected_source_exclusion_count": 0,
        "background_rejected_measured_occlusion_count": 0,
        "background_accepted_sample_count": 0,
        "background_sample_from_excluded_foreground_count": 0,
    }
    scan_axis = np.asarray(canvas.scan_axis, dtype=np.float64)
    min_scan, min_down, _, _ = canvas.world_bounds
    scan_values = min_scan + (
        np.arange(width, dtype=np.float64) + 0.5
    ) / canvas.pixels_per_mm

    for source, masks in zip(sources, source_masks, strict=True):
        pose = source.camera_to_world
        rotation = pose[:3, :3]
        translation = pose[:3, 3]
        camera_scan_position = float(np.dot(translation, scan_axis))
        for row_start in range(0, height, 128):
            row_stop = min(height, row_start + 128)
            down_values = min_down + (
                np.arange(row_start, row_stop, dtype=np.float64) + 0.5
            ) / canvas.pixels_per_mm
            scan_grid, _ = np.meshgrid(scan_values, down_values, indexing="xy")
            world = _wall_plane_world_grid(canvas, plane, scan_values, down_values)
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
                source.color,
                u.astype(np.float32),
                v.astype(np.float32),
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )
            sampled_source_safe = cv2.remap(
                masks.background_sample_safe.astype(np.uint8),
                u.astype(np.float32),
                v.astype(np.float32),
                cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            ) > 0
            sampled_depth = cv2.remap(
                source.depth_mm.astype(np.float32),
                u.astype(np.float32),
                v.astype(np.float32),
                cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            sampled_depth_valid = cv2.remap(
                masks.depth_valid.astype(np.uint8),
                u.astype(np.float32),
                v.astype(np.float32),
                cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            ) > 0
            sampled_verified_foreground = cv2.remap(
                masks.foreground_core.astype(np.uint8),
                u.astype(np.float32),
                v.astype(np.float32),
                cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            ) > 0
            # A raw close depth alone is only a candidate: on this white-wall
            # capture it frequently forms a coherent but false front layer.
            # It may occlude wall texture only after the source-view TSDF
            # z-buffer and same-surface test have made it a verified source
            # foreground core.  That check is deliberately independent of
            # RGB colour, so black hoses remain protected.
            occluded = sampled_depth_valid & sampled_verified_foreground & (
                sampled_depth < z - config.background_occlusion_margin_mm
            )
            audit["background_candidate_in_frame_count"] += int(
                np.count_nonzero(in_frame)
            )
            audit["background_rejected_source_exclusion_count"] += int(
                np.count_nonzero(in_frame & ~sampled_source_safe)
            )
            audit["background_rejected_measured_occlusion_count"] += int(
                np.count_nonzero(in_frame & sampled_source_safe & occluded)
            )
            blocked_by_foreground[row_start:row_stop] |= in_frame & (
                ~sampled_source_safe | occluded
            )
            distance = np.abs(scan_grid - camera_scan_position).astype(np.float32)
            best = owner_distance[row_start:row_stop]
            replace = in_frame & sampled_source_safe & ~occluded & (distance < best)
            if np.any(replace & ~sampled_source_safe):  # defensive owner invariant
                audit["background_sample_from_excluded_foreground_count"] += int(
                    np.count_nonzero(replace & ~sampled_source_safe)
                )
            image_slice = image[row_start:row_stop]
            image_slice[replace] = warped[replace]
            best[replace] = distance[replace]
            valid[row_start:row_stop][replace] = True
            source_id[row_start:row_stop][replace] = source.frame_id
            audit["background_accepted_sample_count"] += int(np.count_nonzero(replace))
    if audit["background_sample_from_excluded_foreground_count"]:
        raise RuntimeError("Dense background sampled an excluded foreground texel")
    return image, valid, source_id, blocked_by_foreground, audit


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


def _depth_point_foreground_geometry(
    all_frames: Sequence[RGBDFrame],
    all_poses: Sequence[np.ndarray],
    intrinsics: PinholeIntrinsics,
    canvas: ProjectionCanvas,
    plane: WallPlaneModel,
    maps: tuple[np.ndarray, np.ndarray] | None,
    config: DenseFusionConfig,
    geometry: ForegroundGeometry | None = None,
    *,
    wall_relative_cache: dict[int, tuple[np.ndarray, float, float]] | None = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, object],
]:
    """Z-buffer independent real-depth foreground and verify it across views.

    The first pass records the closest true depth point on every orthographic
    ray.  The second pass counts only observations that reproduce that same
    world-normal depth from a distinct real camera position.  This rejects
    sparse, image-tied false depths on an otherwise depthless white wall: they
    may look foreground in one RGB-D frame, but do not remain at one metric
    world position as the camera moves.
    """

    if len(all_frames) != len(all_poses) or not all_frames:
        raise ValueError("Dense foreground point fusion requires aligned frames and poses")
    # The measured-depth witness must be independent of the TSDF silhouette.
    # TSDF is later validated against this evidence, never used to create it.
    del geometry
    height, width = canvas.height, canvas.width
    output = np.zeros((height, width, 3), dtype=np.uint8)
    valid = np.zeros((height, width), dtype=bool)
    source_id = np.full((height, width), -1, dtype=np.int32)
    virtual_depth = np.full((height, width), -np.inf, dtype=np.float32)
    source_depth_count = 0
    foreground_count = 0
    projected_count = 0
    rejected_sources: list[dict[str, object]] = []

    def source_candidates(
        frame: RGBDFrame, pose: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, float, SourceForegroundMasks]:
        source = _prepare_session_source(frame, pose, maps)
        masks = _classify_source_foreground(
            source,
            intrinsics,
            plane,
            config,
            tsdf_silhouette=None,
            wall_relative_cache=wall_relative_cache,
        )
        rows, columns = np.nonzero(masks.foreground_core)
        if not rows.size:
            return (
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.float32),
                np.empty((0, 3), dtype=np.uint8),
                source.frame_id,
                float(source.camera_to_world[:3, 3] @ np.asarray(canvas.scan_axis)),
                masks,
            )
        z = source.depth_mm[rows, columns].astype(np.float64)
        camera_points = np.stack(
            (
                (columns.astype(np.float64) - intrinsics.cx) * z / intrinsics.fx,
                (rows.astype(np.float64) - intrinsics.cy) * z / intrinsics.fy,
                z,
            ),
            axis=1,
        )
        world = (
            camera_points @ source.camera_to_world[:3, :3].T
            + source.camera_to_world[:3, 3]
        )
        canvas_points = canvas.world_to_canvas(world)
        x = np.floor(canvas_points[:, 0]).astype(np.int64)
        y = np.floor(canvas_points[:, 1]).astype(np.int64)
        in_canvas = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        if not np.any(in_canvas):
            return (
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.float32),
                np.empty((0, 3), dtype=np.uint8),
                source.frame_id,
                float(source.camera_to_world[:3, 3] @ np.asarray(canvas.scan_axis)),
                masks,
            )
        x = x[in_canvas]
        y = y[in_canvas]
        world = world[in_canvas]
        colors = source.color[rows[in_canvas], columns[in_canvas]]
        flat = y * width + x
        # Positive signed wall distance is by construction camera-side.  Keep
        # one nearest measured point per source/output pixel before combining
        # different real frames, so a single depth map cannot inflate support.
        camera_side_depth = plane.signed_distance_mm(world).astype(np.float32)
        order = np.lexsort((camera_side_depth, flat))
        ordered_flat = flat[order]
        keep = np.empty(order.size, dtype=bool)
        keep[:-1] = ordered_flat[:-1] != ordered_flat[1:]
        keep[-1] = True
        local_winners = order[keep]
        return (
            flat[local_winners],
            camera_side_depth[local_winners],
            colors[local_winners],
            source.frame_id,
            float(source.camera_to_world[:3, 3] @ np.asarray(canvas.scan_axis)),
            masks,
        )

    for frame, pose in zip(all_frames, all_poses, strict=True):
        try:
            candidate_flat, candidate_depth, colors, frame_id, _, masks = source_candidates(
                frame, pose
            )
        except RuntimeError as exc:
            rejected_sources.append({"frame_id": frame.frame_id, "reason": str(exc)})
            continue
        source_depth_count += int(np.count_nonzero(masks.depth_valid))
        foreground_count += int(np.count_nonzero(masks.foreground_core))
        if not candidate_flat.size:
            continue
        zbuffer = virtual_depth.reshape(-1)
        replace = candidate_depth > zbuffer[candidate_flat]
        if not np.any(replace):
            continue
        selected_flat = candidate_flat[replace]
        output.reshape(-1, 3)[selected_flat] = colors[replace]
        source_id.reshape(-1)[selected_flat] = frame_id
        virtual_depth.reshape(-1)[selected_flat] = candidate_depth[replace]
        valid.reshape(-1)[selected_flat] = True
        projected_count += int(np.count_nonzero(replace))

    # Reproject the source-local winners and accept support only when their
    # camera-side depth agrees with the global z-buffer.  This is a metric
    # same-surface check, not a 2-D optical-flow or colour heuristic.
    verified_support = np.zeros((height, width), dtype=np.uint8)
    first_support_scan = np.full((height, width), np.nan, dtype=np.float32)
    support_baseline_mm = np.zeros((height, width), dtype=np.float32)
    zbuffer = virtual_depth.reshape(-1)
    for frame, pose in zip(all_frames, all_poses, strict=True):
        try:
            candidate_flat, candidate_depth, _, _, camera_scan, _ = source_candidates(
                frame, pose
            )
        except RuntimeError:
            continue
        if not candidate_flat.size:
            continue
        consistent = np.abs(candidate_depth - zbuffer[candidate_flat]) <= (
            config.depth_foreground_consistency_tolerance_mm
        )
        if not np.any(consistent):
            continue
        selected_flat = candidate_flat[consistent]
        support_flat = verified_support.reshape(-1)
        anchor_flat = first_support_scan.reshape(-1)
        span_flat = support_baseline_mm.reshape(-1)
        first = support_flat[selected_flat] == 0
        if np.any(first):
            anchor_flat[selected_flat[first]] = np.float32(camera_scan)
        span_flat[selected_flat] = np.maximum(
            span_flat[selected_flat],
            np.abs(np.float32(camera_scan) - anchor_flat[selected_flat]),
        )
        support_flat[selected_flat] = np.minimum(
            np.iinfo(np.uint8).max,
            support_flat[selected_flat].astype(np.uint16) + 1,
        ).astype(np.uint8)
    supported_pixels = valid & (verified_support > 0)
    support_values = verified_support[supported_pixels]
    baseline_values = support_baseline_mm[supported_pixels]
    return (
        output,
        valid,
        source_id,
        virtual_depth,
        verified_support,
        support_baseline_mm,
        {
        "supported_depth_point_count": source_depth_count,
        "foreground_candidate_point_count": foreground_count,
        "foreground_zbuffer_pixel_count": int(np.count_nonzero(valid)),
        "foreground_zbuffer_updates": projected_count,
        "foreground_depth_multiview_support_p50": float(
            np.median(support_values) if support_values.size else 0.0
        ),
        "foreground_depth_multiview_support_p95": float(
            np.percentile(support_values, 95.0) if support_values.size else 0.0
        ),
        "foreground_depth_multiview_baseline_p50_mm": float(
            np.median(baseline_values) if baseline_values.size else 0.0
        ),
        "foreground_depth_multiview_baseline_p95_mm": float(
            np.percentile(baseline_values, 95.0) if baseline_values.size else 0.0
        ),
        "foreground_depth_rejected_source_count": len(rejected_sources),
        "foreground_depth_rejected_sources": rejected_sources,
        },
    )


def _raycast_foreground_geometry(
    mesh: Any,
    canvas: ProjectionCanvas,
    plane: WallPlaneModel,
    camera_to_world: Sequence[np.ndarray],
    config: DenseFusionConfig,
) -> ForegroundGeometry:
    """Raycast TSDF geometry from the camera side, without reading its colours."""

    o3d = _require_open3d()
    vertices_m = np.asarray(mesh.vertices, dtype=np.float64)
    triangles = np.asarray(mesh.triangles, dtype=np.int64)
    if vertices_m.ndim != 2 or vertices_m.shape[1] != 3 or not len(vertices_m):
        raise RuntimeError("Dense TSDF mesh has invalid foreground geometry")
    if triangles.ndim != 2 or triangles.shape[1] != 3 or not len(triangles):
        raise RuntimeError("Dense TSDF mesh has no foreground triangles")
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    height, width = canvas.height, canvas.width
    hard_mask = np.zeros((height, width), dtype=bool)
    surface_hit_mask = np.zeros((height, width), dtype=bool)
    world_points = np.full((height, width, 3), np.nan, dtype=np.float32)
    normal_world = np.full((height, width, 3), np.nan, dtype=np.float32)
    surface_depth = np.full((height, width), np.nan, dtype=np.float32)
    triangle_id = np.full((height, width), -1, dtype=np.int32)
    camera_centers = np.asarray(
        [np.asarray(pose, dtype=np.float64)[:3, 3] for pose in camera_to_world],
        dtype=np.float64,
    )
    camera_signed = plane.signed_distance_mm(camera_centers)
    if not np.isfinite(camera_signed).all() or np.any(camera_signed <= 0.0):
        raise RuntimeError("Dense TSDF wall plane does not put all cameras on its front side")
    # Start just in front of the outermost real camera and traverse toward the
    # wall.  Geometry is metres only in the Open3D raycast bridge.
    origin_signed_mm = float(np.max(camera_signed) + 50.0)
    direction_world = -np.asarray(plane.normal_world, dtype=np.float64)
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
        wall = _wall_plane_world_grid(canvas, plane, x_values, y_values)
        origins_mm = wall + origin_signed_mm * plane.normal_world
        origins_m = origins_mm.astype(np.float32) / 1000.0
        rays = np.concatenate(
            (
                origins_m,
                np.broadcast_to(direction_world.astype(np.float32), origins_m.shape),
            ),
            axis=-1,
        ).reshape(-1, 6)
        answers = scene.cast_rays(o3d.core.Tensor(rays))
        primitive_ids = answers["primitive_ids"].numpy()
        hit = primitive_ids != invalid_primitive
        if not np.any(hit):
            continue
        local_indices = np.flatnonzero(hit)
        t_hit_mm = answers["t_hit"].numpy()[hit].astype(np.float64) * 1000.0
        origins_flat = origins_mm.reshape(-1, 3)[hit]
        hit_world = origins_flat + t_hit_mm[:, None] * direction_world
        normals = None
        if "primitive_normals" in answers:
            normals = answers["primitive_normals"].numpy()[hit]
        if normals is None or normals.shape != hit_world.shape:
            triangle = triangles[primitive_ids[hit]]
            a = vertices_m[triangle[:, 0]]
            b = vertices_m[triangle[:, 1]]
            c = vertices_m[triangle[:, 2]]
            normals = np.cross(b - a, c - a)
        normal_length = np.linalg.norm(normals, axis=1)
        valid_normals = normal_length > 1e-9
        normals[valid_normals] /= normal_length[valid_normals, None]
        local_world = world_points[row_start:row_stop].reshape(-1, 3)
        local_normal = normal_world[row_start:row_stop].reshape(-1, 3)
        local_surface = surface_depth[row_start:row_stop].reshape(-1)
        local_triangle = triangle_id[row_start:row_stop].reshape(-1)
        local_mask = hard_mask[row_start:row_stop].reshape(-1)
        local_hit = surface_hit_mask[row_start:row_stop].reshape(-1)
        local_world[local_indices] = hit_world.astype(np.float32)
        local_normal[local_indices] = normals.astype(np.float32)
        local_surface[local_indices] = (
            hit_world @ np.asarray(canvas.normal_axis, dtype=np.float64)
        ).astype(np.float32)
        local_triangle[local_indices] = primitive_ids[hit].astype(np.int32)
        local_hit[local_indices] = True
        local_mask[local_indices] = (
            plane.signed_distance_mm(hit_world) >= config.foreground_offset_mm
        )
    return ForegroundGeometry(
        hard_mask=hard_mask,
        surface_hit_mask=surface_hit_mask,
        world_points_mm=world_points,
        normal_world=normal_world,
        surface_depth_mm=surface_depth,
        triangle_id=triangle_id,
    )


def _front_surface_mode_shift_mm(
    geometry: ForegroundGeometry,
    plane: WallPlaneModel,
    config: DenseFusionConfig,
) -> float:
    """Find the dominant first-hit wall layer along the camera-side normal.

    TSDFs made from close RGB-D side scans can retain several nearly parallel
    wall layers.  Rendering always sees the *first* layer, so wall ownership
    must be referenced to that layer rather than to a deeper mesh band that
    happens to win a vertex histogram.
    """

    hit = geometry.surface_hit_mask
    if not np.any(hit):
        raise RuntimeError("Dense TSDF raycast found no surface for wall-layer selection")
    normals = geometry.normal_world[hit].astype(np.float64)
    valid_normal = np.isfinite(normals).all(axis=1)
    alignment = np.zeros(len(normals), dtype=np.float64)
    if np.any(valid_normal):
        alignment[valid_normal] = np.abs(normals[valid_normal] @ plane.normal_world)
    points = geometry.world_points_mm[hit].astype(np.float64)
    signed = plane.signed_distance_mm(points)
    usable = np.isfinite(signed) & (alignment >= math.cos(math.radians(35.0)))
    values = signed[usable]
    if values.size < 256:
        values = signed[np.isfinite(signed)]
    if values.size < 256:
        raise RuntimeError("Dense TSDF has too few first-hit wall candidates")
    low, high = np.percentile(values, (1.0, 99.0))
    bin_width = min(10.0, config.plane_histogram_bin_mm)
    edges = np.arange(low, high + bin_width, bin_width, dtype=np.float64)
    if edges.size < 2:
        return float(np.median(values))
    counts, edges = np.histogram(values, bins=edges)
    return float((edges[int(np.argmax(counts))] + edges[int(np.argmax(counts)) + 1]) * 0.5)


def _shift_plane_to_first_surface(
    plane: WallPlaneModel, shift_mm: float
) -> WallPlaneModel:
    """Move the fitted wall plane toward camera-side first-hit geometry."""

    return WallPlaneModel(
        normal_world=plane.normal_world,
        # Moving a plane by +normal*shift changes n·X+d=0 to d-shift.
        offset_mm=float(plane.offset_mm - shift_mm),
        rmse_mm=plane.rmse_mm,
        p95_residual_mm=plane.p95_residual_mm,
        inlier_fraction=plane.inlier_fraction,
        inlier_count=plane.inlier_count,
        tilt_deg=plane.tilt_deg,
    )


def _reclassify_foreground_geometry(
    geometry: ForegroundGeometry,
    plane: WallPlaneModel,
    config: DenseFusionConfig,
) -> ForegroundGeometry:
    hard_mask = np.zeros(geometry.surface_hit_mask.shape, dtype=bool)
    hit = geometry.surface_hit_mask
    if np.any(hit):
        hard_mask[hit] = (
            plane.signed_distance_mm(geometry.world_points_mm[hit])
            >= config.foreground_offset_mm
        )
    return ForegroundGeometry(
        hard_mask=hard_mask,
        surface_hit_mask=geometry.surface_hit_mask,
        world_points_mm=geometry.world_points_mm,
        normal_world=geometry.normal_world,
        surface_depth_mm=geometry.surface_depth_mm,
        triangle_id=geometry.triangle_id,
    )


def _project_foreground_silhouette_to_source(
    geometry: ForegroundGeometry,
    source: PreparedDenseSource,
    intrinsics: PinholeIntrinsics,
    config: DenseFusionConfig,
    *,
    confirmed_foreground: np.ndarray | None = None,
    allow_invalid_depth: bool = False,
) -> np.ndarray:
    """Project source-visible, depth-consistent TSDF foreground into one image.

    A canvas point is not automatically visible in every source camera.  In
    particular, projecting all orthographic hits without a source-view
    z-buffer lets a rear TSDF surface paint a foreground silhouette through a
    closer wall/object observation.  Such a silhouette then over-excludes a
    large part of the wall texture layer.  We instead retain the nearest
    projected TSDF hit per source pixel and require its measured aligned depth
    to agree.  A depth-invalid source pixel is protected only after the canvas
    point has already gained an independently verified foreground texture
    owner; it can never seed foreground evidence by itself.
    """

    silhouette = np.zeros(source.depth_mm.shape, dtype=bool)
    source_foreground = np.asarray(geometry.hard_mask, dtype=bool)
    if confirmed_foreground is not None:
        confirmed = np.asarray(confirmed_foreground, dtype=bool)
        if confirmed.shape != source_foreground.shape:
            raise ValueError("Confirmed foreground mask must match TSDF geometry")
        source_foreground &= confirmed
    points = geometry.world_points_mm[source_foreground]
    if not points.size:
        return silhouette
    pose = source.camera_to_world
    camera = (points.astype(np.float64) - pose[:3, 3]) @ pose[:3, :3]
    z = camera[:, 2]
    safe_z = np.where(z > 1e-6, z, 1.0)
    u = intrinsics.fx * camera[:, 0] / safe_z + intrinsics.cx
    v = intrinsics.fy * camera[:, 1] / safe_z + intrinsics.cy
    x = np.rint(u).astype(np.int64)
    y = np.rint(v).astype(np.int64)
    valid = (
        (z > 1e-6)
        & (x >= 0)
        & (x < intrinsics.width)
        & (y >= 0)
        & (y < intrinsics.height)
    )
    if not np.any(valid):
        return silhouette
    x = x[valid]
    y = y[valid]
    z = z[valid]
    flat = y * intrinsics.width + x
    # Source-view z-buffer: only the first TSDF hit on a source pixel may
    # define an exclusion/foreground association for that source.
    zbuffer = np.full(source.depth_mm.size, np.inf, dtype=np.float64)
    np.minimum.at(zbuffer, flat, z)
    unique_flat = np.unique(flat)
    winner_depth = zbuffer[unique_flat]
    measured_depth = source.depth_mm.reshape(-1)[unique_flat].astype(np.float64)
    measured_valid = (
        source.undistort_valid.reshape(-1)[unique_flat]
        & np.isfinite(measured_depth)
        & (measured_depth > 0.0)
        & (measured_depth <= config.maximum_depth_mm)
    )
    tolerance = np.clip(
        np.maximum(config.foreground_texture_depth_tolerance_mm, winner_depth * 0.02),
        config.foreground_texture_depth_tolerance_mm,
        40.0,
    )
    same_surface = measured_valid & (np.abs(measured_depth - winner_depth) <= tolerance)
    accepted = same_surface
    if allow_invalid_depth and confirmed_foreground is not None:
        # The geometry has a real texture owner elsewhere, so a depth hole in
        # this particular source must not let its wall RGB bleed through.
        accepted |= ~measured_valid & source.undistort_valid.reshape(-1)[unique_flat]
    silhouette.reshape(-1)[unique_flat[accepted]] = True
    return silhouette


def _sample_nearest_points(image: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Nearest-neighbour source samples without OpenCV's 32k remap limit."""

    source = np.asarray(image)
    height, width = source.shape[:2]
    x = np.rint(u).astype(np.int64)
    y = np.rint(v).astype(np.int64)
    inside = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    output_shape = (len(x), *source.shape[2:])
    output = np.zeros(output_shape, dtype=source.dtype)
    output[inside] = source[y[inside], x[inside]]
    return output


def _sample_bilinear_bgr_points(
    image: np.ndarray, u: np.ndarray, v: np.ndarray
) -> np.ndarray:
    """Bilinear BGR samples for arbitrarily many independently projected points."""

    source = np.asarray(image)
    height, width = source.shape[:2]
    x0 = np.floor(u).astype(np.int64)
    y0 = np.floor(v).astype(np.int64)
    inside = (x0 >= 0) & (x0 + 1 < width) & (y0 >= 0) & (y0 + 1 < height)
    output = np.zeros((len(x0), 3), dtype=np.uint8)
    if not np.any(inside):
        return output
    x = x0[inside]
    y = y0[inside]
    dx = (u[inside] - x).astype(np.float32)
    dy = (v[inside] - y).astype(np.float32)
    top = source[y, x].astype(np.float32) * (1.0 - dx[:, None]) + source[
        y, x + 1
    ].astype(np.float32) * dx[:, None]
    bottom = source[y + 1, x].astype(np.float32) * (1.0 - dx[:, None]) + source[
        y + 1, x + 1
    ].astype(np.float32) * dx[:, None]
    output[inside] = np.clip(
        np.rint(top * (1.0 - dy[:, None]) + bottom * dy[:, None]), 0.0, 255.0
    ).astype(np.uint8)
    return output


def _foreground_source_candidates(
    points_world_mm: np.ndarray,
    normals_world: np.ndarray,
    source: PreparedDenseSource,
    masks: SourceForegroundMasks,
    intrinsics: PinholeIntrinsics,
    config: DenseFusionConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return accepted texture samples, BGR values, and geometry-confidence score."""

    pose = source.camera_to_world
    camera = (points_world_mm.astype(np.float64) - pose[:3, 3]) @ pose[:3, :3]
    z = camera[:, 2]
    safe_z = np.where(z > 1e-6, z, 1.0)
    u = intrinsics.fx * camera[:, 0] / safe_z + intrinsics.cx
    v = intrinsics.fy * camera[:, 1] / safe_z + intrinsics.cy
    in_frame = (
        (z > 1e-6)
        & (u >= 0.0)
        & (u < intrinsics.width - 1)
        & (v >= 0.0)
        & (v < intrinsics.height - 1)
    )
    observed_depth = _sample_nearest_points(source.depth_mm, u, v)
    observed_valid = _sample_nearest_points(masks.depth_valid, u, v)
    foreground_core = _sample_nearest_points(masks.foreground_core, u, v)
    tsdf_silhouette = _sample_nearest_points(masks.tsdf_silhouette, u, v)
    undistort_valid = _sample_nearest_points(masks.undistort_valid, u, v)
    tolerance = np.clip(
        np.maximum(config.foreground_texture_depth_tolerance_mm, z * 0.02),
        config.foreground_texture_depth_tolerance_mm,
        40.0,
    )
    residual = np.abs(observed_depth - z)
    measured_same_surface = observed_valid & (residual <= tolerance)
    # A depth-hole colour sample is permitted only where the independently
    # rayed TSDF foreground projects to that source pixel.  It cannot turn a
    # generic depth-invalid white wall into foreground texture.
    allowed = (foreground_core & measured_same_surface) | (
        tsdf_silhouette & ~observed_valid
    )
    accepted = in_frame & undistort_valid & allowed
    color = _sample_bilinear_bgr_points(source.color, u, v)
    camera_center = pose[:3, 3]
    view = camera_center[None, :] - points_world_mm
    view_length = np.linalg.norm(view, axis=1)
    valid_normal = np.isfinite(normals_world).all(axis=1) & (view_length > 1e-6)
    incidence = np.ones(len(points_world_mm), dtype=np.float64)
    if np.any(valid_normal):
        unit_view = view[valid_normal] / view_length[valid_normal, None]
        incidence[valid_normal] = np.abs(
            np.sum(normals_world[valid_normal] * unit_view, axis=1)
        )
    center_distance = np.sqrt(
        ((u - intrinsics.cx) / max(intrinsics.width * 0.5, 1.0)) ** 2
        + ((v - intrinsics.cy) / max(intrinsics.height * 0.5, 1.0)) ** 2
    )
    center = np.clip(1.0 - center_distance / math.sqrt(2.0), 0.0, 1.0)
    depth_score = np.where(
        observed_valid,
        np.exp(-residual / np.maximum(tolerance, 1e-6)),
        0.8,
    )
    score = (
        math.log1p(max(source.sharpness, 0.0))
        * np.maximum(incidence, 0.1) ** 2
        * (0.35 + 0.65 * center)
        * depth_score
    ).astype(np.float32)
    score[~accepted] = -np.inf
    return accepted, color, score


def _texture_foreground_from_real_frames(
    geometry: ForegroundGeometry,
    all_frames: Sequence[RGBDFrame],
    all_poses: Sequence[np.ndarray],
    intrinsics: PinholeIntrinsics,
    plane: WallPlaneModel,
    maps: tuple[np.ndarray, np.ndarray] | None,
    config: DenseFusionConfig,
    *,
    wall_relative_cache: dict[int, tuple[np.ndarray, float, float]] | None = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, object],
]:
    """Texture hard TSDF foreground from the sharpest valid real RGB-D source.

    This routine intentionally has no access to ``mesh.vertex_colors``.  A
    candidate is valid only when its actual depth matches the rayed TSDF
    surface, or when the same TSDF foreground explicitly explains a local
    source-depth hole.
    """

    if len(all_frames) != len(all_poses):
        raise ValueError("Dense foreground texture requires aligned frame poses")
    height, width = geometry.hard_mask.shape
    output = np.zeros((height, width, 3), dtype=np.uint8)
    valid_output = np.zeros((height, width), dtype=bool)
    source_id = np.full((height, width), -1, dtype=np.int32)
    confidence = np.zeros((height, width), dtype=np.float32)
    support_count = np.zeros((height, width), dtype=np.uint8)
    active_flat = np.flatnonzero(geometry.hard_mask.reshape(-1))
    if not active_flat.size:
        return output, valid_output, source_id, confidence, support_count, {
            "foreground_texture_candidate_source_count": 0,
            "foreground_texture_from_mesh_vertex_color_count": 0,
            "foreground_texture_untextured_pixel_count": 0,
            "foreground_component_count": 0,
            "foreground_component_primary_source_count": 0,
        }
    points = geometry.world_points_mm.reshape(-1, 3)[active_flat]
    normals = geometry.normal_world.reshape(-1, 3)[active_flat].astype(np.float64)
    component_count, labels = cv2.connectedComponents(
        geometry.hard_mask.astype(np.uint8), connectivity=8
    )
    active_labels = labels.reshape(-1)[active_flat]
    component_total = np.bincount(active_labels, minlength=component_count)
    best_score = np.full(active_flat.size, -np.inf, dtype=np.float32)
    best_color = np.zeros((active_flat.size, 3), dtype=np.uint8)
    best_source = np.full(active_flat.size, -1, dtype=np.int32)
    best_support = np.zeros(active_flat.size, dtype=np.uint8)
    # Arrays indexed [source-index, component-label] permit a second, stable
    # component pass without retaining every full-resolution source frame.
    coverage = np.zeros((len(all_frames), component_count), dtype=np.int32)
    mean_score = np.zeros((len(all_frames), component_count), dtype=np.float64)
    candidate_sources = 0
    source_stats: list[dict[str, object]] = []

    def process_source(index: int) -> tuple[
        PreparedDenseSource, SourceForegroundMasks, np.ndarray, np.ndarray, np.ndarray
    ] | None:
        prepared = _prepare_session_source(all_frames[index], all_poses[index], maps)
        # Skip an expensive source-space residual fit if this camera cannot see
        # enough target geometry at all.
        camera = (
            points.astype(np.float64) - prepared.camera_to_world[:3, 3]
        ) @ prepared.camera_to_world[:3, :3]
        z = camera[:, 2]
        u = intrinsics.fx * camera[:, 0] / np.where(z > 1e-6, z, 1.0) + intrinsics.cx
        v = intrinsics.fy * camera[:, 1] / np.where(z > 1e-6, z, 1.0) + intrinsics.cy
        if int(
            np.count_nonzero(
                (z > 1e-6)
                & (u >= 0.0)
                & (u < intrinsics.width - 1)
                & (v >= 0.0)
                & (v < intrinsics.height - 1)
            )
        ) < 16:
            return None
        silhouette = _project_foreground_silhouette_to_source(
            geometry, prepared, intrinsics, config
        )
        try:
            masks = _classify_source_foreground(
                prepared,
                intrinsics,
                plane,
                config,
                tsdf_silhouette=silhouette,
                wall_relative_cache=wall_relative_cache,
            )
        except RuntimeError:
            return None
        accepted, color, score = _foreground_source_candidates(
            points, normals, prepared, masks, intrinsics, config
        )
        return prepared, masks, accepted, color, score

    for index in range(len(all_frames)):
        sampled = process_source(index)
        if sampled is None:
            continue
        prepared, masks, accepted, color, score = sampled
        candidate_sources += 1
        if np.any(accepted):
            best_support[accepted] = np.minimum(
                np.iinfo(np.uint8).max, best_support[accepted].astype(np.uint16) + 1
            ).astype(np.uint8)
            bins = np.bincount(
                active_labels[accepted], minlength=component_count
            )
            coverage[index] = bins
            score_bins = np.bincount(
                active_labels[accepted], weights=score[accepted], minlength=component_count
            )
            with np.errstate(divide="ignore", invalid="ignore"):
                mean_score[index] = np.divide(
                    score_bins,
                    bins,
                    out=np.zeros_like(score_bins),
                    where=bins > 0,
                )
            replace = accepted & (score > best_score)
            best_score[replace] = score[replace]
            best_color[replace] = color[replace]
            best_source[replace] = prepared.frame_id
        source_stats.append(
            {
                "frame_id": prepared.frame_id,
                "foreground_core_pixel_count": int(np.count_nonzero(masks.foreground_core)),
                "unknown_band_pixel_count": int(np.count_nonzero(masks.unknown_band)),
                "background_safe_pixel_count": int(np.count_nonzero(masks.background_safe)),
                "wall_residual_rmse_mm": masks.wall_residual_rmse_mm,
                "wall_residual_p95_mm": masks.wall_residual_p95_mm,
            }
        )

    primary_by_component = np.full(component_count, -1, dtype=np.int32)
    for label in range(1, component_count):
        total = int(component_total[label])
        if not total:
            continue
        source_indices = np.flatnonzero(
            coverage[:, label]
            >= math.ceil(config.foreground_component_single_source_coverage * total)
        )
        if source_indices.size:
            primary_by_component[label] = int(
                source_indices[np.argmax(mean_score[source_indices, label])]
            )
    # Enforce one real sharp source across a component whenever it covers the
    # configured fraction.  Re-evaluation avoids caching 101 RGB images.
    primary_count = int(np.count_nonzero(primary_by_component[1:] >= 0))
    for index in np.unique(primary_by_component[primary_by_component >= 0]):
        sampled = process_source(int(index))
        if sampled is None:
            continue
        prepared, _, accepted, color, score = sampled
        primary = primary_by_component[active_labels] == int(index)
        replace = primary & accepted
        best_score[replace] = score[replace]
        best_color[replace] = color[replace]
        best_source[replace] = prepared.frame_id

    # A large valid foreground component can legitimately span several camera
    # views, so the 95% component rule above intentionally does not force one
    # source across it.  Pixel-by-pixel score switching, however, produces
    # visible triangular mosaic texture on a TSDF surface.  Stabilise only the
    # remaining components in small metric canvas tiles: each tile may choose
    # one already verified real source when it accounts for a meaningful share
    # of that tile.  A second geometric candidate check prevents this from
    # copying colour into a pixel the selected source cannot actually see.
    component_has_primary = primary_by_component[active_labels] >= 0
    tile_primary = np.full(active_flat.size, -1, dtype=np.int32)
    tile_group_count = 0
    if np.any(~component_has_primary):
        active_rows = active_flat // width
        active_columns = active_flat % width
        tile_columns = int(
            math.ceil(width / float(config.foreground_texture_tile_size_px))
        )
        tile_index = (
            (active_rows // config.foreground_texture_tile_size_px) * tile_columns
            + active_columns // config.foreground_texture_tile_size_px
        )
        eligible = ~component_has_primary
        group_key = (
            active_labels[eligible].astype(np.int64) * tile_columns * int(
                math.ceil(height / float(config.foreground_texture_tile_size_px))
            )
            + tile_index[eligible].astype(np.int64)
        )
        unique_group, group_inverse = np.unique(group_key, return_inverse=True)
        group_total = np.bincount(group_inverse, minlength=len(unique_group))
        eligible_positions = np.flatnonzero(eligible)
        textured_eligible = best_source[eligible_positions] >= 0
        if np.any(textured_eligible):
            pair_positions = eligible_positions[textured_eligible]
            pair_groups = group_inverse[textured_eligible]
            pair_sources = best_source[pair_positions]
            order = np.lexsort((pair_sources, pair_groups))
            ordered_groups = pair_groups[order]
            ordered_sources = pair_sources[order]
            starts = np.r_[0, np.flatnonzero(
                (np.diff(ordered_groups) != 0)
                | (np.diff(ordered_sources) != 0)
            ) + 1]
            stops = np.r_[starts[1:], len(order)]
            dominant_count = np.zeros(len(unique_group), dtype=np.int32)
            dominant_source = np.full(len(unique_group), -1, dtype=np.int32)
            for start, stop in zip(starts, stops, strict=True):
                group = int(ordered_groups[start])
                count = int(stop - start)
                source = int(ordered_sources[start])
                if count > dominant_count[group]:
                    dominant_count[group] = count
                    dominant_source[group] = source
            required = np.ceil(
                config.foreground_texture_tile_primary_coverage * group_total
            ).astype(np.int32)
            accepted_groups = dominant_count >= required
            selected_by_group = np.where(accepted_groups, dominant_source, -1)
            tile_primary[eligible_positions] = selected_by_group[group_inverse]
            tile_group_count = int(np.count_nonzero(accepted_groups))
            source_index_by_frame_id = {
                frame.frame_id: index for index, frame in enumerate(all_frames)
            }
            for frame_id in np.unique(tile_primary[tile_primary >= 0]):
                index = source_index_by_frame_id.get(int(frame_id))
                if index is None:
                    continue
                sampled = process_source(index)
                if sampled is None:
                    continue
                prepared, _, accepted, color, score = sampled
                replace = (tile_primary == prepared.frame_id) & accepted
                best_score[replace] = score[replace]
                best_color[replace] = color[replace]
                best_source[replace] = prepared.frame_id

    textured = best_source >= 0
    output.reshape(-1, 3)[active_flat[textured]] = best_color[textured]
    valid_output.reshape(-1)[active_flat[textured]] = True
    source_id.reshape(-1)[active_flat[textured]] = best_source[textured]
    support_count.reshape(-1)[active_flat[textured]] = best_support[textured]
    finite_score = best_score[textured]
    if finite_score.size:
        scale = float(np.percentile(finite_score, 95.0))
        confidence.reshape(-1)[active_flat[textured]] = np.clip(
            finite_score / max(scale, 1e-6), 0.0, 1.0
        )
    return output, valid_output, source_id, confidence, support_count, {
        "foreground_texture_candidate_source_count": candidate_sources,
        "foreground_texture_from_mesh_vertex_color_count": 0,
        "foreground_texture_untextured_pixel_count": int(np.count_nonzero(~textured)),
        "foreground_component_count": int(component_count - 1),
        "foreground_component_primary_source_count": primary_count,
        "foreground_texture_tile_primary_group_count": tile_group_count,
        "foreground_texture_multiview_support_p50": float(
            np.median(best_support[textured]) if np.any(textured) else 0.0
        ),
        "foreground_texture_multiview_support_p95": float(
            np.percentile(best_support[textured], 95.0) if np.any(textured) else 0.0
        ),
        "foreground_source_mask_summary": source_stats,
    }


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
    """Find a contiguous crop whose *true* rows and columns meet coverage.

    This is deliberately not a bounding-box fallback.  If no rectangle meets
    the documented threshold, the capture has no safe crop and must fail
    rather than publish a tapered black border under a reassuring name.
    """

    visible = np.asarray(mask, dtype=bool)
    if not np.any(visible):
        raise RuntimeError("Dense RGB-D fusion has no visible crop fallback")
    x, y, width, height = cv2.boundingRect(visible.astype(np.uint8))
    if width <= 0 or height <= 0:
        raise RuntimeError("Dense RGB-D fusion has an empty visible crop fallback")
    row_indices = np.arange(y, y + height, dtype=np.int64)
    column_indices = np.arange(x, x + width, dtype=np.int64)
    # A row can become valid after weak columns are removed and vice versa, so
    # trim the longest qualifying contiguous spans to a fixed point.
    for _ in range(32):
        block = visible[np.ix_(row_indices, column_indices)]
        good_rows = _longest_contiguous_span(
            row_indices[np.mean(block, axis=1) >= minimum_coverage]
        )
        if good_rows is None:
            raise RuntimeError(
                "Dense RGB-D fusion has no row span meeting the visible crop threshold"
            )
        next_rows = np.arange(good_rows[0], good_rows[1] + 1, dtype=np.int64)
        block = visible[np.ix_(next_rows, column_indices)]
        good_columns = _longest_contiguous_span(
            column_indices[np.mean(block, axis=0) >= minimum_coverage]
        )
        if good_columns is None:
            raise RuntimeError(
                "Dense RGB-D fusion has no column span meeting the visible crop threshold"
            )
        next_columns = np.arange(
            good_columns[0], good_columns[1] + 1, dtype=np.int64
        )
        if np.array_equal(next_rows, row_indices) and np.array_equal(
            next_columns, column_indices
        ):
            break
        row_indices = next_rows
        column_indices = next_columns
    else:
        raise RuntimeError("Dense RGB-D visible crop refinement did not converge")
    candidate = visible[np.ix_(row_indices, column_indices)]
    if (
        np.any(np.mean(candidate, axis=1) < minimum_coverage)
        or np.any(np.mean(candidate, axis=0) < minimum_coverage)
    ):
        raise RuntimeError(
            "Dense RGB-D visible crop fallback does not meet its row/column threshold"
        )
    return (
        int(column_indices[0]),
        int(row_indices[0]),
        int(len(column_indices)),
        int(len(row_indices)),
    )


def _safe_visible_mask(valid: np.ndarray, config: DenseFusionConfig) -> np.ndarray:
    """Build crop support from true validity without bridging exterior gaps.

    Only internal invalid islands up to four pixels are filled for rectangle
    selection.  The returned mask never becomes a texture/projection validity
    claim; callers continue to retain the original true valid mask separately.
    """

    support = np.asarray(valid, dtype=bool).copy()
    if config.mask_close_kernel_size > 1:
        invalid = ~support
        component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
            invalid.astype(np.uint8), connectivity=8
        )
        if component_count > 1:
            boundary_labels = np.unique(
                np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1]))
            )
            for label in range(1, component_count):
                if label in boundary_labels:
                    continue
                if int(stats[label, cv2.CC_STAT_AREA]) <= 4:
                    support[labels == label] = True
    binary = support.astype(np.uint8) * 255
    if config.mask_erode_pixels:
        binary = cv2.erode(
            binary,
            np.ones((3, 3), dtype=np.uint8),
            iterations=config.mask_erode_pixels,
            borderType=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
    return binary > 0


def _wall_plane_fov_domain(
    sources: Sequence[PreparedDenseSource],
    intrinsics: PinholeIntrinsics,
    canvas: ProjectionCanvas,
    plane: WallPlaneModel,
    config: DenseFusionConfig,
) -> tuple[tuple[int, int, int, int], dict[str, object]]:
    """Find the true, rectangular wall domain shared by real source FOVs.

    The ends of a side scan naturally form a tapered frustum on the wall.  It
    is neither a black-RGB problem nor foreground geometry, so asking the
    *semantic* composite to discover a rectangle from that full taper makes a
    handful of object exclusions collapse the crop to a narrow strip.  This
    helper first derives the active wall domain solely from each actual camera
    FOV and its undistort-valid mask.  It does not claim an unobserved pixel is
    valid; final cropping still uses the true composite visibility inside this
    domain.
    """

    if not sources:
        raise ValueError("Dense wall FOV domain requires at least one real source")
    height, width = canvas.height, canvas.width
    fov_visible = np.zeros((height, width), dtype=bool)
    min_scan, min_down, _, _ = canvas.world_bounds
    scan_values = min_scan + (
        np.arange(width, dtype=np.float64) + 0.5
    ) / canvas.pixels_per_mm
    for source in sources:
        pose = source.camera_to_world
        rotation = pose[:3, :3]
        translation = pose[:3, 3]
        for row_start in range(0, height, 128):
            row_stop = min(height, row_start + 128)
            down_values = min_down + (
                np.arange(row_start, row_stop, dtype=np.float64) + 0.5
            ) / canvas.pixels_per_mm
            world = _wall_plane_world_grid(canvas, plane, scan_values, down_values)
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
            source_valid = cv2.remap(
                source.undistort_valid.astype(np.uint8),
                u.astype(np.float32),
                v.astype(np.float32),
                cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            ) > 0
            fov_visible[row_start:row_stop] |= in_frame & source_valid
    if not np.any(fov_visible):
        raise RuntimeError("Dense wall plane is outside every true source FOV")
    safe = _safe_visible_mask(fov_visible, config)
    strict = _largest_inscribed_rectangle(safe)
    if strict[2] <= 0 or strict[3] <= 0:
        raise RuntimeError("Dense wall FOV has no safe rectangular domain")
    fallback = _fallback_visible_rectangle(
        fov_visible, config.minimum_row_or_column_coverage
    )
    if fallback[2] * fallback[3] > strict[2] * strict[3]:
        domain = fallback
        strategy = "row_column_visible_coverage_fallback"
    else:
        domain = strict
        strategy = "maximum_safe_inscribed_rectangle"
    bounds = cv2.boundingRect(fov_visible.astype(np.uint8))
    x, y, domain_width, domain_height = domain
    domain_mask = fov_visible[y : y + domain_height, x : x + domain_width]
    return domain, {
        "strategy": strategy,
        "source_count": len(sources),
        "true_fov_pixel_count": int(np.count_nonzero(fov_visible)),
        "true_fov_bounds": {
            "x": int(bounds[0]),
            "y": int(bounds[1]),
            "width": int(bounds[2]),
            "height": int(bounds[3]),
        },
        "domain": {
            "x": int(x),
            "y": int(y),
            "width": int(domain_width),
            "height": int(domain_height),
        },
        "domain_true_fov_coverage": float(np.mean(domain_mask)),
        "strict_candidate": {
            "x": int(strict[0]),
            "y": int(strict[1]),
            "width": int(strict[2]),
            "height": int(strict[3]),
        },
        "fallback_candidate": {
            "x": int(fallback[0]),
            "y": int(fallback[1]),
            "width": int(fallback[2]),
            "height": int(fallback[3]),
        },
    }


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


def _narrow_foreground_alpha(
    foreground_mask: np.ndarray,
    canvas: ProjectionCanvas,
    config: DenseFusionConfig,
) -> tuple[np.ndarray, int]:
    """Create only a narrow antialiasing band; never feather an object interior."""

    binary = np.asarray(foreground_mask, dtype=np.uint8)
    if not np.any(binary):
        return np.zeros(binary.shape, dtype=np.float32), 0
    radius = int(
        min(
            config.maximum_foreground_alpha_radius_px,
            max(
                1,
                round(
                    config.foreground_alpha_radius_px_at_640
                    * canvas.width
                    / 640.0
                ),
            ),
        )
    )
    distance = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
    alpha = binary.astype(np.float32)
    # Just the first ``radius`` interior pixels are softened.  A thin
    # component is restored fully opaque below, so a hose cannot vanish.
    edge = (binary > 0) & (distance < radius)
    alpha[edge] = 0.75 + 0.25 * distance[edge] / max(radius, 1)
    component_count, labels = cv2.connectedComponents(binary, connectivity=8)
    for label in range(1, component_count):
        component = labels == label
        if float(distance[component].max(initial=0.0)) < radius:
            alpha[component] = 1.0
    return alpha, radius


def _propagate_verified_foreground_texture(
    hard_geometry: np.ndarray,
    textured: np.ndarray,
    color: np.ndarray,
    source_id: np.ndarray,
    confidence: np.ndarray,
    *,
    maximum_steps: int = 4,
) -> np.ndarray:
    """Close only tiny texture gaps from adjacent verified foreground texels.

    This is not image inpainting and cannot pull wall texture through an
    object: propagation is one pixel at a time, remains inside the hard TSDF
    owner, and copies a real source ID along with the source RGB value.
    """

    owner = np.asarray(hard_geometry, dtype=bool)
    known = np.asarray(textured, dtype=bool).copy()
    recovered = np.zeros(owner.shape, dtype=bool)
    height, width = owner.shape
    for _ in range(maximum_steps):
        missing = owner & ~known
        if not np.any(missing):
            break
        updated = np.zeros(owner.shape, dtype=bool)
        for row_offset, column_offset in (
            (0, 1),
            (0, -1),
            (1, 0),
            (-1, 0),
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1),
        ):
            source_rows = slice(max(0, -row_offset), min(height, height - row_offset))
            source_columns = slice(
                max(0, -column_offset), min(width, width - column_offset)
            )
            target_rows = slice(max(0, row_offset), min(height, height + row_offset))
            target_columns = slice(
                max(0, column_offset), min(width, width + column_offset)
            )
            source_known = known[source_rows, source_columns]
            target_missing = missing[target_rows, target_columns] & ~updated[
                target_rows, target_columns
            ]
            take = target_missing & source_known
            if not np.any(take):
                continue
            target_color = color[target_rows, target_columns]
            target_source = source_id[target_rows, target_columns]
            target_confidence = confidence[target_rows, target_columns]
            source_color = color[source_rows, source_columns]
            source_source = source_id[source_rows, source_columns]
            source_confidence = confidence[source_rows, source_columns]
            target_color[take] = source_color[take]
            target_source[take] = source_source[take]
            target_confidence[take] = np.minimum(source_confidence[take], 0.6)
            updated[target_rows, target_columns] |= take
        if not np.any(updated):
            break
        known |= updated
        recovered |= updated
    return recovered


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

    true_visible = np.asarray(valid, dtype=bool)
    if not np.any(true_visible):
        raise RuntimeError("Dense RGB-D fusion produced no visible panorama pixels")
    safe_visible = _safe_visible_mask(true_visible, config)
    if not np.any(safe_visible):
        raise RuntimeError("Dense RGB-D fusion has no visible pixels after safe inset")
    true_bounds = cv2.boundingRect(true_visible.astype(np.uint8))
    support_bounds = cv2.boundingRect(safe_visible.astype(np.uint8))
    strict_crop = _largest_inscribed_rectangle(safe_visible)
    minimum_height = config.minimum_inscribed_crop_height_fraction * true_bounds[3]
    if strict_crop[3] >= minimum_height:
        crop = strict_crop
        strategy = "maximum_safe_inscribed_rectangle"
    else:
        fallback_crop = _fallback_visible_rectangle(
            true_visible, config.minimum_row_or_column_coverage
        )
        # This is the documented height-preserving fallback.  Do not silently
        # replace it with a smaller strict rectangle merely because that
        # rectangle has a marginally larger area: the caller explicitly asks
        # for the 98% row/column rule when the loss of strict crop height is
        # excessive.  Formal publication still checks the resulting extent
        # and true validity coverage below.
        crop = fallback_crop
        strategy = "row_column_visible_coverage_fallback"
    x, y, width, height = crop
    if width <= 0 or height <= 0:
        raise RuntimeError("Dense RGB-D fusion produced an empty crop")
    cropped_visible = true_visible[y : y + height, x : x + width]
    cropped_support = safe_visible[y : y + height, x : x + width]
    return (
        image[y : y + height, x : x + width],
        cropped_visible,
        foreground_mask[y : y + height, x : x + width],
        crop,
        {
            "strategy": strategy,
            "true_visible_pixel_count": int(np.count_nonzero(true_visible)),
            "safe_visible_pixel_count": int(np.count_nonzero(safe_visible)),
            "true_crop_coverage": float(np.mean(cropped_visible)),
            "crop_support_coverage": float(np.mean(cropped_support)),
            "crop_height_fraction": float(height / max(1, true_bounds[3])),
            "crop_width_fraction": float(width / max(1, true_bounds[2])),
            "fallback_threshold": float(config.minimum_row_or_column_coverage),
            "fallback_min_row_coverage": float(np.min(np.mean(cropped_visible, axis=1))),
            "fallback_min_column_coverage": float(
                np.min(np.mean(cropped_visible, axis=0))
            ),
            "true_visible_bounds": {
                "x": int(true_bounds[0]),
                "y": int(true_bounds[1]),
                "width": int(true_bounds[2]),
                "height": int(true_bounds[3]),
            },
            "safe_visible_bounds": {
                "x": int(support_bounds[0]),
                "y": int(support_bounds[1]),
                "width": int(support_bounds[2]),
                "height": int(support_bounds[3]),
            },
            "strict_crop": {
                "x": int(strict_crop[0]),
                "y": int(strict_crop[1]),
                "width": int(strict_crop[2]),
                "height": int(strict_crop[3]),
            },
            "fallback_crop": {
                "x": int(fallback_crop[0]) if strict_crop[3] < minimum_height else None,
                "y": int(fallback_crop[1]) if strict_crop[3] < minimum_height else None,
                "width": int(fallback_crop[2]) if strict_crop[3] < minimum_height else None,
                "height": int(fallback_crop[3]) if strict_crop[3] < minimum_height else None,
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
    """Build a hard-owned wall/foreground panorama from real RGB-D evidence."""

    selected_config = (
        config
        if isinstance(config, DenseFusionConfig)
        else DenseFusionConfig.from_mapping(config)
    )
    if len(all_depth_frames) != len(all_camera_to_world):
        raise ValueError("Dense RGB-D fusion requires one real pose per depth frame")
    maps = _undistortion_maps(intrinsics)
    mesh = _integrate_tsdf(
        all_depth_frames,
        all_camera_to_world,
        intrinsics,
        maps,
        selected_config,
    )
    fitted_plane = _fit_wall_plane_model(
        mesh, canvas, all_camera_to_world, selected_config
    )
    # Export may request Open3D to synthesise vertex normals.  Plane fitting
    # must use the reconstruction normals as extracted, not that display-only
    # side effect, or its candidate wall band changes with GLB export order.
    tsdf_mesh_glb = _mesh_to_glb(mesh)
    initial_geometry = (
        _raycast_foreground_geometry(
            mesh, canvas, fitted_plane, all_camera_to_world, selected_config
        )
        if selected_config.foreground_overlay
        else ForegroundGeometry(
            hard_mask=np.zeros((canvas.height, canvas.width), dtype=bool),
            surface_hit_mask=np.zeros((canvas.height, canvas.width), dtype=bool),
            world_points_mm=np.full(
                (canvas.height, canvas.width, 3), np.nan, dtype=np.float32
            ),
            normal_world=np.full(
                (canvas.height, canvas.width, 3), np.nan, dtype=np.float32
            ),
            surface_depth_mm=np.full(
                (canvas.height, canvas.width), np.nan, dtype=np.float32
            ),
            triangle_id=np.full((canvas.height, canvas.width), -1, dtype=np.int32),
        )
    )
    front_surface_shift_mm = (
        _front_surface_mode_shift_mm(initial_geometry, fitted_plane, selected_config)
        if selected_config.foreground_overlay
        else 0.0
    )
    plane = _shift_plane_to_first_surface(fitted_plane, front_surface_shift_mm)
    geometry = _reclassify_foreground_geometry(
        initial_geometry, plane, selected_config
    )

    tsdf_foreground_mask = geometry.hard_mask.copy()
    depth_fallback_mask = np.zeros_like(tsdf_foreground_mask)
    depth_multiview_foreground_mask = np.zeros_like(tsdf_foreground_mask)
    foreground_source_id = np.full(tsdf_foreground_mask.shape, -1, dtype=np.int32)
    foreground_confidence = np.zeros(tsdf_foreground_mask.shape, dtype=np.float32)
    foreground_rgb = np.zeros((*tsdf_foreground_mask.shape, 3), dtype=np.uint8)
    depth_point_metadata: dict[str, object] = {
        "supported_depth_point_count": 0,
        "foreground_candidate_point_count": 0,
        "foreground_zbuffer_pixel_count": 0,
        "foreground_zbuffer_updates": 0,
    }
    foreground_texture_metadata: dict[str, object] = {
        "foreground_texture_candidate_source_count": 0,
        "foreground_texture_from_mesh_vertex_color_count": 0,
        "foreground_texture_untextured_pixel_count": 0,
        "foreground_component_count": 0,
        "foreground_component_primary_source_count": 0,
    }
    local_tsdf_holes = np.zeros_like(tsdf_foreground_mask)
    depth_texture_rescue = np.zeros_like(tsdf_foreground_mask)
    neighbor_texture_recovery = np.zeros_like(tsdf_foreground_mask)
    # The same true frame is sampled for TSDF texture, raw-depth evidence and
    # wall recovery.  Its constrained local wall model is immutable for this
    # fusion run, so cache it by real frame ID rather than re-fitting RANSAC
    # several times and risking numerical drift between ownership stages.
    wall_relative_cache: dict[int, tuple[np.ndarray, float, float]] = {}
    if selected_config.foreground_overlay:
        (
            tsdf_rgb,
            tsdf_texture_valid,
            tsdf_texture_source_id,
            tsdf_confidence,
            tsdf_texture_support,
            foreground_texture_metadata,
        ) = _texture_foreground_from_real_frames(
            geometry,
            all_depth_frames,
            all_camera_to_world,
            intrinsics,
            plane,
            maps,
            selected_config,
            wall_relative_cache=wall_relative_cache,
        )
        (
            depth_rgb,
            depth_raw_valid,
            depth_source_id,
            _,
            depth_support_count,
            depth_support_baseline_mm,
            depth_point_metadata,
        ) = _depth_point_foreground_geometry(
            all_depth_frames,
            all_camera_to_world,
            intrinsics,
            canvas,
            plane,
            maps,
            selected_config,
            geometry,
            wall_relative_cache=wall_relative_cache,
        )
        depth_confirmed_foreground = depth_raw_valid & (
            depth_support_count
            >= selected_config.minimum_depth_foreground_multiview_support
        ) & (
            depth_support_baseline_mm
            >= selected_config.minimum_depth_foreground_baseline_mm
        )
        depth_multiview_foreground_mask = depth_confirmed_foreground
        # A rayed TSDF surface is the primary geometry layer.  It needs a real
        # source RGB-D texture witness at the same surface, but must *not* be
        # discarded merely because a second depth observation happens to land
        # in a different sub-pixel z-buffer bin.  That earlier per-pixel gate
        # turned a complete object into a few isolated speckles on the actual
        # extinguisher sequence.  Multi-view raw-depth evidence is retained as
        # an auditable, local-hole-only complement below.
        raw_tsdf_geometry = tsdf_foreground_mask.copy()
        texture_supported = tsdf_texture_valid & (
            tsdf_texture_support >= selected_config.minimum_foreground_multiview_support
        )
        tsdf_foreground_mask = raw_tsdf_geometry & texture_supported
        foreground_texture_metadata[
            "foreground_geometry_rejected_without_real_source_pixel_count"
        ] = int(np.count_nonzero(raw_tsdf_geometry & ~tsdf_foreground_mask))
        foreground_texture_metadata[
            "foreground_depth_confirmed_pixel_count"
        ] = int(np.count_nonzero(depth_confirmed_foreground))
        foreground_texture_metadata[
            "foreground_tsdf_rejected_without_depth_multiview_evidence_pixel_count"
        ] = int(
            np.count_nonzero(
                raw_tsdf_geometry
                & texture_supported
                & ~depth_confirmed_foreground
            )
        )
        foreground_texture_metadata["foreground_tsdf_raw_geometry_pixel_count"] = int(
            np.count_nonzero(raw_tsdf_geometry)
        )
        foreground_texture_metadata["foreground_tsdf_texture_supported_pixel_count"] = int(
            np.count_nonzero(raw_tsdf_geometry & texture_supported)
        )
        geometry = ForegroundGeometry(
            hard_mask=tsdf_foreground_mask,
            surface_hit_mask=geometry.surface_hit_mask,
            world_points_mm=geometry.world_points_mm,
            normal_world=geometry.normal_world,
            surface_depth_mm=geometry.surface_depth_mm,
            triangle_id=geometry.triangle_id,
        )
        texture_owner = tsdf_foreground_mask & texture_supported
        depth_texture_rescue = (
            tsdf_foreground_mask & ~texture_owner & depth_confirmed_foreground
        )
        foreground_rgb[texture_owner] = tsdf_rgb[texture_owner]
        foreground_source_id[texture_owner] = tsdf_texture_source_id[texture_owner]
        foreground_confidence[texture_owner] = tsdf_confidence[texture_owner]
        foreground_rgb[depth_texture_rescue] = depth_rgb[depth_texture_rescue]
        foreground_source_id[depth_texture_rescue] = depth_source_id[
            depth_texture_rescue
        ]
        foreground_confidence[depth_texture_rescue] = 0.75
        neighbor_texture_recovery = _propagate_verified_foreground_texture(
            tsdf_foreground_mask,
            texture_owner | depth_texture_rescue,
            foreground_rgb,
            foreground_source_id,
            foreground_confidence,
        )
        local_tsdf_holes = _tsdf_local_holes(
            tsdf_foreground_mask, selected_config.foreground_tsdf_hole_kernel_size
        )
        # A local TSDF hole may only be restored by an independently repeated
        # raw-depth foreground point.  The surrounding TSDF bounds where it
        # can land, while the baseline audit prevents one erroneous wall depth
        # sample from painting a coloured foreground island onto the wall.
        depth_fallback_mask = depth_confirmed_foreground & local_tsdf_holes
        foreground_rgb[depth_fallback_mask] = depth_rgb[depth_fallback_mask]
        foreground_source_id[depth_fallback_mask] = depth_source_id[
            depth_fallback_mask
        ]
        foreground_confidence[depth_fallback_mask] = 0.75

    # Only now is the foreground geometry backed by a real RGB-D source.  The
    # planar background may safely use its resulting silhouette as an exclusion
    # mask; rejected ghost geometry therefore releases wall texture rather than
    # punching a black hole through the panorama.
    prepared_background: list[PreparedDenseSource] = []
    background_masks: list[SourceForegroundMasks] = []
    background_source_failures: list[dict[str, object]] = []
    candidate_background_sources: list[PreparedDenseSource] = []
    seen_background_ids: set[int] = set()
    for frame in background_frames:
        source = _prepare_projection_source(frame, maps)
        if source.frame_id not in seen_background_ids:
            candidate_background_sources.append(source)
            seen_background_ids.add(source.frame_id)
    # The normal render selector may use fewer than the 32 real source nodes
    # allowed by the formal projection contract.  Add evenly distributed true
    # ORB poses only for wall recovery: another safe observation is preferable
    # to a black hole, while the cap remains strict and no pose is invented.
    frame_index_by_id = {
        frame.frame_id: index for index, frame in enumerate(all_depth_frames)
    }
    selected_indices = {
        frame_index_by_id[frame_id]
        for frame_id in seen_background_ids
        if frame_id in frame_index_by_id
    }
    # Farthest-point sampling fills the largest missing temporal baseline each
    # time, so adding 19 sources to an existing 13 spans the full scan instead
    # of accidentally concentrating on its first half.
    while (
        len(candidate_background_sources) < selected_config.maximum_background_sources
        and len(selected_indices) < len(all_depth_frames)
    ):
        indices = np.arange(len(all_depth_frames), dtype=np.int64)
        selected_array = np.fromiter(selected_indices, dtype=np.int64)
        if selected_array.size:
            distance = np.min(
                np.abs(indices[:, None] - selected_array[None, :]), axis=1
            )
            distance[list(selected_indices)] = -1
            index = int(np.argmax(distance))
        else:
            index = len(all_depth_frames) // 2
        frame = all_depth_frames[index]
        candidate_background_sources.append(
            _prepare_session_source(frame, all_camera_to_world[index], maps)
        )
        seen_background_ids.add(frame.frame_id)
        selected_indices.add(index)
    for source in candidate_background_sources:
        silhouette = _project_foreground_silhouette_to_source(
            geometry,
            source,
            intrinsics,
            selected_config,
            confirmed_foreground=tsdf_foreground_mask,
            allow_invalid_depth=True,
        )
        try:
            masks = _classify_source_foreground(
                source,
                intrinsics,
                plane,
                selected_config,
                tsdf_silhouette=silhouette,
                wall_relative_cache=wall_relative_cache,
            )
        except RuntimeError as exc:
            background_source_failures.append(
                {"frame_id": source.frame_id, "reason": str(exc)}
            )
            continue
        prepared_background.append(source)
        background_masks.append(masks)
    if not prepared_background:
        raise RuntimeError("No selected source has a reliable local wall-depth model")
    wall_fov_domain, wall_fov_domain_metadata = _wall_plane_fov_domain(
        prepared_background,
        intrinsics,
        canvas,
        plane,
        selected_config,
    )
    (
        background,
        background_valid,
        background_source_id,
        blocked_by_foreground,
        background_audit,
    ) = _dense_plane_background(
        prepared_background,
        background_masks,
        intrinsics,
        canvas,
        plane,
        selected_config,
    )

    foreground_mask = tsdf_foreground_mask | depth_fallback_mask
    foreground_unowned = foreground_mask & (foreground_source_id < 0)
    foreground_multiple_owner = tsdf_foreground_mask & depth_fallback_mask
    if np.any(foreground_unowned) or np.any(foreground_multiple_owner):
        raise RuntimeError(
            "Dense foreground owner audit failed: "
            f"unowned={int(np.count_nonzero(foreground_unowned))}, "
            f"multiple={int(np.count_nonzero(foreground_multiple_owner))}"
        )
    if np.any(background_valid & (background_source_id < 0)):
        raise RuntimeError("Dense background owner audit found a pixel without a source")
    alpha, alpha_radius = _narrow_foreground_alpha(
        foreground_mask, canvas, selected_config
    )
    composite = background.copy()
    owner = foreground_mask
    alpha_values = alpha[owner, None]
    composite[owner] = np.clip(
        foreground_rgb[owner].astype(np.float32) * alpha_values
        + composite[owner].astype(np.float32) * (1.0 - alpha_values),
        0.0,
        255.0,
    ).astype(np.uint8)
    visible = background_valid | foreground_mask
    background_exclusion_mask = foreground_mask | (
        blocked_by_foreground & ~background_valid
    )
    domain_x, domain_y, domain_width, domain_height = wall_fov_domain
    domain_slice = np.s_[
        domain_y : domain_y + domain_height,
        domain_x : domain_x + domain_width,
    ]
    (
        image,
        crop_valid,
        cropped_foreground,
        local_crop,
        crop_metadata,
    ) = _crop_dense_result(
        composite[domain_slice],
        visible[domain_slice],
        foreground_mask[domain_slice],
        config=selected_config,
    )
    local_x, local_y, crop_width, crop_height = local_crop
    x = domain_x + local_x
    y = domain_y + local_y
    crop_slice = np.s_[y : y + crop_height, x : x + crop_width]
    crop_domain_height_fraction = float(crop_height / max(1, domain_height))
    crop_domain_width_fraction = float(crop_width / max(1, domain_width))
    crop_metadata = {
        **crop_metadata,
        "wall_fov_domain": wall_fov_domain_metadata,
        "crop_within_wall_fov_domain": {
            "x": int(local_x),
            "y": int(local_y),
            "width": int(crop_width),
            "height": int(crop_height),
        },
        "crop_domain_height_fraction": crop_domain_height_fraction,
        "crop_domain_width_fraction": crop_domain_width_fraction,
    }
    cropped_background_exclusion = background_exclusion_mask[crop_slice]
    cropped_tsdf_foreground = tsdf_foreground_mask[crop_slice]
    cropped_depth_fallback = depth_fallback_mask[crop_slice]
    cropped_depth_multiview_foreground = depth_multiview_foreground_mask[crop_slice]
    cropped_alpha = alpha[crop_slice]
    cropped_foreground_source_id = foreground_source_id[crop_slice]
    cropped_foreground_confidence = foreground_confidence[crop_slice]
    cropped_background_source_id = background_source_id[crop_slice]
    coverage = float(np.mean(crop_valid))
    foreground_pixels = int(np.count_nonzero(cropped_foreground))
    crop_unowned = int(
        np.count_nonzero(cropped_foreground & (cropped_foreground_source_id < 0))
    )
    wall_normal_depth = -plane.offset_mm / float(
        np.dot(plane.normal_world, np.asarray(canvas.normal_axis, dtype=np.float64))
    )
    quality_pass = bool(
        coverage >= selected_config.minimum_crop_coverage
        and crop_domain_height_fraction
        >= selected_config.minimum_inscribed_crop_height_fraction
        and crop_domain_width_fraction
        >= selected_config.minimum_crop_width_fraction
        and crop_unowned == 0
        and background_audit["background_sample_from_excluded_foreground_count"] == 0
        and int(foreground_texture_metadata["foreground_texture_from_mesh_vertex_color_count"])
        == 0
    )
    metadata: dict[str, object] = {
        "backend": "tsdf_plane_dense_rgbd",
        "config": asdict(selected_config),
        "canvas_width": canvas.width,
        "canvas_height": canvas.height,
        "crop": {"x": x, "y": y, "width": crop_width, "height": crop_height},
        "wall_plane_fit": fitted_plane.as_dict(),
        "wall_plane": plane.as_dict(),
        "wall_plane_first_surface_shift_mm": front_surface_shift_mm,
        "background_plane_normal_depth_mm": wall_normal_depth,
        "tsdf_vertex_count": int(len(mesh.vertices)),
        "tsdf_triangle_count": int(len(mesh.triangles)),
        "tsdf_mesh_glb_byte_count": int(len(tsdf_mesh_glb)),
        "background_valid_pixel_count": int(np.count_nonzero(background_valid)),
        "visible_pixel_count": int(np.count_nonzero(visible)),
        "background_source_failures": background_source_failures,
        "background_source_mask_summary": [
            {
                "frame_id": masks.frame_id,
                "foreground_core_pixel_count": int(np.count_nonzero(masks.foreground_core)),
                "unknown_band_pixel_count": int(np.count_nonzero(masks.unknown_band)),
                "tsdf_silhouette_pixel_count": int(np.count_nonzero(masks.tsdf_silhouette)),
                "background_safe_pixel_count": int(np.count_nonzero(masks.background_safe)),
                "wall_residual_rmse_mm": masks.wall_residual_rmse_mm,
                "wall_residual_p95_mm": masks.wall_residual_p95_mm,
            }
            for masks in background_masks
        ],
        "background_sampling_audit": background_audit,
        "background_sample_from_excluded_foreground_count": background_audit[
            "background_sample_from_excluded_foreground_count"
        ],
        "foreground_tsdf_pixel_count": int(np.count_nonzero(cropped_tsdf_foreground)),
        "foreground_depth_point_fallback_pixel_count": int(
            np.count_nonzero(cropped_depth_fallback)
        ),
        "foreground_depth_multiview_pixel_count": int(
            np.count_nonzero(cropped_depth_multiview_foreground)
        ),
        "foreground_depth_texture_rescue_pixel_count": int(
            np.count_nonzero(depth_texture_rescue[crop_slice])
        ),
        "foreground_neighbor_texture_recovery_pixel_count": int(
            np.count_nonzero(neighbor_texture_recovery[crop_slice])
        ),
        "foreground_tsdf_local_hole_pixel_count": int(np.count_nonzero(local_tsdf_holes)),
        "foreground_overlay_pixel_count": foreground_pixels,
        "foreground_depth_point_audit": depth_point_metadata,
        "foreground_texture_audit": foreground_texture_metadata,
        "foreground_alpha_radius_px": alpha_radius,
        "foreground_unowned_pixel_count": crop_unowned,
        "foreground_multiple_owner_pixel_count": int(
            np.count_nonzero(foreground_multiple_owner[crop_slice])
        ),
        "crop_valid_coverage": coverage,
        "crop_visibility": crop_metadata,
        "quality_metrics": {
            "quality_pass": quality_pass,
            "crop_valid_coverage": coverage,
            "minimum_crop_coverage": selected_config.minimum_crop_coverage,
            "crop_height_fraction": crop_domain_height_fraction,
            "minimum_crop_height_fraction": (
                selected_config.minimum_inscribed_crop_height_fraction
            ),
            "crop_width_fraction": crop_domain_width_fraction,
            "minimum_crop_width_fraction": selected_config.minimum_crop_width_fraction,
            "foreground_overlay_pixel_count": foreground_pixels,
            "foreground_unowned_pixel_count": crop_unowned,
            "background_sample_from_excluded_foreground_count": background_audit[
                "background_sample_from_excluded_foreground_count"
            ],
        },
    }
    return DenseFusionResult(
        image=image,
        valid_mask=crop_valid,
        foreground_mask=cropped_foreground,
        background_exclusion_mask=cropped_background_exclusion,
        tsdf_foreground_mask=cropped_tsdf_foreground,
        depth_fallback_mask=cropped_depth_fallback,
        depth_multiview_foreground_mask=cropped_depth_multiview_foreground,
        foreground_alpha=np.clip(cropped_alpha * 255.0, 0.0, 255.0).astype(np.uint8),
        foreground_source_id=cropped_foreground_source_id,
        foreground_confidence=np.clip(
            cropped_foreground_confidence * 255.0, 0.0, 255.0
        ).astype(np.uint8),
        background_source_id=cropped_background_source_id,
        tsdf_mesh_glb=tsdf_mesh_glb,
        metadata=metadata,
    )
