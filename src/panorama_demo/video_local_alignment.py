"""RGB-only local alignment models for the v6 video candidate renderer.

The functions here only *estimate and audit* a transform from the one cached
DIS correspondence observation for an adjacent pair.  They never sample,
warp, blend, or otherwise modify panorama RGB.  The final one-shot sampler
consumes an accepted model in Step 10; rejected evidence remains a hard-owner
seam/reroute input.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, degrees

import cv2
import numpy as np

from .video_visual_renderer import VideoDISPairEvidence


@dataclass(frozen=True)
class VideoLocalAlignmentConfig:
    """Frozen v6 alignment limits; all distances are full-resolution pixels."""

    background_displacement_target_px: float = 6.0
    background_displacement_hard_px: float = 10.0
    background_held_out_fb_target_px: float = 1.25
    background_held_out_fb_hard_px: float = 2.0
    near_translation_target_px: float = 3.0
    near_translation_hard_px: float = 6.0
    near_rotation_target_deg: float = 1.5
    near_rotation_hard_deg: float = 3.0
    near_affine_scale_min: float = 0.95
    near_affine_scale_max: float = 1.05
    near_affine_anisotropic_ratio_max: float = 1.05
    near_affine_shear_abs_max: float = 0.05
    near_homography_corner_displacement_hard_px: float = 6.0
    near_homography_scale_min: float = 0.94
    near_homography_scale_max: float = 1.06
    near_homography_line_orientation_change_max_deg: float = 1.5
    near_homography_held_out_fb_p95_max_px: float = 1.0
    near_homography_held_out_fb_abs_max_px: float = 2.0
    minimum_training_points: int = 32
    minimum_held_out_points: int = 16
    mesh_grid_columns: int = 16
    mesh_grid_rows: int = 12

    def __post_init__(self) -> None:
        values = tuple(float(value) for value in self.__dict__.values() if isinstance(value, float))
        if not np.isfinite(values).all() or any(value <= 0.0 for value in values):
            raise ValueError("alignment limits must be finite and positive")
        if self.background_displacement_target_px > self.background_displacement_hard_px:
            raise ValueError("background target cannot exceed hard limit")
        if self.near_translation_target_px > self.near_translation_hard_px:
            raise ValueError("near translation target cannot exceed hard limit")
        if self.near_rotation_target_deg > self.near_rotation_hard_deg:
            raise ValueError("near rotation target cannot exceed hard limit")
        if self.minimum_training_points < 8 or self.minimum_held_out_points < 8:
            raise ValueError("alignment needs at least eight train and held-out samples")
        if self.mesh_grid_columns < 3 or self.mesh_grid_rows < 3:
            raise ValueError("mesh grids need a fixed zero-displacement boundary")


@dataclass(frozen=True)
class VideoAlignmentAudit:
    """Model order, train/held-out result, and fail-closed rejection reason."""

    kind: str
    selected_model: str
    accepted: bool
    large_alignment_warning: bool
    held_out_residual_p95_px: float | None
    held_out_residual_abs_max_px: float | None
    held_out_fb_p95_px: float | None
    held_out_fb_abs_max_px: float | None
    maximum_displacement_px: float | None
    rotation_degrees: float | None
    affine_scale_x: float | None
    affine_scale_y: float | None
    affine_shear: float | None
    positive_jacobian: bool | None
    outer_boundary_zero_displacement: bool | None
    rejected_models: tuple[str, ...]
    rejection_reason: str | None


@dataclass(frozen=True)
class VideoLocalAlignment:
    """An audited local model; it is data only until final one-shot sampling."""

    audit: VideoAlignmentAudit
    matrix: np.ndarray | None
    mesh_displacement: np.ndarray | None


def _p95(values: np.ndarray) -> float | None:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return None if finite.size == 0 else float(np.percentile(finite, 95.0))


def _points_from_evidence(
    evidence: VideoDISPairEvidence, support: np.ndarray | None, config: VideoLocalAlignmentConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    reliable = np.asarray(evidence.reliable_mask, dtype=bool)
    if support is not None:
        support = np.asarray(support, dtype=bool)
        if support.shape != reliable.shape:
            raise ValueError("alignment support must match DIS evidence")
        reliable &= support
    y, x = np.nonzero(reliable)
    if x.size == 0:
        empty = np.empty((0, 2), dtype=np.float32)
        return empty, empty, empty, empty
    # A fixed lattice partition is independent of fitted values, so held-out
    # data cannot leak into model fitting.  Cap point count deterministically.
    take = max(1, int(np.ceil(x.size / 4096)))
    x, y = x[::take], y[::take]
    points = np.column_stack((x, y)).astype(np.float32)
    targets = points + evidence.flow_forward[y, x].astype(np.float32)
    held_out = ((x * 17 + y * 31) % 5) == 0
    return points[~held_out], targets[~held_out], points[held_out], targets[held_out]


def _apply_matrix(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack((points, np.ones(len(points), dtype=np.float32)))
    projected = homogeneous @ np.asarray(matrix, dtype=np.float64).T
    return (projected[:, :2] / projected[:, 2:3]).astype(np.float32)


def _translation(train_points: np.ndarray, train_targets: np.ndarray) -> np.ndarray:
    delta = np.median(train_targets - train_points, axis=0)
    return np.array(((1.0, 0.0, delta[0]), (0.0, 1.0, delta[1]), (0.0, 0.0, 1.0)))


def _rotation(train_points: np.ndarray, train_targets: np.ndarray) -> np.ndarray:
    source_centre, target_centre = train_points.mean(axis=0), train_targets.mean(axis=0)
    source, target = train_points - source_centre, train_targets - target_centre
    rotation_u, _, rotation_vt = np.linalg.svd(source.T @ target)
    rotation = rotation_u @ rotation_vt
    if np.linalg.det(rotation) < 0.0:
        rotation_vt[-1] *= -1.0
        rotation = rotation_u @ rotation_vt
    shift = target_centre - source_centre @ rotation
    return np.array(
        ((rotation[0, 0], rotation[1, 0], shift[0]), (rotation[0, 1], rotation[1, 1], shift[1]), (0.0, 0.0, 1.0))
    )


def _affine(train_points: np.ndarray, train_targets: np.ndarray) -> np.ndarray | None:
    estimated, _ = cv2.estimateAffine2D(train_points, train_targets, method=cv2.RANSAC, ransacReprojThreshold=1.0)
    if estimated is None or not np.isfinite(estimated).all():
        return None
    return np.vstack((estimated, (0.0, 0.0, 1.0))).astype(np.float64)


def _homography(train_points: np.ndarray, train_targets: np.ndarray) -> np.ndarray | None:
    estimated, _ = cv2.findHomography(train_points, train_targets, method=cv2.RANSAC, ransacReprojThreshold=1.0)
    if estimated is None or not np.isfinite(estimated).all() or abs(float(estimated[2, 2])) < 1e-8:
        return None
    return (estimated / estimated[2, 2]).astype(np.float64)


def _matrix_metrics(matrix: np.ndarray, held_points: np.ndarray, held_targets: np.ndarray) -> tuple[float | None, float | None, float]:
    error = np.linalg.norm(_apply_matrix(matrix, held_points) - held_targets, axis=1)
    linear = np.asarray(matrix[:2, :2], dtype=np.float64)
    rotation = degrees(atan2(float(linear[1, 0]), float(linear[0, 0])))
    return _p95(error), (None if error.size == 0 else float(np.max(error))), rotation


def _bounded_mesh(
    train_points: np.ndarray, train_targets: np.ndarray, shape: tuple[int, int], config: VideoLocalAlignmentConfig
) -> np.ndarray:
    """Fit a coarse train-only displacement field with an exact zero boundary."""

    height, width = shape
    x_nodes = np.linspace(0.0, width - 1.0, config.mesh_grid_columns)
    y_nodes = np.linspace(0.0, height - 1.0, config.mesh_grid_rows)
    controls = np.zeros((config.mesh_grid_rows, config.mesh_grid_columns, 2), dtype=np.float32)
    delta = train_targets - train_points
    for row in range(1, config.mesh_grid_rows - 1):
        for column in range(1, config.mesh_grid_columns - 1):
            distance = np.linalg.norm(train_points - (x_nodes[column], y_nodes[row]), axis=1)
            nearest = np.argsort(distance)[: min(48, len(distance))]
            weights = 1.0 / np.maximum(distance[nearest], 1.0)
            controls[row, column] = np.average(delta[nearest], axis=0, weights=weights)
    mesh = np.empty((height, width, 2), dtype=np.float32)
    for axis in range(2):
        mesh[..., axis] = cv2.resize(controls[..., axis], (width, height), interpolation=cv2.INTER_LINEAR)
    mesh[[0, -1], :, :] = 0.0
    mesh[:, [0, -1], :] = 0.0
    return mesh


def _apply_mesh(mesh: np.ndarray, points: np.ndarray) -> np.ndarray:
    coordinates = points.reshape((-1, 1, 2)).astype(np.float32)
    sampled = cv2.remap(
        mesh, coordinates[..., 0], coordinates[..., 1], cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(0.0, 0.0),
    ).reshape((-1, 2))
    return points + sampled


def _affine_parameters(matrix: np.ndarray) -> tuple[float, float, float]:
    linear = np.asarray(matrix[:2, :2], dtype=np.float64)
    _, singular, _ = np.linalg.svd(linear)
    scale_x, scale_y = float(singular[0]), float(singular[1])
    return scale_x, scale_y, float(linear[0, 1] + linear[1, 0]) * 0.5


def _empty(kind: str, reason: str, *, rejected_models: tuple[str, ...] = ()) -> VideoLocalAlignment:
    return VideoLocalAlignment(
        audit=VideoAlignmentAudit(
            kind=kind, selected_model="hard_owner_only", accepted=False, large_alignment_warning=False,
            held_out_residual_p95_px=None, held_out_residual_abs_max_px=None, held_out_fb_p95_px=None,
            held_out_fb_abs_max_px=None, maximum_displacement_px=None, rotation_degrees=None,
            affine_scale_x=None, affine_scale_y=None, affine_shear=None, positive_jacobian=None,
            outer_boundary_zero_displacement=None, rejected_models=rejected_models, rejection_reason=reason,
        ), matrix=None, mesh_displacement=None,
    )


def _evaluate_matrix(
    kind: str, model: str, matrix: np.ndarray, train_points: np.ndarray, held_points: np.ndarray,
    held_targets: np.ndarray, evidence: VideoDISPairEvidence, config: VideoLocalAlignmentConfig,
) -> VideoAlignmentAudit:
    held_p95, held_max, rotation = _matrix_metrics(matrix, held_points, held_targets)
    held_y, held_x = held_points[:, 1].astype(int), held_points[:, 0].astype(int)
    fb = evidence.fb_error[held_y, held_x]
    fb_p95, fb_max = _p95(fb), (None if fb.size == 0 else float(np.nanmax(fb)))
    sample = np.vstack((train_points, held_points))
    displacement = np.linalg.norm(_apply_matrix(matrix, sample) - sample, axis=1)
    maximum_displacement = float(np.max(displacement))
    scale_x, scale_y, shear = _affine_parameters(matrix)
    return VideoAlignmentAudit(
        kind=kind, selected_model=model, accepted=False, large_alignment_warning=False,
        held_out_residual_p95_px=held_p95, held_out_residual_abs_max_px=held_max,
        held_out_fb_p95_px=fb_p95, held_out_fb_abs_max_px=fb_max, maximum_displacement_px=maximum_displacement,
        rotation_degrees=rotation, affine_scale_x=scale_x, affine_scale_y=scale_y, affine_shear=shear,
        positive_jacobian=bool(np.linalg.det(matrix[:2, :2]) > 0.0), outer_boundary_zero_displacement=None,
        rejected_models=(), rejection_reason=None,
    )


def _evaluate_mesh(
    train_points: np.ndarray, train_targets: np.ndarray, held_points: np.ndarray, held_targets: np.ndarray,
    evidence: VideoDISPairEvidence, config: VideoLocalAlignmentConfig,
) -> tuple[VideoAlignmentAudit, np.ndarray]:
    mesh = _bounded_mesh(train_points, train_targets, evidence.fb_error.shape, config)
    error = np.linalg.norm(_apply_mesh(mesh, held_points) - held_targets, axis=1)
    held_y, held_x = held_points[:, 1].astype(int), held_points[:, 0].astype(int)
    fb = evidence.fb_error[held_y, held_x]
    jacobian_x = 1.0 + np.gradient(mesh[..., 0], axis=1)
    jacobian_y = 1.0 + np.gradient(mesh[..., 1], axis=0)
    cross_xy = np.gradient(mesh[..., 0], axis=0)
    cross_yx = np.gradient(mesh[..., 1], axis=1)
    determinant = jacobian_x * jacobian_y - cross_xy * cross_yx
    displacement = np.linalg.norm(mesh, axis=2)
    audit = VideoAlignmentAudit(
        kind="background", selected_model="bounded_mesh", accepted=False, large_alignment_warning=False,
        held_out_residual_p95_px=_p95(error),
        held_out_residual_abs_max_px=None if error.size == 0 else float(np.max(error)),
        held_out_fb_p95_px=_p95(fb), held_out_fb_abs_max_px=None if fb.size == 0 else float(np.nanmax(fb)),
        maximum_displacement_px=float(np.max(displacement)), rotation_degrees=None,
        affine_scale_x=None, affine_scale_y=None, affine_shear=None,
        positive_jacobian=bool(np.all(determinant > 0.0)),
        outer_boundary_zero_displacement=bool(
            np.allclose(mesh[[0, -1]], 0.0) and np.allclose(mesh[:, [0, -1]], 0.0)
        ), rejected_models=(), rejection_reason=None,
    )
    return audit, mesh


def _accepted_background(audit: VideoAlignmentAudit, config: VideoLocalAlignmentConfig) -> bool:
    return bool(
        audit.positive_jacobian
        and audit.held_out_residual_p95_px is not None
        and audit.held_out_residual_p95_px <= config.background_held_out_fb_hard_px
        and audit.held_out_fb_p95_px is not None
        and audit.held_out_fb_p95_px <= config.background_held_out_fb_hard_px
        and audit.maximum_displacement_px is not None
        and audit.maximum_displacement_px <= config.background_displacement_hard_px
    )


def _accepted_near(audit: VideoAlignmentAudit, model: str, config: VideoLocalAlignmentConfig) -> bool:
    if not (audit.positive_jacobian and audit.held_out_residual_p95_px is not None and audit.held_out_residual_p95_px <= config.near_translation_hard_px):
        return False
    if audit.maximum_displacement_px is None or audit.maximum_displacement_px > config.near_translation_hard_px:
        return False
    if model in {"rotation", "affine", "restricted_homography"} and (
        audit.rotation_degrees is None or abs(audit.rotation_degrees) > config.near_rotation_hard_deg
    ):
        return False
    if model in {"affine", "restricted_homography"}:
        if audit.affine_scale_x is None or audit.affine_scale_y is None or audit.affine_shear is None:
            return False
        if not (config.near_affine_scale_min <= audit.affine_scale_x <= config.near_affine_scale_max):
            return False
        if not (config.near_affine_scale_min <= audit.affine_scale_y <= config.near_affine_scale_max):
            return False
        if abs(audit.affine_scale_x / audit.affine_scale_y) > config.near_affine_anisotropic_ratio_max:
            return False
        if abs(audit.affine_shear) > config.near_affine_shear_abs_max:
            return False
    return True


def _restricted_homography_geometry_passes(
    matrix: np.ndarray, shape: tuple[int, int], config: VideoLocalAlignmentConfig
) -> bool:
    """Check corner displacement, local scale, and orientation without RGB sampling."""

    height, width = shape
    corners = np.array(
        ((0.0, 0.0), (width - 1.0, 0.0), (0.0, height - 1.0), (width - 1.0, height - 1.0)),
        dtype=np.float32,
    )
    projected = _apply_matrix(matrix, corners)
    if np.max(np.linalg.norm(projected - corners, axis=1)) > config.near_homography_corner_displacement_hard_px:
        return False
    horizontal = np.linalg.norm(projected[[1, 3]] - projected[[0, 2]], axis=1) / max(width - 1.0, 1.0)
    vertical = np.linalg.norm(projected[[2, 3]] - projected[[0, 1]], axis=1) / max(height - 1.0, 1.0)
    scales = np.concatenate((horizontal, vertical))
    if np.any(scales < config.near_homography_scale_min) or np.any(scales > config.near_homography_scale_max):
        return False
    horizontal_angle = np.degrees(np.arctan2(
        projected[[1, 3], 1] - projected[[0, 2], 1], projected[[1, 3], 0] - projected[[0, 2], 0]
    ))
    return bool(np.all(np.abs(horizontal_angle) <= config.near_homography_line_orientation_change_max_deg))


def _with_result(audit: VideoAlignmentAudit, accepted: bool, warning: bool, rejected: list[str], reason: str | None) -> VideoAlignmentAudit:
    return VideoAlignmentAudit(**{**audit.__dict__, "accepted": accepted, "large_alignment_warning": warning, "rejected_models": tuple(rejected), "rejection_reason": reason})


def fit_background_alignment(
    evidence: VideoDISPairEvidence, *, support: np.ndarray | None = None,
    config: VideoLocalAlignmentConfig | None = None,
) -> VideoLocalAlignment:
    """Choose identity, translation, affine, or a future bounded mesh evidence model.

    Only RGB/DIS correspondence evidence is consumed.  The result describes a
    possible later sampling grid and has no effect on RGB at this stage.
    """

    settings = config or VideoLocalAlignmentConfig()
    train, targets, held, held_targets = _points_from_evidence(evidence, support, settings)
    if len(train) < settings.minimum_training_points or len(held) < settings.minimum_held_out_points:
        return _empty("background", "insufficient_independent_dis_correspondence")
    candidates: list[tuple[str, np.ndarray | None]] = [
        ("identity", np.eye(3)), ("translation", _translation(train, targets)), ("affine", _affine(train, targets)),
    ]
    rejected: list[str] = []
    best: tuple[VideoAlignmentAudit, np.ndarray] | None = None
    for name, matrix in candidates:
        if matrix is None:
            rejected.append(name)
            continue
        audit = _evaluate_matrix("background", name, matrix, train, held, held_targets, evidence, settings)
        if not _accepted_background(audit, settings):
            rejected.append(name)
            continue
        residual_warning = bool(
            audit.held_out_residual_p95_px
            and audit.held_out_residual_p95_px > settings.background_held_out_fb_target_px
        )
        warning = bool(
            residual_warning
            or (audit.maximum_displacement_px and audit.maximum_displacement_px > settings.background_displacement_target_px)
        )
        if not residual_warning:
            return VideoLocalAlignment(_with_result(audit, True, warning, rejected, None), matrix, None)
        rejected.append(name)
        if best is None or audit.held_out_residual_p95_px < best[0].held_out_residual_p95_px:
            best = (audit, matrix)
    # Escalate to the bounded, zero-boundary mesh only if the simpler models
    # still leave target-level held-out residual.  It is fit exclusively on
    # the training partition and must pass the same hard limits.
    mesh_audit, mesh = _evaluate_mesh(train, targets, held, held_targets, evidence, settings)
    if _accepted_background(mesh_audit, settings) and mesh_audit.outer_boundary_zero_displacement:
        mesh_warning = bool(
            (mesh_audit.maximum_displacement_px and mesh_audit.maximum_displacement_px > settings.background_displacement_target_px)
            or (mesh_audit.held_out_residual_p95_px and mesh_audit.held_out_residual_p95_px > settings.background_held_out_fb_target_px)
        )
        if best is None or mesh_audit.held_out_residual_p95_px < best[0].held_out_residual_p95_px:
            return VideoLocalAlignment(
                _with_result(mesh_audit, True, mesh_warning, rejected, None), None, mesh
            )
    else:
        rejected.append("bounded_mesh")
    if best is not None:
        audit, matrix = best
        return VideoLocalAlignment(_with_result(audit, True, True, rejected, None), matrix, None)
    return _empty(
        "background", "all_identity_translation_affine_models_exceeded_hard_limits",
        rejected_models=tuple(rejected),
    )


def fit_near_protected_alignment(
    evidence: VideoDISPairEvidence, *, support: np.ndarray | None = None, plane_verified: bool = False,
    config: VideoLocalAlignmentConfig | None = None,
) -> VideoLocalAlignment:
    """Follow the required near-field ladder without inferring planarity.

    ``plane_verified`` is deliberately explicit.  This repository has no
    approved object classifier/segmentation source, so homography fails closed
    unless a later object-protection stage supplies independently verified
    planar support.
    """

    settings = config or VideoLocalAlignmentConfig()
    train, targets, held, held_targets = _points_from_evidence(evidence, support, settings)
    if len(train) < settings.minimum_training_points or len(held) < settings.minimum_held_out_points:
        return _empty("near", "insufficient_independent_dis_correspondence")
    candidates: list[tuple[str, np.ndarray | None]] = [
        ("identity", np.eye(3)), ("translation", _translation(train, targets)),
        ("rotation", _rotation(train, targets)), ("affine", _affine(train, targets)),
    ]
    rejected: list[str] = []
    best: tuple[VideoAlignmentAudit, np.ndarray] | None = None
    for name, matrix in candidates:
        if matrix is None:
            rejected.append(name)
            continue
        audit = _evaluate_matrix("near", name, matrix, train, held, held_targets, evidence, settings)
        if not _accepted_near(audit, name, settings):
            rejected.append(name)
            continue
        residual_warning = bool(
            audit.held_out_residual_p95_px
            and audit.held_out_residual_p95_px > settings.near_translation_target_px
        )
        warning = bool(
            residual_warning
            or (audit.maximum_displacement_px and audit.maximum_displacement_px > settings.near_translation_target_px)
        )
        if not residual_warning:
            return VideoLocalAlignment(_with_result(audit, True, warning, rejected, None), matrix, None)
        rejected.append(name)
        if best is None or audit.held_out_residual_p95_px < best[0].held_out_residual_p95_px:
            best = (audit, matrix)
    if best is not None:
        audit, matrix = best
        return VideoLocalAlignment(_with_result(audit, True, True, rejected, None), matrix, None)
    if plane_verified:
        matrix = _homography(train, targets)
        if matrix is not None:
            audit = _evaluate_matrix("near", "restricted_homography", matrix, train, held, held_targets, evidence, settings)
            # Homography is strictly later and has its tighter independent FB gate.
            valid = _accepted_near(audit, "restricted_homography", settings)
            valid &= _restricted_homography_geometry_passes(matrix, evidence.fb_error.shape, settings)
            valid &= bool(audit.held_out_fb_p95_px is not None and audit.held_out_fb_p95_px <= settings.near_homography_held_out_fb_p95_max_px)
            valid &= bool(audit.held_out_fb_abs_max_px is not None and audit.held_out_fb_abs_max_px <= settings.near_homography_held_out_fb_abs_max_px)
            if valid:
                warning = bool(
                    (audit.maximum_displacement_px and audit.maximum_displacement_px > settings.near_translation_target_px)
                    or (audit.held_out_residual_p95_px and audit.held_out_residual_p95_px > settings.near_translation_target_px)
                )
                return VideoLocalAlignment(_with_result(audit, True, warning, rejected, None), matrix, None)
        rejected.append("restricted_homography")
    return _empty(
        "near", "all_near_models_exceeded_hard_limits_or_planarity_unverified",
        rejected_models=tuple(rejected),
    )


__all__ = [
    "VideoAlignmentAudit", "VideoLocalAlignment", "VideoLocalAlignmentConfig",
    "fit_background_alignment", "fit_near_protected_alignment",
]
