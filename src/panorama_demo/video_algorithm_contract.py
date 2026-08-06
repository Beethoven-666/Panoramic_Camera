"""Strict interfaces shared by candidate and frozen production video algorithms.

The contract makes it impossible for the lifecycle facade to treat an
unstructured dictionary or a legacy renderer return as a production result.
It intentionally contains no image processing, model loading, or publication
code; implementations own those concerns while this module validates their
real-source/provenance boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, runtime_checkable

import numpy as np


class VideoAlgorithmContractError(ValueError):
    """A v2 renderer result or pair plan violates a fail-closed invariant."""


@dataclass(frozen=True)
class PairPlan:
    """Auditable decision for one chronological adjacent real-source pair."""

    left_frame_id: int
    right_frame_id: int
    risk_level: int
    flow_backend: str
    use_raft_backward: bool
    use_depth_mesh: bool
    use_open3d: bool
    object_lock_required: bool
    seam_mode: str
    blend_mode: str

    def __post_init__(self) -> None:
        for name, value in (
            ("left_frame_id", self.left_frame_id),
            ("right_frame_id", self.right_frame_id),
            ("risk_level", self.risk_level),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise VideoAlgorithmContractError(f"{name} must be a non-negative integer")
        if self.right_frame_id <= self.left_frame_id:
            raise VideoAlgorithmContractError("PairPlan must join chronological distinct real source ids")
        if self.risk_level not in (0, 1, 2, 3):
            raise VideoAlgorithmContractError("risk_level must be in [0, 3]")
        if self.flow_backend not in {"none", "dis", "raft_small"}:
            raise VideoAlgorithmContractError("flow_backend is unsupported")
        if self.use_raft_backward and self.flow_backend != "raft_small":
            raise VideoAlgorithmContractError("RAFT backward requires flow_backend=raft_small")
        if self.seam_mode not in {"hard_owner", "curved_hard_owner", "multilabel"}:
            raise VideoAlgorithmContractError("seam_mode is unsupported")
        if self.blend_mode not in {"none", "safe_multiband"}:
            raise VideoAlgorithmContractError("blend_mode is unsupported")
        if self.blend_mode != "none" and self.seam_mode == "hard_owner":
            raise VideoAlgorithmContractError("safe MultiBand needs an audited nontrivial seam plan")

    def as_dict(self) -> dict[str, object]:
        return {
            "left_frame_id": self.left_frame_id,
            "right_frame_id": self.right_frame_id,
            "risk_level": self.risk_level,
            "flow_backend": self.flow_backend,
            "use_raft_backward": self.use_raft_backward,
            "use_depth_mesh": self.use_depth_mesh,
            "use_open3d": self.use_open3d,
            "object_lock_required": self.object_lock_required,
            "seam_mode": self.seam_mode,
            "blend_mode": self.blend_mode,
        }


@dataclass(frozen=True)
class PreparedVideoAlgorithm:
    """Immutable real-source plan passed from ``prepare`` to ``render``."""

    source_frame_ids: tuple[int, ...]
    camera_to_world: tuple[np.ndarray, ...]
    pair_plans: tuple[PairPlan, ...]
    context_audit: dict[str, object]

    def __post_init__(self) -> None:
        ids = self.source_frame_ids
        if len(ids) < 2 or tuple(sorted(ids)) != ids or len(set(ids)) != len(ids):
            raise VideoAlgorithmContractError("prepared algorithm needs >=2 unique chronological real source ids")
        if len(self.camera_to_world) != len(ids):
            raise VideoAlgorithmContractError("every real render source needs a camera_to_world pose")
        if len(self.pair_plans) != len(ids) - 1:
            raise VideoAlgorithmContractError("prepared pair plans must cover every adjacent render source pair")
        for frame_id, pose in zip(ids, self.camera_to_world, strict=True):
            matrix = np.asarray(pose, dtype=np.float64)
            if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
                raise VideoAlgorithmContractError(f"source frame {frame_id} has no finite 4x4 pose")
        expected_pairs = tuple(zip(ids[:-1], ids[1:], strict=True))
        received_pairs = tuple((plan.left_frame_id, plan.right_frame_id) for plan in self.pair_plans)
        if received_pairs != expected_pairs:
            raise VideoAlgorithmContractError("pair plans must exactly cover adjacent chronological source ids")


@dataclass(frozen=True)
class VideoAlgorithmResult:
    """The only v2 renderer result accepted by the publication lifecycle."""

    panorama_bgr: np.ndarray
    owner_frame_id: np.ndarray
    source_frame_ids: tuple[int, ...]
    algorithm_audit: dict[str, object]
    artifact_sources: object | None = None
    # Candidate-only, read-only final inverse-grid deltas.  They are emitted
    # only after rendering and can be consumed solely by post-publication
    # fixed-annotation measurement; they never feed a renderer decision.
    measurement_grid_updates: tuple[dict[str, object], ...] = ()

    def __post_init__(self) -> None:
        panorama = np.asarray(self.panorama_bgr)
        owner = np.asarray(self.owner_frame_id)
        ids = self.source_frame_ids
        if panorama.dtype != np.uint8 or panorama.ndim != 3 or panorama.shape[2] != 3:
            raise VideoAlgorithmContractError("panorama_bgr must be uint8 HxWx3")
        if owner.ndim != 2 or owner.shape != panorama.shape[:2] or not np.issubdtype(owner.dtype, np.integer):
            raise VideoAlgorithmContractError("owner_frame_id must be an integer HxW map matching panorama")
        if not ids or tuple(sorted(ids)) != ids or len(set(ids)) != len(ids):
            raise VideoAlgorithmContractError("result source_frame_ids must be unique chronological real ids")
        valid = owner >= 0
        if np.any(valid) and not set(np.unique(owner[valid]).tolist()).issubset(set(ids)):
            raise VideoAlgorithmContractError("every valid owner must be a declared real render source")
        if not isinstance(self.algorithm_audit, dict):
            raise VideoAlgorithmContractError("algorithm_audit must be an object")
        if not isinstance(self.measurement_grid_updates, tuple):
            raise VideoAlgorithmContractError("measurement_grid_updates must be an immutable tuple")
        if self.algorithm_audit.get("interpolated_pose_count", 0) != 0:
            raise VideoAlgorithmContractError("v2 results cannot contain interpolated poses")


@runtime_checkable
class VideoPanoramaAlgorithm(Protocol):
    """Candidate/production renderer lifecycle; never implemented by photo code."""

    def prepare(
        self,
        *,
        session: Any,
        online_state: Any | None,
        context: Mapping[str, object],
    ) -> PreparedVideoAlgorithm:
        """Plan only real RGB-D sources and real ORB camera_to_world poses."""

    def render(self, prepared: PreparedVideoAlgorithm) -> VideoAlgorithmResult:
        """Render one owner-complete panorama and return its immutable audit."""


def require_v2_algorithm(value: object) -> VideoPanoramaAlgorithm:
    """Reject incomplete objects before a candidate/production invocation."""

    if not isinstance(value, VideoPanoramaAlgorithm):
        raise VideoAlgorithmContractError("video v2 renderer must implement prepare() and render()")
    return value


__all__ = [
    "PairPlan",
    "PreparedVideoAlgorithm",
    "VideoAlgorithmContractError",
    "VideoAlgorithmResult",
    "VideoPanoramaAlgorithm",
    "require_v2_algorithm",
]
