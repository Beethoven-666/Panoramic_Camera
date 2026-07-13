from __future__ import annotations

import importlib
import json
import math
import sys
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, TypeVar, cast

import cv2
import numpy as np

from .errors import AlignmentError
from .paths import LIGHTGLUE_DIR, UNISTITCH_CODES_DIR

if TYPE_CHECKING:
    import torch

torch: Any = None
_Function = TypeVar("_Function", bound=Callable[..., Any])


def _load_torch() -> Any:
    global torch
    if torch is None:
        torch = importlib.import_module("torch")
    return torch


def _inference_mode(function: _Function) -> _Function:
    @wraps(function)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        with _load_torch().inference_mode():
            return function(*args, **kwargs)

    return cast(_Function, wrapped)

@dataclass(frozen=True)
class PairAlignment:
    """Result of aligning ``source`` into ``reference`` coordinates."""

    homography_source_to_reference: np.ndarray
    unistitch_homography_source_to_reference: np.ndarray
    preview_bgr: np.ndarray
    match_count: int
    layout_method: str
    median_reprojection_px: float
    magsac_inliers: int
    inference_size: tuple[int, int]
    metrics: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "homography_source_to_reference": self.homography_source_to_reference.tolist(),
            "unistitch_homography_source_to_reference": (
                self.unistitch_homography_source_to_reference.tolist()
            ),
            "match_count": self.match_count,
            "layout_method": self.layout_method,
            "median_reprojection_px": self.median_reprojection_px,
            "magsac_inliers": self.magsac_inliers,
            "inference_size": list(self.inference_size),
            "metrics": self.metrics,
        }

    def write_report(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.as_dict(), indent=2), encoding="utf-8")


def _normalized_homography(matrix: np.ndarray) -> np.ndarray:
    value = np.asarray(matrix, dtype=np.float64)
    if value.shape != (3, 3) or not np.isfinite(value).all():
        raise AlignmentError("A predicted homography is not a finite 3x3 matrix")
    if abs(value[2, 2]) < 1e-10:
        raise AlignmentError("A predicted homography is singular")
    return value / value[2, 2]


def _reprojection_errors(
    homography: np.ndarray, source_points: np.ndarray, reference_points: np.ndarray
) -> np.ndarray:
    projected = cv2.perspectiveTransform(
        source_points.astype(np.float32).reshape(1, -1, 2),
        _normalized_homography(homography),
    )[0]
    return np.linalg.norm(projected - reference_points, axis=1)


def _choose_layout_method(
    *,
    unistitch_median: float,
    magsac_median: float,
    unistitch_median_on_magsac_inliers: float,
    magsac_inliers: int,
    magsac_inlier_ratio: float,
    min_matches: int,
    min_magsac_inlier_ratio: float,
    max_unistitch_reprojection_px: float,
    allow_magsac_fallback: bool,
    prefer_magsac_layout: bool,
) -> tuple[str, float]:
    """Choose a sequence layout using comparable residual support.

    A low MAGSAC residual is meaningful only when enough of all matches support
    that model.  Preference compares both models on the MAGSAC inlier set; the
    unconstrained UniStitch median remains the independent acceptance check.
    """

    magsac_usable = (
        allow_magsac_fallback
        and magsac_inliers >= min_matches
        and magsac_inlier_ratio >= min_magsac_inlier_ratio
        and np.isfinite(magsac_median)
    )
    if (
        magsac_usable
        and prefer_magsac_layout
        and magsac_median < unistitch_median_on_magsac_inliers
    ):
        return "magsac_preferred", magsac_median
    if unistitch_median <= max_unistitch_reprojection_px:
        return "unistitch_global", unistitch_median
    if magsac_usable:
        return "magsac_fallback", magsac_median
    raise AlignmentError(
        "UniStitch global branch failed validation "
        f"(median {unistitch_median:.2f}px); MAGSAC has {magsac_inliers} inliers "
        f"({magsac_inlier_ratio:.1%})"
    )


def _resize_for_inference(image: np.ndarray, target_width: int) -> np.ndarray:
    height, width = image.shape[:2]
    if target_width < 128:
        raise ValueError("inference_width must be at least 128 pixels")
    target_height = max(128, int(round(height * target_width / width)))
    interpolation = cv2.INTER_AREA if target_width < width else cv2.INTER_LINEAR
    return cv2.resize(image, (target_width, target_height), interpolation=interpolation)


def _to_network_image(image_bgr: np.ndarray, device: torch.device) -> torch.Tensor:
    array = image_bgr.astype(np.float32) / 127.5 - 1.0
    return _load_torch().from_numpy(array.transpose(2, 0, 1)).unsqueeze(0).to(device)


def _scale_homography(
    homography_small: np.ndarray,
    original_size: tuple[int, int],
    inference_size: tuple[int, int],
) -> np.ndarray:
    original_width, original_height = original_size
    inference_width, inference_height = inference_size
    original_to_small = np.array(
        [
            [inference_width / original_width, 0.0, 0.0],
            [0.0, inference_height / original_height, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    result = np.linalg.inv(original_to_small) @ homography_small @ original_to_small
    return _normalized_homography(result)


class UniStitchAligner:
    """In-memory SuperPoint/LightGlue adapter for the official UniStitch model.

    UniStitch is a pair model. The returned TPS/FFD result is therefore a pair
    preview, while ``homography_source_to_reference`` is the composable global
    branch used to lay out an image sequence.
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str = "cuda:0",
        inference_width: int = 640,
        max_points: int = 2048,
        min_matches: int = 40,
        max_pair_canvas: int = 4000,
        max_unistitch_reprojection_px: float = 20.0,
        allow_magsac_fallback: bool = True,
        prefer_magsac_layout: bool = True,
        min_magsac_inlier_ratio: float = 0.5,
    ) -> None:
        _load_torch()
        self.model_path = Path(model_path).expanduser().resolve()
        self.device = torch.device(device)
        self.inference_width = inference_width
        self.max_points = max_points
        self.min_matches = min_matches
        self.max_pair_canvas = max_pair_canvas
        self.max_unistitch_reprojection_px = max_unistitch_reprojection_px
        self.allow_magsac_fallback = allow_magsac_fallback
        self.prefer_magsac_layout = prefer_magsac_layout
        self.min_magsac_inlier_ratio = min_magsac_inlier_ratio

        if not 0.0 <= self.min_magsac_inlier_ratio <= 1.0:
            raise ValueError("min_magsac_inlier_ratio must be between zero and one")

        if self.device.type != "cuda":
            raise RuntimeError(
                "The published UniStitch code contains CUDA-only operators; "
                "use --device cuda:0 on an NVIDIA GPU."
            )
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available to PyTorch")
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"UniStitch model not found: {self.model_path}. "
                "Run scripts/download_unistitch_weights.py first."
            )
        if not UNISTITCH_CODES_DIR.is_dir():
            raise FileNotFoundError(f"UniStitch source not found: {UNISTITCH_CODES_DIR}")
        if not LIGHTGLUE_DIR.is_dir():
            raise FileNotFoundError(f"LightGlue source not found: {LIGHTGLUE_DIR}")

        device_index = self.device.index if self.device.index is not None else 0
        torch.cuda.set_device(device_index)
        self._network_module = self._import_unistitch()
        self._extractor, self._matcher, self._numpy_image_to_torch, self._rbd = (
            self._build_feature_models()
        )
        self._net = self._load_network()

    @staticmethod
    def _import_unistitch():
        code_path = str(UNISTITCH_CODES_DIR)
        if code_path not in sys.path:
            sys.path.insert(0, code_path)
        module = importlib.import_module("network")
        module_path = Path(module.__file__).resolve()
        if UNISTITCH_CODES_DIR.resolve() not in module_path.parents:
            raise ImportError(f"Imported an unexpected 'network' module: {module_path}")
        return module

    def _build_feature_models(self):
        lightglue_path = str(LIGHTGLUE_DIR)
        if lightglue_path not in sys.path:
            sys.path.insert(0, lightglue_path)
        from lightglue import LightGlue, SuperPoint
        from lightglue.utils import numpy_image_to_torch, rbd

        extractor = SuperPoint(max_num_keypoints=self.max_points).eval().to(self.device)
        matcher = LightGlue(features="superpoint").eval().to(self.device)
        return extractor, matcher, numpy_image_to_torch, rbd

    def _load_network(self):
        # The upstream constructor asks torchvision to download a redundant
        # ImageNet ResNet checkpoint before the UniStitch state dict replaces it.
        # Disable that download without modifying the pinned third-party tree.
        resnet_module = self._network_module.models.resnet
        original_resnet18 = resnet_module.resnet18

        def no_pretrained_resnet18(*args, **kwargs):
            kwargs["weights"] = None
            return original_resnet18(*args, **kwargs)

        resnet_module.resnet18 = no_pretrained_resnet18
        try:
            net = self._network_module.Network()
        finally:
            resnet_module.resnet18 = original_resnet18

        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        if not isinstance(checkpoint, dict) or "model" not in checkpoint:
            raise RuntimeError("The UniStitch checkpoint does not contain checkpoint['model']")
        net.load_state_dict(checkpoint["model"], strict=True)
        net = net.to(self.device).eval()
        net.fuse()
        return net

    @_inference_mode
    def _matched_features(
        self, reference_bgr: np.ndarray, source_bgr: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, torch.Tensor, torch.Tensor, np.ndarray]:
        reference_rgb = np.ascontiguousarray(reference_bgr[..., ::-1])
        source_rgb = np.ascontiguousarray(source_bgr[..., ::-1])
        image0 = self._numpy_image_to_torch(reference_rgb).to(self.device)
        image1 = self._numpy_image_to_torch(source_rgb).to(self.device)
        features0 = self._extractor.extract(image0, resize=None)
        features1 = self._extractor.extract(image1, resize=None)
        matches01 = self._matcher({"image0": features0, "image1": features1})
        features0, features1, matches01 = [
            self._rbd(item) for item in (features0, features1, matches01)
        ]
        matches = matches01["matches"]
        if matches.shape[0] < self.min_matches:
            raise AlignmentError(
                f"Only {matches.shape[0]} LightGlue matches; need at least {self.min_matches}"
            )

        scores = matches01.get("scores")
        if scores is None or scores.numel() != matches.shape[0]:
            scores = torch.ones(matches.shape[0], device=matches.device)
        order = torch.argsort(scores, descending=True)
        matches = matches[order]
        scores = scores[order]

        points0 = features0["keypoints"][matches[:, 0]]
        points1 = features1["keypoints"][matches[:, 1]]
        descriptors0 = features0["descriptors"][matches[:, 0]]
        descriptors1 = features1["descriptors"][matches[:, 1]]
        return (
            points0.detach().cpu().numpy().astype(np.float32),
            points1.detach().cpu().numpy().astype(np.float32),
            descriptors0.detach(),
            descriptors1.detach(),
            scores.detach().cpu().numpy().astype(np.float32),
        )

    def _pad_features(
        self,
        points: np.ndarray,
        descriptors: torch.Tensor,
        width: int,
        height: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        count = points.shape[0]
        if count == 0:
            raise AlignmentError("UniStitch cannot run with zero matched points")
        if count >= self.max_points:
            indices = np.arange(self.max_points)
        else:
            indices = np.arange(self.max_points) % count
        selected_points = points[indices].copy()
        selected_points[:, 0] /= max(1, width - 1)
        selected_points[:, 1] /= max(1, height - 1)
        point_tensor = torch.from_numpy(selected_points).unsqueeze(0).to(self.device)
        descriptor_indices = torch.as_tensor(indices, device=descriptors.device)
        descriptor_tensor = descriptors[descriptor_indices].unsqueeze(0).to(self.device)
        return point_tensor.float(), descriptor_tensor.float()

    @staticmethod
    def _blend_pair(output_ref: torch.Tensor, output_tgt: torch.Tensor) -> np.ndarray:
        reference = (
            output_ref[0, :3].permute(1, 2, 0).detach().float().cpu().numpy() * 127.5
        )
        source = (
            output_tgt[0, :3].permute(1, 2, 0).detach().float().cpu().numpy() * 127.5
        )
        reference_mask = (
            output_ref[0, 3:6].mean(0).detach().float().cpu().numpy().clip(0.0, 1.0)
        )
        source_mask = (
            output_tgt[0, 3:6].mean(0).detach().float().cpu().numpy().clip(0.0, 1.0)
        )
        total = reference_mask + source_mask
        result = np.zeros_like(reference, dtype=np.float32)
        valid = total > 1e-6
        result[valid] = (
            reference[valid] * reference_mask[valid, None]
            + source[valid] * source_mask[valid, None]
        ) / total[valid, None]
        return np.clip(result, 0.0, 255.0).astype(np.uint8)

    @_inference_mode
    def _run_unistitch(
        self,
        reference_bgr: np.ndarray,
        source_bgr: np.ndarray,
        points0: np.ndarray,
        points1: np.ndarray,
        descriptors0: torch.Tensor,
        descriptors1: torch.Tensor,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
        height, width = reference_bgr.shape[:2]
        point0_tensor, descriptor0_tensor = self._pad_features(
            points0, descriptors0, width, height
        )
        point1_tensor, descriptor1_tensor = self._pad_features(
            points1, descriptors1, width, height
        )
        input0 = _to_network_image(reference_bgr, self.device)
        input1 = _to_network_image(source_bgr, self.device)
        resized0 = self._network_module.resize_512(input0)
        resized1 = self._network_module.resize_512(input1)

        h_motion, mesh_motion_ref, mesh_motion_tgt, _ = self._net(
            resized0,
            resized1,
            point0_tensor,
            point1_tensor,
            descriptor0_tensor,
            descriptor1_tensor,
        )
        h_motion = h_motion.reshape(1, 4, 2)
        mesh_motion_ref = mesh_motion_ref.reshape(
            1,
            self._network_module.grid_h + 1,
            self._network_module.grid_w + 1,
            2,
        )
        mesh_motion_tgt = mesh_motion_tgt.reshape_as(mesh_motion_ref)
        h_motion = torch.stack(
            [h_motion[..., 0] * width / 512.0, h_motion[..., 1] * height / 512.0],
            dim=2,
        )
        mesh_motion_ref = torch.stack(
            [
                mesh_motion_ref[..., 0] * width / 512.0,
                mesh_motion_ref[..., 1] * height / 512.0,
            ],
            dim=3,
        )
        mesh_motion_tgt = torch.stack(
            [
                mesh_motion_tgt[..., 0] * width / 512.0,
                mesh_motion_tgt[..., 1] * height / 512.0,
            ],
            dim=3,
        )

        corners = np.array(
            [[0.0, 0.0], [float(width), 0.0], [0.0, float(height)], [float(width), float(height)]],
            dtype=np.float32,
        )
        displaced = corners + h_motion[0].detach().float().cpu().numpy()
        predicted_h = cv2.getPerspectiveTransform(corners, displaced.astype(np.float32))
        predicted_h = _normalized_homography(predicted_h)

        src_p = torch.tensor(corners, device=self.device).unsqueeze(0)
        dst_p = src_p + h_motion
        dst_p_tgt = src_p + h_motion / 2.0
        full_h = self._network_module.torch_DLT.tensor_DLT(src_p, dst_p)
        target_h = self._network_module.torch_DLT.tensor_DLT(src_p, dst_p_tgt)
        reference_h = torch.linalg.inv(full_h) @ target_h
        rigid_mesh = self._network_module.get_rigid_mesh(1, height, width)
        mesh_ref = self._network_module.H2Mesh(reference_h, rigid_mesh) + mesh_motion_ref
        mesh_tgt = self._network_module.H2Mesh(target_h, rigid_mesh) + mesh_motion_tgt

        width_max = torch.maximum(mesh_ref[..., 0].max(), mesh_tgt[..., 0].max())
        width_min = torch.minimum(mesh_ref[..., 0].min(), mesh_tgt[..., 0].min())
        height_max = torch.maximum(mesh_ref[..., 1].max(), mesh_tgt[..., 1].max())
        height_min = torch.minimum(mesh_ref[..., 1].min(), mesh_tgt[..., 1].min())
        out_width = int(math.ceil(float((width_max - width_min).detach().cpu())))
        out_height = int(math.ceil(float((height_max - height_min).detach().cpu())))
        if out_width < 1 or out_height < 1:
            raise AlignmentError("UniStitch produced an empty pair canvas")
        if max(out_width, out_height) > self.max_pair_canvas:
            raise AlignmentError(
                f"UniStitch pair canvas {out_width}x{out_height} exceeds "
                f"the {self.max_pair_canvas}px safety limit"
            )

        mesh_trans_ref = torch.stack(
            [mesh_ref[..., 0] - width_min, mesh_ref[..., 1] - height_min], dim=3
        )
        mesh_trans_tgt = torch.stack(
            [mesh_tgt[..., 0] - width_min, mesh_tgt[..., 1] - height_min], dim=3
        )
        norm_rigid_mesh = self._network_module.get_norm_mesh(rigid_mesh, height, width)
        norm_mesh_ref = self._network_module.get_norm_mesh(
            mesh_trans_ref, out_height, out_width
        )
        norm_mesh_tgt = self._network_module.get_norm_mesh(
            mesh_trans_tgt, out_height, out_width
        )
        mask = torch.ones_like(input1)
        transform = self._network_module.torch_tps_transform.transformer
        output_ref = transform(
            torch.cat((input0 + 1.0, mask), dim=1),
            norm_mesh_ref,
            norm_rigid_mesh,
            (out_height, out_width),
        )
        output_tgt = transform(
            torch.cat((input1 + 1.0, mask), dim=1),
            norm_mesh_tgt,
            norm_rigid_mesh,
            (out_height, out_width),
        )
        preview = self._blend_pair(output_ref, output_tgt)
        metrics = {
            "pair_canvas_width": float(out_width),
            "pair_canvas_height": float(out_height),
            "mesh_min_x": float(width_min.detach().cpu()),
            "mesh_min_y": float(height_min.detach().cpu()),
        }
        return predicted_h, preview, metrics

    def align(self, reference_bgr: np.ndarray, source_bgr: np.ndarray) -> PairAlignment:
        if reference_bgr is None or source_bgr is None:
            raise ValueError("Both images must be readable")
        if reference_bgr.ndim != 3 or reference_bgr.shape[2] != 3:
            raise ValueError("Reference image must be an HxWx3 BGR image")
        if source_bgr.shape != reference_bgr.shape:
            raise ValueError(
                "UniStitch sequence inputs must have identical dimensions; "
                f"got {reference_bgr.shape} and {source_bgr.shape}"
            )

        original_height, original_width = reference_bgr.shape[:2]
        reference_small = _resize_for_inference(reference_bgr, self.inference_width)
        source_small = cv2.resize(
            source_bgr,
            (reference_small.shape[1], reference_small.shape[0]),
            interpolation=cv2.INTER_AREA,
        )
        small_height, small_width = reference_small.shape[:2]
        points0, points1, descriptors0, descriptors1, scores = self._matched_features(
            reference_small, source_small
        )
        predicted_h, preview, mesh_metrics = self._run_unistitch(
            reference_small,
            source_small,
            points0,
            points1,
            descriptors0,
            descriptors1,
        )

        candidate_errors: list[tuple[np.ndarray, np.ndarray, str]] = []
        direct_errors = _reprojection_errors(predicted_h, points1, points0)
        candidate_errors.append((predicted_h, direct_errors, "direct"))
        try:
            inverse_h = _normalized_homography(np.linalg.inv(predicted_h))
            inverse_errors = _reprojection_errors(inverse_h, points1, points0)
            candidate_errors.append((inverse_h, inverse_errors, "inverse"))
        except (np.linalg.LinAlgError, AlignmentError):
            pass
        unistitch_h, unistitch_errors, orientation = min(
            candidate_errors, key=lambda item: float(np.median(item[1]))
        )
        unistitch_median = float(np.median(unistitch_errors))

        magsac_h: np.ndarray | None = None
        inlier_mask: np.ndarray | None = None
        magsac_inliers = 0
        magsac_inlier_ratio = 0.0
        magsac_median = float("inf")
        unistitch_median_on_magsac_inliers = float("inf")
        if self.allow_magsac_fallback:
            try:
                magsac_h, inlier_mask = cv2.findHomography(
                    points1,
                    points0,
                    method=cv2.USAC_MAGSAC,
                    ransacReprojThreshold=3.0,
                    maxIters=10_000,
                    confidence=0.999,
                )
                magsac_inliers = (
                    int(inlier_mask.sum()) if inlier_mask is not None else 0
                )
                magsac_inlier_ratio = magsac_inliers / max(1, points0.shape[0])
                if (
                    magsac_h is not None
                    and inlier_mask is not None
                    and magsac_inliers >= 4
                ):
                    magsac_h = _normalized_homography(magsac_h)
                    support = inlier_mask.ravel() > 0
                    magsac_errors = _reprojection_errors(magsac_h, points1, points0)
                    magsac_median = float(np.median(magsac_errors[support]))
                    unistitch_median_on_magsac_inliers = float(
                        np.median(unistitch_errors[support])
                    )
            except (cv2.error, AlignmentError, np.linalg.LinAlgError):
                magsac_h = None
                inlier_mask = None
                magsac_inliers = 0
                magsac_inlier_ratio = 0.0

        layout_method, layout_median = _choose_layout_method(
            unistitch_median=unistitch_median,
            magsac_median=magsac_median,
            unistitch_median_on_magsac_inliers=(
                unistitch_median_on_magsac_inliers
            ),
            magsac_inliers=magsac_inliers,
            magsac_inlier_ratio=magsac_inlier_ratio,
            min_matches=self.min_matches,
            min_magsac_inlier_ratio=self.min_magsac_inlier_ratio,
            max_unistitch_reprojection_px=self.max_unistitch_reprojection_px,
            allow_magsac_fallback=self.allow_magsac_fallback,
            prefer_magsac_layout=self.prefer_magsac_layout,
        )
        if layout_method.startswith("magsac"):
            if magsac_h is None:  # Defensive: usable MAGSAC always has a matrix.
                raise AlignmentError("MAGSAC layout was selected without a homography")
            layout_h = magsac_h
        else:
            layout_h = unistitch_h

        inference_size = (small_width, small_height)
        original_size = (original_width, original_height)
        layout_original = _scale_homography(layout_h, original_size, inference_size)
        unistitch_original = _scale_homography(
            unistitch_h, original_size, inference_size
        )
        scale_to_original = original_width / small_width
        metrics: dict[str, Any] = {
            **mesh_metrics,
            "unistitch_orientation": orientation,
            "unistitch_median_reprojection_px_inference": unistitch_median,
            "magsac_median_reprojection_px_inference": magsac_median,
            "magsac_inlier_ratio": magsac_inlier_ratio,
            "unistitch_median_on_magsac_inliers_px_inference": (
                unistitch_median_on_magsac_inliers
            ),
            "mean_match_score": float(np.mean(scores)),
            "original_size": [original_width, original_height],
        }
        return PairAlignment(
            homography_source_to_reference=layout_original,
            unistitch_homography_source_to_reference=unistitch_original,
            preview_bgr=preview,
            match_count=int(points0.shape[0]),
            layout_method=layout_method,
            median_reprojection_px=float(layout_median * scale_to_original),
            magsac_inliers=magsac_inliers,
            inference_size=inference_size,
            metrics=metrics,
        )
