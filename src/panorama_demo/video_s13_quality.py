"""Local and sequence-relative structure audits for isolated S1.3 stages."""

from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Mapping

import cv2
import numpy as np

if TYPE_CHECKING:
    from .video_s13_m51_r2 import S13M51R2Config


METRIC_NAMES = (
    "color_jump",
    "gradient_jump",
    "double_edge",
    "motion_residual",
    "thin_structure",
    "horizontal_edge_mismatch",
)


@dataclass(frozen=True)
class SeamStructureFeatures:
    """Image features cached once when many seam paths audit one panorama."""

    lab: np.ndarray
    gray: np.ndarray
    gradient: np.ndarray
    laplacian: np.ndarray
    gradient_x: np.ndarray
    gradient_y: np.ndarray

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.gray.shape[0]), int(self.gray.shape[1])


def _finite_summary(values: np.ndarray) -> float | None:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return None if finite.size < 16 else float(np.median(finite))


def prepare_seam_structure(image: np.ndarray) -> SeamStructureFeatures | None:
    """Prepare reusable, black-safe structure features for one BGR image."""

    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        return None
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return SeamStructureFeatures(
        lab=lab,
        gray=gray,
        gradient=cv2.magnitude(gx, gy),
        laplacian=np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3)),
        gradient_x=gx,
        gradient_y=gy,
    )


def _edge_registration_empty(*, reason: str) -> dict[str, object]:
    return {
        "schema": "gemini305-video-s13-oblique-structure-audit/v1",
        "evaluable": False,
        "reason": reason,
        "supported_block_count": 0,
        "supported_component_count": 0,
        "block_best_lag_px": [],
        "median_supported_abs_lag_px": None,
        "p95_supported_abs_lag_px": None,
        "maximum_supported_abs_lag_px": None,
        "minimum_correlation": None,
        "minimum_uniqueness_margin": None,
        "maximum_orientation_difference_degrees": None,
        "multiple_layer_or_ambiguous": False,
        "block_audits": [],
    }


def _edge_features(
    value: np.ndarray | SeamStructureFeatures,
    *,
    name: str,
) -> SeamStructureFeatures:
    if isinstance(value, SeamStructureFeatures):
        return value
    features = prepare_seam_structure(np.asarray(value))
    if features is None:
        raise ValueError(f"{name} must be a uint8 BGR image or SeamStructureFeatures")
    return features


def edge_registration_visual_suspect(
    edge_registration: Mapping[str, object],
    seam_local_lk_p95_px: float | None,
    *,
    config: "S13M51R2Config",
) -> bool:
    """Return whether sufficient, unambiguous pair-local evidence is risky."""

    component_local = getattr(config, "component_local_ambiguity_enabled", False) is True
    if edge_registration.get("evaluable") is not True:
        return False
    if component_local:
        if int(edge_registration.get("actionable_component_count", 0) or 0) < 1:
            return False
    elif edge_registration.get("multiple_layer_or_ambiguous") is True:
        return False

    def exceeds(value: object, threshold: float) -> bool:
        return (
            isinstance(value, (int, float))
            and math.isfinite(float(value))
            and float(value) > float(threshold)
        )

    return bool(
        exceeds(seam_local_lk_p95_px, config.suspect_lk_p95_px)
        or exceeds(
            edge_registration.get("median_supported_abs_lag_px"),
            config.suspect_edge_median_px,
        )
        or exceeds(
            edge_registration.get("p95_supported_abs_lag_px"),
            config.suspect_edge_p95_px,
        )
    )


def _modulo_pi_orientation_difference(
    left_angle: np.ndarray,
    right_angle: np.ndarray,
) -> np.ndarray:
    difference = np.abs(left_angle - right_angle)
    return np.minimum(difference, np.pi - difference)


def _bilinear_feature_samples(
    feature: SeamStructureFeatures,
    valid: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample cached right features only where every bilinear neighbour is valid."""

    height, width = feature.shape
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1, y1 = x0 + 1, y0 + 1
    inside = (x0 >= 0) & (y0 >= 0) & (x1 < width) & (y1 < height)
    safe_x0 = np.clip(x0, 0, width - 1)
    safe_x1 = np.clip(x1, 0, width - 1)
    safe_y0 = np.clip(y0, 0, height - 1)
    safe_y1 = np.clip(y1, 0, height - 1)
    sample_valid = inside.copy()
    sample_valid &= valid[safe_y0, safe_x0]
    sample_valid &= valid[safe_y0, safe_x1]
    sample_valid &= valid[safe_y1, safe_x0]
    sample_valid &= valid[safe_y1, safe_x1]
    wx = (x - x0).astype(np.float32)
    wy = (y - y0).astype(np.float32)

    def sample(array: np.ndarray) -> np.ndarray:
        top = array[safe_y0, safe_x0] * (1.0 - wx) + array[safe_y0, safe_x1] * wx
        bottom = array[safe_y1, safe_x0] * (1.0 - wx) + array[safe_y1, safe_x1] * wx
        return (top * (1.0 - wy) + bottom * wy).astype(np.float64)

    return sample(feature.gradient), sample(feature.gradient_x), sample(feature.gradient_y), sample_valid


def _component_local_edge_summary(
    observations: Sequence[Mapping[str, object]],
    *,
    config: "S13M51R2Config",
) -> dict[str, object]:
    """Cluster overlap-block observations without merging distinct edge layers."""

    rows = [dict(row) for row in observations]
    parents = list(range(len(rows)))

    def root(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = root(left), root(right)
        if left_root != right_root:
            parents[right_root] = left_root

    def orientation_distance(left: float, right: float) -> float:
        difference = abs(left - right) % 180.0
        return min(difference, 180.0 - difference)

    for left_index, left in enumerate(rows):
        for right_index in range(left_index + 1, len(rows)):
            right = rows[right_index]
            vertical_overlap = min(int(left["bbox_y1"]), int(right["bbox_y1"])) - max(
                int(left["bbox_y0"]), int(right["bbox_y0"])
            )
            if vertical_overlap <= 0:
                continue
            centroid_distance = math.hypot(
                float(left["centroid_x"]) - float(right["centroid_x"]),
                float(left["centroid_y"]) - float(right["centroid_y"]),
            )
            if centroid_distance > float(config.component_cluster_maximum_centroid_distance_px):
                continue
            if orientation_distance(
                float(left["normal_angle_degrees_modulo_180"]),
                float(right["normal_angle_degrees_modulo_180"]),
            ) > float(config.component_cluster_maximum_normal_difference_degrees):
                continue
            if abs(float(left["best_lag_px"]) - float(right["best_lag_px"])) > float(
                config.component_cluster_maximum_lag_difference_px
            ):
                continue
            union(left_index, right_index)

    clusters: dict[int, list[dict[str, object]]] = {}
    for index, row in enumerate(rows):
        clusters.setdefault(root(index), []).append(row)
    component_audits: list[dict[str, object]] = []
    for component_id, members in enumerate(clusters.values()):
        weights = np.asarray(
            [max(1, int(row["support_count"])) for row in members], np.float64
        )
        lags = np.asarray([float(row["best_lag_px"]) for row in members], np.float64)
        ambiguous = any(row.get("component_ambiguous") is True for row in members)
        evaluable = any(row.get("supported") is True for row in members)
        component_audits.append({
            "component_id": component_id,
            "ambiguous": ambiguous,
            "evaluable": evaluable,
            "actionable": evaluable and not ambiguous,
            "observation_count": len(members),
            "block_indices": sorted({int(row["block_index"]) for row in members}),
            "best_lag_px": float(np.average(lags, weights=weights)),
            "absolute_lag_p95_px": float(np.percentile(np.abs(lags), 95.0)),
            "minimum_correlation": float(min(float(row["correlation"]) for row in members)),
            "minimum_uniqueness_margin": float(
                min(float(row["uniqueness_margin"]) for row in members)
            ),
            "maximum_orientation_difference_degrees": float(max(
                float(row["orientation_difference_degrees"]) for row in members
            )),
            "bbox_x0": min(int(row["bbox_x0"]) for row in members),
            "bbox_x1": max(int(row["bbox_x1"]) for row in members),
            "bbox_y0": min(int(row["bbox_y0"]) for row in members),
            "bbox_y1": max(int(row["bbox_y1"]) for row in members),
        })
    actionable = [row for row in component_audits if row["actionable"] is True]
    return {
        "component_audits": component_audits,
        "actionable_components": actionable,
        "ambiguous_component_count": len(component_audits) - len(actionable),
    }


def append_s13_exact_component_evidence_from_forward_context(
    context: Mapping[str, object],
    *,
    left_features: SeamStructureFeatures,
    right_features: SeamStructureFeatures,
    config: "S13M51R2Config",
    exact_evidence_sink: list[tuple[object, object]],
    pair_index: int,
    global_x_offset: int,
) -> None:
    """Reuse the ordinary block forward search and add only reverse hypotheses."""

    from dataclasses import replace
    from .video_s13_m51_r4_component_chain import (
        S13ExactEdgeComponentEvidence,
        canonical_s13_support_sha256,
        make_s13_edge_component_observation,
        make_s13_unevaluable_edge_component_observation,
    )

    components = context.get("component_audits")
    supports_value = context.get("support_by_block")
    rows_value = context.get("forward_rows_by_block")
    lags = np.asarray(context.get("lags"), dtype=np.float64)
    if (
        not isinstance(components, Sequence)
        or not isinstance(supports_value, Mapping)
        or not isinstance(rows_value, Mapping)
        or lags.ndim != 1
        or lags.size != 13
    ):
        raise ValueError("S1.3 C2E forward evidence context is invalid")
    for component in components:
        if not isinstance(component, Mapping):
            raise ValueError("S1.3 C2E component forward evidence is invalid")
        blocks = tuple(int(value) for value in component["block_indices"])
        supports = []
        for value in blocks:
            if value not in supports_value:
                continue
            coordinate_pair = supports_value[value]
            if (
                not isinstance(coordinate_pair, tuple)
                or len(coordinate_pair) != 2
            ):
                raise ValueError("S1.3 C2E forward support coordinates are invalid")
            supports.append(np.column_stack(coordinate_pair).astype(np.int32))
        if not supports:
            continue
        support = np.unique(np.concatenate(supports), axis=0)
        score_sum = np.zeros(lags.shape, np.float64)
        correlation_sum = np.zeros(lags.shape, np.float64)
        agreement_sum = np.zeros(lags.shape, np.float64)
        signed_agreement_sum = np.zeros(lags.shape, np.float64)
        counts = np.zeros(lags.shape, np.int32)
        for block in blocks:
            block_rows = rows_value.get(block, ())
            if not isinstance(block_rows, Sequence):
                raise ValueError("S1.3 C2E block forward evidence is invalid")
            for row in block_rows:
                if not isinstance(row, Mapping):
                    raise ValueError("S1.3 C2E forward hypothesis row is invalid")
                lag = float(row["lag"])
                block_normal_x = float(row.get("normal_x", 1.0))
                block_normal_y = float(row.get("normal_y", 0.0))
                if block_normal_x < 0.0 or (
                    block_normal_x == 0.0 and block_normal_y < 0.0
                ):
                    lag = -lag
                indices = np.flatnonzero(np.isclose(lags, lag, rtol=0.0, atol=1e-12))
                if indices.size != 1:
                    raise ValueError("S1.3 C2E forward lag is outside the frozen grid")
                index = int(indices[0])
                count = int(row["support"])
                if count <= 0:
                    continue
                correlation = float(row["correlation"])
                orientation = float(row["orientation_difference"])
                orientation_agreement = float(row.get(
                    "orientation_agreement", math.cos(math.radians(orientation))
                ))
                signed_gradient_agreement = float(row.get(
                    "signed_gradient_agreement", orientation_agreement
                ))
                score_sum[index] += count * float(row["score"])
                correlation_sum[index] += count * correlation
                agreement_sum[index] += count * orientation_agreement
                signed_agreement_sum[index] += count * signed_gradient_agreement
                counts[index] += count
        finite = counts > 0
        forward_scores = np.full(lags.shape, -math.inf, np.float64)
        forward_correlations = np.zeros(lags.shape, np.float64)
        forward_agreements = np.zeros(lags.shape, np.float64)
        forward_signed_agreements = np.zeros(lags.shape, np.float64)
        forward_scores[finite] = score_sum[finite] / counts[finite]
        forward_correlations[finite] = correlation_sum[finite] / counts[finite]
        forward_agreements[finite] = agreement_sum[finite] / counts[finite]
        forward_signed_agreements[finite] = signed_agreement_sum[finite] / counts[finite]
        try:
            observation, evidence = make_s13_edge_component_observation(
                pair_index=int(pair_index),
                component_id=int(component["component_id"]),
                source_indices=(int(pair_index), int(pair_index) + 1),
                support_xy=support,
                left_magnitude=left_features.gradient,
                right_magnitude=right_features.gradient,
                left_gradient_x=left_features.gradient_x,
                left_gradient_y=left_features.gradient_y,
                right_gradient_x=right_features.gradient_x,
                right_gradient_y=right_features.gradient_y,
                lags=lags,
                block_indices=blocks,
                minimum_correlation=float(getattr(config, "minimum_c2e_correlation", 0.75)),
                minimum_uniqueness_fraction=float(getattr(config, "minimum_c2e_uniqueness_fraction", 0.10)),
                maximum_orientation_difference_degrees=float(
                    config.maximum_orientation_difference_degrees
                ),
                maximum_forward_reverse_discrepancy_px=float(
                    getattr(config, "maximum_forward_reverse_discrepancy_px", 0.5)
                ),
                minimum_signed_gradient_agreement=float(
                    getattr(config, "minimum_signed_gradient_agreement", 0.10)
                ),
                precomputed_forward_scores=forward_scores,
                precomputed_forward_correlations=forward_correlations,
                precomputed_forward_agreements=forward_agreements,
                precomputed_forward_signed_agreements=forward_signed_agreements,
                precomputed_forward_support_counts=counts,
            )
        except ValueError as exc:
            observation, evidence = make_s13_unevaluable_edge_component_observation(
                pair_index=int(pair_index),
                component_id=int(component["component_id"]),
                source_indices=(int(pair_index), int(pair_index) + 1),
                support_xy=support,
                left_magnitude=left_features.gradient,
                left_gradient_x=left_features.gradient_x,
                left_gradient_y=left_features.gradient_y,
                lags=lags,
                block_indices=blocks,
                reason=f"symmetric_hypothesis_failed:{type(exc).__name__}",
            )
        if component.get("ambiguous") is True:
            observation = replace(
                observation,
                evidence_state="ambiguous",
                exclusion_reasons=tuple(sorted(set(
                    (*observation.exclusion_reasons, "component_match_ambiguous")
                ))),
            )
        global_support = np.array(evidence.support_xy, copy=True)
        global_support[:, 0] += int(global_x_offset)
        global_evidence = S13ExactEdgeComponentEvidence(
            support_xy=global_support,
            forward_scores=evidence.forward_scores,
            reverse_scores=evidence.reverse_scores,
            forward_correlations=evidence.forward_correlations,
            reverse_correlations=evidence.reverse_correlations,
            forward_support_counts=evidence.forward_support_counts,
            reverse_support_counts=evidence.reverse_support_counts,
        )
        bbox = list(observation.global_bbox_xyxy)
        bbox[0] += int(global_x_offset)
        bbox[2] += int(global_x_offset)
        exact_evidence_sink.append((replace(
            observation,
            global_bbox_xyxy=tuple(bbox),
            fitted_line_offset=(
                observation.fitted_line_offset
                + observation.normal_x * int(global_x_offset)
            ),
            mask_sha256=canonical_s13_support_sha256(global_support),
        ), global_evidence))


def pair_edge_registration_metrics(
    left: np.ndarray | SeamStructureFeatures,
    right: np.ndarray | SeamStructureFeatures,
    left_valid: np.ndarray,
    right_valid: np.ndarray,
    seam_x_by_row: np.ndarray,
    *,
    config: "S13M51R2Config",
    exact_evidence_sink: list[tuple[object, object]] | None = None,
    forward_evidence_sink: list[Mapping[str, object]] | None = None,
    pair_index: int = -1,
    global_x_offset: int = 0,
) -> dict[str, object]:
    """Measure pair-local strong-edge registration along each edge's normal.

    This compares the two real pair sources in their shared canvas/crop domain.
    It deliberately does not inspect an owner-composed preview, infer validity
    from RGB values, or create a warp.
    """

    left_features = _edge_features(left, name="left")
    right_features = _edge_features(right, name="right")
    if left_features.shape != right_features.shape:
        raise ValueError("pair edge feature shapes differ")
    height, width = left_features.shape
    left_mask = np.asarray(left_valid)
    right_mask = np.asarray(right_valid)
    if left_mask.shape != (height, width) or right_mask.shape != (height, width):
        raise ValueError("pair edge valid mask shape differs from the features")
    if left_mask.dtype != np.bool_ or right_mask.dtype != np.bool_:
        raise ValueError("pair edge valid masks must be boolean")
    seam_value = np.asarray(seam_x_by_row)
    if seam_value.shape != (height,) or not np.isfinite(seam_value).all():
        raise ValueError("pair edge seam must contain one finite value per row")
    seam = np.rint(seam_value).astype(np.int32)
    if not np.allclose(seam_value, seam, atol=0.0):
        raise ValueError("pair edge seam coordinates must be integral")

    block_height = int(config.block_height_px)
    block_stride = int(config.block_stride_px)
    seam_radius = int(config.seam_support_radius_px)
    halo_radius = int(config.feature_halo_radius_px)
    minimum_length = int(config.minimum_edge_component_length_px)
    minimum_count = int(config.minimum_edge_support_count)
    maximum_lag = float(config.normal_search_maximum_px)
    lag_step = float(config.normal_search_step_px)
    if height < minimum_length or width < 3:
        return _edge_registration_empty(reason="image_too_small")

    starts = list(range(0, max(height - block_height + 1, 1), block_stride))
    final_start = max(0, height - block_height)
    if not starts or starts[-1] != final_start:
        starts.append(final_start)
    lags = np.arange(-maximum_lag, maximum_lag + 0.25 * lag_step, lag_step)
    accepted: list[dict[str, object]] = []
    component_observations: list[dict[str, object]] = []
    block_audits: list[dict[str, object]] = []
    ambiguous = False
    nonunique_search_observed = False
    component_local = getattr(config, "component_local_ambiguity_enabled", False) is True
    exact_support_by_block: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    forward_rows_by_block: dict[int, tuple[Mapping[str, object], ...]] = {}

    for block_index, y0 in enumerate(starts):
        y1 = min(height, y0 + block_height)
        columns = np.arange(width, dtype=np.int32)[None, :]
        distance = np.abs(columns - seam[y0:y1, None])
        common = left_mask[y0:y1] & right_mask[y0:y1]
        local_values = left_features.gradient[y0:y1][common & (distance <= halo_radius)]
        finite_positive = local_values[np.isfinite(local_values)]
        if finite_positive.size < minimum_count:
            block_audits.append({"block_index": block_index, "y0": y0, "y1": y1,
                                 "supported": False, "reason": "insufficient_local_gradient"})
            continue
        threshold = max(8.0, float(np.percentile(finite_positive, 75.0)))
        strong = (
            common
            & (distance <= halo_radius)
            & np.isfinite(left_features.gradient[y0:y1])
            & (left_features.gradient[y0:y1] >= threshold)
        )
        label_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
            strong.astype(np.uint8), connectivity=8
        )
        candidates: list[tuple[float, int, np.ndarray, np.ndarray]] = []
        for label in range(1, label_count):
            component = labels == label
            component_rows, component_columns = np.nonzero(component)
            count = int(len(component_rows))
            if count < minimum_count:
                continue
            length = float(max(stats[label, cv2.CC_STAT_WIDTH], stats[label, cv2.CC_STAT_HEIGHT]))
            if length < minimum_length:
                continue
            if not np.any(component & (distance <= seam_radius)):
                continue
            strength = float(np.sum(left_features.gradient[y0:y1][component]))
            candidates.append((strength, label, component_rows, component_columns))
        if not candidates:
            block_audits.append({"block_index": block_index, "y0": y0, "y1": y1,
                                 "supported": False, "reason": "no_cross_seam_component",
                                 "strong_gradient_threshold": threshold})
            continue
        candidates.sort(reverse=True, key=lambda item: item[0])
        component_ambiguous = False
        if len(candidates) > 1 and candidates[1][0] >= 0.80 * candidates[0][0]:
            first_center = np.asarray(
                (np.mean(candidates[0][3]), np.mean(candidates[0][2])), dtype=np.float64
            )
            second_center = np.asarray(
                (np.mean(candidates[1][3]), np.mean(candidates[1][2])), dtype=np.float64
            )
            # The two Sobel flanks of one thick antialiased edge are one
            # physical layer.  Only spatially separate comparable components
            # constitute competing layers.
            if float(np.linalg.norm(first_center - second_center)) > 6.0:
                component_ambiguous = True
                ambiguous = True
        _strength, _label, component_rows, component_columns = candidates[0]
        ys = (component_rows + y0).astype(np.float64)
        xs = component_columns.astype(np.float64)
        gx = left_features.gradient_x[ys.astype(np.int32), xs.astype(np.int32)].astype(np.float64)
        gy = left_features.gradient_y[ys.astype(np.int32), xs.astype(np.int32)].astype(np.float64)
        magnitude = np.hypot(gx, gy)
        angles = np.mod(np.arctan2(gy, gx), np.pi)
        doubled = np.sum(magnitude * np.exp(2j * angles))
        if abs(doubled) <= 1e-9:
            ambiguous = True
            block_audits.append({"block_index": block_index, "y0": y0, "y1": y1,
                                 "supported": False, "reason": "ambiguous_normal"})
            continue
        normal_angle = float(np.mod(0.5 * np.angle(doubled), np.pi))
        nx, ny = math.cos(normal_angle), math.sin(normal_angle)
        if exact_evidence_sink is not None or forward_evidence_sink is not None:
            # Retain the already allocated coordinate vectors by reference.
            # Materialize canonical int32 Nx2 support only for a pair that is
            # subsequently confirmed visual-suspect.
            exact_support_by_block[int(block_index)] = (xs, ys)
        lag_rows: list[dict[str, float | int]] = []
        for lag in lags:
            sampled_magnitude, sampled_gx, sampled_gy, sample_valid = _bilinear_feature_samples(
                right_features, right_mask, xs + float(lag) * nx, ys + float(lag) * ny
            )
            keep = sample_valid & np.isfinite(sampled_magnitude)
            if int(keep.sum()) < minimum_count:
                continue
            left_magnitude = magnitude[keep]
            right_magnitude = sampled_magnitude[keep]
            denominator = float(np.linalg.norm(left_magnitude) * np.linalg.norm(right_magnitude))
            if denominator <= 1e-9:
                continue
            correlation = float(np.clip(np.dot(left_magnitude, right_magnitude) / denominator, 0.0, 1.0))
            right_angle = np.mod(np.arctan2(sampled_gy[keep], sampled_gx[keep]), np.pi)
            differences = _modulo_pi_orientation_difference(angles[keep], right_angle)
            orientation_difference = float(np.degrees(np.average(differences, weights=left_magnitude)))
            orientation_agreement = math.cos(math.radians(orientation_difference))
            orientation_denominator = magnitude[keep] * np.hypot(
                sampled_gx[keep], sampled_gy[keep]
            )
            orientation_usable = orientation_denominator > 1e-12
            signed_gradient_agreement = (
                float(np.average(
                    (
                        gx[keep][orientation_usable] * sampled_gx[keep][orientation_usable]
                        + gy[keep][orientation_usable] * sampled_gy[keep][orientation_usable]
                    ) / orientation_denominator[orientation_usable],
                    weights=left_magnitude[orientation_usable],
                ))
                if np.any(orientation_usable) else 0.0
            )
            score = (
                correlation
                * max(0.0, orientation_agreement)
                * max(0.0, signed_gradient_agreement)
            )
            lag_rows.append({"lag": float(lag), "correlation": correlation,
                             "orientation_difference": orientation_difference,
                             "orientation_agreement": orientation_agreement,
                             "signed_gradient_agreement": signed_gradient_agreement,
                             "score": score, "support": int(keep.sum()),
                             "normal_x": float(nx), "normal_y": float(ny)})
        if not lag_rows:
            if exact_evidence_sink is not None or forward_evidence_sink is not None:
                forward_rows_by_block[int(block_index)] = ()
            audit = {
                "block_index": block_index, "y0": y0, "y1": y1,
                "supported": False, "reason": "no_valid_normal_samples",
                "component_ambiguous": False,
                "centroid_x": float(np.mean(xs)), "centroid_y": float(np.mean(ys)),
                "bbox_x0": int(np.floor(np.min(xs))),
                "bbox_x1": int(np.ceil(np.max(xs))) + 1,
                "bbox_y0": int(np.floor(np.min(ys))),
                "bbox_y1": int(np.ceil(np.max(ys))) + 1,
                "normal_angle_degrees_modulo_180": float(math.degrees(normal_angle)),
                "best_lag_px": 0.0, "correlation": 0.0,
                "uniqueness_margin": 0.0, "orientation_difference_degrees": 0.0,
                "support_count": 0,
            }
            block_audits.append(audit)
            if component_local and exact_evidence_sink is not None:
                component_observations.append(audit)
            continue
        if exact_evidence_sink is not None or forward_evidence_sink is not None:
            forward_rows_by_block[int(block_index)] = tuple(lag_rows)
        best = max(lag_rows, key=lambda row: (float(row["score"]), -abs(float(row["lag"]))))
        # Adjacent half-pixel hypotheses sample the same Sobel response and
        # are not independent alternatives.  Compare against a hypothesis at
        # least 1.5 px away (three configured samples) for uniqueness.
        independent_lag_distance = max(1.5, 3.0 * lag_step)
        separated = [
            row for row in lag_rows
            if abs(float(row["lag"]) - float(best["lag"])) >= independent_lag_distance
        ]
        second_score = max((float(row["score"]) for row in separated), default=0.0)
        uniqueness = (float(best["score"]) - second_score) / max(abs(float(best["score"])), 1e-9)
        failure = None
        if float(best["correlation"]) < float(config.minimum_edge_correlation):
            failure = "correlation_below_minimum"
        elif float(best["orientation_difference"]) > float(config.maximum_orientation_difference_degrees):
            failure = "orientation_difference_exceeded"
        elif uniqueness < float(config.minimum_uniqueness_fraction):
            failure = "normal_search_not_unique"
            nonunique_search_observed = True
        audit = {
            "block_index": block_index, "y0": y0, "y1": y1,
            "supported": failure is None, "reason": failure,
            "best_lag_px": float(best["lag"]),
            "correlation": float(best["correlation"]),
            "uniqueness_margin": float(uniqueness),
            "orientation_difference_degrees": float(best["orientation_difference"]),
            "support_count": int(best["support"]),
            "strong_gradient_threshold": threshold,
            "normal_angle_degrees_modulo_180": float(math.degrees(normal_angle)),
        }
        if component_local:
            audit.update({
                "component_ambiguous": bool(
                    component_ambiguous or failure == "normal_search_not_unique"
                ),
                "centroid_x": float(np.mean(xs)),
                "centroid_y": float(np.mean(ys)),
                "bbox_x0": int(np.floor(np.min(xs))),
                "bbox_x1": int(np.ceil(np.max(xs))) + 1,
                "bbox_y0": int(np.floor(np.min(ys))),
                "bbox_y1": int(np.ceil(np.max(ys))) + 1,
            })
        block_audits.append(audit)
        if component_local and (
            failure is None
            or audit.get("component_ambiguous") is True
            or exact_evidence_sink is not None
        ):
            component_observations.append(audit)
        if failure is None:
            accepted.append(audit)

    if component_local and (
        accepted
        or (
            (exact_evidence_sink is not None or forward_evidence_sink is not None)
            and component_observations
        )
    ):
        if accepted:
            legacy_absolute_lags = np.abs(np.asarray(
                [row["best_lag_px"] for row in accepted], dtype=np.float64
            ))
            legacy_pair_decision = {
                "evaluable": True,
                "median_supported_abs_lag_px": float(np.median(legacy_absolute_lags)),
                "p95_supported_abs_lag_px": float(
                    np.percentile(legacy_absolute_lags, 95.0)
                ),
                "multiple_layer_or_ambiguous": bool(ambiguous),
            }
        else:
            legacy_pair_decision = {
                "evaluable": False,
                "median_supported_abs_lag_px": None,
                "p95_supported_abs_lag_px": None,
                "multiple_layer_or_ambiguous": bool(ambiguous),
            }
        component_summary = _component_local_edge_summary(
            component_observations, config=config
        )
        forward_context: Mapping[str, object] = {
            "component_audits": tuple(component_summary["component_audits"]),
            "support_by_block": exact_support_by_block,
            "forward_rows_by_block": dict(forward_rows_by_block),
            "lags": np.asarray(lags, dtype=np.float64),
        }
        if forward_evidence_sink is not None:
            forward_evidence_sink.append(forward_context)
        if exact_evidence_sink is not None:
            append_s13_exact_component_evidence_from_forward_context(
                forward_context,
                left_features=left_features,
                right_features=right_features,
                config=config,
                exact_evidence_sink=exact_evidence_sink,
                pair_index=pair_index,
                global_x_offset=global_x_offset,
            )
        actionable = list(component_summary["actionable_components"])
        if not actionable:
            result = _edge_registration_empty(reason="no_unambiguous_edge_component")
            result.update({
                "schema": "gemini305-video-s13-oblique-structure-audit/v2",
                "multiple_layer_or_ambiguous": True,
                "component_audits": component_summary["component_audits"],
                "actionable_component_count": 0,
                "ambiguous_component_count": component_summary["ambiguous_component_count"],
                "block_audits": block_audits,
                "legacy_pair_decision": legacy_pair_decision,
            })
            return result
        absolute_lags = np.abs(np.asarray(
            [row["best_lag_px"] for row in actionable], dtype=np.float64
        ))
        supported_blocks = {
            int(block_index)
            for row in actionable
            for block_index in row["block_indices"]
        }
        return {
            "schema": "gemini305-video-s13-oblique-structure-audit/v2",
            "evaluable": True,
            "reason": None,
            "supported_block_count": len(supported_blocks),
            "supported_component_count": len(actionable),
            "actionable_component_count": len(actionable),
            "ambiguous_component_count": component_summary["ambiguous_component_count"],
            "component_audits": component_summary["component_audits"],
            "block_best_lag_px": [float(row["best_lag_px"]) for row in actionable],
            "median_supported_abs_lag_px": float(np.median(absolute_lags)),
            "p95_supported_abs_lag_px": float(max(
                float(row["absolute_lag_p95_px"]) for row in actionable
            )),
            "maximum_supported_abs_lag_px": float(max(
                abs(float(row["best_lag_px"])) for row in actionable
            )),
            "minimum_correlation": float(min(
                float(row["minimum_correlation"]) for row in actionable
            )),
            "minimum_uniqueness_margin": float(min(
                float(row["minimum_uniqueness_margin"]) for row in actionable
            )),
            "maximum_orientation_difference_degrees": float(max(
                float(row["maximum_orientation_difference_degrees"])
                for row in actionable
            )),
            "multiple_layer_or_ambiguous": bool(
                component_summary["ambiguous_component_count"]
            ),
            "ambiguity_scope": "component_local",
            "block_audits": block_audits,
            "legacy_pair_decision": legacy_pair_decision,
        }

    if not accepted:
        result = _edge_registration_empty(reason="insufficient_unique_strong_edge_support")
        result["multiple_layer_or_ambiguous"] = bool(
            ambiguous or nonunique_search_observed
        )
        result["block_audits"] = block_audits
        return result
    absolute_lags = np.abs(np.asarray([row["best_lag_px"] for row in accepted], dtype=np.float64))
    return {
        "schema": "gemini305-video-s13-oblique-structure-audit/v1",
        "evaluable": True,
        "reason": None,
        "supported_block_count": len({int(row["block_index"]) for row in accepted}),
        "supported_component_count": len(accepted),
        "block_best_lag_px": [float(row["best_lag_px"]) for row in accepted],
        "median_supported_abs_lag_px": float(np.median(absolute_lags)),
        "p95_supported_abs_lag_px": float(np.percentile(absolute_lags, 95.0)),
        "maximum_supported_abs_lag_px": float(np.max(absolute_lags)),
        "minimum_correlation": float(min(float(row["correlation"]) for row in accepted)),
        "minimum_uniqueness_margin": float(min(float(row["uniqueness_margin"]) for row in accepted)),
        "maximum_orientation_difference_degrees": float(max(
            float(row["orientation_difference_degrees"]) for row in accepted
        )),
        "multiple_layer_or_ambiguous": ambiguous,
        "block_audits": block_audits,
    }


def _horizontal_edge_mismatch(left: np.ndarray, right: np.ndarray) -> float:
    """Measure row-step disagreement only where either side has a real horizontal edge."""

    left_f = np.asarray(left, dtype=np.float64)
    right_f = np.asarray(right, dtype=np.float64)
    strength = np.maximum(np.abs(left_f), np.abs(right_f))
    finite = np.isfinite(strength) & np.isfinite(left_f) & np.isfinite(right_f)
    positive = strength[finite & (strength >= 8.0)]
    if positive.size < 16:
        return 0.0
    threshold = max(8.0, float(np.median(positive)))
    support = finite & (strength >= threshold)
    mismatch = np.abs(left_f[support] - right_f[support])
    return float(np.median(mismatch)) if mismatch.size >= 16 else 0.0


def seam_structure_metrics(
    image: np.ndarray | SeamStructureFeatures,
    seam_x_by_row: np.ndarray,
    *,
    radius_px: int = 4,
) -> dict[str, object]:
    """Measure local hard-owner structure without treating black RGB as invalid."""

    features = image if isinstance(image, SeamStructureFeatures) else prepare_seam_structure(image)
    if features is None:
        return {"evaluable": False, "reason": "invalid_image", **{name: None for name in METRIC_NAMES}}
    height, width = features.shape
    seam = np.asarray(seam_x_by_row, dtype=np.int32)
    if seam.shape != (height,) or width < 5:
        return {"evaluable": False, "reason": "invalid_seam", **{name: None for name in METRIC_NAMES}}
    rows = np.arange(height)
    usable = (seam >= 2) & (seam <= width - 3)
    if int(usable.sum()) < 16:
        return {"evaluable": False, "reason": "insufficient_corridor_rows", **{name: None for name in METRIC_NAMES}}
    rows, seam = rows[usable], seam[usable]
    left = features.lab[rows, seam - 1]
    right = features.lab[rows, seam]
    color = np.linalg.norm(left - right, axis=1)
    grad_jump = np.abs(
        features.gradient[rows, seam - 1] - features.gradient[rows, seam]
    )
    double = np.maximum(
        features.laplacian[rows, seam - 1], features.laplacian[rows, seam]
    )
    motion = np.abs(features.gray[rows, seam - 2] - features.gray[rows, seam + 1])
    thin = np.maximum(
        np.abs(features.gradient_x[rows, seam - 1]),
        np.abs(features.gradient_x[rows, seam]),
    )
    horizontal = _horizontal_edge_mismatch(
        features.gradient_y[rows, seam - 1], features.gradient_y[rows, seam]
    )
    values = {
        "color_jump": _finite_summary(color),
        "gradient_jump": _finite_summary(grad_jump),
        "double_edge": _finite_summary(double),
        "motion_residual": _finite_summary(motion),
        "thin_structure": _finite_summary(thin),
        "horizontal_edge_mismatch": horizontal,
    }
    if any(value is None for value in values.values()):
        return {"evaluable": False, "reason": "nonfinite_or_sparse_metrics", **values}
    score = float(
        0.25 * values["color_jump"]
        + 0.15 * values["gradient_jump"]
        + 0.15 * values["double_edge"]
        + 0.10 * values["motion_residual"]
        + 0.10 * values["thin_structure"]
        + 0.25 * values["horizontal_edge_mismatch"]
    )
    return {"evaluable": True, "reason": None, **values, "score": score, "radius_px": int(radius_px)}


def structurally_non_degrading(
    before: Mapping[str, object],
    after: Mapping[str, object],
    *,
    relative_tolerance: float = 0.02,
) -> tuple[bool, str | None]:
    """Require aggregate and every protected component to remain locally non-worse."""

    if before.get("evaluable") is not True or after.get("evaluable") is not True:
        return False, "insufficient_relative_structure_evidence"
    for name in (*METRIC_NAMES, "score"):
        left, right = before.get(name), after.get(name)
        if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
            return False, f"unevaluable_{name}"
        allowance = max(0.25, abs(float(left)) * float(relative_tolerance))
        if float(right) > float(left) + allowance:
            return False, f"{name}_degraded"
    return True, None


def mean_seam_structure_metrics(
    metrics: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Average already comparable seam audits without hiding an unevaluable path."""

    rows = tuple(metrics)
    if not rows or any(row.get("evaluable") is not True for row in rows):
        return {
            "evaluable": False,
            "reason": "unevaluable_symmetric_seam_path",
            **{name: None for name in METRIC_NAMES},
            "score": None,
        }
    values: dict[str, float] = {}
    for name in (*METRIC_NAMES, "score"):
        samples = [row.get(name) for row in rows]
        if any(not isinstance(value, (int, float)) or not math.isfinite(float(value)) for value in samples):
            return {
                "evaluable": False,
                "reason": f"nonfinite_symmetric_{name}",
                **{metric_name: None for metric_name in METRIC_NAMES},
                "score": None,
            }
        values[name] = float(np.mean(np.asarray(samples, dtype=np.float64)))
    return {
        "evaluable": True,
        "reason": None,
        **values,
        "radius_px": max(int(row.get("radius_px", 0)) for row in rows),
        "path_count": len(rows),
    }


def symmetric_seam_structure_metrics(
    before: np.ndarray | SeamStructureFeatures,
    after: np.ndarray | SeamStructureFeatures,
    base_seam_x_by_row: np.ndarray,
    candidate_seam_x_by_row: np.ndarray,
) -> tuple[dict[str, object], dict[str, object]]:
    """Evaluate both images on both seam paths so moving a seam cannot game selection."""

    paths = (base_seam_x_by_row, candidate_seam_x_by_row)
    before_metrics = mean_seam_structure_metrics(
        tuple(seam_structure_metrics(before, path) for path in paths)
    )
    after_metrics = mean_seam_structure_metrics(
        tuple(seam_structure_metrics(after, path) for path in paths)
    )
    before_metrics["comparison_policy"] = "symmetric_base_and_candidate_paths"
    after_metrics["comparison_policy"] = "symmetric_base_and_candidate_paths"
    return before_metrics, after_metrics


def _horizontal_profile(
    features: SeamStructureFeatures,
    seam: np.ndarray,
    offsets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rows = np.arange(features.shape[0], dtype=np.int32)[:, None]
    columns = seam[:, None] + offsets[None, :]
    in_bounds = (columns >= 0) & (columns < features.shape[1])
    clipped = np.clip(columns, 0, features.shape[1] - 1)
    gx = features.gradient_x[rows, clipped]
    gy = features.gradient_y[rows, clipped]
    horizontal = in_bounds & (np.abs(gy) >= np.abs(gx)) & (np.abs(gy) >= 8.0)
    count = horizontal.sum(axis=1)
    profile = np.divide(
        np.where(horizontal, gy, 0.0).sum(axis=1),
        np.maximum(count, 1),
        dtype=np.float64,
    )
    profile = cv2.GaussianBlur(profile.astype(np.float32)[:, None], (1, 5), 0).ravel()
    return profile.astype(np.float64), count > 0


def _profile_correlation(
    left: np.ndarray,
    right: np.ndarray,
    left_support: np.ndarray,
    right_support: np.ndarray,
    lag: int,
) -> tuple[float | None, int]:
    if lag < 0:
        left_values, right_values = left[:lag], right[-lag:]
        support = left_support[:lag] & right_support[-lag:]
    elif lag > 0:
        left_values, right_values = left[lag:], right[:-lag]
        support = left_support[lag:] & right_support[:-lag]
    else:
        left_values, right_values = left, right
        support = left_support & right_support
    count = int(support.sum())
    minimum_support = max(4, int(math.ceil(len(left) * 0.01)))
    if count < minimum_support:
        return None, count
    a = left_values[support] - float(np.mean(left_values[support]))
    b = right_values[support] - float(np.mean(right_values[support]))
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator <= 1e-6:
        return None, count
    return float(np.clip(np.dot(a, b) / denominator, -1.0, 1.0)), count


def long_horizontal_structure_metrics(
    image: np.ndarray | SeamStructureFeatures,
    seam_x_by_row: np.ndarray,
    *,
    maximum_lag_px: int = 8,
) -> dict[str, object]:
    """Audit long horizontal-edge continuity outside the seam cost's near pixels."""

    features = image if isinstance(image, SeamStructureFeatures) else prepare_seam_structure(image)
    if features is None:
        return {"observed": False, "reason": "invalid_image", "line_score": None}
    seam = np.asarray(seam_x_by_row, dtype=np.int32)
    if seam.shape != (features.shape[0],):
        return {"observed": False, "reason": "invalid_seam", "line_score": None}
    left, left_support = _horizontal_profile(
        features, seam, np.arange(-16, -8, dtype=np.int32)
    )
    right, right_support = _horizontal_profile(
        features, seam, np.arange(9, 17, dtype=np.int32)
    )
    correlations: dict[int, tuple[float, int]] = {}
    for lag in range(-int(maximum_lag_px), int(maximum_lag_px) + 1):
        correlation, support = _profile_correlation(
            left, right, left_support, right_support, lag
        )
        if correlation is not None:
            correlations[lag] = (correlation, support)
    if not correlations:
        return {
            "observed": False,
            "reason": "insufficient_horizontal_edge_support",
            "line_score": None,
            "supported_lag_count": 0,
        }
    best_lag, (best_correlation, best_support) = max(
        correlations.items(), key=lambda item: (item[1][0], -abs(item[0]))
    )
    zero_correlation = correlations.get(0, (-1.0, 0))[0]
    observed = bool(best_correlation >= 0.50)
    line_score = float(
        abs(best_lag) + 4.0 * max(0.0, best_correlation - zero_correlation)
    )
    return {
        "observed": observed,
        "reason": None if observed else "weak_horizontal_edge_correlation",
        "best_vertical_lag_px": int(best_lag),
        "absolute_best_vertical_lag_px": abs(int(best_lag)),
        "best_correlation": float(best_correlation),
        "zero_lag_correlation": float(zero_correlation),
        "best_minus_zero_correlation": float(best_correlation - zero_correlation),
        "line_score": line_score,
        "support_row_count": int(best_support),
        "supported_lag_count": len(correlations),
        "maximum_lag_px": int(maximum_lag_px),
        "held_out_from_near_seam_cost": True,
    }


def long_horizontal_structure_nondegrading(
    before: Mapping[str, object],
    after: Mapping[str, object],
) -> tuple[bool, str | None]:
    """Reject both lost support and newly introduced, visibly displaced structure."""

    if before.get("observed") is not True:
        if after.get("observed") is True:
            after_score = after.get("line_score")
            if not isinstance(after_score, (int, float)):
                return False, "horizontal_structure_score_unevaluable"
            if float(after_score) > 0.25:
                return False, "horizontal_structure_new_misalignment"
        return True, None
    if after.get("observed") is not True:
        return False, "horizontal_structure_support_lost"
    before_score, after_score = before.get("line_score"), after.get("line_score")
    if not isinstance(before_score, (int, float)) or not isinstance(after_score, (int, float)):
        return False, "horizontal_structure_score_unevaluable"
    allowance = max(0.25, abs(float(before_score)) * 0.10)
    if float(after_score) > float(before_score) + allowance:
        return False, "horizontal_structure_continuity_degraded"
    return True, None


def sequence_structure_decision(
    before: Sequence[Mapping[str, object]],
    after: Sequence[Mapping[str, object]],
    *,
    maximum_component_failure_fraction: float = 0.05,
    maximum_pair_score_failure_fraction: float = 0.0,
    minimum_mean_improvement_fraction: float = 0.005,
) -> tuple[bool, dict[str, object]]:
    """Make a robust sequence decision while bounding every pair's total score.

    A small number of protected-component outliers may not veto a clearly
    better long sequence, but no pair may materially regress in total score.
    Short sequences remain strict because their rounded-down failure budget is
    zero.
    """

    left = tuple(before)
    right = tuple(after)
    if len(left) != len(right) or not left:
        return False, {
            "eligible": False,
            "reason": "mismatched_or_empty_sequence_metrics",
            "pair_count": len(left),
        }
    if not 0.0 <= maximum_component_failure_fraction < 1.0:
        raise ValueError("maximum_component_failure_fraction must be in [0, 1)")
    if not 0.0 <= maximum_pair_score_failure_fraction < 1.0:
        raise ValueError("maximum_pair_score_failure_fraction must be in [0, 1)")
    if not 0.0 <= minimum_mean_improvement_fraction < 1.0:
        raise ValueError("minimum_mean_improvement_fraction must be in [0, 1)")

    decisions = [structurally_non_degrading(a, b) for a, b in zip(left, right, strict=True)]
    unevaluable = [
        index for index, (a, b) in enumerate(zip(left, right, strict=True))
        if a.get("evaluable") is not True or b.get("evaluable") is not True
    ]
    failure_indices = [index for index, (passed, _reason) in enumerate(decisions) if not passed]
    score_regressions: list[int] = []
    catastrophic_score_regressions: list[dict[str, object]] = []
    catastrophic_component_regressions: list[dict[str, object]] = []
    before_scores: list[float] = []
    after_scores: list[float] = []
    for index, (a, b) in enumerate(zip(left, right, strict=True)):
        before_score, after_score = a.get("score"), b.get("score")
        if not isinstance(before_score, (int, float)) or not isinstance(after_score, (int, float)):
            score_regressions.append(index)
            continue
        before_value, after_value = float(before_score), float(after_score)
        if not math.isfinite(before_value) or not math.isfinite(after_value):
            score_regressions.append(index)
            continue
        before_scores.append(before_value)
        after_scores.append(after_value)
        allowance = max(0.25, abs(before_value) * 0.02)
        if after_value > before_value + allowance:
            score_regressions.append(index)
        catastrophic_score_allowance = max(2.0, abs(before_value) * 0.25)
        if after_value > before_value + catastrophic_score_allowance:
            catastrophic_score_regressions.append(
                {
                    "pair_index": index,
                    "before": before_value,
                    "after": after_value,
                    "allowance": catastrophic_score_allowance,
                }
            )
        for name in METRIC_NAMES:
            before_component, after_component = a.get(name), b.get(name)
            if not isinstance(before_component, (int, float)) or not isinstance(
                after_component, (int, float)
            ):
                continue
            component_allowance = max(2.0, abs(float(before_component)) * 0.25)
            if float(after_component) > float(before_component) + component_allowance:
                catastrophic_component_regressions.append(
                    {
                        "pair_index": index,
                        "metric": name,
                        "before": float(before_component),
                        "after": float(after_component),
                        "allowance": component_allowance,
                    }
                )

    pair_count = len(left)
    failure_budget = int(math.floor(pair_count * maximum_component_failure_fraction))
    score_failure_budget = int(math.floor(pair_count * maximum_pair_score_failure_fraction))
    before_mean = float(np.mean(before_scores)) if len(before_scores) == pair_count else None
    after_mean = float(np.mean(after_scores)) if len(after_scores) == pair_count else None
    aggregate_improvement = (
        None
        if before_mean is None or after_mean is None or before_mean <= 0.0
        else float((before_mean - after_mean) / before_mean)
    )
    eligible = bool(
        not unevaluable
        and len(failure_indices) <= failure_budget
        and len(score_regressions) <= score_failure_budget
        and not catastrophic_score_regressions
        and not catastrophic_component_regressions
        and aggregate_improvement is not None
        and aggregate_improvement >= minimum_mean_improvement_fraction
    )
    reason = None
    if unevaluable:
        reason = "unevaluable_sequence_pairs"
    elif len(failure_indices) > failure_budget:
        reason = "component_failure_budget_exceeded"
    elif len(score_regressions) > score_failure_budget:
        reason = "pair_total_score_failure_budget_exceeded"
    elif catastrophic_score_regressions:
        reason = "catastrophic_pair_total_score_regression"
    elif catastrophic_component_regressions:
        reason = "catastrophic_component_regression"
    elif aggregate_improvement is None or aggregate_improvement < minimum_mean_improvement_fraction:
        reason = "aggregate_improvement_not_proven"
    return eligible, {
        "eligible": eligible,
        "reason": reason,
        "pair_count": pair_count,
        "component_pass_count": pair_count - len(failure_indices),
        "component_failure_count": len(failure_indices),
        "component_failure_indices": failure_indices,
        "component_failure_reasons": [decisions[index][1] for index in failure_indices],
        "component_failure_budget": failure_budget,
        "maximum_component_failure_fraction": float(maximum_component_failure_fraction),
        "pair_total_score_regression_indices": score_regressions,
        "pair_total_score_failure_budget": score_failure_budget,
        "maximum_pair_score_failure_fraction": float(maximum_pair_score_failure_fraction),
        "catastrophic_pair_total_score_regressions": catastrophic_score_regressions,
        "catastrophic_component_regressions": catastrophic_component_regressions,
        "before_mean_score": before_mean,
        "after_mean_score": after_mean,
        "aggregate_improvement_fraction": aggregate_improvement,
        "minimum_mean_improvement_fraction": float(minimum_mean_improvement_fraction),
    }


__all__ = [
    "METRIC_NAMES",
    "SeamStructureFeatures",
    "edge_registration_visual_suspect",
    "mean_seam_structure_metrics",
    "long_horizontal_structure_metrics",
    "long_horizontal_structure_nondegrading",
    "pair_edge_registration_metrics",
    "prepare_seam_structure",
    "seam_structure_metrics",
    "sequence_structure_decision",
    "structurally_non_degrading",
    "symmetric_seam_structure_metrics",
]
