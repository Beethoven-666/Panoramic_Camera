"""Structural-exact, RGB-tolerant comparison for the isolated S013 v3 path.

This is deliberately an in-memory comparator: benchmark reports may print its
result, but a candidate run never emits comparison assets beside its four PNGs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import cv2
import numpy as np


@dataclass(frozen=True)
class S13RgbTolerance:
    maximum_abs_dn: int
    p999_abs_dn: float
    minimum_psnr_db: float
    minimum_ssim: float


@dataclass(frozen=True)
class S13StructuralEquivalencePolicy:
    stage_tolerances: Mapping[str, S13RgbTolerance]
    exact_structure_fields: tuple[str, ...]


@dataclass(frozen=True)
class S13StageComparison:
    stage_name: str
    structure_equal: bool
    differing_pixels: int
    differing_channels: int
    maximum_abs_dn: int
    p999_abs_dn: float
    mean_abs_dn: float
    psnr_db: float
    ssim: float
    failures: tuple[str, ...]


@dataclass(frozen=True)
class S13StructuralComparison:
    passed: bool
    stages: tuple[S13StageComparison, ...]
    decision_changes: Mapping[str, object]
    structure_failures: tuple[str, ...]


def default_s13_structural_policy() -> S13StructuralEquivalencePolicy:
    p012 = S13RgbTolerance(1, 1.0, 55.0, 0.9998)
    return S13StructuralEquivalencePolicy(
        {"P0": p012, "P1": p012, "P2": p012,
         "P3": S13RgbTolerance(2, 1.0, 50.0, 0.9995)},
        ("schedule", "canvas_shape", "source_assignments", "vertical_solution",
         "vertical_pair_decisions", "valid_mask", "owner_source_index", "source_u",
         "source_v", "geometry_selection", "seam_path", "component_chain",
         "c2e_selection", "photometric_model", "blend_plan"),
    )


def _gray_ssim(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Fixed global grayscale SSIM; deterministic and dependency-free beyond OpenCV."""
    left = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY).astype(np.float64)
    right = cv2.cvtColor(candidate, cv2.COLOR_BGR2GRAY).astype(np.float64)
    mu_l, mu_r = float(left.mean()), float(right.mean())
    var_l, var_r = float(left.var()), float(right.var())
    covariance = float(((left - mu_l) * (right - mu_r)).mean())
    c1, c2 = (0.01 * 255.0) ** 2, (0.03 * 255.0) ** 2
    return float(((2 * mu_l * mu_r + c1) * (2 * covariance + c2)) /
                 ((mu_l * mu_l + mu_r * mu_r + c1) * (var_l + var_r + c2)))


def _value_equal(left: object, right: object) -> bool:
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return isinstance(left, np.ndarray) and isinstance(right, np.ndarray) and np.array_equal(left, right)
    return left == right


def compare_s13_structural_equivalence(
    reference: Mapping[str, Mapping[str, object]], candidate: Mapping[str, Mapping[str, object]],
    *, policy: S13StructuralEquivalencePolicy | None = None,
) -> S13StructuralComparison:
    """Compare P0--P3 images and their decision authority without serializing data."""
    policy = policy or default_s13_structural_policy()
    stages: list[S13StageComparison] = []
    structure_failures: list[str] = []
    decision_changes: dict[str, object] = {}
    for stage, tolerance in policy.stage_tolerances.items():
        expected, actual = reference.get(stage), candidate.get(stage)
        if expected is None or actual is None:
            structure_failures.append(f"{stage}:missing_stage")
            continue
        image, other = expected.get("image"), actual.get("image")
        if not isinstance(image, np.ndarray) or not isinstance(other, np.ndarray) or image.shape != other.shape or image.dtype != np.uint8 or other.dtype != np.uint8:
            structure_failures.append(f"{stage}:image_shape_or_type")
            continue
        field_failures = []
        for field in policy.exact_structure_fields:
            if field in expected or field in actual:
                if not _value_equal(expected.get(field), actual.get(field)):
                    field_failures.append(field)
                    decision_changes[f"{stage}.{field}"] = "different"
        delta = np.abs(image.astype(np.int16) - other.astype(np.int16))
        maximum = int(delta.max())
        p999 = float(np.percentile(delta, 99.9))
        mean = float(delta.mean())
        mse = float(np.mean(delta.astype(np.float64) ** 2))
        psnr = float("inf") if mse == 0 else float(20.0 * np.log10(255.0 / np.sqrt(mse)))
        ssim = _gray_ssim(image, other)
        failures = list(field_failures)
        if maximum > tolerance.maximum_abs_dn:
            failures.append("maximum_abs_dn")
        if p999 > tolerance.p999_abs_dn:
            failures.append("p999_abs_dn")
        if psnr < tolerance.minimum_psnr_db:
            failures.append("psnr")
        if ssim < tolerance.minimum_ssim:
            failures.append("ssim")
        stages.append(S13StageComparison(stage, not field_failures,
            int(np.count_nonzero(np.any(delta != 0, axis=2))), int(np.count_nonzero(delta)),
            maximum, p999, mean, psnr, ssim, tuple(failures)))
        structure_failures.extend(f"{stage}:{field}" for field in field_failures)
    return S13StructuralComparison(
        passed=not structure_failures and len(stages) == len(policy.stage_tolerances)
        and all(not row.failures for row in stages), stages=tuple(stages),
        decision_changes=decision_changes, structure_failures=tuple(structure_failures),
    )


__all__ = ["S13RgbTolerance", "S13StructuralEquivalencePolicy", "S13StageComparison",
           "S13StructuralComparison", "compare_s13_structural_equivalence",
           "default_s13_structural_policy"]
