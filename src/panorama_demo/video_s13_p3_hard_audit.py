"""Independent hard audit for S1.3 P3 photometric/blend assets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np

from .video_s13_bundle import sha256_file, verify_stage
from .video_s13_m6 import S13P3Result
from .video_s13_photometric import S13PhotometricConfig
from .video_s13_replay import S13VerifiedP2


P3_HARD_AUDIT_SCHEMA = "gemini305-video-s13-p3-hard-audit/v1"


def _same_float_with_nan(left: np.ndarray, right: np.ndarray) -> bool:
    return bool(np.array_equal(left, right, equal_nan=True))


def audit_s13_p3_stage(
    p2: S13VerifiedP2,
    p3: S13P3Result,
    *,
    photometric_config: S13PhotometricConfig = S13PhotometricConfig(),
    pending_asset_sha256: Mapping[str, str] | None = None,
) -> dict[str, object]:
    failures: list[str] = []
    primary_fields = (
        "owner_frame_id", "owner_source_index", "source_u", "source_v", "valid",
        "geometry_transaction_id", "seam_transaction_id",
    )
    primary_equal: dict[str, bool] = {}
    for name in primary_fields:
        before = np.asarray(p2.provenance[name])
        after = np.asarray(p3.pixel_provenance[name])
        equal = _same_float_with_nan(before, after) if before.dtype.kind == "f" else bool(np.array_equal(before, after))
        primary_equal[name] = equal
        if not equal:
            failures.append(f"primary_{name}_changed")
    valid_equal = bool(np.array_equal(p2.valid_mask, p3.valid_mask))
    if not valid_equal:
        failures.append("valid_mask_changed")
    if p3.visual_panorama.shape != p2.result_image.shape:
        failures.append("result_shape_changed")
    train = np.asarray(p3.photometric_training_mask, bool)
    heldout = np.asarray(p3.photometric_heldout_mask, bool)
    if train.shape != p2.valid_mask.shape or heldout.shape != p2.valid_mask.shape:
        failures.append("photometric_split_shape_invalid")
    train_heldout_overlap = int(np.count_nonzero(train & heldout)) if train.shape == heldout.shape else -1
    if train_heldout_overlap:
        failures.append("photometric_train_heldout_overlap")
    gains = np.asarray([item.gain_bgr for item in p3.photometric_solution.source_parameters], np.float64)
    biases = np.asarray([item.bias_bgr for item in p3.photometric_solution.source_parameters], np.float64)
    parameters_finite = bool(np.isfinite(gains).all() and np.isfinite(biases).all())
    parameters_in_bounds = bool(
        parameters_finite
        and np.all(gains >= photometric_config.minimum_gain)
        and np.all(gains <= photometric_config.maximum_gain)
        and np.all(np.abs(biases) <= photometric_config.maximum_absolute_bias_linear)
    )
    if not parameters_in_bounds:
        failures.append("photometric_parameters_invalid")
    weight = np.asarray(p3.pixel_provenance["secondary_weight"], np.float32)
    protected = np.asarray(p3.protected_structure_mask, bool)
    protected_blend_pixels = int(np.count_nonzero(protected & (weight > 0.0)))
    if protected_blend_pixels:
        failures.append("protected_structure_blended")
    if not np.isfinite(weight).all() or np.any(weight < 0.0) or np.any(weight > 0.499):
        failures.append("secondary_weight_invalid")
    secondary_frame = np.asarray(p3.pixel_provenance["secondary_frame_id"], np.int32)
    secondary_source = np.asarray(p3.pixel_provenance["secondary_source_index"], np.int32)
    secondary_u = np.asarray(p3.pixel_provenance["secondary_source_u"], np.float32)
    secondary_v = np.asarray(p3.pixel_provenance["secondary_source_v"], np.float32)
    active = weight > 0.0
    owner_only = ~active
    owner_only_sentinels = bool(
        np.all(secondary_frame[owner_only] == -1)
        and np.all(secondary_source[owner_only] == -1)
        and np.all(weight[owner_only] == 0.0)
    )
    if not owner_only_sentinels:
        failures.append("owner_only_secondary_sentinels_invalid")
    if np.any(active & (secondary_frame < 0)) or np.any(active & (secondary_source < 0)):
        failures.append("secondary_source_not_real")
    if np.any(active & (~np.isfinite(secondary_u) | ~np.isfinite(secondary_v))):
        failures.append("secondary_uv_nonfinite")
    owner_source = np.asarray(p3.pixel_provenance["owner_source_index"], np.int32)
    if np.any(active & (np.abs(owner_source - secondary_source) != 1)):
        failures.append("secondary_source_not_adjacent")
    replay_covered = np.zeros_like(active)
    replay_uv_legal = True
    for pair, plan in zip(p2.replay_pairs, p3.blend_plans, strict=True):
        roi = np.s_[:, pair.corridor_x0:pair.corridor_x1]
        local = plan.secondary_weight > 0.0
        if np.any(replay_covered[roi] & local):
            failures.append("blend_corridors_overlap")
        replay_covered[roi] |= local
        expected_valid = np.where(pair.primary_owner_right_mask, pair.left_valid, pair.right_valid)
        if np.any(local & ~expected_valid):
            replay_uv_legal = False
    if not np.array_equal(replay_covered, active):
        failures.append("blend_provenance_outside_replay")
    if not replay_uv_legal:
        failures.append("secondary_uv_not_replay_valid")
    performance = p3.performance
    forbidden_invocations = {
        key: int(performance.get(key, -1))
        for key in (
            "m4_reestimated_in_m6", "m5_reestimated_in_m6",
            "geometry_reestimated_in_m6", "seam_reestimated_in_m6",
            "trajectory_estimation_invocations", "open3d_invocations", "depth_invocations",
            "tsdf_invocations",
        )
    }
    if any(forbidden_invocations.values()):
        failures.append("forbidden_m6_estimation_or_backend_invocation")
    immutable_unchanged = all(
        (p2.root.parent / name).is_file()
        and sha256_file(p2.root.parent / name) == expected
        for name, expected in p2.immutable_sha256.items()
    )
    if not immutable_unchanged:
        failures.append("sealed_p2_or_ancestor_changed")
    source_count = int(p2.completion["source_count"])
    if len(p3.photometric_solution.source_parameters) != source_count:
        failures.append("photometric_source_count_changed")
    maximum_contributors = 2 if np.any(active) else 1
    return {
        "schema": P3_HARD_AUDIT_SCHEMA,
        "passed": not failures,
        "hard_audit_passed": not failures,
        "failures": failures,
        "parent_stage": "P2",
        "parent_completion_sha256": p2.completion_sha256,
        "p2_immutable_sha256": dict(p2.immutable_sha256),
        "p2_immutable_unchanged": immutable_unchanged,
        "primary_provenance_equal": primary_equal,
        "valid_mask_equal": valid_equal,
        "new_hole_count": int(np.count_nonzero(p2.valid_mask & ~p3.valid_mask)),
        "photometric_parameters_finite": parameters_finite,
        "photometric_parameters_in_versioned_bounds": parameters_in_bounds,
        "photometric_bounds_schema": p3.photometric_solution.bounds_schema,
        "train_heldout_overlap_pixel_count": train_heldout_overlap,
        "protected_blend_pixel_count": protected_blend_pixels,
        "blend_pixel_count": int(np.count_nonzero(active)),
        "maximum_real_contributors_per_pixel": maximum_contributors,
        "secondary_sources_real_adjacent_and_uv_legal": (
            "secondary_source_not_real" not in failures
            and "secondary_source_not_adjacent" not in failures
            and "secondary_uv_nonfinite" not in failures
            and replay_uv_legal
        ),
        "owner_only_secondary_sentinels": owner_only_sentinels,
        "uses_depth": False,
        "uses_open3d": False,
        "uses_tsdf": False,
        "uses_virtual_rgb": False,
        "uses_color_hole_fill": False,
        "forbidden_invocations": forbidden_invocations,
        "pending_assets_sha256": dict(pending_asset_sha256 or {}),
    }


def verify_sealed_s13_p3(
    p3_root: Path,
    *,
    completion_schema: str,
) -> Mapping[str, object]:
    completion = verify_stage(
        p3_root, completion_name="P3_completion.json", schema=completion_schema
    )
    if completion.get("stage") != "P3" or completion.get("parent_stage") != "P2":
        raise ValueError("S1.3 P3 completion parent chain is invalid")
    if completion.get("hard_audit_passed") is not True:
        raise ValueError("S1.3 P3 completion did not pass hard audit")
    try:
        audit = json.loads((p3_root / "hard_audit.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("S1.3 P3 hard audit is missing or invalid") from exc
    if audit.get("schema") != P3_HARD_AUDIT_SCHEMA or audit.get("passed") is not True:
        raise ValueError("S1.3 P3 hard audit did not pass")
    if completion.get("hard_audit_sha256") != sha256_file(p3_root / "hard_audit.json"):
        raise ValueError("S1.3 P3 hard audit SHA binding disagrees")
    return completion


__all__ = [
    "P3_HARD_AUDIT_SCHEMA", "audit_s13_p3_stage", "verify_sealed_s13_p3",
]
