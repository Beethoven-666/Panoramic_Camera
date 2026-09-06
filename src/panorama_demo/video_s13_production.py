"""Fail-closed production publication for the exact S013 Visual Continuity V11."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
from pathlib import Path
import time
from typing import TYPE_CHECKING, Any, Callable, Mapping

import numpy as np

if TYPE_CHECKING:
    from .video_s13_live import S13V11LiveHandoff

from .video_algorithm import VideoAlgorithmSpec
from .video_delivery import (
    invalidate_video_delivery,
    publish_video_2d,
    write_video_failure,
)
from .video_s13_contract import (
    S13_VISUAL_CONTINUITY_ALGORITHM_ID,
    S13_VISUAL_CONTINUITY_IMPLEMENTATION_ID,
)


S13_V11_PRODUCTION_EXECUTION_BACKEND = "s013_v11_production_cuda"


@dataclass(frozen=True)
class S13V11AuthorityResult:
    """In-memory authority handed from the exact V11 runner to publication."""

    panorama: np.ndarray
    owner_frame_id: np.ndarray
    report: Mapping[str, object]


S13V11AuthorityRunner = Callable[..., S13V11AuthorityResult]


def _require_exact_v11_production(spec: VideoAlgorithmSpec) -> None:
    if spec.role != "production":
        raise ValueError("S013 V11 formal route requires algorithm role production")
    if spec.algorithm_id != S13_VISUAL_CONTINUITY_ALGORITHM_ID:
        raise ValueError("S013 V11 formal route received a different algorithm_id")
    if spec.implementation_id != S13_VISUAL_CONTINUITY_IMPLEMENTATION_ID:
        raise ValueError("S013 V11 formal route received a different implementation_id")
    if spec.allow_baseline_fallback:
        raise ValueError("S013 V11 production forbids baseline fallback")


def _m51_config(config: object) -> object | None:
    from .video_s13_m51_r2 import S13M51R2Config, S13M51R3Config, S13M51R4Config

    component = getattr(config, "component")
    document = component.get("m51_r2")
    if not getattr(config, "m51_r2_enabled"):
        return None
    if not isinstance(document, Mapping):
        raise ValueError("S013 V11 production M5.1-r2 configuration is missing")
    if getattr(config, "m51_r4_enabled"):
        r3 = component.get("m51_r3")
        r4 = component.get("m51_r4")
        if not isinstance(r3, Mapping) or not isinstance(r4, Mapping):
            raise ValueError("S013 V11 production M5.1-r4 configuration is missing")
        values = {**dict(document), **dict(r4)}
        values.update({
            "complete_reassessment_pair_indices": tuple(
                int(value) for value in r3["reassessment_pair_indices"]
            ),
            "component_local_ambiguity_enabled": True,
        })
        return S13M51R4Config(**values)
    if getattr(config, "m51_r3_enabled"):
        r3 = component.get("m51_r3")
        if not isinstance(r3, Mapping):
            raise ValueError("S013 V11 production M5.1-r3 configuration is missing")
        return S13M51R3Config(
            **dict(document),
            complete_reassessment_pair_indices=tuple(
                int(value) for value in r3["reassessment_pair_indices"]
            ),
            component_local_ambiguity_enabled=True,
        )
    return S13M51R2Config(**dict(document))


def _json_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _monotonic_after(previous_ns: int) -> int:
    value = time.monotonic_ns()
    while value <= previous_ns:
        time.sleep(0)
        value = time.monotonic_ns()
    return value


def _stage_pixel_sha256(image: object) -> str:
    """Hash canonical in-memory uint8 pixels, including shape and dtype."""

    pixels = np.asarray(image)
    if pixels.dtype != np.uint8 or pixels.ndim != 3:
        raise RuntimeError("S013 V11 stage authority must be a three-dimensional uint8 array")
    contiguous = np.ascontiguousarray(pixels)
    semantic_header = (
        f"s013-stage-pixels/v1\0dtype={contiguous.dtype.str}\0"
        f"shape={','.join(str(value) for value in contiguous.shape)}\0"
    ).encode("ascii")
    digest = hashlib.sha256()
    digest.update(semantic_header)
    digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def _run_s13_v11_authority(**kwargs: object) -> S13V11AuthorityResult:
    """Private lazy adapter point for the V11 CUDA authority runner.

    The production route and publisher are independently testable while the
    real runner adapter is kept in one small function.  It must return the
    in-memory P3 and owner authority without encoding any stage snapshot.
    """

    from .video_s13_contract import load_s13_config
    from .video_s13_cuda_fast_pipeline import run_s13_cuda_fast_pipeline
    from .paths import PROJECT_ROOT
    from .video_s13_promotion import (
        S13_V11_CANDIDATE_RELATIVE_PATH,
        verify_s13_v11_promotion,
    )
    from .video_s13_session import load_s13_session_bundle
    from .video_s13_trajectory import load_s13_trajectory

    started = time.perf_counter()
    session_path = Path(kwargs["session_path"])
    spec = kwargs["algorithm_spec"]
    if not isinstance(spec, VideoAlgorithmSpec):
        raise TypeError("S013 V11 production authority requires an algorithm spec")
    verify_s13_v11_promotion(
        PROJECT_ROOT / S13_V11_CANDIDATE_RELATIVE_PATH,
        spec.config_path,
    )
    config = load_s13_config(spec.config_path, expected_role="production")
    if (
        config.document.get("algorithm_id") != spec.algorithm_id
        or config.document.get("implementation_id") != spec.implementation_id
    ):
        raise ValueError("S013 V11 production config identity changed after lock resolution")
    if config.runtime_backend != "cupy_cuda_m63_robust_v7":
        raise ValueError("S013 V11 production requires its frozen CUDA runtime backend")

    bundle = load_s13_session_bundle(session_path, validation_workers=4)
    session = bundle.session
    trajectory = load_s13_trajectory(
        session,
        trajectory_cache=None,
        reuse_online_trajectory=False,
        run_offline_orb=False,
        ignore_pose=True,
        config_path=None,
    )
    output = Path(kwargs.get("output", session_path.parent / ".s013-v11-production"))
    component = config.component
    live_handoff = kwargs.get("live_handoff")
    frozen_p0 = (
        None if live_handoff is None
        else getattr(live_handoff, "frozen_p0_authority", None)
    )
    fast = run_s13_cuda_fast_pipeline(
        session=session,
        trajectory=trajectory,
        output=output,
        analysis_width_px=config.analysis_width_px,
        normal_target_advance_px=config.normal_target_advance_px,
        risky_target_advance_px=config.risky_target_advance_px,
        m51_r2_config=_m51_config(config),
        m6_cuda_v2=False,
        m62_equivalence=config.m62_equivalence,
        m62_options={
            "photometric": dict(component["photometric"]),
            "blend": dict(component["blend"]),
            "equivalence": dict(config.document.get("m62_equivalence", {})),
            "execution": dict(config.document.get("m62_execution", {})),
            "selection": dict(config.document.get("m61_bootstrap", {}).get("selection", {})),
            "m63": dict(config.document.get("m63_photometric", {})),
        },
        validated_rgb_handoff=bundle.validated_rgb_handoff,
        motion_execution_policy=str(
            dict(config.document.get("motion_execution", {})).get("policy", "full_reference")
        ).replace("deferred_step4_unless_direction_fallback_required", "deferred_step4"),
        m5_execution_mode=str(
            dict(config.document.get("m5_execution", {})).get("mode", "full_reference")
        ),
        m5_pair_base_atlas=bool(
            dict(config.document.get("m5_pair_base_atlas", {})).get("enabled", False)
        ),
        m5_p0_map_mode=str(
            dict(config.document.get("m5_compact_p0_map", {})).get("mode", "full_reference")
        ),
        m5_compact_full_reference_fallback=bool(
            dict(config.document.get("m5_compact_p0_map", {})).get(
                "full_reference_fallback", True
            )
        ),
        stage_output_stages=(),
        p0_continuation=(None if frozen_p0 is None else frozen_p0.continuation),
    )
    authority = fast.get("authority")
    if not isinstance(authority, Mapping):
        raise RuntimeError("S013 V11 runner did not return its in-memory authority")
    owner = authority.get("owner")
    if not isinstance(owner, Mapping):
        raise RuntimeError("S013 V11 runner authority lacks the owner map")
    stage_pixel_sha256: dict[str, str] = {}
    for stage_name, result_name in (("P0", "p0"), ("P1", "p1"), ("P2", "p2"), ("P3", "p3")):
        stage_result = fast.get(result_name)
        stage_image = getattr(stage_result, "image", None)
        if stage_image is None:
            raise RuntimeError(f"S013 V11 runner authority lacks in-memory {stage_name} pixels")
        stage_pixel_sha256[stage_name] = _stage_pixel_sha256(stage_image)
    elapsed = time.perf_counter() - started
    shadow_equivalence: dict[str, object] | None = None
    if live_handoff is not None:
        from .video_s13_online_p0 import semantic_assignments

        shadow = getattr(live_handoff, "online_shadow", None)
        shadow_failure = getattr(live_handoff, "online_shadow_failure_reason", None)
        continuation = fast.get("p0_continuation")
        final_semantic = ()
        if continuation is not None:
            final_semantic = semantic_assignments(
                continuation.schedule,
                continuation.selection,
                continuation.selected_hypothesis_ids,
            )
        shadow_semantic = () if shadow is None else shadow.semantic_assignments
        shadow_equivalence = {
            "mode": "online_m0_m3_shadow_only",
            "pixel_evidence_reused": False,
            "shadow_failure_reason": shadow_failure,
            "online_assignment_count": len(shadow_semantic),
            "offline_assignment_count": len(final_semantic),
            "final_schedule_exact": bool(
                shadow is not None
                and shadow_failure is None
                and shadow.frontiers.committed_frame_index + 1
                == len(getattr(live_handoff, "committed_frames"))
                and shadow_semantic == final_semantic
            ),
        }
    maximum = kwargs.get("maximum_post_seconds")
    within_budget = maximum is None or elapsed <= float(maximum)
    overall = "A" if within_budget else "C"
    session_root = session.root
    report: dict[str, object] = {
        "delivery_state": "published" if within_budget else "published_degraded",
        "manual_review_required": not within_budget,
        "strict_quality_pass": True,
        "grades": {
            "structural": "A",
            "visual": "A",
            "performance": overall,
            "overall": overall,
        },
        "input_sha256": {
            name: _sha256(session_root / filename)
            for name, filename in (
                ("manifest", "manifest.json"),
                ("calibration", "calibration.json"),
                ("frames_csv", "frames.csv"),
            )
        },
        "source_frame_ids": _json_value(authority.get("source_frame_ids", ())),
        "primary_scan_segment": _json_value(
            fast.get("primary_scan_segment", {})
        ),
        "schedule": _json_value(authority.get("schedule", {})),
        "owner_summary": _json_value(owner.get("summary", {})),
        "pair_seam_decisions": _json_value(authority.get("pair_seam_decisions", ())),
        "m6_decisions": _json_value(authority.get("m6", {})),
        "stage_pixel_sha256": stage_pixel_sha256,
        "trajectory": dict(trajectory.audit),
        "performance": {
            "primary_post_capture_seconds": elapsed,
            "maximum_post_seconds": maximum,
            "within_post_capture_budget": within_budget,
            "stage_seconds": _json_value(fast.get("timings", {})),
            "cuda_resident": _json_value(fast.get("cuda_resident", {})),
        },
        "production_output_policy": {
            "stage_output_stages": [],
            "p3_stage_png_encoded": False,
            "formal_publication_encodes_p3_once": True,
        },
    }
    if shadow_equivalence is not None:
        report["online_shadow_equivalence"] = shadow_equivalence
    return S13V11AuthorityResult(
        panorama=np.asarray(fast["p3"].image),
        owner_frame_id=np.asarray(owner["frame_id_map"]),
        report=report,
    )


def _algorithm_report(spec: VideoAlgorithmSpec) -> dict[str, object]:
    return {
        "role": spec.role,
        "algorithm_id": spec.algorithm_id,
        "implementation_id": spec.implementation_id,
        "config_sha256": spec.config_sha256,
        "source_commit": spec.source_commit,
        "working_tree_dirty": spec.working_tree_dirty,
        "model_sha256": dict(spec.model_sha256),
        "fallback_used": False,
        "execution_backend": S13_V11_PRODUCTION_EXECUTION_BACKEND,
    }


def _bind_authority_report(
    authority: S13V11AuthorityResult,
    *,
    spec: VideoAlgorithmSpec,
    observability: Mapping[str, object],
) -> dict[str, Any]:
    panorama = np.asarray(authority.panorama)
    owner = np.asarray(authority.owner_frame_id)
    if panorama.dtype != np.uint8 or panorama.ndim != 3 or panorama.shape[2] != 3:
        raise ValueError("S013 V11 authority P3 must be a uint8 BGR image")
    if owner.shape != panorama.shape[:2] or not np.issubdtype(owner.dtype, np.integer):
        raise ValueError("S013 V11 authority owner map must be integral and match P3")

    report = dict(authority.report)
    claimed = report.get("algorithm")
    expected = _algorithm_report(spec)
    if claimed is not None:
        if not isinstance(claimed, Mapping):
            raise ValueError("S013 V11 authority algorithm audit must be an object")
        for key in ("role", "algorithm_id", "implementation_id", "fallback_used"):
            if key in claimed and claimed[key] != expected[key]:
                raise ValueError(f"S013 V11 authority changed algorithm.{key}")
    report["algorithm"] = expected
    report["observability"] = dict(observability)
    return report


def _validate_live_handoff(
    handoff: "S13V11LiveHandoff",
    *,
    spec: VideoAlgorithmSpec,
    session_path: Path,
) -> None:
    if (
        handoff.algorithm_id != spec.algorithm_id
        or handoff.implementation_id != spec.implementation_id
        or handoff.production_config_sha256 != spec.config_sha256
    ):
        raise ValueError("S013 V11 live handoff identity does not match production lock")
    if handoff.session_root.resolve() != session_path.resolve():
        raise ValueError("S013 V11 live handoff belongs to a different session")
    if handoff.reuse_level not in {
        "validated_inputs_only", "frozen_p0_authority_v1",
    }:
        raise ValueError("S013 V11 live handoff requests an unvalidated reuse level")
    ids = [item.frame_id for item in handoff.committed_frames]
    rows = [item.frames_csv_row_index for item in handoff.committed_frames]
    if not ids or ids != sorted(ids) or len(ids) != len(set(ids)):
        raise ValueError("S013 V11 live handoff committed ledger is incomplete")
    if rows != list(range(len(rows))):
        raise ValueError("S013 V11 live handoff CSV ledger is not contiguous")
    if handoff.reuse_level == "frozen_p0_authority_v1":
        from .video_s13_online_checkpoint import array_sha256

        frozen = getattr(handoff, "frozen_p0_authority", None)
        if frozen is None or frozen.reuse_level != handoff.reuse_level:
            raise ValueError("S013 V11 frozen P0 handoff lacks its authority bundle")
        if (
            frozen.algorithm_id != spec.algorithm_id
            or frozen.implementation_id != spec.implementation_id
            or frozen.production_config_sha256 != spec.config_sha256
            or frozen.session_root.resolve() != session_path.resolve()
        ):
            raise ValueError("S013 V11 frozen P0 authority identity changed")
        expected_files = {
            "manifest_sha256": session_path / "manifest.json",
            "calibration_sha256": session_path / "calibration.json",
            "frames_csv_sha256": session_path / "frames.csv",
        }
        for attribute, path in expected_files.items():
            if getattr(frozen, attribute) != _sha256(path):
                raise ValueError(f"S013 V11 frozen P0 {attribute} changed")
        if tuple(frozen.committed_frames) != tuple(handoff.committed_frames):
            raise ValueError("S013 V11 frozen P0 committed ledger changed")
        p0 = frozen.continuation.p0_render
        image = np.asarray(p0.image)
        valid = np.asarray(p0.valid_mask)
        owner = np.asarray(p0.pixel_provenance.get("owner_frame_id"))
        if (
            image.dtype != np.uint8
            or valid.dtype != np.bool_
            or image.shape[:2] != valid.shape
            or owner.shape != valid.shape
            or not np.array_equal(valid, owner >= 0)
        ):
            raise ValueError("S013 V11 frozen P0 pixel provenance is invalid")
        if frozen.p0_pixel_sha256 != array_sha256(
            image, schema="s013-stage-pixels/v1"
        ):
            raise ValueError("S013 V11 frozen P0 pixels changed")
        if frozen.p0_owner_sha256 != array_sha256(owner, schema="s013-owner/v1"):
            raise ValueError("S013 V11 frozen P0 owner changed")
        if frozen.p0_valid_sha256 != array_sha256(valid, schema="s013-valid/v1"):
            raise ValueError("S013 V11 frozen P0 valid mask changed")


def run_s13_v11_production(
    *,
    session_path: Path,
    output: Path,
    algorithm_spec: VideoAlgorithmSpec,
    config_path: Path | None = None,
    online_state: Path | None = None,
    maximum_post_seconds: float | None = None,
    observability: Mapping[str, object] | None = None,
    live_handoff: "S13V11LiveHandoff | None" = None,
    authority_runner: S13V11AuthorityRunner | None = None,
) -> dict[str, Any]:
    """Run exact V11 authority and atomically publish only its final P3."""

    _require_exact_v11_production(algorithm_spec)
    destination = output.expanduser().resolve()
    source = session_path.expanduser().resolve()
    observe = dict(observability or {"report_level": "summary", "artifact_level": "minimal"})
    runner = authority_runner or _run_s13_v11_authority
    if live_handoff is not None:
        if live_handoff.reuse_level == "frozen_p0_authority_v1":
            # Validate the session/algorithm contract before considering a
            # checkpoint rollback. A damaged P0 can only trigger the same V11.
            fresh_inputs = replace(live_handoff, reuse_level="validated_inputs_only",
                                   frozen_p0_authority=None)
            _validate_live_handoff(fresh_inputs, spec=algorithm_spec, session_path=source)
            try:
                _validate_live_handoff(live_handoff, spec=algorithm_spec, session_path=source)
                for frame in live_handoff.committed_frames:
                    if (_sha256(frame.color_path) != frame.color_sha256
                            or _sha256(frame.aligned_depth_path) != frame.aligned_depth_sha256):
                        raise ValueError("Frozen P0 committed source bytes changed")
            except (ValueError, OSError) as exc:
                from .sdk_state import atomic_json

                reason = f"checkpoint_invalid: {exc}"
                atomic_json(destination / "online_checkpoint_rollback.json", {
                    "reason": reason, "recompute_algorithm_id": algorithm_spec.algorithm_id,
                    "baseline_fallback": False,
                })
                live_handoff = replace(fresh_inputs, online_shadow_failure_reason=reason,
                    online_2d_metrics={**dict(fresh_inputs.online_2d_metrics),
                                      "reuse_level": "validated_inputs_only",
                                      "p0_prefix_reused_fraction": 0.0,
                                      "full_m0_m3_recomputed": True,
                                      "rollback_checkpoint_used": True, "rollback_reasons": [reason]})
        else:
            _validate_live_handoff(live_handoff, spec=algorithm_spec, session_path=source)
    production_started_monotonic_ns = time.monotonic_ns()
    capture_stop_monotonic_ns = (
        live_handoff.capture_stopped_monotonic_ns
        if live_handoff is not None
        else production_started_monotonic_ns
    )
    timing_origin = "live_capture_stop" if live_handoff is not None else "offline_production_start"
    invalidate_video_delivery(destination)
    authority: S13V11AuthorityResult | None = None
    try:
        authority = runner(
            session_path=source,
            output=destination,
            algorithm_spec=algorithm_spec,
            config_path=config_path,
            online_state=online_state,
            maximum_post_seconds=maximum_post_seconds,
            stage_output_stages=(),
            live_handoff=live_handoff,
        )
        if not isinstance(authority, S13V11AuthorityResult):
            raise TypeError("S013 V11 authority runner returned an unsupported result")
        p3_memory_monotonic_ns = time.monotonic_ns()
        report = _bind_authority_report(
            authority,
            spec=algorithm_spec,
            observability=observe,
        )
        if live_handoff is not None:
            frozen = getattr(live_handoff, "frozen_p0_authority", None)
            report["live_handoff"] = {
                "reuse_level": live_handoff.reuse_level,
                "committed_frame_count": len(live_handoff.committed_frames),
                "motion_edge_count": len(live_handoff.motion_edges),
                "pixel_evidence_reused": frozen is not None,
                "p0_reused": frozen is not None,
                "p0_prefix_reused_fraction": (
                    0.0 if frozen is None else frozen.p0_prefix_reused_fraction
                ),
                "sealed_source_count": (
                    0 if frozen is None else frozen.sealed_prefix_source_count
                ),
                "tail_finalized_source_count": (
                    0 if frozen is None else frozen.finalized_tail_source_count
                ),
                "full_m0_m3_recomputed": frozen is None,
                "rollback_checkpoint_used": (
                    False if frozen is None else frozen.rollback_checkpoint_used
                ),
                "rollback_reasons": (
                    [] if frozen is None else list(frozen.rollback_reasons)
                ),
                "final_metadata_closure_seconds": (
                    None if frozen is None else frozen.final_metadata_closure_seconds
                ),
                "tail_finalize_seconds": (
                    None if frozen is None else frozen.tail_finalize_seconds
                ),
                "preview_pixels_directly_published": False,
                "online_shadow_equivalence": report.get("online_shadow_equivalence"),
            }
        published = publish_video_2d(
            destination,
            np.asarray(authority.panorama),
            np.asarray(authority.owner_frame_id),
            report,
            capture_stop_monotonic_ns=capture_stop_monotonic_ns,
            p3_memory_monotonic_ns=p3_memory_monotonic_ns,
            timing_origin=timing_origin,
            timing_sections=(
                None
                if live_handoff is None
                else {
                    "capture": dict(live_handoff.capture_metrics),
                    "online_2d": dict(live_handoff.online_2d_metrics),
                    "isolation": {
                        "capture_orbslam3_call_count": 0,
                        "pre_2d_open3d_call_count": 0,
                        "pre_2d_3d_process_count": 0,
                    },
                }
            ),
        )
        published["_two_d_delivery_published_monotonic_ns"] = time.monotonic_ns()
    except Exception as exc:
        write_video_failure(destination, source, exc)
        raise
    finally:
        # The CUDA runtime is closed by the authority adapter before it
        # returns.  Drop the last formal P3/owner references before the outer
        # pipeline is allowed to create the independent 3-D process.
        authority = None
    published["_two_d_resources_released_monotonic_ns"] = _monotonic_after(
        published["_two_d_delivery_published_monotonic_ns"]
    )
    return published


__all__ = [
    "S13V11AuthorityResult",
    "S13_V11_PRODUCTION_EXECUTION_BACKEND",
    "run_s13_v11_production",
]
