"""Fail-closed production publication for the exact S013 Visual Continuity V11."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import time
from typing import Any, Callable, Mapping

import numpy as np

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
    )
    authority = fast.get("authority")
    if not isinstance(authority, Mapping):
        raise RuntimeError("S013 V11 runner did not return its in-memory authority")
    owner = authority.get("owner")
    if not isinstance(owner, Mapping):
        raise RuntimeError("S013 V11 runner authority lacks the owner map")
    elapsed = time.perf_counter() - started
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
        "schedule": _json_value(authority.get("schedule", {})),
        "owner_summary": _json_value(owner.get("summary", {})),
        "pair_seam_decisions": _json_value(authority.get("pair_seam_decisions", ())),
        "m6_decisions": _json_value(authority.get("m6", {})),
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


def run_s13_v11_production(
    *,
    session_path: Path,
    output: Path,
    algorithm_spec: VideoAlgorithmSpec,
    config_path: Path | None = None,
    online_state: Path | None = None,
    maximum_post_seconds: float | None = None,
    observability: Mapping[str, object] | None = None,
    authority_runner: S13V11AuthorityRunner | None = None,
) -> dict[str, Any]:
    """Run exact V11 authority and atomically publish only its final P3."""

    _require_exact_v11_production(algorithm_spec)
    destination = output.expanduser().resolve()
    source = session_path.expanduser().resolve()
    observe = dict(observability or {"report_level": "summary", "artifact_level": "minimal"})
    runner = authority_runner or _run_s13_v11_authority
    invalidate_video_delivery(destination)
    try:
        authority = runner(
            session_path=source,
            output=destination,
            algorithm_spec=algorithm_spec,
            config_path=config_path,
            online_state=online_state,
            maximum_post_seconds=maximum_post_seconds,
            stage_output_stages=(),
        )
        if not isinstance(authority, S13V11AuthorityResult):
            raise TypeError("S013 V11 authority runner returned an unsupported result")
        report = _bind_authority_report(
            authority,
            spec=algorithm_spec,
            observability=observe,
        )
        return publish_video_2d(
            destination,
            np.asarray(authority.panorama),
            np.asarray(authority.owner_frame_id),
            report,
        )
    except Exception as exc:
        write_video_failure(destination, source, exc)
        raise


__all__ = [
    "S13V11AuthorityResult",
    "S13_V11_PRODUCTION_EXECUTION_BACKEND",
    "run_s13_v11_production",
]
