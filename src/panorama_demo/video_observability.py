"""Algorithm-independent video reporting and artifact policy.

The functions in this module consume a completed two-dimensional delivery.
They intentionally have no renderer imports: evidence generation must not be
able to choose sources, alter poses, or feed a measurement back into a seam
decision.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Literal, Mapping

import cv2
import numpy as np


ReportLevel = Literal["summary", "full"]
ArtifactLevel = Literal["minimal", "provenance", "audit"]


_PRIMARY_DELIVERY_NAMES = (
    "video_panorama.jpg",
    "video_panorama.png",
    "video_pixel_provenance.npz",
    "video_report.json",
    "video_delivery.json",
)
_OBSERVABILITY_NAMES = (
    "owner_map_color.png",
    "owner_boundary_overlay.png",
    "owner_component_report.json",
    # Written only by the standalone read-only visual evaluator.  It is an
    # evidence sidecar, so a new primary render must not inherit its result.
    "video_visual_evaluation.json",
    # Candidate-only calibrated inverse-map annotation evidence; it is never
    # primary RGB/provenance and must not survive a new render.
    "candidate_annotation_projection.json",
    "candidate_annotation_projection_masks.npz",
    # Read-only candidate traces written only for full/audit observability.
    # They are derived from the already-published report and must never be an
    # input to a renderer, algorithm selector, or delivery decision.
    "candidate_pair_audits.json",
    "candidate_algorithm_trace.json",
    "audit_manifest.json",
)
_OWNER_COMPONENT_REPORT_SCHEMA = "gemini305-video-owner-components/v1"
_AUDIT_MANIFEST_SCHEMA = "gemini305-video-audit-manifest/v1"
_CANDIDATE_PAIR_AUDIT_SCHEMA = "gemini305-video-candidate-pair-audits/v1"
_CANDIDATE_ALGORITHM_TRACE_SCHEMA = "gemini305-video-candidate-algorithm-trace/v1"


@dataclass(frozen=True)
class ObservabilitySpec:
    """Controls evidence output only; it must never choose an algorithm."""

    report_level: ReportLevel = "summary"
    artifact_level: ArtifactLevel = "minimal"

    def __post_init__(self) -> None:
        if self.report_level not in ("summary", "full"):
            raise ValueError("report_level must be summary or full")
        if self.artifact_level not in ("minimal", "provenance", "audit"):
            raise ValueError("artifact_level must be minimal, provenance, or audit")
        if self.artifact_level == "audit" and self.report_level != "full":
            raise ValueError("artifact_level=audit requires report_level=full")

    @classmethod
    def from_values(
        cls, *, report_level: str = "summary", artifact_level: str = "minimal"
    ) -> "ObservabilitySpec":
        return cls(report_level=report_level, artifact_level=artifact_level)  # type: ignore[arg-type]

    def as_dict(self) -> dict[str, str]:
        return {
            "report_level": self.report_level,
            "artifact_level": self.artifact_level,
        }


def clear_observability_artifacts(output: Path) -> None:
    """Remove sidecars from a prior run without touching a primary delivery.

    ``video_delivery.invalidate_video_delivery`` owns the primary delivery and
    renderer archives.  Keeping this narrow makes the ownership boundary
    explicit and prevents an observability cleanup from revoking a published
    panorama.
    """

    for name in _OBSERVABILITY_NAMES:
        (output / name).unlink(missing_ok=True)


def _require_owner(owner: np.ndarray) -> np.ndarray:
    if not isinstance(owner, np.ndarray) or owner.ndim != 2 or owner.size == 0:
        raise ValueError("Owner map must be a non-empty two-dimensional array")
    if not np.issubdtype(owner.dtype, np.integer):
        raise ValueError("Owner map must use an integer dtype")
    return owner.astype(np.int32, copy=False)


def _require_panorama(panorama_bgr: np.ndarray, owner: np.ndarray) -> np.ndarray:
    if (
        not isinstance(panorama_bgr, np.ndarray)
        or panorama_bgr.dtype != np.uint8
        or panorama_bgr.ndim != 3
        or panorama_bgr.shape[2] != 3
        or panorama_bgr.shape[:2] != owner.shape
    ):
        raise ValueError("Panorama must be an 8-bit BGR image aligned with the owner map")
    return panorama_bgr


def _array_sha256(value: np.ndarray) -> str:
    """Return a hash of an array's layout-independent, decoded contents."""

    contiguous = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(repr(tuple(int(size) for size in contiguous.shape)).encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def colorize_owner_map(owner: np.ndarray) -> np.ndarray:
    """Return a deterministic BGR visualization of a provenance owner map.

    Negative IDs are deliberately black (unowned); every non-negative source
    receives a stable, non-black HSV-derived colour.  This is a visualization
    only and never serves as an alpha or validity mask.
    """

    owner_i32 = _require_owner(owner)
    valid = owner_i32 >= 0
    hsv = np.zeros((*owner_i32.shape, 3), dtype=np.uint8)
    if np.any(valid):
        ids = owner_i32[valid].astype(np.int64, copy=False)
        hsv_values = hsv[valid]
        hsv_values[:, 0] = np.mod(ids * 47 + 29, 180).astype(np.uint8)
        hsv_values[:, 1] = (192 + np.mod(ids * 17 + 13, 64)).astype(np.uint8)
        hsv_values[:, 2] = (192 + np.mod(ids * 31 + 7, 64)).astype(np.uint8)
        hsv[valid] = hsv_values
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def owner_boundaries(owner: np.ndarray) -> np.ndarray:
    """Return a two-sided 4-connected owner-boundary mask."""

    owner_i32 = _require_owner(owner)
    valid = owner_i32 >= 0
    boundary = np.zeros(owner_i32.shape, dtype=bool)
    horizontal = (
        valid[:, :-1]
        & valid[:, 1:]
        & (owner_i32[:, :-1] != owner_i32[:, 1:])
    )
    vertical = (
        valid[:-1, :]
        & valid[1:, :]
        & (owner_i32[:-1, :] != owner_i32[1:, :])
    )
    boundary[:, :-1] |= horizontal
    boundary[:, 1:] |= horizontal
    boundary[:-1, :] |= vertical
    boundary[1:, :] |= vertical
    return boundary


def owner_boundary_overlay(panorama_bgr: np.ndarray, owner: np.ndarray) -> np.ndarray:
    """Overlay only hard-owner transitions on an already-rendered panorama."""

    owner_i32 = _require_owner(owner)
    panorama = _require_panorama(panorama_bgr, owner_i32)
    overlay = panorama.copy()
    # Magenta is visible over both common scene tones and owner-map colours.
    overlay[owner_boundaries(owner_i32)] = (255, 0, 255)
    return overlay


def owner_component_report(owner: np.ndarray) -> dict[str, object]:
    """Describe all 4-connected per-owner components without changing them."""

    owner_i32 = _require_owner(owner)
    valid = owner_i32 >= 0
    owners: list[dict[str, object]] = []
    total_components = 0
    for frame_id in np.unique(owner_i32[valid]):
        mask = (owner_i32 == frame_id).astype(np.uint8)
        count, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=4)
        components: list[dict[str, object]] = []
        for label in range(1, count):
            x, y, width, height, area = stats[label].tolist()
            centroid_x, centroid_y = centroids[label].tolist()
            components.append(
                {
                    "component_id": label - 1,
                    "pixel_count": int(area),
                    "bounding_box": {
                        "x": int(x),
                        "y": int(y),
                        "width": int(width),
                        "height": int(height),
                    },
                    "centroid": {"x": float(centroid_x), "y": float(centroid_y)},
                }
            )
        total_components += len(components)
        owners.append(
            {
                "frame_id": int(frame_id),
                "pixel_count": int(np.count_nonzero(mask)),
                "component_count": len(components),
                "components": components,
            }
        )
    return {
        "schema": _OWNER_COMPONENT_REPORT_SCHEMA,
        "owner_map": {
            "shape": [int(owner_i32.shape[0]), int(owner_i32.shape[1])],
            "dtype": str(owner_i32.dtype),
            "raw_sha256": _array_sha256(owner_i32),
            "valid_pixel_count": int(np.count_nonzero(valid)),
            "unowned_pixel_count": int(np.count_nonzero(~valid)),
            "owner_boundary_pixel_count": int(np.count_nonzero(owner_boundaries(owner_i32))),
        },
        "owner_count": len(owners),
        "component_count": total_components,
        "owners": owners,
    }


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(f".{path.name}.pending")
    try:
        pending.write_bytes(payload)
        os.replace(pending, path)
    finally:
        pending.unlink(missing_ok=True)


def _atomic_write_png(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise OSError(f"Could not encode observability PNG: {path.name}")
    _atomic_write_bytes(path, encoded.tobytes())


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    _atomic_write_bytes(path, encoded)


def _load_primary_delivery(output: Path) -> tuple[np.ndarray, np.ndarray]:
    panorama_path = output / "video_panorama.png"
    provenance_path = output / "video_pixel_provenance.npz"
    panorama = cv2.imread(str(panorama_path), cv2.IMREAD_COLOR)
    if panorama is None:
        raise FileNotFoundError(f"Published video panorama is unavailable: {panorama_path}")
    try:
        with np.load(provenance_path, allow_pickle=False) as loaded:
            if "owner_frame_id" not in loaded:
                raise ValueError("Video provenance lacks owner_frame_id")
            owner = np.asarray(loaded["owner_frame_id"])
    except (OSError, ValueError) as exc:
        raise ValueError(f"Invalid video owner provenance: {provenance_path}") from exc
    owner_i32 = _require_owner(owner)
    return _require_panorama(panorama, owner_i32), owner_i32


def _artifact_record(path: Path) -> dict[str, object]:
    return {"path": path.name, "bytes": path.stat().st_size, "sha256": _file_sha256(path)}


def _primary_records(output: Path) -> dict[str, dict[str, object]]:
    missing = [name for name in _PRIMARY_DELIVERY_NAMES if not (output / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Published primary video delivery is incomplete: {', '.join(missing)}")
    return {name: _artifact_record(output / name) for name in _PRIMARY_DELIVERY_NAMES}


def _load_report_for_audit(output: Path) -> dict[str, object]:
    """Load the published report as evidence, never as a rendering input."""

    path = output / "video_report.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Published video report is not valid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError("Published video report must contain a JSON object")
    return value


def _mapping(value: object) -> Mapping[str, object] | None:
    return value if isinstance(value, Mapping) else None


def _audit_pair_groups(renderer: Mapping[str, object]) -> list[dict[str, object]]:
    """Extract the renderer's existing pair evidence without reinterpreting it.

    Candidate implementations have accumulated several deliberately separate
    per-pair audit lists (C1 owner, mesh, RAFT, object locking, MultiBand and
    local multi-label windows).  The trace preserves their source path and
    record ordering so offline tooling can identify the exact producer rather
    than guessing a common, lossy pair schema.  The only normalisation is the
    enclosing group record; individual evidence records remain byte-for-byte
    JSON values from ``video_report.json``.
    """

    groups: list[dict[str, object]] = []

    def add(path: str, value: object) -> None:
        if not isinstance(value, list):
            return
        # An empty list is meaningful in the primary report, but is not an
        # exported pair-evidence artifact: it proves no pair records exist.
        if not value:
            return
        groups.append(
            {
                "report_path": path,
                "record_count": len(value),
                "records": value,
            }
        )

    quality = _mapping(renderer.get("quality_metrics"))
    if quality is not None:
        for key in sorted(quality):
            # The C-family writes explicit ``*_audits`` arrays.  Requiring
            # both terms avoids accidentally exporting unrelated source-frame
            # or preview lists as if they were pair evidence.
            if "pair" in key or key.endswith("_audits"):
                add(f"renderer.quality_metrics.{key}", quality[key])
    add("renderer.video_photometric_flow_evidence", renderer.get("video_photometric_flow_evidence"))
    photometric = _mapping(renderer.get("video_global_photometric"))
    if photometric is not None:
        add("renderer.video_global_photometric.pairs", photometric.get("pairs"))
    return groups


def _candidate_trace_payload(
    report: Mapping[str, object], renderer: Mapping[str, object], *, report_record: Mapping[str, object]
) -> dict[str, object] | None:
    """Make a small, structured renderer trace from an immutable report.

    This intentionally excludes detailed pair lists; those reside in the
    dedicated pair-audit artifact.  It also does not calculate a quality grade
    or mutate the report.  The resulting data is solely a post-publication
    audit index for a candidate execution.
    """

    algorithm = _mapping(report.get("algorithm"))
    if algorithm is None or algorithm.get("role") != "candidate":
        return None

    pair_groups = _audit_pair_groups(renderer)
    # A renderer object with no pair records has no candidate seam/mesh audit
    # to export.  Retain a trace only when the report actually contains
    # renderer/pair evidence, rather than manufacturing a success record.
    if not pair_groups:
        return None

    quality = _mapping(renderer.get("quality_metrics")) or {}
    quality_summary = {
        key: value
        for key, value in quality.items()
        if not isinstance(value, (list, dict))
    }
    global_photometric = _mapping(renderer.get("video_global_photometric"))
    photometric_summary = (
        {
            key: value
            for key, value in global_photometric.items()
            if key != "pairs"
        }
        if global_photometric is not None
        else None
    )
    renderer_summary = {
        key: value
        for key, value in renderer.items()
        if key not in {"quality_metrics", "video_photometric_flow_evidence", "video_global_photometric"}
    }
    renderer_summary["quality_metrics"] = quality_summary
    if photometric_summary is not None:
        renderer_summary["video_global_photometric"] = photometric_summary

    return {
        "schema": _CANDIDATE_ALGORITHM_TRACE_SCHEMA,
        "source_report": dict(report_record),
        "algorithm": dict(algorithm),
        "report_schema": report.get("schema"),
        "source_frame_ids": report.get("source_frame_ids", []),
        "renderer": renderer_summary,
        "pair_audit": {
            "schema": _CANDIDATE_PAIR_AUDIT_SCHEMA,
            "path": "candidate_pair_audits.json",
            "group_count": len(pair_groups),
            "record_count": sum(int(group["record_count"]) for group in pair_groups),
        },
    }


def _write_candidate_audit_artifacts(output: Path) -> dict[str, dict[str, object]]:
    """Write candidate renderer traces for full/audit evidence only.

    The export is deliberately a one-way projection from ``video_report``.
    No caller receives data that could affect RGB, ownership, pose selection,
    candidate selection, or the published delivery record.
    """

    report = _load_report_for_audit(output)
    renderer = _mapping(report.get("renderer"))
    if renderer is None:
        return {}
    report_record = _artifact_record(output / "video_report.json")
    trace = _candidate_trace_payload(report, renderer, report_record=report_record)
    if trace is None:
        return {}
    pair_groups = _audit_pair_groups(renderer)
    pair_payload: dict[str, object] = {
        "schema": _CANDIDATE_PAIR_AUDIT_SCHEMA,
        "source_report": dict(report_record),
        "algorithm": trace["algorithm"],
        "renderer_backend": renderer.get("backend"),
        "group_count": len(pair_groups),
        "record_count": sum(int(group["record_count"]) for group in pair_groups),
        "groups": pair_groups,
    }
    pair_path = output / "candidate_pair_audits.json"
    trace_path = output / "candidate_algorithm_trace.json"
    _atomic_write_json(pair_path, pair_payload)
    _atomic_write_json(trace_path, trace)
    return {
        pair_path.name: _artifact_record(pair_path),
        trace_path.name: _artifact_record(trace_path),
    }


def write_observability_artifacts(output: Path, spec: ObservabilitySpec) -> dict[str, object]:
    """Write requested read-only evidence after primary delivery publication.

    The primary file digests are verified before and after export.  Therefore
    a sidecar implementation error cannot silently produce a different
    panorama, owner map, report, or delivery marker.
    """

    if spec.artifact_level == "minimal":
        return {"artifact_level": "minimal", "published": False}

    primary_before = _primary_records(output)
    panorama, owner = _load_primary_delivery(output)
    color_path = output / "owner_map_color.png"
    boundary_path = output / "owner_boundary_overlay.png"
    component_path = output / "owner_component_report.json"
    _atomic_write_png(color_path, colorize_owner_map(owner))
    _atomic_write_png(boundary_path, owner_boundary_overlay(panorama, owner))
    _atomic_write_json(component_path, owner_component_report(owner))
    audit_artifacts: dict[str, dict[str, object]] = {}
    if spec.artifact_level == "audit":
        audit_artifacts = _write_candidate_audit_artifacts(output)
    primary_after = _primary_records(output)
    if primary_after != primary_before:
        raise RuntimeError("Observability export modified a primary video delivery artifact")
    exported: dict[str, object] = {
        "artifact_level": spec.artifact_level,
        "published": True,
        "primary_artifacts": primary_after,
        "provenance_artifacts": {
            path.name: _artifact_record(path)
            for path in (color_path, boundary_path, component_path)
        },
    }
    if spec.artifact_level == "audit":
        exported["audit_artifacts"] = audit_artifacts
    return exported


def write_audit_manifest(
    output: Path,
    spec: ObservabilitySpec,
    export: dict[str, object],
    *,
    error: Exception | None = None,
) -> dict[str, object]:
    """Atomically record the audit outcome after all requested sidecars."""

    if spec.artifact_level != "audit":
        raise ValueError("Audit manifest is valid only for artifact_level=audit")
    payload: dict[str, object] = {
        "schema": _AUDIT_MANIFEST_SCHEMA,
        "status": "failed" if error is not None else "published",
        "observability": spec.as_dict(),
        "primary_artifacts": _primary_records(output),
    }
    if error is None:
        payload["provenance_artifacts"] = export.get("provenance_artifacts", {})
        payload["audit_artifacts"] = export.get("audit_artifacts", {})
        payload["audit_archives"] = {
            name: (output / name).is_dir()
            for name in ("central_strips", "central_strips_owner_only")
        }
    else:
        payload["error"] = {"type": type(error).__name__, "message": str(error)}
    _atomic_write_json(output / "audit_manifest.json", payload)
    return payload
