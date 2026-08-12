"""Single-attempt S013 M6.1 P3 runner.

Candidate selection is completed from sealed P2 evidence before the formal
renderer is entered.  The renderer then remaps each real source exactly once;
post-render measurements are reporting-only and cannot change the selection.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import os
import shutil
import uuid
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from .video_s13_m61_blend import (
    M61BlendCandidate,
    compose_and_canonicalize_m61_blend,
    plan_m61_blends,
)
from .video_s13_m61_config import (
    M61_BLEND_MANIFEST_SCHEMA,
    M61_PERFORMANCE_SCHEMA,
    M61_P2_COMPLETION_SCHEMA,
    M61_P3_COMPLETION_SCHEMA,
    S13M61EffectiveConfig,
    load_s13_m61_effective_config,
)
from .video_s13_m61_hard_audit import (
    P2_PARENT_REFERENCE_SCHEMA,
    audit_m61_p3_files,
    verify_promoted_m61_p3,
)
from .video_s13_m61_photometric import (
    S13M61PairEvidence,
    load_sealed_graph_topology,
    solve_and_select_s13_m61_photometric,
)
from .video_s13_m61_streaming import M61ROIStreamer
from .video_s13_bundle import seal_stage, update_current_latest, verify_stage


RUN_SCHEMA = "gemini305-video-s13-m61-run/v1"
SEAL_SCHEMA = "gemini305-video-s13-p3-seal-manifest/v2"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.pending")
    temporary.write_text(json.dumps(dict(value), sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _restore_pointer(path: Path, previous: bytes | None) -> None:
    """Restore the exact pre-run pointer after a failed final publication."""
    if previous is None:
        path.unlink(missing_ok=True)
        return
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex[:12]}.rollback")
    try:
        temporary.write_bytes(previous)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_session_rgb(
    session: Path,
    raw_rgb_paths: Mapping[int, str | Path] | None = None,
) -> tuple[dict[int, Path], dict[int, str]]:
    paths: dict[int, Path] = {}
    hashes: dict[int, str] = {}
    if raw_rgb_paths is not None:
        if not raw_rgb_paths:
            raise ValueError("raw RGB path mapping is empty")
        for raw_frame_id, raw_path in raw_rgb_paths.items():
            if isinstance(raw_frame_id, bool):
                raise ValueError("raw RGB frame ID is invalid")
            frame_id = int(raw_frame_id)
            if frame_id < 0 or frame_id in paths:
                raise ValueError("raw RGB frame IDs must be unique and nonnegative")
            path = Path(raw_path)
            if not path.is_absolute():
                if ".." in path.parts:
                    raise ValueError("unsafe raw RGB path mapping")
                path = session / path
            path = path.resolve()
            if not path.is_file():
                raise ValueError(f"raw RGB is missing: {path}")
            paths[frame_id], hashes[frame_id] = path, _sha(path)
        return paths, hashes
    with (session / "frames.csv").open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            frame_id = int(row["frame_id"])
            relative = Path(row["color_path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("unsafe raw RGB path in frames.csv")
            path = session / relative
            if not path.is_file():
                raise ValueError(f"raw RGB is missing: {path}")
            paths[frame_id], hashes[frame_id] = path, _sha(path)
    return paths, hashes


def _read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"raw RGB is unreadable: {path}")
    return image


class _LazyRawRGB(Mapping[int, np.ndarray]):
    """Audit mapping that deliberately performs no image caching."""

    def __init__(self, paths: Mapping[int, Path]) -> None:
        self._paths = dict(paths)

    def __getitem__(self, frame_id: int) -> np.ndarray:
        return _read_rgb(self._paths[frame_id])

    def __iter__(self) -> Iterator[int]:
        return iter(self._paths)

    def __len__(self) -> int:
        return len(self._paths)


def _rss() -> int:
    try:
        import psutil
        return int(psutil.Process(os.getpid()).memory_info().rss)
    except ImportError:
        return 0


def _remap(image: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return cv2.remap(
        image, np.asarray(u, np.float32), np.asarray(v, np.float32),
        interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
    )


def _correct(image: np.ndarray, gain: np.ndarray, bias: np.ndarray) -> np.ndarray:
    value = np.asarray(image, np.float64) / 255.0
    return np.clip(np.rint(np.clip(value * gain + bias, 0.0, 1.0) * 255.0), 0, 255).astype(np.uint8)


def _apply_active_pixels(
    visual: np.ndarray, canvas_y: np.ndarray, canvas_x: np.ndarray,
    composed: np.ndarray, active: np.ndarray,
) -> None:
    """Apply only a frozen B1 support; inactive pixels remain canonical owner."""
    mask = np.asarray(active, bool)
    visual[np.asarray(canvas_y)[mask], np.asarray(canvas_x)[mask]] = composed[mask]


def _write_p3_transaction_provenance(
    provenance: dict[str, np.ndarray],
    pairs: Sequence[tuple[Mapping[str, Any], Mapping[str, np.ndarray]]],
    finalized: Sequence[Any],
) -> None:
    """Bind every P3 photometric/blend pixel to one immutable transaction."""
    valid = np.asarray(provenance["valid"], bool)
    owner_source = np.asarray(provenance["owner_source_index"], np.int32)
    secondary = {
        "blend_transaction_id": np.full(valid.shape, -1, np.int32),
        "secondary_frame_id": np.full(valid.shape, -1, np.int32),
        "secondary_source_index": np.full(valid.shape, -1, np.int32),
        "secondary_source_u": np.full(valid.shape, np.nan, np.float32),
        "secondary_source_v": np.full(valid.shape, np.nan, np.float32),
        "secondary_weight": np.zeros(valid.shape, np.float32),
    }
    for (metadata, arrays), plan in zip(pairs, finalized, strict=True):
        active = np.asarray(plan.active, bool)
        yy = np.asarray(arrays["canvas_y"])[active]
        xx = np.asarray(arrays["canvas_x"])[active]
        transaction_id = int(plan.transaction.pair_index)
        if transaction_id < 0:
            raise ValueError("M6.1 blend transaction ID must be nonnegative")
        if np.any(secondary["blend_transaction_id"][yy, xx] != -1):
            raise ValueError("M6.1 active pixels belong to more than one blend transaction")
        secondary["blend_transaction_id"][yy, xx] = transaction_id
        owner_right = np.asarray(arrays["primary_owner_right"], bool)[active]
        side = np.where(owner_right, "left", "right")
        for label in ("left", "right"):
            select = side == label
            secondary["secondary_frame_id"][yy[select], xx[select]] = int(
                metadata[f"{label}_frame_id"]
            )
            secondary["secondary_source_index"][yy[select], xx[select]] = int(
                metadata[f"{label}_source_index"]
            )
            secondary["secondary_source_u"][yy[select], xx[select]] = np.asarray(
                arrays[f"{label}_source_u"]
            )[active][select]
            secondary["secondary_source_v"][yy[select], xx[select]] = np.asarray(
                arrays[f"{label}_source_v"]
            )[active][select]
        secondary["secondary_weight"][yy, xx] = np.asarray(plan.secondary_weight)[active]
    provenance.update(secondary)
    if "photometric_transaction_id" in provenance:
        provenance["photometric_transaction_id"] = np.where(
            valid, owner_source, -1,
        ).astype(np.int32)


def _pair_files(p2_root: Path) -> list[tuple[dict[str, Any], dict[str, np.ndarray]]]:
    replay = _json(p2_root / "photometric_replay/manifest.json")
    result = []
    for item in replay.get("pairs", []):
        metadata = _json(p2_root / "photometric_replay" / str(item["metadata"]))
        asset = p2_root / "photometric_replay" / str(item["asset"])
        if _sha(asset) != item["asset_sha256"] or _sha(
            p2_root / "photometric_replay" / str(item["metadata"])
        ) != item["metadata_sha256"]:
            raise ValueError("sealed pair evidence SHA mismatch")
        with np.load(asset, allow_pickle=False) as archive:
            arrays = {name: np.asarray(archive[name]) for name in archive.files}
        result.append((metadata, arrays))
    return sorted(result, key=lambda value: int(value[0]["pair_index"]))


def _verify_raw_bindings(
    pairs: Sequence[tuple[Mapping[str, Any], Mapping[str, np.ndarray]]], hashes: Mapping[int, str]
) -> None:
    for metadata, _arrays in pairs:
        for side in ("left", "right"):
            frame = int(metadata[f"{side}_frame_id"])
            if hashes.get(frame) != metadata["raw_rgb_sha256"][side]:
                raise ValueError("raw RGB provenance SHA mismatch")


def _sample_pair(
    metadata: Mapping[str, Any], arrays: Mapping[str, np.ndarray], gains: np.ndarray, biases: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    output = []
    for side in ("left", "right"):
        source = int(metadata[f"{side}_source_index"])
        lab = np.nan_to_num(np.asarray(arrays[f"{side}_lab"], np.float32), nan=0.0)
        bgr = cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)
        corrected = np.clip(np.rint(np.clip(bgr * gains[source] + biases[source], 0, 1) * 255), 0, 255)
        output.append(corrected.astype(np.uint8))
    return output[0], output[1]


def _solver_evidence(
    pairs: Sequence[tuple[Mapping[str, Any], Mapping[str, np.ndarray]]], accepted_pairs: set[int],
    maximum_samples: int,
) -> tuple[S13M61PairEvidence, ...]:
    result = []
    for metadata, arrays in pairs:
        pair_index = int(metadata["pair_index"])
        if pair_index not in accepted_pairs:
            continue
        left, right = _sample_pair(
            metadata, arrays,
            np.ones((max(int(metadata["left_source_index"]), int(metadata["right_source_index"])) + 1, 3)),
            np.zeros((max(int(metadata["left_source_index"]), int(metadata["right_source_index"])) + 1, 3)),
        )
        safe = np.asarray(arrays["solver_matched_safe"], bool)
        train = safe & np.asarray(arrays["train"], bool)
        validation = safe & np.asarray(arrays["validation"], bool)

        def take(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
            selected = values[mask].astype(np.float64) / 255.0
            if selected.shape[0] > maximum_samples:
                indices = np.linspace(0, selected.shape[0] - 1, maximum_samples, dtype=np.int64)
                selected = selected[indices]
            return selected

        result.append(S13M61PairEvidence(
            pair_index=pair_index,
            left_source_index=int(metadata["left_source_index"]),
            right_source_index=int(metadata["right_source_index"]),
            train_left_rgb_linear=take(left, train), train_right_rgb_linear=take(right, train),
            validation_left_rgb_linear=take(left, validation),
            validation_right_rgb_linear=take(right, validation), actionable=bool(np.any(validation)),
        ))
    return tuple(result)


def _blend_candidate(
    metadata: Mapping[str, Any], arrays: Mapping[str, np.ndarray], left: np.ndarray, right: np.ndarray,
    *, benefit: float, mde: float,
) -> M61BlendCandidate:
    canvas_x = np.asarray(arrays["canvas_x"], np.int32)
    x0, x1 = int(canvas_x.min()), int(canvas_x.max()) + 1
    return M61BlendCandidate(
        pair_index=int(metadata["pair_index"]), seam_x_by_row=np.asarray(arrays["seam_x_by_row"], np.int32),
        corridor_x0=x0, corridor_x1=x1, owner_right=np.asarray(arrays["primary_owner_right"], bool),
        safe=np.asarray(arrays["quality_output_safe"], bool),
        protected=np.asarray(arrays["quality_output_protected"], bool),
        common_valid=np.asarray(arrays["common_audited_support"], bool),
        expected_valid=np.asarray(arrays["expected_valid"], bool), left_rgb=left, right_rgb=right,
        structural_risk=float(np.mean(np.asarray(arrays["quality_output_protected"], bool))),
        evidence_strength=float(np.mean(np.asarray(arrays["quality_metric_validation"], bool))),
        minimum_immediate_benefit_linear=benefit, immediate_benefit_mde_linear=mde,
    )


def _source_frames(p2_root: Path) -> dict[int, int]:
    manifest = _json(p2_root / "photometric_replay/owner_domain/manifest.json")
    return {int(item["source_index"]): int(item["frame_id"]) for item in manifest["sources"]}


def _formal_render(
    p2_root: Path, pairs: Sequence[tuple[Mapping[str, Any], Mapping[str, np.ndarray]]],
    raw_paths: Mapping[int, Path], source_frames: Mapping[int, int], gains: np.ndarray,
    biases: np.ndarray, blend_plans: Sequence[Any], effective: S13M61EffectiveConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], list[Any], dict[str, Any]]:
    with np.load(p2_root / "p2_pixel_provenance.npz", allow_pickle=False) as archive:
        provenance = {name: np.asarray(archive[name]).copy() for name in archive.files}
    valid = np.asarray(provenance["valid"], bool)
    height, width = valid.shape
    if height * width > effective.visual_quality.maximum_canvas_megapixels * 1_000_000:
        raise ValueError("M6.1 canvas resource cap exceeded")
    owner_source = np.asarray(provenance["owner_source_index"], np.int32)
    pair_by_source: dict[int, list[tuple[int, str, Mapping[str, np.ndarray]]]] = {}
    for index, (metadata, arrays) in enumerate(pairs):
        for side in ("left", "right"):
            pair_by_source.setdefault(int(metadata[f"{side}_source_index"]), []).append((index, side, arrays))
    pair_rgb: list[dict[str, np.ndarray]] = [dict() for _ in pairs]
    owner_only = np.zeros((height, width, 3), np.uint8)

    source_bounds: dict[int, tuple[int, int, int, int]] = {}

    def remap_source(source: int) -> np.ndarray:
        required = valid & (owner_source == source)
        coordinates: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        for _index, side, arrays in pair_by_source.get(source, []):
            support = np.asarray(arrays[f"{side}_geometry_support"], bool)
            coordinates.append((np.asarray(arrays["canvas_y"]), np.asarray(arrays["canvas_x"]),
                                np.asarray(arrays[f"{side}_source_u"]), np.asarray(arrays[f"{side}_source_v"])))
            yy, xx = coordinates[-1][:2]
            required[yy[support], xx[support]] = True
        yy, xx = np.nonzero(required)
        if not yy.size:
            raise ValueError("formal source has no required pixels")
        y0, y1, x0, x1 = int(yy.min()), int(yy.max()) + 1, int(xx.min()), int(xx.max()) + 1
        map_u = np.full((y1 - y0, x1 - x0), -1, np.float32)
        map_v = np.full_like(map_u, -1)
        own = required & (owner_source == source)
        oy, ox = np.nonzero(own)
        map_u[oy - y0, ox - x0] = np.asarray(provenance["source_u"])[oy, ox]
        map_v[oy - y0, ox - x0] = np.asarray(provenance["source_v"])[oy, ox]
        for _index, side, arrays in pair_by_source.get(source, []):
            support = np.asarray(arrays[f"{side}_geometry_support"], bool)
            cy, cx = np.asarray(arrays["canvas_y"])[support], np.asarray(arrays["canvas_x"])[support]
            map_u[cy - y0, cx - x0] = np.asarray(arrays[f"{side}_source_u"])[support]
            map_v[cy - y0, cx - x0] = np.asarray(arrays[f"{side}_source_v"])[support]
        source_bounds[source] = (y0, y1, x0, x1)
        image = _read_rgb(raw_paths[source_frames[source]])
        return _correct(_remap(image, map_u, map_v), gains[source], biases[source])

    streamer = M61ROIStreamer(
        remap_source, maximum_resident_source_rois=effective.visual_quality.maximum_resident_source_rois,
        maximum_live_array_bytes=effective.visual_quality.maximum_live_array_bytes,
        maximum_process_rss_bytes=effective.visual_quality.maximum_process_rss_bytes,
    )
    for source in range(len(source_frames)):
        roi = streamer.source_roi(source)
        y0, y1, x0, x1 = source_bounds[source]
        local_owner = valid[y0:y1, x0:x1] & (owner_source[y0:y1, x0:x1] == source)
        owner_only[y0:y1, x0:x1][local_owner] = roi[local_owner]
        for pair_index, side, arrays in pair_by_source.get(source, []):
            yy, xx = np.asarray(arrays["canvas_y"]), np.asarray(arrays["canvas_x"])
            sampled = np.zeros((*yy.shape, 3), np.uint8)
            support = np.asarray(arrays[f"{side}_geometry_support"], bool)
            sampled[support] = roi[yy[support] - y0, xx[support] - x0]
            pair_rgb[pair_index][side] = sampled
        streamer.release(source)
    streaming = asdict(streamer.report())

    visual = owner_only.copy()
    finalized, transactions = [], {}
    for index, ((metadata, arrays), plan) in enumerate(zip(pairs, blend_plans, strict=True)):
        candidate = _blend_candidate(metadata, arrays, pair_rgb[index]["left"], pair_rgb[index]["right"],
                                     benefit=0.0, mde=0.0)
        composed, final_plan, tx_arrays = compose_and_canonicalize_m61_blend(candidate, plan)
        yy, xx = np.asarray(arrays["canvas_y"]), np.asarray(arrays["canvas_x"])
        # B0 and every non-active corridor pixel remain the canonical primary
        # owner.  Only the already-selected B1 transaction may change pixels.
        active = np.asarray(final_plan.active, bool)
        _apply_active_pixels(visual, yy, xx, composed, active)
        finalized.append(final_plan)
        transactions[int(metadata["pair_index"])] = tx_arrays
    _write_p3_transaction_provenance(provenance, pairs, finalized)
    return visual, owner_only, provenance, finalized, streaming


def _solution_document(plan: Any, topology: Any, source_frames: Mapping[int, int]) -> dict[str, Any]:
    components = []
    for component, model in zip(topology.components, plan.selected_model_by_component, strict=True):
        components.append({"component_index": component.component_index,
                           "source_indices": list(component.source_indices), "selected_model": model})
    all_q0 = all(model == "Q0_identity" for model in plan.selected_model_by_component)
    sources = []
    model_by_source = {source: model for component, model in zip(topology.components, plan.selected_model_by_component, strict=True)
                       for source in component.source_indices}
    for source in range(plan.source_count):
        parameter = {
            "source_index": source,
            "frame_id": source_frames[source],
            "model": model_by_source[source],
            "gain_bgr": plan.gains_rgb[source].tolist(),
            "bias_bgr_linear": plan.biases_rgb[source].tolist(),
        }
        parameter_sha256 = hashlib.sha256(json.dumps(
            parameter, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")).hexdigest()
        sources.append({
            **parameter,
            "photometric_transaction_id": source,
            "parameter_sha256": parameter_sha256,
        })
    return {"schema": plan.schema, "graph_topology_sha256": plan.topology_sha256,
            "selected_model": "Q0_identity" if all_q0 else "component_selected",
            "components": components, "sources": sources,
            "identity_fallback_inside_solved_component_count": plan.identity_fallback_inside_solved_component_count}


def _old_image(path: Path | None) -> np.ndarray | None:
    if path is None:
        return None
    if path.is_dir():
        for name in ("visual_panorama.png", "geometry_and_seam_panorama_owner_only.png",
                     "video_panorama.png", "panorama.png"):
            candidate = path / name
            if candidate.is_file():
                path = candidate
                break
        else:
            return None
    return cv2.imread(str(path), cv2.IMREAD_COLOR)


def _diagnostics(
    root: Path, p2: np.ndarray, new: np.ndarray, old_path: Path | None,
    pairs: Sequence[tuple[Mapping[str, Any], Mapping[str, np.ndarray]]], finalized: Sequence[Any],
) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=False)
    old = _old_image(old_path)
    cv2.imwrite(str(root / "new_vs_p2_full.png"), np.concatenate((p2, new), axis=1))
    if old is not None and old.shape == new.shape:
        cv2.imwrite(str(root / "new_vs_old_full.png"), np.concatenate((old, new), axis=1))
    seam_root = root / "seam_comparisons"
    seam_root.mkdir()
    pair_rows = []
    for (metadata, arrays), plan in zip(pairs, finalized, strict=True):
        x = np.asarray(arrays["canvas_x"], np.int32)
        x0, x1 = max(0, int(x.min())), min(new.shape[1], int(x.max()) + 1)
        views = [p2[:, x0:x1], new[:, x0:x1]]
        if old is not None and old.shape == new.shape:
            views.insert(1, old[:, x0:x1])
        asset = seam_root / f"pair_{int(metadata['pair_index']):04d}.png"
        cv2.imwrite(str(asset), np.concatenate(views, axis=1))
        pair_delta = np.max(np.abs(new[:, x0:x1].astype(np.int16) -
                                   p2[:, x0:x1].astype(np.int16)), axis=2)
        pair_rows.append({"pair_index": int(metadata["pair_index"]),
                          "model": plan.transaction.model, "crop_x0": x0, "crop_x1": x1,
                          "changed_pixel_count": int(np.count_nonzero(pair_delta)),
                          "asset": asset.relative_to(root).as_posix()})
    delta = np.max(np.abs(new.astype(np.int16) - p2.astype(np.int16)), axis=2)
    quality = {"schema": "gemini305-video-s13-m61-post-quality/v1", "selection_authority": False,
               "post_render_parameter_changes": 0, "changed_pixel_count": int(np.count_nonzero(delta)),
               "maximum_absolute_dn": int(delta.max(initial=0)),
               "blend_models": [plan.transaction.model for plan in finalized], "pairs": pair_rows}
    _write_json(root / "quality_report.json", quality)
    return quality


def run_s13_m61_acceptance(
    *, p2_root: str | Path, session: str | Path, output: str | Path,
    candidate_config: str | Path, old_m6_p3: str | Path | None = None,
    pointer_root: str | Path | None = None,
    raw_rgb_paths: Mapping[int, str | Path] | None = None,
) -> dict[str, Any]:
    """Run one branch and append P3 without replacing an existing stage.

    ``output`` is the S013 generation root.  ``pointer_root`` is the shared
    output root containing that generation; standalone acceptance runs retain
    their historical layout by defaulting it to ``output``.
    """
    p2_root, session, output = Path(p2_root).resolve(), Path(session).resolve(), Path(output).resolve()
    shared_pointer = pointer_root is not None
    pointer_root = output if pointer_root is None else Path(pointer_root).resolve()
    if shared_pointer:
        if p2_root != output / "P2":
            raise ValueError("M6.1 shared P3 requires p2_root to be output/P2")
        try:
            output.relative_to(pointer_root)
        except ValueError as exc:
            raise ValueError("M6.1 output generation must be inside pointer_root") from exc
    candidate_path = Path(candidate_config).resolve()
    p2_completion = verify_stage(
        p2_root, completion_name="P2_completion.json", schema=M61_P2_COMPLETION_SCHEMA,
    )
    if p2_completion.get("stage") != "P2" or p2_completion.get("hard_audit_passed") is not True:
        raise ValueError("M6.1 parent P2 is not a hard-audited shared stage")
    parent_sha = _sha(p2_root / "P2_completion.json")
    p2_result_asset = Path(str(p2_completion.get("result_asset", "")))
    if (p2_result_asset.is_absolute() or ".." in p2_result_asset.parts
            or p2_result_asset == Path(".")):
        raise ValueError("M6.1 P2 result asset path is unsafe")
    p2_result_path = p2_root / p2_result_asset
    p2_result_sha = _sha(p2_result_path)
    p2_assets = p2_completion.get("assets_sha256")
    if (not isinstance(p2_assets, Mapping)
            or p2_completion.get("result_asset_sha256") != p2_result_sha
            or p2_assets.get(p2_result_asset.as_posix()) != p2_result_sha):
        raise ValueError("M6.1 P2 result asset binding is invalid")
    p2_provenance_sha = _sha(p2_root / "p2_pixel_provenance.npz")
    if p2_assets.get("p2_pixel_provenance.npz") != p2_provenance_sha:
        raise ValueError("M6.1 P2 provenance binding is invalid")
    replay_manifest = _json(p2_root / "photometric_replay/manifest.json")
    effective = load_s13_m61_effective_config(
        candidate_path, photometric_evidence_config=replay_manifest["photometric_evidence_config"],
        config_root=candidate_path.parents[3],
    )
    topology = load_sealed_graph_topology(
        p2_root / "photometric_replay/graph_topology",
        expected_evidence_config_sha256=effective.photometric_evidence_config_sha256,
    )
    invocation_rss = _rss()
    pairs = _pair_files(p2_root)
    raw_paths, raw_hashes = _load_session_rgb(session, raw_rgb_paths)
    _verify_raw_bindings(pairs, raw_hashes)
    accepted = {edge.pair_index for edge in topology.edges if edge.accepted}
    evidence = _solver_evidence(pairs, accepted,
                                effective.photometric_evidence_config.maximum_samples_per_pair)
    photometric_plan = solve_and_select_s13_m61_photometric(
        topology, evidence, solver_config=effective.solver, selection_config=effective.selection,
    )
    predicted = [_blend_candidate(
        metadata, arrays, *_sample_pair(metadata, arrays, photometric_plan.gains_rgb,
                                        photometric_plan.biases_rgb),
        benefit=effective.blend.minimum_immediate_seam_benefit_dn / 255.0,
        mde=effective.blend.minimum_immediate_seam_benefit_mde_dn / 255.0,
    ) for metadata, arrays in pairs]
    # Pair corridors can overlap.  A pixel protected by any frozen pair is
    # globally owner-only even when a neighbouring pair labels it safe.
    global_protected: set[tuple[int, int]] = set()
    for _metadata, arrays in pairs:
        protected = np.asarray(arrays["quality_output_protected"], bool)
        for row, column in zip(
            np.asarray(arrays["canvas_y"])[protected],
            np.asarray(arrays["canvas_x"])[protected], strict=True,
        ):
            global_protected.add((int(row), int(column)))
    globally_safe_predictions = []
    for candidate, (_metadata, arrays) in zip(predicted, pairs, strict=True):
        blocked = np.fromiter(
            ((int(row), int(column)) in global_protected for row, column in zip(
                np.asarray(arrays["canvas_y"]).ravel(),
                np.asarray(arrays["canvas_x"]).ravel(), strict=True,
            )), dtype=bool, count=np.asarray(arrays["canvas_x"]).size,
        ).reshape(np.asarray(arrays["canvas_x"]).shape)
        globally_safe_predictions.append(
            dataclasses.replace(candidate, safe=candidate.safe & ~blocked)
        )
    blend_plans = plan_m61_blends(globally_safe_predictions)  # final selection boundary

    final = output / "P3"
    output.mkdir(parents=True, exist_ok=True)
    if final.exists():
        raise FileExistsError("P3 already exists; formal rerender is forbidden")
    pending = output / f".P3.{uuid.uuid4().hex[:12]}.pending"
    pending.mkdir(parents=True, exist_ok=False)
    pointer_path = pointer_root / "current_latest.json"
    pointer_before = pointer_path.read_bytes() if pointer_path.is_file() else None
    published_final = False
    pointer_update_attempted = False
    try:
        source_frames = _source_frames(p2_root)
        visual, owner_only, provenance, finalized, streaming = _formal_render(
            p2_root, pairs, raw_paths, source_frames, photometric_plan.gains_rgb,
            photometric_plan.biases_rgb, blend_plans, effective,
        )
        p2_image = cv2.imread(str(p2_result_path), cv2.IMREAD_COLOR)
        if p2_image is None or p2_image.shape != visual.shape:
            raise ValueError("M6.1 sealed P2 result image is unreadable or has changed shape")
        cv2.imwrite(str(pending / "visual_panorama.png"), visual)
        cv2.imwrite(str(pending / "photometric_owner_only.png"), owner_only)
        cv2.imwrite(str(pending / "p3_valid_mask.png"), np.asarray(provenance["valid"], np.uint8) * 255)
        np.savez_compressed(pending / "p3_pixel_provenance.npz", **provenance)
        solution = _solution_document(photometric_plan, topology, source_frames)
        _write_json(pending / "photometric_solution.json", solution)
        effective_doc = effective.canonical_document()
        effective_doc.update({"effective_config_sha256": effective.effective_config_sha256,
                              "graph_topology_sha256": topology.topology_sha256,
                              "photometric_bounds": {"minimum_gain": effective.solver.minimum_gain,
                               "maximum_gain": effective.solver.maximum_gain,
                               "maximum_absolute_bias_linear": effective.solver.maximum_absolute_bias_linear}})
        _write_json(pending / "effective_config.json", effective_doc)
        full_masks = {name: np.zeros(provenance["valid"].shape, bool) for name in
                      ("active", "safe", "protected", "common_valid", "expected_valid")}
        for (metadata, arrays), plan in zip(pairs, finalized, strict=True):
            yy, xx = np.asarray(arrays["canvas_y"]), np.asarray(arrays["canvas_x"])
            for name, local in (("active", plan.active), ("safe", plan.safe),
                                ("protected", plan.protected),
                                ("common_valid", arrays["common_audited_support"]),
                                ("expected_valid", arrays["expected_valid"])):
                full_masks[name][yy, xx] |= np.asarray(local, bool)
        np.savez_compressed(pending / "pair_masks.npz", **full_masks)
        shared_parent_layout = p2_root == output / "P2"
        parent_completion_reference = (
            "../P2/P2_completion.json" if shared_parent_layout
            else str((p2_root / "P2_completion.json").resolve())
        )
        parent_result_reference = (
            f"../P2/{p2_result_asset.as_posix()}" if shared_parent_layout
            else str(p2_result_path.resolve())
        )
        parent_provenance_reference = (
            "../P2/p2_pixel_provenance.npz" if shared_parent_layout
            else str((p2_root / "p2_pixel_provenance.npz").resolve())
        )
        _write_json(pending / "p2_parent_reference.json", {
            "schema": P2_PARENT_REFERENCE_SCHEMA,
            "parent_stage": "P2",
            "completion": parent_completion_reference,
            "completion_sha256": parent_sha,
            "p2_completion_sha256": parent_sha,
            "result_asset": parent_result_reference,
            "result_asset_sha256": p2_result_sha,
            "pixel_provenance": parent_provenance_reference,
            "pixel_provenance_sha256": p2_provenance_sha,
            "raw_rgb_provenance_replay": True,
        })
        tx_root = pending / "blend_transactions"
        tx_root.mkdir()
        tx_entries, reasons = [], []
        mask_sha = _sha(pending / "pair_masks.npz")
        for (metadata, _arrays), plan in zip(pairs, finalized, strict=True):
            tx = asdict(plan.transaction)
            tx.update({
                "transaction_id": int(plan.transaction.pair_index),
                "parent_completion_sha256": parent_sha,
                "parent_narrow_replay_sha256": metadata["parent_narrow_replay_sha256"],
                "parent_pair_transaction_sha256": metadata["parent_pair_transaction_sha256"],
                "pair_masks_sha256": mask_sha,
                "selection_completed_before_formal_render": True,
            })
            asset = f"blend_transactions/pair_{plan.transaction.pair_index:04d}.json"
            _write_json(pending / asset, tx)
            tx_entries.append({
                "transaction_id": int(plan.transaction.pair_index),
                "pair_index": plan.transaction.pair_index,
                "model": plan.transaction.model,
                "active_pixel_count": int(plan.transaction.active_pixel_count),
                "asset": asset,
                "asset_sha256": _sha(pending / asset),
            })
            if plan.transaction.fallback_reason:
                reasons.append("blend_benefit_not_detectable" if
                               plan.transaction.fallback_reason in {"immediate_benefit_below_mde", "no_safe_two_pixel_support"}
                               else "blend_ineligible")
        _write_json(pending / "blend_transactions.json", {
            "schema": M61_BLEND_MANIFEST_SCHEMA,
            "parent_completion_sha256": parent_sha, "pair_masks_sha256": mask_sha,
            "pairs": tx_entries, "fallback_reasons": sorted(set(reasons)),
            "all_pairs_reported": len(tx_entries) == len(pairs),
            "candidate_selection_completed_before_formal_render": True,
            "post_render_parameter_changes": 0,
        })
        performance = {"schema": M61_PERFORMANCE_SCHEMA, **streaming, "formal_render_attempt_count": 1,
                       "formal_remap_count_by_source": [1] * topology.source_count,
                       "hard_audit_render_invocations": 0, "fallback_identity_rebuild_count": 0,
                       "forbidden_invocations": {"q4_nonzero_solver": 0, "b2_b3_b4": 0,
                                                 "post_render_reselection": 0, "synthetic_source": 0}}
        performance["peak_process_rss_bytes"] = max(
            int(performance["peak_process_rss_bytes"]), invocation_rss, _rss()
        )
        performance["resource_measurement_scope"] = "runner_invocation_through_seal"
        _write_json(pending / "performance.json", performance)
        assets = {}
        for path in sorted(pending.rglob("*")):
            if path.is_file():
                assets[path.relative_to(pending).as_posix()] = _sha(path)
        _write_json(pending / "seal_manifest.json", {"schema": SEAL_SCHEMA, "assets_sha256": assets})
        audit = audit_m61_p3_files(
            p2_root, pending, expected_parent_sha256=parent_sha,
            expected_effective_config_sha256=effective.effective_config_sha256,
            expected_topology_sha256=topology.topology_sha256,
            raw_rgb_by_frame=_LazyRawRGB({source_frames[source]: raw_paths[source_frames[source]]
                                         for source in source_frames}),
        )
        if not audit["passed"]:
            raise ValueError(f"M6.1 hard audit failed: {audit['failures']}")
        _write_json(pending / "hard_audit.json", audit)
        result_asset = "visual_panorama.png"
        result_sha = _sha(pending / result_asset)
        provenance_sha = _sha(pending / "p3_pixel_provenance.npz")
        completion = seal_stage(
            pending,
            completion_name="P3_completion.json",
            schema=M61_P3_COMPLETION_SCHEMA,
            metadata={
                "generation_id": str(p2_completion["generation_id"]),
                "stage": "P3",
                "hard_audit_passed": True,
                "parent_stage": "P2",
                "parent_completion_sha256": parent_sha,
                "parent_result_sha256": p2_result_sha,
                "parent_pixel_provenance_sha256": p2_provenance_sha,
                "result_asset": result_asset,
                "result_asset_sha256": result_sha,
                "pixel_provenance_sha256": provenance_sha,
                "hard_audit_sha256": _sha(pending / "hard_audit.json"),
                "effective_config_sha256": effective.effective_config_sha256,
                "graph_topology_sha256": topology.topology_sha256,
                "photometric_solution_sha256": _sha(pending / "photometric_solution.json"),
                "blend_transactions_sha256": _sha(pending / "blend_transactions.json"),
                "formal_raw_rgb_unique_sources": topology.source_count,
                "formal_raw_rgb_remap_invocations": sum(
                    int(value) for value in performance["formal_remap_count_by_source"]
                ),
                "maximum_real_contributors_per_pixel": (
                    2 if np.any(np.asarray(provenance["secondary_weight"]) > 0.0) else 1
                ),
                "diagnostic_only": True,
                "production": False,
                "production_eligible": False,
                "production_lock_eligible": False,
            },
        )
        verify_stage(
            pending, completion_name="P3_completion.json", schema=M61_P3_COMPLETION_SCHEMA,
        )
        os.replace(pending, final)
        published_final = True
        verify_promoted_m61_p3(
            p2_root, final, expected_parent_sha256=parent_sha,
            expected_effective_config_sha256=effective.effective_config_sha256,
            expected_topology_sha256=topology.topology_sha256,
            raw_rgb_by_frame=_LazyRawRGB({source_frames[source]: raw_paths[source_frames[source]]
                                         for source in source_frames}),
        )
        diagnostics = output / "validation_m61"
        quality = _diagnostics(
            diagnostics, p2_image, visual, Path(old_m6_p3).resolve() if old_m6_p3 else None,
            pairs, finalized,
        )
        repro = {"schema": RUN_SCHEMA, "p2_completion_sha256": parent_sha,
                 "p3_completion_sha256": _sha(final / "P3_completion.json"),
                 "effective_config_sha256": effective.effective_config_sha256,
                 "graph_topology_sha256": topology.topology_sha256,
                 "raw_rgb_sha256_by_frame": {str(k): v for k, v in sorted(raw_hashes.items())},
                 "formal_render_attempt_count": 1, "post_render_parameter_changes": 0,
                 "quality_report_sha256": _sha(diagnostics / "quality_report.json")}
        _write_json(diagnostics / "reproducibility.json", repro)
        # Pointer publication is deliberately the final fallible output action.
        # Diagnostic quality has no authority over this hard-audited stage.
        pointer_update_attempted = True
        if shared_pointer:
            pointer = update_current_latest(
                pointer_root, output, stage="P3", completion_name="P3_completion.json",
                completion_schema=M61_P3_COMPLETION_SCHEMA, result_asset=result_asset,
            )
        else:
            pointer = {
                "schema": "gemini305-video-s13-current-stage/v2",
                "stage": "P3",
                "generation_id": str(p2_completion["generation_id"]),
                "completion_sha256": _sha(final / "P3_completion.json"),
            }
            _write_json(pointer_path, pointer)
        return {"schema": RUN_SCHEMA, "p3": str(final), "pointer": pointer,
                "completion": completion, "hard_audit_passed": True,
                "quality": quality, "reproducibility": repro}
    except Exception:
        if pointer_update_attempted:
            try:
                current = _json(pointer_path)
            except (OSError, ValueError):
                current = {}
            if published_final and current.get("completion_sha256") == _sha(final / "P3_completion.json"):
                _restore_pointer(pointer_path, pointer_before)
        if published_final and final.exists():
            shutil.rmtree(final)
        raise
    finally:
        if pending.exists():
            shutil.rmtree(pending)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one sealed S013 M6.1 branch")
    parser.add_argument("--p2-root", required=True, type=Path)
    parser.add_argument("--session", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--candidate-config", required=True, type=Path)
    parser.add_argument("--old-m6-p3", type=Path)
    parser.add_argument("--pointer-root", type=Path)
    args = parser.parse_args(argv)
    result = run_s13_m61_acceptance(**vars(args))
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


__all__ = ["RUN_SCHEMA", "main", "run_s13_m61_acceptance"]
