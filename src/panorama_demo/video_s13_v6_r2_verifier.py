"""Fail-closed semantic verifier for sealed S1.3 v6-r2 P2 artifacts."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .video_s13_bundle import sha256_file
from .video_s13_contract import (
    S13_M51_R4_ALGORITHM_ID,
    S13_M51_R4_CONTRACT_SCHEMA,
    S13_M51_R4_IMPLEMENTATION_ID,
    S13_M51_R4_P2_COMPLETION_SCHEMA,
)
from .video_s13_m51_r4_component_chain import (
    SourceMapOracle,
    audit_s13_obligation_coverage,
    source_map_oracle_from_arrays,
)


_PAIR_SCHEMA = "gemini305-video-s13-m5-pair-transaction/v5"
_AGGREGATE_SCHEMA = "gemini305-video-s13-m5-pair-transactions/v4"
_COMPONENT_SCHEMA = "gemini305-video-s13-component-chain-transactions/v1"
_CORRECTION_SCHEMA = "gemini305-video-s13-source-corrections/v1"
_ORACLE_MANIFEST_SCHEMA = "gemini305-video-s13-source-map-oracles/v1"
_REPLAY_SCHEMA = "gemini305-video-s13-p2-replay/v2"
_PROVENANCE_SCHEMA = "gemini305-video-s13-p2-provenance/v6-r2"

_COMPONENT_STABLE_FIELDS = (
    "application_state", "baseline_c2e_obligations", "chains", "segments",
    "dependency_groups", "component_match_matrices",
    "exact_evidence_propagation", "roi_candidate_pixels", "roi_preview_count",
    "evidence_assets", "segment_assets", "obligation_outcomes", "split_lineage",
)


def component_decision_stable_payload(
    component_manifest: Mapping[str, object],
) -> dict[str, object]:
    """Select all decision-authority fields covered by the stable digest."""

    return {field: component_manifest.get(field, [] if field not in {
        "application_state", "roi_candidate_pixels", "roi_preview_count"
    } else ("none" if field == "application_state" else 0))
            for field in _COMPONENT_STABLE_FIELDS}


def component_decision_stable_sha256(
    component_manifest: Mapping[str, object],
) -> str:
    return hashlib.sha256(json.dumps(
        component_decision_stable_payload(component_manifest),
        sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()


def segment_decision_stable_payload(
    segment_document: Mapping[str, object],
) -> dict[str, object]:
    """Canonical segment authority excluding only its own digest."""

    return {
        str(key): value
        for key, value in segment_document.items()
        if key != "decision_payload_stable_sha256"
    }


def segment_decision_stable_sha256(
    segment_document: Mapping[str, object],
) -> str:
    return hashlib.sha256(json.dumps(
        segment_decision_stable_payload(segment_document),
        sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()


def verify_s13_v6_r1_noop_exact_comparison(
    baseline_p2: str | Path,
    candidate_p2: str | Path,
    *,
    output_path: str | Path | None = None,
) -> dict[str, object]:
    """Prove a v6-r2 no-op retained every v6-r1 pixel/map authority array."""

    baseline = Path(baseline_p2).resolve()
    candidate = Path(candidate_p2).resolve()
    component = _json(
        candidate / "component_chain_transactions/manifest.json",
        "component transaction manifest",
    )
    if component.get("application_state") != "none" or component.get(
        "accepted_segment_ids"
    ) not in ([], ()):
        raise ValueError("S1.3 v6-r2 no-op exact comparison requires no applied segment")

    comparisons: list[dict[str, object]] = []

    def array_equal(left: np.ndarray, right: np.ndarray) -> bool:
        left_value, right_value = np.asarray(left), np.asarray(right)
        equal_nan = (
            np.issubdtype(left_value.dtype, np.floating)
            or np.issubdtype(left_value.dtype, np.complexfloating)
        )
        return np.array_equal(left_value, right_value, equal_nan=equal_nan)

    def sampling_maps_equal(
        left_u: np.ndarray,
        left_v: np.ndarray,
        right_u: np.ndarray,
        right_v: np.ndarray,
        authority_mask: np.ndarray,
    ) -> bool:
        """Compare the exact OpenCV INTER_LINEAR sampling authority.

        Legacy v6-r1 evaluated identical global coordinates in independently
        sized crops, so SIMD evaluation could differ by one float32 ULP and
        invalid UV retained noncanonical values.  R4 must canonicalize invalid
        UV and use one oracle.  The relevant rollback authority is therefore
        the fixed-point interpolation cell on valid/active samples, which is
        exactly what OpenCV consumes to produce the byte-exact panorama.
        """

        import cv2

        arrays = tuple(np.asarray(value, dtype=np.float32) for value in (
            left_u, left_v, right_u, right_v
        ))
        mask = np.asarray(authority_mask, dtype=bool)
        if any(value.shape != mask.shape for value in arrays):
            return False
        if any(np.any(~np.isfinite(value[mask])) for value in arrays):
            return False
        normalized = []
        for value in arrays:
            copied = np.array(value, copy=True, order="C")
            copied[~mask] = 0.0
            if copied.ndim == 1:
                copied = copied[None, :]
            normalized.append(copied)
        comparison_mask = mask[None, :] if mask.ndim == 1 else mask
        left_fixed = cv2.convertMaps(
            normalized[0], normalized[1], cv2.CV_16SC2, nninterpolation=False
        )
        right_fixed = cv2.convertMaps(
            normalized[2], normalized[3], cv2.CV_16SC2, nninterpolation=False
        )
        return all(
            np.array_equal(
                left_value[comparison_mask], right_value[comparison_mask]
            )
            for left_value, right_value in zip(left_fixed, right_fixed, strict=True)
        )

    def compare_array_file(
        relative: str,
        fields: Sequence[str] | None = None,
        *,
        candidate_relative: str | None = None,
    ) -> None:
        left_path = baseline / relative
        right_path = candidate / (candidate_relative or relative)
        if not left_path.is_file() or not right_path.is_file():
            raise ValueError(f"S1.3 no-op exact comparison asset is missing: {relative}")
        if left_path.suffix.lower() == ".png":
            import cv2

            left = cv2.imread(str(left_path), cv2.IMREAD_UNCHANGED)
            right = cv2.imread(str(right_path), cv2.IMREAD_UNCHANGED)
            exact = left is not None and right is not None and np.array_equal(left, right)
            names: list[str] = []
        else:
            with np.load(left_path, allow_pickle=False) as left_archive, np.load(
                right_path, allow_pickle=False
            ) as right_archive:
                names = list(fields) if fields is not None else list(left_archive.files)
                provenance_map_names = {"source_u", "source_v", "valid"}
                if (
                    relative == "p2_pixel_provenance.npz"
                    and provenance_map_names <= set(names)
                ):
                    if not all(name in right_archive.files for name in names):
                        exact = False
                    else:
                        valid = np.asarray(left_archive["valid"], dtype=bool)
                        exact = (
                            np.array_equal(valid, np.asarray(right_archive["valid"], dtype=bool))
                            and sampling_maps_equal(
                                left_archive["source_u"], left_archive["source_v"],
                                right_archive["source_u"], right_archive["source_v"], valid,
                            )
                        )
                        secondary_names = {
                            "secondary_source_u", "secondary_source_v"
                        }
                        if secondary_names <= set(names):
                            active = (
                                np.asarray(left_archive["secondary_frame_id"]) >= 0
                            ) & (np.asarray(left_archive["secondary_weight"]) > 0.0)
                            exact = exact and sampling_maps_equal(
                                left_archive["secondary_source_u"],
                                left_archive["secondary_source_v"],
                                right_archive["secondary_source_u"],
                                right_archive["secondary_source_v"],
                                active,
                            )
                        ignored = {
                            "source_u", "source_v", "secondary_source_u",
                            "secondary_source_v",
                        }
                        exact = exact and all(
                            array_equal(left_archive[name], right_archive[name])
                            for name in names if name not in ignored
                        )
                elif relative.startswith("pair_replay/"):
                    if not all(name in right_archive.files for name in names):
                        exact = False
                    else:
                        ignored = {"parent_pair_transaction_sha256"}
                        exact = True
                        for side in ("left", "right"):
                            valid_name = f"{side}_valid"
                            u_name, v_name = (
                                f"{side}_source_u", f"{side}_source_v"
                            )
                            if {valid_name, u_name, v_name} <= set(names):
                                valid = np.asarray(left_archive[valid_name], dtype=bool)
                                exact = exact and (
                                    np.array_equal(
                                        valid,
                                        np.asarray(right_archive[valid_name], dtype=bool),
                                    )
                                    and sampling_maps_equal(
                                        left_archive[u_name], left_archive[v_name],
                                        right_archive[u_name], right_archive[v_name], valid,
                                    )
                                )
                                ignored.update({valid_name, u_name, v_name})
                        exact = exact and all(
                            array_equal(left_archive[name], right_archive[name])
                            for name in names if name not in ignored
                        )
                elif fields is None and left_archive.files != right_archive.files:
                    exact = False
                else:
                    exact = all(
                        name in left_archive.files
                        and name in right_archive.files
                        and left_archive[name].dtype == right_archive[name].dtype
                        and left_archive[name].shape == right_archive[name].shape
                        and array_equal(left_archive[name], right_archive[name])
                        for name in names
                    )
        comparisons.append({
            "asset": relative,
            "fields": names,
            "exact": bool(exact),
        })

    compare_array_file("geometry_and_seam_panorama_owner_only.png")
    compare_array_file("p2_seams.npz")
    # correction_field_id is new v6-r2 authority. Every pre-existing v6-r1
    # provenance field must remain exactly equal in a no-op run.
    baseline_provenance = baseline / "p2_pixel_provenance.npz"
    with np.load(baseline_provenance, allow_pickle=False) as archive:
        provenance_fields = tuple(archive.files)
    compare_array_file("p2_pixel_provenance.npz", provenance_fields)

    baseline_replay = _json(
        baseline / "p2_replay_manifest.json", "baseline replay manifest"
    )
    candidate_replay = _json(
        candidate / "p2_replay_manifest.json", "candidate replay manifest"
    )
    baseline_assets = {
        int(row["pair_index"]): str(row["asset"])
        for row in baseline_replay.get("pairs", ())
        if isinstance(row, Mapping)
    }
    candidate_assets = {
        int(row["pair_index"]): str(row["asset"])
        for row in candidate_replay.get("pairs", ())
        if isinstance(row, Mapping)
    }
    if set(baseline_assets) != set(candidate_assets):
        raise ValueError("S1.3 no-op replay pair universe differs from v6-r1")
    for pair_index in sorted(baseline_assets):
        left_relative = baseline_assets[pair_index]
        right_relative = candidate_assets[pair_index]
        if left_relative != right_relative:
            raise ValueError("S1.3 no-op replay asset identity differs from v6-r1")
        compare_array_file(left_relative, candidate_relative=right_relative)
    passed = bool(comparisons) and all(row["exact"] is True for row in comparisons)
    report = {
        "schema": "gemini305-video-s13-v6-r1-noop-exact-comparison/v1",
        "passed": passed,
        "application_state": "none",
        "comparison_count": len(comparisons),
        "comparisons": comparisons,
    }
    if output_path is not None:
        target = Path(output_path).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.tmp")
        temporary.write_text(
            json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(target)
    return report


def _json(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"S1.3 v6-r2 {label} is missing or invalid") from exc
    if not isinstance(value, dict):
        raise ValueError(f"S1.3 v6-r2 {label} must be an object")
    return value


def _sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def _canonical(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _canonical(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("S1.3 v6-r2 transaction contains a nonfinite float")
        rounded = round(number, 6)
        return 0.0 if rounded == 0.0 else rounded
    return value


def canonical_pair_transaction_sha256(transaction: Mapping[str, object]) -> str:
    """Return the frozen v5 pair digest, excluding its digest field."""

    payload = dict(transaction)
    payload.pop("result_stage_sha256", None)
    return _sha(_canonical(payload))


def canonical_source_map_slice_sha256(
    oracle: SourceMapOracle,
    bbox_xyxy: tuple[int, int, int, int],
) -> str:
    """Hash one canonical global-coordinate slice of a SourceMapOracle."""

    x0, y0, x1, y1 = (int(value) for value in bbox_xyxy)
    ox0, oy0, ox1, oy1 = oracle.domain_xyxy
    if x0 < ox0 or y0 < oy0 or x1 > ox1 or y1 > oy1 or x1 <= x0 or y1 <= y0:
        raise ValueError("S1.3 v6-r2 source-map slice is outside its oracle")
    region = np.s_[y0 - oy0 : y1 - oy0, x0 - ox0 : x1 - ox0]
    header = {
        "schema": "gemini305-video-s13-source-map-oracle-slice/v1",
        "parent_oracle_sha256": oracle.oracle_sha256,
        "source_index": oracle.source_index,
        "bbox_xyxy": [x0, y0, x1, y1],
        "shape": [y1 - y0, x1 - x0],
        "dtypes": {"u": "<f4", "v": "<f4", "valid": "u1", "field_id": "<i4"},
    }
    digest = hashlib.sha256(
        json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    for array in (oracle.u[region], oracle.v[region], oracle.valid[region], oracle.field_id[region]):
        digest.update(np.ascontiguousarray(array).tobytes(order="C"))
    return digest.hexdigest()


def _safe_asset(root: Path, value: object, label: str) -> Path:
    relative = Path(str(value))
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError(f"S1.3 v6-r2 {label} path is unsafe")
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"S1.3 v6-r2 {label} escapes P2") from exc
    return path


def _require_sha(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"S1.3 v6-r2 {label} is not a SHA-256 digest")
    return value


def _verify_completion(p2: Path) -> tuple[dict[str, object], dict[str, object]]:
    completion = _json(p2 / "P2_completion.json", "P2 completion")
    required_identity = {
        "schema": S13_M51_R4_P2_COMPLETION_SCHEMA,
        "algorithm_id": S13_M51_R4_ALGORITHM_ID,
        "implementation_id": S13_M51_R4_IMPLEMENTATION_ID,
        "contract_schema": S13_M51_R4_CONTRACT_SCHEMA,
        "provenance_schema": _PROVENANCE_SCHEMA,
        "p2_replay_schema": _REPLAY_SCHEMA,
        "stage": "P2",
        "sealed": True,
        "hard_audit_passed": True,
        "m6_eligible": False,
        "working_tree_dirty": False,
    }
    if any(completion.get(key) != expected for key, expected in required_identity.items()):
        raise ValueError("S1.3 v6-r2 completion identity is incomplete or invalid")
    for name in (
        "config_sha256",
        "candidate_manifest_sha256",
        "generation_manifest_sha256",
        "component_transaction_manifest_sha256",
        "source_correction_manifest_sha256",
        "source_map_oracle_manifest_sha256",
        "aggregate_pair_transaction_manifest_sha256",
        "p2_replay_manifest_sha256",
    ):
        _require_sha(completion.get(name), f"completion {name}")
    source_commit = completion.get("source_commit")
    if not isinstance(source_commit, str) or len(source_commit) != 40:
        raise ValueError("S1.3 v6-r2 completion source commit is invalid")
    generation_manifest = _json(p2.parent / "generation_manifest.json", "generation manifest")
    if sha256_file(p2.parent / "generation_manifest.json") != completion["generation_manifest_sha256"]:
        raise ValueError("S1.3 v6-r2 generation manifest SHA mismatch")
    algorithm = generation_manifest.get("algorithm")
    if not isinstance(algorithm, Mapping) or any(
        algorithm.get(key) != completion.get(key)
        for key in (
            "algorithm_id",
            "implementation_id",
            "config_sha256",
            "candidate_manifest_sha256",
            "source_commit",
            "working_tree_dirty",
        )
    ):
        raise ValueError("S1.3 v6-r2 generation/completion identity disagrees")
    assets = completion.get("assets_sha256")
    if not isinstance(assets, Mapping) or not assets:
        raise ValueError("S1.3 v6-r2 completion has no asset bindings")
    for relative, expected in assets.items():
        path = _safe_asset(p2, relative, "completion asset")
        if not path.is_file() or sha256_file(path) != _require_sha(expected, "asset SHA"):
            raise ValueError(f"S1.3 v6-r2 completion asset SHA mismatch: {relative}")
    hard_audit = _json(p2 / "hard_audit.json", "hard audit")
    if hard_audit.get("passed") is not True:
        raise ValueError("S1.3 v6-r2 hard audit did not pass")
    return completion, generation_manifest


def _load_oracles(
    p2: Path,
    manifest: Mapping[str, object],
) -> tuple[dict[int, SourceMapOracle], dict[int, SourceMapOracle], dict[int, tuple[int, int]]]:
    rows = manifest.get("sources")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise ValueError("S1.3 v6-r2 source-map manifest has no sources")
    result: dict[int, SourceMapOracle] = {}
    base_result: dict[int, SourceMapOracle] = {}
    raw_sizes: dict[int, tuple[int, int]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("S1.3 v6-r2 source-map row is invalid")
        path = _safe_asset(p2, row.get("asset"), "source-map oracle")
        if sha256_file(path) != _require_sha(row.get("asset_sha256"), "oracle asset SHA"):
            raise ValueError("S1.3 v6-r2 source-map oracle asset SHA mismatch")
        try:
            with np.load(path, allow_pickle=False) as stored:
                arrays = {name: np.asarray(stored[name]).copy() for name in stored.files}
        except (OSError, ValueError, KeyError) as exc:
            raise ValueError("S1.3 v6-r2 source-map oracle asset is invalid") from exc
        required = {"source_index", "domain_xyxy", "u", "v", "valid", "field_id"}
        if set(arrays) != required:
            raise ValueError("S1.3 v6-r2 source-map oracle fields disagree")
        source_index = int(np.asarray(arrays["source_index"]).item())
        domain = tuple(int(value) for value in np.asarray(arrays["domain_xyxy"]).tolist())
        if source_index != row.get("source_index") or list(domain) != row.get("domain_xyxy"):
            raise ValueError("S1.3 v6-r2 source-map oracle header disagrees")
        oracle = source_map_oracle_from_arrays(
            source_index=source_index,
            domain_xyxy=domain,  # type: ignore[arg-type]
            u=arrays["u"],
            v=arrays["v"],
            valid=arrays["valid"],
            field_id=arrays["field_id"],
        )
        if oracle.oracle_sha256 != _require_sha(row.get("oracle_sha256"), "oracle SHA"):
            raise ValueError("S1.3 v6-r2 source-map oracle canonical SHA mismatch")
        if source_index in result:
            raise ValueError("S1.3 v6-r2 source-map source index is duplicated")
        result[source_index] = oracle
        base_path = _safe_asset(p2, row.get("base_asset"), "base source-map oracle")
        if sha256_file(base_path) != _require_sha(
            row.get("base_asset_sha256"), "base oracle asset SHA"
        ):
            raise ValueError("S1.3 v6-r2 base source-map oracle asset SHA mismatch")
        try:
            with np.load(base_path, allow_pickle=False) as stored:
                base_arrays = {
                    name: np.asarray(stored[name]).copy() for name in stored.files
                }
        except (OSError, ValueError, KeyError) as exc:
            raise ValueError("S1.3 v6-r2 base source-map oracle asset is invalid") from exc
        if set(base_arrays) != required:
            raise ValueError("S1.3 v6-r2 base source-map oracle fields disagree")
        base_source_index = int(np.asarray(base_arrays["source_index"]).item())
        base_domain = tuple(
            int(value) for value in np.asarray(base_arrays["domain_xyxy"]).tolist()
        )
        if base_source_index != source_index or base_domain != domain:
            raise ValueError("S1.3 v6-r2 base/final source-map headers disagree")
        base_oracle = source_map_oracle_from_arrays(
            source_index=base_source_index,
            domain_xyxy=base_domain,  # type: ignore[arg-type]
            u=base_arrays["u"],
            v=base_arrays["v"],
            valid=base_arrays["valid"],
            field_id=base_arrays["field_id"],
        )
        if base_oracle.oracle_sha256 != _require_sha(
            row.get("base_oracle_sha256"), "base oracle SHA"
        ):
            raise ValueError("S1.3 v6-r2 base source-map canonical SHA mismatch")
        if np.any(base_oracle.field_id != -1):
            raise ValueError("S1.3 v6-r2 pre-C2E base oracle contains correction labels")
        size = row.get("raw_source_size")
        if (
            not isinstance(size, Sequence)
            or isinstance(size, (str, bytes))
            or len(size) != 2
            or any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in size)
        ):
            raise ValueError("S1.3 v6-r2 raw source size authority is invalid")
        base_result[source_index] = base_oracle
        raw_sizes[source_index] = (int(size[0]), int(size[1]))
    if len(result) != manifest.get("source_count"):
        raise ValueError("S1.3 v6-r2 source-map source count disagrees")
    if set(base_result) != set(result) or set(raw_sizes) != set(result):
        raise ValueError("S1.3 v6-r2 base/final source-map universe disagrees")
    return result, base_result, raw_sizes


def _field_table(corrections: Mapping[str, object]) -> dict[int, Mapping[str, object]]:
    rows = corrections.get("field_id_table")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise ValueError("S1.3 v6-r2 source correction field table is invalid")
    result: dict[int, Mapping[str, object]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("S1.3 v6-r2 source correction field row is invalid")
        field_id = row.get("field_id")
        if not isinstance(field_id, int) or isinstance(field_id, bool) or field_id < 0:
            raise ValueError("S1.3 v6-r2 source correction field ID is invalid")
        if field_id in result:
            raise ValueError("S1.3 v6-r2 source correction field ID is duplicated")
        for name in (
            "segment_transaction_sha256",
            "support_sha256",
            "source_correction_asset_sha256",
        ):
            _require_sha(row.get(name), f"field table {name}")
        segment_id = row.get("segment_id")
        if not isinstance(segment_id, str) or not segment_id:
            raise ValueError("S1.3 v6-r2 source correction segment ID is invalid")
        correction_rows = row.get("correction_rows")
        if not isinstance(correction_rows, Sequence) or isinstance(
            correction_rows, (str, bytes)
        ) or not correction_rows:
            raise ValueError("S1.3 v6-r2 field table correction-row authority is invalid")
        result[field_id] = row
    if tuple(sorted(result)) != tuple(range(len(result))):
        raise ValueError("S1.3 v6-r2 source correction field IDs are not canonical")
    return result


def _segment_support_authorities(
    component: Mapping[str, object],
) -> dict[str, str]:
    rows = component.get("segment_assets", [])
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise ValueError("S1.3 v6-r2 segment support authority table is invalid")
    result: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("S1.3 v6-r2 segment support authority row is invalid")
        segment_id = row.get("segment_id")
        if not isinstance(segment_id, str) or not segment_id or segment_id in result:
            raise ValueError("S1.3 v6-r2 segment support identity is invalid")
        result[segment_id] = _require_sha(
            row.get("support_authority_sha256"), "segment support authority SHA"
        )
    return result


def _all_mapping_keys(value: object) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            keys.add(str(key))
            keys.update(_all_mapping_keys(child))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for child in value:
            keys.update(_all_mapping_keys(child))
    return keys


def _segment_authorities(
    p2: Path, component: Mapping[str, object], *, config_sha256: str
) -> set[str]:
    rows = component.get("segment_transaction_assets", [])
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise ValueError("S1.3 v6-r2 segment transaction asset table is invalid")
    result: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("S1.3 v6-r2 segment transaction asset row is invalid")
        path = _safe_asset(p2, row.get("asset"), "segment transaction asset")
        expected = _require_sha(row.get("sha256"), "segment transaction SHA")
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError("S1.3 v6-r2 segment transaction asset SHA mismatch")
        document = _json(path, "segment transaction")
        if document.get("config_sha256") != config_sha256:
            raise ValueError("S1.3 v6-r2 segment transaction config authority disagrees")
        if document.get("decision_payload_stable_sha256") != (
            segment_decision_stable_sha256(document)
        ):
            raise ValueError("S1.3 v6-r2 segment stable decision authority disagrees")
        if expected in result:
            raise ValueError("S1.3 v6-r2 segment transaction SHA is duplicated")
        result.add(expected)
    return result


def _correction_asset_authorities(
    p2: Path, corrections: Mapping[str, object]
) -> set[str]:
    rows = corrections.get("sources")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise ValueError("S1.3 v6-r2 source correction sources are invalid")
    result: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("S1.3 v6-r2 source correction source row is invalid")
        if "correction_asset" not in row and "correction_asset_sha256" not in row:
            continue
        path = _safe_asset(p2, row.get("correction_asset"), "source correction asset")
        expected = _require_sha(
            row.get("correction_asset_sha256"), "source correction asset SHA"
        )
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError("S1.3 v6-r2 source correction asset SHA mismatch")
        result.add(expected)
    return result


def _verify_correction_asset_semantics(
    p2: Path,
    corrections: Mapping[str, object],
    field_table: Mapping[int, Mapping[str, object]],
    segment_support: Mapping[str, str],
) -> dict[str, object]:
    rows = corrections.get("sources", [])
    maximum_offset = 0.0
    overlap_pixels = 0
    asset_count = 0
    observed_rows: list[dict[str, object]] = []
    for row in rows:
        if not isinstance(row, Mapping) or "correction_asset" not in row:
            continue
        path = _safe_asset(p2, row["correction_asset"], "source correction asset")
        with np.load(path, allow_pickle=False) as stored:
            domains = np.asarray(stored["domains_xyxy"], dtype=np.int32)
            fields = np.asarray(stored["field_id"], dtype=np.int32)
            segment_ids = np.asarray(stored["segment_ids"])
            correction_shas = np.asarray(stored["correction_sha256"])
            support_shas = np.asarray(stored["support_authority_sha256"])
            if (
                domains.shape != (len(fields), 4)
                or len(segment_ids) != len(fields)
                or len(correction_shas) != len(fields)
                or len(support_shas) != len(fields)
            ):
                raise ValueError("S1.3 v6-r2 correction asset authority shapes disagree")
            source_index = row.get("source_index")
            if not isinstance(source_index, int) or isinstance(source_index, bool):
                raise ValueError("S1.3 v6-r2 correction source identity is invalid")
            asset_sha = _require_sha(
                row.get("correction_asset_sha256"), "source correction asset SHA"
            )
            if list(segment_ids.astype(str)) != list(row.get("contributors", [])):
                raise ValueError("S1.3 v6-r2 correction contributors disagree")
            occupied: set[tuple[int, int]] = set()
            for index, domain in enumerate(domains):
                names = (
                    f"delta_u_{index:04d}", f"delta_v_{index:04d}",
                    f"weight_{index:04d}",
                )
                if any(name not in stored.files for name in names):
                    raise ValueError("S1.3 v6-r2 correction asset arrays are incomplete")
                du = np.asarray(stored[names[0]], dtype=np.float64)
                dv = np.asarray(stored[names[1]], dtype=np.float64)
                weight = np.asarray(stored[names[2]], dtype=np.float64)
                x0, y0, x1, y1 = (int(value) for value in domain)
                segment_id = str(segment_ids[index])
                field_id = int(fields[index])
                correction_sha = _require_sha(
                    str(correction_shas[index]), "correction row SHA"
                )
                support_sha = _require_sha(
                    str(support_shas[index]), "correction row support SHA"
                )
                field = field_table.get(field_id)
                if (
                    field is None
                    or field.get("segment_id") != segment_id
                    or segment_support.get(segment_id) != support_sha
                    or field.get("support_sha256") != support_sha
                ):
                    raise ValueError("S1.3 v6-r2 correction row segment/support authority disagrees")
                if (
                    du.shape != dv.shape or du.shape != weight.shape
                    or du.shape != (y1 - y0, x1 - x0)
                    or not np.isfinite(du).all() or not np.isfinite(dv).all()
                    or not np.isfinite(weight).all() or np.any(weight < 0.0)
                ):
                    raise ValueError("S1.3 v6-r2 correction asset numeric domain is invalid")
                active = weight > 0.0
                if np.any(du[~active] != 0.0) or np.any(dv[~active] != 0.0):
                    raise ValueError("S1.3 v6-r2 correction is nonzero outside omega-map")
                if np.any(active):
                    maximum_offset = max(
                        maximum_offset, float(np.max(np.hypot(du[active], dv[active])))
                    )
                active_y, active_x = np.nonzero(active)
                coordinates = {(y0 + int(y), x0 + int(x)) for y, x in zip(active_y, active_x, strict=True)}
                overlap_pixels += len(occupied & coordinates)
                occupied.update(coordinates)
                observed_rows.append({
                    "source_index": source_index,
                    "row_index": index,
                    "field_id": field_id,
                    "segment_id": segment_id,
                    "domain_xyxy": [x0, y0, x1, y1],
                    "correction_sha256": correction_sha,
                    "support_sha256": support_sha,
                    "source_correction_asset_sha256": asset_sha,
                })
            asset_count += 1
    if maximum_offset > 3.0 + 1e-6:
        raise ValueError("S1.3 v6-r2 maximum component offset is exceeded")
    if overlap_pixels:
        raise ValueError("S1.3 v6-r2 resolved correction fields overlap")
    declared_rows: list[dict[str, object]] = []
    for field_id, field in field_table.items():
        for row in field["correction_rows"]:  # validated by _field_table
            if not isinstance(row, Mapping):
                raise ValueError("S1.3 v6-r2 field correction-row entry is invalid")
            declared = {
                "source_index": row.get("source_index"),
                "row_index": row.get("row_index"),
                "field_id": field_id,
                "segment_id": field.get("segment_id"),
                "domain_xyxy": row.get("domain_xyxy"),
                "correction_sha256": row.get("correction_sha256"),
                "support_sha256": row.get("support_sha256"),
                "source_correction_asset_sha256": row.get(
                    "source_correction_asset_sha256"
                ),
            }
            for name in (
                "correction_sha256", "support_sha256",
                "source_correction_asset_sha256",
            ):
                _require_sha(declared[name], f"declared correction row {name}")
            declared_rows.append(declared)
    def row_key(value: Mapping[str, object]) -> tuple[int, int]:
        return int(value["field_id"]), int(value["source_index"])

    if sorted(observed_rows, key=row_key) != sorted(declared_rows, key=row_key):
        raise ValueError("S1.3 v6-r2 field table/correction row authority disagrees")
    return {
        "asset_count": asset_count,
        "correction_row_count": len(observed_rows),
        "maximum_component_offset_px": maximum_offset,
        "resolved_field_overlap_pixel_count": overlap_pixels,
    }


def _verify_pair_transactions(
    p2: Path,
    aggregate: Mapping[str, object],
    authority: Mapping[str, str],
    segment_authorities: set[str],
) -> list[dict[str, object]]:
    if aggregate.get("schema") != _AGGREGATE_SCHEMA:
        raise ValueError("S1.3 v6-r2 aggregate pair schema is invalid")
    for name, expected in authority.items():
        if aggregate.get(name) != expected:
            raise ValueError("S1.3 v6-r2 aggregate pair DAG binding disagrees")
    asset_rows = aggregate.get("pair_transaction_assets")
    embedded = aggregate.get("pairs")
    if (
        not isinstance(asset_rows, Sequence)
        or isinstance(asset_rows, (str, bytes))
        or not isinstance(embedded, Sequence)
        or isinstance(embedded, (str, bytes))
        or len(asset_rows) != len(embedded)
    ):
        raise ValueError("S1.3 v6-r2 aggregate pair transaction assets are incomplete")
    result: list[dict[str, object]] = []
    for pair_index, (asset_row, embedded_pair) in enumerate(
        zip(asset_rows, embedded, strict=True)
    ):
        if not isinstance(asset_row, Mapping) or not isinstance(embedded_pair, Mapping):
            raise ValueError("S1.3 v6-r2 aggregate pair row is invalid")
        if asset_row.get("pair_index") != pair_index:
            raise ValueError("S1.3 v6-r2 aggregate pair ordering disagrees")
        path = _safe_asset(p2, asset_row.get("asset"), "pair transaction asset")
        expected_sha = _require_sha(asset_row.get("sha256"), "pair transaction asset SHA")
        if not path.is_file() or sha256_file(path) != expected_sha:
            raise ValueError("S1.3 v6-r2 pair transaction asset SHA mismatch")
        pair = _json(path, "pair transaction")
        if pair != dict(embedded_pair):
            raise ValueError("S1.3 v6-r2 embedded pair transaction disagrees")
        if pair.get("schema") != _PAIR_SCHEMA or pair.get("transaction_id") != f"m5-pair-{pair_index:04d}":
            raise ValueError("S1.3 v6-r2 pair transaction identity is invalid")
        if pair.get("result_stage_sha256") != canonical_pair_transaction_sha256(pair):
            raise ValueError("S1.3 v6-r2 pair transaction digest mismatch")
        reference = pair.get("component_chain_c2e")
        if not isinstance(reference, Mapping) or any(
            reference.get(name) != expected for name, expected in authority.items()
        ):
            raise ValueError("S1.3 v6-r2 pair transaction DAG binding disagrees")
        refs = reference.get("refs")
        if not isinstance(refs, Sequence) or isinstance(refs, (str, bytes)):
            raise ValueError("S1.3 v6-r2 pair segment references are invalid")
        for row in refs:
            if not isinstance(row, Mapping):
                raise ValueError("S1.3 v6-r2 pair segment reference is invalid")
            digest = _require_sha(
                row.get("segment_transaction_sha256"), "pair segment transaction SHA"
            )
            if digest not in segment_authorities:
                raise ValueError("S1.3 v6-r2 pair references an unknown segment transaction")
        result.append(pair)
    if aggregate.get("all_pairs_reported") is not True:
        raise ValueError("S1.3 v6-r2 aggregate pair coverage is incomplete")
    return result


def _oracle_region(
    oracle: SourceMapOracle, bbox: tuple[int, int, int, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x0, y0, x1, y1 = bbox
    ox0, oy0, _ox1, _oy1 = oracle.domain_xyxy
    region = np.s_[y0 - oy0 : y1 - oy0, x0 - ox0 : x1 - ox0]
    return oracle.u[region], oracle.v[region], oracle.valid[region], oracle.field_id[region]


def _verify_base_final_map_semantics(
    final_oracles: Mapping[int, SourceMapOracle],
    base_oracles: Mapping[int, SourceMapOracle],
    raw_sizes: Mapping[int, tuple[int, int]],
    field_table: Mapping[int, Mapping[str, object]],
) -> dict[str, object]:
    """Recompute every post-C2E map gate from independent base/final assets."""

    exterior_mismatch = 0
    valid_mismatch = 0
    source_oob = 0
    maximum_displacement = 0.0
    minimum_jacobian = math.inf
    minimum_jacobian_ratio = math.inf
    changed_pixels = 0
    labelled_pixels = 0
    for source_index, final in final_oracles.items():
        base = base_oracles.get(source_index)
        if base is None or base.domain_xyxy != final.domain_xyxy:
            raise ValueError("S1.3 v6-r2 base/final source-map authority disagrees")
        base_valid = base.valid != 0
        final_valid = final.valid != 0
        valid_mismatch += int(np.count_nonzero(base_valid != final_valid))
        changed = final_valid & (
            (final.u.view(np.uint32) != base.u.view(np.uint32))
            | (final.v.view(np.uint32) != base.v.view(np.uint32))
        )
        labelled = final.field_id >= 0
        labelled_pixels += int(np.count_nonzero(labelled))
        changed_pixels += int(np.count_nonzero(changed))
        exterior_mismatch += int(np.count_nonzero(changed & ~labelled))
        if np.any(labelled & ~final_valid):
            raise ValueError("S1.3 v6-r2 correction field lies outside valid map support")
        if any(
            int(value) not in field_table
            for value in np.unique(final.field_id[labelled])
        ):
            raise ValueError("S1.3 v6-r2 final oracle field ID is absent from field table")
        displacement = np.hypot(
            final.u.astype(np.float64) - base.u.astype(np.float64),
            final.v.astype(np.float64) - base.v.astype(np.float64),
        )
        if np.any(labelled):
            maximum_displacement = max(
                maximum_displacement, float(np.max(displacement[labelled]))
            )
        raw_width, raw_height = raw_sizes[source_index]
        source_oob += int(np.count_nonzero(
            final_valid
            & (
                (final.u < 0.0) | (final.u > float(raw_width - 1))
                | (final.v < 0.0) | (final.v > float(raw_height - 1))
            )
        ))
        if min(final.u.shape) >= 2:
            final_du_dy, final_du_dx = np.gradient(final.u.astype(np.float64))
            final_dv_dy, final_dv_dx = np.gradient(final.v.astype(np.float64))
            base_du_dy, base_du_dx = np.gradient(base.u.astype(np.float64))
            base_dv_dy, base_dv_dx = np.gradient(base.v.astype(np.float64))
            final_det = final_du_dx * final_dv_dy - final_du_dy * final_dv_dx
            base_det = base_du_dx * base_dv_dy - base_du_dy * base_dv_dx
            interior = final_valid & base_valid & labelled
            interior[1:, :] &= final_valid[:-1, :] & base_valid[:-1, :]
            interior[:-1, :] &= final_valid[1:, :] & base_valid[1:, :]
            interior[:, 1:] &= final_valid[:, :-1] & base_valid[:, :-1]
            interior[:, :-1] &= final_valid[:, 1:] & base_valid[:, 1:]
            evaluable = interior & np.isfinite(final_det) & np.isfinite(base_det)
            if np.any(evaluable):
                minimum_jacobian = min(
                    minimum_jacobian, float(np.min(final_det[evaluable]))
                )
                positive_base = evaluable & (base_det > 0.0)
                if np.any(positive_base):
                    minimum_jacobian_ratio = min(
                        minimum_jacobian_ratio,
                        float(np.min(final_det[positive_base] / base_det[positive_base])),
                    )
    if valid_mismatch:
        raise ValueError("S1.3 v6-r2 base/final valid support changed")
    if exterior_mismatch:
        raise ValueError("S1.3 v6-r2 omega-out exterior source map changed")
    if source_oob:
        raise ValueError("S1.3 v6-r2 final source map is out of bounds")
    if math.isfinite(minimum_jacobian) and minimum_jacobian < 0.5 - 1e-9:
        raise ValueError("S1.3 v6-r2 final inverse-map Jacobian failed")
    if maximum_displacement > 8.0 + 1e-6:
        raise ValueError("S1.3 v6-r2 combined source-map displacement exceeded")
    return {
        "omega_out_exterior_uv_mismatch_count": exterior_mismatch,
        "valid_support_mismatch_count": valid_mismatch,
        "source_out_of_bounds_pixel_count": source_oob,
        "maximum_combined_map_displacement_px": maximum_displacement,
        "minimum_final_inverse_map_jacobian": (
            None if not math.isfinite(minimum_jacobian) else minimum_jacobian
        ),
        "minimum_final_to_base_jacobian_ratio": (
            None if not math.isfinite(minimum_jacobian_ratio)
            else minimum_jacobian_ratio
        ),
        "changed_map_pixel_count": changed_pixels,
        "labelled_map_pixel_count": labelled_pixels,
    }


def _verify_replay(
    p2: Path,
    replay: Mapping[str, object],
    pairs: Sequence[Mapping[str, object]],
    oracles: Mapping[int, SourceMapOracle],
    authority: Mapping[str, str],
    aggregate_sha: str,
    field_table: Mapping[int, Mapping[str, object]],
    segment_authorities: set[str],
) -> None:
    if replay.get("schema") != _REPLAY_SCHEMA:
        raise ValueError("S1.3 v6-r2 replay schema is invalid")
    if any(replay.get(name) != expected for name, expected in authority.items()):
        raise ValueError("S1.3 v6-r2 replay DAG binding disagrees")
    if replay.get("aggregate_pair_transaction_manifest_sha256") != aggregate_sha:
        raise ValueError("S1.3 v6-r2 replay aggregate pair binding disagrees")
    rows = replay.get("pairs")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)) or len(rows) != len(pairs):
        raise ValueError("S1.3 v6-r2 replay pair coverage disagrees")
    for pair_index, row in enumerate(rows):
        if not isinstance(row, Mapping) or row.get("pair_index") != pair_index:
            raise ValueError("S1.3 v6-r2 replay pair row is invalid")
        path = _safe_asset(p2, row.get("asset"), "replay asset")
        try:
            with np.load(path, allow_pickle=False) as stored:
                arrays = {name: np.asarray(stored[name]).copy() for name in stored.files}
        except (OSError, ValueError, KeyError) as exc:
            raise ValueError("S1.3 v6-r2 replay asset is invalid") from exc
        required = {
            "pair_index", "left_source_index", "right_source_index", "corridor_x0",
            "corridor_x1", "left_source_u", "left_source_v", "left_valid",
            "right_source_u", "right_source_v", "right_valid",
            "left_component_correction_field_id", "right_component_correction_field_id",
            "parent_pair_transaction_sha256",
        }
        if not required.issubset(arrays):
            raise ValueError("S1.3 v6-r2 replay-v2 required fields are missing")
        left_index = int(arrays["left_source_index"].item())
        right_index = int(arrays["right_source_index"].item())
        x0, x1 = int(arrays["corridor_x0"].item()), int(arrays["corridor_x1"].item())
        height = arrays["left_source_u"].shape[0]
        bbox = (x0, 0, x1, height)
        pair_asset = _safe_asset(p2, row.get("parent_pair_transaction"), "replay parent pair")
        pair_sha = sha256_file(pair_asset)
        if (
            row.get("parent_pair_transaction_sha256") != pair_sha
            or str(arrays["parent_pair_transaction_sha256"].item()) != pair_sha
        ):
            raise ValueError("S1.3 v6-r2 replay pair transaction binding disagrees")
        for prefix, source_index in (("left", left_index), ("right", right_index)):
            oracle = oracles.get(source_index)
            if oracle is None:
                raise ValueError("S1.3 v6-r2 replay references an unknown oracle")
            if row.get(f"{prefix}_source_map_oracle_sha256") != oracle.oracle_sha256:
                raise ValueError("S1.3 v6-r2 replay oracle authority disagrees")
            expected_slice_sha = canonical_source_map_slice_sha256(oracle, bbox)
            if row.get(f"{prefix}_source_map_slice_sha256") != expected_slice_sha:
                raise ValueError("S1.3 v6-r2 replay oracle slice SHA disagrees")
            expected = _oracle_region(oracle, bbox)
            actual = (
                np.asarray(arrays[f"{prefix}_source_u"], dtype="<f4"),
                np.asarray(arrays[f"{prefix}_source_v"], dtype="<f4"),
                np.asarray(arrays[f"{prefix}_valid"], dtype=np.uint8),
                np.asarray(arrays[f"{prefix}_component_correction_field_id"], dtype="<i4"),
            )
            if any(not np.array_equal(a, b) for a, b in zip(actual, expected, strict=True)):
                raise ValueError("S1.3 v6-r2 replay map disagrees with its oracle")
            labels = actual[3]
            if any(int(value) not in field_table for value in np.unique(labels) if value >= 0):
                raise ValueError("S1.3 v6-r2 replay field ID is absent from field table")
        relevant = row.get("relevant_segment_transaction_sha256")
        if not isinstance(relevant, Sequence) or isinstance(relevant, (str, bytes)):
            raise ValueError("S1.3 v6-r2 replay segment authority is invalid")
        for digest in relevant:
            verified = _require_sha(digest, "replay segment transaction SHA")
            if verified not in segment_authorities:
                raise ValueError("S1.3 v6-r2 replay references an unknown segment transaction")


def _verify_provenance(
    p2: Path,
    oracles: Mapping[int, SourceMapOracle],
    field_table: Mapping[int, Mapping[str, object]],
) -> dict[str, object]:
    try:
        with np.load(p2 / "p2_pixel_provenance.npz", allow_pickle=False) as stored:
            arrays = {name: np.asarray(stored[name]).copy() for name in stored.files}
    except (OSError, ValueError, KeyError) as exc:
        raise ValueError("S1.3 v6-r2 provenance is invalid") from exc
    required = {
        "owner_source_index", "source_u", "source_v", "valid",
        "component_correction_field_id",
    }
    if not required.issubset(arrays):
        raise ValueError("S1.3 v6-r2 provenance required fields are missing")
    valid = np.asarray(arrays["valid"], dtype=bool)
    owner = np.asarray(arrays["owner_source_index"], dtype=np.int32)
    labels = np.asarray(arrays["component_correction_field_id"], dtype=np.int32)
    if labels.shape != valid.shape or np.any(labels[~valid] != -1):
        raise ValueError("S1.3 v6-r2 provenance correction labels are invalid")
    if any(int(value) not in field_table for value in np.unique(labels) if value >= 0):
        raise ValueError("S1.3 v6-r2 provenance field ID is absent from field table")
    rows, columns = np.indices(valid.shape)
    for source_index in np.unique(owner[valid]):
        oracle = oracles.get(int(source_index))
        if oracle is None:
            raise ValueError("S1.3 v6-r2 provenance references an unknown oracle")
        selected = valid & (owner == source_index)
        x = columns[selected]
        y = rows[selected]
        x0, y0, x1, y1 = oracle.domain_xyxy
        if np.any((x < x0) | (x >= x1) | (y < y0) | (y >= y1)):
            raise ValueError("S1.3 v6-r2 provenance lies outside its oracle")
        oy, ox = y - y0, x - x0
        checks = (
            np.array_equal(np.asarray(arrays["source_u"])[selected], oracle.u[oy, ox]),
            np.array_equal(np.asarray(arrays["source_v"])[selected], oracle.v[oy, ox]),
            np.all(oracle.valid[oy, ox] != 0),
            np.array_equal(labels[selected], oracle.field_id[oy, ox]),
        )
        if not all(checks):
            raise ValueError("S1.3 v6-r2 provenance map disagrees with its oracle")
    try:
        with np.load(p2 / "p2_seams.npz", allow_pickle=False) as stored:
            seams = np.asarray(stored["seams_x_by_row"], dtype=np.int32)
    except (OSError, ValueError, KeyError) as exc:
        raise ValueError("S1.3 v6-r2 seam topology asset is invalid") from exc
    if seams.ndim != 2 or seams.shape[1] != valid.shape[0]:
        raise ValueError("S1.3 v6-r2 seam topology shape disagrees")
    if seams.size and (
        np.any(seams < 0)
        or np.any(seams > valid.shape[1])
        or np.any(np.diff(seams, axis=0) < 0)
    ):
        raise ValueError("S1.3 v6-r2 seam family topology is invalid")
    columns = np.arange(valid.shape[1], dtype=np.int32)[None, :]
    derived_owner = np.zeros(valid.shape, dtype=np.int32)
    for seam in seams:
        derived_owner += columns >= seam[:, None]
    if np.any(owner[valid] != derived_owner[valid]) or np.any(owner[~valid] != -1):
        raise ValueError("S1.3 v6-r2 owner provenance disagrees with seam topology")
    return {
        "seam_count": int(seams.shape[0]),
        "valid_owner_pixel_count": int(np.count_nonzero(valid)),
        "owner_valid_topology_unchanged": True,
        "seam_topology_valid": True,
    }


def verify_s13_v6_r2_p2(p2: str | Path) -> dict[str, object]:
    """Verify the complete semantic and hash DAG of one sealed v6-r2 P2."""

    root = Path(p2).resolve()
    component_path = root / "component_chain_transactions/manifest.json"
    component = _json(component_path, "component transaction manifest")
    if component.get("schema") != _COMPONENT_SCHEMA:
        raise ValueError("S1.3 v6-r2 component transaction schema is invalid")
    stable_sha = component.get("decision_payload_stable_sha256")
    if (
        not isinstance(stable_sha, str)
        or stable_sha != component_decision_stable_sha256(component)
    ):
        raise ValueError("S1.3 v6-r2 component stable decision authority disagrees")
    completion, _generation = _verify_completion(root)
    corrections_path = root / "source_corrections/manifest.json"
    oracle_manifest_path = root / "source_maps/manifest.json"
    aggregate_path = root / "pair_transactions.json"
    replay_path = root / "p2_replay_manifest.json"
    corrections = _json(corrections_path, "source correction manifest")
    oracle_manifest = _json(oracle_manifest_path, "source-map oracle manifest")
    aggregate = _json(aggregate_path, "aggregate pair transaction manifest")
    replay = _json(replay_path, "replay-v2 manifest")
    component_sha = sha256_file(component_path)
    corrections_sha = sha256_file(corrections_path)
    oracle_manifest_sha = sha256_file(oracle_manifest_path)
    aggregate_sha = sha256_file(aggregate_path)
    replay_sha = sha256_file(replay_path)
    explicit = {
        "component_transaction_manifest_sha256": component_sha,
        "source_correction_manifest_sha256": corrections_sha,
        "source_map_oracle_manifest_sha256": oracle_manifest_sha,
        "aggregate_pair_transaction_manifest_sha256": aggregate_sha,
        "p2_replay_manifest_sha256": replay_sha,
    }
    if any(completion.get(name) != expected for name, expected in explicit.items()):
        raise ValueError("S1.3 v6-r2 completion manifest binding disagrees")
    if corrections.get("schema") != _CORRECTION_SCHEMA or corrections.get(
        "parent_component_transaction_manifest_sha256"
    ) != component_sha:
        raise ValueError("S1.3 v6-r2 source correction DAG binding disagrees")
    application_states = {
        completion.get("application_state"),
        component.get("application_state"),
        corrections.get("application_state"),
    }
    if len(application_states) != 1 or next(iter(application_states)) not in {
        "complete", "partial", "none"
    }:
        raise ValueError("S1.3 v6-r2 application state lineage disagrees")
    application_state = next(iter(application_states))
    if not isinstance(component.get("repair_complete"), bool) or completion.get(
        "repair_complete"
    ) != component.get("repair_complete"):
        raise ValueError("S1.3 v6-r2 repair-complete lineage disagrees")
    obligation_coverage = audit_s13_obligation_coverage(
        component.get("baseline_c2e_obligations", []),
        component.get("segments", []),
        component.get("accepted_segment_ids", []),
    )
    if obligation_coverage["repair_complete"] is not component.get("repair_complete"):
        raise ValueError("S1.3 v6-r2 repair-complete obligation authority disagrees")
    accepted = component.get("accepted_segment_ids")
    if (
        not isinstance(accepted, Sequence)
        or isinstance(accepted, (str, bytes))
        or any(not isinstance(value, str) or not value for value in accepted)
        or len(set(accepted)) != len(accepted)
    ):
        raise ValueError("S1.3 v6-r2 accepted segment authority is invalid")
    if application_state == "none" and accepted:
        raise ValueError("S1.3 v6-r2 no-op state has accepted correction authority")
    if application_state in {"partial", "complete"} and not accepted:
        raise ValueError("S1.3 v6-r2 applied state has no accepted correction authority")
    if oracle_manifest.get("schema") != _ORACLE_MANIFEST_SCHEMA or oracle_manifest.get(
        "parent_source_correction_manifest_sha256"
    ) != corrections_sha:
        raise ValueError("S1.3 v6-r2 source-map DAG binding disagrees")
    forbidden_upstream = {
        "source_correction_manifest_sha256",
        "source_map_oracle_manifest_sha256",
        "aggregate_pair_transaction_manifest_sha256",
        "p2_replay_manifest_sha256",
        "completion_sha256",
    }
    if forbidden_upstream & _all_mapping_keys(component):
        raise ValueError("S1.3 v6-r2 component manifest contains a reverse DAG binding")
    if {
        "source_map_oracle_manifest_sha256",
        "aggregate_pair_transaction_manifest_sha256",
        "p2_replay_manifest_sha256",
        "completion_sha256",
    } & _all_mapping_keys(corrections):
        raise ValueError("S1.3 v6-r2 source correction manifest contains a reverse DAG binding")
    field_table = _field_table(corrections)
    if application_state == "none" and field_table:
        raise ValueError("S1.3 v6-r2 no-op state has correction fields")
    if application_state in {"partial", "complete"} and not field_table:
        raise ValueError("S1.3 v6-r2 applied state has no correction fields")
    segment_authorities = _segment_authorities(
        root, component, config_sha256=str(completion["config_sha256"])
    )
    segment_support = _segment_support_authorities(component)
    correction_asset_authorities = _correction_asset_authorities(root, corrections)
    correction_semantics = _verify_correction_asset_semantics(
        root, corrections, field_table, segment_support
    )
    for row in field_table.values():
        if row["segment_transaction_sha256"] not in segment_authorities:
            raise ValueError("S1.3 v6-r2 field table references an unknown segment transaction")
        if row["source_correction_asset_sha256"] not in correction_asset_authorities:
            raise ValueError("S1.3 v6-r2 field table references an unknown correction asset")
    oracles, base_oracles, raw_sizes = _load_oracles(root, oracle_manifest)
    map_semantics = _verify_base_final_map_semantics(
        oracles, base_oracles, raw_sizes, field_table
    )
    correction_sources = corrections.get("sources")
    if not isinstance(correction_sources, Sequence) or isinstance(correction_sources, (str, bytes)):
        raise ValueError("S1.3 v6-r2 source correction sources are invalid")
    bound_sources: set[int] = set()
    for row in correction_sources:
        if not isinstance(row, Mapping) or not isinstance(row.get("source_index"), int):
            raise ValueError("S1.3 v6-r2 source correction source row is invalid")
        source_index = int(row["source_index"])
        oracle = oracles.get(source_index)
        base_oracle = base_oracles.get(source_index)
        if (
            oracle is None
            or base_oracle is None
            or row.get("source_map_oracle_sha256") != oracle.oracle_sha256
            or row.get("base_source_map_oracle_sha256")
            != base_oracle.oracle_sha256
        ):
            raise ValueError("S1.3 v6-r2 source correction/oracle binding disagrees")
        bound_sources.add(source_index)
    if bound_sources != set(oracles):
        raise ValueError("S1.3 v6-r2 source correction manifest does not bind every oracle")
    authority = {
        "component_transaction_manifest_sha256": component_sha,
        "source_correction_manifest_sha256": corrections_sha,
        "source_map_oracle_manifest_sha256": oracle_manifest_sha,
    }
    pairs = _verify_pair_transactions(root, aggregate, authority, segment_authorities)
    _verify_replay(
        root,
        replay,
        pairs,
        oracles,
        authority,
        aggregate_sha,
        field_table,
        segment_authorities,
    )
    topology_semantics = _verify_provenance(root, oracles, field_table)
    hard_audit = _json(root / "hard_audit.json", "hard audit")
    component_hard = hard_audit.get("component_chain_c2e")
    if not isinstance(component_hard, Mapping):
        raise ValueError("S1.3 v6-r2 component hard audit authority is missing")
    required_hard_values = {
        "passed": True,
        "correction_arrays_finite": True,
        "correction_domain_valid": True,
        "resolved_field_overlap_pixel_count": 0,
        "p2_full_resolution_render_count": 2,
        "extra_full_resolution_render_count": 0,
        "formal_raw_rgb_remap_invocations": 2 * len(oracles),
        "owner_valid_topology_unchanged": True,
        "base_geometry_provenance_valid": True,
        "secondary_provenance_unchanged": True,
        "seam_topology_valid": True,
    }
    if any(component_hard.get(key) != value for key, value in required_hard_values.items()):
        raise ValueError("S1.3 v6-r2 component hard audit semantic authority disagrees")
    if component_hard.get("repair_complete") is not obligation_coverage["repair_complete"]:
        raise ValueError("S1.3 v6-r2 component hard audit repair authority disagrees")
    hard_coverage = component_hard.get("obligation_coverage")
    if not isinstance(hard_coverage, Mapping) or any(
        hard_coverage.get(key) != value for key, value in obligation_coverage.items()
    ):
        raise ValueError("S1.3 v6-r2 component hard audit coverage disagrees")
    if abs(float(component_hard.get("maximum_component_offset_px", math.inf)) - float(
        correction_semantics["maximum_component_offset_px"]
    )) > 1e-6:
        raise ValueError("S1.3 v6-r2 component hard audit offset disagrees")
    for name, measured in map_semantics.items():
        recorded = component_hard.get(name)
        if measured is None:
            if recorded is not None:
                raise ValueError("S1.3 v6-r2 component hard audit map authority disagrees")
        elif not isinstance(recorded, (int, float)) or abs(
            float(recorded) - float(measured)
        ) > 1e-6:
            raise ValueError("S1.3 v6-r2 component hard audit map authority disagrees")
    for name in ("owner_valid_topology_unchanged", "seam_topology_valid"):
        if component_hard.get(name) is not topology_semantics[name]:
            raise ValueError("S1.3 v6-r2 component hard audit topology disagrees")
    return {
        "schema": "gemini305-video-s13-v6-r2-semantic-verification/v1",
        "passed": True,
        "pair_count": len(pairs),
        "source_count": len(oracles),
        "field_count": len(field_table),
        "obligation_coverage": dict(obligation_coverage),
        "map_semantics": map_semantics,
        "topology_semantics": topology_semantics,
        "manifest_sha256": explicit,
    }


__all__ = [
    "canonical_pair_transaction_sha256",
    "canonical_source_map_slice_sha256",
    "component_decision_stable_payload",
    "component_decision_stable_sha256",
    "segment_decision_stable_payload",
    "segment_decision_stable_sha256",
    "verify_s13_v6_r1_noop_exact_comparison",
    "verify_s13_v6_r2_p2",
]
