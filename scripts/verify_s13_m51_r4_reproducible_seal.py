"""Verify the two-round, four-branch S013 M5.1-r4 P2-only seal."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


BRANCHES = ("fast_direct", "fast_ignore_pose", "slow_direct", "slow_ignore_pose")
ROUNDS = ("round_a", "round_b")
P2_SCHEMA = "gemini305-video-s13-p2-completion/v6-r2"
SEAL_SCHEMA = "gemini305-video-s13-m51-r4-reproducible-seal/v1"
NORMALIZATION_SCHEMA = "gemini305-video-s13-reproducibility-normalization/v1"
NORMALIZATION_POINTERS = (
    "/generation_id",
    "/generation_manifest_sha256",
    "/parent_completion_sha256",
    "/created_at_utc",
    "/output_root",
    "/timing",
    "/performance",
)
BOX_ASSETS = (
    "current_roi.png",
    "candidate_roi.png",
    "current_vs_candidate.png",
    "edge_trace_overlay.png",
    "component_masks_overlay.png",
    "pair_0068_0078_metrics.json",
    "component_chain_audit.json",
    "source_offsets.json",
    "forward_reverse_hypotheses.json",
)
MANIFESTS = {
    "component_chain": "component_chain_transactions/manifest.json",
    "source_corrections": "source_corrections/manifest.json",
    "source_maps": "source_maps/manifest.json",
    "pair_transactions": "pair_transactions.json",
    "replay": "p2_replay_manifest.json",
}
MANIFEST_SCHEMAS = {
    "component_chain": "gemini305-video-s13-component-chain-transactions/v1",
    "source_corrections": "gemini305-video-s13-source-corrections/v1",
    "source_maps": "gemini305-video-s13-source-map-oracles/v1",
    "pair_transactions": "gemini305-video-s13-m5-pair-transactions/v4",
    "replay": "gemini305-video-s13-p2-replay/v2",
}


def _read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid or missing JSON asset: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise ValueError(f"missing required seal asset: {path}") from exc
    return digest.hexdigest()


def _sha_json(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _decode_pointer_token(token: str) -> str:
    return token.replace("~1", "/").replace("~0", "~")


def _remove_pointer(value: object, pointer: str) -> None:
    if not pointer.startswith("/"):
        raise ValueError(f"normalization pointer must be absolute: {pointer}")
    tokens = [_decode_pointer_token(token) for token in pointer[1:].split("/")]
    parent: object = value
    for token in tokens[:-1]:
        if isinstance(parent, dict):
            if token not in parent:
                return
            parent = parent[token]
        elif isinstance(parent, list):
            try:
                parent = parent[int(token)]
            except (ValueError, IndexError):
                return
        else:
            return
    last = tokens[-1]
    if isinstance(parent, dict):
        parent.pop(last, None)
    elif isinstance(parent, list):
        try:
            del parent[int(last)]
        except (ValueError, IndexError):
            return


def normalize_json(
    value: Mapping[str, object],
    pointers: Sequence[str] = NORMALIZATION_POINTERS,
) -> dict[str, object]:
    """Remove only the frozen JSON Pointer allowlist from a payload copy."""

    if tuple(pointers) != NORMALIZATION_POINTERS:
        raise ValueError("S013 r4 normalization pointer allowlist is frozen")
    result = copy.deepcopy(dict(value))
    for pointer in pointers:
        _remove_pointer(result, pointer)
    return result


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot compute percentile of an empty sequence")
    position = (len(ordered) - 1) * percentile
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    fraction = position - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


def summarize_paired_timings(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Summarize paired v6-r1/v6-r2 timings using the frozen performance gates."""

    if len(rows) < 5:
        raise ValueError("paired timing audit requires at least five pairs")
    normalized_rows: list[dict[str, object]] = []
    deltas: list[float] = []
    baselines: list[float] = []
    allowed_orders = {"v6-r1_then_v6-r2", "v6-r2_then_v6-r1"}
    for index, row in enumerate(rows):
        order = str(row.get("order", ""))
        if order not in allowed_orders:
            raise ValueError(f"paired timing row {index} has an invalid order")
        baseline = float(row.get("baseline_seconds", math.nan))
        candidate = float(row.get("candidate_seconds", math.nan))
        if not math.isfinite(baseline) or not math.isfinite(candidate):
            raise ValueError(f"paired timing row {index} is not finite")
        if baseline <= 0.0 or candidate <= 0.0:
            raise ValueError(f"paired timing row {index} must be positive")
        delta = candidate - baseline
        baselines.append(baseline)
        deltas.append(delta)
        normalized_rows.append(
            {
                "pair_id": str(row.get("pair_id", index)),
                "order": order,
                "baseline_seconds": baseline,
                "candidate_seconds": candidate,
                "delta_seconds": delta,
            }
        )
    observed_orders = {str(row["order"]) for row in normalized_rows}
    if observed_orders != allowed_orders:
        raise ValueError("paired timing audit must interleave both execution orders")
    baseline_median = statistics.median(baselines)
    median_delta = statistics.median(deltas)
    maximum_delta = max(deltas)
    median_limit = max(1.0, baseline_median * 0.05)
    return {
        "schema": "gemini305-video-s13-m51-r4-paired-timing-summary/v1",
        "pair_count": len(rows),
        "rows": normalized_rows,
        "baseline_median_seconds": baseline_median,
        "median_delta_seconds": median_delta,
        "maximum_delta_seconds": maximum_delta,
        "p95_delta_seconds": _percentile(deltas, 0.95),
        "median_limit_seconds": median_limit,
        "maximum_limit_seconds": 2.0,
        "median_passed": median_delta <= median_limit,
        "maximum_passed": maximum_delta <= 2.0,
    }


def _safe_generation(run: Path, pointer: Mapping[str, object]) -> Path:
    relative = Path(str(pointer.get("generation", "")))
    if not relative.parts or relative.is_absolute():
        raise ValueError(f"unsafe generation pointer: {run}")
    generation = (run / relative).resolve()
    try:
        generation.relative_to(run.resolve())
    except ValueError as exc:
        raise ValueError(f"generation pointer escapes run root: {run}") from exc
    return generation


def _npz_content(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            return {name: np.array(archive[name], copy=True) for name in archive.files}
    except (OSError, ValueError) as exc:
        raise ValueError(f"invalid NPZ asset: {path}") from exc


def _npz_equal(
    left: Path, right: Path, *, ignored_names: frozenset[str] = frozenset()
) -> bool:
    a = _npz_content(left)
    b = _npz_content(right)
    names_a = set(a) - set(ignored_names)
    names_b = set(b) - set(ignored_names)
    return names_a == names_b and all(
        np.array_equal(
            a[key],
            b[key],
            equal_nan=(a[key].dtype.kind in {"f", "c"}),
        )
        for key in names_a
    )


def _npz_tree_equal(
    left: Path,
    right: Path,
    pattern: str,
    *,
    ignored_names: frozenset[str] = frozenset(),
) -> bool:
    left_files = {path.relative_to(left).as_posix(): path for path in left.glob(pattern)}
    right_files = {path.relative_to(right).as_posix(): path for path in right.glob(pattern)}
    return left_files.keys() == right_files.keys() and all(
        _npz_equal(
            left_files[name], right_files[name], ignored_names=ignored_names
        )
        for name in left_files
    )


def _box_asset_audit(run: Path, p2: Path, *, required: bool) -> dict[str, object]:
    roots = (run / "validation/box_component_chain", p2 / "validation/box_component_chain")
    root = next((candidate for candidate in roots if candidate.is_dir()), roots[0])
    missing = [name for name in BOX_ASSETS if not (root / name).is_file()]
    passed = not missing if required else True
    return {
        "required": required,
        "root": str(root),
        "required_assets": list(BOX_ASSETS),
        "missing_assets": missing,
        "passed": passed,
        "assets_sha256": {
            name: _sha_file(root / name)
            for name in BOX_ASSETS
            if (root / name).is_file()
        },
    }


def _verify_completion_assets(
    p2: Path,
    completion: Mapping[str, object],
    required: Sequence[str],
) -> None:
    assets = completion.get("assets_sha256")
    if not isinstance(assets, Mapping):
        raise ValueError(f"P2 completion assets_sha256 is missing: {p2}")
    for relative in required:
        expected = assets.get(relative)
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError(f"P2 completion does not bind required asset: {relative}")
    for name, expected in assets.items():
        relative = Path(str(name))
        if relative.is_absolute() or not relative.parts:
            raise ValueError(f"unsafe completion asset path: {name}")
        path = (p2 / relative).resolve()
        try:
            path.relative_to(p2.resolve())
        except ValueError as exc:
            raise ValueError(f"completion asset escapes P2: {name}") from exc
        if not isinstance(expected, str) or _sha_file(path) != expected:
            raise ValueError(f"P2 completion asset hash mismatch: {name}")


def _verify_optional_explicit_binding(
    completion: Mapping[str, object],
    names: Sequence[str],
    expected: str,
) -> None:
    for name in names:
        if name in completion and completion[name] != expected:
            raise ValueError(f"P2 completion explicit lineage hash mismatch: {name}")


def _integer_labels(path: Path, names: Sequence[str]) -> set[int]:
    arrays = _npz_content(path)
    labels: set[int] = set()
    for name in names:
        if name not in arrays:
            continue
        values = arrays[name]
        if values.dtype.kind not in {"i", "u"}:
            raise ValueError(f"correction authority label must be integer: {path}:{name}")
        labels.update(int(value) for value in np.unique(values) if int(value) >= 0)
    return labels


def _manifest_asset_rows(
    p2: Path,
    directory: str,
    rows: object,
) -> list[Mapping[str, object]]:
    if not isinstance(rows, list) or not all(isinstance(row, Mapping) for row in rows):
        raise ValueError(f"accepted correction authority rows are invalid: {directory}")
    result: list[Mapping[str, object]] = []
    for row in rows:
        asset_value = row.get("asset", row.get("correction_asset"))
        hash_value = row.get("asset_sha256", row.get("correction_asset_sha256"))
        if asset_value is None and directory == "source_corrections":
            continue
        relative = Path(str(asset_value or ""))
        if relative.is_absolute() or not relative.parts or ".." in relative.parts:
            raise ValueError(f"accepted correction authority asset is unsafe: {relative}")
        path = (
            p2 / relative
            if relative.parts[0] == directory
            else p2 / directory / relative
        )
        expected = hash_value
        if not isinstance(expected, str) or _sha_file(path) != expected:
            raise ValueError(f"accepted correction authority asset hash mismatch: {path}")
        result.append(row)
    return result


def _verify_r4_semantics(
    p2: Path,
    completion: Mapping[str, object],
    manifests: Mapping[str, Mapping[str, object]],
    payloads: Mapping[str, Mapping[str, object]],
) -> None:
    expected_hashes = {name: str(value["sha256"]) for name, value in manifests.items()}
    required_completion_bindings = {
        "component_transaction_manifest_sha256": expected_hashes["component_chain"],
        "source_correction_manifest_sha256": expected_hashes["source_corrections"],
        "source_map_oracle_manifest_sha256": expected_hashes["source_maps"],
        "aggregate_pair_transaction_manifest_sha256": expected_hashes["pair_transactions"],
        "p2_replay_manifest_sha256": expected_hashes["replay"],
    }
    for name, expected in required_completion_bindings.items():
        if completion.get(name) != expected:
            raise ValueError(f"P2 completion explicit lineage binding is missing or invalid: {name}")
    if completion.get("provenance_schema") != "gemini305-video-s13-p2-provenance/v6-r2":
        raise ValueError("P2 completion explicit lineage provenance schema is invalid")

    component = payloads["component_chain"]
    correction = payloads["source_corrections"]
    source_maps = payloads["source_maps"]
    pairs = payloads["pair_transactions"]
    replay = payloads["replay"]
    if correction.get("parent_component_transaction_manifest_sha256") != expected_hashes[
        "component_chain"
    ]:
        raise ValueError("source correction manifest DAG parent is invalid")
    if source_maps.get("parent_source_correction_manifest_sha256") != expected_hashes[
        "source_corrections"
    ]:
        raise ValueError("source-map manifest DAG parent is invalid")
    dag_bindings = {
        "component_transaction_manifest_sha256": expected_hashes["component_chain"],
        "source_correction_manifest_sha256": expected_hashes["source_corrections"],
        "source_map_oracle_manifest_sha256": expected_hashes["source_maps"],
    }
    for document_name, document in (("pair", pairs), ("replay", replay)):
        for key, expected in dag_bindings.items():
            if document.get(key) != expected:
                raise ValueError(f"{document_name} manifest DAG binding is invalid: {key}")
    if replay.get("aggregate_pair_transaction_manifest_sha256") != expected_hashes[
        "pair_transactions"
    ]:
        raise ValueError("replay manifest DAG pair parent is invalid")
    forbidden_upstream = {
        "aggregate_pair_transaction_manifest_sha256",
        "p2_replay_manifest_sha256",
        "completion_sha256",
    }
    for document_name, document in (
        ("component", component),
        ("source correction", correction),
        ("source maps", source_maps),
    ):
        if forbidden_upstream.intersection(document):
            raise ValueError(f"{document_name} manifest DAG contains a reverse lineage edge")

    states = {
        str(component.get("application_state")),
        str(correction.get("application_state")),
        str(completion.get("application_state")),
    }
    if len(states) != 1 or next(iter(states)) not in {"complete", "partial", "none"}:
        raise ValueError("component correction application_state lineage is inconsistent")
    application_state = next(iter(states))
    accepted = component.get("accepted_segment_ids")
    applied = completion.get("applied_segment_ids")
    improved = completion.get("improved_segment_ids")
    resolved = completion.get("resolved_segment_ids")
    if not all(isinstance(value, list) for value in (accepted, applied, improved, resolved)):
        raise ValueError("accepted correction authority segment lists are invalid")
    accepted_ids = {str(value) for value in accepted}
    if accepted_ids != {str(value) for value in applied}:
        raise ValueError("accepted correction authority segment IDs disagree with completion")
    if accepted_ids != ({str(value) for value in improved} | {str(value) for value in resolved}):
        raise ValueError("accepted correction authority quality-state coverage is incomplete")
    if application_state == "none" and accepted_ids:
        raise ValueError("accepted correction authority exists for application_state=none")
    if application_state in {"partial", "complete"} and not accepted_ids:
        raise ValueError("accepted correction authority is missing for applied P2")

    correction_rows = _manifest_asset_rows(
        p2, "source_corrections", correction.get("sources")
    )
    map_rows = _manifest_asset_rows(p2, "source_maps", source_maps.get("sources"))
    field_table = correction.get("field_id_table")
    contributors = correction.get("contributors")
    if not isinstance(field_table, list) or not all(isinstance(row, Mapping) for row in field_table):
        raise ValueError("accepted correction authority field table is invalid")
    if not isinstance(contributors, list):
        raise ValueError("accepted correction authority contributors are invalid")
    field_ids = [int(row.get("field_id", -1)) for row in field_table]
    if len(field_ids) != len(set(field_ids)) or any(value < 0 for value in field_ids):
        raise ValueError("accepted correction authority field IDs are invalid or duplicated")
    table_ids = set(field_ids)
    table_segments = {str(row.get("segment_id")) for row in field_table}
    if table_segments != accepted_ids or {str(value) for value in contributors} != accepted_ids:
        raise ValueError("accepted correction authority field table does not cover accepted segments")
    correction_labels: set[int] = set()
    for row in correction_rows:
        correction_relative = Path(str(row.get("asset", row.get("correction_asset"))))
        correction_labels |= _integer_labels(
            (
                p2 / correction_relative
                if correction_relative.parts[0] == "source_corrections"
                else p2 / "source_corrections" / correction_relative
            ),
            ("field_id",),
        )
    map_labels: set[int] = set()
    for row in map_rows:
        map_relative = Path(str(row["asset"]))
        map_labels |= _integer_labels(
            p2 / map_relative
            if map_relative.parts[0] == "source_maps"
            else p2 / "source_maps" / map_relative,
            ("field_id",),
        )
    replay_labels: set[int] = set()
    replay_segments: set[str] = set()
    replay_rows = replay.get("pairs")
    if not isinstance(replay_rows, list) or not all(isinstance(row, Mapping) for row in replay_rows):
        raise ValueError("accepted replay authority rows are invalid")
    for row in replay_rows:
        relative = Path(str(row.get("asset", "")))
        path = p2 / relative
        replay_labels |= _integer_labels(
            path,
            (
                "left_component_correction_field_id",
                "right_component_correction_field_id",
            ),
        )
        relevant = row.get("relevant_segment_ids")
        if not isinstance(relevant, list):
            raise ValueError("accepted replay authority relevant segment list is missing")
        replay_segments.update(str(value) for value in relevant)
    provenance_labels = _integer_labels(
        p2 / "p2_pixel_provenance.npz", ("component_correction_field_id",)
    )
    for label_name, labels in (
        ("correction", correction_labels),
        ("source-map", map_labels),
        ("replay", replay_labels),
        ("provenance", provenance_labels),
    ):
        if labels != table_ids:
            raise ValueError(
                f"accepted correction authority {label_name} field IDs do not match field table"
            )
    if replay_segments != accepted_ids:
        raise ValueError("accepted replay authority segment coverage is incomplete")
    if int(source_maps.get("source_count", -1)) != len(map_rows):
        raise ValueError("source-map authority source_count is inconsistent")

    pair_rows = pairs.get("pairs")
    if not isinstance(pair_rows, list) or not all(isinstance(row, Mapping) for row in pair_rows):
        raise ValueError("accepted pair correction authority rows are invalid")
    referenced_segments: set[str] = set()
    referenced_fields: set[int] = set()
    for pair in pair_rows:
        authority = pair.get("component_chain_c2e")
        if not isinstance(authority, Mapping):
            raise ValueError("accepted pair correction authority is missing")
        for key, expected in dag_bindings.items():
            if authority.get(key) != expected:
                raise ValueError(f"accepted pair correction authority binding is invalid: {key}")
        refs = authority.get("refs")
        if not isinstance(refs, list):
            raise ValueError("accepted pair correction authority refs are invalid")
        for ref in refs:
            if not isinstance(ref, Mapping):
                raise ValueError("accepted pair correction authority ref is invalid")
            if str(ref.get("segment_id")) in accepted_ids:
                referenced_segments.add(str(ref["segment_id"]))
                referenced_fields.add(int(ref.get("field_id", -1)))
    if application_state != "none" and (
        referenced_segments != accepted_ids or referenced_fields != table_ids
    ):
        raise ValueError("accepted pair/replay correction authority coverage is incomplete")


def _cell(root: Path, round_name: str, branch: str) -> dict[str, object]:
    run = root / round_name / branch
    pointer = _read(run / "current_latest.json")
    if pointer.get("stage") != "P2":
        raise ValueError(f"current_latest is not P2: {run}")
    generation = _safe_generation(run, pointer)
    generation_manifest_path = generation / "generation_manifest.json"
    generation_manifest = _read(generation_manifest_path)
    algorithm = generation_manifest.get("algorithm")
    if not isinstance(algorithm, Mapping):
        raise ValueError(f"generation algorithm binding is invalid: {generation}")
    source_commit = algorithm.get("source_commit")
    if (
        not isinstance(source_commit, str)
        or len(source_commit) != 40
        or any(character not in "0123456789abcdef" for character in source_commit.lower())
        or algorithm.get("working_tree_dirty") is not False
    ):
        raise ValueError(f"seal requires a clean committed worktree: {generation}")
    for forbidden in ("P3", "M6", "delivery.json", "video_delivery.json"):
        if (generation / forbidden).exists() or (run / forbidden).exists():
            raise ValueError(f"forbidden post-P2 output exists: {run / forbidden}")
    p2 = generation / "P2"
    completion_path = p2 / "P2_completion.json"
    completion = _read(completion_path)
    if (
        completion.get("schema") != P2_SCHEMA
        or completion.get("stage") != "P2"
        or completion.get("m6_eligible") is not False
        or completion.get("hard_audit_passed") is not True
    ):
        raise ValueError(f"P2-only v6-r2 completion contract is invalid: {p2}")
    hard_audit = _read(p2 / "hard_audit.json")
    if hard_audit.get("passed") is not True:
        raise ValueError(f"P2 hard audit did not pass: {p2}")
    manifests: dict[str, object] = {}
    manifest_payloads: dict[str, dict[str, Any]] = {}
    for name, relative in MANIFESTS.items():
        path = p2 / relative
        payload = _read(path)
        if payload.get("schema") != MANIFEST_SCHEMAS[name]:
            raise ValueError(f"invalid {name} manifest schema: {path}")
        manifest_payloads[name] = payload
        manifest_sha = _sha_file(path)
        manifests[name] = {"path": relative, "sha256": manifest_sha}
    binding_aliases = {
        "component_chain": (
            "component_transaction_manifest_sha256",
            "component_chain_transaction_manifest_sha256",
        ),
        "source_corrections": ("source_correction_manifest_sha256",),
        "source_maps": (
            "source_map_oracle_manifest_sha256",
            "source_maps_manifest_sha256",
        ),
        "pair_transactions": ("aggregate_pair_transaction_manifest_sha256",),
        "replay": ("replay_manifest_sha256", "p2_replay_manifest_sha256"),
    }
    for name, aliases in binding_aliases.items():
        manifest = manifests[name]
        if not isinstance(manifest, Mapping):
            raise ValueError("internal manifest binding audit is invalid")
        _verify_optional_explicit_binding(completion, aliases, str(manifest["sha256"]))
    typed_manifests = {
        name: value for name, value in manifests.items() if isinstance(value, Mapping)
    }
    typed_payloads = {
        name: value for name, value in manifest_payloads.items() if isinstance(value, Mapping)
    }
    if len(typed_manifests) != len(MANIFESTS) or len(typed_payloads) != len(MANIFESTS):
        raise ValueError("internal r4 manifest audit is invalid")
    _verify_r4_semantics(p2, completion, typed_manifests, typed_payloads)
    performance = _read(p2 / "performance.json")
    if (
        int(performance.get("p2_full_resolution_render_count", -1)) != 2
        or int(performance.get("extra_full_resolution_render_count", -1)) != 0
        or int(performance.get("backward_pyr_lk_call_count", -1)) != 0
    ):
        raise ValueError(f"formal render/LK performance counters are invalid: {p2}")
    required_files = (
        "geometry_and_seam_panorama_owner_only.png",
        "p2_seams.npz",
        "p2_pixel_provenance.npz",
    )
    assets = {name: _sha_file(p2 / name) for name in required_files}
    required_completion_assets = [
        *required_files,
        "hard_audit.json",
        "performance.json",
        *MANIFESTS.values(),
    ]
    required_completion_assets.extend(
        path.relative_to(p2).as_posix()
        for pattern in (
            "component_chain_transactions/*.npz",
            "source_corrections/*.npz",
            "source_maps/*.npz",
            "pair_replay/*.npz",
        )
        for path in p2.glob(pattern)
    )
    _verify_completion_assets(p2, completion, required_completion_assets)
    generation_sha = _sha_file(generation_manifest_path)
    if completion.get("generation_manifest_sha256") != generation_sha:
        raise ValueError("P2 completion explicit lineage generation manifest binding is invalid")
    explicit_identity = {
        "algorithm_id": algorithm.get("algorithm_id"),
        "implementation_id": algorithm.get("implementation_id"),
        "config_sha256": algorithm.get("config_sha256"),
        "candidate_manifest_sha256": algorithm.get("candidate_manifest_sha256"),
    }
    for key, expected in explicit_identity.items():
        if not isinstance(expected, str) or completion.get(key) != expected:
            raise ValueError(f"P2 completion explicit lineage identity binding is invalid: {key}")
    if completion.get("contract_schema") != "gemini305-video-s13-output-first/v6-r2":
        raise ValueError("P2 completion explicit lineage contract schema is invalid")
    box = _box_asset_audit(run, p2, required=branch == "fast_direct")
    if not box["passed"]:
        raise ValueError(f"required box component-chain assets are missing: {run}")
    return {
        "round": round_name,
        "branch": branch,
        "run_root": str(run),
        "p2_root": str(p2),
        "generation_id": pointer.get("generation_id", completion.get("generation_id")),
        "source_commit": source_commit,
        "working_tree_dirty": False,
        "generation_manifest_sha256": generation_sha,
        "completion_sha256": _sha_file(completion_path),
        "completion": completion,
        "manifests": manifests,
        "manifest_payloads": manifest_payloads,
        "assets_sha256": assets,
        "performance_sha256": _sha_file(p2 / "performance.json"),
        "performance": performance,
        "box_asset_audit": box,
    }


def _exact_branch_comparison(left: Mapping[str, object], right: Mapping[str, object]) -> dict[str, object]:
    left_p2 = Path(str(left["p2_root"]))
    right_p2 = Path(str(right["p2_root"]))
    left_manifests = left["manifest_payloads"]
    right_manifests = right["manifest_payloads"]
    if not isinstance(left_manifests, Mapping) or not isinstance(right_manifests, Mapping):
        raise ValueError("internal manifest payload audit is invalid")
    left_component = left_manifests["component_chain"]
    right_component = right_manifests["component_chain"]
    if not isinstance(left_component, Mapping) or not isinstance(right_component, Mapping):
        raise ValueError("component transaction manifest is invalid")
    checks = {
        "png_bytes_exact": _sha_file(
            left_p2 / "geometry_and_seam_panorama_owner_only.png"
        )
        == _sha_file(right_p2 / "geometry_and_seam_panorama_owner_only.png"),
        "seam_arrays_exact": _npz_equal(left_p2 / "p2_seams.npz", right_p2 / "p2_seams.npz"),
        "provenance_arrays_exact": _npz_equal(
            left_p2 / "p2_pixel_provenance.npz",
            right_p2 / "p2_pixel_provenance.npz",
        ),
        "component_correction_arrays_exact": _npz_tree_equal(
            left_p2, right_p2, "source_corrections/*.npz"
        ),
        "source_map_arrays_exact": _npz_tree_equal(left_p2, right_p2, "source_maps/*.npz"),
        "replay_arrays_exact": _npz_tree_equal(
            left_p2,
            right_p2,
            "pair_replay/*.npz",
            ignored_names=frozenset({"parent_pair_transaction_sha256"}),
        ),
        "component_segment_arrays_exact": _npz_tree_equal(
            left_p2, right_p2, "component_chain_transactions/*.npz"
        ),
        "component_decision_payload_exact": (
            normalize_json(left_component) == normalize_json(right_component)
        ),
        "source_correction_manifest_payload_exact": (
            normalize_json(left_manifests["source_corrections"])
            == normalize_json(right_manifests["source_corrections"])
        ),
        "source_map_manifest_payload_exact": (
            normalize_json(left_manifests["source_maps"])
            == normalize_json(right_manifests["source_maps"])
        ),
        "decision_payload_stable_sha256_exact": (
            left_component.get("decision_payload_stable_sha256")
            == right_component.get("decision_payload_stable_sha256")
        ),
    }
    return {"passed": all(checks.values()), "checks": checks}


def _public_cell(cell: Mapping[str, object]) -> dict[str, object]:
    result = {
        key: cell[key]
        for key in (
            "round",
            "branch",
            "generation_id",
            "source_commit",
            "working_tree_dirty",
            "generation_manifest_sha256",
            "completion_sha256",
            "manifests",
            "assets_sha256",
            "performance_sha256",
            "box_asset_audit",
        )
    }
    completion = cell["completion"]
    if not isinstance(completion, Mapping):
        raise ValueError("internal completion audit is invalid")
    result["application_state"] = completion.get("application_state")
    result["repair_complete"] = completion.get("repair_complete")
    return result


def verify(root: Path) -> dict[str, object]:
    """Verify and return the compact reproducibility seal document."""

    root = root.resolve()
    cells = [_cell(root, round_name, branch) for round_name in ROUNDS for branch in BRANCHES]
    commits = {str(cell["source_commit"]) for cell in cells}
    if len(commits) != 1:
        raise ValueError("4x2 seal cells do not share one clean source commit")
    by_key = {(str(cell["round"]), str(cell["branch"])): cell for cell in cells}
    exact_branches: dict[str, object] = {}
    for branch in BRANCHES:
        result = _exact_branch_comparison(
            by_key[("round_a", branch)], by_key[("round_b", branch)]
        )
        exact_branches[branch] = result
        if not result["passed"]:
            failed = [name for name, passed in result["checks"].items() if not passed]
            raise ValueError(f"{branch} exact reproducibility failed: {failed}")
    timing_input = _read(root / "paired_timing_runs.json")
    if timing_input.get("schema") != "gemini305-video-s13-m51-r4-paired-timing-runs/v1":
        raise ValueError("paired timing input schema is invalid")
    timing_rows = timing_input.get("pairs")
    if not isinstance(timing_rows, list) or not all(isinstance(row, Mapping) for row in timing_rows):
        raise ValueError("paired timing input rows are invalid")
    paired_summary = summarize_paired_timings(timing_rows)
    if not paired_summary["median_passed"] or not paired_summary["maximum_passed"]:
        raise ValueError("paired timing performance gate failed")
    run_timings = {
        branch: {
            round_name: {
                "total_m5": float(
                    by_key[(round_name, branch)]["performance"].get("total_m5", math.nan)
                ),
                "component_chain_seconds": float(
                    by_key[(round_name, branch)]["performance"].get(
                        "component_chain_seconds", math.nan
                    )
                ),
            }
            for round_name in ROUNDS
        }
        for branch in BRANCHES
    }
    if not all(
        math.isfinite(value)
        for branch in run_timings.values()
        for round_value in branch.values()
        for value in round_value.values()
    ):
        raise ValueError("4x2 timing values must be finite")
    public_cells = [_public_cell(cell) for cell in cells]
    return {
        "schema": SEAL_SCHEMA,
        "sealed": True,
        "diagnostic_only": True,
        "production_eligible": False,
        "m6_eligible": False,
        "matrix": {"branches": list(BRANCHES), "rounds": list(ROUNDS), "cell_count": 8},
        "normalization": {
            "schema": NORMALIZATION_SCHEMA,
            "excluded_json_pointers": list(NORMALIZATION_POINTERS),
            "normalized_component_payload_sha256": {
                branch: _sha_json(
                    normalize_json(
                        by_key[("round_a", branch)]["manifest_payloads"]["component_chain"]
                    )
                )
                for branch in BRANCHES
            },
        },
        "cells": public_cells,
        "exact_comparison": {
            "passed": all(result["passed"] for result in exact_branches.values()),
            "branches": exact_branches,
        },
        "run_specific_lineage": {
            "passed": True,
            "policy": "each cell is schema-, asset-, and P2-hard-audit-valid; hashes may differ",
        },
        "timing_comparison": {
            "comparison_class": "tolerance",
            "run_timings": run_timings,
            "paired_summary": paired_summary,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    document = verify(args.root)
    _write(args.output, document)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
