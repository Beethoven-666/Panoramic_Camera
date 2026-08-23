"""Write compact S013 M5.1-r4 validation and reproducibility summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

try:
    from scripts.verify_s13_m51_r4_reproducible_seal import BRANCHES, ROUNDS, verify
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    from verify_s13_m51_r4_reproducible_seal import BRANCHES, ROUNDS, verify


SUMMARY_SCHEMA = "gemini305-video-s13-m51-r4-acceptance-summary/v1"


def _write(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid validation JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected validation JSON object: {path}")
    return value


def _find_boolean(value: object, key: str) -> bool | None:
    if isinstance(value, Mapping):
        if key in value and isinstance(value[key], bool):
            return bool(value[key])
        for child in value.values():
            found = _find_boolean(child, key)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find_boolean(child, key)
            if found is not None:
                return found
    return None


def _box_target_state(cell: Mapping[str, object]) -> bool | None:
    audit = cell.get("box_asset_audit")
    if not isinstance(audit, Mapping) or audit.get("required") is not True:
        return None
    root = Path(str(audit["root"]))
    for name in ("component_chain_audit.json", "pair_0068_0078_metrics.json"):
        value = _read(root / name)
        found = _find_boolean(value, "box_target_repair_complete")
        if found is not None:
            return found
    return None


def summarize_acceptance(root: Path, output: Path) -> tuple[Path, ...]:
    """Verify an acceptance root and write three compact, non-generation assets."""

    root = root.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"summary output must be new absent directory: {output}")
    seal = verify(root)
    cells = seal["cells"]
    if not isinstance(cells, list):
        raise ValueError("seal cell matrix is invalid")
    branches: list[dict[str, object]] = []
    for branch in BRANCHES:
        branch_cells = [cell for cell in cells if cell.get("branch") == branch]
        if len(branch_cells) != len(ROUNDS):
            raise ValueError(f"acceptance matrix is incomplete for {branch}")
        branches.append(
            {
                "branch": branch,
                "rounds": [
                    {
                        "round": cell["round"],
                        "generation_id": cell["generation_id"],
                        "completion_sha256": cell["completion_sha256"],
                        "application_state": cell["application_state"],
                        "repair_complete": cell["repair_complete"],
                        "box_target_repair_complete": _box_target_state(cell),
                    }
                    for cell in branch_cells
                ],
                "exact_reproducibility_passed": seal["exact_comparison"]["branches"][
                    branch
                ]["passed"],
            }
        )
    required_box_cells = [
        cell
        for cell in cells
        if isinstance(cell.get("box_asset_audit"), Mapping)
        and cell["box_asset_audit"].get("required") is True
    ]
    summary = {
        "schema": SUMMARY_SCHEMA,
        "diagnostic_only": True,
        "production_eligible": False,
        "matrix": {"branch_count": 4, "round_count": 2, "cell_count": 8},
        "all_p2_hard_passed": seal["run_specific_lineage"]["passed"],
        "all_exact_reproducibility_passed": seal["exact_comparison"]["passed"],
        "all_required_v6_r1_noop_exact_comparisons_passed": seal[
            "v6_r1_noop_exact_comparison"
        ]["passed"],
        "all_box_assets_present": all(
            cell["box_asset_audit"]["passed"] for cell in required_box_cells
        ),
        "branches": branches,
        "paired_timing_gate": {
            key: seal["timing_comparison"]["paired_summary"][key]
            for key in (
                "pair_count",
                "median_delta_seconds",
                "maximum_delta_seconds",
                "p95_delta_seconds",
                "median_limit_seconds",
                "maximum_limit_seconds",
                "median_passed",
                "maximum_passed",
            )
        },
        "claim_policy": (
            "box target is complete only when both fast_direct rounds explicitly report "
            "box_target_repair_complete=true"
        ),
    }
    output.mkdir(parents=True)
    paths = (
        output / "acceptance_summary.json",
        output / "paired_timing_summary.json",
        output / "reproducible_seal.json",
    )
    _write(paths[0], summary)
    _write(paths[1], seal["timing_comparison"]["paired_summary"])
    _write(paths[2], seal)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    summarize_acceptance(args.root, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
