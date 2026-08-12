from __future__ import annotations

import json
from pathlib import Path

import pytest

from panorama_demo.video_s13_m61_summary import (
    ACCEPTANCE_SCHEMA,
    BranchInputs,
    parse_branch_spec,
    summarize_acceptance,
)


BRANCHES = ("fast_direct", "fast_ignore_pose", "slow_direct", "slow_ignore_pose")


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _p3(root: Path) -> None:
    root.mkdir(parents=True)
    for name, value in (
        ("visual_panorama.png", b"png"), ("p3_pixel_provenance.npz", b"npz"),
    ):
        (root / name).write_bytes(value)
    _write_json(root / "photometric_solution.json", {
        "selected_model": "Q0_identity", "components": [{"component_index": 0}],
    })
    tx = root / "blend_transactions/pair_0000.json"
    _write_json(tx, {"model": "B0_owner_only"})
    _write_json(root / "blend_transactions.json", {
        "pairs": [{"pair_index": 0, "asset": "blend_transactions/pair_0000.json"}],
    })
    audit = {"schema": "gemini305-video-s13-p3-hard-audit/v2", "passed": True}
    _write_json(root / "hard_audit.json", audit)
    _write_json(root / "performance.json", {
        "forbidden_invocations": {"q4": 0}, "formal_raw_rgb_remap_invocations": 2,
        "maximum_resident_source_rois": 2, "peak_live_array_bytes": 100,
    })
    _write_json(root / "P3_completion.json", {
        "schema": "gemini305-video-s13-p3-visual-completion/v2",
        "hard_audit_sha256": _sha(root / "hard_audit.json"),
        "effective_config_sha256": "a" * 64,
    })
    _write_json(root.parent / "validation_m61/quality_report.json", {
        "quality_state": "clean_noop", "visible_seam_count": 0,
    })


def _inputs(tmp_path: Path) -> list[BranchInputs]:
    result = []
    for branch in BRANCHES:
        base = tmp_path / branch
        formal, rerun, old, p2 = (base / name for name in ("formal/P3", "rerun/P3", "old/P3", "P2"))
        _p3(formal)
        _p3(rerun)
        old.mkdir(parents=True)
        (old / "visual_panorama.png").write_bytes(b"old")
        p2.mkdir()
        (p2 / "geometry_and_seam_panorama_owner_only.png").write_bytes(b"p2")
        _write_json(p2 / "P2_completion.json", {
            "schema": "gemini305-video-s13-p2-completion/v4",
            "canonical_p2_regression": {"image": {"equal": True}},
        })
        result.append(BranchInputs(branch, formal, rerun, old, p2))
    return result


def test_summary_writes_only_three_assets_to_new_directory(tmp_path: Path) -> None:
    output = tmp_path / "summary"
    written = summarize_acceptance(_inputs(tmp_path), output)
    assert {path.name for path in written} == {
        "acceptance_summary.json", "reproducibility.json", "new_old_seam_atlas_index.json",
    }
    summary = json.loads((output / "acceptance_summary.json").read_text())
    assert summary["schema"] == ACCEPTANCE_SCHEMA
    assert summary["all_reproducible"] is True
    assert len(summary["branches"]) == 4
    assert summary["branches"][0]["blend_model_counts"] == {"B0_owner_only": 1}
    assert "stage_exit" in summary["section_27_report_skeleton"]
    with pytest.raises(FileExistsError, match="new absent"):
        summarize_acceptance(_inputs(tmp_path / "again"), output)


def test_summary_detects_nonreproducible_pixel_asset(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    (inputs[0].rerun / "visual_panorama.png").write_bytes(b"changed")
    summarize_acceptance(inputs, tmp_path / "summary")
    repro = json.loads((tmp_path / "summary/reproducibility.json").read_text())
    assert repro["passed"] is False
    assert repro["branches"][0]["asset_matches"]["visual_panorama.png"] is False


def test_summary_rejects_failed_audit_or_forbidden_invocation(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    _write_json(inputs[0].formal / "performance.json", {"forbidden_invocations": {"q4": 1}})
    with pytest.raises(ValueError, match="forbidden"):
        summarize_acceptance(inputs, tmp_path / "summary")


def test_parse_branch_spec_supports_relative_and_windows_absolute_paths() -> None:
    relative = parse_branch_spec("fast_direct=a:b:c:d")
    assert relative.branch == "fast_direct"
    windows = parse_branch_spec(
        r"fast_direct=D:\formal:D:\rerun:D:\old:D:\p2"
    )
    assert windows.branch == "fast_direct"
    assert len((windows.formal, windows.rerun, windows.old, windows.p2)) == 4
