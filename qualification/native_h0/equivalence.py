"""Exact native SDK/CLI verification against the installed frozen verifier."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_native_equivalence(sdk_root: Path, cli_root: Path, *, session_root: Path,
                              require_standard_quality: bool = False) -> dict:
    """Retain canonical stage checks and also compare full seam/provenance records."""
    from panorama_demo.sdk_acceptance import verify_2d_equivalence
    import numpy as np

    roots = [Path(sdk_root).resolve(), Path(cli_root).resolve()]
    if roots[0] == roots[1]:
        raise ValueError("SDK/CLI self-comparison is forbidden")
    expected = {key: _sha(Path(session_root) / name) for key, name in (
        ("manifest", "manifest.json"), ("calibration", "calibration.json"),
        ("frames_csv", "frames.csv"))}
    reports = []
    for root in roots:
        if (root / "emergency_delivery.json").exists():
            raise ValueError("Normal H0 cannot accept emergency output")
        report = json.loads((root / "video_report.json").read_text(encoding="utf-8"))
        if report.get("input_sha256") != expected:
            raise ValueError("Output does not bind to this captured session")
        if require_standard_quality and report.get("grades", {}).get("overall") not in {"A", "B"}:
            raise ValueError("Normal H0 quality must be A or B")
        if require_standard_quality and report.get("manual_review_required") is not False:
            raise ValueError("Normal H0 requires manual_review_required=false")
        reports.append(report)
    result = verify_2d_equivalence(roots[0], roots[1])
    # The installed comparator intentionally canonicalizes only selected seam
    # fields. H0 must additionally compare every recorded transaction field.
    for field in ("stage_pixel_sha256", "source_frame_ids", "schedule",
                  "pair_seam_decisions", "m6_decisions"):
        if field not in reports[0] or reports[0][field] != reports[1].get(field):
            raise ValueError("Exact H0 evidence mismatch: " + field)
    with np.load(roots[0] / "video_pixel_provenance.npz", allow_pickle=False) as left:
        with np.load(roots[1] / "video_pixel_provenance.npz", allow_pickle=False) as right:
            if set(left.files) != set(right.files):
                raise ValueError("Exact H0 provenance key mismatch")
            for name in left.files:
                a, b = left[name], right[name]
                if a.dtype != b.dtype or a.shape != b.shape or a.tobytes() != b.tobytes():
                    raise ValueError("Exact H0 provenance mismatch: " + name)
    return {"schema": "gemini305-native-h0-equivalence/v1", "status": "PASS",
            "session": str(Path(session_root).resolve()), "canonical": result,
            "full_seam_transactions_exact": True, "all_provenance_exact": True}
