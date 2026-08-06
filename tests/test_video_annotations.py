from __future__ import annotations

import json
from pathlib import Path

import pytest

from panorama_demo.video_annotations import (
    audit_annotation_source_progress,
    load_source_annotations,
    validate_annotation_coordinates,
    verify_annotation_source_progress,
    verify_annotation_source_progress_evidence,
    write_annotation_preview,
)
from panorama_demo.video_split import build_source_progress_evidence
from panorama_demo.video_annotations import VideoAnnotationError


ROOT = Path(__file__).resolve().parents[1]
SESSION = ROOT / "data" / "captures" / "video" / "run_20260804_162340"
ANNOTATIONS = ROOT / "benchmarks" / "run_20260804_162340" / "annotations" / "objects.json"


def test_fixed_source_annotations_cover_required_objects_lines_and_safe_regions(tmp_path):
    annotations = load_source_annotations(ANNOTATIONS)
    validate_annotation_coordinates(annotations, session_root=SESSION)
    assert len(annotations["objects"]) >= 8
    assert len(annotations["lines"]) >= 6
    assert len(annotations["safe_background"]) >= 2
    assert set(annotations["source_frames"]) == {"249", "257"}
    preview = write_annotation_preview(
        annotations, session_root=SESSION, output=tmp_path / "annotation_preview.png"
    )
    assert preview.is_file()
    for index_name, expected_kind in (
        ("lines.json", "lines"),
        ("safe_background.json", "safe_background"),
    ):
        index = json.loads((ANNOTATIONS.parent / index_name).read_text(encoding="utf-8"))
        assert index["kind"] == expected_kind
        assert index["canonical_source"] == "objects.json"
        assert all(item["id"] in index["entry_ids"] for item in annotations[expected_kind])


def test_fixed_annotation_source_progress_requires_explicit_matching_real_mapping():
    annotations = load_source_annotations(ANNOTATIONS)
    mapping = {
        int(frame_id): float(descriptor["scan_progress"])
        for frame_id, descriptor in annotations["source_frames"].items()
    }
    audit = verify_annotation_source_progress(annotations, mapping)
    assert audit["verified"] is True
    mapping[249] += 0.001
    audit = audit_annotation_source_progress(annotations, mapping)
    assert audit["verified"] is False
    assert audit["mismatched_frame_ids"] == [249]
    with pytest.raises(VideoAnnotationError, match="scan_progress mismatch"):
        verify_annotation_source_progress(annotations, mapping)


def test_annotation_progress_evidence_rejects_invalid_or_mismatched_real_map():
    annotations = load_source_annotations(ANNOTATIONS)

    class Motion:
        def __init__(self, dx: float):
            self.dx, self.dy, self.reliable = dx, 0.0, True

    evidence = build_source_progress_evidence(
        (63, 249, 257, 500),
        [Motion(0.3908519084978398), Motion(0.0125236775216051), Motion(0.5966244139805551)],
    )
    audit = verify_annotation_source_progress_evidence(annotations, evidence)
    assert audit["verified"] is True
    assert len(audit["source_progress_evidence_sha256"]) == 64
    evidence["schema"] = "wrong"
    with pytest.raises(VideoAnnotationError, match="Invalid real source progress evidence"):
        verify_annotation_source_progress_evidence(annotations, evidence)


def test_optional_measurement_group_is_validated_without_mutating_fixed_annotations(tmp_path):
    payload = {
        "schema": "gemini305-video-source-annotations/v1",
        "source_frames": {"7": {"color_path": "color/00000007.jpg", "scan_progress": 0.0}},
        "objects": [{"id": "object", "frame_id": 7, "measurement_group": "paired_object", "polygon": [[0, 0], [1, 0], [1, 1]]}],
        "lines": [{"id": "line", "frame_id": 7, "measurement_group": "paired_line", "points": [[0, 0], [1, 0]]}],
        "safe_background": [{"id": "safe", "frame_id": 7, "measurement_group": "paired_safe", "polygon": [[0, 0], [1, 0], [1, 1]]}],
    }
    path = tmp_path / "annotations.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = load_source_annotations(path)
    assert loaded["objects"][0]["measurement_group"] == "paired_object"
    payload["objects"][0]["measurement_group"] = " "
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(VideoAnnotationError, match="measurement_group"):
        load_source_annotations(path)
