from __future__ import annotations

from dataclasses import dataclass
import json

import pytest

from panorama_demo.video_panorama import _restrict_scan_to_progress_interval
from panorama_demo.video_split import (
    build_source_progress_evidence,
    source_progress_by_frame,
    write_or_verify_source_progress_evidence,
)


@dataclass(frozen=True)
class _Motion:
    dx: float
    dy: float = 0.0
    reliable: bool = True


def test_progress_restriction_keeps_only_contiguous_real_sources():
    frames = tuple(range(5))
    qualities = list(range(5))
    motions = [_Motion(1.0) for _ in range(4)]
    selected_frames, selected_quality, selected_motion, audit = _restrict_scan_to_progress_interval(
        frames, qualities, motions, (0.24, 0.76)
    )
    assert selected_frames == (1, 2, 3)
    assert selected_quality == [1, 2, 3]
    assert len(selected_motion) == 2
    assert audit is not None
    assert audit["selection"] == "real_contiguous_sources_only"


def test_progress_restriction_rejects_ranges_without_a_real_edge():
    with pytest.raises(ValueError, match="fewer than two"):
        _restrict_scan_to_progress_interval(
            (0, 1, 2), [0, 1, 2], [_Motion(1.0), _Motion(1.0)], (0.01, 0.49)
        )


def test_progress_restriction_uses_locked_reliable_horizontal_coordinate():
    # A large vertical wobble is risk evidence, not scan-distance progress.
    # Using hypot(dx, dy) would put this same real-source interval in a
    # different split and incorrectly leave fewer than two nodes.
    selected, _, _, audit = _restrict_scan_to_progress_interval(
        (10, 20, 30, 40),
        [0, 1, 2, 3],
        [_Motion(1.0), _Motion(1.0, dy=100.0), _Motion(1.0)],
        (0.30, 0.70),
    )

    assert selected == (20, 30)
    assert audit is not None
    assert audit["progress_coordinate"] == "cumulative_reliable_horizontal_motion"
    assert len(audit["source_progress_evidence_sha256"]) == 64


def test_source_progress_evidence_is_real_serializable_and_immutable(tmp_path):
    evidence = build_source_progress_evidence(
        (10, 20, 30, 40),
        [_Motion(2.0), _Motion(1.0, reliable=False), _Motion(3.0)],
    )
    assert source_progress_by_frame(evidence) == {10: 0.0, 20: 0.4, 30: 0.4, 40: 1.0}
    path = tmp_path / "source_progress.json"
    assert write_or_verify_source_progress_evidence(path, evidence) == evidence
    assert json.loads(path.read_text(encoding="utf-8"))["content_sha256"] == evidence["content_sha256"]
    changed = build_source_progress_evidence(
        (10, 20, 30, 40),
        [_Motion(3.0), _Motion(1.0, reliable=False), _Motion(3.0)],
    )
    with pytest.raises(ValueError, match="immutable"):
        write_or_verify_source_progress_evidence(path, changed)


def test_source_progress_evidence_rejects_reliable_direction_reversal():
    with pytest.raises(ValueError, match="reverses"):
        build_source_progress_evidence((1, 2, 3), [_Motion(2.0), _Motion(-1.0)])
