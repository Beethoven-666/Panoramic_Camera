import csv
import hashlib
import json
from types import SimpleNamespace

import pytest

from panorama_demo.commit_journal import CommitJournal, read_durable_prefix


def _root(root):
    (root / "color").mkdir()
    (root / "depth_aligned").mkdir()
    (root / "calibration.json").write_text('{"calibration":1}')
    (root / "manifest.json").write_text('{"schema":"panorama-demo-session/v2"}')


def _record(journal, handle, frame_id, timestamp=None):
    root = journal.root
    color, depth = root / f"color/{frame_id}.jpg", root / f"depth_aligned/{frame_id}.png"
    color.write_bytes(f"color{frame_id}".encode())
    depth.write_bytes(f"depth{frame_id}".encode())
    row = {"frame_id": frame_id}
    csv.DictWriter(handle, fieldnames=["frame_id"]).writerow(row)
    journal.record(SimpleNamespace(frame_id=frame_id, timestamp_us=frame_id if timestamp is None else timestamp,
                                   color_path=color, aligned_depth_path=depth,
                                   color_sha256=hashlib.sha256(color.read_bytes()).hexdigest(),
                                   aligned_depth_sha256=hashlib.sha256(depth.read_bytes()).hexdigest()), row, handle)


def test_first_frame_durable_and_unconfirmed_tail_excluded(tmp_path):
    _root(tmp_path)
    journal = CommitJournal(tmp_path, interval_frames=30, interval_seconds=3600)
    with (tmp_path / "frames.csv").open("w", newline="") as handle:
        handle.write("frame_id\n")
        _record(journal, handle, 1)
        _record(journal, handle, 2)
    journal.close()
    rows, checkpoint = read_durable_prefix(tmp_path)
    assert [r["frame_id"] for r in rows] == [1]
    assert checkpoint["count"] == 1


def test_corrupt_latest_frame_rolls_back_to_complete_boundary(tmp_path):
    _root(tmp_path)
    journal = CommitJournal(tmp_path, interval_frames=2)
    with (tmp_path / "frames.csv").open("w", newline="") as handle:
        handle.write("frame_id\n")
        _record(journal, handle, 1)
        _record(journal, handle, 2)
    journal.close()
    (tmp_path / "color/2.jpg").write_bytes(b"corrupt")
    rows, checkpoint = read_durable_prefix(tmp_path)
    assert len(rows) == 1
    assert checkpoint["checkpoint_used"] == "previous_commit.json"
    assert checkpoint["rollback_reasons"]
    (tmp_path / "color/1.jpg").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="No valid durable"):
        read_durable_prefix(tmp_path)


def test_regressed_timestamp_is_never_added_to_safe_prefix(tmp_path):
    _root(tmp_path)
    journal = CommitJournal(tmp_path, interval_frames=2)
    with (tmp_path / "frames.csv").open("w", newline="") as handle:
        handle.write("frame_id\n")
        _record(journal, handle, 1, 10)
        _record(journal, handle, 2, 9)
        journal.checkpoint(handle)
    journal.close()
    rows, _ = read_durable_prefix(tmp_path)
    assert len(rows) == 1
    checkpoint = json.loads((tmp_path / "checkpoints/current_commit.json").read_text())
    assert checkpoint["last_timestamp_us"] == 10


def test_csv_corruption_is_not_salvaged_as_confirmed_data(tmp_path):
    _root(tmp_path)
    journal = CommitJournal(tmp_path)
    with (tmp_path / "frames.csv").open("w", newline="") as handle:
        handle.write("frame_id\n")
        _record(journal, handle, 1)
    journal.close()
    (tmp_path / "frames.csv").write_text("frame_id\n99\n")
    with pytest.raises(ValueError, match="CSV prefix differs"):
        read_durable_prefix(tmp_path)
