"""Fresh-root provenance and fixed sample schedule failures must stop qualification."""
from pathlib import Path

import pytest

from qualification.native_h0.campaign import Campaign, LABELS, bound_path, verify_operation
from qualification.native_h0.common import read, write_new


@pytest.fixture
def binding():
    return {"h0_id": "h0-test", "subject": {"source_commit": "2abb132043e7c5ae1805a2ed1dd91516ebf2f874"},
            "qualification_tool": {"commit": "a" * 40}}


def test_operation_is_fresh_bound_and_raw_files_are_rechecked(tmp_path, binding):
    campaign = Campaign(tmp_path / "campaign", binding, create=True)
    with campaign.operation("capture-contracts/fixed") as root:
        (root / "raw.log").write_text("real producer output")
    assert verify_operation(campaign.root, binding, "capture-contracts/fixed") == root
    (root / "raw.log").write_text("changed")
    with pytest.raises(ValueError, match="Evidence changed"):
        verify_operation(campaign.root, binding, "capture-contracts/fixed")


def test_reject_preexisting_root_and_old_operation(tmp_path, binding):
    root = tmp_path / "old"
    root.mkdir()
    with pytest.raises(FileExistsError):
        Campaign(root, binding, create=True)
    campaign = Campaign(tmp_path / "new", binding, create=True)
    with campaign.operation("capture-contracts/photo"):
        pass
    with pytest.raises(FileExistsError):
        with campaign.operation("capture-contracts/photo"):
            pass


def test_other_h0_binding_rejected(tmp_path, binding):
    campaign = Campaign(tmp_path / "new", binding, create=True)
    with campaign.operation("post-3d"):
        pass
    other = {**binding, "h0_id": "other-h0"}
    with pytest.raises(ValueError, match="Unbound or stale"):
        verify_operation(campaign.root, other, "post-3d")


def test_failed_sample_cannot_be_replaced(tmp_path, binding):
    campaign = Campaign(tmp_path / "new", binding, create=True)
    with pytest.raises(RuntimeError):
        with campaign.operation("sdk-physical/run_03"):
            raise RuntimeError("camera disconnected")
    with pytest.raises(ValueError, match="Failed or invalid"):
        verify_operation(campaign.root, binding, "sdk-physical/run_03")
    assert read(campaign.root / "sdk-physical/run_03/operation_end.json")["error"] == "camera disconnected"
    assert LABELS == ("warmup_01", "run_01", "run_02", "run_03", "run_04", "run_05")


@pytest.mark.parametrize("name", ["../old/session", "../../another-h0", str(Path.cwd().resolve())])
def test_foreign_evidence_paths_rejected(tmp_path, name):
    with pytest.raises(ValueError):
        bound_path(tmp_path, name)


def test_added_evidence_rejected(tmp_path, binding):
    campaign = Campaign(tmp_path / "new", binding, create=True)
    with campaign.operation("cli-live") as root:
        pass
    write_new(root / "old_result.json", {"status": "PASS"})
    with pytest.raises(ValueError, match="added evidence"):
        verify_operation(campaign.root, binding, "cli-live")
