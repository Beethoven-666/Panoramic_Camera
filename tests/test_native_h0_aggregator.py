"""No historical PASS or missing native binding can issue hardware qualification."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qualification.native_h0.aggregator import aggregate, publish


def test_historical_pass_cannot_issue_native_v3(tmp_path):
    (tmp_path / "sdk_h0_acceptance.json").write_text(json.dumps({"status": "PASS", "hardware_qualified": True}))
    result = aggregate(tmp_path, tmp_path / "missing-binding", tmp_path / "index", tmp_path / "software")
    assert result["schema"] == "gemini305-sdk-native-acceptance/v3"
    assert result["hardware_qualified"] is False
    assert result["long_duration_qualified"] is False
    assert result["release_ready"] is False
    assert result["signature_status"] == "UNSIGNED"
    assert result["errors"]


def test_verified_staging_subject_preserves_only_software_readiness(tmp_path, monkeypatch):
    from qualification.native_h0 import aggregator
    candidate = {"phase3_freeze": {"path": "freeze"}, "artifacts": {},
                 "extracted_roots": {"base": "base", "three_d_addon": "addon"}, "orb_runtime": {"manifest_path": "orb"}}
    staging = tmp_path / "subject.json"
    staging.write_text(json.dumps({"candidate": candidate, "status": "PASS"}))
    calls = []
    monkeypatch.setattr(aggregator, "verify_candidate", lambda *args: calls.append("raw_candidate") or {})
    monkeypatch.setattr(aggregator, "verify_software", lambda *args: calls.append("software_binding") or {})
    result = aggregate(tmp_path, tmp_path / "missing-binding", tmp_path / "index", tmp_path / "software",
                       subject_verification=staging)
    assert calls == ["raw_candidate", "software_binding"]
    assert result["software_ready"] is True
    assert result["hardware_qualified"] is False and result["long_duration_qualified"] is False
    assert result["h0_tool_commit"] is None
    assert any(error.startswith("binding:") for error in result["errors"])


def test_failure_publication_preserves_prior_result(tmp_path):
    output = tmp_path / "final"
    result = publish(tmp_path, tmp_path / "binding", tmp_path / "index", tmp_path / "software", output)
    before = (output / "native_acceptance_status.json").read_bytes()
    assert result["hardware_qualified"] is False
    with pytest.raises(FileExistsError):
        publish(tmp_path, tmp_path / "binding", tmp_path / "index", tmp_path / "software", output)
    assert (output / "native_acceptance_status.json").read_bytes() == before


@pytest.mark.parametrize("failed_component", ["validate_campaign", "validate_inventory", "validate_fault_campaign",
                                            "validate_retention", "validate_endurance", "final_evidence_seal"])
def test_every_required_raw_validator_can_veto_qualification(tmp_path, monkeypatch, failed_component):
    from qualification.native_h0 import aggregator, campaign, inventory, fault_cases, retention, endurance

    binding = {"h0_id": "fresh", "candidate": {"phase3_freeze": {"path": "freeze"}, "artifacts": {},
                "extracted_roots": {"base": "base", "three_d_addon": "addon"}, "orb_runtime": {"manifest_path": "orb"}},
               "qualification_tool": {"commit": "a" * 40}}
    path = tmp_path / "binding.json"
    path.write_text(json.dumps(binding))
    monkeypatch.setattr(aggregator, "verify_candidate", lambda *a: {})
    monkeypatch.setattr(aggregator, "verify_software", lambda *a: {})
    monkeypatch.setattr(aggregator, "revalidate_binding", lambda *a: {})
    called = []
    def validator(name):
        def run(*args):
            called.append(name)
            return {"errors": ["raw evidence invalid"] if name == failed_component else [], "evidence": {}}
        return run
    for module, name in ((campaign, "validate_campaign"), (inventory, "validate_inventory"),
                         (fault_cases, "validate_fault_campaign"), (retention, "validate_retention"),
                         (endurance, "validate_endurance")):
        monkeypatch.setattr(module, name, validator(name))
    def seal(*args):
        if failed_component == "final_evidence_seal":
            raise ValueError("evidence modified after sealing")
        return {"status": "PASS"}
    monkeypatch.setattr(campaign, "verify_final_evidence_seal", seal)
    result = aggregate(tmp_path, path, tmp_path / "index", tmp_path / "software")
    assert len(called) == 5
    assert result["software_ready"] is True
    assert result["hardware_qualified"] is False
    assert result["long_duration_qualified"] is False
