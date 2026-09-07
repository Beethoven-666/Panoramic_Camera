"""The only native v3 qualification decision; recompute all gates from raw evidence."""
from __future__ import annotations

from pathlib import Path

from .binding import QUALIFICATION_ENVIRONMENT, RUNTIME_VARIANT, SUBJECT_COMMIT, revalidate_binding, verify_candidate, verify_software
from .common import file_identity, read, write_new


def aggregate(raw_root, binding_path, candidate_index, software_acceptance, *, subject_verification=None):
    root = Path(raw_root).resolve()
    errors, evidence = [], {}
    software_ready = False
    binding = {}
    try:
        binding = read(binding_path)
        candidate = binding["candidate"]
        actual = verify_candidate(candidate_index, candidate["phase3_freeze"]["path"],
            {k: v["path"] for k, v in candidate["artifacts"].items()}, candidate["extracted_roots"]["base"],
            candidate["extracted_roots"]["three_d_addon"], candidate["orb_runtime"]["manifest_path"])
        evidence["software_acceptance"] = verify_software(software_acceptance, actual)
        software_ready = True
        revalidate_binding(binding, candidate_index, software_acceptance)
        evidence["binding"] = file_identity(binding_path)
    except (OSError, ValueError, KeyError, TypeError, ImportError, RuntimeError) as exc:
        errors.append("binding: " + str(exc))
    if not software_ready and subject_verification is not None:
        # A staging audit can retain software readiness on an unavailable native
        # host. It cannot substitute for the native binding or enable any gate.
        try:
            candidate = read(subject_verification)["candidate"]
            actual = verify_candidate(candidate_index, candidate["phase3_freeze"]["path"],
                {k: v["path"] for k, v in candidate["artifacts"].items()}, candidate["extracted_roots"]["base"],
                candidate["extracted_roots"]["three_d_addon"], candidate["orb_runtime"]["manifest_path"])
            evidence["software_acceptance"] = verify_software(software_acceptance, actual)
            evidence["staging_subject_verification"] = file_identity(subject_verification)
            software_ready = True
        except (OSError, ValueError, KeyError, TypeError, ImportError, RuntimeError) as exc:
            errors.append("staging_subject_verification: " + str(exc))

    components = {}
    # No SDK imports or evidence processing on a failed candidate/tool binding.
    if not errors:
        from .campaign import validate_campaign, verify_final_evidence_seal
        from .inventory import validate_inventory
        from .fault_cases import validate_fault_campaign
        from .retention import validate_retention
        from .endurance import validate_endurance

        try:
            evidence["final_evidence_seal"] = verify_final_evidence_seal(root, binding)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append("final_evidence_seal: " + str(exc))
        for name, validator in (("native_h0_core", validate_campaign), ("native_inventory", validate_inventory),
                                ("native_fault_campaign", validate_fault_campaign),
                                ("native_retention", validate_retention), ("native_endurance", validate_endurance)):
            try:
                result = validator(root, binding)
                detail_errors = result["errors"]
                if not isinstance(detail_errors, list):
                    raise ValueError("Validator did not return an errors list")
                if result.get("status") in {"FAIL", "NOT_EXECUTED", "BLOCKED"} and not detail_errors:
                    detail_errors = ["Validator did not complete"]
                components[name] = "FAIL" if detail_errors else "PASS"
                errors.extend(name + ": " + str(error) for error in detail_errors)
                evidence[name] = result.get("evidence", {})
            except (OSError, ValueError, KeyError, TypeError, ImportError, RuntimeError) as exc:
                components[name] = "FAIL"
                errors.append(name + ": " + str(exc))
    passed = software_ready and not errors
    status = "PASS" if passed else "FAIL"
    return {"schema": "gemini305-sdk-native-acceptance/v3", "status": status,
            "milestone": "SDK_HARDWARE_QUALIFIED" if passed else "NATIVE_H0_INCOMPLETE",
            "subject_source_commit": SUBJECT_COMMIT,
            "h0_tool_commit": binding.get("qualification_tool", {}).get("commit"),
            "h0_id": binding.get("h0_id"), "runtime_variant": RUNTIME_VARIANT,
            "platform": QUALIFICATION_ENVIRONMENT, "qualification_environment": QUALIFICATION_ENVIRONMENT,
            "bare_metal_qualified": False, "vmware_qualified": passed,
            "software_ready": software_ready, "hardware_qualified": passed, "long_duration_qualified": passed,
            "release_ready": False, "signature_status": "UNSIGNED",
            "native_h0_core": components.get("native_h0_core", "NOT_EXECUTED"),
            "native_fault_campaign": components.get("native_fault_campaign", "NOT_EXECUTED"),
            "native_endurance_30m": components.get("native_endurance", "NOT_EXECUTED"),
            "native_endurance_2h": components.get("native_endurance", "NOT_EXECUTED"),
            "sdk_cli_exact_equivalence": components.get("native_h0_core", "NOT_EXECUTED"),
            "post_3d": components.get("native_h0_core", "NOT_EXECUTED"),
            "post_3d_failure_isolation": components.get("native_h0_core", "NOT_EXECUTED"),
            "orb_distribution_license": "BLOCKED/ORB_LICENSE",
            "project_owner_license": "WAITING_FOR_OWNER_APPROVAL",
            "evidence": evidence, "errors": errors}


def publish(raw_root, binding_path, candidate_index, software_acceptance, output_directory, *, subject_verification=None):
    result = aggregate(raw_root, binding_path, candidate_index, software_acceptance, subject_verification=subject_verification)
    write_new(Path(output_directory) / "native_acceptance_status.json", result)
    return result
