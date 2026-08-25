from __future__ import annotations

from panorama_demo.sdk_doctor import SDKDoctorCheck, SDKDoctorReport


def test_doctor_report_serializes_typed_checks() -> None:
    check = SDKDoctorCheck("PASS", "OK", {"value": 1}, True, True)
    report = SDKDoctorReport("gemini305-sdk-doctor/v1", {"python": check}, True, False, False,
                             "SDK_SOFTWARE_READY")
    assert report.as_dict()["checks"]["python"]["reason_code"] == "OK"
    assert "SDK_SOFTWARE_READY" in str(report)


def test_orb_license_is_not_silently_approved(monkeypatch, tmp_path) -> None:
    from panorama_demo import sdk_doctor

    monkeypatch.setattr(sdk_doctor.sys, "platform", "linux")
    report = sdk_doctor.run_sdk_doctor(
        orbslam3_root=tmp_path,
        orbslam3_executable="rgbd",
        orbslam3_vocabulary="ORBvoc.txt",
        orb_runtime_kind="native_linux",
    )
    assert report.checks["orb_distribution_license"].status == "BLOCKED"
    assert report.release_ready is False
