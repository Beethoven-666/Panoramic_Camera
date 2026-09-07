from qualification.native_h0.retention import snapshot_files, verify_snapshot, validate_retention


def test_preserved_files_rehashed_and_changed_output_detected(tmp_path):
    file = tmp_path / "video_panorama.png"
    file.write_bytes(b"actual preserved pixels")
    snapshot = snapshot_files(tmp_path)
    assert verify_snapshot(tmp_path, snapshot) == []
    file.write_bytes(b"changed")
    assert verify_snapshot(tmp_path, snapshot)


def test_missing_output_and_empty_snapshot_fail(tmp_path):
    assert verify_snapshot(tmp_path, {})
    assert verify_snapshot(tmp_path, {"missing":{"bytes":1,"sha256":"bad"}})


def test_all_retention_cases_required(tmp_path):
    result = validate_retention(tmp_path, {})
    assert len(result["errors"]) == 6
