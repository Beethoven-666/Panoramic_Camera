import hashlib

import pytest

from panorama_demo.sdk_retention import cleanup_owned_session, resume_cleanup
from panorama_demo.sdk_state import JobRecord, RetentionPolicy


def _job(tmp_path, owned=True):
    job = JobRecord.create(tmp_path / "job", source_commit="abc", config_sha="def", owns_session=owned)
    session = job.root / "session/run"
    session.mkdir(parents=True)
    (session / "frame").write_bytes(b"original")
    output = job.root / "2d/image"
    output.write_bytes(b"published")
    return job, session, {str(output): hashlib.sha256(output.read_bytes()).hexdigest()}


@pytest.mark.parametrize("owned,warning,three_d_success", [(False, False, True), (True, True, True), (True, False, False)])
def test_incomplete_or_external_inputs_are_retained(tmp_path, owned, warning, three_d_success):
    job, session, outputs = _job(tmp_path, owned)
    if warning:
        job.update(warnings=["emergency"])
    assert not cleanup_owned_session(job, session, policy=RetentionPolicy.DELETE_AFTER_ALL_SUCCESS,
                                     formal_2d_published=True, three_d_requested=True,
                                     three_d_succeeded=three_d_success, output_sha256=outputs)
    assert session.is_dir()


def test_successful_owned_session_cleanup_and_crash_resume(tmp_path, monkeypatch):
    job, session, outputs = _job(tmp_path)
    from panorama_demo import sdk_retention
    original = sdk_retention.shutil.rmtree

    def fail(_):
        raise OSError("simulated crash after rename")

    monkeypatch.setattr(sdk_retention.shutil, "rmtree", fail)
    with pytest.raises(OSError, match="simulated crash"):
        cleanup_owned_session(job, session, policy=RetentionPolicy.DELETE_AFTER_ALL_SUCCESS,
                              formal_2d_published=True, three_d_requested=False,
                              three_d_succeeded=False, output_sha256=outputs)
    assert not session.exists()
    assert session.with_name("run.delete_pending").exists()
    job.update(warnings=["cleanup_pending: simulated crash after rename"])
    monkeypatch.setattr(sdk_retention.shutil, "rmtree", original)
    assert resume_cleanup(JobRecord.load(job.root))
    assert not session.with_name("run.delete_pending").exists()
    assert not resume_cleanup(JobRecord.load(job.root))
