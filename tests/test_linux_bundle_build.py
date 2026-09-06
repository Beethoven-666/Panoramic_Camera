import hashlib
import importlib.util
from pathlib import Path
import subprocess

import pytest


def module():
    spec = importlib.util.spec_from_file_location(
        "bundle_build",
        Path(__file__).resolve().parents[1] / "scripts/build_linux_sdk_bundles.py",
    )
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_dirty_source_refused_before_build(tmp_path):
    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
    (tmp_path / "changed").write_text("dirty")
    with pytest.raises(ValueError, match="dirty"):
        module().source_identity(tmp_path)


def test_exact_wheel_hash_changes_selection(tmp_path):
    wheel = tmp_path / "x-1.0-py3-none-any.whl"
    wheel.write_bytes(b"wheel")
    lock = tmp_path / "lock"
    lock.write_text("x==1.0 --hash=sha256:" + hashlib.sha256(b"wheel").hexdigest())
    assert module().locked_wheels(lock, [wheel]) == [wheel]
    wheel.write_bytes(b"changed")
    with pytest.raises(ValueError, match="corrupt"):
        module().locked_wheels(lock, [wheel])


def test_signature_uses_only_explicit_existing_key(tmp_path, monkeypatch):
    builder = module()
    calls = []
    monkeypatch.setattr(
        builder.subprocess, "run", lambda command, **kw: calls.append(command)
    )
    builder.sign(tmp_path / "checksums", None)
    assert not calls
    builder.sign(tmp_path / "checksums", "TEST-EXISTING-KEY", True)
    assert "--detach-sign" in calls[0] and calls[0][-2].endswith(".TEST_ONLY.asc")
    key = tmp_path / "private"
    key.write_text("never bundle")
    with pytest.raises(ValueError, match="outside"):
        builder.sign(tmp_path / "checksums", str(key))
