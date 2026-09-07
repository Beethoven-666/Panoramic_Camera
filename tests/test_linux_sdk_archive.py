import hashlib
import importlib.util
import json
from pathlib import Path
import tarfile

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", ["/absolute", "../escape", "top/../escape", "other/file", "C:/file", "top\\evil"])
def test_archive_paths_rejected_before_extraction(name):
    info = tarfile.TarInfo(name)
    with pytest.raises(ValueError, match="UNSAFE_ARCHIVE_PATH"):
        load("verify_linux_sdk_archive").validate_member(info, "top")


def test_archive_links_not_emitted_by_builder_are_rejected():
    info = tarfile.TarInfo("top/link")
    info.type = tarfile.SYMTYPE
    info.linkname = "../../outside"
    with pytest.raises(ValueError, match="UNSUPPORTED_ARCHIVE_MEMBER"):
        load("verify_linux_sdk_archive").validate_member(info, "top")


def test_tar_metadata_and_compression_are_reproducible(tmp_path):
    pytest.importorskip("zstandard")
    builder = load("build_linux_sdk_bundles")
    tree = tmp_path / "input"
    tree.mkdir()
    (tree / "install.sh").write_bytes(b"#!/bin/sh\n")
    (tree / "data").write_bytes(b"payload")
    archives = [tmp_path / "a.tar.zst", tmp_path / "b.tar.zst"]
    builder.deterministic_archive(archives[0], builder.tree_entries(tree, "top"), 1700000000)
    (tree / "data").touch()
    (tree / "data").chmod(0o777)
    builder.deterministic_archive(archives[1], builder.tree_entries(tree, "top"), 1700000000)
    assert archives[0].read_bytes() == archives[1].read_bytes()
    with load("verify_linux_sdk_archive").open_tar(archives[0]) as archive:
        members = list(archive)
    assert [m.name for m in members] == sorted(m.name for m in members)
    assert all((m.uid, m.gid, m.uname, m.gname, m.mtime) == (0, 0, "", "", 1700000000) for m in members)
    assert next(m for m in members if m.name == "top/data").mode == 0o644
    with pytest.raises(FileExistsError):
        builder.deterministic_archive(archives[0], [], 1700000000)


def source_candidate(tmp_path):
    builder = load("build_linux_sdk_bundles")
    top = "gemini305-sdk-source-compliance-0.3.0rc1"
    root = tmp_path / top
    root.mkdir()
    (root / "source.py").write_bytes(b"print('source')\n")
    (root / "checksums.sha256").write_text(builder.checksum_text(root))
    archive = tmp_path / (top + ".tar.zst")
    builder.deterministic_archive(archive, builder.tree_entries(root, top), 1700000000)
    index = tmp_path / "index.json"
    index.write_text(json.dumps({"sdk_version": "0.3.0rc1", "runtime_variant": "test",
                                "artifacts": {"source_compliance": {"filename": archive.name,
                                "size": archive.stat().st_size, "archive_sha256": builder.sha(archive)}}}))
    return index, archive


def test_real_archive_verification_and_tamper_preserve_destination(tmp_path):
    pytest.importorskip("zstandard")
    index, archive = source_candidate(tmp_path)
    verifier = load("verify_linux_sdk_archive")
    result = verifier.verify(index, archive, "source_compliance", tmp_path / "extract")
    assert result["status"] == "PASS"
    assert Path(result["extracted_root"]).is_dir()
    archive.write_bytes(archive.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="ARCHIVE_IDENTITY_MISMATCH"):
        verifier.verify(index, archive, "source_compliance", tmp_path / "must-not-exist")
    assert not (tmp_path / "must-not-exist").exists()


def test_content_verifier_rejects_extra_file(tmp_path):
    payload = tmp_path / "payload"
    payload.write_bytes(b"ok")
    (tmp_path / "checksums.sha256").write_text(hashlib.sha256(b"ok").hexdigest() + "  payload\n")
    (tmp_path / "extra").write_bytes(b"undeclared")
    with pytest.raises(ValueError, match="UNDECLARED"):
        load("verify_linux_sdk_archive").verify_contents(tmp_path, "source_compliance")


def test_build_comparison_rejects_self_comparison(tmp_path):
    with pytest.raises(ValueError, match="Independent"):
        load("compare_linux_sdk_builds").compare(tmp_path, tmp_path)
