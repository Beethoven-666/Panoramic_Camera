import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import pytest


def load_installer(path):
    spec = importlib.util.spec_from_file_location("g305_installer_test", path)
    module = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


def facts():
    return {"os":"ubuntu","release":"22.04","machine":"x86_64","implementation":"cpython",
            "python":[3,10],"glibc":"2.35","gpu_compute_capabilities":["12.0"]}


@pytest.mark.parametrize("field,value,error", [
    ("os","debian","UNSUPPORTED_PLATFORM"), ("release","24.04","UNSUPPORTED_PLATFORM"),
    ("python",[3,11],"UNSUPPORTED_PYTHON_ABI"), ("python",[3,12],"UNSUPPORTED_PYTHON_ABI"),
    ("gpu_compute_capabilities",["8.6"],"UNSUPPORTED_GPU_VARIANT")])
def test_installer_rejects_unsupported_platform_abi_gpu(field, value, error):
    module = load_installer(Path(__file__).resolve().parents[1]/"packaging/linux/install_runtime.py")
    current = facts()
    current[field] = value
    assert error in module.platform_errors(current)


def bundle_fixture(tmp_path):
    root = tmp_path / "bundle"
    (root / "manifests").mkdir(parents=True)
    (root / "wheels").mkdir()
    shutil.copy2(Path(__file__).resolve().parents[1]/"packaging/linux/install_runtime.py", root/"install_runtime.py")
    (root / "manifests/bundle-manifest.json").write_text(json.dumps({
        "kind":"base", "sdk_version":"0.3.0rc1", "source_commit":"a"*40,
        "runtime_variant":"ubuntu22.04-x86_64-py310-sm120"}))
    lines = [hashlib.sha256(p.read_bytes()).hexdigest()+"  "+p.relative_to(root).as_posix()
             for p in root.rglob("*") if p.is_file()]
    (root / "checksums.sha256").write_text("\n".join(lines)+"\n")
    return root, load_installer(root/"install_runtime.py")


def fake_run(command, log, *, cwd):
    log.write(repr(command).encode())
    if command[1:3] == ["-m","venv"]:
        binary = Path(command[3]) / "bin"
        binary.mkdir(parents=True)
        (binary / "g305-capture").write_text("#!"+str(binary/"python")+"\n")


def test_first_repeat_uninstall_and_user_data_preservation(tmp_path, monkeypatch):
    root, module = bundle_fixture(tmp_path)
    monkeypatch.setattr(module, "host_facts", facts)
    monkeypatch.setattr(module, "run", fake_run)
    args = argparse.Namespace(prefix=tmp_path/"install",python=sys.executable,action="install")
    assert module.install(args)["status"] == "INSTALLED"
    assert module.install(args)["status"] == "ALREADY_INSTALLED"
    assert ".install-" not in (args.prefix/"venv/bin/g305-capture").read_text()
    user = args.prefix / "user-capture"
    user.mkdir()
    (user/"frame.jpg").write_bytes(b"user data")
    args.action = "uninstall"
    assert module.install(args)["status"] == "UNINSTALLED"
    assert (user/"frame.jpg").read_bytes() == b"user data"
    assert module.install(args)["status"] == "ALREADY_UNINSTALLED"


def test_failed_install_never_publishes_prefix_and_keeps_failure_log(tmp_path, monkeypatch):
    _, module = bundle_fixture(tmp_path)
    monkeypatch.setattr(module, "host_facts", facts)

    def failure(command, log, *, cwd):
        fake_run(command, log, cwd=cwd)
        if "pip" in command:
            raise RuntimeError("injected offline install failure")

    monkeypatch.setattr(module, "run", failure)
    args = argparse.Namespace(prefix=tmp_path/"install",python=sys.executable,action="install")
    with pytest.raises(RuntimeError, match="injected"):
        module.install(args)
    assert not args.prefix.exists()
    failed = list(tmp_path.glob("install.install-failed-*"))
    assert len(failed) == 1
    assert (failed[0]/"install.log").exists()
    assert (failed[0]/"failure.json").exists()


def test_checksum_failure_precedes_installation(tmp_path):
    root, module = bundle_fixture(tmp_path)
    (root/"manifests/bundle-manifest.json").write_text("changed")
    args = argparse.Namespace(prefix=tmp_path/"install",python=sys.executable,action="install")
    with pytest.raises(ValueError, match="checksum mismatch"):
        module.install(args)
    assert not args.prefix.exists()


def test_installer_entrypoints_never_invoke_sudo():
    root = Path(__file__).resolve().parents[1]/"packaging/linux"
    for name in ("install.sh","uninstall.sh","install-addon.sh","uninstall-addon.sh","install_runtime.py"):
        assert "sudo" not in (root/name).read_text()


@pytest.mark.parametrize("name", ["wheels/numpy-0.0-py3-none-any.whl", "ORBvoc.txt", "acceptance-tools/changed.py"])
def test_undeclared_file_rejected_before_prefix_creation(tmp_path, name):
    root, module = bundle_fixture(tmp_path)
    extra = root / name
    extra.parent.mkdir(exist_ok=True)
    extra.write_bytes(b"unapproved file")
    args = argparse.Namespace(prefix=tmp_path/"install", python=sys.executable, action="install")
    with pytest.raises(ValueError, match="Undeclared bundle file"):
        module.install(args)
    assert not args.prefix.exists()


def test_declared_wrong_runtime_variant_is_rejected(tmp_path, monkeypatch):
    root, module = bundle_fixture(tmp_path)
    manifest_path = root / "manifests/bundle-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["runtime_variant"] = "ubuntu22.04-x86_64-py310-sm86"
    manifest_path.write_text(json.dumps(manifest))
    checksums = root / "checksums.sha256"
    checksums.write_text("\n".join(
        module.digest(p) + "  " + p.relative_to(root).as_posix()
        for p in root.rglob("*") if p.is_file() and p != checksums) + "\n")
    monkeypatch.setattr(module, "host_facts", facts)
    args = argparse.Namespace(prefix=tmp_path/"install", python=sys.executable, action="install")
    with pytest.raises(ValueError, match="UNSUPPORTED_RUNTIME_VARIANT"):
        module.install(args)
    assert not args.prefix.exists()
