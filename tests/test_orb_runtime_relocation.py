import importlib.util
import os
from pathlib import Path
import shutil
import subprocess

import pytest


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts/relocate_orb_runtime.py"
    spec = importlib.util.spec_from_file_location("g305_relocation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runtime_system_allowlist_excludes_build_and_usr_local():
    module = _module()
    assert module.is_system(Path("/lib/x86_64-linux-gnu/libc.so.6"))
    assert not module.is_system(Path("/home/zyh/build/libpango.so"))
    assert not module.is_system(Path("/usr/local/lib/libpango.so"))


@pytest.mark.skipif(os.name != "posix" or not shutil.which("patchelf"), reason="Linux ELF tools")
def test_real_elf_relocation_works_without_original_dependency(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    library = tmp_path / "build"
    library.mkdir()
    (library / "example.c").write_text("int value(void) { return 42; }")
    (source / "main.c").write_text("int value(void); int main(void) { return value() != 42; }")
    subprocess.run(["cc", "-shared", "-fPIC", str(library / "example.c"), "-Wl,-soname,libexample.so",
                    "-o", str(library / "libexample.so")], check=True)
    subprocess.run(["cc", str(source / "main.c"), "-L" + str(library), "-lexample",
                    "-Wl,-rpath," + str(library), "-o", str(source / "runner")], check=True)
    output = tmp_path / "portable"
    report = _module().prepare(source, output, shutil.which("patchelf"))
    assert report["status"] == "PASS"
    (library / "libexample.so").rename(library / "disabled-original")
    relocated = tmp_path / "different-prefix"
    shutil.copytree(output, relocated)
    subprocess.run([str(relocated / "runner")], env={k:v for k,v in os.environ.items() if k != "LD_LIBRARY_PATH"}, check=True)
    assert _module().scan(relocated, shutil.which("patchelf"))["status"] == "PASS"
