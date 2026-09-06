import importlib.util
from pathlib import Path
import sys

import pytest

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]


def test_root_metadata_is_single_cp310_headless_authority():
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert metadata["project"]["requires-python"] == ">=3.10,<3.11"
    dependencies = metadata["project"]["dependencies"]
    assert not any(name.startswith("open3d") for name in dependencies)
    assert any(name.startswith("opencv-python-headless") for name in dependencies)
    assert not (ROOT / "packaging/sdk_pyproject.toml").exists()


def test_resource_hook_refuses_dirty_source_before_build(monkeypatch):
    spec = importlib.util.spec_from_file_location("g305_build_support", ROOT / "build_support.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.subprocess, "check_output", lambda *_args, **_kwargs: b" M tracked.py\n")
    with pytest.raises(RuntimeError, match="Dirty SDK builds"):
        module.source_commit()
