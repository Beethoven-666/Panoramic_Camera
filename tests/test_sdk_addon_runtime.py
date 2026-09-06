import json

from panorama_demo import sdk_runtime


def test_base_has_no_implicit_open3d_addon(tmp_path, monkeypatch):
    monkeypatch.setattr(sdk_runtime.sys,"prefix",str(tmp_path/"venv"))
    monkeypatch.setattr(sdk_runtime,"find_spec",lambda _:None)
    assert sdk_runtime.addon_python() is None


def test_installed_addon_uses_separate_interpreter(tmp_path, monkeypatch):
    monkeypatch.setattr(sdk_runtime.sys,"prefix",str(tmp_path/"venv"))
    interpreter = tmp_path / "addon/venv/bin/python"
    interpreter.parent.mkdir(parents=True)
    interpreter.write_text("interpreter")
    (tmp_path/"addon/addon-install-manifest.json").write_text(json.dumps({
        "schema":"gemini305-owned-install/v1","python":str(interpreter)}))
    assert sdk_runtime.addon_python() == interpreter
