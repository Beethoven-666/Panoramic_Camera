"""Resolve an explicitly installed addon without importing Open3D."""
from importlib.util import find_spec
import json
from pathlib import Path
import sys


def addon_python() -> Path | None:
    marker = Path(sys.prefix).parent / "addon/addon-install-manifest.json"
    if marker.is_file():
        payload = json.loads(marker.read_text())
        interpreter = marker.parent / "venv/bin/python"
        if (payload.get("schema") == "gemini305-owned-install/v1"
                and payload.get("python") == str(interpreter) and interpreter.is_file()):
            return interpreter
        raise RuntimeError("Installed 3-D addon manifest does not match its interpreter")
    return Path(sys.executable) if find_spec("open3d") is not None else None
