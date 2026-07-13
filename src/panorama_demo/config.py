from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from .paths import PROJECT_ROOT


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    default_path = PROJECT_ROOT / "configs" / "demo.yaml"
    with default_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if path is None:
        return config
    custom_path = Path(path).expanduser().resolve()
    with custom_path.open("r", encoding="utf-8") as handle:
        custom = yaml.safe_load(handle) or {}
    custom_capture = custom.get("capture")
    if (
        isinstance(custom_capture, dict)
        and "color_exposure_us" in custom_capture
        and "color_auto_exposure" not in custom_capture
    ):
        # Before color_auto_exposure was configurable, a non-null exposure in a
        # custom file unambiguously selected manual exposure. Preserve that
        # behavior when merging the custom file over today's auto-exposure default.
        custom_capture["color_auto_exposure"] = (
            custom_capture["color_exposure_us"] is None
        )
    return _merge(config, custom)
