from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from .paths import PROJECT_ROOT


_LEGACY_VIDEO_CONTROL_KEYS = (
    "color_auto_exposure",
    "color_exposure_us",
    "color_ae_max_exposure_us",
    "diagnostic_unrestricted_auto_exposure",
    "diagnostic_replaced_auto_cap_us",
    "color_gain",
    "color_auto_white_balance",
    "color_white_balance",
    "lock_color_controls_after_warmup",
    "post_lock_verified_frames",
    "require_locked_control_metadata",
)


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
    if isinstance(custom_capture, dict):
        # Controls used to sit directly under ``capture``. Keep custom video
        # profiles usable, but never let those legacy keys alter photo mode.
        legacy_controls = {
            key: custom_capture.pop(key)
            for key in _LEGACY_VIDEO_CONTROL_KEYS
            if key in custom_capture
        }
        if legacy_controls:
            custom_video = custom_capture.setdefault("video_mode", {})
            if not isinstance(custom_video, dict):
                raise ValueError("capture.video_mode must be a mapping")
            for key, value in legacy_controls.items():
                custom_video.setdefault(key, value)
        custom_video = custom_capture.get("video_mode")
        if (
            isinstance(custom_video, dict)
            and "color_exposure_us" in custom_video
            and "color_auto_exposure" not in custom_video
        ):
            # Preserve the old inference rule within the video namespace.
            custom_video["color_auto_exposure"] = (
                custom_video["color_exposure_us"] is None
            )
    return _merge(config, custom)
