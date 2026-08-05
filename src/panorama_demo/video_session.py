"""Strict, isolated input contract for continuous RGB-D video sessions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .session import RGBDSession, load_rgbd_session


_VIDEO_CAPTURE_MODES = {
    "continuous_rgbd_video_auto",
    "continuous_rgbd_video_fixed_exposure",
}


@dataclass(frozen=True)
class VideoSession:
    rgbd: RGBDSession
    capture_mode: str
    legacy_v1: bool
    product_eligible: bool


def load_video_session(
    input_path: str | Path,
    *,
    validate_frame_files: bool = True,
    validation_workers: int = 1,
) -> VideoSession:
    """Load a complete colour-aligned video session without admitting photo input.

    v1 auto-exposure captures predate the product-eligibility marker and are
    intentionally accepted for C-grade validation only.  They remain rejected
    by the photo pipeline because this module is never imported there.
    """

    rgbd = load_rgbd_session(
        input_path,
        validate_frame_files=validate_frame_files,
        validation_workers=validation_workers,
    )
    manifest: dict[str, Any] = rgbd.manifest or {}
    mode = manifest.get("capture_mode")
    if not isinstance(mode, str) or mode not in _VIDEO_CAPTURE_MODES:
        raise ValueError("g305-video-panorama requires a continuous RGB-D video session")
    schema = manifest.get("schema")
    if schema not in {"panorama-demo-session/v1", "panorama-demo-session/v2"}:
        raise ValueError("Video session has an unsupported manifest schema")
    legacy = schema == "panorama-demo-session/v1"
    eligibility = manifest.get("product_eligibility")
    if legacy:
        eligible = True
    else:
        if not isinstance(eligibility, dict) or eligibility.get("photo_panorama") is not False:
            raise ValueError("Video v2 manifest has invalid product eligibility")
        if eligibility.get("video_panorama") is not True:
            raise ValueError("Video v2 session is not eligible for video panorama")
        eligible = True
    if manifest.get("clean_shutdown") is not True:
        raise ValueError("Video session was not cleanly closed")
    if int(manifest.get("write_errors", 0)) != 0 or manifest.get("writer_errors", []) != []:
        raise ValueError("Video session contains capture/write errors")
    return VideoSession(rgbd=rgbd, capture_mode=mode, legacy_v1=legacy, product_eligible=eligible)
