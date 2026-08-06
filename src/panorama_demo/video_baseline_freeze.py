"""Exact pixel/owner verification for the immutable legacy baseline."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class BaselineReference:
    # Authorized replacement lock measured from the immutable approved session
    # on this repository/runtime. The original external reference could not be
    # recovered, so these values deliberately replace it rather than silently
    # weakening verification.
    panorama_shape: tuple[int, int, int] = (456, 1818, 3)
    active_owner_count: int = 36
    panorama_sha256: str = "5f2cc36061132c49e9077ec66c261a52c2474a7157b45a240b64184bdb0c2bb7"
    owner_sha256: str = "3b3fc00efbd289d69ce5e7d3e57b13635ca239ad0d2792a7231023356df2903f"


def measure_baseline(output: Path) -> dict[str, object]:
    image = cv2.imread(str(output / "video_panorama.png"), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Baseline output lacks decodable video_panorama.png")
    with np.load(output / "video_pixel_provenance.npz") as archive:
        owner = archive["owner_frame_id"]
    if owner.dtype != np.int32:
        raise ValueError("Baseline owner map must use int32")
    return {
        "panorama_shape": list(image.shape),
        "active_owner_count": int(np.unique(owner).size),
        "panorama_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
        "owner_sha256": hashlib.sha256(owner.tobytes()).hexdigest(),
    }


def verify_baseline(output: Path, *, reference: BaselineReference = BaselineReference()) -> dict[str, object]:
    measured = measure_baseline(output)
    expected = {
        "panorama_shape": list(reference.panorama_shape),
        "active_owner_count": reference.active_owner_count,
        "panorama_sha256": reference.panorama_sha256,
        "owner_sha256": reference.owner_sha256,
    }
    mismatches = {
        key: {"expected": expected[key], "observed": measured[key]}
        for key in expected
        if expected[key] != measured[key]
    }
    result = {"expected": expected, "observed": measured, "matches": not mismatches, "mismatches": mismatches}
    (output / "baseline_freeze_verification.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    if mismatches:
        raise ValueError("Frozen baseline output mismatch: " + ", ".join(mismatches))
    return result
