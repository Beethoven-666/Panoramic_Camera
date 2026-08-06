from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from panorama_demo.video_baseline_freeze import BaselineReference, verify_baseline


ROOT = Path(__file__).resolve().parents[1]


def test_baseline_freeze_compares_decoded_pixels_and_raw_owner(tmp_path):
    image = np.full((3, 4, 3), 42, dtype=np.uint8)
    assert cv2.imwrite(str(tmp_path / "video_panorama.png"), image)
    owner = np.arange(12, dtype=np.int32).reshape(3, 4)
    np.savez_compressed(tmp_path / "video_pixel_provenance.npz", owner_frame_id=owner)
    reference = BaselineReference(
        panorama_shape=image.shape,
        active_owner_count=12,
        panorama_sha256=__import__("hashlib").sha256(image.tobytes()).hexdigest(),
        owner_sha256=__import__("hashlib").sha256(owner.tobytes()).hexdigest(),
    )
    assert verify_baseline(tmp_path, reference=reference)["matches"] is True


def test_baseline_freeze_rejects_any_hash_change(tmp_path):
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    assert cv2.imwrite(str(tmp_path / "video_panorama.png"), image)
    np.savez_compressed(tmp_path / "video_pixel_provenance.npz", owner_frame_id=np.zeros((2, 2), dtype=np.int32))
    with pytest.raises(ValueError, match="Frozen baseline output mismatch"):
        verify_baseline(tmp_path)


def test_authorized_rebaseline_lock_matches_the_runtime_reference():
    import json

    lock = json.loads(
        (ROOT / "configs" / "video_algorithms" / "baseline_legacy_fast_b07b561.lock.json").read_text(
            encoding="utf-8"
        )
    )
    reference = BaselineReference()
    assert lock["baseline_output"] == {
        "panorama_shape": list(reference.panorama_shape),
        "active_owner_count": reference.active_owner_count,
        "panorama_sha256": reference.panorama_sha256,
        "owner_sha256": reference.owner_sha256,
    }
