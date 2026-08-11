"""Hash-sealed immutable generations and atomic S1.3 pointers."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np


P0_COMPLETION_SCHEMA = "gemini305-video-s13-p0-completion/v1"
BASE_POINTER_SCHEMA = "gemini305-video-s13-current-base/v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(f".{path.name}.{uuid.uuid4().hex}.pending")
    encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    try:
        with pending.open("wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(pending, path)
    finally:
        pending.unlink(missing_ok=True)


def write_image(path: Path, image: np.ndarray, *, jpeg_quality: int = 95) -> None:
    suffix = path.suffix.lower()
    parameters = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality] if suffix in {".jpg", ".jpeg"} else []
    ok, encoded = cv2.imencode(suffix, image, parameters)
    if not ok:
        raise OSError(f"Failed to encode S1.3 image: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded.tobytes())


def write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        np.savez_compressed(handle, **arrays)


def write_csv(path: Path, rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def seal_p0(p0: Path, *, generation_id: str, metadata: Mapping[str, Any]) -> dict[str, Any]:
    assets = sorted(path for path in p0.rglob("*") if path.is_file() and path != p0 / "P0_completion.json")
    hashes = {path.relative_to(p0).as_posix(): sha256_file(path) for path in assets}
    completion = {
        "schema": P0_COMPLETION_SCHEMA,
        "generation_id": generation_id,
        "stage": "P0",
        "sealed": True,
        "diagnostic_only": True,
        "production_eligible": False,
        "production_lock_eligible": False,
        "assets_sha256": hashes,
        **dict(metadata),
    }
    atomic_write_json(p0 / "P0_completion.json", completion)
    verify_p0_completion(p0)
    return completion


def verify_p0_completion(p0: Path) -> dict[str, Any]:
    path = p0 / "P0_completion.json"
    try:
        completion = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("S1.3 P0 completion is missing or invalid") from exc
    if completion.get("schema") != P0_COMPLETION_SCHEMA or completion.get("sealed") is not True:
        raise ValueError("S1.3 P0 completion is not sealed")
    hashes = completion.get("assets_sha256")
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("S1.3 P0 completion has no asset hashes")
    for name, expected in hashes.items():
        relative = Path(str(name))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"S1.3 sealed P0 asset path is unsafe: {name}")
        asset = p0 / relative
        if not asset.is_file() or sha256_file(asset) != expected:
            raise ValueError(f"S1.3 sealed P0 asset hash mismatch: {name}")
    return completion


def publish_generation(staging: Path, generation: Path) -> None:
    if generation.exists():
        raise FileExistsError(f"S1.3 generation already exists: {generation}")
    generation.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staging, generation)


def update_current_base(root: Path, generation: Path) -> dict[str, Any]:
    completion_path = generation / "P0" / "P0_completion.json"
    completion = verify_p0_completion(completion_path.parent)
    pointer = {
        "schema": BASE_POINTER_SCHEMA,
        "algorithm_id": "S013_output_first_progressive_dense_central_slit_v3",
        "generation_id": completion["generation_id"],
        "generation": str(generation.relative_to(root)).replace("\\", "/"),
        "completion": str(completion_path.relative_to(root)).replace("\\", "/"),
        "completion_sha256": sha256_file(completion_path),
        "p0_assets_sha256": completion["assets_sha256"],
    }
    atomic_write_json(root / "current_base.json", pointer)
    return pointer


def new_generation_staging(root: Path, generation_id: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    staging = root / "generations" / f".{generation_id}.{uuid.uuid4().hex}.pending"
    staging.mkdir(parents=True, exist_ok=False)
    return staging


def discard_staging(staging: Path) -> None:
    if staging.is_dir() and staging.name.startswith(".") and staging.name.endswith(".pending"):
        shutil.rmtree(staging)


__all__ = [
    "atomic_write_json", "discard_staging", "new_generation_staging", "publish_generation", "seal_p0",
    "sha256_file", "update_current_base", "verify_p0_completion", "write_csv", "write_image", "write_npz",
]
