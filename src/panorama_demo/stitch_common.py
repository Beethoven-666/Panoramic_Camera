from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from .paths import PROJECT_ROOT

if TYPE_CHECKING:
    from .unistitch_adapter import UniStitchAligner


def read_bgr(path: str | Path) -> np.ndarray:
    image_path = Path(path).expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    encoded = np.fromfile(image_path, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise OSError(f"OpenCV could not decode image: {image_path}")
    return image


def write_bgr(path: str | Path, image: np.ndarray, *, jpeg_quality: int = 95) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    extension = destination.suffix.lower()
    parameters: list[int] = []
    if extension in {".jpg", ".jpeg"}:
        parameters = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
    success, encoded = cv2.imencode(extension, image, parameters)
    if not success:
        raise OSError(f"OpenCV could not encode image: {destination}")
    encoded.tofile(destination)
    return destination


def resolve_model_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def build_aligner(
    stitch_config: dict[str, Any], **overrides: Any
) -> "UniStitchAligner":
    from .unistitch_adapter import UniStitchAligner

    settings = dict(stitch_config)
    settings.update({key: value for key, value in overrides.items() if value is not None})
    return UniStitchAligner(
        resolve_model_path(settings["model"]),
        device=str(settings.get("device", "cuda:0")),
        inference_width=int(settings.get("inference_width", 640)),
        max_points=int(settings.get("max_points", 2048)),
        min_matches=int(settings.get("min_matches", 40)),
        max_pair_canvas=int(settings.get("max_pair_canvas", 4000)),
        max_unistitch_reprojection_px=float(
            settings.get("max_unistitch_reprojection_px", 20.0)
        ),
        allow_magsac_fallback=bool(settings.get("allow_magsac_fallback", True)),
        prefer_magsac_layout=bool(settings.get("prefer_magsac_layout", True)),
        min_magsac_inlier_ratio=float(
            settings.get("min_magsac_inlier_ratio", 0.5)
        ),
    )
