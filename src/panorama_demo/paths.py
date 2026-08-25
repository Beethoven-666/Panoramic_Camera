from __future__ import annotations

from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
_SOURCE_ROOT = PACKAGE_DIR.parents[1]
_PACKAGED_RUNTIME_ROOT = PACKAGE_DIR / "_runtime"
PROJECT_ROOT = (
    _SOURCE_ROOT
    if (_SOURCE_ROOT / "configs" / "demo.yaml").is_file()
    else _PACKAGED_RUNTIME_ROOT
)
THIRD_PARTY_DIR = PROJECT_ROOT / "third_party"
UNISTITCH_DIR = THIRD_PARTY_DIR / "UniStitch"
UNISTITCH_CODES_DIR = UNISTITCH_DIR / "Codes"
LIGHTGLUE_DIR = THIRD_PARTY_DIR / "LightGlue"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "unistitch" / "epoch_best_model.pth"


def runtime_resource(relative_path: str | Path) -> Path:
    """Resolve a required source-checkout or wheel Runtime resource."""

    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Runtime resource path must be confined and relative")
    resolved = (PROJECT_ROOT / relative).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Packaged Runtime resource is missing: {relative}")
    return resolved
