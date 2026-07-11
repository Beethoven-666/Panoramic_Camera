from __future__ import annotations

from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parents[1]
THIRD_PARTY_DIR = PROJECT_ROOT / "third_party"
UNISTITCH_DIR = THIRD_PARTY_DIR / "UniStitch"
UNISTITCH_CODES_DIR = UNISTITCH_DIR / "Codes"
LIGHTGLUE_DIR = THIRD_PARTY_DIR / "LightGlue"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "unistitch" / "epoch_best_model.pth"
