"""Independent diagnostic entry point for real-pose central-strip rendering.

This module deliberately owns the only import edge to ``central_strip``.  The
formal sequence entry point receives the renderer as a callback and therefore
cannot discover, select, or publish this experimental backend by itself.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from . import stitch_sequence


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a diagnostic-only calibrated central-strip panorama from a "
            "strict Gemini 305 RGB-D session"
        )
    )
    parser.add_argument("input", type=Path, help="Calibrated RGB-D capture session")
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/central_strip_diagnostic")
    )
    parser.add_argument("--config", type=Path)
    return parser


def _central_renderer(**kwargs: Any) -> object:
    """Load the optional backend only after sequence failure handling is active."""

    from .central_strip import render_central_strip_diagnostic

    return render_central_strip_diagnostic(**kwargs)


def main() -> None:
    args = _parser().parse_args()
    try:
        report = stitch_sequence.run(args, diagnostic_renderer=_central_renderer)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print(f"Diagnostic panorama: {report['panorama']}")
    print(f"Diagnostic report: {report['report']}")
    print("Diagnostic only: no delivery.json was published")


if __name__ == "__main__":
    main()
