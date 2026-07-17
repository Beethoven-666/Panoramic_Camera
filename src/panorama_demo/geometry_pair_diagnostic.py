"""Independent RGB-only A/B diagnostic entry point for one adjacent seam.

This module is intentionally only a thin CLI/callback wrapper.  It neither
loads a historical pose sidecar nor runs ORB-SLAM3 itself: the sequence
orchestrator owns strict-session loading, the complete current RGB-D pose
chain, failure handling, and the two-file diagnostic publication contract.

The callback receives the *full* current render chain.  ``pair_index`` merely
selects the adjacent seam to inspect after the helper has rendered with the
complete real-pose layout; it must never turn the diagnostic into a two-frame
trajectory or scale estimate.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Mapping, Sequence

from . import stitch_sequence

if TYPE_CHECKING:
    import numpy as np

    from .session import CameraIntrinsics, RGBDFrame


def _nonnegative_pair_index(value: str) -> int:
    """Parse an adjacent-pair position without accepting negative indices."""

    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("pair index must be an integer") from exc
    if result < 0:
        raise argparse.ArgumentTypeError("pair index must be non-negative")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a diagnostic-only RGB A/B view for one adjacent seam from "
            "a strict Gemini 305 RGB-D session"
        )
    )
    parser.add_argument("input", type=Path, help="Calibrated RGB-D capture session")
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/geometry_pair_diagnostic")
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--pair-index",
        type=_nonnegative_pair_index,
        default=48,
        help=(
            "Zero-based adjacent pair position in the complete current render "
            "chain (default: 48, inspecting nodes 48 and 49)"
        ),
    )
    return parser


def _geometry_pair_renderer(
    *,
    render_frames: Sequence[RGBDFrame],
    render_poses: Sequence[np.ndarray],
    calibration: CameraIntrinsics,
    config: Mapping[str, object],
    rgb_motions: Sequence[object] | None,
    motion_pixels_to_full_resolution: float,
    multiband_levels: int,
    pair_index: int,
) -> object:
    """Delegate one full-layout A/B render to the calibrated helper.

    The delayed import keeps the formal sequence module free from a diagnostic
    entry-point edge.  The helper is responsible for the paired
    baseline/candidate render and returns the renderer-style result consumed
    by ``stitch_sequence``'s diagnostic publication path.
    """

    from .calibrated_rgb_pushbroom import render_geometry_pair_diagnostic

    return render_geometry_pair_diagnostic(
        frames=render_frames,
        poses=render_poses,
        calibration=calibration,
        config=config,
        rgb_motions=rgb_motions,
        motion_pixels_to_full_resolution=motion_pixels_to_full_resolution,
        multiband_levels=multiband_levels,
        pair_index=pair_index,
    )


def main() -> None:
    args = _parser().parse_args()
    try:
        # ``geometry_pair_diagnostic_renderer`` is a deliberately separate
        # injection seam from the reference-plane central-strip callback.  The
        # sequence route will force diagnostic-only publication and insist on
        # the complete current ORB-SLAM3 trajectory before calling us.
        report = stitch_sequence.run(
            args,
            geometry_pair_diagnostic_renderer=_geometry_pair_renderer,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print(f"Diagnostic panorama: {report['panorama']}")
    print(f"Diagnostic report: {report['report']}")
    print("Diagnostic only: no delivery.json was published")


if __name__ == "__main__":
    main()
