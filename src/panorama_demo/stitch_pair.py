from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .config import load_config
from .stitch_common import build_aligner, read_bgr, write_bgr
from .unistitch_adapter import AlignmentError


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stitch one image pair with the official UniStitch model"
    )
    parser.add_argument("reference", type=Path, help="Reference/previous image")
    parser.add_argument("source", type=Path, help="Source/next image")
    parser.add_argument("--output", type=Path, default=Path("outputs/pair"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--inference-width", type=int)
    parser.add_argument(
        "--strict-unistitch",
        action="store_true",
        help="Reject the pair instead of using MAGSAC when the global branch fails validation",
    )
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    config = load_config(args.config)
    stitch_config = dict(config["stitch"])
    if args.strict_unistitch:
        stitch_config["allow_magsac_fallback"] = False
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    reference = read_bgr(args.reference)
    source = read_bgr(args.source)
    if reference.shape != source.shape:
        raise ValueError(
            "Pair images must have identical dimensions; "
            f"got {reference.shape} and {source.shape}"
        )

    started = time.perf_counter()
    aligner = build_aligner(
        stitch_config,
        model=args.model,
        device=args.device,
        inference_width=args.inference_width,
    )
    alignment = aligner.align(reference, source)
    elapsed = time.perf_counter() - started

    preview_path = write_bgr(output / "pair_unistitch.jpg", alignment.preview_bgr)
    report: dict[str, object] = {
        "schema": "gemini305-unistitch-pair/v1",
        "reference": str(args.reference.expanduser().resolve()),
        "source": str(args.source.expanduser().resolve()),
        "elapsed_seconds_including_model_load": elapsed,
        "pair_preview": str(preview_path),
        "depth_used": False,
        **alignment.as_dict(),
    }
    report_path = output / "pair_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    args = _parser().parse_args()
    try:
        report = run(args)
    except (AlignmentError, FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print(f"Pair preview: {report['pair_preview']}")
    print(f"Layout: {report['layout_method']}")
    print(f"Report: {args.output.expanduser().resolve() / 'pair_report.json'}")


if __name__ == "__main__":
    main()
