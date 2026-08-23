"""Compare c3aaa7dd and current M6 on one portable post-P2 fixture."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys


OLD_COMMIT = "c3aaa7ddbaf8aac5e00cded71ba9e4f1689b822a"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--legacy-worktree", type=Path)
    parser.add_argument("--worker", choices=("old", "current"))
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--current-config", type=Path)
    return parser


def _luminance_stats(image, valid):
    import numpy as np

    rgb = np.asarray(image, np.float32)[valid][:, ::-1] / 255.0
    luma = rgb @ np.asarray([0.2126, 0.7152, 0.0722], np.float32)
    return {
        "mean": float(np.mean(luma)),
        "p10": float(np.quantile(luma, 0.10)),
        "p50": float(np.quantile(luma, 0.50)),
        "p90": float(np.quantile(luma, 0.90)),
        "p99": float(np.quantile(luma, 0.99)),
        "p995": float(np.quantile(luma, 0.995)),
        "clipped_fraction": float(np.mean(np.max(rgb, axis=1) >= 1.0)),
    }


def _worker(args: argparse.Namespace) -> int:
    import cv2
    import numpy as np

    source_root = args.source_root.expanduser().resolve()
    sys.path.insert(0, str(source_root / "src"))
    manifest = json.loads((args.fixture / "manifest.json").read_text(encoding="utf-8"))
    image = cv2.imread(str(args.fixture / manifest["p2_image"]), cv2.IMREAD_COLOR)
    with np.load(args.fixture / manifest["p2_maps"], allow_pickle=False) as stored:
        provenance = {name: np.asarray(stored[name]).copy() for name in stored.files}
    from panorama_demo.video_s13_replay import S13P2ReplayPair, S13VerifiedP2

    pairs = []
    old_fields = tuple(S13P2ReplayPair.__dataclass_fields__)
    for relative in manifest["pairs"]:
        with np.load(args.fixture / relative, allow_pickle=False) as stored:
            arrays = {name: np.asarray(stored[name]).copy() for name in stored.files}
        values = {}
        for name in old_fields:
            if name not in arrays:
                continue
            value = arrays[name]
            if value.shape == ():
                value = value.item()
            values[name] = value
        pairs.append(S13P2ReplayPair(**values))
    valid = np.asarray(provenance["valid"], bool)
    p2 = S13VerifiedP2(
        root=args.fixture,
        completion={"source_count": int(manifest["source_count"])},
        completion_sha256="",
        result_image=image,
        valid_mask=valid,
        provenance=provenance,
        transactions=(),
        replay_pairs=tuple(pairs),
        immutable_sha256={},
    )
    by_frame = {}
    session_root = Path(manifest["session_path"])
    with (session_root / "frames.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            by_frame[int(row["frame_id"])] = session_root / row["color_path"]
    cache = {}

    def image_loader(frame_id: int):
        if frame_id not in cache:
            cache[frame_id] = cv2.imread(str(by_frame[frame_id]), cv2.IMREAD_COLOR)
            if cache[frame_id] is None:
                raise ValueError(f"unreadable fixture RGB frame: {frame_id}")
        return cache[frame_id]

    if args.worker == "old":
        from panorama_demo.video_s13_m6 import run_s13_m6

        result = run_s13_m6(p2, image_loader)
    else:
        from panorama_demo.video_s13_m62_runner import run_s13_m62
        from panorama_demo.video_s13_m63_solver import S13M63Config

        m63_config = S13M63Config()
        if args.current_config is not None:
            import yaml

            document = yaml.safe_load(args.current_config.read_text(encoding="utf-8"))
            m63_config = S13M63Config.from_document(document.get("m63_photometric"))

        result, _details = run_s13_m62(
            p2,
            image_loader,
            execution_mode="candidate_single_pass",
            force_owner_only_pair_indices=frozenset(
                int(value) for value in manifest.get("force_owner_only_pair_indices", ())
            ),
            m63_config=m63_config,
        )
    p3 = np.asarray(result.visual_panorama)
    difference = np.abs(p3.astype(np.int16) - image.astype(np.int16))
    parameters = result.photometric_solution.source_parameters
    report = {
        "engine": args.worker,
        "source_root": str(source_root),
        "model": result.photometric_solution.model_family,
        "source_count": len(parameters),
        "pair_count": len(pairs),
        "canvas_shape": list(valid.shape),
        "source_parameters": [
            {
                "source_index": item.source_index,
                "frame_id": item.frame_id,
                "gain_bgr": list(item.gain_bgr),
                "bias_bgr": list(item.bias_bgr),
                "fallback_reason": item.fallback_reason,
            }
            for item in parameters
        ],
        "fallback_source_count": sum(
            item.fallback_reason not in (None, "identity_candidate") for item in parameters
        ),
        "p2_identity": bool(np.array_equal(np.asarray(p2.result_image), image)),
        "p3_vs_p2": {
            "differing_pixel_count": int(np.count_nonzero(np.any(difference != 0, axis=2))),
            "differing_channel_count": int(np.count_nonzero(difference)),
            "maximum_absolute_dn": int(difference.max(initial=0)),
        },
        "p2_luminance": _luminance_stats(image, valid),
        "p3_luminance": _luminance_stats(p3, valid),
        "heldout": dict(result.photometric_solution.heldout_audit),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output / f"{args.worker}_P3.png"), p3)
    (args.output / f"{args.worker}_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


def _orchestrate(args: argparse.Namespace) -> int:
    if args.legacy_worktree is None:
        raise ValueError("--legacy-worktree is required")
    output = args.output.expanduser().resolve()
    fixture = args.fixture.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    current_root = Path(__file__).resolve().parents[1]
    for engine, source_root in (("old", args.legacy_worktree), ("current", current_root)):
        environment = os.environ.copy()
        if engine == "old":
            environment["G305_CUDA"] = "off"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--fixture",
            str(fixture),
            "--output",
            str(output),
            "--worker",
            engine,
            "--source-root",
            str(source_root),
        ]
        if engine == "current" and args.current_config is not None:
            command.extend(("--current-config", str(args.current_config.expanduser().resolve())))
        subprocess.run(
            command,
            check=True,
            env=environment,
        )
    old = json.loads((output / "old_report.json").read_text(encoding="utf-8"))
    current = json.loads((output / "current_report.json").read_text(encoding="utf-8"))
    if any(
        old[key] != current[key]
        for key in ("source_count", "pair_count", "canvas_shape", "p2_identity", "p2_luminance")
    ):
        raise ValueError("same-P2 comparison engines did not consume identical P2 state")
    comparison = {
        "schema": "gemini305-video-s13-m6-same-p2-comparison/v1",
        "old_commit": OLD_COMMIT,
        "fixture": str(fixture),
        "old": old,
        "current": current,
    }
    (output / "comparison.json").write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


def main() -> int:
    args = _parser().parse_args()
    return _worker(args) if args.worker else _orchestrate(args)


if __name__ == "__main__":
    raise SystemExit(main())
