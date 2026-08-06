"""Development-only baseline/candidate entry point for the locked video run."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np

from .config import load_config
from .video_dataset_lock import (
    require_candidate_role_for_diagnostic_session,
    write_or_verify_experiment_dataset_lock,
)
from .video_observability import ObservabilitySpec
from .video_pipeline import run_video_algorithm
from .video_split import SPLIT_DEFINITION, write_or_verify_split


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a locked video baseline or candidate experiment")
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--algorithm", choices=("baseline", "candidate"), required=True)
    parser.add_argument("--candidate-config", type=Path)
    parser.add_argument("--report-level", choices=("summary", "full"), default="summary")
    parser.add_argument("--artifact-level", choices=("minimal", "provenance", "audit"), default="minimal")
    parser.add_argument("--maximum-post-seconds", type=float)
    parser.add_argument("--defer-3d", action="store_true")
    parser.add_argument(
        "--reuse-online-trajectory",
        action="store_true",
        help=(
            "Candidate-only: reuse the capture-bound, strictly verified online ORB "
            "trajectory instead of rerunning it."
        ),
    )
    parser.add_argument(
        "--trajectory-cache",
        type=Path,
        help="Verified real ORB trajectory cache produced by g305-video-freeze-trajectory.",
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--progress-range",
        metavar=("START", "END"),
        type=float,
        nargs=2,
        help="Restrict development work to a closed cumulative-motion interval in [0, 1].",
    )
    parser.add_argument(
        "--split",
        choices=("development", "validation"),
        help="Name the immutable non-holdout split containing --progress-range.",
    )
    return parser


def _seed() -> dict[str, object]:
    seed = 20_260_804
    random.seed(seed)
    np.random.seed(seed)
    state: dict[str, object] = {"seed": seed, "torch": "unavailable"}
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        state.update(
            {
                "torch": torch.__version__,
                "cuda_available": bool(torch.cuda.is_available()),
                "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
                "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
                "tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            }
        )
    except ImportError:
        pass
    return state


def _benchmark_root(session: Path) -> Path:
    """Keep experiment evidence isolated under its immutable source session."""

    return Path("benchmarks") / session.name


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.algorithm == "candidate" and args.candidate_config is None:
        raise ValueError("candidate requires --candidate-config")
    if args.algorithm == "baseline" and args.candidate_config is not None:
        raise ValueError("baseline does not accept --candidate-config")
    reuse_online_trajectory = bool(getattr(args, "reuse_online_trajectory", False))
    if reuse_online_trajectory and args.algorithm != "candidate":
        raise ValueError("--reuse-online-trajectory is candidate-only")
    progress_range = getattr(args, "progress_range", None)
    split = getattr(args, "split", None)
    if (progress_range is None) != (split is None):
        raise ValueError("--progress-range and --split must be provided together")
    if args.algorithm == "candidate" and progress_range is None:
        raise ValueError(
            "candidate experiments require an immutable non-holdout --split and --progress-range"
        )
    if progress_range is not None:
        requested = [float(progress_range[0]), float(progress_range[1])]
        legal = SPLIT_DEFINITION[split]
        if requested not in legal:
            raise ValueError(
                "--progress-range must exactly equal one immutable interval of the named split"
            )
    observe = ObservabilitySpec.from_values(
        report_level=args.report_level, artifact_level=args.artifact_level
    )
    root = args.input.expanduser().resolve()
    root = root if root.is_dir() else root.parent
    benchmark_root = _benchmark_root(root)
    require_candidate_role_for_diagnostic_session(root, args.algorithm)
    # The split is frozen independently for every capture.  In particular,
    # the diagnostic capture cannot inherit or mutate the old run's ledger.
    write_or_verify_split(benchmark_root / "split_definition.json")
    config = load_config(args.config)
    settings = dict(dict(config.get("stitch", {})).get("video_panorama", {}))
    if bool(settings.get("dataset_lock_required_for_experiments", True)):
        write_or_verify_experiment_dataset_lock(root, benchmark_root, role=args.algorithm)
    seed_state = _seed()
    report = run_video_algorithm(
        input_path=args.input,
        output=args.output,
        role=args.algorithm,
        candidate_config=args.candidate_config,
        config_path=args.config,
        observability=observe,
        maximum_post_seconds=args.maximum_post_seconds,
        defer_3d=args.defer_3d,
        reuse_online_trajectory=reuse_online_trajectory,
        trajectory_cache=getattr(args, "trajectory_cache", None),
        scan_progress_interval=(
            (float(progress_range[0]), float(progress_range[1]))
            if progress_range is not None
            else None
        ),
        evaluation_scope=(
            f"{split}_only" if split is not None else "exploratory_full_scan"
        ),
    )
    (args.output / "experiment_environment.json").write_text(
        json.dumps(seed_state, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report


def main() -> None:
    args = _parser().parse_args()
    try:
        report = run(args)
    except Exception as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    print(f"Video experiment: {report['panorama']}")
