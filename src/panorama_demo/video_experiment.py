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
    write_or_verify_v6_tracking_gate_dataset_lock,
    write_or_verify_experiment_dataset_lock,
)
from .video_observability import ObservabilitySpec
from .video_algorithm import build_algorithm_spec, load_algorithm_config
from .video_pipeline import run_video_algorithm
from .video_panorama import run_direct_orb_tracking_gate
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
    parser.add_argument(
        "--run-offline-orb",
        action="store_true",
        help="S1.3 candidate only: explicitly run a new complete offline ORB trajectory.",
    )
    parser.add_argument(
        "--ignore-pose",
        action="store_true",
        help="S1.3 candidate only: do not read any trajectory and use RGB evidence alone.",
    )
    parser.add_argument(
        "--m62-warmup",
        action="store_true",
        help="Mark this S1.3 M6.2 candidate run as benchmark warm-up only.",
    )
    parser.add_argument(
        "--m62-execution-mode",
        choices=("reference", "shadow_audit", "candidate_single_pass", "parity_test"),
        help="S1.3 M6.2 only: override the configured execution mode for a dedicated audit.",
    )
    parser.add_argument(
        "--simulate-optimizer-all-fail",
        action="store_true",
        help="S1.3 test hook: fail all post-P0 optimizers after the immutable base seal.",
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--tracking-gate-only",
        action="store_true",
        help="Run v6 T0/T1/T2 direct-ORB tracking only; do not render a panorama.",
    )
    parser.add_argument(
        "--tracking-fps-candidates",
        type=float,
        nargs="+",
        default=(8.0, 12.0, 16.0),
        metavar="FPS",
        help="Increasing direct-ORB tracking rates for --tracking-gate-only (default: 8 12 16).",
    )
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
    tracking_gate_only = bool(getattr(args, "tracking_gate_only", False))
    if args.algorithm == "candidate" and args.candidate_config is None:
        raise ValueError("candidate requires --candidate-config")
    if args.algorithm == "baseline" and args.candidate_config is not None:
        raise ValueError("baseline does not accept --candidate-config")
    run_offline_orb = bool(getattr(args, "run_offline_orb", False))
    ignore_pose = bool(getattr(args, "ignore_pose", False))
    simulate_optimizer_all_fail = bool(getattr(args, "simulate_optimizer_all_fail", False))
    s13_spec = None
    if args.algorithm == "candidate" and args.candidate_config is not None:
        from .video_s13_contract import (
            S13_ALGORITHM_ID,
            S13_FORMAL_M6_ALGORITHM_ID,
            claims_s13_document,
            is_s13_identity,
            load_s13_config,
        )

        candidate_path = Path(args.candidate_config).expanduser().resolve()
        try:
            candidate_text = candidate_path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            candidate_text = ""
        try:
            candidate_document = load_algorithm_config(candidate_path)
        except (OSError, UnicodeError, ValueError):
            candidate_document = {}
        claims_s13 = claims_s13_document(candidate_document) or any(
            marker in candidate_text
            for marker in (
                S13_ALGORITHM_ID,
                S13_FORMAL_M6_ALGORITHM_ID,
                "s013_output_first_progressive_dense_central_slit:",
            )
        )
        if claims_s13:
            # The sibling manifest, self hashes, and exact identity are all
            # verified before any shared lock, facade, renderer, or publisher.
            s13_spec = build_algorithm_spec(args.candidate_config, expected_role="candidate")
            load_s13_config(args.candidate_config)
            if not is_s13_identity(
                algorithm_id=s13_spec.algorithm_id,
                implementation_id=s13_spec.implementation_id,
                role=s13_spec.role,
            ):
                raise ValueError("S1.3 claimed identity is incomplete")
    reuse_online_trajectory = bool(getattr(args, "reuse_online_trajectory", False))
    if reuse_online_trajectory and args.algorithm != "candidate":
        raise ValueError("--reuse-online-trajectory is candidate-only")
    progress_range = getattr(args, "progress_range", None)
    split = getattr(args, "split", None)
    if (progress_range is None) != (split is None):
        raise ValueError("--progress-range and --split must be provided together")
    if args.algorithm == "candidate" and progress_range is None and s13_spec is None:
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
    if tracking_gate_only:
        if args.algorithm != "baseline" or args.candidate_config is not None:
            raise ValueError("--tracking-gate-only requires --algorithm baseline without --candidate-config")
        if reuse_online_trajectory or getattr(args, "trajectory_cache", None) is not None:
            raise ValueError("--tracking-gate-only always reruns direct ORB and cannot reuse a trajectory")
        if progress_range is not None:
            raise ValueError("--tracking-gate-only requires the complete real scan")
    if s13_spec is not None:
        from .video_s13_experiment import run_s13_experiment

        report = run_s13_experiment(
            input_path=args.input,
            output=args.output,
            candidate_config=args.candidate_config,
            algorithm_spec=s13_spec,
            trajectory_cache=getattr(args, "trajectory_cache", None),
            reuse_online_trajectory=reuse_online_trajectory,
            run_offline_orb=run_offline_orb,
            ignore_pose=ignore_pose,
            config_path=getattr(args, "config", None),
            simulate_optimizer_all_fail=simulate_optimizer_all_fail,
            m62_warmup=bool(getattr(args, "m62_warmup", False)),
            m62_execution_mode=getattr(args, "m62_execution_mode", None),
        )
        return report
    if run_offline_orb or ignore_pose or simulate_optimizer_all_fail or getattr(args, "m62_execution_mode", None):
        raise ValueError("S1.3-only flags cannot be used by legacy baseline/candidates")
    observe = ObservabilitySpec.from_values(
        report_level=args.report_level, artifact_level=args.artifact_level
    )
    root = args.input.expanduser().resolve()
    root = root if root.is_dir() else root.parent
    benchmark_root = _benchmark_root(root)
    if tracking_gate_only:
        write_or_verify_v6_tracking_gate_dataset_lock(root, benchmark_root)
        baseline_document = load_algorithm_config(
            Path(__file__).resolve().parents[2]
            / "configs"
            / "video_algorithms"
            / "baseline_legacy_fast_b07b561.yaml"
        )
        baseline_settings = baseline_document.get("legacy_video_panorama")
        if not isinstance(baseline_settings, dict):
            raise ValueError("Frozen baseline lacks legacy_video_panorama settings")
        fast_orb = baseline_settings.get("fast_orbslam3_rgbd")
        if not isinstance(fast_orb, dict):
            raise ValueError("Frozen baseline lacks fast_orbslam3_rgbd settings")
        return run_direct_orb_tracking_gate(
            input_path=args.input,
            output=args.output,
            fps_candidates=tuple(float(value) for value in args.tracking_fps_candidates),
            config_path=args.config,
            fast_orbslam3_config=fast_orb,
        )
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
    panorama = report.get("panorama")
    if report.get("final_stage") == "P3" and isinstance(panorama, str):
        for path in report.get("stage_images", ()):  # stdout, not an artifact
            print(f"S1.3 stage: {path}")
        selected = report.get("c2e", {}).get("selected", ())
        changed = sum(str(value) != "C0_keep_standard" for value in selected)
        print(f"C2E automatic: enabled ({len(selected)} pairs, {changed} changed)")
        timings = report.get("timings", {})
        print(f"S1.3 timings: {timings}")
        print("Final stage: P3")
        return
    if isinstance(panorama, str):
        print(f"Video experiment: {panorama}")
        return
    if report.get("schema") == "gemini305-video-direct-orb-tracking-gate/v1":
        print(
            "Direct ORB tracking gate: "
            f"{report.get('selected_tracking_candidate_id') or 'no_survivor'}"
        )
        return
    raise SystemExit("ERROR: experiment returned neither a panorama nor a tracking-gate report")
