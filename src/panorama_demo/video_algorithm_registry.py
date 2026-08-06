"""Role-safe resolver for baseline, candidate, and production algorithms."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .video_algorithm import AlgorithmRole, VideoAlgorithmSpec, build_algorithm_spec
from .video_algorithm_lock import VideoAlgorithmLockError, verify_algorithm_lock


class VideoAlgorithmRegistryError(ValueError):
    """Requested role/config combination violates the lifecycle contract."""


@dataclass(frozen=True)
class VideoAlgorithmRegistry:
    """Resolve only immutable locks for baseline/production.

    Candidate YAML is intentionally accepted only for the ``candidate`` role,
    preventing a production invocation from observing mutable experiment files.
    """

    baseline_lock: Path
    production_lock: Path

    def resolve(
        self,
        role: AlgorithmRole,
        *,
        candidate_config: str | Path | None = None,
    ) -> VideoAlgorithmSpec:
        if role == "candidate":
            if candidate_config is None:
                raise VideoAlgorithmRegistryError("candidate requires candidate_config")
            return build_algorithm_spec(candidate_config, expected_role="candidate")
        if candidate_config is not None:
            raise VideoAlgorithmRegistryError(f"{role} does not accept candidate_config")
        lock_path = self.baseline_lock if role == "baseline" else self.production_lock
        try:
            return verify_algorithm_lock(lock_path, expected_role=role)
        except VideoAlgorithmLockError as exc:
            raise VideoAlgorithmRegistryError(str(exc)) from exc


def resolve_video_algorithm(
    role: AlgorithmRole,
    *,
    baseline_lock: str | Path,
    production_lock: str | Path,
    candidate_config: str | Path | None = None,
) -> VideoAlgorithmSpec:
    """Convenience API for CLI/pipeline construction."""

    registry = VideoAlgorithmRegistry(
        baseline_lock=Path(baseline_lock), production_lock=Path(production_lock)
    )
    return registry.resolve(role, candidate_config=candidate_config)
