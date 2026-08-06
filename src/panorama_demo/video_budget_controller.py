"""Ordered, auditable degradation policy for the video production budget."""

from __future__ import annotations

from dataclasses import dataclass, field


_DEGRADATIONS = (
    "raft_backward_risk_only",
    "normal_mesh_cell_32px",
    "multiband_levels_3",
    "disable_low_frequency_straightening",
    "normal_step_14px",
    "defer_complete_open3d_audit",
)


@dataclass
class VideoBudgetController:
    requested_level: int = 0
    applied_level: int = 0
    changes: list[str] = field(default_factory=list)

    def tighten(self) -> bool:
        if self.applied_level >= len(_DEGRADATIONS):
            return False
        change = _DEGRADATIONS[self.applied_level]
        self.changes.append(change)
        self.applied_level += 1
        return True

    def as_dict(self) -> dict[str, object]:
        return {
            "requested_level": self.requested_level,
            "applied_level": self.applied_level,
            "changes": list(self.changes),
        }
