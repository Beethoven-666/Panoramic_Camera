"""Checkpoint chain for committed online S013 P0 assignments."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import TYPE_CHECKING, Sequence

import numpy as np

if TYPE_CHECKING:
    from .video_s13_online_p0 import S13ScheduleSemanticAssignment


def array_sha256(value: np.ndarray, *, schema: str) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(schema.encode("ascii"))
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(str(tuple(int(item) for item in array.shape)).encode("ascii"))
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


@dataclass(frozen=True)
class S13AssignmentCheckpoint:
    assignment_index: int
    semantic: S13ScheduleSemanticAssignment
    image_roi_sha256: str
    valid_roi_sha256: str
    checkpoint_sha256: str


class S13OnlineCheckpointChain:
    def __init__(self) -> None:
        self._checkpoints: list[S13AssignmentCheckpoint] = []

    def advance(
        self,
        semantic: Sequence[S13ScheduleSemanticAssignment],
        renders: Sequence[object],
        sealed_source_count: int,
    ) -> None:
        render_by_index = {
            int(getattr(item, "assignment_index")): item for item in renders
        }
        while len(self._checkpoints) < int(sealed_source_count):
            index = len(self._checkpoints)
            rendered = render_by_index.get(index)
            image_sha = (
                "zero-width"
                if rendered is None else array_sha256(
                    np.asarray(getattr(rendered, "image_roi")), schema="s013-p0-roi/v1"
                )
            )
            valid_sha = (
                "zero-width"
                if rendered is None else array_sha256(
                    np.asarray(getattr(rendered, "valid_roi")), schema="s013-p0-valid/v1"
                )
            )
            payload = {
                "previous": (
                    "" if not self._checkpoints
                    else self._checkpoints[-1].checkpoint_sha256
                ),
                "assignment_index": index,
                "semantic": self._semantic_payload(semantic[index]),
                "image_roi_sha256": image_sha,
                "valid_roi_sha256": valid_sha,
            }
            checkpoint_sha = hashlib.sha256(
                json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            self._checkpoints.append(S13AssignmentCheckpoint(
                assignment_index=index,
                semantic=semantic[index],
                image_roi_sha256=image_sha,
                valid_roi_sha256=valid_sha,
                checkpoint_sha256=checkpoint_sha,
            ))

    @staticmethod
    def _semantic_payload(value: S13ScheduleSemanticAssignment) -> dict[str, object]:
        return {
            "frame_id": value.frame_id,
            "quantized_center_x": value.quantized_center_x,
            "left_x": value.left_x,
            "right_x": value.right_x,
            "placement_method": value.placement_method,
            "selected_hypothesis_id": value.selected_hypothesis_id,
        }

    def longest_consistent_prefix(
        self,
        final_semantic: Sequence[S13ScheduleSemanticAssignment],
        renders: Sequence[object],
    ) -> int:
        render_by_index = {
            int(getattr(item, "assignment_index")): item for item in renders
        }
        consistent = 0
        for checkpoint in self._checkpoints:
            index = checkpoint.assignment_index
            if index >= len(final_semantic) or checkpoint.semantic != final_semantic[index]:
                break
            rendered = render_by_index.get(index)
            if rendered is None:
                if (
                    checkpoint.image_roi_sha256 == "zero-width"
                    and checkpoint.valid_roi_sha256 == "zero-width"
                ):
                    consistent += 1
                    continue
                break
            if checkpoint.image_roi_sha256 != array_sha256(
                np.asarray(getattr(rendered, "image_roi")), schema="s013-p0-roi/v1"
            ):
                break
            if checkpoint.valid_roi_sha256 != array_sha256(
                np.asarray(getattr(rendered, "valid_roi")), schema="s013-p0-valid/v1"
            ):
                break
            consistent += 1
        return consistent

    @property
    def checkpoints(self) -> tuple[S13AssignmentCheckpoint, ...]:
        return tuple(self._checkpoints)

    @property
    def chain_sha256(self) -> str:
        return "" if not self._checkpoints else self._checkpoints[-1].checkpoint_sha256


__all__ = [
    "S13AssignmentCheckpoint",
    "S13OnlineCheckpointChain",
    "array_sha256",
]
