"""Bounded asynchronous writer for the four S1.3 stage snapshots."""

from __future__ import annotations

import os
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np


STAGE_FILENAMES = {
    "P0": "P0_base_panorama.png",
    "P1": "P1_vertical_panorama.png",
    "P2": "P2_geometry_and_seam_panorama.png",
    "P3": "P3_visual_panorama.png",
}


class S13StageImageWriter:
    """Encode one lossless PNG per stage without competing with compute threads."""

    def __init__(
        self, output_root: Path, *, png_compression: int = 0, max_pending: int = 4,
    ) -> None:
        self.output_root = Path(output_root)
        self.png_compression = int(png_compression)
        self.max_pending = int(max_pending)
        if self.max_pending < 1:
            raise ValueError("S1.3 writer max_pending must be positive")
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="s13-png")
        self._pending: list[Future[Path]] = []
        self._written: list[Path] = []
        self.submit_wait_seconds = 0.0
        self.close_wait_seconds = 0.0
        self.submit_blocking_count = 0
        self.pending_peak = 0
        self.snapshot_copy_count = 0

    def _drain_if_full(self) -> None:
        if len(self._pending) < self.max_pending:
            return
        started = time.perf_counter()
        self._written.append(self._pending.pop(0).result())
        self.submit_wait_seconds += time.perf_counter() - started
        self.submit_blocking_count += 1

    @staticmethod
    def _validate(stage: str, image: np.ndarray) -> None:
        if stage not in STAGE_FILENAMES:
            raise ValueError(f"Unsupported S1.3 stage: {stage}")
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError("S1.3 stage PNG requires uint8 BGR image")

    def submit_host_image(self, stage: str, image: np.ndarray) -> None:
        self._validate(stage, image)
        # A bounded queue preserves the intended compute/encode overlap without
        # retaining unbounded full-resolution host copies.
        self._drain_if_full()
        snapshot = np.ascontiguousarray(image).copy()
        self.snapshot_copy_count += 1
        snapshot.setflags(write=False)
        self._pending.append(self._executor.submit(self._write_png, stage, snapshot))
        self.pending_peak = max(self.pending_peak, len(self._pending))

    def submit_owned_host_image(self, stage: str, image: np.ndarray) -> None:
        """Queue an immutable, contiguous buffer whose ownership is transferred.

        CUDA stage downloads use this to avoid a second full-resolution copy.
        The caller must not mutate the buffer after submission.
        """
        self._validate(stage, image)
        if not image.flags.c_contiguous:
            raise ValueError("Owned S1.3 stage image must be contiguous")
        if image.flags.writeable:
            raise ValueError("Owned S1.3 stage image must be immutable")
        self._drain_if_full()
        self._pending.append(self._executor.submit(self._write_png, stage, image))
        self.pending_peak = max(self.pending_peak, len(self._pending))

    def _write_png(self, stage: str, image: np.ndarray) -> Path:
        self.output_root.mkdir(parents=True, exist_ok=True)
        final = self.output_root / STAGE_FILENAMES[stage]
        pending = final.with_name(f".{final.name}.{uuid.uuid4().hex}.tmp")
        ok, encoded = cv2.imencode(
            ".png", image, [cv2.IMWRITE_PNG_COMPRESSION, self.png_compression]
        )
        if not ok:
            raise OSError(f"Failed to encode {stage} PNG")
        try:
            pending.write_bytes(encoded.tobytes())
            os.replace(pending, final)
        finally:
            pending.unlink(missing_ok=True)
        return final

    def flush(self) -> tuple[Path, ...]:
        started = time.perf_counter()
        self._written.extend(future.result() for future in self._pending)
        self.close_wait_seconds += time.perf_counter() - started
        self._pending.clear()
        paths = tuple(self._written)
        self._written.clear()
        return paths

    def close(self) -> tuple[Path, ...]:
        try:
            return self.flush()
        finally:
            self._executor.shutdown(wait=True)


__all__ = ["S13StageImageWriter", "STAGE_FILENAMES"]
