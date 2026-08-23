"""Bounded runtime frame/analysis cache shared by S1.3 M0--M6."""

from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from threading import RLock
from typing import Mapping, Sequence

import cv2
import numpy as np

from .video_s13_session import S13RenderFrame, read_s13_rgb


@dataclass(frozen=True)
class S13FrameStoreReport:
    raw_decode_count: int
    raw_hit_count: int
    gray_build_count: int
    gray_hit_count: int
    gradient_build_count: int
    gradient_hit_count: int
    eviction_count: int
    resident_bytes: int
    peak_resident_bytes: int
    prefetch_workers: int
    adopted_raw_count: int
    adopted_raw_bytes: int
    bulk_adopt_count: int
    retained_raw_count_after_schedule: int
    released_unselected_raw_count: int
    raw_decode_after_adopt_count: int
    raw_bytes_after_schedule_retain: int


class S13FrameStore:
    def __init__(self, *, maximum_bytes: int = 512 * 1024 * 1024) -> None:
        if maximum_bytes <= 0:
            raise ValueError("S1.3 FrameStore maximum bytes must be positive")
        self.maximum_bytes = int(maximum_bytes)
        self._raw: OrderedDict[int, np.ndarray] = OrderedDict()
        self._gray: dict[tuple[int, int], tuple[np.ndarray, float]] = {}
        self._gradient: dict[tuple[int, int], np.ndarray] = {}
        self._lock = RLock()
        self._resident_bytes = 0
        self._raw_bytes = 0
        self._peak_bytes = 0
        self._raw_decode_count = 0
        self._raw_hit_count = 0
        self._gray_build_count = 0
        self._gray_hit_count = 0
        self._gradient_build_count = 0
        self._gradient_hit_count = 0
        self._eviction_count = 0
        self._prefetch_workers = 1
        self._adopted_raw_count = 0
        self._adopted_raw_bytes = 0
        self._bulk_adopt_count = 0
        self._retained_raw_count_after_schedule = 0
        self._released_unselected_raw_count = 0
        self._raw_decode_after_adopt_count = 0
        self._raw_bytes_after_schedule_retain = 0

    def _note_array(self, array: np.ndarray) -> None:
        self._resident_bytes += int(array.nbytes)
        self._peak_bytes = max(self._peak_bytes, self._resident_bytes)

    def _evict_raw(self) -> None:
        while self._raw_bytes > self.maximum_bytes and self._raw:
            _frame_id, image = self._raw.popitem(last=False)
            self._resident_bytes -= int(image.nbytes)
            self._raw_bytes -= int(image.nbytes)
            self._eviction_count += 1

    def raw_bgr(self, frame: S13RenderFrame) -> np.ndarray:
        with self._lock:
            cached = self._raw.get(frame.frame_id)
            if cached is not None:
                self._raw.move_to_end(frame.frame_id)
                self._raw_hit_count += 1
                return cached
        image = read_s13_rgb(frame)
        image.setflags(write=False)
        with self._lock:
            existing = self._raw.get(frame.frame_id)
            if existing is not None:
                self._raw.move_to_end(frame.frame_id)
                self._raw_hit_count += 1
                return existing
            self._raw[frame.frame_id] = image
            self._raw_decode_count += 1
            if self._bulk_adopt_count:
                self._raw_decode_after_adopt_count += 1
            self._note_array(image)
            self._raw_bytes += int(image.nbytes)
            self._evict_raw()
            return image

    def adopt_validated_raw_bulk(
        self,
        frames_by_id: Mapping[int, S13RenderFrame],
        images_by_frame_id: dict[int, np.ndarray],
    ) -> None:
        """Take ownership of one complete validated RGB set without copying pixels."""

        if set(images_by_frame_id) != set(frames_by_id):
            raise ValueError("S1.3 validated RGB handoff frame ids do not match the session")
        prepared: list[tuple[int, np.ndarray]] = []
        total_bytes = 0
        with self._lock:
            duplicate_ids = set(images_by_frame_id).intersection(self._raw)
            if duplicate_ids:
                raise ValueError("S1.3 FrameStore cannot adopt a cached frame twice")
            for frame_id, image in images_by_frame_id.items():
                frame = frames_by_id[frame_id]
                if image.dtype != np.uint8 or image.shape != (frame.height, frame.width, 3):
                    raise ValueError("S1.3 validated RGB handoff dtype or shape changed")
                if not image.flags.c_contiguous:
                    raise ValueError("S1.3 validated RGB handoff must be C contiguous")
                prepared.append((frame_id, image))
                total_bytes += int(image.nbytes)
            if total_bytes > self.maximum_bytes:
                raise ValueError("S1.3 FrameStore raw budget cannot retain the validated handoff")
            for _frame_id, image in prepared:
                image.setflags(write=False)
            for frame_id, image in prepared:
                self._raw[frame_id] = image
            self._resident_bytes += total_bytes
            self._raw_bytes += total_bytes
            self._peak_bytes = max(self._peak_bytes, self._resident_bytes)
            self._adopted_raw_count += len(prepared)
            self._adopted_raw_bytes += total_bytes
            self._bulk_adopt_count += 1
            images_by_frame_id.clear()

    def retain_raw(self, frame_ids: set[int]) -> None:
        """Drop unselected raw sources after M3 without disturbing analysis caches."""

        with self._lock:
            missing = frame_ids.difference(self._raw)
            if missing:
                raise ValueError("S1.3 FrameStore cannot retain missing raw sources")
            retained: OrderedDict[int, np.ndarray] = OrderedDict()
            released_count = 0
            released_bytes = 0
            for frame_id, image in self._raw.items():
                if frame_id in frame_ids:
                    retained[frame_id] = image
                else:
                    released_count += 1
                    released_bytes += int(image.nbytes)
            self._raw = retained
            self._resident_bytes -= released_bytes
            self._raw_bytes -= released_bytes
            self._released_unselected_raw_count += released_count
            self._retained_raw_count_after_schedule = len(retained)
            self._raw_bytes_after_schedule_retain = self._raw_bytes

    def analysis_gray(self, frame: S13RenderFrame, width: int) -> tuple[np.ndarray, float]:
        key = (frame.frame_id, int(width))
        with self._lock:
            cached = self._gray.get(key)
            if cached is not None:
                self._gray_hit_count += 1
                return cached
        image = self.raw_bgr(frame)
        scale = min(1.0, float(width) / image.shape[1])
        analysis = image
        if scale < 1.0:
            analysis = cv2.resize(
                image,
                (
                    max(1, int(round(image.shape[1] * scale))),
                    max(1, int(round(image.shape[0] * scale))),
                ),
                interpolation=cv2.INTER_AREA,
            )
        gray = cv2.cvtColor(analysis, cv2.COLOR_BGR2GRAY)
        gray.setflags(write=False)
        value = (gray, 1.0 / scale)
        with self._lock:
            existing = self._gray.get(key)
            if existing is not None:
                self._gray_hit_count += 1
                return existing
            self._gray[key] = value
            self._gray_build_count += 1
            self._note_array(gray)
            return value

    def analysis_gradient(self, frame: S13RenderFrame, width: int) -> np.ndarray:
        key = (frame.frame_id, int(width))
        with self._lock:
            cached = self._gradient.get(key)
            if cached is not None:
                self._gradient_hit_count += 1
                return cached
        gray, _scale = self.analysis_gray(frame, width)
        gradient = cv2.Sobel(gray, cv2.CV_32F, 1, 1, ksize=3)
        gradient.setflags(write=False)
        with self._lock:
            existing = self._gradient.get(key)
            if existing is not None:
                self._gradient_hit_count += 1
                return existing
            self._gradient[key] = gradient
            self._gradient_build_count += 1
            self._note_array(gradient)
            return gradient

    def prefetch_analysis(
        self,
        frames: Sequence[S13RenderFrame],
        width: int,
        *,
        workers: int = 2,
    ) -> tuple[tuple[np.ndarray, float], ...]:
        if workers not in (1, 2):
            raise ValueError("S1.3 FrameStore supports only 1 or 2 prefetch workers")
        self._prefetch_workers = workers
        if workers == 1:
            return tuple(self.analysis_gray(frame, width) for frame in frames)
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="s13-frame") as pool:
            futures = [pool.submit(self.analysis_gray, frame, width) for frame in frames]
            return tuple(future.result() for future in futures)

    def report(self) -> S13FrameStoreReport:
        with self._lock:
            return S13FrameStoreReport(
                raw_decode_count=self._raw_decode_count,
                raw_hit_count=self._raw_hit_count,
                gray_build_count=self._gray_build_count,
                gray_hit_count=self._gray_hit_count,
                gradient_build_count=self._gradient_build_count,
                gradient_hit_count=self._gradient_hit_count,
                eviction_count=self._eviction_count,
                resident_bytes=self._resident_bytes,
                peak_resident_bytes=self._peak_bytes,
                prefetch_workers=self._prefetch_workers,
                adopted_raw_count=self._adopted_raw_count,
                adopted_raw_bytes=self._adopted_raw_bytes,
                bulk_adopt_count=self._bulk_adopt_count,
                retained_raw_count_after_schedule=self._retained_raw_count_after_schedule,
                released_unselected_raw_count=self._released_unselected_raw_count,
                raw_decode_after_adopt_count=self._raw_decode_after_adopt_count,
                raw_bytes_after_schedule_retain=self._raw_bytes_after_schedule_retain,
            )

    def release_analysis_arrays(self) -> None:
        """Release temporary M0 analysis arrays while retaining raw RGB LRU entries."""

        with self._lock:
            self._resident_bytes -= sum(value[0].nbytes for value in self._gray.values())
            self._resident_bytes -= sum(value.nbytes for value in self._gradient.values())
            self._gray.clear()
            self._gradient.clear()


__all__ = ["S13FrameStore", "S13FrameStoreReport"]
