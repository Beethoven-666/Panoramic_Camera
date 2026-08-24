"""Incremental, non-authoritative S013 V11 capture-time analysis and preview."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import queue
import threading
import time
from types import MappingProxyType
from typing import Callable, Mapping

import cv2
import numpy as np

from .capture_orbbec import (
    CaptureResult,
    LiveFramePacket,
    LiveSessionInfo,
    WrittenRGBDFrame,
)
from .video_s13_contract import (
    S13_VISUAL_CONTINUITY_ALGORITHM_ID,
    S13_VISUAL_CONTINUITY_IMPLEMENTATION_ID,
)
from .video_s13_online_p0 import (
    S13FrozenP0Authority,
    S13OnlineP0Engine,
    S13OnlineShadowSnapshot,
    calibration_from_live_document,
)


LIVE_PREVIEW_SCHEMA = "gemini305-s013-v11-live-preview/v1"
LIVE_PREVIEW_FAILURE_SCHEMA = "gemini305-s013-v11-live-preview-failure/v1"


@dataclass(frozen=True)
class S13LiveMotionEdge:
    left_frame_id: int
    right_frame_id: int
    dx_424_px: float
    reliable: bool
    accepted_monotonic_ns: int


@dataclass(frozen=True)
class CommittedFrameIdentity:
    frame_id: int
    color_path: Path
    aligned_depth_path: Path
    color_sha256: str
    aligned_depth_sha256: str
    timestamp_us: int
    frames_csv_row_index: int
    commit_monotonic_ns: int


@dataclass(frozen=True)
class S13V11LiveHandoff:
    algorithm_id: str
    implementation_id: str
    production_config_sha256: str
    session_root: Path
    committed_frames: tuple[CommittedFrameIdentity, ...]
    analysis_gray: Mapping[int, np.ndarray]
    motion_edges: tuple[S13LiveMotionEdge, ...]
    online_shadow: S13OnlineShadowSnapshot | None
    online_shadow_failure_reason: str | None
    frozen_p0_authority: S13FrozenP0Authority | None
    capture_started_monotonic_ns: int
    capture_stopped_monotonic_ns: int
    capture_metrics: Mapping[str, object]
    online_2d_metrics: Mapping[str, object]
    reuse_level: str = "validated_inputs_only"


@dataclass(frozen=True)
class S13LiveSnapshot:
    accepted_frames: int
    analysed_frames: int
    committed_frames: int
    preview_updates: int
    preview_skipped_due_to_load: int
    preview_failures: int
    first_preview_monotonic_ns: int | None
    preview_latency_p95_ms: float | None
    stable_motion_seconds: float
    cumulative_forward_424_px: float
    direction_consistency: float
    reliable_motion_fraction: float
    preview_queue_peak: int
    analysis_queue_peak: int
    stopped: bool


MotionEstimator = Callable[[np.ndarray, np.ndarray], tuple[float, bool]]


def _phase_motion(previous: np.ndarray, current: np.ndarray) -> tuple[float, bool]:
    shift, response = cv2.phaseCorrelate(
        previous.astype(np.float32, copy=False),
        current.astype(np.float32, copy=False),
    )
    dx = float(shift[0])
    return dx, bool(np.isfinite(dx) and abs(dx) <= previous.shape[1] * 0.5 and response >= 0.05)


class S13V11LiveObserver:
    """Bounded incremental state; never calls ORB, Open3D, TSDF, or a batch pipeline."""

    def __init__(
        self,
        *,
        production_config_sha256: str,
        analysis_width_px: int = 424,
        stable_motion_seconds: float = 0.8,
        minimum_forward_px: float = 32.0,
        minimum_direction_consistency: float = 0.85,
        minimum_reliable_fraction: float = 0.75,
        minimum_source_candidates: int = 5,
        preview_hz: float = 4.0,
        preview_output: Path | None = None,
        motion_estimator: MotionEstimator = _phase_motion,
    ) -> None:
        if analysis_width_px != 424:
            raise ValueError("S013 V11 live analysis width is frozen at 424 pixels")
        self.production_config_sha256 = production_config_sha256
        self.analysis_width_px = analysis_width_px
        self.required_stable_seconds = stable_motion_seconds
        self.minimum_forward_px = minimum_forward_px
        self.minimum_direction_consistency = minimum_direction_consistency
        self.minimum_reliable_fraction = minimum_reliable_fraction
        self.minimum_source_candidates = minimum_source_candidates
        self.preview_interval_ns = int(1_000_000_000 / preview_hz)
        self.preview_output = None if preview_output is None else preview_output.expanduser().resolve()
        self.motion_estimator = motion_estimator
        self._analysis_queue: queue.Queue[LiveFramePacket | None] = queue.Queue(maxsize=64)
        self._preview_queue: queue.Queue[tuple[int, int, np.ndarray] | None] = queue.Queue(maxsize=1)
        self._lock = threading.Lock()
        self._committed_condition = threading.Condition(self._lock)
        self._session: LiveSessionInfo | None = None
        self._capture_result: CaptureResult | None = None
        self._accepting = True
        self._stopped = False
        self._accepted = self._analysed = 0
        self._analysis_queue_peak = self._preview_queue_peak = 0
        self._preview_updates = self._preview_skipped = self._preview_failures = 0
        self._preview_generation = 0
        self._preview_disabled = False
        self._first_preview_ns: int | None = None
        self._preview_latencies_ms: list[float] = []
        self._last_preview_request_ns = 0
        self._committed: list[CommittedFrameIdentity] = []
        self._shadow_engine: S13OnlineP0Engine | None = None
        self._shadow_snapshot: S13OnlineShadowSnapshot | None = None
        self._shadow_failure_reason: str | None = None
        self._shadow_next_index = 0
        self._shadow_stop = False
        self._shadow_post_stop_started_ns: int | None = None
        self._shadow_catchup_seconds = 0.0
        self._analysis_gray: dict[int, np.ndarray] = {}
        self._motion_edges: list[S13LiveMotionEdge] = []
        self._stable_edges: list[S13LiveMotionEdge] = []
        self._stable_segment_started_ns: int | None = None
        self._last_motion_ns: int | None = None
        self._latest_frame_id: int | None = None
        self._latest_panorama_preview: np.ndarray | None = None
        self._analysis_thread = threading.Thread(
            target=self._analysis_loop, name="s013-live-analysis", daemon=False
        )
        self._preview_thread = threading.Thread(
            target=self._preview_loop, name="s013-live-preview", daemon=False
        )
        self._shadow_thread = threading.Thread(
            target=self._shadow_loop, name="s013-live-committed-shadow", daemon=True
        )
        self._analysis_thread.start()
        self._preview_thread.start()
        self._shadow_thread.start()

    def on_session_ready(self, session: LiveSessionInfo) -> None:
        shadow: S13OnlineP0Engine | None = None
        failure: str | None = None
        try:
            shadow = S13OnlineP0Engine(
                session_root=session.root,
                calibration=calibration_from_live_document(session.calibration),
                analysis_width_px=self.analysis_width_px,
            )
        except Exception as exc:
            failure = f"{type(exc).__name__}: {exc}"
        with self._lock:
            self._session = session
            self._shadow_engine = shadow
            self._shadow_failure_reason = failure
            self._committed_condition.notify_all()
        if self.preview_output is not None:
            self.preview_output.mkdir(parents=True, exist_ok=True)

    def _preview_root(self, session: LiveSessionInfo) -> Path:
        return session.root if self.preview_output is None else self.preview_output

    def on_frame_accepted(self, packet: LiveFramePacket) -> None:
        with self._lock:
            if not self._accepting:
                return
            self._accepted += 1
        try:
            self._analysis_queue.put_nowait(packet)
            with self._lock:
                self._analysis_queue_peak = max(
                    self._analysis_queue_peak, self._analysis_queue.qsize()
                )
        except queue.Full:
            with self._lock:
                self._preview_skipped += 1

    def on_frame_committed(self, frame: WrittenRGBDFrame) -> None:
        identity = CommittedFrameIdentity(
            frame_id=frame.frame_id,
            color_path=frame.color_path,
            aligned_depth_path=frame.aligned_depth_path,
            color_sha256=frame.color_sha256,
            aligned_depth_sha256=frame.aligned_depth_sha256,
            timestamp_us=frame.timestamp_us,
            frames_csv_row_index=frame.frames_csv_row_index,
            commit_monotonic_ns=frame.commit_monotonic_ns,
        )
        with self._lock:
            self._committed.append(identity)
            self._committed_condition.notify()

    def on_capture_stopping(self) -> None:
        stop_shadow = False
        with self._lock:
            if self._stopped:
                return
            self._accepting = False
            self._stopped = True
            self._shadow_post_stop_started_ns = time.monotonic_ns()
            if self._session is None:
                self._shadow_stop = True
                stop_shadow = True
                self._committed_condition.notify_all()
        while True:
            try:
                self._analysis_queue.get_nowait()
                self._analysis_queue.task_done()
            except queue.Empty:
                break
        while True:
            try:
                self._preview_queue.get_nowait()
                self._preview_queue.task_done()
            except queue.Empty:
                break
        self._analysis_queue.put(None)
        self._preview_queue.put(None)
        self._analysis_thread.join()
        self._preview_thread.join()
        if stop_shadow:
            self._shadow_thread.join()
        self._write_stopped_state()

    def on_capture_closed(self, result: CaptureResult) -> None:
        self.on_capture_stopping()
        with self._lock:
            self._capture_result = result
            self._shadow_stop = True
            self._committed_condition.notify_all()
        self._shadow_thread.join()
        if self._shadow_post_stop_started_ns is not None:
            self._shadow_catchup_seconds = (
                time.monotonic_ns() - self._shadow_post_stop_started_ns
            ) / 1_000_000_000.0

    def _shadow_loop(self) -> None:
        while True:
            with self._committed_condition:
                self._committed_condition.wait_for(
                    lambda: (
                        self._shadow_stop
                        or self._shadow_failure_reason is not None
                        or (
                            self._shadow_engine is not None
                            and self._shadow_next_index < len(self._committed)
                        )
                    )
                )
                if self._shadow_failure_reason is not None:
                    if self._shadow_stop:
                        return
                    self._committed_condition.wait()
                    continue
                if self._shadow_next_index >= len(self._committed):
                    if self._shadow_stop:
                        return
                    continue
                identity = self._committed[self._shadow_next_index]
                engine = self._shadow_engine
                force_layout = bool(
                    self._shadow_stop
                    and self._shadow_next_index == len(self._committed) - 1
                )
            assert engine is not None
            try:
                shadow_snapshot = engine.append_committed(
                    identity, force_layout=force_layout
                )
            except Exception as exc:
                with self._committed_condition:
                    self._shadow_failure_reason = f"{type(exc).__name__}: {exc}"
                    self._committed_condition.notify_all()
            else:
                with self._committed_condition:
                    self._shadow_snapshot = shadow_snapshot
                    self._shadow_next_index += 1
                    self._committed_condition.notify_all()

    def _analysis_image(self, color: np.ndarray) -> np.ndarray:
        height = max(1, round(color.shape[0] * self.analysis_width_px / color.shape[1]))
        resized = cv2.resize(color, (self.analysis_width_px, height), interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    def _gate(self, packet: LiveFramePacket | SimplePacket) -> tuple[bool, float, float, float, float]:
        reliable = [
            edge for edge in self._stable_edges
            if edge.reliable and abs(edge.dx_424_px) > 0.25
        ]
        all_edges = self._stable_edges
        if not reliable:
            return False, 0.0, 0.0, 0.0, 0.0
        direction = 1.0 if float(np.median([edge.dx_424_px for edge in reliable])) >= 0 else -1.0
        consistent = [edge for edge in reliable if direction * edge.dx_424_px > 0.0]
        consistency = len(consistent) / len(reliable)
        reliable_fraction = len(reliable) / max(1, len(all_edges))
        forward = float(sum(direction * edge.dx_424_px for edge in consistent))
        stable_seconds = (
            0.0
            if self._stable_segment_started_ns is None
            else (
                packet.accepted_monotonic_ns - self._stable_segment_started_ns
            ) / 1_000_000_000.0
        )
        passed = (
            stable_seconds >= self.required_stable_seconds
            and forward >= self.minimum_forward_px
            and consistency >= self.minimum_direction_consistency
            and reliable_fraction >= self.minimum_reliable_fraction
            and self._analysed >= self.minimum_source_candidates
            and packet.writer_queue_fraction <= 0.25
            and packet.writer_queue_drops == 0
        )
        return passed, stable_seconds, forward, consistency, reliable_fraction

    def _request_preview(self, packet: LiveFramePacket) -> None:
        with self._lock:
            if self._preview_disabled or self._stopped:
                return
            shadow = self._shadow_snapshot
        if shadow is None or shadow.current_p0 is None:
            return
        now = time.monotonic_ns()
        if self._last_preview_request_ns and now - self._last_preview_request_ns < self.preview_interval_ns:
            return
        self._last_preview_request_ns = now
        current = np.asarray(shadow.current_p0.image)
        scale = min(1.0, 1600.0 / max(1, current.shape[1]))
        preview = (
            current.copy()
            if scale == 1.0
            else cv2.resize(
                current,
                (1600, max(1, int(round(current.shape[0] * scale)))),
                interpolation=cv2.INTER_AREA,
            )
        )
        preview = np.ascontiguousarray(preview)
        preview.setflags(write=False)
        with self._lock:
            self._latest_panorama_preview = preview
        item = (packet.frame_id, packet.accepted_monotonic_ns, preview)
        try:
            self._preview_queue.put_nowait(item)
        except queue.Full:
            try:
                self._preview_queue.get_nowait()
                self._preview_queue.task_done()
            except queue.Empty:
                pass
            self._preview_queue.put_nowait(item)
            with self._lock:
                self._preview_skipped += 1
        with self._lock:
            self._preview_queue_peak = max(self._preview_queue_peak, self._preview_queue.qsize())

    def capture_preview_image(self) -> np.ndarray | None:
        """Return the latest immutable panorama for the shared capture window."""

        with self._lock:
            return self._latest_panorama_preview

    def _analysis_loop(self) -> None:
        previous: tuple[int, int, np.ndarray] | None = None
        while True:
            packet = self._analysis_queue.get()
            try:
                if packet is None:
                    return
                gray = self._analysis_image(packet.color_bgr)
                with self._lock:
                    self._analysis_gray[packet.frame_id] = gray
                    self._analysed += 1
                    self._latest_frame_id = packet.frame_id
                if previous is not None:
                    dx, reliable = self.motion_estimator(previous[2], gray)
                    edge = S13LiveMotionEdge(
                        left_frame_id=previous[0],
                        right_frame_id=packet.frame_id,
                        dx_424_px=float(dx),
                        reliable=bool(reliable),
                        accepted_monotonic_ns=packet.accepted_monotonic_ns,
                    )
                    with self._lock:
                        self._motion_edges.append(edge)
                        previous_direction = (
                            0.0
                            if not self._stable_edges
                            else float(np.sign(self._stable_edges[-1].dx_424_px))
                        )
                        direction = float(np.sign(edge.dx_424_px))
                        gap_ns = (
                            0
                            if self._last_motion_ns is None
                            else edge.accepted_monotonic_ns - self._last_motion_ns
                        )
                        gap_reset = self._last_motion_ns is not None and gap_ns > 400_000_000
                        reset = (
                            not edge.reliable
                            or abs(edge.dx_424_px) <= 0.25
                            or gap_reset
                            or (previous_direction != 0.0 and direction != previous_direction)
                        )
                        if reset:
                            self._stable_edges.clear()
                            self._stable_segment_started_ns = None
                        if edge.reliable and abs(edge.dx_424_px) > 0.25 and not gap_reset:
                            if not self._stable_edges:
                                self._stable_segment_started_ns = previous[1]
                            self._stable_edges.append(edge)
                        self._last_motion_ns = edge.accepted_monotonic_ns
                        passed, *_ = self._gate(packet)
                    if passed:
                        self._request_preview(packet)
                previous = (packet.frame_id, packet.accepted_monotonic_ns, gray)
            except Exception:
                with self._lock:
                    self._preview_failures += 1
            finally:
                self._analysis_queue.task_done()

    def _preview_loop(self) -> None:
        while True:
            item = self._preview_queue.get()
            try:
                if item is None:
                    return
                frame_id, accepted_ns, preview = item
                with self._lock:
                    session = self._session
                    stopped = self._stopped
                    disabled = self._preview_disabled
                    shadow = self._shadow_snapshot
                if session is None or stopped or disabled:
                    continue
                ok, encoded = cv2.imencode(".jpg", preview, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if not ok:
                    raise OSError("Could not encode S013 live preview")
                preview_root = self._preview_root(session)
                pending = preview_root / ".live_preview.pending.jpg"
                pending.write_bytes(encoded.tobytes())
                with self._lock:
                    if self._stopped:
                        pending.unlink(missing_ok=True)
                        continue
                os.replace(pending, preview_root / "live_preview.jpg")
                published_ns = time.monotonic_ns()
                metadata = {
                    "schema": LIVE_PREVIEW_SCHEMA,
                    "authority": "non_authoritative_live_preview",
                    "algorithm_id": S13_VISUAL_CONTINUITY_ALGORITHM_ID,
                    "preview_generation": self._preview_generation + 1,
                    "latest_frame_id": frame_id,
                    "stable_source_count": (
                        0 if shadow is None else shadow.frontiers.sealed_source_count
                    ),
                    "mutable_source_count": (
                        0 if shadow is None else (
                            shadow.frontiers.selected_source_count
                            - shadow.frontiers.sealed_source_count
                        )
                    ),
                    "current_p0_width": 0 if preview is None else int(preview.shape[1]),
                    "stage_visualization": "s013_online_p0_current/v1",
                    "capture_active": True,
                    "published_monotonic_ns": published_ns,
                }
                metadata_pending = preview_root / ".live_preview.pending.json"
                metadata_pending.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
                os.replace(metadata_pending, preview_root / "live_preview_state.json")
                with self._lock:
                    self._preview_updates += 1
                    self._preview_generation += 1
                    if self._first_preview_ns is None:
                        self._first_preview_ns = published_ns
                    self._preview_latencies_ms.append((published_ns - accepted_ns) / 1_000_000.0)
            except Exception as exc:
                self._disable_preview(exc)
            finally:
                self._preview_queue.task_done()

    def _disable_preview(self, exc: BaseException) -> None:
        with self._lock:
            if self._preview_disabled:
                return
            self._preview_disabled = True
            self._preview_failures += 1
            session = self._session
        if session is None:
            return
        payload = {
            "schema": LIVE_PREVIEW_FAILURE_SCHEMA,
            "authority": "non_authoritative_live_preview",
            "algorithm_id": S13_VISUAL_CONTINUITY_ALGORITHM_ID,
            "error_type": type(exc).__name__,
            "message": str(exc),
            "capture_continues": True,
        }
        preview_root = self._preview_root(session)
        pending = preview_root / ".live_preview_failure.pending.json"
        pending.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(pending, preview_root / "live_preview_failure.json")

    def _write_stopped_state(self) -> None:
        with self._lock:
            session = self._session
            if session is None:
                return
            payload = {
                "schema": LIVE_PREVIEW_SCHEMA,
                "authority": "non_authoritative_live_preview",
                "algorithm_id": S13_VISUAL_CONTINUITY_ALGORITHM_ID,
                "preview_generation": self._preview_generation,
                "latest_frame_id": self._latest_frame_id,
                "stable_source_count": (
                    0
                    if self._shadow_snapshot is None
                    else self._shadow_snapshot.frontiers.sealed_source_count
                ),
                "mutable_source_count": 0,
                "stage_visualization": "s013_online_p0_current/v1",
                "capture_active": False,
            }
        preview_root = self._preview_root(session)
        pending = preview_root / ".live_preview_state.pending.json"
        pending.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(pending, preview_root / "live_preview_state.json")

    def snapshot(self) -> S13LiveSnapshot:
        with self._lock:
            if self._motion_edges:
                packet_ns = self._motion_edges[-1].accepted_monotonic_ns
                dummy = SimplePacket(packet_ns)
                _, stable, forward, consistency, reliable = self._gate(dummy)
            else:
                stable = forward = consistency = reliable = 0.0
            p95 = (
                None
                if not self._preview_latencies_ms
                else float(np.percentile(self._preview_latencies_ms, 95))
            )
            return S13LiveSnapshot(
                accepted_frames=self._accepted,
                analysed_frames=self._analysed,
                committed_frames=len(self._committed),
                preview_updates=self._preview_updates,
                preview_skipped_due_to_load=self._preview_skipped,
                preview_failures=self._preview_failures,
                first_preview_monotonic_ns=self._first_preview_ns,
                preview_latency_p95_ms=p95,
                stable_motion_seconds=stable,
                cumulative_forward_424_px=forward,
                direction_consistency=consistency,
                reliable_motion_fraction=reliable,
                preview_queue_peak=self._preview_queue_peak,
                analysis_queue_peak=self._analysis_queue_peak,
                stopped=self._stopped,
            )

    def freeze_handoff(self) -> S13V11LiveHandoff:
        snapshot = self.snapshot()
        with self._lock:
            if self._session is None or self._capture_result is None or not self._stopped:
                raise RuntimeError("S013 live handoff requires a closed capture")
            committed = tuple(self._committed)
            if len(committed) != self._capture_result.written_frames:
                raise RuntimeError("S013 live committed ledger does not cover every written frame")
            if [item.frame_id for item in committed] != sorted(item.frame_id for item in committed):
                raise RuntimeError("S013 live committed ledger is not chronological")
            capture_seconds = (
                self._capture_result.capture_stopped_monotonic_ns
                - self._capture_result.capture_started_monotonic_ns
            ) / 1_000_000_000.0
            frozen: S13FrozenP0Authority | None = None
            freeze_failure = self._shadow_failure_reason
            if (
                freeze_failure is None
                and self._shadow_engine is not None
                and self._capture_result.queue_drops == 0
                and self._capture_result.write_errors == 0
            ):
                try:
                    frozen = self._shadow_engine.freeze(
                        committed_frames=committed,
                        production_config_sha256=self.production_config_sha256,
                    )
                    self._shadow_snapshot = self._shadow_engine.snapshot()
                except Exception as exc:
                    freeze_failure = f"{type(exc).__name__}: {exc}"
            reuse_level = (
                "frozen_p0_authority_v1"
                if frozen is not None else "validated_inputs_only"
            )
            return S13V11LiveHandoff(
                algorithm_id=S13_VISUAL_CONTINUITY_ALGORITHM_ID,
                implementation_id=S13_VISUAL_CONTINUITY_IMPLEMENTATION_ID,
                production_config_sha256=self.production_config_sha256,
                session_root=self._session.root,
                committed_frames=committed,
                analysis_gray=MappingProxyType(dict(self._analysis_gray)),
                motion_edges=tuple(self._motion_edges),
                online_shadow=self._shadow_snapshot,
                online_shadow_failure_reason=freeze_failure,
                frozen_p0_authority=frozen,
                capture_started_monotonic_ns=self._capture_result.capture_started_monotonic_ns,
                capture_stopped_monotonic_ns=self._capture_result.capture_stopped_monotonic_ns,
                capture_metrics=MappingProxyType({
                    "physical_seconds": capture_seconds,
                    "received_frames": self._capture_result.received_frames,
                    "written_frames": self._capture_result.written_frames,
                    "max_queue_depth": self._capture_result.max_queue_depth,
                    "queue_drops": self._capture_result.queue_drops,
                    "write_errors": self._capture_result.write_errors,
                }),
                online_2d_metrics=MappingProxyType({
                    "first_preview_seconds": (
                        None
                        if snapshot.first_preview_monotonic_ns is None
                        else (
                            snapshot.first_preview_monotonic_ns
                            - self._capture_result.capture_started_monotonic_ns
                        ) / 1_000_000_000.0
                    ),
                    "preview_latency_p95_ms": snapshot.preview_latency_p95_ms,
                    "preview_update_count": snapshot.preview_updates,
                    "preview_skipped_due_to_load": snapshot.preview_skipped_due_to_load,
                    "preview_failures": snapshot.preview_failures,
                    "pair_evidence_reused_count": 0,
                    "reuse_level": reuse_level,
                    "shadow_processed_committed_frames": self._shadow_next_index,
                    "shadow_failure_reason": self._shadow_failure_reason,
                    "shadow_final_selected_source_count": (
                        0
                        if self._shadow_snapshot is None
                        else self._shadow_snapshot.frontiers.selected_source_count
                    ),
                    "p0_current_width": (
                        0
                        if self._shadow_snapshot is None
                        or self._shadow_snapshot.current_p0 is None
                        else int(self._shadow_snapshot.current_p0.image.shape[1])
                    ),
                    "p0_sealed_source_count": (
                        0
                        if self._shadow_snapshot is None
                        else self._shadow_snapshot.frontiers.sealed_source_count
                    ),
                    "p0_mutable_tail_source_count": (
                        0
                        if self._shadow_snapshot is None
                        else (
                            self._shadow_snapshot.frontiers.selected_source_count
                            - self._shadow_snapshot.frontiers.sealed_source_count
                        )
                    ),
                    "p0_prefix_reused_fraction": (
                        0.0 if frozen is None else frozen.p0_prefix_reused_fraction
                    ),
                    "tail_finalize_source_count": (
                        0 if frozen is None else frozen.finalized_tail_source_count
                    ),
                    "final_metadata_closure_seconds": (
                        None if frozen is None else frozen.final_metadata_closure_seconds
                    ),
                    "tail_finalize_seconds": (
                        None if frozen is None else frozen.tail_finalize_seconds
                    ),
                    "rollback_checkpoint_used": (
                        False if frozen is None else frozen.rollback_checkpoint_used
                    ),
                    "rollback_reasons": (
                        [] if frozen is None else list(frozen.rollback_reasons)
                    ),
                    "full_m0_m3_recomputed": frozen is None,
                    "committed_catchup_seconds": self._shadow_catchup_seconds,
                }),
                reuse_level=reuse_level,
            )


@dataclass(frozen=True)
class SimplePacket:
    accepted_monotonic_ns: int
    writer_queue_fraction: float = 0.0
    writer_queue_drops: int = 0


__all__ = [
    "CommittedFrameIdentity",
    "S13LiveMotionEdge",
    "S13LiveSnapshot",
    "S13V11LiveHandoff",
    "S13V11LiveObserver",
]
