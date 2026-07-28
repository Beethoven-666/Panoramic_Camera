"""F0-F27 RapidOCR CUDA adapter equivalence against the retained OCR audit."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import cv2
import numpy as np

from panorama_demo.inspection_multiview import _read_rgbd, _undistortion_maps
from panorama_demo.inspection_ocr_identity import (
    OCRTextDetection,
    audit_waveshare_text,
)
from panorama_demo.rapidocr_onnx_adapter import (
    RapidOCRModels,
    RapidOCROnnxAdapter,
    RapidOCROnnxError,
    RapidOCRRuntime,
)
from panorama_demo.session import load_rgbd_session


FRAME_COUNT = 28
MINIMUM_POLYGON_IOU = 0.95
MAXIMUM_SCORE_DELTA = 0.02


def _polygon_iou(first: np.ndarray, second: np.ndarray) -> float:
    a = cv2.convexHull(np.asarray(first, dtype=np.float32))
    b = cv2.convexHull(np.asarray(second, dtype=np.float32))
    area_a = float(cv2.contourArea(a))
    area_b = float(cv2.contourArea(b))
    intersection, _ = cv2.intersectConvexConvex(a, b)
    union = area_a + area_b - float(intersection)
    return float(intersection / union) if union > 0.0 else 0.0


def _atomic_json(path: Path, value: dict[str, object]) -> None:
    pending = path.with_name(f".{path.name}.pending")
    pending.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(pending, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("session", type=Path)
    parser.add_argument("existing_audit", type=Path)
    parser.add_argument("det_model", type=Path)
    parser.add_argument("cls_model", type=Path)
    parser.add_argument("rec_model", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    started = time.perf_counter()
    output = arguments.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "rapidocr_adapter_f0_f27_report.json"
    if report_path.exists():
        raise RuntimeError("RapidOCR fixed validation report already exists")
    existing_path = arguments.existing_audit.expanduser().resolve()
    existing = json.loads(existing_path.read_text(encoding="utf-8"))
    if existing.get("schema") != "inspection-waveshare-ocr-rgbd-identity/v1":
        raise RuntimeError("Existing OCR audit schema is unsupported")
    expected_frames = list(existing["frame_audits"][:FRAME_COUNT])
    if len(expected_frames) != FRAME_COUNT:
        raise RuntimeError("Existing OCR audit lacks F0-F27 evidence")

    models = RapidOCRModels(
        detection=arguments.det_model,
        classification=arguments.cls_model,
        recognition=arguments.rec_model,
    )
    runtime = RapidOCRRuntime(
        provider="CUDAExecutionProvider",
        device_id=0,
        profile_directory=output,
        allow_shape_control_cpu=True,
    )
    adapter: RapidOCROnnxAdapter | None = None
    try:
        adapter = RapidOCROnnxAdapter(models, runtime)
        session = load_rgbd_session(arguments.session.expanduser().resolve())
        frames = sorted(
            session.frames, key=lambda item: int(item.frame_id)
        )[:FRAME_COUNT]
        maps = _undistortion_maps(session.calibration)
        frame_rows: list[dict[str, object]] = []
        all_equivalent = True
        for frame_index, (frame, expected_frame) in enumerate(
            zip(frames, expected_frames, strict=True)
        ):
            frame_id = int(frame.frame_id)
            if frame_id != int(expected_frame["frame_id"]):
                raise RuntimeError(
                    "Current F0-F27 frame order differs from existing audit"
                )
            image, _, _ = _read_rgbd(frame, session.calibration, maps)
            detections = adapter.predict(image)
            actual_targets: list[dict[str, object]] = []
            for detection in detections:
                identity = OCRTextDetection(
                    polygon_xy=detection.polygon_xy,
                    text=detection.text,
                    confidence=detection.score,
                )
                text_audit = audit_waveshare_text(identity)
                if text_audit["pass"]:
                    actual_targets.append(
                        {
                            "polygon_xy": detection.polygon_xy.tolist(),
                            "text": detection.text,
                            "score": float(detection.score),
                            "text_audit": text_audit,
                        }
                    )
            expected_targets = list(expected_frame["target_detections"])
            expected_count = len(expected_targets)
            actual_count = len(actual_targets)
            matches: list[dict[str, object]] = []
            frame_equivalent = expected_count == actual_count
            for expected_target, actual_target in zip(
                expected_targets, actual_targets
            ):
                expected_text = str(
                    expected_target["text_audit"]["normalized_text"]
                )
                actual_text = str(
                    actual_target["text_audit"]["normalized_text"]
                )
                score_delta = abs(
                    float(expected_target["text_audit"]["confidence"])
                    - float(actual_target["score"])
                )
                polygon_iou = _polygon_iou(
                    np.asarray(
                        expected_target["ocr_polygon_xy"],
                        dtype=np.float32,
                    ),
                    np.asarray(
                        actual_target["polygon_xy"], dtype=np.float32
                    ),
                )
                passed = bool(
                    expected_text == actual_text
                    and score_delta <= MAXIMUM_SCORE_DELTA
                    and polygon_iou >= MINIMUM_POLYGON_IOU
                )
                frame_equivalent &= passed
                matches.append(
                    {
                        "expected_text": expected_text,
                        "actual_text": actual_text,
                        "score_delta": score_delta,
                        "polygon_iou": polygon_iou,
                        "pass": passed,
                    }
                )
            all_equivalent &= frame_equivalent
            frame_rows.append(
                {
                    "frame_index": frame_index,
                    "frame_id": frame_id,
                    "expected_waveshare_detection_count": expected_count,
                    "actual_waveshare_detection_count": actual_count,
                    "matches": matches,
                    "equivalent": bool(frame_equivalent),
                }
            )
            print(
                f"F{frame_index} id={frame_id}: expected={expected_count} "
                f"actual={actual_count} equivalent={frame_equivalent}",
                flush=True,
            )
        runtime_audit = adapter.audit()
        provider_pass = bool(
            runtime_audit["execution_verified"]
            and runtime_audit["execution"]["pass"]
        )
        verdict = bool(all_equivalent and provider_pass)
        report: dict[str, object] = {
            "schema": "rapidocr-onnx-adapter-equivalence/v1",
            "verdict": "equivalent" if verdict else "not_equivalent",
            "frame_range": [0, FRAME_COUNT - 1],
            "frame_count": FRAME_COUNT,
            "existing_audit": str(existing_path),
            "existing_audit_uses_cpu_only_runtime": (
                "CUDAExecutionProvider"
                not in existing["model"]["available_execution_providers"]
            ),
            "waveshare_detection_equivalent": bool(all_equivalent),
            "provider_policy_pass": provider_pass,
            "failed_frame_indices": [
                int(row["frame_index"])
                for row in frame_rows
                if not row["equivalent"]
            ],
            "thresholds": {
                "minimum_polygon_iou": MINIMUM_POLYGON_IOU,
                "maximum_score_delta": MAXIMUM_SCORE_DELTA,
                "normalized_text_exact_match": True,
            },
            "runtime": runtime_audit,
            "frames": frame_rows,
            "formal_renderer_modified": False,
            "pyproject_modified": False,
            "isolated_environment_required": True,
            "elapsed_seconds": time.perf_counter() - started,
        }
    except RapidOCROnnxError as exc:
        report = {
            "schema": "rapidocr-onnx-adapter-equivalence/v1",
            "verdict": "provider_failed",
            "frame_range": [0, FRAME_COUNT - 1],
            "frame_count": FRAME_COUNT,
            "existing_audit": str(existing_path),
            "waveshare_detection_equivalent": False,
            "provider_policy_pass": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "runtime": adapter.audit() if adapter is not None else None,
            "formal_renderer_modified": False,
            "pyproject_modified": False,
            "isolated_environment_required": True,
            "elapsed_seconds": time.perf_counter() - started,
        }
    _atomic_json(report_path, report)
    print(report_path)
    print(
        json.dumps(
            {
                "verdict": report["verdict"],
                "waveshare_detection_equivalent": report[
                    "waveshare_detection_equivalent"
                ],
                "provider_policy_pass": report["provider_policy_pass"],
                "elapsed_seconds": report["elapsed_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
