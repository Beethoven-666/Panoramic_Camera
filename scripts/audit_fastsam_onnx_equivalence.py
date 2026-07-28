from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import time

import cv2
import numpy as np

from panorama_demo.fastsam_onnx import FastSAMOnnxRunner, summarize_onnxruntime_profile


FIXED_SAMPLE_FRAME_IDS = (0, 20, 36, 57, 79, 101, 133, 157)
EQUIVALENCE_THRESHOLDS = {
    "minimum_count_agreement": 0.95,
    "minimum_matched_reference_ratio": 0.90,
    "minimum_median_bbox_iou": 0.90,
    "minimum_median_mask_iou": 0.85,
    "minimum_p10_mask_iou": 0.70,
}


def _read_polygons(path: Path, width: int, height: int) -> list[np.ndarray]:
    polygons: list[np.ndarray] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        values = np.fromstring(line, sep=" ", dtype=np.float32)
        if values.size < 7 or (values.size - 1) % 2:
            continue
        polygon = values[1:].reshape(-1, 2)
        polygon[:, 0] *= width
        polygon[:, 1] *= height
        polygons.append(polygon)
    return polygons


def _polygon_mask(polygon: np.ndarray, width: int, height: int) -> np.ndarray:
    mask = np.zeros((height, width), np.uint8)
    cv2.fillPoly(mask, [np.rint(polygon).astype(np.int32)], 1)
    return mask.astype(bool)


def _bbox(polygon: np.ndarray) -> np.ndarray:
    return np.asarray(
        [polygon[:, 0].min(), polygon[:, 1].min(), polygon[:, 0].max(), polygon[:, 1].max()],
        dtype=np.float32,
    )


def _bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return float(intersection / union) if union else 0.0


def _match(
    generated: list,
    reference_polygons: list[np.ndarray],
    width: int,
    height: int,
) -> dict:
    reference_boxes = [_bbox(polygon) for polygon in reference_polygons]
    generated_boxes = [np.asarray(proposal.bbox_xyxy, np.float32) for proposal in generated]
    candidates: list[tuple[float, int, int]] = []
    for generated_index, generated_box in enumerate(generated_boxes):
        for reference_index, reference_box in enumerate(reference_boxes):
            iou = _bbox_iou(generated_box, reference_box)
            if iou >= 0.25:
                candidates.append((iou, generated_index, reference_index))
    candidates.sort(reverse=True)
    used_generated: set[int] = set()
    used_reference: set[int] = set()
    matches: list[dict] = []
    for box_iou, generated_index, reference_index in candidates:
        if generated_index in used_generated or reference_index in used_reference:
            continue
        used_generated.add(generated_index)
        used_reference.add(reference_index)
        reference_mask = _polygon_mask(reference_polygons[reference_index], width, height)
        generated_mask = generated[generated_index].mask
        intersection = int(np.count_nonzero(reference_mask & generated_mask))
        union = int(np.count_nonzero(reference_mask | generated_mask))
        matches.append(
            {
                "bbox_iou": float(box_iou),
                "mask_iou": float(intersection / union) if union else 0.0,
            }
        )
    return {
        "generated_count": len(generated),
        "reference_count": len(reference_polygons),
        "matched_count": len(matches),
        "bbox_ious": [match["bbox_iou"] for match in matches],
        "mask_ious": [match["mask_iou"] for match in matches],
    }


def _aggregate(frames: list[dict]) -> dict:
    bbox = np.asarray([value for frame in frames for value in frame["bbox_ious"]], np.float64)
    mask = np.asarray([value for frame in frames for value in frame["mask_ious"]], np.float64)
    generated = sum(frame["generated_count"] for frame in frames)
    reference = sum(frame["reference_count"] for frame in frames)
    matched = sum(frame["matched_count"] for frame in frames)
    count_agreement = min(generated, reference) / max(generated, reference) if generated or reference else 1.0
    matched_reference_ratio = matched / reference if reference else 1.0
    result = {
        "frame_count": len(frames),
        "generated_count": generated,
        "reference_count": reference,
        "count_agreement": count_agreement,
        "matched_reference_ratio": matched_reference_ratio,
        "best_bbox_iou": float(np.max(bbox)) if bbox.size else 0.0,
        "mean_bbox_iou": float(np.mean(bbox)) if bbox.size else 0.0,
        "median_bbox_iou": float(np.median(bbox)) if bbox.size else 0.0,
        "p10_bbox_iou": float(np.percentile(bbox, 10)) if bbox.size else 0.0,
        "best_mask_iou": float(np.max(mask)) if mask.size else 0.0,
        "mean_mask_iou": float(np.mean(mask)) if mask.size else 0.0,
        "median_mask_iou": float(np.median(mask)) if mask.size else 0.0,
        "p10_mask_iou": float(np.percentile(mask, 10)) if mask.size else 0.0,
    }
    result["equivalent"] = bool(
        result["count_agreement"] >= EQUIVALENCE_THRESHOLDS["minimum_count_agreement"]
        and result["matched_reference_ratio"]
        >= EQUIVALENCE_THRESHOLDS["minimum_matched_reference_ratio"]
        and result["median_bbox_iou"] >= EQUIVALENCE_THRESHOLDS["minimum_median_bbox_iou"]
        and result["median_mask_iou"] >= EQUIVALENCE_THRESHOLDS["minimum_median_mask_iou"]
        and result["p10_mask_iou"] >= EQUIVALENCE_THRESHOLDS["minimum_p10_mask_iou"]
    )
    return result


def _failure_reasons(summary: dict) -> list[str]:
    reasons: list[str] = []
    for metric, threshold_name in (
        ("count_agreement", "minimum_count_agreement"),
        ("matched_reference_ratio", "minimum_matched_reference_ratio"),
        ("median_bbox_iou", "minimum_median_bbox_iou"),
        ("median_mask_iou", "minimum_median_mask_iou"),
        ("p10_mask_iou", "minimum_p10_mask_iou"),
    ):
        if summary[metric] < EQUIVALENCE_THRESHOLDS[threshold_name]:
            reasons.append(
                f"{metric}={summary[metric]:.9f} below "
                f"{threshold_name}={EQUIVALENCE_THRESHOLDS[threshold_name]:.9f}"
            )
    return reasons


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("session_dir", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("reference_labels", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--sample-only", action="store_true")
    parser.add_argument("--frame-id", type=int, action="append")
    parser.add_argument("--allow-cpu-diagnostic-fallback", action="store_true")
    args = parser.parse_args()
    image_dir = args.session_dir / "color"
    label_paths = sorted(args.reference_labels.glob("*.txt"))
    if not label_paths:
        raise SystemExit("no reference labels found")
    if args.frame_id:
        frame_ids = list(dict.fromkeys(args.frame_id))
    elif args.sample_only:
        frame_ids = list(FIXED_SAMPLE_FRAME_IDS)
    else:
        frame_ids = [int(path.stem) for path in label_paths]
    args.output.mkdir(parents=True, exist_ok=True)
    runner = FastSAMOnnxRunner(
        args.model,
        allow_cpu_diagnostic_fallback=args.allow_cpu_diagnostic_fallback,
        enable_profiling=True,
    )
    frames: list[dict] = []
    started = time.perf_counter()
    for frame_id in frame_ids:
        image_path = next(iter(sorted(image_dir.glob(f"{frame_id:08d}.*"))), None)
        label_path = args.reference_labels / f"{frame_id:08d}.txt"
        if image_path is None or not label_path.is_file():
            raise SystemExit(f"missing image or label for frame {frame_id}")
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise SystemExit(f"failed to read {image_path}")
        generated = runner.predict(image)
        reference = _read_polygons(label_path, image.shape[1], image.shape[0])
        result = _match(generated, reference, image.shape[1], image.shape[0])
        result["frame_id"] = frame_id
        frame_summary = _aggregate([result])
        result["proposal_count_delta"] = result["generated_count"] - result["reference_count"]
        result["iou_summary"] = {
            key: frame_summary[key]
            for key in (
                "count_agreement",
                "matched_reference_ratio",
                "best_bbox_iou",
                "mean_bbox_iou",
                "median_bbox_iou",
                "p10_bbox_iou",
                "best_mask_iou",
                "mean_mask_iou",
                "median_mask_iou",
                "p10_mask_iou",
            )
        }
        result["equivalent"] = frame_summary["equivalent"]
        result["failure_reasons"] = _failure_reasons(frame_summary)
        frames.append(result)
        print(
            f"{frame_id:08d}: generated={result['generated_count']} "
            f"reference={result['reference_count']} matched={result['matched_count']}",
            flush=True,
        )
    profile_source = runner.end_profiling()
    profile = summarize_onnxruntime_profile(profile_source) if profile_source else {}
    if profile_source and profile_source.is_file():
        shutil.copy2(profile_source, args.output / "onnxruntime_profile.json")
    aggregate = _aggregate(frames)
    cuda = profile.get("providers", {}).get("CUDAExecutionProvider", {})
    conv_providers = profile.get("operator_providers", {}).get("Conv", [])
    gpu_execution_verified = bool(cuda.get("node_events", 0) and "CUDAExecutionProvider" in conv_providers)
    verdict = bool(
        aggregate["equivalent"]
        and runner.execution_mode == "cuda_priority"
        and gpu_execution_verified
    )
    report = {
        "schema": "gemini305-fastsam-onnx-equivalence-audit/v1",
        "verdict": "equivalent" if verdict else "not_equivalent",
        "fixed_parameters": {
            "image_size": 1024,
            "retina_masks": True,
            "confidence_threshold": 0.25,
            "nms_iou_threshold": 0.9,
            "max_detections": 300,
            "sample_frame_ids": list(FIXED_SAMPLE_FRAME_IDS),
        },
        "equivalence_thresholds": EQUIVALENCE_THRESHOLDS,
        "model_path": str(args.model.resolve()),
        "model_copied": False,
        "forbidden_framework_imports": False,
        "execution": {
            "mode": runner.execution_mode,
            "active_providers": list(runner.active_providers),
            "diagnostic_cpu_fallback_used": runner.diagnostic_cpu_fallback_used,
            "gpu_execution_verified": gpu_execution_verified,
            "elapsed_seconds": time.perf_counter() - started,
            "profile": profile,
            "cpu_node_reason": (
                "ONNX Runtime assigns unsupported shape/control nodes to CPU; convolution must remain on CUDA."
            ),
        },
        "aggregate": aggregate,
        "failed_frame_ids": [frame["frame_id"] for frame in frames if not frame["equivalent"]],
        "count_mismatch_frame_ids": [
            frame["frame_id"] for frame in frames if frame["proposal_count_delta"] != 0
        ],
        "unmatched_reference_frame_ids": [
            frame["frame_id"]
            for frame in frames
            if frame["matched_count"] != frame["reference_count"]
        ],
        "frames": frames,
    }
    (args.output / "equivalence_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps({"verdict": report["verdict"], **aggregate}, indent=2))
    return 0 if verdict else 2


if __name__ == "__main__":
    raise SystemExit(main())
