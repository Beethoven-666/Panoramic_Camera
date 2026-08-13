from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np


BRANCHES = ("fast_direct", "fast_ignore_pose", "slow_direct", "slow_ignore_pose")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _ssim(left: np.ndarray, right: np.ndarray, mask: np.ndarray) -> float:
    left_f = left.astype(np.float64)
    right_f = right.astype(np.float64)
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    scores: list[float] = []
    for channel in range(3):
        x = left_f[:, :, channel]
        y = right_f[:, :, channel]
        mu_x = cv2.GaussianBlur(x, (11, 11), 1.5)
        mu_y = cv2.GaussianBlur(y, (11, 11), 1.5)
        sigma_x = cv2.GaussianBlur(x * x, (11, 11), 1.5) - mu_x * mu_x
        sigma_y = cv2.GaussianBlur(y * y, (11, 11), 1.5) - mu_y * mu_y
        sigma_xy = cv2.GaussianBlur(x * y, (11, 11), 1.5) - mu_x * mu_y
        numerator = (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)
        denominator = (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
        scores.append(float(np.mean((numerator / denominator)[mask])))
    return float(np.mean(scores))


def _label(image: np.ndarray, text: str) -> np.ndarray:
    bar_height = 46
    labeled = np.zeros((image.shape[0] + bar_height, image.shape[1], 3), np.uint8)
    labeled[bar_height:] = image
    cv2.putText(
        labeled,
        text,
        (14, 31),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.78,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return labeled


def _write_comparison(
    output: Path,
    branch: str,
    p1: np.ndarray,
    p3: np.ndarray,
    common_valid: np.ndarray,
) -> None:
    maximum_difference = np.max(cv2.absdiff(p1, p3), axis=2)
    heatmap = cv2.applyColorMap(maximum_difference, cv2.COLORMAP_TURBO)
    heatmap[~common_valid] = 0
    comparison = np.vstack(
        [
            _label(p1, f"{branch}: P1 vertical owner-only"),
            _label(p3, f"{branch}: final P3 visual panorama"),
            _label(heatmap, "absolute RGB difference (TURBO; black = outside common valid mask)"),
        ]
    )
    if not cv2.imwrite(str(output), comparison, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
        raise RuntimeError(f"failed to write {output}")


def _compare_branch(run_root: Path, output_root: Path, branch: str) -> dict:
    latest = _read_json(run_root / "branches" / branch / "latest_run.json")
    generation = Path(latest["generation"])
    p1_path = Path(latest["m4"]["panorama"])
    p3_path = Path(latest["m6"]["panorama"])
    p1_mask_path = generation / "P1" / "vertical_valid_mask.png"
    p3_provenance_path = Path(latest["m6"]["pixel_provenance"])

    p1 = cv2.imread(str(p1_path), cv2.IMREAD_COLOR)
    p3 = cv2.imread(str(p3_path), cv2.IMREAD_COLOR)
    p1_mask_image = cv2.imread(str(p1_mask_path), cv2.IMREAD_GRAYSCALE)
    if p1 is None or p3 is None or p1_mask_image is None:
        raise FileNotFoundError(f"could not read P1/P3 inputs for {branch}")
    if p1.shape != p3.shape:
        raise ValueError(f"P1/P3 shape mismatch for {branch}: {p1.shape} != {p3.shape}")
    with np.load(p3_provenance_path, allow_pickle=False) as provenance:
        p3_mask = np.asarray(provenance["valid"], dtype=bool)
    p1_mask = p1_mask_image != 0
    if p1_mask.shape != p3_mask.shape or p1_mask.shape != p1.shape[:2]:
        raise ValueError(f"valid-mask shape mismatch for {branch}")

    common_valid = p1_mask & p3_mask
    valid_values = np.abs(p1.astype(np.int16) - p3.astype(np.int16))[common_valid]
    if valid_values.size == 0:
        raise ValueError(f"no common valid pixels for {branch}")
    per_pixel_max = np.max(
        np.abs(p1.astype(np.int16) - p3.astype(np.int16)), axis=2
    )[common_valid]
    squared_error = np.square(
        p1.astype(np.float64)[common_valid] - p3.astype(np.float64)[common_valid]
    )
    mse = float(np.mean(squared_error))
    psnr = math.inf if mse == 0.0 else float(10.0 * math.log10((255.0**2) / mse))
    signed_rgb = (
        p3.astype(np.float64)[common_valid] - p1.astype(np.float64)[common_valid]
    )[:, ::-1]

    comparison_path = output_root / f"{branch}_P1_vs_P3.png"
    _write_comparison(comparison_path, branch, p1, p3, common_valid)
    common_count = int(np.count_nonzero(common_valid))
    metrics = {
        "branch": branch,
        "generation_id": latest["generation_id"],
        "p1_path": str(p1_path),
        "p3_path": str(p3_path),
        "comparison_path": str(comparison_path),
        "width_px": int(p1.shape[1]),
        "height_px": int(p1.shape[0]),
        "p1_valid_pixels": int(np.count_nonzero(p1_mask)),
        "p3_valid_pixels": int(np.count_nonzero(p3_mask)),
        "valid_mask_mismatch_pixels": int(np.count_nonzero(p1_mask ^ p3_mask)),
        "common_valid_pixels": common_count,
        "changed_pixels": int(np.count_nonzero(per_pixel_max)),
        "changed_pixel_percent": float(100.0 * np.count_nonzero(per_pixel_max) / common_count),
        "change_gt_2_percent": float(100.0 * np.count_nonzero(per_pixel_max > 2) / common_count),
        "change_gt_5_percent": float(100.0 * np.count_nonzero(per_pixel_max > 5) / common_count),
        "change_gt_10_percent": float(100.0 * np.count_nonzero(per_pixel_max > 10) / common_count),
        "mean_absolute_error": float(np.mean(valid_values)),
        "rmse": float(math.sqrt(mse)),
        "psnr_db": psnr,
        "ssim": _ssim(p1, p3, common_valid),
        "absolute_difference_percentiles": {
            "p50": float(np.percentile(valid_values, 50)),
            "p90": float(np.percentile(valid_values, 90)),
            "p95": float(np.percentile(valid_values, 95)),
            "p99": float(np.percentile(valid_values, 99)),
            "maximum": int(np.max(valid_values)),
        },
        "mean_signed_P3_minus_P1_rgb": {
            "r": float(np.mean(signed_rgb[:, 0])),
            "g": float(np.mean(signed_rgb[:, 1])),
            "b": float(np.mean(signed_rgb[:, 2])),
        },
    }
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare sealed S013 P1 and final P3 images.")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    output_root = (args.output or (run_root / "comparisons")).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    rows = [_compare_branch(run_root, output_root, branch) for branch in BRANCHES]
    report = {
        "schema": "s013-p1-p3-comparison/v1",
        "run_root": str(run_root),
        "metric_scope": "common valid pixels; 8-bit BGR decoded lossless PNG",
        "comparison_layout": "P1, P3, absolute-difference heatmap",
        "branches": rows,
    }
    json_path = output_root / "p1_p3_metrics.json"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    csv_path = output_root / "p1_p3_metrics.csv"
    columns = [
        "branch",
        "width_px",
        "height_px",
        "common_valid_pixels",
        "valid_mask_mismatch_pixels",
        "changed_pixel_percent",
        "change_gt_2_percent",
        "change_gt_5_percent",
        "change_gt_10_percent",
        "mean_absolute_error",
        "rmse",
        "psnr_db",
        "ssim",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"metrics": str(json_path), "csv": str(csv_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
