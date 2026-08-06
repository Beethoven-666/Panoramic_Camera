from __future__ import annotations

import importlib

import pytest

from panorama_demo.video_cuda_photometric import (
    CudaPhotometricConfig,
    CudaPhotometricError,
    CudaPhotometricOverlap,
    apply_cuda_global_photometric_correction,
    solve_cuda_global_photometric,
)


def _torch():
    return importlib.import_module("torch")


def _srgb(torch, linear):
    return torch.where(
        linear <= 0.0031308,
        linear * 12.92,
        1.055 * linear.pow(1.0 / 2.4) - 0.055,
    )


def _overlap(torch, *, left, right, safe, risk=None):
    zeros = torch.zeros_like(safe) if risk is None else risk
    return CudaPhotometricOverlap(
        left_frame_id=10,
        right_frame_id=20,
        left_bgr_srgb=left,
        right_bgr_srgb=right,
        left_valid_mask=torch.ones_like(safe),
        right_valid_mask=torch.ones_like(safe),
        safe_background_mask=safe,
        protected_mask=zeros,
        risk_mask=zeros,
    )


@pytest.mark.skipif(not _torch().cuda.is_available(), reason="requires actual CUDA")
def test_cuda_global_photometric_fits_safe_held_out_samples_and_applies_only_colour():
    torch = _torch()
    device = torch.device("cuda:0")
    height, width = 48, 64
    yy, xx = torch.meshgrid(
        torch.linspace(0.0, 1.0, height, device=device),
        torch.linspace(0.0, 1.0, width, device=device),
        indexing="ij",
    )
    canonical = torch.stack((0.20 + 0.20 * xx, 0.25 + 0.16 * yy, 0.30 + 0.08 * xx * yy))
    gain = torch.tensor((1.12, 0.94, 1.06), device=device).view(3, 1, 1)
    bias = torch.tensor((0.020, -0.010, 0.012), device=device).view(3, 1, 1)
    left = _srgb(torch, canonical)
    right = _srgb(torch, (canonical - bias) / gain)
    safe = torch.ones((height, width), dtype=torch.bool, device=device)

    result = solve_cuda_global_photometric(
        torch,
        source_frame_ids=(10, 20),
        overlaps=(_overlap(torch, left=left, right=right, safe=safe),),
    )

    assert result.accepted is True
    correction = result.correction_for_frame(20)
    assert correction.gain_bgr.device.type == "cuda"
    assert correction.bias_bgr.device.type == "cuda"
    assert torch.allclose(correction.gain_bgr, gain[:, 0, 0], atol=2.0e-4)
    assert torch.allclose(correction.bias_bgr, bias[:, 0, 0], atol=2.0e-4)
    output, apply_audit = apply_cuda_global_photometric_correction(
        torch,
        real_source_bgr_srgb=right,
        frame_id=20,
        result=result,
    )
    assert output.device.type == "cuda"
    assert torch.allclose(output, left, atol=3.0e-5)
    assert result.audit["dense_host_transfer_count"] == 0
    assert result.audit["held_out_error_p95_linear"] < 1.0e-4
    assert apply_audit["creates_owner"] is False
    assert apply_audit["mutates_source_tensor"] is False
    assert torch.allclose(right, _srgb(torch, (canonical - bias) / gain), atol=0.0)


@pytest.mark.skipif(not _torch().cuda.is_available(), reason="requires actual CUDA")
def test_cuda_global_photometric_solves_a_three_real_source_graph_once():
    torch = _torch()
    device = torch.device("cuda:0")
    height, width = 48, 64
    yy, xx = torch.meshgrid(
        torch.linspace(0.0, 1.0, height, device=device),
        torch.linspace(0.0, 1.0, width, device=device),
        indexing="ij",
    )
    canonical = torch.stack((0.22 + 0.16 * xx, 0.24 + 0.18 * yy, 0.28 + 0.06 * xx * yy))
    gain_20 = torch.tensor((1.10, 0.95, 1.03), device=device).view(3, 1, 1)
    bias_20 = torch.tensor((0.016, -0.009, 0.008), device=device).view(3, 1, 1)
    gain_30 = torch.tensor((0.91, 1.07, 0.97), device=device).view(3, 1, 1)
    bias_30 = torch.tensor((-0.011, 0.012, -0.006), device=device).view(3, 1, 1)
    source_10 = _srgb(torch, canonical)
    source_20 = _srgb(torch, (canonical - bias_20) / gain_20)
    source_30 = _srgb(torch, (canonical - bias_30) / gain_30)
    safe = torch.ones((height, width), dtype=torch.bool, device=device)
    result = solve_cuda_global_photometric(
        torch,
        source_frame_ids=(10, 20, 30),
        overlaps=(
            _overlap(torch, left=source_10, right=source_20, safe=safe),
            CudaPhotometricOverlap(
                left_frame_id=20,
                right_frame_id=30,
                left_bgr_srgb=source_20,
                right_bgr_srgb=source_30,
                left_valid_mask=torch.ones_like(safe),
                right_valid_mask=torch.ones_like(safe),
                safe_background_mask=safe,
                protected_mask=torch.zeros_like(safe),
                risk_mask=torch.zeros_like(safe),
            ),
        ),
    )

    assert result.accepted is True
    assert result.audit["pair_count"] == 2
    corrected, _audit = apply_cuda_global_photometric_correction(
        torch, real_source_bgr_srgb=source_30, frame_id=30, result=result
    )
    assert torch.allclose(corrected, source_10, atol=3.0e-5)


@pytest.mark.skipif(not _torch().cuda.is_available(), reason="requires actual CUDA")
def test_cuda_global_photometric_uses_median_anchor_and_skip_one_overlap_graph():
    """C7 is a graph fit, not the legacy source-0 correction chain."""

    torch = _torch()
    device = torch.device("cuda:0")
    height, width = 48, 64
    yy, xx = torch.meshgrid(
        torch.linspace(0.0, 1.0, height, device=device),
        torch.linspace(0.0, 1.0, width, device=device),
        indexing="ij",
    )
    canonical = torch.stack((0.22 + 0.15 * xx, 0.25 + 0.14 * yy, 0.30 + 0.06 * xx * yy))
    gain_10 = torch.tensor((0.96, 1.04, 0.98), device=device).view(3, 1, 1)
    bias_10 = torch.tensor((-0.008, 0.006, -0.004), device=device).view(3, 1, 1)
    gain_30 = torch.tensor((1.08, 0.94, 1.03), device=device).view(3, 1, 1)
    bias_30 = torch.tensor((0.012, -0.007, 0.005), device=device).view(3, 1, 1)
    source_20 = _srgb(torch, canonical)
    source_10 = _srgb(torch, (canonical - bias_10) / gain_10)
    source_30 = _srgb(torch, (canonical - bias_30) / gain_30)
    safe = torch.ones((height, width), dtype=torch.bool, device=device)
    result = solve_cuda_global_photometric(
        torch,
        source_frame_ids=(10, 20, 30),
        overlaps=(
            _overlap(torch, left=source_10, right=source_20, safe=safe),
            CudaPhotometricOverlap(
                left_frame_id=20,
                right_frame_id=30,
                left_bgr_srgb=source_20,
                right_bgr_srgb=source_30,
                left_valid_mask=torch.ones_like(safe),
                right_valid_mask=torch.ones_like(safe),
                safe_background_mask=safe,
                protected_mask=torch.zeros_like(safe),
                risk_mask=torch.zeros_like(safe),
            ),
            CudaPhotometricOverlap(
                left_frame_id=10,
                right_frame_id=30,
                left_bgr_srgb=source_10,
                right_bgr_srgb=source_30,
                left_valid_mask=torch.ones_like(safe),
                right_valid_mask=torch.ones_like(safe),
                safe_background_mask=safe,
                protected_mask=torch.zeros_like(safe),
                risk_mask=torch.zeros_like(safe),
                edge_kind="skip_one_overlap",
            ),
        ),
        anchor_frame_id=20,
        anchor_policy="median_exposure",
    )

    assert result.accepted is True
    assert result.audit["anchor_frame_id"] == 20
    assert result.audit["anchor_policy"] == "median_exposure"
    assert result.audit["graph_edge_count"] == 3
    assert result.audit["graph_edge_kinds"] == ["adjacent", "skip_one_overlap"]
    corrected, _audit = apply_cuda_global_photometric_correction(
        torch, real_source_bgr_srgb=source_30, frame_id=30, result=result
    )
    assert torch.allclose(corrected, source_20, atol=3.0e-5)


@pytest.mark.skipif(not _torch().cuda.is_available(), reason="requires actual CUDA")
def test_cuda_global_photometric_rejects_error_visible_only_in_held_out_tiles():
    torch = _torch()
    device = torch.device("cuda:0")
    height = width = 48
    yy, xx = torch.meshgrid(
        torch.linspace(0.0, 1.0, height, device=device),
        torch.linspace(0.0, 1.0, width, device=device),
        indexing="ij",
    )
    linear = torch.stack((0.28 + 0.12 * xx, 0.30 + 0.10 * yy, 0.32 + 0.08 * xx * yy))
    left = _srgb(torch, linear)
    right_linear = linear.clone()
    rows = torch.arange(height, device=device).view(height, 1)
    columns = torch.arange(width, device=device).view(1, width)
    held_out = ((rows // 16 + columns // 16) % 5) == 0
    right_linear[:, held_out] = 0.62
    right = _srgb(torch, right_linear)
    safe = torch.ones((height, width), dtype=torch.bool, device=device)

    result = solve_cuda_global_photometric(
        torch,
        source_frame_ids=(10, 20),
        overlaps=(_overlap(torch, left=left, right=right, safe=safe),),
        config=CudaPhotometricConfig(maximum_held_out_error_p95=0.01, maximum_held_out_error_max=0.02),
    )

    assert result.accepted is False
    assert result.audit["rejection_reason"] == "held_out_error_exceeds_gate"
    for correction in result.corrections:
        assert correction.gain_bgr.device.type == "cuda"
        assert torch.equal(correction.gain_bgr, torch.ones(3, device=device))
        assert torch.equal(correction.bias_bgr, torch.zeros(3, device=device))
    with pytest.raises(CudaPhotometricError, match="rejected"):
        apply_cuda_global_photometric_correction(
            torch,
            real_source_bgr_srgb=right,
            frame_id=20,
            result=result,
        )


@pytest.mark.skipif(not _torch().cuda.is_available(), reason="requires actual CUDA")
def test_cuda_global_photometric_rejects_safe_samples_touching_risk_domain():
    torch = _torch()
    device = torch.device("cuda:0")
    image = torch.full((3, 32, 32), 0.5, device=device)
    safe = torch.ones((32, 32), dtype=torch.bool, device=device)
    risk = torch.zeros_like(safe)
    risk[4, 4] = True
    with pytest.raises(CudaPhotometricError, match="protected or risk"):
        solve_cuda_global_photometric(
            torch,
            source_frame_ids=(10, 20),
            overlaps=(_overlap(torch, left=image, right=image, safe=safe, risk=risk),),
        )
