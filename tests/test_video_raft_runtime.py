"""Tests for the candidate-only local torchvision RAFT-small runtime."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from panorama_demo.video_raft_runtime import (
    RAFTSmallRuntimeConfig,
    RAFTSmallRuntimeError,
    RAFTSmallTensorFlowResult,
    TorchvisionRAFTSmallRuntime,
    verify_local_raft_small_weights,
)


torch = pytest.importorskip("torch")


class _MockRAFT(torch.nn.Module):
    def __init__(self, *, non_finite: bool = False) -> None:
        super().__init__()
        self.marker = torch.nn.Parameter(torch.tensor(1.0))
        self.non_finite = non_finite

    def forward(self, first: torch.Tensor, second: torch.Tensor) -> list[torch.Tensor]:
        del second
        flow = torch.zeros((1, 2, first.shape[-2], first.shape[-1]), device=first.device)
        if self.non_finite:
            flow[..., 0, 0] = float("nan")
        return [flow + self.marker * 0.0]


def _checkpoint(path: Path) -> str:
    torch.save(_MockRAFT().state_dict(), path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_runtime_uses_verified_explicit_local_weight_and_returns_audited_flow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "raft_small.pth"
    digest = _checkpoint(path)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    factory_calls: list[tuple[object, object]] = []

    def factory(*, weights: object, progress: object) -> _MockRAFT:
        factory_calls.append((weights, progress))
        return _MockRAFT()

    runtime = TorchvisionRAFTSmallRuntime(
        RAFTSmallRuntimeConfig(weights_path=path, weights_sha256=digest),
        model_factory=factory,
    )
    source = np.zeros((10, 14, 3), dtype=np.uint8)
    target = np.full((10, 14, 3), 12, dtype=np.uint8)
    result = runtime.estimate_pair(source, target, source_frame_id=20, target_frame_id=21)

    assert factory_calls == [(None, False)]
    assert result.flow_xy.shape == (10, 14, 2)
    assert result.flow_xy.dtype == np.float32
    assert np.all(result.flow_xy == 0.0)
    assert result.audit.device == "cpu"
    assert result.audit.precision == "float32"
    assert result.audit.padded_height == 16
    assert result.audit.padded_width == 16
    assert result.audit.downloaded is False
    assert result.audit.as_dict()["weights_sha256"] == digest


def test_runtime_rejects_missing_or_mismatched_local_weights(tmp_path: Path) -> None:
    missing = tmp_path / "missing.pth"
    with pytest.raises(RAFTSmallRuntimeError, match="required"):
        verify_local_raft_small_weights(missing, "0" * 64)

    path = tmp_path / "raft_small.pth"
    _checkpoint(path)
    with pytest.raises(RAFTSmallRuntimeError, match="mismatch"):
        verify_local_raft_small_weights(path, "0" * 64)


def test_runtime_fail_closed_for_invalid_pair_or_nonfinite_flow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "raft_small.pth"
    digest = _checkpoint(path)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    runtime = TorchvisionRAFTSmallRuntime(
        RAFTSmallRuntimeConfig(weights_path=path, weights_sha256=digest),
        model_factory=lambda **_: _MockRAFT(non_finite=True),
    )
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    with pytest.raises(RAFTSmallRuntimeError, match="non-finite"):
        runtime.estimate_pair(image, image, source_frame_id=1, target_frame_id=2)
    with pytest.raises(RAFTSmallRuntimeError, match="distinct"):
        runtime.estimate_pair(image, image, source_frame_id=1, target_frame_id=1)


def test_resident_tensor_api_returns_device_flow_without_host_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "raft_small.pth"
    digest = _checkpoint(path)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    runtime = TorchvisionRAFTSmallRuntime(
        RAFTSmallRuntimeConfig(weights_path=path, weights_sha256=digest),
        model_factory=lambda **_: _MockRAFT(),
    )
    source = torch.zeros((3, 10, 14), dtype=torch.uint8)
    target = torch.full((3, 10, 14), 0.25, dtype=torch.float32)

    # Patch after model construction: an accidental ``.cpu()`` inside the
    # resident API must fail rather than being hidden by the CPU test device.
    def forbid_cpu(self: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
        del self, args, kwargs
        raise AssertionError("resident RAFT tensor API must not call Tensor.cpu()")

    monkeypatch.setattr(torch.Tensor, "cpu", forbid_cpu)
    result = runtime.estimate_pair_tensors(
        source, target, source_frame_id=30, target_frame_id=31
    )

    assert isinstance(result, RAFTSmallTensorFlowResult)
    assert result.flow_xy.shape == (10, 14, 2)
    assert result.flow_xy.dtype == torch.float32
    assert result.flow_xy.device == source.device
    audit = result.audit.as_dict()
    assert audit["output_residency"] == "device_tensor"
    assert audit["host_transfer_count"] == 0
    assert audit["flow_finite"] is None
    assert audit["flow_finite_audit"] == "deferred_no_d2h"


def test_resident_tensor_api_rejects_nonresident_or_malformed_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "raft_small.pth"
    digest = _checkpoint(path)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    runtime = TorchvisionRAFTSmallRuntime(
        RAFTSmallRuntimeConfig(weights_path=path, weights_sha256=digest),
        model_factory=lambda **_: _MockRAFT(),
    )
    image = torch.zeros((3, 8, 8), dtype=torch.float32)
    with pytest.raises(RAFTSmallRuntimeError, match="already-resident"):
        runtime.estimate_pair_tensors(np.zeros((3, 8, 8), dtype=np.uint8), image, source_frame_id=1, target_frame_id=2)
    with pytest.raises(RAFTSmallRuntimeError, match="CHW"):
        runtime.estimate_pair_tensors(image.unsqueeze(0), image.unsqueeze(0), source_frame_id=1, target_frame_id=2)
