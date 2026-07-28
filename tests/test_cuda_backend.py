from __future__ import annotations

import cv2
import numpy as np
import pytest

from panorama_demo import cuda_backend


@pytest.fixture(autouse=True)
def _restore_cuda_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("G305_CUDA", "off")
    cuda_backend.cuda_status(refresh=True)
    yield
    cuda_backend.cuda_status(refresh=True)


def test_cpu_remap_matches_opencv_exactly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("G305_CUDA", "off")
    cuda_backend.cuda_status(refresh=True)
    source = np.arange(96 * 80 * 3, dtype=np.uint8).reshape(80, 96, 3)
    yy, xx = np.indices((73, 91), dtype=np.float32)
    map_x = xx + 0.375
    map_y = yy + 0.125

    actual = cuda_backend.remap(source, map_x, map_y, cv2.INTER_LINEAR)
    expected = cv2.remap(
        source,
        map_x,
        map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    np.testing.assert_array_equal(actual, expected)
    assert cuda_backend.cuda_metadata()["backend"] == "cpu"


def test_default_policy_prefers_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("G305_CUDA", raising=False)

    status = cuda_backend.cuda_status(refresh=True)

    assert status.mode == "prefer"


def test_prefer_uses_cuda_even_for_small_supported_operation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("G305_CUDA", "prefer")
    if not cuda_backend.cuda_status(refresh=True).available:
        pytest.skip("CUDA runtime is unavailable")
    cuda_backend.reset_cuda_audit()
    source = np.arange(64, dtype=np.uint8).reshape(8, 8)
    yy, xx = np.indices(source.shape, dtype=np.float32)

    actual = cuda_backend.remap(source, xx, yy, cv2.INTER_NEAREST)

    np.testing.assert_array_equal(actual, source)
    counters = cuda_backend.cuda_metadata()["counters"]
    assert counters["cupy_calls"] + counters["opencv_cuda_calls"] == 1
    assert counters["cpu_calls"] == 0


def test_cuda_remap_parity_on_real_device(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("G305_CUDA", "auto")
    status = cuda_backend.cuda_status(refresh=True)
    if not status.available:
        pytest.skip("CUDA runtime is unavailable")
    rng = np.random.default_rng(20260726)
    source = rng.integers(0, 256, (320, 384, 3), dtype=np.uint8)
    yy, xx = np.indices((300, 360), dtype=np.float32)
    map_x = xx + 0.37
    map_y = yy + 0.21
    expected = cv2.remap(
        source,
        map_x,
        map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    monkeypatch.setenv("G305_CUDA", "required")
    cuda_backend.cuda_status(refresh=True)

    actual = cuda_backend.remap(source, map_x, map_y, cv2.INTER_LINEAR)

    delta = np.abs(actual.astype(np.int16) - expected.astype(np.int16))
    assert int(delta.max(initial=0)) <= 1
    assert float(np.percentile(delta, 99.0)) == 0.0
    metadata = cuda_backend.cuda_metadata()
    assert metadata["available"] is True
    assert metadata["counters"]["cupy_calls"] + metadata["counters"][
        "opencv_cuda_calls"
    ] >= 1


def test_cuda_nearest_depth_is_exact(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("G305_CUDA", "auto")
    if not cuda_backend.cuda_status(refresh=True).available:
        pytest.skip("CUDA runtime is unavailable")
    source = np.arange(320 * 384, dtype=np.float32).reshape(320, 384)
    yy, xx = np.indices((300, 360), dtype=np.float32)
    map_x = xx + 0.2
    map_y = yy + 0.2
    expected = cv2.remap(
        source,
        map_x,
        map_y,
        cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    monkeypatch.setenv("G305_CUDA", "required")
    cuda_backend.cuda_status(refresh=True)

    actual = cuda_backend.remap(source, map_x, map_y, cv2.INTER_NEAREST)

    np.testing.assert_array_equal(actual, expected)


def test_required_cuda_remap_supports_replicate_border(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("G305_CUDA", "auto")
    if not cuda_backend.cuda_status(refresh=True).available:
        pytest.skip("CUDA runtime is unavailable")
    source = np.arange(64 * 80 * 3, dtype=np.uint8).reshape(64, 80, 3)
    yy, xx = np.indices((68, 84), dtype=np.float32)
    map_x = xx - 2.25
    map_y = yy - 1.75
    expected = cv2.remap(
        source,
        map_x,
        map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    monkeypatch.setenv("G305_CUDA", "required")
    cuda_backend.cuda_status(refresh=True)

    actual = cuda_backend.remap(
        source,
        map_x,
        map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )

    delta = np.abs(actual.astype(np.int16) - expected.astype(np.int16))
    assert int(delta.max(initial=0)) <= 1


def test_required_cuda_uint16_depth_remap_is_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("G305_CUDA", "auto")
    if not cuda_backend.cuda_status(refresh=True).available:
        pytest.skip("CUDA runtime is unavailable")
    source = np.arange(320 * 384, dtype=np.uint16).reshape(320, 384)
    yy, xx = np.indices((300, 360), dtype=np.float32)
    monkeypatch.setenv("G305_CUDA", "required")
    cuda_backend.cuda_status(refresh=True)

    actual = cuda_backend.remap(
        source, xx + 0.2, yy + 0.2, cv2.INTER_NEAREST
    )
    expected = cv2.remap(source, xx + 0.2, yy + 0.2, cv2.INTER_NEAREST)

    np.testing.assert_array_equal(actual, expected)
    assert cuda_backend.cuda_metadata()["counters"]["cupy_calls"] >= 1


def test_required_cuda_rejects_unsupported_remap_dtype(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("G305_CUDA", "auto")
    if not cuda_backend.cuda_status(refresh=True).available:
        pytest.skip("CUDA runtime is unavailable")
    source = np.ones((320, 384), dtype=np.float64)
    yy, xx = np.indices((300, 360), dtype=np.float32)
    monkeypatch.setenv("G305_CUDA", "required")
    cuda_backend.cuda_status(refresh=True)

    with pytest.raises(RuntimeError, match="CUDA remap was required"):
        cuda_backend.remap(source, xx, yy, cv2.INTER_NEAREST)


def test_cuda_float_remap_preserves_nan_border(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("G305_CUDA", "auto")
    if not cuda_backend.cuda_status(refresh=True).available:
        pytest.skip("CUDA runtime is unavailable")
    source = np.arange(320 * 384, dtype=np.float32).reshape(320, 384)
    yy, xx = np.indices((300, 360), dtype=np.float32)
    map_x = xx - 8.25
    map_y = yy - 4.5
    expected = cv2.remap(
        source,
        map_x,
        map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=np.nan,
    )
    monkeypatch.setenv("G305_CUDA", "required")
    cuda_backend.cuda_status(refresh=True)

    actual = cuda_backend.remap(
        source,
        map_x,
        map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=np.nan,
    )

    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    finite = np.isfinite(expected)
    np.testing.assert_allclose(actual[finite], expected[finite], atol=1e-5)


def test_pinhole_cuda_geometry_matches_numpy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("G305_CUDA", "auto")
    if not cuda_backend.cuda_status(refresh=True).available:
        pytest.skip("CUDA runtime is unavailable")
    rng = np.random.default_rng(42)
    depth = rng.uniform(250.0, 3000.0, 100_000)
    u = rng.uniform(0.0, 1279.0, depth.size)
    v = rng.uniform(0.0, 799.0, depth.size)
    points = cuda_backend.pinhole_unproject(
        u, v, depth, fx=910.0, fy=909.0, cx=640.0, cy=400.0
    )
    projected_u, projected_v = cuda_backend.pinhole_project(
        points, fx=910.0, fy=909.0, cx=640.0, cy=400.0
    )

    np.testing.assert_allclose(projected_u, u, atol=1e-10, rtol=1e-12)
    np.testing.assert_allclose(projected_v, v, atol=1e-10, rtol=1e-12)


def test_required_pinhole_initializes_cuda_lazily(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("G305_CUDA", "required")
    monkeypatch.setattr(cuda_backend, "_STATUS", None)
    monkeypatch.setattr(cuda_backend, "_CUPY", None)
    u = np.arange(100_000, dtype=np.float64)
    depth = np.full(u.shape, 1000.0)

    points = cuda_backend.pinhole_unproject(
        u, u, depth, fx=900.0, fy=900.0, cx=640.0, cy=400.0
    )

    assert points.shape == (100_000, 3)
    assert cuda_backend.cuda_metadata()["counters"]["cupy_calls"] >= 1


def test_cuda_linear_rgb_roundtrip_matches_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("G305_CUDA", "auto")
    if not cuda_backend.cuda_status(refresh=True).available:
        pytest.skip("CUDA runtime is unavailable")
    rng = np.random.default_rng(20260726)
    image = rng.integers(0, 256, (320, 384, 3), dtype=np.uint8)
    monkeypatch.setenv("G305_CUDA", "off")
    cuda_backend.cuda_status(refresh=True)
    expected_linear = cuda_backend.srgb_to_linear_bgr(image)
    expected_encoded = cuda_backend.linear_to_srgb_bgr(expected_linear)
    monkeypatch.setenv("G305_CUDA", "required")
    cuda_backend.cuda_status(refresh=True)

    actual_linear = cuda_backend.srgb_to_linear_bgr(image)
    actual_encoded = cuda_backend.linear_to_srgb_bgr(actual_linear)

    np.testing.assert_allclose(actual_linear, expected_linear, atol=2e-7, rtol=2e-6)
    np.testing.assert_array_equal(actual_encoded, expected_encoded)
