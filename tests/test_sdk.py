from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from panorama_demo import (
    CudaMode,
    PanoramaProcessingError,
    PanoramaResult,
    PanoramaSDK,
    SDKConfig,
    SDKConfigurationError,
    SDKInputError,
    get_sdk_version,
)


def _write_formal_output(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "panorama.jpg").write_bytes(b"jpeg")
    (output / "report.json").write_text("{}", encoding="utf-8")
    (output / "delivery.json").write_text(
        json.dumps(
            {
                "delivery_state": "published_degraded",
                "quality_grade": "C",
                "strict_quality_pass": False,
                "manual_review_required": True,
            }
        ),
        encoding="utf-8",
    )


def test_sdk_configuration_validates_yaml_and_cuda_mode(tmp_path: Path) -> None:
    config = tmp_path / "custom.yaml"
    config.write_text("stitch: {}\n", encoding="utf-8")

    actual = SDKConfig(config_path=config, cuda_mode="off")

    assert actual.config_path == config.resolve()
    assert actual.cuda_mode is CudaMode.OFF
    with pytest.raises(SDKConfigurationError, match="cuda_mode"):
        SDKConfig(cuda_mode="gpu")
    invalid_extension = tmp_path / "custom.json"
    invalid_extension.write_text("{}", encoding="utf-8")
    with pytest.raises(SDKConfigurationError, match="YAML"):
        SDKConfig(config_path=invalid_extension)


def test_generate_demo_and_validate_session(tmp_path: Path) -> None:
    client = PanoramaSDK(SDKConfig(cuda_mode=CudaMode.OFF))

    session = client.generate_demo(
        tmp_path / "session", frame_count=2, frame_width=32, frame_height=32, step=4
    )
    summary = client.validate_session(session)

    assert summary.root == session.resolve()
    assert summary.frame_count == 2
    assert (session / "depth_aligned" / "00000000.png").is_file()


def test_build_uses_public_configuration_and_returns_delivery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = PanoramaSDK(SDKConfig(cuda_mode=CudaMode.OFF))
    session = client.generate_demo(
        tmp_path / "session", frame_count=2, frame_width=32, frame_height=32, step=4
    )
    output = tmp_path / "output"
    from panorama_demo import stitch_sequence

    def fake_run(args: object) -> dict[str, object]:
        assert getattr(args, "input") == session.resolve()
        assert getattr(args, "output") == output.resolve()
        assert getattr(args, "diagnostic_force") is False
        _write_formal_output(output)
        return {}

    monkeypatch.setattr(stitch_sequence, "run", fake_run)

    result = client.build(session, output)

    assert result.is_published is True
    assert result.quality_grade == "C"
    assert result.manual_review_required is True
    assert result.delivery_path == output / "delivery.json"


def test_build_rejects_missing_session_before_pipeline(tmp_path: Path) -> None:
    client = PanoramaSDK()

    with pytest.raises(SDKInputError, match="does not exist"):
        client.build(tmp_path / "missing", tmp_path / "output")


def test_capture_builds_validated_capture_arguments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = PanoramaSDK(SDKConfig(cuda_mode=CudaMode.OFF))
    from panorama_demo import capture_orbbec

    def fake_capture(args: object) -> Path:
        assert getattr(args, "output") == (tmp_path / "captures").resolve()
        assert getattr(args, "max_frames") == 3
        assert getattr(args, "photo_mode") is True
        assert getattr(args, "no_preview") is True
        return tmp_path / "captures" / "run_1"

    monkeypatch.setattr(capture_orbbec, "run_capture", fake_capture)

    assert client.capture(tmp_path / "captures", max_frames=3) == tmp_path / "captures" / "run_1"
    with pytest.raises(SDKInputError, match="positive"):
        client.capture(tmp_path / "captures", max_frames=0)
    with pytest.raises(SDKInputError, match="does not support preview"):
        client.capture(tmp_path / "captures", photo_mode=True, preview=True)


def test_result_loads_diagnostic_artifacts(tmp_path: Path) -> None:
    output = tmp_path / "diagnostic"
    output.mkdir()
    (output / "diagnostic_panorama.jpg").write_bytes(b"jpeg")
    (output / "diagnostic_report.json").write_text("{}", encoding="utf-8")

    result = PanoramaResult.load(output)

    assert result.diagnostic_only is True
    assert result.is_published is False


def test_result_rejects_incomplete_delivery_metadata(tmp_path: Path) -> None:
    output = tmp_path / "incomplete"
    output.mkdir()
    (output / "panorama.jpg").write_bytes(b"jpeg")
    (output / "report.json").write_text("{}", encoding="utf-8")
    (output / "delivery.json").write_text("{}", encoding="utf-8")

    with pytest.raises(PanoramaProcessingError, match="delivery_state"):
        PanoramaResult.load(output)


def test_cuda_policy_is_scoped_to_sdk_call(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("G305_CUDA", "prefer")
    client = PanoramaSDK(SDKConfig(cuda_mode=CudaMode.OFF))

    status = client.acceleration_status()

    assert status["mode"] == "off"
    assert os.environ["G305_CUDA"] == "prefer"


def test_version_is_exposed_by_sdk() -> None:
    assert get_sdk_version() == "0.2.0"
    assert PanoramaSDK().version == "0.2.0"
