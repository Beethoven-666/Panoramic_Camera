"""Process an existing capture session with a site-specific YAML overlay."""

from pathlib import Path

from panorama_demo import CudaMode, PanoramaSDK, SDKConfig


def main() -> None:
    sdk = PanoramaSDK(
        SDKConfig(
            config_path=Path("configs/my_site_overlay.yaml"),
            cuda_mode=CudaMode.AUTO,
        )
    )
    result = sdk.build(
        Path("data/captures/run_YYYYMMDD_HHMMSS"),
        Path("outputs/site_run"),
    )
    if result.is_published:
        print(result.delivery_state, result.quality_grade)
    elif result.diagnostic_only:
        print("diagnostic output only")


if __name__ == "__main__":
    main()
