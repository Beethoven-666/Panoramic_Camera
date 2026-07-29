"""Create a demo RGB-D session and process it through the public SDK."""

from pathlib import Path

from panorama_demo import CudaMode, PanoramaSDK, SDKConfig


def main() -> None:
    sdk = PanoramaSDK(SDKConfig(cuda_mode=CudaMode.PREFER))
    session = sdk.generate_demo(Path("data/sdk_demo"), frame_count=10)
    print(sdk.validate_session(session))
    result = sdk.build(session, Path("outputs/sdk_demo"))
    print(result.panorama_path)
    print(result.delivery_state, result.quality_grade)


if __name__ == "__main__":
    main()
