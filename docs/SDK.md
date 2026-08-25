# Gemini 305 Python SDK

`PanoramaSDK` remains the photo-product API. Continuous video uses the separate
`Gemini305VideoSDK`; it exposes machine/runtime settings but no renderer, seam, candidate, fallback
or pose knobs.

```python
from pathlib import Path
from panorama_demo import Gemini305VideoSDK, VideoSDKConfig

sdk = Gemini305VideoSDK(VideoSDKConfig(
    cuda_mode="required",
    orbslam3_root=Path("/home/zyh/opt/g305-orbslam3"),
))

doctor = sdk.doctor()
result_2d = sdk.process_session(
    session=Path("/data/L1-FROZEN-001"),
    output_dir=Path("/data/output/2d"),
    defer_3d=True,
)
result_3d = sdk.run_post_3d(result_2d.session, result_2d.output_dir)
```

For physical capture:

```python
job = sdk.capture_and_process(
    capture_root=Path("/data/captures"),
    output_dir=Path("/data/output/2d"),
    width=848, height=480, fps=60, duration_seconds=3,
    video_exposure_us=100, preview_window=False,
)
result_2d = job.result()
```

`VideoProcessingJob.cancel()` is cooperative and works while waiting for camera or capturing.
The default waits indefinitely for hot-plug; pass `wait_for_camera=False` only when immediate
failure is desired.

`VideoPanoramaResult` loads the actual P3, provenance, report, timing and atomic delivery marker.
It rejects Preview or non-`ignore_pose` authority. `VideoThreeDResult` is separate; post-3D failure
raises `PanoramaProcessingError` while preserving the published 2-D delivery.

The wheel contains byte-identical immutable V11/baseline locks, configurations and approval
resources. It does not contain NVIDIA drivers/toolkit, Orbbec system rules, ORB-SLAM3 binaries or
Vocabulary. `sdk.doctor()` reports these external requirements separately, and intentionally keeps
ORB distribution licensing blocked until an approved GPL distribution plan or commercial license
exists. A missing camera does not by itself negate `SDK_SOFTWARE_READY`.
