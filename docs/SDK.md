# Gemini 305 全景 SDK 使用指南

`gemini305-rgbd-panorama` 提供面向 Python 集成的稳定 SDK。照片产品 `PanoramaSDK` 只封装严格 RGB-D 会话校验、合成会话生成和正式全景交付；位姿、接缝和裁剪等安全门限仍由 fail-closed 流水线管理。

当前候选 SDK 版本：`0.3.0rc1`，Runtime 仅支持 CPython 3.10。
Linux 离线安装遵循 [INSTALL_LINUX.md](INSTALL_LINUX.md)，受控平台见
[SUPPORT_MATRIX.md](SUPPORT_MATRIX.md)。下面源码开发示例须在 CPython 3.10 环境使用；
`capture` extra 的 wrapper metadata variant 应先由离线包安装。

## 安装

```powershell
cd D:\central_strip_Panoramic_Camera
D:\Panoramic_Camera\.conda\python.exe -m pip install -e .
D:\Panoramic_Camera\.conda\python.exe -m pip install -e ".[capture,cuda13]"
```

SDK 构建前应确认导入的是当前工作区：

```powershell
D:\Panoramic_Camera\.conda\python.exe -c "import panorama_demo; print(panorama_demo.__file__, panorama_demo.__version__)"
```

## 照片 SDK 快速入门

```python
from pathlib import Path
from panorama_demo import CudaMode, PanoramaSDK, SDKConfig

sdk = PanoramaSDK(SDKConfig(cuda_mode=CudaMode.OFF))
session_dir = sdk.generate_demo(
    Path("data/sdk_demo"), frame_count=10, frame_width=640,
    frame_height=400, step=120, scene="plane",
)
session = sdk.validate_session(session_dir)
print(f"{session.frame_count} frames, {session.frame_width}x{session.frame_height}")

result = sdk.build(session_dir, Path("outputs/sdk_demo"))
print(result.panorama_path, result.delivery_state, result.quality_grade)
```

只应以 `result.delivery_path` 和其中的 `delivery_state` 判断正式交付状态：`published` 表示 A/B 级，`published_degraded` 表示结构安全但要求人工复核的 C 级。没有 `delivery.json` 就没有正式交付。

## 照片 SDK CUDA 策略

| 值 | 行为 |
| --- | --- |
| `CudaMode.PREFER`（默认） | 优先使用可用 CUDA；不可用时使用等价 CPU 路径并记录审计。 |
| `CudaMode.AUTO` | 对已支持操作比较 CPU/CUDA 性能与等价性；不可用时使用 CPU。 |
| `CudaMode.OFF` | 强制参考 CPU 路径，适用于调试和 CI。 |
| `CudaMode.REQUIRED` | 已接入 CUDA 的边界必须使用 CUDA，失败即报错，不回退。 |

SDK 会在调用结束后恢复进程原有的 `G305_CUDA`，并串行化同一进程中的 CUDA 策略切换。

## 照片 SDK 自定义配置与采集

`SDKConfig.config_path` 是相对 `configs/demo.yaml` 合并的 YAML 覆盖；底层仍执行正式安全校验。`diagnostic_force=True` 不能绕过严格会话、标定、aligned depth、有限 SE(3)、owner 拓扑、资源上限或原子交付。

```python
from panorama_demo import CudaMode, PanoramaProcessingError, PanoramaSDK, SDKConfig

sdk = PanoramaSDK(SDKConfig(cuda_mode=CudaMode.PREFER))
try:
    # 默认逐帧 SOFTWARE_TRIGGERING photo-mode，无预览。
    session_dir = sdk.capture("data/captures", max_frames=120)
    info = sdk.validate_session(session_dir)
    result = sdk.build(session_dir, "outputs/production_run")
except PanoramaProcessingError:
    # 正式失败由底层写 failure.json，不发布 delivery.json。
    raise
```

`PanoramaSDK.capture()` 默认并正式支持照片模式。连续 RGB-D 视频不能传给 `PanoramaSDK.build()` 或 `g305-panorama`；它使用下面独立的视频 facade。

## 照片 SDK API

- `SDKConfig(config_path=None, cuda_mode=CudaMode.PREFER, diagnostic_force=False)`：不可变初始化配置。
- `PanoramaSDK.acceleration_status()`：返回当前 CUDA policy 与后端审计。
- `PanoramaSDK.validate_session(session)`：返回 `SessionSummary`；严格输入无效时抛出 `SDKInputError`。
- `PanoramaSDK.capture(output_dir, *, duration_seconds=None, max_frames=None, photo_mode=True, preview=False)`：采集同步 Gemini 305 RGB-D 会话。
- `PanoramaSDK.build(session, output_dir)`：返回 `PanoramaResult`，失败时抛出 `PanoramaProcessingError`。
- `PanoramaSDK.generate_demo(...)`：创建确定性严格合成 RGB-D 会话。
- `PanoramaResult.load(output_dir)`：从既有正式或诊断输出恢复只读结果。
- `get_sdk_version()`：返回与 `panorama_demo.__version__` 相同的版本。

异常层级保持不变：

```text
PanoramaSDKError
├── SDKConfigurationError
├── SDKInputError
└── PanoramaProcessingError
```

可运行的照片示例位于 `examples/sdk_quickstart.py` 与 `examples/sdk_custom_config.py`。

## 连续视频 SDK

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
exists. Doctor cannot issue or infer software, hardware or release readiness; its compatibility
fields are fixed to null. Qualification requires independently validated acceptance artifacts.
The current phase stops at development bundle portability, with all readiness states false.

Typed jobs, errors, cancellation, ownership, salvage/emergency and retention contracts are documented
in [LINUX_SDK_JOB_LIFECYCLE.md](LINUX_SDK_JOB_LIFECYCLE.md). Emergency output is separate from formal P3.
Only successful SDK-owned inputs meeting all requested delivery and warning gates can be retained
or cleaned according to policy; external sessions are never deleted.
