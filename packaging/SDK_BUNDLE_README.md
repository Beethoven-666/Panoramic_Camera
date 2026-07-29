# Gemini 305 RGB-D Panorama SDK

这是正式 SDK 源码包，不含测试、采集数据、运行输出、历史双图工具或独立诊断 CLI。

## 安装

请在解压后的目录内以 editable 方式安装。当前默认配置由项目根目录的
`configs/demo.yaml` 提供，因此必须保留压缩包的目录结构。

```powershell
python -m pip install -e .
```

连接 Gemini 305 相机时：

```powershell
python -m pip install -e ".[capture]"
```

CUDA 加速是可选的：

```powershell
python -m pip install -e ".[cuda13]"
```

## 最小调用

```python
from panorama_demo import CudaMode, PanoramaSDK, SDKConfig

sdk = PanoramaSDK(SDKConfig(cuda_mode=CudaMode.PREFER))
session = sdk.capture("data/captures", max_frames=120)  # 默认 photo-mode
result = sdk.build(session, "outputs/result")
print(result.delivery_state, result.quality_grade)
```

SDK 的采集入口固定默认使用无预览的 photo-mode：每个正式帧只触发一次，完整收取、对齐和落盘后再继续下一帧。

`CudaMode.PREFER` 与 `CudaMode.AUTO` 在 CUDA 不可用或某个 GPU 操作失败时自动使用等价 CPU 路径；`CudaMode.REQUIRED` 会强制 CUDA 并在失败时终止。

详细 API 参考和完整示例见 `docs/SDK.md`。
