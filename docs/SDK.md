# Gemini 305 全景 SDK 使用指南

`gemini305-rgbd-panorama` 提供面向 Python 集成的稳定 SDK。SDK 只封装已支持的严格 RGB-D 会话校验、合成会话生成和正式全景交付；位姿、接缝和裁剪等安全门限仍由项目的 fail-closed 流水线管理，不向调用方暴露会破坏交付契约的内部实现参数。

当前 SDK 版本：`0.2.0`。

## 安装

Python 版本要求为 3.10–3.12。正式全景构建需要 Open3D、OpenCV、PyYAML，以及可用的 ORB-SLAM3 运行环境。

```powershell
cd D:\central_strip_Panoramic_Camera
D:\Panoramic_Camera\.conda\python.exe -m pip install -e .
```

如需连接 Gemini 305 采集设备，安装采集扩展：

```powershell
D:\Panoramic_Camera\.conda\python.exe -m pip install -e ".[capture]"
```

CUDA 是可选加速能力。若要使用 CuPy CUDA 13 后端：

```powershell
D:\Panoramic_Camera\.conda\python.exe -m pip install -e ".[cuda13]"
```

SDK 构建前应确认导入的是当前工作区：

```powershell
D:\Panoramic_Camera\.conda\python.exe -c "import panorama_demo; print(panorama_demo.__file__, panorama_demo.__version__)"
```

## 快速入门

下面示例生成一个严格 RGB-D 合成会话，并使用 CPU 运行。合成会话适合验证 SDK 集成；正式交付仍需要本机可用的 Open3D 与 ORB-SLAM3。

```python
from pathlib import Path
from panorama_demo import CudaMode, PanoramaSDK, SDKConfig

sdk = PanoramaSDK(SDKConfig(cuda_mode=CudaMode.OFF))
session_dir = sdk.generate_demo(Path("data/sdk_demo"), frame_count=10, frame_width=640, frame_height=400, step=120, scene="plane")
session = sdk.validate_session(session_dir)
print(f"{session.frame_count} frames, {session.frame_width}x{session.frame_height}")

result = sdk.build(session_dir, Path("outputs/sdk_demo"))
print(result.panorama_path)
print(result.delivery_state, result.quality_grade)
if result.manual_review_required:
    print("该结果需要人工复核")
```

成功后只应以 `result.delivery_path` 和其中的 `delivery_state` 判断正式交付状态：`published` 表示 A/B 级交付，`published_degraded` 表示结构安全但要求人工复核的 C 级交付。没有 `delivery.json` 不代表成功。

## CUDA 加速策略

通过 `SDKConfig.cuda_mode` 选择单次 SDK 操作的 CUDA 策略：

| 值 | 行为 |
| --- | --- |
| `CudaMode.PREFER`（默认） | 优先使用可用 CUDA 实现；CUDA 不可用或某个可加速操作失败时，自动使用等价 CPU 路径，并将原因记入审计。 |
| `CudaMode.AUTO` | 针对支持的操作比较 CPU/CUDA 性能与等价性，只保留更快的实现；不可用时使用 CPU。 |
| `CudaMode.OFF` | 强制使用参考 CPU 路径。适用于调试、CI 与可复现实验。 |
| `CudaMode.REQUIRED` | 必须使用可用 CUDA；初始化或已接入 CUDA 操作失败即报错，绝不回退。适用于现场 GPU 验收。 |

`prefer` 与 `auto` 是普通集成推荐值，已经满足“CUDA 可选、失败时继续使用 CPU”的需求。SDK 会在调用结束后恢复进程原有的 `G305_CUDA` 环境变量；同一 Python 进程中的 SDK 计算会串行化 CUDA 策略切换，避免互相污染。

```python
from panorama_demo import CudaMode, PanoramaSDK, SDKConfig

sdk = PanoramaSDK(SDKConfig(cuda_mode=CudaMode.PREFER))
status = sdk.acceleration_status()
print(status["backend"], status["available"], status["mode"])
```

## 自定义配置

SDK 始终以 `configs/demo.yaml` 为基础。可将符合项目安全约束的 YAML 覆盖文件传给 `SDKConfig.config_path`；内部会合并并重新校验，任何放宽正式安全包络的配置都会被拒绝。

```python
from pathlib import Path
from panorama_demo import CudaMode, PanoramaSDK, SDKConfig

sdk = PanoramaSDK(SDKConfig(config_path=Path("configs/my_site_overlay.yaml"), cuda_mode=CudaMode.AUTO))
result = sdk.build("data/captures/run_YYYYMMDD_HHMMSS", "outputs/site_run")
```

`diagnostic_force=True` 仅用于诊断。它不会绕过标定、aligned depth、有限 SE(3)、owner 拓扑、资源上限或原子交付门禁，且诊断成功只产生 `diagnostic_panorama.jpg` 与 `diagnostic_report.json`，不发布 `delivery.json`。

## 完整使用示例

### 处理设备采集的会话

SDK 也可采集 Gemini 305 会话；同步、COLOR_STREAM 对齐、曝光和 manifest 契约仍由经过验证的底层采集器和 YAML 配置执行。随后直接校验并处理：

```python
from pathlib import Path
from panorama_demo import CudaMode, PanoramaProcessingError, PanoramaSDK, SDKConfig

output_dir = Path(r"outputs\production_run")
sdk = PanoramaSDK(SDKConfig(cuda_mode=CudaMode.PREFER))

try:
    # 默认使用逐帧 SOFTWARE_TRIGGERING 的 photo-mode，无预览。
    session_dir = sdk.capture("data/captures", max_frames=120)
    info = sdk.validate_session(session_dir)
    print(f"输入已验证：{info.frame_count} 帧")
    result = sdk.build(session_dir, output_dir)
except PanoramaProcessingError as exc:
    print(f"未交付：{exc}")  # 底层已原子写入 output_dir/failure.json
    raise

if not result.is_published:
    raise RuntimeError("SDK 调用没有发布正式交付")
print(f"全景：{result.panorama_path}")
print(f"报告：{result.report_path}")
print(f"等级：{result.quality_grade}")
if result.manual_review_required:
    print("C 级交付：请人工检查 report.json 后再使用")
```

SDK 的 `capture()` 固定使用照片模式。若需要实时或诊断用途的连续 RGB-D 视频，请从 CLI 调用不带 `--photo-mode` 的 `g305-capture`；该路径独立使用自动曝光（自动快门时间）、自动增益和自动白平衡，并明确标记为非正式全景输入，不能传给 `PanoramaSDK.build()` 或 `g305-panorama` 取得正式交付。

### 加载既有结果

```python
from panorama_demo import PanoramaResult

result = PanoramaResult.load("outputs/production_run")
if result.is_published:
    print(result.delivery_state, result.quality_grade)
elif result.diagnostic_only:
    print("这是诊断输出，不能当作正式交付")
```

可运行的示例文件位于 `examples/sdk_quickstart.py` 与 `examples/sdk_custom_config.py`。

## API 参考

### `CudaMode`

字符串枚举：`PREFER`、`AUTO`、`OFF`、`REQUIRED`。含义见“CUDA 加速策略”。

### `SDKConfig(config_path=None, cuda_mode=CudaMode.PREFER, diagnostic_force=False)`

初始化 SDK 的不可变配置。`config_path` 是可选 YAML 覆盖文件，必须存在且扩展名为 `.yaml` 或 `.yml`；`cuda_mode` 为 CUDA 策略；`diagnostic_force` 必须为布尔值。任何无效字段都会抛出 `SDKConfigurationError`。

### `PanoramaSDK(config=None)`

创建可复用客户端。`config` 必须是 `SDKConfig` 或 `None`。公开属性 `config: SDKConfig` 返回不可变配置，`version: str` 返回 SDK 语义化版本。

#### `acceleration_status()`

返回 `Mapping[str, object]`，包含当前配置下的 `mode`、`available`、`backend`、设备信息、CPU/CUDA 调用计数和回退审计。无可用 GPU 时，`prefer`/`auto` 返回 CPU 状态而不是异常；`required` 会按 fail-closed 规则报错。

#### `validate_session(session)`

验证会话目录或其中的 `frames.csv`，并返回 `SessionSummary`。参数 `session` 为 `str | pathlib.Path`。路径、标定、aligned depth、时间戳、曝光、尺寸或单位契约无效时抛出 `SDKInputError`。

#### `capture(output_dir, *, duration_seconds=None, max_frames=None, photo_mode=True, preview=False)`

从已连接的 Gemini 305 采集一个严格 RGB-D 会话，返回新建会话的根目录 `Path`。默认 `photo_mode=True`，使用逐帧 SOFTWARE_TRIGGERING、每帧一次正式触发且无预览。`output_dir` 为会话父目录；`duration_seconds` 为正数或 `None`；`max_frames` 为正整数或 `None`；`photo_mode` 和 `preview` 必须为布尔值，且照片模式不能开启预览。参数无效时抛出 `SDKInputError`，设备、驱动、写盘或采集契约失败时抛出 `PanoramaProcessingError`。

#### `build(session, output_dir)`

运行正式 RGB-D 全景交付，返回 `PanoramaResult`。`session` 与 `output_dir` 均为 `str | pathlib.Path`；输出路径若为普通文件会抛出 `SDKInputError`。正式链任一门禁未通过时抛出 `PanoramaProcessingError`，并由底层流程原子写入 `failure.json`。

#### `generate_demo(output_dir, *, frame_count=10, frame_width=640, frame_height=400, step=120, scene="plane")`

创建用于集成测试的确定性 RGB-D 合成会话，返回会话根目录 `Path`。`frame_count` 必须至少为 1，`frame_width` 和 `frame_height` 必须至少为 16，`step` 必须为非负整数。`scene` 支持 `plane`、`layered`、`occlusion`、`depth_hole`、`dynamic_object`。非法输入或写入失败抛出 `SDKInputError`。

### `SessionSummary`

只读会话摘要：`root: Path`、`frame_count: int`、`frame_width: int`、`frame_height: int`、`depth_alignment: str`。

### `PanoramaResult`

`build()` 或 `PanoramaResult.load()` 的只读结果，包含 `output_dir`、`panorama_path`、`report_path`、`delivery_path`、`delivery_state`、`quality_grade`、`strict_quality_pass`、`manual_review_required`、`diagnostic_only` 和只读属性 `is_published`。`PanoramaResult.load(output_dir)` 可恢复既有正式或诊断输出；目录既没有完整正式产物也没有完整诊断产物时抛出 `PanoramaProcessingError`。

### 异常层级

```text
PanoramaSDKError
├── SDKConfigurationError  # SDKConfig 或客户端初始化无效
├── SDKInputError          # 调用参数或严格 RGB-D 会话无效
└── PanoramaProcessingError # 处理、发布或结果文件无效/未交付
```

### `get_sdk_version()`

返回当前 SDK 的语义化版本字符串。它与 `panorama_demo.__version__` 保持一致。

## 版本管理与兼容性

包版本以 `src/panorama_demo/version.py` 为单一来源，构建元数据通过 `pyproject.toml` 动态读取它。SDK 采用语义化版本：破坏公开 API 时升级主版本，新增向后兼容 API 时升级次版本，修复时升级补丁版本。

发布前至少运行：

```powershell
D:\Panoramic_Camera\.conda\python.exe -m pytest -q tests/test_sdk.py
ruff check src tests
D:\Panoramic_Camera\.conda\python.exe -m compileall -q src tests
```

完整变更记录见项目根目录的 `CHANGELOG.md`。
