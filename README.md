# Gemini 305 RGB-D 移动侧扫全景

Python 集成请参阅完整的 [SDK 使用指南](docs/SDK.md)，其中包含安装、CUDA 可选加速、API 参考和示例；版本变更见 [CHANGELOG.md](CHANGELOG.md)。

正式方法名称为**基于轨迹约束的深度感知多视点侧扫拼接**
（**Trajectory-Constrained Depth-Aware Multi-Viewpoint Side-Scan
Mosaicing**）。当前正式实现以 ORB-SLAM3 RGB-D 的真实
`camera_to_world` 节点为唯一全局轨迹，并从同一组原始 RGB-D 独立生成两个产品：

- 固定 `2 mm/pixel` 的 2.5D metric mosaic，通过逐像素 RGB-D 反投影、世界坐标
  变换、正交栅格 z-buffer、时间一致性和深度置信度生成；
- 面向人工巡检的 full-FOV multi-view inspection mosaic，通过沿真实轨迹布置的
  重叠虚拟透视 panel、前景单源 owner、曝光补偿、DIS 安全背景恢复、
  GraphCut 引导的单调相邻 panel 链和受保护 MultiBand 生成。

两个产品都直接读取原始 RGB-D 和同一条不可变轨迹，互不采样、互不补洞。正式
V1 不是固定中央条带 pushbroom，也不是普通二维全景或 TSDF 主图。TSDF 只生成
独立的三维浏览附件，不向任何主图回传结果。

物体的可测位置以 metric 产品中的毫米世界坐标、depth、confidence 和
`owner_frame_id` 为准；inspection 是分段透视浏览图，不能把其二维横坐标解释为
所有深度层共享的毫米位置。正式 inspection 使用 RGB-D 深度保护、相邻 panel 的
单一 RGB owner 和安全背景融合来避免物体缺失或半透明重影。实验性的
`foreground_world_anchor_enabled` 默认保持 `false`：它只有在跨场景物体身份、
至少两个独立 RGB 观察、边界与裁剪审计均完成后才可显式启用，不能以未验证的
局部 RGB 替换冒充真实世界定位。

## CUDA 加速

正式流水线提供独立、可审计的 CUDA 后端。CUDA 13 主机安装 CuPy：

```powershell
D:\Panoramic_Camera\.conda\python.exe -m pip install -e ".[cuda13]"
```

`G305_CUDA=prefer`（默认）优先执行已实现并通过等价审计的 CUDA
remap 和批量几何算子；`G305_CUDA=auto` 会对相同形状的
remap 做一次 CPU/CUDA 等价性与耗时校准，只保留更快的实现；
`G305_CUDA=off` 强制参考 CPU 路径；现场验收可用
`G305_CUDA=required` 强制所有已接入的 remap 与批量 pinhole
几何走 CUDA，任何不支持的采样语义都会 fail closed。报告和交付文件的
`acceleration` 字段记录各阶段设备、调用次数与主机/显存传输量。

当前可加速范围包括正式/诊断 inverse remap、ORB-SLAM3 与 Open3D
输入预处理、metric/inspection 的批量 pinhole 反投影与投影、批量 SE(3)
点变换、inspection 的大批量 sRGB/linear 转换，以及 Open3D Tensor
多尺度 RGB-D odometry。CUDA build 会在 `CUDA:0` 上执行迭代
odometry，并将边明确记录为 `open3d_tensor_cuda_rgbd`；Open3D 0.19
仍仅在 CPU 上提供其 6×6 information-matrix reduction。展示用 TSDF
自动切换到 Tensor `VoxelBlockGrid(CUDA:0)`。ORB-SLAM3、
GraphCut、MultiBand、DIS optical flow、单调 panel-chain 求解和发布审计
没有正式 CUDA 实现，因此继续使用 CPU。`G305_CUDA=required` 只强制已经
接入 CUDA 边界的算子，不会把这些 CPU 算法或 ORB-SLAM3 伪装成 GPU 算法。
官方 CPU-only OpenCV wheel 不是 CUDA remap 的阻塞项：正式 CUDA 边界可由
CuPy 实现；只有实际安装了 CUDA-enabled OpenCV 时才优先使用 `cv2.cuda`。

Windows 的官方 Open3D wheel 不含 CUDA。RTX 50 系主机使用并行安装的
CUDA Toolkit 12.8 构建 Open3D 0.19 Release wheel；CUDA 13.3 可继续供
CuPy 使用。仓库内脚本固定 `sm_120` 和动态 CUDA runtime，并带有 Open3D
0.19 第三方依赖兼容补丁。正式 TSDF 在分配 VoxelBlockGrid 前先以真实
aligned depth、真实 `camera_to_world` 和三层 SDF 支持估计唯一 block；
默认以 `1.85` 安全系数、`2.5 GB` 目标显存、最多 `30,000` blocks 规划
capacity，必要时只允许在展示附件内从 `5 mm` 自适应到最多 `8 mm`：

```powershell
conda create -p D:\open3d_cuda_build\cuda128-toolkit `
  -c nvidia/label/cuda-12.8.1 -c conda-forge cuda-toolkit=12.8.1
.\scripts\build_open3d_cuda_windows.ps1
D:\Panoramic_Camera\.conda\python.exe .\scripts\verify_open3d_cuda.py
```

脚本只生成 wheel，不会自动覆盖现有 Open3D。先在隔离环境验证
`open3d.core.cuda.is_available()`、CUDA Tensor 和 VoxelBlockGrid，
再安装进正式环境。正式 TSDF 在 `prefer` 模式下仅允许在处理真实帧前
因 capability/preflight 不通过而选择 CPU；一旦开始 CUDA 集成，运行期
异常会 fail closed，不会通过降低分辨率、跳帧或静默 CPU 回退来发布。

主环境已经安装 CUDA wheel 和本工作区 editable 包时，日常运行不必设置
环境变量，默认就是 CUDA 优先：

```powershell
& 'D:\Panoramic_Camera\.conda\Scripts\g305-panorama.exe' `
  'D:\central_strip_Panoramic_Camera\data\captures\run_20260725_211255_081' `
  --output 'D:\central_strip_Panoramic_Camera\outputs\greenhouse_sequence'
```

正式物体身份候选层只接受纯 ONNX FastSAM 模型。设置
`G305_FASTSAM_ONNX` 会启用该层；它使用 ONNX Runtime CUDA，并对实际
provider 节点分配做 fail-closed 审计。Conv、ConvTranspose、Gemm、MatMul
等主要计算不得落到 CPU；Shape、Reshape、Gather 等控制节点可在 CPU
执行并写入 `identity_owner_runtime.cuda_execution`。该路径不导入 Torch
或 Ultralytics，也不允许静默整图 CPU 回退。现场强制 CUDA 运行示例：

```powershell
$env:G305_CUDA='required'
$env:G305_FASTSAM_ONNX='D:\models\FastSAM-s.onnx'

& 'D:\Panoramic_Camera\.conda\Scripts\g305-panorama.exe' `
  'D:\central_strip_Panoramic_Camera\data\captures\run_20260727_110952_326' `
  --output 'D:\central_strip_Panoramic_Camera\new_outputs\run_20260727_110952_326_combined_cuda'
```

物体候选依次接受 FastSAM mask、DIS 前后向身份、RGB/Lab、aligned depth
世界位置与多视图一致性审计。只有闭合结构、真实 RGB-D 覆盖和物体边界
光度安全均通过时才允许单一真实 RGB owner；否则明确记录为拒绝或 C 级
hard-cut 建议，不将候选贴入全景、不生成颜色，也不修改真实 pose。

若现场验收要求“已接入 CUDA 的算子、Open3D Tensor odometry 或 TSDF
不能回退”，先设置 `$env:G305_CUDA='required'`；这不会把 ORB-SLAM3、
GraphCut、MultiBand、DIS 或标量拓扑审计伪装成 GPU 算法。

本项目在 Windows 上使用奥比中光 Gemini 305 采集同步、标定且对齐到彩色坐标系的 RGB-D 序列，并生成移动侧扫全景图。正式序列流程是：

```text
严格 RGB-D 会话与可选 sidecar 审计
  → Open3D 相邻 RGB-D odometry（局部几何质量）
  → ORB-SLAM3 RGB-D 完整真实全局轨迹
  → 有限、连续、毫米 camera_to_world SE(3)
  ├─→ 固定 2 mm/pixel 的全场景 2.5D metric 重投影
  │     → RGB / world-normal depth / confidence / hard owner
  └─→ 原始 RGB-D 独立 full-FOV 虚拟透视 panels
        → 深度置信度、局部 depth mesh 和前景单源 owner
        → 安全背景曝光补偿、DIS、GraphCut 单调相邻链、MultiBand
  → 双产品完整性与严格质量审计
  → 输出隔离的 TSDF 浏览附件
  → 原子发布 delivery.json
```


默认工况是相机连续单向水平侧移、场景基本静止、最近物体约 `0.5 m`、最高速度约 `1.5 m/s`。用户只需提供采集目录和输出目录，不需要调整曝光、步长、帧号、位姿、接缝或裁剪参数。

## 正式 A/B/C/F 交付

正式配置的 `stitch.handoff_fallback_policy` 固定为 `publish_degraded=true`、`local_apap_flow_enabled=false`、`manual_review_for_grade_c=true`。这不会放宽 RGB-D 会话、真实轨迹、像素来源或接缝拓扑，只会把结构安全的交付与严格质量结果分开报告：

- **A**：会话、真实轨迹、metric 与 inspection 产品隔离、panel 链、前景单源
  owner、安全背景接缝和全部严格质量门均通过。
- **B**：保留给未来经过正式审计的非 anchor 方法；当前 V1 multiview
  classifier 不产生 B。
- **C**：上述结构审计完整，但输入、轨迹或 inspection 的严格质量未通过。
  它仍发布完整双产品以及必需的 `tsdf_mesh.glb` /
  `tsdf_mesh_viewer.html`，并设置 `delivery_state=published_degraded`、
  `manual_review_required=true`。
- **F**：标定、aligned depth/单位/元数据、真实 SE(3)、必需 RGB-D 边/图连通、
  metric/inspection 产品隔离、panel 链、protected blending、前景 owner、资源
  硬限、TSDF 浏览附件或原子交付任一结构项失败。F 不发布 `delivery.json`，
  只写 `failure.json`。

正式 `report.json` 为 `gemini305-dual-mosaic-report/v11`，
`delivery.json` 为 `gemini305-panorama-delivery/v11`，并公开
`delivery_state`、`strict_quality_pass`、`quality_grade`、
`handoff_fallback_summary`、`foreground_owner_continuity_summary`、
`metric_mosaic`、`inspection_mosaic`、`tsdf_visualization` 和
`manual_review_required`。`quality_pass` 只是严格质量的兼容别名；是否已发布
必须以最后原子写入的 `delivery.json` 及其 `delivery_state` 为准。

## 命令概览

| 命令 | 用途 |
|---|---|
| `g305-capture` | 采集连续流或照片模式驱动的低帧率同步 RGB-D 会话 |
| `g305-panorama` | 正式 RGB-D 序列全景入口 |
| `g305-central-strip-diagnostic` | 独立的参考平面中央条带诊断入口；绝不替代正式 V1 双输出 multiview 路径 |
| `g305-geometry-pair-diagnostic` | 独立的完整序列相邻接缝 RGB A/B 诊断；不读取历史位姿或发布交付 |
| `g305-foreground-deformation-diagnostic` | 默认关闭的前景局部 inverse mesh 实验诊断；可发布相邻 pair A/B 或显式全景诊断图及标量审计 |
| `unistitch-sequence` | 一个版本内保留的弃用别名；运行同一 RGB-D 流程，不含 UniStitch 回退 |
| `generate-panorama-demo` | 生成带标定、对齐深度和已知 SE(3) 轨迹的合成会话 |
| `unistitch-pair` | 独立历史双图诊断工具，不进入正式序列流程 |

### 让 `g305-panorama` 指向当前工作区（首次或切换工作区后执行一次）

本项目的正式入口必须来自当前工作区的源码。不要使用系统 Python 目录中旧的
`g305-panorama.exe`，否则它可能仍会把 TSDF/正交深度投影错误地用作主图，
或走旧 fixed-strip 路径。以下命令把当前
工作区以 editable 方式写入正式 Conda 环境，并将该环境的命令目录放在用户级
`PATH` 的最前面：

```powershell
cd D:\central_strip_Panoramic_Camera

$g305Python = 'D:\Panoramic_Camera\.conda\python.exe'
$g305Scripts = 'D:\Panoramic_Camera\.conda\Scripts'

& $g305Python -m pip install --no-deps -e .

$userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
if (-not (($userPath -split ';') -contains $g305Scripts)) {
  [Environment]::SetEnvironmentVariable('Path', "$g305Scripts;$userPath", 'User')
}
$env:Path = "$g305Scripts;$env:Path"
```

本机的主环境是 `D:\Panoramic_Camera\.conda`。若在另一台机器上通过下文的
bootstrap 脚本创建了项目内 `D:\central_strip_Panoramic_Camera\.conda`，请将
上面两处 `D:\Panoramic_Camera\.conda` 替换为该项目内环境后，再执行同一安装和
`PATH` 设置步骤。

关闭并重新打开 PowerShell 后，下面的命令应显示
`D:\Panoramic_Camera\.conda\Scripts\g305-panorama.exe`；源码路径应显示当前
工作区的 `D:\central_strip_Panoramic_Camera\src\panorama_demo`：

```powershell
(Get-Command g305-panorama).Source
& 'D:\Panoramic_Camera\.conda\python.exe' -c "import panorama_demo; print(panorama_demo.__file__)"
```

激活环境（当前终端也可使用）：

```powershell
conda activate D:\Panoramic_Camera\.conda
```

正式零参数命令：

```powershell
g305-capture --output .\data\captures

g305-capture `
  --photo-mode `
  --output .\data\captures

g305-panorama `
  .\data\captures\run_YYYYMMDD_HHMMSS `
  --output .\outputs\greenhouse_sequence

g305-central-strip-diagnostic `
  .\data\captures\run_YYYYMMDD_HHMMSS `
  --output .\outputs\central_strip_diagnostic

g305-geometry-pair-diagnostic `
  .\data\captures\run_YYYYMMDD_HHMMSS `
  --pair-index 48 `
  --output .\outputs\geometry_pair_48_49

g305-foreground-deformation-diagnostic `
  .\data\captures\run_YYYYMMDD_HHMMSS `
  --config .\configs\foreground_deformation_experiment.yaml `
  --pair-index 48 `
  --output .\outputs\foreground_deformation_pair_48_49

# 对所有相邻 pair 执行相同门禁；仍是诊断产物，不会写 delivery.json
g305-foreground-deformation-diagnostic `
  .\data\captures\run_YYYYMMDD_HHMMSS `
  --config .\configs\foreground_deformation_experiment.yaml `
  --whole-panorama `
  --output .\outputs\foreground_deformation_experimental_panorama
```
`unistitch-sequence` 会打印弃用提示，但调用与 `g305-panorama` 完全相同的 RGB-D `main`。它不会加载 UniStitch、Torch、LightGlue 或 MAGSAC。

`g305-central-strip-diagnostic` 只用于评估“真实 RGB-D 轨迹驱动的参考平面中央条带”是否值得继续研究。它复用严格会话、Open3D 相邻边和 ORB-SLAM3 真实轨迹，但通过内部 renderer callback 与正式路径隔离：`g305-panorama` 不提供算法选项、不会导入该后端，也不会把它作为失败回退。

`configs/demo.yaml` 中的 `stitch.central_strip_diagnostic.enabled` 故意默认为 `false`；它不能通过 `g305-panorama` 打开。独立命令本身是唯一显式 opt-in，并向 renderer 传递一个内部启用的、固定且拒绝未知键的配置副本。

`g305-geometry-pair-diagnostic` 是检查某一条真实相邻接缝的独立 A/B 工具，而不是 `g305-panorama` 的算法开关。它始终先运行完整扫描的 Open3D 相邻边与本次 ORB-SLAM3 RGB-D 轨迹，随后用完整真实 pose-node 链各渲染一次：左栏为关闭局部 geometry 的 baseline，右栏为正常 geometry candidate，最后只裁出该 pair 的共同标定 RGB 走廊。它拒绝 `--render-frame-ids`、`--diagnostic-force`、Open3D-only 轨迹、历史 `render_transforms.json` 和历史 gain；因此不能把两帧当作端点，也不能借旧 pose 发布新的结果。`diagnostic_report.json` 记录左右栏坐标、两套 source/remap/gain 标量和 pair audit；若网格未获准，右栏必须是 hard-owner 回退，而不是强行变形。该命令只写 `diagnostic_panorama.jpg` 与 `diagnostic_report.json`，永不写 `delivery.json`。

`g305-foreground-deformation-diagnostic` 是一个独立、默认关闭的实验分支；必须通过 `configs/foreground_deformation_experiment.yaml` 显式开启。默认模式只检查完整当前 Open3D/ORB-SLAM3 链中的一个相邻 `96–160 px` pair corridor；`--whole-panorama` 则在同一完整链中逐一审计全部相邻 pair，并只把通过门禁且不与其它已接受候选重叠的前景 RGB 采样替换进完成后的 baseline 全景。它只在高置信度、无 split/merge、双源完整覆盖的前景 track 上尝试 16/32 px 的边界固定 inverse mesh；真实接头、端点、遮挡、透明/保护域、非原始分辨率证据、尺度/Jacobian/held-out 门禁任一失败都会保持单源 hard owner。默认 A/B 图的左栏是变形前、右栏是候选后；全景模式仍不做 alpha、MultiBand、APAP、全局 flow、pose 改写或颜色生成。成功也仅写 `diagnostic_panorama.jpg` 与 `diagnostic_report.json`（含 `foreground_deformation_audits`），绝不写 `delivery.json`，更不会改变 v11 A/C/F 语义。

该诊断路线的参考平面也采用 fail-closed 门禁：它必须是唯一主导、跨扫描有足够标定图像面积支持的实测平面；竞争平面、面积不足或结构残差过大只会写 `failure.json`。较严格的平面质量阈值只会令 `strip_quality_pass=false`，仍可留下两个诊断文件供 A/B 检查，绝不变成正式交付。

## 正式处理流程

```text
Gemini 305 同步 RGB-D 会话
  ↓
会话、标定、对齐深度、深度比例和图像尺寸硬校验
  ↓
曝光、清晰度、纹理、主扫描段与相邻视觉运动分析
  ↓
主扫描段全部真实 RGB-D pose nodes（最多 160）
  ↓
Open3D 相邻 RGB-D odometry 局部质量验证
  ↓
ORB-SLAM3 RGB-D 完整短基线序列的真实全局轨迹
  ↓
有限、连通、连续单向的 camera_to_world 4×4 SE(3)
  ↓
metric：原始 RGB-D 全场景世界重投影到固定 2 mm/pixel 2.5D 栅格
  ├─ RGB、毫米 world-normal depth、confidence、hard owner
  ↓
inspection：原始 RGB-D 独立重建 full-FOV 虚拟透视 panels
  ├─ 深度 mesh、可见性与前景单源 owner
  └─ 安全背景曝光补偿、DIS、GraphCut 引导单调链、受保护 MultiBand
  ↓
双产品完整性、质量、资源和产品隔离门禁
  ↓
输出隔离的 TSDF 浏览附件和原子发布
```

缩略图视觉运动只用于主扫描段和 pose-node 布局，不产生正式几何变换。渲染源必须具有 pose graph 中真实优化出的位姿；程序不会按时间戳或二维运动伪造中间位姿。

## Windows 安装

### 前置条件

- Windows 10/11 x64；
- Python `3.10–3.12`，推荐 Conda Python 3.12；
- Git；
- Gemini 305 通过 USB 3 直接连接，采集建议使用 SSD/NVMe；
- 实机采集需要 `pyorbbecsdk2`；
- 正式序列依赖 Open3D 0.19 和可从 Windows 调用的 WSL ORB-SLAM3 RGB-D 示例；不依赖 Torch。CUDA 是默认优先、CPU 可审计回退的加速后端；Open3D Tensor TSDF 要求项目构建的 CUDA wheel。Gemini 305 的 `0.1 mm/unit` 深度必须自动写成 ORB-SLAM3 `DepthMapFactor=10000`，不得使用 TUM 示例的 `5000` 默认值。Open3D 0.19 的官方预编译包支持到 Python 3.12，详见 [Open3D 安装说明](https://www.open3d.org/docs/release/getting_started.html)。

Open3D 在首次执行 RGB-D odometry 时才延迟导入。采集、会话检查和不需要默认 Open3D backend 的单元测试不会因模型库或 CUDA 不可用而提前失败。

### 新机器：创建项目内 Conda 环境（备选）

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\bootstrap_conda.ps1
conda activate .\.conda
```

脚本使用 [`environment.yml`](environment.yml) 创建项目内 `.conda`，并安装基础项目、Open3D、采集依赖和测试依赖。默认不会安装 Torch/Kornia/torchvision，不会克隆 UniStitch/LightGlue，也不会下载模型。若要让未激活环境的新 PowerShell 也能直接识别 `g305-panorama`，创建完成后请回到上方“让 `g305-panorama` 指向当前工作区”一节，用该项目内 `.conda` 路径执行 editable 安装和 `PATH` 设置。

```powershell
# 明确删除并重建项目环境；会移除该环境中已有的包
.\scripts\bootstrap_conda.ps1 -Recreate

# 检查正式入口
python -c "import open3d; print(open3d.__version__)"
g305-capture --help
g305-panorama --help
generate-panorama-demo --help
```

### venv（备选）

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\bootstrap_windows.ps1
.\.venv\Scripts\Activate.ps1
```

也可以手工安装：

```powershell
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[capture,test]"
```

### 可选历史 UniStitch 双图诊断

只有确实需要 `unistitch-pair` 时才安装旧诊断依赖：

```powershell
.\scripts\bootstrap_conda.ps1 -WithUnistitchDiagnostic

# 或在 venv 中
.\scripts\bootstrap_windows.ps1 -WithUnistitchDiagnostic
```

该开关才会安装指定 CUDA PyTorch、`unistitch-diagnostic` extra、检出固定版本的 UniStitch/LightGlue，并下载权重。可同时传 `-SkipModel` 暂不下载权重。手工安装 extra 时，应先按目标机器驱动选择合适的 PyTorch wheel，再执行：

```powershell
python -m pip install -e ".[unistitch-diagnostic]"
```

这些依赖和模型只服务于历史双图工具。它们不会成为 `g305-panorama` 的正式依赖、几何来源或失败回退。

### 注册 Windows metadata

首次在一台电脑上使用相机时运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\register_orbbec_metadata.ps1 `
  -Python D:\Panoramic_Camera\.conda\python.exe
```

若使用项目内 `.conda` 或 `.venv`，可以省略 `-Python`；否则应显式传入主环境 Python，避免脚本误选另一个环境。完成后重新插拔相机。短采集后应检查 `manifest.json` 的 `metadata_support`，以及 `frames.csv` 中曝光、增益、帧号、sensor timestamp 和 RGB-D 时间差是否持续有效。

## 采集 Gemini 305 RGB-D

### 照片模式驱动的低帧率 RGB-D 序列

不新增桌面程序；照片模式复用现有采集入口：

```powershell
g305-capture `
  --photo-mode `
  --output .\data\captures
```

该命令不显示视频。它在配置分辨率下枚举彩色与 Y16 深度 profile，选择两者共同支持的最高 FPS，然后以“单张拍照”作为序列帧持续采集，不增加人工限速。实际帧率由软件触发、完整同步 RGB-D 收帧、对齐和同步写盘共同决定，因此表现为较低帧率的视频式采集。按 `Q`、`Esc` 或 `Ctrl+C` 结束；也可使用已有的 `--duration` 或 `--max-frames`：

```powershell
g305-capture --photo-mode --max-frames 120 --output .\data\captures
g305-capture --photo-mode --duration 20 --output .\data\captures
```

如果不存在尺寸一致、可同步的 RGB-D profile，会在发出正式触发前失败，不会退化为只拍 RGB。

准备阶段执行以下固定安全流程：

```text
停止旧流
  → SOFTWARE_TRIGGERING
  → Trigger Out Enable = true
  → frames_per_trigger = 1
  → 物理 Trigger Out gate 关闭
  → 用不会到达外部端口的软件触发预热 RGB-D Pipeline
  → 确认能够取得新的完整同步 RGB-D 帧并回读配置
  → 等待预热触发完全静默且门控稳定
  → 开始逐帧照片序列
```

预热可能需要内部 `trigger_capture()`，但预热期间物理 gate 必须保持关闭；在打开 gate 前还会等待一个完整静默窗口并排空迟到帧，因此内部触发不能成为序列的外部脉冲。无法关闭或回读物理 gate、无法关闭设备定时自动采集、预热/静默验证失败或同步配置回读不一致时，照片模式拒绝就绪。

序列中的每一帧都具有严格的一对一语义：

- 每帧只调用一次正式 `device.trigger_capture()`；`SOFTWARE_TRIGGERING`、`Trigger Out Enable=true` 和 `frames_per_trigger=1` 始终保持有效；
- 等待该触发产生的全新、完整同步 RGB-D 帧，不把预热帧、积压旧帧或只有彩色的帧计入序列；
- 上一帧收帧、对齐和落盘完成后才发下一次触发，不存在并发触发；
- 任一帧的触发调用、收帧、曝光元数据、解码、对齐或写盘失败都会立即终止序列，不自动重触发，也不发布 formal session。

每个成功序列帧都向当前 RGB-D session 追加一张彩色图、一张对齐深度图和一条 `frames.csv` 记录；会话同时包含 `calibration.json`、深度比例、时间戳、逐帧曝光及同步元数据。结束时程序停止 Pipeline、恢复设备设置并最终更新 `manifest.json` 的实际帧数、有效采集 FPS 和 clean-shutdown 状态。正在写入、强制终止或关闭/恢复失败的会话会保持 `formal_stitch_allowed=false`，严格会话加载器也会拒绝 `clean_shutdown!=true`，不能发布部分扫描。

照片模式的安全默认值位于 `capture.photo_mode`：功能默认启用；自动选择共同最高 RGB-D FPS；彩色曝光 `800 µs`；Trigger Out 延时固定为 `17000 µs`；正式触发最长等待 `8000 ms`；Gemini 固件预热最多允许 8 次仅限 gate-off 的内部触发，每次等待 `1500 ms`。取得完整预热 RGB-D 后，物理 gate 仍保持关闭，直到从最后一次内部触发起完整 `8000 ms` 迟到响应窗口结束并确认队列为空；逐帧收帧、对齐和同步写盘完成后才允许下一次正式触发，天然覆盖该硬件延时；gate 状态改变后等待 `250 ms`。这些值是内部安全默认值，普通用户不需要为了得到正式会话而调整它们。

照片序列与下面的连续流模式都必须独占同一台 Gemini 305；运行前关闭 OrbbecViewer、Flash 工具和其它占用相机的进程。

### 连续移动 RGB-D 序列

正式采集使用 [`configs/demo.yaml`](configs/demo.yaml)：

```powershell
g305-capture --output .\data\captures
```

默认配置：

- RGB 与对齐深度均为 `1280×800@30`；
- 软件 D2C 对齐和帧同步开启；
- `PRIMARY` 外部同步输出开启，默认输出与帧率一致的 30 Hz 脉冲；
- 彩色自动曝光和 AWB 在预热期间开启，曝光上限 `800 µs`；固件不支持 AE 上限时退回固定 `800 µs`；
- 预热后回读并锁定 exposure、sensor gain、AWB/white-balance；丢弃过渡帧，只有连续两帧的这四项 metadata 与锁定读回值一致才开始正式扫描。无法回读、锁定或验证即 fail-closed；
- JPEG 质量 95，只默认保存对齐深度，原始深度仅供显式诊断；
- 异步写盘队列为 64 帧，丢帧和写入错误会记录到 manifest。

先采 300 帧进行约 10 秒冒烟测试：

```powershell
g305-capture `
  --output .\data\captures `
  --max-frames 300
```

无预览固定时长采集：

```powershell
g305-capture `
  --output .\data\captures `
  --duration 20 `
  --no-preview
```

预览窗口中按 `Q` 或 `Esc` 停止，也可使用 `Ctrl+C`。采集器会回读验证同步输出和曝光属性；无法验证时明确失败，不会伪造成功。

### 曝光边界

Gemini 305 彩色曝光 metadata 的原始单位按 `100 µs/单位` 解释。`frames.csv` 中的 `color_exposure` 保留设备原始值，例如 `8` 表示 `800 µs`，`301` 表示约 `30.1 ms`，不是 `301 µs`。

正式移动序列的输入拒绝上限是 `1200 µs`。在 `1.5 m/s` 下，`800 µs` 内相机移动约 `1.2 mm`；现场仍过暗时应增加连续补光，而不是提高正式曝光门限。连续三帧曝光超过采集安全上限时，正式采集器应停止。

需要观察设备原生长曝光行为时，必须显式进入诊断模式：

```powershell
g305-capture `
  --output .\data\captures `
  --diagnostic-unrestricted-auto-exposure

# 等价的一体化诊断配置
g305-capture `
  --config .\configs\capture_unrestricted_auto_exposure.yaml `
  --output .\data\captures
```

该模式会解除项目的 `800 µs` AE 上限、请求并回读设备当前 profile 允许的最大值，并把会话标记为 `diagnostic_only=true`、`formal_stitch_allowed=false`。它只用于诊断，不能发布正式交付。

### 会话输出

```text
data/captures/run_YYYYMMDD_HHMMSS/
├─ manifest.json
├─ calibration.json
├─ frames.csv
├─ color/
│  └─ 00000000.jpg
├─ depth_aligned/
│  └─ 00000000.png
└─ depth_raw/                 仅显式 --raw-depth 时存在
   └─ 00000000.png
```

`depth_aligned/*.png` 是 16 位设备深度单位，不可直接假定为毫米。必须使用每行 `depth_scale_mm_per_unit` 转换；项目内部投影与报告始终使用毫米。

## 严格 RGB-D 会话契约

`g305-panorama` 只接受会话目录或该会话的 `frames.csv`。只含 RGB 图片的目录、单独 `color/`、旧合成 RGB 序列和任意图片列表都不能进入正式流程。

每个正式源帧必须具备：

- 可解码的 RGB 图；
- `frames.csv` 中明确且非空、位于 `depth_aligned/` 的 `aligned_depth_path`；
- 有限、正数的 `depth_scale_mm_per_unit`；
- 每帧非负彩色时间戳和正数 `color_exposure` 元数据；
- 与 RGB 和彩色内参尺寸完全相同的对齐深度；
- `calibration.json` 中有限、有效的彩色内参和畸变；
- 标定中明确的 color-target 对齐声明，或本项目 v1 捕获器的严格 `software → COLOR_STREAM` provenance；
- 有效的相对路径，路径不得逃逸会话目录。

`raw_depth_path` 不能替代 `aligned_depth_path`；即使文件尺寸相同也会拒绝。缺少标定、标定损坏、主点越界、RGB/深度尺寸不一致、深度比例错误、深度未对齐或 raw depth 冒充 aligned depth 都是结构性失败，`--diagnostic-force` 也不能绕过。

RGB 使用线性插值去畸变，对齐深度使用最近邻。无效边缘由独立几何 `valid_mask` 表示；RGB 像素为黑色不代表无效。深度进入项目代码时先明确换算为毫米，只有 Open3D 适配层临时转换为米，适配层返回时再换回毫米。

会话可选提供 `camera.yaml` 与 `transforms.json` 作为 V1 输入审计 sidecar。
存在 `camera.yaml` 时，它必须严格声明 `1280×800`、OpenCV 内参/畸变、
RGB 对齐和 depth scale，并与 `calibration.json` 及每一行 `frames.csv`
一致。存在 `transforms.json` 时，它必须提供毫米、有限刚性的
`camera_to_world`、ORB-SLAM3 provenance、edge residuals 和被用帧的完整
tracked coverage。Sidecar 只参与审计，绝不静默替代本次 ORB-SLAM3；两者
缺失时继续使用原有严格会话路径。

## RGB-D odometry 与 Pose Graph

局部相邻边 backend 固定为 `open3d_rgbd`，全局正式轨迹固定来自
ORB-SLAM3 RGB-D；Open3D pose graph 不能替代缺失的 ORB-SLAM3 全局轨迹。
局部边参考 [Open3D RGB-D odometry](https://www.open3d.org/docs/latest/tutorial/pipelines/rgbd_odometry.html)。

相邻 pose node 必须有可靠 RGB-D 边；预计仍有真实重叠的非相邻节点最多跨两个节点增加弱边。这些边仍由 RGB-D odometry 得到，不能用特征匹配补边。每条边审计：

- `source_to_reference` 4×4 SE(3)；
- 收敛状态、fitness、RMSE 与有限、对称、正定的 6×6 信息矩阵；
- 两端有效深度比例；
- 有限性、旋转正交性和行列式；
- 平移、垂直/前后漂移、旋转与扫描方向；
- 图优化后的边残差。

坐标与单位约定：

- 彩色相机坐标采用 OpenCV/Open3D 约定：`+x` 向右、`+y` 向下、`+z` 向前；
- `camera_to_world` 把相机坐标映射到第一个 pose node 的相机坐标系；
- 所有项目侧平移和 RMSE 使用毫米；
- `transforms.json` 中不会出现任何全局/pose/全景级 `3×3` homography，也不会有插值位姿；即使启用 `local_apap_flow`，其局部候选也绝不写入该文件或参与轨迹。

正式流程拒绝必需相邻边失败、图不连通、非有限/非刚体 SE(3)、可靠节点不足两帧、不可审计残差、无效 RGB remap 或不完整 owner 拓扑等结构失败。物理不连续/逆向、步长或扫描跨度异常、上下/前后漂移、旋转及 RGB-D 边残差越界同样使真实轨迹无效，必须为 F。只有在这些结构门均通过后，低 fitness/RMSE、输入清晰度与最终渲染覆盖等严格质量门限未过，才会在 v11 中形成透明的 C 级人工复核，而不是伪造轨迹或静默放宽门限。

## 固定尺度 2.5D metric mosaic

Metric 产品使用主扫描段的全部真实 ORB-SLAM3 pose nodes。每个有效 RGB-D
样本先按彩色内参反投影到相机三维，再使用该帧不可变的
`camera_to_world` 变换到世界坐标，最后写入固定 `2 mm/pixel` 的侧扫正交
栅格。z-buffer、局部深度支持、时间同层支持、遮挡冲突和深度边共同决定
winner 与 confidence。

每个有效 metric 像素严格包含一个真实 RGB owner、毫米
world-normal depth 和 uint16 confidence。无效 depth 在 EXR 中写为 NaN；
不会用外观、inspection 或 TSDF 补洞，也不会平均多个 RGB owner。正式产物为
`mosaic_metric.png`、`mosaic_depth.exr`、`mosaic_confidence.png`、
`mosaic_owner.png` 和 `mosaic_meta.json`。

## Full-FOV multi-view inspection mosaic

Inspection 产品从原始全分辨率 RGB-D 和同一条真实轨迹独立重建，不读取 metric
raster 或 TSDF。renderer 沿稳健侧扫轴布置重叠的虚拟透视 panels，并为每个
panel 选择真实 full-FOV RGB-D 源。参考背景在相邻 panel 间映射到一致的展示
位置；近景保留真实视差，再由深度置信度、可见性和局部 depth mesh 决定安全
采样。任何 pose 都不会被插值或二维累计，正式路径也不会退化为固定中央条带。

前景、软管、叶片、遮挡边界、透明/反光内容和不可靠深度保护域必须由单个真实
frame/panel hard owner 提供；前景 blend 像素必须为零。只有共同有效且不在
protected mask 内的安全背景才可进入曝光补偿和 MultiBand。DIS optical flow
只允许恢复平坦、RGB 一致、前后向误差通过的无效深度背景，不能移动前景或
改写 pose。

“单一 owner”不等于“物体完整”。正式 renderer 还会从全部真实 RGB-D pose
建立独立的近景世界体素集合，再检查最终 inspection 的 RGB owner 是否仍能
回溯到这些实测表面。多视图可见近景的覆盖率固定要求至少 `80%`；低于该值时
必须记录
`multiview_observed_near_world_surface_missing_from_final_rgb_owner` 并降为
C 级人工复核，不能仅凭接缝闭合和每像素唯一 owner 发布 A 级。该审计只读，
不提供颜色、不补洞、不改 pose，也不向 metric 或 TSDF 回传结果。

不要手工给每件物体指定二维全景坐标。可测位置始终来自 RGB-D 世界坐标；若要
在 inspection 中锁定整件近景物体，还必须先得到跨视图稳定的实例身份和完整
轮廓，再选择一个完整、清晰的真实 RGB owner，以 direct SE(3) 投影一次。
纯深度连通块可能把相接的物体与货架合并，也可能把同一设备按深度层拆开；
逐像素“最佳视图”则会把一件物体分给多个 owner。这两种结果都不能作为正式
物体锁定依据。

OpenCV GraphCut 提供颜色/梯度接缝提示，但正式 owner 会被投影到一个从左到右、
只连接相邻 panel 的单调全高链；非相邻 owner、回跳、未覆盖像素或 protected
blend 都是结构失败。随后只在安全背景 owner 边界运行最多三层 MultiBand，
保护域和其它区域直接复制唯一 owner。最终裁剪使用独立 valid mask 的
`largest_valid_rectangle()`，不会因 RGB 为黑色而误删有效内容。正式产物为
`mosaic_inspection.png`、`inspection_owner.png` 和
`inspection_meta.json`；`panorama.jpg` 只是同一 inspection 视觉结果的兼容
JPEG。

Metric、inspection 与 TSDF 三条输出路径相互隔离。Depth 可以决定 metric
几何、inspection 可见性和 owner，但不生成 inspection RGB 颜色；TSDF 只在
两个主产品完成自身审计后生成浏览附件，绝不参与主图的几何、接缝、融合、裁剪
或质量判定。

## 配置安全默认值

普通用户不需要修改 [`configs/demo.yaml`](configs/demo.yaml)。关键内部默认值如下；它们是 fail-closed 安全起点，仍需通过合成数据和现场静止、`0.5`、`1.0`、`1.5 m/s` 验收后才能确认具体硬件/场景的交付范围。

| 项目 | 默认值 |
|---|---:|
| pose backend | `hybrid_orbslam3_rgbd`（Open3D 局部边 + ORB-SLAM3 全局轨迹） |
| odometry working width | `640` |
| 有效深度比例下限 | `0.10` |
| RGB-D fitness 下限 | `0.15` |
| RGB-D RMSE 上限 | `50 mm` |
| 单边平移 / 垂直 / 前后上限 | `750 / 80 / 120 mm` |
| 单边旋转上限 | `6°` |
| 总垂直 / 前后漂移上限 | `120 / 150 mm` |
| 总旋转上限 | `10°` |
| 边平移 / 旋转残差上限 | `30 mm / 2°` |
| pose node 硬预算 | `160` |
| metric 尺度 / 深度范围 | `2.0 mm/pixel` / `200–3000 mm` |
| metric confidence 满支持视图数 | `2` |
| metric depth-edge confidence cap | `0.25` |
| inspection 渲染源 | 沿真实轨迹自动选择重叠 full-FOV panels |
| inspection 前景保护 | reference ratio `0.70`；margin `max(60 mm, 8%)`；guard `12 px` |
| inspection depth mesh | cell `8 px`；Jacobian `0.01–64`；边界 margin `1 px` |
| GraphCut / 相邻链 | preview `0.25`；corridor `128 px`；每行最大步长 `1 px` |
| DIS 安全背景门 | preview `0.25`；motion `≤2 px`；FB error `≤1 px`；RGB residual `≤12` |
| MultiBand | 仅安全背景，最多 `3` 层；protected intersection 必须为 `0` |
| metric / inspection 画布上限 | 各 `200 MP` |

`metric_mosaic` 与 `inspection_multiview` 都不能在正式交付中关闭。配置中的
`sequence_blend_mode=calibrated_rgb_pushbroom`、`calibrated_rgb_pushbroom`
和旧 `scan_seam` 键仍由兼容校验器保留，并供 `--diagnostic-force` 及独立诊断
callback 使用；它们不是非诊断 V1 的正式主图 renderer。正式 V1 固定调用
metric renderer 与 full-FOV inspection renderer。`200 MP`、160 pose nodes、
有限真实 SE(3)、双产品隔离、单一前景 owner、protected blend 为零以及原子
发布仍是不可放宽的结构门。手工 `render_frame_ids` 只允许诊断；正式命令会
拒绝它。

## 产物、报告和原子交付

正式成功目录：

```text
outputs/greenhouse_sequence/
├─ panorama.jpg
├─ mosaic_metric.png
├─ mosaic_depth.exr
├─ mosaic_confidence.png
├─ mosaic_owner.png
├─ mosaic_meta.json
├─ mosaic_inspection.png
├─ inspection_owner.png
├─ inspection_meta.json
├─ tsdf_mesh.glb
├─ tsdf_mesh_viewer.html
├─ transforms.json
├─ render_transforms.json
├─ report.json
└─ delivery.json
```

- `panorama.jpg`：`mosaic_inspection.png` 的兼容 JPEG；不是 metric、TSDF 或第三套 renderer；
- `mosaic_metric.png` / `mosaic_depth.exr` / `mosaic_confidence.png` /
  `mosaic_owner.png`：固定 `2 mm/pixel` 的 RGB、毫米 depth、uint16 confidence
  和 frame-id owner；`mosaic_meta.json` schema 为
  `gemini305-metric-mosaic/v1`；
- `mosaic_inspection.png` / `inspection_owner.png`：full-FOV multiview
  inspection 与其 frame-id owner；`inspection_meta.json` schema 为
  `gemini305-inspection-mosaic/v1`；
- `transforms.json`：`rgbd-pose-graph/v1`，包含坐标约定、毫米单位、pose nodes 的 4×4 `camera_to_world`、RGB-D 边、信息矩阵、残差、优化和连通状态；
- `render_transforms.json`：`trajectory-constrained-rgbd-multiview/v1`，声明
  真实 SE(3) 源、full-FOV panel 选择、无 pose 插值、metric/TSDF 不参与
  inspection RGB，以及深度置信度和背景接缝审计；
- `report.json`：`gemini305-dual-mosaic-report/v11`，汇总严格会话、
  `v1_input_sidecars`、Open3D odometry、ORB-SLAM3 轨迹、metric/inspection
  manifest、前景 owner、背景 seam、CUDA provenance、TSDF 和 publication；
- `delivery.json`：`gemini305-panorama-delivery/v11`，最后发布并列出
  `products.metric`、`products.inspection`、`projection`、`seam_backend`、
  `blend_backend`、`acceleration`、`tsdf_visualization` 以及 A/C/F 状态。

每次任务先使旧 `delivery.json` 失效；双产品、GLB、Viewer、报告、位姿和
`delivery.json` 都先写隐藏 pending 文件，再用 `os.replace` 发布，
`delivery.json` 始终最后写入。C 级仍强制人工复核，并同样交付成对的 TSDF
展示文件。TSDF 不是主图，但其生成、Viewer 或任意发布步骤失败仍会清除正式
文件并原子写入 `failure.json`。没有有效 `delivery.json` 就不是有效交付。

旧诊断文件会在新的正式或失败任务开始时清除。历史 `pairs/` 不是交付目录，不能用于判断本次任务是否成功。

`g305-central-strip-diagnostic` 成功时严格只原子发布 `diagnostic_panorama.jpg` 和 `diagnostic_report.json`（schema: `gemini305-central-strip-diagnostic/v1`）。它绝不写 `panorama.jpg`、`report.json`、`transforms.json`、`delivery.json`、TSDF mesh 或其它正式交付文件；普通异常同样只留下 `failure.json`。ORB-SLAM3 的临时 staging 位于系统临时目录，不会保留在成功输出目录。

## `--diagnostic-force` 的边界

无限制自动曝光会话只能生成诊断结果：

```powershell
g305-panorama `
  .\data\captures\run_20260713_184519 `
  --output .\outputs\run_20260713_184519_diagnostic `
  --diagnostic-force
```

也可以传入一体化诊断配置：

```powershell
g305-panorama `
  .\data\captures\run_20260713_184519 `
  --config .\configs\capture_unrestricted_auto_exposure.yaml `
  --output .\outputs\run_20260713_184519_diagnostic
```

诊断模式可以绕过：

- 输入绝对清晰度、曝光和整体画质门限；
- 正式 RGB-D odometry 边质量门限；
- pose 轨迹质量门限；
- 最终图像画质门限。

它不能绕过：

- 有效标定、彩色对齐深度和深度单位；
- 有限 SE(3) 和 pose graph 连通；
- 有效 RGB inverse remap；
- 严格 owner 拓扑；
- 画布和 aggregate working-set 限制；
- 原子交付语义。

诊断成功只写 `diagnostic_panorama.jpg` 和 `diagnostic_report.json`，绝不写 `panorama.jpg`、正式 JSON 或 `delivery.json`。即使使用了 `--diagnostic-force`，结构性失败仍会明确报错，而不是强行出图。

`run_20260713_184519` 的 `color_exposure=301`，约 `30.1 ms`，远高于 `1200 µs` 正式移动安全门限，因此只能按上面的诊断命令测试。旧温室会话 `run_20260711_213054` 同样只适合作为输入门禁应拒绝的回归样本。源帧已经丢失的纹理不能靠融合恢复。

2026-07-13 使用 Open3D 0.19 和上述 `--diagnostic-force` 命令复测 `run_20260713_184519`：12 条必需相邻 RGB-D 边均收敛，fitness 为 `0.613–0.856`、RMSE 为 `17.4–26.2 mm`；旧深度 renderer 随后因高风险带横断完整相邻 pair corridor 而正确失败。该历史结果不能作为当前 V1 dual multiview 成功样本，也不是算法回退点；应通过补光、降低速度并重新采集解决。

## 无相机合成 RGB-D 数据

```powershell
generate-panorama-demo `
  --output .\data\synthetic\demo `
  --frames 10 `
  --width 640 `
  --height 400 `
  --step 120 `
  --scene layered
```

可选场景：

- `plane`：单平面；
- `layered`：近远两层与真实横移视差；
- `occlusion`：遮挡边界；
- `depth_hole`：对齐深度空洞；
- `dynamic_object`：动态物体失败/风险回归。

合成会话包含 `calibration.json`、`color/`、`depth_aligned/`、带 `aligned_depth_path` 与 `depth_scale_mm_per_unit` 的 `frames.csv`，以及 manifest 中已知的毫米 `camera_to_world` 轨迹。它适合验证严格会话、单位、真实 SE(3) 交接、固定尺度 metric、metric/inspection 产品隔离、前景 hard owner、黑色有效内容和原子交付语义，但不能代替 Gemini 305、Open3D 实机 odometry、现场照明或速度验收。

## 现场验收

新采集数据至少分别验证静止、`0.5 m/s`、`1.0 m/s` 和 `1.5 m/s`：

- `queue_drops == 0`、`write_errors == 0`，写盘队列无持续堆积；
- RGB-D 时间戳同步且没有回退；
- 彩色曝光 metadata 保持在正式上限内；
- 对齐深度有效率、尺寸和单位正确；
- pose graph 连通、真实 SE(3)、连续单向侧移、步长/跨度、漂移、旋转与必需边残差均在物理安全范围内；低 fitness/RMSE 等严格质量不足才可作为 C 级原因；
- metric 的 `2 mm/pixel` 坐标、world-normal depth、confidence 和 owner 对齐；
- inspection 中最近约 `0.5 m` 物体没有半透明重影或 panel 重复，前景组件保持单一 owner；
- inspection 的曝光连续性、GraphCut 单调相邻链、safe-background MultiBand、上下抖动和最终四边通过人工复核，且 protected blend intersection 为 `0`；
- 输出目录存在最后发布的 v11 `delivery.json`，并人工核对 `products`、`delivery_state`、`quality_grade`、`strict_quality_pass` 和 `manual_review_required`。

合成测试通过不等于实机验收完成。动态物体、镜面反射、完全无纹理、严重欠光、深度大面积空洞或源帧已拖影的场景可能形成 C 级人工复核或 F；结构失败仍 fail-closed，不应通过放宽门限、回退平均或未通过审计的 APAP/flow 来掩盖。

`outputs/greenhouse_sequence_optimized/` 是 `2026-07-16` 留存的历史 CLI 输出：101 个真实源、
`2978×782`、裁剪高度 `97.75%`、融合区 `1.264%`。它的 JSON sidecar 早于当前局部网格审计契约
（`/v2`，不含 geometry-assist 和 A/B/C handoff 字段），因此只能作为旧路径与真实轨迹的历史证据，不能当作本版本
`trajectory-constrained-rgbd-multiview/v1`、
`gemini305-dual-mosaic-report/v11` 与
`gemini305-panorama-delivery/v11` 交付的验收结果。当前 Windows 主机恢复
WSL/ORB-SLAM3 后必须重新执行正式 CLI
才能发布当前 v11 schema 的 `delivery.json`。仅存在 `tsdf_mesh.glb` 和 Viewer
是当前正式交付的正常现象；只有 `panorama.jpg`、`mosaic_inspection.png` 或
其 metadata 声称颜色来自 TSDF、orthographic metric raster 或 fixed-strip
renderer，才说明仍调用了错误主图路径。

同日的 [`outputs/greenhouse_geometry_assisted_direct_20260716_v2/diagnostic_panorama.jpg`](outputs/greenhouse_geometry_assisted_direct_20260716_v2/diagnostic_panorama.jpg)
以及 2026-07-17 的 101-node pushbroom 回放/交付都属于正式 V1 dual
multiview 之前的历史证据。它们可以证明当时的会话和真实轨迹，但不能验收当前
metric、full-FOV inspection、产品隔离或 v11 原子交付。当前 V1 实机验收必须
重新运行 CLI，并同时检查八个双输出文件、TSDF 浏览附件、v11
`report.json` 和最后写入的 `delivery.json`。

2026-07-27 使用
`data/captures/run_20260725_211255_081` 完成当前 V1 的完整实拍正式回归。
145 个真实 RGB-D pose nodes 全部由 ORB-SLAM3 跟踪，144 条相邻 Open3D
RGB-D 边全部使用 CUDA；最新巡检图为 `2598×717`，15 个 full-FOV 透视
panel，所有 14 条相邻 owner 边界通过，owner-only 保护像素与融合区交集为
`0`。metric 产品保持 `2 mm/pixel`、
单一 RGB owner、毫米 depth/confidence/坐标 sidecar；展示 TSDF 使用
Open3D Tensor CUDA，`5 mm`、`20,567` block capacity、`14,074` active
blocks。此前仅依据 owner 拓扑和接缝审计得到的
`delivery_state=published`、`quality_grade=A` 已被新增的独立世界覆盖审计
否证，不能继续作为当前正式验收结论：最新正式回归耗时 `205.28 s`，全部近景
世界体素覆盖率为 `32.829%`，至少两视图观测体素覆盖率为 `43.128%`，明显低于
正式 `80%` 门限。当前代码会
将同类结果发布为 `published_degraded`、`quality_grade=C` 并要求人工复核；
在自动跨视图实例身份与完整轮廓问题解决前，不得宣称“所有物体完整”或 A 级。

同日另以该真实 145-node 轨迹只缩放累计纵向位移执行了明确标记为
`renderer_performance_only` 的 20 m 规模压力回放：inspection 为
`78.29 s`，metric 为 `56.00 s`。它证明当前 renderer 的规模与耗时余量，
但 RGB-D 像素仍来自约 `1.69 m` 的真实会话，不能冒充真实 20 m 的跟踪、
几何或物体完整性验收。最终 20 m / 5 min 现场门仍必须用一套真实 20 m
同步 RGB-D 会话重新运行正式 CLI，并以当次真实 `report.json` 中的
`pose_quality.metrics.scan_span_mm`、`elapsed_seconds` 和最终
`delivery.json` 为准。

## 常见问题

### `Open3D is required ... but could not be imported`

确认正在使用项目环境，并重新安装基础依赖：

```powershell
python -m pip install -e ".[capture,test]"
python -c "import open3d; print(open3d.__version__)"
```

### 找不到 `g305-panorama`，或它指向旧全局入口

先按上方“让 `g305-panorama` 指向当前工作区”完成 editable 安装和用户级 `PATH`
设置，关闭并重新打开 PowerShell；随后验证：

```powershell
(Get-Command g305-panorama).Source
python -c "import panorama_demo; print(panorama_demo.__file__)"
```

前者应为主 Conda 环境的 `Scripts\g305-panorama.exe`，后者应位于当前工作区的
`src\panorama_demo`。不要用任意系统 Python 重装同名命令。

### 正式目录没有 `delivery.json`

任务为 F 或只运行了诊断模式。检查 `failure.json`、标准错误或 `diagnostic_report.json`；不要把 `panorama.jpg` 是否存在、旧 `pairs/` 或 JPEG 黑色像素当作成功判断。相反，C 级会有 `delivery.json`，但其中 `delivery_state=published_degraded`、`quality_grade=C` 且 `manual_review_required=true`，不能按 A/B 自动验收。

### GraphCut 或 MultiBand 报错

经审计、完整有效的最低代价 hard cut 会作为 C 级发布，不会自动 Feather、平均或补洞。GraphCut/MultiBand 的结构错误、缺少真实共同 RGB 覆盖、owner 拓扑不完整、有效掩码不完整或超出 aggregate working set 则仍是 F。检查报告与 `failure.json`；若源数据不足，应补光、降低速度或重新采集。

### Windows App Control 阻止 `c10.dll`

`c10.dll` 属于可选历史 Torch/UniStitch 诊断依赖，不应进入正式 `g305-panorama`。若只运行正式 RGB-D 流程，请使用未安装 `unistitch-diagnostic` 的基础环境；若确需历史双图工具，需要由系统策略放行组织认可的签名 Python/PyTorch 环境。

## 测试

```powershell
python -m pytest -q
ruff check src tests
python -m compileall -q src tests
git diff --check
```

修改相机、Open3D、GraphCut、MultiBand、性能或交付语义时，必须明确区分：纯单元/合成验证、历史失败数据回归、真实 Open3D 运行和 Gemini 305 现场验收。

## 第三方项目与许可证

- [Open3D](https://github.com/isl-org/Open3D) 提供正式 RGB-D odometry 与 pose graph 能力；
- [OpenCV](https://opencv.org/) 提供图像处理、GraphCut seam finder 和 MultiBand blender；
- [OrbbecSDK v2 Python wrapper](https://github.com/orbbec/pyorbbecsdk2) 提供 Gemini 305 采集；
- UniStitch 与 LightGlue 仅为可选历史双图诊断依赖。

UniStitch、LightGlue 与 Orbbec wrapper 的固定来源和许可证说明见 [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)；Open3D 与 OpenCV 许可证以各自上游发布为准。
