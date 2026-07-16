# Gemini 305 RGB-D 移动侧扫全景

本项目在 Windows 上使用奥比中光 Gemini 305 采集同步、标定且对齐到彩色坐标系的 RGB-D 序列，并生成移动侧扫全景图。正式序列流程是：

```text
短基线 Open3D RGB-D Odometry（局部质量验证）
  → ORB-SLAM3 RGB-D（完整序列的真实全局相机轨迹）
  → 真实 SE(3) 校平的 RGB 流式中央窄条（每源一次标定 inverse remap）
  → RGB 视差风险 + 单调 hard owner / GraphCut
  → 仅安全白墙区域的窄带 MultiBand
  → valid mask 最大内接矩形、质量门禁和原子交付
```

正式全景像素只来自 RGB。深度仍严格用于会话契约和 Open3D/ORB-SLAM3 的真实轨迹验证；RGB renderer 不读取深度生成像素、不拟合参考平面，也不以深度产生前景。RGB 全景通过质量门禁后，`g305-panorama` 会额外生成仅供浏览的 `tsdf_mesh.glb` 和 `tsdf_mesh_viewer.html`：它使用同一严格 RGB-D 会话和真实轨迹，但不向条带、接缝、融合、裁剪或任何全景质量判定回传结果。正式流程不使用 UniStitch、LightGlue、MAGSAC、Torch、3×3 单应矩阵、二维累计或二维位姿插值。ORB-SLAM3 仅负责输出真实 RGB-D 相机轨迹；有未跟踪帧、位姿异常、RGB 尺度不稳定或条带接缝结构不完整时流程会失败，不会伪造位姿或回退到二维拼接。没有 `delivery.json` 就不是有效交付。

默认工况是相机连续单向水平侧移、场景基本静止、最近物体约 `0.5 m`、最高速度约 `1.5 m/s`。用户只需提供采集目录和输出目录，不需要调整曝光、步长、帧号、位姿、接缝或裁剪参数。

## 命令概览

| 命令 | 用途 |
|---|---|
| `g305-capture` | 采集连续流或照片模式驱动的低帧率同步 RGB-D 会话 |
| `g305-panorama` | 正式 RGB-D 序列全景入口 |
| `g305-central-strip-diagnostic` | 独立的参考平面中央条带诊断入口；绝不替代正式 RGB pushbroom 路径 |
| `unistitch-sequence` | 一个版本内保留的弃用别名；运行同一 RGB-D 流程，不含 UniStitch 回退 |
| `generate-panorama-demo` | 生成带标定、对齐深度和已知 SE(3) 轨迹的合成会话 |
| `unistitch-pair` | 独立历史双图诊断工具，不进入正式序列流程 |

### 让 `g305-panorama` 指向当前工作区（首次或切换工作区后执行一次）

本项目的正式入口必须来自当前工作区的源码。不要使用系统 Python 目录中旧的
`g305-panorama.exe`，否则它可能仍会走旧的 TSDF/深度渲染路径。以下命令把当前
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
```
`unistitch-sequence` 会打印弃用提示，但调用与 `g305-panorama` 完全相同的 RGB-D `main`。它不会加载 UniStitch、Torch、LightGlue 或 MAGSAC。

`g305-central-strip-diagnostic` 只用于评估“真实 RGB-D 轨迹驱动的参考平面中央条带”是否值得继续研究。它复用严格会话、Open3D 相邻边和 ORB-SLAM3 真实轨迹，但通过内部 renderer callback 与正式路径隔离：`g305-panorama` 不提供算法选项、不会导入该后端，也不会把它作为失败回退。

`configs/demo.yaml` 中的 `stitch.central_strip_diagnostic.enabled` 故意默认为 `false`；它不能通过 `g305-panorama` 打开。独立命令本身是唯一显式 opt-in，并向 renderer 传递一个内部启用的、固定且拒绝未知键的配置副本。

该诊断路线的参考平面也采用 fail-closed 门禁：它必须是唯一主导、跨扫描有足够标定图像面积支持的实测平面；竞争平面、面积不足或结构残差过大只会写 `failure.json`。较严格的平面质量阈值只会令 `strip_quality_pass=false`，仍可留下两个诊断文件供 A/B 检查，绝不变成正式交付。

## 正式处理流程

```text
Gemini 305 同步 RGB-D 会话
  ↓
会话、标定、对齐深度、深度比例和图像尺寸硬校验
  ↓
曝光、清晰度、纹理、主扫描段与相邻视觉运动分析
  ↓
自适应选择 RGB-D pose nodes
  ↓
相邻 RGB-D odometry；真实重叠的非相邻节点可增加弱 RGB-D 边
  ↓
仅由 RGB-D 边构建并优化 pose graph
  ↓
有限、连通、连续单向的 camera_to_world 4×4 SE(3)
  ↓
主扫描段的全部真实优化 pose nodes（最多 160 帧）
  ↓
每帧原始全分辨率 RGB 在主点窄带的一次标定 inverse remap 与姿态校平
  ↓
相邻 RGB 条带的白墙安全亮度增益、RGB 风险保护、单调 hard owner / GraphCut
  ↓
风险区直接 owner 复制；仅安全接缝做局部窄带 MultiBand
  ↓
基于 valid mask 的最大内接矩形、最终门禁和原子发布
```

缩略图视觉运动只用于主扫描段和 pose-node 布局，不产生正式几何变换。渲染源必须具有 pose graph 中真实优化出的位姿；程序不会按时间戳或二维运动伪造中间位姿。

## Windows 安装

### 前置条件

- Windows 10/11 x64；
- Python `3.10–3.12`，推荐 Conda Python 3.12；
- Git；
- Gemini 305 通过 USB 3 直接连接，采集建议使用 SSD/NVMe；
- 实机采集需要 `pyorbbecsdk2`；
- 正式序列依赖 Open3D 0.19 和可从 Windows 调用的 WSL ORB-SLAM3 RGB-D 示例；不依赖 Torch 或 CUDA。Gemini 305 的 `0.1 mm/unit` 深度必须自动写成 ORB-SLAM3 `DepthMapFactor=10000`，不得使用 TUM 示例的 `5000` 默认值。Open3D 0.19 的官方预编译包支持到 Python 3.12，详见 [Open3D 安装说明](https://www.open3d.org/docs/release/getting_started.html)。

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

## RGB-D odometry 与 Pose Graph

正式 backend 固定为 `open3d_rgbd`，参考 [Open3D RGB-D odometry](https://www.open3d.org/docs/latest/tutorial/pipelines/rgbd_odometry.html) 和 [Open3D multiway registration](https://www.open3d.org/docs/latest/tutorial/Advanced/multiway_registration.html)。

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
- `transforms.json` 中不会出现 3×3 homography，也不会有插值位姿。

正式流程拒绝必需相邻边失败、图不连通、非有限 SE(3)、逆向运动、明显上下/前后漂移、过大旋转或残差、可靠节点不足两帧，以及清晰渲染源覆盖不足 `95%`。

## 校准 RGB 流式狭缝扫描

优化后的位姿绝不会伪装为二维单应矩阵。正式 renderer 不取样深度：每个原始全分辨率 RGB 源仅在主点附近的窄条上执行一次标定 inverse remap，并用该帧的真实 `camera_to_world` 旋转校平。

```text
RGB (u, v)
  → 标定畸变模型的 inverse remap
  → 真实 SE(3) 的姿态校平
  → 主点附近 RGB 窄条
  → 连续扫描画布 x 坐标
```

画布 `x` 来自真实相机中心沿扫描方向的单调 SE(3) 位移；毫米到像素的唯一标量来自相邻 RGB 局部运动除以对应真实相机中心位移。这个标量仅决定狭缝布局：它不构造二维相机轨迹、不累计二维变换、不插值位姿，也不是深度/平面代理。没有足够稳定的相邻 RGB 测量时会失败。

主扫描段的全部真实优化 pose nodes 都是源（至少 2、最多 160），不再压缩成 32 张全画布投影。中间源的原始中央带不超过 RGB 宽度的 `20%`；首帧和末帧仅向扫描方向外侧扩展至各自校准 RGB 图像边缘，以保留扫描端点的场景。端点扩展仍是一次标定 inverse remap、独占 hard owner 和独立 valid mask；最终裁剪若丢弃某个完整有效的端点外扩列，交付会失败。镜头畸变校正产生的几何无效边缘列无法成为矩形全景的一部分，会在报告中单独审计。中间 hard-owner 区不能放入窄带时，会要求更密的真实采样或更低输出尺度。条带临时落盘，输出阶段只加载相邻 2 条（配置硬上限 5）；画布和流式 aggregate working set 均不超过 `200 MP`。

每个源输出的唯一像素证据是 RGB 和独立 `valid_mask`。黑色 RGB 像素可以有效；不会产生 `surface_depth_mm`、相机深度、点云、TSDF、参考平面或前景 mask。

物理限制：横向移动相机面对多深度场景时，纯 RGB 不可能同时生成对所有深度都严格正确的普通针孔透视图。窄条与 hard owner 能消除半透明双影，但近景仍可能轻微横向拉伸或压缩；若必须同时保证近景比例，需要改变采集方式、采用分层非刚性 RGB 变形，或重新引入可靠几何信息。

## RGB 风险、hard owner 与窄带 MultiBand

正式接缝 backend 固定为 `rgb_monotonic_hard_owner_graphcut`。在相邻条带真实共同有效区，程序从 Lab 残差、对称边缘距离和梯度结构不一致得到纯 RGB 风险；风险连通域经过填充和自适应保护，整块只能属于一个 RGB owner。GraphCut 在与 `2–8 px` 融合带解耦的 `32–64 px` 只读搜索走廊内寻找单调 hard owner 接缝，输出不会再按行重写。`owner_boundary ∩ risk_guard` 必须为空；没有安全通道时会保留可审计 hard cut 或失败，绝不使用 Feather、平均、补洞或透明重影掩盖问题。

光度补偿只从共同有效、低梯度、低饱和、近中性、未过曝/欠曝且不在风险保护带内的白墙候选估计。它在近似线性 RGB 中，对每个颜色通道取 trimmed Huber log-ratio，并一次性解全部帧的三通道全局 log-gain（带二阶平滑），而不是逐相邻对累加标量增益。每张 RGB 条带只施加一次线性三通道补偿后再编码输出；缺少可靠白墙支持或 gain 超出 `0.45–2.20` 均 fail-closed。

owner 审计后，每对相邻条带独立运行局部 `MultiBandBlender`。两个 blender mask 分别来自互补 hard owner 向安全白墙的膨胀，绝不共用同一 mask；风险、软管、标签和保护带直接复制唯一 owner。总融合带宽采用：

```text
clamp(floor(0.20 × 较窄 owner 宽度), 2, 8) px
```

层数最多 3。融合带必须完整、有正权重、位于共同有效的安全白墙，并且与 RGB 风险交集严格为 0；全图融合区不得超过有效像素的 `20%`。区外和风险保护带直接复制唯一 owner。

最终裁剪使用独立 valid mask 的 `largest_valid_rectangle()`，不是 `cv2.boundingRect()` 或 RGB 非黑检测，因此不会误删黑色软管等有效内容。正式门禁要求：严格 owner 分区、融合风险为 `0`、融合面积不超过 `20%`、曝光增益在 `0.45–2.20`、裁剪高度至少 `85%`、宽度至少 `95%`。

## 配置安全默认值

普通用户不需要修改 [`configs/demo.yaml`](configs/demo.yaml)。关键内部默认值如下；它们是 fail-closed 安全起点，仍需通过合成数据和现场静止、`0.5`、`1.0`、`1.5 m/s` 验收后才能确认具体硬件/场景的交付范围。

| 项目 | 默认值 |
|---|---:|
| pose backend | `open3d_rgbd` |
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
| 渲染源 | 主扫描段全部真实优化 pose nodes |
| 原始条带宽度 | 中间帧最多输入 RGB 宽度的 `20%`；首尾帧仅向外侧扩展至校准图像边缘 |
| 流式驻留 RGB 条带 | `2`（硬上限 `5`） |
| 画布 / aggregate working set 上限 | `200 / 200 MP` |
| seam backend | `rgb_monotonic_hard_owner_graphcut` |
| GraphCut 搜索走廊 | `64 px`（正式允许 `32–64 px`） |
| MultiBand 总带宽 / 层数 | `2–8 px` / 最多 `3` |
| 曝光补偿 | `safe_wall_global_linear_rgb` |

配置中的 `pose_backend`、`sequence_blend_mode=calibrated_rgb_pushbroom`、`calibrated_rgb_pushbroom.mode`、`scan_seam.backend`、标定/对齐要求和 pose graph 开关是正式结构约束，不能改为其它值发布交付。正式模式的曝光、RGB 尺度、odometry、pose、风险、GraphCut 和 MultiBand 阈值只能等于或收紧默认安全包络；试图放宽会直接失败。诊断模式可以绕过质量阈值，但 `200 MP` 画布/aggregate、160 pose nodes、5 条流式驻留上限和 RGB-only 像素来源等结构硬限仍不可放宽。手工 `render_frame_ids` 只允许诊断；正式命令会拒绝它。

## 产物、报告和原子交付

正式成功目录：

```text
outputs/greenhouse_sequence/
├─ panorama.jpg
├─ transforms.json
├─ render_transforms.json
├─ report.json
└─ delivery.json
```

- `transforms.json`：`rgbd-pose-graph/v1`，包含坐标约定、毫米单位、pose nodes 的 4×4 `camera_to_world`、RGB-D 边、信息矩阵、残差、优化和连通状态；
- `render_transforms.json`：`calibrated-rgb-pushbroom/v2`，包含 RGB-only 像素来源、真实 SE(3) 源、扫描布局、局部 RGB 像素/毫米标量、选源信息，以及不含 preview、flow、mask 或稠密 map 的残差对齐小参数、held-out 与拓扑审计摘要；
- `report.json`：`gemini305-calibrated-rgb-pushbroom/v2`，汇总 RGB-D 会话、输入质量、odometry、pose graph、pose quality、RGB 条带布局、残差对齐证据、风险、hard owner、亮度增益和 MultiBand 审计；
- `delivery.json`：`gemini305-panorama-delivery/v2`，最后发布；其 `alignment_backend` 与 `alignment_model` 标识最终采用的 RGB 残差对齐后端和模型，且只有 `quality_pass=true` 才代表有效交付。

每次任务先删除旧 `delivery.json`，正式文件先写隐藏 pending 文件，再用 `os.replace` 原子发布；`delivery.json` 最后写入。普通异常会清除正式文件并写 `failure.json`。强制终止可能来不及写失败报告，但没有有效 `delivery.json` 仍表示失败。

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

2026-07-13 使用 Open3D 0.19 和上述 `--diagnostic-force` 命令复测 `run_20260713_184519`：12 条必需相邻 RGB-D 边均收敛，fitness 为 `0.613–0.856`、RMSE 为 `17.4–26.2 mm`；旧深度 renderer 随后因高风险带横断完整相邻 pair corridor 而正确失败。该历史结果不能作为 RGB pushbroom 成功样本，也不是算法回退点；应通过补光、降低速度并重新采集解决。

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

合成会话包含 `calibration.json`、`color/`、`depth_aligned/`、带 `aligned_depth_path` 与 `depth_scale_mm_per_unit` 的 `frames.csv`，以及 manifest 中已知的毫米 `camera_to_world` 轨迹。它适合验证严格会话、单位、真实 SE(3) 交接、RGB-only 条带、风险 hard owner、黑色有效内容和原子交付语义，但不能代替 Gemini 305、Open3D 实机 odometry、现场照明或速度验收。

## 现场验收

新采集数据至少分别验证静止、`0.5 m/s`、`1.0 m/s` 和 `1.5 m/s`：

- `queue_drops == 0`、`write_errors == 0`，写盘队列无持续堆积；
- RGB-D 时间戳同步且没有回退；
- 彩色曝光 metadata 保持在正式上限内；
- 对齐深度有效率、尺寸和单位正确；
- pose graph 连通，位姿和残差满足报告阈值；
- 最近约 `0.5 m` 物体没有半透明重影；轻微狭缝横向拉伸/压缩应按纯 RGB 的物理限制单独评估；
- RGB 风险 hard owner、GraphCut/hard cut、白墙亮度带、上下抖动和最终四边通过人工复核；
- 输出目录存在且只存在最后发布的有效 `delivery.json`。

合成测试通过不等于实机验收完成。动态物体、镜面反射、完全无纹理、严重欠光、深度大面积空洞或源帧已拖影的场景可能被拒绝；这是 fail-closed 设计，不应通过放宽门限或回退平均来掩盖。

本机在 `2026-07-16` 已通过主环境的直接 `g305-panorama` 入口对
`data/run_20260714_132427_262` 复验：101 个真实源、输出 `2978×782`、裁剪高度
`97.75%`、融合区 `1.264%`、融合风险/owner 边界风险/受保护组件拆分均为 `0`。成功
目录为 `outputs/greenhouse_sequence_optimized/`，其 `delivery.json` 必须声明
`projection=calibrated_rgb_pushbroom`、`seam_backend=rgb_monotonic_hard_owner_graphcut`；
出现 `orthographic_side_scan`、TSDF mesh 或深度前景 mask 说明调用到了旧全局入口。

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

任务失败或只运行了诊断模式。检查 `failure.json`、标准错误或 `diagnostic_report.json`；不要把 `panorama.jpg` 是否存在、旧 `pairs/` 或 JPEG 黑色像素当作成功判断。

### GraphCut 或 MultiBand 报错

这是正式失败，没有自动 fallback。检查报告中的真实共同 RGB 覆盖、RGB 风险/保护带、owner 拓扑、有效掩码和 aggregate working set；若源数据不足，应补光、降低速度或重新采集。

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
