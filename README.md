# Gemini 305 RGB-D 移动侧扫全景

本项目在 Windows 上使用奥比中光 Gemini 305 采集同步、标定且对齐到彩色坐标系的 RGB-D 序列，并生成移动侧扫全景图。正式序列流程是：

```text
短基线 Open3D RGB-D Odometry（局部质量验证）
  → ORB-SLAM3 RGB-D（完整序列的真实全局相机轨迹）
  → 全帧 TSDF 几何融合
  → 标定平面真实位姿重投影（补足平整背景的深度空洞）
  → TSDF 前景覆盖
  → 裁剪、质量门禁和原子交付
```

正式流程不使用 UniStitch、LightGlue、MAGSAC、二维单应矩阵或二维位姿插值。ORB-SLAM3 仅负责输出真实 RGB-D 相机轨迹；有未跟踪帧、位姿异常或稠密融合覆盖不足时流程会失败，不会伪造位姿或回退到二维拼接。没有 `delivery.json` 就不是有效交付。

默认工况是相机连续单向水平侧移、场景基本静止、最近物体约 `0.5 m`、最高速度约 `1.5 m/s`。用户只需提供采集目录和输出目录，不需要调整曝光、步长、帧号、位姿、接缝或裁剪参数。

## 命令概览

| 命令 | 用途 |
|---|---|
| `g305-capture` | 采集连续流或照片模式驱动的低帧率同步 RGB-D 会话 |
| `g305-panorama` | 正式 RGB-D 序列全景入口 |
| `unistitch-sequence` | 一个版本内保留的弃用别名；运行同一 RGB-D 流程，不含 UniStitch 回退 |
| `generate-panorama-demo` | 生成带标定、对齐深度和已知 SE(3) 轨迹的合成会话 |
| `unistitch-pair` | 独立历史双图诊断工具，不进入正式序列流程 |

激活环境

```
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
```

`unistitch-sequence` 会打印弃用提示，但调用与 `g305-panorama` 完全相同的 RGB-D `main`。它不会加载 UniStitch、Torch、LightGlue 或 MAGSAC。

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
按真实相机中心、投影足迹、清晰度和覆盖选择渲染源
  ↓
原始全分辨率 RGB-D 各重投影一次到统一正射世界条带
  ↓
相邻 pair corridor + 深度高风险区硬 owner + GraphCut
  ↓
严格 owner 拓扑、真实重叠、边界风险与跨向覆盖验证
  ↓
只在 owner 边界窄带执行 OpenCV MultiBand
  ↓
最大有效矩形裁剪、最终门禁和原子发布
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

### Conda（首选）

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\bootstrap_conda.ps1
conda activate .\.conda
```

脚本使用 [`environment.yml`](environment.yml) 创建项目内 `.conda`，并安装基础项目、Open3D、采集依赖和测试依赖。默认不会安装 Torch/Kornia/torchvision，不会克隆 UniStitch/LightGlue，也不会下载模型。

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
powershell -ExecutionPolicy Bypass -File .\scripts\register_orbbec_metadata.ps1
```

脚本会优先使用项目内 `.conda`，必要时请求管理员权限。完成后重新插拔相机。短采集后应检查 `manifest.json` 的 `metadata_support`，以及 `frames.csv` 中曝光、增益、帧号、sensor timestamp 和 RGB-D 时间差是否持续有效。

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
- 彩色自动曝光开启，上限 `800 µs`；固件不支持 AE 上限时退回固定 `800 µs`；
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

## 深度重投影侧扫条带

优化位姿不会伪装为二维单应矩阵。每个选中的原始全分辨率 RGB-D 源只处理一次：

```text
(u, v, depth_device_unit)
  → depth_mm
  → 彩色相机三维点
  → camera_to_world
  → 统一世界正射条带坐标
  → 每源 point-splat z-buffer
```

扫描方向规范为画布 `x`，相机上方向规范为画布 `y`，世界法向记录为 `z`。像素密度由投影足迹自动估计，用户不设置比例。每源输出：

- `warped_rgb`；
- 独立 `valid_mask`；
- 同一世界法向定义的 `surface_depth_mm`；
- `surface_depth_valid_mask`；
- 与 z-buffer 最终样本对应的源彩色相机 `camera_depth_mm` 及独立 valid mask；
- 投影中心、有效包围盒、投影高度和采样统计。

投影使用 point splat 而不是跨相邻深度样本连三角形，因此不会把前景/背景断层拉成连续表面。同一源落到同一画布像素时用世界法向 z-buffer 保留可见表面；深度空洞不会补造几何，必须由另一张真实源覆盖，否则后续门禁失败。

画布超过 `200 MP`，或所有投影源的 aggregate working set 超过 `200 MP`，会在分配大数组前失败。正式最多选择 32 个渲染源，仍必须维持至少 `95%` 可靠扫描覆盖和 `34%` 相邻投影足迹重叠；预算不足时拒绝，不会发布部分扫描。

## 深度约束 GraphCut

正式接缝 backend 固定为 `graphcut_depth_constrained`。OpenCV [`GraphCutSeamFinder`](https://docs.opencv.org/4.x/d2/d7c/classcv_1_1detail_1_1GraphCutSeamFinder-members.html) 的公开 API 只接受颜色、梯度和 mask，不能接收任意深度代价，所以本项目不会宣称原生 GraphCut “天然深度感知”。

深度约束在 GraphCut 前转成硬 mask：

1. 按投影中心排序，只给相邻帧建立互不允许非相邻帧共同竞争的 `pair corridor`。
2. 在真实共同有效区计算 Lab 色差、梯度差、与世界原点无关的固定毫米世界表面深度差，以及源相机深度小于 `1 m` 的近景/遮挡风险和 MultiBand 保护带风险。
3. 对高风险连通域，根据有效覆盖、投影采样质量、离无效边缘距离、清晰度和确定性帧序选择唯一可靠 owner。
4. 从另一帧的 GraphCut mask 中移除该区域；绝不从双方同时删除。
5. 若没有唯一 owner、风险带横断走廊或已无连续安全通道，直接失败。
6. 对剩余走廊调用 `cv2.detail_GraphCutSeamFinder("COST_COLOR_GRAD")`。

GraphCut 输出必须转换为严格全画布 owner mask。有效区域每个像素恰好一个 owner，单源区只能归该源，非相邻 owner 不得接触，每对相邻源必须存在由真实共同有效区支撑的边界。空洞、多 owner、无效区 owner、缺失边界或 GraphCut 异常都会失败；禁止“按离中心最近帧补洞”，也没有 DP、Feather 或平均回退。

## owner 边界窄带 MultiBand

只有完整 owner 拓扑验证通过后，才运行 OpenCV `MultiBandBlender`；实现参考 [OpenCV stitching detailed sample](https://docs.opencv.org/4.x/d9/dd8/samples_2cpp_2stitching_detailed_8cpp-example.html)。默认 `multiband_levels=5`。

程序从最终 owner 边界生成保守的窄 `blend_zone`：区外直接复制唯一 owner 的全局增益校色 RGB，区内才采用 MultiBand 输出。每一对相邻 owner 使用独立的局部 `MultiBandBlender`，不会把所有源送入同一个全局金字塔；相邻保护带重叠时按离哪条 owner 边界更近分成互斥区域，避免 `i+2` 源的低频颜色串入 `i/i+1` 边界。每个局部 blender 的 output mask 必须完整覆盖它的真实 pair support，不能出现零权重 wedge、黑洞或写入无效区。融合带触及高风险区的比例仍会复核；失败时不会回退到自定义归一化 MultiBand 或整片平均。

最终无条件执行的门禁包括：

- owner 空洞、重叠和无效区 owner；
- 非相邻 owner 接触；
- 边界缺失或没有真实重叠支撑；
- 精确接缝风险和融合保护带风险均不高于 `0.10`；
- 最小跨向覆盖不低于 `0.80`；
- 安全接缝 Lab 残差 P95 不高于 `48`；
- 曝光增益必须保持在 `0.45–2.20`；
- 裁剪保留高度不低于源图中值高度的 `90%`，宽度不低于扫描画布的 `95%`；
- 独立 valid mask 定义的最终区域不能有黑边或缺口。

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
| 渲染源上限 | `32` |
| 渲染覆盖 / 相邻足迹重叠下限 | `0.95 / 0.34` |
| 画布 / aggregate working set 上限 | `200 / 200 MP` |
| seam backend | `graphcut_depth_constrained` |
| MultiBand 层数 | `5` |
| 曝光补偿 | `global_gain` |

配置中的 `pose_backend`、`rgbd_projection.mode`、`scan_seam.backend`、标定/对齐要求和 pose graph 开关是正式结构约束，不能改为其它值发布交付。正式模式的曝光、覆盖、odometry、pose、GraphCut 和 MultiBand 阈值只能等于或收紧默认安全包络；试图放宽会直接失败。诊断模式可以绕过质量阈值，但 `200 MP` 画布/aggregate、160 pose nodes 和 32 render sources 等资源硬限仍不可放宽。手工 `render_frame_ids` 只允许诊断；正式命令会拒绝它。

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
- `render_transforms.json`：`rgbd-side-scan-projection/v1`，包含正射模式、扫描/上/法向轴、像素密度、世界范围、渲染源选择和每帧投影足迹；
- `report.json`：`gemini305-rgbd-side-scan/v3`，汇总 RGB-D 会话、输入质量、布局、odometry、pose graph、pose quality、投影、GraphCut、owner 和 MultiBand 审计；
- `delivery.json`：`gemini305-panorama-delivery/v2`，最后发布，且只有 `quality_pass=true` 才代表有效交付。

每次任务先删除旧 `delivery.json`，正式文件先写隐藏 pending 文件，再用 `os.replace` 原子发布；`delivery.json` 最后写入。普通异常会清除正式文件并写 `failure.json`。强制终止可能来不及写失败报告，但没有有效 `delivery.json` 仍表示失败。

旧诊断文件会在新的正式或失败任务开始时清除。历史 `pairs/` 不是交付目录，不能用于判断本次任务是否成功。

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
- 有效重投影；
- 严格 owner 拓扑；
- 画布和 aggregate working-set 限制；
- 原子交付语义。

诊断成功只写 `diagnostic_panorama.jpg` 和 `diagnostic_report.json`，绝不写 `panorama.jpg`、正式 JSON 或 `delivery.json`。即使使用了 `--diagnostic-force`，结构性失败仍会明确报错，而不是强行出图。

`run_20260713_184519` 的 `color_exposure=301`，约 `30.1 ms`，远高于 `1200 µs` 正式移动安全门限，因此只能按上面的诊断命令测试。旧温室会话 `run_20260711_213054` 同样只适合作为输入门禁应拒绝的回归样本。源帧已经丢失的纹理不能靠融合恢复。

2026-07-13 使用 Open3D 0.19 和上述 `--diagnostic-force` 命令复测 `run_20260713_184519`：12 条必需相邻 RGB-D 边均收敛，fitness 为 `0.613–0.856`、RMSE 为 `17.4–26.2 mm`；随后深度硬约束发现高风险带横断完整相邻 pair corridor，任务按结构门禁失败。输出目录只留下 `failure.json`，没有诊断图、正式文件或 `delivery.json`。这不是算法回退点，应通过补光、降低速度并重新采集解决。

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

合成会话包含 `calibration.json`、`color/`、`depth_aligned/`、带 `aligned_depth_path` 与 `depth_scale_mm_per_unit` 的 `frames.csv`，以及 manifest 中已知的毫米 `camera_to_world` 轨迹。它适合验证单位、SE(3) 组合、z-buffer、深度断层、黑色有效内容、owner 和原子交付语义，但不能代替 Gemini 305、Open3D 实机 odometry、现场照明或速度验收。

## 现场验收

新采集数据至少分别验证静止、`0.5 m/s`、`1.0 m/s` 和 `1.5 m/s`：

- `queue_drops == 0`、`write_errors == 0`，写盘队列无持续堆积；
- RGB-D 时间戳同步且没有回退；
- 彩色曝光 metadata 保持在正式上限内；
- 对齐深度有效率、尺寸和单位正确；
- pose graph 连通，位姿和残差满足报告阈值；
- 最近约 `0.5 m` 物体没有明显重影或前景拉伸；
- GraphCut 边界、亮度带、上下抖动和最终四边通过人工复核；
- 输出目录存在且只存在最后发布的有效 `delivery.json`。

合成测试通过不等于实机验收完成。动态物体、镜面反射、完全无纹理、严重欠光、深度大面积空洞或源帧已拖影的场景可能被拒绝；这是 fail-closed 设计，不应通过放宽门限或回退平均来掩盖。

## 常见问题

### `Open3D is required ... but could not be imported`

确认正在使用项目环境，并重新安装基础依赖：

```powershell
.\.conda\python.exe -m pip install -e ".[capture,test]"
.\.conda\python.exe -c "import open3d; print(open3d.__version__)"
```

### 找不到 `g305-panorama`

重新执行 editable 安装：

```powershell
.\.conda\python.exe -m pip install -e ".[capture,test]"
```

### 正式目录没有 `delivery.json`

任务失败或只运行了诊断模式。检查 `failure.json`、标准错误或 `diagnostic_report.json`；不要把 `panorama.jpg` 是否存在、旧 `pairs/` 或 JPEG 黑色像素当作成功判断。

### GraphCut 或 MultiBand 报错

这是正式失败，没有自动 fallback。检查报告中的真实共同覆盖、深度高风险带、owner 拓扑、输出 mask 和 aggregate working set；若源数据不足，应补光、降低速度或重新采集。

### Windows App Control 阻止 `c10.dll`

`c10.dll` 属于可选历史 Torch/UniStitch 诊断依赖，不应进入正式 `g305-panorama`。若只运行正式 RGB-D 流程，请使用未安装 `unistitch-diagnostic` 的基础环境；若确需历史双图工具，需要由系统策略放行组织认可的签名 Python/PyTorch 环境。

## 测试

```powershell
.\.conda\python.exe -m pytest -q
ruff check src tests
.\.conda\python.exe -m compileall -q src tests
git diff --check
```

修改相机、Open3D、GraphCut、MultiBand、性能或交付语义时，必须明确区分：纯单元/合成验证、历史失败数据回归、真实 Open3D 运行和 Gemini 305 现场验收。

## 第三方项目与许可证

- [Open3D](https://github.com/isl-org/Open3D) 提供正式 RGB-D odometry 与 pose graph 能力；
- [OpenCV](https://opencv.org/) 提供图像处理、GraphCut seam finder 和 MultiBand blender；
- [OrbbecSDK v2 Python wrapper](https://github.com/orbbec/pyorbbecsdk2) 提供 Gemini 305 采集；
- UniStitch 与 LightGlue 仅为可选历史双图诊断依赖。

UniStitch、LightGlue 与 Orbbec wrapper 的固定来源和许可证说明见 [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)；Open3D 与 OpenCV 许可证以各自上游发布为准。
