# Gemini 305 + UniStitch 移动侧扫 Demo

本项目面向 Windows 上的奥比中光 Gemini 305 侧向移动采集：相机安装在水平小车上，光轴朝向小车前进方向的侧面，连续记录同步 RGB、深度、标定参数和时间戳，并使用 UniStitch 完成双图预览和序列 RGB 长图构建。

当前目标是先完成可复现的离线 Demo，而不是承诺无限长度、零漂移或实时三维建图。采集默认保留对齐到 RGB 的深度，原始深度可按需额外保存，便于后续增加 RGB-D 位姿估计、点云或 TSDF；UniStitch 本身只处理 RGB 与二维关键点，不读取深度，也不输出三维位姿。

默认路径面向“最近物体约 0.5 m、相机连续单向侧移、场景基本静止”的交付条件，用户无需修改曝光、步长、帧号、保护框、接缝或裁剪参数。程序只有在输入清晰度、重叠、运动方向、深度接缝和无黑边裁剪全部通过时才发布完整图；严重欠光、动态遮挡、镜面、无纹理或所有源帧已经拖影时，会自动拒绝劣图并写明原因。二维算法无法从这类缺失信息中恢复真实细节。

## 当前实现状态

| 模块 | 命令 | 状态 |
|---|---|---|
| Gemini 305 RGB-D 采集 | `g305-capture` | 已实现 |
| 无相机合成侧扫数据 | `generate-panorama-demo` | 已实现 |
| UniStitch 双图拼接 | `unistitch-pair` | 已实现，使用完整 FFD/TPS 生成双图预览 |
| UniStitch 序列拼接 | `unistitch-sequence` | 已实现，使用 UniStitch global 分支布局并一次渲染原图 |
| 彩色点云、TSDF、网格 | 暂无 | 不在当前 Demo 中 |

## 处理流程

```text
Gemini 305
  └─ g305-capture
       ├─ RGB JPEG
       ├─ D2C 对齐深度 PNG
       ├─ 可选原始深度 PNG
       ├─ 时间戳和 metadata
       └─ 相机内参、畸变和 RGB-D 外参

RGB 会话或合成会话
  ├─ unistitch-pair
  │    └─ 完整双图 homography + FFD/TPS 预览和质量报告
  └─ unistitch-sequence
       ├─ UniStitch global 门限验证与可用 MAGSAC 同支持集择优/回退
       └─ 累积布局后从原始帧一次性渲染 RGB 长图
```

## 50 cm、1.5 m/s 工况的建议参数

默认配置位于 [`configs/demo.yaml`](configs/demo.yaml)，针对 50 cm 拍摄距离和最高 1.5 m/s 速度给出了一组保守起点：

| 参数 | 默认/建议值 | 说明 |
|---|---:|---|
| RGB 与深度 | `1280×800 @ 30 fps` | 必须以实机能同时开启的精确流配置为准 |
| D2C | `software` | 当前采集器只支持软件对齐到彩色坐标系 |
| 帧同步 | 开启 | SDK 不支持时会记录警告，而不是伪造同步成功 |
| 彩色曝光 | 自动曝光，上限 `800 µs` | 自动适应亮度，同时针对最近约 0.5 m 的移动侧扫限制拖影；过暗时应补光 |
| 防闪烁 | 开启 | 大棚使用交流补光时尤其重要 |
| 采集写队列 | 64 帧 | 磁盘持续写入不足时会统计丢弃数 |
| 序列选帧 | 自动 | 根据相邻帧实测位移、覆盖和清晰度自适应，不使用固定步长 |
| UniStitch 推理宽度 | 640 px | 控制显存；先验证稳定性，再尝试原分辨率 |
| 最大关键点 | 2048 | 与公开 UniStitch 模型输入约定一致 |
| 布局安全上限 | 160 帧 | 超过时要求拆分路线，避免无界内存和累计漂移 |

按约 88° 的 RGB-D 公共水平视场估算，50 cm 处横向覆盖约 0.97 m。小车以 1.5 m/s、30 fps 行驶时，相邻原始帧位移约 50 mm，重叠约 95%；每 3 帧选一张时位移约 150 mm、重叠约 84%，每 4 帧约 200 mm、重叠约 79%。

默认配置使用带 `800 µs` 上限的自动曝光，用户不需要再设置曝光参数；按 1.5 m/s 运行时，小车在一次曝光内最多移动约 1.2 mm，对应约 1.6 个原始图像像素。相机会在上限内自动适应亮度，程序再自动选帧并执行质量门禁。若固件不支持自动曝光上限，采集器会自动退回固定 `800 µs`，而不会放任长曝光；现场仍过暗时应增加连续补光。需要固定曝光时可传 `--exposure-us`（或同义的 `--manual-exposure-us`），这会关闭自动曝光并验证设备回读值；不传则继续使用自动模式。

Gemini 305 的彩色曝光控制原始单位是 100 µs；采集器会把手动微秒值四舍五入到最近的 100 µs 后写入，并将请求模式、实际模式和实际值记录在 `manifest.json` 的 `color_exposure_control`。`frames.csv` 的 `color_exposure` 保留设备原始值，例如 `301` 表示约 30.1 ms，而不是 301 µs。手动值超过 `1200 µs` 可用于静态诊断采集，但移动序列的正式拼接会按质量门禁拒绝交付。

现有温室会话 `run_20260711_213054` 的峰值正是 `301`（约 30.1 ms），超过默认 `1200 µs` 移动输入门禁，因此只能作为失败回归样本，不能交付。曝光期间已经丢失的纹理无法靠后处理恢复，必须使用新曝光上限重新采集。

50 cm 处公共垂直覆盖约 0.64 m。若铁架需要扫描的高度更大，应改变安装距离或采用不同高度的多趟扫描。相机应刚性固定，图像水平轴与小车运动方向平行，光轴尽量垂直于铁架。

## 目录结构

```text
Panoramic_Camera/
├─ configs/demo.yaml                  默认采集和拼接配置
├─ environment.yml                    Conda 环境定义（Python 3.12）
├─ scripts/
│  ├─ bootstrap_conda.ps1             首选：创建项目内 Conda 环境
│  ├─ bootstrap_windows.ps1           备选：创建 Python venv
│  ├─ register_orbbec_metadata.ps1    注册 Windows UVC metadata
│  └─ download_unistitch_weights.py   下载并校验官方 UniStitch 权重
├─ configs/capture_640x480.yaml        低带宽 YUYV 采集回退配置
├─ src/panorama_demo/
│  ├─ capture_orbbec.py               RGB-D 采集
│  ├─ synthetic.py                    确定性合成侧扫序列
│  ├─ session.py                      会话和帧发现
│  ├─ unistitch_adapter.py             LightGlue + UniStitch 内存适配器
│  ├─ stitch_pair.py                  完整双图 FFD/TPS 命令
│  ├─ stitch_sequence.py              序列布局和报告命令
│  ├─ stitch_common.py                拼接公共配置和图像 I/O
│  └─ render.py                       原始帧单源分区接缝或羽化渲染
├─ third_party/UniStitch/             固定版本的上游研究代码
├─ third_party/LightGlue/             固定版本的关键点匹配代码
└─ THIRD_PARTY_NOTICES.md             第三方来源和许可证说明
```

## Windows 安装

### 1. 前置条件

- Windows 10/11 x64；
- 首选方案需要 Miniconda 或 Anaconda，且 `conda` 已加入当前 PowerShell；未安装时可使用后文 venv 备选方案；
- Git；
- Gemini 305 连接到主机 USB 3 端口，避免无源 Hub；
- 采集建议使用 SSD/NVMe；
- 运行 UniStitch 需要 NVIDIA CUDA GPU。采集和合成数据生成不依赖 GPU。

首选 Conda 方案由 `environment.yml` 创建项目内的 Python 3.12 环境。备选 venv 方案支持 `pyproject.toml` 声明的 Python `3.10–3.13`。

`pyproject.toml` 要求 PyTorch 2.1 及以上。两个初始化脚本会先从 PyTorch 官方 CUDA 13.0 wheel 源安装锁定的 `torch 2.13.0+cu130` 与 `torchvision 0.28.0+cu130`，再安装其余依赖；当前 RTX 5060 Laptop 与驱动 610.62 已实际验证。若目标电脑驱动不支持 CUDA 13.0，应先升级 NVIDIA 驱动，或按 [PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/) 改成该电脑支持的 CUDA wheel，并确认 `torch.cuda.is_available()` 为 `True`。

### 2. Conda 初始化（首选）

在项目根目录打开 PowerShell：

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\bootstrap_conda.ps1
conda activate .\.conda
```

脚本会：

1. 根据 `environment.yml` 在项目根目录创建 `.conda`，使用 Python 3.12；
2. 以 editable 模式安装项目、采集依赖和测试依赖；
3. 在第三方目录不存在时检出固定版本的 UniStitch 与 LightGlue；
4. 下载并校验官方 UniStitch 模型。

可选参数：

```powershell
# 暂不下载约 321 MiB 的模型
.\scripts\bootstrap_conda.ps1 -SkipModel

# 删除已有 .conda 并完整重建；会移除该环境内已安装的包
.\scripts\bootstrap_conda.ps1 -Recreate
```

如果当前 PowerShell 不能执行 `conda activate`，先运行一次 `conda init powershell` 并重新打开 PowerShell；也可以不激活环境，改用 `conda run --prefix .\.conda ...`。

检查环境：

```powershell
python -c "import torch; print('torch=', torch.__version__, 'cuda=', torch.cuda.is_available())"
g305-capture --help
generate-panorama-demo --help
unistitch-pair --help
unistitch-sequence --help
```

不激活 Conda 环境时，等价检查命令例如：

```powershell
conda run --prefix .\.conda python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
conda run --prefix .\.conda unistitch-sequence --help
```

如果只进行采集、合成数据或 CPU 单元测试，`cuda=False` 不影响这些功能。`unistitch-pair` 和 `unistitch-sequence` 在真正开始对齐时会明确拒绝 CPU 设备。

### 3. venv 初始化（备选）

未安装 Conda 时，可以使用系统 Python 创建 `.venv`：

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\bootstrap_windows.ps1
.\.venv\Scripts\Activate.ps1
```

可选参数：

```powershell
# 暂不下载模型
.\scripts\bootstrap_windows.ps1 -SkipModel

# 仅当系统 Python 已有可用的 CUDA PyTorch，且 .venv 尚不存在时使用
.\scripts\bootstrap_windows.ps1 -ReuseSystemPackages
```

如果 `.venv` 已存在，`-ReuseSystemPackages` 不会改变它；需要切换环境策略时应先自行移走旧环境。

### 4. 手动下载模型

初始化时使用了 `-SkipModel`，或者需要重新校验模型时执行：

```powershell
python .\scripts\download_unistitch_weights.py

# 覆盖校验失败或损坏的已有文件
python .\scripts\download_unistitch_weights.py --force
```

模型保存到 `models/unistitch/epoch_best_model.pth`。下载器验证文件大小 `336,715,309` 字节和 SHA-256：

```text
c7c4184c3ec63e15ed483f7066afdd4ed2fcd12f1178ae27183c9838f9083c19
```

模型不会随本项目再分发。Hugging Face 模型卡目前把权重标记为 `license: other`，且没有提供完整许可文本；商业使用或再次分发前应向作者确认。详见 [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)。

## 注册 Windows metadata

先完成环境初始化并连接相机。下面的脚本会优先使用项目内 `.conda`，不存在时再使用 `.venv`，也可通过 `-Python` 显式指定解释器：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\register_orbbec_metadata.ps1

# 可选：明确指定 Python
powershell -ExecutionPolicy Bypass -File .\scripts\register_orbbec_metadata.ps1 `
  -Python D:\Panoramic_Camera\.conda\python.exe
```

脚本最终运行 `pyorbbecsdk2` 自带的 `shared/setup_env.py`，并可能请求管理员/UAC 权限。每台新的物理相机通常需要执行一次。完成后重新插拔相机；若系统提示需要重启，则先重启再采集。

注册是否生效不能只看脚本退出码。完成一次短采集后检查：

- `manifest.json` 中的 `metadata_support`；
- `frames.csv` 中的曝光、增益、帧号和 sensor timestamp 字段是否持续存在；
- device timestamp 是否单调递增。

未注册 metadata 时，RGB 和深度图仍可能保存成功，但时间同步分析和掉帧诊断不可信。

## 无相机的合成数据测试

下面的命令生成一段确定性的侧扫 RGB 序列，包含重复铁架、随机纹理、文字标记和轻微亮度变化：

```powershell
generate-panorama-demo `
  --output .\data\synthetic\demo `
  --frames 10 `
  --width 640 `
  --height 400 `
  --step 120
```

640 px 画面每帧移动 120 px，对应约 81% 重叠。输出结构为：

```text
data/synthetic/demo/
├─ manifest.json
├─ frames.csv
└─ color/
   ├─ 00000000.jpg
   ├─ 00000001.jpg
   └─ ...
```

这条路径不需要相机或 Orbbec SDK，适合验证安装、会话发现、图像读取和未来的序列拼接入口。它不生成深度，也不能验证真实相机同步、曝光或深度质量。

## 采集 Gemini 305 RGB-D

### 安装检查

采集前确认：

- 相机直接连接 USB 3；
- 相机与支架无松动；
- 拍摄距离约 50 cm；
- 现场补光不会频闪；
- 关闭同时占用相机的 OrbbecViewer 或其他程序；
- 已执行 metadata 注册。

### 推荐命令

```powershell
g305-capture `
  --output .\data\captures
```

程序会在输出目录下新建 `run_YYYYMMDD_HHMMSS` 会话。预览窗口中按 `Q` 或 `Esc` 停止，也可以使用 `Ctrl+C`。无预览模式同样监听控制台的 `Q`/`Esc`。

先录制 300 帧做约 10 秒冒烟测试：

```powershell
g305-capture `
  --config .\configs\demo.yaml `
  --output .\data\captures `
  --max-frames 300
```

常用覆盖参数：

```powershell
# 固定录制 20 秒，不显示预览；默认只保存对齐深度
g305-capture `
  --config .\configs\demo.yaml `
  --output .\data\captures `
  --duration 20 `
  --no-preview

# 短期标定/诊断时额外保存原始深度
g305-capture `
  --config .\configs\demo.yaml `
  --output .\data\captures `
  --duration 5 `
  --raw-depth

# 本次采集固定为 1000 µs；不修改默认自动曝光配置
g305-capture `
  --output .\data\captures `
  --exposure-us 1000

# 如果自定义配置采用手动曝光，可临时强制切回自动曝光
g305-capture `
  --config .\configs\custom.yaml `
  --output .\data\captures `
  --auto-exposure
```

默认配置已经采用适合移动扫描的“自动曝光 + `800 µs` 上限”，无需手动传入参数。`--auto-exposure` 与 `--exposure-us` 互斥；手动曝光只固定曝光时长，增益仍可用 `--gain` 单独指定。还可通过命令行覆盖 `--warmup-frames`、`--queue-size` 和 `--white-balance`。

若要在自定义配置中长期使用手动曝光，可设置：

```yaml
capture:
  color_auto_exposure: false
  color_exposure_us: 1000
```

恢复自动曝光时设为 `color_auto_exposure: true`、`color_exposure_us: null`。

采集器要求彩色和深度都存在与配置完全一致的分辨率和帧率，深度格式为 Y16。如果实机不支持，程序会列出枚举到的 profile；请选择列表中可同时运行的组合。当前仅支持 `align: software`，其它值会明确报错。

当前 Gemini 305（固件 1.0.70、USB 3.2）已实测可同时运行 RGB `1280×800@30` 与 Y16 `1280×800@30`。如果现场电脑或 USB 链路无法稳定运行该组合，可先使用低带宽回退配置验证链路：

```powershell
g305-capture `
  --config .\configs\capture_640x480.yaml `
  --output .\data\captures `
  --max-frames 30 `
  --no-preview
```

该配置使用 YUYV `640×480@30` 与 Y16 `640×480@30`。它适合诊断，不应在没有记录的情况下替代正式分辨率。采集器还会用 `warmup_timeout_seconds` 和 `frame_timeout_seconds` 拒绝“不出帧但无限等待”的会话。

### 采集输出

```text
data/captures/run_YYYYMMDD_HHMMSS/
├─ manifest.json
├─ calibration.json
├─ frames.csv
├─ color/
│  └─ 00000000.jpg
├─ depth_aligned/
│  └─ 00000000.png
└─ depth_raw/           仅使用 --raw-depth 时生成
   └─ 00000000.png
```

- `manifest.json`：设备、Python wrapper、SDK 版本、请求和实际 profile、属性设置结果、metadata 支持情况、接收/写入/丢弃统计及是否正常结束；
- `calibration.json`：彩色和深度内参、畸变、深度到彩色外参；
- `frames.csv`：帧 ID、device/system/host 时间戳、原始 sensor timestamp、曝光、增益、帧号、同步差、深度比例和相对文件路径；
- `color/*.jpg`：JPEG 质量 95 的 BGR 彩色帧；
- `depth_aligned/*.png`：软件对齐到彩色坐标系的 16 位深度；
- `depth_raw/*.png`：可选的原始 16 位深度，仅使用 `--raw-depth` 时生成。

深度 PNG 保存的是相机深度单位，不应直接假定每单位就是 1 mm；应使用 `frames.csv` 中的 `depth_scale_mm_per_unit` 换算。

短采集验收建议：

- `timestamp_regressions == 0`；
- `queue_drops == 0`；
- `write_errors == 0`；
- RGB 与深度时间差稳定；
- 50 cm 目标区域深度连续；
- 最大速度下铁杆边缘运动模糊不超过约 2 px。

## UniStitch 双图拼接

双图命令的两个位置参数依次是“参考/前一张图”和“来源/后一张图”。两张图必须具有相同尺寸：

```powershell
unistitch-pair `
  .\data\synthetic\demo\color\00000000.jpg `
  .\data\synthetic\demo\color\00000001.jpg `
  --config .\configs\demo.yaml `
  --output .\outputs\pair
```

完整参数：

```text
unistitch-pair [--output DIR] [--config YAML] [--model PTH]
               [--device DEVICE] [--inference-width PIXELS]
               [--strict-unistitch]
               REFERENCE SOURCE
```

- `--output` 默认是 `outputs/pair`；
- `--config` 未指定时自动读取 `configs/demo.yaml`；
- `--model`、`--device`、`--inference-width` 可覆盖配置文件；
- `--inference-width` 最小为 128，默认 640，并按原图宽高比确定推理高度；
- `--strict-unistitch` 禁用 MAGSAC 的择优比较和回退：只接受通过重投影门限的 UniStitch global 分支。

适配器直接在内存中运行 SuperPoint + LightGlue，不使用上游面向训练数据集的 LMDB 流程。它把最多 2048 个已匹配的关键点和描述子送入官方模型，并执行完整的全局 homography 与局部 FFD/TPS。双图输出为：

```text
outputs/pair/
├─ pair_unistitch.jpg   完整 FFD/TPS 双向中间平面融合预览
└─ pair_report.json     变换、匹配数、误差、画布、耗时和布局方法
```

`pair_unistitch.jpg` 使用推理分辨率生成，不是原始全分辨率计量图。`pair_report.json` 同时保存 UniStitch 原始 global homography 和最终采用的 `homography_source_to_reference`；`layout_method` 可能为 `unistitch_global`、`magsac_preferred` 或 `magsac_fallback`。MAGSAC 候选只有在内点数不少于 `min_matches` 且内点率至少为 50% 时才可用；择优时会在同一组 MAGSAC 内点上分别计算 MAGSAC 与 UniStitch 的中值重投影误差，避免拿不同支持集的误差直接比较。`magsac_preferred` 表示可用 MAGSAC 在该统一支持集上误差更低；`magsac_fallback` 表示 UniStitch 未通过自身的全匹配门限后采用可用 MAGSAC。无论最终布局方法是哪一种，双图预览都仍由完整 UniStitch FFD/TPS 产生。

匹配少于 `min_matches`、FFD 画布超过安全上限、CUDA 不可用或两图尺寸不同都会返回非零退出码，不会把未对齐图片静默标记为成功。

## UniStitch 序列拼接

输入可以是完整采集会话、`frames.csv`，或者直接包含图片的 `color` 目录。合成数据本身每帧已移动 120 px，应使用 `--stride 1`：

```powershell
unistitch-sequence `
  .\data\synthetic\demo `
  --config .\configs\demo.yaml `
  --output .\outputs\synthetic_sequence `
  --stride 1 `
  --max-frames 10
```

真实相机序列默认不需要设置步长、帧号、接缝位置或裁剪范围；程序会自动分割主扫描段、按实测像素位移布局并选择清晰渲染帧：

```powershell
unistitch-sequence `
  .\data\captures\run_YYYYMMDD_HHMMSS `
  --output .\outputs\greenhouse_sequence
```

完整参数：

```text
unistitch-sequence [--output DIR] [--config YAML] [--model PTH]
                    [--device DEVICE] [--inference-width PIXELS]
                    [--stride N] [--max-frames N]
                    [--max-canvas-megapixels MP]
                    [--blend-mode {feather,scan_seam}]
                    [--render-frame-ids ID,ID,...]
                    [--translation-anchor-y FRACTION]
                    [--scan-seam-margin PIXELS]
                    [--scan-multiband-levels N]
                    [--scan-exposure-mode {none,center_gain,global_gain}]
                    [--scan-seam-mask-sigma PIXELS]
                    [--scan-protect-region FRAME_ID:X0:Y0:X1:Y1]
                    [--motion-model {translation,similarity,homography}]
                    [--no-pair-previews] [--strict-unistitch]
                    [--diagnostic-force]
                    INPUT
```

- `--stride` 和 `--max-frames` 仅用于诊断时显式覆盖自动布局；不传时按实测运动自适应选帧；
- `--max-canvas-megapixels` 始终限制单个几何画布；`scan_seam` 还以“画布 MP × 最终渲染源帧数”检查 aggregate working-set，任一预算超限都会在全分辨率 warp 前失败。这是对同时常驻工作量的保守约束，不是输出 JPEG 大小；
- `--blend-mode scan_seam` 是唯一可发布交付件的序列模式：沿扫描轴给每帧分配单源区域，在相邻重叠带内以颜色、梯度和对齐深度构造风险图，再用单调动态规划让接缝绕开近物体、深度断层和高残差区域，最后执行有限多频段融合。找不到安全接缝时拒绝发布；`feather` 仅为代码级诊断渲染器，命令行不会把它发布为交付件；
- `--render-frame-ids` 是保留的诊断覆盖项，正式序列命令会拒绝用手选帧发布交付件，防止通过少量帧绕过完整覆盖和清晰度选择；默认始终从完整输入自动选帧；
- `--translation-anchor-y` 是诊断用高级覆盖项；默认 `null`，从全画面鲁棒提取平移；
- `--scan-seam-margin` 是诊断用高级覆盖项；默认 `0` 时按图像宽度自动确定，`--scan-multiband-levels` 控制接缝低频平滑层数；
- `--scan-exposure-mode global_gain` 在完整重叠区求解每帧全局亮度增益，减少自动曝光造成的低频亮度带；`center_gain` 保留旧的中心块中值校正；
- `--scan-seam-mask-sigma` 对最终单源 owner mask 做窄高斯软化，默认 `1 px`，只柔化边界而不恢复整片重叠平均；
- `--scan-protect-region FRAME_ID:X0:Y0:X1:Y1` 仅保留为诊断用高级覆盖项；默认流程使用对齐深度自动避开近景，不需要用户画框；
- `--motion-model` 控制长序列累计约束：固定水平侧扫默认使用 `translation`，允许小旋转/尺度变化时使用 `similarity`，一般自由视角才使用 `homography`；
- `--no-pair-previews` 不保存逐对 FFD/TPS 诊断图；
- `--strict-unistitch` 禁用 LightGlue + MAGSAC 的择优比较和回退，任一 UniStitch global 边不合格即终止；
- `--diagnostic-force` 仅用于查看失败场景：放宽 UniStitch/MAGSAC 正式几何门限，并绕过输入和最终渲染质量门禁；仍保留有限矩阵、画布和内存安全限制，只写 `diagnostic_panorama.jpg` 与 `diagnostic_report.json`，永不写正式全景或 `delivery.json`；
- 所有选中帧必须具有相同尺寸，任一相邻对失败都会使本次序列命令返回非零退出码。

序列实现明确区分“局部双图预览”和“可组合全局布局”：

1. 在相邻原始关键帧上运行 LightGlue 和完整 UniStitch；
2. 以匹配点验证 UniStitch global homography 的方向和重投影误差；
3. 同时拟合 USAC/MAGSAC；只有内点数不少于 `min_matches` 且内点率至少为 50% 时才把它视为可用候选。默认在同一组 MAGSAC 内点上比较两种变换，若 MAGSAC 中值误差更低则记录 `magsac_preferred`；
4. 若未触发择优，UniStitch 的全匹配中值误差通过门限时记录 `unistitch_global`；否则采用可用 MAGSAC 并记录 `magsac_fallback`，两者均不可用则终止；
5. 将通过验证的变换投影到配置的运动模型；默认 `translation` 从全画面鲁棒提取平移，同时拒绝方向反转、相邻大跳和过大垂直运动；
6. 将每条受约束的 `source_to_reference` 变换累积到第一帧坐标系；
7. `scan_seam` 从原始全分辨率帧中按覆盖与绝对清晰度选择关键帧，先补偿全局曝光，再以扫描轴分配单源区域；颜色、梯度和对齐深度共同引导单调 DP 接缝绕开近物体与高视差区。最终只对 owner 边界做窄 mask 软化与多频段融合，并按有效 mask 的最大内接矩形自动裁剪黑边；任一输入、几何、接缝或裁剪质量门禁失败都不会发布 `panorama.jpg`。

它不会把不断变宽的累计全景再次输入 UniStitch，也不会把局部 FFD/TPS 网格错误地当作可直接连乘的全局变换。逐对 FFD/TPS 只用于 `pairs/*.jpg` 视觉诊断；最终长图使用经过运动模型约束的 global 布局和原图单次渲染，避免递归重采样。

输出结构：

```text
outputs/sequence/
├─ panorama.jpg       原始帧一次性渲染的 RGB 长图
├─ transforms.json    每帧到第一帧坐标系的 3×3 累计变换
├─ render_transforms.json  最终清晰关键帧及其插值变换（scan_seam）
├─ report.json        画布、选帧、耗时、逐对质量和实际布局方法
├─ delivery.json      最后原子写入的交付成功标记；只有存在它才可使用本次全景
└─ pairs/             非交付诊断目录；开启预览时写入，也可能保留上一次诊断内容
   ├─ 0000_0001.jpg
   └─ ...
```

若质量门禁或普通运行异常失败，正式交付文件会被清除并写入 `failure.json`，其中包含失败原因；旧的 `panorama.jpg` 不会被误当成本次结果。即使进程被强制终止而来不及写失败报告，只要没有 `delivery.json` 就不得把目录视为交付成功。

仅查看不合格结果时，可以显式加入 `--diagnostic-force`。诊断成功后输出结构为：

```text
outputs/diagnostic/
├─ diagnostic_panorama.jpg
└─ diagnostic_report.json
```

诊断模式不会生成 `panorama.jpg`、`report.json` 或 `delivery.json`，其图像不得作为交付件。

`configs/demo.yaml` 中的保护参数已经由真实 CLI 使用：

- `min_matches: 40`：LightGlue 的最低匹配数量，也是 MAGSAC 候选的最低内点数；MAGSAC 还必须达到至少 50% 的内点率；
- `max_unistitch_reprojection_px: 20.0`：推理分辨率下 UniStitch global 分支的中值重投影拒绝门限；
- `allow_magsac_fallback: true`：允许构造 LightGlue + USAC/MAGSAC 候选；这个历史名称同时控制 `magsac_preferred` 的择优路径和 `magsac_fallback` 的回退路径；
- `prefer_magsac_layout: true`：可用 MAGSAC 与 UniStitch 在同一组 MAGSAC 内点上比较时，MAGSAC 中值残差更低便直接采用，而不是等 UniStitch 超阈值后才回退；
- `sequence_motion_model: translation`：按固定水平侧扫模型累计位移，避免自由单应的投影项产生长程梯形漂移；
- `translation_anchor_y: null`：自动从全画面鲁棒提取平移，不绑定某一固定高度的物体层；
- `sequence_blend_mode: scan_seam`：禁止几十张重叠帧整幅平均；
- `scan_max_keyframes: 0`、`scan_seam_margin: 0`：关键帧数量和接缝搜索带宽均按轨迹、清晰度和图像宽度自动确定；`scan_multiband_levels` 控制多频段融合层数；
- `scan_exposure_mode: global_gain`、`scan_seam_mask_sigma: 1.0`：先校正关键帧的全局亮度，再只对最终 owner 边界做约 1 px 的软化；
- `max_pair_canvas: 4000`：限制逐对 FFD/TPS 异常画布；
- `max_canvas_megapixels: 200`：同时限制单个最终几何画布和“画布 MP × 最终渲染源帧数”的工作集，为显式归一化多频段金字塔保留内存余量；
- `save_pair_previews: false`：默认不写大量逐对诊断图，降低长序列 I/O；调试时可在自定义配置中开启。

## RGB 与三维输出的边界

当前采集会话已经保留构建彩色三维地图所需的基础数据：对齐深度、深度比例、内外参和时间戳；需要研究原生深度坐标时可用 `--raw-depth` 额外保存原始深度。但当前仓库没有 RGB-D 3D-3D RANSAC、ICP、位姿图、点云融合或 TSDF 模块，因此不能声称已经输出三维地图。

UniStitch 的 FFD/TPS 是为了视觉对齐而设计的二维非刚性变形，会改变局部形状和尺度。它不能：

- 作为相机 SE(3) 位姿；
- 直接用于扭曲深度图后进行米制测量；
- 为任意长度路线提供无漂移三维轨迹；
- 把多深度层场景变成严格无失真的正射图。

后续三维板块应独立使用有效 RGB-D 对应点估计 SE(3)，再通过 ICP/位姿图和分块点云或 TSDF 融合；RGB 长图可以与三维地图共享经过验证的全局轨迹，但不能用非刚性 RGB warp 代替三维几何。

## 已知限制与风险

- **无编码器、无外接 IMU**：系统完全依赖图像重叠，匀速距离和长期尺度没有独立观测。
- **没有真实闭环**：沿一列铁架单向前进通常不会回到旧区域，位姿图只能平滑，不能凭空消除系统性漂移。
- **周期性铁架**：非常容易错配到相邻重复单元。方向、最大位移、匹配覆盖和全局一致性必须同时检查；宁可断开子地图，也不要接受整周期错误。
- **叶片运动**：风机、人员和自然摆动会引起双影或错误局部变形。
- **金属和深度边缘**：细铁杆、遮挡边界和反射表面可能产生空洞或飞点。
- **域外数据**：公开模型主要使用 UDIS-D 图像对训练，并未证明能可靠处理温室无限侧扫序列。
- **显存**：FFD 输出显存随分辨率快速增加。默认 640 px 推理宽度用于降低风险；原始 `1280×800` 可能需要约 10 GB 级显存。
- **行程长度**：默认不再固定截取 40 帧，而是自动分割连续单向主扫描段并按位移选帧；`layout_max_frames: 160` 和 `max_canvas_megapixels: 200` 分别约束布局规模和渲染工作集。超过预算会明确要求拆分路线，而不是降质输出。
- **二维结果不是计量正射图**：铁架、叶片和背景分处不同深度，视觉全景不能保证各深度层同时保持真实尺寸。

## 常见问题

### PowerShell 不允许执行脚本

仅对当前窗口放开：

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

也可以使用 `powershell -ExecutionPolicy Bypass -File ...` 直接运行指定脚本。

### `No Orbbec camera found`

检查 USB 3 连接、线材、设备管理器，并关闭所有可能占用设备的程序。不要同时运行 OrbbecViewer 和本采集程序。

### 找不到精确的流配置

错误信息会列出实机 profile。修改 `configs/demo.yaml` 或使用 `--width`、`--height`、`--fps`，选择彩色和 Y16 深度都能工作的组合。

### `frames.csv` 的 metadata 字段为空

重新运行 `scripts/register_orbbec_metadata.ps1`，重新插拔相机，再录制一段短数据检查 `manifest.json` 的 `metadata_support`。

### 写队列出现丢帧

优先使用本地 NVMe，关闭实时杀毒扫描和其他高磁盘负载程序。默认只保存对齐深度，当前电脑已实测 300 帧零丢失；同时加 `--raw-depth` 会多编码一张 16 位 PNG，当前单写盘线程可能跟不上 30 fps，因此仅建议短期标定/诊断使用。不要在不记录原因的情况下忽略 `queue_drops`。

### Windows 报 `0xc00d3704` 或硬件 MFT 资源不足

先关闭所有相机程序并重新插拔 Gemini 305。若上一次采集进程被强制结束，相机/UVC 资源可能没有正常释放；重新插拔或通过 SDK 重启相机后再测试。不要因为该错误就盲目优先 MJPG：当前电脑的 MJPG Media Foundation 路径曾触发该错误，而 RGB 和低带宽 YUYV 在相机重启后均已实测成功。

### `torch.cuda.is_available()` 为 `False`

采集和合成仍可使用；UniStitch 不可用。根据 NVIDIA 驱动重新安装匹配的 CUDA PyTorch，再次运行检查命令。

若导入 PyTorch 时提示 Windows App Control/WDAC 阻止 `c10.dll`，需要由系统策略放行该已安装运行库或改用组织认可的签名环境；这不是调低拼接参数可以解决的问题。当前开发机仍受此策略限制，解除前尚未完成真实 CUDA 模型验收。

### CUDA out of memory

把 `stitch.inference_width` 保持在 640 或进一步降低，减少一次处理的帧数，关闭其他 GPU 程序。不要直接用累计长图作为 UniStitch 输入。

### 找不到 `unistitch-pair` 或 `unistitch-sequence` 命令

确认已执行 editable 安装并激活正确环境。Conda 用户可直接运行：

```powershell
conda run --prefix .\.conda unistitch-pair --help
```

### 匹配数不足或 global 分支验证失败

先检查运动模糊、曝光、相邻帧重叠和图片尺寸；默认自动布局会根据实测位移选帧。若仍失败，应补光、降低运动速度或重新采集，不要靠显式 `--stride` 关闭自动布局并强行接受错误边。默认模式同时验证 UniStitch global 与 LightGlue + MAGSAC；MAGSAC 必须满足最低内点数和至少 50% 的内点率，并且择优比较会让两种变换使用同一组 MAGSAC 内点。使用 `--strict-unistitch` 会禁用 MAGSAC 的择优与回退路径。周期铁架场景不要通过盲目放宽误差阈值来强行接受错误边。

## 测试

安装测试依赖后运行：

```powershell
python -m pytest
```

当前开发机已完成以下 smoke：

- 完整纯离线测试套件全部通过（具体数量以当前 `pytest` 输出为准）；
- 合成 6 帧序列的真值步长为 100 px，默认平移约束得到累计位置约 `0, 100.6, 201.6, 302.2, 402.6, 503.7 px`；
- Gemini 305 以 RGB + Y16 `1280×800@30` 连续采集 30 帧，实测约 30.022 fps，写入/丢帧/时间戳回退为 `30/0/0`，RGB-D 设备时间戳差为 0–1 µs；
- 仅保存 RGB 与对齐深度时连续采集 300 帧，写入/丢帧为 `300/0`，写盘队列保持在约 1–2 帧；
- 5 张真实静态 RGB 帧的 UniStitch 匹配数为 470–487，中值重投影误差为 1.15–1.29 px，累计平移保持在约 1 px 内。

这些数字只说明当前电脑和当前静态场景的链路已经跑通，不替代温室运动数据验收。

合成数据测试不能替代实机验收。相机采集至少应分别验证静止、0.5 m/s、1.0 m/s 和 1.5 m/s，并检查时间戳、写入丢帧、深度有效率和运动模糊。

## 第三方项目与许可证

- [UniStitch 代码](https://github.com/MmelodYy/UniStitch)，固定到提交 `78ebe7c07d516c591810337475ccdd4f2beff384`；
- [UniStitch 论文](https://arxiv.org/abs/2603.10568)；
- [UniStitch 模型](https://huggingface.co/Y5Y/UniStitch_model)；
- [LightGlue](https://github.com/cvg/LightGlue)，固定到提交 `746fac2c042e05d1865315b1413419f1c1e7ba55`；
- [Orbbec Python SDK v2](https://github.com/orbbec/pyorbbecsdk)。

第三方代码来源、固定版本和许可证说明见 [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)。

`conda activate .\.conda
g305-capture --output .\data\captures`

``unistitch-sequence `
  .\data\captures\run_20260713_152721 `
  --output .\outputs\greenhouse_sequence `
  --diagnostic-force``
