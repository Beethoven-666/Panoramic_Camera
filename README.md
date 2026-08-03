# Gemini 305 RGB-D 移动侧扫全景

这是一个面向奥比中光 Gemini 305 的 fail-closed RGB-D 侧扫全景程序。项目主要使用**照片模式驱动的低帧率同步 RGB-D 序列**采集输入，再以完整 ORB-SLAM3 RGB-D 真实轨迹、相邻 Open3D RGB-D 几何验证和统一中央窄条 RGB renderer 生成正式全景。

正式输出的每个颜色像素都来自原始 RGB 的一次标定 inverse remap。aligned depth 只在相邻风险走廊中帮助判断可见性、遮挡和受限局部采样，不生成颜色、不补洞、不修改 pose。TSDF 只生成独立的三维浏览附件，不参与全景构图或质量等级。

## 正式程序概览

```text
照片模式驱动的低帧率同步 RGB-D 会话
  → manifest / calibration / frames.csv / aligned depth 严格验证
  → Open3D 相邻短基线 RGB-D odometry
  → ORB-SLAM3 RGB-D 完整序列 camera_to_world
  → SE(3)、边残差、单向侧移和图连通审计
  → unified_calibrated_central_strip/v1
     · 每个真实源一次全分辨率标定 RGB inverse remap
     · 单一画布、valid mask 和 owner-frame map
     · 相邻风险走廊 RGB-D 可见性和受限局部 inverse sampling
     · 单调 hard owner 与安全窄带局部 MultiBand
  → A/B/C/F 发布判定
  → 独立只读 TSDF desktop/mobile GLB + Viewer
  → delivery.json 最后原子发布
```

默认工况是相机连续单向水平侧移、场景基本静止、最近物体约 `0.5 m`、最高速度约 `1.5 m/s`。普通用户不需要调整曝光、位姿、条带、接缝、融合或裁剪参数。

## 快速开始

### 1. 安装当前工作区

正式 Windows 主环境为 `D:\Panoramic_Camera\.conda`，Python 要求 `>=3.10,<3.13`：

```powershell
cd D:\central_strip_Panoramic_Camera

& 'D:\Panoramic_Camera\.conda\python.exe' -m pip install -e '.[capture,test,cuda13]'
```

Gemini 305 采集需要 `pyorbbecsdk2`。正式全景还需要：

- Windows 可用的 Gemini 305 与 metadata 注册；
- WSL 中可运行的 ORB-SLAM3 RGB-D；
- 项目构建的 Open3D 0.19 CUDA wheel；
- NVIDIA GPU 与 CUDA 12.8 runtime（Open3D）；
- CUDA 13 可供 CuPy 的其它已接入算子使用。

确认命令和包来自当前工作区：

```powershell
(Get-Command g305-capture).Source
(Get-Command g305-panorama).Source

& 'D:\Panoramic_Camera\.conda\python.exe' -c `
  "import panorama_demo; print(panorama_demo.__file__)"
```

正式调用应使用主环境 `Scripts` 下的启动器，不要使用系统 Python 目录中的旧同名命令。

### 2. 采集低帧率 RGB-D 照片序列

照片模式是 Gemini 305 的主要正式采集方式：

```powershell
& 'D:\Panoramic_Camera\.conda\Scripts\g305-capture.exe' `
  --photo-mode `
  --max-frames 120 `
  --output 'D:\central_strip_Panoramic_Camera\data\captures'
```

也可以按时长停止：

```powershell
& 'D:\Panoramic_Camera\.conda\Scripts\g305-capture.exe' `
  --photo-mode `
  --duration 20 `
  --output 'D:\central_strip_Panoramic_Camera\data\captures'
```

采集时保持相机连续、单向、近似水平侧移。照片模式不显示视频，不添加人为帧率限制；它会在每张 RGB-D 照片完整接收、对齐和写盘后再触发下一张。

成功后会得到类似目录：

```text
data/captures/run_YYYYMMDD_HHMMSS/
├─ color/
├─ depth_aligned/
├─ calibration.json
├─ frames.csv
└─ manifest.json
```

只有最终 `manifest.json` 同时满足以下条件，该会话才允许正式拼接：

```json
{
  "clean_shutdown": true,
  "formal_stitch_allowed": true,
  "capture_mode": "software_triggered_rgbd_photo_sequence"
}
```

### 2.1 采集连续 RGB-D 视频（独立视频产品输入）

不带 `--photo-mode` 时，`g305-capture` 进入与照片模式隔离的连续 RGB-D 视频路径：

```powershell
& 'D:\Panoramic_Camera\.conda\Scripts\g305-capture.exe' `
  --duration 3 `
  --video-exposure-us 600 `
  --output 'D:\central_strip_Panoramic_Camera\data\captures\video'
```

此模式与照片全景隔离：它不能传给 `g305-panorama`，但可以传给独立的
`g305-video-panorama`。默认始终启用相机自动曝光（快门时间随 AE 自动调节）、自动增益和自动白平衡；不会在预热后锁定控制值。需要固定视频曝光时使用
`--video-exposure-us 800`（不能与 `--photo-mode` 同用）；此时仍保持自动增益和自动白平衡。

默认视频同步配置将 `trigger_to_image_delay_us`（触发到图像采集延时）和
`trigger_out_delay_us` 都设为 `17000 µs`。程序在启动写入后立即回读，并在每个完整对齐
RGB-D 帧进入写盘前再次回读完整同步配置。任一帧读到的模式、Trigger Out gate、图像延时
或 Trigger Out 延时与配置不符，采集会立即失败，最终 `manifest.json` 不会标记为干净关闭。
成功会话在 `external_sync_output.per_frame_readback_verified_frames` 记录逐帧回读通过数。

视频会话会写入 `capture_mode="continuous_rgbd_video_auto"` 或
`continuous_rgbd_video_fixed_exposure`，并保留 `diagnostic_only=true`、
`formal_stitch_allowed=false` 以拒绝照片流程。v2 会话在安全关闭且没有写盘错误时还会写入
`product_eligibility={"photo_panorama": false, "video_panorama": true}`。

### 2.2 生成独立视频全景与三维附件

视频入口会选择最长连续单向扫描段，运行完整段的真实 ORB-SLAM3 RGB-D 轨迹，对全部实际
渲染源执行相邻 Open3D RGB-D 审计，然后复用统一 calibrated central-strip renderer。它不会
插值 pose，也不会使用 Open3D 或二维运动代替缺失 ORB pose。

```powershell
& 'D:\Panoramic_Camera\.conda\Scripts\g305-video-panorama.exe' `
  'D:\central_strip_Panoramic_Camera\data\captures\run_YYYYMMDD_HHMMSS' `
  --output 'D:\central_strip_Panoramic_Camera\outputs\video_sequence'
```

默认在 2-D 发布后生成 GLB；若要延后：

```powershell
& 'D:\Panoramic_Camera\.conda\Scripts\g305-video-panorama.exe' SESSION --output OUTPUT --defer-3d
& 'D:\Panoramic_Camera\.conda\Scripts\g305-video-3d.exe' OUTPUT --input SESSION
```

自动曝光、超过 `1200 µs` 的曝光或严格质量未过但结构完整的视频会发布为 C 级：
`video_delivery.json` 的 `delivery_state` 为 `published_degraded`，并要求人工复核。2-D 交付
包含 `video_panorama.jpg/png`、`video_pixel_provenance.npz`、`video_report.json` 和最后写入的
`video_delivery.json`。三维发布是独立的：`video_tsdf_mesh.glb`、mobile GLB、离线
`video_tsdf_mesh_viewer.html` 及 `video_3d_delivery.json`；3-D 失败只写
`video_3d_failure.json`，不会撤销已经发布的 2-D 交付。

### 3. 验证 CUDA Open3D 与 ORB-SLAM3

正式并行位姿前端要求 Open3D 相邻边实际使用 `open3d_tensor_cuda_rgbd`。先执行：

```powershell
$env:G305_CUDA = 'required'

& 'D:\Panoramic_Camera\.conda\python.exe' `
  .\scripts\verify_open3d_cuda.py
```

输出必须包含：

```text
"cuda_available": true
"odometry_backend": "open3d_tensor_cuda_rgbd"
```

默认 ORB-SLAM3 WSL 路径来自 `configs/demo.yaml`：

```powershell
wsl.exe -e bash -lc `
  'test -f ~/Projects/ORB_SLAM3_WS/ORB_SLAM3/Examples/RGB-D/rgbd_tum && echo orb-executable-ok'

wsl.exe -e bash -lc `
  'test -f ~/Projects/ORB_SLAM3_WS/ORB_SLAM3/Vocabulary/ORBvoc.txt && echo vocabulary-ok'
```

### 4. 运行正式全景

```powershell
$env:G305_CUDA = 'required'

& 'D:\Panoramic_Camera\.conda\Scripts\g305-panorama.exe' `
  '你的会话目录' `
  --output '你的输出目录'
```

实际示例：

```powershell
$env:G305_CUDA = 'required'

& 'D:\Panoramic_Camera\.conda\Scripts\g305-panorama.exe' `
  'D:\central_strip_Panoramic_Camera\data\captures\run_YYYYMMDD_HHMMSS' `
  --output 'D:\central_strip_Panoramic_Camera\outputs\greenhouse_sequence'
```

不要直接调用 PATH 中来源不明的 `g305-panorama`。正式运行前可以再次确认：

```powershell
(Get-Command g305-panorama).Source
```

它应指向：

```text
D:\Panoramic_Camera\.conda\Scripts\g305-panorama.exe
```

### 5. 检查发布结果

正式 A/B/C 交付目录包含：

```text
outputs/greenhouse_sequence/
├─ panorama.jpg
├─ panorama.png
├─ pixel_provenance.npz
├─ transforms.json
├─ render_transforms.json
├─ report.json
├─ tsdf_mesh.glb
├─ tsdf_mesh_mobile.glb
├─ tsdf_mesh_viewer.html
└─ delivery.json
```

检查最终状态：

```powershell
$G305Output = 'D:\central_strip_Panoramic_Camera\outputs\greenhouse_sequence'
$G305Delivery = Get-Content "$G305Output\delivery.json" -Raw | ConvertFrom-Json

$G305Delivery |
  Select-Object delivery_state, quality_grade, strict_quality_pass, manual_review_required
```

`delivery.json` 是最后写入的成功标记。没有它就没有正式交付。

通过本地 HTTP server 查看 TSDF：

```powershell
cd $G305Output
& 'D:\Panoramic_Camera\.conda\python.exe' -m http.server 8080
```

然后访问 [http://localhost:8080/tsdf_mesh_viewer.html](http://localhost:8080/tsdf_mesh_viewer.html)。

## 照片模式的正式采集契约

照片模式由 `g305-capture --photo-mode` 和 `photo_capture.py` 实现，核心规则如下：

- 使用 `SOFTWARE_TRIGGERING`；
- `frames_per_trigger=1`；
- `Trigger Out Enable=true`；
- 触发到图像采集延时 `trigger_to_image_delay_us=17000 µs`；
- Trigger Out 延时固定为 `17000 µs`；
- 彩色曝光不超过 `800 µs`；
- 在指定分辨率下自动选择彩色与 Y16 深度共同支持、且能同时容纳图像延时和 Trigger Out 延时的最高 FPS；
- 不允许偷偷降低采集分辨率；
- 每个正式帧只调用一次 `device.trigger_capture()`；
- 每帧返回后都重新读取同步配置，确认图像延时和 Trigger Out 延时仍为设定值；
- 上一帧必须完整收取、COLOR_STREAM 对齐并写盘成功后，才能触发下一帧；
- 正式失败路径不重触发。

准备阶段允许最多 8 次内部预热触发，但物理输出 gate 必须保持关闭。取得完整预热 RGB-D 后，程序仍保持 gate 关闭，等待从最后一次内部触发开始的完整迟到响应窗口，并确认队列为空，之后才开放正式 Trigger Out。

会话打开期间：

```text
clean_shutdown=false
formal_stitch_allowed=false
```

只有相机、pipeline、writer 和设备配置都安全关闭，且没有采集、队列或写盘错误时，最终 manifest 才会将二者设为 `true`。强制结束、写盘失败、同步不确定或设备恢复失败的会话不能正式拼接。

连续流采集仍可通过不带 `--photo-mode` 的 `g305-capture` 使用，具体控制行为和限制见“采集连续 RGB-D 视频（独立视频产品输入）”。它不能替代照片模式的固定控制 RGB-D 会话，也不能传给 `g305-panorama`；RGB-only 截图、普通照片目录或未对齐深度同样不能替代任何正式 RGB-D 输入。

## 严格 RGB-D 会话

`g305-panorama` 只接受会话目录或它的 `frames.csv`。正式加载要求：

- `manifest.json` schema 为受支持版本；
- `clean_shutdown=true`；
- `calibration.json` 包含有限有效的彩色内参与畸变；
- 明确声明深度已对齐到彩色坐标系；
- 每帧有 RGB；
- 每帧有 `depth_aligned/` 内、与 RGB 同尺寸的单通道 `uint16 PNG`；
- `frames.csv` 有 `aligned_depth_path`；
- `depth_scale_mm_per_unit` 为有限正数；
- 彩色时间戳非负；
- `color_exposure` metadata 为正数。

`raw_depth_path`、`depth_path` 或其它目录不能冒充 `aligned_depth_path`。内部深度与 pose 平移使用毫米，Open3D 适配层才临时转换为米。

设备 `color_exposure` metadata 固定按 `100 µs/单位` 解释。照片模式正式上限为 `800 µs`，正式输入绝对拒绝上限为 `1200 µs`。缺 manifest、标定、aligned depth、单位、对齐 provenance、时间戳或曝光属于结构失败，`--diagnostic-force` 也不能绕过。

## 位姿前端

默认 `pose_backend=hybrid_orbslam3_rgbd`：

1. Open3D 对每个相邻短基线 RGB-D pair 测量 source-to-reference SE(3)、fitness、RMSE 和 `6×6` information matrix。
2. ORB-SLAM3 RGB-D 对完整短基线序列求真实全局 `camera_to_world`。
3. 程序检查每帧覆盖、图连通、SE(3) 刚体性、单向横移、步长、扫描跨度、上下/前后漂移、旋转和 Open3D 边残差。

正式并行前端要求全部 Open3D 边的 backend 都是：

```text
open3d_tensor_cuda_rgbd
```

如果终端最终报告：

```text
Formal parallel pose frontend requires every Open3D RGB-D edge to use
open3d_tensor_cuda_rgbd; observed open3d_rgbd
```

说明当前 Open3D 是 CPU/旧接口或 CUDA runtime 没有生效。不要放宽检查；应重新安装并验证项目的 CUDA Open3D wheel，并使用主 Conda 环境启动器。

ORB-SLAM3 未安装、进程失败或未跟踪全部正式帧时，程序失败，不会回退为 Open3D-only 全局轨迹，也不会插值缺失 pose。

可独立重新运行完整 ORB-SLAM3 并导出轨迹：

```powershell
& 'D:\Panoramic_Camera\.conda\Scripts\g305-orbslam3-trajectory.exe' `
  'D:\central_strip_Panoramic_Camera\data\captures\run_YYYYMMDD_HHMMSS' `
  --output 'D:\central_strip_Panoramic_Camera\outputs\orbslam3_trajectory.json'
```

输出 schema 为 `gemini305-orbslam3-trajectory/v1`；该命令不读取历史 pose sidecar，也不以 Open3D 替代缺失 ORB-SLAM3 pose。

## Unified calibrated central-strip renderer

当前正式 renderer 是：

```text
unified_calibrated_central_strip/v1
```

它具有以下不变量：

- 只调用 `render_calibrated_rgb_pushbroom` 产生正式 RGB 主图；
- 每个真实 pose node 的 RGB 只做一次全分辨率标定 inverse remap；
- 所有内容共享单一画布、valid mask 和 owner-frame map；
- 中间源中央条带不超过输入宽度的 `20%`；
- 首尾源只向扫描外侧扩展到校准图像边缘；
- 画布和 aggregate working set 分别不超过 `200 MP`；
- 保留主扫描段全部真实 pose nodes，不设固定数量上限；
- 同时常驻的 RGB 条带不超过 5。

布局 x 比例由相邻 RGB 局部运动与已审计 SE(3) 相机中心位移的稳健比值决定。这个比例只控制条带位置，不是二维 pose、单应矩阵、深度平面或 pose 修正。

因此，序列帧数不会因固定 pose-node 数量而被截断；实际可交付长度仍受 `200 MP` 画布/aggregate working set 上限约束，超过上限会 fail-closed，而不是增量续拼。

`metric_mosaic` 和 `inspection_multiview` 模块目前仍保留配置兼容验证、历史测试和隔离实现，但在 `unified_content_mode=true` 的正式路径中不会生成第二张 RGB 主图、后渲染 overlay 或失败回退。

## RGB-D 风险走廊、owner 与融合

RGB Lab/梯度风险先决定 owner 和禁止融合区域。只有跨名义 seam 的结构性 raw seed、明显边缘残差或整高 hard cut 指向几何问题时，程序才读取相邻风险走廊的 aligned depth。

风险走廊宽度限制在 `96–160 px`。双向重投影和 z-buffer 把区域分为：

- 同层安全背景；
- 遮挡；
- disocclusion；
- 深度边界；
- 深度孔洞、透明或不可靠区域。

深度一致容差是：

```text
max(20 mm, 2% × depth, 3σ_depth)
```

没有可审计深度噪声 provenance 时，`σ_depth=0`。遮挡、孔洞、透明/反光和强 RGB 结构保持单一 RGB owner，不允许 MultiBand。

局部 inverse mesh 只能作用于双向可见、同层、未保护的安全背景，并必须通过：

- 训练与 held-out 分离；
- 前后向 flow；
- 边界零位移；
- 最大局部位移 `≤8 px`；
- 正 Jacobian；
- 与保护域零交集；
- 全分辨率误差和直线保持审计。

失败时继续使用单一 hard owner，不会半信任地应用 warp。

`local_apap_flow` 默认关闭。即使显式开启，也只允许在一个相邻走廊和单一 RGB owner 内使用，并必须通过全部同层、保护域、对应、held-out、尺度、Jacobian、位移和边界审计。

GraphCut 只让相邻真实源形成单调 hard owner。有效区域每像素恰有一个 owner，无效区域没有 owner。MultiBand 仅用于共同有效、低梯度、无风险的安全背景，带宽为：

```text
clamp(floor(0.20 × 较窄 owner 宽度), 2, 8)
```

最多 3 层。程序不回退到 feather、平均、全局金字塔、全图模糊或补洞。

## A/B/C/F 发布语义

结构安全和严格质量分开表达：

| 等级 | 语义 |
| --- | --- |
| A | 输入、轨迹和最终渲染严格质量全部通过，handoff 为安全 anchor/owner；`delivery_state=published`。 |
| B | 严格质量通过，并使用完整审计的 `flow_mesh`、显式启用的 `apap`，或 photometric pair 使用 `rgb_texture_consistent`；`delivery_state=published`。 |
| C | 严格会话、真实轨迹、一次 remap、owner 拓扑和 pair 审计完整，但严格质量未通过或使用 `hard_cut_degraded`；`delivery_state=published_degraded`，必须人工复核。 |
| F | 会话、轨迹、remap、owner/MultiBand、资源、TSDF/GLB/Viewer 或原子发布任一结构条件失败；不发布 `delivery.json`。 |

正式 policy 固定为：

```yaml
handoff_fallback_policy:
  publish_degraded: true
  local_apap_flow_enabled: false
  manual_review_for_grade_c: true
```

当前 schema：

| 文件 | Schema |
| --- | --- |
| `report.json` | `gemini305-unified-central-strip/v12-r1` |
| `delivery.json` | `gemini305-panorama-delivery/v12-r1` |
| `render_transforms.json` | `unified-calibrated-central-strip/v1` |
| `pixel_provenance.npz` | `owner-frame-id-r1` |
| `failure.json` | `gemini305-panorama-failure/v2` |

`quality_pass` 只是 `strict_quality_pass` 的兼容别名。判断是否发布必须同时检查：

- `delivery.json` 是否存在；
- `delivery_state`；
- `quality_grade`；
- `manual_review_required`。

## TSDF 三维浏览附件

正式 A/B/C 都必须发布：

```text
tsdf_mesh.glb
tsdf_mesh_mobile.glb
tsdf_mesh_viewer.html
```

默认 `colour_mode=rgbd_tsdf_vertex_colour`，Open3D 把 aligned RGB 与 depth 融合到同一 TSDF volume，并以 `COLOR_0` 输出连续顶点色。TSDF 使用已审计真实 pose，但只供浏览：

- 不向全景提供颜色；
- 不改变 RGB owner；
- 不参与 seam、融合或 crop；
- 不影响 A/B/C 的 RGB 决策；
- 任一 GLB/Viewer 构建或发布失败仍属于结构 F。

### Viewer 画面上方的离散噪声分量

正式默认配置 `tsdf_visualization.upper_island_exclusion_maximum_y_mm: -500.0`
会在 GLB 导出前移除**完全断开**且所有顶点原始 TSDF `Y < -500 mm`
的连通三角面分量。glTF 导出会将相机式 `+Y-down` 转成 Viewer 的 `+Y-up`，
所以这正对应 Viewer 画面上方的漂浮噪声；不能把该条件改成较大的 `Y`，否则会误删画面下方。

该过滤只删除满足条件的完整连通分量：不会裁剪与主体相连、跨过阈值的网格，也不会修改
`panorama.jpg`、`panorama.png`、RGB owner、真实 pose 或质量等级。导出审计会记录阈值、删除的
分量/三角面数量及其边界；desktop/mobile GLB 使用同一筛选结果。

## CUDA Open3D 安装与验证

Windows 官方 Open3D wheel 不包含本项目正式位姿前端需要的 CUDA Tensor odometry。项目提供构建脚本，默认针对 CUDA Toolkit 12.8 和 `sm_120`：

```powershell
conda create -p D:\open3d_cuda_build\cuda128-toolkit `
  -c nvidia/label/cuda-12.8.1 `
  -c conda-forge `
  cuda-toolkit=12.8.1

.\scripts\build_open3d_cuda_windows.ps1
```

构建完成后安装生成的 wheel：

```powershell
$G305Wheel = Get-ChildItem `
  'D:\open3d_cuda_build\Open3D-v0.19.0\build-cuda12.8-sm120-v3\lib\python_package\pip_package' `
  -Filter 'open3d-*.whl' |
  Select-Object -First 1 -ExpandProperty FullName

& 'D:\Panoramic_Camera\.conda\python.exe' `
  -m pip install --force-reinstall --no-deps $G305Wheel

$env:G305_CUDA = 'required'
& 'D:\Panoramic_Camera\.conda\python.exe' `
  .\scripts\verify_open3d_cuda.py
```

如果 `import open3d` 报 `No module named 'open3d.cpu'`，说明环境中的 Open3D 安装不完整或 wheel 内容混杂。使用上面的 `--force-reinstall --no-deps` 重新安装项目 CUDA wheel，再运行验证脚本。

`G305_CUDA` 支持：

| 值 | 行为 |
| --- | --- |
| `prefer` | 默认；可用时优先 CUDA。 |
| `auto` | 对已接入的等价算子进行 CPU/CUDA 校准选择。 |
| `off` | 强制参考 CPU 路径；不能满足当前正式 Open3D CUDA edge 契约。 |
| `required` | 已接入 CUDA 的边界不可回退，适合正式运行前 fail-fast 验证。 |

`required` 不会把 ORB-SLAM3、GraphCut、MultiBand、DIS 或标量审计伪装成 GPU 算法。

## 诊断入口

| 命令 | 用途 |
| --- | --- |
| `g305-central-strip-diagnostic` | 独立中央条带诊断 |
| `g305-geometry-pair-diagnostic` | 完整当前轨迹下的相邻 geometry A/B 诊断 |
| `g305-foreground-deformation-diagnostic` | 默认关闭的前景局部 inverse mesh 实验 |
| `unistitch-pair` | 历史双图诊断 |

这些命令不能成为 `g305-panorama` 的 backend 或失败回退。独立诊断成功只发布：

```text
diagnostic_panorama.jpg
diagnostic_report.json
```

它们不写正式 `delivery.json`。

`g305-panorama --diagnostic-force` 可以绕过输入外观、odometry/pose 质量和最终图像质量阈值，但不能绕过：

- manifest、clean shutdown、标定和 aligned depth；
- 深度单位、时间戳和曝光 metadata；
- 有限真实 SE(3)、必需边和图连通；
- 有效 RGB inverse remap；
- owner/MultiBand 拓扑；
- `200 MP` 画布和 aggregate working set 资源上限；
- 原子发布语义。

## 原子交付与失败

每次运行首先使旧 `delivery.json` 失效。所有正式文件先写隐藏 pending 文件，再通过 `os.replace` 发布；`delivery.json` 始终最后写入。

普通异常或结构 F 会清理正式/诊断产物，并原子写：

```text
failure.json
```

强制终止可能来不及写 `failure.json`，但只要没有有效 `delivery.json`，就不能把输出视为正式交付。

## 合成数据

没有相机时可生成严格合成 RGB-D 会话：

```powershell
& 'D:\Panoramic_Camera\.conda\Scripts\generate-panorama-demo.exe' `
  --output 'D:\central_strip_Panoramic_Camera\data\synthetic\demo' `
  --frames 10 `
  --width 640 `
  --height 400 `
  --step 120 `
  --scene layered
```

合成数据可测试 manifest、标定、aligned depth、毫米单位、SE(3)、owner 和原子交付，但不能代替 Gemini 305、真实 Open3D CUDA 边、完整 ORB-SLAM3 或现场速度验收。

## 配置

正式默认配置位于 [configs/demo.yaml](configs/demo.yaml)。普通用户通常不需要传 `--config`。

同步延时使用 SDK 的微秒字段配置。照片模式是正式硬约束，两项都必须保持 `17000`；视频模式
也默认使用相同值，并允许在站点配置中显式设定，但两项都必须是非负整数且小于当前帧周期：

```yaml
capture:
  video_mode:
    trigger_out_delay_us: 17000
    trigger_to_image_delay_us: 17000
  photo_mode:
    trigger_out_delay_us: 17000
    trigger_to_image_delay_us: 17000
```

`trigger_to_image_delay_us` 是触发到图像采集的延时，`trigger_out_delay_us` 是触发到外部
Trigger Out 边沿的延时。它们分别写入并分别回读；默认设为相同值不代表程序用一个字段冒充另一个。

重要固定值：

| 项目 | 默认/硬限制 |
| --- | --- |
| 正式主采集 | 照片模式驱动的低帧率同步 RGB-D 序列 |
| 照片曝光上限 | `800 µs` |
| 输入曝光绝对上限 | `1200 µs` |
| 触发到图像采集延时 | 照片/视频默认 `17000 µs`，逐帧回读确认 |
| Trigger Out 延时 | `17000 µs` |
| 照片预热触发 | 最多 8 次，gate-off |
| 全局 pose | 完整 ORB-SLAM3 RGB-D |
| 相邻边 | Open3D Tensor CUDA RGB-D |
| 正式 renderer | `unified_calibrated_central_strip/v1` |
| 中间中央条带 | 输入宽度 `≤20%` |
| 风险走廊 | `96–160 px` |
| 局部位移 | `≤8 px` |
| MultiBand | 2–8 px，最多 3 层 |
| 画布/aggregate | 各 `≤200 MP` |
| pose nodes | 不设固定数量上限；保留全部真实节点，仍受 `200 MP` 资源上限约束 |
| 常驻 RGB 条带 | 2–5 |
| `local_apap_flow` | 默认关闭 |
| TSDF | 必需、只读、不得反馈 RGB 全景 |

正式配置只能等于或收紧安全默认值。需要放宽质量阈值的实验应进入隔离诊断，不应修改正式发布语义。

## 测试

```powershell
$G305Python = 'D:\Panoramic_Camera\.conda\python.exe'

& $G305Python -m pytest -q
ruff check src tests
& $G305Python -m compileall -q src tests
git diff --check
```

重点测试导航：

| 范围 | 测试 |
| --- | --- |
| 照片采集 | `test_photo_capture.py`、`test_capture_calibration.py` |
| 严格会话 | `test_session.py`、`test_v1_input_contract.py` |
| 位姿/CUDA | `test_rgbd_odometry.py`、`test_orbslam3_bridge.py`、`test_export_orbslam3_trajectory.py`、`test_cuda_backend.py` |
| Unified RGB | `test_calibrated_rgb_pushbroom.py`、`test_geometry_assisted_local_warp.py`、`test_handoff_continuity.py` |
| 发布 | `test_sequence_delivery.py`、`test_sequence_integration.py`、`test_config.py` |
| TSDF | `test_dense_fusion.py` |

## 常见问题

### 推送后仍调用旧 `g305-panorama`

检查：

```powershell
(Get-Command g305-panorama).Source
& 'D:\Panoramic_Camera\.conda\python.exe' -c `
  "import panorama_demo; print(panorama_demo.__file__)"
```

使用显式主环境启动器运行正式程序。

### Open3D 边显示 `open3d_rgbd`

当前正式程序需要 `open3d_tensor_cuda_rgbd`。重新安装项目 CUDA Open3D wheel，设置 `$env:G305_CUDA='required'`，并先运行 `scripts/verify_open3d_cuda.py`。

### 正式目录没有 `delivery.json`

检查 `failure.json` 和终端错误。没有 `delivery.json` 就没有正式交付；单独存在 JPEG、GLB 或历史 sidecar 不能代表成功。

### C 级是否算成功

C 是结构安全但严格质量未通过的降级发布：

```text
delivery_state=published_degraded
quality_grade=C
manual_review_required=true
```

它必须人工检查全景、owner/handoff 审计和 TSDF。

### 可以用 TSDF 修补全景吗

不可以。TSDF 是交付后的只读展示附件，不能提供全景颜色、补洞、修改 pose、选择 seam 或改变等级。

## 开发说明

开发代理约束见 [AGENTS.md](AGENTS.md)，SDK/API 说明见 [docs/SDK.md](docs/SDK.md)，版本变更见 [CHANGELOG.md](CHANGELOG.md)。
