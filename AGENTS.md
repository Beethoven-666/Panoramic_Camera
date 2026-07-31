# AGENTS.md

本文档是 `D:\central_strip_Panoramic_Camera` 的开发约束。开始工作前先阅读本文件，再按需查看 `README.md`、`configs/demo.yaml`、相关源码和测试。若说明与当前可执行源码、默认配置或测试不一致，以源码、默认配置和测试为准，并在同一改动中修正文档。

## 1. 当前目标与正式程序

项目面向奥比中光 Gemini 305，主要使用**照片模式驱动的低帧率同步 RGB-D 序列**生成 fail-closed 的移动侧扫全景。正式工况是相机连续单向水平侧移、场景基本静止、最近物体约 `0.5 m`、速度最高约 `1.5 m/s`。普通用户只提供采集输出目录、会话目录和全景输出目录，不调整曝光、位姿、条带、接缝、融合或裁剪算法参数。

正式入口是 `g305-panorama`，默认程序固定为：

```text
照片模式驱动的低帧率同步 RGB-D 会话
  → 严格 manifest / calibration / frames.csv / aligned depth 验证
  → 每帧短基线 Open3D RGB-D odometry
  → 完整序列 ORB-SLAM3 RGB-D 真实 camera_to_world 轨迹
  → 有限、连通、连续、单向的 SE(3) 与边残差审计
  → unified_calibrated_central_strip/v1
     · 每个真实源一次全分辨率标定 RGB inverse remap
     · 单一画布、valid mask 和 owner-frame map
     · 相邻风险走廊中的受限 RGB-D 可见性/局部 inverse sampling
     · 单调 hard owner 与安全窄带局部 MultiBand
  → A/B/C/F 判定
  → 独立只读 TSDF desktop/mobile GLB 与 Viewer
  → delivery.json 最后原子发布
```

### 1.1 正式 renderer 不变量

- 全景颜色只能来自原始 RGB 的一次标定 inverse remap。黑色 RGB 是有效内容，不能按颜色值删掉。
- 完整 ORB-SLAM3 RGB-D `camera_to_world` 链是唯一全局轨迹。Open3D 相邻边用于局部几何验证；不得以 Open3D-only、特征匹配、二维累计运动、单应矩阵或插值 pose 代替缺失 ORB-SLAM3 pose。
- `calibrated_rgb_pushbroom.unified_content_mode=true` 时，正式 RGB 输出只有一个 renderer、一个目标坐标域、一个 valid mask 和一个严格 owner map。
- `metric_mosaic`、`inspection_multiview` 目前保留配置验证和历史/诊断实现，但不能在 unified 正式路径产生第二幅 RGB 主图、overlay 或失败回退。
- aligned depth 只能在已触发的相邻 `96–160 px` 风险走廊中做双向重投影、z-buffer、层分类、遮挡/孔洞/透明保护和受限局部 inverse sampling。它不得生成颜色、补洞、拟合全局平面、修改 pose、构造全景深度或向 RGB 全景回传 TSDF 结果。
- TSDF 仅在 RGB 全景的结构和质量判定完成后生成展示附件，不参与条带、接缝、融合、裁剪、轨迹或等级决定。正式 GLB 导出可按 `tsdf_visualization.upper_island_exclusion_maximum_y_mm` 删除完全断开且 `max(raw Y) < threshold` 的上方噪声分量；glTF 将 `+Y-down` 翻转为 Viewer 的 `+Y-up`，故默认 `-500 mm` 对应 Viewer 画面上方，绝不可反向删除画面下方或裁剪跨阈值/主体连通分量。
- 禁止把 UniStitch、LightGlue、MAGSAC、Torch、全局/pose/全景级 `3×3` 单应、全局 flow 或时间/二维 pose 插值引入正式路径。

`unistitch-sequence` 是 `g305-panorama` 的弃用别名。`unistitch-pair` 仅用于历史双图诊断。`g305-central-strip-diagnostic`、`g305-geometry-pair-diagnostic` 和 `g305-foreground-deformation-diagnostic` 是隔离诊断入口，不能成为正式 backend、CLI 模式或失败回退。

## 2. 开始工作

1. 工作目录固定为 `D:\central_strip_Panoramic_Camera`。
2. 第一条命令运行 `git status --short`，保留用户和其他代理的所有改动、采集与输出目录。
3. 主环境是 `D:\Panoramic_Camera\.conda`；若项目已有本地 `.conda`，可按项目现状使用。不得无故删除或重建环境。
4. 搜索使用 `rg` / `rg --files`，文件修改使用补丁。
5. 不执行 `git reset --hard`、破坏性 checkout 或批量删除采集/输出数据。

常用验证：

```powershell
$G305Python = 'D:\Panoramic_Camera\.conda\python.exe'
& $G305Python -m pytest -q
ruff check src tests
& $G305Python -m compileall -q src tests
git diff --check
```

Open3D `0.19` 是正式依赖。Torch/Kornia/torchvision 仅属于 `unistitch-diagnostic` extra，不得成为正式依赖。涉及 CUDA 时使用现有 `G305_CUDA=prefer|auto|off|required` 审计；`required` 只强制已有 CUDA 实现的边界，不能把 ORB-SLAM3、GraphCut、MultiBand 或其它 CPU 算法伪装成 GPU。

## 3. 主要模块边界

| 路径 | 当前职责 |
| --- | --- |
| `configs/demo.yaml` | 照片采集、位姿、unified RGB renderer、接缝、TSDF 与发布安全默认值 |
| `capture_orbbec.py` | CLI、连续采集及照片模式路由、同步对齐和会话写盘基础设施 |
| `photo_capture.py` | 正式主采集路径：软件触发、Trigger Out 门控、逐帧同步 RGB-D 照片序列 |
| `session.py` | 严格 manifest、标定、aligned depth、时间戳、曝光与毫米单位契约 |
| `quality.py` | 输入画质、主扫描段、pose-node 与渲染源审计 |
| `rgbd_odometry.py` | Open3D 相邻短基线 RGB-D 边、信息矩阵和轨迹质量 |
| `orbslam3_bridge.py` | WSL ORB-SLAM3 RGB-D staging、执行、重试和真实轨迹解析 |
| `export_orbslam3_trajectory.py` | 独立重新运行完整 ORB-SLAM3 并原子导出真实轨迹 |
| `calibrated_rgb_pushbroom.py` | 唯一正式 unified central-strip RGB renderer |
| `geometry_assisted_local_warp.py` | 相邻 RGB-D 可见性、保护域、held-out 局部 inverse mesh |
| `handoff_continuity.py` / `local_apap_flow.py` | handoff 标量审计及默认关闭的 APAP/flow 候选 |
| `dense_fusion.py` | 交付后只读 TSDF、GLB 和 Viewer；不得向 RGB renderer 回传结果 |
| `stitch_sequence.py` | 正式编排、v12-r1 判定、失败清理和原子发布 |
| `metric_mosaic.py` / `inspection_multiview.py` | 兼容验证、历史/诊断实现；不是 unified 正式 RGB 输出 |
| `*_diagnostic.py`、`central_strip.py`、`rgbd_projection.py` | 隔离诊断或历史回归 |
| `tests/` | 采集、会话、轨迹、渲染、发布、CUDA 与集成回归 |

## 4. 正式采集：照片模式驱动的低帧率 RGB-D 序列

正式采集优先使用：

```powershell
& 'D:\Panoramic_Camera\.conda\Scripts\g305-capture.exe' `
  --photo-mode `
  --max-frames 120 `
  --output 'D:\central_strip_Panoramic_Camera\data\captures'
```

照片模式必须保持以下硬契约：

- 无预览，使用 `SOFTWARE_TRIGGERING`、`frames_per_trigger=1`、`Trigger Out Enable=true` 和固定 `17000 µs` Trigger Out 延时。
- 在配置分辨率下选择彩色与 Y16 深度共同支持且能容纳 Trigger Out 延时的最高 FPS；不允许偷偷降分辨率。
- 准备阶段先关闭并回读物理输出 gate，最多执行 8 次有界内部预热触发。预热期间 gate 始终关闭。
- 获得完整预热 RGB-D 后，gate 继续关闭，直到最后一次内部触发的完整迟到响应窗口结束并确认队列为空。
- 每个正式序列帧只调用一次 `device.trigger_capture()`；上一帧必须完整收取、COLOR_STREAM 对齐、写盘并确认成功后才能触发下一帧。失败路径不得重触发。
- 正式彩色曝光固定不超过 `800 µs`，设备 metadata 单位为 `100 µs/单位`。
- 会话打开期间 `formal_stitch_allowed=false`。只有相机/写盘资源安全关闭、无采集或写盘错误时，最终 manifest 才可写 `clean_shutdown=true`、`formal_stitch_allowed=true`。

连续流采集仍保留，但属于次要路径；不能用 RGB-only 截图替代正式照片 RGB-D 序列。

## 5. 严格 RGB-D 会话

正式输入只能是会话目录或其 `frames.csv`。必须存在受支持的 `manifest.json` 和 `calibration.json`，且 manifest 的 `clean_shutdown` 必须为 `true`。每帧必须包含：

- RGB 文件；
- `depth_aligned/` 内、与 RGB 同尺寸的单通道 `uint16 PNG`；
- `aligned_depth_path` 和有限正数 `depth_scale_mm_per_unit`；
- 有限有效的彩色内参、畸变和 color-target 对齐 provenance；
- 非负彩色时间戳；
- 正数 `color_exposure` metadata。

`raw_depth_path`、`depth_path` 或其它目录不能冒充 aligned depth。项目内部深度和 pose 平移统一使用毫米；只有 Open3D 适配层临时转换为米。缺失 manifest、clean shutdown、标定、aligned depth、单位、对齐声明、时间戳或曝光都是结构失败，`--diagnostic-force` 也不能绕过。

输入移动曝光绝对拒绝上限是 `1200 µs`；照片模式正式采集上限是 `800 µs`。无限 AE 会话必须标为 diagnostic-only，不能发布正式 `delivery.json`。

## 6. 位姿与正式 RGB 渲染

- 正式 `pose_backend=hybrid_orbslam3_rgbd`。每条 Open3D 边记录 source-to-reference SE(3)、收敛、fitness、RMSE、正定 `6×6` information matrix、深度有效率和失败原因。
- 正式并行前端要求每条 Open3D 边实际使用 `open3d_tensor_cuda_rgbd`。若观察到 `open3d_rgbd`，说明 CUDA Open3D 未生效，必须失败，不能静默接受 CPU legacy edge。
- ORB-SLAM3 必须跟踪完整正式序列。缺 pose、非有限/非刚体 SE(3)、图不连通、逆向或不连续运动、步长/跨度异常、垂直/前后漂移、旋转或边残差越界均为 F。
- 正式渲染保留主扫描段全部真实 pose nodes，数量为 2–160。不得固定抽稀为 32 个画布源，不得插值、重排或伪造 pose。
- 近重复节点只能在完整审计后成为零最终 owner；其真实 pose、边和一次 RGB remap 仍须保留。
- 每源只做一次全分辨率标定 inverse remap。中间源中央条带最多为输入宽度的 `20%`；首尾只可向扫描外侧扩展到校准图像边缘。
- 布局比例来自相邻 RGB 局部运动与真实 SE(3) 相机中心位移的稳健标量。它仅决定条带 x 布局，不是二维轨迹、单应矩阵、深度平面或 pose 修正。
- 画布和 aggregate working set 均不超过 `200 MP`；常驻 RGB 条带为 2–5 个，pose node 上限 160。

## 7. 风险走廊、局部几何与 owner

RGB Lab/梯度风险先约束 owner 和 MultiBand。只有跨 seam 的结构性 raw seed、显著边缘残差或整高 hard cut 指向几何问题时，才允许读取相邻走廊 aligned depth。

双向重投影和 z-buffer 使用 `max(20 mm, 2% × depth, 3σ_depth)` 深度一致门；没有可审计噪声 provenance 时 `σ_depth=0`。遮挡、disocclusion、深度边界、孔洞、透明/反光和强 RGB 结构都要扩展保护，并保持单一 RGB owner。

局部 inverse mesh 只能作用于双向可见、同层、未保护的安全背景，并必须通过训练/held-out 分离、前后向 flow、边界零位移、最大 `8 px` 位移、正 Jacobian、保护域零交集、直线和全分辨率误差审计。失败即回到单一 hard owner，不得半信任应用。

`local_apap_flow` 只允许在一个相邻 `96–160 px` 走廊、单一 owner 和安全同层背景中使用。正式默认：

```yaml
handoff_fallback_policy:
  publish_degraded: true
  local_apap_flow_enabled: false
  manual_review_for_grade_c: true
```

即使显式开启，也必须逐 pair 通过 40+ 对应、同层/保护域、held-out、FB flow、`16/32 px` 网格、零边界位移、最大 `8 px` 位移、正 Jacobian 和尺度 `0.80–1.25` 审计。前景仍是 owner-only。

GraphCut 只允许相邻真实源在互斥 corridor 中竞争，形成单调 owner。有效区每像素必须恰有一个 owner，无效区不能有 owner。MultiBand 只用于共同有效、低梯度、无风险的安全背景；每 pair 总带宽为 `clamp(floor(0.20 × 较窄 owner 宽度), 2, 8)`，最多 3 层，融合区不得与风险/保护域相交。禁止 DP、feather、全局金字塔、平均、全图模糊或补洞。

`foreground_deformation_experiment` 默认关闭且仅限独立诊断；不得进入正式 A/B/C/F、`report.json` 或 `delivery.json`。

## 8. A/B/C/F 与原子交付

结构检查先于质量分级：

| 等级 | 当前语义 |
| --- | --- |
| A | 输入、轨迹和最终渲染严格质量全部通过；handoff 为安全 anchor/owner；`delivery_state=published`。 |
| B | 严格质量通过，至少使用完整审计的 `flow_mesh`、显式启用的 `apap`，或 photometric pair 使用 `rgb_texture_consistent`；`delivery_state=published`。 |
| C | 会话、真实轨迹、一次 remap、owner 拓扑和 pair 审计结构完整，但严格质量未过或使用 `hard_cut_degraded`；`delivery_state=published_degraded`、`manual_review_required=true`。 |
| F | 会话、轨迹、remap、owner/MultiBand、资源限制、TSDF/GLB/Viewer 或原子发布任一结构项失败；不发布 `delivery.json`。 |

当前正式 schema：

- `report.json`: `gemini305-unified-central-strip/v12-r1`
- `delivery.json`: `gemini305-panorama-delivery/v12-r1`
- `render_transforms.json`: `unified-calibrated-central-strip/v1`
- `pixel_provenance.npz`: `owner-frame-id-r1`
- `failure.json`: `gemini305-panorama-failure/v2`

A/B/C 目录必须原子发布：

```text
panorama.jpg
panorama.png
pixel_provenance.npz
transforms.json
render_transforms.json
report.json
tsdf_mesh.glb
tsdf_mesh_mobile.glb
tsdf_mesh_viewer.html
delivery.json
```

`delivery.json` 最后写入。每次 `run()` 的第一项输出动作必须使旧 `delivery.json` 失效。普通异常与 F 都应清除正式/诊断产物并原子写 `failure.json`。没有有效 `delivery.json` 就没有正式交付；`quality_pass` 只是 `strict_quality_pass` 的兼容别名，不能单独表示已经发布。

`--diagnostic-force` 可绕过输入外观、odometry/pose 质量和最终图像质量阈值，但不能绕过严格会话、有限 SE(3)、必需边/连通性、有效 remap、owner/MultiBand 拓扑、资源上限或原子交付。诊断成功只写 `diagnostic_panorama.jpg` 和 `diagnostic_report.json`。

## 9. CLI 与验证导航

| CLI | 用途 |
| --- | --- |
| `g305-capture --photo-mode` | 正式主采集：低帧率软件触发同步 RGB-D 照片序列 |
| `g305-capture` | 次要连续流采集 |
| `g305-panorama` | 正式 unified RGB-D 全景 |
| `g305-orbslam3-trajectory` | 独立重新运行完整 ORB-SLAM3 并导出真实轨迹 |
| `g305-central-strip-diagnostic` | 隔离中央条带诊断 |
| `g305-geometry-pair-diagnostic` | 隔离相邻 geometry A/B 诊断 |
| `g305-foreground-deformation-diagnostic` | 默认关闭的前景局部变形诊断 |
| `generate-panorama-demo` | 生成严格合成 RGB-D 会话 |
| `unistitch-pair` | 历史双图诊断 |
| `unistitch-sequence` | `g305-panorama` 的弃用别名 |

修改后按影响范围运行测试：

- 采集：`test_photo_capture.py`、`test_capture_calibration.py`
- 会话：`test_session.py`、`test_v1_input_contract.py`
- 位姿/CUDA：`test_rgbd_odometry.py`、`test_orbslam3_bridge.py`、`test_export_orbslam3_trajectory.py`、`test_cuda_backend.py`
- unified renderer：`test_calibrated_rgb_pushbroom.py`、`test_geometry_assisted_local_warp.py`、`test_handoff_continuity.py`
- 发布：`test_sequence_delivery.py`、`test_sequence_integration.py`、`test_config.py`
- TSDF：`test_dense_fusion.py`

合成测试不等于实机验收。涉及相机、Open3D、CUDA、ORB-SLAM3 或性能的改动，交付说明必须分别注明单元/合成测试、真实 Open3D 边、真实完整 ORB-SLAM3、历史失败数据和现场速度验收状态。历史输出和旧 schema 不能作为当前 v12-r1 正式验收。
