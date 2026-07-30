# AGENTS.md

本文件是 `D:\central_strip_Panoramic_Camera` 的开发约束。修改前先阅读本文件，再按需阅读 `README.md`、`configs/demo.yaml`、关联实现和测试。若文档与当前可执行源码、默认配置或测试不一致，以后者为准，并同时更新文档。

## 1. 项目与正式架构

项目采集奥比中光 Gemini 305 的同步、标定且对齐到彩色坐标系的 RGB-D 会话，生成 fail-closed 的移动侧扫全景。默认工况是连续单向水平侧移、场景基本静止、最近物体约 `0.5 m`、速度最高约 `1.5 m/s`。普通用户仅提供会话和输出目录；不得暴露曝光、位姿、条带、接缝或裁剪的正式算法调参。

当前正式入口为 `g305-panorama`，默认路径是：

```text
严格同步 RGB-D 会话
  → Open3D 相邻短基线 RGB-D odometry（每条边）
  → ORB-SLAM3 RGB-D（完整序列真实 camera_to_world 链）
  → 有限、连续、单向的 SE(3) 轨迹审计
  → unified_calibrated_central_strip/v1
     （每源一次标定 RGB inverse remap、单一画布/valid mask/owner map）
  → 邻接风险走廊的 RGB-D 可见性与受限局部 inverse sampling 审计
  → 单调 hard owner / 安全窄带局部 MultiBand
  → 严格质量与 A/B/C/F 发布判定
  → 独立、只读的 TSDF GLB + Viewer
  → 原子发布 delivery.json
```

正式 renderer 的不变量：

- 全景像素仅能取自原始 RGB 的一次标定 inverse remap；黑色 RGB 仍是有效内容。
- 完整 ORB-SLAM3 RGB-D `camera_to_world` 链是唯一全局轨迹。Open3D 边只做短基线几何验证；二者均不得伪造、插值、重排或用特征匹配替代缺失 pose。
- `unified_content_mode=true` 时只有一个 calibrated central-strip renderer、一个目标坐标域、一个 valid mask 和一个严格 owner map。旧 metric mosaic、inspection multiview、前景锚点或任何后渲染 overlay 均不得进入正式 RGB 输出。
- aligned depth 只能在已触发的相邻 `96–160 px` 风险走廊做双向重投影、z-buffer 可见性、深度分层、遮挡/透明保护和受限局部逆采样审计；不得生成颜色、补洞、构造全景深度、拟合全局面、改 pose 或向全景回传 TSDF 结果。
- 禁止把 UniStitch、LightGlue、MAGSAC、Torch、全局/pose/全景级 `3×3` 单应、二维累计位姿、全局 flow 或时间/二维位姿插值带入正式路径。
- TSDF 只在 RGB 全景的结构与质量判定之后生成展示附件；它绝不影响 RGB 条带、接缝、融合、裁剪、位姿或等级。

`unistitch-sequence` 只是弃用别名，运行同一 RGB-D 流程；`unistitch-pair` 是历史双图诊断。`g305-central-strip-diagnostic`、`g305-geometry-pair-diagnostic`、`g305-foreground-deformation-diagnostic` 均是隔离诊断入口，不能成为 `g305-panorama` 的 backend、选项或失败回退。

## 2. 开始工作与验证

1. 工作目录固定为 `D:\central_strip_Panoramic_Camera`。
2. 首先运行 `git status --short`；保留用户和其他代理的全部改动及输出目录。
3. 主环境为 `D:\Panoramic_Camera\.conda`。本项目已有 `.conda` 时可使用它；不得无故删除或重建环境。
4. 搜索先用 `rg` 或 `rg --files`，编辑使用补丁。不得 `git reset --hard`、破坏性 checkout 或批量删除采集/输出数据。

常用检查：

```powershell
$G305Python = 'D:\Panoramic_Camera\.conda\python.exe'
& $G305Python -m pytest -q
ruff check src tests
& $G305Python -m compileall -q src tests
git diff --check
```

Open3D `0.19` 是主环境依赖。Torch/Kornia/torchvision 仅属于 `unistitch-diagnostic`，不是正式依赖。若涉及 CUDA，使用 `G305_CUDA=prefer|auto|off|required` 的既有审计路径；`required` 不能把尚未实现 CUDA 的 CPU 算法伪装成 GPU。

## 3. 代码边界

| 路径 | 职责 |
| --- | --- |
| `configs/demo.yaml` | 正式零调参默认值与安全包络 |
| `capture_orbbec.py` / `photo_capture.py` | 同步采集、软件对齐、曝光/触发、严格会话落盘 |
| `session.py` | RGB-D 会话、标定、aligned depth、单位/时间/曝光契约 |
| `quality.py` | 画质、主扫描段、pose-node 与渲染源选择 |
| `rgbd_odometry.py` | Open3D 相邻短基线 RGB-D 边和轨迹审计 |
| `orbslam3_bridge.py` | WSL ORB-SLAM3 RGB-D staging、执行和真实轨迹解析 |
| `calibrated_rgb_pushbroom.py` | 唯一正式 unified central-strip RGB renderer |
| `geometry_assisted_local_warp.py` | 邻接 RGB-D 双向可见性、保护域与局部 inverse mesh |
| `handoff_continuity.py` / `local_apap_flow.py` | handoff 审计及默认关闭的局部候选 |
| `dense_fusion.py` | 仅供交付后展示的 TSDF GLB/Viewer |
| `stitch_sequence.py` | 正式编排、v12-r1 报告、原子发布与失败清理 |
| `*_diagnostic.py`、`central_strip.py`、`rgbd_projection.py` | 仅独立诊断或历史回归，不能由正式 renderer 导入/回退 |
| `tests/` | 契约、几何、渲染、交付和集成回归 |

## 4. 严格 RGB-D 会话

正式输入只能是会话目录或其 `frames.csv`。每帧必须有：

- RGB；
- `depth_aligned/` 内的 `aligned_depth_path`，且为与 RGB 同尺寸的 `uint16 PNG`；
- 有限正数 `depth_scale_mm_per_unit`；
- 有限的彩色内参与畸变；
- color-target 对齐标记，或本项目采集器的 `software → COLOR_STREAM` provenance；
- 非负彩色时间戳和正数 `color_exposure`。

`raw_depth_path`、`depth_path` 或其它目录不能冒充 aligned depth。内部深度和位姿平移一律以毫米表示，只有 Open3D 适配层可临时转换为米。缺少标定、aligned depth、单位、对齐 provenance、时间戳或曝光，永远是结构失败，诊断模式也不能绕过。

连续采集使用 `color_ae_max_exposure_us=800`；输入绝对上限为 `1200 µs`，设备 `color_exposure` 单位固定为 `100 µs/单位`。照片模式必须仍产出同一严格 RGB-D 会话：使用软件触发、单帧触发、gate-off 有界预热与迟到响应静默窗口；每个正式帧只触发一次，前一帧完整接收/对齐/落盘前不得触发下一帧。正常关闭前 `formal_stitch_allowed=false`，严格加载器必须要求受支持 manifest、`clean_shutdown=true`。

## 5. 位姿、布局和 RGB 渲染

- 每条相邻边必须保留 source-to-reference SE(3)、收敛、fitness、RMSE、正定 `6×6` information matrix、有效深度率和失败原因。
- ORB-SLAM3 必须完整跟踪正式序列；缺失真实 pose、非有限或非刚体 SE(3)、不连通、逆向/不连续运动、步长/跨度、垂直/前后漂移、旋转或边残差越界均为 F。
- 正式渲染保留主扫描段全部真实 pose nodes（2–160）；不能把它们抽稀为固定画布源。近重复节点最多只能被抑制为零最终 owner，仍须实际 remap、保留 pose 并完整审计。
- 每源全分辨率 RGB 只进行一次标定 inverse remap。中间源最多使用输入宽度 `20%` 的中央条带；首尾仅可向扫描方向外侧扩展到校准图像边缘。有效端点列不能被最终裁剪静默丢弃。
- 画布和 aggregate working set 均不得超过 `200 MP`；常驻 RGB 条带硬上限为 5，pose nodes 上限 160。
- 布局标量来自相邻 RGB 局部运动与已审计 SE(3) 相机中心位移的稳健比例；它只决定条带位置，不是二维 pose 或平面代理。比例不稳定即 F。

## 6. 风险、局部几何与 owner

RGB Lab/梯度风险先约束 owner/MultiBand；只有跨名义 seam 的结构性 raw seed、边缘残差或整高 hard cut 指向几何问题时才允许读取深度。每个风险走廊中，双向重投影与 z-buffer 使用 `max(20 mm, 2% × depth, 3σ_depth)` 一致门；没有噪声 provenance 时 `σ_depth=0`。遮挡、disocclusion、深度边界、孔洞、透明/反光和强 RGB 结构外扩 `8–12 px` 后均为单一 RGB owner。

局部 inverse mesh 仅可用于双向可见、同层、未保护的安全背景，网格为 `16/32 px`，且必须通过 held-out、前后向 flow、正 Jacobian、零边界位移、最大 `8 px` 位移、保护域零交集、直线和误差审计。无法通过即 hard owner；不得半信任应用。

`local_apap_flow` 进一步限制在单个相邻 `96–160 px` 走廊和单一 RGB owner。正式默认 `handoff_fallback_policy.local_apap_flow_enabled=false`；只有显式开启且逐项通过同层、保护域、40+ 对应、held-out、FB flow、Jacobian、尺度 `0.80–1.25`、位移和边界审计后才可使用。前景实例始终 owner-only。`foreground_deformation_experiment` 默认关闭且仅诊断，不能写入正式 A/B/C/F、`report.json` 或 `delivery.json`。

GraphCut 仅让相邻源在互斥 corridor 竞争并形成单调 owner。每个有效像素恰有一个 owner，无效像素没有 owner。安全、低梯度、共同有效的白墙才允许逐 pair 局部 MultiBand：总带宽为 `clamp(floor(0.20 × 较窄 owner 宽度), 2, 8)`，最多三层；融合区不得超过有效画布 `20%`，也不得与风险/保护域相交。禁止 DP、feather、全图平均、全局金字塔、全图模糊或补洞。

## 7. A/B/C/F 与原子交付

正式 policy 必须保持：

```yaml
handoff_fallback_policy:
  publish_degraded: true
  local_apap_flow_enabled: false
  manual_review_for_grade_c: true
```

| 等级 | 语义 |
| --- | --- |
| A | 全部严格质量通过，handoff 都是安全 anchor/owner；`delivery_state=published`。 |
| B | 严格质量通过，至少有已审计 `flow_mesh` 或显式开启并完整审计的 `apap`；`delivery_state=published`。 |
| C | 会话、真实轨迹、一次 remap、owner 拓扑和 pair 审计完整，但严格质量未通过或使用完整审计的 hard cut；`delivery_state=published_degraded` 且必须 `manual_review_required=true`。 |
| F | 会话、轨迹、remap、owner/MultiBand 拓扑、资源限制、TSDF/GLB/Viewer 或原子发布任一结构条件失败；仅保留 `failure.json`。 |

当前正式 schema：`report.json` 为 `gemini305-unified-central-strip/v12-r1`，`delivery.json` 为 `gemini305-panorama-delivery/v12-r1`。二者必须包含 `delivery_state`、`strict_quality_pass`、`quality_grade`、`handoff_fallback_summary`、`handoff_outcomes`、`manual_review_required` 与 `tsdf_visualization`；`quality_pass` 只是严格质量兼容别名，不能单独表示已发布。

A/B/C 成功目录原子发布：

```text
panorama.jpg / panorama.png
pixel_provenance.npz
transforms.json / render_transforms.json
report.json / delivery.json
tsdf_mesh.glb / tsdf_mesh_mobile.glb / tsdf_mesh_viewer.html
```

`delivery.json` 必须最后写入。每次 `run()` 的第一个文件动作必须使旧 `delivery.json` 失效；任何普通异常或 F 都清除正式/诊断产物并原子写 `failure.json`。无有效 `delivery.json` 即未发布。

`--diagnostic-force` 只能绕过输入外观、正式 odometry/pose 质量和最终图像质量阈值；不能绕过会话契约、有限 SE(3)、必需边/连通性、有效 remap、owner/MultiBand 拓扑、资源限制或原子交付。诊断成功严格只写 `diagnostic_panorama.jpg` 和 `diagnostic_report.json`。

## 8. 实机与回归

- 合成测试不等于实机验收。涉及相机、Open3D、CUDA、性能或 ORB-SLAM3 的修改，需分别说明单元测试、真实 Open3D 边、历史失败数据和现场速度验收状态。
- `data/captures/run_20260714_132427_262` 曾验证完整 ORB-SLAM3 与 Open3D 轨迹；历史输出或旧 schema 不能作为当前 v12-r1 的正式验收。
- 修改渲染、局部几何、发布或清理逻辑时，重点运行 `test_calibrated_rgb_pushbroom.py`、`test_geometry_assisted_local_warp.py`、`test_handoff_continuity.py`、`test_sequence_delivery.py` 和 `test_sequence_integration.py` 的相关测试。
