# AGENTS.md

本文档供后续在 `D:\central_strip_Panoramic_Camera` 工作的开发代理阅读。开始修改前先阅读本文，再按需阅读 `README.md`、配置和相关源码。

## 1. 项目目标与正式架构

项目使用奥比中光 Gemini 305 采集同步 RGB-D 序列，并以 fail-closed 方式生成移动侧扫全景。默认工况是连续单向水平侧移、场景基本静止、最近物体约 `0.5 m`、最高速度约 `1.5 m/s`。普通用户只提供会话目录和输出目录，不调整曝光、步长、位姿、接缝或裁剪参数。

正式序列流程固定为：

```text
严格 RGB-D 会话、标定、曝光、清晰度与主扫描段
  → 自适应 RGB-D pose nodes
  → 每帧短基线 Open3D 相邻 RGB-D odometry（局部几何验证）
  → ORB-SLAM3 RGB-D 以完整短基线链求真实全局轨迹
  → 有限 4×4 camera_to_world SE(3)
  → 真实位姿校平的全分辨率 RGB 流式中央窄条（每源一次标定 inverse remap）
  → RGB 视差风险触发的相邻 RGB-D 双向可见性 / 深度分层 / 局部逆网格
  → 单调 hard owner / GraphCut 接缝（前景、透明或不可靠深度只取一个 RGB owner）
  → 仅安全白墙的窄带局部 MultiBand
  → 基于 valid mask 的最大内接矩形、质量门禁和原子交付
```

正式全景输出像素只能取自 RGB。aligned depth 可在 RGB 风险已触发的 `96–160 px` 相邻接缝走廊内正式参与双向重投影、z-buffer 可见性、深度分层、遮挡/透明保护、局部逆网格和 owner 决策；它绝不生成颜色、补洞、拟合全局参考平面、改写/插值真实 pose、构造全景深度或向 TSDF 回传结果。局部网格只修正一次 RGB inverse sampling，不是新的相机轨迹。不得导入或回退到 UniStitch、LightGlue、MAGSAC、Torch、3×3 单应矩阵、二维累计或时间/二维运动位姿插值。完成全景质量门禁后，`g305-panorama` 额外构建仅供浏览的 TSDF 网格并发布 `tsdf_mesh.glb` 与 `tsdf_mesh_viewer.html`；该产物绝不得向全景的条带、接缝、融合、裁剪或质量判定回传任何结果。ORB-SLAM3 仅提供真实 RGB-D 相机轨迹；未安装或未完整跟踪时应失败，不能回退为伪造位姿。`unistitch-pair` 暂时保留为可选历史双图诊断；`unistitch-sequence` 只是 `g305-panorama` 的弃用别名，运行同一 RGB-D 路径。`central_strip_plane_diagnostic` 只能由独立诊断命令通过 renderer callback 调用，绝不能成为正式 pushbroom backend、正式 CLI 选项或失败回退。

## 2. 开始工作前

1. 工作目录固定为 `D:\central_strip_Panoramic_Camera`。
2. 先运行 `git status --short`，保留用户或其他代理的所有改动。
3. 本机主环境是 `D:\Panoramic_Camera\.conda`；若本项目已创建本地 `.conda`，可使用本地环境。不要无故删除或重建任一环境。
4. 搜索优先用 `rg` / `rg --files`；修改文件使用补丁。
5. 不执行 `git reset --hard`、破坏性 checkout 或批量删除数据/输出。

常用验证：

```powershell
$G305Python = 'D:\Panoramic_Camera\.conda\python.exe'
& $G305Python -m pytest -q
ruff check src tests
& $G305Python -m compileall -q src tests
git diff --check
```

主环境（或项目内 `.conda`）中应安装 Open3D 0.19。Torch/Kornia/torchvision 只属于可选 `unistitch-diagnostic` extra，不是正式依赖。

## 3. 目录与模块职责

| 路径 | 职责 |
|---|---|
| `configs/demo.yaml` | 正式零调参采集、RGB-D 位姿验证、RGB pushbroom、接缝与融合安全默认值 |
| `configs/capture_640x480.yaml` | 低带宽诊断配置，不是正式默认 |
| `configs/capture_unrestricted_auto_exposure.yaml` | 无限 AE 与非交付诊断的一体化配置 |
| `capture_orbbec.py` | Gemini 305 同步采集、COLOR_STREAM 软件对齐、曝光限制、元数据和异步写盘 |
| `photo_capture.py` | 无预览 RGB-D 照片序列状态机、软件触发、外部 Trigger Out 门控、最快共同 profile 与逐帧落盘 |
| `session.py` | 严格 RGB-D 会话、标定、aligned depth、曝光/时间戳和毫米单位契约 |
| `quality.py` | 缩略图画质、视觉运动、主扫描段、pose-node 布局和基于真实 SE(3)/足迹的渲染源选择 |
| `rgbd_odometry.py` | Open3D 延迟导入、短基线 RGB-D odometry、边质量、SE(3) 与轨迹审计 |
| `orbslam3_bridge.py` | WSL ORB-SLAM3 RGB-D 调用、标定去畸变、深度比例适配和真实轨迹解析 |
| `calibrated_rgb_pushbroom.py` | 正式 RGB 流式窄条 renderer；真实 SE(3)、一次 RGB inverse remap、深度辅助局部逆网格、hard owner 与窄带 MultiBand |
| `geometry_assisted_local_warp.py` | 相邻 RGB-D 双向重投影、z-buffer/层分类、深度保护和受限 16/32 px 局部逆网格 |
| `rgbd_projection.py` | 仅限历史/独立诊断与回归的 RGB-D 投影模块，正式 renderer 不导入 |
| `dense_fusion.py` | TSDF 仅供正式交付后的独立 3D 展示；正式 RGB renderer 不导入且不读取其结果 |
| `central_strip.py` | 真实轨迹参考平面、扫描坐标、中央条带一次 remap 与诊断结果；只供独立诊断后端 |
| `central_strip_diagnostic.py` | 独立诊断 CLI；本文件是唯一导入 central-strip renderer 的入口 |
| `geometry_pair_diagnostic.py` | 独立的完整序列相邻接缝 RGB A/B 诊断 CLI；不读取历史位姿 sidecar |
| `render.py` | RGB 风险检测、历史深度 renderer、owner/MultiBand 工具和 valid-mask 裁剪 |
| `stitch_sequence.py` | 严格 RGB-D 位姿编排 + 正式 RGB pushbroom、报告、失败清理和原子交付 |
| `synthetic.py` | 带标定、aligned depth、已知 SE(3) 的合成 RGB-D 会话 |
| `stitch_pair.py` / `unistitch_adapter.py` | 可选历史双图诊断，不得进入正式序列 |
| `tests/` | 单元、合成、交付和零参数集成回归 |

CLI：

- `g305-capture`（连续流；`--photo-mode` 使用逐帧拍照实现低帧率序列）
- `g305-panorama`
- `g305-central-strip-diagnostic`（独立、仅参考平面声明的中央条带诊断；只发布两个诊断文件）
- `g305-geometry-pair-diagnostic`（独立、完整当前 ORB-SLAM3 链的相邻接缝 RGB A/B 诊断；只发布两个诊断文件）
- `unistitch-sequence`（弃用别名，同一 RGB-D 流程）
- `unistitch-pair`（可选历史诊断）
- `generate-panorama-demo`

Windows 直接调用 `g305-panorama` 前，必须确认 `(Get-Command g305-panorama).Source` 是主环境的 `Scripts\g305-panorama.exe`，且 `import panorama_demo` 解析到本工作区 `src\panorama_demo`；否则会调用旧全局安装并可能产生不合规的 TSDF/深度产物。

## 4. 严格 RGB-D 会话契约

正式输入只接受会话目录或其 `frames.csv`。每帧必须具备：

- RGB；
- `depth_aligned/` 下明确的 `aligned_depth_path`；
- 有限正数 `depth_scale_mm_per_unit`；
- 与 RGB/彩色内参完全同尺寸的 uint16 PNG；
- 有限有效彩色内参与畸变；
- 标定中的 color-target 对齐标记，或本项目 `panorama-demo-session/v1` 捕获器的 `software → COLOR_STREAM` provenance；
- 非负彩色时间戳和正数 `color_exposure` 元数据。

`raw_depth_path`、`depth_path` 或其它目录不能冒充 aligned depth。RGB 线性去畸变，深度最近邻；黑色 RGB 仍可为有效内容。项目内部深度与位姿平移始终为毫米，只有 Open3D 适配层临时转换为米。缺标定、深度、单位、对齐 provenance、曝光或时间戳是结构失败，诊断模式也不能绕过。

新采集的 `calibration.json` 必须写出明确 `depth_alignment` 标记。不要丢失 CSV 中的曝光、时间戳和深度比例。

照片模式也必须输出同一严格 RGB-D 会话，不能移植成 RGB-only 截图。`g305-capture --photo-mode` 固定使用 `SOFTWARE_TRIGGERING`、`Trigger Out Enable=true`、`frames_per_trigger=1` 和 `17000 µs` Trigger Out 延时；准备阶段只有在 SBU 物理输出门已回读为关闭时才允许最多 8 次有界内部触发预热，取得完整 RGB-D 后仍须让 gate 保持关闭至从最后一次内部触发起的完整迟到响应窗口结束，并确认队列为空。准备完成后以单次拍照循环形成低帧率序列：每个序列帧只调用一次正式 `device.trigger_capture()`，上一帧完整收取、对齐和落盘后才能触发下一帧，正式失败路径不得重触发。该模式不得显示视频，不增加人工限速；profile 自动选择限于配置分辨率下彩色与 Y16 深度共同支持的最高 FPS。照片会话成功关闭前 `formal_stitch_allowed=false`；严格加载器必须拒绝缺失/未知 manifest 或 `clean_shutdown!=true`，诊断模式也不能绕过。

## 5. 位姿、RGB 流式条带和选源边界

- Open3D 相邻 RGB-D 边始终必需并用于局部质量验证；ORB-SLAM3 只能输出完整 RGB-D 序列的真实 camera-to-world 轨迹，不能以特征匹配伪造缺失帧位姿。
- 每条边记录 source-to-reference SE(3)、收敛、真实 fitness/RMSE、正定 6×6 信息矩阵、深度有效率和失败原因。
- 正式配置只能等于或收紧默认 odometry/pose 门限；放宽只能进入诊断输出。
- 优化结果必须有限、旋转正交、行列式为 +1、图连通，并保持连续单向侧移。正式拒绝过大垂直/前后漂移、旋转、逆向和边残差。
- 正式渲染必须使用主扫描段的全部真实优化 pose nodes（至少两帧、最多 160 帧）；不得抽稀成 32 个全画布源、插值、重排或伪造中间位姿。
- 每个源只进行一次全分辨率、标定 inverse remap。中间源只保留其主点附近的窄条；首帧和末帧仅向扫描方向外侧扩展至各自校准 RGB 图像边缘，以保留端点场景。端点扩展仍须由单一 hard owner 和独立 valid mask 支撑；最终裁剪若丢弃完整有效的端点外扩列则失败，畸变校正导致的几何无效边缘列须审计后裁去。临时条带落盘，任一阶段内只保留相邻 2 个 RGB 条带（配置硬上限 5）。
- 深度辅助只作用于触发 RGB 风险的相邻走廊：按真实 `camera_to_world` 双向反投影/重投影、target z-buffer 与 `max(20 mm, 2%×depth, 3σ_depth)` 一致门分为同层、遮挡、disocclusion、深度边界和无效/透明区域。`20 mm` 与 `2%` 是固定项；没有可审计噪声标定 provenance 时 `σ_depth=0`，不得用配置扩大容差。后四类向外保护 `8–12 px`，只允许单一 RGB owner。层分类只依据该走廊的原始像素足迹：至少占局部安全支持 `10%` 的层才能定义远背景，最终只能保留一个占比至少 `50%`、跨名义 seam 的 4 连通背景分量。
- 只有双向同层、深度一致且 RGB 光流验证一致的连通域可拟合 `16/32 px` 局部逆网格。flow 安全应用域包含已接受的 held-out 像素，节点拟合域是排除 held-out 的严格训练子集；留出像素可以验证和应用已通过的场，不能参与拟合。所有强 RGB 结构（包括 flow 已接受者）均先成为 owner-only guard，不能连接或跨越网格；实际非恒等网格像素与 protected/non-same-layer 的交集必须为零。网格边界位移为零、保护样本逐点恒等、最大位移不超过 `8 px`、Jacobian 为正，raw 同层中心线、内部网格边和每格对角线的新增弯曲不得超过 `1 px`；Hough/forward-inverse 实际观察到的基线直线同样不得弯曲超过 `1 px`，无 solver-valid 直线只记录 `not_observed_no_solver_valid_line` 且不能跳过固有网格审计；held-out 强边缘、flow FB 和网格误差均以全分辨率像素审计，P95 `≤0.75 px`、最大 `≤2 px`、相对改善 `≥30%`、至少 30 个对应且至少一个 4-连通组件覆盖 4 个网格单元，否则回退为单一 owner 或失败。启动局部网格还要求膨胀前的边缘位移/梯度 raw seed 经 `3×3` close 后形成跨名义 seam 中线的 8 连通分量，并按 preview scale 等价至少 `72` 个全分辨率 seed 像素、`18` 行和 `26 px` 纵向跨度；Lab-only 风险只约束 RGB owner/MultiBand，不能单独触发深度或网格。
- 条带的 x 坐标按真实相机中心的单调 SE(3) 位移，并以相邻 RGB 局部运动/毫米的稳健标量换算；该标量只决定条带布局，绝不是二维位姿、单应矩阵或深度/平面代理。尺度不稳定时失败。
- 中间源的最大原始中央条带宽度为输入 RGB 宽度的 `20%`；放不下 hard-owner 区间时失败，而不是放宽中间条带。首尾帧只有向扫描外侧的校准半视场可超过此宽度，不能向内侧扩展或绕过 hard-owner、valid-mask、画布和 working-set 门禁。
- 全景输出只包含 RGB、独立 valid mask 和 hard-owner 审计；不得产生 `surface_depth_mm`、相机深度或前景 mask。交付目录可额外包含只读展示用的 `tsdf_mesh.glb` 和 `tsdf_mesh_viewer.html`，但网格不得影响全景。
- 画布与流式 aggregate working set 均不得超过 `200 MP`；诊断也不能提高这些硬限。pose nodes 最多 160。

## 6. RGB 风险、hard owner 与窄带融合

默认正式 backend 为 `calibrated_rgb_pushbroom`。RGB Lab/梯度风险始终先约束 owner/MultiBand；只有满足上条跨 seam 中线的结构性 raw seed，或边缘残差/整高 hard cut 明确指出几何问题时，才读取深度辅助。未触发时沿用纯 RGB 接缝。触发后只有已验证同层能得到局部逆网格，近景、把手、软管、遮挡、深度孔洞、透明/反光内容均为单一 RGB hard owner，不能进入 MultiBand。透明斜带等原始 RGB 内容不是“待修复的错位”，应保留其单源外观。亮度比例仅在共同有效、低梯度、低残差、未过曝/欠曝的白墙候选上以 trimmed median/Huber 估计，并在帧序列上平滑增益曲线。不得把无效去畸变黑边当作纹理。`graphcut_depth_constrained` 仅保留给历史回归；TSDF 仅可在全景门禁后用于独立 3D 展示，二者绝不能作为正式渲染或失败回退。

- 仅相邻扫描源在互斥 pair corridor 内竞争；
- 风险包括 Lab/梯度残差与其膨胀保护带；高风险像素及保护带只允许一个 RGB owner，绝不进入 MultiBand；
- 若组件级 owner 无法找到不穿过保护带的物理安全边界，只有一幅 RGB 条带完整覆盖有效走廊时才允许整走廊单源；此回退适用于 geometry 和纯 RGB 组件拓扑，不能因非 geometry pair 直接失败。否则 fail-closed。任何因这一决定不再贡献颜色的真实节点必须在审计中列出，所有真实 pose 仍参与轨迹、布局和一次 remap；
- GraphCut 在每个相邻条带重叠区寻找单调 hard owner；无法找到安全通道时允许报告 hard cut，绝不以透明重影掩盖；
- MultiBand 宽度为 `clamp(floor(0.20 × 较窄 owner 宽度), 2, 8)` 的总安全带宽，最多 3 层；融合区不得超过有效画布 `20%`，与 RGB 风险交集必须严格为零；
- GraphCut/MultiBand 异常不得回退 DP、Feather、平均、全图高斯模糊或补洞。

GraphCut 后无条件运行 owner 审计：有效区每像素恰好一个 owner、无效区无 owner，每对相邻源的安全融合带都由真实共同有效 RGB 支撑。结构门禁不受 `quality_gate` 或 `--diagnostic-force` 影响。

MultiBand 只能在 owner 验证之后运行。每条相邻 owner 边界使用独立局部 `cv2.detail_MultiBandBlender`，不能把全部源 feed 到一个全局金字塔。区外和风险保护带直接复制唯一 owner；每个局部 output mask 必须完整、无零权重 wedge、无越界写入。最终裁剪必须使用 valid mask 的 `largest_valid_rectangle()`，不得因 RGB 为黑而删除黑色软管等有效内容。

正式最终门限包括：融合区风险严格为 `0`、融合区占有效像素不超过 `20%`、曝光增益 `0.45–2.20`、裁剪高度至少 `85%`、画布宽度至少 `95%`。

## 7. 诊断与原子交付

正式曝光上限为 `800 µs`，输入拒绝上限固定为 `1200 µs`，设备原始 `color_exposure` 当前按 `100 µs/单位`。正式配置不能改变这一单位或放宽门限。

`--diagnostic-force` 可绕过输入绝对画质、正式 odometry/pose 质量和最终图像质量，但不能绕过：

- 标定、aligned depth、单位、曝光/时间元数据；
- 有限 SE(3)、正定信息矩阵、必需相邻边和图连通；
- 有效 RGB inverse remap、owner 拓扑、GraphCut/MultiBand 结构完整性；
- 画布/working-set 硬限；
- 原子交付语义。

诊断成功只写 `diagnostic_panorama.jpg` 和 `diagnostic_report.json`，绝不写正式文件或 `delivery.json`。中央条带诊断同样严格只写这两个文件，不写 `transforms.json`、TSDF mesh 或其它附属产物；其 ORB-SLAM3 staging 必须使用系统临时目录，成功输出目录不得留下 `.orbslam3_rgbd`。其参考平面必须是唯一主导、跨扫描具备足够标定图像面积支持的实测平面；竞争平面、面积不足或结构残差过大是结构失败。更严格的平面残差只令诊断 `strip_quality_pass=false`，不会放宽为正式交付。`g305-geometry-pair-diagnostic` 也严格只发布这两个文件：它必须先以本次完整扫描运行 Open3D 和 ORB-SLAM3，再以全部真实 pose nodes 分别渲染关闭 geometry 的 baseline 与正常 candidate，并仅裁出请求相邻 pair 的共同 RGB 走廊（左 baseline、右 candidate；列映射、两套 gain 统计与 pair audit 写入报告）。它不得接受 `--render-frame-ids`、`--diagnostic-force`、Open3D-only 轨迹、保存的 `render_transforms.json` 或历史 gain；不获准的网格必须如实显示为 hard-owner candidate，而不是强行变形。

每次 `run()` 的第一项文件动作必须使旧 `delivery.json` 失效。正式文件先写隐藏 pending，再 `os.replace`；`delivery.json` 最后写。普通异常无论从 CLI 还是直接调用 `run()`，都应清除正式/诊断产物并原子写 `failure.json`。强制终止可能来不及写失败报告，但没有有效 `delivery.json` 始终代表失败。

## 8. 测试导航

| 测试 | 重点 |
|---|---|
| `test_capture_calibration.py` | 相机属性、AE 上限、对齐 provenance、标定 |
| `test_photo_capture.py` | SOFTWARE_TRIGGERING、SBU gate-off 预热/静默、最快共同 RGB-D FPS、逐帧单次正式触发、失败停止、会话落盘与恢复 |
| `test_session.py` | 严格会话、曝光/时间、aligned/raw、尺寸和毫米单位 |
| `test_quality.py` | 画质、运动、主段、pose nodes、SE(3)/足迹选源 |
| `test_rgbd_odometry.py` | 单位、边、信息矩阵、图连接、SE(3)、轨迹、延迟导入 |
| `test_calibrated_rgb_pushbroom.py` | 全真实帧流式窄条、RGB 像素来源、深度辅助接缝、风险 hard owner、窄带融合、valid-mask 裁剪与资源门禁 |
| `test_geometry_assisted_local_warp.py` | 双向 z-buffer/层分类、遮挡/孔洞/透明保护、局部网格、held-out 与 flow 门禁 |
| `test_geometry_pair_diagnostic.py` | 完整源链 baseline/candidate A/B、共同走廊 RGB 裁切、无历史位姿回退与诊断原子交付 |
| `test_rgbd_projection.py` | 视差、z-buffer、断层、空洞、黑色内容、资源限制 |
| `test_render.py` | RGB 风险、历史深度 GraphCut、owner、逐 pair MultiBand、裁剪与风险 |
| `test_sequence_delivery.py` | 首先失效 delivery、各阶段失败、诊断隔离 |
| `test_sequence_integration.py` | fake RGB-D backend 的零参数完整交付，正式路径不导入旧模型 |
| `test_config.py` | RGB-D 默认值与正式安全包络 |
| `test_synthetic.py` | 合成 RGB-D 场景和已知轨迹 |

合成测试通过不等于实机验收。相机/Open3D/性能改动必须分别说明纯测试、真实 Open3D 边、历史失败数据和现场速度验收的状态。

## 9. 已知真实数据和现场验收

`data/captures/run_20260713_184519` 是无限 AE 诊断会话，`color_exposure=301`，约 `30.1 ms`。2026-07-13 用 Open3D 0.19 和 `--diagnostic-force` 复测：12 条必需相邻边均收敛，fitness `0.613–0.856`，RMSE `17.4–26.2 mm`；随后因高风险 RGB-D 带横断完整相邻 pair corridor 而正确失败。输出只含 `failure.json`，不能作为成功样本或通过放宽门限强行出图。

`data/captures/run_20260714_132427_262` 是真实轨迹验收样本：主扫描段 101 帧均被 ORB-SLAM3 RGB-D 跟踪；100 条短基线 Open3D 边的 fitness 为 `0.939–0.998`、RMSE 为 `7.7–11.5 mm`。2026-07-14 的 TSDF 正式交付证明该会话的输入和真实轨迹可通过旧路径；RGB pushbroom 仍须单独视觉验收其窄条、硬 owner、白墙亮度和裁剪结果，不能将旧 TSDF 成功宣称为新 renderer 成功。

2026-07-15 已以真实 Open3D 边和完整 ORB-SLAM3 RGB-D 轨迹对该会话运行正式 `calibrated_rgb_pushbroom`：101 个真实源各一次 RGB remap，输出 `1729×797`，裁剪高度 `99.625%`、融合区 `11.48%`、融合风险像素 `0`、峰值驻留条带 `2`、亮度 gain `0.888–1.154`，并发布 `delivery.json`。这是该单一会话的新 renderer 实机验收，仍不代表所有场景或速度都必然成功。

`outputs/greenhouse_sequence_optimized/` 是 2026-07-16 留存的历史 CLI 输出：101 个真实源，`2978×782`，裁剪高度 `97.75%`、融合区 `1.264%`、融合风险/owner 边界风险/受保护组件拆分均为 `0`、峰值驻留条带 `2`、亮度 gain `0.755–1.364`。其 JSON sidecar 是早于当前 geometry-assist 审计的 `/v2`，不含本版本要求的深度辅助网格字段；只能证明当时的旧路径/真实轨迹，不得当作当前 `/v7`、`/v8`、`/v8` schema 的正式验收。当前 Windows 主机恢复 WSL/ORB-SLAM3 后须重新执行 CLI 才能发布新 `delivery.json`。

2026-07-16 的 `outputs/greenhouse_geometry_assisted_direct_20260716_v2/diagnostic_panorama.jpg` 是早于当前 raw-structural-trigger、实际 RGB 直线门和通用组件 owner 回退的历史直接诊断，不能作为当前验收。2026-07-17 当前代码以同一已验证 101-node ORB-SLAM3 轨迹做只读内存回放（不写文件、复用历史 gain、关闭最终画质门）得到 `2978×782`：24 对结构性风险触发对均严格 hard-owner 回退，未接受局部网格。48–49 的双向深度残差 P95 约为 `0.46/0.47` 个深度容差，但深度边保护与层连通性不足，正确保持单源而不拉扯灭火器/软管。该回放不可视为新交付：它复用历史 gain，且当前 Windows 主机没有可供 ORB-SLAM3 bridge 使用的 WSL Linux 发行版；正式 CLI 必须在可用 bridge 上以当前 gain/门禁重新运行并发布新 `delivery.json`。

旧 `greenhouse_trial/run_20260711_213054` 同样只适合输入质量应拒绝回归。源帧拖影和已丢纹理无法靠锐化或融合恢复。

新数据至少验收静止、`0.5`、`1.0`、`1.5 m/s`，检查队列丢帧、RGB-D 同步、曝光、深度有效率/单位、pose 残差、0.5 m 近景重影、GraphCut/MultiBand、裁剪四边及最终 `delivery.json`。在这些现场样本完成前，不得宣称所有物理环境必然成功。

## 10. 交付前检查

- [ ] `git status --short` 无意外文件或覆盖用户改动；
- [ ] 定向和完整 pytest 通过；
- [ ] Ruff、compileall、`git diff --check` 通过；
- [ ] 默认 `g305-panorama INPUT --output OUTPUT` 无算法参数；
- [ ] RGB 全景路径无 UniStitch/Torch/LightGlue/MAGSAC、TSDF、RGB-D projection 整幅投影或 central-strip plane renderer 导入/回退；仅允许受审计的相邻 RGB-D 局部可见性/逆网格，TSDF 仅在其后作为独立展示导出；
- [ ] 失败路径先失效旧 delivery 并写 `failure.json`；
- [ ] 成功路径最后发布 `delivery.json`；
- [ ] README、AGENTS、配置、CLI、依赖和测试一致；
- [ ] 明确区分合成、历史失败、真实 Open3D 和现场相机验收。
