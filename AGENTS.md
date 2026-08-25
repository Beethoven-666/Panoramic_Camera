# AGENTS.md

本文档是 `D:\central_strip_Panoramic_Camera` 的开发约束。开始工作前先阅读本文件，再按需查看 `README.md`、`configs/demo.yaml`、相关源码和测试。若说明与当前可执行源码、默认配置或测试不一致，以源码、默认配置和测试为准，并在同一改动中修正文档。

## 0. 范围约束

以下限制约束的是**提出什么修法**，不约束**寻找什么问题**。

凡是这里真的有问题，都要报告——包括听起来罕见但本项目确实会产生的情况。然后把修法收在范围内：

1. 这不是一篇安全攻防论文。可以校验，禁止过度防御。除非本项目另有说明，默认操作者是自己机器上的合作者；如果它真有对手，它会写明，以那个范围为准。
2. 不要添加哈希、校验和或指纹，除非它替代了一个实质上更昂贵的操作，并且结果会改变下一步做什么。
3. 禁止防御性脚手架：不为这里不会发生的情况添加 feature flag、迁移框架、兼容层或包装层。
4. 禁止钻牛角尖：冷门编码、符号链接竞态、RTL 文本和毫秒级竞态一律不在范围内，除非该情况经由本项目**受支持的用法**可达——它的文档输入、公开接口或真实数据。可达即可，不需要复现；但“理论上可以构造”不算。
5. 需要判断时就作出判断，不要用评分表、检查清单或对已经定论的内容重复校验来代替判断。
6. 以上限制不覆盖用户、本项目约定或更高优先级规则明确要求的安全、迁移、校验与审阅。那些是被要求的工作本身，不算范围蔓延。

已经见过的形状仅用于校准，不是检查清单；一个真实问题不会因为“长得像其中一条”就被驳回：

- H：为了比较两个表格的差异，给每一行计算哈希，而直接比较单元格就能回答。
- H：写下一批没有任何代码会读取的校验和文件。
- E：给一个没有用户、没有部署的应用做账号安全加固。
- R：用一整夜反复审计自己的补丁，而功能仍未完成。
- R：一个对任何提交都给出不通过结论的审阅者。
- O：一层守卫的理由只是上一层守卫，而不是实际需求。

另有两种看起来相似、但不属于过度防御的情况；这些问题应当报告：

- ✓ 使用摘要比较来跳过重新读取一个已经拥有的大文件。
- ✓ 本项目自己的文档示例会产生的那种“听起来罕见”的输入。

运行任何检查之前，先回答：这次检查会检测出什么具体失败？如果失败真的发生，下一步会因此做出什么不同处理？答不上来就不要运行。

正确的内容要明确说正确，不要为了交差制造问题。

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

### 1.2 独立连续视频全景产品

照片模式的 `g305-panorama` 契约保持不变。连续 RGB-D 视频通过独立入口
`g305-video-panorama` 或 `g305-video-live` 处理，绝不能传给 `g305-panorama`，也不能复用
照片的 `delivery.json`。公共入口只能读取
`configs/video_algorithms/s013_visual_continuity_v11_production.lock.json`，正式身份固定为：

- `algorithm_id=S013_output_first_progressive_dense_central_slit_v4_cuda_m63_visual_continuity_v11`
- `implementation_id=s013_m5_disjoint_seam_m63_pair_guard_v11`
- 原始 candidate canonical SHA-256 为
  `a03f443aaf72463afc1f06507fe2bfc888d9211f4c4a62634bbfb29fa0f2dab0`
- production config canonical SHA-256 为
  `3a49f8cbaa622592eff41b2a9e34cb518c6067d74d15574cf01faeea0cc8fafc`
- `role=production`、`allow_baseline_fallback=false`、
  `execution_backend=s013_v11_production_cuda`

原始 candidate 不得原地修改。production config/lock 必须保留精确 algorithm 与
implementation identity，记录晋级 provenance，并通过 pixel contract 等价校验。公共二维入口
必须在 legacy renderer 之前直接路由到该 V11；半匹配身份、错误 config SHA、允许 fallback、
ORB trajectory 参数或任何试图进入旧 production renderer 的 V11 请求都必须 fail-closed。

正式连续视频路径如下：

```text
连续 RGB-D 采集
  → 424 px 增量 RGB 运动分析与非正式预览
  → 停采时停止预览、drain writer、冻结 committed ledger handoff
  → S013 V11 ignore-pose authority 在内存完整执行 P0 → P1 → P2 → P3
  → 只发布正式 P3、provenance、report、timing
  → video_delivery.json 最后原子发布
  → 二维资源释放后，才可启动独立 post-3D 子进程
  → 子进程内运行 ORB-SLAM3、Open3D TSDF、GLB 与离线 Viewer
```

- 只接受 `continuous_rgbd_video_auto` 与
  `continuous_rgbd_video_fixed_exposure`；v2 会话必须有
  `product_eligibility.photo_panorama=false`、`video_panorama=true` 和干净关闭证据。
- 正式采集和二维期间 `OnlineORBTracker` 构造、ORB-SLAM3 进程、ORB frame submit、Open3D
  import、TSDF 调用和三维进程调用次数必须全部为 0。普通连续采集默认也不构造在线 ORB；
  `--diagnostic-online-orbslam3` 只属于诊断，并被 `g305-video-live` 拒绝。
- 在线分析必须维护增量状态，不得循环调用整段 `run_s13_fast_pipeline`。预览默认门槛为同一
  稳定运动段持续 `0.8 s`、累计前进 `32` 个 424 宽分析像素、方向一致率 `≥0.85`、可靠运动
  占比 `≥0.75`、至少 5 个源候选、writer queue `≤25%` 且 0 drops。方向反转、不可靠运动或
  超过 `0.4 s` 的 gap 重置稳定段。
- `live_preview.jpg` / `live_preview_state.json` 必须标记
  `non_authoritative_live_preview`。预览使用 latest-only 队列；预览失败只写
  `live_preview_failure.json` 并禁用后续预览，不得终止采集或正式二维。
- 停采必须先停止并 join 预览，再完成 writer drain。live handoff 当前只允许
  `reuse_level=validated_inputs_only`，`pair_evidence_reused_count=0`；在线 gray/motion、pair 或
  M6 evidence 不得冒充最终 authority。正式二维仍从 committed 会话重新计算完整 V11。
- 正式 production 必须执行完整 P0–P3 像素链，但正常首图模式不得落盘 P0/P1/P2 stage PNG。
  live 与 offline 对同一会话及锁定配置的 `video_panorama.png` 必须字节一致；P0–P3 内存像素
  evidence 逐阶段比较，首个差异即停止扩大复用范围并定位该阶段。
- 2-D 主交付为 `video_panorama.jpg`、`video_panorama.png`、
  `video_pixel_provenance.npz`、`video_report.json`、`video_timing.json` 和最后原子写入的
  `video_delivery.json`。只有 delivery 已发布且二维资源释放后，才可创建三维进程。
- post-3D 从连续会话选择约 8 FPS 的真实 RGB-D 帧，只在独立子进程中运行 ORB-SLAM3；三维
  不做预览。所有输出位于二维目录下的 `3d/`：trajectory、trajectory lock、desktop/mobile
  GLB、离线 Viewer、timing 和 `video_3d_delivery.json`。任何 ORB、TSDF、cleanup 或 GLB
  失败只写 `3d/video_3d_failure.json`，不得修改、删除或撤销二维结果。
- 视频 Viewer 必须离线可用，不得依赖 CDN。GLB 的节点将 Open3D `+Y-down` 转为 glTF
  `+Y-up`；自定义 Viewer 必须同样应用 180° X 轴转换。

### 1.3 视频视觉 renderer（用户授权的正式例外）

本节只适用于独立的 `g305-video-panorama` 和 `g305-video-live`；照片 `g305-panorama` 及其 unified
renderer 继续受 1.1 与第 7 节的全部限制。当前正式视频 renderer 只有锁定的 S013 V11，
其二维 authority 明确使用 RGB motion/ignore-pose；不得运行或消费 ORB pose，也不得把二维
motion 写成 SE(3)。它按冻结配置执行真实 RGB source selection、P0 hard owner、P1 vertical、
P2 geometry/seam transaction 和 P3 M6.3 photometric/blend/repair。每个有效像素必须保持可追溯
owner，禁止虚拟 RGB source、外部补色、全景级 flow、全局单应或从三维向二维回传结果。

正式报告必须包含 source、schedule、owner、逐 pair seam/M5 transaction、C2E、M6.3、
photometric candidate audit、B0/B1 和各阶段 pixel evidence。主结果按
`maximum_post_seconds=60` 记录停采到 P3 memory/发布的 SLA；超时可以结构化发布，但必须如实
降级并要求人工复核。candidate/audit 模式可以落盘 P0–P3 用于等价验证，但不得改变正式首图
输出策略。

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
| `capture_orbbec.py` | CLI、连续采集（自动或 `--video-exposure-us` 固定曝光）及照片模式路由、同步对齐和会话写盘基础设施 |
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
| `video_session.py` / `video_scan_segment.py` / `video_source_selection.py` / `video_motion_resampler.py` | 连续视频会话资格、RGB motion、扫描段与真实源选择基础设施 |
| `video_live.py` / `video_s13_live.py` | 正式连续采集编排、424 px 增量分析、非正式 latest-only 预览、停采 handoff |
| `video_s13_contract.py` / `video_s13_promotion.py` / `video_s13_production.py` | 精确 V11 身份与晋级校验、production ignore-pose authority、P3 正式发布 |
| `video_pipeline.py` / `video_delivery.py` / `video_performance.py` | V11 优先路由、二维原子发布与停采到 P3 SLA 审计 |
| `video_online_state.py` / `video_online_orb.py` | 旧在线状态兼容与诊断 ORB；不得进入正式 live 采集或 V11 二维 authority |
| `video_3d_launcher.py` / `video_3d_postprocess.py` / `video_3d.py` | 二维交付后独立进程启动、真实 ORB 轨迹、TSDF/GLB/离线 Viewer 与失败隔离 |
| `video_panorama.py` / `video_visual_renderer.py` | 公共视频 CLI 编排与 legacy/研发 renderer；精确 V11 production 不得经过 legacy renderer |
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

- 无预览，使用 `SOFTWARE_TRIGGERING`、`frames_per_trigger=1` 和 `Trigger Out Enable=true`。`trigger_to_image_delay_us` 默认 `8000 µs`，Trigger Out 延时默认 `7000 µs`；二者可由命令行覆盖，但必须为非负整数并能被所选帧周期安全容纳。
- 彩色格式不得由用户指定，必须按 `RGB → BGR → YUYV → MJPG` 依次尝试并选择指定分辨率/FPS 下第一个 SDK 支持的格式；深度固定为单通道 `uint16 Y16`。FPS 可由命令行精确选择；未指定时选择彩色/深度共同支持且能容纳两种延时的最高 FPS。不得回退分辨率或帧率；SDK 没有精确共同 profile 时必须失败并报告实际 profile。
- 准备阶段先关闭并回读物理输出 gate，最多执行 8 次有界内部预热触发。预热期间 gate 始终关闭。
- 获得完整预热 RGB-D 后，gate 继续关闭，直到最后一次内部触发的完整迟到响应窗口结束并确认队列为空。
- 每个正式序列帧只调用一次 `device.trigger_capture()`；每帧返回后必须重新读取完整同步配置，确认 `trigger_to_image_delay_us` 与 Trigger Out 延时仍为设定值；上一帧必须完整收取、COLOR_STREAM 对齐、写盘并确认成功后才能触发下一帧。失败路径不得重触发。
- 正式彩色曝光固定不超过 `800 µs`，设备 metadata 单位为 `100 µs/单位`。
- 会话打开期间 `formal_stitch_allowed=false`。只有相机/写盘资源安全关闭、无采集或写盘错误时，最终 manifest 才可写 `clean_shutdown=true`、`formal_stitch_allowed=true`。

连续流采集与照片模式隔离：预热期间使用自动曝光、自动增益和自动白平衡，预热后锁定并
逐帧验证曝光、增益和白平衡；自动曝光元数据超过 `800 µs` 时固定回退到 `800 µs` 并保留
最后一个完整预热 FrameSet 的有效自动增益，回退和锁定过渡帧不写盘。会话写入
`capture_mode=continuous_rgbd_video_auto`、`diagnostic_only=true` 与
`formal_stitch_allowed=false`。可用 `--video-exposure-us` 采集固定曝光视频，写入
`continuous_rgbd_video_fixed_exposure`；二者都不能用作 `g305-panorama` 输入。安全关闭且
无写盘错误的 v2 视频会话会写入仅供视频产品使用的 `product_eligibility`，可传给
`g305-video-panorama`；RGB-only 截图仍不能替代 RGB-D 会话。视频的图像延时默认
`8000 µs`，Trigger Out 延时默认 `7000 µs`，也可从命令行覆盖。视频彩色格式按固定优先级
自动选择，深度固定为 `Y16`，FPS 可由命令行覆盖并做 SDK 精确共同 profile 匹配；启动时和每个完整对齐 RGB-D 帧
写盘前都必须回读同步配置，任一帧回读不符都使会话失败。连续视频结束时必须先停止
Pipeline，再切换为 `STANDALONE`、关闭 Trigger Out 并回读确认；关闭失败使
`clean_shutdown=false` 且视频产品不可交付。
普通连续采集默认不得构造 `OnlineORBTracker`，manifest 应将 ORB 标记为
`deferred` / `post_2d_publication_only`。只有显式诊断入口可延迟 import 在线 ORB；
`g305-video-live` 必须拒绝 `--photo-mode` 和 `--diagnostic-online-orbslam3`。

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

## 6. 照片产品的位姿与正式 RGB 渲染

本节仅约束照片 `g305-panorama`。视频 S013 V11 二维路径遵循 1.2/1.3 的 ignore-pose
契约；视频 ORB-SLAM3/Open3D 只在二维交付后的独立 post-3D 进程中运行。

- 正式 `pose_backend=hybrid_orbslam3_rgbd`。每条 Open3D 边记录 source-to-reference SE(3)、收敛、fitness、RMSE、正定 `6×6` information matrix、深度有效率和失败原因。
- 正式并行前端要求每条 Open3D 边实际使用 `open3d_tensor_cuda_rgbd`。若观察到 `open3d_rgbd`，说明 CUDA Open3D 未生效，必须失败，不能静默接受 CPU legacy edge。
- ORB-SLAM3 必须跟踪完整正式序列。缺 pose、非有限/非刚体 SE(3)、图不连通、逆向或不连续运动、步长/跨度异常、垂直/前后漂移、旋转或边残差越界均为 F。
- 正式渲染保留主扫描段全部真实 pose nodes，不设固定的 pose-node 数量上限。不得固定抽稀为 32 个画布源，不得插值、重排或伪造 pose。
- 近重复节点只能在完整审计后成为零最终 owner；其真实 pose、边和一次 RGB remap 仍须保留。
- 每源只做一次全分辨率标定 inverse remap。中间源中央条带最多为输入宽度的 `20%`；首尾只可向扫描外侧扩展到校准图像边缘。
- 布局比例来自相邻 RGB 局部运动与真实 SE(3) 相机中心位移的稳健标量。它仅决定条带 x 布局，不是二维轨迹、单应矩阵、深度平面或 pose 修正。
- 画布和 aggregate working set 均不超过 `200 MP`；常驻 RGB 条带为 2–5 个。序列长度不由固定 pose-node 数量限制，但必须受上述资源上限约束。

## 7. 照片产品的风险走廊、局部几何与 owner

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

## 8. 照片产品的 A/B/C/F 与原子交付

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
| `g305-capture --photo-mode` | 正式主采集：软件触发同步 RGB-D 照片序列；可选 FPS 和两种同步延时，格式自动决定 |
| `g305-capture` | 连续 RGB-D 视频采集；默认不运行 ORB；`--diagnostic-online-orbslam3` 仅供隔离诊断 |
| `g305-panorama` | 正式 unified RGB-D 全景 |
| `g305-video-live` | 连续 RGB-D 采集、S013 V11 增量非正式预览与正式 P3；默认 defer 3-D，`--post-3d` 显式启动独立三维 |
| `g305-video-panorama` | 对已有连续会话执行锁定的 S013 V11 正式二维；默认在发布后 spawn 独立 3-D，`--defer-3d` 可延后 |
| `g305-video-post-3d` | 面向普通使用者的独立后期三维：从会话运行 ORB-SLAM3，再发布 `3d/` TSDF/GLB/Viewer |
| `g305-video-3d` | 低层三维发布器；要求显式 `--trajectory`、`--source-frame-id`、`--two-d-output` 与 `--input` |
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
- 视频正式 V11：`test_video_s13_v11_promotion.py`、`test_video_s13_v11_production_route.py`、`test_video_s13_live_acceptance.py`、`test_video_live.py`、`test_video_s13_live.py`、`test_video_delivery.py`
- 视频后期三维：`test_video_3d_postprocess.py`

合成测试不等于实机验收。涉及相机、Open3D、CUDA、ORB-SLAM3 或性能的改动，交付说明必须分别注明单元/合成测试、真实 Open3D 边、真实完整 ORB-SLAM3、历史失败数据和现场速度验收状态。历史输出和旧 schema 不能作为当前 v12-r1 正式验收。
