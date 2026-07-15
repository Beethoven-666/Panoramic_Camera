# AGENTS.md

本文档供后续在 `D:\Panoramic_Camera` 工作的开发代理阅读。开始修改前先阅读本文，再按需阅读 `README.md`、配置和相关源码。

## 1. 项目目标与正式架构

项目使用奥比中光 Gemini 305 采集同步 RGB-D 序列，并以 fail-closed 方式生成移动侧扫全景。默认工况是连续单向水平侧移、场景基本静止、最近物体约 `0.5 m`、最高速度约 `1.5 m/s`。普通用户只提供会话目录和输出目录，不调整曝光、步长、位姿、接缝或裁剪参数。

正式序列流程固定为：

```text
严格 RGB-D 会话、标定、曝光、清晰度与主扫描段
  → 自适应 RGB-D pose nodes
  → 每帧短基线 Open3D 相邻 RGB-D odometry（局部几何验证）
  → ORB-SLAM3 RGB-D 以完整短基线链求真实全局轨迹
  → 有限 4×4 camera_to_world SE(3)
  → 全分辨率 RGB-D TSDF 几何融合
  → 标定平面真实位姿重投影，恢复无深度白墙等背景纹理
  → TSDF 前景覆盖，保护近景物体的几何位置
  → 裁剪、质量门禁和原子交付
```

正式路径不得导入或回退到 UniStitch、LightGlue、MAGSAC、Torch、3×3 单应矩阵、二维累计或时间/二维运动位姿插值。ORB-SLAM3 仅提供真实 RGB-D 相机轨迹；未安装或未完整跟踪时应失败，不能回退为伪造位姿。`unistitch-pair` 暂时保留为可选历史双图诊断；`unistitch-sequence` 只是 `g305-panorama` 的弃用别名，运行同一 RGB-D 路径。

## 2. 开始工作前

1. 工作目录固定为 `D:\Panoramic_Camera`。
2. 先运行 `git status --short`，保留用户或其他代理的所有改动。
3. 优先使用现有 `.conda`，不要无故删除或重建。
4. 搜索优先用 `rg` / `rg --files`；修改文件使用补丁。
5. 不执行 `git reset --hard`、破坏性 checkout 或批量删除数据/输出。

常用验证：

```powershell
.\.conda\python.exe -m pytest -q
ruff check src tests
.\.conda\python.exe -m compileall -q src tests
git diff --check
```

`.conda` 中应安装 Open3D 0.19。Torch/Kornia/torchvision 只属于可选 `unistitch-diagnostic` extra，不是正式依赖。

## 3. 目录与模块职责

| 路径 | 职责 |
|---|---|
| `configs/demo.yaml` | 正式零调参采集、RGB-D 位姿、投影、GraphCut 与融合安全默认值 |
| `configs/capture_640x480.yaml` | 低带宽诊断配置，不是正式默认 |
| `configs/capture_unrestricted_auto_exposure.yaml` | 无限 AE 与非交付诊断的一体化配置 |
| `capture_orbbec.py` | Gemini 305 同步采集、COLOR_STREAM 软件对齐、曝光限制、元数据和异步写盘 |
| `photo_capture.py` | 无预览 RGB-D 照片序列状态机、软件触发、外部 Trigger Out 门控、最快共同 profile 与逐帧落盘 |
| `session.py` | 严格 RGB-D 会话、标定、aligned depth、曝光/时间戳和毫米单位契约 |
| `quality.py` | 缩略图画质、视觉运动、主扫描段、pose-node 布局和基于真实 SE(3)/足迹的渲染源选择 |
| `rgbd_odometry.py` | Open3D 延迟导入、短基线 RGB-D odometry、边质量、SE(3) 与轨迹审计 |
| `orbslam3_bridge.py` | WSL ORB-SLAM3 RGB-D 调用、标定去畸变、深度比例适配和真实轨迹解析 |
| `rgbd_projection.py` | 全分辨率 RGB-D 正射投影、世界表面深度、相机深度、valid masks、z-buffer 和资源预算 |
| `dense_fusion.py` | 全帧 TSDF 几何与基于真实位姿的稠密平面纹理恢复 |
| `render.py` | 深度硬约束 GraphCut、owner 拓扑、逐相邻边界局部 MultiBand、裁剪与最终门禁 |
| `stitch_sequence.py` | 正式 RGB-D 序列编排、报告、失败清理和原子交付 |
| `synthetic.py` | 带标定、aligned depth、已知 SE(3) 的合成 RGB-D 会话 |
| `stitch_pair.py` / `unistitch_adapter.py` | 可选历史双图诊断，不得进入正式序列 |
| `tests/` | 单元、合成、交付和零参数集成回归 |

CLI：

- `g305-capture`（连续流；`--photo-mode` 使用逐帧拍照实现低帧率序列）
- `g305-panorama`
- `unistitch-sequence`（弃用别名，同一 RGB-D 流程）
- `unistitch-pair`（可选历史诊断）
- `generate-panorama-demo`

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

## 5. 位姿、投影和选源边界

- Open3D 相邻 RGB-D 边始终必需并用于局部质量验证；ORB-SLAM3 只能输出完整 RGB-D 序列的真实 camera-to-world 轨迹，不能以特征匹配伪造缺失帧位姿。
- 每条边记录 source-to-reference SE(3)、收敛、真实 fitness/RMSE、正定 6×6 信息矩阵、深度有效率和失败原因。
- 正式配置只能等于或收紧默认 odometry/pose 门限；放宽只能进入诊断输出。
- 优化结果必须有限、旋转正交、行列式为 +1、图连通，并保持连续单向侧移。正式拒绝过大垂直/前后漂移、旋转、逆向和边残差。
- 渲染源只从具有真实优化位姿的 pose nodes 选择；不得插值或伪造中间位姿。
- 自动渲染源覆盖至少 `95%`，维持至少 `0.34` 相邻投影足迹重叠，至少两源，最多 32 源。
- 每个最终源全分辨率处理一次，不递归重采样。
- 正射投影输出 RGB、独立 valid mask、统一世界法向 `surface_depth_mm`、源相机 `camera_depth_mm` 及各自 valid mask。
- point splat 与每源 z-buffer 不跨深度断层连三角形，不补造深度空洞。
- 画布与 aggregate working set 均不得超过 `200 MP`；诊断也不能提高这些硬限。pose nodes 最多 160。

## 6. 稠密融合与历史 GraphCut 路径

默认正式 backend 为 `tsdf_plane_dense_rgbd`：TSDF 使用所有真实 ORB-SLAM3 位姿构建近景几何；主平面纹理由标定相机模型重投影，填补传感器在平整白墙上的深度空洞。不得把无效去畸变黑边当作纹理。`graphcut_depth_constrained` 仅保留给显式旧 Open3D backend 和回归测试。

- 仅相邻扫描源在互斥 pair corridor 内竞争；
- 风险包括 Lab/梯度、与世界原点无关的固定毫米世界表面深度差、源相机深度 `<1 m` 的近景，以及融合保护带；
- 高风险连通域必须由能够完整覆盖它的唯一可靠源硬拥有；不能从双方同时删掉；
- 没有唯一 owner、高风险带横断完整走廊或无连续安全通道时失败；
- GraphCut 异常不得回退 DP、Feather、平均或补洞。

GraphCut 后无条件运行 owner 审计：有效区每像素恰好一个 owner、无效区无 owner、非相邻 owner 不接触、每对相邻源边界存在且由真实共同有效区支撑。结构门禁不受 `quality_gate` 或 `--diagnostic-force` 影响。

MultiBand 只能在 owner 验证之后运行。每条相邻 owner 边界使用独立局部 `cv2.detail_MultiBandBlender`，不能把全部源 feed 到一个全局金字塔。重叠保护带按离 owner 边界的距离互斥分区，防止非相邻低频串色。区外直接复制唯一 owner；每个局部 output mask 必须完整、无零权重 wedge、无越界写入。不得回退旧自定义融合。

正式最终门限仍包括：精确边界与融合保护带风险 `<=0.10`、跨向覆盖 `>=0.80`、安全 Lab P95 `<=48`、曝光增益 `0.45–2.20`、裁剪高度 `>=90%`、画布宽度 `>=95%`。

## 7. 诊断与原子交付

正式曝光上限为 `800 µs`，输入拒绝上限固定为 `1200 µs`，设备原始 `color_exposure` 当前按 `100 µs/单位`。正式配置不能改变这一单位或放宽门限。

`--diagnostic-force` 可绕过输入绝对画质、正式 odometry/pose 质量和最终图像质量，但不能绕过：

- 标定、aligned depth、单位、曝光/时间元数据；
- 有限 SE(3)、正定信息矩阵、必需相邻边和图连通；
- 有效投影、owner 拓扑、GraphCut/MultiBand 结构完整性；
- 画布/working-set 硬限；
- 原子交付语义。

诊断成功只写 `diagnostic_panorama.jpg` 和 `diagnostic_report.json`，绝不写正式文件或 `delivery.json`。

每次 `run()` 的第一项文件动作必须使旧 `delivery.json` 失效。正式文件先写隐藏 pending，再 `os.replace`；`delivery.json` 最后写。普通异常无论从 CLI 还是直接调用 `run()`，都应清除正式/诊断产物并原子写 `failure.json`。强制终止可能来不及写失败报告，但没有有效 `delivery.json` 始终代表失败。

## 8. 测试导航

| 测试 | 重点 |
|---|---|
| `test_capture_calibration.py` | 相机属性、AE 上限、对齐 provenance、标定 |
| `test_photo_capture.py` | SOFTWARE_TRIGGERING、SBU gate-off 预热/静默、最快共同 RGB-D FPS、逐帧单次正式触发、失败停止、会话落盘与恢复 |
| `test_session.py` | 严格会话、曝光/时间、aligned/raw、尺寸和毫米单位 |
| `test_quality.py` | 画质、运动、主段、pose nodes、SE(3)/足迹选源 |
| `test_rgbd_odometry.py` | 单位、边、信息矩阵、图连接、SE(3)、轨迹、延迟导入 |
| `test_rgbd_projection.py` | 视差、z-buffer、断层、空洞、黑色内容、资源限制 |
| `test_render.py` | 深度硬约束、GraphCut、owner、逐 pair MultiBand、裁剪与风险 |
| `test_sequence_delivery.py` | 首先失效 delivery、各阶段失败、诊断隔离 |
| `test_sequence_integration.py` | fake RGB-D backend 的零参数完整交付，正式路径不导入旧模型 |
| `test_config.py` | RGB-D 默认值与正式安全包络 |
| `test_synthetic.py` | 合成 RGB-D 场景和已知轨迹 |

合成测试通过不等于实机验收。相机/Open3D/性能改动必须分别说明纯测试、真实 Open3D 边、历史失败数据和现场速度验收的状态。

## 9. 已知真实数据和现场验收

`data/captures/run_20260713_184519` 是无限 AE 诊断会话，`color_exposure=301`，约 `30.1 ms`。2026-07-13 用 Open3D 0.19 和 `--diagnostic-force` 复测：12 条必需相邻边均收敛，fitness `0.613–0.856`，RMSE `17.4–26.2 mm`；随后因高风险 RGB-D 带横断完整相邻 pair corridor 而正确失败。输出只含 `failure.json`，不能作为成功样本或通过放宽门限强行出图。

`data/captures/run_20260714_132427_262` 是混合路线的真实正式验收样本：主扫描段 101 帧均被 ORB-SLAM3 RGB-D 跟踪；100 条短基线 Open3D 边的 fitness 为 `0.939–0.998`、RMSE 为 `7.7–11.5 mm`。2026-07-14 正式交付的扫描跨度为约 `1206 mm`、最大单步旋转约 `0.43°`、稠密裁剪覆盖率约 `95.1%`，并发布了 `delivery.json`。这证明该会话可成功处理，不代表所有物理环境都必然成功。

旧 `greenhouse_trial/run_20260711_213054` 同样只适合输入质量应拒绝回归。源帧拖影和已丢纹理无法靠锐化或融合恢复。

新数据至少验收静止、`0.5`、`1.0`、`1.5 m/s`，检查队列丢帧、RGB-D 同步、曝光、深度有效率/单位、pose 残差、0.5 m 近景重影、GraphCut/MultiBand、裁剪四边及最终 `delivery.json`。在这些现场样本完成前，不得宣称所有物理环境必然成功。

## 10. 交付前检查

- [ ] `git status --short` 无意外文件或覆盖用户改动；
- [ ] 定向和完整 pytest 通过；
- [ ] Ruff、compileall、`git diff --check` 通过；
- [ ] 默认 `g305-panorama INPUT --output OUTPUT` 无算法参数；
- [ ] 正式路径无 UniStitch/Torch/LightGlue/MAGSAC 导入或回退；
- [ ] 失败路径先失效旧 delivery 并写 `failure.json`；
- [ ] 成功路径最后发布 `delivery.json`；
- [ ] README、AGENTS、配置、CLI、依赖和测试一致；
- [ ] 明确区分合成、历史失败、真实 Open3D 和现场相机验收。
