# AGENTS.md

本文档供后续在 `D:\Panoramic_Camera` 工作的开发代理阅读。它描述当前程序的目标、架构、不可破坏的交付约束、验证方式和已知环境限制。开始修改前，请先阅读本文，再按需阅读 `README.md` 和相关源码。

## 1. 项目目标

本项目使用奥比中光 Gemini 305 采集同步 RGB-D 序列，并利用 UniStitch、视觉运动估计和 RGB-D 接缝优化生成移动侧扫全景图。

默认交付工况是：

- 相机在水平方向连续、单向侧移；
- 场景基本静止；
- 最近物体约 `0.5 m`；
- 目标最高运动速度约 `1.5 m/s`；
- 用户只提供采集输出目录，不手工调整曝光、步长、帧号、接缝或裁剪参数。

程序的核心原则是 **fail closed（不合格即拒绝交付）**。无法可靠恢复的模糊、错误匹配、高视差接缝、覆盖缺口和黑边必须产生明确失败，不能为了“稳定出图”而发布劣质全景。

二维拼接无法保证动态物体、镜面反射、完全无纹理、严重欠光或源帧已经拖影的场景一定成功。此时正确行为是生成失败报告，要求补光、降低速度或重新采集。

## 2. 开始工作前

1. 工作目录固定为 `D:\Panoramic_Camera`。
2. 先运行 `git status --short`。工作树可能包含用户或其他代理尚未提交的改动，必须保留，不得重置、覆盖或顺手整理无关文件。
3. 优先使用现有 `.conda` 环境，不要无故删除或重建它。
4. 搜索文件和符号优先使用 `rg` / `rg --files`。
5. 修改文件使用补丁方式，并保持改动局限于当前任务。
6. 不执行 `git reset --hard`、破坏性 checkout 或批量删除输出目录。

当前常用验证命令：

```powershell
.\.conda\python.exe -m pytest -q
ruff check src tests
.\.conda\python.exe -m compileall -q src tests
git diff --check
```

注意：当前 `.conda` 环境内不一定安装 Ruff 模块，因此优先使用系统可调用的 `ruff check src tests`，不要假定 `.\.conda\python.exe -m ruff` 可用。

## 3. 目录与模块职责

| 路径 | 职责 |
|---|---|
| `configs/demo.yaml` | 正式零调参采集和序列拼接默认值 |
| `configs/capture_640x480.yaml` | 低带宽链路诊断配置，不是正式交付默认值 |
| `configs/capture_unrestricted_auto_exposure.yaml` | 解除项目曝光上限的诊断采集覆盖，不能用于正式交付 |
| `src/panorama_demo/capture_orbbec.py` | Gemini 305 配置、同步 RGB-D 采集、曝光限制、异步写盘和采集元数据 |
| `src/panorama_demo/session.py` | 读取 `frames.csv`、颜色帧、对齐深度和曝光等会话字段 |
| `src/panorama_demo/quality.py` | 输入质量、相邻视觉运动、主扫描段、自适应布局和清晰渲染源选择 |
| `src/panorama_demo/unistitch_adapter.py` | UniStitch 模型适配、匹配和变换结果封装 |
| `src/panorama_demo/stitch_common.py` | 模型构建、图像读写和双图/序列共用逻辑 |
| `src/panorama_demo/stitch_pair.py` | 双图诊断拼接入口，不代表完整序列交付流程 |
| `src/panorama_demo/stitch_sequence.py` | 完整序列编排、质量门禁、变换累计、渲染和原子交付 |
| `src/panorama_demo/render.py` | 画布、RGB-D 风险图、单调 DP 接缝、owner mask、多频段融合和无黑边裁剪 |
| `src/panorama_demo/synthetic.py` | 无相机合成数据生成与基础验证 |
| `src/panorama_demo/errors.py` | 轻量异常类型，避免非模型路径过早加载 Torch |
| `tests/` | 单元、合成和零参数集成测试 |
| `models/unistitch/` | UniStitch 权重位置，默认模型为 `epoch_best_model.pth` |
| `data/captures/` | 实机采集会话 |
| `outputs/` | 拼接结果、诊断图和成功/失败标记 |

命令行入口由 `pyproject.toml` 定义：

- `g305-capture`
- `unistitch-pair`
- `unistitch-sequence`
- `generate-panorama-demo`

## 4. 正式数据流

### 4.1 采集

正式采集命令不需要调参：

```powershell
g305-capture --output .\data\captures
```

默认采集配置：

- RGB 与对齐深度均为 `1280×800@30`；
- 软件对齐和帧同步开启；
- 彩色自动曝光开启，但曝光上限为 `800 µs`；
- 若固件不支持自动曝光上限，采集器退回固定 `800 µs`；
- `--diagnostic-unrestricted-auto-exposure` 会把 AE 上限设为设备回读范围最大值，只生成标记为 `diagnostic_only` 的诊断会话；
- `frames.csv` 中 `color_exposure` 保留设备原始单位，当前按 `100 µs/单位` 解释；
- 采集过程中连续出现超过安全上限的曝光元数据时应停止，而不是继续记录模糊数据。

典型会话结构：

```text
run_YYYYMMDD_HHMMSS/
├─ manifest.json
├─ calibration.json
├─ frames.csv
├─ color/
├─ depth_aligned/
└─ depth_raw/          仅诊断时可选
```

不要丢失 `frames.csv` 中的曝光、时间戳和 `depth_scale_mm_per_unit`；序列质量判断和深度单位换算依赖这些字段。

### 4.2 序列分析与布局

正式拼接命令同样不需要算法参数：

```powershell
unistitch-sequence `
  .\data\captures\run_YYYYMMDD_HHMMSS `
  --output .\outputs\greenhouse_sequence
```

默认序列流程：

1. 流式读取缩略图，测量清晰度、纹理、欠曝、过曝和相邻视觉运动；
2. 自动截取连续、单向的主扫描段，排除启动、停止和回扫；
3. 检查曝光、整体清晰度、运动方向、垂直跳动和相邻覆盖；
4. 根据实测像素位移选择布局帧，不依赖固定时间间隔或固定速度；
5. 用 UniStitch global 变换和 LightGlue + MAGSAC 候选进行几何验证；
6. 将通过验证的变换约束到默认 `translation` 运动模型并累计；
7. 根据完整轨迹、绝对清晰度和覆盖自动选择全分辨率渲染源；
8. 用实测运动进度插值密集帧位姿，因此可适应扫描速度变化；
9. 读取对齐深度并换算为毫米，执行 RGB-D 风险引导的单调 DP 接缝；
10. 进行有限多频段融合、最大有效矩形裁剪和最终质量门禁；
11. 所有检查通过后才发布正式文件。

显式传入 `--stride` 或 `--max-frames` 会切换到诊断覆盖路径并关闭默认自适应布局。不要把它们写进正式用户流程。

## 5. 不可破坏的交付约束

以下约束属于程序的安全边界。修改算法时不得静默放宽；若确需调整，必须同时提供真实依据、测试和文档更新。

### 5.1 输入质量

- 默认自动曝光上限：`800 µs`。
- 序列输入曝光拒绝上限：`1200 µs`。
- 无限自动曝光只能通过显式诊断模式启用；正式序列必须拒绝该会话，只有 `--diagnostic-force` 可以生成非交付诊断图。
- 运动模糊与 `速度 × 曝光时间` 成正比；物体越近，视差越明显。
- 连续三对缺少可靠视觉重叠、方向反转、过大水平跳变或过大垂直运动必须拒绝。
- 全局已经模糊、无足够纹理、严重欠曝或严重过曝时必须拒绝。

### 5.2 布局与渲染源

- 正式默认使用自适应布局，`layout_max_frames` 是硬预算，包含自动补入的尾帧。
- 最终渲染至少需要两张源图。
- 自动渲染源必须覆盖至少 `95%` 的可靠扫描范围，并维持安全重叠。
- 手工 `render_frame_ids` 可能绕过完整覆盖，因此正式序列命令必须拒绝用它发布交付件。
- 不得将不断变宽的累计全景重新输入模型，也不得递归重采样；最终全景应从原始全分辨率帧各 warp 一次。

### 5.3 接缝、融合和裁剪

- `scan_seam` 是唯一可发布的正式序列渲染模式。
- `feather` 仅用于诊断，不能生成成功交付标记。
- `--diagnostic-force` 可放宽正式 UniStitch/MAGSAC 几何门限，并绕过输入和最终渲染质量门禁；有限矩阵、画布和内存安全限制仍必须保留。它只能写 `diagnostic_panorama.jpg` 和 `diagnostic_report.json`，绝不能写正式文件或成功标记。
- RGB Lab 残差、梯度、深度不一致和近景共同构成接缝风险。
- 最终 owner mask 必须形成严格单一归属：不能有有效区空洞或多 owner 重叠。
- 只允许扫描顺序中相邻帧形成边界；非相邻 owner 接触必须拒绝。
- 每对相邻源的最终边界必须存在，并且必须由两张源图的真实重叠支撑。
- 多频段融合只能围绕最终 owner 边界发生，不能恢复整片重叠平均。
- 彩色 warp 使用边界复制避免亚像素插值引入黑线，但有效区域仍必须由独立 valid mask 限定。
- 自动裁剪后保留高度不得低于源图中值高度的 `90%`，保留宽度不得低于扫描画布的 `95%`。
- 当前接缝门限包括：精确风险和融合保护带风险均不高于 `0.10`，最小跨向覆盖不低于 `0.80`，安全接缝 Lab 残差 P95 不高于 `48`，曝光增益保持在 `0.45–2.20`。

### 5.4 原子交付

正式输出可能包含：

```text
panorama.jpg
transforms.json
render_transforms.json
report.json
delivery.json
```

必须遵守：

- 新任务开始时首先删除旧 `delivery.json`，防止清理中断后留下错误成功状态；
- 正式文件先写入隐藏 pending 文件，再通过 `os.replace` 发布；
- `delivery.json` 必须最后写入；
- 只有 `delivery.json` 存在且其中 `quality_pass=true` 时，目录才是有效交付；
- 普通异常必须清除正式文件并写入 `failure.json`；
- 强制终止可能来不及写 `failure.json`，但没有 `delivery.json` 仍代表失败；
- `pairs/` 是非交付诊断目录，可能保留上一次运行的预览，不能用于判断本次是否成功；
- 即使配置关闭某个诊断门禁，最终发布断言也必须阻止 `quality_pass=false` 的输入或渲染结果写入成功标记。
- 强制诊断输出必须使用独立文件名；新的正式或失败任务开始时应清除旧诊断文件，避免误认本次结果。

## 6. 测试导航

修改对应模块后先跑相关测试，再跑完整测试套件。

| 测试文件 | 重点覆盖 |
|---|---|
| `tests/test_capture_calibration.py` | 相机属性、自动曝光上限、固件回退和标定 |
| `tests/test_session.py` | 会话发现、CSV 元数据与深度单位 |
| `tests/test_quality.py` | 模糊/曝光拒绝、运动估计、主段、布局和清晰覆盖选帧 |
| `tests/test_sequence_motion.py` | 运动模型正规化和变速位姿插值 |
| `tests/test_unistitch_layout.py` | UniStitch 与 MAGSAC 几何验证和选择 |
| `tests/test_render.py` | 画布、DP 接缝、owner 拓扑、融合、裁剪、黑边和质量门禁 |
| `tests/test_sequence_delivery.py` | 旧交付清理、失败报告、诊断模式拒绝和原子标记 |
| `tests/test_sequence_integration.py` | 使用已知平移 aligner 的零调参完整交付路径 |
| `tests/test_config.py` | 默认配置的关键安全值 |
| `tests/test_synthetic.py` | 合成数据生成 |

渲染或交付语义有变化时，至少应增加下列类型之一的回归：

- 黑色本身作为有效内容，不能被误判为无效区；
- 亚像素平移或旋转边缘不能出现插值黑线；
- 非相邻 owner 接触、无重叠支撑边界、owner 空洞和 owner 重叠必须失败；
- 旧 `delivery.json` 必须在其他清理动作之前失效；
- 关闭诊断门禁也不能发布失败结果；
- 默认 CLI 不传曝光、步长、帧号、接缝或裁剪参数仍能完成合成端到端测试。

合成测试通过不等于实机验收完成。涉及相机、CUDA、UniStitch 权重或性能的改动，必须单独说明哪些部分只经过模拟验证。

## 7. 已知真实数据与环境限制

### 7.1 旧温室数据

`data/captures/greenhouse_trial/run_20260711_213054` 的彩色曝光原始值达到 `301`，按 `100 µs/单位` 计算约为 `30.1 ms`，远高于 `1200 µs` 移动安全门限。

该会话只能用于“应被输入质量门禁拒绝”的失败回归，不能作为新算法成功交付样本。源帧已经丢失的纹理不可通过锐化、融合或放宽门限恢复，必须重新采集。

### 7.2 Windows 与 CUDA

当前开发机曾受 Windows App Control/WDAC 策略影响，加载 PyTorch `c10.dll` 时被系统阻止。由于 Torch 在非模型测试路径采用延迟导入，纯离线测试仍可运行。

遇到该问题时：

- 不要通过修改拼接算法或降低质量门限规避；
- 需要系统策略放行运行库，或使用组织认可的签名 Python/PyTorch 环境；
- 在真实 CUDA UniStitch 尚未跑通前，交付说明必须明确“模型实机端到端未验证”。

### 7.3 现场验收

新采集数据至少应验证：静止、`0.5 m/s`、`1.0 m/s` 和 `1.5 m/s`，并检查：

- `queue_drops` 和写盘队列；
- RGB-D 时间戳及回退；
- 彩色曝光元数据是否保持在上限内；
- 对齐深度有效率和单位；
- 最近约 `0.5 m` 物体处是否出现重影；
- 接缝、亮度带、上下抖动和最终四边；
- 输出目录是否存在有效 `delivery.json`。

## 8. 修改规范

- 代码和错误信息沿用现有英文风格；面向用户的主要说明写入中文 `README.md`。
- 新增配置项必须有安全默认值，不能要求普通用户理解或调节才能得到正式结果。
- 诊断无限自动曝光必须显式解除设备上一次可能保留的 AE 上限并回读验证，不能仅跳过上限属性写入；允许固件按当前帧率将通用属性最大值钳制到较低回读值，但该值必须高于被替换的正式安全上限并记录到 manifest。
- 配置、CLI、README 和测试必须保持一致。
- 保持 Torch/UniStitch 延迟加载，使采集、配置、会话、质量和渲染单元测试不依赖可用 CUDA。
- 深度数组进入渲染前必须明确换算成毫米；不要混用设备原始单位和毫米。
- 不允许捕获异常后静默回退到整片重叠平均或发布部分扫描图。
- 不要把诊断输出改名成正式交付文件。
- 不要仅凭 JPEG 像素是否为黑色判断有效区域，应始终使用几何 valid mask。
- 大画布和多源 warp 必须继续受 `max_canvas_megapixels` 及 aggregate working-set 约束。
- 性能优化不能改变源帧单次渲染、严格 owner 分区和最终质量门禁语义。

## 9. 交付前检查清单

完成代码修改后逐项确认：

- [ ] `git status --short` 中没有意外文件或被覆盖的用户改动；
- [ ] 相关定向测试通过；
- [ ] 完整 `pytest` 通过；
- [ ] `ruff check src tests` 通过；
- [ ] `compileall` 与 `git diff --check` 通过；
- [ ] 默认用户命令仍无需算法参数；
- [ ] 失败路径不会留下旧 `delivery.json`；
- [ ] 成功路径最后才发布 `delivery.json`；
- [ ] README、配置和测试已同步；
- [ ] 明确区分合成验证、历史数据回归、真实 CUDA 验证和实机相机验收；
- [ ] 最终说明不宣称“所有物理环境必然成功”，而是说明自动成功或明确拒绝的边界。
