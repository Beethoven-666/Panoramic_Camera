# Panoramic_Camera 交给 Codex 的最终程序修改方案
## 算法生命周期重构 + 单数据集 AI 自动研发、消融、选型、CUDA 常驻与 Production 冻结

**仓库：** `https://github.com/Beethoven-666/Panoramic_Camera`  
**当前仓库基线提交：** `1d58159a429a37b1de9efbd9a053fcaad51813d1`  
**当前可执行视频代码行为基线：** `b07b561d03f2ddd85dcf0c3834ded8ec11c777ae`  
**唯一允许用于研发、调参、验证、性能投影和最终算法选择的数据集：**

```text
D:\central_strip_Panoramic_Camera\data\captures\video\run_20260804_162340
```

**目标硬件：**

```text
Gemini 305
848×480 @ 60 FPS RGB-D
RTX 5060 Laptop GPU
Windows + WSL2 ORB-SLAM3
```

**最终产品目标：**

```text
以 ORB-SLAM3 位姿作为稳定先验
+ 以 60 FPS 稠密图像运动决定全景推进
+ 以 RAFT 和 RGB-D 分层网格修正局部视差
+ 以前景对象锁定保证完整物体
+ 以可弯曲接缝和安全背景 MultiBand 完成合成
+ 以全局 photometric graph 统一颜色
+ 以端到端 CUDA 常驻链压缩停止采集后的处理时间
```

**最终现场验证顺序：**

```text
只用固定 3 m run 完成全部研发和算法选型
→ 冻结唯一 production
→ 满足固定数据集画质、稳定性、CUDA 和性能投影门槛
→ 生成 user_20m_test.ps1
→ 用户再执行一次真实 20 m 测试
```

---

# 0. 本文档的权威性与冲突统一

本文档合并并取代以下两份方案：

```text
Panoramic_Camera_Codex_Algorithm_Lifecycle_Refactor_Plan.md
Panoramic_Camera_AI_Single_Run_Algorithm_Selection_Plan.md
```

若旧方案、README、AGENTS、当前配置或历史注释与本文档冲突，以本文档为本次 Codex 任务的执行依据。

本次合并对冲突项作出以下最终决定：

1. **模式生命周期**采用 `baseline / candidate / production`，彻底删除旧 `fast / quality / audit` 产品模式。
2. **审计**只是一种输出与观测级别，不再是算法模式。
3. **研发数据**只允许 `run_20260804_162340`。
4. **单数据集划分**采用交错的 development/validation 区间和尾部 holdout，而不是简单连续的 55/25/20 切分。
5. **研发 CLI**使用独立的 `g305-video-experiment`；最终公共 `g305-video-panorama` 只运行 production。
6. **候选 C2**只加入 DIS RGB residual mesh；对象锁统一在 C5 加入，避免候选定义重复。
7. **Baseline 冻结**分两步：重构前用旧 `--preset fast` 生成锁；重构后只用新 `baseline` 入口复现。
8. **Production fallback**只用于结构性不可运行故障；视觉不达标、超时或局部降级不得静默回退。
9. **Production**可以使用真实 RGB 帧对应的 SE(3) 插值 pose prior，但必须明确标记为 prior，不得伪装成直接 ORB 节点。
10. 照片模式 `g305-panorama` 的原有约束保持不变；本文档中的 Torch、RAFT、mesh 与 MultiBand 例外只适用于独立视频产品。

---

# 1. 给 Codex 的最高优先级指令

你必须直接修改代码、运行测试、运行实验、计算指标、完成消融、搜索参数并冻结最终算法。

不要只交付：

- 建议；
- 伪代码；
- TODO；
- 单一候选；
- “下一步可以继续”；
- 只修改 Markdown 而不修改程序；
- 只给出一个未经实测的理想架构。

完整闭环必须是：

```text
检查仓库
→ 列出旧模式引用
→ 锁定唯一数据集
→ 冻结当前 baseline
→ 重构算法生命周期和审计输出
→ 建立 benchmark、split、标注和客观指标
→ 实现 C1～C8 候选与消融
→ 建立端到端 CUDA 常驻链
→ 参数搜索
→ validation 选型
→ 一次正式 holdout
→ 冷热重复与长度缩放预测
→ 冻结 production
→ 清理旧模式和死代码
→ 生成用户 20 m 脚本
```

## 1.1 只有以下情况可以询问用户

其余情况应自主继续：

1. 固定数据集任一哈希不匹配；
2. RTX 5060、CUDA、Torch、CuPy 或 Open3D CUDA 无法启动；
3. 下一步操作可能删除或覆盖用户原始数据；
4. RAFT 权重来源、许可证或模型哈希无法确认；
5. 所有合理候选都无法同时满足硬质量与性能门槛；
6. 发现仓库存在未提交的用户改动，与本任务修改同一代码区且无法安全合并。

## 1.2 禁止事项

禁止：

- 修改、重命名、移动、重编码或覆盖唯一 session；
- 使用其他历史 run；
- 使用 `run_20260804_134618_straight`；
- 要求用户重新采集 3 m；
- 使用网络测试视频；
- 使用 iPhone 图像作为数值 ground truth；
- 在 production 冻结前要求用户执行 20 m；
- 通过继续减少渲染源掩盖视差；
- 将明显失败的 pair 标为低风险；
- 关闭对象锁、深度边缘保护或 owner cleanup 换取速度；
- 用大范围羽化掩盖几何错位；
- 修改验收阈值让失败结果变成通过；
- 只依据旧 `quality_grade=A` 宣布成功；
- 在看过正式 holdout 后继续调参，却仍称其为首次盲测；
- 修改 baseline 参数；
- 执行 `git reset --hard`、破坏性 checkout 或批量删除采集/输出数据。

---

# 2. 当前基线事实

固定数据集当前已知 19.46 s 基线：

| 指标 | 当前值 |
|---|---:|
| 后处理总时间 | 19.4614928 s |
| 配置与会话 | 1.130 s |
| 扫描分析 | 1.001 s |
| ORB-SLAM3 | 7.447 s |
| Open3D 边审计 | 5.400 s |
| 渲染与输出 | 4.482 s |
| 运动分析帧 | 384 |
| ORB 跟踪帧 | 66 |
| 渲染源 | 36 |
| 相邻 pair | 35 |
| high-risk edge | 0 |
| normal step | 20 px |
| risk step | 8 px |
| CuPy 调用 | 144 |
| Open3D CUDA 调用 | 35 |
| H2D | 约 579 MB |
| D2H | 约 127 MB |
| blend pixels | 0 |
| deformation pixels | 0 |
| RGB-D local geometry | 未使用 |
| renderer | CPU DIS 证据 + curved hard-owner visual seam |
| geometry assist | 关闭 |

当前 delivery 虽然标记为 A，但它只证明旧结构检查和 20 s 预算通过，不能代表手机式视觉质量已经通过。

## 2.1 当前 baseline 身份

```text
role = baseline
algorithm_id = legacy_fast_b07b561
implementation_id = legacy_visual_seam
origin_commit = b07b561d03f2ddd85dcf0c3834ded8ec11c777ae
```

## 2.2 当前 baseline 输出锁

以当前上传结果作为重构前参考：

```text
panorama size:
1830 × 456

owner dtype:
int32

active owner count:
36

decoded RGB pixel SHA-256:
7171a368a0d0b1c595adfa1096a2b7198983680ada3140399be7c3064203cb1b

owner_frame_id raw bytes SHA-256:
f11e42b4347aa14927b0d99765d68f0aceed24d6c4a41ed9766e5a2fca4a2b6c
```

生命周期重构完成后，新 `baseline` 必须保持：

- 解码后的 panorama 像素完全一致；
- owner array 完全一致；
- source frame IDs 完全一致；
- panorama 尺寸完全一致。

JSON schema、运行时间、压缩容器 metadata 不要求字节一致。

---

# 3. 唯一数据集锁

唯一 session：

```text
D:\central_strip_Panoramic_Camera\data\captures\video\run_20260804_162340
```

已知控制文件 SHA-256：

```text
manifest.json
11e52a86126b7a4445806bb7b8b82abd507d35f90e3c94797e5008d87af89cb0

calibration.json
9e19b8dc506b27834b4fa0166294deecb1c23d93e3b7bb93184b3aa8c5691330

frames.csv
f27d7dd4b675193a3846fa70fd1e8461da7898568b300c6e1e4ea190e1fcb42d
```

Codex 首次运行时必须计算：

- 每张 RGB JPEG；
- 每张 aligned-depth PNG；
- 可选 raw-depth 文件；
- 在线状态和轨迹 cache；

的 SHA-256，生成：

```text
benchmarks\run_20260804_162340\dataset_lock.json
benchmarks\run_20260804_162340\source_files_sha256.json
```

每次正式 benchmark 前自动复核。任何哈希变化都应立即停止。

---

# 4. 删除旧 `fast / quality / audit` 模式

本次不是机械改名。

旧模式同时混合算法、帧密度、运行参数、renderer 和输出，因此必须删除其产品语义。

## 4.1 当前研发阶段

保留：

```text
baseline
candidate
audit 输出能力
```

删除用户可见：

```text
fast
quality
audit preset
```

## 4.2 Production 冻结后

正式保留：

```text
production
baseline fallback
audit 输出能力
```

普通用户只使用：

```text
production
```

## 4.3 新入口定义

| 新入口或能力 | 行为 | 用途 |
|---|---|---|
| `baseline` | 完全冻结当前 19.46 s 旧算法 | 回归比较和结构性故障 fallback |
| `candidate` | 读取明确候选 YAML，记录 changed components | AI 算法搜索和消融 |
| `production` | 读取冻结 lock 的唯一最佳算法 | 最终 3 m 验收和用户 20 m 测试 |
| `report-level` | 控制报告详细程度 | 不改变算法 |
| `artifact-level` | 控制中间证据输出 | 不改变算法 |
| `maximum-post-seconds` | production 预算输入 | 由 production 内部预算控制器处理 |

## 4.4 必须删除的旧配置和分支

从公共配置、CLI、报告与代码中删除：

```text
--preset
default_preset
preset == "fast"
preset == "quality"
preset == "audit"
fast_orb_target_fps
fast_orbslam3_rgbd
fast_rgbd_odometry
fast_odometry_prepare_workers
fast_session_validation_workers
fast_scan_analysis_workers
fast_publish_auxiliary_exports
fast_enable_geometry_assist
fast_renderer
report["preset"]
delivery["preset"]
```

这些值中用于复现当前基线的部分迁移到不可变 baseline YAML，不能丢失。

## 4.5 旧参数兼容

不得将旧参数静默映射。

允许暂时捕获：

```text
--preset fast|quality|audit
```

并立即报错：

```text
--preset 已删除。
研发请使用 g305-video-experiment 的 baseline/candidate；
审计使用 --report-level full --artifact-level audit；
production 冻结后普通用户直接运行 g305-video-panorama。
```

旧参数不得出现在 `--help`。

---

# 5. 最终 CLI 设计

## 5.1 研发入口

新增：

```text
g305-video-experiment
g305-video-benchmark
g305-video-replay
```

### Baseline

```powershell
g305-video-experiment `
  D:\central_strip_Panoramic_Camera\data\captures\video\run_20260804_162340 `
  --algorithm baseline `
  --report-level summary `
  --artifact-level minimal `
  --defer-3d `
  --output benchmarks\run_20260804_162340\baseline\legacy_fast_b07b561
```

### Candidate

```powershell
g305-video-experiment `
  D:\central_strip_Panoramic_Camera\data\captures\video\run_20260804_162340 `
  --algorithm candidate `
  --candidate-config configs\video_candidates\C4_raft_rgbd_layered_mesh.yaml `
  --report-level summary `
  --artifact-level minimal `
  --defer-3d `
  --output benchmarks\run_20260804_162340\experiments\C4
```

### 同一 candidate 的审计输出

```powershell
g305-video-experiment `
  D:\central_strip_Panoramic_Camera\data\captures\video\run_20260804_162340 `
  --algorithm candidate `
  --candidate-config configs\video_candidates\C4_raft_rgbd_layered_mesh.yaml `
  --report-level full `
  --artifact-level audit `
  --defer-3d `
  --output benchmarks\run_20260804_162340\experiments\C4_audit
```

## 5.2 公共 Production 入口

Production 冻结后，普通用户只运行：

```powershell
g305-video-panorama SESSION `
  --maximum-post-seconds 60 `
  --output OUTPUT
```

其默认且唯一正式行为：

```text
algorithm = production
report-level = summary
artifact-level = minimal
```

README 普通章节不得再要求用户选择算法。

## 5.3 参数约束

```text
baseline + candidate-config      → 错误
production + candidate-config    → 错误
candidate 缺 candidate-config    → 错误
production lock 不存在           → 错误
artifact-level=audit             → report-level 至少 full
```

Production 选型完成后，公共 `g305-video-panorama` 不应公开 `--algorithm`；baseline 和 candidate 只留在开发入口。

---

# 6. 配置和算法注册

## 6.1 目录结构

```text
configs/
├── demo.yaml
├── video_algorithms/
│   ├── baseline_legacy_fast_b07b561.yaml
│   ├── baseline_legacy_fast_b07b561.lock.json
│   ├── production_v1.yaml                 # 最终生成
│   └── production.lock.json               # 最终生成
└── video_candidates/
    ├── C0_baseline_reference.yaml
    ├── C1_constrained_owner.yaml
    ├── C2_dis_rgb_mesh.yaml
    ├── C3_raft_rgb_mesh.yaml
    ├── C4_raft_rgbd_layered_mesh.yaml
    ├── C5_object_lock.yaml
    ├── C6_multiband.yaml
    ├── C7_photometric_graph.yaml
    └── C8_multilabel_window.yaml
```

## 6.2 公共 `demo.yaml`

只保留共享设置：

```yaml
stitch:
  video_panorama:
    default_algorithm: baseline        # 研发阶段
    maximum_post_seconds: 60.0
    baseline_lock: configs/video_algorithms/baseline_legacy_fast_b07b561.lock.json
    production_lock: configs/video_algorithms/production.lock.json
    dataset_lock_required_for_experiments: true

    observability:
      report_level: summary
      artifact_level: minimal

    fallback:
      production_to_baseline_on_structural_failure: true
      fallback_grade: C
      manual_review_required: true

  video_runtime:
    session_validation_workers: 4
    scan_analysis_workers: 4
    odometry_prepare_workers: 4
    cuda_device: 0
```

Production 冻结 commit 中改为：

```yaml
default_algorithm: production
```

## 6.3 Algorithm spec

```python
AlgorithmRole = Literal["baseline", "candidate", "production"]

@dataclass(frozen=True)
class VideoAlgorithmSpec:
    role: AlgorithmRole
    algorithm_id: str
    implementation_id: str
    config_path: Path
    config_sha256: str
    source_commit: str
    model_sha256: dict[str, str]
    allow_baseline_fallback: bool
```

## 6.4 Observability spec

```python
@dataclass(frozen=True)
class ObservabilitySpec:
    report_level: Literal["summary", "full"]
    artifact_level: Literal["minimal", "provenance", "audit"]
```

## 6.5 Registry 规则

### Baseline

- 只能加载 baseline lock；
- 不接受 CLI 参数覆盖；
- 任何算法行为变化都必须生成新 baseline ID，不能覆盖旧锁。

### Candidate

必须包含：

```text
candidate_id
parent_candidate_id
changed_components
config_schema
config_sha256
```

### Production

- 只能读取 production lock；
- 不得读取 `configs/video_candidates/` 中的可变文件；
- lock 必须验证代码 commit、配置、权重和数据锁；
- 未冻结时必须明确拒绝运行。

---

# 7. Baseline 冻结过程

## 7.1 重构前运行

在删除旧 `--preset` 以前，先使用当前 CLI 生成 baseline：

```powershell
$Repo = 'D:\central_strip_Panoramic_Camera'
$Session = Join-Path $Repo 'data\captures\video\run_20260804_162340'
$Out = Join-Path $Repo 'benchmarks\run_20260804_162340\baseline\pre_refactor'

& "$Repo\.conda\Scripts\g305-video-panorama.exe" `
  $Session `
  --preset fast `
  --defer-3d `
  --maximum-post-seconds 20 `
  --output $Out
```

保存：

```text
command.txt
git_commit.txt
environment.json
video_panorama.png
video_panorama.jpg
video_pixel_provenance.npz
video_report.json
video_delivery.json
owner_map_color.png
owner_boundary_overlay.png
```

## 7.2 重构后验证

使用新 `baseline` 入口重新运行，验证：

```text
decoded panorama SHA 完全一致
owner raw SHA 完全一致
source IDs 完全一致
panorama size 完全一致
```

Baseline warm median 不得比 19.461 s 慢超过 15%，除非报告证明是环境波动。

---

# 8. Audit 与算法完全解耦

## 8.1 Output levels

### `minimal`

```text
video_panorama.jpg
video_panorama.png
video_pixel_provenance.npz
video_report.json
video_delivery.json
```

### `provenance`

在 minimal 基础上增加：

```text
owner_map_color.png
owner_boundary_overlay.png
owner_component_report.json
```

### `audit`

在 provenance 基础上增加：

```text
central_strips/
central_strips_owner_only/
flow_debug/
depth_layers/
mesh_debug/
object_debug/
seam_debug/
photometric_debug/
pair_audit.json
algorithm_trace.json
audit_manifest.json
```

## 8.2 硬不变量

同一：

```text
dataset lock
algorithm ID
config SHA
model SHA
source commit
random seed
```

下：

```text
summary + minimal
full + audit
```

必须得到相同：

```text
panorama raw pixel hash
owner raw hash
source frame IDs
pose prior
pair plans
flow/mesh/seam 决策
photometric 参数
blend labels
```

Audit 只能复制已经产生的算法证据，不得触发不同 renderer 或增加帧。

## 8.3 时间统计

主 SLA：

```text
primary_post_capture_seconds
```

审计输出：

```text
audit_export_seconds
```

推荐顺序：

```text
完成算法决策
→ 原子发布主图
→ 写 audit artifacts
→ 原子发布 audit manifest
```

Audit 失败不得撤销主图，但必须记录失败。

---

# 9. Report 和 Delivery v2

升级：

```text
gemini305-video-panorama-report/v2
gemini305-video-panorama-delivery/v2
```

## 9.1 Report

```json
{
  "schema": "gemini305-video-panorama-report/v2",
  "algorithm": {
    "role": "candidate",
    "algorithm_id": "C4_raft_rgbd_layered_mesh",
    "implementation_id": "video_visual_renderer_v2",
    "config_sha256": "...",
    "source_commit": "...",
    "model_sha256": {},
    "fallback_used": false
  },
  "observability": {
    "report_level": "summary",
    "artifact_level": "minimal"
  },
  "grades": {
    "structural": "A",
    "visual": "B",
    "performance": "A",
    "overall": "B"
  },
  "performance": {},
  "renderer": {}
}
```

## 9.2 Delivery

```json
{
  "schema": "gemini305-video-panorama-delivery/v2",
  "delivery_state": "published",
  "algorithm_id": "production_v1",
  "algorithm_role": "production",
  "fallback_used": false,
  "structural_grade": "A",
  "visual_grade": "A",
  "performance_grade": "A",
  "overall_grade": "A",
  "report_level": "summary",
  "artifact_level": "minimal",
  "report": "video_report.json"
}
```

禁止再写 `preset`。

旧 v1 输出只读兼容，不重写。

---

# 10. 实验基础设施

新增命令：

```text
g305-video-benchmark
g305-video-replay
```

## 10.1 Benchmark 目录

```text
benchmarks\run_20260804_162340\
├── dataset_lock.json
├── source_files_sha256.json
├── split_definition.json
├── annotations\
│   ├── objects.json
│   ├── lines.json
│   ├── safe_background.json
│   └── annotation_preview.png
├── baseline\
├── experiments\
│   └── <experiment_id>\
│       ├── config.yaml
│       ├── command.txt
│       ├── environment.json
│       ├── result.json
│       ├── video_panorama.png
│       ├── video_pixel_provenance.npz
│       ├── owner_boundary_overlay.png
│       ├── owner_map_color.png
│       ├── visual_metrics.json
│       ├── performance.json
│       └── logs\
├── leaderboard.csv
├── ablation_results.csv
├── tuning_results.csv
├── holdout_state.json
└── final\
```

## 10.2 Benchmark 命令

```powershell
g305-video-benchmark `
  --session D:\central_strip_Panoramic_Camera\data\captures\video\run_20260804_162340 `
  --algorithm candidate `
  --candidate-config configs\video_candidates\C4_raft_rgbd_layered_mesh.yaml `
  --repeat 3 `
  --output benchmarks\run_20260804_162340\experiments\C4
```

自动执行：

1. 数据哈希；
2. config/model hash；
3. 环境记录；
4. cold/warm 运行；
5. provenance 验证；
6. 视觉指标；
7. CUDA 与传输统计；
8. stage timing；
9. leaderboard 更新。

## 10.3 可重复性

固定：

```python
random.seed(20260804)
np.random.seed(20260804)
torch.manual_seed(20260804)
torch.cuda.manual_seed_all(20260804)
```

记录：

```text
deterministic algorithms
cuDNN benchmark
TF32
AMP
CUDA Graph
driver
CUDA runtime
Torch
Torchvision
Open3D
CuPy
model hashes
```

---

# 11. 单数据集 Development / Validation / Holdout

使用 60 FPS 累计可靠水平运动定义扫描进度：

```text
s ∈ [0, 1]
```

最终固定划分：

```json
{
  "development": [
    [0.00, 0.30],
    [0.48, 0.68]
  ],
  "validation": [
    [0.30, 0.48],
    [0.68, 0.84]
  ],
  "holdout": [
    [0.84, 1.00]
  ]
}
```

用途：

- Development：左侧设备、风扇和中部复杂物体；
- Validation：红瓶交接、右立柱等主要已知问题；
- Holdout：最右纸箱、工具、线缆和端点处理。

一旦提交 `split_definition.json`，不得修改。

## 11.1 Holdout 规则

冻结算法和参数前：

- 不显示 holdout crop；
- 不向 Codex 输出 holdout 局部图；
- 不将 holdout 得分用于调参；
- 不运行正式 holdout。

冻结后只允许一次正式 holdout。

若失败：

- 记录 `first_holdout_pass=false`；
- 可以重新研发；
- 后续测试必须标为第二轮，不得冒充首次盲测。

---

# 12. 固定标注

创建一次：

```text
benchmarks\run_20260804_162340\annotations\
├── objects.json
├── lines.json
├── safe_background.json
└── annotation_preview.png
```

## 12.1 对象

至少：

1. 左侧白色设备；
2. 红色瓶子；
3. 黑色风扇；
4. 右侧黑色立柱；
5. 黑色线缆；
6. 中部纸箱；
7. 右侧纸箱和工具；
8. 黄色横梁高风险结构。

对象标注以真实源帧 polygon/mask 和扫描进度保存。

## 12.2 长直线

至少：

- 黄色主横梁上下边缘；
- 横梁凹槽；
- 桌面前沿；
- 顶部货架板；
- 左右立柱。

## 12.3 安全背景

包括：

- 白墙；
- 黄色平面；
- 低纹理且无深度突变区域。

风扇、瓶子、线缆和深度边缘不得作为 photometric fit 样本。

标注一旦提交，不得因候选得分差而修改。

---

# 13. 客观评测指标

算法必须先通过硬门槛，再比较质量分。

## 13.1 结构硬门槛

```text
valid mask 完整
每个有效像素具有真实 frame provenance
owner/source 全部来自固定 session 真实帧
NaN/Inf = 0
输出空洞 = 0
mesh fold = 0
结果可重复
```

MultiBand 时 provenance 至少记录：

- dominant owner；
- 参与融合 source；
- 融合权重或可恢复摘要；
- sampling grid 版本。

## 13.2 对象完整性

```text
object_internal_seam_count = 0
dominant_owner_count = 1
maximum_handoffs <= 1
mask_coverage >= 98%
double_edge_width <= 2 px
```

红瓶、风扇、右立柱为强制门槛对象。

## 13.3 长直线

```text
yellow_beam_line_step_p95 < 1.0 px
table_edge_line_step_p95 < 1.0 px
pillar_edge_step_p95 < 1.0 px
line_orientation_delta_p95 < 3°
```

## 13.4 Owner 拓扑

当前基线参考：

```text
active owner = 36
connected components = 85
split owners = 26
```

最终要求：

```text
owner_connected_components <= 1.5 × active_owner_count
owner component < 128 px 的数量 = 0
最终 owner 像素 < 0.05% 全景面积的 source = 0
单一 owner 最大组件数 <= 2
对象锁内部 owner 岛 = 0
```

## 13.5 Flow 与 Mesh

```text
safe-region flow FB P95 < 1.5 px
mesh fold = 0
minimum Jacobian >= 0.05
local scale ∈ [0.70, 1.40]
unsupported displacement = 0
```

## 13.6 光度

```text
safe background seam ΔE00 P95 < 3
safe background brightness step < 2%
global max gain <= 1.35
global |bias| <= 0.08
扫描方向亮度漂移斜率接近 0
```

## 13.7 细节保持

在固定高纹理 ROI 计算：

```text
Tenengrad
Laplacian variance
edge width
double-edge score
```

MultiBand 或网格不能显著降低基线细节。

## 13.8 三类等级

```text
structural_grade
visual_grade
performance_grade
overall_grade
```

不再使用单一旧 `quality_grade` 表达全部含义。

---

# 14. 性能、CUDA 与 20 m 投影门槛

## 14.1 3 m 最终门槛

```text
warm median <= 8.0 s
warm maximum <= 9.0 s
cold run <= 12.0 s
peak VRAM <= 可用显存的 85%
无 OOM
```

## 14.2 CUDA 常驻门槛

```text
每个 render source H2D <= 1
H2D 总量 < 100 MiB
D2H 总量 < 15 MiB
intermediate D2H count = 0
最终 panorama/provenance 之外无大型 D2H
RAFT 在 CUDA
mesh 在 CUDA
MultiBand 在 CUDA
```

## 14.3 长度前缀

| 子测试 | 扫描进度 | 名义长度 |
|---|---:|---:|
| P25 | 0–0.25 | 0.75 m |
| P50 | 0–0.50 | 1.50 m |
| P75 | 0–0.75 | 2.25 m |
| P100 | 0–1.00 | 3.00 m |

拟合：

```text
T(L) = T_fixed + k × L
```

并计算最大增量斜率保守预测。

要求：

```text
20 m 线性预测 <= 50 s
20 m 保守预测 <= 55 s
保守预测 + 5 s 安全余量 <= 60 s
```

报告只能称为：

```text
fixed-run performance projection
```

不得称真实 20 m 已通过。

---

# 15. 在线回放

新增：

```text
g305-video-replay
```

## 15.1 Realtime

按 `frames.csv` timestamp 回放：

```powershell
g305-video-replay `
  --session D:\central_strip_Panoramic_Camera\data\captures\video\run_20260804_162340 `
  --mode realtime `
  --write-online-state benchmarks\run_20260804_162340\replay\online_video_state.json
```

验证：

- online motion 与离线一致；
- online ORB 可复用；
- GPU ring 不丢帧；
- keyframe selection 一致；
- GPU cache 正确释放；
- 采集期完成的工作不计入 post-capture time。

## 15.2 Unpaced

```text
--mode unpaced
```

测最大吞吐和 CUDA pipeline 纯性能。

---

# 16. 候选算法家族

不得预先认定模块最多的算法最好。

## C0：Baseline

```text
20 px normal step
CuPy remap
CPU DIS
curved hard-owner
source-0 photometric chain
无 mesh
无对象锁
无 MultiBand
```

只作对照。

## C1：约束 Hard-Owner

```text
12/5 px keyframe step
逐 pair 风险
seam corridor
一阶 + 二阶曲率
长直线代价
owner cleanup
```

## C2：DIS RGB Residual Mesh

```text
C1
+ DIS 对应
+ RGB residual mesh
```

不使用深度分层，不使用对象锁。

## C3：RAFT RGB Mesh

```text
C1
+ RAFT-small
+ RGB residual mesh
```

用于比较 RAFT 相对 DIS 的实际收益。

## C4：RAFT + RGB-D 分层 Mesh

```text
C3
+ depth confidence
+ far / mid / near 三层
+ 遮挡与反遮挡
```

不使用对象锁，不使用 MultiBand。

## C5：对象级 Owner 锁

```text
C4
+ 前景对象 track
+ 单一 owner
+ 最多一次受控 handoff
```

## C6：安全背景 MultiBand

```text
C5
+ 3/4/5 层 MultiBand 消融
+ 背景 16～24 px
+ 对象 0～2 px
+ 深度边缘 hard owner
```

## C7：全局 Photometric Graph

```text
C6
+ 取消 source-0 链式增益
+ 中位曝光 anchor
+ 邻接与跨一帧 overlap 图
+ 时间一阶/二阶正则
+ 可选 block illumination
```

## C8：五帧局部多标签 Owner

```text
C7
+ 5 帧局部窗口
+ 多标签 owner
+ 时间顺序与对象约束
```

只有速度仍满足门槛时才能胜出。

---

# 17. 消融实验

## 17.1 Flow

| ID | 方案 |
|---|---|
| F0 | CPU DIS |
| F1 | CUDA 局部相关或常驻轻量 flow |
| F2 | RAFT-small forward |
| F3 | RAFT-small 双向 |
| F4 | forward 全 pair，backward 仅高风险 |

## 17.2 Depth

| ID | 方案 |
|---|---|
| D0 | 不用深度 |
| D1 | depth edge guard |
| D2 | 单层 mesh |
| D3 | 三层 mesh |
| D4 | 三层 mesh + occlusion |

## 17.3 Object

| ID | 方案 |
|---|---|
| O0 | 无对象锁 |
| O1 | depth connected components |
| O2 | depth + flow track |
| O3 | O2 + 一次受控 handoff |

第一阶段不引入 FastSAM，除非 O1/O2 无法通过固定对象门槛。

## 17.4 Seam

| ID | 方案 |
|---|---|
| S0 | 当前 pairwise DP |
| S1 | corridor + curvature |
| S2 | S1 + line constraints |
| S3 | S2 + owner cleanup |
| S4 | 5 帧 multilabel |

## 17.5 Blend

| ID | 方案 |
|---|---|
| B0 | hard owner |
| B1 | 2 px feather |
| B2 | 3 层 MultiBand |
| B3 | 4 层 MultiBand |
| B4 | 5 层 MultiBand |

## 17.6 Photometric

| ID | 方案 |
|---|---|
| P0 | identity |
| P1 | 旧 source-0 chain |
| P2 | global graph |
| P3 | global graph + block illumination |

## 17.7 Open3D

| ID | 方案 |
|---|---|
| O3D0 | 全 pair，当前逐 pair 上传 |
| O3D1 | risk-only |
| O3D2 | DLPack GPU frame cache |
| O3D3 | 主图 risk-only + 发布后完整 audit |

---

# 18. 参数搜索和算法选择

## 18.1 搜索顺序

先选算法家族，再调参数。禁止一次性大网格搜索全部模块。

### 粗搜索

12～16 个有代表性的组合。

### 精搜索

对胜出家族执行最多 24 个确定性 TPE/Optuna trial。

可调参数最多包括：

```text
normal_target_step_pixels
risk_target_step_pixels
RAFT backward policy
mesh cell size
mesh smoothness
object guard pixels
seam curvature penalty
MultiBand levels
MultiBand width
photometric regularization
```

每个 trial：

- 只使用 development；
- validation 只用于选择前三；
- holdout 不运行。

## 18.2 提前停止

任一条件立即淘汰：

```text
mesh fold > 0
对象内部 seam > 0
global gain > 1.50
owner 碎片显著多于基线
3 m warm > 15 s
20 m 预测 > 80 s
CUDA OOM
结果非确定性超限
结构输出不完整
```

## 18.3 综合质量分

先通过全部硬门槛，再计算：

```text
Q =
0.30 × line_continuity
+ 0.25 × object_integrity
+ 0.15 × seam_photometric
+ 0.10 × owner_topology
+ 0.10 × detail_preservation
+ 0.10 × flow_mesh_consistency
```

性能是硬门槛，不能用高 Q 抵消。

## 18.4 最终规则

1. 选择 validation Q 最高且通过性能门槛的候选；
2. 冻结代码、YAML、模型和 SHA；
3. 正式运行一次 holdout；
4. holdout 通过后执行：
   - 1 次 cold；
   - 5 次 warm；
   - P25/P50/P75/P100 各 3 次；
5. 两候选 Q 差异小于 2% 时，选择：
   - 更快；
   - 模块更少；
   - 显存更低；
   - fallback 更简单；
   的候选。

结论只能写：

```text
在 run_20260804_162340 和 RTX 5060 Laptop 上最适合的算法
```

不能写“对所有场景最优”。

---

# 19. 端到端 CUDA 常驻实现

## 19.1 主 GPU 运行时

候选和 production 使用 PyTorch CUDA 作为主链：

```text
Torch CUDA Tensor
→ calibration grid
→ pose/scan initial grid
→ RAFT
→ depth layers
→ residual mesh
→ object mask
→ seam cost
→ MultiBand
→ tile accumulator
```

CuPy legacy remap保留给 baseline 和对照，不作为 production 主链。

## 19.2 GPU Frame

```python
@dataclass
class GpuVideoFrame:
    frame_id: int
    timestamp_us: int
    color_u8: torch.Tensor
    color_linear: torch.Tensor
    depth_mm: torch.Tensor
    depth_valid: torch.Tensor
    pose_prior: torch.Tensor
```

每个 source：

```text
H2D 次数 <= 1
```

禁止中间 `cp.asnumpy()`、flow 前下载、mesh 前下载或 blend 前下载。

## 19.3 三 Stream

```text
upload_stream
compute_stream
output_stream
```

重叠：

```text
下一窗口上传
当前窗口计算
上一 tile 输出
```

仅在窗口依赖、最终输出和小型标量审计处同步。

## 19.4 一次 Warp

将：

```text
镜头标定
+ ORB pose prior
+ 60 FPS scan coordinate
+ residual mesh
```

合并为单一 `grid_sample` grid。

最终 RGB 不得经历多次串联插值。

## 19.5 Open3D

比较：

### Risk-only

L2/L3 pair 执行 Open3D。

### DLPack Cache

Torch CUDA Tensor 通过 DLPack 交给 Open3D，左右相邻 pair 复用同一帧。

完整 information matrix 可延后到发布后 audit，但报告必须明确。

---

# 20. 候选技术起点

算法搜索不预设最终结果，但建议首先实现：

```text
ORB-SLAM3 8 FPS 直接节点
+ 对真实中间帧使用明确标记的 SE(3) prior 插值
+ 60 FPS CUDA 累计图像运动
+ 12/5 px 风险感知真实关键帧
+ RAFT-small 424×240 FP16
+ RGB-D far/mid/near 三层 residual mesh
+ 前景对象单一 owner
+ 5 帧局部多标签 seam
+ 安全背景 4 层 MultiBand
+ global photometric graph
+ Torch CUDA 常驻 tile renderer
+ Open3D risk-only 或 DLPack cache
```

只有消融和指标证明其胜出时，才冻结为 production。

---

# 21. Production 内部预算控制

不得再创建 `fast production` 或 `quality production`。

唯一 production 超预算时按顺序降载：

1. RAFT backward：all → risk-only；
2. normal mesh cell：24 → 32 px；
3. MultiBand：4 → 3 层；
4. 关闭最终 low-frequency straightening；
5. normal step：12 → 14 px；
6. 完整 Open3D audit 移到发布后。

不得优先关闭：

```text
对象锁
深度边缘保护
owner cleanup
前景 hard owner
provenance
```

报告记录：

```json
{
  "budget_policy": {
    "requested_level": 0,
    "applied_level": 2,
    "changes": [
      "raft_backward_risk_only",
      "multiband_levels_3"
    ]
  }
}
```

---

# 22. Baseline Fallback

## 22.1 允许自动 fallback

仅限：

- production renderer 初始化失败；
- 模型文件缺失或 hash 不匹配；
- CUDA OOM 且内部降载仍失败；
- mesh/owner 无法形成结构完整 panorama；
- 输出出现无效像素或不可恢复异常。

## 22.2 不允许自动 fallback

不得因以下问题回退来掩盖 production：

- visual grade 为 B/C；
- 对象内部存在 seam；
- 超过 60 s；
- photometric gain 超限；
- 某些 pair 局部退化；
- 候选尚未达门槛。

## 22.3 Fallback 等级

Fallback 必须：

```text
overall_grade = C
manual_review_required = true
fallback_used = true
```

不得标为 production A。

---

# 23. 代码重构

## 23.1 必改文件

```text
src/panorama_demo/video_panorama.py
src/panorama_demo/video_delivery.py
src/panorama_demo/config.py
src/panorama_demo/video_motion_resampler.py
configs/demo.yaml
README.md
AGENTS.md
pyproject.toml
tests/test_config.py
tests/test_video_delivery.py
```

## 23.2 生命周期与实验基础设施新增

```text
src/panorama_demo/video_experiment.py
src/panorama_demo/video_algorithm.py
src/panorama_demo/video_algorithm_registry.py
src/panorama_demo/video_algorithm_lock.py
src/panorama_demo/video_pipeline.py
src/panorama_demo/video_observability.py
src/panorama_demo/video_dataset_lock.py
src/panorama_demo/video_benchmark.py
src/panorama_demo/video_replay.py
src/panorama_demo/video_algorithm_selection.py
src/panorama_demo/video_visual_metrics.py
```

## 23.3 候选 renderer 可能新增

```text
src/panorama_demo/video_visual_renderer_v2.py
src/panorama_demo/video_gpu_runtime.py
src/panorama_demo/video_gpu_frame_cache.py
src/panorama_demo/video_cuda_motion.py
src/panorama_demo/video_raft.py
src/panorama_demo/video_pair_risk.py
src/panorama_demo/video_depth_layers.py
src/panorama_demo/video_layered_mesh.py
src/panorama_demo/video_object_tracks.py
src/panorama_demo/video_multilabel_seam.py
src/panorama_demo/video_torch_multiband.py
src/panorama_demo/video_photometric_graph.py
src/panorama_demo/open3d_gpu_frame_cache.py
src/panorama_demo/video_budget_controller.py
```

## 23.4 保留但重新定位

```text
calibrated_rgb_pushbroom.py      → baseline / legacy fallback
video_visual_renderer.py         → baseline curved hard-owner
video_photometric.py             → baseline photometric
CuPy remap                       → baseline / 对照
central strip exporters          → audit artifacts
video_pixel_provenance.npz       → production 正式输出
g305-panorama                    → 不修改
g305-video-3d                    → 与 2-D 生命周期独立
```

---

# 24. 关键接口

```python
class VideoPanoramaAlgorithm(Protocol):
    def prepare(
        self,
        *,
        session: VideoSession,
        online_state: OnlineState | None,
        context: VideoRunContext,
    ) -> PreparedVideoAlgorithm:
        ...

    def render(
        self,
        prepared: PreparedVideoAlgorithm,
    ) -> VideoAlgorithmResult:
        ...
```

```python
@dataclass
class VideoAlgorithmResult:
    panorama_bgr: np.ndarray
    owner_frame_id: np.ndarray
    source_frame_ids: tuple[int, ...]
    algorithm_audit: dict[str, object]
    artifact_sources: object | None
```

```python
@dataclass(frozen=True)
class PairPlan:
    left_frame_id: int
    right_frame_id: int
    risk_level: int
    flow_backend: str
    use_raft_backward: bool
    use_depth_mesh: bool
    use_open3d: bool
    object_lock_required: bool
    seam_mode: str
    blend_mode: str
```

---

# 25. 测试要求

新增：

```text
tests/test_video_algorithm_spec.py
tests/test_video_algorithm_cli.py
tests/test_video_baseline_freeze.py
tests/test_video_candidate_config.py
tests/test_video_production_freeze.py
tests/test_video_output_policy.py
tests/test_video_audit_invariance.py
tests/test_video_delivery_v2.py
tests/test_video_dataset_lock.py
tests/test_video_benchmark.py
tests/test_video_replay.py
tests/test_video_visual_metrics.py
tests/test_video_performance_projection.py
tests/test_video_fallback.py
```

必须验证：

```text
--help 不再出现 fast/quality/audit preset
--preset fast|quality|audit 明确报错
baseline 不接受参数覆盖
candidate 缺 config 时失败
production lock 不存在时失败
production 不读取可变 candidate 文件
summary/minimal 与 full/audit panorama hash 相同
summary/minimal 与 full/audit owner hash 相同
report/delivery v2 不含 preset
v2 包含 algorithm ID 和三类 grade
baseline 重构前后像素一致
fallback 必须为 C 且 manual review
candidate 不得自动 baseline fallback
visual 不合格不得自动 fallback
```

---

# 26. 分阶段实施与 Commit 顺序

不得一次性把生命周期重构和新 renderer 混在一个 commit。

## Phase 0：仓库盘点、数据锁和重构前 baseline

第一条命令：

```powershell
cd D:\central_strip_Panoramic_Camera

git status --short
git rev-parse HEAD

rg -n --glob '!outputs/**' --glob '!data/**' `
  'preset|default_preset|fast_|quality|audit' `
  src configs tests README.md AGENTS.md pyproject.toml
```

先生成“旧模式引用清单”，每项标记：

```text
删除
迁移 baseline lock
迁移 candidate config
迁移 observability
保留为普通英文含义
```

然后运行重构前 baseline，并建立 dataset lock。

## Phase 1：生命周期和 CLI 重构

完成：

- 删除旧 preset；
- 新增 algorithm registry；
- baseline/candidate；
- observability；
- report/delivery v2；
- 文档与测试。

验收：

```text
baseline panorama/owner hash 完全不变
```

## Phase 2：Audit 解耦和 Benchmark

完成：

- minimal/provenance/audit；
- audit invariance；
- benchmark 命令；
- replay 命令；
- environment 和 CUDA counters。

## Phase 3：Split、标注和 Visual Metrics

完成：

- split lock；
- objects/lines/background；
- owner topology；
- visual grades；
- prefix scaling；
- leaderboard。

核心 renderer 尚未改造前，评测系统必须先完成。

## Phase 4：C1 低成本修复

完成：

```text
12/5 px
逐 pair 风险
seam corridor
二阶曲率
line constraints
owner cleanup
```

目标：

- 横梁阶梯显著下降；
- owner components 明显少于 85；
- runtime 不显著恶化。

## Phase 5：CUDA 常驻底座

完成：

```text
Torch CUDA frame
pinned ring
GPU calibration grid
GPU linear RGB
GPU tile writer
final-only D2H
```

此阶段可以暂时沿用旧 seam，以验证数据链。

## Phase 6：Flow 与 Mesh 候选

完成 C2、C3、C4 和对应消融。

## Phase 7：对象锁、MultiBand、Photometric 与 Multilabel

完成 C5～C8。

## Phase 8：粗搜索、精搜索和 Validation 选型

生成 leaderboard 和 ablation。

## Phase 9：冻结候选并正式 Holdout

只运行一次首次 holdout。

## Phase 10：最终重复、Production 冻结和清理

执行：

```text
1 cold
5 warm
4 prefixes × 3
```

生成 production lock，修改默认 algorithm，清理旧模式和死分支。

---

# 27. 最终交付文件

```text
benchmarks\run_20260804_162340\final\
├── best_algorithm.yaml
├── best_algorithm_description.md
├── algorithm_selection_report.md
├── ablation_results.csv
├── tuning_results.csv
├── final_metrics.json
├── scaling_projection.json
├── final_video_panorama.png
├── final_video_panorama.jpg
├── final_video_pixel_provenance.npz
├── owner_boundary_overlay.png
├── owner_map_color.png
├── object_debug_overlay.png
├── mesh_debug_overlay.png
├── environment.json
├── model_hashes.json
├── reproduce_run162340.ps1
└── user_20m_test.ps1
```

仓库内：

```text
configs/video_algorithms/baseline_legacy_fast_b07b561.yaml
configs/video_algorithms/baseline_legacy_fast_b07b561.lock.json
configs/video_algorithms/production_v1.yaml
configs/video_algorithms/production.lock.json
docs/algorithm_selection_report.md
```

---

# 28. Production 冻结门槛

Codex 只有全部满足，才可宣布具备用户 20 m 测试条件。

## 28.1 质量

```text
红瓶内部 seam = 0
风扇内部 seam = 0
右立柱内部 seam = 0
对象 dominant owner = 1
对象 handoff <= 1
黄色横梁 line step P95 < 1 px
桌面 line step P95 < 1 px
safe background ΔE00 P95 < 3
owner 小碎片 = 0
mesh fold = 0
global max gain <= 1.35
```

## 28.2 性能

```text
3 m warm median <= 8.0 s
3 m warm max <= 9.0 s
3 m cold <= 12.0 s
20 m linear prediction <= 50 s
20 m conservative prediction <= 55 s
peak VRAM <= 85%
无 OOM
```

## 28.3 CUDA

```text
每个 source H2D <= 1
H2D < 100 MiB
D2H < 15 MiB
intermediate D2H = 0
RAFT CUDA
mesh CUDA
MultiBand CUDA
```

## 28.4 稳定性

```text
5 次 warm 全部通过
关键指标波动 < 3%
owner topology 无随机变化
无偶发 Open3D/RAFT 崩溃
```

---

# 29. 文档和死代码清理

选型完成后：

1. 将旧 `Panoramic_Camera_FastV2_CUDA_Resident_Quality_Recovery_Plan.md` 删除或移入 `docs/archive/`，标记 `SUPERSEDED`。
2. 本文档作为实施阶段唯一方案；最终由 `algorithm_selection_report.md` 作为 production 依据。
3. README 普通用户只展示 production。
4. AGENTS 明确：
   - 照片路径不变；
   - 视频无 fast/quality/audit preset；
   - baseline 不可修改；
   - candidate 只用于固定 run；
   - audit 不改变算法；
   - production lock 是唯一正式算法来源；
   - 视频 candidate/production 允许 Torch、RAFT、局部 mesh 和安全 MultiBand；
   - 真实 RGB 必须来自真实采集帧，provenance 可追溯。
5. 搜索：

```powershell
rg -n --glob '!docs/archive/**' --glob '!benchmarks/**' `
  'default_preset|--preset|preset ==|fast_enable_geometry_assist|fast_renderer|fast_publish_auxiliary_exports|quality preset|audit preset'
```

除明确的迁移报错测试和历史 baseline metadata 外，不应再有旧产品模式引用。

---

# 30. 用户未来 20 m 测试的性质

固定 3 m run 可以证明：

- 该场景上的算法选择；
- 客观画质改善；
- CUDA 链路；
- 长度缩放趋势；
- 在线回放；
- 预验收稳定性。

它不能证明：

- 真实 20 m 无累计漂移；
- 新场景所有物体都能锁定；
- 更长序列的 photometric graph 一定稳定；
- 磁盘、显存和温度在真实 20 m 完全相同；
- 所有农业场景都达到同样质量。

因此最终结论必须是：

> 已在 `run_20260804_162340` 上完成单数据集算法选型，并通过质量、CUDA、稳定性、在线回放和 20 m 性能投影门槛；现在具备由用户执行一次真实 20 m 验收的条件。

不得写：

> 已经证明真实 20 m 一定在 60 秒内完成。

---

# 31. Codex 最终回复格式

全部完成后才能使用：

```text
最终 production 算法：
production algorithm_id：
selected candidate：
代码 commit：
固定数据集：
首次 holdout：
最终 3 m cold：
最终 3 m warm median：
最终 3 m warm max：
20 m 线性预测：
20 m 保守预测：
structural grade：
visual grade：
performance grade：
overall grade：
CUDA H2D：
CUDA D2H：
intermediate D2H：
峰值 VRAM：
baseline fallback 状态：
未解决风险：
用户 20 m 命令：
```

并链接：

```text
algorithm_selection_report.md
best_algorithm.yaml
production.lock.json
reproduce_run162340.ps1
user_20m_test.ps1
```

---

# 32. 最终完成定义

只有以下全部成立，任务才完成：

1. 旧 `fast / quality / audit` 产品模式已从 CLI、配置、README、AGENTS 和 report/delivery 移除；
2. 当前 19.46 s 算法已冻结为不可变 baseline；
3. baseline 重构前后 panorama 和 owner hash 完全一致；
4. candidate 使用独立配置运行，不污染 baseline；
5. audit 输出不改变任何算法结果；
6. 固定数据集、split、标注和指标已锁定；
7. C1～C8 和必要消融实际运行；
8. 参数搜索、validation 和一次正式 holdout 已完成；
9. CUDA 常驻和传输门槛已通过；
10. 3 m 与 20 m 投影门槛已通过；
11. production 配置、模型和 commit 已锁定；
12. 公共 `g305-video-panorama` 默认只运行 production；
13. baseline fallback 只在结构性故障时触发，且明确降级为 C；
14. Codex 已生成用户 20 m 脚本，但未替用户运行真实 20 m。

---

# 33. 一句话总结

这项任务不是把旧模式机械改名，也不是直接实现一个未经比较的“完整 RAFT 方案”。

正确目标是：

> **冻结当前旧 fast 为不可变 baseline；删除 fast/quality/audit 产品模式；把所有画质和加速模块拆成可归因 candidate；只使用 `run_20260804_162340` 完成客观评测、消融和算法选择；将胜出的唯一算法冻结为 production；把 audit 变成不影响像素的输出能力；最终普通用户只运行 production。**
