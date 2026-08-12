# Panoramic Camera S013 M6.1 光度接缝抑制实施方案 v1

> 用途：直接交给 Codex 实施、测试和验收  
> 仓库：D:\central_strip_Panoramic_Camera  
> 基线分支：codex/video-realtime-seam-v6  
> 基线提交：c3aaa7ddbaf8aac5e00cded71ba9e4f1689b822a  
> 文档状态：可实施方案，不是完成报告  
> 适用范围：独立、production-ineligible 的 S013 视频候选实验  
> 固定真实数据：run_20260807_140140 与 run_20260806_153033  
> M6.1 所属阶段：仍然生成 P3；不得新造 P3.1，也不得提前实现 M7/P4

## 0. Codex 必须先读的结论

M6.1 的目标不是“让融合区域看起来更宽”，而是安全地消除 P2 中仍可见的系统性竖向亮度、白平衡和低频色带接缝，同时保持 P2 的几何、owner、真实来源和所有上游证据不变。

当前 M6 没有达到这个目标，原因已经定位为四类确定性问题：

1. YAML 中 photometric、blend 和 selection 参数没有接入 M6 正常或 resume 运行路径；算法实际使用 Python dataclass 默认值。
2. 当前 1 px feather 在整数像素中心上的 secondary weight 恒为零，是一个数学 no-op；大量 pair 的唯一 feather 尝试因此零支持并回退 B0，使可用 blend 候选空间实际上缺失。
3. 光度模型只按全局 held-out RGB 残差 median 选择，忽略 P95、最差 pair、色度、clipping、暗部抬升和参数跳变。slow_ignore_pose 因此选中 Q2，并把 pair 21 的旧灰度接缝从 2 放大到 9。
4. 当前质量门只检查 seam 两侧 x-1/x 的灰度中位数和全图 Laplacian，无法识别宽带竖向亮度条、同亮度异色接缝、局部少数强接缝和全局坡度。

因此实施顺序必须是：

~~~text
版本化与配置接线
→ 在新 generation 生成并封存宽肩部光度证据
→ 基于最终 sidecar 做 metrics-only 基线与阈值冻结
→ robust component 光度求解
→ Q0–Q3 fail-closed component 候选选择
→ 经约束授权后才可启用的可选 Q4 source-u 一维低频亮度场
→ 离散有效的安全窄带融合
→ 流式正式渲染与完整 provenance
→ hard audit v2
→ 四组真实验收
→ 仅在满足分流条件时生成 M7 handoff
~~~

不得通过扩大 feather、全图平滑或把安全门放宽来掩盖当前接缝。

## 1. 现状证据与必须修复的回归

证据来源；固定 generation 映射如下：

| 分支 | generation id |
| --- | --- |
| fast_direct | 20260812T140107-c4db600bf175 |
| fast_ignore_pose | 20260812T140107-fb6aafb7f542 |
| slow_direct | 20260812T140111-5a520ae4f223 |
| slow_ignore_pose | 20260812T140111-6319fb59b194 |

- artifacts\S013_M6_acceptance\M6_validation_summary.json
- artifacts\S013_M6_acceptance\P2_regression.json
- 各组 P3\photometric_solution.json
- 各组 P3\diagnostic_quality_report.json
- 各组 P3\blend_transactions.json
- 各组 P3\hard_audit.json

最直观的失败证据：

- slow_ignore_pose 的 generations\<generation_id>\P3\worst_visual_crops\color_00_pair_0021.png
- slow_ignore_pose 的 validation_m6\p2_vs_p3_full.png
- fast 两组的 validation_m6\p2_vs_photometric_owner_only_full.png

### 1.1 四组当前结果

| 分支 | pair 数 | 光度结果 | 旧 seam median/P95/max | 融合 provenance 像素 | 实际 RGB 改变 | 结论 |
| --- | ---: | --- | --- | ---: | ---: | --- |
| fast_direct | 84 | 85/85 source 为 Q0 | 1/2/11 → 1/2/11 | 3023，约 0.3024% valid | 2148，约 0.2148% | 光度为 no-op，接缝未解决 |
| fast_ignore_pose | 84 | 85/85 source 为 Q0 | 1/2/8 → 1/2/8 | 2687，约 0.2705% valid | 1943，约 0.1956% | 光度为 no-op，接缝未解决 |
| slow_direct | 108 | 109/109 source 为 Q0 | 1/1/2 → 1/1/2 | 0 | 0 | P3 与 P2 逐像素相同 |
| slow_ignore_pose | 109 | 88 source Q2，22 source Q0 | 1/2/2 → 1/2/9 | 0 | 514631，约 60.13% | 新造明显亮条，quality=false |

slow_ignore_pose 的 Q2 相对 Q0：

- held-out median 改善 2.52%；
- held-out P95 恶化 9.04%；
- source 21 到 22 发生 identity 到 Q2 的模型边界；
- red 相邻 log-gain 最大步长约 0.201，等价于约 22% 的突变；
- 最终 pair 21 seam jump 从 2 变成 9。

这不是需要 M7 修的小型几何残差，而是 M6 光度选择制造的系统性回归。M6.1 必须保证此候选被拒绝。

### 1.2 当前 blend 的确定性缺陷

当前代码选择：

- safe fraction 小于 0.35 时，B1 total width 为 1；
- safe fraction 大于等于 0.35 时，B1 total width 为 2；
- B2 只有 safe fraction 达到更高阈值才可用。

但 width=1 时，现有离散权重公式在 seam 中心两侧得到 0 或 1，经过 owner 侧筛选后 secondary weight 全为 0。真实数据中：

- fast_direct 有 74 个 width=1 no-op pair；
- fast_ignore_pose 有 76 个；
- slow_direct 有 108 个；
- slow_ignore_pose 有 109 个。

只有 fast 两组少量 width=2 pair 真正改变像素，两个 slow 分支完全没有融合。已有 sealed transaction 正确把零支持候选降为 B0；M6.1 要修的是 width=1 造成的候选空间缺失，并继续禁止把零像素候选报告为已应用 blend。

### 1.3 当前资源报告不可信

video_s13_m6.py 当前把每个 source 的 full-canvas float32 BGR 保存到 corrected 字典：

- fast 约为 1.02 GB；
- slow 约为 1.12–1.13 GB；
- raw cache 还没有包含在上述估算中。

当前 performance 报告的 41–48 MB 只计算少量单体数组，漏掉整个 corrected 字典；多个 decode、owner compose、blend compose timing 还被固定写成 0。M6.1 必须先修计量，再冻结资源门，不能沿用这个假数字。

## 2. 范围、非目标和不可违反的不变量

### 2.1 M6.1 必须解决

- 相邻真实来源间的曝光、整体亮度和白平衡不连续；
- 安全背景中的低频竖向亮度条带；
- 每 source 的全局白平衡/色度台阶，由 Q2/Q3 处理；
- 已正确对齐且无结构风险的 seam 附近极窄过渡；
- 当前模型选择造成的少数 source 参数突变；
- 现有质量门对宽带色差和尾部回归失明的问题；
- 配置、transaction、hard audit 与真实运行参数未绑定的问题。

### 2.2 M6.1 明确不解决

- 几何错位、台阶、双边、ghost、对象切断；
- seam 穿过对象或错误 owner；
- 深度层、遮挡、disocclusion 或透明物体问题；
- M4 轨迹、M5 几何、seam、placement 或 source 选择；
- P2 画布、P2 owner、P2 primary provenance；
- TSDF、Open3D、depth、ORB-SLAM3、GraphCut；
- M7 的局部结构修复；
- M8 的人工复核和 M9 的最终性能冻结。

结构问题可以被检测、保护并写入 M7 handoff，但 M6.1 不得修改其像素归属或几何。

source-u-dependent 的空间色度场不在 M6.1 模型族内。若全局 Q2/Q3 无法修复这类残差，必须输出 model_family_not_supported blocked reason，不能让 Q4 共通道亮度场冒充色度修复。

### 2.3 硬不变量

每次实现和验收都必须同时满足：

1. 旧 v3 config、旧 artifacts\S013_M6、S013_M6_final 和 S013_M6_acceptance 不覆盖、不改写。
2. 所有历史 v3 generation 的 P0、P1、P2 已封存资产不变；新 M6.1 generation 必须在 P2 seal 前增加 photometric replay sidecar。P3 运行开始后不得再改该 P2，也不得重估 trajectory、motion、placement、geometry 或 seam。
3. P3 immutable_primary_fields 逐数组等于父 P2；photometric_transaction_id、blend_transaction_id 和 secondary_* 是明确的 P3-owned fields，不属于“不变”集合。
4. P3 只能改变视觉 RGB，并在安全 blend 像素增加一个相邻真实 source 的 secondary provenance。
5. 每个有效像素最多两个真实 contributor，primary 始终占主导，secondary weight 小于 0.5。
6. protected、invalid、非共同可见和非 safe 像素始终 owner-only。
7. Q0+B0 路径必须使 P3 PNG 数组逐像素等于 P2 PNG，不能只满足肉眼相似。
8. 不允许任意 y 向或二维自由光度场；不允许全图 blur、feather、flow、homography、补洞或生成颜色。
9. 每个 render attempt 内，每个 source 只做一次正式标定 inverse remap；正常无 fallback 运行累计为 1。唯一授权的 candidate-dependent Q0+B0 rebuild 可产生第二个 attempt，累计为 2，绝不允许大于 2。
10. diagnostic visual quality 不拥有 P3 seal 或 pointer 权限。
11. 当前 AGENTS.md 只授权共同可见安全背景估计的全局线性 RGB gain/bias。Q4 是额外的空间光度模型，在项目约束未显式更新前必须保持 disabled，不能只改 YAML 绕过。

## 3. 版本化策略

不得静默改变现有 v3 行为。新增一套 M6.1 candidate，同时保留 v3 可复现：

~~~text
candidate_id / algorithm_id:
  S013_output_first_progressive_dense_central_slit_m61_v1

implementation_id:
  s013_output_first_progressive_dense_central_slit_m61_preview

contract_schema:
  gemini305-video-s13-output-first/v4

photometric_solution_schema:
  gemini305-video-s13-photometric-solution/v2

photometric_transaction_schema:
  gemini305-video-s13-photometric-transaction/v2

luminance_field_schema:
  gemini305-video-s13-source-u-luminance-field/v1

p2_completion_schema:
  gemini305-video-s13-p2-completion/v4

p2_photometric_replay_schema:
  gemini305-video-s13-p2-photometric-replay/v1

blend_transaction_schema:
  gemini305-video-s13-blend-transaction/v2

blend_transactions_manifest_schema:
  gemini305-video-s13-blend-transactions/v2

photometric_pair_evidence_manifest_schema:
  gemini305-video-s13-photometric-pair-evidence/v2

blend_pair_masks_manifest_schema:
  gemini305-video-s13-blend-pair-masks/v1

effective_config_schema:
  gemini305-video-s13-m61-effective-config/v1

quality_thresholds_schema:
  gemini305-video-s13-m61-quality-thresholds/v1

performance_schema:
  gemini305-video-s13-p3-performance/v2

visual_quality_schema:
  gemini305-video-s13-p3-diagnostic-quality/v2

p3_hard_audit_schema:
  gemini305-video-s13-p3-hard-audit/v2

p3_completion_schema:
  gemini305-video-s13-p3-visual-completion/v2
~~~

实现要求：

- 新增 configs\video_candidates\s013\S013_output_first_progressive_dense_central_slit_m61_v1.yaml。
- Commit A 先实现 identity/schema/parser，但不写带 placeholder threshold 的最终 manifest entry；Commit B 冻结 threshold 后才把最终 config 与 entry 原子加入 candidate_manifest.json，保留原 v3 entry。
- 把 video_s13_contract.py 中硬编码的单一 identity 改为显式 identity registry。
- v3 的 config 解析结果、hash、dispatch 和测试必须保持不变。
- video_s13_experiment.py 与 video_s13_bundle.py 不得再把 v3 algorithm id 写死到新 generation。
- manifest self-hash 和 config SHA 必须通过现有规范重新计算，不得手工填占位值。
- README 只新增 M6.1 诊断候选说明，不得宣称 production eligible，也不得宣称 M7 已实现。
- 历史 v3 P2_completion/v3 不变；只有新 M6.1 generation 使用 P2_completion/v4 并绑定新增 sidecar。

## 4. 三层权限模型

M6.1 必须把“候选能否改像素”“P3 能否封存”“视觉是否真的改善”拆成三层。

| 层 | 权限 | 失败行为 |
| --- | --- | --- |
| candidate safety / selection gate | 决定已授权的 Q0–Q3、可选 Q4 和 B0–B1 哪个候选可改像素 | 淘汰当前候选，回退更简单候选，最坏回 Q0+B0 |
| P3 integrity hard audit | 决定 P3 是否结构合法、可追溯、可 seal | candidate-dependent 首败可重建一次 Q0+B0；integrity 类立即失败 |
| diagnostic visual quality v2 | 决定 M6.1 工程验收、人工复核和 M7 优先级 | 写报告；不触发 runtime fallback，不控制 seal/pointer |

quality report 必须显式记录：

~~~json
{
  "candidate_selection_authority": false,
  "stage_seal_authority": false,
  "contributes_to_engineering_acceptance": true,
  "sole_engineering_acceptance_authority": false
}
~~~

说明：

- 候选选择使用与 quality v2 同源的安全指标，但选择决策在正式 render 前完成。
- quality v2 是正式 render 后的独立诊断，不可反过来修改已选择参数。
- hard audit 只审结构、hash、范围、provenance 和可重放性，不以“看起来更好”作为封存条件。

## 5. 第一阶段：先修配置接线

这是 M6.1 的第一个实现提交。未完成前不得实现 Q4 或新阈值。

### 5.0 Bootstrap 配置闭环

阈值文件在 metrics-only 后才存在，因此禁止在 Commit A 用 placeholder SHA 伪造最终 candidate。采用两个明确分离的 config identity：

1. photometric_evidence_config：只定义 P2 sidecar 的 geometry support、shoulder、mask/split/color metric 输入；生成 canonical photometric_evidence_config_sha。它不含候选阈值，不允许非 Q0/B0，也不 seal P3。
2. p3_effective_config：在 threshold freeze 后由最终 candidate YAML、已验证 threshold asset 内容和 evidence config SHA 共同生成 p3_effective_config_sha。

A/C/B 使用专用 metrics_calibration CLI/role；它不是 candidate manifest entry，只能生成/验证 sidecar 和 Q0+B0 metrics。P2_completion/v4 绑定稳定的 evidence config SHA 与 sidecar SHA，不绑定尚不存在的 P3 threshold。B 生成并显式审查 threshold JSON 后，才创建最终 M6.1 candidate YAML、config SHA 和 manifest entry。最终 P3 加载同一 P2-v4，验证 evidence config/SHA 完全一致，再绑定 p3_effective_config SHA。禁止 placeholder、候选运行前改 sidecar、或 threshold freeze 前 seal P3。

### 5.1 新增严格配置对象

在 video_s13_contract.py 或单独的 video_s13_m61_config.py 中定义并严格验证：

~~~python
S13PhotometricEvidenceConfig
S13PhotometricSolverConfig
S13LuminanceFieldConfig
S13PhotometricSelectionConfig
S13BlendConfig
S13VisualQualityConfig
S13M61EffectiveConfig
~~~

所有对象必须：

- frozen；
- 只接受版本化 YAML 中声明的键；
- 拒绝未知键；
- 拒绝 NaN、Inf、非法 fraction、负尺寸、奇怪层数和互相矛盾的门限；
- validated 后序列化成 canonical JSON；
- 生成 effective_config_sha256。

strict config 不得把关键界限留在源码。至少版本化：minimum train/validation sample count、vertical/shoulder/luminance coverage、maximum dominant-tile fraction、minimum inlier fraction、maximum robust scale、maximum condition number、minimum rank、maximum IRLS iterations、convergence tolerance、identity/smoothness regularization、parameter bounds、adjacent step/curvature、selection minimum absolute/relative improvement、minimum evaluable coverage，以及可选 Q4 的 gauge/slope/curvature/taper canvas 门。

唯一调用链必须变为：

~~~text
run_s13_experiment
  → load_s13_config
  → build_s13_m61_effective_config
  → _run_m6(..., effective_config)
  → run_s13_m6(..., effective_config)
  → audit_s13_p3_stage(..., same effective_config)
  → verify_sealed_s13_p3(..., same config SHA)
~~~

normal、resume、identity fallback 三条路径必须传递同一个不可变配置对象。

### 5.2 禁止死配置

为 YAML 中每一个版本化 M6.1 参数建立至少一个测试，证明修改该参数会发生以下之一：

- 确定性改变候选 eligibility；
- 确定性改变求解边界；
- 确定性改变 blend width/level；
- 确定性使 config validation 失败。

不允许“配置被解析但运行时从不读取”。solution、transaction、quality、hard audit、completion 都必须写入同一个 effective_config_sha256。

## 6. 第二阶段：在最终 evidence 定义完成后做 metrics-only 基线和阈值冻结

本阶段的运行顺序晚于第 7 节 sidecar 和第 8 节最终 mask、tile split、sampling、color metric 实现。章节编号用于说明职责，不表示可以在旧窄 corridor 上先冻结阈值。

在允许任何非恒等候选之前，先实现 quality v2 的测量部分，并对四组固定数据只运行：

~~~text
Q0 identity
B0 owner-only
metrics-only
~~~

输出到全新目录：

~~~text
artifacts\S013_M6_1_metrics_baseline\
~~~

生成：

- quality_metrics_baseline.json；
- 四组汇总表；
- 每 pair 的 evaluability、样本数和安全覆盖；
- P2 的多尺度 Luma/Lab 接缝指标；
- P2 的 clipping、dark、x-column profile 和 protected-only 问题；
- quality_thresholds_v1.proposed.json。

metrics-only 必须读取新 P2_completion/v4 已封存的最终 photometric replay，并使用最终 safe/protected、guard split、sampling、Lab/DeltaE 和 evaluability 定义。由实现者只根据这次 Q0 baseline 和固定物理量一次性校准阈值，然后把最终值写入：

~~~text
configs\video_candidates\s013\quality_thresholds_m61_v1.json
~~~

候选 YAML 必须包含 threshold asset 的相对路径和 SHA256；candidate config SHA 仍按现有 YAML 规范计算，effective config builder 读取并验证外部文件，把完整 canonical threshold 内容纳入 effective_config_sha256。启用 Q1–Q3、B1 或经授权的 Q4 后禁止再看候选结果调阈值；如果阈值必须修改，应 bump threshold version、写原因并整套重跑。

## 7. 第三阶段：扩展冻结的 P2 光度证据

### 7.1 为什么需要扩展

当前 P2 replay 只保存约 17–33 px 的窄 blend corridor。它足够做局部 provenance 重放，但不足以稳定识别 64 px 尺度的白墙亮度条和 source 内低频暗角。

M6.1 不得在 P3 重新估计 geometry。正确做法是在新的 v4 P2 generation 中增加只读光度证据 sidecar，同时证明 P2 canonical result 完全不变。

### 7.2 新 sidecar

保留当前窄 blend replay 不变，新增：

~~~text
P2\photometric_replay\
  pair_0000.npz
  pair_0001.npz
  ...
  manifest.json
~~~

schema：

~~~text
gemini305-video-s13-p2-photometric-replay/v1
~~~

每个 pair 至少包含：

- pair index、left/right source index 和 frame id；
- selected M5 transaction SHA；
- canvas x/y；
- seam x；
- left/right frozen source-u、source-v；
- left/right expected-valid；
- common-valid；
- M5 当前已经存在并审计的 geometry support、candidate support 和风险 reason bit；不得声称 M5 已有 M6 protected mask；
- seam 每侧 64 px 的输出诊断 shoulder 坐标；
- 只在双侧真实 source UV 都处于已审计 geometry support 内时保存 matched-source map；
- 所有数组的 dtype、shape 和 SHA256。

shoulder 坐标单位固定为 canvas pixel；maximum shoulder 为 seam 每侧 64 canvas px，不是总宽 64，也不是 raw source-u 64。每行 corridor 由 seam_x_by_row 正负 64 px 后裁到 canvas border，NPZ 可保存每行有效 x 范围或用 rectangular union 加 row-valid mask；manifest 必须报告 border-clipped row 数、实际最小/中位/P95 coverage 和 rectangular array bounds。

重要：当前 _map_crop 在 candidate.x0/x1 之外会把 local delta 置 0。因此不得把最终 M5 map 自然扩到正负 64 px，并把未经审计的 zero-delta UV 当作同场景 matched-source evidence。宽 shoulder 可以用于 P2/P3 输出条带诊断；Q1–Q4 solver sample 只能来自双侧 UV 均落在已审计 geometry support 的交集。宽域缺少双侧审计 UV 时必须标 insufficient_evidence，Q0/zero field，禁止猜测或外推。

这些资产必须由已选择的最终 M5 geometry/seam transaction 及其真实 support 确定性生成，只增加 evidence，不改变：

- P2 panorama PNG；
- P2 primary provenance NPZ；
- P2 owner；
- P2 pair transaction；
- 原窄 corridor replay；
- P2 result semantics。

新 candidate 为每个分支创建自己的新 generation 和 P2_completion/v4。验收时分别比较该分支的 P2 image、immutable primary provenance、pair transaction 和旧 narrow replay 与对应 v3 基线 hash 相同；新的 completion 因绑定 sidecar 必然使用新 schema/hash，不能要求与旧 completion 相同。P3 开始后该 v4 P2 才成为不可变 sealed parent。

### 7.3 明确禁止

- 不得在 M6.1 内运行 M4/M5；
- 不得重新选 motion hypothesis；
- 不得更改 seam；
- 不得把 RGB matching 结果反写 source UV；
- 不得让 photometric replay 成为第二个 owner 或 geometry 结果。

## 8. 第四阶段：安全样本与 pair evidence

建议在 video_s13_photometric.py 中新增：

~~~python
@dataclass(frozen=True)
class S13PhotometricPairEvidence:
    pair_index: int
    left_source_index: int
    right_source_index: int
    train_samples: object
    validation_samples: object
    support_by_y_bin: tuple[int, ...]
    support_by_shoulder_bin: tuple[int, ...]
    support_by_luminance_bin: tuple[int, ...]
    graph_edge_accepted: bool
    rejection_reasons: tuple[str, ...]
    baseline_metrics: dict[str, object]
~~~

每个 sample 必须保留 canvas xy、左右 source uv、linear BGR、Rec.709 Y、固定 CIELAB/CIEDE2000 需要的值、sample class 和权重。

### 8.1 Train / validation 必须空间独立

- 使用 canvas-global tile split，初始 tile 为 16×16 px。
- tile 分为 train、validation 和 excluded_guard 三类；旧 v1 术语 held-out 在 v2 schema 中统一迁移为 validation。
- 同一 canvas tile 在所有重叠 pair 中永远属于同一集合。
- 先按最大 filter radius 腐蚀 train/validation support，邻接和 halo 区成为 excluded_guard；train 与 validation 的 measurement halo 不得重叠。
- validation 不得参与 mask 阈值学习、pair relation、robust scale、component solve 或 Q4 solve。
- 报告 train_validation_overlap_count 和 guard pixel count，前者硬要求为 0。

### 8.2 确定性分层抽样

删除 flatten 后取前 8192 个样本的逻辑。改为按以下维度做确定性 reservoir 或固定排序抽样：

- y bin；
- seam 左右 shoulder offset bin；
- luminance bin；
- neutral / non-neutral；
- source-u coverage bin。

每个 bin 有明确上限和 minimum coverage，不能偏向画面上方或左侧。随机性如必须存在，只能来自由 candidate id、pair index 和 config SHA 派生的固定 seed。

### 8.3 样本类别

至少分三类：

1. general_safe：共同有效、未饱和、未保护、低到中梯度。
2. neutral_safe：general_safe 中低色度且左右梯度方向一致的稳定背景，主要驱动 luminance 和白平衡。
3. protected_only：存在结构、上游已封存的 risk bit、横向长边、双边、方向不一致、透明/反光风险或 M5 风险，只用于诊断和 M7 handoff。

protected_only 不得驱动 Q1–Q4 参数，也不得通过扩大 blend 解决。

safe/protected 是 M6.1 根据 raw replay、上游 sealed risk 和 effective config 确定性派生的 P3-owned evidence，不伪装成 M5 原生字段。M6.1 不读取 depth；只有 P2 已封存并 hash-bound 的上游 risk bit 可被复用。

### 8.4 Graph edge 证据门

一条相邻 source edge 进入 photometric graph 前必须同时满足：

- train 样本数达到配置下限；
- y 向有效覆盖达到下限；
- shoulder 和 luminance bin 覆盖达到下限；
- robust inlier fraction 达到下限；
- robust scale 有限且不超限；
- 没有单一小区域支配全部样本。

graph topology、relation、robust scale、edge weight、component 拆分和参数求解全部只使用 train。validation 数量不足或 train/validation drift 超限只能令已拟合候选不可选择，不能改变 graph 或参数。不满足时记录精确 reason code，不得静默丢弃。

validation 同时承担模型选择和 M6.1 运行报告，因此必须称 validation 而不是独立 test。四组固定真实数据是本 milestone 的 frozen engineering regression/qualification set，也参与 Q0 baseline 和一次阈值冻结，不是独立泛化 test，文档与最终报告不得宣称 unseen 泛化。非恒等候选首次运行后，任何算法或阈值变更都必须 bump implementation/threshold version，并把本轮标为 development；如果需要独立外部验收，必须另留从未参与 baseline、threshold 或迭代的实机序列。

## 9. 第五阶段：重做 Q0–Q3 component 求解

### 9.1 模型定义

- Q0_identity：gain 精确为 1，bias 精确为 0。
- Q1_scalar_luminance_gain：linear BGR 三通道共用一个 gain。
- Q2_rgb_diagonal_gain：每通道独立 gain，bias 为 0。
- Q3_bounded_rgb_gain_bias：每通道有有界 gain 和 bias。

Q1 使用 Rec.709 linear luminance：

~~~text
Y = 0.0722 B + 0.7152 G + 0.2126 R
~~~

禁止继续使用三通道算术平均冒充 luminance。

### 9.2 Robust component solve

每个有证据的 connected component 独立求解：

- pair relation 使用 Huber/IRLS，不使用一次普通最小二乘；
- edge weight 只综合 train 的 capped sample count、vertical coverage、robust scale 和 inlier stability；
- 所有 source 都加 identity regularization；
- source 参数加一阶和二阶平滑正则，抑制链式累计漂移；
- 明确 component gauge 和 anchor；
- 记录 rank、condition、iteration、convergence 和 residual；
- 所有 gain/bias 通过有界求解器约束，不允许求完以后逐 source 截断。

仓库当前正式依赖不保证 SciPy。实现不得假设传递依赖中碰巧存在 scipy.optimize。首选用 NumPy 实现固定迭代数、固定变量顺序的 deterministic projected-IRLS 或 active-set bounded least squares，并用小维度 reference 测试验证 KKT、bounds 和 determinism；如果选择新增依赖，必须先显式更新 diagnostic extra、lock/安装文档并验证目标环境。

### 9.3 禁止 component 内单节点 identity 回退

当前“某 source 参数越界就只把该 source 改回 identity”的逻辑必须删除。

新规则：

1. 有界求解可行：整个 component 使用该候选。
2. 有界求解不可行或 audit 失败：整个 component 回退到更简单模型。
3. 只有一条 graph edge 因真实证据不足被拒绝时，才允许在那条 edge 拆分 component。
4. 不得为了让参数过界而事后制造 identity 边界。

报告以下连续性指标：

- identity_fallback_inside_solved_component_count，硬要求 0；
- adjacent_log_luminance_gain_step P95/max；
- adjacent_log_chroma_ratio_step P95/max；
- 参数二阶差分 P95/max；
- component boundary reason。

validation 绝不能进入 edge weight、IRLS、regularization、component 拆分或参数求解；它只负责候选淘汰和选择。

初始保护门建议：

- max adjacent log-luminance step 不超过 0.06；
- max adjacent log-chroma-ratio step 不超过 0.04。

M6.1 v1 不允许超出上述连续性门的自动例外。未来如需例外，必须新增默认 false 的版本化 config flag、固定公式、reason schema 和独立测试后再 bump 版本。

### 9.4 模型状态不变量

模型状态拆成：

~~~text
component_selected_model
source_transform_is_identity
source_fallback_reason
~~~

Q0 component 要求 component 内所有 canonical gain=1、bias=0、field=0。非 Q0 候选如果全部参数变化小于版本化 canonical epsilon，必须 canonicalize 为 Q0；否则至少一个参数是被审计的非平凡变化。source_transform_is_identity 不能反向改变 component model，也不能用来制造 source-specific fallback。修复 Q3 继承 Q2 fallback 标志后仍带非 identity 参数的问题。solution、formal application 和 hard audit 必须使用同一份 canonical 参数数组。

## 10. 第六阶段：经约束授权后才可启用的可选 Q4 source-u 一维低频亮度场

重要授权门：当前 AGENTS.md 的视频视觉 renderer 只授权共同可见安全背景估计的全局线性 RGB gain/bias，video_s13_contract.py 也强制 low_frequency_luminance_field=false。因此 Commit A–D 不得启用 Q4。只有用户或项目维护者明确授权扩展约束，并在同一改动中更新 AGENTS.md、README、v4 contract/config validator 和测试后，才执行本节；否则跳过 Commit E，Q4 始终 disabled，M6.1 首版以 robust Q0–Q3 为正式范围。

原计划中的 PH4 必须在代码和报告中统一命名为：

~~~text
Q4_constrained_source_u_luminance_field
~~~

它不是二维场，也不是 canvas 全局曝光拉平。它只描述每个真实 source 随 source-u 缓慢变化的低频 luminance multiplier。它只能修 luminance；source 全局 white balance 由 Q2/Q3 处理，source-u-dependent chroma 不受支持。

### 10.1 数据模型

建议新建 video_s13_luminance_field.py：

~~~python
@dataclass(frozen=True)
class S13SourceLuminanceField:
    source_index: int
    frame_id: int
    source_u_knots: np.ndarray
    log_gain_knots: np.ndarray
    observed_u_min: float
    observed_u_max: float
    support_count: int
    support_weight: float
    field_sha256: str
~~~

### 10.2 求解定义

先应用已选 Q0–Q3 基础模型。对每个 safe matched-source sample：

~~~text
log(Y_left_corrected + epsilon) + f_left(source_u_left)
≈
log(Y_right_corrected + epsilon) + f_right(source_u_right)
~~~

f 使用 source-u 上的分段线性 basis。每个 component 用 train 样本做 Huber/IRLS 求解，validation 只用于选择。

硬约束：

- control spacing 不小于 64 个 raw-source pixel；
- absolute log gain 不超过 0.08；
- B、G、R 共乘 exp(f)，不得改变 chroma ratio；
- 每个 source 的 evidence-weighted field 零均值，整体曝光交给 Q1–Q3；
- 强二阶导 regularization，初值建议 10.0；
- 记录最大 amplitude、slope 和 curvature；
- 不允许 y、object、owner、depth、pose 或 frame time 作为 field 自变量；
- 0 field 始终存在；
- disconnected component 不传播参数；
- validation 不进入任何拟合或正则选择。

### 10.3 禁止无证据外推

field 只在 observed source-u 支持范围内应用。支持范围边缘使用至少 32 source pixel 的平滑 taper 回到 0；范围外精确为 0。不得把少量 seam 样本拟合出的趋势外推到整幅 source。

如果任一有 edge 的 source 的 field shape、gauge、amplitude、slope、curvature 或 train support 失败，则该 component 的 Q4 fit 不具资格；solve 完成后，任一 validation 门失败也只会在 candidate selection 阶段令整个 component Q4 回退到已选 Q0–Q3，不能重选正则、support 或重新拟合。不得局部半应用。没有任何 accepted edge 的 isolated source 自成 component，精确使用 zero field。

raw source-u 的 64 px spacing 不足以证明 canvas 输出低频。Q4 audit 还必须在所有实际 owner/replay UV 上计算 rendered canvas-x amplitude、一阶差分、二阶差分和 taper canvas 宽度；taper 经 map 后不得压缩成新亮线。Q4 对同一 source 的所有正式 owner RGB 一致应用，不能用 safe mask 只给背景局部涂改；protected 内容另做 photometric non-regression 诊断。

## 11. 第七阶段：fail-closed 光度候选选择

选择单位是 component，不再只写一个全局 model_family。每个复杂候选相对该 component 的 Q0 baseline 必须依次通过：

1. 结构、参数、field、rank 和 coverage 安全门；
2. aggregate validation median 有材料性改善；
3. aggregate validation P95 不回归；
4. 每个可评 pair 的 validation tail 不明显回归；
5. replay shoulder 上可在正式 render 前重算的 safe 宽带 Luma/chroma 不回归；
6. replay validation samples 上预测的 clipping、dark lift 和 neutral chroma drift 不越界；
7. source 参数连续性和 sidecar owner-domain audit samples 上预测的 x slope 不越界；
8. 新模型相对更简单模型有最小额外收益；
9. 分数接近时选更简单模型。

正式 render 前的 candidate gate 只允许使用 train 以外的冻结 validation/owner-domain sidecar samples；不得声称已经知道完整 P3 输出。完整全景 clipping、column profile、实际 changed RGB、blend sharpness 和 global slope 只在唯一正式 render 后由 quality v2 测量，不再反向改参数。若要让某项成为 pre-render gate，必须先在 photometric sidecar 中定义足量、确定性、hash-bound 的 owner-domain audit sample，不能偷偷做第二次 full render。

建议初始、待 metrics-only 冻结的选择门：

| 指标 | 初始值 |
| --- | ---: |
| aggregate validation median 相对 Q0 最小改善 | 1% |
| aggregate validation P95 最大回归 | 1% |
| per-pair validation P95 最大回归 | max(5%, 0.001 linear 绝对容差) |
| sidecar/compact replay predicted low-frequency DeltaE 最大回归 | max(10%, 0.5 DeltaE 绝对容差) |
| 新 high/low clipping 绝对增量 | 0.05 个百分点 |
| safe dark-region median L-star lift | 1.0 |
| global x slope 相对 P2 最大新增回归 | 5% |
| blend ROI gradient 最大衰减 | 8% |

关键验收样例：

- slow_ignore_pose 当前 Q2 因 global P95 恶化 9.04% 必须被拒绝；
- 即使 global median 改善，pair 21 出现 2 → 9 也必须被 per-pair/output gate 拒绝；
- slow_direct 如果 baseline 已经低于全部绝对 clean 门，可报告 clean_noop；否则 Q0 no-op 必须是 unresolved，不能 quality=true。

候选报告必须保留每个被拒模型的逐门结果和 reason，不能只保留 winner。

## 12. 第八阶段：修复安全窄带融合

M6.1 的 blend 是光度校正后的最后微调，不是主修复器。禁止为了提高 blended pixel fraction 而扩大到结构风险区。

### 12.1 候选定义

由 config 明确列出，不允许源码根据 safe fraction 硬编码宽度：

~~~text
B0_owner_only
B1_feather_2px
B2_masked_multiband_4px
B3_masked_multiband_6px
B4_masked_multiband_8px
~~~

要求：

- 删除当前 width=1 的 no-op 路径；
- minimum_total_width_px=2 必须真实生效；
- 每个离散宽度有精确 pixel-center 语义和 golden weight 测试；
- 不允许大面积 50/50 平台；
- secondary weight 必须大于 0 且小于 0.5；
- 如果计算出的 active count 为 0，candidate 状态必须是 ineligible_zero_support 或 B0，不能报告为已应用 B1。

### 12.2 Blend eligibility

每 pair 分别产生：

~~~text
pair_safe_mask
pair_protected_mask
pair_common_valid_mask
blend_eligible_mask
applied_blend_mask
~~~

定义：

~~~text
blend_eligible_mask
= pair_safe_mask
AND pair_common_valid_mask
AND expected_valid
AND NOT global_protected_mask

applied_blend_mask
= secondary_weight > 0
~~~

当前 safe_blend_mask 与 protected mask 有约 8 万像素重叠，名称和语义会误导。新 schema 不得继续用一个 union mask 同时表达 candidate-safe 和最终可融合。

### 12.3 候选选择门

每个非 B0 候选必须相对 photometric owner-only 同时满足：

- 至少一个 baseline-actionable 的 validation safe seam Luma/DeltaE 目标改善 5%；
- 所有其它 Luma/chroma/tail 指标通过非回归门；
- sidecar 上可预测的 local gradient、double-edge/ghost 风险不回归；
- active 支持不为空；
- 完整 footprint 与 protected 无交集；
- validation sample 的 predicted clipping 不回归；
- 与相邻 pair 不产生 contributor 或 corridor 冲突。

没有可测量改善时必须保持 B0。selected non-B0 必须 active_count>0；正式 uint8 render 后 actual_changed_rgb_count 也必须大于 0，否则 canonical transaction 降为 B0 并清除 secondary provenance。blend pixel fraction 不设最小值，0 是合法安全结果。

### 12.4 Masked MultiBand

当前 MultiBand 虽把 protected 像素 weight 置零，但 pyramid 卷积仍可能从 protected 邻域把颜色带入 active safe 像素。

M6.1 v1 的正式范围只启用 B1 2px feather；B2–B4 必须显式 ineligible。原因是 MultiBand 的邻域/每层 alpha 无法由当前单一同像素 secondary UV 和 weight 精确表达。

未来如启用 MultiBand，必须先 bump provenance/schema，保存或绑定 per-level alpha、mask、support 和读取邻域，并在 normalized masked support 与严格 geometric halo 两种安全语义中选定一种；不能继续声称单个 secondary_weight/UV 精确重放多频带输出。

### 12.5 全局 corridor 冲突

不要继续简单“先到先得，后 pair 整体 B0”。改为：

1. 按结构风险低、evidence 强、预期改善高的稳定 key 排序；
2. 对冲突 pair 先尝试下一档更窄候选；
3. 仍冲突则回 B0；
4. 最终重新生成 candidate audits，只有实际执行的候选 selected=true；
5. final transaction model、width、level、active count 与 provenance 必须一致。

排序 key 和 shrink 过程必须确定性，不能依赖 dict 顺序或线程完成顺序。

## 13. 第九阶段：唯一正式渲染与资源计量

### 13.1 两阶段执行

M6.1 分为：

1. evidence/decision 阶段：用 replay 的稀疏或紧凑样本决定光度 component、Q4 field 和 blend plan。
2. formal render 阶段：决策冻结后，仅执行一次正式 full-resolution source remap 和 compose。

不得为每个候选生成一整幅全画布 source cache。

### 13.2 Compact ROI / streaming

建议实现：

- 先从 P2 owner 和最终 active corridors 计算每个 source 的 union ROI；
- ROI 包括该 source 的 owner 区、相邻 active blend 区和必要的 interpolation halo；
- source 按扫描顺序处理；
- 常驻 2–3 个相邻 source ROI；
- evidence 阶段原始 RGB decode 也使用最多 2–3 source 的 LRU，结束后彻底释放，与 formal render cache 生命周期隔离；
- 完成相关 pair 后释放不再需要的 source；
- 保留 photometric_owner_only 和 final 两张全画布受控 working buffer；
- pyramid 临时缓冲只在当前 pair ROI 内存在；
- 不保留 source_count 份 full-canvas float32 图。
- 不返回或保留全序列 raw_cache；encoded bytes、decoded raw、remap ROI、canvas 和临时数组全部进入 live-byte/RSS 计量。

每 attempt、每 source 的 formal calibrated inverse remap invocation 必须精确为 1。正常运行累计为 1；唯一授权 identity rebuild 累计为 2，记录 attempt_id、abandoned witness 和 final witness，sealed P3 只绑定 final attempt。独立 hard audit 的 audit_remap 单独计数，不能冒充 formal。evidence 阶段的只读采样不是正式输出 remap，但 decode、sample 和 LRU 行为都要单独计数。

### 13.3 数值一致性

增加小尺寸 synthetic reference：

- full-canvas reference compose；
- compact ROI streaming compose。

两者对 owner-only 和 B1 必须逐数组相同或满足明确的浮点容差，最终 uint8 必须逐像素一致。B2–B4 不进入 v1 formal reference。

### 13.4 实际资源计量

performance.json 必须实测而非填 0：

- stage_total_seconds，截止 P3_completion.json 原子写开始前，包含 hard audit；
- full-resolution decode；
- replay evidence extraction；
- component solve；
- Q4 solve；
- plan selection；
- formal remap；
- owner compose；
- blend compose；
- quality v2；
- hard audit；
- process peak RSS 或 Windows peak working set；
- 显式 live-array peak；
- source ROI resident peak count。

performance.json 必须在 completion 前写入并被 completion hash-bound，因此它不能事后知道 completion 自身写完的时刻。外层 experiment/acceptance harness 在 atomic completion write 返回后，另把 end_to_end_through_completion_seconds 写入 P3 目录之外的动态 acceptance summary/返回值；该动态值不反写 sealed P3。

计时字段测试要求字段存在、有限、非负且 measured/source 不为 placeholder；小 synthetic 可因计时分辨率得到 0。真实四组 acceptance 的 evidence/solve/formal remap/owner compose/quality/hard audit 等实际执行阶段必须大于 0，stage total 与阶段和 overhead 关系合理。

先在 metrics-only 与第一个实现结果中测得真实峰值，再冻结资源目标。不得以当前假 41–48 MB 为基线，也不得让 M9 性能目标反过来削弱 M6.1 的视觉安全门。

建议工程目标：

- 四组每组 end_to_end_through_completion_seconds 不超过 60 s；边界从 sealed P2 load 开始，到 P3_completion.json 原子写返回，包含 evidence、solve、formal render、quality、hard audit 和 completion write，排除 acceptance atlas 汇总与后续 M7 handoff。该值由 P3 外层 acceptance summary 记录并执行门；超时使该组 M6.1 engineering acceptance=false，不得只记录后通过；
- resident full-resolution source ROI 不超过 3；
- 不存在 source_count × full_canvas float cache；
- actual peak RSS 门在第一次正确计量后版本化冻结。

在 RSS 门冻结前也必须有结构性 fail-closed cap：每次 allocation 前以 checked arithmetic 计算 requested live bytes，canvas/ROI/pyramid/source LRU 总量不得超过 config 的 provisional maximum_live_array_bytes；canvas shape、ROI count 和 resident source count 先审后分配。首次真实计量只能收紧或有版本理由地调整 cap，不能无限制运行。

## 14. 输出资产与 provenance

### 14.1 P3 正式结构资产

新 P3 至少包含：

~~~text
P3\
  visual_panorama.png
  photometric_owner_only.png
  p3_pixel_provenance.npz
  effective_m61_config.json
  photometric_solution.json
  photometric_solution.npz
  luminance_fields.npz
  photometric_pair_evidence\
    manifest.json
    pair_0000.npz
    pair_0000.json
    ...
  photometric_transactions.json
  blend_plans.json
  blend_transactions.json
  blend_pair_masks\
    manifest.json
    pair_0000.npz
    ...
  pair_safe_mask.png
  global_protected_mask.png
  blend_eligible_mask.png
  applied_blend_mask.png
  blend_weight_map.png
  diagnostic_quality_report.json
  hard_audit.json
  performance.json
  p2_parent_reference.json
  P3_completion.json
~~~

canonical 文件名保持现有 stage machinery 约定，版本只由文件内 schema 区分。audit-only 可视化 alias 必须位于明确的 diagnostics 子目录并由 canonical 确定性派生；不得创建 completion、provenance、audit 或 quality 的第二个正式真相源。sealed allowlist 只接受表中 canonical 和版本化 manifest 明确列出的资产。

blend_pair_masks 的每个 NPZ 保存该 pair 的 safe、protected、eligible、active、weight 和 corridor bounds；manifest 绑定顺序与 SHA，供 hard audit 逐 pair 重放。全局 PNG 只用于诊断 union，不能替代 per-pair 证据。

photometric_pair_evidence 的每个 NPZ 保存 train/validation/guard、sample class、canvas/source coordinates、safe/protected/support masks 和版本化 metric inputs；配套 JSON 保存 counts/reasons/metrics，manifest 绑定 pair 顺序、dtype、shape 和每数组/file SHA。不能只用一个汇总 JSON 代替可审计数组。

### 14.2 Photometric transaction

每 source transaction 必须绑定：

- source index 和真实 frame id；
- selected component id 和 model；
- gain/bias 数值 SHA；
- field knots/values SHA；
- parent P2 source identity；
- effective config SHA；
- photometric solution SHA；
- formal ROI 和 remap witness；
- Q0 identity invariant audit。

photometric transaction 的 JSON id/sha 必须由上述 canonical 内容生成，不能继续只是 source index。为最小化 provenance migration，NPZ 保留现有 int32 字段名 photometric_transaction_id，但 v2 语义明确为 manifest table index；photometric_transactions.json 提供 index→transaction_sha256 的唯一映射。不得把 SHA 塞入 int32，也不得让 index 自身冒充 content id。

### 14.3 Blend transaction

每 pair transaction 必须绑定：

- pair index 和左右 source/frame；
- parent P2 pair transaction SHA；
- narrow replay asset SHA；
- photometric replay asset SHA；
- photometric solution SHA；
- effective config SHA；
- candidate audit 列表；
- selected model、width、levels；
- safe/protected/eligible/active/weight SHA；
- secondary source/frame/UV SHA；
- active count 和实际 changed RGB count；
- conflict/shrink/fallback reason。

NPZ 保留现有 int32 字段名 blend_transaction_id，v2 语义同样是 manifest table index，blend_transactions.json 绑定 index→transaction_sha256。

不得在独立 hard audit 运行前直接写 hard_audit_passed=true。正式 transaction 写的是 self_audit 或 candidate_gate 结果，独立 P3 hard audit 另行输出。

### 14.4 Primary 和 secondary provenance

provenance 分为两组。

immutable_primary_fields 必须逐字段复用 P2：

- owner source/frame；
- source-u/source-v；
- valid；
- assignment_index；
- selected_motion_hypothesis_id；
- placement_method_code/names；
- geometry transaction id 和 seam transaction id。

p3_owned_fields 包括 photometric_transaction_id、blend_transaction_id 和全部 secondary_*。它们在 P2 是 sentinel，允许且必须由 P3 按下列规则更新：

- photometric_transaction_id 作为 manifest table index，精确映射到 canonical source transform SHA；
- active 像素必须精确等于 replay 中的相邻另一个真实 source；
- frame id、source index、u、v 和 pair transaction 必须逐值匹配；
- owner-only 像素的 secondary frame/source/transaction 为 -1；
- owner-only secondary u/v 为 NaN；
- owner-only secondary weight 为 0。

## 15. 视觉质量 v2

### 15.1 状态而不是单一布尔值

每组最终状态至少为：

~~~text
clean_noop
materially_improved
unresolved
regressed
unevaluable
~~~

建议：

~~~text
quality_pass
= measurement_valid
AND non_regression_pass
AND (
     baseline_actionable == false
     OR material_improvement_pass == true
   )
~~~

这避免三种错误：

- 明显有条带但 Q0 不动仍报 true；
- 没有可修问题时强迫非 identity 修改；
- 样本不足被当作通过。

可执行状态机固定为：

~~~text
coverage 未达到版本化 minimum
  → unevaluable, quality_pass=false

任一 non-regression gate 失败
  → regressed, quality_pass=false

coverage 充分且 baseline_actionable=false
  → clean_noop, quality_pass=true

baseline_actionable=true 且全部 material-improvement gate 通过
  → materially_improved, quality_pass=true

baseline_actionable=true 但改善不足
  → unresolved, quality_pass=false
~~~

threshold schema 必须定义 minimum safe-evaluable pair fraction、minimum row/block coverage、maximum insufficient-evidence pair fraction、protected-only 分类证据和每项 regression tolerance。aggregate 同时输出 pair-uniform macro 与 sample-weighted micro；两者都过门，防止大 corridor 淹没小 pair。baseline_actionable_by_branch 和 baseline_actionable_by_pair 在 Q0 metrics-only 时冻结并 hash-bind，候选不能重分类。

### 15.2 每个指标必须带测量有效性

所有 per-pair、component 和 global 指标都带：

~~~text
evaluable
sample_count
coverage_fraction
reason
~~~

每 pair 必须归类为：

- safe_evaluable；
- protected_only；
- insufficient_evidence。

不能静默从 aggregate 中消失。

before/after 必须在完全相同的冻结 canvas 坐标、safe/protected 分类和 sample mask 上计算。候选不得通过缩小 safe mask、改变 evaluability 或把问题 pair 改标 protected_only 来改善分数。

### 15.3 三尺度安全接缝

在 safe、非 global protected、common-valid 上分别测：

1. immediate：seam 两侧约正负 1 px；
2. shoulder：seam 两侧 2–8 px，另增加 4–16 px 宽肩部指标；
3. low-frequency：固定 CIELAB 后做 mask-normalized sigma=3 Gaussian 局部低通，再测 shoulder。

每个尺度报告：

- absolute 和 signed Delta L-star；
- Delta a-star、Delta b-star；
- Delta C-ab 或版本固定的 DeltaE；
- median、P90、P95、P99、max；
- 有效 block 数和 row coverage。

color_metric_schema 在 v1 固定为：uint8 BGR 转 RGB、IEC sRGB EOTF 解码到 [0,1] linear RGB、D65/2-degree linear RGB→XYZ→CIELAB、DeltaE 使用 CIEDE2000；不得使用 OpenCV uint8 Lab 近似。low-frequency 先逐像素转 Lab，再用 sigma=3、truncate=3 的 separable Gaussian 做 mask-normalized convolution，边界 reflect；分子卷积 Lab×safe_valid，分母卷积 safe_valid，分母不足则 unevaluable，invalid/protected 不得渗入。epsilon、精确矩阵、white point、float64 metric reduction 和实现 schema 都写入 threshold/config 并以 golden vectors 测试。

### 15.4 纵向 block 宽带拟合

- 纵向 block 高度初始为 16 px；
- 排除 invalid、protected、高梯度和饱和像素；
- seam 左右分别对低频 luminance/chroma 做稳健一次拟合；
- 外推到 seam，测 step 和 slope mismatch；
- 记录 actionable_block_count 和 unresolved_block_count。

这项指标是 M6.1 主验收，专门检测“seam 紧邻像素相似，但两侧宽区域是不同色块”的情况。

### 15.5 同场景点 validation 残差

利用 replay 在同一 canvas 坐标比较左右真实来源。在线性 RGB 中：

~~~text
Y = 0.0722 B + 0.7152 G + 0.2126 R
luma_error_stops
= abs(log2((Y_left + epsilon) / (Y_right + epsilon)))
~~~

每 pair、component、global 报告：

- signed median；
- absolute P50/P90/P95/P99；
- Q0、base Q1–Q3、Q4、photometric owner-only、P3；
- improvement；
- worst pair 和 regression pair。

### 15.6 全景低频漂移

对 P2、photometric owner-only 和 P3 生成：

- safe background robust L-star/a-star/b-star x-column profile；
- owner seam overlay；
- robust global x slope；
- seam-correlated step energy；
- source gain/bias 曲线；
- adjacent first difference 和 second difference；
- Q4 field amplitude/slope/curvature。

只限制相对 P2 新增的全局漂移，不能假设真实场景照明必须恒定。

### 15.7 Clipping、暗部和偏色

在相同 valid mask 上报告：

- 各 B/G/R 通道 low/high clipping before/after；
- clipping 绝对增量；
- P2 暗像素集合的 median/P95 dark lift；
- neutral low-chroma 区域 chroma drift；
- 最大 gain/bias；
- 相邻 source 参数跳变。

low/high clipping 阈值、分母，dark-region 在 P2 上的冻结定义，以及 dark lift median/P95 都必须进入 threshold schema。不得保留“held-out 明确改善即可突破 dark-lift 门”这类未数值化例外。

### 15.8 Blend 局部清晰度和 ghost

只在 secondary_weight 大于 0 的 ROI 测：

- owner-only 与 P3 的 Scharr energy ratio；
- gradient attenuation P50/P95；
- seam 法向 profile 的新增双峰；
- double-edge/ghost score；
- actual bandwidth；
- weight min/max；
- protected 邻域颜色扩散。

禁止继续使用全图 Laplacian 证明局部 blend 不糊。

### 15.9 必出视觉诊断

四组各输出一套：

1. P2 | photometric_owner_only | P3 | 8x_abs_delta 全图；
2. 按宽带 DeltaE 排序的 top-10 seam atlas，每项含 P2/photo/P3、seam 和 mask；
3. top-10 regression atlas；
4. source gain/bias/Q4 field 曲线，标 component/fallback 边界；
5. P2/photo/P3 robust L-star/a-star/b-star x-column profile，叠 owner seam；
6. P3 上 applied blend 像素高亮 overlay；
7. blend 局部 owner | blended | delta | weight atlas；
8. clipping/dark-lift atlas；
9. protected_only_diagnostic_atlas。

所有图旁必须标尺度、pair、数值、before/after 和 mask 语义，不能只给没有坐标和指标的肉眼图。

## 16. P3 hard audit v2

### 16.1 父级实体复核

verify 和 audit 都必须从当前 generation 重新读取并计算：

- P2 completion SHA；
- P2 result image SHA；
- P2 full primary provenance SHA；
- P2 pair transactions SHA；
- narrow replay manifest 和每个 asset SHA；
- photometric replay manifest 和每个 asset SHA。

不能只信 p2_parent_reference.json 或 completion 中自报的字符串。

### 16.2 光度审计

逐项验证：

- solution/schema/bounds/config SHA；
- source index 连续且 frame id 与 P2 精确对应；
- 每 source 恰有一个 transaction；
- Q0 当且仅当 gain=1、bias=0、field=0；
- gain/bias/field 有限且在界内；
- component、gauge、rank 和 fallback 连贯；
- identity_fallback_inside_solved_component_count=0；
- field shape、knot spacing、observed domain、taper、amplitude、slope、curvature合法；
- train/validation dtype、shape、互斥和 safe/valid 子集；
- train_validation_overlap_count=0；
- global sample masks 等于 pair masks 的确定性合并；
- parameter SHA、transaction id 和 formal render 使用值一致；
- clipping/dark-lift 统计存在且可重算。

### 16.3 Blend 审计

- plan 数量和顺序等于 replay pair；
- pair/transaction id 唯一；
- parent pair transaction 和 replay SHA 精确匹配；
- applied 是 safe/common/expected-valid/eligible 子集；
- applied 与 protected 交集为 0；
- B2–B4 确认 ineligible，B1 active/protected 交集为 0；
- global masks 等于 pair masks 的确定性合并，不能伪造为空；
- provenance weight 与 plan weight 逐值相等，不能只比 active mask；
- secondary frame/source/u/v 精确等于另一真实 source 的 replay 值；
- owner-only sentinel 全部正确；
- B0 active count=0；
- final selected flag 与实际 model/width/level 一致；
- conflict/shrink 后的 transaction 已重写。

### 16.4 像素内容审计

- 非 blend 像素的 visual_panorama 等于 photometric_owner_only；
- protected 像素两者相等；
- invalid 保持 invalid，不新增孔洞；
- Q0+B0 时 P3 等于 P2；
- actual changed count 与报告一致；
- PNG、NPZ dtype/shape/channel order 正确；
- compact ROI formal render witness 覆盖全部 owner 和 active pixel；
- 每 attempt、每 source formal remap invocation 精确为 1；正常累计 1，授权 fallback 累计 2，绝不大于 2；

hard audit 必须从原始 RGB file SHA、P2 frozen UV、canonical gain/bias、可选 Q4 field 和 blend plan 独立重算 RGB，不能只验证 provenance 与输出之间“自洽”：

- 对全部非 blend owner pixel，独立 streaming remap/apply，最终 uint8 逐像素等于 photometric_owner_only；
- 对全部 active B1 pixel，独立重算 primary、secondary 和离散 weight，逐像素等于 visual_panorama；
- 原始 source file SHA 与 session/input binding 一致；
- audit 路径记录 audit_remap_invocations，与 formal_remap_invocations 分开；
- 该 audit 是只读重放，不改变正式输出。

负向测试必须能检测只篡改一个 owner-only RGB 或只篡改一个 active blend RGB，而其它 provenance/weight 全不变。

### 16.5 Malformed 输入行为

所有 shape/count/schema 错误必须返回结构化 audit failure，不得因 broadcasting、zip strict 或 index error 抛出未分类异常。

### 16.6 Sealed verifier

verify_sealed_s13_p3 还必须：

- 校验当前实体 parent SHA；
- 校验 effective config SHA；
- 校验正式资产 allowlist；
- 拒绝 completion 未绑定的额外正式资产；
- 校验 solution、field、transactions、quality、performance、hard audit 的 schema 和 SHA；
- 确认 completion 最后原子写入；
- 确认 pending generation 清理状态。

## 17. 失败、fallback 和 pointer 状态机

按下列顺序实现，不得混用 quality 与 hard audit：

~~~text
load and verify sealed P2
→ extract evidence
→ choose Q0–Q3, optional authorized Q4, and B0–B1 through candidate gates
→ formal render once
→ write unsealed P3 assets
→ independent hard audit v2

hard audit pass:
  atomically write P3 completion
  current_latest = P3
  diagnostic quality may be pass or fail

first hard audit fail:
  classify failure
  candidate-dependent failure only:
    discard only unsealed P3 attempt
    rebuild exactly once as Q0+B0
    rerun hard audit
  parent/config/schema/provenance/asset tamper or malformed structure:
    do not fallback
    fail closed immediately

second hard audit fail:
  do not seal P3
  current_latest remains P2
  clear pending
  write structured failure
~~~

额外要求：

- candidate gate 失败只是选择更简单模型，不算 hard audit failure；
- 只有 gain/bias/field/blend weight/像素 compose 等 candidate-dependent 失败允许一次 Q0+B0 fallback；
- parent hash、config SHA、schema、P2 primary provenance、replay、asset allowlist、malformed shape/count、forbidden invocation 或原子状态失败不得用 identity rebuild 掩盖，必须立即 fail closed；
- diagnostic quality=false 不触发 identity rebuild；
- diagnostic quality=false 仍可有结构合法的 sealed P3；
- 但四组 engineering acceptance 未全过时，流程禁止开始 M7；
- current_reviewed 和 current_preview 不因 M6.1 自动变化。

## 18. 逐文件实施清单

### 18.1 配置和文档

| 文件 | 改动 |
| --- | --- |
| configs\video_candidates\s013\S013_output_first_progressive_dense_central_slit_m61_v1.yaml | 新增 M6.1 全部版本化参数 |
| configs\video_candidates\s013\quality_thresholds_m61_v1.json | metrics-only 后一次冻结的质量阈值 |
| configs\video_candidates\s013\candidate_manifest.json | threshold freeze 后新增 M6.1 entry，保留 v3；bootstrap 不写 placeholder |
| README.md | 说明 M6.1 候选、production-ineligible、M7 尚未实现 |

### 18.2 核心源码

| 文件 | 改动 |
| --- | --- |
| src\panorama_demo\video_s13_contract.py | identity registry、v4 schema、严格 config builder |
| src\panorama_demo\video_s13_bundle.py | 去除硬编码 v3 identity；新 P2/P3 asset binding |
| src\panorama_demo\video_s13_m5.py | 仅增加冻结 photometric replay sidecar，不改变 M5 决策 |
| src\panorama_demo\video_s13_replay.py | 新 sidecar 的写入、读取、hash 和验证 |
| src\panorama_demo\video_s13_photometric.py | tile split、分层样本、robust Q0–Q3、component selection |
| src\panorama_demo\video_s13_luminance_field.py | 无授权首版只实现 zero-field/disabled contract，拒绝实例化非零 Q4；经约束授权才增加 solver |
| src\panorama_demo\video_s13_blend.py | 移除 1px no-op、config 宽度、safe halo、收益与清晰度门 |
| src\panorama_demo\video_s13_visual_quality.py | quality v2 多尺度、Lab、tail、profile、clipping、ghost |
| src\panorama_demo\video_s13_p3_hard_audit.py | hard audit v2 与实体 parent 复核 |
| src\panorama_demo\video_s13_m6.py | 配置接线、两阶段决策、streaming formal render、实计时 |
| src\panorama_demo\video_s13_experiment.py | normal/resume/fallback 状态机和 config 传递 |

### 18.3 脚本

建议新增：

- scripts\prepare_s13_m61_p2.py：metrics_calibration role，给每个分支创建 P2-v4 sidecar 和 sealed completion；允许在正式 M6.1 runtime 之外确定性重放 P2 serialization，禁止 seal P3；
- scripts\calibrate_s13_m61_quality.py：只允许 Q0+B0 的 metrics-only baseline；
- scripts\summarize_s13_m61_validation.py：四组汇总、阈值、hash、状态；
- scripts\compare_s13_m61_reproducibility.py：忽略动态字段后的逐资产复现比较。

prepare 脚本必须显式接收 branch-specific parent generation/P2 输出目录和 photometric_evidence_config，拒绝覆盖 sealed v3；完成后输出 P2 canonical regression 与 sidecar SHA。随后 calibrate 脚本只 resume 该 sealed P2-v4，formal M4/M5 invocation=0。脚本不得直接改 candidate config；threshold freeze 必须作为一次显式、可审查的文件变更。

## 19. 单元、集成和负向测试矩阵

### 19.1 Contract 和 config

扩展 tests\test_video_s13_contract.py：

- v3 identity/hash/dispatch 保持不变；
- M6.1 identity 和 schema 精确匹配；
- 未知键、NaN、Inf、非法 fraction、非法 width、非法 level 被拒绝；
- normal/resume/fallback 得到同一 effective config SHA；
- 每个版本化配置参数有 runtime effect 测试；
- dead config 被测试发现。
- external threshold JSON 路径逃逸、缺失、内容篡改和 SHA 不匹配被拒绝。

扩展 tests\test_video_s13_dispatch.py 与 test_video_s13_bundle.py：

- 两个 S013 identity 均正确 dispatch；
- M6.1 generation 不写死 v3 id；
- 旧 generation 仍可 verify；
- manifest self-hash 正确。

### 19.2 Photometric evidence 和 Q0–Q3

扩展 tests\test_video_s13_photometric.py：

- canvas-global tile train/validation/guard 无像素、tile 和 halo 泄漏；
- 重叠 corridor 的同一 canvas 坐标分类一致；
- validation 不进入 mask 阈值或拟合，改变 validation 不改变 graph topology 或 solve 参数；
- 分层样本覆盖全部 y/shoulder/luminance bins；
- 不可靠 edge 不进入 graph，并有 reason；
- Q1 使用 Rec.709；
- synthetic gain 和 white-balance 可恢复；
- Huber 对 outlier 稳定；
- median 改善但 P95 恶化的候选被拒；
- global 好但最差 pair 恶化的候选被拒；
- clipping、dark、chroma、slope 回归分别被拒；
- 一个 source 超界不会形成 component 内 identity 断点；
- component 回退到更简单模型；
- Q3 fallback/model/实际参数一致；
- Q0 invariant 精确成立；
- disconnected components 独立。
- P2 photometric replay 缺失、数组篡改、重复/乱序 pair、未审计宽 UV 被拒；
- 候选不能通过改变 frozen safe/protected/evaluable mask 改善分数。

### 19.3 可选 Q4，仅在约束授权后

新增 tests\test_video_s13_luminance_field.py。无授权首版至少验证 Q4 config disabled、非零 field/solver 调用被拒、canonical field 全零；以下 solver 测试只在约束授权后启用：

- 恢复已知平滑 source-u field；
- equal input 选择 zero field；
- RGB 共乘且 chroma ratio 不变；
- 不恢复高频纹理；
- field 不依赖 y；
- knot spacing 不小于 64；
- zero-mean gauge；
- amplitude、slope、curvature 分别越界时回退；
- support 不足 field=0；
- observed range 外精确为 0；
- taper 连续；
- disconnected components 独立；
- validation 不参与拟合；
- Q4 gate 失败只回 Q0–Q3，不影响其它 component。

### 19.4 Blend

扩展 tests\test_video_s13_blend.py：

- 1px no-op 路径不存在；
- 2 px pixel-center 权重 golden；
- active 是 safe/common/valid/eligible 子集；
- 非 protected 但不 safe 仍不得 blend；
- active/protected 交集为 0；
- width=2 有非零且小于 0.5 的 secondary weight；
- zero support 报 ineligible，不报 applied；
- 没有 50/50 平台；
- local blur 或 ghost 风险触发缩窄/B0；
- B2–B4 明确 ineligible；
- global conflict 确定性缩窄或 B0；
- final selected 标志与实际 model 一致；
- weight active count 与 provenance 一致；
- Q0+B0 精确恒等。

### 19.5 Quality v2

扩展 tests\test_video_s13_visual_quality.py：

- seam 相邻像素相同但 8–16 px 宽带 step 明显，必须检出；
- 相同 luminance、不同 chroma，必须检出；
- protected 真实边缘只标 protected_only，不驱动光度；
- 仅 5%–10% 行恶化时 P95/max 检出；
- 全图 Laplacian 不变但 blend ROI 模糊，必须检出；
- double-edge/ghost 增加，必须检出；
- clipping、dark lift、global x slope 回归；
- Q0 在干净输入为 clean_noop；
- Q0 在 actionable baseline 为 unresolved；
- 样本不足为 unevaluable；
- worst pair 排序稳定；
- 同输入报告逐字段确定性；
- stage_seal_authority 永远为 false。
- clean baseline 从 0 或极小值恶化到绝对门仍失败；
- visible seam count 0→1 或 0→2 失败；
- before=0 的相对公式不产生除零、NaN 或假通过；
- insufficient-evidence 超 coverage limit 时全局为 unevaluable；
- macro/micro 均需通过。

### 19.6 Hard audit v2

扩展 tests\test_video_s13_p3_hard_audit.py，参数化篡改：

- 每个 P2 primary 字段；
- source/frame 顺序；
- solution/schema/bounds/config SHA；
- gain/bias/field 非有限或越界；
- Q0 但参数非 identity；
- component fallback 边界；
- field gauge/shape/domain/amplitude/slope/curvature；
- train-validation overlap；
- sample 越出 safe/valid；
- 伪造空 global mask；
- pair/global mask 不一致；
- active 位于 unsafe/protected/corridor 外；
- 非法启用 B2–B4；
- secondary frame/source/UV 虽有限但值错误；
- owner-only UV 非 NaN；
- plan/provenance weight 数值不一致；
- transaction id、parent SHA、pair count 错误；
- B0 非零 active；
- 非 active pixel final 与 owner-only 不一致；
- Q0+B0 不等于 P2；
- formal remap count 错误；
- P2 实体或祖先篡改；
- malformed shape/count 返回 failure，不抛异常。
- 单独篡改一个 owner-only RGB 被独立重放检出；
- 单独篡改一个 active B1 RGB 被独立重放检出。

### 19.7 M6 和 experiment 集成

扩展 tests\test_video_s13_m6.py 和 test_video_s13_experiment.py：

- normal/resume 都传同一 config；
- P3 completion 绑定 config/schema；
- candidate gate 失败回 Q0/B0 后仍可 seal；
- diagnostic false + hard true 仍 seal且不触发 identity rebuild；
- 首次 candidate-dependent hard fail 只重建一次；
- parent/config/schema/provenance/replay/malformed/forbidden/atomic failure 立即 fail closed，不重建；
- 两次 hard fail 后无 P3 completion、pointer 留 P2；
- pending 清理；
- current_reviewed/current_preview 不变；
- 历史 v3 P0/P1/P2 hash 不变；新 P2/v4 在 seal 后不变；
- M4/M5/Open3D/depth/TSDF/GraphCut invocation=0；
- 每 attempt 每 source formal remap=1，normal 累计1、授权 fallback 累计2；
- compact ROI 与 reference compose 最终 uint8 一致；
- illegal overwrite/resume 被拒；
- performance timing 不再硬编码 0；
- live memory accounting 包含全部 resident ROI 和 canvas。
- selected B1 但 final changed RGB count=0 时 canonicalize 为 B0；
- reproducibility comparator 只忽略动态 allowlist，不忽略模型/参数/mask/像素；
- quality true 不能绕过 hard audit，quality false 不能改变已冻结 selection；
- fallback remap counters 按 attempt 分开且累计不大于 2。

## 20. 四组真实验收

### 20.1 固定输入与输出

固定：

~~~text
fast:
  run_20260807_140140

slow:
  run_20260806_153033

branches:
  fast_direct
  fast_ignore_pose
  slow_direct
  slow_ignore_pose
~~~

输出必须写入新目录：

~~~text
artifacts\S013_M6_1_acceptance\
~~~

不得覆盖现有 M6 目录。允许在一次性的 P2-v4 preparation 阶段复用 trajectory cache，并生成新的宽 shoulder sidecar；该准备阶段可以从每个分支自身的 P0/P1/M5 封存结果确定性重放 M5 serialization，但不得从旧窄 replay 猜测或外推宽 UV。正式 M6.1 runtime 和 acceptance invocation 开始后，M4/M5 invocation 必须为 0。每个分支从其自身对应、已确认且 sealed 的 branch-specific P2/v4 parent 重放；同一分支两次复现使用同一个 P2 实体。

运行中硬要求：

- ORB/Open3D/depth/TSDF/mesh/GraphCut 调用均为 0；
- 所有历史 v3 P0/P1/P2 completion 未修改；
- 每分支新建且 seal 一份 P2_completion/v4；M6.1 开始后它不再修改；
- P2 image/provenance/pair transaction/旧 replay hash 与对应基线相同；
- 每 attempt 每 source formal remap 精确 1 次；无 fallback 累计1，授权 fallback 累计2且封存只绑定 final attempt。

### 20.2 结构验收

四组全部满足：

- P3 hard audit v2 pass；
- current_latest=P3；
- immutable primary provenance 与 P2 全字段相同，P3-owned transaction/secondary 字段按 v2 契约更新；
- protected secondary count=0；
- invalid/hole count 不增加；
- train-validation overlap=0；
- identity_fallback_inside_solved_component_count=0；
- config/code/input/parent/asset hash 完整绑定；
- Q0+B0 路径可逐像素复原 P2；
- performance 数值为实测；
- 未调用 forbidden backend。

### 20.3 视觉数值门

以下值是初始建议，必须在第 6 节 metrics-only 后冻结，不能看候选结果再调整：

不能用 after 小于等于 max(绝对门, 比例乘 before) 这种单式门。它会让很干净的 baseline 大幅恶化后仍通过，例如 before=0.2、after=2.0。必须先按 baseline_actionable 分支：

safe-clean baseline：

- 不要求材料性改善；
- final 状态可为 clean_noop；
- after 必须同时满足绝对 clean 门和相对/绝对非回归门；
- visible safe seam count 不得从 0 增至 1 或 2。

actionable baseline：

- after 必须满足绝对目标门；
- 同时相对 before 达到材料性改善；
- 同时每个 pair 满足非回归容差；
- final 状态才可为 materially_improved。

初始门限表：

| 指标 | 绝对目标门 | actionable 改善门 | clean/non-regression 门 |
| --- | ---: | ---: | ---: |
| wide-safe DeltaE P95 | 2.0 | 至少下降 15% | after-before 不大于 max(0.10 DeltaE, 0.01×before)，且 after 过绝对 clean 门 |
| wide-safe DeltaE median | 1.0 | 至少下降 5% | after-before 不大于 max(0.05 DeltaE, 0.01×before)，且 after 过绝对 clean 门 |
| safe-evaluable pair 的 block-P95 最大值 | 3.0 | 不要求统一比例 | after-before 不大于 max(0.25 DeltaE, 0.05×before) |
| 超出逐 pair 非回归容差的数量 | 0 | 0 | 0 |
| visible safe seam count，DeltaE00 大于 2 | 不多于 2 | after 不大于 min(2, floor(0.5×before)) | 不得增加 |
| high/low clipping 绝对新增 | 每通道 0.05 个百分点 | 不适用 | 同左 |
| safe dark L-star median/P95 lift | 1.0/按 baseline 冻结 | 不适用 | 无自动例外 |
| global/local gradient ratio | 全图不低于 0.95，blend ROI 不低于 0.92 | 不适用 | 同左 |
| ghost/double-edge count | 不增加 | 不适用 | 不增加 |

每个 metric 同时满足适用的 absolute target 和 non-regression delta；delta tolerance 使用 max(absolute tolerance, relative tolerance×before)，避免 before=0 被比例项锁死，也避免干净 baseline 恶化到宽松绝对门。绝对 clean/actionable 分类阈值由 metrics-only 文件版本化，并对边界值加单测。

regression pair 定义为：

~~~text
pair_after
> pair_before
  + max(pair_absolute_tolerance,
        pair_relative_tolerance * pair_before)
~~~

其中 pair metric 是冻结 safe blocks 的 per-pair block-P95，不是单像素 raw max。before=0 的公式必须有限且使用 absolute tolerance。visible seam 定义为 per-pair frozen block-P95 DeltaE00>2：clean baseline 只要求 after_count<=before_count；actionable baseline 要求 after_count<=min(2, floor(0.5×before_count))。所有 before/after 使用同一 frozen mask。

若 metrics-only 证明某一绝对值不适配固定数据，允许在候选启用前一次性修订并解释；不允许在候选运行后调门。

### 20.4 分支附加门

fast_direct：

- 当前 legacy max=11 若经 v2 证明完全是 protected-only，可进入 M7 清单；
- 若 metrics-only 冻结为 actionable，safe 宽带必须 materially_improved；若冻结为 clean，允许 clean_noop 但不得回归；
- 不能再用 Q0 加约 0.30% blend 宣称完成。

fast_ignore_pose：

- 当前 legacy max=8 只有证明是 protected-only 才可交 M7；
- 按冻结 baseline actionability 决定 materially_improved 或 clean_noop；
- 不得扩大 blend 侵入横梁、立柱和线缆。

slow_direct：

- P3 不得继续在 baseline actionable 时逐像素等于 P2 并报告 pass；
- 只有 baseline 已低于所有绝对门时，Q0+B0 才可作为 clean_noop。

slow_ignore_pose：

- pair 21 不得回归；
- legacy maximum luminance jump 必须从 9 回到不大于 2.25，P2 baseline 为 2；
- 当前 Q2 形式必须因 P95 和 pair 21 门被拒；
- 不允许再次出现约 60% 像素被大面积校正同时 quality=false；
- adjacent parameter step 必须过连续性门。

### 20.5 可重复性

相同代码、配置、P2 和 input 连续运行两次。排除明确列入 allowlist 的动态字段：

- generation id；
- timestamp；
- wall-clock timing；
- process id；
- peak RSS 的非确定性测量。

复现 harness 的比较分三级：

byte-identical：

- PNG 和 masks；
- integer provenance NPZ；
- parameters/field/weight 浮点 NPZ，要求固定 reduction 顺序、线程数和 dtype 后位级相同；
- sample split。

canonical normalized-equal：

- model/component 选择；
- blend plans；
- transaction canonical payload；
- quality 指标；
- hard audit decision；
- completion。

比较前只剔除明确 allowlist 的 generation id、timestamp、performance SHA、process id 和 pointer path；不能剔除模型、参数、mask、parent semantic identity、像素或 decision 字段。

dynamic-only：

- performance/timing/RSS；
- generation metadata；
- pointers。

dynamic-only 文件要求 schema、字段、有限非负值和关系正确，不要求原文件 SHA 相同。performance 及其 SHA 不得混入 canonical decision payload。浮点 JSON 使用固定 canonical 格式。

## 21. M6.1 与 M7 的严格分流

### 21.1 仍属于 M6.1，不能交给 M7

出现任一情况仍是系统性光度问题：

- residual 跨越至少 3 个连续 pair；
- safe visible seam 超过 pair 总数的 5%；
- safe wide-band Luma/DeltaE 门失败；
- 全局 x slope 或白平衡漂移门失败；
- 参数链出现无物理证据的突变；
- Q0 no-op 但 baseline 仍 actionable；
- blend 只靠侵入结构区才能改善。

四组未全部通过时，不得创建或运行 M7 pixel-changing 实现。

### 21.2 可以交给 M7

四组结构和视觉门全部通过，且残余问题同时满足：

- 每个分支的数量不超过 max(3, ceil(0.05 × 该分支 pair_count))；
- 是 source schedule 上非连续的孤立 pair；任意两个残余 pair index 相邻即不算孤立；
- 位于 protected 或明确结构风险区；
- 是几何、对象边界、遮挡、ghost 或 owner/seam 选择问题；
- safe 背景的光度接缝已经过门。

此时才从各组 protected_only_diagnostic_atlas 聚合生成：

~~~text
artifacts\S013_M6_1_acceptance\M7_handoff.json
artifacts\S013_M6_1_acceptance\M7_handoff_atlas\
~~~

handoff 是跨四分支聚合文件；每项必须写 branch、run id、generation id、P2 parent completion identity、pair index、canvas ROI、frame/source、risk reason、P2/P3 crop、相关 transaction SHA 和建议的 M7 问题类别。pair index 只在 branch 内唯一，不能单独作为全局 id。生成后停止，不得自动实现 M7。

### 21.3 停止盲调

如果连续两轮合法 M6.1 候选迭代：

- wide-safe P95 改善都小于 1%；
- 且仍未过门；

停止调阈值或扩大 blend，输出 root_cause_blocked_report.json，区分：

- 光度证据不足；
- 捕获曝光/白平衡非线性；
- protected 覆盖过高；
- 上游 geometry/owner；
- 当前模型族不适用。

请求人工决策，不得继续为了通过指标修改门。

## 22. 提交拆分和每步完成定义

Codex 应保持小提交、每步可验证；不要求自动 commit，但代码改动建议按以下边界组织。

### Commit A：版本和配置

- 新 identity/schema/parser 和 metrics_calibration bootstrap role；
- 不写 placeholder threshold 或最终 manifest entry；
- strict effective config；
- normal/resume/fallback 全接线；
- v3 回归测试。

完成定义：改变任何关键 YAML 参数会被运行时或 validator 观察到，所有 config 测试通过。

### Commit C：P2 photometric replay

- 宽 shoulder sidecar；
- v4 binding；
- P2 canonical hash regression。
- 最终 safe/protected 派生、train/validation/guard split 和 color metric schema。

完成定义：新增 evidence，但 P2 image/provenance/pair transaction/旧 replay 不变。

### Commit B：quality v2 metrics-only

- 在 Commit C 的最终 sidecar/mask/split/metric 定义上实现新测量和诊断资产；
- baseline 脚本；
- 四组 Q0+B0 metrics；
- threshold freeze 文件。
- 生成最终 candidate YAML/config SHA，并在 freeze 后新增 manifest entry。

完成定义：宽带亮条、同亮度异色和局部少数回归 synthetic 全部被检出，threshold 绑定最终 evidence schema。执行顺序固定为 A→C→B→D。

### Commit D：robust Q0–Q3

- tile split、分层抽样、edge gate；
- IRLS/component solve；
- non-degradation selection；
- fallback invariant。

完成定义：slow_ignore 类 synthetic 和真实 Q2 回归被拒绝。

### Commit E：可选 Q4，仅在约束授权后

- source-u field；
- constraints/taper；
- validation selection；
- tests。

完成定义：AGENTS/README/contract 已经同改授权；恢复 synthetic 低频场，不改变 chroma，不外推，无证据时精确为 0。没有授权时本 commit 跳过且 Q4 disabled。

### Commit F：blend

- 删除 1px no-op；
- 2px feather；
- B2–B4 MultiBand 显式保持 ineligible；
- active/protected/halo/conflict audit。

完成定义：每个 selected non-B0 都有非零 active 和可测改善，结构区仍 owner-only。

### Commit G：streaming render 与 provenance

- compact ROI；
- formal remap once；
- transaction v2；
- 实际 timing/memory。

完成定义：小图与 reference compose 一致，不再有约 1 GB corrected full-canvas dict。

### Commit H：hard audit、状态机和真实验收

- audit/completion v2；
- mutation tests；
- fallback 状态机；
- 四组两次运行；
- M7 分流或 blocked 报告。

完成定义：第 23 节全部通过，并输出明确结论。

## 23. 验证命令和最终完成定义

使用仓库指定 Python：

~~~powershell
$G305Python = 'D:\Panoramic_Camera\.conda\python.exe'

& $G305Python -m pytest -q tests/test_video_s13_contract.py tests/test_video_s13_dispatch.py tests/test_video_s13_bundle.py tests/test_video_s13_photometric.py tests/test_video_s13_luminance_field.py tests/test_video_s13_blend.py tests/test_video_s13_visual_quality.py tests/test_video_s13_p3_hard_audit.py tests/test_video_s13_m6.py tests/test_video_s13_experiment.py

ruff check src tests scripts
& $G305Python -m compileall -q src tests scripts
git diff --check
& $G305Python -m pytest -q
~~~

真实验收必须另外记录：

- 单元/合成测试状态；
- 四组固定历史数据状态；
- 每组模型/component/blend 选择；
- P2 regression；
- hard audit v2；
- visual quality v2；
- reproducibility；
- actual performance；
- forbidden invocation；
- M7 handoff 或 blocked 原因。

M6.1 只有在以下条件全部满足时才算完成：

1. 相关和全量测试通过。
2. v3 行为可复现且历史 artifacts 未改。
3. 配置全接线，无死参数。
4. 四组 P2 canonical 内容不变。
5. 四组 hard audit v2 全过。
6. 四组 visual engineering gates 全过。
7. slow_ignore_pose 不再制造 pair 21 新亮条。
8. fast 两组 safe 宽带接缝有材料性下降，而不是只增加少量 blended pixel。
9. protected 区完全 owner-only，无 ghost/blur 回归。
10. 两次运行除动态 allowlist 外逐 hash 一致。
11. 资源和 timing 是实测值。
12. 输出 M7 handoff 后停止，或明确报告为什么仍属于 M6.1。

## 24. 可直接交给 Codex 的执行指令

将以下内容与本文件路径一起交给 Codex：

~~~text
请在 D:\central_strip_Panoramic_Camera 实施
Panoramic_Camera_S013_M6_1_Photometric_Seam_Suppression_Codex_Implementation_Plan_v1.md。

先阅读 AGENTS.md、README.md、现有 S013 v3 config、M6 源码、测试以及
artifacts\S013_M6_acceptance 的四组证据。第一条命令必须是
git status --short。保留全部用户和历史 artifacts，不覆盖任何现有 M6 目录。

严格按文档的 A→C→B→D→可选E→F→G→H 顺序推进：先修版本化和配置接线，
再生成/封存 P2-v4 photometric sidecar 与最终 mask/split/metric 定义，然后
做 Q0+B0 metrics-only baseline 并冻结阈值；在此之前不得启用 Q1–Q3 或 B1。
Commit E 的 Q4 只有在明确授权并同改 AGENTS.md、README 和 contract 后才执行，
否则保持 disabled。B2–B4 MultiBand 在 M6.1 v1 明确 ineligible。

M6.1 仍生成 P3，只解决安全背景的系统性光度和极窄融合问题。正式 M6.1
runtime 不得重估 trajectory、M4、M5、geometry、seam、owner 或
P2 immutable primary provenance；P2-v4 preparation 只允许确定性生成 sidecar。
正式 runtime 不得调用 Open3D、depth、TSDF、mesh、GraphCut；不得实现 M7。

不要通过扩大融合带解决明显接缝。必须修复 aggregate median 误选、component
内单节点 identity fallback、1px blend no-op、弱 quality gate、hard audit
绑定缺口和虚假内存/计时报告。

每个阶段先补失败测试，再实现，再运行该阶段测试。最终运行文档第 23 节的
相关和全量验证，并在全新 artifacts\S013_M6_1_acceptance 下对固定四组数据
各跑两次。四组未全部通过前不要进入 M7。若两轮合法迭代仍改善不足，按文档
输出 root-cause blocked 报告，不得调松阈值或侵入 protected 区。

最终报告必须逐项列出：改动文件、测试、四组数值、P2 hash、hard audit、
quality 状态、模型和 blend 选择、复现 hash、实际性能、禁止调用计数以及
M7 分流结论。不要只说“测试通过”或“肉眼更好”。
~~~

---

本方案的核心判定很简单：系统性可见亮度/色度接缝必须在 M6.1 解决；M7 只接收少量、孤立、明确属于 protected 结构或几何的残余问题。M8 和 M9 都不会替 M6.1 改善图像。
