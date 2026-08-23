# Panoramic Camera S013 M6.1 最终融合实施方案 v3

## 光度接缝抑制、owner 亮度平台检测与安全窄融合

> 用途：直接交给 Codex 实施、测试和四组真实验收  
> 仓库：D:\central_strip_Panoramic_Camera  
> 当前分支：codex/video-realtime-seam-v6  
> M6 只读语义基线：c3aaa7ddbaf8aac5e00cded71ba9e4f1689b822a  
> 实施 identity：S013_output_first_progressive_dense_central_slit_m61_v2  
> 文档版本与实施 identity 版本相互独立；本次文档修订不要求再造 m61_v3  
> 当前 stage：P0 → P1 → P2 → P3；M6.1 仍生成 P3，不新增 P3_1  
> 适用范围：独立、production-ineligible 的 S013 视频候选实验  
> 固定真实数据：run_20260807_140140 与 run_20260806_153033  
> 严格停止：完成 M6.1、生成 handoff 或 blocked report 后停止，不实现 M7/M8/M9

本文件融合以下两份来源：

- 工作区详细实施合同 v1，SHA256：
  22230BFA9C7D843A8F7C19ECB7C76702EA276B80EB27E7525664BE531DEDEC1E
- 下载目录最终合并方案 v2，SHA256：
  C1F3EE37AEECB733E8AD6971DFF9491DFB87301F9B6BE9A7A560E03A5A10B36C

基线提交只用于比较语义。实施必须在当前工作树继续，保留用户和其它代理的全部改动；禁止 reset、破坏性 checkout、覆盖历史 artifacts 或把当前工作树退回该提交。

---

## 0. 最终融合结论

两份来源不是两套根本不同的算法。下载版 v2 主要是工作区 v1 的压缩重写，并新增了更成熟的门限治理与画质状态。最终取舍如下。

| 主题 | 最终采用 |
| --- | --- |
| 实施骨架 | 使用 v1 的完整配置、资产、replay、provenance、hard audit、负向测试、复现和资源合同 |
| 门限权限 | 吸收 v2 的分级思想，但拆成 A1 candidate safety、A2 P3 integrity、B visual non-regression、C engineering quality；每个指标再显式声明方向和 B/C 角色 |
| 最终画质 | 吸收 target + engineering ceiling，避免“已达到干净线仍因未下降 15% 失败” |
| 质量状态 | 新增 improved_near_target，并拆分 sealed_manual_review_status 与 effective_manual_review_status |
| owner 亮块 | 吸收 owner plateau 指标、曲线和 top-10 atlas |
| 低纹理清晰度 | 吸收 low_gradient_unevaluable，避免梯度分母接近零时假失败 |
| Q4 | 只保留 disabled/zero-field contract；M6.1 v2 不实现非零 Q4 |
| 融合 | 只开放 B0 与离散有效的 B1 2 px feather；B2–B4 明确 ineligible |
| 性能 | 60 s 在 M6.1 是软 SLA，不控制候选、P3 seal 或视觉完成；M9/production lock 后再决定是否硬化 |

下载版 v2 下列内容不原样采用：

1. aggregate P95 的 0.0015 linear 容差太宽。slow_ignore_pose 坏 Q2 的实际增量约为 0.001539，几乎擦线。
2. 固定 0.10/0.06 相邻参数 hard cap 缺少物理证据；改为 evidence-adaptive 连续性门。
3. 下载版只要求 B1 active/changed 非零，缺少实际收益门；本版恢复 immediate seam 的最小可检测改善。
4. small_improvement 在下载版状态机中未定义；本版删除该悬空状态。
5. improved_near_target 不可未经人工接受就算最终完成，也不可自动交给 M7。
6. 下载版的 60 s 硬失败与其最终完成定义、sealed performance 写入顺序冲突；本版修正。

M6.1 的主路径固定为：

~~~text
版本化、config bootstrap 与 v3 回归
→ branch-specific P2-v4 photometric replay sidecar
→ 最终 safe/protected、tile split、color metric 和 evaluability
→ 仅 Q0+B0 metrics-only、MDE 校准与一次性 threshold freeze
→ 创建最终 m61_v2 candidate YAML/manifest entry/effective config
→ robust Q0–Q3 component solve
→ fail-closed candidate selection
→ 仅 B0/B1 2px 安全窄融合
→ compact ROI / streaming 唯一正式渲染
→ diagnostic quality v2
→ independent P3 hard audit v2
→ 四组真实验收与复现重跑
→ M7 handoff、人工接受的 near-target 终态或 root-cause blocked report
→ 严格停止
~~~

不得通过扩大 feather、全图平滑、把 safe 改成 protected、重估 geometry/seam 或调用 M7 算法掩盖接缝。

---

## 1. 当前 M6 的固定证据

### 1.1 固定 generation

| 分支 | generation id |
| --- | --- |
| fast_direct | 20260812T140107-c4db600bf175 |
| fast_ignore_pose | 20260812T140107-fb6aafb7f542 |
| slow_direct | 20260812T140111-5a520ae4f223 |
| slow_ignore_pose | 20260812T140111-6319fb59b194 |

主要证据：

- artifacts\S013_M6_acceptance\M6_validation_summary.json
- artifacts\S013_M6_acceptance\P2_regression.json
- 各组 P3\photometric_solution.json
- 各组 P3\diagnostic_quality_report.json
- 各组 P3\blend_transactions.json
- 各组 P3\hard_audit.json

### 1.2 四组当前结果

| 分支 | pair 数 | 光度结果 | legacy seam median/P95/max | blend provenance px | 实际 RGB 改变 | 结论 |
| --- | ---: | --- | --- | ---: | ---: | --- |
| fast_direct | 84 | Q0×85 | 1/2/11 → 1/2/11 | 3023 | 2148 | 光度无改善，只有约 0.21% 像素真改变 |
| fast_ignore_pose | 84 | Q0×85 | 1/2/8 → 1/2/8 | 2687 | 1943 | 同样基本是光度 no-op |
| slow_direct | 108 | Q0×109 | 1/1/2 → 1/1/2 | 0 | 0 | P3 与 P2 逐像素相同 |
| slow_ignore_pose | 109 | Q0×22、Q2×88 | 1/2/2 → 1/2/9 | 0 | 514631 | Q2 改约 60.13% 像素并新造中央亮块 |

slow_ignore_pose 当前 Q2 相对 Q0：

~~~text
held-out median 改善约 2.52%
held-out P95 恶化约 9.04%
P95 绝对增量约 0.001539 linear
source 21 → 22 出现 identity / Q2 人造边界
red adjacent log-gain step 约 0.201
pair 21 legacy seam jump 2 → 9
~~~

M6.1 必须明确拒绝同类候选。

### 1.3 现有候选 P95 对门限的启示

| 分支/候选 | aggregate validation P95 变化 | 本版结论 |
| --- | ---: | --- |
| fast_direct Q1 | +1.17% | 可进入后续门，不因微小 tail 波动提前淘汰 |
| fast_ignore_pose Q1/Q2 | +14.12% / +12.30% | 拒绝 |
| slow_direct Q1 | +0.16% | 可进入后续门 |
| slow_ignore_pose Q1 | +1.89%，绝对约 +0.000321 | 可进入后续门 |
| slow_ignore_pose Q2 | +9.04%，绝对约 +0.001539 | 拒绝 |

这支持 aggregate P95 使用 2% 加 0.0005 linear 的相对/绝对组合，而不是 v1 的 1% 或下载版 v2 的 0.0015 linear。

### 1.4 当前 blend 的确定性缺陷

现有 width=1 feather 在整数像素中心上产生 0/1 alpha，经过 owner 侧选择后 secondary weight 全为 0。真实数据中 width=1 no-op pair：

- fast_direct：74；
- fast_ignore_pose：76；
- slow_direct：108；
- slow_ignore_pose：109。

M6.1 必须删除 1 px 候选；minimum_total_width_px=2 必须从 config 真正生效。零支持或最终 uint8 零变化的候选必须 canonicalize 为 B0。

### 1.5 当前资源报告不可信

现有 corrected full-canvas float32 cache 约占：

- fast：1.014–1.020 GB；
- slow：1.120–1.130 GB；
- 尚未包含 raw/cache 等其它常驻对象。

旧 performance 报告的 41–48 MB 漏算该字典，部分 timing 还硬编码为 0。M6.1 必须改为 compact ROI/streaming，并同时报告进程 RSS 与显式 live-array bytes。

---

## 2. 目标、非目标与永久不变量

### 2.1 M6.1 必须解决

- Q1–Q3 可表达的相邻 source 曝光、整体亮度、白平衡和色度差异；
- safe 背景中由 source 全局参数不连续造成的竖向亮度/色度条；
- source 参数突然跳变造成的 owner 亮度平台；
- seam 紧邻像素不明显但 4–64 px 宽带明显不同的问题；
- aggregate median 变好但 P95、worst pair 或局部视觉恶化的误选；
- 1 px blend 数学 no-op；
- YAML 未接入 normal/resume/fallback 的 dead config；
- 弱 visual quality、hard audit 和资源计量缺口。

M6.1 必须检测所有 safe 低频条，但只承诺解决 Q0–Q3 模型族可表达的部分。若残差确认为 source-u 空间场且 Q1–Q3 无法表达，必须输出 photometric_model_family_not_supported 或 low_frequency_field_authorization_required，不得假装完成。

### 2.2 M6.1 明确不解决

- geometry 错位、横线台阶、双边、ghost 的几何根因；
- object 被切断、seam 穿物体、错误 owner；
- 遮挡、disocclusion、深度层、透明/反光结构修复；
- trajectory、M4、M5、placement、motion、seam 或 source 选择重估；
- Depth、TSDF、Open3D、ORB-SLAM3、GraphCut、mesh、source rescue；
- M7 对象/结构级像素修复；
- production delivery、production lock、M8 或 M9。

上述问题只允许检测、保护、分类和 handoff，不得在 M6.1 修改其几何或 owner。

### 2.3 永久硬不变量

1. 历史 v3 config、generation 和 artifacts 不覆盖、不改写。
2. 旧 M6/P3 只作为人工与 acceptance 基线，不是新 candidate 的 runtime parent。
3. 每个新分支使用自己的 sealed P2-v4 parent，不得把四分支误称为同一 P2。
4. 新 P2-v4 的 canonical RGB、immutable primary provenance、pair transaction 和原 narrow replay 与对应 v3 P2 逐 hash 相同。
5. P2-v4 只增加 photometric replay/owner-domain/train-only graph-topology sidecar 与新的 completion binding；P3 开始后 P2-v4 不再改变。
6. 正式 M6.1 runtime 的 M4/M5、geometry、seam、ORB、Open3D、Depth、TSDF、GraphCut、mesh 调用均为 0。
7. geometry、seam、owner、placement、motion hypothesis 和 primary UV 不改变。
8. 每个有效像素最多两个真实 contributor；primary 始终占主导，secondary weight 必须为 0 或严格位于 0 与 0.5 之间。
9. protected、invalid、非 common-valid、非 expected-valid 或非 safe 像素始终 owner-only。
10. Q0+B0 必须使 visual_panorama.png 与 photometric_owner_only.png 解码后的 uint8 数组、valid/invalid mask 逐值等于父 P2 canonical PNG；派生 JPG 不参与 identity equality。
11. 禁止生成颜色、补洞、全图 blur、全景 flow、homography 或 Depth/TSDF 回色。
12. 每个 render attempt 内每 source formal calibrated inverse remap 精确 1 次。
13. 每个正式 invocation 的 formal attempt 恰为 1；候选安全退化必须在正式 render 前完成。hard audit 发现任何独立 replay/compose 不一致都视为实现完整性错误，不得以 Q0+B0 二次 render 掩盖。
14. independent audit remap 单独计数，不能冒充 formal remap。
15. diagnostic quality 与 performance 都不拥有 P3 seal/pointer 权限。
16. current_reviewed 与 current_preview 不由 M6.1 自动创建或修改。
17. Q4 非零空间光度场在当前 AGENTS.md 下未授权，必须保持 disabled。

---

## 3. Candidate、schema 与兼容策略

### 3.1 统一 identity

~~~text
candidate_id / algorithm_id:
  S013_output_first_progressive_dense_central_slit_m61_v2

implementation_id:
  s013_output_first_progressive_dense_central_slit_m61_v2_preview
~~~

文档 v3 只修正实施合同；由于 m61_v2 尚未落地，不再无意义创建 m61_v3 identity。

### 3.2 统一 schema

~~~text
contract: gemini305-video-s13-output-first/v4
p2_completion: gemini305-video-s13-p2-completion/v4
p2_photometric_replay: gemini305-video-s13-p2-photometric-replay/v1
p2_photometric_graph_topology: gemini305-video-s13-p2-photometric-graph-topology/v1
photometric_solution: gemini305-video-s13-photometric-solution/v2
photometric_transaction: gemini305-video-s13-photometric-transaction/v2
photometric_pair_evidence: gemini305-video-s13-photometric-pair-evidence/v2
luminance_field_disabled_contract: gemini305-video-s13-source-u-luminance-field/v1
blend_transaction: gemini305-video-s13-blend-transaction/v2
blend_transactions_manifest: gemini305-video-s13-blend-transactions/v2
blend_pair_masks_manifest: gemini305-video-s13-blend-pair-masks/v1
effective_config: gemini305-video-s13-m61-effective-config/v2
quality_thresholds: gemini305-video-s13-m61-quality-thresholds/v2
visual_quality: gemini305-video-s13-p3-diagnostic-quality/v2
p3_hard_audit: gemini305-video-s13-p3-hard-audit/v2
performance: gemini305-video-s13-p3-performance/v2
p3_completion: gemini305-video-s13-p3-visual-completion/v2
m61_diagnostics_manifest: gemini305-video-s13-m61-diagnostics-manifest/v1
m61_manual_review_manifest: gemini305-video-s13-m61-manual-review-manifest/v1
m61_manual_review_decision: gemini305-video-s13-m61-manual-review-decision/v1
m61_acceptance_summary: gemini305-video-s13-m61-acceptance-summary/v1
m61_acceptance_summary_manifest: gemini305-video-s13-m61-acceptance-summary-manifest/v1
m61_threshold_approval: gemini305-video-s13-m61-threshold-approval/v1
~~~

历史 v3 的 identity、parser、dispatch、schema、hash 和 sealed verifier 必须继续复现。

### 3.3 Manifest 时序

- bootstrap 阶段不得把 placeholder threshold SHA 写入最终 candidate manifest；
- 先完成 P2-v4 与 metrics-only；
- threshold v2 冻结后才生成最终 candidate YAML、config SHA 和 manifest entry；
- manifest self-hash 由现有 canonical 规范计算，不手填；
- 外部 threshold 路径逃逸、缺失、内容篡改或 SHA 不匹配必须 fail closed。

---

## 4. 权限模型

下载版把所有 A 类失败都描述为“淘汰候选”，但 parent/config/provenance 失败实际必须终止 stage。本版明确拆分：

| 层 | 负责内容 | 失败行为 |
| --- | --- | --- |
| A1 candidate safety | 参数有限/物理 bounds、rank、condition、component solve、模型支持域 | 淘汰当前候选，回退更简单 Q/B |
| A2 P3 integrity | parent、schema、config、provenance、asset、formal remap、forbidden invocation、像素可重放 | 任何失败立即 fail closed；不得在 hard audit 后重建像素 |
| B-pre candidate non-regression | 正式 render 前，sealed sidecar validation 上的 P95、worst pair、DeltaE、clipping、dark lift、predicted plateau、predicted global x-slope | 淘汰当前复杂候选，回退更简单候选 |
| B-post output non-regression | 唯一正式 P3 上实际 clipping、plateau、blur、ghost、slope 等材料性回归 | quality_state=regressed；不重选参数、不重新 render、不触发 identity rebuild |
| C engineering quality | 是否达到 M6.1 画质目标 | 不控制 seal；写 quality state、人工复核与阶段完成状态 |

performance 是独立工程维度，不属于 A1/A2/B/C，不改变像素选择与 seal。

quality report 必须明确：

~~~json
{
  "candidate_selection_authority": false,
  "stage_seal_authority": false,
  "contributes_to_engineering_acceptance": true,
  "sole_engineering_acceptance_authority": false
}
~~~

candidate selection 可使用与 quality v2 同源、已封存的 sidecar validation 指标，但必须在唯一正式 render 前完成。B-post 与 C 只观察正式输出，不得反向改参数。

---

## 5. Config bootstrap 与全路径接线

### 5.1 两个不可混淆的 config identity

第一阶段 photometric_evidence_config 只定义：

- P2 sidecar geometry support；
- diagnostic shoulder；
- upstream risk/support evidence；
- safe/protected 派生所需输入；
- tile/guard split；
- color metric；
- sampling/evaluability；
- train-only graph-topology 算法版本、edge minimum sample/tile/vertical coverage、inlier fraction、robust-scale、rank/condition、relation uncertainty、component ordering 与 split 规则。

它不包含 Q1–Q3 candidate threshold，不允许非 Q0/B0，不 seal P3。上述 topology gate 在 P2-v4 preparation 前冻结并进入 canonical photometric_evidence_config_sha256；candidate solver 不得拥有第二套 edge/topology gate。

第二阶段 p3_effective_config 在 threshold freeze 后由以下内容共同构造：

- 最终 candidate YAML；
- 完整 canonical threshold JSON 内容及 SHA；
- photometric evidence config SHA；
- solver/blend/quality config；
- zero-Q4 disabled contract。

生成 p3_effective_config_sha256。

### 5.2 Bootstrap 无环流程

~~~text
identity/schema/bootstrap role
→ prepare branch-specific sealed P2-v4
→ Q0+B0 metrics-only
→ freeze threshold asset
→ write final candidate YAML + manifest entry
→ build final effective config
→ resume same P2-v4 and run P3
~~~

prepare/calibrate role 不是 candidate manifest entry，不得 seal P3。禁止 placeholder、候选运行后改 sidecar或同一 threshold version。

### 5.3 严格 config 对象

建议定义 frozen 对象：

~~~text
S13PhotometricEvidenceConfig
S13PhotometricSolverConfig
S13PhotometricSelectionConfig
S13BlendConfig
S13VisualQualityConfig
S13LuminanceFieldDisabledConfig
S13M61EffectiveConfig
~~~

全部对象必须：

- 拒绝未知键、NaN、Inf、非法 fraction、负尺寸和冲突门限；
- validated 后序列化为 canonical JSON；
- 将阈值、MDE、coverage、physical bounds、continuity、blend、metric 定义纳入 SHA；
- normal、resume、fallback、audit、verifier 使用同一个不可变对象。

至少以下参数不得留为源码隐式常数：

- train/validation sample、tile、block、dominant-tile 和 coverage 门；
- tile seed、split schema 与 guard halo；
- Huber delta、IRLS iterations、convergence tolerance；
- rank、condition、robust scale、inlier fraction；
- identity/一阶/二阶 regularization；
- gain/bias bounds、canonical epsilon、MDE 和 actionability；
- candidate benefit、non-regression、blend/gradient/ghost；
- color transform、Gaussian、clipping/dark/plateau/slope；
- canvas/ROI/live-array/resident-source allocation cap。

唯一调用链：

~~~text
run_s13_experiment
→ load candidate config
→ load and verify threshold asset
→ build S13M61EffectiveConfig
→ run_s13_m6 with same object
→ audit_s13_p3_stage with same object
→ verify_sealed_s13_p3 with same config SHA
~~~

### 5.4 Dead config 测试

每个版本化参数必须证明修改它会改变 candidate eligibility、solver bound/regularization、blend 行为，或触发 config validation failure。不允许“成功解析但 runtime 从不读取”。

---

## 6. P2-v4 photometric replay sidecar

### 6.1 资产与字段

~~~text
P2\
  photometric_replay\
    manifest.json
    pair_0000.npz
    pair_0000.json
    owner_domain\
      manifest.json
      source_0000.npz
      source_0000.json
    graph_topology\
      manifest.json
      topology.json
    ...
~~~

root/pair manifest 绑定 pair 顺序、branch/run/generation、P1 parent completion SHA、P2 canonical RGB/provenance/pair-transaction/旧 replay SHA、evidence config SHA、每个 sidecar 文件 SHA、dtype、shape、坐标单位和 support 语义。sidecar 内任何 manifest 都不得绑定尚未生成的 P2_completion SHA。

每 pair 至少保存：

- pair index、左右 source/frame；
- seam_x_by_row 与 canvas row；
- diagnostic canvas x/y；
- 左右 source u/v；
- valid/common-valid/expected-valid；
- 经上游审计的 geometry support；
- upstream sealed risk/support bit；
- raw RGB file SHA；
- narrow replay 和相关 pair transaction SHA；
- border-clipped rows、实际 shoulder min/median/P95 coverage；
- rectangular bounds 与 row-valid mask。

owner_domain 每 source 至少保存 source/frame、P2 owner identity、owner canvas x/y、frozen source u/v、valid/expected-valid/upstream support、safe/protected、train/validation/guard、central stable-domain mask、seam-shoulder exclusion、L/a/b profile input、neighbor-owner relation、sample class、tile/block coverage、dtype/shape 和每数组 SHA。

owner_domain manifest 绑定 source 顺序、evidence config、P1 parent、P2 canonical 实体 SHA 和所有 owner-domain 资产 SHA。它只提供 pre-render predicted plateau/slope 的 hash-bound samples，不包含候选结果。

graph_topology 只从 train evidence 构造，保存 model-independent support/coverage edge acceptance、pair relation witness、component membership、source/frame 顺序和 canonical SHA；它不保存 Q1–Q3 参数，也不运行 candidate solve。Q1/Q2/Q3 特有的 design-matrix rank、condition 和 parameter identifiability 只在同一 frozen component 内淘汰对应模型，不得据此重建 edge 或 component。component actionability 只能从该 topology 与 frozen pair actionability 聚合。

最终 P2_completion/v4 单向绑定 photometric replay root、owner_domain 与 graph_topology manifest SHA。P3 threshold、parent reference 和 verifier 才能绑定已经封存的 P2_completion SHA；禁止任何 P2 内资产反向绑定其容器 completion，避免 hash 自引用。

sidecar 不保存“由 M6 根据 RGB 新计算的 protected 结论”作为上游事实。M6.1 的 safe/protected 必须从 sealed replay、raw RGB、upstream risk bit 和 evidence config 确定性派生并绑定 P3。

### 6.2 Shoulder 与禁止外推

最大 diagnostic shoulder 为 seam 每侧 64 canvas px，不是总宽 64，也不是 raw source-u 64。

当前 M5 map 在 candidate support 外可能把 local delta 置 0。此值不得被解释为已审计 correspondence。solver sample 必须满足：

~~~text
left UV in audited geometry support
AND
right UV in audited geometry support
~~~

宽 canvas shoulder 可以测 P2/P3 输出条带；没有双侧审计 UV 的位置不能进入 matched-source solve，必须标 insufficient_evidence。禁止猜测、外推或把 support 外 delta=0 当真。

### 6.3 Preparation 边界

历史 P2-v3 未含新 sidecar，允许一次 branch-specific P2-v4 preparation：

- 从该分支封存的 P0/P1/M5 结果确定性重放 serialization；
- 不改变 M5 决策、canonical P2、owner、pair transaction 或旧 replay；
- 不覆盖 sealed v3；
- 输出 P2 canonical regression 和 sidecar SHA；
- P2-v4 seal 后，正式 M6.1/acceptance invocation 中 M4/M5=0。

---

## 7. Safe、protected、split 与 evaluability

### 7.1 样本类别

general_safe：

- common-valid、expected-valid；
- 非饱和、非 invalid、非 protected；
- 低到中梯度；
- 左右梯度方向无明显冲突。

neutral_safe 是 general_safe 的低 chroma、稳定背景子集，用于 luminance 和 neutral chroma drift。

protected_only：

- 强结构、长线、管道、叶片边缘；
- 双边、反光、透明风险；
- 左右梯度方向明显不一致；
- upstream sealed risk bit 指向风险。

protected 不驱动 Q 参数，也不允许 blend。M6.1 不重新读取 depth。

### 7.2 Canvas-global split

初始 tile 为 16×16 canvas px，确定性分成 train、validation、excluded_guard。

要求：

- 同一 canvas tile 在所有重叠 pair 中分类一致；
- train/validation halo 不重叠；
- validation 不参与 mask threshold、graph topology、pair relation、robust scale、solver、regularization 或参数 bounds；
- validation 只做 candidate selection 和报告；
- train_validation_overlap_count=0。

### 7.3 分层采样

按 y bin、shoulder offset、luminance bin、neutral/non-neutral、source-u coverage 和 canvas-global tile 确定性分层，不得 flatten 后取前 N。记录 maximum dominant-tile fraction。

### 7.4 初始 evaluability 门

以下 provisional 值必须由 metrics-only、tile jackknife 与真实 coverage 一次性冻结：

~~~text
per-pair validation samples >= 128
independent validation tiles >= 8
vertical blocks >= 4
samples >= 256 → high_confidence

solver matched-source safe-evaluable pair fraction >= 80%
solver matched-source insufficient-evidence fraction <= 20%
~~~

上述 80%/20% 只控制需要双侧审计 UV 的 solver/candidate evidence，不是最终 output quality coverage。post-render seam quality 不需要 matched-source relation，只需要冻结 canvas shoulder、valid/safe/protected mask；每个 pair 必须最终归为 `quality_safe_evaluable`、`protected_only` 或 `quality_insufficient_evidence`。自动 visual completion 初始要求所有含 safe output shoulder 的 pair 都 quality_safe_evaluable，quality_insufficient_evidence safe pair count=0；若 metrics-only 证明边界裁切等不可避免，只有在候选前冻结一个明显小于 5% pairs 的显式 cap、逐 pair reason 和人工批准后才可例外，且不得把 solver 的 20% 沿用过来。超过 cap 时 quality_state=unevaluable。

如真实 sidecar 无法满足，只能在首次非 Q0 候选前修订 threshold version、说明原因并重新冻结。

aggregate 同时输出 pair-uniform macro 与 sample-weighted micro。两者都必须非回归；材料性收益只要求冻结的 actionable macro/composite 通过，不能要求 macro 和 micro 都显著改善。

---

## 8. Metrics-only、MDE 与阈值冻结

### 8.1 数据角色

四组固定数据是 M6.1 的 engineering regression/qualification set，不是假装从未使用过的外部泛化测试。它们可用于一次 Q0 baseline 与 threshold freeze。

若要宣称外部泛化验收，必须另留未参与 sidecar、MDE、threshold 或算法选择的 unseen 实机序列；这不属于本轮完成条件。

### 8.2 冻结顺序

~~~text
完成最终 P2-v4 replay
→ 完成最终 safe/protected
→ 完成 train/validation/guard
→ 完成 CIEDE2000、block、plateau、clipping、ghost 定义
→ 仅运行 Q0+B0
→ block-jackknife / tile-fold / synthetic no-op 与 known-transform 校准
→ 第一次调用只生成 proposed threshold 后退出
→ 暂停并向用户/维护者报告 baseline、MDE、proposed threshold
→ 人工只看 baseline 和固定物理含义并明确批准
→ 通过单独文件变更冻结 quality_thresholds_m61_v2.json
→ 把 canonical MDE 值、方法版本和四组 P2 SHA 内嵌 threshold
→ 绑定 threshold SHA、写最终 manifest entry
→ 第二次正式 candidate 调用只读 final threshold
→ 才允许 Q1/Q2/Q3/B1
~~~

输出：

~~~text
artifacts\S013_M6_1_metrics_baseline\
  quality_metrics_baseline.json
  evaluability_by_pair.json
  mde_calibration.json
  quality_thresholds_m61_v2.proposed.json
  baseline_atlases\
~~~

MDE 至少覆盖 aggregate linear residual median/P95、pair P95、DeltaE00、immediate seam、gradient/ghost detector、owner plateau 和 clipping fraction。

### 8.3 冻结纪律

calibrate CLI 只能写 proposed 文件并退出；不得在同一进程自动批准、写 final threshold、更新 manifest 后继续运行候选。未经用户/维护者明确批准，不得进入 Q1–Q3/B1。

批准必须形成只增不改的外部 witness：

~~~text
artifacts\S013_M6_1_metrics_baseline\threshold_approval.json
schema = gemini305-video-s13-m61-threshold-approval/v1
proposed_threshold_sha256
baseline_quality_manifest_sha256
mde_calibration_sha256
four_p2_parents = canonical array sorted by branch:
  branch
  run_id
  p2_completion_sha256
  p2_canonical_image_sha256
decision = approved | rejected
approver
approved_at
reason
~~~

final threshold、candidate manifest、effective config 和最终报告都绑定 approved witness SHA。缺失、rejected、proposed SHA 不匹配、malformed 或被覆盖时不得写 final threshold/manifest，也不得启动 Q1–Q3/B1。批准 witness 不进入 proposed threshold 自身，避免自引用。

final threshold JSON 必须内嵌 canonical MDE values、MDE method/schema、baseline P2 completion/image SHAs、baseline quality asset SHAs 和 actionability；effective config 与 P3 completion 绑定该同一 threshold 实体。mde_calibration.json 是审计证据，不是可独立漂移的第二真相源。

### 8.4 Baseline actionability

metrics-only 为每 metric、pair 和 branch 冻结：

~~~text
baseline_actionability_by_metric
baseline_actionability_by_pair
baseline_actionability_by_branch
~~~

每个 metric 在 threshold schema 固定 lower_is_better 或 higher_is_better、target/ceiling comparator 和 actionability comparator。初始判定对 lower-is-better 为：

~~~text
metric actionable
= before 超过对应 target
  + max(actionability_relative_margin * target,
        MDE_metric_abs)

pair actionable
= 任一 applicable、required、quality_role=C_target_ceiling
  的 safe metric actionable

branch actionable
= 任一 applicable、required、quality_role=C_target_ceiling
  的 branch metric actionable
   OR actionable pair fraction 超过冻结阈值
~~~

higher-is-better 使用方向相反、量纲一致的 margin 公式。B_nonreg_only 指标不参与 pair/component/branch baseline actionability，只负责材料性回归门。边界、方向、每个 metric 的唯一 quality_role、required metric 集合和 pair-fraction 必须进入 threshold schema 并有测试。候选运行后不得重新分类，protected_only/insufficient_evidence 不得冒充 non-actionable。

component topology 只由 train evidence 构造，因此不得在 topology 产生前伪造或冻结 `baseline_actionability_by_component`。Step 2 先生成并封存 train-only `graph_topology` sidecar，记录 edge acceptance、component membership 与 canonical SHA；Step 3 根据该 sidecar 和已冻结的 pair actionability 确定性聚合 component actionability，并把 topology SHA 与聚合结果写入 final threshold。后续候选 solve 不得改变 membership。

第一次正式启用 Q1–Q3/B1 后：

- 同一 threshold v2 不得修改；
- 若算法或阈值必须改变，bump threshold version、写 reason，四组整套重跑；
- 修改后的轮次属于 development，不得继续声称是同一个 frozen acceptance；
- 禁止用候选结果调松 safe/protected 或 evaluability。

---

## 9. Robust Q0–Q3 component solve

### 9.1 模型

~~~text
Q0_identity
Q1_scalar_luminance_gain
Q2_rgb_diagonal_gain
Q3_bounded_rgb_gain_bias
~~~

Q1 使用 linear Rec.709：

~~~text
Y = 0.0722 B + 0.7152 G + 0.2126 R
~~~

provisional physical bounds：

~~~yaml
minimum_gain: 0.80
maximum_gain: 1.25
maximum_absolute_bias_linear: 0.03
~~~

这些是物理安全初值，不是由当前四组已证明的最佳值；必须版本化。候选运行后不得为通过验收而继续放宽。

### 9.2 Train-only frozen graph edge

edge 是否进入 graph 只在 Step 2 topology preparation 使用 train 决定：

- capped sample count；
- independent tile/vertical coverage；
- inlier fraction；
- robust scale；
- condition/rank；
- pair relation uncertainty。

validation 不得影响 edge topology、robust scale、regularization 或 solve。Step 4 只能加载并验证 sealed graph_topology SHA/membership；当前 evidence 若与它不一致是 integrity failure，不得重选 edge。

### 9.3 Solver

每 connected component：

- Huber/IRLS；
- deterministic variable order 与 reduction order；
- bounded projected solve；
- identity regularization；
- 一阶、二阶平滑；
- 固定 gauge；
- 显式 rank/condition/iteration/convergence；
- NumPy deterministic 实现优先，不假设 SciPy 必然存在。

### 9.4 Component fallback

禁止：

~~~text
一个 source 越界
→ 只把该 source 改成 identity
→ 在 component 内制造参数边界
~~~

规则：

~~~text
candidate solve 合法
→ 整个 component 使用

candidate solve 不合法
→ 整个 component 回退到更简单模型

train edge evidence 不足
→ 只能在 Step 2 topology preparation 时于该 edge 预先拆 component
~~~

硬要求：

~~~text
identity_fallback_inside_solved_component_count = 0
~~~

非 Q0 参数全部低于 canonical epsilon 时必须 canonicalize 为 Q0。

### 9.5 参数连续性

初始 warning/complexity penalty：

~~~text
adjacent log-luminance step > 0.06
adjacent log-chroma-ratio step > 0.04
~~~

不得仅凭固定 0.10/0.06 无条件 hard reject。真正的 hard reject 为：

~~~text
parameter outside physical bounds
OR
measured relation 的置信区间不含 0
AND abs(measured_pair_relation) > calibrated_step_MDE
AND step 与 train pair relation 符号不一致
OR
abs(parameter_step - measured_pair_relation)
  > max(calibrated_step_MDE, 3 * relation_uncertainty)
OR
形成 unsupported identity/complex-model 边界
OR
触发 pre-render validation material regression
~~~

Step 4 发现证据、edge 或 membership 与 sealed topology 不一致时 fail closed；不得拆 component。模型特有的 rank/condition 不足只淘汰该模型并回退更简单模型。validation 可以淘汰已解候选，但不得反过来重选正则、support 或重新拟合。正式 output 的材料性回归属于 B-post，只产生 regressed 状态，不在渲染后追溯修改候选。

报告 adjacent log luminance/chroma P50/P95/max、relation uncertainty、evidence-supported/unsupported、参数一阶/二阶差分和 component/fallback 边界。

---

## 10. Q4 的最终决定

M6.1 v2 默认不实现非零 Q4。

保留：

- gemini305-video-s13-source-u-luminance-field/v1 schema；
- canonical zero field；
- disabled-state config；
- 非零 field/solver 调用被拒的测试。

不保留：

- 非零 source-u solver；
- panorama-domain 二维亮度场；
- y-dependent field；
- 用 field 修复 chroma。

只有满足以下条件后才能另行授权：

1. robust Q0–Q3、evidence、selection、B1 和 hard audit 已完成；
2. frozen quality 证明 safe 残差是连续低频 luminance，不是 chroma/geometry/owner/object；
3. 用户或维护者明确授权；
4. AGENTS.md、README、contract、config、schema 和测试在同一改动更新。

没有授权时，残差输出 low_frequency_field_authorization_required 或 photometric_model_family_not_supported，不能偷偷启用 Q4。

---

## 11. Fail-closed candidate selection

### 11.1 精确非回归公式

第 11 节为每个当前 component 独立选择候选。除明确标为 branch/global audit 的指标外，所有 before/after、aggregate、macro、pair set、predicted target 与 benefit 都只在该 component 冻结的 source/pair/sample 子集上计算；不得用另一 component 的收益掩盖当前 component。branch/global 指标可以淘汰产生全局材料性回归的候选，但不能为单个 component 贡献收益。

aggregate validation P95：

~~~text
after <= before + max(0.02 * before, 0.0005 linear)
~~~

per-pair validation P95：

~~~text
after <= before + max(0.05 * before, 0.0010 linear)
~~~

predicted low-frequency DeltaE00：

~~~text
after <= before + max(0.10 * before, 0.5 DeltaE00)
~~~

所有公式都必须保存 before、after、relative tolerance、absolute tolerance 和 decision；max 表示容差二者取大，再加到 before 上。

material pair regression 定义为任一：

~~~text
pair P95 超过上述容差
OR
DeltaE00 增量 > max(0.10 * before, 0.5)
OR
新增 severe clipping / dark lift / ghost / blur / owner plateau
~~~

要求 material_pair_regression_count=0。所有微小正增量仍记录 raw_regression_count，但不控制候选。

### 11.2 Candidate benefit 不是单一 median

复杂候选的 candidate_benefit_pass 为以下至少一项：

~~~text
before_aggregate_median - after_aggregate_median
  >= max(0.01 * before_aggregate_median,
         MDE_median_abs)
OR
before_actionable_macro_p95 - after_actionable_macro_p95
  >= max(0.05 * before_actionable_macro_p95,
         MDE_p95_abs)
OR
before_predicted_plateau - after_predicted_plateau
  >= max(0.10 * before_predicted_plateau,
         0.5 DeltaE00,
         MDE_plateau_abs)
~~~

同时必须满足全部 A1 与 B-pre 非回归门。上式左侧与右侧单位一致；百分比必须先乘 before，MDE 必须是该 metric 的绝对量。这样允许局部少数严重 seam 的安全改善，不要求全局 median 必然下降 1%。

最终候选收益条件统一为：

~~~text
candidate_acceptance_benefit
= predicted_all_applicable_targets_pass
  OR candidate_benefit_pass
~~~

predicted_all_applicable_targets_pass 只能使用 sidecar validation/owner-domain samples。若预测已达到全部 applicable target，不再强迫其相对 Q0 的额外收益；但更复杂模型替代一个已经合格的更简单模型仍受下述 complexity gate 约束。

复杂度额外收益门固定为：

~~~text
complex_candidate may replace simple_candidate only if
  simple_candidate fails any predicted applicable C target
  AND complex_candidate passes all predicted applicable C targets
OR
  frozen_selection_score(simple) - frozen_selection_score(complex)
    >= MDE_selection_score_abs
~~~

`frozen_selection_score` 必须在 threshold asset 中唯一规定为各 applicable C metric 的 direction-normalized、按 frozen scale/MDE 标准化后的加权和；权重、scale、missing/evaluability 处理、MDE_selection_score_abs 与 tie epsilon 都在 metrics-only 阶段冻结。两者都合格且 score 差落在 MDE 内时必须选择更简单模型；不得用源码隐式权重或模型编号打破实质性平局。

### 11.3 选择顺序

每 component 候选依次通过：

1. 参数、rank、condition、coverage 和 physical bounds；
2. aggregate P95；
3. per-pair material regression；
4. predicted low-frequency DeltaE；
5. clipping、dark lift、neutral chroma drift；
6. predicted owner plateau；
7. predicted global x-slope non-regression；
8. parameter continuity；
9. candidate_acceptance_benefit；
10. 上述冻结 complexity gate。

复杂度顺序 Q0 < Q1 < Q2 < Q3。分数落在 MDE 内时选择更简单模型。报告必须保留所有 rejected candidate、逐门数值和 reason。

各 component winner 组装为 whole-branch plan 后，必须在同一 sealed compact validation/owner-domain evidence 上再运行一次 branch-level A1/B-pre（aggregate tail、clipping、dark/chroma、predicted plateau、predicted global x-slope 等材料性非回归）。同时计算 predicted C_target/C_ceiling/quality_minimum_benefit，但 C target 本身不是 rollback 门；计划达到 predicted target，或在 predicted ceiling 内且达到 minimum benefit，都可进入唯一正式 render。这里对 ceiling/benefit 的预测副本只是 B-pre 的复杂候选安全策略，不赋予 post-render C 类反向改参数的权限。只有组合效应违反 A1/B-pre、超出 predicted ceiling 或 near-target benefit 不足时，才按 threshold 中冻结的确定性 rollback key 逐 component 降一级并重算 whole-branch gate：

~~~text
rollback priority:
  predicted global harm descending
  then component benefit ascending
  then complexity descending
  then component id ascending
~~~

直到全分支通过或全部回到 Q0；保存每次 plan SHA、降级 component、before/after、reason 和最终 rollback trace。若 whole-branch 已全部为 Q0+B0，只要 identity 的 A1/B-pre non-regression 通过，就始终允许执行唯一 formal render；此时 predicted C target/ceiling/benefit 失败只预置并最终产生 quality_state=unresolved，不再阻止 render。该过程只重算 compact prediction，不 formal render、不改 frozen topology/threshold，也不得用另一个 component 的收益抵消材料性回归。

当前 slow_ignore_pose Q2 必须至少因 aggregate P95、pair 21、parameter boundary 或 owner plateau 中一项被拒，正常应多项同时失败。

---

## 12. Owner plateau 专项检测

### 12.1 指标

每个 owner 区域排除左右 seam shoulder，在中央稳定域测：

~~~text
owner_plateau_delta_L
owner_plateau_delta_a
owner_plateau_delta_b
owner_plateau_DeltaE00
owner_plateau_width_px
owner_plateau_signed_delta
owner_first_difference
owner_second_difference
~~~

与左右相邻 owner 的 robust low-frequency profile 比较，检测完整 owner 整体偏亮、偏暗或偏色。

### 12.2 Pre-render 与 post-render 权限

pre-render candidate gate：

- 只能使用 P2 sidecar 中 hash-bound 的 owner-domain audit samples；
- 计算 predicted owner plateau；
- 不得生成第二张 full-resolution panorama。

post-render quality：

- 在唯一正式 photometric_owner_only/P3 上测 actual plateau；
- 生成 atlas；
- 不得反向修改 selection。

### 12.3 诊断

输出 top_10_owner_plateaus_atlas，每项包含 P2、旧 M6 P3、M6.1 photometric owner-only、M6.1 P3、owner boundaries、gain/bias、x profile 和 plateau metrics。

---

## 13. B0/B1 安全窄融合

### 13.1 候选

正式只允许 B0_owner_only 与 B1_feather_2px。B2_multiband_4px、B3_multiband_6px、B4_multiband_8px 明确 ineligible。

MultiBand pyramid 会读取邻域，而当前 provenance 只表达同像素 primary/secondary contributor，不能精确绑定 pyramid 邻域读取。未升级 provenance/schema 前不得启用。

### 13.2 Mask 语义

每 pair 必须分别输出：

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
= pair_safe
AND pair_common_valid
AND expected_valid
AND NOT global_protected

applied_blend_mask
= secondary_weight > 0
~~~

不得继续用一个 safe_blend_mask 混淆 candidate-safe 与实际可融合。

### 13.3 B1 eligibility 与收益门

B1 必须同时满足：

~~~text
active_count > 0
predicted_uint8_changed_count > 0
immediate actionable seam 改善
  >= max(0.02 * before, MDE_immediate)
wide-band non-regression
predicted clipping non-regression
gain-normalized sharpness non-regression
material ghost/double-edge regression count = 0
full footprint 与 protected 交集 = 0
与相邻 pair 无 contributor/corridor 冲突
~~~

2% 是最低可检测收益初值，5% 是 target，不统一作为硬门。最终值由 metrics-only MDE 冻结。

正式 uint8 compose 后 actual_changed_rgb_count=0 时，在同一 attempt 内确定性 canonicalize 为 B0：清零该 pair active/weight/secondary，重建 plan、transaction、per-pair/global masks、manifest、IDs、weight map 和 provenance 后再运行 quality/audit；不重跑 formal remap，也不产生第二张 panorama。transaction 不得声称 applied B1。blend pixel fraction 不设下限，0 是合法结果。

### 13.4 离散权重与冲突

- 删除 width=1；
- width=2 必须有 pixel-center golden test；
- secondary weight 严格小于 0.5；
- 不允许 50/50 平台；
- plan、weight map、provenance 和 final transaction 逐值一致。

冲突 pair 按结构风险更低、evidence 更强、predicted benefit 更高、pair index 的确定性 key 排序。冲突时 B1 直接回 B0；M6.1 没有更窄的 1 px 候选。最终重新生成 transaction。

---

## 14. 唯一正式渲染、资源与性能

### 14.1 两阶段与 streaming

候选阶段只使用 sidecar/compact samples，不生成多个 full-resolution panorama。参数与 blend plan 冻结后只进行一个正式 render attempt。

每 source union ROI：

~~~text
owner ROI
+ active B1 corridor
+ interpolation halo
~~~

目标常驻 2–3 个相邻 source ROI，硬上限不得超过项目允许的 5 个 resident strips。禁止 source_count × full_canvas float32 cache。

所有 shape、stride、canvas/ROI bytes 用 checked arithmetic，在 allocation 前验证 canvas 200 MP 契约、ROI bounds、live-array cap、resident source cap 和 dtype/channel multiplier。

small reference full-canvas compose 与 compact ROI 的最终 uint8 必须逐像素一致。

### 14.2 Remap 计数

~~~text
每 invocation、每 source formal remap = 1
每 invocation formal attempt = 1
formal attempt != 1 = hard failure
audit_remap 单独计数
~~~

记录 attempt_id 和 final witness。任何正式 render 后的失败都不得在同一 invocation 再 render Q0+B0；重新运行必须是新的 immutable generation/invocation，并保留旧 failure witness。

### 14.3 Performance 资产

P3 内 performance.json 是 pre-audit sealed asset，实测：

~~~text
evidence_seconds
sample_seconds
solve_seconds
selection_seconds
formal_decode_seconds
formal_remap_seconds
owner_compose_seconds
blend_compose_seconds
quality_seconds
pre_audit_stage_total_seconds

formal_remap_invocations

peak_process_rss_bytes
peak_live_array_bytes
resident_source_roi_peak
~~~

不得把未测阶段硬编码为 0。

performance.json 在 hard audit 前写完并 hash-bound，pre_audit_stage_total_seconds 截止该文件原子写开始前，不包含 hard audit，也不得包含尚未发生的 audit_remap_invocations、audit_seconds、end-to-end 或 target_met。

hard audit 自己在 hard_audit.json 记录 audit_remap_invocations、audit_seconds、audit_peak_process_rss_bytes、audit_peak_live_array_bytes、audit_resident_source_roi_peak 与 audit_allocation_cap_decision。outer experiment/acceptance harness 在 P3_completion 原子写、pending promote 和 sealed verifier 返回后，把 hard_audit_seconds、end_to_end_through_completion_seconds、`whole_invocation_peak_* = max(pre_audit_peak_*, audit_peak_*)` 与 performance_target_met 写入 P3 目录外的 acceptance summary。外层值不反写 sealed P3，避免 performance↔audit↔completion 自引用。

每个 branch/generation 使用不可变、可追加的 versioned summary chain：

~~~text
artifacts\S013_M6_1_acceptance\summaries\
  <branch>\<generation>\
    manifest.json
    summary_0000.json
    summary_0001.json
    ...
~~~

summary 绑定 P2/P3 completion、effective config、threshold approval、quality、hard audit、diagnostics manifest、manual review manifest（若 applicable）、reproducibility partner 与所有外层 timing。`summary_0000.json` 先以 pending 文件写全、校验后原子 rename；人工 review 等外层状态改变时只追加 `summary_NNNN.json`，每个 N>0 payload 必须显式保存 `supersedes_summary_sha256` 并引用前一有效链头。payload 文件永不覆盖。manifest 记录所有 payload SHA、严格连续序号和唯一 `effective_head_sha256`；更新 manifest 时写新 pending manifest 后原子替换 pointer-like manifest，旧 manifest SHA 进入审计日志。多个有效未串联 summary、序号间隙、分叉、head 不在链尾或 manifest 冲突时 effective status=invalid。

### 14.4 60 s 语义

sealed performance.json 只记录 performance_target_seconds=60；P3 外 acceptance summary 才记录 performance_target_met。M6.1 中它是软 SLA：

- 不改变 Q/B 选择；
- 不触发 Q0；
- 不阻止 P3 seal；
- 不改变 visual quality state；
- 不阻止报告 M6.1 视觉完成；
- 必须在最终报告显式列出。

到 M9/production lock 固定硬件、线程、CUDA mode、计时边界和 atlas 是否排除后，再决定是否硬化。

---

## 15. P3 正式资产与 provenance

### 15.1 Canonical asset tree

~~~text
P3\
  visual_panorama.png
  photometric_owner_only.png
  p3_pixel_provenance.npz
  effective_m61_config.json
  photometric_solution.json
  photometric_solution.npz
  luminance_fields.json
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

P3 内禁止放 atlas、audit-only alias 或其它 diagnostics。第 16.10 节全部诊断都写到 P3 外：

~~~text
artifacts\S013_M6_1_acceptance\diagnostics\
  manifest.json
  <branch>\<generation>\...
~~~

外层 diagnostics manifest 绑定 branch、generation、P3 completion SHA、quality SHA、每个诊断文件 SHA、图像尺度与动态字段 allowlist；acceptance summary 再绑定该 manifest SHA。诊断只能由 sealed P3 确定性派生，不得反写候选选择、quality、completion 或 current_latest。现有 stage machinery 的 canonical 文件名优先，sealed P3 allowlist 拒绝 completion 未绑定的额外正式资产。

photometric_pair_evidence 每 pair NPZ 保存 train/validation/guard、sample class、canvas/source coordinates、safe/protected/support masks 与 metric inputs；JSON 保存 counts/reasons/metrics；manifest 绑定顺序、dtype、shape 和 SHA。

blend_pair_masks 每 pair NPZ 保存 safe、protected、eligible、active、weight、corridor bounds；全局 PNG 只是诊断 union，不能替代 per-pair 证据。

luminance_fields.json 声明 q4_state=disabled、schema、source/frame 顺序、NPZ dtype/shape、canonical-zero invariant 与数组 SHA。luminance_fields.npz 按 source 顺序保存零长度或固定 shape 的全零 field；二者都由 completion 绑定。禁止只发布无 manifest 语义的匿名 NPZ。

### 15.2 Transaction

每 source photometric transaction 绑定：

- source index/真实 frame id；
- component/model；
- gain/bias SHA；
- canonical zero field SHA；
- P2 source identity；
- effective config/solution SHA；
- formal ROI/remap witness；
- Q0 invariant。

每 pair blend transaction 绑定：

- pair 与左右 source/frame；
- parent P2 pair transaction；
- narrow/photometric replay SHA；
- solution/config SHA；
- candidate audit 列表；
- selected model/width；
- safe/protected/eligible/active/weight SHA；
- secondary source/frame/UV SHA；
- active 与 actual changed count；
- conflict/fallback reason。

NPZ 保留现有 int32 字段名 photometric_transaction_id 和 blend_transaction_id，其 v2 语义为 manifest table index。JSON manifest 提供 index → canonical transaction SHA256。不能把 SHA 塞入 int32，也不能让 index 冒充内容身份。

### 15.3 Primary 与 secondary

immutable primary fields 逐数组等于 P2：

- owner source/frame；
- source-u/source-v；
- valid；
- assignment；
- selected motion hypothesis；
- placement；
- geometry transaction；
- seam transaction。

P3-owned：

- photometric_transaction_id；
- blend_transaction_id；
- secondary_frame_id；
- secondary_source_index；
- secondary_source_u/v；
- secondary_weight。

owner-only sentinel：

~~~text
secondary frame/source/transaction = -1
secondary u/v = NaN
secondary weight = 0
~~~

active secondary 必须逐值等于 replay 中相邻另一真实 source。

---

## 16. Quality v2

### 16.1 无矛盾状态机

状态：

~~~text
clean_noop
clean_nonregressed
materially_improved
improved_near_target
unresolved
regressed
unevaluable
~~~

封存的人工状态：

~~~text
not_required
pending
~~~

外层派生的 effective 人工状态：

~~~text
not_required
pending
accepted
rejected
invalid
~~~

固定逻辑：

~~~text
coverage 不足
→ unevaluable

任一 material B 类回归
→ regressed

baseline 非 actionable
且 C_target_pass
且非回归
且 Q0+B0 / decoded pixels 精确等于 P2
→ clean_noop

baseline 非 actionable
且 C_target_pass
且非回归
且 decoded pixels 不再精确等于 P2
→ clean_nonregressed

baseline actionable
且 C_target_pass
且非回归
→ materially_improved
# 一旦跨过 target，不再强迫额外 10%–15% 改善

baseline actionable
且非 C_target_pass、但 C_ceiling_pass
且达到 quality_minimum_benefit_pass
且非回归
→ improved_near_target
→ sealed_manual_review_status=pending

其它
→ unresolved
~~~

P3 内的 sealed_manual_review_status 永远按正式运行事实写 not_required 或 pending，并由 completion hash-bound；不得事后回写 accepted/rejected。外部 review decision 派生 effective_manual_review_status，语义见第 16.11 节；invalid 与 pending/rejected 一样不能完成。

### 16.2 Color pipeline

正式 metric 固定：

~~~text
uint8 BGR
→ RGB
→ IEC sRGB EOTF
→ linear RGB float64
→ D65/2-degree XYZ
→ CIELAB
→ CIEDE2000
~~~

禁止用 OpenCV uint8 Lab 作为正式 metric。

low-frequency：

- 先逐像素转 Lab；
- sigma=3、truncate=3 separable Gaussian；
- reflect boundary；
- mask-normalized convolution；
- numerator=Lab×safe_valid；
- denominator=safe_valid；
- denominator 低于 floor 时 unevaluable；
- invalid/protected 不得渗入；
- 矩阵、white point、epsilon、reduction dtype 全部版本化并有 golden vectors。

### 16.3 三尺度和纵向 block

safe、非 global protected、common-valid 上测：

~~~text
immediate: seam 两侧约 ±1 px
shoulder: 2–8 px
wide shoulder: 4–16 px
band-32: 16–32 px
band-64: 32–64 px
low-frequency: sigma=3
~~~

band-32/band-64 在 frozen sidecar coverage 足够时是正式 applicable metric，不只是 atlas；support/coverage 不足时按 insufficient_evidence 处理，不能外推 correspondence。报告 DeltaL、Deltaa、Deltab、DeltaE00 的 signed/absolute median、P90、P95、P99、max、有效 block 和 row coverage。

纵向 block 初始高度 16 px，左右分别稳健拟合 low-frequency profile，测 seam step、slope mismatch、wide-band DeltaE 和 actionable/unresolved blocks。

### 16.4 Target、ceiling 与材料性改善

以下是 provisional 值，必须在非 Q0 候选前冻结：

| 指标 | target | engineering ceiling |
| --- | ---: | ---: |
| wide-safe DeltaE00 median | <=1.0 | <=1.25 |
| wide-safe DeltaE00 P95 | <=2.0 | <=2.5 |
| safe per-pair block-P95 最大值 | <=3.0 | <=3.5 |
| clipping 每通道新增 | <=0.05 pp | <=0.10 pp |
| gain-normalized B1 ROI gradient ratio | >=0.92 或合法 low-gradient_unevaluable | >=0.90 或合法 low-gradient_unevaluable |

threshold 中每个 metric 条目必须显式保存：

~~~text
metric_id
quality_role = B_nonreg_only | C_target_ceiling
direction = lower_is_better | higher_is_better
applicable_rule
required
target
engineering_ceiling
nonreg_relative_tolerance
nonreg_absolute_tolerance_or_MDE
coverage/evaluability rules
~~~

方向公式统一为：

~~~text
lower_is_better:
  actionable = before > target + tolerance
  target_pass = after <= target
  ceiling_pass = after <= engineering_ceiling

higher_is_better:
  actionable = before < target - tolerance
  target_pass = after >= target
  ceiling_pass = after >= engineering_ceiling

C_target_pass
= 所有 applicable、evaluable、required、
  quality_role=C_target_ceiling 的 metric 均 target_pass

C_ceiling_pass
= 上述同一集合均 ceiling_pass

near_target = NOT C_target_pass AND C_ceiling_pass
~~~

只有 quality_role=C_target_ceiling 的指标进入 C_target_pass/C_ceiling_pass。B_nonreg_only 只进入 B 类材料性非回归，不得因没有 target/ceiling 被伪装成 C 类通过。coverage 不足仍是 unevaluable。

当 C_target_pass 时，不再要求相对改善。

只有 near_target 时，quality_minimum_benefit_pass 才适用：

~~~text
wide-safe P95:
  minimum 5%
  target 10%
  stretch 15%

wide-safe median:
  minimum 3%
  target 5%
~~~

quality_minimum_benefit_pass 为以下至少一项：

~~~text
before_wide_p95 - after_wide_p95
  >= max(0.05 * before_wide_p95,
         MDE_wide_p95_abs)
OR
before_wide_median - after_wide_median
  >= max(0.03 * before_wide_median,
         MDE_wide_median_abs)
OR
before_actual_plateau - after_actual_plateau
  >= max(0.10 * before_actual_plateau,
         0.5 DeltaE00,
         MDE_plateau_abs)
~~~

candidate_benefit_pass 只决定复杂候选可否正式 render；quality_minimum_benefit_pass 只决定未达 target、但在 ceiling 内的正式输出能否成为 near-target。达到全部 absolute target 时二者均不再强制。15% 只是 stretch，不是统一完成硬门。

C_target_pass 与 C_ceiling_pass 的 metric 集合必须在 threshold asset 中显式、可重算。以下各项都必须冻结且只能有一个最终 quality_role；若希望“未改善该问题就不能完成”，必须设为 C_target_ceiling。初始 C 集合至少包括：

- wide-safe median/P95；
- per-pair block-P95；
- visible safe seam count；
- clipping；
- B1 gradient 或合法 low_gradient_unevaluable；
- owner plateau；
- neutral chroma drift；
- global x-slope 新增漂移；
- dark lift。

owner plateau、neutral chroma、global slope、dark 和 ghost 必须在 metrics-only/MDE 时逐项冻结 quality_role；若属于 C_target_ceiling，则必须有方向正确的 target/ceiling；若属于 B_nonreg_only，则必须有方向正确的非回归容差。没有冻结角色和值时不得静默视为通过。lower-is-better 非回归公式为：

~~~text
after <= before + max(relative_tolerance * before,
                      MDE_metric_abs)
~~~

higher-is-better 使用镜像公式 `after >= before - max(relative_tolerance * abs(before), MDE_metric_abs)`。

### 16.5 Visible safe seam

定义为 per-pair frozen block-P95 DeltaE00>2。

~~~text
target <= 2
engineering ceiling <= max(2, ceil(0.02 * pair_count))
clean baseline count 不得增加
~~~

84 pair ceiling=2；108/109 pair ceiling=3。count=3 只能进入 improved_near_target/pending manual review，不能自动完成，也不能自动归为 M7 protected handoff。

### 16.6 Clipping、dark 与 chroma

在相同冻结 mask 上报告 B/G/R low/high clipping before/after、absolute delta、P2 冻结 dark pixels 的 median/P95 L lift、neutral low-chroma drift、max gain/bias 和 parameter first/second differences。

初始：

~~~text
dark median L lift <= 1.0
dark P95 L lift <= 2.0
~~~

P95 必须在 metrics-only 后版本化。禁止“有改善即可自动突破 dark 门”的未数值例外。

### 16.7 Sharpness 与 ghost

只在真实 applied B1 ROI 测 gain-normalized Scharr energy、normalized edge shape、gradient attenuation P50/P95、seam normal profile 新双峰、double-edge/ghost score、actual bandwidth 与 weight range。

低纹理 ROI 的 before gradient 低于 frozen evaluability floor 时标为 low_gradient_unevaluable；不使用不稳定 ratio 判失败，改用 immediate DeltaE、band-step、clipping 和 ghost。

protected 区是 owner-only，但全局 gain 仍可能改变梯度幅值；因此 protected 清晰度比较必须除去预测 gain 或比较 normalized edge shape，不能用未归一化 0.97 ratio 误判合法亮度变化。

ghost/double-edge 要求新增 material/severe count=0；raw detector count 允许 MDE 内抖动并完整报告。

### 16.8 Owner plateau 与全景 profile

对 P2、旧 M6 P3、photometric owner-only 和 M6.1 P3 输出 robust L/a/b x-column profile、owner seam overlay、global slope、seam-correlated energy、source gain/bias curve、owner plateau metrics 和 first/second differences。

只限制相对 P2 新增漂移，不能假设真实场景照明必须全局恒定。

### 16.9 每项都带 evaluability

每个 per-pair/component/global metric：

~~~text
applicable
evaluable
required
quality_role = B_nonreg_only | C_target_ceiling
direction = lower_is_better | higher_is_better
sample_count
independent_tile_count
vertical_block_count
coverage_fraction
reason
~~~

只有 applicable=true 的 required metric 进入 coverage；其中只有 quality_role=C_target_ceiling 的项进入 target/ceiling 聚合，B_nonreg_only 只进入材料性非回归。B0 的 B1 sharpness、disabled Q4 等必须是 not_applicable，不是 unevaluable；applicable=true 但证据不足才是 unevaluable。

solver pair 必须属于 solver_safe_evaluable、protected_only 或 solver_insufficient_evidence；post-render quality pair 必须属于 quality_safe_evaluable、protected_only 或 quality_insufficient_evidence。两套分类不得混用，也不能静默删除差 pair。before/after 必须使用完全相同的 frozen coordinates、safe/protected 和 sample mask。

### 16.10 必出诊断

四组各输出到第 15.1 节定义的 P3 外 diagnostics 目录，并由外层 manifest 绑定：

1. P2 | old M6 P3 | photometric_owner_only | M6.1 P3 | 8x_abs_delta；
2. 按 wide DeltaE 排序 top-10 seam atlas；
3. top-10 material regression atlas；
4. source gain/bias 曲线和 component/fallback 边界；
5. L/a/b x-profile 与 owner seam overlay；
6. applied blend overlay；
7. owner | blended | delta | weight atlas；
8. clipping/dark-lift atlas；
9. protected_only_diagnostic_atlas；
10. top_10_owner_plateaus_atlas。

所有图必须标 pair、ROI、尺度、mask 语义、before/after 数值，不得只给肉眼图。

### 16.11 外部人工复核资产

improved_near_target 的 sealed P3 保持 sealed_manual_review_status=pending。每个 branch/generation 的人工决定写到 P3 目录外、只增不改的：

~~~text
artifacts\S013_M6_1_acceptance\manual_reviews\
  manifest.json
  <branch>_<generation>_decision.json
~~~

至少绑定：

~~~text
generation_id
P3_completion_sha256
quality_report_sha256
threshold_sha256
decision = accepted | rejected
reviewer
timestamp
reason
review_atlas_sha256
~~~

acceptance summary 从该资产派生 effective_manual_review_status；不得改写 sealed quality/completion。decision 缺失时为 pending；SHA 不匹配、malformed、重复有效 decision 或 superseding 链非法时为 invalid；二者都不能视为 accepted。

manifest 只追加新 decision，绑定每个文件 SHA 与 branch/generation 唯一键；已有 decision 文件不得覆盖。若需要推翻旧决定，追加 superseding decision 并显式引用旧 SHA。

---

## 17. Independent P3 hard audit v2

### 17.1 权限

hard audit 只控制真实性、parent/config/schema、provenance、transaction/hash、参数物理范围、formal remap、像素独立可重放和 atomic state。C 类 quality 和 60 s performance 不控制 seal。

### 17.2 父级实体复核

audit/verifier 从磁盘重新读取并计算 P2 completion、P2 RGB、full primary provenance、pair transactions、narrow/photometric replay、owner_domain、graph_topology 的 manifest/逐资产 SHA、component membership 和 raw RGB file SHA。不能只信 p2_parent_reference 或 completion 自报字符串。

### 17.3 光度审计

- solution/schema/bounds/config SHA；
- source index 连续且 frame id 对应 P2；
- 每 source 恰有一个 transaction；
- Q0 当且仅当 gain=1、bias=0、field=0；
- Q4 field 始终 canonical zero/disabled；
- component、gauge、rank、condition、fallback；
- solution 的 edge/component/source membership 与 sealed train-only graph_topology 逐值一致；
- identity_fallback_inside_solved_component_count=0；
- train/validation/guard dtype、shape、互斥与 safe/valid 子集；
- global masks 等于 pair masks 确定性合并；
- parameter SHA、transaction 与 formal render 使用值一致。

### 17.4 Blend 审计

- plan 数量/顺序等于 replay pair；
- transaction 唯一且 parent SHA 匹配；
- active 是 safe/common/expected-valid/eligible 子集；
- active 与 protected 交集=0；
- B2–B4 ineligible；
- global masks 不可伪造为空；
- provenance weight 与 plan weight 逐值相等；
- secondary frame/source/u/v 精确等于 replay；
- owner-only sentinel 正确；
- B0 active=0；
- selected flag/model/width 与实际一致；
- conflict 后 transaction 已重写。

### 17.5 全像素独立重放

从 raw RGB、P2 frozen UV、canonical gain/bias、zero field 和 B1 plan 独立计算：

~~~text
全部 owner-only pixels == photometric_owner_only
全部 non-blend final pixels == photometric_owner_only
全部 active B1 pixels == independent blend replay
protected pixels == owner-only
~~~

检查最终 uint8，不只验证 provenance 与输出自洽。audit_remap_invocations 与 formal_remap_invocations 分开。

hard audit 还必须显式验证：

- decoded visual_panorama/photometric_owner_only dtype、shape、channel order；
- valid/invalid/hole mask 等于 P2，不新增孔洞；
- Q0+B0 decoded uint8 与 P2 canonical PNG 数组全等；
- actual_changed_rgb_count 从最终数组重算并等于 transaction；
- 每 invocation、每 source formal remap=1，formal attempt 恰为 1；
- formal ROI witness 覆盖全部 owner 与 active pixels；
- canvas<=200 MP；
- live-array 与 resident-source 不超过 effective config cap；
- allocation request 使用 checked arithmetic；
- post-compose B1→B0 canonicalization 后全部 plan/mask/ID/weight/provenance 一致。

负向测试必须检出单独篡改一个 owner-only RGB、一个 active B1 RGB、secondary UV/weight 或 gain/bias/transaction/parent/config。

### 17.6 Malformed 与 sealed verifier

shape/count/schema 错误必须返回结构化 failure，不得因 broadcasting、index 或 zip 异常泄漏成未分类错误。

sealed verifier 校验当前实体 parent/effective config SHA、canonical asset allowlist、solution/transactions/quality/performance/hard audit schema/SHA，拒绝 completion 未绑定的额外正式资产，并确认 completion 最后原子写及 pending state。

---

## 18. Fallback 与 pointer 状态机

~~~text
verify sealed P2-v4
→ extract evidence
→ choose Q0–Q3 and B0/B1
→ formal render
→ post-compose B1 no-op canonicalization
→ write unsealed P3 assets including performance.json pre-audit content
→ diagnostic quality
→ independent hard audit

hard audit pass:
  hard_audit.json already contains audit_seconds
  atomically write P3_completion last inside pending
  verify_sealed_s13_p3(pending)
only if pending verifier passes:
    atomically promote pending directory to final immutable P3
    verify_sealed_s13_p3(final)
    only if final verifier passes:
      current_latest = P3
    if final verifier fails:
      atomically rename the entire final generation to
      artifacts/S013_M6_1_acceptance/failure_quarantine/<branch>/<generation>
      verify canonical final path has no P3_completion
      current_latest remains P2

any hard audit / pending verifier / promote / final verifier failure:
  no identity rebuild
  fail closed
  before promote: clear/quarantine pending
  after promote: quarantine the entire final generation as above
  do not leave final P3_completion
  current_latest remains P2
  write structured failure
~~~

pointer update 自身失败时，保留 `verified_but_unpointed` sealed generation，不篡改其内容，pointer 继续指 P2，并写 P3 外结构化 failure/acceptance summary；下次只允许在重新验证同一 completion SHA 后重试 pointer 原子更新。该状态不得冒充 current P3，且必须有负向测试。

candidate gate 失败必须在正式 render 前选择更简单模型，不算 hard audit failure。diagnostic quality=unresolved/regressed/unevaluable 或 performance_target_met=false 都不触发 identity rebuild。

### 18.2 Hard-audit 后禁止 identity rebuild

以下以及任何未列出的 hard-audit reason 都不得 fallback：

~~~text
parent/config/schema/shape/asset tamper
P2 provenance/replay tamper
source/frame/order/count mismatch
formal/audit remap counter mismatch
ROI witness/resource/canvas violation
forbidden invocation
transaction/manifest/weight/UV implementation bug
generic renderer pixel mismatch
atomic/pending/pointer failure
unknown failure reason
~~~

failure reason enum 必须版本化并参数化测试，不能用自由文本把 integrity bug 归为 candidate-dependent。gain/bias、B1 payload、transaction、weight、UV 或 compose 的独立 replay mismatch 一律是实现完整性失败；Q0/B0 安全退化只属于正式 render 前的 A1/B-pre 选择。

---

## 19. 可重复性

同一 code、config、threshold、P2 和 input 连续运行两次。

byte-identical：

- PNG/masks；
- integer provenance NPZ；
- 固定 dtype/reduction/thread 后的 parameter/weight NPZ；
- sample split。

canonical normalized-equal：

- model/component 选择；
- blend plans；
- transaction canonical payload；
- quality metrics/decision；
- hard audit decision；
- completion semantic payload。

dynamic-only：

- timing/RSS；
- process id；
- generation timestamp/id；
- pointer path。

比较只剔除显式 allowlist 的动态字段，不能剔除模型、参数、mask、parent identity、像素或 decision。performance SHA 不得进入 canonical decision payload。

---

## 20. 四组真实验收

### 20.1 输出与运行边界

输出到 artifacts\S013_M6_1_acceptance，不得覆盖现有 S013_M6、S013_M6_final、S013_M6_acceptance。

每分支：

- 使用自己的 sealed P2-v4；
- 正式运行至少一次，每次创建新的 immutable generation/attempt 目录；
- sealed completion 已存在时拒绝覆盖；
- resume 只允许 parent/evidence/effective/threshold SHA 全等的 pending generation；
- 同 config/P2 再建不同 generation 做 reproducibility，两次使用同一 sealed P2 实体；
- pending 使用 atomic rename/cleanup，illegal overwrite/resume 必须有负向测试；
- 正式 invocation 的 M4/M5/ORB/Open3D/Depth/TSDF/GraphCut/mesh=0。

### 20.2 四组结构门

全部满足：

- P2 canonical RGB/provenance/pair transaction/旧 replay 与 v3 对应基线相同；
- P3 hard audit v2 pass；
- current_latest=P3；
- current_reviewed/current_preview 不自动变化；
- immutable primary fields 与 P2 全等；
- protected secondary=0；
- contributor<=2；
- invalid/hole 不增加；
- train-validation overlap=0；
- identity fallback inside solved component=0；
- Q0+B0 可逐像素恢复 P2；
- config/code/input/parent/asset SHA 完整；
- formal/audit counters 合法；
- performance pre-audit、audit timing 与外层 end-to-end 实测；
- forbidden invocation=0。

### 20.3 视觉状态

自动视觉完成：

~~~text
clean_noop
OR
clean_nonregressed
OR
materially_improved
~~~

improved_near_target：

- sealed P3 sealed_manual_review_status=pending；
- 外部 decision accepted 后，stage_exit=completed_manual_near_target；
- 该状态允许本轮 M6.1 以“人工接受的近目标终态”退出，但不表示达到完整 target；
- 不得自动更新 current_reviewed；
- m7_handoff_eligible=false，不得生成 M7 handoff。

unresolved、regressed、unevaluable 均不能报告最终视觉完成。

### 20.4 分支附加门

slow_ignore_pose：

~~~text
旧 Q2 模式必须被拒
pair 21 不得出现 2→9
legacy grayscale jump <=3 DN
无超过冻结 owner-plateau ceiling 的中央亮度块
owner plateau 通过冻结的 actual plateau 数值门
不得出现类似 +9.04% aggregate P95 回归
~~~

如果所有 Q1–Q3 无法安全改善，可以输出结构合法 Q0+B0；但若 baseline actionable，quality=unresolved，不能冒充完成。

slow_direct：

- baseline 达 target 可 clean_noop；
- baseline actionable 时 P3=P2 只能 unresolved。

fast_direct / fast_ignore_pose：

- legacy max 11/8 只有证明为 protected_only/geometry/object risk 才能进入未来 M7 清单；
- 属于 safe photometric 时 M6.1 必须改善；
- 不得扩大 B1 侵入横梁、立柱、线缆或其它 protected。

### 20.5 60 s

每组报告 end_to_end_through_completion_seconds 与 performance_target_met。超过 60 s 不改变视觉/结构结论，但必须进入性能待办，留给 M9 决定生产门。

---

## 21. M7 分流与停止盲调

状态优先级先判定人工接受终态：

~~~text
if stage_exit == completed_manual_near_target:
  作为本轮人工接受终态退出
  skip §21.1 and §21.2
  m7_handoff_eligible = false
else:
  execute §21.1 and §21.2
~~~

`effective_manual_review_status=invalid|pending|rejected` 不能走该例外。

quality_state 与 stage_exit 每个 branch/generation 只能各有一个总状态；component/pair 状态仅是可重算明细，不得产生与 branch 总状态竞争的第二套完成结论。四分支 milestone 再从四个 branch 状态确定性聚合。

### 21.1 仍属于 M6.1

任一情况仍是系统性 M6.1 问题：

- residual 跨至少 3 个连续 pair；
- safe visible seam 未达到 target；
- safe wide-band target 未达到且 near-target 仍为 pending/rejected；
- global x slope/white balance 新回归；
- unsupported parameter boundary；
- Q0 no-op 但 baseline actionable；
- 只有侵入 protected 才能改善。

外部人工 accepted 的 near-target 使用：

~~~text
stage_exit = completed_manual_near_target
m7_handoff_eligible = false
~~~

它是“接受当前近目标结果并结束本轮”的终态，不是完整 target，也不能借此启动 M7。若项目仍要求进入 M7，必须先让 safe photometric 达到完整 target。

### 21.2 才能生成 M7 handoff

四组结构门通过，且 safe photometric 达完整 target、stage_exit=completed_target；residual 同时：

- 位于 protected/geometry/object 风险；
- 是 source schedule 上非连续的孤立 pair；
- 任意两个 residual pair 不相邻；
- 每分支数量 <= max(3, ceil(0.05×pair_count))；
- safe visible seam <=2。

输出：

~~~text
artifacts\S013_M6_1_acceptance\M7_handoff.json
artifacts\S013_M6_1_acceptance\M7_handoff_atlas\
~~~

每项含 branch/run/generation/P2 parent、pair、ROI、source/frame、risk、P2/P3 crop、transaction SHA、建议分类。生成后停止，不实现 M7。

### 21.3 停止盲调

若连续两个合法 implementation/threshold version 满足 wide-safe P95 改善<1% 且仍 unresolved，禁止继续放宽门、扩大 feather、把 safe 改 protected 或增加 blur。completed_manual_near_target 是独立的人工接受终态，不生成 handoff，也不要求 blocked report。

输出 root_cause_blocked_report.json，分类：

~~~text
photometric_evidence_insufficient
capture_exposure_non_linear
capture_white_balance_non_linear
protected_coverage_too_high
upstream_geometry_or_owner
photometric_model_family_not_supported
low_frequency_field_authorization_required
~~~

---

## 22. 实施顺序与每步完成定义

### Step 1：Identity、schema 与 config bootstrap

- m61_v2 identity/registry；
- schema；
- evidence config 与 effective config；
- strict parser；
- normal/resume/fallback/audit/verifier 接线；
- v3 回归；
- 不写 placeholder threshold 或最终 manifest entry。

完成：所有关键 config 有 runtime effect，v3 行为可复现。

### Step 2：P2-v4 replay 与最终 metric 输入

- branch-specific prepare script；
- 64 px diagnostic shoulder；
- audited support intersection；
- safe/protected 派生；
- tile split；
- 只用 train evidence 构造并封存 graph topology sidecar、component membership 与 SHA，不运行 Q1–Q3 solve；
- color pipeline；
- P2 canonical regression。

完成：sidecar 可审计，P2 canonical 内容不变。

### Step 3：Q0+B0 metrics-only 与 freeze

- quality measurement；
- block/plateau/clipping/ghost；
- MDE calibration；
- 从封存 topology 与 pair actionability 确定性聚合 component actionability；
- Q0+B0 四组 baseline；
- 只写 threshold v2 proposed 后暂停；
- 人工明确批准后，先写并校验 append-only threshold_approval witness，再通过第二次文件变更写 final threshold；
- 最终 candidate YAML/config SHA/manifest entry；
- 第二次 invocation 才启用候选。

完成：synthetic 宽亮条、同亮度异色、少数局部回归、plateau 和低纹理假失败测试通过；未经人工批准不得进入 Step 4。

### Step 4：Robust Q0–Q3

- 加载并验证 frozen train-only graph_topology SHA/membership，不重选 edge/拆 component；
- Huber/IRLS；
- bounded component solve；
- evidence-adaptive continuity；
- candidate non-regression/benefit；
- component fallback。

完成：真实 slow_ignore 旧 Q2 和 synthetic median-good/P95-bad 均被拒。

### Step 5：B1 2 px

- 删除 1 px；
- B0/B1 only；
- mask 分离；
- pixel-center golden；
- immediate benefit/MDE；
- conflict；
- actual no-op canonicalize B0。

完成：selected B1 有非零真实变化和可测局部收益，protected 始终 owner-only。

### Step 6：Streaming、provenance 与 performance

- compact ROI；
- resident LRU；
- remap once；
- transactions；
- live memory/RSS；
- sealed/outside timing。

完成：small reference 与 compact uint8 全等，无约 1 GB corrected cache。

### Step 7：Hard audit 与状态机

- independent raw RGB replay；
- mutation tests；
- sealed allowlist/verifier；
- classified fallback；
- pointer/pending。

完成：单像素、UV、weight、parent、config 等篡改均按预期失败。

### Step 8：四组 acceptance 与复现

- 每组正式运行；
- 每组复现重跑；
- summary/atlas；
- performance soft SLA；
- handoff 或 blocked。

完成后严格停止。Q4 非零 solver 不在上述任何 Step 中。

---

## 23. 逐文件实施清单

### 23.1 配置

~~~text
configs\video_candidates\s013\
  S013_output_first_progressive_dense_central_slit_m61_v2.yaml
  quality_thresholds_m61_v2.json
  candidate_manifest.json
~~~

### 23.2 核心源码

| 文件 | 改动 |
| --- | --- |
| video_s13_contract.py | identity registry、v4/v2 schema、严格 config |
| video_s13_bundle.py | 去除新 generation 的 v3 硬编码，资产 binding |
| video_s13_m5.py | 仅支持确定性 photometric replay preparation，不改变 M5 决策 |
| video_s13_replay.py | sidecar read/write/hash/verify |
| video_s13_photometric.py | split、sampling、IRLS、component、selection、plateau prediction |
| video_s13_luminance_field.py | 只实现 zero-field/disabled contract |
| video_s13_blend.py | 删除 1 px、B1 2 px、mask/benefit/conflict |
| video_s13_visual_quality.py | CIEDE2000、wide/block/plateau/clipping/ghost/state |
| video_s13_p3_hard_audit.py | parent 实体复核与全像素独立 replay |
| video_s13_m6.py | effective config、两阶段、streaming、provenance、实计时 |
| video_s13_experiment.py | bootstrap/normal/resume/fallback/pointer |

可新增职责隔离模块：

~~~text
video_s13_m61_config.py
video_s13_m61_evidence.py
video_s13_m61_quality.py
video_s13_m61_streaming.py
~~~

### 23.3 脚本

~~~text
scripts\prepare_s13_m61_p2.py
scripts\calibrate_s13_m61_quality.py
scripts\summarize_s13_m61_validation.py
scripts\compare_s13_m61_reproducibility.py
~~~

prepare 显式接收 branch-specific parent/P2 目录与 evidence config，拒绝覆盖 v3。calibrate 只 resume sealed P2-v4，只允许 Q0+B0。

---

## 24. 必须新增的测试

### 24.1 Contract/config

- v3 identity/hash/dispatch 不变；
- m61_v2 identity/schema；
- unknown key、NaN/Inf、非法 width/level/fraction；
- threshold path traversal/missing/tamper/SHA；
- threshold approval missing/rejected/tamper/proposed-SHA mismatch；
- normal/resume/fallback/audit 同 effective SHA；
- dead parameter test；
- manifest self-hash；
- lower/higher direction comparator 边界；
- B_nonreg_only 与 C_target_ceiling 聚合隔离；
- sealed/effective manual status 的 pending/accepted/rejected/invalid 派生。

### 24.2 Replay/evidence

- P2 RGB/provenance/transaction/旧 replay 不变；
- sidecar tamper/乱序/重复 pair；
- train-only graph topology sidecar membership/SHA tamper；
- component actionability 只从 frozen topology+pair actionability 聚合；
- support 外 UV 不进入 solver；
- canvas tile/halo 无泄漏；
- validation 改变不改变 solve；
- sampling 覆盖 y/shoulder/luminance/source-u；
- coverage/evaluability 边界。
- solver 80/20 coverage 不得冒充 post-render quality coverage；
- safe output pair insufficient-evidence 超过 frozen <5% cap 时 unevaluable。

### 24.3 Q solver/selection

- Rec.709 Q1；
- synthetic gain/WB；
- Huber outlier；
- bounds/rank/condition；
- component fallback；
- no single-source identity boundary；
- median 好、P95 差拒绝；
- global 好、worst pair 差拒绝；
- 多 component 单独合格但组合造成 global regression 时确定性 rollback；
- B_nonreg_only 不得触发 baseline actionable 或 materially_improved；
- 两个已达 target 的模型在 selection-score MDE 内选择更简单模型；
- 仅复杂模型跨过 C target 或 score 改善超过 MDE 时才允许替代；
- slow_ignore Q1 可进入后续门；
- slow_ignore Q2 被拒；
- evidence-adaptive continuity；
- canonical near-identity→Q0；
- disconnected components。

### 24.4 Q4 disabled

- config disabled；
- canonical field 全零；
- 非零 field/solver invocation 被拒；
- schema/transaction 可审计。

### 24.5 Blend

- 1 px 路径不存在；
- 2 px golden；
- active 是 safe/common/expected/eligible 子集；
- active/protected=0；
- immediate benefit/MDE；
- wide/clipping/ghost/sharpness regression 回 B0；
- actual no-op canonicalize B0；
- B2–B4 ineligible；
- conflict 确定性；
- provenance/weight 一致。

### 24.6 Quality

- 8–16/32 px 亮度块检出；
- 同亮度异色检出；
- 5% 行局部恶化检出；
- owner plateau 检出；
- low-gradient ratio 不假失败；
- gain-normalized edge；
- severe ghost；
- clipping/dark/slope；
- target crossing 不再强迫 15%；
- improved_near_target/manual pending/accepted/rejected；
- clean_noop/clean_nonregressed/unresolved/regressed/unevaluable；
- visible seam 0→1 失败；
- before=0 不除零；
- macro/micro 非回归与 composite benefit。

### 24.7 Hard audit/state

- 每个 primary field 篡改；
- source/frame 顺序；
- gain/bias/config/solution；
- split overlap/mask fake empty；
- unsafe/protected active；
- secondary frame/source/UV/weight；
- transaction parent/SHA；
- 单 owner pixel/active pixel；
- malformed shape structured failure；
- 任意 hard-audit/replay mismatch 不 fallback、不二次 render；
- pending verifier/promote/final verifier 任一失败都无 final completion、pointer 留 P2；
- formal/audit counters；
- current_reviewed/preview 不变。
- fallback reason allowlist；
- P3-v4 resource/canvas/ROI witness；
- illegal overwrite/resume；
- diagnostics 只能位于 P3 外且 manifest 绑定 sealed P3；
- acceptance summary canonical path、原子写、append-only manifest 与 superseding chain；

### 24.8 Streaming/reproducibility

- compact vs reference uint8 全等；
- 不存在 source_count×full_canvas cache；
- resident ROI<=5；
- checked allocation；
- timing 不硬编码；
- byte/canonical/dynamic 三层复现。

---

## 25. 验证命令

~~~powershell
$G305Python = 'D:\Panoramic_Camera\.conda\python.exe'

& $G305Python -m pytest -q tests/test_video_s13_contract.py tests/test_video_s13_dispatch.py tests/test_video_s13_bundle.py tests/test_video_s13_photometric.py tests/test_video_s13_luminance_field.py tests/test_video_s13_blend.py tests/test_video_s13_visual_quality.py tests/test_video_s13_p3_hard_audit.py tests/test_video_s13_m6.py tests/test_video_s13_experiment.py tests/test_prepare_s13_m61_p2.py tests/test_calibrate_s13_m61_quality.py

ruff check src tests scripts
& $G305Python -m compileall -q src tests scripts
git diff --check
& $G305Python -m pytest -q
~~~

真实验收必须另列单元/合成、固定四组、P2 regression、hard audit、visual state、复现、实际资源、performance target 和 forbidden invocation；不能只说“测试通过”或“肉眼更好”。

---

## 26. 最终完成定义

### 26.1 工程结构

- m61_v2 identity/schema/config 完成；
- v3 可复现，历史 artifacts 未改；
- P2-v4 sidecar 与 canonical regression 完成；
- threshold 在非 Q0 候选前冻结；
- normal/resume/fallback/audit/verifier 全接线；
- Q4 非零保持 disabled；
- 四组 hard audit pass；
- protected secondary=0；
- streaming 生效；
- timing/RSS/live bytes 实测；
- 两次复现通过；
- forbidden invocation=0。

### 26.2 视觉

四组必须为：

~~~text
clean_noop
OR
clean_nonregressed
OR
materially_improved
OR
improved_near_target with effective_manual_review_status=accepted
~~~

任何 unresolved、regressed、unevaluable 或 pending/rejected near-target 都不能报告 M6.1 最终视觉完成。accepted near-target 必须报告 stage_exit=completed_manual_near_target、m7_handoff_eligible=false；只有达到完整 target 才报告 completed_target 并按第 21.2 节判断 M7 handoff。

slow_ignore_pose 必须拒绝旧 Q2、pair 21 legacy jump<=3、无旧中央亮块。

### 26.3 阶段边界

明确没有：

~~~text
non-zero Q4
GraphCut
Depth
mesh
source rescue
object transaction
M7 pixel changes
M8
M9
production delivery
production lock
current_reviewed 自动更新
~~~

---

## 27. 最终报告字段

最终报告逐项列出：

- 只读基线 commit 与最终 commit；
- candidate/schema；
- 两份源文档 SHA 与本 v3 文档路径；
- P2 canonical regression；
- evidence/effective config SHA；
- threshold/MDE SHA；
- 四组 Q model/component；
- 四组 B0/B1；
- 四组 quality/manual state；
- wide-safe DeltaE median/P95；
- block-P95、visible seam、owner plateau；
- slow_ignore pair 21；
- protected blend count；
- formal/audit remap；
- pre-audit、audit 与 whole-invocation RSS/live bytes/resident ROI/timing/performance target；
- hard audit；
- reproducibility；
- forbidden invocation；
- tests/Ruff/compileall/diff-check；
- stage_exit=completed_target / completed_manual_near_target / blocked；
- M7 handoff、manual-near-target decision witness 或 blocked reason 三选一。

---

## 28. 可直接交给 Codex 的执行指令

~~~text
请在 D:\central_strip_Panoramic_Camera 当前工作树实施
Panoramic_Camera_S013_M6_1_Final_Merged_Photometric_Seam_Suppression_Codex_Implementation_Plan_v3.md。

c3aaa7ddbaf8aac5e00cded71ba9e4f1689b822a 仅为只读语义基线。
不得 reset、破坏性 checkout、覆盖用户改动或历史 artifacts。
第一条命令运行 git status --short，并先阅读 AGENTS.md、README、现有
S013 v3 config、M6 源码/测试与 artifacts\S013_M6_acceptance。

严格按 Step 1→8 实施。先完成 branch-specific P2-v4 replay、最终
safe/protected/split/metric，再只运行 Q0+B0 做 MDE/threshold freeze。
第一次 calibration 调用只写 proposed threshold 后必须暂停并向用户报告；
未经明确批准且未生成有效 threshold_approval witness，不得写 final threshold、manifest 或继续 Q1–Q3/B1。
批准后的第二次调用只读 final threshold。在 freeze 前不得启用 Q1–Q3 或 B1，
不得写 placeholder candidate manifest。

正式 identity 使用：
S013_output_first_progressive_dense_central_slit_m61_v2

本轮 Q4 只实现 zero-field/disabled contract，不得实现非零 solver。
正式 blend 只允许 B0 与 B1 2 px；B2–B4 明确 ineligible。

修复 config threading、aggregate median 误选、component 内单 source identity
边界、1 px no-op、弱 quality、owner plateau、hard audit 和资源假计量。

候选 aggregate P95 初始非回归公式必须为：
after <= before + max(0.02*before, 0.0005 linear)
不得使用下载版 v2 的 0.0015 linear。

候选 selection 必须在唯一正式 render 前完成；post-render quality 不反向改参数。
60 s 仅为 M6.1 soft SLA，不改变候选、seal 或视觉状态。

四组各正式运行一次并复现重跑一次。四组未达到本文件完成条件前不进入 M7。
若两轮合法版本仍无收益，输出 blocked report，不得调松门、扩大 feather、
改变 safe/protected 或增加 blur。

完成后按第 27 节逐项报告，生成 handoff、人工接受的 near-target 终态或
blocked report 后严格停止。
~~~

---

本方案的最终判定：A1/A2 安全和真实性必须严格；B 类只阻止材料性回归；C 类用 target/ceiling 区分“已干净、接近目标、仍未解决”。系统性 safe 光度接缝必须留在 M6.1，M7 只接收少量、孤立、明确属于 protected 结构或几何的残余问题。
