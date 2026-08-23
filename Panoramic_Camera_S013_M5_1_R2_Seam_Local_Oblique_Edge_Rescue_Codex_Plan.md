# Panoramic Camera S1.3 M5.1-r2 接缝局部证据与斜边连续性修复 Codex 实施任务书

> 适用仓库：`D:\central_strip_Panoramic_Camera`<br>
> 编写基线：分支 `codex/video-realtime-seam-v6`，HEAD `0eaaa5eb86d8f86e71d27cc05f2e37db73fa9fe1`<br>
> 目标候选：`S013_output_first_progressive_dense_central_slit_v5`<br>
> 首轮边界：只完成并封存新的 P2-v5；不得进入 M6/P3
> 状态：供 Codex 直接实施、测试和生成验收证据

---

## 0. 给 Codex 的执行指令

你需要在当前仓库内实现本任务书，而不是只做评审或再次输出方案。

开始时必须：

1. 工作目录固定为 `D:\central_strip_Panoramic_Camera`。
2. 第一条命令运行：

   ```powershell
   git status --short
   ```

3. 完整阅读 `AGENTS.md`，然后阅读本任务书列出的源码、配置和测试。
4. 记录实际分支和 HEAD。若 HEAD 晚于本任务书基线，先检查增量；不要覆盖用户或其他代理的修改。
5. 保留旧 v3/v4 配置、旧 generation、旧 artifact、旧 threshold/approval 和 production lock；所有新结果写入新目录。
6. 使用 `rg` 搜索，使用 `apply_patch` 修改文件；不得使用破坏性 checkout、reset 或批量删除。
7. 先写能在旧实现上失败的测试，再实现功能。
8. 首轮严格停止在 P2-v5。不得为了得到 P3 而临时复用旧 M6 阈值。
9. 不要提交、推送或创建 PR，除非用户另行明确要求。

实施完成后必须给出：修改文件、测试结果、四分支 P2 输出、关键 ROI、性能计数、未完成项和停止原因。不得把单元测试通过描述成真实数据已经验收。

---

## 1. 结论与目标结果

本轮不增加独立的 `P2.5`、`M5.2` 或封存后像素补丁，而是在 M5.1 每个 pair 的候选评估内部修复以下问题：

1. 净化现有单次 PyrLK 对应点，去掉无效落点和极端误跟踪；
2. 让 final-corridor geometry 的拟合证据真正跟随每条候选 seam；
3. 使用 pair-local、任意方向强边缘连续性审计发现纸盒斜边等局部错层；
4. 只对 `visual_suspect` pair 继续比较其他现有 C0–C4 和 S2/S1/S0 候选；
5. 缓存现有结构特征，抵消新增小 ROI 检查的时间；
6. 若首轮已解决真实目标 ROI，则停止，不实现新的微变形模型。

目标阶段关系：

```text
P0
 ↓
P1（sealed parent）
 ↓
M5.1-r2
 · 单次现有 GFTT + forward LK
 · LK 尾部净化
 · candidate-specific seam-local evidence
 · 现有 C0–C4
 · 强边缘法向连续性检查
 · suspect pair 才继续候选
 ↓
P2-v5（sealed，P2-only）
 ↓
停止并验收
```

首轮不允许出现：

```text
P2-v5 → 旧 M6 threshold/approval → P3
```

---

## 2. 已确认的根因与真实证据

### 2.1 final-corridor 当前只改变 application band

当前 M5 在 `src/panorama_demo/video_s13_m5.py` 中把 final seam 传给：

```python
reestimate_s13_final_corridor_alignment(...)
```

但仍传入固定的：

```python
alignment_shoulder=(x0, x1)
```

`src/panorama_demo/video_s13_alignment.py` 中的 wrapper 只替换 `application_band`，真正的对应点过滤仍按完整 96 px shoulder 执行。

当前实际语义是：

```text
在完整 96 px 肩区内混合前景、背景和误跟踪点拟合模型
                         ↓
仅把该模型应用于 final seam 周围最多 ±8 px
```

它不等于“用 final seam 邻域证据重新估计”。近景纸盒与背景存在不同视差时，背景点会支配模型。

关键位置：

- `src/panorama_demo/video_s13_m5.py:459-487`
- `src/panorama_demo/video_s13_m5.py:553-565`
- `src/panorama_demo/video_s13_alignment.py:408-423`
- `src/panorama_demo/video_s13_alignment.py:583-610`

### 2.2 现有 PyrLK error 被丢弃

`_pair_correspondences()` 已取得 `_error`，但只检查 status 和 finite；没有检查：

- moving 点是否仍在 crop 内；
- moving 点落点是否属于 `right_valid`；
- LK error 是否属于极端尾部。

关键位置：

- `src/panorama_demo/video_s13_m5.py:169-195`

真实 slow pair 47 已出现：C3 median 很低，但 P95 被少量尾部残差拉高，最终退到仍有大残差的 C1，同时 DP hard gate 通过：

```text
artifacts/S013_M6_final/slow_direct/
  generations/20260812T135613-0bd15b99ce8b/
  P2/pair_transactions/pair_0047.json
```

### 2.3 当前运行时只专门保护长水平结构

M5 已计算一般 seam structure 与 long-horizontal 指标，但运行时只调用：

```python
long_horizontal_structure_catastrophe_guard(...)
```

纸盒顶部斜边的 `thin_structure`、`double_edge` 或 edge-normal 错层只进入 diagnostic，不会拒绝当前候选。

关键位置：

- `src/panorama_demo/video_s13_m5.py:587-600`
- `src/panorama_demo/video_s13_quality.py:81-156`
- `src/panorama_demo/video_s13_quality.py:262-342`

### 2.4 当前首个 hard-safe 候选立即胜出

候选按局部目标排序后，第一个通过现有 hard gate 的候选立即成为结果，其余默认记录为：

```text
skipped_after_higher_rank_safe_candidate
```

如果现有 hard gate 没检测到斜边错层，一个可能更连续的后续 seam 或 geometry map 根本不会被采样。

关键位置：

- `src/panorama_demo/video_s13_m5.py:517-635`

### 2.5 截图对应的真实风险区域

参考截图：

```text
C:\Users\zyh68\Pictures\Screenshots\屏幕截图 2026-08-13 111240.png
```

对应 fast-direct P3 右侧大约：

```text
x = 1436 ... 1648
pair = 65 ... 82
```

重点 pair：

| Pair | 当前 geometry | 当前 held-out P95 |
|---:|---|---:|
| 65 | C0 | 约 24.59 px |
| 66 | C1 | 约 7.28 px |
| 71 | C1 | 约 54.62 px |
| 72–77 | C0 | 不可评估 |
| 78 | C1 | 约 9.23 px |
| 83 | C2 | 约 0.63 px |

基准 generation：

```text
outputs/S013_M1_M9_full_run_20260813_000442/
  branches/fast_direct/generations/20260813T000500-36dc28cb6600
```

首轮真实验收必须固定导出 pair 65、66、71、78 和 `x≈1436...1648` 的 before/after ROI。

---

## 3. 必须保持的范围与不变量

### 3.1 本轮允许增加的计算

仅允许：

- 最多 400 个已有 LK 点的布尔过滤与稳健统计；
- 每条实际检查 seam 的 `±16/±24 px` 对应点过滤；
- 风险 pair 的少量 compact C0–C4 map ROI sampling；
- `32-row` 重叠块或单一强边 component 的 Sobel 法向检查；
- 已有 Lab/gray/Sobel/Laplacian 特征缓存。

### 3.2 本轮禁止

- 第二次或反向 PyrLK；
- DIS、RAFT、dense flow；
- depth、mesh、Open3D、ORB-SLAM3 新调用；
- Hough、LSD、语义分割；
- 新的全图 GraphCut、MultiBand 或 feather；
- 第三次全分辨率 P2 render；
- 新的 raw RGB decode 或正式全分辨率 remap；
- 跨 pair 联合优化、object lock 或 owner expansion；
- 对已封存 P2/P3 修改像素、map、provenance 或 hash；
- 在已 warp 图上再 warp；
- 以 blur 或混合掩盖强结构断裂；
- 恢复 `_select_held_out_safe_seam()` 的 straight 前置授权；
- 把 `selected_as_best` 或完整 `structurally_non_degrading()` 恢复成 P2 总门；
- 修改 production lock 或把 S13 声称为 production eligible。

### 3.3 owner、seam 和 sampling 硬约束

必须继续满足：

- seam 为整数；
- 每行 seam step 仅为 `-1/0/+1`；
- 相对 base seam 位移 `≤8 px`；
- seam family 不交叉；
- 每个 owner 宽度 `≥1 px`；
- valid 像素恰有一个真实 owner；
- `valid == provenance.valid == (owner_frame_id >= 0)`；
- invalid 像素的 owner/source index/transaction 为 sentinel；
- `source_u/source_v` 在 valid 区有限且位于真实源范围；
- source 0 的 geometry/seam transaction 为 `-1`；source `i>0` 对应 transaction 为 `i-1`；
- photometric/blend transaction 在 P2 为 `-1`；
- secondary frame 为 `-1`、secondary UV 为 `NaN`、weight 为 `0`；
- map 总位移 `≤8 px`；
- positive Jacobian、source bounds、support retention 不放宽；
- expected support 内无内部洞；
- 黑色 RGB 仍是有效内容。

主要审计位置：

- `src/panorama_demo/video_s13_hard_audit.py:142-283`
- `src/panorama_demo/video_s13_hard_audit.py:311-452`
- `src/panorama_demo/video_s13_alignment.py:30-47`
- `src/panorama_demo/video_s13_alignment.py:464-470`
- `src/panorama_demo/video_s13_m5.py:824-906`

---

## 4. 版本隔离与 P2-only lineage

### 4.1 不得修改旧 v4

以下内容必须逐 hash 保持不变：

```text
configs/video_candidates/s013/
  S013_output_first_progressive_dense_central_slit_v4.yaml

configs/video_candidates/s013/quality_thresholds_m61_v2.json

旧 threshold approval / M7 handoff / manual authorization
旧 v4 generations
production.lock.json
```

共享 M5 代码不能无条件开启新行为。v3/v4 配置下必须继续走旧路径，合成回归应证明旧输出逐像素不变。

### 4.2 新建 v5 P2-only 身份

首轮新增：

```text
algorithm_id:
  S013_output_first_progressive_dense_central_slit_v5

implementation_id:
  s013_output_first_progressive_dense_central_slit_m51_r2

contract_schema:
  gemini305-video-s13-output-first/v5

P2 completion schema:
  gemini305-video-s13-p2-completion/v5

pair transaction schema:
  gemini305-video-s13-m5-pair-transaction/v3

pair transaction manifest schema:
  gemini305-video-s13-m5-pair-transactions/v2
```

新 config：

```text
configs/video_candidates/s013/
  S013_output_first_progressive_dense_central_slit_v5.yaml
```

必须满足：

```yaml
role: candidate
diagnostic_only: true
production_eligible: false
production_lock_eligible: false
required_output_components: [s013_p2_v5]
forward_pipeline:
  stage_order: [P0, P1, P2]
  default_stop_after: P2
p2_replay:
  enabled: true
  completion_schema: gemini305-video-s13-p2-completion/v5
```

v5 不得包含或引用旧 `m61_bootstrap`、旧 threshold SHA 或旧 approval SHA。

更新：

```text
configs/video_candidates/s013/candidate_manifest.json
```

新增 v5 条目并重新计算：

- v5 config canonical `config_sha256`；
- sibling candidate manifest 的 `manifest_sha256`。

不得修改共享的：

```text
configs/video_candidates/candidate_manifest.json
```

### 4.3 contract 与 runner 拆分 P2 schema 和 M6 资格

修改：

```text
src/panorama_demo/video_s13_contract.py
src/panorama_demo/video_s13_experiment.py
src/panorama_demo/video_s13_m61_config.py（仅在必要处保持旧常量隔离）
src/panorama_demo/video_s13_m61_evidence.py（只增加 v5 schema 常量/拒绝逻辑，不生成旧证据）
```

不要再只用：

```python
configured_algorithm_id == S13_FORMAL_M6_ALGORITHM_ID
```

同时代表 P2 schema、M5 功能和 M6 资格。为 config identity descriptor 增加明确字段或属性：

```text
p2_completion_schema
p2_only
requires_m61_bootstrap
m51_r2_enabled
m6_eligible
```

建议将 `run_s13_experiment()` 的 `run_m6` 参数改成：

```python
run_m6: bool | None = None
```

语义：

- `None`：按 config 的 `default_stop_after` 决定；
- v3/v4 + `None`：保持现有行为；
- v5 + `None`：封存 P2 后停止；
- v5 + 显式 `run_m6=True`：fail-closed；
- `--resume-generation` 尝试把 v5 P2 交给 v4 M6：fail-closed。

P2-v5 completion 必须显式记录：

```json
{
  "m6_eligible": false,
  "m6_blocked_reason": "successor_p2_requires_fresh_four_branch_threshold_lineage"
}
```

P2 hard-pass 后：

- `current_latest` 只能到 P2；
- 不得创建 P3/M6；
- 不得创建 delivery；
- 不得写 `native_p2_v4` 或旧 threshold/approval 绑定。

### 4.4 P2-v5 seal 必须绑定的资产

P2 completion 必须覆盖：

- canonical owner-only PNG；
- valid mask；
- pixel provenance；
- pair transaction manifest；
- 每个 pair transaction JSON；
- hard audit；
- `p2_seams.npz`；
- narrow replay manifest；
- 每个 replay pair NPZ；
- performance；
- diagnostic quality；
- P0/P1 parent completion/result/provenance hash。

`P2_REPLAY_SCHEMA` 字段未改变时可以继续使用 `/v1`，但每个 replay pair 必须精确绑定新的 pair transaction SHA。

任何候选 map 变化都必须同时反映到：

```text
canonical image
source_u/source_v
valid
owner/provenance
pair transaction
narrow replay
completion asset hash
```

禁止只改图像而不改 replay 或 provenance。

---

## 5. 新配置对象

增加一个冻结的 M5.1-r2 配置对象，名称可采用：

```python
@dataclass(frozen=True)
class S13M51R2Config:
    enabled: bool = False
    lk_error_sigma_multiplier: float = 3.0
    lk_error_mad_floor: float = ...
    lk_error_absolute_maximum: float = ...
    minimum_correspondence_count_for_error_filter: int = 16
    evidence_half_widths_px: tuple[int, ...] = (16, 24)
    moving_evidence_margin_px: int = 8
    block_height_px: int = 32
    block_stride_px: int = 16
    seam_support_radius_px: int = 6
    feature_halo_radius_px: int = 12
    normal_search_maximum_px: float = 3.0
    normal_search_step_px: float = 0.5
    minimum_edge_component_length_px: int = 12
    minimum_edge_support_count: int = 12
    minimum_edge_correlation: float = 0.65
    minimum_uniqueness_fraction: float = 0.10
    maximum_orientation_difference_degrees: float = 10.0
    suspect_lk_p95_px: float = 1.25
    suspect_edge_median_px: float = 0.75
    suspect_edge_p95_px: float = 1.0
    equivalent_edge_p95_tolerance_px: float = 0.15
    micro_rescue_enabled: bool = False
```

注意：

- v3/v4 必须传 `enabled=False`；
- v5 必须显式传 `enabled=True`；
- `lk_error_mad_floor` 和 `lk_error_absolute_maximum` 必须先从四分支现有 LK error 分布取得证据再冻结，不得凭空选择；
- 现有封存 artifact 没有保存 PyrLK `_error`；T0 应先做一次只增加统计、不改变候选决策、sampling map、provenance 或像素的 instrumentation baseline pass，采集分布后再冻结阈值，不能用 held-out residual P95 等其他指标冒充 LK error；
- 记录校准样本、分位数和最终冻结值；
- 不允许普通用户通过正式 CLI 修改这些参数。

---

## 6. T0：冻结基线与先写失败测试

### 6.1 记录基线

保存：

- 当前 HEAD；
- `git status --short`；
- 旧 v4 config 和 manifest SHA；
- 四个旧 P2 completion/image SHA；
- 当前定向测试结果；
- 当前 M5 wall time 与调用计数；
- fast-direct pair 65、66、71、78 的 transaction；
- slow-direct pair 47 的 transaction。

旧定向测试参考状态为 `59 passed / 22.52 s`，但实现时应记录实际结果。

### 6.2 先增加失败测试

新建：

```text
tests/test_video_s13_m51_r2.py
```

在修改算法前，至少让以下测试在旧实现上明确失败：

1. final seam 改变时，拟合证据仍来自完整 shoulder；
2. moved 点落在 `right_valid=false` 时仍进入拟合；
3. 极端 LK error 仍进入 held-out；
4. 32-row 局部斜边被全高横梁统计淹没；
5. 首个候选斜边错层时仍 early-stop；
6. v5 错误复用旧 M6 threshold 时未 fail-closed。

---

## 7. T1：净化现有 LK 对应，不增加新光流

修改：

```text
src/panorama_demo/video_s13_m5.py
_pair_correspondences()
```

建议返回结构化结果：

```python
@dataclass(frozen=True)
class S13PairCorrespondences:
    reference_xy: np.ndarray
    moving_xy: np.ndarray
    audit: Mapping[str, object]
```

伪代码：

```python
points = cv2.goodFeaturesToTrack(...)
moved, status, error = cv2.calcOpticalFlowPyrLK(...)

source = points.reshape(-1, 2)
target = moved.reshape(-1, 2)

status_keep = status.reshape(-1) > 0
finite_keep = np.isfinite(source).all(1) & np.isfinite(target).all(1)

# 必须先 finite，后 round/cast/index。
candidate = status_keep & finite_keep
target_in_bounds = (
    (target[:, 0] >= 0.0)
    & (target[:, 0] <= width - 1.0)
    & (target[:, 1] >= 0.0)
    & (target[:, 1] <= height - 1.0)
)
candidate &= target_in_bounds

tx = np.rint(target[candidate, 0]).astype(np.int32)
ty = np.rint(target[candidate, 1]).astype(np.int32)
target_valid = right_valid[ty, tx]

hard_keep = status_keep & finite_keep & target_in_bounds & expanded_target_valid

valid_error = error.reshape(-1)[hard_keep]
if len(valid_error) >= minimum_correspondence_count_for_error_filter:
    median = np.median(valid_error)
    mad = np.median(np.abs(valid_error - median))
    sigma = 1.4826 * mad
    robust_limit = median + multiplier * max(sigma, mad_floor)
    limit = min(absolute_maximum, robust_limit)
    error_keep = error <= limit
else:
    error_keep = np.ones_like(hard_keep)

final_keep = hard_keep & error_keep
```

硬要求：

- GFTT 仍为一次；
- forward PyrLK 仍为一次；
- 不增加 backward LK；
- invalid、nonfinite、OOB、`right_valid=false` 的点永远不能重新加入；
- error 过滤后点数不足时，不能恢复 error 尾部；
- 点不足只会使复杂模型返回 `insufficient_train_held_out_evidence`；
- 不改变黑色内容有效性；
- source/target 加 `x_offset` 前后坐标语义必须写测试。

audit 至少记录：

```json
{
  "detected_count": 0,
  "status_count": 0,
  "finite_count": 0,
  "target_in_bounds_count": 0,
  "target_valid_count": 0,
  "error_filtered_count": 0,
  "final_count": 0,
  "error_median": null,
  "error_mad": null,
  "error_limit": null,
  "gftt_call_count": 1,
  "forward_pyr_lk_call_count": 1,
  "backward_pyr_lk_call_count": 0
}
```

---

## 8. T2：让 final corridor 真正使用 seam-local evidence

修改：

```text
src/panorama_demo/video_s13_alignment.py
reestimate_s13_final_corridor_alignment()
```

新增纯函数：

```python
filter_s13_correspondences_for_final_seam(
    reference_points_xy: np.ndarray,
    non_reference_points_xy: np.ndarray,
    final_seam_x_by_row: np.ndarray,
    *,
    evidence_half_widths_px: tuple[int, ...] = (16, 24),
    moving_margin_px: int = 8,
    minimum_required_points: int = 36,
) -> tuple[np.ndarray, np.ndarray, Mapping[str, object]]
```

坐标规则：

```python
ref_row = np.rint(reference[:, 1]).astype(np.int32)
mov_row = np.rint(moving[:, 1]).astype(np.int32)

ref_distance = np.abs(reference[:, 0] - seam[ref_row])
mov_distance = np.abs(moving[:, 0] - seam[mov_row])

keep = ref_distance <= width
keep &= mov_distance <= width + moving_margin_px
```

硬要求：

- 先验证 y finite/in-bounds，再索引 seam；
- `seam_global` 只能与已经加过 `x_offset` 的 global correspondence 比较；
- candidate preview/quality 使用 crop 内的 `seam_local`；
- 禁止 global/local 坐标混用；
- 只尝试 `±16 → ±24`；
- 首轮不启用 `±32`；
- v5 禁止退回完整 `±48/96 px shoulder`；
- 点数需满足现有 `24 train + 12 held-out = 36`；
- 局部证据不足时，C2–C4 必须明确不可评估；C0 始终保留，C1 只按现有父级与证据规则接受；
- application band 继续最多为 final seam `±8 px`；
- evidence corridor 与 application band 是不同概念；
- 模型仍从 immutable P0 map 构建；
- deterministic train/held-out split 保持；
- 现有 C2–C4 held-out P95 `≤1.5 px` 上限不放宽。

返回 audit：

```json
{
  "requested_half_widths_px": [16, 24],
  "selected_half_width_px": 16,
  "raw_count": 0,
  "selected_count": 0,
  "minimum_required_count": 36,
  "sufficient_for_train_held_out": true,
  "wide_evidence": false,
  "full_shoulder_fallback_used": false,
  "mode": "seam_local"
}
```

若不足：

```json
{
  "selected_half_width_px": null,
  "sufficient_for_train_held_out": false,
  "full_shoulder_fallback_used": false,
  "mode": "seam_local_insufficient"
}
```

不得把不足时的 full-shoulder 结果标为 `reestimated_for_final_seam=true`。

---

## 9. T3：增加 pair-local 任意方向强边缘连续性指标

修改：

```text
src/panorama_demo/video_s13_quality.py
```

新增：

```python
pair_edge_registration_metrics(
    left: np.ndarray | SeamStructureFeatures,
    right: np.ndarray | SeamStructureFeatures,
    left_valid: np.ndarray,
    right_valid: np.ndarray,
    seam_x_by_row: np.ndarray,
    *,
    config: S13M51R2Config,
) -> dict[str, object]
```

### 9.1 检测对象

直接比较同一 pair 的：

```text
left_image
candidate final_right
```

二者已经位于相同 canvas/crop 坐标域，可直接判断 geometry 是否把两源真实结构对齐。

不要以 owner-only preview 作为两源配准证据。preview 可以继续用于既有 seam diagnostic 和 long-horizontal guard。

### 9.2 特征复用

复用 `prepare_seam_structure()` 已有：

```text
gray
gradient
gradient_x
gradient_y
laplacian
```

禁止新增 Hough、LSD 或语义模型。

### 9.3 块与支持域

初始参数：

| 参数 | 值 |
|---|---:|
| block height | 32 px |
| block stride | 16 px |
| seam support radius | 6 px |
| feature halo radius | 12 px |
| normal lag range | `-3...+3 px` |
| lag step | `0.5 px` |
| minimum component length | 12 px |
| minimum support count | 12 |
| strong gradient threshold | `max(8, local P75)` |
| minimum correlation | 0.65 |
| best/second uniqueness | 10% |
| maximum orientation difference | 10° |

不要使用整幅高度 median；局部纸盒边会被货架横梁和墙面淹没。

### 9.4 法向搜索

不能只搜索 vertical lag 并称为任意方向检测。应：

1. 在 seam `±6 px` 和当前 32-row block 中找共同有效强边支持；
2. 以梯度方向 modulo π 估计主法向；
3. 沿该法向执行 `-3...+3 px / 0.5 px` 的 1-D bilinear sampling；
4. 比较 gradient magnitude agreement 和 orientation agreement；
5. 要求相关性与最优/次优唯一性；
6. 记录 edge-normal lag。

示意伪代码：

```python
for block in overlapping_blocks(height=32, stride=16):
    support = common_valid(block, seam_radius=6)
    support &= strong_gradient_support(left_features, right_features)

    component = strongest_cross_seam_component(support)
    if component.length < 12 or component.count < 12:
        record_unevaluable()
        continue

    normal = dominant_gradient_normal_modulo_pi(component)
    scores = []
    for lag in np.arange(-3.0, 3.01, 0.5):
        right_sample = bilinear_sample_along_normal(right_features, component, normal, lag)
        scores.append(gradient_and_orientation_score(left_features, right_sample))

    require_minimum_correlation(scores)
    require_best_second_uniqueness(scores)
    record_best_normal_lag(scores)
```

输出至少包含：

```json
{
  "schema": "gemini305-video-s13-oblique-structure-audit/v1",
  "evaluable": true,
  "supported_block_count": 0,
  "supported_component_count": 0,
  "block_best_lag_px": [],
  "median_supported_abs_lag_px": null,
  "p95_supported_abs_lag_px": null,
  "maximum_supported_abs_lag_px": null,
  "minimum_correlation": null,
  "minimum_uniqueness_margin": null,
  "maximum_orientation_difference_degrees": null,
  "multiple_layer_or_ambiguous": false
}
```

### 9.5 visual_suspect 条件

只有在有充分强边支持时才触发：

```text
edge_registration.evaluable == true
AND
(
    seam_local_lk_p95 > 1.25 px
    OR edge median abs lag > 0.75 px
    OR edge P95 abs lag > 1.0 px
)
```

规则：

- `unevaluable` 不是失败，保持原快速路径；
- 弱纹理不能触发；
- 多边缘、多层或搜索不唯一只记录 ambiguous，不能进入可选 micro rescue；
- 原有 long-horizontal catastrophe guard 完整保留；
- 本指标只控制当前 pair 是否继续检查候选，不控制 P2 能否封存；
- owner/topology/map hard failure 仍必须阻止 P2 seal。

---

## 10. T4：风险 pair 才解除 early-stop 并比较现有候选

修改：

```text
src/panorama_demo/video_s13_m5.py
estimate_s13_m5_transactions()
```

建议封装：

```python
_evaluate_s13_seam_geometry_combination(...)
```

单个组合必须返回冻结的数据结构，包含：

- seam candidate；
- alignment；
- 被采样 geometry candidate；
- map hard safety；
- seam-local evidence；
- correspondence P95；
- edge registration；
- horizontal catastrophe；
- preview metrics；
- 是否 `visual_suspect`；
- 所有失败原因。

### 10.1 快速路径

```python
for seam in ranked_seam_candidates:
    alignment = reestimate_with_seam_local_evidence(seam)
    primary_geometry = alignment.selected
    result = evaluate_small_roi(seam, primary_geometry)

    if result.hard_safe and not result.visual_suspect:
        select(result)
        break
```

正常 pair 仍只检查第一个 hard-safe seam/geometry 组合。

### 10.2 suspect 路径

```python
if result.hard_safe and result.visual_suspect:
    evaluate_other_accepted_C0_to_C4_compact_maps_for_same_seam()

    if a_clean_geometry_exists:
        select_best_clean_geometry()
        break

    continue_to_next_ranked_seam_candidate()
```

几何候选只使用已经生成的 C0–C4 compact maps，不重新调用 GFTT/LK。

候选选择顺序：

1. 先通过所有既有 map hard safety；
2. 再要求 long-horizontal catastrophe guard 通过；
3. 对有充分支持的强边，优先选择 edge-normal P95 最小的候选；
4. edge P95 相差 `≤0.15 px` 时选择更简单模型：

   ```text
   C0 → C1 → C2 → C3 → C4
   ```

5. 同一 seam 没有 clean geometry 时再检查下一条 ranked S2/S1/S0；
6. 不得以旧 straight 候选预先否决 DP；
7. 不得为了增加 DP 数量跳过任何硬门。

### 10.3 所有候选仍 suspect

局部视觉残差不能让整个 P2 自动失败。若至少存在一个结构上 hard-safe 的组合：

- 以原始首个 hard-safe 组合作为 fail-closed baseline；
- 只有其他组合明确非退化时才替换；
- 否则保留 baseline；
- 记录 `unresolved_oblique_structure=true`；
- 继续生成 P2，但保持 diagnostic/manual-review 语义。

若 owner、bounds、Jacobian、support、topology 或 replay 失败，则仍必须阻止 P2 seal。

### 10.4 frozen alignment 的正确替换

`S13PairAlignment` 是 frozen dataclass。选择另一个 geometry candidate 时禁止原地改属性，必须：

```python
selected_alignment = dataclasses.replace(
    alignment,
    selected_model=chosen.model,
    selected_candidate_index=chosen_index,
)
```

并确保：

- `alignment.selected` 返回新选中的 map；
- transaction 的 selected model 与 replay 完全一致；
- final P2 renderer 使用同一 map。

### 10.5 不吞掉实现错误

当前 pair 外层 broad `except Exception` 会把任意程序错误伪装成正常 fallback。v5 路径必须改成：

- 只捕获明确声明的、可恢复的 pair candidate 异常；
- assertion、shape bug、索引错误、schema 错误等必须向上抛出；
- v3/v4 若需兼容可保留旧行为，但 v5 必须记录：

  ```text
  unexpected_exception_fallback=false
  ```

真实四分支必须满足：

```text
unexpected_exception_fallback_count = 0
```

---

## 11. T5：缓存结构特征，保证净耗时不增加

当前 `run_s13_m5()` 在 geometry/final 全景生成后，会为每条 seam 重复构建 Lab、gray、Sobel 和 Laplacian。

修改：

```text
src/panorama_demo/video_s13_m5.py
src/panorama_demo/video_s13_quality.py
```

要求：

1. geometry 全景只调用一次 `prepare_seam_structure()`；
2. final 全景只调用一次 `prepare_seam_structure()`；
3. 后续每条 seam 的 symmetric、普通 structure 和 horizontal audit 传入缓存；
4. pair candidate 的 left features 每 pair 只构建一次；
5. 每个实际采样的 right candidate 最多构建一次；
6. 不缓存所有候选的全 crop，只保留当前/最佳小 ROI；
7. post-render audit 仅索引 seam corridor，不反复转换整幅图。

已观察的性能预算：108 pair 重复整图 audit 约 `15.1 s`，corridor/cached 方式约 `1.2 s`。实现应利用这部分预算覆盖新增风险 ROI 检查，并争取总时间下降。

新增性能计数：

```text
gftt_call_count
forward_pyr_lk_call_count
backward_pyr_lk_call_count
full_canvas_feature_build_count
pair_feature_build_count
risk_pair_count
extra_seam_candidate_evaluation_count
extra_geometry_roi_sample_count
micro_rescue_attempt_count
p2_full_resolution_render_count
extra_full_resolution_render_count
formal_raw_rgb_remap_invocations
depth_call_count
dis_call_count
open3d_call_count
orbslam3_call_count_in_m5
m4_geometry_reestimation_count
gain_enumeration_count
```

硬期望：

```text
gftt_call_count == pair_count
forward_pyr_lk_call_count == pair_count
backward_pyr_lk_call_count == 0
full_canvas_feature_build_count == 2
p2_full_resolution_render_count == 2
extra_full_resolution_render_count == 0
depth_call_count == 0
dis_call_count == 0
open3d_call_count == 0
orbslam3_call_count_in_m5 == 0
m4_geometry_reestimation_count == 0
gain_enumeration_count == 0
micro_rescue_attempt_count == 0  # 首轮
```

候选 ROI sampling 不得错误计为 full-resolution render。

---

## 12. T6：transaction、fallback 与 replay

### 12.1 pair transaction v3

所有 v5 pair 路径，包括正常、局部 fallback 和 topology repair，都必须使用：

```text
gemini305-video-s13-m5-pair-transaction/v3
```

至少新增：

```json
{
  "correspondence_filter": {
    "detected_count": 0,
    "status_count": 0,
    "finite_count": 0,
    "target_in_bounds_count": 0,
    "target_valid_count": 0,
    "error_filtered_count": 0,
    "final_count": 0,
    "error_median": null,
    "error_mad": null,
    "error_limit": null
  },
  "seam_local_evidence": {
    "requested_half_widths_px": [16, 24],
    "selected_half_width_px": null,
    "raw_count": 0,
    "selected_count": 0,
    "minimum_required_count": 36,
    "sufficient_for_train_held_out": false,
    "full_shoulder_fallback_used": false,
    "mode": "seam_local_or_insufficient"
  },
  "edge_registration": {
    "evaluable": false,
    "visual_suspect": false,
    "supported_block_count": 0,
    "supported_component_count": 0,
    "median_supported_abs_lag_px": null,
    "p95_supported_abs_lag_px": null,
    "ambiguous": false
  },
  "selection_continued_for_structure": false,
  "selected_seam_rank": 0,
  "selected_geometry_rank": 0,
  "seam_fallback_used": false,
  "geometry_fallback_used": false,
  "unresolved_oblique_structure": false,
  "unexpected_exception_fallback": false,
  "micro_rescue": {
    "enabled": false,
    "attempted": false,
    "accepted": false
  }
}
```

### 12.2 修正 fallback 语义

不得继续用：

```python
fallback_used = seam == S0 and geometry == C0
```

代替所有 fallback。

至少拆分：

```text
seam_fallback_used
geometry_fallback_used
topology_repair_used
unexpected_exception_fallback
```

DP+C0、DP+C1 仍可能是 geometry fallback，必须如实记录。

更新 selection policy 字符串以匹配真实行为，例如：

```text
minimum_local_objective_with_supported_edge_residual_veto
```

不得继续写与实现不符的旧 policy。

### 12.3 replay 精确一致

必须断言：

- `pair_count == source_count - 1`；
- transaction 严格按 `m5-pair-N` 排序；
- 每个 replay pair 恰有两个相邻真实源；
- replay corridor 和 secondary width 仍不超过 8 px；
- replay seam 与 selected seam 完全一致；
- replay UV/valid 与正式 P2 render 完全一致；
- `primary_owner_right_mask` 与 owner 完全一致；
- `parent_pair_transaction_sha256` 精确匹配 v3 JSON；
- 篡改任一字段后 loader 和 seal 都 fail-closed。

不要另增 sealed `seam_local_edge_report.json`。首轮把所有必需信息写入 pair transactions；验收报告由脚本从 transaction 派生并放在 validation 目录，避免扩张核心 schema 面。

---

## 13. T7：测试矩阵

### 13.1 新增与扩展文件

新建：

```text
tests/test_video_s13_m51_r2.py
```

扩展：

```text
tests/test_video_s13_m5.py
tests/test_video_s13_hard_audit.py
tests/test_video_s13_contract.py
tests/test_video_s13_experiment.py
tests/test_video_s13_m61_streaming.py
tests/test_video_s13_m9.py
scripts/summarize_s13_m51_validation.py
README.md
```

README 需要同步修正当前 peer seam candidate + catastrophic hard gate 的实际语义，不要继续描述旧 straight 前置授权。

### 13.2 LK 测试

必须覆盖：

- status false；
- source/target NaN/Inf；
- target x/y 越界；
- target 落在 `right_valid=false`；
- 单个和多个 error outlier；
- MAD 为 0；
- error 点不足 16；
- 过滤后少于 36 点不恢复坏点；
- GFTT/LK 调用次数不增加；
- global `x_offset` 符号正确。

### 13.3 seam-local evidence 测试

必须覆盖：

- 直 seam；
- 每行变化的 curved seam；
- `±16` 足够；
- `±16` 不足而 `±24` 足够；
- `±24` 仍不足；
- 风险路径不回完整 shoulder；
- reference/moving 分别按自己的 y 索引 seam；
- global/local 坐标不可混用；
- 每条 S2/S1/S0 用自己的 evidence；
- application band 仍最大 ±8；
- immutable P0 不变。

### 13.4 边缘检测测试

合成方向：

```text
0° / 15° / 30° / 45° / 75° / 接近垂直
```

合成错位：

```text
0 / 0.5 / 1 / 2 / 3 px
```

必须覆盖：

- 安全对齐不触发；
- 2 px 斜边错层触发；
- 其余 14 个 block 放强货架横梁，仅一个 block 放断裂斜边仍能检出；
- 32-row block 边界上的斜边不会漏检；
- 黑色有效内容不被删除；
- 弱纹理为 unevaluable 且不触发；
- 多边缘/遮挡为 ambiguous；
- 只做 vertical lag 的错误实现不能通过接近垂直边测试；
- correlation 和 uniqueness 不足时不误判。

### 13.5 候选选择测试

必须覆盖：

- clean 第一个候选保持 early-stop；
- suspect 第一个 geometry 会检查同 seam 其他 accepted C0–C4；
- 同 seam 都 suspect 时检查下一条 seam；
- 指标相差 ≤0.15 px 时选更简单模型；
- 所有候选仍 suspect 时保留最佳 hard-safe baseline 并标 unresolved；
- 普通局部视觉失败不阻止 P2；
- owner/Jacobian/bounds/support/topology 失败仍阻止 P2；
- `dataclasses.replace()` 后 alignment、transaction、renderer、replay 选择一致；
- v5 意外程序异常不会被 broad fallback 吞掉；
- topology repair 的 transaction schema 和新字段完整。

### 13.6 版本与 lineage 测试

必须覆盖：

- v3/v4 identity 仍被接受；
- v3/v4 新功能关闭且固定合成输入输出逐像素不变；
- v5 identity/config/self-hash/manifest-hash 正确；
- v5 P2 completion 使用 `/v5`；
- v5 current_latest 最终为 P2；
- v5 不生成 P3/M6；
- v5 + 显式 `run_m6=True` 失败；
- v5 resume 到 v4 M6 失败；
- v5 config 引用旧 threshold/approval 失败；
- 篡改 P0/P1 parent hash 失败；
- 篡改 transaction、replay seam/UV/valid/transaction SHA 失败；
- 篡改 owner/provenance/completion schema/hash 失败；
- 旧 v4 formal M6 测试仍全部通过。

### 13.7 性能调用计数测试

必须 monkeypatch 断言：

```text
full_canvas_feature_build_count == 2
p2_full_resolution_render_count == 2
extra_full_resolution_render_count == 0
GFTT == pair_count
forward LK == pair_count
backward LK == 0
M4 reestimate == 0
gain enumeration == 0
depth/DIS/Open3D/ORB in M5 == 0
micro rescue == 0
```

---

## 14. T8：测试命令

使用项目正式 Python：

```powershell
$G305Python = 'D:\Panoramic_Camera\.conda\python.exe'
```

先跑定向测试：

```powershell
& $G305Python -m pytest -q `
  tests/test_video_s13_m51_r2.py `
  tests/test_video_s13_m5.py `
  tests/test_video_s13_hard_audit.py `
  tests/test_video_s13_contract.py `
  tests/test_video_s13_experiment.py `
  tests/test_video_s13_m61_streaming.py `
  tests/test_video_s13_m9.py
```

然后：

```powershell
ruff check src tests scripts
& $G305Python -m compileall -q src tests scripts
git diff --check
& $G305Python -m pytest -q
```

不得只跑新测试；完整回归必须通过。

---

## 15. T9：四分支真实 P2-v5 验收

### 15.1 固定输入

fast：

```text
data/captures/video/run_20260807_140140
```

fast direct trajectory：

```text
data/captures/video/run_20260807_140140/orbslam3_trajectory.json
SHA256: 19566989ab091fd1b710f0237f4b0d1689bbf374e3dcb24da60811a2af7b4961
```

slow：

```text
data/captures/video/run_20260806_153033
```

slow direct trajectory：

```text
data/captures/video/run_20260806_153033/orbslam3_trajectory.json
SHA256: 1f0f0bd795e1030fbf1c5c2d916537fc095ab0c4d2ac6447810ac48fd386cb57
```

以上 direct trajectory 必须由现有严格 loader 再验证，不能插值、外推或重新运行 ORB。

四个分支固定为：

```text
fast_direct
fast_ignore_pose
slow_direct
slow_ignore_pose
```

### 15.2 新输出根

```text
artifacts/S013_M5_1_r2_v5_acceptance
```

不得覆盖：

```text
artifacts/S013_M5_1
artifacts/S013_M6_final
outputs/S013_M1_M9_full_run_20260813_000442
```

### 15.3 命令模板

```powershell
$G305Experiment = 'D:\Panoramic_Camera\.conda\Scripts\g305-video-experiment.exe'
$Candidate = 'D:\central_strip_Panoramic_Camera\configs\video_candidates\s013\S013_output_first_progressive_dense_central_slit_v5.yaml'
$Acceptance = 'D:\central_strip_Panoramic_Camera\artifacts\S013_M5_1_r2_v5_acceptance'

& $G305Experiment `
  'D:\central_strip_Panoramic_Camera\data\captures\video\run_20260807_140140' `
  --output "$Acceptance\fast_direct" `
  --algorithm candidate `
  --candidate-config $Candidate `
  --trajectory-cache 'D:\central_strip_Panoramic_Camera\data\captures\video\run_20260807_140140\orbslam3_trajectory.json' `
  --report-level full `
  --artifact-level audit

& $G305Experiment `
  'D:\central_strip_Panoramic_Camera\data\captures\video\run_20260807_140140' `
  --output "$Acceptance\fast_ignore_pose" `
  --algorithm candidate `
  --candidate-config $Candidate `
  --ignore-pose `
  --report-level full `
  --artifact-level audit

& $G305Experiment `
  'D:\central_strip_Panoramic_Camera\data\captures\video\run_20260806_153033' `
  --output "$Acceptance\slow_direct" `
  --algorithm candidate `
  --candidate-config $Candidate `
  --trajectory-cache 'D:\central_strip_Panoramic_Camera\data\captures\video\run_20260806_153033\orbslam3_trajectory.json' `
  --report-level full `
  --artifact-level audit

& $G305Experiment `
  'D:\central_strip_Panoramic_Camera\data\captures\video\run_20260806_153033' `
  --output "$Acceptance\slow_ignore_pose" `
  --algorithm candidate `
  --candidate-config $Candidate `
  --ignore-pose `
  --report-level full `
  --artifact-level audit
```

v5 默认必须停在 P2；若以上命令生成 P3/M6，视为实现失败。

### 15.4 每分支输出

必须保存：

- P0/P1/P2 completion；
- canonical P2 PNG；
- provenance NPZ；
- hard audit；
- performance；
- pair transactions；
- narrow replay；
- current_latest；
- seam model/geometry model 计数；
- risk/suspect/unresolved/fallback 计数；
- LK filter 数量与 error 分布；
- seam-local evidence 宽度与不足计数；
- extra candidate/ROI sample 计数；
- worst 10 oblique-edge pair；
- before/after/contact sheet。

验收报告可由扩展后的：

```text
scripts/summarize_s13_m51_validation.py
```

生成，但只能总结事实，不能自动授予 production 或 M6 资格。

### 15.5 fast-direct 重点 ROI

必须自动导出：

```text
pair 65
pair 66
pair 71
pair 78
x = 1436 ... 1648
```

比较：

```text
旧 P2/P3
新 P2-v5
owner overlay
seam overlay
edge-normal metric
selected seam/geometry
LK filter/evidence audit
```

### 15.6 真实验收门

| 项目 | 硬要求 |
|---|---:|
| P2 hard audit | pass |
| current_latest | P2 |
| P3/M6/delivery | 不存在 |
| seam crossing | 0 |
| invalid Jacobian | 0 |
| source out-of-bounds | 0 |
| internal hole | 0 |
| total map displacement | `≤8 px` |
| target ROI edge-normal P95 | `≤1 px` |
| 断裂/double-edge 长度 | 相对旧结果下降 `≥50%` |
| unexpected exception fallback | 0 |
| full-resolution P2 renders | 2 |
| extra full-resolution renders | 0 |
| raw decode/formal remap | 不增加 |
| GFTT/forward LK | 不增加 |
| backward LK | 0 |
| production lock | 不变 |

对于非触发 pair：

- v5 行为应尽可能保持旧结果；
- 若因 seam-local evidence 设计而改变，必须在 transaction 中给出可解释原因；
- 不允许无原因的大面积 C0 回退。

### 15.7 可复现性

四个分支至少各重复运行一次。允许 generation id、绝对路径和时间字段变化，但以下决策性内容应相同：

- canonical P2 image；
- seam arrays；
- source UV/valid/provenance；
- pair selection payload；
- risk/suspect/unresolved 决策；
- replay pair 内容。

---

## 16. T10：真实性能验收

使用同机、相同输入、相同 Python、相同配置，做 5 次 warm run，比较旧 v4 与新 v5。

硬调用计数优先于 wall time：

```text
P2 full render = 2
extra full render = 0
GFTT = pair_count
forward LK = pair_count
backward LK = 0
raw decode/remap 不增加
M4/gain/depth/DIS/Open3D/ORB in M5 = 0
```

时间验收：

```text
patched M5 median <= baseline M5 median
patched M5 P95 <= baseline M5 P95 + 2% measurement noise
```

若 suspect pair 过多导致耗时增长：

1. 先确认 feature cache 是否真的为 full canvas 仅 2 次；
2. 确认 edge audit 只运行于已采样的小 corridor；
3. 确认 clean pair 仍 early-stop；
4. 不得通过降低结构安全阈值换速度；
5. 若仍无法持平，停止并报告，不扩展算法。

---

## 17. 可选阶段：C2E component-local micro rescue

### 17.1 首轮决策门

完成 T0–T10 后先判断：

```text
如果 fast-direct pair 65/66/71/78 的强斜边 P95 已 ≤1 px，
且断裂/double-edge 长度至少下降 50%，
则立即停止，不实现 C2E。
```

只有仍存在“证据充分、单一强边、搜索唯一、1–3 px 残差”的 pair，才可以另开实现任务。

### 17.2 允许的 C2E 形式

名称：

```text
C2E_edge_normal_micro_translation
```

它不是自由整高 `(du,dv)`，而是一个 component-local edge-normal 标量：

```text
normal lag search: -3 ... +3 px
step: 0.5 px
一次只处理一个最强、单层、跨 seam component
x taper: 在现有 ±8 px application band 边界归零
y taper: component 上下各扩 8–12 行后归零
```

必须从 immutable P0 一次性构建最终 inverse map：

```python
combined_delta = base_candidate_delta + component_local_delta
source_map = compose_from_immutable_p0(combined_delta)
```

禁止：

- 对已经 warp 的图再次 warp；
- 给整高 `target_delta_u/v` 直接加常数；
- 改 seam、owner、source frame 或 pose；
- 把强边交给 blend；
- 多边缘、多层、遮挡或不唯一证据进入 C2E。

### 17.3 C2E 接受条件

必须全部满足：

```text
held-out edge P95 <= 1.0 px
absolute improvement >= 0.5 px
relative improvement >= 30%
forward/reverse edge-correlation lag discrepancy <= 0.5 px
best/second uniqueness >= 10%
orientation difference <= 10°
support retention >= 0.98
minimum Jacobian >= 0.5
micro displacement <= 3 px
total map displacement <= 8 px
horizontal catastrophe guard passes
halo regression <= 0.25 px
finite/in-bounds passes
```

任一失败即 rollback，保留首轮最佳 hard-safe candidate，并记录 unresolved。

C2E 若实施，将改变 P2-v5 算法定义；必须创建新的 successor identity/schema，不得修改已经封存和验收的 v5 generation。

---

## 18. 后续 M6 lineage：本轮不实施

只有 v5 四分支 P2 验收完成后，才允许另开 full-chain successor，建议使用新身份而不是修改 v5，例如：

```text
S013_output_first_progressive_dense_central_slit_v6
```

后续必须：

1. 以四个 sealed P2-v5 为新父级；
2. 重新生成 photometric evidence/calibration inputs；
3. 重新校准 M6 quality thresholds；
4. 重新生成四 P2 parent binding；
5. 取得新的人工 threshold approval；
6. 使用新的 config/threshold/approval SHA；
7. 才允许 P3/M6 replay；
8. 保持 sealed v5 不变。

禁止“暂时”复用：

```text
quality_thresholds_m61_v2.json
旧 threshold_approval.json
旧四 P2 parent hash
```

本任务书完成条件只到 P2-v5，不包含 M6/P3。

---

## 19. 实施顺序

严格按以下顺序：

### T0 基线冻结

- 记录 HEAD、status、旧 hash、旧测试和旧真实性能；
- 保存关键 pair/ROI；
- 不改旧 artifact。

### T1 版本与失败测试

- 新增 v5 P2-only identity skeleton；
- 先写 LK、seam-local、斜边、early-stop 和 lineage 失败测试；
- 确认旧代码不能通过。

### T2 LK 净化

- 实现 hard-valid 与 robust error filter；
- 不增加 LK；
- 完成审计计数。

### T3 seam-local evidence

- 实现 `±16/±24` 过滤；
- 修正 final-corridor 语义；
- 禁止 full shoulder fallback。

### T4 edge-normal audit

- 实现 32-row 重叠块/component-local 法向检查；
- 复用特征；
- 保留 horizontal guard。

### T5 suspect-only candidate continuation

- clean pair 保持 early-stop；
- suspect pair 比较现有 C0–C4，再检查下一 seam；
- 修正 frozen alignment 与 transaction/replay 一致性。

### T6 缓存与性能计数

- full canvas feature build 固定为 2；
- 所有调用计数进入 performance；
- 消除重复整图审计。

### T7 P2-v5 seal 与 lineage

- 完成 v5 config/manifest/completion；
- v5 默认停在 P2；
- 所有旧 M6 入口 fail-closed。

### T8 单元、定向、完整回归

- 定向测试；
- lint/compile/diff check；
- 完整 pytest；
- v3/v4 非回归。

### T9 四分支真实 P2

- 新输出根；
- 导出目标 ROI 和 worst 10；
- hard audit、replay、可复现性。

### T10 性能与停止决策

- 5 次 warm run；
- median/P95；
- 判断是否已经达到目标；
- 达标立即停止，不实现 C2E。

---

## 20. 强制停止条件

发生以下任一情况，立即停止并报告，不自行扩大范围：

- 需要第三次全景 render；
- 需要第二次/反向 LK；
- 需要 depth、DIS、mesh、Open3D、ORB 或跨 pair 优化；
- 无法保持 immutable P0 的一次 inverse sampling；
- 无法隔离 v3/v4；
- P2 hard audit、owner、provenance 或 replay 任一失败；
- 风险 pair 证据显示为多视差层、遮挡或透明问题，而不是单层局部残差；
- edge search 不唯一或支持不足；
- wall time 无法由缓存抵消；
- 只有复用旧 M6 threshold 才能继续；
- 真实目标 ROI 已达标；
- 需要修改 production lock；
- 发现当前工作树中有无法安全合并的用户改动。

---

## 21. 禁止的错误实现

以下任何一项出现都视为任务未完成：

- 只改变 application band，没有过滤拟合 evidence；
- 对 NaN/OOB 点先 round/cast 再检查；
- 点少时恢复 invalid、OOB 或高 error LK 点；
- 风险 pair 最后回完整 96 px shoulder；
- 只搜索 y lag 却声称支持任意方向；
- 用整幅高度 median 检测纸盒斜边；
- 在 owner-only 图上代替两源图判断 geometry；
- 对整高 band 应用自由 `(du,dv)`；
- 在已 warp 图上再次 warp；
- 直接修改 frozen alignment，导致 replay 仍用旧 map；
- `visual_suspect` 被升级为整个 P2 的一般画质失败；
- 删除原 long-horizontal catastrophe guard；
- 恢复 straight 对 DP 的前置授权；
- 通过放宽 `1.5 px`、Jacobian、support 或 8 px 位移门来增加通过数；
- broad `except` 吞掉 v5 程序 bug；
- transaction 仍使用错误的 fallback/policy 语义；
- 新建 v5 YAML，但共享代码无条件改变 v4；
- 修改旧 v4 config/hash/generation；
- v5 继续使用旧 M6 threshold/approval；
- 增加第三次全图渲染或额外 raw decode；
- 只跑合成测试，不跑四分支真实数据；
- 只给报告，不实现代码。

---

## 22. 完成标准

本任务书首轮只有在以下条件全部满足时才算完成：

### 22.1 功能

- LK hard-valid/error filter 已实现；
- final geometry evidence 随 candidate seam 移动；
- 强斜边局部检测已实现；
- clean pair 保持早停；
- suspect pair 会比较其他现有 geometry/seam；
- 所有候选仍 suspect 时 output-first 且明确 unresolved；
- 首轮没有 C2E。

### 22.2 安全

- P2 hard audit 全通过；
- owner/provenance/replay 精确一致；
- seam/topology/Jacobian/bounds/support 不放宽；
- v3/v4 行为隔离；
- v5 严格 P2-only；
- production lock 不变。

### 22.3 质量

- fast-direct pair 65、66、71、78 有完整对照；
- 目标强边 edge-normal P95 `≤1 px`；
- 断裂/double-edge 长度下降 `≥50%`；
- 不出现无解释的大面积新回退；
- 四分支 worst 10 已人工可审查。

### 22.4 性能

- full canvas feature build = 2；
- P2 full render = 2；
- GFTT/forward LK 不增加；
- backward LK = 0；
- raw decode/remap 不增加；
- M5 median 不高于基线；
- M5 P95 仅允许 2% 测量噪声。

### 22.5 验证

- 新测试通过；
- 旧定向测试通过；
- 完整 pytest 通过；
- lint/compile/diff check 通过；
- 四分支各至少两次决策性输出可复现；
- current_latest 均为 P2；
- 没有 P3/M6/delivery。

---

## 23. Codex 最终交付格式

完成后按以下顺序报告：

1. 实际分支和最终 HEAD；
2. 工作树状态；
3. 修改文件列表及职责；
4. 新 v5 identity/config/schema/hash；
5. 新增和修改的测试；
6. 定向测试结果；
7. 完整测试、lint、compile、diff check 结果；
8. 四分支 generation/P2 路径；
9. 每分支 P2 hard audit 与 current_latest；
10. pair 65、66、71、78 before/after ROI；
11. risk/suspect/unresolved/fallback 统计；
12. LK filter 和 seam-local evidence 统计；
13. seam/geometry model 计数；
14. owner/provenance/replay hash 验证；
15. 全分辨率 render、raw remap、GFTT/LK 和 feature-build 计数；
16. 5 次 warm-run median/P95；
17. 是否达到停止门；
18. C2E 是否被禁止或为何需要另开任务；
19. 明确说明本轮未运行 M6/P3、未修改 production lock；
20. 真实 Open3D、真实 ORB、现场速度验收状态分别说明，不能用单元测试代替。
