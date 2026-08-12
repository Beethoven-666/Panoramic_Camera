# Panoramic Camera S1.3 M5.1 单向流水线、独立 DP 接缝与硬安全门改造任务书

## 0. 给 Codex 的执行指令

请直接在当前仓库中完成代码修改、测试和真实数据验证，不要只输出分析或设计建议。

仓库：`Beethoven-666/Panoramic_Camera`

基准分支：`codex/video-realtime-seam-v6`

本轮名称：`S013 M5.1 Forward Pipeline`

本轮严格停止在 M5.1。完成 P2 生成、封存、指针更新和四组真实数据验证后停止。不要进入 M6，不要加入颜色优化、亮度场、羽化、MultiBand、深度补图、网格修复或 production 冻结。

本轮必须实现以下总原则：

> 单向阶段流水线 + 硬安全门 + 评分仅作离线参考。

不要再让 P0、P1、P2 之间的综合评分控制父级、回退、继续执行或运行时指针。

---

## 1. 当前问题

当前实现有两类结构性问题。

### 1.1 M5 重复执行 M4 选择

当前 `src/panorama_demo/video_s13_experiment.py` 中的 `_run_m5()` 会再次调用：

```python
estimate_s13_vertical(...)
select_s13_vertical_parent(...)
```

这会在 M5 内重新估计纵向修正、重新枚举 gain、重新渲染多个全分辨率结果，并再次选择 P0 或 P1。

改造后，M5 只能读取已经封存并通过哈希验证的 P1。M5 不得重新估计 M4，不得重新枚举 gain，不得重新选择 P0 或 P1。

### 1.2 DP 接缝被 shifted straight 前置否决

当前 `src/panorama_demo/video_s13_m5.py` 中 `_select_held_out_safe_seam()` 存在类似逻辑：

```python
if dp is not None and straight is not None and allowed_rank >= 2 and straight_safe:
```

这意味着 shifted straight 必须先通过，DP 才能接受检查。直线失败时，DP 即使已经生成，也不会获得独立验证。

改造后，以下三类接缝必须是同级候选：

- `S0_midpoint_straight`
- `S1_shifted_straight`
- `S2_monotone_dp`

DP 是否检查和是否接受，不得依赖 shifted straight 是否通过。

### 1.3 综合评分仍控制 P2 是否成为运行时结果

当前 `run_s13_m5()` 会使用 `sequence_structure_decision()`、`selected_as_best` 和 `current_preview.json` 决定 P2 是否被选中。

这套逻辑必须退出运行时控制。评分仍可计算并输出，但只能用于离线比较和人工复核。

---

## 2. 本轮目标

本轮必须完成以下结果。

1. P0、P1、P2 按固定方向执行。
2. P1 封存成功后，M5 只从该 P1 继续。
3. P2 只要通过阶段硬安全审计，就必须封存并成为 `current_latest.json`。
4. P2 即使综合评分下降、改善低于 0.5%、改善为 0 或与 P1 哈希接近，也不能因此回退到 P1。
5. 单个 pair 的 geometry、shifted straight 或 DP 失败时，只回退该 pair。
6. midpoint、shifted straight、DP 作为同级候选生成和排序。
7. DP 不再需要直线先通过。
8. 每个实际参与硬审计的候选都在自己的接缝走廊中估计 geometry。
9. 结构灾难、非法采样、非法 Jacobian、过大位移、接缝交叉等问题继续被硬门拦截。
10. 运行时指针与人工推荐指针分开。
11. 删除 M5 中重复的 M4 全图选择，减少全分辨率渲染次数。
12. 保留完整 provenance、parent hash、pair transaction 和 completion seal。

---

## 3. 明确不做的内容

本轮不得做以下工作。

- 不修改 production renderer。
- 不修改 production lock。
- 不自动推广到 production。
- 不进入 M6。
- 不加入 photometric gain、bias 或亮度场。
- 不加入 feather、MultiBand 或其他融合。
- 不使用深度进行精细补图。
- 不加入 mesh、TSDF、source rescue。
- 不改动 P0 的帧选择、运动推进、轨迹来源和画布定义。
- 不重新设计 M4 纵向算法。
- 不删除现有 M4 gain 选择代码。只是不允许 M5 再调用它。
- 不覆盖或删除 `artifacts/S013_M5` 中的旧结果。
- 不以“真实数据必须接受至少 N 条 DP”为硬编码门限。
- 不允许为了提高 DP 数量而放宽 Jacobian、位移、越界或结构灾难保护。

---

## 4. 目标阶段关系

改造后的阶段关系固定为：

```text
P0 sealed
  -> P1 sealed
  -> M5 读取 P1
  -> 每个 pair 生成并检查接缝与 geometry 候选
  -> P2 hard audit
  -> P2 sealed
  -> current_latest = P2
```

评分关系为：

```text
P0/P1/P2 diagnostic metrics
  -> 写入离线报告
  -> 不控制父级
  -> 不控制 P2 封存
  -> 不控制 M6 是否允许继续
  -> 不自动修改 current_reviewed
```

---

## 5. 指针模型

保留现有 `current_base.json`，新增两个指针。

### 5.1 current_base.json

继续指向可用的 P0 基础图。

它只表示基础结果，不表示最新结果，也不表示人工推荐结果。

### 5.2 current_latest.json

表示最新的、已封存且通过硬审计的阶段，供下一阶段继续处理。

允许的阶段顺序：

```text
P0 < P1 < P2 < M6
```

本轮只实现和验证 P0、P1、P2。为 M6 预留阶段顺序，不实现 M6。

建议结构：

```json
{
  "schema": "gemini305-video-s13-current-latest/v1",
  "generation_id": "...",
  "generation": "generations/...",
  "stage": "P2",
  "completion": "generations/.../P2/P2_completion.json",
  "completion_sha256": "...",
  "result_asset": "generations/.../P2/geometry_and_seam_panorama_owner_only.png",
  "result_asset_sha256": "...",
  "parent_stage": "P1",
  "parent_completion_sha256": "..."
}
```

更新要求：

- 必须先调用 `verify_stage()`。
- 必须验证 completion SHA。
- 必须验证结果文件 SHA。
- 不得指向 `.pending` 目录。
- 使用临时文件、flush、fsync 和 `os.replace()` 原子更新。
- 评分字段不得出现在更新条件中。
- 新阶段失败时保留原指针。

### 5.3 current_reviewed.json

表示人工检查或固定离线数据集确认后的推荐结果。

建议结构：

```json
{
  "schema": "gemini305-video-s13-current-reviewed/v1",
  "generation_id": "...",
  "generation": "generations/...",
  "stage": "P1",
  "completion": "generations/.../P1/P1_completion.json",
  "completion_sha256": "...",
  "result_asset": "generations/.../P1/vertical_panorama_owner_only.png",
  "result_asset_sha256": "...",
  "review_method": "manual",
  "reviewer": "...",
  "reviewed_at_utc": "...",
  "note": "..."
}
```

要求：

- 只能由显式人工命令或离线选型命令更新。
- 普通 M4、M5 运行不得自动更新。
- 新 generation 开始时不得清空。
- 评分程序不得自动更新。

### 5.4 current_preview.json 兼容处理

不要继续把 `current_preview.json` 作为流水线父级。

建议做法：

- 保留旧文件读取兼容。
- 新流水线不读取它。
- 新 M5 不自动写入它。
- 可以在显式更新 `current_reviewed.json` 时同步写一个兼容副本。
- 在兼容文件中加入 `deprecated=true` 和 `runtime_authority=false`。

---

## 6. P1 必须成为可重放的封存阶段

当前 P1 已保存图像、provenance、`vertical_solution.npz`、`pair_report.json` 和 completion。请把它整理成可严格重建 `S13VerticalSolution` 的版本化格式。

### 6.1 在 video_s13_vertical.py 中增加统一序列化接口

建议新增：

```python
def save_s13_vertical_solution(
    directory: Path,
    solution: S13VerticalSolution,
) -> dict[str, object]:
    ...


def load_s13_vertical_solution(
    directory: Path,
    *,
    expected_completion_sha256: str | None = None,
) -> S13VerticalSolution:
    ...
```

保存内容至少包括：

- `global_offsets_px`
- 每个 pair 的 `local_row_residuals`
- `pairs` 的全部 dataclass 字段
- `selected_gain`
- `audit`
- pair 数量
- 画布高度
- assignment 数量
- 数组 key 与 pair index 的映射
- 序列化 schema

建议文件：

```text
P1/vertical_solution.npz
P1/vertical_solution.json
```

`vertical_solution.json` 建议结构：

```json
{
  "schema": "gemini305-video-s13-vertical-solution/v2",
  "selected_gain": 1.0,
  "pair_count": 84,
  "assignment_count": 85,
  "canvas_height": 480,
  "global_offsets_key": "global_offsets_px",
  "local_row_residual_keys": [
    "pair_0000_local_row_residual_px"
  ],
  "pairs": [],
  "audit": {},
  "npz_sha256": "..."
}
```

加载时必须验证：

- schema 正确。
- JSON 和 NPZ SHA 一致。
- assignment 数量一致。
- pair 数量等于 assignment 数量减 1。
- global offset 数量正确。
- 每个 local row residual 长度等于画布高度。
- 数组全部有限。
- `selected_gain` 与 P1 completion 一致。
- pair report 与 solution JSON 一致。

如果旧 P1 v1 能被无损重建，可以增加兼容 loader。如果缺少必要字段，必须明确报错 `legacy_p1_not_replayable`，不得偷偷重新运行 M4。

### 6.2 P1 completion 升级

建议升级为：

```text
gemini305-video-s13-p1-vertical-completion/v2
```

completion 中加入：

- `hard_audit_passed=true`
- `vertical_solution_json`
- `vertical_solution_json_sha256`
- `vertical_solution_npz`
- `vertical_solution_npz_sha256`
- `result_asset`
- `result_asset_sha256`
- `pixel_provenance_sha256`
- `p0_parent_sha256`

P1 完成封存和验证后，更新 `current_latest.json` 到 P1。不要根据 P1 相对 P0 的评分决定是否更新。

---

## 7. M5 只读取封存 P1

修改 `src/panorama_demo/video_s13_experiment.py` 中 `_run_m5()`。

### 7.1 删除 M5 内的重复选择

M5 中不得再调用：

```python
estimate_s13_vertical(...)
select_s13_vertical_parent(...)
render_s13_p1_from_raw(...)
```

这些函数可以继续存在并供 M4 使用，但 M5 不得调用。

### 7.2 新的 _run_m5() 流程

参考伪代码：

```python
def _run_m5(...):
    p0_dir = generation / "P0"
    p1_dir = generation / "P1"

    p0_completion = verify_stage(...)
    p1_completion = verify_stage(...)

    p0_completion_sha = sha256_file(p0_completion_path)
    p1_completion_sha = sha256_file(p1_completion_path)

    vertical = load_s13_vertical_solution(
        p1_dir,
        expected_completion_sha256=p1_completion_sha,
    )

    p1_image = read_and_verify_image_from_completion(p1_dir, p1_completion)
    p1_provenance = read_and_verify_npz_from_completion(p1_dir, p1_completion)

    m5 = run_s13_m5(
        schedule,
        calibration,
        image_loader,
        vertical,
        p1_image,
        parent_stage="P1",
        parent_completion_sha256=p1_completion_sha,
        parent_result_sha256=p1_completion["result_asset_sha256"],
        ...,
    )

    write_p2_assets(...)
    hard_audit = audit_s13_p2_stage(...)

    if not hard_audit.passed:
        write_failure_audit(...)
        do_not_update_current_latest()
        raise StageHardAuditError(...)

    seal_p2(...)
    verify_p2(...)
    update_current_latest(... stage="P2")
```

### 7.3 父级绑定

P2 和所有 pair transaction 必须绑定：

- `parent_stage=P1`
- `parent_completion_sha256`
- `parent_result_sha256`
- `p0_ancestor_completion_sha256`

不要再把 P2 的父级哈希设为临时生成的 `vertical_selection.json` 哈希。

### 7.4 P1 参考文件

P2 可以保留 `selected_vertical_parent.png` 兼容文件，但只能从 P1 已封存图像复制或硬链接，不能重新渲染。

建议新增：

```text
P2/p1_parent_reference.json
```

其中记录相对路径和哈希。

---

## 8. 接缝候选改成同级候选

主要修改 `src/panorama_demo/video_s13_m5.py`，必要时修改 `src/panorama_demo/video_s13_seam.py`。

### 8.1 删除旧的串行授权关系

以下关系必须删除：

```text
midpoint 通过后才考虑 shifted
shifted 通过后才考虑 DP
preliminary 降级后 final 不得升级
```

不得再使用 `straight_safe` 作为 DP 检查条件。

不得再使用 preliminary 选出的 `maximum_model` 作为 final iteration 的模型上限。

### 8.2 新增候选数据结构

建议新增：

```python
@dataclass(frozen=True)
class S13SeamCandidate:
    candidate_id: str
    model_code: str
    model_name: str
    seam_x_by_row: np.ndarray
    local_objective: float
    complexity_rank: int
    generation_audit: Mapping[str, object]


@dataclass(frozen=True)
class S13CandidateEvaluation:
    candidate: S13SeamCandidate
    evaluation_status: str
    alignment: S13PairAlignment | None
    geometry_model: str
    geometry_audit: Mapping[str, object]
    seam_hard_audit: Mapping[str, object]
    horizontal_hard_audit: Mapping[str, object]
    diagnostic_metrics: Mapping[str, object]
    hard_gate_passed: bool
    hard_gate_failures: tuple[str, ...]
    result_hash: str | None
```

模型命名：

```text
S0 / midpoint_straight
S1 / shifted_straight
S2 / monotone_dp
```

### 8.3 候选生成

每个 pair 必须先生成可用候选集合。

```python
candidates = build_s13_seam_candidates(...)
```

规则：

- midpoint 必须始终存在。
- shifted straight 在搜索区有效时生成。
- DP 在 cost volume 和有效路径存在时生成。
- 某一候选无法生成时记录明确原因。
- 不得因为其他候选失败而把它标记为未生成。
- `select_s13_seam()` 如未暴露每个候选的总 cost，请扩展返回结构。

每个候选必须记录：

- seam hash
- total path cost
- color cost
- gradient magnitude cost
- gradient direction cost
- double edge cost
- thin strong structure cost
- motion residual cost
- boundary offset cost
- slope cost
- curvature cost
- search bounds
- 最大 shift
- row step 约束

### 8.4 局部排序

局部目标函数可以继续使用。它只决定当前 pair 的候选顺序，不决定 P2 是否允许生成。

建议排序方式：

1. 使用同一 cost volume 下的候选 total cost。
2. slope 和 curvature 已包含在 total cost 中，不要再次重复加权。
3. cost 完全相同或非常接近时，优先选择简单模型。
4. 可以保留 DP 相对最佳直线至少改善 2% 的局部复杂度偏好，但必须满足以下条件：
   - 只用于当前 pair 排序。
   - 不得影响 P2 封存。
   - 不得影响 `current_latest`。
   - 不得影响 M6 是否可继续。
   - 必须写入候选 audit。

建议：

```python
ordered_candidates = rank_candidates_by_local_objective(candidates)
```

### 8.5 高效而独立的检查顺序

为了避免每个 pair 无条件运行三次 geometry，可采用以下方式：

1. 所有候选先生成并完成轻量 seam path 检查。
2. 根据局部目标排好顺序。
3. 按排序依次执行候选自己的 geometry 和硬审计。
4. 第一个通过全部硬门的候选成为当前 pair 结果。
5. 低优先级候选可以记录为 `skipped_after_higher_rank_safe_candidate`。
6. 开启审计模式时，允许检查全部候选。

建议增加开发配置：

```text
G305_S13_M5_AUDIT_ALL_CANDIDATES=0|1
```

默认运行使用 0。真实诊断可对一组数据使用 1。

关键要求：

- 检查顺序只能由局部目标排序决定。
- 不得固定先检查 midpoint，再检查 shifted，再检查 DP。
- DP 排在第一时，必须先独立检查 DP。
- shifted 失败不能跳过 DP。

### 8.6 每个被检查候选使用自己的 geometry 走廊

每个实际进入 geometry 检查的候选必须调用候选自己的 final corridor re-estimation。

不得：

- 使用 midpoint 的 geometry 审计 DP。
- 使用 shifted straight 的 geometry 审计 DP。
- 使用 preliminary 降级后的 model cap 限制 final 候选。

transaction 中记录：

- candidate seam hash
- corridor bounds
- geometry input grid hash
- geometry result hash
- alignment model
- map delta hash
- hard gate结果

---

## 9. 局部失败处理

每个 pair 独立处理。

### 9.1 候选失败

某个候选发生以下问题时，只拒绝该候选：

- 无有效 DP 路径
- 坐标非有限
- 采样越界
- Jacobian 非正
- 位移超过限制
- support retention 不足
- held-out geometry residual 超界
- seam 超出搜索区
- row step 非法
- 与相邻 seam 交叉
- 长水平结构出现灾难性错层

然后继续检查当前 pair 的下一个候选。

### 9.2 当前 pair 全部复杂候选失败

回退顺序：

```text
当前最优安全候选
  -> shifted straight
  -> midpoint straight
  -> midpoint straight + identity geometry
```

最终 fallback 必须绑定 P1 原始采样网格。

### 9.3 midpoint + identity 仍不合法

这不再属于普通 pair 优化失败，而属于父级或全局拓扑损坏。

此时：

- 写入明确的 stage hard failure。
- 不封存 P2。
- 不更新 `current_latest` 到 P2。
- 保持已有 `current_latest`。
- 不得伪造成功 P2。

---

## 10. 把硬安全门与画质评分分开

建议新增：

```text
src/panorama_demo/video_s13_hard_audit.py
```

`video_s13_quality.py` 中的画质指标可以保留，但不能继续同时承担硬安全授权。

### 10.1 三类指标

#### A. 候选局部目标

用于算法内部选择：

- seam total cost
- slope cost
- curvature cost
- boundary offset
- geometry residual
- held-out geometry residual

这类指标可以控制当前 pair 候选顺序。

#### B. 硬安全门

决定候选或阶段是否合法：

- parent seal 和 hash
- sampling map 有限
- source 坐标在范围内
- positive Jacobian
- 最大位移
- support retention
- seam 路径合法
- seam 不交叉
- owner 唯一
- valid 和 owner 一致
- provenance 完整
- 无内部孔洞
- 长水平结构无灾难性错层
- completion 和资产哈希完整

#### C. 离线诊断指标

只记录，不控制运行：

- color jump
- gradient jump
- double edge
- motion residual
- thin structure
- horizontal edge mismatch
- aggregate score
- mean improvement percentage
- legacy selected_as_best

### 10.2 structurally_non_degrading 的处理

当前 `structurally_non_degrading()` 会因为任意单项小幅变差而拒绝候选。

改造要求：

- 保留它用于离线报告或兼容报告。
- 不得直接作为 DP、shifted straight 或 P2 的硬安全门。
- 不得因为 gradient jump、color jump 等轻微变化而直接拒绝一个通过硬安全门的候选。

### 10.3 sequence_structure_decision 的处理

保留统计功能，但取消运行时权限。

建议改名或封装为：

```python
build_sequence_structure_diagnostic(...)
```

输出可以继续包括旧规则下是否会被选中，但必须写：

```json
{
  "runtime_authority": false,
  "legacy_policy_result": true,
  "diagnostic_only": true
}
```

删除“至少改善 0.5%”作为运行时条件。

---

## 11. 长水平结构灾难保护

必须保留防止黄色横梁、管道和长直线被切成波浪的硬保护，但不能继续要求每个细小指标都不变差。

建议新增：

```python
def long_horizontal_structure_catastrophe_guard(
    before: Mapping[str, object],
    after: Mapping[str, object],
) -> tuple[bool, str | None, Mapping[str, object]]:
    ...
```

初始硬门建议使用以下条件，并将常量集中定义和测试：

- before 支持充分时，after 不得完全丢失长水平结构。
- 新增绝对垂直 lag 达到 2 px 或以上时拒绝。
- 相比 before，最佳垂直 lag 增加 2 px 或以上时拒绝。
- zero-lag correlation 下降超过 0.20 时拒绝。
- before 支持行数不少于 64 时，支持行数下降超过 50% 时拒绝。
- 只在支持充分时触发，避免低纹理噪声误判。

这些常量必须放入版本化配置或冻结 dataclass，不得散落在函数内部。

现有 `long_horizontal_structure_nondegrading()` 保留为诊断，不再作为唯一硬门。

必须增加旧坏 P2 回归测试，确保明显的波浪横梁仍会被拒绝。

旧结果参考：

```text
D:\central_strip_Panoramic_Camera\artifacts\S013_M5\fast_direct\generations\20260811T215130-011198de41e8\P2
```

不要要求新输出复现旧 P2。旧 P2 只作为负面结构回归参考。

---

## 12. 接缝和全局拓扑硬审计

### 12.1 单条 seam 审计

每条 seam 必须满足：

- shape 等于画布高度。
- 所有值有限并可转为整数坐标。
- 在当前 pair 搜索范围内。
- 相邻行步长只允许 `-1、0、1`，或沿用现有更严格规则。
- 最大水平偏移不超过配置值，当前默认保持 8 px。
- 不与前后 seam 交叉。
- 不破坏最小 owner 宽度。

### 12.2 seam family 审计

全部 pair 选择完成后执行全局检查：

- 按 pair index 有序。
- 任意行上 seam 坐标严格不交叉。
- 所有 owner 区域有合法宽度。
- seam 数量等于 assignment 数量减 1。

如果发现局部交叉：

1. 找到涉及的 pair。
2. 对该 pair 使用下一个安全候选。
3. 如果仍冲突，回退 midpoint。
4. 最多进行一次确定性修复遍历。
5. 修复后再次执行全局审计。
6. 仍失败才判定 P2 stage hard failure。

### 12.3 owner、valid 和 provenance

P2 必须满足：

- 每个 valid 像素只有一个 owner。
- `valid == owner_frame_id >= 0`。
- `owner_source_index` 在 assignment 范围内。
- valid 像素的 `source_u/source_v` 全部有限。
- valid 像素的 source 坐标在原图范围内。
- `geometry_transaction_id` 和 `seam_transaction_id` 与 owner 区域一致。
- 所有 transaction id 指向存在的 pair transaction。
- secondary 权重继续为 0，因为本轮仍是 owner-only。

### 12.4 无内部孔洞

渲染时累计每个像素可被多少 source 有效覆盖，得到 `expected_support_mask`。

计算：

```text
hole_mask = expected_support_mask AND NOT final_valid_mask
```

与画布外部无效区域连通的部分可以视为外部边界。位于 expected support 内部且不与外部连通的无效连通域视为内部孔洞。

任何内部孔洞都必须导致 stage hard audit 失败，或在封存前进行确定性的局部 owner fallback。不得使用补色、模糊或邻域复制伪造有效像素。

---

## 13. Geometry 硬门

保留当前 C0 到 C4 的有限模型体系和已有安全限制。

每个实际检查的 candidate geometry 必须验证：

- candidate 从不可变基础采样网格构建。
- map 全部有限。
- map 在 source 范围内。
- linear Jacobian 为正。
- sampled minimum Jacobian 为正。
- 最大 map displacement 不超过现有上限。
- translation、rotation、scale、shear 不超过现有上限。
- taper 连续回到 identity。
- support retention 达标。
- held-out residual 满足现有模型硬界限。

本轮不得为了增加 DP 接受数量而放宽这些 geometry 上限。

geometry 某个模型失败时，继续检查该候选的更简单 geometry 模型。候选全部 geometry 模型失败时，再检查下一个 seam 候选。

---

## 14. Pair transaction 新结构

建议升级为：

```text
gemini305-video-s13-m5-pair-transaction/v2
```

每个 pair transaction 至少包含：

```json
{
  "transaction_id": "m5-pair-0000",
  "parent_stage": "P1",
  "parent_completion_sha256": "...",
  "parent_result_sha256": "...",
  "p0_ancestor_completion_sha256": "...",
  "pair_frame_ids": [0, 35],
  "candidate_generation": [],
  "candidate_evaluations": [],
  "selection_policy": "minimum_local_objective_among_hard_safe_candidates",
  "selected_candidate_id": "pair-0000-S2",
  "selected_seam_model": "monotone_dp",
  "selected_geometry_model": "C1_accepted_vertical",
  "pair_hard_audit_passed": true,
  "fallback_used": false,
  "fallback_reason": null,
  "diagnostic_before_metrics": {},
  "diagnostic_after_metrics": {},
  "result_stage_sha256": "..."
}
```

每个 candidate evaluation 至少包含：

```json
{
  "candidate_id": "pair-0000-S2",
  "model_code": "S2",
  "model_name": "monotone_dp",
  "generation_status": "generated",
  "evaluation_status": "selected",
  "local_objective": 519.74,
  "relative_to_best_straight": -0.1732,
  "seam_sha256": "...",
  "geometry_model": "C1_accepted_vertical",
  "geometry_audit": {},
  "seam_hard_audit": {},
  "horizontal_hard_audit": {},
  "hard_gate_passed": true,
  "hard_gate_failures": [],
  "diagnostic_metrics": {}
}
```

允许的 `evaluation_status`：

- `selected`
- `rejected_hard_gate`
- `generation_failed`
- `skipped_after_higher_rank_safe_candidate`
- `fallback_identity`

禁止使用含糊状态：

- `dp_vs_straight = null`，但不说明原因
- `straight_held_out_structure_not_proved` 导致 DP 未检查
- `selected_as_best=false` 导致整个 pair 或 P2 被撤销

对于任何已生成的 DP，必须有明确状态。

---

## 15. P2 阶段输出

建议目录：

```text
P2/
  geometry_panorama_owner_only.png
  geometry_and_seam_panorama_owner_only.png
  geometry_and_seam_panorama_owner_only.jpg
  seam_overlay.png
  p2_valid_mask.png
  p2_pixel_provenance.npz
  p1_parent_reference.json
  pair_transactions.json
  pair_transactions/
    pair_0000.json
    ...
  hard_audit.json
  diagnostic_quality_report.json
  performance.json
  worst_seam_crops/
  P2_completion.json
```

### 15.1 hard_audit.json

必须包含：

- `passed`
- parent verification
- owner topology
- valid/provenance consistency
- source coordinate checks
- internal hole count
- seam family checks
- Jacobian summary
- displacement summary
- horizontal catastrophe summary
- pair fallback count
- fatal failures

### 15.2 diagnostic_quality_report.json

保留现有画质统计，但加入：

```json
{
  "schema": "gemini305-video-s13-m5-diagnostic-quality/v2",
  "diagnostic_only": true,
  "runtime_authority": false,
  "legacy_minimum_improvement_fraction": 0.005,
  "legacy_policy_result": false,
  "before_mean_score": 7.65,
  "after_mean_score": 6.36,
  "relative_change": -0.1684
}
```

无论结果是多少，都不能改变 P2 completion 和 `current_latest`。

### 15.3 P2 completion

建议升级为：

```text
gemini305-video-s13-p2-completion/v2
```

至少包含：

- `stage=P2`
- `sealed=true`
- `hard_audit_passed=true`
- `parent_stage=P1`
- `parent_completion_sha256`
- `parent_result_sha256`
- `p0_ancestor_completion_sha256`
- `hard_audit_sha256`
- `diagnostic_quality_report_sha256`
- 所有资产 SHA
- pair transaction 数量
- selected seam 模型计数
- fallback 计数
- formal raw RGB remap 次数
- source 数量

`selected_as_best` 如因兼容需要暂时保留，必须标记：

```json
{
  "selected_as_best": false,
  "selected_as_best_deprecated": true,
  "selected_as_best_runtime_authority": false
}
```

不能再有：

```python
if completion["selected_as_best"] is True:
    update_pointer(...)
```

P2 指针更新只看 `hard_audit_passed`、seal 和 hash。

---

## 16. M5Result 数据结构

建议把 `S13M5Result` 扩展为：

```python
@dataclass(frozen=True)
class S13M5Result:
    pairs: tuple[S13M5Pair, ...]
    geometry_result: S13P2Result
    final_result: S13P2Result
    seam_overlay: np.ndarray
    hard_audit_passed: bool
    hard_audit: Mapping[str, object]
    diagnostic_quality: Mapping[str, object]
    before_mean_score: float | None
    after_mean_score: float | None
    performance: Mapping[str, float]
```

兼容策略：

- 不要把 `selected_as_best` 重新解释为 hard audit。
- 如果旧测试或读取器需要该字段，可以暂时保留为 diagnostic-only 字段。
- 所有运行时分支改用 `hard_audit_passed`。

---

## 17. 性能改造

本轮性能收益主要来自删除 M5 中重复的 M4 选择，而不是关闭 DP。

### 17.1 禁止的重复工作

M5 不得：

- 再次估计 vertical solution。
- 再次枚举四个 gain。
- 为每个 gain 做多套完整全景渲染。
- 再次比较 P0 和 P1 作为父级。

### 17.2 允许的全分辨率渲染

正常 M5 最多保留：

1. geometry owner-only 全图。
2. final seam owner-only 全图。

候选判断应使用 pair corridor crop 和缓存，不得为每个 seam 候选渲染完整全景。

### 17.3 performance.json

建议字段：

```json
{
  "p1_verify_and_load": 0.0,
  "candidate_generation": 0.0,
  "candidate_geometry": 0.0,
  "pair_hard_audit": 0.0,
  "geometry_global_render": 0.0,
  "final_global_render": 0.0,
  "diagnostic_metrics": 0.0,
  "artifact_export": 0.0,
  "total_m5": 0.0,
  "m4_reestimated_in_m5": false,
  "gain_enumeration_count_in_m5": 0,
  "m4_selection_full_resolution_render_count": 0,
  "p2_full_resolution_render_count": 2
}
```

测试应检查调用次数，不要用固定秒数作为 CI 硬门。真实运行报告前后时间。

---

## 18. 单元测试和集成测试

优先修改：

```text
tests/test_video_s13_m5.py
tests/test_video_s13_experiment.py
tests/test_video_s13_selection.py
tests/test_video_s13_real_regression.py
tests/test_video_s13_contract.py
```

建议新增：

```text
tests/test_video_s13_bundle.py
tests/test_video_s13_hard_audit.py
```

### 18.1 DP 独立检查测试

新增：

```python
def test_dp_is_evaluated_when_shifted_straight_fails():
    ...
```

构造：

- shifted straight 硬门失败。
- DP 已生成且局部目标更优。
- DP geometry 和硬门通过。

断言：

- DP 被独立检查。
- DP audit 非空。
- DP 可以被选中。
- 不出现 `straight_held_out_structure_not_proved` 导致 DP 跳过。

### 18.2 preliminary 不再限制 final

```python
def test_preliminary_downgrade_does_not_cap_final_candidates():
    ...
```

构造 preliminary 选择 midpoint，final corridor 下 DP 安全。

断言 final 仍可选择 DP。

### 18.3 候选自己的 geometry 走廊

```python
def test_candidate_geometry_is_reestimated_in_its_own_corridor():
    ...
```

记录每次 re-estimation 的 seam hash，断言每个被检查候选使用自己的 seam。

### 18.4 局部失败不影响其他 pair

```python
def test_pair_failure_falls_back_only_that_pair():
    ...
```

让一个 pair 的 DP 和 geometry 全部失败。

断言：

- 该 pair 回退 midpoint identity。
- 其他 pair 继续使用各自结果。
- P2 仍生成。
- pair transaction 数量完整。

### 18.5 局部成本排序

新增：

- DP 明显优于直线时，DP 排在前面。
- DP 只改善极小且复杂度偏好生效时，直线排在前面。
- 该规则只影响 pair，不影响 stage。

### 18.6 硬 geometry 门

分别测试：

- nonfinite map 被拒绝。
- source 越界被拒绝。
- nonpositive Jacobian 被拒绝。
- displacement 超限被拒绝。
- support retention 不足被拒绝。
- held-out residual 超界被拒绝。

### 18.7 seam 拓扑门

分别测试：

- row step 非法。
- 超出搜索区。
- 相邻 seam 交叉。
- owner 宽度为 0。
- seam 数量错误。

### 18.8 长横线灾难回归

新增：

```python
def test_wave_beam_is_rejected_by_catastrophe_guard():
    ...
```

使用合成横梁或从旧坏结果提取的小型 fixture。

断言明显的 2 px 以上错层或大幅相关性下降会被拒绝。

再新增：

```python
def test_minor_gradient_change_does_not_reject_safe_dp():
    ...
```

断言轻微 gradient jump 变化只进入 diagnostic，不触发硬回退。

### 18.9 M5 不重复 M4

```python
def test_m5_never_reestimates_or_reselects_m4(monkeypatch):
    ...
```

把 `estimate_s13_vertical` 和 `select_s13_vertical_parent` monkeypatch 为一旦调用就抛错。

M5 应仍能从封存 P1 完成。

### 18.10 P1 round-trip

```python
def test_sealed_p1_solution_round_trip_is_exact():
    ...
```

断言保存后加载得到：

- global offsets 完全一致。
- local residual arrays 完全一致。
- selected gain 一致。
- pair 字段一致。
- audit 一致。
- 渲染像素哈希一致。

### 18.11 评分不再控制 P2

新增三组测试：

- diagnostic 改善为 0.091%，P2 仍封存并更新 latest。
- diagnostic 变差，P2 只要 hard audit 通过仍封存并更新 latest。
- diagnostic 改善 16.84%，但 hard audit 失败，P2 不得封存。

### 18.12 指针测试

测试：

- P0 seal 后 latest=P0。
- P1 seal 后 latest=P1。
- P2 hard pass 后 latest=P2。
- 坏 hash 不更新 latest。
- current_reviewed 不会被普通运行修改。
- 显式 review 命令验证 hash 后才更新 reviewed。
- current_preview 不再作为父级读取。

### 18.13 provenance 和孔洞测试

测试：

- `valid == owner >= 0`。
- valid source 坐标有限。
- owner source index 合法。
- transaction id 可解析。
- 内部孔洞被拒绝。
- 外部边界无效区域不被误判为内部孔洞。

---

## 19. 真实数据验证

使用现有真实数据和现有 S1.3 CLI，不要重新采集数据。

数据：

```text
fast: data/captures/video/run_20260804_162340
slow: data/captures/video/run_20260806_153033
```

运行四组：

```text
fast_direct
fast_ignore_pose
slow_direct
slow_ignore_pose
```

复用生成 `artifacts/S013_M5/validation_summary.json` 时的同一调用方式、配置和轨迹输入。不要自行更换帧集或采样策略。

新输出根目录：

```text
artifacts/S013_M5_1
```

不得覆盖旧 `artifacts/S013_M5`。

### 19.1 每组必须输出

- P0 completion 和图像。
- P1 completion、图像和可重放 solution。
- P2 completion 和图像。
- `current_latest` 状态快照。
- hard audit。
- diagnostic quality report。
- performance report。
- seam overlay。
- pair transaction 汇总。
- DP candidate 状态统计。
- hard gate rejection reason 统计。
- fallback 统计。

### 19.2 DP 验证要求

不设置“每组至少接受一条 DP”的硬门，因为这会再次变成人为配额。

但必须满足：

- 任何已生成 DP 都有明确状态。
- DP 不得因 shifted straight 失败而跳过。
- `fast_direct` 中旧 `pair_0000` 对应区域必须确认 DP 获得独立检查。
- 如果四组最终仍为 0 条 DP，报告必须列出每条 DP 的明确硬失败原因或局部排序跳过原因。
- 不允许出现大量 `dp_vs_straight=null` 且没有原因。
- 必须有合成测试证明安全 DP 能被系统接受。

### 19.3 旧坏 P2 防回归

对照旧目录：

```text
artifacts/S013_M5/fast_direct/generations/20260811T215130-011198de41e8/P2
```

至少生成以下对照：

```text
validation/old_bad_vs_new_contact_sheet.png
validation/high_risk_horizontal_crops/
validation/seam_model_counts.json
```

不能仅用综合 score 声称新图更好。必须说明：

- 波浪横梁硬门是否触发。
- 哪些 pair 被回退。
- 新 P2 是否通过全局 hard audit。
- 人工 review 状态是否仍为 `not_reviewed`。

---

## 20. validation_summary.json

新建：

```text
artifacts/S013_M5_1/validation_summary.json
```

建议结构：

```json
{
  "schema": "gemini305-video-s13-m51-validation/v1",
  "policy": "forward_pipeline_hard_gate_diagnostic_only",
  "runs": {
    "fast_direct": {
      "generation": "...",
      "latest_stage": "P2",
      "hard_audit_passed": true,
      "diagnostic_review_status": "not_reviewed",
      "seam_counts": {
        "midpoint_straight": 0,
        "shifted_straight": 0,
        "monotone_dp": 0
      },
      "dp_generated": 0,
      "dp_evaluated": 0,
      "dp_selected": 0,
      "pair_fallback_count": 0,
      "hard_rejection_histogram": {},
      "time_to_p0": 0.0,
      "time_p1": 0.0,
      "time_m5": 0.0,
      "m4_reestimated_in_m5": false,
      "m5_gain_enumeration_count": 0,
      "m5_full_resolution_render_count": 2
    }
  }
}
```

此文件只能总结事实，不自动选出 reviewed 结果。

---

## 21. 测试命令

先运行定向测试：

```bash
python -m pytest tests/test_video_s13_m5.py -q
python -m pytest tests/test_video_s13_experiment.py -q
python -m pytest tests/test_video_s13_selection.py -q
python -m pytest tests/test_video_s13_hard_audit.py -q
python -m pytest tests/test_video_s13_bundle.py -q
python -m pytest tests/test_video_s13_real_regression.py -q
python -m pytest tests/test_video_s13_contract.py tests/test_video_s13_dispatch.py -q
```

然后运行完整回归：

```bash
python -m pytest -q
```

检查项目现有 `pyproject.toml`、`ruff.toml` 或其他配置。如果项目已经配置 lint 或 type check，运行现有命令。不要自行引入新的格式化工具和大范围格式化。

---

## 22. 完成标准

以下条件必须全部满足。

### 22.1 代码结构

- `_run_m5()` 不再估计 vertical solution。
- `_run_m5()` 不再调用 M4 parent selection。
- M5 从封存 P1 精确加载 solution 和 parent image。
- P2 parent hash 绑定 P1 completion。
- DP 检查不依赖 `straight_safe`。
- preliminary 结果不再限制 final 可用模型。
- hard audit 与 diagnostic quality 分离。
- `selected_as_best` 不再拥有运行时权限。
- M5 不再自动更新 `current_preview`。
- P2 hard pass 后更新 `current_latest`。
- current_reviewed 只允许显式更新。

### 22.2 安全

- 旧波浪横梁问题有回归测试。
- finite、bounds、Jacobian、位移、support、seam topology、owner、provenance、孔洞和 seal 全部有测试。
- 单个 pair 失败只局部回退。
- 全局非法结果不会被封存为 P2。

### 22.3 DP

- 测试证明 shifted straight 失败时 DP 仍可被检查和接受。
- 测试证明 final 可以从 preliminary midpoint 升级到 DP。
- 每个已生成 DP 都有明确状态。
- 不通过硬门的 DP 有具体原因。

### 22.4 性能

- M5 中 M4 重估次数为 0。
- M5 中 gain 枚举次数为 0。
- M5 中 M4 选择全图渲染次数为 0。
- 正常 P2 全图渲染次数不超过 2。
- 四组真实数据给出改造前后时间对比。

### 22.5 真实结果

- 四组均生成 P0、P1、P2 或明确的 hard failure 报告。
- 正常数据不再因综合改善不足而保留父级。
- `current_latest` 指向最新 hard-pass 阶段。
- `current_reviewed` 保持不变，除非显式人工更新。
- 不以单一综合 score 声称视觉成功。

---

## 23. 实施顺序

严格按以下顺序工作。

### T1 基线冻结

- 记录当前 HEAD。
- 保存现有定向测试结果。
- 保存四组旧 validation summary。
- 不修改旧 artifact。

### T2 先写失败测试

先添加以下测试并确认旧代码失败：

- M5 不重复 M4。
- DP 不依赖 shifted straight。
- preliminary 不限制 final。
- 评分不控制 P2。
- pointer 分离。
- 波浪横梁硬门。

### T3 P1 可重放

- 完成 solution v2 保存和加载。
- 完成 round-trip 测试。
- 完成 P1 completion v2。

### T4 指针分离

- 实现 current_latest。
- 实现 current_reviewed。
- 降级 current_preview 权限。
- 完成原子和坏 hash 测试。

### T5 M5 父级改造

- 删除 M5 内 M4 重算和选择。
- 直接读取 P1。
- 改正 parent hash。
- 增加性能调用计数。

### T6 接缝候选改造

- 生成同级候选。
- 局部排序。
- 候选自己的 geometry。
- 取消 straight 前置条件。
- 取消 preliminary model cap。

### T7 硬审计与诊断分离

- 新建 hard audit 模块。
- 迁移运行时安全检查。
- 保留 quality diagnostics。
- 移除 0.5% 运行时门。

### T8 P2 封存和输出

- P2 completion v2。
- hard audit 文件。
- diagnostic quality 文件。
- current_latest 更新。

### T9 全部测试

- 定向测试。
- 完整回归。
- 修复兼容性问题。

### T10 四组真实验证

- 使用新 artifact root。
- 生成 validation summary。
- 生成旧坏 P2 对照图。
- 汇总 DP、fallback、hard gate 和时间。

完成 T10 后严格停止，不进入 M6。

---

## 24. 禁止的错误实现

以下实现视为未完成。

- 只是把 `minimum_mean_improvement_fraction` 从 0.005 改成 0。
- 只是把 `selected_as_best` 永远设为 true。
- 删除所有结构检查。
- 为增加 DP 数量而跳过 Jacobian 或位移检查。
- shifted straight 失败时仍不检查 DP。
- preliminary 选择 midpoint 后 final 仍只允许 midpoint。
- P2 继续从重新生成的 M4 solution 运行。
- P2 pointer 继续依赖 `current_preview`。
- `current_reviewed` 被程序自动覆盖。
- 候选检查使用完整全景反复渲染。
- pair 失败导致整条流水线回退到 P0。
- 评分下降导致不封存 hard-pass P2。
- 只跑单元测试，不跑四组真实数据。
- 只提交报告，不修改代码。

---

## 25. Codex 最终交付格式

完成后给出以下内容。

1. 当前分支和最终 commit SHA。
2. 修改文件列表及每个文件的作用。
3. 新增和修改的测试列表。
4. 定向测试结果。
5. 完整回归结果。
6. 四组真实数据输出路径。
7. 每组 P0、P1、P2 的尺寸、contributors 和时间。
8. 每组 seam 模型计数。
9. DP generated、evaluated、selected 数量。
10. DP hard rejection 原因直方图。
11. pair fallback 数量和原因。
12. hard audit 结果。
13. current_latest 和 current_reviewed 状态。
14. 改造前后 M5 全分辨率渲染次数和运行时间。
15. 旧波浪横梁回归结果和对照图路径。
16. 明确声明已停止在 M5.1，未进入 M6。

不要仅使用“评分改善百分比”作为成功结论。真实图尚未人工确认时，明确标记 `not_reviewed`。
