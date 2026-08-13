# Panoramic Camera S013：M6–M9 Forward Pipeline 完整实施方案 v1

## 面向 Codex 的可直接执行版本

```text
仓库：D:\central_strip_Panoramic_Camera
远端仓库：Beethoven-666/Panoramic_Camera
开发分支：codex/video-realtime-seam-v6
基线提交：8a2b5d8cdc4deccbd5617cedb1c50330a99daa52
算法 ID：S013_output_first_progressive_dense_central_slit_v4
实现 ID：s013_output_first_progressive_dense_central_slit_m61_v2
现行 M6：M6.1 光度接缝抑制（P2-v4 → P3-v2）
规范封存链：P0 → P1 → P2 → P3
当前自动视觉验收：blocked（原始 unresolved/regressed 记录保留）
当前人工推进状态：M6 stage_exit=completed_manual_forward，接受当前 P3
当前 M7 授权：authorized_manual_forward（仅限明确的局部 M7 component）
当前运行指针：current_latest
人工复核指针：current_reviewed
current_preview：已废弃，不得恢复运行权限
角色：独立 diagnostic candidate
production renderer：不得修改
production lock：不得创建或修改
S1、S1.1、S1.2 历史产物：不得作为运行输入
Depth / Open3D / TSDF：默认不参与 M6–M9 主路径
```

本文只规划 **M5.1 之后的 M6、M7、M8、M9**。  
M0–M5.1 的算法逻辑、P0/P1/P2 像素、封存资产和 transaction 不允许被后续阶段静默改写。

`S013_output_first_progressive_dense_central_slit_v4` 是本文所称的正式 S013 M6 路径；“正式”只表示该 diagnostic candidate 内的规范身份、父级、seal、provenance 和 pointer 契约。旧 v3 继续作为历史/兼容输入，不能代表现行 M6，也不能授权 M7。公共 `g305-video-panorama`、production renderer、production lock 与正式 delivery 仍完全在范围外。

---

# 0. 当前基线与为什么必须重新规划

## 0.1 M5.1 已建立的新事实

当前 M5.1 已经完成以下架构修正：

1. 流水线固定为 `P0 → P1 → P2` 单向执行。
2. M5 只校验并加载封存 P1，不再重新运行 M4。
3. M5 不再枚举 M4 gain，也不重新选择 P0/P1 父级。
4. midpoint、shifted straight、DP 是同级 seam 候选。
5. DP 不依赖 shifted straight 或其他候选先通过。
6. 每个被实际评估的 seam 候选拥有自己的 final geometry corridor。
7. hard audit 与 diagnostic quality 完全分离。
8. P2 只要通过 hard audit，就封存并更新 `current_latest=P2`。
9. `current_reviewed` 只能显式写入。
10. completion、parent chain、result、provenance、transaction 都通过 SHA-256 绑定。
11. M5 全分辨率渲染已经从旧路径的多次枚举降到两次。
12. 四组真实数据均完成 P0、P1、P2 封存，并保持 `not_reviewed`。

因此，M6–M9 不能继续沿用旧方案中“后级重新选择前级”“综合评分决定能否发布”“current_preview 代表最佳结果”等旧思路。

## 0.2 旧 M6–M9 中必须修正的地方

| 旧写法 | 更新后的处理 |
| --- | --- |
| M6 目录或阶段可直接命名为 `M6` | M6 是实施里程碑，正式像素阶段必须命名为 `P3` |
| M7 可从 P3 quality 或零散人工标注直接启动 | 只能消费 SHA 绑定的正式 M7 handoff；自动路线要求 `completed_target`，人工继续路线要求单独的 `completed_manual_forward` 授权记录，且 handoff 只能包含明确属于 M7 的孤立局部问题 |
| M8 更新 `current_preview` | `current_preview` 不再有运行权限；M8 只显式更新 `current_reviewed` |
| P5 Reviewed 作为新的像素阶段 | Review 不产生新像素，不创建 P5；它是指针和 evaluation bundle |
| 画质评分可以阻止后级资产发布 | 画质只决定局部候选；hard-safe 的 identity/owner-only 始终可封存 |
| M6 可重新估计 geometry/seam | M6 严禁重估 P1/P2，只能重放封存 P2 |
| M6 可继续枚举 1 px / MultiBand | 现行 M6 只允许 B0 owner-only 或 B1 2 px；B2–B4 全部禁用 |
| M6 可启用低频亮度场/Q4 | 现行 M6 的 Q4 solver 关闭，规范 field 值为 0；需要它属于新身份的 M6.x |
| M9 优化后继续沿用旧 evaluation lock | 任何非精确等价优化都会使旧 lock 失效，必须重新跑 M8 |

## 0.3 更新后的最终阶段关系

```text
M6 产生 P3：Visual
M7 仅在 completed_target 或 completed_manual_forward 的认证 handoff 下运行；有胜出修复时产生 P4，否则保持 P3
M8 不产生新像素：Review + Evaluation Lock
M9 优化实现：Exact-equivalent 优先；发生像素变化则重新执行 M8
```

---

# 1. M6–M9 总目标与固定优先级

```text
第一优先级：绝不破坏已封存 P2 的几何、接缝、owner 和 provenance
第二优先级：减少白墙、横梁和物体内部的亮度与颜色跳变
第三优先级：只在真正安全的背景接缝附近做极窄融合
第四优先级：对仍然明显的问题区域进行证据驱动的局部修复
第五优先级：形成可人工复核、可哈希锁定、可复现的 evaluation bundle
第六优先级：在保持输出正确性的前提下减少运行时间和内存
```

任何后级优化失败，都只能发生以下回退：

```text
P3 局部候选失败 → identity photometric / owner-only blend
P3 整体 hard audit 失败 → 不封存 P3，current_latest 保持 P2
P4 局部修复失败 → 保留 P3 对应像素
P4 整体 hard audit 失败 → 不封存 P4，current_latest 保持 P3
M8 复核不通过 → current_reviewed 不写入
M9 优化不等价 → 新建候选证据链，旧 evaluation lock 不得继续使用
```

---

# 2. 阶段、目录、父级和指针的最终定义

## 2.1 像素阶段

| 实施里程碑 | 像素阶段 | 内容 | 父级 |
| --- | --- | --- | --- |
| M4 | P1 | 纵向校正 | P0 |
| M5/M5.1 | P2 | 局部 geometry + seam | P1 |
| M6 | P3 | 光度校正 + 安全融合 | P2 |
| M7 | P4 | 证据驱动的局部修复 | P3 |

M8 和 M9 不应随意创建新的 `P5`、`P6` 像素阶段。

## 2.2 `_STAGE_ORDER` 必须修改

将当前：

```python
_STAGE_ORDER = {"P0": 0, "P1": 1, "P2": 2, "M6": 3}
```

修改为：

```python
_STAGE_ORDER = {
    "P0": 0,
    "P1": 1,
    "P2": 2,
    "P3": 3,
    "P4": 4,
}
```

父级规则修改为：

```python
expected_parent = {
    "P1": "P0",
    "P2": "P1",
    "P3": "P2",
    "P4": "P3",
}
```

严禁创建：

```text
generations/<id>/M6/
generations/<id>/M7/
```

必须创建 P3；P4 只有在自动达标或人工继续模式的认证 M7 handoff 下存在真实胜出修复时才可创建：

```text
generations/<id>/P3/
generations/<id>/P4/
```

## 2.3 completion schema

建议新增：

```text
gemini305-video-s13-p2-completion/v4
gemini305-video-s13-p3-visual-completion/v2
gemini305-video-s13-p4-repair-completion/v1
gemini305-video-s13-evaluation-lock/v1
```

M7 授权与交接使用独立、可哈希验证的 schema：

```text
gemini305-video-s13-m7-manual-forward-authorization/v1
gemini305-video-s13-m7-handoff/v2
```

`v2` handoff 明确区分 `completed_target` 与 `completed_manual_forward`，避免把人工接受误写成自动视觉达标。

`P2 v4` 在 v3 可重放资产上增加 M6.1 evidence/threshold lineage；P2 canonical image、原 provenance 和 M5.1 candidate selection 仍不得变化。旧 P2-v3 只作兼容读取，不能成为现行 M6/M7 的新父级。

## 2.4 指针语义

### `current_base.json`

- 继续只指向 P0。
- 后续阶段不得修改其语义。

### `current_latest.json`

- 表示同一 generation 内，最新完成且通过 hard audit 的阶段。
- 只由 completion、parent SHA 和 hard audit 决定。
- 不表示肉眼最好。
- M6 成功后更新为 P3。
- M7 只有真实修复胜出、P4 seal 与最终 verify 均通过后才更新为 P4；未授权、无 component、全 R0 或无胜者时保持 P3。

### `current_reviewed.json`

- 只能由 M8 的显式人工复核或冻结离线数据集评审写入。
- 可以指向 P2、P3 或 P4。
- 不要求一定指向 `current_latest`。
- 必须包含 reviewer、review method、review time、note 和全部 SHA。

### `current_preview.json`

- 继续保持 deprecated。
- 不得恢复 runtime authority。
- 默认不写。
- 只有兼容性测试显式要求时，才允许写入：
  - `deprecated=true`
  - `runtime_authority=false`

---

# 3. M6–M9 共用的不可违反契约

## 3.1 单向流水线

```text
P0 → P1 → P2 → P3
                    └─ [自动达标或人工继续的认证 handoff + 真实胜者] → P4
```

后级阶段只能：

1. 校验父级。
2. 加载父级封存资产。
3. 追加新阶段。
4. 在 hard audit 通过后更新 `current_latest`。

后级不得：

- 回头重新选择父级。
- 修改已封存父级目录。
- 把旧 stage 复制后伪装为新估计。
- 读取历史 generation 的图像参与当前像素计算。
- 让 quality score 修改 stage seal 规则。

## 3.2 hard audit 与 diagnostic quality 分离

### hard audit 决定

- 资产能否封存。
- pointer 能否前进。
- provenance 是否完整。
- 采样和权重是否合法。
- owner 拓扑是否安全。
- parent chain 是否完整。
- 是否出现孔洞、越界、非有限参数或非法 contributor。

### diagnostic quality 决定

- 现行 M6 在冻结 Q0–Q3 中选哪个，Q4 始终关闭。
- 现行 M6 在 B0 owner-only 与 B1 2 px 中选哪个。
- 只形成 M6 visual evidence；不能直接决定哪些区域进入 M7。
- 人工复核优先看哪些 crop。
- M8 推荐 P2、P3 还是 P4。

M7 的运行授权只来自 SHA 绑定的正式 handoff，不来自 diagnostic quality 本身。当前路线还必须绑定显式用户决定和 `completed_manual_forward` 授权记录；该人工记录只授权受限 M7，不改变 M6 自动画质结论。

所有 diagnostic 文件必须包含：

```json
{
  "diagnostic_only": true,
  "runtime_authority": false
}
```

## 3.3 真实颜色来源

P3/P4 每个有效像素只能是：

1. 一张真实 RGB source 的正式采样，经明确记录的光度变换。
2. 两张真实相邻 RGB source 的明确融合。

严禁：

- Depth 生成颜色。
- TSDF 或点云回传颜色。
- 邻帧复制补洞。
- virtual RGB。
- foreign fill。
- 生成式补图。
- 三张及以上 source 同时混合。

## 3.4 full-resolution 规则

- 候选分析尽量使用低分辨率、patch、shoulder 或抽样像素。
- 不得为每个 photometric 候选渲染完整全景。
- 不得为每个 blend width 渲染完整全景。
- 最终选定全部参数后，再进行正式 full-resolution sampling。
- 每个真实 contributor source 每个正式 stage 最多一次正式解码和一次联合 remap。
- 同一次 remap 结果应同时服务：
  - photometric owner-only。
  - final blended image。
  - provenance。
  - hard audit 必要统计。

## 3.5 deterministic 与可复现

必须固定：

```text
随机种子
train/held-out 划分
候选顺序
候选 tie-break
pair/component 排序
浮点求解器配置
线程数或确定性说明
CUDA/CPU 路径标识
```

如果 GPU 路径无法保证逐像素确定性：

- 必须报告差异。
- exact-equivalence 测试使用确定性 CPU 或固定 CUDA 路径。
- 不能用“看起来差不多”冒充 hash 等价。

## 3.6 production 隔离

M6–M9 继续固定：

```text
diagnostic_only = true
production_eligible = false
production_lock_eligible = false
```

不得：

- 写 `production.lock.json`。
- 写正式 `video_delivery.json`。
- 修改 production renderer。
- 把 evaluation lock 政名为 production lock。
- 把 `current_reviewed` 冒充 production publication。

---

# 4. M6：Forward Visual Pipeline，产生 P3

## 4.0 现行决议：M6.1 即规范 M6

本章早期草案中的通用 PH/BL、1 px、MultiBand 和低频场枚举，凡与本节冲突均由本节取代。现行路径固定使用 v4 candidate、P2 completion v4 和 P3 completion v2；M6 实现就是 `s013_output_first_progressive_dense_central_slit_m61_v2`。光度只允许 Q0–Q3，Q4/低频 field solver 关闭；融合只允许 B0 或总宽度 2 px 的 B1，B2/B3/B4 与 MultiBand 不得进入 registry。旧 v3/M6 仅保留兼容与回归用途。

现有四组真实证据证明 structural hard audit 可通过，但自动视觉结论仍为 blocked。P3 的 seal/pointer authority 与 visual quality 分离；用户已明确接受当前 P3 继续，因此通过新增、独立的 manual-forward 授权记录允许生成受限 M7 handoff，但不得修改或覆盖原 blocked/unresolved/regressed 证据。

# 4.1 M6 的最终职责

M6 只解决：

- source 间整体亮度不一致。
- source 间 RGB 通道比例不一致。
- 大面积白墙和背景的低频亮度条带。
- 已对齐接缝附近的极窄颜色过渡。

M6 不解决：

- 横线几何阶梯。
- 物体错层。
- seam 位置错误。
- geometry 过强或过弱。
- source 选择错误。
- 大视差。
- 对象跨 pair 归属问题。

几何/owner 根因属于 P2；只有认证 handoff 明确列出的孤立、非相邻 protected/geometry/seam/owner residual 才可属于 M7。系统性 photometric、brightness/chroma 或相邻多 seam 仍属于 M6/M6.x；当前人工继续决定只表示暂不处理，不得把它们转入 M7。

---

## 4.2 M6.0：补齐 P2 可重放契约

### 4.2.1 为什么必须先做

当前 P2 已经保存：

- 主 owner。
- 主 source-u/source-v。
- geometry transaction ID。
- seam transaction ID。
- 空的 secondary provenance 字段。

但选中 seam 和 geometry 的完整双侧 sampling map 主要还存在于 M5 运行时对象中，transaction 只绑定其 SHA。  
M6 要从原始 RGB 对两侧 source 做安全融合，不能重新估计 M5，也不能只依赖进程内存。

因此 M6 的第一项工作不是颜色求解，而是让新生成的 P2 成为可独立重放的父级。

### 4.2.2 P2 v4 可重放与 M6.1 evidence 资产

建议新增：

```text
P2/
├─ p2_replay_manifest.json
├─ p2_seams.npz
└─ pair_replay/
   ├─ pair_0000.npz
   ├─ pair_0001.npz
   └─ ...
```

`p2_seams.npz`：

```text
seams_x_by_row: int32[pair_count, H]
base_boundaries_x: int32[pair_count]
```

每个 `pair_XXXX.npz` 至少保存：

```text
pair_index
left_source_index
right_source_index
left_frame_id
right_frame_id
corridor_x0
corridor_x1
seam_x_by_row
left_source_u
left_source_v
left_valid
right_source_u
right_source_v
right_valid
primary_owner_right_mask
geometry_transaction_numeric_id
seam_transaction_numeric_id
```

推荐直接保存最终 2–8 px 最大融合 corridor 内的双侧采样坐标，而不是让 M6 再次推导 alignment 参数。

### 4.2.3 replay 资产约束

- 所有数组来自 M5 已选中的 final geometry + final seam。
- 不允许再进行 feature matching、LK、DIS、RANSAC 或 candidate selection。
- `pair_replay` 的 seam SHA 必须与 pair transaction 相同。
- geometry map SHA 必须与 pair transaction 的 `map_delta_sha256` 或 replay map SHA 一致。
- replay corridor 之外不得提供 secondary sampling。
- replay 文件必须被 P2 completion 的 `assets_sha256` 覆盖。
- P2 completion 绑定：
  - replay manifest SHA。
  - all-pair replay count。
  - pair transaction count。
  - P1 parent completion SHA。
  - P2 result image SHA。
  - P2 provenance SHA。

### 4.2.4 对既有 M5.1 P2 的处理

既有已封存 P2 不得修改。

M6 开发时应：

1. 更新代码，使未来 P2 原生生成 v4 replay 与 M6.1 evidence 资产。
2. 重新运行四组真实输入，创建新 generation。
3. 验证新 P2：
   - `geometry_and_seam_panorama_owner_only.png` 与 M5.1 基线逐哈希一致。
   - `p2_pixel_provenance.npz` 中原有字段逐数组一致。
   - seam model count、fallback、hard audit 结果一致。
4. 只有 replay 资产和 completion schema 允许变化。

如果 P2 像素发生变化，视为 M5.1 回归，不得继续 M6。

---

## 4.3 M6 输入预检

新增函数建议：

```python
load_verified_s13_p2_for_m6(...)
```

必须执行：

1. 验证 P0 completion。
2. 验证 P1 completion。
3. 验证 P2 completion v4。
4. 验证父级链：
   - P2 → P1。
   - P1 → P0。
5. 验证 P2 hard audit 为 true。
6. 验证 P2 result image SHA。
7. 验证 P2 provenance SHA。
8. 验证全部 pair transaction。
9. 验证全部 replay 文件。
10. 验证 replay pair 数量等于 `source_count - 1`。
11. 验证 seam 和 replay map 与 transaction SHA 一致。
12. 验证 `current_latest` 对同一 generation 至少为 P2。
13. 禁止从 `current_reviewed` 或 `current_preview` 读取运行输入。
14. 禁止调用 M4/M5 估计函数。

预检失败时：

- 不创建 P3。
- 不修改 `current_latest`。
- 写 `P3_preflight_failure.json`。
- 保留 P2 可用。

---

## 4.4 M6 新增模块

| 文件 | 职责 |
| --- | --- |
| `video_s13_replay.py` | P2 replay 序列化、加载与 SHA 验证 |
| `video_s13_photometric.py` | 安全样本、全局 gain/bias、低频亮度场 |
| `video_s13_blend.py` | feather、masked MultiBand、权重与事务 |
| `video_s13_p3_hard_audit.py` | P3 独立硬审计 |
| `video_s13_m6.py` | M6 纯算法编排 |
| `video_s13_experiment.py` | P3 staging、seal、pointer 更新 |
| `scripts/summarize_s13_m6_validation.py` | 四组真实验证汇总 |

不要把 P3 hard audit 继续堆进 `video_s13_hard_audit.py`。  
M5 hard audit 与 P3 visual hard audit 应保持职责分离。

---

## 4.5 建议的数据结构

```python
@dataclass(frozen=True)
class S13PhotometricSampleSet:
    pair_index: int
    left_source_index: int
    right_source_index: int
    train_left_rgb_linear: np.ndarray
    train_right_rgb_linear: np.ndarray
    heldout_left_rgb_linear: np.ndarray
    heldout_right_rgb_linear: np.ndarray
    train_weight: np.ndarray
    heldout_weight: np.ndarray
    safe_mask_sha256: str
    protected_mask_sha256: str

@dataclass(frozen=True)
class S13SourcePhotometricParameters:
    source_index: int
    frame_id: int
    model: str
    gain_rgb: tuple[float, float, float]
    bias_rgb: tuple[float, float, float]
    component_id: int
    evidence_weight: float
    fallback_reason: str | None

@dataclass(frozen=True)
class S13PhotometricSolution:
    model_family: str
    source_parameters: tuple[S13SourcePhotometricParameters, ...]
    luminance_field: np.ndarray | None
    train_audit: Mapping[str, object]
    heldout_audit: Mapping[str, object]
    diagnostic_selection: Mapping[str, object]

@dataclass(frozen=True)
class S13BlendTransaction:
    pair_index: int
    transaction_id: int
    parent_p2_pair_transaction_sha256: str
    model: str
    total_width_px: int
    pyramid_levels: int
    protected_pixel_count: int
    blended_pixel_count: int
    hard_audit_passed: bool
    fallback_reason: str | None

@dataclass(frozen=True)
class S13P3Result:
    photometric_owner_only: np.ndarray
    final_image: np.ndarray
    valid_mask: np.ndarray
    pixel_provenance: Mapping[str, np.ndarray]
    photometric_solution: S13PhotometricSolution
    blend_transactions: tuple[S13BlendTransaction, ...]
    hard_audit: Mapping[str, object]
    diagnostic_quality: Mapping[str, object]
    performance: Mapping[str, float]
```

---

## 4.6 线性颜色域

输入通常是 8 位 BGR 图像。M6 应使用固定 LUT 转换到近似线性 RGB：

```python
linear = srgb_to_linear_lut[image_u8]
```

要求：

- 不在 gamma 编码域直接拟合 gain/bias。
- 全部求解和融合使用 float32 或 float64。
- 只在最终输出时转换回输出颜色域并量化为 uint8。
- 不允许多次 uint8 → float → uint8 往返。
- LUT 版本和 SHA 写入 P3 completion。

---

## 4.7 安全光度样本提取

每个相邻 pair 使用 P2 replay corridor 和较宽只读 shoulder 提取光度样本。

### 4.7.1 必须同时满足

- 左右 source 都有效。
- source-u/source-v 合法。
- 两侧没有饱和。
- 两侧不处于极暗噪声区。
- 不在 M5 protected structure。
- 不在明显双边缘区域。
- 不在高 motion residual 区域。
- 不在强反光或颜色剪切区。
- 不在图像边界无效区。
- 不使用 Depth 作为必要条件。

### 4.7.2 建议默认阈值

所有阈值必须进入版本化配置，不得散落在代码中。

```yaml
photometric_sampling:
  minimum_linear_intensity: 0.02
  maximum_linear_intensity: 0.98
  maximum_gradient_percentile: 0.80
  maximum_pair_color_residual_percentile: 0.80
  minimum_pair_sample_count: 512
  maximum_samples_per_pair: 8192
```

这些是样本选择阈值，不是 P3 发布门。

### 4.7.3 protected mask

protected mask 至少融合：

- P2 seam 附近强梯度。
- 长横向结构检测。
- 线缆、细杆、门框、横梁等细线响应。
- M5 horizontal catastrophe guard 的高风险区域。
- 两侧 edge orientation 明显不一致区域。
- 有明显双边缘可能的区域。

protected mask 只用于：

- 禁止 blend。
- 降低光度训练权重。
- 为自动达标或人工继续模式的 handoff 分类提供证据；mask 或 diagnostic quality 本身不得直接提升或创建 M7 repair。

它不得删除 P3 owner 像素。

---

## 4.8 train / held-out 固定划分

不能随机每次变化。

建议使用：

```text
hash(generation_id, pair_index, canvas_x, canvas_y, seed)
```

固定划分：

```text
75% train
25% held-out
```

要求：

- 同一输入、同一 config、同一 seed 的划分逐像素一致。
- held-out 不能参与参数求解。
- held-out 不能参与候选 seam 或 geometry 选择。
- held-out 只用于 diagnostic model selection。
- 划分规则和 seed 写入 solution 和 completion。

---

## 4.9 光度候选层级

> 历史提案（被 §4.0 取代）：本节保留 PH 命名用于解释旧 v3/回归产物。现行 v4/M6.1 只映射 Q0–Q3；PH4/Q4 不进入 solver、registry 或正式 P3。

旧提案模型代码：

```text
PH0 identity
PH1 scalar_luminance_gain
PH2 per_channel_rgb_gain
PH3 bounded_rgb_gain_bias
PH4 constrained_x_luminance_field
```

复杂度优先级：

```text
PH0 < PH1 < PH2 < PH3 < PH4
```

分数接近时选择更简单模型。

### PH0：identity

```text
gain = 1
bias = 0
field = 0
```

始终存在，永远 hard-safe。

### PH1：单一亮度 gain

同一个 gain 同时作用于 R/G/B。

适合：

- 主要是曝光差。
- 不希望改变颜色比例。
- 首轮安全候选。

### PH2：RGB 三通道 gain

每个 source：

```text
gain_r
gain_g
gain_b
bias = 0
```

适合轻微白平衡差异。

### PH3：有界 RGB gain + bias

只有 PH1/PH2 无法解释稳定黑电平差异时才考虑。

建议默认参数边界：

```yaml
photometric_bounds:
  minimum_gain: 0.80
  maximum_gain: 1.25
  maximum_absolute_bias_linear: 0.03
```

超界不是把整条链判失败，而是：

- 当前候选 hard reject。
- 回到更简单模型。
- source 缺证据时回 identity。

### PH4：受限 x 向低频亮度场

本小节仅为历史设计记录，现行 v4 中 `enabled=false`、`solver=disabled`、规范 field 值为 0。若未来需要非零低频场，必须创建新身份的 M6.x，并重新冻结 evidence/threshold，不能在 M7 中启用。旧提案曾考虑：

```text
L(x)
```

而不是自由二维 `L(x,y)`。

规则：

- 只调整 luminance。
- RGB 三通道共同乘同一低频倍率。
- 控制点间距不小于 64 px。
- 零均值 gauge。
- 二阶平滑正则。
- 最大幅度受限。
- 0 field 始终存在。
- 不根据局部物体边缘快速变化。
- 不用于修几何。

建议：

```yaml
luminance_field:
  enabled: true
  mode: x_only
  control_spacing_px: 64
  maximum_absolute_log_gain: 0.08
  second_derivative_regularization: 10.0
  zero_candidate_enabled: true
```

完整二维场与任何 Q4 都不属于 M7。

---

## 4.10 全局求解方法

### 4.10.1 不允许逐帧传递

禁止：

```text
source 0 校正 source 1
source 1 再校正 source 2
source 2 再校正 source 3
```

这会产生累计漂移。

### 4.10.2 建立 photometric graph

- node：真实 contributor source。
- edge：相邻 source 的安全样本约束。
- edge weight：样本量、残差、覆盖、held-out 稳定性。
- 不可靠 pair 不进入求解，但仍保留 owner-only 输出。

### 4.10.3 component 处理

photometric graph 可能不连通。

每个 component：

1. 选择 evidence 最高的 source 作为 gauge。
2. 对所有 source 加 identity 正则。
3. 解 Huber/IRLS 全局问题。
4. evidence 不足的 source 保持 identity。
5. 不允许跨无证据 gap 强行传播大幅 gain。

### 4.10.4 选择规则

候选模型的 diagnostic 比较至少包括：

- train 颜色残差。
- held-out 颜色残差。
- seam 亮度跳变。
- chroma 跳变。
- 全景左右累计亮度坡度。
- clipping 增量。
- 暗部抬灰。
- source 参数复杂度。
- source 参数二阶变化。

候选模型选择不控制 P3 是否封存。  
若全部复杂模型不可靠，选 PH0。

---

## 4.11 安全融合候选

> 历史提案（被 §4.0 取代）：现行 v4/M6.1 只允许 B0 owner-only 与总宽度 2 px 的 B1；下列 BL1 1 px、BL3/BL4 MultiBand 只用于解释旧草案，禁止进入现行 registry。

旧提案模型代码：

```text
BL0 owner_only
BL1 feather_1px
BL2 feather_2px
BL3 masked_multiband_3_to_4px
BL4 masked_multiband_5_to_8px
```

复杂度优先级：

```text
BL0 < BL1 < BL2 < BL3 < BL4
```

### 4.11.1 BL0 owner-only

- 始终存在。
- 不增加 secondary contributor。
- 是所有 pair 的最终 fallback。

### 4.11.2 feather

- 旧提案总宽 1 或 2 px；现行只允许 B1 总宽 2 px。
- 权重由 seam signed distance 生成。
- 不允许固定多列 50/50。
- protected mask 内强制为 0。

### 4.11.3 masked MultiBand

现行路径始终禁用。以下仅为旧提案曾定义的条件，不构成实现或启用授权：

- 左右 replay map 都有效。
- geometry 已由 P2 固定。
- protected mask 为 false。
- 两侧 residual 足够低。
- 无明显遮挡。
- 不与相邻 pair blend corridor 重叠。
- source 颜色已经完成光度校正。

默认：

```yaml
blend:
  minimum_total_width_px: 2
  maximum_total_width_px: 8
  maximum_levels: 2
  adaptive_levels: true
```

自适应层数：

```text
2 px：不用 MultiBand
3–4 px：最多 1 层
5–8 px：最多 2 层
```

### 4.11.4 pair 级选择

旧草案曾令每个 pair 选择 BL0–BL4；现行只能在 B0/B1 之间选择，并必须满足全局 non-overlap。

若相邻 blend corridor 冲突：

1. 优先保留风险更低、证据更强的 pair。
2. 另一 pair 缩窄。
3. 仍冲突则回 BL0。
4. 不允许扩大 owner 或移动 seam。

---

## 4.12 M6 候选 hard gate

photometric 候选 hard gate：

- 参数有限。
- 参数在配置范围内。
- component gauge 完整。
- 不产生 NaN/Inf。
- 输出不会必然越界。
- source 数和 frame_id 与 P2 一致。

blend 候选 hard gate：

- 权重有限。
- 权重位于 `[0,1]`。
- 最多两个 contributor。
- secondary frame 是当前 pair 的另一真实 source。
- secondary source-u/source-v 合法。
- protected mask 内 `secondary_weight=0`。
- corridor 外 `secondary_weight=0`。
- 不改变 primary owner。
- 不改变 seam。
- 不改变 geometry transaction。

失败只淘汰当前候选。

---

## 4.13 P3 正式 full-resolution 渲染

### 4.13.1 处理顺序

```text
验证 P2
加载 P2 replay
确定全局 photometric solution
确定所有 pair blend transaction
按 source 组织 primary owner + 相邻 blend corridor 的联合 ROI
每个 source 解码一次
每个 source 联合 remap 一次
在线性域应用 source photometric
生成 photometric owner-only
生成 final blended P3
写 provenance
执行 P3 hard audit
封存 P3
更新 current_latest=P3
```

### 4.13.2 不允许

- 从 P2 PNG 再做几何 warp。
- 重新跑 M5。
- 重新寻找特征点。
- 重新选择 seam。
- 重新估计 alignment。
- 为每个候选渲染完整图。
- 先生成 uint8 owner-only，再用它做 MultiBand。

### 4.13.3 可以复用

- P2 封存的 primary source-u/source-v。
- P2 replay 双侧 corridor map。
- P1 作为祖先审计信息。
- 原始 RGB。
- 固定 LUT。
- 已选 photometric 参数。
- 已选 blend 权重。

---

## 4.14 P3 provenance

P3 必须保留 P2 原字段：

```text
owner_frame_id
owner_source_index
assignment_index
source_u
source_v
valid
selected_motion_hypothesis_id
placement_method_code
geometry_transaction_id
seam_transaction_id
```

新增或正式写入：

```text
photometric_transaction_id
blend_transaction_id
secondary_frame_id
secondary_source_u
secondary_source_v
secondary_weight
```

要求：

- primary owner 字段与 P2 逐数组一致。
- primary source-u/source-v 与 P2 逐数组一致。
- owner-only 像素：
  - `secondary_frame_id=-1`
  - `secondary_source_u=NaN`
  - `secondary_source_v=NaN`
  - `secondary_weight=0`
- blended 像素最多一个 secondary source。
- `secondary_weight` 是 secondary 的权重。
- primary 权重为 `1-secondary_weight`。
- photometric transaction 必须绑定 source 参数 SHA。
- blend transaction 必须绑定 pair replay SHA 和 P2 pair transaction SHA。

---

## 4.15 P3 hard audit

新增：

```text
src/panorama_demo/video_s13_p3_hard_audit.py
```

至少检查以下内容。

### 父级与不可变性

- P2 completion SHA 正确。
- P2 result SHA 正确。
- P2 provenance SHA 正确。
- P2 pair transaction SHA 正确。
- P2 replay SHA 正确。
- M6 运行前后 P0/P1/P2 completion SHA 不变。
- P2 result image SHA 不变。
- P2 provenance SHA 不变。

### 图像与拓扑

- P3 尺寸等于 P2。
- P3 valid mask 等于 P2。
- P3 primary owner 等于 P2。
- P3 primary source-u/source-v 等于 P2。
- P3 geometry/seam transaction ID 等于 P2。
- 不新增内部孔洞。
- 不新增 invalid owner。
- 不改变 source count。

### 光度

- 所有 gain/bias/field 有限。
- 参数在版本化上限内。
- field gauge 合法。
- 输出像素有限。
- clipping 统计存在。
- identity fallback 可重放。

### 融合

- secondary weight 位于 `[0,1]`。
- 最多两个真实 source。
- secondary source 必须是相邻 pair。
- secondary source 坐标合法。
- protected mask 内无 blend。
- blend corridor 外无 blend。
- blend corridor 不相互重叠。
- owner-only fallback 完整。

### completion

- `hard_audit_passed=true` 才可 seal。
- completion 绑定所有资产 SHA。
- parent stage 必须为 P2。
- `current_latest` 只能在 seal 后更新。
- diagnostic quality 的结果不得参与 pointer 更新。

---

## 4.16 P3 diagnostic quality

建议文件：

```text
P3/diagnostic_quality_report.json
```

至少报告：

### photometric

- 每个 source 的 model、gain、bias。
- 每个 graph component。
- train/held-out sample count。
- train residual。
- held-out residual。
- seam luminance jump before/after。
- seam chroma jump before/after。
- clipping increase。
- dark lift。
- global x slope before/after。
- field amplitude。
- field curvature。

### blend

- 每个 pair 的 candidate list。
- selected model。
- blended pixel count。
- protected pixel count。
- blur width before/after。
- gradient attenuation。
- ghost/double-edge risk。
- owner-only fallback reason。

### 全局

- legacy/diagnostic visual policy result。
- runtime authority 固定为 false。
- 人工 review priority。
- worst pair 列表。

即使 diagnostic policy 为 false，只要 P3 hard audit 通过，P3 仍可封存。  
这种情况下 P3 很可能选择大量 PH0 + BL0，甚至与 P2 像素一致。

---

## 4.17 P3 输出目录

```text
P3/
├─ p2_parent_reference.json
├─ photometric_owner_only.png
├─ photometric_owner_only.jpg
├─ visual_panorama.png
├─ visual_panorama.jpg
├─ p3_valid_mask.png
├─ p3_pixel_provenance.npz
├─ photometric_solution.json
├─ photometric_solution.npz
├─ photometric_training_mask.png
├─ photometric_heldout_mask.png
├─ protected_structure_mask.png
├─ safe_blend_mask.png
├─ blend_weight_map.png
├─ blend_transactions.json
├─ blend_transactions/
│  ├─ pair_0000.json
│  └─ ...
├─ pair_photometric_report.json
├─ hard_audit.json
├─ diagnostic_quality_report.json
├─ performance.json
├─ worst_visual_crops/
│  ├─ manifest.json
│  └─ ...
└─ P3_completion.json
```

结果资产：

```text
result_asset = visual_panorama.png
```

---

## 4.18 M6 performance 字段

```json
{
  "p2_verify_and_load": 0.0,
  "replay_verify": 0.0,
  "photometric_sample_extraction": 0.0,
  "photometric_solve": 0.0,
  "luminance_field_solve": 0.0,
  "blend_candidate_analysis": 0.0,
  "full_resolution_decode": 0.0,
  "full_resolution_remap": 0.0,
  "photometric_owner_compose": 0.0,
  "blend_compose": 0.0,
  "p3_hard_audit": 0.0,
  "artifact_export": 0.0,
  "total_m6": 0.0,
  "m4_reestimated_in_m6": false,
  "m5_reestimated_in_m6": false,
  "geometry_candidate_count_in_m6": 0,
  "seam_candidate_count_in_m6": 0,
  "p3_full_resolution_candidate_render_count": 1,
  "formal_raw_rgb_unique_sources": 0,
  "formal_raw_rgb_remap_invocations": 0
}
```

---

## 4.19 M6 必须新增的测试

### P2 replay

- P2 v4 result image 与 M5.1/P2-v3 canonical image 逐哈希一致。
- 原 P2 provenance 字段逐数组一致。
- replay pair 数量完整。
- replay seam SHA 与 transaction 一致。
- replay map 被 completion 绑定。
- 修改一个 replay 像素后验证失败。
- 缺少任一 pair replay 时 M6 preflight 失败。
- 已封存旧 P2 不被修改。

### photometric

- identity 输入选 PH0 或产生恒等输出。
- 已知曝光差能恢复合理 gain。
- 已知 RGB 通道差能恢复 PH2。
- 不可靠 bias 候选回退 PH2/PH1/PH0。
- graph disconnected 时各 component 可独立求解。
- 无证据 source 保持 identity。
- held-out 不进入求解。
- 参数超界只拒绝当前候选。
- field zero candidate 永远存在。
- field 不改变高频 owner。
- field 不产生全局单向坡度。

### blend

- protected mask 内权重为 0。
- corridor 外权重为 0。
- 一个像素最多两个 contributor。
- 常量红蓝测试不存在多列固定 50/50 平台。
- 1 px、2 px feather 宽度正确。
- 3–4 px 最多一层。
- 5–8 px 最多两层。
- 相邻 corridor 冲突时缩窄或 owner-only。
- 无安全背景时 BL0。

### P3

- primary owner 与 P2 完全相同。
- primary source-u/source-v 与 P2 完全相同。
- geometry/seam transaction 与 P2 完全相同。
- P3 hard audit 失败不更新 pointer。
- diagnostic false 但 hard audit true 时仍封存 P3。
- current_reviewed 不自动创建。
- current_preview 不自动创建。
- P0/P1/P2 SHA 在 M6 前后不变。
- M4/M5 估计函数被调用即测试失败。

---

## 4.20 M6 四组真实验证

固定输入：

```text
fast_direct
fast_ignore_pose
slow_direct
slow_ignore_pose
```

固定数据：

```text
fast：run_20260807_140140
slow：run_20260806_153033
```

规则：

- 继续复用已确认的真实 trajectory cache。
- 不重新运行 ORB-SLAM3。
- 不运行 Open3D。
- 不进行新采集。
- direct/ignore_pose 的输入帧集和轨迹策略必须与 M5.1 摘要一致。
- 因 P2 replay schema 更新，使用新 generation 重新走 P0→P3。
- 新 P2 结果图必须与 M5.1 P2 基线逐哈希一致。

每组至少输出：

- P2 vs photometric owner-only。
- P2 vs P3。
- full panorama contact sheet。
- 三段高风险横梁 crop。
- 亮度变化最大 10 seam。
- 模糊变化最大 10 seam。
- clipping 变化最大 10 seam。
- protected mask。
- safe blend mask。
- blend weight map。
- source gain/bias 曲线。
- x 向亮度场曲线。
- pair 全状态表。
- M6 性能报告。

### M6 硬验收

四组均满足：

- P0、P1、P2、P3 已封存。
- 新 P2 image 与 M5.1 P2 基线 hash 一致。
- P3 hard audit 通过。
- `current_latest=P3`。
- `current_reviewed` 不存在。
- `current_preview` 不存在或 deprecated 且无权限。
- M4/M5 重估次数为 0。
- geometry/seam candidate count 为 0。
- primary provenance 与 P2 一致。
- 每像素最多两个真实 contributor。
- protected structure 内无 blend。
- 无新增孔洞。
- 无 Depth、Open3D、TSDF、mesh、GraphCut。
- 全分辨率正式 candidate render count 为 1。
- 每个 source 正式 remap 不超过一次。

### M6 视觉验收

只记录，不控制 seal：

- 白墙竖向亮度跳变是否下降。
- 物体内部颜色是否更连续。
- 横梁是否没有变模糊。
- 风扇、线缆、门框和软管是否无双边。
- 是否出现新的全局亮度坡度。
- 是否出现偏色。
- 是否出现高光截断。
- 是否出现暗部抬灰。
- P3 是否肉眼优于 P2。

当前四组自动结论仍为 blocked，而不是 M6 自动视觉完成：fast 分支仍有新增 ghost，slow 分支仍 unresolved，wide-safe P95 连续两个合法轮次改善均小于 1%，且冻结 proxy 与 full post-render metric 存在校准口径不一致。该结论不撤销结构合法 P3，也不得被改写；用户已明确接受当前 P3 继续，故新增 `completed_manual_forward` 授权，只允许 M7 处理 handoff 中明确、孤立的 geometry/seam/owner/protected 问题。

---

## 4.21 M6 完成后严格停止

M6 提交结束时不得进入：

- GraphCut。
- object component transaction。
- bounded mesh。
- Depth 风险辅助。
- source rescue。
- P4。
- current_reviewed。
- evaluation lock。
- production freeze。

这里禁止的是 M6 自动越级进入 M7/P4。M6 已严格结束并封存 P3 后，可以通过独立的 `M6_manual_forward_authorization.json` 和 M7 preflight 开启受限 M7；该授权不得触发任何 M6 参数、阈值或像素重算。

---

# 5. M7：Authorized-Handoff Local Repair（有条件地产生 P4）

## 5.1 定位、版本与当前状态

M7 不是 M6 的兜底，也不是“再跑一次 M5”。M7 只能消费现行正式 S013 M6 的授权 handoff，对 handoff 已逐项定位的少量局部残差执行保守 repair。这里的“正式”仅指 S013 diagnostic candidate 内的规范运行路径，不表示 production renderer、production lock 或正式交付获得授权。

现行父级契约固定为：

- candidate / algorithm：`S013_output_first_progressive_dense_central_slit_v4`；
- implementation：`s013_output_first_progressive_dense_central_slit_m61_v2`；
- P2 completion：`gemini305-video-s13-p2-completion/v4`；
- P3 completion：`gemini305-video-s13-p3-visual-completion/v2`；
- M6 实现：原 M6.1 光度接缝抑制方案，现作为唯一规范 M6；
- 旧 v3 链只保留兼容和历史验证用途，不能生成新的 M7 授权。

当前自动验收结论不是 `completed_target`。`artifacts/S013_M6_1_acceptance/blocked_report.json` 作为不可变原始证据继续明确记录：

- `stage_exit=blocked`；
- `m7_handoff_eligible=false`；
- `manual_acceptance_granted=false`；
- 连续两个合法实现轮次在四分支上的 wide-safe P95 收益均小于 1%；
- fast 分支仍有新增 ghost，slow 分支仍 unresolved；
- 冻结 proxy threshold 与 full post-render 尾部分布存在口径不一致。

用户现已明确接受当前 P3 继续后续阶段。不得回写上述 blocked report；应新增 `artifacts/S013_M6_1_acceptance/M6_manual_forward_authorization.json`，至少记录：

- `schema=gemini305-video-s13-m7-manual-forward-authorization/v1`；
- `m6_stage_exit=completed_manual_forward`；
- `accepted_parent_stage=P3` 及其 completion/result/provenance/hard-audit SHA；
- `automatic_visual_status=blocked` 及原 blocked report SHA；
- `automatic_quality_records_preserved=true`；
- `manual_acceptance_granted=true`、授权人、时间、原始用户决定文本或其 SHA；
- `m7_authorization_state=authorized_manual_forward`；
- allowlist 仅为孤立的 `geometry` / `seam` / `owner` / `protected` 局部问题；
- denylist 至少包含系统性 photometric/brightness/chroma/white-balance、Q0–Q4 重求解、阈值放宽、扩大 feather/MultiBand、相邻多 seam 联合修复和 source-set 变化。

因此当前有效状态是：M6 为 `completed_manual_forward`，M6 自动画质仍保持 blocked/unresolved/regressed 原始事实，M7 获得 `authorized_manual_forward` 的受限 Core 权限。该状态不等于 `completed_target`，也不授权任何未通过 component-specific evidence gate 的 Optional。

---

## 5.2 唯一授权入口：SHA 绑定的 M7 handoff

M7 不从 worst crop、零散人工标注、P3 quality report 或单个指标直接建 component。唯一运行输入是通过授权预检并封存的：

`artifacts/S013_M6_1_acceptance/M7_handoff.json`

handoff schema 固定为 `gemini305-video-s13-m7-handoff/v2`。预检必须同时证明：

1. 授权模式二选一：自动路线为 `stage_exit=completed_target` 且 `m7_handoff_eligible=true`；当前人工路线为独立授权记录中的 `m6_stage_exit=completed_manual_forward`、`manual_acceptance_granted=true` 和 `m7_authorization_state=authorized_manual_forward`。单独的 `blocked`、`unresolved`、`regressed`、`unevaluable` 或旧 `completed_manual_near_target` 均不授权 M7。
2. 人工路线必须绑定原始 blocked report，证明自动画质记录未被覆盖；旧 report 中的 `m7_handoff_eligible=false` 保持原值，由新增 manual-forward 授权记录提供本次受限权限。
3. handoff 文件与对应自动/人工授权文件真实存在并通过 canonical SHA 验证。
4. candidate、implementation、effective config、candidate manifest、threshold approval、quality threshold 的 SHA 与父级完全一致。
5. branch、run、generation、P2-v4 completion、P3-v2 completion、P3 result、P3 provenance、P3 hard audit、automatic quality decision 和 manual-forward authorization 均由 handoff 绑定。
6. `current_latest` 仍指向同一 generation 的已验证 P3；P0/P1/P2/P3 的 completion 和资产在预检前后不变。
7. 每个 residual 已分类为 protected / geometry / seam / owner，且只位于 source schedule 上孤立、彼此不相邻的 pair。
8. 任意两个 residual pair 不相邻；每分支数量不超过 `max(3, ceil(0.05 × pair_count))`；允许修改的 safe visible seam 数量不超过 2，且不得将它们合并为一次 repair。
9. handoff 中每项都有 pair、ROI、source/frame、P2/P3 crop、risk 分类、transaction SHA 和证据 SHA，不允许 M7 自行扩大 ROI 或补录新 component。
10. 系统性 photometric/brightness/chroma/white-balance、连续或相邻多 seam、Q 参数/metric-calibration/Q4 需求必须写入 `known_out_of_scope`，不得出现在 `components`。

任一条件失败时只允许写独立的 preflight / not-authorized 报告。不得创建 `P4.pending`，不得 seal P4，不得更新任何运行指针。

---

## 5.3 M6/M6.x 与 M7 的问题边界

以下情况仍属于 M6 或一个显式新身份的后继 M6.x。当前决定是暂不处理并继续后续阶段；它们可以原样记录在 `known_out_of_scope`，但绝不能通过 M7 handoff 转移为 component：

- safe photometric seam 未达到完整 target；
- 系统性亮度或 chroma 跳变；
- residual 跨至少 3 个连续 pair；
- 多个 residual pair 相邻，或同一结构跨 pair 反复出现；
- wide-safe、visible-safe、全局 x slope 或 white balance 仍回归；
- Q0 no-op 但 baseline 仍 actionable；
- 只有放宽 safe/protected、扩大 feather 或引入 blur 才能改善；
- 需要 Q4 / 非零低频亮度场或扩展 photometric model family；
- threshold、metric calibration 或 evidence definition 本身不一致。

M7 只处理授权 handoff 中非相邻、孤立且已证明为 protected / geometry / seam / owner 的 component。运行中如果发现问题实际是系统性 photometric、相邻多 seam 或证据口径错误，必须拒绝该 component，输出 `deferred_m6_out_of_scope`，保持该区域 P3 像素不变；这不妨碍继续评估 handoff 中其他彼此独立的合法 component。

改变 source set、插入中间帧或重新选择 source schedule 不属于 M7。此类需求只能成为上游请求并创建新的 generation，从 P0 重新执行；M7 不实现 `source_rescue` 模块，也不在当前 generation 表达它。

---

## 5.4 不可变输入与冻结参数

M7 必须只读验证过的 P2-v4、P3-v2、授权记录和 handoff，并固定：

- P0 raw-source sampling grid；
- source 顺序、frame identity 和 source count；
- P2 geometry/seam replay 与 transaction；
- M6 的 Q0–Q3 photometric 选择及全部 gain/bias 参数；
- Q4 / low-frequency field 为关闭且规范值为 0；
- safe/protected mask 定义、阈值和 held-out 划分；
- B1 总宽度为 2 px，B0 为 owner-only。

M7 不得重新求解 Q 参数，不得修改 threshold，不得重跑 M4/M5/ORB-SLAM3/Open3D，不得重新定义 safe/protected，也不得从 P3 PNG 反向估计 source map。

---

## 5.5 repair component

component 只能逐项来自授权 handoff；一个 component 只绑定一个孤立 pair，禁止跨 pair object transaction。最小记录为：

- `handoff_item_id`、branch、run、generation；
- authorization mode 与对应 authorization evidence SHA；
- pair index、左右 source/frame identity；
- `bbox_xyxy` 与精确 affected mask；
- protected / geometry / seam / owner 分类；
- P2/P3 parent、result、provenance、transaction 和 evidence SHA；
- 可运行的 Core 候选集合；
- Optional gate 默认值和明确理由。

component 必须有界、可重放且与场景名称、固定 frame_id 和硬编码 crop 无关。affected mask 不得自动扩展到相邻 pair；发生边界证据不足时直接保留 P3。

---

## 5.6 M7 Core：冻结 Q 的安全降级

Core 只有以下候选，复杂度和风险顺序固定为 `R0 < R1 < R2 < R3`：

### R0：keep P3

R0 是比较基线和 component 回滚结果，不是一个 no-op P4。R0 获胜表示该 component 不修改；如果所有 component 都由 R0 获胜，则本次 M7 没有结果阶段。

### R1：B1 2 px → B0 owner-only

- 只关闭 handoff component 内现有的 B1 2 px feather；
- 不存在 1 px、中间宽度或 MultiBand 候选；
- Q 参数、geometry、seam、primary owner 和 component 外像素不变；
- protected 区始终为 B0。

### R2：受限 seam downgrade

- 只在当前 handoff corridor 和冻结模型族内按更简单方向尝试 `S2 → S1 → S0`；
- seam 一旦变化，必须从 immutable P0 grid 在新 final corridor 内重新估计该候选 geometry；
- Q 参数完全冻结，颜色组合使用 B0；
- 不增加最大偏移、不改变 threshold、不沿用旧 seam 的 geometry map。

### R3：受限 geometry downgrade

- 只在 M6 已冻结的 geometry model family 和界限内向更简单、hard-safe 的模型降级；
- 每个 geometry 候选必须重新评估 seam，不能只替换 map；
- Q 参数完全冻结，默认 B0；
- 不串联修改 source grid，不把局部结果传递到下一 pair。

候选只有在 structural hard audit 通过、handoff 指定 held-out/crop 指标有明确改善、无新增 ghost/blur/孔洞/颜色台阶且 affected 区外精确不变时才能胜出。没有胜者就选择 R0，并且不产生 P4。

---

## 5.7 Optional：仅未来明确局部证据的逐级门

当前三个 Optional 全部固定为 `disabled`。未来只有自动达标或 manual-forward 的授权 handoff 为某个具体、孤立 component 提供新增独立证据，并取得对应的显式扩展授权，且配置、implementation identity、测试和 hard audit 已先封存，才能按下列优先级逐项评估；不得因为列在方案中就预先实现或启用。

### O1：constrained two-label GraphCut

- 仅在 R0–R3 无法解决的单 pair component 内竞争两个真实相邻 owner；
- 输出必须是连通、单调、无 island/checkerboard/crossing/collapse 的 owner；
- 不得穿越 protected structure，不得充当 geometry alignment；
- 失败回 R0，不授权后续像素。

### O2：depth-risk veto

- Depth 只可标记深度边界、遮挡侧并 veto blend/warp 候选；
- Depth 不生成颜色、不补洞、不替代 RGB motion/ORB pose、不改变 layout 或 source；
- 深度缺失等价于 veto 不可用，不能使 P3 失败。

### O3：bounded local mesh

- 仅在 Core、GraphCut 和 depth-risk veto 后仍有密集、稳定 RGB 支持的孤立安全区域内评估；
- 位移严格有界，boundary 回 identity，Jacobian 必须为正，protected 位移为 0；
- 不使用自由 dense flow 直接 remap，不跨 component/pair 传播；
- 失败回 R0。

明确不属于 M7 的图像“优化”包括：MultiBand、1 px feather、Q4/低频场、跨 pair object transaction、source rescue、全景或自由 dense flow、锐化、去噪、CLAHE、gamma 调整、模糊和 inpainting。它们不能作为 Optional 候选。

---

## 5.8 选择、P4 创建与指针

M7 的输出规则是“有真实胜者才有 P4”：

1. 在隔离的 evaluation workspace 中验证 handoff 和候选，不预建规范 P4。
2. 每个 component 将 R0 与合法候选按同一 frozen evidence 比较。
3. 至少一个 component 选择了会改变像素/provenance 的候选，且所有选中 transaction 与聚合 hard audit 通过，才创建 `P4.pending`。
4. P4 使用 `gemini305-video-s13-p4-repair-completion/v1` seal，并绑定 P2-v4、P3-v2、handoff、result、provenance、transaction 和 hard-audit SHA。
5. seal 后独立 verify；只有 verify 成功才原子更新 `current_latest=P4`。

以下情况均不生成 P4，也不改 pointer：

- handoff 不存在、未授权或未通过 SHA 预检；
- handoff component count 为 0；
- 所有 component 选择 R0；
- 所有修复候选均未胜出；
- P4 pending、hard audit、seal 或最终 verify 失败。

因此禁止 no-op P4，禁止用“已执行 M7”作为推进指针的理由。M8 在没有 P4 时直接复核 P3。

---

## 5.9 局部渲染、provenance 与 hard audit

选中真实修复时：

- affected ROI 从真实原始 RGB 按 immutable P0/P2 sampling lineage 重放；
- 使用冻结 Q 参数和选中的局部 owner/geometry/seam；
- affected mask 外图像、valid mask 和全部 provenance 与 P3 逐像素/逐数组一致；
- 边界连续回 P3，不新增孔洞或 invalid owner；
- 每像素最多两个真实相邻 contributor；Core 的修复区默认 B0；
- 任何 source、geometry、seam、photometric、blend/owner 决策都绑定 repair transaction。

P4 provenance 在 P3-v2 字段基础上仅为真实修改区增加：

- `m7_handoff_item_id`；
- `repair_component_id`；
- `repair_transaction_id`；
- `repair_model_code`；
- `repair_parent_stage_code`。

未修改区上述字段为 `-1`，并与 P3 原数组全等。hard audit 至少验证父级 SHA、affected/unaffected 分区、source 坐标、owner 拓扑、Jacobian、protected 规则、transaction DAG、孔洞和边界连续性。source count 变化直接 hard fail。

---

## 5.10 产物契约

授权失败或无胜者时，只在 validation/report 根写：

- `M6_manual_forward_authorization.json` 及其验证结果（人工路线）；
- `M7_handoff.json` 及其验证结果；
- `M7_preflight.json`；
- `M7_not_authorized.json` 或 `M7_no_winning_repair.json`；
- `M7_known_out_of_scope.json`，保留系统性 M6 光度遗留问题但不得把它们列为 component；
- 必要的候选比较与失败证据。

这些报告没有 stage/pointer authority，generation 下不得出现 P4。

只有存在真实胜者时，P4 至少包含：

- `p3_parent_reference.json`；
- `m7_handoff_reference.json`；
- `repaired_panorama.png` / `.jpg`；
- `p4_valid_mask.png`；
- `p4_pixel_provenance.npz`；
- `repair_components.json` 与 `repair_transactions.json`；
- 实际启用时才存在的 optional audit；
- `hard_audit.json`、`diagnostic_quality_report.json`、`performance.json`；
- `P4_completion.json`。

不存在的 Optional 不写伪造的成功 audit；未授权状态不创建空 P4 目录。

---

## 5.11 必须新增的测试

### 授权与版本

- 单独的 blocked、manual-near-target、unresolved/regressed、缺 handoff、坏 handoff SHA 均拒绝，且无 P4/pointer 变化。
- `completed_manual_forward` 只有在显式用户授权、原 blocked report SHA、P3 全部 SHA 和 v2 handoff 同时有效时才接受。
- manual-forward 预检不得修改原 blocked/unresolved/regressed 记录，也不得把它们重写成 `completed_target`。
- 只接受 v4 candidate identity、P2 completion v4、P3 completion v2 和同 generation P3 pointer。
- 旧 v3 或 P3-v1 只能兼容读取，不能授权新 M7。
- 相邻 residual、多 pair 系统性 photometric、brightness/chroma/white-balance、Q0–Q4 重求解或 metric-calibration 需求只能进入 `known_out_of_scope`，不能进入 component。
- handoff 中两个相邻 seam、合并多 seam 的 ROI 或 component 自动扩张必须 fail-closed。

### Core

- R0 不创建 no-op P4。
- R1 只执行 B1 2 px 到 B0，冻结 Q/geometry/seam。
- 1 px 和 MultiBand 候选无法进入 registry。
- R2 seam 变化必从 P0 grid 重估 final-corridor geometry。
- R3 geometry 变化必重新评估 seam。
- 无胜者、hard-audit 失败和 final verify 失败均保持 `current_latest=P3`。
- affected 外 image、valid mask、provenance 精确相等。

### Optional

- 默认三个 Optional 全关闭；无 component-specific gate 时不能加载实现。
- GraphCut 强制 two-label、连通、单调、无 island/crossing/collapse。
- Depth 只能 veto，不能生成颜色或改变 source/pose。
- mesh 位移、identity boundary、正 Jacobian 和 protected-zero 逐项 fail-closed。
- 跨 pair object、source rescue、Q4、1 px、MultiBand 和通用图像增强均被 validator 拒绝。

---

## 5.12 基于当前人工授权的实施决策

当前允许实施并运行 M7 preflight 与 Core。授权基础不是 M6 自动视觉达标，而是用户明确接受当前 P3，并要求系统性 safe photometric / metric-calibration 遗留问题暂不处理。原自动 visual acceptance 继续保持 blocked，不能报告为 `completed_target`。

本轮只生成 manual-forward 授权记录和只含真实 M7 局部问题的 v2 handoff，然后按 R0 → R1 → R2 → R3 评估。不得修复整体亮度、色差或白平衡，不得重求 Q0–Q3，不得启用 Q4/低频场，不得放宽 safe/protected、扩大 feather 或使用 MultiBand，不得合并相邻接缝，不得改变 source set 或插入帧。

只在至少一个 Core 修复对明确局部证据有真实改善且全部 hard audit 通过时生成 P4；否则保留 P3，并允许流程继续进入 M8。O1 → O2 → O3 仍全部关闭，未来只有具体 component 出现明确局部证据并取得扩展授权时才逐项考虑。

---

## 5.13 M7 边界

M7 无论未授权、无胜者或成功产生 P4，均不得自动写 `current_reviewed`，不得创建 evaluation lock、production lock 或正式 delivery，不得修改 production renderer。M7 完成后严格停止，由 M8 显式决定复核 P3 或真实存在的 P4。

---
# 6. M8：Review、Transaction DAG 与 Evaluation Lock

# 6.1 M8 不产生新像素

M8 不创建 P5。

它只做：

```text
完整证据收集
全阶段对照
人工或冻结离线数据集复核
current_reviewed 显式写入
evaluation.lock.json
```

M8 可以选择：

- P2。
- P3。
- 真实存在且验证通过的 P4。

不要求选最新阶段。

---

## 6.2 M8 新增模块与脚本

| 文件 | 职责 |
| --- | --- |
| `video_s13_evaluation.py` | 阶段验证、DAG、evaluation bundle |
| `video_s13_review.py` | 显式 review 记录和 pointer 写入 |
| `scripts/build_s13_review_bundle.py` | 生成完整图、crop、表格 |
| `scripts/review_s13_generation.py` | 显式接受/拒绝某一 stage |
| `scripts/build_s13_evaluation_lock.py` | 构造并验证 lock |
| `scripts/verify_s13_evaluation_lock.py` | 独立复验 lock |
| `scripts/summarize_s13_m8_validation.py` | 四组复核汇总 |

---

## 6.3 transaction DAG

必须把以下关系连成 DAG：

```text
P0 completion
  ↓
P1 completion + vertical solution
  ↓
P2 completion
  ├─ pair geometry/seam transaction
  └─ replay asset
  ↓
P3 completion
  ├─ photometric solution
  ├─ blend transaction
  ├─ automatic visual-quality evidence
  └─ M7 authorization evidence
      ├─ completed_target decision；或
      └─ completed_manual_forward decision + 原 blocked report
          ↓
        M7 handoff
          └─ [仅有真实胜者时]
              ↓
            P4 completion
              └─ repair component transaction
```

每个 node 至少有：

```text
node_id
node_type
stage
asset
sha256
parent_node_ids
generation_id
transaction_id
```

DAG 验证：

- 无环。
- 无 dangling parent。
- parent hash 正确。
- 所有 completion 资产存在。
- all-pair / all-component 报告完整。
- 实际存在的 transaction（含前级合法 no-op）也明确记录；禁止为 M7 伪造 no-op P4 transaction。

---

## 6.4 Review Bundle

每个运行必须生成：

```text
evaluation/<evaluation_id>/<run_name>/
├─ full/
│  ├─ P0.png
│  ├─ P1.png
│  ├─ P2.png
│  ├─ P3.png
│  └─ P4.png
├─ contact_sheets/
├─ fixed_crops/
├─ worst_geometry_crops/
├─ worst_visual_crops/
├─ worst_repair_crops/
├─ owner_overlays/
├─ seam_overlays/
├─ protected_masks/
├─ blend_masks/
├─ metric_tables/
├─ m6_automatic_quality.json
├─ M6_manual_forward_authorization.json
├─ m7_handoff.json
├─ M7_known_out_of_scope.json
├─ transaction_dag.json
├─ stage_integrity.json
└─ review_form.json
```

如果某 stage 未运行：

- 明确标记 `not_run`。
- 不用空白图伪装。
- 不把不存在的 P4 当作失败。
- manual-forward 路线即使没有 P4，也必须展示原 M6 自动失败记录、人工接受决定、handoff 和无胜者/跳过 component 证据。

---

## 6.5 人工复核协议

每组真实数据至少检查：

### 全图

- 横向比例和总体布局。
- 是否有明显黑边或孔洞。
- 是否有全局亮度坡度。
- 是否有明显色偏。

### 固定高风险结构

- 横梁。
- 风扇。
- 线缆。
- 门框。
- 软管。
- 箱体边缘。
- 白墙。
- 叶片或细杆。

### worst crops

- 几何结构最差 10。
- 亮度跳变最差 10。
- 模糊增加最差 10。
- repair 变化最大 10。

每个 stage 的人工结论：

```text
accepted
rejected
not_preferred
not_run
```

必须记录理由，不能只写一个总分。

---

## 6.6 `current_reviewed` 写入规则

只能通过显式命令：

```text
review_s13_generation.py
  --generation ...
  --stage P2|P3|P4
  --review-method manual|offline_dataset
  --reviewer ...
  --note ...
```

写入前：

1. 重新验证所选 stage completion。
2. 验证 result asset SHA。
3. 验证 parent chain。
4. 验证 hard audit。
5. 验证 evaluation bundle 中该 stage 存在。
6. 保存 review record SHA。
7. 原子更新 `current_reviewed.json`。

禁止：

- 自动使用 `current_latest`。
- 自动选分数最高 stage。
- 运行结束时自动 review。
- 写入空 reviewer。
- 让 current_preview 代替 reviewed。

---

## 6.7 四组数据的 review 策略

诊断阶段允许四组分别选择 P2/P3，以及真实存在时的 P4，以了解算法行为。

但 evaluation recommendation 必须同时报告：

```text
同一固定 config 下
P3 相对 P2 的通过率
P4 相对 P3 的通过率（无授权 handoff/P4 时为 not_applicable）
不同 trajectory policy 的一致性
fast/slow 的一致性
是否需要逐运行人工选择
```

如果算法只能靠每次人工挑 stage 才可用：

- 可以保留 diagnostic reviewed pointer。
- 不得声称已经形成自动 production policy。

---

## 6.8 Evaluation Lock

建议文件：

```text
evaluation/<evaluation_id>/evaluation.lock.json
```

schema：

```text
gemini305-video-s13-evaluation-lock/v1
```

必须绑定：

### 代码

- repository。
- branch。
- commit SHA。
- dirty state。
- tracked diff hash。
- Python package lock/environment hash。
- CUDA/OpenCV/CuPy/Torch 版本。
- deterministic 配置。

### 配置

- candidate config path。
- config SHA。
- candidate manifest SHA。
- algorithm ID。
- implementation ID。
- feature flags。

### 数据

- fast/slow session path 标识。
- session manifest SHA。
- frames.csv SHA。
- calibration SHA。
- RGB frame list hash。
- trajectory cache SHA。
- trajectory policy。
- ignore_pose 标记。

### 运行

- 四组 generation ID。
- P0/P1/P2/P3 completion SHA，以及真实存在时的 P4 completion SHA。
- 原 M6 automatic quality / blocked report SHA。
- `completed_manual_forward` 授权记录 SHA（当前路线）。
- M7 handoff SHA、M7 preflight SHA，以及 M7 validation/no-winner summary SHA。
- reviewed stage。
- reviewed pointer SHA。
- validation summary SHA。
- transaction DAG SHA。
- test summary SHA。
- command line。
- environment variables。
- start/end time。

### 审计

- hard audit 状态。
- all-pair completeness。
- all-component completeness。
- manual review record。
- known limitations，包括明确 out-of-scope 的系统性 photometric/brightness/chroma/white-balance 与 metric-calibration 遗留问题。
- production eligibility 固定 false。

---

## 6.9 Evaluation Lock 的语义

该 lock 表示：

> 在固定代码、固定配置、固定真实数据和固定 review 证据下，当前 diagnostic candidate 的结果可重放和可审计。

它不表示：

- production 已冻结。
- 20 m SLA 已达成。
- 新数据必然通过。
- 当前图像已经达到 iPhone 级效果。
- 可以修改 production renderer。
- 可以跳过 M9 后重评审。

---

## 6.10 M8 必须新增的测试

- DAG 无环。
- 缺少 parent node 时验证失败。
- 任一 asset SHA 变化时 lock 失效。
- dirty repository 默认拒绝创建正式 evaluation lock。
- reviewer 为空时拒绝。
- review stage hard audit false 时拒绝。
- current_reviewed 可指向 P2/P3/P4。
- current_reviewed 不要求等于 current_latest。
- current_preview 不被读取。
- lock 明确 `production_eligible=false`。
- 四组运行缺一组时 lock 状态为 incomplete。
- trajectory cache SHA 变化时 lock 失效。
- config SHA 变化时 lock 失效。
- M9 代码变化后旧 lock 验证失败。

---

## 6.11 M8 完成定义

- 四组 review bundle 完整。
- transaction DAG 完整。
- 所有 stage integrity 通过。
- 用户或冻结离线规则明确选择 reviewed stage。
- `current_reviewed` 显式写入。
- evaluation lock 创建并通过独立 verifier。
- lock 不是 production lock。
- 没有新像素。
- 没有 current_preview 运行权限。
- 没有 production delivery。

---

# 7. M9：性能优化、等价性验证与候选冻结

# 7.1 M9 的原则

性能不能反向破坏 M5.1–M8 的真实性和审计链。

优化分两类：

```text
A 类：exact-output-preserving
B 类：budgeted / approximate
```

A 类优先。  
B 类只要改变像素、transaction、候选选择或 provenance，就必须视为新候选并重新执行 M6–M8。

---

## 7.2 M9 基线测量

对四组数据记录：

```text
input_and_preflight
trajectory_verify
motion_measurement
progress_and_layout
time_to_P0
P1
P2_geometry
P2_seam
P2_render
P2_export
P3_replay_verify
P3_photometric_sampling
P3_photometric_solve
P3_blend_analysis
P3_full_resolution_render
P3_export
P4_risk_analysis
P4_candidate_analysis
P4_render
P4_export
evaluation_bundle
total_wall_time
peak_rss
peak_gpu_memory
decoded_source_count
remap_invocation_count
```

M9 不用单一总耗时掩盖某一阶段退化。

---

## 7.3 A 类：精确等价优化

优先顺序：

### 1. 复用静态 map

- 缓存 undistortion map。
- 缓存 calibration lookup。
- 缓存 P2 replay 验证结果。
- 缓存 linearization LUT。
- 缓存 source union ROI。

### 2. 解码复用

- 同一 stage 每个 source 解码一次。
- photometric owner-only 和 final blend 共用 float sample。
- 不重复读取同一 JPEG/PNG。
- 对大数据使用有界 LRU。

### 3. 联合 remap

每个 source 的 ROI 是：

```text
primary owner 区
∪ 左侧相邻 blend corridor
∪ 右侧相邻 blend corridor
∪ 当前 repair component ROI
```

合并后一次 remap。

### 4. 稀疏光度样本

- 样本求解不读取完整全景。
- 在 replay shoulder 中按确定性网格抽样。
- 限制每 pair 最大样本数。
- 保持 train/held-out hash 不变。

### 5. 小型线性系统

- photometric graph 只求 source 参数。
- 使用稀疏矩阵。
- 不建立全像素未知量。
- field 控制点稀疏。

### 6. tile streaming

- 按 x tile 处理全景。
- tile 必须带 blend halo。
- halo 宽度由最大 blend width 和金字塔层数确定。
- 拼接 tile 后与非 tile 输出逐像素一致。
- artifact 写入使用流式或 memory map。

### 7. 并行分析

可以并行：

- pair 光度样本。
- pair blend candidate analysis。
- 仅通过自动达标或 manual-forward 授权预检后的 M7 component analysis；当前 manual-forward 路线只运行 handoff 中的 Core component。

不得并行破坏：

- 候选 tie-break。
- transaction 排序。
- 结果写入顺序。
- deterministic hash。

### 8. GPU

允许：

- CuPy / OpenCV CUDA remap。
- LUT。
- 简单逐像素 photometric。
- blend。

要求：

- 路径写入 performance。
- 与基线的等价规则明确。
- exact 模式必须通过逐像素/逐数组验证。
- 不允许 GPU 非确定性导致 selection 变化而继续使用旧 lock。

---

## 7.4 B 类：预算化优化

只有 A 类不足时考虑。

候选：

- photometric 分析宽度降低。
- 只对风险 pair 运行高级 blend。
- 关闭 PH3/PH4。
- 关闭 MultiBand，只保留 feather。
- M7 只运行 Core。
- 跳过无风险 component。
- 降低 artifact 输出级别。
- tile 大小自适应。
- 预算耗尽后停止可选优化。

必须满足：

- P0/P1/P2 输出和 hard audit不受影响。
- P3 至少有 PH0 + BL0。
- P4 只在真实修复胜出时存在；无有效授权/handoff、全 R0 或无胜者必须保持 P3。
- 预算停止不删除已经封存 stage。
- 任何像素变化都创建新的 config/implementation identity。
- 重新运行四组真实数据。
- 重新人工 review。
- 创建新的 evaluation lock。

---

## 7.5 artifact level

建议：

```text
full
audit
minimal
```

### full

- 全部中间图。
- 全部 mask。
- 全部 pair/component crop。
- 用于开发与 M8。

### audit

- 所有 completion。
- hard audit。
- provenance。
- transaction。
- worst crops。
- performance。
- 用于标准验收。

### minimal

- result image。
- completion。
- hard audit。
- 必要 provenance。
- 不用于首次 evaluation lock。

无论 artifact level：

- 不得省略 seal 必需资产。
- 不得省略 parent SHA。
- 不得省略 hard audit。
- 不得省略真实 contributor provenance。

---

## 7.6 时间目标的正确定位

当前阶段只使用现有 fast/slow 真实数据验证。  
20 m ≤ 60 s 不是 M9 的硬发布门，也不能在没有 20 m 数据时宣称达成。

建议将目标写为 diagnostic target：

```text
P3 增量耗时：
  fast：目标 ≤ 8 s
  slow：目标 ≤ 12 s

P4 有授权 handoff 且修复胜出时：
  单次 Core/Optional 增量必须按 component 单独报告

四组全链：
  按阶段报告，不用单一 hard timeout 阻止出图
```

如果当前 M5 本身占用约 31–33 s，M9 应优先：

- 减少 P3，以及真实存在时的 P4 增量。
- 保持 M5.1 结果。
- 为未来长距离 streaming 架构收集 profile。

不得在没有证据时承诺 20 m 已经实时。

---

## 7.7 exact-equivalence 验收

A 类优化必须在固定输入上满足：

```text
P0 result hash 相同
P1 result hash 相同
P2 result hash 相同
P3 result hash 相同
P4 result hash 相同
全部 provenance array 相同
全部 selected model 相同
全部 transaction ID/decision 相同
hard audit 相同
current_latest stage 相同
current_reviewed 指向相同 completion
```

允许不同：

- performance 时间。
- 内存统计。
- 非确定性的系统时间戳。
- 明确排除在 canonical hash 外的运行元数据。

---

## 7.8 M9 与 evaluation lock 的关系

### exact 等价

如果：

- canonical outputs 全部相同。
- code commit 变化。
- 旧 lock 仍因 code SHA 变化失效。

则仍必须：

1. 建立新 commit。
2. 重新运行 verifier。
3. 生成新 evaluation lock。
4. 可以复用人工视觉结论，但必须记录“pixel-equivalent”证据。

### 非 exact

必须完整重新执行：

```text
M6 真实验证
M7 授权预检；`completed_target` 或有效 `completed_manual_forward` handoff 下有真实胜者时才 repair，否则明确保持 P3
M8 review
新 evaluation lock
```

禁止仅修改 lock 中的 commit SHA。

---

## 7.9 M9 新增脚本

```text
scripts/profile_s13_pipeline.py
scripts/compare_s13_exact_outputs.py
scripts/benchmark_s13_m6_m9.py
scripts/verify_s13_pixel_equivalence.py
scripts/summarize_s13_m9_validation.py
```

输出：

```text
benchmarks/S013_M9/<benchmark_id>/
├─ baseline.json
├─ optimized.json
├─ stage_breakdown.csv
├─ memory.csv
├─ equivalence.json
├─ four_run_summary.json
└─ benchmark_completion.json
```

---

## 7.10 M9 必须新增的测试

- tile 与非 tile 逐像素一致。
- cache 开关不改变输出。
- 单线程与固定并行策略输出一致。
- CPU 与 exact CUDA 路径满足既定等价规则。
- source 解码次数正确。
- remap 次数正确。
- photometric/ blend 不发生 full-resolution candidate 枚举。
- budget 终止时 P3 仍可 identity/owner-only。
- artifact level 不删除 seal 必需资产。
- 非 exact 配置不能复用旧 evaluation lock。
- code SHA 变化使旧 lock 失效。
- exact output comparison 覆盖 image、NPZ、JSON canonical content。

---

## 7.11 M9 完成定义

- 四组 baseline profile 完整。
- A 类优化全部经过 exact-equivalence。
- P3 增量耗时明确；存在 P4 时再报告其增量。
- peak memory 明确。
- 所有 full-resolution decode/remap count 明确。
- 新 evaluation lock 已建立。
- candidate 仍为 diagnostic。
- 没有 production lock。
- 没有 20 m 虚假验收结论。

---

# 8. 更新后的配置草案

建议在现有 config 中追加或修改：

```yaml
components:
  s013_output_first_progressive_dense_central_slit:
    forward_pipeline:
      stage_order: [P0, P1, P2, P3, P4]
      default_stop_after: P3
      allow_resume_from_sealed_stage: true
      forbid_parent_reselection: true
      p4_requires_certified_handoff_and_actual_winner: true
      current_preview_runtime_authority: false

    p2_replay:
      enabled: true
      completion_schema: gemini305-video-s13-p2-completion/v4
      write_seams: true
      write_pair_sampling_corridors: true
      maximum_secondary_corridor_width_px: 8
      require_transaction_hash_match: true

    photometric:
      enabled: true
      color_domain: linear_srgb
      model_candidates:
        - Q0_identity
        - Q1_scalar_luminance_gain
        - Q2_per_channel_rgb_gain
        - Q3_bounded_rgb_gain_bias
      safe_background_only: true
      train_heldout_split: true
      train_fraction: 0.75
      deterministic_split_seed: 20260804
      minimum_pair_sample_count: 512
      maximum_samples_per_pair: 8192
      minimum_gain: 0.80
      maximum_gain: 1.25
      maximum_absolute_bias_linear: 0.03
      disconnected_component_policy: identity_regularized_component
      failure_policy: identity

    luminance_field:
      enabled: false
      solver: disabled
      canonical_field_value: 0
      full_2d_enabled: false

    blend:
      enabled: true
      model_candidates:
        - B0_owner_only
        - B1_narrow_feather_2px
      safe_background_only: true
      b1_total_width_px: 2
      ineligible_models: [B2_safe_masked_multiband, B3, B4]
      maximum_color_contributors_per_pixel: 2
      protected_structure_weight: 0
      failure_policy: owner_only

    repair:
      enabled: true
      authorization_schema: gemini305-video-s13-m7-handoff/v2
      allowed_stage_exits: [completed_target, completed_manual_forward]
      completed_target_requires_m7_handoff_eligible: true
      manual_forward_authorization_schema: gemini305-video-s13-m7-manual-forward-authorization/v1
      manual_forward_requires_explicit_user_authorization: true
      manual_forward_preserve_automatic_quality_records: true
      manual_forward_core_only: true
      allowed_component_classes: [protected, geometry, seam, owner]
      systemic_photometric_component_forbidden: true
      adjacent_or_merged_seam_component_forbidden: true
      core_candidates:
        - keep_p3
        - b1_2px_to_b0_owner_only
        - downgrade_seam_reestimate_geometry
        - downgrade_geometry_reselect_seam
      q_parameters_frozen: true
      optional_priority: [constrained_two_label_graphcut, depth_risk_veto, bounded_local_mesh]
      graphcut_enabled: false
      depth_risk_enabled: false
      bounded_mesh_enabled: false
      no_handoff_or_no_winner_policy: no_p4_keep_p3

    review:
      automatic_review_forbidden: true
      current_reviewed_requires_explicit_command: true
      allowed_review_methods: [manual, offline_dataset]
      current_preview_runtime_authority: false

    evaluation:
      lock_schema: gemini305-video-s13-evaluation-lock/v1
      require_clean_worktree: true
      require_four_real_runs: true
      production_lock: false

    performance:
      exact_equivalence_first: true
      tile_streaming_enabled: false
      artifact_level: full
      budget_timeout_is_quality_degradation_only: true
      preserve_sealed_stage_on_timeout: true
```

config validator 必须拒绝：

- stage 使用 `M6`/`M7` 目录名。
- P3 parent 不是 P2。
- P4 parent 不是 P3。
- `current_preview_runtime_authority=true`。
- photometric 没有 Q0 identity。
- blend 没有 B0 owner-only。
- repair 没有 keep_p3，或未冻结 Q。
- blend contributor > 2。
- full 2D field 默认开启。
- 既没有 `completed_target` 授权，也没有显式用户接受、原自动质量记录和全部 P3 SHA 绑定的 `completed_manual_forward` v2 handoff 时启用 Core。
- manual-forward 模式把系统性 photometric/brightness/chroma/white-balance、Q 重求解、相邻/合并多 seam 或 source/frame 变化写入 component。
- manual-forward 模式在缺少新的 component-specific 扩展授权时启用任何 Optional。
- GraphCut/depth-risk/mesh 未经 component-specific evidence gate 直接开启。
- 注册 1 px、MultiBand、Q4、跨 pair object transaction 或 source rescue 为 M7 候选。
- production eligibility 或 production lock。
- M6 重估 geometry/seam。
- M7 修改 source set；此类需求必须返回上游并创建新 generation。

---

# 9. CLI 与编排建议

## 9.1 用 `--stop-after` 代替随意跳阶段

建议：

```text
--stop-after P2
--stop-after P3
--stop-after P4  # 仅自动达标或 manual-forward 的授权 M7 handoff 已通过预检时
```

规则：

- 不能 `--stop-after P4` 却跳过 P3。
- 当前 blocked 本身不再是拒绝理由；缺少有效 manual-forward 授权或 handoff 时必须拒绝 `--stop-after P4`，不能创建空阶段。
- 已存在 sealed parent 时可以 resume。
- resume 只能追加，不得重新估计 parent。
- parent 不可重放时明确失败。

## 9.2 resume

建议：

```text
--resume-generation <generation_path>
```

必须：

- 验证 generation。
- 查找 current latest sealed stage。
- 只执行其后阶段。
- 不重新运行前级。
- P2→P3 resume 要求原生 P2-v4 replay/evidence 完整。
- P3→P4 resume 还要求自动达标或 manual-forward 授权 handoff、P3-v2 provenance/transaction 完整且至少有一个可评估的孤立局部 component。

## 9.3 Review 命令独立

实验运行命令不得自动 review。

Review 使用独立脚本：

```text
python scripts/review_s13_generation.py ...
```

## 9.4 性能命令独立

```text
python scripts/profile_s13_pipeline.py ...
python scripts/verify_s13_pixel_equivalence.py ...
```

---

# 10. 代码文件修改清单

## 必须修改

```text
src/panorama_demo/video_s13_bundle.py
src/panorama_demo/video_s13_experiment.py
src/panorama_demo/video_s13_contract.py
src/panorama_demo/video_s13_m5.py
configs/video_candidates/s013/S013_output_first_progressive_dense_central_slit_v4.yaml
configs/video_candidates/s013/candidate_manifest.json
```

`video_s13_m5.py` 只允许增加 replay 序列化接口，不允许改变 M5.1 算法选择。

## 当前 manual-forward 授权后可新增的 M7 Core

```text
src/panorama_demo/video_s13_repair.py
src/panorama_demo/video_s13_p4_hard_audit.py
src/panorama_demo/video_s13_m7.py
```

当前可新增上述 M7 preflight/Core 文件，但实现必须先验证 `M6_manual_forward_authorization.json` 和 v2 handoff，只注册 R0–R3。不得在这些文件中重求 M6 光度参数、修改阈值或夹带 Optional。

## 未来 component-specific evidence gate 后才可新增

```text
src/panorama_demo/video_s13_graphcut.py
src/panorama_demo/video_s13_depth_risk.py
src/panorama_demo/video_s13_mesh.py
```

跨 pair object transaction 不再属于 M7 文件清单。

## 脚本

```text
scripts/summarize_s13_m6_validation.py
scripts/summarize_s13_m7_validation.py
scripts/build_s13_review_bundle.py
scripts/review_s13_generation.py
scripts/build_s13_evaluation_lock.py
scripts/verify_s13_evaluation_lock.py
scripts/profile_s13_pipeline.py
scripts/verify_s13_pixel_equivalence.py
scripts/summarize_s13_m9_validation.py
```

---

# 11. 建议的提交与停止顺序

## 已完成/并入规范路径：M6.0 P2 replay contract

完成：

- 原生 P2 v4（兼容读取 P2-v3）。
- replay serialization。
- replay verifier。
- exact P2 regression。
- 单元测试。

真实验证：

- 现有四组 prepared P2-v4 已证明 P2 image/provenance 与 M5.1 基线一致；这些是历史 acceptance preparation 证据。
- v4 同 generation 原生 P2 已通过单元/合成全链验证；四组真实数据仍需按新的规范 full-chain identity 重跑，不能把 detached preparation 冒充本轮现场验收。

严格停止，不做 photometric。

## 已完成/并入规范路径：M6.1 photometric owner-only

完成：

- safe samples。
- linear color。
- Q0–Q3。
- Q4/x-field solver 关闭。
- photometric owner-only。
- P3 provenance 基础。
- 测试。

严格停止，不做 blend。

## 已完成/并入规范路径：M6.1 B0/B1 + P3-v2 seal

完成：

- B0 owner-only / B1 2 px；B2–B4 禁止。
- P3 hard audit。
- P3 completion。
- current_latest=P3。
- v4 同 generation 的合成 normal/resume/fail-closed 验证，以及 standalone 真实分支 smoke。
- 既有四分支 acceptance 用于保留 `blocked` 视觉结论；新的四分支规范 full-chain 现场重跑仍待执行。

严格停止 M6。

## Commit 4：M7 manual-forward authorization + preflight/Core（当前执行）

当前按 `completed_manual_forward` 排期：

- 生成并验证独立 manual-forward 授权记录，原 blocked/unresolved/regressed 证据保持不变。
- 生成只含孤立 geometry/seam/owner/protected 问题的 v2 handoff，系统性光度遗留只写 `known_out_of_scope`。
- 只加载 handoff component。
- R0–R3（冻结 Q）。
- 明确禁止 no-op P4。
- P4 hard audit。
- handoff 与胜者 gate report。

当前允许实现并运行 Core；有真实胜者才更新 `current_latest=P4`，否则保持 `current_latest=P3`。Core 完成后严格停止并进入 M8，本轮不自动进入 Optional。

## Commit 5：M7 Optional（当前跳过；仅未来明确局部证据与扩展授权逐项触发）

只按 O1 constrained two-label GraphCut → O2 depth-risk veto → O3 bounded mesh 的优先级实现 handoff 明确要求的功能；当前三项均 disabled。

严格停止 M7。

## Commit 6：M8 Review + Evaluation Lock

完成：

- review bundle。
- DAG。
- current_reviewed。
- evaluation lock。
- verifier。

严格停止 M8。

## Commit 7：M9 exact performance

完成：

- profile。
- cache。
- decode/remap 联合。
- tile streaming。
- exact equivalence。
- 新 evaluation lock。

严格停止 M9。

---

# 12. 回归与测试命令要求

每个 commit 至少执行：

```text
定向 pytest
受影响子集 pytest
完整 pytest
ruff
compileall
git diff --check
```

M6–M9 完成时要求：

- 完整测试不得少于当前基线覆盖。
- skipped 必须说明原因。
- 真实验证脚本可复现。
- validation summary 不从旧摘要复制。
- 所有真实命令、输入路径、trajectory policy 写入 summary。
- 工作区未跟踪计划文件和大体积 artifact 不得误提交。

---

# 13. 四组真实数据的统一产物

建议根目录：

```text
artifacts/S013_M6_M9/
├─ fast_direct/
├─ fast_ignore_pose/
├─ slow_direct/
├─ slow_ignore_pose/
├─ M6_validation_summary.json
├─ M6_manual_forward_authorization.json # 绑定原自动质量记录与 sealed P3
├─ M7_handoff.json                 # v2，只含真正属于 M7 的局部 component
├─ M7_validation_summary.json      # 有胜者或无胜者均记录；not_authorized 仅用于预检失败
├─ M8_evaluation/
├─ M9_benchmarks/
└─ final_summary.json
```

每组必须记录：

```text
generation_id
P0/P1/P2/P3 completion SHA；P4 仅真实存在时记录
current_latest
current_reviewed
source count
pair count
M7 authorization state
repair component count（未授权时为 not_applicable，不伪造 0 个 P4）
M6 automatic quality status 与原始报告 SHA
manual-forward authorization state 与授权 SHA
known out-of-scope M6 photometric issues
selected photometric model histogram
selected blend model histogram
selected repair model histogram
fallback histogram
hard rejection histogram
time breakdown
remap count
peak memory
manual review state
```

---

# 14. M6–M9 最终完成定义

## M6 完成

- v4 candidate 从原生 P2-v4 进入 P3-v2。
- P2 可独立重放并绑定 M6.1 evidence/threshold lineage。
- P2 result 与 M5.1 基线一致。
- P3 hard-safe 封存。
- 光度限制为 Q0–Q3，融合限制为 B0/B1 2 px；Q4/B2+ 被拒绝。
- M6 不重估 M5。
- 四组 `current_latest=P3`。
- `not_reviewed`。
- 当前 automatic visual acceptance 仍为 blocked，不得报告 `completed_target`；原 unresolved/regressed 记录保持不变。
- 用户明确接受当前 sealed P3，新增独立授权后 M6 推进状态为 `completed_manual_forward`，可生成受限 v2 M7 handoff。

## M7 完成

当前完成定义是执行受限 manual-forward M7 并严格停止：

- manual-forward 授权、原 blocked report 与 sealed P3 的全部 SHA 已验证。
- handoff component 只包含孤立 geometry/seam/owner/protected 问题；系统性 M6 光度遗留仅列为 out-of-scope。
- Core 可局部回滚，Q0–Q3、阈值、source set 和 component 外像素冻结；全部 Optional 保持关闭。
- 至少一个真实修复胜出时才封存 P4 并更新 `current_latest=P4`。
- 全 R0/无胜者不产生 P4，`current_latest` 保持 P3，但流程继续进入 M8。
- `not_reviewed`。

## M8 完成

- 全阶段证据完整。
- transaction DAG 完整。
- 人工或冻结离线复核完成。
- `current_reviewed` 显式写入。
- evaluation lock 可独立验证。
- 无新像素。
- 无 production lock。

## M9 完成

- 性能瓶颈按阶段明确。
- exact 优化保持所有 canonical output。
- 非 exact 优化重新评审。
- 新 evaluation lock 完整。
- diagnostic candidate 冻结。
- 20 m 性能不虚假宣称。
- production 仍在方案外。

---

# 15. 绝对禁止项

1. 不得修改已封存 P0/P1/P2/P3/P4。
2. 不得让 M6 重新运行 M4/M5。
3. 不得让 M6 重新选择 seam 或 geometry。
4. 不得只在内存中保存 M6 所需的 P2 map。
5. 不得从 P2 PNG 再做几何 warp。
6. 不得在 gamma 域直接进行主要光度求解。
7. 不得使用全图 feather。
8. 不得使用全图 MultiBand。
9. 不得在 protected structure 上融合。
10. 不得出现三 source 混色。
11. 不得用 blur 单独换取评分。
12. 不得让 quality score 阻止 hard-safe identity P3。
13. 不得在 M7 静默改变 source set。
14. 不得把 GraphCut 当作几何对齐。
15. 不得逐 pair 串联修改 source grid。
16. 不得使用自由 dense flow 直接作为最终 mesh。
17. 不得使用 Depth 生成颜色。
18. 不得固定 frame_id、crop 或场景名称。
19. 不得自动写 current_reviewed。
20. 不得恢复 current_preview 运行权限。
21. 不得把 evaluation lock 当 production lock。
22. 不得在 M9 输出变化后复用旧 lock。
23. 不得以运行超时删除已封存 stage。
24. 不得在没有 20 m 实测时声称达到 20 m ≤ 60 s。
25. 不得修改 production renderer 或 production lock。
26. 不得让 M7 修复整体亮度、色差或白平衡，也不得重新求解 M6 的 Q0–Q3。
27. 不得让 M7 启用 Q4 或非零低频亮度场。
28. 不得让 M7 放宽 safe/protected 阈值、扩大 feather 或使用 MultiBand。
29. 不得把多个相邻接缝合并为一个 M7 component 或一次联合修复。
30. 不得让 M7 改变 source set、插入新帧或扩大 handoff ROI。
31. 不得修改原 M6 blocked/unresolved/regressed 自动记录来伪造 `completed_target`。
32. 系统性 M6 光度遗留只能作为 out-of-scope known limitation 记录，不得进入 M7 component。

---

# 16. 给 Codex 的最终执行指令

```text
以 S013_output_first_progressive_dense_central_slit_v4 /
s013_output_first_progressive_dense_central_slit_m61_v2 为规范 M6。
旧 v3 只作兼容与回归，不得授权新的 M7。

M6 必须从同一 generation 的原生 sealed P2-v4 进入 P3-v2；不得重估 M4/M5。
P3 通过独立 hard audit 后 seal 并原子更新 current_latest=P3；
diagnostic quality 不得拥有 pointer 权限；current_reviewed 不得自动创建。

保留当前 blocked_report 及 unresolved/regressed 自动记录，不得改写为 completed_target。
依据用户明确接受 sealed P3 的决定，新增 SHA 绑定的 completed_manual_forward 授权和 v2 M7 handoff；
handoff 只列孤立的 geometry/seam/owner/protected component，系统性 photometric/brightness/chroma/
white-balance、相邻多 seam、Q4 或 metric-calibration 问题只记为 out-of-scope known limitation。

通过 manual-forward 预检后，按 R0→R1(B1 2px→B0)→R2/R3(seam/geometry downgrade) 评估 Core；
冻结 Q0–Q3、safe/protected 阈值和 source set，禁止 Q4、扩大 feather、MultiBand、跨 pair 联合修复和插帧。
有真实胜者才生成 P4；无胜者保持 P3 并继续 M8。本轮 GraphCut、Depth-risk 与 mesh 全部关闭，
未来只在具体 component 有明确局部证据并取得扩展授权时逐项考虑。

M8 不产生新像素，不创建 P5。
M8 生成完整 review bundle、transaction DAG、显式 current_reviewed 和 evaluation.lock.json；
当前直接复核 P3，只有真实存在的 P4 才加入比较。
current_preview 不得恢复运行权限。evaluation lock 不是 production lock。

M9 优先 exact-output-preserving 优化。
任何改变 canonical image、provenance、transaction 或 selection 的优化都视为新候选，
必须重新执行四组真实验证、M8 review 和 evaluation lock。

每个 milestone 单独提交、单独测试、单独真实验证并严格停止。
不要提前进入下一个 milestone。
```

---

# 17. 通俗易懂的完整说明

## 17.1 M6 在做什么

现在的 P2 已经把图片的主要几何位置和接缝路线确定了。  
M6 不再碰这些位置，只负责把不同照片的亮度和颜色调得更接近，再在非常安全的接缝附近做一点点过渡。

可以把 P2 想成一张已经拼好、结构基本正确，但不同纸片颜色稍有差异的长画卷。

M6 做两件事：

1. 给每张纸片做轻微曝光和颜色校正。
2. 只在没有横梁、线缆、物体边缘的安全区域，选择硬 owner 或总宽度 2 px 的极窄过渡。

如果某个位置不安全，就完全不融合，继续使用 P2 的硬切。  
因此 M6 最坏的结果只是和 P2 一样，不应该把 P2 变坏。

## 17.2 为什么先增加 P2 replay

现在程序知道最终 seam 和 geometry 是什么，但部分详细采样 map 主要保存在 M5 运行时内存里。

这就像施工完成后，只保存了房子的照片和验收报告，却没有保存后续装修需要的管线图。

M6 需要同时读取接缝左右两张原图，所以必须先把这份“管线图”正式保存到 P2，并用 SHA 封存。  
这一步不改变 P2 的图片，只增加后续可以准确重放的数据。

## 17.3 M6 为什么必须有 identity 和 owner-only

颜色算法可能算错，融合也可能让横梁变糊。

因此每个位置都保留最简单方案：

```text
颜色不改
不做融合
```

复杂方案只有确实更合适时才使用。  
即使全部复杂方案失败，P3 仍可以安全生成，而且像素可以与 P2 相同。

## 17.4 M7 在做什么

M7 不是全图再优化一次，也不是看到 P3 最差位置就能启动。当前 M6 自动画质仍是 blocked，但用户已经人工接受 sealed P3 继续；因此要先把这一决定写成 `completed_manual_forward` 授权，再生成并用 SHA 封存只含真正局部问题的 handoff。

得到 manual-forward 授权 handoff 后，只对其中列出的少数、互不相邻位置先做最保守的修复：

1. 把 2 px 融合关闭为硬 owner。
2. 把弯曲 seam 换成简单 seam。
3. 把复杂 geometry 换成简单 geometry。
4. 每次 seam 变化都重新计算对应局部 geometry。

这些办法都不改 M6 的颜色参数。系统性亮度、色差、白平衡和 metric-calibration 问题继续保留为本轮范围外的已知问题，不能混入 component。只有将来某个具体局部区域有新增证据并取得扩展授权时，才按顺序考虑：

- 受限的两标签 GraphCut。
- 只用于否决危险候选的 Depth 风险提示。
- 位移有界、边界回零的小型局部网格。

跨多个接缝的联合对象修复、Q4、MultiBand、1 px 融合和插入新帧都不属于 M7。需要改变 source 时只能返回上游，新建 generation 并从 P0 重新生成。没有 handoff 或没有任何修复胜出时，不生成一张假装完成的 P4，指针仍留在 P3。

## 17.5 M8 在做什么

M8 不再改图片。

它把 P0、P1、P2、P3，以及真实存在时的 P4 放在一起，生成完整图和最差区域对照，让人明确看出每一步到底变好了还是变坏了。当前没有 P4，就直接复核 P3。

人工可以选择：

- P2 最好。
- P3 最好。
- 真实存在时的 P4 最好。

选择结果写入 `current_reviewed`。  
这和 `current_latest` 不一样：

```text
current_latest = 程序最新安全完成到哪一步
current_reviewed = 人工真正认为哪一步最好
```

M8 最后生成 evaluation lock，把代码、配置、数据、轨迹、图片、报告和人工选择全部用 hash 绑定。  
它只是“这次实验可复现”的锁，不是正式产品锁。

## 17.6 M9 在做什么

M9 才开始集中加速。

先做不改变任何像素的优化，例如：

- 相机标定 map 缓存。
- 每张原图只读取一次。
- 多个区域合并成一次 remap。
- 光度分析只抽样少量像素。
- 分块处理大图。
- 复用 GPU。

这些优化后，图片和 provenance 必须与优化前完全一致。

如果为了更快而关闭某些算法、降低分辨率或减少候选，结果可能变化。  
这种变化不能继续使用旧的人工结论和旧 evaluation lock，必须作为新的候选重新验证。

## 17.7 整个 M6–M9 最终流程

```text
P2：结构和接缝已经确定
  ↓
M6 / P3：Q0–Q3 调亮度/颜色，安全背景只用 B0 或 B1 2 px
  ↓
M6 授权状态？
  ├─ 其他：跳过 M7，保持 P3 → M8
  ├─ completed_target：签发自动达标 handoff → M7
  └─ completed_manual_forward：保留自动失败记录，签发受限人工 handoff → M7（当前路径）
M7：只修 handoff 中的孤立 geometry/seam/owner/protected 局部
  ├─ 有真实胜者：产生 P4 → M8
  └─ 无胜者：保留 P3 → M8
M8：人工对照并选出真正最好的一版，生成 evaluation lock
  ↓
M9：在保持结果正确的前提下加速，再重新锁定
```

核心思想是：

> 后面的每一步都只能在前一步已经安全的基础上增加效果；一旦不确定，就回到更简单的方案，而不是为了让指标好看把图像模糊掉、重新改几何或覆盖前面的封存结果。
