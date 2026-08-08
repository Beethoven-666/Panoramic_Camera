# Panoramic_Camera PNS-Flex — Output-First Pairwise Vertical Warp Plan

## 0. 文档定位

本文是独立连续 RGB-D 视频全景的新候选实施方案，替代 v6/v6.1 当前“视觉门先行、算法尚未执行就退回 hard cut”的研发路线。

本文只适用于：

- `g305-video-experiment` 的 candidate；
- 后续通过 development / validation / first-holdout 后才可能进入的视频 production freeze；
- Gemini 305 连续 RGB-D 视频；
- 单向水平侧扫、场景基本静止的工况。

本文不修改照片 `g305-panorama`，也不授权修改当前 `production.lock.json`。

实施分支：

```text
codex/pns-flex-output-first
```

当前建议基点：

```text
1cacbd17e89b960c8f13bb732f87804aaeffc56a
```

当前工作区已有未提交 tracked 修改。执行者必须先确认归属，不能自行 stash、reset、覆盖或混入新分支。

---

## 1. 用户目标与总原则

用户优先级：

1. 只要输入、真实位姿、owner 和发布结构仍然成立，就应生成全景图；
2. 在能够出图的前提下，尽量选择视觉更好的处理；
3. 局部算法失败不能让整对条带在其它算法尚未尝试前就退回 hard cut；
4. hard owner 是安全基线和最终兜底，不是每对条带的默认唯一结果；
5. 横线、货架、梁、桌沿应得到专门的纵向连续性处理；
6. 重复纹理要保留多个匹配候选，再用单调、平滑和左右一致性筛选；
7. 最终得到随高度变化的 `Delta y(y)`，必要时升级为低分辨率二维 mesh；
8. 局部几何修正后，再尝试 GraphCut 和安全 MultiBand；
9. “肉眼看不出”是相对感知目标，不能被一个固定绝对门限代替；
10. 真实 pose、owner、正 Jacobian、遮挡保护等结构安全门继续 fail-closed。

核心原则：

> **结构门决定候选是否合法；感知评分决定合法候选中哪个更好。局部算法失败只淘汰该候选，不直接阻止其它策略，也不阻止结构完整的全景发布。**

---

## 2. v6/v6.1 当前问题

### 2.1 门禁位置错误

当前行为接近：

```text
alignment / FB / edge / topology 某一项未达到固定阈值
→ 不运行后续局部几何
→ 不运行 GraphCut
→ 不运行 MultiBand
→ 直接 hard cut
```

问题：

- 后续候选没有真实比较机会；
- 某个局部最大值会否决整对条带；
- 重复纹理或局部难区把整个 overlap 判死；
- `0 double-edge / 0 ghost` 可能只是 GraphCut 根本没执行；
- GraphCut 失败后扩大旧 owner，形成几十到上百像素的大块交接；
- 固定门限跨速度、曝光、纹理和物距没有统一视觉含义。

### 2.2 新方案不取消安全门

重新分为：

```text
结构硬门：候选不合法，必须淘汰
感知软门：候选可运行，按相对质量排序
灾难否决：候选产生复制、空洞、fold-over、保护区破坏，必须淘汰
```

因此仍然禁止：

- 负 Jacobian；
- owner 无来源；
- 前景/遮挡保护区被不可信 warp 或 blend 修改；
- 非有限 grid；
- 非相邻 source 竞争。

但不再因为某个 edge P95 略高于固定值，就禁止 GraphCut 或其它合法候选运行。

---

## 3. “必须出图”的边界

### 3.1 可以降级但必须出图

满足以下条件时，候选程序必须生成结构完整的 2-D 全景：

- 严格视频会话有效；
- 存在覆盖扫描起止的连续单向 direct ORB pose source 链；
- 所有 render source 都是真实落盘 RGB-D 帧；
- 每条最终相邻 source 边有真实 Open3D 审计；
- 一次标定 RGB sampling、valid mask 和 owner map 可完整建立；
- 资源上限和原子发布结构有效。

以下情况不能阻止全景生成，只能影响局部策略和等级：

- 某一对没有足够纵向匹配；
- `Delta y(y)` 拟合失败；
- 低分辨率 mesh 被拒绝；
- GraphCut 异常或拓扑不合格；
- 没有安全 MultiBand 区域；
- 某一对仍有肉眼可见残差；
- 可选增强算法超过局部预算。

这些 pair 回到该 pair 当前最好的合法候选，最差仍有 midpoint hard owner。

### 3.2 仍然必须 F 的情况

不得用“先出图”绕过：

- 会话、标定、aligned depth 或输入结构无效；
- 最终 source 缺真实 direct ORB pose；
- pose 非有限、非刚体、断链、逆向或不连续；
- 无法构造覆盖扫描范围的真实 source 链；
- 最终相邻 Open3D 边缺失或使用伪 backend；
- owner / valid mask 拓扑不完整；
- 颜色无真实来源或发生补洞；
- 资源超限；
- 原子发布失败。

### 3.3 分级

保持 A/B/C/F：

- A：结构完整，视觉严格通过，安全 hard owner/anchor；
- B：结构完整，使用完整审计的局部 warp、mesh、GraphCut 或安全融合；
- C：结构完整并成功出图，但部分 pair 只能 degraded hard owner，或视觉仍需复核；
- F：仅结构失败。

算法增强失败不得把本可发布 C 的全景变成 F。

---

## 4. 总体架构

```text
真实连续 RGB-D 扫描
        ↓
尽可能密的 direct ORB source 链
        ↓
所有最终相邻 source 的 CUDA Open3D 审计
        ↓
PNSF-0：midpoint hard-owner 完整基线
        ↓
每个相邻 pair 建立候选池
        ↓
纵向多候选匹配
        ↓
单调对应路径 + 稳健 Delta y(y)
        ↓
必要时低分辨率 2-D mesh
        ↓
在 identity / curve / mesh 上分别构造 seam 候选
        ↓
GraphCut 候选
        ↓
安全背景 0 / 2 / 4 px 融合候选
        ↓
结构过滤 + held-out 感知评分 + 复杂度排序
        ↓
每对选择最好的合法候选
        ↓
一次最终 RGB sampling + 单一 owner provenance
        ↓
结构完整则发布；残差决定 A/B/C，不决定有没有图
```

---

## 5. Source 与基础条带

### 5.1 优先全部真实 direct ORB 帧

第一选择：

```text
tracking_source_mode = all_real_scan_frames
render_source_mode = all_direct_orb_frames
```

对 `run_20260807_140140`：

- 扫描段 frame 31–128；
- 98 个已落盘真实 RGB-D 帧；
- 首个目标 98/98 direct ORB pose；
- 成功时全部 98 帧成为 render source；
- 不允许 ORB 后再按 8/12/16 FPS 抽稀。

### 5.2 灵活但不伪造 pose

如果 all-frame ORB 不完整，继续寻找：

1. 30 FPS 全 direct 链；
2. 更密、覆盖扫描起止的 direct-pose 连通子链；
3. 16/12/8 FPS direct 链。

必须满足：

- 全部 source 仍是真实 direct ORB pose；
- 不插值、不外推、不用 Open3D/DIS 替代 pose；
- 每次降密记录 source gap、原因和视觉风险；
- source 变稀时降级或人工复核，不宣称等同全帧方案。

选择“最密的结构合法链”，不能选择“第一个成功的最低 FPS 链”。

### 5.3 PNSF-0 永远先建立

任一局部算法前，先建立完整基线：

- midpoint 决定 hard owner；
- owner 不因对象或 GraphCut 失败而扩大；
- 首尾只向扫描外侧扩展；
- 有效像素恰有一个真实 owner；
- valid mask 不按颜色删除黑像素；
- 关闭 flow、warp、GraphCut、blend 和 photometric；
- 保存 owner 和采样 grid hash。

这是每个 pair 的最终安全回退和“必须能出图”的基础。

---

## 6. 每对条带使用候选池

### 6.1 候选集合

每个相邻真实 source pair 至少考虑：

```text
B0  identity grid + midpoint hard owner
B1  Delta y(y) grid + midpoint hard owner
B2  low-res mesh grid + midpoint hard owner
B3  identity grid + GraphCut hard owner
B4  Delta y(y) grid + GraphCut hard owner
B5  low-res mesh grid + GraphCut hard owner
B6  B3/B4/B5 + 2 px safe MultiBand
B7  B3/B4/B5 + 4 px safe MultiBand
```

候选可缺失：

- 无可靠纵向曲线时不生成 B1/B4；
- mesh 不合法时不生成 B2/B5；
- 无共同有效 overlap 时不生成 GraphCut；
- 无安全低梯度背景时不生成融合。

但某候选缺失或失败不能阻止其它候选。

### 6.2 不再设置单一 pre-seam 视觉否决

以下仍测量，但进入连续评分：

- forward/backward P50/P95/max；
- edge residual P50/P95/max；
- 横线相位残差；
- Lab 低频阶跃；
- double-edge / ghost；
- 重复纹理歧义；
- 安全背景覆盖率。

禁止用其中一个绝对最大值直接阻止整个 pair 的 GraphCut。

### 6.3 候选级结构硬否决

以下候选必须淘汰：

- 非有限 grid；
- 位移超过结构硬界；
- Jacobian 非正；
- 新产生无来源像素或空洞；
- 修改前景、遮挡、孔洞或深度保护区；
- owner 不单调、回跳或非相邻竞争；
- GraphCut 标签不能覆盖共同有效区；
- blend mask 与保护域相交；
- 输出与声明的 owner/provenance 不一致。

---

## 7. 多候选纵向匹配

核心处理链：

```text
左条带右侧窄带                 右条带左侧窄带
从上到下提取每行/每段特征  ──→  搜索最相似的纵向位置
         ↓                                  ↓
得到多个可靠匹配：(y_left, y_right, confidence)
         ↓
单调、平滑、左右一致的连续对应路径
         ↓
稳健拟合 Delta y(y)
         ↓
仅作用于 overlap 的局部纵向 warp
         ↓
重新检查横线残差，再 GraphCut / MultiBand
```

### 7.1 Corridor

- 输出 owner 宽度仍由 midpoint 决定；
- 匹配 corridor 建议 96–160 px；
- 左侧取左 source 的右 overlap；
- 右侧取右 source 的左 overlap；
- 匹配上下文宽于最终 seam/blend；
- 只读 pair 所需局部 RGB；风险触发才读 aligned depth。

必须把三类 mask 分开，不能再用一个大 `protected` 同时关闭所有算法：

```text
evidence_mask
    允许作为匹配证据；背景长横线可以进入。

warp_safe_mask
    允许修改局部 inverse grid；排除遮挡、不同深度层和不确定前景。

blend_safe_mask
    比 warp_safe 更严格；只保留共同有效、低梯度、低残差安全背景。
```

同一 pair 中，某个高度不能 warp，不代表其它高度不能 warp；不能 blend，也不代表不能 GraphCut。

### 7.2 垂直分段

将 corridor 从上到下分成重叠 segment：

```yaml
segment_height_px: 16
segment_stride_px: 8
descriptor_window_width_px: 32
minimum_valid_fraction: 0.55
```

每段组合特征：

- Sobel-y / Scharr-y 水平边缘强度；
- 梯度方向直方图；
- 局部 Lab 均值、方差和截尾分位；
- 归一化 patch correlation；
- Census/符号梯度；
- 长水平 ridge 的位置、宽度和方向；
- valid mask 与 safe mask 覆盖率。

不能只比较单行亮度。

### 7.3 保留多个候选

每个左 segment 在右 corridor 的有限搜索范围保留 Top-K：

```yaml
maximum_vertical_search_px: 8
top_k_candidates: 5
candidate_nms_distance_px: 1
```

每个候选记录：

```text
y_left
y_right
delta_y
descriptor_cost
gradient_cost
orientation_cost
color_cost
valid_fraction
protection_fraction
ambiguity
confidence
```

重复纹理不能被简单 ratio-test 全丢。多个相似层同时保留，由全高单调路径消歧。

### 7.4 左右一致性

执行：

```text
left segment → right candidate → search back to left
```

建议：

```yaml
mutual_consistency_target_px: 0.75
mutual_consistency_hard_px: 1.5
```

未达到 target 的候选可降权；超过 hard 且无法解释的候选不能独立支撑 warp。

### 7.5 单调、不交叉路径

候选图节点：

```text
node(i, j) = 第 i 个左 segment 的第 j 个右候选
```

路径必须：

- `y_left` 递增；
- 被选 `y_right` 也递增；
- 不交叉；
- `delta_y` 一阶平滑；
- `delta_y` 二阶受罚；
- 允许 unmatched；
- 遮挡或保护区转移被禁止或高惩罚。

可使用动态规划式最短路径，但必须明确：

> **DP 只选择纵向特征对应，不用于最终 RGB owner seam；最终 seam 仍是 midpoint hard owner 或真实 GraphCut。**

建议路径代价：

```text
data_cost
+ lambda_slope * |delta_y_i - delta_y_(i-1)|
+ lambda_curve * |delta_y_i - 2*delta_y_(i-1) + delta_y_(i-2)|
+ lambda_skip * unmatched_count
+ lambda_ambig * ambiguity
+ lambda_guard * protection_risk
```

### 7.6 RANSAC / 稳健曲线

有序对应为：

```text
(y_i, delta_y_i, confidence_i)
```

RANSAC 第一职责是从重复纹理候选中产生少量粗路径假设，不是直接把最终模型限制成 translation。建议最多保留 3 个满足 `|Delta y| <= 8 px`、纵向覆盖充分的粗线性假设，再由单调路径和 held-out 证据决定使用哪一组对应。

可使用：

1. 加权 RANSAC + 分段线性；
2. Huber loss 低阶 B-spline；
3. 二阶平滑正则的稀疏 knot 曲线。

第一版建议：

```yaml
curve_model: robust_piecewise_linear
knot_spacing_px: 48
maximum_vertical_displacement_px: 8
maximum_vertical_slope: 0.12
minimum_inlier_segments: 6
```

不能只输出单一 translation/rotation。多条横线支持不同位移时，曲线要允许各高度独立约束。

### 7.7 Train / held-out 分离

- 按高度交错划分 train / held-out segment；
- train 用于路径和曲线；
- held-out 只用于重投影、横线相位和方向连续性评分；
- segment 少时 leave-one-band-out；
- 无 held-out 时可作为低置信候选，但不能获得高等级视觉声明。

---

## 8. 从 Delta y(y) 生成局部 warp

### 8.1 第一阶段只做纵向

```text
x' = x
y' = y + alpha(x, y) * Delta y(y)
```

其中：

- `alpha=1` 在 seam 邻域；
- 向 corridor 外边界平滑到 0；
- 在前景、遮挡、孔洞、深度边缘和保护区附近到 0；
- 不改 pose、source centre 和 owner 身份。

### 8.2 单次 RGB sampling

标定 inverse map 与局部 map 组合：

```text
final_inverse_grid = compose(calibration_inverse_grid, local_vertical_grid)
```

原始 RGB 只做一次最终采样。禁止先标定 remap 再对已插值图二次 warp。

### 8.3 Jacobian

全分辨率 grid 检查：

```text
J = 1 + alpha*d(Delta y)/dy + Delta y*d(alpha)/dy > 0
```

不能只在 knot 上检查。

### 8.4 保护区

以下不参与匹配或不接受位移：

- depth discontinuity；
- occlusion / disocclusion；
- depth hole；
- 强反光/透明疑似区；
- 细杆、电缆和孤立前景；
- 无纹理且对应不确定区域；
- object owner lock。

这些像素继续单一 hard owner。

---

## 9. 低分辨率二维 mesh

### 9.1 升级条件

只有 `Delta y(y)` 结构安全，但 held-out 残差仍呈明显 x 相关变化时，才构造 mesh。不是所有 pair 都运行。

### 9.2 约束

```yaml
mesh_cell_width_px: 32
mesh_cell_height_px: 32
maximum_vertical_displacement_px: 8
maximum_horizontal_displacement_px: 2
boundary_displacement_zero: true
positive_jacobian_required: true
protected_intersection_pixels: 0
```

第一版以纵向自由度为主，水平自由度只处理微小局部残差。

### 9.3 保留所有基线

必须同时保留：

- identity；
- `Delta y(y)`；
- mesh。

mesh 只有 held-out 改善且无结构退化时才选。mesh 拒绝后继续 curve 或 identity，不得全局失败。

如果完整幅度 mesh 不合法，可以在不重新拟合、不读取 validation 标签的前提下，按固定顺序审计缩放幅度：

```text
1.00 → 0.75 → 0.50 → 0.25
```

每个幅度都重新检查 Jacobian、边界、保护域和 held-out。全部失败才回到 `Delta y(y)`，不能直接把整对条带判死。

---

## 10. Warp 后复检

每个 identity / curve / mesh grid 在未参与拟合区域测量：

- 横线中心位置残差；
- 水平边缘方向差；
- 梯度峰重复数量；
- double-edge；
- ghost；
- Lab 低频阶跃；
- 新空洞数量；
- valid 支持变化；
- protected mask 修改数量；
- Jacobian min / P01；
- 位移 P50/P95/max；
- 清晰度变化。

评价采用：

```text
candidate vs identity baseline
candidate vs 邻近同-owner自然变化
candidate vs 测量重复性噪声
```

不要求视觉指标都等于 0。

---

## 11. GraphCut 的新位置

### 11.1 GraphCut 是候选，不是奖励

只要：

- 两个相邻真实 source 有共同有效 overlap；
- owner 竞争区非空；
- 保护域可建立；
- 输入 grid 有限；

就构造 GraphCut 候选。

不能因某视觉 P95 超过固定值，就在 GraphCut 前判死。

### 11.2 分别尝试

```text
identity + GraphCut
Delta y(y) + GraphCut
mesh + GraphCut
```

curve/mesh 不通过时，identity GraphCut 仍有真实比较机会。

### 11.3 结构审计

GraphCut 必须：

- 只在相邻真实 source 竞争；
- 每个共同有效像素一个 owner；
- 标签按时间单调；
- 不穿越 owner-locked 前景；
- 不产生岛；
- 不覆盖 invalid；
- 不改 pose/source；
- 失败恢复该 grid 的 midpoint hard owner。

禁止：

```text
GraphCut reject
→ 整 corridor choose_new=false
→ 扩大旧 owner
```

正确行为：

```text
GraphCut reject
→ 该候选无效
→ 继续其它候选
→ 最差回原 midpoint hard owner
```

### 11.4 Seam 平滑不再使用整对绝对 row-step 总开关

`maximum_row_step_px=1` 不能继续作为整个 480 px 高 pair 的共同入口门。新的处理是：

- 一阶和二阶 seam 平滑进入 GraphCut 能量；
- 按安全连通分量审计，不让一个局部尖峰否决整对；
- 对孤立尖峰可做有界、单调的 seam 投影修复；
- 修复仍不满足 owner 拓扑时，只回退该分量；
- owner 回跳、非相邻竞争和岛仍是结构硬失败。

任何放宽必须通过 development 人审校准，并保持最终 owner 单调；不能把 DP 用作最终 seam。

---

## 12. MultiBand 的新位置

### 12.1 安全带宽候选

对合法 GraphCut seam 构造：

```text
blend width = 0
blend width = 2 px
blend width = 4 px
```

总带宽不超过：

```text
min(8 px, floor(0.20 * narrower_owner_width))
```

### 12.2 只在安全背景

必须：

- 两侧共同有效；
- 低梯度；
- warp 后残差低；
- 非前景/遮挡/深度边缘/孔洞/反光；
- 与所有保护 mask 零交集。

owner map 仍保留 dominant hard owner。

### 12.3 不隐藏几何错误

若 blend 后新增：

- 双边；
- ghost；
- 清晰度下降；
- 横线变粗；
- 保护域污染；

只淘汰融合候选，无融合 GraphCut 仍保留。

---

## 13. 候选评分与选择

### 13.1 顺序

```text
构造所有可行候选
→ 淘汰结构非法候选
→ 淘汰灾难视觉退化候选
→ 计算 held-out 相对感知分数
→ 分数接近时选更简单、更快、位移更小者
→ 无增强候选胜出时选 B0
```

### 13.2 连续分数

```text
score =
    w_line        * normalized_line_residual
  + w_orientation * normalized_orientation_break
  + w_lab         * normalized_low_frequency_lab_step
  + w_double      * normalized_double_edge
  + w_ghost       * normalized_ghost
  + w_hole        * new_hole_penalty
  + w_blur        * sharpness_loss
  + w_complexity  * model_complexity
  + w_runtime     * local_runtime_cost
```

视觉项用局部自然变化、MAD、分位或测量噪声归一化。

### 13.3 灾难否决

直接淘汰：

- 内容复制；
- 新空洞；
- fold-over；
- owner 无来源；
- 保护区修改；
- 明显新增双轮廓；
- 单线变双线；
- crop/valid/owner 不一致。

### 13.4 简单候选优先

差异小于测量噪声时：

```text
identity hard owner
优先于 curve
优先于 mesh
优先于 GraphCut
优先于 blend
```

这是避免无收益复杂度，不是算法前置门。

---

## 14. Pair 状态机与区域级策略

### 14.1 Pair 状态机

```text
BASELINE_READY
→ MATCH_CANDIDATES_BUILT
→ MONOTONE_PATH_BUILT / MATCH_INSUFFICIENT
→ CURVE_CANDIDATE_BUILT / CURVE_SKIPPED
→ MESH_CANDIDATE_BUILT / MESH_SKIPPED
→ GRAPHCUT_CANDIDATES_EVALUATED
→ BLEND_CANDIDATES_EVALUATED
→ BEST_LEGAL_CANDIDATE_SELECTED
→ PAIR_COMMITTED
```

任一增强异常：

- 记录 pair-local failure；
- 继续下一候选或上一合法候选；
- 不终止整幅全景；
- 不改变其它 pair。

### 14.2 同一 pair 可混合区域策略

例如：

```text
前景20% hard owner
横线安全区35% Delta y(y) + GraphCut
平坦背景30% Delta y(y) + GraphCut + 2px MultiBand
重复纹理歧义区15% hard owner
```

建议新增：

```text
pair_state = mixed_region_strategy
```

并记录各策略像素数，不能只写 `graphcut_accepted` 或 `hard_owner_degraded`。

### 14.3 三类 hard owner

- `hard_owner_protected`：前景、遮挡、深度边缘等正常保护，不自动降级；
- `hard_owner_selected`：合法候选中 baseline 最好或改善小于噪声；
- `hard_owner_fallback_exhausted`：所有增强耗尽且接缝仍可能可见，发布 C 并人工复核。

---

## 15. 时间预算与实时性

### 15.1 Baseline 始终可用

PNSF-0 source plan、owner 和 grid 先完整建立。增强超时时仍可发布 PNSF-0 或部分增强结果，并标 C / manual review。

### 15.2 所有 pair 先获得廉价机会

1. 所有 pair 运行廉价多候选纵向匹配；
2. 所有可行 pair 运行曲线；
3. 所有有 overlap 的 pair 构造 GraphCut；
4. mesh 只给曲线残差明显 pair；
5. MultiBand 只给安全背景；
6. 更昂贵局部 flow/mesh 按风险排序。

不能让前几个 pair 耗完预算，导致后面 pair 连 GraphCut 都未尝试。

### 15.3 预算审计

逐 pair：

```text
matching_ms
curve_fit_ms
mesh_ms
graphcut_ms
blend_ms
evaluation_ms
budget_skipped_components
```

要求：

- 3 m benchmark 继续满足仓库选择门；
- 正式 `maximum_post_seconds=60` 不变；
- 超增强预算时发布最好结构合法结果；
- audit/review 不计主 SLA；
- 不通过抽稀 source 伪造性能。

### 15.4 快速采集链仍要单独优化

`run_20260807_140140` 标称 60 FPS，但扫描段实际只保存约 45 FPS。renderer 改进不能替代真实采样密度，因而本分支仍要保留一个独立性能提交：

- 先用 `--no-preview` 做实机 A/B；
- 将同步 `online_scan.add(...)` 移出 capture hot path；
- preview 限频到约 10–15 Hz；
- 分析 worker 积压时标记 online state 不可用，正式 RGB-D 继续采集；
- 不删除逐帧同步回读、align、writer 和 manifest 契约；
- 连续 3 次固定曝光实机目标为有效保存 FPS `>=58`、sensor frame gap rate `<=1%`、queue/write errors 为 0。

采集改动与 renderer 候选分开提交和验收；没有实机时必须标注未现场验证。

---

## 16. 候选阶段

### PNSF-0 — Dense Direct Midpoint

```text
全部或最密 direct ORB source
+ midpoint hard owner
+ 无增强
```

### PNSF-1 — Monotone Vertical Curve

```text
PNSF-0
+ Top-K 重复纹理候选
+ 左右一致性
+ 单调对应路径
+ robust Delta y(y)
+ overlap-only vertical warp
```

### PNSF-2 — Selective Low-Resolution Mesh

```text
PNSF-1
+ 仅对曲线残差仍显著的 pair 构造低分辨率 2-D mesh
```

### PNSF-3 — Flexible Seam Candidate Pool

```text
PNSF-2
+ identity / curve / mesh GraphCut
+ 0 / 2 / 4 px 安全 MultiBand
+ held-out 相对感知选择
```

### PNSF-4 — Temporal Photometric Continuity

```text
PNSF-3
+ 共同可见安全背景 gain/bias
+ source 时间链低频正则
+ held-out 失败回单位校正
```

光度校正实际执行在 matching/seam 前，但作为独立 ablation 开启。

### PNSF-5 — Object Single-Owner Continuity

```text
PNSF-4
+ depth connected object protection
+ compact object 单真实 source owner
+ 最大一次 handoff
```

对象/遮挡 mask 从 PNSF-1 起作为保护证据；PNSF-5 才主动优化对象 owner。

---

## 17. 配置文件

新增：

```text
configs/video_candidates/PNSF0_dense_direct_midpoint.yaml
configs/video_candidates/PNSF1_monotone_vertical_curve.yaml
configs/video_candidates/PNSF2_selective_lowres_mesh.yaml
configs/video_candidates/PNSF3_flexible_pair_strategy.yaml
configs/video_candidates/PNSF4_temporal_photometric.yaml
configs/video_candidates/PNSF5_object_continuity.yaml
configs/video_candidates/pns_flex_visual_calibration_v1.json
```

每个 YAML：

- `config_schema: gemini305-video-candidate/v1`；
- `role: candidate`；
- `allow_baseline_fallback: false`；
- 明确 parent；
- 明确 changed / evidence / output components；
- 更新 `candidate_manifest.json` 和规范化 SHA；
- 不改 production lock。

核心配置建议：

```yaml
pns_flex:
  output_first: true
  pair_failure_policy: keep_best_legal_candidate
  tracking_source_mode: all_real_scan_frames
  render_source_mode: all_direct_orb_frames
  owner_baseline: midpoint_hard_owner

  vertical_matching:
    corridor_width_px: 128
    segment_height_px: 16
    segment_stride_px: 8
    descriptor_window_width_px: 32
    maximum_vertical_search_px: 8
    top_k_candidates: 5
    mutual_consistency_target_px: 0.75
    mutual_consistency_hard_px: 1.5
    minimum_inlier_segments: 6

  vertical_curve:
    model: robust_piecewise_linear
    knot_spacing_px: 48
    maximum_displacement_px: 8
    maximum_slope: 0.12
    full_resolution_positive_jacobian: true
    overlap_boundary_zero: true

  mesh:
    enabled: false
    selective_only: true
    cell_width_px: 32
    cell_height_px: 32
    maximum_dx_px: 2
    maximum_dy_px: 8
    positive_jacobian: true

  graphcut:
    enabled: false
    attempt_on_identity: true
    attempt_on_curve: true
    attempt_on_mesh: true
    rejection_policy: discard_candidate_not_expand_owner

  multiband:
    enabled: false
    candidate_total_widths_px: [0, 2, 4]
    maximum_levels: 3
    safe_background_only: true

  candidate_selection:
    relative_metrics: true
    held_out_required_for_grade_ab: true
    simpler_on_tie: true
    catastrophic_veto: true
```

后续候选只打开对应组件，不改前一阶段其它参数。

---

## 18. 代码结构

### 18.1 新增模块

```text
src/panorama_demo/video_pns_flex_config.py
src/panorama_demo/video_pns_flex_source_plan.py
src/panorama_demo/video_pns_vertical_features.py
src/panorama_demo/video_pns_monotone_correspondence.py
src/panorama_demo/video_pns_vertical_curve.py
src/panorama_demo/video_pns_lowres_mesh.py
src/panorama_demo/video_pns_pair_candidates.py
src/panorama_demo/video_pns_candidate_scoring.py
src/panorama_demo/video_pns_flex_renderer.py
src/panorama_demo/video_pns_review_package.py
```

职责：

- config：阶段累积关系和结构硬界；
- source plan：全部/最密 direct ORB 链；
- vertical features：segment 特征和 Top-K；
- monotone correspondence：左右一致、单调不交叉路径；
- vertical curve：RANSAC/Huber `Delta y(y)` 与 inverse grid；
- lowres mesh：选择性二维 mesh；
- pair candidates：identity/curve/mesh + seam/blend 候选池；
- scoring：结构过滤、灾难否决、held-out 相对评分；
- renderer：pair 隔离、最佳候选提交、一次最终 sampling；
- review package：发布后只读盲评包。

### 18.2 修改模块

```text
src/panorama_demo/video_pipeline.py
src/panorama_demo/video_panorama.py
src/panorama_demo/video_experiment.py
src/panorama_demo/video_motion_resampler.py
src/panorama_demo/video_algorithm_contract.py
src/panorama_demo/video_algorithm_selection.py
src/panorama_demo/video_visual_metrics.py
configs/video_candidates/candidate_manifest.json
```

### 18.3 可复用原语

可复用但不能继承旧决策逻辑：

```text
video_visual_renderer.video_dis_pair_evidence
video_graphcut_seam.solve_video_graphcut_seam
video_hard_guards
video_near_blend
video_photometric
video_object_mask
video_object_owner_lock
video_final_sampling
video_v2_route 的 CUDA once-sampling / owner / provenance
现有 ORB staging、trajectory cache、Open3D prepared-frame 审计
```

### 18.4 禁止复用的策略

- v6 GraphCut reject 后扩大旧 owner；
- v6 四 source rescue/reroute；
- v6 192 px retry；
- v6 全量背景/近景多级 alignment；
- v6 当前 MultiBand 同 mask 平均；
- v6.1 pre-seam 固定绝对门阻止 GraphCut；
- v6.1 整高 row-step/island 共同入口；
- D1/D3 插值 prior 进入 owner；
- D3 full-canvas RAFT；
- C2–C13 累积复杂链作为隐式依赖。

---

## 19. 报告与 provenance

每个 pair 记录：

```text
left_frame_id / right_frame_id
baseline_owner_interval
candidate_ids_attempted
candidate_ids_structurally_rejected
candidate_ids_visually_rejected
selected_candidate_id
selected_grid_type
selected_seam_type
selected_blend_width
top_k_match_count_by_segment
mutual_consistent_match_count
monotone_path_match_count
unmatched_segment_count
curve_model / knots / inliers
mesh_shape / applied_pixels
held_out_before / after
line residual before / after
double-edge before / after
ghost before / after
Lab step before / after
Jacobian min / P01
protected_changed_pixels
new_hole_pixels
runtime_by_component
fallback_reason
```

全景级记录：

```text
all_scan_frame_count
direct_orb_pose_count
render_source_count
pose provenance counts
source spacing statistics
owner width statistics
Open3D edge count/backend/status
pair strategy histogram
hard-owner fallback pair count
curve selected pair count
mesh selected pair count
GraphCut selected pair count
MultiBand selected pair count
budget-skipped counts
single RGB sampling audit
owner monotonicity audit
primary post-capture seconds
```

`video_pixel_provenance.npz` 继续保存真实 dominant owner。blend 不得删除 owner 权威。

---

## 20. 测试

### 20.1 新增

```text
tests/test_video_pns_vertical_features.py
tests/test_video_pns_monotone_correspondence.py
tests/test_video_pns_vertical_curve.py
tests/test_video_pns_lowres_mesh.py
tests/test_video_pns_pair_candidates.py
tests/test_video_pns_candidate_scoring.py
tests/test_video_pns_flex_renderer.py
tests/test_video_pns_output_first.py
tests/test_video_pns_review_package.py
```

### 20.2 合成场景

1. 多条重复货架线：Top-K 保留多个，路径不跨层；
2. 多层亮度相同：单行亮度失败，方向/颜色/上下文消歧；
3. `Delta y` 随高度线性变化；
4. `Delta y` 分段变化但连续；
5. translation 不够、curve 恢复；
6. curve 不够、mesh 改善；
7. 交叉候选被单调路径拒绝；
8. 左右不一致候选降权/拒绝；
9. 无纹理带允许 unmatched；
10. 前景横穿 seam 时 warp/blend 为 0；
11. 深度边缘、孔洞、透明保护区不改；
12. GraphCut 异常后仍输出 baseline；
13. mesh 负 Jacobian 后仍输出 curve/baseline；
14. MultiBand 重影后保留无 blend 候选；
15. 全部增强异常时仍生成 PNSF-0；
16. hard-owner fallback 不扩大旧 owner；
17. 黑色 RGB 有效；
18. owner 单调唯一；
19. 每 source 一次最终 RGB sampling；
20. 同输入同配置确定输出。

### 20.3 现有回归

```text
tests/test_video_session.py
tests/test_video_scan_segment.py
tests/test_video_motion_resampler.py
tests/test_video_algorithm_contract.py
tests/test_video_candidate_config.py
tests/test_video_graphcut_seam.py
tests/test_video_hard_guards.py
tests/test_video_near_blend.py
tests/test_video_photometric.py
tests/test_video_object_mask.py
tests/test_video_object_owner_lock.py
tests/test_video_final_sampling.py
tests/test_video_visual_renderer_v2.py
tests/test_video_delivery.py
tests/test_video_performance.py
tests/test_rgbd_odometry.py
tests/test_orbslam3_bridge.py
tests/test_cuda_backend.py
```

---

## 21. 真实数据与 split

保持：

```text
development: [0.00, 0.30] + [0.48, 0.68]
validation:  [0.30, 0.48] + [0.68, 0.84]
holdout:     [0.84, 1.00]
```

### 21.1 `run_20260807_140140`

用途：

- 2.154 s 快速扫描；
- all-frame direct ORB；
- PNSF-0 密度；
- 横线阶梯；
- 主性能。

目标：

- 优先 98/98 direct pose；
- 97 条 CUDA Open3D 边；
- PNSF-0 owner P50 约 15 px 内、P95 约 30 px 内；
- PNSF-1 横线残差明显下降；
- PNSF-3 多数可处理 pair 不再默认 hard cut；
- 100% 观看无规律性大块接缝。

### 21.2 `run_20260806_153033`

用途：

- D3 视觉参考；
- 长水平梁；
- 长序列内存和性能。

D3 插值 prior 只作视觉参考，不进入 PNS-Flex pose/owner。

### 21.3 `run_20260804_162340`

用途：

- 重复货架纹理；
- 纸箱、风扇、电缆；
- 丢帧、大 gap、fail-closed 压力；
- 当前未消费 holdout。

未冻结配置前不得查看 holdout。

---

## 22. 人眼标定

### 22.1 观看条件

- 100% 是产品判断；
- 200% 辅助；
- 400–600% 仅定位；
- 不因放大一像素差异判正常观看失败。

### 22.2 隐藏标签 A/B

每阶段随机、隐藏名称比较：

```text
上一通过阶段
当前阶段
D3 参考（153033）
```

检查：

- 能否指出 seam；
- 横线是否台阶、重复或变粗；
- 物体内部是否切块/色带；
- 轮廓是否复制、缺失、重影；
- 当前更好/相同/更差。

### 22.3 视觉配置

`pns_flex_visual_calibration_v1.json` 只来自冻结 development 人审，用于：

- 指标归一化；
- 测量噪声；
- 相对排序；
- JND 附近简单模型优先。

不能改变 pose、source、owner、Jacobian、保护域和 holdout 生命周期。

---

## 23. 分支与提交

### 23.1 分支前

```powershell
Set-Location 'D:\central_strip_Panoramic_Camera'
git status --short
git diff --name-only
git branch --show-current
git rev-parse HEAD
git branch --list 'codex/pns-flex-output-first'
```

存在未提交 tracked 修改时停止并报告，不得自行 stash/reset。

安全后：

```powershell
git switch -c codex/pns-flex-output-first <approved-base-commit>
```

### 23.2 建议提交

```text
docs(video): add PNS-Flex output-first implementation plan
test(video): lock PNS-Flex source and pair candidate contracts
feat(video): add densest direct-ORB output-first baseline
feat(video): add multi-candidate vertical strip matching
feat(video): add monotone correspondence and robust vertical curve
feat(video): add selective low-resolution pair mesh
feat(video): add flexible GraphCut and safe blend candidate pool
feat(video): add relative held-out pair candidate selection
feat(video): add temporal photometric continuity
feat(video): add single-owner object continuity
test(video): lock PNS-Flex development and validation evidence
```

只按明确路径 `git add`，禁止 `git add .`，不提交 data、大型 benchmark 或缓存。

---

## 24. 检查点

### Checkpoint 1 — PNSF-0

展示：

- 全部/最密 direct source 数；
- owner overlay；
- PNSF-0 全景；
- 横线和物体 ROI；
- ORB/Open3D/总耗时。

### Checkpoint 2 — PNSF-1

展示：

- 重复纹理 Top-K；
- 单调路径；
- `Delta y(y)`；
- warp 前后横线；
- protected 区；
- selected/fallback pair。

### Checkpoint 3 — PNSF-3

展示策略直方图：

```text
baseline hard owner
curve hard owner
mesh hard owner
identity GraphCut
curve GraphCut
mesh GraphCut
2px blend
4px blend
```

确认 GraphCut/MultiBand 得到执行机会，而非被固定门挡住。

### Checkpoint 4 — 最终

用户隐藏标签 100% 观看确认。某阶段已肉眼不可见时停止增加复杂度。

---

## 25. Go 条件

### 25.1 结构

- 最终 source 全是真实 direct ORB pose；
- 最终相邻边有 CUDA Open3D；
- 无插值 pose；
- 一次最终 RGB sampling；
- owner 单调、唯一、真实；
- valid/crop/owner 一致；
- accepted warp 正 Jacobian；
- protected changed pixels = 0；
- 无补洞、复制、无来源颜色；
- 原子发布完整。

### 25.2 输出优先

- pair-local 算法失败不阻止其它 pair/全景；
- 全部增强失败仍生成 PNSF-0；
- fallback 不扩大旧 owner；
- 结构完整但视觉残差时发布 C，不是 F；
- 报告算法实际尝试/接受/拒绝/预算跳过。

### 25.3 视觉

- 横线 100% 观看无明显阶梯；
- 重复货架无跨层错误；
- 物体内部无规则色带/大块切换；
- 轮廓无复制、缺失、双边；
- 当前阶段适用 ROI 多数更好，非适用 ROI 不变；
- 用户完成隐藏标签确认。

### 25.4 性能

- 分开记录 ORB、Open3D、matching、curve、mesh、GraphCut、blend、sampling、发布；
- audit/review 不冒充主 SLA；
- 增强超预算保留最好合法结果；
- 不抽稀 source 伪造性能；
- 正式最大 60 s 不变。

---

## 26. No-Go

禁止：

- 一个固定视觉 P95 阻止所有后续候选；
- GraphCut 未运行就报告 GraphCut 视觉通过；
- GraphCut reject 后扩大旧 owner；
- 所有重复特征候选同时当真值；
- 对应关系上下交叉；
- 位移随高度跳几十像素；
- 用 DP 生成最终 seam 或冒充 GraphCut；
- 用 blur/MultiBand 掩盖未对齐横线；
- 在对象、遮挡、深度边缘融合；
- 用 interpolated pose 提高正式 source 密度；
- 全景 flow、全局单应、全图 mesh；
- mesh 失败后终止整幅全景；
- 可选算法异常导致旧 delivery 残留；
- 为好看结果修改 holdout/正式标注；
- 自动修改 production lock。

---

## 27. 验证命令

```powershell
$G305Python = 'D:\Panoramic_Camera\.conda\python.exe'

& $G305Python -m pytest -q <本阶段相关测试>
ruff check src tests
& $G305Python -m compileall -q src tests
git diff --check
```

最终：

```powershell
& $G305Python -m pytest -q
ruff check src tests
& $G305Python -m compileall -q src tests
git diff --check
```

真实 CUDA：

```powershell
$env:G305_CUDA = 'required'
& $G305Python .\scripts\verify_open3d_cuda.py
```

验收报告分别注明：

- 单元/合成；
- 真实完整 ORB-SLAM3；
- 真实 CUDA Open3D；
- 三份历史真实数据；
- 人眼盲评；
- 性能重复；
- 新 60 FPS 现场采集；
- 未完成/未验证项。

---

## 28. Codex 最终交付

必须交付：

1. 新分支与实际基点 commit；
2. 分阶段 commit；
3. 修改文件；
4. PNSF-0 到最终候选的算法/像素差异；
5. direct ORB source 数与覆盖；
6. Open3D edge 数、backend、状态；
7. owner 宽度与 provenance；
8. 每对候选尝试/拒绝/选择统计；
9. `Delta y(y)` / mesh 审计；
10. GraphCut/MultiBand 实际调用和接受数；
11. 每阶段性能；
12. development/validation；
13. 用户视觉包；
14. 现场采集状态；
15. 未完成项与推荐阶段；
16. production lock 未修改声明。

最后：

```powershell
git status --short
git log --oneline <base-commit>..HEAD
git diff --stat <base-commit>...HEAD
```

不得自动合并、推送或冻结 production。生产冻结是用户确认后的独立任务。

---

## 29. 最终结论

PNS-Flex 不再采用：

```text
一个固定门失败
→ 后续算法不运行
→ 立即 hard cut
```

而采用：

```text
先建立一定能出图的 midpoint hard-owner 基线
→ 每对真实条带保留多个纵向候选
→ 左右一致、单调不交叉、平滑约束选对应路径
→ 拟合随高度变化的 Delta y(y)
→ 必要时局部低分辨率 2-D mesh
→ 在 identity / curve / mesh 上真实尝试 GraphCut
→ 对安全背景尝试 0 / 2 / 4 px MultiBand
→ 结构非法淘汰，合法候选按 held-out 相对视觉质量排序
→ 选择最好结果；最差仍有 midpoint hard owner
→ 结构完整就发布，视觉残差决定 A/B/C，不决定有没有图
```

这条路线保留真实 pose、owner 和局部几何安全边界，同时让局部 warp、GraphCut 和 MultiBand 真正得到执行和比较机会，直接针对多层横线阶梯与重复货架纹理。
