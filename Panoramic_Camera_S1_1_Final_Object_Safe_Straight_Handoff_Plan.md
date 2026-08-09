# Panoramic Camera S1.1 最终实施方案

## 对象安全直线交接、局部真实 source 补选和亚像素纵向对齐

---

## 1. 最终技术决定

本方案在现有 `S01_output_first_vertical_alignment_v1` 基础上新增独立的 S1.1 实验路线。

统一名称：

```text
S011_object_safe_straight_handoff_v1
```

建议配置：

```text
configs/video_candidates/S011_object_safe_straight_handoff_v1.yaml
```

继续使用开发专用入口：

```text
g305-video-s1-experiment
```

S1.1 的最终处理顺序如下：

```text
完全复用 S01 的初始 source 和初始 canvas 布局
    ↓
寻找每对 source 中同一结构在左右图中的两个 canvas 位置
    ↓
把两个位置之间标记为禁止切割区
    ↓
先移动整条竖直接缝，使物体完整属于一个 source
    ↓
附近没有安全直线时，只给这个风险 pair 补选真实中间帧
    ↓
仍然没有安全直线时，标记 needs_s2
    ↓
横向 owner 确定后，再做纵向粗对齐和亚像素精修
    ↓
强物体边缘 hard owner，不做混合
    ↓
每个最终 source 只执行一次全分辨率 RGB remap
```

本方案不再默认把整段 source 全局加密到 8 至 12 px。

默认保留现有 S01 的 source 数量和速度基础，只对真正存在对象重复风险、又找不到安全直线的局部 pair 增加真实 source。

这样可以同时处理：

- 箱子斜边出现重复三角形
- 右侧 source 带入左侧已经出现过的内容
- 货架横线仍有约 1 px 阶梯
- 首尾完整画面不能裁掉
- 全局 source 加密导致处理时间大幅增加

---

# 2. 目标

## 2.1 必须解决

1. 首帧左侧和末帧右侧完整画面继续保留。
2. S01 初始 source、source center、support 和 canvas 尺寸保持不变。
3. 内部 owner 边界不再只能使用相邻 source 中点。
4. 同一近景结构的左右投影之间不得放置安全直线接缝。
5. 无安全直线时只在当前风险 pair 内补选真实帧。
6. 强物体边缘和风险区域使用单 owner，不进行双源混合。
7. 横向交接完成后，再处理 `dy(y)`。
8. missing 高度区域必须真正使用 `dy=0`。
9. 前后 pair 的纵向 warp 不能在错误边界相互抵消。
10. 一条 pair correction 必须同时提交或同时回滚。
11. 每个最终 source 只执行一次全分辨率 RGB remap。
12. 两组真实数据无论局部算法是否成功，都必须生成完整实验结果和诊断。

## 2.2 质量目标

横向交接：

```text
安全 pair 的 protected crossing = 0
同一高置信物体结构在最终图中只出现一次
箱子斜边不再出现第二个三角形
强边缘处不出现灰色混合带
```

纵向对齐：

```text
最终亚像素边缘残差 global P95 <= 0.5 px
每个可评估 pair 的 P95 <= 0.75 px
P99 <= 1 px
旧整数阶梯 P95 = 0 px 作为冲刺目标
```

## 2.3 性能目标

第一优先级：

```text
最终最小输出模式不慢于原 S01 的 1.15 倍
```

进一步目标：

```text
total <= 20 s
candidate generation <= 12 s
```

冲刺目标：

```text
total <= 15 s
candidate generation <= 8 s
```

性能不作为阻止出图的门限。

---

# 3. 明确不做的内容

S1.1 不实现：

- 随高度弯曲的 seam
- GraphCut
- 逐行 seam 动态规划
- 多标签 owner
- DIS 或 RAFT 横向形变 RGB
- 二维 mesh
- APAP
- 单应或仿射修图
- 深度分层
- RGB-D 或 TSDF 纹理补图
- 大模型语义分割
- 对象长期跟踪
- production 发布

当一条整高直线无法同时避开不同高度的物体时：

```text
needs_s2 = true
```

不得偷偷加入 S2，也不得通过模糊掩盖。

---

# 4. 工程隔离

S1.1 必须保持：

```text
diagnostic_only: true
production_eligible: false
motion_role: diagnostic_layout_evidence_only
```

不得修改：

```text
g305-video-panorama
production renderer
production lock
video_delivery.json
正式视频产品契约
```

现有 v1 必须继续可复现：

```text
algorithm_id: S01_output_first_vertical_alignment_v1
config: configs/video_candidates/S01_output_first_vertical_alignment_v1.yaml
```

S1.1 不能原地覆盖 v1。

建议 v2：

```text
schema: gemini305-video-s1-experiment/v2
algorithm_id: S011_object_safe_straight_handoff_v1
report_schema: gemini305-video-s1-experiment-report/v2
```

---

# 5. 首尾完整画面不变量

继续保持：

```text
first.owner_left_x == first.support_left_x
last.owner_right_x == last.support_right_x
```

不得：

- 裁掉首帧左侧
- 裁掉尾帧右侧
- 改变首末 source
- 缩短 canvas
- 删除首尾 source
- 为了指标裁边

两组真实数据最终尺寸应保持：

```text
run_20260806_153033: 480 × 1839
run_20260807_140140: 480 × 2155
```

首末真实帧保持：

```text
慢速：60 / 848
快速：31 / 128
```

远离内部 warp 区域的首帧左翼和末帧右翼应与 v1 保持一致。

---

# 6. 三阶段固定输出

S1.1 内部固定输出三个阶段。

## Stage A

```text
s11_nominal_midpoint_owner_only.png
```

含义：

- 使用 S1.1 最终 refined source 集合
- 使用这些 source 的名义 midpoint boundaries
- 不做纵向修正
- 不做过渡

用于判断局部补帧本身带来的变化。

## Stage B

```text
s0_panorama_owner_only.png
```

含义：

- 与 Stage A 使用相同 source
- 使用最终对象安全直线 boundaries
- 不做纵向修正
- 不做过渡

Stage A 到 Stage B 只评价横向 handoff。

## Stage C

```text
s11_panorama_owner_only.png
```

含义：

- 与 Stage B 使用相同 source
- 使用相同 straight boundaries
- 应用最终 accepted `dy(y)`
- 不做过渡

Stage B 到 Stage C 只评价纵向修正。

原 S01 只是外部只读基线，不属于 A、B、C。

---

# 7. 初始 source 必须完全继承 S01

S1.1 第一次布局必须逐项复用 v1：

```text
source frame IDs
source indices
canvas_center_x
support_left_x
support_right_x
初始 midpoint boundaries
canvas width
```

不得默认使用更小的全局 target step 重新选整段 source。

原因：

- 普通区域没有必要增加 pair
- 当前主要可见问题集中在少量对象交接处
- 全局加密会把 33 个 source 增加到约 80 个
- 当前候选生成会因此明显变慢
- 无法判断改善来自对象安全交接还是全局加帧

只有满足以下条件的 pair 才允许局部补选真实 source：

```text
横向证据可评估
存在 hard forbidden interval
当前 midpoint 不安全
允许范围内没有安全直线
```

---

# 8. 唯一坐标域

必须明确四个坐标域：

```text
R：原始带畸变 RGB 像素
A：低分辨率 handoff 分析图
C：全分辨率标定目标图
X：最终 canvas
```

handoff 证据建议直接从 C 域生成低分辨率 A 域图像。

推荐：

```text
analysis_width = 424
analysis_height = full calibrated height
```

只缩小水平方向，不缩小高度。

A 到 C 的像素中心换算：

```text
x_C = (x_A + 0.5) × scale_x - 0.5
y_C = y_A
```

C 到 canvas：

```text
x_canvas = source.support_left_x + x_C
```

禁止：

```text
support_left_x + raw_x
support_left_x + analysis_x
```

直接相加。

所有 handoff interval 在进入 boundary solver 前必须转换为全分辨率 canvas x。

---

# 9. 横向对象安全证据

## 9.1 目标

横向证据只回答：

1. 当前 pair 是否存在背景运动无法解释的局部横向视差？
2. 一条整高直线放在哪里不会把同一结构切成两份？

它不能：

- 修改 pose
- 生成 flow field
- 横向 remap RGB
- 拟合二维形变
- 生成新图像

## 9.2 特征跟踪

每个 pair 在共同 corridor 中执行：

```text
goodFeaturesToTrack
    ↓
pyramidal LK left→right
    ↓
pyramidal LK right→left
    ↓
candidate-specific forward/backward check
```

可信 observation 必须满足：

- 左右跟踪成功
- forward/backward error 合格
- 坐标有限
- 双方 valid
- 垂直误差合格
- 局部纹理足够
- 没有越出共同 corridor

不得用单个全局 RANSAC 删除所有不符合背景运动的近景点。

这些点正是需要保护的对象视差证据。

## 9.3 canvas 投影

对一个左右对应：

```text
left_canvas_x
right_canvas_x
```

定义：

```text
interval_left  = min(left_canvas_x, right_canvas_x)
interval_right = max(left_canvas_x, right_canvas_x)
interval_width = interval_right - interval_left
```

如果 owner boundary 落在两者之间，同一结构可能：

- 左图先画一次
- 右图再画一次

或者反向运动时：

- 左右都不画完整
- 形成漏口

## 9.4 背景主残差

在 canvas 域中，正确背景对应应接近同一 canvas x。

使用可信对应的稳健中位数估计背景 canvas residual。

每个 observation 计算：

```text
relative_dx = observation_canvas_dx - background_canvas_dx
```

近景物体通常具有明显不同的 relative dx。

## 9.5 hard forbidden interval 不能由单点生成

一个普通强边或单个特征不能直接创建 hard interval。

必须形成一致 cluster。

推荐初始条件：

```text
minimum match count: 4
minimum vertical span: 24 px
minimum interval width: 1.5 px
minimum relative dx: 1.0 px
same residual sign
neighbor x <= 24 px
neighbor y <= 32 px
relative dx difference <= 1.0 px
merge gap <= 2 px
guard <= 3 px
```

只有通过 cluster 审计的区域才生成：

```text
hard forbidden interval
```

不足条件的证据只进入 soft cost。

## 9.6 aligned depth 的作用

aligned depth 只允许：

- 给已由 RGB 触发的 observation 增加风险权重
- 标记深度边缘
- 降低不稳定 observation 的置信度

不得：

- 单独创建 hard interval
- 做对象连通分割
- 做遮挡推理
- 决定 owner
- 生成颜色
- 补纹理

depth 无效时继续使用 RGB，不把区域自动视为安全。

---

# 10. solver 和 heldout 分离

横向 evidence 在做任何风险、补帧和 boundary 决策前，按稳定 match ID 分为：

```text
solver
heldout
```

solver 只用于：

- 背景统计
- hard cluster
- forbidden interval
- risk 判断
- source rescue
- boundary cost

heldout 只用于最终成图验收。

不得用 heldout 调整 boundary。

稳定 ID 可以由以下内容的 canonical SHA-256 前 8 字节生成：

```text
calibration ID
origin pair lineage
current pair frame IDs
左右 calibrated quarter-pixel 坐标
```

S1.1 第一轮实现至少必须保证：

- 同一输入重复运行 split 一致
- heldout 不参与 solver
- source rescue 后 origin heldout denominator 不被删除

---

# 11. pair 风险分类

至少分为：

```text
midpoint_safe
straight_boundary_shifted
source_refined_then_safe
unobservable_midpoint
capture_limited
straight_seam_limited
unsafe_fallback_midpoint
global_boundary_conflict
```

## midpoint_safe

- 没有 hard interval 穿过 midpoint
- 继续使用 midpoint

## straight_boundary_shifted

- midpoint 不安全
- 附近存在安全整高直线
- 移动 boundary

## unobservable_midpoint

- 横向特征不足
- 不能证明安全或不安全
- 保持 midpoint
- hard owner
- 纵向 `dy=0`
- `manual_review_required=true`
- 不设置 `needs_s2`

不可观测不等于安全。

## capture_limited

- 已是相邻真实帧
- 或没有满足改善条件的真实中间帧
- 仍无安全直线
- 标记 `needs_s2=true`

## straight_seam_limited

- 已执行允许的真实 source rescue
- 所有整高直线仍命中冲突
- 标记 `needs_s2=true`

---

# 12. 对象安全直线 boundary

## 12.1 候选范围

每个内部 pair 在 midpoint 附近搜索：

```text
±48 full-resolution px
```

候选必须同时位于：

- 左右 source 的共同 support
- 双方有效 corridor
- 全局单调 owner 允许范围

最小内部 owner 宽度：

```text
9 px
```

## 12.2 候选代价

```text
cost(x) =
    hard forbidden crossing
  + soft crossing
  + RGB strong-edge crossing
  + optional depth-edge crossing
  + invalid support
  + midpoint distance
```

规则：

- 穿过 hard interval 的 candidate 不能被当作安全解
- midpoint distance 只用于多个安全候选之间的选择
- 不能为了颜色相似切进箱子
- 不得用最终图像反向优化 boundary

## 12.3 全局一维 DP

所有内部 boundaries 统一优化。

状态只包含：

```text
每条整高直线的 x
```

约束：

- boundary 严格递增
- owner 无 gap
- owner 无 overlap
- 内部 owner 不小于最小宽度
- 首尾边界不改变
- canvas extent 不改变

这是整高直线的全局离散优化，不属于 S2。

## 12.4 强边缘 hard owner

主验收图：

```text
protected_transition_width_px = 0
safe_background_transition_width_px = 0
```

所有区域均使用单 owner。

如需查看 1 px 过渡，只能另行输出：

```text
diagnostic_transition_preview.png
transition_mask.png
```

该预览不参与验收。

---

# 13. 局部真实 source rescue

## 13.1 触发条件

只在以下情况触发：

```text
pair 可评估
midpoint 命中 hard interval
附近所有允许的整高直线都不安全
```

普通 pair、midpoint-safe pair 和 unobservable pair 不补帧。

## 13.2 选择方法

在左右 source 的原始帧开区间中选择真实帧。

先把 scan progress 从 analysis px 换算到 full-resolution px：

```text
progress_full = progress_analysis × calibrated_width / analysis_width
```

候选排序：

1. 最小化两个 child step 的最大值
2. 最小化两侧 child step 差
3. 画质 penalty
4. frame index 升序

候选必须使最大 child step 至少改善：

```text
2 px
```

否则返回无候选。

## 13.3 限制

```text
maximum refinement rounds = 2
maximum insertions per original pair = 2
```

新增 source 必须：

- 是真实落盘帧
- 位于左右 source 时间区间
- 保持时间顺序
- 有真实 RGB 文件
- 有合法 scan progress
- 不改变首尾 source
- 不改变 canvas extent

## 13.4 rescue 后重建

加入真实 source 后重新计算：

- source layout
- child pair
- calibrated handoff cache
- sparse observations
- hard clusters
- forbidden intervals
- boundary candidates
- global DP

不得沿用 parent pair 的旧 interval。

---

# 14. owner 和 provenance

## 14.1 几何 owner

所有 owner 使用半开区间：

```text
[left, right)
```

必须完整覆盖 canvas x，且：

```text
column gap = 0
column overlap = 0
```

## 14.2 像素 owner

最终 valid 像素必须有且只有一个真实 frame owner。

最终 invalid 像素：

```text
owner = -1
```

不能用黑色判断 valid。

## 14.3 禁止全局 foreign fill

S1.1 规范主图：

```text
allow_foreign_owner_fill = false
allow_adjacent_border_fill = false
```

owner source 无效时保留 invalid，不允许远端 source 补入对象内容。

为了观察完整画面，可以额外输出：

```text
s11_presentation_preview.png
presentation_fill_mask.png
```

presentation preview 只允许：

- 直接相邻 source
- 安全背景
- 不穿过 hard interval
- 不参与几何和质量验收

规范结果仍是 owner-only 主图。

## 14.4 Stage A、B、C 必须分别保存

```text
geometric_owner_frame_id
declared_owner_sample_valid
final_owner_frame_id
final_valid
```

真实验收以这些数组为准，不以彩色 owner preview 为准。

---

# 15. 横向 handoff 后的纵向 S1.1

纵向对齐必须在最终 straight boundary 和最终 pair corridor 上重新运行。

## 15.1 coarse-to-fine

普通 pair 粗层：

```text
window height = 32 px
step = 16 px
search center = motion dy 或 0
search radius = ±6 px
Top-K = 3
```

风险 pair 或普通搜索触边：

```text
expanded radius = ±16 px
Top-K = 5
```

细层：

```text
window height = 16 至 24 px
step = 8 px
residual radius = ±2 px
step = 0.25 px
最佳位置再做三点抛物线细化
```

细层只修 residual，不输出 dx。

## 15.2 独立水平子窗口

使用三个近独立子窗口。

要求：

- 至少 2/3 同意
- spread <= 0.75 px
- 每个正向候选单独执行反向检查
- 低纹理、低唯一性、FB 失败、触边时允许 missing

## 15.3 trusted rows

accepted windows 转为 trusted-row mask。

要求：

1. 连续 trusted segments 分段拟合
2. missing window 核心行 `dy_applied=0`
3. 不跨长 missing gap 外推
4. segment 边缘 raised-cosine 回到 0
5. coverage 不足时当前 pair 整体 dy=0

状态：

```text
s11_full_rows
s11_partial_rows
s11_zero_unobservable
s11_atomic_rollback
```

---

# 16. 对象 reference source 和纵向保护

根据最终 boundary 相对 hard intervals 的位置，确定对象由哪一侧提供。

## reference_source = left

对象由左 source 提供：

- 左 source 保持 identity
- 只修正右 source
- 右侧失败时 pair 两侧都回到 identity

## reference_source = right

对象由右 source 提供：

- 右 source 保持 identity
- 只修正左 source
- 左侧失败时 pair 两侧都回到 identity

## reference_source = both

不同 hard clusters 分别由左右 source 提供。

一条 pair 级纵向 warp 无法同时保护两侧对象：

```text
左右都保持 identity
vertical_skipped_mixed_reference
```

## reference_source = none

普通安全背景 pair：

```text
左侧 -0.5dy
右侧 +0.5dy
```

---

# 17. 不重叠的 warp support

匹配 corridor 可以是 128 px。

实际修改 inverse map 的 warp support 必须很窄。

推荐：

```yaml
warp_support_maximum_px: 16
warp_support_owner_fraction: 0.45
warp_support_minimum_px: 4
```

对于内部 source owner 宽度 `w`：

```text
desired = clamp(floor(0.45 × w), 4, 16)
```

有左右两个 pair 时：

```text
left support 和 right support 必须不重叠
至少保留 1 px 不属于任何 pair support
```

在 pair boundary：

```text
当前 pair 权重 = 1
远端相邻 pair 权重 = 0
```

这保证估计的 pair dy 不会被前后 pair 抵消。

---

# 18. pair 原子事务

一个 pair 的左右 component 作为一个事务。

允许：

```text
identity + corrected
corrected + identity
corrected + corrected
identity + identity
```

不允许：

```text
一侧提交
另一侧因审计失败却保留旧半边修正
```

若任一 corrected component 出现：

- 非有限
- 越界
- 垂直映射不单调
- slope 超限
- valid coverage 异常

则：

```text
pair_commit_status = rolled_back
左右 component 都回到 identity
```

只回滚当前 pair。

其他安全 pair 保持不变。

---

# 19. 每个 source 一次全分辨率 remap

继续使用联合 map。

每个 source 同时构造：

```text
identity calibration map
vertical-corrected calibration map
```

在 map 维度拼接后，执行一次：

```python
cv2.remap(...)
```

一次得到：

- identity contribution
- corrected contribution

要求：

```text
full_resolution_remap_invocations == final_source_count
```

Stage A、B 复用 identity contribution。

Stage C 使用 corrected contribution。

不得为三个阶段重复 remap RGB。

---

# 20. 最终成图指标

## 20.1 横向 single-copy

不能只检查 boundary 是否避开 solver 自己的 interval。

必须使用 heldout 对应在最终 owner-only 图中检查：

```text
copy_count = 0  表示漏画
copy_count = 1  表示正确
copy_count >= 2 表示重复
```

安全 pair 必须：

```text
Stage B duplicate = 0
Stage B missing = 0
Stage C duplicate = 0
Stage C missing = 0
heldout evaluable fraction >= 0.90
```

## 20.2 纵向亚像素残差

纵向指标固定比较 Stage B 到 Stage C。

对左右边缘 ridge 做斜率补偿，外推到唯一物理 boundary：

```text
x = B - 0.5
```

避免把连续斜边天然的行差误判为阶梯。

输出：

```text
stage_b_subpixel_edge_residual_p50/p95/p99/max
stage_c_subpixel_edge_residual_p50/p95/p99/max
coverage
```

## 20.3 旧整数指标

保留：

```text
legacy_integer_peak_row_gap_p50
legacy_integer_peak_row_gap_p95
```

不得再静默删除超过 8 px 的样本。

整数 P95=0 只是冲刺目标。

## 20.4 防止模糊作弊

同时报告：

```text
measurement sample count
edge peak strength
edge width
double edge count
```

不能通过把边缘模糊到检测不到来获得更低残差。

---

# 21. 慢速箱子固定回归

已知问题位置：

```text
session: run_20260806_153033
origin pair lineage: 477 -> 501
baseline pair index: 16
baseline boundary x: 951
baseline crop: x=931:993, y=286:397
```

这些数值只用于固定回归定位，不得硬编码进算法。

必须输出：

```text
baseline_v1_crop.png
stage_a_nominal_midpoint_crop.png
stage_b_handoff_only_crop.png
stage_c_final_crop.png
handoff_overlay.png
forbidden_intervals.csv
box_acceptance.json
```

自动检查：

- 箱子斜边只出现一个主边峰
- duplicate row = 0
- gap row = 0
- 斜率补偿后的最大跳变 <= 1 px
- valid coverage >= 90%
- protected crossing after = 0
- protected transition pixel count = 0

人工检查：

- 不再出现第二个重复三角形
- 不出现宽灰色混合带
- 箱子斜边连续
- 箱体尽量完整来自一个 source
- 不能把该固定样例留成 `unobservable`
- 不能把该固定样例直接推给 S2

---

# 22. 快速数据重点回归

重点检查原 S01 中的长尾和近景 pair：

```text
pair 14
pair 31
pair 32
```

source rescue 后 pair index 可能改变，因此必须使用：

```text
origin_pair_lineage
```

跟踪。

如果原始相邻真实帧已经没有可插入帧：

```text
capture_limited = true
needs_s2 = true
```

不能生成虚拟帧。

---

# 23. 性能实现

## 23.1 只对风险 pair 运行完整 handoff

普通 midpoint-safe pair：

- 使用缓存的轻量证据
- 不运行 source rescue
- 不运行完整 fine vertical，除非 Stage B 残差较大

风险 pair：

- 运行完整 handoff
- 必要时 rescue
- 运行 fine vertical

## 23.2 缓存

每个 source 缓存：

- calibrated low-resolution gray
- valid mask
- gradients
- GFTT points
- S1 vertical features

新增 source 只计算自己的缓存。

## 23.3 纵向 cost volume 批量计算

普通粗搜索：

```text
-6 到 +6
```

一次批量构造。

风险扩大搜索：

```text
-16 到 +16
```

只在必要窗口执行。

避免逐窗口、逐 dy 的 Python 内层循环。

## 23.4 两种输出模式

### audit

输出全部 pair 证据、CSV 和 crop。

### minimal

只输出：

- Stage A/B/C 主图
- owner/valid provenance
- report.json
- performance.json
- 最差若干 crop

正式性能比较必须使用 minimal。

---

# 24. 推荐配置

```yaml
schema: gemini305-video-s1-experiment/v2
algorithm_id: S011_object_safe_straight_handoff_v1

diagnostic_only: true
production_eligible: false

scan:
  analysis_width: 320
  motion_backend: lk_ransac
  fallback_motion_backend: phase_correlation
  maximum_workers: 4

source_selection:
  initial_selection_policy: inherit_s01_v1_exact
  rescue_step_units: full_resolution_px
  minimum_rescue_improvement_px: 2.0
  maximum_refinement_rounds: 2
  maximum_insertions_per_original_pair: 2
  preserve_full_fov_endpoints: true

handoff:
  enabled: true
  coordinate_domain: calibrated_target_analysis

  analysis_width_px: 424
  match_corridor_width_px: 128
  maximum_boundary_shift_px: 48
  minimum_internal_owner_width_px: 9

  minimum_match_count: 12
  minimum_protected_match_count: 4
  minimum_protected_vertical_span_px: 24

  cluster_neighbor_x_px: 24.0
  cluster_neighbor_y_px: 32.0
  cluster_relative_dx_tolerance_px: 1.0
  cluster_merge_gap_px: 2.0

  lk_forward_backward_max_px: 0.75
  maximum_vertical_match_error_px: 4.0

  protected_interval_min_width_px: 1.5
  minimum_background_relative_dx_px: 1.0
  conflict_guard_fullres_px: 3.0

  use_aligned_depth_for_risk: true
  straight_seam_only: true

owner:
  protected_transition_width_px: 0
  safe_background_transition_width_px: 0
  allow_foreign_owner_fill: false
  allow_adjacent_border_fill: false

s1:
  horizontal_scale: 0.5
  preserve_full_vertical_resolution: true

  normal_coarse_search_radius_y_px: 6
  expanded_coarse_search_radius_y_px: 16

  coarse_window_height_px: 32
  coarse_window_step_px: 16

  fine_window_height_px: 24
  fine_window_step_px: 8
  fine_search_radius_y_px: 2.0
  fine_search_step_px: 0.25

  normal_top_k_candidates: 3
  expanded_top_k_candidates: 5
  candidate_nms_radius_px: 2

  minimum_independent_subwindow_agreement: 2
  maximum_subwindow_spread_px: 0.75

  trusted_row_taper_px: 8
  maximum_consecutive_missing_windows: 2

  warp_support_maximum_px: 16
  warp_support_owner_fraction: 0.45
  warp_support_minimum_px: 4

  protected_source_reference_warp: true
  maximum_local_slope: 0.08
  fade_function: cosine

output:
  mode: audit

  write_stage_a_nominal_midpoint: true
  write_stage_b_handoff_only: true
  write_stage_c_final: true

  write_stage_provenance: true
  write_pair_reports: true
  write_acceptance_crops: true
  write_performance_report: true
```

配置必须拒绝：

```text
preserve_full_fov_endpoints = false
straight_seam_only = false
protected_transition_width_px > 0
warp_support_owner_fraction >= 0.5
minimum_internal_owner_width_px < 2 × warp_support_minimum_px + 1
```

---

# 25. 建议文件结构

修改：

```text
src/panorama_demo/video_s1_config.py
src/panorama_demo/video_s1_layout.py
src/panorama_demo/video_s1_vertical_alignment.py
src/panorama_demo/video_s1_warp.py
src/panorama_demo/video_s1_renderer.py
src/panorama_demo/video_s1_diagnostics.py
src/panorama_demo/video_s1_experiment.py
README.md
pyproject.toml
```

新增：

```text
src/panorama_demo/video_s1_handoff.py
src/panorama_demo/video_s1_metrics.py

configs/video_candidates/S011_object_safe_straight_handoff_v1.yaml

tests/test_video_s1_handoff.py
tests/test_video_s1_metrics.py
```

扩展：

```text
tests/test_video_s1_config.py
tests/test_video_s1_layout.py
tests/test_video_s1_vertical_alignment.py
tests/test_video_s1_warp.py
tests/test_video_s1_renderer.py
tests/test_video_s1_integration.py
```

暂时不要求实现庞大的发布审批系统。

第一轮只保留必要的：

- baseline/session hash
- source lineage
- required pair report
- Stage A/B/C provenance
- 固定箱子 crop
- 最终成图指标
- production 路线未修改检查

算法稳定后再补完整 acceptance 汇总工具。

---

# 26. 实施顺序

## 任务 0：冻结 v1

- 给 v1 source/layout 增加兼容回归
- 固定两组 v1 baseline report 和图像 hash
- 确认新代码不改变 v1 默认输出

## 任务 1：v2 配置和三阶段骨架

- 新增 v2 schema
- CLI 根据配置选择 v1 或 v2
- 输出 Stage A/B/C
- 暂时保持 midpoint，不改变画面

## 任务 2：calibrated handoff evidence

- 新建 `video_s1_handoff.py`
- 生成 A 域 evidence cache
- 实现 GFTT + LK F/B
- 转换到 C/X 域
- 实现 solver/heldout split
- 实现 background residual 和 cluster
- 输出 forbidden intervals

## 任务 3：对象安全 straight boundary

- 生成候选 x
- hard interval crossing 排除
- soft cost
- 全局一维 DP
- 保持首尾和 canvas 不变

## 任务 4：局部真实 source rescue

- 只处理无安全直线的 hard-risk pair
- 按 full-res progress 选择真实帧
- 最多两轮、每 lineage 两张
- 重建 pair 和 evidence

## 任务 5：hard owner 和 provenance

- Stage A/B/C owner
- transition=0
- 关闭 foreign fill
- 保存真实 frame owner 和 valid

## 任务 6：trusted rows 和 pair 原子 warp

- 普通 ±6、风险 ±16
- fine ±2
- missing 核心 dy=0
- 分段曲线
- reference source
- non-overlap support
- pair transaction

## 任务 7：最终成图指标

- horizontal single-copy heldout
- slope-compensated vertical residual
- legacy integer metric
- 慢速箱子固定 crop

## 任务 8：两组真实数据

运行：

```text
run_20260806_153033
run_20260807_140140
```

输出：

```text
原 S01
Stage A
Stage B
Stage C
箱子 crop
快速风险 crop
needs_s2 清单
```

## 任务 9：性能优化

- 缓存
- batch vertical cost volume
- 风险 pair 才运行完整 handoff/fine
- audit/minimal 两种输出模式

---

# 27. 必须新增的核心测试

## 坐标域

- A→C 像素中心换算
- C→canvas
- 带畸变时禁止 raw x 直接加 support
- horizontal correction 始终为 0

## hard interval

- 背景 residual 约 0
- 前景 residual 约 6 px
- 多点 cluster 才能生成 hard interval
- 单点和普通强边只能 soft
- boundary 在 interval 左端、内部、右端的半开区间行为

## straight boundary

- midpoint 不安全但旁边有安全线
- midpoint 安全保持不动
- 不同高度冲突导致 needs_s2
- 不可观测保持 midpoint、dy=0

## rescue

- 只对无安全线的风险 pair 补帧
- 插入真实帧
- 保持时间顺序
- 保持首尾
- 无候选时 capture_limited

## owner

- 无 gap、无 overlap
- valid 像素唯一真实 owner
- invalid owner=-1
- transition=0
- foreign owner=0

## vertical

- missing 核心 dy=0
- 不跨长 gap 外推
- support 不重叠
- boundary 当前 pair 权重为 1
- reference identity 和 moving corrected 同事务
- 失败时 pair 原子回滚

## synthetic object

构造：

```text
background canvas residual = 0
foreground residual = 6 px
斜边物体
midpoint 位于 forbidden interval 内
```

要求：

- 选择 interval 外直线
- 物体只出现一次
- 不使用横向 warp
- 不使用 blend

---

# 28. 真实运行命令

慢速：

```powershell
& 'D:\Panoramic_Camera\.conda\Scripts\g305-video-s1-experiment.exe' `
  'D:\central_strip_Panoramic_Camera\data\captures\video\run_20260806_153033' `
  --config 'D:\central_strip_Panoramic_Camera\configs\video_candidates\S011_object_safe_straight_handoff_v1.yaml' `
  --baseline 'D:\central_strip_Panoramic_Camera\benchmarks\run_20260806_153033\S01_output_first_vertical_alignment_v1' `
  --output 'D:\central_strip_Panoramic_Camera\benchmarks\run_20260806_153033\S011_object_safe_straight_handoff_v1'
```

快速：

```powershell
& 'D:\Panoramic_Camera\.conda\Scripts\g305-video-s1-experiment.exe' `
  'D:\central_strip_Panoramic_Camera\data\captures\video\run_20260807_140140' `
  --config 'D:\central_strip_Panoramic_Camera\configs\video_candidates\S011_object_safe_straight_handoff_v1.yaml' `
  --baseline 'D:\central_strip_Panoramic_Camera\benchmarks\run_20260807_140140\S01_output_first_vertical_alignment_v1' `
  --output 'D:\central_strip_Panoramic_Camera\benchmarks\run_20260807_140140\S011_object_safe_straight_handoff_v1'
```

---

# 29. 验收要求

## 结构

- 两组完整出图
- 首尾 full-FOV 保留
- canvas 尺寸不变
- 只用真实 source
- 每 source 一次 remap
- straight seam only
- horizontal warp = 0
- owner 无 gap、无 overlap
- foreign owner = 0
- protected transition = 0

## 慢速箱子

- origin lineage `477→501`
- protected crossing after = 0
- duplicate triangle 消失
- gap = 0
- 强边缘没有灰色 blend
- 不能是 unobservable
- 不能直接留给 S2

## 横向

安全 pair：

```text
rendered duplicate = 0
rendered missing = 0
heldout coverage >= 90%
```

## 纵向

```text
global subpixel P95 <= 0.5 px
per-pair P95 <= 0.75 px
P99 <= 1 px
legacy integer P95 = 0 px 为冲刺目标
```

## 性能

第一轮：

```text
minimal total <= v1 total × 1.15
```

优化目标：

```text
minimal total <= 20 s
```

---

# 30. 完成后进入 S2 的条件

完成以下内容后再进入 S2：

1. 慢速箱子固定回归通过。
2. safe pair 不再有横向 duplicate/missing。
3. 普通横线的亚像素残差达到目标。
4. 首尾完整覆盖保持。
5. 性能没有明显恶化。
6. 剩余问题明确属于：

```text
一条整高直线无法同时避开不同高度对象
```

S2 只处理这些：

```text
needs_s2 pair
```

S2 的核心目标是让 seam 随高度弯曲并绕开对象，而不是重新解决普通横线。

---

# 31. 可直接发送给 Codex 的命令

```text
请读取并严格执行仓库根目录中的：

Panoramic_Camera_S1_1_Final_Object_Safe_Straight_Handoff_Plan.md

仓库：
D:\central_strip_Panoramic_Camera

分支：
codex/video-realtime-seam-v6

实施要求：

1. 先冻结并验证现有 S01_output_first_vertical_alignment_v1。
2. 新增独立的 S011_object_safe_straight_handoff_v1。
3. 不修改 production renderer、production lock、g305-video-panorama 或 video_delivery.json。
4. 保留首帧左侧和末帧右侧完整画面，保持两组 canvas 尺寸不变。
5. 初始 source、center、support 和 midpoint layout 必须逐项继承 S01，不得全局加密。
6. 使用 calibrated RGB 稀疏对应建立 hard forbidden intervals。
7. 先移动整高直线 boundary。
8. 只有 hard-risk pair 没有安全直线时，才补选真实中间 source。
9. rescue 最多两轮，每个 origin pair 最多两张真实帧。
10. 仍无安全直线时标记 needs_s2，不得加入弯曲 seam、横向 warp 或模糊。
11. 主验收图 transition=0、foreign owner fill=false。
12. 横向 handoff 确定后再运行纵向 coarse-to-fine。
13. missing 核心行必须 dy=0。
14. pair warp support 不得重叠，pair correction 必须原子提交或回滚。
15. 每个最终 source 只执行一次全分辨率 RGB remap。
16. 按任务 0 至任务 9 分阶段实现并持续运行测试。
17. 最终必须运行慢速和快速两组真实数据。
18. 必须固定输出慢速 origin pair 477→501 的箱子 before/after crop。
19. 最终报告横向 duplicate/missing、亚像素纵向残差、source rescue、needs_s2 和性能。
20. 在 S1.1 验收前不要进入 S2。
```
