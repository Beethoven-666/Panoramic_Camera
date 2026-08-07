# Panoramic_Camera v6.1 — Anchor-Constrained Narrow Strip Recovery Plan

## 0. 目的

仓库：

```text
https://github.com/Beethoven-666/Panoramic_Camera
```

当前实现分支：

```text
codex/video-realtime-seam-v6
```

当前状态：

```text
v6 candidate 已阻塞
未生成 Production lock
AGENTS.md / README.md 暂未修改
```

当前阻塞不是继续调 GraphCut 参数可以解决的问题：

```text
三份冻结数据均无法同时满足：
double-edge = 0
ghost = 0

T2 = 16 FPS direct-ORB 已实测更差
继续调参数会出现：
数据 A 改善
数据 B 退化
```

本方案不再继续优化“宽近景大 patch / 动态 owner span”路线。

新的核心路线：

> **Direct-ORB 只作为稀疏、可靠的全局 Anchor；最终 RGB 使用高密度真实中央窄带。每两个相邻窄带先做 RGB 亚像素几何对齐，再 GraphCut，再极窄融合。宽物体自然横跨更多窄带，但绝不允许某一条带因为物体很宽而持续加宽。**

本轮仍只验证现有 3 m 数据，不要求重新采集。

---

# 1. 当前 v6 的结构问题

当前 v6 source planner 使用动态 owner span：

```text
fixed_owner_pixel_width = None
```

owner 区间根据 source centre 和 frontality overlap 动态规划。

这会造成一个危险链条：

```text
近景物体很宽
→ 当前 source 为保持物体完整而承担更宽 owner
→ source 左右边缘越来越远离图像中央
→ 与下一 source 的真实观察位置差变大
→ 近景视差增大
→ GraphCut 输入本身已有 double-edge / ghost
→ GraphCut 只能选 seam，无法消除几何双边
→ 后续 blend 也只能把双边变成 ghost / blur
```

因此要修改的不是：

```text
N_req 从 3 改 4
或
T2 从 16 FPS 再提高到 24/30 FPS
```

而是：

```text
彻底取消“物体宽度决定 owner span”。
```

---

# 2. v6.1 新架构

## 2.1 两类 source 完全分离

### A. ORB Pose Anchor

要求：

```text
必须是真实采集 RGB-D 帧
必须拥有真实 direct-ORB camera_to_world
不允许插值 pose
不允许 DIS / Open3D 补 pose
```

职责：

```text
提供全局扫描方向
提供长期位置约束
限制 2D strip 链累计漂移
```

ORB FPS 只根据“轨迹是否稳定”选择。

不再要求：

```text
ORB FPS 必须足够高以提供每一条 render strip。
```

如果 8 FPS 最稳定，就用 8 FPS；
如果 12 FPS 更稳定，就用 12 FPS；
16 FPS 已实测更差时，不再因为“需要更多 source”强行选 T2。

### B. RGB Render Strip

要求：

```text
必须是真实采集 RGB 帧
允许没有 direct-ORB pose
绝不能虚构 camera_to_world
必须位于两个真实 ORB Anchor 约束的短时间区间内
```

职责：

```text
提供最终 panorama RGB 像素
把相邻成图视角基线压小
解决宽近景 double-edge / ghost
```

正式 provenance 必须区分：

```text
pose_anchor_source = direct_orb

render_strip_placement =
anchor_constrained_rgb_2d_registration
```

禁止把 strip 的 2D placement 写成：

```text
direct_orb_pose
interpolated_orb_pose
```

---

# 3. 中央窄带规则

## 3.1 物体宽度不再控制 strip 宽度

删除 Production 路径中的：

```text
compact / wide / very-wide / oversized
→ 决定 geometry patch 数量
→ 决定 owner 扩宽
```

`N_req` 如果保留，只允许作为 diagnostic：

```text
object_span_diagnostic
```

不得影响：

```text
strip width
owner span
GraphCut corridor
source selection density
```

## 3.2 宽物体的正确行为

一个非常宽的纸箱：

```text
错误：

AAAAAAAAAAAAAAAAAAAAAAAA | BBBBBBBBBBBBBBB
<----- A 不断扩大 ------->


v6.1：

AAAA | BBBB | CCCC | DDDD | EEEE | FFFF
```

物体可以横跨：

```text
2 条
5 条
20 条
```

都没有问题。

硬原则：

> **物体多宽，只决定它跨多少条 strip；绝不能让单条 strip 跟着物体一起无限变宽。**

---

# 4. Strip 密度与宽度

## 4.1 第一版质量优先参数

从全部真实 60 FPS RGB 帧中根据累计水平推进选择 Render Strip。

第一版先采用：

```yaml
render_strip:
  minimum_advance_px: 3
  normal_target_advance_px: 8
  risk_target_advance_px: 5
  normal_hard_max_advance_px: 16
  emergency_hard_max_advance_px: 24
```

解释：

```text
普通区域：
约每推进 8 px 选一张真实 RGB

高风险近景：
约每推进 5 px 选一张

相邻 strip 推进 >16 px：
优先插入中间真实帧

>24 px：
除采集缺帧 emergency 外直接拒绝
```

这是第一版“先把 double-edge / ghost 清零”的参数。

质量通过后再做速度消融：

```text
8/5
→ 10/6
→ 12/6
→ 12/8
→ 16/8
```

选择满足所有视觉门的最稀配置。

不要一开始为了速度直接把 strip 放宽。

## 4.2 Strip output span 和 alignment corridor 必须分开

最终真正写入 panorama 的 owner strip 可以很窄：

```text
约 5–16 px
```

但做 DIS / edge / GraphCut 时必须看更宽的上下文：

```text
96–160 px corridor
```

即：

```text
用 96–160 px 看清楚两张图应该怎么对齐
                    ↓
最终只让中央 5–16 px 左右贡献到 panorama
```

禁止把：

```text
alignment corridor width
```

误认为：

```text
实际 owner strip width
```

---

# 5. Anchor-Constrained 2D Strip Placement

这是 v6.1 最关键的新模块。

建议新增：

```text
src/panorama_demo/video_anchor_strip_alignment.py
```

## 5.1 每个 ORB Anchor interval 独立求解

例如：

```text
Anchor A0
  |
  S1
  S2
  S3
  S4
  |
Anchor A1
```

只允许在：

```text
A0 ～ A1
```

这个短区间内累计 RGB strip placement。

到达 A1 后重新锁定，下一段重新开始：

```text
A1 ～ A2
```

禁止整个 3 m：

```text
S1 → S2 → ... → S200
```

纯 DIS 长链累计。

## 5.2 每对 strip 的观测

每个 final adjacent render-strip pair：

```text
Forward DIS × 1
Backward DIS × 1
```

输出：

```text
dx / dy
FB error
RGB residual
gradient residual
edge correspondence
occlusion-risk
confidence
```

只在：

```text
96–160 px corridor
```

计算。

## 5.3 2D 对齐模型优先级

每对 strip：

```text
identity
→ subpixel translation
→ translation + tiny rotation
→ restricted affine
→ bounded local mesh
```

第一版门限：

```yaml
pair_alignment:
  translation_target_px: 2.0
  translation_hard_px: 4.0

  rotation_target_deg: 0.75
  rotation_hard_deg: 1.5

  affine_scale_min: 0.97
  affine_scale_max: 1.03
  affine_shear_abs_max: 0.03

  mesh_displacement_target_px: 3.0
  mesh_displacement_hard_px: 6.0
  mesh_positive_jacobian: true
```

原因：

> 相邻 strip 已经非常密。如果还需要 8～10 px 大形变，说明这两个 source 之间太远，应该插入真实中间帧，而不是继续拉图。

---

# 6. Anchor Interval 全局约束

不能让每一对 DIS 独立累计，否则仍会漂移。

每个 ORB Anchor interval 内建立一个小型 2D constrained solve。

变量第一版只保留：

```text
x_i
y_i
theta_i
```

第一版不要做复杂全局非刚性 bundle。

目标：

```text
minimize:

Σ pairwise_DIS_residual
+
λ1 * anchor_endpoint_error
+
λ2 * vertical_drift
+
λ3 * rotation_drift
+
λ4 * second_difference_smoothness
```

硬约束：

```text
A0 位置固定
A1 位置由 direct-ORB + 已冻结 scan scale 约束
strip 顺序严格单调
禁止 reverse owner
```

如果 RGB 累积推进与 ORB Anchor endpoint 有小残差：

```text
只在当前 A0~A1 区间缓慢分配残差
```

禁止：

```text
修改 ORB Anchor pose
把 residual 一次性压到某条 seam
把 strip 2D placement 冒充成 6DoF pose
```

---

# 7. 自适应 Strip 加密

这是解决当前 blocker 的主恢复动作。

## 7.1 GraphCut 前先做 pre-seam quality check

一对 strip 在进入 GraphCut 前先检查：

```text
FB residual
edge mismatch
double-edge predictor
ghost predictor
line mismatch
```

建议初始门：

```yaml
pre_seam:
  fb_p95_target_px: 0.75
  fb_p95_hard_px: 1.25
  edge_normal_residual_p95_px: 0.75
  edge_normal_residual_abs_px: 1.25
```

如果 GraphCut 前已经出现明确：

```text
double-edge risk
ghost risk
```

则：

```text
禁止直接 GraphCut
```

## 7.2 失败恢复顺序

假设：

```text
A ---- B
```

失败。

第一恢复动作必须是：

```text
找到 A/B 之间真实 RGB Frame M
        ↓
A ---- M ---- B
```

重新计算：

```text
A-M F/B DIS
M-B F/B DIS
```

如果仍失败且还有真实中间帧：

```text
A - M1 - M2 - B
```

最多继续到“相邻真实采集帧”粒度。

原则：

> **double-edge / ghost 失败时优先缩短视角基线，而不是扩大 owner、扩大 warp、扩大 blend。**

## 7.3 中间帧选择

候选只允许：

```text
A/B 时间区间内真实采集 RGB 帧
```

评分：

```text
1. 与左右 pair 的累计推进最均衡
2. 清晰度高
3. 曝光差小
4. DIS FB 预测误差低
5. 无明显 motion blur
```

不要固定只取时间 midpoint。

---

# 8. 宽物体处理改为“连续性保护”

删除：

```text
整个大物体 single owner
一个大对象必须 2/3/4 个 geometry patch
```

改成：

> **对象可以跨任意数量窄 strip；保护的是边界连续性，而不是对象 owner 数量。**

每对局部 seam 建立：

```text
strong edge guard
long line guard
thin structure guard
occlusion-risk guard
```

这些 guard 全部由：

```text
RGB
DIS
gradient
line detector
```

生成。

Depth 不参与 v6.1 final renderer。

## 8.1 物体外轮廓

如果 seam 会穿：

```text
纸箱外边
风扇轮廓
线缆
横梁边缘
叶片轮廓
```

优先：

```text
reroute GraphCut seam
```

## 8.2 大物体内部

如果一个超宽物体占满整个 overlap，seam 无法绕到背景：

```text
允许 seam 穿物体内部
```

但前提必须是：

```text
两 strip 已经完成亚像素对齐
FB pass
edge residual pass
无明显遮挡转换
```

然后只做：

```text
2–4 px narrow blend
```

这比：

```text
为了不切物体把一张 source 一直加宽
```

更安全。

---

# 9. GraphCut

继续复用：

```text
src/panorama_demo/video_graphcut_seam.py
```

正式 solver：

```text
cv2.detail.GraphCutSeamFinder("COST_COLOR_GRAD")
```

GraphCut 只负责：

```text
从“已经对齐”的两个 strip 中找最安全交接线
```

GraphCut 不负责：

```text
修几何
消 double-edge
消 ghost
补 pose
```

正常 corridor：

```text
96–160 px
```

保留：

```text
480 px full height
row step <= 1 px
owner island = 0
small fragment = 0
```

GraphCut 失败恢复：

```text
先插入真实中间 RGB strip
→ 重新几何对齐
→ 再 GraphCut
→ 再 seam reroute
→ fail closed
```

不要首先扩到 192 px。

`192 px rescue corridor` 只保留为最后一次 seam 搜索扩展，且不得改变 actual owner strip width。

---

# 10. 融合

顺序必须：

```text
strip selection
→ DIS
→ 2D alignment
→ pre-seam geometry pass
→ GraphCut
→ hard guards
→ blend eligibility
→ narrow blend
```

初始融合宽度：

```yaml
blend:
  safe_background_px: [2, 6]
  aligned_object_interior_px: [2, 4]

  strong_edge_px: 0
  line_guard_px: 0
  thin_structure_px: 0
  occlusion_boundary_px: 0
```

禁止：

```text
double-edge 还存在
→ MultiBand 加到 10/20 px
```

---

# 11. 一次正式 RGB Sampling

正式像素生成必须仍然遵守：

```text
raw RGB → exactly one final sampling
```

实现方式：

1. 每个 strip 先只求：
   ```text
   final inverse grid
   ```
2. grid 由：
   ```text
   central crop geometry
   +
   anchor-constrained 2D correction
   +
   accepted local warp
   ```
   合并。
3. GraphCut 只产生：
   ```text
   owner labels
   ```
4. 最终从原始 RGB：
   ```text
   一次 bilinear/bicubic sampling
   ```
5. mask：
   ```text
   nearest-neighbour
   ```

禁止对 RGB 多次 remap。

---

# 12. RGB-D / Open3D 的新边界

## RGB-D

仅用于：

```text
ORB-SLAM3 direct anchor pose
```

最终 v6.1 render path 不读取：

```text
depth map
point cloud
TSDF
depth edge
depth connected object
```

当前以下旧模块可保留，但新 candidate 不调用：

```text
video_object_owner_lock.py 中的 depth owner logic
video_visual_renderer.py 中的 depth seam logic
旧 geometry-assisted RGB-D path
```

不要删除旧代码，以免破坏 baseline regression。

## Open3D

只保留：

```text
diagnostic
```

不得：

```text
生成 pose
替代 ORB
决定 strip placement
决定 seam
修改 final RGB
单独否决 RGB 已通过的 panorama
```

---

# 13. 代码修改建议

## 13.1 新增

建议新增：

```text
src/panorama_demo/video_anchor_strip_alignment.py
src/panorama_demo/video_render_strip_selection.py
```

### `video_render_strip_selection.py`

职责：

```text
从全部真实 RGB frames 中选择高密度 render strips
计算累计 RGB scan progress
risk-adaptive 8/5 px selection
在 failed pair 中插入真实中间帧
不读 pose
不生成 pose
不做 renderer
```

### `video_anchor_strip_alignment.py`

职责：

```text
接收：
ORB Anchor IDs / poses
Render Strip IDs
pair DIS evidence
scan scale

输出：
每个 strip 的 2D placement
每个 strip 的 inverse correction grid
anchor interval audit
```

禁止输出：

```text
camera_to_world
synthetic 6DoF pose
```

## 13.2 修改

主要修改：

```text
src/panorama_demo/video_panorama.py
src/panorama_demo/video_pipeline.py
src/panorama_demo/video_motion_resampler.py
src/panorama_demo/video_source_selection.py
src/panorama_demo/video_graphcut_seam.py
src/panorama_demo/video_safe_multiband.py
src/panorama_demo/video_visual_metrics.py
src/panorama_demo/video_offline_evaluation.py
src/panorama_demo/video_algorithm_selection.py
src/panorama_demo/video_production_freeze.py
src/panorama_demo/video_delivery.py
```

## 13.3 `video_source_selection.py`

当前动态 owner span 路线保留给历史 v6 candidate，但新 v6.1 不使用：

```text
plan_frontality_owner_spans()
```

决定最终 strip owner。

frontality 只允许作为：

```text
候选 RGB strip 的评分项
```

不允许：

```text
扩宽 owner
```

---

# 14. Candidate 设计

不要覆盖当前失败 candidate。

新增：

```text
V61_anchor_narrow_strip_t0
V61_anchor_narrow_strip_t1
```

T2：

```text
保留现有失败证据
默认不作为 survivor fallback
```

推荐：

```yaml
V61:
  pose_anchor:
    direct_orb_only: true
    candidates_fps: [8, 12]

  render_strip:
    real_rgb_only: true
    direct_orb_pose_required: false
    normal_target_advance_px: 8
    risk_target_advance_px: 5
    hard_max_advance_px: 16
    emergency_max_advance_px: 24

  alignment:
    backend: anchor_constrained_rgb_2d
    dis_forward_once_per_final_pair: true
    dis_backward_once_per_final_pair: true
    max_translation_px: 4
    max_rotation_deg: 1.5
    max_mesh_displacement_px: 6

  graphcut:
    corridor_px: [96, 160]
    rescue_corridor_max_px: 192
    full_height: 480

  depth_final_renderer: false
  open3d_final_gate: false
```

---

# 15. Provenance

每个最终有效像素仍然必须能追踪到：

```text
真实 capture RGB frame_id
```

建议：

```text
owner_frame_id
```

继续保存。

新增 strip report：

```text
render_strip_frame_id
left_anchor_frame_id
right_anchor_frame_id
placement_method = anchor_constrained_rgb_2d_registration
placement_dx
placement_dy
placement_theta
local_warp_enabled
```

必须明确：

```text
render strip 没有 direct ORB pose
```

时不得填 fake pose。

---

# 16. 质量硬门

当前 blocker：

```text
double-edge=0
ghost=0
```

**不要放宽。**

保留：

```text
double_edge_count = 0
ghost_count = 0
line_break_count = 0
```

建议 v6.1 质量门：

```yaml
quality:
  pre_seam_fb_p95_hard_px: 1.25
  pre_seam_edge_residual_p95_hard_px: 0.75
  pre_seam_edge_residual_abs_hard_px: 1.25

  final_line_step_p95_px: 0.75
  final_line_step_abs_px: 1.5

  double_edge_count_max: 0
  ghost_count_max: 0
  line_break_count_max: 0
  visible_wide_blur_count_max: 0
```

重点变化：

> **以前失败后调 GraphCut / owner / patch；v6.1 失败后优先增加真实 RGB strip 密度。**

---

# 17. 最小验证：先证明方向再全量改

不要立即重写整个 pipeline。

先在当前三数据中各选：

```text
1～2 个已确认 double-edge / ghost 的最困难 seam
```

做一个 isolated prototype：

```text
旧 pair：
A ---------------- B
FAIL

v6.1：
A -- S1 -- S2 -- S3 -- B
```

只实现：

```text
真实中间 RGB
+ F/B DIS
+ translation/affine
+ GraphCut
+ 2–4 px blend
```

不做全 panorama。

### POC Go 条件

三份数据的难 seam 中：

```text
double-edge 明显下降
ghost 明显下降
至少主要 blocker seam 达到 0
```

并且：

```text
不需要扩大 owner
不需要 Depth
不需要 T2
```

如果 POC 仍完全无改善，再考虑新采集。

如果 POC 通过，才进入全量架构改造。

---

# 18. 完整实施顺序

## Phase 0 — 冻结失败证据

保留：

```text
V6_rgb_only_graphcut
V6_rgb_only_graphcut_t2
```

当前 blocker report 不删除、不覆盖。

不生成 Production lock。

## Phase 1 — Blocker POC

实现最小：

```text
dense real RGB insert
pair DIS
bounded affine
GraphCut
narrow blend
```

只验证最难 seam。

通过才继续。

## Phase 2 — Pose Anchor / Render Strip 解耦

修改 pipeline：

```text
tracking_frames
!=
render_strip_frames
```

所有 ORB Anchor 必须 direct pose。

Render Strip 只要求真实 RGB。

## Phase 3 — Anchor-Constrained Placement

实现：

```text
A0~A1 interval solve
```

每个 interval 结束时由 ORB Anchor 拉回。

## Phase 4 — Adaptive Strip Densification

任何 pair pre-seam fail：

```text
插入真实中间帧
→重算 pair
```

直到：

```text
pass
```

或：

```text
已经相邻 capture frames仍失败
```

才 fail closed。

## Phase 5 — GraphCut / Guard / Blend

几何 pass 后才 GraphCut。

宽物体自然跨 strip。

删除 N_req 对 owner span 的控制。

## Phase 6 — 三份冻结数据全量

顺序：

```text
run_20260807_140140
→ run_20260804_162340
→ run_20260806_153033
```

同一 candidate lock。

不得单数据集调参。

## Phase 7 — 性能优化

质量通过后再优化：

```text
8/5
→10/6
→12/6
→12/8
→16/8
```

只允许通过降低不必要 strip 密度提速。

禁止通过：

```text
放宽 double-edge / ghost
扩大 blend
删除 full-res GraphCut
```

提速。

## Phase 8 — 文档 / Freeze

只有三份冻结数据同 candidate 通过后：

```text
更新 AGENTS.md
更新 README.md
```

写入新的正式契约：

```text
Direct-ORB Pose Anchor
+
Real RGB Render Strip
+
Anchor-Constrained 2D Placement
```

当前数据通过最多先设置：

```text
development_matrix_pass=true
current_dataset_candidate_pass=true
```

按当前项目生命周期要求决定是否进入 Production Freeze，不要因一次局部 POC 直接生成 lock。

---

# 19. 测试

新增：

```text
tests/test_video_render_strip_selection.py
tests/test_video_anchor_strip_alignment.py
```

扩展：

```text
tests/test_video_motion_resampler.py
tests/test_video_source_selection.py
tests/test_video_visual_renderer.py
tests/test_video_seam_solver_graphcut.py
tests/test_video_offline_evaluation.py
tests/test_video_algorithm_selection.py
tests/test_video_production_freeze.py
tests/test_calibrated_rgb_pushbroom.py
```

必须覆盖：

```text
1. ORB Anchor 全部 direct pose。
2. Render Strip 可以没有 ORB pose，但必须是真实 capture RGB。
3. Render Strip 绝不能伪造 camera_to_world。
4. N_req 不得影响 strip owner width。
5. 超宽物体可以跨任意数量 strip。
6. normal/risk strip advance 生效。
7. pair fail 后优先插真实中间帧。
8. 不允许扩大 owner 作为 double-edge recovery。
9. 每个 final pair 恰好一次 F/B DIS。
10. anchor interval 2D placement 不跨 interval 自由漂移。
11. anchor endpoint residual 有界。
12. pre-seam fail 不得进入 GraphCut。
13. GraphCut 仍是真实 OpenCV max-flow。
14. double-edge=0。
15. ghost=0。
16. line break=0。
17. protected edge blend=0。
18. final RGB only one sampling。
19. final renderer 不读取 depth。
20. Open3D 不参与 final gate。
21. legacy baseline 字节输出不变。
```

运行：

```text
pytest
ruff check .
python -m compileall src tests
git diff --check
```

---

# 20. 性能策略

高密度 strip 会增加 pair 数，因此必须控制计算范围。

必须：

```text
DIS 只算 selected render-strip pairs
只算 96–160 px corridor
不算全图 flow
```

可以：

```text
CPU 预取下一 pair RGB/gradient
DIS pair 串行或小并发
GraphCut 只在 geometry pass 后运行
resident strips <= 5
```

不要恢复：

```text
Open3D final-edge lattice
RAFT
SAM
全图 mesh
```

第一优先级：

```text
先达到质量
```

第二优先级：

```text
再逐步把 8/5 strip 变稀
```

---

# 21. No-Go

任一情况禁止称 v6.1 完成：

```text
继续让物体宽度直接扩大 owner span
N_req 决定 strip 宽度
为了 render source 数量继续强推失败的 T2
给无 ORB pose 的 render strip 伪造 pose
整段 3 m 串联 DIS 不做 anchor reset
pair geometry fail 后直接 GraphCut
double-edge fail 后扩大 MultiBand
ghost fail 后 blur
用 Depth / point cloud 精细补图
用 Open3D 替代 pose
用 DP 冒充 GraphCut
```

---

# 22. Go 条件

v6.1 当前数据 candidate 完成：

```text
三份冻结数据使用同一 candidate lock

double_edge_count = 0
ghost_count = 0
line_break_count = 0

所有 pose anchors = direct ORB
所有 render RGB = real capture frame
synthetic_render_pose_count = 0

N_req 不参与 strip width
动态 owner widening 禁用

所有 failed wide-baseline pair：
优先通过真实中间 strip densification 恢复

final RGB exactly one sampling
depth_final_renderer = false
open3d_final_gate = false

完整 regression / pytest / ruff / compileall 通过
```

---

# 23. Codex 直接执行指令

```text
不要继续调当前 V6_rgb_only_graphcut / T2 来尝试跨数据集碰运气。

先保留当前失败证据，然后建立新的 v6.1 candidate。

第一步不要全量重构：
先从三份冻结数据各取最困难的 double-edge/ghost seam，
实现“真实中间 RGB 窄带插入 + F/B DIS + bounded 2D alignment + GraphCut”
POC。

如果 POC 能把 blocker seam 消掉，再实现 Pose Anchor / Render Strip 解耦和
Anchor-Constrained interval placement。

核心契约：
ORB 只负责稀疏真实 Anchor；
60 FPS 真实 RGB 负责高密度窄带；
物体宽度永远不能扩大单 strip；
pair 几何失败优先增加真实中间 strip；
GraphCut 只处理已经对齐的 pair；
double-edge=0 / ghost=0 不放宽。
```

---

# 24. 最终结论

v6 的失败恢复方向从：

```text
宽物体
→扩大 owner / 增加大 patch
→提高 ORB tracking FPS
→在更远 source 之间找 seam
```

改为：

```text
宽物体
→保持中央窄带硬上限
→使用更多真实 RGB 窄带
→每对窄带先亚像素对齐
→ORB Anchor 定期约束累计漂移
→几何通过后 GraphCut
→最后极窄融合
```

这次修改针对的是当前 `double-edge=0 / ghost=0` 无法跨三数据同时满足的结构性原因，而不是继续调某一组阈值。
