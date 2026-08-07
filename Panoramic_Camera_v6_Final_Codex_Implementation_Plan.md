# Panoramic_Camera v6 最终版 — Codex 直接实施方案

## 0. 目标与边界


## v6 最终门限策略

本版采用“**质量优先，但不过度 fail-closed**”的放宽门限。原则：

```text
1. 直线、双边、ghost、明显阶梯仍是硬失败。
2. DIS、近景小修正、正视角、亮度、局部 mesh 适度放宽。
3. 对超过 target 但未超过 hard 的结果允许继续出图，并在 report 标记 warning。
4. 超过 hard 才进入 reroute / reselect source / rescue / fail-closed。
5. 禁止通过大范围 blur、超宽 MultiBand 或自由 flow warp 掩盖几何错误。
```


仓库：

```text
D:\central_strip_Panoramic_Camera
```

实现基线：

```text
codex/dense-anchor-recovery@8b4107bad115f54cd267937cac645e827754464d
```

建议新分支：

```text
codex/video-realtime-seam-v6
```

当前只实施和验证 **3 m**。不要实现 20 m。

正式目标：

```text
输入：848×480 @ 60 FPS aligned RGB-D
RGB-D：只给 ORB-SLAM3 求 direct pose
最终成图：RGB-only
最大正式 post：60 s
内部 realtime 目标：20 s
优先目标：FAST P50<=5 s，P95<=8 s
```

最终图像必须满足：

```text
接缝基本不可见
长直线无像素台阶
无双边/重影
宽近景尽量正视
近景内部无明显 hard-cut 色缝
不能靠大范围模糊隐藏几何错误
```

---

# 1. 不可违反的硬约束

```text
1. 所有正式 render source 必须是真实 RGB 帧。
2. 所有正式 source pose 必须是 direct ORB camera_to_world。
3. 禁止 interpolated render pose。
4. Depth / point cloud / TSDF 不参与最终 RGB 精细 warp、补图、seam 或 blend。
5. Open3D 不再是 Production 发布硬门，只允许 diagnostic。
6. 每个 final pair 只允许 1 次 Forward DIS + 1 次 Backward DIS。
7. DIS 不允许直接逐像素自由 warp 最终 RGB。
8. GraphCut 必须是真实 binary max-flow，不允许 DP 冒充。
9. GraphCut 在 480 px 高、96–160 px corridor 内一次求解。
10. 最终 RGB 只允许一次 full-resolution inverse sampling。
11. 不允许扩大 MultiBand 来掩盖几何错位。
12. 不允许重新退化成大量固定中央窄条切碎近景物体。
```

正式路径禁止：

```text
RAFT
SAM
Torch / torchvision / Kornia
TSDF RGB 补图
RGB-D 3D reprojection
Open3D pose replacement
全景级 homography
插值 pose
DP seam
大范围 feather / blur
```

---

# 2. Phase 1 — 先恢复完整 Direct-ORB Tracking

在：

```text
g305-video-experiment
```

实现并比较：

```text
T0 = 8 FPS direct ORB
T1 = 12 FPS direct ORB
T2 = 16 FPS direct ORB
```

选择规则：

```text
完整 connected trajectory
→ 所有正式 source 都有 direct pose
→ 满足 frontality coverage
→ 通过质量门
→ 选最低 tracking FPS
```

禁止：

```text
T0 丢轨后插值补 pose
DIS 补 pose
Open3D 补 pose
```

需要新增报告字段：

```text
tracking_candidate_id
tracking_fps
direct_orb_pose_count
direct_orb_pose_coverage
full_direct_orb_chain_available
frontality_coverage_pass
```

---

# 3. Phase 2 — 重写 Source Selection：中央正视优先

修改：

```text
video_motion_resampler.py
video_source_selection.py
```

保留现有 full-resolution dx resampler，但 source 选择加入：

```text
coverage
frontality
sharpness
DIS/common-support quality
risk
```

每张 source 的实际 owner span **不设固定像素宽度**。

使用：

```text
off_axis_angle = atan((x-cx)/fx)
```

最终 Development 门限：

```text
近景：
target <= 4°
hard <= 7°

普通背景：
target <= 6°
hard <= 10°
```

从主点 `cx` 向左右扩展：

```text
<= target → 正常使用
target ~ hard → 允许继续出图，但记录 frontality_warning
> hard → 停止扩大当前 owner span，换下一张 direct-ORB source
```

新增：

```text
frontality_score
valid_frontality_span
```

Tracking gate 除了检查 ORB 是否完整，还要检查 anchor 是否足够密，能否避免单帧 owner span 被迫过宽。

保留：

```text
base_source_hard_maximum <= 44
rescue_per_seam <= 1
rescue_session <= 4
final_sources <= 48
resident_sources <= 5
```

---

# 4. Phase 3 — DIS 成为 RGB 亚像素几何主证据

每个最终相邻 source pair：

```text
Forward DIS ×1
Backward DIS ×1
```

只在：

```text
96–160 px seam corridor
```

计算。

统一生成：

```text
flow_forward
flow_backward
fb_error
rgb_residual
gradient_residual
occlusion_risk_mask
correspondence_confidence
```

F/B 不一致、高 RGB/gradient 冲突区域：

```text
禁止 blend
禁止自由 warp
提高 seam cost
必要时 reroute / rescue
```

DIS 只能生成 correspondence evidence，禁止：

```text
final_rgb = remap(final_rgb, raw_dis_flow)
```

---

# 5. Phase 4 — RGB-only 局部几何修正

## 5.1 背景

优先级：

```text
identity
→ translation / small affine
→ bounded 2D mesh
```

mesh 仅在 DIS/RGB evidence 足够时启用。

门限：

```text
displacement target <= 6 px
displacement hard <= 10 px
positive Jacobian
outer boundary displacement = 0
line guard intersection = 0

held-out FB P95:
target <= 1.25 px
hard <= 2.0 px
```

`6~10 px` 允许继续，但必须记录 `large_alignment_warning=true`，且仍需通过直线/ghost/GraphCut 质量门。

## 5.2 近景

新增：

```text
near_protected_alignment
```

优先级：

```text
identity
→ subpixel translation
→ small rotation
→ small affine
→ restricted homography（仅可靠近平面）
```

最终门限：

```text
translation:
  target <= 3 px
  hard <= 6 px

rotation:
  target <= 1.5°
  hard <= 3°

affine scale:
  [0.95, 1.05]

anisotropic scale ratio:
  <= 1.05

|shear|:
  <= 0.05
```

允许 homography 的对象：

```text
纸箱正面
柜门
平板
```

禁止强制 homography 的对象：

```text
风扇
叶片
果串
软管
线缆
复杂非平面结构
```

所有模型必须用部分 DIS 对应拟合、另一部分 held-out 验证。

restricted homography 仅在 affine 失败后启用，最终门限：

```text
corner displacement hard <= 6 px
local scale in [0.94, 1.06]
line orientation change <= 1.5°
held-out FB P95 <= 1.0 px
held-out FB absolute max <= 2.0 px
```

若超过 hard、直线被破坏或 distortion 明显：

```text
拒绝该 warp
→ 换 source / reroute seam / rescue
```

---

# 6. Phase 5 — 物体保护与宽近景

## 6.1 Object Context Collar

对象保护从：

```text
object mask
```

升级为：

```text
object mask + context collar
```

development 初值：

```text
大对象：8–20 px
普通对象：6–16 px
细结构：4–10 px
```

这些值后续由当前数据冻结。

## 6.2 宽窄定义

不要用固定像素阈值。

定义：

```text
N_req = 保持正视、清晰、RGB 对齐时，
        完整覆盖该物体+保护上下文所需的最少 direct-ORB source 数
```

分类：

```text
N_req=1 → compact
N_req=2 → wide
N_req=3 → very-wide
N_req>3 → oversized
```

## 6.3 N_req>3 的最新规则

**不要直接失败，也不要规定最多只能 3 个 patch。**

处理顺序：

```text
重新选择更靠中央的 direct-ORB source
→ 若 anchor 太稀，尝试 T1/T2 更高 tracking FPS
→ 重新计算 final_replanned_N_req
→ 如果仍 >3，且确实因为物体本身很宽：
   允许使用 final_replanned_N_req 个连续大 patch
```

允许：

```text
AAAAAAAAAA | BBBBBBBBBB | CCCCCCCCCC | DDDDDDDDDD
```

禁止：

```text
A|B|C|D|E|F|G|H|I
```

硬门：

```text
geometry_patch_count <= final_replanned_N_req + approved_rescue_patch_count
redundant_geometry_patch_count = 0
small_fragment_count = 0
patch_island_count = 0
```

所有内部 handoff：

```text
必须位于物体内部安全区域
不得穿外轮廓
不得穿细线
不得穿高 occlusion-risk
必须独立通过 DIS/RGB/line/ghost/seam gate
```

---

# 7. Phase 6 — 真 GraphCut

新增文件：

```text
src/panorama_demo/video_graphcut_seam.py
```

仅包装：

```text
cv2.detail_GraphCutSeamFinder
```

职责只包括：

```text
输入 cost / hard masks
运行真实 binary GraphCut / max-flow
输出 labels
输出 seam_x_by_row
输出 topology audit
```

禁止读取 pose、选 source、搭 canvas、做 compositor。

GraphCut 正常直接在：

```text
480 × 96–160 px corridor
```

运行一次。

若正常 corridor 找不到安全 seam，可进行 **一次 rescue 扩宽**：

```text
maximum rescue corridor = 192 px
```

只能用于 seam reroute，不能因此扩大最终 RGB blend 宽度。

Hard protection：

```text
line_guard
object_outer_boundary
thin_structure
occlusion_risk
invalid_support
```

审计：

```text
adjacent row seam step <= 1 px
owner island = 0
small fragment = 0
valid pixel exactly one geometry owner
```

恢复顺序：

```text
reroute seam
→ expand one real source owner
→ insert one direct-ORB rescue source
→ fail closed
```

禁止：

```text
DP smooth seam
扩大 blur
扩大 MultiBand
降低分辨率
```

---

# 8. Phase 7 — 近景“边缘硬、内部软”

新增：

```text
near_blend_eligible_mask
```

只有全部满足才允许近景 blend：

```text
local FB error:
  target <= 0.75 px
  hard <= 1.25 px

RGB correspondence pass
geometry residual pass
both source pixels valid
not occlusion/disocclusion
not object boundary
not line guard
not thin structure
```

融合宽度：

```text
普通安全背景：2–10 px
近景安全内部：2–8 px
强纹理近景内部：2–6 px

object boundary：0
line guard：0
occlusion boundary：0
cable/thin structure：0
```

`>8 px` 的背景 blend 仅允许在低梯度、无强结构、无 ghost 风险区域；近景 hard maximum 仍为 8 px。

原则：

```text
先几何对齐
→ 再 GraphCut
→ 再生成 blend mask
→ 最后窄 MultiBand
```

严禁：

```text
对不齐 → 扩大 MultiBand
```

---

# 9. Phase 8 — Photometric + 一次正式 RGB Sampling

复用：

```text
video_photometric.py
video_safe_multiband.py
```

Photometric：

```text
pair gain [0.82, 1.22]
pair |bias| <= 0.12 linear-light
global |bias| <= 0.20

DeltaE00 P95:
target <= 2.5
hard <= 4.0
```

将：

```text
calibration grid
+ background correction
+ near correction
+ final geometry owner/seam
```

组合成一个最终 inverse grid。

然后：

```text
raw RGB
→ exactly one full-resolution sampling
```

RGB：

```text
bilinear / bicubic
```

mask：

```text
nearest-neighbour
```

禁止第二次正式 remap。

---

# 10. Open3D 处理

Open3D 从正式路径删除硬依赖。

保留时仅：

```text
diagnostic / audit
```

Open3D：

```text
不能修改 pose
不能修改 RGB warp
不能决定 seam
不能单独否决通过 RGB gate 的 panorama
```

删除/取消正式要求：

```text
all final O3D edges Tensor CUDA pass
Open3D candidate-edge lattice production requirement
RGB-D residual hard gate
depth-edge crossing hard gate
```

---

# 11. Quality Gates

## Geometry / DIS

```text
held-out FB P95:
target <= 1.25 px
hard <= 2.0 px

safe seam residual:
P95 target <= 0.9 px
P95 hard <= 1.25 px
absolute max <= 1.5 px
```

## Anti-Staircase

```text
long-line perpendicular step:
P95 <= 0.75 px
absolute max <= 1.5 px

orientation:
P95 <= 1.5°
absolute max <= 3°

>1.5 px discrete step event = 0
continuous >=1 px staircase run >5 px = 0
line break = 0
double edge = 0
ghost = 0
line_guard ∩ warp = 0
line_guard ∩ blend = 0
```

说明：直线门已适度放宽，但仍禁止肉眼明显的断线、双边、ghost 和连续阶梯。

## Object

```text
compact:
  geometry_patch_count = 1

wide:
  geometry_patch_count = 2

very-wide:
  geometry_patch_count = 3

oversized:
  先 replan / higher tracking
  然后 geometry_patch_count <= final_replanned_N_req + approved_rescue_patch_count

redundant_geometry_patch_count = 0
small_fragment_count = 0
patch_island_count = 0
outer_contour_handoff = 0
thin_structure_blend = 0
```

## Visual

```text
DeltaE00 P95:
target <= 2.5
hard <= 4.0

brightness step:
P95 <= 1.5%
absolute max <= 3.0%

gradient jump:
target <= 1.20
hard <= 1.35

visible vertical band = 0
ghost = 0
wide blur = 0
```

brightness/DeltaE 只在低梯度、共同有效、无强纹理的 photometric evaluation mask 上统计。

---

# 12. 当前数据验证顺序

固定数据：

```text
FAST_PRIMARY_DEVELOPMENT
D:\central_strip_Panoramic_Camera\data\captures\video\run_20260807_140140

FAST_PRESSURE_REGRESSION
D:\central_strip_Panoramic_Camera\data\captures\video\run_20260804_162340

SLOW_DEVELOPMENT_CONTROL
D:\central_strip_Panoramic_Camera\data\captures\video\run_20260806_153033
```

顺序：

```text
1. 140140：比较 T0/T1/T2，选 survivor
2. survivor 跑完整 v6 quality gates
3. survivor 跑 162340 pressure regression
4. survivor 跑 153033 slow veto
5. 同一 candidate lock 做 timing
```

当前阶段允许：

```text
development_matrix_pass=true
current_dataset_candidate_pass=true
```

当前阶段不要：

```text
production_3m_pass=true
production_3m_realtime_pass=true
twenty_metre_validated=true
```

---

# 13. 性能目标

正式产品：

```text
maximum_post_seconds = 60
```

内部 realtime：

```text
FAST：
2 warm-up + 10 formal
P50 target <= 6 s
P95 target <= 10 s
10/10 hard <= 20 s

OLD FAST：
1 warm-up + 5 formal
5/5 hard <= 20 s

SLOW：
1 warm-up + 5 formal
5/5 hard <= 20 s
```

`P95 8 s` 仍作为优化目标，但不再作为 current-dataset candidate 的硬失败线；**20 s 才是 realtime candidate hard limit**。

如果 P95 > 10 s：

```text
先 profile
→ 优化 DIS corridor / source count / sampling / GraphCut / CPU-GPU transfer
```

若 `8 < P95 <= 10 s`，允许 current-dataset candidate 继续，但标记 `realtime_optimization_warning=true`。

不要首先：

```text
降分辨率
降低质量门
扩大 blend
引入 RAFT
```

---

# 14. 主要修改文件

优先修改：

```text
src/panorama_demo/video_pipeline.py
src/panorama_demo/video_panorama.py
src/panorama_demo/video_motion_resampler.py
src/panorama_demo/video_source_selection.py
src/panorama_demo/video_visual_renderer.py
src/panorama_demo/video_object_owner_lock.py
src/panorama_demo/video_photometric.py
src/panorama_demo/video_safe_multiband.py
src/panorama_demo/geometry_assisted_local_warp.py
src/panorama_demo/video_visual_metrics.py
src/panorama_demo/video_offline_evaluation.py
src/panorama_demo/video_algorithm_selection.py
src/panorama_demo/video_production_freeze.py
src/panorama_demo/video_performance.py
src/panorama_demo/video_delivery.py
configs/video_candidates/<v6>.yaml
configs/video_candidates/candidate_manifest.json
```

新增最多：

```text
src/panorama_demo/video_graphcut_seam.py
```

Open3D 相关代码先不删除，改为 diagnostic-only，避免大范围无关重构。

Legacy baseline：

```text
legacy_fast_b07b561
```

字节输出必须保持不变。

---

# 15. 必须新增/修改的测试

```text
tests/test_video_motion_resampler.py
tests/test_video_visual_renderer.py
tests/test_video_offline_evaluation.py
tests/test_video_algorithm_selection.py
tests/test_video_production_freeze.py
tests/test_video_performance.py
```

新增：

```text
tests/test_video_seam_solver_graphcut.py
```

至少覆盖：

```text
1. T0/T1/T2 所有正式 source 都是 direct ORB。
2. 插值 pose 立即失败。
3. owner span 根据 frontality 动态变化。
4. 每 final pair 只调用一次 F/B DIS。
5. depth / Open3D 不进入 final RGB renderer。
6. DIS flow 不直接逐像素自由 warp final RGB。
7. near identity→translation→rotation→affine→restricted homography 顺序正确。
8. homography 对非平面/held-out fail 时拒绝。
9. N_req 分类正确。
10. N_req>3 必须先 replan / higher tracking。
11. oversized 可以 >3 大 patch，但 redundant/small/island=0。
12. GraphCut 是真实 cv2 GraphCut，不允许 DP fallback。
13. seam row-step<=1。
14. long-line staircase / double-edge / ghost gate。
15. near blend mask 不进入 line/contour/thin/occlusion。
16. 一次正式 full-resolution RGB sampling。
17. legacy baseline 字节回归不变。
18. current_dataset_candidate_pass 不得生成 Production lock。
```

实现后运行：

```text
pytest
ruff check .
python -m compileall src tests
git diff --check
```

---

# 16. Codex 执行顺序

严格按下面顺序，不要同时铺开所有模块：

```text
Step 1  建分支，锁 baseline regression
Step 2  完成 T0/T1/T2 direct-ORB tracking gate
Step 3  完成 dynamic frontality source selection
Step 4  完成 final-pair F/B DIS evidence
Step 5  完成 RGB-only background / near alignment
Step 6  完成 N_req + wide/oversized patch planning
Step 7  接入真实 GraphCut
Step 8  接入 line/object/occlusion hard guards
Step 9  接入 near_blend_eligible_mask + narrow MultiBand
Step 10 合成一次 final sampling grid
Step 11 重写 RGB-only quality gates
Step 12 跑 140140 → 162340 → 153033
Step 13 profile 并优化到 P95<=8 s / all<=20 s
Step 14 完整 pytest / ruff / compileall / diff-check
```

每完成一个 Step：

```text
提交代码
运行对应测试
输出：
- 改了哪些文件
- 实际算法路径
- 测试结果
- 当前 blocker
- 下一步
```

不要在某一步失败时静默降级到 legacy DP、depth warp、宽 blur 或插值 pose。

---


# 17. 最终冻结的 v6 Development 门限

```yaml
v6_thresholds:
  dis:
    held_out_fb_p95_target_px: 1.25
    held_out_fb_p95_hard_px: 2.0

  seam:
    residual_p95_target_px: 0.9
    residual_p95_hard_px: 1.25
    residual_abs_max_px: 1.5
    corridor_normal_px: [96, 160]
    corridor_rescue_max_px: 192

  line:
    step_p95_max_px: 0.75
    step_abs_max_px: 1.5
    orientation_p95_max_deg: 1.5
    orientation_abs_max_deg: 3.0
    discrete_step_over_1p5px_max_count: 0
    staircase_run_ge_1px_over_5px_max_count: 0
    break_max_count: 0
    double_edge_max_count: 0
    ghost_max_count: 0

  frontality:
    near_target_deg: 4.0
    near_hard_deg: 7.0
    general_target_deg: 6.0
    general_hard_deg: 10.0

  near_alignment:
    translation_target_px: 3.0
    translation_hard_px: 6.0
    rotation_target_deg: 1.5
    rotation_hard_deg: 3.0

  near_affine:
    scale_min: 0.95
    scale_max: 1.05
    anisotropic_scale_ratio_max: 1.05
    shear_abs_max: 0.05

  near_homography:
    default_enabled: false
    enable_only_after_affine_failure: true
    corner_displacement_hard_px: 6.0
    local_scale_min: 0.94
    local_scale_max: 1.06
    line_orientation_change_max_deg: 1.5
    held_out_fb_p95_max_px: 1.0
    held_out_fb_abs_max_px: 2.0

  background_alignment:
    displacement_target_px: 6.0
    displacement_hard_px: 10.0

  near_blend:
    local_fb_target_px: 0.75
    local_fb_hard_px: 1.25
    width_px: [2, 8]
    strong_texture_width_px: [2, 6]

  background_blend:
    width_px: [2, 10]

  protected_blend:
    object_boundary_px: 0
    line_guard_px: 0
    occlusion_boundary_px: 0
    thin_structure_px: 0

  photometric:
    pair_gain_min: 0.82
    pair_gain_max: 1.22
    pair_bias_abs_max_linear: 0.12
    global_bias_abs_max_linear: 0.20
    delta_e00_p95_target: 2.5
    delta_e00_p95_hard: 4.0
    brightness_step_p95_max_percent: 1.5
    brightness_step_abs_max_percent: 3.0
    gradient_jump_target: 1.20
    gradient_jump_hard: 1.35

  realtime:
    fast_p50_target_seconds: 6.0
    fast_p95_target_seconds: 10.0
    fast_p95_optimization_goal_seconds: 8.0
    per_run_hard_seconds: 20.0
    product_maximum_post_seconds: 60.0
```

# 18. 完成定义

v6 development 完成必须同时满足：

```text
完整 direct-ORB chain
动态正视 source selection 生效
最终 renderer 无 depth / point cloud 精细成图依赖
DIS F/B 每 pair 一次
RGB-only 局部对齐生效
宽近景按最小必要 N_req 规划，无碎片化
真实 480px GraphCut 生效
长直线零阶梯门通过
近景“边缘硬、内部软”生效
final RGB exactly one sampling
三份当前数据同 candidate 通过
FAST P50<=6 s / P95<=10 s / all<=20 s（8 s 仍为优化目标）
全量测试通过
```

达到以上条件后只设置：

```text
development_matrix_pass=true
current_dataset_candidate_pass=true
```

不要在当前阶段生成公共 Production lock。
