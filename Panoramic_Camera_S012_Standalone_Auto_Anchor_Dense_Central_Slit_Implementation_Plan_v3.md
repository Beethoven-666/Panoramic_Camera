# Panoramic Camera S1.2 独立自动锚点密集中央狭缝实施方案 v3

## 0. 方案定位

本文件取代此前所有依赖冻结 S1 source center、S1 canvas 或 S1 baseline hash 的 S1.2 方案。

最新算法名称：

```text
S012_standalone_auto_anchor_dense_central_slit_v1
```

最新核心路线：

> **从原始视频、相机标定和 direct ORB-SLAM3 位姿出发，自动识别有效扫描段，自动选择 pose 合格关键帧作为锚点，使用可靠的相邻和跨帧图像运动建立运动图，求出所有渲染 source 的全景横向位置，再使用固定 Voronoi 列分配生成密集中央狭缝全景。**

S1 不再是：

- 输入
- baseline
- anchor 来源
- canvas 来源
- 轨迹校准来源
- 验收必要条件
- 实验必要条件

代码、配置和 CLI 中不得出现 `--baseline S1`。

现有 S0、S1 和 S1.1 只能作为仓库中已有的独立旧实验保留，不参与 S1.2 的计算。

---

# 1. 最终输入和输出

## 1.1 必需输入

S1.2 正式运行只需要：

```text
连续 RGB 或 RGB-D 视频会话
相机标定
每帧时间戳
direct ORB-SLAM3 camera_to_world 位姿
```

深度可以用于 Open3D 资格审计，但不得用于：

- 生成 RGB
- 补齐 RGB
- TSDF
- 点云回投
- 近景精细补图
- 决定某个物体的纹理来源

## 1.2 正式 source 资格

每一个实际贡献非零宽度条带的 source 都必须满足：

```text
RGB 文件存在
尺寸与标定一致
有 direct ORB-SLAM3 pose
pose 时间戳与 frame 对应
pose origin 和坐标约定一致
camera_to_world 为有限有效 SE(3)
轨道方向和姿态审计通过
与最终相邻 source 的 Open3D 边审计通过
相邻或跨帧图像运动证据可靠
```

没有 direct pose 的帧可以存在于原始视频中，但不得成为正式渲染 source。

禁止：

- 插值 pose
- 外推 pose
- 复制邻帧 pose
- 用二维 dx/dy 伪造 pose
- 用 Open3D odometry 替代缺失的 ORB pose

## 1.3 输出

算法独立计算：

```text
有效扫描段
自动 anchor
最终 source 集
所有 source 的全景中心
最终 canvas 宽度
固定 Voronoi 列边界
Stage A 全景
Stage B 全景
像素级 provenance
全景 held-out track 审计
质量和性能报告
```

最终 canvas 不要求与 S1 相同。

---

# 2. 这条路线如何具有通用性

S1.2 不需要提前知道：

```text
全景应该有多宽
某一帧应该放在 x=多少
需要多少 source
哪个物体跨了多少 pair
哪个 source 应该拥有箱子
```

它自动完成：

```text
ORB pose 判断相机走向和帧资格
↓
标定域图像运动测量像素推进
↓
多帧运动图抑制累计误差
↓
自动选择可靠关键帧作为 anchor
↓
全局求所有 source center
↓
根据中心中点固定分配输出列
↓
中央狭缝渲染
```

这里的 anchor 不是预先知道绝对 x 的外部路标。

Anchor 是算法自动选择的高可信关键帧。它们用于建立可靠的长距离图像运动边，限制单链累计漂移。

只有第一个 source center 固定为坐标原点。其他 source center 全部由运动图求出。

---

# 3. 不再采用的旧设计

以下设计全部禁止：

1. 读取 S1 source center。
2. 读取 S1 canvas 宽度。
3. 使用 S1 source 作为硬 anchor。
4. 对水平 advance 做统一中值滤波。
5. 将整条水平轨迹按首尾宽度统一拉伸。
6. 对不可靠运动边线性插值。
7. 对首尾运动外推。
8. 使用 `max(0, dx)` 掩盖错误运动。
9. 让一个宽物体装进某一个 pair。
10. 使用对象 owner。
11. 使用可移动 pair owner。
12. 使用 S1.1 直线边界 DP。
13. 插入 9 到 12 像素的 owner sliver。
14. 对每个 pair 独立做窄肩纵向 warp。
15. 把每 source 的纵向参数插值为跨 source 共享的 `v_canvas[x]`。
16. 第一版使用颜色增益、羽化或 MultiBand。
17. 第一版生成虚拟中间 RGB 帧。

---

# 4. 代码隔离要求

仓库：

```text
Beethoven-666/Panoramic_Camera
```

开发分支：

```text
codex/video-realtime-seam-v6
```

必须新增独立实验路线。

不得修改：

```text
production renderer
production lock
现有 S0/S1/S1.1 的算法行为
现有 S0/S1/S1.1 的冻结产物
```

建议新增：

```text
src/panorama_demo/video_s12_config.py
src/panorama_demo/video_s12_session.py
src/panorama_demo/video_s12_pose.py
src/panorama_demo/video_s12_scan.py
src/panorama_demo/video_s12_motion_graph.py
src/panorama_demo/video_s12_anchor.py
src/panorama_demo/video_s12_schedule.py
src/panorama_demo/video_s12_tracks.py
src/panorama_demo/video_s12_vertical.py
src/panorama_demo/video_s12_renderer.py
src/panorama_demo/video_s12_metrics.py
src/panorama_demo/video_s12_experiment.py
```

新增配置：

```text
configs/video_candidates/S012_standalone_auto_anchor_dense_central_slit_v1.yaml
configs/video_candidates/S012_validation_regions_v1.json
```

新增测试：

```text
tests/test_video_s12_config.py
tests/test_video_s12_pose.py
tests/test_video_s12_scan.py
tests/test_video_s12_motion_graph.py
tests/test_video_s12_anchor.py
tests/test_video_s12_schedule.py
tests/test_video_s12_tracks.py
tests/test_video_s12_vertical.py
tests/test_video_s12_renderer.py
tests/test_video_s12_metrics.py
tests/test_video_s12_experiment.py
tests/test_video_s12_real_data_regressions.py
```

修改 `pyproject.toml`：

```toml
g305-video-s12-experiment = "panorama_demo.video_s12_experiment:main"
```

---

# 5. 分两次实施

## 第一次交付

Codex 只实现：

```text
M0 独立 session 和 trajectory 读取
M1 pose 资格与扫描段识别
M2 图像运动图
M3 自动 anchor
M4 全局水平 source center
M5 最终 source 选择和固定 Voronoi schedule
M6 无纵向、无颜色、无融合的 Stage A
M7 像素级 provenance
M8 Stage A 审计和锁定
```

完成 Stage A 后停止。

不得提前实现 Stage B。

## 第二次交付

Stage A 经确认后实现：

```text
M9 完整 feature track
M10 solver 和 held-out 整轨迹分离
M11 每 source 全局纵向参数 t_i
M12 Stage B
M13 held-out 横线和 track cluster 验收
M14 性能优化
```

---

# 6. Structural fatal 和 Quality fail

## 6.1 Structural fatal

以下情况不能发布新的正式完整 bundle：

- trajectory 文件缺失或格式不支持
- pose origin 混用
- source 的 direct pose 缺失
- pose 与 frame 时间对应错误
- pose 不是有效 SE(3)
- 自动扫描段长度不足
- 图像运动图不连通
- anchor 之间没有可靠运动约束
- 相邻最终 source 的 Open3D 边失败
- 普通内部条带超过 24 px，且没有更多 pose 合格真实帧可加入
- 同一输出列分配给两个 source
- 内部输出列没有 source
- 同一像素被写入两次
- `valid_mask` 与 `owner_frame_id` 不一致
- 原子发布失败
- 配置 schema 不合法

Structural fatal 时：

```text
CLI 返回非零
不覆盖旧 bundle
删除未完成 staging
写独立 failure.json
```

## 6.2 Quality fail

以下情况可以保留完整审计图：

- 箱子仍有局部切碎
- 横线仍有台阶
- held-out track 出现重复 cluster
- projected span 超过 2 px
- 可评估 track 数量不足
- 空间覆盖不足
- 处理时间未达到目标

报告必须：

```text
structural_valid = true
acceptance.pass = false
production_eligible = false
```

---

# 7. 配置初稿

```yaml
schema: gemini305-video-s12-standalone/v1
algorithm_id: S012_standalone_auto_anchor_dense_central_slit_v1

diagnostic_only: true
production_eligible: false

trajectory:
  require_direct_orb_pose: true
  allow_interpolated_pose: false
  allow_extrapolated_pose: false
  require_uniform_pose_origin: true
  maximum_timestamp_delta_ms: 5.0
  audit_rotation: true
  maximum_relative_roll_deg: 1.0
  maximum_relative_pitch_deg: 1.0
  maximum_relative_yaw_deg: 2.0
  maximum_source_pruning_rounds: 2

scan:
  minimum_pose_count: 16
  minimum_scan_displacement_m: 0.50
  rail_axis_ransac_iterations: 128
  rail_axis_inlier_threshold_m: 0.02
  maximum_cross_track_error_m: 0.03
  maximum_backward_step_m: 0.002
  maximum_pause_span_frames: 30
  require_image_motion_direction_agreement: true

motion_graph:
  analysis_width_px: 424
  model: calibrated_robust_translation
  adjacent_candidate_steps: [1]
  skip_candidate_steps: [2, 4]
  maximum_direct_edge_frame_gap: 64
  lk_forward_backward_max_px: 0.75
  ransac_residual_px: 1.5
  minimum_inlier_count: 16
  minimum_inlier_ratio: 0.45
  minimum_grid_coverage: 0.25
  minimum_vertical_span_fraction: 0.40
  maximum_parallax_cluster_ratio: 0.45
  require_pose_direction_agreement: true
  preserve_reliable_measurements: true
  apply_median_filter: false
  interpolate_unreliable_edges: false
  extrapolate_edges: false

anchors:
  automatic: true
  target_spacing_px: 96
  minimum_spacing_px: 64
  maximum_spacing_px: 144
  target_search_radius_px: 24
  require_direct_anchor_edge: true
  require_two_independent_graph_paths: true
  minimum_anchor_edge_confidence: 0.70
  minimum_anchor_image_quality: 0.50
  preserve_first_and_last_scan_source: true

horizontal_solver:
  parameterization: positive_adjacent_increments
  gauge_first_center_at_calibrated_cx: true
  huber_delta_px: 1.5
  irls_iterations: 4
  minimum_increment_px: 0.20
  horizontal_smoothing_weight: 0.0
  use_adjacent_edges: true
  use_skip_edges: true
  use_anchor_edges: true
  maximum_reliable_edge_adjustment_px: 2.0
  require_strictly_increasing_centers: true

source_selection:
  start_from_all_pose_qualified_frames: true
  duplicate_center_tolerance_px: 0.50
  preferred_internal_strip_width_px: 16
  hard_internal_strip_width_px: 24
  maximum_refinement_rounds: 2
  allow_additional_pose_qualified_real_frames: true
  permit_virtual_rgb_source: false
  preserve_automatic_anchors: true

schedule:
  policy: fixed_voronoi_midpoint
  coordinate_domain: calibrated_target
  source_center: calibrated_principal_point
  preserve_first_full_fov: true
  preserve_last_full_fov: true
  allow_zero_width_source_skip: true
  permit_dynamic_pair_boundary: false
  permit_object_owner: false

open3d_audit:
  enabled: true
  use_for_qualification_only: true
  require_every_adjacent_final_edge: true
  allow_rgb_generation: false
  allow_pose_replacement: false
  allow_tsdf: false

render:
  remap_only_assigned_roi: true
  interpolation: linear
  transition_width_px: 0
  color_gain_enabled: false
  rotation_enabled: false
  synthetic_hole_fill: false
  foreign_source_fill: false
  decode_workers: 4
  frame_cache_size: 8

vertical:
  enabled_for_stage_a: false
  model: per_source_scalar_translation
  calibrated_analysis_width_px: 424
  verify_sign_with_end_to_end_synthetic_test: true
  equation_convention: determined_by_test
  use_skip_constraints: true
  skip_steps: [2, 4]
  skip_stride: 8
  huber_delta_px: 1.0
  irls_iterations: 3
  second_difference_weight: 2.0
  prior_weight: 0.20
  allow_canvas_interpolated_offset: false
  allow_pair_local_warp: false
  allow_row_dependent_warp: false

tracks:
  whole_track_solver_heldout_split: true
  heldout_fraction: 0.20
  split_seed: 12012
  minimum_track_length_frames: 3
  crossing_margin_analysis_px: 4
  maximum_output_cluster_gap_px: 2
  maximum_projected_output_span_px: 2.0
  minimum_evaluable_track_count: 64
  horizontal_bin_count: 8
  minimum_covered_horizontal_bins: 6
  vertical_band_count: 3
  minimum_covered_vertical_bands: 2
  zero_evaluable_tracks_is_failure: true

acceptance:
  stage_a_required_before_stage_b: true
  maximum_unassigned_internal_columns: 0
  maximum_multiply_assigned_columns: 0
  minimum_valid_pixel_fraction: 0.995
  horizontal_line_residual_p95_px: 0.75
  horizontal_line_residual_p99_px: 1.0
  fixed_region_maximum_jump_px: 1.0
  maximum_duplicate_heldout_track_count: 0
  maximum_missing_heldout_track_count: 0

output:
  mode: audit
  write_pose_audit: true
  write_scan_audit: true
  write_motion_graph: true
  write_anchor_audit: true
  write_pixel_provenance: true
  write_source_uv: true
  write_open3d_edge_audit: true
  write_fixed_region_crops: true
  write_track_audit: true
  atomic_publish: true
```

项目已有 pose 和 Open3D gate 时，优先复用现有正式门限。

没有现成门限时，先输出分布，不得根据最终图片是否好看反向调资格门限。

---

# 8. M0：独立 session 和 trajectory

实现：

```text
video_s12_session.py
video_s12_pose.py
```

CLI 不再要求任何 S1 baseline。

需要读取：

```text
manifest.json
calibration.json
frames.csv
RGB 文件
depth 文件，仅在 Open3D 审计时读取
ORB-SLAM3 trajectory
```

Trajectory 记录至少包含：

```text
frame_id
timestamp
pose status
pose kind
pose origin
camera_to_world
tracking state
```

如果现有 trajectory 格式没有明确区分 direct 和 interpolated，必须先扩展轨迹导出格式。

不能通过猜测判断 direct pose。

---

# 9. M1：自动识别有效扫描段

实现：

```text
video_s12_scan.py
```

## 9.1 拟合轨道主轴

从所有 direct ORB camera center 中稳健拟合一条轨道轴。

建议：

- RANSAC 拟合三维直线
- 使用 PCA 细化内点方向
- 方向符号由时间顺序确定

每帧计算：

```text
沿轨道进度 s_i
横向偏离
姿态 roll/pitch/yaw
是否 direct pose
```

## 9.2 有效扫描段

选择最长的连续区间，满足：

```text
沿轨道总体单调前进
横向偏离不过限
姿态不过限
RGB 连续可用
direct pose 密度足够
图像运动方向与 pose 前进方向一致
```

暂停帧可以存在，但不应反复生成输出列。

明显后退时：

- 结束当前扫描段
- 或开始新的独立扫描段

第一版只处理一段最长有效扫描，不合并往返路径。

## 9.3 不足处理

如果 direct pose 太少，或者最长有效扫描不足：

```text
structural fatal
```

不得退回 S1，也不得使用二维运动替代 pose 资格。

---

# 10. M2：建立标定域图像运动图

实现：

```text
video_s12_motion_graph.py
```

## 10.1 节点

初始节点为有效扫描段中的所有 pose 合格真实帧。

## 10.2 图像域

先将 RGB 采样到标定后的分析域。

所有 dx、dy 和 track 都在同一个 calibrated coordinate 中计算。

不得混用：

```text
raw distorted coordinates
calibrated target coordinates
full-resolution canvas coordinates
```

## 10.3 运动边

建立：

```text
相邻 pose 合格帧边
候选索引相差 2 的 skip edge
候选索引相差 4 的 skip edge
后续自动 anchor 之间的 long edge
```

每条边必须直接匹配对应的两张真实图。

不能将若干相邻边相加后冒充 direct edge。

## 10.4 特征和模型

建议：

1. 在 4×6 或 6×8 网格中选 Shi-Tomasi 特征。
2. 使用 pyramidal LK。
3. 进行前后向检查。
4. 使用 RANSAC 求稳健平移。
5. 检查 feature 的水平和垂直覆盖。
6. 分析水平位移是否存在多个明显视差簇。
7. 选取稳定的主运动簇。
8. 多运动过强时降低置信度或拒绝该边。

输出：

```python
@dataclass(frozen=True)
class S12MotionEdge:
    left_node: int
    right_node: int
    left_frame_id: int
    right_frame_id: int
    measured_dx_px: float
    measured_dy_px: float
    inlier_count: int
    inlier_ratio: float
    grid_coverage: float
    vertical_span_fraction: float
    forward_backward_p95_px: float
    parallax_cluster_ratio: float
    pose_progress_m: float
    pose_direction_agreement: bool
    confidence: float
    reliable: bool
    rejection_reasons: tuple[str, ...]
```

## 10.5 可靠测量不得被统一平滑

可靠边的 `measured_dx_px` 原样进入运动图。

禁止：

- 中值滤波
- 全局缩放
- 线性插值
- 首尾外推
- 静默裁剪

不可靠边不进入正式图求解。

图必须由其他可靠 direct edge 保持连通。

---

# 11. M3：两遍自动 anchor 生成

实现：

```text
video_s12_anchor.py
```

Anchor 不是外部绝对坐标。

它是自动选择的高质量关键帧，用于建立长距离直接图像运动边。

## 11.1 第一遍临时位置

先只使用：

```text
可靠相邻边
可靠 skip 2 边
可靠 skip 4 边
```

求一个 provisional center path。

第一帧设：

```python
c_0 = calibration.cx
```

这只用于消除整体平移自由度，不决定全景尺度。

## 11.2 Anchor 候选条件

Anchor 必须：

```text
direct pose 合格
图像质量合格
姿态合格
局部相邻边可靠
至少有一条可靠 skip edge
不位于暂停或回退区
不位于图像运动多簇严重区
```

## 11.3 自动选择

第一和最后一个有效 source 强制作为 anchor。

中间 anchor 按 provisional pixel progress 选择。

从上一个 anchor 出发：

```text
目标距离约 96 px
允许范围 64 到 144 px
```

在目标附近选择候选。

候选排序采用明确的优先级，不使用随意综合分数：

1. 能与上一个 anchor 建立可靠 direct long edge。
2. 与上一个 anchor 之间存在两条独立可靠图路径。
3. long edge confidence 最高。
4. 局部最弱边 confidence 最高。
5. 图像质量最高。
6. 与 96 px 目标距离最接近。
7. frame id 较小，保证确定性。

## 11.4 Anchor long edge

Anchor 确定后，必须直接匹配两个 anchor 图像。

加入：

```text
anchor_i → anchor_(i+1)
```

长距离图像运动边。

由于 anchor 间隔远小于 848 像素图像宽度，理论上仍有大面积重叠。

如果 direct anchor edge 不可靠：

- 尝试附近其他候选
- 或缩短 anchor 间距
- 仍失败则该区间 structural fatal

## 11.5 两条独立路径

Anchor 区间至少需要：

```text
一条直接 anchor long edge
一条由局部相邻/skip edge 组成的路径
```

两者差异用于审计累计漂移。

Anchor 不提供预设 x，只提供更强的图约束。

---

# 12. M4：全局水平位置求解

实现：

```text
video_s12_motion_graph.py
```

## 12.1 未知量

最终候选 source 按时间排序。

定义相邻正向增量：

```text
p_k = c_(k+1) - c_k
```

要求：

```text
p_k >= minimum_increment_px
```

任意可靠边 `i→j` 提供：

```text
Σ p_k = measured_dx_ij
k = i ... j-1
```

## 12.2 目标函数

使用：

```text
相邻边
skip 2 边
skip 4 边
anchor long edge
```

目标：

```text
Σ w_ij ρ(Σ p_k - dx_ij)
```

其中：

- `ρ` 使用 Huber
- `w_ij` 来自 edge confidence
- 不添加水平速度平滑
- 不要求相邻条带宽度相近
- 真实速度变化必须保留

## 12.3 求解方式

不能新增大型依赖。

建议使用：

```text
IRLS
加简单下界 active-set
NumPy lstsq
```

每轮：

1. 根据当前残差更新 Huber 权重。
2. 解加权线性最小二乘。
3. 将违反下界的增量加入 active set。
4. 固定这些增量为最小值。
5. 重新求剩余变量。
6. 直到没有新增违反项。

固定：

```python
c_0 = calibration.cx
```

从 `p_k` 累加得到所有 center。

## 12.4 验收

必须满足：

```text
所有 center 严格递增
所有可靠边残差可审计
anchor direct edge 与局部路径一致
没有全局 normalization
没有可靠 edge 中值滤波
最大 edge correction 不超过门限
```

如果运动图不连通或残差异常：

```text
structural fatal
```

---

# 13. M5：最终 source 选择

初始 source 为所有 pose 合格帧。

## 13.1 去除重复位置 source

若两个 source center 距离小于 0.5 px：

保留：

1. anchor
2. 图像质量更高
3. pose 质量更高
4. Open3D 邻接更稳定
5. frame id 较小

被删除 source 不参与渲染。

## 13.2 保证普通内部狭缝宽度

先生成临时 Voronoi schedule。

普通内部宽度：

```text
preferred <= 16 px
hard <= 24 px
```

超过 16 只记录，不算结构失败。

超过 24 时：

1. 在对应时间区间搜索尚未使用的 pose 合格真实帧。
2. 要求该帧与左右 source 的 direct motion edge 可靠。
3. 要求新的相邻 Open3D 边通过。
4. 加入后重新求局部运动图和 center。
5. 最多两轮。

仍超过 24：

```text
structural fatal
```

禁止虚拟 RGB。

## 13.3 Anchor 保留

自动 anchor 不得在普通去重和补帧阶段被删除。

如果某个 anchor 资格后来失败：

- 重建该区间 anchor
- 不能静默降级为普通 source

---

# 14. Open3D 最终邻边审计

在最终 source 链确定后，对每一对相邻 source 做 Open3D RGB-D 边审计。

Open3D 只检查：

```text
ORB relative pose 是否与 RGB-D 局部证据基本相容
邻接关系是否可信
```

禁止：

- 修改 RGB
- 替换 ORB pose
- TSDF
- 生成点云全景
- 补齐 source
- 精细几何重投影

非 anchor source 导致失败时，可以删除并重建邻接，最多两轮。

Anchor 相关必要边失败时，structural fatal，或者重新选 anchor。

---

# 15. M6：固定 Voronoi 列分配

实现：

```text
video_s12_schedule.py
```

## 15.1 Canvas 自动生成

第一个 center 固定为：

```python
c_0 = calibration.cx
```

第一帧 rectified support 左边因此为 0。

最终 canvas：

```python
canvas_left = 0
canvas_right = ceil(c_last + (calibration.width - calibration.cx))
canvas_width = canvas_right - canvas_left
canvas_height = calibration.height
```

不读取 S1 canvas。

## 15.2 中间边界

相邻中心：

```text
c_i
c_(i+1)
```

边界：

```python
b_i = floor((c_i + c_(i+1)) * 0.5 + 0.5)
```

第 `i` 个 source 负责半开区间：

```text
[b_(i-1), b_i)
```

## 15.3 首尾完整视场

第一个 source 左边界：

```text
0
```

最后一个 source 右边界：

```text
canvas_width
```

首尾条带可以很宽。

首尾宽度单独报告，不进入内部 24 px 门限。

## 15.4 Source 坐标

输出列 `x_canvas` 在 source `i` 中的 rectified x：

```python
source_x = calibration.cx + (x_canvas - c_i)
```

不得使用 `width / 2` 代替 `cx`。

## 15.5 零宽 source

整数化后无输出列的 source：

```text
不解码
不渲染
在 schedule 中标记 zero_width
```

每个内部输出列必须恰好属于一个 source。

---

# 16. M7：Stage A 渲染

实现：

```text
video_s12_renderer.py
```

Stage A 只包含：

```text
自动 source
自动 center
固定 Voronoi
去畸变
窄 ROI remap
```

Stage A 明确不包含：

```text
纵向修正
旋转修正
颜色增益
RGB 混合
羽化
MultiBand
foreign fill
深度补图
对象 owner
```

## 16.1 只 remap 实际 ROI

某个 source 负责 `w` 列，就只处理：

```text
H × w
```

不得 remap 完整帧后再裁剪。

## 16.2 Stage A 坐标

```python
rectified_x = calibration.cx + (canvas_x - source_center_x)
rectified_y = output_y
```

再通过预计算的去畸变 map 得到 raw RGB 坐标。

## 16.3 无混合

```yaml
transition_width_px: 0
```

每个有效输出像素只来自一个真实帧。

---

# 17. M8：像素级 provenance

必须保存：

```text
owner_frame_id[H,W]       int32
valid_mask[H,W]           bool
source_u[H,W]             float32
source_v[H,W]             float32
assignment_index[H,W]     int32
```

规则：

```text
有效像素 owner_frame_id 为真实 frame id
无效像素 owner_frame_id 为 -1
valid_mask == (owner_frame_id >= 0)
同一像素只允许写一次
无效像素不得被其他 source 越权填充
```

同时保存列级索引：

```text
column_frame_id[W]
column_source_u[W]
column_assignment_index[W]
```

列级数据用于快速审计，不能替代像素级 provenance。

---

# 18. Stage A 输出和停止点

第一次交付至少输出：

```text
stage_a_dense_slit.png
owner_frame_id.npz
valid_mask.png
source_uv.npz
column_provenance.npz

pose_qualification.csv
scan_segment.json
motion_edges.csv
motion_graph.json
anchor_selection.json
anchor_edges.csv
horizontal_solution.csv
source_selection.csv
open3d_edge_audit.json
strip_schedule.csv

fixed_regions/
worst_regions/

performance.json
report.json
config_snapshot.yaml
```

Codex 完成 Stage A 后必须停止。

---

# 19. Stage A 验收

第一阶段主要判断密集狭缝本身是否解决：

- 箱子被宽 source 切碎
- 翻盖出现多个尖角副本
- 插线板出现两个位置
- 白线出现双份
- 风扇完整重复
- 跨多个 pair 的设备被重复保留

Stage A 暂时不要求所有横线纵向完全对齐。

## 19.1 固定验证区域

验证区域不再使用 S1 canvas 坐标。

`S012_validation_regions_v1.json` 使用：

```text
run name
reference source frame id
source-local bbox
说明
```

运行后根据 source center 自动换算为 canvas crop。

示例：

```json
{
  "run_20260806_153033": {
    "box_and_flaps": {
      "reference_frame_id": 477,
      "source_bbox": [300, 180, 720, 470]
    }
  }
}
```

具体 frame 和 bbox 由 Codex查看真实帧后确认。

不得根据 S1.2 输出结果移动区域来避开失败。

## 19.2 Stage A lock

用户确认 Stage A 后生成：

```text
S012_stage_a.lock.json
```

锁定：

```text
session hash
trajectory hash
calibration hash
有效扫描段
pose 合格 source
自动 anchor
motion graph edge hash
source center
Voronoi boundary
canvas
Stage A 图 hash
provenance hash
config hash
```

Stage B 必须验证此 lock。

---

# 20. 第二次交付：完整 feature track

实现：

```text
video_s12_tracks.py
```

## 20.1 整轨迹

建立跨多个 source 的 feature track。

每条 track 保存：

```text
track_id
所有 observation
frame id
calibrated u/v
LK 前后向误差
质量
是否完整穿越中央狭缝
```

## 20.2 Solver 和 held-out 整轨迹分离

先完成整条 track，再分组。

同一 track 只能属于：

```text
solver
或 heldout
```

不得按 observation 分割。

使用稳定 hash 和固定 seed，保证重复运行一致。

---

# 21. 每 source 全局纵向参数

实现：

```text
video_s12_vertical.py
```

每个 source 一个 scalar：

```text
t_i
```

第 `i` 个 source 整个条带使用：

```python
source_y = output_y + t_i
```

禁止：

```text
v_canvas[x]
pair-local warp
row-dependent warp
窄肩 cosine warp
左右 source 半量分摊
```

## 21.1 dy 测量

所有 dy 在 calibrated analysis domain 中测量。

相邻 source 和少量 skip source 建立约束。

## 21.2 符号

预期关系可能是：

```text
t_j - t_i = -dy_ij
```

但必须通过端到端合成测试确定。

测试必须覆盖：

```text
右图横线向上
右图横线向下
实际 motion estimator
实际全局求解
实际 inverse remap
最终横线是否重合
```

## 21.3 全局求解

目标：

```text
Σ w_ij ρ((t_j - t_i) - target_ij)
+ λ2 Σ (t_(i-1) - 2t_i + t_(i+1))²
+ λp Σ (t_i - prior_i)²
```

要求：

- 一个 source 固定 `t=0`
- IRLS
- Huber
- NumPy 求解
- solver track 参与
- held-out track 不参与
- source 集和列边界保持 Stage A lock 不变

---

# 22. Stage B 渲染

Stage B 与 Stage A 使用完全相同的：

```text
source 集
source center
anchor
canvas
Voronoi boundary
column owner
```

唯一变化：

```python
rectified_y = output_y + t_i
```

不得在 Stage B 静默删除 source 或改变 schedule。

若发现某 source 需要删除：

```text
Stage B 失败
输出 source pruning 建议
回到 Stage A 重建
```

---

# 23. 全景 held-out track 审计

一个物理点可能连续被多帧采样，所以“只有一个连续区间”不够。

## 23.1 实际采样位置

对 held-out track observation，计算它投到 canvas 的位置。

只有该位置：

```text
落入当前 source 的固定 Voronoi 列
对应像素有效
```

才算被实际采样。

## 23.2 通过条件

每条完整穿越中央狭缝的 held-out track：

```text
cluster_count == 1
projected_x_span <= 2 px
不存在第二个分离 cluster
应该穿越但完全未采样的 track 不允许
```

还报告：

```text
projected_y_span
sampled_source_count
```

连续形成 5 像素小段也必须失败。

## 23.3 有效样本门

至少：

```text
64 条可评估 held-out track
8 个水平分区覆盖至少 6 个
3 个垂直分区覆盖至少 2 个
```

0 条可评估轨迹必须失败。

---

# 24. 自动 anchor 审计

`anchor_selection.json` 必须逐个记录：

```text
anchor index
frame id
pose 资格
provisional center
与上一个 anchor 的预计距离
为什么选择它
被淘汰的附近候选
direct anchor edge 指标
第二独立路径指标
direct edge 与局部路径差异
```

必须证明：

```text
anchor 来自算法自动选择
没有读取 S1
没有人工写死 frame id
```

固定验证数据可以写 anchor 回归断言，但不能把回归 frame id 作为选择规则。

---

# 25. CLI

新增：

```text
g305-video-s12-experiment
```

参数：

```text
input
--trajectory
--config
--output
--stage a|b
--stage-a-lock
--output-mode audit|minimal
```

Stage A 示例：

```powershell
g305-video-s12-experiment `
  D:\central_strip_Panoramic_Camera\data\captures\video\run_20260806_153033 `
  --trajectory D:\central_strip_Panoramic_Camera\data\captures\video\run_20260806_153033\orbslam3_trajectory.json `
  --config D:\central_strip_Panoramic_Camera\configs\video_candidates\S012_standalone_auto_anchor_dense_central_slit_v1.yaml `
  --stage a `
  --output D:\central_strip_Panoramic_Camera\benchmarks\run_20260806_153033\S012_standalone_auto_anchor_dense_central_slit_v1_stage_a `
  --output-mode audit
```

Stage B 必须提供：

```text
--stage-a-lock
```

CLI 中不得有：

```text
--baseline
--s1
--source-layout-from-s1
```

---

# 26. 当前两组数据的处理要求

## 26.1 慢速数据

```text
run_20260806_153033
```

先统计：

```text
direct pose 数
最长有效扫描段
pose 合格候选数
自动 anchor 数
运动图边数
最终 source 数
内部条带 P50/P95/max
自动 canvas 尺寸
```

此前“约 122 个 direct source、条带约 8/11/12”只能作为审查参考。

必须由新算法独立重算。

不得读取 S1 来验证这些数值。

## 26.2 快速数据

```text
run_20260807_140140
```

当前 direct pose 链不完整。

正式实施顺序：

1. 使用现有离线 ORB-SLAM3 入口重新处理完整会话。
2. 输出明确 direct pose 状态。
3. 不允许插值补 pose。
4. 重新执行 M0/M1 资格检查。
5. 如果 pose 合格 source 无法把内部条带控制在 24 px 内，structural fatal。
6. 需要提高 ORB 跟踪覆盖，或重新采集更慢数据。

没有完整 pose 链时，不运行正式 Stage A。

也不退回 S1。

---

# 27. 原子发布

所有输出先写 staging。

Structural fatal：

```text
不替换旧 bundle
删除 staging
写 sibling failure.json
CLI 返回非零
```

Quality fail：

```text
完整审计 bundle 可以原子发布
acceptance.pass=false
```

必须测试：

```text
首次发布
替换旧结果
图像写入失败
NPZ 写入失败
report 写入失败
rename 失败
旧 bundle 恢复
```

---

# 28. Report schema

```json
{
  "schema": "gemini305-video-s12-standalone-report/v1",
  "algorithm_id": "S012_standalone_auto_anchor_dense_central_slit_v1",
  "stage": "a",
  "diagnostic_only": true,
  "production_eligible": false,
  "structural_valid": true,
  "input": "...",
  "trajectory": {
    "path": "...",
    "hash": "...",
    "pose_origin": "...",
    "direct_pose_count": 0
  },
  "scan": {
    "start_frame_id": 0,
    "end_frame_id": 0,
    "pose_displacement_m": 0.0,
    "candidate_frame_count": 0
  },
  "motion_graph": {
    "node_count": 0,
    "adjacent_edge_count": 0,
    "skip_edge_count": 0,
    "anchor_edge_count": 0,
    "rejected_edge_count": 0,
    "connected": true
  },
  "anchors": {
    "count": 0,
    "frame_ids": [],
    "spacing_p50_px": 0.0,
    "spacing_p95_px": 0.0,
    "all_direct_edges_pass": true,
    "all_two_path_checks_pass": true
  },
  "horizontal": {
    "source_count": 0,
    "center_increment_p50_px": 0.0,
    "center_increment_p95_px": 0.0,
    "edge_residual_p95_px": 0.0,
    "global_rescale_applied": false,
    "median_filter_applied": false
  },
  "schedule": {
    "canvas_width": 0,
    "canvas_height": 0,
    "nonzero_source_count": 0,
    "internal_width_p50_px": 0.0,
    "internal_width_p95_px": 0.0,
    "internal_width_max_px": 0,
    "preferred_width_exceed_count": 0,
    "hard_width_exceed_count": 0,
    "unassigned_internal_columns": 0,
    "multiply_assigned_columns": 0
  },
  "provenance": {
    "valid_pixel_fraction": 0.0,
    "duplicate_write_count": 0,
    "consistency_pass": true
  },
  "vertical": {},
  "heldout_tracks": {},
  "performance": {},
  "acceptance": {
    "pass": false,
    "structural_fatal": false,
    "reasons": []
  }
}
```

---

# 29. 第一阶段测试

## 29.1 S1 独立性

```text
test_s12_cli_has_no_baseline_argument
test_s12_does_not_read_s1_report
test_s12_does_not_read_s1_layout
test_s12_canvas_is_computed_from_automatic_centers
test_s12_runs_on_synthetic_session_without_any_s1_artifact
```

## 29.2 Pose 和扫描段

```text
test_missing_direct_pose_frame_is_not_render_source
test_interpolated_pose_is_rejected
test_mixed_pose_origin_is_structural_fatal
test_invalid_se3_is_structural_fatal
test_scan_axis_is_recovered_from_pose
test_pause_does_not_create_repeated_progress
test_backward_motion_splits_scan
test_rotation_is_audited_when_render_rotation_is_disabled
```

## 29.3 Motion graph

```text
test_reliable_dx_is_not_median_filtered
test_unreliable_edge_is_not_interpolated
test_unreliable_edge_is_not_extrapolated
test_adjacent_and_skip_edges_form_connected_graph
test_disconnected_graph_is_structural_fatal
test_multi_motion_edge_is_rejected
test_pose_and_image_direction_must_agree
```

## 29.4 自动 anchor

```text
test_first_and_last_sources_are_anchors
test_anchor_selection_is_automatic_and_deterministic
test_anchor_spacing_uses_provisional_pixel_progress
test_anchor_requires_direct_long_edge
test_anchor_requires_second_independent_path
test_failed_anchor_candidate_selects_nearby_alternative
test_anchor_interval_without_valid_candidate_is_structural_fatal
```

## 29.5 水平求解

```text
test_positive_increment_solver_recovers_known_centers
test_long_edges_reduce_accumulated_drift
test_no_global_rescale_is_applied
test_real_speed_changes_are_preserved
test_edge_adjustment_is_bounded
test_first_center_uses_calibrated_cx_only_as_gauge
```

## 29.6 Schedule

```text
test_voronoi_columns_partition_canvas_exactly_once
test_schedule_uses_half_open_intervals
test_source_x_uses_calibrated_cx
test_zero_width_source_is_skipped
test_wide_object_can_span_many_strips
test_no_object_owner_is_created
test_internal_width_over_24_is_structural_fatal
test_canvas_does_not_depend_on_s1
```

## 29.7 Renderer 和 provenance

```text
test_stage_a_has_no_vertical_correction
test_stage_a_has_no_color_gain
test_stage_a_has_no_blend
test_roi_remap_matches_full_coordinate_reference
test_owner_frame_id_is_pixel_level
test_invalid_owner_is_minus_one
test_valid_mask_matches_owner
test_second_pixel_write_is_structural_fatal
test_foreign_fill_is_disabled
test_depth_never_generates_rgb
```

---

# 30. 第二阶段测试

## 30.1 Track split

```text
test_whole_track_is_solver_or_heldout
test_split_is_deterministic
test_heldout_track_never_enters_solver
test_zero_evaluable_tracks_fails
```

## 30.2 纵向符号

```text
test_positive_dy_end_to_end_horizontal_line
test_negative_dy_end_to_end_horizontal_line
test_equation_sign_matches_inverse_remap
```

## 30.3 每 source t_i

```text
test_each_source_has_one_scalar_translation
test_scalar_applies_to_entire_strip
test_canvas_interpolated_v_is_forbidden
test_pair_local_warp_is_forbidden
test_rendered_line_continuity_not_offset_continuity_is_measured
```

## 30.4 Held-out cluster

```text
test_one_cluster_span_le_2_passes
test_one_cluster_span_gt_2_fails
test_second_cluster_fails
test_expected_crossing_without_sample_is_missing
test_minimum_track_count_is_enforced
test_spatial_coverage_is_enforced
```

---

# 31. 性能要求

第一阶段目标：

```text
慢速 Stage A <= 20 s
快速 Stage A <= 20 s，前提是 direct pose 链合格
```

第二阶段目标：

```text
Stage B 总时间 <= 25 s
```

优化顺序：

1. 零宽 source 不解码。
2. 只 remap 实际条带。
3. 有界 JPEG 解码线程池。
4. 滚动帧缓存。
5. 缓存标定 map。
6. 复用分析图和特征。
7. 诊断输出后置。

不得通过关闭 pose、Open3D、provenance 或 held-out 审计伪造速度达标。

---

# 32. 全仓验证

每个里程碑结束运行：

```powershell
pytest
ruff check src/panorama_demo/video_s12_*.py tests/test_video_s12_*.py
python -m compileall src/panorama_demo
git diff --check
```

还要确认：

```text
production renderer 未修改
production lock 未修改
S0/S1/S1.1 代码行为未修改
```

现有无关 Ruff 错误只能报告，不能越权修改。

---

# 33. 第一次交付完成定义

只有同时满足以下条件，才可声明 Stage A 完成：

```text
没有读取任何 S1 文件
没有 --baseline 参数
有效扫描段由 pose 自动识别
所有渲染 source 都有 direct ORB pose
自动 anchor 已生成
每个 anchor 都有可靠 direct long edge
每个 anchor 区间都有第二独立图路径
水平 center 由多帧运动图求出
没有中值滤波
没有全局首尾拉伸
没有插值不可靠 edge
canvas 自动生成
固定 Voronoi 列恰好分配一次
普通内部条带不超过 24 px
像素级 provenance 一致
Stage A 无纵向、无颜色、无融合
箱子、线缆、风扇固定裁剪已输出
production 未修改
```

完成后停止，等待 Stage A 检查。

---

# 34. 最终候选完成定义

Stage B 通过后，正式候选还需满足：

```text
Stage B 使用 Stage A lock
source、center、anchor、canvas 和列分配不变
每个 source 只有一个 scalar t_i
没有 v_canvas 跨 source 插值
纵向符号由端到端测试确定
solver 和 held-out 按整轨迹分离
held-out cluster_count 为 1
held-out projected span 不超过 2 px
没有第二 cluster
没有应穿越却缺失的 track
可评估 track 数和空间覆盖达标
横线 P95/P99 和固定区域通过
像素级 provenance 通过
没有颜色、融合、对象 owner 和 pair-local warp
```

---

# 35. Codex 第一次交付报告格式

Codex 必须按以下顺序报告：

1. 修改文件。
2. 是否读取或依赖任何 S1 文件。
3. trajectory 格式和 direct pose 数。
4. 自动扫描段起止帧。
5. pose 合格候选数。
6. motion graph 节点和各类边数量。
7. 被拒绝运动边及原因。
8. 自动 anchor frame id 和选择原因。
9. anchor direct edge 和第二路径是否通过。
10. 最终 source 数。
11. source center 增量 P50、P95、最大值。
12. 自动 canvas 尺寸。
13. 内部条带 P50、P95、最大值。
14. Open3D 最终邻边通过情况。
15. provenance 一致性。
16. 箱子、线缆、风扇固定区域截图。
17. 总时间和最慢阶段。
18. pytest、Ruff、compileall、git diff check。
19. 未通过项必须明确列出。
