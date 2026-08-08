# 中央条带全景拼接第一版实施方案

## 文档用途

本文件用于直接交给 Codex 实施。

本版本只实现新的基础路线，不沿用 v6 或 v6.1 的算法框架，也不修改当前 production 行为。目标是在现有工程中建立一个独立、可重复、可诊断的实验入口，先完成稳定出图和局部纵向对齐。

本版本名称统一为：

```text
S01_output_first_vertical_alignment_v1
```

其中：

- S0 负责动态 source 选择、动态中央 owner 区域和稳定出图。
- S1 负责接缝附近随高度变化的局部纵向修正 `Δy(y)`。
- 第一版暂时使用固定中央接缝和极窄过渡。
- GraphCut、可弯曲 seam、自适应 MultiBand、二维 mesh、DIS 残差形变、RAFT、RGB-D 精细补图都不在本次范围内。

---

# 1. 项目和数据范围

## 1.1 代码仓库

```text
https://github.com/Beethoven-666/Panoramic_Camera
```

目标分支：

```text
codex/video-realtime-seam-v6
```

在该分支上实现新实验路线，但不得把新路线写进 v6、v6.1 或 D3 的 renderer 内部。

## 1.2 必测数据

慢速数据：

```text
D:\central_strip_Panoramic_Camera\data\captures\video\run_20260806_153033
```

快速数据：

```text
D:\central_strip_Panoramic_Camera\data\captures\video\run_20260807_140140
```

D3 视觉参考：

```text
D:\central_strip_Panoramic_Camera\benchmarks\run_20260806_153033\view_only\D3_full_scan_preview_run5
```

D3 只作为视觉参考，不允许直接复制 D1、D2、D3 的完整重型处理路径。

---

# 2. 第一版的目标

## 2.1 必须达到的目标

1. 两组数据都能稳定生成完整全景图。
2. S1 任意局部窗口匹配失败时，只让该区域的纵向修正降为零。
3. 某一对 source 的 S1 全部失败时，仍然输出该对 source 的 S0 结果。
4. 不允许因为画质门限不满足而终止整幅全景。
5. source 根据实际 canvas 推进量选择，不使用固定每 N 帧选择一次。
6. 中央 owner 区域根据相邻 source 的实际位置动态计算。
7. 每个 source 在中央 owner 区域左右保留额外匹配肩。
8. S1 只估计随高度变化的纵向修正 `Δy(y)`。
9. warp 只在 owner 边界和 overlap 附近生效，离开接缝后平滑降为零。
10. 低分辨率完成匹配和决策，全分辨率只执行一次最终 RGB remap。
11. 输出完整的 pair 级诊断结果和分阶段耗时。
12. S0 与 S1 必须使用相同 source 布局，方便直接比较 S1 的实际收益。

## 2.2 不允许通过以下方式伪装改善

- 不能通过大范围模糊掩盖横线阶梯。
- 不能通过整幅图宽 MultiBand 掩盖错位。
- 不能用 RGB-D、TSDF 或点云补充 RGB 纹理。
- 不能把匹配失败区域强行拟合成大幅度 warp。
- 不能为了通过指标裁掉问题区域。
- 不能把 S1 失败解释成整对 source 必须 hard cut。
- 不能用人工标注参与 source、warp 或 owner 决策。

---

# 3. 非目标

本次不实现：

- GraphCut seam。
- 动态规划弯曲 seam。
- 局部或全局 MultiBand。
- 二维自由 mesh。
- DIS 稠密光流形变。
- RAFT。
- 深度分层 warp。
- 深度连通物体。
- 跨帧物体跟踪。
- TSDF。
- 全局多标签 owner 优化。
- 全局复杂光度图优化。
- production lock。
- 20 m 数据验证。
- 自动算法选型。
- 画质不达标时阻止出图。

这些模块只能在 S0 和 S1 的效果已经单独验证后再考虑。

---

# 4. 总体处理流程

```text
加载视频会话
  ↓
低分辨率 RGB 运动分析
  ↓
确定单向扫描段
  ↓
按累计 canvas 推进量动态选择真实 source
  ↓
建立 source 中心、动态 owner 边界和左右匹配肩
  ↓
构建每个 source 的基础 inverse map
  ↓
S0 固定中央 owner 合成
  ↓
为每对相邻 source 提取公共 overlap ROI
  ↓
多高度窗口、多水平子窗口、Top-K 纵向候选
  ↓
左右一致性和局部置信度
  ↓
粗略 RANSAC 先验，可用时才使用
  ↓
动态规划寻找连续候选路径
  ↓
拟合平滑的 Δy(y)
  ↓
按局部收益选择 warp 增益
  ↓
把左右接缝修正合并到每个 source 的最终 inverse map
  ↓
全分辨率只 remap 一次
  ↓
固定 owner 边界和 1 至 2 px 极窄过渡
  ↓
输出 S1 全景和诊断文件
```

---

# 5. 工程隔离要求

## 5.1 新增独立实验入口

新增开发专用命令：

```text
g305-video-s1-experiment
```

建议入口文件：

```text
src/panorama_demo/video_s1_experiment.py
```

在 `pyproject.toml` 增加：

```toml
g305-video-s1-experiment = "panorama_demo.video_s1_experiment:main"
```

该入口只用于新路线研发。

不得修改普通用户的：

```text
g305-video-panorama
```

不得修改当前 production lock。

## 5.2 禁止直接依赖的旧模块

新路线不得 import：

```text
video_v6_pair_renderer.py
video_v61_renderer.py
video_v61_geometry_gate.py
video_graphcut_seam.py
video_near_blend.py
video_hard_guards.py
video_visual_renderer_v2.py
```

可以复用以下公共基础能力：

```text
load_video_session
analyse_video_scan
MotionEstimate
run_orbslam3_rgbd
calibration 数据结构
calibrated pushbroom 的基础布局和 inverse mapping
视频性能计时器
原子写 JSON 的公共工具
```

可以复用 `video_motion_resampler.py` 中与累计 RGB 运动和真实帧选择有关的基础逻辑，但必须扩展为 overlap 驱动的 source 选择，不能继续只依赖固定 normal/risk step。

## 5.3 推荐新增文件

```text
src/panorama_demo/video_s1_config.py
src/panorama_demo/video_s1_layout.py
src/panorama_demo/video_s1_vertical_alignment.py
src/panorama_demo/video_s1_warp.py
src/panorama_demo/video_s1_renderer.py
src/panorama_demo/video_s1_diagnostics.py
src/panorama_demo/video_s1_experiment.py
```

推荐测试文件：

```text
tests/test_video_s1_config.py
tests/test_video_s1_layout.py
tests/test_video_s1_vertical_alignment.py
tests/test_video_s1_warp.py
tests/test_video_s1_renderer.py
tests/test_video_s1_integration.py
```

不要把所有逻辑写进一个超长文件。

---

# 6. 配置文件

新增：

```text
configs/video_candidates/S01_output_first_vertical_alignment_v1.yaml
```

建议初始内容：

```yaml
schema: gemini305-video-s1-experiment/v1
algorithm_id: S01_output_first_vertical_alignment_v1

scan:
  analysis_width: 320
  motion_backend: lk_ransac
  fallback_motion_backend: phase_correlation
  maximum_workers: 4

source_selection:
  normal_target_step_px: 12.0
  risk_target_step_px: 8.0
  maximum_preferred_step_px: 18.0
  emergency_step_px: 24.0
  target_common_overlap_px: 128
  minimum_s1_overlap_px: 64
  shoulder_target_px: 64
  shoulder_minimum_px: 32
  shoulder_maximum_px: 96
  adjacent_frame_rescue: true

s0:
  transition_width_px: 1
  fill_invalid_owner_from_adjacent_real_source: true
  synthetic_hole_fill: false

s1:
  horizontal_scale: 0.5
  preserve_full_vertical_resolution: true

  window_height_px: 32
  window_step_px: 16
  minimum_window_height_px: 16
  maximum_window_height_px: 48

  horizontal_subwindow_width_px: 80
  horizontal_subwindow_count: 3
  minimum_horizontal_subwindow_width_px: 48

  search_radius_y_px: 16
  top_k_candidates: 5
  candidate_nms_radius_px: 2
  allow_subpixel_refinement: true

  luminance_zncc_weight: 0.45
  vertical_gradient_zncc_weight: 0.40
  lab_color_weight: 0.15

  minimum_valid_fraction_for_candidate: 0.45
  minimum_2d_texture_score: 0.05

  left_right_sigma_px: 1.5
  candidate_margin_sigma: 0.08
  multi_x_consensus_sigma_px: 1.5

  use_ransac_soft_prior: true
  ransac_minimum_points: 6
  ransac_residual_px: 2.0

  dp_first_order_weight: 1.0
  dp_second_order_weight: 0.5
  dp_missing_cost: 1.25
  dp_soft_prior_weight: 0.25

  smoothing_control_spacing_px: 80
  smoothing_data_weight: 1.0
  smoothing_first_order_weight: 0.25
  smoothing_second_order_weight: 2.0
  maximum_local_slope: 0.08

  gain_candidates: [0.0, 0.25, 0.5, 1.0]
  gain_block_height_px: 64
  gain_smoothness_weight: 0.20

  warp_support_left_px: 64
  warp_support_right_px: 64
  symmetric_pair_warp: true
  fade_function: cosine

output:
  write_s0_panorama: true
  write_s1_panorama: true
  write_owner_map: true
  write_pair_diagnostics: true
  write_candidate_heatmaps: true
  write_before_after_crops: true
  write_performance_report: true
```

这些数值只是第一轮实验起点。

除输入完整性和数值安全外，不能把这些配置变成整对 source 或整幅全景的通过门限。

---

# 7. 数据结构

## 7.1 SourceLayout

在 `video_s1_layout.py` 定义不可变数据结构：

```python
@dataclass(frozen=True)
class S1SourceLayout:
    frame_id: int
    source_index: int
    canvas_center_x: float
    owner_left_x: int
    owner_right_x: int
    support_left_x: int
    support_right_x: int
    match_left_x: int
    match_right_x: int
    measured_progress_from_previous_px: float
    source_quality: dict[str, float]
```

要求：

- `owner_left_x < owner_right_x`
- owner 区域按时间顺序排列。
- 相邻 owner 区域无重叠、无空洞。
- match 区域可以重叠。
- owner 宽度由相邻 source 中心中点决定。
- 第一帧和最后一帧 owner 区域延伸到各自有效支持边界。

## 7.2 PairOverlap

```python
@dataclass(frozen=True)
class S1PairOverlap:
    pair_index: int
    left_frame_id: int
    right_frame_id: int
    owner_boundary_x: int
    roi_left_x: int
    roi_right_x: int
    per_row_valid_left_x: np.ndarray
    per_row_valid_right_x: np.ndarray
    common_valid_mask: np.ndarray
    overlap_width_p5: float
    overlap_width_p50: float
    overlap_width_p95: float
    s1_evaluable_fraction: float
```

如果 `s1_evaluable_fraction` 很低，不得抛异常。该 pair 的 S1 输出 `dy=0`，并记录原因。

## 7.3 VerticalCandidate

```python
@dataclass(frozen=True)
class VerticalCandidate:
    window_index: int
    center_y: float
    dy_px: float
    raw_cost: float
    normalized_cost: float
    luminance_cost: float
    vertical_gradient_cost: float
    color_cost: float
    valid_fraction: float
    texture_score: float
    left_right_error_px: float
    multi_x_consensus_error_px: float
    confidence: float
    is_missing: bool
```

## 7.4 VerticalCurve

```python
@dataclass(frozen=True)
class VerticalCurve:
    y: np.ndarray
    dy_raw: np.ndarray
    dy_smoothed: np.ndarray
    confidence: np.ndarray
    gain: np.ndarray
    dy_applied: np.ndarray
    valid_observation_mask: np.ndarray
```

---

# 8. S0 动态 source 选择

## 8.1 运动估计

使用全部帧的低分辨率 RGB 图像计算相邻帧运动。

第一选择：

```text
Shi-Tomasi + pyramidal LK + RANSAC
```

只需要估计主要水平运动和较粗的垂直运动。

输出每条相邻边：

```text
dx
dy
inlier_ratio
grid_coverage
texture_coverage
reliable
```

当 LK 证据不足时，只对该边使用相位相关作为 fallback。

如果两个方法都不可靠：

- 不终止。
- 使用最近若干可靠边的中位数水平速度预测。
- 将该边标记为低置信度。
- source 选择在该区域自动变密。
- 最差情况下选择相邻真实帧。

## 8.2 累计 canvas 推进量

根据扫描方向计算单调进度：

```text
progress_t = max(progress_t-1, progress_t-1 + direction * dx_t)
```

小幅反向抖动记录到运动不确定性中，但不能让整体 canvas 进度倒退。

保留原始 `dx`，不要丢失诊断信息。

## 8.3 source 选择规则

source 选择必须使用真实帧，不得插值 RGB 帧。

基本规则：

1. 保留扫描段第一帧。
2. 从上一个 source 开始累计 canvas 推进量。
3. 正常区域达到约 12 px 时选择候选 source。
4. 风险区域达到约 8 px 时选择候选 source。
5. 如果单帧推进已经较大，立即选择当前帧。
6. 如果预测再等待一帧会使公共 overlap 低于目标，立即选择当前帧。
7. 目标位置附近有多帧可选时，按以下顺序评分：
   - 公共 overlap 是否充足。
   - 图像是否清晰。
   - 是否严重过曝或欠曝。
   - 局部纹理是否充足。
   - 是否接近目标 canvas 位置。
8. 保留扫描段最后一帧。
9. 快速运动时允许连续选择相邻帧。
10. 慢速运动时允许跳过大量重复帧。

不得固定每 N 帧选择一次。

## 8.4 动态 owner 边界

source 中心为：

```text
p_i
```

内部 source 的 owner 边界：

```text
owner_left_i  = round((p_i-1 + p_i) / 2)
owner_right_i = round((p_i + p_i+1) / 2)
```

第一 source 的左边界和最后 source 的右边界取有效 support 边界。

owner 区域必须满足：

- 按时间顺序单调。
- 不能出现空洞。
- 不能出现两个 source 同时拥有同一个最终像素。
- owner 边界不依赖 S1 匹配结果。

## 8.5 匹配肩

每个 source 在 owner 区域左右保留匹配肩：

```text
[左侧匹配肩] [中央 owner 区域] [右侧匹配肩]
```

默认单侧 64 px。

动态调整规则：

- 公共 overlap 充足时使用目标宽度。
- overlap 较小时缩小到实际可用范围。
- 不能小于 32 px 后仍声称 S1 可正常评估。
- 小于最小值时只把该局部区域标记为 S1 不可评估。
- S0 仍继续输出。

---

# 9. S0 基础渲染

## 9.1 基础 inverse map

复用公共标定和 pushbroom 布局能力，生成每个 source 的：

```text
inverse_x
inverse_y
valid_mask
```

不得从 v6 renderer 调用其私有流程。

如果公共模块缺少稳定的公开接口，可以在公共 calibrated pushbroom 模块中新增只负责生成 inverse map 的函数，例如：

```python
build_calibrated_source_inverse_maps(...)
```

旧调用方保持兼容。

## 9.2 S0 owner 合成

S0 使用动态 owner 区域直接合成。

每个最终像素：

1. 根据 x 坐标确定唯一 owner source。
2. 如果 owner source 在该像素有效，使用该真实 RGB。
3. 如果 owner source 无效，但相邻真实 source 有效，按距离 owner 边界最近的顺序选择相邻 source。
4. 不允许使用插值 RGB 帧。
5. 不允许使用深度补洞。
6. 所有真实 source 都只在最终阶段 remap 一次。

## 9.3 S0 过渡

第一版只允许：

```text
1 至 2 px
```

的简单线性过渡。

默认值为 1 px。

该过渡只用于避免整数 owner 边界的生硬亮度跳变，不能用于掩盖几何错位。

必须同时输出不经过渡的 owner-only 结果，供诊断对比。

---

# 10. S1 特征准备

## 10.1 处理分辨率

第一版建议：

- 水平方向缩小到 0.5。
- 高度方向保持原始高度。
- 最终估计和报告中的 `dy` 均换算到全分辨率像素。

原因是目标主要是修复 1 至 2 px 的纵向阶梯，不能过早损失高度分辨率。

## 10.2 每个 source 只计算一次

缓存：

```text
gray
Lab
gradient_x
gradient_y
Canny edge
structure tensor score
sharpness map
valid mask
```

相邻 pair 直接复用。

不得为同一个 source 在左 pair 和右 pair 重复计算这些特征。

## 10.3 曝光归一化

局部匹配前，对每个窗口使用：

- 局部均值去除。
- 局部标准差归一化。
- 必要时对 Lab 的 L 通道做局部归一化。

不能先做全局强颜色变换。

---

# 11. S1 局部候选生成

## 11.1 高度窗口

初始设置：

```text
窗口高度 32 px
窗口步长 16 px
```

窗口之间必须重叠。

允许根据纹理调整：

- 二维纹理丰富时可以缩小到 16 至 24 px。
- 一般区域使用 32 px。
- 重复横线或低纹理区域扩大到 40 至 48 px。

第一版可以先实现固定 32 px，再保留自适应接口。

## 11.2 水平子窗口

每个高度窗口在公共 overlap 内选择 3 个水平子窗口。

默认每个子窗口全分辨率宽度约 80 px。

选择位置：

- overlap 左部。
- overlap 中部。
- overlap 右部。

要求：

- 子窗口必须落在双方有效区域。
- overlap 过窄时自动缩小。
- 无法形成足够宽的子窗口时，该子窗口跳过。
- 至少一个子窗口可用时仍允许产生候选。
- 不能因为少一个子窗口就让整个高度窗口失败。

## 11.3 纵向搜索

默认搜索：

```text
dy ∈ [-16, +16] px
```

按全分辨率像素定义。

第一轮先计算整数候选。

对 Top-K 候选使用邻近代价做二次曲线拟合，得到亚像素 `dy`。

如果最佳候选位于搜索边界：

- 记录 `search_border_hit=true`。
- 降低该候选置信度。
- 第一版不自动无限扩大搜索范围。
- 可选只对该窗口进行一次扩大搜索，但最大不得超过 ±24 px。

## 11.4 匹配代价

每个子窗口的代价：

```text
0.45 × 亮度 ZNCC 代价
0.40 × 纵向梯度 ZNCC 代价
0.15 × Lab 颜色代价
```

建议统一转换到：

```text
越小越好
```

的代价。

纵向梯度必须保留较高权重，因为货架横线主要表现为纵向梯度峰。

## 11.5 Top-K

每个高度窗口保留：

```text
Top 5
```

候选。

候选之间使用 2 px 非极大值抑制，避免五个候选都来自同一个相关峰。

候选应包含：

- 综合代价。
- 三项子代价。
- 有效像素比例。
- 二维纹理分数。
- 多水平子窗口一致性。
- 搜索边界状态。

---

# 12. 重复横线防误匹配

## 12.1 二维纹理置信度

使用结构张量最小特征值或等价指标。

只有水平线、缺少立柱和角点时：

```text
texture_score 低
```

同时存在横线、立柱、标签、接头等二维结构时：

```text
texture_score 高
```

低二维纹理不代表不能匹配，但要降低候选置信度。

## 12.2 多水平位置一致性

同一高度的 3 个水平子窗口分别产生候选。

对相近 `dy` 聚类。

若多个水平位置支持同一 `dy`：

- 提高置信度。

若只有一个小区域支持：

- 降低置信度。
- 保留为候选，不直接删除。

## 12.3 左右一致性

同时执行：

```text
left → right
right → left
```

候选返回误差作为连续置信度。

不要使用一个固定 1.25 px 门限决定整段通过或失败。

建议：

```text
c_lr = exp(-(lr_error / sigma)^2)
```

默认：

```text
sigma = 1.5 px
```

## 12.4 候选唯一性

比较最佳候选和次优候选的代价差。

差值很小通常说明重复横线存在多个相似峰。

这种情况降低候选唯一性置信度，但仍保留多个候选交给动态规划。

## 12.5 允许 missing

每个高度窗口都必须有一个显式 missing 状态。

以下情况优先进入 missing：

- 双方有效像素不足。
- 完全无纹理。
- 左右一致性很差。
- 多个横线候选几乎同分。
- 遮挡或前景变化严重。
- 所有候选都位于搜索边界。

missing 不等于 pair 失败。

---

# 13. RANSAC 软先验

RANSAC 只用于建立粗略软先验，不得决定最终曲线。

从每个窗口的高置信度候选中拟合：

```text
dy(y) = a × y + b
```

条件：

- 至少 6 个候选点。
- 残差阈值初始为 2 px。
- 如果不足 6 点，跳过 RANSAC。
- 如果 RANSAC 结果不稳定，跳过 RANSAC。
- 跳过后仍继续动态规划。

RANSAC 只向动态规划增加小权重先验项：

```text
candidate_soft_prior_cost
```

不能删除偏离先验但局部证据很强的候选。

---

# 14. 动态规划

## 14.1 状态

每个高度窗口的状态包括：

```text
Top-K 候选
missing
```

## 14.2 代价

总能量：

```text
局部候选代价
+ 相邻位移一阶变化代价
+ 位移二阶变化代价
+ RANSAC 软先验代价
+ missing 代价
```

## 14.3 顺序约束

必须满足基本上下顺序：

```text
y_i + dy_i > y_i-1 + dy_i-1
```

不允许对应窗口大规模交叉。

## 14.4 连续性

使用鲁棒惩罚，不使用无限硬门限。

建议：

```text
Huber
```

一阶项限制突然跳变。

二阶项限制曲率突然变化。

遇到重复横线产生一个货架间距的大跳时，动态规划应更倾向于选择另一个候选或 missing。

## 14.5 无路径时的处理

如果因为数值问题没有合法路径：

- 不抛出整对 source 的异常。
- 将该 pair 的全部 `dy` 设置为 0。
- 标记：
  ```text
  pair_status = s1_no_valid_path
  ```
- 继续生成 S0 等价结果。

---

# 15. 平滑曲线

## 15.1 加权拟合

动态规划得到离散点后，拟合平滑 `Δy(y)`。

目标函数：

```text
高置信度观测误差
+ 一阶平滑
+ 二阶平滑
```

控制点间距初始：

```text
80 px
```

观测权重来自候选置信度。

missing 区域观测权重为 0。

## 15.2 无观测区域

无观测区域：

- 在上下可靠观测之间平滑插值。
- 顶部或底部只有单侧可靠观测时，逐渐减弱。
- 长距离无观测时，逐渐回到 0 修正。
- 不能外推成不断增大的位移。

## 15.3 斜率安全

限制局部映射斜率，保证纵向映射单调。

如果局部斜率过大：

- 不拒绝整条曲线。
- 局部缩小修正幅度。
- 必要时把该局部 gain 降为 0。

---

# 16. 局部 warp 增益选择

## 16.1 候选增益

每个高度块测试：

```text
0.00
0.25
0.50
1.00
```

高度块默认：

```text
64 px
```

## 16.2 评价代价

在预计 owner 边界附近的窄 ROI 中，比较 warp 前后：

- 横线和边缘跨边界的纵向差。
- 纵向梯度残差。
- 亮度 ZNCC。
- 双边缘响应。
- 形变量惩罚。

选择代价最低的 gain。

## 16.3 gain 连续性

相邻高度块的 gain 加入小的平滑代价，避免：

```text
1.0, 0.0, 1.0, 0.0
```

快速跳变。

## 16.4 gain 全部为零

这是正常结果。

当某 pair 的全部 gain 为零时：

- S1 等价于 S0。
- 仍输出 pair 报告。
- 不触发 hard cut。
- 不终止全景。

---

# 17. warp 作用范围

## 17.1 对称修正

第一版使用对称 pair warp。

对一个 pair 的 `Δy(y)`：

- 左 source 在其右匹配肩中应用一半修正。
- 右 source 在其左匹配肩中应用相反方向的一半修正。
- 两侧都向各自 owner 中央区域平滑衰减为零。

这样可以减少单侧 source 的明显拉伸。

## 17.2 x 方向衰减

使用 cosine fade。

接近 owner 边界时权重最大。

离开边界后，在约 64 px 内降为零。

如果实际匹配肩不足 64 px，按实际宽度缩放。

## 17.3 source 同时参与左右 pair

一个中间 source 可能同时具有：

- 左侧 pair 的修正。
- 右侧 pair 的修正。

两种修正分别只在 source 的左右肩中生效。

source 中央 owner 区域保持零修正。

最终把左右修正合并到同一张 inverse map。

## 17.4 全分辨率只 remap 一次

禁止：

```text
先 remap S0
再 remap S1
再 remap blend
```

正确方式：

1. 生成基础 inverse map。
2. 把 S1 canvas 纵向修正组合到 inverse map。
3. 得到最终 inverse map。
4. 每个 source 全分辨率 remap 一次。
5. 使用 owner map 合成 S1。

---

# 18. 失败和降级规则

## 18.1 允许终止的情况

只有以下情况允许整个任务失败：

- 输入目录不存在。
- manifest、frames.csv 或 calibration 文件无法解析。
- 有效真实帧少于 2。
- 图像文件全部无法读取。
- 标定参数非法。
- 无法找到任何单向扫描段。
- 最终 inverse map 出现不可恢复的 NaN 或越界。
- 内存分配失败。
- 输出主图无法写盘。

## 18.2 必须局部降级的情况

以下情况不得终止：

| 情况 | 降级方式 |
|---|---|
| 单条运动边不可靠 | 使用局部速度预测并加密 source |
| source 间推进过大 | 选择相邻真实帧 |
| 公共 overlap 较窄 | 缩小匹配肩 |
| overlap 不足以做 S1 | 当前 pair 使用 dy=0 |
| 某个窗口无纹理 | 选择 missing |
| 左右一致性差 | 降低候选置信度 |
| RANSAC 失败 | 不使用软先验 |
| DP 无合法路径 | 当前 pair 全部 dy=0 |
| 平滑曲线局部斜率过大 | 局部降低 gain |
| warp 后残差变差 | 当前高度块 gain 降为 0 |
| 一侧 inverse map 无效 | 使用 S0 owner 或相邻真实 source |
| 诊断图写入失败 | 主图继续输出，报告诊断写入错误 |

## 18.3 禁止的退化方式

- 不得让整对 source 直接切换成 v6.1 式 hard owner 模式。
- 不得删除这对 source。
- 不得重新运行整条重型流程。
- 不得因为一个局部窗口失败而修改其他高度区域。
- 不得因为一个 pair 失败而修改其他 pair。

---

# 19. 输出目录

每个数据集输出：

```text
benchmarks/<run_name>/S01_output_first_vertical_alignment_v1/
```

目录结构：

```text
S01_output_first_vertical_alignment_v1/
├─ s0_panorama_owner_only.png
├─ s0_panorama.png
├─ s1_panorama_owner_only.png
├─ s1_panorama.png
├─ owner_map.png
├─ valid_mask.png
├─ source_layout.json
├─ source_selection.csv
├─ overlap_summary.csv
├─ pair_summary.csv
├─ performance.json
├─ report.json
├─ config_snapshot.yaml
├─ pair_diagnostics/
│  ├─ pair_0000/
│  │  ├─ report.json
│  │  ├─ overlap_left.png
│  │  ├─ overlap_right.png
│  │  ├─ candidate_cost.png
│  │  ├─ topk_candidates.csv
│  │  ├─ dy_raw.csv
│  │  ├─ dy_curve.csv
│  │  ├─ dy_curve_overlay.png
│  │  ├─ confidence.png
│  │  ├─ gain.png
│  │  ├─ before_boundary_crop.png
│  │  ├─ after_boundary_crop.png
│  │  └─ edge_residual_comparison.json
│  └─ ...
└─ debug/
   ├─ cumulative_motion.csv
   ├─ source_centres.png
   └─ owner_boundary_overlay.png
```

所有诊断输出不得参与算法决策。

---

# 20. report.json

至少包含：

```json
{
  "schema": "gemini305-video-s1-experiment-report/v1",
  "algorithm_id": "S01_output_first_vertical_alignment_v1",
  "input": "",
  "output": "",
  "source_count": 0,
  "pair_count": 0,
  "s0_generated": true,
  "s1_generated": true,
  "complete_panorama_generated": true,
  "source_selection": {
    "method": "cumulative_canvas_motion",
    "fixed_every_n_frames": false,
    "source_frame_ids": [],
    "step_px_p5": 0.0,
    "step_px_p50": 0.0,
    "step_px_p95": 0.0,
    "adjacent_frame_rescue_count": 0
  },
  "overlap": {
    "width_p5": 0.0,
    "width_p50": 0.0,
    "width_p95": 0.0,
    "pair_below_s1_minimum_count": 0
  },
  "s1": {
    "pair_full_correction_count": 0,
    "pair_partial_correction_count": 0,
    "pair_zero_correction_count": 0,
    "window_observation_count": 0,
    "window_missing_count": 0,
    "gain_1_fraction": 0.0,
    "gain_05_fraction": 0.0,
    "gain_025_fraction": 0.0,
    "gain_0_fraction": 0.0,
    "dy_abs_p50_px": 0.0,
    "dy_abs_p95_px": 0.0
  },
  "quality_comparison": {
    "edge_step_before_p50_px": 0.0,
    "edge_step_before_p95_px": 0.0,
    "edge_step_after_p50_px": 0.0,
    "edge_step_after_p95_px": 0.0,
    "relative_p95_improvement": 0.0,
    "double_edge_change": 0
  },
  "performance": {},
  "fatal_quality_gate": false
}
```

`relative_p95_improvement` 只用于比较，不得决定是否发布主图。

---

# 21. 性能记录

使用现有性能计时工具或等价工具，分阶段记录：

```text
session_load
scan_analysis
source_selection
trajectory_or_layout
inverse_map_build
s0_render
feature_precompute
candidate_generation
left_right_check
ransac_prior
dynamic_programming
curve_smoothing
gain_selection
final_inverse_map_compose
final_full_resolution_remap
s1_compose
diagnostic_export
total
```

另外记录：

- 原始帧数。
- 最终 source 数。
- pair 数。
- 每个 pair 的处理时间。
- 最慢 10 个 pair。
- 全分辨率 remap 次数。
- 每个 source 是否只 remap 一次。
- CPU 线程数。
- GPU 是否仅用于 remap。
- 诊断输出耗时和主处理耗时分开统计。

性能不达目标时仍必须输出图像。

---

# 22. 单元测试

## 22.1 source 选择

构造合成运动序列：

1. 慢速稳定运动。
2. 快速稳定运动。
3. 速度突然变化。
4. 少量反向抖动。
5. 某些运动边不可靠。
6. 单帧推进大于目标 step。

检查：

- source ID 单调。
- 第一帧和最后一帧保留。
- 快速数据 source 更密。
- 不使用固定每 N 帧。
- 风险区域自动加密。
- 无重复 source。
- 不生成虚拟 source。

## 22.2 owner 区域

检查：

- owner 区域覆盖完整 canvas 有效范围。
- owner 无重叠。
- owner 无空洞。
- owner 边界位于 source 中心中点附近。
- 匹配肩允许重叠。
- 第一和最后 source 正确覆盖端点。

## 22.3 Top-K 候选

合成包含重复横线的图像：

- 正确纵向位移为 2 px。
- 货架周期为 20 px。
- 局部存在多个高相关峰。

检查：

- Top-K 中包含正确候选。
- 不只保留单个最佳候选。
- 候选之间满足 NMS。
- 重复峰的唯一性置信度降低。

## 22.4 左右一致性

构造：

- 正确匹配。
- 遮挡区域。
- 错误周期匹配。

检查正确候选置信度高于错误周期候选。

## 22.5 动态规划

构造每个高度有以下候选：

```text
正确连续路径
错误货架周期路径
随机跳变路径
missing
```

检查：

- 选择正确连续路径。
- 局部遮挡时使用 missing。
- 不出现几十像素无原因跳变。
- 没有合法路径时返回零修正，而不是抛出全局异常。

## 22.6 曲线拟合

检查：

- 观测区域拟合误差合理。
- missing 区域平滑插值。
- 长距离无观测时逐渐回零。
- 局部斜率保持安全。
- 不产生 NaN。

## 22.7 warp

检查：

- 接缝附近修正最大。
- 离开接缝后降为零。
- source 中央区域保持不变。
- 左右 pair 修正不会在中央叠加。
- inverse map 保持有限。
- 全分辨率只 remap 一次。

## 22.8 局部降级

强制以下失败：

- 无纹理。
- overlap 过窄。
- RANSAC 失败。
- DP 失败。
- gain 全部为零。

检查主全景仍然生成。

---

# 23. 合成回归测试

建立小型合成测试集：

## 场景 A

规则横线，真实 `dy=+2 px`。

预期：

- S1 恢复接近 2 px。
- 横线跨 owner 边界阶梯明显降低。

## 场景 B

随高度变化：

```text
y=0     dy=0
y=120   dy=1.5
y=240   dy=0.5
y=360   dy=-1
y=480   dy=-0.5
```

预期：

- 恢复连续曲线。
- 不使用常数平移代替。

## 场景 C

重复横线周期 24 px。

预期：

- 动态规划不跳到相邻周期。
- Top-K 中可存在错误候选，但最终路径连续。

## 场景 D

中间 96 px 高区域被遮挡。

预期：

- 遮挡区域选择 missing 或低 gain。
- 上下区域仍正常修正。
- 不让整对 source 退化。

## 场景 E

完全无纹理墙面。

预期：

- `dy=0`。
- S1 等价于 S0。
- 全景正常输出。

---

# 24. 真实数据实验

## 24.1 慢速数据

运行：

```powershell
& 'D:\Panoramic_Camera\.conda\Scripts\g305-video-s1-experiment.exe' `
  'D:\central_strip_Panoramic_Camera\data\captures\video\run_20260806_153033' `
  --config 'D:\central_strip_Panoramic_Camera\configs\video_candidates\S01_output_first_vertical_alignment_v1.yaml' `
  --output 'D:\central_strip_Panoramic_Camera\benchmarks\run_20260806_153033\S01_output_first_vertical_alignment_v1'
```

## 24.2 快速数据

运行：

```powershell
& 'D:\Panoramic_Camera\.conda\Scripts\g305-video-s1-experiment.exe' `
  'D:\central_strip_Panoramic_Camera\data\captures\video\run_20260807_140140' `
  --config 'D:\central_strip_Panoramic_Camera\configs\video_candidates\S01_output_first_vertical_alignment_v1.yaml' `
  --output 'D:\central_strip_Panoramic_Camera\benchmarks\run_20260807_140140\S01_output_first_vertical_alignment_v1'
```

如 CLI 需要复用轨迹，可以增加现有工程兼容的：

```text
--reuse-online-trajectory
```

或：

```text
--trajectory-cache <path>
```

但不得把 ORB、Open3D 或轨迹质量变成 S1 的画质 gate。

---

# 25. 真实数据必须检查的区域

固定导出以下类型区域的 S0 和 S1 对比：

- 长货架横线。
- 管道。
- 墙面边缘。
- 门框。
- 灭火器边缘。
- 物体内部 owner 交接位置。
- 重复横线密集区。
- 近景结构。
- 快速数据 overlap 最小的 pair。
- S1 gain 为零的 pair。
- S1 修正最大的 pair。

每种至少导出一个局部 crop。

---

# 26. 第一版完成标准

Codex 完成任务时必须同时满足：

## 26.1 功能

- 新 CLI 可以运行。
- 慢速数据生成 S0 和 S1。
- 快速数据生成 S0 和 S1。
- S0 与 S1 使用相同 source 布局。
- S1 失败时主图仍生成。
- 不调用 v6 或 v6.1 renderer。
- 不使用 GraphCut。
- 不使用 MultiBand。
- 不使用 mesh。
- 不使用 DIS 形变。
- 不使用 RAFT。
- 不使用 RGB-D 精细纹理。
- 不使用 TSDF。

## 26.2 诊断

- source 布局可视化存在。
- 每对 source 有 overlap 报告。
- 每对 source 有 `Δy(y)` 曲线。
- 有候选、置信度和 gain 记录。
- 有 S0 与 S1 局部对比。
- 有完整性能报告。
- 可以明确看出 S1 改善了哪些区域，未改善哪些区域。

## 26.3 稳定性

- 任意单个 pair 的 S1 失败不会终止。
- 任意单个高度窗口失败不会影响其他高度窗口。
- 不出现 NaN warp。
- 不出现 inverse map 翻折。
- 不出现 owner 空洞。
- 不出现 source ID 逆序。
- 不出现虚拟 RGB source。

## 26.4 画质观察目标

以下是实验目标，不是阻止出图的 gate：

- 横线阶梯 P95 相对 S0 有明显下降。
- 目标可以先设为相对下降 30%。
- 双边缘数量不能明显增加。
- 不能通过模糊让指标下降。
- 物体内部不能出现新的明显拉弯。
- 快速数据不能因为固定条带宽度产生大面积无 overlap。

---

# 27. 性能目标

性能目标只用于优化，不得阻止输出。

在复用已有轨迹或只统计图像后处理时，建议目标：

```text
3 m 数据总后处理时间尽量控制在 10 至 15 秒内
```

重点观察：

- S1 候选生成是否成为主要耗时。
- 诊断图写盘是否占用大量时间。
- source 数量是否过多。
- 是否重复解码同一 RGB。
- 是否重复计算同一 source 特征。
- 是否执行了多次全分辨率 remap。

如果未达到性能目标，优先优化：

1. 缓存 source 特征。
2. 只处理真实 overlap。
3. 批量计算纵向候选。
4. pair 并行。
5. 关闭非必要诊断图。
6. 减少水平子窗口数量。
7. 调整动态 source 密度。

不能先删除 S1 再保留宽融合。

---

# 28. 实施顺序

## 任务 1

建立新 CLI、配置、报告结构和输出目录。

此时先输出空白测试报告，不修改旧路线。

## 任务 2

实现累计 canvas 运动、动态 source 选择、动态 owner 边界和匹配肩。

输出：

```text
source_selection.csv
source_layout.json
owner_boundary_overlay.png
```

## 任务 3

实现 S0 owner-only 和 1 px 过渡版本。

先在两组真实数据上确认完整出图。

## 任务 4

实现 source 特征缓存和 pair overlap 提取。

输出 overlap 统计，不做 S1。

## 任务 5

实现二维窗口、多个水平子窗口和 Top-K 纵向候选。

先通过合成重复横线测试。

## 任务 6

实现左右一致性、置信度和 missing 状态。

## 任务 7

实现 RANSAC 软先验和动态规划。

先只输出离散 `dy` 路径。

## 任务 8

实现加权平滑曲线和局部 gain 选择。

## 任务 9

实现左右匹配肩的对称 warp，并组合到最终 inverse map。

## 任务 10

实现全分辨率一次 remap 和 S1 输出。

## 任务 11

补齐诊断、单元测试、合成回归测试和两组真实数据报告。

每完成一个任务都应保持测试通过，不要等全部模块完成后再统一调试。

---

# 29. Codex 最终交付内容

Codex 最终回复必须列出：

1. 新增文件。
2. 修改文件。
3. 新 CLI 的使用方法。
4. S0 的实际实现方式。
5. S1 候选代价和动态规划方式。
6. 局部降级如何保证始终出图。
7. 慢速数据结果路径。
8. 快速数据结果路径。
9. 两组数据的 source 数量和 overlap 分布。
10. S0 与 S1 的横线阶梯对比。
11. 各阶段耗时。
12. 所有测试结果。
13. 仍然存在的问题。
14. 下一步是否值得进入 S2。

不得只回复“代码已完成”。

---

# 30. 本版本的核心判断

第一版的重点不是一次做到最终画质。

本次只验证三个问题：

1. 动态 source 和动态 owner 是否能同时适应慢速与快速采集。
2. 一维 `Δy(y)` 是否能稳定减少横线、管道和墙面边缘的纵向阶梯。
3. 局部匹配失败是否可以只影响小区域，而不会让整对 source 或整幅全景退化。

只有这三项得到明确结果后，才进入下一版：

```text
S2 单路径可弯曲 seam
```

在进入 S2 前，不要加入更重算法。
