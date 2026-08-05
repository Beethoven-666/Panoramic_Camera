# Panoramic_Camera Fast V2 最终修改方案
## 画质恢复、端到端 CUDA 常驻与 20 m / 60 s 目标合并版

**项目：** `Beethoven-666/Panoramic_Camera`  
**代码基线：** `b07b561d03f2ddd85dcf0c3834ded8ec11c777ae`（2026-08-05）  
**测试输入：** `run_20260804_162340`  
**目标硬件：** Gemini 305，848×480@60 FPS，RTX 5060 Laptop GPU  
**方案日期：** 2026-08-05  
**方案性质：** 可分阶段落地的正式工程方案，不再受“RGB 不允许形变”“每个渲染帧必须有原始 ORB 节点”等旧约束限制。

---

# 0. 最终结论

当前版本已经把 3 m 数据的后处理时间降到 **19.461 s**，说明现有加速工作有效；但最新输出出现明显的水平结构断裂、前景物体切碎和锯齿 owner 边界，根本原因不是轨道或 ORB-SLAM3，而是：

1. 渲染源从 53 个进一步减少到 36 个，普通关键帧推进阈值提高到 20 px；
2. 风险检测仍然把全部边判为普通边，`high_risk_edge_count=0`；
3. 35 个 pair 的 DIS 光流只作为接缝代价，不用于 RGB 形变；
4. 最终仍是顺序 pairwise 的可弯曲 hard-owner；
5. RGB-D 分层 mesh、局部形变、对象锁和 MultiBand 均未运行；
6. CuPy remap 每次都将 NumPy 数据上传，再立刻 `cp.asnumpy()` 下载，CUDA 没有形成常驻链；
7. Open3D 每个 pair 重新创建 CUDA Tensor，并在每条边后同步和下载；
8. 全局颜色校正连续映射到 source 0，最终最大通道增益达到约 1.968，存在明显扫描方向漂移。

因此，下一版不能继续以“减少源帧 + hard-owner”换速度。正式方案应改为：

> **以 ORB-SLAM3 轨迹作为低频稳定先验，以 60 FPS 稠密图像运动决定全景推进；选中的真实 RGB-D 帧只上传一次并常驻 GPU；使用 RAFT 和 RGB-D 分层网格修正局部视差；对完整前景对象锁定单一 owner；使用受约束的可弯接缝和安全背景 MultiBand；最终只下载一次全景和 provenance。**

建议新增 `fast_v2` renderer，保留当前 `visual_seam` 作为回退，而不是继续在旧 `calibrated_rgb_pushbroom` 中叠加更多分支。

---

# 1. 最新版本的实测基线

## 1.1 时间

最新报告：

| 阶段 | 时间 |
|---|---:|
| 配置与会话读取 | 1.130 s |
| 扫描分析 | 1.001 s |
| ORB-SLAM3 | 7.447 s |
| 关键帧选择 | 0.001 s |
| Open3D 35 条边 | 5.400 s |
| 标定渲染与交付 | 4.482 s |
| **总计** | **19.461 s** |

当前 3 m 已满足设置的 20 s 预算，但按距离简单线性放大：

```text
19.461 × 20 / 3 ≈ 129.7 s
```

因此当前结构仍未达到 20 m / 60 s。

## 1.2 CUDA 使用

| 项目 | 当前结果 |
|---|---:|
| GPU | RTX 5060 Laptop |
| `G305_CUDA` | `prefer` |
| CuPy 调用 | 144 |
| Open3D CUDA 调用 | 35 |
| OpenCV CUDA 调用 | 0 |
| 统一适配层 CPU 调用 | 0 |
| Host→Device | 579,109,720 B，约 552 MiB |
| Device→Host | 126,757,600 B，约 121 MiB |

这证明 CUDA 实际启用，但仍是“按操作调用 CUDA”，不是常驻链。

## 1.3 渲染结构

| 指标 | 当前结果 |
|---|---:|
| 运动分析帧 | 384 |
| ORB 跟踪帧 | 66 |
| 最终渲染源 | 36 |
| 相邻 pair | 35 |
| 高风险边 | 0 |
| normal step | 20 px |
| risk step | 8 px |
| 曲线 hard-owner pair | 35 |
| GraphCut pair | 0 |
| blend pixels | 0 |
| deformation pixels | 0 |
| 深度参与局部几何 | 否 |
| 全分辨率 remap | 36 |

## 1.4 DIS 可靠性实际上很低

35 条最终接缝的 `reliable_flow_fraction`：

```text
最小值：0.252
中位数：0.469
最大值：0.625
低于 0.5：24 / 35
低于 0.6：33 / 35
低于 0.7：35 / 35
```

也就是说，当前风险分类器把全部边判为低风险，但最终 overlap 中没有任何一条边达到 70% 可靠对应。

## 1.5 Owner 拓扑质量

从 `video_pixel_provenance.npz` 读取：

| 指标 | 当前结果 |
|---|---:|
| panorama owner map | 1830×456 |
| owner 数量 | 36 |
| owner 连通组件总数 | 85 |
| 被分裂为多个组件的 owner | 26 / 36 |
| 单 owner 最大组件数 | 6 |
| frame 318 最终像素 | 374 |
| frame 443 最终像素 | 39 |

这说明顺序 pairwise 合成产生了大量碎片 owner。frame 318 和 frame 443 几乎没有贡献，却仍完成了读取、ORB、Open3D、remap、光流和接缝计算。

## 1.6 光度漂移

全局颜色求解虽然被判定为 accepted，但：

```text
global_gain_max = 1.9684
最后一帧 gain BGR ≈ [1.651, 1.530, 1.968]
```

这不是正常的小范围颜色统一，而是长序列相邻校正累积形成的近 2 倍通道放大。

---

# 2. 最新全景的视觉问题及成因

## 2.1 黄色横梁和桌面出现阶梯断裂

全景中黄色横梁、顶部货架板、桌面前沿存在大量横向台阶。它们与 provenance 中高度弯曲、跨越几十到上百像素的 owner 边界一致。

当前接缝允许相邻行移动最多 4 px，但没有足够的二阶曲率约束、长直线约束和对象约束。对 456 行图像而言，一条 seam 可以在上下方向累计偏移上百像素。

## 2.2 红色瓶子和细线物体被切碎

红色瓶子、黑色线缆、塑料袋边缘被多个时间视角切割。接缝虽然“可弯”，但没有对象级锁定，所以会沿局部灰度低差路径直接穿过物体。

## 2.3 右侧立柱附近出现大面积几何断层

右侧黑色立柱处同时包含：

- 深度突变；
- 近景立柱；
- 远处墙面；
- 黄色横梁；
- 多个相邻时间视角。

当前没有 mesh warp，也没有 depth layer，因此一条 hard seam 无法同时满足这些深度层。

## 2.4 风扇比旧版完整，但仍然偏暗且局部不连续

新版本减少了风扇被等宽窄条切割的问题，但只是因为关键帧更稀疏、接缝路径变化，并不代表已经实现对象锁。只要接缝成本稍变，风扇仍可能再次被穿过。

## 2.5 结构 A 级不等于视觉 A 级

当前 A 级主要检查：

- owner 完整；
- 输出有真实来源；
- 光流证据存在；
- 总时间未超标。

它没有检查：

- 对象内部是否出现 seam；
- owner 是否碎片化；
- 长直线是否跳变；
- 是否真正执行形变和融合；
- 全局 gain 是否异常；
- 接缝可靠率是否足够。

正式报告应拆分为：

```text
structural_grade
visual_grade
performance_grade
```

不能再用一个 A 同时表示三者。

---

# 3. 需要修改的总体架构

```text
848×480@60 FPS RGB-D
        │
        ├── 采集期：低分辨率 CUDA 图像运动
        ├── 采集期：ORB-SLAM3 8～12 FPS 稳定轨迹
        ├── 采集期：曝光、清晰度、深度风险统计
        ▼
60 FPS 累计视觉扫描坐标
        ▼
风险感知真实关键帧选择
        ▼
5 帧 GPU 常驻窗口
        ├── GPU 标定逆映射
        ├── RAFT-small
        ├── RGB-D 深度置信与分层
        ├── 分层 mesh 求解
        ├── 前景对象跟踪与 owner 锁定
        ├── 可弯曲 seam / 多标签 owner
        └── 线性 RGB MultiBand
        ▼
CUDA tile / panorama accumulator
        ▼
一次 Device→Host
        ▼
JPEG/PNG + provenance + 摘要报告
```

---

# 4. 正式技术决策

## 4.1 Fast V2 以 PyTorch CUDA 作为主 GPU 运行时

建议主链从 CuPy 切换为 PyTorch CUDA，原因：

- RAFT-small 原生位于 Torchvision；
- `grid_sample` 可同时完成标定 remap 和 mesh warp；
- 卷积、金字塔、梯度、颜色变换和 MultiBand 都能在同一 Tensor 域完成；
- CUDA Stream、Event、Graph、AMP 均可直接使用；
- Open3D 和 CuPy 可通过 DLPack 与 Torch Tensor 共享显存。

现有 CuPy `remap()` 保留给：

```text
legacy / audit / CPU 对照测试
```

不作为 `fast_v2` 的主图像接口。

## 4.2 Open3D 不再阻塞 fast 主图发布

当前 35 条 Open3D 边耗时 5.4 s，但没有改变最终像素。

Fast V2 改为：

```text
L0/L1 普通 pair：不在主路径运行 Open3D
L2/L3 高风险 pair：按需运行 Open3D CUDA
完整全边审计：全景发布后异步执行
```

如果项目仍要求发布前覆盖所有边，应实现 GPU 帧缓存与 DLPack 零拷贝，不能继续每条边重新上传两帧。

## 4.3 ORB 位姿改为“稳定先验”，允许时间插值

旧约束要求每个渲染源必须拥有直接 ORB pose，导致 ORB 8 FPS 时渲染候选也只能来自 8 FPS。

新约束：

- RGB 帧必须是真实采集帧；
- ORB 节点仍是真实跟踪节点；
- 非 ORB 时间戳可以使用 SE(3) 插值作为初始 prior；
- 最终推进和局部 warp 由 60 FPS 图像运动、RAFT 和 RGB-D 决定；
- 报告明确记录 `pose_prior_origin=orb_interpolated`，不得写成 direct ORB pose。

在轨道近似匀速、姿态平稳的情况下，这比继续减少渲染源更合理。

---

# 5. 第一阶段：立即修复当前画质退化

这一阶段不等待完整 CUDA 常驻链，即可先消除最严重的阶梯和碎片。

## 5.1 降低关键帧跨度

将：

```yaml
normal_target_step_pixels: 20
risk_target_step_pixels: 8
maximum_step_pixels: 24
emergency_step_pixels: 30
```

改为：

```yaml
normal_target_step_pixels: 12
risk_target_step_pixels: 5
maximum_step_pixels: 16
emergency_step_pixels: 20
```

说明：

- 20 px 对未形变的 hard-owner 太大；
- 普通背景 12 px；
- 深度突变、前景和低可靠 flow 区域 5 px；
- 仍比旧版逐帧渲染少得多。

## 5.2 风险判断改为逐 pair 二阶段判断

新增：

```python
@dataclass(frozen=True)
class PairRiskRecord:
    left_frame_id: int
    right_frame_id: int
    motion_reliable: bool
    flow_reliable_fraction: float
    flow_fb_p95: float
    near_depth_fraction: float
    depth_edge_fraction: float
    multimodal_flow_score: float
    line_step_p95: float
    object_crossing_score: float
    risk_level: int
```

第一阶段在 60 FPS 缩略图上预判；第二阶段在已摆放 overlap 上重新判断。

建议阈值：

```text
flow_reliable_fraction < 0.70 → 至少 L1
flow_reliable_fraction < 0.50 → L2
flow_fb_p95 > 1.5 px          → L2
near_depth_fraction > 0.05    → L1/L2
depth_edge_fraction > 0.08    → L2
object_crossing_score > 0     → L3
line_step_p95 > 1.5 px        → L2
```

当前 35 条 pair 全部低于 0.7，因此不应继续得到 `high_risk_edge_count=0`。

## 5.3 接缝增加曲率和直线约束

当前只有：

```text
|seam[y] - seam[y-1]| × step_penalty
```

增加：

```text
一阶项：|x_y - x_{y-1}|
二阶项：|(x_y - x_{y-1}) - (x_{y-1} - x_{y-2})|
水平长线切断项
垂直长线切断项
源图中心距离项
最小 owner 宽度项
```

建议：

```yaml
maximum_row_step_pixels: 2
step_penalty: 15
curvature_penalty: 24
minimum_owner_run_pixels: 8
seam_corridor_half_width_pixels: 48
```

不能让 seam 在整个 169 px overlap 内任意游走，应以几何 owner boundary 为中心限定走廊。

## 5.4 Owner 碎片清理

接缝后执行：

1. 统计每个 owner 连通组件；
2. 小于 128 px 的孤立组件并入邻接 owner；
3. 一个 source 最终少于全景 `0.05%` 像素时，移除该 source；
4. 在局部窗口重新求 seam；
5. 对象锁区域不允许被并入错误 owner。

本次 frame 318 和 443 应在此阶段被自动移除。

---

# 6. 第二阶段：建立真正的 CUDA 常驻数据链

## 6.1 新增 GPU 数据类型

```python
@dataclass
class GpuVideoFrame:
    frame_id: int
    timestamp_us: int
    color_u8: torch.Tensor          # CUDA, H×W×3
    color_linear: torch.Tensor      # CUDA FP16, 3×H×W
    depth_mm: torch.Tensor          # CUDA FP32, 1×H×W
    depth_valid: torch.Tensor       # CUDA bool
    pose_prior: torch.Tensor        # CUDA/CPU 4×4
    sharpness: float
    exposure_score: float
```

```python
@dataclass
class GpuRenderWindow:
    frames: tuple[GpuVideoFrame, ...]
    upload_done: torch.cuda.Event
    flow_done: torch.cuda.Event
    warp_done: torch.cuda.Event
```

## 6.2 使用 pinned host ring

新增固定大小的 pinned buffer：

```text
color slot：848×480×3 uint8
depth slot：848×480 uint16/float32
metadata slot
```

使用：

```python
tensor.cuda(non_blocking=True)
```

环形容量：

```text
fast：8～12 帧
quality：12～20 帧
```

## 6.3 不再上传 map_x / map_y

当前每次 remap 会上传：

```text
source + map_x + map_y
```

Fast V2 在 GPU 中直接生成采样 grid：

1. 缓存虚拟输出像素射线；
2. 使用相机 pose 将射线/点变换到源相机；
3. 在 CUDA Tensor 中执行 Brown-Conrady 投影；
4. 转换到 `grid_sample` 的 `[-1, 1]` 坐标；
5. 同一 grid 同时采样 RGB、depth 和 valid mask。

这样每个源只上传原始 RGB-D 一次。

## 6.4 三条 CUDA Stream

```text
upload_stream
compute_stream
output_stream
```

流程：

```text
upload_stream：
    下一窗口 H2D

compute_stream：
    当前窗口 remap / RAFT / mesh / seam / blend

output_stream：
    已完成 tile 转 uint8 并 D2H
```

通过 Event 建立依赖，不在每个 pair 后执行全设备 `synchronize()`。

## 6.5 目标传输量

本次 36 源当前为：

```text
H2D ≈ 552 MiB
D2H ≈ 121 MiB
```

Fast V2 的 3 m 目标：

```text
H2D < 100 MiB
D2H < 15 MiB
```

原则：

```text
每个源 RGB-D 上传一次
中间结果不下载
只下载最终 panorama、owner map 和小型审计标量
```

---

# 7. 第三阶段：CUDA 光流

## 7.1 60 FPS 推进运动

不需要对 60 FPS 全序列运行完整 RAFT。

新增 `video_cuda_motion.py`：

- 输入 212×120 或 424×240；
- 图像分成 8×4 tile；
- 每个 tile 在 CUDA 上计算水平相位相关或粗到细局部相关；
- 排除深度边缘、饱和区和低纹理 tile；
- 对背景 tile 的 dx 做稳健中位数；
- 输出累计 scan coordinate。

这部分在采集期间执行，不占停止后的 60 s。

## 7.2 RAFT-small 用于渲染 pair

新增 `video_raft.py`：

```python
class RaftSmallCuda:
    def warmup(self) -> None: ...
    def estimate(
        self,
        left: Tensor,
        right: Tensor,
        *,
        backward: bool,
    ) -> FlowResult: ...
```

建议：

```yaml
input_width: 424
input_height: 240
dtype: float16
batch_pairs: 4
forward_all_pairs: true
backward_policy: risk_only
```

第一版：

- 所有 render pair 计算 forward RAFT；
- L2/L3 计算 backward RAFT；
- L0/L1 用 forward photometric residual和邻帧一致性；
- 如果实测预算允许，再改为全双向。

模型权重必须：

- 离线预置；
- 写入 SHA-256；
- 启动时 warm-up；
- 不在正式运行时联网下载。

## 7.3 Flow 可信度

```text
forward-backward error
photometric residual
gradient residual
深度可见性
局部 Jacobian
源图范围
```

输出：

```python
@dataclass
class FlowResult:
    forward: Tensor
    backward: Tensor | None
    confidence: Tensor
    occlusion: Tensor
    fb_error: Tensor | None
```

---

# 8. 第四阶段：RGB-D 分层网格

## 8.1 深度预处理

在已标定的 GPU support tile 上：

1. 0 值设 invalid；
2. 3 帧时域中值，仅在同一像素邻域和相近深度内；
3. 3×3 保边滤波；
4. 计算 depth confidence；
5. 计算 depth edge；
6. 不跨强边缘补洞。

## 8.2 三层深度

```text
far_background
mid_range
near_foreground
```

采用 log-depth 聚类，并加空间连通约束。

不能用一组固定绝对阈值覆盖所有场景；绝对范围仅用于过滤：

```text
200 mm ≤ depth ≤ 3000 mm
```

## 8.3 网格参数

| 参数 | fast | quality |
|---|---:|---:|
| 窗口帧数 | 5 | 7 |
| 普通 cell | 24 px | 16 px |
| 风险 cell | 16 px | 8～16 px |
| 最大位移 | 32 px | 48 px |
| 局部 scale | 0.75～1.35 | 0.70～1.40 |
| Jacobian | >0 | >0 |

## 8.4 优化目标

```text
E =
λ_flow  · 高置信 RAFT 对应误差
+ λ_depth · 同层 RGB-D 重投影误差
+ λ_pose  · ORB 稳定先验
+ λ_smooth · 网格一阶平滑
+ λ_arap · 局部形状保持
+ λ_line · 长直线保持
+ λ_time · 相邻窗口一致性
```

建议实现顺序：

### V1：固定迭代 GPU 平滑

- cell 内加权中位 flow；
- 顶点位移双线性分配；
- 20～30 次 Jacobi 平滑；
- 深度边缘不跨层传播。

### V2：稀疏共轭梯度

- 构建固定拓扑稀疏矩阵；
- Torch CUDA CG；
- 支持 line constraints 和 ARAP。

## 8.5 一次最终 warp

所有几何校正必须合并成一个最终 grid：

```text
标定畸变
+ pose prior
+ scan coordinate
+ layer mesh
```

最终 RGB 只执行一次 `grid_sample`，避免多次插值。

---

# 9. 第五阶段：前景对象锁定

## 9.1 第一版无需 FastSAM

候选对象由以下信息产生：

```text
近景深度连通域
+ depth edge
+ RGB 强边缘
+ RAFT 与背景 mesh 的残差
+ 时序 mask 连通
```

## 9.2 对象跟踪

使用：

- RAFT mask propagation；
- 3D centroid；
- median depth；
- bbox IoU；
- 外观直方图。

```python
@dataclass
class ForegroundTrack:
    track_id: int
    frame_masks: dict[int, Tensor]
    depth_median_mm: float
    owner_frame_id: int | None
    handoff_frame_ids: list[int]
```

## 9.3 最佳 source 评分

```text
score =
+ 完整可见率
+ 清晰度
+ 深度有效率
+ 接近图像中心
+ flow 置信度
- 遮挡率
- mesh 形变量
- 高光/暗部异常
```

## 9.4 锁定规则

- 对象内部只允许一个 owner；
- mask 外扩 8～12 px；
- seam 不得穿过；
- 对象内部不运行宽 MultiBand；
- 超出单帧 support 时允许一次受控 handoff；
- handoff 只能发生在高重叠、低梯度、稳定深度区域。

固定验收对象：

```text
红色瓶子
黑色风扇
黑色立柱
线缆
纸箱开口
```

---

# 10. 第六阶段：接缝与 MultiBand

## 10.1 不再顺序合成整幅旧 panorama

当前逻辑是：

```text
旧 panorama + incoming source
→ 一条 seam
→ 永久覆盖
```

会导致早期 owner 被后续帧不断切碎。

Fast V2 使用 5 帧局部窗口：

```text
source i-2, i-1, i, i+1, i+2
```

在共同 tile 中同时选择 label。

## 10.2 多标签代价

```text
data cost =
warp 后 Lab 残差
+ gradient magnitude 差
+ gradient orientation 差
+ flow uncertainty
+ depth visibility
+ source center distance
+ sharpness penalty
```

```text
smoothness =
颜色边缘感知 Potts
+ seam 一阶平滑
+ seam 二阶曲率
+ 长直线保护
```

```text
hard constraints =
对象锁
无效采样
遮挡层
时间 owner 单调顺序
```

## 10.3 分阶段实现

### 第一版

- pairwise seam；
- mesh 后求 seam；
- 增加曲率、对象锁和 owner 清理。

### 第二版

- 5 帧多标签；
- 1/2 分辨率求解；
- 原分辨率 ±4 px 局部细化。

## 10.4 MultiBand

仅在完成几何对齐后启用。

| 区域 | 策略 |
|---|---|
| 白墙、黄色平面、低风险背景 | 4 层 MultiBand，16～24 px |
| 轻微残差背景 | 8～16 px |
| 前景对象内部 | hard owner |
| 深度边缘 | 0～3 px feather |
| 不可信区域 | hard seam |

使用 Torch 实现 Gaussian/Laplacian pyramid，不调用 CPU OpenCV blender。

## 10.5 Tile 渲染

```yaml
tile_width: 2048
tile_overlap: 128
```

20 m 超宽图无需一次建立所有 source 的全景级 Tensor。

---

# 11. 第七阶段：全局颜色优化

## 11.1 取消 source 0 链式参考

当前所有 correction 递推到 source 0，容易把长序列误差累计到近 2 倍。

改为全局图优化：

```text
相邻 overlap
+ i↔i+2 overlap
+ 时间一阶平滑
+ 时间二阶平滑
+ gain 接近 1
+ bias 接近 0
```

参考域选择：

```text
全序列中位曝光帧
```

而不是第 0 帧。

## 11.2 限制

Fast 建议：

```yaml
global_gain_min: 0.75
global_gain_max: 1.35
global_bias_abs_max: 0.08
```

判级：

```text
gain > 1.35：不能给 visual A
gain > 1.50：拒绝帧级 correction，改用低频照明场或降级
```

## 11.3 低频照明场

对安全背景估计 64×64 或 96×96 block gain field，强平滑，用于：

- 镜头暗角；
- 场景照明渐变；
- 局部墙面亮度变化。

对象内部不使用块级光照场。

## 11.4 处理顺序

```text
sRGB decode
→ linear RGB
→ global gain/bias
→ mesh warp
→ seam / MultiBand
→ tone mapping
→ sRGB encode
```

---

# 12. Open3D CUDA 重构

## 12.1 当前问题

当前 `_estimate_pair_tensor_cuda()` 每个 pair：

1. 从 NumPy 创建 source CUDA Tensor；
2. 从 NumPy 创建 reference CUDA Tensor；
3. 创建 intrinsic Tensor；
4. 创建 initial Tensor；
5. 创建 criteria 和 loss；
6. 运行 odometry；
7. 全局 synchronize；
8. 下载 transformation；
9. 再将两张 depth 建成 CPU Tensor；
10. CPU 计算 information matrix。

因此报告不能证明 frame buffer 在 pair 之间复用。

## 12.2 Fast 推荐策略

```yaml
open3d_fast_policy: risk_only
open3d_full_audit_after_publish: true
```

主发布前：

- 只对 L2/L3 pair 执行 Open3D；
- 普通 pair 使用 ORB prior + RAFT/depth mesh。

发布后：

- 后台跑完整 35/更多条边；
- 原子更新 `geometry_audit_state`；
- 不撤销已经发布的 panorama，除非发现结构性错误。

## 12.3 必须保留全边审计时

新增：

```python
class Open3DGpuFrameCache:
    def put_from_torch(self, frame_id, color, depth): ...
    def get(self, frame_id): ...
    def evict_before(self, frame_id): ...
```

通过 DLPack：

```text
Torch CUDA Tensor
→ Open3D Tensor.from_dlpack
```

零拷贝共享同一显存。

缓存：

- source/reference RGB-D；
- intrinsic；
- criteria；
- loss；
- device。

仅保持滑动窗口 3 帧。

## 12.4 Information matrix

Open3D 0.19 当前代码路径中 information matrix 仍在 CPU。

Fast 主路径改为：

- 使用 CUDA odometry 的 fitness、RMSE；
- 与 ORB prior 的 translation/rotation residual；
- depth support ratio。

完整 6×6 information matrix延后到 audit。

---

# 13. 端到端主流程伪代码

```python
def render_fast_v2(session, online_state, config):
    runtime = TorchCudaRuntime(config.cuda)
    runtime.warmup()

    poses = load_or_run_orb_prior(session, online_state)
    dense_pose_prior = interpolate_se3_to_real_frames(poses, session.timestamps)

    motion = load_online_motion_or_compute_cuda(session)
    candidates = select_visual_keyframes(
        session.frames,
        motion,
        dense_pose_prior,
        normal_step=12,
        risk_step=5,
    )

    pair_plans = preclassify_pair_risk(candidates, motion, session.depth)

    photometric_graph = collect_safe_overlap_statistics_gpu(
        candidates,
        pair_plans,
    )
    color_solution = solve_global_photometric_graph(photometric_graph)

    writer = TorchCudaTileWriter(
        width=planned_width,
        height=480,
        tile_width=2048,
        tile_overlap=128,
    )

    for window in sliding_windows(candidates, size=5):
        gpu_frames = runtime.upload_once(window)

        calibrated = runtime.calibrated_remap(window, gpu_frames)
        flows = runtime.raft_pairs(calibrated, pair_plans)
        layers = build_depth_layers(calibrated.depth)
        meshes = solve_layered_mesh(
            flows=flows,
            layers=layers,
            poses=dense_pose_prior,
        )

        warped = runtime.warp_once(
            calibrated,
            meshes,
            color_solution,
        )

        objects = track_and_lock_foreground(
            warped,
            layers,
            flows,
        )

        labels = solve_multilabel_seam(
            warped,
            objects,
            pair_plans,
        )

        tile = multiband_safe_background(
            warped,
            labels,
            objects,
            levels=4,
        )

        writer.commit(tile)

    panorama_gpu, owner_gpu = writer.finish()
    panorama = runtime.download_final(panorama_gpu)
    owner = runtime.download_final(owner_gpu)

    publish_panorama(panorama, owner)
    schedule_full_open3d_audit_if_required()
```

---

# 14. 文件级修改清单

## 14.1 修改

### `src/panorama_demo/video_panorama.py`

- 新增 `fast_v2`；
- 移除全局 `fast_visual_use_depth`；
- 使用逐 pair `PairPlan`；
- 允许真实帧使用插值 ORB prior；
- Open3D 改为 risk-only 或 post-publish；
- 调用新 GPU renderer；
- 增加性能预算控制。

### `src/panorama_demo/video_motion_resampler.py`

- 接收深度和 flow 风险；
- normal step 20 → 12；
- risk step 8 → 5；
- 从“ORB tracking frame 子集”改为“所有真实帧候选 + ORB prior”；
- 选择目标进度附近质量最好的真实帧。

### `src/panorama_demo/cuda_backend.py`

- 保留 NumPy 返回 API 给 legacy；
- 增加明确告警：该接口不是常驻链；
- 不再被 `fast_v2` 使用。

### `src/panorama_demo/rgbd_odometry.py`

- 增加 DLPack GPU frame cache；
- 拆分 CUDA odometry 与 CPU information matrix；
- risk-only 快速策略；
- 去除 pair 级全局 synchronize；
- 增加 stream/event 审计。

### `src/panorama_demo/video_photometric.py`

- source-0 链式校正改为图优化；
- 加入 i↔i+2 边；
- 加入时间一阶/二阶正则；
- 中位曝光帧作为 anchor；
- 收紧 gain 上限；
- 增加低频 block field。

### `src/panorama_demo/video_delivery.py`

- 区分结构、视觉、性能等级；
- 主图先发布；
- 完整 Open3D audit 可后补；
- 保存 renderer version、模型哈希和 GPU 审计。

### `configs/demo.yaml`

新增 `fast_v2` 配置，见第 16 节。

### `pyproject.toml`

正式依赖组：

```toml
visual-cuda = [
  "torch",
  "torchvision",
  "kornia",
  "scipy",
]
```

版本必须锁定到实机 CUDA 环境。

### `AGENTS.md`

更新正式视频契约：

- 允许 SE(3) prior 插值；
- 允许局部 RGB 非刚性形变；
- 允许 safe-background MultiBand；
- 不再要求 fast 发布前完成全部 Open3D 信息矩阵；
- 仍要求 RGB 来自真实帧，owner/provenance 可追溯。

## 14.2 新增

```text
src/panorama_demo/
├── video_visual_renderer_v2.py
├── video_gpu_runtime.py
├── video_gpu_frame_cache.py
├── video_cuda_motion.py
├── video_raft.py
├── video_pair_risk.py
├── video_depth_layers.py
├── video_layered_mesh.py
├── video_object_tracks.py
├── video_multilabel_seam.py
├── video_torch_multiband.py
├── video_photometric_graph.py
├── video_visual_quality.py
├── video_budget_controller.py
└── open3d_gpu_frame_cache.py
```

---

# 15. 关键接口

```python
@dataclass(frozen=True)
class PairPlan:
    left_frame_id: int
    right_frame_id: int
    risk_level: int
    use_raft_backward: bool
    use_depth_mesh: bool
    use_open3d: bool
    object_lock_required: bool
    seam_mode: str
    blend_mode: str
```

```python
@dataclass
class GpuWarpedSource:
    frame_id: int
    rgb_linear: torch.Tensor
    depth_mm: torch.Tensor
    valid: torch.Tensor
    confidence: torch.Tensor
    source_grid: torch.Tensor
```

```python
@dataclass
class VisualQualityMetrics:
    line_step_p95: float
    seam_delta_e_p95: float
    object_internal_seam_count: int
    owner_component_count: int
    minimum_owner_pixels: int
    mesh_fold_count: int
    maximum_photometric_gain: float
```

---

# 16. 建议配置

```yaml
stitch:
  video_panorama:
    default_preset: fast_v2
    maximum_post_seconds: 60

    orb_prior:
      target_fps: 8
      allow_se3_interpolation: true
      validate_on_held_out_orb_nodes: true
      maximum_interpolation_translation_error_mm: 5
      maximum_interpolation_rotation_error_deg: 0.5

    motion_resampling:
      minimum_step_pixels: 3
      normal_target_step_pixels: 12
      risk_target_step_pixels: 5
      maximum_step_pixels: 16
      emergency_step_pixels: 20

    cuda_runtime:
      backend: torch
      device: cuda:0
      mixed_precision: true
      window_frames: 5
      pinned_ring_frames: 12
      upload_streams: 1
      compute_streams: 1
      output_streams: 1
      maximum_vram_bytes: 5000000000
      final_download_only: true

    cuda_motion:
      enabled: true
      width: 212
      height: 120
      tiles_x: 8
      tiles_y: 4
      backend: phase_correlation
      cpu_dis_fallback: true

    raft:
      enabled: true
      model: raft_small
      model_path: models/raft_small_C_T_V2.pth
      model_sha256: REQUIRED
      width: 424
      height: 240
      batch_pairs: 4
      forward_all_pairs: true
      backward_policy: risk_only
      fp16: true

    pair_risk:
      medium_flow_reliable_fraction: 0.70
      high_flow_reliable_fraction: 0.50
      maximum_flow_fb_p95: 1.5
      near_depth_fraction: 0.05
      depth_edge_fraction: 0.08
      maximum_line_step_p95: 1.5

    depth_layers:
      enabled: true
      minimum_depth_mm: 200
      maximum_depth_mm: 3000
      layer_count: 3
      temporal_radius_frames: 1
      edge_guard_pixels: 8

    mesh:
      enabled: true
      normal_cell_pixels: 24
      risk_cell_pixels: 16
      maximum_displacement_pixels: 48
      minimum_scale: 0.70
      maximum_scale: 1.40
      minimum_jacobian: 0.05
      jacobi_iterations: 24
      use_line_constraints: true

    object_lock:
      enabled: true
      minimum_component_pixels: 32
      guard_pixels: 10
      maximum_owner_handoffs: 1
      semantic_backend: none

    seam:
      backend: local_multilabel
      window_frames: 5
      preview_scale: 0.5
      corridor_half_width_pixels: 48
      maximum_row_step_pixels: 2
      step_penalty: 15
      curvature_penalty: 24
      minimum_owner_run_pixels: 8
      use_line_constraints: true
      use_object_locks: true

    photometric:
      backend: global_graph
      linear_rgb: true
      global_gain_minimum: 0.75
      global_gain_maximum: 1.35
      global_bias_abs_maximum: 0.08
      temporal_first_order_weight: 1.0
      temporal_second_order_weight: 2.0
      block_field_enabled: true
      block_size_pixels: 64

    multiband:
      enabled: true
      levels: 4
      background_width_pixels: 24
      foreground_width_pixels: 2
      tile_width_pixels: 2048
      tile_overlap_pixels: 128

    open3d:
      fast_policy: risk_only
      full_audit_after_publish: true
      dlpack_zero_copy: true
      cache_frames: 3
      defer_information_matrix: true

    delivery:
      export_central_strips: false
      export_owner_only: false
      report_level: summary
      save_owner_map: true
```

---

# 17. 性能预算

以下是工程目标，必须通过实机 profiler 验证。

## 17.1 3 m

| 阶段 | 目标 |
|---|---:|
| 会话与在线状态复用 | 0.5～1.0 s |
| ORB flush / prior | 0～0.5 s |
| decode + 单次 H2D | 0.8～1.5 s |
| RAFT | 1.0～2.0 s |
| depth layers + mesh | 0.8～1.5 s |
| object + seam | 0.8～1.5 s |
| MultiBand + final D2H | 1.0～2.0 s |
| 发布 | 0.3～0.8 s |
| **总计** | **5～10 s** |

## 17.2 20 m

| 阶段 | 目标 |
|---|---:|
| 在线状态结束处理 | 1～2 s |
| decode + upload | 6～10 s |
| RAFT | 8～15 s |
| depth mesh | 6～10 s |
| object + seam | 6～10 s |
| MultiBand | 8～12 s |
| 发布 | 2～4 s |
| **总计** | **37～63 s** |

为了稳定达到 60 s，正常目标应设为 50～55 s。

---

# 18. 超时降载顺序

不能优先牺牲对象完整性。

正确降载顺序：

1. backward RAFT 从全部 pair 改为 L2/L3；
2. mesh normal cell 24 → 32 px；
3. MultiBand 4 → 3 层；
4. 跳过最终 straightening；
5. 普通 step 12 → 14 px；
6. Open3D 全边审计移到发布后。

不得优先关闭：

```text
对象锁
深度边缘保护
owner 拓扑清理
前景 hard owner
```

---

# 19. 质量判级

## 19.1 Structural Grade

检查：

- 输入完整；
- ORB prior 有效；
- owner 覆盖完整；
- provenance 可追溯；
- mesh 无翻折。

## 19.2 Visual Grade

A 级建议：

| 指标 | 阈值 |
|---|---:|
| 横梁接缝跳变 P95 | <1 px |
| 桌面边缘跳变 P95 | <1 px |
| safe background ΔE P95 | <3 |
| 对象内部 seam | 0 |
| 对象主 owner 数 | 1 |
| owner 最大连通组件数 | ≤2 |
| owner 碎片 <128 px | 0 |
| photometric max gain | ≤1.35 |
| mesh fold | 0 |
| flow FB P95 | <1.5 px |

## 19.3 Performance Grade

```text
3 m ≤ 10 s
10 m ≤ 30 s
20 m ≤ 55 s
```

最终 delivery 记录：

```json
{
  "structural_grade": "A",
  "visual_grade": "A",
  "performance_grade": "A",
  "overall_grade": "A"
}
```

---

# 20. 回归测试

## 20.1 固定 ROI

必须保存：

```text
red_bottle_roi
fan_roi
right_pillar_roi
yellow_beam_roi
table_edge_roi
```

## 20.2 自动指标

红瓶：

```text
内部 owner 数 = 1
内部 seam 数 = 0
轮廓断裂 < 2 px
```

风扇：

```text
主体 owner 数 = 1
外圆最大断裂 < 2 px
格栅无明显双影
```

横梁：

```text
接缝处纵向跳变 P95 < 1 px
梯度方向突变 < 5°
```

Owner：

```text
总 connected components <= owner_count × 2
任一 owner 小于 0.05% 时自动清理
```

## 20.3 CUDA 常驻验收

报告必须新增：

```text
source_upload_count
source_reuse_count
torch_cuda_flow_calls
torch_grid_sample_calls
torch_multiband_calls
dlpack_share_count
cuda_stream_count
peak_vram_bytes
h2d_bytes
d2h_bytes
intermediate_d2h_count
```

A 级要求：

```text
intermediate_d2h_count = 0
每个 source H2D 次数 = 1
D2H 只包括最终图、owner map和标量
```

---

# 21. 实施顺序

## 阶段 A：1～2 天

- normal step 20 → 12；
- 增加真实 pair 风险；
- 深度开关改为逐 pair；
- seam 曲率、直线和 corridor 限制；
- owner 碎片清理；
- 增加 visual grade。

目标：立刻消除当前最严重的锯齿和大面积阶梯。

## 阶段 B：3～5 天

- 新建 Torch CUDA runtime；
- GPU 内生成标定 grid；
- 每帧上传一次；
- GPU 线性 RGB；
- GPU final tile；
- 传输审计。

目标：

```text
3 m H2D <100 MiB
3 m D2H <15 MiB
```

## 阶段 C：3～7 天

- RAFT-small；
- batch 和 FP16；
- flow confidence；
- pair risk 与 RAFT 联动。

## 阶段 D：5～10 天

- depth layers；
- residual mesh；
- grid_sample 一次 warp；
- long-line constraints。

## 阶段 E：5～10 天

- object track；
- owner lock；
- 受控 handoff；
- 5 帧多标签 seam。

## 阶段 F：3～5 天

- Torch MultiBand；
- 全局颜色图优化；
- block illumination field。

## 阶段 G：3～5 天

- Open3D DLPack cache；
- risk-only / post-publish；
- 20 m 实机压力测试；
- SLA 自动降载。

---

# 22. 回退策略

```text
Fast V2 全功能
→ RAFT + mesh，无 MultiBand
→ DIS/RAFT + 对象锁 + hard seam
→ 当前 visual_seam
→ legacy hard-owner
```

任何回退必须在报告中明确记录。

不能因为一个 cell 失败，让整个 pair、整段或全序列回退。

---

# 23. 最终推荐

最重要的不是继续减少 render source，而是让每个保留 source 真正被充分利用。

当前 19.46 s 版本的合理下一步是：

1. 立即把 20 px normal step 降回 12 px；
2. 修复 `high_risk_edge_count=0`；
3. 将深度和 flow 风险改为逐 pair；
4. 新建 Torch CUDA 常驻 renderer；
5. 用 RAFT + RGB-D mesh 在接缝前对齐；
6. 以对象锁阻止红瓶、风扇和线缆被切开；
7. 只在安全背景使用 MultiBand；
8. 全局颜色改为图优化；
9. Open3D 全边审计移出 fast 主发布路径；
10. 用 provenance 拓扑、长直线和对象完整性决定 visual grade。

**不能再把“处理时间更短”建立在 `blend_pixel_count=0`、`deformation_pixel_count=0` 和更稀疏关键帧的基础上。**

最终 Fast V2 应真正实现：

```text
RGB 上传一次
→ GPU 标定 remap
→ GPU RAFT
→ GPU RGB-D depth layers
→ GPU mesh warp
→ GPU object/seam
→ GPU MultiBand
→ 最终下载一次
```

---

# 24. 依据与参考

## 项目与测试依据

- GitHub commit：`b07b561d03f2ddd85dcf0c3834ded8ec11c777ae`
- `video_report(2).json`
- `video_delivery(1).json`
- `video_pixel_provenance.npz`
- `video_panorama(1).png`

## 官方实现参考

- Torchvision `raft_small`
- PyTorch CUDA Stream / Event / CUDA semantics
- Open3D Tensor DLPack interface
- CuPy DLPack and memory pool documentation
