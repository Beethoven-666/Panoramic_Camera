# Panoramic_Camera 最终视频全景修改方案  
## 视觉质量与“20 m 采集结束后 1 分钟内出图”合并版

**项目：** `Beethoven-666/Panoramic_Camera`  
**目标硬件：** Gemini 305、848×480、最高 60 FPS、i7、16 GB 内存、RTX 5060 Laptop  
**方案版本：** 1.0  
**日期：** 2026-08-04

---

## 0. 执行摘要

项目的正式视频方案统一改为：

> **以 ORB-SLAM3 位姿作为稳定先验，以 60 FPS 稠密图像运动决定全景推进，以 RAFT 和 RGB-D 分层网格修正局部视差，再使用可弯曲接缝、前景对象锁定、全局颜色优化和 MultiBand 完成最终全景。**

为了同时满足视觉质量与速度，不能把 RAFT、RGB-D 网格、GraphCut 和完整审计无条件应用到所有原始帧。最终系统应采用一套共享架构，并提供三种计算策略：

| 预设 | 用途 | 停止采集后的目标 | 主要策略 |
|---|---|---:|---|
| `fast` | 正式默认快速交付 | 20 m：30～60 s | 全序列廉价运动；仅困难区域使用 RAFT 和深度网格 |
| `quality` | 更高视觉质量 | 20 m：约 90～180 s | 更密渲染帧、更高 RAFT 覆盖率、更细网格与整平 |
| `audit` | 开发、复现、诊断 | 不设时限 | 全量报告、中央条带、owner-only、全部中间结果 |

三种预设必须共享相同的数据结构、坐标体系、颜色域、位姿和渲染实现，只改变采样密度、重型算法覆盖率、网格分辨率和调试输出，避免形成三套难以维护的代码。

**一分钟目标成立的前提：**

1. ORB-SLAM3、低分辨率运动估计和关键帧选择在采集期间同步执行；
2. 60 FPS 全部用于跟踪和运动分析，但只有经过运动重采样的关键帧参与最终渲染；
3. CUDA 路径真实启用，并让图像在显存中完成重映射、光流、网格采样和融合；
4. `fast` 模式不阻塞式输出数千张中央条带 PNG，也不生成每个 pair 的完整审计；
5. 3 m 快速基准必须先达到停止采集后 **7～8 s**，否则 20 m 一分钟目标没有足够余量。

---

# 1. 当前基线与问题诊断

## 1.1 当前性能基线

最新 3 m 测试中：

| 指标 | 当前结果 |
|---|---:|
| 原始采集帧 | 598 帧 |
| 最终 renderer 源帧 | 387 帧 |
| 总处理时间 | 324.120 s |
| ORB-SLAM3 时间 | 31.171 s |
| CPU 调用计数 | 7345 |
| OpenCV CUDA 调用 | 0 |
| CuPy 调用 | 0 |
| Open3D CUDA 调用 | 0 |
| 输出画布 | 1973×480 |
| 最终裁剪 | 1932×460 |

按距离线性外推：

```text
当前速度 = 324.120 s / 3 m
         ≈ 108.04 s/m

20 m 离线全流程 ≈ 2160.8 s
                ≈ 36.0 min

压到 60 s 所需总体加速 ≈ 36.0 倍
```

同样采集密度下，20 m 大约产生：

```text
原始帧约 3987 帧
现有逻辑下渲染源约 2580 帧
```

因此，不能继续让每个真实帧都经历完整 Open3D、光度审计、接缝审计、全分辨率 remap 和双套 PNG 导出。

## 1.2 当前渲染链的主要浪费

现有正式视频入口大致执行：

```text
扫描段分析
→ 保留所有真实帧
→ 采集结束后运行 ORB-SLAM3
→ 对所有相邻帧运行 Open3D RGB-D odometry
→ calibrated_rgb_pushbroom
→ 输出 central_strips
→ 输出 central_strips_owner_only
→ 发布全景和完整报告
```

当前报告显示：

| 项目 | 当前结果 |
|---|---:|
| residual model | `identity` |
| identity hard-owner fallback | 197 对 |
| RGB-D geometry 触发 | 98 对 |
| RGB-D geometry 接受 | 0 对 |
| 最终 blend pixels | 0 |
| 最终 deformation pixels | 0 |
| analysis preview remap | 387 次 |
| geometry/preview remap | 387 次 |
| full-resolution output remap | 387 次 |

即同一批源帧至少经历：

```text
387 + 387 + 387
= 1161 次主要重映射
```

但最终局部形变和融合都没有真正进入输出。大量计算只用于证明“不能修改”，随后回退为硬 owner。

## 1.3 当前视觉结构的问题

历史风扇样例中，连续源帧的 owner-only 宽度约为：

```text
36、36、35、35、34 px
```

这会让一台近景风扇被五个不同观察位置的竖条切开。当前问题不是“羽化不够”，而是：

1. 全景推进主要由轨迹换算和固定 owner 区间控制；
2. owner 接近整高竖切；
3. 前景对象没有真正锁定到单一来源；
4. 局部 APAP/光流默认关闭；
5. 一处 held-out 失败容易让整对图像退化到 identity；
6. 深度只参与窄接缝风险判断，不参与最终局部采样坐标修正；
7. `blend_pixel_count=0`、`deformation_pixel_count=0`，最终接近“很多窄条硬切”。

## 1.4 当前代码中的关键限制

### `video_panorama.py`

当前流程在采集结束后依次执行：

- `analyse_video_scan`
- `select_video_render_sources`
- `run_orbslam3_rgbd`
- 每对相邻源执行 `estimate_pair_rgbd_odometry`
- `render_calibrated_rgb_pushbroom`
- 强制暂存并发布两套中央条带目录

这使轨迹、全部相邻 RGB-D 边和调试导出全部位于关键路径。

### `video_source_selection.py`

当前函数明确“保留每一个真实节点”，不进行运动重采样。这保证了旧方案的审计完整性，但使长距离视频的计算量按帧数线性增长。

### `capture_orbbec.py`

当前每帧独立写入：

- 一张 JPEG；
- 一张 aligned-depth PNG；
- 可选 raw-depth PNG；
- CSV 行；
- 每个文件进行原子替换和 `fsync`。

这种逐帧小文件布局适合可审计采集，但不适合快速长距离视频。

### `configs/demo.yaml`

当前正式配置强调：

- 真实节点全部保留；
- 中央支持区最大约 20%；
- 不插值位姿；
- local APAP/flow 默认关闭；
- 颜色样本不得生成；
- 形变 fail-closed；
- 2～8 px 的安全背景融合。

这些约束与现在“允许视觉形变、追求手机式自然效果”的目标冲突。

### `pyproject.toml`

PyTorch 和 Torchvision 只存在于可选诊断依赖中。新的正式视觉 renderer 需要独立、明确的可选依赖组，而不能继续把 RAFT 伪装成诊断功能。

---

# 2. 最终产品定义

## 2.1 正式输出产品

默认运行 `fast` 预设，停止采集后优先发布：

```text
video_panorama.jpg
video_report.json
video_delivery.json
video_pixel_provenance.npz（精简或分块）
```

不阻塞主结果的内容：

```text
central_strips/
central_strips_owner_only/
pair_debug/
flow_debug/
mesh_debug/
seam_debug/
完整 pair audit
```

它们只在以下条件生成：

```text
--preset audit
--export-central-strips
--export-owner-only
--report-level full
```

## 2.2 时间定义

“20 m 一分钟出图”必须定义为：

> 从用户停止采集，到 `video_panorama.jpg` 和摘要报告原子发布完成，不超过 60 秒。

采集期间运行的在线轨迹和运动估计不计入后处理时间，因为用户本身正在行走。以 0.30 m/s 行走 20 m，采集持续约 66.7 s，这段时间应被充分用于计算，而不是空等。

## 2.3 视觉目标

在静态货架、墙面、农业巡检等场景中：

- 接缝肉眼不明显；
- 风扇、果串、软管、线缆、纸箱等近景物体不被多条竖向 owner 切碎；
- 横梁、桌边、管道等长直线在接缝处无明显台阶；
- 墙面和货架无周期性竖向亮度块；
- 允许局部非刚性形变，以视觉连续性优先；
- 保留真实帧 provenance，但不要求每个输出像素完全不变形。

---

# 3. 统一总体架构

```text
┌─────────────────────────────────────────────────────────────┐
│                    848×480 @ 60 FPS 采集                     │
└────────────────────────────┬────────────────────────────────┘
                             │
                  ┌──────────▼──────────┐
                  │ 在线环形缓冲和批量写盘 │
                  └──────────┬──────────┘
                             │
          ┌──────────────────┼───────────────────┐
          │                  │                   │
┌─────────▼────────┐ ┌───────▼────────┐ ┌────────▼─────────┐
│ 低分辨率图像运动 │ │ ORB-SLAM3 在线 │ │ 曝光/清晰度/深度 │
│ 60 FPS           │ │ 15～30 FPS      │ │ 风险统计         │
└─────────┬────────┘ └───────┬────────┘ └────────┬─────────┘
          └──────────────────┼───────────────────┘
                             │
                  ┌──────────▼──────────┐
                  │ 运动重采样关键帧选择 │
                  └──────────┬──────────┘
                             │
                  ┌──────────▼──────────┐
                  │ 轨迹平滑与视觉推进量 │
                  └──────────┬──────────┘
                             │
              ┌──────────────▼──────────────┐
              │ pair 风险分级：普通 / 困难   │
              └──────────────┬──────────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
   ┌──────────▼─────────┐        ┌──────────▼─────────┐
   │ DIS/相位相关/局部仿射 │        │ RAFT + RGB-D 分层网格 │
   │ 普通 pair           │        │ 困难 pair           │
   └──────────┬─────────┘        └──────────┬─────────┘
              └──────────────┬──────────────┘
                             │
                  ┌──────────▼──────────┐
                  │ 前景对象锁定与可弯接缝 │
                  └──────────┬──────────┘
                             │
                  ┌──────────▼──────────┐
                  │ 全局颜色优化        │
                  └──────────┬──────────┘
                             │
                  ┌──────────▼──────────┐
                  │ CUDA MultiBand 分块 │
                  └──────────┬──────────┘
                             │
                  ┌──────────▼──────────┐
                  │ 低频直线整平与裁剪  │
                  └──────────┬──────────┘
                             │
                  ┌──────────▼──────────┐
                  │ 原子发布与摘要报告  │
                  └─────────────────────┘
```

核心原则：

1. **ORB-SLAM3 只提供稳定先验，不再独占最终像素布局。**
2. **最终横向推进以稠密图像运动为主，轨迹只提供尺度、方向和异常约束。**
3. **深度负责分层、遮挡、可见性和对应点权重，不负责生成颜色。**
4. **普通区域走廉价路径，困难区域才进入 RAFT 和细网格。**
5. **一个局部失败只降级对应网格单元或 pair，不能把整段序列全部退化为 identity。**
6. **最终 owner 是二维 label map 或逐行 seam，不再是每帧一个整高 x 区间。**
7. **所有预设共享同一 renderer，只改变策略参数。**

---

# 4. 采集端修改

## 4.1 正式分辨率与帧率

正式视频模式设置为：

```yaml
capture:
  width: 848
  height: 480
  fps: 60
```

60 FPS 的价值不是把 60 帧全部做高成本渲染，而是：

- 降低相邻帧基线；
- 提供稳定光流；
- 提高遮挡交接密度；
- 允许按图像运动自适应选择关键帧。

## 4.2 扫描速度控制

推荐目标：

```text
最近有效前景的帧间运动：4～8 px/frame
警告阈值：>10 px/frame
强制提示减速：>14 px/frame
```

采集界面实时显示：

```text
TOO SLOW   <3 px/frame
GOOD       4～8 px/frame
FAST       8～10 px/frame
TOO FAST   >10 px/frame
STOP       >14 px/frame
```

建议初始物理速度：

```text
0.25～0.35 m/s
```

实际控制应以图像运动为准，不以物理速度为唯一标准。

## 4.3 曝光、增益和白平衡

预热后必须固定：

- 自动曝光状态；
- 曝光时间；
- 模拟/数字增益；
- 自动白平衡状态；
- 白平衡值；
- Gamma；
- 锐化；
- 对比度和饱和度。

推荐流程：

```text
启动
→ 自动曝光/白平衡预热 30～60 帧
→ 读取收敛值
→ 关闭自动控制
→ 写回固定值
→ 连续验证 3～5 帧 metadata
→ 正式录制
```

如果某项无法写锁，应在报告中标为降级，而不能静默继续。

## 4.4 写盘结构

### 第一阶段：低风险改造

继续保持逐帧文件格式，但修改：

- JPEG 质量从 95 提升到 98；
- depth PNG compression 使用 0 或 1；
- CSV 每 30 帧 flush；
- 不再每一帧执行磁盘 `fsync`，只在 chunk 完成和会话结束时 fsync；
- 颜色与深度写入独立队列；
- 队列满时优先保留在线处理，明确记录 dropped frame。

### 第二阶段：长距离优化

新增 chunk/container 格式：

```text
color_0000.mkv      FFV1 或高质量 MJPEG
depth_0000.bin.zst  连续 uint16 chunk
frames_0000.jsonl   metadata
```

每个 chunk 约 120～300 帧。这样可以减少数千个小文件、目录操作和随机读取。

## 4.5 在线环形缓冲

新增：

```python
class OnlineFrameRing:
    capacity_frames: int
    color: pinned-memory buffers
    depth: pinned-memory buffers
    metadata: fixed-size records
```

建议容量：

```text
fast：120 帧
quality：240 帧
```

选为 render keyframe 的帧才进入长期缓存；其他帧在完成运动统计后即可释放。

---

# 5. ORB-SLAM3 在线化与轨迹处理

## 5.1 当前问题

当前 ORB-SLAM3 在采集结束后重新读取所有源帧。仅 3 m 测试就需要约 31.171 s，按 20 m 线性放大会超过 3 分钟，因此一分钟目标要求 ORB-SLAM3 移到采集期间。

## 5.2 新的流式 runner

新增 ORB-SLAM3 C++ 可执行程序：

```text
Examples/RGB-D/rgbd_stream_headless
```

输入可以采用 stdin：

```text
timestamp color_path depth_path depth_scale
```

或者共享内存/命名管道。输出每帧：

```text
frame_id timestamp tx ty tz qx qy qz qw tracking_state
```

Python 新增：

```python
class OnlineORBBridge:
    start()
    submit(frame_id, timestamp, color_ref, depth_ref)
    poll_poses()
    finish()
```

要求：

- 无 Pangolin；
- 不创建 GUI 线程；
- 支持增量 flush；
- 进程异常时保留已输出轨迹；
- 采集结束时执行一次最终 shutdown/trajectory flush。

## 5.3 Tracking FPS

不必让 ORB-SLAM3 处理完整 60 FPS。

推荐：

| 模式 | ORB-SLAM3 输入 |
|---|---:|
| fast | 20 FPS |
| quality | 30 FPS |
| audit | 30～60 FPS |

未送入 ORB 的帧根据时间戳在 SE(3) 上插值，但只用于运动先验，不伪装成真实 ORB 节点。

## 5.4 轨迹平滑

原始 ORB 位姿保留；新增展示轨迹：

1. 计算相机中心；
2. 用稳健 PCA/IRLS 确定主扫描方向；
3. 主方向位置保持单调；
4. 垂直和前后抖动采用更强平滑；
5. 旋转使用四元数 Slerp 或 Lie algebra spline；
6. 对跟踪质量低的段增加轨迹先验权重。

建议输出两套：

```text
raw_camera_to_world
render_camera_to_world
```

轨迹只参与：

- 初始 warp；
- 搜索窗口；
- 图像运动尺度；
- 异常检测；
- 深度重投影。

最终全景推进不再简单等于：

```text
相机中心位移 × 全局 pixels_per_mm
```

---

# 6. 60 FPS 稠密图像运动与全景推进

## 6.1 全帧廉价运动

所有 60 FPS 帧在 1/4 或 1/2 分辨率计算廉价运动：

优先级：

1. OpenCV DIS Optical Flow；
2. 稀疏 LK + RANSAC；
3. 相位相关作为低纹理补充。

输出：

```python
@dataclass
class FrameMotionRecord:
    frame_id: int
    dx_median: float
    dy_median: float
    foreground_dx_p90: float
    flow_fb_p95: float
    valid_ratio: float
    sharpness: float
    exposure_score: float
    depth_edge_ratio: float
    risk_score: float
```

## 6.2 最终推进量

对每个 render keyframe：

```text
Δs = 稳健水平图像运动
```

融合轨迹先验：

```text
Δs_final =
    w_rgb × Δs_rgb
  + w_pose × scale_local × Δx_pose
```

其中：

- 纹理丰富、光流可信：提高 `w_rgb`；
- 低纹理墙面：提高 `w_pose`；
- 前景占比大：用背景层光流；
- 流向与轨迹方向冲突：标记困难 pair。

## 6.3 Render Keyframe 选择

60 FPS 帧分为：

```text
capture frames
tracking frames
render keyframes
heavy-processing pairs
```

推荐初值：

```yaml
motion_resampling:
  minimum_step_pixels: 3
  normal_target_step_pixels: 16
  risk_target_step_pixels: 8
  maximum_step_pixels: 24
  emergency_step_pixels: 30
```

普通背景每约 16 px 选择一帧；风扇、软管、果串、线缆等高风险区域降到约 8 px。

对于约 13 000 px 宽的 20 m 全景：

```text
普通区域约 600～900 个 render keyframes
近景密集区域约 900～1300 个 render keyframes
```

远低于按当前密度推算的约 2580 个源帧。

---

# 7. 宽支持区与一次重映射

## 7.1 支持区和最终贡献区分离

当前约 20% 的中央支持区限制了匹配、网格和绕缝空间。

新 renderer 使用：

```text
source support fraction：0.55～0.65
```

对 848 px 宽图像约为：

```text
466～551 px
```

这只是分析和采样支持区。每帧最终新增到全景中的宽度仍可能只有 8～20 px。

## 7.2 一次上传、一次标定重映射

每个 render keyframe：

```text
磁盘/环形缓冲
→ pinned host memory
→ 一次 H2D
→ 一次 CUDA 标定 inverse remap
→ 在 GPU 中生成金字塔
```

得到：

```text
full：最终渲染
1/2：接缝、RAFT
1/4：风险、DIS、颜色统计
```

不能再从源文件分别执行 analysis preview、geometry preview 和 final remap。

## 7.3 标定 map 缓存

启动时预计算：

```text
undistort_map_x
undistort_map_y
valid_mask
```

按分辨率和 calibration hash 缓存到：

```text
.cache/calibration/<sha256>/
```

---

# 8. Pair 风险分级与自适应重型算法

## 8.1 风险分数

每对相邻 render keyframe 计算：

```text
risk =
  深度边缘占比
+ 前后向光流误差
+ 图像残差
+ 长直线台阶
+ 近景占比
+ 遮挡/反遮挡占比
+ 曝光差
```

分为：

| 等级 | 条件 | 路径 |
|---|---|---|
| L0 | 平面背景、低残差 | 位姿 + DIS + 局部仿射 |
| L1 | 一般纹理和轻微视差 | DIS + APAP/粗网格 |
| L2 | 明显近景、遮挡、线缆 | RAFT-small + RGB-D 分层网格 |
| L3 | 不可信或运动物体 | 对象 owner 锁定 + 动态硬接缝 |

## 8.2 Fast 模式重型覆盖率

目标：

```text
RAFT / 精细 RGB-D 网格只覆盖 10%～25% pair
```

当预计覆盖率超过预算时：

1. 优先处理近景对象；
2. 合并连续高风险 pair 为一个窗口；
3. 普通墙面继续走廉价路径；
4. 必要时提高普通区 render step，而不是降低前景质量。

## 8.3 不允许整对 fail-closed

新退路必须按网格单元和对象区域生效：

```text
RAFT + depth mesh
→ RAFT mesh
→ DIS/APAP mesh
→ local affine
→ dynamic hard seam
```

某个局部区域失败，只让对应 cell 或对象退化，不能让整对、整段甚至全序列回退到 identity。

---

# 9. RAFT 稠密光流

## 9.1 正式后端

建议：

```text
torchvision.models.optical_flow.raft_small
```

参数：

| 项目 | fast | quality |
|---|---:|---:|
| 输入分辨率 | 424×240 | 848×480 或 424×240 |
| 方向 | 双向 | 双向 |
| 精度 | FP16 | FP16/FP32 |
| batch | 4～8 pair | 2～4 pair |
| 覆盖率 | 10%～25% | 60%～100% |

模型在进程启动时加载一次，并先执行 warm-up，避免第一对图像产生初始化延迟。

## 9.2 光流可信度

有效对应点必须同时满足：

```text
前后向一致性
亮度/梯度一致性
目标范围内
非深度边缘保护区
非遮挡/反遮挡
局部 Jacobian 合理
```

输出：

```python
@dataclass
class FlowResult:
    forward: Tensor
    backward: Tensor
    fb_error: Tensor
    confidence: Tensor
    occlusion: Tensor
```

RAFT 不直接无限制地 warp RGB，而只为网格拟合提供对应关系。

## 9.3 廉价后端

普通 pair 使用：

```text
DIS medium / ultrafast
```

并复用上一对的流作为初值。低纹理区域使用轨迹初始化和相位相关补充。

---

# 10. RGB-D 深度预处理与分层

## 10.1 深度角色

深度用于：

- 前景/背景层划分；
- 遮挡与反遮挡；
- 对应点过滤；
- 网格权重；
- 对象三维中心跟踪；
- 接缝禁止区。

深度不直接产生 RGB，也不把整幅全景强制重建为 TSDF。

## 10.2 深度预处理

每个 render keyframe：

1. 读取 aligned depth；
2. 乘 `depth_scale_mm_per_unit`；
3. 0 值设为 invalid；
4. 小范围时空中值；
5. 保边双边滤波；
6. 计算 depth confidence；
7. 计算 depth edge；
8. 不在大洞上无条件补深度。

## 10.3 分层策略

初始采用三层：

```text
far_background
mid_range
near_foreground
```

按 log-depth 聚类，并要求空间连通。深度无效区域通过 RGB 光流和周边层的置信度分类，但不得穿越强深度边缘。

---

# 11. RGB-D 分层网格形变

## 11.1 网格表示

每个相邻窗口建立规则 mesh：

| 参数 | fast | quality |
|---|---:|---:|
| cell size | 24 px | 16 px |
| window frames | 5 | 7 |
| max displacement | 32 px | 48 px |
| local scale | 0.75～1.35 | 0.70～1.40 |

网格坐标保留浮点，最终 RGB 只执行一次 `grid_sample/remap`。

## 11.2 优化目标

```text
E =
λ_flow   × 光流对应误差
+ λ_depth × 深度重投影误差
+ λ_pose  × ORB 位姿先验误差
+ λ_arap  × 网格 ARAP/平滑误差
+ λ_line  × 长直线保持误差
+ λ_temp  × 相邻窗口时间一致性
```

保护的长线包括：

- 黄色横梁；
- 桌面前沿；
- 立柱；
- 管道；
- 门框。

## 11.3 求解

第一版可采用：

- SciPy sparse least-squares；
- 每窗口一次稀疏求解；
- CUDA 负责图像采样。

后续可改为 Torch 共轭梯度和多窗口 batch。

## 11.4 APAP 的位置

APAP 不再是被禁止的实验后端，而作为：

- L1 普通视差 pair 的主方法；
- L2 网格的初始值；
- RAFT 失败时的退路。

它不直接决定最终接缝和 owner。

---

# 12. 前景对象锁定

## 12.1 第一版不依赖重型语义模型

`fast` 模式通过以下信息生成对象候选：

```text
近景深度连通域
+ depth edge
+ RGB edge
+ 光流与背景模型残差
+ 时序连通
```

可选 FastSAM 只在 `quality` 或显式启用时使用。

## 12.2 对象跟踪

对象跨帧关联使用：

- 光流 mask propagation；
- 3D centroid；
- 深度中位数；
- bbox overlap；
- 外观特征。

## 12.3 最佳来源评分

```text
score =
+ 清晰度
+ 完整可见比例
+ 深度有效率
+ 靠近镜头中心
+ 光流可信度
- 遮挡比例
- 网格形变量
- 曝光异常
```

## 12.4 Owner 锁定

对于风扇、果串、软管、纸箱等完整对象：

- 一个连续时间段内优先使用一个主 source；
- 对象 mask 外扩 8～16 px；
- seam 不得穿过锁区；
- 对象内部不做宽 MultiBand；
- 只有当对象离开当前 source 有效支持区时才允许一次受控 handoff。

---

# 13. 可弯曲接缝与二维 owner map

## 13.1 替换整高 owner 区间

旧表示：

```text
source i owns x0:x1 for all y
```

新表示：

```text
seam_x[y]
或
owner_label[y, x]
```

不同图像行可以在不同位置切换源帧。

## 13.2 接缝成本

```text
C =
w_color × Lab 差
+ w_grad × 梯度差
+ w_flow × 光流不确定性
+ w_depth × 深度边缘
+ w_object × 对象锁惩罚
+ w_center × 离源图中心距离
+ w_sharp × 清晰度损失
```

对象锁区域使用近似无穷大成本，禁止 seam 穿过。

## 13.3 Fast 实现

先在 1/2 分辨率执行单调动态规划：

```text
每行 seam 最大横移 2～3 px
```

再在原分辨率 ±4～8 px 范围局部细化。

## 13.4 Quality 实现

在 5～7 帧局部窗口内做多标签 GraphCut，输出 owner label map，再进行单调和对象完整性修正。

---

# 14. 全局颜色优化

## 14.1 线性 RGB 联合求解

每帧颜色模型：

```text
I_out = gain_rgb × I_linear + bias_rgb
```

所有 render keyframe 一次联合优化。样本只来自共同可见背景、非深度边缘、非对象锁区和非高光暗区。

目标：

```text
重叠颜色一致
+ 参数随时间平滑
+ gain 靠近 1
+ bias 靠近 0
```

## 14.2 低频块级修正

墙面、横梁等低频亮度变化使用 64×64 或 96×96 块级增益场，加入强平滑，避免再次制造条带。

## 14.3 颜色处理顺序

```text
sRGB/JPEG 解码
→ 转线性 RGB
→ 全局 gain/bias
→ 网格 warp
→ seam/blend
→ 转回 sRGB
→ 最终轻量锐化
```

---

# 15. MultiBand 融合与最终整平

## 15.1 融合策略

| 区域 | fast | quality |
|---|---:|---:|
| 安全背景 | 4 层，16～24 px | 5～6 层，24～32 px |
| 轻微几何残差 | 8～16 px | 16～24 px |
| 前景对象内部 | hard owner | hard owner |
| 深度边缘 | 2～4 px feather | 2～4 px feather |
| 不可信区域 | 动态硬 seam | 动态硬 seam |

## 15.2 分块渲染

20 m 全景采用：

```text
tile width = 2048 px
overlap = 128 px
```

每块在 GPU 中完成 source warp、owner masks、Laplacian pyramid 和 blend。

## 15.3 最终整平

融合后执行低频全景 mesh：

```text
grid = 64 px
maximum displacement = 8～12 px
```

根据长线约束横梁水平、立柱垂直、桌面连续。`fast` 可按预算关闭，`quality` 默认启用。

---

# 16. CUDA 与内存加速设计

## 16.1 GPU 常驻路径

以下步骤不得反复 CPU↔GPU：

```text
decode/上传
→ undistort/remap
→ pyramid
→ DIS/RAFT
→ mesh grid_sample
→ seam cost
→ MultiBand
→ output tile
```

每个 keyframe 只上传一次。

## 16.2 后端选择

正式视觉路径以 PyTorch CUDA 为核心：

- RAFT；
- `grid_sample`；
- 图像金字塔；
- 梯度和颜色转换；
- mask 运算；
- 可选融合。

OpenCV CUDA 只作为可用时补充，不能依赖普通 `opencv-python` wheel 提供 CUDA。

## 16.3 显存预算

```text
fast 峰值显存 < 3.0 GB
quality 峰值显存 < 3.8 GB
```

方法：

- 5～7 帧窗口；
- 半分辨率 RAFT；
- FP16；
- 及时释放 flow pyramids；
- tile 渲染；
- 不缓存全序列 full-resolution GPU tensor。

## 16.4 CPU 与 I/O

- 读取线程预取下一个窗口；
- depth 顺序读取；
- 报告分阶段写临时文件；
- 调试输出异步或关闭；
- 不在主路径压缩数千张 PNG。

---

# 17. 在线和后处理任务划分

## 17.1 采集期间完成

```text
相机采集
低分辨率运动
清晰度/曝光统计
深度风险统计
ORB-SLAM3 20～30 FPS
轨迹增量写入
render keyframe 候选选择
颜色重叠统计
```

## 17.2 停止采集后完成

```text
轨迹最终平滑
最终 render keyframe 确认
pair 风险分级
普通 pair 轻量对齐
困难 pair RAFT + depth mesh
对象 owner 锁定
全局颜色求解
动态 seam
CUDA tile blend
整平、裁剪、JPEG
原子发布
```

---

# 18. 20 m 一分钟性能预算

以下是目标预算，不是当前实测保证：

| 阶段 | 目标 |
|---|---:|
| 在线轨迹 flush、轨迹平滑 | 1～2 s |
| 关键帧索引与预取 | 2～4 s |
| 普通 pair DIS/APAP | 5～8 s |
| 困难 pair RAFT/depth mesh | 8～15 s |
| 对象锁与 seam | 5～9 s |
| 全局颜色求解 | 2～4 s |
| CUDA warp/MultiBand | 7～12 s |
| 整平、编码、报告、发布 | 3～5 s |
| **总计** | **33～59 s** |

设计目标应为正常约 45 s、上限 55 s。

## 18.1 性能门槛

| 距离 | 停止采集后的 fast 目标 |
|---|---:|
| 3 m | ≤ 7～8 s |
| 5 m | ≤ 14 s |
| 10 m | ≤ 28 s |
| 20 m | ≤ 55 s |

## 18.2 超预算时的自适应策略

按以下顺序降载：

1. 普通区域 render step 从 16 增至 18～20 px；
2. RAFT 风险阈值提高，但前景对象 pair 不跳过；
3. RAFT 输入保持 424×240；
4. mesh cell 从 24 增至 32 px；
5. MultiBand 从 4 层降至 3 层；
6. 关闭 fast 模式最终整平；
7. 保留动态 seam 和对象锁，不优先牺牲前景完整性。

---

# 19. 代码修改清单

## 19.1 修改现有文件

### `src/panorama_demo/capture_orbbec.py`

新增：

- 848×480@60 正式 profile；
- 在线环形缓冲；
- 低分辨率运动 worker；
- 在线 ORB bridge；
- 预热后完整颜色锁定；
- 批量 flush/fsync；
- chunk writer；
- 实时速度等级；
- 在线轨迹和 motion manifest。

### `src/panorama_demo/video_panorama.py`

改为：

```text
加载在线结果
→ 仅缺失时补跑离线阶段
→ 选择 preset
→ 调用 unified_visual_renderer
→ fast 原子发布
→ debug/audit 可异步
```

拟议 CLI：

```text
--preset fast|quality|audit
--reuse-online-trajectory
--report-level summary|full
--export-central-strips
--export-owner-only
--maximum-post-seconds 60
```

### `src/panorama_demo/video_source_selection.py`

替换“全部帧均为 renderer source”为：

```python
select_tracking_frames(...)
select_render_keyframes(...)
select_heavy_pairs(...)
```

### `src/panorama_demo/video_delivery.py`

新增：

- fast 最小发布集；
- debug 输出延迟发布；
- 预设、算法版本和 SLA 写入 delivery；
- 超时仍发布当前最佳结果，并明确降级原因。

### `src/panorama_demo/config.py`

增加配置校验：

- preset；
- motion resampling；
- online processing；
- optical flow；
- layered mesh；
- object lock；
- seam；
- photometric bundle；
- CUDA budget；
- output policy。

### `src/panorama_demo/calibrated_rgb_pushbroom.py`

保留为 `legacy/audit renderer`，不再作为视频正式默认 backend。

### `pyproject.toml`

新增：

```toml
[project.optional-dependencies]
visual-quality = [
  "torch>=2.7",
  "torchvision>=0.22",
  "scipy>=1.13",
  "kornia>=0.8",
]
video-container = [
  "av>=13",
  "zstandard>=0.23",
]
```

## 19.2 新增模块

```text
src/panorama_demo/
├── unified_visual_renderer.py
├── video_online_pipeline.py
├── online_orbslam3_bridge.py
├── video_motion_resampler.py
├── trajectory_smoothing.py
├── pair_risk.py
├── dense_optical_flow.py
├── depth_layers.py
├── layered_mesh_warp.py
├── object_owner_lock.py
├── spacetime_seam.py
├── photometric_bundle.py
├── cuda_tile_renderer.py
├── panorama_straightening.py
└── performance_profiler.py
```

---

# 20. 核心接口建议

```python
@dataclass(frozen=True)
class RenderKeyframe:
    frame_id: int
    timestamp_us: int
    pose_raw: np.ndarray
    pose_render: np.ndarray
    cumulative_motion_px: float
    support_fraction: float
    risk_score: float
    color_path: Path
    depth_path: Path


@dataclass(frozen=True)
class PairPlan:
    left_frame_id: int
    right_frame_id: int
    risk_level: int
    flow_backend: str
    mesh_mode: str
    object_lock_required: bool
    seam_mode: str
```

统一 renderer：

```python
def render_video_panorama(
    session: VideoSession,
    online_state: OnlineProcessingState | None,
    preset: VisualPreset,
    output_dir: Path,
) -> VisualRenderResult:
    ...
```

---

# 21. 正式配置示例

```yaml
capture:
  width: 848
  height: 480
  fps: 60
  jpeg_quality: 98
  depth_png_compression: 0

  video_mode:
    enabled: true
    lock_color_controls_after_warmup: true
    require_locked_control_metadata: true
    lock_white_balance_after_warmup: true
    post_lock_verified_frames: 5

stitch:
  video_renderer: unified_visual_v2
  video_preset: fast

  online_processing:
    enabled: true
    orb_tracking_fps: 20
    motion_analysis_scale: 0.25
    ring_capacity_frames: 120
    interpolate_non_tracking_poses: true

  motion_resampling:
    minimum_step_pixels: 3.0
    normal_target_step_pixels: 16.0
    risk_target_step_pixels: 8.0
    maximum_step_pixels: 24.0
    emergency_step_pixels: 30.0

  support:
    source_support_fraction: 0.60
    endpoint_outer_half_fov: true

  trajectory:
    smooth_translation: true
    smooth_rotation: true
    translation_strength: 0.70
    rotation_strength: 0.50

  pair_risk:
    heavy_pair_budget_fraction: 0.20

  optical_flow:
    cheap_backend: opencv_dis_medium
    heavy_backend: torchvision_raft_small
    raft_input_scale: 0.50
    bidirectional: true
    mixed_precision: true
    batch_pairs: 6
    maximum_fb_p95_pixels: 2.0

  depth_layers:
    enabled: true
    minimum_depth_mm: 200
    maximum_depth_mm: 3000
    edge_guard_pixels: 8
    temporal_median_radius: 1

  layered_mesh:
    enabled: true
    fast_cell_pixels: 24
    quality_cell_pixels: 16
    fast_window_frames: 5
    quality_window_frames: 7
    maximum_displacement_pixels: 48
    minimum_local_scale: 0.70
    maximum_local_scale: 1.40
    use_line_constraints: true
    per_cell_fallback: true

  object_owner:
    enabled: true
    minimum_component_pixels: 24
    object_guard_pixels: 12
    prefer_central_source: true
    allow_single_controlled_handoff: true

  seam:
    fast_backend: monotone_dp
    quality_backend: graphcut
    preview_scale: 0.50
    corridor_width_pixels: 224
    maximum_row_step_pixels: 3
    use_color_cost: true
    use_gradient_cost: true
    use_flow_confidence: true
    use_depth_edges: true
    use_object_locks: true

  photometric:
    linear_rgb: true
    model: affine_rgb
    temporal_smoothing: true
    block_correction: true
    block_size_pixels: 64

  blending:
    backend: torch_cuda
    fast_multiband_levels: 4
    quality_multiband_levels: 6
    background_blend_width_pixels: 24
    foreground_blend_width_pixels: 3
    tile_width_pixels: 2048
    tile_overlap_pixels: 128

  straightening:
    enabled_in_fast: false
    enabled_in_quality: true
    grid_size_pixels: 64
    maximum_displacement_pixels: 12

  delivery:
    fast_export_central_strips: false
    fast_export_owner_only: false
    fast_report_level: summary
    audit_report_level: full
```

---

# 22. 主流程伪代码

```python
def run_video_panorama(session, preset):
    profiler = PerformanceProfiler()
    online = load_online_state_if_available(session)

    trajectory = finalize_or_recover_trajectory(session, online)
    render_poses = smooth_trajectory(trajectory)

    motion = online.motion if online else compute_cheap_motion(session)
    keyframes = select_render_keyframes(
        session.frames, motion, render_poses, preset.motion_resampling
    )

    pair_plans = classify_pair_risk(
        keyframes,
        motion,
        session.depth_metadata,
        heavy_budget=preset.heavy_pair_budget_fraction,
    )

    photometric = solve_global_photometric_bundle(
        keyframes, pair_plans, preview_scale=0.25
    )

    writer = CudaTilePanoramaWriter(...)

    for window in sliding_windows(keyframes, size=preset.window_frames):
        gpu_sources = load_remap_once(window)
        plans = pair_plans.for_window(window)

        cheap_flow = compute_dis_flow(plans.low_risk)
        raft_flow = compute_raft_flow_batched(plans.high_risk)

        layers = build_depth_layers(window)
        meshes = solve_layered_mesh(
            window, cheap_flow, raft_flow, layers, render_poses
        )

        objects = track_and_lock_objects(window, layers, meshes)
        labels = solve_spacetime_seams(
            window, meshes, objects, photometric
        )

        writer.consume(
            warp_once(window, meshes, photometric),
            labels,
        )

    panorama = writer.finish()
    panorama = optional_straighten(panorama, preset)
    panorama = crop_valid_region(panorama)
    publish_fast_delivery(panorama, profiler.summary())
    return panorama
```

---

# 23. 报告与性能观测

新的 `video_report.json` 至少增加：

```json
{
  "preset": "fast",
  "online_processing": {
    "enabled": true,
    "orb_frames": 0,
    "motion_frames": 0,
    "render_keyframes": 0
  },
  "pair_planning": {
    "total_pairs": 0,
    "dis_pairs": 0,
    "apap_pairs": 0,
    "raft_pairs": 0,
    "object_lock_pairs": 0
  },
  "quality": {
    "seam_delta_e00_p95": 0.0,
    "line_step_p95_pixels": 0.0,
    "flow_fb_p95_pixels": 0.0,
    "object_internal_seam_count": 0
  },
  "performance": {
    "post_capture_seconds": 0.0,
    "peak_ram_bytes": 0,
    "peak_vram_bytes": 0,
    "stage_seconds": {}
  }
}
```

当前只记录总 elapsed，不足以定位剩余开销。

---

# 24. 测试与验收

## 24.1 单元测试

新增：

```text
test_video_motion_resampler.py
test_trajectory_smoothing.py
test_pair_risk.py
test_dense_optical_flow.py
test_depth_layers.py
test_layered_mesh_warp.py
test_object_owner_lock.py
test_spacetime_seam.py
test_photometric_bundle.py
test_cuda_tile_renderer.py
test_video_fast_delivery.py
test_video_performance_budget.py
```

## 24.2 几何验收

| 指标 | A 级目标 |
|---|---:|
| 长水平线接缝跳变 P95 | <1 px |
| 长垂直线接缝跳变 P95 | <1 px |
| 光流前后向误差 P95 | <1.5 px |
| mesh 翻折单元 | 0 |
| 局部尺度 | 0.70～1.40 |
| 未保护穿过深度边缘的 seam | 0 |

## 24.3 前景验收

固定风扇 ROI：

- 外圆轮廓断裂不超过 2 px；
- 格栅无明显双影；
- 风扇主体内部 seam 数为 0；
- 风扇主 source 数为 1；
- 必要时只允许一次受控 handoff；
- 底座无明显宽度突变。

农业场景增加：

- 果串内部 seam 数为 0；
- 软管和吊绳不出现周期性阶梯；
- 叶片边缘不出现双轮廓。

## 24.4 光度验收

| 指标 | 目标 |
|---|---:|
| 安全背景 seam ΔE00 P95 | <3 |
| 白墙中位亮度跳变 | <2% |
| 周期性竖向亮度块 | 肉眼不可见 |
| 饱和/暗区扩大 | 不超过输入基线 |

## 24.5 性能验收

```text
3 m fast：≤8 s
10 m fast：≤28 s
20 m fast：≤55 s
```

不能通过简单跳过前景处理来获得性能通过。

---

# 25. 分阶段实施顺序

## 阶段 0：性能可观测性

实现：

- 分阶段 profiler；
- `fast/quality/audit` CLI；
- fast 禁止中央条带输出；
- 快速报告；
- 3 m 基准脚本。

## 阶段 1：在线化与关键帧重采样

实现：

- 在线 ORB；
- 60 FPS 廉价运动；
- render keyframe selection；
- 一次 remap 和 GPU 金字塔；
- fast 最小发布集。

目标：先把 3 m 后处理降至 15 s 以下。

## 阶段 2：动态 seam 与全局颜色

实现：

- 线性 RGB 全局颜色；
- 逐行 DP seam；
- CUDA tile MultiBand；
- 宽支持区。

目标：大部分背景竖缝消失，3 m 达到 8～12 s。

## 阶段 3：RAFT 与分层 mesh

实现：

- pair 风险；
- RAFT-small batch；
- depth layers；
- per-cell fallback；
- APAP 退路。

## 阶段 4：对象锁定

实现：

- 深度/RGB/flow 对象候选；
- 跨帧 track；
- 单一 owner；
- 受控 handoff。

## 阶段 5：质量模式与最终整平

实现：

- 更细网格；
- 更高 RAFT 覆盖；
- GraphCut 多标签；
- FastSAM 可选；
- 低频直线整平。

---

# 26. 回退与失败策略

新的回退顺序：

```text
深度 + RAFT 分层 mesh
→ RAFT mesh
→ DIS + APAP
→ 局部仿射
→ 可弯曲 hard seam
→ legacy calibrated_rgb_pushbroom
```

必须满足：

- 局部 cell 失败不影响其他 cell；
- 一个 pair 失败不影响其他 pair；
- 对象锁失败时 seam 绕开高风险区；
- 颜色优化失败时使用锁定采集参数和单位 gain；
- RAFT OOM 时自动减 batch；
- 超时前优先完成可发布结果；
- 发布后允许生成 quality 版本，但不能删除已发布 fast 结果。

---

# 27. 建议命令形式

以下为拟议接口：

```powershell
g305-capture `
  --video-mode `
  --width 848 `
  --height 480 `
  --fps 60 `
  --online-panorama fast `
  --output data/captures/video/run_xxx

g305-video-panorama `
  data/captures/video/run_xxx `
  --preset fast `
  --reuse-online-trajectory `
  --maximum-post-seconds 60 `
  --output outputs/video_sequence

g305-video-panorama `
  data/captures/video/run_xxx `
  --preset quality `
  --reuse-online-trajectory `
  --output outputs/video_sequence_quality

g305-video-panorama `
  data/captures/video/run_xxx `
  --preset audit `
  --export-central-strips `
  --export-owner-only `
  --report-level full
```

---

# 28. 最终判断

在当前硬件上，下面两个目标可以同时作为正式工程目标：

```text
静态场景视觉质量接近 iPhone 全景
+
20 m 停止采集后 1 分钟内发布 fast 全景
```

但必须接受以下架构事实：

1. 计算必须部分前移到采集期间；
2. 60 FPS 是运动采样率，不是最终重型渲染帧率；
3. RAFT 和 RGB-D 网格只能自适应应用于困难区域；
4. 当前“全部真实帧 + 全部相邻 Open3D + 多轮 remap + 全量审计 + 两套 PNG”不能进入 fast 关键路径；
5. 接缝必须从整高竖切升级为可弯曲二维 owner；
6. 前景物体必须获得对象级 owner 连续性；
7. CUDA 必须真实运行，而不是只有配置和 DLL 路径；
8. 3 m 快速基准必须先达到 7～8 s，才有资格测试 20 m 一分钟目标。

正式推荐的默认路径是：

> **在线 ORB-SLAM3 + 60 FPS 廉价图像运动 + 自适应关键帧 + 风险驱动 RAFT/RGB-D 分层 mesh + 对象锁 + 可弯 seam + 全局颜色 + CUDA MultiBand。**

原有 `calibrated_rgb_pushbroom` 不删除，保留为 `audit/legacy` 后端，用于对照、复现和出现结构性故障时的最终退路。
