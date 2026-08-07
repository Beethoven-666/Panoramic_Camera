# Panoramic_Camera Phase 3：20 秒内近无痕 RGB-D 视频全景方案

## 0. 文档状态

本文档针对以下固定输入和硬件制定：

```text
会话：D:\central_strip_Panoramic_Camera\data\captures\video\run_20260806_153033
相机：Orbbec Gemini 305
输入：848 × 480 @ 60 FPS 同步 aligned RGB-D
扫描长度：约 3 m
GPU：NVIDIA GeForce RTX 5060 Laptop GPU，8 GB
产品入口：g305-video-panorama
```

本文档只修改独立视频产品的研发与冻结方向，不修改照片模式 `g305-panorama` 的正式契约。

本文档提出候选方案，不代表当前代码已经满足目标。只有本文全部硬门在固定 3 m 会话和后续现场复测上通过后，才允许更新 `production.lock.json`。

---

## 1. 不可协商目标

最终产品必须同时满足以下两个硬目标，不能以一个目标换取另一个目标。

### 1.1 视觉目标

1. 相邻源在接缝附近必须完成真实几何对齐，不能只靠模糊、羽化或宽带融合遮盖错位。
2. 正常观看比例和 1:1 检查下，安全背景接缝应接近不可见。
3. 长直线、货架横梁、立柱边缘、桌面前沿不得出现明显断裂、双边、台阶或弯折。
4. 纸箱、风扇、线缆等紧凑前景不得被多个窄条 owner 切碎；通过对象门的区域必须保持单一真实源 owner。
5. 深度边缘、遮挡、孔洞、反光和透明风险区禁止 MultiBand；这些区域只能采用经过审计的单 owner。
6. 每个有效像素必须恰有一个真实源 provenance owner，禁止合成 owner、插值帧和伪造 pose。

### 1.2 性能目标

```text
从最后一帧 RGB-D 完整写盘并安全关闭会话开始，
到 video_delivery.json 原子发布完成为止：

硬上限：20.0 s
冻结目标：P50 <= 16.0 s
冻结目标：P95 <= 18.5 s
保留抖动余量：至少 1.5 s
```

3-D TSDF/GLB、审计条带导出和离线评价不计入 2-D 主交付，但不得改变已经发布的 2-D 像素、owner 或等级。

采集期并行完成的在线计算必须记录真实耗时、输入哈希和完成水位，不能伪装成零成本。它可以与采集重叠，因此不计入“停采到出图”的 20 秒，但必须由会话写盘并通过完整性绑定才能复用。

### 1.3 同时失败策略

如果候选无法同时满足视觉硬门和 20 秒硬门：

- 不得降低接缝、对象完整性或 provenance 门槛；
- 不得通过大范围羽化掩盖几何错误；
- 不得把超时结果冻结为 production；
- 可以保留诊断预览和失败报告，但不得把它宣称为合格正式交付。

---

## 2. 当前证据与问题定位

当前完整 3 m D3 预览：

```text
目录：benchmarks\run_20260806_153033\view_only\D3_full_scan_preview_run5
评价范围：view_only_full_scan_unscored
渲染源：399
相邻 Open3D 边：398
主交付耗时：509.27 s
```

耗时拆分：

| 阶段 | 当前耗时 | 占总时长 |
|---|---:|---:|
| config/session | 2.48 s | 0.5% |
| render keyframe selection / dense evidence | 125.97 s | 24.7% |
| Open3D render edge audit | 30.11 s | 5.9% |
| calibrated render + D2/D3 | 350.16 s | 68.8% |
| 其他 | 0.55 s | 0.1% |

已确认的结构问题：

1. D1 的 24 FPS 稠密证据源被直接替换成最终渲染源，导致 399 个源都参加 full-resolution remap、Open3D 和 D3。
2. D1 相邻 60 FPS 边在两个方向重复求取 DIS forward/backward flow。
3. D3 为每个源重新构造接近整画布大小的 RGB、depth、valid tile。
4. D3 对 398 个相邻源 pair 跑接近 `1769 × 468` 的整画布 RAFT，最终只改变 3,027 个像素。
5. 当前完整预览的 Open3D backend 为 `open3d_rgbd`，不是正式要求的 `open3d_tensor_cuda_rgbd`。
6. 固定 validation 上 `black_fan` 已是单 owner，但 `central_carton` 和 `black_cable` 仍分别有 34 和 26 个 owner。

因此，不能在当前 399 源 D3 结构上通过局部微优化达到 20 秒。本方案明确停止把该结构视为 production 候选。

---

## 3. 核心决策

### 3.1 三种帧集合彻底分离

后续实现必须有三个不可混用的集合：

| 集合 | 典型数量/频率 | 用途 | 能否拥有最终像素 |
|---|---:|---|---|
| `motion_evidence_frames` | 60 FPS | 运动、风险、稠密 FB/RGB-D 证据 | 否 |
| `orb_anchor_frames` | 8 FPS | 真实 ORB-SLAM3 `camera_to_world` 锚点 | 只有被选为 render source 时可以 |
| `render_sources` | 36–48 帧/3 m | full-resolution remap、owner、接缝、输出颜色 | 是 |

候选期的 audited dense real-frame pose prior 仍可用于 D1/D2 研发证据，但不得进入 production lock 或正式像素 owner。

### 3.2 用空间推进量而不是固定高 FPS 选择渲染源

正式渲染源只从真实 ORB anchors 中选择，并满足：

```text
normal source advance：18–24 full-resolution px
risk source advance：8–12 full-resolution px
最大 render source 数：48
最小相邻共同有效走廊：160 px
首尾源：强制保留
高风险区：只允许增加真实 ORB anchor，不允许插值 pose
```

慢速采集不应因为时间帧更密而增加渲染源数量。只要空间推进量不足，额外真实帧仅用于风险证据，不参与 full-resolution 输出。

### 3.3 seam-first，取消 full-scan object compositor

正式 20 秒候选不再运行“每源整画布 tile + 全 pair RAFT + 全局 persistent component”D3。

替代路径为：

1. 在线阶段检测哪些名义 seam 会穿过深度边缘、强 RGB 结构、遮挡或紧凑对象。
2. 对安全 pair 直接使用标定/pose 布局和低成本局部 DIS 对齐。
3. 只对高风险 pair 的 `96–160 px` 走廊运行 RAFT-small 和 RGB-D 分层 mesh。
4. seam 优化绕开对象/深度保护区。
5. 无法绕开的紧凑对象由一个真实源完整拥有。
6. 只有共同可见的安全背景允许 2–8 px、最多 3 层的局部 MultiBand。

---

## 4. 目标数据流

```text
采集期，和 60 FPS 写盘并行
  ├─ 严格文件/同步配置验证
  ├─ 424 px ORB staging 与真实 ORB anchor
  ├─ 384/424 px 相邻运动、DIS FB 和 RGB-D residual
  ├─ pair 风险、深度边缘、RGB 强结构、遮挡/孔洞 mask
  ├─ 风险触发的对象候选与短窗口 mask track
  ├─ 共同可见背景 photometric 统计量
  └─ 原子写 online_state v3，绑定所有真实输入哈希

安全关闭后，20 秒主交付
  ├─ 0.8 s  严格加载并验证 online_state
  ├─ 0.4 s  从真实 ORB anchors 选择 36–48 个 render sources
  ├─ 1.5 s  处理尾部未完成证据并完成 CUDA Open3D 边审计
  ├─ 4.0 s  每源一次 full-resolution RGB/depth inverse remap
  ├─ 5.0 s  risk-only flow/RGB-D mesh、对象 owner、单调 seam
  ├─ 2.5 s  photometric correction 与安全窄带 MultiBand
  ├─ 1.2 s  JPEG/PNG/provenance/report/delivery 原子发布
  └─ 4.6 s  驱动、I/O、调度与尾延迟余量
```

硬预算合计：

```text
15.4 s 计划执行
+4.6 s 抖动余量
=20.0 s
```

任何阶段不能通过挪到 `offline_evaluation_seconds` 来规避主交付计时。只有已经在安全关闭前完成、落盘并通过输入哈希绑定的在线证据可以从 post-capture 计时中排除。

---

## 5. 采集期在线证据

### 5.1 online_state v3

新增或扩展：

```text
src/panorama_demo/video_online_state.py
src/panorama_demo/video_online_seam_evidence.py
```

`online_state v3` 至少记录：

```json
{
  "schema": "gemini305-video-online-state/v3",
  "input_hashes": {},
  "last_complete_frame_id": 0,
  "orb_anchor_ids": [],
  "adjacent_motion_evidence": [],
  "pair_risk_records": [],
  "object_track_records": [],
  "photometric_sufficient_statistics": [],
  "actual_cuda_calls": {},
  "online_wall_seconds": 0.0,
  "complete_and_cleanly_closed": false
}
```

复用条件：

1. manifest、calibration、frames.csv 和每个被引用 RGB/depth 文件哈希一致。
2. `last_complete_frame_id` 覆盖最终扫描段尾部。
3. ORB anchors、pair evidence 和对象 track 都只引用真实帧。
4. 会话安全关闭后才把 `complete_and_cleanly_closed` 原子写为 true。
5. 任一检查失败则 fail closed；不得部分复用旧会话证据。

### 5.2 DIS 计算去重

每条真实 60 FPS 边只计算一次：

```text
forward DIS
backward DIS
```

同一对 flow 同时生成：

- forward/backward consistency；
- 左源深度的 SE(3) residual；
- 右源深度的 SE(3) residual；
- occlusion/disocclusion 初始 mask；
- seam risk score。

禁止当前“左右方向各重新运行一套 forward/backward DIS”的四次计算。

计算按 4 workers 处理独立只读 pair；最终记录保持时间顺序。在线队列超过界限时只允许降低非风险证据密度，不能丢弃真实 ORB anchors、正式 render pair 必需证据或风险 pair。

### 5.3 自动对象候选

对象候选不得使用 `annotations_v2`、类别名或标注坐标。输入只允许：

- aligned depth 连通表面和深度不连续；
- RGB 梯度闭合区域；
- 遮挡/disocclusion；
- FB flow residual；
- 通用自动分割 mask；
- 多帧稳定性和完整可见性。

第一实现采用两级结构：

1. OpenCV 深度/RGB/flow 生成廉价 proposal。
2. 只有 proposal 会穿过候选 seam 时，才在 424 px 风险帧上运行 SAM 2.1 Hiera Tiny 自动 mask/refinement；每个 3 m 会话最多触发 12 个关键帧。

SAM 权重必须本地固定、记录 SHA-256、无下载回退，并在采用前完成许可证确认。SAM 只生成保护 mask，不提供颜色、pose 或插值帧。代码与模型能力的上游依据固定为 [Meta SAM 2 官方仓库](https://github.com/facebookresearch/sam2)，不得用来源不明的第三方权重替换。

对象 mask 用已计算的 DIS/RAFT 在短窗口双向传播；前后向 IoU、深度层、面积稳定性或边界审计失败时，该 track 不得被信任。

---

## 6. 停采后 20 秒 renderer

### 6.1 一次 remap 和常驻窗口

每个正式 render source：

- 读取一次真实全分辨率 RGB；
- 读取一次 aligned depth；
- 生成一次标定 inverse map；
- 在同一 CUDA/张量路径中生成 RGB、depth、valid；
- RGB 和 depth 共用 map，不重复计算坐标；
- 普通源只保留当前 seam 需要的贡献区；
- 常驻源窗口限制为 3–5 帧；
- 禁止建立 `source_count × panorama_width × panorama_height` 的整画布 tile stack。

最终输出颜色始终来自真实 RGB 的这一次标定 inverse remap。

### 6.2 pair 风险分级

每个相邻正式源 pair 冻结为以下一种：

| 等级 | 条件 | 对齐方式 | seam/blend |
|---|---|---|---|
| R0 | 低梯度、低残差、无深度风险 | pose + 局部 DIS 平移审计 | 单调 seam + 2–8 px MultiBand |
| R1 | 结构/光度风险但共同可见 | DIS FB + 局部可弯曲 seam | 安全背景窄带 MultiBand |
| R2 | 深度层、遮挡或局部视差风险 | 384 px RAFT-small FB + RGB-D far/mid mesh | 保护区 hard owner，背景窄带 MultiBand |
| R3 | 紧凑对象或不可可靠对齐 | 不 warp 对象 | 对象完整单 owner，seam 绕行 |
| RF | 无法通过可见性/残差/owner 门 | 不发布合格候选 | 写结构化失败 |

每个会话允许的重计算上限：

```text
R2 RAFT pair <= 8
R3 object track <= 12
局部 mesh corridor width <= 160 px
局部 mesh displacement <= 8 px
MultiBand levels <= 3
总 MultiBand 宽度 <= 8 px/pair
```

如果实际场景超过这些上限，不能为了守住 20 秒把风险 pair 假报为 R0/R1。该会话应判为本候选不适用，而不是发布有明显接缝的图。

### 6.3 局部几何对齐

R2 pair 只在风险走廊运行：

1. RAFT-small forward/backward，输入宽度 384 px，FP16 CUDA。
2. flow 上采样回 full resolution 时同步缩放位移。
3. aligned depth 双向重投影和 z-buffer。
4. far/mid/near 三层分类。
5. near、遮挡、disocclusion、深度边缘、孔洞和强 RGB 结构全部保护为 owner-only。
6. far/mid 安全背景拟合有界 B-spline/规则 mesh。
7. 通过 held-out FB、RGB-D residual、零边界位移、正 Jacobian、最大 8 px 位移和保护域零交集后才应用。

mesh 不得修改 pose，不得产生颜色，不得覆盖前景，不得跨 pair 或升级为全景级 flow。

### 6.4 seam 求解

每个 pair 的 seam cost 只在互斥 corridor 中计算：

```text
E = 0.30 × Lab residual
  + 0.25 × gradient residual
  + 0.20 × FB/geometry residual
  + 0.15 × depth/occlusion protection
  + 0.10 × curvature/temporal consistency
```

硬约束优先于代价：

- 不得穿过对象锁、深度边缘和 occlusion protection；
- owner label 必须单调；
- 有效像素恰有一个 owner；
- 相邻 pair corridor 互斥；
- seam 不能生成回头、孤岛或小碎片 owner；
- 长直线附近优先沿同侧 owner 绕行，不能横切造成台阶。

如果没有可行路径，则尝试选择一个完整支持走廊的真实源 hard owner；仍不可行则 RF。

### 6.5 光度和 MultiBand

全局 RGB gain/bias 只用共同可见、安全背景样本训练，并保留 held-out 样本。

```text
gain 范围：0.85–1.18
bias 范围：-12–12
held-out ΔE00 P95：< 3.0
超过范围或误差门：单位校正
```

MultiBand 仅处理：

- 两源共同有效；
- 低梯度；
- 无对象锁；
- 无深度/遮挡保护；
- 已通过局部几何对齐；
- seam 两侧 2–8 px 的安全背景。

MultiBand 不得用于修复可见的几何双边。

---

## 7. 接缝近无痕验收门

“几乎看不到接缝”必须同时由自动指标和固定盲评定义，不能只看一张缩略图。

### Gate S1：几何对齐

对每个 R1/R2 pair：

```text
held-out FB P95 <= 1.0 px（目标）
held-out FB P95 <= 1.5 px（硬上限）
RGB-D residual P95 <= 1.5 px
automatic long-line discontinuity P95 <= 1.5 px
mesh minimum Jacobian > 0
mesh maximum displacement <= 8 px
```

任一 pair 超过硬上限即不能作为合格候选发布。

### Gate S2：对象与风险保护

固定 `annotations_v2` 只能在 renderer 完成后评价：

```text
central_carton owner_count = 1
black_fan owner_count = 1
black_cable owner_count = 1
compact-object full-support >= 98%
object internal seam count = 0
object blend pixel count = 0
unowned valid pixels = 0
```

`yellow_beam_*` 和 `table_front_edge`：

- 长线断裂、台阶和双边均必须通过固定测量门；
- 深度/RGB 风险区域 blend pixel count 必须为 0；
- seam 若接近长线，必须沿同侧 owner 绕行而不是横切。

### Gate S3：光度可见性

```text
safe-background seam ΔE00 P95 < 3.0
seam-band/control-band gradient discontinuity ratio <= 1.15
明显亮度跳变 pair count = 0
宽带模糊/ghost pair count = 0
```

### Gate S4：固定盲评

从 validation 和完整 3 m 输出中自动裁出：

- 全部真实 seam 的 160 px strip；
- 数量相同、尺寸相同的非 seam control strip；
- 随机顺序、隐藏 provenance 和算法名。

验收条件：

```text
1:1 显示：不得出现明确错位/双边/断线
fit-width 显示：不得出现一眼可见的周期性竖接缝
seam/control 二分类人工识别率 <= 60%
任何被标记为“明显接缝”的真实 seam 数 = 0
```

盲评结果必须落盘，不能只在聊天或截图中确认。

---

## 8. 20 秒性能验收门

### Gate P1：计时口径

计时起点：

```text
最后一个正式 RGB-D 帧写盘成功、设备和 writer 安全关闭之后
```

计时终点：

```text
video_delivery.json 最后原子写入之后
```

计时必须包含：

- online_state 加载和哈希验证；
- render source 选择；
- 尾部 ORB/Open3D 收尾；
- 正式 remap；
- flow/mesh/seam/photometric/MultiBand；
- PNG/JPEG/provenance/report/delivery 写盘。

### Gate P2：固定会话重复

在 `run_20260806_153033` 上运行：

```text
2 次 warm-up，不计分
10 次正式重复
```

通过条件：

```text
10/10 primary_post_capture_seconds <= 20.0
P50 <= 16.0
P95 <= 18.5
输出 panorama 解码像素 hash 一致
owner array hash 一致
source_frame_ids 一致
所有视觉 Gate 一致通过
```

### Gate P3：资源边界

```text
render sources <= 48
常驻 full-resolution source window <= 5
GPU peak allocated <= 6.5 GB
CPU aggregate working set <= 4 GB
R2 pair <= 8
CUDA OOM count = 0
unexpected CPU renderer fallback count = 0
```

Open3D 正式边必须全部报告：

```text
open3d_tensor_cuda_rgbd
```

观察到 `open3d_rgbd` 即本次正式性能/结构验收失败。

### Gate P4：现场复测

固定会话通过后，还必须在同一硬件上新采至少 3 次 3 m：

- 低纹理背景一次；
- 货架/立柱/长直线丰富场景一次；
- 含纸箱、线缆、风扇等紧凑前景一次。

每次都必须同时通过 20 秒和 Gate S1–S4。合成测试和已有会话重复不能替代现场复测。

---

## 9. 实施阶段

### Phase A：性能骨架，不改变图像算法

目标：先证明 36–48 个真实 ORB render sources 和 20 秒数据面可行。

工作：

1. 把 D1 evidence pool 与 render source pool 分离。
2. 删除 full-scan D3 tile stack。
3. DIS bidirectional 计算去重并移入在线状态。
4. RGB/depth 共用 inverse map。
5. 启用并强制真实 Open3D tensor CUDA backend。
6. 加入逐阶段 CUDA event 和 wall-clock profiler。

Gate：

```text
结构完整 hard-owner panorama <= 12 s post-capture
render sources <= 48
所有真实源/pose/provenance 合法
```

此阶段允许接缝仍明显，只验证性能上限和数据链。

### Phase B：局部对齐和单调 seam

工作：

1. R0/R1/R2/R3/RF pair 风险分类。
2. risk-only 384 px RAFT-small FB。
3. RGB-D far/mid 局部 mesh。
4. 单调 seam 和 owner cleanup。
5. 长直线保护。

Gate：S1，并保持总时长 <=17 s。

### Phase C：对象单 owner

工作：

1. 在线廉价 object proposal。
2. 风险触发的 SAM 2.1 Tiny mask refinement。
3. 短窗口双向 track。
4. 单真实源 full-support owner 选择。
5. annotation-only post-render evaluation。

Gate：S2，并保持总时长 <=18 s。

### Phase D：近无痕合成

工作：

1. 安全背景 global gain/bias。
2. held-out photometric audit。
3. 2–8 px、最多 3 层的局部 MultiBand。
4. seam/control 自动裁片和盲评包。

Gate：S3、S4，并保持 P95 <=18.5 s。

### Phase E：冻结和现场复测

工作：

1. 10 次固定会话重复。
2. 新采 3 次 3 m 现场复测。
3. 全量单元/合成回归。
4. 冻结模型 SHA-256、候选 YAML、source commit 和 production lock。
5. 更新 README、AGENTS、配置和交付 schema 说明。

只有 S1–S4、P1–P4 全部通过才允许进入 D4/D5 或 production freeze。

---

## 10. 预计代码边界

优先修改：

```text
src/panorama_demo/video_panorama.py
src/panorama_demo/video_online_state.py
src/panorama_demo/video_source_selection.py
src/panorama_demo/video_dense_real_frame_layout.py
src/panorama_demo/video_dense_pose_prior.py
src/panorama_demo/video_visual_renderer_v2.py
src/panorama_demo/video_budget_controller.py
src/panorama_demo/video_delivery.py
```

建议新增：

```text
src/panorama_demo/video_online_seam_evidence.py
src/panorama_demo/video_pair_risk.py
src/panorama_demo/video_object_tracks.py
src/panorama_demo/video_multilabel_seam.py
src/panorama_demo/video_torch_multiband.py
src/panorama_demo/video_photometric_graph.py
src/panorama_demo/video_seam_blind_eval.py
```

当前 D3 代码保留为隔离的历史候选，不作为 20 秒 renderer backend 或失败回退：

```text
src/panorama_demo/video_d3_object_first_compositor.py
configs/video_candidates/D3_object_first_dense_source_compositor.yaml
```

---

## 11. 配置草案

以下参数属于冻结候选内部配置，不能暴露给普通用户调节：

```yaml
video_panorama_20s:
  target_post_seconds: 16.0
  maximum_post_seconds: 20.0
  require_complete_online_state: true
  require_direct_orb_render_poses: true
  render_sources:
    normal_advance_pixels: 20
    risk_advance_pixels: 10
    maximum_count: 48
    minimum_shared_corridor_pixels: 160
  pair_risk:
    analysis_width: 384
    maximum_raft_pairs: 8
    maximum_object_tracks: 12
  local_geometry:
    corridor_width_pixels: 160
    maximum_displacement_pixels: 8
    maximum_fb_p95_pixels: 1.5
    maximum_rgbd_residual_p95_pixels: 1.5
    require_positive_jacobian: true
  object_owner:
    minimum_full_support: 0.98
    maximum_handoffs: 0
    allow_object_warp: false
    allow_object_multiband: false
  multiband:
    safe_background_only: true
    maximum_total_width_pixels: 8
    maximum_levels: 3
  resources:
    resident_source_window: 5
    maximum_gpu_gb: 6.5
    maximum_cpu_working_set_gb: 4.0
```

冻结前可在 development split 搜索参数，但 validation/holdout 首次查看后不得继续针对结果调参。

---

## 12. 测试要求

新增测试至少包括：

```text
tests/test_video_online_seam_evidence.py
tests/test_video_render_source_decoupling.py
tests/test_video_pair_risk.py
tests/test_video_object_tracks.py
tests/test_video_multilabel_seam.py
tests/test_video_torch_multiband.py
tests/test_video_seam_blind_eval.py
tests/test_video_20s_budget.py
```

必须覆盖：

1. evidence frame 永远不能成为未授权的 owner。
2. render source pose 全部来自真实 ORB anchors。
3. online state 任一输入哈希变化即拒绝复用。
4. RGB/depth map 坐标只生成一次。
5. 风险 mask 与 MultiBand 区域零交集。
6. 对象区域只含一个 owner，且不含 blend。
7. owner 水平单调、无孤岛、无未归属有效像素。
8. RAFT/mesh 只在配置走廊内执行。
9. 超过 R2/R3 数量上限时 fail closed。
10. `G305_CUDA=required` 下出现 CPU Open3D backend 必须失败。
11. summary/minimal 与 audit 输出 panorama/owner hash 相同。
12. 20 秒计时包含编码与 `video_delivery.json` 原子写入。

常规验证：

```powershell
$G305Python = 'D:\Panoramic_Camera\.conda\python.exe'
& $G305Python -m pytest -q
ruff check src tests
& $G305Python -m compileall -q src tests
git diff --check
```

真实验收必须分别记录：

- 单元/合成测试；
- 真实 Open3D tensor CUDA 边；
- 真实完整 ORB-SLAM3；
- 固定失败数据；
- 固定 3 m 十次性能重复；
- 三类新采 3 m 现场复测。

---

## 13. 最终交付证据

候选通过时必须生成：

```text
video_panorama.jpg
video_panorama.png
video_pixel_provenance.npz
video_report.json
video_timing.json
video_delivery.json

seam_evaluation/
  seam_metrics.json
  object_integrity.json
  line_continuity.json
  blind_eval_manifest.json
  seam_strips/
  control_strips/

performance_evaluation/
  repeated_runs.json
  stage_percentiles.json
  gpu_memory.json
  cuda_backend_audit.json
```

`video_delivery.json` 仍最后写入，并新增或明确记录：

```json
{
  "primary_post_capture_seconds": 0.0,
  "maximum_post_seconds": 20.0,
  "within_post_capture_budget": true,
  "seam_quality_pass": true,
  "object_integrity_pass": true,
  "blind_seam_evaluation_pass": true,
  "direct_orb_render_pose_count": 0,
  "interpolated_render_pose_count": 0,
  "actual_open3d_backend": "open3d_tensor_cuda_rgbd"
}
```

---

## 14. Go / No-Go 判定

### Go

只有同时满足以下条件才继续 D4/D5：

```text
S1 几何对齐通过
S2 对象/风险保护通过
S3 光度接缝通过
S4 固定盲评通过
P1 计时口径通过
P2 固定会话十次重复通过
P3 资源/CUDA 边界通过
P4 三次新采现场复测通过
全量测试通过
```

### No-Go

以下任一情况立即停止冻结：

- 任一真实 seam 有明显双边、断线或台阶；
- 纸箱、风扇或线缆不是单 owner；
- 通过宽带模糊才隐藏接缝；
- 任一次正式重复超过 20 秒；
- 使用 48 个以上 render sources 才能达到视觉门；
- Open3D 实际回退到 CPU；
- annotations 进入 renderer 数据面；
- online state 不完整却被当成正式缓存；
- 任何插值 pose 或非真实源成为正式 owner。

本方案的最终原则是：

```text
用在线证据和风险局部化换时间，
用真实几何对齐、对象单 owner 和窄带安全融合换接缝质量，
不用增加全图源密度、全图 RAFT 或宽带模糊换表面上的“无缝”。
```
