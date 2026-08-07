# Panoramic_Camera 最终合并方案：20 秒内接缝优先 RGB-D 视频全景

## 0. 文档定位

本文档合并并取舍以下两份方案：

```text
D:\central_strip_Panoramic_Camera\Panoramic_Camera_Phase3_20s_Seamless_Video_Panorama_Plan.md
C:\Users\zyh68\Downloads\Panoramic_Camera_Seam_First_Speed_Controlled_Production_Plan.md
```

本文档是后续独立视频产品研发、候选消融、性能验收和 production freeze 的统一执行方案。两份来源文档保留为历史参考；若其内容与本文冲突，以本文和仓库当前可执行源码、默认配置、测试及 `AGENTS.md` 为准。

适用范围：

```text
项目：Panoramic_Camera
入口：g305-video-panorama / g305-video-experiment
相机：Orbbec Gemini 305
输入：848 × 480 @ 60 FPS 同步 aligned RGB-D
目标场景：约 3 m 单向水平侧扫
固定研发会话：
D:\central_strip_Panoramic_Camera\data\captures\video\run_20260806_153033
硬件：RTX 5060 Laptop GPU，8 GB
开发分支：codex/dense-anchor-recovery
```

本文档不修改照片模式 `g305-panorama` 的正式契约。

本文档描述的是待实现、待实测的候选方案，不代表当前程序已经满足质量或 20 秒目标。只有本文全部硬门实际通过后，才允许生成新的 `production.lock.json`。

---

## 1. 最终产品硬目标

### 1.1 优先级

```text
第一优先级：接缝两侧几何真实对齐；
第二优先级：接缝光度连续，正常观看几乎不可见；
第三优先级：紧凑对象完整、长直线连续；
第四优先级：停采后 20 秒内完成 2-D 主交付；
第五优先级：在以上硬门内选择模块最少、显存最低的候选。
```

速度不能通过关闭几何门、对象门或 provenance 门获得；视觉不能通过 509 秒全画布计算获得。

### 1.2 视觉硬目标

1. 每条最终 seam 在融合前必须通过局部几何对齐审计。
2. 横梁、桌面前沿、立柱、货架板、管道和其他自动长线不得在 seam 处出现台阶、断裂、双边或方向突变。
3. 白墙、黄色平面和其他低纹理背景不得出现清楚可见的亮度竖带。
4. `central_carton`、`black_fan`、`black_cable` 在固定评价中必须各自由一个真实源 owner 完整拥有。
5. 深度边缘、遮挡、disocclusion、孔洞、反光/透明风险、对象锁区域禁止融合。
6. 安全背景接缝在正常观看和 1:1 检查下应接近不可见。
7. 每个有效像素必须恰有一个真实源 provenance owner。
8. 禁止插值帧、伪造帧、伪造 pose、二维 motion pose 或非真实源颜色。

### 1.3 性能硬目标

正式计时：

```text
起点：最后一个正式 RGB-D 帧完整写盘，设备和 writer 安全关闭；
终点：video_delivery.json 最后原子写入完成。
```

目标：

```text
伸展目标：cold <= 15.0 s
冻结目标：P50 <= 16.0 s
冻结目标：P95 <= 18.5 s
硬上限：每次 <= 20.0 s
硬上限余量：冻结候选至少保留 1.5 s P95 余量
```

计时必须包含：

- online state 加载与哈希验证；
- 正式渲染源选择和救援源决策；
- 尾部 ORB/Open3D 收尾；
- RGB/depth inverse remap；
- flow、RGB-D 局部 mesh、对象 owner、seam；
- photometric correction、MultiBand；
- JPEG、PNG、provenance、report、timing、delivery 写盘。

TSDF/GLB、3-D viewer、完整审计条带、盲评和离线 annotation evaluation 不计入 2-D 主交付，但不得改变已经发布的 2-D 像素、owner、source IDs 或等级。

### 1.4 失败原则

任一视觉硬门或 20 秒硬门失败时：

- 不生成或更新 production lock；
- 不把明显接缝标为 production A/B；
- 不静默回退旧 hard-owner 并冒充正式结果；
- 不通过宽羽化、降门槛或虚假 backend 报告掩盖失败；
- 可以保留结构化 failure、diagnostic preview 和审计证据。

---

## 2. 当前实测基线

固定新会话的已知状态：

```text
D1 source/pose/dense-evidence gate：已通过
全量回归：用户报告 1142 passed, 2 skipped
D2：真实深度分层 B-spline、RAFT-small FB P95 0.689 px、自动长线审计已通过
D3 validation：black_fan 单 owner 通过
D3 validation：central_carton 34 owners，失败
D3 validation：black_cable 26 owners，失败
```

当前完整 3 m 预览：

```text
目录：benchmarks\run_20260806_153033\view_only\D3_full_scan_preview_run5
评价范围：view_only_full_scan_unscored
scan frames：789
render sources：399
Open3D edges：398
主交付：509.27 s
```

耗时：

| 阶段 | 当前耗时 | 占比 |
|---|---:|---:|
| config/session | 2.48 s | 0.5% |
| dense evidence / render selection | 125.97 s | 24.7% |
| Open3D edge audit | 30.11 s | 5.9% |
| calibrated render + D2/D3 | 350.16 s | 68.8% |
| 其他 | 0.55 s | 0.1% |

已确认根因：

1. 24 FPS 稠密证据源被直接作为最终渲染源。
2. 每条 60 FPS 边重复计算两套 forward/backward DIS。
3. D3 为每个源建立接近整幅全景的 RGB/depth/valid tile。
4. D3 对 398 个 pair 运行接近 `1769 × 468` 的整画布 RAFT。
5. D3 最终只改变 3,027 个像素，纸箱和线缆仍未通过。
6. 当前完整预览的 Open3D backend 是 `open3d_rgbd`，不是正式要求的 `open3d_tensor_cuda_rgbd`。

结论：

```text
停止把 399 源 full-scan D3 视为 production 候选；
停止继续 D4/D5；
先完成 seam-first 的质量门、性能数据面和 D3 对象完整性替代路径。
```

---

## 3. 最终架构决策

### 3.1 Evidence、ORB anchor、Render Source 分离

| 集合 | 密度/数量 | 用途 | 是否可拥有正式像素 |
|---|---:|---|---|
| `motion_evidence_frames` | 60 FPS | motion、DIS FB、RGB-D residual、风险和救援候选 | 否 |
| `orb_anchor_frames` | 8 FPS | 真实 ORB-SLAM3 `camera_to_world` | 被选择后才可以 |
| `base_render_sources` | 36–44/3 m | 正常 full-resolution renderer | 是 |
| `rescue_render_sources` | 全会话最多4 | 修复无法通过的 seam | 是，且必须是 direct ORB anchor |
| `final_render_sources` | 总数最多48 | 最终 remap、owner、Open3D、seam | 是 |

候选期 audited dense real-frame prior 继续只用于 D1/D2 研发证据，不能进入 production lock、正式 render source 或正式 owner。

### 3.2 空间推进量选源

不按慢速会话的时间帧密度增加渲染源。初始冻结范围：

```text
normal source advance：18–24 full-resolution px
risk source advance：8–12 full-resolution px
maximum source advance：24 px
minimum shared corridor：160 px
base source count：36–44
rescue source count：0–4
final source count：<=48
首尾 direct ORB source：强制保留
```

每个候选参数只允许在 development split 搜索；validation 和 holdout 查看后不得继续定向调参。

### 3.3 全局轻量，接缝局部重型

```text
全局：真实 ORB pose、scan coordinate、画布布局、source 顺序、粗 photometric 统计
局部：96–160 px seam corridor 内的 DIS/RAFT、RGB-D 分层、mesh、seam、blend
```

禁止：

- 全图/全 pair RAFT；
- 全景级 flow；
- 全局 homography；
- 全画布自由 mesh；
- 399 源 full-canvas object compositor；
- 5–7 帧全画布联合优化；
- 以 TSDF 修正 RGB panorama。

---

## 4. 目标数据流

```text
采集期，与 60 FPS RGB-D 写盘并行
  ├─ 严格同步/文件/深度完整性验证
  ├─ 424 px ORB staging 和 8 FPS direct ORB anchors
  ├─ 384/424 px 60 FPS motion 和去重 bidirectional DIS
  ├─ 两方向 RGB-D residual、遮挡、孔洞、深度边缘
  ├─ pair 风险和自动长线
  ├─ 对象 proposal、短窗口 track、救援源候选
  ├─ 安全背景 photometric sufficient statistics
  └─ 原子 online_state v3，绑定真实输入和配置哈希
                     ↓
安全关闭，开始 20 秒计时
                     ↓
验证 online_state 和 direct ORB chain
                     ↓
选择 36–44 个 base render sources
                     ↓
全部相邻边真实 Open3D tensor CUDA 审计
                     ↓
每源一次 full-resolution RGB/depth inverse remap
                     ↓
逐 seam 分级升级
  L0 粗布局审计
  L1 廉价局部平移/DIS
  L2 risk-only batched RAFT-small
  L3 RGB-D far/mid 局部 mesh
  L4 direct-ORB rescue source
                     ↓
对象单 owner + 受限 GraphCut/单调 label seam
                     ↓
安全背景 photometric + 2–8 px/<=3层 MultiBand
                     ↓
质量门、性能门、原子 2-D 发布
```

---

## 5. 采集期 online state v3

### 5.1 必需内容

扩展 `src/panorama_demo/video_online_state.py`，新增 `video_online_seam_evidence.py`。

```json
{
  "schema": "gemini305-video-online-state/v3",
  "input_hashes": {},
  "config_sha256": "",
  "model_sha256": {},
  "last_complete_frame_id": 0,
  "orb_anchor_ids": [],
  "adjacent_motion_evidence": [],
  "pair_risk_records": [],
  "automatic_line_records": [],
  "object_track_records": [],
  "rescue_candidate_records": [],
  "open3d_candidate_edge_records": [],
  "photometric_sufficient_statistics": [],
  "actual_backend_counters": {},
  "online_wall_seconds": 0.0,
  "complete_and_cleanly_closed": false
}
```

### 5.2 复用门

只有以下条件全部满足才允许复用：

1. manifest、calibration、frames.csv 哈希一致。
2. 所有被引用 RGB/aligned-depth 文件哈希一致。
3. config 和本地模型 SHA-256 一致。
4. ORB anchors、evidence、track 和 rescue candidate 只引用真实帧。
5. `last_complete_frame_id` 覆盖最终 scan segment 尾部。
6. `complete_and_cleanly_closed=true`。
7. online state 在会话安全关闭后最后原子确认。

失败时不得部分复用、跨会话迁移或使用旧标注坐标补齐。

为同时满足20秒和发布前全边Open3D，采集期必须根据direct-ORB source lattice预计算可能成为正式相邻边的CUDA Open3D记录：

- normal/risk空间推进量可能选择的左右direct-ORB组合；
- 每个候选seam的direct-ORB rescue左右新边；
- 每条记录绑定两端真实RGB-D哈希、ORB pose哈希、working intrinsics和backend；
- 不允许通过组合多条短边冒充一条最终相邻边审计；
- 停采后final pair若没有完全匹配的记录，必须在20秒计时内真实补算。

在线Open3D任务必须有界调度，不得阻塞60 FPS采集、正式写盘或ORB anchor生成。

### 5.3 DIS 去重与并行

每条相邻真实 60 FPS 边只运行：

```text
forward DIS：一次
backward DIS：一次
```

同一对 flow 同时派生：

- FB consistency；
- 左源深度 SE(3) residual；
- 右源深度 SE(3) residual；
- occlusion/disocclusion；
- pair risk；
- object/line propagation evidence。

不得按左右方向重复运行四次 DIS。独立 pair 以最多4 workers并行，最终审计顺序仍按帧时间排序。

### 5.4 自动长线

只使用运行时自动检测：

```text
LSD/HoughLinesP
+梯度方向
+DIS/RAFT 对应
+背景深度层
```

禁止将 frame 389/391 的 `yellow_beam_*`、`table_front_edge` 标注输入 renderer。标注只用于发布后评价。

### 5.5 自动对象 proposal/refinement

廉价 proposal 来源：

- aligned depth 连通表面和深度不连续；
- RGB 梯度闭合区和细长强结构；
- 遮挡/disocclusion；
- FB residual；
- 多帧面积、深度和边界稳定性。

只有 proposal 会与候选 seam 相交时，才允许运行通用自动 mask refinement。初始候选为 SAM 2.1 Hiera Tiny：

```text
输入：424 px 风险关键帧
每个 3 m 会话最多12个 refinement keyframes
本地权重、固定 SHA-256
无下载和无模型回退
采用前完成许可证确认
输出：保护 mask，不提供 pose、颜色或插值帧
```

mask 使用既有 flow 在短窗口双向传播，并通过：

- forward/backward mask IoU；
- 深度层一致性；
- 面积/边界稳定性；
- 完整可见性；
- source support >=98%。

失败 mask 不得参与对象锁。

---

## 6. Render Source 与救援帧

### 6.1 Base sources

base source 只从 direct ORB anchors 中按 scan coordinate、清晰度、深度有效率和空间推进量选取。

不得因为慢速采集包含更多真实文件就提高渲染源数量。

### 6.2 Rescue source

当一条 seam 依次经过 L1、L2、L3 仍无法通过质量门时，可插入一个中间真实帧。

救援源必须：

1. 是左右 base source 之间的真实 RGB-D 帧。
2. 有 direct ORB `camera_to_world`，不得使用 interpolated/refined prior。
3. 清晰度和 aligned-depth 有效率通过。
4. 与左右源均有充分共同有效走廊。
5. 插入后产生的两条新边都通过 Open3D tensor CUDA 审计。
6. 计入总 render source 上限和 20 秒预算。

限制：

```text
每条失败 seam：最多1个 rescue source
全会话：最多4个 rescue sources
final render sources：仍然<=48
```

如果候选需要超过此限制才能通过视觉门，则该候选不适合20秒产品。

---

## 7. 正式 Open3D 审计

与来源方案不同，正式主路径不能关闭全边 Open3D，也不能把完整边审计移到发布后。

要求：

```text
所有 final render source 相邻边都在发布前审计；
实际 backend 必须为 open3d_tensor_cuda_rgbd；
G305_CUDA=required 验收；
观察到 open3d_rgbd 即正式结构/性能验收失败；
Open3D 只验证真实相邻边，不修改 ORB pose。
```

性能措施：

- 只审计最终<=47条相邻边，而不是398条；
- 采集期对direct-ORB source lattice和rescue候选边预计算实际CUDA审计；
- 停采后只复核完全匹配的预计算边，并补算尾部/最终选择缺失边；
- 4 workers并行解码和输入准备；
- 单一实际 CUDA backend 串行/批次执行最终 estimator；
- 已完整绑定的采集期准备结果可复用；
- 不虚报 GraphCut、MultiBand、ORB 或其他 CPU 算法为 CUDA。

---

## 8. 每源一次 inverse remap

每个 final render source：

1. 读取一次真实 full-resolution RGB。
2. 读取一次 aligned depth。
3. 生成一次标定 inverse map。
4. RGB 和 depth 共用 map 坐标。
5. RGB 只来自这一次 inverse remap。
6. 常驻 full-resolution source window 为3–5帧。
7. 禁止建立 `source_count × panorama_width × panorama_height` tile stack。

GPU 数据面：

```text
每个 final source H2D 一次
标定 grid GPU 常驻/复用
corridor flow/warp/seam cost 尽量设备驻留
主路径只在必要边界下载 panorama、owner/provenance和审计标量
```

3 m panorama 可使用单画布；tile renderer 仅作为长距离扩展：

```text
20 m 建议 tile width：2048 px
tile overlap：128 px
只加载影响当前 tile 的 source
```

不得为了3 m提前引入无收益的tile复杂度。

---

## 9. 逐接缝分级对齐

每条 seam 必须从便宜到昂贵升级，已经通过的 pair 不得继续运行更重算法。

### L0：粗布局审计

输入：

```text
direct ORB camera_to_world
+scan coordinate
+calibrated source support
```

只建立名义 boundary 和96–160 px corridor，不修改 pose。

### L1：廉价局部对齐

在半分辨率 corridor 计算：

```text
有界 phase-correlation translation
+既有 DIS FB
+自动长线对应
```

只允许小范围 `dx/dy` 或等价低自由度局部修正；不得构成全图 affine、homography 或二维 pose 替代。

通过全部 seam hard gate 后停止升级。

### L2：risk-only batched RAFT-small

只对L1失败且仍有可恢复共同背景的pair运行：

```text
输入宽度：384 px
走廊：<=160 px full-resolution对应区
FP16 CUDA
单模型实例
batch：4–8 difficult pairs
forward/backward：全部R2 pair
模型和flow尽量GPU驻留
全会话R2 pair：<=8
```

不得对普通pair或整幅全景运行RAFT。

### L3：RGB-D far/mid 局部 mesh

在L2 evidence上执行：

1. aligned depth 双向重投影和z-buffer。
2. far/mid/near三层分类。
3. near、遮挡、disocclusion、深度边缘、孔洞、强RGB结构全部owner-only。
4. 仅far/mid安全同层背景拟合有界单调B-spline/规则mesh。
5. corridor边界平滑回到identity。

硬约束：

```text
maximum displacement <= 8 px
scale in [0.80, 1.25]
minimum Jacobian > 0
corridor outer boundary displacement = 0
protected-region intersection = 0
held-out FB P95 <= 1.5 px
held-out RGB-D residual P95 <= 1.5 px
```

禁止采用来源方案中的48 px residual和0.70–1.40生产范围。超过8 px时进入救援源流程，不扩大形变。

### L4：Direct-ORB rescue source

按第6节插入一个真实 direct-ORB source，形成两条较短基线边，并从L0重新审计。

### LF：Fail closed

L4后仍不能通过时，该 seam 和候选失败。禁止宽羽化或跳过gate。

---

## 10. 风险分级与PairPlan

```python
@dataclass(frozen=True)
class SeamPairPlan:
    left_frame_id: int
    right_frame_id: int
    predicted_boundary_x: float
    corridor_x0: int
    corridor_x1: int
    risk_level: int
    alignment_level: int
    require_open3d: bool
    require_depth_mesh: bool
    require_object_lock: bool
    rescue_candidate_frame_ids: tuple[int, ...]
```

风险等级：

| 等级 | 含义 | 最重允许处理 |
|---|---|---|
| R0 | 低梯度、低残差、无深度风险 | L1 + 安全窄带融合 |
| R1 | 结构/光度风险、共同可见 | L1/L2 + 可弯 seam |
| R2 | 局部视差或深度层风险 | L2/L3 |
| R3 | 紧凑对象或不可warp前景 | 对象单owner + seam绕行 |
| RF | 无可审计路径 | fail closed |

任何失败pair不得为了守时降报风险等级。

---

## 11. 对象单Owner

对象处理不以全图语义重建为目标，只处理会被最终seam切割的紧凑风险对象。

owner选择依据：

```text
完整可见性
+source support
+aligned-depth有效率
+遮挡关系
+清晰度
+source中心距离
```

硬门：

```text
minimum full support = 0.98
maximum handoffs = 0
object flow/warp = false
object MultiBand = false
object internal seam count = 0
```

对象颜色只能从所选真实source的一次inverse remap获得。annotation类别和坐标不能进入数据面。

---

## 12. Seam求解

### 12.1 求解器

采用仓库契约允许的受限GraphCut或等价单调label optimizer。为避免与现有禁止项冲突，不采用字面上的动态规划backend。

要求：

- 只允许相邻真实源在互斥corridor竞争；
- owner沿扫描方向单调；
- 每个有效像素恰有一个owner；
- 无owner回头、孤岛和小碎片；
- 半分辨率求解后仅在full-resolution小半径内细化；
- 不得演变为全景级多标签联合优化。

### 12.2 Seam cost

合并两份方案后采用：

```text
E = 0.25 × Lab residual
  + 0.20 × gradient magnitude residual
  + 0.10 × gradient direction residual
  + 0.15 × flow/RGB-D uncertainty
  + 0.15 × depth/occlusion protection
  + 0.10 × automatic long-line cutting risk
  + 0.05 × sharpness/source-centre preference
```

权重只在development split搜索，validation前冻结。

### 12.3 硬禁止区

软代价不能覆盖硬禁止：

```text
对象锁核心
强深度边缘
occlusion/disocclusion
深度孔洞扩展区
RAFT FB不可靠区
mesh residual超限区
自动长线不一致区
透明/反光保护区
```

无可行seam时依次：

```text
移动seam
→扩大单一真实source owner
→插入direct-ORB rescue source
→fail closed
```

---

## 13. Photometric与MultiBand

### 13.1 全局线性颜色

只使用共同可见、安全背景样本：

```text
per-source linear RGB gain/bias
时间一阶/二阶正则
中位曝光anchor
训练/held-out分离
```

冻结范围：

```text
gain in [0.85, 1.18]
bias in [-12, 12] for uint8 domain
held-out DeltaE00 P95 < 3.0
```

超限则回退单位校正，不允许链式累计失控。

### 13.2 安全背景MultiBand

只有几何门已通过时才允许：

```text
共同有效
低梯度
无对象锁
无深度/遮挡保护
无自动长线风险
已通过局部几何对齐
总带宽2–8 px/pair
最多3层
```

对象、深度边缘和风险区blend pixel count必须为0。

禁止采用12–20 px生产融合带、宽feather或全图金字塔；MultiBand不得把双边变成重影。

---

## 14. 接缝质量硬门

### S1：几何

每条最终seam：

```text
seam-band held-out FB residual P95 <= 1.0 px target
seam-band held-out FB residual P95 <= 1.5 px hard maximum
seam-band maximum residual <= 2.0 px
RGB-D residual P95 <= 1.5 px
automatic line step P95 <= 1.0 px
automatic line orientation delta P95 <= 3 degrees
mesh maximum displacement <= 8 px
mesh minimum Jacobian > 0
protected-region intersection = 0
```

### S2：对象和Owner

`annotations_v2`只在renderer完成后评价：

```text
central_carton owner_count = 1
black_fan owner_count = 1
black_cable owner_count = 1
compact-object full-support >= 98%
object internal seam count = 0
object blend pixel count = 0
unowned valid pixels = 0
invalid owner IDs = 0
```

### S3：光度和显著性

```text
safe-background DeltaE00 P95 < 3.0
brightness step P95 <= 1.5%
gradient jump ratio P95 <= 1.15 target
gradient jump ratio P95 <= 1.25 hard maximum
seam_saliency_p95 <= 0.10
明显亮度竖带 count = 0
ghost/wide-blur pair count = 0
```

`seam_saliency_score`定义为沿seam的颜色/梯度异常相对于两侧8–16 px control band的归一化值。公式和阈值必须在development固定。

### S4：结构

```text
seam crossing strong depth edge count = 0
seam crossing protected line mismatch count = 0
owner temporal reversal count = 0
owner islands/small fragments = 0
unowned pixel count = 0
invalid sampling count = 0
all final sources are real RGB-D = true
all final poses are direct ORB = true
```

### S5：固定盲评

自动导出：

- 每条真实seam的160 px 100% crop；
- 横梁、桌面前沿、立柱、纸箱、风扇、线缆附近crop；
- 数量相同、尺寸相同的非seam control crop；
- 随机顺序，隐藏owner、算法和seam位置。

门槛：

```text
1:1：明确错位、双边、断线、台阶 count = 0
fit-width：周期性竖接缝 count = 0
seam/control人工识别率 <= 60%
被标记为“明显接缝”的真实seam count = 0
```

盲评结果必须落盘，不能只通过聊天截图确认。

---

## 15. 20秒预算

计划预算：

| 阶段 | 目标 |
|---|---:|
| online state严格验证 | 0.8 s |
| base/rescue source plan | 0.4 s |
| 尾部ORB与全部Open3D CUDA边 | 1.8 s |
| 每源一次RGB/depth remap | 3.5 s |
| L1/L2/L3局部对齐 | 3.5 s |
| 对象owner和seam优化 | 2.0 s |
| photometric/MultiBand | 1.8 s |
| 编码、provenance、原子发布 | 1.2 s |
| 调度/I/O/驱动余量 | 5.0 s |
| 合计 | 20.0 s |

预算控制不能降载：

- seam几何门；
- 自动长线门；
- 深度/遮挡保护；
- 对象单owner；
- provenance；
- 全部final edges Open3D审计；
- fail-closed。

可在候选期按顺序降低：

1. 非最高风险pair的额外RAFT iteration。
2. RAFT analysis width 384→320，但必须重新通过S1。
3. MultiBand 3层→2层。
4. normal source advance 18→20→24 px，失败seam仍允许有限rescue。
5. 关闭非主交付审计导出。

不得自动降低输出高度；若未来需要360/400 px产品，必须作为独立规格重新冻结，不得作为超时回退。

---

## 16. 性能熔断

禁止再次未经预测直接运行数百秒完整3 m。

候选执行顺序：

```text
合成corridor测试
→固定validation split
→P25前缀
→P50前缀
→P75前缀
→P100时间/显存投影
→投影<=20 s才允许完整3 m
```

投影必须按实际pair风险分布、source count、RAFT count、rescue count和输出宽度计算，不能只按帧数线性外推。

一旦满足任一条件立即停止完整候选：

- 预测post-capture >20 s；
- render sources >48；
- R2 pairs >8；
- rescue sources >4；
- GPU projected peak >6.5 GB；
- 任何硬视觉门没有可行恢复链；
- 实际Open3D不是tensor CUDA。

---

## 17. 候选消融序列

### SF0：结构与在线基线

```text
direct ORB sources
+online state v3
+每源一次remap
+hard owner
+S1–S5指标
```

目标：证明数据面、source上限和<=12 s结构基线，不要求视觉通过。

### SF1：L1对齐和受限seam

```text
SF0
+phase translation/DIS
+自动长线
+受限GraphCut/单调label seam
```

### SF2：Risk-only RAFT

```text
SF1
+batched RAFT-small for failed corridors
```

### SF3：RGB-D局部分层mesh

```text
SF2
+far/mid bounded mesh
+near/occlusion owner-only
```

### SF4：对象单Owner和救援源

```text
SF3
+risk-triggered object mask refinement
+single-owner selection
+direct-ORB rescue source
```

### SF5：Photometric和安全MultiBand

```text
SF4
+global linear photometric graph
+2–8 px/<=3-level safe MultiBand
```

选择原则：

```text
先淘汰任何S1–S5失败候选；
再淘汰任何一次>20 s候选；
从剩余候选中选择P95最快、显存最低、模块最少者。
```

不得因为后续候选模块更多就默认其更好。

---

## 18. 核心结果接口

```python
@dataclass
class SeamAlignmentResult:
    left_frame_id: int
    right_frame_id: int
    alignment_level: int
    source_grid_left: object
    source_grid_right: object
    valid_left: object
    valid_right: object
    fb_residual_p95_px: float
    rgbd_residual_p95_px: float
    maximum_residual_px: float
    line_step_p95_px: float
    line_orientation_p95_degrees: float
    minimum_jacobian: float
    passed: bool
    failure_reasons: list[str]
```

```python
@dataclass
class SeamQualityResult:
    delta_e00_p95: float
    brightness_step_percent_p95: float
    gradient_jump_ratio_p95: float
    seam_saliency_p95: float
    strong_depth_edge_crossings: int
    protected_line_crossings: int
    object_internal_seam_count: int
    passed: bool
    failure_reasons: list[str]
```

```python
@dataclass
class SeamFirstPanoramaResult:
    panorama_bgr: object
    owner_frame_id: object
    source_frame_ids: tuple[int, ...]
    pair_plans: tuple[SeamPairPlan, ...]
    alignment_results: tuple[SeamAlignmentResult, ...]
    quality_results: tuple[SeamQualityResult, ...]
    performance_audit: dict[str, object]
```

---

## 19. 推荐候选配置

以下均为candidate内部冻结参数，普通用户不能调节：

```yaml
stitch:
  video_panorama:
    candidate_family: seam_first_20s_v1

    performance:
      stretch_cold_seconds: 15.0
      p50_max_seconds: 16.0
      p95_max_seconds: 18.5
      hard_maximum_seconds: 20.0
      abort_full_scan_projection_seconds: 20.0

    source_selection:
      direct_orb_only: true
      normal_advance_pixels: 20
      risk_advance_pixels: 10
      maximum_advance_pixels: 24
      base_source_minimum: 36
      base_source_maximum: 44
      maximum_rescue_sources: 4
      maximum_final_sources: 48
      maximum_rescue_sources_per_seam: 1

    seam_corridor:
      width_pixels: 160
      minimum_width_pixels: 96
      maximum_width_pixels: 160
      analysis_width_pixels: 384

    alignment:
      level1_backend: phase_translation_dis
      level2_backend: raft_small
      raft_fp16: true
      raft_batch_pairs: 6
      maximum_raft_pairs: 8
      require_forward_backward: true

    depth_mesh:
      enabled: true
      layers: [far, mid, near]
      maximum_displacement_pixels: 8
      minimum_scale: 0.80
      maximum_scale: 1.25
      require_positive_jacobian: true
      taper_to_identity: true

    object_owner:
      proposal_only_near_candidate_seams: true
      maximum_refinement_keyframes: 12
      minimum_full_support: 0.98
      maximum_handoffs: 0
      allow_object_warp: false
      allow_object_multiband: false

    seam:
      backend: constrained_graphcut_monotone_labels
      full_resolution_refine_radius_pixels: 4
      forbid_owner_islands: true

    photometric:
      backend: global_linear_rgb_graph
      gain_minimum: 0.85
      gain_maximum: 1.18
      bias_minimum_uint8: -12
      bias_maximum_uint8: 12
      held_out_delta_e00_p95_max: 3.0

    multiband:
      enabled: true
      safe_background_only: true
      total_width_pixels: 8
      maximum_levels: 3

    quality_gate:
      fb_residual_p95_target_pixels: 1.0
      fb_residual_p95_hard_max_pixels: 1.5
      residual_max_pixels: 2.0
      rgbd_residual_p95_max_pixels: 1.5
      line_step_p95_max_pixels: 1.0
      line_orientation_p95_max_degrees: 3.0
      delta_e00_p95_max: 3.0
      brightness_step_p95_max_percent: 1.5
      gradient_jump_ratio_p95_hard_max: 1.25
      seam_saliency_p95_max: 0.10
      strong_depth_edge_crossings_max: 0

    runtime:
      require_complete_online_state: true
      require_open3d_tensor_cuda: true
      resident_source_window: 5
      maximum_gpu_gb: 6.5
      maximum_cpu_working_set_gb: 4.0
```

---

## 20. 预计代码边界

优先修改：

```text
src/panorama_demo/video_panorama.py
src/panorama_demo/video_pipeline.py
src/panorama_demo/video_online_state.py
src/panorama_demo/video_source_selection.py
src/panorama_demo/video_motion_resampler.py
src/panorama_demo/video_dense_real_frame_layout.py
src/panorama_demo/video_dense_pose_prior.py
src/panorama_demo/video_visual_renderer_v2.py
src/panorama_demo/video_visual_metrics.py
src/panorama_demo/video_budget_controller.py
src/panorama_demo/video_delivery.py
configs/demo.yaml
README.md
AGENTS.md
```

建议新增：

```text
src/panorama_demo/video_seam_first_pipeline.py
src/panorama_demo/video_online_seam_evidence.py
src/panorama_demo/video_seam_corridor.py
src/panorama_demo/video_seam_alignment.py
src/panorama_demo/video_seam_raft.py
src/panorama_demo/video_seam_depth_layers.py
src/panorama_demo/video_seam_mesh.py
src/panorama_demo/video_seam_optimizer.py
src/panorama_demo/video_object_tracks.py
src/panorama_demo/video_seam_rescue.py
src/panorama_demo/video_seam_photometric.py
src/panorama_demo/video_seam_multiband.py
src/panorama_demo/video_seam_quality.py
src/panorama_demo/video_seam_blind_eval.py
```

保留为证据或隔离历史候选，不作为20秒正式renderer backend或失败回退：

```text
src/panorama_demo/video_dense_real_frame_layout.py（D1 evidence-only）
src/panorama_demo/video_d2_monotonic_depth_layer_warp.py
src/panorama_demo/video_d3_object_first_compositor.py
configs/video_candidates/D3_object_first_dense_source_compositor.yaml
```

注意：来源方案中的 `video_monotonic_depth_layer.py` 和 `video_object_first_compositor.py` 不是当前实际文件名，实施时必须使用上面的真实模块。

---

## 21. 实施与Commit顺序

### Commit 1：冻结证据与候选身份

```text
记录509秒D3结果和输入哈希
新增seam_first_20s_v1候选家族
禁止旧D3进入production route
```

### Commit 2：质量指标和性能熔断

```text
S1–S5指标
seam_saliency
前缀性能投影
>20秒完整运行熔断
```

### Commit 3：Evidence/Render Source解耦

```text
60 FPS evidence pool
direct ORB render pool
36–44 base / <=48 final source contract
```

### Commit 4：Online state v3和DIS去重

```text
哈希绑定
bidirectional DIS只算一次
风险/长线/photometric统计
```

### Commit 5：一次Remap和CUDA Open3D

```text
RGB/depth共享grid
3–5 source resident window
全部final edges tensor CUDA audit
```

### Commit 6：L1和受限Seam

```text
phase translation + DIS
自动长线
constrained graphcut/monotone labels
```

### Commit 7：Batched RAFT

```text
risk-only corridor
单实例FP16
batch difficult pairs
```

### Commit 8：RGB-D局部Mesh

```text
far/mid bounded mesh
near/occlusion owner-only
8 px/positive Jacobian/identity boundary
```

### Commit 9：对象Owner和救援源

```text
risk-triggered generic mask refinement
compact object single owner
direct-ORB rescue source
```

### Commit 10：Photometric和安全MultiBand

```text
global linear graph
held-out audit
2–8 px/<=3 levels
```

### Commit 11：SF0–SF5消融和冻结

```text
development搜索
validation选型
holdout一次
10次性能重复
3次新采现场复测
production lock
```

每个commit必须独立通过相关测试，不得把生命周期、数据链和全部renderer一次性混在一个commit。

---

## 22. 测试要求

新增测试至少包括：

```text
tests/test_video_online_seam_evidence.py
tests/test_video_render_source_decoupling.py
tests/test_video_seam_corridor.py
tests/test_video_seam_alignment.py
tests/test_video_seam_raft.py
tests/test_video_seam_mesh.py
tests/test_video_object_tracks.py
tests/test_video_seam_rescue.py
tests/test_video_seam_optimizer.py
tests/test_video_seam_photometric.py
tests/test_video_seam_multiband.py
tests/test_video_seam_quality.py
tests/test_video_seam_blind_eval.py
tests/test_video_20s_budget.py
```

必须验证：

1. evidence frame不能成为未授权owner。
2. final render source pose全部为direct ORB。
3. online state任一输入/配置/模型哈希变化即拒绝复用。
4. 每条60 FPS边只计算一套bidirectional DIS。
5. 每个final source只生成一次inverse grid并只上传一次。
6. 全部final edges在发布前通过tensor CUDA Open3D。
7. RAFT只在失败corridor运行且全会话<=8 pair。
8. mesh位移<=8 px、正Jacobian、边界identity。
9. 风险mask与MultiBand区域零交集。
10. 紧凑对象单owner且object blend=0。
11. seam不穿过强深度边缘或保护长线。
12. rescue source必须真实、direct ORB、全会话<=4。
13. owner单调、无孤岛、无未归属有效像素。
14. geometry失败时禁止MultiBand和宽feather。
15. summary/minimal与audit panorama/owner hash相同。
16. 20秒计时包含编码和delivery最后原子写入。
17. 预测>20秒时禁止完整3 m运行。

合成场景至少包括：

```text
3 px横梁台阶
2 px垂直偏差
亮度阶跃
近景/远景双层视差
seam穿过深度边缘
细线缆穿过名义seam
插入中间direct-ORB源可修复pair
无法修复且应fail-closed pair
```

常规验证：

```powershell
$G305Python = 'D:\Panoramic_Camera\.conda\python.exe'
& $G305Python -m pytest -q
ruff check src tests
& $G305Python -m compileall -q src tests
git diff --check
```

---

## 23. 性能与现场验收

### 23.1 固定会话

```text
2次warm-up，不计分
10次正式重复
```

通过条件：

```text
10/10 primary_post_capture_seconds <=20.0
P50 <=16.0
P95 <=18.5
panorama decoded pixel hash一致
owner raw hash一致
source IDs一致
S1–S5每次一致通过
GPU peak <=6.5 GB
CPU aggregate working set <=4 GB
OOM=0
unexpected CPU fallback=0
```

### 23.2 新采现场

固定会话通过后，同一硬件至少新采3次3 m：

1. 低纹理墙面/大平面。
2. 货架、横梁、立柱和长直线丰富场景。
3. 纸箱、风扇、线缆等紧凑前景场景。

每次必须同时通过20秒、S1–S5和真实CUDA/Open3D/ORB审计。合成测试不能替代现场验收。

### 23.3 长度投影

3 m冻结后才允许20 m性能投影和现场脚本：

```text
20 m conservative projection <=55 s：伸展目标
20 m必须启用tile资源边界
20 m不得改变3 m冻结算法、pose、owner或像素规则
```

20 m投影不是当前3 m production freeze的替代条件。

---

## 24. 最终交付证据

正式2-D：

```text
video_panorama.jpg
video_panorama.png
video_pixel_provenance.npz
video_report.json
video_timing.json
video_delivery.json
```

候选验收附加：

```text
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
  prefix_projection.json
  gpu_memory.json
  cuda_backend_audit.json
```

`video_delivery.json`最后写入并记录：

```json
{
  "algorithm_id": "production_seam_first_20s_v1",
  "primary_post_capture_seconds": 0.0,
  "maximum_post_seconds": 20.0,
  "within_post_capture_budget": true,
  "seam_quality_pass": true,
  "object_integrity_pass": true,
  "blind_seam_evaluation_pass": true,
  "base_render_source_count": 0,
  "rescue_render_source_count": 0,
  "direct_orb_render_pose_count": 0,
  "interpolated_render_pose_count": 0,
  "actual_open3d_backend": "open3d_tensor_cuda_rgbd"
}
```

---

## 25. Production Freeze

只在以下条件全部满足时生成：

```text
configs/video_production/production_seam_first_20s_v1.yaml
production.lock.json
user_20m_test.ps1（仅在3 m冻结后）
```

冻结条件：

```text
S1几何通过
S2对象/owner通过
S3光度/显著性通过
S4结构通过
S5盲评通过
固定会话10次性能通过
3次新采现场通过
所有final sources为真实RGB-D
所有final poses为direct ORB
所有final edges为Open3D tensor CUDA审计
无OOM和非法fallback
全量测试通过
```

冻结后公共 `g305-video-panorama` 只能读取该lock，不允许普通用户调节source、flow、mesh、seam、blend、对象或预算参数。

---

## 26. 明确不采用的来源方案内容

为防止后续实施重新引入冲突，明确排除：

```text
不允许对象“个别不保证单owner”
不允许production rescue source使用interpolated/refined pose prior
不允许关闭正式全边Open3D或发布后补审计
不允许48 px局部形变
不允许scale 0.70–1.40作为production范围
不允许12–20 px MultiBand或宽feather
不采用字面DP backend
不自动把输出高度480降至400/360
不允许35–60基础源再无界增加rescue source
不把warm 8–10 s未实测目标写成已达成事实
```

---

## 27. Go / No-Go

### Go

```text
SF0–SF5逐阶段完成
S1–S5全部通过
10/10固定会话<=20 s
P95<=18.5 s
3次新采全部通过
全量测试通过
```

满足后才继续D4/D5或直接进行production freeze评审。

### No-Go

任一情况立即停止：

- 任一真实seam有明确错位、双边、断线或台阶；
- 纸箱、风扇或线缆不是单owner；
- 通过宽带模糊才隐藏接缝；
- 任一次正式重复超过20秒；
- final sources超过48；
- 需要超过8个RAFT pair或4个rescue sources；
- Open3D实际回退CPU；
- annotation进入renderer数据面；
- online state不完整却被正式复用；
- 插值pose或非真实源成为正式owner。

最终原则：

```text
用采集期在线证据、稀疏direct-ORB渲染源和性能熔断控制时间；
用逐级局部对齐、有限救援帧、对象单owner和长线保护保证几何；
用受限seam、全局线性颜色和2–8 px安全融合消除可见接缝；
不用全图RAFT、全画布D3、宽形变或宽羽化换取表面上的“无缝”。
```
