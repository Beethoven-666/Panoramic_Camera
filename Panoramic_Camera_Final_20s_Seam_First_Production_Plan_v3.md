# Panoramic_Camera 最终 Production 方案 v3

## 3 m 停采后 20 秒内近无痕 RGB-D 视频全景，以及受控的 20 m 准入

## 0. 文档权威性

本文档合并并修正以下方案：

```text
D:\central_strip_Panoramic_Camera\Panoramic_Camera_Phase3_20s_Seamless_Video_Panorama_Plan.md
D:\central_strip_Panoramic_Camera\Panoramic_Camera_Final_20s_Seam_First_Production_Plan.md
C:\Users\zyh68\Downloads\Panoramic_Camera_Seam_First_Speed_Controlled_Production_Plan.md
C:\Users\zyh68\Downloads\Panoramic_Camera_Latest_Final_20s_Seam_First_Production_Plan_v2.md
```

从本方案开始，后续程序修改、候选消融、性能验收、3 m Production Freeze 和 20 m 现场准入均以本文为唯一方案依据。旧方案保留作历史参考；实施本方案时应在旧方案首行标记 `SUPERSEDED BY Panoramic_Camera_Final_20s_Seam_First_Production_Plan_v3.md`，不删除原文件和证据。

若本文与当前可执行源码、默认配置、测试或 `AGENTS.md` 不一致，先以源码、配置、测试和 `AGENTS.md` 为准，再在同一改动中修正文档。不得通过重命名算法绕过既有禁令。

适用范围：

```text
产品：独立连续 RGB-D 视频全景
正式入口：g305-video-panorama
研发入口：g305-video-experiment
相机：Orbbec Gemini 305
输入：848 × 480 @ 60 FPS aligned RGB-D
正式 tracking FPS：8.0
目标硬件：RTX 5060 Laptop GPU，8 GB
固定研发会话：
D:\central_strip_Panoramic_Camera\data\captures\video\run_20260806_153033
开发分支：codex/dense-anchor-recovery
```

本文不修改照片模式 `g305-panorama`。

本文描述待实现、待实测的候选，不代表当前代码已经达到 20 秒或近无痕目标。全部硬门通过前，不得创建或更新 Production lock。

---

## 1. 最终决策摘要

### 1.1 只允许真实 Direct-ORB 正式源

Production v1：

```text
base render source：真实 RGB-D + direct ORB camera_to_world
rescue render source：真实 RGB-D + direct ORB camera_to_world
interpolated_render_pose_count：0
```

audited dense real-frame pose prior 仅保留为候选研究能力：

```text
candidate_only = true
production_v1_selection_eligible = false
```

### 1.2 渲染源硬上限

```text
base source 目标范围：36–44/3 m
base source 硬上限：44
每条 seam rescue：最多1张
每个3 m会话 rescue：最多4张
final render sources：最多48张
```

36不是通用结构下限。实际数量由真实扫描跨度、共同有效覆盖和空间推进量决定；不得为了凑36张加入冗余源。

### 1.3 局部形变严格有界

```text
maximum displacement <= 8 px
local scale in [0.80, 1.25]
minimum Jacobian > 0
corridor outer boundary displacement = 0
```

超过8 px时不扩大形变，进入 direct-ORB rescue；仍失败则 fail closed。

### 1.4 固定输出规格

Production v1固定输出高度480 px。同一lock不得自动执行：

```text
480 → 400 → 360
```

未来若需要360p，必须冻结独立产品，例如：

```text
production_seam_first_360p_v1
```

### 1.5 3 m与20 m分级

```text
3 m：单GPU canvas
20 m：2048 px tile + 128 px overlap
```

20 m tile只改变资源调度，不能改变pose、source、risk、warp、seam、owner、blend或质量门。

状态严格拆分：

```text
production_3m_pass
twenty_metre_test_authorized
twenty_metre_validated
```

3 m通过不等于20 m已通过；3 m前缀投影只能授权20 m现场测试，不能宣称20 m已验证。

### 1.6 发布前全部 Final Edges真实审计

每条最终相邻source edge必须在发布前满足：

```text
backend = open3d_tensor_cuda_rgbd
quality_pass = true
```

Open3D只做真实相邻边结构审计，不生成pose、不修改ORB轨迹、不替代seam质量门。

### 1.7 普通用户不调算法参数

公共入口只读取冻结的 `production.lock.json`。普通用户不能通过CLI覆盖20秒、source、flow、mesh、seam、blend或输出规格。

禁止正式用法：

```powershell
g305-video-panorama --maximum-post-seconds 20 ...
```

20秒必须冻结在Production lock中。

---

## 2. 产品硬目标

### 2.1 优先级

```text
第一：接缝几何真实对齐；
第二：接缝光度连续，正常观看几乎不可见；
第三：紧凑对象完整、长直线连续；
第四：3 m停采后20秒内完成2-D主交付；
第五：在全部硬门内选择P95最快、显存最低、模块最少的实现。
```

### 2.2 视觉硬目标

1. 每条最终seam在融合前通过局部几何门。
2. 横梁、桌边、立柱、货架板、管道无明显台阶、断裂、双边或方向突变。
3. 安全背景无明显亮度竖带、颜色跳变或宽模糊。
4. `central_carton`、`black_fan`、`black_cable` 各由一个真实source完整拥有。
5. 对象、深度边缘、遮挡、disocclusion、孔洞、透明/反光风险区禁止MultiBand。
6. 每个最终颜色来自真实RGB的一次标定inverse remap。
7. provenance能诚实解释单源和双源安全融合颜色。
8. 禁止插值帧、伪造帧、伪造pose、二维flow pose和非真实源颜色。

### 2.3 3 m性能硬目标

计时：

```text
起点：最后一帧正式RGB-D完整写盘，设备与writer安全关闭
终点：video_delivery.json最后原子写入完成
```

冻结门：

```text
10/10正式运行 <=20.0 s
P50 <=16.0 s
P95 <=18.5 s
P95余量 >=1.5 s
```

优化目标：

```text
P50 10–12 s
```

优化目标不能通过关闭质量门获得。

### 2.4 失败原则

任一质量或性能硬门失败：

- 不生成/更新Production lock；
- 不把明显接缝标为A/B；
- 不静默回退旧hard-owner冒充Production；
- 不通过宽羽化、扩大形变或降分辨率掩盖失败；
- 结构完整时可按现有产品政策发布 `published_degraded` C级并强制人工复核；
- 结构不完整时写failure，不发布正式delivery；
- 失败结果不能设置 `production_3m_pass=true`。

---

## 3. 生命周期与真实算法身份

继续使用：

```text
baseline
candidate
production
report-level
artifact-level
```

禁止重新引入用户可见 `fast/quality/audit preset`。

当前真实Baseline身份必须保持：

```text
algorithm_id = legacy_fast_b07b561
lock = configs/video_algorithms/baseline_legacy_fast_b07b561.lock.json
```

不得在方案或实现中改名为 `legacy_fast_baseline`。

所有SF0–SF5研发只通过 `g305-video-experiment`。Production未冻结时，`g305-video-panorama`必须明确失败，不得自动把baseline冒充Production。

---

## 4. 当前证据与停止项

固定新会话已知：

```text
D1 source/pose/dense evidence：通过
D2 depth-layer/RAFT/long-line：通过
D3 validation black_fan：单owner通过
D3 validation central_carton：34 owners，失败
D3 validation black_cable：26 owners，失败
```

完整3 m D3：

```text
scan frames = 789
render sources = 399
Open3D edges = 398
primary post-capture = 509.27 s
```

停止进入Production：

```text
399-source full-scan D3
全图/全pair RAFT
停采后398边Open3D
全画布object compositor
全景级flow/free mesh
5–7帧全画布联合优化
```

D1/D2/D3可保留为研究证据，不作为Production v1 renderer或fallback。

---

## 5. 四类帧集合

### 5.1 Motion Evidence Frames

```text
60 FPS真实RGB-D
```

只用于motion、DIS、RGB-D residual、risk、line/object传播和rescue评分，不能直接拥有正式像素。

### 5.2 ORB Anchor Frames

```text
Production tracking FPS = 8.0
全部为direct ORB anchors
```

不得在Production v1自动提升到12 FPS。任何改变都必须建立新candidate并重新验证完整轨迹、性能和现场结果。

### 5.3 Base Render Sources

只从direct ORB anchors选择：

```text
normal advance：18–24 full-resolution px
risk advance：8–12 px
推荐：normal=20，risk=10
target count：36–44/固定3 m
hard maximum：44
首尾source强制保留
```

### 5.4 Rescue Render Sources

只允许未入选的direct ORB anchor：

```text
per seam <=1
per session <=4
base+rescue <=48
```

插入后形成的左右新边都必须通过真实Open3D Tensor CUDA审计。

---

## 6. Online State v3

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

复用必须同时满足：

1. manifest、calibration、frames.csv哈希一致。
2. 所有引用RGB/aligned-depth文件哈希一致。
3. config、模型、Open3D working config哈希一致。
4. 所有记录只引用真实帧。
5. `last_complete_frame_id`覆盖正式扫描尾部。
6. 会话安全关闭后最后原子写入 `complete_and_cleanly_closed=true`。

失败时整份拒绝复用，不部分复用、不跨会话迁移、不使用旧标注坐标补齐。

### 6.1 DIS去重

每条相邻60 FPS edge只运行：

```text
forward DIS一次
backward DIS一次
```

同一证据复用于FB consistency、双向RGB-D residual、occlusion、risk、line/object传播和rescue评分。禁止重复为两个方向各算一套bidirectional DIS。

---

## 7. Open3D Candidate-Edge Lattice

### 7.1 Lattice覆盖

采集期基于当前direct-ORB source plan滚动预计算：

- 当前最优base plan的相邻边；
- normal/risk推进量可能产生的高优先级替代边；
- 每条高风险seam首选rescue形成的左右边；
- 首尾和尾部边。

每条记录绑定：

```text
left/right frame IDs
left/right RGB-D SHA
left/right ORB pose SHA
working intrinsics SHA
Open3D config SHA
actual backend
quality result
```

不得组合多条短边冒充一条final edge。

### 7.2 有界调度

```text
maximum candidate-edge records：128（初始candidate上限，SF0实测后冻结）
maximum pending Open3D jobs：16
优先级：当前base边 > rescue边 > alternative边
```

队列触顶时只能丢弃最低优先级未执行candidate edge，不能丢RGB-D帧、ORB anchor或正式写盘。停采后final edge未命中必须在20秒计时内真实补算。

### 7.3 Gate O3D-PRECOMPUTE

进入20秒完整候选前必须实测：

```text
正式RGB-D dropped frame count = 0
ORB anchor生成不阻塞
Open3D pending queue不持续增长
candidate-edge records <=冻结上限
final edge cache hit rate >=90%
post-capture missing edge fill <=1.8 s
全部final edges发布前审计完成
全部actual backend = open3d_tensor_cuda_rgbd
```

若Gate失败，不能继续使用理论20秒预算；先修正lattice、调度或CUDA实现。

---

## 8. 停采后主数据流与预算

```text
验证online_state
→选择base direct-ORB sources
→匹配Open3D edge cache
→逐seam处理并按需选择0–4 rescue sources
→补算全部缺失final edges
→每source一次full-resolution RGB/depth inverse remap
→pair risk
→L0–L4局部对齐
→对象单owner
→受限seam
→global photometric
→2–8 px safe MultiBand
→sparse provenance v2
→JPEG/PNG/report/timing/delivery原子发布
```

预算：

| 阶段 | 目标 |
|---|---:|
| online state验证 | 0.8 s |
| source/rescue plan | 0.4 s |
| Open3D复核/补算 | 1.8 s |
| 每源一次remap | 3.5 s |
| 局部对齐 | 3.5 s |
| object owner + seam | 2.0 s |
| photometric + MultiBand | 1.8 s |
| provenance/编码/原子发布 | 1.2 s |
| I/O、驱动、尾延迟余量 | 5.0 s |
| 总计 | 20.0 s |

预算必须实测，不得把主路径阶段挪到offline timer规避。

---

## 9. Pair风险与升级链

| 风险 | 含义 | 最重处理 |
|---|---|---|
| R0 | 低梯度、低残差、无深度风险 | L1 + 安全融合 |
| R1 | 结构/光度风险、共同可见 | L1/L2 + seam |
| R2 | 视差、深度层、遮挡 | L2/L3 |
| R3 | 紧凑对象或不可warp前景 | 单owner + seam绕行 |
| RF | 无可审计路径 | fail closed |

上限：

```text
R2 RAFT pairs <=8
R3 object refinements <=12
final sources <=48
resident sources <=5
```

超过上限不得降报风险。

### L0：粗布局

使用direct ORB pose、scan coordinate和calibrated support建立名义boundary和96–160 px corridor，不修改pose。

### L1：廉价局部对齐

半分辨率corridor使用：

```text
bounded phase translation
+既有DIS FB
+自动长线
```

只允许低自由度局部修正，不得构成全图affine/homography或二维pose替代。

### L2：Risk-Only Batched RAFT-Small

仅处理L1失败且仍有可恢复共同背景的pair：

```text
input width=384 px
FP16 CUDA
单模型实例
batch=4–8 pairs
forward/backward=全部R2
R2/session<=8
```

禁止全图和普通pair RAFT。

### L3：RGB-D Far/Mid Mesh

1. aligned depth双向重投影和z-buffer。
2. far/mid/near/occlusion/disocclusion分类。
3. near、对象、深度边缘、孔洞、强RGB结构owner-only。
4. 只对far/mid安全共同背景拟合有界单调B-spline/规则mesh。
5. corridor外边界回到identity。

硬门：

```text
displacement<=8 px
scale=[0.80,1.25]
minimum Jacobian>0
outer boundary displacement=0
protected intersection=0
held-out FB P95<=1.5 px
held-out RGB-D P95<=1.5 px
```

### L4：Direct-ORB Rescue

插入一张真实direct-ORB anchor，形成两条较短边，从L0重新审计。

```text
per seam<=1
per session<=4
```

### LF：Fail Closed

L4后仍失败则RF。禁止扩大corridor超过160 px、扩大warp或宽羽化。

---

## 10. 对象单Owner

运行时proposal只来自：

- depth connected surface/discontinuity；
- RGB closed gradient/细长强结构；
- flow residual；
- occlusion/disocclusion；
- 多帧稳定性；
- 可选通用mask refinement。

固定frame 389/391 annotations严格为post-render evaluation，不能作prompt、mask、source选择、warp约束或训练数据。

初始通用refinement候选可使用本地锁定的SAM 2.1 Hiera Tiny，但必须先完成许可证确认，固定权重SHA，无下载回退；它只能生成保护mask。

硬门：

```text
central_carton owner_count=1
black_fan owner_count=1
black_cable owner_count=1
full support>=98%
handoff=0
internal seam=0
object blend=0
```

对象颜色只来自所选真实source的一次inverse remap。

---

## 11. Seam求解器

### 11.1 正式允许边界

采用当前契约允许的：

```text
constrained binary GraphCut
或经正式契约确认允许的monotone graph-label optimizer
```

当前不采用 `monotone_shortest_path`、动态规划或通过改名伪装的DP backend。若未来要A/B测试此类solver，必须先明确修订 `AGENTS.md` 并建立独立candidate。

### 11.2 Seam Cost

development搜索、validation前冻结：

```text
0.25 Lab residual
+0.20 gradient magnitude residual
+0.10 gradient direction residual
+0.15 flow/RGB-D uncertainty
+0.15 depth/occlusion protection
+0.10 automatic line cutting risk
+0.05 sharpness/source-centre preference
```

硬禁止区优先：对象锁、强深度边缘、遮挡、孔洞、FB不可靠、mesh超限、长线不一致、透明/反光保护。

无可行seam：移动seam→扩大单一真实source owner→direct-ORB rescue→RF。

---

## 12. Photometric与MultiBand

### 12.1 全局线性颜色图

只使用真实共同可见、安全背景：

```text
adjacent overlap
+skip-one overlap（仅在真实共同可见且安全时）
+temporal regularization
+median exposure anchor
+training/held-out split
```

范围：

```text
gain=[0.85,1.18]
bias=[-12,12]（uint8等效）
held-out DeltaE00 P95<3.0
```

超限回退单位校正，禁止链式累计失控。

### 12.2 安全MultiBand

仅当共同有效、几何通过、低梯度、无对象/深度/遮挡/长线风险时允许：

```text
total width=2–8 px/pair
levels<=3
```

对象、深度边缘、细线缆、遮挡和强结构 `blend_pixel_count=0`。MultiBand不得修复几何双边。

---

## 13. Sparse Pixel Provenance v2

### 13.1 语义

每个有效像素仍有且只有一个 `dominant_owner_frame_id`。

单源区域：

```text
contributor_count=1
weight=1
pose_origin=direct_orb
```

安全背景MultiBand：

```text
contributor_count=2
weights sum=1
dominant owner=最大权重源
两源均为真实direct-ORB render sources
```

不得把双源混合颜色错误描述为单一颜色来源。

### 13.2 稀疏文件结构

`video_pixel_provenance.npz`建议：

```text
schema = gemini305-video-pixel-provenance/v2
owner_frame_id：全图int32
blend_linear_indices：仅blend像素
blend_partner_frame_id：仅blend像素
blend_partner_weight_u16：仅blend像素
blend_pair_id：仅blend像素
source_grid_manifest：按source/pair记录，不逐像素重复
pose_origin_by_source：按source记录
```

对象/风险区不得出现在blend索引中。

### 13.3 Provenance性能门

必须单独测量：

```text
provenance build seconds
NPZ encode seconds
file size
round-trip reconstruction
```

并包含在20秒主计时。若密集双贡献数组导致超时，必须使用上述稀疏结构，不能删除来源语义。

---

## 14. 视觉验收 S1–S5

### S1：几何

```text
held-out FB P95<=1.0 px target
held-out FB P95<=1.5 px hard max
RGB-D residual P95<=1.5 px
seam maximum residual<=2.0 px
automatic line step P95<=1.0 px
automatic line orientation P95<=3 degrees
mesh displacement<=8 px
minimum Jacobian>0
protected intersection=0
```

### S2：对象与Owner

```text
3 compact objects均单owner
full support>=98%
handoff=0
internal seam=0
object blend=0
unowned valid pixels=0
invalid owner IDs=0
```

annotations只在renderer完成后评价。

### S3：光度与显著性

```text
safe-background DeltaE00 P95<3.0
brightness step P95<=1.5%
gradient jump ratio target<=1.15
gradient jump hard max<=1.25
seam_saliency_p95<=development锁定阈值
明显亮度竖带=0
ghost/wide blur=0
```

`seam_saliency`在development标定并写入lock，不根据validation调整。

### S4：结构

```text
strong depth-edge crossing=0
protected-line mismatch crossing=0
owner temporal reversal=0
owner islands/small fragments=0
unowned pixel=0
invalid sampling=0
all final sources real RGB-D=true
all final poses direct ORB=true
all final edges Tensor CUDA pass=true
```

### S5：固定盲评（版本冻结门）

导出每条真实seam 160 px crop、同数量非seam control及对象/长线crop，随机顺序并隐藏算法/provenance。

```text
1:1明确错位/双边/断线/台阶=0
fit-width周期性竖缝=0
seam/control识别率<=60%
明显接缝真实seam=0
```

S5只在算法版本冻结和新现场资格验证时执行，不在普通用户每次运行中重新人工评审。普通delivery只能引用lock已通过的S5资格。

---

## 15. 性能验收 P1–P5

### P1：计时口径

包含online state验证、source/rescue plan、final-edge Open3D、remap、flow/mesh/seam、object、photometric/MultiBand、sparse provenance、编码和delivery原子写入。

### P2：固定会话重复

```text
2 warm-up
10正式运行
10/10<=20.0 s
P50<=16.0 s
P95<=18.5 s
panorama/owner/provenance/source IDs确定性一致
S1–S4每次一致通过
```

### P3：资源

```text
base<=44
rescue<=4
final<=48
resident<=5
R2<=8
object refinement<=12
Open3D candidate records<=冻结上限
GPU peak<=6.5 GB
CPU working set<=4 GB
OOM=0
unexpected CPU fallback=0
```

### P4：独立3 m现场

至少三类：低纹理平面、长直线丰富、紧凑前景丰富。每次通过S1–S5、20秒、真实ORB和Open3D Tensor CUDA。

通过后设置：

```text
production_3m_pass=true
```

### P5：20 m测试授权投影

使用3 m固定会话的P25/P50/P75/P100和实际tile成本建立保守模型，至少包含：

- source/edge数量缩放；
- tile数量和overlap；
- Open3D lattice命中/补算；
- RAFT/R3风险比例；
- provenance和编码；
- 长序列内存稳定性余量。

门：

```text
20 m linear projection<=50 s
20 m conservative projection<=55 s
```

通过后只设置：

```text
twenty_metre_test_authorized=true
```

并允许生成 `user_20m_test.ps1`。不得设置 `twenty_metre_validated=true`。

### P6：真实20 m现场验证

执行授权脚本并验证：

- 完整真实ORB轨迹；
- 全部final edges；
- tile边界质量和provenance；
- 长时间显存/内存稳定性；
- S1–S5；
- 实际主交付性能门。

通过后才设置：

```text
twenty_metre_validated=true
```

20 m实际性能上限在首次授权测试前保持独立验收字段，不通过3 m投影伪造实测结论。

---

## 16. 性能熔断

候选顺序：

```text
合成corridor
→development
→validation
→P25
→P50
→P75
→P100投影
→投影<=20 s才允许完整3 m
```

立即熔断：

- predicted 3 m>20 s；
- final sources>48；
- R2>8；
- rescue>4；
- GPU projected peak>6.5 GB；
- O3D-PRECOMPUTE失败；
- 任一seam无恢复路径；
- 实际Open3D非tensor CUDA。

允许候选期依次降载：

1. 非最高风险pair的额外RAFT iterations。
2. RAFT analysis width 384→320，并重新通过S1。
3. MultiBand 3层→2层。
4. normal advance 18→20→24，仍受seam/rescue门约束。
5. 关闭非主交付审计导出。

不得降载：几何门、长线门、对象单owner、深度保护、provenance、全部final-edge审计、固定输出高度和fail-closed。

---

## 17. 候选序列 SF0–SF5

### SF0：性能数据面

```text
online state v3
direct-ORB source separation
O3D lattice
每源一次remap
hard owner
sparse provenance infrastructure
```

目标：结构完整<=12 s；允许seam明显。

### SF1：L1 + Seam

```text
phase/DIS
automatic line
constrained GraphCut
```

### SF2：Risk-Only RAFT

```text
只处理L1失败corridor
```

### SF3：RGB-D Bounded Mesh

```text
far/mid mesh
near/object owner-only
8 px / 0.80–1.25 / positive Jacobian
```

### SF4：对象 + Rescue

```text
risk-triggered object refinement
single owner
每seam一张direct-ORB rescue
```

### SF5：Photometric + MultiBand

```text
adjacent/skip-one global linear graph
2–8 px safe MultiBand
sparse dual-contributor provenance
```

选择：先淘汰S1–S5失败和任一次>20秒者，再选P95最快、显存最低、模块最少、确定性最高者。

---

## 18. 推荐 Candidate 配置

```yaml
video_panorama_20s:
  algorithm_id: candidate_seam_first_20s_v3

  timing:
    maximum_post_seconds: 20.0
    p50_max_seconds: 16.0
    p95_max_seconds: 18.5
    cli_override_allowed: false

  output:
    fixed_height: 480
    allow_runtime_resolution_change: false

  tracking:
    fps: 8.0
    direct_orb_render_sources_only: true

  online_state:
    schema: gemini305-video-online-state/v3
    require_complete_and_cleanly_closed: true
    require_full_hash_binding: true

  open3d_lattice:
    maximum_candidate_edges: 128
    maximum_pending_jobs: 16
    minimum_final_edge_cache_hit_rate: 0.90
    maximum_post_fill_seconds: 1.8
    require_tensor_cuda: true

  source_selection:
    normal_advance_pixels: 20
    risk_advance_pixels: 10
    maximum_advance_pixels: 24
    base_source_target_range: [36, 44]
    base_source_hard_maximum: 44
    rescue_per_seam_maximum: 1
    rescue_session_maximum: 4
    final_source_hard_maximum: 48

  corridor:
    minimum_width_pixels: 96
    default_width_pixels: 160
    maximum_width_pixels: 160

  alignment:
    level1_backend: phase_translation_dis
    level2_backend: raft_small
    raft_input_width: 384
    raft_fp16: true
    raft_batch_pairs: 6
    maximum_raft_pairs: 8
    require_forward_backward: true

  depth_mesh:
    fit_layers: [far, mid]
    protected_layers: [near, occlusion, disocclusion]
    maximum_displacement_pixels: 8
    minimum_scale: 0.80
    maximum_scale: 1.25
    require_positive_jacobian: true
    taper_to_identity: true

  object_owner:
    maximum_refinement_keyframes: 12
    minimum_full_support: 0.98
    maximum_handoffs: 0
    allow_object_warp: false
    allow_object_multiband: false

  seam:
    backend: constrained_binary_graphcut
    full_resolution_refine_radius_pixels: 4
    forbid_owner_islands: true

  photometric:
    backend: global_linear_rgb_graph
    use_safe_skip_one_overlap: true
    gain_minimum: 0.85
    gain_maximum: 1.18
    bias_minimum_uint8: -12
    bias_maximum_uint8: 12
    held_out_delta_e00_p95_maximum: 3.0

  multiband:
    safe_background_only: true
    maximum_total_width_pixels: 8
    maximum_levels: 3

  provenance:
    schema: gemini305-video-pixel-provenance/v2
    sparse_blend_contributors: true
    maximum_contributors: 2
    blend_weight_dtype: uint16
    require_direct_orb_pose_origin: true

  quality_gate:
    fb_p95_target_pixels: 1.0
    fb_p95_hard_maximum_pixels: 1.5
    rgbd_p95_maximum_pixels: 1.5
    seam_maximum_residual_pixels: 2.0
    line_step_p95_maximum_pixels: 1.0
    line_orientation_p95_maximum_degrees: 3.0
    delta_e00_p95_maximum: 3.0
    brightness_step_p95_maximum_percent: 1.5
    gradient_jump_ratio_target: 1.15
    gradient_jump_ratio_hard_maximum: 1.25
    strong_depth_edge_crossings_maximum: 0

  resources:
    resident_source_window: 5
    maximum_gpu_gb: 6.5
    maximum_cpu_working_set_gb: 4.0

  readiness:
    predicted_20m_linear_maximum_seconds: 50.0
    predicted_20m_conservative_maximum_seconds: 55.0
    require_real_20m_for_validation: true
```

---

## 19. Report与Delivery

### 19.1 Report

```json
{
  "algorithm_id": "candidate_seam_first_20s_v3",
  "production_pose_policy": "direct_orb_only",
  "final_source_count": 0,
  "rescue_source_count": 0,
  "open3d_candidate_edge_count": 0,
  "open3d_edge_cache_hit_rate": 0.0,
  "open3d_post_fill_seconds": 0.0,
  "seam_solver": "constrained_binary_graphcut",
  "provenance_schema": "gemini305-video-pixel-provenance/v2",
  "production_3m_pass": false,
  "twenty_metre_test_authorized": false,
  "twenty_metre_validated": false
}
```

### 19.2 合格Delivery

```json
{
  "delivery_state": "published",
  "manual_review_required": false,
  "primary_post_capture_seconds": 0.0,
  "maximum_post_seconds": 20.0,
  "within_post_capture_budget": true,
  "seam_geometry_pass": true,
  "object_integrity_pass": true,
  "seam_photometric_pass": true,
  "production_3m_pass": true,
  "actual_open3d_backend": "open3d_tensor_cuda_rgbd",
  "direct_orb_render_pose_count": 0,
  "interpolated_render_pose_count": 0,
  "production_lock_qualification": {
    "blind_seam_evaluation_pass": true
  }
}
```

### 19.3 退化Delivery

结构完整但质量或性能失败：

```json
{
  "delivery_state": "published_degraded",
  "manual_review_required": true,
  "within_post_capture_budget": false,
  "production_3m_pass": false
}
```

具体pass字段必须反映实际结果，不能在失败时写true。结构失败不发布 `video_delivery.json`。

---

## 20. 代码边界

优先修改：

```text
src/panorama_demo/video_panorama.py
src/panorama_demo/video_pipeline.py
src/panorama_demo/video_online_state.py
src/panorama_demo/video_source_selection.py
src/panorama_demo/video_motion_resampler.py
src/panorama_demo/video_visual_renderer_v2.py
src/panorama_demo/video_budget_controller.py
src/panorama_demo/video_delivery.py
src/panorama_demo/video_visual_metrics.py
configs/demo.yaml
README.md
AGENTS.md
```

建议新增：

```text
src/panorama_demo/video_online_seam_evidence.py
src/panorama_demo/video_open3d_edge_lattice.py
src/panorama_demo/video_pair_risk.py
src/panorama_demo/video_seam_corridor.py
src/panorama_demo/video_seam_alignment.py
src/panorama_demo/video_seam_raft.py
src/panorama_demo/video_seam_depth_layers.py
src/panorama_demo/video_seam_mesh.py
src/panorama_demo/video_seam_solver_graphcut.py
src/panorama_demo/video_object_tracks.py
src/panorama_demo/video_seam_rescue.py
src/panorama_demo/video_photometric_graph.py
src/panorama_demo/video_torch_multiband.py
src/panorama_demo/video_pixel_provenance_v2.py
src/panorama_demo/video_seam_quality.py
src/panorama_demo/video_seam_blind_eval.py
src/panorama_demo/video_20m_projection.py
```

实施前以当前仓库真实模块为准建立映射，禁止重复创建同功能模块。

---

## 21. Commit顺序

1. 标记旧方案SUPERSEDED，冻结D3失败证据和真实baseline身份。
2. 增加S1–S5、P1–P6、性能熔断和正确delivery状态。
3. 分离evidence、ORB anchor、base和rescue sources。
4. 增加online state v3、DIS去重和有界Open3D lattice。
5. 增加one-remap resident数据面和Tensor CUDA final-edge审计。
6. 增加L1和constrained GraphCut框架。
7. 增加persistent batched RAFT-small。
8. 增加bounded RGB-D far/mid mesh。
9. 增加compact-object single owner和direct-ORB rescue。
10. 增加global photometric、narrow MultiBand和sparse provenance v2。
11. 运行SF0–SF5消融、固定会话重复和独立3 m验证。
12. 冻结3 m Production，执行20 m授权投影。
13. 用户授权后执行真实20 m，并单独设置validated状态。

每个commit独立通过相关测试，不能一次性混合全部生命周期和renderer改造。

---

## 22. 测试要求

新增至少：

```text
test_video_online_state_v3.py
test_video_open3d_edge_lattice.py
test_video_open3d_precompute_gate.py
test_video_render_source_decoupling.py
test_video_direct_orb_sources.py
test_video_seam_corridor.py
test_video_seam_alignment.py
test_video_seam_raft.py
test_video_seam_mesh.py
test_video_object_tracks.py
test_video_direct_orb_rescue.py
test_video_seam_solver_graphcut.py
test_video_photometric_graph.py
test_video_multiband_safety.py
test_video_pixel_provenance_v2.py
test_video_seam_quality.py
test_video_seam_blind_eval.py
test_video_20s_budget.py
test_video_20m_projection.py
test_video_20m_readiness_states.py
```

必须验证：

1. evidence frame不能未经授权成为owner。
2. final sources全部direct ORB，tracking FPS固定8.0。
3. rescue direct ORB、per seam<=1、session<=4。
4. online state任一哈希变化整份拒绝。
5. lattice只复用完全匹配边，队列有界且采集不掉帧。
6. 每source只生成一次grid、只上传一次。
7. 全部final edges发布前Tensor CUDA审计。
8. RAFT只在R2 corridor且pair<=8。
9. mesh<=8 px、scale 0.80–1.25、正Jacobian、边界identity。
10. object/深度保护与MultiBand零交集。
11. object单owner、blend=0。
12. GraphCut满足相同cost、硬禁止和owner拓扑。
13. owner单调、无岛、无未归属。
14. sparse contributor/weight可重建blend颜色语义。
15. summary/minimal与full/audit panorama和provenance一致。
16. 20秒包含provenance、编码和delivery原子写入。
17. 预测>20秒禁止完整3 m。
18. public CLI不能覆盖lock内20秒和renderer参数。
19. fixed annotations永不进入renderer。
20. 同一Production lock不改变输出分辨率。
21. degraded结果不能写 `delivery_state=published`。
22. 20 m投影只能授权测试，不能设置validated。

合成场景：3 px横梁台阶、2 px垂直偏差、亮度阶跃、双层视差、深度边缘、细线缆、可被一个rescue修复的pair和必须fail-closed的pair。

常规验证：

```powershell
$G305Python = 'D:\Panoramic_Camera\.conda\python.exe'
& $G305Python -m pytest -q
ruff check src tests
& $G305Python -m compileall -q src tests
git diff --check
```

---

## 23. Production Freeze与20 m Handoff

### Go-3M

```text
SF0–SF5完成
S1–S5通过
P1–P4通过
10/10<=20 s
P50<=16 s
P95<=18.5 s
三类独立3 m通过
全量测试通过
```

允许生成：

```text
configs/video_production/production_seam_first_20s_v1.yaml
production.lock.json
production_3m_pass=true
```

### Go-20M-Test

P5投影通过后：

```text
twenty_metre_test_authorized=true
生成user_20m_test.ps1
```

此时 `twenty_metre_validated=false`。

### Go-20M-Validated

真实20 m现场完成并通过P6后：

```text
twenty_metre_validated=true
```

3 m Production Freeze不依赖20 m实测；20 m Production资格不能只靠3 m投影。

### No-Go

任一情况停止对应冻结：

- 明显seam错位、双边、断线或台阶；
- 任一compact object非单owner；
- 宽模糊掩盖seam；
- 任一次正式3 m>20秒；
- final sources>48、R2>8或rescue>4；
- warp>8 px或scale超出0.80–1.25；
- Open3D回退CPU；
- fixed annotation进入renderer；
- online state不完整仍复用；
- 插值pose成为正式owner；
- provenance不能解释blend；
- 20 m投影未过却生成脚本；
- 未完成真实20 m却设置validated。

---

## 24. 明确排除项

```text
不采用399-source/509-second D3作为Production
不允许Production使用interpolated/refined render pose
不关闭正式全边Open3D，不发布后补正式审计
不允许48 px形变或0.70–1.40生产scale
不允许12–20 px MultiBand或宽feather
不采用DP/monotone_shortest_path backend
不自动降低480 px输出
不把36设为所有会话硬下限
不向普通用户暴露maximum-post-seconds覆盖
不改名真实baseline algorithm_id
不在失败时写delivery_state=published
不把3 m投影当作20 m实际验证
```

---

## 25. 最终授权摘要

```text
停止399-source、509-second D3和全画布RAFT/Open3D/object compositor。

建立candidate_seam_first_20s_v3：

1. tracking固定8.0 FPS，所有base/rescue render source均为direct ORB；
2. base目标36–44但只设硬上限44，rescue per-seam<=1/session<=4，final<=48；
3. corridor 96–160 px，warp<=8 px，scale 0.80–1.25，positive Jacobian；
4. 采集期建立有界Open3D candidate-edge lattice，最终边发布前全部Tensor CUDA审计；
5. L1 phase/DIS，L2 risk-only batched RAFT，L3 RGB-D far/mid mesh，L4 direct-ORB rescue；
6. compact object单owner、零handoff、零blend；
7. 采用constrained GraphCut，不引入被禁止的DP backend；
8. 只有安全背景允许2–8 px、最多3层MultiBand；
9. sparse provenance v2记录dominant owner和真实blend contributor；
10. 3 m每次<=20 s、P50<=16 s、P95<=18.5 s；
11. S5只作版本冻结资格，普通运行引用lock结果；
12. P5投影只授权20 m测试，真实20 m通过后才validated；
13. 普通用户只能读取Production lock，不能覆盖内部算法参数；
14. 任一门失败继续fail-closed或按现有政策发布C级degraded，不能更新Production lock。
```

最终原则：

> 用60 FPS在线证据、8 FPS direct-ORB anchors、有界Open3D预计算和性能熔断控制时间；用96–160 px走廊内逐级对齐、有限direct-ORB救援、对象单owner、受限GraphCut和窄带安全融合实现近无痕接缝；用稀疏双贡献provenance诚实解释融合颜色；用3 m冻结、20 m测试授权和真实20 m验证三个状态防止把投影当成现场结论。
