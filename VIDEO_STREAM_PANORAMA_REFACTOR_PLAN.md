# Gemini 305 视频流全景正式交付改造方案

> 状态：设计冻结，尚未开始实现  
> 适用项目：`D:\central_strip_Panoramic_Camera`  
> 目标设备：Orbbec Gemini 305  
> 方案依据：截至 2026-07-31 已确认的需求、现有源码边界和两段真实连续视频会话的只读检查结果

## 1. 目标

在不改变现有照片模式正式程序默认行为的前提下，为连续视频 RGB-D 会话新增独立正式全景产品。

视频模式必须：

- 默认直接兼容当前自动曝光、自动增益、自动白平衡的连续采集；
- 可选固定彩色曝光，但默认仍为自动曝光；
- 使用 ORB-SLAM3 RGB-D 提供真实 `camera_to_world` 全局轨迹；
- 使用 Open3D RGB-D 边验证实际全景源之间的局部几何；
- 与照片模式使用同一个 unified calibrated central-strip renderer 内核；
- 允许视频 profile 启用经过严格审计的 affine、homography、局部 mesh/flow 残差候选；
- 生成全分辨率二维全景、像素来源数据和二维缩放浏览器；
- 二维全景完成后即可独立正式发布；
- 随后自动或延后生成与照片模式格式和内容规则一致的 TSDF/GLB/三维 Viewer；
- 三维失败不得撤销已经成功发布的二维全景；
- 支持任意总扫描长度，不设置总源数和总画布 `200 MP` 硬上限；
- 对超长扫描使用 BigTIFF、Deep Zoom、分块缓存、断点续跑和空间分块 TSDF；
- 普通用户不调整算法参数。

视频模式当前正式验收速度上限为：

```text
1.0 m/s
```

场景约束与照片模式一致：

- 相机连续单向水平侧移；
- 相机正立，RGB 未镜像；
- 图像坐标向右对应场景物理右方向；
- 场景基本静止；
- 最近物体约 `0.5 m`；
- 不承诺持续人员运动或物体运动的动态场景。

## 2. 核心架构原则

采用：

```text
同一个算法内核
+ 两套独立业务流水线
+ 两套独立发布协议
```

### 2.1 照片流程保持不变

```text
照片 RGB-D 会话
  → 照片输入门控
  → 照片主扫描段与完整 pose nodes
  → 完整 ORB-SLAM3 RGB-D
  → 完整 Open3D RGB-D 边链审计
  → unified_calibrated_central_strip/v1
  → 照片 A/B/C/F
  → TSDF / desktop GLB / mobile GLB / Viewer
  → delivery.json
```

照片模式下列内容必须保持完全兼容：

- 全景像素；
- owner map；
- 默认配置；
- A/B/C/F；
- `gemini305-unified-central-strip/v12-r1`；
- `gemini305-panorama-delivery/v12-r1`；
- 正式交付文件；
- TSDF/GLB/Viewer；
- 所有现有测试。

任何默认照片输出变化均视为回归。

### 2.2 视频流程独立编排

```text
连续视频 RGB-D 会话
  → 视频输入门控
  → 视频预分析与最长单向扫描段
  → 完整扫描段 ORB-SLAM3 RGB-D
  → 视频专用真实源选择
  → 选中源 Open3D RGB-D 边审计
  → 共享 unified central-strip renderer
  → 视频二维 A/B/C/F
  → BigTIFF / JPEG / PNG / provenance / 2-D Viewer
  → video_delivery.json
  → 自动或延后执行只读 TSDF
  → video_3d_delivery.json 或 video_3d_failure.json
```

### 2.3 强制依赖边界

允许视频流程直接复用：

- `session.py`
- `quality.py`
- `orbslam3_bridge.py`
- `rgbd_odometry.py`
- `calibrated_rgb_pushbroom.py`
- `dense_fusion.py` 中只读 TSDF/GLB 导出函数

禁止：

```text
video_panorama.py → stitch_sequence._run_pipeline()
video_panorama.py → 照片正式发布器
stitch_sequence.py → video_*.py
```

视频三维阶段可以调用 `dense_fusion.export_tsdf_mesh_pair()`，但二维视频编排和 renderer 不得导入或调用 TSDF。

## 3. CLI 设计

新增三个正式入口：

```text
g305-video-panorama
g305-video-3d
g305-video-viewer
```

### 3.1 自动曝光视频采集

默认连续视频采集行为保持不变：

```powershell
& 'D:\Panoramic_Camera\.conda\Scripts\g305-capture.exe' `
  --max-frames 120 `
  --output 'D:\central_strip_Panoramic_Camera\data\captures'
```

默认：

- 自动曝光；
- 自动增益；
- 自动白平衡；
- `capture_mode="continuous_rgbd_video_auto"`。

### 3.2 固定曝光视频采集

新增：

```powershell
& 'D:\Panoramic_Camera\.conda\Scripts\g305-capture.exe' `
  --video-exposure-us 800 `
  --max-frames 120 `
  --output 'D:\central_strip_Panoramic_Camera\data\captures'
```

规则：

- `--video-exposure-us` 只适用于连续视频；
- 与 `--photo-mode` 同时使用时直接报错；
- 不传时仍为默认自动模式；
- 固定曝光时自动曝光关闭；
- 增益和白平衡继续自动；
- 输入单位为微秒；
- 非 `100 µs` 整数倍时自动量化到最近设备值；
- 启动前显示请求值和实际量化值；
- 属性写入后必须回读；
- 每帧 exposure metadata 必须与实际量化值一致；
- 连续三帧不一致时采集失败；
- 超过设备当前 FPS 支持范围时在启动前拒绝；
- 超过 `1200 µs` 但设备支持时允许采集，不过视频二维正式交付最高只能为 C。

固定曝光会话标记：

```text
capture_mode="continuous_rgbd_video_fixed_exposure"
```

### 3.3 视频二维正式处理

```powershell
& 'D:\Panoramic_Camera\.conda\Scripts\g305-video-panorama.exe' `
  'D:\central_strip_Panoramic_Camera\data\captures\run_YYYYMMDD_HHMMSS' `
  --output 'D:\central_strip_Panoramic_Camera\outputs\video_sequence'
```

默认在二维发布后继续执行三维。

延后三维：

```powershell
& 'D:\Panoramic_Camera\.conda\Scripts\g305-video-panorama.exe' `
  '会话目录' `
  --output '输出目录' `
  --defer-3d
```

### 3.4 独立三维处理

```powershell
& 'D:\Panoramic_Camera\.conda\Scripts\g305-video-3d.exe' `
  '视频二维交付目录'
```

若原始会话被移动：

```powershell
& 'D:\Panoramic_Camera\.conda\Scripts\g305-video-3d.exe' `
  '视频二维交付目录' `
  --input '移动后的会话目录'
```

新位置的全部实际源 RGB、aligned depth、calibration 和 manifest SHA-256 必须与二维报告一致。

### 3.5 二维浏览器

```powershell
& 'D:\Panoramic_Camera\.conda\Scripts\g305-video-viewer.exe' `
  '视频二维交付目录'
```

默认：

- 仅绑定 `127.0.0.1`；
- 自动选择可用端口；
- 启动只读 HTTP 服务；
- 打开二维浏览器；
- 不向局域网开放；
- 高级用户可以显式指定监听地址。

仍支持手动：

```powershell
cd '视频二维交付目录'
& 'D:\Panoramic_Camera\.conda\python.exe' -m http.server 8080
```

## 4. 会话 schema 与输入资格

### 4.1 新视频采集

只将连续视频会话升级为：

```text
panorama-demo-session/v2
```

照片会话保持当前 schema 和行为。

视频 v2 manifest 保留兼容字段：

```json
{
  "diagnostic_only": true,
  "formal_stitch_allowed": false
}
```

这些字段继续表示“不能进入照片正式流程”。

新增：

```json
{
  "product_eligibility": {
    "photo_panorama": false,
    "video_panorama": true
  }
}
```

`video_panorama=true` 只在以下条件全部满足后写入：

- 相机和写盘资源已经安全关闭；
- `clean_shutdown=true`；
- 无采集异常；
- `write_errors=0`；
- `writer_errors=[]`。

`queue_drops` 不直接使资格为 false，后续按时间戳跨度和轨迹连续性审计。

### 4.2 旧视频会话兼容

必须接受已存在的：

```text
panorama-demo-session/v1
+ capture_mode="continuous_rgbd_video_auto"
```

旧会话的视频产品资格根据以下字段推导：

- `clean_shutdown`
- `received_frames`
- `written_frames`
- `queue_drops`
- `write_errors`
- `writer_errors`
- `timestamp_regressions`

### 4.3 结构验证

视频正式输入仍须通过现有严格 RGB-D 结构验证：

- manifest；
- calibration；
- `frames.csv`；
- RGB 文件；
- `depth_aligned/` 内同尺寸 `uint16 PNG`；
- 正数 `depth_scale_mm_per_unit`；
- depth-to-color 对齐 provenance；
- 非负时间戳；
- 正数曝光 metadata；
- 有限标定；
- 毫米单位。

缺曝光 metadata 为结构失败 F，不能猜测。

### 4.4 模式互斥

`g305-video-panorama` 只接受：

```text
continuous_rgbd_video_auto
continuous_rgbd_video_fixed_exposure
```

照片会话传给视频 CLI 时拒绝。

`g305-panorama` 继续拒绝视频会话。

## 5. 视频主扫描段

先在分析宽度约 `320 px` 的 RGB 缩略图上复用：

- `resize_for_analysis()`
- `analyze_frame_quality()`
- `estimate_translation()`
- `select_primary_scan_segment()`

预分析只用于：

- 去除开始和结束静止段；
- 去除短暂停顿；
- 找到最长连续单向扫描段；
- 识别回程；
- 记录模糊、曝光和纹理；
- 为 ORB 输入确定候选范围。

二维预分析不得：

- 生成正式 pose；
- 替代 ORB-SLAM3；
- 插值 pose；
- 跨两个不连续段拼图。

若视频存在回程，允许选择最长连续单向扫描段。报告必须列出：

- 保留帧；
- 排除帧；
- 排除原因；
- 起止时间；
- 扫描方向。

## 6. ORB-SLAM3 全局轨迹

### 6.1 唯一全局轨迹

ORB-SLAM3 RGB-D 是视频全景唯一全局轨迹来源。

允许 affine、homography、mesh 和 flow 仅作为真实 ORB pose 上的图像残差，不得：

- 替代缺失 pose；
- 构造第二条全局轨迹；
- 修改 ORB pose；
- 使用时间或二维 pose 插值。

### 6.2 完整扫描段

ORB-SLAM3 必须处理主扫描段内全部视频帧，而不是只处理最终 renderer 源。

要求：

```text
主扫描段整体 tracked fraction >= 95%
实际 renderer 源真实 pose 覆盖率 = 100%
```

相邻真实 tracked pose 时间间隔不得超过：

```text
250 ms
```

超过即视为轨迹断点。

若中途丢失跟踪：

- 不能跨断点拼接；
- 可以选择最长连续、连通、单向 ORB 轨迹段；
- 报告必须说明未覆盖的时间范围；
- 若保留段不满足正式跨度和质量要求则 F。

### 6.3 动态超时

ORB 超时按输入长度动态计算：

```text
max(600 秒, 视频时长 × 10)
```

实际计算值写入报告。

### 6.4 极长视频分段 ORB

若单次 ORB-SLAM3 超过资源能力，允许使用多个重叠子序列。

每段：

- 都是完整 ORB-SLAM3 RGB-D 运行；
- 重叠区至少 `2秒`；
- 至少 `20` 个共同真实 tracked 帧；
- 使用共同帧的真实 ORB pose 求稳健 SE(3) 段间对齐；
- RGB-D 保持公制尺度；
- 段间对齐通过有限刚体、位移、旋转和残差审计；
- 无足够共同帧时不得合并。

合并结果仍必须是：

- 有限 SE(3)；
- 连通；
- 连续；
- 单向；
- 毫米单位；
- 每个实际源有真实 ORB pose。

## 7. 视频真实源选择

视频总源数不设固定上限，但不能把所有近重复帧直接送入 renderer。

源选择基于：

- ORB 相机中心真实毫米位移；
- RGB 局部水平位移；
- 图像清晰度；
- 曝光和饱和；
- 纹理覆盖；
- 时间间隔；
- 相邻重叠；
- Open3D 可验证跨度。

相机中心：

```python
camera_center_i = camera_to_world_i[:3, 3]
```

规则：

- 第一和最后有效源保留；
- 使用与照片相同的目标布局尺度：

```yaml
target_displacement_fraction: 0.18
maximum_displacement_fraction: 0.28
```

- 在候选窗口内优先选择更清晰的真实 tracked 帧；
- 不允许虚拟帧；
- 不允许 pose 插值；
- 不允许重排真实空间顺序；
- 若源跨度过大，插入中间真实 tracked 帧；
- 插入后重新进行 Open3D 边审计。

取消的是：

- 总源数 160 上限；
- 总画布 200 MP 上限。

仍保留：

- 每次内存中只驻留相邻 `2–5` 个源；
- 默认总工作集约 `4 GB`；
- 单个 tile 超限时自动减小 tile；
- 画布和源总数使用 64 位索引与尺寸计算。

## 8. Open3D RGB-D 局部边

ORB-SLAM3 提供全局 pose，Open3D 测量实际 renderer 相邻源的局部 RGB-D 边。

ORB 初值：

```python
initial_source_to_reference = (
    np.linalg.inv(reference_camera_to_world)
    @ source_camera_to_world
)
```

复用：

```python
estimate_pair_rgbd_odometry(
    reference,
    source,
    calibration,
    initial_source_to_reference=initial_source_to_reference,
)
```

如果选中源跨度过大导致 Open3D 失败：

1. 在中间寻找一个真实 tracked 视频帧；
2. 插入该帧；
3. 测量两条较短真实边；
4. 禁止合成边和插值 pose。

然后复用：

```python
optimize_rgbd_pose_graph(
    selected_frames,
    edges,
    backend=ORBSLAM3PoseGraphOptimizer(trajectory),
    enforce_edge_quality=False,
)
```

并复用 `validate_pose_trajectory()` 做：

- 连通性；
- 单向性；
- 步长；
- 垂直和前后漂移；
- 旋转；
- Open3D/ORB 边残差；
- 连续不可靠边；
- 有限刚体。

CUDA 规则与现有项目一致：

- `G305_CUDA=required` 时必须实际使用 `open3d_tensor_cuda_rgbd`；
- 不允许静默 CPU 回退；
- `prefer|auto|off|required` 保持可审计；
- OpenCV 图像残差第一版可以使用 CPU。

## 9. 共享 renderer 与改造范围

### 9.1 同一个 renderer 内核

照片和视频都调用：

```text
render_calibrated_rgb_pushbroom()
```

共享：

- 单一目标坐标域；
- 单一画布；
- 单一 valid mask；
- 唯一 owner；
- 原始 RGB；
- 标定 inverse sampling；
- 条带布局；
- RGB 风险；
- depth 可见性；
- GraphCut；
- hard owner；
- 窄带 MultiBand；
- 资源和拓扑审计。

视频不复制另一套 renderer。

### 9.2 照片 profile 锁定

新增能力全部为可选通用能力，照片 profile 默认关闭。

照片调用在默认配置下必须：

- panorama 数组逐像素不变；
- owner map 逐像素不变；
- metadata 不变；
- schema 不变；
- A/B/C/F 不变。

### 9.3 视频 profile 扩展

视频可以启用以下候选：

```text
无残差
  → similarity / affine
  → 受限 homography
  → 局部 mesh / flow
```

模型按从简单到复杂逐级尝试。

复杂模型只有同时通过以下条件才能使用：

- 独立 held-out 对应；
- 前后向 flow 一致性；
- 正 Jacobian；
- 边界和连续性；
- 最大全分辨率位移 `16 px`；
- 前景保护；
- 遮挡保护；
- depth 不连续保护；
- 实际 RGB 直线和结构审计；
- 应用后误差优于更简单模型。

失败时依次回退：

```text
局部 mesh/flow
  → homography
  → affine/similarity
  → 无残差
  → hard owner
```

### 9.4 ORB 与二维残差边界

affine、homography、mesh、flow：

- 只能作用于相邻 pair；
- 不能跨多个 pair 累计为第二条全局运动；
- 不能填补缺失 ORB pose；
- 不能修改 `camera_to_world`；
- 最大位移 `16 px`；
- 超限候选不应用；
- 不得把一个局部模型无审计地扩展到全图。

### 9.5 前景和遮挡

以下区域必须保持单一真实 owner，不做跨帧 RGB 变形：

- 前景物体；
- 遮挡边界；
- disocclusion；
- depth 不连续；
- depth 孔洞；
- 透明或反光保护域；
- 强结构保护域；
- flow 前后向不一致区域。

二维残差只允许应用于：

- 双向可见；
- 同层；
- 未保护；
- held-out 审计通过；
- RGB 与几何一致的区域。

### 9.6 一源一次 RGB 采样

即使启用视频残差，也必须把：

- 标定去畸变；
- ORB 旋转校正；
- affine/homography；
- 局部 mesh/flow；

合成为一次最终 inverse sampling。

禁止连续多次重采样 RGB。

全景颜色只能来自原始 RGB。三维数据不得补全 RGB。

## 10. 视频布局与真实方向

稳健扫描轴由 ORB 相机中心轨迹估计。

布局标量通过：

```text
RGB 局部位移 / ORB 相机中心毫米位移
```

稳健求得 `pixels_per_mm`。

视频小基线较多，允许使用真实 pose pair gap：

```text
1, 2, 4
```

用于布局证据，但必须：

- 忽略近零基线；
- 方向一致；
- 使用真实 ORB pose；
- 保留 held-out pair；
- 验证统一比例；
- 最终仍只产生一个布局标量；
- 不修改 pose。

输出必须保持场景真实空间朝向为左到右：

- 图像未镜像；
- 第一帧相机坐标 `+X` 定义物理右方向；
- 正向扫描按真实空间排列；
- 反向扫描时 render 顺序可以反转；
- `video_render_transforms.json` 同时记录时间顺序和空间 render 顺序；
- 不能简单规定“第一帧永远放左边”。

## 11. 自动曝光、自动增益和自动白平衡

### 11.1 metadata 先验

允许视频 profile 使用每帧：

- exposure；
- gain；
- white balance metadata；

作为颜色关系的初始先验。

最终颜色校正仍必须：

- 从安全背景重叠区重新估计；
- 进行 held-out 验证；
- 使用有界全局线性 RGB 增益；
- 排除前景、遮挡、反光、强结构和 depth 边界；
- 验证失败时回到原始 RGB 和 hard owner。

禁止：

- metadata 直接决定最终 RGB；
- 逐像素无约束亮度补偿；
- 全图色调拉平；
- depth 或 TSDF 提供颜色。

通过安全验证的颜色校正本身不降低视频等级。

### 11.2 曝光等级规则

只要任一实际 renderer 源：

```text
color_exposure_us > 1200
```

则二维全景最高只能为：

```text
C
delivery_state=published_degraded
manual_review_required=true
```

自动曝光和固定曝光采用同一规则。

缺失有效曝光 metadata 为 F。

## 12. 超长二维全景

### 12.1 双执行后端

自动选择：

```text
预计画布 <= 200 MP
  → 现有内存执行后端

预计画布 > 200 MP
  → 分块流式执行后端
```

对同一小型输入，两种后端必须生成：

- 无损 RGB 逐像素一致；
- Alpha/valid mask 逐像素一致；
- owner map 逐像素一致；
- source coordinate provenance 一致；
- seam 和 quality metadata 一致。

`200 MP` 不再是失败限制，只是后端切换点。

### 12.2 分块策略

流式后端使用：

- 64 位画布坐标；
- 空间 tile；
- seam/MultiBand halo；
- 相邻 `2–5` 源驻留窗口；
- 全局布局预计算；
- 全局 photometric pair statistics 预计算；
- 顺序 owner 状态；
- 分块 RGB/Alpha/provenance 同步写入；
- tile 边界一致性审计；
- 中断检查点。

不能因为 tile 边界改变：

- seam；
- owner；
- blend；
- RGB；
- provenance。

### 12.3 主图

正式主文件：

```text
panorama.tif
```

要求：

- BigTIFF；
- 全分辨率；
- 原始采集垂直分辨率；
- 统一全分辨率横向尺度；
- RGBA；
- 8-bit sRGB；
- Alpha 精确保存 valid mask；
- tile；
- 多级金字塔；
- Deflate 或 LZW 无损压缩；
- 不自动降采样。

允许新增正式依赖：

```text
tifffile
imagecodecs（可选加速）
```

### 12.4 JPEG 和 PNG

预览：

```text
panorama.jpg
panorama.png
```

规则：

- 若全景 `<= 200 MP` 且单边尺寸未超过编码器上限，直接输出全分辨率；
- 若超过 `200 MP`，缩放到不超过 `200 MP`；
- 即使像素数不足 `200 MP`，若单边尺寸超过编码器限制，也按尺寸上限缩小；
- 报告记录缩放原因和比例；
- PNG 保留 Alpha valid mask；
- JPEG 将无效区与黑色背景合成；
- BigTIFF 始终保持全分辨率。

## 13. 全分辨率 provenance

每个全分辨率有效像素保存：

```text
frame_id
source_u
source_v
```

要求：

- `frame_id` 为真实采集帧 ID；
- `source_u/source_v` 指向磁盘中原始未去畸变 JPEG；
- 坐标精度为 `1/16 px`；
- 无损压缩；
- 分块 BigTIFF 或等价可随机读取的分块格式；
- 无效像素使用明确 sentinel；
- 与 RGBA Alpha 和 owner 拓扑一致。

建议文件：

```text
pixel_owner.tif
pixel_source_uv.tif
pixel_provenance_preview.npz
```

`pixel_provenance_preview.npz` 只服务缩放预览，不能取代全分辨率 provenance。

## 14. 二维浏览器

正式二维浏览器：

```text
panorama_viewer.html
panorama.dzi
panorama_files/
provenance_tiles/
source_thumbnails/
```

要求：

- Deep Zoom；
- 完全离线；
- OpenSeadragon 脚本随交付保存；
- 不依赖 CDN；
- 通过本地 HTTP 服务访问；
- 支持缩放和平移；
- 显示全景像素坐标；
- 点击像素显示：
  - `frame_id`
  - 时间戳
  - exposure
  - gain
  - white balance
  - 原始 `source_u/source_v`
  - 对应原始 RGB 源缩略图
- 缩略图约 `320 px` 宽；
- 只保存实际 renderer 源缩略图；
- 不依赖原始 `data` 路径即可完成二维溯源；
- 检测到 `video_3d_delivery.json` 后显示“打开三维模型”按钮；
- 三维 pending 或 failed 时隐藏三维按钮。

## 15. 视频二维 A/B/C/F

视频使用 A/B/C/F，但 schema 与照片独立。

### A

- 会话结构通过；
- ORB 和 Open3D 轨迹结构通过；
- owner/valid/provenance 通过；
- 图像严格质量通过；
- 无几何残差，或只使用安全 hard owner；
- 安全颜色校正允许；
- `delivery_state=published`。

### B

- 所有结构检查通过；
- 严格图像质量通过；
- 实际采用了通过完整审计的：
  - affine；
  - homography；
  - mesh；
  - flow；
- `delivery_state=published`。

### C

- 会话、真实 pose、owner、provenance、画布和原子发布结构完整；
- 但严格图像质量未通过，或：
  - 使用 `hard_cut_degraded`；
  - 任一实际源曝光超过 `1200 µs`；
  - 其它明确降级条件；
- `delivery_state=published_degraded`；
- `manual_review_required=true`。

### F

- 会话结构失败；
- 缺曝光 metadata；
- 缺实际源真实 ORB pose；
- SE(3) 非有限或非刚体；
- 轨迹断裂且无合格连续段；
- Open3D 必需边结构失败；
- owner/Alpha/provenance 不一致；
- 分块边界不一致；
- 磁盘或文件系统预检失败；
- BigTIFF/Deep Zoom/二维 Viewer 失败；
- 二维原子发布失败；
- 不发布 `video_delivery.json`。

三维状态不改变二维 A/B/C/F。

## 16. 二维正式发布

### 16.1 独立 schema

```text
video_report.json
  schema=gemini305-video-panorama-report/v1

video_delivery.json
  schema=gemini305-video-panorama-delivery/v1
```

报告引用：

```text
renderer_backend=unified-calibrated-central-strip/v1
renderer_profile=continuous-rgbd-video/v1
same_rgb_algorithm_as_photo=true
```

### 16.2 二维交付文件

建议完整集合：

```text
panorama.tif
panorama.jpg
panorama.png
pixel_owner.tif
pixel_source_uv.tif
pixel_provenance_preview.npz
video_render_transforms.json
video_report.json
panorama_viewer.html
panorama.dzi
panorama_files/
provenance_tiles/
source_thumbnails/
video_delivery.json
```

`video_render_transforms.json` 可以保存：

- 真实 ORB `camera_to_world`；
- Open3D 边；
- 时间顺序；
- 空间 render 顺序；
- 布局标量；
- source selection；
- residual model；
- seam；
- provenance mapping。

这些是审计元数据，不是三维模型。

### 16.3 原子规则

二维任务的第一项输出动作：

```text
使目标任务目录中的旧 video_delivery.json 失效
```

所有二维文件先写 pending/cache，完成验证后发布。

```text
video_delivery.json
```

必须最后原子写入，发布后不可修改。

二维进度和三维状态不得通过修改二维成功标记表达。

## 17. 输出目录路由

若用户指定目录没有照片 `delivery.json`：

```text
直接将视频产物写入该目录
```

若用户指定目录已有有效照片 `delivery.json`：

```text
创建 <输出目录>/video_panorama/
```

若 `video_panorama/` 已有有效 `video_delivery.json`：

```text
创建 <输出目录>/video_panorama_YYYYMMDD_HHMMSS/
```

若用户指定目录本身已有有效 `video_delivery.json`：

```text
创建同级 video_panorama_YYYYMMDD_HHMMSS/
```

不得覆盖历史照片或视频正式交付。

## 18. 独立三维交付

### 18.1 发布解耦

二维成功后立即写：

```text
video_delivery.json
```

三维使用独立状态：

```text
video_3d_status.json
video_3d_delivery.json
video_3d_failure.json
```

二维 `video_delivery.json` 发布后保持不可修改。

三维失败：

- 写 `video_3d_failure.json`；
- 不删除二维文件；
- 不删除 `video_delivery.json`；
- 不降低二维 A/B/C/F；
- `g305-video-panorama` 返回退出码 `0`，但打印明确警告；
- 自动化通过 `video_3d_delivery.json` 判断三维是否成功。

### 18.2 三维格式

与照片模式保持一致：

```text
tsdf_mesh.glb
tsdf_mesh_mobile.glb
tsdf_mesh_viewer.html
```

保持相同：

- RGB-D TSDF；
- voxel；
- truncation；
- depth range；
- depth consistency；
- `rgbd_tsdf_vertex_colour`；
- desktop/mobile triangle target；
- texture/vertex colour规则；
- `+Y-down` 到 Viewer `+Y-up`；
- 上方孤岛排除语义；
- GLB validator；
- Viewer 协议。

### 18.3 三维数据来源

TSDF 只使用：

- 实际参与二维全景的 renderer 源帧；
- 原始 RGB；
- 原始 aligned depth；
- 真实 ORB pose；
- 已通过审计的源顺序。

二维 affine、homography、mesh、flow：

- 不得回传 TSDF；
- 不得修改 depth；
- 不得修改 pose；
- 不得改变三维颜色；
- 不得改变三维裁剪。

### 18.4 两种 TSDF 后端

现有容量内：

```text
直接调用当前 export_tsdf_mesh_pair()
```

保证照片格式和内容规则一致。

超长扫描超过当前 TSDF 容量时：

- 空间分块 TSDF；
- 分块重叠；
- 相同体素和 truncation；
- 相同可见性与深度一致性；
- 边界 mesh 合并审计；
- 相同颜色规则；
- 相同坐标翻转；
- 相同上方孤岛排除；
- 最终仍输出单个 desktop GLB、单个 mobile GLB 和同格式 Viewer。

## 19. 缓存、续跑和状态

### 19.1 缓存边界

只清理程序处理中产生的缓存。

绝不删除：

- `data` 下原始会话；
- RGB；
- aligned depth；
- manifest；
- calibration；
- `frames.csv`。

缓存放在输出目录之外或输出父目录的专用位置，例如：

```text
<输出父目录>/.g305_video_cache/<task-id>/
```

### 19.2 续跑一致性

复用缓存前严格校验：

- 输入会话文件清单；
- 实际源 RGB SHA-256；
- 实际源 aligned depth SHA-256；
- calibration SHA-256；
- manifest SHA-256；
- 配置哈希；
- 程序版本；
- Git commit；
- ORB-SLAM3 配置；
- Open3D 配置；
- renderer profile；
- tile/provenance schema。

任一变化：

```text
拒绝旧缓存，创建新 task-id
```

### 19.3 中断

用户 `Ctrl+C`：

- 写 `interrupted_resumable`；
- 保留检查点；
- 不写正式 `video_failure.json`；
- 下次相同命令自动续跑。

真实异常：

- 写 `video_failure.json`；
- 保留可恢复缓存；
- 不发布 `video_delivery.json`。

### 19.4 成功清理

二维发布后：

- 删除二维 renderer 的大体积临时 tile；
- 保留 SHA-256 清单；
- 保留实际源列表；
- 保留 ORB pose；
- 保留 Open3D 边；
- 保留三维续跑的小型检查点；
- 三维重新读取原始 RGB-D。

三维成功后可继续清理三维大体积缓存。

提供：

```text
--discard-cache
```

仅删除当前任务缓存，不触碰输入数据和正式交付。

### 19.5 状态文件

持续原子更新：

```text
video_status.json
video_3d_status.json
```

记录：

- 当前阶段；
- 已处理帧；
- 已完成 ORB 段；
- 已完成 Open3D 边；
- 已完成 tile；
- 已完成 Deep Zoom 层级；
- 已完成 TSDF chunk；
- 预计剩余量；
- 是否可恢复；
- task-id；
- 缓存路径；
- 最近错误。

## 20. 磁盘和资源预检

运行前估算：

- BigTIFF；
- Alpha；
- owner；
- source coordinates；
- Deep Zoom；
- source thumbnails；
- JPEG/PNG；
- ORB staging；
- renderer cache；
- TSDF cache；
- desktop/mobile GLB。

要求：

```text
可用空间 >= 估算峰值 × 1.5
```

不足时在处理前失败。

若预计 BigTIFF 超过 `4 GB`，输出文件系统必须支持大文件。

不支持时：

- 启动前拒绝；
- 提示改用 NTFS 或其它大文件文件系统。

当前开发机基线：

```text
GPU: NVIDIA GeForce RTX 5060 Laptop GPU
显存: 8 GB
Open3D: 0.19.0 CUDA build
实际 odometry backend: open3d_tensor_cuda_rgbd
WSL ORB-SLAM3 executable: 可用
ORB vocabulary: 可用
```

当前磁盘只作为开发现状参考，正式任务仍实时预检。

## 21. 性能目标

验收场景：

- 约 `3 m` 长物体/场景；
- 视频正式速度 `1.0 m/s`；
- `1280×800 @ 30 FPS`；
- 固定曝光 `800 µs`；
- 与照片模式相同起止位置和场景；
- 冷启动、无可复用缓存。

端到端计时包括：

- 输入验证；
- ORB-SLAM3；
- Open3D；
- 二维 renderer；
- BigTIFF；
- provenance；
- Deep Zoom；
- 二维发布；
- 随后自动三维；
- desktop/mobile GLB；
- 三维 Viewer。

目标：

```text
视频端到端总时间 <= 对应照片模式总时间 × 1.5
```

为达到该目标：

- `<=200 MP` 使用现有内存 renderer；
- ORB 对完整视频段运行；
- Open3D 只测量实际 renderer 源边；
- Deep Zoom 与 render tile 复用；
- 颜色和布局证据只做必要 pass；
- TSDF 现有容量内直接复用当前后端。

## 22. 现有真实视频样本

### 22.1 长曝光/丢帧压力样本

```text
D:\central_strip_Panoramic_Camera\data\captures\run_20260731_134559
```

只读检查结果：

- schema：`panorama-demo-session/v1`
- mode：`continuous_rgbd_video_auto`
- clean shutdown：true
- received：484
- written：333
- queue drops：151
- write errors：0
- timestamp regressions：0
- RGB 文件：333
- aligned depth 文件：333
- exposure：全部 `30,000 µs`
- 最大相邻已写时间间隔：约 `100 ms`
- depth valid median：约 `73.4%`
- RGB pair 匹配可靠率：`100%`
- 可找到连续单向扫描段

用途：

- 旧 v1 会话兼容；
- 自动曝光 metadata；
- 丢帧；
- 长曝光；
- C 级降级；
- 长序列与缓存；
- 不能作为 A/B 正常样本；
- 不能单独作为结构失败 F 样本。

### 22.2 当前较正常视频样本

```text
D:\central_strip_Panoramic_Camera\data\captures\run_20260731_141616
```

只读检查结果：

- schema：`panorama-demo-session/v1`
- mode：`continuous_rgbd_video_auto`
- clean shutdown：true
- frames：87
- queue drops：0
- write errors：0
- timestamp regressions：0
- exposure：`4,800–5,300 µs`
- 帧间隔：约 `33.37 ms`
- 时长：约 `2.87 s`
- depth valid median：约 `78.3%`
- RGB pair 匹配可靠率：`100%`
- 可找到连续单向扫描段

用途：

- 正常结构路径开发；
- ORB/Open3D/renderer 功能；
- 自动曝光颜色变化；
- 反方向扫描；
- 因曝光大于 `1200 µs`，正式等级最高仍为 C；
- 不能作为 A/B 正常样本。

以上检查尚未等同于：

- 完整真实 ORB-SLAM3 验收；
- 真实 Open3D 全边验收；
- 正式二维发布；
- TSDF/GLB 验收。

这些必须在实现后执行。

## 23. 需要补采的实机样本

实现固定曝光选项后补采：

### 23.1 正常左到右

- `1280×800 @ 30 FPS`
- `800 µs`
- 自动增益
- 自动白平衡
- `1.0 m/s`
- 完整约 3 m 场景
- 无丢帧
- 单向左到右

### 23.2 正常右到左

- 与左到右相同；
- 同一静止场景；
- 相同起止空间范围；
- 单向右到左；
- 用于真实空间左到右输出验收。

### 23.3 配对照片样本

对同一场景、相同起止位置和方向采集照片模式。

用途：

- 同一 renderer 算法验证；
- 二维布局和接缝比较；
- 三维 mesh 比较；
- 性能比较。

### 23.4 ORB 结构失败样本

故意：

- 中途完全遮挡镜头超过 `250 ms`；或
- 快速跳转导致轨迹断裂。

用途：

- 不跨断点；
- 最长连续段；
- 不插值；
- 不伪造 pose；
- 必要时 F。

## 24. 配对验收指标

### 24.1 二维照片/视频一致性

验收工具可以为比较共同区域做只读对齐，但该对齐不得回传生产全景。

指标：

```text
强结构边缘误差 median <= 1.5 px
强结构边缘误差 P95 <= 3 px
单位扫描跨度对应的全景尺度差 <= 5%
```

### 24.2 三维照片/视频一致性

共同可见区域：

```text
双向 mesh 距离 median <= 10 mm
双向 mesh 距离 P95 <= 30 mm
扫描方向包围盒跨度差 <= 5%
```

这些值与当前：

```text
voxel_length_mm=5
sdf_truncation_mm=20
```

相匹配。

## 25. 测试计划

新增建议：

```text
tests/test_video_capture_exposure.py
tests/test_video_session.py
tests/test_video_scan_segment.py
tests/test_video_orbslam3_pipeline.py
tests/test_video_orb_segments.py
tests/test_video_source_selection.py
tests/test_video_pose_pipeline.py
tests/test_video_renderer_profile.py
tests/test_video_residual_models.py
tests/test_video_tiled_renderer.py
tests/test_video_bigtiff.py
tests/test_video_provenance.py
tests/test_video_viewer.py
tests/test_video_delivery.py
tests/test_video_3d_delivery.py
tests/test_video_resume.py
tests/test_video_output_routing.py
tests/test_video_pipeline_isolation.py
tests/test_video_integration.py
```

### 25.1 捕获测试

- 默认仍为自动曝光；
- 固定曝光关闭 AE；
- 增益和白平衡仍自动；
- 100 µs 量化；
- readback；
- metadata；
- 三帧不一致失败；
- 设备不支持值拒绝；
- 固定曝光 manifest mode；
- v2 产品资格；
- 照片模式完全不变。

### 25.2 ORB 与轨迹

- 完整主扫描段送入 ORB；
- tracked fraction；
- selected source 100% pose；
- 无插值；
- 250 ms 断点；
- 最长连通段；
- 2秒重叠；
- 20共同帧；
- 段间 SE(3)；
- Open3D 使用 ORB 初值；
- CUDA backend 审计；
- 边残差；
- 双方向；
- 真实空间朝向。

### 25.3 renderer

- 照片默认像素和 owner 完全不变；
- 视频和照片相同输入调用同一 renderer；
- affine/homography/mesh/flow逐级选择；
- 16 px 上限；
- held-out；
- FB flow；
- Jacobian；
- foreground owner-only；
- one inverse sampling；
- 黑色 RGB 有效；
- Alpha/owner/provenance 一致。

### 25.4 分块

- 小图内存/分块逐像素一致；
- tile halo；
- 跨 tile seam；
- 跨 tile MultiBand；
- owner 单调；
- source coordinates；
- 超过200 MP自动切换；
- 4 GB working set；
- 任意源数；
- 中断续跑。

### 25.5 发布

- 二维 marker 最后写；
- 二维 marker 不可修改；
- 3D marker 独立；
- 3D失败不撤销二维；
- 三维失败主CLI退出0并警告；
- `--defer-3d`；
- `g305-video-3d`；
- hash验证；
- output子目录路由；
- 不覆盖照片；
- 不删除 `data`。

### 25.6 Viewer

- 离线；
- HTTP；
- localhost；
- DZI；
- click provenance；
- raw source coordinates；
- thumbnails；
- 3D按钮条件显示。

### 25.7 照片完整回归

必须运行：

```powershell
$G305Python = 'D:\Panoramic_Camera\.conda\python.exe'
& $G305Python -m pytest -q
ruff check src tests
& $G305Python -m compileall -q src tests
git diff --check
```

以及 AGENTS.md 中照片、会话、位姿、renderer、发布和TSDF全部相关测试。

## 26. 计划中的文件变化

新增：

```text
src/panorama_demo/video_config.py
src/panorama_demo/video_session.py
src/panorama_demo/video_scan_segment.py
src/panorama_demo/video_orb_pipeline.py
src/panorama_demo/video_source_selection.py
src/panorama_demo/video_pose_pipeline.py
src/panorama_demo/video_render_adapter.py
src/panorama_demo/video_tiled_render.py
src/panorama_demo/video_bigtiff.py
src/panorama_demo/video_provenance.py
src/panorama_demo/video_viewer.py
src/panorama_demo/video_delivery.py
src/panorama_demo/video_3d.py
src/panorama_demo/video_panorama.py
```

共享 renderer 可能需要内部泛化，但保留 public photo API：

```text
src/panorama_demo/calibrated_rgb_pushbroom.py
```

三维超长后端可能需要扩展：

```text
src/panorama_demo/dense_fusion.py
```

采集和入口：

```text
src/panorama_demo/capture_orbbec.py
pyproject.toml
```

文档和配置在实施时更新：

```text
AGENTS.md
README.md
configs/demo.yaml 或独立视频默认配置
```

普通用户不开放算法阈值；开发配置须严格验证未知键。

## 27. 实施阶段

### 阶段 0：照片基线冻结

- 记录当前照片测试结果；
- 建立固定输入的 panorama/owner/metadata golden；
- 记录照片性能基线；
- 不改变代码语义。

### 阶段 1：视频采集与输入

- `--video-exposure-us`；
- video session v2；
- product eligibility；
- legacy v1兼容；
- 独立CLI；
- 输出路由；
- status/cache骨架。

### 阶段 2：ORB与真实源

- 视频扫描段；
- 完整 ORB；
- 95%跟踪；
- 250 ms断点；
- 分段ORB；
- 源选择；
- Open3D边；
- 双方向空间规范化。

### 阶段 3：共享renderer视频profile

- 先用现有内存renderer；
- 确认照片零回归；
- metadata颜色先验；
- affine；
- homography；
- mesh/flow；
- 模型审计；
- A/B/C/F。

### 阶段 4：BigTIFF与provenance

- RGBA BigTIFF；
- owner/source coordinates；
- JPEG/PNG规则；
- Deep Zoom；
- 离线Viewer；
- thumbnails；
- click provenance。

### 阶段 5：超长分块与续跑

- >200 MP流式后端；
- 2–5源窗口；
- 4 GB工作集；
- tile等价性；
- SHA-256；
- cache；
- resume；
- disk预检。

### 阶段 6：二维正式发布

- 独立schema；
- 原子发布；
- marker不可修改；
- output路由；
- failure/interruption。

### 阶段 7：三维独立交付

- 现有TSDF直接后端；
- `--defer-3d`；
- `g305-video-3d`；
- 3D status/delivery/failure；
- 分块TSDF；
- Viewer联动。

### 阶段 8：实机和性能

阶段 8 必须严格按以下顺序执行，不得提前要求用户补采：

1. 先完成全部单元测试和合成集成测试；
2. 单元/合成测试通过后，使用现有两段视频完成 C 级真实数据实测：
   - `run_20260731_134559`
   - `run_20260731_141616`
3. 两段现有视频的 C 级实测通过后，再通知用户补采：
   - `800 µs` 左到右正常视频；
   - `800 µs` 右到左正常视频；
   - 同场景配对照片；
   - ORB 断点视频；
4. 使用补采数据完成：
   - A/B 正常发布验收；
   - 真实空间左右方向验收；
   - ORB 断点截断/F验收；
   - 二维一致性；
   - 三维一致性；
   - 1.5×性能。

现有两段 C 级实测尚未通过前，不要求用户重新采集数据。

## 28. 明确不做

- 不把视频分支塞入 `g305-panorama`；
- 不让照片接受视频会话；
- 不让视频使用照片 `delivery.json`；
- 不修改照片默认像素；
- 不用三维数据补RGB；
- 不用Open3D替代缺失ORB pose；
- 不用二维模型替代ORB轨迹；
- 不插值pose；
- 不跨ORB断点；
- 不删除输入 `data`；
- 不依赖互联网CDN；
- 第一版不新增Torch、LightGlue、LoFTR等大型依赖；
- 不承诺动态场景；
- 不承诺超过 `1.0 m/s` 的视频正式现场验收。

## 29. 完成定义

只有以下全部满足，改造才算完成：

1. 当前照片模式全量测试通过且默认产物完全兼容；
2. 视频自动曝光旧v1会话可处理；
3. 视频固定曝光v2采集可审计；
4. 完整主扫描段ORB真实运行；
5. Open3D CUDA真实边通过；
6. 视频实际源100%真实pose；
7. 同一共享renderer生成二维全景；
8. 视频残差模型通过完整审计；
9. BigTIFF、RGBA、owner、raw source coordinates一致；
10. Deep Zoom离线Viewer和点击溯源可用；
11. 二维独立A/B/C/F和原子发布通过；
12. 二维成功后可延后或自动生成三维；
13. 三维格式和内容规则与照片一致；
14. 三维失败不撤销二维；
15. 超长分块、缓存、续跑和磁盘预检通过；
16. 现有两个C样本完成实测；
17. 新800 µs双方向样本完成实测；
18. ORB断点样本正确截断或F；
19. 照片/视频二维和三维配对指标通过；
20. 3 m端到端视频时间不超过照片模式1.5倍。

## 30. 当前验收状态

本文件只冻结方案，尚未开始代码改造。

当前已确认：

- 源码和模块边界已检查；
- 当前Open3D CUDA环境可用；
- 当前ORB-SLAM3 executable和vocabulary可用；
- 两段旧视频会话结构和基础RGB/depth指标已只读检查；
- 用户已同意补采正常、反向、配对照片和ORB断点样本。

尚未完成：

- 单元/合成测试；
- 新固定曝光采集；
- 真实完整ORB运行；
- 真实Open3D全边；
- 视频二维正式交付；
- BigTIFF/provenance/Viewer；
- 超长分块；
- TSDF/GLB；
- 现场1.0 m/s；
- 1.5×性能验收。
