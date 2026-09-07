# Gemini 305 全景程序 WSL-first Linux SDK L0–L1 Codex 直接实施方案

> 基线日期：2026-08-25（Asia/Shanghai）
> Windows 仓库：`D:\central_strip_Panoramic_Camera`
> 基线分支：`codex/s013-v11-live-production`
> 基线提交：`f8f624cee310678c4605eb14764c7bc2509158f9`
> 本文范围：先在现有 WSL Ubuntu 22.04 完成 L0 Linux 移植和 L1 SDK 基本开发/回放基线，再到原生 Ubuntu 做最终相机验收。

## 1. 先给出平台结论

1. **现在不安装新的图形化虚拟机，也不把原生 Ubuntu 当作开发前置条件。** 当前 WSL2 Ubuntu 22.04.5 是 L0–L1 的主开发和验收环境。
2. **先在 WSL 中做到软件基本完成。** Linux CLI 移植、视频 SDK API、wheel/资源自包含、Runtime doctor、CUDA/CuPy、Open3D 0.19 CUDA、Linux ELF ORB-SLAM3、真实录制 RGB-D 回放、CLI/SDK 精确等价和重复性能基线都在 WSL 完成。
3. **SDK 基本完成后，再借用或准备一台原生 Ubuntu 机器。** 该机器只承担 Gemini 305 的 non-root USB/udev、精确 `848×480 @ 60 FPS`、metadata、Trigger Out、断连恢复、持续写盘和最终现场性能验收，不承担日常开发。
4. **最终资格机优先选与开发基线一致的 Ubuntu 22.04 LTS x86_64。** 无图形桌面的 Ubuntu Server 即可；若使用 24.04，则把它作为额外兼容目标单独记录，不能自动反推 22.04 或其它发行版。GUI 不是 SDK 运行条件。
5. **WSL 软件通过和原生硬件通过使用两个不同状态。** WSL 完成后可写 `SDK_SOFTWARE_READY`；只有原生 Ubuntu 真机通过后才能写 `SDK_HARDWARE_QUALIFIED`。两者都通过且 ORB 分发许可明确后，才可讨论 `SDK_RELEASE_READY`。
6. **当前约 `34.6 GiB` 的 D 盘可用空间按用户决定直接用于本轮 WSL CUDA/Open3D 构建，不再设置 `80 GiB` 开工硬门。** WSL VHD 继续留在 D 盘，不迁移、不改分区；E 盘保持 `Warning / Full Repair Needed`，本轮不使用。Codex 采用单一源码/构建树和无冗余缓存策略，分阶段记录宿主 D 盘与 WSL 实际用量；只有出现真实 `ENOSPC`、下一步明确无法容纳，或 D 盘可用空间低于 `10 GiB` 安全余量时才暂停并请求用户处理。

推荐路线固定为：

```text
现有 WSL Ubuntu 22.04
  → L0 Linux CLI / CUDA / Open3D / ORB 原生用户态移植
  → L1 视频 SDK、wheel、doctor、真实录制回放与性能基线
  → SDK_SOFTWARE_READY
  → 借用或准备原生 Ubuntu 机器
  → 同一 wheel 的原生 Runtime 与冻结数据回放预检
  → Gemini 305 USB / udev / 60 FPS / 同步 / Preview / Capture / P3 最终验收
  → SDK_HARDWARE_QUALIFIED
```

## 2. Codex 执行规则

把本文交给 Codex 后，Codex 必须遵守以下规则。

可直接使用的开工指令：

```text
按照 docs/LINUX_L0_L1_CODEX_IMPLEMENTATION_PLAN.md 实施 WSL-first L0–L1。
先执行 L0.0 并记录当前基线，再在现有 WSL Ubuntu 22.04 中连续完成 Linux 移植、
视频 SDK、CUDA/Open3D、ORB、wheel/doctor、真实录制数据回放和性能基线；不要只写方案。
sudo 检查点由用户配合；当前磁盘预算已获准直接实施，构建期间按文档监测。原生 Ubuntu 和相机不是 L0–L1 开工条件，
而是 SDK_SOFTWARE_READY 之后的最终 H0 硬件验收。禁止把未执行的 H0 写成通过。
```

### 2.1 每次实施的第一步

在 Windows 主工作区执行并记录：

```powershell
Set-Location -LiteralPath 'D:\central_strip_Panoramic_Camera'
git status --short
git branch --show-current
git rev-parse HEAD
```

- 第一条 Git 命令必须是 `git status --short`。
- 不得覆盖用户现有改动、采集目录或输出目录。
- 不得使用 `git reset --hard`、破坏性 checkout、`git add -A` 或递归删除工作区/环境/数据。
- 每个阶段只暂存该阶段明确列出的文件；验证通过后才提交。
- Windows 工作区是修改源。WSL ext4 checkout 只获取已提交的阶段，用于 Linux 构建和测试；不得同时在两个 checkout 中各自修改同一文件。
- `sudo apt`、udev 安装、磁盘修复、分区、原生系统安装和重启需要用户检查点。Codex 可以先准备并核对命令，但遇到密码/重启/磁盘变更时必须停止并请求用户执行或明确授权。

### 2.2 证据状态

所有验收项只能使用：

```text
PASS
PASS_WITH_WARNING
FAIL
NOT_EXECUTED
BLOCKED
```

同时写 `reason_code`，例如 `NO_CAMERA`、`NO_NATIVE_UBUNTU`、`HOST_DISK_SPACE`、`ORB_LICENSE` 或 `WINDOWS_OPEN3D_BROKEN`。`NOT_EXECUTED` 和 `BLOCKED` 绝不能写成 `PASS`。

### 2.3 不能混淆的证据

- 合成测试不是实机证据。
- Windows 历史输出不是当前 HEAD 的新基线。
- WSL 回放是真实录制数据的 Linux 回放证据，不是 Linux 相机采集证据。
- WSL 通过 `usbipd-win` 看到相机也不能替代原生 Ubuntu USB/udev/60 FPS 验收。
- 不同采集 session 的时间不能互称平台优化。
- GPU 利用率只能作为描述，不能单独证明某算法已经在 GPU 上运行。

## 3. 当前电脑与仓库的已验证状态

### 3.1 宿主机

| 项目 | 当前值 | 对实施的含义 |
| --- | --- | --- |
| 操作系统 | Windows 11 Home，build 26200 | 保留为迁移前参考环境 |
| CPU | Intel Core 7 245HX，14 核 | 可做受限并行构建 |
| 内存 | 约 15.4 GiB | Open3D 构建并行度必须受限，默认不超过 2 |
| GPU | NVIDIA GeForce RTX 5060 Laptop，约 8 GiB，compute capability 12.0 | Open3D 构建目标为 `sm_120` |
| Windows 驱动 | 610.62，CUDA UMD 13.3 | 驱动可供 WSL GPU 透传；不代表 WSL 已装 `nvcc` |
| Gemini 305 | 当前未连接，SDK 查询设备数 0 | 所有新真机项当前必须是 `NOT_EXECUTED/NO_CAMERA` |

### 3.2 磁盘

| 卷 | 容量 | 可用 | 健康状态 | 决策 |
| --- | ---: | ---: | --- | --- |
| C: | 约 274.5 GiB | 约 56.4 GiB | Healthy | 可保留现有系统，不作为重型构建目标 |
| D: | 200 GiB | 约 34.6 GiB | Healthy | 当前工作区和 WSL VHD 所在卷；用户已接受该预算直接构建，保留 `10 GiB` 安全余量 |
| E: | 约 953.9 GiB | 约 855.1 GiB | Warning / Full Repair Needed | 本轮不使用，不作为 L0–L1 前置 |

WSL VHD 位于 `D:\WSL\Ubuntu\ext4.vhdx`，物理文件当前约 9.1 GiB。WSL 内看到约 949 GiB 的逻辑文件系统上限，不等于 D 盘真实可用空间；因此构建过程中以 Windows 宿主 D 盘的实际可用空间为停止依据，而不是 `df` 显示的逻辑上限。

### 3.3 Windows Python 环境

主环境：`D:\Panoramic_Camera\.conda`

| 组件 | 当前值 | 状态 |
| --- | --- | --- |
| Python | 3.12.13 | 可用于迁移前二维基线 |
| NumPy | 2.5.1 | 已安装 |
| OpenCV | 5.0.0 | 已安装 |
| CuPy | 14.1.1 | 可识别 RTX 5060 |
| pyorbbecsdk2 distribution | 2.1.1 | runtime SDK 报告 2.8.6 |
| Open3D | 0.19.0+1e7b174 | **当前导入失败**：`No module named 'open3d.cpu'` |

因此，当前 Windows 可建立正式二维基线，但不能把现有 Windows Open3D/3D 当作当前有效性能基线。L0–L1 不为建立对照而顺手重建 Windows Open3D；三维跨系统对比应如实标为 `NOT_EXECUTED/WINDOWS_OPEN3D_BROKEN`，Linux 三维仍须独立完成正确性和时间记录。

### 3.4 现有 WSL2

| 项目 | 当前值 |
| --- | --- |
| 发行版 | Ubuntu 22.04.5 LTS，WSL2 |
| kernel | 6.18.33.2-microsoft-standard-WSL2 |
| systemd | running |
| 默认用户 | `zyh`，uid 1000 |
| GPU 透传 | `/dev/dxg` 存在，`nvidia-smi` 可见 RTX 5060 |
| Python | 3.10.12 |
| GCC | 11.4 |
| CMake | 3.22.1，低于本次固定 Open3D 构建要求 |
| Ninja | 1.10.1 |
| 当前缺少 | `pip`、`nvcc`、Python `cv2/open3d/cupy/pyorbbecsdk`、`lsusb` |

当前仓库只通过 `/mnt/d` 暴露给 WSL，`/home/zyh` 下没有该仓库的 ext4 checkout。正式 Linux 计时不得从 `/mnt/d` 运行；应在 ext4 checkout 和 ext4 数据副本上执行。

### 3.5 现有 ORB-SLAM3

现有目录：

```text
/home/zyh/Projects/ORB_SLAM3_WS/ORB_SLAM3
```

已验证：

- 上游 HEAD：`4452a3c4ab75b1cde34e5505a36ec3f9edcdc4c4`
- `Examples/RGB-D/rgbd_tum_headless` 存在，当前 `ldd` 未发现缺失依赖。
- `Examples/RGB-D/rgbd_g305_stream_headless` 和 `Vocabulary/ORBvoc.txt` 存在。

阻断点：该源码树不是可重建的干净来源。`CMakeLists.txt` 有未提交修改，两个 headless runner 源码未跟踪，并存在备份文件。现成 executable 只能算“当前开发运行时”，不能宣称为可复现 Linux 安装或未来 SDK 组成部分。

ORB-SLAM3 上游是 GPLv3。L0 可把它作为独立外部进程用于开发和内部验收，但不得在没有 GPL 分发方案或商业许可结论时，把 ORB binary、Vocabulary 或派生 runner 打入未来闭源 SDK、wheel 或 `.deb`。

### 3.6 当前代码中特定的 Linux 阻断面

| 路径 | 当前阻断 |
| --- | --- |
| `src/panorama_demo/orbslam3_bridge.py` | `wsl.exe`、Windows→WSL 路径转换、`orbslam3_rgbd_wsl` backend |
| `src/panorama_demo/online_orbslam3_bridge.py` | 复用 WSL path/process helper |
| `src/panorama_demo/video_online_orb.py` | 诊断在线 ORB 仍绑定 WSL helper |
| `configs/demo.yaml` | ORB 注释和路径按 WSL 书写 |
| `src/panorama_demo/cuda_backend.py` | Windows DLL/cuDNN PATH 配置和 `D:\...` 硬编码 |
| `src/panorama_demo/capture_orbbec.py` | `msvcrt` 单键停止与 Windows metadata registration 提示 |
| `src/panorama_demo/video_3d_launcher.py` | Windows priority flag；Linux 当前虽会落为 0，仍需测试 native spawn |
| `src/panorama_demo/video_holdout.py` | 只生成 PowerShell 辅助脚本；不阻塞五个正式 CLI，但 Linux 文档不能引用它作为标准入口 |
| `src/panorama_demo/sdk.py` / `src/panorama_demo/__init__.py` | 目前公开 SDK 只覆盖照片产品；尚无连续视频 facade、typed job/result 和软件/硬件分层状态 |
| `src/panorama_demo/paths.py` / `src/panorama_demo/config.py` | 默认资源仍按源码树 `PROJECT_ROOT/configs/...` 解析，仓库外 wheel 不能据此宣称可运行 |
| `pyproject.toml` | 尚未声明视频 production lock、阈值 witness 和离线 Viewer 等 Runtime 资源的 wheel 打包契约 |
| `README.md` | 主要示例是 PowerShell，并且仍错误宣传 frozen P0/tail-only 正常路径 |

### 3.7 本轮已做的最小当前源码验证

```text
pytest -q tests/test_video_live.py tests/test_video_s13_live.py
21 passed
```

这证明当前源码和测试的 live handoff 是 `validated_inputs_only`，不证明 Linux 已经可运行。

## 4. 阶段范围与完成顺序

### 4.1 L0：现有 WSL 中完成 Linux Runtime 移植

五个目标 CLI：

```text
g305-capture
g305-video-live
g305-video-panorama
g305-video-post-3d
g305-orbslam3-trajectory
```

它们在 WSL 内必须直接运行 Linux Python、Linux CUDA/Open3D 和 Linux ELF ORB-SLAM3，不得从 Linux 进程反向启动 `wsl.exe`。WSL 中无相机时，采集入口完成 import、参数、缺设备轮询/取消和 mock contract 测试；真实 USB 采集留给 H0。

### 4.2 L1：现有 WSL 中完成视频 SDK 基本开发

L1 必须完成：

1. 保留既有照片 `PanoramaSDK` 行为，新增连续视频 SDK facade 和 typed results。
2. SDK 复用五个既有正式入口/函数，不复制 S013、ORB 或 TSDF 算法。
3. clean wheel 在仓库外 venv 可安装，默认配置、production lock、阈值 witness 和 Viewer 静态资源可加载。
4. `sdk.doctor()` 在 WSL 中审计 Python、V11 identity、CuPy、Open3D CUDA、native ORB、Orbbec wrapper 和相机状态。
5. 对同一冻结真实 RGB-D session，SDK 与 CLI 的 P3、owner、provenance、decision 和 3D 隔离契约精确一致。
6. 建立 WSL 内 SDK/CLI 重复性能基线，二维、SDK overhead、ORB 和 TSDF/GLB 分开报告。
7. 形成可安装 wheel、constraints、Runtime manifest、使用文档和 H0 原生真机验收包。

达到上述条件后状态为 `SDK_SOFTWARE_READY`。它表示软件、打包和录制数据回放已完成，不表示 Gemini 305 已在原生 Ubuntu 通过。

### 4.3 H0：SDK 基本完成后的原生 Ubuntu 最终验收

H0 在借用或准备好原生 Ubuntu 机器后执行：

1. 安装 WSL 阶段产生的同一 wheel，并按冻结依赖版本和构建配方准备 native external Runtime；不在验收机临时改源码。
2. 先对同一冻结 session 做 native CLI/SDK 回放预检，确认移机与 GPU Runtime 没有引入正确性差异。
3. 普通用户发现 Gemini 305，完成 udev、USB 3.x、metadata 和同步 readback。
4. 完成 `848×480 RGB + 848×480 Y16 @ 60 FPS`、3 秒、Preview/Capture/P3 和独立 post-3D 验收。
5. 代码、打包或资源缺陷回到 WSL 修复，并重新生成 `SDK_SOFTWARE_READY` artifact；udev、驱动、USB、线缆、供电或机器特定 Runtime 问题则保持同一 wheel，在原生机修复环境后用新 H0 ID 重测。两类问题都不得覆盖旧失败证据。

H0 通过后状态为 `SDK_HARDWARE_QUALIFIED`。

### 4.4 明确排除项

以下内容不属于本轮“SDK 基本完成”，禁止顺手实现：

- gRPC、systemd service、SQLite、Task Manager、Control Lease；
- `/dev/shm` BGRA triple buffer 或共享内存预览；
- 任务队列、3D 抢占、TSDF checkpoint；
- 长时间 segment、tiled panorama；
- `.deb`、正式 bootstrap installer、`/opt/panoramic-sdk` 布局；
- PLY/PCD/OBJ 新格式；
- 原始数据自动删除；
- 新的 P0、窄全景或 single-frame fallback；
- 为迁移方便而修改 S013 V11 像素算法、阈值或生产身份。

本轮实现的是进程内 Python 视频 SDK、wheel 和 Runtime 诊断，不是 daemon/service SDK。ORB binary、Vocabulary、CUDA driver/toolkit 和 Orbbec 系统权限不能伪装成 wheel 自包含依赖。

## 5. 不可破坏的正式契约

### 5.1 S013 V11 身份

必须保持：

```text
algorithm_id = S013_output_first_progressive_dense_central_slit_v4_cuda_m63_visual_continuity_v11
implementation_id = s013_m5_disjoint_seam_m63_pair_guard_v11
candidate canonical SHA-256 = a03f443aaf72463afc1f06507fe2bfc888d9211f4c4a62634bbfb29fa0f2dab0
production canonical SHA-256 = 3a49f8cbaa622592eff41b2a9e34cb518c6067d74d15574cf01faeea0cc8fafc
role = production
allow_baseline_fallback = false
execution_backend = s013_v11_production_cuda
```

禁止修改：

```text
configs/video_candidates/s013/S013_output_first_progressive_dense_central_slit_v4_cuda_m63_visual_continuity_v11.yaml
configs/video_algorithms/s013_visual_continuity_v11_production.yaml
configs/video_algorithms/s013_visual_continuity_v11_production.lock.json
```

若 Linux 迁移导致这些文件的字节或 canonical identity 改变，立即停止，不得重新生成 lock 来掩盖变化。

### 5.2 必须纠正附件中的 frozen P0 旧描述

当前正式源码和测试要求：

```text
reuse_level = validated_inputs_only
pair_evidence_reused_count = 0
frozen_p0_authority = None
full_m0_m3_recomputed = true
```

权威证据位于 `src/panorama_demo/video_s13_live.py` 的 live handoff 构造，以及 `tests/test_video_s13_live.py` 的 acceptance assertions。README 中 frozen P0/tail-only 正常路径描述已经过时，必须在 L0 文档提交中修正。Linux 迁移不得重新启用 frozen P0 或扩大复用 authority。

### 5.3 时序和隔离

- 正式采集和二维期间，`OnlineORBTracker` 构造、ORB process、ORB submit、Open3D import、TSDF 和 3D process 调用次数全部为 0。
- Preview 是 `non_authoritative_live_preview`；失败只能隔离并记录，不能冒充正式 P3。
- 停采后先停止/join Preview，再 drain writer，再从 committed session 完整重算 V11 P0–P3。
- `video_delivery.json` 已发布且二维资源释放后，才可创建 post-3D 子进程。
- 3D 失败只写 `3d/video_3d_failure.json`，不得修改、删除或撤销二维结果。

### 5.4 Fail-closed 条件

以下任一情况使**视频正式二维**失败，不能增加成功 fallback：

- session 不完整或 `clean_shutdown=false`；
- drops/write errors 非零，或 received 与 written 不相等；
- production identity/SHA 不符；
- `allow_baseline_fallback=true` 或实际发生 fallback；
- `G305_CUDA=required` 边界不可用；
- 没有有效空间扫描段；
- owner、valid mask、provenance 或最终原子发布结构失败；
- 二维任一现有 CUDA-required 边界不可用或记录 CPU fallback。

ORB/Open3D/TSDF 属于视频二维发布后的独立 3D 子交付。ORB 轨迹缺 pose、非有限、非刚体、不完整，使用插值/伪 pose，或 3D 的现有 CUDA-required 边界发生 fallback 时，`video_3d_delivery` 必须失败并写 `3d/video_3d_failure.json`，但已经通过的二维 `video_delivery.json` 继续有效。独立 `g305-orbslam3-trajectory` 命令自身仍应以非零退出失败。照片正式产品的 ORB/Open3D 结构失败继续使照片整体失败。

## 6. L0 直接实施步骤

L0 全部在现有 WSL Ubuntu 22.04 中实施。原生 Ubuntu 和相机不再是 L0 完成前置；当前 D 盘预算已经由用户接受，可以直接进入 CUDA/Open3D 构建，无需先迁移 WSL、修复 E 盘或安装另一套 Ubuntu。

### L0.0 固定迁移前 Windows 基线

**目标**：在任何 Linux 改码前，为当前 HEAD 建立新的二维参考；已有旧输出只作 sanity seed。

优先冻结 session：

```text
D:\central_strip_Panoramic_Camera\data\captures\video8_25_3_right\run_20260825_123714
```

已验证 session 事实，以及现有旧提交输出提供的 workload 参考：

| 项目 | 值 |
| --- | ---: |
| 相机/固件/连接 | Gemini 305 / 1.0.70 / USB 3.2 |
| profile | 848×480 RGB + 848×480 Y16 @ 60 FPS |
| 采集时长 | 3.0 s |
| received / written | 152 / 152 |
| queue drops / write errors | 0 / 0 |
| clean shutdown / video eligible | true / true |
| 扫描 | right，`scan_direction=-1` |
| 可靠运动占比 | 1.0 |
| 旧输出 source / pair | 89 / 88（L0.0 新基线后重新确认） |
| 旧输出 P3 尺寸 | 1945×480（L0.0 新基线后重新确认） |
| 旧输出 invalid owner pixels | 0 |

冻结输入身份：

```text
manifest.json  = 1b6bc991ba86ad75a249857b360388f860ae38274fb15030742388cba5e44ba9
calibration.json = 9e19b8dc506b27834b4fa0166294deecb1c23d93e3b7bb93184b3aa8c5691330
frames.csv = b2a24151c41b1603a563f897f3fab1e954b99ee811ba9e5ee0201b370a942f7b
```

不修改原 session。三个摘要只用于快速确认控制文件身份，不能单独证明 RGB/depth 副本完整；L0.6 还会直接逐文件比较整个 session。不新增长期逐帧摘要清单。

执行：

```powershell
$G305Python = 'D:\Panoramic_Camera\.conda\python.exe'
$FrozenSession = 'D:\central_strip_Panoramic_Camera\data\captures\video8_25_3_right\run_20260825_123714'
$BaselineRoot = 'D:\linux_l0_l1_artifacts\windows_f8f624ce'
$env:G305_CUDA = 'required'

& $G305Python scripts\run_s013_v11_offline_acceptance.py `
  $FrozenSession --output "$BaselineRoot\offline" --maximum-post-seconds 60

& $G305Python scripts\run_s013_v11_live_replay.py `
  $FrozenSession --output "$BaselineRoot\live" `
  --cadence-scale 1.0 --maximum-post-seconds 60

& $G305Python scripts\verify_s13_v11_live_equivalence.py `
  "$BaselineRoot\offline" "$BaselineRoot\live" `
  --output "$BaselineRoot\live_offline_equivalence.json"
```

必须记录 command、stdout、stderr、exit code、Git HEAD、production identity、`video_report.json`、`video_timing.json`、P3 和 provenance。若当前 Windows live/offline 不等价，停止 Linux 改码，先按首个 P0–P3 差异诊断当前基线。

现有 `outputs/video_live8_25_3_right` 的历史单次值约为首次 Preview `1.781 s`、Preview P95 `16 ms`、停采到 P3 发布 `14.422 s`、ORB `18.218 s`、post-3D `36.375 s`，但其代码提交为旧的 `0f5ce584...`，不能冒充上述当前 HEAD 基线。

### L0.1 固化 ORB-SLAM3 可重建来源

**目标**：从干净、固定的上游提交重建 headless runner，不能继续依赖唯一一份 dirty WSL tree。

拟新增：

```text
scripts/build_orbslam3_linux.sh
scripts/patches/orbslam3-g305-headless-runners.patch
third_party/orbslam3-g305-runner/COPYING
THIRD_PARTY_NOTICES.md
tests/test_orbslam3_linux_build_manifest.py
```

实现要求：

1. 在任何清理、移动或重新构建旧 WSL ORB 目录前，把以下内容输出到仓库外证据目录：upstream HEAD、`git status --short`、完整 diff、两个 runner 源文件、各自摘要、binary 摘要和 `ldd`。
2. 构建脚本固定 ORB-SLAM3 upstream commit `4452a3c4ab75b1cde34e5505a36ec3f9edcdc4c4` 和已审计的 Pangolin commit `aff6883c83f3fd7e8268a9715e84266c42e2efe3`；只能从全新 source/build 目录构建，不得读取旧 binary 或旧 build cache。
3. patch 只包含本项目确实需要的 C++ 标准/CMake target 和两个 headless runner；不得顺手改 ORB 算法、Viewer、tracking 或 trajectory 语义。
4. 构建产物记录 source commit、patch identity、compiler、CMake、Pangolin、OpenCV、executable、Vocabulary 和完整 `ldd`。出现 `not found` 必须失败。
5. formal batch runner 必须关闭 Viewer，跟踪完整 RGB-D 序列，只在正常 shutdown 后写真实轨迹。
6. 该 GPL 派生目录必须带清楚的 GPLv3 notice，且不进入 Python wheel。若仓库不能接受 GPL 派生 patch，则将此项标记 `BLOCKED/ORB_LICENSE`，保留外部 runtime 边界，不能伪称可重建问题已解决。

阶段测试：

```bash
bash scripts/build_orbslam3_linux.sh \
  --work-root /home/zyh/build/g305-orbslam3 \
  --install-root /home/zyh/opt/g305-orbslam3

test -x /home/zyh/opt/g305-orbslam3/Examples/RGB-D/rgbd_tum_headless
test -f /home/zyh/opt/g305-orbslam3/Vocabulary/ORBvoc.txt
ldd /home/zyh/opt/g305-orbslam3/Examples/RGB-D/rgbd_tum_headless
```

建议提交：

```text
build(linux): make the G305 ORB-SLAM3 runner reproducible
```

### L0.2 把 ORB bridge 改成 Linux 原生进程

**目标**：在 WSL/Linux 上直接启动 ELF ORB，不再从 Linux 进程调用 `wsl.exe`。现有 Windows 产品若仍需回归，可保留显式的 Windows→WSL legacy backend，但绝不能成为 Linux fallback。

修改：

```text
src/panorama_demo/orbslam3_bridge.py
src/panorama_demo/online_orbslam3_bridge.py
src/panorama_demo/video_online_orb.py
src/panorama_demo/export_orbslam3_trajectory.py
src/panorama_demo/video_3d_postprocess.py
configs/demo.yaml
tests/test_orbslam3_bridge.py
tests/test_online_orbslam3_bridge.py
tests/test_export_orbslam3_trajectory.py
tests/test_video_3d_postprocess.py
```

实现要求：

- `ORBSLAM3Config` 增加显式 `runtime_kind=native_linux|windows_wsl_legacy`；Linux SDK 只接受 `native_linux`。
- native 配置使用 Linux root/executable/vocabulary 字段，不读取 `wsl_executable`、不做 Windows path conversion。
- Windows legacy path 只为现有 Windows 回归保留并隔离；配置选择后 fail-closed，不允许 native 失败后自动回退 legacy。
- 使用 Linux `Path` 解析 executable、Vocabulary、settings、sequence 和 association。
- executable 必须是可执行普通文件；Vocabulary/settings 必须存在。
- 以参数数组直接 `subprocess.run/Popen` native executable，不经 `bash -lc` 或字符串拼接；`cwd` 指向 private staging 目录。
- 保留现有 fresh-staging retry、timeout、轨迹解析、完整 frame coverage、真实 `camera_to_world` 和毫米单位契约。
- Linux backend 明确报告 `orbslam3_rgbd_native_linux`；Windows legacy 继续使用自己不同的 backend 名称，不能混写。
- `g305-video-live` 仍拒绝 diagnostic online ORB，正式二维 authority 仍不消费 ORB。

验证：

```bash
python -m pytest -q \
  tests/test_orbslam3_bridge.py \
  tests/test_online_orbslam3_bridge.py \
  tests/test_export_orbslam3_trajectory.py \
  tests/test_video_3d_postprocess.py
```

测试必须断言：Linux native command 中不存在 `wsl.exe`、`wslpath` 或 `/mnt/...` 转换；Windows legacy command 只能在 Windows backend 测试中出现。两者不得互相 fallback。

真实录制 RGB-D 验收必须证明：tracked frame IDs 完整、pose 有限且为刚体、无插值/外推、失败重试使用新 staging。不得以 Open3D-only、ignore-pose 或二维 motion 替代 ORB。

建议提交：

```text
refactor(orb): launch ORB-SLAM3 as a native Linux process
```

### L0.3 建立 Linux Python、CuPy 和 Open3D 0.19 CUDA 环境

**空间执行决策**：用户已明确接受当前 D 盘约 `34.6 GiB` 可用空间，允许立即在现有 WSL 中安装 CUDA 依赖并编译 Open3D。删除原 `80 GiB` 开工门，不迁移 WSL，不使用状态异常的 E 盘。空间控制属于运行监测，不再是阶段前置：

```powershell
# 每个重型步骤前后都在 Windows 宿主记录；这是实际停止依据。
Get-Volume -DriveLetter D |
  Select-Object DriveLetter, Size, SizeRemaining, HealthStatus
```

```bash
# 同时记录 WSL 视角及本轮明确目录的增长，不把 949 GiB 逻辑上限当成宿主空间。
df -h /
du -sh /home/zyh/build /home/zyh/venvs /home/zyh/g305-data 2>/dev/null || true
```

执行要求：

- 只保留一套固定 Open3D source tree、一套 `build-cuda12.8-sm120` 和一个最终 wheel，不为失败重试复制完整构建树。
- Python 下载使用 `PIP_NO_CACHE_DIR=1`；系统包使用 `--no-install-recommends`，但不得为了省空间删除现有环境、用户数据或失败证据。
- 源码、build tree 和正式计时数据仍放 WSL ext4；不得为了省空间改到 `/mnt/d` 后把性能结果冒充 ext4 基线。
- D 盘实际可用空间保持不少于 `10 GiB`。低于该值、出现真实 `ENOSPC`，或下一已知构建步骤明确无法容纳时，停止当前重型步骤并保留日志；不得自动清理或迁移，交由用户决定。
- 成功完成后记录实际 peak/remaining space，用实测结果替代此前的空间估算。

拟新增/修改：

```text
requirements/linux-l0-constraints.txt
scripts/build_open3d_cuda_linux.sh
scripts/verify_open3d_cuda.py
src/panorama_demo/cuda_backend.py
tests/test_cuda_backend.py
tests/test_rgbd_odometry.py
tests/test_dense_fusion.py
tests/test_video_s13_cuda_runtime.py
tests/test_video_s13_m6_cuda.py
tests/test_video_cuda_renderer.py
docs/LINUX_NATIVE_CLI.md
```

环境边界：

- Python 保持项目支持范围 `>=3.10,<3.13`；WSL 主 Runtime 使用 Python 3.10 venv。
- 升级 venv 内 CMake 到 `>=3.24`，不依赖当前系统 CMake 3.22.1。
- CuPy 使用 `cupy-cuda13x>=14.1,<15`，记录它实际报告的 runtime/device。
- Open3D 固定 0.19 源码提交 `1e7b17438687a0b0c1e5a7187321ac7044afe275`、CUDA compiler toolkit 12.8 和 `CMAKE_CUDA_ARCHITECTURES=120`。
- Open3D 与 CuPy 可以使用不同 toolkit/runtime 边界，报告必须分开；不得把 ORB、GraphCut 或 MultiBand 宣称为 GPU 算法。
- WSL 只安装 CUDA toolkit，**不得安装 Linux NVIDIA display driver**。使用执行当日 NVIDIA 官方 WSL 安装方式，避开会拉取 Linux driver 的 meta-package。
- Linux 不应用 `open3d-0.19-windows-cuda-shutdown.patch`。仅在 clean build 证明需要且 patch 可应用时，使用现有 CUDA13/CCCL 和 stdgpu patch。
- 构建并安装单一 Open3D CUDA wheel；避免 CPU wheel 与 CUDA wheel 混装。
- 16 GiB 主机默认构建并行度不超过 2；发生内存压力时降低为 1，不通过改算法规避。

在 WSL 中先建立 ext4 checkout。若目标目录已存在，先检查其 status 和归属，不得覆盖；只允许 clone 到新的空路径：

```bash
git clone /mnt/d/central_strip_Panoramic_Camera \
  /home/zyh/src/central_strip_Panoramic_Camera
cd /home/zyh/src/central_strip_Panoramic_Camera
git checkout codex/s013-v11-live-production
git rev-parse HEAD  # 必须等于 Windows 侧刚记录的本阶段提交
```

WSL venv 示例（系统包安装是用户检查点）：

```bash
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
  python3-venv python3-pip build-essential git ripgrep pkg-config \
  libusb-1.0-0-dev libudev-dev libgl1 libglib2.0-0

cd /home/zyh/src/central_strip_Panoramic_Camera
python3 -m venv /home/zyh/venvs/g305-l0
source /home/zyh/venvs/g305-l0/bin/activate
export PIP_NO_CACHE_DIR=1
python -m pip install --upgrade pip setuptools wheel 'cmake>=3.24' ninja
python -m pip install -r requirements/linux-l0-constraints.txt
python -m pip install -e . --no-deps
```

Open3D 构建脚本至少固定：

```text
BUILD_CUDA_MODULE=ON
BUILD_PYTHON_MODULE=ON
CMAKE_CUDA_ARCHITECTURES=120
BUILD_WITH_CUDA_STATIC=ON
BUILD_GUI=OFF
BUILD_WEBRTC=OFF
BUILD_PYTORCH_OPS=OFF
BUILD_TENSORFLOW_OPS=OFF
```

验证：

```bash
G305_CUDA=required python scripts/verify_open3d_cuda.py

G305_CUDA=required python -m pytest -q \
  tests/test_cuda_backend.py \
  tests/test_rgbd_odometry.py \
  tests/test_dense_fusion.py \
  tests/test_video_s13_cuda_runtime.py \
  tests/test_video_s13_m6_cuda.py \
  tests/test_video_cuda_renderer.py
```

接受条件：

- `import open3d` 成功，CPU 和 CUDA binding 都完整；
- `open3d._build_config.BUILD_CUDA_MODULE` 为 true；
- `open3d.core.cuda.is_available()` 为 true；
- Tensor odometry 和 VoxelBlockGrid smoke 均成功；
- 正式真实边报告 `open3d_tensor_cuda_rgbd`；
- TSDF 生成非空 blocks、vertices 和 triangles；
- 现有明确标为 CUDA-required 的边界没有 CPU fallback；ORB、GraphCut、MultiBand 等既有 CPU 算法仍如实报告为 CPU。

建议提交：

```text
build(linux): add the pinned CUDA Open3D 0.19 toolchain
```

### L0.4 Linux headless Gemini 305 采集可移植性

拟新增/修改：

```text
src/panorama_demo/capture_orbbec.py
src/panorama_demo/photo_capture.py
configs/demo.yaml
scripts/setup_orbbec_linux.sh
tests/test_capture_calibration.py
tests/test_photo_capture.py
tests/test_video_live.py
tests/test_video_s13_live.py
tests/test_video_delivery.py
docs/LINUX_NATIVE_CLI.md
```

实现要求：

1. Linux 正式入口以 `--duration`、`--max-frames` 或 `Ctrl+C` 停止，不实现复杂的 POSIX 单键监听。
2. `--no-preview` 路径不得调用 `cv2.imshow()`；只有实际创建 GUI 窗口时才调用 GUI cleanup。
3. 保留缺相机默认持续轮询、插入后自动继续，以及可取消的 `--no-wait-for-camera` 立即失败模式。
4. 把 Linux 下的 Windows metadata registration 提示改为实际 Orbbec SDK/udev 诊断。
5. `scripts/setup_orbbec_linux.sh` 只能调用或安装固定版本 Orbbec 官方规则；不得凭记忆手写 VID/PID、metadata API 或不完整规则。
6. 使用普通用户运行相机程序；不得用 `sudo g305-capture` 掩盖 udev/权限错误。
7. V4L2/LibUVC backend 只根据所装 SDK 的真实配置、官方示例和运行时审计决定，不提前臆造。
8. 保持现有同步、对齐、曝光、metadata、每帧 readback、writer drain 和 cleanup fail-closed 契约。

单元/合成验证：

```bash
python -m pytest -q \
  tests/test_capture_calibration.py \
  tests/test_photo_capture.py \
  tests/test_video_live.py \
  tests/test_video_s13_live.py \
  tests/test_video_delivery.py
```

当前没有相机，因此真机 discovery/profile/metadata 项在这一步必须是 `NOT_EXECUTED/NO_CAMERA`。

建议提交：

```text
fix(capture): support headless native Linux Gemini capture
```

### L0.5 修正文档与平台语义

修改：

```text
README.md
docs/LINUX_NATIVE_CLI.md
docs/SDK.md
AGENTS.md
```

必须写清：

- 五个目标 CLI 的 Linux Bash 示例；
- 无需图形虚拟机，WSL 是 L0–L1 主开发、打包和回放环境，原生 Ubuntu 只作最终 H0 硬件验收；
- current live handoff 是 `validated_inputs_only`，不是 frozen P0/tail-only；
- Open3D/CuPy/ORB 各自的 runtime 和许可边界；
- 录制回放、WSL 证据、原生实机证据的区别；
- L0 是 Linux Runtime/CLI，L1 在同一 WSL 中完成进程内视频 SDK；daemon/service 仍不在范围；
- 实际验收产物路径和代表文件。

建议提交：

```text
docs(linux): document the native CLI and current live authority
```

### L0.6 WSL 录制数据回放与严格等价

**目标**：在现有 WSL 中证明 Linux 用户态、S013、native ORB 和 Open3D CUDA 可处理真实录制 RGB-D。该结果仍不是 Linux 相机实采。

复用 L0.3 已建立的 ext4 checkout；若当时因存储门尚未建立，则按 L0.3 的新目录 clone 流程创建。不得覆盖一个已有 dirty checkout。

先核对 checkout：

```bash
cd /home/zyh/src/central_strip_Panoramic_Camera
git status --short
git rev-parse HEAD  # 必须等于 Windows 侧刚记录的本轮 L0 已验证提交
```

把冻结 session 复制到 ext4 的**全新空目录**，保持源目录只读。三个控制文件摘要用于快速身份检查；随后必须用一次 `diff -qr` 直接逐文件比较整个 RGB/depth session，不能只比较三个控制文件。无需写长期逐帧摘要清单。正式计时不得从 `/mnt/d` 读取。

```bash
WINDOWS_SESSION=/mnt/d/central_strip_Panoramic_Camera/data/captures/video8_25_3_right/run_20260825_123714
LINUX_SESSION=/home/zyh/g305-data/L1-FROZEN-001
test ! -e "$LINUX_SESSION"
mkdir -p /home/zyh/g305-data
cp -a "$WINDOWS_SESSION" "$LINUX_SESSION"
diff -qr "$WINDOWS_SESSION" "$LINUX_SESSION"
```

执行：

```bash
export G305_CUDA=required
L0_ROOT=/home/zyh/l0-artifacts/L1-FROZEN-001

python scripts/run_s013_v11_offline_acceptance.py \
  "$LINUX_SESSION" --output "$L0_ROOT/offline" --maximum-post-seconds 60

python scripts/run_s013_v11_live_replay.py \
  "$LINUX_SESSION" --output "$L0_ROOT/live" \
  --cadence-scale 1.0 --maximum-post-seconds 60

python scripts/verify_s13_v11_live_equivalence.py \
  "$L0_ROOT/offline" "$L0_ROOT/live" \
  --output "$L0_ROOT/linux_live_offline_equivalence.json"

python -m panorama_demo.video_3d_postprocess \
  "$LINUX_SESSION" \
  --two-d-output "$L0_ROOT/offline" \
  --output "$L0_ROOT/offline/3d"
```

另用 `g305-orbslam3-trajectory` 对一份严格真实 RGB-D 会话运行 native ORB，确认不经 WSL wrapper。若冻结视频 post-3D 已经覆盖相同批处理桥，可复用该真实轨迹证据，不重复运行无意义工作。

严格正确性门：

- Linux live replay 与 Linux offline 的正式 P3 字节一致，P0–P3 内存像素 evidence 逐阶段精确一致。
- Windows/Linux 不要求 PNG 容器 SHA 一致，但解码后 P3 像素、source schedule、pair decisions、owner、valid mask、provenance arrays、grade 和 fallback decision 必须精确一致。
- 浮点审计指标只按项目已有契约容差比较；不得为 Linux 新建更宽容差。
- 第一个像素差异出现时立即停止扩大迁移范围，定位到 P0/P1/P2/P3 首个阶段，不得解释成“CUDA 正常误差”。
- 生产输出不得为基准额外落盘 P0/P1/P2 stage PNG。
- native ORB 轨迹真实、完整且无插值；Open3D/TSDF backend 满足 CUDA gate。
- 3D 失败注入只生成 `3d/video_3d_failure.json`，二维 delivery 内容保持不变。

### L0.7 WSL Linux Runtime 五入口验收

**目标**：证明五个入口的 Linux 用户态、依赖和录制数据处理已经完成；不要求 WSL 连接相机。

记录 WSL OS/kernel、Python、GPU/driver、CUDA toolkit、CuPy、OpenCV、Open3D、Orbbec wrapper、ORB executable、Git HEAD/status 和 production lock。

执行：

```bash
export G305_CUDA=required
WSL_ROOT=/home/zyh/l0-artifacts/runtime
VIDEO_SESSION=/home/zyh/g305-data/L1-FROZEN-001
ORB_SESSION=/home/zyh/g305-data/photo/run_20260730_195816_975

g305-capture --help
g305-video-live --help

g305-video-panorama \
  "$VIDEO_SESSION" --output "$WSL_ROOT/offline_2d" \
  --maximum-post-seconds 60 --defer-3d

g305-video-post-3d \
  "$VIDEO_SESSION" \
  --two-d-output "$WSL_ROOT/offline_2d" \
  --output "$WSL_ROOT/offline_2d/3d"

g305-orbslam3-trajectory \
  "$ORB_SESSION" \
  --output "$WSL_ROOT/native_orb_trajectory.json"
```

`g305-video-live` 的正式二维链使用 L0.6 的 live replay runner 验证。`g305-capture` 和真实 live capture 使用 no-device/mock tests 验证参数、等待、取消、writer 和 cleanup 契约；因为当前无相机，USB/profile/metadata 项写 `NOT_EXECUTED/NO_CAMERA`，但不阻塞 `L0_WSL_RUNTIME=PASS`。

接受条件：

- 五个 entry point 可 import，`--help` 返回 0；
- 离线 V11、live replay、native ORB 和 post-3D 真实录制回放通过；
- Linux 运行代码不启动 `wsl.exe` 或做 Windows→WSL path conversion；
- Open3D/CuPy 各自 CUDA-required 边界通过；
- 3D 失败注入不修改二维 delivery；
- 真机未执行项和软件通过项分开记录。

### L0.8 L0 收口验证

按影响范围先跑目标测试，再跑全量：

```bash
python -m pytest -q
ruff check src tests
python -m compileall -q src tests
git diff --check
git status --short
```

还必须运行：

```bash
rg -n "wsl\.exe|wslpath|orbslam3_rgbd_wsl|windows_path_to_wsl" \
  src/panorama_demo/orbslam3_bridge.py \
  src/panorama_demo/online_orbslam3_bridge.py \
  src/panorama_demo/video_online_orb.py \
  src/panorama_demo/export_orbslam3_trajectory.py \
  src/panorama_demo/video_3d_postprocess.py \
  configs/demo.yaml
git diff --exit-code f8f624cee310678c4605eb14764c7bc2509158f9 -- \
  configs/video_candidates/s013/S013_output_first_progressive_dense_central_slit_v4_cuda_m63_visual_continuity_v11.yaml \
  configs/video_algorithms/s013_visual_continuity_v11_production.yaml \
  configs/video_algorithms/s013_visual_continuity_v11_production.lock.json
```

L0 完成需要：代码/测试通过、五个 Linux entry point 的软件契约通过、WSL CUDA/Open3D/ORB 通过、真实录制 2D/3D 回放通过、immutable V11 未变。满足后写 `L0_WSL_RUNTIME=PASS`；原生 Ubuntu 和相机状态仍为 `NOT_EXECUTED`，但不阻塞进入 L1 SDK 开发。

## 7. L1 在 WSL 中完成视频 SDK 和回放性能基线

L1 在现有 WSL Ubuntu 22.04 中完成。它可以新增 SDK facade、typed API、资源打包、wheel、doctor 和隔离 benchmark harness，但不得修改 S013、ORB、Open3D、采集算法或正式输出契约。

### L1.0 SDK 基本完成的定义

`SDK_SOFTWARE_READY` 必须同时表示：

- 五个 Linux CLI 的 Runtime 已通过 L0；
- 新视频 SDK 能处理已有 session、启动采集任务、先交付二维并独立启动 3D；
- SDK 与 CLI 对同一输入使用同一 production lock 和同一底层函数；
- clean wheel 在仓库外 venv 可安装并找到全部正式资源；
- doctor 能区分软件就绪、相机缺席和分发许可状态；
- 真实录制 RGB-D 的 SDK/CLI 二维精确等价，ORB/3D 回放通过；
- WSL 内重复性能和 SDK overhead 已报告；
- 原生相机项明确保留为 `NOT_EXECUTED/WAITING_FOR_H0`。

### L1.1 SDK 修改面

拟新增/修改：

```text
src/panorama_demo/video_sdk.py
src/panorama_demo/sdk_doctor.py
src/panorama_demo/sdk.py
src/panorama_demo/__init__.py
src/panorama_demo/paths.py
src/panorama_demo/config.py
src/panorama_demo/video_pipeline.py
pyproject.toml
docs/SDK.md
docs/LINUX_NATIVE_CLI.md
tests/test_video_sdk.py
tests/test_sdk_doctor.py
tests/test_sdk_wheel_smoke.py
scripts/run_sdk_h0_acceptance.py
tests/test_sdk_h0_acceptance.py
```

既有 `PanoramaSDK` 继续只表示照片产品，默认行为和公开类型不变。新增连续视频 facade 命名为 `Gemini305VideoSDK`，不能把视频语义塞进旧照片方法造成兼容性破坏。

### L1.2 视频 SDK 最小公开接口

```python
sdk = Gemini305VideoSDK(VideoSDKConfig(...))

doctor = sdk.doctor()

result_2d = sdk.process_session(
    session=Path(...),
    output_dir=Path(...),
    defer_3d=True,
)

job = sdk.capture_and_process(
    capture_root=Path(...),
    output_dir=Path(...),
    width=848,
    height=480,
    fps=60,
    duration_seconds=3.0,
    video_exposure_us=100,
    preview_window=False,
)

result_3d = sdk.run_post_3d(
    session=result_2d.session,
    two_d_output=result_2d.output_dir,
)
```

公开类型至少包括：

```text
VideoSDKConfig
Gemini305VideoSDK
VideoProcessingJob
VideoPanoramaResult
VideoThreeDResult
SDKDoctorReport
```

契约要求：

- `process_session()` 直接复用锁定的 `g305-video-panorama` production route；不得复制 renderer。
- `capture_and_process()` 复用 `g305-video-live` 编排，默认等待相机且可取消；WSL 无相机时只执行 mock/no-device 测试。
- `VideoProcessingJob` 只保存当前进程内任务状态和结果路径，不引入 SQLite、daemon、gRPC 或持久任务队列。
- 二维结果一旦发布即可返回/通知；3D 结果单独查询，3D 失败不撤销二维。
- 默认配置固定 V11 production lock，不公开 candidate、baseline fallback、ORB pose 参与二维或算法内部调参。
- 所有返回类型给出真实 artifact path、authority、grade、manual review、timing 和 failure 信息，不把 Preview 当正式 P3。

### L1.3 wheel、资源和 Runtime doctor

当前 `PROJECT_ROOT/configs/...` 只适合源码/editable checkout。L1 必须建立统一 resource resolver，并字节不变地随 wheel 携带正式运行所需的：

```text
configs/demo.yaml
configs/video_algorithms/s013_visual_continuity_v11_production.yaml
configs/video_algorithms/s013_visual_continuity_v11_production.lock.json
configs/video_algorithms/baseline_legacy_fast_b07b561.yaml
configs/video_algorithms/baseline_legacy_fast_b07b561.lock.json
configs/video_candidates/s013/quality_thresholds_m61_v2.json
生产配置引用的 threshold approval witness
离线 Viewer 静态资源
```

不得修改 immutable YAML/lock 来适配机器路径。机器特定 ORB executable、Vocabulary、CUDA 和缓存路径通过 `VideoSDKConfig` 的 site/runtime 配置提供。

`sdk.doctor()` 至少报告：

```text
python
package_resources
production_identity
cupy_cuda
open3d_cuda
orbslam3_external_runtime
orbbec_wrapper
camera_device
host_disk_space
orb_distribution_license
```

相机缺席只使 `camera_device=NOT_EXECUTED/NO_CAMERA`，不使 WSL 软件 doctor 失败。production identity、package resources、CuPy/Open3D required boundary 或 ORB external runtime 失败，则 `SDK_SOFTWARE_READY` 不成立。

wheel 不捆绑 NVIDIA driver、CUDA toolkit、Orbbec 系统规则、ORB binary 或 Vocabulary。生成 `runtime-manifest.json`，记录这些外部依赖的版本、路径和状态。

仓库外 smoke：

```bash
python -m build --outdir /home/zyh/build/wheels
python3 -m venv /home/zyh/venvs/g305-wheel-smoke
source /home/zyh/venvs/g305-wheel-smoke/bin/activate
python -m pip install --no-deps /home/zyh/build/wheels/gemini305_rgbd_panorama-*.whl
python -c "from panorama_demo import Gemini305VideoSDK; print(Gemini305VideoSDK().doctor())"
```

clean venv 必须能加载资源并运行五个 CLI 的 `--help`。CUDA/Open3D/ORB 集成 smoke 使用已准备好的 WSL Runtime；不能用 `--no-deps` 后的缺依赖状态冒充完整 Runtime 通过。

建议分两个提交：

```text
feat(sdk): add the video-first in-process SDK facade
build(sdk): package immutable runtime resources and doctor checks
```

### L1.4 新增基准设施

拟新增：

```text
scripts/run_linux_l1_benchmark.py
scripts/summarize_linux_l1_benchmark.py
tests/test_linux_l1_benchmark.py
docs/LINUX_L1_ACCEPTANCE_TEMPLATE.md
```

脚本职责仅限：

- 调用现有正式 CLI 和 replay runner；
- 每次创建全新输出目录；
- 保存 command、stdout、stderr、exit code 和原始 report/timing；
- 每 100 ms 采样根进程与进程树 RSS；
- 每 200 ms 采样 GPU memory/utilization；
- 提取 capture、Preview、P3、ORB、TSDF 和总 wall time；
- 固定计算 median 和 P95；
- 失败运行保留在样本集合和成功率中，不能删掉后重算；
- 生成 Windows/Linux 正确性、工作量和性能对比；
- 不改变公共 CLI、production config、source selection、像素链、ORB 或 TSDF。

`run_linux_l1_benchmark.py` 固定提供四个 subcommand，Windows 和 Linux 分别在本机运行自己的 suite，不从一个系统远程控制另一个系统：

```text
frozen-2d
  --session PATH --artifact-root PATH --runs 5 --warmups 1
  --entry cli|sdk
  --cadence-scale 1.0 --maximum-post-seconds 60

physical-live
  --artifact-root PATH --runs 5 --warmups 1
  --entry cli|sdk
  --width 848 --height 480 --fps 60 --duration 3
  --video-exposure-us 100 --maximum-post-seconds 60 --no-preview

post-3d
  --session PATH --two-d-output PATH --artifact-root PATH
  --entry cli|sdk
  --runs 5 --warmups 1

orb
  --session PATH --artifact-root PATH --runs 5 --warmups 1
```

`physical-live --entry sdk` 只调用公开 `Gemini305VideoSDK.capture_and_process()`；`scripts/run_sdk_h0_acceptance.py` 是它的薄封装，不得复制 capture、S013 或 3D 编排。每个 subcommand 还必须写 `--platform-label` 和当前 Git HEAD；不得允许调用者伪造 HEAD/status。`summarize_linux_l1_benchmark.py` 固定接收：

```text
--windows-root PATH
--linux-root PATH
--output PATH
```

它读取已有 suite，不重跑产品命令；缺 required suite、输入身份不一致或 workload 不一致时 fail-closed，并保留可审计原因。

统计固定为：

```python
statistics.median(values)
numpy.percentile(values, 95, method="linear")
```

脚本测试至少覆盖：失败样本不被丢弃、5 次以上样本的统计、缺字段 fail-closed、状态/reason_code、RSS/GPU 采样终止、artifact schema 和二维/三维字段分离。

建议提交：

```text
test(perf): add the isolated Linux L1 benchmark harness
```

### L1.5 权威冻结 workload

主 workload 固定为 L0.0 的 `L1-FROZEN-001`：3 秒、152 帧、fixed exposure、right scan。`89 sources / 88 pairs / 1945×480 P3` 是旧输出参考；L0.0 先生成 Windows 回归护栏，L0.6 再冻结 WSL 当前 workload。L1 的 SDK、CLI 和 replay 必须使用同一份 ext4 session 和同一 workload。

正式 WSL SDK/CLI 对比必须使用：

- 相同 session 字节；
- SDK 与 CLI 使用同一个已通过 L0 的 Git commit 和同一个 Python Runtime；
- 相同 production identity；
- `G305_CUDA=required`；
- `cadence_scale=1.0`；
- 相同 source/pair/canvas/owner 工作量；
- WSL 代码、session 和输出均位于 ext4，不从 `/mnt/d` 正式计时；
- 同一台电脑、接通电源、固定 Windows/WSL 电源模式；
- 运行时不并发全景、ORB 或 3D 任务。

若 input、source、pair、canvas 或 owner workload 不一致，停止 SDK overhead 和性能归因，先处理正确性差异。Windows 迁移前数据只作补充对照；它不替代 SDK/CLI 同 Runtime 基线，也不等于原生 Ubuntu 最终性能。

### L1.6 重复次数

WSL 内重复运行拆成必需 suite 和补充 suite：

- 每个 suite 先 1 次 smoke/warm-up，不计入统计；
- **必需**：WSL bare CLI/live replay 二维 5 次；
- **必需**：WSL `Gemini305VideoSDK.process_session()` 二维 5 次；
- **必需**：WSL SDK post-3D 5 次；
- **补充且单独判定**：WSL 独立 native ORB trajectory 5 次；
- **补充**：同一 accepted commit 能继续在 Windows 运行时，增加 Windows 二维对照；迁移前 `f8f624ce...` 只作回归护栏；
- 真实 Gemini 305 capture/live suite 不在 L1 执行，明确留给 H0。

5 次是最低正式样本数。若任一运行失败，保留整个 benchmark ID 并修复原因；修复后创建新的 benchmark ID，不能覆盖原始失败。每个 suite 分别报告 `suite_success_rate`。任一必需 suite 低于 `5/5`，`SDK_SOFTWARE_READY` 不成立；ORB/3D 子项失败不能撤销已经通过的二维 delivery。

“独立 native ORB trajectory 5 次”只表示其重复性能不参与 SDK overhead 汇总；L0 的 native ORB 正确性、完整轨迹和真实 recorded replay 仍是硬门，L1 的 SDK post-3D 也必须实际经过同一 native ORB。它们不能因“补充 suite”字样被降成可跳过项。

### L1.7 必须采集的指标

| 类别 | 必须记录 |
| --- | --- |
| 环境 | OS、kernel、CPU、RAM、GPU、driver、CUDA、CuPy、OpenCV、Open3D、Orbbec SDK/wrapper、ORB binary、Git HEAD/status、production lock |
| SDK | wheel/version、resource resolver、doctor、API mode、job state、SDK call overhead、result/failure type |
| Recorded capture | 冻结 manifest 的 profile、physical seconds、received、written、drops、write errors、clean shutdown、eligibility；不冒充本轮 WSL 实采 |
| Preview | first preview seconds、每次运行的 frame latency P95、updates、skipped、failures、authority 标记 |
| P3 | `capture_stop_to_p3_memory_seconds`、`capture_stop_to_p3_published_seconds`、`primary_post_capture_seconds`、各 stage、source/pair/canvas |
| Authority | algorithm/implementation/config、fallback、P0–P3 evidence、owner/valid mask、invalid owner、seam/M5、C2E、M6.3、B0/B1 |
| ORB | tracked frame IDs/count、完整真实 chain、seconds、attempts、interpolation/extrapolation count |
| TSDF | backend、CUDA device、fallback、integration seconds、frame count、capacity、mesh/GLB |
| 总时间 | process start→2D delivery、post-3D process wall、顺序执行→3D delivery |
| RAM | 二维根进程 peak RSS、二维/三维进程树 peak RSS |
| GPU | idle memory、global peak、idle-adjusted peak、可用时的 process peak、utilization median/P95、内部 CUDA pool/device audit |

二维用户等待时间以 `capture_stop_to_p3_published_seconds` 为准，不能用更短的 `primary_post_capture_seconds` 替代。SDK 还要报告从公开方法调用到 typed result 返回的 wall time，并与同 Runtime bare CLI 分开。ORB、TSDF 和 3D 总时间独立报告，不能揉成“P3 时间”。真实 Capture 性能写 `NOT_EXECUTED/WAITING_FOR_H0`。

### L1.8 正确性硬门

每次 WSL CLI 和 SDK 正式二维运行都必须满足：

- production identity 精确，fallback 为 0；
- decoded P3、owner、valid mask、provenance 和 decisions 满足第 L0.6 节等价要求；
- invalid owner pixels 为 0；
- 生产模式不额外落 P0/P1/P2 PNG；
- session/write/capture 契约无错误；
- Preview failure 被隔离，不改变正式 P3/delivery；
- 若本次另执行 post-3D，3D 失败隔离不改变 2D。
- SDK typed result 指向实际 delivery/report/timing/provenance，不复制或伪造状态；
- SDK 与 CLI decoded P3、owner、valid mask、provenance、grade、fallback 和 decisions 精确一致；
- SDK wheel 与 editable checkout 加载的 production identity 完全一致。

任何正确性硬门失败时，不再用该组时间宣称平台性能。

post-3D 和独立 ORB suite 单独要求：native ORB 无缺帧/插值/伪 pose；每个现有 CUDA-required 边界无 CPU fallback；GLB/Viewer delivery 完整。SDK 必须保留相同的 3D 失败隔离。ORB 或 3D 失败使对应子项 `FAIL`，但二维正确性仍按自身证据判定。

### L1.9 性能判定

正式产品已有 `maximum_post_seconds=60`，本次迁移不得为了通过而修改它。当前代码用 `primary_post_capture_seconds` 判定该正式 SLA；超时可以按现有契约结构化发布并降级/要求人工复核，但 L1 性能项应判失败。`capture_stop_to_p3_published_seconds` 是更接近用户真实等待的端到端字段，L1 另外对它设置 60 秒迁移门。两者必须同时报告，不能互相替代。

#### 硬门

- 每次 `primary_post_capture_seconds <= 60 s`，满足现有正式 SLA；
- 每次 `capture_stop_to_p3_published_seconds <= 60 s`，满足 L1 端到端用户等待门；
- 三个必需 suite 各自 5/5 运行成功；
- 冻结 session 的 capture drops/write errors 为 0，received == written；
- SDK/CLI 2D fallback 全为 0；
- 二维 peak RSS 不超过现有 production `2,000,000,000` bytes gate；
- 执行 3D 时，TSDF 满足现有 `2,500,000,000` bytes GPU byte budget/capacity preflight，且现有 CUDA-required 边界不 fallback。

SDK 相对同 Runtime bare CLI 的公开调用到 typed result 返回的 median overhead：

| overhead | 结论 |
| --- | --- |
| ≤ 5% | `PASS` |
| > 5% 且 ≤ 10% | `PASS_WITH_WARNING`，定位 SDK 编排/序列化开销 |
| > 10% | `FAIL`，`SDK_SOFTWARE_READY` 不成立 |

Preview failure 按非权威隔离契约处理：单次失败记录 `preview_run=FAIL_ISOLATED`，正式二维 delivery 仍可 `PASS`。五次内既有有效 Preview 又有隔离失败、且所有权威硬门通过时，聚合 `preview_acceptance=PASS_WITH_WARNING`，L1 overall 最多为 `PASS_WITH_WARNING`；五次均无可用 Preview 更新时才写 `preview_acceptance=FAIL/NO_PREVIEW_EVIDENCE`，此时 `SDK_SOFTWARE_READY` 不成立。

#### 当前 3 秒录制回放 workload 的观察目标

以下是观察目标，不替代 60 秒正式硬门，也不能由一条旧运行直接升级为产品契约：

| 指标 | 观察目标 |
| --- | ---: |
| 首次 Preview | ≤ 2.0 s |
| 单次运行 Preview frame latency P95 | ≤ 25 ms |
| 停采到 P3 published median | ≤ 15 s |
| 停采到 P3 published P95 | ≤ 16 s |

达到硬门但未达到观察目标时可记 `PASS_WITH_WARNING`，并原样报告数值。特别是其它现有 3 秒 session 曾出现约 31 ms 的 Preview P95，所以不能把 `25 ms` 当作未经重新冻结的 fail-closed 产品契约。回放目标不替代 H0 真机 Preview/Capture/P3。

#### 补充的 WSL 相对迁移前 Windows 对照

仅在同一 accepted commit 仍能在 Windows 和 WSL 执行、冻结输入和工作量完全一致时比较：

| Linux 相对 Windows 的同 workload median | 结论 |
| --- | --- |
| 回退 ≤ 5% | `PASS` |
| 回退 > 5% 且 ≤ 10% | `PASS_WITH_WARNING`，必须定位 stage |
| 回退 > 10% | `FAIL`（migration performance） |

同时报告 P95，不用最快一次代替 median。二维 P3、ORB 和 TSDF 分开判定。当前 Windows Open3D 损坏只阻塞 Windows TSDF/Open3D 对照；任一未执行项都不得拿旧提交时间补齐。该对照用于发现 WSL 明显回退，不作为原生 Ubuntu 最终性能结论。

### L1.10 `SDK_SOFTWARE_READY` 状态

L1 的 `acceptance_status.json` 至少包含：

```text
l0_wsl_runtime
sdk_public_api
wheel_and_resources
sdk_doctor
sdk_cli_exact_equivalence
wsl_recorded_replay_2d
wsl_native_orb
wsl_post_3d
sdk_overhead
memory_and_gpu_budget
orb_distribution_license
native_camera_acceptance
overall_software
```

每个状态项必须是带 `status`、`reason_code`、`evidence_paths`、`required_for_software_ready` 和 `required_for_native_final` 的对象，不能只写一个模糊布尔值。`native_camera_acceptance` 在此固定为 `NOT_EXECUTED/WAITING_FOR_H0`，其两个 required 字段分别为 `false` 和 `true`；L0 Runtime、SDK API、wheel/resources、doctor 软件项、SDK/CLI 等价、WSL recorded replay、native ORB、post-3D 和资源门则必须 `required_for_software_ready=true`。其余软件必需项通过后，写：

```text
overall_software = PASS
milestone = SDK_SOFTWARE_READY
hardware_qualified = false
```

ORB 商业分发许可未定时可继续得到内部开发用 `SDK_SOFTWARE_READY`，但 `release_ready=false`，wheel 不包含 ORB binary/Vocabulary。不得把该里程碑写成“原生 Ubuntu 真机通过”或“可直接商业分发”。

### L1.11 L1 收口验证

```bash
python -m pytest -q \
  tests/test_video_sdk.py \
  tests/test_sdk_doctor.py \
  tests/test_sdk_wheel_smoke.py \
  tests/test_sdk_h0_acceptance.py \
  tests/test_linux_l1_benchmark.py \
  tests/test_video_s13_v11_production_route.py \
  tests/test_video_s13_live_acceptance.py \
  tests/test_video_delivery.py \
  tests/test_video_3d_postprocess.py

ruff check src tests \
  scripts/run_linux_l1_benchmark.py \
  scripts/summarize_linux_l1_benchmark.py \
  scripts/run_sdk_h0_acceptance.py

python -m compileall -q src tests \
  scripts/run_linux_l1_benchmark.py \
  scripts/summarize_linux_l1_benchmark.py \
  scripts/run_sdk_h0_acceptance.py

python -m pytest -q
git diff --check
git status --short
```

还必须在仓库外 clean venv 安装最终 wheel，运行 doctor、五 CLI `--help`、SDK/CLI 精确回放和一轮 post-3D smoke。没有 wheel、runtime manifest、artifact root、原始失败样本、`summary.json` 和 `acceptance_status.json`，不能写 `SDK_SOFTWARE_READY`。

## 8. H0：SDK 基本完成后的原生 Ubuntu 最终相机验收

H0 只在 L1 已经产生 `SDK_SOFTWARE_READY` artifact 后开始。优先借用与开发基线一致的原生 Ubuntu 22.04 LTS x86_64；若使用 24.04，则把它明确记为额外兼容目标，不能把 24.04 结果反写成 22.04 已验收。无桌面环境可通过 SSH 执行。验收机只安装冻结 wheel 和受控 external Runtime，不直接编辑源码。代码/打包缺陷回 WSL 生成新候选；机器环境缺陷保持同一 wheel 修复后重测；旧 H0 结果始终保留。

### H0.0 原生验收机回放与 Runtime 预检

插相机前必须先完成：

1. 安装 L1 产出的**同一 wheel SHA**，核对 Git commit、production YAML/lock、package resources 和 Runtime 配方。
2. 在原生机 ext4 上放置 `L1-FROZEN-001` 的同一字节副本，分别运行一次 bare CLI 2D、SDK `process_session()` 和 SDK post-3D。
3. 原生 CLI/SDK 的 decoded P3、owner、valid mask、provenance、grade、fallback 和 decisions 必须与 WSL 软件基线精确一致；ORB/GLB/Viewer 必须满足相同结构契约。首个差异未定位前不得进入真机采集。
4. 保存 native doctor、依赖清单、回放命令、report/timing 和对比报告；这组结果写入 H0 的 `native_replay/`，不能混入五次 physical run。

GPU 处理采用“同 SDK、受控 Runtime variant”：

- 若原生机 GPU 架构与 L1 Runtime 已覆盖范围兼容，按冻结 manifest 安装现成 CuPy/Open3D Runtime。
- 若借用机器使用不同 compute capability，允许只把 `CMAKE_CUDA_ARCHITECTURES` 改为该 GPU 的受支持目标；Open3D `0.19` 来源、CuPy/CUDA 版本和其余 build flags 必须沿用 L1 冻结配方。不得修改 SDK wheel、production config、S013 或降级 `G305_CUDA=required`；目标架构无法构建或 CUDA smoke 失败时，H0 Runtime 预检直接失败。
- 新建 `runtime_variant_id`，记录 GPU、driver、CUDA、CuPy、Open3D commit/build flags 和 wheel SHA。此时 H0 合格只覆盖“该 wheel + 该 Runtime variant + 该原生机器”，不得宣称 WSL external binary 与原生 binary 字节相同。

### H0.1 Gemini 305 SDK 五次真机与 CLI 交叉验收

`SDK_HARDWARE_QUALIFIED` 的主证据必须来自公开 SDK，而不是只由底层 CLI 代替。先执行五次 SDK physical-live；一次 warm-up 不计入五次正式样本，最后一个成功 session 再通过公开 SDK 运行独立 post-3D：

```bash
export G305_CUDA=required
H0_RUN_ID=H0_YYYYMMDDTHHMMSS_SHORTCOMMIT
H0_ROOT="/var/tmp/g305-h0/$H0_RUN_ID"
mkdir -p "$H0_ROOT"

python scripts/run_sdk_h0_acceptance.py \
  --artifact-root "$H0_ROOT/sdk_physical" \
  --runs 5 --warmups 1 \
  --width 848 --height 480 --fps 60 --duration 3 \
  --video-exposure-us 100 --maximum-post-seconds 60 \
  --no-preview --post-3d-last
```

该脚本只调用 `Gemini305VideoSDK.capture_and_process()` 和 `run_post_3d()`，必须创建 `run_01` 到 `run_05` 的独立 session/output，并保留失败样本。随后用同一固定参数执行一次 `g305-video-live` 交叉检查，证明 CLI 和 SDK 共用正式编排：

```bash
CLI_ROOT="$H0_ROOT/cli_crosscheck"
g305-video-live \
  --width 848 --height 480 --fps 60 \
  --duration 3 --video-exposure-us 100 --no-preview \
  --output "$CLI_ROOT/sessions" \
  --panorama-output "$CLI_ROOT/2d" \
  --maximum-post-seconds 60
```

`g305-capture` 也属于 L0 的目标入口，因此 H0 还必须各运行一次 continuous fixed exposure、continuous auto exposure 和照片模式；它们是独立采集契约证据，不冒充五次 SDK P3：

```bash
g305-capture \
  --width 848 --height 480 --fps 60 --duration 3 \
  --video-exposure-us 100 --no-preview \
  --output "$H0_ROOT/capture_cli/video_fixed"

g305-capture \
  --width 848 --height 480 --fps 60 --duration 3 --no-preview \
  --output "$H0_ROOT/capture_cli/video_auto"

g305-capture \
  --photo-mode --width 848 --height 480 --fps 60 --max-frames 10 \
  --output "$H0_ROOT/capture_cli/photo"
```

对 SDK/live 路径，`--no-preview` 只关闭本地图形窗口，不能关闭内部非权威 S13 Preview 的生成和计时；独立 `g305-capture` 不产生 S013 Preview。

现场固定：同一相机、同一 USB 3.x 端口、线缆、供电、场景、照明、起终点和 right 单向扫描。优先使用可重复运动平台；若只能手持，保留全部 5 次，不能挑选最快会话。若 Windows/Linux 真机 session 的位移、source、pair 或 canvas 中位工作量差异超过 5%，真机时间只能作为现场可用性证据，不能用于平台速度归因。

H0 始终执行 60 秒正式 SLA、0 drops/errors、clean shutdown、资源预算和 CUDA-required 绝对硬门。只有在同一台或硬件等价机器、相同输入和相同 workload 下，才应用“相对回退 ≤5% / 5–10% warning / >10% fail”的平台比较；借用机硬件不同则只报告原始 median/P95，不据此给 WSL 或 Windows 做优化/回退归因。

此外必须单独覆盖：

- 相机缺席时持续轮询；插入后自动继续；Ctrl+C 可取消；
- fixed exposure 和 auto exposure 至少各一组；
- non-root discovery、udev 和 metadata；
- sync 每帧 readback；
- cleanup gate/readback；
- 3D 失败隔离。

### H0.2 原生硬件结果状态

H0 的 `native_acceptance_status.json` 至少包含：

```text
wheel_and_runtime_identity
native_replay_equivalence
non_root_camera_discovery
udev_and_usb3
exact_848x480_60_profiles
fixed_exposure_capture
auto_exposure_capture
standalone_video_capture_cli
photo_capture_cli
sync_and_metadata_readback
clean_shutdown_and_writer
preview_acceptance
formal_p3_delivery
sdk_capture_api
video_live_cli_crosscheck
post_3d_and_failure_isolation
native_performance
overall_hardware
```

这些项目同样使用结构化状态对象；软件身份和 native replay 同时 `required_for_software_ready=false`、`required_for_native_final=true`，所有真机项只 `required_for_native_final=true`。H0 不重写 L1 的 `acceptance_status.json`，只在新的 native 状态文件中引用其 artifact ID。

H0 全部硬门通过后写：

```text
overall_hardware = PASS
milestone = SDK_HARDWARE_QUALIFIED
```

没有原生机器或相机时保持 `NOT_EXECUTED/WAITING_FOR_H0`，不撤销 `SDK_SOFTWARE_READY`。H0 失败时保持软件里程碑，但 `hardware_qualified=false`。若失败来自代码、wheel 或 package resources，回 WSL 修复、重跑全部 software required suite 并生成新 wheel；若失败来自 udev、驱动、USB、线缆、供电或 native Runtime 构建，则保持同一 wheel，修复原生环境并用新的 H0 ID 重测。两类情况都保留原失败目录。

### H0.3 原生验收收口

- 验收机安装的 wheel SHA、Git commit 和 production lock 必须与 WSL 交付包一致。
- external Runtime 必须匹配冻结版本/构建配方；若因 GPU 架构生成 native variant，必须有独立 `runtime_variant_id` 和 native manifest。
- 相机接入前，原生冻结回放的 CLI/SDK 精确二维等价及 post-3D 结构契约必须通过。
- `sdk.doctor()` 的 camera、udev/USB、CuPy、Open3D、ORB 和 production identity 均通过。
- 五次 `Gemini305VideoSDK.capture_and_process()` physical-live、最后一次成功 session 的 SDK post-3D，以及一次 `g305-video-live` CLI 交叉检查通过。
- `g305-capture` 的 continuous fixed、continuous auto 和 photo mode 实机契约分别通过。
- 固定曝光、自动曝光、缺相机轮询/插入继续、Ctrl+C、cleanup readback 和 3D 失败隔离都有独立证据。
- 保留五次完整原始 report/timing，不只保留最快一次。
- 运行最终 smoke/target tests、`git diff --check`（若有 source checkout）和 artifact schema 校验。

## 9. 验收矩阵

| 项目 | 环境 | 输入 | 硬门 | 当前能否执行 |
| --- | --- | --- | --- | --- |
| 迁移前 Windows 2D 基线 | Windows | 冻结 3 秒 session | live/offline 精确等价 | 能 |
| 单元/合成测试 | WSL 22.04 | 合成 | 全通过 | 安装 venv 后能 |
| V11 identity | WSL | packaged config | identity/SHA/fallback 精确 | 能 |
| Linux 2D offline/live replay | WSL | 冻结真实 RGB-D | decoded P3/owner/provenance 精确 | CUDA 环境完成后能 |
| direct Linux ELF ORB | WSL | 真实录制 RGB-D | 真 pose、完整、无 wrapper | ORB 固化后能 |
| Open3D CUDA | WSL | smoke/真实边 | CUDA 可用、backend 精确 | **可按当前 34.6 GiB 预算立即实施并监测** |
| post-3D | WSL | 真实录制 RGB-D | delivery/GLB/Viewer、2D 保留 | Open3D 完成后能 |
| 视频 SDK API | WSL | 合成/真实录制 | typed API 与 CLI 同契约 | L0 后执行 |
| wheel/resources | WSL clean venv | wheel | repo 外加载全部正式资源 | L1 执行 |
| SDK doctor | WSL | Runtime | software pass、camera 未执行 | L1 执行 |
| SDK/CLI exact parity | WSL | 同一冻结 session | P3/owner/provenance/decision 精确 | L1 执行 |
| SDK overhead | WSL | 同一冻结 session | 1 warm-up + 5/5，median ≤5% 目标 | L1 执行 |
| `SDK_SOFTWARE_READY` | WSL | L0–L1 全证据 | software required 项全通过 | 可直接推进；以实际构建结果为准 |
| 原生冻结回放预检 | 原生 22.04 优先 | 同一冻结 session | CLI/SDK 精确二维等价、post-3D 结构通过 | H0，`NOT_EXECUTED` |
| `g305-capture` 真机 | 原生 22.04 优先 | Gemini 305 | fixed/auto/photo、profile/sync/clean 均通过 | H0，`NOT_EXECUTED` |
| non-root camera/udev | 原生 22.04 优先 | Gemini 305 | 普通用户发现 | H0，`NOT_EXECUTED` |
| 60 FPS capture/sync | 原生 22.04 优先 | Gemini 305 | 0 drops/errors、clean | H0，`NOT_EXECUTED` |
| live Preview/P3/SDK API | 原生 22.04 优先 | Gemini 305 | Preview 非权威、P3 正式 | H0，`NOT_EXECUTED` |
| `SDK_HARDWARE_QUALIFIED` | 原生 Ubuntu | 冻结 wheel + 真机 | H0 全部硬门通过 | `NOT_EXECUTED` |
| ORB 商业分发 | 许可决策 | 外部 runtime | GPL/商业许可明确 | `BLOCKED`，不阻塞内部 `SDK_SOFTWARE_READY` |

## 10. 产物目录

重型产物、raw session、wheel 和构建缓存不提交 Git。每轮使用新的根目录：

```text
<L0_L1_ARTIFACT_ROOT>/SDK_SOFTWARE_<timestamp>_<short_commit>/
├── benchmark_manifest.json
├── host_inventory.json
├── wsl_inventory.json
├── dependency_manifest.json
├── runtime-manifest.json
├── frozen_session.json
├── wheel/
├── wheel_smoke/
├── sdk_doctor.json
├── tests/
├── windows/
│   ├── environment.json
│   └── migration_guard/
├── wsl/
│   ├── environment.json
│   ├── cli_2d/run_01 ... run_05/
│   ├── sdk_2d/run_01 ... run_05/
│   ├── sdk_post_3d/run_01 ... run_05/
│   ├── native_orb/
│   └── open3d_cuda/
└── comparison/
    ├── runs.csv
    ├── summary.json
    ├── equivalence.json
    ├── acceptance_status.json
    └── acceptance.md
```

H0 使用另一个根，不回写软件基线：

```text
<H0_ARTIFACT_ROOT>/SDK_HARDWARE_<timestamp>_<wheel_sha>/
├── frozen_software_identity.json
├── native_environment.json
├── native_runtime_manifest.json
├── native_replay/
│   ├── cli_2d/
│   ├── sdk_2d/
│   ├── sdk_post_3d/
│   └── equivalence.json
├── sdk_doctor.json
├── sdk_physical/run_01 ... run_05/
├── cli_crosscheck/
├── capture_cli/
│   ├── video_fixed/
│   ├── video_auto/
│   └── photo/
├── failure_injection/
├── native_acceptance_status.json
└── acceptance.md
```

每次运行至少保留：

```text
command.json
stdout.log
stderr.log
exit.json
process_rss.csv
gpu.csv
metrics.json
2d/video_delivery.json
2d/video_report.json
2d/video_timing.json
2d/video_panorama.png
2d/video_pixel_provenance.npz
2d/3d/video_3d_delivery.json（执行 3D 时）
2d/3d/video_3d_timing.json（执行 3D 时）
```

L1 交付说明必须给出软件 artifact root、wheel、Runtime manifest、doctor、WSL CLI/SDK 子目录，以及代表性的 P3、trajectory、desktop/mobile GLB 和离线 Viewer。H0 完成后再追加硬件 artifact root 和五次真机目录。不要只写“测试通过”而不写产物位置。

## 11. 分阶段提交建议

每个提交只包含对应范围，未通过对应测试前不提交。

1. `build(linux): make the G305 ORB-SLAM3 runner reproducible`
2. `refactor(orb): launch ORB-SLAM3 as a native Linux process`
3. `build(linux): add the pinned CUDA Open3D 0.19 toolchain`
4. `fix(capture): support headless native Linux Gemini capture`
5. `docs(linux): document the native CLI and current live authority`
6. `feat(sdk): add the video-first in-process SDK facade`
7. `build(sdk): package immutable runtime resources and doctor checks`
8. `test(perf): add the isolated WSL SDK benchmark harness`
9. `docs(sdk): publish the software-ready and H0 acceptance handoff`

每次提交前：

```bash
git diff --check
git status --short
# 逐个列出本阶段文件；下面以 L1 harness 阶段为例。
git add -- scripts/run_linux_l1_benchmark.py \
  scripts/summarize_linux_l1_benchmark.py \
  tests/test_linux_l1_benchmark.py \
  docs/LINUX_L1_ACCEPTANCE_TEMPLATE.md
git diff --cached --check
```

不得提交：raw RGB-D、生成全景、GLB、benchmark CSV/log、Open3D/ORB build tree、venv、CUDA toolkit、现成 ORB binary 或 Vocabulary。

## 12. 最终停止条件与完成定义

### 12.1 Codex 必须停止并请求用户处理

- 本轮默认不修复 E 盘、不迁移 WSL、不改分区、不安装原生 Ubuntu；若后来确需执行这些动作，必须先经过用户检查点；
- WSL CUDA/Open3D 构建期间，Windows 宿主 D 盘实际可用空间低于 `10 GiB`、出现 `ENOSPC`，或下一已知步骤明确无法容纳；此时停止重型步骤、保留日志和既有文件，不得自动删除或迁移；
- ORB GPL/商业许可边界会影响需要提交或分发的文件；
- 只有进入 H0 后，Gemini 305、原生 Ubuntu、USB 端口/线缆/供电缺失才阻塞硬件项；不得反向阻塞 L0–L1 软件工作；
- immutable S013 V11 identity 发生变化；
- WSL SDK/CLI 或 WSL/Windows 补充对照的首个像素差异尚未定位；
- 为通过测试必须放宽 fail-closed、启用 fallback、降低 CUDA required 或伪造 ORB pose。

### 12.2 L0 完成

只有同时满足以下条件才写 `L0_WSL_RUNTIME=PASS`：

- 五个目标 CLI 均能在 WSL 中作为 Linux process import/启动，运行代码不调用 `wsl.exe`；
- native ORB 可从固定来源重建；
- CuPy 和 Open3D 0.19 CUDA 验证通过；
- WSL 真实录制 2D/ORB/3D 回放通过；
- 全量测试/ruff/compileall/diff check 通过；
- S013 V11 immutable 文件未变。

相机 USB/profile/metadata 仍为 `NOT_EXECUTED/WAITING_FOR_H0`，不影响此状态。

### 12.3 L1 完成

只有同时满足以下条件才写 `L1_SDK=PASS` 和 `SDK_SOFTWARE_READY`：

- L0 WSL Runtime 已通过；
- 视频 SDK API、typed results 和 2D→3D 隔离契约通过；
- clean wheel、正式资源、Runtime manifest 和 doctor 通过；
- WSL SDK/CLI 冻结输入正确性和精确二维等价通过；
- 三个必需 suite（WSL CLI 2D、WSL SDK 2D、WSL SDK post-3D）各自成功率为 5/5；
- SDK overhead、P3、ORB、TSDF、RAM 和 GPU 分开报告并满足 L1 门；
- 60 秒、authority、resource 和 failure-isolation 软件门通过；
- H0 输入包、命令和验收模板完整；
- 未达到的观察目标写为 warning，不美化。

### 12.4 H0 完成

只有冻结的 L1 wheel 与受控 Runtime variant 先通过原生冻结回放，再在原生 Ubuntu 上完成 non-root Gemini 305、USB 3.x、精确 60 FPS、同步/metadata、SDK 公共入口五次 Preview/Capture/P3、SDK post-3D、CLI 交叉检查，以及 `g305-capture` fixed/auto/photo 契约后，才写：

```text
H0_NATIVE_CAMERA = PASS
SDK_HARDWARE_QUALIFIED = true
```

H0 未执行不撤销 `SDK_SOFTWARE_READY`；H0 失败也不能篡改旧软件证据。代码、打包或资源缺陷必须回 WSL 生成新软件候选并重跑 L0–L1；机器环境缺陷使用同一 wheel 修复并以新 H0 ID 重测。

H0 之后仍可继续 systemd、Task Manager、SQLite、Control Lease、gRPC、共享内存预览和 `.deb` 等服务化/部署工作；这些不是本轮 `SDK_SOFTWARE_READY` 的组成部分。商业发布前仍必须明确 ORB GPL/商业许可。

## 13. 官方参考

- [Ubuntu release cycle](https://ubuntu.com/about/release-cycle)：Ubuntu 22.04 和 24.04 生命周期。
- [Orbbec SDK v2](https://github.com/orbbec/OrbbecSDK_v2)：Gemini 305、Linux x64 和 Ubuntu 20.04/22.04/24.04 支持范围。
- [pyorbbecsdk2 installation](https://orbbec.github.io/pyorbbecsdk/source/2_installation/install_the_package.html)：Linux venv、Python 包和官方环境/udev 配置。
- [Microsoft WSL USB](https://learn.microsoft.com/en-us/windows/wsl/connect-usb)：WSL USB 依赖 `usbipd-win`，只作开发实验。
- [NVIDIA CUDA on WSL](https://docs.nvidia.com/cuda/wsl-user-guide/)：WSL 使用 Windows 主机驱动，只安装 WSL CUDA toolkit。
- [Open3D 0.19 build from source](https://www.open3d.org/docs/0.19.0/compilation.html)：Open3D 源码构建依据。
- [ORB-SLAM3 license](https://github.com/UZ-SLAMLab/ORB_SLAM3#license)：GPLv3 与商业许可渠道。
