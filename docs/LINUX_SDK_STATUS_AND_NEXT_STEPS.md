# Gemini 305 Linux SDK 打包现状与后续实施方案

> 审计日期：2026-09-07（Asia/Shanghai）
> Windows 工作区：`D:\central_strip_Panoramic_Camera`
> 当前分支：`codex/s013-v11-live-production`
> 当前 HEAD：`da9687fc7c3e6cbb9ea7ec98bb1d85eebb7e4b21`
> 本文目的：说明 Linux SDK 已经完成到哪一步、哪些结论已有证据、哪些仍未完成，以及下一阶段应按什么顺序实施。

## 1. 结论先行

当前已经完成了一个经过 WSL 软件验收的 **Linux Python SDK 内部候选**：

- Linux 原生 CLI、Linux ELF ORB-SLAM3、CuPy、Open3D 0.19 CUDA、视频 SDK facade、wheel 资源打包、Runtime doctor、重复回放和 SDK/CLI 精确等价均已有实现；
- 提交 `da9687fc` 对应的 `0.2.0` wheel 已在仓库外 clean venv 中通过验收，正式状态为 `SDK_SOFTWARE_READY`；
- 该状态证明“当前 WSL Runtime 上的软件、wheel 和真实录制数据回放可用”，不代表原生 Ubuntu 相机已经验收，也不代表可直接交给普通用户或商业发布。

当前还不能称为“Linux SDK 正式版”，原因有五项：

1. 工作树还有四个 USB/IP/相机启动恢复相关文件未提交，已验收的 `da9687fc` wheel 不包含这些修复；
2. ORB-SLAM3 external Runtime 仍引用构建目录和系统级 Pangolin 动态库，尚未证明可搬到另一台干净机器；
3. 普通 `pip install`、自建 Open3D CUDA wheel、Orbbec wrapper 和项目 wheel 还没有统一为一个可靠安装入口；
4. 原生 Ubuntu H0 尚未执行，不能写 `SDK_HARDWARE_QUALIFIED`；
5. ORB-SLAM3 商业再分发许可仍为 `BLOCKED/ORB_LICENSE`，所以 `release_ready=false`。

因此，最准确的当前定位是：

```text
L0 Linux Runtime                 PASS（仅限 da9687fc 冻结候选）
L1 SDK / thin wheel              PASS（仅限 da9687fc 冻结候选）
当前工作树                       REQUALIFICATION_REQUIRED
完整可搬运 Linux Runtime bundle  INCOMPLETE
原生 Ubuntu H0                  NOT_EXECUTED / WAITING_FOR_H0
正式发布                         BLOCKED_BY_H0_AND_ORB_LICENSE
```

下一步不应直接拿旧 wheel 去做最终发布。应先收口当前补丁和打包基础设施，再产生新的 release candidate，重新签发 `SDK_SOFTWARE_READY`，然后进入原生 Ubuntu H0。

## 2. 状态定义和证据边界

本文使用以下里程碑，避免把不同层次的“通过”混在一起。

| 里程碑 | 含义 | 当前状态 |
| --- | --- | --- |
| `L0_WSL_RUNTIME` | 五个 Linux CLI、CuPy、Open3D CUDA、native ORB 和 recorded replay 在 WSL 运行 | `PASS`，仅绑定 `da9687fc`；当前 capture 改动需重签 |
| `SDK_SOFTWARE_READY` | 冻结 wheel、资源、doctor、SDK/CLI 等价、重复回放和 post-3D 软件验收通过 | `PASS`，仅绑定 `da9687fc` 和对应 wheel |
| `WSL_USBIP_SUPPLEMENTAL` | WSL 经 USB/IP 使用真实 Gemini 305 的补充证据 | 已有单次完整成功；连续稳定性尚未通过 5/5 |
| `SDK_HARDWARE_QUALIFIED` | 同一 wheel 在原生 Ubuntu 上完成 non-root、USB、60 FPS、同步、五次 SDK、CLI 和 3D 验收 | `NOT_EXECUTED` |
| `SDK_RELEASE_READY` | 软件、原生硬件、安装包、支持范围、许可证和发布身份全部关闭 | `false` |

必须始终保留以下边界：

- WSL recorded replay 是 Linux 软件证据，不是新相机采集证据；
- WSL USB/IP 真机成功不是原生 Ubuntu 的 udev、USB 或 60 FPS 资格；
- 单元测试、mock 和单次成功不是五次连续真机验收；
- `delivery.json`/`video_delivery.json`、真实 ORB trajectory、GLB 和 Viewer 必须分别按各自正式契约判断；
- 已发布二维不能因 post-3D 失败被撤销；
- 视频二维始终使用锁定 S013 V11 ignore-pose authority，不能在二维阶段运行或消费 ORB pose；
- 没有当前 commit、wheel SHA 和 Runtime variant 绑定的证据，不能继承旧候选的资格。

## 3. 当前仓库和候选包快照

### 3.1 Git 状态

本次审计读取到：

```text
branch: codex/s013-v11-live-production
HEAD:   da9687fc7c3e6cbb9ea7ec98bb1d85eebb7e4b21
origin/codex/s013-v11-live-production: f8f624cee310678c4605eb14764c7bc2509158f9
local commits ahead of origin: 19
```

创建本文后的当前工作树如下；最后一项就是本次用户要求的新状态文档：

```text
 M src/panorama_demo/capture_orbbec.py
 M src/panorama_demo/video_sdk.py
 M tests/test_capture_calibration.py
 M tests/test_video_sdk.py
?? docs/LINUX_L0_L1_CODEX_IMPLEMENTATION_PLAN.md
?? docs/LINUX_SDK_STATUS_AND_NEXT_STEPS.md
```

四个修改文件实现的是正式帧开始前的 USB/IP/相机 transport、control 和 warm-up 恢复；正式帧开始后的失败仍保持 fail-closed。它们目前只属于工作树，不能说已经进入冻结 wheel。

旧的 `docs/LINUX_L0_L1_CODEX_IMPLEMENTATION_PLAN.md` 是 2026-08-25、`f8f624ce` 基线的实施蓝图。它仍可用于查看证据边界和 H0 流程，但其中把 Linux Runtime、视频 SDK 和 wheel 写为“待实现”的章节已经过时。本文是当前状态文档，不覆盖旧蓝图。

### 3.2 已验收 wheel

当前主机上仍存在下列软件验收包：

```text
/home/zyh/l1-artifacts/SDK_SOFTWARE_20260826_da9687fc/
```

关键身份：

| 项目 | 当前值 |
| --- | --- |
| Git commit | `da9687fc7c3e6cbb9ea7ec98bb1d85eebb7e4b21` |
| SDK version | `0.2.0` |
| Wheel | `gemini305_rgbd_panorama-0.2.0-py3-none-any.whl` |
| Wheel size | `1,775,516 bytes` |
| Wheel SHA-256 | `dfdf7c064745bd671c8ab30bda292319a22a25d1150a024bc61ac9677a40638f` |
| Acceptance schema | `gemini305-linux-l1-acceptance/v1` |
| Software milestone | `SDK_SOFTWARE_READY` |
| Hardware qualified | `false` |
| Release ready | `false` |

证据包包括：

```text
benchmark_manifest.json
dependency_manifest.json
frozen_session.json
host_inventory.json
runtime-manifest.json
sdk_doctor.json
comparison/acceptance_status.json
comparison/summary.json
comparison/sdk_cli_equivalence.json
wheel/
wheel_smoke/
wsl/cli-2d/
wsl/sdk-2d/
wsl/native-orb/
wsl/sdk-post-3d/
wsl/open3d_cuda/verify.json
tests/
```

### 3.3 已完成的软件验收

`da9687fc` 候选的 WSL 重复结果如下。每组均为一次 warm-up 加五次正式运行，正式样本为 5/5：

| Suite | 成功率 | Median | P95 |
| --- | ---: | ---: | ---: |
| CLI frozen 2-D | 5/5 | `8.869 s` | `8.990 s` |
| SDK frozen 2-D | 5/5 | `8.598 s` | `8.994 s` |
| Native ORB CLI | 5/5 | `10.638 s` | `10.754 s` |
| SDK post-3D | 5/5 | `11.268 s` | `11.907 s` |

同时已记录：

- SDK 相对 CLI 的二维 median 没有变慢，记录值约为 `-3.06%`；
- 峰值进程树 RSS 为 `1,408,131,072 bytes`，约 `1.31 GiB`；
- 峰值 GPU memory 为 `1402 MiB`；
- wheel clean venv 中五个正式 CLI 的 `--help` 全部通过；
- SDK/CLI 的 P0–P3、owner、provenance、M6 decision、P3 decoded pixels 和 PNG SHA 精确一致；
- CuPy CUDA、Open3D 0.19 CUDA、native ORB 和 SDK post-3D 均为 `PASS`；
- 当时的完整 Python 回归为 `2002 passed, 36 skipped`，Ruff、compileall 和差异检查通过。

这些结果说明 L0/L1 的核心代码不是停留在设计阶段；已经有可运行 wheel 和重复证据。

### 3.4 当前未提交补丁的验证

本次审计对当前四个修改文件执行了：

```text
pytest tests/test_capture_calibration.py tests/test_video_sdk.py
109 passed

ruff check <四个修改文件>
PASS

compileall <四个修改文件>
PASS

git diff --check
PASS（仅有 Git 的 LF/CRLF 提示）
```

这说明补丁的定向单元和 mock 契约当前通过，但还没有重新构建 wheel、重跑完整 WSL L1 或原生 H0。

### 3.5 WSL 真实相机补充证据

当前主机还保留以下 WSL USB/IP 真机成功样本：

```text
/home/zyh/wsl-camera-tests/SDK_FIXED_FULL_60FPS_20260826T193728/
```

该样本实际记录：

| 项目 | 结果 |
| --- | --- |
| 分辨率/FPS | `848×480 @ 60 FPS` |
| 实际采集时长 | `15.024 s` |
| received/written | `921/921` |
| queue drops | `0` |
| write errors | `0` |
| timestamp regressions | `0` |
| per-frame sync readback | `921` 帧全部验证 |
| clean shutdown | `true` |
| 2-D | `published`、S013 V11、A 级、无 fallback |
| capture stop 到 P3 发布 | `25.701 s`，在 60 秒门内 |
| 3-D | `published`、116 个真实 ORB 帧、CUDA Open3D TSDF、desktop/mobile GLB 和离线 Viewer |

但是，较早的一轮 WSL 五次连续 SDK 相机测试只通过 `3/5`，两次失败发生在 `0/30` warm-up FrameSet。后续 `921/921` 是一个正式样本，不是五次连续稳定性验收。因此当前修复的目标虽然有实机支持，但仍必须在新 wheel 上重新完成 1 次 warm-up 加 5 次正式运行。

该结果只标记为 `WSL_USBIP_SUPPLEMENTAL`；不得写成原生 Ubuntu H0。

## 4. 已经实现的 Linux SDK 组成

### 4.1 Linux 原生 Runtime

已经实现：

- Linux 下直接启动 ELF ORB-SLAM3，不再从 Linux 反向调用 `wsl.exe`；
- 固定 ORB-SLAM3/Pangolin source commit 和 headless runner patch；
- 生成 ORB external Runtime manifest；
- 固定 Open3D 0.19 source commit、CUDA 12.8 和当前 RTX 5060 的 `sm_120` 构建；
- 安装 CuPy CUDA 13、Open3D Python runtime dependencies 和 Orbbec wrapper；
- 使用 Orbbec 官方 v2.8.6 udev 安装脚本，而不是临时手写规则；
- 五个正式 Linux CLI 已经进入根 `pyproject.toml`。

关键文件：

```text
scripts/build_orbslam3_linux.sh
scripts/build_open3d_cuda_linux.sh
scripts/install_linux_python_runtime.sh
scripts/setup_orbbec_linux.sh
requirements/linux-l0-constraints.txt
src/panorama_demo/orbslam3_bridge.py
docs/LINUX_NATIVE_CLI.md
```

### 4.2 视频 SDK API

已经实现并保持照片 API 兼容：

```text
VideoSDKConfig
Gemini305VideoSDK
VideoProcessingJob
VideoPanoramaResult
VideoThreeDResult
SDKDoctorReport
```

公开路径已经覆盖：

- `process_session()`：对已有连续 RGB-D session 执行锁定 S013 V11 正式二维；
- `capture_and_process()`：等待/发现相机、采集并发布二维，可协作取消；
- `run_post_3d()`：二维之后独立执行真实 ORB 和 Open3D TSDF；
- `doctor()`：检查 Python、package resources、V11 identity、CuPy、Open3D、ORB、Orbbec、相机、磁盘和许可证状态；
- typed result 从真实 delivery/report/timing/artifact 加载，不把 Preview 当正式结果。

关键文件：

```text
src/panorama_demo/video_sdk.py
src/panorama_demo/sdk_doctor.py
src/panorama_demo/paths.py
src/panorama_demo/__init__.py
docs/SDK.md
```

### 4.3 Wheel 资源

`setup.py` 已把以下不可变运行资源按原字节放入 wheel：

```text
configs/demo.yaml
configs/video_algorithms/s013_visual_continuity_v11_production.yaml
configs/video_algorithms/s013_visual_continuity_v11_production.lock.json
configs/video_algorithms/baseline_legacy_fast_b07b561.yaml
configs/video_algorithms/baseline_legacy_fast_b07b561.lock.json
configs/video_candidates/s013/S013_output_first_progressive_dense_central_slit_v4_cuda_m63_visual_continuity_v11.yaml
configs/video_candidates/s013/quality_thresholds_m61_v2.json
artifacts/S013_M6_1_metrics_baseline_v2/threshold_approval.json
```

资源 resolver 支持源码树和安装 wheel 两种路径，wheel smoke 已验证这些文件逐字节一致。

### 4.4 验收工具

已经具备：

```text
scripts/run_linux_l1_benchmark.py
scripts/summarize_linux_l1_benchmark.py
scripts/run_sdk_h0_acceptance.py
tests/test_linux_l1_benchmark.py
tests/test_sdk_h0_acceptance.py
tests/test_sdk_wheel_smoke.py
tests/test_sdk_doctor.py
tests/test_video_sdk.py
```

这些工具已经支撑了当前 L1 artifact，但在正式 H0 和发布前还需要加强“成功”的定义，见下一节。

## 5. 当前剩余问题和优先级

| 优先级 | 问题 | 影响 | 处理结论 |
| --- | --- | --- | --- |
| P0 | 当前四文件补丁未提交、未进入 wheel | 旧 wheel 不能代表当前代码 | 先冻结新 commit，再重新验收 |
| P0 | `sdk.doctor()` 可能把“软件通过 + 检测到相机”计算为 hardware qualified | 会错误签发 `SDK_HARDWARE_QUALIFIED` | doctor 只报告能力；硬件资格只能由完整 H0 artifact 签发 |
| P0 | H0 runner 可用一次运行且未做 CLI cross-check 仍写完成 | 单次 WSL smoke 可能被误称 H0 | 正式 H0 固定至少 1 warm-up + 5 runs，并要求完整聚合 |
| P0 | ORB executable 的 RUNPATH 含 `/home/zyh/build/...` | external Runtime 换路径/换机器可能失效 | 复制必需 Pangolin 库并使用相对 `$ORIGIN` RUNPATH |
| P0 | 当前 `libpango_*` 从 `/usr/local/lib` 解析 | 当前主机的全局安装掩盖了 Runtime 缺件 | 在没有 build tree 和 `/usr/local` 辅助库的 clean 环境验证 |
| P0 | 普通 wheel 依赖会拉取 PyPI Open3D/GUI OpenCV | 可能替换正式自建 Open3D CUDA 或引入错误 Runtime | 修正 Linux dependency marker，并用唯一安装器和 `--no-deps` 控制项目 wheel、自建 Open3D、headless OpenCV 和 Orbbec wrapper 顺序 |
| P0 | 项目元数据允许 Python 3.10–3.12，但当前 Open3D CUDA wheel 只有 CPython 3.10 | 3.11/3.12 用户无法复用当前正式 Runtime | 首个 Linux RC 明确只支持 Python 3.10，或为每个 Python ABI 分别构建和验收 Open3D wheel |
| P1 | `install_linux_python_runtime.sh` 不安装 Open3D wheel和项目 wheel | 不是普通用户的一键安装入口 | 新增完整 bundle installer，旧脚本降为 Runtime build helper |
| P1 | L1 runner/summary 主要依赖退出码和 wall time | 无法独立证明 delivery、authority 和像素契约 | 聚合器直接消费正式 report/equivalence/doctor 证据并 fail-closed |
| P1 | `packaging/sdk_pyproject.toml` 和 `SDK_BUNDLE_README.md` 已过时且无引用 | 用户可能构建出只有照片 CLI 的错误包 | 固定根 `pyproject.toml` 为唯一构建入口，删除或明确废弃旧入口 |
| P1 | `docs/SDK.md` Linux 安装说明不足，CHANGELOG 停在旧照片 SDK | 对外使用方式和版本身份不清楚 | 新候选同步更新 SDK 文档、安装说明和 CHANGELOG |
| P1 | WSL 相机连续成功率未达到 5/5 | 当前重试修复尚无稳定性门 | 新 wheel 上重跑 1+5，保留所有失败样本 |
| P1 | 原生 Ubuntu H0 未执行 | 没有 native udev/USB/60 FPS/同步/五次 SDK 证据 | 在新 L1 通过后执行 H0 |
| P1 | 本地分支领先 origin 19 个提交且无 SDK release tag | 外部无法从远端复现当前候选 | 验收后按项目 Git 流程推送、核对远端 SHA 并创建版本 tag |
| Blocked | 项目本身没有明确的发布许可证/EULA 和 wheel license metadata | 即使第三方 notice 完整，也不能定义外部用户的项目使用权 | 发布前选择并加入 proprietary EULA 或批准的开源许可证，同步 wheel metadata |
| Blocked | ORB GPL/商业许可未确定 | 不能把 ORB binary/Vocabulary 作为闭源商业 bundle 发布 | 继续保持 wheel 外置；发布前完成许可决策 |

## 6. 后续实施总览：六个阶段

推荐按以下顺序推进：

```text
阶段 1  冻结当前 USB/IP 修复和新 RC 源码
  ↓
阶段 2  修正 Runtime 可搬运性、安装和资格签发语义
  ↓
阶段 3  从干净提交构建新的 Linux SDK bundle
  ↓
阶段 4  在 WSL 重新签发 SDK_SOFTWARE_READY，并完成 USB/IP 1+5
  ↓
阶段 5  在原生 Ubuntu 执行完整 H0
  ↓
阶段 6  许可证、版本、远端和正式发布收口
```

任何阶段失败都保留原 artifact root，新一轮使用新的 candidate/acceptance ID，不覆盖旧失败记录。

## 7. 阶段 1：冻结当前 USB/IP 修复和新 RC 源码

### 7.1 目标

把当前四个修改文件从“本地补丁”变成可复现 commit，并证明它们没有改变 S013 V11、二维/三维隔离或正式帧后的 fail-closed 行为。

### 7.2 工作项

1. 审阅 `capture_orbbec.py` 的错误分类：只允许 transport、control 和零正式帧 warm-up 失败进入重试；
2. 审阅 `video_sdk.py` 的等待逻辑：`wait_for_camera=true` 时可取消并恢复，`false` 时立即失败；
3. 明确正式帧已经接受后的任何采集、写盘、同步或 metadata 失败不得重试；
4. 保留当前新增的正向和负向测试；
5. 运行定向测试、相关视频/采集测试和全量回归；
6. 检查 immutable S013 V11 YAML/lock/approval 没有变化；
7. 只暂存四个明确代码/测试文件，单独形成修复 commit；不要使用 `git add -A`，不要顺带提交旧的未跟踪蓝图；
8. 版本、打包语义和文档变更在阶段 2 另做明确提交，避免“四文件修复 commit”与实际暂存范围矛盾；
9. 为 Linux 视频 SDK 候选选择新版本。`0.2.0` wheel 已经包含视频 SDK，但 `CHANGELOG.md` 的 `0.2.0` 条目只记录照片 SDK，且版本身份没有区分 Linux RC；因此建议新候选使用 `0.3.0rc1`，最终 H0 后再发布 `0.3.0`。

### 7.3 建议验证命令

```powershell
Set-Location -LiteralPath 'D:\central_strip_Panoramic_Camera'
git status --short
git diff -- src/panorama_demo/capture_orbbec.py src/panorama_demo/video_sdk.py `
  tests/test_capture_calibration.py tests/test_video_sdk.py

$G305Python = 'D:\Panoramic_Camera\.conda\python.exe'
& $G305Python -m pytest -q tests/test_capture_calibration.py tests/test_video_sdk.py
& $G305Python -m pytest -q
& $G305Python -m ruff check src tests scripts/run_linux_l1_benchmark.py `
  scripts/summarize_linux_l1_benchmark.py scripts/run_sdk_h0_acceptance.py
& $G305Python -m compileall -q src tests scripts/run_linux_l1_benchmark.py `
  scripts/summarize_linux_l1_benchmark.py scripts/run_sdk_h0_acceptance.py
git diff --check
```

### 7.4 输出

```text
新 commit SHA
targeted-test.txt
pytest-full.txt
ruff.txt
compileall.txt
git-diff-check.txt
```

### 7.5 完成门

- 工作树中的正式代码修复全部进入一个明确 commit；
- 全量回归通过，未达到的硬件项保持 `NOT_EXECUTED`；
- S013 V11 algorithm/config/implementation identity 未变化；
- 没有 baseline fallback、二维 ORB、正式帧后重试或降级 CUDA required；
- 旧 `da9687fc` artifact 保留为历史证据，但不再代表新 RC。

### 7.6 预计工作量

约 0.5–1 个工程日；不包含重新构建 Open3D。

## 8. 阶段 2：修正 Runtime 可搬运性、安装和资格签发语义

### 8.1 目标

把“在当前 WSL 主机能运行”提升为“可以在同一受支持平台的干净环境复现”，并保证任何 WSL 或单次 smoke 都无法签发原生硬件资格。

### 8.2 ORB external Runtime 可搬运性

当前已现场确认 ORB executable 的 RUNPATH 包含：

```text
/home/zyh/build/g305-orb-build-4f7eec99/...
```

并且 `libpango_display.so`、`libpango_opengl.so` 等从 `/usr/local/lib` 解析。应完成：

1. 将 headless runner 实际需要的 Pangolin 动态库复制到 external Runtime 的受控 `lib/`；
2. 将 executable 和所带动态库的 RUNPATH 改为以 `$ORIGIN` 为基准的相对路径；
3. Runtime manifest 记录 ORB/Pangolin commit、patch、compiler、CMake、平台、文件列表和动态依赖；
4. 把整个 install root 复制到一个不同绝对路径，在不读取原 build tree 的条件下运行 `ldd` 和真实录制回放；
5. clean 验证中任何依赖仍解析到 `/home/zyh/build` 或未声明的 `/usr/local/lib` 均为失败；
6. ORB binary/Vocabulary 继续保持在 Python wheel 外，直到许可证方案批准。

### 8.3 Python/Linux 安装契约

统一根 `pyproject.toml` 为唯一项目 wheel 构建入口。需要消除以下歧义：

- 项目 wheel 当前会声明 PyPI `open3d` 和 GUI `opencv-python`；Linux 元数据应使用明确的 `opencv-python-headless` platform marker；
- 正式 Linux Runtime 实际要求自建 Open3D 0.19 CUDA wheel 和 `opencv-python-headless`；
- `pyorbbecsdk2` 的依赖解析可能拉入不符合正式要求的 Open3D 版本；
- 现有 Runtime install 脚本只装 constraints 和 Orbbec wrapper，没有安装自建 Open3D wheel或项目 wheel。

建议实现一个唯一的 Linux installer，固定执行顺序：

```text
校验目标平台和受支持的 Python ABI
  → 安装冻结的普通 Python dependencies
  → --no-deps 安装匹配 runtime variant 的 Open3D CUDA wheel
  → --no-deps 安装 pyorbbecsdk2 wrapper
  → --no-deps 安装 Gemini 305 project wheel
  → 读取 site/runtime 配置
  → 运行 sdk.doctor()
  → 运行五个 CLI --help 和 CUDA smoke
```

installer 不自动执行 sudo。Orbbec udev 继续由 `setup_orbbec_linux.sh` 在明确检查点调用官方脚本。

当前只有 `cp310` 的 Open3D CUDA wheel。因此第一版 bundle 建议明确固定 Python 3.10；若要兑现项目元数据中的 Python 3.11/3.12 支持，必须为 `cp311`、`cp312` 分别构建 Open3D wheel，并逐个完成同样的 clean-install 和 CUDA 验收，不能只因为纯 Python 项目 wheel 可以导入就宣称完整支持。

现有 `requirements/linux-l0-constraints.txt` 多数使用版本范围，适合作为构建约束，但不是最终 release lock。阶段 2 还应从已验收环境生成精确的 Python 3.10 Runtime lock 和 wheel inventory；安装器只接受该 lock 中的版本，`pip check` 必须通过。

### 8.4 资格签发语义

需要进行三项收口：

1. `sdk.doctor()` 只报告环境能力。检测到相机只能写 `CAMERA_DETECTED`，不能单独把 `hardware_qualified` 设为 true；
2. `SDK_HARDWARE_QUALIFIED` 必须来自一个绑定 wheel SHA、Runtime variant、原生平台和完整 H0 结果的 `native_acceptance_status.json`；
3. 正式 H0 工具必须拒绝 WSL，固定至少一次 warm-up 加五次正式 SDK 运行，并要求 CLI cross-check、fixed/auto/photo、post-3D 和失败隔离证据全部存在。

H0 runner 还必须直接读取 `video_timing.json` 中的
`final_2d.capture_stop_to_p3_published_seconds`。不得把
`primary_post_capture_seconds` 改名后冒充该用户等待指标；两个字段必须按各自语义保存。

同时加强 L1 summary：不要只根据进程退出码，按 operation 分层验收：

- 2-D suite 读取并验证 delivery、authority、fallback、SLA、P0–P3、owner、provenance 和 decision，同时断言二维发布前 ORB/Open3D/3-D 调用为 0；
- ORB suite 验证 native Linux trajectory、真实 pose、完整性和 Runtime identity；
- post-3D suite 验证 ORB、GLB、Viewer、CUDA TSDF、失败隔离和既有二维未被修改。

阶段 2 应把这些规则固化到明确工具中，而不是人工拼 JSON：新增 bundle builder/validator，增强 L1 acceptance aggregator 和 native H0 aggregator。本文后续列出的 bundle manifest、resource equivalence、`acceptance_status.json`、`native_acceptance_status.json` 和 failure-isolation artifact 均由这些工具生成。

### 8.5 文档与版本

- 更新 `docs/SDK.md` 的 Linux wheel/Runtime 安装说明；
- 更新 `CHANGELOG.md`，记录视频 facade、native Linux Runtime、wheel resources 和资格边界；
- 将过时的 `packaging/sdk_pyproject.toml` 与 `packaging/SDK_BUNDLE_README.md` 删除、改名为历史文件或更新为根构建入口说明，只保留一个权威打包方式；
- 为项目本身确定 proprietary EULA 或批准的开源许可证，并写入根目录、wheel metadata 和 bundle；第三方 notice 不能替代项目许可证；
- 明确 Ubuntu、Python、GPU architecture、CUDA、Open3D、Orbbec 和 ORB 的支持矩阵。

### 8.6 完成门

- ORB install root 换绝对路径后仍能通过 `ldd` 和 recorded replay；
- clean 环境不依赖 build tree 或未声明的全局 Pangolin；
- 一个安装入口可完成正式 Python Runtime 和项目 wheel 安装；
- WSL、一次运行或仅检测到相机均不能生成 `SDK_HARDWARE_QUALIFIED`；
- 根 `pyproject.toml` 是唯一权威构建入口；
- 所有变更有正向/负向测试。

### 8.7 预计工作量

约 1–2 个工程日；ORB 重新构建和 clean-copy 回放耗时取决于当前 build cache。

## 9. 阶段 3：从干净提交构建新的 Linux SDK bundle

### 9.1 目标

从阶段 1/2 的干净 commit 构建一个能交给另一位 Linux 使用者安装和复查的 release candidate，而不是只给一个缺少 Runtime 说明的 Python wheel。

### 9.2 推荐交付结构

第一版使用版本化压缩 bundle，不在此阶段扩展到 `.deb`、systemd、gRPC 或后台服务：

```text
gemini305-sdk-0.3.0rc1-ubuntu22.04-x86_64-sm120/
├── install.sh
├── wheels/
│   ├── gemini305_rgbd_panorama-0.3.0rc1-py3-none-any.whl
│   ├── open3d-0.19.0+<commit>-cp310-cp310-manylinux_2_35_x86_64.whl
│   └── dependencies/
│       └── <与 exact lock 匹配的 Python wheelhouse>
├── requirements/
│   ├── linux-l0-constraints.txt
│   └── linux-runtime-lock-py310.txt
├── scripts/
│   ├── setup_orbbec_linux.sh
│   ├── build_orbslam3_linux.sh
│   ├── build_open3d_cuda_linux.sh
│   ├── verify_open3d_cuda.py
│   └── patches/
│       ├── orbslam3-g305-headless-runners.patch
│       ├── open3d-0.19-cmake4-policy.patch
│       ├── open3d-0.19-cuda13-cccl.patch
│       └── stdgpu-cuda13-device-properties.patch
├── acceptance-tools/
│   ├── run_linux_l1_benchmark.py
│   ├── summarize_linux_l1_benchmark.py
│   ├── run_sdk_h0_acceptance.py
│   └── verify_s13_v11_live_equivalence.py
├── config/
│   └── site.example.yaml
├── manifests/
│   ├── sdk-bundle-manifest.json
│   ├── dependency-manifest.json
│   └── runtime-variant-manifest.json
├── docs/
│   ├── INSTALL_LINUX.md
│   ├── SDK.md
│   ├── LINUX_NATIVE_CLI.md
│   └── TROUBLESHOOTING.md
├── licenses/
│   ├── THIRD_PARTY_NOTICES.md
│   ├── orbslam3-g305-runner-COPYING
│   └── PRODUCT_LICENSE_OR_EULA
└── README.md
```

ORB binary/Vocabulary 不进入公开 bundle。ORB external build recipe 必须连同它读取的 patch、GPL COPYING 和 notice 一起提供，否则配方本身不可执行。内部完整 L1 验收需要预先构建一个独立、可搬运的 ORB install root，并通过 `--orb-runtime-root` 或 site YAML 显式传给 SDK；公开无 ORB 安装的 doctor 必须如实报告 full 3-D Runtime 未就绪，不能声称完整 `SDK_SOFTWARE_READY`。二进制是否可分发继续等待许可证结论。

候选 bundle 一经生成即不可修改。阶段 4 的 `acceptance_status.json` 必须放在 bundle 外部的独立 evidence root 中，并引用 bundle SHA；H0 也另建 evidence root。最终发布索引同时引用 bundle、L1 和 H0，而不是把验收结果回写进 bundle 导致 bundle SHA 改变。

### 9.3 构建流程

在 WSL ext4 clean checkout 中执行：

```bash
export G305_CUDA=required
export PIP_NO_CACHE_DIR=1
export G305_BUNDLE_ROOT=/home/zyh/l1-artifacts/SDK_RC_0_3_0_rc1_<shortcommit>

python3 -m venv /home/zyh/venvs/g305-sdk-build-0.3.0rc1
source /home/zyh/venvs/g305-sdk-build-0.3.0rc1/bin/activate
python -m pip install 'build>=1.2,<2'

git status --short
git rev-parse HEAD
python -m build --wheel --outdir "$G305_BUNDLE_ROOT/wheels"
```

随后由新的 bundle builder：

1. 拒绝 dirty worktree；
2. 读取版本、Git SHA、production lock identity 和 Runtime variant；
3. 收集项目 wheel、自建 Open3D wheel、constraints、安装脚本、文档和 notices；
4. 按 exact Python lock 收集与目标 ABI 匹配的 dependency wheelhouse，并生成 bundle manifest 和文件 SHA；
5. 不复制采集数据、输出目录、开发缓存、测试 fixture 或 ORB binary；
6. 生成压缩包，但保留未压缩目录供验收；
7. 许可证或体积原因导致某个依赖 wheel 不能进入 wheelhouse 时，必须把 bundle 明确降为 `network-assisted`，列出网络依赖和来源，不能称为离线/完全自包含。

### 9.4 Clean-install smoke

在仓库外新建 venv，并且只使用 bundle 提供的安装入口：

```bash
python3 -m venv /home/zyh/venvs/g305-sdk-0.3.0rc1-smoke
source /home/zyh/venvs/g305-sdk-0.3.0rc1-smoke/bin/activate
bash "$G305_BUNDLE_ROOT/install.sh" --venv "$VIRTUAL_ENV"

unset PYTHONPATH
python -c 'import panorama_demo; print(panorama_demo.__file__, panorama_demo.__version__)'
python -c 'from panorama_demo import Gemini305VideoSDK; print(Gemini305VideoSDK().doctor())'
python -m pip check
g305-capture --help
g305-video-live --help
g305-video-panorama --help
g305-video-post-3d --help
g305-orbslam3-trajectory --help
```

不能用 editable install，不能通过源码树补资源，不能让 `PYTHONPATH` 指向仓库。上面的公开无 ORB smoke 允许 doctor 如实报告 ORB external Runtime 未就绪；完整 L1 smoke 还要重新运行一次 installer/doctor，并显式传入阶段 2 产生的 relocatable ORB Runtime root，此时所有 software-required 项才允许为 PASS。

### 9.5 输出

```text
项目 wheel
Open3D runtime wheel
版本化 Linux bundle
sdk-bundle-manifest.json
dependency-manifest.json
runtime-variant-manifest.json
linux-runtime-lock-py310.txt
clean-install doctor.json
cli-help.json
wheel/resource byte-equivalence.json
```

### 9.6 完成门

- bundle 来自干净 commit，版本、Git SHA、wheel SHA 和 production lock 一致；
- clean venv 可通过一个入口安装；
- 若标记为 offline bundle，安装期间以 `--no-index` 只使用随包 wheelhouse；若标记为 network-assisted，则 manifest 精确记录所有联网依赖；
- 五个 CLI、V11 resources、CUDA smoke 和 `pip check` 全部通过；
- 公开无 ORB doctor 如实报告外部 3-D Runtime 缺失；绑定 relocatable ORB root 的内部完整 doctor 软件项全部通过；
- 安装过程没有拉取或覆盖错误的 Open3D；
- bundle 明确标明当前 Runtime variant 只覆盖 Ubuntu 22.04 x86_64、Python 3.10 和 `sm_120`；其它 GPU 必须生成独立受控 variant；
- ORB 分发边界和未完成 H0 清楚写入 manifest；
- bundle SHA 冻结后不再回写 L1/H0 结果，所有资格 artifact 在 bundle 外引用其 SHA。

### 9.7 预计工作量

约 0.5–1 个工程日；若必须重建 Open3D，另计数小时构建时间并继续监测 D 盘至少 `10 GiB` 安全余量。

## 10. 阶段 4：重新签发 WSL `SDK_SOFTWARE_READY`

### 10.1 目标

让新的 RC 获得与 `da9687fc` 相同或更强的软件证据，并用当前相机恢复补丁完成 WSL USB/IP 连续 1+5 稳定性补充验证。

### 10.2 必需软件 suites

使用同一冻结真实 RGB-D session、同一 production lock、同一 Runtime variant：

1. CLI frozen 2-D：1 warm-up + 5 runs；
2. SDK frozen 2-D：1 warm-up + 5 runs；
3. native ORB：1 warm-up + 5 runs；
4. SDK post-3D：1 warm-up + 5 runs；
5. clean-wheel SDK/CLI exact equivalence；
6. CuPy/Open3D CUDA required smoke；
7. 全量 pytest、Ruff、compileall；
8. CPU/RAM/GPU、P3、ORB、TSDF/GLB 分开记录。

命令骨架：

```bash
export G305_CUDA=required
export G305_L1_ROOT=/home/zyh/l1-artifacts/SDK_SOFTWARE_<date>_<shortcommit>
export G305_FROZEN_SESSION=/home/zyh/g305-data/L1-FROZEN-001
export G305_BUNDLE_ROOT=/home/zyh/l1-artifacts/SDK_RC_0_3_0_rc1_<shortcommit>
export G305_ACCEPTANCE_VENV=/home/zyh/venvs/g305-sdk-0.3.0rc1-acceptance
export G305_ORB_RUNTIME_ROOT=/home/zyh/opt/g305-orbslam3-0.3.0rc1

python3 -m venv "$G305_ACCEPTANCE_VENV"
source "$G305_ACCEPTANCE_VENV/bin/activate"
bash "$G305_BUNDLE_ROOT/install.sh" \
  --venv "$G305_ACCEPTANCE_VENV" \
  --orb-runtime-root "$G305_ORB_RUNTIME_ROOT"
unset PYTHONPATH
python -c 'import panorama_demo; print(panorama_demo.__file__, panorama_demo.__version__)'
python -m pip check

python "$G305_BUNDLE_ROOT/acceptance-tools/run_linux_l1_benchmark.py" frozen-2d \
  --artifact-root "$G305_L1_ROOT/wsl/cli-2d" \
  --platform-label WSL_UBUNTU_22_04 --entry cli \
  --session "$G305_FROZEN_SESSION" --warmups 1 --runs 5

python "$G305_BUNDLE_ROOT/acceptance-tools/run_linux_l1_benchmark.py" frozen-2d \
  --artifact-root "$G305_L1_ROOT/wsl/sdk-2d" \
  --platform-label WSL_UBUNTU_22_04 --entry sdk \
  --session "$G305_FROZEN_SESSION" --warmups 1 --runs 5

python "$G305_BUNDLE_ROOT/acceptance-tools/run_linux_l1_benchmark.py" orb \
  --artifact-root "$G305_L1_ROOT/wsl/native-orb" \
  --platform-label WSL_UBUNTU_22_04 \
  --session "$G305_FROZEN_SESSION" --warmups 1 --runs 5

python "$G305_BUNDLE_ROOT/acceptance-tools/run_linux_l1_benchmark.py" post-3d \
  --artifact-root "$G305_L1_ROOT/wsl/sdk-post-3d" \
  --platform-label WSL_UBUNTU_22_04 --entry sdk \
  --session "$G305_FROZEN_SESSION" \
  --two-d-output '<frozen-2d-output>' --warmups 1 --runs 5
```

### 10.3 精确正确性门

每轮必须校验：

- production algorithm、implementation、config SHA 和 `allow_baseline_fallback=false`；
- 2-D authority 为正式 S013 V11 ignore-pose，`pose_used=false`；
- P0、P1、P2、P3、valid、owner、provenance、source/schedule、pair seam、M6 decision 一致；
- decoded P3 像素精确一致，PNG SHA 一致；
- 60 秒发布门、manual review 和 grade 语义正确；
- ORB 是真实 native Linux trajectory，不是插值或二维 motion；
- post-3D 在二维发布和资源释放后开始；失败不会撤销二维；
- GLB magic、desktop/mobile mesh、离线 Viewer 和无外部 CDN 契约通过。

### 10.4 WSL USB/IP 1+5 补充验证

保持 WSL 和 `usbipd --auto-attach` 进程存活，使用新 wheel 运行：

```bash
python "$G305_BUNDLE_ROOT/acceptance-tools/run_linux_l1_benchmark.py" physical-live \
  --artifact-root "$G305_L1_ROOT/wsl-usbip-sdk-physical" \
  --platform-label WSL2_USBIP_SUPPLEMENTAL --entry sdk \
  --width 848 --height 480 --fps 60 --duration 3 \
  --video-exposure-us 100 --maximum-post-seconds 60 \
  --no-preview --warmups 1 --runs 5
```

这组运行专门验证：

- 五次正式运行均为 0 drops、0 write errors、0 timestamp regressions；
- 每帧 sync readback、clean shutdown、正式 P3 均通过；
- 所有失败样本保留，不能挑选五次中的最快结果。

上面的 1+5 正常运行不能自动证明每一种恢复分支。还要由阶段 2 新增的 USB/IP supplemental acceptance 工具执行并保存四个独立用例：

1. 相机缺席且 `wait_for_camera=false`，验证立即失败及 reason code；
2. 相机缺席时启动等待，再触发 SDK cancellation，记录取消延迟和干净退出；
3. 正式帧前断开/重新 attach，验证 transport/control/零帧 warm-up 失败被恢复，随后产生干净正式 session；
4. 接受首个正式帧后断开，验证任务 fail-closed、会话不具备产品资格，且程序不自动重试生成拼接后的混合 session。

对应产物至少为：

```text
no-wait-immediate-failure.json
wait-cancel.json
pre-formal-reconnect-recovery.json
post-formal-disconnect-fail-closed.json
```

无论 WSL 真机结果多好，这一组仍只写 `WSL_USBIP_SUPPLEMENTAL`。

### 10.5 新 acceptance artifact

自动生成的 `acceptance_status.json` 至少包含：

```text
source_commit
source_clean
sdk_version
wheel_path
wheel_sha256
bundle_manifest
runtime_variant
l0_wsl_runtime
wheel_and_resources
sdk_doctor
sdk_cli_exact_equivalence
wsl_recorded_replay_2d
wsl_native_orb
wsl_post_3d
sdk_overhead
memory_and_gpu_budget
wsl_usbip_supplemental
native_camera_acceptance
orb_distribution_license
overall_software
hardware_qualified
release_ready
```

### 10.6 完成门

- 四个必需软件 suite 全部 5/5；
- exact equivalence 全部通过；
- clean-wheel smoke 和 doctor 软件项通过；
- WSL USB/IP 1+5 和四个故障用例属于 supplemental；若失败，软件 replay 可保留 PASS，但当前相机恢复补丁不得宣称 WSL USB/IP 稳定；
- 生成绑定新 commit、新 wheel SHA 和新 Runtime variant 的 `SDK_SOFTWARE_READY`；
- `native_camera_acceptance` 保持 `NOT_EXECUTED/WAITING_FOR_H0`；
- `hardware_qualified=false`、`release_ready=false`。

WSL USB/IP supplemental 不是 `required_for_software_ready`，也不是进入原生 H0 的硬前置。若失败原因已明确隔离为 USB/IP/WSL transport，而 clean replay 和软件硬门通过，仍可签发软件里程碑并到原生 H0 验证；若暴露的是 SDK 代码、写盘、同步或 fail-closed 缺陷，则必须先修复和重签。

### 10.7 预计工作量

约 0.5–1 个工程日；若出现 USB/IP 瞬断，定位和重跑使用新 artifact ID，不覆盖失败轮次。

## 11. 阶段 5：原生 Ubuntu H0

### 11.1 用户/设备检查点

H0 需要：

- 一台原生 Ubuntu 22.04 LTS x86_64 机器，优先无虚拟化层；
- NVIDIA GPU、驱动和与该 GPU compute capability 匹配的受控 Runtime variant；
- Gemini 305、固定 USB 3.x 端口、线缆和供电；
- 阶段 3 产生的同一 wheel/bundle；
- 阶段 4 的冻结 session 和 `SDK_SOFTWARE_READY` artifact；
- 执行官方 udev 安装脚本时的一次明确 sudo 配合。

若使用 Ubuntu 24.04 或不同 GPU，只能生成新的 `runtime_variant_id` 并单独声明支持范围，不能把结果反写成 Ubuntu 22.04/`sm_120` 已验收。

### 11.2 H0.0：无相机回放预检

在插入相机前完成：

1. 安装阶段 3 的同一 wheel SHA，不在 native 机器编辑源码；
2. 安装匹配的 Open3D/CuPy/ORB external Runtime；
3. 运行 doctor 和五个 CLI `--help`；
4. 对同一冻结 session 运行 CLI 2-D、SDK 2-D、native ORB 和 SDK post-3D；
5. 与 WSL 的 P0–P3、owner、provenance、decision 和 P3 decoded pixels 精确比较；
6. 保存 native doctor、runtime manifest、依赖清单、命令、report 和 timing；
7. 任一差异未定位前，不进入真机阶段。

原生机应从解压后的冻结 bundle 执行 acceptance tools，并先记录实际导入路径：

```bash
export G305_NATIVE_BUNDLE_ROOT=/var/tmp/g305-h0/gemini305-sdk-0.3.0rc1-ubuntu22.04-x86_64-<variant>
source /var/tmp/g305-h0/venv/bin/activate
unset PYTHONPATH
python -c 'import panorama_demo; print(panorama_demo.__file__, panorama_demo.__version__)'
python -m pip check
```

### 11.3 H0.1：non-root 和相机基础

1. 通过固定 commit 的 Orbbec 官方脚本安装 udev rule；
2. 普通用户枚举相机，禁止用 root capture 掩盖权限问题；
3. 确认 USB 3.x、固件、wrapper、SDK version 和 exact profiles；
4. 确认 `848×480 RGB + 848×480 Y16 @ 60 FPS`；
5. 确认 metadata、Trigger Out、同步字段和每帧 readback；
6. 验证等待相机、热插入继续、Ctrl+C/SDK cancellation；
7. 验证 Pipeline stop、切换 `STANDALONE`、关闭 Trigger Out 和 clean shutdown。

### 11.4 H0.2：公开 SDK 五次正式运行

阶段 2 加强后的正式 H0 工具固定执行：

```bash
export G305_CUDA=required
export G305_H0_ROOT=/var/tmp/g305-h0/H0_<date>_<shortcommit>

python "$G305_NATIVE_BUNDLE_ROOT/acceptance-tools/run_sdk_h0_acceptance.py" \
  --artifact-root "$G305_H0_ROOT/sdk_physical" \
  --runs 5 --warmups 1 \
  --width 848 --height 480 --fps 60 --duration 3 \
  --video-exposure-us 100 --maximum-post-seconds 60 \
  --no-preview --post-3d-last
```

这里的 `--no-preview` 只关闭本地图形窗口，不关闭内部 latest-only 非权威 Preview。H0 仍必须从 `live_preview.jpg`、`live_preview_state.json` 和 `video_timing.online_2d` 验证 Preview 更新、延迟和隔离状态；如果未来 CLI 语义改变为真正禁用内部 Preview，则必须删除该参数或增加独立 Preview suite。

通过要求：

- 一次 warm-up 不计入五次正式样本；
- 五次正式样本全部成功，不能挑选；
- 0 queue drops、0 write errors、0 timestamp regressions；
- 每帧 sync readback 数等于 written frames；
- clean shutdown 和 video product eligibility 正确；
- 非权威 Preview 有可用证据，Preview 独立失败时按正式隔离语义记录；
- 正式 P3 发布、A/B/C/F 和 manual review 语义正确；
- 最后一个成功 session 通过独立 SDK post-3D；
- 3D 失败注入只能写 `3d/video_3d_failure.json`，不能撤销二维。

阶段 2 增强后的 H0 工具必须自动执行 3-D failure-isolation 子步骤：先记录一个已发布二维目录中 `video_delivery.json`、P3 和 provenance 的 SHA，然后对同一 session 使用一个明确不存在的 ORB executable 路径启动新的 post-3D failure 输出。验收要求捕获 `PanoramaProcessingError`、生成 `3d/video_3d_failure.json`、不生成 3-D delivery，并确认原二维文件及 SHA 完全未变。结果写入 `post_3d_failure_isolation.json`。

### 11.5 H0.3：CLI 交叉检查和三种采集契约

使用相同参数执行一次 `g305-video-live`：

```bash
g305-video-live \
  --width 848 --height 480 --fps 60 \
  --duration 3 --video-exposure-us 100 --no-preview \
  --output "$G305_H0_ROOT/cli_crosscheck/sessions" \
  --panorama-output "$G305_H0_ROOT/cli_crosscheck/2d" \
  --maximum-post-seconds 60
```

另外分别执行：

```bash
g305-capture \
  --width 848 --height 480 --fps 60 --duration 3 \
  --video-exposure-us 100 --no-preview \
  --output "$G305_H0_ROOT/capture_cli/video_fixed"

g305-capture \
  --width 848 --height 480 --fps 60 --duration 3 --no-preview \
  --output "$G305_H0_ROOT/capture_cli/video_auto"

g305-capture \
  --photo-mode --width 848 --height 480 --fps 60 --max-frames 10 \
  --output "$G305_H0_ROOT/capture_cli/photo"
```

fixed、auto 和 photo 是三个独立契约，不能互相代替，也不能用 WSL 历史 session 补齐。

### 11.6 H0 输出

```text
native_acceptance_status.json
native runtime-variant-manifest.json
native doctor.json
native replay equivalence.json
run_01 ... run_05 完整目录
CLI cross-check 目录
fixed/auto/photo capture 目录
post-3D success 与 failure-isolation 证据
native performance summary
```

### 11.7 完成门

只有 `native_acceptance_status.json` 中全部 native-required 硬门为 `PASS`，才写：

```text
overall_hardware = PASS
milestone = SDK_HARDWARE_QUALIFIED
hardware_qualified = true
```

以下硬门只接受 `PASS`：wheel/bundle/Runtime identity、原生平台、native replay equivalence、non-root/udev/USB3、精确 profile、五次 SDK、0 drops/errors/regressions、逐帧 sync、clean shutdown、正式 2-D delivery、CLI cross-check、fixed/auto/photo、post-3D 和 3-D failure isolation。

`PASS_WITH_WARNING` 只允许用于两个观察项：

- 不同硬件导致无法进行 Windows/WSL/native 相对性能归因，但本机 60 秒绝对 SLA 已通过并保存原始 median/P95；
- 个别非权威 Preview 隔离失败，但至少有其它正式样本提供有效 Preview 证据，且五次正式 P3 均未受影响。

五次全部没有可用 Preview、绝对 SLA 超时、质量/身份/同步/采集/发布硬门失败、任何 `NOT_EXECUTED`、CLI cross-check 缺失、少于五次或 post-3D 未执行，都不能签发硬件资格。

### 11.8 预计工作量

原生机器和相机准备好后约 1–2 个工程日。机器、USB、驱动或 GPU Runtime 故障使用同一 wheel 修环境；代码或 bundle 缺陷必须回 WSL 生成新候选并重跑阶段 3/4。

## 12. 阶段 6：许可证、版本、远端和正式发布收口

### 12.1 ORB 分发决策

发布前必须在以下两条路径中作出明确决定：

1. 获得适用的 ORB-SLAM3 商业许可；或
2. 采用经过法律/合规确认的 GPL 分发方案，并履行对应源码和许可证义务。

在结论明确之前：

- Python wheel 不包含 ORB binary/Vocabulary；
- bundle 只包含外部构建配方、接口说明和 manifest；
- 内部开发可继续使用 external Runtime；
- `release_ready` 必须保持 `false`。

### 12.2 版本和发布身份

推荐版本路线：

```text
当前历史候选        0.2.0 + da9687fc
新软件候选          0.3.0rc1 + <new commit>
H0 修复候选         0.3.0rcN
最终正式版本        0.3.0
```

最终版本必须绑定：

```text
Git commit/tag
project wheel SHA
bundle SHA
production lock identity/config SHA
Open3D wheel SHA/source commit/build flags
CuPy/CUDA versions
Orbbec wrapper/SDK/udev source commit
ORB external Runtime manifest 或许可包身份
WSL SDK_SOFTWARE_READY artifact ID
native SDK_HARDWARE_QUALIFIED artifact ID
```

### 12.3 Git 和远端

当前本地分支领先 origin 19 个提交。发布前应：

1. 保留并审阅当前所有提交；
2. 按项目流程推送候选分支；
3. 用远端查询核对 branch tip 与本地 SHA 相同；
4. H0 通过后创建正式 tag；
5. 不在 dirty worktree 或仅本地存在的 commit 上发布；
6. 发布说明同时列出已实现、已软件验收、已原生验收和仍受限功能。

### 12.4 用户文档和支持范围

最终至少提供：

- 五分钟安装/doctor/首个 recorded replay；
- 相机 udev、USB 和 profile 检查；
- Python SDK 和五个 CLI 示例；
- 2-D/3-D 结果、失败文件和日志位置；
- 支持的 Ubuntu、Python、GPU、CUDA、Open3D、Orbbec 版本矩阵；
- 不支持项和 Runtime variant 生成方法；
- ORB 许可证和 external Runtime 边界；
- `SDK_SOFTWARE_READY`、`SDK_HARDWARE_QUALIFIED` 与 `SDK_RELEASE_READY` 的区别。

### 12.5 最终完成门

只有同时满足以下条件，才可写 `SDK_RELEASE_READY=true`：

- 新 RC 的 `SDK_SOFTWARE_READY` 为 PASS；
- 同一 wheel 的原生 Ubuntu H0 为 PASS；
- 安装 bundle 在 clean target 上可复现；
- ORB 分发许可结论已批准；
- 项目自身许可证/EULA 已批准并进入 wheel metadata 和 bundle；
- 版本、tag、远端 SHA、wheel/bundle/Runtime manifest 全部一致；
- 文档、CHANGELOG、third-party notices 和支持矩阵完整；
- 没有用 WSL、单次相机 smoke、旧 artifact 或静态测试替代原生验收。

### 12.6 预计工作量

工程收口约 0.5–1 个工程日；许可证等待时间不计入工程工期。

## 13. 推荐总体工期和依赖

| 阶段 | 预计工程量 | 前置依赖 | 是否需要用户/外部条件 |
| --- | ---: | --- | --- |
| 1. 冻结当前修复 | 0.5–1 天 | 当前工作树 | 无，提交/推送按用户流程 |
| 2. Runtime/安装/资格语义 | 1–2 天 | 阶段 1 | ORB 许可不阻塞内部实现 |
| 3. 构建 RC bundle | 0.5–1 天 | 阶段 2 | 磁盘至少保留 10 GiB |
| 4. WSL 软件与 USB/IP 1+5 | 0.5–1 天 | 阶段 3 | Gemini 305、USB/IP 稳定连接 |
| 5. 原生 Ubuntu H0 | 1–2 天 | 阶段 4 | 原生 Ubuntu、GPU、相机、sudo udev 检查点 |
| 6. 正式发布收口 | 0.5–1 天 | 阶段 5 | ORB 分发许可、远端发布权限 |

在没有额外故障的情况下，工程工作约为 **4–8 个工作日**，不包含原生机器排期、完整 Open3D 重建、下载时间和 ORB 许可等待。

## 14. 停止条件和失败处理

以下情况必须停止扩大工作，先保留证据并定位：

- S013 V11 immutable algorithm/config/implementation identity 发生变化；
- SDK/CLI 的首个 P0–P3、owner、provenance、decision 或 decoded P3 差异未定位；
- 需要启用 baseline fallback、二维 ORB、CPU fallback 或放宽 CUDA required 才能通过；
- ORB copied Runtime 仍依赖 build tree 或未声明的全局库；
- clean install 拉入错误 Open3D/Orbbec 依赖；
- D 盘剩余空间低于 `10 GiB`、出现 `ENOSPC` 或下一重型步骤明确无法容纳；
- H0 工具在 WSL、少于五次或 CLI cross-check 缺失时仍可能签发硬件资格；
- 原生机器的 wheel SHA、production lock 或 Runtime variant 与 L1 候选不一致；
- 许可证边界要求分发 ORB 文件，但许可方案尚未批准。

失败处理原则：

- 代码/wheel/bundle 缺陷：回 WSL 修复，产生新 commit、新 RC 和新 L1 artifact；
- 原生机器环境、udev、USB、线缆或驱动缺陷：保留同一 wheel，修环境后用新 H0 ID 重测；
- 3-D 失败：保留已发布二维，只记录独立 3-D failure；
- 所有失败目录保留，不覆盖、不删除、不挑选最快成功样本。

## 15. 建议立即执行的下一项工作

立即从 **阶段 1** 开始，而不是直接做 H0：

1. 收口并提交当前四文件 USB/IP/相机启动恢复补丁；
2. 修正 doctor/H0 的资格签发语义和 ORB Runtime 可搬运性；
3. 以 `0.3.0rc1` 从 clean commit 生成新 bundle；
4. 用新 wheel 重跑 WSL 软件 5/5 和 USB/IP 相机 1+5；
5. 新 `SDK_SOFTWARE_READY` 通过后，再准备原生 Ubuntu H0。

这是当前最短、证据最连续的路线。它复用已经完成的 Linux/CUDA/Open3D/ORB/SDK 工作，不重新设计算法，也不会把旧 wheel、WSL 真机或单次成功误写成最终 Linux SDK 发布。
