# Linux 0.3.0rc1 开发级离线安装

范围固定为 Ubuntu 22.04 x86_64、CPython 3.10、glibc >=2.35 和 sm_120。
WSL 只用于开发级 portability smoke，不是原生 H0。安装器不安装驱动、CUDA、系统包或 udev。
使用 bundle 的 runtime-variant-manifest.json 查看精确依赖；当前组合是 CuPy 14.2.0、
CUDA 13 cuBLAS wheel 与 CUDA 12.8 NVRTC/Open3D toolkit，不能随意替换为其它 GPU variant。

## 二维基础包

先校验 bundle，再在全新专用 prefix 安装：

```bash
python3.10 scripts/verify_linux_sdk_bundle.py /path/to/base-bundle --output /new/path/verify.json
cd /path/to/base-bundle
bash install.sh --prefix /home/me/gemini305-rc1 --python /usr/bin/python3.10 --offline
/home/me/gemini305-rc1/venv/bin/python -m pip check
/home/me/gemini305-rc1/venv/bin/g305-sdk-doctor --json
```

base 包含采集、非权威文件 Preview、S013 V11 正式二维、SDK 和 CLI。
base 不含 Open3D、ORB binary 或 Vocabulary。安装只读取本地 exact wheelhouse，使用
`--no-index --require-hashes --no-deps`；原始 wrapper 依赖不会自动拉取 PyPI Open3D。
安装器创建临时 venv、保存日志、执行 doctor 和五个 CLI help，成功后发布专用 prefix。
相同 bundle 重复安装为幂等；不同内容需要新 prefix。失败目录保留为 `.install-failed-*`。

相机权限单独操作。只有用户明确执行以下命令才安装官方 v2.9.3 udev 规则：

```bash
sudo bash ./setup_orbbec_udev.sh
```

规则固定于 Orbbec commit `2f6561c28255d805b34aa00a690199ce40e96c81`，与 native 2.9.3 配套。
wrapper `2.1.2+g305.1` 是上游 2.1.2 的 metadata 修订；69 个 Runtime 文件未变。
Gemini 305 记录过 firmware 1.0.70，当前允许范围仅该版本；这不表示本次已执行硬件验收。
相机 lock root 必须是本机文件系统中已有的可写目录，由调用者配置；不要用 NFS。

## 独立三维 addon

```bash
cd /path/to/3d-addon-bundle
bash install-addon.sh --prefix /home/me/gemini305-rc1 --python /usr/bin/python3.10 --offline
```

addon 使用独立 venv，通过只读 base package 路径获得二维公共代码，base 看不到 addon。
安装后执行自建 Open3D 0.19 CUDA TSDF smoke。ORB 许可未解决，公开 addon 不包含 ORB
binary/Vocabulary；仅内部验证可显式绑定独立 ORB runtime root。无 Runtime 时三维返回
`THREE_D_RUNTIME_MISSING_OR_UNLICENSED`，二维交付保持有效。

```bash
bash uninstall-addon.sh --prefix /home/me/gemini305-rc1 --python /usr/bin/python3.10
cd /path/to/base-bundle
bash uninstall.sh --prefix /home/me/gemini305-rc1 --python /usr/bin/python3.10
```

卸载只删除安装 manifest 声明的文件，保留用户新增 session/job/output，不卸载 udev、驱动或
外部 ORB。保留用户文件时 prefix 会继续存在，这是预期行为。

## 构建与签名

根 pyproject 是唯一 metadata。构建必须使用干净 Git checkout 或从干净 commit 导出的源文件，
导出应包含 `build_support.RUNTIME_RESOURCES` 的全部八项，尤其是
`artifacts/S013_M6_1_metrics_baseline_v2/threshold_approval.json`，并写入 `.g305-source-commit`。
构建工具：`scripts/build_linux_sdk_bundles.py --help`；输出目录必须全新。
源码合规归档携带构建脚本、patch、第三方 notice 和 SPDX SBOM。
`--signing-key` 只接受调用者已有 GPG key id，工具不生成、复制或提交私钥。
测试签名必须加 `--test-only-signature`，无 key 时为 UNSIGNED。
签名不能代替 acceptance，项目所有者尚未提供 LICENSE/EULA，发布继续 blocked。
