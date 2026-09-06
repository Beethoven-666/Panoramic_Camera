# 受控 Runtime 范围

| 项目 | 0.3.0rc1 阶段 0–2 范围 |
| --- | --- |
| 平台 | Ubuntu 22.04 x86_64，glibc >=2.35 |
| Python | CPython 3.10；其它 ABI 不支持此 bundle |
| GPU | sm_120；不声明所有 NVIDIA GPU 通用 |
| 二维 | 锁定 S013 V11 CUDA，ignore-pose，不允许 baseline fallback |
| 相机 | Gemini 305，VID 0x2bc5/PID 0x0840，firmware 1.0.70 受控组合 |
| Orbbec | upstream wrapper 2.1.2，metadata variant 2.1.2+g305.1，native 2.9.3 |
| Open3D | 独立 addon，自建 0.19.0+1e7b17438，cp310 manylinux_2_35，CUDA 12.8 |
| ORB | 外部独立进程，公共包不携带 binary/Vocabulary；分发许可 blocked |
| WSL | 开发 smoke/真实 recorded replay；不能签发原生 H0 |
| Windows CPython 3.12 | 既有开发测试环境；不是该 Linux bundle 的安装资格 |

`doctor()` 只报告 PASS/FAIL/NOT_CHECKED/BLOCKED，readiness 兼容字段固定 null。
只有 `scripts/aggregate_sdk_acceptance.py` 可以写 software/native acceptance artifact，必须
绑定 source commit、project wheel SHA、bundle checksum manifest SHA、variant、platform 和 raw runs。
阶段 2 不执行资格签发；`software_ready=false`、`hardware_qualified=false`、`release_ready=false`。
原生 Ubuntu H0、最终 1+5 软件验收、速度/长时现场测试、所有者许可证和正式签名属于后续阶段。
