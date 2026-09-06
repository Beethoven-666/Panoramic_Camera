# Linux SDK 故障处理

- 安装 checksum/ABI/GPU/platform 不符：安装器拒绝安装。使用匹配的离线 bundle，不能绕过为成功。
- 安装中途失败：保留 `prefix.install-failed-*` 的 install.log/failure.json；选择新 prefix 重试。
- base doctor 三维 BLOCKED：检查独立 Open3D addon 和外部 ORB root；不需要为二维安装三维包。
- `pip check` 出现 open3d 0.18、opencv-python 或 numpy 冲突：误装了上游 wrapper metadata，
  重新用 exact bundle 在新 prefix 安装，不在现有环境盲目升级。
- CUDA 不可用：核对 variant 的 sm_120、驱动及 CUDA 12.8 NVRTC。doctor 必须实际分配 CUDA 数组。
- 相机占用/锁目录不可写：只读客户端仍可看状态；由相机控制者释放或修正本机 lock root 权限。
  禁止删除锁文件抢占，也不要把锁放在网络文件系统。
- Preview 写盘/编码失败：保留 `live_preview_failure.json`，采集与正式二维继续。
- 低磁盘、写盘失败或进程崩溃：停止采集，验证 durable committed prefix，恢复成新的 job/session；
  首个正式帧提交后不允许重连并追加原 session。timestamp regression 不能恢复成正式合格产品。
- 单帧/窄带 emergency 只生成独立 emergency 交付，要求人工复核，不等于正式 P3。
- 三维失败：查看 `3d/video_3d_failure.json`，原二维 delivery 保持有效；不要删掉二维结果重试三维。
- retention 仅清理 SDK 拥有且请求交付全部成功、无警告的原始数据。外部 session 不清理，
  清理中断通过 cleanup_pending 恢复；有 emergency、降级、取消或三维失败时保留原始帧。

H0 runner 会拒绝 WSL、少于 1+5、缺少 CLI cross-check、fixed/auto/photo 契约或三维成功/失败隔离证据。
L1 aggregator 读取真实 delivery/report/timing/owner/provenance，而不是根据退出码或 wall time 签发资格。
所有失败记录保留；旧 artifact 不覆盖，旧 bundle SHA 对应的资格不能迁移到新包。
