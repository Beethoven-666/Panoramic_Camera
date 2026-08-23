# D3 performance profile

Run: `D3_object_first_dense_source_compositor`, validation interval `[0.30, 0.48]`, with the capture-bound online ORB-SLAM3 RGB-D trajectory reused after integrity verification.

| Stage | Wall time (s) | Execution evidence |
| --- | ---: | --- |
| Config and strict session validation | 1.509 | CPU |
| Scan analysis | 0.000 | CPU |
| Reuse and verify complete real ORB-SLAM3 trajectory | 0.037 | CPU (the accepted capture-time ORB run took 108.924 s and is not charged again) |
| Render-keyframe selection | 17.513 | CPU |
| Open3D RGB-D render-edge audit (59 edges) | 5.006 | CPU; `open3d_rgbd` observed, no Open3D CUDA call |
| Calibrated render, D2 and D3 exports | 38.996 | CPU orchestration/renderer, plus RAFT-small GPU inference as detailed below |
| Quality and report | 0.009 | CPU |
| Atomic 2-D delivery | 0.044 | CPU |
| Offline object evaluation | 0.330 | CPU; outside the primary SLA |

Primary post-capture wall time was **63.509 s** before final report/delivery accounting (or **63.560 s** including those two final CPU stages), exceeding the configured 60.000 s target. The validation/audit performance grade is therefore `NE`; it is not a production performance claim.

## GPU evidence

The locked torchvision RAFT-small model executed on `cuda:0` in FP16 for D2 forward/backward flow and for D3's 59 forward real-source track pairs. GPU is NVIDIA GeForce RTX 5060 Laptop GPU. The current profiler records CPU wall-clock stages but does not insert CUDA events around individual RAFT calls, so it does **not** support a defensible GPU-only millisecond subtotal. No such subtotal is inferred here.

`G305_CUDA=off` was preserved: the calibrated pushbroom acceleration audit reports CPU-only execution with 420 CPU calls, zero Open3D/OpenCV/CuPy CUDA calls, and zero host/device bytes through that acceleration backend. RAFT is separate from that backend and is explicitly audited as `cuda:0`.

## Gate outcome

D1/D2 evidence was retained (zero unowned pixels; all three compact supports at 1.0; RAFT FB P95 0.689 px against 1.5 px). D3 rendered 1,346 real-source object-owner pixels, but the resulting visual grade remains C and D3 Gate 3 remains blocked: carton and cable have 34 and 26 owners respectively. No D4/D5 operation was invoked.
