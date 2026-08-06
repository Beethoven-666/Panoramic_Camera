from __future__ import annotations

import cv2
import numpy as np

from panorama_demo.session import CameraIntrinsics, RGBDFrame
from panorama_demo.video_v2_audit_export import (
    V2CudaAuditExportContext,
    stage_v2_cuda_audit_exports,
)
from panorama_demo.video_visual_renderer_v2 import CudaSourceStrip


def test_v2_cuda_audit_exports_stage_real_bgra_sources_and_owner_only_without_mutation(tmp_path):
    height, width = 4, 8
    first = np.zeros((height, width, 3), dtype=np.uint8)
    first[..., 2] = 40
    second = np.zeros((height, width, 3), dtype=np.uint8)
    second[..., 0] = 90
    first_path, second_path = tmp_path / "first.png", tmp_path / "second.png"
    assert cv2.imwrite(str(first_path), first)
    assert cv2.imwrite(str(second_path), second)
    depth = np.full((height, width), 1000, dtype=np.uint16)
    depth_path = tmp_path / "depth.png"
    assert cv2.imwrite(str(depth_path), depth)
    calibration = CameraIntrinsics(width, height, 20.0, 20.0, 3.5, 1.5, ())
    sources = (
        RGBDFrame(10, first_path, depth_path, 1.0, timestamp_us=10),
        RGBDFrame(20, second_path, depth_path, 1.0, timestamp_us=20),
    )
    context = V2CudaAuditExportContext(
        sources=sources,
        strips=(CudaSourceStrip(10, 0, 0, 4, 3.5), CudaSourceStrip(20, 4, 4, 4, 7.5)),
        calibration=calibration,
        renderer="torch_cuda_c1_constrained_owner_v2",
        include_adjacent_corridors=False,
    )
    panorama = np.zeros((height, width, 3), dtype=np.uint8)
    panorama[:, :4] = first[:, :4]
    panorama[:, 4:] = second[:, 4:]
    owner = np.full((height, width), 20, dtype=np.int32)
    owner[:, :4] = 10
    panorama_before, owner_before = panorama.copy(), owner.copy()

    source_export, owner_export = stage_v2_cuda_audit_exports(
        context,
        panorama_bgr=panorama,
        owner_frame_id=owner,
        central_strip_output_dir=tmp_path / ".central_strips.pending",
        owner_only_output_dir=tmp_path / ".central_strips_owner_only.pending",
    )

    assert source_export["image_count"] == 2
    assert source_export["primary_pixels_or_owner_modified"] is False
    assert owner_export["image_count"] == 2
    assert owner_export["primary_pixels_or_owner_modified"] is False
    central = cv2.imread(str(tmp_path / ".central_strips.pending" / "central_strip_0000_frame_000010.png"), cv2.IMREAD_UNCHANGED)
    owner_only = cv2.imread(str(tmp_path / ".central_strips_owner_only.pending" / "central_strip_0001_frame_000020.png"), cv2.IMREAD_UNCHANGED)
    assert central is not None and central.shape[2] == 4 and np.all(central[..., 3] == 255)
    assert owner_only is not None and owner_only.shape[2] == 4 and np.all(owner_only[..., 3] == 255)
    assert np.array_equal(panorama, panorama_before)
    assert np.array_equal(owner, owner_before)

