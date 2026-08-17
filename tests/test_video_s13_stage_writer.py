from __future__ import annotations

from pathlib import Path

import numpy as np

from panorama_demo.video_s13_stage_writer import S13StageImageWriter, STAGE_FILENAMES


def test_writer_allows_four_pending_images_without_submit_wait(tmp_path: Path) -> None:
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    writer = S13StageImageWriter(tmp_path, max_pending=4)
    for stage in ("P0", "P1", "P2", "P3"):
        writer.submit_host_image(stage, image)
    paths = writer.close()
    assert tuple(path.name for path in paths) == tuple(STAGE_FILENAMES.values())
    assert writer.pending_peak == 4
    assert writer.submit_blocking_count == 0
    assert writer.snapshot_copy_count == 4
