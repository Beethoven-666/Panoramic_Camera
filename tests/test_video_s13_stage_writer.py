import cv2
import numpy as np

from panorama_demo.video_s13_stage_writer import S13StageImageWriter, STAGE_FILENAMES


def test_stage_writer_emits_only_named_lossless_pngs(tmp_path) -> None:
    image = np.full((5, 7, 3), 23, np.uint8)
    writer = S13StageImageWriter(tmp_path)
    for stage in STAGE_FILENAMES:
        writer.submit_host_image(stage, image)
    paths = writer.close()
    assert {path.name for path in paths} == set(STAGE_FILENAMES.values())
    assert {path.name for path in tmp_path.iterdir()} == set(STAGE_FILENAMES.values())
    for path in paths:
        assert np.array_equal(cv2.imread(str(path), cv2.IMREAD_COLOR), image)
