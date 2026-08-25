# Third-party notices

- `third_party/UniStitch` is pinned to commit
  `78ebe7c07d516c591810337475ccdd4f2beff384` and is licensed under
  Apache-2.0. Project: <https://github.com/MmelodYy/UniStitch>.
- `third_party/LightGlue` is pinned to commit
  `746fac2c042e05d1865315b1413419f1c1e7ba55` and is licensed under
  Apache-2.0. Project: <https://github.com/cvg/LightGlue>.
- `pyorbbecsdk2` is the official Orbbec Python SDK v2 wrapper and is licensed
  under Apache-2.0. Project: <https://github.com/orbbec/pyorbbecsdk>.

The UniStitch checkpoint is not redistributed by this demo. The official
Hugging Face model card currently labels the weights as `license: other` and
does not include a license text. `scripts/download_unistitch_weights.py`
downloads the checkpoint directly from the authors' model repository. Obtain
license clarification from the authors before commercial use or redistribution.

## ORB-SLAM3 Gemini 305 headless runners

`scripts/patches/orbslam3-g305-headless-runners.patch` is a GPL-3.0-or-later
derived patch against ORB-SLAM3 commit
`4452a3c4ab75b1cde34e5505a36ec3f9edcdc4c4`. ORB-SLAM3 is maintained by the
University of Zaragoza and distributed under GPLv3; commercial licensing is
available from its authors.

The patch and resulting external executable are not included in the Python
wheel. Commercial distribution remains blocked until an applicable GPL
distribution plan or commercial license is approved.
