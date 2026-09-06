# Third-party notices

## Linux 0.3.0rc1 distribution blockers and evidence

The project owner has not supplied a project LICENSE or EULA. Distribution remains blocked;
this document does not grant a project license. No EULA or private release key is generated.
Public base/addon bundles exclude ORB binaries and Vocabulary while its distribution plan is unresolved.
Open3D source commit is `1e7b17438687a0b0c1e5a7187321ac7044afe275`; the custom CUDA cp310 wheel
is addon-only. Pangolin external source is `aff6883c83f3fd7e8268a9715e84266c42e2efe3`.
Bundle `licenses/` retains wheel-supplied texts; SPDX records unknown expressions as NOASSERTION.
This is an inventory, not a conclusion that all transitive native components can be redistributed.

Orbbec native SDK 2.9.3 and official udev files are pinned at
`2f6561c28255d805b34aa00a690199ce40e96c81`. Their verbatim upstream LICENSE and file provenance
are in `packaging/linux/orbbec-official/`. `pyorbbecsdk2 2.1.2+g305.1` changes only distribution
metadata/RECORD from upstream 2.1.2; all runtime bytes are retained. Original metadata and wheel
digest are included in the variant's dist-info record. No native SDK upgrade is implied.

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
