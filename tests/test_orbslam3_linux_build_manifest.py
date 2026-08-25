from __future__ import annotations

from pathlib import Path


def test_linux_orb_build_is_pinned_and_external_to_wheel() -> None:
    root = Path(__file__).resolve().parents[1]
    script = (root / "scripts" / "build_orbslam3_linux.sh").read_text(encoding="utf-8")
    patch = root / "scripts" / "patches" / "orbslam3-g305-headless-runners.patch"

    assert "4452a3c4ab75b1cde34e5505a36ec3f9edcdc4c4" in script
    assert "aff6883c83f3fd7e8268a9715e84266c42e2efe3" in script
    assert "G305_BUILD_JOBS:-2" in script
    assert "git -C \"$ORB_SOURCE\" apply --check" in script
    assert "runtime-manifest.json" in script
    assert "rgbd_tum_headless.cc" in patch.read_text(encoding="utf-8")
    assert (root / "third_party" / "orbslam3-g305-runner" / "COPYING").is_file()
