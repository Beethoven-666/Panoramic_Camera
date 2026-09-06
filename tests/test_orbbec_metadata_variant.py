import importlib.util
import json
from pathlib import Path
import zipfile


def test_wrapper_metadata_patch_preserves_native_runtime_bytes(tmp_path):
    path = Path(__file__).resolve().parents[1] / "scripts/repack_orbbec_metadata.py"
    spec = importlib.util.spec_from_file_location("g305_repack", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = tmp_path / "pyorbbecsdk2-2.1.2-cp310-cp310-manylinux_2_17_x86_64.whl"
    with zipfile.ZipFile(original, "w") as archive:
        archive.writestr("pyorbbecsdk/native.so", b"unchanged native ELF")
        archive.writestr("pyorbbecsdk/__init__.py", b"upstream wrapper")
        archive.writestr("pyorbbecsdk2-2.1.2.dist-info/METADATA",
            "Metadata-Version: 2.1\nName: pyorbbecsdk2\nVersion: 2.1.2\nRequires-Dist: open3d==0.18.0\n\nOriginal description")
        archive.writestr("pyorbbecsdk2-2.1.2.dist-info/RECORD", "")
    output = module.repack(original, tmp_path / "patched")
    with zipfile.ZipFile(output) as archive:
        assert archive.read("pyorbbecsdk/native.so") == b"unchanged native ELF"
        info = "pyorbbecsdk2-2.1.2+g305.1.dist-info/"
        metadata = archive.read(info + "METADATA").decode()
        assert "Requires-Dist: open3d" not in metadata
        assert "opencv-python-headless==4.14.0.94" in metadata
        assert "Original description" in metadata
        assert json.loads(archive.read(info + "g305_metadata_patch.json"))["native_binary_modified"] is False
