#!/usr/bin/env python3
"""Verify addon wheel members without importing Open3D or executing the SDK."""
import argparse
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import sys
import sysconfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qualification.native_h0.binding import installed_wheel_identity
from qualification.native_h0.common import file_identity, require


def inspect(open3d_wheel, project_wheel, base_prefix):
    base_prefix = Path(base_prefix).resolve()
    project = installed_wheel_identity(project_wheel, "gemini305-rgbd-panorama", allowed_prefix=base_prefix)
    spec = importlib.util.find_spec("panorama_demo")
    require(spec and Path(spec.origin).resolve().is_relative_to(base_prefix), "Addon imports an unbound SDK")
    result = installed_wheel_identity(open3d_wheel, "open3d")
    result["installed_project"] = project
    if Path(sys.prefix).resolve() != base_prefix:
        link = Path(sysconfig.get_paths()["purelib"]) / "g305_base.pth"
        distribution = importlib.metadata.distribution("gemini305-rgbd-panorama")
        base_lib = Path(distribution.locate_file("")).resolve()
        require(base_lib.is_relative_to(base_prefix) and link.read_text().strip() == str(base_lib),
                "Addon .pth does not point to the verified base installation")
        result["base_pth"] = file_identity(link)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("open3d-wheel", "project-wheel", "base-prefix"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(inspect(args.open3d_wheel, args.project_wheel, args.base_prefix)))


if __name__ == "__main__":
    main()
