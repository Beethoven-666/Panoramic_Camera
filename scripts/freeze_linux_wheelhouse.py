"""Freeze the installed, verified CPython 3.10 dependency graph into wheel locks."""
from __future__ import annotations

import argparse
from importlib.metadata import distribution
import hashlib
from pathlib import Path
import subprocess
import sys

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name, parse_wheel_filename

BASE = ("numpy", "opencv-python-headless", "OpenEXR", "PyYAML", "xatlas", "psutil",
        "av", "cupy-cuda13x", "nvidia-cublas")
ADDON = ("configargparse", "dash", "flask", "nbformat", "plotly", "werkzeug")


def installed_closure(seeds):
    found = {}
    todo = list(seeds)
    while todo:
        name = canonicalize_name(todo.pop())
        if name in found:
            continue
        item = distribution(name)
        found[name] = item.version
        for text in item.requires or ():
            requirement = Requirement(text)
            if requirement.marker is None or requirement.marker.evaluate({"extra": ""}):
                actual = distribution(requirement.name).version
                if actual not in requirement.specifier:
                    raise ValueError(f"Installed dependency conflict: {name}: {text}, found {actual}")
                todo.append(requirement.name)
    return found


def write_lock(wheels: Path, target: Path):
    lines = []
    names = set()
    for wheel in sorted(wheels.glob("*.whl")):
        name, version, _, _ = parse_wheel_filename(wheel.name)
        if name in names:
            raise ValueError(f"Multiple wheels for {name}")
        names.add(name)
        lines.append(f"{name}=={version} --hash=sha256:{hashlib.sha256(wheel.read_bytes()).hexdigest()}")
    target.write_text("\n".join(lines) + "\n", encoding="ascii")
    target.with_suffix(".sha256").write_text(
        hashlib.sha256(target.read_bytes()).hexdigest() + "  " + target.name + "\n", encoding="ascii")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("base", "addon"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 10):
        raise SystemExit("The frozen Runtime requires CPython 3.10")
    pins = installed_closure(BASE if args.variant == "base" else ADDON)
    if args.variant == "addon":
        base = installed_closure(BASE)
        pins = {name: version for name, version in pins.items() if name not in base}
    args.output.mkdir(parents=True, exist_ok=False)
    wheels = args.output / "wheels"
    wheels.mkdir()
    subprocess.run([sys.executable, "-m", "pip", "download", "--only-binary=:all:", "--no-deps",
                    "--dest", str(wheels), *[f"{name}=={version}" for name, version in sorted(pins.items())]], check=True)
    write_lock(wheels, args.output / "requirements.txt")


if __name__ == "__main__":
    main()
