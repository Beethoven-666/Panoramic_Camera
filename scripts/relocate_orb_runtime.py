"""Prepare an internal ORB Runtime with local dependencies and relative RPATH."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess


def command(*args):
    env = {name: value for name, value in os.environ.items() if name != "LD_LIBRARY_PATH"}
    return subprocess.check_output(args, text=True, env=env, stderr=subprocess.STDOUT).strip()


def elf_files(root):
    result = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            with path.open("rb") as handle:
                if handle.read(4) == b"\x7fELF":
                    result.append(path)
    return result


def dependencies(path):
    output = command("ldd", str(path))
    if "not found" in output:
        raise RuntimeError(f"Unresolved ELF dependency: {path}\n{output}")
    found = {}
    for line in output.splitlines():
        match = re.search(r"^\s*(\S+) => (/\S+)", line)
        if match:
            found[match[1]] = Path(match[2])
        elif line.strip().startswith("/"):
            dependency = Path(line.split()[0])
            found[dependency.name] = dependency
    return found, output


def is_system(path):
    return any(path.is_relative_to(prefix) for prefix in (Path("/lib"), Path("/lib64"), Path("/usr/lib")))


def scan(root: Path, patchelf: str):
    root = root.resolve()
    records = []
    system = set()
    for path in elf_files(root):
        rpath = command(patchelf, "--print-rpath", str(path))
        if not rpath or any(not entry.startswith("$ORIGIN") for entry in rpath.split(":")):
            raise RuntimeError(f"Runtime requires relative RPATH: {path}: {rpath}")
        resolved, raw = dependencies(path)
        for name, dependency in resolved.items():
            if not dependency.resolve().is_relative_to(root):
                if not is_system(dependency) or str(dependency).startswith("/usr/local"):
                    raise RuntimeError(f"Undeclared external dependency: {name}: {dependency}")
                system.add(str(dependency))
        records.append({"path": str(path.relative_to(root)), "file": command("file", "-b", str(path)),
            "readelf_dynamic": command("readelf", "-d", str(path)), "rpath": rpath,
            "ldd": raw, "resolved": {name: str(value) for name, value in resolved.items()}})
    if not records:
        raise ValueError("ORB Runtime has no ELF files")
    return {"schema": "gemini305-elf-dependencies/v1", "status": "PASS", "root": str(root),
            "ld_library_path_used": False, "elf": records,
            "declared_ubuntu22_04_system_libraries": sorted(system)}


def prepare(source: Path, output: Path, patchelf: str):
    source, output = source.resolve(), output.resolve()
    shutil.copytree(source, output)
    (output / "lib").mkdir(exist_ok=True)
    queued = list(elf_files(source))
    copied = set()
    origins = {}
    while queued:
        original = queued.pop()
        if original.resolve() in copied:
            continue
        copied.add(original.resolve())
        # Flatten every shared library, including ORB's DBoW2/g2o libraries.
        if ".so" in original.name:
            destination = output / "lib" / original.name
            if not destination.exists():
                shutil.copy2(original, destination)
            elif destination.read_bytes() != original.read_bytes():
                raise RuntimeError(f"Conflicting library basename: {original.name}")
            origins[original.name] = str(original)
        resolved, _ = dependencies(original)
        for dependency in resolved.values():
            if not is_system(dependency):
                queued.append(dependency)
    code_evidence = {}
    for path in elf_files(output):
        # .text stays unchanged; only dynamic-link metadata is rewritten.
        before = command("readelf", "-x", ".text", str(path))
        for needed in command(patchelf, "--print-needed", str(path)).splitlines():
            if "/" in needed:
                command(patchelf, "--replace-needed", needed, Path(needed).name, str(path))
        relative = os.path.relpath(output / "lib", path.parent)
        rpath = "$ORIGIN" if relative == "." else "$ORIGIN/" + relative
        command(patchelf, "--set-rpath", rpath, str(path))
        after = command("readelf", "-x", ".text", str(path))
        if before != after:
            raise RuntimeError(f"Relocation modified executable instructions: {path}")
        code_evidence[str(path.relative_to(output))] = hashlib.sha256(after.encode()).hexdigest()
    report = scan(output, patchelf)
    manifest = {"schema": "gemini305-orb-runtime-portability/v1", "internal_only": True,
        "public_distribution_approved": False,
        "orbslam3_commit": "4452a3c4ab75b1cde34e5505a36ec3f9edcdc4c4",
        "pangolin_commit": "aff6883c83f3fd7e8268a9715e84266c42e2efe3",
        "headless_choice": "Retain the validated System library and its Pangolin ABI; no Viewer is started by the headless runner",
        "library_origins": origins, "text_sections_sha256": code_evidence,
        "files": {str(path.relative_to(output)): {"bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in elf_files(output)}}
    (output / "orb-runtime-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "elf-dependency-report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--scan-only", action="store_true")
    parser.add_argument("--patchelf", default="patchelf")
    args = parser.parse_args()
    if args.scan_only:
        result = scan(args.source, args.patchelf)
        print(json.dumps(result, indent=2))
    else:
        if args.output is None:
            parser.error("--output is required")
        result = prepare(args.source, args.output, args.patchelf)
        print(json.dumps({"status": result["status"], "elf_count": len(result["elf"]), "root": result["root"]}))


if __name__ == "__main__":
    main()
