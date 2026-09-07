"""Offline, owned-prefix installer. Does not install drivers, udev or system packages."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import uuid


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024**2), b""):
            h.update(block)
    return h.hexdigest()


def verify_checksums(bundle):
    path = bundle / "checksums.sha256"
    covered = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        sha, relative = line.split("  ", 1)
        target = (bundle / relative).resolve()
        target.relative_to(bundle.resolve())
        if not target.is_file() or digest(target) != sha:
            raise ValueError(f"Bundle checksum mismatch: {relative}")
        covered.add(relative)
    sidecars = {"checksums.sha256", "checksums.sha256.asc", "checksums.sha256.TEST_ONLY.asc"}
    for target in bundle.rglob("*"):
        relative = target.relative_to(bundle).as_posix()
        if target.is_file() and relative not in covered | sidecars:
            raise ValueError(f"Undeclared bundle file: {relative}")
    return digest(path)


def platform_errors(facts):
    errors = []
    if facts["os"] != "ubuntu" or facts["release"] != "22.04" or facts["machine"] != "x86_64":
        errors.append("UNSUPPORTED_PLATFORM")
    if facts["implementation"] != "cpython" or tuple(facts["python"]) != (3, 10):
        errors.append("UNSUPPORTED_PYTHON_ABI")
    if tuple(map(int, facts["glibc"].split("."))) < (2, 35):
        errors.append("UNSUPPORTED_GLIBC")
    if facts["gpu_compute_capabilities"] != ["12.0"]:
        errors.append("UNSUPPORTED_GPU_VARIANT")
    return errors


def host_facts():
    release = {}
    for line in Path("/etc/os-release").read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            release[key] = value.strip('"')
    smi = shutil.which("nvidia-smi") or "/usr/lib/wsl/lib/nvidia-smi"
    result = subprocess.check_output([smi, "--query-gpu=compute_cap", "--format=csv,noheader"], text=True)
    return {"os": release.get("ID"), "release": release.get("VERSION_ID"),
            "machine": platform.machine(), "implementation": sys.implementation.name,
            "python": list(sys.version_info[:2]), "glibc": platform.libc_ver()[1],
            "gpu_compute_capabilities": result.strip().splitlines(),
            "wsl_development_environment": "microsoft" in platform.release().lower()}


def run(command, log, *, cwd):
    env = {key: value for key, value in os.environ.items() if key not in {"PYTHONPATH", "PYTHONHOME"}}
    env["G305_CUDA"] = "required"
    log.write(("RUN " + repr(command) + "\n").encode())
    log.flush()
    subprocess.run(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)


def rewrite_prefix(venv, old, new):
    for path in (venv / "bin").iterdir():
        if not path.is_file() or path.is_symlink():
            continue
        content = path.read_bytes()
        if b"\0" not in content[:4096]:
            path.write_bytes(content.replace(str(old).encode(), str(new).encode()))


def inventory(root):
    paths = list(root.rglob("*"))
    return {"files": sorted(str(p.relative_to(root)) for p in paths if p.is_file() or p.is_symlink()),
            "directories": sorted((str(p.relative_to(root)) for p in paths if p.is_dir() and not p.is_symlink()),
                                  key=lambda value: len(Path(value).parts), reverse=True)}


def remove_owned(root, marker_name):
    marker = root / marker_name
    if not marker.is_file():
        return {"status": "ALREADY_UNINSTALLED", "user_files_preserved": root.exists()}
    saved = json.loads(marker.read_text())
    if saved.get("schema") != "gemini305-owned-install/v1" or saved.get("prefix") != str(root):
        raise ValueError("Installation ownership manifest does not match prefix")
    for relative in saved["owned"]["files"]:
        path = root / relative
        path.parent.resolve().relative_to(root.resolve())
        if path.is_file() or path.is_symlink():
            path.unlink()
    for relative in saved["owned"]["directories"]:
        path = root / relative
        path.resolve().relative_to(root.resolve())
        try:
            path.rmdir()
        except OSError:
            pass  # User-added data is deliberately retained.
    marker.unlink()
    try:
        root.rmdir()
    except OSError:
        pass
    return {"status": "UNINSTALLED", "user_files_preserved": root.exists()}


def install(args):
    bundle = Path(__file__).resolve().parent
    prefix = args.prefix.expanduser().absolute()
    if len(prefix.parts) < 3 or prefix.is_symlink():
        raise ValueError("Use a dedicated absolute installation prefix")
    if args.action == "uninstall-addon":
        return remove_owned(prefix / "addon", "addon-install-manifest.json")
    if args.action == "uninstall":
        if (prefix / "addon/addon-install-manifest.json").exists():
            raise ValueError("Uninstall the SDK addon before its base environment")
        return remove_owned(prefix, "sdk-install-manifest.json")
    checksum = verify_checksums(bundle)
    manifest = json.loads((bundle / "manifests/bundle-manifest.json").read_text())
    facts = host_facts()
    errors = platform_errors(facts)
    if errors:
        raise ValueError(",".join(errors))
    if manifest["runtime_variant"] != "ubuntu22.04-x86_64-py310-sm120":
        raise ValueError("UNSUPPORTED_RUNTIME_VARIANT")
    addon = args.action == "install-addon"
    if manifest["kind"] != ("3d-addon" if addon else "base"):
        raise ValueError("Bundle kind does not match installer action")
    base = prefix
    if addon:
        base_marker = base / "sdk-install-manifest.json"
        if not base_marker.is_file():
            raise ValueError("Install the matching base bundle before the addon")
        installed_base = json.loads(base_marker.read_text())
        if any(installed_base[key] != manifest[key] for key in
               ("sdk_version", "source_commit", "runtime_variant")):
            raise ValueError("Addon and base source or runtime identities differ")
        prefix = base / "addon"
    marker_name = "addon-install-manifest.json" if addon else "sdk-install-manifest.json"
    existing = prefix / marker_name
    if prefix.exists():
        if existing.is_file() and json.loads(existing.read_text())["bundle_checksums_sha256"] == checksum:
            return {"status": "ALREADY_INSTALLED", "prefix": str(prefix)}
        raise ValueError("Prefix exists and is not this exact installation; use a new prefix")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    required = sum(p.stat().st_size for p in (bundle / "wheels").glob("*.whl")) * 3 + 2 * 1024**3
    if shutil.disk_usage(prefix.parent).free < required:
        raise ValueError("INSUFFICIENT_INSTALL_DISK_SPACE")
    pending = prefix.parent / ("." + prefix.name + ".install-" + uuid.uuid4().hex)
    pending.mkdir()
    published = False
    try:
        with (pending / "install.log").open("xb") as log:
            run([args.python, "-m", "venv", str(pending / "venv")], log, cwd=pending)
            python = pending / "venv/bin/python"
            if addon:
                purelib = subprocess.check_output([str(python), "-c", "import sysconfig;print(sysconfig.get_paths()['purelib'])"], text=True).strip()
                base_lib = subprocess.check_output([str(base / "venv/bin/python"), "-c", "import sysconfig;print(sysconfig.get_paths()['purelib'])"], text=True).strip()
                # The addon sees the immutable base packages; the base never
                # imports addon packages during capture or authoritative 2-D.
                (Path(purelib) / "g305_base.pth").write_text(base_lib + "\n")
                locks = ("addon-lock-py310.txt", "open3d-lock-py310.txt")
            else:
                locks = ("linux-runtime-lock-py310.txt", "orbbec-wrapper-lock-py310.txt", "project-lock.txt")
            for lock in locks:
                run([str(python), "-m", "pip", "install", "--no-index", "--find-links", str(bundle / "wheels"),
                     "--require-hashes", "--no-deps", "-r", str(bundle / "requirements" / lock)], log, cwd=pending)
            run([str(python), "-m", "pip", "check"], log, cwd=pending)
            if addon:
                run([str(python), str(bundle / "addon_smoke.py"), "--output", str(pending / "addon-smoke.json")], log, cwd=pending)
            else:
                run([str(python), "-m", "panorama_demo.sdk_doctor", "--json", "--output", str(pending / "doctor-prepublish.json"),
                     "--orb-runtime-root", str(pending / "no-orb")], log, cwd=pending)
                for entry in ("g305-capture", "g305-video-live", "g305-video-panorama", "g305-video-post-3d", "g305-orbslam3-trajectory"):
                    run([str(pending / "venv/bin" / entry), "--help"], log, cwd=pending)
            rewrite_prefix(pending / "venv", pending, prefix)
        os.replace(pending, prefix)
        published = True
        with (prefix / "relocation-smoke.log").open("xb") as log:
            run([str(prefix / "venv/bin/python"), "-c", "import panorama_demo;print(panorama_demo.__file__,panorama_demo.__version__)"], log, cwd=prefix)
        saved = {"schema": "gemini305-owned-install/v1", "prefix": str(prefix), "kind": manifest["kind"],
                 "sdk_version": manifest["sdk_version"], "source_commit": manifest["source_commit"],
                 "bundle_checksums_sha256": checksum, "runtime_variant": manifest["runtime_variant"],
                 "python": str(prefix / "venv/bin/python"), "host": facts,
                 "software_ready": False, "hardware_qualified": False, "release_ready": False,
                 "owned": inventory(prefix)}
        with (prefix / marker_name).open("x") as handle:
            json.dump(saved, handle, indent=2)
        return {"status": "INSTALLED", "prefix": str(prefix), "python": saved["python"]}
    except BaseException as exc:
        failed = prefix.parent / (prefix.name + ".install-failed-" + uuid.uuid4().hex)
        survivor = prefix if published else pending
        if survivor.exists():
            os.replace(survivor, failed)
        (failed / "failure.json").write_text(json.dumps({"type":type(exc).__name__,"message":str(exc),
                                                        "final_prefix_published":False}, indent=2))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--offline", action="store_true", help="All installations are offline")
    parser.add_argument("--action", choices=("install", "uninstall", "install-addon", "uninstall-addon"), default="install")
    args = parser.parse_args()
    # Use exactly the explicitly selected bootstrap interpreter.
    selected = subprocess.check_output([args.python, "-c", "import sys;print(sys.executable)"], text=True).strip()
    if Path(selected).resolve() != Path(sys.executable).resolve():
        raise SystemExit("Run this installer with the --python interpreter")
    try:
        print(json.dumps(install(args), indent=2))
    except Exception as exc:
        print(json.dumps({"status":"FAIL","type":type(exc).__name__,"message":str(exc)}), file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
