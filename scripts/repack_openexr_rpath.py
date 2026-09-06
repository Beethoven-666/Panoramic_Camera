"""Remove the upstream OpenEXR 3.4.15 wheel's stale /tmp build RPATH; retain instructions."""

import argparse
import base64
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import zipfile


def repack(source, destination):
    if destination.exists():
        raise FileExistsError(destination)
    with zipfile.ZipFile(source) as archive:
        files = {
            name: archive.read(name)
            for name in archive.namelist()
            if not name.endswith("/")
        }
    old = "openexr-3.4.15.dist-info"
    new = "openexr-3.4.15+g305.1.dist-info"
    if old + "/METADATA" not in files:
        raise ValueError("Only exact upstream OpenEXR 3.4.15 is supported")
    evidence = []
    for name, data in list(files.items()):
        if data[:4] != b"\x7fELF":
            continue
        with tempfile.NamedTemporaryFile() as handle:
            handle.write(data)
            handle.flush()
            before = subprocess.check_output(["readelf", "-x", ".text", handle.name])
            rpath = subprocess.check_output(
                ["patchelf", "--print-rpath", handle.name], text=True
            ).strip()
            fixed = ":".join(p for p in rpath.split(":") if p.startswith("$ORIGIN"))
            subprocess.run(["patchelf", "--set-rpath", fixed, handle.name], check=True)
            after = subprocess.check_output(["readelf", "-x", ".text", handle.name])
            if before != after:
                raise ValueError("ELF instructions changed")
            files[name] = Path(handle.name).read_bytes()
            evidence.append(
                {
                    "member": name,
                    "original_rpath": rpath,
                    "rpath": fixed,
                    "text_unchanged": True,
                }
            )
    files = {
        name.replace(old, new): data
        for name, data in files.items()
        if not name.endswith("/RECORD")
    }
    files[new + "/METADATA"] = files[new + "/METADATA"].replace(
        b"Version: 3.4.15\n", b"Version: 3.4.15+g305.1\n", 1
    )
    files[new + "/g305_rpath_patch.json"] = json.dumps(
        {
            "upstream_wheel_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "patch": "remove stale absolute build RPATH; preserve $ORIGIN entries",
            "elf": evidence,
        },
        indent=2,
    ).encode()
    rows = []
    for name, data in sorted(files.items()):
        digest = (
            base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip("=")
        )
        rows.append([name, "sha256=" + digest, len(data)])
    rows.append([new + "/RECORD", "", ""])
    stream = io.StringIO(newline="")
    csv.writer(stream, lineterminator="\n").writerows(rows)
    files[new + "/RECORD"] = stream.getvalue().encode()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(files.items()):
            info = zipfile.ZipInfo(name, (2026, 9, 7, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, data)
    return evidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    print(json.dumps(repack(args.source, args.destination), indent=2))
