"""Build a traceable dependency-metadata variant; preserve every runtime byte."""
from __future__ import annotations

import argparse
import base64
import csv
from email import policy
from email.parser import BytesParser
import hashlib
import io
import json
from pathlib import Path
import zipfile

UPSTREAM_VERSION = "2.1.2"
LOCAL_VERSION = "2.1.2+g305.1"
REQUIRES = ("av==12.3.0", "numpy==2.2.6", "opencv-python-headless==4.14.0.94")


def repack(source: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / source.name.replace(f"-{UPSTREAM_VERSION}-", f"-{LOCAL_VERSION}-", 1)
    if target.exists() or target.name == source.name:
        raise ValueError("Expected an upstream 2.1.2 wheel and a new output filename")
    with zipfile.ZipFile(source) as archive:
        entries = {name: archive.read(name) for name in archive.namelist()}
    metadata_name = next(name for name in entries if name.endswith(".dist-info/METADATA"))
    metadata = BytesParser(policy=policy.compat32).parsebytes(entries[metadata_name])
    if metadata["Name"] != "pyorbbecsdk2" or metadata["Version"] != UPSTREAM_VERSION:
        raise ValueError("This patch applies only to pyorbbecsdk2 2.1.2")
    original_requires = metadata.get_all("Requires-Dist", [])
    del metadata["Requires-Dist"]
    metadata.replace_header("Version", LOCAL_VERSION)
    for requirement in REQUIRES:
        metadata["Requires-Dist"] = requirement
    original_info = metadata_name.split("/")[0]
    info = original_info.replace(UPSTREAM_VERSION, LOCAL_VERSION)
    rewritten = {}
    for name, content in entries.items():
        if name.endswith(".dist-info/RECORD"):
            continue
        new_name = name.replace(original_info + "/", info + "/", 1)
        rewritten[new_name] = metadata.as_bytes() if name == metadata_name else content
    runtime_files = {name: hashlib.sha256(value).hexdigest() for name, value in entries.items()
                     if not name.startswith(original_info + "/")}
    audit = {"schema": "gemini305-orbbec-metadata-variant/v1", "upstream_version": UPSTREAM_VERSION,
             "local_version": LOCAL_VERSION, "upstream_wheel_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
             "original_requires": original_requires, "requires": REQUIRES,
             "native_binary_modified": False, "runtime_files_sha256": runtime_files,
             "reason": "Pinned headless production dependencies; upstream sample GUI/Open3D dependencies are excluded",
             "hardware_qualified": False}
    rewritten[info + "/g305_metadata_patch.json"] = json.dumps(audit, indent=2).encode()
    record = io.StringIO(newline="")
    writer = csv.writer(record, lineterminator="\n")
    for name, content in sorted(rewritten.items()):
        digest = base64.urlsafe_b64encode(hashlib.sha256(content).digest()).rstrip(b"=").decode()
        writer.writerow((name, "sha256=" + digest, len(content)))
    writer.writerow((info + "/RECORD", "", ""))
    rewritten[info + "/RECORD"] = record.getvalue().encode()
    with zipfile.ZipFile(target, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in sorted(rewritten.items()):
            entry = zipfile.ZipInfo(name, date_time=(2026, 9, 7, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            entry.external_attr = 0o644 << 16
            archive.writestr(entry, content)
    with zipfile.ZipFile(target) as archive:
        assert all(hashlib.sha256(archive.read(name)).hexdigest() == sha for name, sha in runtime_files.items())
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(repack(args.input, args.output_dir))
