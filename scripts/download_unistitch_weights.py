from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path


URL = "https://huggingface.co/Y5Y/UniStitch_model/resolve/main/epoch_best_model.pth"
EXPECTED_SIZE = 336_715_309
# Hugging Face displays both an Xet object hash and the downloaded file's
# SHA-256.  They are different values; this is the SHA-256 of the .pth bytes.
EXPECTED_SHA256 = "c7c4184c3ec63e15ed483f7066afdd4ed2fcd12f1178ae27183c9838f9083c19"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(destination: Path, force: bool = False) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not force:
        current = sha256_file(destination)
        if current == EXPECTED_SHA256:
            print(f"Model already verified: {destination}")
            return
        raise RuntimeError(
            f"Existing model has unexpected SHA-256: {current}. "
            "Use --force to replace it."
        )

    temporary = destination.with_suffix(destination.suffix + ".partial")
    if temporary.exists():
        temporary.unlink()

    print(f"Downloading UniStitch model ({EXPECTED_SIZE / 1024**2:.1f} MiB)...")

    def report(blocks: int, block_size: int, total: int) -> None:
        known_total = total if total > 0 else EXPECTED_SIZE
        received = min(blocks * block_size, known_total)
        percent = 100.0 * received / known_total
        print(f"\r{percent:6.2f}%  {received / 1024**2:7.1f}/{known_total / 1024**2:.1f} MiB", end="")

    urllib.request.urlretrieve(URL, temporary, reporthook=report)
    print()
    if temporary.stat().st_size != EXPECTED_SIZE:
        temporary.unlink(missing_ok=True)
        raise RuntimeError("Downloaded model size does not match the published file size")
    digest = sha256_file(temporary)
    if digest != EXPECTED_SHA256:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Downloaded model SHA-256 mismatch: {digest}")
    temporary.replace(destination)
    print(f"Verified model: {destination}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and verify the official UniStitch weights")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/unistitch/epoch_best_model.pth"),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    try:
        download(args.output.resolve(), args.force)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
