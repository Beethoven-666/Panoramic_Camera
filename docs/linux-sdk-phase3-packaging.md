# Linux SDK candidate archives

Build a clean commit in two independent Linux ext4 checkouts. Install the build
environment offline with `--no-index --require-hashes` and
`requirements/linux-build-lock-py310.txt`; pass the same exact build wheelhouse
to the builder. The builder checks installed tool versions and every locked wheel
SHA before building. `SOURCE_DATE_EPOCH` is required and must be the same in both
builds. `PYTHONPATH` and `PYTHONHOME` are removed from the wheel build environment.

```bash
python scripts/build_linux_sdk_bundles.py \
  --source /absolute/clean-checkout \
  --output /absolute/new-build-a \
  --wheelhouse /absolute/runtime-wheels \
  --build-wheelhouse /absolute/build-wheels \
  --open3d-wheel /absolute/private-open3d.whl \
  --runtime-manifest /absolute/runtime-variant-manifest.json

python scripts/compare_linux_sdk_builds.py \
  --build-a /absolute/new-build-a --build-b /absolute/new-build-b \
  --output /absolute/reproducibility.json
```

Both builds independently create a project wheel, a base archive, a 3-D addon
archive and a source compliance archive. Tar members have sorted names, fixed
mtime, uid/gid 0, empty owner names and stable modes; zstd uses level 6 and no
worker threads. Existing output directories and archives are never overwritten.

`release-candidate-index.json` records the SHA of the actual compressed archives.
Its `content_checksums_sha256` fields identify the final per-bundle
`checksums.sha256`, which covers every other distributed file. Inside each bundle,
the manifest's `content_checksums_file` identifies `payload-checksums.sha256`:
this covers the payload before the manifest and checksum indexes are created,
avoiding a self-referential manifest checksum. The final checksum index covers
both the payload index and the manifest. Archive SHA values are never embedded
in the archives themselves.

Use the frozen standalone `verify_linux_sdk_archive.py` next to the release index.
It requires Python 3.10 and either the `zstandard` module or the system `zstd`
command, and requires no SDK, source tree, Git metadata or `PYTHONPATH`.

```bash
env -u PYTHONPATH -u PYTHONHOME python verify_linux_sdk_archive.py \
  --candidate-index release-candidate-index.json \
  --archive gemini305-sdk-base-0.3.0rc1-ubuntu22.04-x86_64-py310-sm120.tar.zst \
  --kind base --extract-to /absolute/new-extraction \
  --output /absolute/base-verification.json
```

The verifier checks the archive SHA and all member paths before extracting any
member. It verifies every extracted checksum and rejects undeclared files before
publishing the extraction directory. Only regular files and directories are
distributed. Use `three_d_addon` and `source_compliance` for the other kinds.
Run acceptance tools from the base bundle's `acceptance-tools/` using the installed
SDK interpreter. Do not edit these tools or their archived copies during acceptance.

The builder emits `RC_BUNDLE_BUILT_PENDING_VERIFICATION`. The phase 3 completion
report may issue `RC_BUNDLE_FROZEN` only after reproducibility, installation,
relocation, replay, uninstall and tamper checks pass. This report is external to
the immutable candidate; do not rewrite its release index or bundle manifests.
All candidate files remain `UNSIGNED`, with software, hardware and release
qualification false. Only the separate software acceptance aggregator can issue
software qualification after the required recorded suites.
