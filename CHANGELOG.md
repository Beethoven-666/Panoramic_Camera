# Changelog

All notable changes to `gemini305-rgbd-panorama` are documented here. The project follows Semantic Versioning.

## [0.2.0] - 2026-07-28

### Added

- Public Python SDK: `PanoramaSDK`, `SDKConfig`, `PanoramaResult`, `SessionSummary` and typed SDK exceptions.
- Explicit optional CUDA policies: `prefer`, `auto`, `off` and `required`.
- Strict SDK input validation, result loading and deterministic demo-session generation.
- Complete SDK quick start, API reference and executable integration examples.

### Changed

- Package version is now sourced from `src/panorama_demo/version.py` and read dynamically by package metadata.

## [0.1.0]

### Added

- Initial fail-closed Gemini 305 RGB-D panorama pipeline.
# 0.3.0rc1 Linux SDK candidate — phases 0–2

- Durable typed jobs/results, stable errors, cooperative cancellation, local flock camera ownership,
  disk protection, committed-prefix salvage, separate emergency output and crash-safe retention.
- Online frozen P0 reads canonical decoded committed JPEGs; bounded memory and independent Preview.
  Formal P0–P3/owner/valid/provenance/schedule/seam/M6 equivalence is required.
- CPython 3.10 root metadata, headless base, exact offline locks, isolated CUDA Open3D addon,
  relative ORB RPATH tooling and external-only unlicensed ORB distribution boundary.
- Source-bound manifests/SPDX/checksums/signing entry, capability-only doctor, stricter L1/H0 raw
  evidence tools. No native H0 or software/hardware/release qualification is issued in phase 2.
