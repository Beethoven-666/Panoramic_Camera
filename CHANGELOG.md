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
