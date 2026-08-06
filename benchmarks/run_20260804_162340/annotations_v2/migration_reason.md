# Annotation v2 migration

This is a new, validation-only measurement input.  The original `annotations/`
directory remains immutable and is not rewritten or reinterpreted.

The human-confirmed frame-249 and frame-257 geometry is copied exactly.  v2
adds only read-only evaluator roles:

- `compact_foreground_single_owner` for carton, fan, and cable;
- `extended_background_structure` for the full yellow beam;
- `long_line` for beam/table straight-line measurements; and
- `safe_background` for the beam-face photometric measurement.

The extended beam role removes only its whole-region single-owner requirement.
Its internal-seam and handoff checks remain strict, as do all line and safe
background hard gates.  Roles do not select sources, change poses, alter
rendering, modify a quality threshold, or make a holdout annotation.
