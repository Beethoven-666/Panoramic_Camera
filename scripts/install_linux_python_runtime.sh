#!/usr/bin/env bash
set -euo pipefail

PYTHON=${G305_PYTHON:-python}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)

"$PYTHON" -m pip install -r "$REPO_ROOT/requirements/linux-l0-constraints.txt"

# pyorbbecsdk2 2.1.x declares an exact Open3D 0.18 dependency.  The product
# runtime deliberately supplies its separately built Open3D 0.19 CUDA wheel,
# so install only the wrapper distribution here.  Its runtime dependencies
# (including av) are pinned explicitly in the requirements file above. The
# same file also owns the custom Open3D wheel's Python-only dependencies.
"$PYTHON" -m pip install --no-deps "pyorbbecsdk2==2.1.2"
