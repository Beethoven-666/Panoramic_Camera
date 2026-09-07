#!/usr/bin/env python3
"""Run the external native H0 orchestrator under the installed candidate Python."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qualification.native_h0.campaign import main

if __name__ == "__main__":
    main()
