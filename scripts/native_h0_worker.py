#!/usr/bin/env python3
"""Isolated worker: adds only the H0 tool root, never the SDK source tree."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qualification.native_h0.campaign import worker
from qualification.native_h0.common import read

if __name__ == "__main__":
    worker(read(sys.argv[1]))
