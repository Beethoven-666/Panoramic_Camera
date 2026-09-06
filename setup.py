"""Compatibility shim; all project metadata lives in pyproject.toml."""
from setuptools import setup
from pathlib import Path
from runpy import run_path

setup(cmdclass={"build_py": run_path(str(Path(__file__).with_name("build_support.py")))["BuildPy"]})
