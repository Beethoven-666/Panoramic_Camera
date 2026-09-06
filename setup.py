"""Compatibility shim; all project metadata lives in pyproject.toml."""
from setuptools import setup
from build_support import BuildPy

setup(cmdclass={"build_py": BuildPy})
