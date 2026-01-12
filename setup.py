"""
BenchDrift: Benchmark Drift Detection Pipeline for LLMs
"""
from setuptools import setup, find_packages

setup(
    name="benchdrift",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
)
