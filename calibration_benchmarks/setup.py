#!/usr/bin/env python
# -*- coding: utf-8 -*-
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

description = "Utilities for Benchmarking Calibration Tests in Python"

dist = setup(
    name="calibration_benchmarks",
    version="0.0.1dev0",
    description=description,
    license="BSD 3-Clause License",
    packages=[
        "calibration_benchmarks",
        "calibration_benchmarks.data_generators",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Operating System :: POSIX",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Programming Language :: Python :: Implementation :: CPython",
    ],
    python_requires=">=3.9",
)
