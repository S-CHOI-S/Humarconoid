"""Installation script for the 'humarcscripts' python package."""

import os
import toml

# Installation operation
from setuptools import find_packages, setup

setup(
    name="humarcscripts",
    version="0.1.0",
    packages=["humarcscripts"],
    package_dir={"humarcscripts": "scripts"},
    author="Sol Choi",
    author_email="solchoi@yonsei.ac.kr",
    maintainer="Sol Choi",
    maintainer_email="solchoi@yonsei.ac.kr",
    description="Scripts and utilities for humarcscripts",
    url="https://github.com/S-CHOI-S/Humarconoid.git",
    install_requires=[],
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    include_package_data=True,
    zip_safe=False,
)
