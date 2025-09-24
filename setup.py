#!/usr/bin/env python3
"""Setup configuration for Xencode package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
long_description = (
    readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""
)

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_path.exists():
    requirements = requirements_path.read_text().strip().split('\n')
    requirements = [
        req.strip() for req in requirements if req.strip() and not req.startswith('#')
    ]

setup(
    name="xencode",
    version="1.0.0",
    author="Sreevarshan",
    author_email="sreevarshan@xenoz.dev",
    description="Professional offline-first AI assistant with Claude-style interface",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sreevarshan-xenoz/xencode",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: User Interfaces",
        "Topic :: Terminals",
    ],
    python_requires=">=3.6",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pytest-mock>=3.10.0",
            "mypy>=1.0.0",
            "ruff>=0.1.0",
            "black>=23.0.0",
            "pre-commit>=3.0.0",
            "types-requests>=2.25.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "xencode=xencode_core:main",
        ],
    },
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "mypy>=1.0.0",
            "ruff>=0.1.0",
            "black>=23.0.0",
            "pre-commit>=3.0.0",
            "bandit>=1.7.0",
        ],
        "test": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "mypy>=1.0.0",
        ],
        "docs": [],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.sh"],
    },
    project_urls={
        "Bug Reports": "https://github.com/sreevarshan-xenoz/xencode/issues",
        "Source": "https://github.com/sreevarshan-xenoz/xencode",
    },
)
