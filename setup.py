#!/usr/bin/env python3
"""
Xencode Setup Script

Installation script for the Xencode AI/ML leviathan system.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_path.exists():
    with open(requirements_path, 'r') as f:
        requirements = [
            line.strip() for line in f 
            if line.strip() and not line.startswith('#')
        ]

setup(
    name="xencode",
    version="2.1.0",
    description="The ultimate offline AI assistant that outperforms GitHub Copilot",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Sreevarshan",
    author_email="sreevarshan@xenoz.com",
    url="https://github.com/sreevarshan-xenoz/xencode",
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'xencode=xencode.cli:cli',
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    keywords="ai ml assistant offline ensemble rlhf ollama cli",
    project_urls={
        "Bug Reports": "https://github.com/sreevarshan-xenoz/xencode/issues",
        "Source": "https://github.com/sreevarshan-xenoz/xencode",
        "Documentation": "https://github.com/sreevarshan-xenoz/xencode/blob/main/README.md",
    },
)