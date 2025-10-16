#!/usr/bin/env python3
"""
Xencode CLI Entry Point

Main entry point for the Xencode AI/ML leviathan command-line interface.
"""

import sys
from pathlib import Path

# Add xencode to Python path
sys.path.insert(0, str(Path(__file__).parent))

from xencode.cli import cli

if __name__ == '__main__':
    cli()