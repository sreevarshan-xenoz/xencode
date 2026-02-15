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


def main() -> None:
    """Run the canonical Click CLI entrypoint."""
    cli.main(prog_name="xencode")


if __name__ == '__main__':
    main()