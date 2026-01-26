#!/usr/bin/env python3
"""
Xencode TUI Entry Point

Launch the Xencode Terminal User Interface.
"""

import sys
from pathlib import Path
from xencode.tui.app import run_tui


def main():
    """Main entry point for xencode tui command"""
    # Get root path from args or use current directory
    root_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
    
    print(f"Starting Xencode TUI...")
    print(f"Root: {root_path}")
    print()
    
    run_tui(root_path=root_path)


if __name__ == "__main__":
    main()
