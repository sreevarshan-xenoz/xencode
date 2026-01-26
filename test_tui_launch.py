#!/usr/bin/env python3
"""Test TUI launch to capture any errors"""

import sys
import traceback
from pathlib import Path

try:
    from xencode.tui.app import run_tui
    print("✓ Imports successful")
    
    # Try to initialize the app
    print("✓ Starting TUI...")
    run_tui(root_path=Path.cwd())
    
except Exception as e:
    print(f"✗ Error: {e}")
    print("\nFull traceback:")
    traceback.print_exc()
    sys.exit(1)
