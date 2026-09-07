"""
Xencode TUI Package

VS Code-like Terminal User Interface for Xencode.
"""

try:
    from .app import XencodeApp, run_tui
except ImportError:
    XencodeApp = None
    run_tui = None

__all__ = ["XencodeApp", "run_tui"]
