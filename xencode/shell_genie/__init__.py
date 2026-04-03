"""
Shell Genie package for Xencode.

Natural language to shell command generation with safety guards.
"""

from .genie import ShellGenie, CommandSafety

__all__ = [
    "ShellGenie",
    "CommandSafety",
]
