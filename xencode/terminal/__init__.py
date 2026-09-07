"""
Terminal package for Xencode.

Provides terminal-related components including visual workflow building.
"""

try:
    from .visual_workflow_builder import TerminalWorkflowBuilder
except ImportError:
    TerminalWorkflowBuilder = None

__all__ = [
    "TerminalWorkflowBuilder",
]
