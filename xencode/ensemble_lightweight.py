"""
Backward compatibility shim for ensemble_lightweight.

The ensemble lightweight module has been moved to xencode/ensemble/.
This module re-exports everything from the new location.
"""

from xencode.ensemble.ensemble_lightweight import (
    LightweightTokenVoter,
    ImprovedConsensus,
    EnhancedQualityMetrics,
    create_improved_components,
)

__all__ = [
    "LightweightTokenVoter",
    "ImprovedConsensus",
    "EnhancedQualityMetrics",
    "create_improved_components",
]
