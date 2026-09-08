"""
Ensemble Reasoning package for Xencode.

Combines multiple AI model responses using advanced fusion algorithms
including voting, weighted voting, semantic fusion, consensus, and hybrid methods.
"""

from .ai_ensembles import (
    EnsembleMethod,
    EnsembleReasoner,
    ModelConfig,
    ModelResponse,
    ModelTier,
    QueryRequest,
    QueryResponse,
    TokenVoter,
)
from .ensemble_lightweight import (
    EnhancedQualityMetrics,
    ImprovedConsensus,
    LightweightTokenVoter,
)

__all__ = [
    # Core ensemble types
    "EnsembleMethod",
    "ModelTier",
    "ModelConfig",
    "QueryRequest",
    "ModelResponse",
    "QueryResponse",
    # Core classes
    "TokenVoter",
    "EnsembleReasoner",
    # Improved components
    "LightweightTokenVoter",
    "ImprovedConsensus",
    "EnhancedQualityMetrics",
]
