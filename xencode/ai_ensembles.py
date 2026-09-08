"""
Backward compatibility shim for ai_ensembles.

The ensemble module has been moved to xencode/ensemble/.
This module re-exports everything from the new location.
"""

from xencode.ensemble.ai_ensembles import (
    EnsembleMethod,
    EnsembleReasoner,
    ModelConfig,
    ModelResponse,
    ModelTier,
    QueryRequest,
    QueryResponse,
    TokenVoter,
    create_ensemble_reasoner,
)

__all__ = [
    "EnsembleMethod",
    "ModelTier",
    "ModelConfig",
    "QueryRequest",
    "ModelResponse",
    "QueryResponse",
    "TokenVoter",
    "EnsembleReasoner",
    "create_ensemble_reasoner",
]
