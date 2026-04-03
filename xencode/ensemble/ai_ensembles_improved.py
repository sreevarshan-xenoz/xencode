#!/usr/bin/env python3
"""
Compatibility shim for the legacy improved ensemble module.

The canonical implementation now lives in ai_ensembles.
This module re-exports the shared APIs to avoid duplicate behavior.
"""

from .ai_ensembles import (
    EnsembleMethod,
    ModelTier,
    ModelConfig,
    QueryRequest,
    ModelResponse,
    QueryResponse,
    TokenVoter,
    EnsembleReasoner,
    create_ensemble_reasoner,
    quick_ensemble_query,
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
    "quick_ensemble_query",
]
