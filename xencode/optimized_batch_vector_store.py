#!/usr/bin/env python3
"""
Compatibility shim for the optimized batch vector store.

The canonical implementation now lives in rag.vector_store.
"""

from .rag.vector_store import BatchProcessingConfig, OptimizedVectorStore, BatchIndexer

__all__ = ["BatchProcessingConfig", "OptimizedVectorStore", "BatchIndexer"]
