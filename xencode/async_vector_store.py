#!/usr/bin/env python3
"""
Compatibility shim for the async vector store.

The canonical implementation now lives in rag.vector_store.
"""

from .rag.vector_store import AsyncVectorStore

__all__ = ["AsyncVectorStore"]
