"""
RAG (Retrieval-Augmented Generation) Package

Provides vector storage, graph-based code relationships, and context indexing
for AI-powered code understanding and retrieval.
"""

from .vector_store import (
    VectorStore,
    AsyncVectorStore,
    BatchProcessingConfig,
    OllamaEmbeddingBatchProcessor,
    OptimizedVectorStore,
    BatchIndexer,
    OllamaChromaWrapper,
)

from .indexer import Indexer

from .context_indexer_v2 import (
    ContextIndexerV2,
    IndexStatus,
    FileMetadata,
    Symbol,
    IndexManifest,
)

from .graph_extractor import CodeGraphExtractor

from .graph_store import GraphStore

__all__ = [
    # Vector Store
    "VectorStore",
    "AsyncVectorStore",
    "OptimizedVectorStore",
    "BatchProcessingConfig",
    "OllamaEmbeddingBatchProcessor",
    "BatchIndexer",
    "OllamaChromaWrapper",
    # Indexer
    "Indexer",
    # Context Indexer v2
    "ContextIndexerV2",
    "IndexStatus",
    "FileMetadata",
    "Symbol",
    "IndexManifest",
    # Graph
    "CodeGraphExtractor",
    "GraphStore",
]
