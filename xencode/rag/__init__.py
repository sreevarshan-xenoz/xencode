"""
RAG (Retrieval-Augmented Generation) Package

Provides vector storage, graph-based code relationships, and context indexing
for AI-powered code understanding and retrieval.
"""

try:
    from .vector_store import (
        VectorStore,
        AsyncVectorStore,
        BatchProcessingConfig,
        OllamaEmbeddingBatchProcessor,
        OptimizedVectorStore,
        BatchIndexer,
        OllamaChromaWrapper,
    )
except ImportError:
    VectorStore = AsyncVectorStore = BatchProcessingConfig = None
    OllamaEmbeddingBatchProcessor = OptimizedVectorStore = None
    BatchIndexer = OllamaChromaWrapper = None

try:
    from .indexer import Indexer
except ImportError:
    Indexer = None

try:
    from .context_indexer_v2 import (
        ContextIndexerV2,
        IndexStatus,
        FileMetadata,
        Symbol,
        IndexManifest,
    )
except ImportError:
    ContextIndexerV2 = IndexStatus = FileMetadata = Symbol = IndexManifest = None

try:
    from .graph_extractor import CodeGraphExtractor
except ImportError:
    CodeGraphExtractor = None

try:
    from .graph_store import GraphStore
except ImportError:
    GraphStore = None

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
