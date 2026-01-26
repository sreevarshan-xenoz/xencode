#!/usr/bin/env python3
"""
Optimized Batch Processing for Vector Store Operations

Efficient batch processing system for ChromaDB vector store operations
to improve performance and reduce overhead.
"""

import asyncio
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from langchain_core.documents import Document
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from langchain_ollama import OllamaEmbeddings
# import aiotools  # Commented out to avoid dependency issue

console = Console()


@dataclass
class BatchProcessingConfig:
    """Configuration for batch processing"""
    batch_size: int = 100
    max_concurrent_batches: int = 5
    embedding_batch_size: int = 10  # Smaller for embeddings to manage memory
    timeout_seconds: int = 300
    retry_attempts: int = 3
    backoff_factor: float = 1.0


class OllamaEmbeddingBatchProcessor:
    """Batch processor for Ollama embeddings with optimized performance"""

    def __init__(self, embedding_model: str = "nomic-embed-text", config: BatchProcessingConfig = None):
        self.embedding_model = embedding_model
        self.config = config or BatchProcessingConfig()
        self.embeddings_client = OllamaEmbeddings(model=embedding_model)
        self.executor = ThreadPoolExecutor(max_workers=4)

    async def batch_embed_documents(self, documents: List[str]) -> List[List[float]]:
        """Efficiently embed a batch of documents"""
        if not documents:
            return []

        all_embeddings = []
        
        # Process in smaller batches to manage memory
        for i in range(0, len(documents), self.config.embedding_batch_size):
            batch = documents[i:i + self.config.embedding_batch_size]
            
            # Run embedding in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            batch_embeddings = await loop.run_in_executor(
                self.executor, 
                self.embeddings_client.embed_documents, 
                batch
            )
            
            all_embeddings.extend(batch_embeddings)
            
            # Small delay to prevent overwhelming the embedding service
            await asyncio.sleep(0.01)

        return all_embeddings

    async def batch_embed_query(self, queries: List[str]) -> List[List[float]]:
        """Efficiently embed a batch of queries"""
        if not queries:
            return []

        # Group queries and embed them together for efficiency
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            self.executor,
            self.embeddings_client.embed_documents,
            queries
        )
        
        return embeddings


class OptimizedVectorStore:
    """
    Optimized vector store with efficient batch operations
    """

    def __init__(self,
                 collection_name: str = "xencode_codebase_optimized",
                 persist_directory: str = None,
                 embedding_model: str = "nomic-embed-text",
                 config: BatchProcessingConfig = None):
        """
        Initialize the OptimizedVectorStore.

        Args:
            collection_name: Name of the ChromaDB collection.
            persist_directory: Directory to store the database.
            embedding_model: Ollama model to use for embeddings.
            config: Batch processing configuration.
        """
        if persist_directory is None:
            import os
            from pathlib import Path
            base_dir = os.getcwd()
            persist_directory = os.path.join(base_dir, ".xencode", "rag_store_optimized")

        self.persist_directory = persist_directory
        self.embedding_model = embedding_model
        self.config = config or BatchProcessingConfig()

        # Initialize embeddings and batch processor
        self.embeddings = OllamaEmbeddings(model=embedding_model)
        self.batch_processor = OllamaEmbeddingBatchProcessor(embedding_model, self.config)

        # Initialize Chroma Client
        self.client = chromadb.PersistentClient(path=persist_directory)

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        # Track performance metrics
        self.metrics = {
            "total_documents_processed": 0,
            "total_batches_processed": 0,
            "average_batch_time": 0.0,
            "total_embedding_time": 0.0,
            "total_storage_time": 0.0
        }

    async def add_documents_batch(self, documents: List[Document], 
                                 batch_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Add documents to the vector store in optimized batches.
        """
        start_time = time.time()
        batch_size = batch_size or self.config.batch_size
        
        if not documents:
            return {"processed": 0, "batches": 0, "time_taken": 0.0}

        total_processed = 0
        total_batches = 0
        
        # Prepare all data upfront
        all_texts = [doc.page_content for doc in documents]
        all_metadatas = [doc.metadata for doc in documents]
        all_ids = [f"{doc.metadata.get('filename', 'unknown')}_{i}_{hash(doc.page_content) % 10000}" 
                  for i, doc in enumerate(documents)]

        # Process in batches
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i + batch_size]
            batch_texts = all_texts[i:i + batch_size]
            batch_metadatas = all_metadatas[i:i + batch_size]
            batch_ids = all_ids[i:i + batch_size]
            
            # Embed the batch
            embed_start = time.time()
            batch_embeddings = await self.batch_processor.batch_embed_documents(batch_texts)
            embed_time = time.time() - embed_start
            self.metrics["total_embedding_time"] += embed_time

            # Add to collection
            storage_start = time.time()
            self.collection.add(
                documents=batch_texts,
                embeddings=batch_embeddings,
                metadatas=batch_metadatas,
                ids=batch_ids
            )
            storage_time = time.time() - storage_start
            self.metrics["total_storage_time"] += storage_time

            total_processed += len(batch_docs)
            total_batches += 1

            # Update metrics
            batch_time = embed_time + storage_time
            self.metrics["total_documents_processed"] += len(batch_docs)
            self.metrics["total_batches_processed"] += 1
            
            # Update average time (exponential moving average)
            prev_avg = self.metrics["average_batch_time"]
            if prev_avg == 0:
                self.metrics["average_batch_time"] = batch_time
            else:
                # EMA with alpha = 0.1
                self.metrics["average_batch_time"] = 0.1 * batch_time + 0.9 * prev_avg

        total_time = time.time() - start_time
        
        result = {
            "processed": total_processed,
            "batches": total_batches,
            "time_taken": total_time,
            "documents_per_second": total_processed / total_time if total_time > 0 else 0,
            "average_batch_time": self.metrics["average_batch_time"]
        }
        
        return result

    async def similarity_search_batch(self, queries: List[str], k: int = 4) -> List[List[Document]]:
        """
        Perform similarity search for multiple queries in batch.
        """
        if not queries:
            return [[] for _ in queries]

        results = []
        
        # Embed all queries at once for efficiency
        query_embeddings = await self.batch_processor.batch_embed_query(queries)

        # Process each query
        for i, query in enumerate(queries):
            query_embedding = [query_embeddings[i]]  # Chroma expects a list of embeddings
            
            # Perform search
            search_results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=k
            )

            documents = []
            if search_results['documents'] and search_results['documents'][0]:
                for j in range(len(search_results['documents'][0])):
                    doc = Document(
                        page_content=search_results['documents'][0][j],
                        metadata=search_results['metadatas'][0][j] if search_results['metadatas'] else {}
                    )
                    documents.append(doc)

            results.append(documents)

        return results

    async def add_documents_parallel(self, documents: List[Document], 
                                   max_concurrent: Optional[int] = None) -> Dict[str, Any]:
        """
        Add documents using parallel processing for maximum throughput.
        """
        max_concurrent = max_concurrent or self.config.max_concurrent_batches
        start_time = time.time()
        
        if not documents:
            return {"processed": 0, "time_taken": 0.0}

        # Split documents into batches
        batch_size = self.config.batch_size
        batches = []
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            batches.append(batch)

        # Process batches in parallel with semaphore to limit concurrency
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_batch_semaphore(batch_docs):
            async with semaphore:
                # Create a temporary vector store for this batch to isolate operations
                temp_store = OptimizedVectorStore(
                    collection_name=f"temp_{id(batch_docs)}",
                    persist_directory=self.persist_directory,
                    embedding_model=self.embedding_model,
                    config=self.config
                )
                
                # Add this batch
                result = await temp_store.add_documents_batch(batch_docs, len(batch_docs))
                return result

        # Run all batches in parallel
        tasks = [process_batch_semaphore(batch) for batch in batches]
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Aggregate results
        total_processed = 0
        total_time = time.time() - start_time
        errors = 0

        for result in batch_results:
            if isinstance(result, Exception):
                console.print(f"[red]Error in batch: {result}[/red]")
                errors += 1
            else:
                total_processed += result.get("processed", 0)

        return {
            "processed": total_processed,
            "time_taken": total_time,
            "batches": len(batches),
            "errors": errors,
            "documents_per_second": total_processed / total_time if total_time > 0 else 0
        }

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the vector store operations."""
        return {
            **self.metrics,
            "documents_per_second_overall": (
                self.metrics["total_documents_processed"] / 
                (self.metrics["total_embedding_time"] + self.metrics["total_storage_time"])
                if (self.metrics["total_embedding_time"] + self.metrics["total_storage_time"]) > 0 
                else 0
            )
        }

    def clear(self):
        """Clear the collection."""
        self.client.delete_collection(self.collection.name)
        # Reset metrics
        self.metrics = {
            "total_documents_processed": 0,
            "total_batches_processed": 0,
            "average_batch_time": 0.0,
            "total_embedding_time": 0.0,
            "total_storage_time": 0.0
        }


class BatchIndexer:
    """
    Optimized indexer that uses batch processing for efficient indexing
    """

    def __init__(self, vector_store: OptimizedVectorStore, config: BatchProcessingConfig = None):
        self.vector_store = vector_store
        self.config = config or BatchProcessingConfig()
        
        # Import text splitter
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", " ", ""]
            )
        except ImportError:
            raise ImportError("Please install langchain-text-splitters: pip install langchain-text-splitters")

    async def index_directory_batch(self, root_path: str, verbose: bool = True):
        """
        Index a directory using optimized batch processing.
        """
        import os
        from pathlib import Path
        
        root = Path(root_path)
        all_documents = []

        # Define file extensions and exclusions
        DEFAULT_EXCLUDES = {
            '.git', '__pycache__', 'node_modules', '.venv', 'venv',
            '.env', '.vscode', '.idea', 'dist', 'build', '.xencode'
        }

        DEFAULT_EXTENSIONS = {
            '.py', '.js', '.ts', '.jsx', '.tsx', '.html', '.css',
            '.md', '.txt', '.json', '.yaml', '.yml', '.sql', '.sh'
        }

        # Discovery Phase - collect all files first
        if verbose:
            console.print(f"[blue]🔍 Discovering files in {root_path}...[/blue]")

        files_to_index = []
        for path in root.rglob('*'):
            if path.is_file():
                # Check excludes
                if any(p in path.parts for p in DEFAULT_EXCLUDES):
                    continue

                # Check extension
                if path.suffix not in DEFAULT_EXTENSIONS:
                    continue

                files_to_index.append(path)

        if verbose:
            console.print(f"[green]✅ Found {len(files_to_index)} files to index[/green]")

        # Processing Phase - read and split files
        if verbose:
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                console=console
            ) as progress:
                task = progress.add_task("Processing files...", total=len(files_to_index))

                for file_path in files_to_index:
                    try:
                        docs = await self._process_file_async(file_path)
                        all_documents.extend(docs)
                        progress.advance(task)
                    except Exception as e:
                        if verbose:
                            console.print(f"[yellow]⚠️ Skipping {file_path}: {e}[/yellow]")
        else:
            for file_path in files_to_index:
                try:
                    docs = await self._process_file_async(file_path)
                    all_documents.extend(docs)
                except Exception:
                    pass

        # Batch add to vector store
        if all_documents:
            if verbose:
                console.print(f"[blue]💾 Storing {len(all_documents)} chunks to vector store...[/blue]")
            
            result = await self.vector_store.add_documents_batch(
                all_documents, 
                batch_size=self.config.batch_size
            )
            
            if verbose:
                console.print(f"[green]✅ Indexed {result['processed']} documents in {result['time_taken']:.2f}s[/green]")
                console.print(f"   📊 Throughput: {result['documents_per_second']:.2f} docs/sec")
        else:
            if verbose:
                console.print("[yellow]⚠️ No suitable files found to index[/yellow]")

    async def _process_file_async(self, file_path):
        """Process a single file asynchronously."""
        loop = asyncio.get_event_loop()
        
        def read_and_split():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                metadatas = {"source": str(file_path), "filename": file_path.name}
                docs = self.text_splitter.create_documents([content], metadatas=[metadatas])
                return docs
            except UnicodeDecodeError:
                return []
            except Exception:
                return []

        # Run the synchronous file operations in a thread pool
        return await loop.run_in_executor(self.vector_store.batch_processor.executor, read_and_split)


# Example usage and benchmarking
async def benchmark_batch_operations():
    """Benchmark the batch operations against standard operations."""
    console.print("[bold blue]🚀 Batch Processing Benchmark[/bold blue]")
    
    # Create optimized vector store
    config = BatchProcessingConfig(
        batch_size=50,
        embedding_batch_size=10,
        max_concurrent_batches=3
    )
    
    store = OptimizedVectorStore(config=config)
    
    # Create test documents
    test_docs = []
    for i in range(200):  # 200 test documents
        doc = Document(
            page_content=f"This is test document {i} with some sample content to test the batch processing capabilities of our optimized vector store. " * 5,
            metadata={"source": f"test_source_{i}", "filename": f"test_file_{i}.txt", "chunk_id": i}
        )
        test_docs.append(doc)
    
    console.print(f"[green]✅ Created {len(test_docs)} test documents[/green]")
    
    # Test batch processing
    console.print("[blue]⏱️ Testing batch processing...[/blue]")
    start_time = time.time()
    batch_result = await store.add_documents_batch(test_docs, batch_size=50)
    batch_time = time.time() - start_time
    
    console.print(f"[green]✅ Batch processing completed:[/green]")
    console.print(f"   Documents: {batch_result['processed']}")
    console.print(f"   Time: {batch_time:.2f}s")
    console.print(f"   Throughput: {batch_result['documents_per_second']:.2f} docs/sec")
    
    # Show performance metrics
    metrics = store.get_performance_metrics()
    console.print(f"\n[bold]Performance Metrics:[/bold]")
    console.print(f"   Total docs processed: {metrics['total_documents_processed']}")
    console.print(f"   Avg batch time: {metrics['average_batch_time']:.3f}s")
    console.print(f"   Embedding time: {metrics['total_embedding_time']:.2f}s")
    console.print(f"   Storage time: {metrics['total_storage_time']:.2f}s")
    console.print(f"   Overall throughput: {metrics['documents_per_second_overall']:.2f} docs/sec")


if __name__ == "__main__":
    # Don't run benchmark by default
    # asyncio.run(benchmark_batch_operations())
    pass