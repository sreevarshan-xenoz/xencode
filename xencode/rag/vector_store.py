import os
import asyncio
import time
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from .graph_store import GraphStore
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn

console = Console()

class OllamaChromaWrapper(EmbeddingFunction):
    def __init__(self, ollama_embeddings):
        self.ollama_embeddings = ollama_embeddings
        
    def __call__(self, input: Documents) -> Embeddings:
        return self.ollama_embeddings.embed_documents(list(input))

class VectorStore:
    """
    Wrapper around ChromaDB for storing and retrieving code context.
    Uses OllamaEmbeddings for vectorization.
    """
    
    def __init__(self, 
                 collection_name: str = "xencode_codebase",
                 persist_directory: str = None,
                 embedding_model: str = "nomic-embed-text",
                 graph_store: Optional[GraphStore] = None):
        """
        Initialize the VectorStore.
        
        Args:
            collection_name: Name of the ChromaDB collection.
            persist_directory: Directory to store the database. Defaults to .xencode/rag_store
            embedding_model: Ollama model to use for embeddings.
            graph_store: Optional GraphStore instance for relationship-aware retrieval.
        """
        if persist_directory is None:
            # Default to .xencode/rag_store in the project root
            base_dir = os.getcwd()
            persist_directory = os.path.join(base_dir, ".xencode", "rag_store")
            
        self.persist_directory = persist_directory
        self.embedding_model = embedding_model
        self.graph_store = graph_store or GraphStore()
        
        # Initialize Embeddings
        self.embeddings = OllamaEmbeddings(model=embedding_model)
        
        # Initialize Chroma Client
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Get or create collection with explicit embedding function
        # This prevents Chroma from trying to load sentence-transformers
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=OllamaChromaWrapper(self.embeddings),
            metadata={"hnsw:space": "cosine"}
        )
        
    def add_documents(self, documents: List[Document]):
        """
        Add documents to the vector store.
        """
        if not documents:
            return

        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        ids = [f"{documents[i].metadata.get('filename', 'unknown')}_{i}" for i in range(len(documents))]
        
        # Generate embeddings
        # We can either let Chroma compute them (if we provide an embedding function)
        # or pre-compute them. Let's pre-compute with LangChain's OllamaEmbeddings.
        embeddings = self.embeddings.embed_documents(texts)
        
        self.collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """
        Search for similar documents.
        """
        query_embedding = self.embeddings.embed_query(query)
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )
        
        documents = []
        if results['documents']:
            for i in range(len(results['documents'][0])):
                doc = Document(
                    page_content=results['documents'][0][i],
                    metadata=results['metadatas'][0][i] if results['metadatas'] else {}
                )
                documents.append(doc)
                
        return documents

    def enhanced_similarity_search(self, query: str, k: int = 4, graph_depth: int = 1) -> List[Document]:
        """
        Performs similarity search and enhances results with related nodes from GraphStore.
        """
        # 1. Standard Similarity Search
        base_docs = self.similarity_search(query, k=k)
        
        if not self.graph_store:
            return base_docs
            
        # 2. Graph Enhancement
        enhanced_docs = list(base_docs)
        seen_sources = {doc.metadata.get('source') for doc in base_docs}
        
        for doc in base_docs:
            source_file = doc.metadata.get('source')
            if not source_file:
                continue
                
            # Find related nodes in the graph
            related_nodes = self.graph_store.get_related_nodes(source_file, depth=graph_depth)
            
            for node in related_nodes:
                node_type = node.get('type')
                node_id = node.get('id')
                
                # If the related node is another file, include it
                if node_type == 'file':
                    related_path = node_id
                    if related_path and related_path not in seen_sources:
                        try:
                            with open(related_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                                
                            new_doc = Document(
                                page_content=content[:2000],
                                metadata={
                                    "source": related_path,
                                    "filename": os.path.basename(related_path),
                                    "reason": f"Graph-related to {os.path.basename(source_file)}"
                                }
                            )
                            enhanced_docs.append(new_doc)
                            seen_sources.add(related_path)
                        except Exception:
                            pass
                
                # If it's a function or class, we can be more granular
                elif node_type in ['function', 'class']:
                    # For now, we still reference the file but add metadata about the specific entity
                    # In a more advanced version, we would extract just that code block
                    entity_name = node.get('name')
                    file_part = node_id.split('::')[0] if '::' in node_id else source_file
                    
                    if file_part and file_part not in seen_sources:
                        try:
                            with open(file_part, 'r', encoding='utf-8') as f:
                                lines = f.readlines()
                            
                            start_line = node.get('line', 1) - 1
                            # Take 50 lines or until end of file
                            end_line = min(len(lines), start_line + 50)
                            content = "".join(lines[start_line:end_line])
                            
                            new_doc = Document(
                                page_content=content,
                                metadata={
                                    "source": file_part,
                                    "entity": entity_name,
                                    "type": node_type,
                                    "reason": f"Graph-related {node_type} '{entity_name}'"
                                }
                            )
                            enhanced_docs.append(new_doc)
                            # We don't add to seen_sources here to allow multiple entities from same file
                            # but we should probably limit this.
                        except Exception:
                            pass
                            
        return enhanced_docs

    def clear(self):
        """Delete specific collection."""
        self.client.delete_collection(self.collection.name)


class AsyncVectorStore:
    """
    Async wrapper around ChromaDB for storing and retrieving code context.
    Uses OllamaEmbeddings for vectorization with async support.
    """

    def __init__(self,
                 collection_name: str = "xencode_codebase",
                 persist_directory: str = None,
                 embedding_model: str = "nomic-embed-text"):
        if persist_directory is None:
            base_dir = os.getcwd()
            persist_directory = os.path.join(base_dir, ".xencode", "rag_store")

        self.persist_directory = persist_directory
        self.embedding_model = embedding_model
        self.embeddings = OllamaEmbeddings(model=embedding_model)
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=OllamaChromaWrapper(self.embeddings),
            metadata={"hnsw:space": "cosine"}
        )

    async def add_documents(self, documents: List[Document]) -> None:
        if not documents:
            return

        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        ids = [f"{documents[i].metadata.get('filename', 'unknown')}_{i}" for i in range(len(documents))]

        loop = asyncio.get_event_loop()

        def embed_documents():
            return self.embeddings.embed_documents(texts)

        embeddings = await loop.run_in_executor(None, embed_documents)

        def add_to_collection():
            self.collection.add(
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )

        await loop.run_in_executor(None, add_to_collection)

    async def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        loop = asyncio.get_event_loop()

        def embed_query():
            return self.embeddings.embed_query(query)

        query_embedding = await loop.run_in_executor(None, embed_query)

        def perform_search():
            return self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k
            )

        results = await loop.run_in_executor(None, perform_search)

        documents = []
        if results['documents']:
            for i in range(len(results['documents'][0])):
                doc = Document(
                    page_content=results['documents'][0][i],
                    metadata=results['metadatas'][0][i] if results['metadatas'] else {}
                )
                documents.append(doc)

        return documents

    async def clear(self) -> None:
        loop = asyncio.get_event_loop()

        def delete_collection():
            self.client.delete_collection(self.collection.name)

        await loop.run_in_executor(None, delete_collection)


@dataclass
class BatchProcessingConfig:
    """Configuration for batch processing"""

    batch_size: int = 100
    max_concurrent_batches: int = 5
    embedding_batch_size: int = 10
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
        if not documents:
            return []

        all_embeddings = []

        for i in range(0, len(documents), self.config.embedding_batch_size):
            batch = documents[i:i + self.config.embedding_batch_size]
            loop = asyncio.get_event_loop()
            batch_embeddings = await loop.run_in_executor(
                self.executor,
                self.embeddings_client.embed_documents,
                batch
            )
            all_embeddings.extend(batch_embeddings)
            await asyncio.sleep(0.01)

        return all_embeddings

    async def batch_embed_query(self, queries: List[str]) -> List[List[float]]:
        if not queries:
            return []

        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            self.executor,
            self.embeddings_client.embed_documents,
            queries
        )

        return embeddings


class OptimizedVectorStore:
    """Optimized vector store with efficient batch operations"""

    def __init__(self,
                 collection_name: str = "xencode_codebase_optimized",
                 persist_directory: str = None,
                 embedding_model: str = "nomic-embed-text",
                 config: BatchProcessingConfig = None):
        if persist_directory is None:
            base_dir = os.getcwd()
            persist_directory = os.path.join(base_dir, ".xencode", "rag_store_optimized")

        self.persist_directory = persist_directory
        self.embedding_model = embedding_model
        self.config = config or BatchProcessingConfig()

        self.embeddings = OllamaEmbeddings(model=embedding_model)
        self.batch_processor = OllamaEmbeddingBatchProcessor(embedding_model, self.config)
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        self.metrics = {
            "total_documents_processed": 0,
            "total_batches_processed": 0,
            "average_batch_time": 0.0,
            "total_embedding_time": 0.0,
            "total_storage_time": 0.0,
        }

    async def add_documents_batch(self, documents: List[Document],
                                 batch_size: Optional[int] = None) -> Dict[str, Any]:
        start_time = time.time()
        batch_size = batch_size or self.config.batch_size

        if not documents:
            return {"processed": 0, "batches": 0, "time_taken": 0.0}

        total_processed = 0
        total_batches = 0

        all_texts = [doc.page_content for doc in documents]
        all_metadatas = [doc.metadata for doc in documents]
        all_ids = [
            f"{doc.metadata.get('filename', 'unknown')}_{i}_{hash(doc.page_content) % 10000}"
            for i, doc in enumerate(documents)
        ]

        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i + batch_size]
            batch_texts = all_texts[i:i + batch_size]
            batch_metadatas = all_metadatas[i:i + batch_size]
            batch_ids = all_ids[i:i + batch_size]

            embed_start = time.time()
            batch_embeddings = await self.batch_processor.batch_embed_documents(batch_texts)
            embed_time = time.time() - embed_start
            self.metrics["total_embedding_time"] += embed_time

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

            batch_time = embed_time + storage_time
            self.metrics["total_documents_processed"] += len(batch_docs)
            self.metrics["total_batches_processed"] += 1

            prev_avg = self.metrics["average_batch_time"]
            if prev_avg == 0:
                self.metrics["average_batch_time"] = batch_time
            else:
                self.metrics["average_batch_time"] = 0.1 * batch_time + 0.9 * prev_avg

        total_time = time.time() - start_time

        return {
            "processed": total_processed,
            "batches": total_batches,
            "time_taken": total_time,
            "documents_per_second": total_processed / total_time if total_time > 0 else 0,
            "average_batch_time": self.metrics["average_batch_time"],
        }

    async def similarity_search_batch(self, queries: List[str], k: int = 4) -> List[List[Document]]:
        if not queries:
            return [[] for _ in queries]

        results = []
        query_embeddings = await self.batch_processor.batch_embed_query(queries)

        for i, query in enumerate(queries):
            query_embedding = [query_embeddings[i]]

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
        max_concurrent = max_concurrent or self.config.max_concurrent_batches
        start_time = time.time()

        if not documents:
            return {"processed": 0, "time_taken": 0.0}

        batch_size = self.config.batch_size
        batches = []
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            batches.append(batch)

        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_batch_semaphore(batch_docs):
            async with semaphore:
                temp_store = OptimizedVectorStore(
                    collection_name=f"temp_{id(batch_docs)}",
                    persist_directory=self.persist_directory,
                    embedding_model=self.embedding_model,
                    config=self.config
                )
                result = await temp_store.add_documents_batch(batch_docs, len(batch_docs))
                return result

        tasks = [process_batch_semaphore(batch) for batch in batches]
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)

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
            "documents_per_second": total_processed / total_time if total_time > 0 else 0,
        }

    def get_performance_metrics(self) -> Dict[str, Any]:
        total_time = self.metrics["total_embedding_time"] + self.metrics["total_storage_time"]
        return {
            **self.metrics,
            "documents_per_second_overall": (
                self.metrics["total_documents_processed"] / total_time if total_time > 0 else 0
            ),
        }

    def clear(self) -> None:
        self.client.delete_collection(self.collection.name)
        self.metrics = {
            "total_documents_processed": 0,
            "total_batches_processed": 0,
            "average_batch_time": 0.0,
            "total_embedding_time": 0.0,
            "total_storage_time": 0.0,
        }


class BatchIndexer:
    """Optimized indexer that uses batch processing for efficient indexing"""

    def __init__(self, vector_store: OptimizedVectorStore, config: BatchProcessingConfig = None):
        self.vector_store = vector_store
        self.config = config or BatchProcessingConfig()

        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", " ", ""]
            )
        except ImportError:
            raise ImportError("Please install langchain-text-splitters: pip install langchain-text-splitters")

    async def index_directory_batch(self, root_path: str, verbose: bool = True) -> None:
        from pathlib import Path

        root = Path(root_path)
        all_documents = []

        default_excludes = {
            '.git', '__pycache__', 'node_modules', '.venv', 'venv',
            '.env', '.vscode', '.idea', 'dist', 'build', '.xencode'
        }

        default_extensions = {
            '.py', '.js', '.ts', '.jsx', '.tsx', '.html', '.css',
            '.md', '.txt', '.json', '.yaml', '.yml', '.sql', '.sh'
        }

        if verbose:
            console.print(f"[blue]🔍 Discovering files in {root_path}...[/blue]")

        files_to_index = []
        for path in root.rglob('*'):
            if path.is_file():
                if any(p in path.parts for p in default_excludes):
                    continue
                if path.suffix not in default_extensions:
                    continue
                files_to_index.append(path)

        if verbose:
            console.print(f"[green]✅ Found {len(files_to_index)} files to index[/green]")

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
                        console.print(f"[yellow]⚠️ Skipping {file_path}: {e}[/yellow]")
        else:
            for file_path in files_to_index:
                try:
                    docs = await self._process_file_async(file_path)
                    all_documents.extend(docs)
                except Exception:
                    pass

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

    async def _process_file_async(self, file_path: str) -> List[Document]:
        loop = asyncio.get_event_loop()

        def read_and_split():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                metadatas = {"source": str(file_path), "filename": Path(file_path).name}
                docs = self.text_splitter.create_documents([content], metadatas=[metadatas])
                return docs
            except UnicodeDecodeError:
                return []
            except Exception:
                return []

        return await loop.run_in_executor(self.vector_store.batch_processor.executor, read_and_split)
