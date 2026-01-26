import os
import asyncio
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

class OllamaChromaWrapper(EmbeddingFunction):
    def __init__(self, ollama_embeddings):
        self.ollama_embeddings = ollama_embeddings

    def __call__(self, input: Documents) -> Embeddings:
        # Run the synchronous embedding in a thread pool to avoid blocking
        import concurrent.futures
        import threading
        
        loop = asyncio.get_event_loop()
        
        # Use run_in_executor to run the synchronous operation in a thread pool
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(self.ollama_embeddings.embed_documents, list(input))
            return future.result()


class AsyncVectorStore:
    """
    Async wrapper around ChromaDB for storing and retrieving code context.
    Uses OllamaEmbeddings for vectorization with async support.
    """

    def __init__(self,
                 collection_name: str = "xencode_codebase",
                 persist_directory: str = None,
                 embedding_model: str = "nomic-embed-text"):
        """
        Initialize the AsyncVectorStore.

        Args:
            collection_name: Name of the ChromaDB collection.
            persist_directory: Directory to store the database. Defaults to .xencode/rag_store
            embedding_model: Ollama model to use for embeddings.
        """
        if persist_directory is None:
            # Default to .xencode/rag_store in the project root
            base_dir = os.getcwd()
            persist_directory = os.path.join(base_dir, ".xencode", "rag_store")

        self.persist_directory = persist_directory
        self.embedding_model = embedding_model

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

    async def add_documents(self, documents: List[Document]):
        """
        Add documents to the vector store asynchronously.
        """
        if not documents:
            return

        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        ids = [f"{documents[i].metadata.get('filename', 'unknown')}_{i}" for i in range(len(documents))]

        # Generate embeddings - run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        import concurrent.futures
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Submit the embedding operation to the thread pool
            future = executor.submit(self.embeddings.embed_documents, texts)
            embeddings = await loop.run_in_executor(None, lambda: future.result())

        # Add to collection - also run in thread pool
        def add_to_collection():
            self.collection.add(
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
        
        await loop.run_in_executor(None, add_to_collection)

    async def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """
        Search for similar documents asynchronously.
        """
        # Generate query embedding - run in thread pool
        loop = asyncio.get_event_loop()
        import concurrent.futures
        
        def embed_query():
            return self.embeddings.embed_query(query)
        
        query_embedding = await loop.run_in_executor(None, embed_query)

        # Perform search - run in thread pool
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

    async def clear(self):
        """Delete specific collection asynchronously."""
        loop = asyncio.get_event_loop()
        import concurrent.futures
        
        def delete_collection():
            self.client.delete_collection(self.collection.name)
        
        await loop.run_in_executor(None, delete_collection)