import os
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
        return self.ollama_embeddings.embed_documents(list(input))

class VectorStore:
    """
    Wrapper around ChromaDB for storing and retrieving code context.
    Uses OllamaEmbeddings for vectorization.
    """
    
    def __init__(self, 
                 collection_name: str = "xencode_codebase",
                 persist_directory: str = None,
                 embedding_model: str = "nomic-embed-text"):
        """
        Initialize the VectorStore.
        
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

    def clear(self):
        """Delete specific collection."""
        self.client.delete_collection(self.collection.name)
