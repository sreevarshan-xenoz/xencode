import os
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from .graph_store import GraphStore

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
