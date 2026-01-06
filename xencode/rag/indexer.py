import os
from pathlib import Path
from typing import List, Set
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from rich.progress import Progress, SpinnerColumn, TextColumn

from .vector_store import VectorStore

class Indexer:
    """
    Handles indexing of a codebase into the VectorStore.
    """
    
    DEFAULT_EXCLUDES = {
        '.git', '__pycache__', 'node_modules', '.venv', 'venv', 
        '.env', '.vscode', '.idea', 'dist', 'build', '.xencode'
    }
    
    DEFAULT_EXTENSIONS = {
        '.py', '.js', '.ts', '.jsx', '.tsx', '.html', '.css', 
        '.md', '.txt', '.json', '.yaml', '.yml', '.sql', '.sh'
    }
    
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        
    def index_directory(self, root_path: str, verbose: bool = True):
        """
        Walks the directory and indexes allowed files.
        """
        root = Path(root_path)
        documents: List[Document] = []
        
        files_to_index = []
        
        # Discovery Phase
        if verbose:
            print(f"Scanning {root_path}...")
            
        for path in root.rglob('*'):
            if path.is_file():
                # Check excludes
                if any(p in path.parts for p in self.DEFAULT_EXCLUDES):
                    continue
                
                # Check extension
                if path.suffix not in self.DEFAULT_EXTENSIONS:
                    continue
                    
                files_to_index.append(path)
        
        # Processing Phase
        if verbose:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
            ) as progress:
                task = progress.add_task(f"Indexing {len(files_to_index)} files...", total=len(files_to_index))
                
                for file_path in files_to_index:
                    try:
                        docs = self._process_file(file_path)
                        documents.extend(docs)
                        progress.advance(task)
                    except Exception as e:
                        # Silently fail on individual files but log if needed
                        pass
        else:
             for file_path in files_to_index:
                try:
                    docs = self._process_file(file_path)
                    documents.extend(docs)
                except Exception:
                    pass

        # Batch add to vector store
        if documents:
            if verbose:
                print(f"Storing {len(documents)} chunks to vector store...")
            self.vector_store.add_documents(documents)
            if verbose:
                print("Indexing complete.")
        else:
            if verbose:
                print("No suitable files found to index.")

    def _process_file(self, file_path: Path) -> List[Document]:
        """Read and split a single file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            metadatas = {"source": str(file_path), "filename": file_path.name}
            docs = self.text_splitter.create_documents([content], metadatas=[metadatas])
            return docs
        except UnicodeDecodeError:
            return []
