#!/usr/bin/env python3
"""
Context Indexer v2 - Repo-wide context indexing with incremental updates.

Features:
- Incremental project indexing with symbol metadata
- Stale file invalidation based on modification time
- File change detection using content hashing
- Enhanced symbol extraction for classes, functions, imports
- Efficient batch processing with async support
"""

import asyncio
import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
from enum import Enum

from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
    TaskProgressColumn,
)

from .graph_extractor import CodeGraphExtractor
from .vector_store import VectorStore, OptimizedVectorStore, BatchProcessingConfig
from .graph_store import GraphStore

console = Console()


class IndexStatus(Enum):
    """Index status enumeration"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    STALE = "stale"
    FAILED = "failed"


@dataclass
class FileMetadata:
    """Metadata for indexed file"""
    path: str
    size: int
    modified_time: float
    content_hash: str
    indexed_time: datetime
    symbol_count: int = 0
    line_count: int = 0
    token_estimate: int = 0
    status: IndexStatus = IndexStatus.PENDING

    def to_dict(self) -> Dict[str, Any]:
        return {
            'path': self.path,
            'size': self.size,
            'modified_time': self.modified_time,
            'content_hash': self.content_hash,
            'indexed_time': self.indexed_time.isoformat(),
            'symbol_count': self.symbol_count,
            'line_count': self.line_count,
            'token_estimate': self.token_estimate,
            'status': self.status.value,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FileMetadata':
        return cls(
            path=data['path'],
            size=data['size'],
            modified_time=data['modified_time'],
            content_hash=data['content_hash'],
            indexed_time=datetime.fromisoformat(data['indexed_time']),
            symbol_count=data.get('symbol_count', 0),
            line_count=data.get('line_count', 0),
            token_estimate=data.get('token_estimate', 0),
            status=IndexStatus(data.get('status', 'pending')),
        )


@dataclass
class Symbol:
    """Symbol extracted from code"""
    name: str
    type: str  # class, function, method, import, constant
    file_path: str
    line: int
    end_line: Optional[int] = None
    signature: Optional[str] = None
    docstring: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'type': self.type,
            'file_path': self.file_path,
            'line': self.line,
            'end_line': self.end_line,
            'signature': self.signature,
            'docstring': self.docstring,
            'dependencies': self.dependencies,
            'metadata': self.metadata,
        }


@dataclass
class IndexManifest:
    """Manifest tracking indexed files and their state"""
    project_root: str
    created_at: datetime
    updated_at: datetime
    files: Dict[str, FileMetadata] = field(default_factory=dict)
    symbols: Dict[str, Symbol] = field(default_factory=dict)
    total_tokens: int = 0
    index_version: str = "2.0"

    def to_dict(self) -> Dict[str, Any]:
        return {
            'project_root': self.project_root,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'files': {k: v.to_dict() for k, v in self.files.items()},
            'symbols': {k: v.to_dict() for k, v in self.symbols.items()},
            'total_tokens': self.total_tokens,
            'index_version': self.index_version,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'IndexManifest':
        manifest = cls(
            project_root=data['project_root'],
            created_at=datetime.fromisoformat(data['created_at']),
            updated_at=datetime.fromisoformat(data['updated_at']),
            total_tokens=data.get('total_tokens', 0),
            index_version=data.get('index_version', '2.0'),
        )
        manifest.files = {
            k: FileMetadata.from_dict(v)
            for k, v in data.get('files', {}).items()
        }
        manifest.symbols = {
            k: Symbol(**v)
            for k, v in data.get('symbols', {}).items()
        }
        return manifest


class ContextIndexerV2:
    """
    Advanced context indexer with incremental updates and symbol tracking.
    
    Features:
    - Incremental indexing: Only re-index changed files
    - Symbol extraction: Track classes, functions, imports
    - Stale detection: Automatically detect and invalidate stale entries
    - Batch processing: Efficient async batch operations
    - Progress tracking: Rich progress indicators
    """

    DEFAULT_EXCLUDES = {
        '.git', '__pycache__', 'node_modules', '.venv', 'venv',
        '.env', '.vscode', '.idea', 'dist', 'build', '.xencode',
        'coverage', '.pytest_cache', '.mypy_cache', '.ruff_cache',
        'dist-packages', 'site-packages', 'bower_components',
        '.tox', '.nox', '.eggs', '*.egg-info',
    }

    DEFAULT_EXTENSIONS = {
        '.py', '.js', '.ts', '.jsx', '.tsx', '.html', '.css',
        '.md', '.txt', '.json', '.yaml', '.yml', '.sql', '.sh',
        '.rs', '.go', '.java', '.cpp', '.c', '.h', '.hpp',
        '.rb', '.php', '.swift', '.kt', '.scala',
    }

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        graph_store: Optional[GraphStore] = None,
        persist_directory: Optional[str] = None,
        embedding_model: str = "nomic-embed-text",
    ):
        """
        Initialize the context indexer.
        
        Args:
            vector_store: Optional VectorStore instance (created if not provided)
            graph_store: Optional GraphStore instance (created if not provided)
            persist_directory: Directory for storing index manifest
            embedding_model: Ollama model for embeddings
        """
        self.graph_extractor = CodeGraphExtractor()
        
        # Initialize stores
        if vector_store is None:
            vector_store = VectorStore(
                collection_name="xencode_codebase_v2",
                persist_directory=persist_directory,
                embedding_model=embedding_model,
            )
        self.vector_store = vector_store
        
        if graph_store is None:
            graph_store = GraphStore()
        self.graph_store = graph_store
        
        # Manifest tracking
        self.manifest: Optional[IndexManifest] = None
        self.project_root: Optional[Path] = None
        
        # Configuration
        self.chunk_size = 1000
        self.chunk_overlap = 200
        self.stale_threshold_hours = 24
        
        # Try to import langchain text splitter
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n\n", "\n", " ", ""]
            )
        except ImportError:
            self.text_splitter = None
            console.print("[yellow]⚠️ langchain-text-splitters not installed, using basic splitting[/yellow]")

    def _calculate_content_hash(self, content: str) -> str:
        """Calculate SHA-256 hash of content"""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation: 1 token ≈ 4 chars)"""
        return len(text) // 4

    def _is_stale(self, file_path: Path, metadata: FileMetadata) -> bool:
        """Check if a file is stale compared to its metadata"""
        if not file_path.exists():
            return True
        
        stat = file_path.stat()
        
        # Check if file was modified after indexing
        if stat.st_mtime > metadata.modified_time:
            return True
        
        # Check if content hash changed
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            current_hash = self._calculate_content_hash(content)
            if current_hash != metadata.content_hash:
                return True
        except Exception:
            return True
        
        # Check time-based staleness
        age = datetime.now() - metadata.indexed_time
        if age.total_seconds() > self.stale_threshold_hours * 3600:
            return True
        
        return False

    def _detect_changes(
        self,
        root_path: Path,
        extensions: Set[str],
        excludes: Set[str],
    ) -> Tuple[List[Path], List[str], List[str]]:
        """
        Detect files that need indexing.
        
        Returns:
            Tuple of (new_files, stale_files, removed_files)
        """
        new_files = []
        stale_files = []
        removed_files = []
        
        # Track seen files
        seen_paths: Set[str] = set()
        
        # Scan directory
        for path in root_path.rglob('*'):
            if not path.is_file():
                continue
            
            # Check excludes
            if any(p in path.parts for p in excludes):
                continue
            
            # Check extension
            if path.suffix not in extensions:
                continue
            
            rel_path = str(path.relative_to(root_path))
            seen_paths.add(rel_path)
            
            # Check if file is in manifest
            if rel_path not in self.manifest.files:
                new_files.append(path)
            else:
                # Check if stale
                metadata = self.manifest.files[rel_path]
                if self._is_stale(path, metadata):
                    stale_files.append(rel_path)
        
        # Check for removed files
        for rel_path in self.manifest.files:
            if rel_path not in seen_paths:
                removed_files.append(rel_path)
        
        return new_files, stale_files, removed_files

    def _extract_symbols(self, file_path: Path, content: str) -> List[Symbol]:
        """Extract symbols from file content using AST"""
        symbols = []
        
        if file_path.suffix != '.py':
            return symbols
        
        try:
            tree = ast.parse(content)
            rel_path = str(file_path)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    symbols.append(Symbol(
                        name=node.name,
                        type='class',
                        file_path=rel_path,
                        line=node.lineno,
                        end_line=getattr(node, 'end_lineno', None),
                        signature=f"class {node.name}",
                        docstring=ast.get_docstring(node),
                        metadata={
                            'bases': [self._get_name(base) for base in node.bases],
                            'decorators': [self._get_name(d) for d in node.decorator_list],
                        }
                    ))
                
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    symbols.append(Symbol(
                        name=node.name,
                        type='function',
                        file_path=rel_path,
                        line=node.lineno,
                        end_line=getattr(node, 'end_lineno', None),
                        signature=self._get_function_signature(node),
                        docstring=ast.get_docstring(node),
                        metadata={
                            'args': self._get_function_args(node),
                            'decorators': [self._get_name(d) for d in node.decorator_list],
                            'is_async': isinstance(node, ast.AsyncFunctionDef),
                        }
                    ))
                
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        symbols.append(Symbol(
                            name=alias.name,
                            type='import',
                            file_path=rel_path,
                            line=node.lineno,
                            signature=f"import {alias.name}",
                            metadata={'asname': alias.asname},
                        ))
                
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    for alias in node.names:
                        symbols.append(Symbol(
                            name=alias.name,
                            type='import',
                            file_path=rel_path,
                            line=node.lineno,
                            signature=f"from {module} import {alias.name}",
                            metadata={'module': module, 'asname': alias.asname},
                        ))
                
                elif isinstance(node, ast.Assign):
                    # Extract constants (simple heuristic)
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id.isupper():
                            symbols.append(Symbol(
                                name=target.id,
                                type='constant',
                                file_path=rel_path,
                                line=node.lineno,
                                signature=f"{target.id} = ...",
                            ))
        
        except Exception as e:
            console.print(f"[yellow]⚠️ Symbol extraction failed for {file_path}: {e}[/yellow]")
        
        return symbols

    def _get_name(self, node) -> str:
        """Get name from AST node"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_name(node.value)}.{node.attr}"
        elif isinstance(node, ast.Call):
            return self._get_name(node.func)
        return "?"

    def _get_function_signature(self, node) -> str:
        """Get function signature string"""
        args = self._get_function_args(node)
        async_prefix = "async " if isinstance(node, ast.AsyncFunctionDef) else ""
        return f"{async_prefix}def {node.name}({args})"

    def _get_function_args(self, node) -> str:
        """Get function arguments as string"""
        args = []
        for arg in node.args.args:
            arg_str = arg.arg
            if arg.annotation:
                arg_str += f": {self._get_name(arg.annotation)}"
            args.append(arg_str)
        
        if node.args.vararg:
            args.append(f"*{node.args.vararg.arg}")
        if node.args.kwarg:
            args.append(f"**{node.args.kwarg.arg}")
        
        return ", ".join(args)

    async def _process_file_async(
        self,
        file_path: Path,
        rel_path: str,
    ) -> Optional[FileMetadata]:
        """Process a single file asynchronously"""
        try:
            loop = asyncio.get_event_loop()
            
            def read_and_process():
                # Read file
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Calculate metadata
                stat = file_path.stat()
                content_hash = self._calculate_content_hash(content)
                line_count = content.count('\n') + 1
                token_estimate = self._estimate_tokens(content)
                
                # Extract symbols
                symbols = self._extract_symbols(file_path, content)
                
                # Split into chunks
                if self.text_splitter:
                    docs = self.text_splitter.create_documents(
                        [content],
                        metadatas=[{"source": rel_path, "filename": file_path.name}]
                    )
                else:
                    # Basic splitting
                    from langchain_core.documents import Document
                    chunks = [
                        content[i:i+self.chunk_size]
                        for i in range(0, len(content), self.chunk_size - self.chunk_overlap)
                    ]
                    docs = [
                        Document(page_content=chunk, metadata={"source": rel_path, "filename": file_path.name})
                        for chunk in chunks
                    ]
                
                return {
                    'content': content,
                    'docs': docs,
                    'symbols': symbols,
                    'stat': stat,
                    'content_hash': content_hash,
                    'line_count': line_count,
                    'token_estimate': token_estimate,
                }
            
            result = await loop.run_in_executor(None, read_and_process)
            
            # Create metadata
            metadata = FileMetadata(
                path=rel_path,
                size=result['stat'].st_size,
                modified_time=result['stat'].st_mtime,
                content_hash=result['content_hash'],
                indexed_time=datetime.now(),
                symbol_count=len(result['symbols']),
                line_count=result['line_count'],
                token_estimate=result['token_estimate'],
                status=IndexStatus.COMPLETED,
            )
            
            # Store symbols in manifest
            for symbol in result['symbols']:
                symbol_key = f"{rel_path}::{symbol.name}"
                self.manifest.symbols[symbol_key] = symbol
            
            # Add to vector store
            if result['docs']:
                if isinstance(self.vector_store, OptimizedVectorStore):
                    await self.vector_store.add_documents_batch(result['docs'])
                else:
                    self.vector_store.add_documents(result['docs'])
            
            # Extract graph relationships
            self.graph_extractor.extract_from_file(str(file_path))
            
            return metadata
        
        except Exception as e:
            console.print(f"[red]❌ Failed to process {file_path}: {e}[/red]")
            return None

    async def index_directory_async(
        self,
        root_path: str,
        incremental: bool = True,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Index a directory with incremental support.
        
        Args:
            root_path: Root directory to index
            incremental: If True, only index new/changed files
            verbose: Show progress indicators
        
        Returns:
            Indexing statistics
        """
        start_time = time.time()
        root = Path(root_path).resolve()
        self.project_root = root
        
        # Load or create manifest
        manifest_path = root / ".xencode" / "index_manifest.json"
        if incremental and manifest_path.exists():
            try:
                with open(manifest_path, 'r') as f:
                    manifest_data = json.load(f)
                self.manifest = IndexManifest.from_dict(manifest_data)
                if verbose:
                    console.print(f"[green]✓ Loaded existing manifest with {len(self.manifest.files)} files[/green]")
            except Exception as e:
                if verbose:
                    console.print(f"[yellow]⚠️ Could not load manifest: {e}, creating new[/yellow]")
                self.manifest = None
        
        if self.manifest is None:
            self.manifest = IndexManifest(
                project_root=str(root),
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )
            incremental = False  # Force full index
        
        # Detect changes
        if verbose:
            console.print(f"[blue]🔍 Scanning {root_path}...[/blue]")
        
        new_files, stale_files, removed_files = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self._detect_changes(root, self.DEFAULT_EXTENSIONS, self.DEFAULT_EXCLUDES)
        )
        
        if verbose:
            console.print(f"[green]✓ Found {len(new_files)} new, {len(stale_files)} stale, {len(removed_files)} removed files[/green]")
        
        # Remove deleted files from manifest
        for rel_path in removed_files:
            del self.manifest.files[rel_path]
        
        # Prepare files to index
        files_to_index = new_files + [Path(root) / f for f in stale_files]
        
        if not files_to_index:
            if verbose:
                console.print("[green]✓ All files up to date, no indexing needed[/green]")
            return {
                'indexed': 0,
                'new': len(new_files),
                'stale': len(stale_files),
                'removed': len(removed_files),
                'total_files': len(self.manifest.files),
                'time_taken': time.time() - start_time,
            }
        
        # Index files
        if verbose:
            console.print(f"[blue]📝 Indexing {len(files_to_index)} files...[/blue]")
        
        indexed_count = 0
        failed_count = 0
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Indexing...", total=len(files_to_index))
            
            # Process in batches
            batch_size = 10
            for i in range(0, len(files_to_index), batch_size):
                batch = files_to_index[i:i + batch_size]
                tasks = [
                    self._process_file_async(
                        file_path,
                        str(file_path.relative_to(root)) if file_path.is_relative_to(root) else str(file_path)
                    )
                    for file_path in batch
                ]
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for file_path, result in zip(batch, results):
                    rel_path = str(file_path.relative_to(root)) if file_path.is_relative_to(root) else str(file_path)
                    
                    if isinstance(result, Exception):
                        console.print(f"[red]❌ {rel_path}: {result}[/red]")
                        failed_count += 1
                    elif result:
                        self.manifest.files[rel_path] = result
                        indexed_count += 1
                    else:
                        failed_count += 1
                    
                    progress.advance(task)
        
        # Update manifest
        self.manifest.updated_at = datetime.now()
        self.manifest.total_tokens = sum(
            m.token_estimate for m in self.manifest.files.values()
        )
        
        # Store graph data
        nodes, rels = self.graph_extractor.get_data()
        for node_id, node_type, metadata in nodes:
            self.graph_store.add_node(node_id, node_type, metadata)
        for src, target, rel_type, metadata in rels:
            self.graph_store.add_relationship(src, target, rel_type, metadata)
        self.graph_store.persist()
        
        # Save manifest
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, 'w') as f:
            json.dump(self.manifest.to_dict(), f, indent=2)
        
        elapsed = time.time() - start_time
        
        if verbose:
            console.print(f"[green]✓ Indexed {indexed_count} files in {elapsed:.2f}s[/green]")
            console.print(f"   📊 Total files: {len(self.manifest.files)}")
            console.print(f"   📊 Total symbols: {len(self.manifest.symbols)}")
            console.print(f"   📊 Total tokens: {self.manifest.total_tokens:,}")
            if failed_count > 0:
                console.print(f"[yellow]⚠️ Failed: {failed_count}[/yellow]")
        
        return {
            'indexed': indexed_count,
            'new': len(new_files),
            'stale': len(stale_files),
            'removed': len(removed_files),
            'failed': failed_count,
            'total_files': len(self.manifest.files),
            'total_symbols': len(self.manifest.symbols),
            'total_tokens': self.manifest.total_tokens,
            'time_taken': elapsed,
        }

    def index_directory(
        self,
        root_path: str,
        incremental: bool = True,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Synchronous wrapper for index_directory_async"""
        return asyncio.run(self.index_directory_async(root_path, incremental, verbose))

    def search(
        self,
        query: str,
        k: int = 5,
        use_graph: bool = True,
        filter_by_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search indexed content.
        
        Args:
            query: Search query
            k: Number of results
            use_graph: Use graph-enhanced retrieval
            filter_by_type: Filter results by symbol type (class, function, etc.)
        
        Returns:
            List of search results with metadata
        """
        if use_graph and isinstance(self.vector_store, VectorStore):
            docs = self.vector_store.enhanced_similarity_search(query, k=k)
        else:
            docs = self.vector_store.similarity_search(query, k=k)
        
        results = []
        for doc in docs:
            source = doc.metadata.get('source', '')
            
            # Filter by symbol type if requested
            if filter_by_type and source in self.manifest.symbols:
                symbol = self.manifest.symbols[source]
                if symbol.type != filter_by_type:
                    continue
            
            result = {
                'content': doc.page_content,
                'source': source,
                'filename': doc.metadata.get('filename', ''),
                'metadata': doc.metadata,
            }
            
            # Add symbol info if available
            if source in self.manifest.symbols:
                result['symbol'] = self.manifest.symbols[source].to_dict()
            
            results.append(result)
        
        return results

    def get_symbol(self, name: str, file_path: Optional[str] = None) -> Optional[Symbol]:
        """Get symbol by name and optional file path"""
        if file_path:
            key = f"{file_path}::{name}"
            return self.manifest.symbols.get(key)
        
        # Search all symbols
        for key, symbol in self.manifest.symbols.items():
            if symbol.name == name:
                return symbol
        
        return None

    def get_file_info(self, file_path: str) -> Optional[FileMetadata]:
        """Get metadata for indexed file"""
        return self.manifest.files.get(file_path)

    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics"""
        if not self.manifest:
            return {'status': 'not_indexed'}
        
        # Count by symbol type
        symbol_counts = {}
        for symbol in self.manifest.symbols.values():
            symbol_counts[symbol.type] = symbol_counts.get(symbol.type, 0) + 1
        
        # Count by file extension
        ext_counts = {}
        for metadata in self.manifest.files.values():
            ext = Path(metadata.path).suffix
            ext_counts[ext] = ext_counts.get(ext, 0) + 1
        
        return {
            'total_files': len(self.manifest.files),
            'total_symbols': len(self.manifest.symbols),
            'total_tokens': self.manifest.total_tokens,
            'symbol_breakdown': symbol_counts,
            'file_types': ext_counts,
            'created_at': self.manifest.created_at.isoformat(),
            'updated_at': self.manifest.updated_at.isoformat(),
        }

    def clear_index(self) -> None:
        """Clear the index"""
        if isinstance(self.vector_store, VectorStore):
            self.vector_store.clear()
        self.graph_store.clear()
        self.manifest = None


# Import ast at module level for symbol extraction
import ast
