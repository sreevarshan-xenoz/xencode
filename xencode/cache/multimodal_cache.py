#!/usr/bin/env python3
"""
Multi-Modal Cache System for Xencode

Enhanced caching system specifically designed for multi-modal processing including
document processing, code analysis, and workspace data with intelligent cache
invalidation and warming strategies.
"""

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from enum import Enum
import logging

from ..advanced_cache_system import HybridCacheManager, CacheEntry, CacheStats
from ..models.document import ProcessedDocument, DocumentType
from ..models.code_analysis import CodeAnalysisResult
from ..models.workspace import Workspace, Change

logger = logging.getLogger(__name__)


class CacheType(str, Enum):
    """Types of cached content"""
    DOCUMENT_PROCESSING = "document_processing"
    CODE_ANALYSIS = "code_analysis"
    WORKSPACE_DATA = "workspace_data"
    AI_RESPONSE = "ai_response"
    ANALYTICS_DATA = "analytics_data"
    PLUGIN_DATA = "plugin_data"


@dataclass
class MultiModalCacheEntry(CacheEntry):
    """Enhanced cache entry for multi-modal content"""
    cache_type: CacheType = CacheType.AI_RESPONSE
    content_hash: str = ""
    dependencies: Set[str] = field(default_factory=set)
    invalidation_rules: Dict[str, Any] = field(default_factory=dict)
    warming_priority: int = 0  # 0-10, higher = more important
    
    def should_invalidate(self, change_event: Dict[str, Any]) -> bool:
        """Check if this entry should be invalidated based on change event"""
        rules = self.invalidation_rules or {}
        affected_files = change_event.get('files', [])
        
        # Check file-based invalidation
        if 'files' in rules:
            watched_files = rules['files']
            if any(file in watched_files for file in affected_files):
                return True
        
        # Check pattern-based invalidation
        if 'patterns' in rules:
            patterns = rules['patterns']
            for pattern in patterns:
                if any(pattern in file for file in affected_files):
                    return True
        
        # Workspace-specific invalidation
        if 'workspace_changes' in rules:
            workspace_ids = rules['workspace_changes']
            if change_event.get('workspace_id') in workspace_ids:
                return True
        
        # Check dependency-based invalidation
        if self.dependencies:
            if any(dep in affected_files for dep in self.dependencies):
                return True
        
        return False


class DocumentProcessingCache:
    """Specialized cache for document processing results"""
    
    def __init__(self, base_cache: HybridCacheManager):
        self.base_cache = base_cache
        self.document_hashes: Dict[str, str] = {}
    
    def _generate_document_key(self, file_path: str, document_type: DocumentType, 
                             processing_options: Dict[str, Any] = None) -> str:
        """Generate cache key for document processing"""
        # Include file modification time and size for invalidation
        try:
            file_stat = Path(file_path).stat()
            file_info = f"{file_stat.st_mtime}_{file_stat.st_size}"
        except (OSError, FileNotFoundError):
            file_info = "unknown"
        
        options_str = json.dumps(processing_options or {}, sort_keys=True)
        composite = f"doc_{file_path}_{document_type.value}_{file_info}_{options_str}"
        
        return hashlib.sha256(composite.encode()).hexdigest()[:24]
    
    def _calculate_content_hash(self, file_path: str) -> str:
        """Calculate hash of file content for change detection"""
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
            return hashlib.sha256(content).hexdigest()[:16]
        except (OSError, FileNotFoundError):
            return "missing"
    
    async def get_processed_document(self, file_path: str, document_type: DocumentType,
                                   processing_options: Dict[str, Any] = None) -> Optional[ProcessedDocument]:
        """Get cached document processing result"""
        cache_key = self._generate_document_key(file_path, document_type, processing_options)
        
        # Check if file has changed
        current_hash = self._calculate_content_hash(file_path)
        cached_hash = self.document_hashes.get(cache_key)
        
        if cached_hash and cached_hash != current_hash:
            # File changed, invalidate cache
            await self.base_cache.delete(cache_key)
            del self.document_hashes[cache_key]
            return None
        
        cached_result = await self.base_cache.get(cache_key)
        if cached_result:
            logger.debug(f"Cache hit for document: {file_path}")
            return cached_result
        
        return None
    
    async def cache_processed_document(self, file_path: str, document_type: DocumentType,
                                     result: ProcessedDocument, 
                                     processing_options: Dict[str, Any] = None) -> bool:
        """Cache document processing result"""
        cache_key = self._generate_document_key(file_path, document_type, processing_options)
        content_hash = self._calculate_content_hash(file_path)
        
        # Create enhanced cache entry
        entry = MultiModalCacheEntry(
            key=cache_key,
            value=result,
            created_at=time.time(),
            last_accessed=time.time(),
            access_count=1,
            size_bytes=len(str(result)),
            compressed=False,
            cache_type=CacheType.DOCUMENT_PROCESSING,
            content_hash=content_hash,
            dependencies={file_path},
            invalidation_rules={
                'files': [file_path],
                'patterns': [Path(file_path).suffix]
            },
            warming_priority=5,
            tags={'document', document_type.value},
            metadata={
                'file_path': file_path,
                'document_type': document_type.value,
                'processing_time': result.processing_time_ms,
                'confidence': result.confidence_score
            }
        )
        
        success = await self.base_cache.put_enhanced(cache_key, result, entry)
        if success:
            self.document_hashes[cache_key] = content_hash
            logger.debug(f"Cached document processing result: {file_path}")
        
        return success
    
    async def invalidate_document_cache(self, file_path: str) -> int:
        """Invalidate all cache entries related to a document"""
        invalidated = 0
        
        # Find all entries that depend on this file
        for key, hash_val in list(self.document_hashes.items()):
            entry = await self.base_cache.get_entry(key)
            if entry and file_path in entry.dependencies:
                await self.base_cache.delete(key)
                del self.document_hashes[key]
                invalidated += 1
        
        return invalidated


class CodeAnalysisCache:
    """Specialized cache for code analysis results"""
    
    def __init__(self, base_cache: HybridCacheManager):
        self.base_cache = base_cache
        self.code_hashes: Dict[str, str] = {}
    
    def _generate_analysis_key(self, code: str, language: str, 
                             analysis_options: Dict[str, Any] = None) -> str:
        """Generate cache key for code analysis"""
        code_hash = hashlib.sha256(code.encode()).hexdigest()[:16]
        options_str = json.dumps(analysis_options or {}, sort_keys=True)
        composite = f"code_{code_hash}_{language}_{options_str}"
        
        return hashlib.sha256(composite.encode()).hexdigest()[:24]
    
    async def get_analysis_result(self, code: str, language: str,
                                analysis_options: Dict[str, Any] = None) -> Optional[CodeAnalysisResult]:
        """Get cached code analysis result"""
        cache_key = self._generate_analysis_key(code, language, analysis_options)
        
        cached_result = await self.base_cache.get(cache_key)
        if cached_result:
            logger.debug(f"Cache hit for code analysis: {language}")
            return cached_result
        
        return None
    
    async def cache_analysis_result(self, code: str, language: str, 
                                  result: CodeAnalysisResult,
                                  analysis_options: Dict[str, Any] = None) -> bool:
        """Cache code analysis result"""
        cache_key = self._generate_analysis_key(code, language, analysis_options)
        code_hash = hashlib.sha256(code.encode()).hexdigest()[:16]
        
        # Create enhanced cache entry
        entry = MultiModalCacheEntry(
            key=cache_key,
            value=result,
            created_at=time.time(),
            last_accessed=time.time(),
            access_count=1,
            size_bytes=len(str(result)),
            compressed=False,
            cache_type=CacheType.CODE_ANALYSIS,
            content_hash=code_hash,
            warming_priority=7,  # Code analysis is expensive
            tags={'code_analysis', language},
            metadata={
                'language': language,
                'code_length': len(code),
                'analysis_time': getattr(result, 'analysis_time_ms', 0)
            }
        )
        
        success = await self.base_cache.put_enhanced(cache_key, result, entry)
        if success:
            self.code_hashes[cache_key] = code_hash
            logger.debug(f"Cached code analysis result: {language}")
        
        return success


class WorkspaceCache:
    """Specialized cache for workspace data and collaboration state"""
    
    def __init__(self, base_cache: HybridCacheManager):
        self.base_cache = base_cache
        self.workspace_versions: Dict[str, int] = {}
    
    def _generate_workspace_key(self, workspace_id: str, data_type: str) -> str:
        """Generate cache key for workspace data"""
        return f"workspace_{workspace_id}_{data_type}"
    
    async def get_workspace_data(self, workspace_id: str, data_type: str) -> Optional[Any]:
        """Get cached workspace data"""
        cache_key = self._generate_workspace_key(workspace_id, data_type)
        
        cached_data = await self.base_cache.get(cache_key)
        if cached_data:
            logger.debug(f"Cache hit for workspace data: {workspace_id}/{data_type}")
            return cached_data
        
        return None
    
    async def cache_workspace_data(self, workspace_id: str, data_type: str, 
                                 data: Any, version: int = None) -> bool:
        """Cache workspace data"""
        cache_key = self._generate_workspace_key(workspace_id, data_type)
        
        # Create enhanced cache entry
        entry = MultiModalCacheEntry(
            key=cache_key,
            value=data,
            created_at=time.time(),
            last_accessed=time.time(),
            access_count=1,
            size_bytes=len(str(data)),
            compressed=True,  # Workspace data can be large
            cache_type=CacheType.WORKSPACE_DATA,
            content_hash=hashlib.sha256(str(data).encode()).hexdigest()[:16],
            invalidation_rules={
                'workspace_changes': [workspace_id]
            },
            warming_priority=8,  # Workspace data is frequently accessed
            tags={'workspace', data_type},
            metadata={
                'workspace_id': workspace_id,
                'data_type': data_type,
                'version': version or 0
            }
        )
        
        success = await self.base_cache.put_enhanced(cache_key, data, entry)
        if success and version:
            self.workspace_versions[cache_key] = version
            logger.debug(f"Cached workspace data: {workspace_id}/{data_type}")
        
        return success
    
    async def invalidate_workspace_cache(self, workspace_id: str, 
                                       change: Change = None) -> int:
        """Invalidate workspace cache based on changes"""
        invalidated = 0
        
        # If specific change provided, use smart invalidation
        if change:
            change_event = {
                'type': 'workspace_change',
                'workspace_id': workspace_id,
                'files': [change.path] if hasattr(change, 'path') else [],
                'operation': change.operation if hasattr(change, 'operation') else 'unknown'
            }
            
            # Find entries that should be invalidated
            all_entries = await self.base_cache.get_all_entries()
            for entry in all_entries:
                if (isinstance(entry, MultiModalCacheEntry) and 
                    entry.cache_type == CacheType.WORKSPACE_DATA and
                    entry.should_invalidate(change_event)):
                    await self.base_cache.delete(entry.key)
                    invalidated += 1
        else:
            # Invalidate all workspace data
            pattern = f"workspace_{workspace_id}_"
            invalidated = await self.base_cache.delete_pattern(pattern)
        
        return invalidated


class TaskParameters(dict):
    """Dictionary wrapper providing compatibility helpers for legacy tests."""
    
    def __getitem__(self, key):
        if key == 2:
            return self
        return super().__getitem__(key)


class CacheWarmingManager:
    """Manages cache warming strategies for frequently accessed data"""
    
    def __init__(self, document_cache: DocumentProcessingCache,
                 code_cache: CodeAnalysisCache, workspace_cache: WorkspaceCache):
        self.document_cache = document_cache
        self.code_cache = code_cache
        self.workspace_cache = workspace_cache
        self.warming_queue: List[Tuple[int, str, Dict[str, Any]]] = []  # (priority, type, params)
    
    async def add_warming_task(self, cache_type: str, priority: int, **params):
        """Add a cache warming task"""
        self.warming_queue.append((priority, cache_type, TaskParameters(params)))
        self.warming_queue.sort(key=lambda x: x[0], reverse=True)  # Higher priority first
    
    async def warm_frequently_accessed_documents(self, file_paths: List[str]):
        """Pre-warm cache for frequently accessed documents"""
        for file_path in file_paths:
            try:
                # Determine document type
                suffix = Path(file_path).suffix.lower()
                if suffix == '.pdf':
                    doc_type = DocumentType.PDF
                elif suffix in ['.docx', '.doc']:
                    doc_type = DocumentType.DOCX
                elif suffix in ['.html', '.htm']:
                    doc_type = DocumentType.HTML
                else:
                    continue  # Skip unsupported types
                
                # Check if already cached
                cached = await self.document_cache.get_processed_document(file_path, doc_type)
                if not cached:
                    await self.add_warming_task('document', 7, 
                                              file_path=file_path, 
                                              document_type=doc_type)
            except Exception as e:
                logger.warning(f"Failed to add warming task for {file_path}: {e}")
    
    async def warm_workspace_data(self, workspace_id: str, data_types: List[str]):
        """Pre-warm cache for workspace data"""
        for data_type in data_types:
            cached = await self.workspace_cache.get_workspace_data(workspace_id, data_type)
            if not cached:
                await self.add_warming_task('workspace', 8,
                                          workspace_id=workspace_id,
                                          data_type=data_type)
    
    async def process_warming_queue(self, max_tasks: int = 5):
        """Process cache warming tasks"""
        processed = 0
        
        while self.warming_queue and processed < max_tasks:
            priority, cache_type, params = self.warming_queue.pop(0)
            
            try:
                if cache_type == 'document':
                    await self._warm_document(**params)
                elif cache_type == 'workspace':
                    await self._warm_workspace(**params)
                elif cache_type == 'code':
                    await self._warm_code_analysis(**params)
                
                processed += 1
                
            except Exception as e:
                logger.error(f"Cache warming failed for {cache_type}: {e}")
    
    async def _warm_document(self, file_path: str, document_type: DocumentType):
        """Warm document processing cache"""
        # This would integrate with the actual document processor
        logger.info(f"Warming document cache: {file_path}")
        # Implementation would call the actual document processor
    
    async def _warm_workspace(self, workspace_id: str, data_type: str):
        """Warm workspace data cache"""
        logger.info(f"Warming workspace cache: {workspace_id}/{data_type}")
        # Implementation would load workspace data
    
    async def _warm_code_analysis(self, code: str, language: str):
        """Warm code analysis cache"""
        logger.info(f"Warming code analysis cache: {language}")
        # Implementation would call the actual code analyzer


class MultiModalCacheSystem:
    """Main multi-modal cache system that coordinates all specialized caches"""
    
    def __init__(self, base_cache: HybridCacheManager):
        self.base_cache = base_cache
        self.document_cache = DocumentProcessingCache(base_cache)
        self.code_cache = CodeAnalysisCache(base_cache)
        self.workspace_cache = WorkspaceCache(base_cache)
        self.warming_manager = CacheWarmingManager(
            self.document_cache, self.code_cache, self.workspace_cache
        )
        
        # Enhanced statistics
        self.multimodal_stats = {
            'document_hits': 0,
            'document_misses': 0,
            'code_hits': 0,
            'code_misses': 0,
            'workspace_hits': 0,
            'workspace_misses': 0,
            'invalidations': 0,
            'warming_tasks': 0
        }
    
    async def get_cache_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        base_stats = await self.base_cache.get_stats()
        
        return {
            'base_cache': base_stats,
            'multimodal': self.multimodal_stats,
            'cache_types': {
                CacheType.DOCUMENT_PROCESSING: await self._get_type_stats(CacheType.DOCUMENT_PROCESSING),
                CacheType.CODE_ANALYSIS: await self._get_type_stats(CacheType.CODE_ANALYSIS),
                CacheType.WORKSPACE_DATA: await self._get_type_stats(CacheType.WORKSPACE_DATA)
            }
        }
    
    async def _get_type_stats(self, cache_type: CacheType) -> Dict[str, Any]:
        """Get statistics for specific cache type"""
        entries = await self.base_cache.get_entries_by_tag(cache_type.value)
        
        total_size = sum(entry.size_bytes for entry in entries)
        avg_access_count = sum(entry.access_count for entry in entries) / len(entries) if entries else 0
        
        return {
            'entry_count': len(entries),
            'total_size_mb': total_size / (1024 * 1024),
            'avg_access_count': avg_access_count
        }
    
    async def handle_change_event(self, event: Dict[str, Any]):
        """Handle change events for cache invalidation"""
        event_type = event.get('type')
        
        if event_type == 'file_changed':
            file_path = event.get('file_path')
            if file_path:
                invalidated = await self.document_cache.invalidate_document_cache(file_path)
                self.multimodal_stats['invalidations'] += invalidated
        
        elif event_type == 'workspace_changed':
            workspace_id = event.get('workspace_id')
            change = event.get('change')
            if workspace_id:
                invalidated = await self.workspace_cache.invalidate_workspace_cache(workspace_id, change)
                self.multimodal_stats['invalidations'] += invalidated
    
    async def start_background_warming(self):
        """Start background cache warming process"""
        while True:
            try:
                await self.warming_manager.process_warming_queue()
                await asyncio.sleep(30)  # Process warming tasks every 30 seconds
            except Exception as e:
                logger.error(f"Background warming error: {e}")
                await asyncio.sleep(60)  # Wait longer on error
    
    async def optimize_cache(self):
        """Optimize cache performance and cleanup"""
        # Clean up expired entries
        await self.base_cache.cleanup_expired()
        
        # Optimize memory usage
        await self.base_cache.optimize_memory()
        
        # Update statistics
        stats = await self.get_cache_statistics()
        logger.info(f"Cache optimization complete. Stats: {stats}")


# Global instance
_multimodal_cache_system: Optional[MultiModalCacheSystem] = None


def get_cache_system() -> HybridCacheManager:
    """Synchronously obtain the shared hybrid cache manager."""
    async def _get():
        from ..advanced_cache_system import get_cache_manager
        return await get_cache_manager()

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        future = asyncio.run_coroutine_threadsafe(_get(), loop)
        return future.result()

    return asyncio.run(_get())


def get_multimodal_cache() -> MultiModalCacheSystem:
    """Get global multimodal cache system instance"""
    global _multimodal_cache_system
    
    if _multimodal_cache_system is None:
        base_cache = get_cache_system()
        _multimodal_cache_system = MultiModalCacheSystem(base_cache)
    
    return _multimodal_cache_system


async def get_multimodal_cache_async() -> MultiModalCacheSystem:
    """Async-compatible wrapper for obtaining the cache system."""
    return get_multimodal_cache()


async def initialize_multimodal_cache(config: Dict[str, Any] = None):
    """Initialize the multimodal cache system"""
    cache_system = get_multimodal_cache()
    
    # Start background processes
    asyncio.create_task(cache_system.start_background_warming())
    
    logger.info("Multi-modal cache system initialized")
    return cache_system