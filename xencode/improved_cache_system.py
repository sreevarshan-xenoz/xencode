#!/usr/bin/env python3
"""
Improved Advanced Response Caching System for Xencode Phase 2

High-performance caching with intelligent cache management, compression,
and multiple storage backends for optimal response times.
Fixed: All synchronous operations converted to async operations.
"""

import asyncio
import hashlib
import json
import lzma
import pickle
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import aiofiles
import psutil
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
# import aiosqlite  # Commented out to avoid dependency issue
import sqlite3  # Use standard library instead
from concurrent.futures import ThreadPoolExecutor

console = Console()

DEFAULT_MEMORY_CACHE_MB = 256
DEFAULT_DISK_CACHE_MB = 1024
COMPRESSION_THRESHOLD_BYTES = 1024
CACHE_EXPIRY_SECONDS = 86400


@dataclass
class CacheEntry:
    """Represents a cached response with metadata"""
    key: str
    value: Any
    created_at: float
    last_accessed: float
    access_count: int
    size_bytes: int
    compressed: bool
    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CacheStats:
    """Cache performance statistics"""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_entries: int = 0
    memory_usage_mb: float = 0.0
    disk_usage_mb: float = 0.0
    compression_ratio: float = 0.0
    avg_response_time_ms: float = 0.0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return (self.hits / total * 100) if total > 0 else 0.0

    @property
    def efficiency_score(self) -> float:
        """Combined efficiency score (0-100)"""
        hit_score = min(self.hit_rate, 100)
        speed_score = max(0, 100 - (self.avg_response_time_ms / 10))
        compression_score = min(self.compression_ratio * 10, 100)
        return (hit_score * 0.5 + speed_score * 0.3 + compression_score * 0.2)


class CacheKeyGenerator:
    """Generates consistent cache keys from various inputs"""

    @staticmethod
    def generate_key(prompt: str, model: str, parameters: Dict[str, Any] = None) -> str:
        """Generate a consistent cache key"""
        # Normalize inputs
        prompt_normalized = prompt.strip().lower()
        model_normalized = model.strip().lower()

        # Include relevant parameters
        if parameters:
            # Only include parameters that affect output
            relevant_params = {
                k: v for k, v in parameters.items()
                if k in ['temperature', 'top_p', 'top_k', 'max_tokens', 'system_prompt']
            }
            params_str = json.dumps(relevant_params, sort_keys=True)
        else:
            params_str = ""

        # Create composite string
        composite = f"{prompt_normalized}|{model_normalized}|{params_str}"

        # Generate hash
        return hashlib.sha256(composite.encode()).hexdigest()[:16]

    @staticmethod
    def generate_context_key(conversation_history: List[Dict[str, str]],
                           model: str, max_history: int = 5) -> str:
        """Generate key for conversation context"""
        # Take last N messages for context
        recent_history = conversation_history[-max_history:] if conversation_history else []

        # Create normalized history string
        history_parts = []
        for msg in recent_history:
            role = msg.get('role', '').strip().lower()
            content = msg.get('content', '').strip().lower()
            history_parts.append(f"{role}:{content}")

        history_str = "|".join(history_parts)
        composite = f"{history_str}|{model.strip().lower()}"

        return hashlib.sha256(composite.encode()).hexdigest()[:16]


class CompressionManager:
    """Handles data compression/decompression for cache entries"""

    @staticmethod
    def compress_data(data: Any) -> Tuple[bytes, bool]:
        """Compress data if beneficial"""
        # Serialize data
        serialized = pickle.dumps(data)
        original_size = len(serialized)

        # Only compress if data is large enough
        if original_size < COMPRESSION_THRESHOLD_BYTES:
            return serialized, False

        # Compress with LZMA
        compressed = lzma.compress(serialized, preset=1)
        compression_ratio = len(compressed) / original_size

        # Use compressed version if it saves significant space
        if compression_ratio < 0.8:  # 20% savings threshold
            return compressed, True
        else:
            return serialized, False

    @staticmethod
    def decompress_data(data: bytes, compressed: bool) -> Any:
        """Decompress and deserialize data"""
        if compressed:
            decompressed = lzma.decompress(data)
            return pickle.loads(decompressed)
        else:
            return pickle.loads(data)


class MemoryCache:
    """In-memory LRU cache with size limits"""

    def __init__(self, max_size_mb: int = DEFAULT_MEMORY_CACHE_MB):
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.cache: Dict[str, CacheEntry] = {}
        self.access_order: List[str] = []
        self.current_size = 0
        self.stats = CacheStats()

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        start_time = time.perf_counter()

        if key in self.cache:
            entry = self.cache[key]
            entry.last_accessed = time.time()
            entry.access_count += 1

            # Move to end (most recently used)
            self.access_order.remove(key)
            self.access_order.append(key)

            self.stats.hits += 1
            response_time = (time.perf_counter() - start_time) * 1000
            self._update_avg_response_time(response_time)

            return entry.value

        self.stats.misses += 1
        return None

    def put(self, key: str, value: Any, tags: Set[str] = None,
            metadata: Dict[str, Any] = None) -> bool:
        """Store value in cache"""
        # Compress if beneficial
        compressed_data, is_compressed = CompressionManager.compress_data(value)
        size_bytes = len(compressed_data)

        # Check if it fits
        if size_bytes > self.max_size_bytes:
            return False  # Too large to cache

        # Evict if necessary
        while (self.current_size + size_bytes > self.max_size_bytes and
               self.access_order):
            self._evict_lru()

        # Create cache entry
        entry = CacheEntry(
            key=key,
            value=value,
            created_at=time.time(),
            last_accessed=time.time(),
            access_count=1,
            size_bytes=size_bytes,
            compressed=is_compressed,
            tags=tags or set(),
            metadata=metadata or {}
        )

        # Store entry
        self.cache[key] = entry
        self.access_order.append(key)
        self.current_size += size_bytes
        self.stats.total_entries = len(self.cache)
        self.stats.memory_usage_mb = self.current_size / (1024 * 1024)

        return True

    def _evict_lru(self):
        """Evict least recently used entry"""
        if not self.access_order:
            return

        lru_key = self.access_order.pop(0)
        if lru_key in self.cache:
            entry = self.cache.pop(lru_key)
            self.current_size -= entry.size_bytes
            self.stats.evictions += 1

    def _update_avg_response_time(self, response_time_ms: float):
        """Update average response time"""
        total_requests = self.stats.hits + self.stats.misses
        if total_requests == 1:
            self.stats.avg_response_time_ms = response_time_ms
        else:
            # Exponential moving average
            alpha = 0.1
            self.stats.avg_response_time_ms = (
                alpha * response_time_ms +
                (1 - alpha) * self.stats.avg_response_time_ms
            )

    def clear_by_tags(self, tags: Set[str]):
        """Clear entries matching any of the given tags"""
        keys_to_remove = []
        for key, entry in self.cache.items():
            if entry.tags.intersection(tags):
                keys_to_remove.append(key)

        for key in keys_to_remove:
            self._remove_key(key)

    def _remove_key(self, key: str):
        """Remove specific key from cache"""
        if key in self.cache:
            entry = self.cache.pop(key)
            self.current_size -= entry.size_bytes
            if key in self.access_order:
                self.access_order.remove(key)

    def get_stats(self) -> CacheStats:
        """Get current cache statistics"""
        self.stats.total_entries = len(self.cache)
        self.stats.memory_usage_mb = self.current_size / (1024 * 1024)
        return self.stats


class AsyncDiskCache:
    """Async persistent disk-based cache with SQLite backend"""

    def __init__(self, cache_dir: Path, max_size_mb: int = DEFAULT_DISK_CACHE_MB):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.cache_dir / "cache_async.db"
        self.data_dir = self.cache_dir / "data"
        self.data_dir.mkdir(exist_ok=True)
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.stats = CacheStats()
        self.executor = ThreadPoolExecutor(max_workers=4)  # Thread pool for blocking operations

        # Initialize database
        asyncio.create_task(self._init_database())

    async def _init_database(self):
        """Initialize SQLite database asynchronously"""
        def db_operation():
            with sqlite3.connect(self.db_path) as db:
                db.execute("""
                    CREATE TABLE IF NOT EXISTS cache_entries (
                        key TEXT PRIMARY KEY,
                        created_at REAL,
                        last_accessed REAL,
                        access_count INTEGER,
                        size_bytes INTEGER,
                        compressed BOOLEAN,
                        tags TEXT,
                        metadata TEXT,
                        data_file TEXT
                    )
                """)
                db.execute("""
                    CREATE INDEX IF NOT EXISTS idx_last_accessed
                    ON cache_entries(last_accessed)
                """)
                db.execute("""
                    CREATE INDEX IF NOT EXISTS idx_created_at
                    ON cache_entries(created_at)
                """)
                db.commit()

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, db_operation)

    async def get(self, key: str) -> Optional[Any]:
        """Get value from disk cache asynchronously"""
        start_time = time.perf_counter()

        def db_operation():
            with sqlite3.connect(self.db_path) as db:
                cursor = db.execute(
                    "SELECT data_file, compressed, access_count FROM cache_entries WHERE key = ?",
                    (key,)
                )
                row = cursor.fetchone()
                return row

        loop = asyncio.get_event_loop()
        row = await loop.run_in_executor(None, db_operation)

        if row:
            data_file, compressed, access_count = row
            data_path = self.data_dir / data_file

            # Check if file exists
            loop = asyncio.get_event_loop()
            exists = await loop.run_in_executor(self.executor, data_path.exists)

            if exists:
                # Load and decompress data
                async with aiofiles.open(data_path, 'rb') as f:
                    data = await f.read()

                value = CompressionManager.decompress_data(data, compressed)

                # Update access statistics
                def update_stats():
                    with sqlite3.connect(self.db_path) as db:
                        db.execute(
                            "UPDATE cache_entries SET last_accessed = ?, access_count = ? WHERE key = ?",
                            (time.time(), access_count + 1, key)
                        )
                        db.commit()

                await loop.run_in_executor(None, update_stats)

                self.stats.hits += 1
                response_time = (time.perf_counter() - start_time) * 1000
                self._update_avg_response_time(response_time)

                return value

        self.stats.misses += 1
        return None

    async def put(self, key: str, value: Any, tags: Set[str] = None,
                  metadata: Dict[str, Any] = None) -> bool:
        """Store value in disk cache asynchronously"""
        # Compress data
        compressed_data, is_compressed = CompressionManager.compress_data(value)
        size_bytes = len(compressed_data)

        # Check size limits
        if size_bytes > self.max_size_bytes * 0.1:  # Max 10% of cache for single entry
            return False

        # Cleanup if needed
        await self._cleanup_if_needed(size_bytes)

        # Save data to file
        data_file = f"{key}.cache"
        data_path = self.data_dir / data_file

        async with aiofiles.open(data_path, 'wb') as f:
            await f.write(compressed_data)

        # Store metadata in database
        tags_str = json.dumps(list(tags)) if tags else "[]"
        metadata_str = json.dumps(metadata) if metadata else "{}"

        def db_operation():
            with sqlite3.connect(self.db_path) as db:
                db.execute("""
                    INSERT OR REPLACE INTO cache_entries
                    (key, created_at, last_accessed, access_count, size_bytes,
                     compressed, tags, metadata, data_file)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    key, time.time(), time.time(), 1, size_bytes,
                    is_compressed, tags_str, metadata_str, data_file
                ))
                db.commit()

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, db_operation)

        self.stats.total_entries += 1
        return True

    async def _cleanup_if_needed(self, new_entry_size: int):
        """Clean up old entries if cache is full"""
        current_size = await self._get_current_size()

        if current_size + new_entry_size > self.max_size_bytes:
            # Remove oldest entries until we have space
            def db_operation():
                nonlocal current_size  # Declare that we're modifying the outer current_size variable
                with sqlite3.connect(self.db_path) as db:
                    cursor = db.execute("""
                        SELECT key, data_file, size_bytes
                        FROM cache_entries
                        ORDER BY last_accessed ASC
                    """)

                    rows = cursor.fetchall()
                    for row in rows:
                        key, data_file, size_bytes = row

                        # Remove data file
                        data_path = self.data_dir / data_file
                        loop = asyncio.get_event_loop()
                        # Synchronously remove the file
                        if data_path.exists():
                            data_path.unlink()

                        # Remove database entry
                        db.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
                        db.commit()

                        current_size -= size_bytes
                        self.stats.evictions += 1

                        if current_size + new_entry_size <= self.max_size_bytes:
                            break

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, db_operation)

    async def _get_current_size(self) -> int:
        """Get current cache size in bytes"""
        def db_operation():
            with sqlite3.connect(self.db_path) as db:
                cursor = db.execute("SELECT SUM(size_bytes) FROM cache_entries")
                result = cursor.fetchone()
                return result[0] if result[0] else 0

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, db_operation)

    def _update_avg_response_time(self, response_time_ms: float):
        """Update average response time"""
        total_requests = self.stats.hits + self.stats.misses
        if total_requests == 1:
            self.stats.avg_response_time_ms = response_time_ms
        else:
            alpha = 0.1
            self.stats.avg_response_time_ms = (
                alpha * response_time_ms +
                (1 - alpha) * self.stats.avg_response_time_ms
            )

    async def clear_by_tags(self, tags: Set[str]):
        """Clear entries matching any of the given tags"""
        def db_operation():
            with sqlite3.connect(self.db_path) as db:
                cursor = db.execute("SELECT key, data_file, tags FROM cache_entries")
                keys_to_remove = []

                rows = cursor.fetchall()
                for row in rows:
                    key, data_file, entry_tags_str = row
                    entry_tags = set(json.loads(entry_tags_str))

                    if entry_tags.intersection(tags):
                        keys_to_remove.append((key, data_file))

                for key, data_file in keys_to_remove:
                    # Remove data file
                    data_path = self.data_dir / data_file
                    if data_path.exists():
                        data_path.unlink()

                    # Remove database entry
                    db.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
                    db.commit()

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, db_operation)

    def get_stats(self) -> CacheStats:
        """Get current disk cache statistics"""
        # Since this is called from sync context, we'll return cached stats
        # For full async stats, use get_stats_async
        return self.stats

    async def get_stats_async(self) -> CacheStats:
        """Get current disk cache statistics - async version"""
        self.stats.disk_usage_mb = await self._get_current_size() / (1024 * 1024)

        def db_operation():
            with sqlite3.connect(self.db_path) as db:
                cursor = db.execute("SELECT COUNT(*) FROM cache_entries")
                count = cursor.fetchone()
                return count[0] if count else 0

        loop = asyncio.get_event_loop()
        count = await loop.run_in_executor(None, db_operation)
        self.stats.total_entries = count

        return self.stats

    async def get_entry(self, key: str) -> Optional[CacheEntry]:
        """Get cache entry with metadata"""
        def db_operation():
            with sqlite3.connect(self.db_path) as db:
                cursor = db.execute("""
                    SELECT created_at, last_accessed, access_count, size_bytes,
                           compressed, tags, metadata
                    FROM cache_entries WHERE key = ?
                """, (key,))
                row = cursor.fetchone()
                return row

        loop = asyncio.get_event_loop()
        row = await loop.run_in_executor(None, db_operation)

        if row:
            created_at, last_accessed, access_count, size_bytes, compressed, tags_str, metadata_str = row

            # Get the actual value
            value = await self.get(key)
            if value is not None:
                return CacheEntry(
                    key=key,
                    value=value,
                    created_at=created_at,
                    last_accessed=last_accessed,
                    access_count=access_count,
                    size_bytes=size_bytes,
                    compressed=compressed,
                    tags=set(json.loads(tags_str)),
                    metadata=json.loads(metadata_str)
                )

        return None

    async def delete(self, key: str) -> bool:
        """Delete a specific cache entry"""
        def db_operation():
            with sqlite3.connect(self.db_path) as db:
                cursor = db.execute("SELECT data_file FROM cache_entries WHERE key = ?", (key,))
                row = cursor.fetchone()

                if row:
                    data_file = row[0]
                    data_path = self.data_dir / data_file
                    if data_path.exists():
                        data_path.unlink()

                    db.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
                    db.commit()
                    return True
            return False

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, db_operation)


class ImprovedHybridCacheManager:
    """Improved hybrid cache with async disk operations"""

    def __init__(self,
                 memory_cache_mb: int = DEFAULT_MEMORY_CACHE_MB,
                 disk_cache_mb: int = DEFAULT_DISK_CACHE_MB,
                 cache_dir: Optional[Path] = None):

        self.cache_dir = cache_dir or Path.home() / ".xencode" / "cache"
        self.memory_cache = MemoryCache(memory_cache_mb)
        self.disk_cache = AsyncDiskCache(self.cache_dir, disk_cache_mb)
        self.key_generator = CacheKeyGenerator()

        # Thread safety locks
        self._memory_lock = asyncio.Lock()
        self._disk_lock = asyncio.Lock()

        # Performance tracking
        self.total_requests = 0
        self.cache_hits = 0
        self.memory_hits = 0
        self.disk_hits = 0

        console.print(f"[green]✅ Improved hybrid cache initialized: {memory_cache_mb}MB memory + {disk_cache_mb}MB disk[/green]")

    async def get_response(self, prompt: str, model: str,
                          parameters: Dict[str, Any] = None) -> Optional[Any]:
        """Get cached response for a prompt"""
        key = self.key_generator.generate_key(prompt, model, parameters)
        self.total_requests += 1

        # Try memory cache first (fastest)
        async with self._memory_lock:
            result = self.memory_cache.get(key)
            if result is not None:
                self.cache_hits += 1
                self.memory_hits += 1
                return result

        # Try disk cache (slower but persistent)
        async with self._disk_lock:
            result = await self.disk_cache.get(key)
            if result is not None:
                # Promote to memory cache for future fast access
                async with self._memory_lock:
                    self.memory_cache.put(key, result)
                self.cache_hits += 1
                self.disk_hits += 1
                return result

        return None

    async def store_response(self, prompt: str, model: str, response: Any,
                           parameters: Dict[str, Any] = None,
                           tags: Set[str] = None) -> bool:
        """Store response in cache"""
        key = self.key_generator.generate_key(prompt, model, parameters)

        # Add automatic tags
        auto_tags = {f"model:{model}", f"type:response"}
        if tags:
            auto_tags.update(tags)

        # Store in both caches
        async with self._memory_lock:
            memory_success = self.memory_cache.put(key, response, auto_tags)
        async with self._disk_lock:
            disk_success = await self.disk_cache.put(key, response, auto_tags)

        return memory_success or disk_success

    async def get_context_response(self, conversation_history: List[Dict[str, str]],
                                 model: str, max_history: int = 5) -> Optional[Any]:
        """Get cached response for conversation context"""
        key = self.key_generator.generate_context_key(conversation_history, model, max_history)
        self.total_requests += 1

        # Try memory first
        result = self.memory_cache.get(key)
        if result is not None:
            self.cache_hits += 1
            self.memory_hits += 1
            return result

        # Try disk
        result = await self.disk_cache.get(key)
        if result is not None:
            self.memory_cache.put(key, result)
            self.cache_hits += 1
            self.disk_hits += 1
            return result

        return None

    async def store_context_response(self, conversation_history: List[Dict[str, str]],
                                   model: str, response: Any,
                                   max_history: int = 5) -> bool:
        """Store response for conversation context"""
        key = self.key_generator.generate_context_key(conversation_history, model, max_history)
        tags = {f"model:{model}", "type:context"}

        memory_success = self.memory_cache.put(key, response, tags)
        disk_success = await self.disk_cache.put(key, response, tags)

        return memory_success or disk_success

    async def invalidate_model(self, model: str):
        """Invalidate all cache entries for a specific model"""
        tags = {f"model:{model}"}
        self.memory_cache.clear_by_tags(tags)
        await self.disk_cache.clear_by_tags(tags)

    async def invalidate_by_tags(self, tags: Set[str]):
        """Invalidate cache entries by tags"""
        self.memory_cache.clear_by_tags(tags)
        await self.disk_cache.clear_by_tags(tags)

    async def cleanup_expired(self):
        """Clean up expired cache entries"""
        await self._cleanup_old_entries()

    async def _cleanup_old_entries(self):
        """Internal method to clean up old entries"""
        # For now, just clean up memory cache
        # In a real implementation, we'd check timestamps
        pass

    async def optimize_memory(self):
        """Optimize memory usage"""
        await self._reduce_memory_cache()

    async def _reduce_memory_cache(self):
        """Internal method to reduce memory cache size if needed"""
        # If memory usage is too high, evict some entries
        target_size = self.memory_cache.max_size_bytes * 0.8  # Target 80% of max size
        while self.memory_cache.current_size > target_size and self.memory_cache.access_order:
            self.memory_cache._evict_lru()

    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache performance statistics"""
        memory_stats = self.memory_cache.get_stats()
        disk_stats = await self.disk_cache.get_stats_async()

        total_hit_rate = (self.cache_hits / self.total_requests * 100) if self.total_requests > 0 else 0
        memory_hit_rate = (self.memory_hits / self.total_requests * 100) if self.total_requests > 0 else 0
        disk_hit_rate = (self.disk_hits / self.total_requests * 100) if self.total_requests > 0 else 0

        return {
            "total_requests": self.total_requests,
            "total_hit_rate": total_hit_rate,
            "memory_hit_rate": memory_hit_rate,
            "disk_hit_rate": disk_hit_rate,
            "memory_cache": {
                "entries": memory_stats.total_entries,
                "size_mb": memory_stats.memory_usage_mb,
                "hit_rate": memory_stats.hit_rate,
                "avg_response_ms": memory_stats.avg_response_time_ms
            },
            "disk_cache": {
                "entries": disk_stats.total_entries,
                "size_mb": disk_stats.disk_usage_mb,
                "hit_rate": disk_stats.hit_rate,
                "avg_response_ms": disk_stats.avg_response_time_ms
            },
            "efficiency_score": (memory_stats.efficiency_score + disk_stats.efficiency_score) / 2
        }

    async def optimize_cache(self):
        """Run cache optimization routines"""
        await self.optimize_memory()

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache (compatibility method)"""
        # Check memory cache first
        if key in self.memory_cache.cache:
            entry = self.memory_cache.cache[key]
            entry.last_accessed = time.time()
            entry.access_count += 1
            return entry.value

        # Check disk cache
        return await self.disk_cache.get(key)

    async def delete(self, key: str) -> bool:
        """Delete cache entry"""
        deleted = False

        # Remove from memory cache
        if key in self.memory_cache.cache:
            self.memory_cache._remove_key(key)
            deleted = True

        # Remove from disk cache
        result = await self.disk_cache.delete(key)
        return deleted or result

    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        memory_stats = self.memory_cache.get_stats()
        disk_stats = await self.disk_cache.get_stats_async()

        return {
            'memory_usage_mb': memory_stats.memory_usage_mb,
            'disk_usage_mb': disk_stats.disk_usage_mb,
            'hit_rate': ((self.memory_hits + self.disk_hits) / max(1, self.total_requests)) * 100,
            'total_entries': memory_stats.total_entries + disk_stats.total_entries,
            'memory_hits': self.memory_hits,
            'disk_hits': self.disk_hits,
            'total_requests': self.total_requests
        }


class AdvancedCacheSystem(ImprovedHybridCacheManager):
    """Alias for ImprovedHybridCacheManager to maintain compatibility"""
    pass


# Global cache instance
_cache_manager: Optional[ImprovedHybridCacheManager] = None


async def get_cache_manager(memory_mb: int = DEFAULT_MEMORY_CACHE_MB, disk_mb: int = DEFAULT_DISK_CACHE_MB) -> ImprovedHybridCacheManager:
    """Get or create global cache manager"""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = ImprovedHybridCacheManager(memory_mb, disk_mb)
    return _cache_manager


async def cached_response(prompt: str, model: str, response_func,
                         parameters: Dict[str, Any] = None,
                         tags: Set[str] = None) -> Any:
    """Decorator-like function for caching AI responses"""
    cache = await get_cache_manager()

    # Try to get cached response
    cached = await cache.get_response(prompt, model, parameters)
    if cached is not None:
        return cached

    # Generate new response and cache it
    response = await response_func()
    await cache.store_response(prompt, model, response, parameters, tags)

    return response


if __name__ == "__main__":
    # Demo and testing
    async def demo():
        cache = await get_cache_manager()

        # Demo responses
        responses = [
            ("Hello world", "llama3.1:8b", "Hello! How can I help you today?"),
            ("What is Python?", "llama3.1:8b", "Python is a programming language..."),
            ("Hello world", "llama3.1:8b", "This should be cached!"),  # Duplicate
        ]

        console.print("[bold blue]🚀 Improved Cache Performance Demo[/bold blue]")

        for prompt, model, response in responses:
            # Check cache first
            cached = await cache.get_response(prompt, model)
            if cached:
                console.print(f"[green]✅ Cache HIT: {prompt[:30]}...[/green]")
            else:
                console.print(f"[yellow]⚠️  Cache MISS: {prompt[:30]}...[/yellow]")
                await cache.store_response(prompt, model, response)

        # Show statistics
        stats = await cache.get_performance_stats()
        console.print(f"\n[bold]Cache Performance:[/bold]")
        console.print(f"Hit Rate: {stats['total_hit_rate']:.1f}%")
        console.print(f"Memory Entries: {stats['memory_cache']['entries']}")
        console.print(f"Disk Entries: {stats['disk_cache']['entries']}")

    asyncio.run(demo())