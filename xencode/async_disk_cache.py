#!/usr/bin/env python3
"""
Async Disk Cache Implementation

Asynchronous version of disk cache to fix synchronous operations in async contexts.
FIXED VERSION: Uses standard sqlite3 with thread pool for async operations.
"""

import asyncio
import json
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
import aiofiles
from .advanced_cache_system import CacheEntry, CacheStats, CompressionManager, COMPRESSION_THRESHOLD_BYTES


@dataclass
class AsyncCacheEntry:
    """Async version of cache entry"""
    key: str
    value: Any
    created_at: float
    last_accessed: float
    access_count: int
    size_bytes: int
    compressed: bool
    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)


class AsyncDiskCache:
    """Async version of disk cache with proper async operations"""

    def __init__(self, cache_dir: Path, max_size_mb: int = 1024):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.cache_dir / "cache_async.db"
        self.data_dir = self.cache_dir / "data_async"
        self.data_dir.mkdir(exist_ok=True)
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.stats = CacheStats()

        # Initialize database schema
        self._init_database()

    def _init_database(self):
        """Initialize SQLite database synchronously"""
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

    async def get(self, key: str) -> Optional[Any]:
        """Get value from disk cache asynchronously"""
        start_time = time.perf_counter()

        # Run the synchronous database operation in a thread pool
        loop = asyncio.get_event_loop()
        import concurrent.futures
        
        def db_operation():
            with sqlite3.connect(self.db_path) as db:
                cursor = db.execute(
                    "SELECT data_file, compressed, access_count FROM cache_entries WHERE key = ?",
                    (key,)
                )
                row = cursor.fetchone()

                if row:
                    data_file, compressed, access_count = row
                    data_path = self.data_dir / data_file

                    # Check if file exists
                    if data_path.exists():
                        # Load and decompress data
                        with open(data_path, 'rb') as f:
                            data = f.read()

                        value = CompressionManager.decompress_data(data, compressed)

                        # Update access statistics
                        db.execute(
                            "UPDATE cache_entries SET last_accessed = ?, access_count = ? WHERE key = ?",
                            (time.time(), access_count + 1, key)
                        )
                        db.commit()

                        return value, True  # value, hit
                return None, False  # no value, miss

        value, hit = await loop.run_in_executor(None, db_operation)

        if hit:
            self.stats.hits += 1
            response_time = (time.perf_counter() - start_time) * 1000
            self._update_avg_response_time(response_time)
            return value
        else:
            self.stats.misses += 1
            return None

    async def _file_exists(self, path: Path) -> bool:
        """Asynchronously check if file exists"""
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, path.exists)
        except:
            return False

    async def _read_file_async(self, path: Path) -> bytes:
        """Asynchronously read file contents"""
        async with aiofiles.open(path, 'rb') as f:
            return await f.read()

    async def _write_file_async(self, path: Path, data: bytes):
        """Asynchronously write file contents"""
        async with aiofiles.open(path, 'wb') as f:
            await f.write(data)

    async def _unlink_file_async(self, path: Path):
        """Asynchronously remove file"""
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, path.unlink)
        except FileNotFoundError:
            pass  # File already removed

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
        await self._cleanup_if_needed_async(size_bytes)

        # Save data to file asynchronously
        data_file = f"{key}.cache"
        data_path = self.data_dir / data_file

        await self._write_file_async(data_path, compressed_data)

        # Store metadata in database
        tags_str = json.dumps(list(tags)) if tags else "[]"
        metadata_str = json.dumps(metadata) if metadata else "{}"

        # Run database operation in thread pool
        loop = asyncio.get_event_loop()
        
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

        await loop.run_in_executor(None, db_operation)

        self.stats.total_entries += 1
        return True

    async def _cleanup_if_needed_async(self, new_entry_size: int):
        """Clean up old entries if cache is full - async version"""
        current_size = await self._get_current_size_async()

        if current_size + new_entry_size > self.max_size_bytes:
            # Remove oldest entries until we have space
            loop = asyncio.get_event_loop()
            
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
                        if data_path.exists():
                            data_path.unlink()

                        # Remove database entry
                        db.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
                        db.commit()

                        current_size -= size_bytes

                        if current_size + new_entry_size <= self.max_size_bytes:
                            break

            await loop.run_in_executor(None, db_operation)

    async def _get_current_size_async(self) -> int:
        """Get current cache size in bytes - async version"""
        loop = asyncio.get_event_loop()
        
        def db_operation():
            with sqlite3.connect(self.db_path) as db:
                cursor = db.execute("SELECT SUM(size_bytes) FROM cache_entries")
                result = cursor.fetchone()
                return result[0] if result[0] else 0

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
        """Clear entries matching any of the given tags - async version"""
        loop = asyncio.get_event_loop()
        
        def db_operation():
            with sqlite3.connect(self.db_path) as db:
                cursor = db.execute("SELECT key, data_file, tags FROM cache_entries")
                rows = cursor.fetchall()
                keys_to_remove = []

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

        await loop.run_in_executor(None, db_operation)

    def get_stats(self) -> CacheStats:
        """Get current disk cache statistics"""
        # Since this is called from sync context, we'll return cached stats
        # For full async stats, use get_stats_async
        return self.stats

    async def get_stats_async(self) -> CacheStats:
        """Get current disk cache statistics - async version"""
        self.stats.disk_usage_mb = await self._get_current_size_async() / (1024 * 1024)

        loop = asyncio.get_event_loop()
        
        def db_operation():
            with sqlite3.connect(self.db_path) as db:
                cursor = db.execute("SELECT COUNT(*) FROM cache_entries")
                count = cursor.fetchone()
                return count[0] if count else 0

        self.stats.total_entries = await loop.run_in_executor(None, db_operation)
        return self.stats

    async def get_entry(self, key: str) -> Optional[AsyncCacheEntry]:
        """Get cache entry with metadata - async version"""
        loop = asyncio.get_event_loop()
        
        def db_operation():
            with sqlite3.connect(self.db_path) as db:
                cursor = db.execute("""
                    SELECT created_at, last_accessed, access_count, size_bytes,
                           compressed, tags, metadata
                    FROM cache_entries WHERE key = ?
                """, (key,))
                row = cursor.fetchone()

                if row:
                    created_at, last_accessed, access_count, size_bytes, compressed, tags_str, metadata_str = row

                    # Get the actual value
                    value = self._get_sync(key)  # Use sync method to avoid infinite recursion
                    if value is not None:
                        return AsyncCacheEntry(
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

        return await loop.run_in_executor(None, db_operation)

    def _get_sync(self, key: str) -> Optional[Any]:
        """Synchronous get for internal use"""
        with sqlite3.connect(self.db_path) as db:
            cursor = db.execute(
                "SELECT data_file, compressed FROM cache_entries WHERE key = ?",
                (key,)
            )
            row = cursor.fetchone()

            if row:
                data_file, compressed = row
                data_path = self.data_dir / data_file

                if data_path.exists():
                    with open(data_path, 'rb') as f:
                        data = f.read()
                    return CompressionManager.decompress_data(data, compressed)

        return None

    async def delete(self, key: str) -> bool:
        """Delete a specific cache entry"""
        loop = asyncio.get_event_loop()
        
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

        return await loop.run_in_executor(None, db_operation)

    async def delete_pattern(self, pattern: str) -> int:
        """Delete cache entries matching pattern - async version"""
        loop = asyncio.get_event_loop()
        
        def db_operation():
            deleted = 0
            with sqlite3.connect(self.db_path) as db:
                cursor = db.execute("""
                    SELECT key, data_file FROM cache_entries
                    WHERE key LIKE ?
                """, (f"%{pattern}%",))
                
                rows = cursor.fetchall()
                for row in rows:
                    key, data_file = row
                    data_path = self.data_dir / data_file
                    
                    if data_path.exists():
                        data_path.unlink()

                    db.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
                    db.commit()
                    deleted += 1

            return deleted

        return await loop.run_in_executor(None, db_operation)