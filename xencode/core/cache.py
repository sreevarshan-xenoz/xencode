"""
Response caching module for Xencode
"""
import hashlib
import json
import lzma
import pickle
import time
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Union

# Performance and caching configuration
CACHE_ENABLED = True
CACHE_DIR = Path.home() / ".xencode" / "cache"
MAX_CACHE_SIZE = 100  # Maximum cached responses
CACHE_TTL_SECONDS = 86400  # 24 hours TTL
COMPRESSION_ENABLED = True


class CacheStats(NamedTuple):
    hits: int
    misses: int
    evictions: int
    size: int


class CacheLevel(Enum):
    MEMORY = "memory"
    DISK = "disk"
    DISTRIBUTED = "distributed"  # Future extension


class LRUCacheItem:
    """Represents an item in the LRU cache with metadata"""
    def __init__(self, key: str, value: str, model: str, timestamp: float):
        self.key = key
        self.value = value
        self.model = model
        self.timestamp = timestamp
        self.access_count = 1
        self.last_access = timestamp


class InMemoryCache:
    """Simple in-memory LRU cache for frequently accessed items"""
    def __init__(self, capacity: int = 50):
        self.capacity = capacity
        self.cache: Dict[str, LRUCacheItem] = {}
        self.access_order: List[str] = []  # Keys ordered by access time
        self.stats = CacheStats(hits=0, misses=0, evictions=0, size=0)

    def get(self, key: str) -> Optional[LRUCacheItem]:
        """Get an item from the cache and update access order"""
        if key in self.cache:
            item = self.cache[key]
            # Update access count and time
            item.access_count += 1
            item.last_access = time.time()

            # Move to end (most recently used)
            if key in self.access_order:
                self.access_order.remove(key)
            self.access_order.append(key)

            self.stats = self.stats._replace(hits=self.stats.hits + 1)
            return item

        self.stats = self.stats._replace(misses=self.stats.misses + 1)
        return None

    def put(self, key: str, value: str, model: str) -> None:
        """Put an item in the cache, evicting if necessary"""
        current_time = time.time()

        if key in self.cache:
            # Update existing item
            self.cache[key].value = value
            self.cache[key].timestamp = current_time
            self.cache[key].access_count += 1
            self.cache[key].last_access = current_time
        else:
            # Add new item
            if len(self.cache) >= self.capacity:
                # Evict least recently used item
                if self.access_order:
                    lru_key = self.access_order.pop(0)
                    del self.cache[lru_key]
                    self.stats = self.stats._replace(evictions=self.stats.evictions + 1)

            self.cache[key] = LRUCacheItem(key, value, model, current_time)

        # Update access order
        if key in self.access_order:
            self.access_order.remove(key)
        self.access_order.append(key)

        self.stats = self.stats._replace(size=len(self.cache))

    def get_stats(self) -> CacheStats:
        """Get current cache statistics"""
        return self.stats

    def clear(self) -> None:
        """Clear the cache"""
        self.cache.clear()
        self.access_order.clear()
        self.stats = CacheStats(hits=0, misses=0, evictions=0, size=0)


class ResponseCache:
    """Sophisticated multi-level response caching for performance optimization"""

    def __init__(
        self,
        cache_dir: Path = CACHE_DIR,
        max_size: int = MAX_CACHE_SIZE,
        ttl_seconds: int = CACHE_TTL_SECONDS,
        compression_enabled: bool = COMPRESSION_ENABLED
    ) -> None:
        self.cache_dir: Path = Path(cache_dir)
        self.max_size: int = max_size
        self.ttl_seconds: int = ttl_seconds
        self.compression_enabled: bool = compression_enabled

        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize in-memory cache for hot items
        self.memory_cache = InMemoryCache(capacity=max(10, max_size // 4))

        # Statistics tracking
        self.hit_count = 0
        self.miss_count = 0
        self.eviction_count = 0

    def _get_cache_key(self, prompt: str, model: str) -> str:
        """Generate cache key from prompt and model"""
        content = f"{prompt}:{model}".encode('utf-8')
        return hashlib.md5(content).hexdigest()

    def _compress_data(self, data: str) -> bytes:
        """Compress data using LZMA if enabled"""
        if self.compression_enabled:
            return lzma.compress(data.encode('utf-8'))
        return data.encode('utf-8')

    def _decompress_data(self, compressed_data: bytes) -> str:
        """Decompress data using LZMA if enabled"""
        if self.compression_enabled:
            return lzma.decompress(compressed_data).decode('utf-8')
        return compressed_data.decode('utf-8')

    def get(self, prompt: str, model: str) -> Optional[str]:
        """Get cached response if available, checking memory first then disk"""
        if not CACHE_ENABLED:
            return None

        cache_key = self._get_cache_key(prompt, model)

        # First, check in-memory cache
        item = self.memory_cache.get(cache_key)
        if item is not None:
            # Verify TTL
            if time.time() - item.timestamp < self.ttl_seconds:
                return item.value
            else:
                # Item expired, remove from memory cache
                if cache_key in self.memory_cache.cache:
                    del self.memory_cache.cache[cache_key]
                if cache_key in self.memory_cache.access_order:
                    self.memory_cache.access_order.remove(cache_key)
                self.memory_cache.stats = self.memory_cache.stats._replace(
                    size=len(self.memory_cache.cache)
                )

        # Then check disk cache
        try:
            cache_file = self.cache_dir / f"{cache_key}.json.xz"  # Use .xz extension for compressed files

            if cache_file.exists():
                # Read and decompress the file
                with open(cache_file, 'rb') as f:
                    compressed_data = f.read()

                # Decompress and parse
                data_str = self._decompress_data(compressed_data)
                data = json.loads(data_str)

                # Check if cache is still valid (TTL)
                if time.time() - data['timestamp'] < self.ttl_seconds:
                    # Add to memory cache for faster access next time
                    self.memory_cache.put(cache_key, data['response'], model)
                    self.hit_count += 1
                    return data['response']  # type: ignore
                else:
                    # Remove expired file
                    cache_file.unlink()
        except Exception:
            pass

        self.miss_count += 1
        return None

    def set(self, prompt: str, model: str, response: str) -> None:
        """Cache a response in both memory and disk"""
        if not CACHE_ENABLED:
            return

        cache_key = self._get_cache_key(prompt, model)

        # Add to memory cache
        self.memory_cache.put(cache_key, response, model)

        try:
            cache_file = self.cache_dir / f"{cache_key}.json.xz"  # Use .xz extension for compressed files

            data = {
                'prompt': prompt,
                'model': model,
                'response': response,
                'timestamp': time.time(),
                'created_at': datetime.now().isoformat(),
            }

            # Serialize and compress the data
            data_str = json.dumps(data)
            compressed_data = self._compress_data(data_str)

            # Write compressed data to file
            with open(cache_file, 'wb') as f:
                f.write(compressed_data)

            # Clean up old cache files if exceeding limit
            self._cleanup_cache()
        except Exception:
            pass

    def _cleanup_cache(self) -> None:
        """Remove old cache files to maintain size limit"""
        try:
            cache_files = list(self.cache_dir.glob("*.json.xz"))
            if len(cache_files) > self.max_size:
                # Sort by modification time and remove oldest
                cache_files.sort(key=lambda x: x.stat().st_mtime)
                for old_file in cache_files[: -self.max_size]:
                    old_file.unlink()
                    self.eviction_count += 1
        except Exception:
            pass

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        disk_files = list(self.cache_dir.glob("*.json.xz"))
        disk_usage = sum(f.stat().st_size for f in disk_files)

        memory_stats = self.memory_cache.get_stats()

        total_requests = self.hit_count + self.miss_count
        hit_rate = (self.hit_count / total_requests * 100) if total_requests > 0 else 0

        return {
            'memory': {
                'hits': memory_stats.hits,
                'misses': memory_stats.misses,
                'evictions': memory_stats.evictions,
                'size': memory_stats.size,
                'capacity': self.memory_cache.capacity,
            },
            'disk': {
                'total_files': len(disk_files),
                'disk_usage_bytes': disk_usage,
                'disk_usage_mb': round(disk_usage / (1024 * 1024), 2),
            },
            'overall': {
                'total_hits': self.hit_count,
                'total_misses': self.miss_count,
                'hit_rate_percent': round(hit_rate, 2),
                'total_evictions': self.eviction_count,
            },
            'configuration': {
                'enabled': CACHE_ENABLED,
                'max_size': self.max_size,
                'ttl_seconds': self.ttl_seconds,
                'compression_enabled': self.compression_enabled,
            }
        }

    def clear_cache(self) -> None:
        """Clear both memory and disk caches"""
        self.memory_cache.clear()

        # Remove all cache files
        cache_files = list(self.cache_dir.glob("*.json.xz"))
        for cache_file in cache_files:
            cache_file.unlink()

        # Reset counters
        self.hit_count = 0
        self.miss_count = 0
        self.eviction_count = 0