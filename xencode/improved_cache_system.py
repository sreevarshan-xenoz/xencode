#!/usr/bin/env python3
"""
Compatibility shim for the legacy improved cache system.

The canonical implementation now lives in advanced_cache_system.
This module re-exports the shared APIs to avoid duplicate behavior.
"""

from .advanced_cache_system import (
    CACHE_EXPIRY_SECONDS,
    COMPRESSION_THRESHOLD_BYTES,
    DEFAULT_DISK_CACHE_MB,
    DEFAULT_MEMORY_CACHE_MB,
    CacheEntry,
    CacheKeyGenerator,
    CacheStats,
    CompressionManager,
    DiskCache,
    HybridCacheManager,
    MemoryCache,
    cached_response,
    get_cache_manager,
)


class ImprovedHybridCacheManager(HybridCacheManager):
    """Backward-compatible alias for the canonical hybrid cache manager."""


__all__ = [
    "CacheEntry",
    "CacheStats",
    "CacheKeyGenerator",
    "CompressionManager",
    "MemoryCache",
    "DiskCache",
    "HybridCacheManager",
    "ImprovedHybridCacheManager",
    "DEFAULT_MEMORY_CACHE_MB",
    "DEFAULT_DISK_CACHE_MB",
    "COMPRESSION_THRESHOLD_BYTES",
    "CACHE_EXPIRY_SECONDS",
    "get_cache_manager",
    "cached_response",
]
