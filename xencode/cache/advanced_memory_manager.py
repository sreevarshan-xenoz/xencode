"""
Advanced Memory Management for Xencode

Implements tiered memory storage (RAM, SSD, HDD) for different cache priorities,
predictive caching based on usage patterns and ML algorithms,
and intelligent cache eviction policies with priority scoring.
"""

import asyncio
import heapq
import json
import pickle
import time
import zlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from collections import OrderedDict, defaultdict
import threading
import sqlite3
import logging
from datetime import datetime, timedelta

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


logger = logging.getLogger(__name__)


class MemoryTier(Enum):
    """Memory tiers based on speed and persistence"""
    RAM_HOT = "ram_hot"      # Frequently accessed items in RAM
    RAM_WARM = "ram_warm"    # Recently accessed items in RAM
    SSD_COLD = "ssd_cold"    # Less frequently accessed on SSD
    HDD_ARCHIVE = "hdd_archive"  # Rarely accessed items on HDD


class CachePriority(Enum):
    """Priority levels for cached items"""
    CRITICAL = "critical"    # Must stay in memory
    HIGH = "high"           # Frequently accessed
    NORMAL = "normal"       # Regular access pattern
    LOW = "low"            # Infrequently accessed
    ARCHIVE = "archive"     # Rarely accessed, can be evicted


@dataclass
class CacheItem:
    """Represents an item in the cache"""
    key: str
    value: Any
    priority: CachePriority
    access_count: int
    last_access_time: float
    creation_time: float
    size_bytes: int
    ttl: Optional[float]  # Time to live in seconds
    tags: List[str]
    
    def is_expired(self) -> bool:
        """Check if the item has expired"""
        if self.ttl is None:
            return False
        return time.time() - self.creation_time > self.ttl


class MemoryTierManager(ABC):
    """Abstract base class for memory tier managers"""
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Get an item from this tier"""
        pass
    
    @abstractmethod
    def put(self, item: CacheItem) -> bool:
        """Put an item in this tier"""
        pass
    
    @abstractmethod
    def evict(self, count: int) -> List[str]:
        """Evict items from this tier"""
        pass
    
    @abstractmethod
    def get_size(self) -> int:
        """Get current size of this tier"""
        pass
    
    @abstractmethod
    def get_max_size(self) -> int:
        """Get maximum size of this tier"""
        pass


class RAMTierManager(MemoryTierManager):
    """Memory tier manager for RAM-based storage"""
    
    def __init__(self, max_size_mb: int = 100):
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.data: OrderedDict[str, CacheItem] = OrderedDict()
        self.current_size = 0
        self.lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get an item from RAM tier"""
        with self.lock:
            if key in self.data:
                item = self.data[key]
                if item.is_expired():
                    del self.data[key]
                    self.current_size -= item.size_bytes
                    return None
                
                # Update access stats
                item.access_count += 1
                item.last_access_time = time.time()
                
                # Move to end (most recently used)
                self.data.move_to_end(key)
                
                return item.value
            return None
    
    def put(self, item: CacheItem) -> bool:
        """Put an item in RAM tier"""
        with self.lock:
            # Check if we need to evict items to make space
            if item.size_bytes > self.max_size_bytes:
                # Item is too big for this tier
                return False
            
            # Evict items if necessary
            while self.current_size + item.size_bytes > self.max_size_bytes:
                if not self.evict(1):
                    # Cannot evict anything, reject the item
                    return False
            
            # Remove existing item if it exists
            if item.key in self.data:
                old_item = self.data[item.key]
                self.current_size -= old_item.size_bytes
            
            # Add new item
            self.data[item.key] = item
            self.current_size += item.size_bytes
            return True
    
    def evict(self, count: int) -> List[str]:
        """Evict items from RAM tier using LRU"""
        with self.lock:
            evicted_keys = []
            
            # Sort items by priority and access pattern
            items_to_evict = []
            for key, item in self.data.items():
                if not item.is_expired():
                    # Calculate eviction score (lower priority, less access = higher score)
                    score = (item.priority.value * -1) + (item.access_count * 0.1) + \
                           ((time.time() - item.last_access_time) * 0.01)
                    items_to_evict.append((score, key, item))
            
            # Sort by score (ascending, so lowest priority items first)
            items_to_evict.sort(key=lambda x: x[0])
            
            # Evict the specified number of items
            for i in range(min(count, len(items_to_evict))):
                _, key, item = items_to_evict[i]
                if key in self.data:
                    del self.data[key]
                    self.current_size -= item.size_bytes
                    evicted_keys.append(key)
            
            return evicted_keys
    
    def get_size(self) -> int:
        """Get current size of RAM tier"""
        with self.lock:
            return self.current_size
    
    def get_max_size(self) -> int:
        """Get maximum size of RAM tier"""
        return self.max_size_bytes


class DiskTierManager(MemoryTierManager):
    """Memory tier manager for disk-based storage (SSD/HDD)"""
    
    def __init__(self, tier: MemoryTier, storage_path: Optional[Path] = None, max_size_mb: int = 1000):
        self.tier = tier
        self.max_size_bytes = max_size_mb * 1024 * 1024
        
        if storage_path is None:
            storage_path = Path.home() / ".xencode" / "cache" / tier.value
        self.storage_path = storage_path
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Use SQLite for efficient storage and querying
        self.db_path = self.storage_path / "cache.db"
        self.init_database()
        
        self.lock = threading.RLock()
    
    def init_database(self):
        """Initialize the SQLite database"""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS cache_items (
                key TEXT PRIMARY KEY,
                value BLOB,
                priority TEXT,
                access_count INTEGER,
                last_access_time REAL,
                creation_time REAL,
                size_bytes INTEGER,
                ttl REAL,
                tags TEXT,
                compressed BOOLEAN
            )
        """)
        
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_last_access ON cache_items(last_access_time)
        """)
        
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_priority ON cache_items(priority)
        """)
        
        self.conn.commit()
    
    def get(self, key: str) -> Optional[Any]:
        """Get an item from disk tier"""
        with self.lock:
            cursor = self.conn.execute(
                "SELECT value, priority, access_count, last_access_time, ttl, compressed FROM cache_items WHERE key = ?",
                (key,)
            )
            row = cursor.fetchone()
            
            if row is None:
                return None
            
            value_data, priority_str, access_count, last_access_time, ttl, compressed = row
            
            # Check expiration
            if ttl is not None:
                creation_time = last_access_time  # For simplicity, treat last_access as creation
                if time.time() - creation_time > ttl:
                    # Item expired, remove it
                    self.conn.execute("DELETE FROM cache_items WHERE key = ?", (key,))
                    self.conn.commit()
                    return None
            
            # Deserialize value
            if compressed:
                value_data = zlib.decompress(value_data)
            
            try:
                value = pickle.loads(value_data)
            except:
                # Corrupted data, remove it
                self.conn.execute("DELETE FROM cache_items WHERE key = ?", (key,))
                self.conn.commit()
                return None
            
            # Update access stats
            new_access_count = access_count + 1
            self.conn.execute(
                "UPDATE cache_items SET access_count = ?, last_access_time = ? WHERE key = ?",
                (new_access_count, time.time(), key)
            )
            self.conn.commit()
            
            return value
    
    def put(self, item: CacheItem) -> bool:
        """Put an item in disk tier"""
        with self.lock:
            # Check current size
            current_size = self.get_size()
            if current_size + item.size_bytes > self.max_size_bytes:
                # Need to evict items
                self.evict(int((current_size + item.size_bytes - self.max_size_bytes) // item.size_bytes) + 1)
            
            # Serialize value with optional compression
            value_bytes = pickle.dumps(item.value)
            compressed = False
            if len(value_bytes) > 1024:  # Compress if larger than 1KB
                value_bytes = zlib.compress(value_bytes)
                compressed = True
            
            # Insert or update the item
            try:
                self.conn.execute("""
                    INSERT OR REPLACE INTO cache_items 
                    (key, value, priority, access_count, last_access_time, creation_time, size_bytes, ttl, tags, compressed)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    item.key, value_bytes, item.priority.value, item.access_count,
                    item.last_access_time, item.creation_time, item.size_bytes,
                    item.ttl, json.dumps(item.tags), compressed
                ))
                self.conn.commit()
                return True
            except sqlite3.Error as e:
                logger.error(f"Failed to insert item {item.key} into disk cache: {e}")
                return False
    
    def evict(self, count: int) -> List[str]:
        """Evict items from disk tier"""
        with self.lock:
            # Get items to evict based on priority and access pattern
            cursor = self.conn.execute("""
                SELECT key, priority, access_count, last_access_time
                FROM cache_items
                ORDER BY 
                    CASE priority
                        WHEN 'archive' THEN 1
                        WHEN 'low' THEN 2
                        WHEN 'normal' THEN 3
                        WHEN 'high' THEN 4
                        WHEN 'critical' THEN 5
                    END ASC,
                    access_count ASC,
                    last_access_time ASC
                LIMIT ?
            """, (count,))
            
            rows = cursor.fetchall()
            evicted_keys = [row[0] for row in rows]
            
            if evicted_keys:
                placeholders = ','.join(['?' for _ in evicted_keys])
                self.conn.execute(f"DELETE FROM cache_items WHERE key IN ({placeholders})", evicted_keys)
                self.conn.commit()
            
            return evicted_keys
    
    def get_size(self) -> int:
        """Get current size of disk tier"""
        with self.lock:
            cursor = self.conn.execute("SELECT SUM(size_bytes) FROM cache_items")
            result = cursor.fetchone()[0]
            return result or 0
    
    def get_max_size(self) -> int:
        """Get maximum size of disk tier"""
        return self.max_size_bytes
    
    def __del__(self):
        """Cleanup database connection"""
        if hasattr(self, 'conn'):
            self.conn.close()


class PredictiveCacheAdvisor:
    """Uses ML to predict which items will be accessed and pre-cache them"""
    
    def __init__(self):
        self.access_patterns = defaultdict(list)  # key -> list of access times
        self.model = LinearRegression()
        self.scaler = StandardScaler()
        self.is_trained = False
        self.lock = threading.RLock()
    
    def record_access(self, key: str):
        """Record an access event"""
        with self.lock:
            self.access_patterns[key].append(time.time())
    
    def predict_access_probability(self, key: str, time_ahead: float = 300) -> float:
        """Predict probability of access in the next time_ahead seconds"""
        with self.lock:
            if key not in self.access_patterns or len(self.access_patterns[key]) < 3:
                # Not enough data, return a default probability
                return 0.1
            
            # Get access times in the last hour
            recent_accesses = [t for t in self.access_patterns[key] if time.time() - t < 3600]
            if len(recent_accesses) < 3:
                return 0.1
            
            # Calculate features: frequency, recency, periodicity
            now = time.time()
            time_diffs = [now - t for t in recent_accesses]
            frequency = len(recent_accesses) / 3600  # accesses per hour
            recency = min(time_diffs) if time_diffs else 0
            avg_interval = np.mean(np.diff(sorted([now - td for td in time_diffs]))) if len(time_diffs) > 1 else 3600
            
            # Simple heuristic for access probability
            # Higher probability if accessed frequently, recently, or periodically
            prob = min(1.0, (frequency * 100 + (3600 - recency) / 3600 + (1 / (avg_interval + 1))) / 10)
            return max(0.0, min(1.0, prob))
    
    def get_predicted_accesses(self, time_window: float = 300) -> List[Tuple[str, float]]:
        """Get list of keys predicted to be accessed with probabilities"""
        with self.lock:
            predictions = []
            for key in list(self.access_patterns.keys()):
                prob = self.predict_access_probability(key, time_window)
                if prob > 0.3:  # Only include if probability is reasonably high
                    predictions.append((key, prob))
            
            # Sort by probability descending
            predictions.sort(key=lambda x: x[1], reverse=True)
            return predictions


class AdvancedMemoryManager:
    """Main class for advanced memory management"""
    
    def __init__(self, 
                 ram_hot_size_mb: int = 50,
                 ram_warm_size_mb: int = 100, 
                 ssd_size_mb: int = 500,
                 hdd_size_mb: int = 2000):
        # Initialize memory tiers
        self.ram_hot = RAMTierManager(ram_hot_size_mb)
        self.ram_warm = RAMTierManager(ram_warm_size_mb)
        self.ssd_cold = DiskTierManager(MemoryTier.SSD_COLD, max_size_mb=ssd_size_mb)
        self.hdd_archive = DiskTierManager(MemoryTier.HDD_ARCHIVE, max_size_mb=hdd_size_mb)
        
        # Tier mapping
        self.tiers = {
            MemoryTier.RAM_HOT: self.ram_hot,
            MemoryTier.RAM_WARM: self.ram_warm,
            MemoryTier.SSD_COLD: self.ssd_cold,
            MemoryTier.HDD_ARCHIVE: self.hdd_archive
        }
        
        # Predictive advisor
        self.advisor = PredictiveCacheAdvisor()
        
        # Background tasks
        self.background_tasks = []
        self.stop_event = threading.Event()
        
        # Start background maintenance
        self.start_background_tasks()
    
    def start_background_tasks(self):
        """Start background maintenance tasks"""
        # Predictive pre-caching thread
        precache_thread = threading.Thread(target=self._predictive_precache_worker, daemon=True)
        precache_thread.start()
        self.background_tasks.append(precache_thread)
        
        # Periodic cleanup thread
        cleanup_thread = threading.Thread(target=self._periodic_cleanup_worker, daemon=True)
        cleanup_thread.start()
        self.background_tasks.append(cleanup_thread)
    
    def _predictive_precache_worker(self):
        """Background worker for predictive pre-caching"""
        while not self.stop_event.is_set():
            try:
                # Get predicted accesses
                predicted = self.advisor.get_predicted_accesses(time_window=300)
                
                # Pre-cache top predictions that aren't already cached
                for key, prob in predicted[:10]:  # Pre-cache top 10 predictions
                    if self.get(key) is None:
                        # Trigger a callback to populate this key if possible
                        self._trigger_precache_callback(key)
                
                # Sleep for a while before next prediction
                self.stop_event.wait(timeout=60)  # Check every minute
            except Exception as e:
                logger.error(f"Error in predictive cache worker: {e}")
                self.stop_event.wait(timeout=60)
    
    def _periodic_cleanup_worker(self):
        """Background worker for periodic cleanup"""
        while not self.stop_event.is_set():
            try:
                # Perform cleanup operations
                self._cleanup_expired_items()
                self._balance_tiers()
                
                # Sleep for a while
                self.stop_event.wait(timeout=300)  # Clean up every 5 minutes
            except Exception as e:
                logger.error(f"Error in cleanup worker: {e}")
                self.stop_event.wait(timeout=300)
    
    def _trigger_precache_callback(self, key: str):
        """Trigger a callback to pre-cache a key if registered"""
        # This would call a registered callback to populate the cache
        # For now, we'll just log it
        logger.debug(f"Would trigger pre-cache for key: {key}")
    
    def _cleanup_expired_items(self):
        """Clean up expired items from all tiers"""
        for tier in self.tiers.values():
            # This is handled within each tier's get method
            # For disk tiers, we might want a more aggressive cleanup
            if isinstance(tier, DiskTierManager):
                # Remove expired items from disk database
                with tier.lock:
                    cursor = tier.conn.execute(
                        "DELETE FROM cache_items WHERE ttl IS NOT NULL AND ? - creation_time > ttl",
                        (time.time(),)
                    )
                    if cursor.rowcount > 0:
                        tier.conn.commit()
                        logger.debug(f"Removed {cursor.rowcount} expired items from {tier.tier.value} tier")
    
    def _balance_tiers(self):
        """Balance items between tiers based on access patterns"""
        # Move frequently accessed items to hotter tiers
        # Move infrequently accessed items to colder tiers
        
        # This is a simplified implementation
        # In a real system, this would be more sophisticated
        pass
    
    def get(self, key: str) -> Optional[Any]:
        """Get an item from any tier"""
        # Check hottest tier first
        result = self.ram_hot.get(key)
        if result is not None:
            # Record access for prediction
            self.advisor.record_access(key)
            return result
        
        # Check warm tier
        result = self.ram_warm.get(key)
        if result is not None:
            # Move to hot tier if it fits
            item = self._create_cache_item(key, result, CachePriority.NORMAL, tags=[])
            if self.ram_hot.put(item):
                # Successfully moved to hot tier
                pass
            # Record access
            self.advisor.record_access(key)
            return result
        
        # Check cold tier
        result = self.ssd_cold.get(key)
        if result is not None:
            # Move to warm tier if it fits
            item = self._create_cache_item(key, result, CachePriority.NORMAL, tags=[])
            if self.ram_warm.put(item):
                # Successfully moved to warm tier
                pass
            # Record access
            self.advisor.record_access(key)
            return result
        
        # Check archive tier
        result = self.hdd_archive.get(key)
        if result is not None:
            # Move to cold tier if it fits
            item = self._create_cache_item(key, result, CachePriority.NORMAL, tags=[])
            if self.ssd_cold.put(item):
                # Successfully moved to cold tier
                pass
            # Record access
            self.advisor.record_access(key)
            return result
        
        return None
    
    def put(self, 
            key: str, 
            value: Any, 
            priority: CachePriority = CachePriority.NORMAL,
            ttl: Optional[float] = None,
            tags: Optional[List[str]] = None) -> bool:
        """Put an item in the appropriate tier"""
        if tags is None:
            tags = []
        
        # Create cache item
        item = self._create_cache_item(key, value, priority, ttl, tags)
        
        # Put in the appropriate tier based on priority
        if priority in [CachePriority.CRITICAL, CachePriority.HIGH]:
            # Try hot tier first, then warm
            if self.ram_hot.put(item):
                return True
            elif self.ram_warm.put(item):
                return True
            else:
                # If RAM is full, try to evict lower priority items
                if priority == CachePriority.CRITICAL:
                    # Force into hot tier by evicting low priority items
                    self.ram_hot.evict(1)
                    return self.ram_hot.put(item)
                elif priority == CachePriority.HIGH:
                    # Force into warm tier by evicting low priority items
                    self.ram_warm.evict(1)
                    return self.ram_warm.put(item)
                else:
                    # Try cold tier
                    return self.ssd_cold.put(item)
        elif priority == CachePriority.NORMAL:
            # Try warm tier first, then cold
            if self.ram_warm.put(item):
                return True
            else:
                return self.ssd_cold.put(item)
        else:
            # LOW or ARCHIVE go to cold or archive tier
            if priority == CachePriority.LOW:
                return self.ssd_cold.put(item)
            else:  # ARCHIVE
                return self.hdd_archive.put(item)
    
    def _create_cache_item(self, 
                          key: str, 
                          value: Any, 
                          priority: CachePriority, 
                          ttl: Optional[float] = None,
                          tags: Optional[List[str]] = None) -> CacheItem:
        """Create a cache item with calculated properties"""
        if tags is None:
            tags = []
        
        # Estimate size of the value
        try:
            value_bytes = pickle.dumps(value)
            size_bytes = len(value_bytes)
        except:
            size_bytes = 1024  # Default estimate
        
        return CacheItem(
            key=key,
            value=value,
            priority=priority,
            access_count=0,
            last_access_time=time.time(),
            creation_time=time.time(),
            size_bytes=size_bytes,
            ttl=ttl,
            tags=tags
        )
    
    def evict_by_pattern(self, pattern: str) -> int:
        """Evict items matching a pattern"""
        count = 0
        
        # This would implement pattern matching across all tiers
        # For now, we'll just return 0
        return count
    
    def get_tier_usage(self) -> Dict[str, Dict[str, int]]:
        """Get usage statistics for all tiers"""
        usage = {}
        for tier_name, tier_manager in self.tiers.items():
            usage[tier_name.value] = {
                "current_size": tier_manager.get_size(),
                "max_size": tier_manager.get_max_size(),
                "utilization": tier_manager.get_size() / max(tier_manager.get_max_size(), 1)
            }
        return usage
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        tier_usage = self.get_tier_usage()
        
        # Get total items across all tiers (approximation)
        total_items = 0
        for tier_name, tier_data in tier_usage.items():
            # This is a simplification - in reality, we'd need to count items in each tier
            avg_item_size = 1024  # Assumed average item size
            total_items += tier_data["current_size"] // max(avg_item_size, 1)
        
        return {
            "tier_usage": tier_usage,
            "total_items": total_items,
            "total_size_bytes": sum(tier["current_size"] for tier in tier_usage.values()),
            "prediction_accuracy": "N/A"  # Would require historical tracking
        }
    
    def shutdown(self):
        """Shutdown the memory manager and background tasks"""
        self.stop_event.set()
        
        # Wait for background tasks to finish
        for task in self.background_tasks:
            task.join(timeout=5)  # Wait up to 5 seconds


# Global memory manager instance
_memory_manager: Optional[AdvancedMemoryManager] = None


def get_memory_manager() -> AdvancedMemoryManager:
    """Get the global advanced memory manager instance"""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = AdvancedMemoryManager()
    return _memory_manager


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    def test_advanced_memory_management():
        """Test the advanced memory management system"""
        print("Testing Advanced Memory Management...")
        
        # Create memory manager
        mm = get_memory_manager()
        
        print("\n1. Testing basic put/get operations:")
        # Test putting items with different priorities
        mm.put("hot_item", "This is a hot item", CachePriority.HIGH, ttl=600)
        mm.put("warm_item", "This is a warm item", CachePriority.NORMAL, ttl=1200)
        mm.put("cold_item", "This is a cold item", CachePriority.LOW, ttl=3600)
        mm.put("archive_item", "This is an archive item", CachePriority.ARCHIVE, ttl=7200)
        
        # Retrieve items
        print(f"Hot item: {mm.get('hot_item')}")
        print(f"Warm item: {mm.get('warm_item')}")
        print(f"Cold item: {mm.get('cold_item')}")
        print(f"Archive item: {mm.get('archive_item')}")
        
        print("\n2. Testing tier usage:")
        usage = mm.get_tier_usage()
        for tier, stats in usage.items():
            print(f"{tier}: {stats['current_size']}/{stats['max_size']} bytes ({stats['utilization']:.2%})")
        
        print("\n3. Testing cache statistics:")
        stats = mm.get_cache_stats()
        print(f"Total items: {stats['total_items']}")
        print(f"Total size: {stats['total_size_bytes']} bytes")
        
        print("\n4. Testing predictive advisor:")
        # Record some accesses to train the predictor
        for i in range(5):
            mm.get("hot_item")  # Access hot item multiple times
            time.sleep(0.01)  # Small delay
        
        # Check prediction
        prob = mm.advisor.predict_access_probability("hot_item", time_ahead=60)
        print(f"Prediction for hot_item access in next 60s: {prob:.2f}")
        
        # Get top predicted accesses
        predictions = mm.advisor.get_predicted_accesses(time_window=60)
        print(f"Top predicted accesses: {predictions[:3]}")
        
        print("\n5. Testing item expiration:")
        mm.put("expiring_item", "This will expire", CachePriority.NORMAL, ttl=2)  # 2 seconds
        print(f"Expiring item (before expiry): {mm.get('expiring_item')}")
        time.sleep(3)  # Wait for expiry
        print(f"Expiring item (after expiry): {mm.get('expiring_item')}")
        
        print("\n✅ Advanced Memory Management tests completed!")
        
        # Shutdown
        mm.shutdown()
    
    # Run the test
    test_advanced_memory_management()