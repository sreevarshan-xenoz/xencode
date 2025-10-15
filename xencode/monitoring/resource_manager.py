#!/usr/bin/env python3
"""
Resource Management System

Comprehensive resource management with memory monitoring, cleanup automation,
garbage collection optimization, and resource pooling for expensive operations.
"""

import asyncio
import gc
import logging
import os
import threading
import time
import weakref
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from enum import Enum
import tracemalloc

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

logger = logging.getLogger(__name__)


class ResourceType(str, Enum):
    """Types of managed resources"""
    MEMORY = "memory"
    FILE_HANDLES = "file_handles"
    NETWORK_CONNECTIONS = "network_connections"
    THREAD_POOL = "thread_pool"
    PROCESS_POOL = "process_pool"
    CACHE_ENTRIES = "cache_entries"
    TEMPORARY_FILES = "temporary_files"


class CleanupPriority(str, Enum):
    """Priority levels for cleanup operations"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ResourceLimit:
    """Resource usage limits and thresholds"""
    resource_type: ResourceType
    soft_limit: float  # Warning threshold
    hard_limit: float  # Critical threshold
    unit: str = "bytes"  # bytes, count, percentage
    enabled: bool = True


@dataclass
class ResourceUsage:
    """Current resource usage information"""
    resource_type: ResourceType
    current_usage: float
    peak_usage: float
    limit: Optional[ResourceLimit]
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CleanupTask:
    """Represents a cleanup task"""
    task_id: str
    resource_type: ResourceType
    priority: CleanupPriority
    cleanup_function: Callable[[], bool]
    description: str
    estimated_savings: float = 0.0
    last_executed: Optional[datetime] = None
    execution_count: int = 0
    success_count: int = 0


class MemoryTracker:
    """Advanced memory tracking and analysis"""
    
    def __init__(self):
        self.tracking_enabled = False
        self.snapshots: List[Tuple[datetime, Any]] = []
        self.memory_history: deque = deque(maxlen=1000)
        self.allocation_patterns: Dict[str, List[float]] = defaultdict(list)
        
    def start_tracking(self):
        """Start memory tracking"""
        if not self.tracking_enabled:
            tracemalloc.start()
            self.tracking_enabled = True
            logger.info("Memory tracking started")
    
    def stop_tracking(self):
        """Stop memory tracking"""
        if self.tracking_enabled:
            tracemalloc.stop()
            self.tracking_enabled = False
            logger.info("Memory tracking stopped")
    
    def take_snapshot(self, label: str = None) -> Optional[Any]:
        """Take a memory snapshot"""
        if not self.tracking_enabled:
            return None
        
        snapshot = tracemalloc.take_snapshot()
        self.snapshots.append((datetime.now(), snapshot))
        
        # Keep only recent snapshots
        if len(self.snapshots) > 10:
            self.snapshots.pop(0)
        
        return snapshot
    
    def analyze_memory_growth(self) -> Dict[str, Any]:
        """Analyze memory growth patterns"""
        if len(self.snapshots) < 2:
            return {"error": "Insufficient snapshots for analysis"}
        
        current_snapshot = self.snapshots[-1][1]
        previous_snapshot = self.snapshots[-2][1]
        
        top_stats = current_snapshot.compare_to(previous_snapshot, 'lineno')
        
        analysis = {
            "total_growth_mb": sum(stat.size_diff for stat in top_stats) / (1024 * 1024),
            "top_growing_files": [],
            "memory_leaks_detected": False
        }
        
        # Analyze top growing allocations
        for stat in top_stats[:10]:
            if stat.size_diff > 0:
                analysis["top_growing_files"].append({
                    "file": str(stat.traceback.format()[0]) if stat.traceback else "unknown",
                    "growth_mb": stat.size_diff / (1024 * 1024),
                    "count_diff": stat.count_diff
                })
        
        # Simple leak detection (consistent growth over multiple snapshots)
        if len(self.snapshots) >= 5:
            recent_growth = [
                self.snapshots[i][1].compare_to(self.snapshots[i-1][1], 'lineno')
                for i in range(-4, 0)
            ]
            
            # Check for consistent growth patterns
            consistent_growers = defaultdict(int)
            for growth in recent_growth:
                for stat in growth[:5]:
                    if stat.size_diff > 1024 * 1024:  # > 1MB growth
                        key = str(stat.traceback.format()[0]) if stat.traceback else "unknown"
                        consistent_growers[key] += 1
            
            # Flag potential leaks (growing in 3+ consecutive snapshots)
            analysis["memory_leaks_detected"] = any(
                count >= 3 for count in consistent_growers.values()
            )
        
        return analysis
    
    def get_current_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics"""
        usage = {}
        
        if PSUTIL_AVAILABLE:
            process = psutil.Process()
            memory_info = process.memory_info()
            
            usage.update({
                "rss_mb": memory_info.rss / (1024 * 1024),
                "vms_mb": memory_info.vms / (1024 * 1024),
                "percent": process.memory_percent(),
                "available_mb": psutil.virtual_memory().available / (1024 * 1024)
            })
        
        # Add Python-specific memory info
        if self.tracking_enabled:
            current, peak = tracemalloc.get_traced_memory()
            usage.update({
                "traced_current_mb": current / (1024 * 1024),
                "traced_peak_mb": peak / (1024 * 1024)
            })
        
        return usage


class ResourcePool:
    """Generic resource pool for expensive operations"""
    
    def __init__(self, resource_factory: Callable, max_size: int = 10, 
                 cleanup_function: Optional[Callable] = None):
        self.resource_factory = resource_factory
        self.cleanup_function = cleanup_function
        self.max_size = max_size
        self.pool: deque = deque()
        self.in_use: Set[Any] = set()
        self.created_count = 0
        self.reuse_count = 0
        self._lock = threading.Lock()
    
    def acquire(self) -> Any:
        """Acquire a resource from the pool"""
        with self._lock:
            if self.pool:
                resource = self.pool.popleft()
                self.in_use.add(resource)
                self.reuse_count += 1
                return resource
            else:
                # Create new resource
                resource = self.resource_factory()
                self.in_use.add(resource)
                self.created_count += 1
                return resource
    
    def release(self, resource: Any):
        """Release a resource back to the pool"""
        with self._lock:
            if resource in self.in_use:
                self.in_use.remove(resource)
                
                if len(self.pool) < self.max_size:
                    self.pool.append(resource)
                else:
                    # Pool is full, cleanup resource
                    if self.cleanup_function:
                        try:
                            self.cleanup_function(resource)
                        except Exception as e:
                            logger.warning(f"Resource cleanup failed: {e}")
    
    def cleanup_all(self):
        """Cleanup all pooled resources"""
        with self._lock:
            if self.cleanup_function:
                for resource in self.pool:
                    try:
                        self.cleanup_function(resource)
                    except Exception as e:
                        logger.warning(f"Resource cleanup failed: {e}")
            
            self.pool.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics"""
        with self._lock:
            return {
                "pool_size": len(self.pool),
                "in_use": len(self.in_use),
                "created_count": self.created_count,
                "reuse_count": self.reuse_count,
                "reuse_ratio": self.reuse_count / max(self.created_count, 1)
            }


class GarbageCollectionManager:
    """Intelligent garbage collection management"""
    
    def __init__(self):
        self.gc_stats: List[Dict[str, Any]] = []
        self.auto_gc_enabled = True
        self.gc_thresholds = gc.get_threshold()
        self.last_gc_time = time.time()
        
    def optimize_gc_thresholds(self):
        """Optimize garbage collection thresholds based on usage patterns"""
        # Analyze recent GC performance
        if len(self.gc_stats) < 5:
            return
        
        recent_stats = self.gc_stats[-5:]
        avg_collection_time = sum(stat['duration'] for stat in recent_stats) / len(recent_stats)
        avg_objects_collected = sum(stat['collected'] for stat in recent_stats) / len(recent_stats)
        
        # Adjust thresholds based on performance
        current_thresholds = list(self.gc_thresholds)
        
        if avg_collection_time > 0.1:  # GC taking too long
            # Increase thresholds to reduce frequency
            current_thresholds = [int(t * 1.2) for t in current_thresholds]
        elif avg_collection_time < 0.01 and avg_objects_collected < 100:
            # GC running too frequently with little benefit
            current_thresholds = [int(t * 1.5) for t in current_thresholds]
        
        # Apply new thresholds
        gc.set_threshold(*current_thresholds)
        self.gc_thresholds = tuple(current_thresholds)
        
        logger.info(f"GC thresholds optimized: {self.gc_thresholds}")
    
    def force_garbage_collection(self) -> Dict[str, Any]:
        """Force garbage collection and return statistics"""
        start_time = time.time()
        
        # Collect statistics before GC
        before_counts = gc.get_count()
        before_objects = len(gc.get_objects())
        
        # Force collection for all generations
        collected = [gc.collect(generation) for generation in range(3)]
        
        # Collect statistics after GC
        after_counts = gc.get_count()
        after_objects = len(gc.get_objects())
        duration = time.time() - start_time
        
        stats = {
            "timestamp": datetime.now(),
            "duration": duration,
            "before_counts": before_counts,
            "after_counts": after_counts,
            "collected": sum(collected),
            "objects_before": before_objects,
            "objects_after": after_objects,
            "memory_freed": before_objects - after_objects
        }
        
        self.gc_stats.append(stats)
        self.last_gc_time = time.time()
        
        # Keep only recent stats
        if len(self.gc_stats) > 50:
            self.gc_stats.pop(0)
        
        return stats
    
    def should_force_gc(self) -> bool:
        """Determine if garbage collection should be forced"""
        # Force GC if it's been a while since last collection
        time_since_gc = time.time() - self.last_gc_time
        if time_since_gc > 300:  # 5 minutes
            return True
        
        # Force GC if memory usage is high
        if PSUTIL_AVAILABLE:
            memory_percent = psutil.virtual_memory().percent
            if memory_percent > 85:
                return True
        
        return False


class TemporaryFileManager:
    """Manages temporary files and cleanup"""
    
    def __init__(self, base_temp_dir: Optional[Path] = None):
        self.base_temp_dir = base_temp_dir or Path.cwd() / "temp"
        self.base_temp_dir.mkdir(exist_ok=True)
        self.tracked_files: Dict[str, Path] = {}
        self.file_metadata: Dict[str, Dict[str, Any]] = {}
        
    def create_temp_file(self, prefix: str = "xencode_", suffix: str = ".tmp",
                        auto_cleanup: bool = True) -> Tuple[str, Path]:
        """Create a temporary file and return ID and path"""
        import uuid
        
        file_id = str(uuid.uuid4())
        filename = f"{prefix}{file_id}{suffix}"
        file_path = self.base_temp_dir / filename
        
        # Create the file
        file_path.touch()
        
        # Track the file
        self.tracked_files[file_id] = file_path
        self.file_metadata[file_id] = {
            "created_at": datetime.now(),
            "auto_cleanup": auto_cleanup,
            "size_bytes": 0,
            "last_accessed": datetime.now()
        }
        
        return file_id, file_path
    
    def cleanup_temp_file(self, file_id: str) -> bool:
        """Cleanup a specific temporary file"""
        if file_id not in self.tracked_files:
            return False
        
        file_path = self.tracked_files[file_id]
        try:
            if file_path.exists():
                file_path.unlink()
            
            del self.tracked_files[file_id]
            del self.file_metadata[file_id]
            return True
        except Exception as e:
            logger.error(f"Failed to cleanup temp file {file_path}: {e}")
            return False
    
    def cleanup_old_files(self, max_age_hours: int = 24) -> int:
        """Cleanup temporary files older than specified age"""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        cleaned_count = 0
        
        for file_id, metadata in list(self.file_metadata.items()):
            if metadata["created_at"] < cutoff_time and metadata["auto_cleanup"]:
                if self.cleanup_temp_file(file_id):
                    cleaned_count += 1
        
        return cleaned_count
    
    def get_temp_usage(self) -> Dict[str, Any]:
        """Get temporary file usage statistics"""
        total_files = len(self.tracked_files)
        total_size = 0
        
        for file_id, file_path in self.tracked_files.items():
            try:
                if file_path.exists():
                    size = file_path.stat().st_size
                    total_size += size
                    self.file_metadata[file_id]["size_bytes"] = size
            except Exception:
                pass
        
        return {
            "total_files": total_files,
            "total_size_mb": total_size / (1024 * 1024),
            "base_directory": str(self.base_temp_dir)
        }


class ResourceManager:
    """Main resource management system"""
    
    def __init__(self):
        self.memory_tracker = MemoryTracker()
        self.gc_manager = GarbageCollectionManager()
        self.temp_file_manager = TemporaryFileManager()
        self.resource_pools: Dict[str, ResourcePool] = {}
        self.cleanup_tasks: Dict[str, CleanupTask] = {}
        
        # Resource limits
        self.resource_limits = {
            ResourceType.MEMORY: ResourceLimit(
                ResourceType.MEMORY, 
                soft_limit=75.0,  # 75% memory usage
                hard_limit=90.0,  # 90% memory usage
                unit="percentage"
            ),
            ResourceType.TEMPORARY_FILES: ResourceLimit(
                ResourceType.TEMPORARY_FILES,
                soft_limit=100,   # 100 MB
                hard_limit=500,   # 500 MB
                unit="megabytes"
            )
        }
        
        # Background tasks
        self.monitoring_task: Optional[asyncio.Task] = None
        self.cleanup_task: Optional[asyncio.Task] = None
        self.running = False
        
        # Statistics
        self.cleanup_stats = {
            "total_cleanups": 0,
            "memory_freed_mb": 0,
            "files_cleaned": 0,
            "last_cleanup": None
        }
    
    async def start(self):
        """Start the resource management system"""
        if self.running:
            return
        
        self.running = True
        self.memory_tracker.start_tracking()
        
        # Register default cleanup tasks
        self._register_default_cleanup_tasks()
        
        # Start background monitoring
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        logger.info("Resource management system started")
    
    async def stop(self):
        """Stop the resource management system"""
        self.running = False
        
        # Cancel background tasks
        if self.monitoring_task:
            self.monitoring_task.cancel()
        if self.cleanup_task:
            self.cleanup_task.cancel()
        
        # Wait for tasks to complete
        tasks = [t for t in [self.monitoring_task, self.cleanup_task] if t]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        
        # Cleanup resources
        self.memory_tracker.stop_tracking()
        for pool in self.resource_pools.values():
            pool.cleanup_all()
        
        logger.info("Resource management system stopped")
    
    def _register_default_cleanup_tasks(self):
        """Register default cleanup tasks"""
        
        # Memory cleanup task
        self.register_cleanup_task(
            "memory_gc",
            ResourceType.MEMORY,
            CleanupPriority.HIGH,
            self._cleanup_memory,
            "Force garbage collection to free memory"
        )
        
        # Temporary files cleanup
        self.register_cleanup_task(
            "temp_files",
            ResourceType.TEMPORARY_FILES,
            CleanupPriority.MEDIUM,
            self._cleanup_temp_files,
            "Remove old temporary files"
        )
        
        # Cache cleanup (if available)
        self.register_cleanup_task(
            "cache_cleanup",
            ResourceType.CACHE_ENTRIES,
            CleanupPriority.LOW,
            self._cleanup_cache,
            "Clean up expired cache entries"
        )
    
    def register_cleanup_task(self, task_id: str, resource_type: ResourceType,
                            priority: CleanupPriority, cleanup_function: Callable,
                            description: str, estimated_savings: float = 0.0):
        """Register a cleanup task"""
        task = CleanupTask(
            task_id=task_id,
            resource_type=resource_type,
            priority=priority,
            cleanup_function=cleanup_function,
            description=description,
            estimated_savings=estimated_savings
        )
        
        self.cleanup_tasks[task_id] = task
        logger.debug(f"Registered cleanup task: {task_id}")
    
    def create_resource_pool(self, pool_name: str, resource_factory: Callable,
                           max_size: int = 10, cleanup_function: Optional[Callable] = None) -> ResourcePool:
        """Create a new resource pool"""
        pool = ResourcePool(resource_factory, max_size, cleanup_function)
        self.resource_pools[pool_name] = pool
        return pool
    
    def get_resource_pool(self, pool_name: str) -> Optional[ResourcePool]:
        """Get an existing resource pool"""
        return self.resource_pools.get(pool_name)
    
    async def get_resource_usage(self) -> Dict[ResourceType, ResourceUsage]:
        """Get current resource usage for all monitored resources"""
        usage = {}
        
        # Memory usage
        memory_stats = self.memory_tracker.get_current_memory_usage()
        if memory_stats:
            usage[ResourceType.MEMORY] = ResourceUsage(
                resource_type=ResourceType.MEMORY,
                current_usage=memory_stats.get("percent", 0),
                peak_usage=memory_stats.get("traced_peak_mb", 0),
                limit=self.resource_limits.get(ResourceType.MEMORY),
                metadata=memory_stats
            )
        
        # Temporary files usage
        temp_stats = self.temp_file_manager.get_temp_usage()
        usage[ResourceType.TEMPORARY_FILES] = ResourceUsage(
            resource_type=ResourceType.TEMPORARY_FILES,
            current_usage=temp_stats["total_size_mb"],
            peak_usage=temp_stats["total_size_mb"],  # TODO: Track peak
            limit=self.resource_limits.get(ResourceType.TEMPORARY_FILES),
            metadata=temp_stats
        )
        
        return usage
    
    async def check_resource_limits(self) -> List[ResourceUsage]:
        """Check if any resources exceed their limits"""
        violations = []
        usage_data = await self.get_resource_usage()
        
        for resource_type, usage in usage_data.items():
            if not usage.limit or not usage.limit.enabled:
                continue
            
            limit = usage.limit
            
            # Check if limits are exceeded
            if usage.current_usage >= limit.hard_limit:
                violations.append(usage)
                logger.warning(
                    f"Resource {resource_type.value} exceeded hard limit: "
                    f"{usage.current_usage:.1f} >= {limit.hard_limit:.1f} {limit.unit}"
                )
            elif usage.current_usage >= limit.soft_limit:
                violations.append(usage)
                logger.info(
                    f"Resource {resource_type.value} exceeded soft limit: "
                    f"{usage.current_usage:.1f} >= {limit.soft_limit:.1f} {limit.unit}"
                )
        
        return violations
    
    async def trigger_cleanup(self, priority_threshold: CleanupPriority = CleanupPriority.MEDIUM) -> Dict[str, Any]:
        """Trigger cleanup operations based on priority"""
        results = {
            "tasks_executed": 0,
            "tasks_successful": 0,
            "memory_freed_mb": 0,
            "errors": []
        }
        
        # Sort tasks by priority
        priority_order = {
            CleanupPriority.CRITICAL: 4,
            CleanupPriority.HIGH: 3,
            CleanupPriority.MEDIUM: 2,
            CleanupPriority.LOW: 1
        }
        
        threshold_value = priority_order[priority_threshold]
        
        eligible_tasks = [
            task for task in self.cleanup_tasks.values()
            if priority_order[task.priority] >= threshold_value
        ]
        
        eligible_tasks.sort(key=lambda t: priority_order[t.priority], reverse=True)
        
        # Execute cleanup tasks
        for task in eligible_tasks:
            try:
                results["tasks_executed"] += 1
                
                # Record memory before cleanup
                memory_before = self.memory_tracker.get_current_memory_usage()
                
                # Execute cleanup
                success = task.cleanup_function()
                
                if success:
                    results["tasks_successful"] += 1
                    task.success_count += 1
                    
                    # Calculate memory freed
                    memory_after = self.memory_tracker.get_current_memory_usage()
                    if memory_before and memory_after:
                        memory_freed = memory_before.get("rss_mb", 0) - memory_after.get("rss_mb", 0)
                        if memory_freed > 0:
                            results["memory_freed_mb"] += memory_freed
                
                task.execution_count += 1
                task.last_executed = datetime.now()
                
            except Exception as e:
                error_msg = f"Cleanup task {task.task_id} failed: {e}"
                logger.error(error_msg)
                results["errors"].append(error_msg)
        
        # Update statistics
        self.cleanup_stats["total_cleanups"] += results["tasks_executed"]
        self.cleanup_stats["memory_freed_mb"] += results["memory_freed_mb"]
        self.cleanup_stats["last_cleanup"] = datetime.now()
        
        return results
    
    def _cleanup_memory(self) -> bool:
        """Memory cleanup implementation"""
        try:
            # Force garbage collection
            gc_stats = self.gc_manager.force_garbage_collection()
            
            # Optimize GC thresholds
            self.gc_manager.optimize_gc_thresholds()
            
            logger.info(f"Memory cleanup completed: {gc_stats['collected']} objects collected")
            return True
        except Exception as e:
            logger.error(f"Memory cleanup failed: {e}")
            return False
    
    def _cleanup_temp_files(self) -> bool:
        """Temporary files cleanup implementation"""
        try:
            cleaned_count = self.temp_file_manager.cleanup_old_files(max_age_hours=24)
            self.cleanup_stats["files_cleaned"] += cleaned_count
            
            logger.info(f"Temporary files cleanup completed: {cleaned_count} files removed")
            return True
        except Exception as e:
            logger.error(f"Temporary files cleanup failed: {e}")
            return False
    
    def _cleanup_cache(self) -> bool:
        """Cache cleanup implementation"""
        try:
            # Try to cleanup multimodal cache if available
            asyncio.create_task(self._async_cache_cleanup())
            return True
        except Exception as e:
            logger.error(f"Cache cleanup failed: {e}")
            return False
    
    async def _async_cache_cleanup(self):
        """Async cache cleanup"""
        try:
            from ..cache.multimodal_cache import get_multimodal_cache
            cache_system = await get_multimodal_cache()
            await cache_system.optimize_cache()
        except Exception as e:
            logger.warning(f"Cache cleanup not available: {e}")
    
    async def _monitoring_loop(self):
        """Background monitoring loop"""
        while self.running:
            try:
                # Check resource limits
                violations = await self.check_resource_limits()
                
                # Trigger cleanup if needed
                if violations:
                    critical_violations = [v for v in violations if v.current_usage >= v.limit.hard_limit]
                    
                    if critical_violations:
                        await self.trigger_cleanup(CleanupPriority.CRITICAL)
                    else:
                        await self.trigger_cleanup(CleanupPriority.HIGH)
                
                # Take memory snapshot periodically
                if len(self.memory_tracker.snapshots) == 0 or \
                   (datetime.now() - self.memory_tracker.snapshots[-1][0]).seconds > 300:
                    self.memory_tracker.take_snapshot("periodic")
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")
                await asyncio.sleep(60)
    
    async def _cleanup_loop(self):
        """Background cleanup loop"""
        while self.running:
            try:
                # Periodic cleanup (every 10 minutes)
                await self.trigger_cleanup(CleanupPriority.LOW)
                
                # Check if forced GC is needed
                if self.gc_manager.should_force_gc():
                    self.gc_manager.force_garbage_collection()
                
                await asyncio.sleep(600)  # Run every 10 minutes
                
            except Exception as e:
                logger.error(f"Background cleanup error: {e}")
                await asyncio.sleep(600)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive resource management statistics"""
        stats = {
            "resource_usage": {},
            "cleanup_stats": self.cleanup_stats.copy(),
            "gc_stats": self.gc_manager.gc_stats[-5:] if self.gc_manager.gc_stats else [],
            "memory_analysis": self.memory_tracker.analyze_memory_growth(),
            "resource_pools": {},
            "temp_files": self.temp_file_manager.get_temp_usage()
        }
        
        # Add resource pool statistics
        for pool_name, pool in self.resource_pools.items():
            stats["resource_pools"][pool_name] = pool.get_stats()
        
        return stats


# Global resource manager instance
_global_resource_manager: Optional[ResourceManager] = None


async def get_resource_manager() -> ResourceManager:
    """Get or create the global resource manager"""
    global _global_resource_manager
    if _global_resource_manager is None:
        _global_resource_manager = ResourceManager()
        await _global_resource_manager.start()
    return _global_resource_manager


async def initialize_resource_management():
    """Initialize the resource management system"""
    manager = await get_resource_manager()
    logger.info("Resource management system initialized")
    return manager