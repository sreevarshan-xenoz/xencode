#!/usr/bin/env python3
"""
Tests for Resource Management System

Comprehensive tests for memory tracking, resource pooling, cleanup operations,
and resource limit monitoring.
"""

import asyncio
import gc
import pytest
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch

from xencode.monitoring.resource_manager import (
    ResourceManager,
    MemoryTracker,
    ResourcePool,
    GarbageCollectionManager,
    TemporaryFileManager,
    ResourceType,
    ResourceLimit,
    CleanupPriority,
    get_resource_manager
)


class TestMemoryTracker:
    """Test memory tracking functionality"""
    
    def test_memory_tracker_initialization(self):
        """Test memory tracker initialization"""
        tracker = MemoryTracker()
        assert not tracker.tracking_enabled
        assert len(tracker.snapshots) == 0
        assert len(tracker.memory_history) == 0
    
    def test_start_stop_tracking(self):
        """Test starting and stopping memory tracking"""
        tracker = MemoryTracker()
        
        tracker.start_tracking()
        assert tracker.tracking_enabled
        
        tracker.stop_tracking()
        assert not tracker.tracking_enabled
    
    def test_take_snapshot(self):
        """Test taking memory snapshots"""
        tracker = MemoryTracker()
        tracker.start_tracking()
        
        snapshot = tracker.take_snapshot("test")
        assert snapshot is not None
        assert len(tracker.snapshots) == 1
        
        tracker.stop_tracking()
    
    def test_memory_growth_analysis(self):
        """Test memory growth analysis"""
        tracker = MemoryTracker()
        tracker.start_tracking()
        
        # Take initial snapshot
        tracker.take_snapshot("initial")
        
        # Create some objects
        large_list = [i for i in range(1000)]
        
        # Take second snapshot
        tracker.take_snapshot("after_allocation")
        
        # Analyze growth
        analysis = tracker.analyze_memory_growth()
        
        assert "total_growth_mb" in analysis
        assert "top_growing_files" in analysis
        assert "memory_leaks_detected" in analysis
        
        tracker.stop_tracking()
        del large_list
    
    @patch('xencode.monitoring.resource_manager.psutil')
    def test_get_current_memory_usage(self, mock_psutil):
        """Test getting current memory usage"""
        # Mock psutil
        mock_process = Mock()
        mock_memory_info = Mock()
        mock_memory_info.rss = 100 * 1024 * 1024  # 100 MB
        mock_memory_info.vms = 200 * 1024 * 1024  # 200 MB
        mock_process.memory_info.return_value = mock_memory_info
        mock_process.memory_percent.return_value = 5.0
        
        mock_virtual_memory = Mock()
        mock_virtual_memory.available = 1024 * 1024 * 1024  # 1 GB
        
        mock_psutil.Process.return_value = mock_process
        mock_psutil.virtual_memory.return_value = mock_virtual_memory
        
        tracker = MemoryTracker()
        usage = tracker.get_current_memory_usage()
        
        assert "rss_mb" in usage
        assert "vms_mb" in usage
        assert "percent" in usage
        assert "available_mb" in usage
        assert usage["rss_mb"] == 100.0
        assert usage["percent"] == 5.0


class TestResourcePool:
    """Test resource pooling functionality"""
    
    def test_resource_pool_creation(self):
        """Test resource pool creation"""
        def create_resource():
            return {"id": time.time()}
        
        pool = ResourcePool(create_resource, max_size=5)
        assert pool.max_size == 5
        assert len(pool.pool) == 0
        assert len(pool.in_use) == 0
    
    def test_acquire_and_release(self):
        """Test acquiring and releasing resources"""
        def create_resource():
            return {"id": time.time()}
        
        pool = ResourcePool(create_resource, max_size=3)
        
        # Acquire resources
        resource1 = pool.acquire()
        resource2 = pool.acquire()
        
        assert len(pool.in_use) == 2
        assert pool.created_count == 2
        
        # Release resources
        pool.release(resource1)
        pool.release(resource2)
        
        assert len(pool.in_use) == 0
        assert len(pool.pool) == 2
    
    def test_pool_reuse(self):
        """Test resource reuse from pool"""
        def create_resource():
            return {"id": time.time()}
        
        pool = ResourcePool(create_resource, max_size=2)
        
        # Acquire and release
        resource1 = pool.acquire()
        pool.release(resource1)
        
        # Acquire again (should reuse)
        resource2 = pool.acquire()
        
        assert resource1 is resource2
        assert pool.reuse_count == 1
    
    def test_cleanup_function(self):
        """Test resource cleanup function"""
        cleanup_called = []
        
        def create_resource():
            return {"id": time.time()}
        
        def cleanup_resource(resource):
            cleanup_called.append(resource)
        
        pool = ResourcePool(create_resource, max_size=1, cleanup_function=cleanup_resource)
        
        # Fill pool beyond capacity
        resource1 = pool.acquire()
        resource2 = pool.acquire()
        
        pool.release(resource1)  # Goes to pool
        pool.release(resource2)  # Should trigger cleanup
        
        assert len(cleanup_called) == 1
        assert cleanup_called[0] is resource2
    
    def test_get_stats(self):
        """Test getting pool statistics"""
        def create_resource():
            return {"id": time.time()}
        
        pool = ResourcePool(create_resource, max_size=3)
        
        resource1 = pool.acquire()
        resource2 = pool.acquire()
        pool.release(resource1)
        
        stats = pool.get_stats()
        
        assert stats["pool_size"] == 1
        assert stats["in_use"] == 1
        assert stats["created_count"] == 2
        assert stats["reuse_count"] == 0


class TestGarbageCollectionManager:
    """Test garbage collection management"""
    
    def test_gc_manager_initialization(self):
        """Test GC manager initialization"""
        manager = GarbageCollectionManager()
        assert manager.auto_gc_enabled
        assert len(manager.gc_stats) == 0
    
    def test_force_garbage_collection(self):
        """Test forcing garbage collection"""
        manager = GarbageCollectionManager()
        
        # Create some objects to collect
        objects = [[] for _ in range(100)]
        del objects
        
        stats = manager.force_garbage_collection()
        
        assert "timestamp" in stats
        assert "duration" in stats
        assert "collected" in stats
        assert stats["collected"] >= 0
    
    def test_should_force_gc(self):
        """Test GC forcing logic"""
        manager = GarbageCollectionManager()
        
        # Should not force GC immediately after creation
        assert not manager.should_force_gc()
        
        # Simulate old last GC time
        manager.last_gc_time = time.time() - 400  # 400 seconds ago
        assert manager.should_force_gc()


class TestTemporaryFileManager:
    """Test temporary file management"""
    
    def test_temp_file_manager_initialization(self):
        """Test temporary file manager initialization"""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = TemporaryFileManager(Path(temp_dir) / "test_temp")
            assert manager.base_temp_dir.exists()
            assert len(manager.tracked_files) == 0
    
    def test_create_temp_file(self):
        """Test creating temporary files"""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = TemporaryFileManager(Path(temp_dir) / "test_temp")
            
            file_id, file_path = manager.create_temp_file(prefix="test_", suffix=".txt")
            
            assert file_id in manager.tracked_files
            assert file_path.exists()
            assert file_path.name.startswith("test_")
            assert file_path.name.endswith(".txt")
    
    def test_cleanup_temp_file(self):
        """Test cleaning up temporary files"""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = TemporaryFileManager(Path(temp_dir) / "test_temp")
            
            file_id, file_path = manager.create_temp_file()
            assert file_path.exists()
            
            success = manager.cleanup_temp_file(file_id)
            assert success
            assert not file_path.exists()
            assert file_id not in manager.tracked_files
    
    def test_cleanup_old_files(self):
        """Test cleaning up old temporary files"""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = TemporaryFileManager(Path(temp_dir) / "test_temp")
            
            # Create files
            file_id1, _ = manager.create_temp_file()
            file_id2, _ = manager.create_temp_file()
            
            # Simulate old files by modifying metadata
            from datetime import datetime, timedelta
            old_time = datetime.now() - timedelta(hours=25)
            manager.file_metadata[file_id1]["created_at"] = old_time
            
            # Cleanup old files
            cleaned_count = manager.cleanup_old_files(max_age_hours=24)
            
            assert cleaned_count == 1
            assert file_id1 not in manager.tracked_files
            assert file_id2 in manager.tracked_files
    
    def test_get_temp_usage(self):
        """Test getting temporary file usage statistics"""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = TemporaryFileManager(Path(temp_dir) / "test_temp")
            
            # Create some files
            file_id1, file_path1 = manager.create_temp_file()
            file_id2, file_path2 = manager.create_temp_file()
            
            # Write data to files
            file_path1.write_text("test data 1")
            file_path2.write_text("test data 2")
            
            usage = manager.get_temp_usage()
            
            assert usage["total_files"] == 2
            assert usage["total_size_mb"] > 0
            assert "base_directory" in usage


class TestResourceManager:
    """Test main resource manager functionality"""
    
    @pytest.fixture
    async def resource_manager(self):
        """Create a resource manager for testing"""
        manager = ResourceManager()
        await manager.start()
        yield manager
        await manager.stop()
    
    @pytest.mark.asyncio
    async def test_resource_manager_initialization(self, resource_manager):
        """Test resource manager initialization"""
        assert resource_manager.running
        assert resource_manager.memory_tracker is not None
        assert resource_manager.gc_manager is not None
        assert resource_manager.temp_file_manager is not None
    
    @pytest.mark.asyncio
    async def test_get_resource_usage(self, resource_manager):
        """Test getting resource usage"""
        usage = await resource_manager.get_resource_usage()
        
        assert ResourceType.MEMORY in usage
        assert ResourceType.TEMPORARY_FILES in usage
        
        memory_usage = usage[ResourceType.MEMORY]
        assert memory_usage.resource_type == ResourceType.MEMORY
        assert memory_usage.current_usage >= 0
    
    @pytest.mark.asyncio
    async def test_check_resource_limits(self, resource_manager):
        """Test checking resource limits"""
        # Set very low limits for testing
        resource_manager.resource_limits[ResourceType.MEMORY] = ResourceLimit(
            ResourceType.MEMORY,
            soft_limit=0.1,
            hard_limit=0.2,
            unit="percentage"
        )
        
        violations = await resource_manager.check_resource_limits()
        
        # Should have violations due to low limits
        assert len(violations) > 0
        assert violations[0].resource_type == ResourceType.MEMORY
    
    @pytest.mark.asyncio
    async def test_register_cleanup_task(self, resource_manager):
        """Test registering cleanup tasks"""
        cleanup_called = []
        
        def test_cleanup():
            cleanup_called.append(True)
            return True
        
        resource_manager.register_cleanup_task(
            "test_cleanup",
            ResourceType.MEMORY,
            CleanupPriority.LOW,
            test_cleanup,
            "Test cleanup task"
        )
        
        assert "test_cleanup" in resource_manager.cleanup_tasks
        
        # Trigger cleanup
        results = await resource_manager.trigger_cleanup(CleanupPriority.LOW)
        
        assert len(cleanup_called) > 0
        assert results["tasks_executed"] > 0
    
    @pytest.mark.asyncio
    async def test_create_resource_pool(self, resource_manager):
        """Test creating resource pools"""
        def create_test_resource():
            return {"test": True}
        
        pool = resource_manager.create_resource_pool(
            "test_pool",
            create_test_resource,
            max_size=3
        )
        
        assert pool is not None
        assert resource_manager.get_resource_pool("test_pool") is pool
        
        # Test pool usage
        resource = pool.acquire()
        assert resource["test"] is True
        
        pool.release(resource)
    
    @pytest.mark.asyncio
    async def test_trigger_cleanup(self, resource_manager):
        """Test triggering cleanup operations"""
        results = await resource_manager.trigger_cleanup(CleanupPriority.LOW)
        
        assert "tasks_executed" in results
        assert "tasks_successful" in results
        assert "memory_freed_mb" in results
        assert "errors" in results
        
        assert results["tasks_executed"] >= 0
        assert results["tasks_successful"] >= 0
    
    @pytest.mark.asyncio
    async def test_get_statistics(self, resource_manager):
        """Test getting comprehensive statistics"""
        stats = resource_manager.get_statistics()
        
        assert "resource_usage" in stats
        assert "cleanup_stats" in stats
        assert "gc_stats" in stats
        assert "memory_analysis" in stats
        assert "resource_pools" in stats
        assert "temp_files" in stats
        
        cleanup_stats = stats["cleanup_stats"]
        assert "total_cleanups" in cleanup_stats
        assert "memory_freed_mb" in cleanup_stats


class TestIntegration:
    """Integration tests for resource management system"""
    
    @pytest.mark.asyncio
    async def test_global_resource_manager(self):
        """Test global resource manager singleton"""
        manager1 = await get_resource_manager()
        manager2 = await get_resource_manager()
        
        assert manager1 is manager2
        assert manager1.running
    
    @pytest.mark.asyncio
    async def test_memory_pressure_scenario(self):
        """Test system behavior under memory pressure"""
        resource_manager = await get_resource_manager()
        
        # Create memory pressure
        large_objects = []
        for i in range(10):
            large_objects.append([j for j in range(1000)])
        
        # Take snapshot
        resource_manager.memory_tracker.take_snapshot("pressure_test")
        
        # Trigger cleanup
        results = await resource_manager.trigger_cleanup(CleanupPriority.HIGH)
        
        # Cleanup should have been executed
        assert results["tasks_executed"] > 0
        
        # Clean up test objects
        del large_objects
        gc.collect()
    
    @pytest.mark.asyncio
    async def test_resource_pool_under_load(self):
        """Test resource pool behavior under load"""
        resource_manager = await get_resource_manager()
        
        def create_expensive_resource():
            time.sleep(0.01)  # Simulate expensive creation
            return {"created_at": time.time()}
        
        pool = resource_manager.create_resource_pool(
            "load_test_pool",
            create_expensive_resource,
            max_size=5
        )
        
        # Simulate concurrent access
        resources = []
        for i in range(10):
            resource = pool.acquire()
            resources.append(resource)
        
        # Release all resources
        for resource in resources:
            pool.release(resource)
        
        # Check pool statistics
        stats = pool.get_stats()
        assert stats["created_count"] == 10
        assert stats["pool_size"] <= 5  # Pool size should be limited
        assert stats["reuse_ratio"] >= 0  # Some reuse should occur


if __name__ == "__main__":
    pytest.main([__file__, "-v"])