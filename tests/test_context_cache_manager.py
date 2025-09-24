#!/usr/bin/env python3

import unittest
import tempfile
import shutil
import os
import json
import time
import threading
import multiprocessing
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

from xencode.context_cache_manager import (
    ContextCacheManager,
    ConversationCache,
    ContextData,
    CacheLock,
    CacheVersion,
    get_cache_manager,
)
class TestContextCacheManager(unittest.TestCase):
    """Comprehensive test suite for ContextCacheManager"""
    
    def setUp(self):
        """Set up test environment with temporary directory"""
        self.test_dir = tempfile.mkdtemp()
        self.cache_manager = ContextCacheManager(cache_base_dir=self.test_dir)
        self.test_project_hash = "test_project_123"
        
        # Sample context data
        self.sample_context = ContextData(
            project_hash=self.test_project_hash,
            project_root="/test/project",
            files=[
                {"path": "main.py", "size": 1024, "type": "python"},
                {"path": "README.md", "size": 512, "type": "markdown"}
            ],
            file_type_breakdown={"python": 1, "markdown": 1},
            excluded_files=["secret.env"],
            security_alerts=[],
            scan_duration_ms=150,
            total_size_mb=1.5,
            estimated_tokens=500
        )
    
    def tearDown(self):
        """Clean up test environment"""
        # Release any active locks
        for project_hash in list(self.cache_manager._active_locks.keys()):
            self.cache_manager.release_cache_lock(project_hash)
        
        # Remove test directory
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_cache_manager_initialization(self):
        """Test cache manager initialization"""
        self.assertTrue(self.cache_manager.cache_base_dir.exists())
        self.assertEqual(len(self.cache_manager._active_locks), 0)
        self.assertIn(CacheVersion.V1, self.cache_manager._migration_handlers)
    
    def test_context_data_serialization(self):
        """Test ContextData serialization and deserialization"""
        # Test to_dict
        data_dict = self.sample_context.to_dict()
        self.assertIn('version', data_dict)
        self.assertIn('project_hash', data_dict)
        self.assertIn('timestamp', data_dict)
        self.assertIsInstance(data_dict['timestamp'], str)
        
        # Test from_dict
        restored_context = ContextData.from_dict(data_dict)
        self.assertEqual(restored_context.project_hash, self.sample_context.project_hash)
        self.assertEqual(restored_context.project_root, self.sample_context.project_root)
        self.assertEqual(len(restored_context.files), len(self.sample_context.files))
    
    def test_cache_lock_serialization(self):
        """Test CacheLock serialization and deserialization"""
        lock = CacheLock(
            project_hash="test_hash",
            pid=12345,
            timestamp=datetime.now(),
            lock_file_path="/test/path"
        )
        
        # Test to_dict
        lock_dict = lock.to_dict()
        self.assertIn('pid', lock_dict)
        self.assertIn('timestamp', lock_dict)
        self.assertIsInstance(lock_dict['timestamp'], str)
        
        # Test from_dict
        restored_lock = CacheLock.from_dict(lock_dict)
        self.assertEqual(restored_lock.pid, lock.pid)
        self.assertEqual(restored_lock.project_hash, lock.project_hash)
    
    def test_basic_save_and_load(self):
        """Test basic save and load operations"""
        # Save context
        success = self.cache_manager.save_context(self.test_project_hash, self.sample_context)
        self.assertTrue(success)
        
        # Load context
        loaded_context = self.cache_manager.load_context(self.test_project_hash)
        self.assertIsNotNone(loaded_context)
        self.assertEqual(loaded_context.project_hash, self.sample_context.project_hash)
        self.assertEqual(loaded_context.project_root, self.sample_context.project_root)
        self.assertEqual(len(loaded_context.files), len(self.sample_context.files))
    
    def test_checksum_validation(self):
        """Test checksum validation for data integrity"""
        # Save context
        self.cache_manager.save_context(self.test_project_hash, self.sample_context)
        
        # Verify cache file has checksum
        cache_file = self.cache_manager._get_cache_file_path(self.test_project_hash)
        with open(cache_file, 'r') as f:
            data = json.load(f)
        self.assertIn('checksum', data)
        
        # Verify integrity validation passes
        self.assertTrue(self.cache_manager.validate_cache_integrity(cache_file))
        
        # Corrupt the file and verify validation fails
        data['project_root'] = "corrupted_path"
        with open(cache_file, 'w') as f:
            json.dump(data, f)
        self.assertFalse(self.cache_manager.validate_cache_integrity(cache_file))
    
    def test_atomic_write_operations(self):
        """Test atomic write operations using temporary files"""
        # Mock a failure during write to test cleanup
        original_replace = Path.replace
        
        def mock_replace_failure(self, target):
            raise OSError("Simulated write failure")
        
        with patch.object(Path, 'replace', mock_replace_failure):
            success = self.cache_manager.save_context(self.test_project_hash, self.sample_context)
            self.assertFalse(success)
        
        # Verify no temporary files left behind
        cache_dir = self.cache_manager._get_project_cache_dir(self.test_project_hash)
        temp_files = list(cache_dir.glob("*.tmp"))
        self.assertEqual(len(temp_files), 0)
        
        # Verify normal operation works
        success = self.cache_manager.save_context(self.test_project_hash, self.sample_context)
        self.assertTrue(success)
    
    def test_lock_acquisition_and_release(self):
        """Test lock acquisition and release mechanisms"""
        # Acquire lock
        success = self.cache_manager.acquire_cache_lock(self.test_project_hash)
        self.assertTrue(success)
        self.assertIn(self.test_project_hash, self.cache_manager._active_locks)
        
        # Verify lock file exists
        lock_file = self.cache_manager._get_lock_file_path(self.test_project_hash)
        self.assertTrue(lock_file.exists())
        
        # Release lock
        self.cache_manager.release_cache_lock(self.test_project_hash)
        self.assertNotIn(self.test_project_hash, self.cache_manager._active_locks)
        self.assertFalse(lock_file.exists())
    
    def test_concurrent_lock_prevention(self):
        """Test that concurrent access is properly prevented"""
        # Acquire lock in main thread
        success1 = self.cache_manager.acquire_cache_lock(self.test_project_hash)
        self.assertTrue(success1)
        
        # Try to acquire same lock (should fail)
        success2 = self.cache_manager.acquire_cache_lock(self.test_project_hash)
        self.assertFalse(success2)
        
        # Release first lock
        self.cache_manager.release_cache_lock(self.test_project_hash)
        
        # Now second acquisition should succeed
        success3 = self.cache_manager.acquire_cache_lock(self.test_project_hash)
        self.assertTrue(success3)
        
        # Clean up
        self.cache_manager.release_cache_lock(self.test_project_hash)
    
    def test_stale_lock_cleanup(self):
        """Test cleanup of stale locks from dead processes"""
        # Create a fake stale lock with non-existent PID
        fake_pid = 999999  # Very unlikely to exist
        lock_file = self.cache_manager._get_lock_file_path(self.test_project_hash)
        
        stale_lock = CacheLock(
            project_hash=self.test_project_hash,
            pid=fake_pid,
            timestamp=datetime.now() - timedelta(minutes=10),
            lock_file_path=str(lock_file)
        )
        
        # Write stale lock
        lock_file.parent.mkdir(parents=True, exist_ok=True)
        with open(lock_file, 'w') as f:
            json.dump(stale_lock.to_dict(), f)
        
        # Cleanup should remove the stale lock
        cleaned_count = self.cache_manager.cleanup_stale_locks()
        self.assertGreaterEqual(cleaned_count, 1)
        self.assertFalse(lock_file.exists())
    
    def test_lock_timeout_handling(self):
        """Test handling of timed-out locks"""
        # Create an old lock that should be considered timed out
        old_timestamp = datetime.now() - timedelta(seconds=self.cache_manager._lock_timeout_seconds + 10)
        lock_file = self.cache_manager._get_lock_file_path(self.test_project_hash)
        
        old_lock = CacheLock(
            project_hash=self.test_project_hash,
            pid=os.getpid(),  # Use current PID so it appears "running"
            timestamp=old_timestamp,
            lock_file_path=str(lock_file)
        )
        
        # Write old lock
        lock_file.parent.mkdir(parents=True, exist_ok=True)
        with open(lock_file, 'w') as f:
            json.dump(old_lock.to_dict(), f)
        
        # Should be able to acquire lock despite existing file (due to timeout)
        success = self.cache_manager.acquire_cache_lock(self.test_project_hash)
        self.assertTrue(success)
        
        # Clean up
        self.cache_manager.release_cache_lock(self.test_project_hash)
    
    def test_cache_version_migration(self):
        """Test cache version migration from V1 to V2"""
        # Create V1 format data
        v1_data = {
            'version': 1,
            'project_hash': self.test_project_hash,
            'project_root': '/test/project',
            'timestamp': datetime.now().isoformat(),
            'files': [{'path': 'test.py', 'size': 100}],
            'file_type_breakdown': {'python': 1},
            'excluded_files': [],
            'scan_duration_ms': 100,
            'total_size_mb': 0.1
        }
        
        # Test migration
        migrated_data = self.cache_manager.migrate_cache_version(v1_data)
        
        # Verify V2 fields were added
        self.assertEqual(migrated_data['version'], CacheVersion.V2.value)
        self.assertIn('security_alerts', migrated_data)
        self.assertIn('estimated_tokens', migrated_data)
        self.assertIn('semantic_index', migrated_data)
        
        # Verify original fields preserved
        self.assertEqual(migrated_data['project_hash'], v1_data['project_hash'])
        self.assertEqual(migrated_data['files'], v1_data['files'])
    
    def test_backup_and_recovery(self):
        """Test backup creation and recovery mechanisms"""
        # Save initial context
        self.cache_manager.save_context(self.test_project_hash, self.sample_context)
        
        # Modify and save again to trigger backup creation
        modified_context = ContextData(
            project_hash=self.test_project_hash,
            project_root="/modified/project",
            files=[{"path": "modified.py", "size": 2048}]
        )
        self.cache_manager.save_context(self.test_project_hash, modified_context)
        
        # Verify backup was created
        backup_file = self.cache_manager._get_backup_file_path(self.test_project_hash)
        self.assertTrue(backup_file.exists())
        
        # Corrupt the main cache file
        cache_file = self.cache_manager._get_cache_file_path(self.test_project_hash)
        with open(cache_file, 'w') as f:
            f.write("corrupted data")
        
        # Load should trigger recovery from backup
        loaded_context = self.cache_manager.load_context(self.test_project_hash)
        self.assertIsNotNone(loaded_context)
        # Should load the original context from backup
        self.assertEqual(loaded_context.project_hash, self.test_project_hash)
    
    def test_old_cache_cleanup(self):
        """Test cleanup of old cache files"""
        # Create an old cache file
        old_context = ContextData(
            project_hash="old_project",
            project_root="/old/project",
            timestamp=datetime.now() - timedelta(hours=25)  # Older than 24 hours
        )
        
        self.cache_manager.save_context("old_project", old_context)
        
        # Manually set file modification time to be old
        old_cache_file = self.cache_manager._get_cache_file_path("old_project")
        old_time = time.time() - (25 * 3600)  # 25 hours ago
        os.utime(old_cache_file, (old_time, old_time))
        
        # Run cleanup
        cleaned_count = self.cache_manager.cleanup_old_caches(max_age_hours=24)
        self.assertGreaterEqual(cleaned_count, 1)
        
        # Verify old cache was removed
        self.assertFalse(old_cache_file.exists())
    
    def test_cache_stats(self):
        """Test cache statistics generation"""
        # Save some test data
        self.cache_manager.save_context(self.test_project_hash, self.sample_context)
        
        # Get stats
        stats = self.cache_manager.get_cache_stats()
        
        # Verify stats structure
        self.assertIn('total_projects', stats)
        self.assertIn('total_cache_size_mb', stats)
        self.assertIn('active_locks', stats)
        self.assertIn('cache_base_dir', stats)
        
        # Verify stats values
        self.assertGreaterEqual(stats['total_projects'], 1)
        self.assertGreater(stats['total_cache_size_mb'], 0)
        self.assertEqual(stats['cache_base_dir'], str(self.cache_manager.cache_base_dir))
    
    def test_concurrent_access_simulation(self):
        """Test concurrent access using threading"""
        results = []
        errors = []
        
        def worker_save(worker_id):
            try:
                context = ContextData(
                    project_hash=f"concurrent_test_{worker_id}",
                    project_root=f"/test/worker_{worker_id}",
                    files=[{"path": f"worker_{worker_id}.py", "size": 100}]
                )
                success = self.cache_manager.save_context(f"concurrent_test_{worker_id}", context)
                results.append((worker_id, success))
            except Exception as e:
                errors.append((worker_id, str(e)))
        
        def worker_load(worker_id):
            try:
                loaded = self.cache_manager.load_context(f"concurrent_test_{worker_id}")
                results.append((worker_id, loaded is not None))
            except Exception as e:
                errors.append((worker_id, str(e)))
        
        # Create multiple threads for saving
        save_threads = []
        for i in range(5):
            thread = threading.Thread(target=worker_save, args=(i,))
            save_threads.append(thread)
            thread.start()
        
        # Wait for all save threads
        for thread in save_threads:
            thread.join()
        
        # Create multiple threads for loading
        load_threads = []
        for i in range(5):
            thread = threading.Thread(target=worker_load, args=(i,))
            load_threads.append(thread)
            thread.start()
        
        # Wait for all load threads
        for thread in load_threads:
            thread.join()
        
        # Verify no errors occurred
        self.assertEqual(len(errors), 0, f"Errors occurred: {errors}")
        
        # Verify all operations succeeded
        save_results = [r for r in results if len(results) >= 5]
        self.assertGreaterEqual(len(save_results), 5)
    
    def test_same_project_concurrent_access(self):
        """Test concurrent access to the same project (should be serialized)"""
        results = []
        
        def worker_same_project(worker_id):
            try:
                # All workers try to access the same project
                context = ContextData(
                    project_hash=self.test_project_hash,
                    project_root=f"/test/worker_{worker_id}",
                    files=[{"path": f"worker_{worker_id}.py", "size": 100}]
                )
                success = self.cache_manager.save_context(self.test_project_hash, context)
                results.append((worker_id, success))
                time.sleep(0.1)  # Hold lock briefly
            except Exception as e:
                results.append((worker_id, False))
        
        # Create multiple threads accessing same project
        threads = []
        for i in range(3):
            thread = threading.Thread(target=worker_same_project, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # At least one should succeed, others may fail due to locking
        successful_operations = sum(1 for _, success in results if success)
        self.assertGreaterEqual(successful_operations, 1)
        
        # Verify final state is consistent
        final_context = self.cache_manager.load_context(self.test_project_hash)
        self.assertIsNotNone(final_context)
    
    def test_global_cache_manager_instance(self):
        """Test global cache manager instance"""
        manager1 = get_cache_manager()
        manager2 = get_cache_manager()
        
        # Should return the same instance
        self.assertIs(manager1, manager2)
        self.assertIsInstance(manager1, ContextCacheManager)
    
    def test_corrupted_lock_file_handling(self):
        """Test handling of corrupted lock files"""
        lock_file = self.cache_manager._get_lock_file_path(self.test_project_hash)
        lock_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Write corrupted lock file
        with open(lock_file, 'w') as f:
            f.write("corrupted json data")
        
        # Should be able to acquire lock despite corrupted file
        success = self.cache_manager.acquire_cache_lock(self.test_project_hash)
        self.assertTrue(success)
        
        # Clean up
        self.cache_manager.release_cache_lock(self.test_project_hash)
    
    def test_missing_cache_directory_creation(self):
        """Test automatic creation of missing cache directories"""
        # Remove cache directory
        shutil.rmtree(self.test_dir)
        
        # Create new manager (should recreate directory)
        new_manager = ContextCacheManager(cache_base_dir=self.test_dir)
        self.assertTrue(new_manager.cache_base_dir.exists())
        
        # Should be able to save context
        success = new_manager.save_context(self.test_project_hash, self.sample_context)
        self.assertTrue(success)
    
    def test_edge_case_empty_context_data(self):
        """Test handling of edge cases with empty context data"""
        empty_context = ContextData(
            project_hash="empty_test",
            project_root="/empty"
        )
        
        # Should handle empty context gracefully
        success = self.cache_manager.save_context("empty_test", empty_context)
        self.assertTrue(success)
        
        loaded_context = self.cache_manager.load_context("empty_test")
        self.assertIsNotNone(loaded_context)
        self.assertEqual(loaded_context.project_hash, "empty_test")
        self.assertEqual(len(loaded_context.files), 0)


class TestConcurrentProcessAccess(unittest.TestCase):
    """Test concurrent access from multiple processes"""
    
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.test_project_hash = "multiprocess_test"
    
    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def worker_process(self, worker_id, results_queue):
        """Worker process for multiprocess testing"""
        try:
            cache_manager = ContextCacheManager(cache_base_dir=self.test_dir)
            
            context = ContextData(
                project_hash=f"{self.test_project_hash}_{worker_id}",
                project_root=f"/test/process_{worker_id}",
                files=[{"path": f"process_{worker_id}.py", "size": 200}]
            )
            
            success = cache_manager.save_context(f"{self.test_project_hash}_{worker_id}", context)
            results_queue.put((worker_id, success))
            
        except Exception as e:
            results_queue.put((worker_id, False, str(e)))
    
    def test_multiprocess_access(self):
        """Test access from multiple processes"""
        results_queue = multiprocessing.Queue()
        processes = []
        
        # Create multiple processes
        for i in range(3):
            process = multiprocessing.Process(
                target=self.worker_process, 
                args=(i, results_queue)
            )
            processes.append(process)
            process.start()
        
        # Wait for all processes
        for process in processes:
            process.join()
        
        # Collect results
        results = []
        while not results_queue.empty():
            results.append(results_queue.get())
        
        # Verify all processes succeeded
        self.assertEqual(len(results), 3)
        for result in results:
            self.assertTrue(result[1], f"Process {result[0]} failed")


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)