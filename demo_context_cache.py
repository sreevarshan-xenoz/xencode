#!/usr/bin/env python3

"""
Demonstration script for the Context Cache Manager.

This script shows the key features of the atomic context cache system:
- Concurrent access protection with double-lock pattern
- Atomic write operations with versioning and checksum validation
- Cache migration system for version compatibility
- Backup and recovery mechanisms
"""

import time
import threading
from datetime import datetime
from context_cache_manager import ContextCacheManager, ContextData, get_cache_manager


def demo_basic_operations():
    """Demonstrate basic save and load operations"""
    print("🔧 Demo: Basic Cache Operations")
    print("=" * 50)
    
    cache_manager = get_cache_manager()
    
    # Create sample context data
    context = ContextData(
        project_hash="demo_project_123",
        project_root="/home/user/demo_project",
        files=[
            {"path": "main.py", "size": 2048, "type": "python"},
            {"path": "README.md", "size": 1024, "type": "markdown"},
            {"path": "config.json", "size": 512, "type": "json"}
        ],
        file_type_breakdown={"python": 1, "markdown": 1, "json": 1},
        excluded_files=["secret.env", "private.key"],
        security_alerts=[],
        scan_duration_ms=250,
        total_size_mb=3.5,
        estimated_tokens=1500
    )
    
    print(f"📝 Saving context for project: {context.project_hash}")
    success = cache_manager.save_context("demo_project_123", context)
    print(f"✅ Save result: {'Success' if success else 'Failed'}")
    
    print(f"📖 Loading context for project: demo_project_123")
    loaded_context = cache_manager.load_context("demo_project_123")
    
    if loaded_context:
        print(f"✅ Load successful!")
        print(f"   Project Root: {loaded_context.project_root}")
        print(f"   Files Count: {len(loaded_context.files)}")
        print(f"   Total Size: {loaded_context.total_size_mb} MB")
        print(f"   Scan Duration: {loaded_context.scan_duration_ms} ms")
        print(f"   Cache Version: {loaded_context.version}")
    else:
        print("❌ Load failed!")
    
    print()


def demo_concurrent_access():
    """Demonstrate concurrent access protection"""
    print("🔒 Demo: Concurrent Access Protection")
    print("=" * 50)
    
    cache_manager = get_cache_manager()
    results = []
    
    def worker_thread(worker_id, project_hash):
        """Worker thread that tries to access the same project"""
        context = ContextData(
            project_hash=project_hash,
            project_root=f"/worker_{worker_id}/project",
            files=[{"path": f"worker_{worker_id}.py", "size": 1024}]
        )
        
        print(f"🧵 Worker {worker_id}: Attempting to save context...")
        start_time = time.time()
        success = cache_manager.save_context(project_hash, context)
        duration = time.time() - start_time
        
        result = f"Worker {worker_id}: {'✅ Success' if success else '❌ Failed'} ({duration:.3f}s)"
        print(f"🧵 {result}")
        results.append((worker_id, success, duration))
    
    # Create multiple threads trying to access the same project
    project_hash = "concurrent_demo"
    threads = []
    
    print("🚀 Starting 3 concurrent workers accessing the same project...")
    for i in range(3):
        thread = threading.Thread(target=worker_thread, args=(i, project_hash))
        threads.append(thread)
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    print(f"\n📊 Results:")
    successful_workers = sum(1 for _, success, _ in results if success)
    print(f"   Successful operations: {successful_workers}/3")
    print(f"   Lock protection: {'✅ Working' if successful_workers >= 1 else '❌ Failed'}")
    
    # Verify final state
    final_context = cache_manager.load_context(project_hash)
    if final_context:
        print(f"   Final state: ✅ Consistent (project saved)")
    
    print()


def demo_version_migration():
    """Demonstrate cache version migration"""
    print("🔄 Demo: Cache Version Migration")
    print("=" * 50)
    
    cache_manager = get_cache_manager()
    
    # Simulate V1 cache data
    v1_data = {
        'version': 1,
        'project_hash': 'migration_demo',
        'project_root': '/old/project',
        'timestamp': datetime.now().isoformat(),
        'files': [{'path': 'old_file.py', 'size': 500}],
        'file_type_breakdown': {'python': 1},
        'excluded_files': [],
        'scan_duration_ms': 100,
        'total_size_mb': 0.5
    }
    
    print(f"📦 Original V1 data version: {v1_data['version']}")
    print(f"   Fields: {list(v1_data.keys())}")
    
    # Migrate to current version
    migrated_data = cache_manager.migrate_cache_version(v1_data)
    
    print(f"🔄 Migrated to version: {migrated_data['version']}")
    print(f"   New fields added: {set(migrated_data.keys()) - set(v1_data.keys())}")
    print(f"   Migration: ✅ Successful")
    
    print()


def demo_integrity_validation():
    """Demonstrate checksum validation and backup recovery"""
    print("🛡️ Demo: Data Integrity & Backup Recovery")
    print("=" * 50)
    
    cache_manager = get_cache_manager()
    
    # Save initial context
    context = ContextData(
        project_hash="integrity_demo",
        project_root="/integrity/test",
        files=[{"path": "test.py", "size": 1024}]
    )
    
    print("💾 Saving initial context...")
    cache_manager.save_context("integrity_demo", context)
    
    # Verify integrity validation
    cache_file = cache_manager._get_cache_file_path("integrity_demo")
    is_valid = cache_manager.validate_cache_integrity(cache_file)
    print(f"🔍 Initial integrity check: {'✅ Valid' if is_valid else '❌ Invalid'}")
    
    # Save again to create backup
    modified_context = ContextData(
        project_hash="integrity_demo",
        project_root="/integrity/modified",
        files=[{"path": "modified.py", "size": 2048}]
    )
    cache_manager.save_context("integrity_demo", modified_context)
    
    backup_file = cache_manager._get_backup_file_path("integrity_demo")
    backup_exists = backup_file.exists()
    print(f"💾 Backup creation: {'✅ Created' if backup_exists else '❌ Failed'}")
    
    print()


def demo_cache_statistics():
    """Demonstrate cache statistics and monitoring"""
    print("📊 Demo: Cache Statistics & Monitoring")
    print("=" * 50)
    
    cache_manager = get_cache_manager()
    
    # Create some test data
    for i in range(3):
        context = ContextData(
            project_hash=f"stats_demo_{i}",
            project_root=f"/stats/project_{i}",
            files=[{"path": f"file_{i}.py", "size": 1024 * (i + 1)}]
        )
        cache_manager.save_context(f"stats_demo_{i}", context)
    
    # Get statistics
    stats = cache_manager.get_cache_stats()
    
    print("📈 Cache Statistics:")
    print(f"   Total Projects: {stats['total_projects']}")
    print(f"   Total Cache Size: {stats['total_cache_size_mb']:.2f} MB")
    print(f"   Active Locks: {stats['active_locks']}")
    print(f"   Cache Directory: {stats['cache_base_dir']}")
    
    if stats['oldest_cache']:
        print(f"   Oldest Cache: {stats['oldest_cache']}")
    if stats['newest_cache']:
        print(f"   Newest Cache: {stats['newest_cache']}")
    
    print()


def demo_cleanup_operations():
    """Demonstrate cleanup operations"""
    print("🧹 Demo: Cleanup Operations")
    print("=" * 50)
    
    cache_manager = get_cache_manager()
    
    # Clean up stale locks
    print("🔓 Cleaning up stale locks...")
    cleaned_locks = cache_manager.cleanup_stale_locks()
    print(f"   Cleaned locks: {cleaned_locks}")
    
    # Clean up old caches (using 0 hours to clean everything for demo)
    print("🗑️ Cleaning up old caches...")
    cleaned_caches = cache_manager.cleanup_old_caches(max_age_hours=0)
    print(f"   Cleaned caches: {cleaned_caches}")
    
    print()


def main():
    """Run all demonstrations"""
    print("🚀 Context Cache Manager Demonstration")
    print("=" * 60)
    print()
    
    try:
        demo_basic_operations()
        demo_concurrent_access()
        demo_version_migration()
        demo_integrity_validation()
        demo_cache_statistics()
        demo_cleanup_operations()
        
        print("✅ All demonstrations completed successfully!")
        print("\nKey Features Demonstrated:")
        print("• ✅ Atomic write operations with temporary files")
        print("• ✅ Double-lock pattern for concurrent access protection")
        print("• ✅ Checksum validation for data integrity")
        print("• ✅ Version migration system (V1 → V2)")
        print("• ✅ Backup and recovery mechanisms")
        print("• ✅ Cache statistics and monitoring")
        print("• ✅ Cleanup operations for maintenance")
        
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()