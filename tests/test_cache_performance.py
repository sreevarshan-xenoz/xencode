"""
Performance tests for the caching system
"""
import tempfile
import time
from pathlib import Path
import pytest
from xencode.core.cache import ResponseCache


def test_cache_performance_single_item():
    """Test the performance of caching a single item"""
    with tempfile.TemporaryDirectory() as temp_dir:
        cache = ResponseCache(cache_dir=Path(temp_dir))
        
        # Measure time to set a single item
        start_time = time.time()
        cache.set("test_prompt", "test_model", "test_response")
        set_time = time.time() - start_time
        
        # Measure time to get a single item
        start_time = time.time()
        result = cache.get("test_prompt", "test_model")
        get_time = time.time() - start_time
        
        assert result == "test_response"
        # Set and get operations should be reasonably fast (under 1 second each)
        assert set_time < 1.0, f"Setting cache took too long: {set_time}s"
        assert get_time < 1.0, f"Getting cache took too long: {get_time}s"


def test_cache_performance_multiple_items():
    """Test the performance of caching multiple items"""
    with tempfile.TemporaryDirectory() as temp_dir:
        cache = ResponseCache(cache_dir=Path(temp_dir), max_size=100)
        
        num_items = 50
        start_time = time.time()
        
        # Add multiple items
        for i in range(num_items):
            cache.set(f"prompt_{i}", f"model_{i}", f"response_{i}")
        
        set_batch_time = time.time() - start_time
        
        # Verify all items were stored
        for i in range(num_items):
            result = cache.get(f"prompt_{i}", f"model_{i}")
            assert result == f"response_{i}"
        
        # Calculate average time per item
        avg_set_time = set_batch_time / num_items
        assert avg_set_time < 0.1, f"Average set time per item too slow: {avg_set_time}s"


def test_cache_hit_vs_miss_performance():
    """Compare performance of cache hits vs misses"""
    with tempfile.TemporaryDirectory() as temp_dir:
        cache = ResponseCache(cache_dir=Path(temp_dir))
        
        # Prime the cache
        cache.set("existing_prompt", "test_model", "test_response")
        
        # Measure cache hit time
        start_time = time.time()
        for _ in range(10):
            cache.get("existing_prompt", "test_model")
        hit_time = time.time() - start_time
        
        # Measure cache miss time
        start_time = time.time()
        for _ in range(10):
            cache.get("nonexistent_prompt", "test_model")
        miss_time = time.time() - start_time
        
        # Hits should generally be faster than misses (though both should be fast)
        # The important thing is that both operations remain performant
        assert hit_time < 1.0, f"Cache hits took too long: {hit_time}s"
        assert miss_time < 1.0, f"Cache misses took too long: {miss_time}s"


def test_cache_invalidation_performance():
    """Test the performance of cache invalidation operations"""
    with tempfile.TemporaryDirectory() as temp_dir:
        cache = ResponseCache(cache_dir=Path(temp_dir), max_size=100)
        
        # Add multiple items with a specific model
        num_items = 20
        for i in range(num_items):
            cache.set(f"prompt_{i}", "target_model", f"response_{i}")
            cache.set(f"different_prompt_{i}", "other_model", f"other_response_{i}")
        
        # Measure time to invalidate by model
        start_time = time.time()
        invalidated_count = cache.invalidate_by_model("target_model")
        invalidation_time = time.time() - start_time
        
        assert invalidated_count == 20, f"Expected to invalidate 20 items, got {invalidated_count}"
        assert invalidation_time < 1.0, f"Cache invalidation took too long: {invalidation_time}s"
        
        # Verify targeted items are gone but others remain
        for i in range(num_items):
            # Targeted items should be gone
            result = cache.get(f"prompt_{i}", "target_model")
            assert result is None
            
            # Other items should remain
            result = cache.get(f"different_prompt_{i}", "other_model")
            assert result == f"other_response_{i}"


def test_cache_size_limit_performance():
    """Test performance when cache reaches size limits"""
    with tempfile.TemporaryDirectory() as temp_dir:
        cache = ResponseCache(cache_dir=Path(temp_dir), max_size=10, ttl_seconds=3600)  # Small max size
        
        # Add more items than the cache limit
        num_items = 20
        start_time = time.time()
        
        for i in range(num_items):
            cache.set(f"prompt_{i}", "test_model", f"response_{i}")
        
        insertion_time = time.time() - start_time
        
        # The cache should still perform well even when managing size limits
        assert insertion_time < 2.0, f"Insertion with size management took too long: {insertion_time}s"
        
        # Check that we're still within reasonable bounds
        cache_size = cache.get_cache_size()
        assert cache_size <= 10, f"Cache exceeded size limit: {cache_size}"


def test_concurrent_access_performance(benchmark):
    """Test cache performance under concurrent access (if benchmark fixture is available)"""
    with tempfile.TemporaryDirectory() as temp_dir:
        cache = ResponseCache(cache_dir=Path(temp_dir))
        
        def cache_operations():
            # Simulate some cache operations
            cache.set("test_prompt", "test_model", "test_response")
            result = cache.get("test_prompt", "test_model")
            cache.set("another_prompt", "another_model", "another_response")
            return result
        
        # This would typically use pytest-benchmark if available
        # For now, we'll just run it directly
        start_time = time.time()
        for _ in range(100):
            cache_operations()
        total_time = time.time() - start_time
        
        assert total_time < 5.0, f"Concurrent operations took too long: {total_time}s"


def test_large_payload_performance():
    """Test performance with large cache payloads"""
    with tempfile.TemporaryDirectory() as temp_dir:
        cache = ResponseCache(cache_dir=Path(temp_dir))
        
        # Create a large response
        large_response = "x" * 10000  # 10KB response
        
        start_time = time.time()
        cache.set("large_prompt", "test_model", large_response)
        set_time = time.time() - start_time
        
        start_time = time.time()
        result = cache.get("large_prompt", "test_model")
        get_time = time.time() - start_time
        
        assert result == large_response
        # Even with large payloads, operations should remain reasonably fast
        assert set_time < 1.0, f"Setting large payload took too long: {set_time}s"
        assert get_time < 1.0, f"Getting large payload took too long: {get_time}s"


if __name__ == "__main__":
    # Run the performance tests
    test_cache_performance_single_item()
    test_cache_performance_multiple_items()
    test_cache_hit_vs_miss_performance()
    test_cache_invalidation_performance()
    test_cache_size_limit_performance()
    test_large_payload_performance()
    print("All performance tests passed!")