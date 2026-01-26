"""
Performance regression tests for Xencode
Tests to ensure performance doesn't degrade over time
"""
import time
import statistics
import tempfile
from pathlib import Path
import pytest
from xencode.core.files import create_file, read_file, write_file, delete_file
from xencode.core.memory import ConversationMemory
from xencode.core.cache import ResponseCache
from xencode.core.models import ModelManager
from xencode.core.performance_benchmarks import run_simple_benchmark, BenchmarkResult


class TestPerformanceRegression:
    """Performance regression tests to catch performance degradations"""
    
    def test_file_operations_performance_baseline(self, tmp_path):
        """Baseline performance test for file operations"""
        # Define acceptable performance thresholds (seconds)
        CREATE_THRESHOLD = 0.1  # 100ms
        READ_THRESHOLD = 0.1    # 100ms
        
        file_path = tmp_path / "perf_test.txt"
        content = "This is test content for performance evaluation." * 100  # Larger content
        
        # Test file creation performance
        start_time = time.time()
        create_file(str(file_path), content)
        create_time = time.time() - start_time
        
        assert create_time <= CREATE_THRESHOLD, f"File creation took {create_time}s, exceeds threshold {CREATE_THRESHOLD}s"
        
        # Test file reading performance
        start_time = time.time()
        read_file(str(file_path))
        read_time = time.time() - start_time
        
        assert read_time <= READ_THRESHOLD, f"File reading took {read_time}s, exceeds threshold {READ_THRESHOLD}s"
    
    def test_memory_operations_performance_baseline(self):
        """Baseline performance test for memory operations"""
        # Define acceptable performance thresholds (seconds)
        ADD_MESSAGE_THRESHOLD = 0.01  # 10ms per message
        GET_CONTEXT_THRESHOLD = 0.005  # 5ms
        
        memory = ConversationMemory()
        
        # Test adding multiple messages
        num_messages = 50
        start_time = time.time()
        for i in range(num_messages):
            memory.add_message("user", f"Test message {i}", f"model_{i}")
        add_time = time.time() - start_time
        
        avg_add_time = add_time / num_messages
        assert avg_add_time <= ADD_MESSAGE_THRESHOLD, f"Average message addition took {avg_add_time}s, exceeds threshold {ADD_MESSAGE_THRESHOLD}s"
        
        # Test getting context
        start_time = time.time()
        context = memory.get_context()
        get_context_time = time.time() - start_time
        
        assert get_context_time <= GET_CONTEXT_THRESHOLD, f"Getting context took {get_context_time}s, exceeds threshold {GET_CONTEXT_THRESHOLD}s"
        assert len(context) == num_messages
    
    def test_cache_operations_performance_baseline(self, tmp_path):
        """Baseline performance test for cache operations"""
        # Define acceptable performance thresholds (seconds)
        SET_THRESHOLD = 0.01  # 10ms
        GET_THRESHOLD = 0.01  # 10ms
        
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = ResponseCache(cache_dir=Path(temp_dir))
            
            # Test cache set performance
            start_time = time.time()
            cache.set("test_prompt", "test_model", "test_response")
            set_time = time.time() - start_time
            
            assert set_time <= SET_THRESHOLD, f"Cache set took {set_time}s, exceeds threshold {SET_THRESHOLD}s"
            
            # Test cache get performance
            start_time = time.time()
            result = cache.get("test_prompt", "test_model")
            get_time = time.time() - start_time
            
            assert get_time <= GET_THRESHOLD, f"Cache get took {get_time}s, exceeds threshold {GET_THRESHOLD}s"
            assert result == "test_response"
    
    def test_cache_bulk_operations_performance(self, tmp_path):
        """Test performance of bulk cache operations"""
        # Define acceptable performance thresholds
        OPERATIONS_PER_SECOND = 100  # Expect at least 100 operations per second
        
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = ResponseCache(cache_dir=Path(temp_dir))
            
            num_operations = 50
            start_time = time.time()
            
            # Perform multiple set/get operations
            for i in range(num_operations):
                cache.set(f"prompt_{i}", f"model_{i}", f"response_{i}")
                result = cache.get(f"prompt_{i}", f"model_{i}")
                assert result == f"response_{i}"
            
            total_time = time.time() - start_time
            operations_per_second = (num_operations * 2) / total_time  # 2 operations per iteration (set + get)
            
            assert operations_per_second >= OPERATIONS_PER_SECOND, f"Performance: {operations_per_second:.2f} ops/sec, below threshold {OPERATIONS_PER_SECOND}"
    
    def test_memory_large_context_performance(self):
        """Test performance with large memory contexts"""
        # Define acceptable performance thresholds
        ADD_THRESHOLD = 0.05  # 50ms for adding to large context
        GET_THRESHOLD = 0.02  # 20ms for getting large context
        
        memory = ConversationMemory(max_items=1000)  # Large capacity
        
        # Fill memory with many messages
        for i in range(500):
            memory.add_message("user", f"Message {i} " + "x" * 50, f"model_{i % 10}")
        
        # Test adding to a large context
        start_time = time.time()
        memory.add_message("assistant", "Final response", "final_model")
        add_time = time.time() - start_time
        
        assert add_time <= ADD_THRESHOLD, f"Adding to large context took {add_time}s, exceeds threshold {ADD_THRESHOLD}s"
        
        # Test getting a large context
        start_time = time.time()
        context = memory.get_context(max_messages=100)  # Get last 100 messages
        get_time = time.time() - start_time
        
        assert get_time <= GET_THRESHOLD, f"Getting large context took {get_time}s, exceeds threshold {GET_THRESHOLD}s"
        assert len(context) == 100
    
    def test_concurrent_cache_performance(self, tmp_path):
        """Test cache performance under simulated concurrent access"""
        import threading
        import queue
        
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = ResponseCache(cache_dir=Path(temp_dir))
            
            # Define performance threshold
            TOTAL_TIME_THRESHOLD = 2.0  # 2 seconds for all operations
            
            def worker(worker_id, num_ops, result_queue):
                start_time = time.time()
                for i in range(num_ops):
                    prompt = f"worker_{worker_id}_prompt_{i}"
                    model = f"model_{worker_id}"
                    response = f"response_{worker_id}_{i}"
                    
                    cache.set(prompt, model, response)
                    result = cache.get(prompt, model)
                    assert result == response
                elapsed = time.time() - start_time
                result_queue.put(elapsed)
            
            # Run multiple threads concurrently
            num_threads = 5
            ops_per_thread = 10
            result_queue = queue.Queue()
            
            threads = []
            for i in range(num_threads):
                thread = threading.Thread(target=worker, args=(i, ops_per_thread, result_queue))
                threads.append(thread)
            
            start_time = time.time()
            for thread in threads:
                thread.start()
            
            for thread in threads:
                thread.join()
            
            total_time = time.time() - start_time
            
            # Collect individual times
            individual_times = []
            while not result_queue.empty():
                individual_times.append(result_queue.get())
            
            assert total_time <= TOTAL_TIME_THRESHOLD, f"Concurrent cache operations took {total_time}s, exceeds threshold {TOTAL_TIME_THRESHOLD}s"
    
    def test_cache_invalidation_performance(self, tmp_path):
        """Test performance of cache invalidation operations"""
        # Define acceptable performance thresholds
        INVALIDATION_THRESHOLD = 0.5  # 500ms for invalidation operations
        
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = ResponseCache(cache_dir=Path(temp_dir), max_size=100)
            
            # Fill cache with many entries
            num_entries = 50
            for i in range(num_entries):
                cache.set(f"prompt_{i}", f"model_A", f"response_{i}")
                cache.set(f"prompt_{i}_2", f"model_B", f"response_{i}_2")
            
            # Test invalidation by model
            start_time = time.time()
            invalidated_count = cache.invalidate_by_model("model_A")
            invalidation_time = time.time() - start_time
            
            assert invalidation_time <= INVALIDATION_THRESHOLD, f"Invalidation by model took {invalidation_time}s, exceeds threshold {INVALIDATION_THRESHOLD}s"
            assert invalidated_count == 50  # Should invalidate 50 entries
    
    def test_memory_session_switching_performance(self):
        """Test performance of memory session switching"""
        # Define acceptable performance thresholds
        SWITCH_THRESHOLD = 0.005  # 5ms per switch
        
        memory = ConversationMemory()
        
        # Create multiple sessions
        num_sessions = 10
        for i in range(num_sessions):
            session_id = f"session_{i}"
            memory.start_session(session_id)
            for j in range(5):
                memory.add_message("user", f"Message {j} in session {i}", "test_model")
        
        # Test switching between sessions
        start_time = time.time()
        for i in range(num_sessions):
            session_id = f"session_{i}"
            result = memory.switch_session(session_id)
            assert result is True, f"Failed to switch to session {session_id}"
            
            # Verify we're in the right session
            context = memory.get_context()
            assert len(context) == 5  # Should have 5 messages
        
        total_time = time.time() - start_time
        avg_switch_time = total_time / num_sessions
        
        assert avg_switch_time <= SWITCH_THRESHOLD, f"Average session switch took {avg_switch_time}s, exceeds threshold {SWITCH_THRESHOLD}s"


class TestBenchmarkComparison:
    """Tests that compare current performance to known baselines"""
    
    def test_cache_performance_against_baseline(self, tmp_path):
        """Compare cache performance to known baseline"""
        # Known baseline from previous measurements (these would be updated as needed)
        BASELINE_SET_TIME = 0.005  # seconds
        BASELINE_GET_TIME = 0.003  # seconds
        
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = ResponseCache(cache_dir=Path(temp_dir))
            
            # Measure current performance
            num_iterations = 20
            set_times = []
            get_times = []
            
            for i in range(num_iterations):
                # Measure set time
                start = time.time()
                cache.set(f"prompt_{i}", "model", f"response_{i}")
                set_times.append(time.time() - start)
                
                # Measure get time
                start = time.time()
                cache.get(f"prompt_{i}", "model")
                get_times.append(time.time() - start)
            
            avg_set_time = statistics.mean(set_times)
            avg_get_time = statistics.mean(get_times)
            
            # Check against baselines (allowing for some variance)
            REGRESSION_THRESHOLD = 2.0  # Allow up to 2x slower
            
            assert avg_set_time <= BASELINE_SET_TIME * REGRESSION_THRESHOLD, \
                f"Cache set performance regressed: {avg_set_time:.4f}s vs baseline {BASELINE_SET_TIME:.4f}s"
            
            assert avg_get_time <= BASELINE_GET_TIME * REGRESSION_THRESHOLD, \
                f"Cache get performance regressed: {avg_get_time:.4f}s vs baseline {BASELINE_GET_TIME:.4f}s"
    
    def test_memory_performance_against_baseline(self):
        """Compare memory performance to known baseline"""
        # Known baseline from previous measurements
        BASELINE_ADD_TIME = 0.002  # seconds
        BASELINE_GET_TIME = 0.001  # seconds
        
        memory = ConversationMemory()
        
        # Measure current performance
        num_iterations = 50
        add_times = []
        get_times = []
        
        for i in range(num_iterations):
            # Measure add time
            start = time.time()
            memory.add_message("user", f"Message {i}", "model")
            add_times.append(time.time() - start)
            
            # Measure get time
            start = time.time()
            memory.get_context()
            get_times.append(time.time() - start)
        
        # Exclude outliers for fair comparison
        if len(add_times) > 10:
            add_times = sorted(add_times)[len(add_times)//10:-len(add_times)//10]  # Remove top and bottom 10%
        if len(get_times) > 10:
            get_times = sorted(get_times)[len(get_times)//10:-len(get_times)//10]
        
        avg_add_time = statistics.mean(add_times) if add_times else 0
        avg_get_time = statistics.mean(get_times) if get_times else 0
        
        # Check against baselines (allowing for some variance)
        REGRESSION_THRESHOLD = 2.0  # Allow up to 2x slower
        
        assert avg_add_time <= BASELINE_ADD_TIME * REGRESSION_THRESHOLD, \
            f"Memory add performance regressed: {avg_add_time:.4f}s vs baseline {BASELINE_ADD_TIME:.4f}s"
        
        assert avg_get_time <= BASELINE_GET_TIME * REGRESSION_THRESHOLD, \
            f"Memory get performance regressed: {avg_get_time:.4f}s vs baseline {BASELINE_GET_TIME:.4f}s"


def test_performance_with_pytest_benchmark(benchmark):
    """Test performance using pytest-benchmark if available"""
    with tempfile.TemporaryDirectory() as temp_dir:
        cache = ResponseCache(cache_dir=Path(temp_dir))
        
        def cache_operation():
            cache.set("benchmark_prompt", "benchmark_model", "benchmark_response")
            result = cache.get("benchmark_prompt", "benchmark_model")
            return result
        
        # This would be measured by pytest-benchmark if available
        result = benchmark(cache_operation)
        assert result == "benchmark_response"


if __name__ == "__main__":
    # Run the performance regression tests
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    test_instance = TestPerformanceRegression()
    
    # Run a few tests manually to demonstrate
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_path_str:
        tmp_path = Path(tmp_path_str)
        test_instance.test_file_operations_performance_baseline(tmp_path)
        print("File operations performance test passed")
        
        test_instance.test_cache_operations_performance_baseline(tmp_path)
        print("Cache operations performance test passed")
    
    test_instance.test_memory_operations_performance_baseline()
    print("Memory operations performance test passed")
    
    print("All performance regression tests passed!")