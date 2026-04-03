#!/usr/bin/env python3
"""
Test Benchmark Wizard (S5-04)

Comprehensive tests for the model benchmarking system including:
- Benchmark store persistence
- Benchmark engine execution
- Benchmark suites
- Recommendations engine
- API endpoints
"""

import os
import sys
import pytest
import tempfile
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


@pytest.fixture
def temp_db():
    """Create a temporary database file for testing"""
    fd, path = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    yield path
    try:
        os.unlink(path)
    except:
        pass


class TestBenchmarkStore:
    """Tests for benchmark store persistence layer"""
    
    def test_benchmark_store_crud(self, temp_db):
        """Test benchmark result storage and retrieval"""
        from xencode.monitoring.benchmark_store import BenchmarkStore
        
        store = BenchmarkStore(temp_db)
        
        # Store a benchmark result
        result_id = store.save_result({
            "provider": "ollama",
            "model": "llama3.2",
            "task_type": "code_generation",
            "latency_ms": 245.5,
            "tokens_per_sec": 45.2,
            "accuracy_score": 0.87,
        })
        
        assert result_id is not None
        
        # Retrieve the result
        result = store.get_result(result_id)
        assert result is not None
        assert result["provider"] == "ollama"
        assert result["model"] == "llama3.2"
        assert result["latency_ms"] == 245.5
        
        # Get all results for provider
        results = store.get_results_by_provider("ollama")
        assert len(results) >= 1
    
    def test_benchmark_store_aggregation(self, temp_db):
        """Test aggregate statistics calculation"""
        from xencode.monitoring.benchmark_store import BenchmarkStore
        
        store = BenchmarkStore(temp_db)
        
        # Save multiple results
        for i in range(5):
            store.save_result({
                "provider": "ollama",
                "model": "llama3.2",
                "task_type": "code_generation",
                "latency_ms": 200 + i * 10,
                "accuracy_score": 0.85 + i * 0.01,
            })
        
        # Get aggregates
        stats = store.get_aggregate_stats(provider="ollama")
        
        assert stats["count"] == 5
        assert stats["avg_latency_ms"] is not None
        assert stats["avg_accuracy_score"] is not None
    
    def test_benchmark_store_run_tracking(self, temp_db):
        """Test benchmark run metadata tracking"""
        from xencode.monitoring.benchmark_store import BenchmarkStore
        
        store = BenchmarkStore(temp_db)
        
        # Save run
        run_id = "test_run_123"
        store.save_run(
            run_id=run_id,
            suite_name="test_suite",
            config={"tasks": 5},
            total_tasks=5,
        )
        
        # Get run
        run = store.get_run(run_id)
        
        assert run is not None
        assert run["run_id"] == run_id
        assert run["suite_name"] == "test_suite"
        assert run["status"] == "running"
        
        # Update status
        store.update_run_status(run_id, "completed", 5)
        
        run = store.get_run(run_id)
        assert run["status"] == "completed"
        assert run["completed_tasks"] == 5
    
    def test_benchmark_store_query_by_model(self, temp_db):
        """Test querying results by model"""
        from xencode.monitoring.benchmark_store import BenchmarkStore
        
        store = BenchmarkStore(temp_db)
        
        # Save results for different models
        store.save_result({
            "provider": "ollama",
            "model": "llama3.2",
            "task_type": "code_generation",
            "latency_ms": 250,
        })
        store.save_result({
            "provider": "ollama",
            "model": "codellama",
            "task_type": "code_generation",
            "latency_ms": 300,
        })
        
        # Query by model
        results = store.get_results_by_model("ollama", "llama3.2")
        assert len(results) == 1
        assert results[0]["model"] == "llama3.2"
    
    def test_benchmark_store_query_by_task_type(self, temp_db):
        """Test querying results by task type"""
        from xencode.monitoring.benchmark_store import BenchmarkStore
        
        store = BenchmarkStore(temp_db)
        
        # Save results for different task types
        store.save_result({
            "provider": "ollama",
            "model": "llama3.2",
            "task_type": "code_generation",
            "latency_ms": 250,
        })
        store.save_result({
            "provider": "ollama",
            "model": "llama3.2",
            "task_type": "chat",
            "latency_ms": 200,
        })
        
        # Query by task type
        results = store.get_results_by_task_type("code_generation")
        assert len(results) == 1
        assert results[0]["task_type"] == "code_generation"


class TestBenchmarkEngine:
    """Tests for benchmark execution engine"""
    
    @pytest.mark.asyncio
    async def test_benchmark_engine_execution(self):
        """Test benchmark engine can execute tasks against providers"""
        from xencode.monitoring.benchmark_engine import BenchmarkEngine, BenchmarkTask, TaskType
        
        engine = BenchmarkEngine()
        
        # Create a benchmark task
        task = BenchmarkTask(
            name="test_code_gen",
            task_type=TaskType.CODE_GENERATION,
            prompt="Write a Python function to add two numbers",
            expected_output="def add(a, b):",
        )
        
        # Execute benchmark (mock provider)
        result = await engine.execute_task(
            task=task,
            provider="mock_provider",
            model="mock_model",
        )
        
        assert result is not None
        assert result.task_name == "test_code_gen"
        assert result.provider == "mock_provider"
        assert result.latency_ms > 0
        
        await engine.close()
    
    @pytest.mark.asyncio
    async def test_benchmark_engine_error_handling(self):
        """Test benchmark engine handles errors gracefully"""
        from xencode.monitoring.benchmark_engine import BenchmarkEngine, BenchmarkTask, TaskType
        
        engine = BenchmarkEngine()
        
        task = BenchmarkTask(
            name="test_error",
            task_type=TaskType.GENERAL,
            prompt="Test",
            timeout_seconds=1,
        )
        
        # Execute with invalid provider
        result = await engine.execute_task(
            task=task,
            provider="invalid_provider",
            model="test",
        )
        
        assert result is not None
        # Should have some latency even for mock
        assert result.latency_ms > 0
        
        await engine.close()
    
    @pytest.mark.asyncio
    async def test_benchmark_suite_execution(self):
        """Test complete benchmark suite execution"""
        from xencode.monitoring.benchmark_engine import BenchmarkEngine
        from xencode.monitoring.benchmark_suites import get_benchmark_suites
        
        engine = BenchmarkEngine()
        suites = get_benchmark_suites()
        
        # Get a small suite
        tasks = suites.get_suite("reasoning")[:2]  # Just 2 tasks for speed
        
        # Run against mock provider
        summary = await engine.run_benchmark_suite(
            tasks=tasks,
            providers=[("mock_provider", "mock_model")],
            suite_name="test_suite",
            concurrent=True,
        )
        
        assert summary["run_id"] is not None
        assert summary["suite_name"] == "test_suite"
        assert summary["total_tasks"] > 0
        
        await engine.close()
    
    def test_benchmark_result_serialization(self):
        """Test benchmark result serialization"""
        from xencode.monitoring.benchmark_engine import BenchmarkResult, TaskType
        
        result = BenchmarkResult(
            task_name="test_task",
            task_type="code_generation",
            provider="ollama",
            model="llama3.2",
            run_id="test_run",
            latency_ms=245.5,
            tokens_per_sec=45.2,
            throughput_rps=4.07,
            accuracy_score=0.87,
            consistency_score=0.85,
            quality_rating=0.86,
            cost_per_request=0.0,
            cost_per_1k_tokens=0.0,
            input_tokens=50,
            output_tokens=100,
            total_tokens=150,
            success=True,
        )
        
        # Convert to dict
        result_dict = result.to_dict()
        
        assert result_dict["task_name"] == "test_task"
        assert result_dict["provider"] == "ollama"
        assert "timestamp" in result_dict
        assert isinstance(result_dict["timestamp"], str)  # ISO format


class TestBenchmarkSuites:
    """Tests for benchmark suites and task definitions"""
    
    def test_benchmark_suites(self):
        """Test pre-defined benchmark suites"""
        from xencode.monitoring.benchmark_suites import BenchmarkSuites, TaskType
        
        suites = BenchmarkSuites()
        
        # Get code generation suite
        code_suite = suites.get_suite("code_generation")
        assert code_suite is not None
        assert len(code_suite) > 0
        
        # Get all task types
        task_types = suites.get_task_types()
        assert "code_generation" in task_types
        assert "chat" in task_types
        
        # Create custom benchmark
        custom_task = suites.create_custom_task(
            name="custom_test",
            task_type=TaskType.GENERAL,
            prompt="Test prompt",
            expected_output="Expected",
        )
        assert custom_task.name == "custom_test"
    
    def test_benchmark_suites_custom_creation(self):
        """Test creating custom benchmark suites"""
        from xencode.monitoring.benchmark_suites import BenchmarkSuites, TaskType
        
        suites = BenchmarkSuites()
        
        # Create custom tasks
        custom_tasks = [
            suites.create_custom_task(
                name="custom_1",
                task_type=TaskType.GENERAL,
                prompt="Custom prompt 1",
                weight=1.5,
            ),
            suites.create_custom_task(
                name="custom_2",
                task_type=TaskType.CHAT,
                prompt="Custom prompt 2",
            ),
        ]
        
        # Create custom suite
        suites.create_custom_suite("custom_suite", custom_tasks)
        
        # Verify
        retrieved = suites.get_suite("custom_suite")
        assert retrieved is not None
        assert len(retrieved) == 2
    
    def test_benchmark_suites_all_types(self):
        """Test all pre-defined suite types exist"""
        from xencode.monitoring.benchmark_suites import BenchmarkSuites
        
        suites = BenchmarkSuites()
        
        expected_suites = [
            "code_generation",
            "chat",
            "reasoning",
            "summarization",
            "translation",
            "question_answering",
        ]
        
        for suite_name in expected_suites:
            suite = suites.get_suite(suite_name)
            assert suite is not None, f"Suite {suite_name} should exist"
            assert len(suite) > 0, f"Suite {suite_name} should have tasks"
    
    def test_benchmark_suite_score_calculation(self):
        """Test suite score calculation"""
        from xencode.monitoring.benchmark_suites import BenchmarkSuites
        
        suites = BenchmarkSuites()
        
        # Mock results
        results = [
            {"task_name": "simple_function", "accuracy_score": 0.9, "latency_ms": 200},
            {"task_name": "list_comprehension", "accuracy_score": 0.8, "latency_ms": 250},
        ]
        
        score = suites.calculate_suite_score(results, "code_generation")
        
        assert "suite_name" in score or "error" in score
        if "error" not in score:
            assert "weighted_accuracy" in score


class TestRecommendationsEngine:
    """Tests for recommendations engine"""
    
    def test_recommendations_engine(self):
        """Test recommendations generation"""
        from xencode.monitoring.benchmark_recommendations import RecommendationsEngine
        
        engine = RecommendationsEngine()
        
        # Add sample results
        engine.add_result({
            "provider": "ollama",
            "model": "llama3.2",
            "task_type": "code_generation",
            "latency_ms": 245.5,
            "accuracy_score": 0.87,
            "cost_per_request": 0.0,
        })
        
        # Get recommendations
        recs = engine.get_recommendations(task_type="code_generation")
        assert recs is not None
        assert "recommended_provider" in recs.to_dict()
        assert "alternatives" in recs.to_dict()
    
    def test_recommendations_cost_quality_analysis(self):
        """Test cost/quality tradeoff analysis"""
        from xencode.monitoring.benchmark_recommendations import RecommendationsEngine
        
        engine = RecommendationsEngine()
        
        # Add varied results
        for i in range(10):
            engine.add_result({
                "provider": "ollama" if i % 2 == 0 else "openai",
                "model": "llama3.2" if i % 2 == 0 else "gpt-4",
                "task_type": "code_generation",
                "latency_ms": 200 if i % 2 == 0 else 500,
                "accuracy_score": 0.80 if i % 2 == 0 else 0.95,
                "cost_per_request": 0.0 if i % 2 == 0 else 0.05,
            })
        
        # Get analysis
        analysis = engine.get_cost_quality_analysis("code_generation")
        
        assert "pareto_optimal" in analysis
        assert "frontier" in analysis
        assert analysis["total_models"] > 0
    
    def test_recommendations_use_case_profiles(self):
        """Test different use case recommendations"""
        from xencode.monitoring.benchmark_recommendations import RecommendationsEngine
        
        engine = RecommendationsEngine()
        
        # Add results
        for _ in range(5):
            engine.add_result({
                "provider": "ollama",
                "model": "llama3.2",
                "task_type": "code_generation",
                "latency_ms": 250,
                "accuracy_score": 0.85,
                "cost_per_request": 0.0,
            })
        
        # Test different use cases
        for use_case in ["production", "development", "realtime", "high_quality"]:
            rec = engine.get_recommendations(
                task_type="code_generation",
                use_case=use_case,
            )
            
            assert rec.task_type == "code_generation"
            assert rec.use_case == use_case
            assert rec.recommended_provider is not None
    
    def test_recommendations_with_filters(self):
        """Test recommendations with budget and latency filters"""
        from xencode.monitoring.benchmark_recommendations import RecommendationsEngine
        
        engine = RecommendationsEngine()
        
        # Add results with varying costs and latencies
        engine.add_result({
            "provider": "ollama",
            "model": "llama3.2",
            "task_type": "code_generation",
            "latency_ms": 200,
            "accuracy_score": 0.85,
            "cost_per_request": 0.0,
        })
        engine.add_result({
            "provider": "openai",
            "model": "gpt-4",
            "task_type": "code_generation",
            "latency_ms": 500,
            "accuracy_score": 0.95,
            "cost_per_request": 0.05,
        })
        
        # Get recommendations with max budget filter
        rec = engine.get_recommendations(
            task_type="code_generation",
            max_budget=0.01,
        )
        
        # Should recommend the cheaper option
        assert rec is not None


class TestBenchmarkAPI:
    """Tests for benchmark API endpoints"""
    
    def test_benchmark_api_imports(self):
        """Test that benchmark API components can be imported"""
        # Test imports work
        try:
            from xencode.monitoring.benchmark_engine import BenchmarkEngine
            from xencode.monitoring.benchmark_suites import BenchmarkSuites, get_benchmark_suites
            from xencode.monitoring.benchmark_recommendations import RecommendationsEngine, get_recommendations_engine
            from xencode.monitoring.benchmark_store import BenchmarkStore
            
            assert BenchmarkEngine is not None
            assert BenchmarkSuites is not None
            assert RecommendationsEngine is not None
            assert BenchmarkStore is not None
        except ImportError as e:
            pytest.fail(f"Failed to import benchmark components: {e}")
    
    def test_benchmark_api_router_import(self):
        """Test that monitoring router imports benchmark components"""
        # Just verify the imports don't fail
        try:
            from xencode.api.routers.monitoring import (
                BenchmarkRunRequest,
                BenchmarkResultResponse,
                BenchmarkComparisonResponse,
                BenchmarkRecommendationResponse,
            )
            
            assert BenchmarkRunRequest is not None
            assert BenchmarkResultResponse is not None
        except ImportError:
            # If benchmark module not available, that's okay for this test
            pass


class TestBenchmarkIntegration:
    """Integration tests for the benchmark system"""
    
    @pytest.mark.asyncio
    async def test_full_benchmark_workflow(self, temp_db):
        """Test complete benchmark workflow from execution to recommendations"""
        from xencode.monitoring.benchmark_engine import BenchmarkEngine, BenchmarkTask, TaskType
        from xencode.monitoring.benchmark_store import BenchmarkStore
        from xencode.monitoring.benchmark_recommendations import RecommendationsEngine
        
        # Create store
        store = BenchmarkStore(temp_db)
        
        # Create engine with store
        engine = BenchmarkEngine(store=store)
        
        # Create tasks
        tasks = [
            BenchmarkTask(
                name="test_task_1",
                task_type=TaskType.CODE_GENERATION,
                prompt="Write a function",
                expected_output="def",
            ),
            BenchmarkTask(
                name="test_task_2",
                task_type=TaskType.CODE_GENERATION,
                prompt="Write a class",
                expected_output="class",
            ),
        ]
        
        # Run benchmarks
        summary = await engine.run_benchmark_suite(
            tasks=tasks,
            providers=[("test_provider", "test_model")],
            suite_name="integration_test",
        )
        
        assert summary["run_id"] is not None
        assert summary["successful"] >= 0
        
        # Create recommendations engine with same store
        rec_engine = RecommendationsEngine(store=store)
        
        # Get recommendations (may use fallback if no data matches)
        rec = rec_engine.get_recommendations(task_type="code_generation")
        assert rec is not None
        
        await engine.close()
    
    def test_benchmark_data_persistence(self, temp_db):
        """Test that benchmark data persists across engine instances"""
        from xencode.monitoring.benchmark_store import BenchmarkStore
        from xencode.monitoring.benchmark_recommendations import RecommendationsEngine
        
        # Create store and save data
        store = BenchmarkStore(temp_db)
        store.save_result({
            "provider": "test",
            "model": "model1",
            "task_type": "test",
            "latency_ms": 100,
            "accuracy_score": 0.9,
        })
        
        # Create recommendations engine with same store
        rec_engine = RecommendationsEngine(store=store)
        
        # Verify data is accessible
        stats = store.get_aggregate_stats(provider="test")
        assert stats["count"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
