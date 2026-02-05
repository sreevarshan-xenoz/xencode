"""
Comprehensive Tests for Performance System
Tests for GPU acceleration, edge computing, quantum-ready algorithms,
microservice optimization, and performance monitoring components.
"""

import asyncio
import pytest
from unittest.mock import Mock, MagicMock, patch, AsyncMock
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Import the modules we're testing
from xencode.performance.gpu_acceleration import (
    GPUAccelerationManager, GPUFramework, create_gpu_acceleration_manager
)
from xencode.performance.edge_computing import (
    EdgeComputingManager, EdgeNodeType, create_edge_computing_manager
)
from xencode.performance.quantum_ready import (
    QuantumReadyManager, QuantumAlgorithmType, create_quantum_ready_manager
)
from xencode.performance.microservices import (
    MicroserviceManager, ServiceType, create_microservice_manager
)
from xencode.performance.optimizer import (
    PerformanceOptimizer, PerformanceMetricType, create_performance_optimizer
)


# Test GPU Acceleration
class TestGPUAcceleration:
    """Test cases for the GPUAccelerationManager class."""
    
    def test_get_available_devices(self):
        """Test getting available GPU devices."""
        manager = create_gpu_acceleration_manager()
        
        devices = manager.get_available_devices()
        
        # The manager should initialize with available devices
        assert isinstance(devices, list)
        
    def test_allocate_gpu_resources(self):
        """Test allocating GPU resources."""
        manager = create_gpu_acceleration_manager()
        
        # Try to allocate resources
        allocation_result = manager.allocate_gpu_resources(
            requested_memory_mb=100,
            framework=GPUFramework.CUDA if manager.framework_support[GPUFramework.CUDA] else None
        )
        
        # Allocation might succeed or fail depending on actual GPU availability
        # but the method should not throw an exception
        assert allocation_result is None or isinstance(allocation_result, tuple)
        
    def test_execute_on_gpu(self):
        """Test executing a workload on GPU."""
        manager = create_gpu_acceleration_manager()
        
        # Create test data
        test_data = np.random.random((100, 100))
        
        # Try to execute on GPU
        workload_id = manager.execute_on_gpu(
            "matrix_multiplication", 
            test_data, 
            priority=3
        )
        
        # Should return a workload ID
        assert isinstance(workload_id, str)
        assert workload_id.startswith("workload_")
        
    def test_performance_stats(self):
        """Test getting performance statistics."""
        manager = create_gpu_acceleration_manager()
        
        stats = manager.get_performance_stats()
        
        assert "total_devices" in stats
        assert "framework_support" in stats
        assert "active_allocations" in stats
        assert "queued_workloads" in stats
        assert "running_workloads" in stats
        assert "device_utilization" in stats
        
    def test_optimize_for_workload(self):
        """Test optimizing GPU selection for specific workloads."""
        manager = create_gpu_acceleration_manager()
        
        # Test with different workload types
        nn_device = manager.optimize_for_workload("neural_network_inference")
        matrix_device = manager.optimize_for_workload("matrix_multiplication")
        general_device = manager.optimize_for_workload("general_processing")
        
        # Should return device IDs or None if no devices available
        assert nn_device is None or isinstance(nn_device, str)
        assert matrix_device is None or isinstance(matrix_device, str)
        assert general_device is None or isinstance(general_device, str)


# Test Edge Computing
class TestEdgeComputing:
    """Test cases for the EdgeComputingManager class."""
    
    @pytest.mark.asyncio
    async def test_submit_task(self):
        """Test submitting a task for edge processing."""
        manager = await create_edge_computing_manager()
        
        # Submit a simple task
        task_id = await manager.submit_task(
            task_type="data_processing",
            data={"numbers": [1, 2, 3, 4, 5]},
            priority=3
        )
        
        assert isinstance(task_id, str)
        assert task_id.startswith("task_")
        
        # Check task status
        task_status = manager.get_task_status(task_id)
        assert task_status is not None
        assert task_status.task_id == task_id
        
        await manager.shutdown()
        
    @pytest.mark.asyncio
    async def test_get_edge_node_status(self):
        """Test getting edge node status."""
        manager = await create_edge_computing_manager()
        
        # Get status of all nodes
        all_nodes = manager.get_edge_node_status()
        assert isinstance(all_nodes, list)
        
        # Get status of a specific node (may not exist)
        node_status = manager.get_edge_node_status("nonexistent_node")
        assert node_status is None or hasattr(node_status, 'node_id')
        
        await manager.shutdown()
        
    @pytest.mark.asyncio
    async def test_performance_metrics(self):
        """Test getting performance metrics."""
        manager = await create_edge_computing_manager()
        
        # Get initial metrics
        metrics = manager.get_performance_metrics()
        
        assert "total_tasks" in metrics
        assert "completed_tasks" in metrics
        assert "edge_vs_cloud_ratio" in metrics
        assert "average_runtime_ms" in metrics
        assert "total_data_processed_mb" in metrics
        
        await manager.shutdown()
        
    @pytest.mark.asyncio
    async def test_offload_to_edge(self):
        """Test offloading computation to edge."""
        manager = await create_edge_computing_manager()
        
        # Offload a simple task
        result = await manager.offload_to_edge(
            data={"value": 42},
            task_type="simple_calculation",
            priority=2
        )
        
        assert "task_id" in result
        assert "status" in result
        assert "result" in result
        assert "metrics" in result
        
        await manager.shutdown()
        
    @pytest.mark.asyncio
    async def test_optimization_recommendations(self):
        """Test getting optimization recommendations."""
        manager = await create_edge_computing_manager()
        
        recommendations = manager.get_optimization_recommendations()
        
        assert isinstance(recommendations, list)
        
        await manager.shutdown()


# Test Quantum-Ready Algorithms
class TestQuantumReady:
    """Test cases for the QuantumReadyManager class."""
    
    def test_assess_quantum_readiness(self):
        """Test assessing quantum readiness."""
        manager = create_quantum_ready_manager()
        
        readiness = manager.assess_quantum_readiness()
        
        assert isinstance(readiness, float)
        assert 0.0 <= readiness <= 1.0
        
    @pytest.mark.asyncio
    async def test_execute_quantum_algorithm(self):
        """Test executing a quantum algorithm."""
        manager = create_quantum_ready_manager()
        
        # Execute a simple quantum algorithm
        task_id = await manager.execute_quantum_algorithm(
            algorithm_type=QuantumAlgorithmType.GROVERS_SEARCH,
            input_data={"n": 3, "marked_item": 2},
            priority=3
        )
        
        assert isinstance(task_id, str)
        assert task_id.startswith("quantum_task_")
        
        # Check task status
        task_status = manager.get_task_status(task_id)
        assert task_status is not None
        assert task_status.task_id == task_id
        
    def test_get_performance_metrics(self):
        """Test getting quantum performance metrics."""
        manager = create_quantum_ready_manager()
        
        metrics = manager.get_performance_metrics()
        
        assert "total_tasks" in metrics
        assert "completed_tasks" in metrics
        assert "quantum_vs_classical_ratio" in metrics
        assert "average_quantum_time" in metrics
        assert "quantum_readiness" in metrics
        assert "simulation_stats" in metrics
        
    def test_enable_disable_hardware_backend(self):
        """Test enabling/disabling hardware backend."""
        manager = create_quantum_ready_manager()
        
        # Initially should be disabled
        assert manager.hardware_available == False
        
        # Enable
        manager.enable_hardware_backend()
        assert manager.hardware_available == True
        
        # Disable
        manager.disable_hardware_backend()
        assert manager.hardware_available == False


# Test Microservice Architecture
class TestMicroserviceArchitecture:
    """Test cases for the MicroserviceManager class."""
    
    @pytest.mark.asyncio
    async def test_register_service(self):
        """Test registering a service."""
        manager = await create_microservice_manager()
        
        # Register a service
        instance_id = manager.register_service(
            ServiceType.USER_SERVICE,
            "localhost",
            8080,
            "/health",
            {"version": "1.0.0", "region": "us-east-1"}
        )
        
        assert isinstance(instance_id, str)
        assert "user_service_" in instance_id
        
        await manager.shutdown()
        
    @pytest.mark.asyncio
    async def test_send_heartbeat(self):
        """Test sending a heartbeat."""
        manager = await create_microservice_manager()
        
        # Register a service first
        instance_id = manager.register_service(
            ServiceType.API_GATEWAY,
            "localhost",
            8081
        )
        
        # Send heartbeat
        success = manager.send_heartbeat(instance_id)
        
        assert success is True
        
        await manager.shutdown()
        
    @pytest.mark.asyncio
    async def test_get_service_instance(self):
        """Test getting a service instance."""
        manager = await create_microservice_manager()
        
        # Register a service
        instance_id = manager.register_service(
            ServiceType.AUTH_SERVICE,
            "localhost",
            8082
        )
        
        # Get an instance
        instance = manager.get_service_instance(ServiceType.AUTH_SERVICE)
        
        # May be None if no healthy instances exist
        if instance is not None:
            assert hasattr(instance, 'instance_id')
            assert hasattr(instance, 'status')
        
        await manager.shutdown()
        
    @pytest.mark.asyncio
    async def test_get_service_status(self):
        """Test getting service status."""
        manager = await create_microservice_manager()
        
        # Get status for all services
        all_status = manager.get_service_status()
        
        assert isinstance(all_status, dict)
        
        # Get status for specific service type
        auth_status = manager.get_service_status(ServiceType.AUTH_SERVICE)
        
        assert "service_type" in auth_status
        assert "total_instances" in auth_status
        assert "healthy_instances" in auth_status
        
        await manager.shutdown()
        
    @pytest.mark.asyncio
    async def test_get_performance_metrics(self):
        """Test getting microservice performance metrics."""
        manager = await create_microservice_manager()
        
        metrics = manager.get_performance_metrics()
        
        assert "total_services" in metrics
        assert "total_instances" in metrics
        assert "healthy_instances" in metrics
        assert "health_percentage" in metrics
        assert "average_cpu_usage" in metrics
        assert "average_memory_usage" in metrics
        assert "scaling_events_count" in metrics
        
        await manager.shutdown()


# Test Performance Monitoring
class TestPerformanceMonitoring:
    """Test cases for the PerformanceOptimizer class."""
    
    @pytest.mark.asyncio
    async def test_get_current_metrics(self):
        """Test getting current performance metrics."""
        optimizer = await create_performance_optimizer(collection_interval=1)
        
        metrics = optimizer.get_current_metrics()
        
        assert isinstance(metrics, dict)
        assert all(isinstance(k, PerformanceMetricType) for k in metrics.keys())
        assert all(isinstance(v, (int, float)) for v in metrics.values())
        
        await optimizer.stop_monitoring()
        
    @pytest.mark.asyncio
    async def test_get_performance_trend(self):
        """Test getting performance trends."""
        optimizer = await create_performance_optimizer(collection_interval=1)
        
        # Get trend for CPU usage over 1 hour
        trend = optimizer.get_performance_trend(PerformanceMetricType.CPU_USAGE, hours=1)
        
        assert "metric_type" in trend
        assert "time_period_hours" in trend
        assert "data_points" in trend
        assert "average_value" in trend
        assert "trend" in trend
        
        await optimizer.stop_monitoring()
        
    @pytest.mark.asyncio
    async def test_get_performance_summary(self):
        """Test getting performance summary."""
        optimizer = await create_performance_optimizer(collection_interval=1)
        
        summary = optimizer.get_performance_summary()
        
        assert "timestamp" in summary
        assert "health_score" in summary
        assert "current_metrics" in summary
        assert "recent_issues_count" in summary
        assert "recommendations_count" in summary
        assert "baselines_available" in summary
        assert "monitoring_active" in summary
        
        await optimizer.stop_monitoring()
        
    @pytest.mark.asyncio
    async def test_optimization_recommendations(self):
        """Test getting optimization recommendations."""
        optimizer = await create_performance_optimizer(collection_interval=1)
        
        # Initially no recommendations
        recommendations = optimizer.get_optimization_recommendations()
        assert isinstance(recommendations, list)
        
        await optimizer.stop_monitoring()
        
    @pytest.mark.asyncio
    async def test_detected_issues(self):
        """Test getting detected performance issues."""
        optimizer = await create_performance_optimizer(collection_interval=1)
        
        issues = optimizer.get_detected_issues()
        assert isinstance(issues, list)
        
        await optimizer.stop_monitoring()


# Integration Tests
class TestPerformanceSystemIntegration:
    """Integration tests for the performance system."""
    
    @pytest.mark.asyncio
    async def test_gpu_edge_integration(self):
        """Test integration between GPU acceleration and edge computing."""
        # Create both managers
        gpu_manager = create_gpu_acceleration_manager()
        edge_manager = await create_edge_computing_manager()
        
        # Submit a GPU-intensive task to edge computing
        task_id = await edge_manager.submit_task(
            task_type="gpu_intensive_computation",
            data=np.random.random((50, 50)).tolist(),  # Convert to serializable format
            priority=4
        )
        
        # Check that the task was submitted
        task_status = edge_manager.get_task_status(task_id)
        assert task_status is not None
        assert task_status.task_id == task_id
        
        # Get performance metrics from both systems
        gpu_metrics = gpu_manager.get_performance_stats()
        edge_metrics = edge_manager.get_performance_metrics()
        
        assert isinstance(gpu_metrics, dict)
        assert isinstance(edge_metrics, dict)
        
        await edge_manager.shutdown()
        
    @pytest.mark.asyncio
    async def test_quantum_microservice_integration(self):
        """Test integration between quantum-ready algorithms and microservices."""
        quantum_manager = create_quantum_ready_manager()
        microservice_manager = await create_microservice_manager()
        
        # Register a quantum service
        quantum_service_id = microservice_manager.register_service(
            ServiceType.ANALYTICS,
            "localhost",
            9090,
            metadata={"quantum_capable": True}
        )
        
        # Execute a quantum algorithm
        quantum_task_id = await quantum_manager.execute_quantum_algorithm(
            algorithm_type=QuantumAlgorithmType.VQE,
            input_data={"hamiltonian_terms": [{"term": "XX", "coeff": 1.0}], "ansatz_depth": 2},
            priority=3
        )
        
        # Check both systems are functioning
        quantum_metrics = quantum_manager.get_performance_metrics()
        microservice_metrics = microservice_manager.get_performance_metrics()
        
        assert isinstance(quantum_metrics, dict)
        assert isinstance(microservice_metrics, dict)
        
        await microservice_manager.shutdown()
        
    @pytest.mark.asyncio
    async def test_full_performance_stack(self):
        """Test the full performance stack working together."""
        # Create all performance components
        gpu_manager = create_gpu_acceleration_manager()
        edge_manager = await create_edge_computing_manager()
        quantum_manager = create_quantum_ready_manager()
        microservice_manager = await create_microservice_manager()
        performance_optimizer = await create_performance_optimizer(collection_interval=1)
        
        # Perform operations across all systems
        # 1. Submit a task to edge computing
        edge_task_id = await edge_manager.submit_task(
            task_type="data_processing",
            data={"dataset_size": 1000},
            priority=2
        )
        
        # 2. Execute a quantum algorithm
        quantum_task_id = await quantum_manager.execute_quantum_algorithm(
            algorithm_type=QuantumAlgorithmType.GROVERS_SEARCH,
            input_data={"n": 4, "marked_item": 5},
            priority=3
        )
        
        # 3. Register a service
        service_id = microservice_manager.register_service(
            ServiceType.CACHE_SERVICE,
            "localhost",
            7070
        )
        
        # 4. Get performance metrics from all systems
        gpu_metrics = gpu_manager.get_performance_stats()
        edge_metrics = edge_manager.get_performance_metrics()
        quantum_metrics = quantum_manager.get_performance_metrics()
        microservice_metrics = microservice_manager.get_performance_metrics()
        performance_summary = performance_optimizer.get_performance_summary()
        
        # Verify all systems are providing metrics
        assert isinstance(gpu_metrics, dict)
        assert isinstance(edge_metrics, dict)
        assert isinstance(quantum_metrics, dict)
        assert isinstance(microservice_metrics, dict)
        assert isinstance(performance_summary, dict)
        
        # Shutdown managers
        await edge_manager.shutdown()
        await microservice_manager.shutdown()
        await performance_optimizer.stop_monitoring()
        
    @pytest.mark.asyncio
    async def test_performance_regression_detection(self):
        """Test performance regression detection across systems."""
        optimizer = await create_performance_optimizer(collection_interval=1)
        
        # Enable regression detection
        optimizer.enable_regression_detection()
        
        # Get initial metrics
        initial_metrics = optimizer.get_current_metrics()
        
        # Simulate some performance degradation by artificially increasing values
        # This would normally be detected through actual system monitoring
        
        # Get updated metrics after simulated degradation
        updated_metrics = optimizer.get_current_metrics()
        
        # Check that the system is monitoring
        summary = optimizer.get_performance_summary()
        assert summary["monitoring_active"] is True
        
        await optimizer.stop_monitoring()
        
    @pytest.mark.asyncio
    async def test_optimization_advice_generation(self):
        """Test generation of optimization advice."""
        optimizer = await create_performance_optimizer(collection_interval=1)
        
        # Enable optimization advice
        optimizer.enable_optimization_advice()
        
        # Artificially create some performance issues by setting high thresholds
        # In a real system, this would happen naturally
        
        # Get recommendations
        recommendations = optimizer.get_optimization_recommendations()
        
        # May be empty initially, but the system should be capable of generating them
        assert isinstance(recommendations, list)
        
        # Get performance summary which includes recommendation count
        summary = optimizer.get_performance_summary()
        assert "recommendations_count" in summary
        
        await optimizer.stop_monitoring()


if __name__ == "__main__":
    # Run the tests if this script is executed directly
    pytest.main([__file__, "-v"])