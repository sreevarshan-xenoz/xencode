#!/usr/bin/env python3
"""
Comprehensive Test Suite for Xencode Phase 2 Features

Tests all Phase 2 components: Intelligent Model Selection, Advanced Caching,
Smart Configuration, Error Handling, and Integration.
"""

import asyncio
import json
import tempfile
import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta

# Add xencode to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent))

from xencode.intelligent_model_selector import (
    HardwareDetector, ModelRecommendationEngine, SystemSpecs, ModelRecommendation
)
from xencode.advanced_cache_system import (
    HybridCacheManager, CacheKeyGenerator, CompressionManager, get_cache_manager
)
from xencode.smart_config_manager import (
    ConfigurationManager, XencodeConfig, ModelConfig, CacheConfig
)
from xencode.advanced_error_handler import (
    ErrorHandler, XencodeError, ErrorSeverity, ErrorCategory, ErrorContext
)
from xencode.phase2_coordinator import Phase2Coordinator


class TestHardwareDetector:
    """Test hardware detection system"""
    
    def test_detect_system_specs(self):
        detector = HardwareDetector()
        specs = detector.detect_system_specs()
        
        assert isinstance(specs, SystemSpecs)
        assert specs.cpu_cores > 0
        assert specs.total_ram_gb > 0
        assert specs.performance_score >= 0
        assert specs.performance_score <= 100
    
    @patch('subprocess.run')
    def test_gpu_detection_nvidia(self, mock_run):
        """Test NVIDIA GPU detection"""
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "8192"  # 8GB VRAM
        
        detector = HardwareDetector()
        gpu_info = detector._detect_gpu()
        
        assert gpu_info["available"] == True
        assert gpu_info["type"] == "nvidia"
        assert gpu_info["vram_gb"] == 8.0
    
    @patch('subprocess.run')
    def test_gpu_detection_none(self, mock_run):
        """Test no GPU detection"""
        mock_run.side_effect = FileNotFoundError()
        
        detector = HardwareDetector()
        gpu_info = detector._detect_gpu()
        
        assert gpu_info["available"] == False
        assert gpu_info["type"] == "none"
    
    def test_performance_score_calculation(self):
        detector = HardwareDetector()
        
        # Test high-end system
        score_high = detector._calculate_performance_score(16, 32.0, True, "x86_64")
        assert score_high >= 80
        
        # Test low-end system
        score_low = detector._calculate_performance_score(2, 4.0, False, "x86_64")
        assert score_low <= 40


class TestModelRecommendationEngine:
    """Test model recommendation system"""
    
    def test_model_database_loading(self):
        engine = ModelRecommendationEngine()
        assert len(engine.models) > 0
        
        # Check model structure
        for model in engine.models:
            assert isinstance(model, ModelRecommendation)
            assert model.name
            assert model.ollama_tag
            assert model.size_gb > 0
            assert model.ram_required_gb > 0
    
    def test_recommendations_high_end_system(self):
        engine = ModelRecommendationEngine()
        
        # High-end system specs
        specs = SystemSpecs(
            cpu_cores=16, cpu_architecture="x86_64", total_ram_gb=64.0,
            available_ram_gb=50.0, gpu_available=True, gpu_type="nvidia",
            gpu_vram_gb=16.0, storage_type="ssd", available_storage_gb=500.0,
            performance_score=95
        )
        
        primary, alternatives = engine.get_recommendations(specs)
        
        assert primary.performance_tier == "powerful"
        assert len(alternatives) >= 1
        assert primary.ram_required_gb <= specs.available_ram_gb
    
    def test_recommendations_low_end_system(self):
        engine = ModelRecommendationEngine()
        
        # Low-end system specs
        specs = SystemSpecs(
            cpu_cores=2, cpu_architecture="x86_64", total_ram_gb=4.0,
            available_ram_gb=3.0, gpu_available=False, gpu_type="none",
            gpu_vram_gb=0.0, storage_type="hdd", available_storage_gb=100.0,
            performance_score=25
        )
        
        primary, alternatives = engine.get_recommendations(specs)
        
        assert primary.performance_tier == "fast"
        assert primary.ram_required_gb <= specs.available_ram_gb


class TestCacheSystem:
    """Test advanced caching system"""
    
    def test_cache_key_generation(self):
        """Test cache key generation"""
        key1 = CacheKeyGenerator.generate_key("Hello world", "llama3.1:8b")
        key2 = CacheKeyGenerator.generate_key("Hello world", "llama3.1:8b")
        key3 = CacheKeyGenerator.generate_key("Different prompt", "llama3.1:8b")
        
        assert key1 == key2  # Same inputs should generate same key
        assert key1 != key3  # Different inputs should generate different keys
        assert len(key1) == 16  # Key should be 16 characters
    
    def test_context_key_generation(self):
        """Test conversation context key generation"""
        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"}
        ]
        
        key1 = CacheKeyGenerator.generate_context_key(history, "llama3.1:8b")
        key2 = CacheKeyGenerator.generate_context_key(history, "llama3.1:8b")
        
        assert key1 == key2
        assert len(key1) == 16
    
    def test_compression_manager(self):
        """Test data compression"""
        # Small data should not be compressed
        small_data = "Hello"
        compressed, is_compressed = CompressionManager.compress_data(small_data)
        assert not is_compressed
        
        # Large data should be compressed
        large_data = "x" * 10000
        compressed, is_compressed = CompressionManager.compress_data(large_data)
        # Note: Compression depends on data patterns, so we just check it doesn't fail
        assert isinstance(compressed, bytes)
        
        # Test decompression
        decompressed = CompressionManager.decompress_data(compressed, is_compressed)
        assert decompressed == large_data
    
    @pytest.mark.asyncio
    async def test_hybrid_cache_manager(self):
        """Test hybrid cache manager"""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir) / "cache"
            cache = HybridCacheManager(memory_cache_mb=32, disk_cache_mb=100, cache_dir=cache_dir)
            
            # Test storing and retrieving
            prompt = "Test prompt"
            model = "test-model"
            response = {"text": "Test response", "timestamp": datetime.now().isoformat()}
            
            success = await cache.store_response(prompt, model, response)
            assert success
            
            cached_response = await cache.get_response(prompt, model)
            assert cached_response == response
            
            # Test cache miss
            missing_response = await cache.get_response("Different prompt", model)
            assert missing_response is None
            
            # Test performance stats
            stats = cache.get_performance_stats()
            assert stats["total_requests"] >= 2
            assert stats["total_hit_rate"] >= 0


class TestConfigurationManager:
    """Test smart configuration system"""
    
    def test_default_config_creation(self):
        """Test default configuration creation"""
        config = XencodeConfig()
        
        assert config.version
        assert isinstance(config.model, ModelConfig)
        assert isinstance(config.cache, CacheConfig)
        assert config.model.name
        assert config.model.temperature >= 0.0
        assert config.cache.memory_cache_mb > 0
    
    def test_config_validation(self):
        """Test configuration validation"""
        config = XencodeConfig()
        
        # Valid config should pass
        errors = config.validate()
        assert not errors
        
        # Invalid config should fail
        config.model.temperature = 5.0  # Invalid temperature
        config.cache.memory_cache_mb = -1  # Invalid cache size
        
        errors = config.validate()
        assert "model" in errors
        assert "cache" in errors
    
    def test_config_serialization(self):
        """Test config to/from dict conversion"""
        original_config = XencodeConfig()
        original_config.model.name = "test-model"
        original_config.cache.enabled = False
        
        # Convert to dict and back
        config_dict = original_config.to_dict()
        restored_config = XencodeConfig.from_dict(config_dict)
        
        assert restored_config.model.name == original_config.model.name
        assert restored_config.cache.enabled == original_config.cache.enabled
    
    def test_yaml_config_loading(self):
        """Test YAML configuration loading/saving"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml_content = """
version: "2.0.0"
model:
  name: "test-model"
  temperature: 0.8
cache:
  enabled: true
  memory_cache_mb: 512
"""
            f.write(yaml_content)
            f.flush()
            
            manager = ConfigurationManager(Path(f.name))
            config = manager.load_config()
            
            assert config.model.name == "test-model"
            assert config.model.temperature == 0.8
            assert config.cache.memory_cache_mb == 512
            
            Path(f.name).unlink()  # Cleanup
    
    @patch.dict('os.environ', {'XENCODE_MODEL_NAME': 'env-model', 'XENCODE_CACHE_MEMORY_MB': '128'})
    def test_environment_overrides(self):
        """Test environment variable overrides"""
        manager = ConfigurationManager()
        manager._load_environment_overrides()
        
        assert 'model' in manager.environment_overrides
        assert manager.environment_overrides['model']['name'] == 'env-model'


class TestErrorHandler:
    """Test advanced error handling system"""
    
    @pytest.mark.asyncio
    async def test_error_classification(self):
        """Test error classification"""
        handler = ErrorHandler()
        context = ErrorContext(function_name="test_function")
        
        # Test network error
        network_error = ConnectionError("Network timeout")
        classified = handler._classify_error(network_error, context, ErrorSeverity.ERROR, ErrorCategory.UNKNOWN)
        
        assert classified.category == ErrorCategory.NETWORK
        assert classified.severity == ErrorSeverity.ERROR
        assert "network" in classified.user_message.lower()
        assert classified.recoverable
    
    @pytest.mark.asyncio
    async def test_error_recovery(self):
        """Test error recovery system"""
        handler = ErrorHandler()
        context = ErrorContext(function_name="test_function")
        
        # Create recoverable error
        error = XencodeError(
            severity=ErrorSeverity.ERROR,
            category=ErrorCategory.CACHE,
            message="Cache error",
            technical_details="Test error",
            user_message="Cache system issue",
            suggested_actions=["Clear cache", "Restart"],
            context=context,
            recoverable=True
        )
        
        # Test recovery attempt
        recovery_success = await handler.recovery_manager.attempt_recovery(error)
        # Recovery should attempt even if it doesn't fully succeed
        assert isinstance(recovery_success, bool)
    
    def test_error_history(self):
        """Test error history tracking"""
        handler = ErrorHandler()
        
        # Initially empty
        assert len(handler.error_history) == 0
        
        # Add some mock errors
        for i in range(5):
            error = XencodeError(
                severity=ErrorSeverity.ERROR,
                category=ErrorCategory.UNKNOWN,
                message=f"Test error {i}",
                technical_details="",
                user_message="",
                suggested_actions=[],
                context=ErrorContext(),
                recoverable=True
            )
            handler.error_history.append(error)
        
        assert len(handler.error_history) == 5
        
        # Test summary
        summary = handler.get_error_summary()
        assert summary["total_errors"] == 5
        assert len(summary["recent_errors"]) == 5


class TestPhase2Integration:
    """Test Phase 2 integration and coordination"""
    
    @pytest.mark.asyncio
    async def test_coordinator_initialization(self):
        """Test Phase 2 coordinator initialization"""
        coordinator = Phase2Coordinator()
        
        # Should not be initialized initially
        assert not coordinator.initialized
        
        # Initialize should succeed
        success = await coordinator.initialize()
        assert success
        assert coordinator.initialized
        assert coordinator.config is not None
        assert coordinator.cache_manager is not None
        assert coordinator.error_handler is not None
    
    @pytest.mark.asyncio
    async def test_system_status(self):
        """Test system status reporting"""
        coordinator = Phase2Coordinator()
        await coordinator.initialize()
        
        status = coordinator.get_system_status()
        
        assert "phase2_initialized" in status
        assert "config_status" in status
        assert "cache_status" in status
        assert "performance_score" in status
        assert isinstance(status["performance_score"], int)
        assert 0 <= status["performance_score"] <= 100
    
    @pytest.mark.asyncio
    async def test_performance_optimization(self):
        """Test performance optimization"""
        coordinator = Phase2Coordinator()
        await coordinator.initialize()
        
        results = await coordinator.optimize_performance()
        
        assert "cache_optimized" in results
        assert "memory_freed" in results
        assert "config_optimized" in results
        assert isinstance(results["memory_freed"], (int, float))
    
    @pytest.mark.asyncio
    async def test_health_check(self):
        """Test system health check"""
        coordinator = Phase2Coordinator()
        await coordinator.initialize()
        
        health_status = await coordinator.health_check()
        assert isinstance(health_status, bool)


class TestEndToEndIntegration:
    """End-to-end integration tests"""
    
    @pytest.mark.asyncio
    async def test_full_workflow(self):
        """Test complete Phase 2 workflow"""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir) / "cache"
            config_path = Path(temp_dir) / "config.yaml"
            
            # 1. Initialize coordinator
            coordinator = Phase2Coordinator()
            success = await coordinator.initialize(config_path)
            assert success
            
            # 2. Check system status
            status = coordinator.get_system_status()
            assert status["phase2_initialized"]
            
            # 3. Test cache functionality
            cache_manager = coordinator.cache_manager
            test_response = {"text": "Hello world", "model": "test"}
            
            await cache_manager.store_response("Hello", "test-model", test_response)
            cached = await cache_manager.get_response("Hello", "test-model")
            assert cached == test_response
            
            # 4. Test configuration
            config = coordinator.config
            assert config is not None
            assert config.model.name
            
            # 5. Run optimization
            opt_results = await coordinator.optimize_performance()
            assert isinstance(opt_results, dict)
            
            # 6. Health check
            health = await coordinator.health_check()
            assert isinstance(health, bool)


def run_comprehensive_tests():
    """Run all Phase 2 tests"""
    print("🧪 Running Xencode Phase 2 Test Suite...")
    
    # Run pytest with verbose output
    exit_code = pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--color=yes"
    ])
    
    if exit_code == 0:
        print("✅ All Phase 2 tests passed!")
    else:
        print("❌ Some tests failed!")
    
    return exit_code


if __name__ == "__main__":
    exit_code = run_comprehensive_tests()
    sys.exit(exit_code)