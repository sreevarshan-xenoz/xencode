"""
Tests for Custom AI Models feature
"""

import shutil
import tempfile
from pathlib import Path

import pytest

from xencode.features.base import FeatureConfig
from xencode.features.custom_models import (
    CodebaseAnalyzer,
    CodebasePattern,
    CustomModelManager,
    ModelStatus,
    ModelTrainer,
    ModelVersion,
    PerformanceMetrics,
    PerformanceMonitor,
    TaskType,
)


@pytest.fixture
def temp_codebase():
    """Create a temporary codebase for testing"""
    temp_dir = tempfile.mkdtemp()

    # Create some Python files
    (Path(temp_dir) / "main.py").write_text("""
import os
import sys

def hello_world():
    return "Hello, World!"

class MyClass:
    def __init__(self):
        self.value = 42
""")

    (Path(temp_dir) / "utils.py").write_text("""
import json
from typing import List

def process_data(data: List[int]) -> int:
    return sum(data)

def format_output(value: int) -> str:
    return f"Result: {value}"
""")

    yield temp_dir

    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def feature_config():
    """Create feature configuration"""
    return FeatureConfig(
        name="custom_models",
        enabled=True,
        config={
            'enabled': True,
            'max_models': 10,
            'training': {
                'max_epochs': 5,
                'batch_size': 16
            },
            'performance': {
                'min_accuracy': 0.85
            }
        }
    )


@pytest.mark.asyncio
async def test_codebase_analyzer(temp_codebase):
    """Test codebase analysis"""
    analyzer = CodebaseAnalyzer()
    patterns = await analyzer.analyze(temp_codebase)

    assert len(patterns) > 0
    assert any(p.pattern_type.startswith('file_type_') for p in patterns)



@pytest.mark.asyncio
async def test_model_trainer():
    """Test model training"""
    trainer = ModelTrainer(max_epochs=3, batch_size=16, min_accuracy=0.80)

    patterns = [
        {'pattern_type': 'test', 'frequency': 10, 'examples': ['example1']}
    ]

    result = await trainer.train(
        model_name="test_model",
        version="v1",
        patterns=patterns,
        task_type=TaskType.CODE_COMPLETION
    )

    assert result['success'] is True
    assert result['accuracy'] >= 0.80
    assert result['epochs_trained'] <= 3


@pytest.mark.asyncio
async def test_performance_monitor():
    """Test performance monitoring"""
    monitor = PerformanceMonitor()

    # Record metrics
    await monitor.record_metrics(
        model_name="test_model",
        version="v1",
        speed_ms=150.0,
        accuracy=0.90,
        memory_mb=800.0,
        samples_tested=100
    )

    # Get metrics
    metrics = await monitor.get_metrics("test_model", "v1")
    assert metrics is not None
    assert metrics.speed_ms == 150.0
    assert metrics.accuracy == 0.90

    # Generate report
    report = await monitor.generate_report("test_model", "v1")
    assert 'grade' in report
    assert 'recommendations' in report


@pytest.mark.asyncio
async def test_custom_model_manager_analyze(feature_config, temp_codebase):
    """Test CustomModelManager analyze method"""
    manager = CustomModelManager(feature_config)
    await manager.initialize()

    result = await manager.analyze(temp_codebase)

    assert 'codebase_path' in result
    assert 'patterns' in result
    assert 'total_patterns' in result
    assert result['total_patterns'] > 0

    await manager.shutdown()


@pytest.mark.asyncio
async def test_custom_model_manager_train(feature_config, temp_codebase):
    """Test CustomModelManager train method"""
    manager = CustomModelManager(feature_config)
    await manager.initialize()

    result = await manager.train(
        model_name="test_model",
        codebase_path=temp_codebase,
        task_type="code_completion"
    )

    assert 'model_name' in result
    assert 'version' in result
    assert 'status' in result
    assert result['model_name'] == "test_model"

    await manager.shutdown()


@pytest.mark.asyncio
async def test_custom_model_manager_list_models(feature_config, temp_codebase):
    """Test listing models"""
    manager = CustomModelManager(feature_config)
    await manager.initialize()

    # Train a model first
    await manager.train(
        model_name="test_model",
        codebase_path=temp_codebase,
        task_type="code_completion"
    )

    # List models
    models = await manager.list_models()
    assert len(models) > 0
    assert models[0]['name'] == "test_model"

    await manager.shutdown()


@pytest.mark.asyncio
async def test_custom_model_manager_delete_model(feature_config, temp_codebase):
    """Test deleting a model"""
    manager = CustomModelManager(feature_config)
    await manager.initialize()

    # Train a model first
    await manager.train(
        model_name="test_model",
        codebase_path=temp_codebase,
        task_type="code_completion"
    )

    # Delete the model
    result = await manager.delete_model("test_model")
    assert result['success'] is True

    # Verify it's deleted
    models = await manager.list_models()
    assert len(models) == 0

    await manager.shutdown()


@pytest.mark.asyncio
async def test_custom_model_manager_rollback(feature_config, temp_codebase):
    """Test rolling back to a previous version"""
    manager = CustomModelManager(feature_config)
    await manager.initialize()

    # Train first version
    result1 = await manager.train(
        model_name="test_model",
        codebase_path=temp_codebase,
        task_type="code_completion"
    )
    version1 = result1['version']

    # Train second version
    result2 = await manager.train(
        model_name="test_model",
        codebase_path=temp_codebase,
        task_type="code_completion"
    )
    result2['version']

    # Rollback to version1
    rollback_result = await manager.rollback("test_model", version1)
    assert rollback_result['success'] is True
    assert rollback_result['new_version'] == version1

    await manager.shutdown()


def test_codebase_pattern_to_dict():
    """Test CodebasePattern to_dict method"""
    pattern = CodebasePattern(
        pattern_type="test_pattern",
        frequency=10,
        examples=["ex1", "ex2", "ex3"],
        confidence=0.95
    )

    result = pattern.to_dict()
    assert result['pattern_type'] == "test_pattern"
    assert result['frequency'] == 10
    assert len(result['examples']) <= 5


def test_model_version_to_dict():
    """Test ModelVersion to_dict method"""
    version = ModelVersion(
        version="v1",
        created_at="2024-01-01T00:00:00",
        status=ModelStatus.READY,
        task_type=TaskType.CODE_COMPLETION,
        accuracy=0.90,
        training_samples=100
    )

    result = version.to_dict()
    assert result['version'] == "v1"
    assert result['status'] == "ready"
    assert result['task_type'] == "code_completion"


def test_performance_metrics_to_dict():
    """Test PerformanceMetrics to_dict method"""
    metrics = PerformanceMetrics(
        model_id="test_model",
        version="v1",
        speed_ms=150.0,
        accuracy=0.90,
        memory_mb=800.0,
        samples_tested=100
    )

    result = metrics.to_dict()
    assert result['model_id'] == "test_model"
    assert result['speed_ms'] == 150.0
    assert result['accuracy'] == 0.90
