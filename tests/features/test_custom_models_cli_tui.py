#!/usr/bin/env python3
"""
Tests for Custom Models CLI and TUI Components

Tests cover:
- CLI command availability and functionality
- TUI component rendering and interaction
- Model management workflows
- Performance monitoring
- Version management
"""


import pytest
import pytest_asyncio

from pathlib import Path

from xencode.features.custom_models import CustomModelManager, FeatureConfig


@pytest.fixture(autouse=True)
def _isolate_models_storage():
    """Isolate CustomModelManager's persistent storage per test."""
    models_file = Path.home() / '.xencode' / 'custom_models.json'
    backup = models_file.read_text() if models_file.exists() else None
    models_file.parent.mkdir(parents=True, exist_ok=True)
    models_file.unlink(missing_ok=True)

    yield

    if backup is None:
        models_file.unlink(missing_ok=True)
    else:
        models_file.write_text(backup)


@pytest.fixture
def feature_config():
    """Create a test feature configuration"""
    return FeatureConfig(
        name="custom_models",
        enabled=True,
        config={
            'max_models': 10,
            'training': {
                'max_epochs': 10,
                'batch_size': 32
            },
            'performance': {
                'min_accuracy': 0.85
            }
        }
    )


@pytest_asyncio.fixture
async def custom_models_feature(feature_config, _isolate_models_storage):
    """Create and initialize a custom models feature"""
    feature = CustomModelManager(feature_config)
    await feature.initialize()
    yield feature
    await feature.shutdown()


class TestCLICommands:
    """Test CLI command functionality"""

    @pytest.mark.asyncio
    async def test_cli_commands_available(self, custom_models_feature):
        """Test that CLI commands are available"""
        commands = custom_models_feature.get_cli_commands()

        assert commands is not None
        assert len(commands) > 0
        assert hasattr(commands[0], 'name')

    @pytest.mark.asyncio
    async def test_analyze_command(self, custom_models_feature, tmp_path):
        """Test analyze command functionality"""
        # Create a test codebase
        test_file = tmp_path / "test.py"
        test_file.write_text("def hello():\n    print('Hello')\n")

        result = await custom_models_feature.analyze(str(tmp_path))

        assert 'codebase_path' in result
        assert 'patterns' in result
        assert 'total_patterns' in result
        assert result['codebase_path'] == str(tmp_path)

    @pytest.mark.asyncio
    async def test_train_command(self, custom_models_feature, tmp_path):
        """Test train command functionality"""
        # Create a test codebase
        test_file = tmp_path / "test.py"
        test_file.write_text("def hello():\n    print('Hello')\n")

        result = await custom_models_feature.train(
            model_name="test-model",
            codebase_path=str(tmp_path),
            task_type="code_completion"
        )

        assert 'model_name' in result
        assert 'version' in result
        assert 'status' in result
        assert 'accuracy' in result
        assert result['model_name'] == "test-model"

    @pytest.mark.asyncio
    async def test_list_command(self, custom_models_feature, tmp_path):
        """Test list command functionality"""
        # Train a model first
        test_file = tmp_path / "test.py"
        test_file.write_text("def hello():\n    print('Hello')\n")

        await custom_models_feature.train(
            model_name="test-model",
            codebase_path=str(tmp_path),
            task_type="code_completion"
        )

        models = await custom_models_feature.list_models()

        assert isinstance(models, list)
        assert len(models) > 0
        assert models[0]['name'] == "test-model"

    @pytest.mark.asyncio
    async def test_performance_command(self, custom_models_feature, tmp_path):
        """Test performance command functionality"""
        # Train a model first
        test_file = tmp_path / "test.py"
        test_file.write_text("def hello():\n    print('Hello')\n")

        result = await custom_models_feature.train(
            model_name="test-model",
            codebase_path=str(tmp_path),
            task_type="code_completion"
        )

        # Record metrics
        await custom_models_feature.performance_monitor.record_metrics(
            model_name="test-model",
            version=result['version'],
            speed_ms=150.0,
            accuracy=0.9,
            memory_mb=800.0,
            samples_tested=100
        )

        performance = await custom_models_feature.monitor("test-model", result['version'])

        assert 'model_name' in performance
        assert 'version' in performance
        assert 'metrics' in performance
        assert performance['metrics'] is not None

    @pytest.mark.asyncio
    async def test_delete_command(self, custom_models_feature, tmp_path):
        """Test delete command functionality"""
        # Train a model first
        test_file = tmp_path / "test.py"
        test_file.write_text("def hello():\n    print('Hello')\n")

        await custom_models_feature.train(
            model_name="test-model",
            codebase_path=str(tmp_path),
            task_type="code_completion"
        )

        # Delete the model
        result = await custom_models_feature.delete_model("test-model")

        assert result['success'] is True
        assert result['model_name'] == "test-model"

        # Verify it's deleted
        models = await custom_models_feature.list_models()
        assert len([m for m in models if m['name'] == "test-model"]) == 0

    @pytest.mark.asyncio
    async def test_rollback_command(self, custom_models_feature, tmp_path):
        """Test rollback command functionality"""
        # Train two versions
        test_file = tmp_path / "test.py"
        test_file.write_text("def hello():\n    print('Hello')\n")

        result1 = await custom_models_feature.train(
            model_name="test-model",
            codebase_path=str(tmp_path),
            task_type="code_completion"
        )

        await custom_models_feature.train(
            model_name="test-model",
            codebase_path=str(tmp_path),
            task_type="code_completion"
        )

        # Rollback to first version
        rollback_result = await custom_models_feature.rollback(
            "test-model",
            result1['version']
        )

        assert rollback_result['success'] is True
        assert rollback_result['new_version'] == result1['version']

    @pytest.mark.asyncio
    async def test_compare_command(self, custom_models_feature, tmp_path):
        """Test compare command functionality"""
        # Train two versions
        test_file = tmp_path / "test.py"
        test_file.write_text("def hello():\n    print('Hello')\n")

        result1 = await custom_models_feature.train(
            model_name="test-model",
            codebase_path=str(tmp_path),
            task_type="code_completion"
        )

        result2 = await custom_models_feature.train(
            model_name="test-model",
            codebase_path=str(tmp_path),
            task_type="code_completion"
        )

        # Record metrics for both
        await custom_models_feature.performance_monitor.record_metrics(
            model_name="test-model",
            version=result1['version'],
            speed_ms=150.0,
            accuracy=0.9,
            memory_mb=800.0,
            samples_tested=100
        )

        await custom_models_feature.performance_monitor.record_metrics(
            model_name="test-model",
            version=result2['version'],
            speed_ms=120.0,
            accuracy=0.92,
            memory_mb=750.0,
            samples_tested=100
        )

        # Compare versions
        comparison = await custom_models_feature.compare_versions(
            "test-model",
            result1['version'],
            result2['version']
        )

        assert 'model_name' in comparison
        assert 'version1' in comparison
        assert 'version2' in comparison
        assert 'comparison' in comparison


class TestTUIComponents:
    """Test TUI component functionality"""

    def test_tui_components_available(self, custom_models_feature):
        """Test that TUI components are available"""
        components = custom_models_feature.get_tui_components()

        assert components is not None
        assert len(components) > 0

    @pytest.mark.asyncio
    async def test_custom_models_panel_creation(self):
        """Test CustomModelsPanel can be created"""
        from xencode.tui.widgets.custom_models_panel import CustomModelsPanel

        panel = CustomModelsPanel()
        assert panel is not None

    @pytest.mark.asyncio
    async def test_training_dashboard_creation(self):
        """Test TrainingDashboard can be created"""
        from xencode.tui.widgets.custom_models_panel import TrainingDashboard

        dashboard = TrainingDashboard()
        assert dashboard is not None

    @pytest.mark.asyncio
    async def test_performance_viewer_creation(self):
        """Test PerformanceViewer can be created"""
        from xencode.tui.widgets.custom_models_panel import PerformanceViewer

        viewer = PerformanceViewer()
        assert viewer is not None

    @pytest.mark.asyncio
    async def test_version_history_creation(self):
        """Test VersionHistory can be created"""
        from xencode.tui.widgets.custom_models_panel import VersionHistory

        history = VersionHistory()
        assert history is not None

    @pytest.mark.asyncio
    async def test_codebase_analysis_results_creation(self):
        """Test CodebaseAnalysisResults can be created"""
        from xencode.tui.widgets.custom_models_panel import CodebaseAnalysisResults

        results = CodebaseAnalysisResults()
        assert results is not None


class TestModelWorkflow:
    """Test complete model workflow"""

    @pytest.mark.asyncio
    async def test_complete_workflow(self, custom_models_feature, tmp_path):
        """Test complete model workflow from analysis to monitoring"""
        # Create test codebase
        test_file = tmp_path / "test.py"
        test_file.write_text("def hello():\n    print('Hello')\n")

        # Step 1: Analyze
        analysis = await custom_models_feature.analyze(str(tmp_path))
        assert 'patterns' in analysis

        # Step 2: Train
        training_result = await custom_models_feature.train(
            model_name="workflow-test",
            codebase_path=str(tmp_path),
            task_type="code_completion"
        )
        assert training_result['model_name'] == "workflow-test"

        # Step 3: Record metrics
        await custom_models_feature.performance_monitor.record_metrics(
            model_name="workflow-test",
            version=training_result['version'],
            speed_ms=150.0,
            accuracy=0.9,
            memory_mb=800.0,
            samples_tested=100
        )

        # Step 4: Monitor
        performance = await custom_models_feature.monitor(
            "workflow-test",
            training_result['version']
        )
        assert performance['metrics'] is not None

        # Step 5: List
        models = await custom_models_feature.list_models()
        assert len([m for m in models if m['name'] == "workflow-test"]) == 1


class TestCodebaseAnalyzer:
    """Test codebase analyzer functionality"""

    @pytest.mark.asyncio
    async def test_analyze_python_files(self, custom_models_feature, tmp_path):
        """Test analyzing Python files"""
        # Create test files
        (tmp_path / "test1.py").write_text("def hello():\n    pass\n")
        (tmp_path / "test2.py").write_text("class MyClass:\n    pass\n")

        result = await custom_models_feature.analyze(str(tmp_path))

        assert result['total_patterns'] > 0
        assert any(p['pattern_type'].startswith('file_type') for p in result['patterns'])

    @pytest.mark.asyncio
    async def test_analyze_empty_directory(self, custom_models_feature, tmp_path):
        """Test analyzing empty directory"""
        result = await custom_models_feature.analyze(str(tmp_path))

        assert 'patterns' in result
        assert 'total_patterns' in result


class TestPerformanceMonitor:
    """Test performance monitor functionality"""

    @pytest.mark.asyncio
    async def test_record_metrics(self, custom_models_feature):
        """Test recording performance metrics"""
        await custom_models_feature.performance_monitor.record_metrics(
            model_name="test-model",
            version="v1",
            speed_ms=150.0,
            accuracy=0.9,
            memory_mb=800.0,
            samples_tested=100
        )

        metrics = await custom_models_feature.performance_monitor.get_metrics(
            "test-model",
            "v1"
        )

        assert metrics is not None
        assert metrics.speed_ms == 150.0
        assert metrics.accuracy == 0.9

    @pytest.mark.asyncio
    async def test_compare_metrics(self, custom_models_feature):
        """Test comparing metrics between versions"""
        # Record metrics for two versions
        await custom_models_feature.performance_monitor.record_metrics(
            model_name="test-model",
            version="v1",
            speed_ms=150.0,
            accuracy=0.9,
            memory_mb=800.0,
            samples_tested=100
        )

        await custom_models_feature.performance_monitor.record_metrics(
            model_name="test-model",
            version="v2",
            speed_ms=120.0,
            accuracy=0.92,
            memory_mb=750.0,
            samples_tested=100
        )

        comparison = await custom_models_feature.performance_monitor.compare_metrics(
            "test-model",
            "v1",
            "v2"
        )

        assert 'speed_improvement' in comparison
        assert 'accuracy_improvement' in comparison
        assert 'memory_improvement' in comparison


class TestVersionManagement:
    """Test version management functionality"""

    @pytest.mark.asyncio
    async def test_multiple_versions(self, custom_models_feature, tmp_path):
        """Test creating multiple versions of a model"""
        test_file = tmp_path / "test.py"
        test_file.write_text("def hello():\n    print('Hello')\n")

        # Train multiple versions
        await custom_models_feature.train(
            model_name="multi-version",
            codebase_path=str(tmp_path),
            task_type="code_completion"
        )

        await custom_models_feature.train(
            model_name="multi-version",
            codebase_path=str(tmp_path),
            task_type="code_completion"
        )

        # List models
        models = await custom_models_feature.list_models()
        multi_version_model = next(m for m in models if m['name'] == "multi-version")

        assert len(multi_version_model['versions']) == 2

    @pytest.mark.asyncio
    async def test_delete_specific_version(self, custom_models_feature, tmp_path):
        """Test deleting a specific version"""
        test_file = tmp_path / "test.py"
        test_file.write_text("def hello():\n    print('Hello')\n")

        # Train two versions
        result1 = await custom_models_feature.train(
            model_name="version-test",
            codebase_path=str(tmp_path),
            task_type="code_completion"
        )

        result2 = await custom_models_feature.train(
            model_name="version-test",
            codebase_path=str(tmp_path),
            task_type="code_completion"
        )

        # Delete first version
        await custom_models_feature.delete_model("version-test", result1['version'])

        # Verify only one version remains
        models = await custom_models_feature.list_models()
        version_test_model = next(m for m in models if m['name'] == "version-test")

        assert len(version_test_model['versions']) == 1
        assert version_test_model['versions'][0]['version'] == result2['version']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
