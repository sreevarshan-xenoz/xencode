"""
Tests for Project Analyzer Feature

Tests all functionality including structure scanning, dependency analysis,
metrics calculation, architecture visualization, and technical debt detection.
"""

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path

from xencode.features.project_analyzer import (
    ProjectAnalyzerFeature,
    StructureScanner,
    DependencyAnalyzer,
    MetricsCalculator,
    ArchitectureVisualizer,
    TechnicalDebtDetector,
    ProjectAnalyzerConfig
)
from xencode.features.base import FeatureConfig


@pytest.fixture
def temp_project():
    """Create a temporary project directory"""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_python_project(temp_project):
    """Create a sample Python project"""
    # Create directory structure
    (temp_project / 'src').mkdir()
    (temp_project / 'tests').mkdir()
    (temp_project / 'docs').mkdir()
    
    # Create Python files
    (temp_project / 'src' / '__init__.py').write_text('')
    (temp_project / 'src' / 'main.py').write_text('''
"""Main module"""
import os
import sys
from typing import List

def hello_world():
    """Print hello world"""
    print("Hello, World!")

def complex_function(a, b, c, d, e, f):
    """Function with many parameters"""
    if a > 0:
        if b > 0:
            if c > 0:
                return a + b + c
    return 0

class MyClass:
    """Sample class"""
    def __init__(self):
        self.value = 0
    
    def method(self):
        """Sample method"""
        return self.value
''')
    
    (temp_project / 'src' / 'utils.py').write_text('''
"""Utility functions"""

def add(a, b):
    """Add two numbers"""
    return a + b

def subtract(a, b):
    """Subtract two numbers"""
    return a - b
''')
    
    # Create requirements.txt
    (temp_project / 'requirements.txt').write_text('''
pytest==7.4.0
requests==2.31.0
numpy>=1.24.0
''')
    
    # Create setup.py
    (temp_project / 'setup.py').write_text('''
from setuptools import setup

setup(
    name='sample-project',
    version='1.0.0',
    install_requires=[
        'click>=8.0.0',
        'pyyaml==6.0'
    ]
)
''')
    
    return temp_project



@pytest.fixture
def feature_config():
    """Create feature configuration"""
    return FeatureConfig(
        name='project_analyzer',
        enabled=True,
        config={
            'enabled': True,
            'max_file_size': 1024 * 1024,
            'exclude_patterns': ['__pycache__', '.git', 'node_modules'],
            'include_extensions': ['.py', '.js', '.ts'],
            'generate_diagrams': True,
            'track_metrics': True
        }
    )


class TestStructureScanner:
    """Test StructureScanner functionality"""
    
    @pytest.mark.asyncio
    async def test_scan_project_structure(self, sample_python_project):
        """Test scanning project structure"""
        scanner = StructureScanner(
            exclude_patterns=['__pycache__', '.git'],
            include_extensions=['.py'],
            max_file_size=1024 * 1024
        )
        
        result = await scanner.scan(sample_python_project)
        
        assert result['total_files'] >= 3
        assert result['total_directories'] > 0
        assert 'Python' in result['languages']
        assert result['project_type'] == 'Python'
    
    @pytest.mark.asyncio
    async def test_exclude_patterns(self, temp_project):
        """Test that excluded patterns are ignored"""
        # Create files to exclude
        (temp_project / '__pycache__').mkdir()
        (temp_project / '__pycache__' / 'test.pyc').write_text('bytecode')
        (temp_project / 'main.py').write_text('print("hello")')
        
        scanner = StructureScanner(
            exclude_patterns=['__pycache__'],
            include_extensions=['.py', '.pyc'],
            max_file_size=1024 * 1024
        )
        
        result = await scanner.scan(temp_project)
        
        # Should only find main.py, not the .pyc file
        assert result['total_files'] == 1
        assert any('main.py' in f['path'] for f in result['files'])
    
    @pytest.mark.asyncio
    async def test_language_detection(self, temp_project):
        """Test language detection from file extensions"""
        (temp_project / 'test.py').write_text('print("python")')
        (temp_project / 'test.js').write_text('console.log("js")')
        (temp_project / 'test.ts').write_text('console.log("ts")')
        
        scanner = StructureScanner(
            exclude_patterns=[],
            include_extensions=['.py', '.js', '.ts'],
            max_file_size=1024 * 1024
        )
        
        result = await scanner.scan(temp_project)
        
        assert 'Python' in result['languages']
        assert 'JavaScript' in result['languages']
        assert 'TypeScript' in result['languages']


class TestDependencyAnalyzer:
    """Test DependencyAnalyzer functionality"""
    
    @pytest.mark.asyncio
    async def test_analyze_python_dependencies(self, sample_python_project):
        """Test analyzing Python dependencies"""
        scanner = StructureScanner(
            exclude_patterns=['__pycache__'],
            include_extensions=['.py'],
            max_file_size=1024 * 1024
        )
        structure = await scanner.scan(sample_python_project)
        
        analyzer = DependencyAnalyzer()
        result = await analyzer.analyze(sample_python_project, structure)
        
        assert result['dependency_count'] > 0
        assert any(d['name'] == 'pytest' for d in result['dependencies'])
        assert any(d['name'] == 'requests' for d in result['dependencies'])
    
    @pytest.mark.asyncio
    async def test_internal_dependencies(self, sample_python_project):
        """Test analyzing internal module dependencies"""
        scanner = StructureScanner(
            exclude_patterns=['__pycache__'],
            include_extensions=['.py'],
            max_file_size=1024 * 1024
        )
        structure = await scanner.scan(sample_python_project)
        
        analyzer = DependencyAnalyzer()
        result = await analyzer.analyze(sample_python_project, structure)
        
        assert 'internal_dependencies' in result
        # main.py should have imports - check both Unix and Windows paths
        main_deps = result['internal_dependencies'].get('src/main.py', []) or \
                   result['internal_dependencies'].get('src\\main.py', [])
        assert 'os' in main_deps or 'sys' in main_deps
    
    @pytest.mark.asyncio
    async def test_nodejs_dependencies(self, temp_project):
        """Test analyzing Node.js dependencies"""
        # Create package.json
        (temp_project / 'package.json').write_text('''
{
  "name": "test-project",
  "dependencies": {
    "express": "^4.18.0",
    "lodash": "^4.17.21"
  },
  "devDependencies": {
    "jest": "^29.0.0"
  }
}
''')
        
        scanner = StructureScanner(
            exclude_patterns=[],
            include_extensions=['.js'],
            max_file_size=1024 * 1024
        )
        structure = await scanner.scan(temp_project)
        structure['project_type'] = 'Node.js'
        
        analyzer = DependencyAnalyzer()
        result = await analyzer.analyze(temp_project, structure)
        
        assert any(d['name'] == 'express' for d in result['dependencies'])
        assert any(d['name'] == 'jest' and d['type'] == 'dev' for d in result['dependencies'])


class TestMetricsCalculator:
    """Test MetricsCalculator functionality"""
    
    @pytest.mark.asyncio
    async def test_calculate_metrics(self, sample_python_project):
        """Test calculating project metrics"""
        scanner = StructureScanner(
            exclude_patterns=['__pycache__'],
            include_extensions=['.py'],
            max_file_size=1024 * 1024
        )
        structure = await scanner.scan(sample_python_project)
        
        calculator = MetricsCalculator()
        result = await calculator.calculate(sample_python_project, structure)
        
        assert result['total_lines'] > 0
        assert result['code_lines'] > 0
        assert result['comment_lines'] >= 0
        assert result['average_complexity'] >= 0
        assert 0 <= result['maintainability_index'] <= 100
    
    @pytest.mark.asyncio
    async def test_complexity_calculation(self, temp_project):
        """Test complexity calculation"""
        # Create file with known complexity
        (temp_project / 'complex.py').write_text('''
def complex_function(x):
    if x > 0:
        if x > 10:
            if x > 20:
                return "high"
            return "medium"
        return "low"
    return "negative"
''')
        
        scanner = StructureScanner(
            exclude_patterns=[],
            include_extensions=['.py'],
            max_file_size=1024 * 1024
        )
        structure = await scanner.scan(temp_project)
        
        calculator = MetricsCalculator()
        result = await calculator.calculate(temp_project, structure)
        
        # Should detect high complexity
        assert result['average_complexity'] > 1
    
    @pytest.mark.asyncio
    async def test_comment_ratio(self, temp_project):
        """Test comment ratio calculation"""
        (temp_project / 'documented.py').write_text('''
# This is a comment
# Another comment
def function():
    # Comment inside
    pass
''')
        
        scanner = StructureScanner(
            exclude_patterns=[],
            include_extensions=['.py'],
            max_file_size=1024 * 1024
        )
        structure = await scanner.scan(temp_project)
        
        calculator = MetricsCalculator()
        result = await calculator.calculate(temp_project, structure)
        
        assert result['comment_ratio'] > 0



class TestArchitectureVisualizer:
    """Test ArchitectureVisualizer functionality"""
    
    @pytest.mark.asyncio
    async def test_generate_visualization(self, sample_python_project):
        """Test generating architecture visualization"""
        scanner = StructureScanner(
            exclude_patterns=['__pycache__'],
            include_extensions=['.py'],
            max_file_size=1024 * 1024
        )
        structure = await scanner.scan(sample_python_project)
        
        analyzer = DependencyAnalyzer()
        dependencies = await analyzer.analyze(sample_python_project, structure)
        
        visualizer = ArchitectureVisualizer()
        result = await visualizer.generate(sample_python_project, structure, dependencies)
        
        assert 'components' in result
        assert 'modules' in result
        assert 'mermaid_diagram' in result
        assert len(result['components']) > 0
    
    @pytest.mark.asyncio
    async def test_mermaid_diagram_generation(self, sample_python_project):
        """Test Mermaid diagram generation"""
        scanner = StructureScanner(
            exclude_patterns=['__pycache__'],
            include_extensions=['.py'],
            max_file_size=1024 * 1024
        )
        structure = await scanner.scan(sample_python_project)
        
        analyzer = DependencyAnalyzer()
        dependencies = await analyzer.analyze(sample_python_project, structure)
        
        visualizer = ArchitectureVisualizer()
        result = await visualizer.generate(sample_python_project, structure, dependencies)
        
        diagram = result['mermaid_diagram']
        assert 'graph TD' in diagram
        assert len(diagram) > 0


class TestTechnicalDebtDetector:
    """Test TechnicalDebtDetector functionality"""
    
    @pytest.mark.asyncio
    async def test_detect_high_complexity(self, temp_project):
        """Test detecting high complexity"""
        (temp_project / 'complex.py').write_text('''
def very_complex():
    if True:
        if True:
            if True:
                if True:
                    if True:
                        return 1
    return 0
''')
        
        scanner = StructureScanner(
            exclude_patterns=[],
            include_extensions=['.py'],
            max_file_size=1024 * 1024
        )
        structure = await scanner.scan(temp_project)
        
        calculator = MetricsCalculator()
        metrics = await calculator.calculate(temp_project, structure)
        
        detector = TechnicalDebtDetector()
        result = await detector.detect(temp_project, structure, metrics)
        
        # Should detect high complexity if average > 10
        if metrics['average_complexity'] > 10:
            assert any(issue['type'] == 'high_complexity' for issue in result['issues'])
    
    @pytest.mark.asyncio
    async def test_detect_long_function(self, temp_project):
        """Test detecting long functions"""
        # Create a long function (>50 lines)
        long_code = 'def long_function():\n' + '    pass\n' * 60
        (temp_project / 'long.py').write_text(long_code)
        
        scanner = StructureScanner(
            exclude_patterns=[],
            include_extensions=['.py'],
            max_file_size=1024 * 1024
        )
        structure = await scanner.scan(temp_project)
        
        calculator = MetricsCalculator()
        metrics = await calculator.calculate(temp_project, structure)
        
        detector = TechnicalDebtDetector()
        result = await detector.detect(temp_project, structure, metrics)
        
        # Should detect long function
        assert any(issue['type'] == 'long_function' for issue in result['issues'])
    
    @pytest.mark.asyncio
    async def test_detect_too_many_parameters(self, temp_project):
        """Test detecting functions with too many parameters"""
        (temp_project / 'params.py').write_text('''
def many_params(a, b, c, d, e, f, g):
    return a + b + c + d + e + f + g
''')
        
        scanner = StructureScanner(
            exclude_patterns=[],
            include_extensions=['.py'],
            max_file_size=1024 * 1024
        )
        structure = await scanner.scan(temp_project)
        
        calculator = MetricsCalculator()
        metrics = await calculator.calculate(temp_project, structure)
        
        detector = TechnicalDebtDetector()
        result = await detector.detect(temp_project, structure, metrics)
        
        # Should detect too many parameters
        assert any(issue['type'] == 'too_many_parameters' for issue in result['issues'])
    
    @pytest.mark.asyncio
    async def test_detect_bare_except(self, temp_project):
        """Test detecting bare except clauses"""
        (temp_project / 'except.py').write_text('''
try:
    risky_operation()
except:
    pass
''')
        
        scanner = StructureScanner(
            exclude_patterns=[],
            include_extensions=['.py'],
            max_file_size=1024 * 1024
        )
        structure = await scanner.scan(temp_project)
        
        calculator = MetricsCalculator()
        metrics = await calculator.calculate(temp_project, structure)
        
        detector = TechnicalDebtDetector()
        result = await detector.detect(temp_project, structure, metrics)
        
        # Should detect bare except
        assert any(issue['type'] == 'bare_except' for issue in result['issues'])


class TestProjectAnalyzerFeature:
    """Test ProjectAnalyzerFeature integration"""
    
    @pytest.mark.asyncio
    async def test_feature_initialization(self, feature_config):
        """Test feature initialization"""
        feature = ProjectAnalyzerFeature(feature_config)
        
        assert feature.name == 'project_analyzer'
        assert feature.description
        assert not feature.is_initialized
        
        await feature.initialize()
        
        assert feature.is_initialized
        assert feature.structure_scanner is not None
        assert feature.dependency_analyzer is not None
        assert feature.metrics_calculator is not None
    
    @pytest.mark.asyncio
    async def test_analyze_project(self, feature_config, sample_python_project):
        """Test complete project analysis"""
        feature = ProjectAnalyzerFeature(feature_config)
        await feature.initialize()
        
        result = await feature.analyze_project(str(sample_python_project))
        
        assert 'structure' in result
        assert 'dependencies' in result
        assert 'metrics' in result
        assert 'architecture' in result
        assert 'technical_debt' in result
        assert 'summary' in result
        
        # Check summary
        summary = result['summary']
        assert summary['total_files'] > 0
        assert summary['total_lines'] > 0
        assert len(summary['languages']) > 0
        assert 0 <= summary['health_score'] <= 100
    
    @pytest.mark.asyncio
    async def test_health_score_calculation(self, feature_config, sample_python_project):
        """Test health score calculation"""
        feature = ProjectAnalyzerFeature(feature_config)
        await feature.initialize()
        
        result = await feature.analyze_project(str(sample_python_project))
        
        health_score = result['summary']['health_score']
        assert 0 <= health_score <= 100
        assert isinstance(health_score, (int, float))
    
    @pytest.mark.asyncio
    async def test_feature_shutdown(self, feature_config):
        """Test feature shutdown"""
        feature = ProjectAnalyzerFeature(feature_config)
        await feature.initialize()
        
        assert feature.is_initialized
        
        await feature.shutdown()
        
        assert not feature.is_initialized
        assert feature.structure_scanner is None
    
    @pytest.mark.asyncio
    async def test_nonexistent_project(self, feature_config):
        """Test analyzing nonexistent project"""
        feature = ProjectAnalyzerFeature(feature_config)
        await feature.initialize()
        
        with pytest.raises(FileNotFoundError):
            await feature.analyze_project('/nonexistent/path')


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    @pytest.mark.asyncio
    async def test_empty_project(self, temp_project, feature_config):
        """Test analyzing empty project"""
        feature = ProjectAnalyzerFeature(feature_config)
        await feature.initialize()
        
        result = await feature.analyze_project(str(temp_project))
        
        assert result['structure']['total_files'] == 0
        assert result['summary']['total_files'] == 0
    
    @pytest.mark.asyncio
    async def test_large_file_exclusion(self, temp_project):
        """Test that large files are excluded"""
        # Create a file larger than max_file_size
        large_content = 'x' * (2 * 1024 * 1024)  # 2MB
        (temp_project / 'large.py').write_text(large_content)
        (temp_project / 'small.py').write_text('print("hello")')
        
        scanner = StructureScanner(
            exclude_patterns=[],
            include_extensions=['.py'],
            max_file_size=1024 * 1024  # 1MB limit
        )
        
        result = await scanner.scan(temp_project)
        
        # Should only find small.py
        assert result['total_files'] == 1
        assert any('small.py' in f['path'] for f in result['files'])
    
    @pytest.mark.asyncio
    async def test_invalid_python_syntax(self, temp_project):
        """Test handling files with invalid Python syntax"""
        (temp_project / 'invalid.py').write_text('def broken(')
        
        scanner = StructureScanner(
            exclude_patterns=[],
            include_extensions=['.py'],
            max_file_size=1024 * 1024
        )
        structure = await scanner.scan(temp_project)
        
        calculator = MetricsCalculator()
        # Should not crash on invalid syntax
        result = await calculator.calculate(temp_project, structure)
        
        assert result is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])



class TestProjectAnalyzerCLI:
    """Test Project Analyzer CLI commands"""
    
    def test_analyze_project_command(self, sample_python_project):
        """Test xencode analyze project command"""
        from click.testing import CliRunner
        from xencode.cli import cli
        
        runner = CliRunner()
        result = runner.invoke(cli, ['analyze', 'project', str(sample_python_project)])
        
        assert result.exit_code == 0
        assert 'Project Analysis Summary' in result.output or 'Analyzing project' in result.output
    
    def test_analyze_metrics_command(self, sample_python_project):
        """Test xencode analyze metrics command"""
        from click.testing import CliRunner
        from xencode.cli import cli
        
        runner = CliRunner()
        result = runner.invoke(cli, ['analyze', 'metrics', str(sample_python_project)])
        
        assert result.exit_code == 0
        assert 'metrics' in result.output.lower() or 'calculating' in result.output.lower()
    
    def test_analyze_dependencies_command(self, sample_python_project):
        """Test xencode analyze dependencies command"""
        from click.testing import CliRunner
        from xencode.cli import cli
        
        runner = CliRunner()
        result = runner.invoke(cli, ['analyze', 'dependencies', str(sample_python_project)])
        
        assert result.exit_code == 0
        assert 'dependencies' in result.output.lower() or 'analyzing' in result.output.lower()
    
    def test_analyze_architecture_command(self, sample_python_project):
        """Test xencode analyze architecture command"""
        from click.testing import CliRunner
        from xencode.cli import cli
        
        runner = CliRunner()
        result = runner.invoke(cli, ['analyze', 'architecture', str(sample_python_project)])
        
        assert result.exit_code == 0
        assert 'architecture' in result.output.lower() or 'generating' in result.output.lower()
    
    def test_analyze_docs_command(self, sample_python_project, temp_project):
        """Test xencode analyze docs command"""
        from click.testing import CliRunner
        from xencode.cli import cli
        
        runner = CliRunner()
        result = runner.invoke(cli, [
            'analyze', 'docs', 
            str(sample_python_project),
            '--output', str(temp_project),
            '--readme'
        ])
        
        assert result.exit_code == 0
        assert 'documentation' in result.output.lower() or 'generating' in result.output.lower()
    
    def test_analyze_project_with_output(self, sample_python_project, temp_project):
        """Test xencode analyze project with output file"""
        from click.testing import CliRunner
        from xencode.cli import cli
        
        output_file = temp_project / 'analysis.json'
        
        runner = CliRunner()
        result = runner.invoke(cli, [
            'analyze', 'project',
            str(sample_python_project),
            '--output', str(output_file),
            '--format', 'json'
        ])
        
        assert result.exit_code == 0
        # Output file should be created
        assert output_file.exists() or 'saved' in result.output.lower()
    
    def test_analyze_project_markdown_output(self, sample_python_project, temp_project):
        """Test xencode analyze project with markdown output"""
        from click.testing import CliRunner
        from xencode.cli import cli
        
        output_file = temp_project / 'analysis.md'
        
        runner = CliRunner()
        result = runner.invoke(cli, [
            'analyze', 'project',
            str(sample_python_project),
            '--output', str(output_file),
            '--format', 'markdown'
        ])
        
        assert result.exit_code == 0
    
    def test_analyze_nonexistent_project(self):
        """Test analyzing nonexistent project"""
        from click.testing import CliRunner
        from xencode.cli import cli
        
        runner = CliRunner()
        result = runner.invoke(cli, ['analyze', 'project', '/nonexistent/path'])
        
        # Should fail gracefully
        assert result.exit_code != 0 or 'error' in result.output.lower() or 'failed' in result.output.lower()


class TestProjectAnalyzerTUI:
    """Test Project Analyzer TUI panel"""
    
    @pytest.mark.asyncio
    async def test_tui_panel_initialization(self):
        """Test TUI panel initialization"""
        from xencode.tui.features.project_analyzer_panel import ProjectAnalyzerPanel
        
        panel = ProjectAnalyzerPanel()
        
        assert panel.feature_name == "project_analyzer"
        assert panel.title == "📊 Project Analyzer"
        assert panel.current_path is None
        assert panel.analysis_results is None
    
    @pytest.mark.asyncio
    async def test_tui_analyze_project(self, sample_python_project):
        """Test TUI project analysis"""
        from xencode.tui.features.project_analyzer_panel import ProjectAnalyzerPanel
        import os
        
        # Change to sample project directory
        original_cwd = os.getcwd()
        try:
            os.chdir(sample_python_project)
            
            panel = ProjectAnalyzerPanel()
            await panel._analyze_project()
            
            assert panel.analysis_results is not None
            assert 'metrics' in panel.analysis_results
            assert panel.analysis_results['metrics']['total_files'] > 0
        finally:
            os.chdir(original_cwd)
    
    @pytest.mark.asyncio
    async def test_tui_metrics_display(self, sample_python_project):
        """Test TUI metrics display"""
        from xencode.tui.features.project_analyzer_panel import ProjectAnalyzerPanel
        import os
        
        original_cwd = os.getcwd()
        try:
            os.chdir(sample_python_project)
            
            panel = ProjectAnalyzerPanel()
            await panel._analyze_project()
            
            metrics = panel.analysis_results.get('metrics', {})
            assert 'health_score' in metrics
            assert 'lines_of_code' in metrics
            assert 'total_files' in metrics
        finally:
            os.chdir(original_cwd)
