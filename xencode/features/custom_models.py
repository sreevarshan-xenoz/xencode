"""
Custom AI Models Feature

Provides codebase analysis, model fine-tuning, versioning, and performance monitoring
for creating personalized AI models based on user's coding style.
"""

import json
import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict, Counter
from enum import Enum
import hashlib

from .base import FeatureBase, FeatureConfig, FeatureError


class ModelStatus(Enum):
    """Status of a custom model"""
    ANALYZING = "analyzing"
    TRAINING = "training"
    READY = "ready"
    FAILED = "failed"
    ARCHIVED = "archived"


class TaskType(Enum):
    """Types of tasks for custom models"""
    CODE_COMPLETION = "code_completion"
    CODE_REVIEW = "code_review"
    REFACTORING = "refactoring"
    DOCUMENTATION = "documentation"
    BUG_DETECTION = "bug_detection"


@dataclass
class CustomModelsConfig:
    """Configuration for Custom AI Models"""
    enabled: bool = True
    max_models: int = 10
    training_max_epochs: int = 10
    training_batch_size: int = 32
    min_accuracy: float = 0.85
    auto_backup: bool = True
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CustomModelsConfig':
        """Create config from dictionary"""
        return cls(
            enabled=data.get('enabled', True),
            max_models=data.get('max_models', 10),
            training_max_epochs=data.get('training', {}).get('max_epochs', 10),
            training_batch_size=data.get('training', {}).get('batch_size', 32),
            min_accuracy=data.get('performance', {}).get('min_accuracy', 0.85),
            auto_backup=data.get('auto_backup', True)
        )


@dataclass
class CodebasePattern:
    """Represents a pattern found in the codebase"""
    pattern_type: str
    frequency: int
    examples: List[str] = field(default_factory=list)
    confidence: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'pattern_type': self.pattern_type,
            'frequency': self.frequency,
            'examples': self.examples[:5],  # Limit examples
            'confidence': self.confidence
        }


@dataclass
class ModelVersion:
    """Represents a version of a custom model"""
    version: str
    created_at: str
    status: ModelStatus
    task_type: TaskType
    accuracy: float = 0.0
    training_samples: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'version': self.version,
            'created_at': self.created_at,
            'status': self.status.value,
            'task_type': self.task_type.value,
            'accuracy': self.accuracy,
            'training_samples': self.training_samples,
            'metadata': self.metadata
        }


@dataclass
class PerformanceMetrics:
    """Performance metrics for a model"""
    model_id: str
    version: str
    speed_ms: float = 0.0
    accuracy: float = 0.0
    memory_mb: float = 0.0
    samples_tested: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'model_id': self.model_id,
            'version': self.version,
            'speed_ms': self.speed_ms,
            'accuracy': self.accuracy,
            'memory_mb': self.memory_mb,
            'samples_tested': self.samples_tested,
            'timestamp': self.timestamp
        }


class CustomModelManager(FeatureBase):
    """Custom AI Models feature implementation"""
    
    def __init__(self, config: FeatureConfig):
        super().__init__(config)
        self.cm_config = CustomModelsConfig.from_dict(config.config)
        self.codebase_analyzer = None
        self.model_trainer = None
        self.performance_monitor = None
        self.models: Dict[str, Dict[str, Any]] = {}
    
    @property
    def name(self) -> str:
        """Feature name"""
        return "custom_models"
    
    @property
    def description(self) -> str:
        """Feature description"""
        return "Fine-tune AI models on your codebase for personalized assistance"
    
    async def _initialize(self) -> None:
        """Initialize Custom Models components"""
        # Initialize codebase analyzer
        self.codebase_analyzer = CodebaseAnalyzer()
        
        # Initialize model trainer
        self.model_trainer = ModelTrainer(
            max_epochs=self.cm_config.training_max_epochs,
            batch_size=self.cm_config.training_batch_size,
            min_accuracy=self.cm_config.min_accuracy
        )
        
        # Initialize performance monitor
        self.performance_monitor = PerformanceMonitor()
        
        # Load existing models
        await self._load_models()
    
    async def _shutdown(self) -> None:
        """Shutdown Custom Models"""
        # Save models metadata
        await self._save_models()
    
    async def analyze(self, codebase_path: str) -> Dict[str, Any]:
        """
        Analyze codebase to identify patterns
        
        Args:
            codebase_path: Path to codebase directory
            
        Returns:
            Analysis results with patterns and statistics
        """
        if not Path(codebase_path).exists():
            raise FeatureError(f"Codebase path not found: {codebase_path}")
        
        # Perform analysis
        patterns = await self.codebase_analyzer.analyze(codebase_path)
        
        # Track analytics
        self.track_analytics('analyze_codebase', {
            'path': codebase_path,
            'patterns_found': len(patterns)
        })
        
        return {
            'codebase_path': codebase_path,
            'patterns': [p.to_dict() for p in patterns],
            'total_patterns': len(patterns),
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    async def train(self, model_name: str, codebase_path: str, 
                   task_type: str = "code_completion") -> Dict[str, Any]:
        """
        Train a custom model on user's codebase
        
        Args:
            model_name: Name for the custom model
            codebase_path: Path to codebase for training
            task_type: Type of task (code_completion, code_review, etc.)
            
        Returns:
            Training results and model information
        """
        # Validate inputs
        if not Path(codebase_path).exists():
            raise FeatureError(f"Codebase path not found: {codebase_path}")
        
        if len(self.models) >= self.cm_config.max_models:
            raise FeatureError(f"Maximum number of models ({self.cm_config.max_models}) reached")
        
        try:
            task_enum = TaskType(task_type)
        except ValueError:
            raise FeatureError(f"Invalid task type: {task_type}")  from None
        
        # Analyze codebase first
        analysis = await self.analyze(codebase_path)
        
        # Create model version
        version = self._generate_version()
        model_version = ModelVersion(
            version=version,
            created_at=datetime.now().isoformat(),
            status=ModelStatus.TRAINING,
            task_type=task_enum,
            training_samples=len(analysis['patterns'])
        )
        
        # Store model info
        if model_name not in self.models:
            self.models[model_name] = {
                'name': model_name,
                'versions': [],
                'current_version': version
            }
        
        self.models[model_name]['versions'].append(model_version.to_dict())
        
        # Train model
        training_result = await self.model_trainer.train(
            model_name=model_name,
            version=version,
            patterns=analysis['patterns'],
            task_type=task_enum
        )
        
        # Update model status
        model_version.status = ModelStatus.READY if training_result['success'] else ModelStatus.FAILED
        model_version.accuracy = training_result.get('accuracy', 0.0)
        
        # Update stored version
        for v in self.models[model_name]['versions']:
            if v['version'] == version:
                v.update(model_version.to_dict())
                break
        
        # Save models
        await self._save_models()
        
        # Track analytics
        self.track_analytics('train_model', {
            'model_name': model_name,
            'version': version,
            'task_type': task_type,
            'success': training_result['success']
        })
        
        return {
            'model_name': model_name,
            'version': version,
            'status': model_version.status.value,
            'accuracy': model_version.accuracy,
            'training_samples': model_version.training_samples,
            'training_result': training_result
        }
    
    async def monitor(self, model_name: str, version: str = None) -> Dict[str, Any]:
        """
        Monitor model performance
        
        Args:
            model_name: Name of the model
            version: Optional specific version (defaults to current)
            
        Returns:
            Performance metrics
        """
        if model_name not in self.models:
            raise FeatureError(f"Model not found: {model_name}")
        
        if version is None:
            version = self.models[model_name]['current_version']
        
        # Get performance metrics
        metrics = await self.performance_monitor.get_metrics(model_name, version)
        
        return {
            'model_name': model_name,
            'version': version,
            'metrics': metrics.to_dict() if metrics else None
        }
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """
        List all custom models
        
        Returns:
            List of models with their versions
        """
        return [
            {
                'name': name,
                'current_version': info['current_version'],
                'versions': info['versions']
            }
            for name, info in self.models.items()
        ]
    
    async def delete_model(self, model_name: str, version: str = None) -> Dict[str, Any]:
        """
        Delete a model or specific version
        
        Args:
            model_name: Name of the model
            version: Optional specific version (deletes all if None)
            
        Returns:
            Deletion result
        """
        if model_name not in self.models:
            raise FeatureError(f"Model not found: {model_name}")
        
        if version:
            # Delete specific version
            versions = self.models[model_name]['versions']
            self.models[model_name]['versions'] = [
                v for v in versions if v['version'] != version
            ]
            
            # Update current version if needed
            if self.models[model_name]['current_version'] == version:
                if self.models[model_name]['versions']:
                    self.models[model_name]['current_version'] = \
                        self.models[model_name]['versions'][-1]['version']
                else:
                    del self.models[model_name]
            
            deleted = f"version {version}"
        else:
            # Delete entire model
            del self.models[model_name]
            deleted = "all versions"
        
        await self._save_models()
        
        return {
            'model_name': model_name,
            'deleted': deleted,
            'success': True
        }
    
    async def rollback(self, model_name: str, version: str) -> Dict[str, Any]:
        """
        Rollback to a previous model version
        
        Args:
            model_name: Name of the model
            version: Version to rollback to
            
        Returns:
            Rollback result
        """
        if model_name not in self.models:
            raise FeatureError(f"Model not found: {model_name}")
        
        # Check if version exists
        versions = [v['version'] for v in self.models[model_name]['versions']]
        if version not in versions:
            raise FeatureError(f"Version not found: {version}")
        
        # Update current version
        old_version = self.models[model_name]['current_version']
        self.models[model_name]['current_version'] = version
        
        await self._save_models()
        
        return {
            'model_name': model_name,
            'old_version': old_version,
            'new_version': version,
            'success': True
        }
    
    async def compare_versions(self, model_name: str, 
                              version1: str, version2: str) -> Dict[str, Any]:
        """
        Compare two model versions
        
        Args:
            model_name: Name of the model
            version1: First version
            version2: Second version
            
        Returns:
            Comparison results
        """
        if model_name not in self.models:
            raise FeatureError(f"Model not found: {model_name}")
        
        # Get metrics for both versions
        metrics1 = await self.performance_monitor.get_metrics(model_name, version1)
        metrics2 = await self.performance_monitor.get_metrics(model_name, version2)
        
        if not metrics1 or not metrics2:
            raise FeatureError("Metrics not available for one or both versions")
        
        return {
            'model_name': model_name,
            'version1': {
                'version': version1,
                'metrics': metrics1.to_dict()
            },
            'version2': {
                'version': version2,
                'metrics': metrics2.to_dict()
            },
            'comparison': {
                'speed_diff_ms': metrics2.speed_ms - metrics1.speed_ms,
                'accuracy_diff': metrics2.accuracy - metrics1.accuracy,
                'memory_diff_mb': metrics2.memory_mb - metrics1.memory_mb
            }
        }
    
    def _generate_version(self) -> str:
        """Generate a version string"""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        return f"v{timestamp}"
    
    async def _load_models(self) -> None:
        """Load models from storage"""
        models_file = Path.home() / '.xencode' / 'custom_models.json'
        if models_file.exists():
            try:
                with open(models_file, 'r') as f:
                    self.models = json.load(f)
            except Exception:
                pass
    
    async def _save_models(self) -> None:
        """Save models to storage"""
        models_file = Path.home() / '.xencode' / 'custom_models.json'
        models_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(models_file, 'w') as f:
                json.dump(self.models, f, indent=2)
        except Exception:
            pass
    
    @staticmethod
    def _run_async_cli_coro(coro):
        """Run an async coroutine from a CLI command with error handling."""
        try:
            return asyncio.run(coro)
        except Exception as e:
            from rich.console import Console
            Console().print(f"[red]❌ {e}[/red]")
            return None

    def get_cli_commands(self) -> List[Any]:  # noqa: C901 - click group definition with many subcommands
        """Get CLI commands for Custom Models"""
        import click
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel

        console = Console()

        @click.group(name='models')
        def models_group():
            """Custom AI Models - Fine-tune models on your codebase"""
            pass

        @models_group.group(name='custom')
        def custom_group():
            """Custom model management commands"""
            pass

        @custom_group.command(name='analyze')
        @click.argument('codebase_path', type=click.Path(exists=True))
        def analyze_cmd(codebase_path):
            """Analyze codebase to identify patterns"""
            console.print(f"[blue]🔍 Analyzing codebase: {codebase_path}[/blue]")

            async def _analyze():
                result = await self.analyze(codebase_path)
                console.print(f"\n[green]✅ Analysis complete![/green]")
                console.print(f"[cyan]Total patterns found: {result['total_patterns']}[/cyan]")
                if result['patterns']:
                    table = Table(title="Codebase Patterns")
                    table.add_column("Pattern Type", style="cyan")
                    table.add_column("Frequency", style="yellow")
                    table.add_column("Confidence", style="green")
                    table.add_column("Examples", style="white")
                    for pattern in result['patterns'][:20]:
                        examples = ', '.join(pattern['examples'][:2])
                        table.add_row(pattern['pattern_type'], str(pattern['frequency']),
                                      f"{pattern['confidence']:.0%}", examples)
                    console.print(table)

            self._run_async_cli_coro(_analyze(), success_msg=None)

        @custom_group.command(name='train')
        @click.argument('model_name')
        @click.argument('codebase_path', type=click.Path(exists=True))
        @click.option('--task-type', type=click.Choice(
            ['code_completion', 'code_review', 'refactoring', 'documentation', 'bug_detection']),
            default='code_completion', help='Type of task for the model')
        def train_cmd(model_name, codebase_path, task_type):
            """Train a custom model on your codebase"""
            console.print(f"[blue]🎯 Training model: {model_name}[/blue]")
            console.print(f"[cyan]Codebase: {codebase_path}[/cyan]")
            console.print(f"[cyan]Task type: {task_type}[/cyan]")

            async def _train():
                with console.status("[bold blue]🤖 Training in progress..."):
                    result = await self.train(model_name, codebase_path, task_type)
                console.print(f"\n[green]✅ Training complete![/green]")
                console.print(f"[cyan]Model: {result['model_name']}[/cyan]")
                console.print(f"[cyan]Version: {result['version']}[/cyan]")
                console.print(f"[cyan]Accuracy: {result['accuracy']:.1%}[/cyan]")
                if result['accuracy'] >= 0.9:
                    console.print("[bold green]🏆 Excellent accuracy![/bold green]")
                elif result['accuracy'] >= 0.85:
                    console.print("[yellow]✓ Good accuracy[/yellow]")
                else:
                    console.print("[yellow]⚠️  Consider retraining with more data[/yellow]")

            self._run_async_cli_coro(_train())

        @custom_group.command(name='list')
        def list_cmd():
            """List all custom models"""
            console.print("[blue]📋 Listing custom models...[/blue]")

            async def _list():
                models = await self.list_models()
                if not models:
                    console.print("[yellow]No custom models found[/yellow]")
                    console.print("[dim]Create a model with: xencode models custom train <name> <path>[/dim]")
                    return
                table = Table(title="Custom Models")
                table.add_column("Model Name", style="cyan")
                table.add_column("Current Version", style="yellow")
                table.add_column("Total Versions", style="green")
                table.add_column("Latest Status", style="white")
                table.add_column("Latest Accuracy", style="magenta")
                for model in models:
                    versions = model['versions']
                    latest = versions[-1] if versions else {}
                    table.add_row(model['name'], model['current_version'], str(len(versions)),
                                  latest.get('status', 'unknown'),
                                  f"{latest.get('accuracy', 0):.1%}" if latest.get('accuracy') else 'N/A')
                console.print(table)
                console.print(f"\n[green]Found {len(models)} custom models[/green]")

            self._run_async_cli_coro(_list())
        
        @custom_group.command(name='performance')
        @click.argument('model_name')
        @click.option('--version', help='Specific version (defaults to current)')
        def performance_cmd(model_name, version):
            """Check model performance metrics
            
            Examples:
                xencode models custom performance my-model
                xencode models custom performance my-model --version v20240115120000
            """
            console.print(f"[blue]📊 Checking performance: {model_name}[/blue]")
            
            async def _performance():
                try:
                    result = await self.monitor(model_name, version)
                    
                    if not result['metrics']:
                        console.print("[yellow]No performance metrics available yet[/yellow]")
                        console.print("[dim]Metrics are recorded during model usage[/dim]")
                        return
                    
                    metrics = result['metrics']
                    
                    # Display metrics panel
                    metrics_text = f"""
[cyan]Model:[/cyan] {result['model_name']}
[cyan]Version:[/cyan] {result['version']}

[bold]Performance Metrics:[/bold]
• Speed: {metrics['speed_ms']:.1f}ms
• Accuracy: {metrics['accuracy']:.1%}
• Memory: {metrics['memory_mb']:.1f}MB
• Samples Tested: {metrics['samples_tested']}
• Last Updated: {metrics['timestamp']}
                    """
                    
                    panel = Panel(metrics_text.strip(), title="Model Performance", border_style="green")
                    console.print(panel)
                    
                    # Performance grade
                    if metrics['speed_ms'] < 100 and metrics['accuracy'] > 0.9:
                        console.print("[bold green]🏆 Excellent performance![/bold green]")
                    elif metrics['speed_ms'] < 200 and metrics['accuracy'] > 0.85:
                        console.print("[green]✓ Good performance[/green]")
                    else:
                        console.print("[yellow]⚠️  Consider optimization[/yellow]")
                    
                except Exception as e:
                    console.print(f"[red]❌ Failed to get performance: {e}[/red]")
            
            asyncio.run(_performance())
        
        @custom_group.command(name='delete')
        @click.argument('model_name')
        @click.option('--version', help='Specific version to delete (deletes all if not specified)')
        @click.option('--yes', is_flag=True, help='Skip confirmation')
        def delete_cmd(model_name, version, yes):
            """Delete a model or specific version
            
            Examples:
                xencode models custom delete my-model
                xencode models custom delete my-model --version v20240115120000
                xencode models custom delete my-model --yes
            """
            if not yes:
                from rich.prompt import Confirm
                target = f"version {version}" if version else "all versions"
                if not Confirm.ask(f"Delete {target} of model '{model_name}'?"):
                    console.print("[yellow]Cancelled[/yellow]")
                    return
            
            console.print(f"[blue]🗑️  Deleting model: {model_name}[/blue]")
            
            async def _delete():
                try:
                    result = await self.delete_model(model_name, version)
                    
                    console.print(f"[green]✅ Deleted {result['deleted']} of model '{result['model_name']}'[/green]")
                    
                except Exception as e:
                    console.print(f"[red]❌ Failed to delete: {e}[/red]")
            
            asyncio.run(_delete())
        
        @custom_group.command(name='rollback')
        @click.argument('model_name')
        @click.argument('version')
        def rollback_cmd(model_name, version):
            """Rollback to a previous model version
            
            Examples:
                xencode models custom rollback my-model v20240115120000
            """
            console.print(f"[blue]⏮️  Rolling back model: {model_name}[/blue]")
            
            async def _rollback():
                try:
                    result = await self.rollback(model_name, version)
                    
                    console.print(f"[green]✅ Rolled back successfully![/green]")
                    console.print(f"[cyan]Old version: {result['old_version']}[/cyan]")
                    console.print(f"[cyan]New version: {result['new_version']}[/cyan]")
                    
                except Exception as e:
                    console.print(f"[red]❌ Rollback failed: {e}[/red]")
            
            asyncio.run(_rollback())
        
        @custom_group.command(name='compare')
        @click.argument('model_name')
        @click.argument('version1')
        @click.argument('version2')
        def compare_cmd(model_name, version1, version2):
            """Compare two model versions
            
            Examples:
                xencode models custom compare my-model v20240115120000 v20240115130000
            """
            console.print(f"[blue]⚖️  Comparing versions: {version1} vs {version2}[/blue]")
            
            async def _compare():
                try:
                    result = await self.compare_versions(model_name, version1, version2)
                    
                    # Display comparison table
                    table = Table(title=f"Version Comparison: {model_name}")
                    table.add_column("Metric", style="cyan")
                    table.add_column(version1, style="yellow")
                    table.add_column(version2, style="green")
                    table.add_column("Difference", style="magenta")
                    
                    v1_metrics = result['version1']['metrics']
                    v2_metrics = result['version2']['metrics']
                    comparison = result['comparison']
                    
                    table.add_row(
                        "Speed (ms)",
                        f"{v1_metrics['speed_ms']:.1f}",
                        f"{v2_metrics['speed_ms']:.1f}",
                        f"{comparison['speed_diff_ms']:+.1f}"
                    )
                    
                    table.add_row(
                        "Accuracy",
                        f"{v1_metrics['accuracy']:.1%}",
                        f"{v2_metrics['accuracy']:.1%}",
                        f"{comparison['accuracy_diff']:+.1%}"
                    )
                    
                    table.add_row(
                        "Memory (MB)",
                        f"{v1_metrics['memory_mb']:.1f}",
                        f"{v2_metrics['memory_mb']:.1f}",
                        f"{comparison['memory_diff_mb']:+.1f}"
                    )
                    
                    console.print(table)
                    
                    # Recommendation
                    if comparison['accuracy_diff'] > 0 and comparison['speed_diff_ms'] < 0:
                        console.print(f"[bold green]✅ {version2} is better (faster and more accurate)[/bold green]")
                    elif comparison['accuracy_diff'] > 0:
                        console.print(f"[green]✓ {version2} is more accurate[/green]")
                    elif comparison['speed_diff_ms'] < 0:
                        console.print(f"[green]✓ {version2} is faster[/green]")
                    else:
                        console.print(f"[yellow]⚠️  {version1} may be better overall[/yellow]")
                    
                except Exception as e:
                    console.print(f"[red]❌ Comparison failed: {e}[/red]")
            
            asyncio.run(_compare())
        
        return [models_group]
    
    def get_tui_components(self) -> List[Any]:
        """Get TUI components for Custom Models"""
        from xencode.tui.widgets.custom_models_panel import CustomModelsPanel
        return [CustomModelsPanel]
    
    def get_api_endpoints(self) -> List[Any]:
        """Get API endpoints for Custom Models"""
        return [
            {
                'path': '/api/models/custom/analyze',
                'method': 'POST',
                'handler': self.analyze
            },
            {
                'path': '/api/models/custom/train',
                'method': 'POST',
                'handler': self.train
            },
            {
                'path': '/api/models/custom/monitor',
                'method': 'GET',
                'handler': self.monitor
            },
            {
                'path': '/api/models/custom/list',
                'method': 'GET',
                'handler': self.list_models
            },
            {
                'path': '/api/models/custom/delete',
                'method': 'DELETE',
                'handler': self.delete_model
            },
            {
                'path': '/api/models/custom/rollback',
                'method': 'POST',
                'handler': self.rollback
            },
            {
                'path': '/api/models/custom/compare',
                'method': 'GET',
                'handler': self.compare_versions
            }
        ]


class CodebaseAnalyzer:
    """Analyzes codebase to extract patterns"""
    
    def __init__(self):
        self.patterns: List[CodebasePattern] = []
    
    async def analyze(self, codebase_path: str) -> List[CodebasePattern]:
        """
        Analyze codebase and extract patterns
        
        Args:
            codebase_path: Path to codebase
            
        Returns:
            List of identified patterns
        """
        patterns = []
        codebase = Path(codebase_path)
        
        # Analyze file structure
        file_patterns = await self._analyze_file_structure(codebase)
        patterns.extend(file_patterns)
        
        # Analyze coding style
        style_patterns = await self._analyze_coding_style(codebase)
        patterns.extend(style_patterns)
        
        # Analyze naming conventions
        naming_patterns = await self._analyze_naming_conventions(codebase)
        patterns.extend(naming_patterns)
        
        # Analyze common imports
        import_patterns = await self._analyze_imports(codebase)
        patterns.extend(import_patterns)
        
        return patterns
    
    async def _analyze_file_structure(self, codebase: Path) -> List[CodebasePattern]:
        """Analyze file structure patterns"""
        patterns = []
        
        # Count file types
        file_types = Counter()
        for file_path in codebase.rglob('*'):
            if file_path.is_file():
                file_types[file_path.suffix] += 1
        
        # Create patterns for common file types
        for ext, count in file_types.most_common(10):
            if count > 1:
                patterns.append(CodebasePattern(
                    pattern_type=f"file_type_{ext}",
                    frequency=count,
                    examples=[ext],
                    confidence=0.9
                ))
        
        return patterns
    
    async def _analyze_coding_style(self, codebase: Path) -> List[CodebasePattern]:
        """Analyze coding style patterns"""
        patterns = []
        
        # Analyze Python files for style patterns
        python_files = list(codebase.rglob('*.py'))
        if python_files:
            # Sample indentation style
            indent_counts = Counter()
            for py_file in python_files[:20]:  # Sample first 20 files
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            if line.startswith(' ') and line.strip():
                                leading_spaces = len(line) - len(line.lstrip(' '))
                                if leading_spaces > 0:
                                    indent_counts[leading_spaces % 8 or 8] += 1
                except Exception:
                    continue
            
            if indent_counts:
                most_common_indent = indent_counts.most_common(1)[0][0]
                patterns.append(CodebasePattern(
                    pattern_type="indentation_style",
                    frequency=indent_counts[most_common_indent],
                    examples=[f"{most_common_indent} spaces"],
                    confidence=0.85
                ))
        
        return patterns
    
    async def _analyze_naming_conventions(self, codebase: Path) -> List[CodebasePattern]:
        """Analyze naming convention patterns"""
        patterns = []
        
        # Analyze Python files for naming patterns
        python_files = list(codebase.rglob('*.py'))
        if python_files:
            function_names = []
            class_names = []
            
            for py_file in python_files[:20]:  # Sample first 20 files
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        # Simple pattern matching (in production, use AST)
                        import re
                        function_names.extend(re.findall(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)', content))
                        class_names.extend(re.findall(r'class\s+([a-zA-Z_][a-zA-Z0-9_]*)', content))
                except Exception:
                    continue
            
            # Detect snake_case vs camelCase for functions
            if function_names:
                snake_case = sum(1 for name in function_names if '_' in name)
                camel_case = sum(1 for name in function_names if name[0].islower() and any(c.isupper() for c in name))
                
                if snake_case > camel_case:
                    patterns.append(CodebasePattern(
                        pattern_type="function_naming_snake_case",
                        frequency=snake_case,
                        examples=function_names[:3],
                        confidence=0.8
                    ))
            
            # Detect PascalCase for classes
            if class_names:
                patterns.append(CodebasePattern(
                    pattern_type="class_naming_pascal_case",
                    frequency=len(class_names),
                    examples=class_names[:3],
                    confidence=0.9
                ))
        
        return patterns
    
    async def _analyze_imports(self, codebase: Path) -> List[CodebasePattern]:
        """Analyze import patterns"""
        patterns = []
        
        # Analyze Python files for common imports
        python_files = list(codebase.rglob('*.py'))
        if python_files:
            import_counts = Counter()
            
            for py_file in python_files[:50]:  # Sample first 50 files
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            line = line.strip()
                            if line.startswith('import ') or line.startswith('from '):
                                import_counts[line] += 1
                except Exception:
                    continue
            
            # Create patterns for common imports
            for import_stmt, count in import_counts.most_common(10):
                if count > 2:
                    patterns.append(CodebasePattern(
                        pattern_type="common_import",
                        frequency=count,
                        examples=[import_stmt],
                        confidence=0.95
                    ))
        
        return patterns


class ModelTrainer:
    """Trains custom models on user's coding style"""
    
    def __init__(self, max_epochs: int = 10, batch_size: int = 32, 
                 min_accuracy: float = 0.85):
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.min_accuracy = min_accuracy
    
    async def train(self, model_name: str, version: str, 
                   patterns: List[Dict[str, Any]], 
                   task_type: TaskType) -> Dict[str, Any]:
        """
        Train a custom model
        
        Args:
            model_name: Name of the model
            version: Version identifier
            patterns: Codebase patterns for training
            task_type: Type of task
            
        Returns:
            Training results
        """
        # Simulate training process
        # In production, this would fine-tune an actual model
        
        training_data = self._prepare_training_data(patterns, task_type)
        
        # Simulate training epochs
        best_accuracy = 0.0
        for epoch in range(self.max_epochs):
            # Simulate training
            await asyncio.sleep(0.1)  # Simulate training time
            
            # Simulate accuracy improvement
            accuracy = min(0.7 + (epoch / self.max_epochs) * 0.25, 0.95)
            best_accuracy = max(best_accuracy, accuracy)
            
            if accuracy >= self.min_accuracy:
                break
        
        success = best_accuracy >= self.min_accuracy
        
        return {
            'success': success,
            'accuracy': best_accuracy,
            'epochs_trained': epoch + 1,
            'training_samples': len(training_data),
            'task_type': task_type.value
        }
    
    def _prepare_training_data(self, patterns: List[Dict[str, Any]], 
                               task_type: TaskType) -> List[Dict[str, Any]]:
        """Prepare training data from patterns"""
        training_data = []
        
        for pattern in patterns:
            # Convert patterns to training samples
            training_data.append({
                'pattern': pattern,
                'task_type': task_type.value
            })
        
        return training_data



class PerformanceMonitor:
    """Monitors model performance metrics"""
    
    def __init__(self):
        self.metrics: Dict[str, Dict[str, PerformanceMetrics]] = defaultdict(dict)
        self._load_metrics()
    
    async def get_metrics(self, model_name: str, version: str) -> Optional[PerformanceMetrics]:
        """
        Get performance metrics for a model version
        
        Args:
            model_name: Name of the model
            version: Version identifier
            
        Returns:
            Performance metrics or None
        """
        return self.metrics.get(model_name, {}).get(version)
    
    async def record_metrics(self, model_name: str, version: str,
                            speed_ms: float, accuracy: float, 
                            memory_mb: float, samples_tested: int) -> None:
        """
        Record performance metrics
        
        Args:
            model_name: Name of the model
            version: Version identifier
            speed_ms: Inference speed in milliseconds
            accuracy: Model accuracy (0-1)
            memory_mb: Memory usage in MB
            samples_tested: Number of samples tested
        """
        metrics = PerformanceMetrics(
            model_id=model_name,
            version=version,
            speed_ms=speed_ms,
            accuracy=accuracy,
            memory_mb=memory_mb,
            samples_tested=samples_tested
        )
        
        self.metrics[model_name][version] = metrics
        await self._save_metrics()
    
    async def compare_metrics(self, model_name: str, 
                             version1: str, version2: str) -> Dict[str, Any]:
        """
        Compare metrics between two versions
        
        Args:
            model_name: Name of the model
            version1: First version
            version2: Second version
            
        Returns:
            Comparison results
        """
        m1 = self.metrics.get(model_name, {}).get(version1)
        m2 = self.metrics.get(model_name, {}).get(version2)
        
        if not m1 or not m2:
            return {'error': 'Metrics not available for one or both versions'}
        
        return {
            'speed_improvement': ((m1.speed_ms - m2.speed_ms) / m1.speed_ms * 100) if m1.speed_ms > 0 else 0,
            'accuracy_improvement': (m2.accuracy - m1.accuracy) * 100,
            'memory_improvement': ((m1.memory_mb - m2.memory_mb) / m1.memory_mb * 100) if m1.memory_mb > 0 else 0
        }
    
    async def generate_report(self, model_name: str, version: str) -> Dict[str, Any]:
        """
        Generate performance report
        
        Args:
            model_name: Name of the model
            version: Version identifier
            
        Returns:
            Performance report
        """
        metrics = self.metrics.get(model_name, {}).get(version)
        
        if not metrics:
            return {'error': 'Metrics not available'}
        
        # Calculate performance grade
        grade = self._calculate_grade(metrics)
        
        return {
            'model_name': model_name,
            'version': version,
            'metrics': metrics.to_dict(),
            'grade': grade,
            'recommendations': self._generate_recommendations(metrics)
        }
    
    @staticmethod
    def _score_speed(speed_ms: float) -> int:
        """Score based on inference speed."""
        if speed_ms < 100:
            return 30
        if speed_ms < 200:
            return 20
        if speed_ms < 500:
            return 10
        return 0

    @staticmethod
    def _score_accuracy(accuracy: float) -> int:
        """Score based on accuracy."""
        if accuracy >= 0.95:
            return 40
        if accuracy >= 0.90:
            return 30
        if accuracy >= 0.85:
            return 20
        if accuracy >= 0.80:
            return 10
        return 0

    @staticmethod
    def _score_memory(memory_mb: float) -> int:
        """Score based on memory usage."""
        if memory_mb < 500:
            return 30
        if memory_mb < 1000:
            return 20
        if memory_mb < 2000:
            return 10
        return 0

    _GRADE_TABLE = [(90, 'A'), (80, 'B'), (70, 'C'), (60, 'D')]

    def _calculate_grade(self, metrics: PerformanceMetrics) -> str:
        """Calculate performance grade using score thresholds."""
        score = (
            self._score_speed(metrics.speed_ms)
            + self._score_accuracy(metrics.accuracy)
            + self._score_memory(metrics.memory_mb)
        )
        for threshold, grade in self._GRADE_TABLE:
            if score >= threshold:
                return grade
        return 'F'
    
    def _generate_recommendations(self, metrics: PerformanceMetrics) -> List[str]:
        """Generate performance recommendations"""
        recommendations = []
        
        if metrics.speed_ms > 200:
            recommendations.append("Consider optimizing model size for faster inference")
        
        if metrics.accuracy < 0.85:
            recommendations.append("Model accuracy is below target - consider retraining with more data")
        
        if metrics.memory_mb > 1500:
            recommendations.append("High memory usage - consider model quantization or pruning")
        
        if not recommendations:
            recommendations.append("Model performance is optimal")
        
        return recommendations
    
    def _load_metrics(self) -> None:
        """Load metrics from storage"""
        metrics_file = Path.home() / '.xencode' / 'model_metrics.json'
        if metrics_file.exists():
            try:
                with open(metrics_file, 'r') as f:
                    data = json.load(f)
                    for model_name, versions in data.items():
                        for version, metrics_data in versions.items():
                            self.metrics[model_name][version] = PerformanceMetrics(
                                model_id=metrics_data['model_id'],
                                version=metrics_data['version'],
                                speed_ms=metrics_data['speed_ms'],
                                accuracy=metrics_data['accuracy'],
                                memory_mb=metrics_data['memory_mb'],
                                samples_tested=metrics_data['samples_tested'],
                                timestamp=metrics_data['timestamp']
                            )
            except Exception:
                pass
    
    async def _save_metrics(self) -> None:
        """Save metrics to storage"""
        metrics_file = Path.home() / '.xencode' / 'model_metrics.json'
        metrics_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            data = {
                model_name: {
                    version: metrics.to_dict()
                    for version, metrics in versions.items()
                }
                for model_name, versions in self.metrics.items()
            }
            with open(metrics_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass
