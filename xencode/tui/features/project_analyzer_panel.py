"""Project Analyzer TUI panel."""

from textual.containers import Container, Vertical, Horizontal, ScrollableContainer
from textual.widgets import Button, Label, Tree, Static, DataTable
from textual.reactive import reactive
from pathlib import Path
from typing import Optional, Dict, Any

from .base_feature_panel import BaseFeaturePanel


class ProjectMetricsCard(Static):
    """Card displaying project metrics."""
    
    DEFAULT_CSS = """
    ProjectMetricsCard {
        height: auto;
        padding: 1;
        margin: 1;
        border: solid $accent;
        background: $panel;
    }
    
    .metric-title {
        text-style: bold;
        color: $accent;
    }
    
    .metric-value {
        color: $success;
        text-style: bold;
    }
    """
    
    def __init__(self, title: str, value: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metric_title = title
        self.metric_value = value
    
    def render(self) -> str:
        return f"[bold]{self.metric_title}[/bold]\n{self.metric_value}"


class ProjectAnalyzerPanel(BaseFeaturePanel):
    """Panel for project analysis and documentation generation."""
    
    DEFAULT_CSS = """
    ProjectAnalyzerPanel {
        height: 100%;
    }
    
    .analyzer-controls {
        height: auto;
        padding: 1;
        background: $panel;
    }
    
    .analyzer-results {
        height: 1fr;
        padding: 1;
    }
    
    .metrics-grid {
        height: auto;
        layout: grid;
        grid-size: 2;
        grid-gutter: 1;
    }
    
    .structure-tree {
        height: 1fr;
        border: solid $primary;
        margin-top: 1;
    }
    """
    
    analyzing = reactive(False)
    
    def __init__(self, *args, **kwargs):
        super().__init__(
            feature_name="project_analyzer",
            title="📊 Project Analyzer",
            *args,
            **kwargs
        )
        self.current_path: Optional[Path] = None
        self.analysis_results: Optional[Dict[str, Any]] = None
    
    def compose(self):
        """Compose the project analyzer panel."""
        yield from super().compose()
    
    def on_mount(self) -> None:
        """Initialize panel on mount."""
        self.set_status("enabled")
        self._build_content()
    
    def _build_content(self) -> None:
        """Build the panel content."""
        if not self.content_container:
            return
        
        self.content_container.remove_children()
        
        with self.content_container:
            # Controls
            with Horizontal(classes="analyzer-controls"):
                yield Button("Analyze Project", id="btn-analyze", variant="primary")
                yield Button("Generate Docs", id="btn-generate-docs")
                yield Button("Health Check", id="btn-health")
                yield Button("Metrics", id="btn-metrics")
            
            # Results area
            with ScrollableContainer(classes="analyzer-results"):
                if self.analysis_results:
                    self._render_results()
                else:
                    yield Label(
                        "Select a project directory and click 'Analyze Project' to begin.",
                        classes="feature-empty"
                    )
    
    def _render_results(self) -> None:
        """Render analysis results."""
        if not self.analysis_results:
            return
        
        # Metrics grid
        with Container(classes="metrics-grid"):
            metrics = self.analysis_results.get("metrics", {})
            yield ProjectMetricsCard("Total Files", str(metrics.get("total_files", 0)))
            yield ProjectMetricsCard("Lines of Code", f"{metrics.get('lines_of_code', 0):,}")
            yield ProjectMetricsCard("Languages", str(metrics.get("languages", 0)))
            yield ProjectMetricsCard("Health Score", f"{metrics.get('health_score', 0):.1f}%")
            yield ProjectMetricsCard("Maintainability", f"{metrics.get('maintainability', 0):.1f}/100")
            yield ProjectMetricsCard("Complexity", f"{metrics.get('complexity', 0):.1f}")
            yield ProjectMetricsCard("Dependencies", str(metrics.get("dependencies", 0)))
            yield ProjectMetricsCard("Tech Debt", str(metrics.get("tech_debt", 0)))
        
        # Project structure tree
        structure = self.analysis_results.get("structure", {})
        if structure:
            tree = Tree("Project Structure", classes="structure-tree")
            self._build_tree(tree.root, structure)
            yield tree
    
    def _build_tree(self, node, data: Dict[str, Any]) -> None:
        """Recursively build project structure tree."""
        for key, value in data.items():
            if isinstance(value, dict):
                child = node.add(key)
                self._build_tree(child, value)
            else:
                node.add_leaf(f"{key}: {value}")
    
    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_id = event.button.id
        
        if button_id == "btn-analyze":
            await self._analyze_project()
        elif button_id == "btn-generate-docs":
            await self._generate_docs()
        elif button_id == "btn-health":
            await self._check_health()
        elif button_id == "btn-metrics":
            await self._show_metrics()
    
    async def _analyze_project(self) -> None:
        """Analyze the current project."""
        self.set_status("loading")
        self.analyzing = True
        
        try:
            from xencode.features.project_analyzer import ProjectAnalyzerFeature
            from xencode.features import FeatureConfig
            from pathlib import Path
            
            # Initialize feature
            config = FeatureConfig(name="project_analyzer", enabled=True)
            feature = ProjectAnalyzerFeature(config)
            await feature._initialize()
            
            # Get current project path
            project_path = Path.cwd()
            
            # Analyze project
            results = await feature.analyze_project(str(project_path))
            
            # Transform results for display
            summary = results.get('summary', {})
            metrics = results.get('metrics', {})
            structure = results.get('structure', {})
            
            self.analysis_results = {
                "metrics": {
                    "total_files": summary.get('total_files', 0),
                    "lines_of_code": summary.get('total_lines', 0),
                    "languages": len(summary.get('languages', [])),
                    "health_score": summary.get('health_score', 0),
                    "maintainability": metrics.get('maintainability_index', 0),
                    "complexity": metrics.get('average_complexity', 0),
                    "dependencies": summary.get('dependency_count', 0),
                    "tech_debt": summary.get('tech_debt_items', 0),
                },
                "structure": structure,
                "full_results": results
            }
            
            self._build_content()
            self.set_status("enabled")
            
            await feature._shutdown()
            
        except Exception as e:
            self.show_empty_state(f"Error analyzing project: {e}")
            self.set_status("disabled")
        finally:
            self.analyzing = False
    
    async def _generate_docs(self) -> None:
        """Generate project documentation."""
        self.set_status("loading")
        
        try:
            from xencode.features.project_analyzer import ProjectAnalyzerFeature
            from xencode.features import FeatureConfig
            from pathlib import Path
            
            # Initialize feature
            config = FeatureConfig(name="project_analyzer", enabled=True)
            feature = ProjectAnalyzerFeature(config)
            await feature._initialize()
            
            # Get current project path
            project_path = Path.cwd()
            
            # Analyze project first if not already done
            if not self.analysis_results:
                results = await feature.analyze_project(str(project_path))
            else:
                results = self.analysis_results.get('full_results', {})
            
            # Generate README
            readme_path = project_path / "README.md"
            if not readme_path.exists() or self.confirm_overwrite(readme_path):
                readme_content = self._generate_readme_content(results)
                with open(readme_path, 'w') as f:
                    f.write(readme_content)
            
            self.set_status("enabled")
            await feature._shutdown()
            
        except Exception as e:
            self.show_empty_state(f"Error generating docs: {e}")
            self.set_status("disabled")
    
    def _generate_readme_content(self, results: Dict[str, Any]) -> str:
        """Generate README content from analysis results"""
        summary = results.get('summary', {})
        structure = results.get('structure', {})
        
        return f"""# Project

## Overview

This project contains {summary.get('total_files', 0)} files with {summary.get('total_lines', 0):,} lines of code.

## Languages

{', '.join(summary.get('languages', []))}

## Project Type

{structure.get('project_type', 'Unknown')}

## Metrics

- **Health Score:** {summary.get('health_score', 0):.1f}/100
- **Maintainability:** {summary.get('maintainability_index', 0):.1f}/100
- **Dependencies:** {summary.get('dependency_count', 0)}

## Getting Started

[Add your getting started instructions here]

## License

[Add your license information here]
"""
    
    def confirm_overwrite(self, path: Path) -> bool:
        """Confirm overwrite of existing file"""
        # In TUI, we'll just return True for now
        # In a real implementation, show a confirmation dialog
        return True
    
    async def _check_health(self) -> None:
        """Check project health."""
        self.set_status("loading")
        
        try:
            # Use existing analysis results or analyze now
            if not self.analysis_results:
                await self._analyze_project()
            
            # Display health metrics
            metrics = self.analysis_results.get('metrics', {})
            health_score = metrics.get('health_score', 0)
            
            # Show health status
            if health_score >= 80:
                status = "Excellent"
                color = "green"
            elif health_score >= 60:
                status = "Good"
                color = "yellow"
            else:
                status = "Needs Improvement"
                color = "red"
            
            # In a real TUI, we'd show a modal or update the display
            # For now, just update status
            self.set_status("enabled")
            
        except Exception as e:
            self.show_empty_state(f"Error checking health: {e}")
            self.set_status("disabled")
    
    async def _show_metrics(self) -> None:
        """Show detailed metrics."""
        self.set_status("loading")
        
        try:
            # Use existing analysis results or analyze now
            if not self.analysis_results:
                await self._analyze_project()
            
            # Metrics are already displayed in the main view
            # In a real implementation, we might show a detailed modal
            self.set_status("enabled")
            
        except Exception as e:
            self.show_empty_state(f"Error showing metrics: {e}")
            self.set_status("disabled")
