#!/usr/bin/env python3
"""
Custom Models Panel Widget for Xencode TUI

Interactive custom models interface with training dashboard, performance viewer,
version history, and codebase analysis results display.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime

from textual.app import ComposeResult
from textual.containers import Container, Vertical, Horizontal, ScrollableContainer
from textual.widgets import Static, Label, Button, Input, ListView, ListItem, ProgressBar, DataTable
from textual.reactive import reactive
from textual.message import Message
from textual.binding import Binding
from rich.text import Text
from rich.panel import Panel


class TrainingDashboard(Container):
    """Dashboard showing model training progress and status"""
    
    DEFAULT_CSS = """
    TrainingDashboard {
        height: 100%;
        border: solid $primary;
        background: $surface;
        padding: 1;
    }
    
    #training-header {
        dock: top;
        height: 3;
        background: $primary;
        padding: 0 2;
    }
    
    #training-content {
        height: 1fr;
        padding: 1;
    }
    
    .training-row {
        height: auto;
        padding: 1;
        margin: 0 1;
    }
    
    .training-label {
        width: 20;
    }
    
    .training-value {
        width: 1fr;
    }
    
    .progress-container {
        height: 3;
        margin: 1 0;
    }
    """
    
    status = reactive("idle")
    progress = reactive(0.0)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.training_info = {
            'model_name': 'N/A',
            'version': 'N/A',
            'task_type': 'N/A',
            'epoch': 0,
            'max_epochs': 10,
            'accuracy': 0.0,
            'samples': 0
        }
    
    def compose(self) -> ComposeResult:
        """Compose the training dashboard"""
        yield Label("🎯 Training Dashboard", id="training-header")
        
        with ScrollableContainer(id="training-content"):
            # Model info
            with Horizontal(classes="training-row"):
                yield Label("Model Name:", classes="training-label")
                yield Label(self.training_info['model_name'], id="training-model-name", classes="training-value")
            
            with Horizontal(classes="training-row"):
                yield Label("Version:", classes="training-label")
                yield Label(self.training_info['version'], id="training-version", classes="training-value")
            
            with Horizontal(classes="training-row"):
                yield Label("Task Type:", classes="training-label")
                yield Label(self.training_info['task_type'], id="training-task-type", classes="training-value")
            
            # Training progress
            with Container(classes="progress-container"):
                yield Label("Training Progress:", classes="dim")
                yield ProgressBar(total=100, show_eta=True, id="training-progress")
            
            # Training metrics
            with Horizontal(classes="training-row"):
                yield Label("Epoch:", classes="training-label")
                yield Label(f"{self.training_info['epoch']}/{self.training_info['max_epochs']}", 
                          id="training-epoch", classes="training-value")
            
            with Horizontal(classes="training-row"):
                yield Label("Accuracy:", classes="training-label")
                yield Label(f"{self.training_info['accuracy']:.1%}", id="training-accuracy", classes="training-value")
            
            with Horizontal(classes="training-row"):
                yield Label("Training Samples:", classes="training-label")
                yield Label(str(self.training_info['samples']), id="training-samples", classes="training-value")
            
            # Status
            with Horizontal(classes="training-row"):
                yield Label("Status:", classes="training-label")
                yield Label(self.status, id="training-status", classes="training-value")
    
    def update_training_info(self, info: Dict[str, Any]) -> None:
        """Update training information"""
        self.training_info.update(info)
        
        # Update labels
        try:
            self.query_one("#training-model-name", Label).update(str(info.get('model_name', 'N/A')))
            self.query_one("#training-version", Label).update(str(info.get('version', 'N/A')))
            self.query_one("#training-task-type", Label).update(str(info.get('task_type', 'N/A')))
            self.query_one("#training-epoch", Label).update(
                f"{info.get('epoch', 0)}/{info.get('max_epochs', 10)}"
            )
            self.query_one("#training-accuracy", Label).update(f"{info.get('accuracy', 0):.1%}")
            self.query_one("#training-samples", Label).update(str(info.get('samples', 0)))
        except Exception:
                pass  # Silently ignore

    def update_progress(self, progress: float, status: str = None) -> None:
        """Update training progress"""
        self.progress = progress
        
        try:
            progress_bar = self.query_one("#training-progress", ProgressBar)
            progress_bar.update(progress=int(progress * 100))
        except Exception:
                pass  # Silently ignore

        if status:
            self.status = status
            try:
                self.query_one("#training-status", Label).update(status)
            except Exception:
                    pass  # Silently ignore

class PerformanceViewer(Container):
    """Viewer for model performance metrics"""
    
    DEFAULT_CSS = """
    PerformanceViewer {
        height: 100%;
        border: solid $accent;
        background: $surface;
        padding: 1;
    }
    
    #performance-header {
        dock: top;
        height: 3;
        background: $accent;
        padding: 0 2;
    }
    
    #performance-content {
        height: 1fr;
    }
    
    .metric-card {
        height: 8;
        border: solid $primary;
        background: $panel;
        padding: 1;
        margin: 1;
    }
    
    .metric-title {
        height: 1;
        text-style: bold;
    }
    
    .metric-value {
        height: 3;
        content-align: center middle;
        text-style: bold;
    }
    
    .metric-label {
        height: 1;
        content-align: center middle;
        text-style: dim;
    }
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.metrics = {
            'speed_ms': 0.0,
            'accuracy': 0.0,
            'memory_mb': 0.0,
            'samples_tested': 0
        }
    
    def compose(self) -> ComposeResult:
        """Compose the performance viewer"""
        yield Label("📊 Performance Metrics", id="performance-header")
        
        with ScrollableContainer(id="performance-content"):
            # Speed metric
            with Container(classes="metric-card"):
                yield Label("⚡ Speed", classes="metric-title")
                yield Label(f"{self.metrics['speed_ms']:.1f}ms", id="metric-speed", classes="metric-value")
                yield Label("Inference Time", classes="metric-label")
            
            # Accuracy metric
            with Container(classes="metric-card"):
                yield Label("🎯 Accuracy", classes="metric-title")
                yield Label(f"{self.metrics['accuracy']:.1%}", id="metric-accuracy", classes="metric-value")
                yield Label("Model Accuracy", classes="metric-label")
            
            # Memory metric
            with Container(classes="metric-card"):
                yield Label("💾 Memory", classes="metric-title")
                yield Label(f"{self.metrics['memory_mb']:.1f}MB", id="metric-memory", classes="metric-value")
                yield Label("Memory Usage", classes="metric-label")
            
            # Samples metric
            with Container(classes="metric-card"):
                yield Label("📈 Samples", classes="metric-title")
                yield Label(str(self.metrics['samples_tested']), id="metric-samples", classes="metric-value")
                yield Label("Samples Tested", classes="metric-label")
    
    def update_metrics(self, metrics: Dict[str, Any]) -> None:
        """Update performance metrics"""
        self.metrics.update(metrics)
        
        try:
            self.query_one("#metric-speed", Label).update(f"{metrics.get('speed_ms', 0):.1f}ms")
            self.query_one("#metric-accuracy", Label).update(f"{metrics.get('accuracy', 0):.1%}")
            self.query_one("#metric-memory", Label).update(f"{metrics.get('memory_mb', 0):.1f}MB")
            self.query_one("#metric-samples", Label).update(str(metrics.get('samples_tested', 0)))
        except Exception:
                pass  # Silently ignore

class VersionHistoryItem(ListItem):
    """A single version in the history"""
    
    DEFAULT_CSS = """
    VersionHistoryItem {
        height: auto;
        padding: 1;
        margin: 0 1;
        background: $panel;
        border-left: thick $primary;
    }
    
    VersionHistoryItem:hover {
        background: $boost;
    }
    
    VersionHistoryItem.current {
        border-left: thick $success;
        background: $success-darken-3;
    }
    
    VersionHistoryItem.failed {
        border-left: thick $error;
    }
    """
    
    def __init__(self, version_info: Dict[str, Any], is_current: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.version_info = version_info
        if is_current:
            self.add_class('current')
        if version_info.get('status') == 'failed':
            self.add_class('failed')
    
    def compose(self) -> ComposeResult:
        """Compose the version item"""
        version = self.version_info.get('version', 'unknown')
        status = self.version_info.get('status', 'unknown')
        accuracy = self.version_info.get('accuracy', 0)
        created_at = self.version_info.get('created_at', '')
        
        # Format timestamp
        try:
            dt = datetime.fromisoformat(created_at)
            time_str = dt.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            time_str = created_at
        
        # Status emoji
        status_emoji = {
            'ready': '✅',
            'training': '🔄',
            'analyzing': '🔍',
            'failed': '❌',
            'archived': '📦'
        }.get(status, '❓')
        
        yield Label(f"{status_emoji} {version}", classes="bold")
        yield Label(f"   Status: {status} | Accuracy: {accuracy:.1%} | {time_str}", classes="dim")



class VersionHistory(Container):
    """Panel showing version history"""
    
    DEFAULT_CSS = """
    VersionHistory {
        height: 100%;
        border: solid $primary;
        background: $surface;
    }
    
    VersionHistory > #history-header {
        dock: top;
        height: 3;
        background: $primary;
        padding: 0 2;
    }
    
    VersionHistory > #history-list {
        height: 1fr;
    }
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.versions: List[Dict[str, Any]] = []
        self.current_version: str = None
    
    def compose(self) -> ComposeResult:
        """Compose the version history"""
        yield Label("📜 Version History", id="history-header")
        
        with ScrollableContainer(id="history-list"):
            yield Label("No versions yet", classes="dim")
    
    def update_versions(self, versions: List[Dict[str, Any]], current_version: str = None) -> None:
        """Update the version history"""
        self.versions = versions
        self.current_version = current_version
        
        history_list = self.query_one("#history-list", ScrollableContainer)
        history_list.remove_children()
        
        if versions:
            for version in reversed(versions):  # Show newest first
                is_current = version.get('version') == current_version
                history_list.mount(VersionHistoryItem(version, is_current))
        else:
            history_list.mount(Label("No versions yet", classes="dim"))


class CodebaseAnalysisResults(Container):
    """Panel showing codebase analysis results"""
    
    DEFAULT_CSS = """
    CodebaseAnalysisResults {
        height: 100%;
        border: solid $accent;
        background: $surface;
        padding: 1;
    }
    
    #analysis-header {
        dock: top;
        height: 3;
        background: $accent;
        padding: 0 2;
    }
    
    #analysis-content {
        height: 1fr;
    }
    
    #analysis-table {
        height: 1fr;
    }
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.analysis_results = None
    
    def compose(self) -> ComposeResult:
        """Compose the analysis results"""
        yield Label("🔍 Codebase Analysis", id="analysis-header")
        
        with ScrollableContainer(id="analysis-content"):
            table = DataTable(id="analysis-table")
            table.add_columns("Pattern Type", "Frequency", "Confidence", "Examples")
            yield table
    
    def update_results(self, results: Dict[str, Any]) -> None:
        """Update analysis results"""
        self.analysis_results = results
        
        try:
            table = self.query_one("#analysis-table", DataTable)
            table.clear()
            
            if results and 'patterns' in results:
                for pattern in results['patterns'][:50]:  # Show top 50
                    examples = ', '.join(pattern.get('examples', [])[:2])
                    table.add_row(
                        pattern.get('pattern_type', 'unknown'),
                        str(pattern.get('frequency', 0)),
                        f"{pattern.get('confidence', 0):.0%}",
                        examples
                    )
        except Exception:
                pass  # Silently ignore

class ModelListItem(ListItem):
    """A single model in the list"""
    
    DEFAULT_CSS = """
    ModelListItem {
        height: auto;
        padding: 1;
        margin: 0 1;
        background: $panel;
        border-left: thick $accent;
    }
    
    ModelListItem:hover {
        background: $boost;
    }
    
    ModelListItem.selected {
        border-left: thick $success;
        background: $success-darken-3;
    }
    """
    
    def __init__(self, model_info: Dict[str, Any], **kwargs):
        super().__init__(**kwargs)
        self.model_info = model_info
    
    def compose(self) -> ComposeResult:
        """Compose the model item"""
        name = self.model_info.get('name', 'unknown')
        current_version = self.model_info.get('current_version', 'N/A')
        versions = self.model_info.get('versions', [])
        
        # Get latest version info
        latest = versions[-1] if versions else {}
        status = latest.get('status', 'unknown')
        accuracy = latest.get('accuracy', 0)
        
        yield Label(f"📦 {name}", classes="bold")
        yield Label(f"   Version: {current_version} | Status: {status} | Accuracy: {accuracy:.1%} | Versions: {len(versions)}", classes="dim")


class ModelList(Container):
    """Panel showing list of models"""
    
    DEFAULT_CSS = """
    ModelList {
        height: 100%;
        border: solid $primary;
        background: $surface;
    }
    
    ModelList > #model-list-header {
        dock: top;
        height: 3;
        background: $primary;
        padding: 0 2;
    }
    
    ModelList > #model-list-content {
        height: 1fr;
    }
    """
    
    class ModelSelected(Message):
        """Message sent when a model is selected"""
        def __init__(self, model_name: str):
            super().__init__()
            self.model_name = model_name
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.models: List[Dict[str, Any]] = []
        self.selected_model: str = None
    
    def compose(self) -> ComposeResult:
        """Compose the model list"""
        yield Label("📋 Custom Models", id="model-list-header")
        
        with ScrollableContainer(id="model-list-content"):
            yield Label("No models yet", classes="dim")
    
    def update_models(self, models: List[Dict[str, Any]]) -> None:
        """Update the model list"""
        self.models = models
        
        list_content = self.query_one("#model-list-content", ScrollableContainer)
        list_content.remove_children()
        
        if models:
            for model in models:
                item = ModelListItem(model)
                list_content.mount(item)
        else:
            list_content.mount(Label("No models yet", classes="dim"))


class CustomModelsPanel(Container):
    """Main custom models panel with all components"""
    
    DEFAULT_CSS = """
    CustomModelsPanel {
        height: 100%;
        border: solid $primary;
        background: $surface;
    }
    
    CustomModelsPanel > #cm-header {
        dock: top;
        height: 3;
        background: $primary;
        padding: 0 2;
    }
    
    CustomModelsPanel > #cm-controls {
        dock: top;
        height: 3;
        background: $panel;
        padding: 0 2;
    }
    
    CustomModelsPanel > #cm-tabs {
        dock: top;
        height: 3;
        background: $panel;
    }
    
    CustomModelsPanel > #cm-content {
        height: 1fr;
    }
    
    .control-button {
        width: 1fr;
        margin: 0 1;
    }
    
    .tab-button {
        width: 1fr;
        margin: 0 1;
    }
    
    .tab-button.active {
        background: $primary;
    }
    """
    
    BINDINGS = [
        Binding("ctrl+a", "analyze", "Analyze"),
        Binding("ctrl+t", "train", "Train"),
        Binding("1", "show_models", "Models"),
        Binding("2", "show_training", "Training"),
        Binding("3", "show_performance", "Performance"),
        Binding("4", "show_versions", "Versions"),
        Binding("5", "show_analysis", "Analysis"),
    ]
    
    class ModelAction(Message):
        """Message sent when a model action is requested"""
        def __init__(self, action: str, model_name: str = None, **kwargs):
            super().__init__()
            self.action = action
            self.model_name = model_name
            self.kwargs = kwargs
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.current_tab = "models"
        self.selected_model = None
    
    def compose(self) -> ComposeResult:
        """Compose the custom models panel"""
        yield Label("🤖 Custom AI Models", id="cm-header")
        
        with Horizontal(id="cm-controls"):
            yield Button("🔍 Analyze", classes="control-button", id="btn-analyze")
            yield Button("🎯 Train", classes="control-button", id="btn-train")
            yield Button("📊 Performance", classes="control-button", id="btn-performance")
            yield Button("🗑️  Delete", classes="control-button", id="btn-delete")
        
        with Horizontal(id="cm-tabs"):
            yield Button("Models", classes="tab-button active", id="tab-models")
            yield Button("Training", classes="tab-button", id="tab-training")
            yield Button("Performance", classes="tab-button", id="tab-performance")
            yield Button("Versions", classes="tab-button", id="tab-versions")
            yield Button("Analysis", classes="tab-button", id="tab-analysis")
        
        with Container(id="cm-content"):
            yield ModelList(id="models-panel")
            yield TrainingDashboard(id="training-panel", classes="hidden")
            yield PerformanceViewer(id="performance-panel", classes="hidden")
            yield VersionHistory(id="versions-panel", classes="hidden")
            yield CodebaseAnalysisResults(id="analysis-panel", classes="hidden")
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses"""
        button_id = event.button.id
        
        if button_id == "btn-analyze":
            self.action_analyze()
        elif button_id == "btn-train":
            self.action_train()
        elif button_id == "btn-performance":
            self.action_show_performance()
        elif button_id == "btn-delete":
            self._delete_model()
        elif button_id.startswith("tab-"):
            tab_name = button_id.replace("tab-", "")
            self._show_tab(tab_name)
    
    def _show_tab(self, tab_name: str) -> None:
        """Show a specific tab"""
        self.current_tab = tab_name
        
        # Update button styles
        for button in self.query(".tab-button"):
            if button.id == f"tab-{tab_name}":
                button.add_class("active")
            else:
                button.remove_class("active")
        
        # Show/hide panels
        panels = {
            "models": "models-panel",
            "training": "training-panel",
            "performance": "performance-panel",
            "versions": "versions-panel",
            "analysis": "analysis-panel"
        }
        
        for name, panel_id in panels.items():
            panel = self.query_one(f"#{panel_id}")
            if name == tab_name:
                panel.remove_class("hidden")
            else:
                panel.add_class("hidden")
    
    def action_analyze(self) -> None:
        """Analyze codebase"""
        self.post_message(self.ModelAction("analyze"))
    
    def action_train(self) -> None:
        """Train model"""
        self.post_message(self.ModelAction("train"))
    
    def action_show_performance(self) -> None:
        """Show performance metrics"""
        self._show_tab("performance")
    
    def _delete_model(self) -> None:
        """Delete selected model"""
        if self.selected_model:
            self.post_message(self.ModelAction("delete", self.selected_model))
    
    def action_show_models(self) -> None:
        """Show models tab"""
        self._show_tab("models")
    
    def action_show_training(self) -> None:
        """Show training tab"""
        self._show_tab("training")
    
    def action_show_versions(self) -> None:
        """Show versions tab"""
        self._show_tab("versions")
    
    def action_show_analysis(self) -> None:
        """Show analysis tab"""
        self._show_tab("analysis")
    
    def on_model_list_model_selected(self, message: ModelList.ModelSelected) -> None:
        """Handle model selection"""
        self.selected_model = message.model_name
    
    # Convenience methods for updating components
    
    def update_models(self, models: List[Dict[str, Any]]) -> None:
        """Update model list"""
        model_list = self.query_one("#models-panel", ModelList)
        model_list.update_models(models)
    
    def update_training_info(self, info: Dict[str, Any]) -> None:
        """Update training dashboard"""
        training_panel = self.query_one("#training-panel", TrainingDashboard)
        training_panel.update_training_info(info)
    
    def update_training_progress(self, progress: float, status: str = None) -> None:
        """Update training progress"""
        training_panel = self.query_one("#training-panel", TrainingDashboard)
        training_panel.update_progress(progress, status)
    
    def update_performance(self, metrics: Dict[str, Any]) -> None:
        """Update performance metrics"""
        performance_panel = self.query_one("#performance-panel", PerformanceViewer)
        performance_panel.update_metrics(metrics)
    
    def update_versions(self, versions: List[Dict[str, Any]], current_version: str = None) -> None:
        """Update version history"""
        versions_panel = self.query_one("#versions-panel", VersionHistory)
        versions_panel.update_versions(versions, current_version)
    
    def update_analysis(self, results: Dict[str, Any]) -> None:
        """Update analysis results"""
        analysis_panel = self.query_one("#analysis-panel", CodebaseAnalysisResults)
        analysis_panel.update_results(results)
