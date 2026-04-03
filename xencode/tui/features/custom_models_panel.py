"""Custom AI Models TUI panel."""

from textual.containers import Container, Vertical, Horizontal, ScrollableContainer
from textual.widgets import Button, Label, Static, DataTable, ProgressBar
from textual.reactive import reactive
from typing import Optional, List, Dict, Any

from .base_feature_panel import BaseFeaturePanel


class ModelCard(Static):
    """Card for a custom model."""
    
    DEFAULT_CSS = """
    ModelCard {
        height: auto;
        padding: 1;
        margin: 0 0 1 0;
        border: solid $accent;
        background: $panel;
    }
    
    ModelCard:hover {
        background: $primary;
    }
    """
    
    def __init__(self, name: str, accuracy: float, version: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_name = name
        self.accuracy = accuracy
        self.version = version
    
    def render(self) -> str:
        return (
            f"[bold]{self.model_name}[/bold]\n"
            f"Version: {self.version} | Accuracy: {self.accuracy:.1f}%"
        )


class CustomModelsPanel(BaseFeaturePanel):
    """Panel for custom AI model management."""
    
    DEFAULT_CSS = """
    CustomModelsPanel {
        height: 100%;
    }
    
    .models-controls {
        height: auto;
        padding: 1;
        background: $panel;
    }
    
    .models-content {
        height: 1fr;
        padding: 1;
    }
    
    .training-progress {
        height: auto;
        padding: 1;
        margin-top: 1;
        border: solid $warning;
    }
    """
    
    training = reactive(False)
    
    def __init__(self, *args, **kwargs):
        super().__init__(
            feature_name="custom_models",
            title="🤖 Custom AI Models",
            *args,
            **kwargs
        )
        self.models: List[Dict[str, Any]] = []
    
    def compose(self):
        """Compose the custom models panel."""
        yield from super().compose()
    
    def on_mount(self) -> None:
        """Initialize panel on mount."""
        self.set_status("enabled")
        self._load_models()
        self._build_content()
    
    def _load_models(self) -> None:
        """Load custom models."""
        # TODO: Load from actual custom models feature
        self.models = [
            {"name": "my-python-model", "accuracy": 92.5, "version": "1.0.0"},
            {"name": "js-style-model", "accuracy": 88.3, "version": "0.5.0"},
        ]
    
    def _build_content(self) -> None:
        """Build the panel content."""
        if not self.content_container:
            return
        
        self.content_container.remove_children()
        
        with self.content_container:
            # Controls
            with Horizontal(classes="models-controls"):
                yield Button("Analyze Codebase", id="btn-analyze", variant="primary")
                yield Button("Train Model", id="btn-train")
                yield Button("List Models", id="btn-list")
                yield Button("Performance", id="btn-performance")
            
            # Content area
            with ScrollableContainer(classes="models-content"):
                if self.training:
                    self._render_training()
                elif self.models:
                    self._render_models()
                else:
                    yield Label(
                        "No custom models yet. Analyze your codebase to create one.",
                        classes="feature-empty"
                    )
    
    def _render_models(self) -> None:
        """Render models list."""
        for model in self.models:
            yield ModelCard(model["name"], model["accuracy"], model["version"])
    
    def _render_training(self) -> None:
        """Render training progress."""
        with Container(classes="training-progress"):
            yield Label("[bold]Training in progress...[/bold]")
            yield ProgressBar(total=100, show_eta=True)
    
    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_id = event.button.id
        
        if button_id == "btn-analyze":
            await self._analyze_codebase()
        elif button_id == "btn-train":
            await self._train_model()
        elif button_id == "btn-list":
            self._build_content()
        elif button_id == "btn-performance":
            await self._show_performance()
    
    async def _analyze_codebase(self) -> None:
        """Analyze codebase for model training."""
        self.set_status("loading")
        # TODO: Implement codebase analysis
        self.set_status("enabled")
    
    async def _train_model(self) -> None:
        """Train a custom model."""
        self.training = True
        self.set_status("loading")
        self._build_content()
        # TODO: Implement model training
        self.training = False
        self.set_status("enabled")
    
    async def _show_performance(self) -> None:
        """Show model performance metrics."""
        # TODO: Implement performance display
        pass
