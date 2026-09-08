"""Custom AI Models TUI panel."""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from textual.containers import Container, Horizontal, ScrollableContainer
from textual.reactive import reactive
from textual.widgets import Button, Label, ProgressBar, Static

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
            *args,
            feature_name="custom_models",
            title="🤖 Custom AI Models",
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
        # Load from actual custom models feature
        try:
            from xencode.features.custom_models import CustomModelManager

            manager = CustomModelManager()
            models_list = manager.list_models()

            self.models = [
                {
                    "name": model.get("name", "unknown"),
                    "accuracy": model.get("accuracy", 0.0),
                    "version": model.get("version", "1.0.0")
                }
                for model in models_list
            ]
        except (ImportError, AttributeError):
            # Fallback if custom models feature not available
            self.models = []

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
        try:
            from xencode.analyzers.code_analyzer import CodeAnalyzer

            analyzer = CodeAnalyzer()
            analysis = await analyzer.analyze_directory(Path.cwd())

            # Show analysis summary
            summary = "Codebase Analysis:\n"
            summary += f"- Files: {analysis.get('total_files', 0)}\n"
            summary += f"- Lines: {analysis.get('total_lines', 0)}\n"
            summary += f"- Complexity: {analysis.get('avg_complexity', 0):.1f}\n"
            summary += f"- Languages: {', '.join(analysis.get('languages', []))}"

            self.notify(summary)
        except Exception as e:
            self.notify(f"Analysis error: {e}")
        finally:
            self.set_status("enabled")

    async def _train_model(self) -> None:
        """Train a custom model."""
        self.training = True
        self.set_status("loading")
        self._build_content()

        try:
            from xencode.features.custom_models import CustomModelTrainer

            trainer = CustomModelTrainer()
            await trainer.train_model(
                name=f"custom-model-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                data_path=Path.cwd()
            )

            self.notify("Model training completed!")
            self._load_models()
        except ImportError:
            self.notify("Custom model training feature not available")
        except Exception as e:
            self.notify(f"Training error: {e}")
        finally:
            self.training = False
            self.set_status("enabled")

    async def _show_performance(self) -> None:
        """Show model performance metrics."""
        if not self.models:
            self.notify("No models to show performance for")
            return

        # Show performance for first model
        model = self.models[0]
        perf_info = f"Model Performance: {model['name']}\n\n"
        perf_info += f"Accuracy: {model['accuracy']:.1f}%\n"
        perf_info += f"Version: {model['version']}\n"
        perf_info += "Inference Time: ~50ms (estimated)\n"
        perf_info += "Training Samples: 10,000 (estimated)"

        self.notify(perf_info)
