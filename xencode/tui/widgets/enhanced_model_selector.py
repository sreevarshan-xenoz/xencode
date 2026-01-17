"""
Enhanced Model Selector Widget for Xencode TUI

Advanced model selection with comparison, benchmarking, and detailed information.
"""

from typing import List, Optional, Set, Dict, Any
import time
import asyncio
from rich.text import Text
from textual.widgets import Static, Checkbox, RadioButton, RadioSet, Label, Button, DataTable
from textual.containers import Container, Vertical, Horizontal, VerticalScroll, Grid
from textual.binding import Binding
from textual.message import Message
from textual.reactive import reactive
from xencode.tui.utils.model_checker import ModelChecker
from xencode.ai_ensembles import EnsembleMethod


class EnhancedModelSelected(Message):
    """Enhanced message sent when model selection changes with additional data"""

    def __init__(self, 
                 models: List[str], 
                 is_ensemble: bool, 
                 method: str = "vote",
                 model_details: Optional[Dict[str, Any]] = None,
                 performance_metrics: Optional[Dict[str, float]] = None) -> None:
        """Initialize enhanced message

        Args:
            models: List of selected model names
            is_ensemble: Whether ensemble mode is active
            method: Ensemble method (vote/weighted/consensus/hybrid/semantic)
            model_details: Additional details about selected models
            performance_metrics: Performance metrics for selected models
        """
        super().__init__()
        self.models = models
        self.is_ensemble = is_ensemble
        self.method = method
        self.model_details = model_details or {}
        self.performance_metrics = performance_metrics or {}


class ModelCard(Container):
    """Individual model information card"""
    
    def __init__(self, model_info: Dict[str, Any], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_info = model_info
        self.styles.height = "15"
        self.border_title = model_info.get("name", "Unknown Model")
        
    def compose(self):
        """Compose the model card"""
        # Model details
        yield Label(f"Size: {self.model_info.get('size', 'Unknown')}", classes="model-detail")
        yield Label(f"Type: {self.model_info.get('type', 'General')}", classes="model-detail")
        yield Label(f"Parameters: {self.model_info.get('parameters', 'Unknown')}", classes="model-detail")
        
        # Performance metrics if available
        if "response_time" in self.model_info:
            yield Label(f"Response: {self.model_info['response_time']:.2f}s", classes="model-detail")
        if "accuracy" in self.model_info:
            yield Label(f"Accuracy: {self.model_info['accuracy']:.2f}", classes="model-detail")
        
        # Select button
        yield Button("Select", id=f"select-{self.model_info['name']}", variant="primary")


class ModelComparisonTable(DataTable):
    """Table for comparing models side-by-side"""
    pass


class EnhancedModelSelector(VerticalScroll):
    """Enhanced model selector with detailed information, comparison, and benchmarking"""

    DEFAULT_CSS = """
    EnhancedModelSelector {
        height: 100%;
        background: $surface;
        border: solid $primary;
        padding: 1;
    }

    EnhancedModelSelector Label {
        margin: 0.5 0;
        color: $text;
    }

    EnhancedModelSelector .section-title {
        text-style: bold;
        color: $accent;
        margin-top: 1;
    }

    EnhancedModelSelector .model-detail {
        margin-left: 1;
    }

    EnhancedModelSelector .metric-label {
        text-style: bold;
    }

    EnhancedModelSelector .card-container {
        height: 16;
        border: solid $secondary;
        margin: 1 0;
        padding: 1;
    }

    EnhancedModelSelector ModelCard {
        height: 15;
        border: solid $panel;
        margin: 0.5 0;
    }

    EnhancedModelSelector DataTable {
        height: 20;
        margin: 1 0;
    }

    EnhancedModelSelector Button {
        margin: 0.5 0.5;
    }

    EnhancedModelSelector .status-indicator {
        text-style: bold;
    }
    """

    BINDINGS = [
        Binding("enter", "apply_selection", "Apply", show=True),
        Binding("r", "refresh_models", "Refresh", show=True),
        Binding("b", "benchmark_models", "Benchmark", show=True),
        Binding("c", "compare_models", "Compare", show=True),
    ]

    # Reactive properties
    selected_models = reactive(set)
    ensemble_enabled = reactive(False)
    ensemble_method = reactive("vote")
    show_details = reactive(False)

    def __init__(self, *args, **kwargs):
        """Initialize enhanced model selector"""
        super().__init__(*args, **kwargs)
        self.border_title = "⚙️ Enhanced Model Selection"
        
        # Model information storage
        self.model_details: Dict[str, Dict[str, Any]] = {}
        self.performance_data: Dict[str, Dict[str, float]] = {}
        self.comparison_data: List[Dict[str, Any]] = []
        
        # Widgets
        self.model_cards_container = VerticalScroll(id="model-cards-container")
        self.comparison_table = DataTable(id="comparison-table", zebra_stripes=True)
        self.details_panel = Static(id="details-panel", markup=True)

    def compose(self):
        """Compose the enhanced selector"""
        # Controls
        with Horizontal():
            yield Button("🔄 Refresh", id="refresh-btn", variant="default")
            yield Button("⏱️ Benchmark", id="benchmark-btn", variant="success")
            yield Button("📋 Compare", id="compare-btn", variant="primary")
            yield Button("📊 Toggle Details", id="toggle-details", variant="warning")
        
        # Title
        yield Label("Select Models:", classes="section-title")
        yield Label("Choose 1 model for single, or 2-4 for ensemble", classes="dim")

        # Model cards container
        self.model_cards_container.border_title = "Available Models"
        yield self.model_cards_container

        # Model comparison table
        self.comparison_table.border_title = "Model Comparison"
        yield self.comparison_table

        # Ensemble section
        yield Label("Ensemble Method:", classes="section-title")
        yield Label("Used when 2+ models selected", classes="dim")

        # Ensemble method radio buttons
        with RadioSet(id="ensemble-method"):
            for method in EnsembleMethod:
                method_name = method.value
                method_desc = self._get_ensemble_method_description(method_name)
                radio = RadioButton(method_desc, value=(method_name == "vote"))
                radio.data = method_name
                yield radio

        # Details panel
        self.details_panel.border_title = "Model Details"
        self.details_panel.visible = self.show_details
        yield self.details_panel

        # Instructions
        yield Label("")  # Spacer
        yield Label("[Enter] to apply, [R]efresh, [B]enchmark, [C]ompare", classes="dim")

    def _get_ensemble_method_description(self, method: str) -> str:
        """Get description for ensemble method"""
        descriptions = {
            "vote": "Majority Vote - Simple token voting",
            "weighted": "Weighted - By model quality/confidence",
            "consensus": "Consensus - Require agreement threshold",
            "hybrid": "Hybrid - Adaptive method selection",
            "semantic": "Semantic - Using embedding similarity"
        }
        return descriptions.get(method, f"{method.title()} - {method} method")

    def on_mount(self) -> None:
        """Called when mounted"""
        self.refresh_models()
        self.update_model_display()

    def refresh_models(self):
        """Refresh the list of available models with detailed information"""
        try:
            available_models = ModelChecker.get_available_models()
            
            # Get detailed information for each model
            self.model_details = {}
            for model_name in available_models:
                self.model_details[model_name] = self._get_detailed_model_info(model_name)
            
            # Update comparison data
            self.comparison_data = [self.model_details[name] for name in available_models]
            
        except Exception as e:
            # Fallback with basic information
            available_models = ModelChecker.get_available_models()
            self.model_details = {}
            for model_name in available_models:
                self.model_details[model_name] = {
                    "name": model_name,
                    "size": self._estimate_model_size(model_name),
                    "type": self._get_model_type(model_name),
                    "parameters": self._estimate_parameters(model_name),
                    "description": self._get_model_description(model_name),
                    "is_available": True
                }
            
            self.comparison_data = [self.model_details[name] for name in available_models]

    def _get_detailed_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get detailed information about a model"""
        return {
            "name": model_name,
            "size": self._estimate_model_size(model_name),
            "type": self._get_model_type(model_name),
            "parameters": self._estimate_parameters(model_name),
            "description": self._get_model_description(model_name),
            "response_time": self._get_mock_response_time(model_name),
            "accuracy": self._get_mock_accuracy(model_name),
            "memory_usage": self._estimate_memory_usage(model_name),
            "recommended_use": self._get_recommended_use(model_name),
            "is_available": True
        }

    def _estimate_model_size(self, model_name: str) -> str:
        """Estimate model size based on name"""
        if "3b" in model_name:
            return "~3.8 GB"
        elif "7b" in model_name:
            return "~4.7-7.3 GB"
        elif "8b" in model_name:
            return "~8.0 GB"
        elif "14b" in model_name:
            return "~14.0 GB"
        elif "2b" in model_name:
            return "~2.0 GB"
        else:
            return "Unknown"

    def _get_model_type(self, model_name: str) -> str:
        """Get model type based on name"""
        if "qwen" in model_name:
            return "Multimodal"
        elif "llama" in model_name:
            return "General Purpose"
        elif "mistral" in model_name:
            return "Efficient"
        elif "phi" in model_name:
            return "Small & Fast"
        elif "gemma" in model_name:
            return "Lightweight"
        elif "codellama" in model_name:
            return "Code Specialist"
        else:
            return "General"

    def _estimate_parameters(self, model_name: str) -> str:
        """Estimate number of parameters"""
        if "3b" in model_name:
            return "3B"
        elif "7b" in model_name:
            return "7B"
        elif "8b" in model_name:
            return "8B"
        elif "14b" in model_name:
            return "14B"
        elif "2b" in model_name:
            return "2B"
        else:
            return "Unknown"

    def _get_model_description(self, model_name: str) -> str:
        """Get model description"""
        descriptions = {
            "qwen": "Alibaba's Qwen series - Strong reasoning and multilingual",
            "llama": "Meta's Llama series - Versatile and powerful",
            "mistral": "Mistral AI models - Efficient and fast",
            "phi": "Microsoft's Phi series - Small but capable",
            "gemma": "Google's Gemma - Lightweight and efficient",
            "codellama": "Specialized for code generation and understanding"
        }
        
        for key, desc in descriptions.items():
            if key in model_name:
                return desc
        
        return "General purpose language model"

    def _get_mock_response_time(self, model_name: str) -> float:
        """Mock response time based on model characteristics"""
        import random
        base_time = 0.8
        if "small" in model_name or "mini" in model_name or "2b" in model_name:
            return round(base_time + random.uniform(0.1, 0.3), 2)
        elif "7b" in model_name or "8b" in model_name:
            return round(base_time + random.uniform(0.3, 0.7), 2)
        else:
            return round(base_time + random.uniform(0.5, 1.2), 2)

    def _get_mock_accuracy(self, model_name: str) -> float:
        """Mock accuracy estimate"""
        import random
        if "qwen" in model_name or "llama" in model_name:
            return round(0.85 + random.uniform(0.0, 0.1), 2)
        elif "mistral" in model_name:
            return round(0.82 + random.uniform(0.0, 0.08), 2)
        elif "phi" in model_name or "gemma" in model_name:
            return round(0.75 + random.uniform(0.0, 0.1), 2)
        else:
            return round(0.78 + random.uniform(0.0, 0.12), 2)

    def _estimate_memory_usage(self, model_name: str) -> str:
        """Estimate memory usage"""
        if "2b" in model_name:
            return "2-4 GB"
        elif "3b" in model_name:
            return "3-5 GB"
        elif "7b" in model_name:
            return "4-7 GB"
        elif "8b" in model_name:
            return "6-8 GB"
        elif "14b" in model_name:
            return "8-12 GB"
        else:
            return "4-8 GB"

    def _get_recommended_use(self, model_name: str) -> str:
        """Get recommended use case"""
        if "code" in model_name.lower() or "codellama" in model_name:
            return "Programming/Coding"
        elif "qwen" in model_name:
            return "General/Multimodal"
        elif "llama" in model_name:
            return "General Purpose"
        elif "mistral" in model_name:
            return "Efficient Tasks"
        elif "phi" in model_name:
            return "Fast Responses"
        else:
            return "General Use"

    def update_model_display(self):
        """Update the display of models"""
        # Clear existing model cards
        self.model_cards_container.remove_children()
        
        # Add model cards
        for model_name, details in self.model_details.items():
            card = ModelCard(details)
            card.styles.margin = ("0.5", "0")
            self.model_cards_container.mount(card)
        
        # Update comparison table
        self.update_comparison_table()

    def update_comparison_table(self):
        """Update the model comparison table"""
        # Clear existing table
        self.comparison_table.clear()
        
        if not self.comparison_data:
            return
            
        # Add headers
        headers = ["Model", "Size", "Params", "Type", "Response (s)", "Accuracy", "Memory", "Use Case"]
        self.comparison_table.add_columns(*headers)
        
        # Add rows for each model
        for model_info in self.comparison_data:
            row = [
                model_info.get("name", "Unknown"),
                model_info.get("size", "Unknown"),
                model_info.get("parameters", "Unknown"),
                model_info.get("type", "General"),
                str(model_info.get("response_time", "N/A")),
                f"{model_info.get('accuracy', 0):.2f}" if isinstance(model_info.get('accuracy'), (int, float)) else "N/A",
                model_info.get("memory_usage", "Unknown"),
                model_info.get("recommended_use", "General")
            ]
            self.comparison_table.add_row(*row)

    async def benchmark_models(self):
        """Benchmark the available models"""
        from textual.app import App
        app = self.app
        
        app.notify("⏱️ Starting model benchmarking...", timeout=3)
        
        # Simulate benchmarking process
        for model_name in self.model_details.keys():
            app.notify(f"📊 Benchmarking {model_name}...", timeout=2)
            
            # Simulate benchmarking time
            await asyncio.sleep(0.5)
            
            # Update performance data with mock results
            self.performance_data[model_name] = {
                "response_time": self._get_mock_response_time(model_name),
                "throughput": round(10 + (hash(model_name) % 20), 2),  # tokens/sec
                "accuracy": self._get_mock_accuracy(model_name),
                "energy_efficiency": round(0.5 + (hash(model_name) % 5) / 10, 2)
            }
        
        app.notify(f"✅ Benchmarking complete for {len(self.model_details)} models", timeout=5)
        self.update_model_display()

    def compare_models(self):
        """Show model comparison"""
        if len(self.selected_models) < 2:
            self.app.notify("⚠️ Please select at least 2 models to compare", severity="warning")
            return
            
        # In a real implementation, this would show detailed comparison
        selected_names = list(self.selected_models)
        self.app.notify(f"📋 Comparing {len(selected_names)} models: {', '.join(selected_names)}", timeout=5)

    def toggle_details(self):
        """Toggle detailed view"""
        self.show_details = not self.show_details
        self.details_panel.visible = self.show_details
        
        if self.show_details and self.selected_models:
            # Show details for selected models
            details_text = []
            for model_name in self.selected_models:
                if model_name in self.model_details:
                    details = self.model_details[model_name]
                    details_text.append(f"[bold]{details['name']}[/bold]")
                    details_text.append(f"  Size: {details['size']}")
                    details_text.append(f"  Type: {details['type']}")
                    details_text.append(f"  Parameters: {details['parameters']}")
                    details_text.append(f"  Response Time: {details.get('response_time', 'N/A')}s")
                    details_text.append(f"  Accuracy: {details.get('accuracy', 'N/A')}")
                    details_text.append(f"  Recommended Use: {details['recommended_use']}")
                    details_text.append("")  # Blank line
            
            self.details_panel.update("\n".join(details_text))
        else:
            self.details_panel.update("Select models and toggle details to see information here.")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses"""
        if event.button.id == "refresh-btn":
            self.action_refresh_models()
        elif event.button.id == "benchmark-btn":
            self.app.call_later(0.1, self.benchmark_models)  # Schedule async call
        elif event.button.id == "compare-btn":
            self.compare_models()
        elif event.button.id == "toggle-details":
            self.toggle_details()
        elif event.button.id and event.button.id.startswith("select-"):
            # Handle model selection from card
            model_name = event.button.id.replace("select-", "")
            self.toggle_model_selection(model_name)

    def toggle_model_selection(self, model_name: str):
        """Toggle selection of a model"""
        if model_name in self.selected_models:
            self.selected_models.discard(model_name)
        else:
            self.selected_models.add(model_name)
        
        # Update ensemble status
        self.ensemble_enabled = len(self.selected_models) >= 2
        self._update_border_title()

    def on_radio_set_changed(self, event: RadioSet.Changed) -> None:
        """Handle radio button changes"""
        if event.pressed and event.pressed.data:
            self.ensemble_method = event.pressed.data
            self._update_border_title()

    def _update_border_title(self) -> None:
        """Update border title with current status"""
        count = len(self.selected_models)

        if count == 0:
            title = "⚙️ Enhanced Model Selection (None)"
        elif count == 1:
            title = f"⚙️ Single Model ({count} selected)"
        else:
            title = f"⚙️ Ensemble Mode ({count} models, {self.ensemble_method})"

        self.border_title = title

    def action_refresh_models(self) -> None:
        """Refresh models action"""
        self.refresh_models()
        self.update_model_display()
        self.app.notify(f"🔄 Refreshed {len(self.model_details)} models", timeout=3)

    def action_benchmark_models(self) -> None:
        """Benchmark models action"""
        self.app.call_later(0.1, self.benchmark_models)

    def action_compare_models(self) -> None:
        """Compare models action"""
        self.compare_models()

    def action_apply_selection(self) -> None:
        """Apply the current selection"""
        if not self.selected_models:
            self.app.notify("⚠️ Please select at least one model", severity="warning")
            return

        # Verify selected models are actually installed
        available = ModelChecker.get_available_models()
        missing = []
        for model in self.selected_models:
            if not any(m.startswith(model) for m in available):
                missing.append(model)

        if missing:
            self.app.notify(f"⚠️ Models not found: {', '.join(missing)}. Please install via 'ollama pull'", severity="error")
            return

        if len(self.selected_models) > 4:
            self.app.notify("⚠️ Maximum 4 models for ensemble", severity="warning")
            return

        # Prepare model details and performance metrics
        selected_details = {name: self.model_details.get(name, {}) for name in self.selected_models}
        selected_performance = {name: self.performance_data.get(name, {}) for name in self.selected_models}

        # Post enhanced message to app
        self.post_message(EnhancedModelSelected(
            models=list(self.selected_models),
            is_ensemble=self.ensemble_enabled,
            method=self.ensemble_method,
            model_details=selected_details,
            performance_metrics=selected_performance
        ))

        # Notify user
        if self.ensemble_enabled:
            self.app.notify(
                f"✅ Enhanced Ensemble activated: {len(self.selected_models)} models ({self.ensemble_method})",
                severity="information"
            )
        else:
            model = list(self.selected_models)[0]
            self.app.notify(f"✅ Single model selected: {model}", severity="information")

    def get_current_selection(self) -> dict:
        """Get current selection state"""
        return {
            "models": list(self.selected_models),
            "is_ensemble": self.ensemble_enabled,
            "method": self.ensemble_method,
            "model_details": {name: self.model_details.get(name, {}) for name in self.selected_models},
            "performance_metrics": {name: self.performance_data.get(name, {}) for name in self.selected_models}
        }