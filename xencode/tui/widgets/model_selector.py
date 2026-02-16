#!/usr/bin/env python3
"""
Model Selector Widget for Xencode TUI

Allows selecting models and configuring ensembles.
"""

from typing import List, Optional, Set

from rich.text import Text
from textual.widgets import Static, Checkbox, RadioButton, RadioSet, Label
from textual.containers import Container, Vertical, Horizontal, VerticalScroll
from textual.binding import Binding
from textual.message import Message
from xencode.tui.utils.model_checker import ModelChecker


class ModelSelected(Message):
    """Message sent when model selection changes"""
    
    def __init__(self, models: List[str], is_ensemble: bool, method: str = "vote") -> None:
        """Initialize message
        
        Args:
            models: List of selected model names
            is_ensemble: Whether ensemble mode is active
            method: Ensemble method (vote/weighted/consensus/hybrid)
        """
        super().__init__()
        self.models = models
        self.is_ensemble = is_ensemble
        self.method = method


class ModelSelector(VerticalScroll):
    """Model selector with ensemble configuration"""
    
    DEFAULT_CSS = """
    ModelSelector {
        height: 100%;
        background: $surface;
        border: solid $primary;
        padding: 1;
    }
    
    ModelSelector Label {
        margin: 1 0;
        color: $text;
    }
    
    ModelSelector .section-title {
        text-style: bold;
        color: $accent;
    }
    
    ModelSelector Checkbox {
        margin: 0 0 0 2;
    }
    
    ModelSelector RadioButton {
        margin: 0 0 0 2;
    }
    """
    
    BINDINGS = [
        Binding("enter", "apply_selection", "Apply", show=True),
    ]
    
    # Available models list
    AVAILABLE_MODELS = [
        ("openrouter:openai/gpt-4o-mini", "OpenRouter GPT-4o Mini (Cloud)"),
        ("openrouter:anthropic/claude-3.5-sonnet", "OpenRouter Claude 3.5 Sonnet (Cloud)"),
        ("openrouter:meta-llama/llama-3.1-8b-instruct", "OpenRouter Llama 3.1 8B (Cloud)"),
        ("qwen:qwen-max-coder-7b-instruct", "Qwen Max Coder 7B (Cloud)"),
        ("qwen:qwen-plus", "Qwen Plus (Cloud)"),
        ("qwen:qwen-max", "Qwen Max (Cloud)"),
        ("qwen2.5:7b", "Qwen 2.5 7B (Fast, Balanced)"),
        ("qwen2.5:14b", "Qwen 2.5 14B (Powerful)"),
        ("llama3.1:8b", "Llama 3.1 8B (High Quality)"),
        ("llama3.2:3b", "Llama 3.2 3B (Very Fast)"),
        ("mistral:7b", "Mistral 7B (Fast)"),
        ("phi3:mini", "Phi-3 Mini (Ultra Fast)"),
        ("gemma2:2b", "Gemma 2 2B (Tiny, Fast)"),
        ("codellama:7b", "CodeLlama 7B (Code Specialist)"),
    ]
    
    # Ensemble methods
    ENSEMBLE_METHODS = [
        ("vote", "Majority Vote - Simple token voting"),
        ("weighted", "Weighted - By model quality"),
        ("consensus", "Consensus - Require 70% agreement"),
        ("hybrid", "Hybrid - Adaptive method selection"),
    ]
    
    def __init__(self, *args, **kwargs):
        """Initialize model selector"""
        super().__init__(*args, **kwargs)
        self.border_title = "⚙️ Model Selection"
        self.selected_models: Set[str] = set()
        self.ensemble_enabled = False
        self.ensemble_method = "vote"
        
        # Widgets
        self.model_checkboxes: dict = {}
        self.ensemble_radios: Optional[RadioSet] = None
    
    def compose(self):
        """Compose the selector"""
        # Title
        yield Label("Select Models:", classes="section-title")
        yield Label("Choose 1 model for single, or 2-4 for ensemble", classes="dim")
        
        # Model checkboxes
        available_system_models = ModelChecker.get_available_models()
        
        # Track which system models are covered by our hardcoded list
        covered_system_models = set()
        
        for model_id, model_name in self.AVAILABLE_MODELS:
            is_cloud_provider = model_id.startswith("qwen:") or model_id.startswith("openrouter:")

            # Check if model is installed (exact or prefix match)
            # We match if any system model starts with our ID, OR if our ID starts with the system model (less likely)
            # Actually, let's find the best match
            matched_sys_model = None
            if not is_cloud_provider:
                for sys_model in available_system_models:
                    if sys_model.startswith(model_id) or model_id.startswith(sys_model.split(':')[0]):
                        matched_sys_model = sys_model
                        covered_system_models.add(sys_model)
                        break
            
            is_installed = matched_sys_model is not None or is_cloud_provider
            
            display_name = model_name
            if model_id.startswith("qwen:"):
                display_name += " (Qwen Auth Required)"
            elif model_id.startswith("openrouter:"):
                display_name += " (OpenRouter API Key Required)"
            elif not is_installed:
                display_name += " (Not Installed)"
            else:
                # Use the actual installed name if it differs slightly
                if matched_sys_model != model_id:
                    display_name += f" [{matched_sys_model}]"
                
            checkbox = Checkbox(display_name, value=(is_installed and "qwen" in model_id))
            # Use the actual installed ID if found, otherwise the hardcoded one
            checkbox.data = matched_sys_model if matched_sys_model else model_id
            
            self.model_checkboxes[model_id] = checkbox
            yield checkbox
            
        # Add any other installed models that weren't in our list
        for sys_model in available_system_models:
            if sys_model not in covered_system_models:
                checkbox = Checkbox(f"{sys_model} (Installed)", value=False)
                checkbox.data = sys_model
                self.model_checkboxes[sys_model] = checkbox
                yield checkbox
            
        # Add option to refresh/check models
        yield Label("Tip: Run 'ollama pull <model>' in terminal to install", classes="dim")
        
        # Ensemble section
        yield Label("")  # Spacer
        yield Label("Ensemble Method:", classes="section-title")
        yield Label("Used when 2+ models selected", classes="dim")
        
        # Ensemble method radio buttons
        with RadioSet(id="ensemble-method"):
            for method_id, method_desc in self.ENSEMBLE_METHODS:
                radio = RadioButton(method_desc, value=(method_id == "vote"))
                radio.data = method_id
                yield radio
        
        # Instructions
        yield Label("")  # Spacer
        yield Label("[Enter] to apply changes", classes="dim")
    
    def on_mount(self) -> None:
        """Called when mounted"""
        self.ensemble_radios = self.query_one("#ensemble-method", RadioSet)
        
        # Initialize with default if available
        available = ModelChecker.get_available_models()
        
        # If we have any available models, select the first one if nothing else is selected
        if available and not self.selected_models:
            # Prefer qwen or llama
            preferred = [m for m in available if "qwen" in m or "llama" in m]
            if preferred:
                default_model = preferred[0]
            else:
                default_model = available[0]
                
            self.selected_models.add(default_model)
            
            # Update checkboxes
            for cb in self.query(Checkbox):
                if cb.data == default_model:
                    cb.value = True
                    break
    
    def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        """Handle checkbox changes"""
        model_id = event.checkbox.data
        
        if event.value:
            self.selected_models.add(model_id)
        else:
            self.selected_models.discard(model_id)
        
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
            title = "⚙️ Model Selection (None)"
        elif count == 1:
            title = f"⚙️ Single Model ({count} selected)"
        else:
            title = f"⚙️ Ensemble Mode ({count} models, {self.ensemble_method})"
        
        self.border_title = title
    
    def action_apply_selection(self) -> None:
        """Apply the current selection"""
        if not self.selected_models:
            self.app.notify("⚠️ Please select at least one model", severity="warning")
            return
            
        # Verify selected models are actually installed
        available = ModelChecker.get_available_models()
        missing = []
        for model in self.selected_models:
            if model.startswith("qwen:") or model.startswith("openrouter:"):
                continue
            if not any(m.startswith(model) for m in available):
                missing.append(model)
        
        if missing:
            self.app.notify(f"⚠️ Models not found: {', '.join(missing)}. Please install via 'ollama pull'", severity="error")
            return
        
        if len(self.selected_models) > 4:
            self.app.notify("⚠️ Maximum 4 models for ensemble", severity="warning")
            return
        
        # Post message to app
        self.post_message(ModelSelected(
            models=list(self.selected_models),
            is_ensemble=self.ensemble_enabled,
            method=self.ensemble_method
        ))
        
        # Notify user
        if self.ensemble_enabled:
            self.app.notify(
                f"✅ Ensemble activated: {len(self.selected_models)} models ({self.ensemble_method})",
                severity="information"
            )
        else:
            model = list(self.selected_models)[0]
            self.app.notify(f"✅ Single model: {model}", severity="information")
    
    def get_current_selection(self) -> dict:
        """Get current selection state"""
        return {
            "models": list(self.selected_models),
            "is_ensemble": self.ensemble_enabled,
            "method": self.ensemble_method
        }
