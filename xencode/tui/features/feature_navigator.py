"""Feature navigation widget for TUI."""

from typing import Dict

from textual.containers import Container, Vertical
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import Button, Label


class FeatureNavigator(Container):
    """Navigation widget for switching between features."""

    DEFAULT_CSS = """
    FeatureNavigator {
        height: 100%;
        width: 100%;
        border: solid $primary;
        background: $surface;
    }

    .nav-header {
        height: 3;
        padding: 1;
        background: $primary;
        color: $text;
        text-style: bold;
        text-align: center;
    }

    .nav-buttons {
        height: 1fr;
        padding: 1;
        overflow-y: auto;
    }

    .feature-button {
        width: 100%;
        margin: 0 0 1 0;
    }

    .feature-button.active {
        background: $success;
    }
    """

    current_feature = reactive(None)

    class FeatureSelected(Message):
        """Message sent when a feature is selected."""

        def __init__(self, feature_id: str) -> None:
            super().__init__()
            self.feature_id = feature_id

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.features: Dict[str, Dict[str, str]] = {
            "code_review": {"name": "Code Review", "icon": "🔍"},
            "terminal_assistant": {"name": "Terminal Assistant", "icon": "💻"},
            "project_analyzer": {"name": "Project Analyzer", "icon": "📊"},
            "learning_mode": {"name": "Learning Mode", "icon": "🎓"},
            "multi_language": {"name": "Multi-language", "icon": "🌍"},
            "voice_interface": {"name": "Voice Interface", "icon": "🎤"},
            "custom_models": {"name": "Custom Models", "icon": "🤖"},
            "security_auditor": {"name": "Security Auditor", "icon": "🔒"},
            "performance_profiler": {"name": "Performance Profiler", "icon": "⚡"},
            "collaborative_coding": {"name": "Collaborative Coding", "icon": "👥"},
        }

    def compose(self):
        """Compose the navigator."""
        yield Label("Features", classes="nav-header")

        with Vertical(classes="nav-buttons"):
            for feature_id, feature_info in self.features.items():
                button = Button(
                    f"{feature_info['icon']} {feature_info['name']}",
                    id=f"nav-{feature_id}",
                    classes="feature-button"
                )
                yield button

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle feature button press."""
        if event.button.id and event.button.id.startswith("nav-"):
            feature_id = event.button.id[4:]  # Remove "nav-" prefix
            self.current_feature = feature_id

            # Update button states
            for button in self.query(Button):
                if button.id == event.button.id:
                    button.add_class("active")
                else:
                    button.remove_class("active")

            # Send message
            self.post_message(self.FeatureSelected(feature_id))
