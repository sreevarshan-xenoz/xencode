"""Base class for feature panels with consistent UI patterns."""

from textual.containers import Container, Vertical
from textual.widgets import Static, Label
from textual.reactive import reactive
from typing import Optional


class FeatureStatus(Static):
    """Status indicator for feature panels."""
    
    DEFAULT_CSS = """
    FeatureStatus {
        height: 1;
        padding: 0 1;
        background: $panel;
        color: $text;
    }
    
    FeatureStatus.enabled {
        background: $success;
        color: $text;
    }
    
    FeatureStatus.disabled {
        background: $error;
        color: $text;
    }
    
    FeatureStatus.loading {
        background: $warning;
        color: $text;
    }
    """
    
    status = reactive("disabled")
    
    def __init__(self, feature_name: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.feature_name = feature_name
    
    def render(self) -> str:
        status_icons = {
            "enabled": "✓",
            "disabled": "✗",
            "loading": "⟳",
        }
        icon = status_icons.get(self.status, "?")
        return f"{icon} {self.feature_name}: {self.status.upper()}"
    
    def watch_status(self, new_status: str) -> None:
        """Update CSS class when status changes."""
        self.remove_class("enabled", "disabled", "loading")
        self.add_class(new_status)


class BaseFeaturePanel(Container):
    """Base class for all feature panels with consistent UI patterns.
    
    Provides:
    - Status indicator
    - Title bar
    - Content area
    - Consistent styling
    - 60 FPS rendering support
    """
    
    DEFAULT_CSS = """
    BaseFeaturePanel {
        height: 100%;
        border: solid $primary;
        background: $surface;
    }
    
    BaseFeaturePanel > Vertical {
        height: 100%;
    }
    
    .feature-title {
        height: 3;
        padding: 1;
        background: $primary;
        color: $text;
        text-style: bold;
    }
    
    .feature-content {
        height: 1fr;
        padding: 1;
        overflow-y: auto;
    }
    
    .feature-empty {
        height: 100%;
        align: center middle;
        color: $text-muted;
    }
    """
    
    def __init__(
        self,
        feature_name: str,
        title: str,
        *args,
        **kwargs
    ):
        """Initialize base feature panel.
        
        Args:
            feature_name: Internal feature name
            title: Display title for the panel
        """
        super().__init__(*args, **kwargs)
        self.feature_name = feature_name
        self.title = title
        self.status_indicator: Optional[FeatureStatus] = None
        self.content_container: Optional[Container] = None
    
    def compose(self):
        """Compose the base panel structure."""
        with Vertical():
            # Status indicator
            self.status_indicator = FeatureStatus(self.feature_name)
            yield self.status_indicator
            
            # Title
            yield Label(self.title, classes="feature-title")
            
            # Content area (to be filled by subclasses)
            self.content_container = Container(classes="feature-content")
            yield self.content_container
    
    def set_status(self, status: str) -> None:
        """Update feature status.
        
        Args:
            status: One of "enabled", "disabled", "loading"
        """
        if self.status_indicator:
            self.status_indicator.status = status
    
    def show_empty_state(self, message: str) -> None:
        """Show empty state message.
        
        Args:
            message: Message to display
        """
        if self.content_container:
            self.content_container.remove_children()
            self.content_container.mount(
                Label(message, classes="feature-empty")
            )
