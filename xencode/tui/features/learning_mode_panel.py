"""Learning Mode TUI panel."""

from textual.containers import Container, Vertical, Horizontal, ScrollableContainer
from textual.widgets import Button, Label, Static, ProgressBar, ListView, ListItem
from textual.reactive import reactive
from typing import Optional, List, Dict, Any

from .base_feature_panel import BaseFeaturePanel


class TopicCard(ListItem):
    """Card for a learning topic."""
    
    DEFAULT_CSS = """
    TopicCard {
        height: auto;
        padding: 1;
        margin: 0 0 1 0;
        border: solid $accent;
        background: $panel;
    }
    
    TopicCard:hover {
        background: $primary;
    }
    
    .topic-title {
        text-style: bold;
        color: $accent;
    }
    
    .topic-progress {
        color: $success;
    }
    """
    
    def __init__(self, title: str, progress: int, difficulty: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.topic_title = title
        self.progress = progress
        self.difficulty = difficulty
    
    def compose(self):
        yield Label(f"[bold]{self.topic_title}[/bold]")
        yield Label(f"Difficulty: {self.difficulty} | Progress: {self.progress}%")
        yield ProgressBar(total=100, show_eta=False)


class LearningModePanel(BaseFeaturePanel):
    """Panel for interactive learning and tutorials."""
    
    DEFAULT_CSS = """
    LearningModePanel {
        height: 100%;
    }
    
    .learning-controls {
        height: auto;
        padding: 1;
        background: $panel;
    }
    
    .learning-content {
        height: 1fr;
        padding: 1;
    }
    
    .topics-list {
        height: 1fr;
        border: solid $primary;
    }
    
    .tutorial-view {
        height: 1fr;
        padding: 1;
        border: solid $accent;
        background: $panel;
    }
    
    .progress-section {
        height: auto;
        padding: 1;
        margin-top: 1;
        border: solid $success;
    }
    """
    
    learning = reactive(False)
    current_topic = reactive(None)
    
    def __init__(self, *args, **kwargs):
        super().__init__(
            feature_name="learning_mode",
            title="🎓 Learning Mode",
            *args,
            **kwargs
        )
        self.topics: List[Dict[str, Any]] = []
        self.current_tutorial: Optional[Dict[str, Any]] = None
    
    def compose(self):
        """Compose the learning mode panel."""
        yield from super().compose()
    
    def on_mount(self) -> None:
        """Initialize panel on mount."""
        self.set_status("enabled")
        self._load_topics()
        self._build_content()
    
    def _load_topics(self) -> None:
        """Load available learning topics."""
        # TODO: Load from actual learning mode feature
        self.topics = [
            {"title": "Python Basics", "progress": 75, "difficulty": "Beginner"},
            {"title": "JavaScript Fundamentals", "progress": 30, "difficulty": "Beginner"},
            {"title": "Rust Ownership", "progress": 0, "difficulty": "Intermediate"},
            {"title": "Advanced TypeScript", "progress": 50, "difficulty": "Advanced"},
        ]
    
    def _build_content(self) -> None:
        """Build the panel content."""
        if not self.content_container:
            return
        
        self.content_container.remove_children()
        
        with self.content_container:
            # Controls
            with Horizontal(classes="learning-controls"):
                yield Button("Browse Topics", id="btn-topics", variant="primary")
                yield Button("My Progress", id="btn-progress")
                yield Button("Start Tutorial", id="btn-start")
            
            # Content area
            with ScrollableContainer(classes="learning-content"):
                if self.current_tutorial:
                    self._render_tutorial()
                else:
                    self._render_topics()
    
    def _render_topics(self) -> None:
        """Render available topics."""
        yield Label("Available Learning Topics", classes="topic-title")
        
        topics_list = ListView(classes="topics-list")
        for topic in self.topics:
            topics_list.append(
                TopicCard(
                    topic["title"],
                    topic["progress"],
                    topic["difficulty"]
                )
            )
        yield topics_list
    
    def _render_tutorial(self) -> None:
        """Render current tutorial."""
        if not self.current_tutorial:
            return
        
        with Container(classes="tutorial-view"):
            yield Label(self.current_tutorial["title"], classes="topic-title")
            yield Label(self.current_tutorial.get("content", "Tutorial content..."))
            
            with Horizontal():
                yield Button("Previous", id="btn-prev")
                yield Button("Next", id="btn-next", variant="primary")
                yield Button("Complete", id="btn-complete", variant="success")
    
    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_id = event.button.id
        
        if button_id == "btn-topics":
            self.current_tutorial = None
            self._build_content()
        elif button_id == "btn-progress":
            await self._show_progress()
        elif button_id == "btn-start":
            await self._start_tutorial()
        elif button_id == "btn-next":
            await self._next_step()
        elif button_id == "btn-prev":
            await self._prev_step()
        elif button_id == "btn-complete":
            await self._complete_tutorial()
    
    async def _start_tutorial(self) -> None:
        """Start a tutorial."""
        if self.topics:
            self.current_tutorial = {
                "title": self.topics[0]["title"],
                "content": "Welcome to the tutorial! Let's begin...",
            }
            self._build_content()
    
    async def _show_progress(self) -> None:
        """Show learning progress."""
        # TODO: Implement progress display
        pass
    
    async def _next_step(self) -> None:
        """Move to next tutorial step."""
        # TODO: Implement step navigation
        pass
    
    async def _prev_step(self) -> None:
        """Move to previous tutorial step."""
        # TODO: Implement step navigation
        pass
    
    async def _complete_tutorial(self) -> None:
        """Complete current tutorial."""
        self.current_tutorial = None
        self._build_content()
