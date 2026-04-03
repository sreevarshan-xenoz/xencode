"""Learning Mode TUI panel."""

from textual.containers import Container, Vertical, Horizontal, ScrollableContainer
from textual.widgets import Button, Label, Static, ProgressBar, ListView, ListItem, TextArea, Select
from textual.reactive import reactive
from typing import Optional, List, Dict, Any
import asyncio

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
    
    .exercise-workspace {
        height: 1fr;
        padding: 1;
        border: solid $warning;
    }
    
    .code-editor {
        height: 1fr;
        border: solid $accent;
    }
    
    .exercise-info {
        height: auto;
        padding: 1;
        background: $panel;
        margin-bottom: 1;
    }
    """
    
    learning = reactive(False)
    current_topic = reactive(None)
    current_exercise = reactive(None)
    view_mode = reactive("topics")  # topics, tutorial, exercises, progress
    
    def __init__(self, *args, **kwargs):
        super().__init__(
            feature_name="learning_mode",
            title="🎓 Learning Mode",
            *args,
            **kwargs
        )
        self.topics: List[Dict[str, Any]] = []
        self.current_tutorial: Optional[Dict[str, Any]] = None
        self.exercises: List[Dict[str, Any]] = []
        self.progress_data: Optional[Dict[str, Any]] = None
    
    def compose(self):
        """Compose the learning mode panel."""
        yield from super().compose()
    
    def on_mount(self) -> None:
        """Initialize panel on mount."""
        self.set_status("enabled")
        asyncio.create_task(self._load_data())
    
    async def _load_data(self) -> None:
        """Load topics and progress data."""
        try:
            from xencode.features import FeatureManager
            
            feature_manager = FeatureManager()
            learning_feature = feature_manager.get_feature('learning_mode')
            
            if learning_feature:
                # Load topics
                self.topics = await learning_feature.get_topics()
                
                # Load progress
                self.progress_data = await learning_feature.get_progress()
                
                # Build initial content
                self._build_content()
        except Exception as e:
            self.set_status("error")
    
    def _build_content(self) -> None:
        """Build the panel content based on current view mode."""
        if not self.content_container:
            return
        
        self.content_container.remove_children()
        
        with self.content_container:
            # Navigation controls
            with Horizontal(classes="learning-controls"):
                yield Button("📚 Topics", id="btn-topics", variant="primary" if self.view_mode == "topics" else "default")
                yield Button("📊 Progress", id="btn-progress", variant="primary" if self.view_mode == "progress" else "default")
                yield Button("📝 Exercises", id="btn-exercises", variant="primary" if self.view_mode == "exercises" else "default")
            
            # Content area based on view mode
            with ScrollableContainer(classes="learning-content"):
                if self.view_mode == "topics":
                    self._render_topics()
                elif self.view_mode == "tutorial":
                    self._render_tutorial()
                elif self.view_mode == "exercises":
                    self._render_exercises()
                elif self.view_mode == "progress":
                    self._render_progress()
    
    def _render_topics(self) -> None:
        """Render available topics."""
        yield Label("📚 Available Learning Topics", classes="topic-title")
        
        if not self.topics:
            yield Label("No topics available. Enable the learning mode feature first.")
            return
        
        topics_list = ListView(classes="topics-list")
        for topic in self.topics:
            progress = 0
            if topic.get('progress'):
                prog = topic['progress']
                if prog['exercises_total'] > 0:
                    progress = int((prog['exercises_completed'] / prog['exercises_total']) * 100)
            
            topics_list.append(
                TopicCard(
                    topic["name"],
                    progress,
                    topic["difficulty"]
                )
            )
        yield topics_list
        
        yield Label("\n💡 Click a topic to start learning", classes="dim")
    
    def _render_tutorial(self) -> None:
        """Render current tutorial."""
        if not self.current_tutorial:
            yield Label("No tutorial selected")
            return
        
        with Container(classes="tutorial-view"):
            yield Label(self.current_tutorial["title"], classes="topic-title")
            yield Label(self.current_tutorial.get("content", "Tutorial content..."))
            
            # Show key concepts
            if self.current_tutorial.get("key_concepts"):
                yield Label("\n🔑 Key Concepts:")
                for concept in self.current_tutorial["key_concepts"]:
                    yield Label(f"  • {concept}")
            
            # Show examples
            if self.current_tutorial.get("examples"):
                yield Label("\n💡 Examples:")
                for example in self.current_tutorial["examples"]:
                    yield Label(example, classes="code-example")
            
            with Horizontal():
                yield Button("⬅️ Previous", id="btn-prev")
                yield Button("➡️ Next", id="btn-next", variant="primary")
                yield Button("✅ Complete", id="btn-complete", variant="success")
    
    def _render_exercises(self) -> None:
        """Render exercise workspace."""
        if not self.current_exercise:
            yield Label("📝 Select a topic to get exercises")
            
            if self.topics:
                yield Label("\nAvailable topics:")
                for topic in self.topics:
                    yield Button(f"Get exercises for {topic['name']}", id=f"exercises-{topic['id']}")
            return
        
        with Container(classes="exercise-workspace"):
            # Exercise info
            with Container(classes="exercise-info"):
                yield Label(f"📝 {self.current_exercise['title']}", classes="topic-title")
                yield Label(self.current_exercise['description'])
                yield Label(f"Difficulty: {self.current_exercise['difficulty']}")
                
                if self.current_exercise.get('hints'):
                    yield Label("\n💡 Hints:")
                    for hint in self.current_exercise['hints']:
                        yield Label(f"  • {hint}")
            
            # Code editor
            yield Label("Your Solution:")
            code_area = TextArea(
                text=self.current_exercise.get('code_template', ''),
                language="python",
                theme="monokai",
                classes="code-editor"
            )
            code_area.id = "exercise-code"
            yield code_area
            
            # Action buttons
            with Horizontal():
                yield Button("▶️ Run Tests", id="btn-run-tests", variant="primary")
                yield Button("💡 Show Hint", id="btn-hint")
                yield Button("✅ Submit", id="btn-submit", variant="success")
                yield Button("⬅️ Back", id="btn-back-exercises")
    
    def _render_progress(self) -> None:
        """Render progress dashboard."""
        yield Label("📊 Your Learning Progress", classes="topic-title")
        
        if not self.progress_data or not self.progress_data.get('topics'):
            yield Label("\nNo progress yet. Start learning to track your progress!")
            return
        
        # Overall stats
        with Container(classes="progress-section"):
            yield Label("Overall Statistics:")
            yield Label(f"  • Overall Mastery: {self.progress_data.get('overall_mastery', 0)*100:.1f}%")
            yield Label(f"  • Total Time: {self.progress_data.get('total_time', 0)} minutes")
            yield Label(f"  • Topics Started: {len(self.progress_data['topics'])}")
        
        # Per-topic progress
        yield Label("\nTopic Progress:")
        for topic_progress in self.progress_data['topics']:
            with Container(classes="progress-section"):
                yield Label(f"📚 {topic_progress['topic_id']}")
                yield Label(f"  • Mastery: {topic_progress['mastery_level']}")
                yield Label(f"  • Exercises: {topic_progress['exercises_completed']}/{topic_progress['exercises_total']}")
                yield Label(f"  • Accuracy: {topic_progress['accuracy']*100:.1f}%")
                yield Label(f"  • Time: {topic_progress['time_spent']} minutes")
                
                # Progress bar
                if topic_progress['exercises_total'] > 0:
                    progress_pct = (topic_progress['exercises_completed'] / topic_progress['exercises_total']) * 100
                    yield ProgressBar(total=100, progress=progress_pct, show_eta=False)
    
    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_id = event.button.id
        
        if button_id == "btn-topics":
            self.view_mode = "topics"
            self._build_content()
        elif button_id == "btn-progress":
            self.view_mode = "progress"
            await self._refresh_progress()
            self._build_content()
        elif button_id == "btn-exercises":
            self.view_mode = "exercises"
            self._build_content()
        elif button_id == "btn-start":
            await self._start_tutorial()
        elif button_id == "btn-next":
            await self._next_step()
        elif button_id == "btn-prev":
            await self._prev_step()
        elif button_id == "btn-complete":
            await self._complete_tutorial()
        elif button_id == "btn-run-tests":
            await self._run_tests()
        elif button_id == "btn-hint":
            await self._show_hint()
        elif button_id == "btn-submit":
            await self._submit_exercise()
        elif button_id == "btn-back-exercises":
            self.current_exercise = None
            self._build_content()
        elif button_id and button_id.startswith("exercises-"):
            topic_id = button_id.replace("exercises-", "")
            await self._load_exercises(topic_id)
    
    async def on_list_view_selected(self, event: ListView.Selected) -> None:
        """Handle topic selection."""
        if isinstance(event.item, TopicCard):
            # Get the topic index
            topic_idx = self.topics.index(next(t for t in self.topics if t['name'] == event.item.topic_title))
            await self._start_topic(self.topics[topic_idx]['id'])
    
    async def _start_topic(self, topic_id: str) -> None:
        """Start a tutorial for a topic."""
        try:
            from xencode.features import FeatureManager
            
            feature_manager = FeatureManager()
            learning_feature = feature_manager.get_feature('learning_mode')
            
            if learning_feature:
                result = await learning_feature.start_topic(topic_id)
                self.current_tutorial = {
                    "title": result['topic']['name'],
                    "content": result['lesson'].get('content', ''),
                    "key_concepts": result['lesson'].get('key_concepts', []),
                    "examples": result['lesson'].get('examples', [])
                }
                self.current_topic = topic_id
                self.view_mode = "tutorial"
                self._build_content()
        except Exception as e:
            pass
    
    async def _load_exercises(self, topic_id: str) -> None:
        """Load exercises for a topic."""
        try:
            from xencode.features import FeatureManager
            
            feature_manager = FeatureManager()
            learning_feature = feature_manager.get_feature('learning_mode')
            
            if learning_feature:
                self.exercises = await learning_feature.get_exercises(topic_id, 5)
                if self.exercises:
                    self.current_exercise = self.exercises[0]
                    self.current_topic = topic_id
                    self._build_content()
        except Exception as e:
            pass
    
    async def _refresh_progress(self) -> None:
        """Refresh progress data."""
        try:
            from xencode.features import FeatureManager
            
            feature_manager = FeatureManager()
            learning_feature = feature_manager.get_feature('learning_mode')
            
            if learning_feature:
                self.progress_data = await learning_feature.get_progress()
        except Exception as e:
            pass
    
    async def _start_tutorial(self) -> None:
        """Start a tutorial."""
        if self.topics:
            await self._start_topic(self.topics[0]['id'])
    
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
        self.view_mode = "topics"
        self._build_content()
    
    async def _run_tests(self) -> None:
        """Run tests for current exercise."""
        # TODO: Implement test execution
        pass
    
    async def _show_hint(self) -> None:
        """Show a hint for current exercise."""
        # TODO: Implement hint display
        pass
    
    async def _submit_exercise(self) -> None:
        """Submit current exercise solution."""
        try:
            from xencode.features import FeatureManager
            
            # Get code from editor
            code_area = self.query_one("#exercise-code", TextArea)
            solution = code_area.text
            
            feature_manager = FeatureManager()
            learning_feature = feature_manager.get_feature('learning_mode')
            
            if learning_feature and self.current_exercise:
                result = await learning_feature.submit_exercise(
                    self.current_exercise['id'],
                    solution
                )
                
                # Show result
                # TODO: Display result in a modal or panel
                
                # Move to next exercise if passed
                if result['passed'] and self.exercises:
                    current_idx = self.exercises.index(self.current_exercise)
                    if current_idx < len(self.exercises) - 1:
                        self.current_exercise = self.exercises[current_idx + 1]
                        self._build_content()
        except Exception as e:
            pass

