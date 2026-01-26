#!/usr/bin/env python3
"""
Agent Panel Widget for Xencode TUI

Interactive agent interface with status display and tool usage tracking.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any

from rich.console import RenderableType

from rich.text import Text
from rich.table import Table
from textual import events
from textual.widgets import Input, Static, Label, Button
from textual.containers import Container, VerticalScroll, Horizontal
from textual.message import Message


class AgentStatus(Static):
    """Display current agent status"""
    
    DEFAULT_CSS = """
    AgentStatus {
        height: 5;
        padding: 1;
        background: $panel;
        border: solid $accent;
    }
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_model = "qwen3:4b"
        self.agent_type = "general"
        self.is_thinking = False
        
    def render(self) -> RenderableType:
        """Render agent status"""
        status_text = Text()
        
        status_text.append("🤖 Agent: ", style="bold")
        status_text.append(f"{self.agent_type}\n", style="cyan")
        
        status_text.append("📦 Model: ", style="bold")
        status_text.append(f"{self.current_model}\n", style="green")
        
        status_text.append("💭 Status: ", style="bold")
        if self.is_thinking:
            status_text.append("Thinking...", style="yellow blink")
        else:
            status_text.append("Ready", style="green")
        
        return status_text
    
    def update_status(self, model: str = None, agent_type: str = None, is_thinking: bool = None):
        """Update agent status"""
        if model:
            self.current_model = model
        if agent_type:
            self.agent_type = agent_type
        if is_thinking is not None:
            self.is_thinking = is_thinking
        self.refresh()


class ToolUsageLog(VerticalScroll):
    """Display tool usage log"""
    
    DEFAULT_CSS = """
    ToolUsageLog {
        height: 1fr;
        padding: 1;
        background: $surface;
        border: solid $accent;
    }
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.border_title = "🔧 Tool Usage"
    
    def add_tool_call(self, tool_name: str, tool_input: str, result: str = None):
        """Add a tool call to the log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        log_entry = Text()
        log_entry.append(f"[{timestamp}] ", style="dim")
        log_entry.append(f"{tool_name}", style="bold cyan")
        log_entry.append(f"({tool_input[:30]}...)\n", style="yellow")
        
        if result:
            log_entry.append(f"  → {result[:100]}...\n", style="green")
        
        entry_widget = Static(log_entry)
        self.mount(entry_widget)
        self.scroll_end(animate=True)


class AgentPanel(Container):
    """Complete agent interface panel"""

    DEFAULT_CSS = """
    AgentPanel {
        height: 100%;
        background: $surface;
    }

    AgentPanel Horizontal {
        height: auto;
        margin: 1 0;
    }

    AgentPanel Button {
        margin: 0 1;
    }

    AgentPanel #collaboration-options {
        height: auto;
        padding: 1;
        background: $panel;
        border: solid $secondary;
        margin: 1 0;
    }
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.status: Optional[AgentStatus] = None
        self.tool_log: Optional[ToolUsageLog] = None
        self.input_field: Optional[Input] = None
        self.collaboration_selector: Optional[Input] = None
        self.show_collaboration_options = False

    def compose(self):
        """Compose the agent panel"""
        self.status = AgentStatus()
        self.tool_log = ToolUsageLog()
        self.input_field = Input(placeholder="Enter task for agent... (Enter to submit)")

        # Collaboration options (initially hidden)
        self.collaboration_selector = Input(
            placeholder="Enter agent sequence (comma-separated): e.g., planning,code,general",
            id="collaboration-input",
            classes="hidden"
        )

        # Control buttons
        control_panel = Horizontal(
            Button("Run Task", id="run_task", variant="primary"),
            Button("Multi-Agent", id="multi_agent", variant="success"),
            Button("Sequential", id="sequential", variant="warning"),
            Button("Adaptive", id="adaptive", variant="default"),
            Button("Clear Log", id="clear_log", variant="warning"),
        )

        # Collaboration options panel (initially hidden)
        collaboration_options = Horizontal(
            self.collaboration_selector,
            Button("Apply", id="apply_collab", variant="primary"),
            Button("Cancel", id="cancel_collab", variant="error"),
            id="collaboration-options",
            classes="hidden"
        )

        yield self.status
        yield self.input_field
        yield control_panel
        yield collaboration_options
        yield self.tool_log

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses"""
        if event.button.id == "run_task":
            self.submit_task()
        elif event.button.id == "multi_agent":
            self.run_multi_agent()
        elif event.button.id == "sequential":
            self.toggle_sequential_mode()
        elif event.button.id == "adaptive":
            self.run_adaptive_collaboration()
        elif event.button.id == "clear_log":
            self.clear_log()
        elif event.button.id == "apply_collab":
            self.apply_sequential_collaboration()
        elif event.button.id == "cancel_collab":
            self.hide_collaboration_options()

    def toggle_sequential_mode(self):
        """Toggle sequential collaboration options"""
        self.show_collaboration_options = not self.show_collaboration_options

        collab_input = self.query_one("#collaboration-input")
        collab_options = self.query_one("#collaboration-options")

        if self.show_collaboration_options:
            collab_input.remove_class("hidden")
            collab_options.remove_class("hidden")
            collab_input.focus()
        else:
            collab_input.add_class("hidden")
            collab_options.add_class("hidden")

    def hide_collaboration_options(self):
        """Hide collaboration options"""
        self.show_collaboration_options = False
        collab_input = self.query_one("#collaboration-input")
        collab_options = self.query_one("#collaboration-options")
        collab_input.add_class("hidden")
        collab_options.add_class("hidden")
        self.input_field.focus()

    def apply_sequential_collaboration(self):
        """Apply sequential collaboration with specified agents"""
        if not self.collaboration_selector:
            return

        agent_sequence_str = self.collaboration_selector.value.strip()
        if not agent_sequence_str:
            self.notify("Please enter an agent sequence", severity="warning")
            return

        # Parse agent sequence
        agent_names = [name.strip() for name in agent_sequence_str.split(',')]

        # Convert to AgentType enums
        from xencode.agentic.coordinator import AgentType
        agent_types = []

        for name in agent_names:
            try:
                agent_type = AgentType(name.upper())
                agent_types.append(agent_type)
            except ValueError:
                self.notify(f"Invalid agent type: {name}", severity="error")
                return

        if not agent_types:
            self.notify("No valid agent types specified", severity="error")
            return

        # Submit task with sequential collaboration
        task = self.input_field.value.strip()
        if not task:
            self.notify("Please enter a task", severity="warning")
            return

        self.input_field.value = ""
        self.hide_collaboration_options()

        # Update status
        if self.status:
            self.status.update_status(is_thinking=True)

        # Post message for sequential processing
        self.post_message(AgentTaskSubmitted(task, use_multi_agent=False, collaboration_type="sequential", agent_sequence=agent_types))

    def run_adaptive_collaboration(self):
        """Run task with adaptive collaboration system"""
        if not self.input_field:
            return

        task = self.input_field.value.strip()
        if not task:
            return

        # Clear input
        self.input_field.value = ""

        # Update status
        if self.status:
            self.status.update_status(is_thinking=True)

        # Post message for adaptive processing
        self.post_message(AgentTaskSubmitted(task, use_multi_agent=False, collaboration_type="adaptive"))

    def submit_task(self):
        """Submit task to agent"""
        if not self.input_field:
            return

        task = self.input_field.value.strip()
        if not task:
            return

        # Clear input
        self.input_field.value = ""

        # Update status
        if self.status:
            self.status.update_status(is_thinking=True)

        # Post message for processing
        self.post_message(AgentTaskSubmitted(task, use_multi_agent=False))

    def run_multi_agent(self):
        """Run task with multi-agent system"""
        if not self.input_field:
            return

        task = self.input_field.value.strip()
        if not task:
            return

        # Clear input
        self.input_field.value = ""

        # Post message for multi-agent processing
        self.post_message(AgentTaskSubmitted(task, use_multi_agent=True))
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses"""
        if event.button.id == "run_task":
            self.submit_task()
        elif event.button.id == "multi_agent":
            self.run_multi_agent()
        elif event.button.id == "clear_log":
            self.clear_log()
    
    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle input submission"""
        self.submit_task()
    
    def submit_task(self):
        """Submit task to agent"""
        if not self.input_field:
            return
        
        task = self.input_field.value.strip()
        if not task:
            return
        
        # Clear input
        self.input_field.value = ""
        
        # Update status
        if self.status:
            self.status.update_status(is_thinking=True)
        
        # Post message for processing
        self.post_message(AgentTaskSubmitted(task, use_multi_agent=False))
    
    def run_multi_agent(self):
        """Run task with multi-agent system"""
        if not self.input_field:
            return
        
        task = self.input_field.value.strip()
        if not task:
            return
        
        # Clear input
        self.input_field.value = ""
        
        # Post message for multi-agent processing
        self.post_message(AgentTaskSubmitted(task, use_multi_agent=True))
    
    def clear_log(self):
        """Clear tool usage log"""
        if self.tool_log:
            for child in list(self.tool_log.children):
                child.remove()
    
    def log_tool_call(self, tool_name: str, tool_input: str, result: str = None):
        """Log a tool call"""
        if self.tool_log:
            self.tool_log.add_tool_call(tool_name, tool_input, result)
    
    def update_agent_status(self, model: str = None, agent_type: str = None, is_thinking: bool = None):
        """Update agent status display"""
        if self.status:
            self.status.update_status(model, agent_type, is_thinking)


class AgentTaskSubmitted(Message):
    """Message sent when agent task is submitted"""

    def __init__(self, task: str, use_multi_agent: bool = False, collaboration_type: str = None, agent_sequence: List = None) -> None:
        super().__init__()
        self.task = task
        self.use_multi_agent = use_multi_agent
        self.collaboration_type = collaboration_type  # "sequential", "parallel", "adaptive", or None
        self.agent_sequence = agent_sequence or []  # For sequential collaboration
