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
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.status: Optional[AgentStatus] = None
        self.tool_log: Optional[ToolUsageLog] = None
        self.input_field: Optional[Input] = None
    
    def compose(self):
        """Compose the agent panel"""
        self.status = AgentStatus()
        self.tool_log = ToolUsageLog()
        self.input_field = Input(placeholder="Enter task for agent... (Enter to submit)")
        
        # Control buttons
        control_panel = Horizontal(
            Button("Run Task", id="run_task", variant="primary"),
            Button("Multi-Agent", id="multi_agent", variant="success"),
            Button("Clear Log", id="clear_log", variant="warning"),
        )
        
        yield self.status
        yield self.input_field
        yield control_panel
        yield self.tool_log
    
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
    
    def __init__(self, task: str, use_multi_agent: bool = False) -> None:
        super().__init__()
        self.task = task
        self.use_multi_agent = use_multi_agent
