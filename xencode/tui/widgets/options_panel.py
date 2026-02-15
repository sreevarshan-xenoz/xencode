#!/usr/bin/env python3
"""
Options Panel Widget for Xencode TUI

Lists all CLI commands and allows triggering them from within TUI.
"""

from textual.containers import Container, VerticalScroll
from textual.message import Message
from textual.widgets import Button, Label


class OptionsPanel(VerticalScroll):
    """Options panel with command shortcuts."""

    DEFAULT_CSS = """
    OptionsPanel {
        height: 100%;
        background: $surface;
        border: solid $primary;
        padding: 1;
    }

    OptionsPanel .section-title {
        text-style: bold;
        color: $accent;
        margin-bottom: 1;
    }

    OptionsPanel Button {
        width: 100%;
        margin-bottom: 1;
    }
    """

    class CommandRequested(Message):
        """Command execution requested from options panel."""

        def __init__(self, command: str) -> None:
            super().__init__()
            self.command = command

    COMMANDS = [
        ("tui", "Launch TUI"),
        ("agentic", "Start interactive agent session"),
        ("query", "Run AI ensemble query"),
        ("init system", "Initialize Xencode systems"),
        ("status", "Show system status"),
        ("health", "Run health check"),
        ("optimize", "Optimize performance"),
        ("version", "Show version"),

        ("ollama list", "List local models"),
        ("ollama optimize", "Optimize model setup"),
        ("rlhf train", "Run RLHF training"),
        ("rag index", "Index project for RAG"),
        ("review file", "Run file code review"),
        ("terminal suggest", "Get terminal suggestions"),
        ("features list", "List feature status"),
        ("bytebot", "Run ByteBot intent engine"),
        ("devops", "Generate DevOps files"),
        ("shadow", "Start shadow mode"),
        ("shell", "Natural language shell"),
    ]

    def compose(self):
        yield Label("🧭 Options", classes="section-title")
        yield Label("Run CLI capabilities from TUI", classes="dim")

        for command, description in self.COMMANDS:
            btn = Button(f"{command}  —  {description}", id=f"cmd-{command.replace(' ', '-')}")
            btn.data = command
            yield btn

    def on_button_pressed(self, event: Button.Pressed) -> None:
        command = getattr(event.button, "data", None)
        if command:
            self.post_message(self.CommandRequested(command))
