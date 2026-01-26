#!/usr/bin/env python3
"""
ByteBot Panel Widget for Xencode TUI

Interactive ByteBot interface with mode selection and execution log.
"""

from datetime import datetime
from typing import Optional

from rich.text import Text
from textual.widgets import Input, Static, Label, Button
from textual.containers import Container, VerticalScroll, Horizontal, Vertical
from textual.message import Message


class ByteBotLog(VerticalScroll):
    """Scrollable log for ByteBot results"""

    DEFAULT_CSS = """
    ByteBotLog {
        height: 1fr;
        padding: 1;
        background: $surface;
        border: solid $accent;
    }
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.border_title = "🧠 ByteBot Log"

    def add_entry(self, content: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        text = Text()
        text.append(f"[{timestamp}] ", style="dim")
        text.append(content)
        self.mount(Static(text))
        self.scroll_end(animate=True)

    def clear(self) -> None:
        for child in list(self.children):
            child.remove()


class ByteBotTaskSubmitted(Message):
    """Message sent when ByteBot task is submitted"""

    def __init__(self, intent: str, mode: str) -> None:
        super().__init__()
        self.intent = intent
        self.mode = mode


class ByteBotPanel(Container):
    """ByteBot control panel"""

    DEFAULT_CSS = """
    ByteBotPanel {
        height: 100%;
        background: $surface;
        border: solid $primary;
        padding: 1;
    }

    ByteBotPanel .section-title {
        text-style: bold;
        color: $accent;
        margin-bottom: 1;
    }

    ByteBotPanel #controls {
        margin: 1 0;
    }

    ByteBotPanel Button {
        margin: 0 1;
    }
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.status_label: Optional[Label] = None
        self.input_field: Optional[Input] = None
        self.bytebot_log: Optional[ByteBotLog] = None
        self.current_mode = "assist"

    def compose(self):
        yield Label("🧠 ByteBot", classes="section-title")
        self.status_label = Label("Mode: assist | Status: idle", id="bytebot-status")
        yield self.status_label

        self.input_field = Input(placeholder="Enter intent... (Enter to run)")
        yield self.input_field

        with Horizontal(id="controls"):
            yield Button("Assist", id="mode-assist", variant="default")
            yield Button("Execute", id="mode-execute", variant="warning")
            yield Button("Autonomous", id="mode-autonomous", variant="error")
            yield Button("Run", id="run", variant="primary")
            yield Button("Clear", id="clear", variant="default")

        self.bytebot_log = ByteBotLog()
        yield self.bytebot_log

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id
        if button_id == "mode-assist":
            self.set_mode("assist")
        elif button_id == "mode-execute":
            self.set_mode("execute")
        elif button_id == "mode-autonomous":
            self.set_mode("autonomous")
        elif button_id == "run":
            self.submit_intent()
        elif button_id == "clear":
            if self.bytebot_log:
                self.bytebot_log.clear()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.submit_intent()

    def set_mode(self, mode: str) -> None:
        if mode in ("assist", "execute", "autonomous"):
            self.current_mode = mode
            self._update_status("idle")

    def _update_status(self, status: str) -> None:
        if self.status_label:
            self.status_label.update(f"Mode: {self.current_mode} | Status: {status}")

    def submit_intent(self) -> None:
        if not self.input_field:
            return

        intent = self.input_field.value.strip()
        if not intent:
            return

        self.input_field.value = ""
        self._update_status("running")
        self.post_message(ByteBotTaskSubmitted(intent, self.current_mode))

    def log_result(self, content: str) -> None:
        if self.bytebot_log:
            self.bytebot_log.add_entry(content)

    def set_idle(self) -> None:
        self._update_status("idle")
