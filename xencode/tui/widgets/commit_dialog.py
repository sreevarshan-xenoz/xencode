#!/usr/bin/env python3
"""
Commit Dialog Widget
"""

from textual.app import ComposeResult
from textual.containers import Container, Vertical, Horizontal
from textual.widgets import Button, Label, TextArea, Static
from textual.screen import ModalScreen
from textual.binding import Binding
from textual.message import Message

class CommitDialog(ModalScreen):
    """Modal dialog for committing changes with AI assistance"""
    
    DEFAULT_CSS = """
    CommitDialog {
        align: center middle;
    }
    
    #dialog {
        width: 80%;
        height: 80%;
        border: solid $accent;
        background: $surface;
        padding: 1 2;
    }
    
    #title {
        text-style: bold;
        margin-bottom: 1;
    }
    
    #diff-preview {
        height: 1fr;
        border: solid $primary;
        margin-bottom: 1;
        background: $surface-darken-1;
    }
    
    #message-input {
        height: 10;
        margin-bottom: 1;
    }
    
    #buttons {
        height: 3;
        align: right middle;
    }
    
    Button {
        margin-left: 2;
    }
    """
    
    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
        Binding("ctrl+enter", "commit", "Commit"),
    ]
    
    def __init__(self, diff: str, ai_generator=None):
        """Initialize dialog
        
        Args:
            diff: The staged diff content
            ai_generator: Async function to generate message from diff
        """
        super().__init__()
        self.diff = diff
        self.ai_generator = ai_generator
        self.generated_message = ""
        
    def compose(self) -> ComposeResult:
        with Container(id="dialog"):
            yield Label("Commit Changes", id="title")
            
            yield Label("Staged Changes:", classes="dim")
            yield TextArea(self.diff, id="diff-preview", read_only=True, language="diff")
            
            yield Label("Commit Message (Ctrl+Enter to commit):", classes="dim")
            yield TextArea("", id="message-input")
            
            with Horizontal(id="buttons"):
                yield Button("✨ Generate with AI", id="btn-generate", variant="primary")
                yield Button("Cancel", id="btn-cancel")
                yield Button("Commit", id="btn-commit", variant="success")
    
    def on_mount(self) -> None:
        """Focus input on mount"""
        self.query_one("#message-input").focus()
        
    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-cancel":
            self.dismiss(None)
        elif event.button.id == "btn-commit":
            self._submit()
        elif event.button.id == "btn-generate":
            await self._generate_message()
            
    def action_cancel(self) -> None:
        self.dismiss(None)
        
    def action_commit(self) -> None:
        self._submit()
        
    def _submit(self) -> None:
        message = self.query_one("#message-input", TextArea).text
        if message.strip():
            self.dismiss(message)
        else:
            self.notify("Please enter a commit message", severity="error")
            
    async def _generate_message(self) -> None:
        """Generate message using AI"""
        if not self.ai_generator:
            self.notify("AI generator not available", severity="error")
            return
            
        input_area = self.query_one("#message-input", TextArea)
        input_area.text = "Generating..."
        input_area.disabled = True
        
        try:
            message = await self.ai_generator(self.diff)
            input_area.text = message
        except Exception as e:
            self.notify(f"Failed to generate: {e}", severity="error")
            input_area.text = ""
        finally:
            input_area.disabled = False
            input_area.focus()
