#!/usr/bin/env python3
"""
Commit Wizard - Interactive multi-step commit flow

A guided wizard for creating commits with AI-generated messages.
"""

from typing import Optional, Callable, Awaitable
from textual.app import ComposeResult
from textual.containers import Container, Vertical, Horizontal, Center
from textual.widgets import Button, Label, TextArea, Static, ProgressBar, Footer
from textual.screen import ModalScreen
from textual.binding import Binding
from textual.reactive import reactive
from rich.text import Text

from xencode.tui.widgets.diff_viewer import DiffViewer


class WizardStep(Static):
    """A step indicator in the wizard"""
    
    DEFAULT_CSS = """
    WizardStep {
        width: auto;
        height: 3;
        padding: 0 2;
        content-align: center middle;
    }
    
    WizardStep.active {
        background: $accent;
        color: $text;
        text-style: bold;
    }
    
    WizardStep.completed {
        background: $success;
        color: $text;
    }
    
    WizardStep.pending {
        background: $surface-darken-1;
        color: $text-muted;
    }
    """
    
    def __init__(self, step_number: int, label: str, **kwargs):
        super().__init__(**kwargs)
        self.step_number = step_number
        self.label = label
        self.add_class("pending")
        
    def render(self) -> Text:
        return Text(f"{self.step_number}. {self.label}")
        
    def set_active(self):
        self.remove_class("pending", "completed")
        self.add_class("active")
        
    def set_completed(self):
        self.remove_class("pending", "active")
        self.add_class("completed")
        
    def set_pending(self):
        self.remove_class("active", "completed")
        self.add_class("pending")


class CommitWizard(ModalScreen):
    """Multi-step commit wizard with AI assistance"""
    
    DEFAULT_CSS = """
    CommitWizard {
        align: center middle;
    }
    
    #wizard-container {
        width: 90%;
        height: 90%;
        border: solid $accent;
        background: $surface;
    }
    
    #wizard-header {
        dock: top;
        height: 5;
        background: $primary-darken-2;
        padding: 1;
    }
    
    #step-indicators {
        height: 3;
        align: center middle;
    }
    
    #wizard-content {
        height: 1fr;
        padding: 1 2;
    }
    
    #wizard-footer {
        dock: bottom;
        height: 3;
        background: $surface-darken-1;
        padding: 0 2;
        align: right middle;
    }
    
    .step-content {
        height: 100%;
    }
    
    #diff-step, #message-step, #confirm-step {
        display: none;
    }
    
    #diff-step.visible, #message-step.visible, #confirm-step.visible {
        display: block;
    }
    
    #message-input {
        height: 1fr;
        margin: 1 0;
    }
    
    #ai-status {
        height: 3;
        background: $primary-darken-1;
        padding: 0 1;
        margin-bottom: 1;
    }
    
    #summary-panel {
        border: solid $success;
        padding: 1;
        height: 1fr;
    }
    
    Button {
        margin-left: 1;
    }
    """
    
    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
        Binding("left", "prev_step", "Previous"),
        Binding("right", "next_step", "Next"),
    ]
    
    current_step = reactive(1)
    
    def __init__(self, diff: str, ai_generator: Optional[Callable[[str], Awaitable[str]]] = None):
        """Initialize wizard
        
        Args:
            diff: The staged diff content
            ai_generator: Async function to generate commit message
        """
        super().__init__()
        self.diff = diff
        self.ai_generator = ai_generator
        self.commit_message = ""
        self.ai_generated = False
        
    def compose(self) -> ComposeResult:
        with Container(id="wizard-container"):
            # Header with step indicators
            with Container(id="wizard-header"):
                yield Label("🎯 Commit Wizard", classes="title")
                with Horizontal(id="step-indicators"):
                    yield WizardStep(1, "Review Changes", id="step-1")
                    yield WizardStep(2, "Write Message", id="step-2")
                    yield WizardStep(3, "Confirm", id="step-3")
            
            # Content area
            with Container(id="wizard-content"):
                # Step 1: Review Diff
                with Vertical(id="diff-step", classes="step-content visible"):
                    yield Label("📝 Review your staged changes:", classes="dim")
                    yield DiffViewer(self.diff)
                
                # Step 2: Message
                with Vertical(id="message-step", classes="step-content"):
                    yield Label("✍️ Enter your commit message:", classes="dim")
                    with Container(id="ai-status"):
                        yield Label("💡 Click 'Generate with AI' for suggestions", id="ai-hint")
                    yield TextArea("", id="message-input", language="markdown")
                    yield Button("✨ Generate with AI", id="btn-generate", variant="primary")
                
                # Step 3: Confirm
                with Vertical(id="confirm-step", classes="step-content"):
                    yield Label("✅ Ready to commit:", classes="dim")
                    with Container(id="summary-panel"):
                        yield Label("", id="summary-message")
                        yield Label("", id="summary-stats")
            
            # Footer with navigation
            with Horizontal(id="wizard-footer"):
                yield Button("Cancel", id="btn-cancel", variant="default")
                yield Button("← Back", id="btn-prev", variant="default")
                yield Button("Next →", id="btn-next", variant="primary")
                yield Button("🚀 Commit", id="btn-commit", variant="success")
                
    def on_mount(self) -> None:
        """Initialize wizard state"""
        self._update_step_display()
        
    def watch_current_step(self, step: int) -> None:
        """React to step changes"""
        self._update_step_display()
        
    def _update_step_display(self) -> None:
        """Update UI based on current step"""
        # Update step indicators
        for i in range(1, 4):
            indicator = self.query_one(f"#step-{i}", WizardStep)
            if i < self.current_step:
                indicator.set_completed()
            elif i == self.current_step:
                indicator.set_active()
            else:
                indicator.set_pending()
                
        # Show/hide content
        for step_id in ["diff-step", "message-step", "confirm-step"]:
            container = self.query_one(f"#{step_id}")
            container.remove_class("visible")
            
        if self.current_step == 1:
            self.query_one("#diff-step").add_class("visible")
        elif self.current_step == 2:
            self.query_one("#message-step").add_class("visible")
            self.query_one("#message-input").focus()
        elif self.current_step == 3:
            self.query_one("#confirm-step").add_class("visible")
            self._update_summary()
            
        # Update button visibility
        self.query_one("#btn-prev").display = self.current_step > 1
        self.query_one("#btn-next").display = self.current_step < 3
        self.query_one("#btn-commit").display = self.current_step == 3
        
    def _update_summary(self) -> None:
        """Update the confirmation summary"""
        message = self.query_one("#message-input", TextArea).text
        self.commit_message = message
        
        # Calculate stats
        lines = self.diff.splitlines()
        additions = sum(1 for l in lines if l.startswith('+') and not l.startswith('+++'))
        deletions = sum(1 for l in lines if l.startswith('-') and not l.startswith('---'))
        
        summary_msg = self.query_one("#summary-message", Label)
        summary_msg.update(f"📝 Message:\n{message[:200]}{'...' if len(message) > 200 else ''}")
        
        summary_stats = self.query_one("#summary-stats", Label)
        ai_tag = " (AI Generated)" if self.ai_generated else ""
        summary_stats.update(f"\n📊 Changes: +{additions} -{deletions}{ai_tag}")
        
    async def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id
        
        if button_id == "btn-cancel":
            self.dismiss(None)
        elif button_id == "btn-prev":
            self.action_prev_step()
        elif button_id == "btn-next":
            self.action_next_step()
        elif button_id == "btn-commit":
            self._do_commit()
        elif button_id == "btn-generate":
            await self._generate_message()
            
    def action_cancel(self) -> None:
        self.dismiss(None)
        
    def action_prev_step(self) -> None:
        if self.current_step > 1:
            self.current_step -= 1
            
    def action_next_step(self) -> None:
        if self.current_step == 2:
            # Validate message before proceeding
            message = self.query_one("#message-input", TextArea).text
            if not message.strip():
                self.notify("Please enter a commit message", severity="warning")
                return
        if self.current_step < 3:
            self.current_step += 1
            
    def _do_commit(self) -> None:
        """Submit the commit"""
        if self.commit_message.strip():
            self.dismiss(self.commit_message)
        else:
            self.notify("No commit message!", severity="error")
            
    async def _generate_message(self) -> None:
        """Generate commit message using AI"""
        if not self.ai_generator:
            self.notify("AI generator not available", severity="error")
            return
            
        input_area = self.query_one("#message-input", TextArea)
        ai_hint = self.query_one("#ai-hint", Label)
        
        ai_hint.update("⏳ Generating message...")
        input_area.disabled = True
        
        try:
            message = await self.ai_generator(self.diff)
            input_area.text = message
            self.ai_generated = True
            ai_hint.update("✅ AI message generated! Feel free to edit.")
        except Exception as e:
            self.notify(f"Failed: {e}", severity="error")
            ai_hint.update("❌ Generation failed. Try again or write manually.")
        finally:
            input_area.disabled = False
            input_area.focus()
