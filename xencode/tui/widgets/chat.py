#!/usr/bin/env python3
"""
Chat Panel Widget for Xencode TUI

AI chat interface with message history and streaming responses.
"""

from datetime import datetime
from typing import Optional

from rich.console import RenderableType, Group
from rich.markdown import Markdown as RichMarkdown
from rich.text import Text
from textual import events
from textual.widgets import Input, Static, Label, Markdown as MarkdownWidget
from textual.containers import Container, VerticalScroll
from textual.message import Message


class ChatMessage(Static):
    """A single chat message widget"""
    
    DEFAULT_CSS = """
    ChatMessage {
        padding: 1;
        margin: 0 1;
    }
    
    ChatMessage.user {
        background: $primary 20%;
        border-left: thick $primary;
    }
    
    ChatMessage.assistant {
        background: $secondary 20%;
        border-left: thick $secondary;
    }
    
    ChatMessage.system {
        background: $warning 20%;
        border-left: thick $warning;
    }
    """
    
    def __init__(
        self,
        role: str,
        content: str,
        timestamp: Optional[datetime] = None,
        *args,
        **kwargs
    ):
        """Initialize chat message
        
        Args:
            role: Message role (user/assistant/system)
            content: Message content
            timestamp: Message timestamp
        """
        super().__init__(*args, **kwargs)
        self.role = role
        self.content_text = content
        self.timestamp = timestamp or datetime.now()
        self.add_class(role)
    
    def render(self) -> RenderableType:
        """Render the message"""
        time_str = self.timestamp.strftime("%H:%M")
        header = Text(f"{self.role.upper()} @ {time_str}", style="bold")
        
        # Render content as markdown if it's from assistant
        if self.role == "assistant":
            content = RichMarkdown(self.content_text)
        else:
            content = Text(self.content_text)
        
        return Group(header, content)


class ChatHistory(VerticalScroll):
    """Scrollable chat history container"""
    
    DEFAULT_CSS = """
    ChatHistory {
        height: 1fr;
        background: $surface;
        border: solid $accent;
    }
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize chat history"""
        super().__init__(*args, **kwargs)
        self.border_title = "💬 Chat History"
    
    def add_message(self, role: str, content: str, sender: str = None) -> ChatMessage:
        """Add a message to the history
        
        Args:
            role: Message role (user/assistant/system)
            content: Message content
            sender: Optional sender name for user messages
            
        Returns:
            The created message widget
        """
        if role == "user":
            display_name = sender if sender else "You"
            message = UserMessage(content, display_name)
            message.add_class("user")
        else:
            message = ChatMessage(role, content)
            
        self.mount(message)
        
        # Scroll to bottom
        self.scroll_end(animate=True)
        
        return message
    
    def update_last_message(self, content: str) -> None:
        """Update the last message content (for streaming)
        
        Args:
            content: New content
        """
        if self.children:
            last_message = self.children[-1]
            if isinstance(last_message, ChatMessage):
                last_message.content_text = content
                last_message.refresh()
    
    def clear_history(self) -> None:
        """Clear all messages"""
        for child in list(self.children):
            child.remove()


class ChatInput(Input):
    """Chat input field"""
    
    DEFAULT_CSS = """
    ChatInput {
        dock: bottom;
        height: 3;
        border: solid $accent;
    }
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize chat input"""
        super().__init__(
            placeholder="Ask Xencode AI anything... (Ctrl+Enter to send)",
            *args,
            **kwargs
        )


class ChatPanel(Container):
    """Complete chat panel with history and input"""
    
    DEFAULT_CSS = """
    ChatPanel {
        height: 100%;
        background: $surface;
        border: none;
    }
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize chat panel"""
        super().__init__(*args, **kwargs)
        self.history: Optional[ChatHistory] = None
        self.input_field: Optional[ChatInput] = None
    
    def compose(self):
        """Compose the chat panel"""
        self.history = ChatHistory()
        self.input_field = ChatInput()
        
        yield self.history
        yield self.input_field
    
    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle input submission
        
        Args:
            event: The submission event
        """
        if not self.input_field:
            return
        
        user_message = self.input_field.value.strip()
        
        if not user_message:
            return
        
        # Clear input
        self.input_field.value = ""
        
        # Add user message to history
        if self.history:
            self.history.add_message("user", user_message)
        
        # Post message to app for processing
        self.post_message(ChatSubmitted(user_message))
    
    def add_message(self, role: str, content: str, sender: str = None) -> None:
        """Add a message to the chat
        
        Args:
            role: 'user' or 'assistant'
            content: Message content
            sender: Optional name of sender (for collaboration)
        """
        if self.history:
            self.history.add_message(role, content, sender)
    
    def add_user_message(self, content: str, sender: str = None) -> None:
        """Add a user message
        
        Args:
            content: Message content
            sender: Optional sender name
        """
        if self.history:
            self.history.add_message("user", content, sender)
    
    def add_assistant_message(self, content: str) -> ChatMessage:
        """Add an assistant message
        
        Args:
            content: Message content
            
        Returns:
            The created message widget
        """
        if self.history:
            return self.history.add_message("assistant", content)
        return None
    
    def add_system_message(self, content: str) -> None:
        """Add a system message
        
        Args:
            content: Message content
        """
        if self.history:
            self.history.add_message("system", content)
    
    def update_streaming_message(self, content: str) -> None:
        """Update the streaming assistant message
        
        Args:
            content: Updated content
        """
        if self.history:
            self.history.update_last_message(content)


class UserMessage(Static):
    """User message bubble"""
    
    def __init__(self, content: str, sender: str = "You"):
        super().__init__()
        self.content_text = content  # Rename to content_text to be consistent if needed, but 'content' is fine
        self.sender = sender
        
    def compose(self):
        yield Label(f"👤 {self.sender}", classes="message-author")
        yield MarkdownWidget(self.content_text)


class ChatSubmitted(Message):
    """Message sent when chat is submitted"""
    
    def __init__(self, content: str) -> None:
        """Initialize message
        
        Args:
            content: The submitted message content
        """
        super().__init__()
        self.content = content
