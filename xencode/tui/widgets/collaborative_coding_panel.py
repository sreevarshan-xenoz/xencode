#!/usr/bin/env python3
"""
Collaborative Coding Panel Widget for Xencode TUI

Real-time collaborative editing with presence indicators, chat interface,
conflict resolution panel, and user list.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, ScrollableContainer, Vertical
from textual.css.query import NoMatches
from textual.message import Message
from textual.widgets import (
    Button,
    Input,
    Label,
    ListItem,
    ListView,
    TextArea,
)


class PresenceIndicator(Container):
    """Visual indicator for user presence in the session"""

    DEFAULT_CSS = """
    PresenceIndicator {
        height: auto;
        border: solid $primary;
        background: $surface;
        padding: 1;
    }

    .user-item {
        height: 3;
        padding: 0 1;
        margin: 0 0 1 0;
        background: $panel;
        border-left: thick $accent;
    }

    .user-item.active {
        border-left: thick $success;
    }

    .user-item.idle {
        border-left: thick $warning;
    }

    .user-item.offline {
        border-left: thick $error;
    }

    .user-item:hover {
        background: $boost;
    }

    #presence-title {
        dock: top;
        height: 1;
        text-style: bold;
        color: $accent;
    }
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.users: Dict[str, Dict[str, Any]] = {}

    def compose(self) -> ComposeResult:
        """Compose the presence indicator"""
        yield Label("👥 Active Users", id="presence-title")
        yield ListView(id="user-list")

    def add_user(self, user_id: str, username: str, color: str, status: str = "online") -> None:
        """Add a user to the presence list"""
        self.users[user_id] = {
            'username': username,
            'color': color,
            'status': status
        }
        self._refresh_user_list()

    def update_user_status(self, user_id: str, status: str) -> None:
        """Update user status"""
        if user_id in self.users:
            self.users[user_id]['status'] = status
            self._refresh_user_list()

    def remove_user(self, user_id: str) -> None:
        """Remove a user from the presence list"""
        if user_id in self.users:
            del self.users[user_id]
            self._refresh_user_list()

    def _refresh_user_list(self) -> None:
        """Refresh the user list display"""
        try:
            user_list = self.query_one("#user-list", ListView)
        except NoMatches:
            return
        user_list.clear()

        for _user_id, user_data in self.users.items():
            status_icon = {
                'online': '🟢',
                'idle': '🟡',
                'away': '🟠',
                'offline': '⚫'
            }.get(user_data['status'], '⚪')

            user_label = Label(f"{status_icon} {user_data['username']}")
            item = ListItem(user_label)

            # Add status class
            if user_data['status'] == 'online':
                item.add_class('active')
            elif user_data['status'] in ['idle', 'away']:
                item.add_class('idle')
            else:
                item.add_class('offline')

            item.add_class('user-item')
            user_list.append(item)


class CursorTracker(Container):
    """Visual tracker for user cursor positions"""

    DEFAULT_CSS = """
    CursorTracker {
        height: auto;
        border: solid $accent;
        background: $surface;
        padding: 1;
    }

    #cursor-title {
        dock: top;
        height: 1;
        text-style: bold;
        color: $accent;
    }

    .cursor-item {
        height: 2;
        padding: 0 1;
        margin: 0 0 1 0;
    }
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cursors: Dict[str, Dict[str, Any]] = {}

    def compose(self) -> ComposeResult:
        """Compose the cursor tracker"""
        yield Label("📍 Cursor Positions", id="cursor-title")
        yield ScrollableContainer(id="cursor-container")

    def update_cursor(self, user_id: str, username: str, position: int,
                     file_path: Optional[str] = None) -> None:
        """Update cursor position for a user"""
        self.cursors[user_id] = {
            'username': username,
            'position': position,
            'file_path': file_path
        }
        self._refresh_cursors()

    def _refresh_cursors(self) -> None:
        """Refresh cursor display"""
        try:
            container = self.query_one("#cursor-container", ScrollableContainer)
        except NoMatches:
            return
        container.remove_children()

        for _user_id, cursor_data in self.cursors.items():
            file_info = f" in {cursor_data['file_path']}" if cursor_data['file_path'] else ""
            cursor_text = f"{cursor_data['username']}: Line {cursor_data['position']}{file_info}"
            label = Label(cursor_text, classes="cursor-item")
            container.mount(label)


class ChatInterface(Container):
    """Chat interface for collaboration"""

    DEFAULT_CSS = """
    ChatInterface {
        height: 1fr;
        border: solid $primary;
        background: $surface;
        padding: 1;
    }

    #chat-title {
        dock: top;
        height: 1;
        text-style: bold;
        color: $accent;
    }

    #chat-messages {
        height: 1fr;
        border: solid $secondary;
        background: $panel;
        margin: 1 0;
    }

    #chat-input-container {
        dock: bottom;
        height: 3;
    }

    #chat-input {
        width: 1fr;
    }

    #send-button {
        width: 10;
    }

    .message-item {
        height: auto;
        padding: 0 1;
        margin: 0 0 1 0;
    }

    .message-self {
        color: $accent;
    }

    .message-other {
        color: $text;
    }

    .message-system {
        color: $warning;
        text-style: italic;
    }
    """

    class SendMessage(Message):
        """Message sent event"""
        def __init__(self, text: str):
            self.text = text
            super().__init__()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.messages: List[Dict[str, Any]] = []

    def compose(self) -> ComposeResult:
        """Compose the chat interface"""
        yield Label("💬 Chat", id="chat-title")
        yield ScrollableContainer(id="chat-messages")

        with Horizontal(id="chat-input-container"):
            yield Input(placeholder="Type a message...", id="chat-input")
            yield Button("Send", id="send-button", variant="primary")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle send button press"""
        if event.button.id == "send-button":
            self._send_message()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle input submission"""
        if event.input.id == "chat-input":
            self._send_message()

    def _send_message(self) -> None:
        """Send a chat message"""
        try:
            chat_input = self.query_one("#chat-input", Input)
        except NoMatches:
            return
        message_text = chat_input.value.strip()

        if message_text:
            self.post_message(self.SendMessage(message_text))
            chat_input.value = ""

    def add_message(self, username: str, text: str, is_self: bool = False,
                   is_system: bool = False) -> None:
        """Add a message to the chat"""
        timestamp = datetime.now().strftime("%H:%M:%S")

        self.messages.append({
            'username': username,
            'text': text,
            'timestamp': timestamp,
            'is_self': is_self,
            'is_system': is_system
        })

        self._refresh_messages()

    def _refresh_messages(self) -> None:
        """Refresh message display"""
        try:
            container = self.query_one("#chat-messages", ScrollableContainer)
        except NoMatches:
            return
        container.remove_children()

        for msg in self.messages:
            if msg['is_system']:
                message_text = f"[{msg['timestamp']}] {msg['text']}"
                label = Label(message_text, classes="message-item message-system")
            else:
                message_text = f"[{msg['timestamp']}] {msg['username']}: {msg['text']}"
                css_class = "message-self" if msg['is_self'] else "message-other"
                label = Label(message_text, classes=f"message-item {css_class}")

            container.mount(label)

        # Scroll to bottom
        container.scroll_end(animate=False)


class ConflictResolutionPanel(Container):
    """Panel for resolving merge conflicts"""

    DEFAULT_CSS = """
    ConflictResolutionPanel {
        height: auto;
        border: solid $error;
        background: $surface;
        padding: 1;
    }

    #conflict-title {
        dock: top;
        height: 1;
        text-style: bold;
        color: $error;
    }

    #conflict-info {
        height: auto;
        margin: 1 0;
    }

    #conflict-actions {
        dock: bottom;
        height: 3;
    }

    .conflict-item {
        height: auto;
        padding: 1;
        margin: 0 0 1 0;
        background: $panel;
        border-left: thick $error;
    }
    """

    class ResolveConflict(Message):
        """Conflict resolution event"""
        def __init__(self, strategy: str):
            self.strategy = strategy
            super().__init__()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conflicts: List[Dict[str, Any]] = []
        self.visible = False

    def compose(self) -> ComposeResult:
        """Compose the conflict resolution panel"""
        yield Label("⚠️  Merge Conflicts Detected", id="conflict-title")
        yield ScrollableContainer(id="conflict-info")

        with Horizontal(id="conflict-actions"):
            yield Button("Smart Merge", id="btn-smart-merge", variant="primary")
            yield Button("Last Write Wins", id="btn-last-write", variant="default")
            yield Button("Manual Review", id="btn-manual", variant="warning")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle resolution button press"""
        strategy_map = {
            'btn-smart-merge': 'smart_merge',
            'btn-last-write': 'last_write_wins',
            'btn-manual': 'manual'
        }

        if event.button.id in strategy_map:
            self.post_message(self.ResolveConflict(strategy_map[event.button.id]))

    def show_conflicts(self, conflicts: List[Dict[str, Any]]) -> None:
        """Show conflicts in the panel"""
        self.conflicts = conflicts
        self.visible = True
        self.display = True
        self._refresh_conflicts()

    def hide_conflicts(self) -> None:
        """Hide the conflict panel"""
        self.visible = False
        self.display = False
        self.conflicts = []

    def _refresh_conflicts(self) -> None:
        """Refresh conflict display"""
        try:
            container = self.query_one("#conflict-info", ScrollableContainer)
        except NoMatches:
            return
        container.remove_children()

        for i, conflict in enumerate(self.conflicts, 1):
            conflict_text = f"Conflict {i}: {conflict.get('description', 'Unknown conflict')}"
            label = Label(conflict_text, classes="conflict-item")
            container.mount(label)


class RealTimeEditor(Container):
    """Real-time collaborative editor"""

    DEFAULT_CSS = """
    RealTimeEditor {
        height: 1fr;
        border: solid $primary;
        background: $surface;
        padding: 1;
    }

    #editor-title {
        dock: top;
        height: 1;
        text-style: bold;
        color: $accent;
    }

    #editor-content {
        height: 1fr;
        border: solid $secondary;
        background: $panel;
        margin: 1 0;
    }

    #editor-status {
        dock: bottom;
        height: 1;
        color: $text-muted;
    }
    """

    class ContentChanged(Message):
        """Content changed event"""
        def __init__(self, content: str, position: int):
            self.content = content
            self.position = position
            super().__init__()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.document_version = 0
        self.is_syncing = False

    def compose(self) -> ComposeResult:
        """Compose the real-time editor"""
        yield Label("📝 Collaborative Editor", id="editor-title")
        yield TextArea("", id="editor-content")
        yield Label("Ready | Version: 0 | Synced", id="editor-status")

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        """Handle content changes"""
        if not self.is_syncing and event.text_area.id == "editor-content":
            cursor_position = event.text_area.cursor_location[0]
            self.post_message(self.ContentChanged(event.text_area.text, cursor_position))

    def update_content(self, content: str, version: int) -> None:
        """Update editor content from remote"""
        self.is_syncing = True
        try:
            editor = self.query_one("#editor-content", TextArea)
            editor.text = content
        except NoMatches:
            pass
        self.document_version = version
        self._update_status()
        self.is_syncing = False

    def _update_status(self) -> None:
        """Update status bar"""
        try:
            status = self.query_one("#editor-status", Label)
        except NoMatches:
            return
        sync_status = "Syncing..." if self.is_syncing else "Synced"
        status.update(f"Ready | Version: {self.document_version} | {sync_status}")


class CollaborativeCodingPanel(Container):
    """Main collaborative coding panel"""

    DEFAULT_CSS = """
    CollaborativeCodingPanel {
        height: 100%;
        background: $surface;
    }

    #main-container {
        height: 100%;
    }

    #left-sidebar {
        width: 30;
        dock: left;
    }

    #center-content {
        width: 1fr;
    }

    #right-sidebar {
        width: 40;
        dock: right;
    }

    #session-controls {
        height: 5;
        dock: top;
        border: solid $primary;
        background: $panel;
        padding: 1;
    }

    #session-info {
        height: 1;
        color: $accent;
    }

    .control-button {
        margin: 0 1;
    }
    """

    BINDINGS = [
        Binding("ctrl+s", "sync", "Sync Changes"),
        Binding("ctrl+u", "toggle_users", "Toggle Users"),
        Binding("ctrl+m", "toggle_chat", "Toggle Chat"),
        Binding("escape", "leave_session", "Leave Session"),
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.session_id: Optional[str] = None
        self.user_id: Optional[str] = None
        self.username: str = "User"
        self.is_connected = False

    def compose(self) -> ComposeResult:
        """Compose the collaborative coding panel"""
        with Container(id="main-container"):
            # Session controls at top
            with Horizontal(id="session-controls"):
                yield Label("Not Connected", id="session-info")
                yield Button("Start Session", id="btn-start", variant="primary", classes="control-button")
                yield Button("Join Session", id="btn-join", variant="default", classes="control-button")
                yield Button("Leave", id="btn-leave", variant="error", classes="control-button", disabled=True)

            # Left sidebar - presence and cursors
            with Vertical(id="left-sidebar"):
                yield PresenceIndicator()
                yield CursorTracker()

            # Center - editor and conflicts
            with Vertical(id="center-content"):
                yield RealTimeEditor()
                conflict_panel = ConflictResolutionPanel()
                conflict_panel.display = False
                yield conflict_panel

            # Right sidebar - chat
            with Vertical(id="right-sidebar"):
                yield ChatInterface()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses"""
        if event.button.id == "btn-start":
            self._show_start_dialog()
        elif event.button.id == "btn-join":
            self._show_join_dialog()
        elif event.button.id == "btn-leave":
            self.action_leave_session()

    def _show_start_dialog(self) -> None:
        """Show dialog to start a session"""
        # This would typically show a modal dialog
        # For now, we'll just simulate starting a session
        import uuid
        self.session_id = str(uuid.uuid4())
        self.user_id = str(uuid.uuid4())
        self.username = "Host"
        self._connect_session()

    def _show_join_dialog(self) -> None:
        """Show dialog to join a session"""
        # This would typically show a modal dialog
        # For now, we'll just simulate joining
        import uuid
        self.user_id = str(uuid.uuid4())
        self.username = "Guest"
        self._connect_session()

    def _connect_session(self) -> None:
        """Connect to a session"""
        self.is_connected = True

        # Update UI
        session_info = self.query_one("#session-info", Label)
        session_info.update(f"Connected: {self.username} | Session: {self.session_id[:8] if self.session_id else 'N/A'}")

        self.query_one("#btn-start", Button).disabled = True
        self.query_one("#btn-join", Button).disabled = True
        self.query_one("#btn-leave", Button).disabled = False

        # Add self to presence
        presence = self.query_one(PresenceIndicator)
        presence.add_user(self.user_id, self.username, "#FF6B6B", "online")

        # Add system message to chat
        chat = self.query_one(ChatInterface)
        chat.add_message("System", f"{self.username} joined the session", is_system=True)

    def action_leave_session(self) -> None:
        """Leave the current session"""
        if not self.is_connected:
            return

        self.is_connected = False

        # Update UI
        session_info = self.query_one("#session-info", Label)
        session_info.update("Not Connected")

        self.query_one("#btn-start", Button).disabled = False
        self.query_one("#btn-join", Button).disabled = False
        self.query_one("#btn-leave", Button).disabled = True

        # Clear presence
        presence = self.query_one(PresenceIndicator)
        if self.user_id:
            presence.remove_user(self.user_id)

        # Add system message to chat
        chat = self.query_one(ChatInterface)
        chat.add_message("System", f"{self.username} left the session", is_system=True)

        self.session_id = None
        self.user_id = None

    def action_sync(self) -> None:
        """Sync changes"""
        if self.is_connected:
            self.query_one(RealTimeEditor)
            # Trigger sync logic here
            pass

    def action_toggle_users(self) -> None:
        """Toggle user list visibility"""
        sidebar = self.query_one("#left-sidebar")
        sidebar.display = not sidebar.display

    def action_toggle_chat(self) -> None:
        """Toggle chat visibility"""
        sidebar = self.query_one("#right-sidebar")
        sidebar.display = not sidebar.display
