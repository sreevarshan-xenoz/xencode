"""
Enhanced Collaboration Panel for Xencode TUI

Advanced real-time collaboration features with shared editing, presence indicators, and more.
"""

from typing import Dict, List, Optional, Any
import asyncio
import json
from datetime import datetime
from rich.text import Text
from textual.widgets import Static, Button, Input, Label, ListView, ListItem, TextArea, TabbedContent, TabPane
from textual.containers import Container, Vertical, Horizontal, Grid
from textual.message import Message
from textual.binding import Binding
from textual.reactive import reactive
from textual.css.query import DOMQuery
import websockets


class EnhancedCollaborationPanel(Container):
    """Enhanced panel for managing collaboration sessions with real-time features"""

    DEFAULT_CSS = """
    EnhancedCollaborationPanel {
        height: 100%;
        background: $surface;
        border: solid $primary;
        padding: 1;
    }

    .section-title {
        text-style: bold;
        color: $accent;
        margin-bottom: 1;
    }

    .subsection-title {
        text-style: bold italic;
        color: $secondary;
        margin-top: 1;
        margin-bottom: 0.5;
    }

    #user-list {
        height: 1fr;
        border: solid $secondary;
        margin-bottom: 1;
    }

    #activity-feed {
        height: 10;
        border: solid $secondary;
        padding: 1;
        background: $boost;
        margin-bottom: 1;
    }

    #shared-editor {
        height: 20;
        border: solid $secondary;
        margin-bottom: 1;
    }

    .status-online {
        color: $success;
    }

    .status-offline {
        color: $error;
    }

    .status-away {
        color: $warning;
    }

    .user-presence {
        margin-right: 1;
    }

    .activity-item {
        margin-bottom: 0.5;
    }

    .tab-content {
        height: 1fr;
    }

    .collaboration-controls {
        margin-bottom: 1;
    }

    .notification-badge {
        background: $error;
        color: $text;
        padding: 0 1;
        border-radius: 10;
        text-style: bold;
    }
    """

    class HostSession(Message):
        """Request to host a session"""
        def __init__(self, session_name: str = "", description: str = ""):
            self.session_name = session_name
            self.description = description
            super().__init__()

    class JoinSession(Message):
        """Request to join a session"""
        def __init__(self, invite_code: str, username: str, session_password: str = ""):
            self.invite_code = invite_code
            self.username = username
            self.session_password = session_password
            super().__init__()

    class SendMessage(Message):
        """Send a message in the session"""
        def __init__(self, content: str, message_type: str = "chat"):
            self.content = content
            self.message_type = message_type
            super().__init__()

    class ShareCode(Message):
        """Share code snippet with collaborators"""
        def __init__(self, code: str, language: str = "python", description: str = ""):
            self.code = code
            self.language = language
            self.description = description
            super().__init__()

    class FileSync(Message):
        """Synchronize file changes with collaborators"""
        def __init__(self, file_path: str, content: str, action: str = "update"):
            self.file_path = file_path
            self.content = content
            self.action = action  # "update", "create", "delete", "rename"
            super().__init__()

    # Reactive properties
    is_connected = reactive(False)
    current_role = reactive(None)  # "host" or "guest"
    invite_code = reactive(None)
    session_name = reactive("")
    user_count = reactive(0)
    unread_messages = reactive(0)
    active_users = reactive(lambda: {})
    activity_log = reactive(lambda: [])

    def __init__(self):
        super().__init__()
        self.border_title = "🤝 Enhanced Collaboration"
        self.websocket = None
        self.session_id = None
        self.username = ""
        self.users: Dict[str, Dict[str, Any]] = {}
        self.shared_code_buffer = ""
        self.file_sync_buffer = {}

    def compose(self):
        """Compose the enhanced collaboration panel"""
        yield Label("🤝 Enhanced Collaboration", classes="section-title")

        # Status section
        self.status_label = Label("Status: Offline", classes="status-offline")
        yield self.status_label

        # Session info (when connected)
        with Vertical(id="session-info", classes="hidden"):
            self.session_info_label = Label("", classes="subsection-title")
            yield self.session_info_label
            
            # User presence indicators
            with Horizontal():
                self.user_count_label = Label("Users: 0", classes="user-count")
                yield self.user_count_label
                self.unread_badge = Label("", classes="notification-badge", visible=False)
                yield self.unread_badge

        # Connection controls
        with Vertical(id="connection-controls", classes="collaboration-controls"):
            with TabbedContent():
                with TabPane("Host Session", id="host-tab"):
                    self.session_name_input = Input(placeholder="Session Name (optional)", id="session-name")
                    yield self.session_name_input
                    self.session_desc_input = Input(placeholder="Description (optional)", id="session-desc")
                    yield self.session_desc_input
                    yield Button("Host New Session", id="btn-host", variant="primary")
                
                with TabPane("Join Session", id="join-tab"):
                    self.invite_code_input = Input(placeholder="Invite Code", id="invite-code")
                    yield self.invite_code_input
                    self.username_input = Input(placeholder="Your Username", id="username")
                    yield self.username_input
                    self.password_input = Input(placeholder="Password (if required)", id="password", password=True)
                    yield self.password_input
                    yield Button("Join Session", id="btn-join", variant="success")

        # Collaboration features (hidden when not connected)
        with Vertical(id="collaboration-features", classes="hidden"):
            # Tabs for different collaboration features
            with TabbedContent(id="collab-tabs"):
                # Chat tab
                with TabPane("💬 Chat", id="chat-tab"):
                    with Vertical(classes="tab-content"):
                        # Activity feed
                        self.activity_feed = Static(id="activity-feed", classes="activity-feed")
                        yield self.activity_feed
                        
                        # Message input
                        with Horizontal():
                            self.message_input = Input(placeholder="Type your message...", id="message-input")
                            yield self.message_input
                            yield Button("Send", id="send-message", variant="primary")

                # Shared Editor tab
                with TabPane("📝 Shared Editor", id="editor-tab"):
                    with Vertical(classes="tab-content"):
                        self.shared_editor = TextArea(
                            code="", 
                            language="python", 
                            id="shared-editor",
                            classes="shared-editor"
                        )
                        yield self.shared_editor
                        
                        with Horizontal():
                            yield Button("Sync Code", id="sync-code", variant="success")
                            yield Button("Reset Buffer", id="reset-buffer", variant="warning")

                # File Sync tab
                with TabPane("📁 File Sync", id="files-tab"):
                    with Vertical(classes="tab-content"):
                        self.file_sync_status = Static("No files synced", id="file-sync-status")
                        yield self.file_sync_status
                        
                        with Horizontal():
                            self.file_path_input = Input(placeholder="File path to sync", id="file-path")
                            yield self.file_path_input
                            yield Button("Sync File", id="sync-file", variant="success")
                            yield Button("List Files", id="list-files", variant="default")

                # Presence tab
                with TabPane("👥 Presence", id="presence-tab"):
                    with Vertical(classes="tab-content"):
                        self.presence_list = ListView(id="user-list")
                        yield self.presence_list
                        
                        self.presence_status = Static("Presence indicators for all collaborators", id="presence-status")
                        yield self.presence_status

        # Disconnect button
        with Horizontal():
            self.disconnect_btn = Button("Disconnect", id="btn-disconnect", variant="error", visible=False)
            yield self.disconnect_btn

    def on_mount(self) -> None:
        """Called when widget is mounted"""
        # Set up periodic updates for presence indicators
        self.set_interval(5, self.update_presence_indicators)

    def update_presence_indicators(self):
        """Update presence indicators for all users"""
        if self.is_connected and self.users:
            # In a real implementation, this would update based on user activity
            # For now, we'll just update the user count display
            self.user_count_label.update(f"Users: {len(self.users)}")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses"""
        if event.button.id == "btn-host":
            session_name = self.session_name_input.value
            description = self.session_desc_input.value
            self.post_message(self.HostSession(session_name, description))
        elif event.button.id == "btn-join":
            code = self.invite_code_input.value
            username = self.username_input.value
            password = self.password_input.value
            if code and username:
                self.post_message(self.JoinSession(code, username, password))
        elif event.button.id == "btn-disconnect":
            self.disconnect_from_session()
        elif event.button.id == "send-message":
            message = self.message_input.value
            if message.strip():
                self.post_message(self.SendMessage(message))
                self.message_input.value = ""
        elif event.button.id == "sync-code":
            code = self.shared_editor.text
            self.post_message(self.ShareCode(code))
        elif event.button.id == "sync-file":
            file_path = self.file_path_input.value
            if file_path:
                # In a real implementation, this would read the file and sync it
                self.post_message(self.FileSync(file_path, "file_content_placeholder"))
        elif event.button.id == "list-files":
            self.update_file_sync_status()
        elif event.button.id == "reset-buffer":
            self.shared_editor.text = ""

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle input submissions"""
        if event.input.id == "message-input":
            message = event.input.value
            if message.strip():
                self.post_message(self.SendMessage(message))
                event.input.value = ""

    def set_connected(self, role: str, code: str, session_name: str = ""):
        """Update UI for connected state"""
        self.is_connected = True
        self.current_role = role
        self.invite_code = code
        self.session_name = session_name

        # Update status
        session_display = f"{session_name} ({code})" if session_name else code
        self.status_label.update(f"Status: Connected as {role.title()} - {session_display}")
        self.status_label.classes = "status-online"

        # Show session info
        self.session_info_label.update(f"Session: {session_display}")
        self.query_one("#session-info").remove_class("hidden")

        # Hide connection controls
        self.query_one("#connection-controls").add_class("hidden")

        # Show collaboration features
        self.query_one("#collaboration-features").remove_class("hidden")

        # Show disconnect button
        self.disconnect_btn.visible = True

    def add_user(self, username: str, status: str = "online", avatar: str = "👤"):
        """Add user to presence list"""
        self.users[username] = {
            "status": status,
            "avatar": avatar,
            "last_seen": datetime.now(),
            "cursor_position": (0, 0)  # For shared editor
        }
        
        # Update user count
        self.user_count = len(self.users)
        
        # Add to list view
        user_item = ListItem(
            Label(f"{avatar} {username} - {status.title()}", classes=f"status-{status}")
        )
        user_item.data = {"username": username, "status": status}
        self.presence_list.append(user_item)

    def remove_user(self, username: str):
        """Remove user from list"""
        if username in self.users:
            del self.users[username]
            self.user_count = len(self.users)
            
            # Remove from list view (simplified)
            for item in self.presence_list.children:
                if hasattr(item, 'data') and item.data.get('username') == username:
                    item.remove()
                    break

    def add_activity(self, user: str, action: str, details: str = ""):
        """Add activity to the feed"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        activity_text = f"[{timestamp}] {user} {action}"
        if details:
            activity_text += f": {details}"
        
        # Add to activity log
        self.activity_log.append({
            "timestamp": timestamp,
            "user": user,
            "action": action,
            "details": details
        })
        
        # Update display (show last 10 activities)
        recent_activities = self.activity_log[-10:]
        activity_display = "\\n".join([
            f"[{item['timestamp']}] {item['user']} {item['action']}: {item['details']}" 
            for item in recent_activities
        ])
        
        self.activity_feed.update(activity_display)
        
        # Update unread counter if not focused
        self.unread_messages += 1
        if self.unread_messages > 0:
            self.unread_badge.update(str(self.unread_messages))
            self.unread_badge.visible = True

    def update_file_sync_status(self):
        """Update file sync status display"""
        if self.file_sync_buffer:
            files_list = "\\n".join([f"📄 {path}" for path in self.file_sync_buffer.keys()])
            self.file_sync_status.update(f"Synced Files:\\n{files_list}")
        else:
            self.file_sync_status.update("No files synced")

    def receive_message(self, user: str, content: str, message_type: str = "chat"):
        """Receive a message from another user"""
        if message_type == "chat":
            self.add_activity(user, "sent", f"Message: {content}")
        elif message_type == "code_share":
            self.add_activity(user, "shared code", content[:50] + "..." if len(content) > 50 else content)
            # Update shared editor if appropriate
            if self.current_role == "guest":  # Only guests update from host
                self.shared_editor.text = content
        elif message_type == "file_sync":
            self.add_activity(user, "synced file", content)

    def disconnect_from_session(self):
        """Disconnect from the current session"""
        # Close websocket connection if active
        if self.websocket:
            # In a real implementation, we would close the connection
            pass
        
        # Reset state
        self.is_connected = False
        self.current_role = None
        self.invite_code = None
        self.session_name = ""
        self.users.clear()
        self.activity_log.clear()
        self.unread_messages = 0
        
        # Update UI
        self.status_label.update("Status: Offline")
        self.status_label.classes = "status-offline"
        
        self.query_one("#session-info").add_class("hidden")
        self.query_one("#connection-controls").remove_class("hidden")
        self.query_one("#collaboration-features").add_class("hidden")
        self.disconnect_btn.visible = False
        self.unread_badge.visible = False
        
        # Clear lists
        self.presence_list.clear()
        self.activity_feed.update("")

    def get_session_info(self) -> Dict[str, Any]:
        """Get current session information"""
        return {
            "is_connected": self.is_connected,
            "role": self.current_role,
            "invite_code": self.invite_code,
            "session_name": self.session_name,
            "user_count": self.user_count,
            "users": list(self.users.keys()),
            "unread_messages": self.unread_messages
        }