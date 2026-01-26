from textual.widgets import Static, Button, Input, Label, ListView, ListItem
from textual.containers import Container, Vertical, Horizontal
from textual.message import Message
from textual.binding import Binding

class CollaborationPanel(Container):
    """Panel for managing collaboration sessions"""
    
    DEFAULT_CSS = """
    CollaborationPanel {
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
    
    #user-list {
        height: 1fr;
        border: solid $secondary;
        margin-bottom: 1;
    }
    
    .status-online {
        color: $success;
    }
    
    .status-offline {
        color: $error;
    }
    """
    
    class HostSession(Message):
        """Request to host a session"""
        pass
        
    class JoinSession(Message):
        """Request to join a session"""
        def __init__(self, invite_code: str, username: str):
            self.invite_code = invite_code
            self.username = username
            super().__init__()
            
    def __init__(self):
        super().__init__()
        self.is_connected = False
        self.current_role = None # "host" or "guest"
        self.invite_code = None
        
    def compose(self):
        yield Label("🤝 Collaboration", classes="section-title")
        
        # Status section
        self.status_label = Label("Status: Offline", classes="status-offline")
        yield self.status_label
        
        # Controls (Host/Join)
        with Vertical(id="controls"):
            yield Button("Host Session", id="btn-host", variant="primary")
            yield Label("Or Join Existing:", classes="dim")
            self.input_code = Input(placeholder="Invite Code")
            self.input_username = Input(placeholder="Your Username")
            yield Button("Join Session", id="btn-join", variant="default")
            
        # Active Session Info (Hidden by default)
        with Vertical(id="session-info", classes="hidden"):
            yield Label("Invite Code:", classes="dim")
            self.code_display = Label("----", id="invite-code")
            yield Label("Connected Users:", classes="section-title")
            self.user_list = ListView(id="user-list")
            yield self.user_list
            yield Button("Disconnect", id="btn-disconnect", variant="error")
            
    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-host":
            self.post_message(self.HostSession())
        elif event.button.id == "btn-join":
            code = self.input_code.value
            username = self.input_username.value
            if code and username:
                self.post_message(self.JoinSession(code, username))
        elif event.button.id == "btn-disconnect":
            # Implement disconnect logic
            self.is_connected = False
            self.current_role = None
            self.invite_code = None
            self.status_label.update("Status: Disconnected")
            self.status_label.classes = "status-offline"
            self.query_one("#controls").display = True
            self.query_one("#btn-start").disabled = False
            self.query_one("#btn-join").disabled = False
            self.input_username.disabled = False
            self.input_code.disabled = False
            
    def set_connected(self, role: str, code: str):
        """Update UI for connected state"""
        self.is_connected = True
        self.current_role = role
        self.invite_code = code
        
        self.status_label.update(f"Status: Online ({role.title()})")
        self.status_label.classes = "status-online"
        
        self.query_one("#controls").display = False
        self.query_one("#session-info").display = True
        self.query_one("#session-info").remove_class("hidden")
        
        self.code_display.update(code)
        
    def add_user(self, username: str):
        """Add user to list"""
        self.user_list.append(ListItem(Label(f"👤 {username}")))
        
    def remove_user(self, username: str):
        """Remove user from list (simplified)"""
        # In a real app, we'd need ID-based removal
        pass
