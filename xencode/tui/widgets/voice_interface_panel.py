#!/usr/bin/env python3
"""
Voice Interface Panel Widget for Xencode TUI

Interactive voice interface with status indicator, audio meter, command history,
and settings panel for voice configuration.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime

from textual.app import ComposeResult
from textual.containers import Container, Vertical, Horizontal, ScrollableContainer
from textual.widgets import Static, Label, Button, Input, ListView, ListItem, ProgressBar, Switch
from textual.reactive import reactive
from textual.message import Message
from textual.binding import Binding
from rich.text import Text
from rich.panel import Panel


class VoiceStatusIndicator(Container):
    """Visual indicator for voice interface status"""
    
    DEFAULT_CSS = """
    VoiceStatusIndicator {
        height: 5;
        border: solid $primary;
        background: $surface;
        padding: 1;
    }
    
    VoiceStatusIndicator.listening {
        border: solid $success;
        background: $success-darken-3;
    }
    
    VoiceStatusIndicator.processing {
        border: solid $warning;
        background: $warning-darken-3;
    }
    
    VoiceStatusIndicator.speaking {
        border: solid $accent;
        background: $accent-darken-3;
    }
    
    VoiceStatusIndicator.error {
        border: solid $error;
        background: $error-darken-3;
    }
    
    #status-icon {
        width: 10;
        height: 3;
        content-align: center middle;
    }
    
    #status-text {
        width: 1fr;
        height: 3;
        content-align: left middle;
    }
    """
    
    status = reactive("idle")
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.status_icons = {
            'idle': '🎤',
            'listening': '🔴',
            'processing': '⚙️',
            'speaking': '🔊',
            'error': '❌'
        }
        self.status_messages = {
            'idle': 'Ready to listen',
            'listening': 'Listening...',
            'processing': 'Processing voice input...',
            'speaking': 'Speaking response...',
            'error': 'Error occurred'
        }
    
    def compose(self) -> ComposeResult:
        """Compose the status indicator"""
        with Horizontal():
            yield Label(self.status_icons['idle'], id="status-icon")
            yield Label(self.status_messages['idle'], id="status-text")
    
    def watch_status(self, new_status: str) -> None:
        """Update status display"""
        # Update classes
        self.remove_class('listening', 'processing', 'speaking', 'error')
        if new_status != 'idle':
            self.add_class(new_status)
        
        # Update icon and text
        icon_label = self.query_one("#status-icon", Label)
        text_label = self.query_one("#status-text", Label)
        
        icon_label.update(self.status_icons.get(new_status, '🎤'))
        text_label.update(self.status_messages.get(new_status, 'Unknown status'))
    
    def set_status(self, status: str, message: str = None) -> None:
        """Set the current status"""
        self.status = status
        if message:
            text_label = self.query_one("#status-text", Label)
            text_label.update(message)


class AudioLevelMeter(Container):
    """Visual audio level meter"""
    
    DEFAULT_CSS = """
    AudioLevelMeter {
        height: 8;
        border: solid $accent;
        background: $surface;
        padding: 1;
    }
    
    #meter-title {
        dock: top;
        height: 1;
    }
    
    #meter-current {
        height: 1;
        margin: 1 0;
    }
    
    #meter-peak {
        height: 1;
        margin: 1 0;
    }
    
    #meter-average {
        height: 1;
        margin: 1 0;
    }
    
    .meter-bar {
        width: 100%;
    }
    """
    
    current_level = reactive(0.0)
    peak_level = reactive(0.0)
    average_level = reactive(0.0)
    is_speaking = reactive(False)
    
    def compose(self) -> ComposeResult:
        """Compose the audio meter"""
        yield Label("🎚️  Audio Level", id="meter-title")
        
        with Horizontal(id="meter-current"):
            yield Label("Current: ", classes="dim")
            yield ProgressBar(total=100, show_eta=False, classes="meter-bar")
        
        with Horizontal(id="meter-peak"):
            yield Label("Peak:    ", classes="dim")
            yield ProgressBar(total=100, show_eta=False, classes="meter-bar")
        
        with Horizontal(id="meter-average"):
            yield Label("Average: ", classes="dim")
            yield ProgressBar(total=100, show_eta=False, classes="meter-bar")
    
    def update_levels(self, current: float, peak: float, average: float, speaking: bool) -> None:
        """Update audio levels"""
        self.current_level = current
        self.peak_level = peak
        self.average_level = average
        self.is_speaking = speaking
        
        # Update progress bars
        bars = list(self.query(ProgressBar))
        if len(bars) >= 3:
            bars[0].update(progress=int(current * 100))
            bars[1].update(progress=int(peak * 100))
            bars[2].update(progress=int(average * 100))


class VoiceCommandItem(ListItem):
    """A single voice command in the history"""
    
    DEFAULT_CSS = """
    VoiceCommandItem {
        height: auto;
        padding: 1;
        margin: 0 1;
        background: $panel;
        border-left: thick $primary;
    }
    
    VoiceCommandItem:hover {
        background: $boost;
    }
    
    VoiceCommandItem.high-confidence {
        border-left: thick $success;
    }
    
    VoiceCommandItem.low-confidence {
        border-left: thick $warning;
    }
    """
    
    def __init__(self, command: Dict[str, Any], **kwargs):
        super().__init__(**kwargs)
        self.command = command
        confidence = command.get('confidence', 0)
        if confidence > 0.8:
            self.add_class('high-confidence')
        elif confidence < 0.5:
            self.add_class('low-confidence')
    
    def compose(self) -> ComposeResult:
        """Compose the command item"""
        text = self.command.get('text', '')
        command_type = self.command.get('command_type', 'unknown')
        confidence = self.command.get('confidence', 0)
        timestamp = self.command.get('timestamp', '')
        
        # Format timestamp
        try:
            dt = datetime.fromisoformat(timestamp)
            time_str = dt.strftime("%H:%M:%S")
        except Exception:
            time_str = timestamp
        
        # Confidence indicator
        conf_emoji = '🟢' if confidence > 0.8 else '🟡' if confidence > 0.5 else '🔴'
        
        yield Label(f"{conf_emoji} {text}", classes="bold")
        yield Label(f"   Type: {command_type} | Confidence: {confidence:.0%} | {time_str}", classes="dim")


class VoiceCommandHistory(Container):
    """Panel showing voice command history"""
    
    DEFAULT_CSS = """
    VoiceCommandHistory {
        height: 100%;
        border: solid $primary;
        background: $surface;
    }
    
    VoiceCommandHistory > #history-header {
        dock: top;
        height: 3;
        background: $primary;
        padding: 0 2;
    }
    
    VoiceCommandHistory > #history-list {
        height: 1fr;
    }
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.commands: List[Dict[str, Any]] = []
    
    def compose(self) -> ComposeResult:
        """Compose the history panel"""
        yield Label("📜 Command History", id="history-header")
        
        with ScrollableContainer(id="history-list"):
            yield Label("No voice commands yet", classes="dim")
    
    def update_history(self, commands: List[Dict[str, Any]]) -> None:
        """Update the command history"""
        self.commands = commands
        
        history_list = self.query_one("#history-list", ScrollableContainer)
        history_list.remove_children()
        
        if commands:
            for command in reversed(commands[-20:]):  # Show last 20
                history_list.mount(VoiceCommandItem(command))
        else:
            history_list.mount(Label("No voice commands yet", classes="dim"))
    
    def add_command(self, command: Dict[str, Any]) -> None:
        """Add a new command to history"""
        self.commands.append(command)
        self.update_history(self.commands)


class VoiceSettingsPanel(Container):
    """Panel for voice interface settings"""
    
    DEFAULT_CSS = """
    VoiceSettingsPanel {
        height: 100%;
        border: solid $accent;
        background: $surface;
        padding: 1;
    }
    
    #settings-header {
        height: 3;
        background: $accent;
        padding: 0 2;
    }
    
    #settings-content {
        height: 1fr;
        padding: 1;
    }
    
    .setting-row {
        height: auto;
        padding: 1;
        margin: 0 1;
    }
    
    .setting-label {
        width: 30;
    }
    
    .setting-value {
        width: 1fr;
    }
    """
    
    class SettingChanged(Message):
        """Message sent when a setting is changed"""
        def __init__(self, setting: str, value: Any):
            super().__init__()
            self.setting = setting
            self.value = value
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.settings = {
            'language': 'en',
            'voice': 'en-US',
            'rate': 150,
            'volume': 0.9,
            'noise_reduction': True,
            'silence_threshold': 0.5,
            'silence_duration': 1.0
        }
    
    def compose(self) -> ComposeResult:
        """Compose the settings panel"""
        yield Label("⚙️  Voice Settings", id="settings-header")
        
        with ScrollableContainer(id="settings-content"):
            # Language setting
            with Horizontal(classes="setting-row"):
                yield Label("Language:", classes="setting-label")
                yield Input(value=self.settings['language'], id="setting-language", classes="setting-value")
            
            # Voice setting
            with Horizontal(classes="setting-row"):
                yield Label("Voice:", classes="setting-label")
                yield Input(value=self.settings['voice'], id="setting-voice", classes="setting-value")
            
            # Rate setting
            with Horizontal(classes="setting-row"):
                yield Label("Speech Rate:", classes="setting-label")
                yield Input(value=str(self.settings['rate']), id="setting-rate", classes="setting-value")
            
            # Volume setting
            with Horizontal(classes="setting-row"):
                yield Label("Volume (0.0-1.0):", classes="setting-label")
                yield Input(value=str(self.settings['volume']), id="setting-volume", classes="setting-value")
            
            # Noise reduction toggle
            with Horizontal(classes="setting-row"):
                yield Label("Noise Reduction:", classes="setting-label")
                yield Switch(value=self.settings['noise_reduction'], id="setting-noise-reduction")
            
            # Silence threshold
            with Horizontal(classes="setting-row"):
                yield Label("Silence Threshold:", classes="setting-label")
                yield Input(value=str(self.settings['silence_threshold']), id="setting-silence-threshold", classes="setting-value")
            
            # Silence duration
            with Horizontal(classes="setting-row"):
                yield Label("Silence Duration (s):", classes="setting-label")
                yield Input(value=str(self.settings['silence_duration']), id="setting-silence-duration", classes="setting-value")
    
    def on_input_changed(self, event: Input.Changed) -> None:
        """Handle input changes"""
        input_id = event.input.id
        value = event.value
        
        if input_id == "setting-language":
            self.settings['language'] = value
            self.post_message(self.SettingChanged('language', value))
        elif input_id == "setting-voice":
            self.settings['voice'] = value
            self.post_message(self.SettingChanged('voice', value))
        elif input_id == "setting-rate":
            try:
                self.settings['rate'] = int(value)
                self.post_message(self.SettingChanged('rate', int(value)))
            except ValueError:
                pass
        elif input_id == "setting-volume":
            try:
                volume = float(value)
                self.settings['volume'] = max(0.0, min(1.0, volume))
                self.post_message(self.SettingChanged('volume', self.settings['volume']))
            except ValueError:
                pass
        elif input_id == "setting-silence-threshold":
            try:
                self.settings['silence_threshold'] = float(value)
                self.post_message(self.SettingChanged('silence_threshold', float(value)))
            except ValueError:
                pass
        elif input_id == "setting-silence-duration":
            try:
                self.settings['silence_duration'] = float(value)
                self.post_message(self.SettingChanged('silence_duration', float(value)))
            except ValueError:
                pass
    
    def on_switch_changed(self, event: Switch.Changed) -> None:
        """Handle switch changes"""
        if event.switch.id == "setting-noise-reduction":
            self.settings['noise_reduction'] = event.value
            self.post_message(self.SettingChanged('noise_reduction', event.value))
    
    def get_settings(self) -> Dict[str, Any]:
        """Get current settings"""
        return self.settings.copy()
    
    def update_settings(self, settings: Dict[str, Any]) -> None:
        """Update settings from external source"""
        self.settings.update(settings)
        
        # Update UI
        for key, value in settings.items():
            input_id = f"setting-{key.replace('_', '-')}"
            try:
                if key == 'noise_reduction':
                    switch = self.query_one(f"#{input_id}", Switch)
                    switch.value = value
                else:
                    input_widget = self.query_one(f"#{input_id}", Input)
                    input_widget.value = str(value)
            except Exception:
                    pass  # Silently ignore

class VoiceInterfacePanel(Container):
    """Main voice interface panel with all components"""
    
    DEFAULT_CSS = """
    VoiceInterfacePanel {
        height: 100%;
        border: solid $primary;
        background: $surface;
    }
    
    VoiceInterfacePanel > #vi-header {
        dock: top;
        height: 3;
        background: $primary;
        padding: 0 2;
    }
    
    VoiceInterfacePanel > #vi-controls {
        dock: top;
        height: 3;
        background: $panel;
        padding: 0 2;
    }
    
    VoiceInterfacePanel > #vi-status {
        dock: top;
        height: 5;
    }
    
    VoiceInterfacePanel > #vi-meter {
        dock: top;
        height: 8;
    }
    
    VoiceInterfacePanel > #vi-tabs {
        dock: top;
        height: 3;
        background: $panel;
    }
    
    VoiceInterfacePanel > #vi-content {
        height: 1fr;
    }
    
    .control-button {
        width: 1fr;
        margin: 0 1;
    }
    
    .tab-button {
        width: 1fr;
        margin: 0 1;
    }
    
    .tab-button.active {
        background: $primary;
    }
    """
    
    BINDINGS = [
        Binding("ctrl+v", "toggle_listening", "Toggle Voice"),
        Binding("ctrl+t", "test_voice", "Test Voice"),
        Binding("1", "show_history", "History"),
        Binding("2", "show_settings", "Settings"),
    ]
    
    class VoiceCommand(Message):
        """Message sent when a voice command is recognized"""
        def __init__(self, command: Dict[str, Any]):
            super().__init__()
            self.command = command
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.current_tab = "history"
        self.is_listening = False
    
    def compose(self) -> ComposeResult:
        """Compose the voice interface panel"""
        yield Label("🎤 Voice Interface", id="vi-header")
        
        with Horizontal(id="vi-controls"):
            yield Button("▶️  Start", classes="control-button", id="btn-start")
            yield Button("⏸️  Stop", classes="control-button", id="btn-stop")
            yield Button("🔊 Test", classes="control-button", id="btn-test")
        
        yield VoiceStatusIndicator(id="vi-status")
        yield AudioLevelMeter(id="vi-meter")
        
        with Horizontal(id="vi-tabs"):
            yield Button("History", classes="tab-button active", id="tab-history")
            yield Button("Settings", classes="tab-button", id="tab-settings")
        
        with Container(id="vi-content"):
            yield VoiceCommandHistory(id="history-panel")
            yield VoiceSettingsPanel(id="settings-panel", classes="hidden")
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses"""
        button_id = event.button.id
        
        if button_id == "btn-start":
            self.action_toggle_listening()
        elif button_id == "btn-stop":
            self._stop_listening()
        elif button_id == "btn-test":
            self.action_test_voice()
        elif button_id == "tab-history":
            self._show_tab("history")
        elif button_id == "tab-settings":
            self._show_tab("settings")
    
    def _show_tab(self, tab_name: str) -> None:
        """Show a specific tab"""
        self.current_tab = tab_name
        
        # Update button styles
        for button in self.query(".tab-button"):
            if button.id == f"tab-{tab_name}":
                button.add_class("active")
            else:
                button.remove_class("active")
        
        # Show/hide panels
        history_panel = self.query_one("#history-panel")
        settings_panel = self.query_one("#settings-panel")
        
        if tab_name == "history":
            history_panel.remove_class("hidden")
            settings_panel.add_class("hidden")
        else:
            history_panel.add_class("hidden")
            settings_panel.remove_class("hidden")
    
    def action_toggle_listening(self) -> None:
        """Toggle voice listening"""
        if self.is_listening:
            self._stop_listening()
        else:
            self._start_listening()
    
    def _start_listening(self) -> None:
        """Start listening for voice input"""
        self.is_listening = True
        status_indicator = self.query_one("#vi-status", VoiceStatusIndicator)
        status_indicator.set_status("listening", "Listening for voice input...")
        
        # Update button
        start_btn = self.query_one("#btn-start", Button)
        start_btn.label = "⏸️  Pause"
    
    def _stop_listening(self) -> None:
        """Stop listening for voice input"""
        self.is_listening = False
        status_indicator = self.query_one("#vi-status", VoiceStatusIndicator)
        status_indicator.set_status("idle", "Ready to listen")
        
        # Update button
        start_btn = self.query_one("#btn-start", Button)
        start_btn.label = "▶️  Start"
    
    def action_test_voice(self) -> None:
        """Test voice output"""
        status_indicator = self.query_one("#vi-status", VoiceStatusIndicator)
        status_indicator.set_status("speaking", "Testing voice output...")
        
        # Simulate speaking
        self.set_timer(2.0, lambda: status_indicator.set_status("idle", "Test complete"))
    
    def action_show_history(self) -> None:
        """Show history tab"""
        self._show_tab("history")
    
    def action_show_settings(self) -> None:
        """Show settings tab"""
        self._show_tab("settings")
    
    # Convenience methods for updating components
    
    def update_status(self, status: str, message: str = None) -> None:
        """Update voice status"""
        status_indicator = self.query_one("#vi-status", VoiceStatusIndicator)
        status_indicator.set_status(status, message)
    
    def update_audio_levels(self, current: float, peak: float, average: float, speaking: bool) -> None:
        """Update audio level meter"""
        meter = self.query_one("#vi-meter", AudioLevelMeter)
        meter.update_levels(current, peak, average, speaking)
    
    def add_command(self, command: Dict[str, Any]) -> None:
        """Add a command to history"""
        history_panel = self.query_one("#history-panel", VoiceCommandHistory)
        history_panel.add_command(command)
        
        # Post message for parent to handle
        self.post_message(self.VoiceCommand(command))
    
    def update_command_history(self, commands: List[Dict[str, Any]]) -> None:
        """Update command history"""
        history_panel = self.query_one("#history-panel", VoiceCommandHistory)
        history_panel.update_history(commands)
    
    def get_settings(self) -> Dict[str, Any]:
        """Get current settings"""
        settings_panel = self.query_one("#settings-panel", VoiceSettingsPanel)
        return settings_panel.get_settings()
    
    def update_settings(self, settings: Dict[str, Any]) -> None:
        """Update settings"""
        settings_panel = self.query_one("#settings-panel", VoiceSettingsPanel)
        settings_panel.update_settings(settings)
