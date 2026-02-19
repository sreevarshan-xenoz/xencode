"""Multi-language Support TUI panel."""

from textual.containers import Container, Vertical, Horizontal, ScrollableContainer
from textual.widgets import Button, Label, Static, Select, ListView, ListItem
from textual.reactive import reactive
from typing import Optional, List, Dict

from .base_feature_panel import BaseFeaturePanel


class LanguageCard(ListItem):
    """Card for a language option."""
    
    DEFAULT_CSS = """
    LanguageCard {
        height: auto;
        padding: 1;
        margin: 0 0 1 0;
        border: solid $accent;
        background: $panel;
    }
    
    LanguageCard:hover {
        background: $primary;
    }
    
    LanguageCard.active {
        border: solid $success;
        background: $success-darken-1;
    }
    """
    
    def __init__(self, code: str, name: str, native_name: str, is_active: bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.code = code
        self.name = name
        self.native_name = native_name
        self.is_active = is_active
        if is_active:
            self.add_class("active")
    
    def compose(self):
        status = "✓ " if self.is_active else ""
        yield Label(f"{status}[bold]{self.native_name}[/bold] ({self.name})")


class MultiLanguagePanel(BaseFeaturePanel):
    """Panel for multi-language support and translation."""
    
    DEFAULT_CSS = """
    MultiLanguagePanel {
        height: 100%;
    }
    
    .language-controls {
        height: auto;
        padding: 1;
        background: $panel;
    }
    
    .language-content {
        height: 1fr;
        padding: 1;
    }
    
    .languages-list {
        height: 1fr;
        border: solid $primary;
    }
    
    .current-language {
        height: auto;
        padding: 1;
        margin-bottom: 1;
        border: solid $success;
        background: $success-darken-2;
        text-align: center;
    }
    """
    
    current_language = reactive("en")
    
    def __init__(self, *args, **kwargs):
        super().__init__(
            feature_name="multi_language",
            title="🌍 Multi-language Support",
            *args,
            **kwargs
        )
        self.languages: List[Dict[str, str]] = []
    
    def compose(self):
        """Compose the multi-language panel."""
        yield from super().compose()
    
    def on_mount(self) -> None:
        """Initialize panel on mount."""
        self.set_status("enabled")
        self._load_languages()
        self._build_content()
    
    def _load_languages(self) -> None:
        """Load supported languages."""
        # TODO: Load from actual i18n feature
        self.languages = [
            {"code": "en", "name": "English", "native": "English"},
            {"code": "es", "name": "Spanish", "native": "Español"},
            {"code": "fr", "name": "French", "native": "Français"},
            {"code": "de", "name": "German", "native": "Deutsch"},
            {"code": "zh", "name": "Chinese", "native": "中文"},
            {"code": "ja", "name": "Japanese", "native": "日本語"},
            {"code": "ko", "name": "Korean", "native": "한국어"},
            {"code": "ru", "name": "Russian", "native": "Русский"},
            {"code": "ar", "name": "Arabic", "native": "العربية"},
            {"code": "pt", "name": "Portuguese", "native": "Português"},
        ]
    
    def _build_content(self) -> None:
        """Build the panel content."""
        if not self.content_container:
            return
        
        self.content_container.remove_children()
        
        with self.content_container:
            # Current language display
            current_lang = next(
                (lang for lang in self.languages if lang["code"] == self.current_language),
                self.languages[0]
            )
            yield Static(
                f"Current Language: {current_lang['native']} ({current_lang['name']})",
                classes="current-language"
            )
            
            # Controls
            with Horizontal(classes="language-controls"):
                yield Button("Auto-detect", id="btn-detect", variant="primary")
                yield Button("Translate Text", id="btn-translate")
                yield Button("Settings", id="btn-settings")
            
            # Languages list
            with ScrollableContainer(classes="language-content"):
                yield Label("Available Languages", classes="topic-title")
                
                languages_list = ListView(classes="languages-list")
                for lang in self.languages:
                    languages_list.append(
                        LanguageCard(
                            lang["code"],
                            lang["name"],
                            lang["native"],
                            is_active=(lang["code"] == self.current_language)
                        )
                    )
                yield languages_list
    
    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_id = event.button.id
        
        if button_id == "btn-detect":
            await self._auto_detect()
        elif button_id == "btn-translate":
            await self._translate_text()
        elif button_id == "btn-settings":
            await self._show_settings()
    
    async def on_list_view_selected(self, event: ListView.Selected) -> None:
        """Handle language selection."""
        if isinstance(event.item, LanguageCard):
            await self._change_language(event.item.code)
    
    async def _change_language(self, language_code: str) -> None:
        """Change the current language."""
        self.current_language = language_code
        self._build_content()
        # TODO: Apply language change to entire TUI
    
    async def _auto_detect(self) -> None:
        """Auto-detect user language."""
        # TODO: Implement language detection
        pass
    
    async def _translate_text(self) -> None:
        """Open translation dialog."""
        # TODO: Implement translation dialog
        pass
    
    async def _show_settings(self) -> None:
        """Show language settings."""
        # TODO: Implement settings dialog
        pass
