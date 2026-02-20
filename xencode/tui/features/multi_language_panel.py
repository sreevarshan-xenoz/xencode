"""Multi-language Support TUI panel."""

from textual.containers import Container, Vertical, Horizontal, ScrollableContainer
from textual.widgets import Button, Label, Static, Select, ListView, ListItem, Input
from textual.reactive import reactive
from textual.screen import ModalScreen
from typing import Optional, List, Dict
import logging

from .base_feature_panel import BaseFeaturePanel

logger = logging.getLogger(__name__)


class TranslationDialog(ModalScreen):
    """Modal dialog for text translation."""
    
    DEFAULT_CSS = """
    TranslationDialog {
        align: center middle;
    }
    
    TranslationDialog > Container {
        width: 80;
        height: auto;
        border: thick $primary;
        background: $surface;
        padding: 1;
    }
    
    .dialog-title {
        text-align: center;
        text-style: bold;
        color: $accent;
        margin-bottom: 1;
    }
    
    .input-section {
        height: auto;
        margin-bottom: 1;
    }
    
    .output-section {
        height: auto;
        margin-top: 1;
        padding: 1;
        border: solid $success;
        background: $success-darken-2;
    }
    
    .dialog-buttons {
        height: auto;
        margin-top: 1;
    }
    """
    
    def __init__(self, current_language: str = "en"):
        super().__init__()
        self.current_language = current_language
        self.translation_result = None
    
    def compose(self):
        with Container():
            yield Label("🌐 Translate Text", classes="dialog-title")
            
            with Vertical(classes="input-section"):
                yield Label("Text to translate:")
                yield Input(placeholder="Enter text...", id="input-text")
                
                with Horizontal():
                    yield Label("From:")
                    yield Input(value="en", id="input-from", max_length=2)
                    yield Label("To:")
                    yield Input(value=self.current_language, id="input-to", max_length=2)
            
            with Vertical(classes="output-section", id="output-section"):
                yield Label("Translation will appear here", id="output-text")
            
            with Horizontal(classes="dialog-buttons"):
                yield Button("Translate", variant="primary", id="btn-translate")
                yield Button("Close", variant="default", id="btn-close")
    
    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "btn-translate":
            await self._do_translation()
        elif event.button.id == "btn-close":
            self.dismiss(self.translation_result)
    
    async def _do_translation(self) -> None:
        """Perform translation."""
        try:
            from xencode.i18n import TranslationEngine
            
            input_text = self.query_one("#input-text", Input).value
            from_lang = self.query_one("#input-from", Input).value
            to_lang = self.query_one("#input-to", Input).value
            
            if not input_text:
                self.query_one("#output-text", Label).update("Please enter text to translate")
                return
            
            engine = TranslationEngine()
            result = engine.translate(input_text, to_lang, from_lang)
            
            self.translation_result = result
            self.query_one("#output-text", Label).update(
                f"[green]{result.translated_text}[/green]\n\n"
                f"[dim]Confidence: {result.confidence:.2%}[/dim]"
            )
            
        except Exception as e:
            logger.error(f"Translation error: {e}")
            self.query_one("#output-text", Label).update(f"[red]Error: {e}[/red]")


class GlossaryDialog(ModalScreen):
    """Modal dialog for technical term glossary."""
    
    DEFAULT_CSS = """
    GlossaryDialog {
        align: center middle;
    }
    
    GlossaryDialog > Container {
        width: 80;
        height: 30;
        border: thick $primary;
        background: $surface;
        padding: 1;
    }
    
    .dialog-title {
        text-align: center;
        text-style: bold;
        color: $accent;
        margin-bottom: 1;
    }
    
    .glossary-search {
        height: auto;
        margin-bottom: 1;
    }
    
    .glossary-list {
        height: 1fr;
        border: solid $primary;
    }
    
    .dialog-buttons {
        height: auto;
        margin-top: 1;
    }
    """
    
    def compose(self):
        with Container():
            yield Label("📚 Technical Term Glossary", classes="dialog-title")
            
            with Horizontal(classes="glossary-search"):
                yield Input(placeholder="Search terms...", id="search-input")
                yield Button("Search", id="btn-search", variant="primary")
            
            with ScrollableContainer(classes="glossary-list"):
                yield ListView(id="terms-list")
            
            with Horizontal(classes="dialog-buttons"):
                yield Button("Close", variant="default", id="btn-close")
    
    def on_mount(self) -> None:
        """Load glossary on mount."""
        self._load_glossary()
    
    def _load_glossary(self, search_term: str = "") -> None:
        """Load technical terms into the list."""
        try:
            from xencode.i18n import TranslationEngine
            
            engine = TranslationEngine()
            terms = sorted(list(engine.technical_terms))
            
            if search_term:
                terms = [t for t in terms if search_term.lower() in t.lower()]
            
            terms_list = self.query_one("#terms-list", ListView)
            terms_list.clear()
            
            for term in terms[:100]:  # Limit to 100 terms
                terms_list.append(ListItem(Label(f"• {term}")))
            
        except Exception as e:
            logger.error(f"Error loading glossary: {e}")
    
    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "btn-search":
            search_input = self.query_one("#search-input", Input)
            self._load_glossary(search_input.value)
        elif event.button.id == "btn-close":
            self.dismiss()


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
    
    def __init__(self, code: str, name: str, native_name: str, is_active: bool = False, is_rtl: bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lang_code = code
        self.lang_name = name
        self.native_name = native_name
        self.is_active = is_active
        self.is_rtl = is_rtl
        if is_active:
            self.add_class("active")
    
    @property
    def code(self):
        """Get language code."""
        return self.lang_code
    
    def compose(self):
        status = "✓ " if self.is_active else ""
        rtl_marker = " [RTL]" if self.is_rtl else ""
        yield Label(f"{status}[bold]{self.native_name}[/bold] ({self.lang_name}){rtl_marker}")


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
        self.language_manager = None
        self.context_adapter = None
    
    def compose(self):
        """Compose the multi-language panel."""
        yield from super().compose()
    
    def on_mount(self) -> None:
        """Initialize panel on mount."""
        self.set_status("enabled")
        self._initialize_i18n()
        self._load_languages()
        self._build_content()
    
    def _initialize_i18n(self) -> None:
        """Initialize i18n components."""
        try:
            from xencode.i18n import LanguageManager, ContextAdapter
            
            self.language_manager = LanguageManager()
            self.context_adapter = ContextAdapter()
            self.current_language = self.language_manager.get_current_language()
            
        except Exception as e:
            logger.error(f"Failed to initialize i18n: {e}")
    
    def _load_languages(self) -> None:
        """Load supported languages."""
        try:
            if self.language_manager:
                lang_infos = self.language_manager.list_languages()
                self.languages = [
                    {
                        "code": lang.code,
                        "name": lang.name,
                        "native": lang.native_name,
                        "rtl": lang.rtl,
                        "enabled": lang.enabled
                    }
                    for lang in lang_infos
                ]
            else:
                # Fallback to hardcoded list
                self.languages = [
                    {"code": "en", "name": "English", "native": "English", "rtl": False, "enabled": True},
                    {"code": "es", "name": "Spanish", "native": "Español", "rtl": False, "enabled": True},
                    {"code": "fr", "name": "French", "native": "Français", "rtl": False, "enabled": True},
                    {"code": "de", "name": "German", "native": "Deutsch", "rtl": False, "enabled": True},
                    {"code": "zh", "name": "Chinese", "native": "中文", "rtl": False, "enabled": True},
                    {"code": "ja", "name": "Japanese", "native": "日本語", "rtl": False, "enabled": True},
                    {"code": "ko", "name": "Korean", "native": "한국어", "rtl": False, "enabled": True},
                    {"code": "ru", "name": "Russian", "native": "Русский", "rtl": False, "enabled": True},
                    {"code": "ar", "name": "Arabic", "native": "العربية", "rtl": True, "enabled": True},
                    {"code": "pt", "name": "Portuguese", "native": "Português", "rtl": False, "enabled": True},
                ]
        except Exception as e:
            logger.error(f"Failed to load languages: {e}")
    
    def _build_content(self) -> None:
        """Build the panel content."""
        if not self.content_container:
            return
        
        self.content_container.remove_children()
        
        with self.content_container:
            # Current language display
            current_lang = next(
                (lang for lang in self.languages if lang["code"] == self.current_language),
                self.languages[0] if self.languages else {"code": "en", "name": "English", "native": "English", "rtl": False}
            )
            
            rtl_indicator = " (RTL)" if current_lang.get("rtl", False) else ""
            yield Static(
                f"Current Language: {current_lang['native']} ({current_lang['name']}){rtl_indicator}",
                classes="current-language"
            )
            
            # Controls
            with Horizontal(classes="language-controls"):
                yield Button("Auto-detect", id="btn-detect", variant="primary")
                yield Button("Translate Text", id="btn-translate")
                yield Button("Glossary", id="btn-glossary")
            
            # Languages list
            with ScrollableContainer(classes="language-content"):
                yield Label("Available Languages (click to select)", classes="topic-title")
                
                languages_list = ListView(classes="languages-list")
                for lang in self.languages:
                    if lang.get("enabled", True):  # Only show enabled languages
                        languages_list.append(
                            LanguageCard(
                                lang["code"],
                                lang["name"],
                                lang["native"],
                                is_active=(lang["code"] == self.current_language),
                                is_rtl=lang.get("rtl", False)
                            )
                        )
                yield languages_list
                
                # RTL support info
                yield Label("\n[dim]RTL Support:[/dim]")
                yield Label("[dim]Languages marked with RTL support right-to-left text rendering[/dim]")
    
    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_id = event.button.id
        
        if button_id == "btn-detect":
            await self._auto_detect()
        elif button_id == "btn-translate":
            await self._translate_text()
        elif button_id == "btn-glossary":
            await self._show_glossary()
    
    async def on_list_view_selected(self, event: ListView.Selected) -> None:
        """Handle language selection."""
        if isinstance(event.item, LanguageCard):
            await self._change_language(event.item.code)
    
    async def _change_language(self, language_code: str) -> None:
        """Change the current language."""
        try:
            if self.language_manager:
                success = self.language_manager.set_language(language_code)
                if success:
                    self.current_language = language_code
                    self._build_content()
                    
                    # Show notification
                    lang_info = self.language_manager.get_language_info(language_code)
                    if lang_info:
                        self.notify(
                            f"Language changed to {lang_info.native_name}",
                            severity="information",
                            timeout=3
                        )
                else:
                    self.notify("Failed to change language", severity="error")
            else:
                self.current_language = language_code
                self._build_content()
                
        except Exception as e:
            logger.error(f"Error changing language: {e}")
            self.notify(f"Error: {e}", severity="error")
    
    async def _auto_detect(self) -> None:
        """Auto-detect user language."""
        try:
            if self.language_manager:
                # Detect from system
                detected_lang = self.language_manager.get_current_language()
                lang_info = self.language_manager.get_language_info(detected_lang)
                
                if lang_info:
                    self.notify(
                        f"Detected language: {lang_info.native_name} ({detected_lang})",
                        severity="information",
                        timeout=5
                    )
                else:
                    self.notify("Could not detect language", severity="warning")
            else:
                self.notify("Language detection not available", severity="warning")
                
        except Exception as e:
            logger.error(f"Error detecting language: {e}")
            self.notify(f"Error: {e}", severity="error")
    
    async def _translate_text(self) -> None:
        """Open translation dialog."""
        try:
            dialog = TranslationDialog(self.current_language)
            result = await self.app.push_screen(dialog, wait_for_dismiss=True)
            
            if result:
                self.notify(
                    f"Translation complete (confidence: {result.confidence:.2%})",
                    severity="information",
                    timeout=3
                )
                
        except Exception as e:
            logger.error(f"Error opening translation dialog: {e}")
            self.notify(f"Error: {e}", severity="error")
    
    async def _show_glossary(self) -> None:
        """Show technical term glossary."""
        try:
            dialog = GlossaryDialog()
            await self.app.push_screen(dialog, wait_for_dismiss=True)
            
        except Exception as e:
            logger.error(f"Error opening glossary: {e}")
            self.notify(f"Error: {e}", severity="error")
