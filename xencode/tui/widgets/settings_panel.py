#!/usr/bin/env python3
"""
Settings Panel Widget for Xencode TUI
"""

from typing import Dict, Any

from rich.text import Text
from textual.containers import Container, Vertical, Horizontal
from textual.message import Message
from textual.widgets import Button, Checkbox, Label, RadioSet, RadioButton


class SettingsPanel(Container):
    """Settings panel for TUI preferences and account actions."""

    DEFAULT_CSS = """
    SettingsPanel {
        height: 100%;
        background: $surface;
        border: solid $primary;
        padding: 1;
    }

    SettingsPanel .section-title {
        text-style: bold;
        color: $accent;
        margin-bottom: 1;
    }

    SettingsPanel Checkbox {
        margin: 0 0 1 0;
    }

    #settings-actions {
        margin-top: 1;
        height: auto;
    }

    #settings-auth {
        margin-top: 1;
        height: auto;
    }
    """

    class SaveRequested(Message):
        """Settings save requested."""

        def __init__(self, settings: Dict[str, Any]) -> None:
            super().__init__()
            self.settings = settings

    class LoginRequested(Message):
        """Login action requested."""

    class SignupRequested(Message):
        """Signup action requested."""

    class ThemeChanged(Message):
        """Theme selection changed (preview apply)."""

        def __init__(self, theme: str) -> None:
            super().__init__()
            self.theme = theme

    THEMES = [
        ("midnight", "Midnight"),
        ("ocean", "Ocean"),
        ("forest", "Forest"),
        ("sunset", "Sunset"),
        ("violet", "Violet"),
        ("slate", "Slate"),
        ("terminal", "Terminal"),
        ("desert", "Desert"),
        ("arctic", "Arctic"),
        ("rose", "Rose"),
    ]

    THEME_PREVIEWS = {
        "midnight": "Preview: Deep dark navy with soft gray text",
        "ocean": "Preview: Deep blue ocean with bright cool text",
        "forest": "Preview: Dark green surface with natural contrast",
        "sunset": "Preview: Warm dusk tones with creamy text",
        "violet": "Preview: Purple night palette with light text",
        "slate": "Preview: Neutral dark slate with crisp text",
        "terminal": "Preview: Retro green terminal on black",
        "desert": "Preview: Earthy brown base with sand text",
        "arctic": "Preview: Light icy background with deep blue text",
        "rose": "Preview: Dark rose surface with pink-white text",
    }

    THEME_SWATCHES = {
        "midnight": ("#0f172a", "#1e293b", "#cbd5e1"),
        "ocean": ("#0b1f3a", "#0f4c75", "#bbe1fa"),
        "forest": ("#0b1f15", "#1b4332", "#d8f3dc"),
        "sunset": ("#3b1f2b", "#b56576", "#ffe8d6"),
        "violet": ("#2b1b3f", "#5a189a", "#e0aaff"),
        "slate": ("#111827", "#374151", "#e5e7eb"),
        "terminal": ("#000000", "#14532d", "#22c55e"),
        "desert": ("#3f2d20", "#7f5539", "#f4e1c1"),
        "arctic": ("#eaf4ff", "#bfdbfe", "#1e3a8a"),
        "rose": ("#3f1d2e", "#9d174d", "#ffe4ec"),
    }

    @classmethod
    def _build_swatch_text(cls, theme_name: str) -> Text:
        colors = cls.THEME_SWATCHES.get(theme_name, cls.THEME_SWATCHES["midnight"])
        swatch = Text("Swatch: ", style="dim")
        for color in colors:
            swatch.append("  ", style=f"on {color}")
            swatch.append(" ")
        return swatch

    def compose(self):
        yield Label("⚙️ Settings", classes="section-title")
        yield Label("TUI preferences", classes="dim")

        self.chk_show_explorer = Checkbox("Show explorer on startup", value=True)
        self.chk_show_models = Checkbox("Show model selector on startup", value=False)
        self.chk_use_ensemble = Checkbox("Enable ensemble by default", value=False)
        self.chk_prompt_auth = Checkbox("Prompt Qwen login on first run", value=True)

        yield self.chk_show_explorer
        yield self.chk_show_models
        yield self.chk_use_ensemble
        yield self.chk_prompt_auth

        yield Label("Theme", classes="section-title")
        with RadioSet(id="theme-radio"):
            for idx, (_, label) in enumerate(self.THEMES):
                yield RadioButton(label, value=(idx == 0))

        self.theme_preview = Label(
            self.THEME_PREVIEWS["midnight"],
            id="theme-preview",
            classes="dim"
        )
        yield self.theme_preview

        self.theme_swatch = Label(id="theme-swatch")
        self.theme_swatch.update(self._build_swatch_text("midnight"))
        yield self.theme_swatch

        with Horizontal(id="settings-actions"):
            yield Button("Save", id="btn-save", variant="primary")

        with Vertical(id="settings-auth"):
            yield Label("Qwen account", classes="section-title")
            with Horizontal():
                yield Button("Login", id="btn-login", variant="success")
                yield Button("Sign Up", id="btn-signup", variant="default")

    def set_settings(self, settings: Dict[str, Any]) -> None:
        """Apply settings into form controls."""
        self.chk_show_explorer.value = bool(settings.get("show_explorer", True))
        self.chk_show_models.value = bool(settings.get("show_model_selector", False))
        self.chk_use_ensemble.value = bool(settings.get("use_ensemble_default", False))
        self.chk_prompt_auth.value = bool(settings.get("prompt_qwen_auth_on_first_run", True))

        selected_theme = settings.get("theme", "midnight")
        radio_set = self.query_one("#theme-radio", RadioSet)
        theme_index = 0
        for idx, (theme_id, _) in enumerate(self.THEMES):
            if theme_id == selected_theme:
                theme_index = idx
                break

        buttons = list(radio_set.query(RadioButton))
        if 0 <= theme_index < len(buttons):
            buttons[theme_index].value = True

        self.theme_preview.update(self.THEME_PREVIEWS.get(selected_theme, self.THEME_PREVIEWS["midnight"]))
        self.theme_swatch.update(self._build_swatch_text(selected_theme))

    def collect_settings(self) -> Dict[str, Any]:
        """Collect current settings from controls."""
        selected_theme = "midnight"
        radio_set = self.query_one("#theme-radio", RadioSet)
        if radio_set.pressed_button:
            buttons = list(radio_set.query(RadioButton))
            if radio_set.pressed_button in buttons:
                idx = buttons.index(radio_set.pressed_button)
                selected_theme = self.THEMES[idx][0]

        return {
            "show_explorer": self.chk_show_explorer.value,
            "show_model_selector": self.chk_show_models.value,
            "use_ensemble_default": self.chk_use_ensemble.value,
            "prompt_qwen_auth_on_first_run": self.chk_prompt_auth.value,
            "theme": selected_theme,
        }

    def on_radio_set_changed(self, event: RadioSet.Changed) -> None:
        """Update preview and request live theme apply when selection changes."""
        if not event.pressed:
            return

        radio_set = self.query_one("#theme-radio", RadioSet)
        buttons = list(radio_set.query(RadioButton))
        if event.pressed not in buttons:
            return

        theme_index = buttons.index(event.pressed)
        theme_name = self.THEMES[theme_index][0]
        self.theme_preview.update(self.THEME_PREVIEWS.get(theme_name, self.THEME_PREVIEWS["midnight"]))
        self.theme_swatch.update(self._build_swatch_text(theme_name))
        self.post_message(self.ThemeChanged(theme_name))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-save":
            self.post_message(self.SaveRequested(self.collect_settings()))
        elif event.button.id == "btn-login":
            self.post_message(self.LoginRequested())
        elif event.button.id == "btn-signup":
            self.post_message(self.SignupRequested())
