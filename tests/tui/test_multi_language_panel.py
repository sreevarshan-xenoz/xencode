"""
Tests for multi-language TUI panel.
"""

import pytest
from textual.widgets import Button, ListView
from xencode.tui.features.multi_language_panel import (
    MultiLanguagePanel,
    LanguageCard,
    TranslationDialog,
    GlossaryDialog
)


@pytest.mark.asyncio
async def test_multi_language_panel_creation():
    """Test creating a multi-language panel."""
    panel = MultiLanguagePanel()
    assert panel.feature_name == "multi_language"
    assert "Multi-language" in panel.title


@pytest.mark.asyncio
async def test_language_card_creation():
    """Test creating a language card."""
    card = LanguageCard("es", "Spanish", "Español", is_active=True, is_rtl=False)
    assert card.code == "es"
    assert card.lang_name == "Spanish"
    assert card.native_name == "Español"
    assert card.is_active is True
    assert card.is_rtl is False


@pytest.mark.asyncio
async def test_language_card_rtl():
    """Test creating an RTL language card."""
    card = LanguageCard("ar", "Arabic", "العربية", is_active=False, is_rtl=True)
    assert card.code == "ar"
    assert card.is_rtl is True


@pytest.mark.asyncio
async def test_translation_dialog_creation():
    """Test creating a translation dialog."""
    dialog = TranslationDialog(current_language="es")
    assert dialog.current_language == "es"
    assert dialog.translation_result is None


@pytest.mark.asyncio
async def test_glossary_dialog_creation():
    """Test creating a glossary dialog."""
    dialog = GlossaryDialog()
    assert dialog is not None


def test_panel_has_required_methods():
    """Test that panel has required methods."""
    panel = MultiLanguagePanel()
    assert hasattr(panel, '_change_language')
    assert hasattr(panel, '_auto_detect')
    assert hasattr(panel, '_translate_text')
    assert hasattr(panel, '_show_glossary')
    assert hasattr(panel, '_load_languages')
    assert hasattr(panel, '_build_content')


def test_panel_initializes_with_languages():
    """Test that panel initializes with language list."""
    panel = MultiLanguagePanel()
    panel.on_mount()
    assert len(panel.languages) > 0
    assert any(lang['code'] == 'en' for lang in panel.languages)
    assert any(lang['code'] == 'es' for lang in panel.languages)


def test_panel_current_language_default():
    """Test that panel has default current language."""
    panel = MultiLanguagePanel()
    panel.on_mount()
    assert panel.current_language is not None
    # Should be 'en' or system detected language
    assert isinstance(panel.current_language, str)
    assert len(panel.current_language) == 2


def test_language_card_active_class():
    """Test that active language card has active class."""
    card = LanguageCard("en", "English", "English", is_active=True)
    assert "active" in card.classes


def test_language_card_inactive_no_class():
    """Test that inactive language card doesn't have active class."""
    card = LanguageCard("es", "Spanish", "Español", is_active=False)
    assert "active" not in card.classes
