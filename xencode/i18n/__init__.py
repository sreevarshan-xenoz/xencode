"""
Multi-language support module for Xencode.

This module provides internationalization (i18n) capabilities including:
- Translation engine with AI-powered translations
- Language detection and switching
- Technical term handling
- RTL language support
"""

from .translation_engine import TranslationEngine
from .language_manager import LanguageManager
from .translation_dict import TranslationDictionary
from .context_adapter import ContextAdapter

__all__ = [
    "TranslationEngine",
    "LanguageManager",
    "TranslationDictionary",
    "ContextAdapter",
]
