"""
Multi-language support module for Xencode.

This module provides internationalization (i18n) capabilities including:
- Translation engine with AI-powered translations
- Language detection and switching
- Technical term handling
- RTL language support
"""

from .context_adapter import ContextAdapter
from .language_manager import LanguageManager
from .translation_dict import TranslationDictionary
from .translation_engine import TranslationEngine

__all__ = [
    "TranslationEngine",
    "LanguageManager",
    "TranslationDictionary",
    "ContextAdapter",
]
