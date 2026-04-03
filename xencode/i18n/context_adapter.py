"""
Context adapter for UI translation and RTL support.

Provides context-aware translation with technical accuracy and RTL handling.
"""

import logging
from typing import Dict, Optional, Any, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TranslationContext:
    """Context information for translation."""
    
    feature: Optional[str] = None
    component: Optional[str] = None
    user_role: Optional[str] = None
    technical_level: str = 'intermediate'  # beginner, intermediate, advanced
    preserve_formatting: bool = True


class ContextAdapter:
    """
    Adapts translations based on context with RTL support.
    
    Provides context-aware translation for UI elements while maintaining
    technical accuracy and supporting right-to-left languages.
    """
    
    def __init__(
        self,
        translation_engine=None,
        language_manager=None,
        translation_dict=None
    ):
        """
        Initialize the context adapter.
        
        Args:
            translation_engine: TranslationEngine instance
            language_manager: LanguageManager instance
            translation_dict: TranslationDictionary instance
        """
        from .translation_engine import TranslationEngine
        from .language_manager import LanguageManager
        from .translation_dict import TranslationDictionary
        
        self.translation_engine = translation_engine or TranslationEngine()
        self.language_manager = language_manager or LanguageManager()
        self.translation_dict = translation_dict or TranslationDictionary()
        
        # RTL configuration
        self.rtl_languages = {'ar', 'he', 'fa', 'ur'}
        self.rtl_markers = {
            'start': '\u202B',  # Right-to-left embedding
            'end': '\u202C',    # Pop directional formatting
            'ltr': '\u202A',    # Left-to-right embedding
        }
    
    def translate_ui(
        self,
        text: str,
        context: Optional[TranslationContext] = None,
        language: Optional[str] = None
    ) -> str:
        """
        Translate UI text with context awareness.
        
        Args:
            text: Text to translate
            context: Translation context
            language: Target language (uses current if None)
            
        Returns:
            Translated text
        """
        target_lang = language or self.language_manager.get_current_language()
        
        # Check dictionary first for common UI elements
        if self.translation_dict.has(text, target_lang):
            translated = self.translation_dict.get(text, target_lang)
        else:
            # Use translation engine for dynamic content
            result = self.translation_engine.translate(
                text,
                target_lang,
                preserve_code=context.preserve_formatting if context else True
            )
            translated = result.translated_text
        
        # Apply RTL formatting if needed
        if self.language_manager.is_rtl(target_lang):
            translated = self._apply_rtl_formatting(translated)
        
        return translated
    
    def translate_response(
        self,
        response: str,
        context: Optional[TranslationContext] = None,
        language: Optional[str] = None
    ) -> str:
        """
        Translate AI response with technical accuracy.
        
        Args:
            response: Response text to translate
            context: Translation context
            language: Target language (uses current if None)
            
        Returns:
            Translated response
        """
        target_lang = language or self.language_manager.get_current_language()
        
        # Skip translation if already in target language
        detected_lang = self.translation_engine.detect_language(response)
        if detected_lang == target_lang:
            return response
        
        # Translate with code preservation
        result = self.translation_engine.translate(
            response,
            target_lang,
            preserve_code=True
        )
        
        translated = result.translated_text
        
        # Apply RTL formatting if needed
        if self.language_manager.is_rtl(target_lang):
            translated = self._apply_rtl_formatting(translated, preserve_code=True)
        
        return translated
    
    def translate_error(
        self,
        error_message: str,
        language: Optional[str] = None
    ) -> str:
        """
        Translate error message.
        
        Args:
            error_message: Error message to translate
            language: Target language (uses current if None)
            
        Returns:
            Translated error message
        """
        target_lang = language or self.language_manager.get_current_language()
        
        # Check dictionary for common errors
        error_key = error_message.lower().replace(' ', '_')
        if self.translation_dict.has(error_key, target_lang):
            return self.translation_dict.get(error_key, target_lang)
        
        # Translate with technical term preservation
        result = self.translation_engine.translate(
            error_message,
            target_lang,
            preserve_code=True
        )
        
        return result.translated_text
    
    def _apply_rtl_formatting(
        self,
        text: str,
        preserve_code: bool = True
    ) -> str:
        """
        Apply RTL formatting to text.
        
        Args:
            text: Text to format
            preserve_code: Whether to preserve LTR for code snippets
            
        Returns:
            RTL-formatted text
        """
        if not preserve_code:
            # Simple RTL wrapping
            return f"{self.rtl_markers['start']}{text}{self.rtl_markers['end']}"
        
        # Complex RTL with code preservation
        import re
        
        # Find code blocks and inline code
        code_pattern = r'(```[\s\S]*?```|`[^`]+`)'
        parts = re.split(code_pattern, text)
        
        formatted_parts = []
        for i, part in enumerate(parts):
            if i % 2 == 0:
                # Regular text - apply RTL
                if part.strip():
                    formatted_parts.append(
                        f"{self.rtl_markers['start']}{part}{self.rtl_markers['end']}"
                    )
            else:
                # Code - keep LTR
                formatted_parts.append(
                    f"{self.rtl_markers['ltr']}{part}{self.rtl_markers['end']}"
                )
        
        return ''.join(formatted_parts)
    
    def get_ui_direction(self, language: Optional[str] = None) -> str:
        """
        Get UI direction for a language.
        
        Args:
            language: Language code (uses current if None)
            
        Returns:
            'rtl' or 'ltr'
        """
        lang = language or self.language_manager.get_current_language()
        return 'rtl' if self.language_manager.is_rtl(lang) else 'ltr'
    
    def format_message(
        self,
        key: str,
        language: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Format a message with parameters.
        
        Args:
            key: Message key
            language: Target language (uses current if None)
            **kwargs: Format parameters
            
        Returns:
            Formatted message
        """
        target_lang = language or self.language_manager.get_current_language()
        return self.translation_dict.get(key, target_lang, **kwargs)
    
    def translate_list(
        self,
        items: List[str],
        context: Optional[TranslationContext] = None,
        language: Optional[str] = None
    ) -> List[str]:
        """
        Translate a list of items.
        
        Args:
            items: List of items to translate
            context: Translation context
            language: Target language (uses current if None)
            
        Returns:
            List of translated items
        """
        return [
            self.translate_ui(item, context, language)
            for item in items
        ]
    
    def translate_dict(
        self,
        data: Dict[str, Any],
        keys_to_translate: List[str],
        context: Optional[TranslationContext] = None,
        language: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Translate specific keys in a dictionary.
        
        Args:
            data: Dictionary to translate
            keys_to_translate: List of keys to translate
            context: Translation context
            language: Target language (uses current if None)
            
        Returns:
            Dictionary with translated values
        """
        result = data.copy()
        
        for key in keys_to_translate:
            if key in result and isinstance(result[key], str):
                result[key] = self.translate_ui(
                    result[key],
                    context,
                    language
                )
        
        return result
    
    def get_technical_term_translation(
        self,
        term: str,
        language: Optional[str] = None
    ) -> str:
        """
        Get translation for a technical term.
        
        Technical terms are often kept in English or have specific
        translations in the target language.
        
        Args:
            term: Technical term
            language: Target language (uses current if None)
            
        Returns:
            Translated term (may be same as input)
        """
        target_lang = language or self.language_manager.get_current_language()
        
        # Check if term should be preserved
        if term.lower() in self.translation_engine.technical_terms:
            return term
        
        # Check dictionary for technical term translation
        tech_key = f"tech_{term.lower()}"
        if self.translation_dict.has(tech_key, target_lang):
            return self.translation_dict.get(tech_key, target_lang)
        
        # Return original term
        return term
    
    def set_language(self, language_code: str) -> bool:
        """
        Set the current language.
        
        Args:
            language_code: Language code to set
            
        Returns:
            True if successful, False otherwise
        """
        return self.language_manager.set_language(language_code)
    
    def get_current_language(self) -> str:
        """
        Get the current language code.
        
        Returns:
            Current language code
        """
        return self.language_manager.get_current_language()
    
    def get_supported_languages(self) -> List[Dict[str, Any]]:
        """
        Get list of supported languages.
        
        Returns:
            List of language information dictionaries
        """
        languages = self.language_manager.list_languages()
        return [
            {
                'code': lang.code,
                'name': lang.name,
                'native_name': lang.native_name,
                'rtl': lang.rtl,
                'enabled': lang.enabled,
            }
            for lang in languages
        ]
    
    def create_context(
        self,
        feature: Optional[str] = None,
        component: Optional[str] = None,
        technical_level: str = 'intermediate'
    ) -> TranslationContext:
        """
        Create a translation context.
        
        Args:
            feature: Feature name
            component: Component name
            technical_level: User's technical level
            
        Returns:
            TranslationContext object
        """
        return TranslationContext(
            feature=feature,
            component=component,
            technical_level=technical_level
        )
    
    def wrap_rtl_text(self, text: str) -> str:
        """
        Wrap text with RTL markers if current language is RTL.
        
        Args:
            text: Text to wrap
            
        Returns:
            Wrapped text
        """
        if self.language_manager.is_rtl():
            return f"{self.rtl_markers['start']}{text}{self.rtl_markers['end']}"
        return text
    
    def unwrap_rtl_text(self, text: str) -> str:
        """
        Remove RTL markers from text.
        
        Args:
            text: Text to unwrap
            
        Returns:
            Text without RTL markers
        """
        for marker in self.rtl_markers.values():
            text = text.replace(marker, '')
        return text
