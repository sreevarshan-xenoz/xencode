"""
Language manager for handling language detection, switching, and configuration.
"""

import logging
import os
import locale
from typing import Dict, List, Optional
from dataclasses import dataclass
from pathlib import Path
import json

logger = logging.getLogger(__name__)


@dataclass
class LanguageInfo:
    """Information about a supported language."""
    
    code: str
    name: str
    native_name: str
    rtl: bool = False
    enabled: bool = True


class LanguageManager:
    """
    Manages language settings, detection, and switching.
    
    Supports 10+ languages with runtime switching and fallback mechanisms.
    """
    
    # Supported languages
    SUPPORTED_LANGUAGES = {
        'en': LanguageInfo('en', 'English', 'English', rtl=False),
        'es': LanguageInfo('es', 'Spanish', 'Español', rtl=False),
        'fr': LanguageInfo('fr', 'French', 'Français', rtl=False),
        'de': LanguageInfo('de', 'German', 'Deutsch', rtl=False),
        'zh': LanguageInfo('zh', 'Chinese', '中文', rtl=False),
        'ja': LanguageInfo('ja', 'Japanese', '日本語', rtl=False),
        'ko': LanguageInfo('ko', 'Korean', '한국어', rtl=False),
        'ru': LanguageInfo('ru', 'Russian', 'Русский', rtl=False),
        'ar': LanguageInfo('ar', 'Arabic', 'العربية', rtl=True),
        'pt': LanguageInfo('pt', 'Portuguese', 'Português', rtl=False),
        'it': LanguageInfo('it', 'Italian', 'Italiano', rtl=False),
        'nl': LanguageInfo('nl', 'Dutch', 'Nederlands', rtl=False),
        'pl': LanguageInfo('pl', 'Polish', 'Polski', rtl=False),
        'tr': LanguageInfo('tr', 'Turkish', 'Türkçe', rtl=False),
        'he': LanguageInfo('he', 'Hebrew', 'עברית', rtl=True),
    }
    
    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initialize the language manager.
        
        Args:
            config_dir: Directory for storing language configuration
        """
        self.config_dir = config_dir or Path.home() / '.xencode' / 'i18n'
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        self.config_file = self.config_dir / 'language_config.json'
        self.current_language = 'en'
        self.fallback_language = 'en'
        
        self._load_config()
    
    def _load_config(self):
        """Load language configuration from file."""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    self.current_language = config.get('current_language', 'en')
                    self.fallback_language = config.get('fallback_language', 'en')
                    logger.info(f"Loaded language config: {self.current_language}")
            except Exception as e:
                logger.error(f"Failed to load language config: {e}")
                self._detect_system_language()
        else:
            self._detect_system_language()
            self._save_config()
    
    def _save_config(self):
        """Save language configuration to file."""
        try:
            config = {
                'current_language': self.current_language,
                'fallback_language': self.fallback_language,
            }
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            logger.debug("Saved language config")
        except Exception as e:
            logger.error(f"Failed to save language config: {e}")
    
    def _detect_system_language(self):
        """Detect language from system settings."""
        try:
            # Try to get system locale
            system_locale = locale.getdefaultlocale()[0]
            if system_locale:
                # Extract language code (e.g., 'en_US' -> 'en')
                lang_code = system_locale.split('_')[0].lower()
                if lang_code in self.SUPPORTED_LANGUAGES:
                    self.current_language = lang_code
                    logger.info(f"Detected system language: {lang_code}")
                    return
        except Exception as e:
            logger.warning(f"Failed to detect system language: {e}")
        
        # Try environment variables
        for env_var in ['LANG', 'LANGUAGE', 'LC_ALL']:
            lang = os.environ.get(env_var, '')
            if lang:
                lang_code = lang.split('_')[0].split('.')[0].lower()
                if lang_code in self.SUPPORTED_LANGUAGES:
                    self.current_language = lang_code
                    logger.info(f"Detected language from {env_var}: {lang_code}")
                    return
        
        # Default to English
        self.current_language = 'en'
        logger.info("Using default language: en")
    
    def get_current_language(self) -> str:
        """
        Get the current language code.
        
        Returns:
            Current language code (e.g., 'en', 'es')
        """
        return self.current_language
    
    def set_language(self, language_code: str) -> bool:
        """
        Set the current language.
        
        Args:
            language_code: Language code to set (e.g., 'es', 'fr')
            
        Returns:
            True if language was set successfully, False otherwise
        """
        if language_code not in self.SUPPORTED_LANGUAGES:
            logger.error(f"Unsupported language: {language_code}")
            return False
        
        if not self.SUPPORTED_LANGUAGES[language_code].enabled:
            logger.error(f"Language not enabled: {language_code}")
            return False
        
        self.current_language = language_code
        self._save_config()
        logger.info(f"Language set to: {language_code}")
        return True
    
    def get_fallback_language(self) -> str:
        """
        Get the fallback language code.
        
        Returns:
            Fallback language code
        """
        return self.fallback_language
    
    def set_fallback_language(self, language_code: str) -> bool:
        """
        Set the fallback language.
        
        Args:
            language_code: Language code to set as fallback
            
        Returns:
            True if fallback was set successfully, False otherwise
        """
        if language_code not in self.SUPPORTED_LANGUAGES:
            logger.error(f"Unsupported fallback language: {language_code}")
            return False
        
        self.fallback_language = language_code
        self._save_config()
        logger.info(f"Fallback language set to: {language_code}")
        return True
    
    def list_languages(self) -> List[LanguageInfo]:
        """
        Get list of all supported languages.
        
        Returns:
            List of LanguageInfo objects
        """
        return list(self.SUPPORTED_LANGUAGES.values())
    
    def get_language_info(self, language_code: str) -> Optional[LanguageInfo]:
        """
        Get information about a specific language.
        
        Args:
            language_code: Language code to query
            
        Returns:
            LanguageInfo object or None if not found
        """
        return self.SUPPORTED_LANGUAGES.get(language_code)
    
    def is_rtl(self, language_code: Optional[str] = None) -> bool:
        """
        Check if a language uses right-to-left text direction.
        
        Args:
            language_code: Language code to check (uses current if None)
            
        Returns:
            True if language is RTL, False otherwise
        """
        code = language_code or self.current_language
        lang_info = self.get_language_info(code)
        return lang_info.rtl if lang_info else False
    
    def is_supported(self, language_code: str) -> bool:
        """
        Check if a language is supported.
        
        Args:
            language_code: Language code to check
            
        Returns:
            True if language is supported, False otherwise
        """
        return language_code in self.SUPPORTED_LANGUAGES
    
    def get_language_name(
        self,
        language_code: str,
        native: bool = False
    ) -> Optional[str]:
        """
        Get the name of a language.
        
        Args:
            language_code: Language code to query
            native: If True, return native name; otherwise English name
            
        Returns:
            Language name or None if not found
        """
        lang_info = self.get_language_info(language_code)
        if not lang_info:
            return None
        return lang_info.native_name if native else lang_info.name
    
    def detect_language(self, text: str) -> str:
        """
        Detect language from text.
        
        This is a simple wrapper around the TranslationEngine's detection.
        For more accurate detection, use TranslationEngine directly.
        
        Args:
            text: Text to analyze
            
        Returns:
            Detected language code
        """
        # Import here to avoid circular dependency
        from .translation_engine import TranslationEngine
        
        engine = TranslationEngine()
        return engine.detect_language(text)
    
    def get_enabled_languages(self) -> List[LanguageInfo]:
        """
        Get list of enabled languages.
        
        Returns:
            List of enabled LanguageInfo objects
        """
        return [
            lang for lang in self.SUPPORTED_LANGUAGES.values()
            if lang.enabled
        ]
    
    def enable_language(self, language_code: str) -> bool:
        """
        Enable a language.
        
        Args:
            language_code: Language code to enable
            
        Returns:
            True if successful, False otherwise
        """
        if language_code not in self.SUPPORTED_LANGUAGES:
            return False
        
        self.SUPPORTED_LANGUAGES[language_code].enabled = True
        logger.info(f"Enabled language: {language_code}")
        return True
    
    def disable_language(self, language_code: str) -> bool:
        """
        Disable a language.
        
        Args:
            language_code: Language code to disable
            
        Returns:
            True if successful, False otherwise
        """
        if language_code not in self.SUPPORTED_LANGUAGES:
            return False
        
        if language_code == 'en':
            logger.error("Cannot disable English (default language)")
            return False
        
        self.SUPPORTED_LANGUAGES[language_code].enabled = False
        logger.info(f"Disabled language: {language_code}")
        return True
