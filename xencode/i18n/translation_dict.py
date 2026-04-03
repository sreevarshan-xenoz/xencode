"""
Translation dictionary for storing and managing translations.
"""

import logging
from typing import Dict, Optional, List
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class TranslationDictionary:
    """
    Manages translation dictionaries for UI elements and common phrases.
    
    Provides fast lookup of pre-translated strings with fallback support.
    """
    
    def __init__(self, translations_dir: Optional[Path] = None):
        """
        Initialize the translation dictionary.
        
        Args:
            translations_dir: Directory containing translation files
        """
        self.translations_dir = translations_dir or Path(__file__).parent / 'translations'
        self.translations_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache of loaded translations: {language_code: {key: translation}}
        self.translations: Dict[str, Dict[str, str]] = {}
        
        # Load default English translations
        self._load_default_translations()
    
    def _load_default_translations(self):
        """Load default English translations."""
        self.translations['en'] = {
            # Common UI elements
            'welcome': 'Welcome to Xencode',
            'loading': 'Loading...',
            'error': 'Error',
            'success': 'Success',
            'cancel': 'Cancel',
            'confirm': 'Confirm',
            'save': 'Save',
            'delete': 'Delete',
            'edit': 'Edit',
            'close': 'Close',
            'back': 'Back',
            'next': 'Next',
            'previous': 'Previous',
            'finish': 'Finish',
            'help': 'Help',
            'settings': 'Settings',
            'language': 'Language',
            
            # Feature names
            'code_review': 'Code Review',
            'terminal_assistant': 'Terminal Assistant',
            'project_analyzer': 'Project Analyzer',
            'learning_mode': 'Learning Mode',
            'multi_language': 'Multi-language Support',
            'voice_interface': 'Voice Interface',
            'custom_models': 'Custom AI Models',
            'security_auditor': 'Security Auditor',
            'performance_profiler': 'Performance Profiler',
            'collaborative_coding': 'Collaborative Coding',
            
            # Common messages
            'feature_enabled': 'Feature enabled',
            'feature_disabled': 'Feature disabled',
            'language_changed': 'Language changed to {language}',
            'translation_failed': 'Translation failed',
            'unsupported_language': 'Unsupported language: {language}',
            'loading_translations': 'Loading translations...',
            'no_translation_available': 'No translation available',
            
            # Code review
            'analyzing_code': 'Analyzing code...',
            'review_complete': 'Review complete',
            'issues_found': '{count} issues found',
            'no_issues': 'No issues found',
            'severity_critical': 'Critical',
            'severity_high': 'High',
            'severity_medium': 'Medium',
            'severity_low': 'Low',
            
            # Terminal assistant
            'command_suggestion': 'Command suggestion',
            'command_explanation': 'Command explanation',
            'error_fix': 'Error fix suggestion',
            'no_suggestions': 'No suggestions available',
            
            # Project analyzer
            'analyzing_project': 'Analyzing project...',
            'generating_docs': 'Generating documentation...',
            'project_health': 'Project health',
            'metrics': 'Metrics',
            
            # Learning mode
            'start_learning': 'Start learning',
            'learning_progress': 'Learning progress',
            'exercise_complete': 'Exercise complete',
            'next_topic': 'Next topic',
            
            # Errors
            'error_occurred': 'An error occurred',
            'invalid_input': 'Invalid input',
            'file_not_found': 'File not found',
            'permission_denied': 'Permission denied',
            'network_error': 'Network error',
            'timeout': 'Operation timed out',
        }
    
    def load_language(self, language_code: str) -> bool:
        """
        Load translations for a specific language.
        
        Args:
            language_code: Language code to load (e.g., 'es', 'fr')
            
        Returns:
            True if loaded successfully, False otherwise
        """
        if language_code in self.translations:
            return True
        
        translation_file = self.translations_dir / f'{language_code}.json'
        
        if not translation_file.exists():
            logger.warning(f"Translation file not found: {translation_file}")
            return False
        
        try:
            with open(translation_file, 'r', encoding='utf-8') as f:
                self.translations[language_code] = json.load(f)
            logger.info(f"Loaded translations for: {language_code}")
            return True
        except Exception as e:
            logger.error(f"Failed to load translations for {language_code}: {e}")
            return False
    
    def get(
        self,
        key: str,
        language_code: str = 'en',
        fallback_language: str = 'en',
        **kwargs
    ) -> str:
        """
        Get a translation for a key.
        
        Args:
            key: Translation key
            language_code: Target language code
            fallback_language: Fallback language if translation not found
            **kwargs: Format arguments for the translation string
            
        Returns:
            Translated string
        """
        # Try to load language if not already loaded
        if language_code not in self.translations:
            self.load_language(language_code)
        
        # Try target language
        if language_code in self.translations:
            translation = self.translations[language_code].get(key)
            if translation:
                try:
                    return translation.format(**kwargs)
                except KeyError as e:
                    logger.warning(f"Missing format key in translation: {e}")
                    return translation
        
        # Try fallback language
        if fallback_language in self.translations:
            translation = self.translations[fallback_language].get(key)
            if translation:
                try:
                    return translation.format(**kwargs)
                except KeyError:
                    return translation
        
        # Return key as last resort
        logger.warning(f"No translation found for key: {key}")
        return key
    
    def set(self, key: str, value: str, language_code: str = 'en'):
        """
        Set a translation for a key.
        
        Args:
            key: Translation key
            value: Translation value
            language_code: Language code
        """
        if language_code not in self.translations:
            self.translations[language_code] = {}
        
        self.translations[language_code][key] = value
        logger.debug(f"Set translation: {key} = {value} ({language_code})")
    
    def has(self, key: str, language_code: str = 'en') -> bool:
        """
        Check if a translation exists for a key.
        
        Args:
            key: Translation key
            language_code: Language code
            
        Returns:
            True if translation exists, False otherwise
        """
        if language_code not in self.translations:
            self.load_language(language_code)
        
        return (
            language_code in self.translations and
            key in self.translations[language_code]
        )
    
    def get_all(self, language_code: str = 'en') -> Dict[str, str]:
        """
        Get all translations for a language.
        
        Args:
            language_code: Language code
            
        Returns:
            Dictionary of all translations
        """
        if language_code not in self.translations:
            self.load_language(language_code)
        
        return self.translations.get(language_code, {})
    
    def save_language(self, language_code: str) -> bool:
        """
        Save translations for a language to file.
        
        Args:
            language_code: Language code to save
            
        Returns:
            True if saved successfully, False otherwise
        """
        if language_code not in self.translations:
            logger.error(f"No translations loaded for: {language_code}")
            return False
        
        translation_file = self.translations_dir / f'{language_code}.json'
        
        try:
            with open(translation_file, 'w', encoding='utf-8') as f:
                json.dump(
                    self.translations[language_code],
                    f,
                    indent=2,
                    ensure_ascii=False
                )
            logger.info(f"Saved translations for: {language_code}")
            return True
        except Exception as e:
            logger.error(f"Failed to save translations for {language_code}: {e}")
            return False
    
    def add_translations(
        self,
        translations: Dict[str, str],
        language_code: str = 'en'
    ):
        """
        Add multiple translations at once.
        
        Args:
            translations: Dictionary of key-value pairs
            language_code: Language code
        """
        if language_code not in self.translations:
            self.translations[language_code] = {}
        
        self.translations[language_code].update(translations)
        logger.info(
            f"Added {len(translations)} translations for: {language_code}"
        )
    
    def get_keys(self, language_code: str = 'en') -> List[str]:
        """
        Get all translation keys for a language.
        
        Args:
            language_code: Language code
            
        Returns:
            List of translation keys
        """
        if language_code not in self.translations:
            self.load_language(language_code)
        
        return list(self.translations.get(language_code, {}).keys())
    
    def get_missing_keys(
        self,
        target_language: str,
        reference_language: str = 'en'
    ) -> List[str]:
        """
        Get keys that are missing in target language.
        
        Args:
            target_language: Language to check
            reference_language: Reference language (usually 'en')
            
        Returns:
            List of missing keys
        """
        if reference_language not in self.translations:
            self.load_language(reference_language)
        
        if target_language not in self.translations:
            self.load_language(target_language)
        
        reference_keys = set(self.translations.get(reference_language, {}).keys())
        target_keys = set(self.translations.get(target_language, {}).keys())
        
        return list(reference_keys - target_keys)
    
    def get_coverage(
        self,
        target_language: str,
        reference_language: str = 'en'
    ) -> float:
        """
        Get translation coverage percentage.
        
        Args:
            target_language: Language to check
            reference_language: Reference language (usually 'en')
            
        Returns:
            Coverage percentage (0.0 to 1.0)
        """
        if reference_language not in self.translations:
            self.load_language(reference_language)
        
        if target_language not in self.translations:
            self.load_language(target_language)
        
        reference_keys = set(self.translations.get(reference_language, {}).keys())
        target_keys = set(self.translations.get(target_language, {}).keys())
        
        if not reference_keys:
            return 0.0
        
        return len(target_keys & reference_keys) / len(reference_keys)
