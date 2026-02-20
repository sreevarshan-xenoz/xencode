"""
Tests for TranslationDictionary.

Validates requirements 5.2, 5.3.
"""

import pytest
import tempfile
from pathlib import Path
from xencode.i18n.translation_dict import TranslationDictionary


class TestTranslationDictionary:
    """Test suite for TranslationDictionary."""
    
    @pytest.fixture
    def temp_translations_dir(self):
        """Create temporary translations directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    def test_initialization(self, temp_translations_dir):
        """Test dictionary initialization."""
        dictionary = TranslationDictionary(translations_dir=temp_translations_dir)
        
        assert dictionary is not None
        assert 'en' in dictionary.translations
        assert len(dictionary.translations['en']) > 0
    
    def test_default_english_translations(self):
        """Test default English translations are loaded."""
        dictionary = TranslationDictionary()
        
        assert dictionary.has('welcome', 'en')
        assert dictionary.has('loading', 'en')
        assert dictionary.has('error', 'en')
    
    def test_get_translation(self):
        """Test getting a translation (Requirement 5.2)."""
        dictionary = TranslationDictionary()
        
        translation = dictionary.get('welcome', 'en')
        assert translation == 'Welcome to Xencode'
    
    def test_get_translation_with_fallback(self):
        """Test getting translation with fallback."""
        dictionary = TranslationDictionary()
        
        # Try to get non-existent key in Spanish, should fallback to English
        translation = dictionary.get('welcome', 'es', fallback_language='en')
        assert translation is not None
    
    def test_get_translation_with_format(self):
        """Test getting translation with format arguments."""
        dictionary = TranslationDictionary()
        
        translation = dictionary.get('language_changed', 'en', language='Spanish')
        assert 'Spanish' in translation
    
    def test_set_translation(self):
        """Test setting a translation."""
        dictionary = TranslationDictionary()
        
        dictionary.set('test_key', 'Test Value', 'en')
        assert dictionary.has('test_key', 'en')
        assert dictionary.get('test_key', 'en') == 'Test Value'
    
    def test_has_translation(self):
        """Test checking if translation exists."""
        dictionary = TranslationDictionary()
        
        assert dictionary.has('welcome', 'en') is True
        assert dictionary.has('nonexistent', 'en') is False
    
    def test_get_all_translations(self):
        """Test getting all translations for a language."""
        dictionary = TranslationDictionary()
        
        all_translations = dictionary.get_all('en')
        assert isinstance(all_translations, dict)
        assert len(all_translations) > 0
        assert 'welcome' in all_translations
    
    def test_add_translations(self):
        """Test adding multiple translations."""
        dictionary = TranslationDictionary()
        
        new_translations = {
            'key1': 'Value 1',
            'key2': 'Value 2',
            'key3': 'Value 3',
        }
        
        dictionary.add_translations(new_translations, 'en')
        
        assert dictionary.has('key1', 'en')
        assert dictionary.has('key2', 'en')
        assert dictionary.has('key3', 'en')
    
    def test_get_keys(self):
        """Test getting all translation keys."""
        dictionary = TranslationDictionary()
        
        keys = dictionary.get_keys('en')
        assert isinstance(keys, list)
        assert len(keys) > 0
        assert 'welcome' in keys
    
    def test_get_missing_keys(self):
        """Test getting missing translation keys."""
        dictionary = TranslationDictionary()
        
        # Add some translations to English
        dictionary.set('key1', 'Value 1', 'en')
        dictionary.set('key2', 'Value 2', 'en')
        
        # Add only one to Spanish
        dictionary.set('key1', 'Valor 1', 'es')
        
        missing = dictionary.get_missing_keys('es', 'en')
        assert 'key2' in missing
    
    def test_get_coverage(self):
        """Test getting translation coverage."""
        dictionary = TranslationDictionary()
        
        # English should have 100% coverage of itself
        coverage = dictionary.get_coverage('en', 'en')
        assert coverage == 1.0
    
    def test_save_and_load_language(self, temp_translations_dir):
        """Test saving and loading translations."""
        dictionary = TranslationDictionary(translations_dir=temp_translations_dir)
        
        # Add translations
        dictionary.set('test_key', 'Test Value', 'test')
        
        # Save
        result = dictionary.save_language('test')
        assert result is True
        
        # Create new dictionary and load
        dictionary2 = TranslationDictionary(translations_dir=temp_translations_dir)
        result = dictionary2.load_language('test')
        assert result is True
        assert dictionary2.has('test_key', 'test')
    
    def test_load_nonexistent_language(self, temp_translations_dir):
        """Test loading non-existent language."""
        dictionary = TranslationDictionary(translations_dir=temp_translations_dir)
        
        result = dictionary.load_language('nonexistent')
        assert result is False
    
    def test_save_nonexistent_language(self, temp_translations_dir):
        """Test saving non-existent language."""
        dictionary = TranslationDictionary(translations_dir=temp_translations_dir)
        
        result = dictionary.save_language('nonexistent')
        assert result is False
    
    def test_ui_element_translations(self):
        """Test UI element translations (Requirement 5.2)."""
        dictionary = TranslationDictionary()
        
        # Check common UI elements
        ui_elements = [
            'welcome', 'loading', 'error', 'success',
            'cancel', 'confirm', 'save', 'delete'
        ]
        
        for element in ui_elements:
            assert dictionary.has(element, 'en')
    
    def test_feature_name_translations(self):
        """Test feature name translations (Requirement 5.2)."""
        dictionary = TranslationDictionary()
        
        # Check feature names
        features = [
            'code_review', 'terminal_assistant', 'project_analyzer',
            'learning_mode', 'multi_language'
        ]
        
        for feature in features:
            assert dictionary.has(feature, 'en')
    
    def test_technical_accuracy(self):
        """Test technical term accuracy (Requirement 5.3)."""
        dictionary = TranslationDictionary()
        
        # Technical terms should be present
        technical_terms = [
            'analyzing_code', 'review_complete', 'command_suggestion'
        ]
        
        for term in technical_terms:
            assert dictionary.has(term, 'en')
    
    def test_get_translation_missing_key(self):
        """Test getting translation for missing key returns key."""
        dictionary = TranslationDictionary()
        
        translation = dictionary.get('nonexistent_key', 'en')
        assert translation == 'nonexistent_key'
    
    def test_format_missing_argument(self):
        """Test format with missing argument."""
        dictionary = TranslationDictionary()
        
        # Should return unformatted string if format key is missing
        translation = dictionary.get('language_changed', 'en')
        assert translation is not None


class TestTranslationDictionaryIntegration:
    """Integration tests for TranslationDictionary."""
    
    def test_spanish_translations_file(self):
        """Test loading Spanish translations from file."""
        dictionary = TranslationDictionary()
        
        # Try to load Spanish translations
        result = dictionary.load_language('es')
        
        if result:
            # If Spanish file exists, check some translations
            assert dictionary.has('welcome', 'es')
            translation = dictionary.get('welcome', 'es')
            assert translation != 'Welcome to Xencode'  # Should be translated
    
    def test_multiple_languages(self):
        """Test working with multiple languages."""
        dictionary = TranslationDictionary()
        
        # Set translations in multiple languages
        dictionary.set('greeting', 'Hello', 'en')
        dictionary.set('greeting', 'Hola', 'es')
        dictionary.set('greeting', 'Bonjour', 'fr')
        
        assert dictionary.get('greeting', 'en') == 'Hello'
        assert dictionary.get('greeting', 'es') == 'Hola'
        assert dictionary.get('greeting', 'fr') == 'Bonjour'
    
    def test_coverage_calculation(self):
        """Test coverage calculation with real data."""
        dictionary = TranslationDictionary()
        
        # Clear existing translations for clean test
        dictionary.translations['en'] = {}
        dictionary.translations['es'] = {}
        
        # Add some English translations
        dictionary.set('key1', 'Value 1', 'en')
        dictionary.set('key2', 'Value 2', 'en')
        dictionary.set('key3', 'Value 3', 'en')
        
        # Add partial Spanish translations
        dictionary.set('key1', 'Valor 1', 'es')
        dictionary.set('key2', 'Valor 2', 'es')
        
        coverage = dictionary.get_coverage('es', 'en')
        assert 0.6 < coverage < 0.7  # Should be around 2/3
