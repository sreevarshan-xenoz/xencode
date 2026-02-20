"""
Tests for LanguageManager.

Validates requirements 5.1, 5.4.
"""

import pytest
import tempfile
from pathlib import Path
from xencode.i18n.language_manager import LanguageManager, LanguageInfo


class TestLanguageManager:
    """Test suite for LanguageManager."""
    
    @pytest.fixture
    def temp_config_dir(self):
        """Create temporary config directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    def test_initialization(self, temp_config_dir):
        """Test manager initialization."""
        manager = LanguageManager(config_dir=temp_config_dir)
        
        assert manager is not None
        assert manager.current_language in manager.SUPPORTED_LANGUAGES
        assert manager.fallback_language == 'en'
    
    def test_supported_languages_count(self):
        """Test that 10+ languages are supported (Requirement 5.1)."""
        manager = LanguageManager()
        languages = manager.list_languages()
        
        assert len(languages) >= 10
    
    def test_supported_languages_list(self):
        """Test specific supported languages."""
        manager = LanguageManager()
        
        # Required languages from spec
        required_langs = ['en', 'es', 'fr', 'de', 'zh', 'ja']
        for lang in required_langs:
            assert manager.is_supported(lang)
    
    def test_get_current_language(self, temp_config_dir):
        """Test getting current language."""
        manager = LanguageManager(config_dir=temp_config_dir)
        current = manager.get_current_language()
        
        assert current in manager.SUPPORTED_LANGUAGES
    
    def test_set_language_valid(self, temp_config_dir):
        """Test setting valid language (Requirement 5.4)."""
        manager = LanguageManager(config_dir=temp_config_dir)
        
        result = manager.set_language('es')
        assert result is True
        assert manager.get_current_language() == 'es'
    
    def test_set_language_invalid(self, temp_config_dir):
        """Test setting invalid language."""
        manager = LanguageManager(config_dir=temp_config_dir)
        
        result = manager.set_language('invalid')
        assert result is False
        assert manager.get_current_language() != 'invalid'
    
    def test_set_language_persistence(self, temp_config_dir):
        """Test language setting persists."""
        manager1 = LanguageManager(config_dir=temp_config_dir)
        manager1.set_language('fr')
        
        # Create new manager with same config dir
        manager2 = LanguageManager(config_dir=temp_config_dir)
        assert manager2.get_current_language() == 'fr'
    
    def test_get_fallback_language(self, temp_config_dir):
        """Test getting fallback language."""
        manager = LanguageManager(config_dir=temp_config_dir)
        fallback = manager.get_fallback_language()
        
        assert fallback == 'en'
    
    def test_set_fallback_language(self, temp_config_dir):
        """Test setting fallback language."""
        manager = LanguageManager(config_dir=temp_config_dir)
        
        result = manager.set_fallback_language('es')
        assert result is True
        assert manager.get_fallback_language() == 'es'
    
    def test_list_languages(self):
        """Test listing all languages."""
        manager = LanguageManager()
        languages = manager.list_languages()
        
        assert len(languages) > 0
        assert all(isinstance(lang, LanguageInfo) for lang in languages)
    
    def test_get_language_info(self):
        """Test getting language information."""
        manager = LanguageManager()
        info = manager.get_language_info('es')
        
        assert info is not None
        assert info.code == 'es'
        assert info.name == 'Spanish'
        assert info.native_name == 'Español'
    
    def test_is_rtl_arabic(self):
        """Test RTL detection for Arabic (Requirement 5.5)."""
        manager = LanguageManager()
        
        assert manager.is_rtl('ar') is True
    
    def test_is_rtl_hebrew(self):
        """Test RTL detection for Hebrew (Requirement 5.5)."""
        manager = LanguageManager()
        
        assert manager.is_rtl('he') is True
    
    def test_is_rtl_english(self):
        """Test RTL detection for English."""
        manager = LanguageManager()
        
        assert manager.is_rtl('en') is False
    
    def test_is_rtl_current_language(self, temp_config_dir):
        """Test RTL detection for current language."""
        manager = LanguageManager(config_dir=temp_config_dir)
        manager.set_language('ar')
        
        assert manager.is_rtl() is True
    
    def test_is_supported(self):
        """Test language support checking."""
        manager = LanguageManager()
        
        assert manager.is_supported('en') is True
        assert manager.is_supported('es') is True
        assert manager.is_supported('invalid') is False
    
    def test_get_language_name_english(self):
        """Test getting language name in English."""
        manager = LanguageManager()
        name = manager.get_language_name('es', native=False)
        
        assert name == 'Spanish'
    
    def test_get_language_name_native(self):
        """Test getting language name in native language."""
        manager = LanguageManager()
        name = manager.get_language_name('es', native=True)
        
        assert name == 'Español'
    
    def test_get_language_name_invalid(self):
        """Test getting name for invalid language."""
        manager = LanguageManager()
        name = manager.get_language_name('invalid')
        
        assert name is None
    
    def test_detect_language(self):
        """Test language detection."""
        manager = LanguageManager()
        detected = manager.detect_language("This is English text")
        
        assert detected in manager.SUPPORTED_LANGUAGES
    
    def test_get_enabled_languages(self):
        """Test getting enabled languages."""
        manager = LanguageManager()
        enabled = manager.get_enabled_languages()
        
        assert len(enabled) > 0
        assert all(lang.enabled for lang in enabled)
    
    def test_enable_language(self):
        """Test enabling a language."""
        manager = LanguageManager()
        
        result = manager.enable_language('es')
        assert result is True
    
    def test_disable_language(self):
        """Test disabling a language."""
        manager = LanguageManager()
        
        result = manager.disable_language('es')
        assert result is True
    
    def test_disable_english_fails(self):
        """Test that English cannot be disabled."""
        manager = LanguageManager()
        
        result = manager.disable_language('en')
        assert result is False
    
    def test_language_switching_without_restart(self, temp_config_dir):
        """Test language switching without restart (Requirement 5.4)."""
        manager = LanguageManager(config_dir=temp_config_dir)
        
        # Enable languages first
        manager.enable_language('es')
        manager.enable_language('fr')
        manager.enable_language('de')
        
        # Switch languages multiple times
        manager.set_language('es')
        assert manager.get_current_language() == 'es'
        
        manager.set_language('fr')
        assert manager.get_current_language() == 'fr'
        
        manager.set_language('de')
        assert manager.get_current_language() == 'de'
    
    def test_rtl_languages_supported(self):
        """Test RTL language support (Requirement 5.5)."""
        manager = LanguageManager()
        
        # Check that RTL languages are supported
        rtl_languages = ['ar', 'he']
        for lang in rtl_languages:
            assert manager.is_supported(lang)
            assert manager.is_rtl(lang)


class TestLanguageInfo:
    """Test suite for LanguageInfo."""
    
    def test_language_info_creation(self):
        """Test LanguageInfo creation."""
        info = LanguageInfo(
            code='es',
            name='Spanish',
            native_name='Español',
            rtl=False,
            enabled=True
        )
        
        assert info.code == 'es'
        assert info.name == 'Spanish'
        assert info.native_name == 'Español'
        assert info.rtl is False
        assert info.enabled is True
    
    def test_language_info_defaults(self):
        """Test LanguageInfo default values."""
        info = LanguageInfo(
            code='en',
            name='English',
            native_name='English'
        )
        
        assert info.rtl is False
        assert info.enabled is True
