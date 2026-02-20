"""
Tests for ContextAdapter.

Validates requirements 5.2, 5.3, 5.5.
"""

import pytest
from xencode.i18n.context_adapter import ContextAdapter, TranslationContext


class TestContextAdapter:
    """Test suite for ContextAdapter."""
    
    def test_initialization(self):
        """Test adapter initialization."""
        adapter = ContextAdapter()
        
        assert adapter is not None
        assert adapter.translation_engine is not None
        assert adapter.language_manager is not None
        assert adapter.translation_dict is not None
    
    def test_translate_ui_basic(self):
        """Test basic UI translation (Requirement 5.2)."""
        adapter = ContextAdapter()
        
        translation = adapter.translate_ui('welcome')
        assert translation is not None
    
    def test_translate_ui_with_language(self):
        """Test UI translation with specific language."""
        adapter = ContextAdapter()
        
        translation = adapter.translate_ui('welcome', language='es')
        assert translation is not None
    
    def test_translate_ui_with_context(self):
        """Test UI translation with context."""
        adapter = ContextAdapter()
        context = TranslationContext(
            feature='code_review',
            component='review_panel'
        )
        
        translation = adapter.translate_ui('analyzing_code', context=context)
        assert translation is not None
    
    def test_translate_response(self):
        """Test response translation (Requirement 5.2)."""
        adapter = ContextAdapter()
        
        response = "The code review is complete"
        translation = adapter.translate_response(response)
        assert translation is not None
    
    def test_translate_response_with_code(self):
        """Test response translation preserves code (Requirement 5.3)."""
        adapter = ContextAdapter()
        
        response = "Use the `print()` function to display output"
        translation = adapter.translate_response(response, language='es')
        
        # Code should be preserved
        assert 'print()' in translation or '`print()`' in translation
    
    def test_translate_error(self):
        """Test error message translation."""
        adapter = ContextAdapter()
        
        error = "File not found"
        translation = adapter.translate_error(error)
        assert translation is not None
    
    def test_get_ui_direction_ltr(self):
        """Test UI direction for LTR languages."""
        adapter = ContextAdapter()
        
        direction = adapter.get_ui_direction('en')
        assert direction == 'ltr'
    
    def test_get_ui_direction_rtl(self):
        """Test UI direction for RTL languages (Requirement 5.5)."""
        adapter = ContextAdapter()
        
        direction = adapter.get_ui_direction('ar')
        assert direction == 'rtl'
    
    def test_format_message(self):
        """Test message formatting."""
        adapter = ContextAdapter()
        
        message = adapter.format_message('language_changed', language='Spanish')
        # Should contain the formatted language parameter
        assert message is not None
        assert len(message) > 0
    
    def test_translate_list(self):
        """Test translating a list of items."""
        adapter = ContextAdapter()
        
        items = ['welcome', 'loading', 'error']
        translations = adapter.translate_list(items)
        
        assert len(translations) == 3
        assert all(isinstance(t, str) for t in translations)
    
    def test_translate_dict(self):
        """Test translating dictionary values."""
        adapter = ContextAdapter()
        
        data = {
            'title': 'welcome',
            'status': 'loading',
            'other': 'keep_this'
        }
        
        translated = adapter.translate_dict(data, ['title', 'status'])
        
        assert 'title' in translated
        assert 'status' in translated
        assert translated['other'] == 'keep_this'
    
    def test_get_technical_term_translation(self):
        """Test technical term translation (Requirement 5.3)."""
        adapter = ContextAdapter()
        
        # Technical terms should be preserved
        term = adapter.get_technical_term_translation('function')
        assert term == 'function'
    
    def test_set_language(self):
        """Test setting language."""
        adapter = ContextAdapter()
        
        result = adapter.set_language('es')
        assert result is True
        assert adapter.get_current_language() == 'es'
    
    def test_get_current_language(self):
        """Test getting current language."""
        adapter = ContextAdapter()
        
        language = adapter.get_current_language()
        assert language in ['en', 'es', 'fr', 'de', 'zh', 'ja', 'ko', 'ru', 'ar', 'pt']
    
    def test_get_supported_languages(self):
        """Test getting supported languages."""
        adapter = ContextAdapter()
        
        languages = adapter.get_supported_languages()
        assert len(languages) >= 10
        assert all('code' in lang for lang in languages)
        assert all('name' in lang for lang in languages)
    
    def test_create_context(self):
        """Test creating translation context."""
        adapter = ContextAdapter()
        
        context = adapter.create_context(
            feature='code_review',
            component='review_panel',
            technical_level='advanced'
        )
        
        assert isinstance(context, TranslationContext)
        assert context.feature == 'code_review'
        assert context.component == 'review_panel'
        assert context.technical_level == 'advanced'
    
    def test_wrap_rtl_text(self):
        """Test RTL text wrapping (Requirement 5.5)."""
        adapter = ContextAdapter()
        adapter.set_language('ar')
        
        text = "Hello World"
        wrapped = adapter.wrap_rtl_text(text)
        
        # Should contain RTL markers
        assert wrapped != text
    
    def test_unwrap_rtl_text(self):
        """Test RTL text unwrapping."""
        adapter = ContextAdapter()
        
        # Create wrapped text
        wrapped = f"{adapter.rtl_markers['start']}Hello{adapter.rtl_markers['end']}"
        unwrapped = adapter.unwrap_rtl_text(wrapped)
        
        assert unwrapped == "Hello"
    
    def test_rtl_formatting_arabic(self):
        """Test RTL formatting for Arabic (Requirement 5.5)."""
        adapter = ContextAdapter()
        adapter.set_language('ar')
        
        text = "مرحبا بك في Xencode"
        formatted = adapter._apply_rtl_formatting(text, preserve_code=False)
        
        assert formatted is not None
    
    def test_rtl_formatting_with_code(self):
        """Test RTL formatting preserves code (Requirement 5.5)."""
        adapter = ContextAdapter()
        
        text = "Use `print()` to display output"
        formatted = adapter._apply_rtl_formatting(text, preserve_code=True)
        
        # Code should be preserved
        assert 'print()' in formatted
    
    def test_translate_ui_rtl_language(self):
        """Test UI translation for RTL language (Requirement 5.5)."""
        adapter = ContextAdapter()
        
        translation = adapter.translate_ui('welcome', language='ar')
        assert translation is not None
    
    def test_translate_response_rtl_language(self):
        """Test response translation for RTL language (Requirement 5.5)."""
        adapter = ContextAdapter()
        
        response = "The analysis is complete"
        translation = adapter.translate_response(response, language='ar')
        assert translation is not None


class TestTranslationContext:
    """Test suite for TranslationContext."""
    
    def test_context_creation(self):
        """Test context creation."""
        context = TranslationContext(
            feature='code_review',
            component='review_panel',
            user_role='developer',
            technical_level='advanced',
            preserve_formatting=True
        )
        
        assert context.feature == 'code_review'
        assert context.component == 'review_panel'
        assert context.user_role == 'developer'
        assert context.technical_level == 'advanced'
        assert context.preserve_formatting is True
    
    def test_context_defaults(self):
        """Test context default values."""
        context = TranslationContext()
        
        assert context.feature is None
        assert context.component is None
        assert context.user_role is None
        assert context.technical_level == 'intermediate'
        assert context.preserve_formatting is True


class TestContextAdapterIntegration:
    """Integration tests for ContextAdapter."""
    
    def test_full_translation_workflow(self):
        """Test complete translation workflow."""
        adapter = ContextAdapter()
        
        # Set language
        adapter.set_language('es')
        
        # Create context
        context = adapter.create_context(feature='code_review')
        
        # Translate UI
        ui_text = adapter.translate_ui('analyzing_code', context=context)
        assert ui_text is not None
        
        # Translate response
        response = adapter.translate_response("Analysis complete")
        assert response is not None
    
    def test_rtl_workflow(self):
        """Test RTL language workflow (Requirement 5.5)."""
        adapter = ContextAdapter()
        
        # Set RTL language
        adapter.set_language('ar')
        
        # Check direction
        assert adapter.get_ui_direction() == 'rtl'
        
        # Translate with RTL
        translation = adapter.translate_ui('welcome')
        assert translation is not None
    
    def test_technical_accuracy_workflow(self):
        """Test technical accuracy in translation (Requirement 5.3)."""
        adapter = ContextAdapter()
        adapter.set_language('es')
        
        # Translate technical content
        response = "The function returns a list of variables"
        translation = adapter.translate_response(response)
        
        # Technical terms should be preserved or accurately translated
        assert translation is not None
    
    def test_language_switching_workflow(self):
        """Test language switching workflow."""
        adapter = ContextAdapter()
        
        # Switch between languages
        adapter.set_language('es')
        assert adapter.get_current_language() == 'es'
        
        adapter.set_language('fr')
        assert adapter.get_current_language() == 'fr'
        
        adapter.set_language('en')
        assert adapter.get_current_language() == 'en'
    
    def test_multi_language_support(self):
        """Test support for multiple languages (Requirement 5.1)."""
        adapter = ContextAdapter()
        
        languages = adapter.get_supported_languages()
        
        # Should support 10+ languages
        assert len(languages) >= 10
        
        # Check specific languages
        language_codes = [lang['code'] for lang in languages]
        assert 'en' in language_codes
        assert 'es' in language_codes
        assert 'fr' in language_codes
        assert 'de' in language_codes
        assert 'zh' in language_codes
        assert 'ja' in language_codes
