"""
Demo of multi-language support (i18n) functionality.

This example demonstrates:
- Language detection and switching
- UI translation
- Response translation with code preservation
- RTL language support
- Technical term handling
"""

from xencode.i18n import (
    TranslationEngine,
    LanguageManager,
    TranslationDictionary,
    ContextAdapter
)


def demo_translation_engine():
    """Demonstrate TranslationEngine capabilities."""
    print("=" * 60)
    print("Translation Engine Demo")
    print("=" * 60)
    
    engine = TranslationEngine()
    
    # Translate text
    print("\n1. Basic Translation:")
    result = engine.translate("Hello, welcome to Xencode!", "es", "en")
    print(f"   Original: {result.original_text}")
    print(f"   Translated: {result.translated_text}")
    print(f"   Confidence: {result.confidence}")
    
    # Translate with code preservation
    print("\n2. Translation with Code Preservation:")
    text = "Use the `print()` function to display output"
    result = engine.translate(text, "es", "en", preserve_code=True)
    print(f"   Original: {text}")
    print(f"   Translated: {result.translated_text}")
    print(f"   Technical terms: {result.technical_terms}")
    
    # Language detection
    print("\n3. Language Detection:")
    texts = [
        "This is English text",
        "Este es texto en español",
        "Ceci est un texte en français",
        "这是中文文本",
        "これは日本語です",
        "هذا نص عربي"
    ]
    
    for text in texts:
        detected = engine.detect_language(text)
        print(f"   '{text[:30]}...' -> {detected}")
    
    # Batch translation
    print("\n4. Batch Translation:")
    texts = ["Hello", "World", "Welcome"]
    results = engine.batch_translate(texts, "es", "en")
    for i, result in enumerate(results):
        print(f"   {texts[i]} -> {result.translated_text}")


def demo_language_manager():
    """Demonstrate LanguageManager capabilities."""
    print("\n" + "=" * 60)
    print("Language Manager Demo")
    print("=" * 60)
    
    manager = LanguageManager()
    
    # List supported languages
    print("\n1. Supported Languages:")
    languages = manager.list_languages()
    print(f"   Total: {len(languages)} languages")
    for lang in languages[:5]:  # Show first 5
        print(f"   - {lang.code}: {lang.name} ({lang.native_name})")
    
    # Get current language
    print(f"\n2. Current Language: {manager.get_current_language()}")
    
    # Switch languages
    print("\n3. Language Switching:")
    for lang_code in ['es', 'fr', 'de']:
        manager.enable_language(lang_code)
        if manager.set_language(lang_code):
            print(f"   Switched to: {lang_code}")
            lang_info = manager.get_language_info(lang_code)
            print(f"   Name: {lang_info.name} ({lang_info.native_name})")
            print(f"   RTL: {lang_info.rtl}")
    
    # RTL language support
    print("\n4. RTL Language Support:")
    rtl_languages = ['ar', 'he']
    for lang in rtl_languages:
        is_rtl = manager.is_rtl(lang)
        lang_info = manager.get_language_info(lang)
        print(f"   {lang_info.name}: RTL = {is_rtl}")


def demo_translation_dictionary():
    """Demonstrate TranslationDictionary capabilities."""
    print("\n" + "=" * 60)
    print("Translation Dictionary Demo")
    print("=" * 60)
    
    dictionary = TranslationDictionary()
    
    # Get UI translations
    print("\n1. UI Element Translations:")
    ui_keys = ['welcome', 'loading', 'error', 'success']
    for key in ui_keys:
        en_text = dictionary.get(key, 'en')
        print(f"   {key}: {en_text}")
    
    # Get feature translations
    print("\n2. Feature Name Translations:")
    features = ['code_review', 'terminal_assistant', 'learning_mode']
    for feature in features:
        en_text = dictionary.get(feature, 'en')
        print(f"   {feature}: {en_text}")
    
    # Format messages
    print("\n3. Message Formatting:")
    message = dictionary.get('language_changed', 'en', language='Spanish')
    print(f"   {message}")
    
    # Translation coverage
    print("\n4. Translation Coverage:")
    if dictionary.load_language('es'):
        coverage = dictionary.get_coverage('es', 'en')
        print(f"   Spanish coverage: {coverage * 100:.1f}%")
        
        missing = dictionary.get_missing_keys('es', 'en')
        if missing:
            print(f"   Missing keys: {len(missing)}")


def demo_context_adapter():
    """Demonstrate ContextAdapter capabilities."""
    print("\n" + "=" * 60)
    print("Context Adapter Demo")
    print("=" * 60)
    
    adapter = ContextAdapter()
    
    # UI translation
    print("\n1. UI Translation:")
    ui_text = adapter.translate_ui('welcome')
    print(f"   English: {ui_text}")
    
    adapter.set_language('es')
    ui_text = adapter.translate_ui('welcome')
    print(f"   Spanish: {ui_text}")
    
    # Response translation with code preservation
    print("\n2. Response Translation (with code):")
    adapter.set_language('en')
    response = "The `analyze()` function completed successfully"
    translated = adapter.translate_response(response, language='es')
    print(f"   Original: {response}")
    print(f"   Translated: {translated}")
    
    # RTL support
    print("\n3. RTL Language Support:")
    adapter.set_language('ar')
    direction = adapter.get_ui_direction()
    print(f"   UI Direction: {direction}")
    
    text = "Welcome to Xencode"
    wrapped = adapter.wrap_rtl_text(text)
    print(f"   Original: {text}")
    print(f"   RTL wrapped: {repr(wrapped)}")
    
    # Technical term translation
    print("\n4. Technical Term Handling:")
    adapter.set_language('es')
    terms = ['function', 'class', 'variable', 'python']
    for term in terms:
        translated = adapter.get_technical_term_translation(term)
        print(f"   {term} -> {translated}")
    
    # Context-aware translation
    print("\n5. Context-Aware Translation:")
    context = adapter.create_context(
        feature='code_review',
        component='review_panel',
        technical_level='advanced'
    )
    
    text = adapter.translate_ui('analyzing_code', context=context)
    print(f"   Context: {context.feature}/{context.component}")
    print(f"   Translation: {text}")
    
    # List translation
    print("\n6. List Translation:")
    items = ['welcome', 'loading', 'error']
    translated_items = adapter.translate_list(items)
    for original, translated in zip(items, translated_items):
        print(f"   {original} -> {translated}")


def demo_full_workflow():
    """Demonstrate complete multi-language workflow."""
    print("\n" + "=" * 60)
    print("Complete Multi-Language Workflow")
    print("=" * 60)
    
    adapter = ContextAdapter()
    
    # Get supported languages
    print("\n1. Available Languages:")
    languages = adapter.get_supported_languages()
    print(f"   Total: {len(languages)} languages")
    
    # Switch to Spanish
    print("\n2. Switching to Spanish:")
    adapter.set_language('es')
    print(f"   Current language: {adapter.get_current_language()}")
    
    # Translate UI elements
    print("\n3. Translating UI:")
    ui_elements = {
        'title': 'welcome',
        'button': 'confirm',
        'status': 'loading'
    }
    translated = adapter.translate_dict(ui_elements, ['title', 'button', 'status'])
    for key, value in translated.items():
        print(f"   {key}: {value}")
    
    # Translate error message
    print("\n4. Translating Error:")
    error = adapter.translate_error("File not found")
    print(f"   Error: {error}")
    
    # Switch to RTL language
    print("\n5. Switching to Arabic (RTL):")
    adapter.set_language('ar')
    direction = adapter.get_ui_direction()
    print(f"   UI Direction: {direction}")
    
    # Translate with RTL
    text = adapter.translate_ui('welcome')
    print(f"   Translated: {text}")
    
    # Switch back to English
    print("\n6. Switching back to English:")
    adapter.set_language('en')
    print(f"   Current language: {adapter.get_current_language()}")


def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("XENCODE MULTI-LANGUAGE SUPPORT DEMO")
    print("=" * 60)
    
    try:
        demo_translation_engine()
        demo_language_manager()
        demo_translation_dictionary()
        demo_context_adapter()
        demo_full_workflow()
        
        print("\n" + "=" * 60)
        print("Demo completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
