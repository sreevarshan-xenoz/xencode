# Multi-Language Support (i18n)

Comprehensive internationalization (i18n) module for Xencode, providing translation, language management, and RTL support.

## Features

- **10+ Language Support**: English, Spanish, French, German, Chinese, Japanese, Korean, Russian, Arabic, Portuguese, Italian, Dutch, Polish, Turkish, Hebrew
- **AI-Powered Translation**: Intelligent translation with technical term preservation
- **Code Preservation**: Maintains code snippets and technical terms during translation
- **RTL Support**: Full support for right-to-left languages (Arabic, Hebrew)
- **Runtime Language Switching**: Change languages without restart
- **Context-Aware Translation**: Adapts translations based on context
- **Translation Dictionary**: Fast lookup for common UI elements
- **Language Detection**: Automatic language detection from text

## Requirements Validation

This module implements requirements 5.1-5.5 from the xencode-features spec:

- ✅ **5.1**: Support 10+ languages (15 languages supported)
- ✅ **5.2**: Translate all UI elements and responses
- ✅ **5.3**: Maintain technical accuracy in translations
- ✅ **5.4**: Allow language switching without restart
- ✅ **5.5**: Support RTL languages (Arabic, Hebrew)

## Components

### TranslationEngine

AI-powered translation with technical term preservation.

```python
from xencode.i18n import TranslationEngine

engine = TranslationEngine()

# Translate text
result = engine.translate("Hello World", "es", "en")
print(result.translated_text)  # "Hola Mundo" (or similar)

# Translate with code preservation
text = "Use the `print()` function"
result = engine.translate(text, "es", "en", preserve_code=True)
# Code snippets are preserved

# Detect language
detected = engine.detect_language("Este es español")
print(detected)  # "es"

# Batch translation
texts = ["Hello", "World", "Test"]
results = engine.batch_translate(texts, "es", "en")
```

### LanguageManager

Manages language settings, detection, and switching.

```python
from xencode.i18n import LanguageManager

manager = LanguageManager()

# Get current language
current = manager.get_current_language()

# Set language
manager.set_language('es')

# List supported languages
languages = manager.list_languages()
for lang in languages:
    print(f"{lang.code}: {lang.name} ({lang.native_name})")

# Check RTL
is_rtl = manager.is_rtl('ar')  # True

# Get language info
info = manager.get_language_info('es')
print(info.name)  # "Spanish"
print(info.native_name)  # "Español"
```

### TranslationDictionary

Fast lookup for pre-translated UI elements.

```python
from xencode.i18n import TranslationDictionary

dictionary = TranslationDictionary()

# Get translation
text = dictionary.get('welcome', 'es')
print(text)  # "Bienvenido a Xencode"

# Format message
message = dictionary.get('language_changed', 'en', language='Spanish')
print(message)  # "Language changed to Spanish"

# Add custom translations
dictionary.set('custom_key', 'Custom Value', 'en')

# Check coverage
coverage = dictionary.get_coverage('es', 'en')
print(f"Coverage: {coverage * 100}%")
```

### ContextAdapter

Context-aware translation with RTL support.

```python
from xencode.i18n import ContextAdapter

adapter = ContextAdapter()

# Translate UI
text = adapter.translate_ui('welcome')

# Translate with specific language
text = adapter.translate_ui('loading', language='es')

# Translate response (preserves code)
response = "Use `git commit` to save"
translated = adapter.translate_response(response, language='es')

# Get UI direction
direction = adapter.get_ui_direction('ar')  # "rtl"

# Create context
context = adapter.create_context(
    feature='code_review',
    component='review_panel'
)

# Translate with context
text = adapter.translate_ui('analyzing_code', context=context)

# RTL text wrapping
adapter.set_language('ar')
wrapped = adapter.wrap_rtl_text("Hello")

# Translate list
items = ['welcome', 'loading', 'error']
translated = adapter.translate_list(items)

# Translate dictionary
data = {'title': 'welcome', 'status': 'loading'}
translated = adapter.translate_dict(data, ['title', 'status'])
```

## Supported Languages

| Code | Language | Native Name | RTL |
|------|----------|-------------|-----|
| en | English | English | No |
| es | Spanish | Español | No |
| fr | French | Français | No |
| de | German | Deutsch | No |
| zh | Chinese | 中文 | No |
| ja | Japanese | 日本語 | No |
| ko | Korean | 한국어 | No |
| ru | Russian | Русский | No |
| ar | Arabic | العربية | Yes |
| pt | Portuguese | Português | No |
| it | Italian | Italiano | No |
| nl | Dutch | Nederlands | No |
| pl | Polish | Polski | No |
| tr | Turkish | Türkçe | No |
| he | Hebrew | עברית | Yes |

## Translation Files

Translation files are stored in `xencode/i18n/translations/` as JSON files:

```json
{
  "welcome": "Bienvenido a Xencode",
  "loading": "Cargando...",
  "error": "Error",
  "success": "Éxito"
}
```

To add a new language:

1. Create `{language_code}.json` in `translations/` directory
2. Add translations for all keys from `en.json`
3. The language will be automatically available

## Technical Term Preservation

The translation engine preserves technical terms during translation:

```python
engine = TranslationEngine()

# Technical terms are preserved
text = "The function returns a list"
result = engine.translate(text, "es", "en")
# "function" and "list" are preserved or accurately translated

# Add custom technical terms
engine.add_technical_term("xencode")
engine.add_technical_term("ensemble")
```

Default technical terms include:
- Programming concepts: function, class, method, variable, etc.
- Common tools: git, docker, python, javascript, etc.
- Xencode-specific: xencode, ensemble, rag, ollama, etc.

## RTL Language Support

Full support for right-to-left languages:

```python
adapter = ContextAdapter()
adapter.set_language('ar')

# Check direction
direction = adapter.get_ui_direction()  # "rtl"

# Wrap text with RTL markers
text = adapter.wrap_rtl_text("مرحبا")

# RTL formatting preserves code
text = "Use `print()` to display"
formatted = adapter._apply_rtl_formatting(text, preserve_code=True)
# Code remains LTR, text is RTL
```

## Configuration

Language settings are stored in `~/.xencode/i18n/language_config.json`:

```json
{
  "current_language": "es",
  "fallback_language": "en"
}
```

## Usage Examples

### Basic Translation

```python
from xencode.i18n import ContextAdapter

adapter = ContextAdapter()

# Set language
adapter.set_language('es')

# Translate UI
welcome = adapter.translate_ui('welcome')
print(welcome)  # "Bienvenido a Xencode"
```

### Code Review Feature

```python
from xencode.i18n import ContextAdapter

adapter = ContextAdapter()
adapter.set_language('es')

# Create context
context = adapter.create_context(feature='code_review')

# Translate UI elements
status = adapter.translate_ui('analyzing_code', context=context)
complete = adapter.translate_ui('review_complete', context=context)

# Translate response with code
response = "Found issue in `main.py` at line 42"
translated = adapter.translate_response(response)
```

### RTL Language

```python
from xencode.i18n import ContextAdapter

adapter = ContextAdapter()
adapter.set_language('ar')

# Check if RTL
if adapter.get_ui_direction() == 'rtl':
    # Apply RTL layout
    text = adapter.wrap_rtl_text("مرحبا بك في Xencode")
```

### Multi-Language Application

```python
from xencode.i18n import ContextAdapter

adapter = ContextAdapter()

# Get available languages
languages = adapter.get_supported_languages()

# Display language selector
for lang in languages:
    print(f"{lang['code']}: {lang['native_name']}")

# User selects language
adapter.set_language('fr')

# All UI is now in French
ui_text = adapter.translate_ui('welcome')
```

## Testing

Comprehensive test suite with 105 tests covering all functionality:

```bash
# Run all i18n tests
pytest tests/i18n/ -v

# Run specific test file
pytest tests/i18n/test_translation_engine.py -v

# Run with coverage
pytest tests/i18n/ --cov=xencode.i18n --cov-report=html
```

## Performance

- Translation caching for improved performance
- Fast dictionary lookup for common UI elements
- Minimal memory footprint (<100MB)
- Response time <100ms for most operations

## Future Enhancements

- Integration with external translation APIs (Google Translate, DeepL)
- Machine learning-based translation quality improvement
- Automatic translation file generation
- Translation memory for consistency
- Glossary management for technical terms
- Translation validation and quality checks

## Contributing

To add support for a new language:

1. Create translation file in `xencode/i18n/translations/{code}.json`
2. Add language info to `LanguageManager.SUPPORTED_LANGUAGES`
3. Add tests for the new language
4. Update documentation

## License

Part of the Xencode project. See main LICENSE file.
