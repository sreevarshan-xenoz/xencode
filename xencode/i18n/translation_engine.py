"""
Translation engine for multi-language support.

Provides AI-powered translation with technical term preservation.
"""

import logging
from typing import Dict, List, Optional, Set
from dataclasses import dataclass
import re

logger = logging.getLogger(__name__)


@dataclass
class TranslationResult:
    """Result of a translation operation."""
    
    original_text: str
    translated_text: str
    source_language: str
    target_language: str
    technical_terms: List[str]
    confidence: float


class TranslationEngine:
    """
    AI-powered translation engine with technical term preservation.
    
    Supports translation between multiple languages while maintaining
    technical accuracy for programming terms and code snippets.
    """
    
    def __init__(self, model_provider=None):
        """
        Initialize the translation engine.
        
        Args:
            model_provider: Optional AI model provider for translations
        """
        self.model_provider = model_provider
        self.technical_terms: Set[str] = self._load_technical_terms()
        self.cache: Dict[tuple, TranslationResult] = {}
        
    def _load_technical_terms(self) -> Set[str]:
        """Load common technical terms that should not be translated."""
        return {
            # Programming concepts
            "function", "class", "method", "variable", "parameter",
            "return", "import", "export", "module", "package",
            "array", "list", "dict", "tuple", "set", "map",
            "string", "integer", "float", "boolean", "null",
            "async", "await", "promise", "callback", "closure",
            "interface", "abstract", "static", "const", "let", "var",
            
            # Common tools and technologies
            "git", "github", "gitlab", "docker", "kubernetes",
            "python", "javascript", "typescript", "rust", "go",
            "api", "rest", "graphql", "json", "xml", "yaml",
            "cli", "tui", "gui", "ide", "vscode",
            
            # Xencode-specific terms
            "xencode", "ensemble", "rag", "ollama", "bitnet",
            "terminal", "assistant", "analyzer", "profiler",
        }
    
    def translate(
        self,
        text: str,
        target_language: str,
        source_language: str = "en",
        preserve_code: bool = True
    ) -> TranslationResult:
        """
        Translate text to target language.
        
        Args:
            text: Text to translate
            target_language: Target language code (e.g., 'es', 'fr', 'zh')
            source_language: Source language code (default: 'en')
            preserve_code: Whether to preserve code snippets and technical terms
            
        Returns:
            TranslationResult with translated text and metadata
        """
        # Check cache
        cache_key = (text, source_language, target_language, preserve_code)
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Extract technical terms and code snippets
        technical_terms = []
        code_snippets = []
        
        if preserve_code:
            # Extract code blocks (```...```)
            code_pattern = r'```[\s\S]*?```'
            code_snippets = re.findall(code_pattern, text)
            
            # Extract inline code (`...`)
            inline_code_pattern = r'`[^`]+`'
            code_snippets.extend(re.findall(inline_code_pattern, text))
            
            # Find technical terms
            words = re.findall(r'\b\w+\b', text.lower())
            technical_terms = [w for w in words if w in self.technical_terms]
        
        # Perform translation
        if self.model_provider:
            translated = self._ai_translate(
                text, source_language, target_language,
                technical_terms, code_snippets
            )
        else:
            # Fallback to simple translation (placeholder)
            translated = self._simple_translate(
                text, source_language, target_language,
                technical_terms, code_snippets
            )
        
        result = TranslationResult(
            original_text=text,
            translated_text=translated,
            source_language=source_language,
            target_language=target_language,
            technical_terms=technical_terms,
            confidence=0.9 if self.model_provider else 0.5
        )
        
        # Cache result
        self.cache[cache_key] = result
        
        return result
    
    def _ai_translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        technical_terms: List[str],
        code_snippets: List[str]
    ) -> str:
        """
        Use AI model to translate text.
        
        Args:
            text: Text to translate
            source_lang: Source language code
            target_lang: Target language code
            technical_terms: List of technical terms to preserve
            code_snippets: List of code snippets to preserve
            
        Returns:
            Translated text
        """
        # Create prompt for AI translation
        prompt = f"""Translate the following text from {source_lang} to {target_lang}.

IMPORTANT RULES:
1. Preserve all code snippets exactly as they are (text in backticks or code blocks)
2. Do not translate these technical terms: {', '.join(technical_terms)}
3. Maintain the original formatting and structure
4. Ensure technical accuracy for programming concepts

Text to translate:
{text}

Provide only the translated text, no explanations."""

        try:
            # Use model provider to translate
            response = self.model_provider.generate(prompt)
            return response.strip()
        except Exception as e:
            logger.error(f"AI translation failed: {e}")
            return self._simple_translate(
                text, source_lang, target_lang,
                technical_terms, code_snippets
            )
    
    def _simple_translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        technical_terms: List[str],
        code_snippets: List[str]
    ) -> str:
        """
        Simple fallback translation (returns original text).
        
        In a production system, this would use a translation library
        like googletrans or a translation API.
        
        Args:
            text: Text to translate
            source_lang: Source language code
            target_lang: Target language code
            technical_terms: List of technical terms to preserve
            code_snippets: List of code snippets to preserve
            
        Returns:
            Translated text (currently returns original)
        """
        # For now, return original text as fallback
        # In production, integrate with translation library
        logger.warning(
            f"Using fallback translation for {source_lang} -> {target_lang}"
        )
        return text
    
    def detect_language(self, text: str) -> str:
        """
        Detect the language of the given text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Language code (e.g., 'en', 'es', 'fr')
        """
        # Simple heuristic-based detection
        # In production, use langdetect or similar library
        
        # Check for common words in different languages
        text_lower = text.lower()
        
        # Spanish indicators
        if any(word in text_lower for word in ['el', 'la', 'los', 'las', 'de', 'que', 'y', 'es']):
            return 'es'
        
        # French indicators
        if any(word in text_lower for word in ['le', 'la', 'les', 'de', 'que', 'et', 'est']):
            return 'fr'
        
        # German indicators
        if any(word in text_lower for word in ['der', 'die', 'das', 'und', 'ist', 'ein', 'eine']):
            return 'de'
        
        # Chinese indicators (simplified)
        if any('\u4e00' <= char <= '\u9fff' for char in text):
            return 'zh'
        
        # Japanese indicators
        if any('\u3040' <= char <= '\u309f' or '\u30a0' <= char <= '\u30ff' for char in text):
            return 'ja'
        
        # Korean indicators
        if any('\uac00' <= char <= '\ud7af' for char in text):
            return 'ko'
        
        # Arabic indicators
        if any('\u0600' <= char <= '\u06ff' for char in text):
            return 'ar'
        
        # Default to English
        return 'en'
    
    def batch_translate(
        self,
        texts: List[str],
        target_language: str,
        source_language: str = "en"
    ) -> List[TranslationResult]:
        """
        Translate multiple texts efficiently.
        
        Args:
            texts: List of texts to translate
            target_language: Target language code
            source_language: Source language code
            
        Returns:
            List of TranslationResult objects
        """
        results = []
        for text in texts:
            result = self.translate(text, target_language, source_language)
            results.append(result)
        return results
    
    def clear_cache(self):
        """Clear the translation cache."""
        self.cache.clear()
        logger.info("Translation cache cleared")
    
    def add_technical_term(self, term: str):
        """
        Add a custom technical term to preserve during translation.
        
        Args:
            term: Technical term to add
        """
        self.technical_terms.add(term.lower())
        logger.debug(f"Added technical term: {term}")
    
    def remove_technical_term(self, term: str):
        """
        Remove a technical term from the preservation list.
        
        Args:
            term: Technical term to remove
        """
        self.technical_terms.discard(term.lower())
        logger.debug(f"Removed technical term: {term}")
