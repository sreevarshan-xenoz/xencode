#!/usr/bin/env python3
"""
Unified Model Client for Xencode

Bridges the ModelProviderManager with all subsystems (ensemble, agentic, TUI, core).
Provides a single interface for generating text from any LLM provider.

Supported providers: ollama, openai, anthropic, huggingface, google_gemini, openrouter, qwen
"""

import asyncio
import os
from typing import AsyncIterator, Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum


class ProviderType(Enum):
    """All supported LLM provider types."""
    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    HUGGINGFACE = "huggingface"
    GOOGLE_GEMINI = "google_gemini"
    OPENROUTER = "openrouter"
    QWEN = "qwen"


@dataclass
class ProviderConfig:
    """Configuration for a single LLM provider."""
    name: str
    provider_type: ProviderType
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    enabled: bool = True
    models: Optional[List[str]] = None


@dataclass
class ModelClientConfig:
    """Configuration for the unified model client."""
    default_provider: str = "ollama"
    default_model: str = "llama3.1:8b"
    provider_configs: Optional[List[ProviderConfig]] = None
    fallback_enabled: bool = True
    max_retries: int = 2
    timeout_seconds: int = 60
    def __post_init__(self):
        if self.provider_configs is None:
            self.provider_configs = self._default_providers()

    def _default_providers(self) -> List[ProviderConfig]:
        return [
            ProviderConfig(
                name="ollama",
                provider_type=ProviderType.OLLAMA,
                base_url=os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
                enabled=True,
                models=["llama3.1:8b", "qwen3:4b", "mistral:7b", "phi3:mini"],
            ),
            ProviderConfig(
                name="openai",
                provider_type=ProviderType.OPENAI,
                api_key=os.environ.get("OPENAI_API_KEY", ""),
                enabled=bool(os.environ.get("OPENAI_API_KEY")),
                models=["gpt-4o", "gpt-4o-mini", "o3-mini"],
            ),
            ProviderConfig(
                name="anthropic",
                provider_type=ProviderType.ANTHROPIC,
                api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
                enabled=bool(os.environ.get("ANTHROPIC_API_KEY")),
                models=["claude-sonnet-4-20250514", "claude-haiku-4-20250514"],
            ),
            ProviderConfig(
                name="google_gemini",
                provider_type=ProviderType.GOOGLE_GEMINI,
                api_key=os.environ.get("GOOGLE_API_KEY", ""),
                enabled=bool(os.environ.get("GOOGLE_API_KEY")),
                models=["gemini-2.0-flash", "gemini-2.0-pro"],
            ),
            ProviderConfig(
                name="openrouter",
                provider_type=ProviderType.OPENROUTER,
                api_key=os.environ.get("OPENROUTER_API_KEY", ""),
                enabled=bool(os.environ.get("OPENROUTER_API_KEY")),
                models=["openai/gpt-4o", "anthropic/claude-sonnet-4", "google/gemini-2.0-flash"],
            ),
            ProviderConfig(
                name="qwen",
                provider_type=ProviderType.QWEN,
                enabled=True,  # Qwen uses OAuth2 device flow
                models=["qwen-max", "qwen-plus", "qwen-turbo"],
            ),
            ProviderConfig(
                name="huggingface",
                provider_type=ProviderType.HUGGINGFACE,
                api_key=os.environ.get("HUGGINGFACE_API_KEY", ""),
                enabled=bool(os.environ.get("HUGGINGFACE_API_KEY")),
                models=["meta-llama/Meta-Llama-3-8B-Instruct", "mistralai/Mixtral-8x7B-Instruct-v0.1"],
            ),
        ]


class UnifiedModelClient:
    """
    Unified client for all LLM providers.

    Uses ModelProviderManager under the hood and provides:
    - Provider-agnostic generate() and chat() methods
    - Automatic fallback to next provider on failure
    - Provider discovery and health checking
    """

    def __init__(self, config: Optional[ModelClientConfig] = None):
        self.config = config or ModelClientConfig()
        self._manager = None
        self._provider_order = self._build_provider_order()

    def _get_manager(self):
        """Lazy-initialize the provider manager."""
        if self._manager is None:
            from xencode.model_providers import get_model_provider_manager
            self._manager = get_model_provider_manager()
            # Initialize all configured providers
            for pc in self.config.provider_configs:
                if pc.enabled:
                    kwargs = {}
                    if pc.api_key:
                        kwargs["api_key"] = pc.api_key
                    if pc.base_url:
                        kwargs["base_url"] = pc.base_url
                    self._manager.add_provider(pc.name, pc.name, kwargs)
            self._manager.initialize_providers()
        return self._manager

    def _build_provider_order(self) -> List[str]:
        """Build ordered list of enabled providers, default first."""
        enabled = [
            pc.name for pc in self.config.provider_configs
            if pc.enabled
        ]
        # Put default provider first if enabled
        default = self.config.default_provider
        if default in enabled:
            enabled.remove(default)
            enabled.insert(0, default)
        return enabled

    def get_available_providers(self) -> Dict[str, Any]:
        """Get all configured providers and their status."""
        manager = self._get_manager()
        result = {}
        for name in self._provider_order:
            provider = manager.get_provider(name)
            if provider is not None:
                pc = next((p for p in self.config.provider_configs if p.name == name), None)
                result[name] = {
                    "enabled": pc.enabled if pc else True,
                    "models": pc.models if pc else [],
                    "healthy": True,  # TODO: add health check
                }
        return result

    async def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        stream: bool = False,
    ) -> str:
        """
        Generate text from any provider with automatic fallback.

        Args:
            prompt: Input prompt
            model: Model name (provider-specific format)
            provider: Specific provider to use (overrides default)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stream: Whether to stream response

        Returns:
            Generated text
        """
        model = model or self.config.default_model
        providers_to_try = [provider] if provider else list(self._provider_order)

        last_error = None
        for provider_name in providers_to_try:
            manager = self._get_manager()
            prov = manager.get_provider(provider_name)
            if prov is None:
                continue

            try:
                full_response = ""
                async for chunk in prov.generate(
                    prompt, model, max_tokens=max_tokens,
                    temperature=temperature, stream=stream
                ):
                    full_response += chunk
                return full_response
            except Exception as e:
                last_error = e
                if self.config.fallback_enabled:
                    raise
                # Continue to next provider
                continue

        raise RuntimeError(
            f"All providers failed. Last error: {last_error}"
        ) if last_error else RuntimeError("No providers available")

    async def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        provider: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        stream: bool = False,
    ) -> str:
        """
        Chat with any provider with automatic fallback.

        Args:
            messages: Chat messages list [{"role": "user", "content": "..."}]
            model: Model name
            provider: Specific provider to use
            max_tokens: Maximum tokens
            temperature: Sampling temperature
            stream: Whether to stream

        Returns:
            Assistant response
        """
        model = model or self.config.default_model
        providers_to_try = [provider] if provider else list(self._provider_order)

        last_error = None
        for provider_name in providers_to_try:
            manager = self._get_manager()
            prov = manager.get_provider(provider_name)
            if prov is None:
                continue

            try:
                full_response = ""
                async for chunk in prov.chat(
                    messages, model, max_tokens=max_tokens,
                    temperature=temperature, stream=stream
                ):
                    full_response += chunk
                return full_response
            except Exception as e:
                last_error = e
                if self.config.fallback_enabled:
                    raise
                continue

        raise RuntimeError(
            f"All providers failed. Last error: {last_error}"
        ) if last_error else RuntimeError("No providers available")

    def generate_sync(
        self,
        prompt: str,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> str:
        """Synchronous wrapper for generate()."""
        return asyncio.get_event_loop().run_until_complete(
            self.generate(prompt, model, provider, max_tokens, temperature)
        )

    def chat_sync(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        provider: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> str:
        """Synchronous wrapper for chat()."""
        return asyncio.get_event_loop().run_until_complete(
            self.chat(messages, model, provider, max_tokens, temperature)
        )


# Global singleton
_client = None


def get_model_client(config: Optional[ModelClientConfig] = None) -> UnifiedModelClient:
    """Get or create the global unified model client."""
    global _client
    if _client is None:
        _client = UnifiedModelClient(config)
    return _client
