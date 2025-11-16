"""Provider registry and factory for managing AI providers."""

from typing import Dict, Type, Optional, Any, List
import logging

from xencode.crush.providers.base import Provider, ProviderError

logger = logging.getLogger(__name__)


class ProviderRegistry:
    """Registry for AI model providers."""
    
    def __init__(self):
        """Initialize provider registry."""
        self._providers: Dict[str, Type[Provider]] = {}
        self._instances: Dict[str, Provider] = {}
        
        # Register built-in providers
        self._register_builtin_providers()
    
    def _register_builtin_providers(self):
        """Register built-in provider implementations."""
        try:
            from xencode.crush.providers.openai_provider import OpenAIProvider
            self.register("openai", OpenAIProvider)
            logger.debug("Registered OpenAI provider")
        except ImportError as e:
            logger.warning(f"Could not register OpenAI provider: {e}")
        
        try:
            from xencode.crush.providers.anthropic_provider import AnthropicProvider
            self.register("anthropic", AnthropicProvider)
            logger.debug("Registered Anthropic provider")
        except ImportError as e:
            logger.warning(f"Could not register Anthropic provider: {e}")
    
    def register(self, name: str, provider_class: Type[Provider]) -> None:
        """
        Register a provider class.
        
        Args:
            name: Provider name (e.g., "openai", "anthropic")
            provider_class: Provider class to register
            
        Raises:
            ValueError: If provider name is already registered
        """
        if name in self._providers:
            logger.warning(f"Provider '{name}' is already registered, overwriting")
        
        if not issubclass(provider_class, Provider):
            raise ValueError(f"Provider class must inherit from Provider base class")
        
        self._providers[name] = provider_class
        logger.info(f"Registered provider: {name}")
    
    def unregister(self, name: str) -> None:
        """
        Unregister a provider.
        
        Args:
            name: Provider name to unregister
        """
        if name in self._providers:
            del self._providers[name]
            logger.info(f"Unregistered provider: {name}")
        
        if name in self._instances:
            del self._instances[name]
    
    def get_provider_class(self, name: str) -> Type[Provider]:
        """
        Get a registered provider class.
        
        Args:
            name: Provider name
            
        Returns:
            Provider class
            
        Raises:
            ProviderError: If provider is not registered
        """
        if name not in self._providers:
            available = ", ".join(self._providers.keys())
            raise ProviderError(
                f"Provider '{name}' is not registered. Available providers: {available}",
                provider=name,
                error_type="configuration"
            )
        
        return self._providers[name]
    
    def create_provider(
        self,
        name: str,
        model: str,
        api_key: str,
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        timeout: int = 60,
        cache_key: Optional[str] = None,
        **kwargs
    ) -> Provider:
        """
        Create a provider instance.
        
        Args:
            name: Provider name
            model: Model identifier
            api_key: API key for authentication
            base_url: Optional base URL for API
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            timeout: Request timeout in seconds
            cache_key: Optional key for caching instances
            **kwargs: Additional provider-specific parameters
            
        Returns:
            Provider instance
            
        Raises:
            ProviderError: If provider creation fails
        """
        # Check cache if key provided
        if cache_key and cache_key in self._instances:
            logger.debug(f"Returning cached provider instance: {cache_key}")
            return self._instances[cache_key]
        
        # Get provider class
        provider_class = self.get_provider_class(name)
        
        # Create instance
        try:
            provider = provider_class(
                model=model,
                api_key=api_key,
                base_url=base_url,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
                **kwargs
            )
            
            # Validate configuration
            provider.validate_config()
            
            # Cache instance if key provided
            if cache_key:
                self._instances[cache_key] = provider
                logger.debug(f"Cached provider instance: {cache_key}")
            
            logger.info(f"Created provider: {name} (model: {model})")
            return provider
        
        except Exception as e:
            if isinstance(e, ProviderError):
                raise
            raise ProviderError(
                f"Failed to create provider '{name}': {str(e)}",
                provider=name,
                error_type="configuration",
                original_error=e
            )
    
    def list_providers(self) -> List[str]:
        """
        List all registered provider names.
        
        Returns:
            List of provider names
        """
        return list(self._providers.keys())
    
    def is_registered(self, name: str) -> bool:
        """
        Check if a provider is registered.
        
        Args:
            name: Provider name
            
        Returns:
            True if provider is registered
        """
        return name in self._providers
    
    def clear_cache(self, cache_key: Optional[str] = None) -> None:
        """
        Clear cached provider instances.
        
        Args:
            cache_key: Optional specific key to clear, or None to clear all
        """
        if cache_key:
            if cache_key in self._instances:
                del self._instances[cache_key]
                logger.debug(f"Cleared cached provider: {cache_key}")
        else:
            self._instances.clear()
            logger.debug("Cleared all cached providers")


# Global registry instance
_global_registry: Optional[ProviderRegistry] = None


def get_registry() -> ProviderRegistry:
    """
    Get the global provider registry.
    
    Returns:
        Global ProviderRegistry instance
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = ProviderRegistry()
    return _global_registry


def create_provider(
    name: str,
    model: str,
    api_key: str,
    **kwargs
) -> Provider:
    """
    Convenience function to create a provider using the global registry.
    
    Args:
        name: Provider name
        model: Model identifier
        api_key: API key
        **kwargs: Additional provider parameters
        
    Returns:
        Provider instance
    """
    registry = get_registry()
    return registry.create_provider(name, model, api_key, **kwargs)


def register_provider(name: str, provider_class: Type[Provider]) -> None:
    """
    Convenience function to register a provider in the global registry.
    
    Args:
        name: Provider name
        provider_class: Provider class
    """
    registry = get_registry()
    registry.register(name, provider_class)


def list_providers() -> List[str]:
    """
    Convenience function to list providers in the global registry.
    
    Returns:
        List of provider names
    """
    registry = get_registry()
    return registry.list_providers()
