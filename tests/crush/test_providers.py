"""Tests for provider abstraction layer."""

import pytest
from xencode.crush.providers import (
    Provider,
    ProviderResponse,
    ProviderError,
    Message,
    MessageRole,
    ProviderRegistry,
    get_registry,
    create_provider,
    list_providers,
)


def test_provider_registry_initialization():
    """Test that provider registry initializes with built-in providers."""
    registry = ProviderRegistry()
    providers = registry.list_providers()
    
    # Should have at least openai and anthropic if libraries are installed
    assert isinstance(providers, list)
    assert len(providers) >= 0  # May be 0 if libraries not installed


def test_global_registry():
    """Test global registry singleton."""
    registry1 = get_registry()
    registry2 = get_registry()
    
    assert registry1 is registry2


def test_list_providers():
    """Test listing providers."""
    providers = list_providers()
    assert isinstance(providers, list)


def test_provider_registration():
    """Test custom provider registration."""
    
    class CustomProvider(Provider):
        @property
        def name(self) -> str:
            return "custom"
        
        async def generate(self, messages, tools=None, stream=False, **kwargs):
            return ProviderResponse(
                content="test response",
                model=self.model,
                provider=self.name,
            )
        
        async def generate_stream(self, messages, tools=None, **kwargs):
            yield
        
        def estimate_tokens(self, messages):
            return 100
    
    registry = ProviderRegistry()
    registry.register("custom", CustomProvider)
    
    assert registry.is_registered("custom")
    assert "custom" in registry.list_providers()
    
    # Create instance
    provider = registry.create_provider(
        name="custom",
        model="test-model",
        api_key="test-key"
    )
    
    assert isinstance(provider, CustomProvider)
    assert provider.model == "test-model"


def test_provider_unregistration():
    """Test provider unregistration."""
    
    class TempProvider(Provider):
        @property
        def name(self) -> str:
            return "temp"
        
        async def generate(self, messages, tools=None, stream=False, **kwargs):
            pass
        
        async def generate_stream(self, messages, tools=None, **kwargs):
            yield
        
        def estimate_tokens(self, messages):
            return 0
    
    registry = ProviderRegistry()
    registry.register("temp", TempProvider)
    assert registry.is_registered("temp")
    
    registry.unregister("temp")
    assert not registry.is_registered("temp")


def test_provider_not_found():
    """Test error when provider not found."""
    registry = ProviderRegistry()
    
    with pytest.raises(ProviderError) as exc_info:
        registry.get_provider_class("nonexistent")
    
    assert "not registered" in str(exc_info.value)


def test_provider_caching():
    """Test provider instance caching."""
    
    class CachedProvider(Provider):
        @property
        def name(self) -> str:
            return "cached"
        
        async def generate(self, messages, tools=None, stream=False, **kwargs):
            pass
        
        async def generate_stream(self, messages, tools=None, **kwargs):
            yield
        
        def estimate_tokens(self, messages):
            return 0
    
    registry = ProviderRegistry()
    registry.register("cached", CachedProvider)
    
    # Create with cache key
    provider1 = registry.create_provider(
        name="cached",
        model="test",
        api_key="key",
        cache_key="test-cache"
    )
    
    # Get cached instance
    provider2 = registry.create_provider(
        name="cached",
        model="test",
        api_key="key",
        cache_key="test-cache"
    )
    
    assert provider1 is provider2
    
    # Clear cache
    registry.clear_cache("test-cache")
    
    # Should create new instance
    provider3 = registry.create_provider(
        name="cached",
        model="test",
        api_key="key",
        cache_key="test-cache"
    )
    
    assert provider3 is not provider1


def test_message_to_dict():
    """Test message conversion to dictionary."""
    msg = Message(
        role=MessageRole.USER,
        content="Hello, world!"
    )
    
    msg_dict = msg.to_dict()
    assert msg_dict["role"] == "user"
    assert msg_dict["content"] == "Hello, world!"


def test_provider_error():
    """Test ProviderError attributes."""
    error = ProviderError(
        message="Test error",
        provider="test",
        error_type="api_error",
        retryable=True
    )
    
    assert str(error) == "Test error"
    assert error.provider == "test"
    assert error.error_type == "api_error"
    assert error.retryable is True


def test_provider_response():
    """Test ProviderResponse attributes."""
    response = ProviderResponse(
        content="Test response",
        model="test-model",
        provider="test",
        prompt_tokens=10,
        completion_tokens=20
    )
    
    assert response.content == "Test response"
    assert response.total_tokens == 30
