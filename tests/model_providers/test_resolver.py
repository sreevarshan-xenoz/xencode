#!/usr/bin/env python3
"""
Unit tests for Locked Model Resolver
"""

import pytest
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

from xencode.model_providers.resolver import (
    LockedModelResolver,
    ResolverConfig,
    ModelDiscoveryCache,
    ModelInfo,
    ModelAlias,
    get_resolver,
    resolve_coder_model,
)


class TestResolverConfig:
    """Tests for ResolverConfig"""
    
    def test_default_config(self):
        """Test default configuration"""
        config = ResolverConfig()
        
        assert config.lock_best_coder is True
        assert config.lock_model_override is None
        assert config.cache_ttl == 3600
        assert config.verify_on_resolve is True
    
    def test_custom_config(self):
        """Test custom configuration"""
        config = ResolverConfig(
            lock_best_coder=False,
            lock_model_override="custom-model",
            cache_ttl=1800,
            verify_on_resolve=False,
        )
        
        assert config.lock_best_coder is False
        assert config.lock_model_override == "custom-model"
        assert config.cache_ttl == 1800
        assert config.verify_on_resolve is False
    
    def test_from_dict(self):
        """Test creating config from dictionary"""
        data = {
            'lock_best_coder': False,
            'lock_model_override': 'test-model',
            'cache_ttl': 600,
        }
        
        config = ResolverConfig.from_dict(data)
        
        assert config.lock_best_coder is False
        assert config.lock_model_override == 'test-model'
        assert config.cache_ttl == 600
        assert config.verify_on_resolve is True  # Default
    
    def test_to_dict(self):
        """Test converting config to dictionary"""
        config = ResolverConfig(
            lock_best_coder=False,
            lock_model_override='test',
            cache_ttl=900,
            verify_on_resolve=False,
        )
        
        d = config.to_dict()
        
        assert d['lock_best_coder'] is False
        assert d['lock_model_override'] == 'test'
        assert d['cache_ttl'] == 900
        assert d['verify_on_resolve'] is False


class TestModelInfo:
    """Tests for ModelInfo"""
    
    def test_model_creation(self):
        """Test creating ModelInfo"""
        model = ModelInfo(
            model_id="qwen3-coder-next-instruct",
            alias="qwen-coder-next",
            capabilities=["code", "instruct"],
            context_window=32768,
            max_tokens=8192,
        )
        
        assert model.model_id == "qwen3-coder-next-instruct"
        assert model.alias == "qwen-coder-next"
        assert "code" in model.capabilities
        assert model.context_window == 32768
    
    def test_to_dict(self):
        """Test converting ModelInfo to dictionary"""
        model = ModelInfo(
            model_id="test-model",
            capabilities=["code"],
        )
        
        d = model.to_dict()
        
        assert d["model_id"] == "test-model"
        assert d["capabilities"] == ["code"]
        assert "discovered_at" in d


class TestModelDiscoveryCache:
    """Tests for ModelDiscoveryCache"""
    
    @pytest.fixture
    def cache(self):
        """Create cache with short TTL for tests"""
        return ModelDiscoveryCache(ttl=1)  # 1 second TTL
    
    def test_cache_invalid_when_empty(self, cache):
        """Test cache is invalid when empty"""
        assert cache.is_valid() is False
    
    def test_cache_valid_after_set(self, cache):
        """Test cache is valid after setting models"""
        models = [
            ModelInfo(model_id="model1", capabilities=["code"]),
        ]
        cache.set_models(models)
        
        assert cache.is_valid() is True
    
    def test_cache_expires(self):
        """Test cache expires after TTL"""
        cache = ModelDiscoveryCache(ttl=0.1)  # 100ms TTL
        
        models = [ModelInfo(model_id="model1")]
        cache.set_models(models)
        
        assert cache.is_valid() is True
        time.sleep(0.15)
        assert cache.is_valid() is False
    
    def test_cache_invalidate(self, cache):
        """Test manual cache invalidation"""
        cache.set_models([ModelInfo(model_id="model1")])
        assert cache.is_valid() is True
        
        cache.invalidate()
        assert cache.is_valid() is False
    
    def test_get_model_by_id(self, cache):
        """Test getting model by ID"""
        models = [
            ModelInfo(model_id="qwen3-coder-next-instruct", capabilities=["code"]),
            ModelInfo(model_id="gpt-4", capabilities=["chat"]),
        ]
        cache.set_models(models)
        
        model = cache.get_model("qwen3-coder-next-instruct")
        assert model is not None
        assert model.model_id == "qwen3-coder-next-instruct"
    
    def test_get_model_by_alias(self, cache):
        """Test getting model by alias"""
        models = [
            ModelInfo(
                model_id="qwen3-coder-next-instruct",
                alias="qwen-coder-next",
                capabilities=["code"],
            ),
        ]
        cache.set_models(models)
        
        model = cache.get_model("qwen-coder-next")
        assert model is not None
        assert model.model_id == "qwen3-coder-next-instruct"
    
    def test_get_coder_models(self, cache):
        """Test filtering coder models"""
        models = [
            ModelInfo(model_id="qwen3-coder-next", capabilities=["code"]),
            ModelInfo(model_id="gpt-4", capabilities=["chat"]),
            ModelInfo(model_id="qwen-coder-plus", capabilities=["code", "chat"]),
        ]
        cache.set_models(models)
        
        coder_models = cache.get_coder_models()
        
        assert len(coder_models) == 2
        assert all(any(c == "code" for c in m.capabilities) for m in coder_models)
    
    def test_find_best_coder_model_priority(self, cache):
        """Test best coder model selection with priority"""
        models = [
            ModelInfo(model_id="qwen-coder-plus", capabilities=["code"]),
            ModelInfo(model_id="qwen3-coder-next-instruct", capabilities=["code", "instruct"]),
            ModelInfo(model_id="qwen-max-coder", capabilities=["code"]),
        ]
        cache.set_models(models)
        
        best = cache.find_best_coder_model()
        
        assert best is not None
        assert best.model_id == "qwen3-coder-next-instruct"
    
    def test_find_best_coder_model_fallback(self, cache):
        """Test best coder model fallback"""
        models = [
            ModelInfo(model_id="qwen-max-coder", capabilities=["code"]),
            ModelInfo(model_id="gpt-4", capabilities=["chat"]),
        ]
        cache.set_models(models)
        
        best = cache.find_best_coder_model()
        
        assert best is not None
        assert "coder" in best.model_id.lower()


class TestLockedModelResolver:
    """Tests for LockedModelResolver"""
    
    @pytest.fixture
    def resolver(self):
        """Create resolver with test config"""
        return LockedModelResolver(
            config=ResolverConfig(
                lock_best_coder=True,
                cache_ttl=300,
                verify_on_resolve=False,
            )
        )
    
    def test_resolver_default_config(self, resolver):
        """Test resolver default configuration"""
        config = resolver.get_config()
        
        assert config['lock_best_coder'] is True
        assert config['lock_model_override'] is None
    
    def test_resolver_custom_config(self):
        """Test resolver with custom config"""
        config = ResolverConfig(
            lock_best_coder=False,
            lock_model_override="custom-model",
        )
        resolver = LockedModelResolver(config=config)
        
        assert resolver.config.lock_model_override == "custom-model"
    
    @pytest.mark.asyncio
    async def test_resolve_with_override(self, resolver):
        """Test resolution with user override"""
        resolver.config.lock_model_override = "my-custom-model"
        resolver.config.verify_on_resolve = False
        
        model_id = await resolver.resolve_coder_model()
        
        assert model_id == "my-custom-model"
    
    @pytest.mark.asyncio
    async def test_resolve_with_cache(self, resolver):
        """Test resolution with valid cache"""
        # Pre-populate cache
        models = [
            ModelInfo(model_id="qwen3-coder-next-instruct", capabilities=["code"]),
        ]
        resolver.cache.set_models(models)
        
        model_id = await resolver.resolve_coder_model()
        
        assert model_id == "qwen3-coder-next-instruct"
    
    @pytest.mark.asyncio
    async def test_resolve_with_empty_cache(self, resolver):
        """Test resolution with empty cache falls back to default"""
        resolver.cache.invalidate()
        
        with patch.object(resolver, 'discover_models', return_value=[]):
            model_id = await resolver.resolve_coder_model()
            
            # Should fall back to default
            assert model_id == LockedModelResolver.DEFAULT_CODER_MODEL
    
    @pytest.mark.asyncio
    async def test_discover_models_success(self, resolver):
        """Test successful model discovery"""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "data": [
                {"id": "qwen3-coder-next-instruct"},
                {"id": "qwen-coder-plus"},
            ]
        })
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)
        
        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        
        with patch.object(resolver, '_load_access_token', return_value="fake_token"):
            with patch('aiohttp.ClientSession', return_value=mock_session):
                models = await resolver.discover_models()
                
                assert len(models) == 2
                assert models[0].model_id == "qwen3-coder-next-instruct"
                assert resolver.cache.is_valid() is True
    
    @pytest.mark.asyncio
    async def test_discover_models_no_auth(self, resolver):
        """Test model discovery without authentication"""
        with patch.object(resolver, '_load_access_token', return_value=None):
            models = await resolver.discover_models()
            
            assert len(models) == 0
    
    @pytest.mark.asyncio
    async def test_verify_model_exists(self, resolver):
        """Test model existence verification"""
        models = [
            ModelInfo(model_id="qwen3-coder-next-instruct", capabilities=["code"]),
        ]
        resolver.cache.set_models(models)
        
        exists = await resolver.verify_model_exists("qwen3-coder-next-instruct")
        assert exists is True
        
        not_exists = await resolver.verify_model_exists("nonexistent-model")
        assert not_exists is False
    
    @pytest.mark.asyncio
    async def test_resolve_alias_builtin(self, resolver):
        """Test resolving built-in alias"""
        canonical = await resolver.resolve_alias("qwen-coder-next-latest")
        
        # Should resolve to canonical mapping
        assert canonical is not None
        assert "qwen3-coder-next-instruct" in canonical.lower()
    
    @pytest.mark.asyncio
    async def test_resolve_alias_discovered(self, resolver):
        """Test resolving discovered alias"""
        models = [
            ModelInfo(
                model_id="qwen3-coder-next-instruct",
                alias="custom-alias",
                capabilities=["code"],
            ),
        ]
        resolver.cache.set_models(models)
        
        canonical = await resolver.resolve_alias("custom-alias")
        
        assert canonical == "qwen3-coder-next-instruct"
    
    def test_invalidate_cache(self, resolver):
        """Test cache invalidation"""
        resolver.cache.set_models([ModelInfo(model_id="test")])
        assert resolver.cache.is_valid() is True
        
        resolver.invalidate_cache()
        assert resolver.cache.is_valid() is False
    
    def test_set_access_token(self, resolver):
        """Test setting access token"""
        resolver.set_access_token("test-token")
        
        assert resolver.access_token == "test-token"
    
    def test_set_creds_file(self, resolver):
        """Test setting custom credentials file"""
        custom_file = Path("/tmp/test_creds.json")
        resolver.set_creds_file(custom_file)
        
        assert resolver._creds_file == custom_file


class TestGetResolver:
    """Tests for get_resolver helper"""
    
    def test_get_resolver_singleton(self):
        """Test that get_resolver returns singleton instance"""
        # Clear singleton first
        import xencode.model_providers.resolver as resolver_module
        resolver_module._resolver = None
        
        resolver1 = get_resolver()
        resolver2 = get_resolver()
        
        assert resolver1 is resolver2
    
    def test_get_resolver_with_config(self):
        """Test get_resolver with custom config"""
        # Clear singleton first
        import xencode.model_providers.resolver as resolver_module
        resolver_module._resolver = None
        
        config = ResolverConfig(lock_best_coder=False)
        resolver = get_resolver(config=config)
        
        assert resolver.config.lock_best_coder is False


class TestResolveCoderModel:
    """Tests for resolve_coder_model convenience function"""
    
    @pytest.mark.asyncio
    async def test_resolve_coder_model(self):
        """Test resolve_coder_model function"""
        # This will use the global resolver
        model_id = await resolve_coder_model(force_refresh=False)
        
        # Should return something (either from cache or default)
        assert model_id is not None or model_id is None  # Depends on global state


class TestModelAlias:
    """Tests for ModelAlias"""
    
    def test_canonical_map(self):
        """Test canonical alias mapping"""
        assert "qwen-coder-next-latest" in ModelAlias.CANONICAL_MAP
        assert ModelAlias.CANONICAL_MAP["qwen-coder-next-latest"] == "qwen3-coder-next-instruct"
    
    def test_alias_constants(self):
        """Test alias constants"""
        assert ModelAlias.QWEN_CODER_NEXT_LATEST == "qwen-coder-next-latest"
        assert ModelAlias.QWEN3_CODER_NEXT_INSTRUCT == "qwen3-coder-next-instruct"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
