#!/usr/bin/env python3
"""
Unit tests for Provider Diagnostics Service
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import aiohttp

from xencode.model_providers.diagnostics import (
    ProviderDiagnostics,
    ProviderTestResult,
    ProviderStatus,
    get_diagnostics,
)


class TestProviderTestResult:
    """Tests for ProviderTestResult dataclass"""
    
    def test_success_result(self):
        """Test successful test result"""
        result = ProviderTestResult(
            provider="TestProvider",
            status=ProviderStatus.OK,
            latency_ms=150.5,
            endpoint="http://test.com",
            model_count=10,
        )
        
        assert result.is_success is True
        assert result.provider == "TestProvider"
        assert result.latency_ms == 150.5
    
    def test_error_result(self):
        """Test error test result"""
        result = ProviderTestResult(
            provider="TestProvider",
            status=ProviderStatus.AUTH_ERROR,
            error_message="Invalid API key",
            remediation="Check your API key",
        )
        
        assert result.is_success is False
        assert result.error_message == "Invalid API key"
        assert result.remediation == "Check your API key"
    
    def test_to_dict(self):
        """Test converting result to dictionary"""
        result = ProviderTestResult(
            provider="Qwen",
            status=ProviderStatus.OK,
            latency_ms=200.0,
            model_count=5,
        )
        
        d = result.to_dict()
        
        assert d["provider"] == "Qwen"
        assert d["status"] == "ok"
        assert d["latency_ms"] == 200.0
        assert d["model_count"] == 5
        assert d["is_success"] is True


class TestProviderStatus:
    """Tests for ProviderStatus enum"""
    
    def test_status_values(self):
        """Test status enum values"""
        assert ProviderStatus.OK.value == "ok"
        assert ProviderStatus.AUTH_ERROR.value == "auth_error"
        assert ProviderStatus.CONNECTION_ERROR.value == "connection_error"
        assert ProviderStatus.NOT_CONFIGURED.value == "not_configured"
        assert ProviderStatus.UNKNOWN.value == "unknown"


class TestProviderDiagnostics:
    """Tests for ProviderDiagnostics"""
    
    @pytest.fixture
    def diagnostics(self):
        """Create diagnostics instance"""
        return ProviderDiagnostics()
    
    @pytest.mark.asyncio
    async def test_qwen_not_configured(self, diagnostics):
        """Test Qwen when not configured"""
        with patch.object(diagnostics, '_load_qwen_token', return_value=None):
            result = await diagnostics.test_qwen()
            
            assert result.provider == "Qwen"
            assert result.status == ProviderStatus.NOT_CONFIGURED
            assert result.is_success is False
            assert "No Qwen authentication" in result.error_message
            assert "Login" in result.remediation
    
    @pytest.mark.asyncio
    async def test_qwen_success(self, diagnostics):
        """Test successful Qwen connection"""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"data": [{"id": "model1"}, {"id": "model2"}]})
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)
        
        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        
        with patch.object(diagnostics, '_load_qwen_token', return_value="fake_token"):
            with patch('aiohttp.ClientSession', return_value=mock_session):
                result = await diagnostics.test_qwen()
                
                assert result.provider == "Qwen"
                assert result.status == ProviderStatus.OK
                assert result.is_success is True
                assert result.latency_ms is not None
                assert result.model_count == 2
    
    @pytest.mark.asyncio
    async def test_qwen_auth_error(self, diagnostics):
        """Test Qwen with expired/invalid token"""
        mock_response = AsyncMock()
        mock_response.status = 401
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)
        
        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        
        with patch.object(diagnostics, '_load_qwen_token', return_value="expired_token"):
            with patch('aiohttp.ClientSession', return_value=mock_session):
                result = await diagnostics.test_qwen()
                
                assert result.provider == "Qwen"
                assert result.status == ProviderStatus.AUTH_ERROR
                assert result.is_success is False
                assert "Authentication failed" in result.error_message
    
    @pytest.mark.asyncio
    async def test_openrouter_not_configured(self, diagnostics):
        """Test OpenRouter when not configured"""
        result = await diagnostics.test_openrouter(None)
        
        assert result.provider == "OpenRouter"
        assert result.status == ProviderStatus.NOT_CONFIGURED
        assert result.is_success is False
        assert "No API key" in result.error_message
    
    @pytest.mark.asyncio
    async def test_openrouter_invalid_key_format(self, diagnostics):
        """Test OpenRouter with invalid key format"""
        result = await diagnostics.test_openrouter("invalid-key")
        
        assert result.provider == "OpenRouter"
        assert result.status == ProviderStatus.NOT_CONFIGURED
        assert "Invalid API key format" in result.error_message
    
    @pytest.mark.asyncio
    async def test_openrouter_success(self, diagnostics):
        """Test successful OpenRouter connection"""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"data": [{"id": "model1"}]})
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)
        
        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        
        with patch('aiohttp.ClientSession', return_value=mock_session):
            result = await diagnostics.test_openrouter("sk-or-v1-fake-key")
            
            assert result.provider == "OpenRouter"
            assert result.status == ProviderStatus.OK
            assert result.is_success is True
            assert result.latency_ms is not None
    
    @pytest.mark.asyncio
    async def test_openrouter_auth_error(self, diagnostics):
        """Test OpenRouter with invalid API key"""
        mock_response = AsyncMock()
        mock_response.status = 401
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)
        
        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        
        with patch('aiohttp.ClientSession', return_value=mock_session):
            result = await diagnostics.test_openrouter("sk-or-v1-invalid-key")
            
            assert result.provider == "OpenRouter"
            assert result.status == ProviderStatus.AUTH_ERROR
            assert result.is_success is False
            assert "Invalid API key" in result.error_message
    
    @pytest.mark.asyncio
    async def test_ollama_success(self, diagnostics):
        """Test successful Ollama connection"""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"models": [{"name": "llama3"}, {"name": "mistral"}]})
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)
        
        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        
        with patch('aiohttp.ClientSession', return_value=mock_session):
            result = await diagnostics.test_ollama()
            
            assert result.provider == "Ollama"
            assert result.status == ProviderStatus.OK
            assert result.is_success is True
            assert result.latency_ms is not None
            assert result.model_count == 2
    
    @pytest.mark.asyncio
    async def test_ollama_connection_error(self, diagnostics):
        """Test Ollama when service not running"""
        mock_session = AsyncMock()
        mock_session.get = MagicMock(side_effect=aiohttp.ClientConnectionError())
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        
        with patch('aiohttp.ClientSession', return_value=mock_session):
            result = await diagnostics.test_ollama()
            
            assert result.provider == "Ollama"
            assert result.status == ProviderStatus.CONNECTION_ERROR
            assert result.is_success is False
            assert "Cannot connect" in result.error_message
            assert "ollama serve" in result.remediation
    
    @pytest.mark.asyncio
    async def test_ollama_timeout(self, diagnostics):
        """Test Ollama timeout"""
        mock_session = AsyncMock()
        mock_session.get = MagicMock(side_effect=asyncio.TimeoutError())
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        
        with patch('aiohttp.ClientSession', return_value=mock_session):
            result = await diagnostics.test_ollama()
            
            assert result.provider == "Ollama"
            assert result.status == ProviderStatus.CONNECTION_ERROR
            assert result.is_success is False
            assert "timed out" in result.error_message.lower()
    
    @pytest.mark.asyncio
    async def test_test_all_providers(self, diagnostics):
        """Test testing all providers concurrently"""
        # Mock all three providers
        with (
            patch.object(diagnostics, 'test_qwen') as mock_qwen,
            patch.object(diagnostics, 'test_openrouter') as mock_openrouter,
            patch.object(diagnostics, 'test_ollama') as mock_ollama,
        ):
            mock_qwen.return_value = ProviderTestResult(
                provider="Qwen",
                status=ProviderStatus.OK,
                latency_ms=100.0,
            )
            mock_openrouter.return_value = ProviderTestResult(
                provider="OpenRouter",
                status=ProviderStatus.NOT_CONFIGURED,
                error_message="No API key",
            )
            mock_ollama.return_value = ProviderTestResult(
                provider="Ollama",
                status=ProviderStatus.OK,
                latency_ms=50.0,
                model_count=3,
            )
            
            results = await diagnostics.test_all_providers()
            
            assert len(results) == 3
            assert results[0].provider == "Qwen"
            assert results[1].provider == "OpenRouter"
            assert results[2].provider == "Ollama"
            
            # Verify all were called concurrently
            mock_qwen.assert_called_once()
            mock_openrouter.assert_called_once()
            mock_ollama.assert_called_once()


class TestGetDiagnostics:
    """Tests for get_diagnostics helper"""
    
    def test_get_diagnostics_singleton(self):
        """Test that get_diagnostics returns singleton instance"""
        diag1 = get_diagnostics()
        diag2 = get_diagnostics()
        
        assert diag1 is diag2
    
    def test_diagnostics_instance(self):
        """Test that diagnostics instance is created correctly"""
        diag = get_diagnostics()
        
        assert isinstance(diag, ProviderDiagnostics)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
