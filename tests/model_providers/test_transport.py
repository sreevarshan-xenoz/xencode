#!/usr/bin/env python3
"""
Unit tests for Provider Transport Layer
"""

import pytest
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch
from types import AsyncGeneratorType
import aiohttp

from xencode.model_providers.transport import (
    ProviderTransport,
    ProviderTransportPolicy,
    TransportEvent,
    TransportEventType,
    UnifiedErrorEnvelope,
    TransportError,
    RetryableErrorCode,
)


class AsyncContextManagerMock:
    """Mock for async context managers (async with)"""
    def __init__(self, return_value=None):
        self.return_value = return_value
        self.enter_count = 0
        
    async def __aenter__(self):
        self.enter_count += 1
        return self.return_value
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


class TestProviderTransportPolicy:
    """Tests for ProviderTransportPolicy"""
    
    def test_default_policy(self):
        """Test default policy values"""
        policy = ProviderTransportPolicy()
        
        assert policy.timeout == 30.0
        assert policy.max_retries == 3
        assert policy.backoff_base == 1.0
        assert policy.backoff_max == 60.0
        assert 429 in policy.retryable_status_codes
        assert 500 in policy.retryable_status_codes
        assert 502 in policy.retryable_status_codes
        assert 503 in policy.retryable_status_codes
        assert 504 in policy.retryable_status_codes
    
    def test_is_retryable_status(self):
        """Test retryable status code detection"""
        policy = ProviderTransportPolicy()
        
        assert policy.is_retryable_status(429) is True
        assert policy.is_retryable_status(500) is True
        assert policy.is_retryable_status(503) is True
        assert policy.is_retryable_status(400) is False
        assert policy.is_retryable_status(401) is False
        assert policy.is_retryable_status(404) is False
    
    def test_is_retryable_exception(self):
        """Test retryable exception detection"""
        policy = ProviderTransportPolicy()
        
        # Should be retryable
        assert policy.is_retryable_exception(aiohttp.ClientConnectionError()) is True
        assert policy.is_retryable_exception(aiohttp.ClientTimeout()) is True
        assert policy.is_retryable_exception(asyncio.TimeoutError()) is True
        
        # Should not be retryable
        assert policy.is_retryable_exception(ValueError()) is False
        assert policy.is_retryable_exception(KeyError()) is False
    
    def test_exponential_backoff(self):
        """Test exponential backoff delay calculation"""
        policy = ProviderTransportPolicy(backoff_base=1.0, backoff_max=60.0)
        
        # Should double each attempt
        assert policy.get_delay(1) == 1.0
        assert policy.get_delay(2) == 2.0
        assert policy.get_delay(3) == 4.0
        assert policy.get_delay(4) == 8.0
        
        # Should cap at max
        assert policy.get_delay(10) == 60.0
    
    def test_custom_policy(self):
        """Test custom policy configuration"""
        policy = ProviderTransportPolicy(
            timeout=60.0,
            max_retries=5,
            backoff_base=2.0,
            backoff_max=120.0,
        )
        
        assert policy.timeout == 60.0
        assert policy.max_retries == 5
        assert policy.get_delay(1) == 2.0
        assert policy.get_delay(2) == 4.0


class TestUnifiedErrorEnvelope:
    """Tests for UnifiedErrorEnvelope"""
    
    def test_default_envelope(self):
        """Test default error envelope"""
        envelope = UnifiedErrorEnvelope()
        
        assert envelope.success is False
        assert envelope.error_type == "unknown"
        assert envelope.message == "An unexpected error occurred"
        assert envelope.is_retryable is False
    
    def test_envelope_to_exception(self):
        """Test converting envelope to exception"""
        envelope = UnifiedErrorEnvelope(
            error_type="timeout",
            error_code="TIMEOUT",
            http_status=504,
            message="Request timed out",
            user_message="Please try again",
            is_retryable=True,
            provider="test",
            endpoint="http://test.com/api",
        )
        
        exc = envelope.to_exception()
        
        assert isinstance(exc, TransportError)
        assert exc.error_type == "timeout"
        assert exc.error_code == "TIMEOUT"
        assert exc.http_status == 504
        assert exc.message == "Request timed out"
        assert exc.user_message == "Please try again"
        assert exc.is_retryable is True
        assert exc.provider == "test"
    
    def test_retry_after_envelope(self):
        """Test envelope with retry-after"""
        envelope = UnifiedErrorEnvelope(
            error_type="rate_limit",
            error_code="429",
            http_status=429,
            message="Rate limit exceeded",
            retry_after=30.0,
            is_retryable=True,
        )
        
        assert envelope.retry_after == 30.0
        assert envelope.is_retryable is True


class TestTransportError:
    """Tests for TransportError"""
    
    def test_error_creation(self):
        """Test creating TransportError"""
        exc = TransportError(
            error_type="http_error",
            http_status=500,
            message="Internal server error",
            provider="openai",
        )
        
        assert exc.error_type == "http_error"
        assert exc.http_status == 500
        assert exc.message == "Internal server error"
        assert exc.provider == "openai"
    
    def test_error_to_dict(self):
        """Test converting error to dictionary"""
        exc = TransportError(
            error_type="timeout",
            error_code="TIMEOUT",
            http_status=504,
            message="Timeout",
            user_message="Please retry",
            is_retryable=True,
            provider="qwen",
            endpoint="http://qwen.ai/api",
        )
        
        d = exc.to_dict()
        
        assert d["error_type"] == "timeout"
        assert d["error_code"] == "TIMEOUT"
        assert d["http_status"] == 504
        assert d["is_retryable"] is True
        assert d["provider"] == "qwen"


class TestProviderTransport:
    """Tests for ProviderTransport"""
    
    @pytest.fixture
    def transport(self):
        """Create test transport"""
        return ProviderTransport(
            provider="test",
            base_url="http://test.com/api",
            policy=ProviderTransportPolicy(
                timeout=5.0,
                max_retries=3,
                backoff_base=0.1,  # Fast for tests
                backoff_max=1.0,
            ),
        )
    
    @pytest.fixture
    def transport_with_events(self):
        """Create transport with event callback"""
        events = []
        
        def callback(event):
            events.append(event)
        
        transport = ProviderTransport(
            provider="test",
            base_url="http://test.com/api",
            policy=ProviderTransportPolicy(max_retries=2, backoff_base=0.1),
            event_callback=callback,
        )
        transport._events = events
        return transport
    
    @pytest.mark.asyncio
    async def test_successful_request(self, transport):
        """Test successful HTTP request"""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"result": "success"})
        
        mock_request_cm = AsyncContextManagerMock(mock_response)
        
        with patch.object(transport, 'session') as mock_session_ctx:
            mock_session = MagicMock()
            mock_session.request = MagicMock(return_value=mock_request_cm)
            mock_session_ctx.return_value.__aenter__.return_value = mock_session
            
            async with transport.session() as session:
                result = await transport.request(session, "GET", "/test")
            
            assert result == {"result": "success"}
    
    @pytest.mark.asyncio
    async def test_retry_on_429(self, transport):
        """Test retry on 429 Too Many Requests"""
        mock_response_429 = MagicMock()
        mock_response_429.status = 429
        mock_response_429.text = AsyncMock(return_value="Rate limited")
        
        mock_response_200 = MagicMock()
        mock_response_200.status = 200
        mock_response_200.json = AsyncMock(return_value={"result": "success"})
        
        call_count = 0
        
        def make_request(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return AsyncContextManagerMock(mock_response_429)
            return AsyncContextManagerMock(mock_response_200)
        
        with patch.object(transport, 'session') as mock_session_ctx:
            mock_session = MagicMock()
            mock_session.request = make_request
            mock_session_ctx.return_value.__aenter__.return_value = mock_session
            
            async with transport.session() as session:
                result = await transport.request(session, "GET", "/test")
            
            assert result == {"result": "success"}
            assert call_count == 2  # First failed, second succeeded
    
    @pytest.mark.asyncio
    async def test_retry_on_connection_error(self, transport):
        """Test retry on connection error"""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"result": "success"})
        
        call_count = 0
        
        def make_request(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise aiohttp.ClientConnectionError("Connection lost")
            return AsyncContextManagerMock(mock_response)
        
        with patch.object(transport, 'session') as mock_session_ctx:
            mock_session = MagicMock()
            mock_session.request = make_request
            mock_session_ctx.return_value.__aenter__.return_value = mock_session
            
            async with transport.session() as session:
                result = await transport.request(session, "GET", "/test")
            
            assert result == {"result": "success"}
            assert call_count == 2
    
    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self, transport):
        """Test failure after max retries"""
        def make_request(*args, **kwargs):
            raise aiohttp.ClientConnectionError("Connection failed")
        
        with patch.object(transport, 'session') as mock_session_ctx:
            mock_session = MagicMock()
            mock_session.request = make_request
            mock_session_ctx.return_value.__aenter__.return_value = mock_session
            
            with pytest.raises(TransportError) as exc_info:
                async with transport.session() as session:
                    await transport.request(session, "GET", "/test")
            
            # Error should be caught and converted
            assert exc_info.value.message is not None
    
    @pytest.mark.asyncio
    async def test_non_retryable_status(self, transport):
        """Test non-retryable status codes"""
        mock_response = MagicMock()
        mock_response.status = 401
        mock_response.text = AsyncMock(return_value="Unauthorized")
        
        mock_request_cm = AsyncContextManagerMock(mock_response)
        
        with patch.object(transport, 'session') as mock_session_ctx:
            mock_session = MagicMock()
            mock_session.request = MagicMock(return_value=mock_request_cm)
            mock_session_ctx.return_value.__aenter__.return_value = mock_session
            
            with pytest.raises(TransportError) as exc_info:
                async with transport.session() as session:
                    await transport.request(session, "GET", "/test")
            
            assert exc_info.value.http_status == 401
            assert exc_info.value.is_retryable is False
    
    @pytest.mark.asyncio
    async def test_event_emission(self, transport_with_events):
        """Test telemetry event emission"""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={})
        
        mock_request_cm = AsyncContextManagerMock(mock_response)
        
        with patch.object(transport_with_events, 'session') as mock_session_ctx:
            mock_session = MagicMock()
            mock_session.request = MagicMock(return_value=mock_request_cm)
            mock_session_ctx.return_value.__aenter__.return_value = mock_session
            
            async with transport_with_events.session() as session:
                await transport_with_events.request(session, "GET", "/test")
        
        # Should have at least REQUEST_START and REQUEST_SUCCESS events
        event_types = [e.event_type for e in transport_with_events._events]
        assert TransportEventType.REQUEST_START in event_types
        assert TransportEventType.REQUEST_SUCCESS in event_types
    
    @pytest.mark.asyncio
    async def test_user_friendly_messages(self, transport):
        """Test user-friendly error messages"""
        # Test 429 message
        msg = transport._get_user_message(429)
        assert "Rate limit" in msg or "wait" in msg.lower()
        
        # Test 500 message
        msg = transport._get_user_message(500)
        assert "try again later" in msg.lower()
        
        # Test unknown message
        msg = transport._get_user_message(999)
        assert "unexpected" in msg.lower() or "try again" in msg.lower()
    
    @pytest.mark.asyncio
    async def test_timeout_envelope(self, transport):
        """Test timeout exception to envelope conversion"""
        exc = aiohttp.ClientTimeout(total=5.0)
        envelope = transport._exception_to_envelope(exc, "http://test.com/api")
        
        assert envelope.error_type == "timeout"
        assert envelope.is_retryable is True
        assert "timed out" in envelope.message.lower()
    
    @pytest.mark.asyncio
    async def test_connection_error_envelope(self, transport):
        """Test connection error to envelope conversion"""
        exc = aiohttp.ClientConnectionError("Connection refused")
        envelope = transport._exception_to_envelope(exc, "http://test.com/api")
        
        assert envelope.error_type == "connection_error"
        assert envelope.is_retryable is True
        assert "connection" in envelope.message.lower()


class TestTransportEvent:
    """Tests for TransportEvent"""
    
    def test_event_creation(self):
        """Test creating transport event"""
        event = TransportEvent(
            event_type=TransportEventType.REQUEST_SUCCESS,
            timestamp=time.time(),
            provider="openai",
            endpoint="http://api.openai.com/v1/chat",
            attempt=1,
            status_code=200,
            latency_ms=150.5,
        )
        
        assert event.event_type == TransportEventType.REQUEST_SUCCESS
        assert event.provider == "openai"
        assert event.latency_ms == 150.5
    
    def test_event_to_dict(self):
        """Test converting event to dictionary"""
        event = TransportEvent(
            event_type=TransportEventType.REQUEST_FAILURE,
            timestamp=1234567890.0,
            provider="qwen",
            endpoint="http://qwen.ai/api",
            attempt=3,
            error_message="Timeout",
            latency_ms=5000.0,
        )
        
        d = event.to_dict()
        
        assert d["event_type"] == "request_failure"
        assert d["provider"] == "qwen"
        assert d["attempt"] == 3
        assert d["error_message"] == "Timeout"


class TestIntegration:
    """Integration tests for transport layer"""
    
    @pytest.mark.asyncio
    async def test_retry_chain_with_recovery(self):
        """Test full retry chain with eventual success"""
        transport = ProviderTransport(
            provider="test",
            base_url="http://test.com",
            policy=ProviderTransportPolicy(
                max_retries=3,
                backoff_base=0.01,  # Very fast for tests
            ),
        )
        
        # Use same exception type for simplicity
        call_sequence = [
            aiohttp.ClientConnectionError("Error 1"),
            aiohttp.ClientConnectionError("Error 2"),
            {"status": "success"},  # Third call succeeds
        ]
        
        call_index = 0
        
        def make_request(*args, **kwargs):
            nonlocal call_index
            result = call_sequence[call_index]
            call_index += 1
            
            if isinstance(result, dict):
                mock_response = MagicMock()
                mock_response.status = 200
                mock_response.json = AsyncMock(return_value=result)
                return AsyncContextManagerMock(mock_response)
            else:
                # Raise exception instance
                raise result
        
        with patch.object(transport, 'session') as mock_session_ctx:
            mock_session = MagicMock()
            mock_session.request = make_request
            mock_session_ctx.return_value.__aenter__.return_value = mock_session
            
            async with transport.session() as session:
                result = await transport.request(session, "GET", "/api")
            
            assert result == {"status": "success"}
            assert call_index == 3  # All three calls were made
    
    @pytest.mark.asyncio
    async def test_streaming_request(self):
        """Test streaming request"""
        transport = ProviderTransport(
            provider="test",
            base_url="http://test.com",
            policy=ProviderTransportPolicy(max_retries=2, backoff_base=0.01),
        )
        
        # Mock streaming response content
        class MockContent:
            def __init__(self):
                self.chunks = [b'data: {"chunk": 1}\n', b'data: {"chunk": 2}\n', b'data: {"chunk": 3}\n']
                self.index = 0
            
            def __aiter__(self):
                return self
            
            async def __anext__(self):
                if self.index < len(self.chunks):
                    chunk = self.chunks[self.index]
                    self.index += 1
                    return chunk
                raise StopAsyncIteration
        
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.content = MockContent()
        
        mock_request_cm = AsyncContextManagerMock(mock_response)
        
        with patch.object(transport, 'session') as mock_session_ctx:
            mock_session = MagicMock()
            mock_session.request = MagicMock(return_value=mock_request_cm)
            mock_session_ctx.return_value.__aenter__.return_value = mock_session
            
            chunks = []
            async with transport.session() as session:
                async for chunk in transport.stream_request(session, "GET", "/stream"):
                    chunks.append(chunk)
            
            assert len(chunks) == 3
            assert '{"chunk": 1}' in chunks[0]
            assert '{"chunk": 2}' in chunks[1]
            assert '{"chunk": 3}' in chunks[2]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
