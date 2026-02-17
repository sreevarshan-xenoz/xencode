#!/usr/bin/env python3
"""
Provider Transport Layer with Reliability Features

Implements resilient transport for cloud providers with:
- Retry logic with exponential backoff
- Timeout policies
- Unified error envelope
- Stream interruption detection
- Safe reconnect hooks
- Structured telemetry events
"""

import asyncio
import json
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, AsyncIterator, Union
import aiohttp
from rich.console import Console

console = Console()


class RetryableErrorCode(Enum):
    """HTTP status codes that are retryable"""
    TOO_MANY_REQUESTS = 429
    INTERNAL_SERVER_ERROR = 500
    BAD_GATEWAY = 502
    SERVICE_UNAVAILABLE = 503
    GATEWAY_TIMEOUT = 504


class TransportEventType(Enum):
    """Transport telemetry event types"""
    REQUEST_START = "request_start"
    REQUEST_SUCCESS = "request_success"
    REQUEST_RETRY = "request_retry"
    REQUEST_FAILURE = "request_failure"
    STREAM_INTERRUPTED = "stream_interrupted"
    STREAM_RECONNECTED = "stream_reconnected"
    TIMEOUT_OCCURRED = "timeout_occurred"


@dataclass
class ProviderTransportPolicy:
    """
    Configuration for provider transport behavior
    
    Attributes:
        timeout: Request timeout in seconds
        max_retries: Maximum number of retry attempts
        backoff_base: Base delay for exponential backoff (seconds)
        backoff_max: Maximum delay between retries (seconds)
        retryable_status_codes: List of HTTP status codes to retry
        retryable_exceptions: List of exception types to retry
    """
    timeout: float = 30.0
    max_retries: int = 3
    backoff_base: float = 1.0
    backoff_max: float = 60.0
    retryable_status_codes: List[int] = field(default_factory=lambda: [
        code.value for code in RetryableErrorCode
    ])
    retryable_exceptions: List[type] = field(default_factory=lambda: [
        aiohttp.ClientConnectionError,
        aiohttp.ClientTimeout,
        asyncio.TimeoutError,
    ])
    
    def is_retryable_status(self, status_code: int) -> bool:
        """Check if HTTP status code is retryable"""
        return status_code in self.retryable_status_codes
    
    def is_retryable_exception(self, exception: Exception) -> bool:
        """Check if exception type is retryable"""
        return any(isinstance(exception, exc_type) for exc_type in self.retryable_exceptions)
    
    def get_delay(self, attempt: int) -> float:
        """Calculate delay for retry attempt using exponential backoff"""
        delay = self.backoff_base * (2 ** (attempt - 1))
        return min(delay, self.backoff_max)


@dataclass
class TransportEvent:
    """Structured telemetry event for transport operations"""
    event_type: TransportEventType
    timestamp: float
    provider: str
    endpoint: str
    attempt: int = 1
    status_code: Optional[int] = None
    error_message: Optional[str] = None
    latency_ms: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for logging/telemetry"""
        return {
            "event_type": self.event_type.value,
            "timestamp": self.timestamp,
            "provider": self.provider,
            "endpoint": self.endpoint,
            "attempt": self.attempt,
            "status_code": self.status_code,
            "error_message": self.error_message,
            "latency_ms": self.latency_ms,
            "details": self.details,
        }


@dataclass
class UnifiedErrorEnvelope:
    """
    Unified error response envelope for all provider errors
    
    Provides consistent error structure across all providers
    """
    success: bool = False
    error_type: str = "unknown"
    error_code: Optional[str] = None
    http_status: Optional[int] = None
    message: str = "An unexpected error occurred"
    user_message: str = "Please try again or contact support"
    retry_after: Optional[float] = None
    is_retryable: bool = False
    provider: str = ""
    endpoint: str = ""
    raw_error: Optional[Any] = None
    
    def to_exception(self) -> "TransportError":
        """Convert error envelope to exception"""
        return TransportError(
            error_type=self.error_type,
            error_code=self.error_code,
            http_status=self.http_status,
            message=self.message,
            user_message=self.user_message,
            retry_after=self.retry_after,
            is_retryable=self.is_retryable,
            provider=self.provider,
            endpoint=self.endpoint,
            raw_error=self.raw_error,
        )


class TransportError(Exception):
    """
    Unified transport exception
    
    Attributes can be used to implement smart fallback and retry logic
    """
    def __init__(
        self,
        error_type: str = "unknown",
        error_code: Optional[str] = None,
        http_status: Optional[int] = None,
        message: str = "Transport error occurred",
        user_message: str = "Please try again or contact support",
        retry_after: Optional[float] = None,
        is_retryable: bool = False,
        provider: str = "",
        endpoint: str = "",
        raw_error: Optional[Any] = None,
    ):
        self.error_type = error_type
        self.error_code = error_code
        self.http_status = http_status
        self.message = message
        self.user_message = user_message
        self.retry_after = retry_after
        self.is_retryable = is_retryable
        self.provider = provider
        self.endpoint = endpoint
        self.raw_error = raw_error
        super().__init__(message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging"""
        return {
            "error_type": self.error_type,
            "error_code": self.error_code,
            "http_status": self.http_status,
            "message": self.message,
            "user_message": self.user_message,
            "retry_after": self.retry_after,
            "is_retryable": self.is_retryable,
            "provider": self.provider,
            "endpoint": self.endpoint,
        }


class ProviderTransport:
    """
    Resilient provider transport with retry and timeout handling
    
    Usage:
        transport = ProviderTransport("openai", policy=ProviderTransportPolicy())
        async with transport.session() as session:
            response = await transport.request(
                session,
                "POST",
                "/chat/completions",
                json={"model": "gpt-4", "messages": [...]}
            )
    """
    
    def __init__(
        self,
        provider: str,
        base_url: str,
        policy: Optional[ProviderTransportPolicy] = None,
        default_headers: Optional[Dict[str, str]] = None,
        event_callback: Optional[Callable[[TransportEvent], None]] = None,
    ):
        """
        Initialize provider transport
        
        Args:
            provider: Provider name (e.g., "openai", "qwen", "ollama")
            base_url: Base URL for API requests
            policy: Transport policy configuration
            default_headers: Default headers for all requests
            event_callback: Optional callback for telemetry events
        """
        self.provider = provider
        self.base_url = base_url.rstrip("/")
        self.policy = policy or ProviderTransportPolicy()
        self.default_headers = default_headers or {}
        self.event_callback = event_callback
        self._session: Optional[aiohttp.ClientSession] = None
    
    def _emit_event(self, event: TransportEvent) -> None:
        """Emit telemetry event"""
        if self.event_callback:
            self.event_callback(event)
        else:
            # Default logging for critical events
            if event.event_type in [
                TransportEventType.REQUEST_FAILURE,
                TransportEventType.STREAM_INTERRUPTED,
                TransportEventType.TIMEOUT_OCCURRED,
            ]:
                console.print(
                    f"[yellow]⚠️  [{self.provider}] {event.event_type.value}: "
                    f"{event.error_message}[/yellow]"
                )
            elif event.event_type == TransportEventType.REQUEST_SUCCESS:
                console.print(
                    f"[green]✓ [{self.provider}] Request successful "
                    f"({event.latency_ms:.0f}ms)[/green]"
                )
    
    @asynccontextmanager
    async def session(self):
        """Async context manager for aiohttp session"""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.policy.timeout)
            self._session = aiohttp.ClientSession(timeout=timeout)
        
        try:
            yield self._session
        finally:
            pass  # Don't close session here, allow reuse
    
    async def close(self):
        """Close the underlying session"""
        if self._session and not self._session.closed:
            await self._session.close()
    
    async def request(
        self,
        session: aiohttp.ClientSession,
        method: str,
        path: str,
        event_name: str = "api_call",
        **kwargs,
    ) -> Any:
        """
        Make HTTP request with retry logic
        
        Args:
            session: aiohttp session to use
            method: HTTP method
            path: API path (will be joined with base_url)
            event_name: Name for telemetry events
            **kwargs: Additional arguments for aiohttp request
            
        Returns:
            Response data (parsed JSON or text)
            
        Raises:
            TransportError: If request fails after all retries
        """
        url = f"{self.base_url}/{path.lstrip('/')}"
        headers = {**self.default_headers, **kwargs.pop("headers", {})}
        
        last_error: Optional[Exception] = None
        last_envelope: Optional[UnifiedErrorEnvelope] = None
        
        for attempt in range(1, self.policy.max_retries + 1):
            start_time = time.time()
            
            # Emit request start event
            self._emit_event(TransportEvent(
                event_type=TransportEventType.REQUEST_START,
                timestamp=start_time,
                provider=self.provider,
                endpoint=url,
                attempt=attempt,
                details={"method": method, "event_name": event_name},
            ))
            
            try:
                async with session.request(method, url, headers=headers, **kwargs) as response:
                    latency_ms = (time.time() - start_time) * 1000
                    
                    # Check for retryable status codes
                    if response.status in self.policy.retryable_status_codes:
                        error_data = await response.text()
                        
                        # Emit retry event
                        self._emit_event(TransportEvent(
                            event_type=TransportEventType.REQUEST_RETRY,
                            timestamp=time.time(),
                            provider=self.provider,
                            endpoint=url,
                            attempt=attempt,
                            status_code=response.status,
                            error_message=f"Retryable status: {response.status}",
                            latency_ms=latency_ms,
                        ))
                        
                        if attempt < self.policy.max_retries:
                            delay = self.policy.get_delay(attempt)
                            console.print(
                                f"[yellow]⏳ [{self.provider}] Retrying in {delay:.1f}s "
                                f"(attempt {attempt}/{self.policy.max_retries})...[/yellow]"
                            )
                            await asyncio.sleep(delay)
                            continue
                        else:
                            # Max retries reached
                            last_envelope = UnifiedErrorEnvelope(
                                error_type="http_error",
                                error_code=f"HTTP_{response.status}",
                                http_status=response.status,
                                message=f"Request failed after {attempt} attempts: {response.status}",
                                user_message=self._get_user_message(response.status),
                                is_retryable=False,
                                provider=self.provider,
                                endpoint=url,
                                raw_error=error_data,
                            )
                            break
                    
                    # Success - parse response
                    if response.status == 200:
                        self._emit_event(TransportEvent(
                            event_type=TransportEventType.REQUEST_SUCCESS,
                            timestamp=time.time(),
                            provider=self.provider,
                            endpoint=url,
                            attempt=attempt,
                            status_code=response.status,
                            latency_ms=latency_ms,
                        ))
                        
                        # Try to parse as JSON, fallback to text
                        try:
                            return await response.json()
                        except (aiohttp.ContentTypeError, json.JSONDecodeError):
                            return await response.text()
                    else:
                        # Non-retryable error
                        error_data = await response.text()
                        last_envelope = UnifiedErrorEnvelope(
                            error_type="http_error",
                            error_code=f"HTTP_{response.status}",
                            http_status=response.status,
                            message=f"Request failed: {response.status}",
                            user_message=self._get_user_message(response.status),
                            is_retryable=False,
                            provider=self.provider,
                            endpoint=url,
                            raw_error=error_data,
                        )
                        break
                        
            except Exception as e:
                latency_ms = (time.time() - start_time) * 1000
                last_error = e
                
                # Check if exception is retryable
                if self.policy.is_retryable_exception(e) and attempt < self.policy.max_retries:
                    delay = self.policy.get_delay(attempt)
                    
                    self._emit_event(TransportEvent(
                        event_type=TransportEventType.REQUEST_RETRY,
                        timestamp=time.time(),
                        provider=self.provider,
                        endpoint=url,
                        attempt=attempt,
                        error_message=str(e),
                        latency_ms=latency_ms,
                        details={"exception_type": type(e).__name__},
                    ))
                    
                    console.print(
                        f"[yellow]⏳ [{self.provider}] Retrying in {delay:.1f}s "
                        f"(attempt {attempt}/{self.policy.max_retries})...[/yellow]"
                    )
                    await asyncio.sleep(delay)
                    continue
                else:
                    # Non-retryable exception or max retries reached
                    self._emit_event(TransportEvent(
                        event_type=TransportEventType.REQUEST_FAILURE,
                        timestamp=time.time(),
                        provider=self.provider,
                        endpoint=url,
                        attempt=attempt,
                        error_message=str(e),
                        latency_ms=latency_ms,
                        details={"exception_type": type(e).__name__},
                    ))
                    
                    last_envelope = self._exception_to_envelope(e, url)
                    break
        
        # All retries exhausted or non-retryable error
        if last_envelope:
            raise last_envelope.to_exception()
        elif last_error:
            raise TransportError(
                error_type="unknown",
                message=str(last_error),
                provider=self.provider,
                endpoint=url,
                raw_error=last_error,
            )
        else:
            raise TransportError(
                error_type="unknown",
                message="Request failed with unknown error",
                provider=self.provider,
                endpoint=url,
            )
    
    async def stream_request(
        self,
        session: aiohttp.ClientSession,
        method: str,
        path: str,
        event_name: str = "stream_call",
        reconnect_hook: Optional[Callable[[], None]] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """
        Make streaming HTTP request with reconnection support
        
        Args:
            session: aiohttp session to use
            method: HTTP method
            path: API path
            event_name: Name for telemetry events
            reconnect_hook: Optional callback for reconnection events
            **kwargs: Additional arguments for aiohttp request
            
        Yields:
            Response chunks (text or JSON)
            
        Raises:
            TransportError: If stream fails and cannot reconnect
        """
        url = f"{self.base_url}/{path.lstrip('/')}"
        headers = {**self.default_headers, **kwargs.pop("headers", {})}
        
        last_error: Optional[Exception] = None
        
        for attempt in range(1, self.policy.max_retries + 1):
            start_time = time.time()
            
            self._emit_event(TransportEvent(
                event_type=TransportEventType.REQUEST_START,
                timestamp=start_time,
                provider=self.provider,
                endpoint=url,
                attempt=attempt,
                details={"method": method, "event_name": event_name, "streaming": True},
            ))
            
            try:
                async with session.request(method, url, headers=headers, **kwargs) as response:
                    if response.status != 200:
                        error_data = await response.text()
                        raise TransportError(
                            error_type="http_error",
                            http_status=response.status,
                            message=f"Stream request failed: {response.status}",
                            user_message=self._get_user_message(response.status),
                            provider=self.provider,
                            endpoint=url,
                            raw_error=error_data,
                        )
                    
                    # Stream successful - yield chunks
                    async for line in response.content:
                        if line.strip():
                            try:
                                # Try to parse as SSE or JSON
                                line_str = line.decode('utf-8').strip()
                                if line_str.startswith("data: "):
                                    yield line_str[6:]  # Remove "data: " prefix
                                else:
                                    yield line_str
                            except UnicodeDecodeError:
                                yield line.decode('utf-8', errors='replace')
                    
                    # Stream completed successfully
                    self._emit_event(TransportEvent(
                        event_type=TransportEventType.REQUEST_SUCCESS,
                        timestamp=time.time(),
                        provider=self.provider,
                        endpoint=url,
                        attempt=attempt,
                        status_code=response.status,
                        latency_ms=(time.time() - start_time) * 1000,
                    ))
                    return
                    
            except (aiohttp.ClientConnectionError, asyncio.TimeoutError) as e:
                last_error = e
                
                # Stream interrupted - attempt reconnect
                self._emit_event(TransportEvent(
                    event_type=TransportEventType.STREAM_INTERRUPTED,
                    timestamp=time.time(),
                    provider=self.provider,
                    endpoint=url,
                    attempt=attempt,
                    error_message=str(e),
                    details={"exception_type": type(e).__name__},
                ))
                
                if attempt < self.policy.max_retries:
                    if reconnect_hook:
                        reconnect_hook()
                    
                    delay = self.policy.get_delay(attempt)
                    console.print(
                        f"[yellow]⏳ [{self.provider}] Stream interrupted. "
                        f"Reconnecting in {delay:.1f}s...[/yellow]"
                    )
                    await asyncio.sleep(delay)
                    
                    self._emit_event(TransportEvent(
                        event_type=TransportEventType.STREAM_RECONNECTED,
                        timestamp=time.time(),
                        provider=self.provider,
                        endpoint=url,
                        attempt=attempt + 1,
                    ))
                    continue
                else:
                    # Max retries reached
                    break
            
            except Exception as e:
                last_error = e
                self._emit_event(TransportEvent(
                    event_type=TransportEventType.REQUEST_FAILURE,
                    timestamp=time.time(),
                    provider=self.provider,
                    endpoint=url,
                    attempt=attempt,
                    error_message=str(e),
                    details={"exception_type": type(e).__name__},
                ))
                break
        
        # Stream failed
        if last_error:
            raise self._exception_to_envelope(last_error, url).to_exception()
        else:
            raise TransportError(
                error_type="unknown",
                message="Stream failed with unknown error",
                provider=self.provider,
                endpoint=url,
            )
    
    def _get_user_message(self, status_code: int) -> str:
        """Generate user-friendly message for HTTP status code"""
        messages = {
            400: "Invalid request. Please check your input.",
            401: "Authentication failed. Please check your API key.",
            403: "Access denied. Please check your permissions.",
            404: "Resource not found.",
            429: "Rate limit exceeded. Please wait a moment and try again.",
            500: "Server error. Please try again later.",
            502: "Bad gateway. The service may be temporarily unavailable.",
            503: "Service unavailable. Please try again later.",
            504: "Gateway timeout. The service is taking too long to respond.",
        }
        return messages.get(status_code, "An unexpected error occurred. Please try again.")
    
    def _exception_to_envelope(self, exception: Exception, url: str) -> UnifiedErrorEnvelope:
        """Convert exception to unified error envelope"""
        if isinstance(exception, aiohttp.ClientTimeout):
            return UnifiedErrorEnvelope(
                error_type="timeout",
                error_code="TIMEOUT",
                message=f"Request timed out after {self.policy.timeout}s",
                user_message="The request took too long. Please try again or check your connection.",
                is_retryable=True,
                provider=self.provider,
                endpoint=url,
                raw_error=exception,
            )
        elif isinstance(exception, aiohttp.ClientConnectionError):
            return UnifiedErrorEnvelope(
                error_type="connection_error",
                error_code="CONNECTION_ERROR",
                message=f"Connection error: {str(exception)}",
                user_message="Cannot connect to the service. Please check your internet connection.",
                is_retryable=True,
                provider=self.provider,
                endpoint=url,
                raw_error=exception,
            )
        elif isinstance(exception, asyncio.TimeoutError):
            return UnifiedErrorEnvelope(
                error_type="timeout",
                error_code="ASYNC_TIMEOUT",
                message="Async operation timed out",
                user_message="The operation took too long. Please try again.",
                is_retryable=True,
                provider=self.provider,
                endpoint=url,
                raw_error=exception,
            )
        else:
            return UnifiedErrorEnvelope(
                error_type="unknown",
                message=str(exception),
                user_message="An unexpected error occurred.",
                is_retryable=False,
                provider=self.provider,
                endpoint=url,
                raw_error=exception,
            )

