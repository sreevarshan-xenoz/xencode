"""Base provider interface for AI models."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncIterator, Dict, List, Optional, Any
import time


class MessageRole(str, Enum):
    """Message role in conversation."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


@dataclass
class ToolCall:
    """Represents a tool call from the AI."""
    id: str
    name: str
    input: Dict[str, Any]


@dataclass
class ToolResult:
    """Represents the result of a tool execution."""
    tool_call_id: str
    name: str
    content: str
    is_error: bool = False
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class Message:
    """Represents a message in the conversation."""
    role: MessageRole
    content: str
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary format."""
        result = {
            "role": self.role.value,
            "content": self.content,
        }
        if self.tool_calls:
            result["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": tc.input,
                    }
                }
                for tc in self.tool_calls
            ]
        if self.tool_call_id:
            result["tool_call_id"] = self.tool_call_id
        if self.name:
            result["name"] = self.name
        return result


@dataclass
class ProviderChunk:
    """Represents a streaming chunk from the provider."""
    text: str = ""
    tool_calls: Optional[List[ToolCall]] = None
    finish_reason: Optional[str] = None
    tokens: int = 0


@dataclass
class ProviderResponse:
    """Response from a provider."""
    content: str
    tool_calls: Optional[List[ToolCall]] = None
    model: str = ""
    provider: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def total_tokens(self) -> int:
        """Total tokens used."""
        return self.prompt_tokens + self.completion_tokens


class ProviderError(Exception):
    """Base exception for provider errors."""
    
    def __init__(
        self,
        message: str,
        provider: str = "",
        error_type: str = "unknown",
        retryable: bool = False,
        original_error: Optional[Exception] = None
    ):
        super().__init__(message)
        self.provider = provider
        self.error_type = error_type
        self.retryable = retryable
        self.original_error = original_error


class Provider(ABC):
    """Abstract base class for AI model providers."""
    
    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        timeout: int = 60,
        **kwargs
    ):
        """
        Initialize provider.
        
        Args:
            model: Model identifier
            api_key: API key for authentication
            base_url: Optional base URL for API
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            timeout: Request timeout in seconds
            **kwargs: Additional provider-specific parameters
        """
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.extra_params = kwargs
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name."""
        pass
    
    @abstractmethod
    async def generate(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        stream: bool = False,
        **kwargs
    ) -> ProviderResponse:
        """
        Generate a response from the model.
        
        Args:
            messages: Conversation history
            tools: Available tools for function calling
            stream: Whether to stream the response
            **kwargs: Additional generation parameters
            
        Returns:
            ProviderResponse with generated content
            
        Raises:
            ProviderError: If generation fails
        """
        pass
    
    @abstractmethod
    async def generate_stream(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> AsyncIterator[ProviderChunk]:
        """
        Generate a streaming response from the model.
        
        Args:
            messages: Conversation history
            tools: Available tools for function calling
            **kwargs: Additional generation parameters
            
        Yields:
            ProviderChunk objects with incremental content
            
        Raises:
            ProviderError: If generation fails
        """
        pass
    
    @abstractmethod
    def estimate_tokens(self, messages: List[Message]) -> int:
        """
        Estimate token count for messages.
        
        Args:
            messages: Messages to estimate
            
        Returns:
            Estimated token count
        """
        pass
    
    def validate_config(self) -> None:
        """
        Validate provider configuration.
        
        Raises:
            ProviderError: If configuration is invalid
        """
        if not self.api_key:
            raise ProviderError(
                "API key is required",
                provider=self.name,
                error_type="configuration"
            )
        if not self.model:
            raise ProviderError(
                "Model is required",
                provider=self.name,
                error_type="configuration"
            )
    
    def _handle_error(self, error: Exception, context: str = "") -> ProviderError:
        """
        Convert exceptions to ProviderError.
        
        Args:
            error: Original exception
            context: Additional context about the error
            
        Returns:
            ProviderError with appropriate details
        """
        if isinstance(error, ProviderError):
            return error
        
        message = f"{context}: {str(error)}" if context else str(error)
        
        # Determine if error is retryable
        retryable = False
        error_type = "unknown"
        
        error_str = str(error).lower()
        if "timeout" in error_str or "timed out" in error_str:
            error_type = "timeout"
            retryable = True
        elif "rate limit" in error_str or "429" in error_str:
            error_type = "rate_limit"
            retryable = True
        elif "connection" in error_str or "network" in error_str:
            error_type = "network"
            retryable = True
        elif "authentication" in error_str or "401" in error_str:
            error_type = "authentication"
            retryable = False
        elif "not found" in error_str or "404" in error_str:
            error_type = "not_found"
            retryable = False
        
        return ProviderError(
            message=message,
            provider=self.name,
            error_type=error_type,
            retryable=retryable,
            original_error=error
        )
