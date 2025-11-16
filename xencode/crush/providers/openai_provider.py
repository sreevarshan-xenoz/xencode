"""OpenAI provider implementation."""

import json
import asyncio
from typing import AsyncIterator, Dict, List, Optional, Any

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

try:
    from openai import AsyncOpenAI, OpenAIError, APIError, RateLimitError, APITimeoutError
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from xencode.crush.providers.base import (
    Provider,
    ProviderResponse,
    ProviderError,
    ProviderChunk,
    Message,
    MessageRole,
    ToolCall,
)


class OpenAIProvider(Provider):
    """OpenAI API provider."""
    
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
        """Initialize OpenAI provider."""
        if not OPENAI_AVAILABLE:
            raise ProviderError(
                "OpenAI library not installed. Install with: pip install openai",
                provider="openai",
                error_type="configuration"
            )
        
        super().__init__(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            **kwargs
        )
        
        # Initialize OpenAI client
        client_kwargs = {
            "api_key": api_key,
            "timeout": timeout,
        }
        if base_url:
            client_kwargs["base_url"] = base_url
        
        self.client = AsyncOpenAI(**client_kwargs)
        
        # Initialize tokenizer if available
        self.tokenizer = None
        if TIKTOKEN_AVAILABLE:
            try:
                self.tokenizer = tiktoken.encoding_for_model(model)
            except KeyError:
                # Fallback to cl100k_base for unknown models
                try:
                    self.tokenizer = tiktoken.get_encoding("cl100k_base")
                except Exception:
                    pass
    
    @property
    def name(self) -> str:
        """Provider name."""
        return "openai"
    
    def _convert_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Convert internal messages to OpenAI format."""
        openai_messages = []
        
        for msg in messages:
            openai_msg = {
                "role": msg.role.value,
                "content": msg.content,
            }
            
            # Add tool calls if present
            if msg.tool_calls:
                openai_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.input) if isinstance(tc.input, dict) else tc.input,
                        }
                    }
                    for tc in msg.tool_calls
                ]
            
            # Add tool call ID for tool responses
            if msg.tool_call_id:
                openai_msg["tool_call_id"] = msg.tool_call_id
            
            # Add name for tool responses
            if msg.name:
                openai_msg["name"] = msg.name
            
            openai_messages.append(openai_msg)
        
        return openai_messages
    
    def _convert_tools(self, tools: Optional[List[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
        """Convert tools to OpenAI function calling format."""
        if not tools:
            return None
        
        openai_tools = []
        for tool in tools:
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("parameters", {}),
                }
            })
        
        return openai_tools
    
    async def generate(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        stream: bool = False,
        **kwargs
    ) -> ProviderResponse:
        """Generate a response from OpenAI."""
        self.validate_config()
        
        if stream:
            # Collect streaming response
            content = ""
            tool_calls_dict = {}
            finish_reason = None
            prompt_tokens = 0
            completion_tokens = 0
            
            async for chunk in self.generate_stream(messages, tools, **kwargs):
                if chunk.text:
                    content += chunk.text
                if chunk.tool_calls:
                    for tc in chunk.tool_calls:
                        if tc.id not in tool_calls_dict:
                            tool_calls_dict[tc.id] = tc
                        else:
                            # Merge tool call data
                            existing = tool_calls_dict[tc.id]
                            if tc.name:
                                existing.name = tc.name
                            if tc.input:
                                existing.input.update(tc.input)
                if chunk.finish_reason:
                    finish_reason = chunk.finish_reason
                completion_tokens += chunk.tokens
            
            tool_calls = list(tool_calls_dict.values()) if tool_calls_dict else None
            
            return ProviderResponse(
                content=content,
                tool_calls=tool_calls,
                model=self.model,
                provider=self.name,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                finish_reason=finish_reason,
            )
        
        # Non-streaming generation
        try:
            openai_messages = self._convert_messages(messages)
            openai_tools = self._convert_tools(tools)
            
            request_params = {
                "model": self.model,
                "messages": openai_messages,
                "temperature": self.temperature,
            }
            
            if self.max_tokens:
                request_params["max_tokens"] = self.max_tokens
            
            if openai_tools:
                request_params["tools"] = openai_tools
                request_params["tool_choice"] = "auto"
            
            # Add extra parameters
            request_params.update(kwargs)
            
            response = await self.client.chat.completions.create(**request_params)
            
            # Extract response data
            choice = response.choices[0]
            message = choice.message
            
            content = message.content or ""
            tool_calls = None
            
            if message.tool_calls:
                tool_calls = []
                for tc in message.tool_calls:
                    try:
                        arguments = json.loads(tc.function.arguments) if isinstance(tc.function.arguments, str) else tc.function.arguments
                    except json.JSONDecodeError:
                        arguments = {"raw": tc.function.arguments}
                    
                    tool_calls.append(ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        input=arguments,
                    ))
            
            return ProviderResponse(
                content=content,
                tool_calls=tool_calls,
                model=response.model,
                provider=self.name,
                prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
                completion_tokens=response.usage.completion_tokens if response.usage else 0,
                finish_reason=choice.finish_reason,
            )
        
        except Exception as e:
            raise self._handle_openai_error(e)
    
    async def generate_stream(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> AsyncIterator[ProviderChunk]:
        """Generate a streaming response from OpenAI."""
        self.validate_config()
        
        try:
            openai_messages = self._convert_messages(messages)
            openai_tools = self._convert_tools(tools)
            
            request_params = {
                "model": self.model,
                "messages": openai_messages,
                "temperature": self.temperature,
                "stream": True,
            }
            
            if self.max_tokens:
                request_params["max_tokens"] = self.max_tokens
            
            if openai_tools:
                request_params["tools"] = openai_tools
                request_params["tool_choice"] = "auto"
            
            # Add extra parameters
            request_params.update(kwargs)
            
            stream = await self.client.chat.completions.create(**request_params)
            
            # Track tool calls being built
            tool_calls_in_progress = {}
            
            async for chunk in stream:
                if not chunk.choices:
                    continue
                
                choice = chunk.choices[0]
                delta = choice.delta
                
                provider_chunk = ProviderChunk()
                
                # Extract text content
                if delta.content:
                    provider_chunk.text = delta.content
                
                # Extract tool calls
                if delta.tool_calls:
                    tool_calls = []
                    for tc_delta in delta.tool_calls:
                        tc_id = tc_delta.id or f"call_{tc_delta.index}"
                        
                        # Initialize or update tool call
                        if tc_id not in tool_calls_in_progress:
                            tool_calls_in_progress[tc_id] = {
                                "id": tc_id,
                                "name": "",
                                "arguments": "",
                            }
                        
                        if tc_delta.function:
                            if tc_delta.function.name:
                                tool_calls_in_progress[tc_id]["name"] = tc_delta.function.name
                            if tc_delta.function.arguments:
                                tool_calls_in_progress[tc_id]["arguments"] += tc_delta.function.arguments
                        
                        # Parse arguments if complete
                        try:
                            arguments = json.loads(tool_calls_in_progress[tc_id]["arguments"])
                        except json.JSONDecodeError:
                            arguments = {}
                        
                        tool_calls.append(ToolCall(
                            id=tc_id,
                            name=tool_calls_in_progress[tc_id]["name"],
                            input=arguments,
                        ))
                    
                    provider_chunk.tool_calls = tool_calls
                
                # Extract finish reason
                if choice.finish_reason:
                    provider_chunk.finish_reason = choice.finish_reason
                
                yield provider_chunk
        
        except Exception as e:
            raise self._handle_openai_error(e)
    
    def estimate_tokens(self, messages: List[Message]) -> int:
        """Estimate token count for messages."""
        if not self.tokenizer:
            # Rough approximation: 1 token ≈ 4 characters
            total_chars = sum(len(msg.content) for msg in messages)
            return total_chars // 4
        
        try:
            total_tokens = 0
            for msg in messages:
                # Count message overhead (role, formatting)
                total_tokens += 4
                
                # Count content tokens
                if msg.content:
                    total_tokens += len(self.tokenizer.encode(msg.content))
                
                # Count tool call tokens
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        total_tokens += len(self.tokenizer.encode(tc.name))
                        total_tokens += len(self.tokenizer.encode(json.dumps(tc.input)))
            
            # Add overhead for message list
            total_tokens += 3
            
            return total_tokens
        
        except Exception:
            # Fallback to character-based estimation
            total_chars = sum(len(msg.content) for msg in messages)
            return total_chars // 4
    
    def _handle_openai_error(self, error: Exception) -> ProviderError:
        """Convert OpenAI errors to ProviderError."""
        if isinstance(error, ProviderError):
            return error
        
        if not OPENAI_AVAILABLE:
            return self._handle_error(error, "OpenAI error")
        
        message = str(error)
        error_type = "unknown"
        retryable = False
        
        if isinstance(error, RateLimitError):
            error_type = "rate_limit"
            retryable = True
            message = "Rate limit exceeded. Please try again later."
        elif isinstance(error, APITimeoutError):
            error_type = "timeout"
            retryable = True
            message = "Request timed out. Please try again."
        elif isinstance(error, APIError):
            error_type = "api_error"
            retryable = error.status_code >= 500
            message = f"OpenAI API error: {error.message}"
        elif isinstance(error, OpenAIError):
            error_type = "openai_error"
            message = f"OpenAI error: {str(error)}"
        
        return ProviderError(
            message=message,
            provider=self.name,
            error_type=error_type,
            retryable=retryable,
            original_error=error
        )
