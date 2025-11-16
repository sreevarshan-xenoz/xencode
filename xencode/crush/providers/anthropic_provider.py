"""Anthropic provider implementation."""

import json
from typing import AsyncIterator, Dict, List, Optional, Any

try:
    from anthropic import AsyncAnthropic, AnthropicError, APIError, RateLimitError, APITimeoutError
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

from xencode.crush.providers.base import (
    Provider,
    ProviderResponse,
    ProviderError,
    ProviderChunk,
    Message,
    MessageRole,
    ToolCall,
)


class AnthropicProvider(Provider):
    """Anthropic API provider."""
    
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
        """Initialize Anthropic provider."""
        if not ANTHROPIC_AVAILABLE:
            raise ProviderError(
                "Anthropic library not installed. Install with: pip install anthropic",
                provider="anthropic",
                error_type="configuration"
            )
        
        super().__init__(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens or 4096,  # Anthropic requires max_tokens
            timeout=timeout,
            **kwargs
        )
        
        # Initialize Anthropic client
        client_kwargs = {
            "api_key": api_key,
            "timeout": timeout,
        }
        if base_url:
            client_kwargs["base_url"] = base_url
        
        self.client = AsyncAnthropic(**client_kwargs)
    
    @property
    def name(self) -> str:
        """Provider name."""
        return "anthropic"
    
    def _convert_messages(self, messages: List[Message]) -> tuple[Optional[str], List[Dict[str, Any]]]:
        """
        Convert internal messages to Anthropic format.
        
        Returns:
            Tuple of (system_prompt, messages)
        """
        system_prompt = None
        anthropic_messages = []
        
        for msg in messages:
            # Extract system message separately
            if msg.role == MessageRole.SYSTEM:
                system_prompt = msg.content
                continue
            
            # Convert role
            role = "user" if msg.role == MessageRole.USER else "assistant"
            
            # Build content
            content = []
            
            # Add text content
            if msg.content:
                content.append({
                    "type": "text",
                    "text": msg.content,
                })
            
            # Add tool use (Anthropic's version of tool calls)
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    content.append({
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.name,
                        "input": tc.input,
                    })
            
            # Add tool result
            if msg.tool_call_id:
                content = [{
                    "type": "tool_result",
                    "tool_use_id": msg.tool_call_id,
                    "content": msg.content,
                }]
                role = "user"  # Tool results must be from user
            
            anthropic_messages.append({
                "role": role,
                "content": content if len(content) > 1 or msg.tool_calls or msg.tool_call_id else msg.content,
            })
        
        return system_prompt, anthropic_messages
    
    def _convert_tools(self, tools: Optional[List[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
        """Convert tools to Anthropic tool format."""
        if not tools:
            return None
        
        anthropic_tools = []
        for tool in tools:
            anthropic_tools.append({
                "name": tool.get("name", ""),
                "description": tool.get("description", ""),
                "input_schema": tool.get("parameters", {}),
            })
        
        return anthropic_tools
    
    async def generate(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        stream: bool = False,
        **kwargs
    ) -> ProviderResponse:
        """Generate a response from Anthropic."""
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
            system_prompt, anthropic_messages = self._convert_messages(messages)
            anthropic_tools = self._convert_tools(tools)
            
            request_params = {
                "model": self.model,
                "messages": anthropic_messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
            
            if system_prompt:
                request_params["system"] = system_prompt
            
            if anthropic_tools:
                request_params["tools"] = anthropic_tools
            
            # Add extra parameters
            request_params.update(kwargs)
            
            response = await self.client.messages.create(**request_params)
            
            # Extract response data
            content = ""
            tool_calls = []
            
            for block in response.content:
                if block.type == "text":
                    content += block.text
                elif block.type == "tool_use":
                    tool_calls.append(ToolCall(
                        id=block.id,
                        name=block.name,
                        input=block.input,
                    ))
            
            return ProviderResponse(
                content=content,
                tool_calls=tool_calls if tool_calls else None,
                model=response.model,
                provider=self.name,
                prompt_tokens=response.usage.input_tokens,
                completion_tokens=response.usage.output_tokens,
                finish_reason=response.stop_reason,
            )
        
        except Exception as e:
            raise self._handle_anthropic_error(e)
    
    async def generate_stream(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> AsyncIterator[ProviderChunk]:
        """Generate a streaming response from Anthropic."""
        self.validate_config()
        
        try:
            system_prompt, anthropic_messages = self._convert_messages(messages)
            anthropic_tools = self._convert_tools(tools)
            
            request_params = {
                "model": self.model,
                "messages": anthropic_messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "stream": True,
            }
            
            if system_prompt:
                request_params["system"] = system_prompt
            
            if anthropic_tools:
                request_params["tools"] = anthropic_tools
            
            # Add extra parameters
            request_params.update(kwargs)
            
            stream = await self.client.messages.create(**request_params)
            
            # Track tool calls being built
            tool_calls_in_progress = {}
            
            async for event in stream:
                provider_chunk = ProviderChunk()
                
                if event.type == "content_block_start":
                    # Start of a content block
                    if hasattr(event, "content_block"):
                        block = event.content_block
                        if block.type == "tool_use":
                            tool_calls_in_progress[block.id] = {
                                "id": block.id,
                                "name": block.name,
                                "input": {},
                            }
                
                elif event.type == "content_block_delta":
                    # Delta update for content block
                    if hasattr(event, "delta"):
                        delta = event.delta
                        
                        if delta.type == "text_delta":
                            provider_chunk.text = delta.text
                        
                        elif delta.type == "input_json_delta":
                            # Tool input is being streamed
                            if hasattr(event, "index"):
                                # Find the tool call by index
                                tool_ids = list(tool_calls_in_progress.keys())
                                if event.index < len(tool_ids):
                                    tool_id = tool_ids[event.index]
                                    # Accumulate JSON string
                                    if "input_json" not in tool_calls_in_progress[tool_id]:
                                        tool_calls_in_progress[tool_id]["input_json"] = ""
                                    tool_calls_in_progress[tool_id]["input_json"] += delta.partial_json
                
                elif event.type == "content_block_stop":
                    # End of a content block
                    if hasattr(event, "index"):
                        # Finalize tool call if it exists
                        tool_ids = list(tool_calls_in_progress.keys())
                        if event.index < len(tool_ids):
                            tool_id = tool_ids[event.index]
                            tc_data = tool_calls_in_progress[tool_id]
                            
                            # Parse accumulated JSON
                            if "input_json" in tc_data:
                                try:
                                    tc_data["input"] = json.loads(tc_data["input_json"])
                                except json.JSONDecodeError:
                                    tc_data["input"] = {}
                            
                            provider_chunk.tool_calls = [ToolCall(
                                id=tc_data["id"],
                                name=tc_data["name"],
                                input=tc_data["input"],
                            )]
                
                elif event.type == "message_delta":
                    # Message-level delta (e.g., stop reason)
                    if hasattr(event, "delta"):
                        if hasattr(event.delta, "stop_reason") and event.delta.stop_reason:
                            provider_chunk.finish_reason = event.delta.stop_reason
                    
                    # Usage information
                    if hasattr(event, "usage"):
                        provider_chunk.tokens = event.usage.output_tokens
                
                elif event.type == "message_stop":
                    # End of message
                    pass
                
                yield provider_chunk
        
        except Exception as e:
            raise self._handle_anthropic_error(e)
    
    def estimate_tokens(self, messages: List[Message]) -> int:
        """
        Estimate token count for messages.
        
        Anthropic uses a different tokenization than OpenAI.
        This is a rough approximation: 1 token ≈ 3.5 characters for English.
        """
        total_chars = 0
        
        for msg in messages:
            # Count content
            if msg.content:
                total_chars += len(msg.content)
            
            # Count tool calls
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    total_chars += len(tc.name)
                    total_chars += len(json.dumps(tc.input))
            
            # Add overhead for message structure
            total_chars += 20
        
        # Convert characters to tokens (rough approximation)
        return int(total_chars / 3.5)
    
    def _handle_anthropic_error(self, error: Exception) -> ProviderError:
        """Convert Anthropic errors to ProviderError."""
        if isinstance(error, ProviderError):
            return error
        
        if not ANTHROPIC_AVAILABLE:
            return self._handle_error(error, "Anthropic error")
        
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
            message = f"Anthropic API error: {error.message}"
        elif isinstance(error, AnthropicError):
            error_type = "anthropic_error"
            message = f"Anthropic error: {str(error)}"
        
        return ProviderError(
            message=message,
            provider=self.name,
            error_type=error_type,
            retryable=retryable,
            original_error=error
        )
