# Provider Abstraction Layer

This module provides a unified interface for interacting with different AI model providers (OpenAI, Anthropic, etc.).

## Features

- **Unified Interface**: Single API for all providers
- **Streaming Support**: Real-time response streaming
- **Tool Calling**: Function calling / tool use support
- **Token Estimation**: Estimate costs before making requests
- **Error Handling**: Consistent error handling across providers
- **Provider Registry**: Easy registration and discovery of providers
- **Caching**: Optional instance caching for performance

## Installation

Install the required provider libraries:

```bash
# For OpenAI
pip install openai tiktoken

# For Anthropic
pip install anthropic

# Or install both
pip install openai anthropic tiktoken
```

## Quick Start

### Using the Registry

```python
from xencode.crush.providers import create_provider, Message, MessageRole

# Create a provider instance
provider = create_provider(
    name="openai",
    model="gpt-4",
    api_key="your-api-key",
    temperature=0.7,
    max_tokens=1000
)

# Generate a response
messages = [
    Message(role=MessageRole.USER, content="Hello, how are you?")
]

response = await provider.generate(messages)
print(response.content)
print(f"Tokens used: {response.total_tokens}")
```

### Streaming Responses

```python
# Stream the response
async for chunk in provider.generate_stream(messages):
    if chunk.text:
        print(chunk.text, end="", flush=True)
    if chunk.finish_reason:
        print(f"\nFinished: {chunk.finish_reason}")
```

### Tool Calling

```python
# Define tools
tools = [
    {
        "name": "get_weather",
        "description": "Get the current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name"
                }
            },
            "required": ["location"]
        }
    }
]

# Generate with tools
response = await provider.generate(messages, tools=tools)

if response.tool_calls:
    for tool_call in response.tool_calls:
        print(f"Tool: {tool_call.name}")
        print(f"Input: {tool_call.input}")
```

## Supported Providers

### OpenAI

```python
from xencode.crush.providers import create_provider

provider = create_provider(
    name="openai",
    model="gpt-4",
    api_key="sk-...",
    base_url="https://api.openai.com/v1",  # Optional
    temperature=0.7,
    max_tokens=1000,
    timeout=60
)
```

**Features:**
- Streaming support
- Function calling
- Token counting with tiktoken
- Automatic retry on rate limits

### Anthropic

```python
from xencode.crush.providers import create_provider

provider = create_provider(
    name="anthropic",
    model="claude-3-opus-20240229",
    api_key="sk-ant-...",
    temperature=0.7,
    max_tokens=4096,  # Required for Anthropic
    timeout=60
)
```

**Features:**
- Streaming support
- Tool use (Anthropic's version of function calling)
- Token estimation
- System prompt support

## Custom Providers

You can register custom providers:

```python
from xencode.crush.providers import Provider, ProviderResponse, register_provider

class MyCustomProvider(Provider):
    @property
    def name(self) -> str:
        return "custom"
    
    async def generate(self, messages, tools=None, stream=False, **kwargs):
        # Your implementation
        return ProviderResponse(
            content="Generated text",
            model=self.model,
            provider=self.name,
            prompt_tokens=10,
            completion_tokens=20
        )
    
    async def generate_stream(self, messages, tools=None, **kwargs):
        # Your streaming implementation
        yield ProviderChunk(text="chunk")
    
    def estimate_tokens(self, messages):
        # Your token estimation
        return 100

# Register the provider
register_provider("custom", MyCustomProvider)

# Use it
provider = create_provider(
    name="custom",
    model="my-model",
    api_key="key"
)
```

## Error Handling

All providers raise `ProviderError` with consistent attributes:

```python
from xencode.crush.providers import ProviderError

try:
    response = await provider.generate(messages)
except ProviderError as e:
    print(f"Provider: {e.provider}")
    print(f"Error type: {e.error_type}")
    print(f"Retryable: {e.retryable}")
    print(f"Message: {str(e)}")
```

Error types:
- `configuration`: Invalid configuration
- `authentication`: Invalid API key
- `rate_limit`: Rate limit exceeded (retryable)
- `timeout`: Request timeout (retryable)
- `network`: Network error (retryable)
- `api_error`: API error
- `unknown`: Unknown error

## Token Estimation

Estimate tokens before making requests:

```python
messages = [
    Message(role=MessageRole.USER, content="Hello, world!")
]

estimated_tokens = provider.estimate_tokens(messages)
print(f"Estimated tokens: {estimated_tokens}")
```

## Provider Registry

The registry manages provider instances:

```python
from xencode.crush.providers import get_registry

registry = get_registry()

# List available providers
providers = registry.list_providers()
print(f"Available: {providers}")

# Check if provider is registered
if registry.is_registered("openai"):
    print("OpenAI is available")

# Create with caching
provider = registry.create_provider(
    name="openai",
    model="gpt-4",
    api_key="key",
    cache_key="my-provider"
)

# Clear cache
registry.clear_cache("my-provider")
```

## Best Practices

1. **Reuse Instances**: Cache provider instances when possible
2. **Handle Errors**: Always catch `ProviderError` and check `retryable`
3. **Estimate Tokens**: Use token estimation to avoid surprises
4. **Stream Large Responses**: Use streaming for better UX
5. **Set Timeouts**: Configure appropriate timeouts for your use case
6. **Validate Config**: Call `validate_config()` after creating providers

## Architecture

```
Provider (Abstract Base)
├── OpenAIProvider
├── AnthropicProvider
└── CustomProvider

ProviderRegistry
├── register()
├── create_provider()
└── list_providers()
```

## Thread Safety

- Provider instances are thread-safe for reading
- Use separate instances for concurrent writes
- Registry is thread-safe for all operations
