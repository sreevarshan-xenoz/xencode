"""
Model provider abstraction system for Xencode
Supports multiple AI model providers (OpenAI, Anthropic, Hugging Face, etc.)
"""
import asyncio
import json
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, AsyncIterator
from dataclasses import dataclass
import aiohttp
import time


@dataclass
class ModelInfo:
    """Information about a specific model."""
    name: str
    provider: str
    max_tokens: int
    context_window: int
    capabilities: List[str]  # e.g., ["text", "code", "chat", "instruct"]
    pricing: Optional[Dict[str, float]] = None  # cost per 1k tokens: {"input": 0.01, "output": 0.03}


class ModelProvider(ABC):
    """Abstract base class for all model providers."""
    
    def __init__(self, api_key: str, base_url: Optional[str] = None):
        self.api_key = api_key
        self.base_url = base_url
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    @abstractmethod
    async def list_models(self) -> List[ModelInfo]:
        """List available models from this provider."""
        pass
    
    @abstractmethod
    async def generate(self, 
                     prompt: str, 
                     model: str, 
                     max_tokens: int = 1024,
                     temperature: float = 0.7,
                     stream: bool = False) -> AsyncIterator[str]:
        """Generate text from the model."""
        pass
    
    @abstractmethod
    async def chat(self,
                  messages: List[Dict[str, str]],
                  model: str,
                  max_tokens: int = 1024,
                  temperature: float = 0.7,
                  stream: bool = False) -> AsyncIterator[str]:
        """Chat with the model."""
        pass
    
    @abstractmethod
    def get_provider_name(self) -> str:
        """Get the name of this provider."""
        pass


class OllamaProvider(ModelProvider):
    """Ollama model provider implementation."""
    
    def __init__(self, api_key: str = "", base_url: str = "http://localhost:11434"):
        super().__init__(api_key, base_url)
    
    async def list_models(self) -> List[ModelInfo]:
        """List available models from Ollama."""
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        try:
            async with self.session.get(f"{self.base_url}/api/tags") as response:
                if response.status == 200:
                    data = await response.json()
                    models = []
                    
                    for model_data in data.get("models", []):
                        name = model_data.get("name", "")
                        # Extract model info from Ollama response
                        models.append(ModelInfo(
                            name=name,
                            provider="ollama",
                            max_tokens=2048,  # Default for most models
                            context_window=2048,
                            capabilities=["text", "chat", "instruct"]
                        ))
                    
                    return models
                else:
                    print(f"Error listing Ollama models: {response.status}")
                    return []
        except Exception as e:
            print(f"Error connecting to Ollama: {e}")
            return []
    
    async def generate(self, 
                     prompt: str, 
                     model: str, 
                     max_tokens: int = 1024,
                     temperature: float = 0.7,
                     stream: bool = False) -> AsyncIterator[str]:
        """Generate text from Ollama model."""
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        payload = {
            "model": model,
            "prompt": prompt,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            },
            "stream": stream
        }
        
        if stream:
            async with self.session.post(f"{self.base_url}/api/generate", json=payload) as response:
                async for line in response.content:
                    if line.strip():
                        try:
                            data = json.loads(line)
                            if "response" in data:
                                yield data["response"]
                            if data.get("done", False):
                                break
                        except json.JSONDecodeError:
                            continue
        else:
            async with self.session.post(f"{self.base_url}/api/generate", json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    yield data.get("response", "")
                else:
                    yield f"Error: {response.status}"
    
    async def chat(self,
                  messages: List[Dict[str, str]],
                  model: str,
                  max_tokens: int = 1024,
                  temperature: float = 0.7,
                  stream: bool = False) -> AsyncIterator[str]:
        """Chat with Ollama model."""
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        payload = {
            "model": model,
            "messages": messages,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            },
            "stream": stream
        }
        
        if stream:
            async with self.session.post(f"{self.base_url}/api/chat", json=payload) as response:
                async for line in response.content:
                    if line.strip():
                        try:
                            data = json.loads(line)
                            if "message" in data and "content" in data["message"]:
                                yield data["message"]["content"]
                            if data.get("done", False):
                                break
                        except json.JSONDecodeError:
                            continue
        else:
            async with self.session.post(f"{self.base_url}/api/chat", json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    yield data.get("message", {}).get("content", "")
                else:
                    yield f"Error: {response.status}"
    
    def get_provider_name(self) -> str:
        """Get the name of this provider."""
        return "ollama"


class OpenAIProvider(ModelProvider):
    """OpenAI model provider implementation."""
    
    def __init__(self, api_key: str, base_url: str = "https://api.openai.com/v1"):
        super().__init__(api_key, base_url)
    
    async def list_models(self) -> List[ModelInfo]:
        """List available models from OpenAI."""
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            async with self.session.get(f"{self.base_url}/models", headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    models = []
                    
                    for model_data in data.get("data", []):
                        id = model_data.get("id", "")
                        # Determine capabilities based on model name
                        capabilities = ["text"]
                        if "gpt" in id.lower():
                            capabilities.extend(["chat", "instruct"])
                        if "instruct" in id.lower():
                            capabilities.append("instruct")
                        
                        models.append(ModelInfo(
                            name=id,
                            provider="openai",
                            max_tokens=4096,  # Varies by model
                            context_window=128000 if "128k" in id else 128000 if "gpt-4" in id else 4096,
                            capabilities=capabilities
                        ))
                    
                    return models
                else:
                    print(f"Error listing OpenAI models: {response.status}")
                    return []
        except Exception as e:
            print(f"Error connecting to OpenAI: {e}")
            return []
    
    async def generate(self, 
                     prompt: str, 
                     model: str, 
                     max_tokens: int = 1024,
                     temperature: float = 0.7,
                     stream: bool = False) -> AsyncIterator[str]:
        """Generate text from OpenAI model."""
        # For OpenAI, we'll use the chat endpoint as completions are deprecated
        messages = [{"role": "user", "content": prompt}]
        async for chunk in self.chat(messages, model, max_tokens, temperature, stream):
            yield chunk
    
    async def chat(self,
                  messages: List[Dict[str, str]],
                  model: str,
                  max_tokens: int = 1024,
                  temperature: float = 0.7,
                  stream: bool = False) -> AsyncIterator[str]:
        """Chat with OpenAI model."""
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream
        }
        
        if stream:
            async with self.session.post(f"{self.base_url}/chat/completions", headers=headers, json=payload) as response:
                async for line in response.content:
                    if line.strip():
                        try:
                            line_str = line.decode('utf-8')
                            if line_str.startswith("data: "):
                                data_str = line_str[6:]  # Remove "data: " prefix
                                if data_str.strip() == "[DONE]":
                                    break
                                data = json.loads(data_str)
                                
                                choices = data.get("choices", [])
                                if choices:
                                    delta = choices[0].get("delta", {})
                                    content = delta.get("content", "")
                                    if content:
                                        yield content
                        except (json.JSONDecodeError, IndexError):
                            continue
        else:
            async with self.session.post(f"{self.base_url}/chat/completions", headers=headers, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    content = data["choices"][0]["message"]["content"]
                    yield content
                else:
                    error_data = await response.json()
                    yield f"Error: {error_data.get('error', {}).get('message', 'Unknown error')}"
    
    def get_provider_name(self) -> str:
        """Get the name of this provider."""
        return "openai"


class AnthropicProvider(ModelProvider):
    """Anthropic model provider implementation."""
    
    def __init__(self, api_key: str, base_url: str = "https://api.anthropic.com/v1"):
        super().__init__(api_key, base_url)
        self.api_version = "2023-06-01"  # Anthropic API version
    
    async def list_models(self) -> List[ModelInfo]:
        """List available models from Anthropic."""
        # Anthropic doesn't have a public models endpoint, so we'll return known models
        known_models = [
            ModelInfo(
                name="claude-3-opus-20240229",
                provider="anthropic",
                max_tokens=4096,
                context_window=200000,
                capabilities=["text", "chat", "instruct"],
                pricing={"input": 0.015, "output": 0.075}
            ),
            ModelInfo(
                name="claude-3-sonnet-20240229",
                provider="anthropic",
                max_tokens=4096,
                context_window=200000,
                capabilities=["text", "chat", "instruct"],
                pricing={"input": 0.003, "output": 0.015}
            ),
            ModelInfo(
                name="claude-3-haiku-20240307",
                provider="anthropic",
                max_tokens=4096,
                context_window=200000,
                capabilities=["text", "chat", "instruct"],
                pricing={"input": 0.00025, "output": 0.00125}
            )
        ]
        return known_models
    
    async def generate(self, 
                     prompt: str, 
                     model: str, 
                     max_tokens: int = 1024,
                     temperature: float = 0.7,
                     stream: bool = False) -> AsyncIterator[str]:
        """Generate text from Anthropic model."""
        # For Anthropic, we'll format the prompt as needed
        formatted_prompt = f"\n\nHuman: {prompt}\n\nAssistant:"
        messages = [{"role": "user", "content": prompt}]
        async for chunk in self.chat(messages, model, max_tokens, temperature, stream):
            yield chunk
    
    async def chat(self,
                  messages: List[Dict[str, str]],
                  model: str,
                  max_tokens: int = 1024,
                  temperature: float = 0.7,
                  stream: bool = False) -> AsyncIterator[str]:
        """Chat with Anthropic model."""
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": self.api_version
        }
        
        # Convert messages to Anthropic format
        system_message = ""
        user_assistant_messages = []
        
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                system_message = content
            else:
                user_assistant_messages.append(msg)
        
        payload = {
            "model": model,
            "messages": user_assistant_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream
        }
        
        if system_message:
            payload["system"] = system_message
        
        if stream:
            async with self.session.post(f"{self.base_url}/messages", headers=headers, json=payload) as response:
                buffer = ""
                async for line in response.content:
                    if line.strip():
                        try:
                            line_str = line.decode('utf-8')
                            if line_str.startswith("data: "):
                                data_str = line_str[6:]  # Remove "data: " prefix
                                if data_str.strip() == "[DONE]":
                                    break
                                data = json.loads(data_str)
                                
                                if data.get("type") == "content_block_delta":
                                    text = data.get("delta", {}).get("text", "")
                                    if text:
                                        yield text
                        except (json.JSONDecodeError, IndexError):
                            continue
        else:
            async with self.session.post(f"{self.base_url}/messages", headers=headers, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    content_blocks = data.get("content", [])
                    full_response = ""
                    for block in content_blocks:
                        if block.get("type") == "text":
                            full_response += block.get("text", "")
                    yield full_response
                else:
                    error_data = await response.json()
                    yield f"Error: {error_data.get('error', {}).get('message', 'Unknown error')}"
    
    def get_provider_name(self) -> str:
        """Get the name of this provider."""
        return "anthropic"


class HuggingFaceProvider(ModelProvider):
    """Hugging Face model provider implementation."""

    def __init__(self, api_key: str, base_url: str = "https://api-inference.huggingface.co/models"):
        super().__init__(api_key, base_url)

    async def list_models(self) -> List[ModelInfo]:
        """List available models from Hugging Face (top models)."""
        # For now, return some popular models
        # In a real implementation, this would call the HF API to search models
        popular_models = [
            ModelInfo(
                name="meta-llama/Llama-2-7b-chat-hf",
                provider="huggingface",
                max_tokens=4096,
                context_window=4096,
                capabilities=["text", "chat", "instruct"]
            ),
            ModelInfo(
                name="google/gemma-7b-it",
                provider="huggingface",
                max_tokens=2048,
                context_window=2048,
                capabilities=["text", "chat", "instruct"]
            ),
            ModelInfo(
                name="mistralai/Mistral-7B-Instruct-v0.1",
                provider="huggingface",
                max_tokens=4096,
                context_window=4096,
                capabilities=["text", "instruct"]
            )
        ]
        return popular_models

    async def generate(self,
                     prompt: str,
                     model: str,
                     max_tokens: int = 1024,
                     temperature: float = 0.7,
                     stream: bool = False) -> AsyncIterator[str]:
        """Generate text from Hugging Face model."""
        if not self.session:
            self.session = aiohttp.ClientSession()

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "return_full_text": False
            }
        }

        # Note: Hugging Face Inference API doesn't support streaming for all models
        # This is a simplified implementation
        async with self.session.post(f"{self.base_url}/{model}", headers=headers, json=payload) as response:
            if response.status == 200:
                data = await response.json()
                if isinstance(data, list) and len(data) > 0:
                    generated_text = data[0].get("generated_text", "")
                    yield generated_text
                else:
                    yield str(data)
            else:
                yield f"Error: {response.status}"

    async def chat(self,
                  messages: List[Dict[str, str]],
                  model: str,
                  max_tokens: int = 1024,
                  temperature: float = 0.7,
                  stream: bool = False) -> AsyncIterator[str]:
        """Chat with Hugging Face model."""
        # Format messages for HF models (many expect specific formats)
        formatted_prompt = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                formatted_prompt += f"User: {content}\n\nAssistant:"
            elif role == "assistant":
                formatted_prompt += f" {content}\n\n"
            elif role == "system":
                formatted_prompt = f"<<SYS>>\n{content}\n<</SYS>>\n\n" + formatted_prompt

        async for chunk in self.generate(formatted_prompt, model, max_tokens, temperature, stream):
            yield chunk

    def get_provider_name(self) -> str:
        """Get the name of this provider."""
        return "huggingface"


class GoogleGeminiProvider(ModelProvider):
    """Google Gemini model provider implementation."""

    def __init__(self, api_key: str, base_url: str = "https://generativelanguage.googleapis.com/v1beta"):
        super().__init__(api_key, base_url)

    async def list_models(self) -> List[ModelInfo]:
        """List available models from Google Gemini."""
        if not self.session:
            self.session = aiohttp.ClientSession()

        try:
            async with self.session.get(f"{self.base_url}/models?key={self.api_key}") as response:
                if response.status == 200:
                    data = await response.json()
                    models = []

                    for model_data in data.get("models", []):
                        name = model_data.get("name", "").replace("models/", "")  # Remove "models/" prefix
                        # Extract capabilities from model info
                        capabilities = ["text"]
                        if "generateContent" in model_data.get("supportedGenerationMethods", []):
                            capabilities.append("chat")
                        if "code" in model_data.get("displayName", "").lower() or "code" in name.lower():
                            capabilities.append("code")

                        # Estimate context window and max tokens based on model name
                        context_window = 32768  # Default for most Gemini models
                        if "1.5" in name:
                            context_window = 1048576  # Gemini 1.5 has 1M+ context
                        elif "flash" in name:
                            context_window = 32768

                        max_tokens = 8192  # Default max output tokens
                        if "1.5" in name:
                            max_tokens = 8192

                        models.append(ModelInfo(
                            name=name,
                            provider="google_gemini",
                            max_tokens=max_tokens,
                            context_window=context_window,
                            capabilities=capabilities
                        ))

                    return models
                else:
                    print(f"Error listing Gemini models: {response.status}")
                    return []
        except Exception as e:
            print(f"Error connecting to Google Gemini: {e}")
            return []

    async def generate(self,
                     prompt: str,
                     model: str,
                     max_tokens: int = 1024,
                     temperature: float = 0.7,
                     stream: bool = False) -> AsyncIterator[str]:
        """Generate text from Google Gemini model."""
        # For Gemini, we'll use the chat endpoint as it's more versatile
        messages = [{"role": "user", "parts": [{"text": prompt}]}]
        async for chunk in self.chat([{"role": "user", "content": prompt}], model, max_tokens, temperature, stream):
            yield chunk

    async def chat(self,
                  messages: List[Dict[str, str]],
                  model: str,
                  max_tokens: int = 1024,
                  temperature: float = 0.7,
                  stream: bool = False) -> AsyncIterator[str]:
        """Chat with Google Gemini model."""
        if not self.session:
            self.session = aiohttp.ClientSession()

        # Convert messages to Gemini format
        gemini_messages = []
        for msg in messages:
            role = "user" if msg["role"] in ["user", "system"] else "model"  # Gemini uses "user" and "model"
            content = msg["content"]

            # For system messages, we'll include them as part of the first user message
            if msg["role"] == "system":
                if gemini_messages and gemini_messages[0]["role"] == "user":
                    # Append to the first user message
                    if gemini_messages[0]["parts"]:
                        gemini_messages[0]["parts"][0]["text"] = f"{content}\n\n{gemini_messages[0]['parts'][0]['text']}"
                    else:
                        gemini_messages[0]["parts"].append({"text": content})
                else:
                    # If no user message exists yet, create one with system content
                    gemini_messages.append({
                        "role": "user",
                        "parts": [{"text": content}]
                    })
            else:
                gemini_messages.append({
                    "role": role,
                    "parts": [{"text": content}]
                })

        # Prepare the request payload
        payload = {
            "contents": gemini_messages,
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": temperature,
                "stopSequences": [],
                "candidateCount": 1
            },
            "safetySettings": [
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                }
            ]
        }

        if stream:
            # For streaming, we need to call the streaming endpoint
            url = f"{self.base_url}/models/{model}:streamGenerateContent?key={self.api_key}"
            # Note: Streaming with Gemini is more complex, so we'll implement non-streaming for now
            # and yield the full response at once
            async with self.session.post(f"{self.base_url}/models/{model}:generateContent?key={self.api_key}",
                                       json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    candidates = data.get("candidates", [])
                    if candidates:
                        content_parts = candidates[0].get("content", {}).get("parts", [])
                        full_response = ""
                        for part in content_parts:
                            if "text" in part:
                                full_response += part["text"]
                        yield full_response
                else:
                    error_data = await response.json()
                    yield f"Error: {error_data.get('error', {}).get('message', 'Unknown error')}"
        else:
            async with self.session.post(f"{self.base_url}/models/{model}:generateContent?key={self.api_key}",
                                       json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    candidates = data.get("candidates", [])
                    if candidates:
                        content_parts = candidates[0].get("content", {}).get("parts", [])
                        full_response = ""
                        for part in content_parts:
                            if "text" in part:
                                full_response += part["text"]
                        yield full_response
                else:
                    error_data = await response.json()
                    yield f"Error: {error_data.get('error', {}).get('message', 'Unknown error')}"

    def get_provider_name(self) -> str:
        """Get the name of this provider."""
        return "google_gemini"


class OpenRouterProvider(ModelProvider):
    """OpenRouter model provider implementation."""

    def __init__(self, api_key: str, base_url: str = "https://openrouter.ai/api/v1"):
        super().__init__(api_key, base_url)

    async def list_models(self) -> List[ModelInfo]:
        """List available models from OpenRouter."""
        if not self.session:
            self.session = aiohttp.ClientSession()

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        try:
            async with self.session.get(f"{self.base_url}/models", headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    models = []

                    for model_data in data.get("data", []):
                        id = model_data.get("id", "")
                        context_length = model_data.get("context_length", 4096)
                        pricing = model_data.get("pricing", {})

                        # Convert pricing from OpenRouter format to our format ($ per 1M tokens -> $ per 1K tokens)
                        converted_pricing = {}
                        if pricing:
                            # OpenRouter pricing is per million tokens, convert to per thousand
                            input_cost = float(pricing.get("prompt", "0")) / 1000.0
                            output_cost = float(pricing.get("completion", "0")) / 1000.0
                            converted_pricing = {"input": input_cost, "output": output_cost}

                        # Determine capabilities based on model characteristics
                        capabilities = ["text"]
                        if "instruct" in id.lower() or "chat" in id.lower():
                            capabilities.append("chat")
                        if "code" in id.lower():
                            capabilities.append("code")

                        models.append(ModelInfo(
                            name=id,
                            provider="openrouter",
                            max_tokens=context_length // 2,  # Conservative max output tokens
                            context_window=context_length,
                            capabilities=capabilities,
                            pricing=converted_pricing
                        ))

                    return models
                else:
                    print(f"Error listing OpenRouter models: {response.status}")
                    return []
        except Exception as e:
            print(f"Error connecting to OpenRouter: {e}")
            return []

    async def generate(self,
                     prompt: str,
                     model: str,
                     max_tokens: int = 1024,
                     temperature: float = 0.7,
                     stream: bool = False) -> AsyncIterator[str]:
        """Generate text from OpenRouter model."""
        # For OpenRouter, we'll use the chat endpoint as most models support it
        messages = [{"role": "user", "content": prompt}]
        async for chunk in self.chat(messages, model, max_tokens, temperature, stream):
            yield chunk

    async def chat(self,
                  messages: List[Dict[str, str]],
                  model: str,
                  max_tokens: int = 1024,
                  temperature: float = 0.7,
                  stream: bool = False) -> AsyncIterator[str]:
        """Chat with OpenRouter model."""
        if not self.session:
            self.session = aiohttp.ClientSession()

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream
        }

        if stream:
            async with self.session.post(f"{self.base_url}/chat/completions", headers=headers, json=payload) as response:
                async for line in response.content:
                    if line.strip():
                        try:
                            line_str = line.decode('utf-8')
                            if line_str.startswith("data: "):
                                data_str = line_str[6:]  # Remove "data: " prefix
                                if data_str.strip() == "[DONE]":
                                    break
                                data = json.loads(data_str)

                                choices = data.get("choices", [])
                                if choices:
                                    delta = choices[0].get("delta", {})
                                    content = delta.get("content", "")
                                    if content:
                                        yield content
                        except (json.JSONDecodeError, IndexError):
                            continue
        else:
            async with self.session.post(f"{self.base_url}/chat/completions", headers=headers, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    content = data["choices"][0]["message"]["content"]
                    yield content
                else:
                    error_data = await response.json()
                    yield f"Error: {error_data.get('error', {}).get('message', 'Unknown error')}"

    def get_provider_name(self) -> str:
        """Get the name of this provider."""
        return "openrouter"


class ModelProviderManager:
    """Manages multiple model providers."""
    
    def __init__(self):
        self.providers: Dict[str, ModelProvider] = {}
        self.provider_configs: Dict[str, Dict[str, str]] = {}
    
    def register_provider(self, name: str, provider: ModelProvider):
        """Register a model provider."""
        self.providers[name] = provider
    
    def configure_provider(self, name: str, api_key: str, base_url: Optional[str] = None):
        """Configure a model provider."""
        self.provider_configs[name] = {
            "api_key": api_key,
            "base_url": base_url or ""
        }
    
    def get_provider(self, name: str) -> Optional[ModelProvider]:
        """Get a model provider by name."""
        return self.providers.get(name)
    
    async def initialize_providers(self):
        """Initialize all configured providers."""
        for name, config in self.provider_configs.items():
            api_key = config["api_key"]
            base_url = config.get("base_url")

            if name == "ollama":
                provider = OllamaProvider(api_key, base_url or "http://localhost:11434")
            elif name == "openai":
                provider = OpenAIProvider(api_key, base_url or "https://api.openai.com/v1")
            elif name == "anthropic":
                provider = AnthropicProvider(api_key, base_url or "https://api.anthropic.com/v1")
            elif name == "huggingface":
                provider = HuggingFaceProvider(api_key, base_url or "https://api-inference.huggingface.co/models")
            elif name == "google_gemini":
                provider = GoogleGeminiProvider(api_key, base_url or "https://generativelanguage.googleapis.com/v1beta")
            elif name == "openrouter":
                provider = OpenRouterProvider(api_key, base_url or "https://openrouter.ai/api/v1")
            else:
                print(f"Unknown provider: {name}")
                continue

            self.register_provider(name, provider)
    
    async def list_all_models(self) -> Dict[str, List[ModelInfo]]:
        """List models from all providers."""
        all_models = {}
        
        for name, provider in self.providers.items():
            try:
                models = await provider.list_models()
                all_models[name] = models
            except Exception as e:
                print(f"Error listing models for {name}: {e}")
                all_models[name] = []
        
        return all_models
    
    async def generate_with_provider(self, 
                                   prompt: str, 
                                   provider_name: str, 
                                   model: str, 
                                   **kwargs) -> str:
        """Generate text using a specific provider."""
        provider = self.get_provider(provider_name)
        if not provider:
            return f"Provider {provider_name} not found"
        
        full_response = ""
        async for chunk in provider.generate(prompt, model, **kwargs):
            full_response += chunk
        
        return full_response
    
    async def chat_with_provider(self,
                                messages: List[Dict[str, str]],
                                provider_name: str,
                                model: str,
                                **kwargs) -> str:
        """Chat using a specific provider."""
        provider = self.get_provider(provider_name)
        if not provider:
            return f"Provider {provider_name} not found"
        
        full_response = ""
        async for chunk in provider.chat(messages, model, **kwargs):
            full_response += chunk
        
        return full_response


# Global manager instance
_provider_manager = ModelProviderManager()


def get_model_provider_manager() -> ModelProviderManager:
    """Get the global model provider manager."""
    return _provider_manager


__all__ = [
    'ModelInfo',
    'ModelProvider',
    'OllamaProvider',
    'OpenAIProvider',
    'AnthropicProvider',
    'HuggingFaceProvider',
    'GoogleGeminiProvider',
    'OpenRouterProvider',
    'ModelProviderManager',
    'get_model_provider_manager'
]