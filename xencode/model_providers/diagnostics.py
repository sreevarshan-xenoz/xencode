#!/usr/bin/env python3
"""
Provider Diagnostics Service

Provides connectivity and authentication tests for AI providers:
- Qwen (OAuth2)
- OpenRouter (API key)
- Ollama (local service)

Returns latency, endpoint status, and auth readiness with friendly remediation hints.
"""

import asyncio
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any, List
import aiohttp
from rich.console import Console

console = Console()


class ProviderStatus(Enum):
    """Provider connection status"""
    OK = "ok"
    AUTH_ERROR = "auth_error"
    CONNECTION_ERROR = "connection_error"
    NOT_CONFIGURED = "not_configured"
    UNKNOWN = "unknown"


@dataclass
class ProviderTestResult:
    """Result of a provider connectivity test"""
    provider: str
    status: ProviderStatus
    latency_ms: Optional[float] = None
    endpoint: Optional[str] = None
    model_count: Optional[int] = None
    error_message: Optional[str] = None
    remediation: Optional[str] = None
    details: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.details is None:
            self.details = {}
    
    @property
    def is_success(self) -> bool:
        """Check if test was successful"""
        return self.status == ProviderStatus.OK
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for display/logging"""
        return {
            "provider": self.provider,
            "status": self.status.value,
            "latency_ms": self.latency_ms,
            "endpoint": self.endpoint,
            "model_count": self.model_count,
            "error_message": self.error_message,
            "remediation": self.remediation,
            "is_success": self.is_success,
            **self.details
        }


class ProviderDiagnostics:
    """
    Diagnostic service for testing provider connectivity
    
    Usage:
        diagnostics = ProviderDiagnostics()
        result = await diagnostics.test_qwen()
        result = await diagnostics.test_openrouter(api_key)
        result = await diagnostics.test_ollama()
    """
    
    # Provider endpoints
    QWEN_BASE_URL = "https://chat.qwen.ai/api/v1"
    QWEN_MODELS_URL = f"{QWEN_BASE_URL}/models"
    QWEN_CREDS_FILE = None  # Will be set dynamically
    
    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
    OPENROUTER_MODELS_URL = f"{OPENROUTER_BASE_URL}/models"
    
    OLLAMA_BASE_URL = "http://localhost:11434"
    OLLAMA_TAGS_URL = f"{OLLAMA_BASE_URL}/api/tags"
    
    def __init__(self):
        self._qwen_creds_file = None
    
    def set_qwen_creds_file(self, creds_file):
        """Set the path to Qwen credentials file"""
        self._qwen_creds_file = creds_file
    
    async def test_qwen(self) -> ProviderTestResult:
        """
        Test Qwen connectivity and authentication
        
        Returns:
            ProviderTestResult with connection status
        """
        start_time = time.time()
        
        try:
            # Try to load cached credentials
            access_token = self._load_qwen_token()
            
            if not access_token:
                return ProviderTestResult(
                    provider="Qwen",
                    status=ProviderStatus.NOT_CONFIGURED,
                    endpoint=self.QWEN_BASE_URL,
                    error_message="No Qwen authentication found",
                    remediation="Click 'Login' to authenticate with Qwen",
                )
            
            # Test the models endpoint
            headers = {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
            }
            
            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(self.QWEN_MODELS_URL, headers=headers) as response:
                    latency_ms = (time.time() - start_time) * 1000
                    
                    if response.status == 200:
                        data = await response.json()
                        models = data.get("data", []) if isinstance(data, dict) else []
                        
                        return ProviderTestResult(
                            provider="Qwen",
                            status=ProviderStatus.OK,
                            latency_ms=latency_ms,
                            endpoint=self.QWEN_BASE_URL,
                            model_count=len(models) if models else None,
                            details={"models_found": len(models)},
                        )
                    elif response.status == 401:
                        return ProviderTestResult(
                            provider="Qwen",
                            status=ProviderStatus.AUTH_ERROR,
                            latency_ms=latency_ms,
                            endpoint=self.QWEN_BASE_URL,
                            error_message="Authentication failed or token expired",
                            remediation="Click 'Login' to re-authenticate with Qwen",
                        )
                    else:
                        error_text = await response.text()
                        return ProviderTestResult(
                            provider="Qwen",
                            status=ProviderStatus.UNKNOWN,
                            latency_ms=latency_ms,
                            endpoint=self.QWEN_BASE_URL,
                            error_message=f"HTTP {response.status}",
                            remediation="Check your internet connection and try again",
                            details={"raw_error": error_text[:200]},
                        )
                        
        except aiohttp.ClientConnectionError as e:
            latency_ms = (time.time() - start_time) * 1000
            return ProviderTestResult(
                provider="Qwen",
                status=ProviderStatus.CONNECTION_ERROR,
                latency_ms=latency_ms,
                endpoint=self.QWEN_BASE_URL,
                error_message=f"Cannot connect: {str(e)}",
                remediation="Check your internet connection",
            )
        except asyncio.TimeoutError:
            latency_ms = (time.time() - start_time) * 1000
            return ProviderTestResult(
                provider="Qwen",
                status=ProviderStatus.CONNECTION_ERROR,
                latency_ms=latency_ms,
                endpoint=self.QWEN_BASE_URL,
                error_message="Request timed out",
                remediation="Check your internet connection or try again later",
            )
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return ProviderTestResult(
                provider="Qwen",
                status=ProviderStatus.UNKNOWN,
                latency_ms=latency_ms,
                endpoint=self.QWEN_BASE_URL,
                error_message=str(e),
                remediation="An unexpected error occurred",
            )
    
    async def test_openrouter(self, api_key: Optional[str] = None) -> ProviderTestResult:
        """
        Test OpenRouter connectivity and API key
        
        Args:
            api_key: OpenRouter API key (sk-or-v1-...)
            
        Returns:
            ProviderTestResult with connection status
        """
        start_time = time.time()
        
        try:
            # Check if API key is provided
            if not api_key:
                return ProviderTestResult(
                    provider="OpenRouter",
                    status=ProviderStatus.NOT_CONFIGURED,
                    endpoint=self.OPENROUTER_BASE_URL,
                    error_message="No API key configured",
                    remediation="Enter your OpenRouter API key and click 'Save'",
                )
            
            # Validate API key format
            if not api_key.startswith("sk-or-v"):
                return ProviderTestResult(
                    provider="OpenRouter",
                    status=ProviderStatus.NOT_CONFIGURED,
                    endpoint=self.OPENROUTER_BASE_URL,
                    error_message="Invalid API key format",
                    remediation="OpenRouter API keys start with 'sk-or-v'",
                )
            
            # Test the models endpoint
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            
            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(self.OPENROUTER_MODELS_URL, headers=headers) as response:
                    latency_ms = (time.time() - start_time) * 1000
                    
                    if response.status == 200:
                        data = await response.json()
                        models = data.get("data", []) if isinstance(data, dict) else []
                        
                        return ProviderTestResult(
                            provider="OpenRouter",
                            status=ProviderStatus.OK,
                            latency_ms=latency_ms,
                            endpoint=self.OPENROUTER_BASE_URL,
                            model_count=len(models) if models else None,
                            details={"models_found": len(models)},
                        )
                    elif response.status == 401:
                        return ProviderTestResult(
                            provider="OpenRouter",
                            status=ProviderStatus.AUTH_ERROR,
                            latency_ms=latency_ms,
                            endpoint=self.OPENROUTER_BASE_URL,
                            error_message="Invalid API key",
                            remediation="Check your API key and try again",
                        )
                    elif response.status == 403:
                        return ProviderTestResult(
                            provider="OpenRouter",
                            status=ProviderStatus.AUTH_ERROR,
                            latency_ms=latency_ms,
                            endpoint=self.OPENROUTER_BASE_URL,
                            error_message="API key has insufficient permissions",
                            remediation="Check your API key permissions on openrouter.ai",
                        )
                    else:
                        error_text = await response.text()
                        return ProviderTestResult(
                            provider="OpenRouter",
                            status=ProviderStatus.UNKNOWN,
                            latency_ms=latency_ms,
                            endpoint=self.OPENROUTER_BASE_URL,
                            error_message=f"HTTP {response.status}",
                            remediation="Check your internet connection and try again",
                            details={"raw_error": error_text[:200]},
                        )
                        
        except aiohttp.ClientConnectionError as e:
            latency_ms = (time.time() - start_time) * 1000
            return ProviderTestResult(
                provider="OpenRouter",
                status=ProviderStatus.CONNECTION_ERROR,
                latency_ms=latency_ms,
                endpoint=self.OPENROUTER_BASE_URL,
                error_message=f"Cannot connect: {str(e)}",
                remediation="Check your internet connection",
            )
        except asyncio.TimeoutError:
            latency_ms = (time.time() - start_time) * 1000
            return ProviderTestResult(
                provider="OpenRouter",
                status=ProviderStatus.CONNECTION_ERROR,
                latency_ms=latency_ms,
                endpoint=self.OPENROUTER_BASE_URL,
                error_message="Request timed out",
                remediation="Check your internet connection or try again later",
            )
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return ProviderTestResult(
                provider="OpenRouter",
                status=ProviderStatus.UNKNOWN,
                latency_ms=latency_ms,
                endpoint=self.OPENROUTER_BASE_URL,
                error_message=str(e),
                remediation="An unexpected error occurred",
            )
    
    async def test_ollama(self) -> ProviderTestResult:
        """
        Test Ollama local service connectivity
        
        Returns:
            ProviderTestResult with connection status
        """
        start_time = time.time()
        
        try:
            timeout = aiohttp.ClientTimeout(total=5)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(self.OLLAMA_TAGS_URL) as response:
                    latency_ms = (time.time() - start_time) * 1000
                    
                    if response.status == 200:
                        data = await response.json()
                        models = data.get("models", []) if isinstance(data, dict) else []
                        
                        return ProviderTestResult(
                            provider="Ollama",
                            status=ProviderStatus.OK,
                            latency_ms=latency_ms,
                            endpoint=self.OLLAMA_BASE_URL,
                            model_count=len(models) if models else 0,
                            details={
                                "models_found": len(models),
                                "local": True,
                            },
                        )
                    else:
                        return ProviderTestResult(
                            provider="Ollama",
                            status=ProviderStatus.UNKNOWN,
                            latency_ms=latency_ms,
                            endpoint=self.OLLAMA_BASE_URL,
                            error_message=f"HTTP {response.status}",
                            remediation="Ollama service may not be running properly",
                        )
                        
        except aiohttp.ClientConnectionError:
            latency_ms = (time.time() - start_time) * 1000
            return ProviderTestResult(
                provider="Ollama",
                status=ProviderStatus.CONNECTION_ERROR,
                latency_ms=latency_ms,
                endpoint=self.OLLAMA_BASE_URL,
                error_message="Cannot connect to Ollama service",
                remediation="Start Ollama with 'ollama serve' or install from ollama.ai",
            )
        except asyncio.TimeoutError:
            latency_ms = (time.time() - start_time) * 1000
            return ProviderTestResult(
                provider="Ollama",
                status=ProviderStatus.CONNECTION_ERROR,
                latency_ms=latency_ms,
                endpoint=self.OLLAMA_BASE_URL,
                error_message="Request timed out",
                remediation="Ollama service is not responding",
            )
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return ProviderTestResult(
                provider="Ollama",
                status=ProviderStatus.UNKNOWN,
                latency_ms=latency_ms,
                endpoint=self.OLLAMA_BASE_URL,
                error_message=str(e),
                remediation="An unexpected error occurred",
            )
    
    def _load_qwen_token(self) -> Optional[str]:
        """Load Qwen access token from credentials file"""
        import json
        from pathlib import Path
        
        try:
            creds_file = self._qwen_creds_file or Path.home() / ".xencode_qwen_creds.json"
            if not creds_file.exists():
                return None
            
            with open(creds_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Check if token is still valid (with 5-minute buffer)
            import time
            created_at = data.get('created_at', 0)
            expires_in = data.get('expires_in', 0)
            elapsed = time.time() - created_at
            
            if elapsed >= (expires_in - 300):  # 5 minutes buffer
                return None  # Token expired
            
            return data.get('access_token')
            
        except (json.JSONDecodeError, IOError, KeyError):
            return None
    
    async def test_all_providers(
        self,
        openrouter_api_key: Optional[str] = None,
    ) -> List[ProviderTestResult]:
        """
        Test all providers concurrently
        
        Args:
            openrouter_api_key: Optional OpenRouter API key
            
        Returns:
            List of ProviderTestResult for each provider
        """
        tasks = [
            self.test_qwen(),
            self.test_openrouter(openrouter_api_key),
            self.test_ollama(),
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert any exceptions to error results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                provider_names = ["Qwen", "OpenRouter", "Ollama"]
                processed_results.append(ProviderTestResult(
                    provider=provider_names[i],
                    status=ProviderStatus.UNKNOWN,
                    error_message=str(result),
                    remediation="An unexpected error occurred during testing",
                ))
            else:
                processed_results.append(result)
        
        return processed_results


# Global instance
_diagnostics: Optional[ProviderDiagnostics] = None


def get_diagnostics() -> ProviderDiagnostics:
    """Get or create global diagnostics instance"""
    global _diagnostics
    if _diagnostics is None:
        _diagnostics = ProviderDiagnostics()
    return _diagnostics


if __name__ == "__main__":
    # Provider Diagnostics - Run with --demo flag for testing
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        async def demo():
            console.print("[bold blue]Provider Diagnostics Test[/bold blue]\n")

            diagnostics = get_diagnostics()

            # Test all providers
            results = await diagnostics.test_all_providers()

            for result in results:
                status_icon = "OK" if result.is_success else "FAIL"
                console.print(f"\n{status_icon} **{result.provider}**")
                console.print(f"   Status: {result.status.value}")

                if result.latency_ms:
                    console.print(f"   Latency: {result.latency_ms:.0f}ms")

                if result.endpoint:
                    console.print(f"   Endpoint: {result.endpoint}")

                if result.model_count is not None:
                    console.print(f"   Models: {result.model_count}")

                if result.error_message:
                    console.print(f"   Error: {result.error_message}")

                if result.remediation:
                    console.print(f"   Hint: {result.remediation}")
        
        import asyncio
        asyncio.run(demo())
    else:
        print("Provider Diagnostics module")
        print("Usage: python -m xencode.model_providers.diagnostics --demo")
