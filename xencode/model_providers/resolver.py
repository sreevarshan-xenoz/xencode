#!/usr/bin/env python3
"""
Locked Model Resolver for Cloud Coder

Implements locked model resolution for cloud coder with:
- Default to qwen3-coder-next-instruct (or qwen-coder-next-latest)
- User override via config
- Alias resolution fallback
- Model ID verification via /v1/models post-auth
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, List, Set
from enum import Enum
import aiohttp
from rich.console import Console

console = Console()


class ModelAlias:
    """
    Model alias constants and canonical mappings
    
    Usage:
        canonical = ModelAlias.CANONICAL_MAP.get("qwen-coder-next-latest")
    """
    # Known model aliases
    QWEN_CODER_NEXT_LATEST = "qwen-coder-next-latest"
    QWEN3_CODER_NEXT_INSTRUCT = "qwen3-coder-next-instruct"
    QWEN_CODER_PLUS = "qwen-coder-plus"
    QWEN_MAX_CODER = "qwen-max-coder"
    
    # Canonical mappings (may change over time)
    CANONICAL_MAP: Dict[str, str] = {
        "qwen-coder-next-latest": "qwen3-coder-next-instruct",
        "qwen3-coder-next": "qwen3-coder-next-instruct",
        "qwen-coder-next": "qwen3-coder-next-instruct",
    }


@dataclass
class ModelInfo:
    """Information about a discovered model"""
    model_id: str
    alias: Optional[str] = None
    capabilities: List[str] = field(default_factory=list)
    context_window: Optional[int] = None
    max_tokens: Optional[int] = None
    discovered_at: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "alias": self.alias,
            "capabilities": self.capabilities,
            "context_window": self.context_window,
            "max_tokens": self.max_tokens,
            "discovered_at": self.discovered_at,
        }


@dataclass
class ResolverConfig:
    """
    Configuration for locked model resolver
    
    Attributes:
        lock_best_coder: If True, always resolve to best coder model
        lock_model_override: Optional user override model ID
        cache_ttl: Cache time-to-live in seconds
        verify_on_resolve: If True, verify model exists via /v1/models
    """
    lock_best_coder: bool = True
    lock_model_override: Optional[str] = None
    cache_ttl: int = 3600  # 1 hour
    verify_on_resolve: bool = True
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ResolverConfig':
        return cls(
            lock_best_coder=data.get('lock_best_coder', True),
            lock_model_override=data.get('lock_model_override'),
            cache_ttl=data.get('cache_ttl', 3600),
            verify_on_resolve=data.get('verify_on_resolve', True),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'lock_best_coder': self.lock_best_coder,
            'lock_model_override': self.lock_model_override,
            'cache_ttl': self.cache_ttl,
            'verify_on_resolve': self.verify_on_resolve,
        }


class ModelDiscoveryCache:
    """
    Cache for discovered models from /v1/models
    
    Provides TTL-based invalidation and fast lookups
    """
    
    def __init__(self, ttl: int = 3600):
        self.ttl = ttl
        self._models: List[ModelInfo] = []
        self._discovered_at: float = 0
        self._alias_map: Dict[str, str] = {}
    
    def is_valid(self) -> bool:
        """Check if cache is still valid"""
        if not self._models:
            return False
        return (time.time() - self._discovered_at) < self.ttl
    
    def invalidate(self):
        """Invalidate the cache"""
        self._models = []
        self._discovered_at = 0
        self._alias_map = {}
    
    def set_models(self, models: List[ModelInfo]):
        """Set discovered models and rebuild alias map"""
        self._models = models
        self._discovered_at = time.time()
        
        # Build alias map
        self._alias_map = {}
        for model in models:
            self._alias_map[model.model_id.lower()] = model.model_id
            if model.alias:
                self._alias_map[model.alias.lower()] = model.model_id
    
    def get_model(self, model_id_or_alias: str) -> Optional[ModelInfo]:
        """Get model by ID or alias"""
        if not self.is_valid():
            return None
        
        key = model_id_or_alias.lower()
        canonical_id = self._alias_map.get(key)
        
        if not canonical_id:
            return None
        
        for model in self._models:
            if model.model_id == canonical_id:
                return model
        
        return None
    
    def get_coder_models(self) -> List[ModelInfo]:
        """Get all coder-capable models"""
        if not self.is_valid():
            return []
        
        return [
            m for m in self._models
            if any(cap in m.capabilities for cap in ['code', 'coder', 'coding'])
        ]
    
    def find_best_coder_model(self) -> Optional[ModelInfo]:
        """
        Find the best coder model based on priority
        
        Priority order:
        1. qwen3-coder-next-instruct (or canonical equivalent)
        2. qwen-coder-next-latest
        3. Any model with 'coder' and 'next' in name
        4. Any model with 'coder' in name
        """
        if not self.is_valid():
            return None
        
        coder_models = self.get_coder_models()
        
        if not coder_models:
            return None
        
        # Priority 1: Next instruct
        for model in coder_models:
            if 'qwen3-coder-next-instruct' in model.model_id.lower():
                return model
        
        # Priority 2: Next latest
        for model in coder_models:
            if 'qwen-coder-next-latest' in model.model_id.lower():
                return model
        
        # Priority 3: Next variant
        for model in coder_models:
            if 'next' in model.model_id.lower() and 'coder' in model.model_id.lower():
                return model
        
        # Priority 4: Any coder model
        return coder_models[0]


class LockedModelResolver:
    """
    Locked model resolver for cloud coder
    
    Resolves model requests to the best available coder model,
    with support for user overrides and alias drift handling.
    
    Usage:
        resolver = LockedModelResolver(config=ResolverConfig(...))
        model_id = await resolver.resolve_coder_model()
    """
    
    # Default best coder model
    DEFAULT_CODER_MODEL = "qwen3-coder-next-instruct"
    
    # Alternative default (fallback)
    FALLBACK_CODER_MODEL = "qwen-coder-next-latest"
    
    # Qwen API endpoints
    QWEN_BASE_URL = "https://chat.qwen.ai/api/v1"
    QWEN_MODELS_URL = f"{QWEN_BASE_URL}/models"
    
    def __init__(
        self,
        config: Optional[ResolverConfig] = None,
        access_token: Optional[str] = None,
        cache: Optional[ModelDiscoveryCache] = None,
    ):
        """
        Initialize locked model resolver
        
        Args:
            config: Resolver configuration
            access_token: Qwen access token for authenticated requests
            cache: Optional model discovery cache
        """
        self.config = config or ResolverConfig()
        self.access_token = access_token
        self.cache = cache or ModelDiscoveryCache(ttl=self.config.cache_ttl)
        self._creds_file = Path.home() / ".xencode_qwen_creds.json"
    
    def set_access_token(self, token: str):
        """Set access token for authenticated requests"""
        self.access_token = token
    
    def set_creds_file(self, creds_file: Path):
        """Set custom credentials file path"""
        self._creds_file = creds_file
    
    def _load_access_token(self) -> Optional[str]:
        """Load access token from credentials file"""
        if self.access_token:
            return self.access_token
        
        try:
            if not self._creds_file.exists():
                return None
            
            with open(self._creds_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Check expiry
            created_at = data.get('created_at', 0)
            expires_in = data.get('expires_in', 0)
            elapsed = time.time() - created_at
            
            if elapsed >= (expires_in - 300):  # 5 min buffer
                return None
            
            return data.get('access_token')
            
        except (json.JSONDecodeError, IOError, KeyError):
            return None
    
    async def discover_models(self) -> List[ModelInfo]:
        """
        Discover available models from Qwen /v1/models
        
        Returns:
            List of discovered ModelInfo objects
        """
        token = self._load_access_token()
        
        if not token:
            console.print("[yellow]Warning: No Qwen authentication available for model discovery[/yellow]")
            return []
        
        try:
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            }
            
            timeout = aiohttp.ClientTimeout(total=15)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(self.QWEN_MODELS_URL, headers=headers) as response:
                    if response.status != 200:
                        console.print(f"[yellow]Warning: Model discovery failed: HTTP {response.status}[/yellow]")
                        return []
                    
                    data = await response.json()
                    models_data = data.get("data", []) if isinstance(data, dict) else []
                    
                    models = []
                    for model_data in models_data:
                        model_id = model_data.get("id", "")
                        
                        # Extract capabilities
                        capabilities = []
                        if "coder" in model_id.lower() or "code" in model_id.lower():
                            capabilities.append("code")
                        if "instruct" in model_id.lower():
                            capabilities.append("instruct")
                        if "chat" in model_id.lower():
                            capabilities.append("chat")
                        
                        # Extract context window if available
                        context_window = None
                        if "context_length" in model_data:
                            context_window = model_data["context_length"]
                        elif "context_window" in model_data:
                            context_window = model_data["context_window"]
                        
                        models.append(ModelInfo(
                            model_id=model_id,
                            capabilities=capabilities,
                            context_window=context_window,
                        ))
                    
                    # Update cache
                    self.cache.set_models(models)
                    console.print(f"[green]OK: Discovered {len(models)} models from Qwen[/green]")
                    
                    return models
                    
        except aiohttp.ClientConnectionError as e:
            console.print(f"[yellow]Warning: Cannot connect to Qwen for model discovery: {e}[/yellow]")
            return []
        except asyncio.TimeoutError:
            console.print("[yellow]Warning: Model discovery timed out[/yellow]")
            return []
        except Exception as e:
            console.print(f"[yellow]Warning: Model discovery failed: {e}[/yellow]")
            return []
    
    async def ensure_cache_valid(self) -> bool:
        """Ensure model cache is valid, discover if needed"""
        if self.cache.is_valid():
            return True
        
        models = await self.discover_models()
        return len(models) > 0
    
    async def resolve_coder_model(self, force_refresh: bool = False) -> Optional[str]:
        """
        Resolve to the best coder model
        
        Resolution order:
        1. User override (if configured)
        2. Best discovered coder model (from cache)
        3. Default model (qwen3-coder-next-instruct)
        
        Args:
            force_refresh: Force cache refresh
            
        Returns:
            Resolved model ID or None if resolution fails
        """
        # Check for user override first
        if self.config.lock_model_override:
            override_model = self.config.lock_model_override
            
            # Verify override model exists if requested
            if self.config.verify_on_resolve:
                await self.ensure_cache_valid()
                cached_model = self.cache.get_model(override_model)
                
                if cached_model:
                    console.print(f"[green]OK: Using override model: {override_model}[/green]")
                    return override_model
                else:
                    console.print(f"[yellow]Warning: Override model '{override_model}' not found, using default[/yellow]")
            else:
                console.print(f"[green]OK: Using override model: {override_model}[/green]")
                return override_model
        
        # Ensure cache is valid
        if force_refresh or not self.cache.is_valid():
            await self.ensure_cache_valid()
        
        # Try to find best coder model from cache
        if self.cache.is_valid():
            best_model = self.cache.find_best_coder_model()
            
            if best_model:
                console.print(f"[green]OK: Resolved coder model: {best_model.model_id}[/green]")
                return best_model.model_id
        
        # Fallback to default model
        console.print(f"[blue]Info: Using default coder model: {self.DEFAULT_CODER_MODEL}[/blue]")
        return self.DEFAULT_CODER_MODEL
    
    async def verify_model_exists(self, model_id: str) -> bool:
        """
        Verify that a model exists via /v1/models
        
        Args:
            model_id: Model ID to verify
            
        Returns:
            True if model exists, False otherwise
        """
        await self.ensure_cache_valid()
        return self.cache.get_model(model_id) is not None
    
    async def resolve_alias(self, alias: str) -> Optional[str]:
        """
        Resolve a model alias to canonical model ID
        
        Args:
            alias: Model alias (e.g., "qwen-coder-next-latest")
            
        Returns:
            Canonical model ID or None if alias not found
        """
        # Check built-in alias map first
        canonical = ModelAlias.CANONICAL_MAP.get(alias.lower())
        if canonical:
            return canonical
        
        # Check discovered models
        await self.ensure_cache_valid()
        model = self.cache.get_model(alias)
        
        if model:
            return model.model_id
        
        return None
    
    def invalidate_cache(self):
        """Invalidate the model cache"""
        self.cache.invalidate()
        console.print("[blue]Info: Model cache invalidated[/blue]")
    
    def get_config(self) -> Dict[str, Any]:
        """Get current resolver configuration"""
        return self.config.to_dict()
    
    def set_config(self, config_dict: Dict[str, Any]):
        """Update resolver configuration"""
        self.config = ResolverConfig.from_dict(config_dict)


# Global resolver instance
_resolver: Optional[LockedModelResolver] = None


def get_resolver(config: Optional[ResolverConfig] = None) -> LockedModelResolver:
    """Get or create global locked model resolver"""
    global _resolver
    if _resolver is None:
        _resolver = LockedModelResolver(config=config)
    return _resolver


async def resolve_coder_model(
    config: Optional[ResolverConfig] = None,
    force_refresh: bool = False,
) -> Optional[str]:
    """
    Convenience function to resolve coder model
    
    Args:
        config: Optional resolver configuration
        force_refresh: Force cache refresh
        
    Returns:
        Resolved model ID or None
    """
    resolver = get_resolver(config)
    return await resolver.resolve_coder_model(force_refresh=force_refresh)


if __name__ == "__main__":
    # Locked Model Resolver - Run with --demo flag for testing
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        async def demo():
            console.print("[bold blue]Locked Model Resolver Demo[/bold blue]\n")

            resolver = LockedModelResolver(
                config=ResolverConfig(
                    lock_best_coder=True,
                    lock_model_override=None,
                    cache_ttl=300,
                )
            )

            # Try to resolve coder model
            console.print("\n[bold]Resolving coder model...[/bold]")
            model_id = await resolver.resolve_coder_model(force_refresh=True)

            if model_id:
                console.print(f"[green]OK: Resolved to: {model_id}[/green]")
            else:
                console.print("[red]FAIL: Failed to resolve coder model[/red]")

            # Show cache status
            console.print(f"\n[bold]Cache valid: {resolver.cache.is_valid()}[/bold]")

            if resolver.cache.is_valid():
                coder_models = resolver.cache.get_coder_models()
                console.print(f"\n[bold]Discovered {len(coder_models)} coder models:[/bold]")
                for model in coder_models:
                    console.print(f"  - {model.model_id} (capabilities: {', '.join(model.capabilities)})")
        
        import asyncio
        asyncio.run(demo())
    else:
        print("Locked Model Resolver module")
        print("Usage: python -m xencode.model_providers.resolver --demo")
