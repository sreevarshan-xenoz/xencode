#!/usr/bin/env python3
"""
Marketplace Client

Standalone client for interacting with the Xencode plugin marketplace.
This is a reference implementation that can be used independently of the main plugin system.
"""

import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp


class MarketplaceClient:
    """Standalone marketplace client for plugin discovery and management"""
    
    def __init__(self, marketplace_url: str = "https://api.xencode.dev/plugins", 
                 cache_dir: Optional[Path] = None):
        self.marketplace_url = marketplace_url
        self.cache_dir = cache_dir or Path.home() / ".xencode" / "marketplace_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_ttl = timedelta(hours=6)
        self.session = None
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def search_plugins(self, query: str = "", tags: List[str] = None, 
                           category: str = "", sort_by: str = "downloads") -> List[Dict[str, Any]]:
        """Search for plugins in the marketplace"""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            params = {
                "query": query,
                "tags": ",".join(tags) if tags else "",
                "category": category,
                "sort": sort_by
            }
            
            async with self.session.get(f"{self.marketplace_url}/search", params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("plugins", [])
                else:
                    return self._get_fallback_plugins(query, tags)
                    
        except Exception as e:
            print(f"Error searching marketplace: {e}")
            return self._get_fallback_plugins(query, tags)
    
    async def get_plugin_info(self, plugin_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific plugin"""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            async with self.session.get(f"{self.marketplace_url}/plugins/{plugin_name}") as response:
                if response.status == 200:
                    return await response.json()
                return None
        except Exception as e:
            print(f"Error getting plugin info: {e}")
            return None
    
    async def get_plugin_versions(self, plugin_name: str) -> List[Dict[str, Any]]:
        """Get available versions for a plugin"""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            async with self.session.get(f"{self.marketplace_url}/plugins/{plugin_name}/versions") as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("versions", [])
                return []
        except Exception as e:
            print(f"Error getting plugin versions: {e}")
            return []
    
    async def download_plugin(self, plugin_name: str, version: str = "latest") -> Optional[bytes]:
        """Download plugin package"""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            url = f"{self.marketplace_url}/plugins/{plugin_name}/download"
            params = {"version": version} if version != "latest" else {}
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    return await response.read()
                return None
        except Exception as e:
            print(f"Error downloading plugin: {e}")
            return None
    
    def _get_fallback_plugins(self, query: str = "", tags: List[str] = None) -> List[Dict[str, Any]]:
        """Get fallback plugin list when marketplace is unavailable"""
        fallback_plugins = [
            {
                "name": "code-formatter",
                "version": "1.0.0",
                "description": "Advanced code formatting and linting",
                "author": "Xencode Team",
                "tags": ["formatting", "linting", "productivity"],
                "downloads": 1250,
                "rating": 4.8,
                "marketplace_id": "xencode-code-formatter",
                "security_verified": True,
                "last_updated": "2024-10-01T00:00:00Z"
            },
            {
                "name": "ai-translator",
                "version": "2.1.0",
                "description": "Multi-language translation using AI",
                "author": "Community",
                "tags": ["translation", "ai", "language"],
                "downloads": 890,
                "rating": 4.6,
                "marketplace_id": "community-ai-translator",
                "security_verified": True,
                "last_updated": "2024-09-15T00:00:00Z"
            },
            {
                "name": "git-integration",
                "version": "1.5.2",
                "description": "Enhanced Git integration and workflow tools",
                "author": "DevTools Inc",
                "tags": ["git", "version-control", "workflow"],
                "downloads": 2100,
                "rating": 4.9,
                "marketplace_id": "devtools-git-integration",
                "security_verified": True,
                "last_updated": "2024-10-10T00:00:00Z"
            }
        ]
        
        # Apply filters
        if query:
            fallback_plugins = [p for p in fallback_plugins 
                              if query.lower() in p["name"].lower() or 
                                 query.lower() in p["description"].lower()]
        
        if tags:
            fallback_plugins = [p for p in fallback_plugins 
                              if any(tag in p["tags"] for tag in tags)]
        
        return fallback_plugins
    
    async def close(self):
        """Close the client session"""
        if self.session:
            await self.session.close()


async def main():
    """Demo the marketplace client"""
    async with MarketplaceClient() as client:
        print("🏪 Xencode Marketplace Client Demo")
        
        # Search for plugins
        plugins = await client.search_plugins(query="formatter")
        print(f"Found {len(plugins)} plugins matching 'formatter'")
        
        for plugin in plugins:
            print(f"  - {plugin['name']} v{plugin['version']} by {plugin['author']}")
        
        # Get plugin info
        if plugins:
            plugin_name = plugins[0]['name']
            info = await client.get_plugin_info(plugin_name)
            if info:
                print(f"\nPlugin info for {plugin_name}:")
                print(f"  Description: {info.get('description', 'N/A')}")
            
            # Get versions
            versions = await client.get_plugin_versions(plugin_name)
            print(f"  Available versions: {len(versions)}")


if __name__ == "__main__":
    asyncio.run(main())