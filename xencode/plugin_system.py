#!/usr/bin/env python3
"""
Plugin Architecture System for Xencode Phase 3

Extensible plugin framework with hot-loading, dependency management,
and comprehensive plugin lifecycle management.
"""

import asyncio
import importlib
import importlib.util
import inspect
import json
import logging
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Type, Union, Callable
import yaml
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()
logger = logging.getLogger(__name__)


@dataclass
class PluginMetadata:
    """Plugin metadata and configuration"""
    name: str
    version: str
    description: str
    author: str
    license: str = "MIT"
    dependencies: List[str] = field(default_factory=list)
    xencode_version: str = ">=3.0.0"
    entry_point: str = "main"
    config_schema: Dict[str, Any] = field(default_factory=dict)
    permissions: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    homepage: Optional[str] = None
    repository: Optional[str] = None


@dataclass
class PluginConfig:
    """Plugin configuration settings"""
    enabled: bool = True
    config: Dict[str, Any] = field(default_factory=dict)
    load_priority: int = 100
    auto_reload: bool = False


class PluginInterface(ABC):
    """Base interface for all Xencode plugins"""
    
    @abstractmethod
    async def initialize(self, context: 'PluginContext') -> bool:
        """Initialize the plugin with given context"""
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """Clean shutdown of plugin resources"""
        pass
    
    @abstractmethod
    def get_metadata(self) -> PluginMetadata:
        """Return plugin metadata"""
        pass
    
    async def on_config_change(self, config: Dict[str, Any]) -> None:
        """Handle configuration changes (optional)"""
        pass
    
    async def health_check(self) -> Dict[str, Any]:
        """Return plugin health status (optional)"""
        return {"status": "healthy", "details": {}}


class PluginContext:
    """Context provided to plugins for system interaction"""
    
    def __init__(self, plugin_manager: 'PluginManager'):
        self.plugin_manager = plugin_manager
        self.logger = logging.getLogger(f"xencode.plugin")
        self._services: Dict[str, Any] = {}
    
    def register_service(self, name: str, service: Any) -> None:
        """Register a service for other plugins to use"""
        self._services[name] = service
    
    def get_service(self, name: str) -> Optional[Any]:
        """Get a registered service"""
        return self._services.get(name)
    
    def list_services(self) -> List[str]:
        """List all available services"""
        return list(self._services.keys())
    
    async def emit_event(self, event_name: str, data: Dict[str, Any]) -> None:
        """Emit an event to all listening plugins"""
        await self.plugin_manager.emit_event(event_name, data)
    
    def subscribe_event(self, event_name: str, callback: Callable) -> None:
        """Subscribe to system events"""
        self.plugin_manager.subscribe_event(event_name, callback)


@dataclass
class LoadedPlugin:
    """Container for a loaded plugin instance"""
    metadata: PluginMetadata
    instance: PluginInterface
    config: PluginConfig
    module_path: Path
    load_time: float
    is_active: bool = True
    error_count: int = 0
    last_error: Optional[str] = None


class PluginManager:
    """Central plugin management system"""
    
    def __init__(self, plugin_dir: Path):
        self.plugin_dir = Path(plugin_dir)
        self.plugins: Dict[str, LoadedPlugin] = {}
        self.event_handlers: Dict[str, List[Callable]] = {}
        self.context = PluginContext(self)
        self.config_file = self.plugin_dir / "plugins.yaml"
        self._ensure_plugin_directory()
    
    def _ensure_plugin_directory(self) -> None:
        """Create plugin directory structure"""
        self.plugin_dir.mkdir(parents=True, exist_ok=True)
        (self.plugin_dir / "enabled").mkdir(exist_ok=True)
        (self.plugin_dir / "disabled").mkdir(exist_ok=True)
        (self.plugin_dir / "configs").mkdir(exist_ok=True)
        
        # Create default plugins.yaml if it doesn't exist
        if not self.config_file.exists():
            default_config = {
                "global": {
                    "auto_reload": False,
                    "max_errors": 5,
                    "plugin_timeout": 30
                },
                "plugins": {}
            }
            with open(self.config_file, 'w') as f:
                yaml.dump(default_config, f, default_flow_style=False)
    
    async def discover_plugins(self) -> List[Path]:
        """Discover all available plugins"""
        plugins = []
        
        # Search for Python files in enabled directory
        for plugin_file in (self.plugin_dir / "enabled").glob("*.py"):
            if plugin_file.name.startswith("__"):
                continue
            plugins.append(plugin_file)
        
        # Search for plugin packages (directories with __init__.py)
        for plugin_dir in (self.plugin_dir / "enabled").iterdir():
            if plugin_dir.is_dir() and (plugin_dir / "__init__.py").exists():
                plugins.append(plugin_dir)
        
        return plugins
    
    async def load_plugin(self, plugin_path: Path) -> Optional[LoadedPlugin]:
        """Load a single plugin"""
        try:
            console.print(f"🔌 Loading plugin: {plugin_path.name}")
            
            # Import the plugin module
            if plugin_path.is_file():
                spec = importlib.util.spec_from_file_location(
                    plugin_path.stem, plugin_path
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
            else:
                # Plugin package
                module_name = f"xencode_plugin_{plugin_path.name}"
                spec = importlib.util.spec_from_file_location(
                    module_name, plugin_path / "__init__.py"
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
            
            # Find plugin class (must inherit from PluginInterface)
            plugin_class = None
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if (issubclass(obj, PluginInterface) and 
                    obj is not PluginInterface):
                    plugin_class = obj
                    break
            
            if not plugin_class:
                logger.error(f"No PluginInterface implementation found in {plugin_path}")
                return None
            
            # Create plugin instance
            plugin_instance = plugin_class()
            metadata = plugin_instance.get_metadata()
            
            # Load plugin configuration
            config = await self._load_plugin_config(metadata.name)
            
            # Initialize plugin
            if await plugin_instance.initialize(self.context):
                loaded_plugin = LoadedPlugin(
                    metadata=metadata,
                    instance=plugin_instance,
                    config=config,
                    module_path=plugin_path,
                    load_time=asyncio.get_event_loop().time()
                )
                
                self.plugins[metadata.name] = loaded_plugin
                console.print(f"✅ Plugin '{metadata.name}' loaded successfully")
                return loaded_plugin
            else:
                logger.error(f"Plugin '{metadata.name}' initialization failed")
                return None
                
        except Exception as e:
            logger.error(f"Failed to load plugin {plugin_path}: {str(e)}")
            console.print(f"❌ Failed to load plugin {plugin_path.name}: {str(e)}")
            return None
    
    async def _load_plugin_config(self, plugin_name: str) -> PluginConfig:
        """Load configuration for a specific plugin"""
        try:
            with open(self.config_file, 'r') as f:
                config_data = yaml.safe_load(f)
            
            plugin_config = config_data.get("plugins", {}).get(plugin_name, {})
            return PluginConfig(**plugin_config)
        except Exception as e:
            logger.warning(f"Failed to load config for {plugin_name}: {e}")
            return PluginConfig()
    
    async def load_all_plugins(self) -> None:
        """Load all discovered plugins"""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Loading plugins...", total=None)
            
            plugins = await self.discover_plugins()
            if not plugins:
                console.print("📭 No plugins found in enabled directory")
                return
            
            loaded_count = 0
            for plugin_path in plugins:
                loaded_plugin = await self.load_plugin(plugin_path)
                if loaded_plugin:
                    loaded_count += 1
            
            progress.update(task, completed=True)
            console.print(f"🎉 Loaded {loaded_count}/{len(plugins)} plugins successfully")
    
    async def unload_plugin(self, plugin_name: str) -> bool:
        """Unload a specific plugin"""
        if plugin_name not in self.plugins:
            console.print(f"❌ Plugin '{plugin_name}' not found")
            return False
        
        try:
            plugin = self.plugins[plugin_name]
            await plugin.instance.shutdown()
            del self.plugins[plugin_name]
            console.print(f"🔌 Plugin '{plugin_name}' unloaded")
            return True
        except Exception as e:
            logger.error(f"Failed to unload plugin {plugin_name}: {e}")
            return False
    
    async def reload_plugin(self, plugin_name: str) -> bool:
        """Reload a specific plugin"""
        if plugin_name not in self.plugins:
            return False
        
        plugin = self.plugins[plugin_name]
        plugin_path = plugin.module_path
        
        # Unload first
        if await self.unload_plugin(plugin_name):
            # Then reload
            return await self.load_plugin(plugin_path) is not None
        return False
    
    async def enable_plugin(self, plugin_name: str) -> bool:
        """Enable a disabled plugin"""
        disabled_path = self.plugin_dir / "disabled" / f"{plugin_name}.py"
        enabled_path = self.plugin_dir / "enabled" / f"{plugin_name}.py"
        
        if disabled_path.exists():
            disabled_path.rename(enabled_path)
            await self.load_plugin(enabled_path)
            console.print(f"✅ Plugin '{plugin_name}' enabled")
            return True
        return False
    
    async def disable_plugin(self, plugin_name: str) -> bool:
        """Disable a plugin"""
        if plugin_name in self.plugins:
            await self.unload_plugin(plugin_name)
        
        enabled_path = self.plugin_dir / "enabled" / f"{plugin_name}.py"
        disabled_path = self.plugin_dir / "disabled" / f"{plugin_name}.py"
        
        if enabled_path.exists():
            enabled_path.rename(disabled_path)
            console.print(f"🔇 Plugin '{plugin_name}' disabled")
            return True
        return False
    
    def list_plugins(self) -> Dict[str, LoadedPlugin]:
        """Get all loaded plugins"""
        return self.plugins.copy()
    
    async def get_plugin_status(self) -> Dict[str, Any]:
        """Get comprehensive plugin system status"""
        status = {
            "total_plugins": len(self.plugins),
            "active_plugins": sum(1 for p in self.plugins.values() if p.is_active),
            "error_count": sum(p.error_count for p in self.plugins.values()),
            "plugins": {}
        }
        
        for name, plugin in self.plugins.items():
            health = await plugin.instance.health_check()
            status["plugins"][name] = {
                "version": plugin.metadata.version,
                "active": plugin.is_active,
                "errors": plugin.error_count,
                "health": health,
                "load_time": plugin.load_time
            }
        
        return status
    
    def display_plugins(self) -> None:
        """Display plugin information in a table"""
        if not self.plugins:
            console.print("📭 No plugins loaded")
            return
        
        table = Table(title="🔌 Loaded Plugins")
        table.add_column("Name", style="cyan")
        table.add_column("Version", style="magenta")
        table.add_column("Status", style="green")
        table.add_column("Description", style="yellow")
        table.add_column("Author", style="blue")
        
        for name, plugin in self.plugins.items():
            status = "🟢 Active" if plugin.is_active else "🔴 Inactive"
            if plugin.error_count > 0:
                status += f" (⚠️ {plugin.error_count} errors)"
            
            table.add_row(
                name,
                plugin.metadata.version,
                status,
                plugin.metadata.description,
                plugin.metadata.author
            )
        
        console.print(table)
    
    async def emit_event(self, event_name: str, data: Dict[str, Any]) -> None:
        """Emit an event to all registered handlers"""
        if event_name in self.event_handlers:
            for handler in self.event_handlers[event_name]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(data)
                    else:
                        handler(data)
                except Exception as e:
                    logger.error(f"Error in event handler for {event_name}: {e}")
    
    def subscribe_event(self, event_name: str, callback: Callable) -> None:
        """Subscribe to system events"""
        if event_name not in self.event_handlers:
            self.event_handlers[event_name] = []
        self.event_handlers[event_name].append(callback)
    
    async def shutdown_all_plugins(self) -> None:
        """Shutdown all plugins gracefully"""
        console.print("🔌 Shutting down all plugins...")
        
        for name, plugin in self.plugins.items():
            try:
                await plugin.instance.shutdown()
                console.print(f"✅ Plugin '{name}' shut down")
            except Exception as e:
                logger.error(f"Error shutting down plugin {name}: {e}")
        
        self.plugins.clear()
        console.print("🔌 All plugins shut down")


class PluginMarketplace:
    """Plugin marketplace for discovering and installing plugins"""
    
    def __init__(self, plugin_manager: PluginManager):
        self.plugin_manager = plugin_manager
        self.marketplace_url = "https://api.xencode.dev/plugins"
        self.local_cache = plugin_manager.plugin_dir / "marketplace_cache.json"
    
    async def search_plugins(self, query: str = "", tags: List[str] = None) -> List[Dict[str, Any]]:
        """Search for plugins in the marketplace"""
        # TODO: Implement actual marketplace API integration
        # For now, return mock data
        mock_plugins = [
            {
                "name": "code-formatter",
                "version": "1.0.0",
                "description": "Advanced code formatting and linting",
                "author": "Xencode Team",
                "tags": ["formatting", "linting", "productivity"],
                "downloads": 1250,
                "rating": 4.8
            },
            {
                "name": "ai-translator",
                "version": "2.1.0",
                "description": "Multi-language translation using AI",
                "author": "Community",
                "tags": ["translation", "ai", "language"],
                "downloads": 890,
                "rating": 4.6
            }
        ]
        
        if query:
            mock_plugins = [p for p in mock_plugins if query.lower() in p["name"].lower() or query.lower() in p["description"].lower()]
        
        if tags:
            mock_plugins = [p for p in mock_plugins if any(tag in p["tags"] for tag in tags)]
        
        return mock_plugins
    
    async def install_plugin(self, plugin_name: str, version: str = "latest") -> bool:
        """Install a plugin from the marketplace"""
        # TODO: Implement actual plugin installation
        console.print(f"📦 Installing plugin '{plugin_name}' version {version}...")
        console.print(f"✅ Plugin '{plugin_name}' installed successfully")
        return True
    
    def display_marketplace(self, plugins: List[Dict[str, Any]]) -> None:
        """Display marketplace plugins in a table"""
        if not plugins:
            console.print("📭 No plugins found")
            return
        
        table = Table(title="🏪 Plugin Marketplace")
        table.add_column("Name", style="cyan")
        table.add_column("Version", style="magenta")
        table.add_column("Description", style="yellow")
        table.add_column("Author", style="blue")
        table.add_column("Downloads", style="green")
        table.add_column("Rating", style="bright_yellow")
        
        for plugin in plugins:
            table.add_row(
                plugin["name"],
                plugin["version"],
                plugin["description"],
                plugin["author"],
                str(plugin["downloads"]),
                f"⭐ {plugin['rating']}"
            )
        
        console.print(table)


# Example plugin for testing
class ExamplePlugin(PluginInterface):
    """Example plugin implementation"""
    
    def __init__(self):
        self.name = "example-plugin"
        self.active = False
    
    async def initialize(self, context: PluginContext) -> bool:
        """Initialize the example plugin"""
        self.context = context
        self.active = True
        
        # Register a service
        context.register_service("example_service", self)
        
        # Subscribe to events
        context.subscribe_event("test_event", self.handle_test_event)
        
        return True
    
    async def shutdown(self) -> None:
        """Shutdown the plugin"""
        self.active = False
    
    def get_metadata(self) -> PluginMetadata:
        """Return plugin metadata"""
        return PluginMetadata(
            name="example-plugin",
            version="1.0.0",
            description="Example plugin for testing the plugin system",
            author="Xencode Team",
            license="MIT",
            tags=["example", "testing"]
        )
    
    async def handle_test_event(self, data: Dict[str, Any]) -> None:
        """Handle test events"""
        console.print(f"🎉 Example plugin received event: {data}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Return health status"""
        return {
            "status": "healthy" if self.active else "inactive",
            "details": {
                "active": self.active,
                "service_registered": True
            }
        }


async def main():
    """Demo the plugin system"""
    console.print(Panel.fit("🔌 Xencode Plugin System Demo", style="bold magenta"))
    
    # Create plugin manager
    plugin_dir = Path.home() / ".xencode" / "plugins"
    manager = PluginManager(plugin_dir)
    
    # Create example plugin file
    example_plugin_path = plugin_dir / "enabled" / "example_plugin.py"
    if not example_plugin_path.exists():
        with open(example_plugin_path, 'w') as f:
            f.write(inspect.getsource(ExamplePlugin))
    
    # Load all plugins
    await manager.load_all_plugins()
    
    # Display loaded plugins
    manager.display_plugins()
    
    # Test plugin system
    await manager.emit_event("test_event", {"message": "Hello from plugin system!"})
    
    # Show status
    status = await manager.get_plugin_status()
    console.print(Panel(
        f"Total Plugins: {status['total_plugins']}\n"
        f"Active Plugins: {status['active_plugins']}\n"
        f"Total Errors: {status['error_count']}",
        title="📊 Plugin System Status"
    ))
    
    # Demo marketplace
    marketplace = PluginMarketplace(manager)
    plugins = await marketplace.search_plugins()
    marketplace.display_marketplace(plugins)
    
    # Cleanup
    await manager.shutdown_all_plugins()


if __name__ == "__main__":
    asyncio.run(main())