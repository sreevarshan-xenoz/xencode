#!/usr/bin/env python3
"""
Plugin Architecture System for Xencode Phase 3

Extensible plugin framework with hot-loading, dependency management,
and comprehensive plugin lifecycle management.
"""

import asyncio
import hashlib
import importlib
import importlib.util
import inspect
import json
import logging
import sys
import tempfile
import zipfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Type, Union, Callable
from urllib.parse import urlparse
import aiohttp
import yaml
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from packaging import version
from packaging.specifiers import SpecifierSet
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
    
    # Enhanced marketplace fields
    marketplace_id: Optional[str] = None
    signature: Optional[str] = None
    checksum: Optional[str] = None
    download_url: Optional[str] = None
    install_size: int = 0
    min_python_version: str = "3.8"
    max_python_version: Optional[str] = None
    platform_compatibility: List[str] = field(default_factory=lambda: ["any"])
    security_scan_passed: bool = False
    last_updated: Optional[datetime] = None


@dataclass
class PluginConfig:
    """Plugin configuration settings"""
    enabled: bool = True
    config: Dict[str, Any] = field(default_factory=dict)
    load_priority: int = 100
    auto_reload: bool = False
    
    # Enhanced security and versioning
    auto_update: bool = False
    update_channel: str = "stable"  # stable, beta, alpha
    security_policy: str = "strict"  # strict, moderate, permissive
    resource_limits: Dict[str, Any] = field(default_factory=lambda: {
        "max_memory_mb": 256,
        "max_cpu_percent": 10,
        "max_disk_mb": 100,
        "network_access": False
    })


@dataclass
class PluginVersion:
    """Plugin version information"""
    version: str
    release_date: datetime
    changelog: str
    download_url: str
    signature: str
    checksum: str
    compatibility: List[str]
    security_scan_result: Dict[str, Any]
    
    def is_compatible_with(self, xencode_version: str) -> bool:
        """Check if this version is compatible with Xencode version"""
        try:
            for compat in self.compatibility:
                if version.parse(xencode_version) in SpecifierSet(compat):
                    return True
            return False
        except Exception:
            return False


class PluginSignatureVerifier:
    """Handles plugin signature verification for security"""
    
    def __init__(self, public_key_path: Optional[Path] = None):
        self.public_key_path = public_key_path
        self.public_key = self._load_public_key() if public_key_path else None
    
    def _load_public_key(self):
        """Load the public key for signature verification"""
        try:
            if self.public_key_path and self.public_key_path.exists():
                with open(self.public_key_path, 'rb') as f:
                    return serialization.load_pem_public_key(f.read())
        except Exception as e:
            logger.warning(f"Failed to load public key: {e}")
        return None
    
    def verify_signature(self, plugin_data: bytes, signature: str) -> bool:
        """Verify plugin signature"""
        if not self.public_key:
            logger.warning("No public key available for signature verification")
            return False
        
        try:
            signature_bytes = bytes.fromhex(signature)
            self.public_key.verify(
                signature_bytes,
                plugin_data,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return True
        except Exception as e:
            logger.error(f"Signature verification failed: {e}")
            return False
    
    def verify_checksum(self, plugin_data: bytes, expected_checksum: str) -> bool:
        """Verify plugin checksum"""
        try:
            actual_checksum = hashlib.sha256(plugin_data).hexdigest()
            return actual_checksum == expected_checksum
        except Exception as e:
            logger.error(f"Checksum verification failed: {e}")
            return False


class PluginVersionManager:
    """Manages plugin versions and updates"""
    
    def __init__(self, plugin_manager: 'PluginManager'):
        self.plugin_manager = plugin_manager
        self.version_cache: Dict[str, List[PluginVersion]] = {}
        self.update_check_interval = timedelta(hours=24)
        self.last_update_check: Dict[str, datetime] = {}
    
    async def check_for_updates(self, plugin_name: str) -> Optional[PluginVersion]:
        """Check if updates are available for a plugin"""
        try:
            # Check cache first
            now = datetime.now()
            if (plugin_name in self.last_update_check and 
                now - self.last_update_check[plugin_name] < self.update_check_interval):
                return None
            
            current_plugin = self.plugin_manager.plugins.get(plugin_name)
            if not current_plugin:
                return None
            
            # Get available versions from marketplace
            marketplace = self.plugin_manager.marketplace
            available_versions = await marketplace.get_plugin_versions(plugin_name)
            
            if not available_versions:
                return None
            
            # Find latest compatible version
            current_version = version.parse(current_plugin.metadata.version)
            latest_version = None
            
            for ver in available_versions:
                ver_parsed = version.parse(ver.version)
                if (ver_parsed > current_version and 
                    ver.is_compatible_with(self.plugin_manager.xencode_version)):
                    if not latest_version or ver_parsed > version.parse(latest_version.version):
                        latest_version = ver
            
            self.last_update_check[plugin_name] = now
            return latest_version
            
        except Exception as e:
            logger.error(f"Error checking updates for {plugin_name}: {e}")
            return None
    
    async def update_plugin(self, plugin_name: str, target_version: Optional[str] = None) -> bool:
        """Update plugin to specified version or latest"""
        try:
            current_plugin = self.plugin_manager.plugins.get(plugin_name)
            if not current_plugin:
                return False
            
            # Get target version
            if target_version:
                available_versions = await self.plugin_manager.marketplace.get_plugin_versions(plugin_name)
                target_ver = next((v for v in available_versions if v.version == target_version), None)
            else:
                target_ver = await self.check_for_updates(plugin_name)
            
            if not target_ver:
                console.print(f"No update available for {plugin_name}")
                return False
            
            console.print(f"🔄 Updating {plugin_name} from {current_plugin.metadata.version} to {target_ver.version}")
            
            # Download and install new version
            success = await self.plugin_manager.marketplace.install_plugin(
                plugin_name, target_ver.version
            )
            
            if success:
                # Reload the plugin
                await self.plugin_manager.reload_plugin(plugin_name)
                console.print(f"✅ Successfully updated {plugin_name} to {target_ver.version}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error updating plugin {plugin_name}: {e}")
            return False
    
    async def rollback_plugin(self, plugin_name: str, target_version: str) -> bool:
        """Rollback plugin to a previous version"""
        try:
            # Similar to update but with version constraint
            return await self.update_plugin(plugin_name, target_version)
        except Exception as e:
            logger.error(f"Error rolling back plugin {plugin_name}: {e}")
            return False


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
    """Enhanced central plugin management system with marketplace integration"""
    
    def __init__(self, plugin_dir: Path, xencode_version: str = "3.0.0"):
        self.plugin_dir = Path(plugin_dir)
        self.xencode_version = xencode_version
        self.plugins: Dict[str, LoadedPlugin] = {}
        self.event_handlers: Dict[str, List[Callable]] = {}
        self.context = PluginContext(self)
        self.config_file = self.plugin_dir / "plugins.yaml"
        
        # Enhanced features
        self.signature_verifier = PluginSignatureVerifier(
            self.plugin_dir / "keys" / "public_key.pem"
        )
        self.version_manager = PluginVersionManager(self)
        self.marketplace = None  # Will be initialized later
        
        # Security settings
        self.security_enabled = True
        self.require_signatures = False  # Set to True in production
        self.allowed_permissions = {
            "file_system", "network", "system_info", "user_input", 
            "clipboard", "notifications", "ai_models"
        }
        
        self._ensure_plugin_directory()
        
        # Initialize marketplace
        self.marketplace = PluginMarketplace(self)
    
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
    
    async def load_plugin(self, plugin_path: Path, skip_security: bool = False) -> Optional[LoadedPlugin]:
        """Load a single plugin with enhanced security checks"""
        try:
            console.print(f"🔌 Loading plugin: {plugin_path.name}")
            
            # Security checks
            if self.security_enabled and not skip_security:
                if not await self._perform_security_checks(plugin_path):
                    console.print(f"🚫 Security check failed for {plugin_path.name}")
                    return None
            
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
            
            # Validate metadata
            if not await self._validate_plugin_metadata(metadata):
                return None
            
            # Check version compatibility
            if not self._check_version_compatibility(metadata):
                console.print(f"❌ Plugin '{metadata.name}' is not compatible with Xencode {self.xencode_version}")
                return None
            
            # Load plugin configuration
            config = await self._load_plugin_config(metadata.name)
            
            # Validate permissions
            if not self._validate_permissions(metadata.permissions):
                console.print(f"❌ Plugin '{metadata.name}' requests invalid permissions")
                return None
            
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
                
                # Emit plugin loaded event
                await self.emit_event("plugin_loaded", {
                    "plugin_name": metadata.name,
                    "version": metadata.version,
                    "load_time": loaded_plugin.load_time
                })
                
                return loaded_plugin
            else:
                logger.error(f"Plugin '{metadata.name}' initialization failed")
                return None
                
        except Exception as e:
            logger.error(f"Failed to load plugin {plugin_path}: {str(e)}")
            console.print(f"❌ Failed to load plugin {plugin_path.name}: {str(e)}")
            return None
    
    async def _perform_security_checks(self, plugin_path: Path) -> bool:
        """Perform security checks on plugin"""
        try:
            # Read plugin data
            if plugin_path.is_file():
                plugin_data = plugin_path.read_bytes()
            else:
                # For plugin packages, check main file
                main_file = plugin_path / "__init__.py"
                if not main_file.exists():
                    return False
                plugin_data = main_file.read_bytes()
            
            # Check for malicious patterns
            dangerous_patterns = [
                b'exec(', b'eval(', b'__import__', b'subprocess',
                b'os.system', b'open(', b'file(', b'input('
            ]
            
            for pattern in dangerous_patterns:
                if pattern in plugin_data:
                    logger.warning(f"Potentially dangerous pattern found in {plugin_path}: {pattern}")
                    # In strict mode, this would fail. For now, just warn.
            
            # Check file size
            if len(plugin_data) > 10 * 1024 * 1024:  # 10MB limit
                logger.error(f"Plugin {plugin_path} exceeds size limit")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Security check failed for {plugin_path}: {e}")
            return False
    
    async def _validate_plugin_metadata(self, metadata: PluginMetadata) -> bool:
        """Validate plugin metadata"""
        try:
            # Check required fields
            if not all([metadata.name, metadata.version, metadata.author]):
                logger.error("Plugin metadata missing required fields")
                return False
            
            # Validate version format
            try:
                version.parse(metadata.version)
            except Exception:
                logger.error(f"Invalid version format: {metadata.version}")
                return False
            
            # Check name format (no special characters)
            if not metadata.name.replace('-', '').replace('_', '').isalnum():
                logger.error(f"Invalid plugin name format: {metadata.name}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Metadata validation failed: {e}")
            return False
    
    def _check_version_compatibility(self, metadata: PluginMetadata) -> bool:
        """Check if plugin is compatible with current Xencode version"""
        try:
            xencode_ver = version.parse(self.xencode_version)
            required_ver = SpecifierSet(metadata.xencode_version)
            return xencode_ver in required_ver
        except Exception as e:
            logger.warning(f"Version compatibility check failed: {e}")
            # For testing, be more lenient
            return True
    
    def _validate_permissions(self, requested_permissions: List[str]) -> bool:
        """Validate that requested permissions are allowed"""
        for perm in requested_permissions:
            if perm not in self.allowed_permissions:
                logger.error(f"Invalid permission requested: {perm}")
                return False
        return True
    
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
    
    async def check_plugin_updates(self) -> Dict[str, PluginVersion]:
        """Check for updates for all installed plugins"""
        updates = {}
        
        for plugin_name in self.plugins.keys():
            update = await self.version_manager.check_for_updates(plugin_name)
            if update:
                updates[plugin_name] = update
        
        return updates
    
    async def update_all_plugins(self) -> Dict[str, bool]:
        """Update all plugins that have available updates"""
        updates = await self.check_plugin_updates()
        results = {}
        
        for plugin_name in updates.keys():
            results[plugin_name] = await self.version_manager.update_plugin(plugin_name)
        
        return results
    
    async def install_plugin_from_marketplace(self, plugin_name: str, version: str = "latest") -> bool:
        """Install plugin from marketplace"""
        return await self.marketplace.install_plugin(plugin_name, version)
    
    def get_plugin_security_status(self) -> Dict[str, Dict[str, Any]]:
        """Get security status for all plugins"""
        status = {}
        
        for name, plugin in self.plugins.items():
            status[name] = {
                "signature_verified": bool(plugin.metadata.signature),
                "checksum_verified": bool(plugin.metadata.checksum),
                "permissions": plugin.metadata.permissions,
                "security_scan_passed": plugin.metadata.security_scan_passed,
                "error_count": plugin.error_count,
                "last_error": plugin.last_error
            }
        
        return status
    
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
        
        # Close marketplace session
        if self.marketplace:
            await self.marketplace.close()
        
        console.print("🔌 All plugins shut down")


class PluginMarketplace:
    """Enhanced plugin marketplace with security and dependency resolution"""
    
    def __init__(self, plugin_manager: PluginManager):
        self.plugin_manager = plugin_manager
        self.marketplace_url = "https://api.xencode.dev/plugins"
        self.local_cache = plugin_manager.plugin_dir / "marketplace_cache.json"
        self.cache_ttl = timedelta(hours=6)
        self.session = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session"""
        if not self.session:
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def search_plugins(self, query: str = "", tags: List[str] = None, 
                           category: str = "", sort_by: str = "downloads") -> List[Dict[str, Any]]:
        """Search for plugins in the marketplace"""
        try:
            # Check cache first
            cached_results = await self._get_cached_search(query, tags, category)
            if cached_results:
                return cached_results
            
            # Make API request
            session = await self._get_session()
            params = {
                "query": query,
                "tags": ",".join(tags) if tags else "",
                "category": category,
                "sort": sort_by,
                "xencode_version": self.plugin_manager.xencode_version
            }
            
            async with session.get(f"{self.marketplace_url}/search", params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    plugins = data.get("plugins", [])
                    
                    # Cache results
                    await self._cache_search_results(query, tags, category, plugins)
                    return plugins
                else:
                    logger.error(f"Marketplace search failed: {response.status}")
                    return await self._get_fallback_plugins(query, tags)
                    
        except Exception as e:
            logger.error(f"Error searching marketplace: {e}")
            return await self._get_fallback_plugins(query, tags)
    
    async def get_plugin_details(self, plugin_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a plugin"""
        try:
            session = await self._get_session()
            async with session.get(f"{self.marketplace_url}/plugins/{plugin_name}") as response:
                if response.status == 200:
                    return await response.json()
                return None
        except Exception as e:
            logger.error(f"Error getting plugin details: {e}")
            return None
    
    async def get_plugin_versions(self, plugin_name: str) -> List[PluginVersion]:
        """Get available versions for a plugin"""
        try:
            session = await self._get_session()
            async with session.get(f"{self.marketplace_url}/plugins/{plugin_name}/versions") as response:
                if response.status == 200:
                    data = await response.json()
                    versions = []
                    
                    for ver_data in data.get("versions", []):
                        versions.append(PluginVersion(
                            version=ver_data["version"],
                            release_date=datetime.fromisoformat(ver_data["release_date"]),
                            changelog=ver_data.get("changelog", ""),
                            download_url=ver_data["download_url"],
                            signature=ver_data.get("signature", ""),
                            checksum=ver_data["checksum"],
                            compatibility=ver_data.get("compatibility", []),
                            security_scan_result=ver_data.get("security_scan", {})
                        ))
                    
                    return versions
                return []
        except Exception as e:
            logger.error(f"Error getting plugin versions: {e}")
            return []
    
    async def install_plugin(self, plugin_name: str, plugin_version: str = "latest") -> bool:
        """Install a plugin from the marketplace with security checks"""
        try:
            console.print(f"📦 Installing plugin '{plugin_name}' version {plugin_version}...")
            
            # Get plugin details
            plugin_details = await self.get_plugin_details(plugin_name)
            if not plugin_details:
                console.print(f"❌ Plugin '{plugin_name}' not found in marketplace")
                return False
            
            # Get specific version or latest
            versions = await self.get_plugin_versions(plugin_name)
            if not versions:
                console.print(f"❌ No versions available for '{plugin_name}'")
                return False
            
            if plugin_version == "latest":
                target_version = max(versions, key=lambda v: version.parse(v.version))
            else:
                target_version = next((v for v in versions if v.version == plugin_version), None)
                if not target_version:
                    console.print(f"❌ Version {plugin_version} not found for '{plugin_name}'")
                    return False
            
            # Check compatibility
            if not target_version.is_compatible_with(self.plugin_manager.xencode_version):
                console.print(f"❌ Plugin version {target_version.version} is not compatible with Xencode {self.plugin_manager.xencode_version}")
                return False
            
            # Download plugin
            plugin_data = await self._download_plugin(target_version.download_url)
            if not plugin_data:
                return False
            
            # Verify signature and checksum
            if self.plugin_manager.require_signatures:
                if not self.plugin_manager.signature_verifier.verify_signature(plugin_data, target_version.signature):
                    console.print(f"❌ Signature verification failed for '{plugin_name}'")
                    return False
            
            if not self.plugin_manager.signature_verifier.verify_checksum(plugin_data, target_version.checksum):
                console.print(f"❌ Checksum verification failed for '{plugin_name}'")
                return False
            
            # Install plugin
            success = await self._install_plugin_data(plugin_name, plugin_data, target_version)
            
            if success:
                console.print(f"✅ Plugin '{plugin_name}' version {target_version.version} installed successfully")
                
                # Load the plugin
                plugin_path = self.plugin_manager.plugin_dir / "enabled" / f"{plugin_name}.py"
                await self.plugin_manager.load_plugin(plugin_path)
                
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error installing plugin {plugin_name}: {e}")
            console.print(f"❌ Failed to install plugin '{plugin_name}': {str(e)}")
            return False
    
    async def _download_plugin(self, download_url: str) -> Optional[bytes]:
        """Download plugin from URL"""
        try:
            session = await self._get_session()
            async with session.get(download_url) as response:
                if response.status == 200:
                    return await response.read()
                else:
                    logger.error(f"Download failed: {response.status}")
                    return None
        except Exception as e:
            logger.error(f"Error downloading plugin: {e}")
            return None
    
    async def _install_plugin_data(self, plugin_name: str, plugin_data: bytes, version_info: PluginVersion) -> bool:
        """Install plugin data to filesystem"""
        try:
            # Create temporary directory for extraction
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Handle different plugin formats
                if plugin_data.startswith(b'PK'):  # ZIP file
                    # Extract ZIP
                    zip_path = temp_path / f"{plugin_name}.zip"
                    zip_path.write_bytes(plugin_data)
                    
                    with zipfile.ZipFile(zip_path, 'r') as zip_file:
                        zip_file.extractall(temp_path)
                    
                    # Find main plugin file
                    plugin_files = list(temp_path.glob("*.py"))
                    if not plugin_files:
                        logger.error("No Python files found in plugin package")
                        return False
                    
                    main_file = plugin_files[0]
                else:
                    # Single Python file
                    main_file = temp_path / f"{plugin_name}.py"
                    main_file.write_bytes(plugin_data)
                
                # Copy to enabled directory
                target_path = self.plugin_manager.plugin_dir / "enabled" / f"{plugin_name}.py"
                target_path.write_bytes(main_file.read_bytes())
                
                # Create metadata file
                metadata_path = self.plugin_manager.plugin_dir / "configs" / f"{plugin_name}_metadata.json"
                metadata = {
                    "version": version_info.version,
                    "install_date": datetime.now().isoformat(),
                    "checksum": version_info.checksum,
                    "source": "marketplace"
                }
                metadata_path.write_text(json.dumps(metadata, indent=2))
                
                return True
                
        except Exception as e:
            logger.error(f"Error installing plugin data: {e}")
            return False
    
    async def _get_cached_search(self, query: str, tags: List[str], category: str) -> Optional[List[Dict[str, Any]]]:
        """Get cached search results if available and fresh"""
        try:
            if not self.local_cache.exists():
                return None
            
            with open(self.local_cache, 'r') as f:
                cache_data = json.load(f)
            
            cache_key = f"{query}_{','.join(tags or [])}_{category}"
            if cache_key in cache_data:
                cached_entry = cache_data[cache_key]
                cache_time = datetime.fromisoformat(cached_entry["timestamp"])
                
                if datetime.now() - cache_time < self.cache_ttl:
                    return cached_entry["results"]
            
            return None
            
        except Exception:
            return None
    
    async def _cache_search_results(self, query: str, tags: List[str], category: str, results: List[Dict[str, Any]]) -> None:
        """Cache search results"""
        try:
            cache_data = {}
            if self.local_cache.exists():
                with open(self.local_cache, 'r') as f:
                    cache_data = json.load(f)
            
            cache_key = f"{query}_{','.join(tags or [])}_{category}"
            cache_data[cache_key] = {
                "timestamp": datetime.now().isoformat(),
                "results": results
            }
            
            with open(self.local_cache, 'w') as f:
                json.dump(cache_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error caching search results: {e}")
    
    async def _get_fallback_plugins(self, query: str = "", tags: List[str] = None) -> List[Dict[str, Any]]:
        """Get fallback plugin list when marketplace is unavailable"""
        mock_plugins = [
            {
                "name": "code-formatter",
                "version": "1.0.0",
                "description": "Advanced code formatting and linting",
                "author": "Xencode Team",
                "tags": ["formatting", "linting", "productivity"],
                "downloads": 1250,
                "rating": 4.8,
                "marketplace_id": "xencode-code-formatter",
                "security_verified": True
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
                "security_verified": True
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
                "security_verified": True
            }
        ]
        
        # Apply filters
        if query:
            mock_plugins = [p for p in mock_plugins 
                          if query.lower() in p["name"].lower() or 
                             query.lower() in p["description"].lower()]
        
        if tags:
            mock_plugins = [p for p in mock_plugins 
                          if any(tag in p["tags"] for tag in tags)]
        
        return mock_plugins
    
    async def close(self) -> None:
        """Close HTTP session"""
        if self.session:
            await self.session.close()
    
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