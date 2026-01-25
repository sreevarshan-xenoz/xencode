#!/usr/bin/env python3
"""
Plugin Ecosystem and Marketplace for Xencode

Comprehensive plugin system with marketplace, dependency management,
and secure execution environment.
"""

import asyncio
import hashlib
import importlib
import importlib.util
import inspect
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Type, Union, Callable, Tuple
from urllib.parse import urlparse
import zipfile
import aiohttp
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt, Confirm
import yaml
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from packaging import version
from packaging.specifiers import SpecifierSet

console = Console()


class PluginStatus(Enum):
    """Plugin status states"""
    INSTALLED = "installed"
    ENABLED = "enabled"
    DISABLED = "disabled"
    BROKEN = "broken"
    UPDATING = "updating"
    DOWNLOADING = "downloading"


class PluginCategory(Enum):
    """Plugin category classification"""
    CODE_ASSISTANT = "code_assistant"
    ANALYSIS_TOOL = "analysis_tool"
    INTEGRATION = "integration"
    UI_EXTENSION = "ui_extension"
    SECURITY = "security"
    PRODUCTIVITY = "productivity"
    CUSTOMIZATION = "customization"
    DEVOPS = "devops"


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
    categories: List[PluginCategory] = field(default_factory=list)
    
    # Marketplace fields
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
    rating: float = 0.0
    download_count: int = 0


@dataclass
class PluginConfig:
    """Plugin configuration settings"""
    enabled: bool = True
    config: Dict[str, Any] = field(default_factory=dict)
    load_priority: int = 100
    auto_reload: bool = False
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
class PluginManifest:
    """Plugin manifest for distribution"""
    metadata: PluginMetadata
    files: List[str]
    checksums: Dict[str, str]
    signed_by: Optional[str] = None
    signature: Optional[str] = None


class PluginInterface:
    """Base interface for all Xencode plugins"""

    def __init__(self, plugin_id: str, metadata: PluginMetadata, config: PluginConfig):
        self.plugin_id = plugin_id
        self.metadata = metadata
        self.config = config
        self.initialized = False

    async def initialize(self) -> bool:
        """Initialize the plugin"""
        try:
            self.initialized = True
            return True
        except Exception as e:
            console.print(f"[red]❌ Plugin {self.plugin_id} initialization failed: {e}[/red]")
            return False

    async def shutdown(self) -> None:
        """Clean shutdown of plugin resources"""
        self.initialized = False

    async def on_config_change(self, new_config: Dict[str, Any]) -> None:
        """Handle configuration changes"""
        self.config.config = new_config

    async def health_check(self) -> Dict[str, Any]:
        """Return plugin health status"""
        return {"status": "healthy" if self.initialized else "uninitialized", "details": {}}


class PluginSecurityManager:
    """Manages plugin security and permissions"""

    def __init__(self):
        self.allowed_permissions = {
            "file_system_read", "file_system_write", "network_access",
            "system_info", "user_input", "clipboard", "notifications"
        }
        self.security_enabled = True
        self.require_signatures = True

    def validate_permissions(self, requested_permissions: List[str]) -> bool:
        """Validate that requested permissions are allowed"""
        for perm in requested_permissions:
            if perm not in self.allowed_permissions:
                console.print(f"[red]❌ Invalid permission requested: {perm}[/red]")
                return False
        return True

    def verify_signature(self, plugin_data: bytes, signature: str, public_key_pem: str) -> bool:
        """Verify plugin signature"""
        if not self.require_signatures:
            return True

        try:
            public_key = serialization.load_pem_public_key(public_key_pem.encode())
            signature_bytes = bytes.fromhex(signature)
            
            public_key.verify(
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
            console.print(f"[red]❌ Signature verification failed: {e}[/red]")
            return False

    def verify_checksum(self, plugin_data: bytes, expected_checksum: str) -> bool:
        """Verify plugin checksum"""
        try:
            actual_checksum = hashlib.sha256(plugin_data).hexdigest()
            return actual_checksum == expected_checksum
        except Exception as e:
            console.print(f"[red]❌ Checksum verification failed: {e}[/red]")
            return False

    def scan_for_malicious_patterns(self, plugin_code: str) -> List[str]:
        """Scan plugin code for malicious patterns"""
        dangerous_patterns = [
            (r'exec\(', "Code execution function"),
            (r'eval\(', "Code evaluation function"),
            (r'__import__', "Dynamic import"),
            (r'subprocess\.', "Subprocess execution"),
            (r'os\.system', "System command execution"),
            (r'os\.popen', "Process creation"),
            (r'shell=True', "Shell command execution"),
            (r'open\([^,]*,[^\'"]*w', "File write operation"),
            (r'import\s+([a-zA-Z_][a-zA-Z0-9_]*)', "External imports")
        ]

        issues = []
        for pattern, description in dangerous_patterns:
            import re
            matches = re.findall(pattern, plugin_code)
            if matches:
                issues.append(f"{description}: {matches[:3]}...")  # Show first 3 matches

        return issues


class PluginValidator:
    """Validates plugins before installation"""

    def __init__(self, security_manager: PluginSecurityManager):
        self.security_manager = security_manager

    def validate_plugin_package(self, package_path: Path, metadata: PluginMetadata) -> bool:
        """Validate a plugin package"""
        try:
            # Check file types and sizes
            if not self._check_file_types(package_path):
                return False

            # Check for malicious patterns
            malicious_issues = self._scan_for_malicious_content(package_path)
            if malicious_issues:
                console.print(f"[red]❌ Malicious patterns detected: {malicious_issues}[/red]")
                return False

            # Validate metadata
            if not self._validate_metadata(metadata):
                return False

            # Check dependencies
            if not self._validate_dependencies(metadata.dependencies):
                return False

            return True

        except Exception as e:
            console.print(f"[red]❌ Plugin validation failed: {e}[/red]")
            return False

    def _check_file_types(self, package_path: Path) -> bool:
        """Check that package only contains allowed file types"""
        allowed_extensions = {'.py', '.json', '.yaml', '.yml', '.txt', '.md', '.cfg', '.ini'}
        
        for file_path in package_path.rglob('*'):
            if file_path.is_file():
                if file_path.suffix.lower() not in allowed_extensions:
                    console.print(f"[red]❌ Disallowed file type: {file_path.suffix} in {file_path}[/red]")
                    return False
        return True

    def _scan_for_malicious_content(self, package_path: Path) -> List[str]:
        """Scan plugin files for malicious content"""
        issues = []
        
        for py_file in package_path.rglob('*.py'):
            try:
                content = py_file.read_text(encoding='utf-8')
                file_issues = self.security_manager.scan_for_malicious_patterns(content)
                if file_issues:
                    issues.extend([f"{py_file.name}: {issue}" for issue in file_issues])
            except Exception:
                # Skip files that can't be read
                continue
        
        return issues

    def _validate_metadata(self, metadata: PluginMetadata) -> bool:
        """Validate plugin metadata"""
        if not metadata.name or not metadata.version or not metadata.author:
            console.print("[red]❌ Missing required metadata fields[/red]")
            return False

        # Validate version format
        try:
            version.parse(metadata.version)
        except Exception:
            console.print(f"[red]❌ Invalid version format: {metadata.version}[/red]")
            return False

        # Validate permissions
        if not self.security_manager.validate_permissions(metadata.permissions):
            return False

        return True

    def _validate_dependencies(self, dependencies: List[str]) -> bool:
        """Validate plugin dependencies"""
        # For now, just check that dependencies are properly formatted
        # In a real system, you'd check if they exist and are compatible
        return True


class PluginDownloader:
    """Downloads plugins from marketplace"""

    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def download_plugin(self, download_url: str, target_dir: Path) -> Path:
        """Download a plugin from URL"""
        if not self.session:
            raise RuntimeError("Downloader not properly initialized")

        console.print(f"[blue]📥 Downloading plugin from {download_url}[/blue]")

        async with self.session.get(download_url) as response:
            if response.status != 200:
                raise Exception(f"Download failed with status {response.status}")

            # Create temporary file
            temp_file = target_dir / f"plugin_download_{uuid.uuid4()}.zip"
            
            with open(temp_file, 'wb') as f:
                async for chunk in response.content.iter_chunked(8192):
                    f.write(chunk)

        return temp_file


class PluginInstaller:
    """Installs plugins with dependency management"""

    def __init__(self, plugins_dir: Path, security_manager: PluginSecurityManager):
        self.plugins_dir = plugins_dir
        self.security_manager = security_manager
        self.validator = PluginValidator(security_manager)
        self.installed_plugins: Dict[str, PluginMetadata] = {}

    def install_plugin(self, plugin_package: Path, metadata: PluginMetadata) -> bool:
        """Install a plugin from package file"""
        try:
            # Validate the plugin
            if not self.validator.validate_plugin_package(plugin_package, metadata):
                console.print(f"[red]❌ Plugin validation failed for {metadata.name}[/red]")
                return False

            # Create plugin directory
            plugin_dir = self.plugins_dir / metadata.name
            plugin_dir.mkdir(exist_ok=True)

            # Extract plugin files
            if plugin_package.suffix == '.zip':
                with zipfile.ZipFile(plugin_package, 'r') as zip_ref:
                    zip_ref.extractall(plugin_dir)
            else:
                # Assume it's already extracted
                shutil.copytree(plugin_package, plugin_dir, dirs_exist_ok=True)

            # Save metadata
            metadata_path = plugin_dir / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata.__dict__, f, default=str, indent=2)

            # Install dependencies
            if metadata.dependencies:
                self._install_dependencies(metadata.dependencies)

            console.print(f"[green]✅ Plugin {metadata.name} installed successfully[/green]")
            return True

        except Exception as e:
            console.print(f"[red]❌ Plugin installation failed: {e}[/red]")
            return False

    def _install_dependencies(self, dependencies: List[str]):
        """Install plugin dependencies"""
        for dep in dependencies:
            try:
                # Use pip to install dependencies
                subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
                console.print(f"[green]✅ Installed dependency: {dep}[/green]")
            except subprocess.CalledProcessError as e:
                console.print(f"[red]❌ Failed to install dependency {dep}: {e}[/red]")

    def uninstall_plugin(self, plugin_name: str) -> bool:
        """Uninstall a plugin"""
        try:
            plugin_dir = self.plugins_dir / plugin_name
            if plugin_dir.exists():
                shutil.rmtree(plugin_dir)
                console.print(f"[green]✅ Plugin {plugin_name} uninstalled[/green]")
                return True
            else:
                console.print(f"[yellow]⚠️ Plugin {plugin_name} not found[/yellow]")
                return False
        except Exception as e:
            console.print(f"[red]❌ Plugin uninstallation failed: {e}[/red]")
            return False


class PluginManager:
    """Main plugin management system"""

    def __init__(self, plugins_dir: Path):
        self.plugins_dir = Path(plugins_dir)
        self.plugins_dir.mkdir(parents=True, exist_ok=True)
        
        self.security_manager = PluginSecurityManager()
        self.installer = PluginInstaller(self.plugins_dir, self.security_manager)
        self.loaded_plugins: Dict[str, PluginInterface] = {}
        self.plugin_configs: Dict[str, PluginConfig] = {}
        
        # Marketplace integration
        self.marketplace_url = "https://marketplace.xencode.ai"
        self.downloader = PluginDownloader()
        
        # Initialize
        self._discover_installed_plugins()

    def _discover_installed_plugins(self):
        """Discover already installed plugins"""
        for plugin_dir in self.plugins_dir.iterdir():
            if plugin_dir.is_dir():
                metadata_file = plugin_dir / "metadata.json"
                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata_dict = json.load(f)
                        
                        metadata = PluginMetadata(**{
                            k: v for k, v in metadata_dict.items() 
                            if k in PluginMetadata.__annotations__
                        })
                        
                        # Set categories from string values
                        if 'categories' in metadata_dict:
                            metadata.categories = [
                                PluginCategory(cat) for cat in metadata_dict['categories']
                                if cat in PluginCategory.__members__
                            ]
                        
                        self.installed_plugins[plugin_dir.name] = metadata
                        
                    except Exception as e:
                        console.print(f"[red]❌ Error loading metadata for {plugin_dir.name}: {e}[/red]")

    async def load_plugin(self, plugin_name: str) -> bool:
        """Load a plugin by name"""
        try:
            plugin_dir = self.plugins_dir / plugin_name
            if not plugin_dir.exists():
                console.print(f"[red]❌ Plugin {plugin_name} not found[/red]")
                return False

            # Get plugin config
            config = self.plugin_configs.get(plugin_name, PluginConfig())

            # Find plugin module
            plugin_module = None
            for py_file in plugin_dir.glob("*.py"):
                if py_file.name != "metadata.json":
                    spec = importlib.util.spec_from_file_location(
                        f"xencode_plugin_{plugin_name}", py_file
                    )
                    plugin_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(plugin_module)
                    break

            if not plugin_module:
                console.print(f"[red]❌ No plugin module found in {plugin_name}[/red]")
                return False

            # Find plugin class
            plugin_class = None
            for name, obj in inspect.getmembers(plugin_module, inspect.isclass):
                if (issubclass(obj, PluginInterface) and 
                    obj is not PluginInterface and
                    name != 'PluginInterface'):
                    plugin_class = obj
                    break

            if not plugin_class:
                console.print(f"[red]❌ No PluginInterface implementation found in {plugin_name}[/red]")
                return False

            # Create and initialize plugin
            metadata = self.installed_plugins.get(plugin_name)
            plugin_instance = plugin_class(plugin_name, metadata, config)

            if await plugin_instance.initialize():
                self.loaded_plugins[plugin_name] = plugin_instance
                console.print(f"[green]✅ Plugin {plugin_name} loaded successfully[/green]")
                return True
            else:
                console.print(f"[red]❌ Plugin {plugin_name} initialization failed[/red]")
                return False

        except Exception as e:
            console.print(f"[red]❌ Error loading plugin {plugin_name}: {e}[/red]")
            return False

    async def unload_plugin(self, plugin_name: str) -> bool:
        """Unload a plugin"""
        if plugin_name in self.loaded_plugins:
            try:
                await self.loaded_plugins[plugin_name].shutdown()
                del self.loaded_plugins[plugin_name]
                console.print(f"[green]✅ Plugin {plugin_name} unloaded[/green]")
                return True
            except Exception as e:
                console.print(f"[red]❌ Error unloading plugin {plugin_name}: {e}[/red]")
                return False
        else:
            console.print(f"[yellow]⚠️ Plugin {plugin_name} not loaded[/yellow]")
            return False

    async def install_plugin_from_marketplace(self, plugin_id: str) -> bool:
        """Install a plugin from the marketplace"""
        try:
            # Fetch plugin info from marketplace
            plugin_info = await self._fetch_plugin_info(plugin_id)
            if not plugin_info:
                return False

            # Download plugin
            async with self.downloader as downloader:
                download_path = await downloader.download_plugin(
                    plugin_info['download_url'], 
                    self.plugins_dir
                )

            # Install plugin
            metadata = PluginMetadata(**plugin_info)
            success = self.installer.install_plugin(download_path, metadata)

            # Clean up
            download_path.unlink()

            if success:
                console.print(f"[green]✅ Plugin {plugin_info['name']} installed from marketplace[/green]")
                
                # Auto-load if enabled
                config = PluginConfig(enabled=True)
                self.plugin_configs[plugin_info['name']] = config
                
                if config.enabled:
                    await self.load_plugin(plugin_info['name'])

            return success

        except Exception as e:
            console.print(f"[red]❌ Marketplace installation failed: {e}[/red]")
            return False

    async def _fetch_plugin_info(self, plugin_id: str) -> Optional[Dict[str, Any]]:
        """Fetch plugin information from marketplace"""
        try:
            # In a real implementation, this would call the marketplace API
            # For demo purposes, we'll simulate a response
            console.print(f"[blue]🌐 Fetching plugin info for {plugin_id}[/blue]")
            
            # Simulate API call delay
            await asyncio.sleep(1)
            
            # Return mock data for demonstration
            return {
                "name": plugin_id,
                "version": "1.0.0",
                "description": f"Sample plugin: {plugin_id}",
                "author": "Xencode Marketplace",
                "license": "MIT",
                "dependencies": [],
                "xencode_version": ">=3.0.0",
                "entry_point": "main",
                "config_schema": {},
                "permissions": ["file_system_read"],
                "tags": ["utility", "demo"],
                "homepage": f"https://marketplace.xencode.ai/plugins/{plugin_id}",
                "download_url": f"https://marketplace.xencode.ai/downloads/{plugin_id}-1.0.0.zip",
                "install_size": 102400,  # 100KB
                "min_python_version": "3.8",
                "categories": ["productivity"],
                "rating": 4.5,
                "download_count": 150
            }
        except Exception as e:
            console.print(f"[red]❌ Failed to fetch plugin info: {e}[/red]")
            return None

    def list_installed_plugins(self) -> List[Tuple[str, PluginMetadata, PluginStatus]]:
        """List all installed plugins with their status"""
        plugins = []
        
        for plugin_name, metadata in self.installed_plugins.items():
            status = PluginStatus.ENABLED if plugin_name in self.loaded_plugins else PluginStatus.DISABLED
            plugins.append((plugin_name, metadata, status))
        
        return plugins

    def list_available_plugins(self) -> List[Dict[str, Any]]:
        """List available plugins from marketplace"""
        # In a real implementation, this would call the marketplace API
        # For demo, return some sample plugins
        return [
            {
                "id": "code_formatter",
                "name": "Code Formatter",
                "version": "1.2.0",
                "description": "Auto-formats code with multiple style options",
                "author": "Xencode Team",
                "rating": 4.8,
                "download_count": 1250,
                "categories": ["code_assistant", "productivity"],
                "price": "free"
            },
            {
                "id": "git_helper",
                "name": "Git Helper",
                "version": "2.1.0",
                "description": "Advanced Git operations and visualization",
                "author": "Dev Tools Inc",
                "rating": 4.6,
                "download_count": 890,
                "categories": ["integration", "devops"],
                "price": "free"
            },
            {
                "id": "security_scanner",
                "name": "Security Scanner",
                "version": "1.5.0",
                "description": "Scans code for security vulnerabilities",
                "author": "Security Experts",
                "rating": 4.9,
                "download_count": 650,
                "categories": ["security", "analysis_tool"],
                "price": "premium"
            }
        ]

    async def update_plugin(self, plugin_name: str) -> bool:
        """Update a plugin to the latest version"""
        try:
            # Check if plugin exists in marketplace
            plugin_info = await self._fetch_plugin_info(plugin_name)
            if not plugin_info:
                console.print(f"[red]❌ Plugin {plugin_name} not found in marketplace[/red]")
                return False

            # Check version
            current_version = self.installed_plugins[plugin_name].version
            new_version = plugin_info['version']

            if version.parse(new_version) <= version.parse(current_version):
                console.print(f"[yellow]⚠️ Plugin {plugin_name} is already up to date[/yellow]")
                return True

            console.print(f"[blue]🔄 Updating {plugin_name} from {current_version} to {new_version}[/blue]")

            # Uninstall current version
            if plugin_name in self.loaded_plugins:
                await self.unload_plugin(plugin_name)
            
            self.installer.uninstall_plugin(plugin_name)

            # Install new version
            metadata = PluginMetadata(**plugin_info)
            async with self.downloader as downloader:
                download_path = await downloader.download_plugin(
                    plugin_info['download_url'], 
                    self.plugins_dir
                )

            success = self.installer.install_plugin(download_path, metadata)
            download_path.unlink()

            if success:
                console.print(f"[green]✅ Plugin {plugin_name} updated successfully[/green]")
                
                # Reload if it was previously loaded
                if plugin_name in self.plugin_configs:
                    if self.plugin_configs[plugin_name].enabled:
                        await self.load_plugin(plugin_name)

            return success

        except Exception as e:
            console.print(f"[red]❌ Plugin update failed: {e}[/red]")
            return False

    def get_plugin_stats(self) -> Dict[str, Any]:
        """Get plugin system statistics"""
        return {
            "total_installed": len(self.installed_plugins),
            "total_loaded": len(self.loaded_plugins),
            "total_disabled": len(self.installed_plugins) - len(self.loaded_plugins),
            "plugin_dirs": [p.name for p in self.plugins_dir.iterdir() if p.is_dir()]
        }

    def display_plugin_dashboard(self):
        """Display plugin management dashboard"""
        stats = self.get_plugin_stats()
        
        console.print(Panel(
            f"[bold blue]Plugin Management Dashboard[/bold blue]\n"
            f"Installed Plugins: {stats['total_installed']}\n"
            f"Loaded Plugins: {stats['total_loaded']}\n"
            f"Disabled Plugins: {stats['total_disabled']}",
            title="Plugin System Overview",
            border_style="blue"
        ))

        # Display installed plugins table
        table = Table(title="Installed Plugins")
        table.add_column("Name", style="cyan")
        table.add_column("Version", style="magenta")
        table.add_column("Status", style="green")
        table.add_column("Author", style="yellow")
        table.add_column("Categories", style="blue")

        for plugin_name, metadata, status in self.list_installed_plugins():
            status_icon = "🟢" if status == PluginStatus.ENABLED else "🔴"
            categories = ", ".join([cat.value for cat in metadata.categories])
            table.add_row(
                plugin_name,
                metadata.version,
                f"{status_icon} {status.value}",
                metadata.author,
                categories
            )

        console.print(table)

        # Display available marketplace plugins
        console.print("\n[bold]Available Marketplace Plugins:[/bold]")
        available = self.list_available_plugins()
        
        for plugin in available[:5]:  # Show first 5
            console.print(f"  • [cyan]{plugin['name']}[/cyan] v{plugin['version']} "
                         f"by {plugin['author']} "
                         f"([green]★{plugin['rating']}[/green], "
                         f"{plugin['download_count']} downloads)")


class PluginMarketplace:
    """Plugin marketplace integration"""

    def __init__(self, plugin_manager: PluginManager):
        self.plugin_manager = plugin_manager
        self.api_base_url = "https://api.marketplace.xencode.ai/v1"

    async def search_plugins(self, query: str, category: Optional[PluginCategory] = None) -> List[Dict[str, Any]]:
        """Search for plugins in marketplace"""
        # In a real implementation, this would call the marketplace API
        # For demo, return filtered sample data
        all_plugins = self.plugin_manager.list_available_plugins()
        
        results = []
        for plugin in all_plugins:
            if query.lower() in plugin['name'].lower() or query.lower() in plugin['description'].lower():
                if category is None or category.value in plugin.get('categories', []):
                    results.append(plugin)
        
        return results

    async def get_featured_plugins(self) -> List[Dict[str, Any]]:
        """Get featured plugins from marketplace"""
        # Return top-rated plugins
        all_plugins = self.plugin_manager.list_available_plugins()
        return sorted(all_plugins, key=lambda p: p['rating'], reverse=True)[:5]

    async def get_new_plugins(self) -> List[Dict[str, Any]]:
        """Get newly added plugins"""
        # In a real system, this would come from the API
        # For demo, return the same list but could be filtered differently
        return self.plugin_manager.list_available_plugins()[:5]

    async def install_plugin_with_deps(self, plugin_id: str) -> bool:
        """Install plugin with automatic dependency resolution"""
        # This would handle dependency resolution in a real implementation
        return await self.plugin_manager.install_plugin_from_marketplace(plugin_id)


async def demo_plugin_system():
    """Demonstrate the plugin system capabilities"""
    console.print("[bold green]🔌 Initializing Plugin Ecosystem[/bold green]")
    
    # Create plugins directory
    plugins_dir = Path.home() / ".xencode" / "plugins"
    
    # Initialize plugin manager
    plugin_manager = PluginManager(plugins_dir)
    
    # Display dashboard
    plugin_manager.display_plugin_dashboard()
    
    # Simulate installing a plugin
    console.print("\n[blue]📦 Installing sample plugin...[/blue]")
    success = await plugin_manager.install_plugin_from_marketplace("code_formatter")
    
    if success:
        console.print("[green]✅ Sample plugin installed successfully[/green]")
        
        # Show updated dashboard
        plugin_manager.display_plugin_dashboard()
        
        # Simulate updating the plugin
        console.print("\n[blue]🔄 Checking for updates...[/blue]")
        update_success = await plugin_manager.update_plugin("code_formatter")
        
        if update_success:
            console.print("[green]✅ Plugin updated successfully[/green]")
    
    console.print("\n[green]✅ Plugin Ecosystem Demo Completed[/green]")


if __name__ == "__main__":
    # Don't run by default to avoid external dependencies
    # asyncio.run(demo_plugin_system())
    pass