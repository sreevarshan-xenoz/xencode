"""
Plugin system for Xencode
Provides extensibility through a plugin architecture
"""
import os
import sys
import importlib
import inspect
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type, Protocol
from pathlib import Path
from dataclasses import dataclass
import json


@dataclass
class PluginMetadata:
    """Metadata for a plugin."""
    name: str
    version: str
    description: str
    author: str
    license: str
    dependencies: List[str]


class PluginInterface(ABC):
    """Abstract base class for all plugins."""
    
    def __init__(self, metadata: PluginMetadata):
        self.metadata = metadata
        self.enabled = True
    
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the plugin. Return True if successful."""
        pass
    
    @abstractmethod
    def execute(self, *args, **kwargs) -> Any:
        """Execute the plugin's main functionality."""
        pass
    
    @abstractmethod
    def shutdown(self) -> bool:
        """Shutdown the plugin. Return True if successful."""
        pass


class ToolPlugin(PluginInterface):
    """Base class for plugins that provide tools to the agentic system."""
    
    @abstractmethod
    def get_tools(self) -> List[Any]:
        """Return a list of tools provided by this plugin."""
        pass


class AgentPlugin(PluginInterface):
    """Base class for plugins that extend agent capabilities."""
    
    @abstractmethod
    def extend_agent(self, agent: Any) -> Any:
        """Extend an agent with additional capabilities."""
        pass


class ModelProviderPlugin(PluginInterface):
    """Base class for plugins that add new AI model providers."""
    
    @abstractmethod
    def get_model_interface(self):
        """Return an interface for the model provider."""
        pass


class PluginManager:
    """Manages loading, initializing, and executing plugins."""
    
    def __init__(self, plugin_dirs: Optional[List[Path]] = None):
        self.plugin_dirs = plugin_dirs or [Path("plugins"), Path.home() / ".xencode" / "plugins"]
        self.loaded_plugins: Dict[str, PluginInterface] = {}
        self.plugin_metadata: Dict[str, PluginMetadata] = {}
        self.enabled_plugins: List[str] = []
    
    def discover_plugins(self) -> List[Path]:
        """Discover available plugins in plugin directories."""
        plugin_paths = []
        
        for plugin_dir in self.plugin_dirs:
            if plugin_dir.exists():
                # Look for Python files in the plugin directory
                for py_file in plugin_dir.glob("*.py"):
                    plugin_paths.append(py_file)
                
                # Also look for plugin directories (packages)
                for plugin_subdir in plugin_dir.iterdir():
                    if plugin_subdir.is_dir() and (plugin_subdir / "__init__.py").exists():
                        plugin_paths.append(plugin_subdir)
        
        return plugin_paths
    
    def load_plugin_from_file(self, plugin_path: Path) -> Optional[PluginInterface]:
        """Load a plugin from a Python file."""
        try:
            # Add the plugin directory to the Python path
            plugin_dir = str(plugin_path.parent)
            if plugin_dir not in sys.path:
                sys.path.insert(0, plugin_dir)
            
            # Import the module
            module_name = plugin_path.stem if plugin_path.is_file() else plugin_path.name
            spec = importlib.util.spec_from_file_location(module_name, plugin_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Look for plugin classes in the module
            for name in dir(module):
                obj = getattr(module, name)
                if (inspect.isclass(obj) and 
                    issubclass(obj, PluginInterface) and 
                    obj != PluginInterface):
                    # Instantiate the plugin
                    plugin_instance = obj
                    return plugin_instance
            
            print(f"No plugin class found in {plugin_path}")
            return None
            
        except Exception as e:
            print(f"Error loading plugin from {plugin_path}: {e}")
            return None
    
    def load_plugin_from_package(self, plugin_path: Path) -> Optional[PluginInterface]:
        """Load a plugin from a package directory."""
        try:
            # Add the parent directory to the Python path
            parent_dir = str(plugin_path.parent)
            if parent_dir not in sys.path:
                sys.path.insert(0, parent_dir)
            
            # Import the package
            package_name = plugin_path.name
            module = importlib.import_module(package_name)
            
            # Look for plugin classes in the package
            for name in dir(module):
                obj = getattr(module, name)
                if (inspect.isclass(obj) and 
                    issubclass(obj, PluginInterface) and 
                    obj != PluginInterface):
                    # Instantiate the plugin
                    plugin_instance = obj
                    return plugin_instance
            
            print(f"No plugin class found in package {plugin_path}")
            return None
            
        except Exception as e:
            print(f"Error loading plugin package from {plugin_path}: {e}")
            return None
    
    def load_plugin(self, plugin_path: Path) -> Optional[PluginInterface]:
        """Load a plugin from a file or package."""
        if plugin_path.is_file() and plugin_path.suffix == ".py":
            return self.load_plugin_from_file(plugin_path)
        elif plugin_path.is_dir():
            return self.load_plugin_from_package(plugin_path)
        else:
            print(f"Invalid plugin path: {plugin_path}")
            return None
    
    def register_plugin(self, plugin: PluginInterface) -> bool:
        """Register a plugin with the manager."""
        try:
            plugin_name = plugin.metadata.name
            
            if plugin_name in self.loaded_plugins:
                print(f"Plugin {plugin_name} already registered")
                return False
            
            # Initialize the plugin
            if plugin.initialize():
                self.loaded_plugins[plugin_name] = plugin
                self.plugin_metadata[plugin_name] = plugin.metadata
                if plugin.enabled:
                    self.enabled_plugins.append(plugin_name)
                print(f"Plugin {plugin_name} registered successfully")
                return True
            else:
                print(f"Failed to initialize plugin {plugin_name}")
                return False
                
        except Exception as e:
            print(f"Error registering plugin: {e}")
            return False
    
    def load_and_register_plugins(self) -> int:
        """Discover and load all available plugins."""
        plugin_paths = self.discover_plugins()
        loaded_count = 0
        
        for plugin_path in plugin_paths:
            plugin = self.load_plugin(plugin_path)
            if plugin and self.register_plugin(plugin):
                loaded_count += 1
        
        return loaded_count
    
    def execute_plugin(self, plugin_name: str, *args, **kwargs) -> Any:
        """Execute a plugin."""
        if plugin_name not in self.loaded_plugins:
            raise ValueError(f"Plugin {plugin_name} not loaded")
        
        plugin = self.loaded_plugins[plugin_name]
        if not plugin.enabled:
            raise ValueError(f"Plugin {plugin_name} is disabled")
        
        return plugin.execute(*args, **kwargs)
    
    def get_plugin(self, plugin_name: str) -> Optional[PluginInterface]:
        """Get a plugin instance."""
        return self.loaded_plugins.get(plugin_name)
    
    def enable_plugin(self, plugin_name: str) -> bool:
        """Enable a plugin."""
        if plugin_name in self.loaded_plugins:
            self.loaded_plugins[plugin_name].enabled = True
            if plugin_name not in self.enabled_plugins:
                self.enabled_plugins.append(plugin_name)
            return True
        return False
    
    def disable_plugin(self, plugin_name: str) -> bool:
        """Disable a plugin."""
        if plugin_name in self.loaded_plugins:
            plugin = self.loaded_plugins[plugin_name]
            plugin.enabled = False
            if plugin_name in self.enabled_plugins:
                self.enabled_plugins.remove(plugin_name)
            
            # Shutdown the plugin
            plugin.shutdown()
            return True
        return False
    
    def get_enabled_plugins(self) -> List[str]:
        """Get list of enabled plugins."""
        return self.enabled_plugins[:]
    
    def get_all_plugins(self) -> List[str]:
        """Get list of all loaded plugins."""
        return list(self.loaded_plugins.keys())
    
    def get_plugin_metadata(self, plugin_name: str) -> Optional[PluginMetadata]:
        """Get metadata for a plugin."""
        return self.plugin_metadata.get(plugin_name)
    
    def shutdown_all_plugins(self):
        """Shutdown all loaded plugins."""
        for plugin_name in list(self.loaded_plugins.keys()):
            plugin = self.loaded_plugins[plugin_name]
            try:
                plugin.shutdown()
            except Exception as e:
                print(f"Error shutting down plugin {plugin_name}: {e}")
        
        self.loaded_plugins.clear()
        self.enabled_plugins.clear()


class PluginRegistry:
    """Global registry for plugins."""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.plugin_manager = PluginManager()
            self._initialized = True
    
    def get_manager(self) -> PluginManager:
        """Get the plugin manager instance."""
        return self.plugin_manager


def get_plugin_manager() -> PluginManager:
    """Get the global plugin manager."""
    registry = PluginRegistry()
    return registry.get_manager()


# Example plugin implementations
class ExampleToolPlugin(ToolPlugin):
    """Example tool plugin."""
    
    def __init__(self):
        metadata = PluginMetadata(
            name="example_tool",
            version="1.0.0",
            description="Example tool plugin",
            author="Xencode Team",
            license="MIT",
            dependencies=[]
        )
        super().__init__(metadata)
    
    def initialize(self) -> bool:
        print(f"Initializing {self.metadata.name} plugin")
        return True
    
    def execute(self, *args, **kwargs) -> Any:
        print(f"Executing {self.metadata.name} plugin")
        return "Example tool plugin executed successfully"
    
    def shutdown(self) -> bool:
        print(f"Shutting down {self.metadata.name} plugin")
        return True
    
    def get_tools(self) -> List[Any]:
        # In a real implementation, this would return actual LangChain tools
        return ["example_tool_1", "example_tool_2"]


class ExampleAgentPlugin(AgentPlugin):
    """Example agent plugin."""
    
    def __init__(self):
        metadata = PluginMetadata(
            name="example_agent",
            version="1.0.0",
            description="Example agent plugin",
            author="Xencode Team",
            license="MIT",
            dependencies=[]
        )
        super().__init__(metadata)
    
    def initialize(self) -> bool:
        print(f"Initializing {self.metadata.name} plugin")
        return True
    
    def execute(self, *args, **kwargs) -> Any:
        print(f"Executing {self.metadata.name} plugin")
        return "Example agent plugin executed successfully"
    
    def shutdown(self) -> bool:
        print(f"Shutting down {self.metadata.name} plugin")
        return True
    
    def extend_agent(self, agent: Any) -> Any:
        print(f"Extending agent with {self.metadata.name} capabilities")
        return agent  # Return extended agent


# Register the plugin types
__all__ = [
    'PluginInterface',
    'ToolPlugin',
    'AgentPlugin',
    'ModelProviderPlugin',
    'PluginMetadata',
    'PluginManager',
    'PluginRegistry',
    'get_plugin_manager',
    'ExampleToolPlugin',
    'ExampleAgentPlugin'
]