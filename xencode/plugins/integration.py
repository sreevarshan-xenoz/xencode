"""
Plugin integration system for Xencode
Integrates plugins with the existing agentic system
"""
from typing import List, Dict, Any, Optional
from ..agentic import EnhancedToolRegistry, AgentCoordinator
from . import PluginManager, ToolPlugin, AgentPlugin, get_plugin_manager
from .config import get_plugin_config_manager, is_plugin_enabled


class PluginIntegrator:
    """Integrates plugins with the existing Xencode system."""
    
    def __init__(self):
        self.plugin_manager = get_plugin_manager()
        self.config_manager = get_plugin_config_manager()
        self.tool_registry = None
        self.agent_coordinator = None
    
    def set_tool_registry(self, tool_registry: EnhancedToolRegistry):
        """Set the tool registry for plugin integration."""
        self.tool_registry = tool_registry
    
    def set_agent_coordinator(self, agent_coordinator: AgentCoordinator):
        """Set the agent coordinator for plugin integration."""
        self.agent_coordinator = agent_coordinator
    
    def load_and_integrate_plugins(self):
        """Load plugins and integrate them with the system."""
        # Load all available plugins
        loaded_count = self.plugin_manager.load_and_register_plugins()
        print(f"Loaded {loaded_count} plugins")
        
        # Integrate tool plugins
        self._integrate_tool_plugins()
        
        # Integrate agent plugins
        self._integrate_agent_plugins()
    
    def _integrate_tool_plugins(self):
        """Integrate tool plugins with the tool registry."""
        if not self.tool_registry:
            print("Tool registry not set, skipping tool plugin integration")
            return
        
        for plugin_name in self.plugin_manager.get_enabled_plugins():
            plugin = self.plugin_manager.get_plugin(plugin_name)
            
            if isinstance(plugin, ToolPlugin):
                try:
                    tools = plugin.get_tools()
                    for tool in tools:
                        # In a real implementation, we would register actual tools
                        # For now, we'll just acknowledge the integration
                        print(f"Registered tool from {plugin_name}: {tool}")
                        # self.tool_registry.register_tool(tool)  # Actual registration
                except Exception as e:
                    print(f"Error integrating tool plugin {plugin_name}: {e}")
    
    def _integrate_agent_plugins(self):
        """Integrate agent plugins with the agent coordinator."""
        if not self.agent_coordinator:
            print("Agent coordinator not set, skipping agent plugin integration")
            return
        
        for plugin_name in self.plugin_manager.get_enabled_plugins():
            plugin = self.plugin_manager.get_plugin(plugin_name)
            
            if isinstance(plugin, AgentPlugin):
                try:
                    # Extend the agent coordinator with plugin capabilities
                    self.agent_coordinator = plugin.extend_agent(self.agent_coordinator)
                    print(f"Extended agent coordinator with {plugin_name}")
                except Exception as e:
                    print(f"Error integrating agent plugin {plugin_name}: {e}")
    
    def execute_plugin_task(self, plugin_name: str, *args, **kwargs) -> Any:
        """Execute a task using a specific plugin."""
        if not is_plugin_enabled(plugin_name):
            raise ValueError(f"Plugin {plugin_name} is not enabled")
        
        return self.plugin_manager.execute_plugin(plugin_name, *args, **kwargs)
    
    def get_available_plugins(self) -> List[str]:
        """Get list of available plugins."""
        return self.plugin_manager.get_all_plugins()
    
    def get_enabled_plugins(self) -> List[str]:
        """Get list of enabled plugins."""
        return self.plugin_manager.get_enabled_plugins()
    
    def reload_plugins(self):
        """Reload all plugins."""
        # Shutdown all current plugins
        self.plugin_manager.shutdown_all_plugins()
        
        # Load and integrate plugins again
        self.load_and_integrate_plugins()


# Global integrator instance
_integrator = PluginIntegrator()


def get_plugin_integrator() -> PluginIntegrator:
    """Get the global plugin integrator."""
    return _integrator


def initialize_plugin_system(tool_registry=None, agent_coordinator=None):
    """Initialize the plugin system with system components."""
    integrator = get_plugin_integrator()
    
    if tool_registry:
        integrator.set_tool_registry(tool_registry)
    
    if agent_coordinator:
        integrator.set_agent_coordinator(agent_coordinator)
    
    integrator.load_and_integrate_plugins()


def execute_plugin_task(plugin_name: str, *args, **kwargs) -> Any:
    """Execute a task using a specific plugin."""
    integrator = get_plugin_integrator()
    return integrator.execute_plugin_task(plugin_name, *args, **kwargs)


__all__ = [
    'PluginIntegrator',
    'get_plugin_integrator',
    'initialize_plugin_system',
    'execute_plugin_task'
]