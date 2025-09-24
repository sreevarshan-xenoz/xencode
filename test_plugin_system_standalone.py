#!/usr/bin/env python3
"""
Standalone test for Plugin System

Tests plugin functionality without importing the main xencode package.
"""

import asyncio
import pytest
import tempfile
import shutil
from pathlib import Path
import sys

# Add the plugin system directly to path
sys.path.insert(0, str(Path(__file__).parent / "xencode"))

from plugin_system import (
    PluginManager, PluginInterface, PluginMetadata, PluginConfig,
    PluginContext, LoadedPlugin, PluginMarketplace
)


class TestPlugin(PluginInterface):
    """Test plugin implementation"""
    
    def __init__(self, name="test-plugin", should_fail=False):
        self.name = name
        self.should_fail = should_fail
        self.initialized = False
        self.shutdown_called = False
    
    async def initialize(self, context: PluginContext) -> bool:
        if self.should_fail:
            return False
        self.context = context
        self.initialized = True
        return True
    
    async def shutdown(self) -> None:
        self.shutdown_called = True
    
    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name=self.name,
            version="1.0.0",
            description="Test plugin",
            author="Test Author"
        )


@pytest.fixture
def temp_plugin_dir():
    """Create temporary plugin directory"""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def plugin_manager(temp_plugin_dir):
    """Create plugin manager with temporary directory"""
    return PluginManager(temp_plugin_dir)


class TestPluginManagerBasics:
    """Test basic plugin manager functionality"""
    
    def test_plugin_directory_creation(self, temp_plugin_dir):
        """Test plugin directory structure creation"""
        manager = PluginManager(temp_plugin_dir)
        
        assert (temp_plugin_dir / "enabled").exists()
        assert (temp_plugin_dir / "disabled").exists()
        assert (temp_plugin_dir / "configs").exists()
        assert (temp_plugin_dir / "plugins.yaml").exists()
    
    @pytest.mark.asyncio
    async def test_plugin_discovery(self, plugin_manager, temp_plugin_dir):
        """Test plugin discovery functionality"""
        # Create test plugin file
        plugin_file = temp_plugin_dir / "enabled" / "test_plugin.py"
        plugin_file.write_text("""
class DummyPlugin:
    pass
""")
        
        plugins = await plugin_manager.discover_plugins()
        assert len(plugins) == 1
        assert plugins[0].name == "test_plugin.py"
    
    @pytest.mark.asyncio
    async def test_plugin_loading_success(self, plugin_manager, temp_plugin_dir):
        """Test successful plugin loading"""
        # Create working plugin file
        plugin_file = temp_plugin_dir / "enabled" / "working_plugin.py"
        plugin_content = """
from plugin_system import PluginInterface, PluginMetadata

class WorkingPlugin(PluginInterface):
    async def initialize(self, context): 
        return True
    
    async def shutdown(self): 
        pass
    
    def get_metadata(self): 
        return PluginMetadata(
            name="working-plugin", 
            version="1.0.0", 
            description="Working test plugin", 
            author="Test"
        )
"""
        plugin_file.write_text(plugin_content)
        
        loaded_plugin = await plugin_manager.load_plugin(plugin_file)
        
        assert loaded_plugin is not None
        assert loaded_plugin.metadata.name == "working-plugin"
        assert "working-plugin" in plugin_manager.plugins
    
    @pytest.mark.asyncio
    async def test_plugin_loading_failure(self, plugin_manager, temp_plugin_dir):
        """Test plugin loading failure handling"""
        # Create invalid plugin file
        plugin_file = temp_plugin_dir / "enabled" / "broken_plugin.py"
        plugin_file.write_text("this is not valid python code {{{")
        
        loaded_plugin = await plugin_manager.load_plugin(plugin_file)
        assert loaded_plugin is None


class TestPluginContext:
    """Test plugin context functionality"""
    
    @pytest.mark.asyncio
    async def test_service_registration(self, plugin_manager):
        """Test service registration and retrieval"""
        context = plugin_manager.context
        
        test_service = {"name": "test_service"}
        context.register_service("test_service", test_service)
        
        retrieved_service = context.get_service("test_service")
        assert retrieved_service == test_service
        
        services = context.list_services()
        assert "test_service" in services
    
    @pytest.mark.asyncio
    async def test_event_system(self, plugin_manager):
        """Test event emission and subscription"""
        context = plugin_manager.context
        
        # Track event calls
        received_events = []
        
        def event_handler(data):
            received_events.append(data)
        
        context.subscribe_event("test_event", event_handler)
        
        # Emit event
        test_data = {"message": "test"}
        await context.emit_event("test_event", test_data)
        
        # Verify handler was called
        assert len(received_events) == 1
        assert received_events[0] == test_data


class TestPluginMetadata:
    """Test plugin metadata"""
    
    def test_metadata_creation(self):
        """Test plugin metadata creation"""
        metadata = PluginMetadata(
            name="test-plugin",
            version="1.0.0",
            description="Test plugin",
            author="Test Author"
        )
        
        assert metadata.name == "test-plugin"
        assert metadata.version == "1.0.0"
        assert metadata.license == "MIT"  # default
        assert metadata.dependencies == []  # default
    
    def test_metadata_with_options(self):
        """Test metadata with optional fields"""
        metadata = PluginMetadata(
            name="advanced-plugin",
            version="2.0.0",
            description="Advanced plugin",
            author="Author",
            dependencies=["requests", "pydantic"],
            permissions=["file_access"],
            tags=["productivity", "ai"]
        )
        
        assert len(metadata.dependencies) == 2
        assert "requests" in metadata.dependencies
        assert len(metadata.permissions) == 1
        assert len(metadata.tags) == 2


class TestPluginMarketplace:
    """Test plugin marketplace"""
    
    @pytest.mark.asyncio
    async def test_plugin_search(self, plugin_manager):
        """Test plugin search functionality"""
        marketplace = PluginMarketplace(plugin_manager)
        
        # Search all plugins
        all_plugins = await marketplace.search_plugins()
        assert len(all_plugins) >= 2  # Mock data has at least 2 plugins
        
        # Search with query
        formatter_plugins = await marketplace.search_plugins("formatter")
        assert len(formatter_plugins) > 0
        
        # Search with tags
        ai_plugins = await marketplace.search_plugins(tags=["ai"])
        assert len(ai_plugins) > 0


@pytest.mark.asyncio
async def test_plugin_system_integration():
    """Integration test for the plugin system"""
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        manager = PluginManager(temp_dir)
        
        # Create a test plugin file
        plugin_file = temp_dir / "enabled" / "integration_test.py"
        plugin_content = """
from plugin_system import PluginInterface, PluginMetadata

class IntegrationTestPlugin(PluginInterface):
    def __init__(self):
        self.events_received = []
    
    async def initialize(self, context):
        self.context = context
        context.register_service("integration_test", self)
        context.subscribe_event("test_event", self.handle_event)
        return True
    
    async def shutdown(self):
        pass
    
    def get_metadata(self):
        return PluginMetadata(
            name="integration-test",
            version="1.0.0",
            description="Integration test plugin",
            author="Test"
        )
    
    async def handle_event(self, data):
        self.events_received.append(data)
"""
        plugin_file.write_text(plugin_content)
        
        # Load and test the plugin
        await manager.load_all_plugins()
        assert len(manager.plugins) == 1
        assert "integration-test" in manager.plugins
        
        # Test service registration
        service = manager.context.get_service("integration_test")
        assert service is not None
        
        # Test event system
        await manager.emit_event("test_event", {"test": "data"})
        
        # Get plugin status
        status = await manager.get_plugin_status()
        assert status["total_plugins"] == 1
        assert status["active_plugins"] == 1
        
        # Cleanup
        await manager.shutdown_all_plugins()
        assert len(manager.plugins) == 0
        
    finally:
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])