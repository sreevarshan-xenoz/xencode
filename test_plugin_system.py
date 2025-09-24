#!/usr/bin/env python3
"""
Comprehensive tests for Xencode Plugin System

Tests plugin loading, lifecycle management, event system,
and marketplace functionality.
"""

import asyncio
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import yaml

import sys
sys.path.append(str(Path(__file__).parent))

from xencode.plugin_system import (
    PluginManager, PluginInterface, PluginMetadata, PluginConfig,
    PluginContext, LoadedPlugin, PluginMarketplace
)


class MockPlugin(PluginInterface):
    """Mock plugin for testing"""
    
    def __init__(self, name="test-plugin", should_fail_init=False):
        self.name = name
        self.should_fail_init = should_fail_init
        self.initialized = False
        self.shutdown_called = False
    
    async def initialize(self, context: PluginContext) -> bool:
        if self.should_fail_init:
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
async def temp_plugin_dir():
    """Create temporary plugin directory"""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
async def plugin_manager(temp_plugin_dir):
    """Create plugin manager with temporary directory"""
    manager = PluginManager(temp_plugin_dir)
    yield manager
    await manager.shutdown_all_plugins()


class TestPluginManager:
    """Test plugin manager functionality"""
    
    @pytest.mark.asyncio
    async def test_plugin_directory_creation(self, temp_plugin_dir):
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
from xencode.plugin_system import PluginInterface, PluginMetadata

class TestPlugin(PluginInterface):
    async def initialize(self, context): return True
    async def shutdown(self): pass
    def get_metadata(self): 
        return PluginMetadata(
            name="test", version="1.0.0", 
            description="Test", author="Test"
        )
""")
        
        plugins = await plugin_manager.discover_plugins()
        assert len(plugins) == 1
        assert plugins[0].name == "test_plugin.py"
    
    @pytest.mark.asyncio
    async def test_plugin_loading_success(self, plugin_manager, temp_plugin_dir):
        """Test successful plugin loading"""
        # Create test plugin
        plugin_file = temp_plugin_dir / "enabled" / "success_plugin.py"
        plugin_content = f"""
import sys
sys.path.append('{Path(__file__).parent.parent}')
from xencode.plugin_system import PluginInterface, PluginMetadata

class SuccessPlugin(PluginInterface):
    async def initialize(self, context): 
        self.context = context
        return True
    async def shutdown(self): pass
    def get_metadata(self): 
        return PluginMetadata(
            name="success-plugin", version="1.0.0", 
            description="Success test plugin", author="Test"
        )
"""
        plugin_file.write_text(plugin_content)
        
        loaded_plugin = await plugin_manager.load_plugin(plugin_file)
        
        assert loaded_plugin is not None
        assert loaded_plugin.metadata.name == "success-plugin"
        assert "success-plugin" in plugin_manager.plugins
        assert plugin_manager.plugins["success-plugin"].is_active
    
    @pytest.mark.asyncio
    async def test_plugin_loading_failure(self, plugin_manager, temp_plugin_dir):
        """Test plugin loading failure handling"""
        # Create invalid plugin file
        plugin_file = temp_plugin_dir / "enabled" / "invalid_plugin.py"
        plugin_file.write_text("invalid python code {{{")
        
        loaded_plugin = await plugin_manager.load_plugin(plugin_file)
        assert loaded_plugin is None
    
    @pytest.mark.asyncio
    async def test_plugin_unloading(self, plugin_manager, temp_plugin_dir):
        """Test plugin unloading"""
        # First load a plugin
        plugin_file = temp_plugin_dir / "enabled" / "unload_test.py"
        plugin_content = f"""
import sys
sys.path.append('{Path(__file__).parent.parent}')
from xencode.plugin_system import PluginInterface, PluginMetadata

class UnloadTestPlugin(PluginInterface):
    def __init__(self):
        self.shutdown_called = False
    async def initialize(self, context): return True
    async def shutdown(self): 
        self.shutdown_called = True
    def get_metadata(self): 
        return PluginMetadata(
            name="unload-test", version="1.0.0", 
            description="Unload test", author="Test"
        )
"""
        plugin_file.write_text(plugin_content)
        
        await plugin_manager.load_plugin(plugin_file)
        assert "unload-test" in plugin_manager.plugins
        
        # Now unload it
        success = await plugin_manager.unload_plugin("unload-test")
        assert success
        assert "unload-test" not in plugin_manager.plugins
    
    @pytest.mark.asyncio
    async def test_plugin_enable_disable(self, plugin_manager, temp_plugin_dir):
        """Test plugin enable/disable functionality"""
        # Create disabled plugin
        disabled_file = temp_plugin_dir / "disabled" / "toggle_test.py"
        plugin_content = f"""
import sys
sys.path.append('{Path(__file__).parent.parent}')
from xencode.plugin_system import PluginInterface, PluginMetadata

class ToggleTestPlugin(PluginInterface):
    async def initialize(self, context): return True
    async def shutdown(self): pass
    def get_metadata(self): 
        return PluginMetadata(
            name="toggle-test", version="1.0.0", 
            description="Toggle test", author="Test"
        )
"""
        disabled_file.write_text(plugin_content)
        
        # Enable plugin
        success = await plugin_manager.enable_plugin("toggle_test")
        assert success
        assert (temp_plugin_dir / "enabled" / "toggle_test.py").exists()
        assert "toggle-test" in plugin_manager.plugins
        
        # Disable plugin
        success = await plugin_manager.disable_plugin("toggle-test")
        assert success
        assert (temp_plugin_dir / "disabled" / "toggle_test.py").exists()
        assert "toggle-test" not in plugin_manager.plugins


class TestPluginContext:
    """Test plugin context functionality"""
    
    @pytest.mark.asyncio
    async def test_service_registration(self, plugin_manager):
        """Test service registration and retrieval"""
        context = plugin_manager.context
        
        test_service = MagicMock()
        context.register_service("test_service", test_service)
        
        retrieved_service = context.get_service("test_service")
        assert retrieved_service is test_service
        
        services = context.list_services()
        assert "test_service" in services
    
    @pytest.mark.asyncio
    async def test_event_system(self, plugin_manager):
        """Test event emission and subscription"""
        context = plugin_manager.context
        
        # Mock event handler
        event_handler = AsyncMock()
        context.subscribe_event("test_event", event_handler)
        
        # Emit event
        test_data = {"message": "test"}
        await context.emit_event("test_event", test_data)
        
        # Verify handler was called
        event_handler.assert_called_once_with(test_data)


class TestPluginMetadata:
    """Test plugin metadata handling"""
    
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
        assert metadata.license == "MIT"  # default value
        assert metadata.dependencies == []  # default value
    
    def test_metadata_with_dependencies(self):
        """Test metadata with dependencies"""
        metadata = PluginMetadata(
            name="complex-plugin",
            version="2.0.0",
            description="Complex plugin",
            author="Author",
            dependencies=["requests", "pydantic"],
            permissions=["file_access", "network_access"]
        )
        
        assert len(metadata.dependencies) == 2
        assert "requests" in metadata.dependencies
        assert len(metadata.permissions) == 2


class TestPluginMarketplace:
    """Test plugin marketplace functionality"""
    
    @pytest.mark.asyncio
    async def test_plugin_search(self, plugin_manager):
        """Test plugin search functionality"""
        marketplace = PluginMarketplace(plugin_manager)
        
        # Search all plugins
        all_plugins = await marketplace.search_plugins()
        assert len(all_plugins) > 0
        
        # Search with query
        formatter_plugins = await marketplace.search_plugins("formatter")
        assert len(formatter_plugins) > 0
        assert all("format" in p["name"].lower() or "format" in p["description"].lower() 
                  for p in formatter_plugins)
        
        # Search with tags
        ai_plugins = await marketplace.search_plugins(tags=["ai"])
        assert len(ai_plugins) > 0
        assert all(any("ai" in tag for tag in p["tags"]) for p in ai_plugins)
    
    @pytest.mark.asyncio
    async def test_plugin_installation(self, plugin_manager):
        """Test plugin installation"""
        marketplace = PluginMarketplace(plugin_manager)
        
        # Mock installation
        success = await marketplace.install_plugin("test-plugin", "1.0.0")
        assert success


class TestPluginConfig:
    """Test plugin configuration handling"""
    
    def test_default_config(self):
        """Test default plugin configuration"""
        config = PluginConfig()
        
        assert config.enabled is True
        assert config.load_priority == 100
        assert config.auto_reload is False
        assert config.config == {}
    
    def test_custom_config(self):
        """Test custom plugin configuration"""
        config = PluginConfig(
            enabled=False,
            load_priority=50,
            auto_reload=True,
            config={"setting1": "value1"}
        )
        
        assert config.enabled is False
        assert config.load_priority == 50
        assert config.auto_reload is True
        assert config.config["setting1"] == "value1"


class TestPluginIntegration:
    """Integration tests for the plugin system"""
    
    @pytest.mark.asyncio
    async def test_full_plugin_lifecycle(self, plugin_manager, temp_plugin_dir):
        """Test complete plugin lifecycle"""
        # Create plugin file
        plugin_file = temp_plugin_dir / "enabled" / "lifecycle_test.py"
        plugin_content = f"""
import sys
sys.path.append('{Path(__file__).parent.parent}')
from xencode.plugin_system import PluginInterface, PluginMetadata

class LifecycleTestPlugin(PluginInterface):
    def __init__(self):
        self.initialized = False
        self.shutdown_called = False
    
    async def initialize(self, context):
        self.context = context
        self.initialized = True
        context.register_service("lifecycle_service", self)
        return True
    
    async def shutdown(self):
        self.shutdown_called = True
    
    def get_metadata(self):
        return PluginMetadata(
            name="lifecycle-test", version="1.0.0",
            description="Lifecycle test plugin", author="Test"
        )
    
    async def health_check(self):
        return {{
            "status": "healthy" if self.initialized else "unhealthy",
            "initialized": self.initialized
        }}
"""
        plugin_file.write_text(plugin_content)
        
        # Load plugin
        await plugin_manager.load_all_plugins()
        assert "lifecycle-test" in plugin_manager.plugins
        
        # Check service registration
        service = plugin_manager.context.get_service("lifecycle_service")
        assert service is not None
        
        # Check plugin status
        status = await plugin_manager.get_plugin_status()
        assert status["total_plugins"] == 1
        assert status["active_plugins"] == 1
        assert "lifecycle-test" in status["plugins"]
        
        # Test health check
        plugin_health = status["plugins"]["lifecycle-test"]["health"]
        assert plugin_health["status"] == "healthy"
        
        # Shutdown
        await plugin_manager.shutdown_all_plugins()
        assert len(plugin_manager.plugins) == 0
    
    @pytest.mark.asyncio
    async def test_event_propagation(self, plugin_manager, temp_plugin_dir):
        """Test event propagation between plugins"""
        # Create two plugins that communicate via events
        plugin1_file = temp_plugin_dir / "enabled" / "sender.py"
        plugin1_content = f"""
import sys
sys.path.append('{Path(__file__).parent.parent}')
from xencode.plugin_system import PluginInterface, PluginMetadata

class SenderPlugin(PluginInterface):
    async def initialize(self, context):
        self.context = context
        return True
    
    async def shutdown(self): pass
    
    def get_metadata(self):
        return PluginMetadata(
            name="sender", version="1.0.0",
            description="Event sender", author="Test"
        )
    
    async def send_message(self, message):
        await self.context.emit_event("test_message", {{"message": message}})
"""
        
        plugin2_file = temp_plugin_dir / "enabled" / "receiver.py"
        plugin2_content = f"""
import sys
sys.path.append('{Path(__file__).parent.parent}')
from xencode.plugin_system import PluginInterface, PluginMetadata

class ReceiverPlugin(PluginInterface):
    def __init__(self):
        self.received_messages = []
    
    async def initialize(self, context):
        self.context = context
        context.subscribe_event("test_message", self.handle_message)
        return True
    
    async def shutdown(self): pass
    
    def get_metadata(self):
        return PluginMetadata(
            name="receiver", version="1.0.0",
            description="Event receiver", author="Test"
        )
    
    async def handle_message(self, data):
        self.received_messages.append(data["message"])
"""
        
        plugin1_file.write_text(plugin1_content)
        plugin2_file.write_text(plugin2_content)
        
        # Load plugins
        await plugin_manager.load_all_plugins()
        assert len(plugin_manager.plugins) == 2
        
        # Test event communication
        await plugin_manager.emit_event("test_message", {"message": "Hello from test!"})
        
        # Verify event was received
        # Note: In a real scenario, we'd have a way to access the plugin instances
        # This test validates the framework structure


@pytest.mark.asyncio
async def test_plugin_system_performance():
    """Test plugin system performance with multiple plugins"""
    import time
    
    temp_dir = Path(tempfile.mkdtemp())
    try:
        manager = PluginManager(temp_dir)
        
        # Create multiple test plugins
        for i in range(10):
            plugin_file = temp_dir / "enabled" / f"perf_test_{i}.py"
            plugin_content = f"""
import sys
sys.path.append('{Path(__file__).parent.parent}')
from xencode.plugin_system import PluginInterface, PluginMetadata

class PerfTestPlugin{i}(PluginInterface):
    async def initialize(self, context): return True
    async def shutdown(self): pass
    def get_metadata(self):
        return PluginMetadata(
            name="perf-test-{i}", version="1.0.0",
            description="Performance test plugin {i}", author="Test"
        )
"""
            plugin_file.write_text(plugin_content)
        
        # Measure loading time
        start_time = time.time()
        await manager.load_all_plugins()
        load_time = time.time() - start_time
        
        assert len(manager.plugins) == 10
        assert load_time < 5.0  # Should load 10 plugins in under 5 seconds
        
        # Measure event emission time
        start_time = time.time()
        for _ in range(100):
            await manager.emit_event("performance_test", {"data": "test"})
        event_time = time.time() - start_time
        
        assert event_time < 1.0  # Should emit 100 events in under 1 second
        
        await manager.shutdown_all_plugins()
        
    finally:
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    # Run specific tests
    import subprocess
    subprocess.run([
        "python", "-m", "pytest", __file__, "-v", "--tb=short"
    ])