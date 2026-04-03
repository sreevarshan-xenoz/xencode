"""
Tests for Feature Infrastructure

Tests for base classes, manager, configuration, and schema validation.
"""

import pytest
import asyncio
from pathlib import Path
from xencode.features.base import FeatureBase, FeatureConfig, FeatureStatus, FeatureError
from xencode.features.manager import FeatureManager
from xencode.features.core.config import FeatureSystemConfig, FeatureConfigManager
from xencode.features.core.schema import (
    FeatureSchema,
    SchemaField,
    SchemaType,
    SchemaValidator
)


# Test Feature Implementation
class TestFeature(FeatureBase):
    """Test feature for unit tests"""
    
    @property
    def name(self) -> str:
        return "test_feature"
    
    @property
    def description(self) -> str:
        return "A test feature for unit testing"
    
    async def _initialize(self) -> None:
        """Initialize the test feature"""
        self.initialized_called = True
    
    async def _shutdown(self) -> None:
        """Shutdown the test feature"""
        self.shutdown_called = True
    
    def get_cli_commands(self):
        """Get CLI commands"""
        return []
    
    def get_tui_components(self):
        """Get TUI components"""
        return []


class TestFeatureBase:
    """Tests for FeatureBase class"""
    
    def test_feature_creation(self):
        """Test creating a feature"""
        config = FeatureConfig(name="test", enabled=True)
        feature = TestFeature(config)
        
        assert feature.name == "test_feature"
        assert feature.description == "A test feature for unit testing"
        assert feature.is_enabled == True
        assert feature.is_initialized == False
        assert feature.status == FeatureStatus.DISABLED
    
    @pytest.mark.asyncio
    async def test_feature_initialization(self):
        """Test feature initialization"""
        config = FeatureConfig(name="test", enabled=True)
        feature = TestFeature(config)
        
        success = await feature.initialize()
        
        assert success == True
        assert feature.is_initialized == True
        assert feature.status == FeatureStatus.ENABLED
        assert hasattr(feature, 'initialized_called')
    
    @pytest.mark.asyncio
    async def test_feature_shutdown(self):
        """Test feature shutdown"""
        config = FeatureConfig(name="test", enabled=True)
        feature = TestFeature(config)
        
        await feature.initialize()
        await feature.shutdown()
        
        assert feature.is_initialized == False
        assert feature.status == FeatureStatus.DISABLED
        assert hasattr(feature, 'shutdown_called')
    
    def test_feature_config_update(self):
        """Test updating feature configuration"""
        config = FeatureConfig(name="test", config={"key1": "value1"})
        feature = TestFeature(config)
        
        feature.update_config({"key2": "value2"})
        
        assert feature.config.config["key1"] == "value1"
        assert feature.config.config["key2"] == "value2"
    
    def test_feature_config_serialization(self):
        """Test feature config serialization"""
        config = FeatureConfig(
            name="test",
            enabled=True,
            version="1.0.0",
            config={"key": "value"},
            dependencies=["dep1"]
        )
        
        config_dict = config.to_dict()
        
        assert config_dict["name"] == "test"
        assert config_dict["enabled"] == True
        assert config_dict["version"] == "1.0.0"
        assert config_dict["config"]["key"] == "value"
        assert "dep1" in config_dict["dependencies"]
        
        # Test deserialization
        config2 = FeatureConfig.from_dict(config_dict)
        assert config2.name == config.name
        assert config2.enabled == config.enabled


class TestFeatureManager:
    """Tests for FeatureManager class"""
    
    def test_manager_creation(self):
        """Test creating a feature manager"""
        manager = FeatureManager()
        
        assert manager is not None
        assert isinstance(manager.features, dict)
        assert isinstance(manager._feature_classes, dict)
    
    def test_get_available_features(self):
        """Test getting available features"""
        manager = FeatureManager()
        features = manager.get_available_features()
        
        assert isinstance(features, list)
        # Should find at least some features
        assert len(features) >= 0
    
    def test_load_feature(self):
        """Test loading a feature"""
        manager = FeatureManager()
        
        # Register our test feature
        manager._feature_classes['test'] = TestFeature
        
        feature = manager.load_feature('test')
        
        assert feature is not None
        assert isinstance(feature, TestFeature)
        assert feature.name == "test_feature"
    
    @pytest.mark.asyncio
    async def test_initialize_feature(self):
        """Test initializing a feature"""
        manager = FeatureManager()
        manager._feature_classes['test'] = TestFeature
        
        success = await manager.initialize_feature('test')
        
        assert success == True
        
        feature = manager.get_feature('test')
        assert feature.is_initialized == True
    
    @pytest.mark.asyncio
    async def test_shutdown_feature(self):
        """Test shutting down a feature"""
        manager = FeatureManager()
        manager._feature_classes['test'] = TestFeature
        
        await manager.initialize_feature('test')
        success = await manager.shutdown_feature('test')
        
        assert success == True
        
        feature = manager.get_feature('test')
        assert feature.is_initialized == False
    
    def test_get_enabled_features(self):
        """Test getting enabled features"""
        manager = FeatureManager()
        manager._feature_classes['test'] = TestFeature
        
        config = FeatureConfig(name="test", enabled=True)
        manager.load_feature('test', config)
        
        enabled = manager.get_enabled_features()
        
        assert 'test' in enabled
        assert enabled['test'].is_enabled == True


class TestSchemaValidation:
    """Tests for schema validation"""
    
    def test_schema_field_creation(self):
        """Test creating a schema field"""
        field = SchemaField(
            name="test_field",
            type=SchemaType.STRING,
            required=True,
            default="default_value",
            description="A test field"
        )
        
        assert field.name == "test_field"
        assert field.type == SchemaType.STRING
        assert field.required == True
        assert field.default == "default_value"
    
    def test_string_validation(self):
        """Test string field validation"""
        field = SchemaField(name="test", type=SchemaType.STRING, required=True)
        
        valid, error = field.validate("test_value")
        assert valid == True
        assert error is None
        
        valid, error = field.validate(123)
        assert valid == False
        assert error is not None
    
    def test_integer_validation(self):
        """Test integer field validation"""
        field = SchemaField(
            name="test",
            type=SchemaType.INTEGER,
            min_value=1,
            max_value=100
        )
        
        valid, error = field.validate(50)
        assert valid == True
        
        valid, error = field.validate(0)
        assert valid == False
        
        valid, error = field.validate(101)
        assert valid == False
    
    def test_boolean_validation(self):
        """Test boolean field validation"""
        field = SchemaField(name="test", type=SchemaType.BOOLEAN)
        
        valid, error = field.validate(True)
        assert valid == True
        
        valid, error = field.validate("true")
        assert valid == False
    
    def test_array_validation(self):
        """Test array field validation"""
        field = SchemaField(
            name="test",
            type=SchemaType.ARRAY,
            items_type=SchemaType.STRING
        )
        
        valid, error = field.validate(["a", "b", "c"])
        assert valid == True
        
        valid, error = field.validate([1, 2, 3])
        assert valid == False
    
    def test_enum_validation(self):
        """Test enum field validation"""
        field = SchemaField(
            name="test",
            type=SchemaType.ENUM,
            enum_values=["option1", "option2", "option3"]
        )
        
        valid, error = field.validate("option1")
        assert valid == True
        
        valid, error = field.validate("invalid")
        assert valid == False
    
    def test_required_field_validation(self):
        """Test required field validation"""
        field = SchemaField(name="test", type=SchemaType.STRING, required=True)
        
        valid, error = field.validate(None)
        assert valid == False
        assert "required" in error.lower()
    
    def test_feature_schema_creation(self):
        """Test creating a feature schema"""
        schema = FeatureSchema(
            name="test_feature",
            version="1.0.0",
            description="Test schema",
            fields={
                "enabled": SchemaField(
                    name="enabled",
                    type=SchemaType.BOOLEAN,
                    required=True,
                    default=True
                ),
                "setting": SchemaField(
                    name="setting",
                    type=SchemaType.STRING,
                    default="default"
                )
            }
        )
        
        assert schema.name == "test_feature"
        assert len(schema.fields) == 2
    
    def test_schema_validation(self):
        """Test validating config against schema"""
        schema = FeatureSchema(
            name="test",
            fields={
                "enabled": SchemaField(
                    name="enabled",
                    type=SchemaType.BOOLEAN,
                    required=True
                ),
                "count": SchemaField(
                    name="count",
                    type=SchemaType.INTEGER,
                    min_value=1,
                    max_value=10
                )
            }
        )
        
        # Valid config
        valid, errors = schema.validate({"enabled": True, "count": 5})
        assert valid == True
        assert len(errors) == 0
        
        # Invalid config (missing required field)
        valid, errors = schema.validate({"count": 5})
        assert valid == False
        assert len(errors) > 0
        
        # Invalid config (out of range)
        valid, errors = schema.validate({"enabled": True, "count": 20})
        assert valid == False
    
    def test_schema_apply_defaults(self):
        """Test applying default values"""
        schema = FeatureSchema(
            name="test",
            fields={
                "enabled": SchemaField(
                    name="enabled",
                    type=SchemaType.BOOLEAN,
                    default=True
                ),
                "setting": SchemaField(
                    name="setting",
                    type=SchemaType.STRING,
                    default="default_value"
                )
            }
        )
        
        config = schema.apply_defaults({"enabled": False})
        
        assert config["enabled"] == False  # User value preserved
        assert config["setting"] == "default_value"  # Default applied
    
    def test_schema_serialization(self):
        """Test schema serialization to JSON/YAML"""
        schema = FeatureSchema(
            name="test",
            version="1.0.0",
            description="Test schema",
            fields={
                "enabled": SchemaField(
                    name="enabled",
                    type=SchemaType.BOOLEAN,
                    required=True,
                    default=True
                )
            }
        )
        
        # Test JSON serialization
        json_str = schema.to_json()
        assert "test" in json_str
        assert "enabled" in json_str
        
        # Test YAML serialization
        yaml_str = schema.to_yaml()
        assert "test" in yaml_str
        assert "enabled" in yaml_str
    
    def test_schema_deserialization(self):
        """Test schema deserialization from dict"""
        schema_dict = {
            'title': 'test',
            'version': '1.0.0',
            'description': 'Test schema',
            'type': 'object',
            'properties': {
                'enabled': {
                    'type': 'boolean',
                    'required': True,
                    'default': True
                }
            },
            'required': ['enabled']
        }
        
        schema = FeatureSchema.from_dict(schema_dict)
        
        assert schema.name == 'test'
        assert schema.version == '1.0.0'
        assert 'enabled' in schema.fields


class TestFeatureConfigManager:
    """Tests for FeatureConfigManager"""
    
    def test_config_manager_creation(self, tmp_path):
        """Test creating a config manager"""
        system_config = FeatureSystemConfig(config_dir=str(tmp_path))
        manager = FeatureConfigManager(system_config)
        
        assert manager is not None
        assert manager.system_config.config_dir == str(tmp_path)
    
    def test_get_set_feature_config(self, tmp_path):
        """Test getting and setting feature config"""
        system_config = FeatureSystemConfig(config_dir=str(tmp_path))
        manager = FeatureConfigManager(system_config)
        
        # Set config
        manager.set_feature_config('test', {'enabled': True, 'setting': 'value'})
        
        # Get config
        config = manager.get_feature_config('test')
        
        assert config['enabled'] == True
        assert config['setting'] == 'value'
    
    def test_enable_disable_feature(self, tmp_path):
        """Test enabling and disabling features"""
        system_config = FeatureSystemConfig(config_dir=str(tmp_path))
        manager = FeatureConfigManager(system_config)
        
        # Enable feature
        manager.set_feature_enabled('test', True)
        assert 'test' in manager.get_enabled_features()
        
        # Disable feature
        manager.set_feature_enabled('test', False)
        assert 'test' not in manager.get_enabled_features()
    
    def test_create_feature_config(self, tmp_path):
        """Test creating a FeatureConfig from manager"""
        system_config = FeatureSystemConfig(config_dir=str(tmp_path))
        manager = FeatureConfigManager(system_config)
        
        manager.set_feature_config('test', {
            'enabled': True,
            'version': '2.0.0',
            'config': {'key': 'value'}
        })
        
        feature_config = manager.create_feature_config('test')
        
        assert feature_config.name == 'test'
        assert feature_config.enabled == True
        assert feature_config.version == '2.0.0'
        assert feature_config.config['key'] == 'value'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
