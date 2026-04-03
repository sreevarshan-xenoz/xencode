#!/usr/bin/env python3
"""
Feature System Demonstration

Demonstrates the feature module architecture capabilities.
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from xencode.features.base import FeatureBase, FeatureConfig, FeatureStatus
from xencode.features.manager import FeatureManager
from xencode.features.core.schema import (
    FeatureSchema,
    SchemaField,
    SchemaType,
    SchemaValidator
)


# Example Feature Implementation
class DemoFeature(FeatureBase):
    """Demo feature to showcase the system"""
    
    @property
    def name(self) -> str:
        return "demo_feature"
    
    @property
    def description(self) -> str:
        return "A demonstration feature showing the capabilities of the feature system"
    
    async def _initialize(self) -> None:
        """Initialize the demo feature"""
        print(f"  ✓ Initializing {self.name}...")
        print(f"    Config: {self.config.config}")
        self.track_analytics("initialized", {"version": self.version})
    
    async def _shutdown(self) -> None:
        """Shutdown the demo feature"""
        print(f"  ✓ Shutting down {self.name}...")
        self.track_analytics("shutdown")
    
    def get_cli_commands(self):
        """Get CLI commands"""
        return []
    
    def get_tui_components(self):
        """Get TUI components"""
        return []


async def demo_basic_feature():
    """Demonstrate basic feature creation and lifecycle"""
    print("\n" + "="*60)
    print("DEMO 1: Basic Feature Creation and Lifecycle")
    print("="*60)
    
    # Create a feature config
    config = FeatureConfig(
        name="demo",
        enabled=True,
        version="1.0.0",
        config={
            "setting1": "value1",
            "setting2": 42
        }
    )
    
    # Create feature instance
    feature = DemoFeature(config)
    
    print(f"\n1. Feature Created:")
    print(f"   Name: {feature.name}")
    print(f"   Description: {feature.description}")
    print(f"   Status: {feature.status.value}")
    print(f"   Enabled: {feature.is_enabled}")
    print(f"   Initialized: {feature.is_initialized}")
    
    # Initialize feature
    print(f"\n2. Initializing Feature:")
    await feature.initialize()
    print(f"   Status: {feature.status.value}")
    print(f"   Initialized: {feature.is_initialized}")
    
    # Update configuration
    print(f"\n3. Updating Configuration:")
    feature.update_config({"setting3": "new_value"})
    print(f"   Updated config: {feature.config.config}")
    
    # Shutdown feature
    print(f"\n4. Shutting Down Feature:")
    await feature.shutdown()
    print(f"   Status: {feature.status.value}")
    print(f"   Initialized: {feature.is_initialized}")


async def demo_feature_manager():
    """Demonstrate feature manager capabilities"""
    print("\n" + "="*60)
    print("DEMO 2: Feature Manager")
    print("="*60)
    
    # Create manager
    manager = FeatureManager()
    
    # Register demo feature
    manager._feature_classes['demo'] = DemoFeature
    
    print(f"\n1. Available Features:")
    features = manager.get_available_features()
    for feature_name in features[:5]:  # Show first 5
        print(f"   - {feature_name}")
    if len(features) > 5:
        print(f"   ... and {len(features) - 5} more")
    
    # Load a feature
    print(f"\n2. Loading Feature:")
    config = FeatureConfig(
        name="demo",
        config={"demo_setting": "demo_value"}
    )
    feature = manager.load_feature('demo', config)
    print(f"   Loaded: {feature.name}")
    
    # Initialize feature
    print(f"\n3. Initializing Feature:")
    success = await manager.initialize_feature('demo')
    print(f"   Success: {success}")
    
    # Get feature
    print(f"\n4. Getting Feature:")
    loaded_feature = manager.get_feature('demo')
    print(f"   Feature: {loaded_feature.name}")
    print(f"   Status: {loaded_feature.status.value}")
    
    # Get enabled features
    print(f"\n5. Enabled Features:")
    enabled = manager.get_enabled_features()
    print(f"   Count: {len(enabled)}")
    
    # Shutdown
    print(f"\n6. Shutting Down Feature:")
    await manager.shutdown_feature('demo')


def demo_schema_validation():
    """Demonstrate schema validation"""
    print("\n" + "="*60)
    print("DEMO 3: Schema Validation")
    print("="*60)
    
    # Create a schema
    print(f"\n1. Creating Schema:")
    schema = FeatureSchema(
        name="demo_feature",
        version="1.0.0",
        description="Demo feature configuration schema",
        fields={
            "enabled": SchemaField(
                name="enabled",
                type=SchemaType.BOOLEAN,
                required=True,
                default=True,
                description="Enable or disable the feature"
            ),
            "max_items": SchemaField(
                name="max_items",
                type=SchemaType.INTEGER,
                min_value=1,
                max_value=100,
                default=10,
                description="Maximum number of items"
            ),
            "mode": SchemaField(
                name="mode",
                type=SchemaType.ENUM,
                enum_values=["fast", "balanced", "accurate"],
                default="balanced",
                description="Processing mode"
            ),
            "tags": SchemaField(
                name="tags",
                type=SchemaType.ARRAY,
                items_type=SchemaType.STRING,
                description="Feature tags"
            )
        }
    )
    print(f"   Schema: {schema.name} v{schema.version}")
    print(f"   Fields: {len(schema.fields)}")
    
    # Validate valid config
    print(f"\n2. Validating Valid Configuration:")
    valid_config = {
        "enabled": True,
        "max_items": 50,
        "mode": "fast",
        "tags": ["demo", "test"]
    }
    valid, errors = schema.validate(valid_config)
    print(f"   Valid: {valid}")
    print(f"   Errors: {errors}")
    
    # Validate invalid config
    print(f"\n3. Validating Invalid Configuration:")
    invalid_config = {
        "enabled": "yes",  # Should be boolean
        "max_items": 200,  # Out of range
        "mode": "invalid"  # Not in enum
    }
    valid, errors = schema.validate(invalid_config)
    print(f"   Valid: {valid}")
    print(f"   Errors:")
    for error in errors:
        print(f"     - {error}")
    
    # Apply defaults
    print(f"\n4. Applying Default Values:")
    partial_config = {"enabled": False}
    full_config = schema.apply_defaults(partial_config)
    print(f"   Input: {partial_config}")
    print(f"   Output: {full_config}")
    
    # Export schema
    print(f"\n5. Exporting Schema:")
    print(f"   JSON format:")
    json_str = schema.to_json(indent=2)
    print(f"   {len(json_str)} characters")
    
    print(f"\n   YAML format:")
    yaml_str = schema.to_yaml()
    print(f"   {len(yaml_str)} characters")


def demo_config_serialization():
    """Demonstrate configuration serialization"""
    print("\n" + "="*60)
    print("DEMO 4: Configuration Serialization")
    print("="*60)
    
    # Create config
    print(f"\n1. Creating Configuration:")
    config = FeatureConfig(
        name="demo",
        enabled=True,
        version="2.0.0",
        config={
            "setting1": "value1",
            "setting2": 42,
            "nested": {
                "key": "value"
            }
        },
        dependencies=["dep1", "dep2"]
    )
    print(f"   Name: {config.name}")
    print(f"   Version: {config.version}")
    
    # Serialize to dict
    print(f"\n2. Serializing to Dictionary:")
    config_dict = config.to_dict()
    print(f"   Keys: {list(config_dict.keys())}")
    print(f"   Config: {config_dict}")
    
    # Deserialize from dict
    print(f"\n3. Deserializing from Dictionary:")
    config2 = FeatureConfig.from_dict(config_dict)
    print(f"   Name: {config2.name}")
    print(f"   Version: {config2.version}")
    print(f"   Enabled: {config2.enabled}")
    print(f"   Dependencies: {config2.dependencies}")


async def main():
    """Run all demonstrations"""
    print("\n" + "="*60)
    print("XENCODE FEATURE SYSTEM DEMONSTRATION")
    print("="*60)
    print("\nThis demo showcases the feature module architecture:")
    print("  • Feature base classes and lifecycle")
    print("  • Feature discovery and management")
    print("  • Configuration schemas and validation")
    print("  • Analytics integration")
    
    try:
        # Run demos
        await demo_basic_feature()
        await demo_feature_manager()
        demo_schema_validation()
        demo_config_serialization()
        
        print("\n" + "="*60)
        print("DEMONSTRATION COMPLETE")
        print("="*60)
        print("\nThe feature system is ready for use!")
        print("See xencode/features/README.md for more information.")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
