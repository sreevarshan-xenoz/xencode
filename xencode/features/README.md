# Xencode Features System

A comprehensive, modular feature system for the Xencode AI-powered development platform.

## Overview

The Xencode Features System provides a plugin-like architecture for adding new capabilities to Xencode. Each feature is self-contained with its own configuration, CLI commands, TUI components, and API endpoints.

## Architecture

```
xencode/features/
├── base.py                 # Base classes for all features
├── manager.py              # Feature discovery and lifecycle management
├── core/                   # Core infrastructure
│   ├── config.py          # Configuration management
│   ├── schema.py          # YAML/JSON schema validation
│   ├── cli.py             # CLI integration
│   ├── tui.py             # TUI integration
│   └── api.py             # API integration
├── schemas/               # Configuration schemas
│   ├── code_review.yaml
│   ├── terminal_assistant.yaml
│   └── ...
└── [feature_modules]/     # Individual feature implementations
    ├── code_review/
    ├── terminal_assistant/
    └── ...
```

## Creating a New Feature

### 1. Define Your Feature Class

Create a new Python file in `xencode/features/` or a subdirectory:

```python
from xencode.features.base import FeatureBase, FeatureConfig, FeatureStatus
from typing import List, Any

class MyFeature(FeatureBase):
    """My awesome new feature"""
    
    @property
    def name(self) -> str:
        return "my_feature"
    
    @property
    def description(self) -> str:
        return "Description of what my feature does"
    
    async def _initialize(self) -> None:
        """Initialize the feature"""
        # Setup code here
        self.track_analytics("initialized")
    
    async def _shutdown(self) -> None:
        """Shutdown the feature"""
        # Cleanup code here
        self.track_analytics("shutdown")
    
    def get_cli_commands(self) -> List[Any]:
        """Return CLI commands for this feature"""
        import click
        
        @click.command()
        def my_command():
            """My feature command"""
            print("Hello from my feature!")
        
        return [my_command]
    
    def get_tui_components(self) -> List[Any]:
        """Return TUI components for this feature"""
        # Return Textual widgets/screens
        return []
```

### 2. Create a Configuration Schema (Optional)

Create a YAML schema in `xencode/features/schemas/my_feature.yaml`:

```yaml
$schema: http://json-schema.org/draft-07/schema#
title: my_feature
version: 1.0.0
description: Configuration schema for My Feature
type: object

properties:
  enabled:
    type: boolean
    description: Enable or disable the feature
    default: true
  
  setting1:
    type: string
    description: A string setting
    default: "default_value"
  
  setting2:
    type: integer
    description: A numeric setting
    default: 42
    minimum: 1
    maximum: 100

required:
  - enabled
```

### 3. Use the Feature

The feature will be automatically discovered by the FeatureManager:

```python
from xencode.features import FeatureManager

# Create manager
manager = FeatureManager()

# List available features
features = manager.get_available_features()
print(features)  # ['my_feature', 'code_review', ...]

# Load and initialize a feature
await manager.initialize_feature('my_feature')

# Get the feature instance
feature = manager.get_feature('my_feature')

# Use the feature
feature.track_analytics('custom_event', {'key': 'value'})
```

## Feature Configuration

### Configuration Files

Features can be configured via JSON files in `.xencode/features.json`:

```json
{
  "my_feature": {
    "enabled": true,
    "version": "1.0.0",
    "config": {
      "setting1": "custom_value",
      "setting2": 50
    },
    "dependencies": []
  }
}
```

### Programmatic Configuration

```python
from xencode.features import FeatureConfig

config = FeatureConfig(
    name="my_feature",
    enabled=True,
    version="1.0.0",
    config={
        "setting1": "custom_value",
        "setting2": 50
    }
)

await manager.initialize_feature('my_feature', config)
```

### Schema Validation

Configurations are automatically validated against schemas:

```python
from xencode.features.core import schema_validator

# Validate a configuration
valid, errors = schema_validator.validate_config('my_feature', {
    'enabled': True,
    'setting1': 'value',
    'setting2': 50
})

if not valid:
    print("Validation errors:", errors)

# Apply default values
config = schema_validator.apply_defaults('my_feature', {
    'enabled': True
})
# config now includes default values for setting1 and setting2
```

## Analytics Integration

Features automatically integrate with Xencode's analytics system:

```python
class MyFeature(FeatureBase):
    async def do_something(self):
        # Track an event
        self.track_analytics('action_performed', {
            'action_type': 'example',
            'success': True
        })
```

Analytics events include:
- Feature name and version
- Feature status
- Custom properties
- Automatic tagging

## CLI Integration

Features can register CLI commands that are automatically added to the `xencode` CLI:

```bash
# List all features
xencode features list

# Enable a feature
xencode features enable my_feature

# Disable a feature
xencode features disable my_feature

# Check feature status
xencode features status my_feature

# Get feature info
xencode features info my_feature

# Feature-specific commands
xencode my-feature-command
```

## TUI Integration

Features can provide Textual widgets and screens for the TUI:

```python
from textual.widgets import Static
from textual.containers import Container

class MyFeature(FeatureBase):
    def get_tui_components(self) -> List[Any]:
        class MyFeaturePanel(Container):
            def compose(self):
                yield Static("My Feature Panel")
        
        return [MyFeaturePanel]
```

## API Integration

Features can expose REST API endpoints:

```python
from fastapi import APIRouter

class MyFeature(FeatureBase):
    def get_api_endpoints(self) -> List[Any]:
        router = APIRouter(prefix="/my-feature")
        
        @router.get("/status")
        async def get_status():
            return {"status": "ok"}
        
        return [router]
```

## Feature Lifecycle

1. **Discovery**: FeatureManager scans for feature classes
2. **Loading**: Feature class is instantiated with configuration
3. **Initialization**: `_initialize()` is called, feature becomes active
4. **Usage**: Feature provides functionality via CLI/TUI/API
5. **Shutdown**: `_shutdown()` is called, resources are cleaned up

## Best Practices

### 1. Keep Features Self-Contained
- Each feature should be independent
- Minimize dependencies on other features
- Use the plugin architecture for extensibility

### 2. Provide Clear Configuration
- Define schemas for all configuration options
- Use sensible defaults
- Document all settings

### 3. Track Analytics
- Track important user actions
- Track errors and failures
- Track performance metrics

### 4. Handle Errors Gracefully
- Catch and log exceptions
- Provide helpful error messages
- Don't crash the entire system

### 5. Write Tests
- Unit tests for core functionality
- Integration tests for CLI/TUI/API
- Test configuration validation

## Available Features

### 1. AI Code Reviewer
Automated, intelligent code reviews with security checks and style analysis.

**CLI**: `xencode review pr <url>`, `xencode review file <path>`

### 2. Terminal Assistant
Context-aware command suggestions and intelligent error handling.

**CLI**: `xencode terminal suggest`, `xencode terminal explain <command>`

### 3. Project Analyzer
Automatic project documentation and architecture analysis.

**CLI**: `xencode analyze project <path>`, `xencode analyze generate-docs`

### 4. Learning Mode
Interactive tutorials and adaptive learning paths.

**CLI**: `xencode learn start <topic>`, `xencode learn progress`

### 5. Multi-language Support
10+ language support with context-aware translation.

**CLI**: `xencode lang set <language>`, `xencode lang list`

### 6. Voice Interface
Voice commands and speech synthesis for hands-free interaction.

**CLI**: `xencode voice start`, `xencode voice commands`

### 7. Custom AI Models
Fine-tune models on your codebase for personalized assistance.

**CLI**: `xencode models custom train <name>`, `xencode models custom list`

### 8. Security Auditor
Proactive vulnerability scanning and security analysis.

**CLI**: `xencode security scan <path>`, `xencode security audit`

### 9. Performance Profiler
Code performance analysis and optimization suggestions.

**CLI**: `xencode profile run <path>`, `xencode profile optimize`

### 10. Collaborative Coding
Real-time collaborative editing with conflict resolution.

**CLI**: `xencode collab start <room>`, `xencode collab join <room>`

## Development

### Running Tests

```bash
pytest tests/features/
```

### Adding Dependencies

Update `pyproject.toml` or `requirements.txt` with feature-specific dependencies.

### Documentation

Each feature should include:
- Docstrings in the feature class
- Configuration schema with descriptions
- Usage examples in CLI help text
- README in the feature directory (for complex features)

## Troubleshooting

### Feature Not Loading

1. Check that the feature class inherits from `FeatureBase`
2. Verify the module is in the `xencode/features/` directory
3. Check for import errors in the feature module

### Configuration Errors

1. Validate configuration against schema
2. Check for required fields
3. Verify data types match schema

### Analytics Not Working

1. Ensure analytics system is initialized
2. Check that `event_tracker` is available
3. Verify analytics storage path is writable

## Contributing

To contribute a new feature:

1. Create a feature branch
2. Implement the feature following the guidelines above
3. Add tests for the feature
4. Create a configuration schema
5. Update documentation
6. Submit a pull request

## License

See the main Xencode LICENSE file.
