# Crush Configuration System

This module provides configuration management for the Crush integration in xencode.

## Features

- **Multi-level Configuration**: Supports both user-level (`~/.xencode/config.json`) and project-level (`.xencode.json`) configuration files
- **Configuration Merging**: Project-level configuration takes precedence over user-level
- **Environment Variable Resolution**: Supports `$VAR`, `${VAR}`, and `$(command)` syntax
- **Validation**: Comprehensive validation with helpful error messages
- **Type Safety**: Strongly typed configuration models using dataclasses

## Configuration Structure

```json
{
  "models": {
    "large": {
      "provider": "openai",
      "model": "gpt-4"
    },
    "small": {
      "provider": "openai",
      "model": "gpt-3.5-turbo"
    }
  },
  "providers": {
    "openai": {
      "api_key": "${OPENAI_API_KEY}",
      "base_url": "https://api.openai.com/v1",
      "price_per_1m_input": 30.0,
      "price_per_1m_output": 60.0,
      "timeout": 60,
      "max_retries": 3
    }
  },
  "lsp": {
    "python": {
      "command": "pylsp",
      "args": [],
      "file_patterns": ["*.py"],
      "root_markers": ["pyproject.toml", "setup.py"]
    }
  },
  "mcp": {
    "filesystem": {
      "command": "uvx",
      "args": ["mcp-server-filesystem"],
      "disabled": false,
      "auto_approve": ["read_file"]
    }
  },
  "options": {
    "auto_summarize": true,
    "max_context_tokens": 100000,
    "log_level": "INFO",
    "command_timeout": 60,
    "background_job_timeout": 300,
    "permission_timeout": 300
  },
  "permissions": {
    "skip_requests": false,
    "allowed_tools": ["view", "grep", "ls"],
    "allowed_directories": [],
    "blocked_commands": ["sudo", "rm -rf", "dd"]
  },
  "agents": {
    "default": {
      "name": "default",
      "system_prompt": "You are a helpful AI coding assistant.",
      "tools": ["bash", "view", "edit", "write", "grep", "ls"]
    }
  }
}
```

## Usage

### Loading Configuration

```python
from xencode.crush.config import ConfigLoader

# Load configuration from current directory
config = ConfigLoader.load()

# Load from specific directory
config = ConfigLoader.load(working_dir="/path/to/project")

# Load without validation (not recommended)
config = ConfigLoader.load(validate=False)
```

### Accessing Configuration

```python
# Get models
large_model = config.get_large_model()
small_model = config.get_small_model()

# Get providers
openai_provider = config.get_provider("openai")
api_key = openai_provider.api_key

# Get options
max_tokens = config.options.max_context_tokens
log_level = config.options.log_level

# Check permissions
if config.permissions.is_tool_allowed("bash"):
    # Execute bash tool
    pass

if config.permissions.is_command_blocked("sudo rm -rf /"):
    # Block dangerous command
    pass
```

### Environment Variable Resolution

The configuration system supports three types of environment variable expansion:

1. **Simple variables**: `$VAR`
2. **Braced variables**: `${VAR}`
3. **Command substitution**: `$(command)`

Example:
```json
{
  "providers": {
    "openai": {
      "api_key": "${OPENAI_API_KEY}",
      "base_url": "$(echo https://api.openai.com/v1)"
    }
  }
}
```

### Validation

```python
from xencode.crush.config import ConfigValidator, ValidationError

try:
    warnings = ConfigValidator.validate(config)
    for warning in warnings:
        print(f"Warning: {warning}")
except ValidationError as e:
    print(f"Configuration error: {e}")
```

## Configuration Files

### User-Level Configuration

Location: `~/.xencode/config.json`

This file contains user-wide defaults that apply to all projects.

### Project-Level Configuration

Location: `.xencode.json` (in project root)

This file contains project-specific configuration that overrides user-level settings.

## Models

### Config
Main configuration object containing all settings.

### ProviderConfig
AI provider configuration (API keys, endpoints, pricing).

### SelectedModel
Model selection (provider + model name).

### LSPConfig
Language Server Protocol client configuration.

### MCPConfig
Model Context Protocol server configuration.

### Options
General options (timeouts, logging, etc.).

### Permissions
Permission settings (allowed tools, blocked commands).

### Agent
Agent configuration (system prompt, tools).

## Error Handling

The configuration system provides three types of exceptions:

- **ConfigurationError**: General configuration errors (invalid JSON, missing files)
- **ValidationError**: Validation errors (invalid values, missing required fields)
- **ResolverError**: Environment variable resolution errors

## Best Practices

1. **Use environment variables for secrets**: Never commit API keys to version control
2. **Validate configuration**: Always validate configuration after loading
3. **Use project-level config for project-specific settings**: Keep user config minimal
4. **Set appropriate timeouts**: Adjust timeouts based on your use case
5. **Configure permissions carefully**: Use allowed_tools and blocked_commands to control access

## Example

See `.xencode.example.json` in the project root for a complete example configuration.
