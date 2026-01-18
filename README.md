# Xencode - AI-Powered Command Line Interface

Xencode is an advanced command-line interface that leverages large language models to assist with coding tasks, file operations, and project management. Built with modularity, security, and performance in mind.

## Features

- **Modular Architecture**: Clean separation of concerns with dedicated modules for files, models, memory, and caching
- **Multi-Model Support**: Works with Ollama, OpenAI, Google Gemini, and other model providers
- **Conversation Memory**: Maintains context across multiple interactions
- **Smart Caching**: Efficient caching system with disk and memory tiers
- **Security First**: Built-in input validation, sanitization, and rate limiting
- **Performance Optimized**: Connection pooling, lazy loading, and resource management
- **Extensible**: Plugin system and API-ready architecture

## Architecture Overview

```
xencode/
├── core/                 # Core functionality modules
│   ├── files.py          # File operations
│   ├── models.py         # Model management
│   ├── memory.py         # Conversation memory
│   ├── cache.py          # Caching system
│   ├── connection_pool.py # Connection pooling
│   └── __init__.py
├── security/             # Security modules
│   ├── validation.py     # Input validation
│   ├── authentication.py # Authentication
│   ├── rate_limiting.py  # Rate limiting
│   └── data_encryption.py # Data encryption
├── testing/              # Testing utilities
│   └── mock_services.py  # Mock services for testing
└── xencode_core.py       # Main application logic
```

## Core Modules

### Files Module (`xencode/core/files.py`)
Handles all file operations with security validation:
- `create_file(path, content)`
- `read_file(path)`
- `write_file(path, content)`
- `delete_file(path)`

### Models Module (`xencode/core/models.py`)
Manages AI model interactions:
- `ModelManager` class for model lifecycle
- Health checks and performance monitoring
- Support for local (Ollama) and cloud (OpenAI, Gemini) models

### Memory Module (`xencode/core/memory.py`)
Maintains conversation context:
- Session management
- Message history with configurable limits
- Context preservation across interactions

### Cache Module (`xencode/core/cache.py`)
Multi-tier caching system:
- In-memory LRU cache for hot items
- Disk-based persistence
- Compression and TTL management
- Advanced invalidation strategies

## Security Features

### Input Validation (`xencode/security/validation.py`)
- File path validation to prevent directory traversal
- Model name validation
- Prompt injection detection
- API request validation

### Authentication (`xencode/security/authentication.py`)
- API key management
- JWT token support
- HMAC request signing
- Scope-based permissions

### Rate Limiting (`xencode/security/rate_limiting.py`)
- Sliding window rate limiting
- Token bucket algorithm
- Endpoint-specific limits
- Middleware integration

### Data Encryption (`xencode/security/data_encryption.py`)
- Fernet symmetric encryption
- AES-GCM support
- Secure configuration storage
- Sensitive data management

## Performance Optimizations

### Connection Pooling (`xencode/core/connection_pool.py`)
- Thread-safe HTTP connection pools
- Async support with aiohttp
- Retry strategies and backoff
- Resource cleanup

### Lazy Loading (`xencode/core/lazy_loader.py`)
- Deferred loading of heavy components
- Component registry pattern
- Memory efficiency

### Resource Management (`xencode/core/resource_monitor.py`)
- Memory usage monitoring
- CPU and disk usage tracking
- Automatic cleanup mechanisms
- Performance benchmarking

## Testing Strategy

### Unit Tests
Located in `tests/` directory:
- `test_core_modules.py` - Tests for core functionality
- `test_cache_performance.py` - Performance benchmarks
- `test_integration.py` - Integration tests
- `test_property_based.py` - Property-based tests using Hypothesis
- `test_performance_regression.py` - Regression tests
- `test_final_integration.py` - Full system integration tests

### Mock Services (`xencode/testing/mock_services.py`)
- Mock Ollama service
- Mock OpenAI service
- Mock filesystem
- Mock HTTP client
- Mock subprocess operations

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/xencode.git
cd xencode

# Install dependencies
pip install -r requirements.txt

# Install Ollama (for local models)
curl https://ollama.ai/install.sh | sh

# Pull a model
ollama pull qwen2.5:3b
```

## Usage

```bash
# Interactive chat mode
./xencode.sh

# Inline query
./xencode.sh "Explain how to reverse a linked list in Python"

# File operations
./xencode.sh file create myfile.py "print('Hello World')"
./xencode.sh file read myfile.py
```

## Configuration

Xencode uses a configuration system with secure storage:

```python
from xencode.security.data_encryption import set_secure_config, get_secure_config_value

# Store sensitive configuration
set_secure_config("openai_api_key", "your-api-key-here")

# Retrieve configuration
api_key = get_secure_config_value("openai_api_key")
```

## Extending Xencode

### Adding New Commands
1. Add command handler in `xencode_core.py`
2. Update argument parsing
3. Write corresponding tests

### Adding New Model Providers
1. Create provider in `xencode/model_providers/`
2. Implement the provider interface
3. Register the provider in the model manager

### Creating Plugins
1. Follow the plugin architecture in `xencode/plugins/`
2. Implement required interfaces
3. Register the plugin in the system

## Best Practices

1. **Security**: Always validate and sanitize user inputs
2. **Performance**: Use caching appropriately for expensive operations
3. **Testing**: Write unit tests for new functionality
4. **Documentation**: Add docstrings to all public APIs
5. **Error Handling**: Implement graceful degradation for failures

## Contributing

1. Fork the repository
2. Create a feature branch
3. Write tests for your changes
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with ❤️ for the open-source community
- Inspired by the need for secure, efficient AI-powered development tools
- Thanks to all contributors who help make Xencode better