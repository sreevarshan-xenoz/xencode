# Xencode API Documentation

## Overview

Xencode is an AI-powered development assistant platform that integrates with local language models through Ollama. This document describes the public APIs and modules available in the Xencode platform.

## Core Modules

### Files Module (`xencode.core.files`)

Functions for file operations within the Xencode platform.

#### `create_file(path: Union[str, Path], content: str) -> None`
Creates a file with the given content.

- **Parameters**:
  - `path`: Path to the file to create
  - `content`: Content to write to the file
- **Returns**: None
- **Side Effects**: Creates a file and prints a success/failure message to the console

#### `read_file(path: Union[str, Path]) -> str`
Reads the content of a file.

- **Parameters**:
  - `path`: Path to the file to read
- **Returns**: String content of the file, or empty string if an error occurs
- **Side Effects**: Prints file content to the console in a panel

#### `write_file(path: Union[str, Path], content: str) -> None`
Writes content to a file (equivalent to `create_file`).

- **Parameters**:
  - `path`: Path to the file to write
  - `content`: Content to write to the file
- **Returns**: None

#### `delete_file(path: Union[str, Path]) -> bool`
Deletes a file.

- **Parameters**:
  - `path`: Path to the file to delete
- **Returns**: True if deletion was successful, False otherwise
- **Side Effects**: Prints success/failure message to the console

### Models Module (`xencode.core.models`)

Classes and functions for managing AI models.

#### `ModelManager` Class
Manages available models and their health status.

##### Methods:
- `refresh_models()` - Refreshes the list of available models from Ollama
- `check_model_health(model: str) -> bool` - Checks if a model is responsive
- `get_best_model() -> str` - Gets the fastest healthy model
- `switch_model(model: str) -> Tuple[bool, str]` - Attempts to switch to a different model

#### `get_available_models() -> List[str]`
Gets a list of available models from Ollama.

- **Returns**: List of model names

#### `get_smart_default_model() -> Optional[str]`
Intelligently selects the best available model based on preferences and availability.

- **Returns**: Name of the selected model, or None if no models are available

#### `list_models() -> None`
Displays a table of installed models with their status and performance metrics.

- **Side Effects**: Prints a table of models to the console

#### `update_model(model: str) -> None`
Downloads or updates a model from the Ollama library.

- **Parameters**:
  - `model`: Name of the model to update
- **Side Effects**: Downloads the model and prints progress/status to the console

### Memory Module (`xencode.core.memory`)

Classes for managing conversation memory.

#### `ConversationMemory` Class
Manages conversation history with context preservation.

##### Constructor:
- `ConversationMemory(max_items: int = 50)` - Initializes with a maximum number of items to retain

##### Methods:
- `load_memory()` - Loads conversation history from disk
- `save_memory()` - Saves conversation history to disk
- `start_session(session_id: Optional[str] = None) -> str` - Starts a new conversation session
- `add_message(role: str, content: str, model: Optional[str] = None)` - Adds a message to the current session
- `get_context(max_messages: int = 10) -> List[Dict[str, Any]]` - Gets recent conversation context
- `list_sessions() -> List[str]` - Lists all conversation sessions
- `switch_session(session_id: str) -> bool` - Switches to a different session

### Cache Module (`xencode.core.cache`)

Classes for response caching with multiple levels.

#### `ResponseCache` Class
Sophisticated multi-level response caching system with compression and statistics.

##### Constructor:
- `ResponseCache(cache_dir: Path = CACHE_DIR, max_size: int = 100, ttl_seconds: int = 86400, compression_enabled: bool = True)`

##### Methods:
- `get(prompt: str, model: str) -> Optional[str]` - Gets a cached response
- `set(prompt: str, model: str, response: str)` - Caches a response
- `get_stats() -> Dict[str, Any]` - Gets comprehensive cache statistics
- `clear_cache()` - Clears both memory and disk caches

### Connection Pool Module (`xencode.core.connection_pool`)

Classes for managing HTTP connections efficiently.

#### `APIClient` Class
API client with connection pooling for Ollama and other services.

##### Constructor:
- `APIClient(base_url: str = "http://localhost:11434", max_connections: int = 10, max_retries: int = 3)`

##### Methods:
- `sync_request(endpoint: str, method: str = "POST", json_data: Optional[Dict] = None, timeout: int = 30) -> requests.Response` - Makes a synchronous request
- `async_request(endpoint: str, method: str = "POST", json_data: Optional[Dict] = None, timeout: int = 30) -> aiohttp.ClientResponse` - Makes an asynchronous request
- `generate(model: str, prompt: str, stream: bool = False) -> Union[Dict, str]` - Generates a response from a model
- `list_models() -> Dict` - Lists available models
- `close()` - Closes all connections

#### Helper Functions:
- `get_api_client(base_url: str = "http://localhost:11434") -> APIClient` - Gets or creates a singleton API client
- `close_api_client()` - Closes the global API client

## Security Modules

### Validation Module (`xencode.security.validation`)

Functions for input validation and sanitization.

#### `InputValidator` Class
Validates and sanitizes various types of input.

##### Methods:
- `sanitize_input(input_text: str) -> str` - Removes dangerous patterns from input
- `validate_file_path(file_path: str) -> bool` - Validates file paths to prevent directory traversal
- `validate_url(url: str) -> bool` - Validates URLs to prevent SSRF attacks
- `validate_model_name(model_name: str) -> bool` - Validates model names to prevent injection

#### Standalone Functions:
- `sanitize_user_input(user_input: str) -> str` - Convenience function to sanitize user input
- `validate_file_operation(file_path: str, operation: str = "read") -> bool` - Validates file operations

### API Validation Module (`xencode.security.api_validation`)

Functions for validating API responses.

#### `APIResponseValidator` Class
Validates API responses to ensure they meet expected formats.

##### Methods:
- `validate_ollama_response(response_data: Union[Dict[str, Any], str]) -> bool` - Validates Ollama API responses
- `validate_model_list_response(response_data: Union[Dict[str, Any], str]) -> bool` - Validates model list responses
- `sanitize_response_content(content: str) -> str` - Sanitizes response content

#### Standalone Functions:
- `validate_api_response(response_data: Union[Dict[str, Any], str], api_type: str = "ollama") -> bool` - Validates API responses by type
- `sanitize_api_response(content: str) -> str` - Sanitizes API response content

## Benchmarking Module (`benchmarks.performance_benchmarks`)

Tools for measuring performance of Xencode components.

#### `PerformanceBenchmarkSuite` Class
Suite of performance benchmarks for Xencode components.

##### Methods:
- `benchmark_function(func: Callable, name: str, iterations: int = 10, *args, **kwargs) -> BenchmarkResult` - Benchmarks a synchronous function
- `benchmark_async_function(func, name: str, iterations: int = 10, *args, **kwargs) -> BenchmarkResult` - Benchmarks an asynchronous function
- `benchmark_cache_performance(cache: ResponseCache, iterations: int = 100) -> BenchmarkResult` - Benchmarks cache performance
- `benchmark_api_response_time(api_endpoint: str, headers: Dict[str, str], payload: Dict[str, Any], iterations: int = 10) -> BenchmarkResult` - Benchmarks API response time
- `benchmark_concurrent_api_calls(api_endpoint: str, headers: Dict[str, str], payload: Dict[str, Any], concurrent_requests: int = 10) -> BenchmarkResult` - Benchmarks concurrent API calls
- `print_results()` - Prints benchmark results in a formatted table

#### `run_performance_benchmarks()` Function
Runs a comprehensive set of performance benchmarks.

## Usage Examples

### Basic Usage
```python
from xencode.core import get_api_client, ConversationMemory

# Initialize components
client = get_api_client()
memory = ConversationMemory()

# Add a message to memory
memory.add_message("user", "Hello, how are you?", "llama3.1:8b")

# Get recent context
context = memory.get_context(max_messages=5)
print(f"Context has {len(context)} messages")
```

### File Operations
```python
from xencode.core.files import create_file, read_file

# Create a file
create_file("example.txt", "Hello, world!")

# Read the file
content = read_file("example.txt")
print(content)  # Outputs: Hello, world!
```

### Model Management
```python
from xencode.core.models import ModelManager, list_models

# List available models
list_models()

# Create a model manager
manager = ModelManager()
print(f"Best model: {manager.get_best_model()}")
```

### Caching
```python
from xencode.core.cache import ResponseCache

# Create a cache
cache = ResponseCache()

# Store a response
cache.set("What is the capital of France?", "llama3.1:8b", "The capital of France is Paris.")

# Retrieve the response
response = cache.get("What is the capital of France?", "llama3.1:8b")
print(response)  # Outputs: The capital of France is Paris.

# Get cache statistics
stats = cache.get_stats()
print(f"Cache hit rate: {stats['overall']['hit_rate_percent']}%")
```

## Error Handling

Xencode follows a consistent error handling approach:

1. Functions that return values typically return `None` or appropriate defaults on error
2. Functions with side effects print error messages to the console using Rich panels
3. Validation functions return boolean values indicating success/failure
4. Exceptions are caught internally and converted to appropriate return values or console messages

## Security Considerations

1. All user inputs are sanitized using the `InputValidator` class
2. File operations validate paths to prevent directory traversal
3. URLs are validated to prevent Server-Side Request Forgery (SSRF)
4. Model names are validated to prevent injection attacks
5. API responses are validated to ensure expected formats

## Performance Tips

1. Use the connection pooling system for multiple API calls
2. Leverage the multi-level caching system for repeated requests
3. Monitor cache statistics to optimize cache settings
4. Use the benchmarking tools to identify performance bottlenecks