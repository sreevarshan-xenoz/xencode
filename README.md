# Xencode - AI-Powered Development Platform

Xencode is a cutting-edge AI assistant platform that transforms how developers interact with their command-line environment. It combines intelligent model selection, advanced caching, robust error handling, and innovative workflow capabilities to create a superior development experience.

## Core Features

### 1. Intelligent Model Selection
- Automatically detects system hardware (CPU, GPU, RAM, storage)
- Recommends optimal AI models based on available resources
- Interactive setup wizard for first-time users
- Performance optimization for different system configurations

### 2. Advanced Caching System
- Hybrid memory/disk caching with LRU eviction
- LZMA compression for efficient storage
- Cache analytics and monitoring
- Multi-level caching strategy

### 3. Robust Error Handling
- Intelligent error classification and recovery
- Automatic retry mechanisms with exponential backoff
- Context-aware error messages
- Comprehensive error monitoring

### 4. Performance Optimizations
- Parallel model availability checking
- Efficient consensus calculation algorithms
- Streamlined confidence scoring in single pass
- Optimized parallel inference with named tasks
- Enhanced caching with hybrid memory/disk system
- LZMA compression for efficient storage

### 5. Smart Configuration Management
- Multi-format support (YAML, TOML, JSON, INI)
- Schema validation with Pydantic
- Interactive configuration wizard
- Hot-reload configuration changes

### 6. Plugin Architecture
- Dynamic plugin loading and lifecycle management
- Service registration and discovery
- Event-driven plugin communication
- Secure plugin context with permissions

### 7. Advanced Analytics Dashboard
- Real-time performance metrics monitoring
- SQLite-based metrics persistence
- Usage pattern analysis
- Cost tracking and optimization recommendations

### 8. Hybrid Model Architecture
- Dynamic switching between local and cloud models
- Model chaining for complex workflows
- Privacy-aware routing based on sensitivity levels
- Fallback mechanisms for high availability

### 9. Advanced Memory Management
- Tiered storage system (RAM Hot/Warm, SSD Cold, HDD Archive)
- Predictive caching based on usage patterns
- Intelligent cache eviction policies
- Cross-tier balancing algorithms

### 10. Visual Workflow Builder
- Drag-and-drop interface for creating AI workflows
- Multiple node types (input, process, decision, model call)
- Template library for common workflow patterns
- Interactive execution and visualization

### 11. Enhanced Xencode Terminal
- Structured command blocks with rich output rendering
- AI-powered command suggestions
- Session persistence with crash recovery
- Advanced UI components and command palette

### 12. Advanced Multi-Agent Collaboration
- Market-based resource allocation system
- Negotiation protocols between agents
- Swarm intelligence behaviors for coordination
- Human-in-the-loop supervision capabilities
- Cross-domain expertise combination system
- Advanced coordination strategies (hierarchical, market-based, swarm intelligence)

### 13. Performance Optimizations
- Parallel model availability checking
- Efficient consensus calculation algorithms
- Streamlined confidence scoring in single pass
- Optimized parallel inference with named tasks
- Enhanced caching with hybrid memory/disk system
- LZMA compression for efficient storage

## Installation

```bash
pip install xencode
```

### Install via npm

Install the Node wrapper package directly from GitHub:

```bash
npm install -g github:sreevarshan-xenoz/xencode
```

Requirements for npm install path:
- Node.js 18+
- Python 3.8+ available in `PATH`

The npm wrapper runs the Python CLI (`python -m xencode.cli`) under the hood.

If you prefer Python-native installation:

```bash
pip install xencode
```

## Usage

```bash
# Run Xencode (opens TUI by default)
xencode

# Start an interactive agent session
xencode agentic --model qwen3:4b

# Launch the TUI
xencode tui

# Launch the TUI (module entrypoint)
python -m xencode.tui

# Or use as a module
python -m xencode
```

### First-run TUI onboarding

- On first launch, Xencode shows onboarding in TUI with login/signup options.
- Open the settings widget anytime with `Ctrl+,`.
- Open the options panel (all major CLI commands) with `Ctrl+O`.
- Choose from 10 themes in Settings and apply instantly.
- Use explicit CLI subcommands (for example `xencode version`) to run non-TUI flows.

## Architecture

Xencode follows a modular architecture with clear separation of concerns:

- **Core**: Fundamental components and utilities
- **Models**: AI model management and selection
- **Cache**: Advanced caching mechanisms
- **Analytics**: Performance monitoring and analytics
- **Plugins**: Extension system
- **Security**: Security and validation utilities
- **TUI**: Terminal user interface components
- **Workflows**: Workflow management and execution

## Contributing

We welcome contributions to Xencode! Please see our contributing guidelines for more information.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

Xencode builds upon the excellent work of the open-source community and integrates with various AI model providers to deliver the best possible experience for developers.