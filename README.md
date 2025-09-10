# Xencode

A professional offline-first AI assistant with Claude-style interface, featuring local LLM integration with Ollama and elegant streaming terminal output.

## Features

- **Claude-Style Interface**: Real-time streaming with thinking sections and formatted responses
- **Offline-First**: Complete local operation with optional internet connectivity
- **Conversation Memory**: Persistent sessions with context awareness
- **Multi-Model Support**: Compatible with all Ollama models
- **Professional UI**: Rich terminal interface with dynamic status updates
- **Cross-Platform**: Support for major Linux distributions

## Installation

### Quick Start
```bash
git clone https://github.com/sreevarshan-xenoz/xencode.git
cd xencode
./install.sh
```

### Manual Installation
See `docs/INSTALL_MANUAL.md` for detailed instructions.

## Usage

### Interactive Chat Mode
```bash
./xencode.sh
```

### Inline Queries
```bash
./xencode.sh "explain quantum computing briefly"
```

### Model Management
```bash
./xencode.sh --list-models
./xencode.sh --update
```

## Requirements

- **System**: Linux (Arch, Debian/Ubuntu, Fedora, RHEL/CentOS)
- **Python**: 3.6+
- **Dependencies**: `requests`, `rich`, `prompt_toolkit`
- **AI Engine**: Ollama with local models

## Project Structure

```
xencode/
├── xencode.sh              # Main executable
├── xencode_core.py         # Core logic
├── install.sh              # Installation script
├── requirements.txt        # Python dependencies
├── docs/                   # Documentation
├── scripts/                # Utility scripts
└── tests/                  # Test suite
```

## Testing

```bash
# Run all tests
./scripts/test.sh

# Test Claude-style features
./scripts/test_claude_style.sh

# Enhanced features test
./scripts/test_enhanced_features.sh
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Run tests with `./scripts/test.sh`
4. Submit a pull request

## License

MIT License - see LICENSE file for details.