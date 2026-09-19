# 🤖 Xencode CLI Guide

The command-line interface for the Xencode AI assistant (Rust binary).
Running `xencode` with no subcommand launches the TUI.

## 🚀 Installation

```bash
# From source
git clone https://github.com/sreevarshan-xenoz/xencode
cd xencode
./install.sh        # Linux/macOS — builds + installs the binary

# Or build directly
cd rust && cargo build --release -p xencode-cli
cp target/release/xencode ~/.local/bin/
```

## 🎯 Quick Start

```bash
# Launch the TUI (default)
xencode

# One-shot query
xencode query "Explain clean code principles"

# Analyze a path
xencode analyze ./src

# Show version
xencode --version
```

## 📋 Command Reference

### `xencode` / `xencode tui`
Launch the immersive terminal UI (default when no subcommand is given).

### `xencode query <prompt>`
Send a one-shot query to the configured model.

```bash
xencode query "Explain microservices architecture"

# Pin a model, skip the cache, attach a session
xencode query "Explain async programming" \
  --model qwen3:4b \
  --no-cache \
  --session-id demo

# llama.cpp sampling controls
xencode query "Write a haiku" \
  --temperature 0.7 \
  --top-k 40 \
  --min-p 0.05 \
  --max-tokens 256
```

### `xencode analyze <path> [--format text|json]`
Analyze a file or directory for code issues and vulnerabilities. Image
files take the intake path: format, dimensions, and byte size are reported
(`--format json` returns the `ImageMeta` for a single image).

```bash
xencode analyze ./src
xencode analyze ./assets/logo.png --format json
```

### `xencode scan [path] [--hidden] [--max-depth N]`
List workspace entries (kind, size, path).

```bash
xencode scan . --max-depth 2
```

### `xencode models <action>`
Local model management (Ollama & llama.cpp).

```bash
xencode models list       # All installed Ollama models
xencode models health <name>  # Check one model
xencode models default     # Show the smart-selected default
```

### `xencode llamacpp <action>`
llama.cpp server management: `status`, `start`, `stop`, `load`, `unload`.

```bash
xencode llamacpp status
xencode llamacpp start --model mymodel.gguf --port 8080
```

### `xencode config <action>`
Configuration management.

```bash
xencode config show
xencode config set default_model qwen3:4b
xencode config reset
```

### `xencode cache <action>`
Response cache management (`stats`, `clear`, …).

```bash
xencode cache stats
```

### `xencode memory <action>`
Conversation memory management (`list`, …).

```bash
xencode memory list
```

### `xencode server [--port 8765]`
Start the collaboration HTTP/WebSocket server.

```bash
xencode server --port 8765
```

### `xencode plugin <action>`
Plugin management: `list`, `install <path>`, `remove <name>`.

```bash
xencode plugin list
```

## 🎯 Usage Examples

### Development Workflow
```bash
# 1. Index the project (inside the TUI)
xencode
# › /init

# 2. Check model health
xencode models list

# 3. Query for code help
xencode query "How do I parse JSON in Rust?"

# 4. Analyze before committing
xencode analyze ./src --format text
```

### Scripting
```bash
#!/bin/bash
# Ask and fail loudly on error
if ! xencode query "$1" --no-cache; then
    echo "❌ Query failed" >&2
    exit 1
fi
```

### JSON output for tooling
```bash
xencode analyze ./assets/logo.png --format json | jq .format
```
