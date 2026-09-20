# Xencode User Manual

## Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Getting Started](#getting-started)
4. [Rust CLI](#rust-cli)
5. [Rust TUI](#rust-tui)
6. [Command Reference](#command-reference)
7. [Troubleshooting](#troubleshooting)
8. [Examples](#examples)

## Introduction

Xencode is an AI-powered development assistant platform that integrates with local language models through Ollama. It provides intelligent code analysis, document processing, workspace collaboration, and plugin management with a focus on privacy and offline operation.

### Architecture
Xencode is a **Rust-only** workspace (`rust/crates/*`, 13 crates):
- **Rust core** — Primary CLI/TUI, server, code analysis, security scanning, plugin system, multi-provider routing (Ollama, Anthropic, Gemini, Qwen, OpenRouter) with retry middleware, and collaboration sync

The Rust binary (`xencode`) is the entry point.

### Key Features
- **Rust TUI**: Ratatui-based terminal interface with 23 interactive feature panels and overlays
- **Multi-Provider AI Routing**: Ollama local + Anthropic, Gemini, Qwen, OpenRouter cloud with token-aware exponential retry middleware
- **Code Analysis**: Language-aware AST analysis (Python, JS/TS, Rust) + OWASP vulnerability scanning
- **HTTP/WebSocket Server**: Axum-based collaboration server with session management
- **Plugin System**: Plugin trait, host, registry with lifecycle management
- **Conversation Memory & Cache**: Persistent session history with compressed hybrid caching
- **Secure Authentication**: Encrypted credential vault, SQLite store, refresh token rotation, email verification

## Installation

### Rust Binary

```bash
# Build from source (requires Rust 1.75+)
cd rust && cargo build --release -p xencode-cli
./target/release/xencode --help
```

### Prerequisites
- **Rust 1.75+** (for building from source)
- **Ollama** installed and running (`ollama serve`)
- **A model** installed: `ollama pull qwen3:4b`
- **4GB+ RAM** recommended

## Getting Started

### Rust CLI

The Rust binary is the primary entry point:

```bash
# Verify installation
xencode --help

# Launch TUI (default experience)
xencode tui

# Scan workspace
xencode scan . --max-depth 2

# List models
xencode models list

# Analyze code for issues and vulnerabilities
xencode analyze src/
xencode analyze src/main.rs --format json

# Start the collaboration server
xencode server --port 8765

# Manage plugins
xencode plugin list
xencode plugin install ./my-plugin/
xencode plugin remove my-plugin

# Run a quick query
xencode query "Explain clean architecture"

# View conversation memory
xencode memory list

# Manage background tasks
xencode tasks start "cargo test" --name tests
xencode tasks list
xencode tasks poll 1 --lines 20

# Check cache stats
xencode cache stats

# Show config
xencode config show
```

### Rust TUI

Launch the interactive TUI with `xencode tui`:

```
┌─────────────────┬──────────────────────────────────────┬──────────────────┐
│  File Explorer   │          Code Editor                  │    Chat Panel    │
│    (Tab)        │                                       │    (Tab)        │
│                 │                                       │                  │
│ src/            │   // Edit your code here             │  You > Hello!    │
│   main.rs      │                                        │                  │
│   lib.rs       │                                        │  AI > Hi there!  │
│ tests/          │                                       │                  │
└─────────────────┴──────────────────────────────────────┴──────────────────┘

```

**TUI keyboard shortcuts:**

Global (work in every panel, any mode):

| Key | Action |
|-----|--------|
| `Ctrl+C` | Quit |
| `Ctrl+G` | Refresh git status |
| `Ctrl+E` | File explorer |
| `Ctrl+S` | Save editor / Git commit panel |
| `Ctrl+R` | AI code review (current file) |
| `Ctrl+Y` | PR review dashboard |
| `Ctrl+K` | Background tasks panel |
| `Ctrl+O` | Worktree panel (list/add/remove git worktrees) |
| `Ctrl+F` | Feature Navigator (16 panels) |
| `Ctrl+D` | Performance dashboard |
| `Ctrl+P` | Project analyzer |
| `Ctrl+B` | ByteBot agent panel |
| `Ctrl+H` | Run provider health check |
| `Ctrl+T` | Toggle embedded terminal strip |
| `Ctrl+W` | Close panel → chat |
| `Ctrl+,` | Settings |

Panel-independent (Normal mode):

| Key | Action |
|-----|--------|
| `?` / `F1` | Keybinding help overlay (lists the focused panel's keys) |
| `Tab` | Cycle explorer → editor → chat |
| `Esc` | Close popup / leave edit mode |
| `i` or `/` | Edit the chat input |
| `m` | Model selector (`r` refresh, `l`/`u` llama.cpp load/unload) |
| `s` | Settings (except in panels that bind `s` themselves: Security Auditor sort, Custom Models save) |
| `q` | Quit |
| `e` | Code editor: start typing edits |
| `↑ ↓` / `j k` | Scroll or move selection (all scrollable panels) |
| Mouse wheel | Scroll the panel under the cursor |

Chat editing (after `i`):

| Key | Action |
|-----|--------|
| `Enter` | Send message |
| `Alt+Enter` / `Ctrl+J` | Insert newline (multiline prompts) |
| `Alt+↑` / `Alt+↓` | Recall previous / next sent prompt |
| `Tab` | Complete a `/` command, else insert 4 spaces |

**Feature Navigator panels (Ctrl+F → select → Enter):**

| Panel | Description |
|-------|-------------|
| Performance Dashboard | Session stats, file breakdown |
| Provider Health | Health checks with status icons |
| Project Analyzer | Workspace file type analysis |
| Git Commit | Commit message input with cursor |
| ByteBot Agent | Step-through autonomous task execution |
| Collaboration Hub | Session sharing, member status, sync |
| Voice Interface | Audio level meter, commands, transcript |
| Terminal Assistant | Shell command suggestions, risk badges |
| Security Auditor | Vulnerability findings, severity bars |
| Performance Profiler | Gauges, function timing, hot paths |
| Custom Models | Profile list, parameter sliders |
| Learning Mode | Lesson viewer, code examples |
| Multi-Language | Language detection, translation |
| PR Review | Per-file diff browsing, base toggle |
| Background Tasks | Live registry of background commands (stop/remove) |
| Worktrees | Git worktree list, add (path+branch) and remove with confirm |

### First-Time Setup

On first run, Xencode will guide you through:
1. Verifying Ollama installation
2. Checking available models
3. Installing a recommended model if none found
4. Configuring default settings

## Advanced Features

### Code Analysis

Analyze source code for style issues, bugs, and security vulnerabilities:

```bash
# Analyze a directory (recursive)
xencode analyze src/

# Analyze a single file with JSON output
xencode analyze src/main.rs --format json

# Results show: issue type, severity (Low/Medium/High/Critical),
# file location, description, and suggestion
```

### Security Scanning

The Rust analyzer includes OWASP-focused vulnerability scanning:
- Hardcoded secrets (passwords, API keys, tokens)
- SQL injection patterns
- Command injection risks
- Weak cryptography (MD5, SHA1, weak RNG)
- Path traversal vulnerabilities
- SSRF patterns

### Collaboration Server

Start a real-time collaboration session:

```bash
# Start server on default port
xencode server

# Start on a specific port
xencode server --port 8765

# Server provides:
# - WebSocket peer broadcast
# - Session management
# - Health and status endpoints
# - Model listing
```

### Plugin System

Extend functionality with plugins:

```bash
# List installed plugins
xencode plugin list

# Install a plugin from a directory
xencode plugin install ./my-plugin/

# Remove a plugin
xencode plugin remove my-plugin
```

Plugins implement the `XencodePlugin` trait with lifecycle methods:
- `initialize` — Called when plugin is loaded
- `handle_event` — Process an event and return a response
- `shutdown` — Clean up resources

### Conversation Memory

```bash
# List all conversation sessions
xencode memory list

# Show specific session
xencode memory show <session-id>
```

### Cache Management

```bash
# Show cache stats (hits, misses, evictions)
xencode cache stats

# Clear all cached responses
xencode cache clear
```

### Background Tasks

```bash
# Start a task that survives this process (state in .xencode/tasks/)
xencode tasks start "cargo test" --name tests

# List tasks with derived status (running / exited(code) / killed)
xencode tasks list
xencode tasks list --json

# Show a task's status and trailing output
xencode tasks poll 1 --lines 20

# Stop a running task; remove a finished one (refused while running)
xencode tasks stop 1
xencode tasks rm 1
```

## Command Reference

### Rust CLI Commands

```
Usage: xencode [COMMAND]

Commands:
  scan      Scan a workspace and list all entries
  config    Configuration management
  models    Local model management (Ollama & llama.cpp)
  cache     Response cache management
  query     Send a query to a model
  memory    Manage conversation memory
  tasks     Manage background tasks (file-backed, survives this process)
  server    Start the collaboration server
  analyze   Analyze code for issues and vulnerabilities
  fetch     Fetch a web page and extract research-ready text
  review    Review the diff between a base branch and HEAD, file by file
  plugin    Manage plugins
  llamacpp  llama.cpp server management (status/start/stop/load/unload)
  tui       Launch the Terminal User Interface
  help      Print this message or the help of the given subcommand(s)

Options:
  -h, --help     Print help
  -V, --version  Print version
```

## Examples

### Example 1: Code Analysis
```bash
# Analyze a Python project for issues
$ xencode analyze src/
📊 Analysis Report for src/
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📄 main.py
  ⚠️ [MEDIUM] Bare except clause on line 23
     Suggestion: Use 'except Exception:' to catch only expected exceptions
  ⚠️ [LOW] Line too long (>100 chars) on line 15
     Suggestion: Break line into multiple lines

📄 utils.py
  🔴 [CRITICAL] Hardcoded API key on line 5
     Suggestion: Remove the hardcoded value and use environment variables or a vault service
  🔴 [HIGH] Mutable default argument on line 8
     Suggestion: Use None as default and initialize inside the function

📊 Summary: 4 issues found (1 critical, 1 high, 1 medium, 1 low)
```

### Example 2: Collaboration Server
```bash
# Start the server
$ xencode server --port 8765
🚀 Xencode server starting on http://0.0.0.0:8765

# In another terminal, check health
$ curl http://localhost:8765/
{"status":"online","service":"Xencode Server","version":"0.1.0"}

# List available models via API
$ curl http://localhost:8765/api/models
{"models":{"qwen3:4b":{"name":"qwen3:4b","status":"healthy","response_time":"1.2s"}}}
```

### Example 3: Running the TUI
```bash
$ xencode tui

# The TUI opens with 3 panels: File Explorer | Code Editor | Chat
# Press Ctrl+F to open the Feature Navigator
# Select any panel with arrow keys + Enter
# Press Esc to close an overlay
# Press Tab to cycle focus between main panels
```

### Example 4: Plugin Management
```bash
$ xencode plugin list
🔌 Installed Plugins
  No plugins installed.

$ xencode plugin install ./my-custom-plugin/
✅ Plugin 'my-custom-plugin' installed successfully

$ xencode plugin list
🔌 Installed Plugins
  my-custom-plugin v1.0.0 — Custom analysis plugin
```

### Example 5: Short Query
```bash
$ xencode query "What does this Rust code do?"
[Query runs against configured model...]
```

## Troubleshooting

### Common Issues

#### Ollama Not Running
**Problem:** "Cannot connect to Ollama service"
**Solution:** 
1. Start Ollama: `ollama serve`
2. Verify it's running: `curl http://localhost:11434/api/tags`

#### No Models Available
**Problem:** "No models found"
**Solution:**
1. Check available models: `ollama list`
2. Install a model: `ollama pull qwen3:4b`

#### Slow Responses
**Problem:** Long response times
**Solution:**
1. Check model health: `xencode models health`
2. List installed models: `xencode models list`
3. Check system resources: `htop` or Task Manager

#### File Operation Errors
**Problem:** "Permission denied" or "File not found"
**Solution:**
1. Check file permissions
2. Verify file paths are correct
3. Ensure you have read/write permissions for the directory

### Performance Tips

1. **Use Caching**: Xencode caches responses to speed up repeated queries
2. **Choose Efficient Models**: Smaller models often respond faster
3. **Manage Memory**: Clear old sessions if experiencing slowdowns
4. **Optimize Prompts**: Clear, specific prompts yield faster responses

### Getting Help

- Use `/help` for command reference
- Check system status with `/status`
- Report issues on GitHub: https://github.com/sreevarshan-xenoz/xencode/issues
- Join discussions: https://github.com/sreevarshan-xenoz/xencode/discussions

## Best Practices

### Effective Prompting
- Be specific about what you need
- Provide context when relevant
- Break complex tasks into smaller queries
- Ask for explanations of code you don't understand

### Security
- Don't share sensitive information in prompts
- Verify code suggestions before running
- Keep your system updated
- Review file operations before confirming

### Productivity
- Use conversation memory to maintain context
- Leverage file operations for code generation
- Switch models based on task requirements
- Export important conversations for reference

---

For more information, visit the official documentation at https://github.com/sreevarshan-xenoz/xencode