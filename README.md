# Xencode

A powerful offline-first AI assistant for Arch Linux developers, featuring local LLM integration with Ollama and beautiful Claude-style terminal output.

## ✨ Features

- **💬 Dual Mode Operation**: Persistent chat mode for extended sessions + inline mode for quick queries
- **🔒 Offline First**: Works completely offline with Qwen 3:4B model
- **🌐 Smart Internet Detection**: Automatically detects connection status
- **🧠 Claude Code Style**: Clean output with separated thinking sections
- **🎨 Rich Formatting**: Syntax highlighting, emojis, and beautiful panels
- **🔄 Model Management**: Easy switching and updating of models
- **⚡ Hyprland Integration**: Super + Enter hotkey with floating terminal support
- **🛠️ Smart Installation**: Automated setup with fallback mechanisms
- **🔍 Service Detection**: Automatically detects and manages Ollama service

## 📦 Installation

### 🚀 Quick Install (Recommended)
```bash
git clone https://github.com/sreevarshan-xenoz/xencode.git
cd xencode
./install.sh
```

The install script will:
- ✅ Install Python dependencies via pacman (Arch Linux) or pip
- ✅ Install and configure Ollama service  
- ✅ Detect if Ollama is already running
- ✅ Pull the Qwen 3:4B model
- ✅ Verify everything is working

### 🔧 Manual Install
If the automated installer fails, see [INSTALL_MANUAL.md](INSTALL_MANUAL.md) for detailed troubleshooting steps.

### 🧪 Test Installation
```bash
./test.sh
```

### 🎮 Hyprland Integration
Add to `~/.config/hypr/hyprland.conf`:
```bash
# Hotkey for persistent chat mode
bind = SUPER, Return, exec, /path/to/xencode/xencode.sh

# Window rule for floating terminal
windowrulev2 = float, title:Xencode AI
windowrulev2 = center, title:Xencode AI
windowrulev2 = size 1200 800, title:Xencode AI
```

**Note**: The persistent chat mode requires Kitty terminal. If Kitty is not available, xencode will fall back to your default terminal or inline mode.

## 🚀 Usage

### 💬 Persistent Chat Mode (NEW)
Launch an interactive chat session in a floating terminal window:
```bash
# Start persistent chat mode
./xencode.sh

# This opens a Kitty terminal with title "Xencode AI"
# Perfect for extended conversations and iterative development
```

The chat interface provides:
- Interactive `[You] >` prompts
- Continuous conversation without re-launching
- Dynamic connectivity status updates
- Multiple exit options: `exit`, `quit`, `q`, Ctrl+C, or Ctrl+D

### 📝 Inline Mode
For quick one-off queries in your current terminal:
```bash
# Simple query (existing behavior)
./xencode.sh "Hello, how are you?"

# Explicit inline mode (NEW)
./xencode.sh --inline "write a simple python function to calculate fibonacci"

# Code generation
./xencode.sh "write a simple python function to calculate fibonacci"

# Technical explanation
./xencode.sh "explain why arch users are cooler"
```

### Advanced Usage
```bash
# Use specific model
./xencode.sh -m mistral "make a todo app in python"

# List available models
./xencode.sh --list-models

# Update model (requires internet)
./xencode.sh --update

# Update specific model
./xencode.sh --update -m qwen3:4b
```

### Sample Output
```
🧠 Thinking...
The user is asking for a simple Python function to calculate Fibonacci numbers.
I should provide an efficient iterative solution with clear explanation.

📄 Answer
Here's a simple and efficient Python function to calculate the n-th Fibonacci number:

def fib(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a
```

## 🏗️ Architecture

| File | Purpose |
|------|---------|
| `xencode.sh` | Bash launcher with internet detection |
| `xencode_core.py` | Python core with Rich formatting and Ollama integration |
| `install.sh` | Automated installation script with smart service detection |
| `test.sh` | Installation verification and testing |
| `INSTALL_MANUAL.md` | Detailed troubleshooting guide |

## 🔧 Requirements

- **Python 3.6+** with `requests` and `rich` libraries
- **Ollama** for local LLM inference
- **Kitty terminal** (recommended for persistent chat mode, falls back to default terminal)
- **Internet connection** (only for model updates)
- **Arch Linux** (recommended, but works on other Linux distros)
- **curl** (for service health checks)

## 🎯 Key Features Explained

### 🧠 Thinking Sections
Xencode parses and displays the model's reasoning process separately from the final answer, mimicking Claude's thinking style.

### 🔄 Smart Service Management  
- Automatically detects if Ollama is already running
- Falls back to manual `ollama serve` if systemd fails
- Provides clear error messages and recovery suggestions

### 🎨 Rich Terminal Output
- Syntax-highlighted code blocks
- Emoji indicators for different sections
- Clean panels and formatting via Rich library

### 🌐 Offline-First Design
- Works completely offline once models are downloaded
- Automatically detects internet connectivity
- Only requires internet for model updates

## 🐛 Troubleshooting

If you encounter issues:

1. **Run the test script**: `./test.sh`
2. **Check the manual guide**: [INSTALL_MANUAL.md](INSTALL_MANUAL.md)
3. **Verify Ollama is running**: `curl http://localhost:11434/api/tags`
4. **Check Python dependencies**: `python3 -c "import requests, rich"`

## 🤝 Contributing

Feel free to submit issues and pull requests to improve Xencode!