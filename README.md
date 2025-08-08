# Xencode

A powerful offline-first AI assistant for Arch Linux developers, featuring local LLM integration with Ollama and beautiful Claude-style terminal output.

## ✨ Features

- **🔒 Offline First**: Works completely offline with Qwen 3:4B model
- **🌐 Smart Internet Detection**: Automatically detects connection status
- **🧠 Claude Code Style**: Clean output with separated thinking sections
- **🎨 Rich Formatting**: Syntax highlighting, emojis, and beautiful panels
- **🔄 Model Management**: Easy switching and updating of models
- **⚡ Hyprland Integration**: Super + Enter hotkey ready
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
```
bind = SUPER, Return, exec, /path/to/xencode/xencode.sh
```

## 🚀 Usage

### Basic Examples
```bash
# Simple query
./xencode.sh "Hello, how are you?"

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