# Xencode

A powerful offline-first AI assistant for Arch Linux developers, featuring local LLM integration with Ollama and beautiful terminal output.

## Features

- **Offline First**: Works completely offline with Qwen 4B model
- **Internet Detection**: Automatically detects connection status
- **Claude Code Style**: Clean, structured output with thinking sections
- **Code Highlighting**: Syntax highlighting for code blocks with Rich library
- **Model Management**: Easy switching and updating of models
- **Hyprland Integration**: Super + Enter hotkey ready
- **Rich Formatting**: Beautiful terminal output with emojis and panels

## Installation

1. **Clone and setup**:
```bash
git clone <repo-url>
cd xencode
chmod +x xencode.sh xencode_core.py
```

2. **Install dependencies**:
```bash
pip install requests rich
sudo pacman -S ollama
```

3. **Pull the default model**:
```bash
ollama pull qwen:4b
```

4. **Setup Hyprland keybinding** (optional):
Add to `~/.config/hypr/hyprland.conf`:
```
bind = SUPER, Return, exec, /path/to/xencode/xencode.sh
```

## Usage

```bash
# Basic query
./xencode.sh "explain why arch users are cooler"

# With specific model
./xencode.sh -m mistral "make a todo app in python"

# List available models
./xencode.sh --list-models

# Update model (requires internet)
./xencode.sh --update -m qwen:4b
```

## Architecture

- `xencode.sh`: Bash launcher that detects internet connectivity
- `xencode_core.py`: Python core with Rich formatting and Ollama integration

## Requirements

- Python 3.6+
- Ollama
- Internet connection (for model updates only)
- Arch Linux (recommended)