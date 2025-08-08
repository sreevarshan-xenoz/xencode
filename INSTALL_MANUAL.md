# Manual Installation Guide

If the automatic installation script fails, follow these steps:

## 1. Install Python Dependencies

### On Arch Linux (Recommended)
```bash
sudo pacman -S python-requests python-rich
```

### On other systems or with pip
```bash
# Option 1: Using pip3 (user install)
pip3 install --user requests rich

# Option 2: Using python3 -m pip
python3 -m pip install --user requests rich

# Option 3: Install pip first if missing
sudo pacman -S python-pip
pip3 install --user requests rich
```

## 2. Install Ollama

```bash
# On Arch Linux
sudo pacman -S ollama

# Or download from https://ollama.ai/download
```

## 3. Start Ollama Service

```bash
# Check if already running
curl -s http://localhost:11434/api/tags

# If not running, try systemd service
sudo systemctl enable ollama
sudo systemctl start ollama

# OR run manually in a separate terminal
ollama serve

# OR run in background
nohup ollama serve > /dev/null 2>&1 &
```

## 4. Pull the Model

```bash
# Wait for Ollama to start, then pull model
ollama pull qwen3:4b
```

## 5. Test Installation

```bash
./test.sh
```

## Troubleshooting

### Ollama not responding
- Check if service is running: `systemctl status ollama`
- Try manual start: `ollama serve` in a separate terminal
- Check port 11434 is not blocked: `curl http://localhost:11434/api/tags`

### Python import errors
- Verify Python 3 is installed: `python3 --version`
- Check if packages are installed: `python3 -c "import requests, rich"`
- Try installing in user space: `pip3 install --user requests rich`

### Permission errors
- Make scripts executable: `chmod +x xencode.sh xencode_core.py`
- Check file permissions: `ls -la`