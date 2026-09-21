# Manual Installation Guide

Xencode is **Rust-only**: one Cargo workspace under `rust/crates/*` producing a
single `xencode` binary. There is no Python stack to install.

---

## Install the Rust Binary

Quickest path: `./install.sh` (Linux/macOS) or `.\install.ps1` — it checks the
toolchain, builds the release binary, sets Ollama up, pulls `qwen3:4b`,
smoke-tests and installs `xencode` into `~/.local/bin`. To do it by hand:

### Prerequisites
- **Rust** (stable) toolchain — install from https://rustup.rs
- **Ollama** for local AI models (install from https://ollama.ai)

### Build from Source

```bash
# Clone repository
git clone https://github.com/sreevarshan-xenoz/xencode
cd xencode

# Build the release binary (package xencode-cli → binary named `xencode`)
cd rust
cargo build --release -p xencode-cli

# Copy to PATH
# Linux/macOS:
cp target/release/xencode ~/.local/bin/        # or /usr/local/bin, needs sudo
# Windows (PowerShell):
copy target\release\xencode.exe $env:LOCALAPPDATA\Microsoft\WindowsApps\
```

### Verify Installation

```bash
xencode --help
xencode scan . --max-depth 1
```

---

## Ollama & Model Setup

### 1. Install Ollama

```bash
# On Arch Linux
sudo pacman -S ollama

# Or download from https://ollama.ai/download
```

### 2. Start Ollama Service

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

### 3. Pull the Model

```bash
# Wait for Ollama to start, then pull model
ollama pull qwen3:4b
```

---

## Troubleshooting

### Ollama not responding
- Check if service is running: `systemctl status ollama`
- Try manual start: `ollama serve` in a separate terminal
- Check port 11434 is not blocked: `curl http://localhost:11434/api/tags`

### Permission errors
- Check file permissions: `ls -la`

### Rust build errors
- Ensure Rust toolchain is up to date: `rustup update`
- Check Cargo workspace: `cd rust && cargo build -p xencode-cli 2>&1`

---

> 📄 **Reference:** [README.md](../README.md) for the crate layout and current test
> counts, and its agent-loop diagram for how a turn flows;
> [CLI_GUIDE.md](../CLI_GUIDE.md) for every command and config key.