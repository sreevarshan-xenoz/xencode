# Manual Installation Guide

Xencode has a **dual-stack architecture** — a Rust binary (primary) and a Python stack (legacy/plugins).

---

## Option A: Install Rust Binary (Recommended)

The Rust binary is a single-file executable with no Python dependency. It provides the TUI, CLI, server, analysis, and plugin management.

### Prerequisites
- **Rust 1.75+** toolchain (install from https://rustup.rs)
- **Ollama** for local AI models (install from https://ollama.ai)

### Build from Source

```bash
# Clone repository
git clone https://github.com/sreevarshan-xenoz/xencode
cd xencode

# Build release binary (output: rust/target/release/xencode-cli.exe)
cd rust
cargo build --release -p xencode-cli

# Copy to PATH
# Linux/macOS:
cp target/release/xencode-cli /usr/local/bin/xencode
# Windows:
copy target\release\xencode-cli.exe C:\Windows\System32\xencode.exe
```

### Verify Installation

```bash
xencode --help
xencode scan . --max-depth 1
```

---

## Option B: Install Python Stack (Legacy / Plugin Development)

### 1. Install Python Dependencies

```bash
# On Arch Linux (Recommended)
sudo pacman -S python-requests python-rich

# On other systems or with pip
pip3 install --user requests rich
```

### 2. Install Ollama

```bash
# On Arch Linux
sudo pacman -S ollama

# Or download from https://ollama.ai/download
```

### 3. Start Ollama Service

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

### 4. Pull the Model

```bash
# Wait for Ollama to start, then pull model
ollama pull qwen3:4b
```

### 5. Test Installation

```bash
./test.sh
```

---

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

### Rust build errors
- Ensure Rust toolchain is up to date: `rustup update`
- Check Cargo workspace: `cd rust && cargo build -p xencode-cli 2>&1`
- See `docs/RUST_MIGRATION_STATUS.md` for crate details

---

> 📄 **Reference:** [`xencode-codebase-reference.html`](../xencode-codebase-reference.html) — Complete codebase reference with all 12 crate descriptions, test counts, architecture diagrams, and remaining work items.