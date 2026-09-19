# 🚀 Xencode - Quick Start Guide

## Installation

```bash
# Already installed? Skip to Usage!

# If not, clone and run the installer:
git clone <your-repo>
cd xencode
./install.sh        # Linux/macOS — builds the Rust binary, checks Ollama
```

Windows (PowerShell): `.\install.ps1`

**What happens:**
1. ✅ Checks Rust toolchain, curl, git
2. ✅ Builds the release binary (`cargo build --release -p xencode-cli`)
3. ✅ Checks Ollama (installs + starts it if missing)
4. ✅ Pulls the starter model (`qwen3:4b`)
5. ✅ Smoke-tests the binary and installs the `xencode` command

## First Run

```bash
xencode
```

Launches the immersive TUI. Run `/init` once per project for project-aware
answers, then just ask.

## Usage

### TUI (default experience)
```bash
xencode
```
Full-screen terminal UI: chat, file explorer (Space attaches files —
including images, which the model actually sees), `/ctx` retrieval,
`/advise` repo insights, model picker, and more.

### One-shot query
```bash
xencode query "what is recursion?"
```
Get an instant answer without entering the TUI.

### Analyze a path
```bash
xencode analyze ./src
```
Code issues + security findings, plus an inventory of any images found.

## Commands

### In the TUI
```
/help       - Show all commands
/models     - Show available models
/model <name> - Switch model
/init       - Index the current project
/ctx        - Retrieve project context
/advise     - Repository insights (cycles, hubs, orphans, broken imports)
/clear      - Clear conversation
```

## Examples

### Example 1: Basic Chat
```bash
$ xencode

You › what is recursion?
Xencode › [streams answer in real-time]

You › /exit
```

### Example 2: Switch Models
```bash
$ xencode

You › /models
[Shows all models with health status]

You › /model qwen2.5:7b
✅ Model switched!

You › explain async/await
Xencode › [uses new model]
```

### Example 3: Project Context
```bash
$ cd /path/to/your/project
$ xencode

You › /init
[Indexes the project]

You › how can I improve this code?
Xencode › [includes project context in response]
```

## Tips

1. **Maximize terminal** for best experience
2. **Install multiple models** for flexibility
3. **Use `/models`** to check health
4. **Run `/init`** in each project directory for auto-context
5. **Type `/help`** to see all commands

## Troubleshooting

### Ollama Not Running
```bash
ollama serve
# or
systemctl start ollama
```

### No Models
```bash
ollama pull qwen3:4b
```

### Slow Responses
```
/model phi3:mini  # Switch to faster model
```

## That's It!

**You're ready to use Xencode!** 🎉

```bash
xencode
```

**Your immersive AI assistant awaits!** 🤖✨
