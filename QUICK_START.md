# 🚀 Xencode - Quick Start Guide

## Installation

```bash
# Already installed? Skip to Usage!

# If not, clone and setup:
git clone <your-repo>
cd xencode
chmod +x xencode.sh test_fixes.sh
pip install -r requirements.txt
```

## First Run

```bash
./xencode.sh
```

**What happens:**
1. ✅ Checks Ollama
2. ✅ Detects models
3. ✅ Offers to install if none found
4. ✅ Starts immersive chat

## Usage

### Chat Mode (Immersive)
```bash
./xencode.sh
```
Takes over your terminal with full-screen experience!

### Inline Mode (Quick Query)
```bash
./xencode.sh "what is python?"
```
Get instant answer without entering chat mode.

## Commands

### In Chat Mode
```
/help       - Show all commands
/models     - Show available models
/model <name> - Switch model
/project    - Show project context
/status     - System status
/clear      - Clear conversation
exit        - Exit chat
```

## Examples

### Example 1: Basic Chat
```bash
$ ./xencode.sh

You › what is recursion?
Xencode › [streams answer in real-time]

You › exit
```

### Example 2: Switch Models
```bash
$ ./xencode.sh

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
$ ./xencode.sh

You › /project
[Shows project type, git status, files, dependencies]

You › how can I improve this code?
Xencode › [includes project context in response]
```

## Tips

1. **Maximize terminal** for best experience
2. **Install multiple models** for flexibility
3. **Use `/models`** to check health
4. **Work in project directory** for auto-context
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
ollama pull qwen2.5:3b
```

### Slow Responses
```
/model phi3:mini  # Switch to faster model
```

## That's It!

**You're ready to use Xencode!** 🎉

```bash
./xencode.sh
```

**Your immersive AI assistant awaits!** 🤖✨
