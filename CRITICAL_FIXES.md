# Critical Fixes for Xencode

## ✅ Good News: No Syntax Errors!

After comprehensive analysis, **all Python files are syntactically correct** with no compilation errors. The "incomplete" code I initially saw was due to file reading truncation, not actual bugs.

## 🐛 Real Issues Found

### 1. **Missing Startup Health Check** ⚠️ HIGH PRIORITY
**File**: `xencode_core.py`
**Issue**: No check if Ollama is running before starting chat mode

**Current Behavior**:
```
User starts xencode → Chat mode starts → User types message → ERROR: Connection refused
```

**Fix**: Add health check at startup
```python
def check_ollama_health():
    """Check if Ollama is running and accessible"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            return True, "Ollama is running"
        else:
            return False, f"Ollama returned status {response.status_code}"
    except requests.ConnectionError:
        return False, "Ollama is not running. Start it with: ollama serve"
    except requests.Timeout:
        return False, "Ollama is not responding (timeout)"
    except Exception as e:
        return False, f"Unexpected error: {e}"

# Add to main() function:
def main():
    # ... existing code ...
    
    # Add health check before chat mode
    if chat_mode_enabled:
        is_healthy, message = check_ollama_health()
        if not is_healthy:
            console.print(Panel(
                f"❌ {message}\n\n"
                "Please start Ollama:\n"
                "• Run: ollama serve\n"
                "• Or: systemctl start ollama",
                title="Ollama Not Available",
                style="red"
            ))
            return
        
        chat_mode(model, online)
        return
```

### 2. **Confusing Entry Points** ⚠️ MEDIUM PRIORITY
**Files**: `xencode.sh`, `xencode_core.py`, `xencode_cli.py`, `xencode/cli.py`
**Issue**: Multiple entry points confuse users

**Current State**:
- `xencode.sh` - Shell wrapper (main entry)
- `xencode_core.py` - Core logic (can be run directly)
- `xencode_cli.py` - Wrapper that calls `xencode/cli.py`
- `xencode/cli.py` - Click-based CLI (different from core)

**Problem**: Users don't know which to use!

**Fix**: Create single, clear entry point
```bash
# Rename xencode.sh to just 'xencode'
mv xencode.sh xencode
chmod +x xencode

# Update shebang and simplify
#!/usr/bin/env bash
# Xencode - AI Assistant CLI

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
    echo "❌ Ollama is not running"
    echo "Start it with: ollama serve"
    exit 1
fi

# Run Python core
exec python3 "$(dirname "$0")/xencode_core.py" "$@"
```

### 3. **No First-Run Experience** ⚠️ HIGH PRIORITY
**File**: `xencode_core.py`
**Issue**: No setup wizard for new users

**Current Behavior**:
```
User runs xencode → Assumes everything is configured → Errors if not
```

**Fix**: Add first-run detection and setup
```python
def is_first_run():
    """Check if this is the first run"""
    config_file = Path.home() / ".xencode" / "config.json"
    return not config_file.exists()

def run_first_time_setup():
    """Run interactive setup for first-time users"""
    console.print(Panel(
        "👋 Welcome to Xencode!\n\n"
        "Let's get you set up in 30 seconds...",
        title="First Run Setup",
        style="blue"
    ))
    
    # Check Ollama installation
    try:
        subprocess.run(["ollama", "--version"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        console.print(Panel(
            "❌ Ollama is not installed\n\n"
            "Install Ollama:\n"
            "• Visit: https://ollama.ai\n"
            "• Or: curl https://ollama.ai/install.sh | sh",
            title="Ollama Not Found",
            style="red"
        ))
        sys.exit(1)
    
    # Check if Ollama is running
    is_healthy, message = check_ollama_health()
    if not is_healthy:
        console.print(Panel(
            f"❌ {message}\n\n"
            "Start Ollama:\n"
            "• Run: ollama serve\n"
            "• Or: systemctl start ollama",
            title="Ollama Not Running",
            style="red"
        ))
        sys.exit(1)
    
    # Check for models
    models = get_available_models()
    if not models:
        console.print(Panel(
            "⚠️ No models installed\n\n"
            "Would you like to install a recommended model?",
            title="No Models Found",
            style="yellow"
        ))
        
        if Confirm.ask("Install recommended model (qwen3:4b)?"):
            update_model("qwen3:4b")
        else:
            console.print("[yellow]You can install models later with: ollama pull <model>[/yellow]")
            sys.exit(0)
    
    # Save config
    config_dir = Path.home() / ".xencode"
    config_dir.mkdir(exist_ok=True)
    config_file = config_dir / "config.json"
    
    config = {
        "default_model": models[0] if models else "qwen3:4b",
        "setup_completed": True,
        "setup_date": datetime.now().isoformat()
    }
    
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    console.print("[green]✅ Setup complete! Starting Xencode...[/green]\n")

# Add to main():
def main():
    # Check for first run
    if is_first_run():
        run_first_time_setup()
    
    # ... rest of main() ...
```

### 4. **Buffered Streaming** ⚠️ MEDIUM PRIORITY
**File**: `xencode_core.py` - `run_streaming_query()`
**Issue**: Response is buffered before streaming, not true real-time

**Current Code**:
```python
def run_streaming_query(model, prompt):
    # Collect the full response first, then stream it
    full_response = ""
    
    for line in response.iter_lines():
        # ... collect all chunks ...
        full_response += token
    
    # Now stream the complete response
    stream_claude_response(thinking, answer)
```

**Problem**: This defeats the purpose of streaming! User sees nothing until complete response is ready.

**Fix**: Stream tokens immediately
```python
def run_streaming_query(model, prompt):
    """True real-time streaming query"""
    memory.add_message("user", prompt, model)
    
    url = "http://localhost:11434/api/generate"
    context = memory.get_context(max_messages=5)
    
    if context:
        context_prompt = "\n\n".join([f"{msg['role']}: {msg['content']}" for msg in context])
        enhanced_prompt = f"{context_prompt}\n\nuser: {prompt}"
    else:
        enhanced_prompt = prompt
    
    payload = {"model": model, "prompt": enhanced_prompt, "stream": True}
    
    try:
        response = requests.post(url, json=payload, stream=True, timeout=RESPONSE_TIMEOUT)
        response.raise_for_status()
        
        full_response = ""
        in_thinking = False
        thinking_buffer = ""
        
        # Stream tokens immediately as they arrive
        for line in response.iter_lines():
            if line:
                try:
                    chunk = json.loads(line.decode('utf-8'))
                    
                    if chunk.get('done', False):
                        break
                    
                    if 'response' in chunk and chunk['response']:
                        token = chunk['response']
                        full_response += token
                        
                        # Detect thinking section
                        if '<think>' in token:
                            in_thinking = True
                            console.print("\n🧠 Thinking...", style="dim italic yellow")
                            continue
                        elif '</think>' in token:
                            in_thinking = False
                            console.print("\n\n📄 Answer", style="bold green")
                            continue
                        
                        # Stream token immediately
                        if in_thinking:
                            console.print(token, style="dim italic yellow", end="", highlight=False)
                        else:
                            console.print(token, end="", highlight=False)
                        
                        sys.stdout.flush()  # Force immediate output
                
                except json.JSONDecodeError:
                    continue
        
        console.print()  # New line at end
        
        # Add to memory
        memory.add_message("assistant", full_response, model)
        
        return full_response
    
    except Exception as e:
        # ... error handling ...
```

### 5. **No Project Context** ⚠️ LOW PRIORITY
**File**: New file needed
**Issue**: Doesn't detect project type or load relevant context

**Fix**: Create project context manager
```python
# Create: xencode/project_context.py

from pathlib import Path
from typing import Dict, List, Optional
import subprocess

class ProjectContextManager:
    """Detect and manage project context"""
    
    def __init__(self):
        self.cwd = Path.cwd()
        self.context = None
    
    def detect_project(self) -> Dict:
        """Detect project type and gather context"""
        context = {
            "type": self._detect_type(),
            "files": self._get_relevant_files(),
            "git": self._get_git_info(),
            "dependencies": self._get_dependencies()
        }
        
        self.context = context
        return context
    
    def _detect_type(self) -> str:
        """Detect project type"""
        if (self.cwd / "package.json").exists():
            return "javascript"
        elif (self.cwd / "requirements.txt").exists() or (self.cwd / "pyproject.toml").exists():
            return "python"
        elif (self.cwd / "Cargo.toml").exists():
            return "rust"
        elif (self.cwd / "go.mod").exists():
            return "go"
        elif (self.cwd / "pom.xml").exists():
            return "java"
        return "unknown"
    
    def _get_relevant_files(self) -> List[str]:
        """Get list of relevant files"""
        # Get recently modified files
        try:
            result = subprocess.run(
                ["git", "ls-files", "-m"],
                cwd=self.cwd,
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode == 0:
                return result.stdout.strip().split('\n')[:10]
        except:
            pass
        
        return []
    
    def _get_git_info(self) -> Dict:
        """Get git information"""
        try:
            # Get current branch
            branch_result = subprocess.run(
                ["git", "branch", "--show-current"],
                cwd=self.cwd,
                capture_output=True,
                text=True,
                timeout=2
            )
            
            # Get status
            status_result = subprocess.run(
                ["git", "status", "--short"],
                cwd=self.cwd,
                capture_output=True,
                text=True,
                timeout=2
            )
            
            return {
                "branch": branch_result.stdout.strip() if branch_result.returncode == 0 else "unknown",
                "status": status_result.stdout.strip() if status_result.returncode == 0 else ""
            }
        except:
            return {"branch": "unknown", "status": ""}
    
    def _get_dependencies(self) -> List[str]:
        """Get project dependencies"""
        deps = []
        
        # Python
        if (self.cwd / "requirements.txt").exists():
            with open(self.cwd / "requirements.txt") as f:
                deps = [line.strip() for line in f if line.strip() and not line.startswith('#')][:10]
        
        # JavaScript
        elif (self.cwd / "package.json").exists():
            import json
            with open(self.cwd / "package.json") as f:
                data = json.load(f)
                deps = list(data.get("dependencies", {}).keys())[:10]
        
        return deps
    
    def get_context_prompt(self) -> str:
        """Generate context prompt for AI"""
        if not self.context:
            self.detect_project()
        
        prompt = f"Project Context:\n"
        prompt += f"- Type: {self.context['type']}\n"
        prompt += f"- Git Branch: {self.context['git']['branch']}\n"
        
        if self.context['files']:
            prompt += f"- Modified Files: {', '.join(self.context['files'][:5])}\n"
        
        if self.context['dependencies']:
            prompt += f"- Dependencies: {', '.join(self.context['dependencies'][:5])}\n"
        
        return prompt
```

### 6. **Terminal Dependency** ⚠️ LOW PRIORITY
**File**: `xencode.sh`
**Issue**: Requires Kitty terminal, complex fallback logic

**Fix**: Work in any terminal
```bash
#!/usr/bin/env bash
# Xencode - Works in ANY terminal

# No terminal detection needed!
# Just run Python directly
exec python3 "$(dirname "$0")/xencode_core.py" "$@"
```

The Python code should handle terminal capabilities automatically using Rich library.

---

## 📝 Implementation Priority

### Week 1 (Critical)
1. ✅ Add startup health check
2. ✅ Add first-run setup
3. ✅ Simplify entry point
4. ✅ Fix streaming to be real-time

### Week 2 (Important)
5. ✅ Add project context detection
6. ✅ Remove terminal dependency
7. ✅ Improve error messages
8. ✅ Add session management UI

### Week 3 (Nice to Have)
9. ✅ Add export/import
10. ✅ Performance dashboard
11. ✅ Plugin system
12. ✅ Documentation

---

## 🧪 Testing Checklist

After implementing fixes:

- [ ] Fresh install works (no config)
- [ ] Ollama not running shows clear error
- [ ] No models installed shows clear error
- [ ] First-run setup completes successfully
- [ ] Streaming shows tokens immediately
- [ ] Chat mode works in any terminal
- [ ] Project context detected correctly
- [ ] Session switching works
- [ ] Export/import works
- [ ] All existing tests pass

---

## 🚀 Quick Start After Fixes

```bash
# User experience after fixes:

# First time
$ ./xencode
👋 Welcome to Xencode!
Let's get you set up in 30 seconds...

✅ Ollama detected
✅ Model installed: qwen3:4b
✅ Setup complete!

# Regular use
$ ./xencode
╔══════════════════════════════════════════╗
║ Xencode AI (Claude-Code Style | qwen3:4b)║
╚══════════════════════════════════════════╝

[You] > explain async/await in python

🧠 Thinking...
[Real-time streaming of thinking process]

📄 Answer
[Real-time streaming of answer]

# Inline mode
$ ./xencode "what is recursion?"
[Immediate response, no chat mode]
```

Much better! 🎉
