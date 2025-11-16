# Quick Fix Guide - Xencode

## 🎯 Priority Fixes (Do These First)

### Fix #1: Real-Time Streaming (2 hours)
**File**: `xencode_core.py` - function `run_streaming_query()`

**Current Problem**: Response is buffered before streaming
**Fix**: Stream tokens immediately as they arrive

**Replace this section** (around line 700):
```python
# OLD CODE - Buffers everything first
full_response = ""
for line in response.iter_lines():
    if line:
        chunk = json.loads(line.decode('utf-8'))
        if chunk.get('done', False):
            break
        if 'response' in chunk and chunk['response']:
            token = chunk['response']
            full_response += token

# Then streams the buffered response
thinking, answer = extract_thinking_and_answer(full_response)
stream_claude_response(thinking, answer)
```

**With this**:
```python
# NEW CODE - Streams immediately
full_response = ""
in_thinking = False

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
```

**Test**:
```bash
./xencode.sh
# Type a message
# Should see tokens appear immediately, not all at once
```

---

### Fix #2: Simplify Entry Point (30 minutes)
**File**: `xencode.sh`

**Current Problem**: Complex terminal detection, multiple modes
**Fix**: Simple, works everywhere

**Replace entire file with**:
```bash
#!/usr/bin/env bash
# Xencode - AI Assistant CLI

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
    echo "❌ Ollama is not running"
    echo ""
    echo "Start Ollama with one of these commands:"
    echo "  • ollama serve"
    echo "  • systemctl start ollama"
    echo ""
    exit 1
fi

# Detect online/offline
if ping -q -c 1 -W 1 8.8.8.8 >/dev/null 2>&1; then
    export XENCODE_ONLINE=true
else
    export XENCODE_ONLINE=false
fi

# Run Python core
exec python3 "$(dirname "$0")/xencode_core.py" "$@"
```

**Update xencode_core.py** to read environment variable:
```python
# In main() function, replace:
online = "false"

# With:
online = os.environ.get('XENCODE_ONLINE', 'false')
```

**Test**:
```bash
./xencode.sh                    # Chat mode
./xencode.sh "what is python?"  # Inline mode
```

---

### Fix #3: Project Context Detection (3 hours)
**Create new file**: `xencode/project_context.py`

```python
#!/usr/bin/env python3
"""
Project Context Detection for Xencode

Automatically detects project type and gathers relevant context.
"""

import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

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
        """Detect project type from files"""
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
        elif (self.cwd / "Gemfile").exists():
            return "ruby"
        return "unknown"
    
    def _get_relevant_files(self) -> List[str]:
        """Get list of recently modified files"""
        try:
            result = subprocess.run(
                ["git", "ls-files", "-m"],
                cwd=self.cwd,
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode == 0:
                files = result.stdout.strip().split('\n')
                return [f for f in files if f][:10]
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
                "status": status_result.stdout.strip() if status_result.returncode == 0 else "",
                "has_changes": bool(status_result.stdout.strip())
            }
        except:
            return {"branch": "unknown", "status": "", "has_changes": False}
    
    def _get_dependencies(self) -> List[str]:
        """Get project dependencies"""
        deps = []
        
        # Python
        if (self.cwd / "requirements.txt").exists():
            try:
                with open(self.cwd / "requirements.txt") as f:
                    deps = [line.strip().split('==')[0] for line in f 
                           if line.strip() and not line.startswith('#')][:10]
            except:
                pass
        
        # JavaScript
        elif (self.cwd / "package.json").exists():
            try:
                with open(self.cwd / "package.json") as f:
                    data = json.load(f)
                    deps = list(data.get("dependencies", {}).keys())[:10]
            except:
                pass
        
        return deps
    
    def get_context_prompt(self) -> str:
        """Generate context prompt for AI"""
        if not self.context:
            self.detect_project()
        
        if self.context['type'] == 'unknown':
            return ""
        
        prompt = f"\n[Project Context]\n"
        prompt += f"Type: {self.context['type']}\n"
        
        if self.context['git']['branch'] != 'unknown':
            prompt += f"Git Branch: {self.context['git']['branch']}\n"
        
        if self.context['git']['has_changes']:
            prompt += f"Status: Uncommitted changes\n"
        
        if self.context['files']:
            prompt += f"Modified Files: {', '.join(self.context['files'][:5])}\n"
        
        if self.context['dependencies']:
            prompt += f"Dependencies: {', '.join(self.context['dependencies'][:5])}\n"
        
        prompt += "[/Project Context]\n\n"
        
        return prompt
    
    def should_include_context(self, prompt: str) -> bool:
        """Determine if context should be included"""
        # Include context for code-related queries
        code_keywords = ['code', 'function', 'class', 'bug', 'error', 'fix', 
                        'implement', 'refactor', 'test', 'debug']
        
        prompt_lower = prompt.lower()
        return any(keyword in prompt_lower for keyword in code_keywords)


# Global instance
_project_context = None

def get_project_context() -> ProjectContextManager:
    """Get or create global project context manager"""
    global _project_context
    if _project_context is None:
        _project_context = ProjectContextManager()
    return _project_context
```

**Integrate into xencode_core.py**:
```python
# Add import at top
try:
    from xencode.project_context import get_project_context
    PROJECT_CONTEXT_AVAILABLE = True
except ImportError:
    PROJECT_CONTEXT_AVAILABLE = False

# In run_streaming_query() and run_query(), add context:
def run_streaming_query(model, prompt):
    """Enhanced streaming query with project context"""
    memory.add_message("user", prompt, model)
    
    # Build context-aware prompt
    context = memory.get_context(max_messages=5)
    context_prompt = "\n\n".join([f"{msg['role']}: {msg['content']}" for msg in context])
    
    # Add project context if available and relevant
    if PROJECT_CONTEXT_AVAILABLE:
        project_ctx = get_project_context()
        if project_ctx.should_include_context(prompt):
            project_info = project_ctx.get_context_prompt()
            enhanced_prompt = f"{project_info}{context_prompt}\n\nuser: {prompt}"
        else:
            enhanced_prompt = f"{context_prompt}\n\nuser: {prompt}"
    else:
        enhanced_prompt = f"{context_prompt}\n\nuser: {prompt}"
    
    # ... rest of function
```

**Test**:
```bash
cd /path/to/python/project
./xencode.sh
# Ask: "how can I improve this code?"
# Should include project context in response
```

---

## 🧪 Testing Checklist

After implementing each fix:

### Real-Time Streaming
- [ ] Start chat mode
- [ ] Ask a question
- [ ] Tokens appear immediately (not all at once)
- [ ] Thinking section streams character by character
- [ ] Answer section streams character by character

### Simplified Entry Point
- [ ] `./xencode.sh` starts chat mode
- [ ] `./xencode.sh "question"` gives inline response
- [ ] Works in any terminal (not just Kitty)
- [ ] Clear error if Ollama not running

### Project Context
- [ ] In Python project, context detected
- [ ] In JavaScript project, context detected
- [ ] Git branch shown in context
- [ ] Modified files shown in context
- [ ] Dependencies shown in context
- [ ] Context only included for code questions

---

## 📊 Expected Results

### Before Fixes
```bash
$ ./xencode.sh
[Complex terminal detection]
[Launches Kitty or fails]
[Buffers response, then shows all at once]
[No project awareness]
```

### After Fixes
```bash
$ ./xencode.sh
╔══════════════════════════════════════════╗
║ Xencode AI (Claude-Code Style | qwen3:4b)║
╚══════════════════════════════════════════╝

[Project Context]
Type: python
Git Branch: main
Modified Files: xencode_core.py, README.md
Dependencies: requests, rich, prompt_toolkit
[/Project Context]

[You] > how can I improve error handling?

🧠 Thinking...
[Tokens appear immediately as they're generated]

📄 Answer
[Tokens appear immediately as they're generated]
```

Much better! 🎉

---

## ⏱️ Time Estimates

- **Fix #1 (Streaming)**: 2 hours
- **Fix #2 (Entry Point)**: 30 minutes
- **Fix #3 (Project Context)**: 3 hours
- **Testing**: 1 hour
- **Total**: ~6.5 hours

---

## 🚀 Quick Implementation Order

1. **Morning**: Fix streaming (2 hours)
2. **Before Lunch**: Simplify entry point (30 min)
3. **Afternoon**: Add project context (3 hours)
4. **End of Day**: Test everything (1 hour)

**Result**: Production-ready Xencode in one day! 🎉

---

## 💡 Pro Tips

1. **Test incrementally** - Don't implement all fixes at once
2. **Keep backups** - `git commit` after each fix
3. **Test in real projects** - Use actual Python/JS projects
4. **Get user feedback** - Have someone else test it
5. **Document changes** - Update README.md

---

## 🆘 If Something Breaks

### Streaming not working?
- Check `sys.stdout.flush()` is called
- Verify no buffering in terminal
- Test with `python3 -u` (unbuffered)

### Entry point not working?
- Check file permissions: `chmod +x xencode.sh`
- Verify Python path: `which python3`
- Test Ollama connection: `curl http://localhost:11434/api/tags`

### Project context not detected?
- Check if in git repository: `git status`
- Verify files exist: `ls package.json` or `ls requirements.txt`
- Test subprocess calls: `git branch --show-current`

---

## ✅ Success Criteria

You'll know it's working when:

1. ✅ Tokens appear immediately (not buffered)
2. ✅ Works in any terminal
3. ✅ Shows project context for code questions
4. ✅ Clear errors if Ollama not running
5. ✅ Smooth first-run experience

---

**Ready to implement? Start with Fix #1 (Streaming) - it has the biggest UX impact!**

Good luck! 🚀
