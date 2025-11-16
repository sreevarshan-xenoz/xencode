# 🤖 Smart Model Selection - Xencode

## What's New?

Xencode now **automatically detects and selects the best available model** on your system! No more hardcoded defaults or manual configuration.

---

## ✨ Features

### 1. **Automatic Model Detection**
- Scans all installed Ollama models
- Selects the best one based on preferences
- Falls back gracefully if preferred models aren't available

### 2. **Smart Preferences**
Models are selected in this order:
1. `qwen2.5:7b` - Best quality (if you have RAM)
2. `qwen2.5:3b` - Great balance (recommended)
3. `qwen3:4b` - Fast and efficient
4. `llama3.2:3b` - Compact and quick
5. `llama3.1:8b` - Powerful (needs more RAM)
6. `mistral:7b` - Code-focused
7. `phi3:mini` - Ultra-fast
8. `gemma2:2b` - Minimal resources

### 3. **Interactive Model Browser**
New `/models` command shows all available models with:
- ✅ Health status
- ⏱️ Response time
- ⭐ Current model indicator

### 4. **Easy Model Switching**
```
/model qwen2.5:3b
```
Switch to any installed model instantly!

---

## 🚀 How It Works

### On First Run
```bash
$ ./xencode.sh

👋 Welcome to Xencode!
Let's get you set up in 30 seconds...

✅ Ollama detected
⚠️ No models installed

Would you like to install a recommended model?
Install recommended model (qwen2.5:3b)? [y/N]: y

📥 Installing qwen2.5:3b...
✅ Setup complete! Using model: qwen2.5:3b
```

### On Subsequent Runs
```bash
$ ./xencode.sh

[Automatically detects best available model]
[Uses qwen2.5:7b if available, otherwise qwen2.5:3b, etc.]

╔═══════════════════════════════════════════════════════════════╗
║                    🤖 XENCODE AI ASSISTANT                    ║
╚═══════════════════════════════════════════════════════════════╝

                        Model: qwen2.5:3b
                         🌐 Online Mode
```

---

## 💡 Usage

### View Available Models
```
You › /models
```

**Output:**
```
╔═══════════════════════════════════════════════════════════════╗
║                    🤖 Available Models                        ║
╚═══════════════════════════════════════════════════════════════╝

#  Model           Status      Response Time  Current
1  qwen2.5:7b      ✅ Healthy  0.45s         
2  qwen2.5:3b      ✅ Healthy  0.23s         ⭐
3  llama3.2:3b     ✅ Healthy  0.31s         
4  mistral:7b      ❌ Error    N/A           

💡 Tip: Use /model <name> to switch models
Current model: qwen2.5:3b
```

### Switch Models
```
You › /model qwen2.5:7b

✅ Model switched to qwen2.5:7b
```

### Check System Status
```
You › /status
```

Shows current model, health, and system info.

---

## 🎯 Smart Selection Logic

### Preference Order
```python
preferred_models = [
    "qwen2.5:7b",    # Best quality
    "qwen2.5:3b",    # Recommended
    "qwen3:4b",      # Fast
    "llama3.2:3b",   # Compact
    "llama3.1:8b",   # Powerful
    "mistral:7b",    # Code-focused
    "phi3:mini",     # Ultra-fast
    "gemma2:2b",     # Minimal
]
```

### Selection Process
1. **Check available models** - Scan Ollama
2. **Match preferences** - Find best match
3. **Health check** - Verify model works
4. **Select best** - Use highest priority healthy model
5. **Fallback** - Use first available if no match

---

## 🔧 Advanced Features

### Model Health Monitoring
Every model is checked for:
- ✅ Availability
- ⏱️ Response time
- 🔄 Status (healthy/error/unavailable)

### Automatic Fallback
If your preferred model fails:
1. Tries next preferred model
2. Falls back to any available model
3. Shows clear error if none work

### Configuration Persistence
Selected model is saved to:
```
~/.xencode/config.json
```

---

## 📊 Comparison

### Before (Old Way)
```python
DEFAULT_MODEL = "qwen3:4b"  # Hardcoded
```

**Problems:**
- ❌ Model might not be installed
- ❌ No flexibility
- ❌ No health checking
- ❌ Manual switching only

### After (New Way)
```python
DEFAULT_MODEL = get_smart_default_model()  # Dynamic
```

**Benefits:**
- ✅ Automatically finds best model
- ✅ Adapts to your system
- ✅ Health monitoring
- ✅ Easy switching
- ✅ Graceful fallbacks

---

## 🎊 Examples

### Example 1: Multiple Models Installed
```bash
$ ollama list
qwen2.5:7b
qwen2.5:3b
llama3.2:3b

$ ./xencode.sh
# Automatically selects qwen2.5:7b (best quality)
```

### Example 2: Only Small Models
```bash
$ ollama list
phi3:mini
gemma2:2b

$ ./xencode.sh
# Automatically selects phi3:mini (preferred over gemma2)
```

### Example 3: Custom Model
```bash
$ ollama list
my-custom-model:latest

$ ./xencode.sh
# Uses my-custom-model:latest (only available)
```

---

## 🚀 Best Practices

### 1. **Install Multiple Models**
```bash
ollama pull qwen2.5:3b   # Fast, efficient
ollama pull qwen2.5:7b   # High quality
ollama pull llama3.2:3b  # Backup option
```

### 2. **Check Model Health**
```
/models
```
See which models are working properly.

### 3. **Switch Based on Task**
- **Code tasks**: Use `mistral:7b` or `qwen2.5:7b`
- **Quick queries**: Use `qwen2.5:3b` or `phi3:mini`
- **Complex reasoning**: Use `qwen2.5:7b` or `llama3.1:8b`

### 4. **Monitor Performance**
```
/status
```
Check response times and system health.

---

## 🐛 Troubleshooting

### No Models Found
```
⚠️ No models installed

Install a model:
ollama pull qwen2.5:3b
```

### Model Not Responding
```
❌ Model qwen2.5:7b is not responding

Try:
1. Restart Ollama: systemctl restart ollama
2. Switch model: /model qwen2.5:3b
3. Check health: /models
```

### Slow Response Times
```
Switch to a smaller model:
/model qwen2.5:3b
or
/model phi3:mini
```

---

## 🎯 Summary

### What Changed
- ✅ **Removed hardcoded model** - No more `DEFAULT_MODEL = "qwen3:4b"`
- ✅ **Added smart selection** - Automatically picks best model
- ✅ **Added health monitoring** - Checks if models work
- ✅ **Added interactive browser** - `/models` command
- ✅ **Added easy switching** - `/model <name>` command

### Benefits
- 🚀 **Works out of the box** - No configuration needed
- 🎯 **Adapts to your system** - Uses what you have
- 💪 **Reliable** - Falls back if model fails
- 🔄 **Flexible** - Easy to switch models
- 📊 **Transparent** - See all models and their status

---

**Xencode now intelligently manages models for you!** 🤖✨

No more guessing which model to use - Xencode picks the best one automatically!
