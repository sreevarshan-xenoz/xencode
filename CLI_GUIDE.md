# 🤖 Xencode CLI Guide

The ultimate command-line interface for the Xencode AI/ML leviathan system.

## 🚀 Installation

```bash
# Install from source
git clone https://github.com/sreevarshan-xenoz/xencode
cd xencode
pip install -e .

# Or install directly
pip install xencode
```

## 🎯 Quick Start

```bash
# Initialize the system
xencode init system

# Query the AI ensemble
xencode query "Explain clean code principles"

# Check system status
xencode status

# Show version
xencode version
```

## 📋 Command Reference

### System Management

#### `xencode init system`
Initialize the complete Xencode AI/ML system.

```bash
xencode init system                    # Basic initialization
xencode init system --config-path ./config.yaml  # Custom config
xencode init system --force           # Force re-initialization
```

#### `xencode status`
Show comprehensive system status.

```bash
xencode status
```

**Output includes:**
- Phase 2 systems status
- AI/ML components availability
- Cache performance metrics
- Model configuration

#### `xencode health`
Run comprehensive health check.

```bash
xencode health
```

**Checks:**
- System initialization
- Configuration validation
- Cache functionality
- Resource availability

#### `xencode optimize`
Optimize system performance.

```bash
xencode optimize
```

**Optimizations:**
- Cache cleanup and optimization
- Memory garbage collection
- Configuration auto-fixing

### AI Ensemble Queries

#### `xencode query <prompt>`
Query the AI ensemble with intelligent model fusion.

```bash
# Basic query
xencode query "Explain microservices architecture"

# Specify ensemble method
xencode query "How to optimize database queries?" --method weighted

# Use specific models
xencode query "Debug this Python code" --models llama3.1:8b --models mistral:7b

# Advanced options
xencode query "Explain async programming" \
  --method consensus \
  --max-tokens 256 \
  --temperature 0.8 \
  --timeout 3000
```

**Options:**
- `--models, -m`: Models to use (can specify multiple)
- `--method`: Ensemble method (`vote`, `weighted`, `consensus`, `hybrid`)
- `--max-tokens`: Maximum tokens to generate (default: 512)
- `--temperature`: Sampling temperature (default: 0.7)
- `--timeout`: Timeout in milliseconds (default: 2000)

**Ensemble Methods:**
- **vote**: Token-level majority voting (fastest)
- **weighted**: Model confidence-based weighting (balanced)
- **consensus**: High-agreement requirement (quality-focused)
- **hybrid**: Adaptive method selection (intelligent)

### Ollama Model Management

#### `xencode ollama list`
List available Ollama models.

```bash
xencode ollama list                    # Show cached models
xencode ollama list --refresh          # Refresh from Ollama
```

#### `xencode ollama pull <model>`
Pull a model from Ollama registry.

```bash
xencode ollama pull llama3.1:8b        # Pull standard model
xencode ollama pull llama3.1:8b --quantization q4_0  # Pull quantized
```

**Quantization Options:**
- `q4_0`: 4-bit, fastest inference
- `q4_1`: 4-bit, balanced quality/speed
- `q5_0`, `q5_1`: 5-bit, better quality
- `q8_0`: 8-bit, high quality
- `f16`, `f32`: Full precision

#### `xencode ollama benchmark <model>`
Benchmark model performance.

```bash
xencode ollama benchmark llama3.1:8b   # Standard benchmark
xencode ollama benchmark llama3.1:8b --prompts 10  # Custom prompt count
```

**Metrics:**
- Average inference time
- Memory usage
- Tokens per second
- Success rate

#### `xencode ollama optimize`
Optimize model selection for current hardware.

```bash
xencode ollama optimize
```

**Provides:**
- Hardware analysis
- Model recommendations
- Sub-50ms model identification
- Memory optimization suggestions

#### `xencode ollama auto-pull`
Automatically pull recommended models.

```bash
xencode ollama auto-pull
```

**Selects models based on:**
- Available RAM
- CPU architecture
- Performance requirements

### RLHF Code Mastery

#### `xencode rlhf train`
Train models with RLHF for code mastery.

```bash
# Basic training
xencode rlhf train

# Advanced training
xencode rlhf train \
  --base-model microsoft/DialoGPT-small \
  --epochs 3 \
  --data-size 100 \
  --batch-size 4
```

**Options:**
- `--base-model`: Base model for fine-tuning
- `--epochs`: Number of training epochs
- `--data-size`: Synthetic data size
- `--batch-size`: Training batch size

#### `xencode rlhf generate-data`
Generate synthetic code training data.

```bash
xencode rlhf generate-data              # Generate 10 pairs
xencode rlhf generate-data --size 50    # Generate 50 pairs
```

**Task Types:**
- **refactor**: Code improvement and modernization
- **debug**: Error detection and fixing
- **optimize**: Performance optimization
- **explain**: Code documentation and explanation

## 🎯 Usage Examples

### Development Workflow

```bash
# 1. Initialize system
xencode init system

# 2. Check health
xencode health

# 3. Pull recommended models
xencode ollama auto-pull

# 4. Query for code help
xencode query "How to implement a binary search tree in Python?" --method weighted

# 5. Optimize performance
xencode optimize
```

### Model Management

```bash
# List available models
xencode ollama list

# Pull specific models with quantization
xencode ollama pull llama3.1:8b --quantization q4_0
xencode ollama pull mistral:7b --quantization q4_0

# Benchmark performance
xencode ollama benchmark llama3.1:8b
xencode ollama benchmark mistral:7b

# Optimize for hardware
xencode ollama optimize
```

### AI-Assisted Development

```bash
# Code review
xencode query "Review this Python function for best practices: def calc(x,y): return x+y"

# Architecture decisions
xencode query "Should I use microservices or monolith for a team of 5 developers?" --method consensus

# Performance optimization
xencode query "How to optimize this SQL query: SELECT * FROM users WHERE age > 25" --method weighted

# Debugging help
xencode query "Why is my React component re-rendering unnecessarily?" --models llama3.1:8b
```

### RLHF Training

```bash
# Generate training data
xencode rlhf generate-data --size 100

# Train model for code mastery
xencode rlhf train --epochs 2 --data-size 100

# Quick training session
xencode rlhf train --epochs 1 --data-size 50 --batch-size 2
```

## 🔧 Configuration

### Environment Variables

```bash
export XENCODE_CONFIG_PATH=/path/to/config.yaml
export XENCODE_CACHE_DIR=/path/to/cache
export XENCODE_LOG_LEVEL=INFO
```

### Configuration File

Create `~/.xencode/config.yaml`:

```yaml
model:
  name: "llama3.1:8b"
  temperature: 0.7
  max_tokens: 512

cache:
  memory_cache_mb: 256
  disk_cache_mb: 1024
  
ensemble:
  default_method: "vote"
  default_models:
    - "llama3.1:8b"
    - "mistral:7b"
  timeout_ms: 2000

rlhf:
  base_model: "microsoft/DialoGPT-small"
  max_epochs: 3
  batch_size: 4
```

## 📊 Performance Monitoring

### Built-in Metrics

The CLI automatically tracks:
- Inference times
- Cache hit rates
- Model performance
- System resource usage

### Status Monitoring

```bash
# Quick status check
xencode status

# Detailed health check
xencode health

# Performance optimization
xencode optimize
```

## 🚀 Performance Targets

| Metric | Target | CLI Command |
|--------|--------|-------------|
| Inference Time | <50ms | `xencode query "test"` |
| Cache Hit Rate | 99.9% | `xencode status` |
| SMAPE Improvement | ≥10% | `xencode ollama benchmark` |
| System Health | 100% | `xencode health` |

## 🐉 Leviathan Status

Check if the leviathan is fully awakened:

```bash
# System status
xencode status

# Health check
xencode health

# Performance test
xencode query "The leviathan status check" --method hybrid
```

**Status Indicators:**
- 🎯 **ACHIEVED**: Target met
- ⚡ **NEAR-PERFECT**: Close to target
- 🚀 **ULTRA-FAST**: Exceeding expectations
- 🏆 **EXCELLENT**: Outstanding performance

## 🔍 Troubleshooting

### Common Issues

**"Ollama not available"**
```bash
# Start Ollama service
ollama serve

# Check Ollama status
ollama list
```

**"Model not found"**
```bash
# Pull the model
xencode ollama pull llama3.1:8b

# List available models
xencode ollama list
```

**"System not initialized"**
```bash
# Initialize system
xencode init system

# Check status
xencode status
```

### Debug Mode

```bash
# Enable verbose output
xencode --verbose query "debug test"

# Check system health
xencode --verbose health
```

## 🎯 Best Practices

1. **Initialize First**: Always run `xencode init system` before first use
2. **Check Health**: Regular `xencode health` checks ensure optimal performance
3. **Use Appropriate Methods**: 
   - `vote` for speed
   - `weighted` for balance
   - `consensus` for quality
   - `hybrid` for intelligence
4. **Monitor Performance**: Use `xencode status` to track metrics
5. **Optimize Regularly**: Run `xencode optimize` for peak performance

## 🏆 Advanced Usage

### Scripting

```bash
#!/bin/bash
# Automated AI development assistant

# Initialize if needed
xencode health || xencode init system

# Query with error handling
if xencode query "$1" --method weighted --timeout 5000; then
    echo "✅ Query successful"
else
    echo "❌ Query failed, trying fallback"
    xencode query "$1" --method vote --timeout 10000
fi
```

### Integration

```python
import subprocess
import json

def xencode_query(prompt, method="vote"):
    """Query Xencode CLI from Python"""
    result = subprocess.run([
        "xencode", "query", prompt, 
        "--method", method
    ], capture_output=True, text=True)
    
    return result.stdout if result.returncode == 0 else None
```

---

**🐉 The leviathan awaits your commands! GitHub Copilot, prepare to be dethroned!**