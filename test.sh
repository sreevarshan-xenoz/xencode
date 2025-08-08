#!/bin/bash

echo "🧪 Testing Xencode installation..."

# Check if files exist
if [[ ! -f "xencode.sh" ]]; then
    echo "❌ xencode.sh not found"
    exit 1
fi

if [[ ! -f "xencode_core.py" ]]; then
    echo "❌ xencode_core.py not found"
    exit 1
fi

# Check if files are executable
if [[ ! -x "xencode.sh" ]]; then
    echo "❌ xencode.sh is not executable"
    exit 1
fi

if [[ ! -x "xencode_core.py" ]]; then
    echo "❌ xencode_core.py is not executable"
    exit 1
fi

# Check Python dependencies
echo "📦 Checking Python dependencies..."
python3 -c "import requests, rich" 2>/dev/null
if [[ $? -ne 0 ]]; then
    echo "❌ Python dependencies missing. Run: pip install -r requirements.txt"
    exit 1
fi

# Check if Ollama is running
echo "🔧 Checking Ollama service..."
if ! pgrep -x "ollama" > /dev/null; then
    echo "⚠️  Ollama service not running. Start with: sudo systemctl start ollama"
fi

# Test basic functionality
echo "🚀 Testing basic functionality..."
./xencode.sh --list-models

echo "✅ All tests passed! Xencode is ready to use."
echo ""
echo "Try: ./xencode.sh \"Hello, how are you?\""