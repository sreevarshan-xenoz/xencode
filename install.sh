#!/bin/bash

echo "🚀 Installing Xencode..."

# Check if running on Arch Linux
if ! command -v pacman &> /dev/null; then
    echo "⚠️  Warning: This script is optimized for Arch Linux"
fi

# Check for required tools
if ! command -v curl &> /dev/null; then
    echo "❌ curl is required but not installed. Install with:"
    echo "   sudo pacman -S curl"
    exit 1
fi

# Install Python dependencies
echo "📦 Installing Python dependencies..."
if command -v pip3 &> /dev/null; then
    pip3 install -r requirements.txt
elif command -v pip &> /dev/null; then
    pip install -r requirements.txt
elif command -v python3 -m pip &> /dev/null; then
    python3 -m pip install -r requirements.txt
else
    echo "❌ No pip found. Please install pip3 first:"
    echo "   sudo pacman -S python-pip"
    exit 1
fi

# Install Ollama if not present
if ! command -v ollama &> /dev/null; then
    echo "🔧 Installing Ollama..."
    if command -v pacman &> /dev/null; then
        sudo pacman -S ollama
    else
        echo "❌ Please install Ollama manually: https://ollama.ai/download"
        exit 1
    fi
fi

# Start Ollama service
echo "🔄 Starting Ollama service..."
sudo systemctl enable ollama
sudo systemctl start ollama

# Wait for Ollama to start
echo "⏳ Waiting for Ollama service to start..."
for i in {1..30}; do
    if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
        echo "✅ Ollama service is running"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "❌ Ollama service failed to start. Try manually: ollama serve"
        exit 1
    fi
    sleep 1
done

# Pull default model
echo "📥 Pulling default model (qwen:4b)..."
if ollama pull qwen:4b; then
    echo "✅ Model pulled successfully"
else
    echo "❌ Failed to pull model. Check your internet connection."
    exit 1
fi

echo "✅ Installation complete!"
echo ""
echo "Usage examples:"
echo "  ./xencode.sh \"explain quantum computing\""
echo "  ./xencode.sh -m mistral \"write a python function\""
echo "  ./xencode.sh --list-models"
echo ""
echo "For Hyprland integration, add to ~/.config/hypr/hyprland.conf:"
echo "  bind = SUPER, Return, exec, $(pwd)/xencode.sh"