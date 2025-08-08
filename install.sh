#!/bin/bash

echo "🚀 Installing Xencode..."

# Check if running on Arch Linux
if ! command -v pacman &> /dev/null; then
    echo "⚠️  Warning: This script is optimized for Arch Linux"
fi

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

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

# Pull default model
echo "📥 Pulling default model (qwen:4b)..."
ollama pull qwen:4b

echo "✅ Installation complete!"
echo ""
echo "Usage examples:"
echo "  ./xencode.sh \"explain quantum computing\""
echo "  ./xencode.sh -m mistral \"write a python function\""
echo "  ./xencode.sh --list-models"
echo ""
echo "For Hyprland integration, add to ~/.config/hypr/hyprland.conf:"
echo "  bind = SUPER, Return, exec, $(pwd)/xencode.sh"