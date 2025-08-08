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

# Try system packages first (Arch Linux way)
if command -v pacman &> /dev/null; then
    echo "🔧 Installing Python packages via pacman..."
    sudo pacman -S --needed python-requests python-rich
else
    # Fallback to pip methods for other systems
    if command -v pip3 &> /dev/null; then
        pip3 install --user -r requirements.txt
    elif command -v pip &> /dev/null; then
        pip install --user -r requirements.txt
    elif command -v python3 -m pip &> /dev/null; then
        python3 -m pip install --user -r requirements.txt
    else
        echo "❌ No pip found. Please install pip3 first:"
        echo "   sudo pacman -S python-pip"
        exit 1
    fi
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

# Check if Ollama is already running
echo "🔍 Checking Ollama service status..."
if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
    echo "✅ Ollama is already running"
else
    echo "🔄 Starting Ollama service..."
    
    # Try systemd service first
    if systemctl is-active --quiet ollama; then
        echo "📋 Ollama service is active but not responding, restarting..."
        sudo systemctl restart ollama
    else
        # Enable and start systemd service
        sudo systemctl enable ollama
        sudo systemctl start ollama
    fi
    
    # Wait for Ollama to start
    echo "⏳ Waiting for Ollama service to start..."
    for i in {1..30}; do
        if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
            echo "✅ Ollama service is now running"
            break
        fi
        if [ $i -eq 30 ]; then
            echo "❌ Ollama service failed to start via systemd"
            echo "🔧 Trying to start Ollama manually..."
            
            # Try to start ollama serve in background
            if command -v ollama &> /dev/null; then
                echo "🚀 Starting 'ollama serve' in background..."
                nohup ollama serve > /dev/null 2>&1 &
                sleep 5
                
                # Check if it's running now
                if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
                    echo "✅ Ollama is now running via 'ollama serve'"
                else
                    echo "❌ Failed to start Ollama. Please run 'ollama serve' manually in another terminal"
                    exit 1
                fi
            else
                echo "❌ Ollama command not found. Please install Ollama first"
                exit 1
            fi
            break
        fi
        sleep 1
    done
fi

# Pull default model
echo "📥 Pulling default model (qwen3:4b)..."
if ollama pull qwen3:4b; then
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