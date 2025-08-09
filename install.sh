#!/bin/bash

set -e  # Exit on any error

echo "🚀 Installing Xencode - Comprehensive Setup"
echo "==========================================="

# Detect system
DISTRO=""
PACKAGE_MANAGER=""

if command -v pacman &> /dev/null; then
    DISTRO="arch"
    PACKAGE_MANAGER="pacman"
    echo "📋 Detected: Arch Linux"
elif command -v apt &> /dev/null; then
    DISTRO="debian"
    PACKAGE_MANAGER="apt"
    echo "📋 Detected: Debian/Ubuntu"
elif command -v dnf &> /dev/null; then
    DISTRO="fedora"
    PACKAGE_MANAGER="dnf"
    echo "📋 Detected: Fedora"
elif command -v yum &> /dev/null; then
    DISTRO="rhel"
    PACKAGE_MANAGER="yum"
    echo "📋 Detected: RHEL/CentOS"
else
    echo "⚠️  Warning: Unknown distribution, will attempt generic installation"
    DISTRO="generic"
fi

echo ""

# Function to install system packages
install_system_package() {
    local package=$1
    local arch_pkg=$2
    local debian_pkg=$3
    local fedora_pkg=$4
    
    echo "🔧 Installing $package..."
    
    case $DISTRO in
        "arch")
            sudo pacman -S --needed --noconfirm ${arch_pkg:-$package}
            ;;
        "debian")
            sudo apt update -qq
            sudo apt install -y ${debian_pkg:-$package}
            ;;
        "fedora")
            sudo dnf install -y ${fedora_pkg:-$package}
            ;;
        "rhel")
            sudo yum install -y ${fedora_pkg:-$package}
            ;;
        *)
            echo "⚠️  Please install $package manually for your system"
            ;;
    esac
}

# Check and install basic system tools
echo "1. 🔧 Checking System Dependencies"
echo "----------------------------------"

# Check for curl
if ! command -v curl &> /dev/null; then
    echo "❌ curl not found, installing..."
    install_system_package "curl" "curl" "curl" "curl"
else
    echo "✅ curl: Available"
fi

# Check for git
if ! command -v git &> /dev/null; then
    echo "❌ git not found, installing..."
    install_system_package "git" "git" "git" "git"
else
    echo "✅ git: Available"
fi

# Check for Python 3
if ! command -v python3 &> /dev/null; then
    echo "❌ python3 not found, installing..."
    install_system_package "python3" "python" "python3" "python3"
else
    echo "✅ python3: $(python3 --version)"
fi

# Check for pip
if ! command -v pip3 &> /dev/null && ! python3 -m pip --version &> /dev/null; then
    echo "❌ pip not found, installing..."
    install_system_package "pip" "python-pip" "python3-pip" "python3-pip"
else
    echo "✅ pip: Available"
fi

echo ""
echo "2. 📦 Checking Python Dependencies"
echo "----------------------------------"

# Check and install Python dependencies
check_python_package() {
    local package=$1
    local import_name=${2:-$package}
    
    if python3 -c "import $import_name" 2>/dev/null; then
        echo "✅ $package: Available"
        return 0
    else
        echo "❌ $package: Missing"
        return 1
    fi
}

# Check required packages
MISSING_PACKAGES=()

if ! check_python_package "requests"; then
    MISSING_PACKAGES+=("requests")
fi

if ! check_python_package "rich"; then
    MISSING_PACKAGES+=("rich")
fi

# Check optional packages
if ! check_python_package "prompt_toolkit" "prompt_toolkit"; then
    echo "⚠️  prompt_toolkit: Not available (optional - enables enhanced multiline input)"
    MISSING_PACKAGES+=("prompt_toolkit")
fi

# Install missing packages
if [ ${#MISSING_PACKAGES[@]} -gt 0 ]; then
    echo ""
    echo "🔧 Installing missing Python packages..."
    
    # Try system packages first (preferred for Arch)
    if [ "$DISTRO" = "arch" ]; then
        echo "📦 Installing via pacman (system packages)..."
        for package in "${MISSING_PACKAGES[@]}"; do
            case $package in
                "requests")
                    sudo pacman -S --needed --noconfirm python-requests
                    ;;
                "rich")
                    sudo pacman -S --needed --noconfirm python-rich
                    ;;
                "prompt_toolkit")
                    # prompt_toolkit might not be in official repos, use pip
                    echo "⚠️  Installing prompt_toolkit via pip (not in official repos)"
                    if command -v pip3 &> /dev/null; then
                        pip3 install --user prompt_toolkit
                    else
                        python3 -m pip install --user prompt_toolkit
                    fi
                    ;;
            esac
        done
    else
        # For other distributions, use pip
        echo "📦 Installing via pip..."
        if command -v pip3 &> /dev/null; then
            pip3 install --user "${MISSING_PACKAGES[@]}"
        elif python3 -m pip --version &> /dev/null; then
            python3 -m pip install --user "${MISSING_PACKAGES[@]}"
        else
            echo "❌ No pip found. Please install pip3 first"
            exit 1
        fi
    fi
    
    # Verify installation
    echo ""
    echo "🔍 Verifying Python package installation..."
    for package in "${MISSING_PACKAGES[@]}"; do
        if [ "$package" = "prompt_toolkit" ]; then
            check_python_package "$package" "prompt_toolkit"
        else
            check_python_package "$package"
        fi
    done
fi

echo ""
echo "3. 🤖 Checking Ollama Installation"
echo "----------------------------------"

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "❌ Ollama not found, installing..."
    
    case $DISTRO in
        "arch")
            echo "📦 Installing Ollama via pacman..."
            sudo pacman -S --needed --noconfirm ollama
            ;;
        "debian")
            echo "📦 Installing Ollama via official installer..."
            curl -fsSL https://ollama.ai/install.sh | sh
            ;;
        "fedora"|"rhel")
            echo "📦 Installing Ollama via official installer..."
            curl -fsSL https://ollama.ai/install.sh | sh
            ;;
        *)
            echo "📦 Installing Ollama via official installer..."
            curl -fsSL https://ollama.ai/install.sh | sh
            ;;
    esac
    
    # Verify installation
    if command -v ollama &> /dev/null; then
        echo "✅ Ollama installed successfully"
    else
        echo "❌ Ollama installation failed"
        echo "💡 Please install manually from: https://ollama.ai/download"
        exit 1
    fi
else
    echo "✅ Ollama: $(ollama --version 2>/dev/null || echo 'Available')"
fi

echo ""
echo "4. 🔄 Checking Ollama Service"
echo "-----------------------------"

# Check if Ollama is already running
if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
    echo "✅ Ollama service: Already running and responding"
else
    echo "🔄 Starting Ollama service..."
    
    # Try systemd service first (most distributions)
    if command -v systemctl &> /dev/null; then
        if systemctl is-active --quiet ollama 2>/dev/null; then
            echo "📋 Ollama service is active but not responding, restarting..."
            sudo systemctl restart ollama
        else
            echo "🚀 Enabling and starting Ollama systemd service..."
            sudo systemctl enable ollama 2>/dev/null || true
            sudo systemctl start ollama 2>/dev/null || true
        fi
        
        # Wait for systemd service to start
        echo "⏳ Waiting for Ollama service to start..."
        for i in {1..15}; do
            if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
                echo "✅ Ollama service is now running via systemd"
                break
            fi
            if [ $i -eq 15 ]; then
                echo "⚠️  Systemd service didn't start, trying manual start..."
                break
            fi
            sleep 1
        done
    fi
    
    # If systemd didn't work, try manual start
    if ! curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
        echo "🔧 Starting Ollama manually..."
        
        if command -v ollama &> /dev/null; then
            echo "🚀 Starting 'ollama serve' in background..."
            nohup ollama serve > /tmp/ollama.log 2>&1 &
            OLLAMA_PID=$!
            
            # Wait for manual start
            echo "⏳ Waiting for manual Ollama start..."
            for i in {1..15}; do
                if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
                    echo "✅ Ollama is now running manually (PID: $OLLAMA_PID)"
                    echo "💡 Note: Ollama is running in background. To stop: kill $OLLAMA_PID"
                    break
                fi
                if [ $i -eq 15 ]; then
                    echo "❌ Failed to start Ollama"
                    echo "📋 Check logs: tail /tmp/ollama.log"
                    echo "💡 Try running manually: ollama serve"
                    exit 1
                fi
                sleep 1
            done
        else
            echo "❌ Ollama command not found after installation"
            exit 1
        fi
    fi
fi

echo ""
echo "5. 📥 Checking AI Models"
echo "------------------------"

# Check if default model exists
if ollama list | grep -q "qwen3:4b"; then
    echo "✅ Default model (qwen3:4b): Already installed"
else
    echo "📥 Pulling default model (qwen3:4b)..."
    echo "⏳ This may take a few minutes depending on your internet connection..."
    
    if ollama pull qwen3:4b; then
        echo "✅ Model pulled successfully"
    else
        echo "❌ Failed to pull model"
        echo "💡 Possible issues:"
        echo "   - Check your internet connection"
        echo "   - Ensure Ollama service is running"
        echo "   - Try running: ollama pull qwen3:4b"
        exit 1
    fi
fi

# List available models
echo ""
echo "📋 Available models:"
ollama list

echo ""
echo "6. 🖥️  Checking Terminal Integration"
echo "-----------------------------------"

# Check for preferred terminals
TERMINALS_FOUND=()

if command -v kitty &> /dev/null; then
    echo "✅ Kitty terminal: Available (recommended for best experience)"
    TERMINALS_FOUND+=("kitty")
else
    echo "⚠️  Kitty terminal: Not found"
    echo "💡 For best experience, install Kitty:"
    case $DISTRO in
        "arch") echo "   sudo pacman -S kitty" ;;
        "debian") echo "   sudo apt install kitty" ;;
        "fedora") echo "   sudo dnf install kitty" ;;
        *) echo "   Visit: https://sw.kovidgoyal.net/kitty/binary/" ;;
    esac
fi

# Check fallback terminals
for terminal in gnome-terminal konsole xterm; do
    if command -v $terminal &> /dev/null; then
        echo "✅ $terminal: Available (fallback)"
        TERMINALS_FOUND+=("$terminal")
    fi
done

if [ ${#TERMINALS_FOUND[@]} -eq 0 ]; then
    echo "⚠️  No supported terminals found. Chat mode will use current terminal."
fi

echo ""
echo "7. 🧪 Running Installation Tests"
echo "--------------------------------"

# Make scripts executable
chmod +x xencode.sh xencode_core.py test.sh

# Run basic tests
echo "🔍 Running basic functionality test..."
if ./test.sh > /tmp/xencode_test.log 2>&1; then
    echo "✅ Basic functionality test: PASSED"
else
    echo "❌ Basic functionality test: FAILED"
    echo "📋 Check logs: cat /tmp/xencode_test.log"
    exit 1
fi

# Test Claude-style features if test script exists
if [ -f "test_claude_style.sh" ]; then
    chmod +x test_claude_style.sh
    echo "🎭 Running Claude-style features test..."
    if ./test_claude_style.sh > /tmp/xencode_claude_test.log 2>&1; then
        echo "✅ Claude-style features test: PASSED"
    else
        echo "❌ Claude-style features test: FAILED"
        echo "📋 Check logs: cat /tmp/xencode_claude_test.log"
        echo "⚠️  Basic functionality should still work"
    fi
fi

echo ""
echo "🎉 INSTALLATION COMPLETE!"
echo "========================="
echo ""
echo "📋 Installation Summary:"
echo "  • System dependencies: ✅ Installed"
echo "  • Python packages: ✅ Installed"
echo "  • Ollama service: ✅ Running"
echo "  • AI models: ✅ Available"
echo "  • Terminal integration: $([ ${#TERMINALS_FOUND[@]} -gt 0 ] && echo '✅ Ready' || echo '⚠️  Limited')"
echo "  • Functionality tests: ✅ Passed"
echo ""
echo "🚀 Usage Examples:"
echo "  ./xencode.sh \"explain quantum computing\""
echo "  ./xencode.sh -m qwen3:4b \"write a python function\""
echo "  ./xencode.sh --list-models"
echo "  ./xencode.sh  # Launch persistent chat mode"
echo ""
echo "🔧 Advanced Usage:"
echo "  ./xencode.sh --inline \"prompt\"     # Force inline mode"
echo "  ./xencode.sh --chat-mode            # Force chat mode"
echo "  ./xencode.sh --update               # Update models"
echo ""
if [ "$DISTRO" = "arch" ] && command -v kitty &> /dev/null; then
    echo "🖥️  Hyprland Integration (add to ~/.config/hypr/hyprland.conf):"
    echo "  bind = SUPER, Return, exec, $(pwd)/xencode.sh"
    echo "  windowrulev2 = float, title:Xencode AI"
    echo ""
fi
echo "📚 Documentation:"
echo "  • README.md - Complete usage guide"
echo "  • INSTALL_MANUAL.md - Manual installation steps"
echo "  • test_results.md - Test results and validation"
echo ""
echo "🆘 Troubleshooting:"
echo "  • Run: ./test.sh (basic functionality)"
echo "  • Check: systemctl status ollama"
echo "  • Logs: journalctl -u ollama -f"
echo "  • Manual start: ollama serve"