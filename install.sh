#!/bin/bash

set -e  # Exit on any error

echo "🚀 Installing Xencode (Rust) - Comprehensive Setup"
echo "=================================================="

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

# Check for Rust toolchain (needed to build xencode)
if ! command -v cargo &> /dev/null; then
    echo "❌ cargo not found, installing Rust via rustup..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    # shellcheck disable=SC1091
    source "$HOME/.cargo/env"
else
    echo "✅ cargo: $(cargo --version)"
fi

if ! command -v rustc &> /dev/null; then
    echo "❌ rustc missing after rustup install — is ~/.cargo/bin on PATH?"
    exit 1
fi

echo ""
echo "2. 🏗️  Building Xencode (release)"
echo "----------------------------------"

if [ ! -f "rust/Cargo.toml" ]; then
    echo "❌ rust/Cargo.toml not found — run install.sh from the repo root."
    exit 1
fi

echo "⏳ cargo build --release -p xencode-cli (first build takes a few minutes)..."
cargo build --release -p xencode-cli --manifest-path rust/Cargo.toml

BIN="rust/target/release/xencode"
if [ ! -x "$BIN" ]; then
    echo "❌ Build failed: $BIN not found."
    exit 1
fi
echo "✅ Build succeeded: $BIN"

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
        "debian"|"fedora"|"rhel"|*)
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
    echo "⚠️  No supported terminals found. The TUI will use the current terminal."
fi

echo ""
echo "7. 🧪 Running Installation Tests"
echo "--------------------------------"

# Make scripts executable
chmod +x scripts/smoke-test.sh

# Smoke-test the freshly built binary
echo "🔍 Smoke-testing the release binary..."
if ./scripts/smoke-test.sh "./$BIN" > /tmp/xencode_test.log 2>&1; then
    echo "✅ Smoke test: PASSED"
else
    echo "❌ Smoke test: FAILED"
    echo "📋 Check logs: cat /tmp/xencode_test.log"
    exit 1
fi

echo ""
echo "8. 🔧 Installing System Command"
echo "-------------------------------"

# Determine installation path
if [ -w "/usr/local/bin" ] 2>/dev/null; then
    INSTALL_PATH="/usr/local/bin"
    echo "📦 Installing to system-wide location: $INSTALL_PATH"
else
    INSTALL_PATH="$HOME/.local/bin"
    echo "📦 Installing to user location: $INSTALL_PATH"
    # Ensure ~/.local/bin exists and is in PATH
    mkdir -p "$INSTALL_PATH"

    # Add to PATH if not already there
    if [[ ":$PATH:" != *":$INSTALL_PATH:"* ]]; then
        echo "💡 Adding $INSTALL_PATH to PATH..."

        # Add to appropriate shell config
        if [ -n "$ZSH_VERSION" ]; then
            echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc
            echo "   Added to ~/.zshrc"
        elif [ -n "$BASH_VERSION" ]; then
            echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
            echo "   Added to ~/.bashrc"
        else
            echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.profile
            echo "   Added to ~/.profile"
        fi

        echo "   ⚠️  Restart your terminal or run: source ~/.bashrc (or ~/.zshrc)"
    fi
fi

# Install the compiled binary directly (no wrapper needed — it is the TUI+CLI)
echo "🚀 Installing xencode binary..."
if [ -w "$INSTALL_PATH" ]; then
    cp "$BIN" "$INSTALL_PATH/xencode"
else
    sudo cp "$BIN" "$INSTALL_PATH/xencode"
fi
chmod +x "$INSTALL_PATH/xencode"

# Verify installation
if [ -x "$INSTALL_PATH/xencode" ]; then
    echo "✅ xencode command installed successfully"
    echo "📍 Location: $INSTALL_PATH/xencode"

    # Test if it's in PATH
    if command -v xencode >/dev/null 2>&1; then
        echo "✅ xencode is available in PATH"
    else
        echo "⚠️  xencode not yet in PATH (restart terminal or source shell config)"
    fi
else
    echo "❌ Failed to install xencode command"
    exit 1
fi

echo ""
echo "🎉 INSTALLATION COMPLETE!"
echo "========================="
echo ""
echo "📋 Installation Summary:"
echo "  • Rust toolchain: ✅ Available"
echo "  • Release binary: ✅ Built"
echo "  • Ollama service: ✅ Running"
echo "  • AI models: ✅ Available"
echo "  • Terminal integration: $([ ${#TERMINALS_FOUND[@]} -gt 0 ] && echo '✅ Ready' || echo '⚠️  Limited')"
echo "  • Smoke test: ✅ Passed"
echo ""
echo "🚀 Usage Examples:"
echo "  xencode                              # Launch the TUI"
echo "  xencode query \"explain quantum computing\""
echo "  xencode analyze ./src                # Code + image inventory"
echo "  xencode scan .                       # Workspace listing"
echo "  xencode server --port 8765           # Collaboration server"
echo ""
if [ "$DISTRO" = "arch" ] && command -v kitty &> /dev/null; then
    echo "🖥️  Hyprland Integration (add to ~/.config/hypr/hyprland.conf):"
    echo "  bind = SUPER, Return, exec, xencode"
    echo ""
fi
echo "📚 Documentation:"
echo "  • README.md - Complete usage guide"
echo "  • docs/INSTALL_MANUAL.md - Manual installation steps"
echo ""
echo "🆘 Troubleshooting:"
echo "  • Run: ./scripts/smoke-test.sh ./rust/target/release/xencode"
echo "  • Test command: xencode query \"hello\""
echo "  • Check: systemctl status ollama"
echo "  • Logs: journalctl -u ollama -f"
echo "  • Manual start: ollama serve"
