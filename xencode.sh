#!/bin/bash

# Detect if online
if ping -q -c 1 google.com >/dev/null 2>&1; then
    ONLINE="true"
else
    ONLINE="false"
fi

# Mode detection and argument parsing
if [ $# -eq 0 ]; then
    # No arguments provided - launch persistent chat mode
    if command -v kitty >/dev/null 2>&1; then
        # Set environment variable for chat mode
        export XENCODE_MODE=chat
        # Launch Kitty terminal with proper title and dimensions
        if ! kitty --title "Xencode AI" --class XencodeAI \
              -o remember_window_size=no \
              -o initial_window_width=1200 \
              -o initial_window_height=800 \
              python3 "$(dirname "$0")/xencode_core.py" --chat-mode --online=$ONLINE; then
            echo "❌ Failed to launch Kitty terminal. Falling back to inline mode."
            echo "💡 You can try running with a prompt: ./xencode.sh --inline \"your prompt\""
            exit 1
        fi
    else
        # Kitty not available - try fallback terminals
        echo "⚠️  Kitty terminal not found. Attempting fallback to default terminal..."
        echo "💡 For best experience, install Kitty: sudo pacman -S kitty"
        
        # Try common terminals as fallback
        if command -v gnome-terminal >/dev/null 2>&1; then
            export XENCODE_MODE=chat
            gnome-terminal --title="Xencode AI" -- python3 "$(dirname "$0")/xencode_core.py" --chat-mode --online=$ONLINE
        elif command -v konsole >/dev/null 2>&1; then
            export XENCODE_MODE=chat
            konsole --title "Xencode AI" -e python3 "$(dirname "$0")/xencode_core.py" --chat-mode --online=$ONLINE
        elif command -v xterm >/dev/null 2>&1; then
            export XENCODE_MODE=chat
            xterm -title "Xencode AI" -e python3 "$(dirname "$0")/xencode_core.py" --chat-mode --online=$ONLINE
        else
            echo "❌ No suitable terminal found. Falling back to inline mode in current terminal."
            echo "💡 You can continue with: ./xencode.sh --inline \"your prompt\""
            echo "🔧 Or install a supported terminal: sudo pacman -S kitty"
            # Fall back to inline mode without prompt (will show help/usage)
            python3 "$(dirname "$0")/xencode_core.py" --chat-mode --online=$ONLINE
        fi
    fi
elif [ "$1" = "--inline" ]; then
    # Explicit inline mode - shift to remove --inline flag and process remaining args
    shift
    if [ $# -eq 0 ]; then
        echo "❌ Error: --inline flag requires a prompt"
        exit 1
    fi
    # Pass remaining args to Python core for inline processing
    python3 "$(dirname "$0")/xencode_core.py" "$@" --online=$ONLINE
elif [ "$1" = "--chat-mode" ]; then
    # Explicit chat mode request - try to launch terminal
    shift
    PROMPT_ARG=""
    if [ $# -gt 0 ]; then
        PROMPT_ARG="$1"
    fi
    
    if command -v kitty >/dev/null 2>&1; then
        export XENCODE_MODE=chat
        if ! kitty --title "Xencode AI" --class XencodeAI \
              -o remember_window_size=no \
              -o initial_window_width=1200 \
              -o initial_window_height=800 \
              python3 "$(dirname "$0")/xencode_core.py" --chat-mode --online=$ONLINE; then
            echo "❌ Failed to launch Kitty terminal."
            if [ -n "$PROMPT_ARG" ]; then
                echo "🔄 Falling back to inline mode with your prompt..."
                python3 "$(dirname "$0")/xencode_core.py" "$PROMPT_ARG" --online=$ONLINE
            else
                echo "💡 Try running with: ./xencode.sh --inline \"your prompt\""
                exit 1
            fi
        fi
    else
        echo "❌ Kitty terminal not found for --chat-mode."
        if [ -n "$PROMPT_ARG" ]; then
            echo "🔄 Falling back to inline mode with your prompt..."
            python3 "$(dirname "$0")/xencode_core.py" "$PROMPT_ARG" --online=$ONLINE
        else
            echo "💡 Install Kitty: sudo pacman -S kitty"
            echo "💡 Or use inline mode: ./xencode.sh --inline \"your prompt\""
            exit 1
        fi
    fi
else
    # Default behavior - pass all args to Python core (backward compatibility)
    python3 "$(dirname "$0")/xencode_core.py" "$@" --online=$ONLINE
fi