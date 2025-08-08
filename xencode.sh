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
        kitty --title "Xencode AI" --class XencodeAI \
              -o remember_window_size=no \
              -o initial_window_width=1200 \
              -o initial_window_height=800 \
              python3 "$(dirname "$0")/xencode_core.py" --chat-mode --online=$ONLINE
    else
        echo "❌ Kitty terminal not found. Please install Kitty or use --inline mode."
        echo "Install Kitty: sudo pacman -S kitty"
        exit 1
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
else
    # Default behavior - pass all args to Python core (backward compatibility)
    python3 "$(dirname "$0")/xencode_core.py" "$@" --online=$ONLINE
fi