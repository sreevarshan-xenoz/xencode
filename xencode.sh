#!/usr/bin/env bash
# Xencode - AI Assistant CLI (Immersive Mode)

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
    echo "❌ Ollama is not running"
    echo ""
    echo "Start Ollama with one of these commands:"
    echo "  • ollama serve"
    echo "  • systemctl start ollama"
    echo ""
    exit 1
fi

# Detect online/offline status
if ping -q -c 1 -W 1 8.8.8.8 >/dev/null 2>&1; then
    export XENCODE_ONLINE=true
else
    export XENCODE_ONLINE=false
fi

# If no arguments, force chat mode in current terminal
if [ $# -eq 0 ]; then
    export XENCODE_FORCE_CHAT=true
fi

# Run Python core in current terminal (immersive mode!)
exec python3 "$(dirname "$0")/xencode_core.py" "$@"