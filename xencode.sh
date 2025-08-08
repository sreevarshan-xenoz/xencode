#!/bin/bash

# Detect if online
if ping -q -c 1 google.com >/dev/null 2>&1; then
    ONLINE="true"
else
    ONLINE="false"
fi

# Pass args to Python core
python3 "$(dirname "$0")/xencode_core.py" "$@" --online=$ONLINE