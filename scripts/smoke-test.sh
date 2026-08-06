#!/usr/bin/env bash
# Smoke test for xencode binary
set -euo pipefail

BINARY=${1:-"./rust/target/release/xencode"}
ERRORS=0

echo "🔍 Smoke testing $BINARY..."
echo ""

# Test 1: Version flag
if $BINARY --version 2>&1 | grep -qi "xencode"; then
    echo "  ✅ --version"
else
    echo "  ❌ --version"
    ERRORS=$((ERRORS+1))
fi

# Test 2: Help flag
if $BINARY --help 2>&1 | grep -qE "Commands:|Subcommand"; then
    echo "  ✅ --help"
else
    echo "  ❌ --help"
    ERRORS=$((ERRORS+1))
fi

# Test 3: Config show
if $BINARY config show 2>&1 | grep -q "default_model"; then
    echo "  ✅ config show"
else
    echo "  ❌ config show"
    ERRORS=$((ERRORS+1))
fi

# Test 4: Scan current directory
if $BINARY scan . --max-depth 1 2>&1 | grep -qE "(file|dir|kind)"; then
    echo "  ✅ scan"
else
    echo "  ❌ scan"
    ERRORS=$((ERRORS+1))
fi

# Test 5: Analyze help
if $BINARY analyze --help 2>&1 | grep -q "analyze"; then
    echo "  ✅ analyze --help"
else
    echo "  ❌ analyze --help"
    ERRORS=$((ERRORS+1))
fi

# Test 6: Plugin help
if $BINARY plugin --help 2>&1 | grep -q "plugin"; then
    echo "  ✅ plugin --help"
else
    echo "  ❌ plugin --help"
    ERRORS=$((ERRORS+1))
fi

# Test 7: Server help
if $BINARY server --help 2>&1 | grep -q "server"; then
    echo "  ✅ server --help"
else
    echo "  ❌ server --help"
    ERRORS=$((ERRORS+1))
fi

echo ""
if [ "$ERRORS" -eq 0 ]; then
    echo "🎉 All smoke tests passed!"
else
    echo "❌ $ERRORS smoke test(s) failed."
fi
exit $ERRORS
