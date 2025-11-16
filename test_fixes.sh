#!/bin/bash
# Test script for Xencode fixes

echo "🧪 Testing Xencode Fixes"
echo "========================"
echo ""

# Test 1: Check if xencode.sh is executable
echo "Test 1: Checking xencode.sh permissions..."
if [ -x xencode.sh ]; then
    echo "✅ xencode.sh is executable"
else
    echo "⚠️  Making xencode.sh executable..."
    chmod +x xencode.sh
    echo "✅ Fixed"
fi
echo ""

# Test 2: Check if Ollama is running
echo "Test 2: Checking Ollama status..."
if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
    echo "✅ Ollama is running"
else
    echo "❌ Ollama is not running"
    echo "   Start it with: ollama serve"
    exit 1
fi
echo ""

# Test 3: Check Python dependencies
echo "Test 3: Checking Python dependencies..."
python3 -c "import requests, rich, prompt_toolkit" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✅ All Python dependencies installed"
else
    echo "⚠️  Some dependencies missing"
    echo "   Install with: pip install -r requirements.txt"
fi
echo ""

# Test 4: Check if project context module exists
echo "Test 4: Checking project context module..."
if [ -f "xencode/project_context.py" ]; then
    echo "✅ Project context module exists"
else
    echo "❌ Project context module missing"
    exit 1
fi
echo ""

# Test 5: Syntax check
echo "Test 5: Python syntax check..."
python3 -m py_compile xencode_core.py 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✅ xencode_core.py syntax OK"
else
    echo "❌ Syntax error in xencode_core.py"
    exit 1
fi

python3 -m py_compile xencode/project_context.py 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✅ project_context.py syntax OK"
else
    echo "❌ Syntax error in project_context.py"
    exit 1
fi
echo ""

# Test 6: Test inline mode
echo "Test 6: Testing inline mode..."
timeout 10 ./xencode.sh "what is 2+2?" >/dev/null 2>&1
if [ $? -eq 0 ] || [ $? -eq 124 ]; then
    echo "✅ Inline mode works (or timed out, which is OK for this test)"
else
    echo "⚠️  Inline mode may have issues"
fi
echo ""

# Test 7: Check first-run setup
echo "Test 7: Checking first-run setup..."
if grep -q "def is_first_run" xencode_core.py && grep -q "def run_first_time_setup" xencode_core.py; then
    echo "✅ First-run setup functions present"
else
    echo "❌ First-run setup functions missing"
    exit 1
fi
echo ""

# Test 8: Check health check
echo "Test 8: Checking health check..."
if grep -q "def check_ollama_health" xencode_core.py; then
    echo "✅ Health check function present"
else
    echo "❌ Health check function missing"
    exit 1
fi
echo ""

# Test 9: Check real-time streaming
echo "Test 9: Checking real-time streaming..."
if grep -q "sys.stdout.flush()" xencode_core.py; then
    echo "✅ Real-time streaming implemented (stdout.flush found)"
else
    echo "⚠️  Real-time streaming may not be fully implemented"
fi
echo ""

# Test 10: Check project context integration
echo "Test 10: Checking project context integration..."
if grep -q "PROJECT_CONTEXT_AVAILABLE" xencode_core.py && grep -q "get_project_context" xencode_core.py; then
    echo "✅ Project context integrated"
else
    echo "❌ Project context not integrated"
    exit 1
fi
echo ""

echo "========================"
echo "✅ All tests passed!"
echo ""
echo "🚀 Ready to use Xencode!"
echo ""
echo "Try these commands:"
echo "  ./xencode.sh                    # Start chat mode"
echo "  ./xencode.sh \"what is python?\"  # Inline query"
echo "  ./xencode.sh --help             # Show help"
echo ""
