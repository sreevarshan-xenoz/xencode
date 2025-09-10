#!/bin/bash

# Enhanced Features Test Script for Xencode
# Tests all new advanced features

set -e

echo "🧪 Testing Xencode Enhanced Features"
echo "===================================="
echo

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Function to print colored output
print_status() {
    echo -e "${GREEN}[PASS]${NC} $1"
}

print_error() {
    echo -e "${RED}[FAIL]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

# Test counter
TESTS_PASSED=0
TESTS_FAILED=0
TESTS_TOTAL=0

# Function to run a test
run_test() {
    local test_name="$1"
    local test_command="$2"
    local expected_output="$3"
    
    TESTS_TOTAL=$((TESTS_TOTAL + 1))
    echo -n "Testing $test_name... "
    
    if eval "$test_command" >/dev/null 2>&1; then
        print_status "$test_name"
        TESTS_PASSED=$((TESTS_PASSED + 1))
    else
        print_error "$test_name"
        TESTS_FAILED=$((TESTS_FAILED + 1))
    fi
}

# Function to test file existence
test_file_exists() {
    local file_path="$1"
    local test_name="$2"
    
    TESTS_TOTAL=$((TESTS_TOTAL + 1))
    echo -n "Testing $test_name... "
    
    if [ -f "$file_path" ]; then
        print_status "$test_name"
        TESTS_PASSED=$((TESTS_PASSED + 1))
    else
        print_error "$test_name"
        TESTS_FAILED=$((TESTS_FAILED + 1))
    fi
}

# Function to test directory existence
test_dir_exists() {
    local dir_path="$1"
    local test_name="$2"
    
    TESTS_TOTAL=$((TESTS_TOTAL + 1))
    echo -n "Testing $test_name... "
    
    if [ -d "$dir_path" ]; then
        print_status "$test_name"
        TESTS_PASSED=$((TESTS_PASSED + 1))
    else
        print_error "$test_name"
        TESTS_FAILED=$((TESTS_FAILED + 1))
    fi
}

echo "1. 🔧 Testing Core Scripts"
echo "--------------------------"

# Test if main scripts exist and are executable
test_file_exists "xencode.sh" "Main launcher script exists"
test_file_exists "xencode_core.py" "Core Python script exists"
test_file_exists "install.sh" "Installation script exists"

# Test if scripts are executable
run_test "Main launcher is executable" "test -x xencode.sh" ""
run_test "Core script is executable" "test -x xencode_core.py" ""
run_test "Install script is executable" "test -x install.sh" ""

echo
echo "2. 🐍 Testing Python Dependencies"
echo "--------------------------------"

# Test Python imports
run_test "Rich library import" "python3 -c 'import rich'" ""
run_test "Requests library import" "python3 -c 'import requests'" ""
run_test "Pathlib import" "python3 -c 'from pathlib import Path'" ""

# Test optional dependencies
if python3 -c "import prompt_toolkit" 2>/dev/null; then
    print_status "Prompt toolkit available (enhanced input support)"
else
    print_warning "Prompt toolkit not available (basic input support only)"
fi

echo
echo "3. 🤖 Testing Ollama Integration"
echo "--------------------------------"

# Test if Ollama is running
if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
    print_status "Ollama service is running"
    
    # Test model listing
    run_test "Model listing works" "ollama list" ""
    
    # Test if default model exists
    if ollama list | grep -q "qwen3:4b"; then
        print_status "Default model (qwen3:4b) is available"
    else
        print_warning "Default model not found - run: ollama pull qwen3:4b"
    fi
else
    print_error "Ollama service is not running"
    print_info "Start with: systemctl start ollama"
fi

echo
echo "4. 🧠 Testing Enhanced Features"
echo "-------------------------------"

# Test if Xencode directories exist
test_dir_exists "$HOME/.xencode" "Xencode home directory exists"
test_dir_exists "$HOME/.xencode/cache" "Cache directory exists"
test_dir_exists "$HOME/.xencode/exports" "Exports directory exists"

# Test if conversation memory file exists (will be created on first run)
if [ -f "$HOME/.xencode/conversation_memory.json" ]; then
    print_status "Conversation memory file exists"
else
    print_info "Conversation memory file will be created on first run"
fi

echo
echo "5. 🎯 Testing Command Line Interface"
echo "-----------------------------------"

# Test help command
run_test "Help command works" "./xencode.sh --help 2>/dev/null || echo 'help' | ./xencode.sh" ""

# Test model listing
run_test "Model listing command works" "./xencode.sh --list-models" ""

# Test status command
run_test "Status command works" "./xencode.sh --status" ""

# Test memory command
run_test "Memory command works" "./xencode.sh --memory" ""

# Test sessions command
run_test "Sessions command works" "./xencode.sh --sessions" ""

# Test cache command
run_test "Cache command works" "./xencode.sh --cache" ""

echo
echo "6. 🔍 Testing Python Core Features"
echo "----------------------------------"

# Test Python script syntax
run_test "Python script syntax is valid" "python3 -m py_compile xencode_core.py" ""

# Test if enhanced classes can be imported
run_test "Enhanced classes can be imported" "python3 -c 'from xencode_core import ConversationMemory, ResponseCache, ModelManager'" ""

echo
echo "7. 📊 Testing Performance Features"
echo "---------------------------------"

# Test if cache directory is writable
run_test "Cache directory is writable" "test -w $HOME/.xencode/cache" ""

# Test if exports directory is writable
run_test "Exports directory is writable" "test -w $HOME/.xencode/exports" ""

# Test if memory file is writable (if it exists)
if [ -f "$HOME/.xencode/conversation_memory.json" ]; then
    run_test "Memory file is writable" "test -w $HOME/.xencode/conversation_memory.json" ""
fi

echo
echo "8. 🎨 Testing UI Enhancements"
echo "-----------------------------"

# Test if Rich library features work
run_test "Rich console creation" "python3 -c 'from rich.console import Console; Console()'" ""
run_test "Rich table creation" "python3 -c 'from rich.table import Table; Table()'" ""
run_test "Rich panel creation" "python3 -c 'from rich.panel import Panel; Panel(\"test\")'" ""
run_test "Rich progress creation" "python3 -c 'from rich.progress import Progress; Progress()'" ""

echo
echo "📊 Test Results Summary"
echo "======================"
echo -e "Total Tests: ${TESTS_TOTAL}"
echo -e "Passed: ${GREEN}${TESTS_PASSED}${NC}"
echo -e "Failed: ${RED}${TESTS_FAILED}${NC}"
echo -e "Success Rate: $(( (TESTS_PASSED * 100) / TESTS_TOTAL ))%"

echo
if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}🎉 All tests passed! Xencode Enhanced is ready to use!${NC}"
    echo
    echo "🚀 Quick Start:"
    echo "  • Chat mode: ./xencode.sh"
    echo "  • Inline mode: ./xencode.sh \"your prompt\""
    echo "  • Check status: ./xencode.sh --status"
    echo "  • View help: ./xencode.sh --help"
else
    echo -e "${YELLOW}⚠️  Some tests failed. Please check the errors above.${NC}"
    echo
    echo "🔧 Troubleshooting:"
    echo "  • Run: ./install.sh to fix dependencies"
    echo "  • Check: systemctl status ollama"
    echo "  • Verify: python3 -m pip list"
fi

echo
echo "📚 For more information, see:"
echo "  • ENHANCED_FEATURES.md - Complete feature guide"
echo "  • README.md - Basic usage and installation"
echo "  • INSTALL_MANUAL.md - Manual installation steps"

# Exit with error code if any tests failed
[ $TESTS_FAILED -eq 0 ] || exit 1
