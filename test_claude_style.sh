#!/bin/bash

# Test script for Claude-style functionality in xencode
# Tests streaming timing, formatting, prompt_toolkit integration, banner display, and error panels

# Don't exit on first error, we want to run all tests
set +e

# Colors for test output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test counters
TESTS_RUN=0
TESTS_PASSED=0
TESTS_FAILED=0

# Helper functions
print_test_header() {
    echo -e "\n${BLUE}=== $1 ===${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
    ((TESTS_PASSED++))
}

print_failure() {
    echo -e "${RED}✗ $1${NC}"
    ((TESTS_FAILED++))
}

run_test() {
    ((TESTS_RUN++))
    echo -e "${YELLOW}Running: $1${NC}"
}

# Test 1: Streaming timing configuration
print_test_header "Testing Streaming Timing Configuration"

run_test "Verify streaming timing constants are defined"
if python3 -c "
import sys
sys.path.append('.')
try:
    import xencode_core
    assert hasattr(xencode_core, 'THINKING_STREAM_DELAY')
    assert hasattr(xencode_core, 'ANSWER_STREAM_DELAY')
    assert hasattr(xencode_core, 'THINKING_TO_ANSWER_PAUSE')
    assert hasattr(xencode_core, 'THINKING_LINE_PAUSE')
    assert 0.040 <= xencode_core.THINKING_STREAM_DELAY <= 0.060
    assert 0.020 <= xencode_core.ANSWER_STREAM_DELAY <= 0.040
    assert xencode_core.THINKING_TO_ANSWER_PAUSE == 0.5
    assert 0.100 <= xencode_core.THINKING_LINE_PAUSE <= 0.150
    print('All timing constants are correctly configured')
except Exception as e:
    print(f'Error: {e}')
    sys.exit(1)
" 2>/dev/null; then
    print_success "Streaming timing constants are properly configured"
else
    print_failure "Streaming timing constants are missing or incorrect"
fi

# Test 2: Extract thinking and answer functionality
run_test "Test thinking and answer extraction"
if python3 -c "
import sys
sys.path.append('.')
try:
    import xencode_core

    # Test with <think> tags
    text1 = '<think>This is thinking</think>This is the answer'
    thinking1, answer1 = xencode_core.extract_thinking_and_answer(text1)
    assert thinking1 == 'This is thinking'
    assert answer1 == 'This is the answer'

    # Test with emoji format
    text2 = '🧠 Thinking: This is thinking\n\nThis is the answer'
    thinking2, answer2 = xencode_core.extract_thinking_and_answer(text2)
    assert thinking2 == 'This is thinking'
    assert answer2 == 'This is the answer'

    # Test with no thinking section
    text3 = 'Just an answer'
    thinking3, answer3 = xencode_core.extract_thinking_and_answer(text3)
    assert thinking3 == ''
    assert answer3 == 'Just an answer'

    print('All extraction tests passed')
except Exception as e:
    print(f'Error: {e}')
    sys.exit(1)
" 2>/dev/null; then
    print_success "Thinking and answer extraction works correctly"
else
    print_failure "Thinking and answer extraction failed"
fi

# Test 3: Banner display functionality
run_test "Test banner display function"
if python3 -c "
import sys
sys.path.append('.')
try:
    import xencode_core
    
    # Just test that the function exists and runs without error
    xencode_core.display_chat_banner('qwen3:4b', 'true')
    xencode_core.display_chat_banner('qwen3:4b', 'false')
    
    print('Banner display tests passed')
except Exception as e:
    print(f'Error: {e}')
    sys.exit(1)
" 2>/dev/null; then
    print_success "Banner display functionality works correctly"
else
    print_failure "Banner display functionality failed"
fi

# Test 4: Exit command detection
run_test "Test exit command detection"
if python3 -c "
import sys
sys.path.append('.')
import xencode_core

# Test various exit commands
assert xencode_core.is_exit_command('exit') == True
assert xencode_core.is_exit_command('quit') == True
assert xencode_core.is_exit_command('q') == True
assert xencode_core.is_exit_command('EXIT') == True
assert xencode_core.is_exit_command('  quit  ') == True
assert xencode_core.is_exit_command('hello') == False
assert xencode_core.is_exit_command('') == False

print('Exit command detection tests passed')
"; then
    print_success "Exit command detection works correctly"
else
    print_failure "Exit command detection failed"
fi

# Test 5: Online status checking
run_test "Test online status checking"
if python3 -c "
import sys
sys.path.append('.')
import xencode_core

# Test online status function (should return 'true' or 'false')
status = xencode_core.update_online_status()
assert status in ['true', 'false']
print(f'Online status check returned: {status}')
"; then
    print_success "Online status checking works correctly"
else
    print_failure "Online status checking failed"
fi

# Test 6: Error panel formatting
run_test "Test error panel formatting"
if python3 -c "
import sys
sys.path.append('.')
import xencode_core
from io import StringIO
from unittest.mock import patch
import subprocess

# Test connection error handling
with patch('requests.post') as mock_post:
    mock_post.side_effect = xencode_core.requests.exceptions.ConnectionError()
    with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
        try:
            xencode_core.run_query('test', 'test prompt')
        except SystemExit:
            pass  # Expected due to sys.exit(1)
        output = mock_stdout.getvalue()
        assert '❌ Cannot connect to Ollama service' in output
        assert 'Connection Error' in output

print('Error panel formatting tests passed')
"; then
    print_success "Error panel formatting works correctly"
else
    print_failure "Error panel formatting failed"
fi

# Test 7: prompt_toolkit integration
print_test_header "Testing prompt_toolkit Integration"

run_test "Test prompt_toolkit availability detection"
if python3 -c "
import sys
sys.path.append('.')
import xencode_core

# Check if PROMPT_TOOLKIT_AVAILABLE is properly set
print(f'prompt_toolkit available: {xencode_core.PROMPT_TOOLKIT_AVAILABLE}')
assert hasattr(xencode_core, 'PROMPT_TOOLKIT_AVAILABLE')
assert isinstance(xencode_core.PROMPT_TOOLKIT_AVAILABLE, bool)
"; then
    print_success "prompt_toolkit availability detection works"
else
    print_failure "prompt_toolkit availability detection failed"
fi

run_test "Test multiline input function exists"
if python3 -c "
import sys
sys.path.append('.')
import xencode_core

# Check if get_multiline_input function exists and is callable
assert hasattr(xencode_core, 'get_multiline_input')
assert callable(xencode_core.get_multiline_input)
print('get_multiline_input function is available')
"; then
    print_success "Multiline input function is properly defined"
else
    print_failure "Multiline input function is missing"
fi

# Test 8: Streaming functions
print_test_header "Testing Streaming Functions"

run_test "Test streaming functions exist and are callable"
if python3 -c "
import sys
sys.path.append('.')
import xencode_core

# Check streaming functions
assert hasattr(xencode_core, 'stream_thinking_section')
assert callable(xencode_core.stream_thinking_section)
assert hasattr(xencode_core, 'stream_answer_section')
assert callable(xencode_core.stream_answer_section)
assert hasattr(xencode_core, 'stream_claude_response')
assert callable(xencode_core.stream_claude_response)

print('All streaming functions are available')
"; then
    print_success "Streaming functions are properly defined"
else
    print_failure "Streaming functions are missing"
fi

run_test "Test format_output with streaming parameter"
if python3 -c "
import sys
sys.path.append('.')
import xencode_core
from io import StringIO
from unittest.mock import patch
import time

# Test format_output with streaming=True
test_text = '<think>Testing thinking</think>Testing answer'
start_time = time.time()

with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
    xencode_core.format_output(test_text, streaming=True)
    output = mock_stdout.getvalue()
    
end_time = time.time()
duration = end_time - start_time

# Verify streaming took some time (should be > 0.1s due to delays)
assert duration > 0.1, f'Streaming should take time, but took only {duration}s'
print(f'Streaming took {duration:.3f}s as expected')
"; then
    print_success "Streaming format_output works with timing"
else
    print_failure "Streaming format_output failed"
fi

# Test 9: Chat mode argument parsing
print_test_header "Testing Chat Mode Integration"

run_test "Test chat mode flag parsing"
if python3 -c "
import sys
sys.path.append('.')
import xencode_core

# Test that chat_mode function exists
assert hasattr(xencode_core, 'chat_mode')
assert callable(xencode_core.chat_mode)

print('Chat mode function is available')
"; then
    print_success "Chat mode function is properly defined"
else
    print_failure "Chat mode function is missing"
fi

# Test 10: Backward compatibility
print_test_header "Testing Backward Compatibility"

run_test "Test format_output without streaming (backward compatibility)"
if python3 -c "
import sys
sys.path.append('.')
try:
    import xencode_core
    
    # Test format_output without streaming parameter (default behavior)
    test_text = '<think>Testing thinking</think>Testing answer'
    
    # Just test that it runs without error
    xencode_core.format_output(test_text)  # No streaming parameter
    
    print('Backward compatibility maintained')
except Exception as e:
    print(f'Error: {e}')
    sys.exit(1)
" 2>/dev/null; then
    print_success "Backward compatibility is maintained"
else
    print_failure "Backward compatibility is broken"
fi

# Test 11: Model list and update functions with error panels
print_test_header "Testing Model Management with Error Panels"

run_test "Test model list function with error handling"
if python3 -c "
import sys
sys.path.append('.')
import xencode_core
from io import StringIO
from unittest.mock import patch
import subprocess

# Test list_models function exists
assert hasattr(xencode_core, 'list_models')
assert callable(xencode_core.list_models)

# Test error handling when ollama is not found
with patch('subprocess.check_output') as mock_subprocess:
    mock_subprocess.side_effect = FileNotFoundError()
    with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
        xencode_core.list_models()
        output = mock_stdout.getvalue()
        # Check for either the old error message or the new enhanced message
        assert any(msg in output for msg in ['❌ Ollama not found', '❌ No models found', 'Missing Dependency'])

print('Model list error handling works')
"; then
    print_success "Model list function with error panels works"
else
    print_failure "Model list function with error panels failed"
fi

run_test "Test model update function with error handling"
if python3 -c "
import sys
sys.path.append('.')
import xencode_core
from io import StringIO
from unittest.mock import patch
import subprocess

# Test update_model function exists
assert hasattr(xencode_core, 'update_model')
assert callable(xencode_core.update_model)

# Test error handling when ollama is not found
with patch('subprocess.run') as mock_subprocess:
    mock_subprocess.side_effect = FileNotFoundError()
    with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
        xencode_core.update_model('test-model')
        output = mock_stdout.getvalue()
        # Check for either the old error message or the new enhanced message
        assert any(msg in output for msg in ['❌ Ollama not found', 'Missing Dependency'])

print('Model update error handling works')
"; then
    print_success "Model update function with error panels works"
else
    print_failure "Model update function with error panels failed"
fi

# Test Summary
print_test_header "Test Summary"
echo -e "\n${BLUE}Test Results:${NC}"
echo -e "Tests Run: ${TESTS_RUN}"
echo -e "${GREEN}Tests Passed: ${TESTS_PASSED}${NC}"
echo -e "${RED}Tests Failed: ${TESTS_FAILED}${NC}"

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "\n${GREEN}🎉 All Claude-style functionality tests passed!${NC}"
    exit 0
else
    echo -e "\n${RED}❌ Some tests failed. Please review the output above.${NC}"
    exit 1
fi