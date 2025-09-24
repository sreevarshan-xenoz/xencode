"""Test fixtures and utilities for Xencode tests."""

import tempfile
from pathlib import Path
from unittest.mock import Mock

import pytest


@pytest.fixture
def mock_ollama_response():
    """Mock successful Ollama API response."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "response": "This is a test response from the AI model.",
        "done": True,
        "context": [1, 2, 3, 4],
        "total_duration": 1000000000,
        "load_duration": 500000000,
        "prompt_eval_count": 10,
        "eval_count": 20,
    }
    mock_response.iter_lines.return_value = [
        b'{"response": "This is ", "done": false}',
        b'{"response": "a test ", "done": false}',
        b'{"response": "response.", "done": true}',
    ]
    return mock_response


@pytest.fixture
def mock_ollama_error():
    """Mock Ollama API error response."""
    mock_response = Mock()
    mock_response.status_code = 500
    mock_response.json.return_value = {"error": "Internal server error"}
    mock_response.raise_for_status.side_effect = Exception("Server error")
    return mock_response


@pytest.fixture
def temp_config_dir():
    """Create temporary configuration directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = Path(temp_dir) / ".xencode"
        config_path.mkdir()
        yield config_path


@pytest.fixture
def mock_console():
    """Mock Rich console for testing."""
    mock = Mock()
    mock.print = Mock()
    mock.input = Mock(return_value="test input")
    return mock


@pytest.fixture
def sample_conversation():
    """Sample conversation data for testing."""
    return {
        "id": "test_conversation_1",
        "created": "2024-01-01T00:00:00",
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there! How can I help you today?"},
            {"role": "user", "content": "What is Python?"},
            {"role": "assistant", "content": "Python is a programming language..."},
        ],
    }


@pytest.fixture
def mock_requests_session():
    """Mock requests session."""
    session = Mock()
    session.post = Mock()
    session.get = Mock()
    return session


class MockOllamaServer:
    """Mock Ollama server for testing."""

    def __init__(self, working=True):
        self.working = working
        self.models = ["qwen3:4b", "llama2:7b", "mistral:7b"]

    def is_running(self):
        return self.working

    def list_models(self):
        if not self.working:
            raise Exception("Ollama server not running")
        return {"models": [{"name": model} for model in self.models]}

    def generate_response(self, prompt, model="qwen3:4b"):
        if not self.working:
            raise Exception("Ollama server not running")
        return f"Mock response to: {prompt}"


@pytest.fixture
def mock_ollama_server():
    """Mock Ollama server fixture."""
    return MockOllamaServer()


@pytest.fixture
def mock_broken_ollama_server():
    """Mock broken Ollama server fixture."""
    return MockOllamaServer(working=False)
