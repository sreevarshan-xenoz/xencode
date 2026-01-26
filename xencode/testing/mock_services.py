"""
Mock objects for external services used in Xencode
Provides mock implementations for testing without external dependencies
"""
from unittest.mock import Mock, MagicMock
from typing import Any, Dict, List, Optional, Union
import json
import time
from datetime import datetime


class MockOllamaService:
    """Mock implementation of Ollama service for testing"""
    
    def __init__(self):
        self.models = ["llama2:7b", "mistral:7b", "gemma:2b"]
        self.responses = {}
        self.health_status = True
        self.api_call_log = []
    
    def set_mock_response(self, model: str, prompt: str, response: str):
        """Set a mock response for a specific model and prompt
        
        Args:
            model: Model name
            prompt: Input prompt
            response: Expected response
        """
        key = f"{model}:{prompt}"
        self.responses[key] = response
    
    def generate(self, model: str, prompt: str, stream: bool = False):
        """Mock implementation of the generate API call"""
        # Log the API call
        call_info = {
            'endpoint': '/api/generate',
            'method': 'POST',
            'params': {'model': model, 'prompt': prompt, 'stream': stream},
            'timestamp': time.time()
        }
        self.api_call_log.append(call_info)
        
        # Create a mock response
        key = f"{model}:{prompt}"
        response_text = self.responses.get(key, f"Mock response for: {prompt}")
        
        if stream:
            # For streaming, return a mock generator
            def mock_stream():
                yield {"response": response_text, "done": False}
                yield {"response": "", "done": True}
            return mock_stream()
        else:
            return {
                "model": model,
                "created_at": datetime.now().isoformat(),
                "response": response_text,
                "done": True,
                "total_duration": 123456789,
                "load_duration": 12345678,
                "prompt_eval_count": 10,
                "prompt_eval_duration": 12345678,
                "eval_count": 5,
                "eval_duration": 87654321
            }
    
    def list_models(self):
        """Mock implementation of the list models API call"""
        # Log the API call
        call_info = {
            'endpoint': '/api/tags',
            'method': 'GET',
            'timestamp': time.time()
        }
        self.api_call_log.append(call_info)
        
        return {
            "models": [
                {
                    "name": model,
                    "model": model,
                    "modified_at": datetime.now().isoformat(),
                    "size": 1024 * 1024 * 1024,  # 1GB
                    "digest": "sha256:abc123"
                } for model in self.models
            ]
        }
    
    def check_health(self):
        """Check if the mock service is healthy"""
        return self.health_status
    
    def reset_api_log(self):
        """Reset the API call log"""
        self.api_call_log = []


class MockOpenAIService:
    """Mock implementation of OpenAI service for testing"""
    
    def __init__(self):
        self.responses = {}
        self.completions = []
        self.api_call_log = []
    
    def set_mock_response(self, model: str, prompt: str, response: str):
        """Set a mock response for a specific model and prompt
        
        Args:
            model: Model name
            prompt: Input prompt
            response: Expected response
        """
        key = f"{model}:{prompt}"
        self.responses[key] = response
    
    def chat_completions_create(self, model: str, messages: List[Dict], **kwargs):
        """Mock implementation of chat completions create"""
        # Log the API call
        call_info = {
            'endpoint': 'chat/completions',
            'method': 'POST',
            'params': {'model': model, 'messages': messages, **kwargs},
            'timestamp': time.time()
        }
        self.api_call_log.append(call_info)
        
        # Get the prompt from messages
        prompt = " ".join([msg.get('content', '') for msg in messages])
        
        # Create a mock response
        key = f"{model}:{prompt}"
        response_text = self.responses.get(key, f"Mock OpenAI response for: {prompt}")
        
        # Store the completion for later inspection
        completion = {
            'id': f'cmpl-{int(time.time())}',
            'object': 'chat.completion',
            'created': int(time.time()),
            'model': model,
            'choices': [{
                'index': 0,
                'message': {
                    'role': 'assistant',
                    'content': response_text
                },
                'finish_reason': 'stop'
            }],
            'usage': {
                'prompt_tokens': 10,
                'completion_tokens': len(response_text.split()),
                'total_tokens': 10 + len(response_text.split())
            }
        }
        
        self.completions.append(completion)
        return MagicMock(**completion)
    
    def models_list(self):
        """Mock implementation of models list"""
        call_info = {
            'endpoint': 'models/list',
            'method': 'GET',
            'timestamp': time.time()
        }
        self.api_call_log.append(call_info)
        
        return MagicMock(data=[
            MagicMock(id="gpt-4", created=int(time.time()), owned_by="openai"),
            MagicMock(id="gpt-3.5-turbo", created=int(time.time()), owned_by="openai"),
        ])
    
    def reset_api_log(self):
        """Reset the API call log"""
        self.api_call_log = []


class MockGoogleGenAIService:
    """Mock implementation of Google Generative AI service for testing"""
    
    def __init__(self):
        self.responses = {}
        self.generations = []
        self.api_call_log = []
    
    def set_mock_response(self, model: str, prompt: str, response: str):
        """Set a mock response for a specific model and prompt
        
        Args:
            model: Model name
            prompt: Input prompt
            response: Expected response
        """
        key = f"{model}:{prompt}"
        self.responses[key] = response
    
    def generate_content(self, contents: Union[str, List]):
        """Mock implementation of generate content"""
        # Log the API call
        call_info = {
            'endpoint': 'generateContent',
            'method': 'POST',
            'params': {'contents': contents},
            'timestamp': time.time()
        }
        self.api_call_log.append(call_info)
        
        # Get the prompt from contents
        if isinstance(contents, str):
            prompt = contents
        elif isinstance(contents, list):
            prompt = " ".join([item if isinstance(item, str) else str(item) for item in contents])
        else:
            prompt = str(contents)
        
        # Create a mock response
        response_text = self.responses.get(f"gemini-pro:{prompt}", f"Mock Gemini response for: {prompt}")
        
        # Create mock response object
        response_obj = MagicMock()
        response_obj.text = response_text
        response_obj.candidates = [MagicMock()]
        response_obj.candidates[0].content = MagicMock(parts=[MagicMock(text=response_text)])
        
        self.generations.append({
            'input': contents,
            'output': response_text,
            'timestamp': time.time()
        })
        
        return response_obj
    
    def reset_api_log(self):
        """Reset the API call log"""
        self.api_call_log = []


class MockHTTPClient:
    """Mock HTTP client for testing network requests"""
    
    def __init__(self):
        self.responses = {}
        self.request_log = []
    
    def set_mock_response(self, url: str, method: str = "GET", response: Union[Dict, str] = None, status_code: int = 200):
        """Set a mock response for a specific URL and method
        
        Args:
            url: URL to mock
            method: HTTP method to mock
            response: Response to return
            status_code: Status code to return
        """
        key = f"{method}:{url}"
        self.responses[key] = {
            'response': response or {},
            'status_code': status_code
        }
    
    def get(self, url: str, **kwargs):
        """Mock GET request"""
        return self._make_request("GET", url, **kwargs)
    
    def post(self, url: str, **kwargs):
        """Mock POST request"""
        return self._make_request("POST", url, **kwargs)
    
    def put(self, url: str, **kwargs):
        """Mock PUT request"""
        return self._make_request("PUT", url, **kwargs)
    
    def delete(self, url: str, **kwargs):
        """Mock DELETE request"""
        return self._make_request("DELETE", url, **kwargs)
    
    def _make_request(self, method: str, url: str, **kwargs):
        """Internal method to handle requests"""
        # Log the request
        request_info = {
            'method': method,
            'url': url,
            'params': kwargs.get('params'),
            'json': kwargs.get('json'),
            'headers': kwargs.get('headers'),
            'timestamp': time.time()
        }
        self.request_log.append(request_info)
        
        # Get the mock response
        key = f"{method}:{url}"
        mock_resp = self.responses.get(key, {
            'response': {'message': f'Mock {method} response for {url}'},
            'status_code': 200
        })
        
        # Create a mock response object
        response = MagicMock()
        response.status_code = mock_resp['status_code']
        response.json.return_value = mock_resp['response']
        response.text = json.dumps(mock_resp['response'])
        response.raise_for_status = Mock() if mock_resp['status_code'] < 400 else Mock(side_effect=Exception(f"HTTP {mock_resp['status_code']}"))
        
        return response
    
    def reset_log(self):
        """Reset the request log"""
        self.request_log = []


class MockFileSystem:
    """Mock file system for testing file operations"""
    
    def __init__(self):
        self.files = {}
        self.directories = set()
        self.operations_log = []
    
    def set_file_content(self, path: str, content: str):
        """Set content for a mock file
        
        Args:
            path: Path of the file
            content: Content to set
        """
        self.files[path] = content
        # Automatically create parent directories
        parts = path.split('/')
        current_path = ''
        for part in parts[:-1]:
            current_path += part + '/'
            self.directories.add(current_path)
    
    def exists(self, path: str) -> bool:
        """Check if a file or directory exists"""
        self.operations_log.append({'operation': 'exists', 'path': path, 'timestamp': time.time()})
        return path in self.files or path in self.directories
    
    def read_file(self, path: str) -> str:
        """Read content from a mock file"""
        self.operations_log.append({'operation': 'read', 'path': path, 'timestamp': time.time()})
        if path not in self.files:
            raise FileNotFoundError(f"Mock file not found: {path}")
        return self.files[path]
    
    def write_file(self, path: str, content: str):
        """Write content to a mock file"""
        self.operations_log.append({'operation': 'write', 'path': path, 'content_len': len(content), 'timestamp': time.time()})
        self.set_file_content(path, content)
    
    def delete_file(self, path: str):
        """Delete a mock file"""
        self.operations_log.append({'operation': 'delete', 'path': path, 'timestamp': time.time()})
        if path in self.files:
            del self.files[path]
    
    def listdir(self, path: str) -> List[str]:
        """List contents of a mock directory"""
        self.operations_log.append({'operation': 'listdir', 'path': path, 'timestamp': time.time()})
        # Find all files and directories that start with the path
        contents = set()
        for file_path in self.files:
            if file_path.startswith(path) and '/' in file_path[len(path):]:
                relative_path = file_path[len(path):].split('/')[1]  # Get first level
                if relative_path:
                    contents.add(relative_path)
        for dir_path in self.directories:
            if dir_path.startswith(path) and dir_path != path:
                relative_path = dir_path[len(path):].split('/')[1] if '/' in dir_path[len(path):] else dir_path[len(path):]
                if relative_path:
                    contents.add(relative_path)
        return list(contents)
    
    def reset_log(self):
        """Reset the operations log"""
        self.operations_log = []


class MockSubprocess:
    """Mock subprocess module for testing command execution"""
    
    def __init__(self):
        self.command_outputs = {}
        self.command_logs = []
    
    def set_command_output(self, command: str, output: str, returncode: int = 0):
        """Set the output for a specific command
        
        Args:
            command: Command string to mock
            output: Output to return
            returncode: Return code to return
        """
        self.command_outputs[command] = {
            'stdout': output,
            'stderr': '',
            'returncode': returncode
        }
    
    def run(self, args, **kwargs):
        """Mock subprocess.run"""
        command = ' '.join(args) if isinstance(args, list) else str(args)
        self.command_logs.append({
            'command': command,
            'args': args,
            'kwargs': kwargs,
            'timestamp': time.time()
        })
        
        result = self.command_outputs.get(command, {
            'stdout': f'Mock output for: {command}',
            'stderr': '',
            'returncode': 0
        })
        
        completed_process = MagicMock()
        completed_process.stdout = result['stdout']
        completed_process.stderr = result['stderr']
        completed_process.returncode = result['returncode']
        completed_process.check_returncode = Mock() if result['returncode'] == 0 else Mock(side_effect=subprocess.CalledProcessError(result['returncode'], command))
        
        return completed_process
    
    def check_output(self, args, **kwargs):
        """Mock subprocess.check_output"""
        result = self.run(args, **kwargs)
        if result.returncode != 0:
            raise subprocess.CalledProcessError(result.returncode, ' '.join(args) if isinstance(args, list) else str(args))
        return result.stdout
    
    def reset_log(self):
        """Reset the command logs"""
        self.command_logs = []


# Global mock service instances
mock_ollama_service = MockOllamaService()
mock_openai_service = MockOpenAIService()
mock_google_genai_service = MockGoogleGenAIService()
mock_http_client = MockHTTPClient()
mock_filesystem = MockFileSystem()
mock_subprocess = MockSubprocess()


def get_mock_ollama_service() -> MockOllamaService:
    """Get the global mock Ollama service instance"""
    return mock_ollama_service


def get_mock_openai_service() -> MockOpenAIService:
    """Get the global mock OpenAI service instance"""
    return mock_openai_service


def get_mock_google_genai_service() -> MockGoogleGenAIService:
    """Get the global mock Google GenAI service instance"""
    return mock_google_genai_service


def get_mock_http_client() -> MockHTTPClient:
    """Get the global mock HTTP client instance"""
    return mock_http_client


def get_mock_filesystem() -> MockFileSystem:
    """Get the global mock filesystem instance"""
    return mock_filesystem


def get_mock_subprocess() -> MockSubprocess:
    """Get the global mock subprocess instance"""
    return mock_subprocess


# For compatibility with the actual subprocess module
try:
    import subprocess
except ImportError:
    # If subprocess isn't available, use our mock
    subprocess = mock_subprocess