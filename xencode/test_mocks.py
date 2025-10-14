#!/usr/bin/env python3
"""
Test Mocks for Xencode Integration Testing

Provides mock services for external dependencies like Ollama, Redis, 
and other services to enable isolated integration testing.
"""

import asyncio
import json
import time
from typing import Any, Dict, List, Optional, Union
from unittest.mock import Mock, AsyncMock
from dataclasses import dataclass, field


@dataclass
class MockResponse:
    """Mock HTTP response"""
    status_code: int = 200
    text: str = ""
    json_data: Optional[Dict[str, Any]] = None
    headers: Dict[str, str] = field(default_factory=dict)
    
    def json(self) -> Dict[str, Any]:
        return self.json_data or {}


class MockOllamaService:
    """Mock Ollama service for testing"""
    
    def __init__(self):
        self.models: List[str] = [
            'llama3.2:3b',
            'codellama:7b', 
            'mistral:7b',
            'qwen2.5-coder:7b'
        ]
        self.running_models: List[str] = []
        self.response_delay: float = 0.1  # Simulate network delay
        
    async def list_models(self) -> Dict[str, Any]:
        """Mock list models endpoint"""
        await asyncio.sleep(self.response_delay)
        return {
            'models': [
                {
                    'name': model,
                    'size': 4000000000,  # 4GB
                    'digest': f'sha256:mock_digest_{model.replace(":", "_")}',
                    'modified_at': '2024-01-01T00:00:00Z'
                }
                for model in self.models
            ]
        }
    
    async def generate(self, 
                      model: str, 
                      prompt: str, 
                      stream: bool = False,
                      **kwargs) -> Union[Dict[str, Any], AsyncMock]:
        """Mock generate endpoint"""
        await asyncio.sleep(self.response_delay)
        
        if model not in self.models:
            raise Exception(f"Model {model} not found")
        
        # Add to running models if not already there
        if model not in self.running_models:
            self.running_models.append(model)
        
        response_text = f"Mock response for prompt: {prompt[:50]}..."
        
        if stream:
            # Return async generator mock
            async def mock_stream():
                for chunk in response_text.split():
                    yield {
                        'model': model,
                        'response': chunk + ' ',
                        'done': False
                    }
                    await asyncio.sleep(0.01)
                
                yield {
                    'model': model,
                    'response': '',
                    'done': True,
                    'total_duration': 1000000000,  # 1 second in nanoseconds
                    'load_duration': 100000000,
                    'prompt_eval_count': len(prompt.split()),
                    'eval_count': len(response_text.split())
                }
            
            return mock_stream()
        else:
            return {
                'model': model,
                'response': response_text,
                'done': True,
                'total_duration': 1000000000,
                'load_duration': 100000000,
                'prompt_eval_count': len(prompt.split()),
                'eval_count': len(response_text.split())
            }
    
    async def pull_model(self, model: str) -> Dict[str, Any]:
        """Mock pull model endpoint"""
        await asyncio.sleep(self.response_delay * 10)  # Pulling takes longer
        
        if model not in self.models:
            self.models.append(model)
        
        return {
            'status': 'success',
            'digest': f'sha256:mock_digest_{model.replace(":", "_")}'
        }
    
    async def show_model(self, model: str) -> Dict[str, Any]:
        """Mock show model info endpoint"""
        await asyncio.sleep(self.response_delay)
        
        if model not in self.models:
            raise Exception(f"Model {model} not found")
        
        return {
            'license': 'Mock License',
            'modelfile': f'FROM {model}\nPARAMETER temperature 0.7',
            'parameters': 'temperature 0.7\ntop_p 0.9',
            'template': '{{ .Prompt }}',
            'details': {
                'format': 'gguf',
                'family': 'llama',
                'families': ['llama'],
                'parameter_size': '7B',
                'quantization_level': 'Q4_0'
            }
        }
    
    def is_running(self) -> bool:
        """Check if Ollama service is running"""
        return True  # Always running in tests
    
    def get_running_models(self) -> List[str]:
        """Get currently running models"""
        return self.running_models.copy()


class MockRedisService:
    """Mock Redis service for testing"""
    
    def __init__(self):
        self.data: Dict[str, Any] = {}
        self.expiry: Dict[str, float] = {}
        self.response_delay: float = 0.001  # Very fast for local cache
    
    async def get(self, key: str) -> Optional[str]:
        """Mock Redis GET"""
        await asyncio.sleep(self.response_delay)
        
        # Check if key has expired
        if key in self.expiry and time.time() > self.expiry[key]:
            del self.data[key]
            del self.expiry[key]
            return None
        
        return self.data.get(key)
    
    async def set(self, 
                 key: str, 
                 value: str, 
                 ex: Optional[int] = None) -> bool:
        """Mock Redis SET"""
        await asyncio.sleep(self.response_delay)
        
        self.data[key] = value
        
        if ex:
            self.expiry[key] = time.time() + ex
        
        return True
    
    async def delete(self, key: str) -> int:
        """Mock Redis DELETE"""
        await asyncio.sleep(self.response_delay)
        
        if key in self.data:
            del self.data[key]
            if key in self.expiry:
                del self.expiry[key]
            return 1
        return 0
    
    async def exists(self, key: str) -> int:
        """Mock Redis EXISTS"""
        await asyncio.sleep(self.response_delay)
        
        # Check expiry
        if key in self.expiry and time.time() > self.expiry[key]:
            del self.data[key]
            del self.expiry[key]
            return 0
        
        return 1 if key in self.data else 0
    
    async def flushall(self) -> bool:
        """Mock Redis FLUSHALL"""
        await asyncio.sleep(self.response_delay)
        self.data.clear()
        self.expiry.clear()
        return True
    
    async def keys(self, pattern: str = "*") -> List[str]:
        """Mock Redis KEYS"""
        await asyncio.sleep(self.response_delay)
        
        # Simple pattern matching (just * for now)
        if pattern == "*":
            return list(self.data.keys())
        else:
            # Basic pattern matching
            import fnmatch
            return [key for key in self.data.keys() if fnmatch.fnmatch(key, pattern)]
    
    def is_connected(self) -> bool:
        """Check if Redis is connected"""
        return True  # Always connected in tests


class MockFileSystem:
    """Mock file system for testing"""
    
    def __init__(self):
        self.files: Dict[str, bytes] = {}
        self.directories: set = set()
        
    def write_file(self, path: str, content: bytes) -> None:
        """Mock file write"""
        # Create parent directories
        import os
        parent = os.path.dirname(path)
        if parent:
            self.directories.add(parent)
        
        self.files[path] = content
    
    def read_file(self, path: str) -> bytes:
        """Mock file read"""
        if path not in self.files:
            raise FileNotFoundError(f"File not found: {path}")
        return self.files[path]
    
    def exists(self, path: str) -> bool:
        """Check if file exists"""
        return path in self.files or path in self.directories
    
    def delete_file(self, path: str) -> None:
        """Mock file deletion"""
        if path in self.files:
            del self.files[path]
    
    def list_files(self, directory: str = "") -> List[str]:
        """List files in directory"""
        if not directory:
            return list(self.files.keys())
        
        return [
            path for path in self.files.keys() 
            if path.startswith(directory + "/")
        ]


class MockDatabaseService:
    """Mock database service for testing"""
    
    def __init__(self):
        self.tables: Dict[str, List[Dict[str, Any]]] = {}
        self.connected = True
    
    async def execute(self, query: str, params: Optional[tuple] = None) -> int:
        """Mock SQL execution"""
        await asyncio.sleep(0.001)  # Simulate DB latency
        
        query_lower = query.lower().strip()
        
        if query_lower.startswith('create table'):
            # Extract table name
            table_name = query.split()[2]
            self.tables[table_name] = []
            return 0
        
        elif query_lower.startswith('insert into'):
            # Simple insert mock
            table_name = query.split()[2]
            if table_name not in self.tables:
                self.tables[table_name] = []
            
            # Mock row data
            row = {'id': len(self.tables[table_name]) + 1}
            if params:
                for i, param in enumerate(params):
                    row[f'col_{i}'] = param
            
            self.tables[table_name].append(row)
            return 1
        
        elif query_lower.startswith('select'):
            # Mock select - return number of rows found
            # For simplicity, return 1 if any tables exist
            return 1 if self.tables else 0
        
        return 0
    
    async def fetchall(self, query: str, params: Optional[tuple] = None) -> List[Dict[str, Any]]:
        """Mock fetch all results"""
        await asyncio.sleep(0.001)
        
        # Return mock data based on query
        if 'users' in query.lower():
            return [
                {'id': 1, 'username': 'test_user', 'email': 'test@example.com'},
                {'id': 2, 'username': 'admin', 'email': 'admin@example.com'}
            ]
        elif 'test_table' in query.lower():
            # Return data from our mock test table
            return [{'id': 1, 'name': 'test_name'}]
        
        # Return some data if any tables exist
        if self.tables:
            return [{'id': 1, 'mock_column': 'mock_value'}]
        
        return []
    
    async def fetchone(self, query: str, params: Optional[tuple] = None) -> Optional[Dict[str, Any]]:
        """Mock fetch one result"""
        results = await self.fetchall(query, params)
        return results[0] if results else None
    
    def is_connected(self) -> bool:
        """Check if database is connected"""
        return self.connected


class MockServiceRegistry:
    """Registry for all mock services"""
    
    def __init__(self):
        self.ollama = MockOllamaService()
        self.redis = MockRedisService()
        self.filesystem = MockFileSystem()
        self.database = MockDatabaseService()
        
        # HTTP mocks
        self.http_responses: Dict[str, MockResponse] = {}
        self.request_history: List[Dict[str, Any]] = []
    
    def add_http_response(self, 
                         url: str, 
                         response: MockResponse,
                         method: str = 'GET') -> None:
        """Add mock HTTP response"""
        key = f"{method}:{url}"
        self.http_responses[key] = response
    
    def get_http_response(self, url: str, method: str = 'GET') -> MockResponse:
        """Get mock HTTP response"""
        key = f"{method}:{url}"
        
        # Record request
        self.request_history.append({
            'method': method,
            'url': url,
            'timestamp': time.time()
        })
        
        return self.http_responses.get(key, MockResponse())
    
    def reset_all(self) -> None:
        """Reset all mock services"""
        self.ollama = MockOllamaService()
        self.redis = MockRedisService()
        self.filesystem = MockFileSystem()
        self.database = MockDatabaseService()
        self.http_responses.clear()
        self.request_history.clear()
    
    def get_request_count(self, url_pattern: str = None) -> int:
        """Get number of requests made"""
        if not url_pattern:
            return len(self.request_history)
        
        return len([
            req for req in self.request_history 
            if url_pattern in req['url']
        ])


# Global mock registry instance
mock_registry = MockServiceRegistry()


def setup_integration_mocks():
    """Set up all integration test mocks"""
    from unittest.mock import patch
    
    # Mock HTTP requests
    def mock_requests_get(url, **kwargs):
        return mock_registry.get_http_response(url, 'GET')
    
    def mock_requests_post(url, **kwargs):
        return mock_registry.get_http_response(url, 'POST')
    
    # Apply patches
    patches = [
        patch('requests.get', side_effect=mock_requests_get),
        patch('requests.post', side_effect=mock_requests_post),
        patch('aiohttp.ClientSession.get', side_effect=mock_requests_get),
        patch('aiohttp.ClientSession.post', side_effect=mock_requests_post),
    ]
    
    for p in patches:
        p.start()
    
    return patches


def teardown_integration_mocks(patches):
    """Tear down integration test mocks"""
    for patch_obj in patches:
        patch_obj.stop()
    
    mock_registry.reset_all()


# Convenience functions for common test scenarios
def create_mock_ollama_response(model: str, prompt: str, response: str) -> Dict[str, Any]:
    """Create a mock Ollama response"""
    return {
        'model': model,
        'response': response,
        'done': True,
        'total_duration': 1000000000,
        'load_duration': 100000000,
        'prompt_eval_count': len(prompt.split()),
        'eval_count': len(response.split())
    }


def create_mock_cache_data(key: str, value: Any, ttl: int = 3600) -> None:
    """Add data to mock Redis cache"""
    import json
    mock_registry.redis.data[key] = json.dumps(value)
    if ttl > 0:
        mock_registry.redis.expiry[key] = time.time() + ttl


def create_mock_file(path: str, content: str) -> None:
    """Create a mock file"""
    mock_registry.filesystem.write_file(path, content.encode('utf-8'))


def simulate_network_delay(delay: float = 0.1) -> None:
    """Simulate network delay for all mock services"""
    mock_registry.ollama.response_delay = delay
    mock_registry.redis.response_delay = delay / 10  # Redis is faster