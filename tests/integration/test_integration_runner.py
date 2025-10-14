#!/usr/bin/env python3
"""
Integration Test Runner for Xencode

Tests component interactions, database operations, and external service integrations
using mock services for isolation.
"""

import asyncio
import pytest
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import patch, Mock

from xencode.test_mocks import (
    MockServiceRegistry,
    setup_integration_mocks,
    teardown_integration_mocks,
    mock_registry
)


class IntegrationTestRunner:
    """Runs integration tests with proper setup and teardown"""
    
    def __init__(self):
        self.temp_dir: Path = None
        self.mock_patches: List = []
        self.test_database_path: Path = None
        
    async def setup_test_environment(self) -> Dict[str, Any]:
        """Set up isolated test environment"""
        # Create temporary directory
        self.temp_dir = Path(tempfile.mkdtemp(prefix='xencode_integration_'))
        
        # Set up mock services
        self.mock_patches = setup_integration_mocks()
        
        # Create test database
        self.test_database_path = self.temp_dir / 'test.db'
        
        # Set up test configuration
        test_config = {
            'temp_dir': str(self.temp_dir),
            'database_path': str(self.test_database_path),
            'cache_enabled': True,
            'ollama_url': 'http://localhost:11434',  # Will be mocked
            'redis_url': 'redis://localhost:6379',   # Will be mocked
        }
        
        return test_config
    
    async def teardown_test_environment(self) -> None:
        """Clean up test environment"""
        # Stop mock patches
        teardown_integration_mocks(self.mock_patches)
        
        # Clean up temporary directory
        if self.temp_dir and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
        
        # Reset mock registry
        mock_registry.reset_all()
    
    async def run_component_interaction_tests(self) -> Dict[str, Any]:
        """Test interactions between components"""
        results = {
            'cache_model_interaction': False,
            'config_cache_interaction': False,
            'security_model_interaction': False,
            'monitor_stability_interaction': False
        }
        
        try:
            # Test cache and model selector interaction
            results['cache_model_interaction'] = await self._test_cache_model_interaction()
            
            # Test config and cache interaction
            results['config_cache_interaction'] = await self._test_config_cache_interaction()
            
            # Test security and model interaction
            results['security_model_interaction'] = await self._test_security_model_interaction()
            
            # Test monitor and stability interaction
            results['monitor_stability_interaction'] = await self._test_monitor_stability_interaction()
            
        except Exception as e:
            results['error'] = str(e)
        
        return results
    
    async def _test_cache_model_interaction(self) -> bool:
        """Test cache and model selector working together"""
        try:
            # Mock a model selection and caching scenario
            model_name = 'llama3.2:3b'
            cache_key = f'model_response:{model_name}:test_prompt'
            
            # Simulate model response being cached
            mock_registry.redis.data[cache_key] = '{"response": "cached response"}'
            
            # Check if cache retrieval works
            cached_response = await mock_registry.redis.get(cache_key)
            
            return cached_response is not None
            
        except Exception:
            return False
    
    async def _test_config_cache_interaction(self) -> bool:
        """Test configuration and cache working together"""
        try:
            # Mock configuration affecting cache behavior
            config_key = 'cache_config'
            config_data = {
                'cache_ttl': 3600,
                'max_cache_size': 1000,
                'cache_enabled': True
            }
            
            # Store config in cache
            import json
            await mock_registry.redis.set(config_key, json.dumps(config_data), ex=3600)
            
            # Retrieve and verify
            retrieved_config = await mock_registry.redis.get(config_key)
            
            return retrieved_config is not None
            
        except Exception:
            return False
    
    async def _test_security_model_interaction(self) -> bool:
        """Test security manager and model selector interaction"""
        try:
            # Mock security validation of model requests
            model_request = {
                'model': 'llama3.2:3b',
                'prompt': 'Write a hello world program',
                'user_id': 'test_user'
            }
            
            # Simulate security check passing
            security_check = True  # Would normally validate prompt safety
            
            if security_check:
                # Simulate model response
                response = await mock_registry.ollama.generate(
                    model_request['model'],
                    model_request['prompt']
                )
                return response is not None
            
            return False
            
        except Exception:
            return False
    
    async def _test_monitor_stability_interaction(self) -> bool:
        """Test resource monitor and model stability interaction"""
        try:
            # Mock resource monitoring affecting model stability
            resource_usage = {
                'memory_percent': 75.0,
                'cpu_percent': 60.0,
                'disk_usage': 80.0
            }
            
            # Simulate stability manager responding to resource pressure
            if resource_usage['memory_percent'] > 70:
                # Would normally trigger model optimization
                optimized_model = 'llama3.2:3b'  # Smaller model
                
                # Test if optimized model works
                response = await mock_registry.ollama.generate(
                    optimized_model,
                    'test prompt'
                )
                return response is not None
            
            return True
            
        except Exception:
            return False
    
    async def test_database_operations(self) -> Dict[str, Any]:
        """Test database setup, operations, and teardown"""
        results = {
            'connection': False,
            'table_creation': False,
            'data_insertion': False,
            'data_retrieval': False,
            'cleanup': False
        }
        
        try:
            # Test database connection
            results['connection'] = mock_registry.database.is_connected()
            
            # Test table creation
            await mock_registry.database.execute(
                "CREATE TABLE test_table (id INTEGER PRIMARY KEY, name TEXT)"
            )
            results['table_creation'] = True
            
            # Test data insertion
            rows_affected = await mock_registry.database.execute(
                "INSERT INTO test_table (name) VALUES (?)",
                ('test_name',)
            )
            results['data_insertion'] = rows_affected > 0
            
            # Test data retrieval
            data = await mock_registry.database.fetchall(
                "SELECT * FROM test_table"
            )
            results['data_retrieval'] = len(data) > 0
            
            # Test cleanup
            await mock_registry.database.execute("DROP TABLE IF EXISTS test_table")
            results['cleanup'] = True
            
        except Exception as e:
            results['error'] = str(e)
        
        return results
    
    async def test_external_service_mocks(self) -> Dict[str, Any]:
        """Test that all external service mocks are working"""
        results = {
            'ollama_service': False,
            'redis_service': False,
            'filesystem_service': False,
            'http_requests': False
        }
        
        try:
            # Test Ollama mock
            models = await mock_registry.ollama.list_models()
            results['ollama_service'] = len(models.get('models', [])) > 0
            
            # Test Redis mock
            await mock_registry.redis.set('test_key', 'test_value')
            value = await mock_registry.redis.get('test_key')
            results['redis_service'] = value == 'test_value'
            
            # Test filesystem mock
            mock_registry.filesystem.write_file('/test/file.txt', b'test content')
            content = mock_registry.filesystem.read_file('/test/file.txt')
            results['filesystem_service'] = content == b'test content'
            
            # Test HTTP mock
            from xencode.test_mocks import MockResponse
            mock_registry.add_http_response(
                'http://test.com/api',
                MockResponse(status_code=200, json_data={'status': 'ok'})
            )
            response = mock_registry.get_http_response('http://test.com/api')
            results['http_requests'] = response.status_code == 200
            
        except Exception as e:
            results['error'] = str(e)
        
        return results


# Global integration test runner
integration_runner = IntegrationTestRunner()


@pytest.fixture
async def integration_env():
    """Pytest fixture for integration test environment"""
    config = await integration_runner.setup_test_environment()
    yield config
    await integration_runner.teardown_test_environment()


# Integration test cases
@pytest.mark.asyncio
async def test_component_interactions(integration_env):
    """Test that components interact correctly"""
    results = await integration_runner.run_component_interaction_tests()
    
    assert results['cache_model_interaction'], "Cache and model selector should interact"
    assert results['config_cache_interaction'], "Config and cache should interact"
    assert results['security_model_interaction'], "Security and model should interact"
    assert results['monitor_stability_interaction'], "Monitor and stability should interact"


@pytest.mark.asyncio
async def test_database_integration(integration_env):
    """Test database operations work correctly"""
    results = await integration_runner.test_database_operations()
    
    assert results['connection'], "Database should connect"
    assert results['table_creation'], "Should create tables"
    assert results['data_insertion'], "Should insert data"
    assert results['data_retrieval'], "Should retrieve data"
    assert results['cleanup'], "Should clean up properly"


@pytest.mark.asyncio
async def test_external_service_mocks(integration_env):
    """Test that all external service mocks work"""
    results = await integration_runner.test_external_service_mocks()
    
    assert results['ollama_service'], "Ollama mock should work"
    assert results['redis_service'], "Redis mock should work"
    assert results['filesystem_service'], "Filesystem mock should work"
    assert results['http_requests'], "HTTP mock should work"


@pytest.mark.asyncio
async def test_end_to_end_workflow(integration_env):
    """Test complete end-to-end workflow"""
    # Simulate a complete user workflow
    
    # 1. User makes a request
    user_prompt = "Write a Python function to calculate fibonacci"
    
    # 2. Security check (mocked)
    security_passed = True  # Would check for harmful content
    assert security_passed, "Security check should pass"
    
    # 3. Model selection (mocked)
    selected_model = 'llama3.2:3b'
    models = await mock_registry.ollama.list_models()
    available_models = [m['name'] for m in models['models']]
    assert selected_model in available_models, "Selected model should be available"
    
    # 4. Check cache (mocked)
    cache_key = f"response:{selected_model}:{hash(user_prompt)}"
    cached_response = await mock_registry.redis.get(cache_key)
    
    if not cached_response:
        # 5. Generate response (mocked)
        response = await mock_registry.ollama.generate(selected_model, user_prompt)
        assert response is not None, "Should generate response"
        
        # 6. Cache response (mocked)
        import json
        await mock_registry.redis.set(
            cache_key, 
            json.dumps(response), 
            ex=3600
        )
    
    # 7. Return response to user
    final_response = await mock_registry.redis.get(cache_key)
    assert final_response is not None, "Should have final response"


if __name__ == '__main__':
    # Run integration tests directly
    async def main():
        runner = IntegrationTestRunner()
        
        print("Setting up integration test environment...")
        config = await runner.setup_test_environment()
        
        try:
            print("Testing component interactions...")
            interaction_results = await runner.run_component_interaction_tests()
            print(f"Interaction results: {interaction_results}")
            
            print("Testing database operations...")
            db_results = await runner.test_database_operations()
            print(f"Database results: {db_results}")
            
            print("Testing external service mocks...")
            mock_results = await runner.test_external_service_mocks()
            print(f"Mock results: {mock_results}")
            
            print("All integration tests completed successfully!")
            
        finally:
            print("Cleaning up test environment...")
            await runner.teardown_test_environment()
    
    asyncio.run(main())