"""
Connection pooling module for Xencode
Provides efficient connection management for API calls
"""
import asyncio
import threading
from collections import deque
from contextlib import contextmanager
from typing import Dict, Optional, Union
from urllib.parse import urlparse

import aiohttp
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry


class ConnectionPool:
    """
    Thread-safe connection pool for HTTP requests
    """
    def __init__(self, 
                 max_connections: int = 10,
                 max_retries: int = 3,
                 backoff_factor: float = 0.3,
                 pool_timeout: float = 30):
        self.max_connections = max_connections
        self.pool_timeout = pool_timeout
        self._pool = deque(maxlen=max_connections)
        self._lock = threading.Lock()
        
        # Create a session with retry strategy
        self.session = requests.Session()
        
        # Define retry strategy
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        
        # Mount adapter with retry strategy
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=max_connections,
            pool_maxsize=max_connections
        )
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
    
    def get_connection(self):
        """
        Get a connection from the pool (returns the session object)
        """
        return self.session
    
    def return_connection(self, conn):
        """
        Return a connection to the pool (not needed for requests.Session)
        """
        # requests.Session handles connection reuse internally
        pass
    
    def close(self):
        """
        Close all connections in the pool
        """
        self.session.close()


class AsyncConnectionPool:
    """
    Async connection pool for aiohttp
    """
    def __init__(self,
                 max_connections: int = 20,
                 max_keepalive_connections: int = 10,
                 keepalive_expiry: float = 30.0):
        self.max_connections = max_connections
        self.max_keepalive_connections = max_keepalive_connections
        self.keepalive_expiry = keepalive_expiry
        self._session: Optional[aiohttp.ClientSession] = None
        self._connector: Optional[aiohttp.TCPConnector] = None
    
    async def get_session(self) -> aiohttp.ClientSession:
        """
        Get an aiohttp session with connection pooling
        """
        if self._session is None or self._session.closed:
            # Create a new connector with connection pooling
            self._connector = aiohttp.TCPConnector(
                limit=self.max_connections,
                limit_per_host=self.max_connections // 2,
                keepalive_timeout=self.keepalive_expiry,
                enable_cleanup_closed=True
            )
            
            # Create timeout configuration
            timeout = aiohttp.ClientTimeout(total=30, connect=10)
            
            # Create the session
            self._session = aiohttp.ClientSession(
                connector=self._connector,
                timeout=timeout
            )
        
        return self._session
    
    async def close(self):
        """
        Close the connection pool
        """
        if self._session and not self._session.closed:
            await self._session.close()
        if self._connector:
            await self._connector.close()


class APIClient:
    """
    API client with connection pooling for Ollama and other services
    """
    def __init__(self,
                 base_url: str = "http://localhost:11434",
                 max_connections: int = 10,
                 max_retries: int = 3):
        self.base_url = base_url
        self.sync_pool = ConnectionPool(
            max_connections=max_connections,
            max_retries=max_retries
        )
        self.async_pool = AsyncConnectionPool(
            max_connections=max_connections * 2,
            max_keepalive_connections=max_connections
        )
        self._loop = None
    
    def sync_request(self, 
                     endpoint: str, 
                     method: str = "POST", 
                     json_data: Optional[Dict] = None,
                     timeout: int = 30) -> requests.Response:
        """
        Make a synchronous request using connection pooling
        """
        url = f"{self.base_url}{endpoint}"
        session = self.sync_pool.get_connection()
        
        try:
            response = session.request(
                method=method,
                url=url,
                json=json_data,
                timeout=timeout
            )
            return response
        finally:
            # Connection reuse is handled by requests internally
            pass
    
    async def async_request(self,
                           endpoint: str,
                           method: str = "POST",
                           json_data: Optional[Dict] = None,
                           timeout: int = 30) -> aiohttp.ClientResponse:
        """
        Make an asynchronous request using connection pooling
        """
        url = f"{self.base_url}{endpoint}"
        session = await self.async_pool.get_session()
        
        timeout_obj = aiohttp.ClientTimeout(total=timeout)
        
        async with session.request(
            method=method,
            url=url,
            json=json_data,
            timeout=timeout_obj
        ) as response:
            # Return response for caller to handle
            # We don't await response.json() here to allow caller flexibility
            return response
    
    async def generate(self, model: str, prompt: str, stream: bool = False) -> Union[Dict, str]:
        """
        Generate response from Ollama model
        """
        data = {
            "model": model,
            "prompt": prompt,
            "stream": stream
        }
        
        response = await self.async_request("/api/generate", json_data=data)
        
        if response.status == 200:
            json_response = await response.json()
            return json_response
        else:
            raise Exception(f"API request failed with status {response.status}")
    
    def list_models(self) -> Dict:
        """
        List available models
        """
        response = self.sync_request("/api/tags")
        response.raise_for_status()
        return response.json()
    
    async def close(self):
        """
        Close all connections
        """
        await self.async_pool.close()


# Global API client instance for reuse
_api_client: Optional[APIClient] = None


def get_api_client(base_url: str = "http://localhost:11434") -> APIClient:
    """
    Get or create a singleton API client instance
    """
    global _api_client
    if _api_client is None:
        _api_client = APIClient(base_url=base_url)
    return _api_client


async def close_api_client():
    """
    Close the global API client
    """
    global _api_client
    if _api_client:
        await _api_client.close()
        _api_client = None


# Example usage and testing
async def test_connection_pooling():
    """
    Test function to demonstrate connection pooling
    """
    import time
    
    # Create API client
    client = get_api_client()
    
    # Example: Make multiple requests
    start_time = time.time()
    
    # Synchronous requests
    for i in range(5):
        try:
            resp = client.list_models()
            print(f"Synchronous request {i+1}: {len(resp.get('models', []))} models")
        except Exception as e:
            print(f"Synchronous request {i+1} failed: {e}")
    
    # Asynchronous requests
    async def make_async_request(req_num):
        try:
            response = await client.generate("test-model", f"Test prompt {req_num}", stream=False)
            print(f"Asynchronous request {req_num}: Success")
            return response
        except Exception as e:
            print(f"Asynchronous request {req_num} failed: {e}")
            return None
    
    # Run multiple async requests concurrently
    tasks = [make_async_request(i) for i in range(5)]
    await asyncio.gather(*tasks)
    
    end_time = time.time()
    print(f"Completed all requests in {end_time - start_time:.2f} seconds")
    
    # Close the client
    await close_api_client()


if __name__ == "__main__":
    # Run the test
    asyncio.run(test_connection_pooling())