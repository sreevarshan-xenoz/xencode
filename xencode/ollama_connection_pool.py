#!/usr/bin/env python3
"""
Ollama Connection Pool for Xencode

Implements connection pooling for Ollama API calls to improve performance
and reduce connection overhead.
"""

import asyncio
import time
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
import ollama
from rich.console import Console

console = Console()


@dataclass
class ConnectionInfo:
    """Information about a pooled connection"""
    client: ollama.AsyncClient
    last_used: float
    is_busy: bool = False
    generation_count: int = 0


class OllamaConnectionPool:
    """Connection pool for Ollama clients to reduce overhead"""
    
    def __init__(self, max_connections: int = 10, timeout: int = 30):
        self.max_connections = max_connections
        self.timeout = timeout
        self.connections: asyncio.Queue = asyncio.Queue()
        self.active_connections: Dict[str, ConnectionInfo] = {}
        self._lock = asyncio.Lock()
        self._initialized = False

    async def initialize(self):
        """Initialize the connection pool with clients"""
        if self._initialized:
            return
            
        async with self._lock:
            if self._initialized:  # Double-check locking
                return
                
            # Create initial connections
            for i in range(min(3, self.max_connections)):  # Start with 3 connections
                client = ollama.AsyncClient()
                conn_info = ConnectionInfo(
                    client=client,
                    last_used=time.time(),
                    is_busy=False,
                    generation_count=0
                )
                await self.connections.put(conn_info)
            
            self._initialized = True
            console.print(f"[green]✅ Ollama connection pool initialized with {min(3, self.max_connections)} connections[/green]")

    @asynccontextmanager
    async def get_connection(self):
        """Get a connection from the pool"""
        if not self._initialized:
            await self.initialize()
            
        conn_info: Optional[ConnectionInfo] = None
        start_time = time.time()
        
        # Try to get an available connection
        while time.time() - start_time < self.timeout:
            try:
                # Try to get from queue first
                conn_info = await asyncio.wait_for(self.connections.get(), timeout=0.1)
                if not conn_info.is_busy:
                    break
                else:
                    # Put it back if busy and continue looking
                    await self.connections.put(conn_info)
                    conn_info = None
            except asyncio.TimeoutError:
                # No available connections, try to create a new one if possible
                if len(self.active_connections) < self.max_connections:
                    client = ollama.AsyncClient()
                    conn_info = ConnectionInfo(
                        client=client,
                        last_used=time.time(),
                        is_busy=True,
                        generation_count=0
                    )
                    conn_id = f"conn_{len(self.active_connections)}"
                    self.active_connections[conn_id] = conn_info
                    break
                else:
                    # Wait a bit before trying again
                    await asyncio.sleep(0.01)
                    continue
        
        if conn_info is None:
            raise TimeoutError(f"Could not acquire Ollama connection within {self.timeout} seconds")
        
        # Mark as busy
        conn_info.is_busy = True
        conn_info.last_used = time.time()
        conn_info.generation_count += 1
        
        try:
            yield conn_info.client
        finally:
            # Release the connection back to the pool
            conn_info.is_busy = False
            conn_info.last_used = time.time()
            
            # If we created this connection dynamically, keep track of it
            if hasattr(conn_info, 'dynamic') and conn_info.dynamic:
                # For now, just keep it in the active connections
                pass
            else:
                # Put it back in the queue for reuse
                try:
                    await self.connections.put(conn_info)
                except asyncio.QueueFull:
                    # Queue is full, close this connection
                    pass

    async def generate(self, model: str, prompt: str, **options) -> Dict[str, Any]:
        """Generate response using a pooled connection"""
        async with self.get_connection() as client:
            try:
                response = await client.generate(
                    model=model,
                    prompt=prompt,
                    options=options,
                    stream=False
                )
                return response
            except Exception as e:
                console.print(f"[red]Error during generation: {e}[/red]")
                raise

    async def embed(self, model: str, input_text: str) -> Dict[str, Any]:
        """Create embeddings using a pooled connection"""
        async with self.get_connection() as client:
            try:
                response = await client.embed(
                    model=model,
                    input=input_text
                )
                return response
            except Exception as e:
                console.print(f"[red]Error during embedding: {e}[/red]")
                raise

    async def close_all(self):
        """Close all connections in the pool"""
        # Close all connections in the queue
        while not self.connections.empty():
            try:
                conn_info = self.connections.get_nowait()
                # No direct way to close ollama client, so just let it go out of scope
            except asyncio.QueueEmpty:
                break
        
        # Close all active connections
        for conn_info in self.active_connections.values():
            # No direct way to close ollama client, so just clear the dict
            pass
        
        self.active_connections.clear()
        self._initialized = False


# Global connection pool instance
_ollama_pool: Optional[OllamaConnectionPool] = None


async def get_ollama_pool(max_connections: int = 10) -> OllamaConnectionPool:
    """Get or create the global Ollama connection pool"""
    global _ollama_pool
    if _ollama_pool is None:
        _ollama_pool = OllamaConnectionPool(max_connections=max_connections)
        await _ollama_pool.initialize()
    return _ollama_pool


async def pooled_generate(model: str, prompt: str, **options) -> Dict[str, Any]:
    """Generate response using the connection pool"""
    pool = await get_ollama_pool()
    return await pool.generate(model, prompt, **options)


async def pooled_embed(model: str, input_text: str) -> Dict[str, Any]:
    """Create embeddings using the connection pool"""
    pool = await get_ollama_pool()
    return await pool.embed(model, input_text)


# Example usage
if __name__ == "__main__":
    async def demo():
        console.print("[bold blue]🚀 Ollama Connection Pool Demo[/bold blue]")
        
        # Test the connection pool
        try:
            response = await pooled_generate(
                model="llama3.1:8b",
                prompt="Say hello in Spanish",
                options={"temperature": 0.7}
            )
            console.print(f"[green]✅ Response: {response.get('response', 'No response')}[/green]")
        except Exception as e:
            console.print(f"[red]❌ Error: {e}[/red]")
            console.print("[yellow]Note: This demo requires Ollama to be running with a model[/yellow]")

    # Don't run the demo by default since it requires Ollama to be running
    # asyncio.run(demo())