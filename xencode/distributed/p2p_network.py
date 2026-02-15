"""
Peer-to-Peer Agent Network
Implements P2PNetworkManager for agent communication, distributed hash table for agent discovery,
encrypted communication channels, and network topology optimization.
"""

import asyncio
import logging
import hashlib
import json
import socket
import ssl
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple, Callable
from enum import Enum
from datetime import datetime
import secrets
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import serialization
import aiohttp
from aiohttp import web
import threading
import time


logger = logging.getLogger(__name__)


class NodeType(Enum):
    """Types of nodes in the P2P network."""
    AGENT_NODE = "agent_node"
    SUPER_NODE = "super_node"  # Node that helps with discovery
    BOOTSTRAP_NODE = "bootstrap_node"  # Initial connection point


class MessagePriority(Enum):
    """Priority levels for network messages."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class NodeInfo:
    """Information about a node in the P2P network."""
    node_id: str
    ip_address: str
    port: int
    node_type: NodeType
    public_key: str
    last_seen: datetime
    reputation_score: float
    capabilities: List[str]


@dataclass
class NetworkMessage:
    """A message in the P2P network."""
    message_id: str
    sender_id: str
    recipient_id: str
    message_type: str
    content: str
    timestamp: datetime
    priority: MessagePriority
    signature: str
    encrypted_content: Optional[str] = None


class DHTNode:
    """Node in the distributed hash table for agent discovery."""
    
    def __init__(self, node_id: str, ip: str, port: int):
        self.node_id = node_id
        self.ip = ip
        self.port = port
        self.data = {}  # key -> value storage
        self.neighbors = {}  # node_id -> NodeInfo
        self.bucket_size = 20  # Kademlia bucket size
        
    def put(self, key: str, value: Any):
        """Store a key-value pair in the DHT."""
        self.data[key] = value
        logger.debug(f"DHT Put: {key} -> {value}")
        
    def get(self, key: str) -> Optional[Any]:
        """Retrieve a value from the DHT."""
        return self.data.get(key)
        
    def find_node(self, target_id: str) -> List[NodeInfo]:
        """Find nodes closest to the target ID."""
        # Calculate distances to all known nodes
        distances = []
        for node_id, node_info in self.neighbors.items():
            distance = self._xor_distance(node_id, target_id)
            distances.append((distance, node_info))
            
        # Return k closest nodes
        distances.sort(key=lambda x: x[0])
        return [info for _, info in distances[:self.bucket_size]]
        
    def _xor_distance(self, id1: str, id2: str) -> int:
        """Calculate XOR distance between two node IDs."""
        # Convert hex IDs to integers for XOR operation
        int1 = int(id1, 16) if len(id1) == 16 else int(hashlib.sha256(id1.encode()).hexdigest()[:16], 16)
        int2 = int(id2, 16) if len(id2) == 16 else int(hashlib.sha256(id2.encode()).hexdigest()[:16], 16)
        return int1 ^ int2


class EncryptionManager:
    """Manages encryption for secure communication."""
    
    def __init__(self):
        self.cipher_suite_map: Dict[str, Fernet] = {}  # node_id -> cipher_suite
        self.public_keys: Dict[str, str] = {}  # node_id -> public_key
        self.private_key = None
        self._generate_local_keys()
        
    def _generate_local_keys(self):
        """Generate local RSA key pair."""
        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        self.public_key = self.private_key.public_key()
        
    def add_peer_public_key(self, node_id: str, public_key_pem: str):
        """Add a peer's public key to enable encrypted communication."""
        # Store the public key
        self.public_keys[node_id] = public_key_pem
        
        # Generate a symmetric key for this peer and encrypt it with their public key
        symmetric_key = Fernet.generate_key()
        cipher_suite = Fernet(symmetric_key)
        
        # Encrypt the symmetric key with the peer's public key
        public_key = serialization.load_pem_public_key(public_key_pem.encode())
        encrypted_symmetric_key = public_key.encrypt(
            symmetric_key,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        
        # Store the cipher suite for this node
        self.cipher_suite_map[node_id] = cipher_suite
        
    def encrypt_message(self, message: str, recipient_id: str) -> str:
        """Encrypt a message for a specific recipient."""
        if recipient_id not in self.cipher_suite_map:
            raise ValueError(f"No encryption key available for node {recipient_id}")
            
        cipher_suite = self.cipher_suite_map[recipient_id]
        encrypted_bytes = cipher_suite.encrypt(message.encode())
        return base64.b64encode(encrypted_bytes).decode()
        
    def decrypt_message(self, encrypted_message: str, sender_id: str) -> str:
        """Decrypt a message from a specific sender."""
        if sender_id not in self.cipher_suite_map:
            raise ValueError(f"No decryption key available for node {sender_id}")
            
        cipher_suite = self.cipher_suite_map[sender_id]
        encrypted_bytes = base64.b64decode(encrypted_message.encode())
        decrypted_bytes = cipher_suite.decrypt(encrypted_bytes)
        return decrypted_bytes.decode()
        
    def sign_message(self, message: str) -> str:
        """Sign a message with the local private key."""
        signature = self.private_key.sign(
            message.encode(),
            padding.PKCS1v15(),
            hashes.SHA256()
        )
        return base64.b64encode(signature).decode()
        
    def verify_signature(self, message: str, signature: str, public_key_pem: str) -> bool:
        """Verify a message signature with the sender's public key."""
        public_key = serialization.load_pem_public_key(public_key_pem.encode())
        signature_bytes = base64.b64decode(signature.encode())
        
        try:
            public_key.verify(
                signature_bytes,
                message.encode(),
                padding.PKCS1v15(),
                hashes.SHA256()
            )
            return True
        except Exception:
            return False


class NetworkTopologyOptimizer:
    """Optimizes the network topology for efficient communication."""
    
    def __init__(self):
        self.node_connections: Dict[str, List[str]] = {}  # node_id -> [connected_node_ids]
        self.connection_quality: Dict[Tuple[str, str], float] = {}  # (node1, node2) -> quality_score
        self.preferred_paths: Dict[str, List[str]] = {}  # source -> [path_to_destination]
        
    def update_connection_quality(self, node1_id: str, node2_id: str, quality_score: float):
        """Update the quality score for a connection between two nodes."""
        # Ensure symmetric storage
        self.connection_quality[(node1_id, node2_id)] = quality_score
        self.connection_quality[(node2_id, node1_id)] = quality_score
        
        # Update connection lists
        if node1_id not in self.node_connections:
            self.node_connections[node1_id] = []
        if node2_id not in self.node_connections[node1_id]:
            self.node_connections[node1_id].append(node2_id)
            
        if node2_id not in self.node_connections:
            self.node_connections[node2_id] = []
        if node1_id not in self.node_connections[node2_id]:
            self.node_connections[node2_id].append(node1_id)
            
    def find_optimal_path(self, source_id: str, destination_id: str) -> List[str]:
        """Find the optimal path from source to destination using Dijkstra's algorithm."""
        # Implementation of Dijkstra's algorithm to find shortest path
        # based on connection quality scores
        
        # Initialize distances and previous nodes
        distances = {node: float('infinity') for node in self.node_connections}
        previous_nodes = {node: None for node in self.node_connections}
        distances[source_id] = 0
        
        unvisited_nodes = set(self.node_connections.keys())
        
        while unvisited_nodes:
            # Find the unvisited node with minimum distance
            current_node = min(unvisited_nodes, key=lambda node: distances[node])
            
            if current_node == destination_id:
                break
                
            unvisited_nodes.remove(current_node)
            
            # Check neighbors
            for neighbor in self.node_connections.get(current_node, []):
                if neighbor in unvisited_nodes:
                    # Calculate tentative distance
                    edge_quality = self.connection_quality.get((current_node, neighbor), 0.0)
                    # Use inverse of quality as "distance" (higher quality = shorter distance)
                    tentative_distance = distances[current_node] + (1.0 / (edge_quality + 0.001))
                    
                    if tentative_distance < distances[neighbor]:
                        distances[neighbor] = tentative_distance
                        previous_nodes[neighbor] = current_node
        
        # Reconstruct path
        path = []
        current = destination_id
        while current is not None:
            path.insert(0, current)
            current = previous_nodes[current]
            
        return path if path[0] == source_id else []
        
    def optimize_topology(self):
        """Optimize the network topology based on current conditions."""
        # This would involve analyzing connection qualities and suggesting
        # new connections or removing poor-quality ones
        logger.info("Optimizing network topology...")
        
        # For now, just log the current topology
        for node_id, connections in self.node_connections.items():
            logger.debug(f"Node {node_id} connected to: {connections}")


class P2PNetworkManager:
    """
    Peer-to-peer network manager for agent communication with distributed hash table,
    encrypted communication channels, and network topology optimization.
    """
    
    def __init__(self, node_id: str, ip: str, port: int, node_type: NodeType = NodeType.AGENT_NODE):
        self.node_id = node_id
        self.ip = ip
        self.port = port
        self.node_type = node_type
        self.dht = DHTNode(node_id, ip, port)
        self.encryption_manager = EncryptionManager()
        self.topology_optimizer = NetworkTopologyOptimizer()
        self.connected_nodes: Dict[str, NodeInfo] = {}
        self.message_handlers: Dict[str, Callable] = {}
        self.bootstrap_nodes: List[Tuple[str, int]] = []  # List of (ip, port)
        self.app = web.Application()
        self.runner = None
        self.site = None
        self.peer_discovery_task = None
        
        # Setup routes
        self.app.router.add_post('/message', self.handle_incoming_message)
        self.app.router.add_get('/discover', self.handle_discovery_request)
        self.app.router.add_post('/register', self.handle_registration_request)
        
    async def start_network(self):
        """Start the P2P network services."""
        # Start the web server
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, self.ip, self.port)
        await self.site.start()
        
        logger.info(f"P2P Network started on {self.ip}:{self.port}")
        
        # Connect to bootstrap nodes if any
        for ip, port in self.bootstrap_nodes:
            await self.connect_to_node(ip, port)
            
        # Start peer discovery task
        self.peer_discovery_task = asyncio.create_task(self._peer_discovery_loop())
        
    async def stop_network(self):
        """Stop the P2P network services."""
        if self.peer_discovery_task:
            self.peer_discovery_task.cancel()
            try:
                await self.peer_discovery_task
            except asyncio.CancelledError:
                pass
                
        if self.site:
            await self.site.stop()
        if self.runner:
            await self.runner.cleanup()
            
        logger.info("P2P Network stopped")
        
    async def connect_to_node(self, ip: str, port: int):
        """Connect to another node in the network."""
        try:
            # Send a registration request to the other node
            url = f"http://{ip}:{port}/register"
            payload = {
                "node_id": self.node_id,
                "ip": self.ip,
                "port": self.port,
                "node_type": self.node_type.value,
                "public_key": self.encryption_manager.public_key.public_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PublicFormat.SubjectPublicKeyInfo
                ).decode(),
                "capabilities": ["computation", "storage", "communication"]
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        response_data = await response.json()
                        logger.info(f"Connected to node {response_data.get('node_id', 'unknown')}")
                        
                        # Add to connected nodes
                        node_info = NodeInfo(
                            node_id=response_data.get('node_id', ''),
                            ip=ip,
                            port=port,
                            node_type=NodeType(response_data.get('node_type', 'agent_node')),
                            public_key=response_data.get('public_key', ''),
                            last_seen=datetime.now(),
                            reputation_score=response_data.get('reputation_score', 0.5),
                            capabilities=response_data.get('capabilities', [])
                        )
                        
                        self.connected_nodes[node_info.node_id] = node_info
                        
                        # Add their public key for encryption
                        self.encryption_manager.add_peer_public_key(node_info.node_id, node_info.public_key)
                        
                        # Update topology
                        self.topology_optimizer.update_connection_quality(
                            self.node_id, node_info.node_id, 0.9  # Assume good initial connection
                        )
                        
                        return True
                    else:
                        logger.error(f"Failed to connect to {ip}:{port}, status: {response.status}")
                        return False
        except Exception as e:
            logger.error(f"Error connecting to {ip}:{port}: {str(e)}")
            return False
            
    async def broadcast_message(self, message: NetworkMessage, exclude_sender: bool = True):
        """Broadcast a message to all connected nodes."""
        tasks = []
        for node_id, node_info in self.connected_nodes.items():
            if exclude_sender and node_id == message.sender_id:
                continue
                
            task = asyncio.create_task(
                self.send_message_to_node(node_info, message)
            )
            tasks.append(task)
            
        # Wait for all sends to complete
        await asyncio.gather(*tasks, return_exceptions=True)
        
    async def send_message_to_node(self, node_info: NodeInfo, message: NetworkMessage):
        """Send a message to a specific node."""
        try:
            # Encrypt the message content if not already encrypted
            if not message.encrypted_content:
                encrypted_content = self.encryption_manager.encrypt_message(
                    message.content, node_info.node_id
                )
                message.encrypted_content = encrypted_content
            else:
                encrypted_content = message.encrypted_content
                
            # Prepare the message payload
            message_payload = {
                "message_id": message.message_id,
                "sender_id": message.sender_id,
                "recipient_id": message.recipient_id,
                "message_type": message.message_type,
                "content": encrypted_content,
                "timestamp": message.timestamp.isoformat(),
                "priority": message.priority.value,
                "signature": message.signature
            }
            
            url = f"http://{node_info.ip}:{node_info.port}/message"
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=message_payload) as response:
                    if response.status == 200:
                        logger.debug(f"Message sent to {node_info.node_id}")
                        
                        # Update connection quality
                        self.topology_optimizer.update_connection_quality(
                            self.node_id, node_info.node_id, 0.95
                        )
                    else:
                        logger.error(f"Failed to send message to {node_info.node_id}, status: {response.status}")
                        
                        # Update connection quality negatively
                        self.topology_optimizer.update_connection_quality(
                            self.node_id, node_info.node_id, 0.1
                        )
                        
        except Exception as e:
            logger.error(f"Error sending message to {node_info.node_id}: {str(e)}")
            
            # Update connection quality negatively
            self.topology_optimizer.update_connection_quality(
                self.node_id, node_info.node_id, 0.1
            )
            
    async def send_direct_message(self, recipient_id: str, message_type: str, content: str, priority: MessagePriority = MessagePriority.NORMAL):
        """Send a direct message to a specific node."""
        if recipient_id not in self.connected_nodes:
            # Try to discover the node first
            await self.discover_node(recipient_id)
            
        if recipient_id not in self.connected_nodes:
            raise ValueError(f"Recipient node {recipient_id} not connected")
            
        node_info = self.connected_nodes[recipient_id]
        
        # Create message
        message = NetworkMessage(
            message_id=f"msg_{secrets.token_hex(8)}",
            sender_id=self.node_id,
            recipient_id=recipient_id,
            message_type=message_type,
            content=content,
            timestamp=datetime.now(),
            priority=priority,
            signature=""  # Will be added after encryption
        )
        
        # Sign the message
        message_content = json.dumps({
            "message_id": message.message_id,
            "sender_id": message.sender_id,
            "recipient_id": message.recipient_id,
            "message_type": message.message_type,
            "content": content,
            "timestamp": message.timestamp.isoformat(),
            "priority": message.priority.value
        }, sort_keys=True)
        
        message.signature = self.encryption_manager.sign_message(message_content)
        
        # Send the message
        await self.send_message_to_node(node_info, message)
        
    async def discover_node(self, target_node_id: str) -> Optional[NodeInfo]:
        """Discover a node in the network using DHT."""
        # First check if we already know about this node
        if target_node_id in self.connected_nodes:
            return self.connected_nodes[target_node_id]
            
        # Query our DHT for nodes close to the target
        closest_nodes = self.dht.find_node(target_node_id)
        
        # Ask each close node if they know about the target
        for node_info in closest_nodes:
            try:
                url = f"http://{node_info.ip}:{node_info.port}/discover?target_id={target_node_id}"
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(url) as response:
                        if response.status == 200:
                            node_data = await response.json()
                            if node_data and node_data.get('found'):
                                discovered_info = NodeInfo(
                                    node_id=node_data['node_id'],
                                    ip=node_data['ip'],
                                    port=node_data['port'],
                                    node_type=NodeType(node_data['node_type']),
                                    public_key=node_data['public_key'],
                                    last_seen=datetime.fromisoformat(node_data['last_seen']),
                                    reputation_score=node_data['reputation_score'],
                                    capabilities=node_data['capabilities']
                                )
                                
                                # Add to our records
                                self.connected_nodes[discovered_info.node_id] = discovered_info
                                
                                # Add their public key for encryption
                                self.encryption_manager.add_peer_public_key(
                                    discovered_info.node_id, discovered_info.public_key
                                )
                                
                                return discovered_info
            except Exception as e:
                logger.error(f"Error discovering node {target_node_id} via {node_info.node_id}: {str(e)}")
                
        return None
        
    async def register_message_handler(self, message_type: str, handler: Callable):
        """Register a handler for a specific message type."""
        self.message_handlers[message_type] = handler
        
    async def handle_incoming_message(self, request):
        """Handle incoming message requests."""
        try:
            data = await request.json()
            
            # Reconstruct the message object
            message = NetworkMessage(
                message_id=data['message_id'],
                sender_id=data['sender_id'],
                recipient_id=data['recipient_id'],
                message_type=data['message_type'],
                content=data.get('content', ''),
                timestamp=datetime.fromisoformat(data['timestamp']),
                priority=MessagePriority(data['priority']),
                signature=data['signature'],
                encrypted_content=data.get('content')  # Encrypted content
            )
            
            # Verify the signature
            message_content = json.dumps({
                "message_id": message.message_id,
                "sender_id": message.sender_id,
                "recipient_id": message.recipient_id,
                "message_type": message.message_type,
                "content": message.content if not message.encrypted_content else message.encrypted_content,
                "timestamp": message.timestamp.isoformat(),
                "priority": message.priority.value
            }, sort_keys=True)
            
            sender_public_key = self.dht.get(f"pubkey_{message.sender_id}")
            if not sender_public_key:
                # Look up in connected nodes
                sender_node = self.connected_nodes.get(message.sender_id)
                if sender_node:
                    sender_public_key = sender_node.public_key
            
            if sender_public_key:
                is_valid = self.encryption_manager.verify_signature(
                    message_content, message.signature, sender_public_key
                )
                if not is_valid:
                    logger.warning(f"Invalid signature for message from {message.sender_id}")
                    return web.Response(status=400, text="Invalid signature")
            else:
                logger.warning(f"Unknown sender {message.sender_id}, skipping signature verification")
            
            # Decrypt the message if it's encrypted
            if message.encrypted_content:
                try:
                    decrypted_content = self.encryption_manager.decrypt_message(
                        message.encrypted_content, message.sender_id
                    )
                    message.content = decrypted_content
                except Exception as e:
                    logger.error(f"Error decrypting message: {str(e)}")
                    return web.Response(status=400, text="Decryption failed")
            
            # Update sender's last seen time
            if message.sender_id in self.connected_nodes:
                self.connected_nodes[message.sender_id].last_seen = datetime.now()
            
            # Route to appropriate handler
            handler = self.message_handlers.get(message.message_type)
            if handler:
                try:
                    result = await handler(message)
                    return web.json_response({"status": "success", "result": result})
                except Exception as e:
                    logger.error(f"Error in message handler: {str(e)}")
                    return web.Response(status=500, text=f"Handler error: {str(e)}")
            else:
                logger.warning(f"No handler for message type: {message.message_type}")
                return web.Response(status=400, text=f"No handler for {message.message_type}")
                
        except Exception as e:
            logger.error(f"Error handling incoming message: {str(e)}")
            return web.Response(status=500, text=f"Server error: {str(e)}")
            
    async def handle_discovery_request(self, request):
        """Handle node discovery requests."""
        try:
            target_id = request.query.get('target_id')
            if not target_id:
                return web.json_response({
                    "found": False,
                    "error": "target_id parameter required"
                })
                
            # Check if we know about this node
            if target_id in self.connected_nodes:
                node_info = self.connected_nodes[target_id]
                return web.json_response({
                    "found": True,
                    "node_id": node_info.node_id,
                    "ip": node_info.ip,
                    "port": node_info.port,
                    "node_type": node_info.node_type.value,
                    "public_key": node_info.public_key,
                    "last_seen": node_info.last_seen.isoformat(),
                    "reputation_score": node_info.reputation_score,
                    "capabilities": node_info.capabilities
                })
            else:
                return web.json_response({
                    "found": False,
                    "error": f"Node {target_id} not found"
                })
        except Exception as e:
            logger.error(f"Error handling discovery request: {str(e)}")
            return web.Response(status=500, text=f"Server error: {str(e)}")
            
    async def handle_registration_request(self, request):
        """Handle node registration requests."""
        try:
            data = await request.json()
            
            node_info = NodeInfo(
                node_id=data['node_id'],
                ip=data['ip'],
                port=data['port'],
                node_type=NodeType(data['node_type']),
                public_key=data['public_key'],
                last_seen=datetime.now(),
                reputation_score=data.get('reputation_score', 0.5),
                capabilities=data.get('capabilities', [])
            )
            
            # Add to connected nodes
            self.connected_nodes[node_info.node_id] = node_info
            
            # Add their public key for encryption
            self.encryption_manager.add_peer_public_key(node_info.node_id, node_info.public_key)
            
            # Add to DHT
            self.dht.put(f"node_{node_info.node_id}", node_info)
            self.dht.put(f"pubkey_{node_info.node_id}", node_info.public_key)
            
            logger.info(f"Registered new node: {node_info.node_id}")
            
            return web.json_response({
                "node_id": self.node_id,
                "node_type": self.node_type.value,
                "public_key": self.encryption_manager.public_key.public_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PublicFormat.SubjectPublicKeyInfo
                ).decode(),
                "reputation_score": 0.8,  # Default reputation for new connections
                "capabilities": ["computation", "storage", "communication"]
            })
        except Exception as e:
            logger.error(f"Error handling registration request: {str(e)}")
            return web.Response(status=500, text=f"Server error: {str(e)}")
            
    def add_bootstrap_node(self, ip: str, port: int):
        """Add a bootstrap node for initial network connection."""
        self.bootstrap_nodes.append((ip, port))
        
    async def _peer_discovery_loop(self):
        """Continuously discover new peers in the network."""
        while True:
            try:
                # Periodically optimize topology
                self.topology_optimizer.optimize_topology()
                
                # Sleep for a while before next iteration
                await asyncio.sleep(30)  # Every 30 seconds
            except asyncio.CancelledError:
                logger.info("Peer discovery loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in peer discovery loop: {str(e)}")
                await asyncio.sleep(5)  # Wait before retrying


# Convenience function for easy use
async def create_p2p_network_node(
    node_id: str,
    ip: str = "127.0.0.1",
    port: int = 8080,
    node_type: NodeType = NodeType.AGENT_NODE
) -> P2PNetworkManager:
    """
    Convenience function to create and start a P2P network node.
    
    Args:
        node_id: Unique identifier for the node
        ip: IP address to bind to
        port: Port to listen on
        node_type: Type of node to create
        
    Returns:
        P2PNetworkManager instance
    """
    network_manager = P2PNetworkManager(node_id, ip, port, node_type)
    await network_manager.start_network()
    return network_manager