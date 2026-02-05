"""
Edge Computing Integration
Implements EdgeComputingManager for distributed processing, edge node discovery and management,
workload distribution algorithms, and edge-cloud hybrid processing.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple, Union
from enum import Enum
import json
import secrets
import hashlib
from datetime import datetime, timedelta
import aiohttp
from aiohttp import web
import threading
import time
import socket
import psutil
from dataclasses import dataclass


logger = logging.getLogger(__name__)


class EdgeNodeType(Enum):
    """Types of edge computing nodes."""
    IOT_DEVICE = "iot_device"
    SMART_GATEWAY = "smart_gateway"
    EDGE_SERVER = "edge_server"
    MOBILE_DEVICE = "mobile_device"
    FOG_NODE = "fog_node"
    MICRO_DATA_CENTER = "micro_data_center"


class EdgeResourceType(Enum):
    """Types of resources available at edge nodes."""
    COMPUTE = "compute"
    STORAGE = "storage"
    BANDWIDTH = "bandwidth"
    MEMORY = "memory"
    GPU = "gpu"


class EdgeTaskStatus(Enum):
    """Status of edge computing tasks."""
    PENDING = "pending"
    ASSIGNED = "assigned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class EdgeNode:
    """Represents an edge computing node."""
    node_id: str
    node_type: EdgeNodeType
    ip_address: str
    port: int
    location: str  # Geographic location
    capabilities: Dict[EdgeResourceType, float]  # Resource availability
    status: str  # online, offline, maintenance
    last_seen: datetime
    workload_capacity: int  # Number of concurrent tasks
    current_workload: int
    latency_to_cloud: float  # ms
    bandwidth_to_cloud: float  # Mbps
    metadata: Dict[str, Any]


@dataclass
class EdgeTask:
    """Represents a task to be executed at the edge."""
    task_id: str
    task_type: str  # e.g., "data_processing", "ml_inference", "video_analysis"
    data_size_kb: int
    computation_intensity: float  # CPU/GPU cycles per KB
    deadline: Optional[datetime]
    priority: int  # 1-5 scale, 5 being highest
    assigned_node: Optional[str]
    submitted_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    status: EdgeTaskStatus
    result: Optional[Any]
    metrics: Dict[str, Any]  # Performance metrics


@dataclass
class EdgeResourceAllocation:
    """Represents allocation of resources at an edge node."""
    allocation_id: str
    node_id: str
    resource_type: EdgeResourceType
    requested_amount: float
    allocated_amount: float
    allocated_at: datetime
    expires_at: datetime
    task_id: str
    status: str  # active, expired, released
    metadata: Dict[str, Any]


class EdgeNodeDiscovery:
    """Discovers and manages edge computing nodes."""
    
    def __init__(self, discovery_port: int = 8765):
        self.nodes: Dict[str, EdgeNode] = {}
        self.discovery_port = discovery_port
        self.broadcast_address = '<broadcast>'
        self.discovery_socket = None
        self.discovery_task = None
        
    async def start_discovery(self):
        """Start the node discovery process."""
        self.discovery_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.discovery_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self.discovery_socket.settimeout(0.5)  # Non-blocking
        
        # Start discovery loop
        self.discovery_task = asyncio.create_task(self._discovery_loop())
        
        logger.info(f"Started edge node discovery on port {self.discovery_port}")
        
    async def stop_discovery(self):
        """Stop the node discovery process."""
        if self.discovery_task:
            self.discovery_task.cancel()
            try:
                await self.discovery_task
            except asyncio.CancelledError:
                pass
                
        if self.discovery_socket:
            self.discovery_socket.close()
            
        logger.info("Stopped edge node discovery")
        
    async def _discovery_loop(self):
        """Main discovery loop."""
        while True:
            try:
                # Broadcast discovery message
                discovery_msg = {
                    "type": "discovery_request",
                    "sender_id": "cloud_controller",
                    "timestamp": datetime.now().isoformat()
                }
                
                # Send broadcast
                try:
                    self.discovery_socket.sendto(
                        json.dumps(discovery_msg).encode(),
                        (self.broadcast_address, self.discovery_port)
                    )
                except Exception as e:
                    logger.error(f"Error sending discovery broadcast: {str(e)}")
                    
                # Listen for responses
                await self._listen_for_responses()
                
                # Update node status
                self._update_node_status()
                
                # Wait before next discovery cycle
                await asyncio.sleep(30)  # Discover every 30 seconds
                
            except asyncio.CancelledError:
                logger.info("Discovery loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in discovery loop: {str(e)}")
                await asyncio.sleep(5)  # Wait before retrying
                
    async def _listen_for_responses(self):
        """Listen for discovery responses from edge nodes."""
        try:
            # Set socket to non-blocking temporarily
            self.discovery_socket.setblocking(False)
            
            while True:
                try:
                    data, addr = self.discovery_socket.recvfrom(1024)
                    response = json.loads(data.decode())
                    
                    if response.get("type") == "discovery_response":
                        await self._handle_node_response(response, addr)
                        
                except BlockingIOError:
                    # No data available, break the inner loop
                    break
                except json.JSONDecodeError:
                    logger.warning("Received malformed discovery response")
                    break
                except Exception as e:
                    logger.error(f"Error processing discovery response: {str(e)}")
                    break
                    
        except Exception as e:
            logger.error(f"Error listening for responses: {str(e)}")
            
        # Reset socket to blocking
        self.discovery_socket.setblocking(True)
        
    async def _handle_node_response(self, response: Dict, addr: Tuple[str, int]):
        """Handle a discovery response from an edge node."""
        node_id = response.get("node_id")
        node_type = response.get("node_type", "unknown")
        capabilities = response.get("capabilities", {})
        location = response.get("location", "unknown")
        workload_capacity = response.get("workload_capacity", 1)
        
        # Update or create node
        if node_id in self.nodes:
            node = self.nodes[node_id]
            node.last_seen = datetime.now()
            node.status = "online"
            node.capabilities.update(capabilities)
        else:
            node = EdgeNode(
                node_id=node_id,
                node_type=EdgeNodeType(node_type),
                ip_address=addr[0],
                port=addr[1],
                location=location,
                capabilities=capabilities,
                status="online",
                last_seen=datetime.now(),
                workload_capacity=workload_capacity,
                current_workload=0,
                latency_to_cloud=0.0,  # Will be measured later
                bandwidth_to_cloud=0.0,  # Will be measured later
                metadata=response.get("metadata", {})
            )
            self.nodes[node_id] = node
            
        logger.info(f"Discovered/updated edge node: {node_id} at {addr[0]}:{addr[1]}")
        
    def _update_node_status(self):
        """Update status of nodes based on last seen time."""
        current_time = datetime.now()
        timeout = timedelta(minutes=2)  # Node is considered offline after 2 minutes
        
        for node_id, node in self.nodes.items():
            if current_time - node.last_seen > timeout:
                node.status = "offline"
                
    def get_available_nodes(self, resource_type: EdgeResourceType = None, min_capacity: float = 0) -> List[EdgeNode]:
        """Get available edge nodes that meet resource requirements."""
        available_nodes = []
        
        for node in self.nodes.values():
            if node.status == "online":
                if resource_type is None or (
                    resource_type in node.capabilities and 
                    node.capabilities[resource_type] >= min_capacity and
                    node.current_workload < node.workload_capacity
                ):
                    available_nodes.append(node)
                    
        return available_nodes


class WorkloadDistributionAlgorithm:
    """Algorithms for distributing workloads across edge nodes."""
    
    def __init__(self):
        self.algorithms = {
            "round_robin": self._round_robin_distribution,
            "least_workload": self._least_workload_distribution,
            "latency_aware": self._latency_aware_distribution,
            "resource_aware": self._resource_aware_distribution,
            "hybrid": self._hybrid_distribution
        }
        
    def distribute_tasks(
        self, 
        tasks: List[EdgeTask], 
        nodes: List[EdgeNode], 
        algorithm: str = "hybrid"
    ) -> Dict[str, str]:
        """
        Distribute tasks among nodes using the specified algorithm.
        
        Returns:
            Dict mapping task_id to node_id
        """
        if algorithm not in self.algorithms:
            raise ValueError(f"Unknown algorithm: {algorithm}")
            
        return self.algorithms[algorithm](tasks, nodes)
        
    def _round_robin_distribution(self, tasks: List[EdgeTask], nodes: List[EdgeNode]) -> Dict[str, str]:
        """Distribute tasks using round-robin algorithm."""
        if not nodes:
            return {}
            
        distribution = {}
        node_index = 0
        
        for task in tasks:
            node = nodes[node_index % len(nodes)]
            distribution[task.task_id] = node.node_id
            node_index += 1
            
        return distribution
        
    def _least_workload_distribution(self, tasks: List[EdgeTask], nodes: List[EdgeNode]) -> Dict[str, str]:
        """Distribute tasks to nodes with least current workload."""
        if not nodes:
            return {}
            
        distribution = {}
        
        for task in tasks:
            # Find node with least current workload
            best_node = min(nodes, key=lambda n: n.current_workload)
            distribution[task.task_id] = best_node.node_id
            best_node.current_workload += 1  # Simulate assignment
            
        return distribution
        
    def _latency_aware_distribution(self, tasks: List[EdgeTask], nodes: List[EdgeNode]) -> Dict[str, str]:
        """Distribute tasks considering latency requirements."""
        if not nodes:
            return {}
            
        distribution = {}
        
        for task in tasks:
            # For latency-sensitive tasks, choose node with lowest latency
            # For others, use general selection
            if task.priority >= 4:  # High priority tasks are latency-sensitive
                best_node = min(nodes, key=lambda n: n.latency_to_cloud)
            else:
                # Choose based on workload balance
                best_node = min(nodes, key=lambda n: n.current_workload)
                
            distribution[task.task_id] = best_node.node_id
            best_node.current_workload += 1
            
        return distribution
        
    def _resource_aware_distribution(self, tasks: List[EdgeTask], nodes: List[EdgeNode]) -> Dict[str, str]:
        """Distribute tasks considering resource requirements."""
        if not nodes:
            return {}
            
        distribution = {}
        
        for task in tasks:
            # Calculate resource requirements
            required_compute = task.data_size_kb * task.computation_intensity
            
            # Find node with sufficient resources
            suitable_nodes = [
                n for n in nodes 
                if n.capabilities.get(EdgeResourceType.COMPUTE, 0) >= required_compute
                and n.current_workload < n.workload_capacity
            ]
            
            if suitable_nodes:
                # Among suitable nodes, choose one with least workload
                best_node = min(suitable_nodes, key=lambda n: n.current_workload)
            else:
                # If no suitable node, choose the one with most compute
                best_node = max(nodes, key=lambda n: n.capabilities.get(EdgeResourceType.COMPUTE, 0))
                
            distribution[task.task_id] = best_node.node_id
            best_node.current_workload += 1
            
        return distribution
        
    def _hybrid_distribution(self, tasks: List[EdgeTask], nodes: List[EdgeNode]) -> Dict[str, str]:
        """Hybrid distribution considering multiple factors."""
        if not nodes:
            return {}
            
        distribution = {}
        
        for task in tasks:
            # Calculate a score for each node based on multiple factors
            scored_nodes = []
            for node in nodes:
                # Calculate score based on available resources, workload, and priority
                resource_score = node.capabilities.get(EdgeResourceType.COMPUTE, 0) / (node.current_workload + 1)
                priority_factor = 1.0 + (task.priority / 10.0)  # Higher priority gets higher weight
                availability_score = (node.workload_capacity - node.current_workload) / node.workload_capacity
                
                score = resource_score * priority_factor * availability_score
                scored_nodes.append((node, score))
                
            # Choose node with highest score
            best_node = max(scored_nodes, key=lambda x: x[1])[0]
            distribution[task.task_id] = best_node.node_id
            best_node.current_workload += 1
            
        return distribution


class EdgeResourceAllocator:
    """Manages allocation of resources at edge nodes."""
    
    def __init__(self):
        self.allocations: Dict[str, EdgeResourceAllocation] = {}
        self.node_resources: Dict[str, Dict[EdgeResourceType, float]] = {}
        
    def allocate_resources(
        self, 
        node_id: str, 
        resource_type: EdgeResourceType, 
        amount: float, 
        task_id: str,
        duration_minutes: int = 10
    ) -> Optional[EdgeResourceAllocation]:
        """Allocate resources at an edge node."""
        allocation_id = f"alloc_{secrets.token_hex(8)}"
        
        # Initialize node resources if not already done
        if node_id not in self.node_resources:
            self.node_resources[node_id] = {}
            
        # Check if sufficient resources are available
        current_usage = sum(
            alloc.allocated_amount for alloc in self.allocations.values()
            if alloc.node_id == node_id and 
               alloc.resource_type == resource_type and 
               alloc.status == "active"
        )
        
        total_available = self.node_resources[node_id].get(resource_type, 0)
        
        if current_usage + amount > total_available:
            logger.warning(f"Insufficient {resource_type.value} resources on {node_id}. "
                          f"Requested: {amount}, Available: {total_available - current_usage}")
            return None
            
        # Create allocation
        allocation = EdgeResourceAllocation(
            allocation_id=allocation_id,
            node_id=node_id,
            resource_type=resource_type,
            requested_amount=amount,
            allocated_amount=amount,
            allocated_at=datetime.now(),
            expires_at=datetime.now() + timedelta(minutes=duration_minutes),
            task_id=task_id,
            status="active",
            metadata={"allocation_method": "resource_allocator"}
        )
        
        self.allocations[allocation_id] = allocation
        
        logger.info(f"Allocated {amount} {resource_type.value} on {node_id} for task {task_id}")
        return allocation
        
    def release_allocation(self, allocation_id: str):
        """Release an allocation."""
        if allocation_id in self.allocations:
            allocation = self.allocations[allocation_id]
            allocation.status = "released"
            logger.info(f"Released allocation {allocation_id}")
            
    def cleanup_expired_allocations(self):
        """Clean up expired allocations."""
        current_time = datetime.now()
        expired_allocations = [
            alloc_id for alloc_id, alloc in self.allocations.items()
            if alloc.status == "active" and alloc.expires_at < current_time
        ]
        
        for alloc_id in expired_allocations:
            self.allocations[alloc_id].status = "expired"
            logger.info(f"Expired allocation {alloc_id}")
            
        return len(expired_allocations)


class EdgeComputingManager:
    """
    Edge computing manager for distributed processing with node discovery,
    workload distribution, and edge-cloud hybrid processing.
    """
    
    def __init__(self, cloud_endpoint: str = "http://localhost:8000"):
        self.discovery = EdgeNodeDiscovery()
        self.distribution_algorithm = WorkloadDistributionAlgorithm()
        self.resource_allocator = EdgeResourceAllocator()
        self.cloud_endpoint = cloud_endpoint
        self.tasks: Dict[str, EdgeTask] = {}
        self.active_sessions: Dict[str, Dict[str, Any]] = {}  # task_id -> session_info
        self.edge_to_cloud_latency = {}  # node_id -> latency
        self.performance_metrics = {}
        self.workload_history = []
        
    async def initialize(self):
        """Initialize the edge computing manager."""
        await self.discovery.start_discovery()
        logger.info("Edge computing manager initialized")
        
    async def shutdown(self):
        """Shutdown the edge computing manager."""
        await self.discovery.stop_discovery()
        logger.info("Edge computing manager shutdown")
        
    async def submit_task(
        self, 
        task_type: str, 
        data: Any, 
        priority: int = 3,
        deadline: Optional[datetime] = None,
        required_resources: Dict[EdgeResourceType, float] = None
    ) -> str:
        """Submit a task for edge processing."""
        task_id = f"task_{secrets.token_hex(8)}"
        
        # Calculate task characteristics
        if isinstance(data, (list, dict)):
            data_size = len(json.dumps(data).encode()) / 1024  # KB
        elif isinstance(data, str):
            data_size = len(data.encode()) / 1024  # KB
        else:
            data_size = len(str(data).encode()) / 1024  # KB
            
        # Estimate computation intensity (arbitrary formula)
        computation_intensity = 0.1 * priority  # Higher priority = more intensive
        
        task = EdgeTask(
            task_id=task_id,
            task_type=task_type,
            data_size_kb=data_size,
            computation_intensity=computation_intensity,
            deadline=deadline,
            priority=priority,
            assigned_node=None,
            submitted_at=datetime.now(),
            started_at=None,
            completed_at=None,
            status=EdgeTaskStatus.PENDING,
            result=None,
            metrics={}
        )
        
        self.tasks[task_id] = task
        
        logger.info(f"Submitted task {task_id} of type {task_type}")
        
        # Automatically schedule the task
        await self.schedule_task(task_id, required_resources)
        
        return task_id
        
    async def schedule_task(
        self, 
        task_id: str, 
        required_resources: Dict[EdgeResourceType, float] = None,
        distribution_algorithm: str = "hybrid"
    ):
        """Schedule a task to an appropriate edge node."""
        if task_id not in self.tasks:
            raise ValueError(f"Task {task_id} not found")
            
        task = self.tasks[task_id]
        
        # Get available nodes
        if required_resources:
            # Filter nodes based on required resources
            resource_type = list(required_resources.keys())[0] if required_resources else EdgeResourceType.COMPUTE
            min_capacity = list(required_resources.values())[0] if required_resources else 0
            available_nodes = self.discovery.get_available_nodes(resource_type, min_capacity)
        else:
            available_nodes = self.discovery.get_available_nodes()
            
        if not available_nodes:
            logger.warning(f"No available edge nodes for task {task_id}, processing in cloud")
            # Fallback to cloud processing
            await self._process_task_in_cloud(task)
            return
            
        # Use distribution algorithm to assign node
        task_assignment = self.distribution_algorithm.distribute_tasks([task], available_nodes, distribution_algorithm)
        
        if task_id in task_assignment:
            assigned_node_id = task_assignment[task_id]
            
            # Update task
            task.assigned_node = assigned_node_id
            task.status = EdgeTaskStatus.ASSIGNED
            
            # Update node workload
            for node in available_nodes:
                if node.node_id == assigned_node_id:
                    node.current_workload += 1
                    break
                    
            logger.info(f"Assigned task {task_id} to node {assigned_node_id}")
            
            # Start processing
            await self._start_task_processing(task_id, assigned_node_id)
        else:
            logger.error(f"Could not assign node for task {task_id}")
            
    async def _start_task_processing(self, task_id: str, node_id: str):
        """Start processing a task on an edge node."""
        task = self.tasks[task_id]
        task.status = EdgeTaskStatus.RUNNING
        task.started_at = datetime.now()
        
        # In a real implementation, this would send the task to the edge node
        # For this demo, we'll simulate processing
        await asyncio.sleep(0.1)  # Simulate network delay
        
        # Simulate task processing
        processing_time = min(task.data_size_kb * task.computation_intensity / 100, 2.0)  # Cap at 2 seconds
        await asyncio.sleep(processing_time)
        
        # Simulate result
        task.result = f"Processed {task.data_size_kb:.2f}KB of {task.task_type} data on {node_id}"
        task.completed_at = datetime.now()
        task.status = EdgeTaskStatus.COMPLETED
        
        # Record metrics
        runtime = (task.completed_at - task.started_at).total_seconds() * 1000  # ms
        task.metrics = {
            "runtime_ms": runtime,
            "node_id": node_id,
            "data_processed_kb": task.data_size_kb,
            "throughput_kbps": task.data_size_kb / (runtime / 1000) if runtime > 0 else 0
        }
        
        # Update node workload
        node = self.discovery.nodes.get(node_id)
        if node:
            node.current_workload -= 1  # Decrement workload counter
            
        logger.info(f"Completed task {task_id} on {node_id} in {runtime:.2f}ms")
        
    async def _process_task_in_cloud(self, task: EdgeTask):
        """Process a task in the cloud when no edge resources are available."""
        task.status = EdgeTaskStatus.RUNNING
        task.started_at = datetime.now()
        
        # Simulate cloud processing
        processing_time = min(task.data_size_kb * task.computation_intensity / 50, 5.0)  # Slower cloud processing
        await asyncio.sleep(processing_time)
        
        # Simulate result
        task.result = f"Processed {task.data_size_kb:.2f}KB of {task.task_type} data in cloud"
        task.completed_at = datetime.now()
        task.status = EdgeTaskStatus.COMPLETED
        
        # Record metrics
        runtime = (task.completed_at - task.started_at).total_seconds() * 1000  # ms
        task.metrics = {
            "runtime_ms": runtime,
            "node_id": "cloud",
            "data_processed_kb": task.data_size_kb,
            "throughput_kbps": task.data_size_kb / (runtime / 1000) if runtime > 0 else 0,
            "processed_in_cloud": True
        }
        
        logger.info(f"Completed task {task.task_id} in cloud in {runtime:.2f}ms")
        
    def get_edge_node_status(self, node_id: str = None) -> Union[EdgeNode, List[EdgeNode]]:
        """Get status of edge nodes."""
        if node_id:
            return self.discovery.nodes.get(node_id)
        else:
            return list(self.discovery.nodes.values())
            
    def get_task_status(self, task_id: str) -> Optional[EdgeTask]:
        """Get status of a specific task."""
        return self.tasks.get(task_id)
        
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for edge computing."""
        completed_tasks = [t for t in self.tasks.values() if t.status == EdgeTaskStatus.COMPLETED]
        
        if not completed_tasks:
            return {
                "total_tasks": len(self.tasks),
                "completed_tasks": 0,
                "edge_vs_cloud_ratio": 0.0,
                "average_runtime_ms": 0.0,
                "total_data_processed_mb": 0.0
            }
            
        total_runtime = sum(t.metrics.get("runtime_ms", 0) for t in completed_tasks)
        total_data = sum(t.data_size_kb for t in completed_tasks) / 1024.0  # MB
        
        edge_tasks = [t for t in completed_tasks if t.metrics.get("node_id") != "cloud"]
        cloud_tasks = [t for t in completed_tasks if t.metrics.get("node_id") == "cloud"]
        
        return {
            "total_tasks": len(self.tasks),
            "completed_tasks": len(completed_tasks),
            "edge_tasks": len(edge_tasks),
            "cloud_tasks": len(cloud_tasks),
            "edge_vs_cloud_ratio": len(edge_tasks) / len(completed_tasks) if completed_tasks else 0,
            "average_runtime_ms": total_runtime / len(completed_tasks) if completed_tasks else 0,
            "total_data_processed_mb": total_data,
            "average_throughput_kbps": total_data * 1024 / (total_runtime / 1000) if total_runtime > 0 else 0
        }
        
    async def offload_to_edge(
        self, 
        data: Any, 
        task_type: str, 
        priority: int = 3,
        deadline: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Offload computation to edge nodes.
        
        Returns:
            Result of the computation
        """
        task_id = await self.submit_task(task_type, data, priority, deadline)
        
        # Wait for task completion
        while task_id in self.tasks:
            task = self.tasks[task_id]
            if task.status in [EdgeTaskStatus.COMPLETED, EdgeTaskStatus.FAILED]:
                break
            await asyncio.sleep(0.1)  # Poll every 100ms
            
        task = self.tasks[task_id]
        
        return {
            "task_id": task_id,
            "status": task.status.value,
            "result": task.result,
            "metrics": task.metrics,
            "processed_at": task.completed_at.isoformat() if task.completed_at else None
        }
        
    def get_optimization_recommendations(self) -> List[str]:
        """Get recommendations for optimizing edge computing performance."""
        metrics = self.get_performance_metrics()
        recommendations = []
        
        if metrics["edge_vs_cloud_ratio"] < 0.5:
            recommendations.append("Consider deploying more edge nodes to reduce cloud dependency")
            
        if metrics["average_runtime_ms"] > 1000:  # More than 1 second
            recommendations.append("Task runtimes are high; consider optimizing task distribution algorithm")
            
        # Check node utilization
        nodes = self.get_edge_node_status()
        for node in nodes:
            if node.status == "online":
                utilization = node.current_workload / node.workload_capacity if node.workload_capacity > 0 else 0
                if utilization > 0.8:
                    recommendations.append(f"Node {node.node_id} is highly utilized ({utilization:.1%}); consider load balancing")
                    
        return recommendations


# Convenience function for easy use
async def create_edge_computing_manager(
    cloud_endpoint: str = "http://localhost:8000"
) -> EdgeComputingManager:
    """
    Convenience function to create an edge computing manager.
    
    Args:
        cloud_endpoint: Endpoint for cloud fallback processing
        
    Returns:
        EdgeComputingManager instance
    """
    manager = EdgeComputingManager(cloud_endpoint)
    await manager.initialize()
    return manager