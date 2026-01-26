"""
Resource management and cost optimization system for multi-agent systems in Xencode
"""
from typing import Dict, List, Optional, Any, Set, Tuple
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import uuid
import json
import sqlite3
import threading
import time
import heapq
from pathlib import Path
from collections import defaultdict, deque
import random


class ResourceType(Enum):
    """Types of resources that can be managed."""
    COMPUTE = "compute"
    MEMORY = "memory"
    STORAGE = "storage"
    NETWORK_BANDWIDTH = "network_bandwidth"
    GPU = "gpu"
    SPECIALIZED_HARDWARE = "specialized_hardware"


class ResourcePoolType(Enum):
    """Types of resource pools."""
    STATIC = "static"
    DYNAMIC = "dynamic"
    HYBRID = "hybrid"


class TaskPriority(Enum):
    """Priority levels for tasks."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class ResourceAllocationStatus(Enum):
    """Status of resource allocation."""
    PENDING = "pending"
    ALLOCATED = "allocated"
    IN_USE = "in_use"
    RELEASED = "released"
    FAILED = "failed"


@dataclass
class Resource:
    """Represents a resource that can be allocated."""
    resource_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    resource_type: ResourceType = ResourceType.COMPUTE
    capacity: float = 1.0  # Total capacity
    available: float = 1.0  # Available capacity
    cost_per_unit: float = 1.0  # Cost per unit of resource
    location: str = "local"  # Physical location of resource
    agent_id: Optional[str] = None  # Agent currently using this resource
    status: str = "available"  # available, in_use, maintenance, unavailable
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class ResourcePool:
    """Group of resources of the same type."""
    pool_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    pool_type: ResourcePoolType = ResourcePoolType.STATIC
    resource_type: ResourceType = ResourceType.COMPUTE
    resources: List[Resource] = field(default_factory=list)
    total_capacity: float = 0.0
    available_capacity: float = 0.0
    min_size: int = 1  # Minimum number of resources in pool
    max_size: int = 10  # Maximum number of resources in pool
    current_size: int = 1  # Current number of resources in pool
    cost_per_hour: float = 1.0  # Cost per hour for resources in this pool
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResourceRequest:
    """Request for resources from an agent."""
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str = ""
    resource_type: ResourceType = ResourceType.COMPUTE
    required_amount: float = 1.0
    priority: TaskPriority = TaskPriority.MEDIUM
    deadline: Optional[datetime] = None
    estimated_duration: float = 3600.0  # in seconds
    created_at: datetime = field(default_factory=datetime.now)
    status: ResourceAllocationStatus = ResourceAllocationStatus.PENDING
    allocated_resources: List[str] = field(default_factory=list)  # resource_ids
    cost_estimate: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResourceAllocation:
    """Details of a resource allocation."""
    allocation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    request_id: str = ""
    resource_id: str = ""
    agent_id: str = ""
    amount_allocated: float = 0.0
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    cost_incurred: float = 0.0
    status: ResourceAllocationStatus = ResourceAllocationStatus.ALLOCATED
    metadata: Dict[str, Any] = field(default_factory=dict)


class ResourceManager:
    """Manages resources across the multi-agent system."""
    
    def __init__(self, db_path: str = "resource_management.db"):
        self.db_path = db_path
        self.resources: Dict[str, Resource] = {}
        self.resource_pools: Dict[str, ResourcePool] = {}
        self.active_allocations: Dict[str, ResourceAllocation] = {}
        self.pending_requests: List[ResourceRequest] = []
        self.access_lock = threading.RLock()
        
        # Initialize database
        self._init_db()
        
        # Initialize default resource pools
        self._init_default_pools()
    
    def _init_db(self):
        """Initialize the resource management database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create resources table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS resources (
                resource_id TEXT PRIMARY KEY,
                resource_type TEXT,
                capacity REAL,
                available REAL,
                cost_per_unit REAL,
                location TEXT,
                agent_id TEXT,
                status TEXT,
                metadata TEXT,
                created_at TEXT
            )
        ''')
        
        # Create resource_pools table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS resource_pools (
                pool_id TEXT PRIMARY KEY,
                pool_type TEXT,
                resource_type TEXT,
                total_capacity REAL,
                available_capacity REAL,
                min_size INTEGER,
                max_size INTEGER,
                current_size INTEGER,
                cost_per_hour REAL,
                metadata TEXT
            )
        ''')
        
        # Create resource_requests table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS resource_requests (
                request_id TEXT PRIMARY KEY,
                agent_id TEXT,
                resource_type TEXT,
                required_amount REAL,
                priority INTEGER,
                deadline TEXT,
                estimated_duration REAL,
                created_at TEXT,
                status TEXT,
                allocated_resources TEXT,
                cost_estimate REAL,
                metadata TEXT
            )
        ''')
        
        # Create resource_allocations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS resource_allocations (
                allocation_id TEXT PRIMARY KEY,
                request_id TEXT,
                resource_id TEXT,
                agent_id TEXT,
                amount_allocated REAL,
                start_time TEXT,
                end_time TEXT,
                cost_incurred REAL,
                status TEXT,
                metadata TEXT
            )
        ''')
        
        # Create indexes
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_resources_type ON resources(resource_type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_resources_status ON resources(status)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_resources_agent ON resources(agent_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_requests_agent ON resource_requests(agent_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_requests_status ON resource_requests(status)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_requests_priority ON resource_requests(priority)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_allocations_agent ON resource_allocations(agent_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_allocations_status ON resource_allocations(status)')
        
        conn.commit()
        conn.close()
    
    def _init_default_pools(self):
        """Initialize default resource pools."""
        default_pools = [
            ResourcePool(
                pool_type=ResourcePoolType.DYNAMIC,
                resource_type=ResourceType.COMPUTE,
                total_capacity=10.0,
                available_capacity=10.0,
                min_size=1,
                max_size=20,
                current_size=1,
                cost_per_hour=0.1
            ),
            ResourcePool(
                pool_type=ResourcePoolType.DYNAMIC,
                resource_type=ResourceType.MEMORY,
                total_capacity=32.0,  # GB
                available_capacity=32.0,
                min_size=1,
                max_size=50,
                current_size=1,
                cost_per_hour=0.05
            ),
            ResourcePool(
                pool_type=ResourcePoolType.STATIC,
                resource_type=ResourceType.STORAGE,
                total_capacity=1000.0,  # GB
                available_capacity=1000.0,
                min_size=1,
                max_size=5,
                current_size=1,
                cost_per_hour=0.01
            )
        ]
        
        for pool in default_pools:
            self.create_resource_pool(pool)
    
    def create_resource_pool(self, pool: ResourcePool) -> str:
        """Create a new resource pool."""
        with self.access_lock:
            pool_id = str(uuid.uuid4())
            pool.pool_id = pool_id
            
            # Create initial resources based on pool size
            for i in range(pool.current_size):
                resource = Resource(
                    resource_type=pool.resource_type,
                    capacity=pool.total_capacity / pool.current_size,
                    available=pool.total_capacity / pool.current_size,
                    cost_per_unit=pool.cost_per_hour / 3600,  # Convert to per second
                    location=f"pool_{pool_id}_resource_{i}"
                )
                pool.resources.append(resource)
                self.resources[resource.resource_id] = resource
            
            self.resource_pools[pool_id] = pool
            
            # Store in database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO resource_pools
                (pool_id, pool_type, resource_type, total_capacity, available_capacity, 
                 min_size, max_size, current_size, cost_per_hour, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                pool_id,
                pool.pool_type.value,
                pool.resource_type.value,
                pool.total_capacity,
                pool.available_capacity,
                pool.min_size,
                pool.max_size,
                pool.current_size,
                pool.cost_per_hour,
                json.dumps(pool.metadata)
            ))
            
            # Store resources
            for resource in pool.resources:
                cursor.execute('''
                    INSERT INTO resources
                    (resource_id, resource_type, capacity, available, cost_per_unit, location, status, metadata, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    resource.resource_id,
                    resource.resource_type.value,
                    resource.capacity,
                    resource.available,
                    resource.cost_per_unit,
                    resource.location,
                    resource.status,
                    json.dumps(resource.metadata),
                    resource.created_at.isoformat()
                ))
            
            conn.commit()
            conn.close()
        
        return pool_id
    
    def request_resources(self, agent_id: str, resource_type: ResourceType, 
                         required_amount: float, priority: TaskPriority = TaskPriority.MEDIUM,
                         deadline: Optional[datetime] = None, 
                         estimated_duration: float = 3600.0,
                         metadata: Dict[str, Any] = None) -> str:
        """Request resources for an agent."""
        metadata = metadata or {}
        
        request = ResourceRequest(
            agent_id=agent_id,
            resource_type=resource_type,
            required_amount=required_amount,
            priority=priority,
            deadline=deadline,
            estimated_duration=estimated_duration,
            metadata=metadata
        )
        
        with self.access_lock:
            # Add to pending requests
            self.pending_requests.append(request)
            
            # Store in database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO resource_requests
                (request_id, agent_id, resource_type, required_amount, priority, deadline, 
                 estimated_duration, created_at, status, allocated_resources, cost_estimate, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                request.request_id,
                request.agent_id,
                request.resource_type.value,
                request.required_amount,
                request.priority.value,
                request.deadline.isoformat() if request.deadline else None,
                request.estimated_duration,
                request.created_at.isoformat(),
                request.status.value,
                json.dumps(request.allocated_resources),
                request.cost_estimate,
                json.dumps(request.metadata)
            ))
            
            conn.commit()
            conn.close()
        
        return request.request_id
    
    def allocate_resources(self) -> int:
        """Allocate resources to pending requests based on priority and availability."""
        with self.access_lock:
            # Sort requests by priority (higher first) and then by creation time
            sorted_requests = sorted(
                self.pending_requests,
                key=lambda r: (r.priority.value, -r.created_at.timestamp()),
                reverse=True
            )
            
            allocated_count = 0
            
            for request in sorted_requests:
                # Find appropriate resource pool
                pool = self._find_pool_for_request(request)
                if not pool:
                    continue
                
                # Try to allocate resources
                allocated_resources = self._allocate_from_pool(request, pool)
                if allocated_resources:
                    # Update request status
                    request.status = ResourceAllocationStatus.ALLOCATED
                    request.allocated_resources = [r.resource_id for r in allocated_resources]
                    
                    # Calculate cost estimate
                    cost_per_second = sum(r.cost_per_unit for r in allocated_resources)
                    request.cost_estimate = cost_per_second * request.estimated_duration
                    
                    # Create allocations
                    for resource in allocated_resources:
                        allocation = ResourceAllocation(
                            request_id=request.request_id,
                            resource_id=resource.resource_id,
                            agent_id=request.agent_id,
                            amount_allocated=min(resource.available, request.required_amount),
                            cost_incurred=cost_per_second * request.estimated_duration
                        )
                        self.active_allocations[allocation.allocation_id] = allocation
                        
                        # Update resource availability
                        resource.available -= min(resource.available, request.required_amount)
                        resource.agent_id = request.agent_id
                        resource.status = "in_use"
                        
                        # Update pool availability
                        pool.available_capacity -= min(resource.available + resource.capacity - resource.available, request.required_amount)
                    
                    # Remove from pending requests
                    self.pending_requests = [r for r in self.pending_requests if r.request_id != request.request_id]
                    
                    # Update database
                    self._update_request_in_db(request)
                    self._update_resource_in_db(allocated_resources[0])  # Update one resource to trigger pool update
                    self._add_allocation_to_db(allocation)
                    
                    allocated_count += 1
            
            return allocated_count
    
    def _find_pool_for_request(self, request: ResourceRequest) -> Optional[ResourcePool]:
        """Find an appropriate resource pool for a request."""
        for pool in self.resource_pools.values():
            if (pool.resource_type == request.resource_type and 
                pool.available_capacity >= request.required_amount):
                return pool
        return None
    
    def _allocate_from_pool(self, request: ResourceRequest, pool: ResourcePool) -> List[Resource]:
        """Allocate resources from a pool."""
        allocated_resources = []
        remaining_amount = request.required_amount
        
        # Sort resources by availability (most available first)
        available_resources = [r for r in pool.resources if r.status == "available"]
        available_resources.sort(key=lambda r: r.available, reverse=True)
        
        for resource in available_resources:
            if remaining_amount <= 0:
                break
                
            alloc_amount = min(resource.available, remaining_amount)
            if alloc_amount > 0:
                # Update resource
                resource.available -= alloc_amount
                resource.agent_id = request.agent_id
                resource.status = "in_use"
                
                allocated_resources.append(resource)
                remaining_amount -= alloc_amount
                
                # Update database
                self._update_resource_in_db(resource)
        
        return allocated_resources if remaining_amount <= 0 else []
    
    def _update_resource_in_db(self, resource: Resource):
        """Update a resource in the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE resources 
            SET available = ?, agent_id = ?, status = ?
            WHERE resource_id = ?
        ''', (
            resource.available,
            resource.agent_id,
            resource.status,
            resource.resource_id
        ))
        
        conn.commit()
        conn.close()
    
    def _update_request_in_db(self, request: ResourceRequest):
        """Update a request in the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE resource_requests 
            SET status = ?, allocated_resources = ?, cost_estimate = ?
            WHERE request_id = ?
        ''', (
            request.status.value,
            json.dumps(request.allocated_resources),
            request.cost_estimate,
            request.request_id
        ))
        
        conn.commit()
        conn.close()
    
    def _add_allocation_to_db(self, allocation: ResourceAllocation):
        """Add an allocation to the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO resource_allocations
            (allocation_id, request_id, resource_id, agent_id, amount_allocated, start_time, cost_incurred, status, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            allocation.allocation_id,
            allocation.request_id,
            allocation.resource_id,
            allocation.agent_id,
            allocation.amount_allocated,
            allocation.start_time.isoformat(),
            allocation.cost_incurred,
            allocation.status.value,
            json.dumps(allocation.metadata)
        ))
        
        conn.commit()
        conn.close()
    
    def release_resources(self, allocation_id: str) -> bool:
        """Release allocated resources."""
        with self.access_lock:
            if allocation_id not in self.active_allocations:
                return False
            
            allocation = self.active_allocations[allocation_id]
            allocation.status = ResourceAllocationStatus.RELEASED
            allocation.end_time = datetime.now()
            
            # Find the resource and update it
            resource = self.resources.get(allocation.resource_id)
            if resource:
                resource.available += allocation.amount_allocated
                resource.agent_id = None
                resource.status = "available"
                
                # Update database
                self._update_resource_in_db(resource)
            
            # Update allocation in database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE resource_allocations 
                SET status = ?, end_time = ?
                WHERE allocation_id = ?
            ''', (
                ResourceAllocationStatus.RELEASED.value,
                allocation.end_time.isoformat(),
                allocation.allocation_id
            ))
            
            conn.commit()
            conn.close()
            
            # Remove from active allocations
            del self.active_allocations[allocation_id]
            
            return True
    
    def get_resource_utilization(self) -> Dict[str, Any]:
        """Get resource utilization statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get total resources by type
        cursor.execute('SELECT resource_type, SUM(capacity), SUM(available) FROM resources GROUP BY resource_type')
        resource_stats = cursor.fetchall()
        
        # Get active allocations
        cursor.execute('SELECT COUNT(*), SUM(amount_allocated) FROM resource_allocations WHERE status = ?', ('allocated',))
        active_allocations = cursor.fetchone()
        
        # Get pending requests
        cursor.execute('SELECT COUNT(*), SUM(required_amount) FROM resource_requests WHERE status = ?', ('pending',))
        pending_requests = cursor.fetchone()
        
        conn.close()
        
        utilization = {
            'resource_types': {},
            'active_allocations': active_allocations[0],
            'active_allocation_amount': active_allocations[1] or 0.0,
            'pending_requests': pending_requests[0],
            'pending_request_amount': pending_requests[1] or 0.0
        }
        
        for stat in resource_stats:
            resource_type = stat[0]
            total_capacity = stat[1]
            available_capacity = stat[2]
            utilized = total_capacity - available_capacity if total_capacity else 0
            
            utilization['resource_types'][resource_type] = {
                'total': total_capacity,
                'available': available_capacity,
                'utilized': utilized,
                'utilization_rate': utilized / total_capacity if total_capacity else 0
            }
        
        return utilization
    
    def scale_pool(self, pool_id: str, new_size: int) -> bool:
        """Scale a resource pool up or down."""
        with self.access_lock:
            if pool_id not in self.resource_pools:
                return False
            
            pool = self.resource_pools[pool_id]
            
            # Check if new size is within bounds
            if new_size < pool.min_size or new_size > pool.max_size:
                return False
            
            size_difference = new_size - pool.current_size
            
            if size_difference > 0:
                # Scale up - add resources
                for i in range(size_difference):
                    new_resource = Resource(
                        resource_type=pool.resource_type,
                        capacity=pool.total_capacity / new_size,
                        available=pool.total_capacity / new_size,
                        cost_per_unit=pool.cost_per_hour / 3600,
                        location=f"pool_{pool_id}_resource_{pool.current_size + i}"
                    )
                    pool.resources.append(new_resource)
                    self.resources[new_resource.resource_id] = new_resource
                    
                    # Add to database
                    conn = sqlite3.connect(self.db_path)
                    cursor = conn.cursor()
                    
                    cursor.execute('''
                        INSERT INTO resources
                        (resource_id, resource_type, capacity, available, cost_per_unit, location, status, metadata, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        new_resource.resource_id,
                        new_resource.resource_type.value,
                        new_resource.capacity,
                        new_resource.available,
                        new_resource.cost_per_unit,
                        new_resource.location,
                        new_resource.status,
                        json.dumps(new_resource.metadata),
                        new_resource.created_at.isoformat()
                    ))
                    
                    conn.commit()
                    conn.close()
                
                pool.current_size = new_size
                pool.available_capacity = sum(r.available for r in pool.resources)
                pool.total_capacity = sum(r.capacity for r in pool.resources)
                
            elif size_difference < 0:
                # Scale down - remove resources (only if not in use)
                resources_to_remove = []
                for resource in pool.resources:
                    if resource.status == "available" and len(resources_to_remove) < abs(size_difference):
                        resources_to_remove.append(resource)
                
                for resource in resources_to_remove:
                    pool.resources.remove(resource)
                    del self.resources[resource.resource_id]
                    
                    # Remove from database
                    conn = sqlite3.connect(self.db_path)
                    cursor = conn.cursor()
                    
                    cursor.execute('DELETE FROM resources WHERE resource_id = ?', (resource.resource_id,))
                    
                    conn.commit()
                    conn.close()
                
                pool.current_size = new_size
                pool.available_capacity = sum(r.available for r in pool.resources)
                pool.total_capacity = sum(r.capacity for r in pool.resources)
            
            # Update pool in database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE resource_pools 
                SET current_size = ?, total_capacity = ?, available_capacity = ?
                WHERE pool_id = ?
            ''', (
                pool.current_size,
                pool.total_capacity,
                pool.available_capacity,
                pool_id
            ))
            
            conn.commit()
            conn.close()
            
            return True


class CostOptimizer:
    """Optimizes costs for resource allocation."""
    
    def __init__(self, resource_manager: ResourceManager):
        self.resource_manager = resource_manager
        self.cost_history: List[Dict[str, Any]] = []
        self.access_lock = threading.RLock()
    
    def calculate_optimal_allocation(self, agent_id: str, resource_type: ResourceType, 
                                   required_amount: float, budget_limit: Optional[float] = None) -> Dict[str, Any]:
        """Calculate optimal resource allocation considering cost."""
        with self.access_lock:
            # Get available resources of the required type
            available_resources = [
                r for r in self.resource_manager.resources.values()
                if r.resource_type == resource_type and r.status == "available"
            ]
            
            # Sort by cost efficiency (lowest cost per unit first)
            available_resources.sort(key=lambda r: r.cost_per_unit)
            
            total_cost = 0.0
            allocated_resources = []
            remaining_amount = required_amount
            
            for resource in available_resources:
                if remaining_amount <= 0:
                    break
                
                alloc_amount = min(resource.available, remaining_amount)
                cost = alloc_amount * resource.cost_per_unit
                
                if budget_limit and (total_cost + cost) > budget_limit:
                    # If adding this resource would exceed budget, skip
                    continue
                
                allocated_resources.append({
                    'resource_id': resource.resource_id,
                    'amount': alloc_amount,
                    'cost': cost,
                    'cost_per_unit': resource.cost_per_unit
                })
                
                total_cost += cost
                remaining_amount -= alloc_amount
            
            result = {
                'success': remaining_amount <= 0,
                'allocated_resources': allocated_resources,
                'total_cost': total_cost,
                'remaining_amount': remaining_amount,
                'budget_limit': budget_limit,
                'cost_efficiency': total_cost / required_amount if required_amount > 0 else 0
            }
            
            # Add to cost history
            self.cost_history.append({
                'timestamp': datetime.now(),
                'agent_id': agent_id,
                'resource_type': resource_type.value,
                'required_amount': required_amount,
                'allocated_amount': required_amount - remaining_amount,
                'total_cost': total_cost,
                'budget_limit': budget_limit
            })
            
            return result
    
    def get_cost_optimization_report(self) -> Dict[str, Any]:
        """Get a report on cost optimization."""
        with self.access_lock:
            if not self.cost_history:
                return {'message': 'No cost history available'}
            
            total_cost = sum(entry['total_cost'] for entry in self.cost_history)
            total_resources = sum(entry['allocated_amount'] for entry in self.cost_history)
            avg_cost_per_unit = total_cost / total_resources if total_resources > 0 else 0
            
            # Calculate savings compared to worst-case scenario (highest cost resources)
            potential_max_cost = 0.0
            for entry in self.cost_history:
                # Find most expensive resources
                expensive_resources = [
                    r for r in self.resource_manager.resources.values()
                    if r.resource_type == ResourceType(entry['resource_type']) and r.status == "available"
                ]
                expensive_resources.sort(key=lambda r: r.cost_per_unit, reverse=True)
                
                remaining = entry['allocated_amount']
                max_cost = 0.0
                for resource in expensive_resources:
                    if remaining <= 0:
                        break
                    alloc_amount = min(resource.available, remaining)
                    max_cost += alloc_amount * resource.cost_per_unit
                    remaining -= alloc_amount
                
                potential_max_cost += max_cost
            
            actual_cost = total_cost
            potential_savings = potential_max_cost - actual_cost if potential_max_cost > actual_cost else 0
            
            return {
                'total_cost_incurred': actual_cost,
                'average_cost_per_unit': avg_cost_per_unit,
                'total_resources_allocated': total_resources,
                'potential_maximum_cost': potential_max_cost,
                'potential_savings': potential_savings,
                'optimization_efficiency': (potential_savings / potential_max_cost) * 100 if potential_max_cost > 0 else 0,
                'transaction_count': len(self.cost_history)
            }


class PriorityScheduler:
    """Manages priority-based scheduling of resource requests."""
    
    def __init__(self, resource_manager: ResourceManager):
        self.resource_manager = resource_manager
        self.request_queue: List[Tuple[int, datetime, str]] = []  # (priority, timestamp, request_id)
        self.access_lock = threading.RLock()
    
    def submit_request_with_priority(self, request_id: str, priority: TaskPriority):
        """Submit a request to the priority queue."""
        with self.access_lock:
            # Add to heap queue: higher priority value = higher priority
            heapq.heappush(self.request_queue, (-priority.value, datetime.now(), request_id))
    
    def get_next_request(self) -> Optional[str]:
        """Get the next highest priority request."""
        with self.access_lock:
            if not self.request_queue:
                return None
            
            # Pop the highest priority request
            priority_neg, timestamp, request_id = heapq.heappop(self.request_queue)
            return request_id
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get the status of the priority queue."""
        with self.access_lock:
            queue_info = {
                'total_requests': len(self.request_queue),
                'by_priority': {
                    'critical': len([r for r in self.request_queue if r[0] == -TaskPriority.CRITICAL.value]),
                    'high': len([r for r in self.request_queue if r[0] == -TaskPriority.HIGH.value]),
                    'medium': len([r for r in self.request_queue if r[0] == -TaskPriority.MEDIUM.value]),
                    'low': len([r for r in self.request_queue if r[0] == -TaskPriority.LOW.value])
                }
            }
            return queue_info


class ResourceManagementSystem:
    """Main system for resource management and cost optimization."""
    
    def __init__(self, db_path: str = "resource_management.db"):
        self.resource_manager = ResourceManager(db_path)
        self.cost_optimizer = CostOptimizer(self.resource_manager)
        self.priority_scheduler = PriorityScheduler(self.resource_manager)
        self.access_lock = threading.RLock()
    
    def request_resources(self, agent_id: str, resource_type: ResourceType, 
                         required_amount: float, priority: TaskPriority = TaskPriority.MEDIUM,
                         deadline: Optional[datetime] = None, 
                         estimated_duration: float = 3600.0,
                         budget_limit: Optional[float] = None) -> Dict[str, Any]:
        """Request resources with cost optimization."""
        # Submit to priority scheduler
        request_id = self.resource_manager.request_resources(
            agent_id, resource_type, required_amount, priority, deadline, estimated_duration
        )
        self.priority_scheduler.submit_request_with_priority(request_id, priority)
        
        # If budget is specified, use cost optimizer
        if budget_limit:
            optimization_result = self.cost_optimizer.calculate_optimal_allocation(
                agent_id, resource_type, required_amount, budget_limit
            )
            
            return {
                'request_id': request_id,
                'optimization_result': optimization_result,
                'message': 'Request submitted with cost optimization'
            }
        
        return {
            'request_id': request_id,
            'message': 'Request submitted'
        }
    
    def process_resource_requests(self) -> Dict[str, Any]:
        """Process pending resource requests."""
        with self.access_lock:
            # Process requests based on priority
            processed_count = 0
            results = []
            
            while True:
                next_request_id = self.priority_scheduler.get_next_request()
                if not next_request_id:
                    break
                
                # For now, just allocate resources directly
                # In a real system, we'd look up the request details
                allocated = self.resource_manager.allocate_resources()
                processed_count += allocated
                results.append(f"Processed {allocated} allocations")
                
                # Limit to prevent infinite loop
                if processed_count > 10:
                    break
        
        return {
            'processed_requests': processed_count,
            'results': results,
            'queue_status': self.priority_scheduler.get_queue_status()
        }
    
    def release_resources(self, allocation_id: str) -> bool:
        """Release allocated resources."""
        return self.resource_manager.release_resources(allocation_id)
    
    def get_utilization_report(self) -> Dict[str, Any]:
        """Get resource utilization report."""
        return self.resource_manager.get_resource_utilization()
    
    def get_cost_report(self) -> Dict[str, Any]:
        """Get cost optimization report."""
        return self.cost_optimizer.get_cost_optimization_report()
    
    def scale_resource_pool(self, pool_id: str, new_size: int) -> bool:
        """Scale a resource pool."""
        return self.resource_manager.scale_pool(pool_id, new_size)
    
    def get_optimal_allocation(self, agent_id: str, resource_type: ResourceType, 
                              required_amount: float, budget_limit: Optional[float] = None) -> Dict[str, Any]:
        """Get optimal resource allocation considering cost."""
        return self.cost_optimizer.calculate_optimal_allocation(
            agent_id, resource_type, required_amount, budget_limit
        )


# Helper functions for common operations
def create_compute_resource_pool(initial_size: int = 5, total_capacity: float = 10.0) -> ResourcePool:
    """Create a compute resource pool."""
    return ResourcePool(
        pool_type=ResourcePoolType.DYNAMIC,
        resource_type=ResourceType.COMPUTE,
        total_capacity=total_capacity,
        available_capacity=total_capacity,
        min_size=1,
        max_size=50,
        current_size=initial_size,
        cost_per_hour=0.1
    )


def create_memory_resource_pool(initial_size: int = 2, total_capacity: float = 32.0) -> ResourcePool:
    """Create a memory resource pool."""
    return ResourcePool(
        pool_type=ResourcePoolType.DYNAMIC,
        resource_type=ResourceType.MEMORY,
        total_capacity=total_capacity,
        available_capacity=total_capacity,
        min_size=1,
        max_size=20,
        current_size=initial_size,
        cost_per_hour=0.05
    )


def request_resources_with_budget(agent_id: str, resource_type: ResourceType, 
                                required_amount: float, budget: float) -> Dict[str, Any]:
    """Request resources with a specific budget constraint."""
    system = ResourceManagementSystem()
    return system.request_resources(
        agent_id, resource_type, required_amount, 
        budget_limit=budget
    )