"""
Microservice Architecture Optimization
Implements MicroserviceManager for service coordination, service mesh integration,
load balancing and auto-scaling, and service discovery and health monitoring.
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
import random
from dataclasses import dataclass
from collections import defaultdict, deque
import psutil
import os


logger = logging.getLogger(__name__)


class ServiceStatus(Enum):
    """Status of microservices."""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    STARTING = "starting"
    STOPPING = "stopping"
    MAINTENANCE = "maintenance"
    UNKNOWN = "unknown"


class ServiceType(Enum):
    """Types of microservices."""
    API_GATEWAY = "api_gateway"
    AUTH_SERVICE = "auth_service"
    USER_SERVICE = "user_service"
    ORDER_SERVICE = "order_service"
    PAYMENT_SERVICE = "payment_service"
    NOTIFICATION_SERVICE = "notification_service"
    DATA_PROCESSING = "data_processing"
    ANALYTICS = "analytics"
    CACHE_SERVICE = "cache_service"
    DATABASE_PROXY = "database_proxy"


class LoadBalancingStrategy(Enum):
    """Strategies for load balancing."""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    IP_HASH = "ip_hash"
    LEAST_RESPONSE_TIME = "least_response_time"


class AutoScalingPolicy(Enum):
    """Policies for auto-scaling."""
    CPU_BASED = "cpu_based"
    MEMORY_BASED = "memory_based"
    REQUEST_RATE_BASED = "request_rate_based"
    CUSTOM_METRIC_BASED = "custom_metric_based"


@dataclass
class MicroserviceInstance:
    """Represents an instance of a microservice."""
    instance_id: str
    service_type: ServiceType
    host: str
    port: int
    status: ServiceStatus
    weight: int
    current_load: int
    max_concurrent_requests: int
    cpu_usage: float  # Percentage
    memory_usage: float  # Percentage
    response_time_avg: float  # milliseconds
    requests_per_minute: float
    last_heartbeat: datetime
    metadata: Dict[str, Any]


@dataclass
class ServiceRegistration:
    """Information for service registration."""
    service_id: str
    service_type: ServiceType
    host: str
    port: int
    health_check_url: str
    metadata: Dict[str, Any]
    registered_at: datetime


@dataclass
class LoadBalancingConfig:
    """Configuration for load balancing."""
    strategy: LoadBalancingStrategy
    sticky_sessions: bool
    health_check_interval: int  # seconds
    unhealthy_threshold: int
    healthy_threshold: int
    timeout: int  # seconds


@dataclass
class AutoScalingConfig:
    """Configuration for auto-scaling."""
    policy: AutoScalingPolicy
    min_instances: int
    max_instances: int
    target_utilization: float  # For CPU/memory based scaling
    scale_up_threshold: float
    scale_down_threshold: float
    cooldown_period: int  # seconds
    custom_metric_name: Optional[str]


class ServiceRegistry:
    """Manages registration and discovery of microservices."""
    
    def __init__(self):
        self.services: Dict[str, ServiceRegistration] = {}
        self.service_instances: Dict[str, List[MicroserviceInstance]] = defaultdict(list)
        self.instance_heartbeats: Dict[str, datetime] = {}
        self.lock = threading.Lock()
        
    def register_service(self, registration: ServiceRegistration) -> str:
        """Register a service instance."""
        with self.lock:
            # Generate instance ID
            instance_id = f"{registration.service_id}_{secrets.token_hex(8)[:8]}"
            
            instance = MicroserviceInstance(
                instance_id=instance_id,
                service_type=registration.service_type,
                host=registration.host,
                port=registration.port,
                status=ServiceStatus.STARTING,
                weight=1,
                current_load=0,
                max_concurrent_requests=100,
                cpu_usage=0.0,
                memory_usage=0.0,
                response_time_avg=0.0,
                requests_per_minute=0.0,
                last_heartbeat=datetime.now(),
                metadata=registration.metadata
            )
            
            # Add to service instances
            self.service_instances[registration.service_type.value].append(instance)
            self.instance_heartbeats[instance_id] = datetime.now()
            
            # Store registration info
            self.services[instance_id] = registration
            
            logger.info(f"Registered service instance: {instance_id} ({registration.service_type.value})")
            return instance_id
            
    def deregister_service(self, instance_id: str):
        """Deregister a service instance."""
        with self.lock:
            if instance_id in self.services:
                registration = self.services[instance_id]
                
                # Remove from service instances
                service_type_list = self.service_instances[registration.service_type.value]
                self.service_instances[registration.service_type.value] = [
                    inst for inst in service_type_list if inst.instance_id != instance_id
                ]
                
                # Remove from registries
                del self.services[instance_id]
                if instance_id in self.instance_heartbeats:
                    del self.instance_heartbeats[instance_id]
                    
                logger.info(f"Deregistered service instance: {instance_id}")
                
    def send_heartbeat(self, instance_id: str) -> bool:
        """Send a heartbeat from a service instance."""
        with self.lock:
            if instance_id in self.instance_heartbeats:
                self.instance_heartbeats[instance_id] = datetime.now()
                
                # Update instance status
                for service_type, instances in self.service_instances.items():
                    for instance in instances:
                        if instance.instance_id == instance_id:
                            instance.last_heartbeat = datetime.now()
                            instance.status = ServiceStatus.HEALTHY
                            break
                            
                return True
            return False
            
    def get_service_instances(self, service_type: ServiceType) -> List[MicroserviceInstance]:
        """Get all instances of a service type."""
        with self.lock:
            return self.service_instances[service_type.value].copy()
            
    def get_healthy_instances(self, service_type: ServiceType) -> List[MicroserviceInstance]:
        """Get healthy instances of a service type."""
        with self.lock:
            healthy_instances = []
            for instance in self.service_instances[service_type.value]:
                # Check if instance is healthy based on heartbeat
                heartbeat_threshold = datetime.now() - timedelta(seconds=30)  # 30 sec threshold
                if (instance.status == ServiceStatus.HEALTHY and 
                    instance.last_heartbeat > heartbeat_threshold):
                    healthy_instances.append(instance)
            return healthy_instances
            
    def update_instance_metrics(
        self, 
        instance_id: str, 
        cpu_usage: float, 
        memory_usage: float, 
        response_time: float,
        requests_per_minute: float
    ):
        """Update metrics for a service instance."""
        with self.lock:
            for service_type, instances in self.service_instances.items():
                for instance in instances:
                    if instance.instance_id == instance_id:
                        instance.cpu_usage = cpu_usage
                        instance.memory_usage = memory_usage
                        instance.response_time_avg = response_time
                        instance.requests_per_minute = requests_per_minute
                        break


class LoadBalancer:
    """Distributes requests across service instances."""
    
    def __init__(self, config: LoadBalancingConfig):
        self.config = config
        self.service_registry = None
        self.current_index = defaultdict(int)  # For round-robin per service type
        self.session_affinity = {}  # For sticky sessions
        self.response_times = defaultdict(deque)  # Track response times for least_response_time
        self.lock = threading.Lock()
        
    def set_service_registry(self, registry: ServiceRegistry):
        """Set the service registry."""
        self.service_registry = registry
        
    def select_instance(self, service_type: ServiceType, client_ip: str = None) -> Optional[MicroserviceInstance]:
        """Select an instance based on the load balancing strategy."""
        if not self.service_registry:
            return None
            
        healthy_instances = self.service_registry.get_healthy_instances(service_type)
        if not healthy_instances:
            return None
            
        # Filter out overloaded instances
        available_instances = [
            inst for inst in healthy_instances 
            if inst.current_load < inst.max_concurrent_requests
        ]
        
        if not available_instances:
            # All instances are overloaded, return the least loaded one
            return min(healthy_instances, key=lambda x: x.current_load)
            
        if self.config.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return self._round_robin_select(available_instances, service_type)
        elif self.config.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            return self._least_connections_select(available_instances)
        elif self.config.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
            return self._weighted_round_robin_select(available_instances)
        elif self.config.strategy == LoadBalancingStrategy.IP_HASH:
            return self._ip_hash_select(available_instances, client_ip)
        elif self.config.strategy == LoadBalancingStrategy.LEAST_RESPONSE_TIME:
            return self._least_response_time_select(available_instances)
        else:
            # Default to round-robin
            return self._round_robin_select(available_instances, service_type)
            
    def _round_robin_select(self, instances: List[MicroserviceInstance], service_type: ServiceType) -> MicroserviceInstance:
        """Select instance using round-robin algorithm."""
        if not instances:
            return None
            
        idx = self.current_index[service_type.value]
        selected = instances[idx % len(instances)]
        self.current_index[service_type.value] = (idx + 1) % len(instances)
        return selected
        
    def _least_connections_select(self, instances: List[MicroserviceInstance]) -> MicroserviceInstance:
        """Select instance with least current connections."""
        if not instances:
            return None
        return min(instances, key=lambda x: x.current_load)
        
    def _weighted_round_robin_select(self, instances: List[MicroserviceInstance]) -> MicroserviceInstance:
        """Select instance using weighted round-robin algorithm."""
        if not instances:
            return None
            
        # Create a list with repeated instances based on their weights
        weighted_list = []
        for instance in instances:
            weight = instance.weight
            weighted_list.extend([instance] * weight)
            
        if not weighted_list:
            return self._round_robin_select(instances, instances[0].service_type)
            
        idx = self.current_index[f"weighted_{instances[0].service_type.value}"]
        selected = weighted_list[idx % len(weighted_list)]
        self.current_index[f"weighted_{instances[0].service_type.value}"] = (idx + 1) % len(weighted_list)
        return selected
        
    def _ip_hash_select(self, instances: List[MicroserviceInstance], client_ip: str) -> MicroserviceInstance:
        """Select instance using IP hash algorithm."""
        if not instances or not client_ip:
            return self._round_robin_select(instances, instances[0].service_type) if instances else None
            
        hash_value = hash(client_ip) % len(instances)
        return instances[hash_value]
        
    def _least_response_time_select(self, instances: List[MicroserviceInstance]) -> MicroserviceInstance:
        """Select instance with least average response time."""
        if not instances:
            return None
            
        # Use the stored response times to make selection
        return min(instances, key=lambda x: x.response_time_avg)
        
    def record_request_completion(self, instance_id: str, response_time: float):
        """Record completion of a request to update metrics."""
        with self.lock:
            # Update response time history
            self.response_times[instance_id].append(response_time)
            if len(self.response_times[instance_id]) > 100:  # Keep last 100 measurements
                self.response_times[instance_id].popleft()


class AutoScaler:
    """Manages auto-scaling of microservice instances."""
    
    def __init__(self, config: AutoScalingConfig):
        self.config = config
        self.service_registry = None
        self.scaling_history = deque(maxlen=100)
        self.last_scaling_action = {}
        self.scaling_lock = threading.Lock()
        
    def set_service_registry(self, registry: ServiceRegistry):
        """Set the service registry."""
        self.service_registry = registry
        
    def should_scale(self, service_type: ServiceType) -> Tuple[bool, str, int]:
        """
        Determine if scaling is needed.
        
        Returns:
            Tuple of (should_scale, reason, desired_count)
        """
        if not self.service_registry:
            return False, "No service registry", 0
            
        instances = self.service_registry.get_service_instances(service_type)
        current_count = len(instances)
        
        if current_count < self.config.min_instances:
            return True, f"Below minimum instances ({self.config.min_instances})", self.config.min_instances
            
        if current_count > self.config.max_instances:
            return True, f"Above maximum instances ({self.config.max_instances})", self.config.max_instances
            
        # Evaluate scaling based on policy
        if self.config.policy == AutoScalingPolicy.CPU_BASED:
            avg_cpu = sum(inst.cpu_usage for inst in instances) / len(instances) if instances else 0
            if avg_cpu > self.config.scale_up_threshold:
                new_count = min(current_count + 1, self.config.max_instances)
                return True, f"CPU usage {avg_cpu:.1f}% exceeds threshold {self.config.scale_up_threshold}%", new_count
            elif avg_cpu < self.config.scale_down_threshold:
                new_count = max(current_count - 1, self.config.min_instances)
                return True, f"CPU usage {avg_cpu:.1f}% below threshold {self.config.scale_down_threshold}%", new_count
                
        elif self.config.policy == AutoScalingPolicy.MEMORY_BASED:
            avg_memory = sum(inst.memory_usage for inst in instances) / len(instances) if instances else 0
            if avg_memory > self.config.scale_up_threshold:
                new_count = min(current_count + 1, self.config.max_instances)
                return True, f"Memory usage {avg_memory:.1f}% exceeds threshold {self.config.scale_up_threshold}%", new_count
            elif avg_memory < self.config.scale_down_threshold:
                new_count = max(current_count - 1, self.config.min_instances)
                return True, f"Memory usage {avg_memory:.1f}% below threshold {self.config.scale_down_threshold}%", new_count
                
        elif self.config.policy == AutoScalingPolicy.REQUEST_RATE_BASED:
            avg_requests = sum(inst.requests_per_minute for inst in instances) / len(instances) if instances else 0
            if avg_requests > self.config.scale_up_threshold:
                new_count = min(current_count + 1, self.config.max_instances)
                return True, f"Request rate {avg_requests:.1f} exceeds threshold {self.config.scale_up_threshold}", new_count
            elif avg_requests < self.config.scale_down_threshold:
                new_count = max(current_count - 1, self.config.min_instances)
                return True, f"Request rate {avg_requests:.1f} below threshold {self.config.scale_down_threshold}", new_count
                
        return False, "No scaling needed", current_count
        
    def scale_service(self, service_type: ServiceType, desired_count: int) -> bool:
        """Scale a service to the desired count."""
        if not self.service_registry:
            return False
            
        current_instances = self.service_registry.get_service_instances(service_type)
        current_count = len(current_instances)
        
        if current_count == desired_count:
            return True  # Already at desired count
            
        if desired_count > current_count:
            # Scale up - create new instances
            for _ in range(desired_count - current_count):
                # In a real system, this would start a new service instance
                # For this demo, we'll just simulate it
                logger.info(f"Scaling up {service_type.value}: {current_count} -> {desired_count}")
        else:
            # Scale down - terminate instances
            instances_to_remove = current_instances[:current_count - desired_count]
            for instance in instances_to_remove:
                # In a real system, this would terminate the service instance
                # For this demo, we'll just simulate it
                self.service_registry.deregister_service(instance.instance_id)
                logger.info(f"Scaling down {service_type.value}: Terminated {instance.instance_id}")
                
        # Record scaling action
        scaling_record = {
            "timestamp": datetime.now(),
            "service_type": service_type.value,
            "from_count": current_count,
            "to_count": desired_count,
            "reason": "auto_scaling_policy"
        }
        self.scaling_history.append(scaling_record)
        
        return True


class ServiceMesh:
    """Manages service mesh functionality."""
    
    def __init__(self):
        self.service_to_service_communication = {}
        self.traffic_policies = {}
        self.security_policies = {}
        self.telemetry_collectors = {}
        
    def configure_traffic_policy(self, source_service: ServiceType, dest_service: ServiceType, policy: Dict[str, Any]):
        """Configure traffic policy between services."""
        key = f"{source_service.value}->{dest_service.value}"
        self.traffic_policies[key] = policy
        logger.info(f"Configured traffic policy: {key}")
        
    def configure_security_policy(self, service_type: ServiceType, policy: Dict[str, Any]):
        """Configure security policy for a service."""
        self.security_policies[service_type.value] = policy
        logger.info(f"Configured security policy for {service_type.value}")
        
    def enable_telemetry(self, service_type: ServiceType, collector_config: Dict[str, Any]):
        """Enable telemetry collection for a service."""
        self.telemetry_collectors[service_type.value] = collector_config
        logger.info(f"Enabled telemetry for {service_type.value}")


class MicroserviceManager:
    """
    Microservice manager for service coordination with service mesh integration,
    load balancing, auto-scaling, and health monitoring.
    """
    
    def __init__(self):
        self.service_registry = ServiceRegistry()
        self.load_balancer = None
        self.auto_scaler = None
        self.service_mesh = ServiceMesh()
        self.health_monitor_task = None
        self.scaling_monitor_task = None
        self.metrics_collector_task = None
        self.stop_event = asyncio.Event()
        
    async def initialize(
        self, 
        load_balancing_config: LoadBalancingConfig = None,
        auto_scaling_config: AutoScalingConfig = None
    ):
        """Initialize the microservice manager."""
        # Set up load balancer
        if load_balancing_config:
            self.load_balancer = LoadBalancer(load_balancing_config)
            self.load_balancer.set_service_registry(self.service_registry)
        else:
            # Default to round-robin
            default_lb_config = LoadBalancingConfig(
                strategy=LoadBalancingStrategy.ROUND_ROBIN,
                sticky_sessions=False,
                health_check_interval=30,
                unhealthy_threshold=3,
                healthy_threshold=2,
                timeout=10
            )
            self.load_balancer = LoadBalancer(default_lb_config)
            self.load_balancer.set_service_registry(self.service_registry)
            
        # Set up auto-scaler
        if auto_scaling_config:
            self.auto_scaler = AutoScaler(auto_scaling_config)
            self.auto_scaler.set_service_registry(self.service_registry)
        else:
            # Default auto-scaling config
            default_as_config = AutoScalingConfig(
                policy=AutoScalingPolicy.CPU_BASED,
                min_instances=1,
                max_instances=10,
                target_utilization=70.0,
                scale_up_threshold=80.0,
                scale_down_threshold=30.0,
                cooldown_period=300
            )
            self.auto_scaler = AutoScaler(default_as_config)
            self.auto_scaler.set_service_registry(self.service_registry)
            
        # Start monitoring tasks
        self.health_monitor_task = asyncio.create_task(self._health_monitor_loop())
        self.scaling_monitor_task = asyncio.create_task(self._scaling_monitor_loop())
        self.metrics_collector_task = asyncio.create_task(self._metrics_collection_loop())
        
        logger.info("Microservice manager initialized")
        
    async def shutdown(self):
        """Shutdown the microservice manager."""
        self.stop_event.set()
        
        # Cancel monitoring tasks
        if self.health_monitor_task:
            self.health_monitor_task.cancel()
            try:
                await self.health_monitor_task
            except asyncio.CancelledError:
                pass
                
        if self.scaling_monitor_task:
            self.scaling_monitor_task.cancel()
            try:
                await self.scaling_monitor_task
            except asyncio.CancelledError:
                pass
                
        if self.metrics_collector_task:
            self.metrics_collector_task.cancel()
            try:
                await self.metrics_collector_task
            except asyncio.CancelledError:
                pass
                
        logger.info("Microservice manager shutdown")
        
    def register_service(
        self, 
        service_type: ServiceType, 
        host: str, 
        port: int, 
        health_check_url: str = "/health",
        metadata: Dict[str, Any] = None
    ) -> str:
        """Register a service instance."""
        registration = ServiceRegistration(
            service_id=f"{service_type.value}_{secrets.token_hex(8)[:8]}",
            service_type=service_type,
            host=host,
            port=port,
            health_check_url=health_check_url,
            metadata=metadata or {},
            registered_at=datetime.now()
        )
        
        instance_id = self.service_registry.register_service(registration)
        return instance_id
        
    def send_heartbeat(self, instance_id: str):
        """Send a heartbeat from a service instance."""
        return self.service_registry.send_heartbeat(instance_id)
        
    def get_service_instance(self, service_type: ServiceType, client_ip: str = None) -> Optional[MicroserviceInstance]:
        """Get an appropriate service instance for a request."""
        if not self.load_balancer:
            return None
            
        return self.load_balancer.select_instance(service_type, client_ip)
        
    def record_request_completion(self, instance_id: str, response_time: float):
        """Record completion of a request."""
        if self.load_balancer:
            self.load_balancer.record_request_completion(instance_id, response_time)
            
        # Update instance metrics
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent
        
        self.service_registry.update_instance_metrics(
            instance_id,
            cpu_percent,
            memory_percent,
            response_time,
            1.0  # requests per minute placeholder
        )
        
    async def _health_monitor_loop(self):
        """Monitor service health."""
        while not self.stop_event.is_set():
            try:
                # Check for unhealthy instances based on heartbeat
                current_time = datetime.now()
                heartbeat_threshold = current_time - timedelta(seconds=30)
                
                for service_type, instances in self.service_registry.service_instances.items():
                    for instance in instances:
                        if instance.last_heartbeat < heartbeat_threshold:
                            if instance.status != ServiceStatus.UNHEALTHY:
                                logger.warning(f"Service instance {instance.instance_id} is unhealthy")
                                instance.status = ServiceStatus.UNHEALTHY
                                
                # Wait before next check
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except asyncio.CancelledError:
                logger.info("Health monitor loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in health monitor loop: {str(e)}")
                await asyncio.sleep(5)  # Wait before retrying
                
    async def _scaling_monitor_loop(self):
        """Monitor and perform auto-scaling."""
        while not self.stop_event.is_set():
            try:
                # Check each service type for scaling needs
                for service_type in ServiceType:
                    should_scale, reason, desired_count = self.auto_scaler.should_scale(service_type)
                    
                    if should_scale:
                        logger.info(f"Scaling {service_type.value}: {reason}")
                        success = self.auto_scaler.scale_service(service_type, desired_count)
                        
                        if success:
                            logger.info(f"Scaled {service_type.value} to {desired_count} instances")
                        else:
                            logger.error(f"Failed to scale {service_type.value}")
                            
                # Wait before next check
                await asyncio.sleep(60)  # Check every minute
                
            except asyncio.CancelledError:
                logger.info("Scaling monitor loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in scaling monitor loop: {str(e)}")
                await asyncio.sleep(30)  # Wait before retrying
                
    async def _metrics_collection_loop(self):
        """Collect and aggregate metrics."""
        while not self.stop_event.is_set():
            try:
                # Collect system metrics
                system_metrics = {
                    "timestamp": datetime.now().isoformat(),
                    "cpu_percent": psutil.cpu_percent(interval=1),
                    "memory_percent": psutil.virtual_memory().percent,
                    "disk_usage_percent": psutil.disk_usage("/").percent,
                    "network_io": psutil.net_io_counters()._asdict() if hasattr(psutil.net_io_counters(), '_asdict') else {}
                }
                
                # Collect service-specific metrics
                service_metrics = {}
                for service_type, instances in self.service_registry.service_instances.items():
                    if instances:
                        avg_cpu = sum(inst.cpu_usage for inst in instances) / len(instances)
                        avg_memory = sum(inst.memory_usage for inst in instances) / len(instances)
                        avg_response_time = sum(inst.response_time_avg for inst in instances) / len(instances) if instances else 0
                        
                        service_metrics[service_type] = {
                            "instance_count": len(instances),
                            "avg_cpu_percent": avg_cpu,
                            "avg_memory_percent": avg_memory,
                            "avg_response_time_ms": avg_response_time,
                            "total_requests_per_minute": sum(inst.requests_per_minute for inst in instances)
                        }
                
                # In a real system, these metrics would be sent to a monitoring system
                # For this demo, we'll just log them periodically
                logger.debug(f"System metrics: {system_metrics}")
                logger.debug(f"Service metrics: {service_metrics}")
                
                # Wait before next collection
                await asyncio.sleep(30)  # Collect every 30 seconds
                
            except asyncio.CancelledError:
                logger.info("Metrics collection loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in metrics collection loop: {str(e)}")
                await asyncio.sleep(10)  # Wait before retrying
                
    def get_service_status(self, service_type: ServiceType = None) -> Dict[str, Any]:
        """Get status of services."""
        if service_type:
            instances = self.service_registry.get_service_instances(service_type)
            healthy_instances = self.service_registry.get_healthy_instances(service_type)
            
            return {
                "service_type": service_type.value,
                "total_instances": len(instances),
                "healthy_instances": len(healthy_instances),
                "unhealthy_instances": len(instances) - len(healthy_instances),
                "instances": [
                    {
                        "instance_id": inst.instance_id,
                        "status": inst.status.value,
                        "host": f"{inst.host}:{inst.port}",
                        "cpu_usage": inst.cpu_usage,
                        "memory_usage": inst.memory_usage,
                        "response_time_avg": inst.response_time_avg
                    }
                    for inst in instances
                ]
            }
        else:
            # Return status for all service types
            all_status = {}
            for service_type in ServiceType:
                all_status[service_type.value] = self.get_service_status(service_type)
            return all_status
            
    def get_load_balancing_info(self) -> Dict[str, Any]:
        """Get information about load balancing."""
        if not self.load_balancer:
            return {}
            
        return {
            "strategy": self.load_balancer.config.strategy.value,
            "sticky_sessions": self.load_balancer.config.sticky_sessions,
            "health_check_interval": self.load_balancer.config.health_check_interval
        }
        
    def get_auto_scaling_info(self) -> Dict[str, Any]:
        """Get information about auto-scaling."""
        if not self.auto_scaler:
            return {}
            
        return {
            "policy": self.auto_scaler.config.policy.value,
            "min_instances": self.auto_scaler.config.min_instances,
            "max_instances": self.auto_scaler.config.max_instances,
            "scale_up_threshold": self.auto_scaler.config.scale_up_threshold,
            "scale_down_threshold": self.auto_scaler.config.scale_down_threshold,
            "cooldown_period": self.auto_scaler.config.cooldown_period
        }
        
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the microservice architecture."""
        service_status = self.get_service_status()
        
        # Calculate aggregate metrics
        total_instances = 0
        total_healthy = 0
        avg_cpu = 0
        avg_memory = 0
        avg_response_time = 0
        total_requests = 0
        
        for service_info in service_status.values():
            if isinstance(service_info, dict) and "instances" in service_info:
                instances = service_info["instances"]
                total_instances += len(instances)
                total_healthy += service_info["healthy_instances"]
                
                if instances:
                    avg_cpu += sum(inst.get("cpu_usage", 0) for inst in instances) / len(instances)
                    avg_memory += sum(inst.get("memory_usage", 0) for inst in instances) / len(instances)
                    avg_response_time += sum(inst.get("response_time_avg", 0) for inst in instances) / len(instances)
                    total_requests += sum(inst.get("requests_per_minute", 0) for inst in instances)
        
        avg_cpu = avg_cpu / len([s for s in service_status.values() if isinstance(s, dict) and s.get("instances")]) if total_instances > 0 else 0
        avg_memory = avg_memory / len([s for s in service_status.values() if isinstance(s, dict) and s.get("instances")]) if total_instances > 0 else 0
        avg_response_time = avg_response_time / len([s for s in service_status.values() if isinstance(s, dict) and s.get("instances")]) if total_instances > 0 else 0
        
        return {
            "total_services": len(service_status),
            "total_instances": total_instances,
            "healthy_instances": total_healthy,
            "unhealthy_instances": total_instances - total_healthy,
            "health_percentage": (total_healthy / total_instances * 100) if total_instances > 0 else 0,
            "average_cpu_usage": avg_cpu,
            "average_memory_usage": avg_memory,
            "average_response_time_ms": avg_response_time,
            "total_requests_per_minute": total_requests,
            "scaling_events_count": len(self.auto_scaler.scaling_history) if self.auto_scaler else 0
        }


# Convenience function for easy use
async def create_microservice_manager(
    load_balancing_config: LoadBalancingConfig = None,
    auto_scaling_config: AutoScalingConfig = None
) -> MicroserviceManager:
    """
    Convenience function to create a microservice manager.
    
    Args:
        load_balancing_config: Configuration for load balancing
        auto_scaling_config: Configuration for auto-scaling
        
    Returns:
        MicroserviceManager instance
    """
    manager = MicroserviceManager()
    await manager.initialize(load_balancing_config, auto_scaling_config)
    return manager