"""
AI Model Orchestration Layer
Implements ModelOrchestrator for coordinating multiple AI systems, load balancing,
model health monitoring, failover, and unified AI API interface.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Callable, Union
from enum import Enum
import json
import time
import random
from datetime import datetime
import aiohttp
import threading
from concurrent.futures import ThreadPoolExecutor
import queue


logger = logging.getLogger(__name__)


class ModelStatus(Enum):
    """Status of AI models in the orchestration system."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    OVERLOADED = "overloaded"
    MAINTENANCE = "maintenance"


class LoadBalancingStrategy(Enum):
    """Strategies for distributing requests across models."""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    PRIORITY_BASED = "priority_based"
    PERFORMANCE_BASED = "performance_based"


@dataclass
class ModelInfo:
    """Information about an AI model in the orchestration system."""
    model_id: str
    endpoint_url: str
    api_key: Optional[str]
    provider: str
    capabilities: List[str]
    weight: int = 1  # For weighted load balancing
    status: ModelStatus = ModelStatus.HEALTHY
    current_load: int = 0
    max_concurrent_requests: int = 10
    response_time_avg: float = 0.0
    last_health_check: datetime = datetime.min
    health_check_interval: int = 30  # seconds


@dataclass
class RequestRoutingInfo:
    """Information about how a request was routed."""
    model_id: str
    routing_strategy: LoadBalancingStrategy
    timestamp: datetime
    response_time: float
    success: bool


@dataclass
class OrchestrationMetrics:
    """Metrics for the orchestration system."""
    total_requests: int
    successful_requests: int
    failed_requests: int
    average_response_time: float
    throughput: float  # requests per second
    load_distribution: Dict[str, int]  # model_id -> request count
    uptime_percentage: float


class HealthMonitor:
    """Monitors the health of AI models."""
    
    def __init__(self, check_interval: int = 30):
        self.check_interval = check_interval
        self.model_status: Dict[str, ModelStatus] = {}
        self.last_check_time: Dict[str, datetime] = {}
        self.monitoring_task = None
        self.stop_event = threading.Event()
        
    def start_monitoring(self, models: List[ModelInfo]):
        """Start health monitoring for the provided models."""
        self.models = {model.model_id: model for model in models}
        
        # Start monitoring thread
        self.monitoring_task = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitoring_task.start()
        
    def stop_monitoring(self):
        """Stop health monitoring."""
        self.stop_event.set()
        if self.monitoring_task:
            self.monitoring_task.join()
            
    def _monitor_loop(self):
        """Main monitoring loop."""
        while not self.stop_event.is_set():
            for model_id, model in self.models.items():
                try:
                    status = self._check_model_health(model)
                    self.model_status[model_id] = status
                    self.last_check_time[model_id] = datetime.now()
                except Exception as e:
                    logger.error(f"Error checking health for model {model_id}: {str(e)}")
                    self.model_status[model_id] = ModelStatus.UNAVAILABLE
                    
            # Wait for the next check interval or stop event
            if self.stop_event.wait(timeout=self.check_interval):
                break
                
    def _check_model_health(self, model: ModelInfo) -> ModelStatus:
        """Check the health of a specific model."""
        # In a real implementation, this would make an actual health check request
        # For now, we'll simulate the health check
        
        # Simulate different health statuses based on model properties
        if model.status == ModelStatus.MAINTENANCE:
            return ModelStatus.MAINTENANCE
            
        # Simulate occasional failures
        if random.random() < 0.05:  # 5% chance of failure
            return ModelStatus.UNAVAILABLE
            
        # Check if model is overloaded based on current load
        if model.current_load >= model.max_concurrent_requests:
            return ModelStatus.OVERLOADED
            
        # Otherwise, it's healthy
        return ModelStatus.HEALTHY
        
    def get_model_status(self, model_id: str) -> Optional[ModelStatus]:
        """Get the current status of a model."""
        return self.model_status.get(model_id)
        
    def is_model_healthy(self, model_id: str) -> bool:
        """Check if a model is healthy."""
        status = self.get_model_status(model_id)
        return status == ModelStatus.HEALTHY


class LoadBalancer:
    """Distributes requests across AI models using various strategies."""
    
    def __init__(self, strategy: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN):
        self.strategy = strategy
        self.model_weights = {}
        self.request_counts = {}
        self.current_index = 0
        self.priority_queue = []  # For priority-based strategy
        self.performance_scores = {}  # For performance-based strategy
        
    def set_models(self, models: List[ModelInfo]):
        """Set the available models for load balancing."""
        self.models = {model.model_id: model for model in models}
        
        # Initialize weights and counters
        for model in models:
            self.model_weights[model.model_id] = model.weight
            self.request_counts[model.model_id] = 0
            self.performance_scores[model.model_id] = 1.0  # Default performance score
            
    def select_model(self) -> Optional[str]:
        """Select a model based on the current strategy."""
        # Filter to only healthy models
        healthy_models = [
            model_id for model_id, model in self.models.items()
            if model.status == ModelStatus.HEALTHY and model.current_load < model.max_concurrent_requests
        ]
        
        if not healthy_models:
            return None  # No healthy models available
            
        if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return self._round_robin_select(healthy_models)
        elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            return self._least_connections_select(healthy_models)
        elif self.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
            return self._weighted_round_robin_select(healthy_models)
        elif self.strategy == LoadBalancingStrategy.PRIORITY_BASED:
            return self._priority_based_select(healthy_models)
        elif self.strategy == LoadBalancingStrategy.PERFORMANCE_BASED:
            return self._performance_based_select(healthy_models)
        else:
            # Default to round-robin
            return self._round_robin_select(healthy_models)
            
    def _round_robin_select(self, healthy_models: List[str]) -> str:
        """Round-robin selection of models."""
        if not healthy_models:
            return None
            
        selected = healthy_models[self.current_index % len(healthy_models)]
        self.current_index += 1
        return selected
        
    def _least_connections_select(self, healthy_models: List[str]) -> str:
        """Select the model with the least current connections."""
        if not healthy_models:
            return None
            
        return min(
            healthy_models,
            key=lambda model_id: self.models[model_id].current_load
        )
        
    def _weighted_round_robin_select(self, healthy_models: List[str]) -> str:
        """Weighted round-robin selection of models."""
        if not healthy_models:
            return None
            
        # Create a list with repeated model IDs based on their weights
        weighted_list = []
        for model_id in healthy_models:
            weight = self.model_weights.get(model_id, 1)
            weighted_list.extend([model_id] * weight)
            
        if not weighted_list:
            return self._round_robin_select(healthy_models)
            
        selected = weighted_list[self.current_index % len(weighted_list)]
        self.current_index += 1
        return selected
        
    def _priority_based_select(self, healthy_models: List[str]) -> str:
        """Select model based on priority."""
        if not healthy_models:
            return None
            
        # For simplicity, we'll use the model weight as priority
        # Higher weight means higher priority
        return max(
            healthy_models,
            key=lambda model_id: self.model_weights.get(model_id, 1)
        )
        
    def _performance_based_select(self, healthy_models: List[str]) -> str:
        """Select model based on performance scores."""
        if not healthy_models:
            return None
            
        return max(
            healthy_models,
            key=lambda model_id: self.performance_scores.get(model_id, 1.0)
        )
        
    def record_request(self, model_id: str, response_time: float, success: bool):
        """Record a request to update load balancing metrics."""
        if model_id in self.request_counts:
            self.request_counts[model_id] += 1
            
        # Update performance score based on response time and success
        if success and response_time > 0:
            # Lower response time = better performance
            # Normalize to 0-1 scale (lower is better)
            current_score = self.performance_scores.get(model_id, 1.0)
            new_score = 0.7 * current_score + 0.3 * (1.0 / (1.0 + response_time))  # Higher score for faster response
            self.performance_scores[model_id] = new_score


class FailoverManager:
    """Manages failover when primary models become unavailable."""
    
    def __init__(self):
        self.failover_chains: Dict[str, List[str]] = {}  # primary_model -> [backup_models]
        self.failure_history: Dict[str, List[datetime]] = {}  # model_id -> [failure_times]
        self.max_failures_before_blacklist = 3
        self.blacklisted_models: Dict[str, datetime] = {}  # model_id -> blacklist_until
        self.blacklist_duration = 300  # 5 minutes in seconds
        
    def set_failover_chain(self, primary_model: str, backup_models: List[str]):
        """Set the failover chain for a primary model."""
        self.failover_chains[primary_model] = backup_models
        
    def record_failure(self, model_id: str):
        """Record a model failure."""
        if model_id not in self.failure_history:
            self.failure_history[model_id] = []
            
        self.failure_history[model_id].append(datetime.now())
        
        # Check if model should be blacklisted
        recent_failures = [
            t for t in self.failure_history[model_id]
            if (datetime.now() - t).total_seconds() < self.blacklist_duration
        ]
        
        if len(recent_failures) >= self.max_failures_before_blacklist:
            self.blacklist_model(model_id)
            
    def blacklist_model(self, model_id: str):
        """Blacklist a model temporarily."""
        self.blacklisted_models[model_id] = datetime.now()
        logger.warning(f"Model {model_id} has been blacklisted due to repeated failures")
        
    def is_blacklisted(self, model_id: str) -> bool:
        """Check if a model is currently blacklisted."""
        if model_id not in self.blacklisted_models:
            return False
            
        blacklist_start = self.blacklisted_models[model_id]
        elapsed = (datetime.now() - blacklist_start).total_seconds()
        
        if elapsed >= self.blacklist_duration:
            # Remove from blacklist if duration has passed
            del self.blacklisted_models[model_id]
            return False
            
        return True
        
    def get_backup_model(self, failed_model: str) -> Optional[str]:
        """Get a backup model for a failed model."""
        if failed_model in self.failover_chains:
            backups = self.failover_chains[failed_model]
            for backup in backups:
                if not self.is_blacklisted(backup):
                    return backup
                    
        return None


class ModelOrchestrator:
    """
    AI model orchestrator that coordinates multiple AI systems, handles load balancing,
    monitors model health, manages failover, and provides a unified AI API interface.
    """
    
    def __init__(
        self, 
        load_balancing_strategy: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN
    ):
        self.models: Dict[str, ModelInfo] = {}
        self.health_monitor = HealthMonitor()
        self.load_balancer = LoadBalancer(strategy=load_balancing_strategy)
        self.failover_manager = FailoverManager()
        self.routing_history: List[RequestRoutingInfo] = []
        self.request_queue = asyncio.Queue()
        self.executor = ThreadPoolExecutor(max_workers=10)
        
        # Metrics tracking
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.response_times = []
        self.start_time = datetime.now()
        
    def register_model(self, model_info: ModelInfo):
        """Register a new AI model with the orchestrator."""
        self.models[model_info.model_id] = model_info
        
        # Update load balancer with new models
        self.load_balancer.set_models(list(self.models.values()))
        
        # Restart health monitoring with updated models
        self.health_monitor.stop_monitoring()
        self.health_monitor.start_monitoring(list(self.models.values()))
        
    def set_failover_chain(self, primary_model: str, backup_models: List[str]):
        """Set the failover chain for a primary model."""
        self.failover_manager.set_failover_chain(primary_model, backup_models)
        
    async def process_request(
        self, 
        request_data: Dict[str, Any], 
        required_capabilities: Optional[List[str]] = None,
        timeout: float = 30.0
    ) -> Dict[str, Any]:
        """
        Process a request by routing it to an appropriate model.
        
        Args:
            request_data: The request data to process
            required_capabilities: List of required model capabilities
            timeout: Request timeout in seconds
            
        Returns:
            Dictionary containing the response and metadata
        """
        self.total_requests += 1
        start_time = time.time()
        
        try:
            # Select an appropriate model
            model_id = await self._select_model(required_capabilities)
            if not model_id:
                raise Exception("No available models to handle the request")
                
            # Increment model load
            self.models[model_id].current_load += 1
            
            # Process the request with the selected model
            response = await self._call_model(model_id, request_data, timeout)
            
            # Decrement model load
            self.models[model_id].current_load -= 1
            
            # Record successful routing
            response_time = time.time() - start_time
            self.response_times.append(response_time)
            self.successful_requests += 1
            
            self.load_balancer.record_request(model_id, response_time, True)
            
            routing_info = RequestRoutingInfo(
                model_id=model_id,
                routing_strategy=self.load_balancer.strategy,
                timestamp=datetime.now(),
                response_time=response_time,
                success=True
            )
            self.routing_history.append(routing_info)
            
            return {
                "result": response,
                "model_used": model_id,
                "routing_info": routing_info,
                "success": True
            }
            
        except Exception as e:
            # Decrement model load if it was incremented
            if 'model_id' in locals():
                self.models[model_id].current_load -= 1
                
            # Record failed request
            response_time = time.time() - start_time
            self.response_times.append(response_time)
            self.failed_requests += 1
            
            # Record failure for failover management
            if 'model_id' in locals():
                self.failover_manager.record_failure(model_id)
                
            logger.error(f"Request failed: {str(e)}")
            
            # Try failover if possible
            if 'model_id' in locals():
                backup_model = self.failover_manager.get_backup_model(model_id)
                if backup_model:
                    logger.info(f"Attempting failover to backup model: {backup_model}")
                    try:
                        backup_response = await self._call_model(backup_model, request_data, timeout)
                        
                        # Record successful failover
                        failover_response_time = time.time() - start_time
                        self.response_times.append(failover_response_time)
                        self.successful_requests += 1
                        
                        self.load_balancer.record_request(backup_model, failover_response_time, True)
                        
                        routing_info = RequestRoutingInfo(
                            model_id=backup_model,
                            routing_strategy=self.load_balancer.strategy,
                            timestamp=datetime.now(),
                            response_time=failover_response_time,
                            success=True
                        )
                        self.routing_history.append(routing_info)
                        
                        return {
                            "result": backup_response,
                            "model_used": backup_model,
                            "routing_info": routing_info,
                            "failover_used": True,
                            "original_error": str(e),
                            "success": True
                        }
                    except Exception as failover_error:
                        logger.error(f"Failover also failed: {str(failover_error)}")
            
            # If no failover or failover also failed
            routing_info = RequestRoutingInfo(
                model_id=model_id if 'model_id' in locals() else "none",
                routing_strategy=self.load_balancer.strategy,
                timestamp=datetime.now(),
                response_time=response_time,
                success=False
            )
            self.routing_history.append(routing_info)
            
            return {
                "error": str(e),
                "model_used": model_id if 'model_id' in locals() else "none",
                "routing_info": routing_info,
                "success": False
            }
            
    async def _select_model(self, required_capabilities: Optional[List[str]] = None) -> Optional[str]:
        """Select an appropriate model based on capabilities and availability."""
        # First, filter models by required capabilities if specified
        candidate_models = []
        for model_id, model in self.models.items():
            if model.status != ModelStatus.HEALTHY:
                continue
                
            if required_capabilities:
                # Check if model has all required capabilities
                has_capabilities = all(cap in model.capabilities for cap in required_capabilities)
                if not has_capabilities:
                    continue
                    
            # Check if model is not overloaded
            if model.current_load >= model.max_concurrent_requests:
                continue
                
            # Check if model is not blacklisted
            if self.failover_manager.is_blacklisted(model_id):
                continue
                
            candidate_models.append(model_id)
            
        if not candidate_models:
            return None
            
        # Use load balancer to select from candidates
        # For this, we'll temporarily update the load balancer's model list
        original_models = self.load_balancer.models
        candidate_model_objects = [self.models[mid] for mid in candidate_models]
        self.load_balancer.set_models(candidate_model_objects)
        
        selected_model_id = self.load_balancer.select_model()
        
        # Restore original models
        self.load_balancer.models = original_models
        
        return selected_model_id
        
    async def _call_model(self, model_id: str, request_data: Dict[str, Any], timeout: float) -> Any:
        """Call a specific model with the request data."""
        model = self.models[model_id]
        
        # In a real implementation, this would make an actual API call to the model
        # For now, we'll simulate the call
        try:
            # Simulate API call delay
            await asyncio.sleep(random.uniform(0.1, 1.0))
            
            # Simulate response
            simulated_response = {
                "response": f"Processed by {model_id} at {datetime.now().isoformat()}",
                "model_info": {
                    "id": model.model_id,
                    "provider": model.provider,
                    "capabilities": model.capabilities
                },
                "input_data_summary": f"Received {len(str(request_data))} characters of input data"
            }
            
            return simulated_response
        except Exception as e:
            raise Exception(f"Failed to call model {model_id}: {str(e)}")
            
    def get_orchestration_metrics(self) -> OrchestrationMetrics:
        """Get metrics about the orchestration system."""
        total_time = (datetime.now() - self.start_time).total_seconds()
        throughput = self.total_requests / total_time if total_time > 0 else 0.0
        
        avg_response_time = (
            sum(self.response_times) / len(self.response_times) 
            if self.response_times else 0.0
        )
        
        uptime_percentage = (
            (self.successful_requests / self.total_requests) * 100 
            if self.total_requests > 0 else 100.0
        )
        
        # Calculate load distribution
        load_distribution = {}
        for routing_info in self.routing_history:
            model_id = routing_info.model_id
            if model_id not in load_distribution:
                load_distribution[model_id] = 0
            load_distribution[model_id] += 1
            
        return OrchestrationMetrics(
            total_requests=self.total_requests,
            successful_requests=self.successful_requests,
            failed_requests=self.failed_requests,
            average_response_time=avg_response_time,
            throughput=throughput,
            load_distribution=load_distribution,
            uptime_percentage=uptime_percentage
        )
        
    def update_model_status(self, model_id: str, new_status: ModelStatus):
        """Manually update the status of a model."""
        if model_id in self.models:
            self.models[model_id].status = new_status
            logger.info(f"Updated status for model {model_id} to {new_status.value}")
            
    def get_model_info(self, model_id: str) -> Optional[ModelInfo]:
        """Get information about a specific model."""
        return self.models.get(model_id)
        
    def get_all_model_statuses(self) -> Dict[str, ModelStatus]:
        """Get the status of all registered models."""
        return {
            model_id: self.health_monitor.get_model_status(model_id) or model.status
            for model_id, model in self.models.items()
        }
        
    async def graceful_shutdown(self):
        """Perform graceful shutdown of the orchestrator."""
        self.health_monitor.stop_monitoring()
        self.executor.shutdown(wait=True)
        logger.info("Model orchestrator shut down gracefully")


# Convenience function for easy use
async def process_with_model_orchestration(
    request_data: Dict[str, Any],
    orchestrator: Optional[ModelOrchestrator] = None,
    required_capabilities: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Convenience function to process a request through the model orchestrator.
    
    Args:
        request_data: The request data to process
        orchestrator: Optional orchestrator instance (will create one if not provided)
        required_capabilities: List of required model capabilities
        
    Returns:
        Dictionary containing the response and metadata
    """
    if orchestrator is None:
        orchestrator = ModelOrchestrator()
        
    return await orchestrator.process_request(request_data, required_capabilities)