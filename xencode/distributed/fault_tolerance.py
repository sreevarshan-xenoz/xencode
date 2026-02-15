"""
Fault Tolerance and Recovery System
Implements FaultToleranceManager for system resilience, automatic node failure detection,
workload redistribution mechanisms, and system state recovery protocols.
"""

import asyncio
import logging
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Callable, Set
from enum import Enum
from datetime import datetime, timedelta
import secrets
import copy
import threading
from collections import deque
import time


logger = logging.getLogger(__name__)


class FailureType(Enum):
    """Types of failures that can occur in the system."""
    NODE_FAILURE = "node_failure"
    NETWORK_PARTITION = "network_partition"
    DATA_CORRUPTION = "data_corruption"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    SOFTWARE_ERROR = "software_error"
    HARDWARE_ERROR = "hardware_error"


class RecoveryStrategy(Enum):
    """Strategies for recovering from failures."""
    FAIL_OVER = "fail_over"
    RESTART_SERVICE = "restart_service"
    ROLLBACK_STATE = "rollback_state"
    REBUILD_COMPONENT = "rebuild_component"
    MIGRATE_WORKLOAD = "migrate_workload"


class ComponentState(Enum):
    """States of system components."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    RECOVERING = "recovering"
    RECOVERED = "recovered"


@dataclass
class FailureEvent:
    """Represents a failure event in the system."""
    event_id: str
    component_id: str
    failure_type: FailureType
    timestamp: datetime
    severity: int  # 1-5 scale, 5 being most severe
    description: str
    recovery_strategy: Optional[RecoveryStrategy]
    resolved: bool = False
    resolution_time: Optional[datetime] = None


@dataclass
class SystemStateSnapshot:
    """A snapshot of the system state for recovery purposes."""
    snapshot_id: str
    timestamp: datetime
    components_state: Dict[str, ComponentState]
    active_workloads: Dict[str, List[str]]  # component_id -> [workload_ids]
    resource_utilization: Dict[str, float]  # component_id -> utilization %
    network_topology: Dict[str, List[str]]  # node_id -> [connected_node_ids]
    metadata: Dict[str, Any]


@dataclass
class RecoveryPlan:
    """A plan for recovering from a failure."""
    plan_id: str
    failure_event_id: str
    affected_components: List[str]
    recovery_strategy: RecoveryStrategy
    priority: int  # 1-5 scale, 5 being highest priority
    estimated_recovery_time: float  # in seconds
    steps: List[Dict[str, Any]]  # List of recovery steps
    status: str  # planned, in_progress, completed, failed


class HealthMonitor:
    """Monitors the health of system components."""
    
    def __init__(self, check_interval: int = 10):
        self.check_interval = check_interval
        self.component_health: Dict[str, ComponentState] = {}
        self.last_check_time: Dict[str, datetime] = {}
        self.health_callbacks: Dict[str, Callable] = {}
        self.monitoring_task = None
        self.stop_event = asyncio.Event()
        
    async def start_monitoring(self):
        """Start the health monitoring loop."""
        self.monitoring_task = asyncio.create_task(self._monitor_loop())
        
    async def stop_monitoring(self):
        """Stop the health monitoring loop."""
        self.stop_event.set()
        if self.monitoring_task:
            await self.monitoring_task
            
    async def _monitor_loop(self):
        """Main monitoring loop."""
        while not self.stop_event.is_set():
            try:
                # Check health of all registered components
                for component_id in list(self.component_health.keys()):
                    await self._check_component_health(component_id)
                    
                # Wait for the next check interval or stop event
                try:
                    await asyncio.wait_for(self.stop_event.wait(), timeout=self.check_interval)
                except asyncio.TimeoutError:
                    # Timeout means it's time for the next check, continue the loop
                    continue
                    
            except asyncio.CancelledError:
                logger.info("Health monitoring loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in health monitoring loop: {str(e)}")
                await asyncio.sleep(5)  # Wait before retrying
                
    async def _check_component_health(self, component_id: str):
        """Check the health of a specific component."""
        try:
            # Use registered callback if available
            if component_id in self.health_callbacks:
                is_healthy = await self.health_callbacks[component_id]()
                new_state = ComponentState.HEALTHY if is_healthy else ComponentState.DEGRADED
            else:
                # Default health check - just mark as healthy if we can reach it
                new_state = ComponentState.HEALTHY
                
            # Update health status
            old_state = self.component_health.get(component_id, ComponentState.HEALTHY)
            self.component_health[component_id] = new_state
            self.last_check_time[component_id] = datetime.now()
            
            # Log state changes
            if old_state != new_state:
                logger.info(f"Component {component_id} state changed: {old_state.value} -> {new_state.value}")
                
        except Exception as e:
            logger.error(f"Error checking health of component {component_id}: {str(e)}")
            self.component_health[component_id] = ComponentState.FAILED
            self.last_check_time[component_id] = datetime.now()
            
    def register_component(self, component_id: str, health_callback: Optional[Callable] = None):
        """Register a component for health monitoring."""
        self.component_health[component_id] = ComponentState.HEALTHY
        self.health_callbacks[component_id] = health_callback
        
    def unregister_component(self, component_id: str):
        """Unregister a component from health monitoring."""
        if component_id in self.component_health:
            del self.component_health[component_id]
        if component_id in self.health_callbacks:
            del self.health_callbacks[component_id]
            
    def get_component_health(self, component_id: str) -> Optional[ComponentState]:
        """Get the health status of a component."""
        return self.component_health.get(component_id)
        
    def is_component_healthy(self, component_id: str) -> bool:
        """Check if a component is healthy."""
        state = self.get_component_health(component_id)
        return state == ComponentState.HEALTHY


class WorkloadRedistributor:
    """Redistributes workloads when failures occur."""
    
    def __init__(self):
        self.workload_assignments: Dict[str, str] = {}  # workload_id -> component_id
        self.component_capacity: Dict[str, int] = {}   # component_id -> capacity
        self.component_load: Dict[str, int] = {}       # component_id -> current_load
        
    def assign_workload(self, workload_id: str, component_id: str) -> bool:
        """Assign a workload to a component."""
        if component_id not in self.component_capacity:
            logger.error(f"Component {component_id} not registered")
            return False
            
        if self.component_load.get(component_id, 0) >= self.component_capacity[component_id]:
            logger.error(f"Component {component_id} is at capacity")
            return False
            
        self.workload_assignments[workload_id] = component_id
        self.component_load[component_id] = self.component_load.get(component_id, 0) + 1
        
        logger.info(f"Assigned workload {workload_id} to component {component_id}")
        return True
        
    def remove_workload_assignment(self, workload_id: str) -> Optional[str]:
        """Remove a workload assignment and return the previous component."""
        if workload_id not in self.workload_assignments:
            return None
            
        old_component = self.workload_assignments.pop(workload_id)
        self.component_load[old_component] = max(0, self.component_load[old_component] - 1)
        
        logger.info(f"Removed workload {workload_id} from component {old_component}")
        return old_component
        
    def redistribute_workloads(self, failed_component: str, backup_components: List[str]) -> Dict[str, str]:
        """Redistribute workloads from a failed component to backup components."""
        redistributed = {}
        
        # Get workloads assigned to the failed component
        workloads_to_move = [
            wl_id for wl_id, comp_id in self.workload_assignments.items()
            if comp_id == failed_component
        ]
        
        if not workloads_to_move:
            logger.info(f"No workloads to redistribute from {failed_component}")
            return redistributed
            
        # Distribute workloads among backup components
        for i, workload_id in enumerate(workloads_to_move):
            target_component = backup_components[i % len(backup_components)]
            
            # Check if target component has capacity
            if self.component_load.get(target_component, 0) < self.component_capacity.get(target_component, 0):
                # Move the workload
                old_component = self.remove_workload_assignment(workload_id)
                success = self.assign_workload(workload_id, target_component)
                
                if success:
                    redistributed[workload_id] = target_component
                    logger.info(f"Redistributed workload {workload_id} from {old_component} to {target_component}")
                else:
                    logger.error(f"Failed to redistribute workload {workload_id} to {target_component}")
            else:
                logger.error(f"Backup component {target_component} is at capacity, cannot accept workload {workload_id}")
                
        return redistributed
        
    def register_component_capacity(self, component_id: str, capacity: int):
        """Register the capacity of a component."""
        self.component_capacity[component_id] = capacity
        if component_id not in self.component_load:
            self.component_load[component_id] = 0


class StateRecoveryManager:
    """Manages system state recovery from snapshots."""
    
    def __init__(self):
        self.snapshots: Dict[str, SystemStateSnapshot] = {}
        self.state_history: deque = deque(maxlen=10)  # Keep last 10 states
        self.recovery_points: Dict[str, SystemStateSnapshot] = {}  # component_id -> recovery_point
        
    def create_snapshot(self) -> SystemStateSnapshot:
        """Create a snapshot of the current system state."""
        snapshot_id = f"snapshot_{secrets.token_hex(8)}"
        
        # This would capture the actual system state in a real implementation
        # For now, we'll create a mock snapshot
        snapshot = SystemStateSnapshot(
            snapshot_id=snapshot_id,
            timestamp=datetime.now(),
            components_state={},  # Would be populated with actual component states
            active_workloads={},  # Would be populated with actual workload assignments
            resource_utilization={},  # Would be populated with actual resource usage
            network_topology={},  # Would be populated with actual network topology
            metadata={"created_by": "state_recovery_manager"}
        )
        
        self.snapshots[snapshot_id] = snapshot
        self.state_history.append(snapshot)
        
        logger.info(f"Created system state snapshot: {snapshot_id}")
        return snapshot
        
    def restore_from_snapshot(self, snapshot_id: str) -> bool:
        """Restore the system state from a snapshot."""
        if snapshot_id not in self.snapshots:
            logger.error(f"Snapshot {snapshot_id} not found")
            return False
            
        snapshot = self.snapshots[snapshot_id]
        
        # In a real implementation, this would restore the actual system state
        # For now, we'll just log the restoration
        logger.info(f"Restoring system state from snapshot: {snapshot_id}")
        
        # This is where the actual restoration logic would go
        # - Restore component states
        # - Restore workload assignments  
        # - Restore network connections
        # - etc.
        
        return True
        
    def get_latest_snapshot(self) -> Optional[SystemStateSnapshot]:
        """Get the most recent system state snapshot."""
        if not self.state_history:
            return None
        return self.state_history[-1]
        
    def cleanup_old_snapshots(self, retention_hours: int = 24):
        """Clean up snapshots older than the retention period."""
        cutoff_time = datetime.now() - timedelta(hours=retention_hours)
        old_snapshots = [
            sid for sid, snap in self.snapshots.items()
            if snap.timestamp < cutoff_time
        ]
        
        for sid in old_snapshots:
            del self.snapshots[sid]
            
        logger.info(f"Cleaned up {len(old_snapshots)} old snapshots")


class RecoveryPlanner:
    """Plans recovery actions for system failures."""
    
    def __init__(self):
        self.recovery_plans: Dict[str, RecoveryPlan] = {}
        self.failure_history: List[FailureEvent] = []
        
    def create_recovery_plan(self, failure_event: FailureEvent) -> RecoveryPlan:
        """Create a recovery plan for a failure event."""
        plan_id = f"plan_{secrets.token_hex(8)}"
        
        # Determine the appropriate recovery strategy based on failure type
        if failure_event.failure_type == FailureType.NODE_FAILURE:
            strategy = RecoveryStrategy.FAI_OVER
            priority = 5  # High priority
            est_recovery_time = 30.0  # 30 seconds
            steps = [
                {"action": "detect_failed_node", "details": failure_event.component_id},
                {"action": "activate_backup_node", "details": "find_and_activate_backup"},
                {"action": "redistribute_workloads", "details": "move_workloads_to_backup"},
                {"action": "verify_recovery", "details": "confirm_backup_is_operational"}
            ]
        elif failure_event.failure_type == FailureType.DATA_CORRUPTION:
            strategy = RecoveryStrategy.ROLLBACK_STATE
            priority = 4  # High priority
            est_recovery_time = 120.0  # 2 minutes
            steps = [
                {"action": "isolate_corrupted_data", "details": failure_event.component_id},
                {"action": "restore_from_backup", "details": "find_recent_clean_backup"},
                {"action": "validate_restored_data", "details": "checksum_verification"},
                {"action": "resume_operations", "details": "bring_component_back_online"}
            ]
        elif failure_event.failure_type == FailureType.RESOURCE_EXHAUSTION:
            strategy = RecoveryStrategy.MIGRATE_WORKLOAD
            priority = 3  # Medium priority
            est_recovery_time = 60.0  # 1 minute
            steps = [
                {"action": "identify_overloaded_component", "details": failure_event.component_id},
                {"action": "find_underutilized_resources", "details": "scan_available_nodes"},
                {"action": "migrate_workloads", "details": "move_processes_to_new_nodes"},
                {"action": "monitor_stabilization", "details": "ensure_system_stability"}
            ]
        else:
            # Default strategy
            strategy = RecoveryStrategy.RESTART_SERVICE
            priority = 2  # Medium priority
            est_recovery_time = 15.0  # 15 seconds
            steps = [
                {"action": "restart_component", "details": failure_event.component_id},
                {"action": "verify_restart_success", "details": "check_component_responsiveness"},
                {"action": "resume_normal_operations", "details": "restore_workload_assignments"}
            ]
        
        plan = RecoveryPlan(
            plan_id=plan_id,
            failure_event_id=failure_event.event_id,
            affected_components=[failure_event.component_id],
            recovery_strategy=strategy,
            priority=priority,
            estimated_recovery_time=est_recovery_time,
            steps=steps,
            status="planned"
        )
        
        self.recovery_plans[plan_id] = plan
        logger.info(f"Created recovery plan {plan_id} for failure {failure_event.event_id}")
        
        return plan
        
    def execute_recovery_plan(self, plan_id: str) -> bool:
        """Execute a recovery plan."""
        if plan_id not in self.recovery_plans:
            logger.error(f"Recovery plan {plan_id} not found")
            return False
            
        plan = self.recovery_plans[plan_id]
        plan.status = "in_progress"
        
        logger.info(f"Executing recovery plan {plan_id}")
        
        try:
            # Execute each step in the plan
            for step in plan.steps:
                logger.debug(f"Executing step: {step['action']} - {step['details']}")
                
                # In a real implementation, this would execute the actual recovery step
                # For now, we'll simulate with a delay
                time.sleep(0.1)  # Simulate time taken for each step
                
            plan.status = "completed"
            logger.info(f"Successfully completed recovery plan {plan_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error executing recovery plan {plan_id}: {str(e)}")
            plan.status = "failed"
            return False


class FaultToleranceManager:
    """
    Fault tolerance manager for system resilience with automatic failure detection,
    workload redistribution, and state recovery protocols.
    """
    
    def __init__(self, recovery_timeout: int = 300):  # 5 minutes
        self.health_monitor = HealthMonitor()
        self.workload_redistributor = WorkloadRedistributor()
        self.state_recovery_manager = StateRecoveryManager()
        self.recovery_planner = RecoveryPlanner()
        self.failure_log: List[FailureEvent] = []
        self.backup_components: Dict[str, List[str]] = {}  # component_id -> [backup_component_ids]
        self.recovery_timeout = recovery_timeout
        self.active_recovery_tasks: Dict[str, asyncio.Task] = {}
        self.failure_detection_enabled = True
        
    async def initialize(self):
        """Initialize the fault tolerance manager."""
        await self.health_monitor.start_monitoring()
        logger.info("Fault tolerance manager initialized")
        
    async def shutdown(self):
        """Shutdown the fault tolerance manager."""
        await self.health_monitor.stop_monitoring()
        
        # Cancel any active recovery tasks
        for task in self.active_recovery_tasks.values():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
                
        logger.info("Fault tolerance manager shutdown")
        
    def register_component(
        self, 
        component_id: str, 
        capacity: int = 10, 
        health_callback: Optional[Callable] = None
    ):
        """Register a component for fault tolerance monitoring."""
        self.health_monitor.register_component(component_id, health_callback)
        self.workload_redistributor.register_component_capacity(component_id, capacity)
        
        logger.info(f"Registered component {component_id} with capacity {capacity}")
        
    def set_backup_components(self, component_id: str, backup_ids: List[str]):
        """Set backup components for a primary component."""
        self.backup_components[component_id] = backup_ids
        logger.info(f"Set backups for {component_id}: {backup_ids}")
        
    async def detect_failure(self, component_id: str, failure_type: FailureType, description: str = ""):
        """Manually detect a failure (normally this would be automatic)."""
        if not self.failure_detection_enabled:
            return
            
        event_id = f"failure_{secrets.token_hex(8)}"
        
        # Determine severity based on failure type
        severity_map = {
            FailureType.NODE_FAILURE: 5,
            FailureType.NETWORK_PARTITION: 4,
            FailureType.DATA_CORRUPTION: 4,
            FailureType.RESOURCE_EXHAUSTION: 3,
            FailureType.SOFTWARE_ERROR: 2,
            FailureType.HARDWARE_ERROR: 5
        }
        severity = severity_map.get(failure_type, 3)
        
        failure_event = FailureEvent(
            event_id=event_id,
            component_id=component_id,
            failure_type=failure_type,
            timestamp=datetime.now(),
            severity=severity,
            description=description,
            recovery_strategy=None  # Will be set by recovery planner
        )
        
        self.failure_log.append(failure_event)
        
        logger.error(f"Detected failure in {component_id}: {failure_type.value} - {description}")
        
        # Create and execute recovery plan
        recovery_plan = self.recovery_planner.create_recovery_plan(failure_event)
        failure_event.recovery_strategy = recovery_plan.recovery_strategy
        
        # Execute recovery asynchronously
        recovery_task = asyncio.create_task(self._execute_recovery_for_event(failure_event, recovery_plan))
        self.active_recovery_tasks[event_id] = recovery_task
        
        # Clean up task when done
        recovery_task.add_done_callback(lambda t: self.active_recovery_tasks.pop(event_id, None))
        
    async def _execute_recovery_for_event(self, failure_event: FailureEvent, recovery_plan: RecoveryPlan):
        """Execute recovery for a specific failure event."""
        try:
            success = self.recovery_planner.execute_recovery_plan(recovery_plan.plan_id)
            
            if success:
                failure_event.resolved = True
                failure_event.resolution_time = datetime.now()
                logger.info(f"Successfully recovered from failure {failure_event.event_id}")
            else:
                logger.error(f"Failed to recover from failure {failure_event.event_id}")
                
        except Exception as e:
            logger.error(f"Error during recovery from failure {failure_event.event_id}: {str(e)}")
            
    def get_component_health(self, component_id: str) -> Optional[ComponentState]:
        """Get the health status of a component."""
        return self.health_monitor.get_component_health(component_id)
        
    def is_component_healthy(self, component_id: str) -> bool:
        """Check if a component is healthy."""
        return self.health_monitor.is_component_healthy(component_id)
        
    def assign_workload(self, workload_id: str, component_id: str) -> bool:
        """Assign a workload to a component."""
        return self.workload_redistributor.assign_workload(workload_id, component_id)
        
    def trigger_manual_recovery(self, component_id: str) -> bool:
        """Trigger manual recovery for a component."""
        # Check if the component is actually failed
        state = self.get_component_health(component_id)
        if state != ComponentState.FAILED:
            logger.warning(f"Component {component_id} is not in FAILED state, current state: {state.value}")
            return False
            
        # Treat this as a node failure
        asyncio.create_task(
            self.detect_failure(component_id, FailureType.NODE_FAILURE, "Manual recovery triggered")
        )
        return True
        
    async def redistribute_workloads_after_failure(self, failed_component: str) -> Dict[str, str]:
        """Manually redistribute workloads after a component failure."""
        backup_components = self.backup_components.get(failed_component, [])
        if not backup_components:
            logger.warning(f"No backup components configured for {failed_component}")
            return {}
            
        return self.workload_redistributor.redistribute_workloads(failed_component, backup_components)
        
    def create_system_snapshot(self) -> SystemStateSnapshot:
        """Create a snapshot of the current system state."""
        return self.state_recovery_manager.create_snapshot()
        
    def restore_system_state(self, snapshot_id: str) -> bool:
        """Restore the system state from a snapshot."""
        return self.state_recovery_manager.restore_from_snapshot(snapshot_id)
        
    def get_recovery_status(self) -> Dict[str, Any]:
        """Get the current status of recovery operations."""
        active_recovery_count = len(self.active_recovery_tasks)
        failed_components = [
            comp_id for comp_id, state in self.health_monitor.component_health.items()
            if state == ComponentState.FAILED
        ]
        
        return {
            "active_recovery_tasks": active_recovery_count,
            "failed_components": failed_components,
            "total_failures_detected": len(self.failure_log),
            "recent_failures": [
                {
                    "event_id": fe.event_id,
                    "component": fe.component_id,
                    "type": fe.failure_type.value,
                    "time": fe.timestamp.isoformat(),
                    "resolved": fe.resolved
                }
                for fe in self.failure_log[-5:]  # Last 5 failures
            ]
        }
        
    def enable_failure_detection(self):
        """Enable automatic failure detection."""
        self.failure_detection_enabled = True
        logger.info("Failure detection enabled")
        
    def disable_failure_detection(self):
        """Disable automatic failure detection."""
        self.failure_detection_enabled = False
        logger.info("Failure detection disabled")


# Convenience function for easy use
async def setup_fault_tolerance(
    components: List[Dict[str, Any]]
) -> FaultToleranceManager:
    """
    Convenience function to set up fault tolerance for a set of components.
    
    Args:
        components: List of component specifications with keys:
                   - 'id': Component ID
                   - 'capacity': Component capacity
                   - 'backups': List of backup component IDs (optional)
                   - 'health_callback': Health check callback (optional)
        
    Returns:
        FaultToleranceManager instance
    """
    manager = FaultToleranceManager()
    await manager.initialize()
    
    for comp_spec in components:
        comp_id = comp_spec['id']
        capacity = comp_spec.get('capacity', 10)
        health_callback = comp_spec.get('health_callback')
        
        manager.register_component(comp_id, capacity, health_callback)
        
        # Set up backups if provided
        backups = comp_spec.get('backups')
        if backups:
            manager.set_backup_components(comp_id, backups)
    
    return manager