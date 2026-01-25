#!/usr/bin/env python3
"""
Advanced Multi-Agent Collaboration System for Xencode

Implements sophisticated inter-agent communication, coordination, and collaboration
mechanisms for complex development tasks.
"""

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Set
from datetime import datetime
import logging
from concurrent.futures import ThreadPoolExecutor

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.tree import Tree

console = Console()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentRole(Enum):
    """Agent roles in the collaboration system"""
    COORDINATOR = "coordinator"
    SPECIALIST = "specialist"
    GENERALIST = "generalist"
    MONITOR = "monitor"
    VALIDATOR = "validator"
    RESOURCE_MANAGER = "resource_manager"


class MessageType(Enum):
    """Types of messages in the communication protocol"""
    TASK_ASSIGNMENT = "task_assignment"
    TASK_UPDATE = "task_update"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    REQUEST_HELP = "request_help"
    OFFER_ASSISTANCE = "offer_assistance"
    STATUS_REPORT = "status_report"
    RESOURCE_REQUEST = "resource_request"
    RESOURCE_RESPONSE = "resource_response"
    CONSULTATION_REQUEST = "consultation_request"
    CONSULTATION_RESPONSE = "consultation_response"
    COORDINATION_BROADCAST = "coordination_broadcast"


class TaskStatus(Enum):
    """Status of tasks in the collaboration system"""
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Message:
    """Communication message between agents"""
    id: str
    sender_id: str
    receiver_id: str
    message_type: MessageType
    content: Dict[str, Any]
    timestamp: float
    correlation_id: Optional[str] = None
    priority: int = 0  # 0-10, 10 is highest priority


@dataclass
class Task:
    """Collaborative task with dependencies and requirements"""
    id: str
    description: str
    assigned_agent: Optional[str] = None
    status: TaskStatus = TaskStatus.PENDING
    dependencies: List[str] = field(default_factory=list)
    required_resources: Dict[str, Any] = field(default_factory=dict)
    assigned_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    priority: int = 5  # 0-10, 10 is highest priority
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentCapabilities:
    """Capabilities and skills of an agent"""
    skills: List[str]
    processing_power: int  # 1-10 scale
    available_resources: Dict[str, Any]
    specialization: str
    max_concurrent_tasks: int = 3


class CommunicationProtocol:
    """Inter-Agent Communication Protocol Implementation"""

    def __init__(self):
        self.message_queue = asyncio.Queue()
        self.subscribers: Dict[str, List[Callable]] = {}
        self.message_history: List[Message] = []
        self.executor = ThreadPoolExecutor(max_workers=10)

    async def send_message(self, message: Message):
        """Send a message to the communication bus"""
        await self.message_queue.put(message)
        self.message_history.append(message)
        
        # Notify subscribers
        if message.receiver_id in self.subscribers:
            for callback in self.subscribers[message.receiver_id]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(message)
                    else:
                        await asyncio.get_event_loop().run_in_executor(
                            self.executor, callback, message
                        )
                except Exception as e:
                    logger.error(f"Error notifying subscriber: {e}")

    def subscribe(self, agent_id: str, callback: Callable):
        """Subscribe an agent to receive messages"""
        if agent_id not in self.subscribers:
            self.subscribers[agent_id] = []
        self.subscribers[agent_id].append(callback)

    async def get_messages_for_agent(self, agent_id: str, timeout: float = 1.0) -> List[Message]:
        """Get messages for a specific agent"""
        messages = []
        try:
            while True:
                message = await asyncio.wait_for(self.message_queue.get(), timeout=0.1)
                if message.receiver_id == agent_id or message.receiver_id == "broadcast":
                    messages.append(message)
                else:
                    # Put it back if not for this agent
                    await self.message_queue.put(message)
                    break
        except asyncio.TimeoutError:
            pass
        
        return messages

    def get_message_stats(self) -> Dict[str, Any]:
        """Get communication statistics"""
        return {
            "total_messages": len(self.message_history),
            "queued_messages": self.message_queue.qsize(),
            "active_subscribers": len(self.subscribers),
            "message_types": {
                msg_type.value: len([m for m in self.message_history if m.message_type == msg_type])
                for msg_type in MessageType
            }
        }


class AgentMemory:
    """Shared memory system for agents"""

    def __init__(self):
        self.shared_memory: Dict[str, Any] = {}
        self.agent_memories: Dict[str, Dict[str, Any]] = {}
        self.access_logs: List[Dict[str, Any]] = []
        self.locks: Dict[str, asyncio.Lock] = {}

    def get_lock(self, key: str) -> asyncio.Lock:
        """Get a lock for a specific memory key"""
        if key not in self.locks:
            self.locks[key] = asyncio.Lock()
        return self.locks[key]

    async def write_shared(self, key: str, value: Any, agent_id: str):
        """Write to shared memory"""
        async with self.get_lock(key):
            self.shared_memory[key] = value
            self.access_logs.append({
                "timestamp": time.time(),
                "agent_id": agent_id,
                "operation": "write",
                "key": key,
                "value_type": type(value).__name__
            })

    async def read_shared(self, key: str, agent_id: str) -> Optional[Any]:
        """Read from shared memory"""
        async with self.get_lock(key):
            value = self.shared_memory.get(key)
            self.access_logs.append({
                "timestamp": time.time(),
                "agent_id": agent_id,
                "operation": "read",
                "key": key,
                "value_type": type(value).__name__ if value else None
            })
            return value

    async def write_agent_memory(self, agent_id: str, key: str, value: Any):
        """Write to agent-specific memory"""
        if agent_id not in self.agent_memories:
            self.agent_memories[agent_id] = {}
        self.agent_memories[agent_id][key] = value

    async def read_agent_memory(self, agent_id: str, key: str) -> Optional[Any]:
        """Read from agent-specific memory"""
        agent_memory = self.agent_memories.get(agent_id, {})
        return agent_memory.get(key)

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory system statistics"""
        return {
            "shared_keys": len(self.shared_memory),
            "agent_memories": len(self.agent_memories),
            "total_accesses": len(self.access_logs),
            "recent_accesses": len([log for log in self.access_logs 
                                  if time.time() - log["timestamp"] < 300])  # Last 5 minutes
        }


class Agent:
    """Base agent class with collaboration capabilities"""

    def __init__(self, agent_id: str, role: AgentRole, capabilities: AgentCapabilities):
        self.id = agent_id
        self.role = role
        self.capabilities = capabilities
        self.communication_protocol: Optional[CommunicationProtocol] = None
        self.memory: Optional[AgentMemory] = None
        self.assigned_tasks: List[Task] = []
        self.status = "idle"
        self.last_heartbeat = time.time()
        self.collaboration_history: List[Dict[str, Any]] = []

    def connect(self, communication_protocol: CommunicationProtocol, memory: AgentMemory):
        """Connect the agent to communication and memory systems"""
        self.communication_protocol = communication_protocol
        self.memory = memory
        communication_protocol.subscribe(self.id, self.receive_message)

    async def receive_message(self, message: Message):
        """Handle incoming messages"""
        self.collaboration_history.append({
            "timestamp": time.time(),
            "message_type": message.message_type.value,
            "sender": message.sender_id,
            "content_keys": list(message.content.keys()) if isinstance(message.content, dict) else []
        })

        # Route message to appropriate handler
        handler_name = f"handle_{message.message_type.value.replace('-', '_')}"
        handler = getattr(self, handler_name, self.handle_generic_message)
        
        try:
            if asyncio.iscoroutinefunction(handler):
                await handler(message)
            else:
                handler(message)
        except Exception as e:
            logger.error(f"Error handling message {message.id}: {e}")
            await self.send_error_response(message, str(e))

    async def handle_task_assignment(self, message: Message):
        """Handle task assignment message"""
        task_data = message.content
        task = Task(
            id=task_data["id"],
            description=task_data["description"],
            assigned_agent=self.id,
            required_resources=task_data.get("required_resources", {}),
            priority=task_data.get("priority", 5),
            metadata=task_data.get("metadata", {})
        )
        
        self.assigned_tasks.append(task)
        task.assigned_at = time.time()
        self.status = "working"
        
        # Start processing the task
        await self.process_task(task)

    async def process_task(self, task: Task):
        """Process an assigned task"""
        try:
            # Simulate task processing
            await asyncio.sleep(0.1)  # Simulated processing time
            
            # Complete the task
            task.status = TaskStatus.COMPLETED
            task.completed_at = time.time()
            task.result = f"Task {task.id} completed by {self.id}"
            
            # Send completion message
            completion_message = Message(
                id=f"completion_{uuid.uuid4()}",
                sender_id=self.id,
                receiver_id="coordinator",  # Default to coordinator
                message_type=MessageType.TASK_COMPLETED,
                content={
                    "task_id": task.id,
                    "result": task.result,
                    "processing_time": task.completed_at - task.assigned_at
                },
                timestamp=time.time(),
                correlation_id=task.id
            )
            await self.communication_protocol.send_message(completion_message)
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            await self.send_error_response(
                Message(
                    id=f"error_{uuid.uuid4()}",
                    sender_id=self.id,
                    receiver_id="coordinator",
                    message_type=MessageType.TASK_FAILED,
                    content={"task_id": task.id, "error": str(e)},
                    timestamp=time.time()
                ),
                str(e)
            )

    async def handle_generic_message(self, message: Message):
        """Handle unrecognized message types"""
        logger.warning(f"Agent {self.id} received unrecognized message type: {message.message_type}")

    async def send_error_response(self, original_message: Message, error: str):
        """Send an error response"""
        error_message = Message(
            id=f"error_{uuid.uuid4()}",
            sender_id=self.id,
            receiver_id=original_message.sender_id,
            message_type=MessageType.TASK_FAILED,
            content={"original_message_id": original_message.id, "error": error},
            timestamp=time.time(),
            correlation_id=original_message.id
        )
        await self.communication_protocol.send_message(error_message)

    async def request_help(self, task: Task, required_skills: List[str] = None):
        """Request help from other agents"""
        help_request = Message(
            id=f"help_{uuid.uuid4()}",
            sender_id=self.id,
            receiver_id="broadcast",
            message_type=MessageType.REQUEST_HELP,
            content={
                "task_id": task.id,
                "task_description": task.description,
                "required_skills": required_skills or [],
                "urgency": task.priority
            },
            timestamp=time.time()
        )
        await self.communication_protocol.send_message(help_request)

    async def offer_assistance(self, task_id: str, offering_agent_id: str):
        """Offer assistance to another agent"""
        assistance_offer = Message(
            id=f"assist_{uuid.uuid4()}",
            sender_id=self.id,
            receiver_id=offering_agent_id,
            message_type=MessageType.OFFER_ASSISTANCE,
            content={"task_id": task_id, "capabilities": self.capabilities.skills},
            timestamp=time.time()
        )
        await self.communication_protocol.send_message(assistance_offer)

    def get_agent_stats(self) -> Dict[str, Any]:
        """Get agent statistics"""
        return {
            "id": self.id,
            "role": self.role.value,
            "status": self.status,
            "assigned_tasks": len(self.assigned_tasks),
            "completed_tasks": len([t for t in self.assigned_tasks if t.status == TaskStatus.COMPLETED]),
            "failed_tasks": len([t for t in self.assigned_tasks if t.status == TaskStatus.FAILED]),
            "collaboration_count": len(self.collaboration_history),
            "last_heartbeat": self.last_heartbeat
        }


class CollaborationOrchestrator:
    """Central orchestrator for multi-agent collaboration"""

    def __init__(self):
        self.agents: Dict[str, Agent] = {}
        self.tasks: Dict[str, Task] = {}
        self.communication_protocol = CommunicationProtocol()
        self.memory = AgentMemory()
        self.active_teams: List[List[str]] = []
        self.coordination_strategies = {}
        self.running = False

    def register_agent(self, agent: Agent):
        """Register an agent with the collaboration system"""
        self.agents[agent.id] = agent
        agent.connect(self.communication_protocol, self.memory)
        logger.info(f"Registered agent: {agent.id} ({agent.role.value})")

    def create_team(self, agent_ids: List[str], team_purpose: str) -> str:
        """Create a dynamic team of agents"""
        team_id = f"team_{uuid.uuid4()}"
        team = {
            "id": team_id,
            "agents": agent_ids,
            "purpose": team_purpose,
            "formation_time": time.time(),
            "status": "active"
        }
        
        # Store team information in shared memory
        asyncio.create_task(
            self.memory.write_shared(f"team_{team_id}", team, "orchestrator")
        )
        
        self.active_teams.append(agent_ids)
        logger.info(f"Created team {team_id} with agents: {agent_ids}")
        return team_id

    async def assign_task(self, task: Task, target_agents: List[str] = None) -> str:
        """Assign a task to appropriate agents"""
        task.id = f"task_{uuid.uuid4()}"
        self.tasks[task.id] = task

        # If no specific agents requested, find suitable ones
        if not target_agents:
            target_agents = self.find_suitable_agents(task)

        if not target_agents:
            logger.warning(f"No suitable agents found for task {task.id}")
            return task.id

        # Assign to the most suitable agent
        primary_agent = target_agents[0]
        assignment_message = Message(
            id=f"assign_{uuid.uuid4()}",
            sender_id="orchestrator",
            receiver_id=primary_agent,
            message_type=MessageType.TASK_ASSIGNMENT,
            content={
                "id": task.id,
                "description": task.description,
                "required_resources": task.required_resources,
                "priority": task.priority,
                "metadata": task.metadata
            },
            timestamp=time.time()
        )
        
        await self.communication_protocol.send_message(assignment_message)
        logger.info(f"Assigned task {task.id} to agent {primary_agent}")

        return task.id

    def find_suitable_agents(self, task: Task) -> List[str]:
        """Find agents suitable for a given task"""
        suitable_agents = []
        
        for agent_id, agent in self.agents.items():
            # Check if agent has required skills
            required_skills = task.metadata.get("required_skills", [])
            if all(skill in agent.capabilities.skills for skill in required_skills):
                # Check if agent has available capacity
                active_tasks = len([t for t in agent.assigned_tasks 
                                  if t.status in [TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS]])
                
                if active_tasks < agent.capabilities.max_concurrent_tasks:
                    suitable_agents.append(agent_id)

        # Sort by processing power (descending) to get best agents first
        suitable_agents.sort(
            key=lambda aid: self.agents[aid].capabilities.processing_power,
            reverse=True
        )
        
        return suitable_agents

    async def coordinate_agents(self):
        """Main coordination loop"""
        self.running = True
        logger.info("Starting multi-agent coordination...")

        while self.running:
            try:
                # Process any pending messages
                await asyncio.sleep(0.1)  # Yield to other coroutines
                
                # Check for agent heartbeats and task status
                current_time = time.time()
                for agent_id, agent in self.agents.items():
                    # Update heartbeat
                    agent.last_heartbeat = current_time
                    
                    # Check for stuck tasks
                    for task in agent.assigned_tasks:
                        if (task.status == TaskStatus.IN_PROGRESS and 
                            current_time - task.assigned_at > 300):  # 5 minutes timeout
                            logger.warning(f"Task {task.id} timeout for agent {agent_id}")
                            task.status = TaskStatus.FAILED
                            task.error = "Task timeout"
                            
                            # Reassign if possible
                            new_agents = self.find_suitable_agents(task)
                            if new_agents and new_agents[0] != agent_id:
                                await self.assign_task(task, [new_agents[0]])

            except Exception as e:
                logger.error(f"Error in coordination loop: {e}")
                await asyncio.sleep(1)  # Brief pause before continuing

    def stop_coordination(self):
        """Stop the coordination system"""
        self.running = False

    def get_collaboration_stats(self) -> Dict[str, Any]:
        """Get overall collaboration statistics"""
        completed_tasks = len([t for t in self.tasks.values() if t.status == TaskStatus.COMPLETED])
        failed_tasks = len([t for t in self.tasks.values() if t.status == TaskStatus.FAILED])
        
        return {
            "total_agents": len(self.agents),
            "total_tasks": len(self.tasks),
            "completed_tasks": completed_tasks,
            "failed_tasks": failed_tasks,
            "active_teams": len(self.active_teams),
            "communication_stats": self.communication_protocol.get_message_stats(),
            "memory_stats": self.memory.get_memory_stats(),
            "agent_stats": {aid: agent.get_agent_stats() for aid, agent in self.agents.items()}
        }

    def display_collaboration_dashboard(self):
        """Display collaboration dashboard"""
        stats = self.get_collaboration_stats()
        
        console.print(Panel(
            f"[bold blue]Multi-Agent Collaboration Dashboard[/bold blue]\n"
            f"Total Agents: {stats['total_agents']}\n"
            f"Total Tasks: {stats['total_tasks']}\n"
            f"Completed: {stats['completed_tasks']}\n"
            f"Failed: {stats['failed_tasks']}\n"
            f"Active Teams: {stats['active_teams']}",
            title="Collaboration Overview",
            border_style="blue"
        ))

        # Display agent status table
        table = Table(title="Agent Status")
        table.add_column("Agent ID", style="cyan")
        table.add_column("Role", style="magenta")
        table.add_column("Status", style="green")
        table.add_column("Tasks", style="yellow")
        table.add_column("Skills", style="blue")

        for agent_id, agent_stats in stats['agent_stats'].items():
            agent = self.agents[agent_id]
            table.add_row(
                agent_id,
                agent.role.value,
                agent_stats['status'],
                f"{agent_stats['assigned_tasks']}/{agent_stats['completed_tasks']}",
                ", ".join(agent.capabilities.skills[:3]) + "..." if len(agent.capabilities.skills) > 3 else ", ".join(agent.capabilities.skills)
            )

        console.print(table)


async def create_sample_agents(orchestrator: CollaborationOrchestrator):
    """Create sample agents for demonstration"""
    # Coordinator agent
    coordinator_capabilities = AgentCapabilities(
        skills=["task_management", "coordination", "decision_making"],
        processing_power=8,
        available_resources={"memory": "high", "cpu": "high"},
        specialization="orchestration",
        max_concurrent_tasks=5
    )
    coordinator = Agent("coordinator_001", AgentRole.COORDINATOR, coordinator_capabilities)
    orchestrator.register_agent(coordinator)

    # Specialist agents
    coding_specialist_capabilities = AgentCapabilities(
        skills=["python", "javascript", "code_review", "debugging"],
        processing_power=7,
        available_resources={"memory": "medium", "cpu": "high"},
        specialization="software_development",
        max_concurrent_tasks=3
    )
    coding_specialist = Agent("coder_001", AgentRole.SPECIALIST, coding_specialist_capabilities)
    orchestrator.register_agent(coding_specialist)

    # Generalist agent
    generalist_capabilities = AgentCapabilities(
        skills=["research", "writing", "analysis", "planning"],
        processing_power=6,
        available_resources={"memory": "medium", "cpu": "medium"},
        specialization="general_purpose",
        max_concurrent_tasks=4
    )
    generalist = Agent("generalist_001", AgentRole.GENERALIST, generalist_capabilities)
    orchestrator.register_agent(generalist)

    # Validator agent
    validator_capabilities = AgentCapabilities(
        skills=["validation", "testing", "quality_assurance", "verification"],
        processing_power=5,
        available_resources={"memory": "low", "cpu": "medium"},
        specialization="validation",
        max_concurrent_tasks=2
    )
    validator = Agent("validator_001", AgentRole.VALIDATOR, validator_capabilities)
    orchestrator.register_agent(validator)

    # Resource manager agent
    resource_manager_capabilities = AgentCapabilities(
        skills=["resource_allocation", "monitoring", "optimization", "scheduling"],
        processing_power=9,
        available_resources={"memory": "high", "cpu": "high"},
        specialization="resource_management",
        max_concurrent_tasks=6
    )
    resource_manager = Agent("resource_mgr_001", AgentRole.RESOURCE_MANAGER, resource_manager_capabilities)
    orchestrator.register_agent(resource_manager)


async def demo_collaboration():
    """Demonstrate the multi-agent collaboration system"""
    console.print("[bold green]🚀 Initializing Multi-Agent Collaboration System[/bold green]")
    
    orchestrator = CollaborationOrchestrator()
    
    # Create sample agents
    await create_sample_agents(orchestrator)
    
    # Create a team
    team_id = orchestrator.create_team(
        ["coder_001", "generalist_001", "validator_001"],
        "Software Development Team"
    )
    
    # Create and assign tasks
    complex_task = Task(
        id="",
        description="Implement a new feature with testing and documentation",
        required_resources={"memory": "medium", "time": "long"},
        priority=8,
        metadata={"required_skills": ["python", "testing", "documentation"]}
    )
    
    task_id = await orchestrator.assign_task(complex_task)
    console.print(f"[blue]✅ Assigned task {task_id}[/blue]")
    
    # Start coordination (in background)
    coord_task = asyncio.create_task(orchestrator.coordinate_agents())
    
    # Let it run for a bit to simulate activity
    await asyncio.sleep(2)
    
    # Display dashboard
    orchestrator.display_collaboration_dashboard()
    
    # Stop coordination
    orchestrator.stop_coordination()
    await coord_task  # Wait for coordination task to complete
    
    console.print("[green]✅ Multi-Agent Collaboration Demo Completed[/green]")


if __name__ == "__main__":
    # Don't run by default to avoid requiring external dependencies
    # asyncio.run(demo_collaboration())
    pass