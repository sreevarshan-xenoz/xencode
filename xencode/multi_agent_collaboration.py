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


class MarketBasedAllocation:
    """Market-based resource allocation system for agents"""

    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.resource_market: Dict[str, Dict] = {}  # Resource pools
        self.agent_bids: Dict[str, Dict] = {}  # Bids for resources
        self.auction_history: List[Dict] = []

    async def auction_resource(self, resource_type: str, required_by: str, priority: int = 5) -> str:
        """Auction a resource to agents based on bids"""
        eligible_agents = []

        for agent_id, agent in self.orchestrator.agents.items():
            if agent_id != required_by and self._can_provide_resource(agent, resource_type):
                bid_value = self._calculate_bid_value(agent, resource_type, priority)
                eligible_agents.append((agent_id, bid_value))

        if not eligible_agents:
            return required_by  # Original agent keeps resource if no bidders

        # Sort by bid value (highest first)
        eligible_agents.sort(key=lambda x: x[1], reverse=True)
        winner = eligible_agents[0][0]

        # Record auction
        self.auction_history.append({
            "resource_type": resource_type,
            "requested_by": required_by,
            "winner": winner,
            "winning_bid": eligible_agents[0][1],
            "timestamp": time.time()
        })

        return winner

    def _can_provide_resource(self, agent: Agent, resource_type: str) -> bool:
        """Check if agent can provide the requested resource"""
        return resource_type in agent.capabilities.available_resources

    def _calculate_bid_value(self, agent: Agent, resource_type: str, priority: int) -> float:
        """Calculate bid value based on agent capabilities and current load"""
        base_value = agent.capabilities.processing_power

        # Factor in current workload
        active_tasks = len([t for t in agent.assigned_tasks
                           if t.status in [TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS]])
        workload_factor = max(0.1, 1.0 - (active_tasks / agent.capabilities.max_concurrent_tasks))

        # Factor in priority
        priority_factor = priority / 10.0

        return base_value * workload_factor * priority_factor


class NegotiationProtocol:
    """Negotiation protocols between agents"""

    def __init__(self):
        self.negotiation_sessions: Dict[str, Dict] = {}
        self.proposals_history: List[Dict] = []

    async def initiate_negotiation(self, initiator: str, responder: str,
                                 proposal: Dict[str, Any]) -> str:
        """Initiate a negotiation session between agents"""
        session_id = f"nego_{uuid.uuid4()}"

        session = {
            "id": session_id,
            "initiator": initiator,
            "responder": responder,
            "proposal": proposal,
            "counter_proposals": [],
            "status": "active",
            "created_at": time.time()
        }

        self.negotiation_sessions[session_id] = session
        return session_id

    async def submit_counter_proposal(self, session_id: str, proposer: str,
                                    counter_proposal: Dict[str, Any]) -> bool:
        """Submit a counter proposal in a negotiation"""
        if session_id not in self.negotiation_sessions:
            return False

        session = self.negotiation_sessions[session_id]
        if proposer not in [session["initiator"], session["responder"]]:
            return False

        session["counter_proposals"].append({
            "proposer": proposer,
            "proposal": counter_proposal,
            "timestamp": time.time()
        })

        return True

    async def reach_agreement(self, session_id: str, agreed_terms: Dict[str, Any]) -> bool:
        """Mark a negotiation as concluded with agreed terms"""
        if session_id not in self.negotiation_sessions:
            return False

        session = self.negotiation_sessions[session_id]
        session["status"] = "agreed"
        session["agreed_terms"] = agreed_terms
        session["concluded_at"] = time.time()

        # Add to history
        self.proposals_history.append(session.copy())

        return True

    async def reject_proposal(self, session_id: str, reason: str = "") -> bool:
        """Reject a proposal and conclude negotiation"""
        if session_id not in self.negotiation_sessions:
            return False

        session = self.negotiation_sessions[session_id]
        session["status"] = "rejected"
        session["rejection_reason"] = reason
        session["concluded_at"] = time.time()

        # Add to history
        self.proposals_history.append(session.copy())

        return True


class SwarmIntelligence:
    """Swarm intelligence behaviors for agent coordination"""

    def __init__(self):
        self.swarm_behaviors: Dict[str, Callable] = {
            "foraging": self._foraging_behavior,
            "consensus": self._consensus_behavior,
            "task_allocation": self._task_allocation_behavior
        }

    async def execute_swarm_behavior(self, behavior_type: str, agents: List[str],
                                  context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a swarm intelligence behavior"""
        if behavior_type not in self.swarm_behaviors:
            return {"error": f"Unknown behavior: {behavior_type}"}

        return await self.swarm_behaviors[behavior_type](agents, context)

    async def _foraging_behavior(self, agents: List[str], context: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate foraging behavior for resource discovery"""
        # Agents explore and share information about resources
        discoveries = []
        for agent_id in agents:
            # Simulate agent exploration
            discovery = {
                "agent": agent_id,
                "resources_found": len(agents) * 2,  # Simulated discovery
                "quality_score": context.get("quality_threshold", 0.5)
            }
            discoveries.append(discovery)

        return {
            "behavior": "foraging",
            "discoveries": discoveries,
            "best_discovery": max(discoveries, key=lambda x: x["quality_score"]) if discoveries else None
        }

    async def _consensus_behavior(self, agents: List[str], context: Dict[str, Any]) -> Dict[str, Any]:
        """Reach consensus through swarm behavior"""
        # Agents influence each other to reach agreement
        opinions = [hash(agent_id) % 100 for agent_id in agents]  # Simulated opinions
        avg_opinion = sum(opinions) / len(opinions) if opinions else 0

        return {
            "behavior": "consensus",
            "initial_opinions": opinions,
            "consensus_value": avg_opinion,
            "agreement_level": 0.8  # Simulated agreement level
        }

    async def _task_allocation_behavior(self, agents: List[str], context: Dict[str, Any]) -> Dict[str, Any]:
        """Distribute tasks using swarm intelligence"""
        tasks = context.get("tasks", [])
        allocation = {}

        for i, task in enumerate(tasks):
            # Assign task to agent based on fitness (simulated)
            agent_idx = i % len(agents)
            allocation[task["id"]] = agents[agent_idx]

        return {
            "behavior": "task_allocation",
            "allocation": allocation,
            "efficiency": 0.9  # Simulated efficiency
        }


class HumanInLoopSupervisor:
    """Human-in-the-loop supervision capabilities"""

    def __init__(self):
        self.approval_queue: List[Dict] = []
        self.human_decisions: List[Dict] = []
        self.supervision_enabled = True

    async def request_approval(self, decision_point: str, options: List[Dict],
                             context: Dict[str, Any], urgency: int = 5) -> str:
        """Request human approval for critical decisions"""
        if not self.supervision_enabled:
            # Return first option as default if supervision disabled
            return options[0]["id"] if options else ""

        approval_request = {
            "id": f"approval_{uuid.uuid4()}",
            "decision_point": decision_point,
            "options": options,
            "context": context,
            "urgency": urgency,
            "submitted_at": time.time(),
            "status": "pending"
        }

        self.approval_queue.append(approval_request)
        return approval_request["id"]

    async def record_human_decision(self, request_id: str, decision: str,
                                 feedback: str = "") -> bool:
        """Record a human decision"""
        for req in self.approval_queue:
            if req["id"] == request_id:
                req["status"] = "approved" if decision.startswith("approve") else "rejected"
                req["human_decision"] = decision
                req["feedback"] = feedback
                req["decided_at"] = time.time()

                # Move to decisions history
                self.human_decisions.append(req.copy())
                self.approval_queue.remove(req)

                return True

        return False

    def get_pending_approvals(self) -> List[Dict]:
        """Get all pending approval requests"""
        return [req for req in self.approval_queue if req["status"] == "pending"]


class CrossDomainExpertise:
    """Cross-domain expertise combination system"""

    def __init__(self):
        self.domain_knowledge: Dict[str, Dict] = {}
        self.expertise_bridge_agents: List[str] = []

    async def register_domain_expertise(self, agent_id: str, domain: str,
                                     expertise_level: float, topics: List[str]):
        """Register an agent's expertise in a specific domain"""
        if domain not in self.domain_knowledge:
            self.domain_knowledge[domain] = {}

        self.domain_knowledge[domain][agent_id] = {
            "expertise_level": expertise_level,
            "topics": topics,
            "last_updated": time.time()
        }

    async def find_cross_domain_solution(self, required_domains: List[str],
                                      problem_description: str) -> Dict[str, Any]:
        """Find solution requiring expertise from multiple domains"""
        solution_path = []

        for domain in required_domains:
            if domain in self.domain_knowledge:
                # Find best expert in this domain
                domain_experts = self.domain_knowledge[domain]
                best_expert = max(domain_experts.items(),
                                key=lambda x: x[1]["expertise_level"])

                solution_path.append({
                    "domain": domain,
                    "expert_agent": best_expert[0],
                    "expertise_level": best_expert[1]["expertise_level"],
                    "relevant_topics": best_expert[1]["topics"]
                })

        return {
            "problem": problem_description,
            "solution_path": solution_path,
            "domains_covered": required_domains,
            "feasibility_score": min(len(solution_path), len(required_domains)) / len(required_domains) if required_domains else 0
        }

    async def create_bridge_agent(self, agent_id: str, connecting_domains: List[str]):
        """Create an agent that bridges multiple domains"""
        self.expertise_bridge_agents.append({
            "id": agent_id,
            "connecting_domains": connecting_domains,
            "created_at": time.time()
        })


class CollaborationOrchestrator:
    """Central orchestrator for multi-agent collaboration"""

    def __init__(self):
        self.agents: Dict[str, Agent] = {}
        self.tasks: Dict[str, Task] = {}
        self.communication_protocol = CommunicationProtocol()
        self.memory = AgentMemory()
        self.active_teams: List[List[str]] = []
        self.coordination_strategies = {}

        # Advanced coordination components
        self.market_allocator = MarketBasedAllocation(self)
        self.negotiation_protocol = NegotiationProtocol()
        self.swarm_intelligence = SwarmIntelligence()
        self.human_supervisor = HumanInLoopSupervisor()
        self.cross_domain_expertise = CrossDomainExpertise()

        self.running = False

    def register_agent(self, agent: Agent):
        """Register an agent with the collaboration system"""
        self.agents[agent.id] = agent
        agent.connect(self.communication_protocol, self.memory)
        logger.info(f"Registered agent: {agent.id} ({agent.role.value})")

    async def allocate_resource_market_based(self, resource_type: str, requesting_agent: str,
                                          priority: int = 5) -> str:
        """Allocate resources using market-based mechanism"""
        return await self.market_allocator.auction_resource(resource_type, requesting_agent, priority)

    async def initiate_agent_negotiation(self, initiator: str, responder: str,
                                       proposal: Dict[str, Any]) -> str:
        """Initiate negotiation between agents"""
        return await self.negotiation_protocol.initiate_negotiation(initiator, responder, proposal)

    async def execute_swarm_behavior(self, behavior_type: str, participating_agents: List[str],
                                   context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute swarm intelligence behavior"""
        return await self.swarm_intelligence.execute_swarm_behavior(behavior_type, participating_agents, context)

    async def request_human_approval(self, decision_point: str, options: List[Dict],
                                   context: Dict[str, Any], urgency: int = 5) -> str:
        """Request human approval for critical decisions"""
        return await self.human_supervisor.request_approval(decision_point, options, context, urgency)

    async def register_agent_expertise(self, agent_id: str, domain: str,
                                    expertise_level: float, topics: List[str]):
        """Register an agent's domain expertise"""
        await self.cross_domain_expertise.register_domain_expertise(agent_id, domain, expertise_level, topics)

    async def find_cross_domain_solution(self, required_domains: List[str],
                                       problem_description: str) -> Dict[str, Any]:
        """Find solution requiring cross-domain expertise"""
        return await self.cross_domain_expertise.find_cross_domain_solution(required_domains, problem_description)

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
    console.print("[bold green]Initializing Multi-Agent Collaboration System[/bold green]")

    orchestrator = CollaborationOrchestrator()

    # Create sample agents
    await create_sample_agents(orchestrator)

    # Create a team
    team_id = orchestrator.create_team(
        ["coder_001", "generalist_001", "validator_001"],
        "Software Development Team"
    )

    # Register agent expertise for cross-domain functionality
    await orchestrator.register_agent_expertise("coder_001", "software_development", 0.9, ["python", "javascript", "algorithms"])
    await orchestrator.register_agent_expertise("generalist_001", "research_analysis", 0.8, ["research", "writing", "analysis"])
    await orchestrator.register_agent_expertise("validator_001", "quality_assurance", 0.85, ["testing", "validation", "verification"])

    # Create and assign tasks
    complex_task = Task(
        id="",
        description="Implement a new feature with testing and documentation",
        required_resources={"memory": "medium", "time": "long"},
        priority=8,
        metadata={"required_skills": ["python", "testing", "documentation"]}
    )

    task_id = await orchestrator.assign_task(complex_task)
    console.print(f"[blue]Assigned task {task_id}[/blue]")

    # Demonstrate market-based resource allocation
    console.print("\n[yellow]Demonstrating Market-Based Resource Allocation...[/yellow]")
    allocated_agent = await orchestrator.allocate_resource_market_based("computing_power", "coder_001", priority=9)
    console.print(f"[blue]Allocated resource to agent: {allocated_agent}[/blue]")

    # Demonstrate negotiation between agents
    console.print("\nDemonstrating Agent Negotiation...")
    negotiation_id = await orchestrator.initiate_agent_negotiation(
        "coder_001", "validator_001",
        {"resource": "testing_environment", "duration": "2_hours", "compensation": "priority_queue"}
    )
    console.print(f"[blue]Started negotiation: {negotiation_id}[/blue]")

    # Demonstrate swarm intelligence behavior
    console.print("\nDemonstrating Swarm Intelligence Behavior...")
    swarm_result = await orchestrator.execute_swarm_behavior(
        "consensus",
        ["coder_001", "generalist_001", "validator_001"],
        {"topic": "feature_priority", "options": ["high", "medium", "low"]}
    )
    console.print(f"[blue]Swarm consensus reached: {swarm_result['consensus_value']:.2f}[/blue]")

    # Demonstrate cross-domain expertise solution
    console.print("\nDemonstrating Cross-Domain Expertise...")
    cross_domain_solution = await orchestrator.find_cross_domain_solution(
        ["software_development", "research_analysis", "quality_assurance"],
        "Build a comprehensive feature with research-backed implementation and thorough validation"
    )
    console.print(f"[blue]Cross-domain solution feasibility: {cross_domain_solution['feasibility_score']:.2f}[/blue]")

    # Request human approval for critical decision
    console.print("\nDemonstrating Human-in-the-Loop Supervision...")
    approval_request_id = await orchestrator.request_human_approval(
        "deploy_to_production",
        [
            {"id": "option1", "description": "Deploy with full testing", "risk": "low"},
            {"id": "option2", "description": "Deploy with minimal testing", "risk": "high"}
        ],
        {"feature": "new_feature", "deadline": "urgent", "impact": "high"},
        urgency=8
    )
    console.print(f"[blue]Approval requested: {approval_request_id}[/blue]")

    # Start coordination (in background)
    coord_task = asyncio.create_task(orchestrator.coordinate_agents())

    # Let it run for a bit to simulate activity
    await asyncio.sleep(3)

    # Display dashboard
    orchestrator.display_collaboration_dashboard()

    # Stop coordination
    orchestrator.stop_coordination()
    await coord_task  # Wait for coordination task to complete

    console.print("[green]Multi-Agent Collaboration Demo Completed[/green]")


if __name__ == "__main__":
    # Don't run by default to avoid requiring external dependencies
    # asyncio.run(demo_collaboration())
    pass