"""
Advanced coordination strategies for multi-agent systems in Xencode
"""
from typing import Dict, List, Optional, Any, Tuple, Set
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
import uuid
import heapq
from abc import ABC, abstractmethod
import random
import math


class CoordinationStrategy(Enum):
    """Types of coordination strategies."""
    MARKET_BASED = "market_based"
    SWARM_INTELLIGENCE = "swarm_intelligence"
    HIERARCHICAL = "hierarchical"
    NEGOTIATION = "negotiation"


class ResourceType(Enum):
    """Types of resources that can be allocated."""
    COMPUTE = "compute"
    MEMORY = "memory"
    TIME = "time"
    SPECIALIZED_SKILL = "specialized_skill"
    DATA_ACCESS = "data_access"


@dataclass
class Resource:
    """Represents a resource that can be allocated to agents."""
    resource_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    resource_type: ResourceType = ResourceType.COMPUTE
    capacity: float = 1.0  # Amount of resource available
    current_usage: float = 0.0
    owner_id: Optional[str] = None
    cost_per_unit: float = 1.0  # Cost per unit of resource
    quality_score: float = 1.0  # Quality/reliability of resource


@dataclass
class Bid:
    """Represents a bid in the market-based allocation system."""
    agent_id: str
    resource_id: str
    bid_amount: float  # Amount willing to pay
    resource_quantity: float  # Amount of resource requested
    timestamp: datetime = field(default_factory=datetime.now)
    priority: int = 0  # Higher priority gets preference


@dataclass
class Task:
    """Represents a task that needs to be coordinated."""
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    description: str = ""
    required_resources: Dict[ResourceType, float] = field(default_factory=dict)
    priority: int = 0  # Higher number = higher priority
    deadline: Optional[datetime] = None
    assigned_agent: Optional[str] = None
    status: str = "pending"  # pending, in_progress, completed, failed
    dependencies: List[str] = field(default_factory=list)  # Task IDs this depends on


@dataclass
class AgentState:
    """Represents the state of an agent in the coordination system."""
    agent_id: str
    performance_score: float = 0.5  # 0.0 to 1.0
    available_resources: Dict[ResourceType, float] = field(default_factory=dict)
    current_tasks: List[str] = field(default_factory=list)
    reputation: float = 0.5  # 0.0 to 1.0
    trust_level: float = 0.5  # 0.0 to 1.0
    specialization: Set[str] = field(default_factory=set)


class CoordinationStrategyBase(ABC):
    """Base class for coordination strategies."""
    
    @abstractmethod
    def coordinate(self, agents: List[AgentState], tasks: List[Task], 
                   resources: List[Resource]) -> Dict[str, Any]:
        """Coordinate agents to accomplish tasks."""
        pass


class MarketBasedAllocation(CoordinationStrategyBase):
    """Implements market-based resource allocation among agents."""
    
    def __init__(self):
        self.resources: Dict[str, Resource] = {}
        self.bids: List[Bid] = []
        self.agent_balances: Dict[str, float] = {}  # Virtual currency for agents
        
    def add_resource(self, resource: Resource):
        """Add a resource to the market."""
        self.resources[resource.resource_id] = resource
        
    def set_agent_balance(self, agent_id: str, balance: float):
        """Set an agent's virtual currency balance."""
        self.agent_balances[agent_id] = balance
        
    def submit_bid(self, bid: Bid) -> bool:
        """Submit a bid for resources."""
        # Check if agent has sufficient balance
        total_cost = bid.bid_amount
        if self.agent_balances.get(bid.agent_id, 0) < total_cost:
            return False
            
        # Check if resource is available
        resource = self.resources.get(bid.resource_id)
        if not resource or (resource.capacity - resource.current_usage) < bid.resource_quantity:
            return False
            
        self.bids.append(bid)
        return True
        
    def allocate_resources(self) -> List[Tuple[str, str, float]]:
        """Allocate resources based on bids."""
        # Sort bids by bid amount (highest first) and priority
        sorted_bids = sorted(self.bids, key=lambda b: (b.bid_amount, b.priority), reverse=True)
        
        allocations = []
        processed_bids = []
        
        for bid in sorted_bids:
            resource = self.resources.get(bid.resource_id)
            if not resource:
                continue
                
            # Check if enough resource is available
            available = resource.capacity - resource.current_usage
            if available >= bid.resource_quantity:
                # Allocate resource
                resource.current_usage += bid.resource_quantity
                resource.owner_id = bid.agent_id
                
                # Deduct from agent's balance
                self.agent_balances[bid.agent_id] = max(
                    0, 
                    self.agent_balances.get(bid.agent_id, 0) - bid.bid_amount
                )
                
                allocations.append((bid.agent_id, bid.resource_id, bid.resource_quantity))
                processed_bids.append(bid)
        
        # Remove processed bids
        for bid in processed_bids:
            self.bids.remove(bid)
            
        return allocations
        
    def coordinate(self, agents: List[AgentState], tasks: List[Task], 
                   resources: List[Resource]) -> Dict[str, Any]:
        """Coordinate using market-based allocation."""
        # Initialize resources and balances
        for resource in resources:
            self.add_resource(resource)
            
        for agent in agents:
            if agent.agent_id not in self.agent_balances:
                self.set_agent_balance(agent.agent_id, 100.0)  # Default balance
        
        # Create bids based on agent needs
        for agent in agents:
            for task_id in agent.current_tasks:
                task = next((t for t in tasks if t.task_id == task_id), None)
                if task:
                    for req_type, req_amount in task.required_resources.items():
                        # Find matching resources
                        for resource in resources:
                            if resource.resource_type == req_type:
                                # Calculate bid amount based on urgency and agent's financial status
                                urgency_factor = 1.0
                                if task.deadline:
                                    time_left = (task.deadline - datetime.now()).total_seconds()
                                    urgency_factor = max(1.0, 10.0 / max(1.0, time_left / 3600))  # More urgent = higher bid
                                
                                bid_amount = req_amount * resource.cost_per_unit * urgency_factor * agent.performance_score
                                bid = Bid(
                                    agent_id=agent.agent_id,
                                    resource_id=resource.resource_id,
                                    bid_amount=min(bid_amount, self.agent_balances.get(agent.agent_id, 0)),
                                    resource_quantity=req_amount,
                                    priority=task.priority
                                )
                                self.submit_bid(bid)
        
        # Allocate resources
        allocations = self.allocate_resources()
        
        return {
            'strategy': CoordinationStrategy.MARKET_BASED,
            'allocations': allocations,
            'remaining_balances': self.agent_balances.copy(),
            'unallocated_tasks': [t.task_id for t in tasks if not any(t.task_id in a.current_tasks for a in agents)]
        }


class SwarmIntelligence(CoordinationStrategyBase):
    """Implements swarm intelligence behaviors for coordination."""
    
    def __init__(self):
        self.pheromone_trails: Dict[str, float] = {}  # Maps task_id to pheromone level
        self.swarm_behavior_params = {
            'alpha': 1.0,  # Pheromone importance
            'beta': 2.0,   # Heuristic importance
            'rho': 0.1,    # Evaporation rate
            'q': 100.0     # Pheromone constant
        }
    
    def calculate_heuristic_value(self, agent: AgentState, task: Task) -> float:
        """Calculate heuristic value for agent-task assignment."""
        # Higher value if agent has relevant specialization
        specialization_bonus = 0.0
        for req_type in task.required_resources.keys():
            if req_type.value in agent.specialization:
                specialization_bonus += 1.0
        
        # Consider agent's performance and availability
        availability = sum(agent.available_resources.values()) / max(1, len(agent.available_resources))
        performance_factor = agent.performance_score
        
        return specialization_bonus + availability + performance_factor
    
    def update_pheromone_trails(self, successful_assignments: List[Tuple[str, str]]):
        """Update pheromone trails based on successful assignments."""
        for agent_id, task_id in successful_assignments:
            key = f"{agent_id}-{task_id}"
            current = self.pheromone_trails.get(key, 0.1)
            self.pheromone_trails[key] = current + 1.0  # Deposit pheromone
    
    def evaporate_pheromones(self):
        """Evaporate pheromones over time."""
        for key in self.pheromone_trails:
            self.pheromone_trails[key] *= (1 - self.swarm_behavior_params['rho'])
    
    def coordinate(self, agents: List[AgentState], tasks: List[Task], 
                   resources: List[Resource]) -> Dict[str, Any]:
        """Coordinate using swarm intelligence."""
        assignments = []
        
        # For each task, find the best agent using ant colony optimization approach
        for task in tasks:
            if task.status != 'pending':
                continue
                
            # Calculate probabilities for each agent
            probabilities = []
            total_prob = 0.0
            
            for agent in agents:
                # Check if agent has required resources
                has_resources = True
                for req_type, req_amount in task.required_resources.items():
                    if agent.available_resources.get(req_type, 0) < req_amount:
                        has_resources = False
                        break
                
                if not has_resources:
                    continue
                
                # Calculate pheromone and heuristic values
                pheromone_val = self.pheromone_trails.get(f"{agent.agent_id}-{task.task_id}", 0.1)
                heuristic_val = self.calculate_heuristic_value(agent, task)
                
                # Calculate probability using ACO formula
                prob = (pheromone_val ** self.swarm_behavior_params['alpha']) * \
                       (heuristic_val ** self.swarm_behavior_params['beta'])
                
                probabilities.append((agent.agent_id, prob))
                total_prob += prob
            
            # Select agent based on probabilities
            if total_prob > 0:
                # Normalize probabilities
                normalized_probs = [(aid, prob/total_prob) for aid, prob in probabilities]
                
                # Select agent using roulette wheel selection
                rand_val = random.random()
                cumulative_prob = 0.0
                selected_agent = None
                
                for agent_id, prob in normalized_probs:
                    cumulative_prob += prob
                    if rand_val <= cumulative_prob:
                        selected_agent = agent_id
                        break
                
                if selected_agent:
                    assignments.append((selected_agent, task.task_id))
                    task.assigned_agent = selected_agent
                    task.status = 'in_progress'
                    
                    # Update agent's available resources
                    selected_agent_state = next(a for a in agents if a.agent_id == selected_agent)
                    for req_type, req_amount in task.required_resources.items():
                        selected_agent_state.available_resources[req_type] = max(
                            0, 
                            selected_agent_state.available_resources.get(req_type, 0) - req_amount
                        )
        
        # Update pheromone trails based on assignments
        self.update_pheromone_trails(assignments)
        
        # Evaporate some pheromones
        self.evaporate_pheromones()
        
        return {
            'strategy': CoordinationStrategy.SWARM_INTELLIGENCE,
            'assignments': assignments,
            'pheromone_levels': self.pheromone_trails.copy(),
            'completed_tasks': [t.task_id for t in tasks if t.status == 'in_progress']
        }


class HierarchicalCoordinator(CoordinationStrategyBase):
    """Implements hierarchical decision-making structures."""
    
    def __init__(self):
        self.leadership_hierarchy: Dict[str, str] = {}  # agent_id -> leader_id
        self.leader_decisions: Dict[str, Any] = {}
        
    def establish_hierarchy(self, agents: List[AgentState]) -> Dict[str, List[str]]:
        """Establish a hierarchy based on agent capabilities."""
        # Sort agents by performance score to determine leadership
        sorted_agents = sorted(agents, key=lambda a: a.performance_score, reverse=True)
        
        # Create hierarchy: top 20% are leaders, rest are followers
        num_leaders = max(1, len(sorted_agents) // 5)  # At least 1 leader
        leaders = sorted_agents[:num_leaders]
        followers = sorted_agents[num_leaders:]
        
        hierarchy = {}
        for leader in leaders:
            hierarchy[leader.agent_id] = []
            
        # Assign followers to leaders based on specialization similarity
        for follower in followers:
            # Find leader with most similar specialization
            best_leader = leaders[0].agent_id  # Default to top performer
            best_similarity = 0
            
            for leader in leaders:
                similarity = len(follower.specialization.intersection(leader.specialization))
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_leader = leader.agent_id
            
            hierarchy[best_leader].append(follower.agent_id)
            self.leadership_hierarchy[follower.agent_id] = best_leader
        
        return hierarchy
    
    def coordinate(self, agents: List[AgentState], tasks: List[Task], 
                   resources: List[Resource]) -> Dict[str, Any]:
        """Coordinate using hierarchical structure."""
        hierarchy = self.establish_hierarchy(agents)
        
        assignments = []
        decisions = {}
        
        # Leaders assign tasks to their followers
        for leader_id, followers in hierarchy.items():
            leader = next((a for a in agents if a.agent_id == leader_id), None)
            if not leader:
                continue
            
            # Get tasks assigned to this leader's group
            leader_tasks = [t for t in tasks if t.priority >= 5 or t.assigned_agent == leader_id]
            
            # Distribute tasks among followers
            for i, task in enumerate(leader_tasks):
                if followers and i < len(followers):
                    follower_id = followers[i % len(followers)]
                    assignments.append((follower_id, task.task_id))
                    task.assigned_agent = follower_id
                    task.status = 'in_progress'
                    
                    # Record the decision
                    if leader_id not in decisions:
                        decisions[leader_id] = []
                    decisions[leader_id].append({
                        'task_id': task.task_id,
                        'assigned_to': follower_id,
                        'decision_time': datetime.now().isoformat()
                    })
        
        return {
            'strategy': CoordinationStrategy.HIERARCHICAL,
            'hierarchy': hierarchy,
            'assignments': assignments,
            'leadership_decisions': decisions,
            'completed_tasks': [t.task_id for t in tasks if t.status == 'in_progress']
        }


class NegotiationProtocol(CoordinationStrategyBase):
    """Implements negotiation protocols between agents."""
    
    def __init__(self):
        self.proposals: Dict[str, List[Dict[str, Any]]] = {}  # task_id -> proposals
        self.negotiation_history: List[Dict[str, Any]] = []
        
    def propose_solution(self, proposer_id: str, task_id: str, 
                        offer: Dict[str, Any], target_agents: List[str]) -> str:
        """Make a proposal to other agents."""
        proposal_id = str(uuid.uuid4())
        
        proposal = {
            'proposal_id': proposal_id,
            'proposer_id': proposer_id,
            'task_id': task_id,
            'offer': offer,
            'target_agents': target_agents,
            'timestamp': datetime.now().isoformat(),
            'status': 'pending'
        }
        
        if task_id not in self.proposals:
            self.proposals[task_id] = []
        self.proposals[task_id].append(proposal)
        
        return proposal_id
    
    def evaluate_proposal(self, agent_id: str, proposal: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """Evaluate a proposal and return acceptance and counter-offer."""
        # Simple evaluation based on agent's needs and capabilities
        agent_needs = random.random()  # Simulate agent's evaluation
        accept_probability = min(1.0, agent_needs + 0.2)  # 20% base acceptance chance
        
        accepted = random.random() < accept_probability
        
        counter_offer = {}
        if not accepted:
            # Make a counter-offer
            original_offer = proposal['offer']
            for key, value in original_offer.items():
                if isinstance(value, (int, float)):
                    counter_offer[key] = value * random.uniform(0.8, 1.2)  # ±20%
                else:
                    counter_offer[key] = value
        
        return accepted, counter_offer if not accepted else {}
    
    def coordinate(self, agents: List[AgentState], tasks: List[Task], 
                   resources: List[Resource]) -> Dict[str, Any]:
        """Coordinate using negotiation protocols."""
        negotiations = []
        
        # Each agent proposes solutions for tasks they're interested in
        for agent in agents:
            for task in tasks:
                if task.status != 'pending':
                    continue
                
                # Check if agent is interested in this task
                if random.random() < 0.3:  # 30% chance to propose
                    # Create an offer
                    offer = {
                        'resources_offered': dict(task.required_resources),
                        'expected_completion_time': random.randint(1, 10),  # hours
                        'quality_guarantee': agent.performance_score,
                        'payment_requested': len(task.required_resources) * 10
                    }
                    
                    # Propose to other agents who might be interested
                    other_agents = [a.agent_id for a in agents if a.agent_id != agent.agent_id]
                    proposal_id = self.propose_solution(
                        agent.agent_id, task.task_id, offer, other_agents[:2]  # Propose to 2 others
                    )
                    
                    # Process responses
                    proposal = next(p for p in self.proposals[task.task_id] if p['proposal_id'] == proposal_id)
                    
                    responses = []
                    for target_agent in proposal['target_agents']:
                        accepted, counter = self.evaluate_proposal(target_agent, proposal)
                        responses.append({
                            'agent_id': target_agent,
                            'accepted': accepted,
                            'counter_offer': counter
                        })
                        
                        if accepted:
                            task.assigned_agent = agent.agent_id
                            task.status = 'in_progress'
                            negotiations.append({
                                'task_id': task.task_id,
                                'agreement': 'reached',
                                'participants': [agent.agent_id, target_agent],
                                'terms': offer
                            })
                            break  # Agreement reached
                    
                    proposal['responses'] = responses
                    proposal['status'] = 'completed'
        
        return {
            'strategy': CoordinationStrategy.NEGOTIATION,
            'negotiations': negotiations,
            'proposals_made': len([p for proposals_list in self.proposals.values() for p in proposals_list]),
            'agreements_reached': len([n for n in negotiations if n['agreement'] == 'reached'])
        }


class AdvancedCoordinationEngine:
    """Main engine for coordinating advanced multi-agent strategies."""
    
    def __init__(self):
        self.market_allocator = MarketBasedAllocation()
        self.swarm_intelligence = SwarmIntelligence()
        self.hierarchical_coordinator = HierarchicalCoordinator()
        self.negotiation_protocol = NegotiationProtocol()
        self.agents: Dict[str, AgentState] = {}
        self.tasks: Dict[str, Task] = {}
        self.resources: Dict[str, Resource] = {}
        
    def register_agent(self, agent_state: AgentState):
        """Register an agent with the coordination engine."""
        self.agents[agent_state.agent_id] = agent_state
        
    def register_task(self, task: Task):
        """Register a task with the coordination engine."""
        self.tasks[task.task_id] = task
        
    def register_resource(self, resource: Resource):
        """Register a resource with the coordination engine."""
        self.resources[resource.resource_id] = resource
        
    def coordinate_by_strategy(self, strategy: CoordinationStrategy) -> Dict[str, Any]:
        """Coordinate agents using a specific strategy."""
        agents_list = list(self.agents.values())
        tasks_list = list(self.tasks.values())
        resources_list = list(self.resources.values())
        
        if strategy == CoordinationStrategy.MARKET_BASED:
            return self.market_allocator.coordinate(agents_list, tasks_list, resources_list)
        elif strategy == CoordinationStrategy.SWARM_INTELLIGENCE:
            return self.swarm_intelligence.coordinate(agents_list, tasks_list, resources_list)
        elif strategy == CoordinationStrategy.HIERARCHICAL:
            return self.hierarchical_coordinator.coordinate(agents_list, tasks_list, resources_list)
        elif strategy == CoordinationStrategy.NEGOTIATION:
            return self.negotiation_protocol.coordinate(agents_list, tasks_list, resources_list)
        else:
            raise ValueError(f"Unknown coordination strategy: {strategy}")
    
    def get_coordination_recommendation(self) -> CoordinationStrategy:
        """Recommend the best coordination strategy based on current situation."""
        # Simple heuristic: if many agents have similar capabilities, use swarm
        # If clear leadership structure exists, use hierarchical
        # If resources are scarce, use market-based
        # Otherwise, use negotiation
        
        if len(self.resources) < len(self.agents) * 0.5:  # Scarce resources
            return CoordinationStrategy.MARKET_BASED
        elif len(set(a.performance_score for a in self.agents.values())) > len(self.agents) * 0.7:  # Diverse performance
            return CoordinationStrategy.HIERARCHICAL
        elif len(self.agents) > 5:  # Many agents
            return CoordinationStrategy.SWARM_INTELLIGENCE
        else:
            return CoordinationStrategy.NEGOTIATION


# Helper functions for creating common scenarios
def create_compute_resource(agent_id: str, capacity: float = 1.0) -> Resource:
    """Create a compute resource for an agent."""
    return Resource(
        resource_type=ResourceType.COMPUTE,
        capacity=capacity,
        cost_per_unit=1.0,
        owner_id=agent_id
    )


def create_memory_resource(agent_id: str, capacity: float = 1.0) -> Resource:
    """Create a memory resource for an agent."""
    return Resource(
        resource_type=ResourceType.MEMORY,
        capacity=capacity,
        cost_per_unit=0.5,
        owner_id=agent_id
    )


def create_specialized_task(description: str, required_resources: Dict[ResourceType, float], 
                          priority: int = 1) -> Task:
    """Create a specialized task with required resources."""
    return Task(
        description=description,
        required_resources=required_resources,
        priority=priority
    )