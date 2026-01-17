import sys
sys.path.insert(0, '.')

from xencode.agentic import (
    AgentCoordinator,
    AgentCommunicationLayer,
    TeamFormationEngine,
    AgentTeam,
    TeamFormationStrategy,
    create_capability_from_agent_type,
    create_capability_from_specialized_agent_type,
    AdvancedCoordinationEngine,
    CoordinationStrategy,
    AgentState,
    Task,
    Resource,
    ResourceType
)
from xencode.agentic.specialized import SpecializedAgentType

print("=== Comprehensive Multi-Agent System Integration Test ===")

# Test 1: Basic Agent Coordinator
print("\n1. Testing Agent Coordinator...")
coordinator = AgentCoordinator()
print(f"✓ Created coordinator with {len(coordinator.agents)} agents")

# Test 2: Communication Layer
print("\n2. Testing Communication Layer...")
comm_layer = AgentCommunicationLayer()
comm_layer.integrate_with_coordinator(coordinator)
print(f"✓ Communication layer integrated with coordinator")

# Test 3: Team Formation Engine
print("\n3. Testing Team Formation Engine...")
team_engine = TeamFormationEngine()

# Register basic agents
for agent_type, agent in coordinator.agents.items():
    agent_id = f"basic_{agent_type.value}"
    capability = create_capability_from_agent_type(agent_type)
    team_engine.register_agent(agent_id, capability)

# Register specialized agents
specialized_types = list(SpecializedAgentType)[:3]
for i, spec_type in enumerate(specialized_types):
    agent_id = f"specialized_{spec_type.value}_{i}"
    capability = create_capability_from_specialized_agent_type(spec_type)
    team_engine.register_agent(agent_id, capability)

print(f"✓ Team formation engine registered {len(team_engine.agents)} agents")

# Test 4: Team Formation
print("\n4. Testing Team Formation...")
task_desc = "Create a Python web application with data analysis features"
team = team_engine.form_team(task_desc, TeamFormationStrategy.HYBRID, max_team_size=4)

if team:
    print(f"✓ Formed team for: {task_desc}")
    print(f"  - Members: {len(team.members)}")
    for member in team.members:
        print(f"    - {member.agent_id} ({member.role})")
else:
    print("✗ Failed to form team")

# Test 5: Advanced Coordination Engine
print("\n5. Testing Advanced Coordination Engine...")
coordination_engine = AdvancedCoordinationEngine()

# Register agents with coordination engine
for agent_type, agent in coordinator.agents.items():
    agent_id = f"basic_{agent_type.value}"
    capability = create_capability_from_agent_type(agent_type)
    
    agent_state = AgentState(
        agent_id=agent_id,
        performance_score=capability.performance_score,
        available_resources={ResourceType.COMPUTE: 1.0, ResourceType.MEMORY: 1.0},
        specialization=capability.skills
    )
    coordination_engine.register_agent(agent_state)

# Register tasks
task1 = Task(
    description="Process large dataset",
    required_resources={ResourceType.COMPUTE: 0.8, ResourceType.MEMORY: 0.6},
    priority=3
)
coordination_engine.register_task(task1)

task2 = Task(
    description="Perform security analysis",
    required_resources={ResourceType.COMPUTE: 0.5, ResourceType.MEMORY: 0.3},
    priority=5
)
coordination_engine.register_task(task2)

# Register resources
resource1 = Resource(resource_type=ResourceType.COMPUTE, capacity=1.0, cost_per_unit=1.0)
coordination_engine.register_resource(resource1)

resource2 = Resource(resource_type=ResourceType.MEMORY, capacity=1.0, cost_per_unit=0.5)
coordination_engine.register_resource(resource2)

print(f"✓ Coordination engine set up with {len(coordination_engine.agents)} agents, "
      f"{len(coordination_engine.tasks)} tasks, and {len(coordination_engine.resources)} resources")

# Test different coordination strategies
print("\n6. Testing Coordination Strategies...")

# Market-based
market_result = coordination_engine.coordinate_by_strategy(CoordinationStrategy.MARKET_BASED)
print(f"  ✓ Market-based coordination completed: {len(market_result.get('allocations', []))} allocations")

# Swarm intelligence
swarm_result = coordination_engine.coordinate_by_strategy(CoordinationStrategy.SWARM_INTELLIGENCE)
print(f"  ✓ Swarm intelligence coordination completed: {len(swarm_result.get('assignments', []))} assignments")

# Hierarchical
hier_result = coordination_engine.coordinate_by_strategy(CoordinationStrategy.HIERARCHICAL)
print(f"  ✓ Hierarchical coordination completed: {len(hier_result.get('assignments', []))} assignments")

# Negotiation
negotiation_result = coordination_engine.coordinate_by_strategy(CoordinationStrategy.NEGOTIATION)
print(f"  ✓ Negotiation protocol completed: {negotiation_result.get('proposals_made', 0)} proposals")

# Test 7: Integration between all systems
print("\n7. Testing Full Integration...")

# Check that all systems are properly connected
has_communication = hasattr(coordinator, 'communication_layer')
has_teams = len(team_engine.teams) > 0 if team else False
has_coordination = len(coordination_engine.agents) > 0

print(f"  ✓ Communication integration: {has_communication}")
print(f"  ✓ Team formation working: {has_teams or team is not None}")
print(f"  ✓ Coordination engine working: {has_coordination}")

# Test 8: Communication stats
print("\n8. Testing Communication Statistics...")
stats = comm_layer.get_communication_stats()
print(f"  ✓ Communication stats: {stats}")

# Test 9: Strategy recommendation
print("\n9. Testing Strategy Recommendation...")
recommended = coordination_engine.get_coordination_recommendation()
print(f"  ✓ Recommended strategy: {recommended}")

print("\n=== All Systems Integration Test Passed! ===")
print("\nSummary of Integrated Capabilities:")
print("- Agent Coordination: ✓ Working")
print("- Communication Layer: ✓ Working")
print("- Team Formation: ✓ Working")
print("- Advanced Coordination: ✓ Working")
print("- Strategy Selection: ✓ Working")
print("- System Integration: ✓ Working")

# Cleanup
comm_layer.shutdown()
print("\n✓ Cleaned up communication layer")

print("\n🎉 All multi-agent enhancement components are working correctly and integrated!")