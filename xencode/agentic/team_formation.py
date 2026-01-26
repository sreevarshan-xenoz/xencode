"""
Team formation system for dynamic agent team assembly in Xencode
"""
from typing import Dict, List, Optional, Set, Tuple, Any
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
import uuid
import re
from .coordinator import AgentType
from .specialized import SpecializedAgentType


class TeamFormationStrategy(Enum):
    """Strategies for forming agent teams."""
    SKILL_MATCHING = "skill_matching"
    EXPERTISE_BASED = "expertise_based"
    LOAD_BALANCED = "load_balanced"
    HYBRID = "hybrid"


class TeamRole(Enum):
    """Roles that agents can play in a team."""
    LEADER = "leader"
    SPECIALIST = "specialist"
    COORDINATOR = "coordinator"
    SUPPORT = "support"
    REVIEWER = "reviewer"


@dataclass
class AgentCapability:
    """Represents the capabilities of an agent."""
    agent_type: str  # Either AgentType or SpecializedAgentType value
    skills: Set[str]
    expertise_domains: Set[str]
    performance_score: float = 1.0  # 0.0 to 1.0 scale
    availability: bool = True
    current_workload: int = 0
    max_capacity: int = 5  # Max concurrent tasks


@dataclass
class TeamAssignment:
    """Represents an agent's assignment to a team."""
    agent_id: str
    role: TeamRole
    assigned_at: datetime = field(default_factory=datetime.now)
    estimated_completion: Optional[datetime] = None


@dataclass
class AgentTeam:
    """Represents a dynamically formed team of agents."""
    team_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    members: List[TeamAssignment] = field(default_factory=list)
    formation_strategy: TeamFormationStrategy = TeamFormationStrategy.SKILL_MATCHING
    created_at: datetime = field(default_factory=datetime.now)
    status: str = "active"  # active, completed, failed
    task_requirements: Dict[str, Any] = field(default_factory=dict)
    
    def add_member(self, agent_id: str, role: TeamRole):
        """Add an agent to the team."""
        assignment = TeamAssignment(agent_id=agent_id, role=role)
        self.members.append(assignment)
        
    def remove_member(self, agent_id: str):
        """Remove an agent from the team."""
        self.members = [member for member in self.members if member.agent_id != agent_id]
        
    def get_members_by_role(self, role: TeamRole) -> List[TeamAssignment]:
        """Get all members with a specific role."""
        return [member for member in self.members if member.role == role]
        
    def get_leader(self) -> Optional[TeamAssignment]:
        """Get the team leader."""
        leaders = self.get_members_by_role(TeamRole.LEADER)
        return leaders[0] if leaders else None
        
    def get_specialists(self) -> List[TeamAssignment]:
        """Get all specialists in the team."""
        return self.get_members_by_role(TeamRole.SPECIALIST)


class SkillMatcher:
    """Matches agents to tasks based on skills and expertise."""
    
    def __init__(self):
        self.skill_weights = {
            'programming': 1.0,
            'debugging': 1.0,
            'testing': 0.9,
            'documentation': 0.8,
            'security': 0.95,
            'optimization': 0.9,
            'data_analysis': 0.95,
            'web_development': 0.9,
            'devops': 0.85,
            'research': 0.8
        }
    
    def calculate_skill_match_score(self, agent_capability: AgentCapability, 
                                  required_skills: Set[str]) -> float:
        """Calculate how well an agent matches required skills."""
        if not required_skills:
            return agent_capability.performance_score
            
        matched_skills = agent_capability.skills.intersection(required_skills)
        if not matched_skills:
            return 0.0
            
        # Calculate weighted score based on importance of matched skills
        total_weight = sum(self.skill_weights.get(skill, 0.5) for skill in required_skills)
        matched_weight = sum(self.skill_weights.get(skill, 0.5) for skill in matched_skills)
        
        skill_score = matched_weight / total_weight if total_weight > 0 else 0.0
        availability_factor = 1.0 if agent_capability.availability else 0.3
        capacity_factor = max(0.1, 1.0 - (agent_capability.current_workload / agent_capability.max_capacity))
        
        final_score = (skill_score * 0.6 + 
                      agent_capability.performance_score * 0.3 + 
                      availability_factor * 0.1) * capacity_factor
        
        return final_score
    
    def extract_required_skills_from_task(self, task_description: str) -> Set[str]:
        """Extract required skills from a task description."""
        task_lower = task_description.lower()
        skills = set()
        
        # Programming languages and technologies
        if re.search(r'\b(python|javascript|typescript|java|c\+\+|go|rust|php|ruby|swift|kotlin)\b', task_lower):
            skills.add('programming')
        
        if 'debug' in task_lower or 'fix' in task_lower or 'bug' in task_lower:
            skills.add('debugging')
            
        if 'test' in task_lower or 'unit test' in task_lower or 'integration test' in task_lower:
            skills.add('testing')
            
        if 'document' in task_lower or 'readme' in task_lower or 'tutorial' in task_lower:
            skills.add('documentation')
            
        if 'security' in task_lower or 'vulnerability' in task_lower or 'secure' in task_lower:
            skills.add('security')
            
        if 'optimize' in task_lower or 'performance' in task_lower or 'efficiency' in task_lower:
            skills.add('optimization')
            
        if ('data' in task_lower and 'analysis' in task_lower) or 'pandas' in task_lower or 'numpy' in task_lower:
            skills.add('data_analysis')
            
        if 'web' in task_lower or 'html' in task_lower or 'css' in task_lower or 'react' in task_lower:
            skills.add('web_development')
            
        if 'docker' in task_lower or 'kubernetes' in task_lower or 'ci/cd' in task_lower:
            skills.add('devops')
            
        if 'research' in task_lower or 'find' in task_lower or 'investigate' in task_lower:
            skills.add('research')
            
        # Add generic skills if none were detected
        if not skills:
            if any(word in task_lower for word in ['function', 'code', 'implement', 'create']):
                skills.add('programming')
            elif any(word in task_lower for word in ['analyze', 'review', 'check']):
                skills.add('research')
            else:
                skills.add('general')
                
        return skills


class TeamFormationEngine:
    """Engine for dynamically forming agent teams based on task requirements."""
    
    def __init__(self):
        self.agents: Dict[str, AgentCapability] = {}
        self.skill_matcher = SkillMatcher()
        self.teams: Dict[str, AgentTeam] = {}
        
    def register_agent(self, agent_id: str, capability: AgentCapability):
        """Register an agent with its capabilities."""
        self.agents[agent_id] = capability
        
    def unregister_agent(self, agent_id: str):
        """Unregister an agent."""
        if agent_id in self.agents:
            del self.agents[agent_id]
            
    def form_team(self, task_description: str, 
                  strategy: TeamFormationStrategy = TeamFormationStrategy.HYBRID,
                  max_team_size: int = 5) -> Optional[AgentTeam]:
        """Form a team of agents based on task requirements."""
        # Extract required skills from task
        required_skills = self.skill_matcher.extract_required_skills_from_task(task_description)
        
        # Find suitable agents based on strategy
        if strategy == TeamFormationStrategy.SKILL_MATCHING:
            selected_agents = self._select_by_skill_matching(required_skills, max_team_size)
        elif strategy == TeamFormationStrategy.EXPERTISE_BASED:
            selected_agents = self._select_by_expertise(task_description, max_team_size)
        elif strategy == TeamFormationStrategy.LOAD_BALANCED:
            selected_agents = self._select_by_load_balancing(max_team_size)
        else:  # HYBRID
            selected_agents = self._select_by_hybrid_approach(task_description, required_skills, max_team_size)
        
        if not selected_agents:
            return None
            
        # Create team
        team = AgentTeam(
            name=f"Team for: {task_description[:50]}{'...' if len(task_description) > 50 else ''}",
            formation_strategy=strategy,
            task_requirements={
                'description': task_description,
                'required_skills': list(required_skills),
                'timestamp': datetime.now().isoformat()
            }
        )
        
        # Assign roles to team members
        self._assign_roles(team, selected_agents)
        
        # Register team
        self.teams[team.team_id] = team
        
        return team
    
    def _select_by_skill_matching(self, required_skills: Set[str], max_team_size: int) -> List[str]:
        """Select agents based on skill matching."""
        scored_agents = []
        
        for agent_id, capability in self.agents.items():
            if capability.availability:
                score = self.skill_matcher.calculate_skill_match_score(capability, required_skills)
                if score > 0.1:  # Only consider agents with decent match
                    scored_agents.append((agent_id, score))
        
        # Sort by score descending
        scored_agents.sort(key=lambda x: x[1], reverse=True)
        
        # Return top agents up to max team size
        return [agent_id for agent_id, score in scored_agents[:max_team_size]]
    
    def _select_by_expertise(self, task_description: str, max_team_size: int) -> List[str]:
        """Select agents based on expertise domains."""
        task_lower = task_description.lower()
        scored_agents = []
        
        for agent_id, capability in self.agents.items():
            if capability.availability:
                # Score based on expertise domain match
                expertise_score = 0.0
                for domain in capability.expertise_domains:
                    if domain.lower() in task_lower:
                        expertise_score += 0.5
                    elif any(word in domain.lower() for word in task_description.lower().split()):
                        expertise_score += 0.3
                        
                # Factor in performance and availability
                total_score = (expertise_score * 0.6 + 
                              capability.performance_score * 0.3 +
                              (1.0 - capability.current_workload / capability.max_capacity) * 0.1)
                
                if total_score > 0.1:
                    scored_agents.append((agent_id, total_score))
        
        # Sort by score descending
        scored_agents.sort(key=lambda x: x[1], reverse=True)
        
        return [agent_id for agent_id, score in scored_agents[:max_team_size]]
    
    def _select_by_load_balancing(self, max_team_size: int) -> List[str]:
        """Select agents based on current workload."""
        available_agents = [(agent_id, capability) 
                           for agent_id, capability in self.agents.items() 
                           if capability.availability and capability.current_workload < capability.max_capacity]
        
        # Sort by lowest workload
        available_agents.sort(key=lambda x: x[1].current_workload)
        
        return [agent_id for agent_id, capability in available_agents[:max_team_size]]
    
    def _select_by_hybrid_approach(self, task_description: str, required_skills: Set[str], 
                                  max_team_size: int) -> List[str]:
        """Select agents using a hybrid approach combining multiple strategies."""
        # Get candidates from skill matching
        skill_candidates = self._select_by_skill_matching(required_skills, max_team_size * 2)
        
        # Get candidates from expertise matching
        expertise_candidates = self._select_by_expertise(task_description, max_team_size * 2)
        
        # Combine and rank
        all_candidates = set(skill_candidates + expertise_candidates)
        scored_candidates = []
        
        for agent_id in all_candidates:
            if agent_id in self.agents and self.agents[agent_id].availability:
                capability = self.agents[agent_id]
                
                # Calculate combined score
                skill_score = self.skill_matcher.calculate_skill_match_score(capability, required_skills)
                expertise_score = 0.0
                
                for domain in capability.expertise_domains:
                    if domain.lower() in task_description.lower():
                        expertise_score += 0.5
                        
                load_factor = 1.0 - (capability.current_workload / capability.max_capacity)
                
                combined_score = (skill_score * 0.4 + 
                                expertise_score * 0.3 + 
                                capability.performance_score * 0.2 + 
                                load_factor * 0.1)
                
                scored_candidates.append((agent_id, combined_score))
        
        # Sort by combined score
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        return [agent_id for agent_id, score in scored_candidates[:max_team_size]]
    
    def _assign_roles(self, team: AgentTeam, selected_agents: List[str]):
        """Assign roles to agents in the team."""
        if not selected_agents:
            return
            
        # Assign leader (best overall match or first agent)
        team.add_member(selected_agents[0], TeamRole.LEADER)
        
        # Assign specialists and other roles based on skills
        for i, agent_id in enumerate(selected_agents[1:], 1):
            # Determine role based on agent type and capabilities
            capability = self.agents[agent_id]
            
            # Default to specialist, but adjust based on specific capabilities
            role = TeamRole.SPECIALIST
            
            # If this is the second agent and has complementary skills, make coordinator
            if i == 1 and len(selected_agents) > 2:
                role = TeamRole.COORDINATOR
            elif i == len(selected_agents) - 1:  # Last agent often serves as reviewer
                role = TeamRole.REVIEWER
            else:
                role = TeamRole.SPECIALIST
                
            team.add_member(agent_id, role)
    
    def update_agent_workload(self, agent_id: str, change: int):
        """Update an agent's workload."""
        if agent_id in self.agents:
            self.agents[agent_id].current_workload = max(0, 
                self.agents[agent_id].current_workload + change)
    
    def disband_team(self, team_id: str):
        """Disband a team and update agent workloads."""
        if team_id in self.teams:
            team = self.teams[team_id]
            
            # Decrease workload for all team members
            for member in team.members:
                self.update_agent_workload(member.agent_id, -1)
                
            # Update team status
            team.status = "completed"
            del self.teams[team_id]


# Utility function to create agent capabilities from existing agent types
def create_capability_from_agent_type(agent_type: AgentType) -> AgentCapability:
    """Create agent capability from basic agent type."""
    skill_map = {
        AgentType.CODE: {"programming", "debugging", "code_review", "testing"},
        AgentType.RESEARCH: {"research", "analysis", "information_gathering", "summarization"},
        AgentType.EXECUTION: {"execution", "command_line", "file_operations", "system_tasks"},
        AgentType.GENERAL: {"general_assistance", "communication", "planning", "coordination"},
        AgentType.PLANNING: {"planning", "organization", "strategy", "workflow_design"}
    }
    
    expertise_map = {
        AgentType.CODE: {"software_development", "programming_languages", "algorithms"},
        AgentType.RESEARCH: {"information_retrieval", "data_analysis", "academic_research"},
        AgentType.EXECUTION: {"system_administration", "automation", "task_execution"},
        AgentType.GENERAL: {"general_assistance", "communication", "problem_solving"},
        AgentType.PLANNING: {"project_management", "workflow_design", "strategic_planning"}
    }
    
    return AgentCapability(
        agent_type=agent_type.value,
        skills=skill_map.get(agent_type, {"general"}),
        expertise_domains=expertise_map.get(agent_type, {"general"}),
        performance_score=0.85
    )


def create_capability_from_specialized_agent_type(agent_type: SpecializedAgentType) -> AgentCapability:
    """Create agent capability from specialized agent type."""
    skill_map = {
        SpecializedAgentType.DATA_SCIENCE: {"data_analysis", "machine_learning", "statistics", "visualization", "pandas", "numpy"},
        SpecializedAgentType.WEB_DEVELOPMENT: {"web_development", "html", "css", "javascript", "react", "api_development"},
        SpecializedAgentType.SECURITY_ANALYSIS: {"security", "vulnerability_assessment", "secure_coding", "penetration_testing"},
        SpecializedAgentType.DEVOPS: {"devops", "ci/cd", "docker", "kubernetes", "infrastructure", "automation"},
        SpecializedAgentType.TESTING: {"testing", "qa", "unit_testing", "integration_testing", "test_automation"},
        SpecializedAgentType.DOCUMENTATION: {"documentation", "technical_writing", "tutorials", "knowledge_management"}
    }
    
    expertise_map = {
        SpecializedAgentType.DATA_SCIENCE: {"data_science", "machine_learning", "statistical_analysis", "data_visualization"},
        SpecializedAgentType.WEB_DEVELOPMENT: {"web_technologies", "frontend", "backend", "full_stack_development"},
        SpecializedAgentType.SECURITY_ANALYSIS: {"cybersecurity", "application_security", "compliance", "risk_assessment"},
        SpecializedAgentType.DEVOPS: {"cloud_infrastructure", "deployment_automation", "monitoring", "containerization"},
        SpecializedAgentType.TESTING: {"software_quality", "test_strategy", "automated_testing", "quality_assurance"},
        SpecializedAgentType.DOCUMENTATION: {"technical_documentation", "knowledge_sharing", "content_creation", "information_architecture"}
    }
    
    return AgentCapability(
        agent_type=agent_type.value,
        skills=skill_map.get(agent_type, {"general"}),
        expertise_domains=expertise_map.get(agent_type, {"general"}),
        performance_score=0.9
    )