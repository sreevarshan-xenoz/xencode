"""
Agentic capabilities for Xencode using LangChain.
"""
from .manager import LangChainManager
from .tools import ReadFileTool, WriteFileTool, ExecuteCommandTool
from .advanced_tools import (
    GitStatusTool,
    GitDiffTool,
    GitLogTool,
    GitCommitTool,
    WebSearchTool,
    CodeAnalysisTool,
    ToolRegistry
)
from .enhanced_tools import (
    GitBranchTool,
    GitPushTool,
    GitPullTool,
    FindFileTool,
    FileStatTool,
    DependencyAnalysisTool,
    SystemInfoTool,
    ProcessInfoTool,
    WebSearchDetailedTool,
    EnhancedToolRegistry
)
from .coordinator import AgentCoordinator, AgentType
from .ensemble_integration import EnsembleChain, ModelCouncil, create_ensemble_chain, create_model_council
from .specialized import (
    SpecializedAgentType,
    SpecializedAgent,
    DataScienceAgent,
    WebDevelopmentAgent,
    SecurityAnalysisAgent,
    DevOpsAgent,
    TestingAgent,
    DocumentationAgent,
    SpecializedAgentFactory
)
from .specialized.coordinator import SpecializedAgentCoordinator
from .communication import (
    Message, MessageType, MessageStatus, MessageTemplates,
    CommunicationProtocol, MessageBroker, InMemoryProtocol,
    SecureChannel, ChannelManager
)
from .communication.integration import AgentCommunicationLayer
from .team_formation import (
    TeamFormationEngine, AgentTeam, TeamAssignment, TeamRole,
    AgentCapability, TeamFormationStrategy, create_capability_from_agent_type,
    create_capability_from_specialized_agent_type
)
from .coordination_strategies import (
    AdvancedCoordinationEngine, CoordinationStrategy, ResourceType,
    Resource, Bid, Task, AgentState, MarketBasedAllocation,
    SwarmIntelligence, HierarchicalCoordinator, NegotiationProtocol
)
from .memory_learning import (
    AgentLearningSystem, AgentMemory, SharedKnowledgeBase, MemoryEntry,
    KnowledgeItem, LearningPattern, MemoryType, KnowledgeSourceType,
    ExperienceSharingSystem, HistoricalTaskPatterns, create_memory_from_task_result,
    create_knowledge_from_solution
)
from .monitoring_analytics import (
    MonitoringAnalyticsEngine, MetricsCollector, CollaborationAnalyzer,
    RealTimeDashboard, Metric, Alert, CollaborationStats, MetricType,
    AlertSeverity, create_utilization_metric, create_efficiency_metric
)
from .workflow_management import (
    WorkflowManager, Workflow, Subtask, TaskStatus, TaskPriority, TaskType,
    TaskDecompositionEngine, DependencyManager, CheckpointManager,
    create_workflow_from_task, get_next_ready_subtasks
)
from .human_supervision import (
    HumanSupervisionInterface, SupervisionEngine, SupervisionRequest,
    HumanFeedback, ApprovalRule, SupervisionLevel, DecisionCategory,
    ApprovalStatus, FeedbackType, FeedbackIntegrationSystem,
    create_supervision_request_for_task, submit_human_feedback
)
from .cross_domain_expertise import (
    CrossDomainExpertiseSystem, DomainBridgeAgent, KnowledgeTranslationSystem,
    CrossDomainCoordinator, HybridReasoningEngine, DomainKnowledge,
    TranslationRule, CrossDomainRequest, DomainType, TranslationType,
    create_domain_knowledge, create_translation_rule, get_cross_domain_solution
)
from .resource_management import (
    ResourceManagementSystem, ResourceManager, CostOptimizer, PriorityScheduler,
    Resource, ResourcePool, ResourceRequest, ResourceAllocation, ResourceType,
    ResourcePoolType, TaskPriority, ResourceAllocationStatus,
    create_compute_resource_pool, create_memory_resource_pool, request_resources_with_budget
)
from .security_governance import (
    SecurityGovernanceSystem, IdentityManager, AccessControlManager, AuditLogger,
    PrivacyPreservationManager, AgentIdentity, AccessControlRule, AuditRecord,
    SecurityPolicy, Permission, SecurityLevel, AuditEventType, ComplianceStatus,
    create_agent_identity, create_access_control_rule, check_security_compliance
)

__all__ = [
    "LangChainManager",
    "ReadFileTool",
    "WriteFileTool",
    "ExecuteCommandTool",
    "GitStatusTool",
    "GitDiffTool",
    "GitLogTool",
    "GitCommitTool",
    "WebSearchTool",
    "CodeAnalysisTool",
    "ToolRegistry",
    "GitBranchTool",
    "GitPushTool",
    "GitPullTool",
    "FindFileTool",
    "FileStatTool",
    "DependencyAnalysisTool",
    "SystemInfoTool",
    "ProcessInfoTool",
    "WebSearchDetailedTool",
    "EnhancedToolRegistry",
    "AgentCoordinator",
    "AgentType",
    "EnsembleChain",
    "ModelCouncil",
    "create_ensemble_chain",
    "create_model_council",
    "SpecializedAgentType",
    "SpecializedAgent",
    "DataScienceAgent",
    "WebDevelopmentAgent",
    "SecurityAnalysisAgent",
    "DevOpsAgent",
    "TestingAgent",
    "DocumentationAgent",
    "SpecializedAgentFactory",
    "SpecializedAgentCoordinator",
    "Message",
    "MessageType",
    "MessageStatus",
    "MessageTemplates",
    "CommunicationProtocol",
    "MessageBroker",
    "InMemoryProtocol",
    "SecureChannel",
    "ChannelManager",
    "AgentCommunicationLayer",
    "TeamFormationEngine",
    "AgentTeam",
    "TeamAssignment",
    "TeamRole",
    "AgentCapability",
    "TeamFormationStrategy",
    "create_capability_from_agent_type",
    "create_capability_from_specialized_agent_type",
    "AdvancedCoordinationEngine",
    "CoordinationStrategy",
    "ResourceType",
    "Resource",
    "Bid",
    "Task",
    "AgentState",
    "MarketBasedAllocation",
    "SwarmIntelligence",
    "HierarchicalCoordinator",
    "NegotiationProtocol",
    "AgentLearningSystem",
    "AgentMemory",
    "SharedKnowledgeBase",
    "MemoryEntry",
    "KnowledgeItem",
    "LearningPattern",
    "MemoryType",
    "KnowledgeSourceType",
    "ExperienceSharingSystem",
    "HistoricalTaskPatterns",
    "create_memory_from_task_result",
    "create_knowledge_from_solution",
    "MonitoringAnalyticsEngine",
    "MetricsCollector",
    "CollaborationAnalyzer",
    "RealTimeDashboard",
    "Metric",
    "Alert",
    "CollaborationStats",
    "MetricType",
    "AlertSeverity",
    "create_utilization_metric",
    "create_efficiency_metric",
    "WorkflowManager",
    "Workflow",
    "Subtask",
    "TaskStatus",
    "TaskPriority",
    "TaskType",
    "TaskDecompositionEngine",
    "DependencyManager",
    "CheckpointManager",
    "create_workflow_from_task",
    "get_next_ready_subtasks",
    "HumanSupervisionInterface",
    "SupervisionEngine",
    "SupervisionRequest",
    "HumanFeedback",
    "ApprovalRule",
    "SupervisionLevel",
    "DecisionCategory",
    "ApprovalStatus",
    "FeedbackType",
    "FeedbackIntegrationSystem",
    "create_supervision_request_for_task",
    "submit_human_feedback",
    "CrossDomainExpertiseSystem",
    "DomainBridgeAgent",
    "KnowledgeTranslationSystem",
    "CrossDomainCoordinator",
    "HybridReasoningEngine",
    "DomainKnowledge",
    "TranslationRule",
    "CrossDomainRequest",
    "DomainType",
    "TranslationType",
    "create_domain_knowledge",
    "create_translation_rule",
    "get_cross_domain_solution",
    "ResourceManagementSystem",
    "ResourceManager",
    "CostOptimizer",
    "PriorityScheduler",
    "Resource",
    "ResourcePool",
    "ResourceRequest",
    "ResourceAllocation",
    "ResourceType",
    "ResourcePoolType",
    "TaskPriority",
    "ResourceAllocationStatus",
    "create_compute_resource_pool",
    "create_memory_resource_pool",
    "request_resources_with_budget",
    "SecurityGovernanceSystem",
    "IdentityManager",
    "AccessControlManager",
    "AuditLogger",
    "PrivacyPreservationManager",
    "AgentIdentity",
    "AccessControlRule",
    "AuditRecord",
    "SecurityPolicy",
    "Permission",
    "SecurityLevel",
    "AuditEventType",
    "ComplianceStatus",
    "create_agent_identity",
    "create_access_control_rule",
    "check_security_compliance"
]
