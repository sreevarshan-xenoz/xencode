"""
Agentic capabilities for Xencode using LangChain.
All imports are guarded for graceful degradation when langchain is not installed.
"""

# Core manager (requires langchain)
try:
    from .manager import LangChainManager
except ImportError:
    LangChainManager = None

# Basic tools
try:
    from .tools import ReadFileTool, WriteFileTool, ExecuteCommandTool
except ImportError:
    ReadFileTool = WriteFileTool = ExecuteCommandTool = None

# Advanced tools
try:
    from .advanced_tools import (
        GitStatusTool, GitDiffTool, GitLogTool, GitCommitTool,
        WebSearchTool, CodeAnalysisTool, ToolRegistry
    )
except ImportError:
    GitStatusTool = GitDiffTool = GitLogTool = GitCommitTool = None
    WebSearchTool = CodeAnalysisTool = ToolRegistry = None

# Enhanced tools
try:
    from .enhanced_tools import (
        GitBranchTool, GitPushTool, GitPullTool, FindFileTool,
        FileStatTool, DependencyAnalysisTool, SystemInfoTool,
        ProcessInfoTool, WebSearchDetailedTool, EnhancedToolRegistry
    )
except ImportError:
    GitBranchTool = GitPushTool = GitPullTool = FindFileTool = None
    FileStatTool = DependencyAnalysisTool = SystemInfoTool = None
    ProcessInfoTool = WebSearchDetailedTool = EnhancedToolRegistry = None

# Coordinator
try:
    from .coordinator import AgentCoordinator, AgentType
except ImportError:
    AgentCoordinator = AgentType = None

# Ensemble integration
try:
    from .ensemble_integration import EnsembleChain, ModelCouncil, create_ensemble_chain, create_model_council
except ImportError:
    EnsembleChain = ModelCouncil = create_ensemble_chain = create_model_council = None

# Specialized agents
try:
    from .specialized import (
        SpecializedAgentType, SpecializedAgent, DataScienceAgent,
        WebDevelopmentAgent, SecurityAnalysisAgent, DevOpsAgent,
        TestingAgent, DocumentationAgent, SpecializedAgentFactory
    )
except ImportError:
    SpecializedAgentType = SpecializedAgent = DataScienceAgent = None
    WebDevelopmentAgent = SecurityAnalysisAgent = DevOpsAgent = None
    TestingAgent = DocumentationAgent = SpecializedAgentFactory = None

try:
    from .specialized.coordinator import SpecializedAgentCoordinator
except ImportError:
    SpecializedAgentCoordinator = None

# Communication
try:
    from .communication import (
        Message, MessageType, MessageStatus, MessageTemplates,
        CommunicationProtocol, MessageBroker, InMemoryProtocol,
        SecureChannel, ChannelManager
    )
except ImportError:
    Message = MessageType = MessageStatus = MessageTemplates = None
    CommunicationProtocol = MessageBroker = InMemoryProtocol = None
    SecureChannel = ChannelManager = None

try:
    from .communication.integration import AgentCommunicationLayer
except ImportError:
    AgentCommunicationLayer = None

# Team formation
try:
    from .team_formation import (
        TeamFormationEngine, AgentTeam, TeamAssignment, TeamRole,
        AgentCapability, TeamFormationStrategy,
        create_capability_from_agent_type, create_capability_from_specialized_agent_type
    )
except ImportError:
    TeamFormationEngine = AgentTeam = TeamAssignment = TeamRole = None
    AgentCapability = TeamFormationStrategy = None
    create_capability_from_agent_type = create_capability_from_specialized_agent_type = None

# Coordination strategies
try:
    from .coordination_strategies import (
        AdvancedCoordinationEngine, CoordinationStrategy, ResourceType,
        Resource, Bid, Task, AgentState, MarketBasedAllocation,
        SwarmIntelligence, HierarchicalCoordinator, NegotiationProtocol
    )
except ImportError:
    AdvancedCoordinationEngine = CoordinationStrategy = ResourceType = None
    Resource = Bid = Task = AgentState = MarketBasedAllocation = None
    SwarmIntelligence = HierarchicalCoordinator = NegotiationProtocol = None

# Memory & learning
try:
    from .memory_learning import (
        AgentLearningSystem, AgentMemory, SharedKnowledgeBase, MemoryEntry,
        KnowledgeItem, LearningPattern, MemoryType, KnowledgeSourceType,
        ExperienceSharingSystem, HistoricalTaskPatterns,
        create_memory_from_task_result, create_knowledge_from_solution
    )
except ImportError:
    AgentLearningSystem = AgentMemory = SharedKnowledgeBase = None
    MemoryEntry = KnowledgeItem = LearningPattern = MemoryType = None
    KnowledgeSourceType = ExperienceSharingSystem = HistoricalTaskPatterns = None
    create_memory_from_task_result = create_knowledge_from_solution = None

# Monitoring
try:
    from .monitoring_analytics import (
        MonitoringAnalyticsEngine, MetricsCollector, CollaborationAnalyzer,
        RealTimeDashboard, Metric, Alert, CollaborationStats, MetricType,
        AlertSeverity, create_utilization_metric, create_efficiency_metric
    )
except ImportError:
    MonitoringAnalyticsEngine = MetricsCollector = CollaborationAnalyzer = None
    RealTimeDashboard = Metric = Alert = CollaborationStats = None
    MetricType = AlertSeverity = create_utilization_metric = create_efficiency_metric = None

# Workflow
try:
    from .workflow_management import (
        WorkflowManager, Workflow, Subtask, TaskStatus, TaskPriority, TaskType,
        TaskDecompositionEngine, DependencyManager, CheckpointManager,
        create_workflow_from_task, get_next_ready_subtasks
    )
except ImportError:
    WorkflowManager = Workflow = Subtask = TaskStatus = None
    TaskPriority = TaskType = TaskDecompositionEngine = None
    DependencyManager = CheckpointManager = None
    create_workflow_from_task = get_next_ready_subtasks = None

# Human supervision
try:
    from .human_supervision import (
        HumanSupervisionInterface, SupervisionEngine, SupervisionRequest,
        HumanFeedback, ApprovalRule, SupervisionLevel, DecisionCategory,
        ApprovalStatus, FeedbackType, FeedbackIntegrationSystem,
        create_supervision_request_for_task, submit_human_feedback
    )
except ImportError:
    HumanSupervisionInterface = SupervisionEngine = SupervisionRequest = None
    HumanFeedback = ApprovalRule = SupervisionLevel = DecisionCategory = None
    ApprovalStatus = FeedbackType = FeedbackIntegrationSystem = None
    create_supervision_request_for_task = submit_human_feedback = None

# Cross-domain expertise
try:
    from .cross_domain_expertise import (
        CrossDomainExpertiseSystem, DomainBridgeAgent, KnowledgeTranslationSystem,
        CrossDomainCoordinator, HybridReasoningEngine, DomainKnowledge,
        TranslationRule, CrossDomainRequest, DomainType, TranslationType,
        create_domain_knowledge, create_translation_rule, get_cross_domain_solution
    )
except ImportError:
    CrossDomainExpertiseSystem = DomainBridgeAgent = KnowledgeTranslationSystem = None
    CrossDomainCoordinator = HybridReasoningEngine = DomainKnowledge = None
    TranslationRule = CrossDomainRequest = DomainType = TranslationType = None
    create_domain_knowledge = create_translation_rule = get_cross_domain_solution = None

# Resource management
try:
    from .resource_management import (
        ResourceManagementSystem, ResourceManager, CostOptimizer, PriorityScheduler,
        Resource, ResourcePool, ResourceRequest, ResourceAllocation, ResourceType,
        ResourcePoolType, TaskPriority, ResourceAllocationStatus,
        create_compute_resource_pool, create_memory_resource_pool, request_resources_with_budget
    )
except ImportError:
    ResourceManagementSystem = ResourceManager = CostOptimizer = None
    PriorityScheduler = Resource = ResourcePool = ResourceRequest = None
    ResourceAllocation = ResourceType = ResourcePoolType = TaskPriority = None
    ResourceAllocationStatus = create_compute_resource_pool = None
    create_memory_resource_pool = request_resources_with_budget = None

# Security governance
try:
    from .security_governance import (
        SecurityGovernanceSystem, IdentityManager, AccessControlManager, AuditLogger,
        PrivacyPreservationManager, AgentIdentity, AccessControlRule, AuditRecord,
        SecurityPolicy, Permission, SecurityLevel, AuditEventType, ComplianceStatus,
        create_agent_identity, create_access_control_rule, check_security_compliance
    )
except ImportError:
    SecurityGovernanceSystem = IdentityManager = AccessControlManager = None
    AuditLogger = PrivacyPreservationManager = AgentIdentity = None
    AccessControlRule = AuditRecord = SecurityPolicy = Permission = None
    SecurityLevel = AuditEventType = ComplianceStatus = None
    create_agent_identity = create_access_control_rule = check_security_compliance = None
