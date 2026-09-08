"""
Xencode - Professional AI Assistant Package

A comprehensive offline-first AI assistant with Claude-style interface,
enhanced with user-centric development framework, technical debt management,
and AI ethics monitoring.
"""

import logging

logger = logging.getLogger(__name__)
__version__ = "2.1.0"  # Updated for enhancement systems
__author__ = "Sreevarshan"
__license__ = "MIT"

# Core systems
from .context_cache_manager import ContextCacheManager
from .model_stability_manager import ModelStabilityManager
from .smart_context_system import SmartContextManager

# Phase 2 systems (with optional imports)
try:
    from .intelligent_model_selector import HardwareDetector, ModelRecommendationEngine
except ImportError as e:
    logger.warning("Failed to import ModelRecommendationEngine: %s", e)
    HardwareDetector = ModelRecommendationEngine = None

try:
    from .advanced_cache_system import HybridCacheManager, get_cache_manager
except ImportError as e:
    logger.warning("Failed to import get_cache_manager: %s", e)
    HybridCacheManager = get_cache_manager = None

try:
    from .smart_config_manager import ConfigurationManager, XencodeConfig
except ImportError as e:
    logger.warning("Failed to import XencodeConfig: %s", e)
    ConfigurationManager = XencodeConfig = None

try:
    from .advanced_error_handler import ErrorCategory, ErrorHandler
except ImportError as e:
    logger.warning("Failed to import ErrorCategory: %s", e)
    ErrorHandler = ErrorCategory = None

try:
    from .phase2_coordinator import Phase2Coordinator
except ImportError as e:
    logger.warning("Failed to import Phase2Coordinator: %s", e)
    Phase2Coordinator = None

# AI/ML Phase 6 systems - with optional imports
try:
    from .ai_ensembles import (
        EnsembleMethod,
        EnsembleReasoner,
        ModelResponse,
        ModelTier,
        QueryRequest,
        QueryResponse,
        TokenVoter,
        create_ensemble_reasoner,
    )
except ImportError as e:
    logger.warning("Failed to import ModelResponse: %s", e)
    EnsembleReasoner = QueryRequest = QueryResponse = ModelResponse = None
    EnsembleMethod = ModelTier = TokenVoter = create_ensemble_reasoner = None

try:
    from .ollama_optimizer import (
        BenchmarkResult,
        ModelInfo,
        ModelStatus,
        OllamaOptimizer,
        QuantizationLevel,
        create_ollama_optimizer,
    )
except ImportError as e:
    logger.warning("Failed to import QuantizationLevel: %s", e)
    OllamaOptimizer = ModelInfo = BenchmarkResult = QuantizationLevel = None
    ModelStatus = create_ollama_optimizer = None

# Ollama Fallback Manager for auto-start and installation
try:
    from .ollama_fallback import OllamaFallbackManager, ensure_ollama
except ImportError as e:
    logger.warning("Failed to import ensure_ollama: %s", e)
    OllamaFallbackManager = ensure_ollama = None

try:
    from .rlhf_tuner import (
        CodePair,
        RLHFConfig,
        RLHFTuner,
        SyntheticDataGenerator,
        TrainingMetrics,
        create_rlhf_tuner,
    )
except ImportError as e:
    logger.warning("Failed to import TrainingMetrics: %s", e)
    RLHFTuner = RLHFConfig = CodePair = TrainingMetrics = None
    SyntheticDataGenerator = create_rlhf_tuner = None

# Enhancement systems (Phase 3+) - with optional imports
try:
    from .user_feedback_system import (
        FeedbackType,
        UserFeedbackManager,
        UserJourneyEvent,
        collect_user_feedback,
        get_feedback_manager,
        track_user_event,
    )
except ImportError as e:
    logger.warning("Failed to import UserJourneyEvent: %s", e)
    UserFeedbackManager = FeedbackType = UserJourneyEvent = None
    get_feedback_manager = collect_user_feedback = track_user_event = None

try:
    from .technical_debt_manager import (
        DebtSeverity,
        DebtType,
        TechnicalDebtManager,
        get_debt_manager,
    )
except ImportError as e:
    logger.warning("Failed to import get_debt_manager: %s", e)
    TechnicalDebtManager = DebtType = DebtSeverity = get_debt_manager = None

try:
    from .ai_ethics_framework import (
        BiasType,
        EthicsFramework,
        EthicsViolationType,
        analyze_ai_interaction,
        get_ethics_framework,
    )
except ImportError as e:
    logger.warning("Failed to import EthicsViolationType: %s", e)
    EthicsFramework = BiasType = EthicsViolationType = None
    get_ethics_framework = analyze_ai_interaction = None

try:
    from .enhancement_integration import (
        EnhancementSystemsIntegration,
        collect_response_feedback,
        get_enhancement_integration,
        get_system_insights,
        report_system_error,
        track_model_selection,
        track_query_response,
    )
except ImportError as e:
    logger.warning("Failed to import get_enhancement_integration: %s", e)
    EnhancementSystemsIntegration = get_enhancement_integration = None
    track_model_selection = track_query_response = collect_response_feedback = None
    report_system_error = get_system_insights = None

# Warp Terminal (Phase 3.5+) - with optional imports
try:
    from .warp_terminal import (
        CommandBlock,
        GPUAcceleratedRenderer,
        LazyCommandBlock,
        StreamingOutputParser,
        WarpTerminal,
        example_ai_suggester,
    )
except ImportError as e:
    logger.warning("Failed to import StreamingOutputParser: %s", e)
    WarpTerminal = CommandBlock = StreamingOutputParser = None
    LazyCommandBlock = GPUAcceleratedRenderer = example_ai_suggester = None

try:
    from .enhanced_command_palette import (
        CommandSuggestion,
        EnhancedCommandPalette,
        FuzzyMatcher,
        WarpTerminalWithPalette,
    )
except ImportError as e:
    logger.warning("Failed to import WarpTerminalWithPalette: %s", e)
    EnhancedCommandPalette = WarpTerminalWithPalette = None
    CommandSuggestion = FuzzyMatcher = None

try:
    from .warp_ui_components import OutputRenderer, WarpLayoutManager
except ImportError as e:
    logger.warning("Failed to import WarpLayoutManager: %s", e)
    OutputRenderer = WarpLayoutManager = None

try:
    from .warp_testing_harness import (
        CommandTestingHarness,
        TestResult,
        run_comprehensive_test,
    )
except ImportError as e:
    logger.warning("Failed to import run_comprehensive_test: %s", e)
    CommandTestingHarness = TestResult = run_comprehensive_test = None

try:
    from .warp_ai_integration import (
        AdvancedAISuggester,
        CommandSuggestionContext,
        ProjectAnalyzer,
        ProjectContext,
        WarpAIIntegration,
        get_warp_ai_integration,
    )
except ImportError as e:
    logger.warning("Failed to import AdvancedAISuggester: %s", e)
    WarpAIIntegration = ProjectAnalyzer = AdvancedAISuggester = None
    ProjectContext = CommandSuggestionContext = get_warp_ai_integration = None

# Feature system
try:
    from .features import (
        FeatureBase,
        FeatureConfig,
        FeatureConfigManager,
        FeatureError,
        FeatureManager,
        FeatureStatus,
        FeatureSystemConfig,
    )
except ImportError as e:
    logger.warning("Failed to import FeatureError: %s", e)
    FeatureBase = FeatureConfig = FeatureStatus = FeatureError = None
    FeatureManager = FeatureSystemConfig = FeatureConfigManager = None

__all__ = [
    # Core systems
    "ContextCacheManager", "ModelStabilityManager", "SmartContextManager",

    # Phase 2 systems
    "HardwareDetector", "ModelRecommendationEngine",
    "HybridCacheManager", "get_cache_manager",
    "ConfigurationManager", "XencodeConfig",
    "ErrorHandler", "ErrorCategory",
    "Phase2Coordinator",

    # AI/ML Phase 6 systems
    "EnsembleReasoner", "QueryRequest", "QueryResponse", "ModelResponse",
    "EnsembleMethod", "ModelTier", "TokenVoter", "create_ensemble_reasoner",
    "OllamaOptimizer", "ModelInfo", "BenchmarkResult", "QuantizationLevel",
    "ModelStatus", "create_ollama_optimizer",
    "OllamaFallbackManager", "ensure_ollama",
    "RLHFTuner", "RLHFConfig", "CodePair", "TrainingMetrics",
    "SyntheticDataGenerator", "create_rlhf_tuner",

    # Enhancement systems
    "UserFeedbackManager", "FeedbackType", "UserJourneyEvent",
    "get_feedback_manager", "collect_user_feedback", "track_user_event",
    "TechnicalDebtManager", "DebtType", "DebtSeverity", "get_debt_manager",
    "EthicsFramework", "BiasType", "EthicsViolationType",
    "get_ethics_framework", "analyze_ai_interaction",
    "EnhancementSystemsIntegration", "get_enhancement_integration",
    "track_model_selection", "track_query_response", "collect_response_feedback",
    "report_system_error", "get_system_insights",

    # Warp Terminal systems
    "WarpTerminal", "CommandBlock", "StreamingOutputParser",
    "LazyCommandBlock", "GPUAcceleratedRenderer", "example_ai_suggester",
    "EnhancedCommandPalette", "WarpTerminalWithPalette",
    "CommandSuggestion", "FuzzyMatcher",
    "OutputRenderer", "WarpLayoutManager",
    "CommandTestingHarness", "TestResult", "run_comprehensive_test",
    "WarpAIIntegration", "ProjectAnalyzer", "AdvancedAISuggester",
    "ProjectContext", "CommandSuggestionContext", "get_warp_ai_integration",

    # Feature system
    "FeatureBase", "FeatureConfig", "FeatureStatus", "FeatureError",
    "FeatureManager", "FeatureSystemConfig", "FeatureConfigManager"
]
