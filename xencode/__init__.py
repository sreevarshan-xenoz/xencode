"""
Xencode - Professional AI Assistant Package

A comprehensive offline-first AI assistant with Claude-style interface,
enhanced with user-centric development framework, technical debt management,
and AI ethics monitoring.
"""

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
except ImportError:
    HardwareDetector = ModelRecommendationEngine = None

try:
    from .advanced_cache_system import HybridCacheManager, get_cache_manager
except ImportError:
    HybridCacheManager = get_cache_manager = None

try:
    from .smart_config_manager import ConfigurationManager, XencodeConfig
except ImportError:
    ConfigurationManager = XencodeConfig = None

try:
    from .advanced_error_handler import ErrorHandler, ErrorCategory
except ImportError:
    ErrorHandler = ErrorCategory = None

try:
    from .phase2_coordinator import Phase2Coordinator
except ImportError:
    Phase2Coordinator = None

# Enhancement systems (Phase 3+) - with optional imports
try:
    from .user_feedback_system import (
        UserFeedbackManager, FeedbackType, UserJourneyEvent,
        get_feedback_manager, collect_user_feedback, track_user_event
    )
except ImportError:
    UserFeedbackManager = FeedbackType = UserJourneyEvent = None
    get_feedback_manager = collect_user_feedback = track_user_event = None

try:
    from .technical_debt_manager import (
        TechnicalDebtManager, DebtType, DebtSeverity, get_debt_manager
    )
except ImportError:
    TechnicalDebtManager = DebtType = DebtSeverity = get_debt_manager = None

try:
    from .ai_ethics_framework import (
        EthicsFramework, BiasType, EthicsViolationType,
        get_ethics_framework, analyze_ai_interaction
    )
except ImportError:
    EthicsFramework = BiasType = EthicsViolationType = None
    get_ethics_framework = analyze_ai_interaction = None

try:
    from .enhancement_integration import (
        EnhancementSystemsIntegration, get_enhancement_integration,
        track_model_selection, track_query_response, collect_response_feedback,
        report_system_error, get_system_insights
    )
except ImportError:
    EnhancementSystemsIntegration = get_enhancement_integration = None
    track_model_selection = track_query_response = collect_response_feedback = None
    report_system_error = get_system_insights = None

# Warp Terminal (Phase 3.5+) - with optional imports
try:
    from .warp_terminal import (
        WarpTerminal, CommandBlock, StreamingOutputParser, 
        LazyCommandBlock, GPUAcceleratedRenderer, example_ai_suggester
    )
except ImportError:
    WarpTerminal = CommandBlock = StreamingOutputParser = None
    LazyCommandBlock = GPUAcceleratedRenderer = example_ai_suggester = None

try:
    from .enhanced_command_palette import (
        EnhancedCommandPalette, WarpTerminalWithPalette, 
        CommandSuggestion, FuzzyMatcher
    )
except ImportError:
    EnhancedCommandPalette = WarpTerminalWithPalette = None
    CommandSuggestion = FuzzyMatcher = None

try:
    from .warp_ui_components import (
        OutputRenderer, WarpLayoutManager
    )
except ImportError:
    OutputRenderer = WarpLayoutManager = None

try:
    from .warp_testing_harness import (
        CommandTestingHarness, TestResult, run_comprehensive_test
    )
except ImportError:
    CommandTestingHarness = TestResult = run_comprehensive_test = None

try:
    from .warp_ai_integration import (
        WarpAIIntegration, ProjectAnalyzer, AdvancedAISuggester,
        ProjectContext, CommandSuggestionContext, get_warp_ai_integration
    )
except ImportError:
    WarpAIIntegration = ProjectAnalyzer = AdvancedAISuggester = None
    ProjectContext = CommandSuggestionContext = get_warp_ai_integration = None

__all__ = [
    # Core systems
    "ContextCacheManager", "ModelStabilityManager", "SmartContextManager",
    
    # Phase 2 systems
    "HardwareDetector", "ModelRecommendationEngine",
    "HybridCacheManager", "get_cache_manager",
    "ConfigurationManager", "XencodeConfig",
    "ErrorHandler", "ErrorCategory",
    "Phase2Coordinator",
    
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
    "ProjectContext", "CommandSuggestionContext", "get_warp_ai_integration"
]
