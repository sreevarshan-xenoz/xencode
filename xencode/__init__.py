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

# Phase 2 systems
from .intelligent_model_selector import HardwareDetector, ModelRecommendationEngine
from .advanced_cache_system import HybridCacheManager, get_cache_manager
from .smart_config_manager import ConfigurationManager, XencodeConfig
from .advanced_error_handler import ErrorHandler, ErrorCategory
from .phase2_coordinator import Phase2Coordinator

# Enhancement systems (Phase 3+)
from .user_feedback_system import (
    UserFeedbackManager, FeedbackType, UserJourneyEvent,
    get_feedback_manager, collect_user_feedback, track_user_event
)
from .technical_debt_manager import (
    TechnicalDebtManager, DebtType, DebtSeverity, get_debt_manager
)
from .ai_ethics_framework import (
    EthicsFramework, BiasType, EthicsViolationType,
    get_ethics_framework, analyze_ai_interaction
)
from .enhancement_integration import (
    EnhancementSystemsIntegration, get_enhancement_integration,
    track_model_selection, track_query_response, collect_response_feedback,
    report_system_error, get_system_insights
)

# Warp Terminal (Phase 3.5+)
from .warp_terminal import (
    WarpTerminal, CommandBlock, StreamingOutputParser, 
    LazyCommandBlock, GPUAcceleratedRenderer, example_ai_suggester
)
from .enhanced_command_palette import (
    EnhancedCommandPalette, WarpTerminalWithPalette, 
    CommandSuggestion, FuzzyMatcher
)
from .warp_ui_components import (
    OutputRenderer, WarpLayoutManager
)
from .warp_testing_harness import (
    CommandTestingHarness, TestResult, run_comprehensive_test
)
from .warp_ai_integration import (
    WarpAIIntegration, ProjectAnalyzer, AdvancedAISuggester,
    ProjectContext, CommandSuggestionContext, get_warp_ai_integration
)

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
