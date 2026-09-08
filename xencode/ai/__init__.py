"""
AI package for Xencode.

Provides advanced AI capabilities including hybrid model architecture,
adaptive reasoning, prompt optimization, and knowledge transfer.
"""

try:
    from .hybrid_model_architecture import HybridModelManager, ModelChain, ModelRouter
except ImportError:
    ModelRouter = ModelChain = HybridModelManager = None

try:
    from .hybrid_model_config import HybridModelConfig
except ImportError:
    HybridModelConfig = None

try:
    from .hybrid_model_integration import HybridModelIntegration
except ImportError:
    HybridModelIntegration = None

try:
    from .orchestrator import ModelOrchestrator
except ImportError:
    ModelOrchestrator = None

try:
    from .prompt_optimizer import PromptOptimizer
except ImportError:
    PromptOptimizer = None

try:
    from .adaptive_reasoning import AdaptiveReasoningEngine
except ImportError:
    AdaptiveReasoningEngine = None

try:
    from .knowledge_transfer import KnowledgeTransferEngine
except ImportError:
    KnowledgeTransferEngine = None

try:
    from .finetuned_models import FineTunedModelManager
except ImportError:
    FineTunedModelManager = None

__all__ = [
    "ModelRouter",
    "ModelChain",
    "HybridModelManager",
    "HybridModelConfig",
    "HybridModelIntegration",
    "ModelOrchestrator",
    "PromptOptimizer",
    "AdaptiveReasoningEngine",
    "KnowledgeTransferEngine",
    "FineTunedModelManager",
]
