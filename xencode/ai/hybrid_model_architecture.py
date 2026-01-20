"""
Hybrid Model Architecture for Xencode

Implements the ability to switch between local and cloud models based on task complexity,
with model chaining for complex workflows and dynamic model selection based on context,
performance requirements, and privacy needs.
"""

import asyncio
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Union
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor
import threading

from ..intelligent_model_selector import HardwareDetector, ModelRecommendationEngine
from ..advanced_cache_system import get_cache_manager
from ..multi_model_system import MultiModelManager
from ..ollama_fallback import OllamaFallbackManager


logger = logging.getLogger(__name__)


class ModelProvider(Enum):
    """Enumeration of model providers"""
    LOCAL_OLLAMA = "local_ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    COHERE = "cohere"
    HUGGINGFACE = "huggingface"
    CUSTOM = "custom"


class ModelType(Enum):
    """Types of models based on capabilities"""
    GENERAL = "general"
    CODE = "code"
    CHAT = "chat"
    EMBEDDING = "embedding"
    SPECIALIZED = "specialized"


@dataclass
class ModelSpec:
    """Specification for a model"""
    name: str
    provider: ModelProvider
    model_type: ModelType
    capabilities: List[str]  # e.g., ["reasoning", "coding", "math", "creative"]
    performance_score: float  # 0-100 scale
    privacy_level: int  # 1-5, 5 being most private (local)
    cost_per_token: float  # in USD
    max_context_length: int
    estimated_response_time: float  # in seconds


@dataclass
class TaskContext:
    """Context for a specific task"""
    task_type: str  # "coding", "analysis", "creative", "research", etc.
    sensitivity_level: int  # 1-5, 5 being most sensitive
    complexity_level: int  # 1-5, 5 being most complex
    urgency_level: int  # 1-5, 5 being most urgent
    context_size: int  # approximate token count
    required_capabilities: List[str]
    user_preferences: Dict[str, Any]


class ModelInterface(ABC):
    """Abstract interface for model providers"""
    
    @abstractmethod
    async def generate(self, prompt: str, context: Optional[Dict] = None) -> str:
        """Generate response for the given prompt"""
        pass
    
    @abstractmethod
    def get_model_spec(self) -> ModelSpec:
        """Get specification for this model"""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the model is healthy and responsive"""
        pass


class LocalOllamaModel(ModelInterface):
    """Wrapper for local Ollama models"""

    def __init__(self, model_name: str, model_selector=None):
        self.model_name = model_name
        self.model_selector = model_selector
        self.spec = self._create_spec(model_name)
    
    def _create_spec(self, model_name: str) -> ModelSpec:
        """Create model spec based on model characteristics"""
        # This would be populated based on actual model characteristics
        return ModelSpec(
            name=model_name,
            provider=ModelProvider.LOCAL_OLLAMA,
            model_type=ModelType.GENERAL,
            capabilities=["reasoning", "coding", "math"],
            performance_score=85.0,
            privacy_level=5,
            cost_per_token=0.0,  # Local models have no per-token cost
            max_context_length=32768,
            estimated_response_time=2.0
        )
    
    async def generate(self, prompt: str, context: Optional[Dict] = None) -> str:
        """Generate response using local Ollama model"""
        try:
            # Use the existing Ollama integration
            from .model_providers.ollama_provider import OllamaProvider
            provider = OllamaProvider()
            return await provider.generate(prompt, self.model_name)
        except Exception as e:
            logger.error(f"Error generating with local model {self.model_name}: {e}")
            raise
    
    def get_model_spec(self) -> ModelSpec:
        return self.spec
    
    async def health_check(self) -> bool:
        """Check if local Ollama is running"""
        try:
            from .model_providers.ollama_provider import OllamaProvider
            provider = OllamaProvider()
            return await provider.health_check()
        except:
            return False


class CloudModel(ModelInterface):
    """Base class for cloud model providers"""
    
    def __init__(self, model_name: str, provider: ModelProvider, api_key: str):
        self.model_name = model_name
        self.provider = provider
        self.api_key = api_key
        self.spec = self._create_spec(model_name, provider)
    
    def _create_spec(self, model_name: str, provider: ModelProvider) -> ModelSpec:
        """Create model spec based on provider and model"""
        # This would be populated based on actual provider characteristics
        if provider == ModelProvider.OPENAI:
            return ModelSpec(
                name=model_name,
                provider=provider,
                model_type=ModelType.CHAT,
                capabilities=["reasoning", "coding", "creative"],
                performance_score=95.0,
                privacy_level=1,
                cost_per_token=0.00001,  # Example cost
                max_context_length=128000,
                estimated_response_time=1.5
            )
        elif provider == ModelProvider.ANTHROPIC:
            return ModelSpec(
                name=model_name,
                provider=provider,
                model_type=ModelType.CHAT,
                capabilities=["reasoning", "long_context"],
                performance_score=92.0,
                privacy_level=1,
                cost_per_token=0.00002,
                max_context_length=200000,
                estimated_response_time=2.0
            )
        else:
            # Default spec
            return ModelSpec(
                name=model_name,
                provider=provider,
                model_type=ModelType.GENERAL,
                capabilities=["reasoning"],
                performance_score=80.0,
                privacy_level=1,
                cost_per_token=0.000015,
                max_context_length=32768,
                estimated_response_time=2.0
            )
    
    async def generate(self, prompt: str, context: Optional[Dict] = None) -> str:
        """Generate response using cloud provider - to be implemented by subclasses"""
        raise NotImplementedError
    
    def get_model_spec(self) -> ModelSpec:
        return self.spec
    
    async def health_check(self) -> bool:
        """Check if cloud provider is accessible"""
        try:
            # Test with a simple request
            await self.generate("health check", {"max_tokens": 5})
            return True
        except:
            return False


class OpenAIModel(CloudModel):
    """OpenAI model implementation"""
    
    def __init__(self, model_name: str, api_key: str):
        super().__init__(model_name, ModelProvider.OPENAI, api_key)
    
    async def generate(self, prompt: str, context: Optional[Dict] = None) -> str:
        """Generate response using OpenAI API"""
        try:
            import openai
            from openai import AsyncOpenAI
            
            client = AsyncOpenAI(api_key=self.api_key)
            
            params = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7
            }
            
            if context:
                if "max_tokens" in context:
                    params["max_tokens"] = context["max_tokens"]
                if "temperature" in context:
                    params["temperature"] = context["temperature"]
            
            response = await client.chat.completions.acreate(**params)
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating with OpenAI model {self.model_name}: {e}")
            raise


class AnthropicModel(CloudModel):
    """Anthropic model implementation"""
    
    def __init__(self, model_name: str, api_key: str):
        super().__init__(model_name, ModelProvider.ANTHROPIC, api_key)
    
    async def generate(self, prompt: str, context: Optional[Dict] = None) -> str:
        """Generate response using Anthropic API"""
        try:
            import anthropic
            from anthropic import AsyncAnthropic
            
            client = AsyncAnthropic(api_key=self.api_key)
            
            params = {
                "model": self.model_name,
                "prompt": f"\n\nHuman: {prompt}\n\nAssistant:",
                "max_tokens_to_sample": 1000,
                "temperature": 0.7
            }
            
            if context:
                if "max_tokens" in context:
                    params["max_tokens_to_sample"] = context["max_tokens"]
                if "temperature" in context:
                    params["temperature"] = context["temperature"]
            
            response = await client.completions.create(**params)
            return response.completion
        except Exception as e:
            logger.error(f"Error generating with Anthropic model {self.model_name}: {e}")
            raise


class ModelRouter:
    """Routes tasks to appropriate models based on context and requirements"""

    def __init__(self, model_selector=None):
        self.model_selector = model_selector
        self.local_models = {}
        self.cloud_models = {}
        self.executor = ThreadPoolExecutor(max_workers=10)
        self._lock = threading.Lock()

        # Initialize default models
        self._initialize_default_models()
    
    def _initialize_default_models(self):
        """Initialize default models"""
        # Add local models
        try:
            # For now, just add a default model
            # In a real implementation, we'd detect available models
            self.local_models["default"] = LocalOllamaModel("default", self.model_selector)
        except Exception as e:
            logger.warning(f"Could not initialize local models: {e}")
    
    def register_cloud_model(self, model: CloudModel):
        """Register a cloud model"""
        with self._lock:
            self.cloud_models[model.model_name] = model
    
    def register_local_model(self, model: LocalOllamaModel):
        """Register a local model"""
        with self._lock:
            self.local_models[model.model_name] = model
    
    def select_best_model(self, task_context: TaskContext) -> ModelInterface:
        """Select the best model based on task context"""
        with self._lock:
            # Consider privacy requirements first
            if task_context.sensitivity_level >= 4:
                # High sensitivity - prefer local models
                for model_name, model in self.local_models.items():
                    spec = model.get_model_spec()
                    if self._model_meets_requirements(spec, task_context):
                        return model
            
            # Consider complexity and urgency
            candidate_models = []
            
            # Add local models
            for model_name, model in self.local_models.items():
                spec = model.get_model_spec()
                if self._model_meets_requirements(spec, task_context):
                    score = self._calculate_model_score(spec, task_context)
                    candidate_models.append((model, score))
            
            # Add cloud models if privacy allows
            if task_context.sensitivity_level <= 3:
                for model_name, model in self.cloud_models.items():
                    spec = model.get_model_spec()
                    if self._model_meets_requirements(spec, task_context):
                        score = self._calculate_model_score(spec, task_context)
                        candidate_models.append((model, score))
            
            # Sort by score (higher is better)
            candidate_models.sort(key=lambda x: x[1], reverse=True)
            
            if candidate_models:
                return candidate_models[0][0]
            else:
                # Fallback to first available local model
                if self.local_models:
                    return list(self.local_models.values())[0]
                elif self.cloud_models:
                    return list(self.cloud_models.values())[0]
                else:
                    raise RuntimeError("No models available")
    
    def _model_meets_requirements(self, spec: ModelSpec, task_context: TaskContext) -> bool:
        """Check if model meets task requirements"""
        # Check if model has required capabilities
        for req_cap in task_context.required_capabilities:
            if req_cap not in spec.capabilities:
                return False
        
        # Check context length
        if task_context.context_size > spec.max_context_length:
            return False
        
        return True
    
    def _calculate_model_score(self, spec: ModelSpec, task_context: TaskContext) -> float:
        """Calculate a score for how well a model fits the task"""
        score = 0.0
        
        # Performance score (0-100)
        score += spec.performance_score * 0.3
        
        # Privacy consideration (more private is better for sensitive tasks)
        if task_context.sensitivity_level >= 4:
            score += spec.privacy_level * 10
        else:
            score += (5 - spec.privacy_level) * 2  # Lower privacy is OK for non-sensitive tasks
        
        # Urgency consideration (faster response preferred for urgent tasks)
        if task_context.urgency_level >= 4:
            score += (10 / spec.estimated_response_time) * 5
        else:
            score += (10 / spec.estimated_response_time) * 2
        
        # Complexity consideration (better models for complex tasks)
        if task_context.complexity_level >= 4:
            score += spec.performance_score * 0.2
        
        # Cost consideration (lower cost preferred unless performance is critical)
        cost_factor = 1.0 / (spec.cost_per_token + 0.000001)  # Small constant to avoid division by zero
        score += cost_factor * 0.1
        
        return score
    
    async def route_request(self, prompt: str, task_context: TaskContext) -> str:
        """Route a request to the best model and execute it"""
        model = self.select_best_model(task_context)
        
        try:
            return await model.generate(prompt)
        except Exception as e:
            logger.error(f"Model {model.get_model_spec().name} failed: {e}")
            
            # Try fallback models
            with self._lock:
                all_models = list(self.local_models.values()) + list(self.cloud_models.values())
                
                for fallback_model in all_models:
                    if fallback_model != model:  # Don't retry the same model
                        try:
                            spec = fallback_model.get_model_spec()
                            if self._model_meets_requirements(spec, task_context):
                                logger.info(f"Trying fallback model: {spec.name}")
                                return await fallback_model.generate(prompt)
                        except Exception as fallback_e:
                            logger.error(f"Fallback model {spec.name} also failed: {fallback_e}")
                            continue
            
            raise RuntimeError(f"All models failed for task: {e}")


class ModelChain:
    """Chains multiple models for complex workflows"""
    
    def __init__(self, router: ModelRouter):
        self.router = router
        self.chain_steps = []
    
    def add_step(self, task_type: str, prompt_template: str, required_capabilities: List[str] = None):
        """Add a step to the model chain"""
        step = {
            "task_type": task_type,
            "prompt_template": prompt_template,
            "required_capabilities": required_capabilities or []
        }
        self.chain_steps.append(step)
    
    async def execute_chain(self, initial_input: str, context: TaskContext) -> str:
        """Execute the model chain"""
        current_input = initial_input
        
        for i, step in enumerate(self.chain_steps):
            # Update context for this step
            step_context = TaskContext(
                task_type=step["task_type"],
                sensitivity_level=context.sensitivity_level,
                complexity_level=context.complexity_level,
                urgency_level=context.urgency_level,
                context_size=len(current_input),
                required_capabilities=step["required_capabilities"],
                user_preferences=context.user_preferences
            )
            
            # Format prompt with current input
            prompt = step["prompt_template"].format(input=current_input)
            
            # Route to appropriate model
            result = await self.router.route_request(prompt, step_context)
            
            # Update current input for next step
            current_input = result
            
            logger.info(f"Completed chain step {i+1}/{len(self.chain_steps)}")
        
        return current_input


class HybridModelManager:
    """Main class managing the hybrid model architecture"""

    def __init__(self):
        self.model_selector = None  # Will initialize with actual class if needed
        self.router = ModelRouter(self.model_selector)
        self.cache_manager = get_cache_manager()
        self.default_api_keys = {}

        # Initialize with common cloud providers
        self._setup_cloud_providers()
    
    def _setup_cloud_providers(self):
        """Setup common cloud providers if API keys are available"""
        # Check for API keys in environment or config
        import os
        
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            # Add common OpenAI models
            for model_name in ["gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"]:
                try:
                    model = OpenAIModel(model_name, openai_key)
                    self.router.register_cloud_model(model)
                except Exception as e:
                    logger.warning(f"Could not register OpenAI model {model_name}: {e}")
        
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        if anthropic_key:
            # Add common Anthropic models
            for model_name in ["claude-3-opus-20240229", "claude-3-sonnet-20240229"]:
                try:
                    model = AnthropicModel(model_name, anthropic_key)
                    self.router.register_cloud_model(model)
                except Exception as e:
                    logger.warning(f"Could not register Anthropic model {model_name}: {e}")
    
    def set_api_key(self, provider: ModelProvider, api_key: str):
        """Set API key for a cloud provider"""
        self.default_api_keys[provider] = api_key
        
        # Register models for this provider
        if provider == ModelProvider.OPENAI:
            for model_name in ["gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"]:
                try:
                    model = OpenAIModel(model_name, api_key)
                    self.router.register_cloud_model(model)
                except Exception as e:
                    logger.warning(f"Could not register OpenAI model {model_name}: {e}")
        elif provider == ModelProvider.ANTHROPIC:
            for model_name in ["claude-3-opus-20240229", "claude-3-sonnet-20240229"]:
                try:
                    model = AnthropicModel(model_name, api_key)
                    self.router.register_cloud_model(model)
                except Exception as e:
                    logger.warning(f"Could not register Anthropic model {model_name}: {e}")
    
    async def generate(self, prompt: str, task_context: Optional[TaskContext] = None) -> str:
        """Generate response using the best available model"""
        if task_context is None:
            task_context = TaskContext(
                task_type="general",
                sensitivity_level=2,
                complexity_level=3,
                urgency_level=2,
                context_size=len(prompt),
                required_capabilities=[],
                user_preferences={}
            )
        
        # Check cache first
        cache_key = f"hybrid_gen:{hash(prompt)}:{task_context.task_type}"
        cached_result = await self.cache_manager.aget(cache_key)
        
        if cached_result is not None:
            logger.info("Returning cached result for hybrid model generation")
            return cached_result
        
        # Generate with appropriate model
        result = await self.router.route_request(prompt, task_context)
        
        # Cache the result
        await self.cache_manager.aset(cache_key, result, ttl=3600)  # Cache for 1 hour
        
        return result
    
    def create_chain(self) -> ModelChain:
        """Create a new model chain for complex workflows"""
        return ModelChain(self.router)
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all registered models"""
        results = {
            "local_models": {},
            "cloud_models": {},
            "overall_status": "healthy"
        }
        
        # Check local models
        for name, model in self.router.local_models.items():
            try:
                is_healthy = await model.health_check()
                results["local_models"][name] = is_healthy
                if not is_healthy:
                    results["overall_status"] = "degraded"
            except Exception as e:
                results["local_models"][name] = False
                results["overall_status"] = "degraded"
                logger.error(f"Health check failed for local model {name}: {e}")
        
        # Check cloud models
        for name, model in self.router.cloud_models.items():
            try:
                is_healthy = await model.health_check()
                results["cloud_models"][name] = is_healthy
                if not is_healthy:
                    results["overall_status"] = "degraded"
            except Exception as e:
                results["cloud_models"][name] = False
                results["overall_status"] = "degraded"
                logger.error(f"Health check failed for cloud model {name}: {e}")
        
        return results


# Global hybrid model manager instance
_hybrid_manager: Optional[HybridModelManager] = None


def get_hybrid_model_manager() -> HybridModelManager:
    """Get the global hybrid model manager instance"""
    global _hybrid_manager
    if _hybrid_manager is None:
        _hybrid_manager = HybridModelManager()
    return _hybrid_manager


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    async def test_hybrid_architecture():
        """Test the hybrid model architecture"""
        print("Testing Hybrid Model Architecture...")
        
        manager = get_hybrid_model_manager()
        
        # Test basic generation
        print("\n1. Testing basic generation:")
        context = TaskContext(
            task_type="general",
            sensitivity_level=2,
            complexity_level=2,
            urgency_level=2,
            context_size=100,
            required_capabilities=["reasoning"],
            user_preferences={}
        )
        
        result = await manager.generate("What is the capital of France?", context)
        print(f"Response: {result}")
        
        # Test with high sensitivity (should prefer local model)
        print("\n2. Testing with high sensitivity (privacy-focused):")
        sensitive_context = TaskContext(
            task_type="analysis",
            sensitivity_level=5,  # Very sensitive
            complexity_level=2,
            urgency_level=2,
            context_size=200,
            required_capabilities=["reasoning"],
            user_preferences={}
        )
        
        result = await manager.generate("Analyze this sensitive code snippet for vulnerabilities", sensitive_context)
        print(f"Response: {result}")
        
        # Test model chaining
        print("\n3. Testing model chaining:")
        chain = manager.create_chain()
        chain.add_step("summarization", "Summarize the following: {input}")
        chain.add_step("analysis", "Analyze the sentiment of this summary: {input}")
        
        chained_result = await chain.execute_chain(
            "Artificial intelligence is a wonderful field that combines computer science and cognitive psychology to create systems that can perform tasks typically requiring human intelligence.",
            context
        )
        print(f"Chained response: {chained_result}")
        
        # Test health check
        print("\n4. Testing health check:")
        health = await manager.health_check()
        print(f"Health status: {health['overall_status']}")
        print(f"Local models: {len(health['local_models'])} checked")
        print(f"Cloud models: {len(health['cloud_models'])} checked")
        
        print("\n✅ Hybrid Model Architecture tests completed!")
    
    # Run the test
    asyncio.run(test_hybrid_architecture())