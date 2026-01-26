"""
Hybrid Model Architecture Integration

Integrates the hybrid model architecture with the existing Xencode system.
"""

import asyncio
from typing import Dict, List, Optional, Any, Union
from pathlib import Path

from ..warp_terminal import WarpTerminal
from .hybrid_model_architecture import (
    HybridModelManager,
    TaskContext,
    ModelProvider,
    get_hybrid_model_manager
)
from .hybrid_model_config import (
    get_hybrid_config_manager,
    ModelPreferenceType
)


class HybridModelIntegration:
    """Integration layer for hybrid model architecture with Xencode"""
    
    def __init__(self):
        self.model_manager = get_hybrid_model_manager()
        self.config_manager = get_hybrid_config_manager()
        self.default_task_context = TaskContext(
            task_type="general",
            sensitivity_level=2,
            complexity_level=3,
            urgency_level=2,
            context_size=100,
            required_capabilities=["reasoning"],
            user_preferences={}
        )
    
    def set_api_key(self, provider: Union[ModelProvider, str], api_key: str):
        """Set API key for a cloud provider"""
        if isinstance(provider, ModelProvider):
            provider_name = provider.value
        else:
            provider_name = provider
        
        self.config_manager.set_provider_api_key(provider_name, api_key)
        self.model_manager.set_api_key(provider, api_key)
    
    async def generate_response(
        self, 
        prompt: str, 
        task_type: str = "general",
        sensitivity_level: int = 2,
        complexity_level: int = 3,
        urgency_level: int = 2,
        required_capabilities: Optional[List[str]] = None
    ) -> str:
        """Generate response using the hybrid model architecture"""
        if required_capabilities is None:
            required_capabilities = ["reasoning"]
        
        task_context = TaskContext(
            task_type=task_type,
            sensitivity_level=sensitivity_level,
            complexity_level=complexity_level,
            urgency_level=urgency_level,
            context_size=len(prompt),
            required_capabilities=required_capabilities,
            user_preferences={}
        )
        
        return await self.model_manager.generate(prompt, task_context)
    
    def create_workflow_chain(self):
        """Create a model chain for complex workflows"""
        return self.model_manager.create_chain()
    
    async def analyze_task_requirements(self, prompt: str) -> TaskContext:
        """Analyze a prompt to determine task requirements"""
        # This would use AI to analyze the prompt and determine requirements
        # For now, we'll use simple heuristics
        
        task_type = "general"
        required_capabilities = ["reasoning"]
        
        # Analyze prompt for task type
        prompt_lower = prompt.lower()
        if any(word in prompt_lower for word in ["code", "program", "function", "algorithm", "debug"]):
            task_type = "coding"
            required_capabilities.extend(["coding", "logic"])
        elif any(word in prompt_lower for word in ["analyze", "summarize", "explain", "describe"]):
            task_type = "analysis"
            required_capabilities.extend(["analysis", "comprehension"])
        elif any(word in prompt_lower for word in ["creative", "story", "write", "draft", "compose"]):
            task_type = "creative"
            required_capabilities.extend(["creativity", "writing"])
        
        # Estimate complexity based on prompt length and keywords
        complexity_level = min(5, max(1, len(prompt) // 200 + 1))
        
        # Default context
        return TaskContext(
            task_type=task_type,
            sensitivity_level=2,  # Default to moderate sensitivity
            complexity_level=complexity_level,
            urgency_level=2,  # Default to moderate urgency
            context_size=len(prompt),
            required_capabilities=required_capabilities,
            user_preferences={}
        )
    
    async def smart_generate(self, prompt: str) -> str:
        """Generate response with automatic task analysis"""
        task_context = await self.analyze_task_requirements(prompt)
        return await self.model_manager.generate(prompt, task_context)
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the hybrid model system"""
        return await self.model_manager.health_check()
    
    def update_preference(self, preference_type: ModelPreferenceType, weight: float):
        """Update model selection preferences"""
        self.config_manager.update_preference(preference_type, weight)
    
    def get_available_providers(self) -> List[str]:
        """Get list of available providers"""
        return self.config_manager.get_provider_priority()


# Global integration instance
_integration: Optional[HybridModelIntegration] = None


def get_hybrid_integration() -> HybridModelIntegration:
    """Get the global hybrid model integration instance"""
    global _integration
    if _integration is None:
        _integration = HybridModelIntegration()
    return _integration


# Integration with Warp Terminal
def integrate_with_warp_terminal(warp_terminal):
    """Integrate hybrid model architecture with Warp terminal"""
    integration = get_hybrid_integration()
    
    # Enhance the AI suggester to use hybrid models
    original_suggester = warp_terminal.ai_suggester
    
    async def enhanced_ai_suggester(recent_commands: List[str]) -> List[str]:
        """Enhanced AI suggester using hybrid model architecture"""
        if not recent_commands:
            return []
        
        # Analyze recent commands to generate context-aware suggestions
        context_str = " ".join(recent_commands[-5:])  # Last 5 commands
        
        # Generate suggestions using hybrid model
        prompt = f"""
        Based on these recent commands: "{context_str}"
        
        Suggest 5 relevant terminal commands that would logically follow.
        Focus on commands that are commonly used in development workflows.
        Respond with only the commands, one per line, without explanations.
        """
        
        try:
            response = await integration.smart_generate(prompt)
            suggestions = [line.strip() for line in response.split('\n') if line.strip()]
            return suggestions[:5]  # Return up to 5 suggestions
        except Exception as e:
            # Fall back to original suggester if hybrid model fails
            print(f"Hybrid model suggestion failed: {e}")
            if original_suggester:
                return original_suggester(recent_commands)
            return []
    
    # Replace the AI suggester
    warp_terminal.ai_suggester = enhanced_ai_suggester
    
    return warp_terminal


# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def test_integration():
        """Test the hybrid model integration"""
        print("Testing Hybrid Model Integration...")
        
        integration = get_hybrid_integration()
        
        # Test basic generation
        print("\n1. Testing basic generation:")
        response = await integration.generate_response(
            "What is the capital of France?",
            task_type="knowledge"
        )
        print(f"Response: {response}")
        
        # Test with high sensitivity (should prefer local model)
        print("\n2. Testing with high sensitivity:")
        response = await integration.generate_response(
            "Analyze this sensitive configuration data for security issues",
            task_type="analysis",
            sensitivity_level=5
        )
        print(f"Response: {response}")
        
        # Test smart generation with automatic analysis
        print("\n3. Testing smart generation:")
        response = await integration.smart_generate(
            "Write a Python function to calculate factorial"
        )
        print(f"Response: {response}")
        
        # Test workflow chaining
        print("\n4. Testing workflow chaining:")
        chain = integration.create_workflow_chain()
        chain.add_step("draft", "Draft a response to this: {input}")
        chain.add_step("refine", "Refine this response: {input}")
        
        chained_response = await chain.execute_chain(
            "How do I center a div in CSS?",
            integration.default_task_context
        )
        print(f"Chained response: {chained_response}")
        
        # Test health check
        print("\n5. Testing health check:")
        health = await integration.health_check()
        print(f"Health status: {health['overall_status']}")
        print(f"Available providers: {integration.get_available_providers()}")
        
        # Test preference update
        print("\n6. Testing preference update:")
        integration.update_preference(ModelPreferenceType.PRIVACY, 0.9)
        print("Updated privacy preference to 0.9")
        
        print("\n✅ Hybrid Model Integration tests completed!")
    
    # Run the test
    asyncio.run(test_integration())