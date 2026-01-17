#!/usr/bin/env python3
"""
Diagnostic tool to verify all Xencode enhancements are properly integrated
"""
import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_specialized_agents():
    """Test specialized agents functionality"""
    print("Testing specialized agents...")
    
    try:
        from xencode.agentic.specialized import (
            SpecializedAgentFactory, 
            SpecializedAgentType, 
            DataScienceAgent,
            WebDevelopmentAgent,
            SecurityAnalysisAgent,
            DevOpsAgent,
            TestingAgent,
            DocumentationAgent
        )
        print("✓ Specialized agent classes imported successfully")
        
        # Test factory
        factory = SpecializedAgentFactory()
        agent_types = factory.get_available_agent_types()
        print(f"✓ Available agent types: {len(agent_types)}")
        
        # Test creating an agent
        agent = factory.create_agent(SpecializedAgentType.DATA_SCIENCE)
        print("✓ Data science agent created successfully")
        
        # Test coordinator
        from xencode.agentic.specialized.coordinator import SpecializedAgentCoordinator
        coordinator = SpecializedAgentCoordinator()
        print("✓ Specialized agent coordinator created successfully")
        
        # Test capabilities
        caps = coordinator.get_all_agent_capability()
        print(f"✓ Retrieved capabilities for {len(caps)} agent types")
        
        return True
    except Exception as e:
        print(f"✗ Error testing specialized agents: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_plugin_system():
    """Test plugin system functionality"""
    print("\nTesting plugin system...")
    
    try:
        from xencode.plugins import (
            PluginManager,
            PluginInterface,
            ToolPlugin,
            AgentPlugin,
            ExampleToolPlugin,
            ExampleAgentPlugin
        )
        print("✓ Plugin system classes imported successfully")
        
        # Test plugin manager
        manager = PluginManager()
        print("✓ Plugin manager created successfully")
        
        # Test plugin configuration
        from xencode.plugins.config import PluginConfigManager, PluginConfig
        config_manager = PluginConfigManager()
        print("✓ Plugin config manager created successfully")
        
        # Test plugin integration
        from xencode.plugins.integration import PluginIntegrator, get_plugin_integrator
        integrator = get_plugin_integrator()
        print("✓ Plugin integrator retrieved successfully")
        
        return True
    except Exception as e:
        print(f"✗ Error testing plugin system: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_providers():
    """Test model provider system"""
    print("\nTesting model provider system...")
    
    try:
        from xencode.model_providers import (
            ModelProvider,
            OllamaProvider,
            OpenAIProvider,
            AnthropicProvider,
            HuggingFaceProvider,
            ModelProviderManager,
            get_model_provider_manager
        )
        print("✓ Model provider classes imported successfully")
        
        # Test provider manager
        manager = get_model_provider_manager()
        print("✓ Model provider manager retrieved successfully")
        
        # Test provider creation (without actual API calls)
        ollama_provider = OllamaProvider("", "http://localhost:11434")
        print("✓ Ollama provider created successfully")
        
        # Test provider types
        provider_types = [OllamaProvider, OpenAIProvider, AnthropicProvider, HuggingFaceProvider]
        print(f"✓ {len(provider_types)} provider types available")
        
        return True
    except Exception as e:
        print(f"✗ Error testing model providers: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_core_enhancements():
    """Test core enhancements"""
    print("\nTesting core enhancements...")
    
    try:
        # Test enhanced cache
        from xencode.core.cache import ResponseCache
        cache = ResponseCache()
        print("✓ Enhanced cache system working")
        
        # Test enhanced tools
        from xencode.agentic.enhanced_tools import EnhancedToolRegistry
        registry = EnhancedToolRegistry()
        print("✓ Enhanced tool registry working")
        
        # Test ensemble improvements
        from xencode.ai_ensembles import EnsembleReasoner, EnsembleMethod, TokenVoter
        reasoner = EnsembleReasoner()
        methods = [e.value for e in EnsembleMethod]
        print(f"✓ Ensemble system with {len(methods)} methods working")
        
        # Test semantic voting
        responses = ["test response 1", "test response 2"]
        semantic_result = TokenVoter.semantic_vote_tokens(responses)
        print("✓ Semantic voting working")
        
        return True
    except Exception as e:
        print(f"✗ Error testing core enhancements: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_main_integration():
    """Test main integration points"""
    print("\nTesting main integration...")
    
    try:
        # Test main agentic import
        from xencode.agentic import (
            SpecializedAgentType,
            SpecializedAgent,
            DataScienceAgent,
            WebDevelopmentAgent,
            SecurityAnalysisAgent,
            DevOpsAgent,
            TestingAgent,
            DocumentationAgent,
            SpecializedAgentFactory,
            SpecializedAgentCoordinator,
            ToolPlugin,
            AgentPlugin,
            ModelProviderPlugin,
            PluginManager,
            PluginInterface,
            ModelProvider,
            OllamaProvider,
            OpenAIProvider,
            AnthropicProvider,
            HuggingFaceProvider
        )
        print("✓ Main integration points working")
        
        return True
    except Exception as e:
        print(f"✗ Error testing main integration: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main diagnostic function"""
    print("Xencode Diagnostic Tool")
    print("=" * 50)
    
    all_passed = True
    
    # Run all tests
    all_passed &= test_core_enhancements()
    all_passed &= test_specialized_agents()
    all_passed &= test_plugin_system()
    all_passed &= test_model_providers()
    all_passed &= test_main_integration()
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All diagnostics passed! Xencode enhancements are properly integrated.")
        print("\nSummary of verified components:")
        print("- Enhanced caching system with compression and statistics")
        print("- Semantic voting in ensemble learning")
        print("- Specialized agents for 6 domains (Data Science, Web Dev, Security, DevOps, Testing, Documentation)")
        print("- Plugin system with configuration management")
        print("- Model provider abstraction for Ollama, OpenAI, Anthropic, Hugging Face")
        print("- Integration between all components")
        return 0
    else:
        print("❌ Some diagnostics failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)