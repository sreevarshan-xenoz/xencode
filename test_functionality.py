#!/usr/bin/env python3
"""
Test script to verify Xencode functionality
"""
import asyncio
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all modules can be imported without errors"""
    print("Testing module imports...")
    
    try:
        from xencode.agentic import (
            LangChainManager,
            ReadFileTool,
            WriteFileTool,
            ExecuteCommandTool,
            GitStatusTool,
            GitDiffTool,
            GitLogTool,
            GitCommitTool,
            WebSearchTool,
            CodeAnalysisTool,
            ToolRegistry,
            GitBranchTool,
            GitPushTool,
            GitPullTool,
            FindFileTool,
            FileStatTool,
            DependencyAnalysisTool,
            SystemInfoTool,
            ProcessInfoTool,
            WebSearchDetailedTool,
            EnhancedToolRegistry,
            AgentCoordinator,
            AgentType,
            EnsembleChain,
            ModelCouncil,
            create_ensemble_chain,
            create_model_council
        )
        print("✓ Agentic modules imported successfully")
    except Exception as e:
        print(f"✗ Error importing agentic modules: {e}")
        return False
    
    try:
        from xencode.ai_ensembles import (
            EnsembleReasoner,
            QueryRequest,
            EnsembleMethod,
            ModelResponse,
            QueryResponse
        )
        print("✓ Ensemble modules imported successfully")
    except Exception as e:
        print(f"✗ Error importing ensemble modules: {e}")
        return False
    
    try:
        from xencode.core import (
            create_file,
            read_file,
            write_file,
            delete_file,
            ModelManager,
            get_smart_default_model,
            get_available_models,
            list_models,
            update_model,
            ConversationMemory,
            ResponseCache,
            APIClient,
            get_api_client,
            close_api_client
        )
        print("✓ Core modules imported successfully")
    except Exception as e:
        print(f"✗ Error importing core modules: {e}")
        return False
    
    try:
        from xencode.security import (
            InputValidator,
            sanitize_user_input,
            validate_file_operation,
            APIResponseValidator,
            validate_api_response,
            sanitize_api_response
        )
        print("✓ Security modules imported successfully")
    except Exception as e:
        print(f"✗ Error importing security modules: {e}")
        return False
    
    return True

def test_ensemble_functionality():
    """Test ensemble functionality"""
    print("\nTesting ensemble functionality...")
    
    try:
        from xencode.ai_ensembles import EnsembleMethod
        
        # Check that all methods exist
        methods = [e.value for e in EnsembleMethod]
        expected_methods = ["vote", "weighted", "consensus", "hybrid", "semantic"]
        
        for method in expected_methods:
            if method not in methods:
                print(f"✗ Missing ensemble method: {method}")
                return False
        
        print("✓ All ensemble methods available")
        
        # Test creating an ensemble reasoner
        from xencode.ai_ensembles import EnsembleReasoner
        reasoner = EnsembleReasoner()
        print("✓ EnsembleReasoner created successfully")
        
        return True
    except Exception as e:
        print(f"✗ Error testing ensemble functionality: {e}")
        return False

def test_agentic_functionality():
    """Test agentic functionality"""
    print("\nTesting agentic functionality...")
    
    try:
        from xencode.agentic.coordinator import AgentCoordinator, AgentType
        
        # Test creating coordinator
        coordinator = AgentCoordinator()
        print("✓ AgentCoordinator created successfully")
        
        # Check agent types
        expected_types = ["code", "research", "execution", "general", "planning"]
        available_types = [agent_type.value for agent_type in AgentType]
        
        for agent_type in expected_types:
            if agent_type not in available_types:
                print(f"✗ Missing agent type: {agent_type}")
                return False
        
        print("✓ All agent types available")
        
        # Test enhanced tools
        from xencode.agentic.enhanced_tools import EnhancedToolRegistry
        registry = EnhancedToolRegistry()
        tools = registry.get_all_tools()
        print(f"✓ EnhancedToolRegistry created with {len(tools)} tools")
        
        return True
    except Exception as e:
        print(f"✗ Error testing agentic functionality: {e}")
        return False

def test_core_functionality():
    """Test core functionality"""
    print("\nTesting core functionality...")
    
    try:
        # Test cache functionality
        from xencode.core.cache import ResponseCache
        cache = ResponseCache()
        print("✓ ResponseCache created successfully")
        
        # Test memory functionality
        from xencode.core.memory import ConversationMemory
        memory = ConversationMemory()
        print("✓ ConversationMemory created successfully")
        
        # Test model manager
        from xencode.core.models import ModelManager
        manager = ModelManager()
        print("✓ ModelManager created successfully")
        
        return True
    except Exception as e:
        print(f"✗ Error testing core functionality: {e}")
        return False

def test_security_functionality():
    """Test security functionality"""
    print("\nTesting security functionality...")
    
    try:
        from xencode.security.validation import InputValidator
        validator = InputValidator()
        
        # Test basic validation
        test_input = "normal input"
        sanitized = validator.sanitize_input(test_input)
        assert sanitized == test_input, "Sanitization changed normal input"
        print("✓ Input sanitization works correctly")
        
        # Test dangerous input detection
        dangerous_input = "rm -rf / dangerous command"
        sanitized = validator.sanitize_input(dangerous_input)
        assert "[FILTERED]" in sanitized, "Dangerous input not filtered"
        print("✓ Dangerous input filtering works correctly")
        
        return True
    except Exception as e:
        print(f"✗ Error testing security functionality: {e}")
        return False

async def test_async_functionality():
    """Test async functionality"""
    print("\nTesting async functionality...")
    
    try:
        from xencode.ai_ensembles import create_ensemble_reasoner
        reasoner = await create_ensemble_reasoner()
        print("✓ Async ensemble reasoner created successfully")
        
        # Test ensemble chain creation
        from xencode.agentic.ensemble_integration import create_ensemble_chain
        ensemble_chain = await create_ensemble_chain()
        print("✓ Async ensemble chain created successfully")
        
        return True
    except Exception as e:
        print(f"✗ Error testing async functionality: {e}")
        return False

async def main():
    """Main test function"""
    print("Xencode Functionality Test Suite")
    print("=" * 40)
    
    all_passed = True
    
    # Run all tests
    all_passed &= test_imports()
    all_passed &= test_ensemble_functionality()
    all_passed &= test_agentic_functionality()
    all_passed &= test_core_functionality()
    all_passed &= test_security_functionality()
    all_passed &= await test_async_functionality()
    
    print("\n" + "=" * 40)
    if all_passed:
        print("🎉 All tests passed! Xencode is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)