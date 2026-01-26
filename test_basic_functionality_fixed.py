#!/usr/bin/env python3
"""
Basic test script to verify Xencode core functionality
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_core_imports():
    """Test that core modules can be imported without errors"""
    print("Testing core module imports...")
    
    try:
        from xencode.core import (
            create_file,
            read_file,
            write_file,
            delete_file,
            ModelManager,
            get_smart_default_model,
            get_available_models,
            ConversationMemory,
            ResponseCache
        )
        print("[OK] Core modules imported successfully")
    except Exception as e:
        print(f"[ERROR] Error importing core modules: {e}")
        return False
    
    try:
        from xencode.security import (
            InputValidator,
            sanitize_user_input,
            validate_file_operation
        )
        print("[OK] Security modules imported successfully")
    except Exception as e:
        print(f"[ERROR] Error importing security modules: {e}")
        return False
    
    try:
        from xencode.ai_ensembles import (
            EnsembleReasoner,
            QueryRequest,
            EnsembleMethod
        )
        print("[OK] Ensemble modules imported successfully")
    except Exception as e:
        print(f"[ERROR] Error importing ensemble modules: {e}")
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
                print(f"[ERROR] Missing ensemble method: {method}")
                return False
        
        print("[OK] All ensemble methods available")
        
        # Test creating an ensemble reasoner
        from xencode.ai_ensembles import EnsembleReasoner
        reasoner = EnsembleReasoner()
        print("[OK] EnsembleReasoner created successfully")
        
        return True
    except Exception as e:
        print(f"[ERROR] Error testing ensemble functionality: {e}")
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
        print("[OK] Input sanitization works correctly")
        
        # Test dangerous input detection
        dangerous_input = "rm -rf / dangerous command"
        sanitized = validator.sanitize_input(dangerous_input)
        assert "[FILTERED]" in sanitized, "Dangerous input not filtered"
        print("[OK] Dangerous input filtering works correctly")
        
        return True
    except Exception as e:
        print(f"[ERROR] Error testing security functionality: {e}")
        return False

def test_cache_functionality():
    """Test cache functionality"""
    print("\nTesting cache functionality...")
    
    try:
        from xencode.core.cache import ResponseCache
        cache = ResponseCache()
        
        # Test basic cache operations
        test_prompt = "test prompt"
        test_model = "test-model"
        test_response = "test response"
        
        # Set a value
        cache.set(test_prompt, test_model, test_response)
        print("[OK] Cache set operation successful")
        
        # Get the value
        retrieved = cache.get(test_prompt, test_model)
        assert retrieved == test_response, f"Expected {test_response}, got {retrieved}"
        print("[OK] Cache get operation successful")
        
        # Test cache stats
        stats = cache.get_stats()
        print(f"[OK] Cache stats retrieved: {stats['overall']['total_hits']} hits")
        
        return True
    except Exception as e:
        print(f"[ERROR] Error testing cache functionality: {e}")
        return False

def test_memory_functionality():
    """Test memory functionality"""
    print("\nTesting memory functionality...")
    
    try:
        from xencode.core.memory import ConversationMemory
        memory = ConversationMemory()
        
        # Test adding a message
        memory.add_message("user", "Hello, world!", "test-model")
        print("[OK] Memory add message successful")
        
        # Test getting context
        context = memory.get_context()
        assert len(context) == 1, f"Expected 1 message, got {len(context)}"
        assert context[0]["content"] == "Hello, world!", "Message content mismatch"
        print("[OK] Memory get context successful")
        
        return True
    except Exception as e:
        print(f"[ERROR] Error testing memory functionality: {e}")
        return False

def main():
    """Main test function"""
    print("Xencode Basic Functionality Test Suite")
    print("=" * 45)
    
    all_passed = True
    
    # Run basic tests
    all_passed &= test_core_imports()
    all_passed &= test_ensemble_functionality()
    all_passed &= test_security_functionality()
    all_passed &= test_cache_functionality()
    all_passed &= test_memory_functionality()
    
    print("\n" + "=" * 45)
    if all_passed:
        print("[SUCCESS] All basic tests passed! Core functionality is working correctly.")
        return 0
    else:
        print("[FAILURE] Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)