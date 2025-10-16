#!/usr/bin/env python3
"""
CLI Integration Test

Tests the complete Xencode CLI functionality without requiring
actual Ollama models or heavy ML dependencies.
"""

import subprocess
import sys
from pathlib import Path

def run_cli_command(cmd_args):
    """Run CLI command and return result"""
    try:
        result = subprocess.run(
            [sys.executable, "xencode_cli.py"] + cmd_args,
            capture_output=True,
            text=True,
            timeout=30
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Command timed out"
    except Exception as e:
        return -1, "", str(e)

def test_basic_commands():
    """Test basic CLI commands"""
    tests = [
        (["--help"], "Should show help"),
        (["version"], "Should show version"),
        (["--version"], "Should show version flag"),
        (["ollama", "--help"], "Should show ollama help"),
        (["query", "--help"], "Should show query help"),
        (["rlhf", "--help"], "Should show rlhf help"),
    ]
    
    print("🧪 Testing basic CLI commands...")
    
    for cmd_args, description in tests:
        print(f"  Testing: xencode {' '.join(cmd_args)}")
        returncode, stdout, stderr = run_cli_command(cmd_args)
        
        if returncode == 0:
            print(f"    ✅ {description}")
        else:
            print(f"    ❌ {description} - Error: {stderr}")
            return False
    
    return True

def test_token_voting():
    """Test token voting functionality"""
    print("\n🧠 Testing token voting system...")
    
    try:
        # Import and test token voting directly
        from xencode.ai_ensembles import TokenVoter
        
        voter = TokenVoter()
        
        # Test basic voting
        responses = ["Python is great", "Python is good", "Python is great"]
        result = voter.vote_tokens(responses)
        
        if "Python is great" in result:
            print("    ✅ Token voting works correctly")
        else:
            print(f"    ❌ Token voting failed: {result}")
            return False
        
        # Test consensus calculation
        consensus = voter.calculate_consensus(responses)
        if 0 <= consensus <= 1:
            print(f"    ✅ Consensus calculation works: {consensus:.3f}")
        else:
            print(f"    ❌ Consensus calculation failed: {consensus}")
            return False
        
        return True
        
    except Exception as e:
        print(f"    ❌ Token voting test failed: {e}")
        return False

def test_cli_structure():
    """Test CLI structure and imports"""
    print("\n🏗️ Testing CLI structure...")
    
    try:
        # Test imports
        from xencode.cli import cli
        from xencode.ai_ensembles import TokenVoter, EnsembleMethod
        from xencode.phase2_coordinator import Phase2Coordinator
        
        print("    ✅ All imports successful")
        
        # Test enum values
        methods = [EnsembleMethod.VOTE, EnsembleMethod.WEIGHTED, 
                  EnsembleMethod.CONSENSUS, EnsembleMethod.HYBRID]
        
        if len(methods) == 4:
            print("    ✅ Ensemble methods available")
        else:
            print("    ❌ Ensemble methods incomplete")
            return False
        
        return True
        
    except Exception as e:
        print(f"    ❌ CLI structure test failed: {e}")
        return False

def test_help_content():
    """Test help content quality"""
    print("\n📖 Testing help content...")
    
    returncode, stdout, stderr = run_cli_command(["--help"])
    
    if returncode != 0:
        print(f"    ❌ Help command failed: {stderr}")
        return False
    
    # Check for key content
    required_content = [
        "Xencode AI/ML Leviathan",
        "Ultimate Offline AI Assistant",
        "GitHub Copilot",
        "leviathan has awakened",
        "query",
        "ollama",
        "rlhf",
        "status"
    ]
    
    missing_content = []
    for content in required_content:
        if content not in stdout:
            missing_content.append(content)
    
    if missing_content:
        print(f"    ❌ Missing help content: {missing_content}")
        return False
    
    print("    ✅ Help content is comprehensive")
    return True

def test_version_info():
    """Test version information"""
    print("\n📋 Testing version information...")
    
    returncode, stdout, stderr = run_cli_command(["version"])
    
    if returncode != 0:
        print(f"    ❌ Version command failed: {stderr}")
        return False
    
    # Check version content
    version_content = [
        "Xencode AI/ML Leviathan v2.1.0",
        "GitHub Copilot",
        "<50ms inference",
        "10% SMAPE improvements",
        "100% privacy",
        "leviathan has awakened"
    ]
    
    missing_content = []
    for content in version_content:
        if content not in stdout:
            missing_content.append(content)
    
    if missing_content:
        print(f"    ❌ Missing version content: {missing_content}")
        return False
    
    print("    ✅ Version information is complete")
    return True

def main():
    """Run all CLI integration tests"""
    print("🐉 Xencode CLI Integration Test Suite")
    print("=" * 50)
    
    tests = [
        test_basic_commands,
        test_token_voting,
        test_cli_structure,
        test_help_content,
        test_version_info
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"    ⚠️ Test {test_func.__name__} failed")
        except Exception as e:
            print(f"    💥 Test {test_func.__name__} crashed: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎯 ALL TESTS PASSED - CLI is ready for action!")
        print("🐉 The leviathan's CLI interface is fully operational!")
        return 0
    else:
        print(f"❌ {total - passed} tests failed - CLI needs attention")
        return 1

if __name__ == "__main__":
    sys.exit(main())