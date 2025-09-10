#!/usr/bin/env python3
"""
Security Test Runner

Runs security tests during development to ensure all security protections work correctly.
This script should be run before committing any changes to the security manager.
"""

import sys
import subprocess
import os


def run_command(command, description):
    """Run a command and return success status"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} - PASSED")
            return True
        else:
            print(f"❌ {description} - FAILED")
            print(f"Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ {description} - ERROR: {e}")
        return False


def main():
    """Run all security tests"""
    print("🔒 Security Manager Test Suite")
    print("=" * 50)
    
    tests = [
        ("python test_security_manager.py", "Unit Tests"),
        ("python -c 'from security_manager import SecurityManager; sm = SecurityManager(); print(\"Import test passed\")'", "Import Test"),
        ("python demo_security_manager.py > /dev/null 2>&1", "Demo Test"),
    ]
    
    passed = 0
    total = len(tests)
    
    for command, description in tests:
        if run_command(command, description):
            passed += 1
        print()
    
    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All security tests passed! Ready for integration.")
        return 0
    else:
        print("❌ Some tests failed. Please fix issues before proceeding.")
        return 1


if __name__ == "__main__":
    exit(main())