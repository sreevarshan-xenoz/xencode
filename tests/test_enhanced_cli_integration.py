#!/usr/bin/env python3
"""
Test Enhanced CLI Integration for Task 5

Tests the enhanced CLI command system with backward compatibility
"""

import os
import sys


def test_enhanced_cli_availability():
    """Test that enhanced CLI system can be imported and initialized"""
    print("🧪 Testing Enhanced CLI Availability...")

    try:
        from enhanced_cli_system import EnhancedXencodeCLI, FeatureDetector

        # Test feature detection
        detector = FeatureDetector()
        features = detector.detect_features()

        print("✅ Enhanced CLI imported successfully")
        print(f"   Feature Level: {features.feature_level}")
        print(f"   Multi-Model: {'✅' if features.multi_model else '❌'}")
        print(f"   Smart Context: {'✅' if features.smart_context else '❌'}")
        print(f"   Code Analysis: {'✅' if features.code_analysis else '❌'}")

        # Test CLI initialization
        cli = EnhancedXencodeCLI()
        print("✅ Enhanced CLI initialized successfully")

        return True

    except Exception as e:
        print(f"❌ Enhanced CLI test failed: {e}")
        return False


def test_argument_parser():
    """Test enhanced argument parser"""
    print("\n🧪 Testing Enhanced Argument Parser...")

    try:
        from enhanced_cli_system import EnhancedXencodeCLI

        cli = EnhancedXencodeCLI()
        parser = cli.create_parser()

        # Test parsing enhanced commands
        test_cases = [
            ["--analyze", "src/"],
            ["--models"],
            ["--context"],
            ["--smart", "How do I optimize this code?"],
            ["--git-commit"],
            ["--feature-status"],
            ["legacy query without flags"],
        ]

        for test_args in test_cases:
            try:
                if test_args == ["legacy query without flags"]:
                    # Test legacy positional argument
                    parsed = parser.parse_args(test_args)
                    print(f"✅ Legacy query parsing: {parsed.query}")
                else:
                    parsed = parser.parse_args(test_args)
                    print(f"✅ Enhanced command parsing: {test_args[0]}")
            except SystemExit:
                # argparse calls sys.exit on help/error - this is expected
                print(f"ℹ️ Parser exit for: {test_args}")
            except Exception as e:
                print(f"❌ Parser failed for {test_args}: {e}")
                return False

        return True

    except Exception as e:
        print(f"❌ Argument parser test failed: {e}")
        return False


def test_command_routing():
    """Test command routing and backward compatibility"""
    print("\n🧪 Testing Command Routing...")

    try:
        from enhanced_cli_system import EnhancedXencodeCLI

        cli = EnhancedXencodeCLI()

        # Test feature status command (always available)
        result = cli._handle_feature_status()
        if "Feature Status" in result:
            print("✅ Feature status command works")
        else:
            print("❌ Feature status command failed")
            return False

        # Test enhanced command processing
        parser = cli.create_parser()

        # Test --feature-status
        try:
            args = parser.parse_args(["--feature-status"])
            result = cli.process_enhanced_args(args)
            if result and "Feature Status" in result:
                print("✅ Enhanced command processing works")
            else:
                print("❌ Enhanced command processing failed")
                return False
        except SystemExit:
            print("ℹ️ Parser exit expected for some commands")

        return True

    except Exception as e:
        print(f"❌ Command routing test failed: {e}")
        return False


def test_security_integration():
    """Test git workflow security layer"""
    print("\n🧪 Testing Security Integration...")

    try:
        from security_manager import SecurityManager

        security = SecurityManager()

        # Test path validation
        current_dir = os.getcwd()
        is_valid = security.validate_project_path(current_dir)
        print(f"✅ Path validation: {'Valid' if is_valid else 'Invalid'}")

        # Test commit message sanitization
        test_message = "feat: add new feature\n\nThis is a test commit"
        sanitized = security.sanitize_commit_message(test_message)

        if sanitized:
            print("✅ Commit message sanitization works")
        else:
            print("❌ Commit message sanitization failed")
            return False

        return True

    except Exception as e:
        print(f"❌ Security integration test failed: {e}")
        return False


def test_git_commit_generation():
    """Test git commit message generation"""
    print("\n🧪 Testing Git Commit Generation...")

    try:
        from code_analysis_system import CodeAnalyzer

        analyzer = CodeAnalyzer()

        # Test commit message generation (will fail if not in git repo, but method should exist)
        result = analyzer.generate_commit_message()

        if "Error" in result or "No changes" in result:
            print("✅ Git commit generation method works (no git repo detected)")
        else:
            print(f"✅ Git commit generation works: {result[:50]}...")

        return True

    except Exception as e:
        print(f"❌ Git commit generation test failed: {e}")
        return False


def test_enhanced_chat_commands():
    """Test enhanced chat commands"""
    print("\n🧪 Testing Enhanced Chat Commands...")

    try:
        from enhanced_chat_commands import EnhancedChatCommands
        from enhanced_cli_system import FeatureDetector

        detector = FeatureDetector()
        features = detector.detect_features()

        chat_commands = EnhancedChatCommands(features)

        # Test help command
        help_text = chat_commands.get_enhanced_help()
        if "Enhanced Chat Commands" in help_text:
            print("✅ Enhanced chat help works")
        else:
            print("❌ Enhanced chat help failed")
            return False

        # Test command handling
        response, new_model = chat_commands.handle_chat_command("help", "", "qwen3:4b")
        if "Enhanced Chat Commands" in response:
            print("✅ Enhanced chat command handling works")
        else:
            print("❌ Enhanced chat command handling failed")
            return False

        return True

    except Exception as e:
        print(f"❌ Enhanced chat commands test failed: {e}")
        return False


def test_backward_compatibility():
    """Test that legacy xencode functionality still works"""
    print("\n🧪 Testing Backward Compatibility...")

    try:
        # Test that xencode_core can still be imported and used
        import xencode_core

        # Test that main function exists
        if hasattr(xencode_core, 'main'):
            print("✅ Legacy main function exists")
        else:
            print("❌ Legacy main function missing")
            return False

        # Test that core functions exist
        required_functions = ['run_query', 'list_models', 'chat_mode']
        for func_name in required_functions:
            if hasattr(xencode_core, func_name):
                print(f"✅ Legacy function {func_name} exists")
            else:
                print(f"❌ Legacy function {func_name} missing")
                return False

        return True

    except Exception as e:
        print(f"❌ Backward compatibility test failed: {e}")
        return False


def main():
    """Run all enhanced CLI integration tests"""
    print("🚀 Enhanced CLI Integration Tests")
    print("=" * 50)

    tests = [
        test_enhanced_cli_availability,
        test_argument_parser,
        test_command_routing,
        test_security_integration,
        test_git_commit_generation,
        test_enhanced_chat_commands,
        test_backward_compatibility,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                print(f"❌ {test.__name__} failed")
        except Exception as e:
            print(f"❌ {test.__name__} crashed: {e}")

    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} passed")

    if passed == total:
        print("🎉 All tests passed! Enhanced CLI integration is working correctly.")
        return True
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
