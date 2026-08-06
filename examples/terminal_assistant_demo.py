#!/usr/bin/env python3
"""
Terminal Assistant Feature Demo

Demonstrates the Terminal Assistant CLI and TUI functionality.
"""

import asyncio

from xencode.features import FeatureConfig
from xencode.features.terminal_assistant import TerminalAssistantFeature


async def demo_terminal_assistant():
    """Demo terminal assistant functionality"""
    print("🖥️  Terminal Assistant Feature Demo\n")
    print("=" * 60)

    # Initialize feature
    config = FeatureConfig(
        name="terminal_assistant",
        enabled=True,
        config={
            'history_size': 1000,
            'context_aware': True,
            'learning_enabled': True,
            'suggestion_limit': 5
        }
    )

    feature = TerminalAssistantFeature(config)
    await feature._initialize()

    print("\n✅ Terminal Assistant initialized successfully!")
    print(f"   Name: {feature.name}")
    print(f"   Description: {feature.description}")
    print(f"   Status: {feature.get_status().value}")

    # Demo 1: Command Suggestions
    print("\n" + "=" * 60)
    print("Demo 1: Command Suggestions")
    print("=" * 60)

    # Record some commands first
    await feature.record_command('git status', success=True)
    await feature.record_command('git add .', success=True)
    await feature.record_command('git commit -m "test"', success=True)
    await feature.record_command('npm install', success=True)
    await feature.record_command('npm test', success=True)

    print("\n📝 Recorded 5 sample commands")

    # Get suggestions
    suggestions = await feature.suggest_commands(partial='git')
    print("\n💡 Suggestions for 'git':")
    for i, suggestion in enumerate(suggestions[:3], 1):
        print(f"   {i}. {suggestion['command']}")
        print(f"      Score: {suggestion.get('score', 0):.1f} | Source: {suggestion.get('source', 'unknown')}")
        if suggestion.get('explanation'):
            print(f"      {suggestion['explanation']}")

    # Demo 2: Command Explanation
    print("\n" + "=" * 60)
    print("Demo 2: Command Explanation")
    print("=" * 60)

    explanation = await feature.explain_command('git commit -m "message"')
    print(f"\n📖 Explanation for: {explanation['command']}")
    print(f"   Description: {explanation['description']}")
    if explanation.get('arguments'):
        print(f"   Arguments: {len(explanation['arguments'])} found")
    if explanation.get('examples'):
        print("   Examples:")
        for ex in explanation['examples'][:2]:
            print(f"      • {ex}")
    if explanation.get('warnings'):
        print(f"   ⚠️  Warnings: {len(explanation['warnings'])}")

    # Demo 3: Error Fixing
    print("\n" + "=" * 60)
    print("Demo 3: Error Fix Suggestions")
    print("=" * 60)

    fixes = await feature.fix_error(
        command='npm install',
        error='command not found: npm'
    )
    print("\n🔧 Fix suggestions for 'command not found: npm':")
    for i, fix in enumerate(fixes[:3], 1):
        print(f"   {i}. {fix['fix']}")
        print(f"      Confidence: {fix['confidence']:.0%}")
        print(f"      {fix['explanation']}")
        if fix.get('requires_install'):
            print(f"      📦 Install: {fix.get('install_command', 'N/A')}")

    # Demo 4: Command History
    print("\n" + "=" * 60)
    print("Demo 4: Command History Search")
    print("=" * 60)

    history = await feature.search_history('git')
    print(f"\n📜 Found {len(history)} commands matching 'git':")
    for i, cmd in enumerate(history[:3], 1):
        status = "✅" if cmd.get('success', True) else "❌"
        print(f"   {i}. {status} {cmd.get('command', '')}")
        print(f"      Timestamp: {cmd.get('timestamp', 'N/A')}")

    # Demo 5: Statistics
    print("\n" + "=" * 60)
    print("Demo 5: Command Statistics")
    print("=" * 60)

    stats = await feature.get_statistics()
    print("\n📊 Overall Statistics:")
    print(f"   Total Commands: {stats.get('total_commands', 0)}")
    print(f"   Unique Commands: {stats.get('unique_commands', 0)}")
    print(f"   Success Rate: {stats.get('success_rate', 0):.1%}")
    print(f"   Patterns Detected: {stats.get('patterns_detected', 0)}")

    if stats.get('most_frequent'):
        print("\n   Most Frequent Commands:")
        for cmd, count in stats['most_frequent'][:3]:
            print(f"      • {cmd}: {count} times")

    # Demo 6: Pattern Analysis
    print("\n" + "=" * 60)
    print("Demo 6: Pattern Analysis")
    print("=" * 60)

    patterns = await feature.analyze_patterns()
    print("\n🔍 Pattern Analysis:")

    if patterns.get('command_patterns'):
        print(f"   Command Patterns: {len(patterns['command_patterns'])} categories")
        for base, info in list(patterns['command_patterns'].items())[:2]:
            print(f"      • {base}: {info['count']} patterns")

    if patterns.get('sequence_patterns'):
        print(f"   Sequence Patterns: {len(patterns['sequence_patterns'])} found")
        for seq in patterns['sequence_patterns'][:2]:
            print(f"      • {seq['from']} → {seq['to']} ({seq['frequency']} times)")

    # Demo 7: Learning Progress
    print("\n" + "=" * 60)
    print("Demo 7: Learning Progress")
    print("=" * 60)

    if feature.learning_engine:
        learning_stats = await feature.learning_engine.get_learning_stats()
        print("\n🎓 Learning Statistics:")
        print(f"   Commands Learned: {learning_stats.get('total_commands_learned', 0)}")
        print(f"   Total Executions: {learning_stats.get('total_executions', 0)}")
        print(f"   Mastered Commands: {len(learning_stats.get('mastered_commands', []))}")

        if learning_stats.get('skill_levels'):
            print("\n   Skill Levels:")
            for cmd, level in list(learning_stats['skill_levels'].items())[:3]:
                bar = "█" * int(level * 10) + "░" * (10 - int(level * 10))
                print(f"      • {cmd}: {bar} {level:.0%}")

    # Cleanup
    await feature._shutdown()

    print("\n" + "=" * 60)
    print("✅ Demo completed successfully!")
    print("=" * 60)

    print("\n📚 CLI Commands Available:")
    print("   • xencode terminal suggest [--partial <text>]")
    print("   • xencode terminal explain <command>")
    print("   • xencode terminal fix <error> [--command <cmd>]")
    print("   • xencode terminal history <pattern>")
    print("   • xencode terminal learn")
    print("   • xencode terminal statistics [--command <cmd>]")
    print("   • xencode terminal patterns")

    print("\n🖥️  TUI Access:")
    print("   • Launch TUI: xencode")
    print("   • Toggle Terminal Assistant: Ctrl+Y")
    print("   • Navigate tabs: 1-5 keys or click buttons")

    print("\n💡 Try it yourself:")
    print("   1. Run 'xencode terminal suggest' to get command suggestions")
    print("   2. Run 'xencode' to launch the TUI and press Ctrl+Y")
    print("   3. Explore the 5 tabs: Suggestions, Explanation, Fixes, Progress, History")


if __name__ == "__main__":
    asyncio.run(demo_terminal_assistant())
