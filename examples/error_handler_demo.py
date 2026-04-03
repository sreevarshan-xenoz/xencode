"""
Error Handler Demo

Demonstrates the intelligent error handling capabilities of the Terminal Assistant.
"""

import asyncio
from xencode.features.terminal_assistant import TerminalAssistantFeature
from xencode.features.base import FeatureConfig


async def demo_error_handling():
    """Demonstrate error handling capabilities"""
    
    # Initialize Terminal Assistant
    config = FeatureConfig(
        name="terminal_assistant",
        enabled=True,
        config={
            'error_fix_enabled': True,
            'history_size': 1000,
            'context_aware': True
        }
    )
    
    assistant = TerminalAssistantFeature(config)
    await assistant.initialize()
    
    print("=" * 80)
    print("Terminal Assistant - Intelligent Error Handling Demo")
    print("=" * 80)
    print()
    
    # Demo 1: Command not found (typo)
    print("Demo 1: Command Not Found (Typo Correction)")
    print("-" * 80)
    command = "pyhton script.py"
    error = "bash: pyhton: command not found"
    print(f"Command: {command}")
    print(f"Error: {error}")
    print()
    
    fixes = await assistant.fix_error(command, error)
    print(f"Found {len(fixes)} fix suggestions:")
    for i, fix in enumerate(fixes, 1):
        print(f"\n{i}. {fix['explanation']}")
        if fix['fix']:
            print(f"   Fix: {fix['fix']}")
        print(f"   Confidence: {fix['confidence']:.2%}")
        if fix['alternative_commands']:
            print(f"   Alternatives: {', '.join(fix['alternative_commands'][:2])}")
    
    print("\n" + "=" * 80 + "\n")
    
    # Demo 2: Permission denied
    print("Demo 2: Permission Denied")
    print("-" * 80)
    command = "./script.sh"
    error = "bash: ./script.sh: Permission denied"
    print(f"Command: {command}")
    print(f"Error: {error}")
    print()
    
    fixes = await assistant.fix_error(command, error)
    print(f"Found {len(fixes)} fix suggestions:")
    for i, fix in enumerate(fixes, 1):
        print(f"\n{i}. {fix['explanation']}")
        if fix['fix']:
            print(f"   Fix: {fix['fix']}")
        print(f"   Confidence: {fix['confidence']:.2%}")
        if fix['requires_sudo']:
            print(f"   ⚠️  Requires sudo privileges")
    
    print("\n" + "=" * 80 + "\n")
    
    # Demo 3: Module not found (with context)
    print("Demo 3: Module Not Found (Context-Aware)")
    print("-" * 80)
    command = "python app.py"
    error = "ModuleNotFoundError: No module named 'flask'"
    context = {
        'project_type': 'python',
        'directory': '/home/user/myproject'
    }
    print(f"Command: {command}")
    print(f"Error: {error}")
    print(f"Context: Python project")
    print()
    
    fixes = await assistant.fix_error(command, error, context)
    print(f"Found {len(fixes)} fix suggestions:")
    for i, fix in enumerate(fixes, 1):
        print(f"\n{i}. {fix['explanation']}")
        if fix['fix']:
            print(f"   Fix: {fix['fix']}")
        print(f"   Confidence: {fix['confidence']:.2%}")
        if fix['requires_install']:
            print(f"   📦 Requires installation")
        if fix['alternative_commands']:
            print(f"   Alternatives:")
            for alt in fix['alternative_commands'][:3]:
                print(f"      - {alt}")
    
    print("\n" + "=" * 80 + "\n")
    
    # Demo 4: Port already in use
    print("Demo 4: Port Already In Use")
    print("-" * 80)
    command = "npm start"
    error = "Error: listen EADDRINUSE: address already in use :::3000"
    print(f"Command: {command}")
    print(f"Error: {error}")
    print()
    
    fixes = await assistant.fix_error(command, error)
    print(f"Found {len(fixes)} fix suggestions:")
    for i, fix in enumerate(fixes, 1):
        print(f"\n{i}. {fix['explanation']}")
        if fix['fix']:
            print(f"   Fix: {fix['fix']}")
        print(f"   Confidence: {fix['confidence']:.2%}")
        if fix['category'] == 'port_management':
            print(f"   🔌 Port management fix")
    
    print("\n" + "=" * 80 + "\n")
    
    # Demo 5: Learning from successful fixes
    print("Demo 5: Learning from Successful Fixes")
    print("-" * 80)
    command = "gti status"
    error = "bash: gti: command not found"
    print(f"Command: {command}")
    print(f"Error: {error}")
    print()
    
    # Record that "git status" worked
    print("Recording successful fix: git status")
    await assistant.record_successful_fix(command, error, "git status")
    
    # Get suggestions again
    fixes = await assistant.fix_error(command, error)
    print(f"\nFound {len(fixes)} fix suggestions (with learning):")
    for i, fix in enumerate(fixes, 1):
        print(f"\n{i}. {fix['explanation']}")
        if fix['fix']:
            print(f"   Fix: {fix['fix']}")
        print(f"   Confidence: {fix['confidence']:.2%}")
        if fix['category'] == 'learned_fix':
            print(f"   🧠 Learned from previous success")
    
    print("\n" + "=" * 80 + "\n")
    
    # Demo 6: Git repository error
    print("Demo 6: Git Repository Error")
    print("-" * 80)
    command = "git status"
    error = "fatal: not a git repository (or any of the parent directories): .git"
    print(f"Command: {command}")
    print(f"Error: {error}")
    print()
    
    fixes = await assistant.fix_error(command, error)
    print(f"Found {len(fixes)} fix suggestions:")
    for i, fix in enumerate(fixes, 1):
        print(f"\n{i}. {fix['explanation']}")
        if fix['fix']:
            print(f"   Fix: {fix['fix']}")
        print(f"   Confidence: {fix['confidence']:.2%}")
        if fix['documentation_url']:
            print(f"   📚 Documentation: {fix['documentation_url']}")
    
    print("\n" + "=" * 80 + "\n")
    
    # Show statistics
    print("Error Handler Statistics")
    print("-" * 80)
    stats = assistant.error_handler.get_error_statistics()
    print(f"Total errors processed: {stats['total_errors_seen']}")
    print(f"Unique error patterns: {stats['unique_errors']}")
    print(f"Registered error patterns: {stats['patterns_registered']}")
    print(f"Learned fixes: {stats['learned_fixes']}")
    
    print("\n" + "=" * 80)
    
    await assistant.shutdown()


if __name__ == '__main__':
    asyncio.run(demo_error_handling())
