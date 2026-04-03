"""
Context Analyzer Demo

Demonstrates the ContextAnalyzer functionality for the Terminal Assistant feature.
Shows how the analyzer detects:
- Directory and project type
- OS and shell environment
- Git repository information
- File system context
- Running processes
- Network context
"""

import asyncio
import json
from pathlib import Path
from xencode.features.terminal_assistant import ContextAnalyzer


async def demo_context_analyzer():
    """Demonstrate context analyzer capabilities"""
    
    print("=" * 80)
    print("Context Analyzer Demo")
    print("=" * 80)
    print()
    
    # Create context analyzer
    analyzer = ContextAnalyzer(enabled=True)
    
    # Analyze current context
    print("Analyzing current context...")
    context = await analyzer.analyze()
    
    # Display results
    print("\n" + "=" * 80)
    print("CONTEXT ANALYSIS RESULTS")
    print("=" * 80)
    
    # Directory information
    print(f"\n📁 Current Directory: {context['directory']}")
    print(f"🖥️  Operating System: {context['os']}")
    
    # Project type detection
    if context['project_type']:
        print(f"📦 Project Type: {context['project_type']}")
    else:
        print("📦 Project Type: Not detected")
    
    # Git information
    print("\n" + "-" * 80)
    print("Git Repository Information:")
    print("-" * 80)
    git_info = context['git_info']
    if git_info['is_repo']:
        print(f"✓ Git repository detected")
        print(f"  Branch: {git_info['branch']}")
        print(f"  Remotes: {', '.join(git_info['remotes']) if git_info['remotes'] else 'None'}")
        print(f"  Has changes: {git_info['has_changes']}")
    else:
        print("✗ Not a git repository")
    
    # Environment variables
    print("\n" + "-" * 80)
    print("Environment Variables:")
    print("-" * 80)
    env = context['environment']
    
    if env['development']:
        print("\n  Development:")
        for key, value in list(env['development'].items())[:3]:
            print(f"    {key}: {value[:50]}..." if len(value) > 50 else f"    {key}: {value}")
    
    if env['cloud']:
        print("\n  Cloud:")
        for key, value in env['cloud'].items():
            print(f"    {key}: {value}")
    
    if env['shell']:
        print("\n  Shell:")
        for key, value in list(env['shell'].items())[:3]:
            print(f"    {key}: {value[:50]}..." if len(value) > 50 else f"    {key}: {value}")
    
    # File system information
    print("\n" + "-" * 80)
    print("File System:")
    print("-" * 80)
    fs = context['filesystem']
    
    if fs['disk_usage']:
        usage = fs['disk_usage']
        print(f"  Disk Usage: {usage['percent_used']:.1f}% used")
        print(f"  Free Space: {usage['free'] / (1024**3):.1f} GB")
    
    if fs['permissions']:
        perms = fs['permissions']
        print(f"  Permissions: R:{perms['readable']} W:{perms['writable']} X:{perms['executable']}")
    
    if fs['file_counts']:
        print(f"\n  File Types:")
        for ext, count in list(fs['file_counts'].items())[:5]:
            print(f"    {ext}: {count} files")
    
    if fs['recent_files']:
        print(f"\n  Recent Files (last 24h):")
        for file_info in fs['recent_files'][:5]:
            print(f"    {file_info['name']} ({file_info['age_hours']:.1f}h ago)")
    
    # Process information
    print("\n" + "-" * 80)
    print("Running Processes:")
    print("-" * 80)
    processes = context['processes']
    
    if processes['development_servers']:
        print("\n  Development Servers:")
        for proc in processes['development_servers'][:3]:
            print(f"    PID {proc['pid']}: {proc['name']}")
            if proc['cmdline']:
                print(f"      {proc['cmdline']}")
    
    if processes['databases']:
        print("\n  Databases:")
        for proc in processes['databases'][:3]:
            print(f"    PID {proc['pid']}: {proc['name']}")
    
    if processes['containers']:
        print("\n  Containers:")
        for proc in processes['containers'][:3]:
            print(f"    PID {proc['pid']}: {proc['name']}")
    
    if not any([processes['development_servers'], processes['databases'], processes['containers']]):
        print("  No relevant processes detected")
    
    # Network information
    print("\n" + "-" * 80)
    print("Network:")
    print("-" * 80)
    network = context['network']
    
    if network['localhost_ports']:
        print("\n  Listening Ports:")
        for port_info in network['localhost_ports']:
            print(f"    Port {port_info['port']} on {port_info['address']}")
    else:
        print("  No common development ports listening")
    
    if network['vpn_active']:
        print("\n  VPN: Active")
    
    if network['network_interfaces']:
        print(f"\n  Network Interfaces: {len(network['network_interfaces'])} detected")
    
    # Files in directory
    print("\n" + "-" * 80)
    print("Files in Directory:")
    print("-" * 80)
    files = context['files']
    if files:
        print(f"  {len(files)} files detected")
        print(f"  Sample: {', '.join(files[:5])}")
        if len(files) > 5:
            print(f"  ... and {len(files) - 5} more")
    else:
        print("  No files detected")
    
    print("\n" + "=" * 80)
    print("Demo Complete!")
    print("=" * 80)
    
    # Save full context to JSON file
    output_file = Path("context_analysis_output.json")
    with open(output_file, 'w') as f:
        json.dump(context, f, indent=2, default=str)
    print(f"\nFull context saved to: {output_file}")


async def demo_project_type_detection():
    """Demonstrate project type detection"""
    
    print("\n" + "=" * 80)
    print("Project Type Detection Demo")
    print("=" * 80)
    
    analyzer = ContextAnalyzer(enabled=True)
    
    # Test different project types
    test_cases = [
        ("Python Poetry", "pyproject.toml"),
        ("Python setuptools", "setup.py"),
        ("Node.js", "package.json"),
        ("Rust", "Cargo.toml"),
        ("Go", "go.mod"),
        ("Java Maven", "pom.xml"),
        ("Docker", "Dockerfile"),
    ]
    
    print("\nSupported project types:")
    for name, marker_file in test_cases:
        print(f"  • {name} (detected by {marker_file})")
    
    print(f"\nCurrent directory project type: {analyzer._detect_project_type(Path.cwd())}")


if __name__ == "__main__":
    # Run the demos
    asyncio.run(demo_context_analyzer())
    asyncio.run(demo_project_type_detection())
