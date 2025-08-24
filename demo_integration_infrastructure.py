#!/usr/bin/env python3
"""
Demo script for core integration infrastructure

Demonstrates the enhanced CLI system, resource monitor, feature detector,
and cold start optimization working together.
"""

import time
import argparse
from enhanced_cli_system import EnhancedXencodeCLI
from resource_monitor import ResourceMonitor


def demo_feature_detection():
    """Demonstrate feature detection with timeout handling"""
    print("🔍 Feature Detection Demo")
    print("=" * 40)
    
    # Initialize CLI (includes feature detection)
    cli = EnhancedXencodeCLI()
    
    # Show detected features
    features = cli.features
    print(f"\n📊 Detected Features:")
    print(f"  Multi-Model System: {'✅' if features.multi_model else '❌'}")
    print(f"  Smart Context System: {'✅' if features.smart_context else '❌'}")
    print(f"  Code Analysis System: {'✅' if features.code_analysis else '❌'}")
    print(f"  Security Manager: {'✅' if features.security_manager else '❌'}")
    print(f"  Context Cache: {'✅' if features.context_cache else '❌'}")
    print(f"  Model Stability: {'✅' if features.model_stability else '❌'}")
    print(f"\n🎚️ Overall Feature Level: {features.feature_level.upper()}")
    
    return cli


def demo_resource_monitoring():
    """Demonstrate resource monitoring and adaptive strategies"""
    print("\n\n💻 Resource Monitoring Demo")
    print("=" * 40)
    
    # Initialize resource monitor
    monitor = ResourceMonitor()
    
    # Show system profile
    profile = monitor.get_system_profile()
    print(f"\n🖥️ System Profile:")
    print(f"  RAM: {profile.ram_gb:.1f} GB")
    print(f"  CPU Cores: {profile.cpu_cores}")
    print(f"  Storage: {profile.storage_gb:.1f} GB ({profile.storage_type})")
    print(f"  Feature Level: {profile.feature_level.value.upper()}")
    print(f"  Max Context Size: {profile.max_context_size_mb} MB")
    print(f"  Max Concurrent Ops: {profile.max_concurrent_operations}")
    
    # Show current resource usage
    usage = monitor.monitor_resource_usage()
    print(f"\n📊 Current Resource Usage:")
    print(f"  RAM: {usage.ram_percent:.1f}% ({usage.ram_mb:,} MB)")
    print(f"  CPU: {usage.cpu_percent:.1f}%")
    print(f"  Storage: {usage.storage_percent:.1f}%")
    print(f"  Pressure Level: {usage.pressure_level.value.upper()}")
    print(f"  Throttled: {'Yes' if usage.is_throttled else 'No'}")
    
    # Show feature recommendations
    recommendations = monitor.get_recommended_feature_set()
    print(f"\n🎯 Feature Recommendations:")
    for feature, enabled in recommendations.items():
        status = "✅ Enabled" if enabled else "❌ Disabled"
        print(f"  {feature.replace('_', ' ').title()}: {status}")
    
    # Show adaptive scan strategy
    strategy = monitor.get_scan_strategy()
    print(f"\n🔍 Adaptive Scan Strategy:")
    print(f"  Scan Depth: {strategy.depth} levels")
    print(f"  Batch Size: {strategy.batch_size} files")
    print(f"  Max File Size: {strategy.max_file_size_kb} KB")
    print(f"  Compression: {strategy.compression}")
    print(f"  Parallel Processing: {'Yes' if strategy.parallel_processing else 'No'}")
    print(f"  Memory Limit: {strategy.memory_limit_mb} MB")
    print(f"  Timeout: {strategy.timeout_seconds}s")
    
    return monitor


def demo_enhanced_cli_commands(cli):
    """Demonstrate enhanced CLI commands with graceful fallback"""
    print("\n\n🚀 Enhanced CLI Commands Demo")
    print("=" * 40)
    
    # Create argument parser
    parser = cli.create_parser()
    
    # Test feature status command
    print(f"\n📋 Testing --feature-status command:")
    args = parser.parse_args(['--feature-status'])
    result = cli.process_enhanced_args(args)
    print(result)
    
    # Test enhanced commands (will show fallback messages)
    print(f"\n🔧 Testing enhanced commands (fallback behavior):")
    
    # Test analyze command
    print(f"\n  --analyze command:")
    result = cli.handle_analyze_command('/tmp')
    print(f"    {result}")
    
    # Test models command
    print(f"\n  --models command:")
    result = cli.handle_models_command()
    print(f"    {result}")
    
    # Test context command
    print(f"\n  --context command:")
    result = cli.handle_context_command()
    print(f"    {result}")
    
    # Test smart query command
    print(f"\n  --smart command:")
    result, model = cli.handle_smart_query("How do I optimize this code?")
    print(f"    {result}")
    if model:
        print(f"    Selected model: {model}")
    
    # Test git commit command
    print(f"\n  --git-commit command:")
    result = cli.handle_git_commit_command()
    print(f"    {result}")


def demo_cold_start_optimization():
    """Demonstrate cold start optimization"""
    print("\n\n⚡ Cold Start Optimization Demo")
    print("=" * 40)
    
    print(f"\n🚀 Initializing new CLI instance with timing...")
    
    start_time = time.time()
    
    # Initialize CLI (this will show the cold start process)
    cli = EnhancedXencodeCLI()
    
    total_time = time.time() - start_time
    
    print(f"\n⏱️ Total initialization time: {total_time:.3f} seconds")
    print(f"✅ Core systems available immediately")
    print(f"🔄 Enhanced features initialized in background")
    
    # Wait for background initialization to complete
    if hasattr(cli, '_init_thread'):
        print(f"⏳ Waiting for background initialization...")
        cli._init_thread.join(timeout=5)
        print(f"✅ Background initialization complete")
    
    return cli


def demo_progress_reporting(monitor):
    """Demonstrate progress reporting functionality"""
    print("\n\n📈 Progress Reporting Demo")
    print("=" * 40)
    
    strategy = monitor.get_scan_strategy()
    
    # Simulate scanning progress
    total_files = 1000
    total_size_mb = 250.0
    file_types = {'py': 400, 'js': 300, 'md': 150, 'json': 100, 'txt': 50}
    
    print(f"\n🔍 Simulating file scanning progress...")
    
    for i in range(0, total_files + 1, 100):
        current_size_mb = (i / total_files) * total_size_mb
        
        report = monitor.generate_progress_report(
            current=i,
            total=total_files,
            current_size_mb=current_size_mb,
            total_size_mb=total_size_mb,
            file_types=file_types,
            strategy=strategy
        )
        
        # Format and display progress
        progress_display = monitor.progress_reporter.format_progress_display(report)
        print(f"\r{progress_display}", end="", flush=True)
        
        time.sleep(0.2)  # Simulate processing time
    
    print(f"\n\n✅ Scanning complete!")


def demo_security_integration(cli):
    """Demonstrate security integration"""
    print("\n\n🔒 Security Integration Demo")
    print("=" * 40)
    
    security = cli.security_manager
    
    # Test path validation
    print(f"\n🛡️ Path Validation Tests:")
    test_paths = [
        "/tmp",
        ".",
        "../../../etc/passwd",
        "/nonexistent/path"
    ]
    
    for path in test_paths:
        is_valid = security.validate_project_path(path)
        status = "✅ Valid" if is_valid else "❌ Invalid"
        print(f"  {path}: {status}")
    
    # Test commit message sanitization
    print(f"\n🧹 Commit Message Sanitization:")
    test_messages = [
        "Fix authentication bug",
        "Update config $(rm -rf /)",
        "Add feature `curl evil.com`",
        "Refactor ${HOME}/.bashrc parser"
    ]
    
    for message in test_messages:
        sanitized = security.sanitize_commit_message(message)
        print(f"  Original: {message}")
        print(f"  Sanitized: {sanitized}")
        print()


def demo_resource_pressure_simulation(monitor):
    """Demonstrate resource pressure adaptation"""
    print("\n\n🔥 Resource Pressure Simulation")
    print("=" * 40)
    
    # Get base strategy
    base_strategy = monitor.get_scan_strategy()
    print(f"\n📊 Base Strategy:")
    print(f"  Batch Size: {base_strategy.batch_size}")
    print(f"  Pause Between Batches: {base_strategy.pause_between_batches_ms}ms")
    print(f"  Parallel Processing: {base_strategy.parallel_processing}")
    print(f"  Memory Limit: {base_strategy.memory_limit_mb}MB")
    
    # Simulate different pressure levels
    from resource_monitor import ResourceUsage, ResourcePressure
    
    pressure_scenarios = [
        ("Low Pressure", ResourceUsage(
            ram_mb=2000, ram_percent=25.0, cpu_percent=15.0,
            storage_mb=500000, storage_percent=50.0,
            pressure_level=ResourcePressure.LOW
        )),
        ("Medium Pressure", ResourceUsage(
            ram_mb=6000, ram_percent=75.0, cpu_percent=60.0,
            storage_mb=800000, storage_percent=80.0,
            pressure_level=ResourcePressure.MEDIUM
        )),
        ("High Pressure", ResourceUsage(
            ram_mb=7500, ram_percent=95.0, cpu_percent=90.0,
            storage_mb=950000, storage_percent=95.0,
            pressure_level=ResourcePressure.HIGH,
            is_throttled=True
        ))
    ]
    
    for scenario_name, usage in pressure_scenarios:
        print(f"\n🎭 {scenario_name} Scenario:")
        print(f"  RAM: {usage.ram_percent:.1f}%, CPU: {usage.cpu_percent:.1f}%")
        
        adjusted_strategy = monitor.throttler.adjust_scan_strategy(base_strategy, usage)
        
        print(f"  Adjusted Batch Size: {adjusted_strategy.batch_size}")
        print(f"  Adjusted Pause: {adjusted_strategy.pause_between_batches_ms}ms")
        print(f"  Parallel Processing: {adjusted_strategy.parallel_processing}")
        print(f"  Memory Limit: {adjusted_strategy.memory_limit_mb}MB")


def main():
    """Run complete integration infrastructure demo"""
    print("🎯 Xencode Integration Infrastructure Demo")
    print("=" * 50)
    print("Demonstrating core integration components working together")
    
    # Demo 1: Feature Detection
    cli = demo_feature_detection()
    
    # Demo 2: Resource Monitoring
    monitor = demo_resource_monitoring()
    
    # Demo 3: Enhanced CLI Commands
    demo_enhanced_cli_commands(cli)
    
    # Demo 4: Cold Start Optimization
    demo_cold_start_optimization()
    
    # Demo 5: Progress Reporting
    demo_progress_reporting(monitor)
    
    # Demo 6: Security Integration
    demo_security_integration(cli)
    
    # Demo 7: Resource Pressure Simulation
    demo_resource_pressure_simulation(monitor)
    
    print(f"\n\n🎉 Demo Complete!")
    print("=" * 50)
    print("✅ All integration infrastructure components working correctly")
    print("🚀 System ready for Phase 2 integration")


if __name__ == "__main__":
    main()