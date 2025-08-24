#!/usr/bin/env python3
"""
Demo script for Model Stability Manager
Shows real-world usage scenarios and capabilities
"""

import time
import json
from model_stability_manager import ModelStabilityManager, StabilityStatus

def print_section(title):
    """Print a formatted section header"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def print_subsection(title):
    """Print a formatted subsection header"""
    print(f"\n{'-'*40}")
    print(f"  {title}")
    print(f"{'-'*40}")

def demo_basic_functionality():
    """Demonstrate basic model stability functionality"""
    print_section("🔍 Model Stability Manager Demo")
    
    # Initialize manager
    manager = ModelStabilityManager()
    print("✅ Model Stability Manager initialized")
    
    # Show configuration
    print(f"📋 Configuration:")
    print(f"  • Stability timeout: {manager.stability_timeout_ms}ms")
    print(f"  • Degradation duration: {manager.degradation_duration_minutes} minutes")
    print(f"  • Max consecutive failures: {manager.max_consecutive_failures}")
    print(f"  • Consecutive query threshold: {manager.consecutive_query_threshold_seconds}s")
    print(f"  • Emergency model: {manager.emergency_model}")
    
    return manager

def demo_fallback_chains(manager):
    """Demonstrate fallback chain functionality"""
    print_section("🔄 Fallback Chain System")
    
    query_types = ["code", "creative", "analysis", "explanation", "general"]
    
    for query_type in query_types:
        chain = manager.get_fallback_chain(query_type)
        print(f"📝 {query_type.capitalize()} queries:")
        print(f"   {' → '.join(chain)}")
    
    print(f"\n🚨 Emergency model: {manager.get_emergency_model()}")

def demo_model_testing(manager):
    """Demonstrate model stability testing"""
    print_section("🧪 Model Stability Testing")
    
    # Test models (these will likely fail since we're not running ollama)
    test_models = ["qwen3:4b", "llama2:7b", "nonexistent:model"]
    
    for model in test_models:
        print_subsection(f"Testing {model}")
        
        try:
            result = manager.test_model_stability(model)
            
            print(f"🎯 Results for {model}:")
            print(f"  • Stable: {'✅' if result.is_stable else '❌'} {result.is_stable}")
            print(f"  • Status: {result.status.value}")
            print(f"  • Response time: {result.response_time_ms}ms")
            print(f"  • Memory usage: {result.memory_usage_mb}MB")
            print(f"  • OOM detected: {'⚠️' if result.oom_detected else '✅'} {result.oom_detected}")
            
            if result.error_message:
                print(f"  • Error: {result.error_message}")
                
        except Exception as e:
            print(f"❌ Error testing {model}: {e}")

def demo_oom_detection(manager):
    """Demonstrate OOM detection capabilities"""
    print_section("🔍 OOM Detection System")
    
    print("🔧 OOM Detection Methods:")
    print("  1. journalctl (systemd systems)")
    print("  2. dmesg (older/non-systemd systems)")
    print("  3. ollama logs (universal)")
    
    # Test OOM detection (will likely return False since no actual OOM)
    test_model = "qwen3:4b"
    print(f"\n🧪 Testing OOM detection for {test_model}:")
    
    try:
        oom_detected = manager.detect_oom_crash(test_model)
        print(f"  • Overall OOM detected: {'⚠️' if oom_detected else '✅'} {oom_detected}")
        
        # Test individual methods
        journalctl_oom = manager.check_journalctl_oom(test_model)
        print(f"  • journalctl OOM: {'⚠️' if journalctl_oom else '✅'} {journalctl_oom}")
        
        dmesg_oom = manager.check_dmesg_oom(test_model)
        print(f"  • dmesg OOM: {'⚠️' if dmesg_oom else '✅'} {dmesg_oom}")
        
        ollama_logs_oom = manager.check_ollama_logs(test_model)
        print(f"  • ollama logs OOM: {'⚠️' if ollama_logs_oom else '✅'} {ollama_logs_oom}")
        
    except Exception as e:
        print(f"❌ Error in OOM detection: {e}")

def demo_degradation_tracking(manager):
    """Demonstrate model degradation tracking"""
    print_section("📉 Model Degradation Tracking")
    
    test_model = "demo_model"
    
    print(f"🧪 Simulating degradation for {test_model}:")
    
    # Mark model as degraded
    manager.mark_model_degraded(test_model, duration_minutes=1/60)  # 1 second for demo
    print(f"  • Marked {test_model} as degraded for 1 second")
    
    # Check availability
    available = manager.is_model_available(test_model)
    print(f"  • Model available: {'✅' if available else '❌'} {available}")
    
    # Wait for degradation to expire
    print("  • Waiting for degradation to expire...")
    time.sleep(1.1)
    
    # Simulate successful operation to restore model
    manager._update_model_success(test_model, 100, 1000)
    available = manager.is_model_available(test_model)
    print(f"  • Model available after recovery: {'✅' if available else '❌'} {available}")

def demo_consecutive_queries(manager):
    """Demonstrate 3-second rule for consecutive queries"""
    print_section("⏱️ Consecutive Query Rule (3-second rule)")
    
    test_model = "demo_model"
    
    print(f"🧪 Testing consecutive queries for {test_model}:")
    
    # First query
    print("  1. First query:")
    should_use_previous = manager.should_use_previous_model(test_model)
    print(f"     • Should use previous model: {'✅' if should_use_previous else '❌'} {should_use_previous}")
    manager.update_query_timing(test_model)
    
    # Second query within 3 seconds
    time.sleep(0.5)
    print("  2. Second query (0.5s later):")
    should_use_previous = manager.should_use_previous_model(test_model)
    print(f"     • Should use previous model: {'✅' if should_use_previous else '❌'} {should_use_previous}")
    manager.update_query_timing(test_model)
    
    # Third query after 3 seconds
    print("  3. Waiting 3.1 seconds...")
    time.sleep(3.1)
    print("  4. Third query (3.1s later):")
    should_use_previous = manager.should_use_previous_model(test_model)
    print(f"     • Should use previous model: {'✅' if should_use_previous else '❌'} {should_use_previous}")

def demo_health_monitoring(manager):
    """Demonstrate health monitoring and reporting"""
    print_section("📊 Health Monitoring & Reporting")
    
    # Add some demo model states
    print("🧪 Creating demo model states...")
    
    # Simulate different model states
    manager._update_model_success("healthy_model", 120, 2000)
    
    manager._update_model_failure("failing_model", "Connection timeout")
    manager._update_model_failure("failing_model", "OOM detected")
    manager._update_model_failure("failing_model", "HTTP 500")
    
    manager.mark_model_degraded("degraded_model", duration_minutes=5)
    
    # Show health summary
    print("\n📋 Health Summary:")
    summary = manager.get_model_health_summary()
    
    for model_name, health in summary.items():
        status_icon = "✅" if health['status'] == 'healthy' else "❌"
        print(f"  {status_icon} {model_name}:")
        print(f"     • Status: {health['status']}")
        print(f"     • Stability score: {health['stability_score']:.2f}")
        print(f"     • Avg response time: {health['avg_response_time_ms']}ms")
        print(f"     • Memory usage: {health['memory_usage_mb']}MB")
        print(f"     • Failure count: {health['failure_count']}")
        
        if health['is_degraded']:
            print(f"     • Degraded until: {health['degraded_until']}")

def demo_background_monitoring(manager):
    """Demonstrate background monitoring"""
    print_section("🔄 Background Monitoring")
    
    print("🚀 Starting background monitoring (5-second interval for demo)...")
    
    # Start background monitoring with short interval for demo
    manager.start_background_monitoring(interval_minutes=5/60)  # 5 seconds
    
    print("  • Background monitoring started")
    print("  • Monitoring will test all known models every 5 seconds")
    print("  • In production, use longer intervals (5+ minutes)")
    
    # Let it run for a bit
    print("\n⏳ Letting background monitoring run for 10 seconds...")
    time.sleep(10)
    
    # Stop monitoring
    manager.stop_background_monitoring()
    print("✅ Background monitoring stopped")

def demo_persistence(manager):
    """Demonstrate state persistence"""
    print_section("💾 State Persistence")
    
    print("🧪 Testing state persistence...")
    
    # Show current state file location
    print(f"📁 State file location: {manager.state_file}")
    
    # Save current states
    manager.save_model_states()
    print("✅ Model states saved to disk")
    
    # Show what's saved
    if manager.state_file.exists():
        with open(manager.state_file, 'r') as f:
            data = json.load(f)
        
        print(f"📊 Saved {len(data)} model states:")
        for model_name in data.keys():
            print(f"  • {model_name}")
    else:
        print("❌ No state file found")

def main():
    """Run the complete demo"""
    print("🚀 Model Stability Manager - Complete Demo")
    print("This demo shows all capabilities of the Model Stability Manager")
    
    try:
        # Initialize
        manager = demo_basic_functionality()
        
        # Demo each feature
        demo_fallback_chains(manager)
        demo_model_testing(manager)
        demo_oom_detection(manager)
        demo_degradation_tracking(manager)
        demo_consecutive_queries(manager)
        demo_health_monitoring(manager)
        demo_persistence(manager)
        demo_background_monitoring(manager)
        
        print_section("✅ Demo Complete")
        print("The Model Stability Manager provides:")
        print("  • Cross-platform OOM detection")
        print("  • Model health monitoring and degradation tracking")
        print("  • Intelligent fallback chains")
        print("  • 3-second rule for conversational flow")
        print("  • Background monitoring")
        print("  • Persistent state management")
        print("\nReady for integration into Xencode Phase 2! 🎉")
        
    except KeyboardInterrupt:
        print("\n\n⏹️ Demo interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Demo error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()