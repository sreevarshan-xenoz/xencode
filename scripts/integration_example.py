#!/usr/bin/env python3
"""
Integration Example: Model Stability Manager with Xencode
Shows how the Model Stability Manager integrates with the main xencode system
"""

import time
from typing import Tuple

from model_stability_manager import ModelStabilityManager


class EnhancedModelManager:
    """
    Enhanced model manager that integrates stability management
    This shows how the Model Stability Manager would be used in xencode_core.py
    """

    def __init__(self):
        self.stability_manager = ModelStabilityManager()
        self.current_model = "qwen3:4b"
        self.smart_mode_enabled = False

        # Start background monitoring
        self.stability_manager.start_background_monitoring(interval_minutes=5)
        print("🔄 Background model monitoring started")

    def __del__(self):
        """Cleanup background monitoring"""
        if hasattr(self, 'stability_manager'):
            self.stability_manager.stop_background_monitoring()

    def switch_model_with_stability_check(self, new_model: str) -> Tuple[bool, str]:
        """
        Switch model with stability verification
        Implements Requirements 4.6, 4.7, 4.8
        """
        print(f"🔄 Attempting to switch to {new_model}...")

        # Check if model is available and stable
        if not self.stability_manager.is_model_available(new_model):
            print(f"❌ Model {new_model} is not available or degraded")

            # Get fallback chain
            fallback_chain = self.stability_manager.get_fallback_chain("general")

            for fallback_model in fallback_chain:
                if self.stability_manager.is_model_available(fallback_model):
                    print(f"🔄 Using fallback model: {fallback_model}")
                    self.current_model = fallback_model
                    return True, f"Switched to fallback model {fallback_model}"

            # Use emergency model as last resort
            emergency_model = self.stability_manager.get_emergency_model()
            print(f"🚨 Using emergency model: {emergency_model}")
            self.current_model = emergency_model
            return True, f"Using emergency model {emergency_model}"

        # Test model stability before switching
        print(f"🧪 Testing stability of {new_model}...")
        result = self.stability_manager.test_model_stability(new_model)

        if result.is_stable:
            self.current_model = new_model
            print(f"✅ Successfully switched to {new_model}")
            return True, f"Switched to {new_model}"
        else:
            print(f"❌ Model {new_model} failed stability test: {result.error_message}")

            if result.oom_detected:
                # Mark as degraded for longer period due to OOM
                self.stability_manager.mark_model_degraded(
                    new_model, duration_minutes=10
                )
                print(f"⚠️ Marked {new_model} as degraded due to OOM")

            return False, f"Model {new_model} is unstable: {result.error_message}"

    def smart_model_selection(self, query: str, query_type: str = "general") -> str:
        """
        Smart model selection with stability consideration
        Implements Requirements 4.9, 4.10, 4.11
        """
        if not self.smart_mode_enabled:
            return self.current_model

        print(f"🧠 Smart model selection for {query_type} query...")

        # Check 3-second rule for consecutive queries
        if self.stability_manager.should_use_previous_model(self.current_model):
            print(f"⚡ Using previous model {self.current_model} (3-second rule)")
            self.stability_manager.update_query_timing(self.current_model)
            return self.current_model

        # Get fallback chain for query type
        fallback_chain = self.stability_manager.get_fallback_chain(query_type)

        # Find first available and stable model
        for model in fallback_chain:
            if self.stability_manager.is_model_available(model):
                print(f"🎯 Selected {model} for {query_type} query")

                # Update timing for 3-second rule
                self.stability_manager.update_query_timing(model)

                return model

        # Fallback to emergency model
        emergency_model = self.stability_manager.get_emergency_model()
        print(f"🚨 Using emergency model {emergency_model}")
        return emergency_model

    def execute_query_with_stability(
        self, query: str, query_type: str = "general"
    ) -> str:
        """
        Execute query with full stability management
        This simulates how xencode would use the stability manager
        """
        print(f"\n📝 Executing query: '{query[:50]}...'")

        # Select best model
        selected_model = self.smart_model_selection(query, query_type)

        # Switch to selected model if different
        if selected_model != self.current_model:
            success, message = self.switch_model_with_stability_check(selected_model)
            if not success:
                print(f"⚠️ Model switch failed: {message}")
                # Continue with current model

        # Simulate query execution
        print(f"🤖 Executing query with {self.current_model}...")

        # In real implementation, this would call the actual model
        # For demo, we'll simulate different outcomes

        # Test current model stability
        result = self.stability_manager.test_model_stability(self.current_model)

        if result.is_stable:
            response = f"✅ Query executed successfully with {self.current_model}"
            print(
                f"📊 Performance: {result.response_time_ms}ms, {result.memory_usage_mb}MB"
            )
        else:
            print(
                f"❌ Model {self.current_model} failed during query: {result.error_message}"
            )

            if result.oom_detected:
                print(f"🚨 OOM detected! Marking {self.current_model} as degraded")
                self.stability_manager.mark_model_degraded(
                    self.current_model, duration_minutes=5
                )

            # Try fallback
            fallback_chain = self.stability_manager.get_fallback_chain(query_type)
            for fallback_model in fallback_chain:
                if (
                    fallback_model != self.current_model
                    and self.stability_manager.is_model_available(fallback_model)
                ):

                    print(f"🔄 Trying fallback model: {fallback_model}")
                    self.current_model = fallback_model
                    response = f"✅ Query executed with fallback model {fallback_model}"
                    break
            else:
                response = "❌ All models failed, using emergency response"

        return response

    def get_system_status(self) -> dict:
        """Get comprehensive system status"""
        health_summary = self.stability_manager.get_model_health_summary()

        status = {
            'current_model': self.current_model,
            'smart_mode': self.smart_mode_enabled,
            'total_models': len(health_summary),
            'healthy_models': sum(
                1 for h in health_summary.values() if h['status'] == 'healthy'
            ),
            'degraded_models': sum(
                1 for h in health_summary.values() if h['status'] == 'degraded'
            ),
            'emergency_model': self.stability_manager.get_emergency_model(),
            'health_details': health_summary,
        }

        return status


def demo_integration_scenarios():
    """Demonstrate real-world integration scenarios"""
    print("🚀 Model Stability Manager Integration Demo")
    print("=" * 60)

    # Initialize enhanced model manager
    manager = EnhancedModelManager()
    manager.smart_mode_enabled = True

    # Scenario 1: Normal operation
    print("\n📋 Scenario 1: Normal Query Execution")
    print("-" * 40)

    response = manager.execute_query_with_stability(
        "Write a Python function to calculate fibonacci numbers", query_type="code"
    )
    print(f"Response: {response}")

    # Scenario 2: Consecutive queries (3-second rule)
    print("\n📋 Scenario 2: Consecutive Queries (3-second rule)")
    print("-" * 40)

    print("First query:")
    manager.execute_query_with_stability("What is Python?", "explanation")

    print("\nSecond query (within 3 seconds):")
    time.sleep(0.5)
    manager.execute_query_with_stability("How do I install Python?", "explanation")

    print("\nThird query (after 3 seconds):")
    time.sleep(3.1)
    manager.execute_query_with_stability("Python vs Java comparison", "analysis")

    # Scenario 3: Model degradation and fallback
    print("\n📋 Scenario 3: Model Degradation and Fallback")
    print("-" * 40)

    # Simulate model degradation
    current_model = manager.current_model
    manager.stability_manager.mark_model_degraded(
        current_model, duration_minutes=1 / 60
    )
    print(f"🧪 Simulated degradation of {current_model}")

    response = manager.execute_query_with_stability(
        "Create a creative story about AI", query_type="creative"
    )
    print(f"Response: {response}")

    # Scenario 4: System status monitoring
    print("\n📋 Scenario 4: System Status Monitoring")
    print("-" * 40)

    status = manager.get_system_status()
    print("📊 System Status:")
    print(f"  • Current model: {status['current_model']}")
    print(f"  • Smart mode: {status['smart_mode']}")
    print(f"  • Total models tracked: {status['total_models']}")
    print(f"  • Healthy models: {status['healthy_models']}")
    print(f"  • Degraded models: {status['degraded_models']}")
    print(f"  • Emergency model: {status['emergency_model']}")

    # Show detailed health for degraded models
    degraded_models = [
        name
        for name, health in status['health_details'].items()
        if health['status'] == 'degraded'
    ]

    if degraded_models:
        print("\n⚠️ Degraded Models:")
        for model in degraded_models:
            health = status['health_details'][model]
            print(
                f"  • {model}: score {health['stability_score']:.2f}, "
                f"failures {health['failure_count']}"
            )

    # Scenario 5: Manual model switching with stability check
    print("\n📋 Scenario 5: Manual Model Switching")
    print("-" * 40)

    test_models = ["llama2:7b", "mistral:7b", "nonexistent:model"]

    for model in test_models:
        success, message = manager.switch_model_with_stability_check(model)
        print(f"  • {model}: {'✅' if success else '❌'} {message}")

    print(f"\nFinal model: {manager.current_model}")

    # Cleanup
    print("\n🧹 Cleaning up...")
    del manager  # This will stop background monitoring
    print("✅ Demo complete!")


def demo_chat_mode_integration():
    """Demonstrate chat mode integration"""
    print("\n" + "=" * 60)
    print("💬 Chat Mode Integration Demo")
    print("=" * 60)

    manager = EnhancedModelManager()
    manager.smart_mode_enabled = True

    # Simulate chat conversation with different query types
    chat_queries = [
        ("Hello, how are you?", "general"),
        ("Write a function to sort an array", "code"),
        ("Tell me a story about dragons", "creative"),
        ("Compare Python and JavaScript", "analysis"),
        ("Explain quantum computing", "explanation"),
        ("What's 2+2?", "general"),  # Quick follow-up (3-second rule)
    ]

    print("🗣️ Simulating chat conversation:")

    for i, (query, query_type) in enumerate(chat_queries, 1):
        print(f"\n💬 Message {i}: {query}")
        print(f"   Type: {query_type}")

        # Small delay between messages (except for quick follow-up)
        if i > 1 and i != 6:
            time.sleep(1)
        elif i == 6:
            time.sleep(0.5)  # Quick follow-up

        selected_model = manager.smart_model_selection(query, query_type)
        print(f"   🤖 Selected model: {selected_model}")

        # Show if 3-second rule applied
        if i == 6:  # Quick follow-up
            print("   ⚡ 3-second rule applied: using previous model")

    # Cleanup
    del manager
    print("\n✅ Chat mode demo complete!")


if __name__ == "__main__":
    try:
        demo_integration_scenarios()
        demo_chat_mode_integration()

        print("\n🎉 All integration demos completed successfully!")
        print("\nThe Model Stability Manager is ready for Phase 2 integration!")

    except KeyboardInterrupt:
        print("\n\n⏹️ Demo interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Demo error: {e}")
        import traceback

        traceback.print_exc()
