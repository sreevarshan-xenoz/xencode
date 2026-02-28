#!/usr/bin/env python3
"""
Voice Interface CLI and TUI Demo

Demonstrates the voice interface CLI commands and TUI components for hands-free coding.

Requirements:
- Voice interface feature enabled
- Microphone access
- Audio output

CLI Commands:
    xencode voice start              # Start voice listening mode
    xencode voice stop               # Stop voice listening
    xencode voice commands           # Show recent voice commands
    xencode voice status             # Show voice interface status
    xencode voice test               # Test voice output
    xencode voice settings --show    # Show current settings

TUI Components:
- VoiceInterfacePanel: Main panel with all voice interface features
- VoiceStatusIndicator: Visual status indicator (idle, listening, processing, speaking)
- AudioLevelMeter: Real-time audio level visualization
- VoiceCommandHistory: History of recognized voice commands
- VoiceSettingsPanel: Configuration panel for voice settings
"""

import asyncio
from pathlib import Path
from typing import Dict, Any

# Example 1: Using Voice Interface Feature Programmatically
async def demo_voice_interface_feature():
    """Demonstrate voice interface feature usage"""
    from xencode.features import FeatureManager
    from xencode.features.voice_interface import VoiceInterfaceFeature, FeatureConfig
    
    print("=" * 60)
    print("Voice Interface Feature Demo")
    print("=" * 60)
    
    # Create feature configuration
    config = FeatureConfig(
        name="voice_interface",
        enabled=True,
        config={
            'speech_to_text': {
                'provider': 'whisper',
                'model': 'base',
                'language': 'en'
            },
            'text_to_speech': {
                'provider': 'pyttsx3',
                'voice': 'en-US',
                'rate': 150,
                'volume': 0.9
            },
            'noise_reduction': True,
            'silence_threshold': 0.5,
            'silence_duration': 1.0
        }
    )
    
    # Create feature instance
    feature = VoiceInterfaceFeature(config)
    
    # Initialize feature
    print("\n1. Initializing voice interface...")
    await feature.initialize()
    print("   ✅ Voice interface initialized")
    
    # Get status
    print("\n2. Getting voice interface status...")
    status = feature.get_status()
    print(f"   Status: {status['status']}")
    print(f"   Listening: {status['listening']}")
    print(f"   Commands: {status['command_count']}")
    
    # Test text-to-speech
    print("\n3. Testing text-to-speech...")
    await feature.speak("Hello, this is the voice interface demo", wait=True)
    print("   ✅ Speech test completed")
    
    # Simulate voice command processing
    print("\n4. Processing voice command...")
    command = await feature.process("open file explorer")
    print(f"   Command: {command.text}")
    print(f"   Type: {command.command_type.value}")
    print(f"   Confidence: {command.confidence:.0%}")
    
    # Get command history
    print("\n5. Getting command history...")
    history = feature.get_command_history(limit=5)
    print(f"   Total commands: {len(history)}")
    for cmd in history:
        print(f"   - {cmd['text']} ({cmd['command_type']})")
    
    # Shutdown
    print("\n6. Shutting down voice interface...")
    await feature.shutdown()
    print("   ✅ Voice interface shut down")


# Example 2: Using TUI Components
def demo_tui_components():
    """Demonstrate TUI components"""
    from textual.app import App, ComposeResult
    from xencode.tui.widgets.voice_interface_panel import (
        VoiceInterfacePanel,
        VoiceStatusIndicator,
        AudioLevelMeter,
        VoiceCommandHistory,
        VoiceSettingsPanel
    )
    
    class VoiceInterfaceDemo(App):
        """Demo app for voice interface TUI components"""
        
        CSS = """
        Screen {
            background: $surface;
        }
        """
        
        BINDINGS = [
            ("q", "quit", "Quit"),
            ("ctrl+v", "toggle_voice", "Toggle Voice"),
        ]
        
        def compose(self) -> ComposeResult:
            """Compose the demo app"""
            yield VoiceInterfacePanel()
        
        def on_mount(self) -> None:
            """Handle mount event"""
            # Simulate some voice commands
            panel = self.query_one(VoiceInterfacePanel)
            
            # Add sample commands
            sample_commands = [
                {
                    'id': 'cmd1',
                    'text': 'open file explorer',
                    'command_type': 'navigation',
                    'confidence': 0.95,
                    'timestamp': '2024-01-15T10:30:00',
                    'parameters': {'target': 'file explorer'}
                },
                {
                    'id': 'cmd2',
                    'text': 'write a function to sort a list',
                    'command_type': 'code',
                    'confidence': 0.88,
                    'timestamp': '2024-01-15T10:31:00',
                    'parameters': {'entity_type': 'function'}
                },
                {
                    'id': 'cmd3',
                    'text': 'explain this code',
                    'command_type': 'query',
                    'confidence': 0.92,
                    'timestamp': '2024-01-15T10:32:00',
                    'parameters': {}
                }
            ]
            
            panel.update_command_history(sample_commands)
            
            # Simulate audio levels
            self.set_interval(0.5, self.update_audio_levels)
        
        def update_audio_levels(self) -> None:
            """Update audio levels periodically"""
            import random
            panel = self.query_one(VoiceInterfacePanel)
            
            # Simulate varying audio levels
            current = random.uniform(0.1, 0.8)
            peak = random.uniform(0.5, 1.0)
            average = random.uniform(0.2, 0.6)
            speaking = current > 0.5
            
            panel.update_audio_levels(current, peak, average, speaking)
        
        def action_toggle_voice(self) -> None:
            """Toggle voice listening"""
            panel = self.query_one(VoiceInterfacePanel)
            panel.action_toggle_listening()
    
    print("\n" + "=" * 60)
    print("Voice Interface TUI Demo")
    print("=" * 60)
    print("\nStarting TUI demo...")
    print("Press 'q' to quit, Ctrl+V to toggle voice")
    print("\nFeatures:")
    print("- Voice status indicator (idle, listening, processing, speaking)")
    print("- Real-time audio level meter")
    print("- Command history with confidence scores")
    print("- Settings panel for voice configuration")
    print("\nStarting in 3 seconds...")
    
    import time
    time.sleep(3)
    
    app = VoiceInterfaceDemo()
    app.run()


# Example 3: CLI Commands Usage
def demo_cli_commands():
    """Demonstrate CLI commands"""
    print("\n" + "=" * 60)
    print("Voice Interface CLI Commands")
    print("=" * 60)
    
    print("\n1. Start voice listening:")
    print("   $ xencode voice start")
    print("   $ xencode voice start --duration 10.0")
    print("   $ xencode voice start --no-silence-detection")
    
    print("\n2. Stop voice listening:")
    print("   $ xencode voice stop")
    
    print("\n3. Show recent voice commands:")
    print("   $ xencode voice commands")
    print("   $ xencode voice commands --limit 20")
    
    print("\n4. Show voice interface status:")
    print("   $ xencode voice status")
    
    print("\n5. Test voice output:")
    print("   $ xencode voice test")
    print("   $ xencode voice test --text 'Hello, this is a test'")
    
    print("\n6. Configure voice settings:")
    print("   $ xencode voice settings --show")
    print("   $ xencode voice settings --set-language es")
    print("   $ xencode voice settings --set-voice en-GB")
    print("   $ xencode voice settings --set-rate 180")
    print("   $ xencode voice settings --set-volume 0.8")


# Example 4: Integration with Feature Manager
async def demo_feature_manager_integration():
    """Demonstrate integration with feature manager"""
    from xencode.features import FeatureManager
    
    print("\n" + "=" * 60)
    print("Feature Manager Integration")
    print("=" * 60)
    
    # Create feature manager
    manager = FeatureManager()
    
    # Enable voice interface feature
    print("\n1. Enabling voice interface feature...")
    success = await manager.initialize_feature("voice_interface")
    if success:
        print("   ✅ Voice interface feature enabled")
    else:
        print("   ❌ Failed to enable voice interface feature")
        return
    
    # Get feature instance
    print("\n2. Getting feature instance...")
    feature = manager.get_feature("voice_interface")
    if feature:
        print(f"   Feature: {feature.name}")
        print(f"   Description: {feature.description}")
        print(f"   Version: {feature.version}")
        print(f"   Status: {feature.get_status().value}")
    
    # Use feature
    print("\n3. Using voice interface...")
    if feature:
        # Get CLI commands
        cli_commands = feature.get_cli_commands()
        print(f"   CLI commands available: {len(cli_commands)}")
        
        # Get TUI components
        tui_components = feature.get_tui_components()
        print(f"   TUI components available: {len(tui_components)}")
        
        # Get API endpoints
        api_endpoints = feature.get_api_endpoints()
        print(f"   API endpoints available: {len(api_endpoints)}")
    
    # Disable feature
    print("\n4. Disabling voice interface feature...")
    success = await manager.shutdown_feature("voice_interface")
    if success:
        print("   ✅ Voice interface feature disabled")


# Example 5: Voice Command Processing
async def demo_voice_command_processing():
    """Demonstrate voice command processing"""
    from xencode.features.voice_interface import VoiceInterfaceFeature, FeatureConfig, CommandType
    
    print("\n" + "=" * 60)
    print("Voice Command Processing Demo")
    print("=" * 60)
    
    # Create feature
    config = FeatureConfig(name="voice_interface", enabled=True, config={})
    feature = VoiceInterfaceFeature(config)
    await feature.initialize()
    
    # Test various voice commands
    test_commands = [
        "write a function to calculate fibonacci",
        "open the file explorer",
        "search for TODO comments",
        "edit the main function",
        "save the current file",
        "explain how recursion works"
    ]
    
    print("\nProcessing voice commands:")
    for text in test_commands:
        command = await feature.process(text)
        print(f"\n  Input: {text}")
        print(f"  Type: {command.command_type.value}")
        print(f"  Confidence: {command.confidence:.0%}")
        if command.parameters:
            print(f"  Parameters: {command.parameters}")
    
    await feature.shutdown()


# Main demo runner
async def main():
    """Run all demos"""
    print("\n" + "=" * 60)
    print("VOICE INTERFACE CLI AND TUI DEMO")
    print("=" * 60)
    
    # Demo 1: Feature usage
    await demo_voice_interface_feature()
    
    # Demo 2: CLI commands
    demo_cli_commands()
    
    # Demo 3: Feature manager integration
    await demo_feature_manager_integration()
    
    # Demo 4: Voice command processing
    await demo_voice_command_processing()
    
    # Demo 5: TUI components (interactive)
    print("\n" + "=" * 60)
    print("Would you like to see the TUI demo? (y/n)")
    response = input("> ").strip().lower()
    if response == 'y':
        demo_tui_components()
    
    print("\n" + "=" * 60)
    print("Demo completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
