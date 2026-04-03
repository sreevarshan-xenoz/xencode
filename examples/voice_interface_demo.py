"""
Voice Interface Demo

Demonstrates the voice interface feature for hands-free coding.
"""

import asyncio
from xencode.features.voice_interface import (
    VoiceInterfaceFeature,
    VoiceStatus,
    CommandType
)
from xencode.features.base import FeatureConfig


async def demo_basic_usage():
    """Demonstrate basic voice interface usage"""
    print("=== Voice Interface Demo ===\n")
    
    # Create configuration
    config = FeatureConfig(
        name='voice_interface',
        enabled=True,
        config={
            'enabled': True,
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
            'silence_duration': 1.0,
            'command_timeout': 5.0
        }
    )
    
    # Initialize voice interface
    print("1. Initializing voice interface...")
    voice = VoiceInterfaceFeature(config)
    await voice.initialize()
    print(f"   ✓ Voice interface initialized: {voice.description}\n")
    
    # Check status
    print("2. Checking status...")
    status = voice.get_status()
    print(f"   Status: {status['status']}")
    print(f"   Listening: {status['listening']}")
    print(f"   Commands processed: {status['command_count']}\n")
    
    # Demonstrate listening (mock)
    print("3. Listening for voice input...")
    text = await voice.listen(duration=0.1)
    print(f"   ✓ Heard: '{text}'\n")
    
    # Process commands
    print("4. Processing voice commands...")
    commands = [
        "create a function",
        "open file main.py",
        "search for class User",
        "edit line 42",
        "save file",
        "what is this function"
    ]
    
    for cmd_text in commands:
        command = await voice.process(cmd_text)
        print(f"   Command: '{command.text}'")
        print(f"   Type: {command.command_type.value}")
        print(f"   Confidence: {command.confidence:.2f}")
        if command.parameters:
            print(f"   Parameters: {command.parameters}")
        print()
    
    # Demonstrate text-to-speech
    print("5. Speaking response...")
    await voice.speak("All commands processed successfully", wait=True)
    print("   ✓ Speech completed\n")
    
    # Show command history
    print("6. Command history:")
    history = voice.get_command_history(limit=5)
    for i, cmd in enumerate(history, 1):
        print(f"   {i}. {cmd['text']} ({cmd['command_type']})")
    print()
    
    # Demonstrate continuous listening mode
    print("7. Testing continuous listening mode...")
    await voice.start_listening()
    print("   ✓ Started listening")
    
    status = voice.get_status()
    print(f"   Listening: {status['listening']}")
    
    await voice.stop_listening()
    print("   ✓ Stopped listening\n")
    
    # Shutdown
    print("8. Shutting down...")
    await voice.shutdown()
    print("   ✓ Voice interface shut down\n")


async def demo_command_types():
    """Demonstrate different command types"""
    print("=== Command Type Recognition Demo ===\n")
    
    config = FeatureConfig(
        name='voice_interface',
        enabled=True,
        config={}
    )
    
    voice = VoiceInterfaceFeature(config)
    await voice.initialize()
    
    # Test different command types
    test_commands = {
        CommandType.CODE: [
            "create a new function",
            "write a class",
            "generate code for sorting"
        ],
        CommandType.NAVIGATION: [
            "go to line 50",
            "open the settings file",
            "show me the main function"
        ],
        CommandType.SEARCH: [
            "search for TODO comments",
            "find all imports",
            "where is the User class"
        ],
        CommandType.EDIT: [
            "edit this line",
            "change the variable name",
            "delete this function"
        ],
        CommandType.SYSTEM: [
            "save the file",
            "close the editor",
            "help me with commands"
        ],
        CommandType.QUERY: [
            "what does this function do",
            "how do I use this API",
            "explain this code"
        ]
    }
    
    for cmd_type, examples in test_commands.items():
        print(f"{cmd_type.value.upper()} Commands:")
        for example in examples:
            command = await voice.process(example)
            match = "✓" if command.command_type == cmd_type else "✗"
            print(f"  {match} '{example}'")
            print(f"     Detected as: {command.command_type.value} (confidence: {command.confidence:.2f})")
        print()
    
    await voice.shutdown()


async def demo_feedback_system():
    """Demonstrate voice feedback system"""
    print("=== Voice Feedback System Demo ===\n")
    
    config = FeatureConfig(
        name='voice_interface',
        enabled=True,
        config={}
    )
    
    voice = VoiceInterfaceFeature(config)
    await voice.initialize()
    
    # Process some commands
    print("Processing commands with feedback...\n")
    
    commands = [
        "create a function named calculate",
        "open the config file",
        "search for error handling"
    ]
    
    for cmd_text in commands:
        print(f"Command: '{cmd_text}'")
        
        # Process command
        command = await voice.process(cmd_text)
        
        # Get feedback
        feedback = voice.feedback_system.get_feedback()
        print(f"  Status: {feedback['status']}")
        print(f"  Recent commands: {len(feedback['recent_commands'])}")
        print()
    
    # Show all feedback
    feedback = voice.feedback_system.get_feedback()
    print("Complete Feedback State:")
    print(f"  Status: {feedback['status']}")
    print(f"  Message: {feedback['message']}")
    print(f"  Commands recognized: {len(feedback['recent_commands'])}")
    print(f"  Responses given: {len(feedback['recent_responses'])}")
    print()
    
    await voice.shutdown()


async def demo_audio_levels():
    """Demonstrate audio level monitoring"""
    print("=== Audio Level Monitoring Demo ===\n")
    
    config = FeatureConfig(
        name='voice_interface',
        enabled=True,
        config={}
    )
    
    voice = VoiceInterfaceFeature(config)
    await voice.initialize()
    
    print("Audio Level Information:")
    audio_level = voice.get_audio_level()
    print(f"  Current: {audio_level.current:.2f}")
    print(f"  Peak: {audio_level.peak:.2f}")
    print(f"  Average: {audio_level.average:.2f}")
    print(f"  Is Speaking: {audio_level.is_speaking}")
    print()
    
    await voice.shutdown()


async def main():
    """Run all demos"""
    try:
        await demo_basic_usage()
        print("\n" + "="*50 + "\n")
        
        await demo_command_types()
        print("\n" + "="*50 + "\n")
        
        await demo_feedback_system()
        print("\n" + "="*50 + "\n")
        
        await demo_audio_levels()
        
        print("\n✓ All demos completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
