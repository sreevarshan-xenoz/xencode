"""
Tests for Voice Interface Feature
"""


import pytest

from xencode.features.base import FeatureConfig
from xencode.features.voice_interface import (
    CommandType,
    SpeechToText,
    TextToSpeech,
    VoiceCommand,
    VoiceCommandProcessor,
    VoiceFeedbackSystem,
    VoiceInterfaceConfig,
    VoiceInterfaceFeature,
    VoiceStatus,
)


@pytest.fixture
def voice_config():
    """Create voice interface config"""
    return FeatureConfig(
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


@pytest.fixture
async def voice_interface(voice_config):
    """Create voice interface feature"""
    feature = VoiceInterfaceFeature(voice_config)
    await feature.initialize()
    yield feature
    await feature.shutdown()


class TestVoiceInterfaceFeature:
    """Test VoiceInterfaceFeature class"""

    @pytest.mark.asyncio
    async def test_initialization(self, voice_interface):
        """Test feature initialization"""
        assert voice_interface.name == "voice_interface"
        assert voice_interface.description == "Hands-free coding with voice commands and responses"
        assert voice_interface.is_initialized
        assert voice_interface.speech_to_text is not None
        assert voice_interface.text_to_speech is not None
        assert voice_interface.command_processor is not None
        assert voice_interface.feedback_system is not None

    @pytest.mark.asyncio
    async def test_listen(self, voice_interface):
        """Test listening for voice input"""
        text = await voice_interface.listen(duration=0.1)
        assert text is not None
        assert isinstance(text, str)

    @pytest.mark.asyncio
    async def test_speak(self, voice_interface):
        """Test text-to-speech"""
        await voice_interface.speak("Hello, world!", wait=True)
        # Should complete without error

    @pytest.mark.asyncio
    async def test_process_command(self, voice_interface):
        """Test command processing"""
        command = await voice_interface.process("create a function")
        assert isinstance(command, VoiceCommand)
        assert command.text == "create a function"
        assert command.command_type == CommandType.CODE
        assert 0 <= command.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_start_stop_listening(self, voice_interface):
        """Test continuous listening mode"""
        await voice_interface.start_listening()
        assert voice_interface._listening is True

        await voice_interface.stop_listening()
        assert voice_interface._listening is False

    @pytest.mark.asyncio
    async def test_get_status(self, voice_interface):
        """Test status retrieval"""
        status = voice_interface.get_status()
        assert 'status' in status
        assert 'listening' in status
        assert 'audio_level' in status
        assert 'command_count' in status

    @pytest.mark.asyncio
    async def test_command_history(self, voice_interface):
        """Test command history"""
        # Process some commands
        await voice_interface.process("create a function")
        await voice_interface.process("open file")

        history = voice_interface.get_command_history(limit=10)
        assert len(history) == 2
        assert all('text' in cmd for cmd in history)


class TestSpeechToText:
    """Test SpeechToText class"""

    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test STT initialization"""
        stt = SpeechToText(provider='whisper', model='base', language='en')
        await stt.initialize()
        assert stt._initialized is True
        await stt.shutdown()

    @pytest.mark.asyncio
    async def test_transcribe(self):
        """Test audio transcription"""
        stt = SpeechToText(provider='whisper', model='base', language='en')
        await stt.initialize()

        text = await stt.transcribe(b"audio_data")
        assert isinstance(text, str)
        assert len(text) > 0

        await stt.shutdown()


class TestTextToSpeech:
    """Test TextToSpeech class"""

    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test TTS initialization"""
        tts = TextToSpeech(provider='pyttsx3', voice='en-US', rate=150, volume=0.9)
        await tts.initialize()
        assert tts._initialized is True
        await tts.shutdown()

    @pytest.mark.asyncio
    async def test_speak(self):
        """Test text-to-speech"""
        tts = TextToSpeech(provider='pyttsx3', voice='en-US', rate=150, volume=0.9)
        await tts.initialize()

        await tts.speak("Hello, world!", wait=True)
        # Should complete without error

        await tts.shutdown()


class TestVoiceCommandProcessor:
    """Test VoiceCommandProcessor class"""

    @pytest.mark.asyncio
    async def test_process_code_command(self):
        """Test processing code commands"""
        processor = VoiceCommandProcessor()
        command = await processor.process("create a function")

        assert command.command_type == CommandType.CODE
        assert command.text == "create a function"
        assert command.confidence > 0

    @pytest.mark.asyncio
    async def test_process_navigation_command(self):
        """Test processing navigation commands"""
        processor = VoiceCommandProcessor()
        command = await processor.process("go to main.py")

        assert command.command_type == CommandType.NAVIGATION
        assert 'target' in command.parameters

    @pytest.mark.asyncio
    async def test_process_search_command(self):
        """Test processing search commands"""
        processor = VoiceCommandProcessor()
        command = await processor.process("search for function")

        assert command.command_type == CommandType.SEARCH

    @pytest.mark.asyncio
    async def test_process_edit_command(self):
        """Test processing edit commands"""
        processor = VoiceCommandProcessor()
        command = await processor.process("edit this line")

        assert command.command_type == CommandType.EDIT

    @pytest.mark.asyncio
    async def test_process_system_command(self):
        """Test processing system commands"""
        processor = VoiceCommandProcessor()
        command = await processor.process("save file")

        assert command.command_type == CommandType.SYSTEM

    @pytest.mark.asyncio
    async def test_process_query_command(self):
        """Test processing query commands"""
        processor = VoiceCommandProcessor()
        command = await processor.process("what is this function")

        assert command.command_type == CommandType.QUERY

    @pytest.mark.asyncio
    async def test_noise_reduction(self):
        """Test noise reduction in command processing"""
        processor = VoiceCommandProcessor(noise_reduction=True)
        command = await processor.process("  create   a   function  ")

        assert command.text == "create a function"

    @pytest.mark.asyncio
    async def test_confidence_calculation(self):
        """Test confidence score calculation"""
        processor = VoiceCommandProcessor()

        # Strong match
        command1 = await processor.process("create a new function")
        assert command1.confidence >= 0.7

        # Weak match
        command2 = await processor.process("do something")
        assert command2.confidence >= 0.5


class TestVoiceFeedbackSystem:
    """Test VoiceFeedbackSystem class"""

    @pytest.mark.asyncio
    async def test_update_status(self):
        """Test status updates"""
        feedback = VoiceFeedbackSystem()
        await feedback.update_status(VoiceStatus.LISTENING, "Listening...")

        assert feedback.current_status == VoiceStatus.LISTENING
        assert feedback.status_message == "Listening..."

    @pytest.mark.asyncio
    async def test_show_command(self):
        """Test showing recognized commands"""
        feedback = VoiceFeedbackSystem()
        command = VoiceCommand(
            id='test123',
            text='create function',
            command_type=CommandType.CODE,
            confidence=0.9,
            timestamp='2024-01-01T00:00:00'
        )

        await feedback.show_command(command)
        assert len(feedback.recognized_commands) == 1

    @pytest.mark.asyncio
    async def test_show_response(self):
        """Test showing responses"""
        feedback = VoiceFeedbackSystem()
        await feedback.show_response("Function created successfully")

        assert len(feedback.responses) == 1
        assert feedback.responses[0] == "Function created successfully"

    @pytest.mark.asyncio
    async def test_get_feedback(self):
        """Test getting feedback state"""
        feedback = VoiceFeedbackSystem()
        await feedback.update_status(VoiceStatus.PROCESSING, "Processing...")

        state = feedback.get_feedback()
        assert 'status' in state
        assert 'message' in state
        assert 'recent_commands' in state
        assert 'recent_responses' in state


class TestVoiceInterfaceConfig:
    """Test VoiceInterfaceConfig class"""

    def test_from_dict(self):
        """Test creating config from dictionary"""
        data = {
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

        config = VoiceInterfaceConfig.from_dict(data)
        assert config.enabled is True
        assert config.speech_to_text['provider'] == 'whisper'
        assert config.text_to_speech['provider'] == 'pyttsx3'
        assert config.noise_reduction is True

    def test_default_config(self):
        """Test default configuration"""
        config = VoiceInterfaceConfig()
        assert config.enabled is True
        assert config.speech_to_text['provider'] == 'whisper'
        assert config.text_to_speech['provider'] == 'pyttsx3'
        assert config.noise_reduction is True


class TestIntegration:
    """Integration tests for voice interface"""

    @pytest.mark.asyncio
    async def test_full_voice_workflow(self, voice_interface):
        """Test complete voice interaction workflow"""
        # Start listening
        await voice_interface.start_listening()
        assert voice_interface._listening is True

        # Listen for input
        text = await voice_interface.listen(duration=0.1)
        assert text is not None

        # Process command
        command = await voice_interface.process(text)
        assert isinstance(command, VoiceCommand)

        # Speak response
        await voice_interface.speak("Command received", wait=True)

        # Stop listening
        await voice_interface.stop_listening()
        assert voice_interface._listening is False

        # Check history
        history = voice_interface.get_command_history()
        assert len(history) > 0

    @pytest.mark.asyncio
    async def test_multiple_commands(self, voice_interface):
        """Test processing multiple commands"""
        commands_text = [
            "create a function",
            "open file main.py",
            "search for class",
            "save file"
        ]

        for text in commands_text:
            command = await voice_interface.process(text)
            assert isinstance(command, VoiceCommand)

        history = voice_interface.get_command_history()
        assert len(history) == len(commands_text)

    @pytest.mark.asyncio
    async def test_error_handling(self, voice_interface):
        """Test error handling in voice interface"""
        # Test with invalid duration
        try:
            await voice_interface.listen(duration=-1)
        except Exception:
            pass  # Expected to handle gracefully

        # Status should not be in error state permanently
        status = voice_interface.get_status()
        assert status['status'] in ['idle', 'listening', 'processing', 'speaking']
