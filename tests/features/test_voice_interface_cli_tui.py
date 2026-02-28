#!/usr/bin/env python3
"""
Tests for Voice Interface CLI and TUI Components

Tests the CLI commands and TUI widgets for the voice interface feature.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

from xencode.features.voice_interface import (
    VoiceInterfaceFeature,
    FeatureConfig,
    VoiceStatus,
    CommandType,
    VoiceCommand
)


class TestVoiceInterfaceCLI:
    """Tests for voice interface CLI commands"""
    
    @pytest.fixture
    def feature(self):
        """Create a voice interface feature instance"""
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
                }
            }
        )
        return VoiceInterfaceFeature(config)
    
    @pytest.mark.asyncio
    async def test_cli_commands_available(self, feature):
        """Test that CLI commands are available"""
        await feature.initialize()
        
        cli_commands = feature.get_cli_commands()
        
        assert len(cli_commands) > 0
        assert cli_commands[0].name == 'voice'
        
        await feature.shutdown()
    
    @pytest.mark.asyncio
    async def test_voice_status_command(self, feature):
        """Test voice status command functionality"""
        await feature.initialize()
        
        status = feature.get_status()
        
        assert 'status' in status
        assert 'listening' in status
        assert 'audio_level' in status
        assert 'command_count' in status
        
        assert status['status'] == 'idle'
        assert status['listening'] is False
        assert status['command_count'] == 0
        
        await feature.shutdown()
    
    @pytest.mark.asyncio
    async def test_voice_commands_history(self, feature):
        """Test voice commands history functionality"""
        await feature.initialize()
        
        # Add some commands
        cmd1 = await feature.process("open file explorer")
        cmd2 = await feature.process("write a function")
        
        # Get history
        history = feature.get_command_history(limit=10)
        
        assert len(history) == 2
        assert history[0]['text'] == "open file explorer"
        assert history[1]['text'] == "write a function"
        
        await feature.shutdown()
    
    @pytest.mark.asyncio
    async def test_voice_settings_get(self, feature):
        """Test getting voice settings"""
        await feature.initialize()
        
        # Check settings are accessible
        assert feature.vi_config.speech_to_text['provider'] == 'whisper'
        assert feature.vi_config.speech_to_text['model'] == 'base'
        assert feature.vi_config.text_to_speech['voice'] == 'en-US'
        assert feature.vi_config.text_to_speech['rate'] == 150
        
        await feature.shutdown()
    
    @pytest.mark.asyncio
    async def test_voice_settings_update(self, feature):
        """Test updating voice settings"""
        await feature.initialize()
        
        # Update settings
        feature.vi_config.speech_to_text['language'] = 'es'
        feature.vi_config.text_to_speech['rate'] = 180
        feature.vi_config.text_to_speech['volume'] = 0.8
        
        # Verify updates
        assert feature.vi_config.speech_to_text['language'] == 'es'
        assert feature.vi_config.text_to_speech['rate'] == 180
        assert feature.vi_config.text_to_speech['volume'] == 0.8
        
        await feature.shutdown()
    
    @pytest.mark.asyncio
    async def test_start_stop_listening(self, feature):
        """Test start and stop listening functionality"""
        await feature.initialize()
        
        # Start listening
        await feature.start_listening()
        assert feature._listening is True
        
        status = feature.get_status()
        assert status['listening'] is True
        
        # Stop listening
        await feature.stop_listening()
        assert feature._listening is False
        
        status = feature.get_status()
        assert status['listening'] is False
        
        await feature.shutdown()
    
    @pytest.mark.asyncio
    async def test_voice_command_processing(self, feature):
        """Test voice command processing"""
        await feature.initialize()
        
        # Test different command types
        test_cases = [
            ("write a function", CommandType.CODE),
            ("open file explorer", CommandType.NAVIGATION),
            ("search for TODO", CommandType.SEARCH),
            ("edit main function", CommandType.EDIT),
            ("save file", CommandType.SYSTEM),
            ("explain recursion", CommandType.QUERY)
        ]
        
        for text, expected_type in test_cases:
            command = await feature.process(text)
            assert command.text == text
            assert command.command_type == expected_type
            assert 0 <= command.confidence <= 1
            assert command.id is not None
            assert command.timestamp is not None
        
        await feature.shutdown()


class TestVoiceInterfaceTUI:
    """Tests for voice interface TUI components"""
    
    def test_tui_components_available(self):
        """Test that TUI components are available"""
        config = FeatureConfig(name="voice_interface", enabled=True, config={})
        feature = VoiceInterfaceFeature(config)
        
        tui_components = feature.get_tui_components()
        
        assert len(tui_components) > 0
        assert tui_components[0].__name__ == 'VoiceInterfacePanel'
    
    def test_voice_status_indicator(self):
        """Test VoiceStatusIndicator component"""
        from xencode.tui.widgets.voice_interface_panel import VoiceStatusIndicator
        
        indicator = VoiceStatusIndicator()
        
        # Test status changes
        indicator.set_status('listening')
        assert indicator.status == 'listening'
        
        indicator.set_status('processing', 'Processing command...')
        assert indicator.status == 'processing'
        
        indicator.set_status('speaking')
        assert indicator.status == 'speaking'
        
        indicator.set_status('idle')
        assert indicator.status == 'idle'
    
    def test_audio_level_meter(self):
        """Test AudioLevelMeter component"""
        from xencode.tui.widgets.voice_interface_panel import AudioLevelMeter
        
        meter = AudioLevelMeter()
        
        # Test level updates
        meter.update_levels(0.5, 0.8, 0.6, True)
        
        assert meter.current_level == 0.5
        assert meter.peak_level == 0.8
        assert meter.average_level == 0.6
        assert meter.is_speaking is True
    
    def test_voice_command_history(self):
        """Test VoiceCommandHistory component"""
        from xencode.tui.widgets.voice_interface_panel import VoiceCommandHistory
        
        history = VoiceCommandHistory()
        
        # Add commands
        commands = [
            {
                'id': 'cmd1',
                'text': 'open file',
                'command_type': 'navigation',
                'confidence': 0.95,
                'timestamp': datetime.now().isoformat()
            },
            {
                'id': 'cmd2',
                'text': 'write function',
                'command_type': 'code',
                'confidence': 0.88,
                'timestamp': datetime.now().isoformat()
            }
        ]
        
        history.update_history(commands)
        assert len(history.commands) == 2
        
        # Add single command
        new_command = {
            'id': 'cmd3',
            'text': 'search TODO',
            'command_type': 'search',
            'confidence': 0.92,
            'timestamp': datetime.now().isoformat()
        }
        
        history.add_command(new_command)
        assert len(history.commands) == 3
    
    def test_voice_settings_panel(self):
        """Test VoiceSettingsPanel component"""
        from xencode.tui.widgets.voice_interface_panel import VoiceSettingsPanel
        
        panel = VoiceSettingsPanel()
        
        # Get default settings
        settings = panel.get_settings()
        assert 'language' in settings
        assert 'voice' in settings
        assert 'rate' in settings
        assert 'volume' in settings
        
        # Update settings
        new_settings = {
            'language': 'es',
            'rate': 180,
            'volume': 0.8
        }
        
        panel.update_settings(new_settings)
        updated_settings = panel.get_settings()
        
        assert updated_settings['language'] == 'es'
        assert updated_settings['rate'] == 180
        assert updated_settings['volume'] == 0.8
    
    def test_voice_interface_panel_integration(self):
        """Test VoiceInterfacePanel integration"""
        from xencode.tui.widgets.voice_interface_panel import VoiceInterfacePanel
        
        panel = VoiceInterfacePanel()
        
        # Test status update
        panel.update_status('listening', 'Listening for input...')
        
        # Test audio level update
        panel.update_audio_levels(0.6, 0.9, 0.5, True)
        
        # Test command addition
        command = {
            'id': 'cmd1',
            'text': 'test command',
            'command_type': 'query',
            'confidence': 0.9,
            'timestamp': datetime.now().isoformat()
        }
        
        panel.add_command(command)
        
        # Test settings
        settings = panel.get_settings()
        assert settings is not None
        
        # Test tab switching
        panel._show_tab('settings')
        assert panel.current_tab == 'settings'
        
        panel._show_tab('history')
        assert panel.current_tab == 'history'


class TestVoiceCommandTypes:
    """Tests for voice command type detection"""
    
    @pytest.mark.asyncio
    async def test_code_command_detection(self):
        """Test detection of code commands"""
        config = FeatureConfig(name="voice_interface", enabled=True, config={})
        feature = VoiceInterfaceFeature(config)
        await feature.initialize()
        
        code_commands = [
            "write a function",
            "create a class",
            "generate code",
            "code a solution"
        ]
        
        for text in code_commands:
            command = await feature.process(text)
            assert command.command_type == CommandType.CODE
        
        await feature.shutdown()
    
    @pytest.mark.asyncio
    async def test_navigation_command_detection(self):
        """Test detection of navigation commands"""
        config = FeatureConfig(name="voice_interface", enabled=True, config={})
        feature = VoiceInterfaceFeature(config)
        await feature.initialize()
        
        nav_commands = [
            "go to main",
            "open file",
            "navigate to settings",
            "show panel"
        ]
        
        for text in nav_commands:
            command = await feature.process(text)
            assert command.command_type == CommandType.NAVIGATION
        
        await feature.shutdown()
    
    @pytest.mark.asyncio
    async def test_search_command_detection(self):
        """Test detection of search commands"""
        config = FeatureConfig(name="voice_interface", enabled=True, config={})
        feature = VoiceInterfaceFeature(config)
        await feature.initialize()
        
        search_commands = [
            "search for TODO",
            "find references",
            "look for bugs",
            "where is the config"
        ]
        
        for text in search_commands:
            command = await feature.process(text)
            assert command.command_type == CommandType.SEARCH
        
        await feature.shutdown()
    
    @pytest.mark.asyncio
    async def test_edit_command_detection(self):
        """Test detection of edit commands"""
        config = FeatureConfig(name="voice_interface", enabled=True, config={})
        feature = VoiceInterfaceFeature(config)
        await feature.initialize()
        
        edit_commands = [
            "edit function",
            "change variable",
            "modify code",
            "delete line"
        ]
        
        for text in edit_commands:
            command = await feature.process(text)
            assert command.command_type == CommandType.EDIT
        
        await feature.shutdown()
    
    @pytest.mark.asyncio
    async def test_system_command_detection(self):
        """Test detection of system commands"""
        config = FeatureConfig(name="voice_interface", enabled=True, config={})
        feature = VoiceInterfaceFeature(config)
        await feature.initialize()
        
        system_commands = [
            "save file",
            "close window",
            "exit program",
            "help me"
        ]
        
        for text in system_commands:
            command = await feature.process(text)
            assert command.command_type == CommandType.SYSTEM
        
        await feature.shutdown()
    
    @pytest.mark.asyncio
    async def test_query_command_detection(self):
        """Test detection of query commands"""
        config = FeatureConfig(name="voice_interface", enabled=True, config={})
        feature = VoiceInterfaceFeature(config)
        await feature.initialize()
        
        query_commands = [
            "what is recursion",
            "how does this work",
            "why is this failing",
            "explain the code"
        ]
        
        for text in query_commands:
            command = await feature.process(text)
            assert command.command_type == CommandType.QUERY
        
        await feature.shutdown()


class TestVoiceInterfaceIntegration:
    """Integration tests for voice interface"""
    
    @pytest.mark.asyncio
    async def test_full_voice_workflow(self):
        """Test complete voice interface workflow"""
        config = FeatureConfig(name="voice_interface", enabled=True, config={})
        feature = VoiceInterfaceFeature(config)
        
        # Initialize
        await feature.initialize()
        assert feature.is_enabled
        
        # Start listening
        await feature.start_listening()
        assert feature._listening is True
        
        # Process commands
        cmd1 = await feature.process("write a function")
        assert cmd1.command_type == CommandType.CODE
        
        cmd2 = await feature.process("open file explorer")
        assert cmd2.command_type == CommandType.NAVIGATION
        
        # Check history
        history = feature.get_command_history()
        assert len(history) == 2
        
        # Check status
        status = feature.get_status()
        assert status['listening'] is True
        assert status['command_count'] == 2
        
        # Stop listening
        await feature.stop_listening()
        assert feature._listening is False
        
        # Shutdown
        await feature.shutdown()
        assert not feature.is_enabled
    
    @pytest.mark.asyncio
    async def test_cli_and_tui_integration(self):
        """Test CLI and TUI integration"""
        config = FeatureConfig(name="voice_interface", enabled=True, config={})
        feature = VoiceInterfaceFeature(config)
        await feature.initialize()
        
        # Get CLI commands
        cli_commands = feature.get_cli_commands()
        assert len(cli_commands) > 0
        
        # Get TUI components
        tui_components = feature.get_tui_components()
        assert len(tui_components) > 0
        
        # Get API endpoints
        api_endpoints = feature.get_api_endpoints()
        assert len(api_endpoints) > 0
        
        # Verify they work together
        command = await feature.process("test command")
        assert command is not None
        
        status = feature.get_status()
        assert status is not None
        
        await feature.shutdown()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
