#!/usr/bin/env python3
"""
Tests for Collaborative Coding CLI and TUI

Tests the CLI commands and TUI components for the collaborative coding feature.
"""

from unittest.mock import AsyncMock

import pytest
from click.testing import CliRunner

from xencode.features.collaborative_coding import CollaborativeCodingFeature


class TestCollaborativeCodingCLI:
    """Test collaborative coding CLI commands"""

    @pytest.fixture
    def feature(self):
        """Create a collaborative coding feature instance"""
        from xencode.features.base import FeatureConfig
        config = FeatureConfig(
            name="collaborative_coding",
            enabled=True,
            config={}
        )
        return CollaborativeCodingFeature(config)

    @pytest.fixture
    def runner(self):
        """Create a CLI runner"""
        return CliRunner()

    def test_get_cli_commands_returns_list(self, feature):
        """Test that get_cli_commands returns a list"""
        commands = feature.get_cli_commands()
        assert isinstance(commands, list)
        assert len(commands) > 0

    def test_cli_commands_have_collab_group(self, feature):
        """Test that CLI commands include collab group"""
        commands = feature.get_cli_commands()
        assert len(commands) == 1

        # Check that it's a click group
        collab_group = commands[0]
        assert hasattr(collab_group, 'name')
        assert collab_group.name == 'collab'

    def test_collab_group_has_start_command(self, feature):
        """Test that collab group has start command"""
        commands = feature.get_cli_commands()
        collab_group = commands[0]

        # Check for start command
        assert 'start' in collab_group.commands

    def test_collab_group_has_join_command(self, feature):
        """Test that collab group has join command"""
        commands = feature.get_cli_commands()
        collab_group = commands[0]

        # Check for join command
        assert 'join' in collab_group.commands

    def test_collab_group_has_leave_command(self, feature):
        """Test that collab group has leave command"""
        commands = feature.get_cli_commands()
        collab_group = commands[0]

        # Check for leave command
        assert 'leave' in collab_group.commands

    def test_collab_group_has_list_command(self, feature):
        """Test that collab group has list command"""
        commands = feature.get_cli_commands()
        collab_group = commands[0]

        # Check for list command
        assert 'list' in collab_group.commands

    def test_collab_group_has_users_command(self, feature):
        """Test that collab group has users command"""
        commands = feature.get_cli_commands()
        collab_group = commands[0]

        # Check for users command
        assert 'users' in collab_group.commands

    @pytest.mark.asyncio
    async def test_start_command_creates_session(self, feature, runner):
        """Test that start command creates a session"""
        # Mock the start method
        feature.start = AsyncMock(return_value={
            'success': True,
            'session': {
                'session_id': 'test-session-123',
                'name': 'Test Room',
                'owner_id': 'user-123',
                'participant_count': 1,
                'participants': {}
            }
        })

        commands = feature.get_cli_commands()
        collab_group = commands[0]

        # Run the start command
        result = runner.invoke(collab_group, ['start', 'Test Room', '--username', 'TestUser'])

        # Check that command executed
        assert result.exit_code == 0
        assert 'Starting Collaboration Session' in result.output or 'Session started' in result.output

    @pytest.mark.asyncio
    async def test_join_command_joins_session(self, feature, runner):
        """Test that join command joins a session"""
        # Mock the join method
        feature.join = AsyncMock(return_value={
            'success': True,
            'session': {
                'session_id': 'test-session-123',
                'name': 'Test Room',
                'owner_id': 'user-123',
                'participant_count': 2,
                'participants': {}
            }
        })

        commands = feature.get_cli_commands()
        collab_group = commands[0]

        # Run the join command
        result = runner.invoke(collab_group, ['join', 'test-session-123', '--username', 'TestUser'])

        # Check that command executed
        assert result.exit_code == 0
        assert 'Joining Collaboration Session' in result.output or 'Joined session' in result.output


class TestCollaborativeCodingTUI:
    """Test collaborative coding TUI components"""

    @pytest.fixture
    def feature(self):
        """Create a collaborative coding feature instance"""
        from xencode.features.base import FeatureConfig
        config = FeatureConfig(
            name="collaborative_coding",
            enabled=True,
            config={}
        )
        return CollaborativeCodingFeature(config)

    def test_get_tui_components_returns_list(self, feature):
        """Test that get_tui_components returns a list"""
        components = feature.get_tui_components()
        assert isinstance(components, list)
        assert len(components) > 0

    def test_tui_components_include_panel(self, feature):
        """Test that TUI components include CollaborativeCodingPanel"""
        components = feature.get_tui_components()

        # Check that we have the panel class
        from xencode.tui.widgets.collaborative_coding_panel import (
            CollaborativeCodingPanel,
        )
        assert CollaborativeCodingPanel in components

    def test_collaborative_coding_panel_can_be_instantiated(self, feature):
        """Test that CollaborativeCodingPanel can be instantiated"""
        from xencode.tui.widgets.collaborative_coding_panel import (
            CollaborativeCodingPanel,
        )

        panel = CollaborativeCodingPanel()
        assert panel is not None
        assert hasattr(panel, 'session_id')
        assert hasattr(panel, 'user_id')
        assert hasattr(panel, 'username')
        assert hasattr(panel, 'is_connected')

    def test_presence_indicator_can_be_instantiated(self):
        """Test that PresenceIndicator can be instantiated"""
        from xencode.tui.widgets.collaborative_coding_panel import PresenceIndicator

        indicator = PresenceIndicator()
        assert indicator is not None
        assert hasattr(indicator, 'users')

    def test_presence_indicator_add_user(self):
        """Test adding a user to presence indicator"""
        from xencode.tui.widgets.collaborative_coding_panel import PresenceIndicator

        indicator = PresenceIndicator()
        indicator.add_user('user-123', 'TestUser', '#FF6B6B', 'online')

        assert 'user-123' in indicator.users
        assert indicator.users['user-123']['username'] == 'TestUser'
        assert indicator.users['user-123']['status'] == 'online'

    def test_presence_indicator_update_status(self):
        """Test updating user status in presence indicator"""
        from xencode.tui.widgets.collaborative_coding_panel import PresenceIndicator

        indicator = PresenceIndicator()
        indicator.add_user('user-123', 'TestUser', '#FF6B6B', 'online')
        indicator.update_user_status('user-123', 'idle')

        assert indicator.users['user-123']['status'] == 'idle'

    def test_presence_indicator_remove_user(self):
        """Test removing a user from presence indicator"""
        from xencode.tui.widgets.collaborative_coding_panel import PresenceIndicator

        indicator = PresenceIndicator()
        indicator.add_user('user-123', 'TestUser', '#FF6B6B', 'online')
        indicator.remove_user('user-123')

        assert 'user-123' not in indicator.users

    def test_cursor_tracker_can_be_instantiated(self):
        """Test that CursorTracker can be instantiated"""
        from xencode.tui.widgets.collaborative_coding_panel import CursorTracker

        tracker = CursorTracker()
        assert tracker is not None
        assert hasattr(tracker, 'cursors')

    def test_cursor_tracker_update_cursor(self):
        """Test updating cursor position"""
        from xencode.tui.widgets.collaborative_coding_panel import CursorTracker

        tracker = CursorTracker()
        tracker.update_cursor('user-123', 'TestUser', 42, 'test.py')

        assert 'user-123' in tracker.cursors
        assert tracker.cursors['user-123']['position'] == 42
        assert tracker.cursors['user-123']['file_path'] == 'test.py'

    def test_chat_interface_can_be_instantiated(self):
        """Test that ChatInterface can be instantiated"""
        from xencode.tui.widgets.collaborative_coding_panel import ChatInterface

        chat = ChatInterface()
        assert chat is not None
        assert hasattr(chat, 'messages')

    def test_chat_interface_add_message(self):
        """Test adding a message to chat"""
        from xencode.tui.widgets.collaborative_coding_panel import ChatInterface

        chat = ChatInterface()
        chat.add_message('TestUser', 'Hello, world!', is_self=True)

        assert len(chat.messages) == 1
        assert chat.messages[0]['username'] == 'TestUser'
        assert chat.messages[0]['text'] == 'Hello, world!'
        assert chat.messages[0]['is_self'] is True

    def test_conflict_resolution_panel_can_be_instantiated(self):
        """Test that ConflictResolutionPanel can be instantiated"""
        from xencode.tui.widgets.collaborative_coding_panel import (
            ConflictResolutionPanel,
        )

        panel = ConflictResolutionPanel()
        assert panel is not None
        assert hasattr(panel, 'conflicts')
        assert hasattr(panel, 'visible')

    def test_conflict_resolution_panel_show_conflicts(self):
        """Test showing conflicts in panel"""
        from xencode.tui.widgets.collaborative_coding_panel import (
            ConflictResolutionPanel,
        )

        panel = ConflictResolutionPanel()
        conflicts = [
            {'description': 'Conflict at line 10'},
            {'description': 'Conflict at line 25'}
        ]
        panel.show_conflicts(conflicts)

        assert panel.visible is True
        assert len(panel.conflicts) == 2

    def test_conflict_resolution_panel_hide_conflicts(self):
        """Test hiding conflicts panel"""
        from xencode.tui.widgets.collaborative_coding_panel import (
            ConflictResolutionPanel,
        )

        panel = ConflictResolutionPanel()
        conflicts = [{'description': 'Test conflict'}]
        panel.show_conflicts(conflicts)
        panel.hide_conflicts()

        assert panel.visible is False
        assert len(panel.conflicts) == 0

    def test_real_time_editor_can_be_instantiated(self):
        """Test that RealTimeEditor can be instantiated"""
        from xencode.tui.widgets.collaborative_coding_panel import RealTimeEditor

        editor = RealTimeEditor()
        assert editor is not None
        assert hasattr(editor, 'document_version')
        assert hasattr(editor, 'is_syncing')

    def test_real_time_editor_update_content(self):
        """Test updating editor content"""
        from xencode.tui.widgets.collaborative_coding_panel import RealTimeEditor

        editor = RealTimeEditor()
        editor.update_content('Test content', 5)

        assert editor.document_version == 5


class TestCollaborativeCodingIntegration:
    """Integration tests for collaborative coding CLI and TUI"""

    @pytest.fixture
    def feature(self):
        """Create a collaborative coding feature instance"""
        from xencode.features.base import FeatureConfig
        config = FeatureConfig(
            name="collaborative_coding",
            enabled=True,
            config={}
        )
        return CollaborativeCodingFeature(config)

    def test_feature_has_both_cli_and_tui(self, feature):
        """Test that feature provides both CLI and TUI components"""
        cli_commands = feature.get_cli_commands()
        tui_components = feature.get_tui_components()

        assert len(cli_commands) > 0
        assert len(tui_components) > 0

    def test_cli_commands_are_properly_structured(self, feature):
        """Test that CLI commands follow proper structure"""
        commands = feature.get_cli_commands()

        # Should have one group
        assert len(commands) == 1

        # Group should have multiple commands
        collab_group = commands[0]
        assert len(collab_group.commands) >= 5  # start, join, leave, list, users

    def test_tui_components_are_properly_structured(self, feature):
        """Test that TUI components follow proper structure"""
        components = feature.get_tui_components()

        # Should have at least one component
        assert len(components) >= 1

        # Component should be a class
        from xencode.tui.widgets.collaborative_coding_panel import (
            CollaborativeCodingPanel,
        )
        assert CollaborativeCodingPanel in components


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
