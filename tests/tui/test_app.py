#!/usr/bin/env python3
"""
Tests for Xencode TUI Application
"""

import pytest
from pathlib import Path
from textual.pilot import Pilot

from xencode.tui.app import XencodeApp
from xencode.tui.widgets.file_explorer import FileExplorer
from xencode.tui.widgets.editor import CodeEditor
from xencode.tui.widgets.chat import ChatPanel, ChatMessage


class TestXencodeApp:
    """Test Xencode TUI App"""
    
    @pytest.mark.asyncio
    async def test_app_launches(self):
        """Test that app launches successfully"""
        app = XencodeApp()
        async with app.run_test() as pilot:
            # Check that app is running
            assert app.is_running
            
            # Check that main widgets exist
            assert app.file_explorer is not None
            assert app.code_editor is not None
            assert app.chat_panel is not None
    
    @pytest.mark.asyncio
    async def test_file_explorer_populated(self):
        """Test that file explorer shows files"""
        test_dir = Path(__file__).parent.parent
        app = XencodeApp(root_path=test_dir)
        
        async with app.run_test() as pilot:
            # Wait for mount
            await pilot.pause()
            
            # Check that file explorer has children
            assert app.file_explorer is not None
            assert len(app.file_explorer.root.children) > 0


class TestChatPanel:
    """Test Chat Panel Widget"""
    
    def test_chat_message_creation(self):
        """Test creating chat messages"""
        msg = ChatMessage("user", "Hello AI")
        
        assert msg.role == "user"
        assert msg.content_text == "Hello AI"
        assert "user" in msg.classes
    
    @pytest.mark.asyncio
    async def test_chat_interaction(self):
        """Test chat message interaction"""
        app = XencodeApp()
        
        async with app.run_test() as pilot:
            # Wait for mount
            await pilot.pause()
            
            # Add a message
            if app.chat_panel:
                app.chat_panel.add_user_message("Test message")
                await pilot.pause()
                
                # Check message was added
                assert len(app.chat_panel.history.children) > 1  # Welcome + test message


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
