import pytest
import asyncio
from xencode.tui.app import XencodeApp
from xencode.tui.widgets.chat import ChatSubmitted
from xencode.tui.utils.model_checker import ModelChecker

@pytest.mark.asyncio
async def test_real_chat_flow():
    """Test real chat flow with actual Ollama model"""
    
    # 1. Check if we have any model installed
    models = ModelChecker.get_available_models()
    if not models:
        pytest.skip("No Ollama models installed to test with")
        
    model = models[0]
    print(f"Testing with model: {model}")
    
    # 2. Setup App
    app = XencodeApp()
    app.current_model = model
    app.use_ensemble = False
    
    # Mock chat panel just to capture output, but logic remains real
    class MockChatPanel:
        def __init__(self):
            self.messages = []
            self.streamed_content = ""
            
        def add_assistant_message(self, content):
            self.messages.append(content)
            return "msg_id"
            
        def update_streaming_message(self, content):
            self.streamed_content = content
            
    app.chat_panel = MockChatPanel()
    
    # 3. Trigger Chat
    event = ChatSubmitted("What is 2+2? Answer briefly.")
    # Manually set content as the event constructor might be different or handled by Textual
    # event.content is already set by constructor
    
    await app.on_chat_submitted(event)
    
    # 4. Verify
    print(f"Streamed content: {app.chat_panel.streamed_content}")
    assert len(app.chat_panel.streamed_content) > 0
    assert "4" in app.chat_panel.streamed_content or "four" in app.chat_panel.streamed_content.lower()
