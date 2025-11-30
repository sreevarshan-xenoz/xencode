#!/usr/bin/env python3
"""
Verification script for Chat and Model Checker fixes.
"""
import sys
import asyncio
from pathlib import Path

# Add project root to path
sys.path.append(str(Path.cwd()))

from xencode.tui.utils.model_checker import ModelChecker
from xencode.tui.widgets.chat import ChatSubmitted

def test_model_checker():
    print("Testing ModelChecker...")
    try:
        models = ModelChecker.get_available_models()
        print(f"Available models: {models}")
        
        is_ollama = ModelChecker.is_ollama_installed()
        print(f"Ollama installed: {is_ollama}")
        
        if models:
            first = models[0]
            available = ModelChecker.check_model_availability(first)
            print(f"Check availability of '{first}': {available}")
            
        unavailable = ModelChecker.check_model_availability("nonexistent_model_xyz")
        print(f"Check availability of 'nonexistent_model_xyz': {unavailable}")
        
    except Exception as e:
        print(f"ModelChecker failed: {e}")

def test_chat_event():
    print("\nTesting ChatSubmitted event...")
    try:
        event = ChatSubmitted("Hello World")
        # Verify content attribute exists (this was the bug)
        print(f"Event content: {event.content}")
        
        # Verify message attribute does NOT exist (to confirm the bug was real and fix is needed)
        try:
            print(f"Event message: {event.message}")
        except AttributeError:
            print("Confirmed: 'message' attribute does not exist (as expected)")
            
    except Exception as e:
        print(f"ChatSubmitted test failed: {e}")

if __name__ == "__main__":
    test_model_checker()
    test_chat_event()
