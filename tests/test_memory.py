import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from xencode.agentic.memory import ConversationMemory

def test_memory_system():
    print("Testing Memory System...")
    
    # Create memory
    memory = ConversationMemory(db_path="test_memory.db")
    
    # Start session
    session_id = memory.start_session(model_name="test-model")
    print(f"✅ Started session: {session_id}")
    
    # Add messages
    memory.add_message(role="user", content="Hello, how are you?")
    memory.add_message(role="assistant", content="I'm doing well, thank you!")
    memory.add_message(role="user", content="Can you help me with code?")
    memory.add_message(role="assistant", content="Of course! I'd be happy to help.")
    
    print("✅ Added 4 messages")
    
    # Get recent messages
    messages = memory.get_recent_messages(limit=10)
    print(f"✅ Retrieved {len(messages)} messages")
    
    for msg in messages:
        print(f"  - {msg['role']}: {msg['content'][:50]}...")
    
    # Get context
    context = memory.get_conversation_context(max_tokens=1000)
    print(f"\n✅ Generated context ({len(context)} chars)")
    
    memory.close()
    
    # Clean up test DB
    import os
    if os.path.exists("test_memory.db"):
        os.remove("test_memory.db")
        print("\n✅ Cleaned up test database")
    
    print("\n✅ All memory tests passed!")

if __name__ == "__main__":
    test_memory_system()
