"""Memory and context management for agentic conversations."""

from typing import List, Dict, Any, Optional
import uuid
from datetime import datetime

from .database import AgenticDatabase


class ConversationMemory:
    """Manages conversation memory and history."""
    
    def __init__(self, db_path: str = "agentic_memory.db"):
        self.db = AgenticDatabase(db_path)
        self.current_session_id: Optional[str] = None
        self.current_conversation_id: Optional[int] = None
    
    def start_session(self, model_name: str, metadata: Optional[Dict] = None) -> str:
        """Start a new conversation session."""
        session_id = str(uuid.uuid4())
        
        conversation_id = self.db.create_conversation(
            session_id=session_id,
            model_name=model_name,
            metadata=metadata
        )
        
        self.current_session_id = session_id
        self.current_conversation_id = conversation_id
        
        return session_id
    
    def load_session(self, session_id: str) -> bool:
        """Load an existing conversation session."""
        conversation = self.db.get_conversation_by_session(session_id)
        
        if conversation:
            self.current_session_id = session_id
            self.current_conversation_id = conversation['id']
            return True
        
        return False
    
    def add_message(self, role: str, content: str, tool_calls: Optional[List[Dict]] = None):
        """Add a message to the current conversation."""
        if not self.current_conversation_id:
            raise ValueError("No active session. Call start_session() first.")
        
        # Simple token counting (rough estimate: ~4 chars per token)
        token_count = len(content) // 4
        
        message_id = self.db.add_message(
            conversation_id=self.current_conversation_id,
            role=role,
            content=content,
            tool_calls=tool_calls,
            token_count=token_count
        )
        
        return message_id
    
    def get_recent_messages(self, limit: int = 10) -> List[Dict]:
        """Get recent messages from the current conversation."""
        if not self.current_conversation_id:
            return []
        
        messages = self.db.get_conversation_messages(
            conversation_id=self.current_conversation_id,
            limit=limit
        )
        
        # Reverse to get chronological order
        return list(reversed(messages))
    
    def get_conversation_context(self, max_tokens: int = 4000) -> str:
        """Get conversation context within token limit."""
        messages = self.get_recent_messages(limit=50)
        
        context_parts = []
        token_count = 0
        
        # Add messages until we hit the token limit
        for message in reversed(messages):
            msg_tokens = message.get('token_count', 0)
            if token_count + msg_tokens > max_tokens:
                break
            
            role = message['role']
            content = message['content']
            context_parts.insert(0, f"{role.capitalize()}: {content}")
            token_count += msg_tokens
        
        return "\n\n".join(context_parts)
    
    def close(self):
        """Close the database connection."""
        self.db.close()


class ContextManager:
    """Manages context window and token limits."""
    
    def __init__(self, max_tokens: int = 8192):
        self.max_tokens = max_tokens
        self.reserved_tokens = 1000  # Reserve for system prompt and response
    
    def get_available_tokens(self) -> int:
        """Get available tokens for context."""
        return self.max_tokens - self.reserved_tokens
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        # Rough estimate: ~4 characters per token
        return len(text) // 4
    
    def trim_context(self, messages: List[Dict], max_tokens: Optional[int] = None) -> List[Dict]:
        """Trim messages to fit within token limit."""
        if max_tokens is None:
            max_tokens = self.get_available_tokens()
        
        trimmed = []
        token_count = 0
        
        # Start from most recent messages
        for message in reversed(messages):
            msg_tokens = self.estimate_tokens(message.get('content', ''))
            
            if token_count + msg_tokens > max_tokens:
                break
            
            trimmed.insert(0, message)
            token_count += msg_tokens
        
        return trimmed
    
    def format_for_langchain(self, messages: List[Dict]) -> List[Dict[str, str]]:
        """Format messages for LangChain."""
        formatted = []
        
        for message in messages:
            role = message['role']
            content = message['content']
            
            # Map roles to LangChain format
            if role == 'user':
                formatted.append({"role": "human", "content": content})
            elif role == 'assistant':
                formatted.append({"role": "ai", "content": content})
            elif role == 'system':
                formatted.append({"role": "system", "content": content})
        
        return formatted
