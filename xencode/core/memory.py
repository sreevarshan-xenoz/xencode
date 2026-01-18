"""
Conversation memory module for Xencode
"""
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Conversation memory configuration
MAX_MEMORY_ITEMS: int = 50
MEMORY_FILE: Path = Path.home() / ".xencode" / "conversation_memory.json"


class ConversationMemory:
    """Advanced conversation memory with context management"""

    def __init__(self, max_items: int = MAX_MEMORY_ITEMS) -> None:
        """Initialize the ConversationMemory with default values.

        Args:
            max_items: Maximum number of items to store in memory
        """
        self.max_items: int = max_items
        self.conversations: Dict[str, Any] = {}
        self.current_session: Optional[str] = None
        self.load_memory()

    def load_memory(self) -> None:
        """Load conversation memory from disk"""
        try:
            if MEMORY_FILE.exists():
                with open(MEMORY_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.conversations = data.get('conversations', {})
                    self.current_session = data.get('current_session')
        except Exception:
            self.conversations = {}
            self.current_session = None

    def save_memory(self) -> None:
        """Save conversation memory to disk"""
        try:
            cache_dir = Path.home() / ".xencode" / "cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            with open(MEMORY_FILE, 'w', encoding='utf-8') as f:
                json.dump(
                    {
                        'conversations': self.conversations,
                        'current_session': self.current_session,
                        'last_updated': datetime.now().isoformat(),
                    },
                    f,
                    indent=2,
                )
        except Exception:
            pass

    def start_session(self, session_id: Optional[str] = None) -> str:
        """Start a new conversation session

        Args:
            session_id: Optional session ID to use, otherwise generates one

        Returns:
            The session ID that was created or used
        """
        if session_id is None:
            session_id = f"session_{int(time.time())}"

        self.current_session = session_id
        if session_id not in self.conversations:
            self.conversations[session_id] = {
                'messages': [],
                'model': None,  # Will be set dynamically
                'created': datetime.now().isoformat(),
                'last_updated': datetime.now().isoformat(),
            }
        return session_id

    def add_message(self, role: str, content: str, model: Optional[str] = None) -> None:
        """Add a message to current session

        Args:
            role: Role of the message sender (e.g., 'user', 'assistant')
            content: Content of the message
            model: Optional model associated with this message
        """
        if self.current_session is None:
            self.start_session()

        message = {
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat(),
            'model': model,
        }

        self.conversations[self.current_session]['messages'].append(message)
        self.conversations[self.current_session][
            'last_updated'
        ] = datetime.now().isoformat()

        # Trim old messages if exceeding limit
        if len(self.conversations[self.current_session]['messages']) > self.max_items:
            self.conversations[self.current_session]['messages'] = self.conversations[
                self.current_session
            ]['messages'][-self.max_items :]

        self.save_memory()

    def get_context(self, max_messages: int = 10) -> List[Dict[str, Any]]:
        """Get recent conversation context for model input

        Args:
            max_messages: Maximum number of messages to return

        Returns:
            List of recent messages in the current session
        """
        if (
            self.current_session is None
            or self.current_session not in self.conversations
        ):
            return []

        messages = self.conversations[self.current_session]['messages']
        return messages[-max_messages:] if len(messages) > max_messages else messages

    def list_sessions(self) -> List[str]:
        """List all conversation sessions

        Returns:
            List of all session IDs
        """
        return list(self.conversations.keys())

    def switch_session(self, session_id: str) -> bool:
        """Switch to a different conversation session

        Args:
            session_id: Session ID to switch to

        Returns:
            True if the switch was successful, False otherwise
        """
        if session_id in self.conversations:
            self.current_session = session_id
            return True
        return False