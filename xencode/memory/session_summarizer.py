#!/usr/bin/env python3
"""
Long-Session Memory Summarizer

Summarizes and pins context for long multi-turn sessions to prevent context overflow
and maintain conversation continuity.

Features:
- Automatic conversation summarization
- Context pinning for important information
- Sliding window memory management
- Hierarchical summarization (turn-level, section-level, session-level)
- Memory consolidation with importance scoring
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import deque

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

console = Console()


class MemoryType(Enum):
    """Types of memory entries"""
    TURN = "turn"  # Single turn pair (user + assistant)
    SECTION = "section"  # Grouped turns by topic
    SESSION = "session"  # Overall session summary
    PINNED = "pinned"  # Important information to retain
    ACTION = "action"  # Executed actions (file edits, commands, etc.)
    ERROR = "error"  # Errors encountered and fixes


class ImportanceLevel(Enum):
    """Importance levels for memory entries"""
    CRITICAL = 5  # Must retain (user preferences, key decisions)
    HIGH = 4  # Very important (major code changes, architecture)
    MEDIUM = 3  # Moderately important (significant discussions)
    LOW = 2  # Low importance (minor details)
    TRIVIAL = 1  # Can be summarized/discarded


@dataclass
class Turn:
    """Represents a single turn in the conversation"""
    id: str
    timestamp: datetime
    user_message: str
    assistant_message: str
    model_used: str
    tokens_used: int
    importance: ImportanceLevel = ImportanceLevel.MEDIUM
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'user_message': self.user_message,
            'assistant_message': self.assistant_message,
            'model_used': self.model_used,
            'tokens_used': self.tokens_used,
            'importance': self.importance.value,
            'tags': self.tags,
            'metadata': self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Turn':
        return cls(
            id=data['id'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            user_message=data['user_message'],
            assistant_message=data['assistant_message'],
            model_used=data['model_used'],
            tokens_used=data['tokens_used'],
            importance=ImportanceLevel(data.get('importance', 3)),
            tags=data.get('tags', []),
            metadata=data.get('metadata', {}),
        )


@dataclass
class Section:
    """Represents a section of conversation (grouped turns)"""
    id: str
    topic: str
    start_time: datetime
    end_time: datetime
    turn_ids: List[str]
    summary: str
    key_points: List[str] = field(default_factory=list)
    decisions: List[str] = field(default_factory=list)
    actions: List[Dict[str, Any]] = field(default_factory=list)
    importance: ImportanceLevel = ImportanceLevel.MEDIUM

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'topic': self.topic,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat(),
            'turn_ids': self.turn_ids,
            'summary': self.summary,
            'key_points': self.key_points,
            'decisions': self.decisions,
            'actions': self.actions,
            'importance': self.importance.value,
        }


@dataclass
class PinnedMemory:
    """Important information that should always be retained"""
    id: str
    content: str
    category: str  # user_preference, project_info, key_decision, etc.
    created_at: datetime
    expires_at: Optional[datetime] = None
    references: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'content': self.content,
            'category': self.category,
            'created_at': self.created_at.isoformat(),
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'references': self.references,
        }


@dataclass
class SessionSummary:
    """Overall session summary"""
    session_id: str
    start_time: datetime
    end_time: datetime
    total_turns: int
    total_tokens: int
    summary: str
    objectives: List[str]
    achievements: List[str]
    challenges: List[str]
    key_decisions: List[str]
    files_modified: List[str]
    commands_executed: List[str]
    errors_encountered: List[str]
    section_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'session_id': self.session_id,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat(),
            'total_turns': self.total_turns,
            'total_tokens': self.total_tokens,
            'summary': self.summary,
            'objectives': self.objectives,
            'achievements': self.achievements,
            'challenges': self.challenges,
            'key_decisions': self.key_decisions,
            'files_modified': self.files_modified,
            'commands_executed': self.commands_executed,
            'errors_encountered': self.errors_encountered,
            'section_ids': self.section_ids,
        }


class MemorySummarizer:
    """
    Summarizes long conversation sessions to maintain context within limits

    Features:
    - Hierarchical summarization (turn → section → session)
    - Importance-based retention
    - Automatic consolidation
    - Context pinning for critical information
    """

    # Default thresholds
    DEFAULT_CONTEXT_WINDOW_TOKENS = 128000  # Typical LLM context window
    DEFAULT_SUMMARY_THRESHOLD_TURNS = 10  # Summarize after N turns
    DEFAULT_SECTION_MAX_TURNS = 20  # Max turns per section
    DEFAULT_PINNED_MEMORY_LIMIT = 50  # Max pinned items

    def __init__(
        self,
        session_id: str,
        context_window_tokens: int = DEFAULT_CONTEXT_WINDOW_TOKENS,
        summary_threshold: int = DEFAULT_SUMMARY_THRESHOLD_TURNS,
        llm_client: Optional[Any] = None,
    ):
        """
        Initialize memory summarizer

        Args:
            session_id: Unique session identifier
            context_window_tokens: Maximum tokens for context window
            summary_threshold: Number of turns before triggering summary
            llm_client: Optional LLM client for generating summaries
        """
        self.session_id = session_id
        self.context_window_tokens = context_window_tokens
        self.summary_threshold = summary_threshold

        # Memory storage
        self.turns: deque[Turn] = deque()
        self.sections: Dict[str, Section] = {}
        self.pinned_memories: Dict[str, PinnedMemory] = {}
        self.session_summary: Optional[SessionSummary] = None

        # Tracking
        self.total_tokens = 0
        self.current_section_turns: List[str] = []
        self.current_section_topic: Optional[str] = None

        # LLM client for summarization
        self.llm_client = llm_client

        # Topic detection (simple keyword-based)
        self.topic_keywords = {
            'code_generation': ['write', 'create', 'implement', 'function', 'class', 'code'],
            'debugging': ['error', 'bug', 'fix', 'issue', 'problem', 'exception'],
            'review': ['review', 'analyze', 'check', 'improve', 'optimize'],
            'explanation': ['explain', 'how does', 'what is', 'understand'],
            'refactoring': ['refactor', 'restructure', 'clean', 'organize'],
            'testing': ['test', 'unit test', 'integration', 'coverage'],
            'documentation': ['document', 'comment', 'docstring', 'readme'],
        }

    def add_turn(
        self,
        user_message: str,
        assistant_message: str,
        model_used: str = "unknown",
        tokens_used: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Add a new turn to the conversation

        Args:
            user_message: User's message
            assistant_message: Assistant's response
            model_used: Model that generated the response
            tokens_used: Tokens consumed by this turn
            metadata: Additional metadata

        Returns:
            Turn ID
        """
        turn_id = f"turn_{len(self.turns) + 1}"

        # Detect topic and importance
        topic = self._detect_topic(user_message)
        importance = self._assess_importance(user_message, assistant_message, metadata)

        # Extract tags
        tags = self._extract_tags(user_message, assistant_message)

        turn = Turn(
            id=turn_id,
            timestamp=datetime.now(),
            user_message=user_message,
            assistant_message=assistant_message,
            model_used=model_used,
            tokens_used=tokens_used,
            importance=importance,
            tags=tags,
            metadata=metadata or {},
        )

        self.turns.append(turn)
        self.total_tokens += tokens_used

        # Check if we need to summarize
        if len(self.turns) >= self.summary_threshold:
            self._consolidate_memory()

        # Check section boundary
        if self.current_section_topic and topic != self.current_section_topic:
            self._finalize_section()

        self.current_section_turns.append(turn_id)
        self.current_section_topic = topic

        return turn_id

    def pin_memory(
        self,
        content: str,
        category: str = "general",
        expires_at: Optional[datetime] = None,
    ) -> str:
        """
        Pin important information to always retain

        Args:
            content: Content to pin
            category: Category of pinned item
            expires_at: Optional expiration time

        Returns:
            Pinned memory ID
        """
        if len(self.pinned_memories) >= self.DEFAULT_PINNED_MEMORY_LIMIT:
            # Remove oldest pinned item
            oldest_id = min(
                self.pinned_memories.keys(),
                key=lambda k: self.pinned_memories[k].created_at,
            )
            del self.pinned_memories[oldest_id]

        pin_id = f"pin_{len(self.pinned_memories) + 1}"
        pinned = PinnedMemory(
            id=pin_id,
            content=content,
            category=category,
            created_at=datetime.now(),
            expires_at=expires_at,
        )

        self.pinned_memories[pin_id] = pinned
        return pin_id

    def record_action(
        self,
        action_type: str,
        details: Dict[str, Any],
        success: bool = True,
    ):
        """Record an action taken during the conversation"""
        if not self.current_section_turns:
            return

        latest_turn_id = self.current_section_turns[-1]
        if latest_turn_id in self.sections:
            section = self.sections[latest_turn_id]
            section.actions.append({
                'type': action_type,
                'details': details,
                'success': success,
                'timestamp': datetime.now().isoformat(),
            })

    def _detect_topic(self, text: str) -> str:
        """Detect topic from text using keyword matching"""
        text_lower = text.lower()
        topic_scores = {}

        for topic, keywords in self.topic_keywords.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            topic_scores[topic] = score

        if not topic_scores or max(topic_scores.values()) == 0:
            return "general"

        return max(topic_scores, key=topic_scores.get)

    def _assess_importance(
        self,
        user_message: str,
        assistant_message: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ImportanceLevel:
        """Assess importance of a turn"""
        # Check for critical indicators
        critical_keywords = [
            'architecture', 'design decision', 'important', 'critical',
            'must', 'should', 'requirement', 'constraint',
        ]

        combined = f"{user_message} {assistant_message}".lower()

        if any(kw in combined for kw in critical_keywords):
            return ImportanceLevel.HIGH

        # Check metadata for importance signals
        if metadata:
            if metadata.get('is_decision'):
                return ImportanceLevel.CRITICAL
            if metadata.get('files_modified'):
                return ImportanceLevel.HIGH
            if metadata.get('error_occurred'):
                return ImportanceLevel.MEDIUM

        # Default to medium
        return ImportanceLevel.MEDIUM

    def _extract_tags(self, user_message: str, assistant_message: str) -> List[str]:
        """Extract tags from conversation turn"""
        tags = []

        # Extract programming language mentions
        languages = ['python', 'javascript', 'typescript', 'java', 'rust', 'go', 'cpp']
        combined = f"{user_message} {assistant_message}".lower()

        for lang in languages:
            if lang in combined:
                tags.append(f"lang:{lang}")

        # Extract file references
        import re
        file_pattern = r'[\w\-]+\.(py|js|ts|java|rs|go|cpp|md|json|yaml|yml)'
        files = re.findall(file_pattern, combined, re.IGNORECASE)
        tags.extend([f"file:{f}" for f in set(files)])

        return list(set(tags))

    def _finalize_section(self):
        """Finalize current section and create summary"""
        if not self.current_section_turns:
            return

        section_id = f"section_{len(self.sections) + 1}"

        # Get turns in section
        section_turns = [
            turn for turn in self.turns
            if turn.id in self.current_section_turns
        ]

        # Generate summary
        summary = self._generate_section_summary(section_turns)
        key_points = self._extract_key_points(section_turns)
        decisions = self._extract_decisions(section_turns)

        section = Section(
            id=section_id,
            topic=self.current_section_topic or "general",
            start_time=section_turns[0].timestamp,
            end_time=section_turns[-1].timestamp,
            turn_ids=self.current_section_turns.copy(),
            summary=summary,
            key_points=key_points,
            decisions=decisions,
        )

        self.sections[section_id] = section

        # Clear current section
        self.current_section_turns = []
        self.current_section_topic = None

    def _generate_section_summary(self, turns: List[Turn]) -> str:
        """Generate summary for a section of turns"""
        if not turns:
            return ""

        # Simple extractive summary
        key_messages = []
        for turn in turns[:3]:  # First 3 turns
            key_messages.append(f"User: {turn.user_message[:100]}")
            key_messages.append(f"Assistant: {turn.assistant_message[:200]}")

        return "\n".join(key_messages)

    def _extract_key_points(self, turns: List[Turn]) -> List[str]:
        """Extract key points from turns"""
        key_points = []

        for turn in turns:
            # Look for important statements
            if turn.importance.value >= ImportanceLevel.HIGH.value:
                key_points.append(turn.assistant_message[:150])

        return key_points[:5]  # Top 5 key points

    def _extract_decisions(self, turns: List[Turn]) -> List[str]:
        """Extract decisions made in turns"""
        decisions = []

        for turn in turns:
            if turn.metadata.get('is_decision'):
                decisions.append(turn.assistant_message[:150])

        return decisions

    def _consolidate_memory(self):
        """Consolidate memory by summarizing old turns"""
        # Keep last N turns in detail
        keep_turns = self.summary_threshold // 2

        # Finalize current section if needed
        if self.current_section_turns:
            self._finalize_section()

        # Remove old turns (they're summarized in sections)
        while len(self.turns) > keep_turns:
            self.turns.popleft()

    def get_context(self, max_tokens: Optional[int] = None) -> Dict[str, Any]:
        """
        Get current context for LLM

        Args:
            max_tokens: Maximum tokens to return

        Returns:
            Context dictionary with pinned memories, recent turns, and summaries
        """
        max_tokens = max_tokens or self.context_window_tokens

        # Start with pinned memories (always included)
        context = {
            'pinned_memories': [
                pm.to_dict() for pm in self.pinned_memories.values()
            ],
            'section_summaries': [
                section.to_dict() for section in self.sections.values()
            ],
            'recent_turns': [],
            'session_summary': None,
        }

        # Add recent turns
        tokens_used = sum(pm.content.count(' ') * 1.3 for pm in self.pinned_memories.values())

        for turn in reversed(self.turns):
            turn_tokens = turn.tokens_used
            if tokens_used + turn_tokens > max_tokens * 0.8:  # Leave room for new response
                break

            context['recent_turns'].insert(0, turn.to_dict())
            tokens_used += turn_tokens

        # Add session summary if exists
        if self.session_summary:
            context['session_summary'] = self.session_summary.to_dict()

        return context

    def generate_session_summary(self) -> SessionSummary:
        """Generate comprehensive session summary"""
        all_turns = list(self.turns) + [
            turn for section in self.sections.values()
            for turn_id in section.turn_ids
            for turn in self.turns if turn.id == turn_id
        ]

        # Extract key information
        objectives = []
        achievements = []
        challenges = []
        key_decisions = []
        files_modified = set()
        commands_executed = []
        errors_encountered = []

        for turn in all_turns:
            # Extract from metadata
            if 'objectives' in turn.metadata:
                objectives.extend(turn.metadata['objectives'])
            if 'achievements' in turn.metadata:
                achievements.extend(turn.metadata['achievements'])
            if 'files_modified' in turn.metadata:
                files_modified.update(turn.metadata['files_modified'])
            if 'commands_executed' in turn.metadata:
                commands_executed.extend(turn.metadata['commands_executed'])
            if 'errors' in turn.metadata:
                errors_encountered.extend(turn.metadata['errors'])

            # Extract from content
            if turn.importance == ImportanceLevel.CRITICAL:
                key_decisions.append(turn.assistant_message[:150])

        # Generate summary text
        summary_parts = []
        if objectives:
            summary_parts.append(f"Objectives: {', '.join(objectives)}")
        if achievements:
            summary_parts.append(f"Achievements: {', '.join(achievements)}")
        if files_modified:
            summary_parts.append(f"Files modified: {', '.join(files_modified)}")

        self.session_summary = SessionSummary(
            session_id=self.session_id,
            start_time=all_turns[0].timestamp if all_turns else datetime.now(),
            end_time=datetime.now(),
            total_turns=len(self.turns) + len(self.sections),
            total_tokens=self.total_tokens,
            summary=". ".join(summary_parts) if summary_parts else "Ongoing conversation",
            objectives=list(set(objectives)),
            achievements=list(set(achievements)),
            challenges=list(set(challenges)),
            key_decisions=list(set(key_decisions))[:10],
            files_modified=list(files_modified),
            commands_executed=list(set(commands_executed)),
            errors_encountered=list(set(errors_encountered)),
            section_ids=list(self.sections.keys()),
        )

        return self.session_summary

    def export_session(self, output_path: Path) -> Dict[str, Any]:
        """Export complete session to file"""
        if not self.session_summary:
            self.generate_session_summary()

        export_data = {
            'session_id': self.session_id,
            'exported_at': datetime.now().isoformat(),
            'summary': self.session_summary.to_dict(),
            'sections': [s.to_dict() for s in self.sections.values()],
            'recent_turns': [t.to_dict() for t in self.turns],
            'pinned_memories': [pm.to_dict() for pm in self.pinned_memories.values()],
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        return export_data

    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        return {
            'total_turns': len(self.turns),
            'total_sections': len(self.sections),
            'pinned_memories': len(self.pinned_memories),
            'total_tokens': self.total_tokens,
            'session_summary_exists': self.session_summary is not None,
            'context_window_usage': f"{(self.total_tokens / self.context_window_tokens * 100):.1f}%",
        }


# Global session manager
_sessions: Dict[str, MemorySummarizer] = {}


def get_session_summarizer(session_id: str, **kwargs) -> MemorySummarizer:
    """Get or create session summarizer"""
    if session_id not in _sessions:
        _sessions[session_id] = MemorySummarizer(session_id, **kwargs)
    return _sessions[session_id]


def export_all_sessions(output_dir: Path) -> List[Path]:
    """Export all sessions to directory"""
    output_dir.mkdir(parents=True, exist_ok=True)
    exported = []

    for session_id, summarizer in _sessions.items():
        output_path = output_dir / f"{session_id}.json"
        summarizer.export_session(output_path)
        exported.append(output_path)

    return exported


if __name__ == "__main__":
    # Demo
    console.print("[bold blue]Long-Session Memory Summarizer Demo[/bold blue]\n")

    summarizer = MemorySummarizer(session_id="demo_session")

    # Add some turns
    test_turns = [
        ("Write a Python function to sort a list", "Here's a simple sorting function..."),
        ("Now add type hints", "Here's the function with type hints..."),
        ("What's the time complexity?", "The time complexity is O(n log n)..."),
        ("Can you optimize it?", "Here's an optimized version..."),
    ]

    for user_msg, assistant_msg in test_turns:
        summarizer.add_turn(user_msg, assistant_msg, tokens_used=100)

    # Pin important info
    summarizer.pin_memory(
        "User prefers Python 3.10+ with type hints",
        category="user_preference",
    )

    # Get stats
    stats = summarizer.get_stats()
    console.print(f"\n[bold]Session Stats:[/bold]")
    for key, value in stats.items():
        console.print(f"  {key}: {value}")
