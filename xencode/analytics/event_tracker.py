#!/usr/bin/env python3
"""
Event Tracker

Tracks user interactions, system events, and analytics data
for comprehensive usage analysis and insights.
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import threading


class EventCategory(str, Enum):
    """Categories of events that can be tracked"""
    USER_ACTION = "user_action"
    SYSTEM_EVENT = "system_event"
    PERFORMANCE = "performance"
    ERROR = "error"
    SECURITY = "security"
    AI_INTERACTION = "ai_interaction"
    PLUGIN_EVENT = "plugin_event"
    WORKSPACE_EVENT = "workspace_event"


class EventPriority(str, Enum):
    """Priority levels for events"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class EventContext:
    """Context information for events"""
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    workspace_id: Optional[str] = None
    plugin_id: Optional[str] = None
    request_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    platform: Optional[str] = None
    version: Optional[str] = None


@dataclass
class TrackedEvent:
    """Represents a tracked event with full context"""
    event_id: str
    event_type: str
    category: EventCategory
    timestamp: datetime
    context: EventContext
    properties: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    priority: EventPriority = EventPriority.MEDIUM
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization"""
        return {
            'event_id': self.event_id,
            'event_type': self.event_type,
            'category': self.category.value,
            'timestamp': self.timestamp.isoformat(),
            'context': {
                'user_id': self.context.user_id,
                'session_id': self.context.session_id,
                'workspace_id': self.context.workspace_id,
                'plugin_id': self.context.plugin_id,
                'request_id': self.context.request_id,
                'ip_address': self.context.ip_address,
                'user_agent': self.context.user_agent,
                'platform': self.context.platform,
                'version': self.context.version
            },
            'properties': self.properties,
            'metrics': self.metrics,
            'priority': self.priority.value,
            'tags': self.tags
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrackedEvent':
        """Create event from dictionary"""
        context = EventContext(**data.get('context', {}))
        
        return cls(
            event_id=data['event_id'],
            event_type=data['event_type'],
            category=EventCategory(data['category']),
            timestamp=datetime.fromisoformat(data['timestamp']),
            context=context,
            properties=data.get('properties', {}),
            metrics=data.get('metrics', {}),
            priority=EventPriority(data.get('priority', 'medium')),
            tags=data.get('tags', [])
        )


class EventFilter:
    """Filter for querying events"""
    
    def __init__(self,
                 event_types: Optional[List[str]] = None,
                 categories: Optional[List[EventCategory]] = None,
                 user_ids: Optional[List[str]] = None,
                 workspace_ids: Optional[List[str]] = None,
                 start_time: Optional[datetime] = None,
                 end_time: Optional[datetime] = None,
                 priorities: Optional[List[EventPriority]] = None,
                 tags: Optional[List[str]] = None):
        
        self.event_types = event_types
        self.categories = categories
        self.user_ids = user_ids
        self.workspace_ids = workspace_ids
        self.start_time = start_time
        self.end_time = end_time
        self.priorities = priorities
        self.tags = tags
    
    def matches(self, event: TrackedEvent) -> bool:
        """Check if event matches filter criteria"""
        
        if self.event_types and event.event_type not in self.event_types:
            return False
        
        if self.categories and event.category not in self.categories:
            return False
        
        if self.user_ids and event.context.user_id not in self.user_ids:
            return False
        
        if self.workspace_ids and event.context.workspace_id not in self.workspace_ids:
            return False
        
        if self.start_time and event.timestamp < self.start_time:
            return False
        
        if self.end_time and event.timestamp > self.end_time:
            return False
        
        if self.priorities and event.priority not in self.priorities:
            return False
        
        if self.tags and not any(tag in event.tags for tag in self.tags):
            return False
        
        return True


class EventTracker:
    """
    Comprehensive event tracking system for analytics and monitoring
    
    Tracks user interactions, system events, and performance metrics
    with real-time processing and storage capabilities.
    """
    
    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path.home() / ".xencode" / "analytics" / "events"
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Event storage
        self.events: List[TrackedEvent] = []
        self.event_callbacks: Dict[str, List[Callable[[TrackedEvent], None]]] = {}
        
        # Configuration
        self.max_events_in_memory = 10000
        self.batch_size = 100
        self.flush_interval = 30  # seconds
        self.max_event_age_days = 90
        
        # Session tracking
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.session_timeout = timedelta(hours=2)
        
        # Background tasks
        self._flush_task = None
        self._cleanup_task = None
        self._running = False
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Event statistics
        self.stats = {
            'total_events': 0,
            'events_by_category': {},
            'events_by_type': {},
            'unique_users': set(),
            'unique_sessions': set()
        }
    
    async def start(self) -> None:
        """Start the event tracker"""
        if self._running:
            return
        
        self._running = True
        
        # Start background tasks
        self._flush_task = asyncio.create_task(self._flush_loop())
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        print("EventTracker started")
    
    async def stop(self) -> None:
        """Stop the event tracker"""
        self._running = False
        
        # Cancel background tasks
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
        
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Flush remaining events
        await self._flush_events_to_disk()
        
        print("EventTracker stopped")
    
    def track_event(self,
                   event_type: str,
                   category: EventCategory,
                   context: Optional[EventContext] = None,
                   properties: Optional[Dict[str, Any]] = None,
                   metrics: Optional[Dict[str, float]] = None,
                   priority: EventPriority = EventPriority.MEDIUM,
                   tags: Optional[List[str]] = None) -> str:
        """Track a new event"""
        
        event_id = str(uuid.uuid4())
        event_context = context or EventContext()
        
        event = TrackedEvent(
            event_id=event_id,
            event_type=event_type,
            category=category,
            timestamp=datetime.now(),
            context=event_context,
            properties=properties or {},
            metrics=metrics or {},
            priority=priority,
            tags=tags or []
        )
        
        with self._lock:
            self.events.append(event)
            
            # Update statistics
            self.stats['total_events'] += 1
            
            category_key = category.value
            if category_key not in self.stats['events_by_category']:
                self.stats['events_by_category'][category_key] = 0
            self.stats['events_by_category'][category_key] += 1
            
            if event_type not in self.stats['events_by_type']:
                self.stats['events_by_type'][event_type] = 0
            self.stats['events_by_type'][event_type] += 1
            
            if event_context.user_id:
                self.stats['unique_users'].add(event_context.user_id)
            
            if event_context.session_id:
                self.stats['unique_sessions'].add(event_context.session_id)
            
            # Rotate events if too many
            if len(self.events) > self.max_events_in_memory:
                self.events = self.events[-self.max_events_in_memory//2:]
        
        # Call event callbacks
        self._call_event_callbacks(event)
        
        return event_id
    
    def track_user_action(self,
                         action: str,
                         user_id: str,
                         session_id: Optional[str] = None,
                         properties: Optional[Dict[str, Any]] = None,
                         metrics: Optional[Dict[str, float]] = None) -> str:
        """Track a user action event"""
        
        context = EventContext(
            user_id=user_id,
            session_id=session_id or self._get_or_create_session(user_id)
        )
        
        return self.track_event(
            event_type=f"user_action_{action}",
            category=EventCategory.USER_ACTION,
            context=context,
            properties=properties,
            metrics=metrics,
            tags=["user_action", action]
        )
    
    def track_system_event(self,
                          event: str,
                          component: str,
                          properties: Optional[Dict[str, Any]] = None,
                          metrics: Optional[Dict[str, float]] = None,
                          priority: EventPriority = EventPriority.MEDIUM) -> str:
        """Track a system event"""
        
        context = EventContext(version="3.0.0")  # Add system version
        
        return self.track_event(
            event_type=f"system_{event}",
            category=EventCategory.SYSTEM_EVENT,
            context=context,
            properties={**(properties or {}), 'component': component},
            metrics=metrics,
            priority=priority,
            tags=["system", component, event]
        )
    
    def track_performance_event(self,
                              operation: str,
                              duration_ms: float,
                              component: str,
                              success: bool = True,
                              properties: Optional[Dict[str, Any]] = None) -> str:
        """Track a performance event"""
        
        context = EventContext()
        
        metrics = {
            'duration_ms': duration_ms,
            'success': 1.0 if success else 0.0
        }
        
        return self.track_event(
            event_type=f"performance_{operation}",
            category=EventCategory.PERFORMANCE,
            context=context,
            properties={**(properties or {}), 'component': component, 'operation': operation},
            metrics=metrics,
            tags=["performance", component, operation]
        )
    
    def track_error_event(self,
                         error_type: str,
                         error_message: str,
                         component: str,
                         user_id: Optional[str] = None,
                         properties: Optional[Dict[str, Any]] = None) -> str:
        """Track an error event"""
        
        context = EventContext(user_id=user_id)
        
        return self.track_event(
            event_type=f"error_{error_type}",
            category=EventCategory.ERROR,
            context=context,
            properties={
                **(properties or {}),
                'component': component,
                'error_message': error_message,
                'error_type': error_type
            },
            priority=EventPriority.HIGH,
            tags=["error", component, error_type]
        )
    
    def track_ai_interaction(self,
                           model: str,
                           operation: str,
                           user_id: str,
                           duration_ms: float,
                           success: bool = True,
                           properties: Optional[Dict[str, Any]] = None) -> str:
        """Track an AI model interaction"""
        
        context = EventContext(user_id=user_id)
        
        metrics = {
            'duration_ms': duration_ms,
            'success': 1.0 if success else 0.0
        }
        
        return self.track_event(
            event_type=f"ai_{operation}",
            category=EventCategory.AI_INTERACTION,
            context=context,
            properties={**(properties or {}), 'model': model, 'operation': operation},
            metrics=metrics,
            tags=["ai", model, operation]
        )
    
    def add_event_callback(self, event_type: str, callback: Callable[[TrackedEvent], None]) -> None:
        """Add callback for specific event type"""
        if event_type not in self.event_callbacks:
            self.event_callbacks[event_type] = []
        self.event_callbacks[event_type].append(callback)
    
    def remove_event_callback(self, event_type: str, callback: Callable[[TrackedEvent], None]) -> None:
        """Remove event callback"""
        if event_type in self.event_callbacks:
            callbacks = self.event_callbacks[event_type]
            if callback in callbacks:
                callbacks.remove(callback)
    
    def _call_event_callbacks(self, event: TrackedEvent) -> None:
        """Call registered callbacks for event"""
        # Call specific event type callbacks
        if event.event_type in self.event_callbacks:
            for callback in self.event_callbacks[event.event_type]:
                try:
                    callback(event)
                except Exception as e:
                    print(f"Error in event callback: {e}")
        
        # Call wildcard callbacks
        if "*" in self.event_callbacks:
            for callback in self.event_callbacks["*"]:
                try:
                    callback(event)
                except Exception as e:
                    print(f"Error in wildcard event callback: {e}")
    
    def _get_or_create_session(self, user_id: str) -> str:
        """Get or create session for user"""
        now = datetime.now()
        
        # Clean up expired sessions
        expired_sessions = []
        for session_id, session_data in self.active_sessions.items():
            if now - session_data['last_activity'] > self.session_timeout:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            del self.active_sessions[session_id]
        
        # Find existing active session for user
        for session_id, session_data in self.active_sessions.items():
            if session_data['user_id'] == user_id:
                session_data['last_activity'] = now
                return session_id
        
        # Create new session
        session_id = str(uuid.uuid4())
        self.active_sessions[session_id] = {
            'user_id': user_id,
            'created_at': now,
            'last_activity': now
        }
        
        return session_id
    
    def query_events(self, event_filter: EventFilter, limit: int = 1000) -> List[TrackedEvent]:
        """Query events with filter"""
        with self._lock:
            matching_events = []
            
            for event in reversed(self.events):  # Most recent first
                if event_filter.matches(event):
                    matching_events.append(event)
                    if len(matching_events) >= limit:
                        break
            
            return matching_events
    
    def get_events_by_user(self, user_id: str, limit: int = 100) -> List[TrackedEvent]:
        """Get events for specific user"""
        event_filter = EventFilter(user_ids=[user_id])
        return self.query_events(event_filter, limit)
    
    def get_events_by_type(self, event_type: str, limit: int = 100) -> List[TrackedEvent]:
        """Get events by type"""
        event_filter = EventFilter(event_types=[event_type])
        return self.query_events(event_filter, limit)
    
    def get_recent_events(self, hours: int = 24, limit: int = 1000) -> List[TrackedEvent]:
        """Get recent events"""
        start_time = datetime.now() - timedelta(hours=hours)
        event_filter = EventFilter(start_time=start_time)
        return self.query_events(event_filter, limit)
    
    async def _flush_loop(self) -> None:
        """Background loop for flushing events to disk"""
        while self._running:
            try:
                await asyncio.sleep(self.flush_interval)
                await self._flush_events_to_disk()
            except Exception as e:
                print(f"Error in flush loop: {e}")
                await asyncio.sleep(5)
    
    async def _cleanup_loop(self) -> None:
        """Background loop for cleanup tasks"""
        while self._running:
            try:
                await asyncio.sleep(3600)  # Run every hour
                await self._cleanup_old_events()
            except Exception as e:
                print(f"Error in cleanup loop: {e}")
                await asyncio.sleep(300)
    
    async def _flush_events_to_disk(self) -> None:
        """Flush events to disk storage"""
        if not self.events:
            return
        
        try:
            # Get events to flush
            with self._lock:
                events_to_flush = self.events[-self.batch_size:] if len(self.events) > self.batch_size else self.events.copy()
            
            if not events_to_flush:
                return
            
            # Create filename with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H')
            events_file = self.storage_path / f"events_{timestamp}.jsonl"
            
            # Append events to file
            with open(events_file, 'a') as f:
                for event in events_to_flush:
                    f.write(json.dumps(event.to_dict()) + '\n')
            
        except Exception as e:
            print(f"Error flushing events to disk: {e}")
    
    async def _cleanup_old_events(self) -> None:
        """Clean up old events from memory and disk"""
        cutoff_date = datetime.now() - timedelta(days=self.max_event_age_days)
        
        # Clean up memory
        with self._lock:
            self.events = [event for event in self.events if event.timestamp > cutoff_date]
        
        # Clean up disk files
        try:
            for events_file in self.storage_path.glob("events_*.jsonl"):
                # Parse date from filename
                try:
                    date_str = events_file.stem.split('_')[1]  # events_YYYYMMDD_HH.jsonl
                    file_date = datetime.strptime(date_str[:8], '%Y%m%d')
                    
                    if file_date < cutoff_date:
                        events_file.unlink()
                        
                except (ValueError, IndexError):
                    # Skip files with invalid names
                    continue
                    
        except Exception as e:
            print(f"Error cleaning up old event files: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get event tracking statistics"""
        with self._lock:
            return {
                'total_events': self.stats['total_events'],
                'events_in_memory': len(self.events),
                'events_by_category': dict(self.stats['events_by_category']),
                'events_by_type': dict(self.stats['events_by_type']),
                'unique_users': len(self.stats['unique_users']),
                'unique_sessions': len(self.stats['unique_sessions']),
                'active_sessions': len(self.active_sessions),
                'storage_path': str(self.storage_path),
                'running': self._running
            }


# Global event tracker instance
event_tracker = EventTracker()