#!/usr/bin/env python3
"""
User Feedback and Analytics System for Xencode

Implements user-centric development framework with feedback collection,
user journey tracking, and satisfaction metrics.
"""

import asyncio
import json
import sqlite3
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class FeedbackType(Enum):
    """Types of user feedback"""
    BUG_REPORT = "bug_report"
    FEATURE_REQUEST = "feature_request"
    SATISFACTION = "satisfaction"
    USABILITY = "usability"
    PERFORMANCE = "performance"
    GENERAL = "general"


class UserJourneyEvent(Enum):
    """User journey tracking events"""
    FIRST_LAUNCH = "first_launch"
    MODEL_SELECTION = "model_selection"
    FIRST_QUERY = "first_query"
    PLUGIN_INSTALL = "plugin_install"
    CONFIG_CHANGE = "config_change"
    ERROR_ENCOUNTERED = "error_encountered"
    FEATURE_DISCOVERY = "feature_discovery"
    SESSION_END = "session_end"


@dataclass
class UserFeedback:
    """User feedback data structure"""
    id: str
    user_id: str
    feedback_type: FeedbackType
    rating: Optional[int]  # 1-5 scale
    message: str
    context: Dict[str, Any]
    timestamp: datetime
    resolved: bool = False
    resolution_notes: Optional[str] = None


@dataclass
class UserJourneyStep:
    """User journey step tracking"""
    id: str
    user_id: str
    event: UserJourneyEvent
    context: Dict[str, Any]
    timestamp: datetime
    session_id: str
    duration_ms: Optional[int] = None


@dataclass
class UserSatisfactionMetrics:
    """User satisfaction metrics"""
    user_id: str
    nps_score: Optional[int]  # Net Promoter Score (-100 to 100)
    satisfaction_rating: Optional[float]  # 1-5 scale
    feature_adoption_rate: float
    session_frequency: float
    avg_session_duration: float
    last_active: datetime
    total_sessions: int


class UserFeedbackManager:
    """Manages user feedback collection and analysis"""
    
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or Path.home() / ".xencode" / "user_feedback.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
        
    def _init_database(self):
        """Initialize the feedback database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    feedback_type TEXT NOT NULL,
                    rating INTEGER,
                    message TEXT NOT NULL,
                    context TEXT,
                    timestamp TEXT NOT NULL,
                    resolved BOOLEAN DEFAULT FALSE,
                    resolution_notes TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_journey (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    event TEXT NOT NULL,
                    context TEXT,
                    timestamp TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    duration_ms INTEGER
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_metrics (
                    user_id TEXT PRIMARY KEY,
                    nps_score INTEGER,
                    satisfaction_rating REAL,
                    feature_adoption_rate REAL,
                    session_frequency REAL,
                    avg_session_duration REAL,
                    last_active TEXT,
                    total_sessions INTEGER
                )
            """)
            
            conn.commit()
    
    async def collect_feedback(
        self,
        user_id: str,
        feedback_type: FeedbackType,
        message: str,
        rating: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Collect user feedback"""
        feedback = UserFeedback(
            id=str(uuid.uuid4()),
            user_id=user_id,
            feedback_type=feedback_type,
            rating=rating,
            message=message,
            context=context or {},
            timestamp=datetime.now()
        )
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO feedback 
                (id, user_id, feedback_type, rating, message, context, timestamp, resolved)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                feedback.id,
                feedback.user_id,
                feedback.feedback_type.value,
                feedback.rating,
                feedback.message,
                json.dumps(feedback.context),
                feedback.timestamp.isoformat(),
                feedback.resolved
            ))
            conn.commit()
        
        logger.info(f"Collected feedback: {feedback.feedback_type.value} from user {user_id}")
        return feedback.id
    
    async def track_user_journey(
        self,
        user_id: str,
        event: UserJourneyEvent,
        session_id: str,
        context: Optional[Dict[str, Any]] = None,
        duration_ms: Optional[int] = None
    ) -> str:
        """Track user journey events"""
        journey_step = UserJourneyStep(
            id=str(uuid.uuid4()),
            user_id=user_id,
            event=event,
            context=context or {},
            timestamp=datetime.now(),
            session_id=session_id,
            duration_ms=duration_ms
        )
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO user_journey 
                (id, user_id, event, context, timestamp, session_id, duration_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                journey_step.id,
                journey_step.user_id,
                journey_step.event.value,
                json.dumps(journey_step.context),
                journey_step.timestamp.isoformat(),
                journey_step.session_id,
                journey_step.duration_ms
            ))
            conn.commit()
        
        return journey_step.id
    
    async def calculate_user_satisfaction(self, user_id: str) -> UserSatisfactionMetrics:
        """Calculate comprehensive user satisfaction metrics"""
        with sqlite3.connect(self.db_path) as conn:
            # Get satisfaction ratings
            satisfaction_data = conn.execute("""
                SELECT AVG(rating), COUNT(*) FROM feedback 
                WHERE user_id = ? AND feedback_type = ? AND rating IS NOT NULL
            """, (user_id, FeedbackType.SATISFACTION.value)).fetchone()
            
            # Get NPS score (latest)
            nps_data = conn.execute("""
                SELECT rating FROM feedback 
                WHERE user_id = ? AND feedback_type = ? AND rating IS NOT NULL
                ORDER BY timestamp DESC LIMIT 1
            """, (user_id, FeedbackType.SATISFACTION.value)).fetchone()
            
            # Calculate session metrics
            session_data = conn.execute("""
                SELECT COUNT(DISTINCT session_id), 
                       AVG(duration_ms),
                       MAX(timestamp),
                       COUNT(*)
                FROM user_journey 
                WHERE user_id = ?
            """, (user_id,)).fetchone()
            
            # Calculate feature adoption rate
            total_features = len(UserJourneyEvent)
            used_features = conn.execute("""
                SELECT COUNT(DISTINCT event) FROM user_journey WHERE user_id = ?
            """, (user_id,)).fetchone()[0]
            
            feature_adoption_rate = used_features / total_features if total_features > 0 else 0
            
            # Calculate session frequency (sessions per week)
            first_session = conn.execute("""
                SELECT MIN(timestamp) FROM user_journey WHERE user_id = ?
            """, (user_id,)).fetchone()[0]
            
            if first_session:
                first_date = datetime.fromisoformat(first_session)
                weeks_active = max(1, (datetime.now() - first_date).days / 7)
                session_frequency = (session_data[0] or 0) / weeks_active
            else:
                session_frequency = 0
            
            # Convert NPS rating (1-5) to NPS score (-100 to 100)
            nps_score = None
            if nps_data and nps_data[0]:
                # Convert 1-5 rating to NPS: 1-2=Detractor(-100), 3=Passive(0), 4-5=Promoter(+100)
                rating = nps_data[0]
                if rating <= 2:
                    nps_score = -100
                elif rating == 3:
                    nps_score = 0
                else:
                    nps_score = 100
            
            return UserSatisfactionMetrics(
                user_id=user_id,
                nps_score=nps_score,
                satisfaction_rating=satisfaction_data[0] if satisfaction_data[0] else None,
                feature_adoption_rate=feature_adoption_rate,
                session_frequency=session_frequency,
                avg_session_duration=(session_data[1] or 0) / 1000,  # Convert to seconds
                last_active=datetime.fromisoformat(session_data[2]) if session_data[2] else datetime.now(),
                total_sessions=session_data[0] or 0
            )
    
    async def get_feedback_summary(self, days: int = 30) -> Dict[str, Any]:
        """Get feedback summary for the last N days"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        with sqlite3.connect(self.db_path) as conn:
            # Feedback by type
            feedback_by_type = {}
            for feedback_type in FeedbackType:
                count = conn.execute("""
                    SELECT COUNT(*) FROM feedback 
                    WHERE feedback_type = ? AND timestamp > ?
                """, (feedback_type.value, cutoff_date.isoformat())).fetchone()[0]
                feedback_by_type[feedback_type.value] = count
            
            # Average ratings
            avg_ratings = {}
            for feedback_type in FeedbackType:
                avg_rating = conn.execute("""
                    SELECT AVG(rating) FROM feedback 
                    WHERE feedback_type = ? AND rating IS NOT NULL AND timestamp > ?
                """, (feedback_type.value, cutoff_date.isoformat())).fetchone()[0]
                avg_ratings[feedback_type.value] = avg_rating
            
            # Top issues (unresolved feedback)
            top_issues = conn.execute("""
                SELECT message, COUNT(*) as count FROM feedback 
                WHERE resolved = FALSE AND timestamp > ?
                GROUP BY message 
                ORDER BY count DESC 
                LIMIT 10
            """, (cutoff_date.isoformat(),)).fetchall()
            
            return {
                "period_days": days,
                "feedback_by_type": feedback_by_type,
                "average_ratings": avg_ratings,
                "top_unresolved_issues": [{"message": issue[0], "count": issue[1]} for issue in top_issues],
                "total_feedback": sum(feedback_by_type.values())
            }


class UserPersonaManager:
    """Manages user personas and segmentation"""
    
    def __init__(self, feedback_manager: UserFeedbackManager):
        self.feedback_manager = feedback_manager
    
    async def identify_user_persona(self, user_id: str) -> str:
        """Identify user persona based on behavior patterns"""
        metrics = await self.feedback_manager.calculate_user_satisfaction(user_id)
        
        # Simple persona classification based on usage patterns
        if metrics.session_frequency > 5 and metrics.feature_adoption_rate > 0.7:
            return "power_user"
        elif metrics.session_frequency > 2 and metrics.avg_session_duration > 300:
            return "regular_user"
        elif metrics.total_sessions < 5:
            return "new_user"
        elif metrics.session_frequency < 1:
            return "inactive_user"
        else:
            return "casual_user"
    
    async def get_persona_insights(self) -> Dict[str, Any]:
        """Get insights about user personas"""
        # This would be implemented with more sophisticated analysis
        # For now, return a placeholder structure
        return {
            "personas": {
                "power_user": {"count": 0, "satisfaction": 0, "retention": 0},
                "regular_user": {"count": 0, "satisfaction": 0, "retention": 0},
                "casual_user": {"count": 0, "satisfaction": 0, "retention": 0},
                "new_user": {"count": 0, "satisfaction": 0, "retention": 0},
                "inactive_user": {"count": 0, "satisfaction": 0, "retention": 0}
            },
            "recommendations": []
        }


# Global feedback manager instance
_feedback_manager: Optional[UserFeedbackManager] = None


def get_feedback_manager() -> UserFeedbackManager:
    """Get the global feedback manager instance"""
    global _feedback_manager
    if _feedback_manager is None:
        _feedback_manager = UserFeedbackManager()
    return _feedback_manager


async def collect_user_feedback(
    user_id: str,
    feedback_type: FeedbackType,
    message: str,
    rating: Optional[int] = None,
    context: Optional[Dict[str, Any]] = None
) -> str:
    """Convenience function to collect user feedback"""
    manager = get_feedback_manager()
    return await manager.collect_feedback(user_id, feedback_type, message, rating, context)


async def track_user_event(
    user_id: str,
    event: UserJourneyEvent,
    session_id: str,
    context: Optional[Dict[str, Any]] = None,
    duration_ms: Optional[int] = None
) -> str:
    """Convenience function to track user journey events"""
    manager = get_feedback_manager()
    return await manager.track_user_journey(user_id, event, session_id, context, duration_ms)