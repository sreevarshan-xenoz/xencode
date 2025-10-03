#!/usr/bin/env python3
"""
Enhancement Systems Integration for Xencode Core

Integrates the User Feedback System, Technical Debt Manager, and AI Ethics Framework
with the main Xencode application for seamless operation.
"""

import asyncio
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path
import logging

from .user_feedback_system import (
    get_feedback_manager, FeedbackType, UserJourneyEvent,
    collect_user_feedback, track_user_event
)
from .technical_debt_manager import get_debt_manager
from .ai_ethics_framework import get_ethics_framework, analyze_ai_interaction

logger = logging.getLogger(__name__)


class EnhancementSystemsIntegration:
    """Integration layer for enhancement systems with Xencode core"""
    
    def __init__(self, user_id: Optional[str] = None):
        self.user_id = user_id or self._generate_user_id()
        self.session_id = f"session_{int(datetime.now().timestamp())}"
        
        # Initialize managers
        self.feedback_manager = get_feedback_manager()
        self.debt_manager = get_debt_manager()
        self.ethics_framework = get_ethics_framework()
        
        # Track session start
        asyncio.create_task(self._track_session_start())
    
    def _generate_user_id(self) -> str:
        """Generate a unique user ID"""
        return f"user_{uuid.uuid4().hex[:8]}"
    
    async def _track_session_start(self):
        """Track the start of a user session"""
        try:
            await track_user_event(
                self.user_id,
                UserJourneyEvent.FIRST_LAUNCH,
                self.session_id,
                context={"timestamp": datetime.now().isoformat()}
            )
        except Exception as e:
            logger.warning(f"Failed to track session start: {e}")
    
    async def track_model_selection(self, model_name: str, selection_method: str = "auto"):
        """Track when user selects an AI model"""
        try:
            await track_user_event(
                self.user_id,
                UserJourneyEvent.MODEL_SELECTION,
                self.session_id,
                context={
                    "model_name": model_name,
                    "selection_method": selection_method,
                    "timestamp": datetime.now().isoformat()
                }
            )
        except Exception as e:
            logger.warning(f"Failed to track model selection: {e}")
    
    async def track_user_query(self, query: str, response_time_ms: Optional[int] = None):
        """Track user query and response time"""
        try:
            await track_user_event(
                self.user_id,
                UserJourneyEvent.FIRST_QUERY,
                self.session_id,
                context={
                    "query_length": len(query),
                    "response_time_ms": response_time_ms,
                    "timestamp": datetime.now().isoformat()
                },
                duration_ms=response_time_ms
            )
        except Exception as e:
            logger.warning(f"Failed to track user query: {e}")
    
    async def analyze_ai_response(
        self, 
        user_query: str, 
        ai_response: str,
        context: Optional[Dict[str, Any]] = None
    ) -> List[Any]:
        """Analyze AI response for ethics violations"""
        try:
            violations = await analyze_ai_interaction(
                user_query,
                ai_response,
                context=context or {}
            )
            
            if violations:
                logger.warning(f"Ethics violations detected: {len(violations)}")
                # Track error event if violations found
                await track_user_event(
                    self.user_id,
                    UserJourneyEvent.ERROR_ENCOUNTERED,
                    self.session_id,
                    context={
                        "error_type": "ethics_violation",
                        "violation_count": len(violations),
                        "timestamp": datetime.now().isoformat()
                    }
                )
            
            return violations
        except Exception as e:
            logger.error(f"Failed to analyze AI response: {e}")
            return []
    
    async def collect_user_feedback_on_response(
        self,
        rating: int,
        feedback_message: str = "",
        context: Optional[Dict[str, Any]] = None
    ):
        """Collect user feedback on AI response"""
        try:
            await collect_user_feedback(
                self.user_id,
                FeedbackType.SATISFACTION,
                feedback_message or f"Rating: {rating}/5",
                rating=rating,
                context=context or {}
            )
        except Exception as e:
            logger.warning(f"Failed to collect user feedback: {e}")
    
    async def report_error(self, error_message: str, error_context: Optional[Dict[str, Any]] = None):
        """Report an error encountered by the user"""
        try:
            # Track error event
            await track_user_event(
                self.user_id,
                UserJourneyEvent.ERROR_ENCOUNTERED,
                self.session_id,
                context={
                    "error_message": error_message,
                    "error_context": error_context or {},
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            # Collect error feedback
            await collect_user_feedback(
                self.user_id,
                FeedbackType.BUG_REPORT,
                error_message,
                rating=1,  # Low rating for errors
                context=error_context or {}
            )
        except Exception as e:
            logger.warning(f"Failed to report error: {e}")
    
    async def track_feature_usage(self, feature_name: str, usage_context: Optional[Dict[str, Any]] = None):
        """Track usage of specific features"""
        try:
            await track_user_event(
                self.user_id,
                UserJourneyEvent.FEATURE_DISCOVERY,
                self.session_id,
                context={
                    "feature_name": feature_name,
                    "usage_context": usage_context or {},
                    "timestamp": datetime.now().isoformat()
                }
            )
        except Exception as e:
            logger.warning(f"Failed to track feature usage: {e}")
    
    async def end_session(self):
        """Track the end of user session"""
        try:
            await track_user_event(
                self.user_id,
                UserJourneyEvent.SESSION_END,
                self.session_id,
                context={"timestamp": datetime.now().isoformat()}
            )
        except Exception as e:
            logger.warning(f"Failed to track session end: {e}")
    
    async def get_user_insights(self) -> Dict[str, Any]:
        """Get comprehensive user insights"""
        try:
            # Get user satisfaction metrics
            user_metrics = await self.feedback_manager.calculate_user_satisfaction(self.user_id)
            
            # Get recent feedback
            feedback_summary = await self.feedback_manager.get_feedback_summary(days=7)
            
            # Get ethics metrics
            ethics_metrics = await self.ethics_framework.get_ethics_metrics(days=7)
            
            return {
                "user_id": self.user_id,
                "satisfaction_rating": user_metrics.satisfaction_rating,
                "feature_adoption_rate": user_metrics.feature_adoption_rate,
                "total_sessions": user_metrics.total_sessions,
                "recent_feedback_count": feedback_summary["total_feedback"],
                "ethics_violations": ethics_metrics.total_violations,
                "last_active": user_metrics.last_active.isoformat() if user_metrics.last_active else None
            }
        except Exception as e:
            logger.error(f"Failed to get user insights: {e}")
            return {"error": str(e)}
    
    async def run_background_maintenance(self):
        """Run background maintenance tasks"""
        try:
            # Run technical debt scan (lightweight)
            debt_metrics = await self.debt_manager.get_debt_metrics()
            
            # Log system health
            logger.info(f"System health check - Debt items: {debt_metrics.total_items}, "
                       f"User satisfaction: {await self._get_avg_satisfaction():.1f}")
            
        except Exception as e:
            logger.error(f"Background maintenance failed: {e}")
    
    async def _get_avg_satisfaction(self) -> float:
        """Get average user satisfaction (simplified)"""
        try:
            user_metrics = await self.feedback_manager.calculate_user_satisfaction(self.user_id)
            return user_metrics.satisfaction_rating or 0.0
        except Exception:
            return 0.0


# Global integration instance
_integration: Optional[EnhancementSystemsIntegration] = None


def get_enhancement_integration(user_id: Optional[str] = None) -> EnhancementSystemsIntegration:
    """Get the global enhancement integration instance"""
    global _integration
    if _integration is None:
        _integration = EnhancementSystemsIntegration(user_id)
    return _integration


# Convenience functions for easy integration
async def track_model_selection(model_name: str, selection_method: str = "auto"):
    """Convenience function to track model selection"""
    integration = get_enhancement_integration()
    await integration.track_model_selection(model_name, selection_method)


async def track_query_response(query: str, response: str, response_time_ms: Optional[int] = None):
    """Convenience function to track query and analyze response"""
    integration = get_enhancement_integration()
    
    # Track the query
    await integration.track_user_query(query, response_time_ms)
    
    # Analyze the response for ethics violations
    violations = await integration.analyze_ai_response(query, response)
    
    return violations


async def collect_response_feedback(rating: int, message: str = ""):
    """Convenience function to collect feedback on AI response"""
    integration = get_enhancement_integration()
    await integration.collect_user_feedback_on_response(rating, message)


async def report_system_error(error_message: str, context: Optional[Dict[str, Any]] = None):
    """Convenience function to report system errors"""
    integration = get_enhancement_integration()
    await integration.report_error(error_message, context)


async def get_system_insights() -> Dict[str, Any]:
    """Convenience function to get system insights"""
    integration = get_enhancement_integration()
    return await integration.get_user_insights()