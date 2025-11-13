#!/usr/bin/env python3
"""
Advanced Analytics Engine

Machine learning-powered analytics engine for Xencode that provides:
- Usage pattern analysis and user behavior insights
- Cost tracking and optimization recommendations  
- ML-powered trend analysis and anomaly detection
- Predictive analytics for resource planning
- Advanced reporting and data visualization

Key Features:
- Real-time usage pattern detection
- Cost optimization recommendations
- Anomaly detection using statistical methods
- Predictive trend analysis
- User behavior clustering and segmentation
- Resource utilization forecasting
"""

import asyncio
import time
import statistics
import json
import sqlite3
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set
from pathlib import Path
from collections import defaultdict, Counter
from enum import Enum
import math

# Import for ML-powered analysis
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

# Import existing analytics components
try:
    from .advanced_analytics_dashboard import MetricsCollector, PerformanceMetrics, CostMetrics
    DASHBOARD_AVAILABLE = True
except ImportError:
    DASHBOARD_AVAILABLE = False
    # Mock classes for standalone operation
    class MetricsCollector:
        def __init__(self, *args, **kwargs): pass
    class PerformanceMetrics:
        def __init__(self, *args, **kwargs): pass
    class CostMetrics:
        def __init__(self, *args, **kwargs): pass


class AnalysisType(str, Enum):
    """Types of analytics analysis"""
    USAGE_PATTERNS = "usage_patterns"
    COST_OPTIMIZATION = "cost_optimization"
    TREND_ANALYSIS = "trend_analysis"
    ANOMALY_DETECTION = "anomaly_detection"
    USER_BEHAVIOR = "user_behavior"
    RESOURCE_FORECASTING = "resource_forecasting"


class AnomalyType(str, Enum):
    """Types of anomalies that can be detected"""
    PERFORMANCE_SPIKE = "performance_spike"
    USAGE_ANOMALY = "usage_anomaly"
    COST_ANOMALY = "cost_anomaly"
    ERROR_SPIKE = "error_spike"
    RESOURCE_EXHAUSTION = "resource_exhaustion"


@dataclass
class UsagePattern:
    """Represents a detected usage pattern"""
    pattern_id: str
    pattern_type: str
    description: str
    frequency: float
    confidence: float
    users_affected: List[str]
    time_periods: List[Tuple[datetime, datetime]]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CostOptimization:
    """Cost optimization recommendation"""
    optimization_id: str
    optimization_type: str
    title: str
    description: str
    potential_savings: float
    implementation_effort: str  # low, medium, high
    impact_score: float
    affected_components: List[str]
    recommended_actions: List[str]


@dataclass
class TrendAnalysis:
    """Trend analysis result"""
    metric_name: str
    trend_direction: str  # increasing, decreasing, stable, volatile
    trend_strength: float  # 0-1
    predicted_values: List[Tuple[datetime, float]]
    confidence_interval: Tuple[float, float]
    seasonality_detected: bool
    anomalies_detected: List[datetime]


@dataclass
class Anomaly:
    """Detected anomaly"""
    anomaly_id: str
    anomaly_type: AnomalyType
    timestamp: datetime
    metric_name: str
    expected_value: float
    actual_value: float
    severity: float  # 0-1
    description: str
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UserBehaviorProfile:
    """User behavior analysis profile"""
    user_id: str
    usage_frequency: str  # low, medium, high
    preferred_models: List[str]
    peak_usage_hours: List[int]
    average_session_length: float
    cost_efficiency_score: float
    behavior_cluster: str
    recommendations: List[str]


class UsagePatternAnalyzer:
    """Analyzes usage patterns and user behavior"""
    
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or Path("analytics.db")
        self._init_database()
    
    def _init_database(self) -> None:
        """Initialize analytics database"""
        with sqlite3.connect(self.db_path) as conn:
            # Usage events table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS usage_events (
                    id TEXT PRIMARY KEY,
                    timestamp REAL,
                    user_id TEXT,
                    event_type TEXT,
                    model TEXT,
                    tokens INTEGER,
                    cost REAL,
                    duration_ms REAL,
                    success BOOLEAN,
                    metadata TEXT
                )
            """)
            
            # User sessions table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    start_time REAL,
                    end_time REAL,
                    total_requests INTEGER,
                    total_tokens INTEGER,
                    total_cost REAL,
                    models_used TEXT
                )
            """)
            
            # Detected patterns table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS detected_patterns (
                    pattern_id TEXT PRIMARY KEY,
                    pattern_type TEXT,
                    description TEXT,
                    frequency REAL,
                    confidence REAL,
                    detected_at REAL,
                    metadata TEXT
                )
            """)
    
    def analyze_usage_patterns(self, hours: int = 24) -> List[UsagePattern]:
        """Analyze usage patterns over specified time period"""
        patterns = []
        cutoff_time = time.time() - (hours * 3600)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Analyze temporal patterns
            temporal_patterns = self._analyze_temporal_patterns(cursor, cutoff_time)
            patterns.extend(temporal_patterns)
            
            # Analyze model usage patterns
            model_patterns = self._analyze_model_usage_patterns(cursor, cutoff_time)
            patterns.extend(model_patterns)
            
            # Analyze user behavior patterns
            user_patterns = self._analyze_user_behavior_patterns(cursor, cutoff_time)
            patterns.extend(user_patterns)
        
        return patterns
    
    def _analyze_temporal_patterns(self, cursor, cutoff_time: float) -> List[UsagePattern]:
        """Analyze temporal usage patterns"""
        patterns = []
        
        # Get hourly usage distribution
        cursor.execute("""
            SELECT strftime('%H', datetime(timestamp, 'unixepoch')) as hour,
                   COUNT(*) as request_count,
                   AVG(tokens) as avg_tokens,
                   COUNT(DISTINCT user_id) as unique_users
            FROM usage_events 
            WHERE timestamp > ?
            GROUP BY hour
            ORDER BY hour
        """, (cutoff_time,))
        
        hourly_data = cursor.fetchall()
        
        if len(hourly_data) >= 3:
            # Detect peak usage hours
            request_counts = [row[1] for row in hourly_data]
            avg_requests = statistics.mean(request_counts)
            std_requests = statistics.stdev(request_counts) if len(request_counts) > 1 else 0
            
            peak_hours = []
            for hour_data in hourly_data:
                hour, count, avg_tokens, unique_users = hour_data
                if count > avg_requests + std_requests:
                    peak_hours.append(int(hour))
            
            if peak_hours:
                patterns.append(UsagePattern(
                    pattern_id=f"peak_hours_{int(time.time())}",
                    pattern_type="temporal_peak",
                    description=f"Peak usage detected during hours: {', '.join(map(str, peak_hours))}",
                    frequency=len(peak_hours) / 24,
                    confidence=0.8,
                    users_affected=[],
                    time_periods=[],
                    metadata={"peak_hours": peak_hours, "avg_requests": avg_requests}
                ))
        
        return patterns
    
    def _analyze_model_usage_patterns(self, cursor, cutoff_time: float) -> List[UsagePattern]:
        """Analyze model usage patterns"""
        patterns = []
        
        # Get model usage distribution
        cursor.execute("""
            SELECT model, 
                   COUNT(*) as usage_count,
                   AVG(tokens) as avg_tokens,
                   AVG(cost) as avg_cost,
                   COUNT(DISTINCT user_id) as unique_users,
                   AVG(CASE WHEN success THEN 1 ELSE 0 END) as success_rate
            FROM usage_events 
            WHERE timestamp > ?
            GROUP BY model
            ORDER BY usage_count DESC
        """, (cutoff_time,))
        
        model_data = cursor.fetchall()
        
        if model_data:
            total_usage = sum(row[1] for row in model_data)
            
            # Detect dominant models
            for model, count, avg_tokens, avg_cost, unique_users, success_rate in model_data:
                usage_percentage = (count / total_usage) * 100
                
                if usage_percentage > 50:  # Dominant model
                    patterns.append(UsagePattern(
                        pattern_id=f"dominant_model_{model}_{int(time.time())}",
                        pattern_type="model_dominance",
                        description=f"Model '{model}' dominates usage with {usage_percentage:.1f}% of requests",
                        frequency=usage_percentage / 100,
                        confidence=0.9,
                        users_affected=[],
                        time_periods=[],
                        metadata={
                            "model": model,
                            "usage_percentage": usage_percentage,
                            "avg_tokens": avg_tokens,
                            "success_rate": success_rate
                        }
                    ))
                
                # Detect inefficient model usage
                if avg_cost > 0 and success_rate < 0.8:  # High cost, low success
                    patterns.append(UsagePattern(
                        pattern_id=f"inefficient_model_{model}_{int(time.time())}",
                        pattern_type="model_inefficiency",
                        description=f"Model '{model}' shows inefficient usage: high cost, low success rate",
                        frequency=usage_percentage / 100,
                        confidence=0.7,
                        users_affected=[],
                        time_periods=[],
                        metadata={
                            "model": model,
                            "avg_cost": avg_cost,
                            "success_rate": success_rate
                        }
                    ))
        
        return patterns
    
    def _analyze_user_behavior_patterns(self, cursor, cutoff_time: float) -> List[UsagePattern]:
        """Analyze user behavior patterns"""
        patterns = []
        
        # Get user behavior data
        cursor.execute("""
            SELECT user_id,
                   COUNT(*) as request_count,
                   AVG(tokens) as avg_tokens,
                   SUM(cost) as total_cost,
                   COUNT(DISTINCT model) as models_used,
                   AVG(duration_ms) as avg_duration
            FROM usage_events 
            WHERE timestamp > ?
            GROUP BY user_id
            HAVING request_count > 5
            ORDER BY request_count DESC
        """, (cutoff_time,))
        
        user_data = cursor.fetchall()
        
        if len(user_data) >= 2:
            # Analyze user clustering
            request_counts = [row[1] for row in user_data]
            avg_requests = statistics.mean(request_counts)
            
            power_users = []
            casual_users = []
            
            for user_id, count, avg_tokens, total_cost, models_used, avg_duration in user_data:
                if count > avg_requests * 2:
                    power_users.append(user_id)
                elif count < avg_requests * 0.5:
                    casual_users.append(user_id)
            
            if power_users:
                patterns.append(UsagePattern(
                    pattern_id=f"power_users_{int(time.time())}",
                    pattern_type="user_segmentation",
                    description=f"Identified {len(power_users)} power users with high usage",
                    frequency=len(power_users) / len(user_data),
                    confidence=0.8,
                    users_affected=power_users,
                    time_periods=[],
                    metadata={"user_type": "power", "threshold": avg_requests * 2}
                ))
            
            if casual_users:
                patterns.append(UsagePattern(
                    pattern_id=f"casual_users_{int(time.time())}",
                    pattern_type="user_segmentation", 
                    description=f"Identified {len(casual_users)} casual users with low usage",
                    frequency=len(casual_users) / len(user_data),
                    confidence=0.8,
                    users_affected=casual_users,
                    time_periods=[],
                    metadata={"user_type": "casual", "threshold": avg_requests * 0.5}
                ))
        
        return patterns
    
    def generate_user_profiles(self, hours: int = 168) -> List[UserBehaviorProfile]:
        """Generate user behavior profiles (default: 1 week)"""
        profiles = []
        cutoff_time = time.time() - (hours * 3600)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT user_id,
                       COUNT(*) as total_requests,
                       AVG(tokens) as avg_tokens,
                       SUM(cost) as total_cost,
                       AVG(duration_ms) as avg_duration,
                       GROUP_CONCAT(DISTINCT model) as models,
                       strftime('%H', datetime(MIN(timestamp), 'unixepoch')) as first_hour,
                       strftime('%H', datetime(MAX(timestamp), 'unixepoch')) as last_hour
                FROM usage_events 
                WHERE timestamp > ?
                GROUP BY user_id
                HAVING total_requests >= 3
            """, (cutoff_time,))
            
            for row in cursor.fetchall():
                user_id, total_requests, avg_tokens, total_cost, avg_duration, models, first_hour, last_hour = row
                
                # Determine usage frequency
                requests_per_day = total_requests / (hours / 24)
                if requests_per_day > 10:
                    usage_frequency = "high"
                elif requests_per_day > 3:
                    usage_frequency = "medium"
                else:
                    usage_frequency = "low"
                
                # Parse preferred models
                preferred_models = models.split(',') if models else []
                
                # Calculate cost efficiency (tokens per dollar)
                cost_efficiency = (avg_tokens / total_cost) if total_cost > 0 else float('inf')
                cost_efficiency_score = min(cost_efficiency / 1000, 1.0)  # Normalize to 0-1
                
                # Determine behavior cluster
                if usage_frequency == "high" and cost_efficiency_score > 0.7:
                    behavior_cluster = "efficient_power_user"
                elif usage_frequency == "high" and cost_efficiency_score < 0.3:
                    behavior_cluster = "inefficient_power_user"
                elif usage_frequency == "medium":
                    behavior_cluster = "regular_user"
                else:
                    behavior_cluster = "casual_user"
                
                # Generate recommendations
                recommendations = self._generate_user_recommendations(
                    usage_frequency, cost_efficiency_score, preferred_models
                )
                
                profiles.append(UserBehaviorProfile(
                    user_id=user_id,
                    usage_frequency=usage_frequency,
                    preferred_models=preferred_models,
                    peak_usage_hours=[int(first_hour), int(last_hour)],
                    average_session_length=avg_duration / 1000 if avg_duration else 0,
                    cost_efficiency_score=cost_efficiency_score,
                    behavior_cluster=behavior_cluster,
                    recommendations=recommendations
                ))
        
        return profiles
    
    def _generate_user_recommendations(self, usage_frequency: str, cost_efficiency: float, models: List[str]) -> List[str]:
        """Generate personalized recommendations for users"""
        recommendations = []
        
        if usage_frequency == "high" and cost_efficiency < 0.5:
            recommendations.append("Consider using more cost-effective models for routine tasks")
            recommendations.append("Review your model selection strategy to optimize costs")
        
        if len(models) == 1:
            recommendations.append("Explore other models that might be better suited for different tasks")
        
        if usage_frequency == "low":
            recommendations.append("Consider batch processing to improve efficiency")
        
        if cost_efficiency > 0.8:
            recommendations.append("Great cost efficiency! Consider sharing your usage patterns with the team")
        
        return recommendations


class CostOptimizationEngine:
    """Analyzes costs and generates optimization recommendations"""
    
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or Path("analytics.db")
        
        # Model cost database (cost per 1K tokens)
        self.model_costs = {
            "gpt-4": 0.03,
            "gpt-4-turbo": 0.01,
            "gpt-3.5-turbo": 0.002,
            "claude-3-opus": 0.015,
            "claude-3-sonnet": 0.003,
            "claude-3-haiku": 0.00025,
            "gemini-pro": 0.001,
            "gemini-pro-vision": 0.002,
            "local-llama": 0.0,
            "local-mistral": 0.0
        }
    
    def analyze_cost_optimization_opportunities(self, hours: int = 24) -> List[CostOptimization]:
        """Analyze cost optimization opportunities"""
        optimizations = []
        cutoff_time = time.time() - (hours * 3600)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Analyze model cost efficiency
            model_optimizations = self._analyze_model_cost_efficiency(cursor, cutoff_time)
            optimizations.extend(model_optimizations)
            
            # Analyze user cost patterns
            user_optimizations = self._analyze_user_cost_patterns(cursor, cutoff_time)
            optimizations.extend(user_optimizations)
            
            # Analyze temporal cost patterns
            temporal_optimizations = self._analyze_temporal_cost_patterns(cursor, cutoff_time)
            optimizations.extend(temporal_optimizations)
        
        return sorted(optimizations, key=lambda x: x.potential_savings, reverse=True)
    
    def _analyze_model_cost_efficiency(self, cursor, cutoff_time: float) -> List[CostOptimization]:
        """Analyze model cost efficiency and suggest alternatives"""
        optimizations = []
        
        cursor.execute("""
            SELECT model,
                   COUNT(*) as usage_count,
                   SUM(cost) as total_cost,
                   AVG(tokens) as avg_tokens,
                   SUM(tokens) as total_tokens,
                   AVG(CASE WHEN success THEN 1 ELSE 0 END) as success_rate
            FROM usage_events 
            WHERE timestamp > ? AND cost > 0
            GROUP BY model
            ORDER BY total_cost DESC
        """, (cutoff_time,))
        
        model_data = cursor.fetchall()
        
        for model, usage_count, total_cost, avg_tokens, total_tokens, success_rate in model_data:
            if total_cost > 1.0:  # Only analyze significant costs
                
                # Check if there are cheaper alternatives
                current_cost_per_token = total_cost / total_tokens if total_tokens > 0 else 0
                
                # Find cheaper alternatives
                cheaper_alternatives = []
                for alt_model, alt_cost in self.model_costs.items():
                    if alt_model != model and alt_cost < current_cost_per_token:
                        potential_savings = (current_cost_per_token - alt_cost) * total_tokens
                        if potential_savings > 0.1:  # Minimum $0.10 savings
                            cheaper_alternatives.append((alt_model, potential_savings))
                
                if cheaper_alternatives:
                    best_alternative = max(cheaper_alternatives, key=lambda x: x[1])
                    alt_model, savings = best_alternative
                    
                    optimizations.append(CostOptimization(
                        optimization_id=f"model_switch_{model}_{alt_model}_{int(time.time())}",
                        optimization_type="model_substitution",
                        title=f"Switch from {model} to {alt_model}",
                        description=f"Replace {model} with {alt_model} for similar tasks to reduce costs",
                        potential_savings=savings,
                        implementation_effort="low",
                        impact_score=min(savings / total_cost, 1.0),
                        affected_components=[model],
                        recommended_actions=[
                            f"Test {alt_model} with representative workloads",
                            f"Gradually migrate from {model} to {alt_model}",
                            "Monitor quality and performance during transition"
                        ]
                    ))
                
                # Check for overuse of expensive models
                if model in ["gpt-4", "claude-3-opus"] and usage_count > 100:
                    optimizations.append(CostOptimization(
                        optimization_id=f"reduce_expensive_{model}_{int(time.time())}",
                        optimization_type="usage_reduction",
                        title=f"Reduce {model} usage for routine tasks",
                        description=f"Use {model} only for complex tasks requiring its advanced capabilities",
                        potential_savings=total_cost * 0.3,  # Assume 30% reduction
                        implementation_effort="medium",
                        impact_score=0.3,
                        affected_components=[model],
                        recommended_actions=[
                            "Implement task complexity classification",
                            "Route simple tasks to cheaper models",
                            "Reserve expensive models for complex reasoning"
                        ]
                    ))
        
        return optimizations
    
    def _analyze_user_cost_patterns(self, cursor, cutoff_time: float) -> List[CostOptimization]:
        """Analyze user cost patterns for optimization"""
        optimizations = []
        
        cursor.execute("""
            SELECT user_id,
                   SUM(cost) as total_cost,
                   COUNT(*) as request_count,
                   AVG(tokens) as avg_tokens,
                   COUNT(DISTINCT model) as models_used
            FROM usage_events 
            WHERE timestamp > ? AND cost > 0
            GROUP BY user_id
            HAVING total_cost > 5.0
            ORDER BY total_cost DESC
        """, (cutoff_time,))
        
        user_data = cursor.fetchall()
        
        if user_data:
            # Calculate cost statistics
            costs = [row[1] for row in user_data]
            avg_cost = statistics.mean(costs)
            
            for user_id, total_cost, request_count, avg_tokens, models_used in user_data:
                cost_per_request = total_cost / request_count
                
                # Identify high-cost users
                if total_cost > avg_cost * 2:
                    optimizations.append(CostOptimization(
                        optimization_id=f"high_cost_user_{user_id}_{int(time.time())}",
                        optimization_type="user_education",
                        title=f"Cost optimization for user {user_id}",
                        description=f"User {user_id} has high costs ({total_cost:.2f}) - provide cost awareness training",
                        potential_savings=total_cost * 0.2,  # Assume 20% reduction with training
                        implementation_effort="low",
                        impact_score=0.2,
                        affected_components=[user_id],
                        recommended_actions=[
                            "Provide cost awareness training",
                            "Share cost-effective usage patterns",
                            "Implement cost budgets and alerts"
                        ]
                    ))
                
                # Check for model diversity (might indicate inefficient model selection)
                if models_used > 5 and cost_per_request > 0.1:
                    optimizations.append(CostOptimization(
                        optimization_id=f"model_consolidation_{user_id}_{int(time.time())}",
                        optimization_type="model_consolidation",
                        title=f"Consolidate model usage for user {user_id}",
                        description=f"User {user_id} uses {models_used} different models - consolidation may reduce costs",
                        potential_savings=total_cost * 0.15,
                        implementation_effort="medium",
                        impact_score=0.15,
                        affected_components=[user_id],
                        recommended_actions=[
                            "Analyze task types and model performance",
                            "Recommend 2-3 optimal models for user's needs",
                            "Provide model selection guidelines"
                        ]
                    ))
        
        return optimizations
    
    def _analyze_temporal_cost_patterns(self, cursor, cutoff_time: float) -> List[CostOptimization]:
        """Analyze temporal cost patterns"""
        optimizations = []
        
        cursor.execute("""
            SELECT strftime('%H', datetime(timestamp, 'unixepoch')) as hour,
                   SUM(cost) as hourly_cost,
                   COUNT(*) as hourly_requests
            FROM usage_events 
            WHERE timestamp > ? AND cost > 0
            GROUP BY hour
            ORDER BY hourly_cost DESC
        """, (cutoff_time,))
        
        hourly_data = cursor.fetchall()
        
        if len(hourly_data) >= 3:
            costs = [row[1] for row in hourly_data]
            avg_hourly_cost = statistics.mean(costs)
            
            # Identify peak cost hours
            peak_cost_hours = []
            for hour, cost, requests in hourly_data:
                if cost > avg_hourly_cost * 1.5:
                    peak_cost_hours.append((int(hour), cost))
            
            if peak_cost_hours:
                total_peak_cost = sum(cost for _, cost in peak_cost_hours)
                
                optimizations.append(CostOptimization(
                    optimization_id=f"peak_hour_optimization_{int(time.time())}",
                    optimization_type="temporal_optimization",
                    title="Optimize peak hour usage",
                    description=f"High costs during peak hours: {', '.join(str(h) for h, _ in peak_cost_hours)}",
                    potential_savings=total_peak_cost * 0.25,  # 25% reduction through load balancing
                    implementation_effort="medium",
                    impact_score=0.25,
                    affected_components=["scheduling"],
                    recommended_actions=[
                        "Implement usage scheduling and load balancing",
                        "Encourage off-peak usage with incentives",
                        "Use batch processing during low-cost hours"
                    ]
                ))
        
        return optimizations
    
    def calculate_roi_projections(self, optimizations: List[CostOptimization], months: int = 12) -> Dict[str, Any]:
        """Calculate ROI projections for optimizations"""
        total_potential_savings = sum(opt.potential_savings for opt in optimizations)
        monthly_savings = total_potential_savings * 30  # Assume daily savings * 30
        annual_savings = monthly_savings * 12
        
        # Estimate implementation costs
        implementation_costs = {
            "low": 100,    # $100 for low effort
            "medium": 500, # $500 for medium effort  
            "high": 2000   # $2000 for high effort
        }
        
        total_implementation_cost = sum(
            implementation_costs.get(opt.implementation_effort, 500) 
            for opt in optimizations
        )
        
        # Calculate ROI
        net_savings = (monthly_savings * months) - total_implementation_cost
        roi_percentage = (net_savings / total_implementation_cost * 100) if total_implementation_cost > 0 else 0
        
        return {
            "total_optimizations": len(optimizations),
            "potential_monthly_savings": monthly_savings,
            "potential_annual_savings": annual_savings,
            "implementation_cost": total_implementation_cost,
            "net_savings_12_months": net_savings,
            "roi_percentage": roi_percentage,
            "payback_period_months": total_implementation_cost / monthly_savings if monthly_savings > 0 else float('inf'),
            "high_impact_optimizations": len([opt for opt in optimizations if opt.impact_score > 0.5])
        }


class MLTrendAnalyzer:
    """Machine learning-powered trend analysis and anomaly detection"""
    
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or Path("analytics.db")
        self.min_data_points = 10  # Minimum points needed for analysis
    
    def analyze_trends(self, metric_name: str, hours: int = 168) -> TrendAnalysis:
        """Analyze trends for a specific metric using statistical methods"""
        cutoff_time = time.time() - (hours * 3600)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get time series data
            cursor.execute("""
                SELECT timestamp, value 
                FROM metrics 
                WHERE metric_name = ? AND timestamp > ?
                ORDER BY timestamp
            """, (metric_name, cutoff_time))
            
            data_points = cursor.fetchall()
        
        if len(data_points) < self.min_data_points:
            return TrendAnalysis(
                metric_name=metric_name,
                trend_direction="insufficient_data",
                trend_strength=0.0,
                predicted_values=[],
                confidence_interval=(0.0, 0.0),
                seasonality_detected=False,
                anomalies_detected=[]
            )
        
        # Extract timestamps and values
        timestamps = [datetime.fromtimestamp(point[0]) for point in data_points]
        values = [point[1] for point in data_points]
        
        # Perform trend analysis
        trend_direction, trend_strength = self._calculate_trend(values)
        
        # Detect anomalies
        anomalies = self._detect_anomalies(timestamps, values)
        
        # Generate predictions
        predicted_values = self._generate_predictions(timestamps, values, hours=24)
        
        # Calculate confidence interval
        confidence_interval = self._calculate_confidence_interval(values)
        
        # Detect seasonality
        seasonality_detected = self._detect_seasonality(values)
        
        return TrendAnalysis(
            metric_name=metric_name,
            trend_direction=trend_direction,
            trend_strength=trend_strength,
            predicted_values=predicted_values,
            confidence_interval=confidence_interval,
            seasonality_detected=seasonality_detected,
            anomalies_detected=[ts for ts, _ in anomalies]
        )
    
    def _calculate_trend(self, values: List[float]) -> Tuple[str, float]:
        """Calculate trend direction and strength using linear regression"""
        if len(values) < 2:
            return "stable", 0.0
        
        n = len(values)
        x = list(range(n))
        
        # Calculate linear regression slope
        x_mean = statistics.mean(x)
        y_mean = statistics.mean(values)
        
        numerator = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return "stable", 0.0
        
        slope = numerator / denominator
        
        # Calculate correlation coefficient (trend strength)
        if len(values) > 1:
            try:
                # Simple correlation calculation
                x_std = statistics.stdev(x) if len(x) > 1 else 0
                y_std = statistics.stdev(values) if len(values) > 1 else 0
                
                if x_std > 0 and y_std > 0:
                    correlation = numerator / (n * x_std * y_std)
                    trend_strength = abs(correlation)
                else:
                    trend_strength = 0.0
            except (ZeroDivisionError, ValueError, OverflowError):
                trend_strength = 0.0
        else:
            trend_strength = 0.0
        
        # Determine trend direction
        if abs(slope) < 0.01:  # Very small slope
            trend_direction = "stable"
        elif slope > 0:
            if trend_strength > 0.7:
                trend_direction = "strongly_increasing"
            else:
                trend_direction = "increasing"
        else:
            if trend_strength > 0.7:
                trend_direction = "strongly_decreasing"
            else:
                trend_direction = "decreasing"
        
        return trend_direction, min(trend_strength, 1.0)
    
    def _detect_anomalies(self, timestamps: List[datetime], values: List[float]) -> List[Tuple[datetime, float]]:
        """Detect anomalies using statistical methods"""
        if len(values) < 5:
            return []
        
        anomalies = []
        
        # Calculate moving statistics
        window_size = min(10, len(values) // 3)
        
        for i in range(window_size, len(values)):
            # Get window data
            window_values = values[i-window_size:i]
            current_value = values[i]
            
            # Calculate statistics
            window_mean = statistics.mean(window_values)
            window_std = statistics.stdev(window_values) if len(window_values) > 1 else 0
            
            # Detect anomaly using z-score
            if window_std > 0:
                z_score = abs(current_value - window_mean) / window_std
                if z_score > 2.5:  # 2.5 standard deviations
                    anomalies.append((timestamps[i], current_value))
        
        return anomalies
    
    def _generate_predictions(self, timestamps: List[datetime], values: List[float], hours: int = 24) -> List[Tuple[datetime, float]]:
        """Generate simple predictions using trend extrapolation"""
        if len(values) < 3:
            return []
        
        # Calculate trend
        trend_direction, trend_strength = self._calculate_trend(values)
        
        # Get recent trend
        recent_values = values[-min(10, len(values)):]
        recent_mean = statistics.mean(recent_values)
        
        # Calculate hourly change rate
        if len(values) >= 2:
            time_diff = (timestamps[-1] - timestamps[0]).total_seconds() / 3600  # hours
            value_diff = values[-1] - values[0]
            hourly_change = value_diff / time_diff if time_diff > 0 else 0
        else:
            hourly_change = 0
        
        # Generate predictions
        predictions = []
        last_timestamp = timestamps[-1]
        last_value = values[-1]
        
        for hour in range(1, hours + 1):
            future_timestamp = last_timestamp + timedelta(hours=hour)
            
            # Simple linear extrapolation with some noise reduction
            predicted_value = last_value + (hourly_change * hour * trend_strength)
            
            # Add some bounds to prevent unrealistic predictions
            predicted_value = max(0, predicted_value)  # No negative values
            predicted_value = min(predicted_value, recent_mean * 3)  # Cap at 3x recent mean
            
            predictions.append((future_timestamp, predicted_value))
        
        return predictions
    
    def _calculate_confidence_interval(self, values: List[float]) -> Tuple[float, float]:
        """Calculate confidence interval for predictions"""
        if len(values) < 2:
            return (0.0, 0.0)
        
        mean_val = statistics.mean(values)
        std_val = statistics.stdev(values)
        
        # 95% confidence interval (approximately 2 standard deviations)
        lower_bound = mean_val - (2 * std_val)
        upper_bound = mean_val + (2 * std_val)
        
        return (lower_bound, upper_bound)
    
    def _detect_seasonality(self, values: List[float]) -> bool:
        """Simple seasonality detection"""
        if len(values) < 24:  # Need at least 24 data points
            return False
        
        # Check for patterns in different periods
        periods_to_check = [24, 12, 8, 6]  # Daily, bi-daily, etc.
        
        for period in periods_to_check:
            if len(values) >= period * 2:
                # Calculate autocorrelation for this period
                correlation = self._calculate_autocorrelation(values, period)
                if correlation > 0.5:  # Strong correlation indicates seasonality
                    return True
        
        return False
    
    def _calculate_autocorrelation(self, values: List[float], lag: int) -> float:
        """Calculate autocorrelation at given lag"""
        if len(values) <= lag:
            return 0.0
        
        n = len(values) - lag
        if n <= 1:
            return 0.0
        
        # Calculate correlation between values and lagged values
        original = values[:-lag]
        lagged = values[lag:]
        
        if len(original) != len(lagged) or len(original) < 2:
            return 0.0
        
        try:
            orig_mean = statistics.mean(original)
            lagged_mean = statistics.mean(lagged)
            
            numerator = sum((original[i] - orig_mean) * (lagged[i] - lagged_mean) for i in range(len(original)))
            
            orig_var = sum((x - orig_mean) ** 2 for x in original)
            lagged_var = sum((x - lagged_mean) ** 2 for x in lagged)
            
            denominator = math.sqrt(orig_var * lagged_var)
            
            if denominator == 0:
                return 0.0
            
            return numerator / denominator
        except (ZeroDivisionError, ValueError, OverflowError):
            return 0.0


class AdvancedAnalyticsEngine:
    """Main advanced analytics engine that coordinates all analysis components"""
    
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or Path("advanced_analytics.db")
        
        # Initialize analysis components
        self.usage_analyzer = UsagePatternAnalyzer(self.db_path)
        self.cost_optimizer = CostOptimizationEngine(self.db_path)
        self.trend_analyzer = MLTrendAnalyzer(self.db_path)
        
        # Initialize database
        self._init_database()
    
    def _init_database(self) -> None:
        """Initialize the advanced analytics database"""
        with sqlite3.connect(self.db_path) as conn:
            # Metrics table for trend analysis
            conn.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    id TEXT PRIMARY KEY,
                    timestamp REAL,
                    metric_name TEXT,
                    value REAL,
                    metadata TEXT
                )
            """)
            
            # Analysis results table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS analysis_results (
                    id TEXT PRIMARY KEY,
                    analysis_type TEXT,
                    timestamp REAL,
                    results TEXT,
                    metadata TEXT
                )
            """)
    
    async def run_comprehensive_analysis(self, hours: int = 24) -> Dict[str, Any]:
        """Run comprehensive analytics analysis"""
        results = {
            "analysis_timestamp": datetime.now().isoformat(),
            "analysis_period_hours": hours,
            "usage_patterns": [],
            "cost_optimizations": [],
            "trend_analyses": {},
            "user_profiles": [],
            "summary": {}
        }
        
        try:
            # Analyze usage patterns
            usage_patterns = self.usage_analyzer.analyze_usage_patterns(hours)
            results["usage_patterns"] = [
                {
                    "pattern_id": p.pattern_id,
                    "type": p.pattern_type,
                    "description": p.description,
                    "frequency": p.frequency,
                    "confidence": p.confidence,
                    "metadata": p.metadata
                }
                for p in usage_patterns
            ]
            
            # Generate user profiles
            user_profiles = self.usage_analyzer.generate_user_profiles(hours * 7)  # Week of data
            results["user_profiles"] = [
                {
                    "user_id": p.user_id,
                    "usage_frequency": p.usage_frequency,
                    "preferred_models": p.preferred_models,
                    "cost_efficiency_score": p.cost_efficiency_score,
                    "behavior_cluster": p.behavior_cluster,
                    "recommendations": p.recommendations
                }
                for p in user_profiles
            ]
            
            # Analyze cost optimizations
            cost_optimizations = self.cost_optimizer.analyze_cost_optimization_opportunities(hours)
            results["cost_optimizations"] = [
                {
                    "optimization_id": opt.optimization_id,
                    "type": opt.optimization_type,
                    "title": opt.title,
                    "description": opt.description,
                    "potential_savings": opt.potential_savings,
                    "implementation_effort": opt.implementation_effort,
                    "impact_score": opt.impact_score,
                    "recommended_actions": opt.recommended_actions
                }
                for opt in cost_optimizations
            ]
            
            # Calculate ROI projections
            roi_projections = self.cost_optimizer.calculate_roi_projections(cost_optimizations)
            results["roi_projections"] = roi_projections
            
            # Analyze trends for key metrics
            key_metrics = ["cpu_usage", "memory_usage", "response_time", "error_rate", "cost"]
            for metric in key_metrics:
                try:
                    trend_analysis = self.trend_analyzer.analyze_trends(metric, hours)
                    results["trend_analyses"][metric] = {
                        "trend_direction": trend_analysis.trend_direction,
                        "trend_strength": trend_analysis.trend_strength,
                        "seasonality_detected": trend_analysis.seasonality_detected,
                        "anomalies_count": len(trend_analysis.anomalies_detected),
                        "predictions_count": len(trend_analysis.predicted_values)
                    }
                except Exception as e:
                    results["trend_analyses"][metric] = {"error": str(e)}
            
            # Generate summary
            results["summary"] = {
                "patterns_detected": len(usage_patterns),
                "users_analyzed": len(user_profiles),
                "optimizations_found": len(cost_optimizations),
                "total_potential_savings": sum(opt.potential_savings for opt in cost_optimizations),
                "high_impact_optimizations": len([opt for opt in cost_optimizations if opt.impact_score > 0.5]),
                "trends_analyzed": len([t for t in results["trend_analyses"].values() if "error" not in t])
            }
            
        except Exception as e:
            results["error"] = str(e)
        
        # Store analysis results
        await self._store_analysis_results(results)
        
        return results
    
    async def _store_analysis_results(self, results: Dict[str, Any]) -> None:
        """Store analysis results in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO analysis_results (id, analysis_type, timestamp, results, metadata)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    f"comprehensive_{int(time.time())}",
                    "comprehensive_analysis",
                    time.time(),
                    json.dumps(results),
                    json.dumps({"version": "1.0"})
                ))
        except Exception as e:
            print(f"Error storing analysis results: {e}")
    
    def generate_sample_data(self, days: int = 7) -> None:
        """Generate sample data for demonstration"""
        import random
        
        current_time = time.time()
        
        # Generate sample usage events
        users = [f"user_{i}" for i in range(20)]
        models = ["gpt-4", "gpt-3.5-turbo", "claude-3-sonnet", "gemini-pro", "local-llama"]
        
        with sqlite3.connect(self.db_path) as conn:
            for day in range(days):
                day_start = current_time - (day * 86400)
                
                # Generate events for this day
                events_per_day = random.randint(50, 200)
                
                for _ in range(events_per_day):
                    timestamp = day_start + random.uniform(0, 86400)
                    user_id = random.choice(users)
                    model = random.choice(models)
                    tokens = random.randint(50, 2000)
                    
                    # Calculate cost based on model
                    cost_per_token = self.cost_optimizer.model_costs.get(model, 0.001)
                    cost = (tokens / 1000) * cost_per_token
                    
                    duration_ms = random.uniform(500, 5000)
                    success = random.random() > 0.05  # 95% success rate
                    
                    # Insert usage event
                    conn.execute("""
                        INSERT INTO usage_events 
                        (id, timestamp, user_id, event_type, model, tokens, cost, duration_ms, success, metadata)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        f"event_{timestamp}_{user_id}",
                        timestamp,
                        user_id,
                        "chat_completion",
                        model,
                        tokens,
                        cost,
                        duration_ms,
                        success,
                        json.dumps({"sample": True})
                    ))
            
            # Generate sample metrics
            metrics = ["cpu_usage", "memory_usage", "response_time", "error_rate"]
            
            for day in range(days):
                day_start = current_time - (day * 86400)
                
                for hour in range(24):
                    timestamp = day_start + (hour * 3600)
                    
                    for metric in metrics:
                        if metric == "cpu_usage":
                            value = random.uniform(20, 80) + random.uniform(-10, 10)  # Add some trend
                        elif metric == "memory_usage":
                            value = random.uniform(30, 70) + (day * 2)  # Increasing trend
                        elif metric == "response_time":
                            value = random.uniform(0.5, 3.0)
                        else:  # error_rate
                            value = random.uniform(0, 5)
                        
                        conn.execute("""
                            INSERT INTO metrics (id, timestamp, metric_name, value, metadata)
                            VALUES (?, ?, ?, ?, ?)
                        """, (
                            f"metric_{timestamp}_{metric}",
                            timestamp,
                            metric,
                            max(0, value),  # Ensure non-negative
                            json.dumps({"sample": True})
                        ))


# Demo function
async def run_advanced_analytics_demo():
    """Run advanced analytics engine demo"""
    from rich.console import Console
    
    console = Console()
    console.print("🚀 [bold cyan]Advanced Analytics Engine Demo[/bold cyan]\n")
    
    # Create analytics engine
    engine = AdvancedAnalyticsEngine()
    
    # Generate sample data
    console.print("📊 Generating sample analytics data...")
    engine.generate_sample_data(days=7)
    console.print("✅ Sample data generated\n")
    
    # Run comprehensive analysis
    console.print("🔍 Running comprehensive analytics analysis...")
    results = await engine.run_comprehensive_analysis(hours=168)  # 1 week
    
    # Display results
    console.print("📈 [bold green]Analysis Results:[/bold green]\n")
    
    summary = results.get("summary", {})
    console.print(f"   📋 Patterns detected: {summary.get('patterns_detected', 0)}")
    console.print(f"   👥 Users analyzed: {summary.get('users_analyzed', 0)}")
    console.print(f"   💡 Optimizations found: {summary.get('optimizations_found', 0)}")
    console.print(f"   💰 Potential savings: ${summary.get('total_potential_savings', 0):.2f}")
    console.print(f"   🎯 High-impact optimizations: {summary.get('high_impact_optimizations', 0)}")
    
    # Show top optimizations
    optimizations = results.get("cost_optimizations", [])
    if optimizations:
        console.print("\n💡 [bold yellow]Top Cost Optimizations:[/bold yellow]")
        for i, opt in enumerate(optimizations[:3]):
            console.print(f"   {i+1}. {opt['title']}")
            console.print(f"      💰 Savings: ${opt['potential_savings']:.2f}")
            console.print(f"      🔧 Effort: {opt['implementation_effort']}")
    
    # Show usage patterns
    patterns = results.get("usage_patterns", [])
    if patterns:
        console.print("\n📊 [bold blue]Usage Patterns Detected:[/bold blue]")
        for i, pattern in enumerate(patterns[:3]):
            console.print(f"   {i+1}. {pattern['description']}")
            console.print(f"      🎯 Confidence: {pattern['confidence']:.1%}")
    
    console.print("\n✨ [green]Advanced analytics analysis complete![/green]")


if __name__ == "__main__":
    asyncio.run(run_advanced_analytics_demo())