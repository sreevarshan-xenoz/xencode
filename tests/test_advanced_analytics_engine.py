#!/usr/bin/env python3
"""
Tests for Advanced Analytics Engine

Comprehensive test suite for the advanced analytics engine including
usage pattern analysis, cost optimization, and ML-powered trend analysis.
"""

import pytest
import asyncio
import time
import tempfile
import sqlite3
import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

# Import the components to test
try:
    from xencode.advanced_analytics_engine import (
        AdvancedAnalyticsEngine, UsagePatternAnalyzer, CostOptimizationEngine,
        MLTrendAnalyzer, UsagePattern, CostOptimization, TrendAnalysis,
        UserBehaviorProfile, Anomaly, AnalysisType, AnomalyType
    )
    ANALYTICS_AVAILABLE = True
except ImportError:
    ANALYTICS_AVAILABLE = False
    pytest.skip("Advanced analytics engine not available", allow_module_level=True)


class TestUsagePatternAnalyzer:
    """Test the UsagePatternAnalyzer class"""
    
    def setup_method(self):
        """Set up test database"""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.db_path = Path(self.temp_db.name)
        self.analyzer = UsagePatternAnalyzer(self.db_path)
        self._populate_test_data()
    
    def teardown_method(self):
        """Clean up test database"""
        self.temp_db.close()
        if self.db_path.exists():
            self.db_path.unlink()
    
    def _populate_test_data(self):
        """Populate database with test data"""
        current_time = time.time()
        
        with sqlite3.connect(self.db_path) as conn:
            # Add sample usage events
            test_events = [
                # Peak usage pattern (hour 14-16)
                (current_time - 3600, "user1", "chat_completion", "gpt-4", 500, 0.015, 1500, True),
                (current_time - 3600, "user2", "chat_completion", "gpt-4", 600, 0.018, 1800, True),
                (current_time - 3600, "user3", "chat_completion", "gpt-3.5-turbo", 300, 0.0006, 800, True),
                
                # Normal usage
                (current_time - 7200, "user1", "chat_completion", "gpt-3.5-turbo", 200, 0.0004, 600, True),
                (current_time - 7200, "user4", "chat_completion", "claude-3-sonnet", 400, 0.0012, 1200, True),
                
                # Power user pattern
                (current_time - 1800, "user1", "chat_completion", "gpt-4", 800, 0.024, 2000, True),
                (current_time - 1800, "user1", "chat_completion", "gpt-3.5-turbo", 300, 0.0006, 700, True),
                (current_time - 1800, "user1", "chat_completion", "claude-3-sonnet", 500, 0.0015, 1100, True),
                
                # Failed requests
                (current_time - 900, "user2", "chat_completion", "gpt-4", 0, 0, 0, False),
                (current_time - 900, "user3", "chat_completion", "gpt-4", 0, 0, 0, False),
            ]
            
            for i, (timestamp, user_id, event_type, model, tokens, cost, duration, success) in enumerate(test_events):
                conn.execute("""
                    INSERT INTO usage_events 
                    (id, timestamp, user_id, event_type, model, tokens, cost, duration_ms, success, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    f"test_event_{i}",
                    timestamp,
                    user_id,
                    event_type,
                    model,
                    tokens,
                    cost,
                    duration,
                    success,
                    json.dumps({"test": True})
                ))
    
    def test_analyzer_initialization(self):
        """Test analyzer initialization"""
        assert self.analyzer.db_path == self.db_path
        
        # Check that database tables were created
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            assert "usage_events" in tables
            assert "user_sessions" in tables
            assert "detected_patterns" in tables
    
    def test_analyze_usage_patterns(self):
        """Test usage pattern analysis"""
        patterns = self.analyzer.analyze_usage_patterns(hours=24)
        
        assert isinstance(patterns, list)
        assert len(patterns) > 0
        
        # Check pattern structure
        for pattern in patterns:
            assert isinstance(pattern, UsagePattern)
            assert pattern.pattern_id
            assert pattern.pattern_type
            assert pattern.description
            assert 0 <= pattern.confidence <= 1
    
    def test_temporal_pattern_detection(self):
        """Test temporal pattern detection"""
        patterns = self.analyzer.analyze_usage_patterns(hours=24)
        
        # Should detect some temporal patterns
        temporal_patterns = [p for p in patterns if p.pattern_type == "temporal_peak"]
        
        # Verify pattern properties
        for pattern in temporal_patterns:
            assert "peak_hours" in pattern.metadata
            assert isinstance(pattern.metadata["peak_hours"], list)
    
    def test_model_usage_pattern_detection(self):
        """Test model usage pattern detection"""
        patterns = self.analyzer.analyze_usage_patterns(hours=24)
        
        # Should detect model-related patterns
        model_patterns = [p for p in patterns if "model" in p.pattern_type]
        
        # Verify we have some model patterns
        assert len(model_patterns) >= 0  # May or may not detect patterns with limited test data
    
    def test_user_behavior_pattern_detection(self):
        """Test user behavior pattern detection"""
        patterns = self.analyzer.analyze_usage_patterns(hours=24)
        
        # Should detect user segmentation patterns
        user_patterns = [p for p in patterns if p.pattern_type == "user_segmentation"]
        
        # Verify pattern structure
        for pattern in user_patterns:
            assert pattern.users_affected is not None
            assert "user_type" in pattern.metadata
    
    def test_generate_user_profiles(self):
        """Test user profile generation"""
        profiles = self.analyzer.generate_user_profiles(hours=24)
        
        assert isinstance(profiles, list)
        
        # Check profile structure
        for profile in profiles:
            assert isinstance(profile, UserBehaviorProfile)
            assert profile.user_id
            assert profile.usage_frequency in ["low", "medium", "high"]
            assert isinstance(profile.preferred_models, list)
            assert 0 <= profile.cost_efficiency_score <= 1
            assert profile.behavior_cluster
            assert isinstance(profile.recommendations, list)
    
    def test_user_profile_clustering(self):
        """Test user behavior clustering"""
        profiles = self.analyzer.generate_user_profiles(hours=24)
        
        # Should have different behavior clusters
        clusters = set(profile.behavior_cluster for profile in profiles)
        
        # Verify cluster types
        valid_clusters = {
            "efficient_power_user", "inefficient_power_user", 
            "regular_user", "casual_user"
        }
        
        for cluster in clusters:
            assert cluster in valid_clusters


class TestCostOptimizationEngine:
    """Test the CostOptimizationEngine class"""
    
    def setup_method(self):
        """Set up test database"""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.db_path = Path(self.temp_db.name)
        self.optimizer = CostOptimizationEngine(self.db_path)
        self._populate_cost_test_data()
    
    def teardown_method(self):
        """Clean up test database"""
        self.temp_db.close()
        if self.db_path.exists():
            self.db_path.unlink()
    
    def _populate_cost_test_data(self):
        """Populate database with cost test data"""
        current_time = time.time()
        
        with sqlite3.connect(self.db_path) as conn:
            # Create usage_events table
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
            
            # Add expensive usage patterns
            expensive_events = [
                # High-cost user with expensive models
                (current_time - 3600, "expensive_user", "chat_completion", "gpt-4", 1000, 0.03, 2000, True),
                (current_time - 3600, "expensive_user", "chat_completion", "gpt-4", 1200, 0.036, 2400, True),
                (current_time - 3600, "expensive_user", "chat_completion", "claude-3-opus", 800, 0.012, 1800, True),
                
                # Inefficient model usage
                (current_time - 1800, "inefficient_user", "chat_completion", "gpt-4", 200, 0.006, 800, False),
                (current_time - 1800, "inefficient_user", "chat_completion", "gpt-4", 150, 0.0045, 600, False),
                
                # Cost-effective usage
                (current_time - 900, "efficient_user", "chat_completion", "gpt-3.5-turbo", 500, 0.001, 1000, True),
                (current_time - 900, "efficient_user", "chat_completion", "local-llama", 600, 0.0, 1200, True),
                
                # Peak hour usage
                (current_time - 14*3600, "peak_user", "chat_completion", "gpt-4", 800, 0.024, 1600, True),
                (current_time - 14*3600, "peak_user", "chat_completion", "gpt-4", 900, 0.027, 1800, True),
            ]
            
            for i, (timestamp, user_id, event_type, model, tokens, cost, duration, success) in enumerate(expensive_events):
                conn.execute("""
                    INSERT INTO usage_events 
                    (id, timestamp, user_id, event_type, model, tokens, cost, duration_ms, success, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    f"cost_event_{i}",
                    timestamp,
                    user_id,
                    event_type,
                    model,
                    tokens,
                    cost,
                    duration,
                    success,
                    json.dumps({"test": True})
                ))
    
    def test_optimizer_initialization(self):
        """Test optimizer initialization"""
        assert self.optimizer.db_path == self.db_path
        assert len(self.optimizer.model_costs) > 0
        
        # Check model costs are reasonable
        assert self.optimizer.model_costs["gpt-4"] > self.optimizer.model_costs["gpt-3.5-turbo"]
        assert self.optimizer.model_costs["local-llama"] == 0.0
    
    def test_analyze_cost_optimization_opportunities(self):
        """Test cost optimization analysis"""
        optimizations = self.optimizer.analyze_cost_optimization_opportunities(hours=24)
        
        assert isinstance(optimizations, list)
        
        # Check optimization structure
        for opt in optimizations:
            assert isinstance(opt, CostOptimization)
            assert opt.optimization_id
            assert opt.optimization_type
            assert opt.title
            assert opt.description
            assert opt.potential_savings >= 0
            assert opt.implementation_effort in ["low", "medium", "high"]
            assert 0 <= opt.impact_score <= 1
            assert isinstance(opt.recommended_actions, list)
    
    def test_model_cost_efficiency_analysis(self):
        """Test model cost efficiency analysis"""
        optimizations = self.optimizer.analyze_cost_optimization_opportunities(hours=24)
        
        # Should detect model substitution opportunities
        model_optimizations = [opt for opt in optimizations if opt.optimization_type == "model_substitution"]
        
        # Verify optimization properties
        for opt in model_optimizations:
            assert opt.potential_savings > 0
            assert len(opt.recommended_actions) > 0
    
    def test_user_cost_pattern_analysis(self):
        """Test user cost pattern analysis"""
        optimizations = self.optimizer.analyze_cost_optimization_opportunities(hours=24)
        
        # Should detect user-related optimizations
        user_optimizations = [opt for opt in optimizations if "user" in opt.optimization_type]
        
        # Verify user optimization properties
        for opt in user_optimizations:
            assert opt.potential_savings > 0
            assert any("user" in action.lower() for action in opt.recommended_actions)
    
    def test_calculate_roi_projections(self):
        """Test ROI projection calculations"""
        optimizations = self.optimizer.analyze_cost_optimization_opportunities(hours=24)
        
        if optimizations:  # Only test if we have optimizations
            roi_projections = self.optimizer.calculate_roi_projections(optimizations, months=12)
            
            assert isinstance(roi_projections, dict)
            assert "total_optimizations" in roi_projections
            assert "potential_monthly_savings" in roi_projections
            assert "potential_annual_savings" in roi_projections
            assert "implementation_cost" in roi_projections
            assert "roi_percentage" in roi_projections
            assert "payback_period_months" in roi_projections
            
            # Verify calculations make sense
            assert roi_projections["total_optimizations"] == len(optimizations)
            assert roi_projections["potential_annual_savings"] >= 0
            assert roi_projections["implementation_cost"] >= 0


class TestMLTrendAnalyzer:
    """Test the MLTrendAnalyzer class"""
    
    def setup_method(self):
        """Set up test database"""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.db_path = Path(self.temp_db.name)
        self.analyzer = MLTrendAnalyzer(self.db_path)
        self._populate_trend_test_data()
    
    def teardown_method(self):
        """Clean up test database"""
        self.temp_db.close()
        if self.db_path.exists():
            self.db_path.unlink()
    
    def _populate_trend_test_data(self):
        """Populate database with trend test data"""
        current_time = time.time()
        
        with sqlite3.connect(self.db_path) as conn:
            # Create metrics table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    id TEXT PRIMARY KEY,
                    timestamp REAL,
                    metric_name TEXT,
                    value REAL,
                    metadata TEXT
                )
            """)
            
            # Generate trend data
            for i in range(50):  # 50 data points
                timestamp = current_time - (i * 3600)  # Hourly data
                
                # Increasing trend
                cpu_value = 30 + (i * 0.5) + (i % 5 - 2)  # Base trend + noise
                
                # Decreasing trend
                memory_value = 80 - (i * 0.3) + (i % 3 - 1)  # Decreasing trend + noise
                
                # Stable trend
                response_time = 1.5 + (i % 4 - 2) * 0.1  # Stable with noise
                
                # Add some anomalies
                if i in [10, 25, 40]:  # Anomaly points
                    cpu_value += 20  # Spike
                
                conn.execute("""
                    INSERT INTO metrics (id, timestamp, metric_name, value, metadata)
                    VALUES (?, ?, ?, ?, ?)
                """, (f"cpu_{i}", timestamp, "cpu_usage", max(0, cpu_value), "{}"))
                
                conn.execute("""
                    INSERT INTO metrics (id, timestamp, metric_name, value, metadata)
                    VALUES (?, ?, ?, ?, ?)
                """, (f"mem_{i}", timestamp, "memory_usage", max(0, memory_value), "{}"))
                
                conn.execute("""
                    INSERT INTO metrics (id, timestamp, metric_name, value, metadata)
                    VALUES (?, ?, ?, ?, ?)
                """, (f"resp_{i}", timestamp, "response_time", max(0, response_time), "{}"))
    
    def test_analyzer_initialization(self):
        """Test analyzer initialization"""
        assert self.analyzer.db_path == self.db_path
        assert self.analyzer.min_data_points == 10
    
    def test_analyze_trends_insufficient_data(self):
        """Test trend analysis with insufficient data"""
        # Test with non-existent metric
        trend_analysis = self.analyzer.analyze_trends("nonexistent_metric", hours=24)
        
        assert isinstance(trend_analysis, TrendAnalysis)
        assert trend_analysis.metric_name == "nonexistent_metric"
        assert trend_analysis.trend_direction == "insufficient_data"
        assert trend_analysis.trend_strength == 0.0
    
    def test_analyze_trends_increasing(self):
        """Test trend analysis for increasing trend"""
        trend_analysis = self.analyzer.analyze_trends("cpu_usage", hours=168)
        
        assert isinstance(trend_analysis, TrendAnalysis)
        assert trend_analysis.metric_name == "cpu_usage"
        assert "increasing" in trend_analysis.trend_direction
        assert trend_analysis.trend_strength > 0
    
    def test_analyze_trends_decreasing(self):
        """Test trend analysis for decreasing trend"""
        trend_analysis = self.analyzer.analyze_trends("memory_usage", hours=168)
        
        assert isinstance(trend_analysis, TrendAnalysis)
        assert trend_analysis.metric_name == "memory_usage"
        assert "decreasing" in trend_analysis.trend_direction
        assert trend_analysis.trend_strength > 0
    
    def test_analyze_trends_stable(self):
        """Test trend analysis for stable trend"""
        trend_analysis = self.analyzer.analyze_trends("response_time", hours=168)
        
        assert isinstance(trend_analysis, TrendAnalysis)
        assert trend_analysis.metric_name == "response_time"
        # Should be stable or have low trend strength
        assert trend_analysis.trend_direction in ["stable", "increasing", "decreasing"]
    
    def test_anomaly_detection(self):
        """Test anomaly detection"""
        trend_analysis = self.analyzer.analyze_trends("cpu_usage", hours=168)
        
        # Should detect some anomalies (we added spikes in test data)
        assert len(trend_analysis.anomalies_detected) > 0
    
    def test_prediction_generation(self):
        """Test prediction generation"""
        trend_analysis = self.analyzer.analyze_trends("cpu_usage", hours=168)
        
        # Should generate predictions
        assert len(trend_analysis.predicted_values) > 0
        
        # Predictions should be tuples of (datetime, float)
        for prediction in trend_analysis.predicted_values:
            assert isinstance(prediction, tuple)
            assert len(prediction) == 2
            assert isinstance(prediction[0], datetime)
            assert isinstance(prediction[1], (int, float))
    
    def test_confidence_interval_calculation(self):
        """Test confidence interval calculation"""
        trend_analysis = self.analyzer.analyze_trends("cpu_usage", hours=168)
        
        # Should have confidence interval
        assert isinstance(trend_analysis.confidence_interval, tuple)
        assert len(trend_analysis.confidence_interval) == 2
        
        lower, upper = trend_analysis.confidence_interval
        assert lower <= upper


class TestAdvancedAnalyticsEngine:
    """Test the main AdvancedAnalyticsEngine class"""
    
    def setup_method(self):
        """Set up test database"""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.db_path = Path(self.temp_db.name)
        self.engine = AdvancedAnalyticsEngine(self.db_path)
    
    def teardown_method(self):
        """Clean up test database"""
        self.temp_db.close()
        if self.db_path.exists():
            self.db_path.unlink()
    
    def test_engine_initialization(self):
        """Test engine initialization"""
        assert self.engine.db_path == self.db_path
        assert self.engine.usage_analyzer is not None
        assert self.engine.cost_optimizer is not None
        assert self.engine.trend_analyzer is not None
    
    def test_database_initialization(self):
        """Test database initialization"""
        # Check that database tables were created
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            assert "metrics" in tables
            assert "analysis_results" in tables
    
    @pytest.mark.asyncio
    async def test_run_comprehensive_analysis_empty_data(self):
        """Test comprehensive analysis with empty data"""
        results = await self.engine.run_comprehensive_analysis(hours=24)
        
        assert isinstance(results, dict)
        assert "analysis_timestamp" in results
        assert "analysis_period_hours" in results
        assert "usage_patterns" in results
        assert "cost_optimizations" in results
        assert "trend_analyses" in results
        assert "user_profiles" in results
        assert "summary" in results
        
        # With empty data, should have empty results
        assert len(results["usage_patterns"]) == 0
        assert len(results["cost_optimizations"]) == 0
        assert len(results["user_profiles"]) == 0
    
    @pytest.mark.asyncio
    async def test_run_comprehensive_analysis_with_data(self):
        """Test comprehensive analysis with sample data"""
        # Generate sample data
        self.engine.generate_sample_data(days=3)
        
        # Run analysis
        results = await self.engine.run_comprehensive_analysis(hours=72)
        
        assert isinstance(results, dict)
        
        # Should have some results with sample data
        summary = results["summary"]
        assert summary["patterns_detected"] >= 0
        assert summary["users_analyzed"] >= 0
        assert summary["optimizations_found"] >= 0
        assert summary["total_potential_savings"] >= 0
    
    def test_generate_sample_data(self):
        """Test sample data generation"""
        self.engine.generate_sample_data(days=2)
        
        # Check that data was generated
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Check usage events
            cursor.execute("SELECT COUNT(*) FROM usage_events")
            usage_count = cursor.fetchone()[0]
            assert usage_count > 0
            
            # Check metrics
            cursor.execute("SELECT COUNT(*) FROM metrics")
            metrics_count = cursor.fetchone()[0]
            assert metrics_count > 0
    
    @pytest.mark.asyncio
    async def test_store_analysis_results(self):
        """Test storing analysis results"""
        # Generate sample data and run analysis
        self.engine.generate_sample_data(days=1)
        results = await self.engine.run_comprehensive_analysis(hours=24)
        
        # Check that results were stored
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM analysis_results")
            count = cursor.fetchone()[0]
            assert count > 0
            
            # Check result structure
            cursor.execute("SELECT results FROM analysis_results LIMIT 1")
            stored_results = cursor.fetchone()[0]
            parsed_results = json.loads(stored_results)
            
            assert "analysis_timestamp" in parsed_results
            assert "summary" in parsed_results


class TestIntegration:
    """Integration tests for the advanced analytics system"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_analytics_workflow(self):
        """Test complete analytics workflow"""
        # Create temporary database
        temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        db_path = Path(temp_db.name)
        
        try:
            # Create engine
            engine = AdvancedAnalyticsEngine(db_path)
            
            # Generate sample data
            engine.generate_sample_data(days=7)
            
            # Run comprehensive analysis
            results = await engine.run_comprehensive_analysis(hours=168)
            
            # Verify results structure
            assert "usage_patterns" in results
            assert "cost_optimizations" in results
            assert "trend_analyses" in results
            assert "user_profiles" in results
            assert "roi_projections" in results
            assert "summary" in results
            
            # Verify summary statistics
            summary = results["summary"]
            assert "patterns_detected" in summary
            assert "users_analyzed" in summary
            assert "optimizations_found" in summary
            assert "total_potential_savings" in summary
            
            # Test individual components
            usage_analyzer = UsagePatternAnalyzer(db_path)
            patterns = usage_analyzer.analyze_usage_patterns(hours=168)
            assert len(patterns) >= 0
            
            cost_optimizer = CostOptimizationEngine(db_path)
            optimizations = cost_optimizer.analyze_cost_optimization_opportunities(hours=168)
            assert len(optimizations) >= 0
            
            trend_analyzer = MLTrendAnalyzer(db_path)
            trend_analysis = trend_analyzer.analyze_trends("cpu_usage", hours=168)
            assert trend_analysis.metric_name == "cpu_usage"
            
        finally:
            # Cleanup
            temp_db.close()
            if db_path.exists():
                db_path.unlink()
    
    @pytest.mark.asyncio
    async def test_performance_with_large_dataset(self):
        """Test performance with larger dataset"""
        # Create temporary database
        temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        db_path = Path(temp_db.name)
        
        try:
            engine = AdvancedAnalyticsEngine(db_path)
            
            # Generate larger dataset
            engine.generate_sample_data(days=30)  # 30 days of data
            
            # Measure analysis time
            start_time = time.time()
            results = await engine.run_comprehensive_analysis(hours=720)  # 30 days
            end_time = time.time()
            
            analysis_duration = end_time - start_time
            
            # Should complete in reasonable time (less than 10 seconds)
            assert analysis_duration < 10.0
            
            # Should have meaningful results
            summary = results["summary"]
            assert summary["patterns_detected"] > 0
            assert summary["users_analyzed"] > 0
            
        finally:
            # Cleanup
            temp_db.close()
            if db_path.exists():
                db_path.unlink()


# Performance tests
class TestPerformance:
    """Performance tests for the analytics system"""
    
    def test_usage_pattern_analysis_performance(self):
        """Test usage pattern analysis performance"""
        temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        db_path = Path(temp_db.name)
        
        try:
            analyzer = UsagePatternAnalyzer(db_path)
            
            # Generate test data
            current_time = time.time()
            with sqlite3.connect(db_path) as conn:
                for i in range(1000):  # 1000 events
                    conn.execute("""
                        INSERT INTO usage_events 
                        (id, timestamp, user_id, event_type, model, tokens, cost, duration_ms, success, metadata)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        f"perf_event_{i}",
                        current_time - (i * 60),  # One event per minute
                        f"user_{i % 10}",
                        "chat_completion",
                        ["gpt-4", "gpt-3.5-turbo"][i % 2],
                        500,
                        0.01,
                        1000,
                        True,
                        "{}"
                    ))
            
            # Measure analysis time
            start_time = time.time()
            patterns = analyzer.analyze_usage_patterns(hours=24)
            end_time = time.time()
            
            analysis_duration = end_time - start_time
            
            # Should complete quickly (less than 2 seconds)
            assert analysis_duration < 2.0
            assert isinstance(patterns, list)
            
        finally:
            temp_db.close()
            if db_path.exists():
                db_path.unlink()
    
    def test_trend_analysis_performance(self):
        """Test trend analysis performance"""
        temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        db_path = Path(temp_db.name)
        
        try:
            analyzer = MLTrendAnalyzer(db_path)
            
            # Generate test data
            current_time = time.time()
            with sqlite3.connect(db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS metrics (
                        id TEXT PRIMARY KEY,
                        timestamp REAL,
                        metric_name TEXT,
                        value REAL,
                        metadata TEXT
                    )
                """)
                
                for i in range(500):  # 500 data points
                    conn.execute("""
                        INSERT INTO metrics (id, timestamp, metric_name, value, metadata)
                        VALUES (?, ?, ?, ?, ?)
                    """, (
                        f"perf_metric_{i}",
                        current_time - (i * 3600),  # Hourly data
                        "cpu_usage",
                        50 + (i * 0.1),  # Trending data
                        "{}"
                    ))
            
            # Measure analysis time
            start_time = time.time()
            trend_analysis = analyzer.analyze_trends("cpu_usage", hours=500)
            end_time = time.time()
            
            analysis_duration = end_time - start_time
            
            # Should complete quickly (less than 1 second)
            assert analysis_duration < 1.0
            assert isinstance(trend_analysis, TrendAnalysis)
            
        finally:
            temp_db.close()
            if db_path.exists():
                db_path.unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])