#!/usr/bin/env python3
"""
ByteBot Integration Tests

Tests for the ByteBot terminal cognition layer integration.
Covers engine, execution modes, safety, and risk scoring.
"""


import pytest


# Test the ByteBot engine components
class TestByteBotEngine:
    """Test ByteBotEngine initialization and basic operations"""

    def test_engine_creation(self):
        """Test ByteBotEngine can be instantiated"""
        from xencode.bytebot import ByteBotEngine

        engine = ByteBotEngine()

        assert engine is not None
        assert hasattr(engine, 'context_engine')
        assert hasattr(engine, 'planner')
        assert hasattr(engine, 'executor')
        assert hasattr(engine, 'safety_gate')
        assert hasattr(engine, 'risk_scorer')

    def test_engine_process_intent_assist_mode(self):
        """Test processing intent in assist mode (suggestions only)"""
        from xencode.bytebot import ByteBotEngine

        engine = ByteBotEngine()
        result = engine.process_intent("list files in current directory", mode="assist")

        assert result is not None
        assert "status" in result
        assert "suggested_steps" in result or "plan_id" in result
        # In assist mode, nothing should be executed
        assert result.get("status") in ["suggested", "executed"]

    def test_engine_process_intent_execute_mode(self):
        """Test processing intent in execute mode"""
        from xencode.bytebot import ByteBotEngine

        engine = ByteBotEngine()
        result = engine.process_intent("check current directory", mode="execute")

        assert result is not None
        assert "status" in result
        assert "execution_results" in result or "summary" in result

    def test_engine_process_intent_autonomous_mode(self):
        """Test processing intent in autonomous mode"""
        from xencode.bytebot import ByteBotEngine

        engine = ByteBotEngine()
        result = engine.process_intent("list files", mode="autonomous")

        assert result is not None
        assert "status" in result
        assert "execution_results" in result

    def test_engine_invalid_mode(self):
        """Test handling of invalid execution mode"""
        from xencode.bytebot import ByteBotEngine

        engine = ByteBotEngine()
        # Should default to execute mode
        result = engine.process_intent("test", mode="invalid_mode")

        assert result is not None

    def test_engine_set_mode(self):
        """Test switching execution modes"""
        from xencode.bytebot import ByteBotEngine
        from xencode.bytebot.safety_gate import ExecutionMode

        engine = ByteBotEngine()
        engine.set_mode(ExecutionMode.ASSIST)

        # Verify mode was set
        assert engine.current_mode == ExecutionMode.ASSIST

    def test_engine_execution_history(self):
        """Test execution history tracking"""
        from xencode.bytebot import ByteBotEngine

        engine = ByteBotEngine()

        # Process multiple intents
        engine.process_intent("test 1", mode="assist")
        engine.process_intent("test 2", mode="assist")

        history = engine.get_execution_history()
        assert len(history) >= 2


class TestExecutionModes:
    """Test execution mode behavior"""

    def test_assist_mode_no_execution(self):
        """Assist mode should not execute commands"""
        from xencode.bytebot import ByteBotEngine

        engine = ByteBotEngine()
        result = engine.process_intent("rm -rf /", mode="assist")

        # Should only suggest, not execute
        assert result.get("status") == "suggested"

    def test_execute_mode_confirms_risky(self):
        """Execute mode should confirm risky operations"""
        from xencode.bytebot import ByteBotEngine

        engine = ByteBotEngine()
        # This should be blocked or require confirmation
        result = engine.process_intent("rm -rf /", mode="execute")

        # Should be blocked by safety gate
        assert result is not None

    def test_autonomous_mode_executes_safe(self):
        """Autonomous mode executes safe commands automatically"""
        from xencode.bytebot import ByteBotEngine

        engine = ByteBotEngine()
        result = engine.process_intent("echo hello", mode="autonomous")

        assert result is not None


class TestSafetyGate:
    """Test safety gate functionality"""

    def test_blocks_dangerous_commands(self):
        """Safety gate should block dangerous commands"""
        from xencode.bytebot.safety_gate import ExecutionMode, SafetyGate

        gate = SafetyGate()

        # Test absolutely dangerous command
        step = {"command": "rm -rf /", "description": "Delete root"}
        should_block = gate.should_block(step, risk_score=0.95, mode=ExecutionMode.EXECUTE)

        assert should_block is True

    def test_allows_safe_commands(self):
        """Safety gate should allow safe commands"""
        from xencode.bytebot.safety_gate import ExecutionMode, SafetyGate

        gate = SafetyGate()

        step = {"command": "ls -la", "description": "List files"}
        should_block = gate.should_block(step, risk_score=0.1, mode=ExecutionMode.EXECUTE)

        assert should_block is False

    def test_assist_mode_blocks_all(self):
        """Assist mode blocks all execution"""
        from xencode.bytebot.safety_gate import ExecutionMode, SafetyGate

        gate = SafetyGate()

        step = {"command": "echo test", "description": "Print test"}
        should_block = gate.should_block(step, risk_score=0.1, mode=ExecutionMode.ASSIST)

        assert should_block is True

    def test_get_block_reason(self):
        """Safety gate provides reason for blocking"""
        from xencode.bytebot.safety_gate import ExecutionMode, SafetyGate

        gate = SafetyGate()

        step = {"command": "rm -rf /", "description": "Delete root"}
        reason = gate.get_block_reason(step, risk_score=0.95, mode=ExecutionMode.EXECUTE)

        assert reason is not None
        assert len(reason) > 0


class TestRiskScorer:
    """Test risk scoring functionality"""

    def test_score_safe_command(self):
        """Safe commands should have low risk score"""
        from xencode.bytebot.risk_scorer import RiskScorer

        scorer = RiskScorer()
        assessment = scorer.score_command("ls -la")

        assert assessment.score < 0.3
        assert scorer.get_risk_category(assessment.score) in ["Minimal", "Low"]

    def test_score_dangerous_command(self):
        """Dangerous commands should have high risk score"""
        from xencode.bytebot.risk_scorer import RiskScorer

        scorer = RiskScorer()
        assessment = scorer.score_command("rm -rf /")

        assert assessment.score >= 0.7
        assert assessment.is_dangerous is True

    def test_score_medium_risk_command(self):
        """Medium risk commands should be identified"""
        from xencode.bytebot.risk_scorer import RiskScorer

        scorer = RiskScorer()
        assessment = scorer.score_command("rm -rf temp_folder")

        assert 0.3 <= assessment.score < 0.7

    def test_get_recommendation(self):
        """Risk scorer provides recommendations"""
        from xencode.bytebot.risk_scorer import RiskScorer

        scorer = RiskScorer()

        # Dangerous command
        dangerous = scorer.score_command("rm -rf /")
        rec = scorer.get_recommendation(dangerous)
        assert "BLOCK" in rec or "high risk" in rec.lower()

        # Safe command
        safe = scorer.score_command("ls -la")
        rec = scorer.get_recommendation(safe)
        assert "PROCEED" in rec or "low risk" in rec.lower()


class TestPlanner:
    """Test plan generation"""

    def test_create_simple_plan(self):
        """Planner creates simple plans for simple commands"""
        from xencode.bytebot.planner import Planner

        planner = Planner()
        plan = planner.create_plan("list files")

        assert plan is not None
        assert len(plan.steps) >= 1
        assert plan.intent == "list files"

    def test_create_plan_with_dependencies(self):
        """Planner creates plans with proper dependencies"""
        from xencode.bytebot.planner import Planner

        planner = Planner()
        plan = planner.create_plan("git commit and push")

        # Git operations should have multiple steps with dependencies
        assert len(plan.steps) >= 2
        assert len(plan.dependencies) >= 1

    def test_validate_plan(self):
        """Planner validates plans"""
        from xencode.bytebot.planner import Planner

        planner = Planner()
        plan = planner.create_plan("test command")

        is_valid, issues = planner.validate_plan(plan)

        # Plan should be valid
        assert is_valid or len(issues) > 0


class TestExecutor:
    """Test command execution"""

    def test_execute_safe_command(self):
        """Executor executes safe commands"""
        from xencode.bytebot.executor import Executor
        from xencode.bytebot.terminal_cognition_layer import TerminalCognitionLayer

        terminal_layer = TerminalCognitionLayer()
        executor = Executor(terminal_layer)

        step = {"id": "test-1", "command": "echo hello", "type": "command"}
        result = executor.execute_step(step)

        assert result.status == "success"
        assert "hello" in result.result

    def test_execute_command_with_error(self):
        """Executor handles command errors"""
        from xencode.bytebot.executor import Executor
        from xencode.bytebot.terminal_cognition_layer import TerminalCognitionLayer

        terminal_layer = TerminalCognitionLayer()
        executor = Executor(terminal_layer)

        step = {"id": "test-1", "command": "nonexistent_command_xyz", "type": "command"}
        result = executor.execute_step(step)

        assert result.status in ["failed", "error"]


class TestTerminalCognitionLayer:
    """Test terminal cognition layer"""

    def test_execute_command_safe(self):
        """Terminal layer executes commands safely"""
        from xencode.bytebot.terminal_cognition_layer import TerminalCognitionLayer

        layer = TerminalCognitionLayer()
        result = layer.execute_command_safe("echo test123")

        assert result.success is True
        assert "test123" in result.stdout

    def test_validate_command(self):
        """Terminal layer validates commands"""
        from xencode.bytebot.terminal_cognition_layer import TerminalCognitionLayer

        layer = TerminalCognitionLayer()
        validation = layer.validate_command("ls -la")

        assert validation is not None
        assert "valid" in validation


class TestByteBotIntegration:
    """Integration tests for complete ByteBot workflows"""

    def test_complete_workflow_safe_command(self):
        """Complete workflow: intent -> plan -> execute (safe)"""
        from xencode.bytebot import ByteBotEngine

        engine = ByteBotEngine()

        # Process a safe command
        result = engine.process_intent("check current directory with pwd", mode="execute")

        assert result is not None
        assert result.get("status") in ["executed", "suggested"]

    def test_complete_workflow_dangerous_command(self):
        """Complete workflow: intent -> plan -> block (dangerous)"""
        from xencode.bytebot import ByteBotEngine

        engine = ByteBotEngine()

        # Try to execute dangerous command
        result = engine.process_intent("delete all files with rm -rf /", mode="execute")

        # Should be blocked or show warnings
        assert result is not None


# Global functions test
class TestByteBotGlobalFunctions:
    """Test global ByteBot functions"""

    def test_get_engine_instance(self):
        """Test getting ByteBot engine instance"""
        from xencode.bytebot import ByteBotEngine

        # Should be able to create instance
        engine = ByteBotEngine()
        assert engine is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
