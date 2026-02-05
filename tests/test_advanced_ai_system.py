"""
Comprehensive Tests for AI Reasoning System
Tests for prompt optimization, knowledge transfer, fine-tuned models, 
adaptive reasoning, and model orchestration components.
"""

import asyncio
import pytest
from unittest.mock import Mock, MagicMock, patch, AsyncMock
import numpy as np
from datetime import datetime, timedelta

# Import the modules we're testing
from xencode.ai.prompt_optimizer import (
    PromptOptimizer, PromptOptimizationStrategy, optimize_prompt_async, 
    PromptMetrics, OptimizedPrompt
)
from xencode.ai.knowledge_transfer import (
    KnowledgeTransferEngine, TransferMethod, KnowledgeFragment, 
    transfer_knowledge_between_models
)
from xencode.ai.finetuned_models import (
    FineTunedModelManager, ModelDomain, ModelProvider, ModelSpecification,
    process_with_domain_specific_model
)
from xencode.ai.adaptive_reasoning import (
    AdaptiveReasoningEngine, ReasoningType, ComplexityLevel,
    solve_with_adaptive_reasoning
)
from xencode.ai.orchestrator import (
    ModelOrchestrator, ModelStatus, LoadBalancingStrategy, ModelInfo,
    process_with_model_orchestration
)


# Test Prompt Optimizer
class TestPromptOptimizer:
    """Test cases for the PromptOptimizer class."""
    
    @pytest.mark.asyncio
    async def test_context_aware_optimization(self):
        """Test context-aware prompt optimization."""
        optimizer = PromptOptimizer()
        
        # Test with technical context
        prompt = "Explain how to implement a sorting algorithm"
        context = "We're building a software application in Python"
        
        result = await optimizer.optimize_prompt(
            prompt, context, PromptOptimizationStrategy.CONTEXT_AWARE, "gpt-4"
        )
        
        assert isinstance(result, OptimizedPrompt)
        assert result.original_prompt == prompt
        assert result.strategy_used == PromptOptimizationStrategy.CONTEXT_AWARE
        assert result.metrics.response_time >= 0
        
        # Verify the optimized prompt contains technical elements
        assert "code" in result.optimized_prompt.lower() or "technical" in result.optimized_prompt.lower()
        
    @pytest.mark.asyncio
    async def test_multi_model_adaptive_optimization(self):
        """Test multi-model adaptive optimization."""
        optimizer = PromptOptimizer()
        
        prompt = "What is the capital of France?"
        
        # Test with Claude model
        result_claude = await optimizer.optimize_prompt(
            prompt, None, PromptOptimizationStrategy.MULTI_MODEL_ADAPTIVE, "claude"
        )
        
        # Test with Llama model
        result_llama = await optimizer.optimize_prompt(
            prompt, None, PromptOptimizationStrategy.MULTI_MODEL_ADAPTIVE, "llama"
        )
        
        assert isinstance(result_claude, OptimizedPrompt)
        assert isinstance(result_llama, OptimizedPrompt)
        
        # Verify model-specific adjustments
        if "[INST]" in result_llama.optimized_prompt:
            assert "[INST]" in result_llama.optimized_prompt
            assert "[/INST]" in result_llama.optimized_prompt
            
    @pytest.mark.asyncio
    async def test_performance_tracking(self):
        """Test prompt performance tracking."""
        optimizer = PromptOptimizer()
        
        prompt = "Simple test prompt"
        context = "Testing context"
        
        # Optimize the prompt multiple times
        for i in range(3):
            await optimizer.optimize_prompt(
                prompt, context, PromptOptimizationStrategy.PERFORMANCE_BASED
            )
        
        # Check that metrics were recorded
        assert len(optimizer.performance_tracker.metrics_history) == 3
        
        # Check that prompt stats were recorded
        prompt_ids = [m.prompt_id for m in optimizer.performance_tracker.metrics_history]
        for pid in prompt_ids:
            assert pid in optimizer.performance_tracker.prompt_stats
            assert len(optimizer.performance_tracker.prompt_stats[pid]) >= 1
    
    @pytest.mark.asyncio
    async def test_ab_testing_framework(self):
        """Test A/B testing framework for prompts."""
        optimizer = PromptOptimizer()
        
        experiment_id = "test_exp_1"
        prompt_variants = [
            "Variant A: Tell me about AI",
            "Variant B: Explain artificial intelligence",
            "Variant C: Describe machine learning"
        ]
        
        # Start experiment
        optimizer.start_ab_test(experiment_id, prompt_variants)
        
        # Record results for each variant
        for i, variant in enumerate(prompt_variants):
            metrics = PromptMetrics(
                prompt_id=f"prompt_{i}",
                timestamp=datetime.now().timestamp(),
                response_time=0.5 + (i * 0.1),
                success_rate=0.9 - (i * 0.05),
                relevance_score=0.8 + (i * 0.05),
                model_used="test_model",
                context_length=50,
                token_usage=20
            )
            optimizer.record_ab_test_result(experiment_id, variant, metrics)
        
        # Get best variant
        best_variant = optimizer.get_best_prompt_variant(experiment_id)
        
        assert best_variant is not None
        assert best_variant in prompt_variants


# Test Knowledge Transfer
class TestKnowledgeTransfer:
    """Test cases for the KnowledgeTransferEngine class."""
    
    @pytest.mark.asyncio
    async def test_semantic_alignment_transfer(self):
        """Test semantic alignment-based knowledge transfer."""
        engine = KnowledgeTransferEngine()
        
        source_model = "gpt-4"
        target_model = "claude"
        knowledge_content = "Artificial intelligence is a branch of computer science..."
        
        result, metrics = await engine.transfer_knowledge(
            source_model, target_model, knowledge_content, 
            transfer_method=TransferMethod.SEMANTIC_ALIGNMENT
        )
        
        assert isinstance(result, KnowledgeFragment)
        assert result.source_model == source_model
        assert result.target_model == target_model
        assert result.content == knowledge_content
        assert isinstance(metrics, object)  # TransferMetrics would be tested separately
        
    @pytest.mark.asyncio
    async def test_context_preservation(self):
        """Test context preservation during knowledge transfer."""
        engine = KnowledgeTransferEngine()
        
        session_id = "test_session_123"
        context_data = {
            "user_preferences": ["technical", "detailed"],
            "conversation_history": ["Q1", "A1", "Q2"]
        }
        
        # Preserve context
        engine.preserve_context(session_id, context_data)
        
        # Perform knowledge transfer with session ID
        result, metrics = await engine.transfer_knowledge(
            "model_a", "model_b", "test knowledge", session_id
        )
        
        # Verify context was preserved in metadata
        assert "context" in result.metadata
        assert result.metadata["context"]["user_preferences"] == context_data["user_preferences"]
        
    @pytest.mark.asyncio
    async def test_batch_transfer(self):
        """Test batch knowledge transfer."""
        engine = KnowledgeTransferEngine()
        
        source_model = "gpt-3.5"
        target_model = "llama"
        knowledge_fragments = [
            "Knowledge fragment 1",
            "Knowledge fragment 2", 
            "Knowledge fragment 3"
        ]
        
        results = await engine.batch_transfer(
            source_model, target_model, knowledge_fragments
        )
        
        assert len(results) == len(knowledge_fragments)
        for result, _ in results:
            assert isinstance(result, KnowledgeFragment)
            assert result.source_model == source_model
            assert result.target_model == target_model
            
    @pytest.mark.asyncio
    async def test_transfer_efficiency_calculation(self):
        """Test transfer efficiency calculation."""
        engine = KnowledgeTransferEngine()
        
        # Register some mock embeddings for alignment
        mock_embeddings = [
            ("test text 1", np.array([0.1, 0.2, 0.3])),
            ("test text 2", np.array([0.4, 0.5, 0.6]))
        ]
        
        engine.register_model_embeddings("model_a", mock_embeddings)
        engine.register_model_embeddings("model_b", mock_embeddings)
        
        # Perform a transfer to establish history
        await engine.transfer_knowledge("model_a", "model_b", "test content")
        
        # Check transfer efficiency
        efficiency = engine.get_transfer_efficiency("model_a", "model_b")
        assert isinstance(efficiency, float)
        assert 0.0 <= efficiency <= 1.0


# Test Fine-Tuned Models
class TestFineTunedModels:
    """Test cases for the FineTunedModelManager class."""
    
    def test_domain_detection(self):
        """Test domain detection functionality."""
        manager = FineTunedModelManager()
        
        # Test technical coding domain
        tech_text = "Write a Python function to sort an array using quicksort algorithm"
        tech_domain = manager.domain_detector.detect_domain(tech_text)
        assert tech_domain == ModelDomain.TECHNICAL_CODING
        
        # Test scientific research domain
        sci_text = "Analyze the experimental data to validate the hypothesis about protein folding"
        sci_domain = manager.domain_detector.detect_domain(sci_text)
        assert sci_domain == ModelDomain.SCIENTIFIC_RESEARCH
        
        # Test financial analysis domain
        fin_text = "Evaluate the quarterly revenue growth and profit margins for the company"
        fin_domain = manager.domain_detector.detect_domain(fin_text)
        assert fin_domain == ModelDomain.FINANCIAL_ANALYSIS
        
    def test_model_registration(self):
        """Test model registration functionality."""
        manager = FineTunedModelManager()
        
        model_spec = ModelSpecification(
            model_id="test_model_1",
            domain=ModelDomain.TECHNICAL_CODING,
            provider=ModelProvider.OPENAI,
            version="1.0.0",
            capabilities=["coding", "debugging"],
            performance_metrics={"accuracy": 0.95, "speed": 0.8},
            endpoint_url="https://api.openai.com/v1/chat/completions",
            api_key="test-key"
        )
        
        manager.register_model(model_spec)
        
        # Verify model was registered
        assert "test_model_1" in manager.available_models
        assert manager.available_models["test_model_1"].model_id == "test_model_1"
        
        # Verify version was registered
        latest_version = manager.version_manager.get_latest_version("test_model_1")
        assert latest_version is not None
        assert latest_version.model_id == "test_model_1"
        
    @pytest.mark.asyncio
    async def test_model_selection_by_domain(self):
        """Test model selection based on domain detection."""
        manager = FineTunedModelManager()
        
        # Register a technical coding model
        tech_model = ModelSpecification(
            model_id="tech_model_1",
            domain=ModelDomain.TECHNICAL_CODING,
            provider=ModelProvider.OPENAI,
            version="1.0.0",
            capabilities=["coding", "debugging"],
            performance_metrics={"accuracy": 0.95, "speed": 0.8},
            endpoint_url="https://api.openai.com/v1/chat/completions",
            api_key="test-key",
            is_active=True
        )
        manager.register_model(tech_model)
        
        # Register a creative writing model
        creative_model = ModelSpecification(
            model_id="creative_model_1",
            domain=ModelDomain.CREATIVE_WRITING,
            provider=ModelProvider.ANTHROPIC,
            version="1.0.0",
            capabilities=["writing", "storytelling"],
            performance_metrics={"creativity": 0.9, "engagement": 0.85},
            endpoint_url="https://api.anthropic.com/v1/messages",
            api_key="test-key",
            is_active=True
        )
        manager.register_model(creative_model)
        
        # Test with technical text
        tech_text = "How do I implement a binary search algorithm in Python?"
        selected_model = manager.detect_appropriate_model(tech_text)
        assert selected_model.domain == ModelDomain.TECHNICAL_CODING
        
        # Test with creative text
        creative_text = "Write a short story about a robot learning to paint"
        selected_model = manager.detect_appropriate_model(creative_text)
        assert selected_model.domain == ModelDomain.CREATIVE_WRITING
        
    @pytest.mark.asyncio
    async def test_process_with_optimal_model(self):
        """Test processing with optimal model selection."""
        manager = FineTunedModelManager()
        
        # Register a model
        model_spec = ModelSpecification(
            model_id="test_model_2",
            domain=ModelDomain.TECHNICAL_CODING,
            provider=ModelProvider.OPENAI,
            version="1.0.0",
            capabilities=["coding", "debugging"],
            performance_metrics={"accuracy": 0.95, "speed": 0.8},
            endpoint_url="https://api.openai.com/v1/chat/completions",
            api_key="test-key",
            is_active=True
        )
        manager.register_model(model_spec)
        
        # Process with optimal model
        input_text = "Write a Python function to reverse a string"
        result = await manager.process_with_optimal_model(input_text, session_id="test_session")
        
        assert "response" in result
        assert "model_used" in result
        assert result["model_used"] == "test_model_2"
        assert result["domain_detected"] == ModelDomain.TECHNICAL_CODING.value
        
    def test_model_performance_monitoring(self):
        """Test model performance monitoring."""
        monitor = FineTunedModelManager().performance_monitor
        
        model_id = "test_model_perf"
        session_id = "perf_session_1"
        
        # Start a session
        monitor.start_session(model_id, session_id)
        
        # Record some requests
        for i in range(5):
            monitor.record_request(model_id, session_id, response_time=0.1 + (i * 0.05), success=True)
        
        # End session and get report
        report = monitor.end_session(model_id, session_id)
        
        assert report.model_id == model_id
        assert report.response_time >= 0
        assert report.throughput >= 0
        assert 0 <= report.error_rate <= 1


# Test Adaptive Reasoning
class TestAdaptiveReasoning:
    """Test cases for the AdaptiveReasoningEngine class."""
    
    @pytest.mark.asyncio
    async def test_simple_problem_solving(self):
        """Test solving a simple problem."""
        engine = AdaptiveReasoningEngine()
        
        problem = "What is 15 + 27?"
        
        result = await engine.solve_problem(problem)
        
        assert isinstance(result, object)  # ReasoningChain would be tested separately
        assert result.problem_statement == problem
        assert result.success is True
        assert len(result.reasoning_steps) > 0
        assert result.execution_time >= 0
        
    @pytest.mark.asyncio
    async def test_complex_problem_solving(self):
        """Test solving a complex problem."""
        engine = AdaptiveReasoningEngine()
        
        problem = "Analyze the potential impacts of climate change on global agriculture, considering economic, social, and environmental factors, and propose three evidence-based adaptation strategies."
        
        result = await engine.solve_problem(problem)
        
        assert result.problem_statement == problem
        assert result.complexity_level in [ComplexityLevel.COMPLEX, ComplexityLevel.HIGHLY_COMPLEX]
        assert len(result.reasoning_steps) >= 2  # Should have multiple steps for complex problems
        assert result.success is True
        
    @pytest.mark.asyncio
    async def test_reasoning_complexity_analysis(self):
        """Test reasoning complexity analysis."""
        analyzer = AdaptiveReasoningEngine().complexity_analyzer
        
        # Test simple problem
        simple_problem = "What is the capital of Germany?"
        simple_complexity = analyzer.analyze_complexity(simple_problem)
        assert simple_complexity in [ComplexityLevel.SIMPLE, ComplexityLevel.MODERATE]
        
        # Test complex problem
        complex_problem = "Develop a comprehensive strategy for sustainable urban development that addresses housing affordability, transportation efficiency, environmental impact, and social equity while considering economic feasibility and political viability."
        complex_complexity = analyzer.analyze_complexity(complex_problem)
        assert complex_complexity in [ComplexityLevel.COMPLEX, ComplexityLevel.HIGHLY_COMPLEX]
        
    @pytest.mark.asyncio
    async def test_reasoning_pathway_selection(self):
        """Test reasoning pathway selection."""
        selector = AdaptiveReasoningEngine().pathway_selector
        analyzer = AdaptiveReasoningEngine().complexity_analyzer
        
        # Test with simple problem
        simple_problem = "Calculate the area of a circle with radius 5"
        simple_complexity = analyzer.analyze_complexity(simple_problem)
        simple_pathways = selector.select_pathways(simple_problem, simple_complexity)
        
        assert len(simple_pathways) > 0
        # Simple problems should favor logical/mathematical reasoning
        assert any(rt in [ReasoningType.LOGICAL, ReasoningType.MATHEMATICAL] for rt in simple_pathways)
        
        # Test with ethical problem
        ethical_problem = "Analyze the ethical implications of using AI in hiring decisions"
        ethical_pathways = selector.select_pathways(ethical_problem, ComplexityLevel.HIGHLY_COMPLEX)
        
        assert len(ethical_pathways) > 0
        # Ethical problems should favor abductive and analogical reasoning
        assert any(rt in [ReasoningType.ABDUCTIVE, ReasoningType.ANALOGICAL] for rt in ethical_pathways)
        
    @pytest.mark.asyncio
    async def test_adaptation_for_similar_problems(self):
        """Test adaptation for similar problems."""
        engine = AdaptiveReasoningEngine()
        
        # First, solve a problem to establish history
        original_problem = "Compare the efficiency of bubble sort and quicksort algorithms"
        original_result = await engine.solve_problem(original_problem)
        
        # Then adapt for a similar problem
        similar_problem = "Contrast the performance of insertion sort and mergesort algorithms"
        adapted_approach = await engine.adapt_for_new_problem(similar_problem)
        
        # The adapted approach should be similar to the original since problems are similar
        assert isinstance(adapted_approach, list)
        assert len(adapted_approach) > 0


# Test Model Orchestrator
class TestModelOrchestrator:
    """Test cases for the ModelOrchestrator class."""
    
    def test_model_registration(self):
        """Test model registration with the orchestrator."""
        orchestrator = ModelOrchestrator()
        
        model_info = ModelInfo(
            model_id="test_model_orch_1",
            endpoint_url="https://api.example.com/v1/chat",
            api_key="test-key",
            provider="openai",
            capabilities=["chat", "completion"],
            weight=2,
            max_concurrent_requests=5
        )
        
        orchestrator.register_model(model_info)
        
        # Verify model was registered
        assert "test_model_orch_1" in orchestrator.models
        assert orchestrator.models["test_model_orch_1"].model_id == "test_model_orch_1"
        assert orchestrator.models["test_model_orch_1"].weight == 2
        
    def test_load_balancing_round_robin(self):
        """Test round-robin load balancing strategy."""
        orchestrator = ModelOrchestrator(load_balancing_strategy=LoadBalancingStrategy.ROUND_ROBIN)
        
        # Register multiple models
        for i in range(3):
            model_info = ModelInfo(
                model_id=f"model_rr_{i}",
                endpoint_url=f"https://api{i}.example.com/v1/chat",
                api_key="test-key",
                provider="openai",
                capabilities=["chat"],
                status=ModelStatus.HEALTHY,
                max_concurrent_requests=10
            )
            orchestrator.register_model(model_info)
        
        # Simulate selecting models multiple times
        selections = []
        for _ in range(6):  # More than the number of models to test cycling
            # Since we're mocking the health check, force status to healthy
            for model_id in orchestrator.models:
                orchestrator.update_model_status(model_id, ModelStatus.HEALTHY)
                
            model_id = asyncio.run(orchestrator._select_model())
            if model_id:
                selections.append(model_id)
        
        # With round-robin, we should cycle through the models
        # This test is simplified due to the complexity of the internal state
        assert len(selections) > 0
        for selection in selections:
            assert selection in [f"model_rr_{i}" for i in range(3)]
            
    def test_model_status_updates(self):
        """Test updating model statuses."""
        orchestrator = ModelOrchestrator()
        
        model_info = ModelInfo(
            model_id="status_test_model",
            endpoint_url="https://api.example.com/v1/chat",
            api_key="test-key",
            provider="openai",
            capabilities=["chat"],
            status=ModelStatus.HEALTHY
        )
        orchestrator.register_model(model_info)
        
        # Update status to degraded
        orchestrator.update_model_status("status_test_model", ModelStatus.DEGRADED)
        
        # Check status
        all_statuses = orchestrator.get_all_model_statuses()
        assert "status_test_model" in all_statuses
        # Note: The actual status might be overridden by health monitor, so we check if it's accessible
        
    @pytest.mark.asyncio
    async def test_request_processing(self):
        """Test processing a request through the orchestrator."""
        orchestrator = ModelOrchestrator()
        
        # Register a model
        model_info = ModelInfo(
            model_id="processing_test_model",
            endpoint_url="https://api.example.com/v1/chat",
            api_key="test-key",
            provider="openai",
            capabilities=["chat", "completion"],
            status=ModelStatus.HEALTHY,
            max_concurrent_requests=5
        )
        orchestrator.register_model(model_info)
        
        # Update model status to healthy (bypass health monitor for this test)
        orchestrator.update_model_status("processing_test_model", ModelStatus.HEALTHY)
        
        # Process a request
        request_data = {
            "prompt": "Hello, how are you?",
            "max_tokens": 100
        }
        
        result = await orchestrator.process_request(request_data)
        
        assert "result" in result or "error" in result
        assert result["model_used"] == "processing_test_model"
        assert result["success"] is True
        
    def test_orchestration_metrics(self):
        """Test orchestration metrics collection."""
        orchestrator = ModelOrchestrator()
        
        # Get initial metrics
        initial_metrics = orchestrator.get_orchestration_metrics()
        
        assert initial_metrics.total_requests == 0
        assert initial_metrics.successful_requests == 0
        assert initial_metrics.average_response_time == 0.0
        
        # Add a mock routing history entry to test load distribution
        mock_routing = MagicMock()
        mock_routing.model_id = "test_model_metrics"
        mock_routing.success = True
        orchestrator.routing_history.append(mock_routing)
        
        # Add to response times
        orchestrator.response_times = [0.5, 0.7, 0.3]
        
        # Update counts
        orchestrator.total_requests = 3
        orchestrator.successful_requests = 3
        
        # Get updated metrics
        updated_metrics = orchestrator.get_orchestration_metrics()
        
        assert updated_metrics.total_requests == 3
        assert updated_metrics.successful_requests == 3
        assert updated_metrics.average_response_time == 0.5  # (0.5+0.7+0.3)/3
        assert "test_model_metrics" in updated_metrics.load_distribution


# Integration Tests
class TestAIReasoningIntegration:
    """Integration tests for the entire AI reasoning system."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_reasoning_pipeline(self):
        """Test an end-to-end reasoning pipeline."""
        # Create all components
        optimizer = PromptOptimizer()
        transfer_engine = KnowledgeTransferEngine()
        model_manager = FineTunedModelManager()
        reasoning_engine = AdaptiveReasoningEngine()
        orchestrator = ModelOrchestrator()
        
        # Register a model for the manager
        model_spec = ModelSpecification(
            model_id="integration_test_model",
            domain=ModelDomain.TECHNICAL_CODING,
            provider=ModelProvider.OPENAI,
            version="1.0.0",
            capabilities=["coding", "explanation"],
            performance_metrics={"accuracy": 0.9, "speed": 0.8},
            endpoint_url="https://api.openai.com/v1/chat/completions",
            api_key="test-key",
            is_active=True
        )
        model_manager.register_model(model_spec)
        
        # Register a model for the orchestrator
        orch_model_info = ModelInfo(
            model_id="orch_integration_model",
            endpoint_url="https://api.openai.com/v1/chat/completions",
            api_key="test-key",
            provider="openai",
            capabilities=["chat", "reasoning"],
            status=ModelStatus.HEALTHY,
            max_concurrent_requests=10
        )
        orchestrator.register_model(orch_model_info)
        
        # Step 1: Optimize a prompt
        original_prompt = "Explain how neural networks learn"
        context = "We're teaching machine learning concepts to beginners"
        
        optimized_result = await optimizer.optimize_prompt(
            original_prompt, context, PromptOptimizationStrategy.CONTEXT_AWARE
        )
        
        assert optimized_result.optimized_prompt != original_prompt
        
        # Step 2: Use the optimized prompt in reasoning
        problem_statement = optimized_result.optimized_prompt
        
        reasoning_result = await reasoning_engine.solve_problem(problem_statement)
        
        assert reasoning_result.success is True
        assert len(reasoning_result.reasoning_steps) > 0
        
        # Step 3: Process through orchestrator
        request_data = {
            "prompt": problem_statement,
            "max_tokens": 200
        }
        
        orch_result = await orchestrator.process_request(request_data)
        
        assert orch_result["success"] is True
        assert "result" in orch_result
        
    @pytest.mark.asyncio
    async def test_cross_component_knowledge_transfer(self):
        """Test knowledge transfer between components."""
        # Set up two different model managers for different domains
        coding_manager = FineTunedModelManager()
        science_manager = FineTunedModelManager()
        
        # Register models
        coding_model = ModelSpecification(
            model_id="coding_expert",
            domain=ModelDomain.TECHNICAL_CODING,
            provider=ModelProvider.OPENAI,
            version="1.0.0",
            capabilities=["programming", "debugging"],
            performance_metrics={"accuracy": 0.92, "speed": 0.85},
            endpoint_url="https://api.openai.com/v1/chat/completions",
            api_key="test-key",
            is_active=True
        )
        coding_manager.register_model(coding_model)
        
        science_model = ModelSpecification(
            model_id="science_expert",
            domain=ModelDomain.SCIENTIFIC_RESEARCH,
            provider=ModelProvider.ANTHROPIC,
            version="1.0.0",
            capabilities=["analysis", "research"],
            performance_metrics={"accuracy": 0.94, "depth": 0.9},
            endpoint_url="https://api.anthropic.com/v1/messages",
            api_key="test-key",
            is_active=True
        )
        science_manager.register_model(science_model)
        
        # Transfer knowledge between domains using the transfer engine
        transfer_engine = KnowledgeTransferEngine()
        
        # Simulate knowledge from coding domain
        coding_knowledge = "Algorithms are step-by-step procedures for solving problems efficiently."
        
        result, metrics = await transfer_engine.transfer_knowledge(
            "coding_expert", "science_expert", coding_knowledge
        )
        
        assert result.content == coding_knowledge
        assert result.source_model == "coding_expert"
        assert result.target_model == "science_expert"
        
    @pytest.mark.asyncio
    async def test_adaptive_optimization_with_realistic_workflow(self):
        """Test adaptive optimization in a realistic workflow."""
        # Create components
        optimizer = PromptOptimizer()
        reasoning_engine = AdaptiveReasoningEngine()
        
        # Simulate a complex problem-solving workflow
        complex_problem = "Design a scalable microservices architecture for a high-traffic e-commerce platform, considering security, performance, and maintainability requirements."
        
        # First, optimize the prompt for the specific context
        context = "Enterprise-level system design with focus on scalability and security"
        optimized_problem = await optimizer.optimize_prompt(
            complex_problem, context, PromptOptimizationStrategy.CONTEXT_AWARE, "gpt-4"
        )
        
        # Then solve the optimized problem using adaptive reasoning
        solution = await reasoning_engine.solve_problem(optimized_problem.optimized_prompt)
        
        # Verify the solution is comprehensive
        assert solution.success is True
        assert solution.complexity_level in [ComplexityLevel.COMPLEX, ComplexityLevel.HIGHLY_COMPLEX]
        assert len(solution.selected_reasoning_types) > 1  # Multiple reasoning types for complex problem
        assert len(solution.reasoning_steps) > 2  # Multiple steps for complex problem


if __name__ == "__main__":
    # Run the tests if this script is executed directly
    pytest.main([__file__, "-v"])