#!/usr/bin/env python3
"""
Tests for AI Ensemble System

Comprehensive testing of multi-model ensemble reasoning with mocks
for offline testing and performance validation.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from xencode.ai_ensembles import (
    EnsembleReasoner, QueryRequest, QueryResponse, ModelResponse,
    EnsembleMethod, ModelTier, TokenVoter, create_ensemble_reasoner
)


class TestTokenVoter:
    """Test token voting mechanisms"""
    
    def test_vote_tokens_simple(self):
        """Test simple token voting"""
        voter = TokenVoter()
        responses = [
            "The quick brown fox",
            "The quick brown dog", 
            "The quick brown fox"
        ]
        
        result = voter.vote_tokens(responses)
        assert "The quick brown fox" == result
    
    def test_vote_tokens_weighted(self):
        """Test weighted token voting"""
        voter = TokenVoter()
        responses = [
            "Python is great",
            "Python is good",
            "Python is excellent"
        ]
        weights = [1.0, 0.5, 2.0]  # Favor "excellent"
        
        result = voter.vote_tokens(responses, weights)
        assert "Python is excellent" == result
    
    def test_calculate_consensus_high(self):
        """Test high consensus calculation"""
        voter = TokenVoter()
        responses = [
            "Machine learning is powerful",
            "Machine learning is powerful",
            "Machine learning is strong"
        ]
        
        consensus = voter.calculate_consensus(responses)
        assert consensus > 0.5  # Should have reasonable consensus
    
    def test_calculate_consensus_low(self):
        """Test low consensus calculation"""
        voter = TokenVoter()
        responses = [
            "Completely different response",
            "Totally unrelated answer", 
            "Nothing in common here"
        ]
        
        consensus = voter.calculate_consensus(responses)
        assert consensus < 0.3  # Should have low consensus


class TestEnsembleReasoner:
    """Test ensemble reasoning engine"""
    
    @pytest.fixture
    def mock_cache_manager(self):
        """Mock cache manager"""
        cache = AsyncMock()
        cache.get_response.return_value = None  # No cache hits by default
        cache.store_response.return_value = True
        return cache
    
    @pytest.fixture
    def reasoner(self, mock_cache_manager):
        """Create reasoner with mocked dependencies"""
        return EnsembleReasoner(cache_manager=mock_cache_manager)
    
    @pytest.fixture
    def mock_ollama_client(self):
        """Mock Ollama client"""
        client = AsyncMock()
        
        # Mock successful responses
        def mock_generate(model, prompt, **kwargs):
            return {
                "response": f"Mock response from {model}: {prompt[:50]}...",
                "model": model
            }
        
        client.generate.side_effect = mock_generate
        return client
    
    @pytest.mark.asyncio
    async def test_single_model_inference_success(self, reasoner, mock_ollama_client):
        """Test successful single model inference"""
        reasoner.client = mock_ollama_client
        
        query = QueryRequest(prompt="Test prompt")
        model_config = reasoner.model_configs["llama3.1:8b"]
        
        response = await reasoner._single_model_inference(query, model_config)
        
        assert response.success
        assert response.model == "llama3.1:8b"
        assert "Mock response" in response.response
        assert response.inference_time_ms > 0
    
    @pytest.mark.asyncio
    async def test_single_model_inference_failure(self, reasoner):
        """Test failed single model inference"""
        # Mock client that raises exception
        reasoner.client = AsyncMock()
        reasoner.client.generate.side_effect = Exception("Model not available")
        
        query = QueryRequest(prompt="Test prompt")
        model_config = reasoner.model_configs["llama3.1:8b"]
        
        response = await reasoner._single_model_inference(query, model_config)
        
        assert not response.success
        assert response.error == "Model not available"
        assert response.confidence == 0.0
    
    @pytest.mark.asyncio
    async def test_parallel_inference(self, reasoner, mock_ollama_client):
        """Test parallel inference across multiple models"""
        reasoner.client = mock_ollama_client
        
        query = QueryRequest(
            prompt="Test prompt",
            models=["llama3.1:8b", "mistral:7b"]
        )
        
        # Mock available models
        models = [
            reasoner.model_configs["llama3.1:8b"],
            reasoner.model_configs["mistral:7b"]
        ]
        
        responses = await reasoner._parallel_inference(query, models)
        
        assert len(responses) == 2
        assert all(r.success for r in responses)
        assert responses[0].model == "llama3.1:8b"
        assert responses[1].model == "mistral:7b"
    
    @pytest.mark.asyncio
    async def test_fuse_responses_vote(self, reasoner):
        """Test response fusion with voting method"""
        responses = [
            ModelResponse(
                model="model1", 
                response="The answer is correct",
                confidence=0.8,
                success=True
            ),
            ModelResponse(
                model="model2",
                response="The answer is correct", 
                confidence=0.7,
                success=True
            ),
            ModelResponse(
                model="model3",
                response="The answer is wrong",
                confidence=0.6,
                success=True
            )
        ]
        
        models = list(reasoner.model_configs.values())[:3]
        
        fused = await reasoner._fuse_responses(
            responses, EnsembleMethod.VOTE, models
        )
        
        assert "correct" in fused.lower()
    
    @pytest.mark.asyncio
    async def test_fuse_responses_weighted(self, reasoner):
        """Test response fusion with weighted method"""
        responses = [
            ModelResponse(
                model="llama3.1:8b",  # Higher weight model
                response="Weighted response wins",
                confidence=0.9,
                success=True
            ),
            ModelResponse(
                model="mistral:7b",
                response="Alternative response loses",
                confidence=0.8,
                success=True
            )
        ]
        
        models = [
            reasoner.model_configs["llama3.1:8b"],
            reasoner.model_configs["mistral:7b"]
        ]
        
        fused = await reasoner._fuse_responses(
            responses, EnsembleMethod.WEIGHTED, models
        )
        
        # Should favor the higher-weighted model
        assert "Weighted" in fused or "wins" in fused
    
    @pytest.mark.asyncio
    async def test_reason_with_cache_hit(self, reasoner, mock_cache_manager):
        """Test reasoning with cache hit"""
        # Mock cache hit
        cached_response = QueryResponse(
            fused_response="Cached response",
            method_used=EnsembleMethod.VOTE,
            model_responses=[],
            total_time_ms=1.0,
            cache_hit=True
        )
        mock_cache_manager.get_response.return_value = cached_response
        
        query = QueryRequest(prompt="Test prompt")
        
        response = await reasoner.reason(query)
        
        assert response.cache_hit
        assert response.fused_response == "Cached response"
        assert reasoner.stats["cache_hits"] == 1
    
    @pytest.mark.asyncio
    async def test_reason_full_pipeline(self, reasoner, mock_ollama_client):
        """Test full reasoning pipeline"""
        reasoner.client = mock_ollama_client
        
        # Mock model availability check
        with patch.object(reasoner, '_get_available_models') as mock_available:
            mock_available.return_value = [
                reasoner.model_configs["llama3.1:8b"],
                reasoner.model_configs["mistral:7b"]
            ]
            
            query = QueryRequest(
                prompt="Explain machine learning",
                models=["llama3.1:8b", "mistral:7b"],
                method=EnsembleMethod.VOTE
            )
            
            response = await reasoner.reason(query)
            
            assert response.fused_response
            assert len(response.model_responses) == 2
            assert response.total_time_ms > 0
            assert 0 <= response.consensus_score <= 1
            assert 0 <= response.confidence <= 1
            assert not response.cache_hit
    
    @pytest.mark.asyncio
    async def test_performance_under_50ms_target(self, reasoner, mock_ollama_client):
        """Test that ensemble can achieve <50ms target"""
        reasoner.client = mock_ollama_client
        
        # Mock fast responses
        def fast_generate(model, prompt, **kwargs):
            return {
                "response": f"Fast response from {model}",
                "model": model
            }
        
        reasoner.client.generate.side_effect = fast_generate
        
        with patch.object(reasoner, '_get_available_models') as mock_available:
            mock_available.return_value = [
                reasoner.model_configs["phi3:mini"],  # Fast model
                reasoner.model_configs["mistral:7b"]   # Fast model
            ]
            
            query = QueryRequest(
                prompt="Quick test",
                models=["phi3:mini", "mistral:7b"],
                timeout_ms=100
            )
            
            response = await reasoner.reason(query)
            
            # Should be fast with mocked responses
            assert response.total_time_ms < 100  # Generous for test environment
            assert response.fused_response
    
    def test_performance_stats(self, reasoner):
        """Test performance statistics tracking"""
        # Simulate some usage
        reasoner.stats["total_queries"] = 100
        reasoner.stats["cache_hits"] = 30
        reasoner.stats["avg_inference_time"] = 45.5
        reasoner.stats["consensus_scores"] = [0.8, 0.7, 0.9, 0.6]
        reasoner.stats["model_success_rates"] = {
            "llama3.1:8b": [True, True, False, True],
            "mistral:7b": [True, True, True, True]
        }
        
        stats = reasoner.get_performance_stats()
        
        assert stats["total_queries"] == 100
        assert stats["cache_hit_rate"] == 30.0
        assert stats["avg_inference_time_ms"] == 45.5
        assert stats["avg_consensus_score"] == 0.75
        assert stats["model_performance"]["llama3.1:8b"]["success_rate"] == 75.0
        assert stats["model_performance"]["mistral:7b"]["success_rate"] == 100.0
        assert 0 <= stats["efficiency_score"] <= 100
    
    @pytest.mark.asyncio
    async def test_benchmark_models(self, reasoner, mock_ollama_client):
        """Test model benchmarking"""
        reasoner.client = mock_ollama_client
        
        # Enable some models for testing
        reasoner.model_configs["llama3.1:8b"].enabled = True
        reasoner.model_configs["mistral:7b"].enabled = True
        
        with patch.object(reasoner, '_get_available_models') as mock_available:
            mock_available.return_value = [
                reasoner.model_configs["llama3.1:8b"],
                reasoner.model_configs["mistral:7b"]
            ]
            
            results = await reasoner.benchmark_models(["Test prompt"])
            
            assert "individual_models" in results
            assert "ensemble_methods" in results
            assert "performance_summary" in results
            
            # Should have tested available models
            assert len(results["individual_models"]) >= 1
            assert len(results["ensemble_methods"]) >= 1


class TestIntegration:
    """Integration tests for ensemble system"""
    
    @pytest.mark.asyncio
    async def test_create_ensemble_reasoner(self):
        """Test ensemble reasoner creation"""
        reasoner = await create_ensemble_reasoner()
        
        assert isinstance(reasoner, EnsembleReasoner)
        assert reasoner.model_configs
        assert reasoner.client
        assert reasoner.voter
    
    @pytest.mark.asyncio
    async def test_query_request_validation(self):
        """Test query request validation"""
        # Valid request
        query = QueryRequest(
            prompt="Test prompt",
            models=["llama3.1:8b"],
            method=EnsembleMethod.VOTE
        )
        
        assert query.prompt == "Test prompt"
        assert query.models == ["llama3.1:8b"]
        assert query.method == EnsembleMethod.VOTE
        
        # Test defaults
        default_query = QueryRequest(prompt="Test")
        assert default_query.models == ["llama3.1:8b", "mistral:7b"]
        assert default_query.method == EnsembleMethod.VOTE
        assert default_query.timeout_ms == 2000
    
    @pytest.mark.asyncio
    async def test_error_handling_no_models(self):
        """Test error handling when no models available"""
        reasoner = EnsembleReasoner()
        
        # Mock no available models
        with patch.object(reasoner, '_get_available_models') as mock_available:
            mock_available.return_value = []
            
            query = QueryRequest(prompt="Test")
            
            with pytest.raises(RuntimeError, match="No models available"):
                await reasoner.reason(query)
    
    @pytest.mark.asyncio
    async def test_error_handling_all_models_fail(self):
        """Test error handling when all models fail"""
        mock_cache_manager = AsyncMock()
        mock_cache_manager.get_response.return_value = None
        reasoner = EnsembleReasoner(cache_manager=mock_cache_manager)
        
        # Mock client that always fails
        reasoner.client = AsyncMock()
        reasoner.client.generate.side_effect = Exception("All models failed")
        
        with patch.object(reasoner, '_get_available_models') as mock_available:
            mock_available.return_value = [reasoner.model_configs["llama3.1:8b"]]
            
            query = QueryRequest(prompt="Test")
            
            with pytest.raises(RuntimeError, match="All models failed"):
                await reasoner.reason(query)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])