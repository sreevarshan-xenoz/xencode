#!/usr/bin/env python3
"""
Tests for RLHF Tuner System

Comprehensive testing of the RLHF tuning system with mocks
for offline testing and performance validation.
"""

import asyncio
import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, mock_open
import torch

from xencode.rlhf_tuner import (
    RLHFTuner, RLHFConfig, CodePair, TrainingMetrics,
    SyntheticDataGenerator, create_rlhf_tuner
)


class TestRLHFConfig:
    """Test RLHF configuration"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = RLHFConfig()
        
        assert config.base_model == "microsoft/DialoGPT-small"
        assert config.lora_rank == 16
        assert config.lora_alpha == 32
        assert config.learning_rate == 2e-4
        assert config.max_epochs == 3
        assert config.synthetic_data_size == 100
    
    def test_custom_config(self):
        """Test custom configuration"""
        config = RLHFConfig(
            base_model="custom/model",
            lora_rank=8,
            max_epochs=5,
            learning_rate=1e-4
        )
        
        assert config.base_model == "custom/model"
        assert config.lora_rank == 8
        assert config.max_epochs == 5
        assert config.learning_rate == 1e-4


class TestCodePair:
    """Test CodePair data structure"""
    
    def test_code_pair_creation(self):
        """Test creating code pairs"""
        pair = CodePair(
            input_code="def test(): pass",
            output_code="def test() -> None: pass",
            task_type="refactor",
            quality_score=0.8
        )
        
        assert pair.input_code == "def test(): pass"
        assert pair.output_code == "def test() -> None: pass"
        assert pair.task_type == "refactor"
        assert pair.quality_score == 0.8
        assert pair.human_feedback is None
        assert isinstance(pair.metadata, dict)
    
    def test_code_pair_with_metadata(self):
        """Test code pair with metadata"""
        metadata = {"source": "synthetic", "difficulty": "easy"}
        pair = CodePair(
            input_code="x = 1",
            output_code="x: int = 1",
            task_type="refactor",
            metadata=metadata
        )
        
        assert pair.metadata == metadata


class TestSyntheticDataGenerator:
    """Test synthetic data generation"""
    
    @pytest.fixture
    def generator(self):
        """Create data generator"""
        return SyntheticDataGenerator()
    
    def test_load_code_templates(self, generator):
        """Test loading code templates"""
        templates = generator._load_code_templates()
        
        assert "refactor" in templates
        assert "debug" in templates
        assert "optimize" in templates
        assert "explain" in templates
        
        # Check that each category has templates
        for category, template_list in templates.items():
            assert isinstance(template_list, list)
            assert len(template_list) > 0
            assert all(isinstance(template, str) for template in template_list)
    
    @pytest.mark.asyncio
    async def test_generate_pairs(self, generator):
        """Test generating code pairs"""
        pairs = await generator.generate_pairs(5)
        
        assert len(pairs) == 5
        assert all(isinstance(pair, CodePair) for pair in pairs)
        
        # Check that all pairs have required fields
        for pair in pairs:
            assert pair.input_code
            assert pair.output_code
            assert pair.task_type in ["refactor", "debug", "optimize", "explain"]
            assert 0 <= pair.quality_score <= 1
            assert pair.metadata.get("generated") is True
    
    @pytest.mark.asyncio
    async def test_generate_improved_code(self, generator):
        """Test generating improved code"""
        input_code = "def calculate_sum(numbers):\n    result = 0\n    for num in numbers:\n        result = result + num\n    return result"
        
        improved = await generator._generate_improved_code(input_code, "refactor")
        
        assert improved != input_code
        assert len(improved) > 0
        
        # Should contain improvements for known templates
        if input_code in generator.code_templates["refactor"]:
            assert "sum(numbers)" in improved or "List[float]" in improved


class TestRLHFTuner:
    """Test RLHF tuner system"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    @pytest.fixture
    def config(self, temp_dir):
        """Create test configuration"""
        return RLHFConfig(
            output_dir=str(temp_dir / "rlhf_test"),
            max_epochs=1,
            synthetic_data_size=5,
            batch_size=1
        )
    
    @pytest.fixture
    def tuner(self, config):
        """Create RLHF tuner"""
        return RLHFTuner(config)
    
    def test_tuner_initialization(self, tuner, config):
        """Test tuner initialization"""
        assert tuner.config == config
        assert tuner.model is None
        assert tuner.tokenizer is None
        assert tuner.peft_model is None
        assert isinstance(tuner.training_history, list)
        assert len(tuner.training_history) == 0
        assert tuner.output_dir.exists()
    
    @pytest.mark.asyncio
    async def test_initialize_model_mock(self, tuner):
        """Test model initialization with mocks"""
        with patch('xencode.rlhf_tuner.AutoTokenizer') as mock_tokenizer, \
             patch('xencode.rlhf_tuner.AutoModelForCausalLM') as mock_model, \
             patch('xencode.rlhf_tuner.get_peft_model') as mock_peft:
            
            # Setup mocks
            mock_tokenizer.from_pretrained.return_value = MagicMock()
            mock_tokenizer.from_pretrained.return_value.pad_token = None
            mock_tokenizer.from_pretrained.return_value.eos_token = "<eos>"
            
            mock_model.from_pretrained.return_value = MagicMock()
            mock_peft.return_value = MagicMock()
            mock_peft.return_value.print_trainable_parameters = MagicMock()
            
            # Test initialization
            result = await tuner.initialize_model()
            
            assert result is True
            assert tuner.tokenizer is not None
            assert tuner.model is not None
            assert tuner.peft_model is not None
            
            # Verify calls
            mock_tokenizer.from_pretrained.assert_called_once()
            mock_model.from_pretrained.assert_called_once()
            mock_peft.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_generate_training_data(self, tuner):
        """Test training data generation"""
        with patch.object(tuner.data_generator, 'generate_pairs') as mock_generate:
            mock_pairs = [
                CodePair("input1", "output1", "refactor", 0.8),
                CodePair("input2", "output2", "debug", 0.9)
            ]
            mock_generate.return_value = mock_pairs
            
            # Mock file operations
            with patch('builtins.open', mock_open()) as mock_file:
                pairs = await tuner.generate_training_data(2)
                
                assert len(pairs) == 2
                assert pairs == mock_pairs
                mock_generate.assert_called_once_with(2)
                mock_file.assert_called_once()
    
    def test_prepare_dataset(self, tuner):
        """Test dataset preparation"""
        # Mock tokenizer
        tuner.tokenizer = MagicMock()
        tuner.tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]])
        }
        
        code_pairs = [
            CodePair("def test(): pass", "def test() -> None: pass", "refactor", 0.8)
        ]
        
        with patch('xencode.rlhf_tuner.Dataset') as mock_dataset:
            mock_dataset.from_dict.return_value.map.return_value = MagicMock()
            
            dataset = tuner._prepare_dataset(code_pairs)
            
            mock_dataset.from_dict.assert_called_once()
            assert dataset is not None
    
    def test_calculate_quality_score(self, tuner):
        """Test quality score calculation"""
        input_code = "def test(): pass"
        
        # Test basic code
        basic_output = "def test(): return None"
        score1 = tuner._calculate_quality_score(input_code, basic_output)
        assert 0.5 <= score1 <= 1.0
        
        # Test improved code with docstring and type hints
        improved_output = '''def test() -> None:
            """Test function with docstring."""
            return None'''
        score2 = tuner._calculate_quality_score(input_code, improved_output)
        assert score2 > score1
        
        # Test code with error handling
        robust_output = '''def test() -> None:
            """Test function with error handling."""
            try:
                return None
            except Exception:
                pass'''
        score3 = tuner._calculate_quality_score(input_code, robust_output)
        assert score3 >= score2
    
    @pytest.mark.asyncio
    async def test_save_and_load_model(self, tuner, temp_dir):
        """Test model saving and loading"""
        # Mock model components
        tuner.peft_model = MagicMock()
        tuner.peft_model.save_pretrained = MagicMock()
        tuner.tokenizer = MagicMock()
        tuner.tokenizer.save_pretrained = MagicMock()
        
        save_path = temp_dir / "test_model"
        
        # Test saving
        with patch('builtins.open', mock_open()) as mock_file:
            result = await tuner.save_model(str(save_path))
            
            assert result is True
            tuner.peft_model.save_pretrained.assert_called_once()
            tuner.tokenizer.save_pretrained.assert_called_once()
            mock_file.assert_called()
        
        # Test loading (mock the file existence and content)
        with patch.object(Path, 'exists', return_value=True), \
             patch('builtins.open', mock_open(read_data='{"base_model": "test/model"}')), \
             patch.object(tuner, 'initialize_model', return_value=True), \
             patch('xencode.rlhf_tuner.PeftModel') as mock_peft_model:
            
            mock_peft_model.from_pretrained.return_value = MagicMock()
            
            result = await tuner.load_model(str(save_path))
            
            assert result is True
            mock_peft_model.from_pretrained.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_evaluate_model(self, tuner):
        """Test model evaluation"""
        # Mock model components
        tuner.peft_model = MagicMock()
        tuner.tokenizer = MagicMock()
        tuner.tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
        tuner.tokenizer.decode.return_value = "def improved_function(): pass"
        tuner.tokenizer.eos_token_id = 0
        
        # Mock model generation
        tuner.peft_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
        
        # Mock data generator
        test_pairs = [
            CodePair("def test(): pass", "def test() -> None: pass", "refactor", 0.8)
        ]
        
        with patch.object(tuner.data_generator, 'generate_pairs', return_value=test_pairs):
            results = await tuner.evaluate_model()
            
            assert "total_samples" in results
            assert "avg_quality_score" in results
            assert "inference_time_ms" in results
            assert "task_type_performance" in results
            
            assert results["total_samples"] > 0
            assert 0 <= results["avg_quality_score"] <= 1
            assert results["inference_time_ms"] >= 0


class TestTrainingMetrics:
    """Test training metrics"""
    
    def test_training_metrics_creation(self):
        """Test creating training metrics"""
        metrics = TrainingMetrics(
            epoch=1,
            loss=0.5,
            perplexity=1.65,
            learning_rate=2e-4,
            grad_norm=1.0,
            step=100
        )
        
        assert metrics.epoch == 1
        assert metrics.loss == 0.5
        assert metrics.perplexity == 1.65
        assert metrics.learning_rate == 2e-4
        assert metrics.grad_norm == 1.0
        assert metrics.step == 100
        assert metrics.timestamp is not None


class TestIntegration:
    """Integration tests"""
    
    @pytest.mark.asyncio
    async def test_create_rlhf_tuner(self):
        """Test creating RLHF tuner"""
        with patch('xencode.rlhf_tuner.AutoTokenizer'), \
             patch('xencode.rlhf_tuner.AutoModelForCausalLM'), \
             patch('xencode.rlhf_tuner.get_peft_model'):
            
            tuner = await create_rlhf_tuner()
            
            assert isinstance(tuner, RLHFTuner)
            assert tuner.config is not None
    
    @pytest.mark.asyncio
    async def test_end_to_end_workflow_mock(self):
        """Test end-to-end workflow with mocks"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = RLHFConfig(
                output_dir=str(Path(tmpdir) / "test_rlhf"),
                max_epochs=1,
                synthetic_data_size=2,
                batch_size=1
            )
            
            tuner = RLHFTuner(config)
            
            # Mock all external dependencies
            with patch('xencode.rlhf_tuner.AutoTokenizer') as mock_tokenizer, \
                 patch('xencode.rlhf_tuner.AutoModelForCausalLM') as mock_model, \
                 patch('xencode.rlhf_tuner.get_peft_model') as mock_peft, \
                 patch('xencode.rlhf_tuner.Trainer') as mock_trainer:
                
                # Setup mocks
                mock_tokenizer.from_pretrained.return_value = MagicMock()
                mock_tokenizer.from_pretrained.return_value.pad_token = None
                mock_tokenizer.from_pretrained.return_value.eos_token = "<eos>"
                
                mock_model.from_pretrained.return_value = MagicMock()
                mock_peft.return_value = MagicMock()
                mock_peft.return_value.print_trainable_parameters = MagicMock()
                
                mock_trainer_instance = MagicMock()
                mock_trainer_instance.train.return_value = None
                mock_trainer_instance.save_model.return_value = None
                mock_trainer.return_value = mock_trainer_instance
                
                # Test workflow
                assert await tuner.initialize_model()
                
                code_pairs = await tuner.generate_training_data(2)
                assert len(code_pairs) == 2
                
                # Mock the training process
                with patch.object(tuner, '_prepare_dataset') as mock_prepare, \
                     patch('torch.utils.data.random_split') as mock_split:
                    
                    mock_prepare.return_value = MagicMock()
                    mock_split.return_value = (MagicMock(), MagicMock())
                    
                    result = await tuner.train(code_pairs)
                    assert result is True
                    
                    mock_trainer.assert_called_once()
                    mock_trainer_instance.train.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])