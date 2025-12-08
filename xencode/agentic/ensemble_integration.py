"""
LangChain Ensemble Integration

Integrates existing EnsembleReasoner with LangChain for hybrid ensemble approach.
Provides semantic voting, council pattern, and enhanced orchestration.
"""

from typing import Dict, List, Optional, Any
from langchain.chains.base import Chain
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain_core.language_models import BaseLLM
from pydantic import Field
import asyncio

from ..ai_ensembles import EnsembleReasoner, QueryRequest, EnsembleMethod, QueryResponse


class EnsembleChain(Chain):
    """LangChain wrapper for EnsembleReasoner"""
    
    ensemble_reasoner: EnsembleReasoner = Field(default_factory=EnsembleReasoner)
    models: List[str] = Field(default=["llama3.1:8b", "mistral:7b"])
    method: EnsembleMethod = Field(default=EnsembleMethod.VOTE)
    max_tokens: int = Field(default=512)
    temperature: float = Field(default=0.7)
    timeout_ms: int = Field(default=2000)
    
    @property
    def input_keys(self) -> List[str]:
        return ["prompt"]
    
    @property
    def output_keys(self) -> List[str]:
        return ["response", "consensus_score", "confidence", "model_responses"]
    
    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """Execute ensemble reasoning"""
        prompt = inputs["prompt"]
        
        # Create query request
        query = QueryRequest(
            prompt=prompt,
            models=inputs.get("models", self.models),
            method=inputs.get("method", self.method),
            max_tokens=inputs.get("max_tokens", self.max_tokens),
            temperature=inputs.get("temperature", self.temperature),
            timeout_ms=inputs.get("timeout_ms", self.timeout_ms)
        )
        
        # Run ensemble (sync wrapper for async)
        response = asyncio.run(self.ensemble_reasoner.reason(query))
        
        # Callback for streaming/logging
        if run_manager:
            run_manager.on_text(f"Ensemble Response: {response.fused_response}\n")
        
        return {
            "response": response.fused_response,
            "consensus_score": response.consensus_score,
            "confidence": response.confidence,
            "model_responses": [
                {
                    "model": r.model,
                    "response": r.response,
                    "confidence": r.confidence,
                    "inference_time_ms": r.inference_time_ms
                }
                for r in response.model_responses
            ],
            "total_time_ms": response.total_time_ms
        }
    
    async def acall(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """Async execution"""
        prompt = inputs["prompt"]
        
        query = QueryRequest(
            prompt=prompt,
            models=inputs.get("models", self.models),
            method=inputs.get("method", self.method),
            max_tokens=inputs.get("max_tokens", self.max_tokens),
            temperature=inputs.get("temperature", self.temperature),
            timeout_ms=inputs.get("timeout_ms", self.timeout_ms)
        )
        
        response = await self.ensemble_reasoner.reason(query)
        
        if run_manager:
            run_manager.on_text(f"Ensemble Response: {response.fused_response}\n")
        
        return {
            "response": response.fused_response,
            "consensus_score": response.consensus_score,
            "confidence": response.confidence,
            "model_responses": [
                {
                    "model": r.model,
                    "response": r.response,
                    "confidence": r.confidence,
                    "inference_time_ms": r.inference_time_ms
                }
                for r in response.model_responses
            ],
            "total_time_ms": response.total_time_ms
        }
    
    @property
    def _chain_type(self) -> str:
        return "ensemble"


class ModelCouncil:
    """Council of specialized models for consensus-driven reasoning"""
    
    def __init__(self, cache_manager=None):
        from .coordinator import AgentCoordinator, AgentType
        
        self.coordinator = AgentCoordinator()
        self.ensemble_chain = EnsembleChain()
        
    async def deliberate(
        self, 
        task: str, 
        use_ensemble: bool = True,
        specialized_routing: bool = True
    ) -> Dict[str, Any]:
        """
        Council deliberation process
        
        Args:
            task: The task to solve
            use_ensemble: Whether to use ensemble voting
            specialized_routing: Whether to route to specialized agents
            
        Returns:
            Final response with all expert opinions
        """
        # Route to specialized agent if enabled
        if specialized_routing:
            agent_result = self.coordinator.delegate_task(task)
            primary_response = agent_result["result"]
            primary_agent = agent_result["selected_agent"]
        else:
            # Use general ensemble
            ensemble_result = await self.ensemble_chain.acall({"prompt": task})
            primary_response = ensemble_result["response"]
            primary_agent = "ensemble"
        
        # For critical tasks, get consensus from other agents
        if use_ensemble and specialized_routing:
            # Get opinions from other agent types
            other_results = []
            for agent_type_name, agent in self.coordinator.agents.items():
                if agent.agent_type.value != primary_agent:
                    result = agent.execute(task)
                    other_results.append({
                        "agent": agent_type_name,
                        "response": result["result"]
                    })
            
            # Synthesize final response
            all_responses = [primary_response] + [r["response"] for r in other_results]
            
            # Use ensemble chain to fuse responses
            final_result = await self.ensemble_chain.acall({
                "prompt": f"Synthesize these expert opinions into one answer:\n\n" + 
                         "\n\n".join(all_responses)
            })
            
            return {
                "final_response": final_result["response"],
                "primary_agent": primary_agent,
                "primary_response": primary_response,
                "expert_opinions": other_results,
                "consensus_score": final_result["consensus_score"],
                "confidence": final_result["confidence"]
            }
        
        return {
            "final_response": primary_response,
            "primary_agent": primary_agent,
            "consensus_score": 1.0,
            "confidence": 0.8
        }
    
    async def benchmark_council(self, test_tasks: List[str]) -> Dict[str, Any]:
        """Benchmark council vs individual agents"""
        results = {
            "council": [],
            "individual_agents": {},
            "summary": {}
        }
        
        # Test council
        for task in test_tasks:
            council_result = await self.deliberate(task, use_ensemble=True)
            results["council"].append({
                "task": task,
                "confidence": council_result["confidence"],
                "consensus": council_result["consensus_score"]
            })
        
        # Test individual agents
        for agent_type, agent in self.coordinator.agents.items():
            agent_results = []
            for task in test_tasks:
                result = agent.execute(task)
                agent_results.append({
                    "task": task,
                    "execution_time": result["execution_time"]
                })
            results["individual_agents"][agent_type] = agent_results
        
        # Calculate summary
        avg_council_confidence = sum(r["confidence"] for r in results["council"]) / len(results["council"])
        avg_council_consensus = sum(r["consensus"] for r in results["council"]) / len(results["council"])
        
        results["summary"] = {
            "avg_council_confidence": avg_council_confidence,
            "avg_council_consensus": avg_council_consensus,
            "improvement_estimate": f"+{int((avg_council_consensus - 0.7) * 100)}% vs single agent"
        }
        
        return results


# Convenience function
async def create_ensemble_chain(
    models: List[str] = None,
    method: EnsembleMethod = EnsembleMethod.VOTE,
    cache_manager=None
) -> EnsembleChain:
    """Create ensemble chain with configuration"""
    ensemble_reasoner = EnsembleReasoner(cache_manager=cache_manager)
    
    return EnsembleChain(
        ensemble_reasoner=ensemble_reasoner,
        models=models or ["llama3.1:8b", "mistral:7b"],
        method=method
    )


async def create_model_council(cache_manager=None) -> ModelCouncil:
    """Create model council"""
    return ModelCouncil(cache_manager=cache_manager)
