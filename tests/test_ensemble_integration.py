#!/usr/bin/env python3
"""
Test ensemble integration with LangChain
"""

import sys
import os
import asyncio

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

print("=" * 80)
print("TESTING LANGCHAIN ENSEMBLE INTEGRATION")
print("=" * 80)

async def test_ensemble_chain():
    """Test EnsembleChain wrapper"""
    print("\n[1/3] Testing EnsembleChain...")
    
    try:
        from xencode.agentic import create_ensemble_chain
        from xencode.ai_ensembles import EnsembleMethod
        
        # Create ensemble chain
        chain = await create_ensemble_chain(
            models=["mistral:7b", "phi3:mini"],  # Using smaller models for testing
            method=EnsembleMethod.VOTE
        )
        
        # Test query
        result = await chain.acall({
            "prompt": "What is 2+2?",
            "max_tokens": 100
        })
        
        print(f"✅ EnsembleChain created successfully")
        print(f"   Response: {result['response'][:50]}...")
        print(f"   Consensus: {result['consensus_score']:.2f}")
        print(f"   Confidence: {result['confidence']:.2f}")
        print(f"   Models: {len(result['model_responses'])}")
        
        return True
    except Exception as e:
        print(f"❌ EnsembleChain FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_model_council():
    """Test ModelCouncil"""
    print("\n[2/3] Testing ModelCouncil...")
    
    try:
        from xencode.agentic import create_model_council
        
        # Create council (just test initialization, no actual inference)
        council = await create_model_council()
        
        print(f"✅ ModelCouncil created successfully")
        print(f"   Coordinator agents: {len(council.coordinator.agents)}")
        
        return True
    except Exception as e:
        print(f"❌ ModelCouncil FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_imports():
    """Test module imports"""
    print("\n[3/3] Testing imports...")
    
    try:
        from xencode.agentic import (
            EnsembleChain,
            ModelCouncil,
            create_ensemble_chain,
            create_model_council
        )
        
        print("✅ All imports successful")
        return True
    except Exception as e:
        print(f"❌ Import FAILED: {e}")
        return False

async def main():
    results = []
    
    # Test imports first
    results.append(await test_imports())
    
    # Test ensemble chain (requires Ollama)
    # Commented out by default to avoid requiring running Ollama
    # results.append(await test_ensemble_chain())
    
    # Test council creation
    results.append(await test_model_council())
    
    print("\n" + "=" * 80)
    if all(results):
        print("✅ ALL TESTS PASSED")
        print("=" * 80)
        print("\n🎉 Ensemble integration ready to use!")
    else:
        print("❌ SOME TESTS FAILED")
        print("=" * 80)

if __name__ == "__main__":
    asyncio.run(main())
