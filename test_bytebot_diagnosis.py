import sys
import traceback
sys.path.insert(0, 'D:/xencode')

print("Starting ByteBot diagnosis...")

try:
    # Test importing all components
    print("Testing imports...")
    from xencode.bytebot import ByteBotEngine
    print("✓ ByteBotEngine imported")
    
    from xencode.bytebot.context_engine import ContextEngine
    print("✓ ContextEngine imported")
    
    from xencode.bytebot.executor import Executor
    print("✓ Executor imported")
    
    from xencode.bytebot.planner import Planner
    print("✓ Planner imported")
    
    from xencode.bytebot.risk_scorer import RiskScorer
    print("✓ RiskScorer imported")
    
    from xencode.bytebot.safety_gate import SafetyGate, ExecutionMode
    print("✓ SafetyGate imported")
    
    from xencode.bytebot.terminal_cognition_layer import TerminalCognitionLayer
    print("✓ TerminalCognitionLayer imported")
    
    from xencode.bytebot.execution_modes import ModeAwareByteBot
    print("✓ ModeAwareByteBot imported")
    
    from xencode.bytebot.plan_graph_storage import PlanGraphManager
    print("✓ PlanGraphManager imported")
    
    from xencode.bytebot.replay_debug import ReplayAndDebugManager
    print("✓ ReplayAndDebugManager imported")
    
    print("\nAll imports successful!")

    # Test creating instances (without full initialization to avoid dependency issues)
    print("\nTesting instantiation...")
    
    try:
        engine = ByteBotEngine()
        print("✓ ByteBotEngine instance created successfully")
    except Exception as e:
        print(f"✗ Error creating ByteBotEngine: {e}")
        traceback.print_exc()

    try:
        context = ContextEngine()
        print("✓ ContextEngine instance created successfully")
    except Exception as e:
        print(f"✗ Error creating ContextEngine: {e}")
        traceback.print_exc()
        
    try:
        risk_scorer = RiskScorer()
        print("✓ RiskScorer instance created successfully")
    except Exception as e:
        print(f"✗ Error creating RiskScorer: {e}")
        traceback.print_exc()
        
    try:
        safety_gate = SafetyGate()
        print("✓ SafetyGate instance created successfully")
    except Exception as e:
        print(f"✗ Error creating SafetyGate: {e}")
        traceback.print_exc()
        
    print('\nBasic instantiation test completed!')
    
    # Test some basic functionality
    print("\nTesting basic functionality...")
    
    try:
        rs = RiskScorer()
        assessment = rs.score_command("ls -la", {})
        print(f"✓ Risk assessment works: score = {assessment.score}")
    except Exception as e:
        print(f"✗ Risk assessment failed: {e}")
        traceback.print_exc()
        
    try:
        sg = SafetyGate()
        should_block = sg.should_block({"command": "ls -la"}, 0.1, ExecutionMode.EXECUTE)
        print(f"✓ Safety gate works: should_block = {should_block}")
    except Exception as e:
        print(f"✗ Safety gate failed: {e}")
        traceback.print_exc()
        
    print("\nDiagnosis completed successfully!")
    
except Exception as e:
    print(f"Diagnosis failed with error: {e}")
    traceback.print_exc()