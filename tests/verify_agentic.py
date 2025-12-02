import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from xencode.agentic.manager import LangChainManager

def test_agent_initialization():
    print("Testing Agent Initialization...")
    try:
        # Use a dummy base url if ollama is not running, but we hope it is or we mock it
        # For this test, we just check if class instantiates and tools are loaded
        manager = LangChainManager(model_name="qwen3:4b")
        print("✅ Manager initialized")
        
        tool_names = [t.name for t in manager.tools]
        print(f"Tools found: {tool_names}")
        
        expected_tools = ["read_file", "write_file", "execute_command"]
        if all(t in tool_names for t in expected_tools):
            print("✅ All expected tools present")
        else:
            print("❌ Missing tools")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        # If it fails due to connection error (Ollama not running), we might want to catch that
        # But for now let's see if it runs
        if "Connection refused" in str(e):
             print("⚠️  Ollama might not be running. Skipping actual connection test.")
        else:
             sys.exit(1)

if __name__ == "__main__":
    test_agent_initialization()
