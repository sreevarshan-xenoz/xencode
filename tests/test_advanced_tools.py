import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from xencode.agentic.advanced_tools import ToolRegistry

def test_tool_registry():
    print("Testing Tool Registry...")
    registry = ToolRegistry()
    
    all_tools = registry.get_all_tools()
    print(f"✅ Total tools registered: {len(all_tools)}")
    
    tool_names = [t.name for t in all_tools]
    print(f"Tools: {', '.join(tool_names)}")
    
    # Test categorization
    git_tools = registry.get_tools_by_category("git")
    print(f"\n✅ Git tools: {len(git_tools)} - {[t.name for t in git_tools]}")
    
    web_tools = registry.get_tools_by_category("web")
    print(f"✅ Web tools: {len(web_tools)} - {[t.name for t in web_tools]}")
    
    code_tools = registry.get_tools_by_category("code")
    print(f"✅ Code tools: {len(code_tools)} - {[t.name for t in code_tools]}")
    
    print("\n✅ All tests passed!")

if __name__ == "__main__":
    test_tool_registry()
