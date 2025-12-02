import sys
import os
import traceback

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from xencode.agentic.manager import LangChainManager
    print("Success")
except Exception:
    traceback.print_exc()
